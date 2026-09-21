// The hero attention band, hand written for the deep-bank lineage (heroD4: d 640,
// 128 heads x hd 32 = a 4096-wide attention bank), replacing lc0's `fusedMHA` in
// hero_forward.cu.
//
// WHY (docs/audit/herod4_engine_384.md): at mb 384 `fusedMHA` costs 22.55 ms of a
// 78.6 ms forward -- 536 GB/s, 26% of the A100's HBM, with ncu reading SM 47% and
// DRAM 30%. It is neither compute- nor bandwidth-bound: its grid is (1, H, N) =
// 49,152 blocks of FOUR warps, each doing one 64x64x32 attention through the generic
// CUTLASS fMHA machinery (online-softmax rescaling, logsumexp, epilogue iterators) --
// ~2 bytes moved per instruction issued. Our own Triton band on the identical layout
// runs at 70% of HBM. This kernel is the lean version of the same math:
//
//   one WARP owns one (board, head): q,k,v [64 x 32] fp16 in, o [64 x 32] fp16 out,
//   12 KB in + 4 KB out per warp against ~900 instructions = ~19 B/instruction, 8x
//   the 2.46 B/instruction an A100 needs to stay HBM-bound at 4 IPC.
//
// Math, in the engine's order (hero_forward.cu:504 -> cutlass_kernels.cu:121 and
// third_party/fused_multi_head_attention/kernel_forward.h:816-822):
//   S = (q k^T) * 1/sqrt(hd)  THEN  + the per-head 64x64 stencil bias, softmax over
//   the 64 keys in fp32 (max subtracted -- the accumulators reach fp16 through P),
//   O = P v, normalised by the row sum at the end.
//
// Layouts (all unchanged from what hero_forward.cu already holds):
//   q,k,v,o : [R = N*64, bank = H*hd] fp16 row-major, head-major columns
//             (head h owns columns 32h..32h+31), row = board*64 + square.
//   bias    : [H, 64, 64] fp16, query-major, batch-independent (broadcast).
//
// Tensor-core plumbing: mma.sync.m16n8k16.f32.f16.f16.f32, 4 m-atoms x 8 n-atoms for
// S and 4 m-atoms x 4 n-atoms x 4 k-steps for O. k is the B operand straight out of
// an SMEM tile stored [key][dim] (ldmatrix, non-trans); v is the B operand of the
// second mma out of a tile stored [key][dim] through ldmatrix.trans; and the C
// fragment of two adjacent n-atoms, in order, IS the A fragment of the next mma, so
// P never leaves the register file. SMEM rows are padded to 40 halves (tiles) and 72
// (bias) so that 8 consecutive rows land in 8 distinct 4-bank groups.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace lczero {
namespace hero {
namespace {

constexpr int kWarps = 4;                 // boards per block; they share the bias tile
constexpr int kTS = 40;                   // SMEM halves per tile row (32 data + 8 pad)
constexpr int kTSz = 64 * kTS;
constexpr int kBS = 72;                   // SMEM halves per bias row (64 data + 8 pad)
constexpr int kBSz = 64 * kBS;
constexpr int kSmem = (int)((kBSz + kWarps * 2 * kTSz) * sizeof(__half));   // 49 KB -> 3 blocks/SM
constexpr float kL2E = 1.4426950408889634f;

__device__ __forceinline__ uint32_t sptr(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldm4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldm4t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ float ex2(float x) {
  float r; asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float qmax(float x) {   // reduce over the 4 lanes of a quad
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
  return fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
}
__device__ __forceinline__ float qsum(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 1);
  return x + __shfl_xor_sync(0xffffffffu, x, 2);
}

__global__ __launch_bounds__(kWarps * 32, 1) void hero_band_kernel(
    __half* __restrict__ o, const __half* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, const __half* __restrict__ bias, int N, int bank,
    float scale) {
  extern __shared__ __half sm[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int h = blockIdx.y, n = blockIdx.x * kWarps + warp;

  // the head's 64x64 stencil bias, once per block (all kWarps boards share it)
  for (int c = threadIdx.x; c < 512; c += kWarps * 32)      // 512 x 16 B = 64x64 halves
    *(uint4*)&sm[(c >> 3) * kBS + ((c & 7) << 3)] =
        *(const uint4*)&bias[(size_t)h * 4096 + (c >> 3) * 64 + ((c & 7) << 3)];
  __syncthreads();
  if (n >= N) return;                      // guarded tail: N need not be a multiple of kWarps

  // TWO tiles per warp, not three: k and v share one buffer (k is dead the moment its
  // B fragments are in registers). That is what takes SMEM from 129 KB to 49 KB per
  // block, i.e. from ONE resident block per SM to THREE -- ncu on the 3-tile version
  // read DRAM 50.5%, occupancy 12.0%, stall long-scoreboard 2.31: memory latency with
  // too few warps to hide it, which is the same disease fusedMHA has.
  __half* sq = sm + kBSz + warp * 2 * kTSz;   // q, then the output tile
  __half* skv = sq + kTSz;                    // k, then v
  const size_t base = (size_t)n * 64 * bank + (size_t)h * 32;
  const int lrow = lane >> 2, lcol = (lane & 3) << 3;
  {   // q and k: 16 copies in flight (64 rows x 64 B, four lanes per row)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      const int s = lrow + i * 8;
      cp16(sptr(&sq[s * kTS + lcol]), &q[base + (size_t)s * bank + lcol]);
      cp16(sptr(&skv[s * kTS + lcol]), &k[base + (size_t)s * bank + lcol]);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
  }
  __syncwarp();
  uint32_t kb[8][4];        // B fragments of k: [n-atom][k-step 0 (2 regs), k-step 1]
  #pragma unroll
  for (int j = 0; j < 8; j++)
    ldm4(kb[j], sptr(&skv[(j * 8 + (lane & 7)) * kTS + (lane >> 3) * 8]));
  __syncwarp();             // k is consumed; v may now overwrite the buffer
  {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      const int s = lrow + i * 8;
      cp16(sptr(&skv[s * kTS + lcol]), &v[base + (size_t)s * bank + lcol]);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
  }
  __syncwarp();
  uint32_t vb[4][2][4];     // B fragments of v: [k-step][dims 0-15 / 16-31][2 n-atoms]
  #pragma unroll
  for (int t = 0; t < 4; t++) {
    const int kr = t * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), dc = ((lane >> 4) & 1) * 8;
    ldm4t(vb[t][0], sptr(&skv[kr * kTS + dc]));
    ldm4t(vb[t][1], sptr(&skv[kr * kTS + 16 + dc]));
  }

  const int rA = lane >> 2, rB = rA + 8;          // the two query rows this lane owns
  #pragma unroll 1
  for (int m = 0; m < 4; m++) {                   // 4 m-atoms of 16 queries
    uint32_t qa[2][4];
    #pragma unroll
    for (int t = 0; t < 2; t++)
      ldm4(qa[t], sptr(&sq[(m * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * kTS
                           + t * 16 + ((lane >> 4) & 1) * 8]));
    float s[8][4];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
      mma(s[j], qa[0], &kb[j][0]);
      mma(s[j], qa[1], &kb[j][2]);
    }
    // scale, then the stencil bias (the engine's order), both folded into log2 space
    const int qA = m * 16 + rA, qB = m * 16 + rB;
    float mA = -3.0e38f, mB = -3.0e38f;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      const int key = j * 8 + ((lane & 3) << 1);
      const __half2 bA = *(const __half2*)&sm[qA * kBS + key];
      const __half2 bB = *(const __half2*)&sm[qB * kBS + key];
      s[j][0] = fmaf(s[j][0], scale, __low2float(bA) * kL2E);
      s[j][1] = fmaf(s[j][1], scale, __high2float(bA) * kL2E);
      s[j][2] = fmaf(s[j][2], scale, __low2float(bB) * kL2E);
      s[j][3] = fmaf(s[j][3], scale, __high2float(bB) * kL2E);
      mA = fmaxf(mA, fmaxf(s[j][0], s[j][1]));
      mB = fmaxf(mB, fmaxf(s[j][2], s[j][3]));
    }
    mA = qmax(mA); mB = qmax(mB);
    float sA = 0.f, sB = 0.f;
    uint32_t pa[4][4];      // P as A fragments: [k-step][a0a1, a2a3, a4a5, a6a7]
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      const float e0 = ex2(s[j][0] - mA), e1 = ex2(s[j][1] - mA);
      const float e2 = ex2(s[j][2] - mB), e3 = ex2(s[j][3] - mB);
      sA += e0 + e1; sB += e2 + e3;
      const __half2 h0 = __floats2half2_rn(e0, e1), h1 = __floats2half2_rn(e2, e3);
      pa[j >> 1][(j & 1) * 2 + 0] = *(const uint32_t*)&h0;
      pa[j >> 1][(j & 1) * 2 + 1] = *(const uint32_t*)&h1;
    }
    sA = qsum(sA); sB = qsum(sB);
    float acc[4][4];
    #pragma unroll
    for (int d = 0; d < 4; d++) acc[d][0] = acc[d][1] = acc[d][2] = acc[d][3] = 0.f;
    #pragma unroll
    for (int t = 0; t < 4; t++)
      #pragma unroll
      for (int d = 0; d < 4; d++) mma(acc[d], pa[t], &vb[t][d >> 1][(d & 1) * 2]);
    // normalise and park the tile in the (now dead) q slot so the global store coalesces
    const float iA = 1.f / sA, iB = 1.f / sB;
    #pragma unroll
    for (int d = 0; d < 4; d++) {
      const int col = d * 8 + ((lane & 3) << 1);
      *(__half2*)&sq[qA * kTS + col] = __floats2half2_rn(acc[d][0] * iA, acc[d][1] * iA);
      *(__half2*)&sq[qB * kTS + col] = __floats2half2_rn(acc[d][2] * iB, acc[d][3] * iB);
    }
  }
  __syncwarp();
  #pragma unroll
  for (int i = 0; i < 8; i++)
    *(uint4*)&o[base + (size_t)(lrow + i * 8) * bank + lcol] =
        *(const uint4*)&sq[(lrow + i * 8) * kTS + lcol];
}

}  // namespace

// hd must be 32 (one mma k-step pair) and heads*hd the bank width. Stream 0 in the
// engine; no device state is cached here -- only the per-device SMEM opt-in flag,
// which is idempotent (hero_forward.cu runs one HeroForward per GPU under its mutex).
void heroBand(__half* o, const __half* q, const __half* k, const __half* v,
              const __half* bias, int N, int H, int hd, cudaStream_t stream) {
  static unsigned char optin[64] = {0};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev >= 0 && dev < 64 && !optin[dev]) {
    cudaFuncSetAttribute(hero_band_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    optin[dev] = 1;
  }
  const float scale = (float)(1.0 / sqrt((double)hd)) * kL2E;
  dim3 grid((N + kWarps - 1) / kWarps, H);
  hero_band_kernel<<<grid, kWarps * 32, kSmem, stream>>>(o, q, k, v, bias, N, H * hd, scale);
}

}  // namespace hero
}  // namespace lczero
