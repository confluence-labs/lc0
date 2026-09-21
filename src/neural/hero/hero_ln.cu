// The trunk's "scaled residual add, then LayerNorm" as ONE bandwidth-rate kernel,
// replacing lc0's generic `LayerNorm<half_t>` at hero_forward.cu:519 and :555.
//
// WHY (docs/audit/hero3_engine_384.md): those 30 calls cost 4.18 ms of a 36.8 ms
// forward on hero3 and 4.24 of 58.2 on heroD4 -- 1.08 TB/s = 53% of an A100's HBM,
// 2.6x the 1.61 ms the traffic is worth. `ncu` reads the stock kernel at SM 47.5% /
// DRAM 58.3% in a (3072) x (32,2,8) grid: 8 rows per 512-thread block, TWO
// __syncthreads-based block reductions per row, and gamma re-read per row. It is a
// reduction wearing a stream's clothes.
//
// This kernel: ONE WARP OWNS ONE ROW. The row is read once through 16-byte vectors
// into registers (CH chunks of 8 halves per lane), both reductions are
// `__shfl_xor` butterflies -- no shared memory and no __syncthreads anywhere -- and
// the second pass re-reads the row from registers, not from HBM. Traffic is exactly
// what the operator is worth: read input + skip, write out.
//
// ARITHMETIC, faithful to `common_kernels.cu:945-1105` term for term, because the
// gate is `hero_parity_gate` against the torch oracle and this must not move it:
//   v   = (float)input * alpha + (float)skip           (bias is zbuf = 0, act NONE)
//   mu  = sum(v)/C ; var = sum((v-mu)^2)/C             (both fp32, two passes)
//   out = ((v - mu) / sqrt(var + eps)) * gamma         (beta is zbuf = 0)
// The scale is applied in fp32 on the converted input, NOT as a half-precision add
// -- that is lc0's order and the engine's ladder is calibrated on it. Only the
// reduction ORDER differs from lc0's (a warp butterfly instead of a shared-memory
// tree), which is worth a few fp32 ULPs on mu and var.
//
// SHAPE-GENERIC: any C with C % 16 == 0 and C <= 2048 (CH <= 8 chunks per lane),
// any row count. hero3 d 1024 -> CH 4 (32 floats/lane, exact); heroD4 d 640 -> CH 3
// with the last chunk half-masked; d 768/1280/1536 all land on an exact or masked
// chunk count. The caller's fallback keeps lc0's kernel for anything else.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lczero {
namespace hero {
namespace {

constexpr int kWarpsPerBlock = 8;          // 256 threads; one row each

template <int CH>
__global__ __launch_bounds__(kWarpsPerBlock * 32, 4) void hero_addln_kernel(
    __half* __restrict__ o, const __half* __restrict__ x,
    const __half* __restrict__ s, const __half* __restrict__ gam, int N, int C,
    float eps, float alpha) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= N) return;                    // warp-uniform: no barrier is skipped
  const int nv = C >> 3;                   // 16-byte vectors per row
  const size_t base = (size_t)row * (size_t)C;

  float v[CH][8];
  float sum = 0.f;
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    const int c = lane + i * 32;
    if (c < nv) {
      const uint4 xi = *(const uint4*)(x + base + (size_t)c * 8);
      const uint4 si = *(const uint4*)(s + base + (size_t)c * 8);
      const __half* xh = (const __half*)&xi;
      const __half* sh = (const __half*)&si;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        v[i][j] = fmaf(__half2float(xh[j]), alpha, __half2float(sh[j]));
        sum += v[i][j];
      }
    }
  }
#pragma unroll
  for (int k = 16; k; k >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, k);
  const float mean = sum / (float)C;

  float q = 0.f;
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    if (lane + i * 32 < nv) {
#pragma unroll
      for (int j = 0; j < 8; ++j) { const float d = v[i][j] - mean; q += d * d; }
    }
  }
#pragma unroll
  for (int k = 16; k; k >>= 1) q += __shfl_xor_sync(0xffffffffu, q, k);
  const float den = sqrtf(q / (float)C + eps);

#pragma unroll
  for (int i = 0; i < CH; ++i) {
    const int c = lane + i * 32;
    if (c < nv) {
      const uint4 gi = *(const uint4*)(gam + (size_t)c * 8);
      const __half* gh = (const __half*)&gi;
      __half out8[8];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        out8[j] = __float2half((v[i][j] - mean) / den * __half2float(gh[j]));
      *(uint4*)(o + base + (size_t)c * 8) = *(const uint4*)out8;
    }
  }
}

}  // namespace

// true when this kernel covers the shape; the caller keeps lc0's LayerNorm otherwise.
bool heroAddLNSupported(int C) { return C % 16 == 0 && C >= 16 && C <= 2048; }

void heroAddLN(__half* o, const __half* x, const __half* skip, const __half* gam,
               int N, int C, float eps, float alpha, cudaStream_t stream) {
  const int ch = ((C >> 3) + 31) / 32;               // chunks of 8 halves per lane
  const dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock), blk(kWarpsPerBlock * 32);
#define L(K) hero_addln_kernel<K><<<grid, blk, 0, stream>>>(o, x, skip, gam, N, C, eps, alpha)
  switch (ch) {
    case 1: L(1); break;  case 2: L(2); break;  case 3: L(3); break;  case 4: L(4); break;
    case 5: L(5); break;  case 6: L(6); break;  case 7: L(7); break;  default: L(8); break;
  }
#undef L
}

}  // namespace hero
}  // namespace lczero
