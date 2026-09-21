// The expert FFN as TWO GROUPED GEMMs, replacing hero_forward.cu's
// gather -> 28 per-expert cuBLAS launches on 8 streams -> scatter (:521, :536-552, :554).
//
// WHY (docs/audit/hero3_engine_384.md): on hero3 that phase is
// `experts 10.85 + gather 0.99 + scatter 0.92 = 12.76 ms of a 36.81 ms forward`
// against a 6.19 ms compute floor, and `ncu` says exactly why: the typical expert
// launches **96 blocks on 108 SMs** and reads SM 30.5% (up) / 42.3% (down). No
// individual expert fills the device; only the 8 streams keep it busy, and the
// per-class row counts are skewed so the round-robin over those streams is wrong
// by construction. `k_gather`/`k_scatter` are already at 81-85% of DRAM: they
// cannot be made faster, only deleted.
//
// THE KERNEL is the sm_80 GEMM loop from `ops/mfu60/ffn_ln.py` (the merged tree,
// branch claude/qband-lnpers @ 5cff548), lifted here TORCH-FREE and verbatim except
// for the two engine-safety fixes marked `lc0bench 0006` below. 128x128 tile, 4
// warps x 64x64 fp32 accumulators, BK 32, cp.async + ldmatrix + mma.m16n8k16,
// CUTLASS Swizzle<3,3,3>, the paired-B lane map, warp-private SMEM-staged
// `st.global.cs` epilogue. GRP=1 takes the row tile's class from a DEVICE-side
// tile->class map and its B tile from that class's weights; EPI=1 fuses mish on the
// fp32 accumulator (algebraically the same expression as lc0's `mishActivate`:
// both are z*n/(n+2) with n = e^2+2e, only the grouping differs); SCAT=1 sends each
// row's store through `dOrder`, which IS `k_scatter`.
//
// SHAPE-GENERIC. Nothing here knows 28 classes, d 1024 or dff 1280. The column
// remainder is handled the way the merged tree measured to be cheapest: the up
// weights are ZERO-PADDED to a multiple of 128 columns, so the last column tile is
// a full tile whose pad columns hold mish(0) = 0, and the down GEMM's K simply
// stops at the real dff. hero3 (dff 1280) pads by nothing; heroD4 (dff 832 -> 896)
// pays 7.7% of the up GEMM's MACs and saves a second launch.
//
// Requirements, checked by `heroGFFNSupported`: d % 128 == 0 (the down GEMM's
// column tiles), d % 32 == 0 and dff % 32 == 0 (BK), and the A buffers must carry
// kGFFNPadRows spare rows because a partial class tile reads up to BM-1 rows past
// its class (its stores are masked).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
using bf16 = __nv_bfloat16;

namespace lczero {
namespace hero {

// lc0bench 0006: the row tile. A partial class tile reads up to kGFFNTile-1 rows past
// its class (its stores are masked), so every A buffer carries kGFFNPadRows spare rows.
static const int kGFFNTile = 128;
static const int kGFFNPadRows = 256;

// HERO_GFFN_PIPE=3 selects the shallower ring for the A/B; 4 is the default.
static int gffn_pipe() {
    static const int p = getenv("HERO_GFFN_PIPE") ? atoi(getenv("HERO_GFFN_PIPE")) : 4;
    return p;
}

namespace {


#define CP16(dst, src)  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory")
#define CPCOMMIT()      asm volatile("cp.async.commit_group;\n" ::: "memory")
template<int N> __device__ __forceinline__ void cpwait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void ldsm4(uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(a));
}
__device__ __forceinline__ uint32_t swz(uint32_t c) { return c ^ ((c >> 3) & 7); }     // CUTLASS Swizzle<3,3,3>

// THE OPERAND DTYPE IS A COMPILE-TIME SWITCH. bf16 is what the torch gate compares against; f16 is what the lc0
// fork's CUDA backend runs heroD4 in. Same fragment layouts, same ldmatrix, same swizzle, same cp.async -- only the
// mma qualifier and the scalar converts change, so the kernel drops into the engine unchanged.
struct BF {
    using T = bf16; using T2 = __nv_bfloat162;
    static __device__ __forceinline__ float f(T x) { return __bfloat162float(x); }
    static __device__ __forceinline__ T c(float x) { return __float2bfloat16(x); }
    static __device__ __forceinline__ T2 c2(float a, float b) { return __floats2bfloat162_rn(a, b); }
    static __device__ __forceinline__ void mma(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
};
struct HF {
    using T = __half; using T2 = __half2;
    static __device__ __forceinline__ float f(T x) { return __half2float(x); }
    static __device__ __forceinline__ T c(float x) { return __float2half(x); }
    static __device__ __forceinline__ T2 c2(float a, float b) { return __floats2half2_rn(a, b); }
    static __device__ __forceinline__ void mma(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
};

// mish, `_ffn_up`'s expression: x*(u-1)/(u+1) with u = (1+e^x)^2, the exponent clamped at 20.
// __expf / __fdividef, i.e. ONE MUFU.EX2 and ONE MUFU.RCP, are what Triton emits and they are the whole story for
// this epilogue: with libdevice's accurate expf and div.rn the SAME kernel ran at 41.9% of peak against 73.1% for
// the identical loop with a plain store (MEASURED, 2,048 boards) -- ~35 instructions per accumulator element x 128
// elements per lane is larger than the entire k-loop.
__device__ __forceinline__ float mish(float z) {
    float e = __expf(fminf(z, 20.f));
    float t = fmaf(e, e, e + e);                          // u - 1 with u = (1+e)^2, without the (1+e)^2 - 1
    return z * __fdividef(t, t + 2.f);                    // cancellation; at the clamp t/(t+2) == 1, so z passes through
}
// The residual add, EXACTLY `_add_ln`'s: ay = round(alpha * a) then a HALF-PRECISION add onto x. Faithful to hero.py
// under autocast; doing this add in fp32 is a different function (policy argmax 98.4% -> 95.7%).
template<typename E> __device__ __forceinline__ typename E::T radd(typename E::T x, typename E::T a, float alpha) {
    return E::c(E::f(x) + E::f(E::c(E::f(a) * alpha)));
}

struct XA {
    const void *A, *B, *R;          // A [*, lda]; B per-class [C, N, K] (nn.Linear layout: B is [N, K]); R residual
    void *C;                        // out [*, ldc]
    const int *tmap, *offs, *row;   // (class, tile) per row tile; class row offsets; row scatter map
    const float *st, *cc, *gm;      // PUBLISHED [M,2] (mu, rstd) -- ONE extra fp32 buffer per layer, nothing else;
    float *part;                    // [C,N] fold correction; [N] gamma; [np, M, 2] scratch partials
    int M, N, K, lda, ldc, n0, np;
    float alpha;
};

// The C stage, shared by both kernels. SWZ=0: the +8-padded 64x72 warp buffer the plain kernel lays over its (by
// then dead) pipeline SMEM. SWZ=1: an XOR-swizzled 64x64 buffer, 8 KB per warp instead of 9, conflict-free for BOTH
// the 4-B accumulator writes and the 16-B stores -- which is what makes ring 48 KB + stage 32 KB = 80 KB still fit
// TWO blocks on an SM in the persistent kernel, and what lets the SAME 8 KB hold the epilogue's PREFETCHED residual
// tile before it holds C.
template<int SWZ> __device__ __forceinline__ uint32_t cso(int row, int q) {
    return SWZ ? (uint32_t)((((row << 3) + q) ^ (row & 7)) << 4) : (uint32_t)(row * 144 + (q << 4));
}

// EPI 0 plain | 1 mish | 2 LN-fold + mish | 3 residual add (raw R) | 4 residual add (LN of R).
// The lane op is FUSED INTO THE CONVERT -- nothing is written back to the accumulator, which is what keeps this off
// the register cap (the CuTe lane measured an 8-byte spill from the write-back form, SS11c.1).
// THE STORE LOOP'S GEOMETRY, spelled out because it is what makes the prefetch and the gamma hoist possible: chunk
// c = i*32 + lane, so q = c & 7 = lane & 7 and row = c >> 3 = 4*i + (lane >> 3). The COLUMN a lane touches is the
// same for all 16 chunks -- so gamma is 8 loads per epilogue, not 8 per chunk -- and the row advances by 4.
// PF=1: the residual tile has already been cp.async'd into `cw` during this tile's last k-tiles. The epilogue reads
// it 16 rows at a time and then REUSES those same 2 KB for C, which is the only way a full-tile prefetch fits beside
// a 48 KB ring and still leaves two blocks on an SM.
template<typename E, int WN, int EPI, int SCAT, int STATS, int SWZ, int PF>
__device__ __forceinline__ void epi(const XA &g, float acc[4][8][4], char *cw, int cls, int rbase, int hiR,
                                    int cbase, int lane, int wn, int nblk) {
    using T = typename E::T;
    const int q = lane & 7, rq = lane >> 3, col = cbase + q * 8;
    float gmv[8], mu[4][2], rs[4][2];
    if constexpr (EPI == 4) {
#pragma unroll
        for (int e = 0; e < 8; ++e) gmv[e] = g.gm[col + e];   // the lane's column is fixed: hoisted out of the loop
    }
    if constexpr (EPI == 2) {
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int r = rbase + m * 16 + (lane >> 2) + h * 8;
                if (r >= hiR) r = hiR - 1;                    // a masked row still reads something in range
                mu[m][h] = g.st[2 * r]; rs[m][h] = g.st[2 * r + 1];
            }
    }
#define CVT(M0, M1) { _Pragma("unroll") for (int n = 0; n < 8; ++n) { \
        float c0 = 0.f, c1 = 0.f; \
        if constexpr (EPI == 2) { const float *cp = g.cc + (size_t)cls * g.N + cbase + n * 8 + (lane & 3) * 2; \
                                  c0 = cp[0]; c1 = cp[1]; } \
        _Pragma("unroll") for (int m = M0; m < M1; ++m) { \
            float z0 = acc[m][n][0], z1 = acc[m][n][1], z2 = acc[m][n][2], z3 = acc[m][n][3]; \
            if constexpr (EPI == 2) {                         /* rstd * (r W' - mu * c): the whole LN, two FFMA */ \
                z0 = (z0 - mu[m][0] * c0) * rs[m][0]; z1 = (z1 - mu[m][0] * c1) * rs[m][0]; \
                z2 = (z2 - mu[m][1] * c0) * rs[m][1]; z3 = (z3 - mu[m][1] * c1) * rs[m][1]; } \
            if constexpr (EPI == 1 || EPI == 2) { z0 = mish(z0); z1 = mish(z1); z2 = mish(z2); z3 = mish(z3); } \
            char *p_ = cw + cso<SWZ>(m * 16 + (lane >> 2), n) + (lane & 3) * 4; \
            *(typename E::T2 *)p_                             = E::c2(z0, z1); \
            *(typename E::T2 *)(p_ + (SWZ ? 1024 : 8 * 144))  = E::c2(z2, z3); } } }
#define STORE(I0, I1, RV) { _Pragma("unroll") for (int i = I0; i < I1; ++i) { \
        const int row = rbase + 4 * i + rq; \
        const bool ok = row < hiR;                            /* a partial class tile: rows past it are not stored */ \
        int orow = row; float ss = 0.f, sq = 0.f; \
        if (ok) { \
            uint4 v = *(const uint4 *)(cw + cso<SWZ>(4 * i + rq, q)); \
            if constexpr (EPI >= 3) { \
                const uint4 rv_ = (RV); \
                const T *rp = (const T *)&rv_; T *vp = (T *)&v; \
                float m0 = 0.f, r0_ = 0.f; \
                if constexpr (EPI == 4) { m0 = g.st[2 * row]; r0_ = g.st[2 * row + 1]; } \
                _Pragma("unroll") for (int e = 0; e < 8; ++e) { \
                    T xx = rp[e]; \
                    if constexpr (EPI == 4)                   /* x2 = round(LN1(r1)): hero's residual is half-precision */ \
                        xx = E::c((E::f(xx) - m0) * r0_ * gmv[e]); \
                    vp[e] = radd<E>(xx, vp[e], g.alpha); \
                    if constexpr (STATS) { float f = E::f(vp[e]); ss += f; sq += f * f; } } } \
            if constexpr (SCAT) orow = g.row[row]; \
            T *gp = (T *)g.C + (size_t)orow * g.ldc + col;     /* st.global.cs: C streams past an L2 holding W */ \
            asm volatile("st.global.cs.v4.b32 [%0], {%1,%2,%3,%4};\n" :: "l"(gp), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); \
        } \
        if constexpr (STATS) {                                /* OUTSIDE the predicate: every lane must shuffle */ \
            _Pragma("unroll") for (int b = 1; b < 8; b <<= 1) {  /* 8 lanes hold the warp's 64 columns of this row */ \
                ss += __shfl_xor_sync(0xffffffff, ss, b); \
                sq += __shfl_xor_sync(0xffffffff, sq, b); } \
            if (ok && (lane & 7) == 0) { \
                float *pp = g.part + ((size_t)(nblk * WN + wn) * g.M + orow) * 2; \
                pp[0] = ss; pp[1] = sq; } } } }
    if constexpr (PF) {
#pragma unroll
        for (int mm = 0; mm < 4; ++mm) {                      // read R out of cw, then hand the same 2 KB to C
            uint4 rv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) rv[j] = *(const uint4 *)(cw + cso<SWZ>(4 * (mm * 4 + j) + rq, q));
            __syncwarp();
            CVT(mm, mm + 1)
            __syncwarp();
            STORE(mm * 4, mm * 4 + 4, rv[i - mm * 4])
        }
    } else {
        CVT(0, 4)
        __syncwarp();
        STORE(0, 16, *(const uint4 *)((const T *)g.R + (size_t)row * g.ldc + col))
    }
#undef CVT
#undef STORE
}

// EPI 0 plain | 1 mish | 2 LN-fold + mish | 3 residual add (raw R) | 4 residual add (LN of R)
template<typename E, int WM, int WN, int PIPE, int MINB, int EPI, int GRP, int SCAT, int STATS, int PF = 0,
         int AGAT = 0>
__global__ __launch_bounds__(WM * WN * 32, MINB)
void xgemm(XA g) {
    using T = typename E::T;
    constexpr int BK = 32, BM = WM * 64, BN = WN * 64, NT = WM * WN * 32;
    constexpr int ACH = BM * (BK / 8) / NT, BCH = BN * (BK / 8) / NT, RPI = NT / (BK / 8);
    constexpr int ABYTES = BM * BK * 2, STAGE = (BM + BN) * BK * 2;
    extern __shared__ char smem[];
    const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, wm = wid / WN, wn = wid % WN;

    int cls = 0, r0, hiR;
    if constexpr (GRP) {                                  // the tile->class map is DEVICE-side: the host never learns sizes
        cls = g.tmap[2 * blockIdx.y];
        // lc0bench 0006: the engine over-provisions the grid to R/BM + E row tiles so the
        // REAL tile count never has to cross to the host. Surplus tiles carry class -1 and
        // the whole block leaves here, uniformly, before any barrier.
        if (cls < 0) return;
        r0  = g.offs[cls] + g.tmap[2 * blockIdx.y + 1] * BM;
        hiR = g.offs[cls + 1];
    } else { r0 = blockIdx.y * BM; hiR = g.M; }

    const int grow = tid >> 2, gq = tid & 3;              // 4 lanes cover one row's 64-B k-window
    const T *ap = (const T *)g.A + (size_t)(r0 + grow) * g.lda + gq * 8;
    // lc0bench 0006b: AGAT GATHERS the A rows. The row tile's BM source rows are
    // g.row[r0 .. r0+BM-1] (the engine's `dOrder`), so the GEMM reads A where the trunk
    // already wrote it and the staging copy `k_gather` used to build -- one full write plus
    // one full read of an [R,d] buffer per layer, 0.95 ms on hero3 -- disappears. The tile
    // already reads BM unrelated 64-byte windows (consecutive rows are lda*2 B apart), so
    // permuting which rows they are costs no locality; it only costs ACH base pointers
    // instead of one. A partial class tile's masked rows clamp to a valid index.
    const T *apr[ACH];
    if constexpr (AGAT) {
#pragma unroll
        for (int i = 0; i < ACH; ++i) {
            int rr = r0 + grow + i * RPI;
            if (rr >= g.M) rr = g.M - 1;
            apr[i] = (const T *)g.A + (size_t)g.row[rr] * g.lda + gq * 8;
        }
    }
    const T *bp = (const T *)g.B + (size_t)cls * g.N * g.K + (size_t)(g.n0 + blockIdx.x * BN + grow) * g.K + gq * 8;
    uint32_t adst[ACH], bdst[BCH], aldm[4], bldm[4];
#pragma unroll
    for (int i = 0; i < ACH; ++i) { uint32_t c = (grow + i * RPI) * 4 + gq; adst[i] = sb + swz(c) * 16; }
#pragma unroll
    for (int i = 0; i < BCH; ++i) { uint32_t c = (grow + i * RPI) * 4 + gq; bdst[i] = sb + ABYTES + swz(c) * 16; }
    const int lr = lane & 15, lq = lane >> 4;
#pragma unroll
    for (int m = 0; m < 4; ++m) { uint32_t c = (wm * 64 + m * 16 + lr) * 4 + lq; aldm[m] = sb + swz(c) * 16; }
    const int br = (lane >> 4) * 8 + (lane & 7), bq = (lane >> 3) & 1;    // BP=1: two ADJACENT register pairs, 0 MOVs
#pragma unroll
    for (int n = 0; n < 4; ++n) { uint32_t c = (wn * 64 + n * 16 + br) * 4 + bq; bldm[n] = sb + ABYTES + swz(c) * 16; }

    float acc[4][8][4];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[m][n][i] = 0.f;
    uint32_t ra[2][4][4], rb[2][4][4];

#define ISSUE(OFF) { _Pragma("unroll") for (int i = 0; i < ACH; ++i) { \
                         if constexpr (AGAT) CP16(adst[i] + (OFF), apr[i]); \
                         else                CP16(adst[i] + (OFF), ap + (size_t)i * RPI * g.lda); } \
                     _Pragma("unroll") for (int i = 0; i < BCH; ++i) CP16(bdst[i] + (OFF), bp + (size_t)i * RPI * g.K); }
#define AADV() { ap += BK; bp += BK; \
                 if constexpr (AGAT) { _Pragma("unroll") for (int i = 0; i < ACH; ++i) apr[i] += BK; } }
#define LDF(BUF, KB) { _Pragma("unroll") for (int m = 0; m < 4; ++m) ldsm4(ra[BUF][m][0], ra[BUF][m][1], ra[BUF][m][2], ra[BUF][m][3], (aldm[m] + roff) ^ (KB * 32)); \
                       _Pragma("unroll") for (int n = 0; n < 4; ++n) ldsm4(rb[BUF][n][0], rb[BUF][n][1], rb[BUF][n][2], rb[BUF][n][3], (bldm[n] + roff) ^ (KB * 32)); }
#define MMA(BUF) { _Pragma("unroll") for (int m = 0; m < 4; ++m) _Pragma("unroll") for (int n = 0; n < 8; ++n) \
                       E::mma(acc[m][n], &ra[BUF][m][0], rb[BUF][n >> 1][(n & 1) * 2], rb[BUF][n >> 1][(n & 1) * 2 + 1]); }

#define KSTEP() { LDF(1, 1) \
        if (ktile < NKT) { ISSUE(woff); AADV(); } \
        CPCOMMIT();                                       /* ALWAYS: wait_group counts GROUPS, empty ones included */ \
        ++ktile; \
        woff = (woff + STAGE == PIPE * STAGE) ? 0 : woff + STAGE; \
        MMA(0) \
        roff = (roff + STAGE == PIPE * STAGE) ? 0 : roff + STAGE; \
        cpwait<PIPE - 2>(); __syncthreads(); LDF(0, 0) MMA(1) }
    // PF 1: the epilogue's own tile-local input, issued as ONE cp.async group two k-tiles from the end. The
    // k-loop's next `cp.async.wait_group PIPE-2` retires it on the way past (groups complete in commit order), so
    // the epilogue's residual read is pipelined exactly like the GEMM's own operands instead of being exposed. It
    // costs SMEM the ring wants: the stage may no longer alias it, so PF 1 is only affordable at PIPE 3.
    // PF 2: the same lead, no SMEM at all -- `prefetch.global.L2` pulls the tile's 64 lines into L2 and the
    // epilogue's ordinary `ld.global` then hits L2 instead of HBM. Zero registers, zero SMEM, so PIPE stays 4.
#define PREFA(I) ((const T *)g.R + (size_t)(r0 + wm * 64 + 4 * (I) + (lane >> 3)) * g.ldc \
                  + g.n0 + blockIdx.x * BN + wn * 64 + (lane & 7) * 8)
#define PREF() { if constexpr (PF == 1) { \
        _Pragma("unroll") for (int i = 0; i < 16; ++i) \
            CP16(sb + PIPE * STAGE + wid * 8192 + cso<1>(4 * i + (lane >> 3), lane & 7), PREFA(i)); \
        CPCOMMIT(); \
    } else {                                              /* NO memory clobber: it re-materialises the whole */ \
        const T *pb_ = PREFA(0);                          /* k-loop's cached state and cost a 96-B spill */ \
        _Pragma("unroll") for (int i = 0; i < 16; ++i) \
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(pb_ + (size_t)i * 4 * g.ldc)); \
    } }
    const int NKT = g.K / BK;
    uint32_t woff = 0, roff = 0;
    int ktile = PIPE - 1;
#pragma unroll
    for (int s = 0; s < PIPE - 1; ++s) { ISSUE(woff); CPCOMMIT(); woff += STAGE; AADV(); }
    cpwait<PIPE - 2>();
    __syncthreads();
    LDF(0, 0)
    constexpr int PLEAD = PF == 1 ? 2 : (PF == 2 ? 4 : 0);
#pragma unroll 1
    for (int t = 0; t < NKT - PLEAD; ++t) KSTEP()
    if constexpr (PF) {
        PREF()
#pragma unroll 1
        for (int t = NKT - PLEAD; t < NKT; ++t) KSTEP()
    }
#undef ISSUE
#undef AADV
#undef LDF
#undef MMA
#undef KSTEP
#undef PREF
#undef PREFA

    // ---------------- epilogue ----------------
    if constexpr (PF == 1) {                              // the stage does NOT alias the ring: no barrier, R is there
        epi<E, WN, EPI, SCAT, STATS, 1, 1>(g, acc, (char *)smem + PIPE * STAGE + wid * 8192, cls, r0 + wm * 64, hiR,
                                           g.n0 + blockIdx.x * BN + wn * 64, lane, wn, blockIdx.x);
    } else {                                              // ... otherwise it is laid over the now-dead pipeline SMEM
        __syncthreads();                                  // and other warps are still reading the ring
        epi<E, WN, EPI, SCAT, STATS, 0, 0>(g, acc, (char *)smem + wid * 64 * 72 * 2, cls, r0 + wm * 64, hiR,
                                           g.n0 + blockIdx.x * BN + wn * 64, lane, wn, blockIdx.x);
    }
}

template<typename E, int WM, int WN, int PIPE, int MINB, int EPI, int GRP, int SCAT, int STATS, int PF = 0,
         int AGAT = 0>
static void go(const XA &g, int ncol, int nrow, cudaStream_t s, int *occ) {
    constexpr int BM = WM * 64, BN = WN * 64, NT = WM * WN * 32;
    constexpr int SE = (NT / 32) * 64 * 72 * 2, SP = (BM + BN) * 32 * 2 * PIPE;
    constexpr int SMEM = PF == 1 ? SP + (NT / 32) * 8192 : (SP > SE ? SP : SE);  // PF 1: stage cannot alias the ring
    auto k = xgemm<E, WM, WN, PIPE, MINB, EPI, GRP, SCAT, STATS, PF, AGAT>;
    // lc0bench 0006: PER DEVICE, not once per process. The multiplexing backend builds one
    // HeroForward per GPU and the >48 KB dynamic-SMEM opt-in is a per-device property, so a
    // process-wide flag silently skips it on every GPU but the first (same scar as
    // hero_band.cu's `optin[]`; a missed opt-in is an invalid-argument launch failure).
    static unsigned char optin[64] = {0};
    int dev_ = 0;
    cudaGetDevice(&dev_);
    if (dev_ >= 0 && dev_ < 64 && !optin[dev_]) {
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        optin[dev_] = 1;
    }
    if (occ) { cudaOccupancyMaxActiveBlocksPerMultiprocessor(occ, k, NT, SMEM); occ[1] = NT; occ[2] = SMEM; return; }
    k<<<dim3(ncol, nrow), NT, SMEM, s>>>(g);
}


// ---------------------------------------------------------------- engine glue
// The tile->class map, built ON DEVICE from the counting sort's own histogram, so the
// 28 counts never cross to the host: `hero_forward.cu:486`'s blocking D2H and `:488`'s
// H2D both disappear, and with them the one synchronous point between route and trunk.
// One thread is right here: E is 28 and the whole map is ~220 entries, once per forward
// (the route runs once, not per layer).
__global__ void k_gffn_map(const int *cnt, int *offs, int *cur, int *tmap, int E, int ntmax, int bm) {
    if (threadIdx.x) return;
    int acc = 0, t = 0;
    for (int e = 0; e < E; ++e) {
        offs[e] = acc; cur[e] = acc;                      // `cur` is k_scatter_order's cursor
        const int c = cnt[e], nt = (c + bm - 1) / bm;     // an EMPTY class gets ZERO tiles
        for (int i = 0; i < nt && t < ntmax; ++i) { tmap[2 * t] = e; tmap[2 * t + 1] = i; ++t; }
        acc += c;
    }
    offs[E] = acc;
    for (; t < ntmax; ++t) { tmap[2 * t] = -1; tmap[2 * t + 1] = 0; }   // surplus tiles exit
}

// debug only (HERO_GFFN_CHECK): max |a-b| and max |a-b|/(|b|+eps) over n halves.
__global__ void k_gffn_cmp(const __half *a, const __half *b, int n, float *out) {
    float ma = 0.f, mr = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float x = __half2float(a[i]), y = __half2float(b[i]), d = fabsf(x - y);
        ma = fmaxf(ma, d); mr = fmaxf(mr, d / (fabsf(y) + 1e-3f));
    }
    for (int k = 16; k; k >>= 1) {
        ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, k));
        mr = fmaxf(mr, __shfl_xor_sync(0xffffffffu, mr, k));
    }
    if ((threadIdx.x & 31) == 0) { atomicMax((int *)out, __float_as_int(ma)); atomicMax((int *)(out + 1), __float_as_int(mr)); }
}

}  // namespace

int  heroGFFNPad(int dff) { return (dff + 127) / 128 * 128; }
int  heroGFFNTileRows() { return kGFFNTile; }
int  heroGFFNPadRows()  { return kGFFNPadRows; }
bool heroGFFNSupported(int d, int dff) {
    return d >= 128 && d % 128 == 0 && dff >= 32 && dff % 32 == 0 && d % 32 == 0;
}

void heroGFFNMap(const int *cnt, int *offs, int *cur, int *tmap, int E, int ntmax, cudaStream_t s) {
    k_gffn_map<<<1, 32, 0, s>>>(cnt, offs, cur, tmap, E, ntmax, kGFFNTile);
}

// h[r, 0:dp] = mish(xs[r] . up[class(r)]^T), rows class-SORTED, dp = heroGFFNPad(dff).
// `row` (the engine's dOrder) non-null GATHERS the A rows in the kernel's own load, which is
// what deletes k_gather; pass nullptr to read `xs` already class-sorted (the old contract).
void heroGFFNUp(__half *h, const __half *xs, const __half *upw, const int *tmap, const int *offs,
                const int *row, int R, int dp, int d, int ntiles, cudaStream_t s) {
    XA g{};
    g.A = xs; g.B = upw; g.C = h; g.tmap = tmap; g.offs = offs; g.row = row;
    g.M = R; g.N = dp; g.K = d; g.lda = d; g.ldc = dp;
    const int p = gffn_pipe();
    if (row) { if (p == 3) go<HF, 2, 2, 3, 2, 1, 1, 0, 0, 0, 1>(g, dp / 128, ntiles, s, nullptr);
               else        go<HF, 2, 2, 4, 2, 1, 1, 0, 0, 0, 1>(g, dp / 128, ntiles, s, nullptr); }
    else     { if (p == 3) go<HF, 2, 2, 3, 2, 1, 1, 0, 0>(g, dp / 128, ntiles, s, nullptr);
               else        go<HF, 2, 2, 4, 2, 1, 1, 0, 0>(g, dp / 128, ntiles, s, nullptr); }
}

// out[dOrder[r]] = h[r, 0:dff] . dn[class(r)]^T -- the scatter IS the epilogue (SCAT=1).
void heroGFFNDown(__half *out, const __half *h, const __half *dnw, const int *tmap, const int *offs,
                  const int *row, int R, int d, int dff, int ldh, int ntiles, cudaStream_t s) {
    XA g{};
    g.A = h; g.B = dnw; g.C = out; g.tmap = tmap; g.offs = offs; g.row = row;
    g.M = R; g.N = d; g.K = dff; g.lda = ldh; g.ldc = d;
    if (gffn_pipe() == 3) go<HF, 2, 2, 3, 2, 0, 1, 1, 0>(g, d / 128, ntiles, s, nullptr);
    else                  go<HF, 2, 2, 4, 2, 0, 1, 1, 0>(g, d / 128, ntiles, s, nullptr);
}

void heroGFFNCompare(const __half *a, const __half *b, int n, float *scratch, float *host2) {
    cudaMemset(scratch, 0, 2 * sizeof(float));
    k_gffn_cmp<<<256, 256>>>(a, b, n, scratch);
    cudaMemcpy(host2, scratch, 2 * sizeof(float), cudaMemcpyDeviceToHost);
}

}  // namespace hero
}  // namespace lczero
