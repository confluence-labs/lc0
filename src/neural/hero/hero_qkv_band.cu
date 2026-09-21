// The fused q|k|v projection + attention band for heroD4 / hero3 on sm_80, distilled from ops/mfu60/qkv_band.py's
// variant 10 (8 warps, 128x128 tile, PIPE 4, q and k in registers, v through SMEM, stencil bias staged by cp.async).
// No torch, no ThunderKittens, no CUTLASS: `nvcc -arch=sm_80 -c hero_qkv_band.cu` is the whole build.
//
// ONE BLOCK = 2 boards x 4 heads = a 128x128 tile. The GEMM agent's sm_80 main loop (BK 32, cp.async pipeline,
// CUTLASS Swizzle<3,3,3>, ldmatrix + mma.m16n8k16, paired-B lane map -- docs/audit/herod4_a100/gemm_agent_report.md)
// is run THREE times over the same x rows, and the band then runs inside the same program:
//   * the q accumulator IS the next mma's A operand (the FlashAttention-2 fragment identity), so q never leaves
//     registers;
//   * the k accumulator IS its B operand (mma.row.col wants B as (k=dim, n=key) = K[key][dim] row-major, which is
//     the accumulator's own lane map), so S = q.k^T runs entirely out of registers -- no k tile, no ldmatrix;
//   * only v round-trips, through a 32 KB SMEM tile read back with ldmatrix.trans (O = P.V needs V^T);
//   * P (the softmax output) becomes an A operand by the q identity again.
// Pass order is q, v, k: whatever is held in registers must not be alive across another pass's accumulator.
//
// SCARS (each one cost a run; docs/audit/herod4_a100.md SS8-SS11c):
//   * every k-tile commits a cp.async group even when it copies nothing -- `wait_group` counts GROUPS, and skipping
//     the commit makes the wait return early and the last stages are read before they land (a silent race);
//   * the B-fragment lane map is permuted (row = (lane>>4)*8 + (lane&7), k-group = (lane>>3)&1) or ptxas emits two
//     MOVs per HMMA: 66.0% -> 72.7% of peak;
//   * the XOR swizzle is on 16-byte chunks so a stage is exactly (BM+BN)*BK*2 bytes;
//   * the SMEM bias tile has row stride 72, not 64, or the band's LDS.32 is 8-way bank-conflicted;
//   * addresses are computed once (on sm_80 instructions and registers are one budget: 224 regs, no spills);
//   * out-of-range rows CLAMP on the read (a cp.async from a bad address faults) and are masked on the store.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include "hero_qkv_band.h"

namespace {

using bf16 = __nv_bfloat16;
template <int ET> struct et_t { using T = bf16; };
template <> struct et_t<1> { using T = __half; };

#define CP16(dst, src) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory")
#define CPCOMMIT() asm volatile("cp.async.commit_group;\n" ::: "memory")
template <int N> __device__ __forceinline__ void cpwait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm4(uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(a));
}
// v is stored [key][dim]; the B operand of O = P.V wants (k = key, n = dim), i.e. the transpose: one instruction.
__device__ __forceinline__ void ldsm4t(uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(a));
}
template <int ET> __device__ __forceinline__ void mma16816(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <> __device__ __forceinline__ void mma16816<1>(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float ex2(float x) { float r; asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
template <int ET> __device__ __forceinline__ uint32_t pk2(float a, float b) {
    union { __nv_bfloat162 h; uint32_t u; } c; c.h = __floats2bfloat162_rn(a, b); return c.u;
}
template <> __device__ __forceinline__ uint32_t pk2<1>(float a, float b) {
    union { __half2 h; uint32_t u; } c; c.h = __floats2half2_rn(a, b); return c.u;
}
template <int ET> __device__ __forceinline__ float un2(const void *p, int i) {
    return __bfloat162float(((const bf16 *)p)[i]);
}
template <> __device__ __forceinline__ float un2<1>(const void *p, int i) { return __half2float(((const __half *)p)[i]); }

__device__ __forceinline__ uint32_t swz(uint32_t c) { return c ^ ((c >> 3) & 7); }    // BK=32 stage: 4 chunks per row
__device__ __forceinline__ uint32_t swzw(uint32_t c) { return c ^ ((c >> 4) & 7); }   // 128-col tile: 16 chunks per row

#define QRED_MAX(v) { v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1)); v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2)); }
#define QRED_SUM(v) { v += __shfl_xor_sync(0xffffffff, v, 1); v += __shfl_xor_sync(0xffffffff, v, 2); }

// ===================== lc0bench 0011: the WARP-SPECIALIZED arm (qkv_band.py v52, claude/qband-wspec @ 8651bfc)
// Same contract, same output, a different schedule: warps 0-3 PRODUCE q, k, v on the agent's 4-warp 128x128 loop
// and hand them over as three 32 KB SMEM tiles; warps 4-7 CONSUME one head each and interleave BOTH boards of the
// tile, so one (board, head) chain's dependent stalls are filled by the other's work. Named barriers carry the
// handoff. 160 KB of dynamic SMEM (ring 64 + handoff 96) against the fused arm's 132, 232 registers, no spills;
// MEASURED 1.03x of variant 10 per layer at heroD4's shape.
//
// It is placed BEFORE the fused arm's namespace-scope constants ON PURPOSE: every constant this kernel needs it
// declares itself, and at namespace scope a name that is not yet declared cannot be found, so nothing of v10's
// (VOFF, BOFF, BSTR, CS, WTN ...) can silently shadow into it.
template <int L> __device__ __forceinline__ uint32_t swzwL(uint32_t c) { return c ^ ((c >> L) & 7); }
#define EXP2(x) (BMODE == 2 ? (x) : ex2(x))
#define BARA(id, n) asm volatile("bar.arrive %0, %1;\n" :: "n"(id), "n"(n) : "memory")
#define BARS(id, n) asm volatile("bar.sync %0, %1;\n" :: "n"(id), "n"(n) : "memory")

// FBUF = fragment buffers in the producer's main loop (2 = ldmatrix of k-atom j+1 overlaps the mma of j; 1 = one
// buffer, 32 registers cheaper). SPLIT picks how much of the two chains is live at once: 0 = both boards' scores
// interleaved at the mma (the design; 64 score registers on top of P's 128), 1 = the same with a scheduling
// barrier between query atoms (ptxas interleaves the unrolled m-loop and that is what overruns the file), 2 = one
// board at a time (32 score registers -- the control for what the interleave is worth). OP = query atoms per P.V
// group (accumulator reuse distance 8 x OP). HALF = how many query-atom groups the band is peeled into:
// P for the whole 64x64 band of both boards is 128 registers, which is the single largest thing the consumer
// holds, and HALF 2 halves it -- at the price of only the FIRST group's scores hiding under the v pass. OSTG 1 = o staged block-wide in the now-dead cp.async ring for 16-B stores; 0 = 4-B st.global.cs
// straight out of the accumulator (PERSIST has no dead ring to stage in). BMODE 1 = band ablated: the consumers
// only copy the handoff out, so what is left is the projections + the handoff + the o store; 3 = no row max.
template<int PIPE, int BBF, int DBG, int ET = 0, int BMODE = 0, int SPLIT = 0, int OP = 2, int HALF = 2,
         int FBUF = 2, int OSTG = 1, int PERSIST = 0, int PHASED = 0>
__global__ __launch_bounds__(256, 1)
void qkvband_ws(const typename et_t<ET>::T *__restrict__ X, const typename et_t<ET>::T *__restrict__ WQ,
                const typename et_t<ET>::T *__restrict__ WK, const typename et_t<ET>::T *__restrict__ WV,
                const void *__restrict__ BIAS, typename et_t<ET>::T *__restrict__ O,
                typename et_t<ET>::T *__restrict__ DQ, typename et_t<ET>::T *__restrict__ DK,
                typename et_t<ET>::T *__restrict__ DV, int M, int N, int K, float scale) {
    using E = typename et_t<ET>::T;
    constexpr int BM = 128, BN = 128, BK = 32, NT = 128;        // NT = threads in ONE warp group
    constexpr int CPR = BN / 8, LCPR = 4;                       // 16-B chunks per handoff row, and its log2
    constexpr int ACH = BM * (BK / 8) / NT, BCH = BN * (BK / 8) / NT, RPI = NT / (BK / 8);
    constexpr int ABYTES = BM * BK * 2, STAGE = (BM + BN) * BK * 2, SPIPE = PIPE * STAGE, TILE = BM * BN * 2;
    constexpr int QOFF = SPIPE, KOFF = QOFF + TILE, VOFF = KOFF + TILE, CS = BN + 8;
    constexpr int RSTR = 16 * CPR * 16;                         // bytes per 16 handoff rows = one atom = 4096
    extern __shared__ char smem[];
    const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int NCOL = N / BN, NTILE = PERSIST ? ((M + BM - 1) / BM) * NCOL : 1;

    // ================= PHASE A: all 8 warps project q and k (PHASED only) =========================
    if constexpr (PHASED) {
        constexpr int NT8 = 256, MA = 4, NA8 = 4, WTN8 = 32;
        constexpr int ACH8 = BM * (BK / 8) / NT8, BCH8 = BN * (BK / 8) / NT8, RPI8 = NT8 / (BK / 8);
        const int tid = threadIdx.x, wm = wid >> 2, wn = wid & 3;     // v10's grid: 2 rows x 4 cols, 64 x 32
        const int lr = lane & 15, lq = lane >> 4, qr = lane >> 2, qc = (lane & 3) * 2;
        const int grow = tid >> 2, gq = tid & 3;
        const int br = (lane >> 4) * 8 + (lane & 7), bq = (lane >> 3) & 1;
        const uint32_t adst0 = sb + swz((uint32_t)(grow * 4 + gq)) * 16;
        const uint32_t bdst0 = sb + ABYTES + swz((uint32_t)(grow * 4 + gq)) * 16;
        const uint32_t aldm0 = sb + swz((uint32_t)((wm * 64 + lr) * 4 + lq)) * 16;
        const uint32_t bldm0 = sb + ABYTES + swz((uint32_t)((wn * WTN8 + br) * 4 + bq)) * 16;
        const uint32_t s8 = qr & 7;
        const uint32_t wb = (wm * 64 + qr) * (CPR * 16) + (((wn * 4) ^ (s8 & 4)) * 16) + qc * 2, sx = (s8 & 3) * 16;
        const int bcol = (int)blockIdx.x, brow = (int)blockIdx.y;
        const int arow = brow * BM + grow < M ? brow * BM + grow : M - 1;
        const E *const ap0 = X + (size_t)arow * K + gq * 8;
        const size_t boff = (size_t)(bcol * BN + grow) * K + gq * 8;
        float acc[MA][NA8][4];
        uint32_t ra[FBUF][4][4], rb[FBUF][NA8 / 2][4];
        const E *ap, *bp;
        uint32_t woff, roff;
        int ktile;
        const int NKT = K / BK;
#define AISSUE(OFF) { _Pragma("unroll") for (int i = 0; i < ACH8; ++i) CP16(adst0 + i * (RPI8 * 4 * 16) + (OFF), ap + (size_t)i * RPI8 * K); \
                      _Pragma("unroll") for (int i = 0; i < BCH8; ++i) CP16(bdst0 + i * (RPI8 * 4 * 16) + (OFF), bp + (size_t)i * RPI8 * K); }
#define ALDF(BUF, KB) { _Pragma("unroll") for (int m = 0; m < 4; ++m) ldsm4(ra[BUF][m][0], ra[BUF][m][1], ra[BUF][m][2], ra[BUF][m][3], (aldm0 + m * 1024 + roff) ^ (KB * 32)); \
                        _Pragma("unroll") for (int n = 0; n < NA8 / 2; ++n) ldsm4(rb[BUF][n][0], rb[BUF][n][1], rb[BUF][n][2], rb[BUF][n][3], (bldm0 + n * 1024 + roff) ^ (KB * 32)); }
#define AMMA(BUF) { _Pragma("unroll") for (int m = 0; m < MA; ++m) _Pragma("unroll") for (int n = 0; n < NA8; ++n) \
                        mma16816<ET>(acc[m][n], &ra[BUF][m][0], rb[BUF][n >> 1][(n & 1) * 2], rb[BUF][n >> 1][(n & 1) * 2 + 1]); }
#define APASS(W) { \
    ap = ap0; bp = (W) + boff; woff = 0; roff = 0; ktile = PIPE - 1; \
    _Pragma("unroll") for (int m = 0; m < MA; ++m) _Pragma("unroll") for (int n = 0; n < NA8; ++n) \
        _Pragma("unroll") for (int i = 0; i < 4; ++i) acc[m][n][i] = 0.f; \
    _Pragma("unroll") for (int t = 0; t < PIPE - 1; ++t) { AISSUE(woff); CPCOMMIT(); woff += STAGE; ap += BK; bp += BK; } \
    cpwait<PIPE - 2>(); __syncthreads(); ALDF(0, 0) \
    _Pragma("unroll 1") for (int t = 0; t < NKT; ++t) { \
        ALDF(1, 1) \
        if (ktile < NKT) { AISSUE(woff); ap += BK; bp += BK; } \
        CPCOMMIT(); ++ktile; \
        woff = (woff + STAGE == SPIPE) ? 0 : woff + STAGE; \
        AMMA(0) \
        roff = (roff + STAGE == SPIPE) ? 0 : roff + STAGE; \
        cpwait<PIPE - 2>(); __syncthreads(); \
        ALDF(0, 0) AMMA(1) \
    } \
    cpwait<0>(); __syncthreads(); }
#define ASTAGE(BASE) { _Pragma("unroll") for (int m = 0; m < MA; ++m) _Pragma("unroll") for (int n = 0; n < NA8; ++n) { \
        char *pp = smem + (BASE) + wb + m * RSTR + ((n * 16) ^ sx); \
        *(uint32_t *)pp = pk2<ET>(acc[m][n][0], acc[m][n][1]); \
        *(uint32_t *)(pp + 8 * CPR * 16) = pk2<ET>(acc[m][n][2], acc[m][n][3]); } }
#define ADBG(P) if (DBG && (P)) { _Pragma("unroll") for (int m = 0; m < MA; ++m) _Pragma("unroll") for (int n = 0; n < NA8; ++n) { \
        if (brow * BM + wm * 64 + m * 16 + qr + 8 >= M) continue; \
        E *g = (P) + (size_t)(brow * BM + wm * 64 + m * 16 + qr) * N + bcol * BN + wn * WTN8 + n * 8 + qc; \
        *(uint32_t *)g = pk2<ET>(acc[m][n][0], acc[m][n][1]); *(uint32_t *)(g + 8 * (size_t)N) = pk2<ET>(acc[m][n][2], acc[m][n][3]); } }
        APASS(WQ) ADBG(DQ) ASTAGE(QOFF)
        APASS(WK) ADBG(DK) ASTAGE(KOFF)
        __syncthreads();                   // q and k are visible to every warp; no handoff barrier needed
#undef AISSUE
#undef ALDF
#undef AMMA
#undef APASS
#undef ASTAGE
#undef ADBG
    }

    if (wid < 4) {
    // ================= PRODUCERS =================================================================
        const int tid = threadIdx.x, wm = wid >> 1, wn = wid & 1;     // 2 x 2 warp grid, warp tile 64 x 64
        constexpr int MA = 4, NA = 8, WTN = 64;
        const int lr = lane & 15, lq = lane >> 4, qr = lane >> 2, qc = (lane & 3) * 2;
        const int grow = tid >> 2, gq = tid & 3;                      // 4 lanes cover one row's 64-B k-window
        const int br = (lane >> 4) * 8 + (lane & 7), bq = (lane >> 3) & 1;    // paired-B lane map
        // ---- one base per stream; the +i*RPI rows / +m*16 rows strides are affine THROUGH the swizzle
        const uint32_t adst0 = sb + swz((uint32_t)(grow * 4 + gq)) * 16;
        const uint32_t bdst0 = sb + ABYTES + swz((uint32_t)(grow * 4 + gq)) * 16;
        const uint32_t aldm0 = sb + swz((uint32_t)((wm * 64 + lr) * 4 + lq)) * 16;
        const uint32_t bldm0 = sb + ABYTES + swz((uint32_t)((wn * 64 + br) * 4 + bq)) * 16;
        const uint32_t wb = (wm * 64 + qr) * (CPR * 16) + wn * (WTN / 8 * 16) + qc * 2, sx = (qr & 7) * 16;
        float acc[MA][NA][4];
        uint32_t ra[FBUF][4][4], rb[FBUF][NA / 2][4];
        const E *ap, *bp;
        uint32_t woff, roff;
        int ktile;
        const int NKT = K / BK;
#define WISSUE(OFF) { _Pragma("unroll") for (int i = 0; i < ACH; ++i) CP16(adst0 + i * (RPI * 4 * 16) + (OFF), ap + (size_t)i * RPI * K); \
                      _Pragma("unroll") for (int i = 0; i < BCH; ++i) CP16(bdst0 + i * (RPI * 4 * 16) + (OFF), bp + (size_t)i * RPI * K); }
#define WLDF(BUF, KB) { _Pragma("unroll") for (int m = 0; m < 4; ++m) ldsm4(ra[BUF][m][0], ra[BUF][m][1], ra[BUF][m][2], ra[BUF][m][3], (aldm0 + m * 1024 + roff) ^ (KB * 32)); \
                        _Pragma("unroll") for (int n = 0; n < NA / 2; ++n) ldsm4(rb[BUF][n][0], rb[BUF][n][1], rb[BUF][n][2], rb[BUF][n][3], (bldm0 + n * 1024 + roff) ^ (KB * 32)); }
#define WMMA(BUF) { _Pragma("unroll") for (int m = 0; m < MA; ++m) _Pragma("unroll") for (int n = 0; n < NA; ++n) \
                        mma16816<ET>(acc[m][n], &ra[BUF][m][0], rb[BUF][n >> 1][(n & 1) * 2], rb[BUF][n >> 1][(n & 1) * 2 + 1]); }
#define WHEAD(W) ap = ap0; bp = (W) + boff; woff = 0; roff = 0; ktile = PIPE - 1; \
    _Pragma("unroll") for (int m = 0; m < MA; ++m) _Pragma("unroll") for (int n = 0; n < NA; ++n) \
        _Pragma("unroll") for (int i = 0; i < 4; ++i) acc[m][n][i] = 0.f; \
    _Pragma("unroll") for (int s = 0; s < PIPE - 1; ++s) { WISSUE(woff); CPCOMMIT(); woff += STAGE; ap += BK; bp += BK; }
#define WSTEP { if (ktile < NKT) { WISSUE(woff); ap += BK; bp += BK; } \
                CPCOMMIT();                       /* ALWAYS: wait_group counts GROUPS, empty ones included */ \
                ++ktile; woff = (woff + STAGE == SPIPE) ? 0 : woff + STAGE; }
// One projection pass, producer-only: every barrier in it is the 128-thread named barrier 1, so the consumers
// (which are inside the band of the SAME tile) are never dragged into the pipeline's step.
#define WPASS(W) { WHEAD(W) \
    if constexpr (FBUF == 2) { \
        cpwait<PIPE - 2>(); BARS(1, NT); WLDF(0, 0) \
        _Pragma("unroll 1") for (int t = 0; t < NKT; ++t) { \
            WLDF(1, 1) WSTEP WMMA(0) \
            roff = (roff + STAGE == SPIPE) ? 0 : roff + STAGE; \
            cpwait<PIPE - 2>(); BARS(1, NT); \
            WLDF(0, 0) WMMA(1) \
        } \
    } else { \
        _Pragma("unroll 1") for (int t = 0; t < NKT; ++t) { \
            cpwait<PIPE - 2>(); BARS(1, NT); \
            WLDF(0, 0) WSTEP WMMA(0) \
            WLDF(0, 1) WMMA(0) \
            roff = (roff + STAGE == SPIPE) ? 0 : roff + STAGE; \
        } \
    } \
    cpwait<0>(); BARS(1, NT); }
// acc -> the bf16 handoff tile at BASE, [row][col]; swzw's flipped bits come from qr alone, so the whole 64x64
// store is ONE base register + constants, and the column swizzle is a constant XOR on the chunk offset.
#define WSTAGE(BASE) { _Pragma("unroll") for (int m = 0; m < MA; ++m) _Pragma("unroll") for (int n = 0; n < NA; ++n) { \
        char *pp = smem + (BASE) + wb + m * RSTR + ((n * 16) ^ sx); \
        *(uint32_t *)pp = pk2<ET>(acc[m][n][0], acc[m][n][1]); \
        *(uint32_t *)(pp + 8 * CPR * 16) = pk2<ET>(acc[m][n][2], acc[m][n][3]); } }
#define WDBG(P) if (DBG && (P)) { _Pragma("unroll") for (int m = 0; m < MA; ++m) _Pragma("unroll") for (int n = 0; n < NA; ++n) { \
        if (brow * BM + wm * 64 + m * 16 + qr + 8 >= M) continue; \
        E *g = (P) + (size_t)(brow * BM + wm * 64 + m * 16 + qr) * N + bcol * BN + wn * WTN + n * 8 + qc; \
        *(uint32_t *)g = pk2<ET>(acc[m][n][0], acc[m][n][1]); *(uint32_t *)(g + 8 * (size_t)N) = pk2<ET>(acc[m][n][2], acc[m][n][3]); } }
        for (int tile = PERSIST ? (int)blockIdx.x : 0; tile < NTILE; tile += (PERSIST ? (int)gridDim.x : NTILE)) {
            const int bcol = PERSIST ? tile % NCOL : (int)blockIdx.x, brow = PERSIST ? tile / NCOL : (int)blockIdx.y;
            // A 64-row tail is legal (R = boards*64): out-of-range rows clamp on the read, and mask on the store.
            const int arow = brow * BM + grow < M ? brow * BM + grow : M - 1;
            const E *const ap0 = X + (size_t)arow * K + gq * 8;
            const size_t boff = (size_t)(bcol * BN + grow) * K + gq * 8;
            if constexpr (!PHASED) {                                       // PHASED: phase A already did q, k
                if constexpr (PERSIST) { if (tile != (int)blockIdx.x) BARS(5, 256); }     // q free
                WPASS(WQ) WDBG(DQ) WSTAGE(QOFF)
                BARA(2, 256);                                                             // q ready
                if constexpr (PERSIST) { if (tile != (int)blockIdx.x) BARS(6, 256); }     // k free
                WPASS(WK) WDBG(DK) WSTAGE(KOFF)
                BARA(3, 256);                                                             // k ready
                if constexpr (PERSIST) { if (tile != (int)blockIdx.x) BARS(7, 256); }     // v free
            }
            WPASS(WV) WDBG(DV) WSTAGE(VOFF)
            BARA(4, 256);                                                             // v ready
        }
#undef WISSUE
#undef WLDF
#undef WMMA
#undef WHEAD
#undef WSTEP
#undef WPASS
#undef WSTAGE
#undef WDBG
    } else {
    // ================= CONSUMERS: head (wid-4), BOTH boards, the two chains interleaved ============
        const int hh = wid - 4, hc = hh * 32;
        const int lr = lane & 15, lq = lane >> 4, qr = lane >> 2, qc = (lane & 3) * 2;
        const int br = (lane >> 4) * 8 + (lane & 7), bq = (lane >> 3) & 1;
        const int vk = ((lane >> 3) & 1) * 8 + (lane & 7);          // the key row ldmatrix.trans wants
        // Same affine-through-the-swizzle trick: one base per (tensor, k-atom); +board, +atom are constants.
        uint32_t qa[2], ka[2], va[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            qa[j] = sb + QOFF + lr * (CPR * 16) + (((hh * 4 + j * 2 + lq) ^ (lr & 7)) * 16);
            ka[j] = sb + KOFF + br * (CPR * 16) + (((hh * 4 + j * 2 + bq) ^ (br & 7)) * 16);
            va[j] = sb + VOFF + vk * (CPR * 16) + (((hh * 4 + j * 2 + (lane >> 4)) ^ (lane & 7)) * 16);
        }
        E *const cs = (E *)smem;                       // the o stage: the cp.async ring, dead once v has landed
        for (int tile = PERSIST ? (int)blockIdx.x : 0; tile < NTILE; tile += (PERSIST ? (int)gridDim.x : NTILE)) {
            const int bcol = PERSIST ? tile % NCOL : (int)blockIdx.x, brow = PERSIST ? tile / NCOL : (int)blockIdx.y;
            const int h = bcol * 4 + hh;
            if constexpr (!PHASED) { BARS(2, 256); BARS(3, 256); }             // q ready, k ready
            if constexpr (BMODE == 1) {                // PROBE: no band. Copy the handoff out so the producers'
                BARS(4, 256);                          // three passes and their SMEM stores stay live.
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int c = i * 32 + lane, r = c >> 2, q = c & 3, col = hc + q * 8;
                    const int src = (i & 1) ? ((i & 2) ? VOFF : KOFF) : QOFF;
                    *(uint4 *)(cs + (size_t)r * CS + col) =
                        *(const uint4 *)(smem + src + swzwL<LCPR>((uint32_t)r * CPR + (col >> 3)) * 16);
                }
                if constexpr (PERSIST) { BARA(5, 256); BARA(6, 256); BARA(7, 256); }
                continue;
            }
            constexpr int NB = SPLIT == 2 ? 1 : 2;     // boards whose SCORES are live at once (P always holds both)
            constexpr int MG = 4 / HALF, OPX = OP < MG ? OP : MG;    // query atoms per P group / per P.V group
#pragma unroll
            for (int hg = 0; hg < HALF; ++hg) {
            uint32_t pf[2][MG][4][4];                  // P for MG query atoms of BOTH boards: [b][atom][katom][4]
#pragma unroll
            for (int mi = 0; mi < MG; ++mi) {
              const int m = hg * MG + mi;
#pragma unroll
              for (int bo = 0; bo < 2 / NB; ++bo) {
                float S[NB][8][4];
#pragma unroll
                for (int b = 0; b < NB; ++b)
#pragma unroll
                    for (int p = 0; p < 8; ++p)
#pragma unroll
                        for (int i = 0; i < 4; ++i) S[b][p][i] = 0.f;
#pragma unroll
                for (int j = 0; j < 2; ++j) {          // two k-atoms of 16 dims = hd 32
                    uint32_t af[NB][4];
#pragma unroll
                    for (int b = 0; b < NB; ++b)
                        ldsm4(af[b][0], af[b][1], af[b][2], af[b][3], qa[j] + (bo * NB + b) * (64 * CPR * 16) + m * RSTR);
                    // k comes back TWO key-atom pairs at a time: 32 live B registers would cost the slack that
                    // holding P for both boards (128 registers) leaves, and S's reuse distance is 16 either way.
#pragma unroll
                    for (int gp = 0; gp < 2; ++gp) {
                        uint32_t kb[NB][2][4];
#pragma unroll
                        for (int b = 0; b < NB; ++b)
#pragma unroll
                            for (int np = 0; np < 2; ++np)
                                ldsm4(kb[b][np][0], kb[b][np][1], kb[b][np][2], kb[b][np][3],
                                      ka[j] + (bo * NB + b) * (64 * CPR * 16) + (gp * 2 + np) * RSTR);
#pragma unroll                                         // key-atom pp: even -> c0,c1 of pair pp/2, odd -> c2,c3;
                        for (int pp = 0; pp < 4; ++pp) // b inner = the two chains alternating at the mma
#pragma unroll
                            for (int b = 0; b < NB; ++b)
                                mma16816<ET>(S[b][gp * 4 + pp], &af[b][0], kb[b][pp >> 1][(pp & 1) * 2], kb[b][pp >> 1][(pp & 1) * 2 + 1]);
                    }
                }
                // ---- scale, + the geometry stencil (per HEAD, so the warp's two boards share one tile through L1)
#pragma unroll
                for (int p = 0; p < 8; ++p) {
                    const int r = m * 16 + qr, c = p * 8 + qc;
                    float b0, b1, b2, b3;
                    if constexpr (BBF) {
                        const E *bs = (const E *)BIAS + (size_t)h * 4096;
                        uint32_t u0 = *(const uint32_t *)(bs + r * 64 + c), u1 = *(const uint32_t *)(bs + (r + 8) * 64 + c);
                        b0 = un2<ET>(&u0, 0); b1 = un2<ET>(&u0, 1); b2 = un2<ET>(&u1, 0); b3 = un2<ET>(&u1, 1);
                    } else {
                        const float *bs = (const float *)BIAS + (size_t)h * 4096;
                        float2 f0 = *(const float2 *)(bs + r * 64 + c), f1 = *(const float2 *)(bs + (r + 8) * 64 + c);
                        b0 = f0.x; b1 = f0.y; b2 = f1.x; b3 = f1.y;
                    }
#pragma unroll
                    for (int b = 0; b < NB; ++b) {
                        S[b][p][0] = S[b][p][0] * scale + b0; S[b][p][1] = S[b][p][1] * scale + b1;
                        S[b][p][2] = S[b][p][2] * scale + b2; S[b][p][3] = S[b][p][3] * scale + b3;
                    }
                }
                // ---- softmax in registers, quad shuffles; P -> A fragments by the FA2 identity
#pragma unroll
                for (int b = 0; b < NB; ++b) {
                    float mx0 = 0.f, mx1 = 0.f;
                    if constexpr (BMODE != 3) {
#pragma unroll
                        for (int p = 0; p < 8; ++p) {
                            mx0 = fmaxf(mx0, fmaxf(S[b][p][0], S[b][p][1]));
                            mx1 = fmaxf(mx1, fmaxf(S[b][p][2], S[b][p][3]));
                        }
                        QRED_MAX(mx0) QRED_MAX(mx1)
                    }
                    float s0 = 0.f, s1 = 0.f;
#pragma unroll
                    for (int p = 0; p < 8; ++p) {
                        S[b][p][0] = ex2((S[b][p][0] - mx0) * 1.4426950408889634f);
                        S[b][p][1] = ex2((S[b][p][1] - mx0) * 1.4426950408889634f);
                        S[b][p][2] = ex2((S[b][p][2] - mx1) * 1.4426950408889634f);
                        S[b][p][3] = ex2((S[b][p][3] - mx1) * 1.4426950408889634f);
                        s0 += S[b][p][0] + S[b][p][1]; s1 += S[b][p][2] + S[b][p][3];
                    }
                    QRED_SUM(s0) QRED_SUM(s1)
                    s0 = 1.f / s0; s1 = 1.f / s1;
#pragma unroll
                    for (int kj = 0; kj < 4; ++kj) {
                        pf[bo * NB + b][mi][kj][0] = pk2<ET>(S[b][2 * kj][0] * s0, S[b][2 * kj][1] * s0);
                        pf[bo * NB + b][mi][kj][1] = pk2<ET>(S[b][2 * kj][2] * s1, S[b][2 * kj][3] * s1);
                        pf[bo * NB + b][mi][kj][2] = pk2<ET>(S[b][2 * kj + 1][0] * s0, S[b][2 * kj + 1][1] * s0);
                        pf[bo * NB + b][mi][kj][3] = pk2<ET>(S[b][2 * kj + 1][2] * s1, S[b][2 * kj + 1][3] * s1);
                    }
                }
              }
              // ptxas interleaves the fully unrolled query-atom loop and the live set overruns the file; this
              // stops it moving loads across the boundary (SS8a: instructions and registers are one budget).
              if constexpr (SPLIT == 1) { asm volatile("" ::: "memory"); }
            }
            if constexpr (PERSIST) { if (hg == HALF - 1) { BARA(5, 256); BARA(6, 256); } }   // q and k released
            if (hg == 0) BARS(4, 256);                                         // v ready (once)
            // ---- O = P.V, v as the B operand through ldmatrix.trans, OPX query atoms at a time
#pragma unroll
            for (int mg = 0; mg < MG / OPX; ++mg) {
                float Ov[2][OPX][4][4];
#pragma unroll
                for (int b = 0; b < 2; ++b)
#pragma unroll
                    for (int mm = 0; mm < OPX; ++mm)
#pragma unroll
                        for (int n = 0; n < 4; ++n)
#pragma unroll
                            for (int i = 0; i < 4; ++i) Ov[b][mm][n][i] = 0.f;
#pragma unroll
                for (int kj = 0; kj < 4; ++kj) {       // k-atom = 16 keys
                    uint32_t vb[2][2][4];
#pragma unroll
                    for (int b = 0; b < 2; ++b)
#pragma unroll
                        for (int g = 0; g < 2; ++g)
                            ldsm4t(vb[b][g][0], vb[b][g][1], vb[b][g][2], vb[b][g][3], va[g] + b * (64 * CPR * 16) + kj * RSTR);
#pragma unroll
                    for (int mm = 0; mm < OPX; ++mm)
#pragma unroll
                        for (int n = 0; n < 4; ++n)
#pragma unroll
                            for (int b = 0; b < 2; ++b)
                                mma16816<ET>(Ov[b][mm][n], &pf[b][mg * OPX + mm][kj][0], vb[b][n >> 1][(n & 1) * 2], vb[b][n >> 1][(n & 1) * 2 + 1]);
                }
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    if (OSTG == 0 && brow * BM + b * 64 >= M) continue;
#pragma unroll
                    for (int mm = 0; mm < OPX; ++mm)
#pragma unroll
                        for (int n = 0; n < 4; ++n) {
                            const int r = b * 64 + (hg * MG + mg * OPX + mm) * 16 + qr, c = hc + n * 8 + qc;
                            if constexpr (OSTG) {      // block-wide 16-B stores out of the stage
                                E *p = cs + (size_t)r * CS + c;
                                *(uint32_t *)p = pk2<ET>(Ov[b][mm][n][0], Ov[b][mm][n][1]);
                                *(uint32_t *)(p + 8 * CS) = pk2<ET>(Ov[b][mm][n][2], Ov[b][mm][n][3]);
                            } else {                   // PERSIST: the ring is the next tile's, so store direct
                                E *g = O + (size_t)(brow * BM + r) * N + bcol * BN + c;
                                uint32_t u0 = pk2<ET>(Ov[b][mm][n][0], Ov[b][mm][n][1]);
                                uint32_t u1 = pk2<ET>(Ov[b][mm][n][2], Ov[b][mm][n][3]);
                                asm volatile("st.global.cs.b32 [%0], %1;\n" :: "l"(g), "r"(u0) : "memory");
                                asm volatile("st.global.cs.b32 [%0], %1;\n" :: "l"(g + 8 * (size_t)N), "r"(u1) : "memory");
                            }
                        }
                }
            }
            }
            if constexpr (PERSIST) { BARA(7, 256); }                           // v released
        }
    }
    // ---- the o epilogue: 16-B coalesced stores out of the stage, all 256 threads (the ring is dead)
    if constexpr (OSTG && !PERSIST) {
        __syncthreads();
        E *const cs = (E *)smem;
#pragma unroll
        for (int i = 0; i < BM * BN / 8 / 256; ++i) {
            const int c = i * 256 + (int)threadIdx.x, r = c / (BN / 8), q = c % (BN / 8);
            if ((int)blockIdx.y * BM + r >= M) continue;
            E *g = O + (size_t)((int)blockIdx.y * BM + r) * N + (int)blockIdx.x * BN + q * 8;
            const uint4 v = *(const uint4 *)(cs + (size_t)r * CS + q * 8);
            asm volatile("st.global.cs.v4.b32 [%0], {%1,%2,%3,%4};\n" :: "l"(g), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
        }
    }
}


#undef EXP2
#undef BARA
#undef BARS

// v52 = go_ws<PIPE 4, BBF 1, DBG 0, ET, BMODE 0, SPLIT 0, OP 2, HALF 2>, i.e. the kernel's
// <4,1,0,ET,0,0,2,2, FBUF 2, OSTG 1, PERSIST 0, PHASED 0>. 160 KB, 256 threads, one block per SM.
template <int ET>
void launch_ws(const void *x, const void *wq, const void *wk, const void *wv, const void *bias, void *o,
               int rows, int d, int bank, int heads, cudaStream_t s) {
    using E = typename et_t<ET>::T;
    if (!hero_qkv_band_supported(rows, d, bank, heads)) {
        fprintf(stderr, "hero_qkv_band_ws: unsupported shape rows=%d d=%d bank=%d heads=%d\n", rows, d, bank, heads);
        return;
    }
    const uintptr_t mask = (uintptr_t)x | (uintptr_t)wq | (uintptr_t)wk | (uintptr_t)wv | (uintptr_t)bias | (uintptr_t)o;
    if (mask & 15u) { fprintf(stderr, "hero_qkv_band_ws: pointers must be 16-byte aligned\n"); return; }
    constexpr int WBM = 128, WBN = 128, WPIPE = 4;
    constexpr int WS_SMEM = WPIPE * (WBM + WBN) * 32 * 2 + 3 * WBM * WBN * 2;   // ring + the q|k|v handoff = 160 KB
    static_assert(WS_SMEM <= 166912, "A100 allows 163 KB of dynamic SMEM per block");
    static_assert(WPIPE * (WBM + WBN) * 32 * 2 >= WBM * (WBN + 8) * 2, "the o stage must fit the dead ring");
    auto kern = qkvband_ws<WPIPE, 1, 0, ET, 0, 0, 2, 2, 2, 1, 0, 0>;
    int dev = 0;
    cudaGetDevice(&dev);
    static unsigned done = 0;                                // ONE bit per device, as the fused arm does
    if (dev >= 32 || !((done >> dev) & 1u)) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, WS_SMEM);
        cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (dev < 32) done |= 1u << dev;
    }
    const float scale = (float)(1.0 / sqrt((double)(bank / heads)));
    kern<<<dim3(bank / WBN, (rows + WBM - 1) / WBM), 256, WS_SMEM, s>>>(
        (const E *)x, (const E *)wq, (const E *)wk, (const E *)wv, bias, (E *)o,
        (E *)nullptr, (E *)nullptr, (E *)nullptr, rows, bank, d, scale);
}
// ===================== end lc0bench 0011 =====================

constexpr int BM = 128, BN = 128, BK = 32, NW = 8, NT = NW * 32, PIPE = 4;
constexpr int CPR = BN / 8;                                  // 16-B chunks per tile row
constexpr int WNW = NW / 2, WTN = BN / WNW;                  // warp grid 2 x 4; warp tile 64 rows x 32 cols
constexpr int MA = 4, NA = WTN / 8;                          // accumulator atoms: 4 m-atoms x 4 n-atoms
constexpr int ACH = BM * (BK / 8) / NT, BCH = BN * (BK / 8) / NT, RPI = NT / (BK / 8);
constexpr int ABYTES = BM * BK * 2, STAGE = (BM + BN) * BK * 2, SPIPE = PIPE * STAGE, TILE = BM * BN * 2;
constexpr int VOFF = SPIPE, BOFF = VOFF + TILE, BSTR = 72;   // v tile, then 4 bias heads x 64 rows of stride BSTR
constexpr int CS = BN + 8;                                   // o staging row stride (conflict-free both ways)
constexpr int SMEM_BYTES = BOFF + 4 * 64 * BSTR * 2;         // 132 KB

template <int ET>
__global__ __launch_bounds__(NT, 1)
void hero_band_kernel(const typename et_t<ET>::T *__restrict__ X, const typename et_t<ET>::T *__restrict__ WQ,
                      const typename et_t<ET>::T *__restrict__ WK, const typename et_t<ET>::T *__restrict__ WV,
                      const typename et_t<ET>::T *__restrict__ BIAS, typename et_t<ET>::T *__restrict__ O,
                      int M, int N, int K, float scale) {
    using E = typename et_t<ET>::T;
    extern __shared__ char smem[];
    const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, wm = wid / WNW, wn = wid % WNW;
    const int lr = lane & 15, lq = lane >> 4, qr = lane >> 2, qc = (lane & 3) * 2;
    const int grow = tid >> 2, gq = tid & 3;                 // 4 lanes cover one row's 64-B k-window
    const int bcol = blockIdx.x, brow = blockIdx.y;
    const E *ap, *bp;

    // ---- SMEM destinations and ldmatrix sources: computed ONCE
    uint32_t adst[ACH], bdst[BCH], aldm[4], bldm[NA / 2];
#pragma unroll
    for (int i = 0; i < ACH; ++i) { uint32_t c = (grow + i * RPI) * 4 + gq; adst[i] = sb + swz(c) * 16; }
#pragma unroll
    for (int i = 0; i < BCH; ++i) { uint32_t c = (grow + i * RPI) * 4 + gq; bdst[i] = sb + ABYTES + swz(c) * 16; }
#pragma unroll
    for (int m = 0; m < 4; ++m) { uint32_t c = (wm * 64 + m * 16 + lr) * 4 + lq; aldm[m] = sb + swz(c) * 16; }
    const int br = (lane >> 4) * 8 + (lane & 7), bq = (lane >> 3) & 1;      // paired-B lane map
#pragma unroll
    for (int n = 0; n < NA / 2; ++n) { uint32_t c = (wn * WTN + n * 16 + br) * 4 + bq; bldm[n] = sb + ABYTES + swz(c) * 16; }

    float acc[MA][NA][4];
    uint32_t ra[2][4][4], rb[2][NA / 2][4], qp[MA][NA][2], kp[MA][NA][2];
    const int NKT = K / BK;
    uint32_t woff, roff;
    int ktile;

    // This block's 4 bias heads -> SMEM, issued BEFORE the first projection so the band reads them at SMEM latency
    // instead of taking 64 dependent L2 loads per warp-head at the end. ONE extra cp.async group at the HEAD of the
    // queue: it is the oldest, so every later `wait_group PIPE-2` has already retired it.
    {
        const E *bg = BIAS + (size_t)bcol * 4 * 4096;
#pragma unroll
        for (int i = 0; i < 4 * 64 * 8 / NT; ++i) {
            const int cc = i * NT + tid, rr = cc >> 3, qq = cc & 7;
            CP16(sb + BOFF + rr * (BSTR * 2) + qq * 16, bg + (size_t)rr * 64 + qq * 8);
        }
        CPCOMMIT();
    }
    const int arow = brow * BM + grow < M ? brow * BM + grow : M - 1;       // the 64-row tail clamps on the read
    const E *const ap0 = X + (size_t)arow * K + gq * 8;
    const size_t boff = (size_t)(bcol * BN + grow) * K + gq * 8;

#define ISSUE(OFF) { _Pragma("unroll") for (int i = 0; i < ACH; ++i) CP16(adst[i] + (OFF), ap + (size_t)i * RPI * K); \
                     _Pragma("unroll") for (int i = 0; i < BCH; ++i) CP16(bdst[i] + (OFF), bp + (size_t)i * RPI * K); }
#define LDF(BUF, KB) { _Pragma("unroll") for (int m = 0; m < 4; ++m) ldsm4(ra[BUF][m][0], ra[BUF][m][1], ra[BUF][m][2], ra[BUF][m][3], (aldm[m] + roff) ^ (KB * 32)); \
                       _Pragma("unroll") for (int n = 0; n < NA / 2; ++n) ldsm4(rb[BUF][n][0], rb[BUF][n][1], rb[BUF][n][2], rb[BUF][n][3], (bldm[n] + roff) ^ (KB * 32)); }
#define MMA(BUF) { _Pragma("unroll") for (int m = 0; m < MA; ++m) _Pragma("unroll") for (int n = 0; n < NA; ++n) \
                       mma16816<ET>(acc[m][n], &ra[BUF][m][0], rb[BUF][n >> 1][(n & 1) * 2], rb[BUF][n >> 1][(n & 1) * 2 + 1]); }
// One projection pass over the same x rows. The x tile is L2-resident after the first pass; the weights stream once each.
#define PASS(W) { \
    ap = ap0; bp = (W) + boff; woff = 0; roff = 0; ktile = PIPE - 1; \
    _Pragma("unroll") for (int m = 0; m < MA; ++m) _Pragma("unroll") for (int n = 0; n < NA; ++n) \
        _Pragma("unroll") for (int i = 0; i < 4; ++i) acc[m][n][i] = 0.f; \
    _Pragma("unroll") for (int s = 0; s < PIPE - 1; ++s) { ISSUE(woff); CPCOMMIT(); woff += STAGE; ap += BK; bp += BK; } \
    cpwait<PIPE - 2>(); __syncthreads(); LDF(0, 0) \
    _Pragma("unroll 1") for (int t = 0; t < NKT; ++t) { \
        LDF(1, 1) \
        if (ktile < NKT) { ISSUE(woff); ap += BK; bp += BK; } \
        CPCOMMIT();                       /* ALWAYS: wait_group counts GROUPS, empty ones included */ \
        ++ktile; \
        woff = (woff + STAGE == SPIPE) ? 0 : woff + STAGE; \
        MMA(0) \
        roff = (roff + STAGE == SPIPE) ? 0 : roff + STAGE; \
        cpwait<PIPE - 2>(); __syncthreads(); \
        LDF(0, 0) \
        MMA(1) \
    } \
    cpwait<0>(); __syncthreads(); }

    // ---- pass 1: q, packed to 16-bit A-fragments and kept in registers
    PASS(WQ)
#pragma unroll
    for (int m = 0; m < MA; ++m)
#pragma unroll
        for (int n = 0; n < NA; ++n) { qp[m][n][0] = pk2<ET>(acc[m][n][0], acc[m][n][1]); qp[m][n][1] = pk2<ET>(acc[m][n][2], acc[m][n][3]); }
    // ---- pass 2: v -> SMEM (the only tile that must round-trip: O = P.V needs V^T)
    PASS(WV)
#pragma unroll
    for (int m = 0; m < MA; ++m)
#pragma unroll
        for (int n = 0; n < NA; ++n) {
            uint32_t r0 = wm * 64 + m * 16 + qr, c0 = wn * WTN + n * 8 + qc;
            *(uint32_t *)(smem + VOFF + swzw(r0 * CPR + (c0 >> 3)) * 16 + (c0 & 7) * 2) = pk2<ET>(acc[m][n][0], acc[m][n][1]);
            *(uint32_t *)(smem + VOFF + swzw((r0 + 8) * CPR + (c0 >> 3)) * 16 + (c0 & 7) * 2) = pk2<ET>(acc[m][n][2], acc[m][n][3]);
        }
    __syncwarp();                                            // a warp reads back only what it wrote, but not in lockstep
    // ---- pass 3: k, consumed immediately as B fragments straight out of the accumulator
    PASS(WK)
#pragma unroll
    for (int m = 0; m < MA; ++m)
#pragma unroll
        for (int n = 0; n < NA; ++n) { kp[m][n][0] = pk2<ET>(acc[m][n][0], acc[m][n][1]); kp[m][n][1] = pk2<ET>(acc[m][n][2], acc[m][n][3]); }

    // ---- the band: this warp's own board (64 queries) x the same board's 64 keys, for ONE head, all on chip
    E *cs = (E *)smem;                                       // o staging reuses the (now dead) pipeline region
    const int hc = wn * WTN;                                 // this head's column base inside the tile
    float S[MA][8][4];
#pragma unroll
    for (int m = 0; m < MA; ++m)
#pragma unroll
        for (int p = 0; p < 8; ++p)
#pragma unroll
            for (int i = 0; i < 4; ++i) S[m][p][i] = 0.f;
#pragma unroll
    for (int j = 0; j < 2; ++j) {                            // two k-atoms of 16 dims
#pragma unroll
        for (int m = 0; m < MA; ++m) {
            uint32_t af[4] = {qp[m][2 * j][0], qp[m][2 * j][1], qp[m][2 * j + 1][0], qp[m][2 * j + 1][1]};
#pragma unroll
            for (int p = 0; p < 8; ++p)                      // key-atom p: even -> c0,c1 of k-atom p/2, odd -> c2,c3
                mma16816<ET>(S[m][p], af, kp[p >> 1][2 * j][p & 1], kp[p >> 1][2 * j + 1][p & 1]);
        }
    }
    // ---- scale, + the geometry stencil / free table (in that order), then softmax by quad shuffles (FA2's pattern)
    const E *bs = (const E *)(smem + BOFF) + (size_t)wn * 64 * BSTR;
#pragma unroll
    for (int m = 0; m < MA; ++m)
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int r = m * 16 + qr, c = p * 8 + qc;
            uint32_t u0 = *(const uint32_t *)(bs + r * BSTR + c), u1 = *(const uint32_t *)(bs + (r + 8) * BSTR + c);
            // __fmaf_rn, not `* scale + b`: whether nvcc contracts that into an FFMA depends on the surrounding
            // pressure, and this file must stay BITWISE equal to ops/mfu60/qkv_band.py's variant 10.
            S[m][p][0] = __fmaf_rn(S[m][p][0], scale, un2<ET>(&u0, 0));
            S[m][p][1] = __fmaf_rn(S[m][p][1], scale, un2<ET>(&u0, 1));
            S[m][p][2] = __fmaf_rn(S[m][p][2], scale, un2<ET>(&u1, 0));
            S[m][p][3] = __fmaf_rn(S[m][p][3], scale, un2<ET>(&u1, 1));
        }
    uint32_t pf[MA][4][4];                                   // P as 16-bit A-fragments; built per m-atom so S dies early
#pragma unroll
    for (int m = 0; m < MA; ++m) {
        float mx0 = -3.4e38f, mx1 = -3.4e38f;
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            mx0 = fmaxf(mx0, fmaxf(S[m][p][0], S[m][p][1]));
            mx1 = fmaxf(mx1, fmaxf(S[m][p][2], S[m][p][3]));
        }
        QRED_MAX(mx0) QRED_MAX(mx1)                          // a row lives in 4 lanes of the quad
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            S[m][p][0] = ex2((S[m][p][0] - mx0) * 1.4426950408889634f);
            S[m][p][1] = ex2((S[m][p][1] - mx0) * 1.4426950408889634f);
            S[m][p][2] = ex2((S[m][p][2] - mx1) * 1.4426950408889634f);
            S[m][p][3] = ex2((S[m][p][3] - mx1) * 1.4426950408889634f);
            s0 += S[m][p][0] + S[m][p][1]; s1 += S[m][p][2] + S[m][p][3];
        }
        QRED_SUM(s0) QRED_SUM(s1)
        s0 = 1.f / s0; s1 = 1.f / s1;
#pragma unroll
        for (int kj = 0; kj < 4; ++kj) {                     // the FA2 identity again: C fragment -> A fragment, k = 16
            pf[m][kj][0] = pk2<ET>(S[m][2 * kj][0] * s0, S[m][2 * kj][1] * s0);
            pf[m][kj][1] = pk2<ET>(S[m][2 * kj][2] * s1, S[m][2 * kj][3] * s1);
            pf[m][kj][2] = pk2<ET>(S[m][2 * kj + 1][0] * s0, S[m][2 * kj + 1][1] * s0);
            pf[m][kj][3] = pk2<ET>(S[m][2 * kj + 1][2] * s1, S[m][2 * kj + 1][3] * s1);
        }
    }
    // ---- O = P.V, v as the B operand through ldmatrix.trans
    float Ov[MA][4][4];
#pragma unroll
    for (int m = 0; m < MA; ++m)
#pragma unroll
        for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) Ov[m][n][i] = 0.f;
#pragma unroll
    for (int kj = 0; kj < 4; ++kj) {                         // k-atom = 16 keys
        uint32_t vb[2][4];
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const uint32_t key = kj * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), dim = hc + g * 16 + (lane >> 4) * 8;
            ldsm4t(vb[g][0], vb[g][1], vb[g][2], vb[g][3], sb + VOFF + swzw((wm * 64 + key) * CPR + (dim >> 3)) * 16);
        }
#pragma unroll
        for (int m = 0; m < MA; ++m)
#pragma unroll
            for (int n = 0; n < 4; ++n) mma16816<ET>(Ov[m][n], &pf[m][kj][0], vb[n >> 1][(n & 1) * 2], vb[n >> 1][(n & 1) * 2 + 1]);
    }
    // ---- stage o block-wide, then 16-B evict-first stores (the epilogue is 23 points of this problem)
#pragma unroll
    for (int m = 0; m < MA; ++m)
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            E *p = cs + (size_t)(wm * 64 + m * 16 + qr) * CS + hc + n * 8 + qc;
            *(uint32_t *)p = pk2<ET>(Ov[m][n][0], Ov[m][n][1]);
            *(uint32_t *)(p + 8 * CS) = pk2<ET>(Ov[m][n][2], Ov[m][n][3]);
        }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < BM * BN / 8 / NT; ++i) {
        const int c = i * NT + tid, r = c / (BN / 8), q = c % (BN / 8);
        if (brow * BM + r >= M) continue;                    // the 64-row tail is masked on the store
        E *g = O + (size_t)(brow * BM + r) * N + bcol * BN + q * 8;
        const uint4 v = *(const uint4 *)(cs + (size_t)r * CS + q * 8);
        asm volatile("st.global.cs.v4.b32 [%0], {%1,%2,%3,%4};\n" ::"l"(g), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
    }
#undef ISSUE
#undef LDF
#undef MMA
#undef PASS
}

template <int ET>
void launch(const void *x, const void *wq, const void *wk, const void *wv, const void *bias, void *o,
            int rows, int d, int bank, int heads, cudaStream_t s) {
    using E = typename et_t<ET>::T;
    if (!hero_qkv_band_supported(rows, d, bank, heads)) {
        fprintf(stderr, "hero_qkv_band: unsupported shape rows=%d d=%d bank=%d heads=%d\n", rows, d, bank, heads);
        return;
    }
    const uintptr_t mask = (uintptr_t)x | (uintptr_t)wq | (uintptr_t)wk | (uintptr_t)wv | (uintptr_t)bias | (uintptr_t)o;
    if (mask & 15u) {                                        // cp.async and the epilogue both move 16 B at a time
        fprintf(stderr, "hero_qkv_band: pointers must be 16-byte aligned\n");
        return;
    }
    auto kern = hero_band_kernel<ET>;
    int dev = 0;
    cudaGetDevice(&dev);
    static unsigned done = 0;                                // ONE bit per device: the >48 KB opt-in is per device
    if (dev >= 32 || !((done >> dev) & 1u)) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (dev < 32) done |= 1u << dev;
    }
    const float scale = (float)(1.0 / sqrt((double)(bank / heads)));    // applied BEFORE the bias
    kern<<<dim3(bank / BN, (rows + BM - 1) / BM), NT, SMEM_BYTES, s>>>(
        (const E *)x, (const E *)wq, (const E *)wk, (const E *)wv, (const E *)bias, (E *)o, rows, bank, d, scale);
}

}  // namespace

bool hero_qkv_band_supported(int rows, int d, int bank, int heads) {
    return rows > 0 && rows % 64 == 0 && d % BK == 0 && bank % BN == 0 && heads > 0 && bank == heads * 32;
}

void hero_qkv_band(const __half *x, const __half *wq, const __half *wk, const __half *wv, const __half *bias,
                   __half *o, int rows, int d, int bank, int heads, cudaStream_t s) {
    launch<1>(x, wq, wk, wv, bias, o, rows, d, bank, heads, s);
}

void hero_qkv_band_bf16(const __nv_bfloat16 *x, const __nv_bfloat16 *wq, const __nv_bfloat16 *wk,
                        const __nv_bfloat16 *wv, const __nv_bfloat16 *bias, __nv_bfloat16 *o,
                        int rows, int d, int bank, int heads, cudaStream_t s) {
    launch<0>(x, wq, wk, wv, bias, o, rows, d, bank, heads, s);
}

// lc0bench 0011: the warp-specialized arm. Same shapes, same conventions, same output.
void hero_qkv_band_ws(const __half *x, const __half *wq, const __half *wk, const __half *wv, const __half *bias,
                      __half *o, int rows, int d, int bank, int heads, cudaStream_t s) {
    launch_ws<1>(x, wq, wk, wv, bias, o, rows, d, bank, heads, s);
}
void hero_qkv_band_ws_bf16(const __nv_bfloat16 *x, const __nv_bfloat16 *wq, const __nv_bfloat16 *wk,
                           const __nv_bfloat16 *wv, const __nv_bfloat16 *bias, __nv_bfloat16 *o,
                           int rows, int d, int bank, int heads, cudaStream_t s) {
    launch_ws<0>(x, wq, wk, wv, bias, o, rows, d, bank, heads, s);
}
