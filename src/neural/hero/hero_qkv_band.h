// The fused q|k|v projection + attention band for heroD4 / hero3 on sm_80 (A100), as one kernel.
// Drop-in for lc0's hero backend: it replaces the three q|k|v GEMMs AND fusedMHA with a single launch, and q, k, v
// never reach HBM. Measured on A100 (docs/audit/herod4_a100.md SS11b/SS11c): 1.06x of "3 cuBLAS GEMMs + band" per
// layer at heroD4's shape, with the projections themselves at 82-85% of peak, and q|k|v BITWISE equal to cuBLAS.
//
// CONVENTIONS (all row-major, all device pointers, all 16-byte aligned):
//   x    [rows, d]        half   the layer input, board-ordered (row = board*64 + square)
//   wq/wk/wv [bank, d]    half   the engine's own t.qw / t.kw / t.vw (torch [out, in]); no transpose needed
//   bias [heads, 64, 64]  half   the geometry stencil + free table; scale 1/sqrt(hd) is applied BEFORE adding it
//   o    [rows, bank]     half   head-major columns -- exactly what the out projection consumes
//   rows must be a multiple of 64 (a 64-row tail, i.e. an odd minibatch, is legal and exact)
//   d    % 32 == 0,  bank % 128 == 0,  bank == heads * 32   (hd = 32; heroD4 640/4096/128, hero3 1024/1024/32)
//
// The kernel needs 132 KB of dynamic shared memory, so it opts in with cudaFuncSetAttribute. That is done lazily
// inside, once per device (the only file-static state); call it under whatever mutex the backend already holds.
// Everything else is stateless: no workspace, no scratch buffer, any stream, any device.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// True if this shape is supported (see the constraints above). Check once at network load.
bool hero_qkv_band_supported(int rows, int d, int bank, int heads);

// fp16 build: what the engine links against. A no-op (with one line on stderr) if the shape is unsupported.
void hero_qkv_band(const __half *x, const __half *wq, const __half *wk, const __half *wv, const __half *bias,
                   __half *o, int rows, int d, int bank, int heads, cudaStream_t s);

// lc0bench 0011: the WARP-SPECIALIZED entry (qkv_band.py v52). Identical contract and identical output; warps 0-3
// project q|k|v and hand them to warps 4-7 through 96 KB of SMEM, which fills the band's dependent stalls.
// 160 KB of dynamic SMEM (same lazy per-device opt-in). MEASURED 1.03x of the fused entry above, per layer.
void hero_qkv_band_ws(const __half *x, const __half *wq, const __half *wk, const __half *wv, const __half *bias,
                      __half *o, int rows, int d, int bank, int heads, cudaStream_t s);
void hero_qkv_band_ws_bf16(const __nv_bfloat16 *x, const __nv_bfloat16 *wq, const __nv_bfloat16 *wk,
                           const __nv_bfloat16 *wv, const __nv_bfloat16 *bias, __nv_bfloat16 *o,
                           int rows, int d, int bank, int heads, cudaStream_t s);

// bf16 twin, same source, used by the torch-side gate (ops/mfu60/qkv_band.py) to prove this file bitwise-equal
// to the in-tree kernel it was distilled from.
void hero_qkv_band_bf16(const __nv_bfloat16 *x, const __nv_bfloat16 *wq, const __nv_bfloat16 *wk,
                        const __nv_bfloat16 *wv, const __nv_bfloat16 *bias, __nv_bfloat16 *o,
                        int rows, int d, int bank, int heads, cudaStream_t s);
