# Hero backend (`-hero`) — build notes

This branch (`hero-backend`) adds a CUDA backend for the **Hero** chess net — a
BT4-skeleton transformer whose per-layer dense FFN is replaced by a bank of E
experts (E=13 or 28), each of the 64 squares routed to one expert by a
deterministic board fact. Hero can't ride the `.pb` path (no expert axis), so it
loads a custom `.htw` side-file. Full analysis + the prize (Hero torch ~5k pos/s
vs ~13.8k BT4 vs ~23k roofline on A100): `confluence-labs/hero-inference`
`docs/PLAN.md`. The `.htw` exporter + the torch oracle live there too.

All file:line refs are this tree (upstream clone, `src/neural/`).

## The 6 changes

1. **FFN swap** — `backends/cuda/layers.cc:2053-2073`. The dense FFN is two
   `cublasXgemm` (up w/ fused mish `addBiasBatched` :2061; down :2065). Replace
   both with the routed grouped GEMM. Keep LN1 (:2047), the fused mish, LN2
   (:2075, residual=LN1 out via `skip`, DeepNorm `alpha_`).
2. **FFN kernel** — CUTLASS **v4.4.1** (vendored, `subprojects/cutlass.wrap`)
   `GemmUniversalMode::kGrouped`. Add `groupedFFN<T>` in
   `backends/cuda/cutlass_kernels.cu` next to `fusedMHA` (same half_t/bfloat16_t
   dispatch, same build gate `cutlass && max_cuda>=800`). Host-side: stable-sort
   rows by expert -> counts/offsets -> device ptr arrays -> 2 grouped launches +
   fused mish -> scatter. cuBLAS per-group loop = the correctness reference +
   non-Hopper fallback.
3. **Smolgen -> static bias** — delete `layers.cc:1852-1915`; preload Hero's
   `[heads,64,64]` bias (= `free + einsum(alpha, geo_basis())`) and pass it as
   the `input2` of the softmax (:2010). Per-head, so tweak the softmax index in
   `common_kernels.cu:787` to drop the N stride (or broadcast-materialize).
4. **LayerNorm** — `common_kernels.cu:1106`. Pass eps **1e-3**; supply a zero
   buffer for `bias`/`betas` (Hero beta=0, no FFN bias; only `skip` is
   null-guarded natively).
5. **Weights** — `.htw` loader. Magic-sniff branch in `loader.cc:214`
   (`LoadWeights`) -> `LoadHeroWeights` -> a `HeroWeights` struct (mirrors
   `MultiHeadWeights` but per-layer carries `ffn_up[E,dff,d]`, `ffn_down[E,d,dff]`,
   `alpha[heads,18]`, `free[heads,64,64]`, gamma-only LNs); factory branch
   `factory.cc:105`. Register `-hero`/`-hero-fp16` in `network_cuda.cc:1466`
   (half already instantiated). FFN weight layout is `[out,in]` row-major =
   torch = lc0's `CUBLAS_OP_T` — no transpose.
6. **Runtime recompute (CUDA, not weights):** `geo_basis()` (18 masks),
   `hero.routes` (13c argmax of planes[:12]; 28c also `hero.attackers` sliders),
   `policy_gather[1858]`.

## `.htw` format (little-endian; written by hero-inference/export/export_hero_htw.py)

```
magic u32 'HERW'(0x48455257), version u32, dtype u32 (1=fp16)
config: d,layers,heads,hd,dff,embed_dff,pol_d,classes  (8x u32)
n_tensors u32
directory[n]: name char[48], ndim u32, shape u32[6], offset u64, nbytes u64
blobs: raw LE fp16, C-contiguous, page(4096)-aligned, directory order
       (offsets relative to blob_base = first page-aligned pos after directory)
```
Tensor order = stem (preproc.0/1, embed, embed_ln, embed_up/down, embed_ffn_ln),
then per layer i: attn.q/k/v/out, attn.alpha, attn.free, ln1, ffn.up, ffn.down,
ln2; then policy.embed(+bias)/q(+bias)/k(+bias)/ppo, value.embed/q/k/v.
Reference `.htw` + oracle (planes/policy/wdl npy) in GCS `hero/`.

## Build sequence (each gates against the oracle)

- **S1 DONE** — fork builds clean; BT4 `.pb` still runs (no regression).
- **S3a** — `.htw` loader + `HeroWeights` + `-hero` registration; parse + upload,
  run stem + attention (static bias) + heads with a **cuBLAS-loop FFN** (slow,
  correct). Gate: policy/wdl within 1e-2 of the oracle (bf16, L=15).
- **S3b** — routes()/geo_basis()/attackers() as CUDA (or host precompute the
  route from planes; it's batch-static per square index? NO — route depends on
  the board, i.e. per (N, square). Compute per position.).
- **S4** — swap in CUTLASS `groupedFFN`; gate bit-close vs the loop; measure nps.
- **S5** — fp8 (arena-gated).

## Build

`meson setup build --buildtype=release -Dcc_cuda=<80|90> -Dcudnn=false`; the
CUTLASS grouped path needs `max_cuda>=800` (Ampere) — the fast kGrouped kernels
are SM90 (Hopper). Build+gate on a GCP A100 (hero-inference launch pattern).
