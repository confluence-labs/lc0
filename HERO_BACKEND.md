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
- **INC1 DONE** — `.htw` loader + `HeroWeights` (`src/neural/hero/hero_weights.{h,cc}`):
  loads the real 1.08 GB net, fp16->fp32 bit-exact vs a Python decode.
- **INC2 DONE** — `-hero` registration + `.htw` routing
  (`src/neural/hero/network_hero.cc`; sniff in `loader.cc:LoadWeights`; path
  injected in `wrapper.cc`). `--backend=hero --weights=x.htw` loads the net
  through the real lc0 binary ("Hero net loaded: d=1024 ..."), forward stubs.
### INC3 blueprint (fork lc0's CudaNetwork — near-identical, 3 deltas)

`CudaNetwork` = a `vector<BaseLayer>` each with
`Eval(N, out, in, in2, scratch, ..., cublas, stream, ...)`; forward rotates
`tensor_mem[0..2]` + `scratch_mem`. Fork it as `HeroCudaNetwork` with a shorter
list (no conv/residual, no MLH). Feed layers from `HeroWeights` via the same
`allocAndUpload` (fp32->fp16 on GPU).

- **Stem** = lc0's `is_pe_dense_embedding_` path, `layers.cc:2349-2470`,
  step-for-step Hero's: preproc (`inputPreprocessForAttentionBody`) -> embed
  `cublasXgemm`+mish+`LayerNorm(1e-3)` -> **drop the input-gating `2438`** ->
  embed-FFN d1/mish/d2 + DeepNorm `LayerNorm(alpha=(2L)^-0.25)`. Delta: LN
  **beta=0** (zero buffer). REUSE with Hero weight names.
- **expandPlanes** `network_cuda.cc:845` -> `(N,112,8,8)` — REUSE as-is; compute
  `route[N,64]` (int, device) right after (argmax 13c / +attackers 28c).
- **EncoderBlock** `layers.cc:1842`: attention QKV/AV/out + LN1/LN2 REUSE;
  smolgen(`1852-1915`)->static `[heads,64,64]` bias into softmax `input2`(`2010`);
  FFN(`2053-2073`)->routed loop (INC3)/CUTLASS(INC4); LN eps 1e-3 + zero beta.
- **Policy** `AttentionPolicyHead`+`PolicyMapLayer` (`2083`, kAttnPolicyMap) ->
  1858: REUSE near-verbatim (Hero policy has biases, matches). **Value**: ADAPT
  bias-free attention-read WDL(3), `wdl_=true`; drop MLH.
- **Gate harness** feeds the oracle's raw `(N,64,112)` planes straight to the
  forward (bypassing board->planes), compares to `post_stem`/`post_layer0`/
  `post_trunk`/`policy`/`wdl` npy (GCS `hero/oracle/`). So the forward needs an
  internal `raw planes -> outputs` entry the harness calls directly.

### INC3 stem — exact op mapping (hero.py -> lc0 cuda kernels), PINNED

KEY: lc0's `LayerNorm(N,C,out,input,bias,skip,g,b,eps,alpha,act)` computes
**`normalize(activate(input+bias)*alpha + skip)*g + b`** (common_kernels.cu) —
the act is PRE-normalize, matching Hero's `LN(mish(...))` / DeepNorm exactly.
So Hero's stem is lc0 kernels in this order (all weights bias-free; pass a zero
buffer for LN bias/beta; eps 1e-3):

1. preproc (factorised, 2 gemms, NO activation, board-level over 768=64*12):
   `convertNCHWtoNHWC`(12-plane slice) -> `cublasXgemm(preproc.0 [128,768])`
   -> `cublasXgemm(preproc.1 [8192,128])` giving pos[N,64,128];
   then `inputPreprocessForAttentionBody(scratch, planes, pos, N, 112, 128,
   true)` concats -> [N,64,240]. (lc0's ip_emb_pre is the SAME shape but ONE
   gemm; Hero adds the 128 bottleneck gemm.)
2. embed: `cublasXgemm(embed [d,240])` ->
   `LayerNorm(N*64, d, ..., bias=0, skip=null, embed_ln_g, beta=0, 1e-3,
   alpha=1, act=MISH)`  == normalize(mish(gemm)).
3. embed-ffn (DeepNorm): `cublasXgemm(embed_up [embed_dff,d])` -> mish
   (`addBiasBatched`/Activate) -> `cublasXgemm(embed_down [d,embed_dff])` ->
   `LayerNorm(..., bias=0, skip=x, embed_ffn_ln_g, beta=0, 1e-3, alpha=(2L)^-0.25,
   act=NONE)` == normalize(x + alpha*down).

Gate 3a: DONE (2026-09-17) — stem reproduces the oracle, mean|d| 0.00013,
worst 0.1% rel (fp16-vs-fp32). Whole stem->lc0-kernel mapping validated.
Same LN identity powers the encoder LN1 (act=MISH) / LN2 (DeepNorm skip) and
value/policy head mishes — so the whole non-FFN path reuses lc0 kernels; only
static-bias + expert-FFN are genuinely new.

- **INC3 (next, the big one)** — the CUDA forward: fork lc0's `CudaNetwork`
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
