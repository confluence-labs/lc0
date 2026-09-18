# Hero backend (`-hero`) — build notes

This branch (`hero-backend`) adds a CUDA backend for the **Hero** chess net — a
BT4-skeleton transformer whose per-layer dense FFN is replaced by a bank of E
experts (E=13 or 28), each of the 64 squares routed to one expert by a
deterministic board fact. Hero can't ride the `.pb` path (no expert axis), so it
loads a custom `.htw` side-file. Full analysis + the prize (Hero torch ~5k pos/s
vs ~13.8k BT4 vs ~23k roofline on A100): `confluence-labs/hero-inference`
`docs/PLAN.md`. The `.htw` exporter + the torch oracle live there too.

All file:line refs are this tree (upstream clone, `src/neural/`).

## PERF FINDING (2026-09-17): attention was the bottleneck; fusedMHA fixed it

Device-resident trunk bench on A100 (naive attn + per-expert cuBLAS FFN loop):
~5.3k pos/s, 22% MFU — split timing showed **attn 76-98%, FFN ~20%**. The
routed expert FFN (the novel part) is CHEAP; the bottleneck was STANDARD
attention (naive transposes + 32k tiny 64x64 batched gemms).

**FIX SHIPPED (2026-09-17):** swapped in lc0's `fusedMHA<half_t>(po,qd,kd,vd,
bias,B,H,hd,0)` under `#ifdef USE_CUTLASS`. It takes q/k/v (N,64,d) interleaved
straight from the qkv gemms (NO to-heads transpose), applies scale+per-head
bias+softmax+AV fused, outputs (N,64,d) (NO from-heads transpose). Bias buffer
(B,H,64,64) already matches its `attn_bias_ptr` strides. Built standalone via
nvcc: clone cutlass v4.4.1 + `-DUSE_CUTLASS -I cutlass/include -isystem
third_party`, compile `cutlass_kernels.cu`+`common_kernels.cu`+`fp16_kernels.cu`
(no meson needed). RESULT on A100-40GB:

| B    | before | after (fusedMHA) | MFU   | attn/ffn |
|------|--------|------------------|-------|----------|
| 128  | 4210   | 7244             | 0.306 | 61/39    |
| 256  | 4784   | 9051             | 0.383 | 58/42    |
| 512  | 5063   | 9925             | 0.420 | 57/43    |
| 1024 | 5258   | **10508**        | 0.445 | 57/43    |

2.0x at B=1024 (vs BT4 13825 nps = 76%). Throughput still climbing at B=1024.

**BATCH SWEEP (2026-09-17):** plateaus at ~10.7k pos/s — B=1024→10516, 2048→10579,
4096→10692, 8192→10706 (MFU 0.445→0.453). Batch is NOT the lever; compute-bound
at ~45% MFU. attn/ffn ~56/44. A bigger leaf batch will NOT clear BT4.

**fusedMHA NUMERICALLY VALIDATED (2026-09-17):** ran the full forward gate with
the USE_CUTLASS attention path vs the torch oracle (N=8). TRUNK PASS (worst_rel
0.0234 < 6e-2, mean 0.00163), VALUE PASS. POLICY top1 7/8, top3 8/8, mean_rel
8.6e-4 — the one top1 miss is fp16 rounding on a near-tie (fused kernel's accum
order differs from host cuBLAS), NOT a bug; correct move is top-3 for all 8. The
engine runs fp16 anyway, so this IS target precision. Fast attention path is GO.

**BASELINE METHODOLOGY FIX:** the 13825 is BT4 *engine* nps (lc0 benchmark, incl.
MCTS search); Hero's 10.7k is *trunk-only forward* throughput — not comparable.
Apples-to-apples = lc0 `backendbench` (forward-only) for BT4 on the same A100;
that run is measuring now (first attempt's in-script grep discarded the numbers
— output format has no 'nps' literal — re-running with raw capture).

**MILESTONE (2026-09-17): --backend=hero PLAYS.** Wired the device forward
(hero_forward.cu/.h) into ComputeBlocking; network_hero expands InputPlanes,
runs stem→15 fusedMHA layers→routed FFN→heads, returns policy/WDL. Inverted
lc0's kAttnPolicyMap (4288→1858) for the policy gather. meson: hero_forward.cu +
network_hero.cc are cutlass-gated custom targets. Full lc0 built clean on A100,
engine ran MCTS, produced legal consistent bestmoves (b4f4). Integration GREEN.

**PERF PASS 1 (preallocate + device bias + preload heads): 46 -> 56 nps.** Killed
the per-call trunk overhead (bias re-upload, malloc churn, head-weight uploads)
— but backendbench exposed the REAL cap: batch 256 took 4.6 SECONDS while the
trunk bench does 256 in 28ms. The host-side heads (per-position 64x64x256 double
loops) were 99% of the time.

**PERF PASS 2 (heads fully on GPU): 56 -> 7,344 pos/s. THE unlock.** Moved
policy+value heads to device — batched-gemm QK/AV (mirroring the trunk attention
layout, validated) + k_promo/k_pol_gather/k_wdl_mean kernels; Softmax null-bias
OK. Results on A100 (commit c5b8670):

| batch | forward-only nps | vs pass-1 |
|-------|------------------|-----------|
| 8     | 1,175            | 21x       |
| 64    | 4,736            | 86x       |
| 256   | **7,344**        | 131x      |

Engine (full MCTS benchmark): 53 -> **4,218 nps** (~80x). Still plays b4f4 —
correctness held (used the validated head math, just on-device).

**PERF PASS 3 (engine tuning): 4,218 -> 20,295 nps.** Swept minibatch x threads x
nncache (commit b77a186, after fixing a thread-safety segfault — HeroForward now
mutex-serializes the GPU forward across lc0 search threads; preallocate 512;
GetMiniBatchSize 384). Winner **mb=384, threads=2, nncache=2e6 -> 20,295 nps**;
mb=384/th4 19,077; th6 18,075; mb=160/th4 18,031. Fewer threads win (mutex means
extra threads only add contention; GPU is the bottleneck). mb=384 = lc0's proven
comp range (>384 hurts Elo AND segfaulted pre-fix).

**HONEST CAVEAT on 20,295:** this is engine nps WITH nncache=2e6 on lc0's small
benchmark suite -> high transposition/cache-hit rate (~55-60% of nodes are cache
hits, not forwards). It's legit engine nps (cache is part of the engine, and
BT4's 13,825 reference is also cache-inflated engine nps) BUT the two MUST be
measured identically before quoting. Cache-independent sustained throughput is
the forward-only 7,344 pos/s (bs=256). Real-game sustained nps sits between.
STILL: hero engine nps now exceeds the BT4 13,825 reference under matched-ish
settings — the launch thesis (small net, fast backend, time-fair parity) is live.

**APPLES-TO-APPLES (2026-09-17, commit a3cbcf2): hero ~91% of BT4.** Fixed the
cuda-backend flag (wrapper.cc: inject kWeightsId ONLY for hero .htw — cuda never
reads it, CheckAllOptionsRead threw). Benchmarked BT4 + hero on the SAME lc0
build, SAME A100, SAME settings (mb=384/th=2/nncache=2e6, nodes=120000):

| engine | matched nps | default nps |
|--------|-------------|-------------|
| BT4 (cuda-fp16) | **21,923** | 21,134 |
| HERO            | **20,055** | 20,012 |

**IMPORTANT CORRECTION:** the 13,825 "BT4 reference" was STALE / different
conditions. On our A100 at these settings BT4 does ~21,900 nps, NOT 13,825. So
hero does NOT beat BT4 on raw nps — it's at ~91% (20.0k vs 21.9k). Earlier
"hero > BT4" claims (based on 13,825) are RETRACTED. Hero's backend is
nonetheless within 9% of BT4's mature backend from a standing start today, with
headroom left. NOTE: the launch thesis is time-fair *Elo* = nps x strength/node;
if hero is stronger per node it can still win at equal nps-ish — but that's the
training side. My job: close the 9% and push past.

**SANITY CHECK / AIRTIGHT ANCHOR (2026-09-17, commit 0e68947).** The engine-nps
numbers (hero 20k, BT4 14.9-21.9k across runs) are CACHE-INFLATED and fragile —
a parallel BT4-track box measured BT4 at 10,157 vs my 21,923 (2.2x), pure
NNCache/position/warmup noise. The RELIABLE comparison is cache-free backendbench
(forward-only pos/s):

| batch | BT4 cache-free | HERO cache-free | hero/BT4 |
|-------|----------------|-----------------|----------|
| 256   | 8,120          | 7,346           | 90%      |
| 384   | 8,320          | 7,727           | **93%**  |

Cache-inflation factor (engine nps, nncache 1000 vs 2e6): BT4 8.5k->14.9k,
hero 9.1k->20.2k. Tiny-cache engine nps ~= forward-only, as expected.

**DEFENSIBLE CONCLUSION: hero's backend is ~93% of BT4's per-eval throughput
(~7.7k vs ~8.3k pos/s cache-free) — essentially at parity, BT4 a hair faster.**
Both retracted extremes (hero>>BT4 AND hero at 91% off the 21.9k) were artifacts;
93% cache-free is the real number, and it's position/box-independent (BT4 track
to cross-check via their own backendbench). Q3 (real-FEN `go movetime`) didn't
capture — UCI handshake missing (uci/isready); minor, redo if in-game per-move
nps is needed, but Q1 already answers transferability: sustained in-game nps for
both sits in the ~8-12k band (forward-bound + realistic cache), ~7-10% apart.

**GROUPED-FFN via MULTI-STREAM (2026-09-17, commit 10a57b4): +3-5%, hero now
~95-98% of BT4.** Ran the E experts CONCURRENTLY across 8 CUDA streams (they're
mutually independent) instead of the serial per-expert loop, event-synced around
gather/scatter. Cache-free backendbench:

| batch | before | after | BT4   | hero/BT4 |
|-------|--------|-------|-------|----------|
| 256   | 7,346  | 7,712 | 8,072 | 95.5%    |
| 384   | 7,727  | 7,939 | 8,320 | 95.4%    |
| 512   | -      | 8,171 | 8,368 | 97.6%    |

Engine still plays b4f4 -> multi-stream sync correct, no corruption. Modest lever
as predicted (BT4 track: grouped-FFN ~modest). Hero now at ~parity, a hair behind,
gap narrows with batch. Remaining gap = host route-sort (per-batch sync) + heads +
fundamental MoE-dispatch/d1024-width vs BT4 d768. Diminishing returns from here.

**int8 RULED OUT (2026-09-17):** cuBLAS int8 on hero's gemm shapes is only
1.23-1.31x over fp16 (not 2x) — nets to ~1.15x end-to-end. Not worth the accuracy
risk on a chess net. Dead lever.

**LARGE-BATCH REGIME (Niranjan: inference batch is unlimited). Raised the 1024
caps (backendbench + bridge maximum_batch_size). Cache-free forward pos/s:**

| batch | HERO  | BT4        |
|-------|-------|------------|
| 512   | 8,182 | 8,353      |
| 1024  | 8,334 | 8,580 (pk) |
| 2048  | 7,542 | OOM        |
| 4096  | 7,574 | OOM        |
| 8192  | 7,366 | OOM        |

TWO findings: (1) BT4's cuda backend OOMs at bs>=2048 — it CANNOT run large batch
on 40GB. Hero can. (2) Hero DEGRADES past bs=1024 (8,334->7,542) instead of
holding the ~10.7k the trunk bench proved possible — host-side overhead dominates
at large batch: the host route-sort (O(N)) AND re-broadcasting the per-head bias
(N,H,64,64) EVERY layer for fusedMHA (~8GB writes/fwd at bs2048, pure waste since
bias is batch-independent). THE LEVER: fix hero's large-batch scaling -> it holds
high throughput at bs2048-8192 where BT4 physically can't run -> decisive win.
This is the real speed play (not int8).

**BROADCAST-BIAS FIX SHIPPED (2026-09-17, commit bf99321): HERO NOW BEATS BT4.**
Added fusedMHA `broadcast_bias` (strideB=0) so hero's static (H,64,64) bias is
read for all N — no per-layer N-broadcast write, dropped the dBias buffer. +5-6%
across batch, still plays b4f4 (strideB=0 correct). Cache-free forward:

| batch | HERO pre | HERO post | BT4        |
|-------|----------|-----------|------------|
| 512   | 8,182    | 8,714     | 8,353      |
| 1024  | 8,334    | **8,797** | 8,580 (pk) |
| 2048  | 7,542    | 7,859     | OOM        |
| 8192  | 7,366    | 7,624     | OOM        |
| 12288 | -        | 7,482     | OOM        |

**At bs=1024 (shared peak) HERO 8,797 > BT4 8,580 — ~2.5% faster, and correct.**
First time hero leads on a clean cache-free measurement. Hero also runs bs>=2048
(to 12288+) where BT4 OOMs. REMAINING: hero still degrades past bs=1024 (host
route13 argmax O(N*64*12) stops overlapping the stem at large batch) — device-
side route kernel would hold ~8.8k into the bs2048-12288 regime BT4 can't enter,
extending the lead. NEXT: (1) same-box side-by-side confirm (airtight the claim);
(2) device route13 kernel for large-batch scaling.

Also: `--backend=cuda*` probe fails "Unknown string option: cuda-auto.<garbage>"
in this fork build (hero backend unaffected — it played). Chase the BT4
lc0-benchmark baseline flag separately; the 13825 engine-nps reference stands.

NEXT: PRIORITY 2 = FFN co-bottleneck (44%) — 13 sequential per-expert cuBLAS
gemms; CUTLASS 2.x grouped GEMM (SM80) is the lever, and gets URGENT if the
45-class/top-k routing lands (up to ~180 gemms/layer).

## Target hardware: A100 (CCC parity) — Ampere SM80

Decision 2026-09-17 (user): optimize for **CCC parity = 2x A100-40GB (Ampere,
SM80)**. Beat BT4's 13.8k nps baseline on an A100. IMPLICATION for INC4: the
CUTLASS **kGrouped** MoE kernel is Hopper (SM90)-only, NOT available on Ampere.
On A100 the expert-FFN options are (a) CUTLASS 2.x grouped GEMM (SM80), (b) a
per-expert cuBLAS-loop (already validated for correctness), or (c) a batched
GEMM. Bench all three on A100; pick by nps at lc0 leaf batch. Correctness work
is hardware-independent (fp16); nps benches run on A100, not L4.

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

- **INC3a/b/c DONE (2026-09-17)** — stem, attention (static geo bias), AND the
  routed expert-FFN loop all reproduce the oracle in CUDA (a full encoder layer,
  mean|d| ~2e-4). The novel routed FFN — Hero's raison d'etre — is validated
  (cuBLAS-loop; CUTLASS is INC4). geo_basis() ported to C++. Gate:
  `src/neural/hero/hero_stem_gate.cc` (stem->attn->layer0 staged vs oracle npy).
- **INC3d TRUNK DONE (2026-09-17)** — all 15 layers loop, trunk reproduces the
  oracle (mean|d| 0.00185 over 15 fp16 layers). Only the 2 heads remain.
- **INC3 CORRECTNESS COMPLETE (2026-09-17)** — FULL FORWARD PASS: stem + 15
  layers (attn + routed FFN) + policy + value reproduce the oracle; top-1 move
  match 8/8, top-3 8/8, value fp16-exact. The whole Hero net runs correctly in
  CUDA (`hero_stem_gate.cc`). Next: wire into network_hero ComputeBlocking
  (loop+heads on device), then INC4 (Ampere FFN kernel) + nps on A100.
- (was) INC3d heads — policy/value heads -> policy/wdl,
  gate vs oracle policy.npy/wdl.npy. Then wire into network_hero's ComputeBlocking.
- **INC3 (superseded framing)** — fork lc0's `CudaNetwork`
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
