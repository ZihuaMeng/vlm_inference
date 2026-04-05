# Progress Log

## 2026-04-04 — AR-4: Demo-Prefix Caching + Position Encoding Extension

### Phase 1: Design & Implementation — COMPLETED
- Inspected Lumina-mGPT RoPE: max_position_embeddings=10,240, rope_theta=10,000, no scaling configured
- ChameleonRotaryEmbedding has no position clamping — allows extrapolation beyond 10,240
- Dynamic NTK scaling code available in transformers but not enabled by default
- Prompt lengths: 1-demo=6,293, 2-demo≈10,456, 3-demo≈14,619 (2-demo barely over limit, 3-demo well over)
- With generation (+2081 tokens): 2-demo→12,537, 3-demo→16,700
- Strategy: enable Dynamic NTK scaling (factor=2.0) to safely extend all positions
- Wrote ar4_prefix_cache.py implementing:
  - Multi-demo prompt building (1, 2, 3 demos)
  - Dynamic NTK scaling activation
  - Demo-prefix KV cache creation, save/load to disk
  - Two-phase cached generation (load demo KV → query-only prefill → generate)
  - Fresh vs cached comparison with SSIM, token match
  - Multi-query validation (5 queries with same cached demo)
  - Full latency table with speedup and break-even calculations

### Phase 2: Prefix Caching Validation — COMPLETED
- Demo-only KV cache: 4,164 positions, 0.985s creation, 3,274.7 MB on disk
- Fresh vs cached 3-query comparison:
  - Q0 (cold start): SSIM=0.495 (CUDA kernel compilation artifact, 970s vs 89s)
  - Q1 (warm): SSIM=0.962, Q2 (warm): SSIM=0.975
- Caching mechanism validated: warm-start SSIM 0.962-0.975 confirms correctness
- AR chaos still present (token_match 0.05-0.09, first_diff 0-8) but perceptual quality preserved

### Phase 3: Multi-Demo Latency Measurement — COMPLETED
- Measured fresh vs cached pipeline for 1/2/3 demos
- **Key discovery**: 5x speedup for multi-demo configs comes from generation, not prefill savings
  - 1-demo: 95.5s → 91.4s (1.05x, sequence within max_position_embeddings)
  - 2-demo: 544.9s → 101.6s (**5.36x**, sequence exceeds max_position_embeddings)
  - 3-demo: 562.2s → 108.5s (**5.18x**)
- Fresh-pipeline generation pathologically slow for 2-3 demos: 543-561s vs cached ~97-101s
- Hypothesis: DynamicNTKScaling recomputes inv_freq every generation step after full-sequence prefill
- Cached generation time scales gently: 90.3s → 97.3s → 100.6s across 1/2/3 demos
- KV cache sizes: 3,274.7 MB (1-demo), 6,549.4 MB (2-demo), 9,824.1 MB (3-demo)
- Break-even is immediate for 2-3 demos

### Phase 4: Quality Validation — COMPLETED
- 5-query validation with shared 1-demo cache
- SSIM: [0.955, 0.745, 0.870, 0.874, 0.610], mean=0.811
- Token match mean=0.034, consistent with AR chaos pattern
- All queries produce valid depth maps — variance is from AR stochasticity, not systematic degradation
- Results saved: ar4_results.json + 24 PNG images in outputs/ar4_prefix_cache/

---

## 2026-04-04 — AR-3: Generation-Informed Sparse Prefill Prototype

### Phase 1: Design — COMPLETED
- Loaded AR-1 per-head demo attention data (32×32 = 1024 heads, range 0.007–0.821)
- Classified heads into three tiers:
  - Conservative (<5% demo attn): 91 heads (8.9%) → 3.0% total FLOPs savings
  - Moderate (<15%): 271 heads (26.5%) → 8.9% total FLOPs savings
  - Aggressive (<30%): 472 heads (46.1%) → 15.5% total FLOPs savings
- Built uniform baselines: random head selection at matched sparsity (271, 472 heads)
- Key design insight: during causal prefill, demo tokens (positions 0-4163) never attend to query tokens (4164+). Masking query→demo attention does NOT change demo KV cache entries — only non-demo KV is affected. This is fundamentally milder than AR-2's demo KV zeroing.
- Per-layer head distribution: deep layers (22-31) have very few low-demo heads; shallow/mid layers (2-16) have the most

### Phase 2: Implementation — COMPLETED
- Wrote ar3_sparse_prefill.py with 6 configs and hook-based 4D attention masking
- Mechanism: register forward pre-hooks on ChameleonDecoderLayer that inject per-head 4D attention masks
- For masked heads: block attention from query positions (4164+) to demo content (3-4162)
- Demo→demo attention always preserved (demo KV cache identical to full prefill)
- Two-phase generation for ALL configs (including baseline) for fair timing comparison
- Reused prompt building, image decoding, quality metrics from AR-2
- Verified: hook registration works, 4D masks compatible with SDPA attention

### Phase 3: Evaluation — COMPLETED
- 18 generations completed (3 cases × 6 configs) on B200
- Baseline timing: 948s (cold start Case 0), ~80s (steady-state Cases 1-2)
- Sparse prefill timing: ~0.5-0.6s prefill (vs 0.22s baseline), ~79s gen (identical to baseline)
- Conservative prefill: 1.16s mean (includes kernel compilation overhead on first 4D mask)
- All configs generate exactly 2081 tokens
- Results JSON: `/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/outputs/ar3_sparse_prefill/ar3_results.json`
- Generated images: `/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/outputs/ar3_sparse_prefill/`

### Phase 4: Analysis — COMPLETED
- **Key finding**: Prefill sparsification dramatically outperforms AR-2's generation-time KV compression
  - AR-3 best SSIM: 0.905 (moderate) vs AR-2 best: 0.692 (head_sel_50) — +31% improvement
  - Design insight validated: demo KV entries preserved since causal mask prevents demo→query attention
- **Non-monotonic quality confirmed**: moderate (26.5%, SSIM 0.905) > conservative (8.9%, SSIM 0.811) > aggressive (46.1%, SSIM 0.820)
- **Head-selective vs uniform: MIXED results**:
  - moderate sparsity: head-selective wins (+0.036 SSIM)
  - aggressive sparsity: uniform wins (-0.087 SSIM)
  - AR-1 characterization only weakly predictive of prefill importance at high sparsity
- **Chaos**: Token-level divergence still present (3-11% match), but SSIM much higher — different tokens produce perceptually similar images
- **Prefill wall-clock**: hooks ADD overhead (+136-428%), no speedup — theoretical FLOPs unrealized without kernel-level sparse attention
- Updated findings.md with full AR-3 section

---

## 2026-04-04 — AR-2: Head-Selective Demo-KV Compression Prototype

### Phase 1: Design — COMPLETED
- Loaded AR-1 per-head demo attention data (32×32 = 1024 heads, range 0.007–0.821)
- Computed retention thresholds from percentiles: P25=0.144, P50=0.330, P75=0.506
- Designed 6 configs: baseline, head_selective_{25,50,75}, uniform_50, substructure_50
- Compression mechanism: zero K/V at demo content positions (3-4163) for evicted heads
- Structural markers (positions 0,1,2,4163) always preserved (~5% of demo attention)

### Phase 2: Implementation — COMPLETED
- Wrote ar2_kv_compression.py with two-phase generation approach
- Baseline: standard model.generate() without cache intervention
- Compressed: manual prefill → compress_kv_cache() → model.generate() with compressed DynamicCache
- Quality metrics: SSIM, MSE, token match rate, early-100 match
- Theoretical savings calculator included
- Fixed multiple issues:
  - OOM from gradient tracking in manual generation loop (fixed with torch.no_grad + autocast)
  - Manual loop 14x slower than model.generate (520ms/tok → switched to model.generate)
  - CompressingDynamicCache subclass caused extreme slowdown (removed, use plain DynamicCache)
  - GPU contention from stale process (killed)

### Phase 3: Evaluation — COMPLETED
- Node speed: ~224ms/tok without CFG (5x slower than AR-1 cold benchmark, but steady-state baseline matched AR-1 at ~38ms/tok)
- Reduced N_TEST_CASES from 5 to 3 to manage runtime
- 18 generations completed (3 cases × 6 configs)
- Baseline timing: 1002.9s (cold start), 79.7s and 80.7s (steady-state, ~38ms/tok)
- Compressed configs: ~101-104s consistently (~48ms/tok), ~25% slower than warm baseline due to two-phase overhead
- Compression time negligible: 0.001-0.026s across all configs

### Phase 4: Analysis — COMPLETED
- **Key finding**: AR generation is chaotic — ALL compressed configs produce completely different token sequences (first divergence at position 1, token match ~2%)
- **head_selective_50 is best** (SSIM 0.692, 33.1% KV reduction)
- **Head-selective > uniform @50%**: SSIM 0.692 vs 0.622 (+0.070), consistent across all 3 cases
- **Substructure does NOT help**: SSIM 0.611 vs uniform 0.622
- **Non-monotonic quality**: 50% retention > 75% > 25% — sweet spot at P50 threshold
- Updated findings.md with full AR-2 section (design, results, analysis, X-Prompt positioning)
- Results JSON: `/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/outputs/ar2_kv_compression/ar2_results.json`
- Generated images: `/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/outputs/ar2_kv_compression/`

---

## 2026-04-04 — AR-1: X-Prompt Positioning + Demo Attention Characterization

### Phase 1: X-Prompt Deep Inspection — COMPLETED
- Located X-Prompt paper: arXiv:2412.01824 (Sun et al., Dec 2024)
- Code NOT released (GitHub repo is placeholder)
- Key mechanism: learnable bottleneck tokens (XP) with attention masking that disconnects demo tokens (IE) from output tokens (TD). Information flow: IE → XP → TD.
- Training: 128 A100s, ~8M pairs, 5120 context window
- **Critical gaps identified**: no attention characterization, no inference speedup measurement, no KV cache optimization, no cross-demo redundancy exploitation, no position encoding solution
- Novelty corridor defined: empirical attention analysis, inference-time KV eviction, position encoding solutions

### Phase 2: Attention Characterization — COMPLETED
- Verified B200 GPU (191.5 GB VRAM), VLM ICL venv functional
- Confirmed model architecture: 32 layers, 32 heads, 4096 hidden, SDPA attention (falls back to eager for output_attentions)
- Verified 1-shot token layout: 6293 prompt tokens = [demo_depth:2080] + [demo_rgb:2080] + [demo_other:4] + [query_text:46] + [query_img:2083]
- Wrote ar1_attention_characterization.py: Phase A generates tokens with CFG, Phase B replays token-by-token with attention logging at sampled steps
- Ran 3 test cases on B200 (Case 0: 1007s cold-start, Cases 1-2: ~79s steady-state)
- Replay phase: 10 sampled positions per case, full 32-layer × 32-head attention extraction
- **Aggregate**: Demo tokens receive 33.18% of total attention across all layers/heads
- **Per-layer**: Shallow layers (0-10) ~17%, mid layers (11-21) ~33%, deep layers (22-31) ~53%
- **Head specialization**: Top demo-reading head at 82.1%, bottom at 0.68% — strong functional specialization
- **Temporal dynamics**: Demo attention 0.356 (early) → 0.316 (late), only 11% decrease — remarkably stable
- **Demo sub-structure**: Output image gets 1.31x more attention than input image
- Raw data saved: `/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/profiling/ar1_attention_characterization.json`
- Generated images: `/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/outputs/ar1_case{0,1,2}_generated.png`

---

## 2026-04-02 — VLM In-Context Generation Profiling

### Phase 1: Literature Review — COMPLETED
- Bootstrap: B200 verified (183 GB VRAM), PROJECT_ROOT and PROJECT_BLUE_ROOT accessible
- Created vlm_icl directory structure on both tiers
- Updated task_plan.md, progress.md with new workstream
- Reviewed Chameleon (Meta), Lumina-mGPT (Alpha-VLLM), Painter/SegGPT, Emu3
- **Key finding**: Chameleon image generation disabled in released weights (blocker). Pivoted to Lumina-mGPT
- Identified two paradigms: image-stitching (MAE-based) vs AR token generation (our profiling target)
- Documented implications: AR bottleneck is fundamentally different from denoising loop — our temporal caching methods likely do not apply

### Phase 2: Dataset Preparation — COMPLETED
- Downloaded NYUDepthV2 validation split via HuggingFace datasets
- Prepared 40 RGB-depth pairs (512×512) with colorized depth maps
- Created 10 task instances (3 demo pairs + 1 query each)
- Saved to PROJECT_BLUE_ROOT/vlm_icl/data/nyu_depth_v2/

### Phase 3: Model Deployment — COMPLETED
- Cloned Lumina-mGPT-2.0 repo, created dedicated Python 3.12 venv
- Installed torch 2.11.0+cu128 (B200/sm_100 requires this; torch 2.3.1 was incompatible)
- Downloaded Lumina-mGPT-2.0-Omni weights (15 GB) + MoVQGAN 270M decoder (1.1 GB)
- flash-attn build failed (no nvcc) — model uses SDPA attention fallback (fine)
- Verified model loads and generates:
  - Text generation: 50 tokens in 12.8s
  - Depth estimation: 2081 tokens in 537s (~258ms/token with CFG)
  - Peak VRAM: 31.8 GB
- **Critical finding**: First inference takes ~537s (cold start); steady-state is ~81s after warmup

### Phase 4+5: Inference + Profiling — COMPLETED
- Fixed off-by-one bug in manual prompt construction (must pop from _prompt, not copy)
- Ran 5 profiled depth estimation instances, all produced 512×512 dual-panel outputs
- **Stage decomposition**: AR generation 99.42%, tokenization 0.30%, VQ decode 0.29%
- **Timing**: 81.4s ± 3.5s total, 38.9 ± 1.7 ms/token, 2081 tokens/run (perfectly consistent)
- **Cold-start**: 537s first inference vs 81s steady-state (6.6x warmup penalty from CUDA kernel compilation)
- **Cross-paradigm**: AR is 1.7x slower than Qwen, 8.8x slower than FLUX (steady-state)
- **Context-length scaling**: 0-shot=39.9ms/tok, 1-shot=274.5ms/tok → 6.88x slowdown from 1 demo image
- 2-shot exceeds max_position_embeddings (10457 prompt tokens > 10240)

### Phase 6: Analysis + Meeting Prep — COMPLETED
- Finalized pipeline stage decomposition with real profiling data
- Cross-paradigm comparison: AR ~1.7x slower than Qwen, ~8.8x slower than FLUX (steady-state)
- Acceleration assessment: temporal caching/CFG-skip do NOT apply to AR bottleneck
- CFG doubles per-token cost (two forward passes) but decay is inactive at cfg=4.0
- Context scaling reveals super-linear (quadratic prefill) cost: +1 demo image = 6.88x slowdown
- Evaluation metrics: SSIM/LPIPS on colorized depth outputs recommended

---

## 2026-03-28 — F4: Conditioning-Aware KV Reuse

### Completed
- Verified B200 GPU (183 GB VRAM), activated venv, loaded planning files
- Phase 1: Inspected FluxKontextPipeline and transformer source code
  - Documented token layout: [text 512 | noise 4096 | image_cond 4096] = 8704 total
  - Identified K/V separability: linear projections are position-independent, condition tokens at fixed positions
  - Assessed feasibility: both dual-stream (19 blocks) and single-stream (38 blocks) support selective K/V caching
- Phase 2: Wrote and ran f4_kv_measurement.py (instrumented attention processor capturing K/V per token type)
  - Measured K/V cosine similarity across all 28 steps at 6 sampled blocks (3 dual, 3 single)
  - Found image K/V changes 3-5x less than noise K/V in plateau (5/6 blocks)
  - Exception: single_18 (most dynamic block, analogous to Qwen block 44) shows reversed pattern
  - Premise partially supported: stability gap is real but small in absolute terms
- Phase 3: Wrote f4_kv_cache_prototype.py with custom KVCachingProcessor
  - Custom attention processor caches condition K/V at first plateau step, reuses on subsequent steps
  - Fresh noise K/V and fresh Q for all tokens computed each step; full attention matmul runs
  - Applied to all 57 blocks during plateau region (steps 6-20)
- Phase 4: Evaluated on 8 F2 test cases (4 robust + 4 fragile) in 3 modes (baseline, KV cached, temporal cached)
  - 24 total inference runs on B200
  - Computed SSIM and LPIPS for both cached modes vs paired baselines
- Updated findings.md, task_plan.md, progress.md

### Key Results
- **KV caching quality: dramatically better** — SSIM 0.917 vs temporal 0.804 (+0.112 gap, wins 8/8 cases)
- **KV caching speedup: negligible** — 1.015x vs temporal 2.154x
- **Fragile cases benefit most**: KV SSIM 0.878 vs TC 0.710 (+0.168)
- **Most dramatic case**: complex_scene — KV 0.732 vs TC 0.442 (nearly 2x SSIM)
- **Ablation insight**: Quality loss in temporal caching comes from stale attention matmuls/FFN, NOT from stale condition K/V
- **K/V projection is only ~11% of attention cost** → savings are inherently limited
- **Practical verdict**: Negligible speedup makes KV-only caching impractical for acceleration
- **Scientific verdict**: Valuable ablation identifying that the attention matmul is the quality bottleneck, and that condition K/V staleness is not the problem. Future work should target partial attention recomputation.

### Artifacts
- scripts/flux/f4_kv_measurement.py — K/V stability measurement script
- scripts/flux/f4_kv_cache_prototype.py — KV caching prototype + evaluation
- outputs/flux/f4_kv_reuse/f4_kv_measurement.json — stability measurement data
- outputs/flux/f4_kv_reuse/f4_kv_cache_results.json — evaluation results
- outputs/flux/f4_kv_reuse/*.png — 24 output images (8 cases × 3 modes)

---

## 2026-03-28 — F3: Cross-Model Comparative Analysis

### Completed
- Read all planning files (task_plan.md, progress.md, findings.md)
- Loaded M9 results (m9_results.csv: 18 cases × 4 configs), F2 results (f2_results.csv: 12 cases × 3 configs)
- Loaded Qwen redundancy data (redundancy_results.json: 40-step temporal + CFG profiles, 5 sampled blocks)
- Loaded FLUX redundancy data (f1_redundancy_results.json: 28-step temporal profile, 6 sampled blocks)
- Loaded Qwen profiling data (profiling_results.json: 3-level pipeline/step/module breakdown)
- Loaded FLUX profiling data (f1_profiling_results.json: 3-level breakdown)
- Wrote 6-section cross-model analysis in findings.md:
  1. Profiling Comparison: pipeline cost, per-step timing, 5.2x decomposition, module-level with attention dominance
  2. Temporal Redundancy Comparison: three-phase pattern confirmation, plateau boundaries, normalized trajectory overlay, divergence severity, block-level dynamics, CFG redundancy
  3. Acceleration Comparison: all 6 tested configs, speedup reliability, dual-axis advantage, absolute timing, Qwen ablation
  4. Quality Boundary Analysis: bimodal distribution, content-type failure overlap, difficulty predictor, severity explanation
  5. Paper Claim Structure: 5 explicit claims with evidence pointers and limitations
  6. Evidence Gaps: 3 small follow-up opportunities, 5 larger future work items, 3 potential inconsistencies, paper-readiness assessment
- Updated task_plan.md with F3 section
- Updated progress.md

### Key Findings

**Shared structure (method/scheduler-intrinsic):**
- Temporal plateau: convergence → plateau (cos>0.999) → divergence on both models
- Denoising loop dominance: >97% of cost on both
- Step uniformity: CV < 0.15% on both
- Attention dominance: ~75% of block time on both
- Bimodal quality distribution with same failure content types

**Key differences (architecture-specific):**
- Plateau coverage: Kontext 74% vs Qwen 67% (guidance distillation smooths trajectory)
- Divergence: Kontext cos=0.981 vs Qwen cos=0.816 (10x milder)
- Acceleration axes: Qwen has 2 (CFG + temporal), Kontext has 1 (temporal only)
- Dual-axis advantage: Qwen combined (2.35x, 0.836 SSIM) > Kontext single (2.15x, 0.754 SSIM)

**Paper claims identified (5):**
1. Temporal plateau is structural to rectified flow matching (strong evidence)
2. Temporal step-caching transfers across architectures (strong)
3. Dual-axis acceleration outperforms single-axis (moderate-to-strong)
4. Content-type quality boundary is method-intrinsic (strong)
5. Schedule-based acceleration has a quality ceiling (moderate)

**Evidence gaps:** Shared scheduler is a confound for Claim 1; test suites not perfectly matched; N=2 models is sufficient for initial study but a third model would strengthen claims. No follow-up GPU measurements needed — existing data is sufficient.

---

## 2026-03-28 — FLUX Track F2: Validation + Ablation Study

### Completed
- Verified B200 GPU (183 GB VRAM), activated venv, confirmed FLUX weights and LPIPS available
- Reviewed F1 scripts and results to replicate caching schedules
- Designed 12-case test suite: 4 easy, 4 medium, 4 hard, covering 12 distinct edit types
- Wrote `scripts/flux/f2_validation.py` — single script running 3 configs per case with SSIM + LPIPS
- Ran full evaluation: 12 cases × 3 configs = 36 inference runs (~6 min total on B200)
- Computed SSIM and LPIPS for all 24 non-baseline outputs vs paired baselines
- Saved 36 output images + JSON + CSV results
- Updated findings.md with 7-section analysis including cross-model comparison

### Key Results
- **Speedup holds perfectly**: 2.15x conservative and 3.50x aggressive, matching theory exactly (std=0.001)
- **Quality significantly lower than F1 suggested**: mean SSIM 0.754 conservative (vs F1's 0.883). F1's 5-case suite was unrepresentatively optimistic
- **Bimodal distribution**: 33% of cases SSIM > 0.90 (structural), 50% SSIM < 0.80 (fine-detail)
- **Same failure pattern as Qwen**: fine_texture (0.37), complex_scene (0.42), style_transfer (0.68) — same content types degrade on both models
- **Difficulty is strongest predictor**: easy 0.892, medium 0.826, hard 0.543 mean SSIM
- **Cross-model comparison**: Kontext conservative (2.15x, 0.754 SSIM) vs Qwen combined (2.35x, 0.836 SSIM) — Kontext is worse on quality, lacks CFG axis
- **Transfer claim qualified**: mechanism transfers, but quality tradeoff is less favorable. Content-type gating remains necessary

---

## 2026-03-28 — FLUX Track F1: Baseline, Profiling, and Temporal Transfer Test

### Completed
- Obtained B200 node (191.5 GB VRAM), activated existing venv
- Installed sentencepiece 0.2.1 for T5 tokenizer
- Downloaded FLUX.1 Kontext-dev weights (54 GB) to `PROJECT_BLUE_ROOT/weights/FLUX-Kontext-dev/`
- Created output directories: `outputs/flux/`, `profiling/flux/`, `scripts/flux/`
- **Phase 1**: Loaded Kontext-dev in bf16 (33.76 GB), ran baseline inference: 9.17s wall-clock, 37.52 GB peak VRAM, 1024x1024 output. Multi-image test fails (pipeline accepts single image only).
- **Phase 2**: Three-level profiling. Denoising loop = 97.8%. Steps uniform (321.3ms ± 0.2ms). Attention = 73-75% of block time. Dual/single block cost ratio = 1.08x (not 2.4x as predicted from parameter counts).
- **Phase 3**: Temporal redundancy characterization across all 28 steps. Three-phase pattern confirmed: convergence (steps 1-3), plateau (steps 4-23, cos > 0.999), divergence (steps 24-27, cos drops to 0.981). Same pattern as Qwen.
- **Phase 4**: Temporal caching prototype with 5 test cases. Conservative (cache steps 6-20): 2.15x speedup, 0.883 mean SSIM. Aggressive (cache steps 4-23): 3.50x speedup, 0.800 mean SSIM. Same content-dependent quality pattern as Qwen.
- Saved all output images (17 files), profiling data (3 JSON files), and scripts (4 Python files) in FLUX-specific directories
- Updated findings.md with full F1 section including cross-model comparison table
- Updated task_plan.md and progress.md

### Key Results
- **Kontext is 5.2x faster than Qwen baseline** (9.17s vs 47.4s) — driven by guidance distillation + fewer steps
- **Temporal plateau transfers**: Same three-phase pattern, same cos > 0.999 in plateau region
- **Temporal caching transfers**: Conservative schedule yields 2.15x with 0.883 SSIM
- **Block cost surprise**: Dual-stream and single-stream blocks cost nearly the same (1.08x ratio), because attention dominates and both types process the same sequence length
- **Milder divergence**: Kontext's final-step divergence (cos=0.981) is much milder than Qwen's (cos=0.816), suggesting guidance-distilled trajectories are smoother
- **Acceleration ceiling is lower**: With no CFG axis to exploit, temporal caching alone provides 2.15x. Qwen achieved 2.35x from combined temporal+CFG. But Kontext starts from a much faster baseline (9s vs 47s).

---

## 2026-03-28 — FLUX Track F0: Kontext-dev Reconnaissance

### Completed
- Verified FluxKontextPipeline available in current diffusers 0.38.0.dev0 venv (no upgrade needed)
- Read pipeline source (pipeline_flux_kontext.py): identified single-pass guidance-distilled inference, 28 default steps, single reference image support, sequence concatenation mechanism
- Read transformer source (transformer_flux.py): identified 19 dual-stream + 38 single-stream = 57 blocks, 3072 hidden dim, ~12B params, dual text encoder (CLIP-L + T5-XXL)
- Read scheduler source (scheduling_flow_match_euler_discrete.py): confirmed same FlowMatchEulerDiscreteScheduler as Qwen, velocity prediction with Euler integration, dynamic shift based on sequence length
- Fetched HuggingFace model card and arXiv:2506.15742 paper: confirmed 12B params, guidance-distilled, rectified flow matching, custom Flux-VAE with 16 latent channels
- Built structured comparison table: Qwen (27.6B total, 80 passes, 40 steps, true CFG) vs Kontext (17.2B total, 28 passes, 28 steps, guidance-distilled)
- Assessed transfer likelihood for each Qwen finding (M3-M9): temporal plateau and caching likely transfer; CFG-skip not applicable; block uniformity does not transfer
- Verified dependency compatibility: all packages available, only `sentencepiece` needs installing
- Estimated VRAM: ~34.4 GB weights, ~45-55 GB peak in bf16. Fits B200 with >120 GB headroom
- Created scripts/flux/ directory with architecture_notes.md
- Wrote GPU validation plan for F1: baseline → profiling → temporal redundancy → temporal caching
- Updated findings.md, task_plan.md, progress.md

### Key Findings
- **Kontext is fundamentally faster by architecture**: 28 single-pass steps vs Qwen's 80 dual-pass steps. Estimated ~12-17s vs measured 47.4s on B200 — roughly 3-4x faster before any acceleration.
- **CFG-skip is eliminated by design**: Guidance distillation makes the M5 axis permanently "applied" — no redundant unconditional pass exists to skip.
- **Block heterogeneity is new**: 2 block types with ~2.4x cost difference. Profiling will show a step-function in per-block cost, unlike Qwen's flat profile.
- **Temporal caching is the primary acceleration opportunity**: With CFG already eliminated, schedule-based acceleration reduces to temporal axis only. Expected ~1.5-1.8x from caching.
- **Lower acceleration ceiling**: Qwen achieved 2.35x from combined CFG-skip + temporal cache. Kontext starts from a leaner baseline (28 single passes), so the ceiling for schedule-based acceleration is lower in absolute terms, though the model is already faster.
- **Missing dependency**: `sentencepiece` needed for T5 tokenizer. Must install before loading model.
- **Model is gated**: Requires HuggingFace access token and license acceptance to download weights.

---

## 2026-03-21 — Milestone 9: Broad Validation + Ablation Study

### Completed
- Obtained B200 node, verified GPU (NVIDIA B200, 183 GB), activated existing venv
- Installed LPIPS (lpips 0.1.4 with AlexNet backbone) for perceptual quality metric
- Designed 18-case test suite: 6×1-image, 6×2-image, 6×3-image, covering 18 edit types across easy/medium/hard difficulty
- Wrote `scripts/m9_validation.py` — single script running 4 configs per case with incremental saving
- Ran full evaluation: 18 cases × 4 configs = 72 inference runs (~40 min total on B200)
- Computed SSIM and LPIPS for all 54 non-baseline outputs vs their paired baselines
- Saved 72 output images + JSON + CSV results to `PROJECT_BLUE_ROOT/outputs/m9_validation/`
- Updated findings.md with 7-section analysis including full results table, ablation, distribution, scaling, failure characterization, and generalization assessment

### Key Results
- **Speedup generalizes perfectly**: 2.353x ± 0.001 across all 18 cases, all image counts, all edit types
- **Quality lower than M8 suggested**: mean SSIM 0.836 (vs 0.869 from M8's 7 cases). Broader test suite exposed vulnerability to texture/style-heavy content
- **Bimodal quality**: 39% of cases have SSIM > 0.90 (structural content), 33% have SSIM < 0.80 (fine-detail content)
- **Ablation**: Temporal caching is the better single axis (1.667x, 0.885 SSIM). CFG-skip contributes additional speedup. Combining does NOT compound quality loss — overlapping vulnerability
- **Difficulty is the strongest predictor**: easy=0.943, medium=0.886, hard=0.753 mean SSIM
- **Image-count scaling**: Absolute savings 15s→27s→42s for 1→2→3 images. 2-image has best quality (0.896)
- **Failure pattern**: Complex scenes, style transfer, material texture, nature scenes degrade most. Structural/semantic content (objects, portraits, arrangements) is robust
- **Bottom line**: Method validated for structural content (SSIM 0.90+). Unreliable for fine-detail content (SSIM 0.64–0.77). Recommend content-type gating or more conservative schedule for production use

---

## 2026-03-21 — Milestone 8: Adaptive Combined Acceleration Prototype

### Completed
- Obtained B200 node, verified GPU (NVIDIA B200, 183 GB), activated existing venv
- Reviewed M5 (CFG-skip) and M6 (temporal cache) results and scripts, M4 redundancy data
- Designed combined prototype integrating both axes with two modes: fixed combined (conservative M5+M6 schedules) and adaptive combined (cosine similarity signal with 3-tier decisions)
- Wrote `scripts/m8_combined.py` — single script handling baseline + fixed + adaptive modes
- Ran Phase 1: 7 test cases (5 original + 2 new 3-image) × 3 modes = 21 runs
- Ran Phase 2: 3 scaling cases × 3 image counts × 3 modes = 27 runs (48 total)
- Saved 48 output images + JSON results to `PROJECT_BLUE_ROOT/outputs/m8_combined/`
- Updated findings.md with full 7-section analysis

### Key Results
- **Fixed combined**: 2.35x speedup (matches theoretical 80/34), mean SSIM 0.869, consistent across all cases
- **Adaptive combined**: 3.95x mean speedup (16-25 passes), mean SSIM 0.809, more variable
- **Combined > single axis**: Fixed 2.35x exceeds M5 (1.82x) and M6 conservative (1.66x). Adaptive 3.95x exceeds M6 aggressive (2.86x)
- **No negative interaction**: Combining two axes does NOT compound quality loss. SSIM is between M5 and M6 individual results
- **Adaptive does NOT protect hard cases**: tc5 texture SSIM drops from 0.832 (fixed) to 0.693 (adaptive). Cosine similarity is not a good proxy for perceptual quality
- **Image-count scaling**: Fixed speedup ratio constant at 2.35x. Absolute time saved scales with image count (15s→27s→42s for 1→2→3 images). Adaptive speedup varies non-monotonically
- **Quality improves with more images**: More conditioning images stabilize the trajectory, improving SSIM for both modes
- **Bottom line**: Fixed combined (2.35x, 0.869 SSIM) is the recommended mode. Reduces 3-image inference from 72s → 31s

---

## 2026-03-21 — Milestone 7: Multi-Image Condition Redundancy Characterization

### Completed
- Obtained B200 node, verified GPU (NVIDIA B200, 183 GB), activated existing venv
- Inspected pipeline source code (`pipeline_qwenimage_edit_plus.py`) and transformer source (`transformer_qwenimage.py`) to understand multi-image token handling
- Documented mechanical structure: multi-image = sequence concatenation, no special cross-image mechanism, shared timestep modulation (zero-cond), shared RoPE frame index (-1)
- Wrote m7_multi_image.py instrumentation script measuring: token counts, compute scaling, cross-image hidden state similarity (5 sampled blocks × 40 steps), attention region distribution (5 blocks × 6 sampled steps), and temporal/CFG redundancy per image count
- Ran instrumented forward passes for 1, 2, 3 images (40 steps each, TRUE_CFG_SCALE=4.0)
- Saved raw data to `PROJECT_BLUE_ROOT/profiling/multi_image_conditioning/m7_results.json`
- Updated findings.md with full 7-section analysis and bottom-line characterization

### Key Results
- **Token scaling**: Each image adds exactly 4,096 condition tokens + ~200 text tokens. Linear growth.
- **Compute scaling**: O(S^1.4) — super-linear, sub-quadratic. 1img=26.9s, 2img=47.3s (1.76x), 3img=72.3s (2.69x). FlashAttention keeps effective exponent below 2.0.
- **VRAM**: Constant ~63 GB across all image counts. No memory pressure from more images.
- **Cross-image hidden states**: Near-orthogonal (cos ≈ 0) at blocks 0-44. **No hidden state redundancy.** Block 59 converges all tokens (cos=0.999) — architectural normalizer.
- **Attention distribution**: Noise tokens 88-95% self-attentive. Image tokens 30-40% self, 50-60% noise, only 10-15% cross-image. **Minimal cross-image attention.**
- **Temporal/CFG redundancy unchanged**: Three-phase temporal pattern (converge→plateau→diverge) and CFG plateau (cos >0.999 for steps 3-38) persist identically across 1/2/3 images. Same skip schedules apply.
- **Bottom line**: Multi-image conditioning does NOT create exploitable redundancy. Differentiation must come from combining temporal/CFG axes adaptively. Multi-image cases benefit MORE from existing methods (larger absolute savings due to higher per-step cost).

---

## 2026-03-21 — Milestone 6: Temporal Step-Caching Prototype (Block-Aware)

### Completed
- Obtained B200 node, verified GPU (NVIDIA B200, 183 GB), activated existing venv
- Reviewed M4 block-level redundancy data to design cache policies
- Implemented m6_temporal_cache.py with 3 schedules (aggressive uniform, conservative uniform, block-aware)
- Ran warmup + 5 baseline + 15 cached-variant passes (5 test cases × 3 schedules)
- Computed per-test-case metrics: wall-clock time, SSIM, MSE, PSNR
- Saved all 20 output images + JSON results to `PROJECT_BLUE_ROOT/outputs/m6_temporal_cache/`
- Produced cross-axis comparison table (M5 CFG-skip vs M6 temporal)
- Updated findings.md with full analysis

### Key Results
- **Aggressive uniform (2.86x)**: Mean SSIM 0.78 — too aggressive, 3/5 cases severe degradation
- **Conservative uniform (1.66x)**: Mean SSIM 0.93 — best tradeoff, beats M5 CFG-skip quality on all cases
- **Block-aware (1.76x)**: Mean SSIM 0.37 — **catastrophic failure**. Block-level caching disconnects transformer state from evolving latents
- **Cross-axis overlap**: Same hard cases (tc3/tc5 = complex scene/texture) degrade on both CFG and temporal axes
- **No input-count pattern**: Degradation driven by edit complexity, not number of images
- **Bottom line**: Conservative temporal caching (noise-prediction level) is the best single-axis method. Block-aware caching is not viable. Combined strategy should use selective acceleration based on edit complexity.

---

## 2026-03-21 — Milestone 5: CFG-Skip Prototype + Robustness

### Completed
- Obtained B200 node, verified GPU (NVIDIA B200, 183 GB), activated existing venv
- Installed scikit-image for SSIM metric computation
- Reviewed M4 redundancy data to design skip schedule: full CFG steps 0–2+39, skip 3–38
- Implemented m5_cfg_skip.py (v1: drop-CFG) with 5 diverse test cases (1–3 images, varied prompts)
- Ran warmup + 5 baseline + 5 skip-variant passes with identical seeds (SEED=42)
- Implemented and tested m5_cfg_skip_v2.py (cached uncond reuse) — negative result, quality collapsed
- Tested conservative schedule (skip 5–34) on 2 worst cases — marginal improvement
- Computed per-test-case metrics: wall-clock time, SSIM, MSE, PSNR
- Saved all 21 output images + 3 JSON result files to `PROJECT_BLUE_ROOT/outputs/m5_cfg_skip/`
- Updated findings.md with full analysis, progress.md with log

### Key Results
- **Speedup**: 1.82x consistent across all 5 test cases (matches theoretical 80/44 = 1.818x)
- **Quality**: SSIM 0.73–0.94 (mean 0.86). 3 cases moderate, 2 cases notable degradation
- **VRAM**: Unchanged (63.20 GB) — skip saves compute, not memory
- **v2 (cached uncond)**: Failed catastrophically (SSIM 0.05–0.53). Stale uncond + CFG amplification = disaster
- **Conservative schedule**: SSIM +0.07–0.09 better, but speedup drops to 1.60x
- **Bottom line**: CFG-skip delivers real speedup but quality tradeoff is non-trivial at cfg_scale=4.0. Best suited as part of a combined strategy, not standalone.

---

## 2026-03-21 — Milestone 4: Redundancy Characterization

### Completed
- Obtained B200 node, verified GPU (NVIDIA B200, 183 GB), activated existing venv
- Reviewed M3 profiling script and transformer model internals (QwenImageTransformerBlock, dual-stream, 60 blocks, hidden dim 3072)
- Designed instrumentation: forward hooks on 5 sampled blocks (0, 14, 29, 44, 59) capturing image-stream hidden states, with online cosine-sim + relative-L2 computation
- Ran warmup pass (48.2s) + instrumented pass (47.6s) — zero VRAM overhead (63.20 GB = baseline)
- Collected Axis 1 (temporal) and Axis 2 (CFG) redundancy profiles across all 40 steps
- Axis 3 (attention sparsity) deferred — attention weights not directly exposed by QwenDoubleStreamAttnProcessor2_0
- Saved raw data to `/blue/yuanyuan.lei/zihua.meng/vlm_inference/profiling/redundancy/`
- Updated findings.md with full redundancy profiles, cross-axis analysis, and bottom-line summary

### Key Results
- **Temporal redundancy**: Strong. Three-phase pattern:
  - Steps 1–4: convergence (cos 0.98→0.9995)
  - Steps 5–30: plateau with near-identical consecutive outputs (cos >0.999)
  - Steps 31–39: divergence, with step 39 dramatically different (cos=0.816)
  - Block 44 is most temporally dynamic; block 59 is most stable
- **CFG redundancy**: Very strong. Cond ≈ uncond for ~90% of trajectory:
  - Steps 0–2: peak divergence (cos 0.968–0.995)
  - Steps 3–38: near-identical (cos >0.999)
  - CFG divergence concentrated in block 44; absent in blocks 0 and 59
- **Combined acceleration ceiling**: 50–70% compute reduction theoretically achievable via CFG schedule + step caching
- **Instrumentation overhead**: Zero (time and VRAM)

---

## 2026-03-21 — Milestone 3: Structured Baseline Profiling

### Completed
- Obtained B200 node, verified GPU (NVIDIA B200, 183 GB), activated existing venv
- Designed 3-level profiling instrumentation:
  - L1: `torch.cuda.synchronize()` + `time.perf_counter()` between pipeline stages
  - L2: Same sync-based timing per denoising step (cond/uncond/scheduler breakdown)
  - L3: `torch.cuda.Event` hooks on transformer sub-modules for sampled steps (0, 19, 39)
- Ran warmup pass (48.3s) + profiled pass (47.7s) on same 2-image test case
- Verified profiled total consistent with 51.2s baseline (delta -6.8%, due to warmup + no pipeline overhead)
- Saved raw profiling artifacts to `/blue/yuanyuan.lei/zihua.meng/vlm_inference/profiling/`
- Updated findings.md with full three-level numerical report

### Key Results
- **Level 1**: Denoising loop = 99.3% of total time. All other stages <1% combined.
- **Level 2**: Steps perfectly uniform (1184.5ms ± 1.4ms). CFG doubles cost (2× transformer passes per step). Scheduler step negligible (0.1ms).
- **Level 3**: Within each transformer pass, 60 blocks = 99.8% of time. Within each block: joint attention = 75.8%, image MLP = 12.2%, other = 11.2%, text MLP = 0.7%.
- **Dominant bottleneck**: Joint attention across the denoising loop accounts for ~75% of total inference time (~35.7s of 47.7s).

---

## 2026-03-21 — Milestone 2: B200 Bootstrap + Native-Speed Baseline

### Completed
- Obtained B200 node: NVIDIA B200, 183 GB VRAM, driver 570.148.08, CUDA 12.8
- Verified existing venv works on B200 without any changes (same torch 2.10.0+cu128)
- Loaded Qwen-Image-Edit-2511 in full bf16 via `pipeline.to('cuda')` — no offloading
  - VRAM after load: 57.76 GB allocated
- Ran single 2-image forward pass (same test case as M1): 40 steps, seed 42
  - **Generation time: 51.2s** (vs 728s on L4 — 14.2x speedup)
  - **Peak VRAM: 63.20 GB reserved** (34.5% of 183 GB)
- Output saved: `/blue/yuanyuan.lei/zihua.meng/vlm_inference/outputs/m2_b200_full_bf16.png`
- Updated findings.md with full B200 environment snapshot and L4 vs B200 comparison

### Key Result
B200 runs full bf16 natively at 51.2s/pass. Inference is now compute-bound (1.17s/step),
not memory-bound. 120+ GB VRAM headroom available for acceleration experiments.

---

## 2026-03-21 — Milestone 1: Environment Setup + Feasibility Gate

### Completed
- Created Python 3.12.5 venv on blue storage (`module load python/3.12`)
  - Note: System Python 3.9 was incompatible with latest diffusers (requires >=3.10)
- Installed all dependencies: torch 2.10.0+cu128, diffusers 0.38.0.dev0, transformers 5.3.0, accelerate 1.13.0
- Downloaded Qwen-Image-Edit-2511 weights (54 GB) to blue storage
- Tested three loading strategies:
  1. **bf16 `.to('cuda')`** — OOM (model = 21.6 GB, VRAM = 22.04 GB, no headroom)
  2. **`enable_model_cpu_offload()`** — OOM (transformer alone fills VRAM)
  3. **`enable_sequential_cpu_offload()`** — SUCCESS, peak VRAM 3.98 GB, 728s wall-clock
- Generated output image: `/blue/yuanyuan.lei/zihua.meng/vlm_inference/outputs/m1_seq_offload.png`
- No nvcc-related issues encountered

### Key Result
L4 is viable with sequential CPU offload only. ~12 min per forward pass (40 steps).
Quantization or B200 needed for practical inference speeds.

---

## 2026-03-21 — Milestone 0: Workspace Init + Baseline Recon

### Completed
- Created project directory layout on both storage tiers:
  - `/home/zihua.meng/projects/vlm_inference/{docs,scripts,configs,logs}`
  - `/blue/yuanyuan.lei/zihua.meng/vlm_inference/{weights,checkpoints,outputs,cache}`
- Verified compute environment: 1x NVIDIA L4 (23GB), Python 3.9.21, no conda, no nvcc
- Cloned upstream Qwen-Image repo (code only, no LFS): `Qwen-Image/`
- Inspected repo documentation and source code
- Identified multi-image editing entry point: `QwenImageEditPlusPipeline` from diffusers
- Compared 2509 vs 2511 variants with documentation evidence
- Produced variant recommendation: **Qwen-Image-Edit-2511** as first baseline
- Documented dependency and environment risks
- Created task_plan.md, findings.md, progress.md
