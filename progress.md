# Progress Log

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
