# Task Plan

## FLUX Track — F1: Baseline, Profiling, and Temporal Transfer Test — COMPLETED

Completed 2026-03-28. See findings.md "FLUX Track — F1" section for full results.

### Answers to Scientific Questions
1. **Baseline performance?** 9.17s wall-clock (28 steps, single pass). 5.2x faster than Qwen baseline. Peak VRAM 37.52 GB.
2. **Multi-image support?** No. Pipeline rejects multi-image input. Single reference image only.
3. **Profiling structure?** Denoising loop = 97.8%. Steps perfectly uniform (321.3ms ± 0.2ms). Attention = 73-75% of block time. Dual/single block cost ratio = 1.08x (not 2.4x as predicted).
4. **Temporal plateau?** CONFIRMED. Steps 4-23 (74% of trajectory). Same three-phase pattern as Qwen. Milder divergence (cos=0.981 vs Qwen's 0.816).
5. **Does temporal caching transfer?** YES. Conservative: 2.15x speedup, 0.883 SSIM. Aggressive: 3.50x, 0.800 SSIM. Same content-dependent quality pattern.
6. **Transfer verdict?** Temporal plateau is a property of rectified flow matching, not model-specific. Caching transfers directly.

### Tasks
1. [x] Bootstrap: install sentencepiece, download FLUX Kontext-dev weights (54 GB), verify B200
2. [x] Phase 1: Baseline timing (9.17s, 37.52 GB VRAM, multi-image test → fails)
3. [x] Phase 2: Three-level profiling (pipeline stages, per-step dynamics, module-level breakdown)
4. [x] Phase 3: Temporal redundancy characterization (plateau steps 4-23, 74% of trajectory)
5. [x] Phase 4: Temporal caching prototype (conservative 2.15x/0.883, aggressive 3.50x/0.800)
6. [x] Update findings.md, progress.md, task_plan.md with full results

---

## FLUX Track — F0: Kontext-dev Reconnaissance — COMPLETED

Completed 2026-03-28. See findings.md "FLUX Track — F0" section for full results.

### Scientific Questions Answered
1. **Architecture?** 12B rectified flow transformer. 19 dual-stream + 38 single-stream = 57 blocks. Hidden dim 3072. CLIP-L + T5-XXL text encoders. Guidance-distilled (single pass).
2. **Multi-image conditioning?** Sequence concatenation with per-image time-step offsets in 3D RoPE. Current release: single image only. Multi-image: architectural but unreleased.
3. **Guidance vs Qwen?** Guidance-distilled (1 pass/step) vs Qwen's true CFG (2 passes/step). CFG-skip (M5) is NOT APPLICABLE — the unconditional pass doesn't exist.
4. **Default steps/scheduler?** 28 steps, FlowMatchEulerDiscreteScheduler (same as Qwen). Velocity prediction with Euler integration.
5. **VRAM?** ~34.4 GB weights in bf16, ~45-55 GB peak estimated. Fits on B200 (183 GB). Does NOT fit on L4 (23 GB) without offloading.
6. **Dependencies?** Current venv compatible. Only missing: `sentencepiece` for T5 tokenizer.
7. **Transfer assessment?** Temporal plateau (M4) and temporal caching (M6): LIKELY. CFG-skip (M5): NOT APPLICABLE. Block uniformity (M3): does NOT transfer. Multi-image scaling (M7): likely but untestable.

### Tasks
1. [x] Read FluxKontextPipeline source code
2. [x] Read FluxTransformer2DModel source code (19 dual + 38 single blocks)
3. [x] Read FlowMatchEulerDiscreteScheduler source code
4. [x] Fetch HuggingFace model card and arXiv paper details
5. [x] Produce structured Qwen vs Kontext comparison table
6. [x] Assess transfer likelihood for each Qwen finding (M3-M9)
7. [x] Check dependency compatibility and estimate VRAM
8. [x] Create scripts/flux/ directory with architecture notes
9. [x] Write GPU validation plan for F1
10. [x] Update findings.md, progress.md, task_plan.md

---

## Qwen Track — Milestone 0: Workspace Init + Baseline Recon — COMPLETED

Completed 2026-03-21. See findings.md for full results.
- Created project layout on both storage tiers
- Verified compute environment (1x L4, 23GB, CUDA 12.8, Python 3.9.21, no nvcc)
- Cloned Qwen-Image repo, inspected architecture
- Recommended **Qwen-Image-Edit-2511** as first baseline

## Milestone 1: Environment Setup + Feasibility Gate — COMPLETED

Completed 2026-03-21. See findings.md for full results.

### Answers to Scientific Questions
1. **Can 2511 load and run on L4?** Yes, but only with `enable_sequential_cpu_offload()`.
2. **Minimum viable config?** Sequential CPU offload (bf16 weights, layer-by-layer GPU execution). Model-level offload is insufficient — the transformer alone exceeds VRAM.
3. **Peak VRAM?** 3,980 MiB (~3.9 GB) during sequential offload inference.
4. **nvcc issues?** None. Pre-compiled PyTorch CUDA wheels work without nvcc.

### Tasks
1. [x] Update task_plan.md with Milestone 1 objectives
2. [x] Create Python 3.12 venv, install torch 2.10+cu128, diffusers 0.38.0.dev0, transformers 5.3.0, accelerate 1.13.0
3. [x] Download 2511 weights (54 GB) to blue storage
4. [x] Tested 3 strategies: bf16 OOM, model_cpu_offload OOM, sequential_cpu_offload SUCCESS (728s)
5. [x] Peak VRAM recorded, output image saved, findings documented

## Milestone 2: B200 Bootstrap + Native-Speed Baseline — COMPLETED

Completed 2026-03-21. See findings.md for full results.

### Answers to Scientific Questions
1. **Does the existing venv work on B200?** Yes, without any changes.
2. **Does the model load in full bf16 without offloading?** Yes. 57.76 GB allocated after load.
3. **Native-speed wall-clock time?** 51.2s for 40 steps (1.17s/step). 14.2x faster than L4 seq-offload.
4. **Peak VRAM?** 63.20 GB reserved (34.5% of 183 GB).
5. **Output visually comparable?** Yes. Red and blue objects on light surface, coherent generation.

### Tasks
1. [x] Verify B200 GPU and environment (191.5 GB VRAM, driver 570.148.08, CUDA 12.8)
2. [x] Confirm existing venv works on B200 without changes
3. [x] Load model in full bf16 via pipeline.to('cuda') — no offloading
4. [x] Run single 2-image forward pass (same test case as M1)
5. [x] Capture peak VRAM and wall-clock time
6. [x] Save output image, update findings.md and progress.md

## Milestone 3: Structured Baseline Profiling — COMPLETED

Completed 2026-03-21. See findings.md for full results.

### Answers to Scientific Questions
1. **Level 1 — Pipeline stages**: Denoising loop = 99.3%. Text encoding 0.3%, VAE decode 0.2%, image preprocess 0.1%, VAE encode 0.1%. Non-denoising stages are negligible.
2. **Level 2 — Denoising step dynamics**: Perfectly uniform. Mean 1184.5ms ± 1.4ms. No variation across steps. CFG doubles cost (592ms cond + 592ms uncond per step).
3. **Level 3 — Module-level within a step**: Joint attention = 75.8% of block time. Image MLP = 12.2%. Other (mod/norm/gate) = 11.2%. Text MLP = 0.7%. All 60 blocks identical. Dominant cost: joint attention (~75% of total inference time).

### Tasks
1. [x] Update task_plan.md with Milestone 3 objectives
2. [x] Write 3-level profiling script (pipeline stages, per-step, module-level)
3. [x] Run warmup + profiled pass on B200, verify consistency with 51.2s baseline
4. [x] Analyze results, update findings.md and progress.md with full report

## Milestone 4: Redundancy Characterization — COMPLETED

Completed 2026-03-21. See findings.md for full results.

### Answers to Scientific Questions
1. **Temporal redundancy?** Yes, strong. Steps 5–30 show cos >0.999 (near-identical consecutive outputs). Three-phase pattern: convergence → plateau → final-step divergence.
2. **Do representations stabilize?** Yes. Peak similarity at steps 20–29 (cos >0.9999). But representations diverge again in final steps, with step 39 dramatically different (cos=0.816).
3. **CFG redundancy?** Yes, very strong. After step 2, cond ≈ uncond (cos >0.999). The unconditional pass is redundant for ~90% of the trajectory.
4. **Block-level pattern?** Block 44 is most dynamic on both axes. Block 59 is most stable (normalizer). CFG divergence is bell-shaped across blocks, peaking at block 44.
5. **Instrumentation overhead?** Zero (0.00 GB VRAM, no measurable time overhead).

### Tasks
1. [x] Update task_plan.md with Milestone 4 objectives
2. [x] Write m4_redundancy.py instrumentation script
3. [x] Run warmup + instrumented pass on B200, collect metrics
4. [x] Analyze results, update findings.md and progress.md with redundancy profiles

## Milestone 5: CFG-Skip Prototype + Robustness — COMPLETED

Completed 2026-03-21. See findings.md for full results.

### Answers to Scientific Questions
1. **What speedup?** 1.82x consistent across all 5 test cases. Matches theoretical 80/44 = 1.818x.
2. **Does it degrade quality?** Yes. SSIM 0.73–0.94 (mean 0.86). 3/5 moderate, 2/5 notable degradation.
3. **Consistent across test cases?** Speedup: perfectly consistent. Quality: varies — complex/textured prompts degrade more.
4. **Simplest effective schedule?** v1 drop-CFG (steps 3–38 skip) is simplest. v2 cached-uncond reuse failed catastrophically. Conservative schedule (steps 5–34 skip) slightly better quality, lower speedup (1.60x).

### Tasks
1. [x] Update task_plan.md with Milestone 5 objectives
2. [x] Activate venv, verify environment, install scikit-image for SSIM
3. [x] Write m5_cfg_skip.py (v1 drop-CFG) + m5_cfg_skip_v2.py (cached uncond reuse)
4. [x] Run experiments: v1 aggressive (5 cases), v2 (5 cases), v1 conservative (2 worst cases)
5. [x] Analyze results, update findings.md and progress.md

## Milestone 6: Temporal Step-Caching Prototype (Block-Aware) — COMPLETED

Started 2026-03-21. Temporal axis acceleration: cache transformer outputs across denoising steps.

### Scientific Questions
1. What speedup does temporal step-caching achieve?
2. Does block-aware caching (recompute dynamic blocks, cache stable) outperform uniform caching?
3. What is the quality cost, and which test cases degrade?
4. Do quality failure modes overlap with M5's CFG-skip failures (same hard cases)?
5. Is degradation correlated with input count or edit complexity?

### Design Decisions (from M4 block-level data)
- **Block 44**: Most dynamic (temporal cos ~0.973 in plateau). Always recompute.
- **Block 59**: Most stable (cos ~0.9999). Safe to cache.
- **Blocks 0, 14**: Very stable (cos >0.9996). Safe to cache.
- **Block 29**: Moderately stable (cos ~0.998). Safe to cache.
- **Dynamic zone**: Blocks 40-59 always recomputed (conservative — includes block 44 and all downstream so fresh output reaches norm_out/proj_out)
- **Cached zone**: Blocks 0-39 (all measured blocks in this range have cos >0.996)

### Three Schedules
- **A: Aggressive uniform** — Cache steps 5-30 (skip entire transformer, reuse prev noise pred). 28 passes vs 80 baseline → ~2.86x theoretical.
- **B: Conservative uniform** — Cache steps 10-25 (narrower window). 48 passes vs 80 → ~1.67x theoretical.
- **C: Block-aware** — Cache steps 5-30, but recompute blocks 40-59 (dynamic zone). Cache blocks 0-39. ~1.7-1.8x theoretical.

### Answers to Scientific Questions
1. **What speedup?** Aggressive uniform: 2.86x. Conservative uniform: 1.66x. Block-aware: 1.76x.
2. **Does block-aware outperform uniform?** No. Block-aware FAILED (SSIM 0.37). Uniform conservative is the best method (SSIM 0.93).
3. **Quality cost?** Conservative uniform: SSIM 0.89-0.97 (mean 0.93). Aggressive: SSIM 0.65-0.92 (mean 0.78).
4. **Overlap with M5 failures?** Yes. tc3 (complex scene) and tc5 (texture) are hardest on both axes.
5. **Input count correlation?** No. Degradation driven by edit complexity, not image count.

### Tasks
1. [x] Update task_plan.md with M6 objectives
2. [x] Write m6_temporal_cache.py
3. [x] Run experiments: baseline + 3 schedules × 5 test cases
4. [x] Analyze results, cross-axis comparison with M5
5. [x] Update findings.md and progress.md

## Milestone 7: Multi-Image Condition Redundancy Characterization — COMPLETED

Started 2026-03-21. Determine whether multi-image conditioning introduces structured redundancy absent in single-image conditioning.

### Scientific Questions
1. How are multiple image conditions represented internally? (Concatenated? Independent? Where does cross-image interaction occur?)
2. How does compute cost scale with input image count? (Linear? Super-linear?)
3. Is there quantifiable redundancy in how the model processes multi-image conditions? (Attention convergence? Hidden state similarity?)
4. Does the temporal or CFG redundancy structure change with more images?
5. Does multi-image conditioning create exploitable redundancy, and if so, what kind?

### Mechanical Structure (from code reading)
- **Text encoder**: Each image gets `"Picture N: <|vision_start|><|image_pad|><|vision_end|>"` in prompt. Qwen2.5-VL processes all images jointly.
- **VAE path**: Each image individually VAE-encoded, packed to 2x2 patch tokens, then all image latents **concatenated along sequence dimension** (dim=1).
- **Transformer input**: `hidden_states = [noise_latents, img1_latents, img2_latents, ...]` (single concatenated sequence).
- **Timestep modulation** (`zero_cond_t`): noise tokens get full timestep modulation (index=0), ALL condition image tokens get zero-timestep modulation (index=1). No per-image differentiation.
- **RoPE**: Output noise tokens get positive frame index. ALL condition images get negative frame index (-1). Condition images share frame position, differ only in spatial (h,w) positions.
- **Joint attention**: Text and image streams separately projected to Q,K,V, then concatenated `[text, image]` for full joint attention. Attention cost is O(n²) in total sequence length.
- **Output**: Only the first `noise_latents.size(1)` tokens of transformer output used as noise prediction.

### Key Architectural Insight
Multi-image conditioning is pure sequence concatenation with no special cross-image mechanism. Adding N images linearly increases sequence length, causing quadratic attention cost scaling. Cross-image interaction is entirely implicit through the joint attention mechanism.

### Answers to Scientific Questions
1. **How are multi-image conditions represented?** Pure sequence concatenation. Each image → 4,096 tokens, all concatenated along dim=1. Shared frame index (-1), shared zero-timestep modulation. No special cross-image mechanism.
2. **How does compute scale?** O(S^1.4): 1img=26.9s, 2img=47.3s (1.76x), 3img=72.3s (2.69x). Super-linear due to quadratic attention, sub-quadratic due to FlashAttention + linear MLP cost.
3. **Is there cross-image redundancy?** No. Hidden states are near-orthogonal (cos ≈ 0) at blocks 0-44. Cross-image attention is only 10-15%. Each image carries genuinely distinct information.
4. **Does temporal/CFG redundancy change with image count?** No. Same three-phase temporal pattern, same CFG plateau. Skip schedules transfer directly.
5. **Bottom line?** No exploitable multi-image redundancy exists. Differentiation must come from adaptive temporal/CFG combination.

### Tasks
1. [x] Inspect pipeline and transformer code for multi-image handling
2. [x] Write m7_multi_image.py instrumentation script
3. [x] Run instrumented passes for 1, 2, 3 images
4. [x] Analyze results: compute scaling, attention, cross-image redundancy, temporal/CFG interaction
5. [x] Update findings.md and progress.md with full analysis

## Milestone 8: Adaptive Combined Acceleration Prototype — COMPLETED

Started 2026-03-21. Combined temporal caching + CFG-skip with adaptive quality-aware control.

### Scientific Questions
1. Does combining temporal caching + CFG-skip achieve higher speedup than either axis alone?
2. Does an adaptive policy (modulating skip/cache decisions based on a runtime signal) improve quality robustness compared to fixed schedules?
3. How does the combined speedup scale across 1, 2, and 3-image inputs?
4. What is the quality/speedup Pareto frontier of the combined method?
5. Can the adaptive policy preserve quality on the hard cases (complex/textured edits)?

### Design
- **Fixed combined**: M6 conservative temporal (cache steps 10-25) + M5 conservative CFG-skip (skip steps 5-9, 26-34). Total: 34/80 passes.
- **Adaptive combined**: Per-step cosine similarity signal. Three tiers: temporal cache (cos>0.999, reuse prev), CFG-skip (cos>0.995, cond only), full compute (low similarity). Max consecutive cache: 4 steps.

### Answers to Scientific Questions
1. **Combined > single axis?** Yes. Fixed combined 2.35x > M5 1.82x and M6 conservative 1.66x individually. Adaptive 3.95x > M6 aggressive 2.86x.
2. **Adaptive improve quality?** No. Adaptive is MORE aggressive and degrades quality MORE on hard cases (tc5 SSIM: fixed 0.832 vs adaptive 0.693). Fixed mode has lower SSIM variance (0.085 vs 0.093).
3. **Image-count scaling?** Fixed: constant 2.35x regardless of image count. Adaptive: variable (3.19-4.98x). Absolute time saved scales with images (15s→27s→42s for 1→2→3 imgs).
4. **Pareto frontier?** Two operating points: quality-first (fixed: 2.35x, 0.869 SSIM) and speed-first (adaptive: 3.95x, 0.809 SSIM).
5. **Adaptive protect hard cases?** No. Cosine similarity in latent space does not correlate with perceptual quality for complex/textured content.

### Tasks
1. [x] Review M5/M6 results, M4 redundancy data, design combined schedules
2. [x] Write m8_combined.py with fixed and adaptive modes
3. [x] Run Phase 1: 7 test cases × 3 modes (21 runs)
4. [x] Run Phase 2: 3 scaling cases × 3 image counts × 3 modes (27 runs)
5. [x] Analyze results, update findings.md and progress.md

## Milestone 9: Broad Validation + Ablation Study — COMPLETED

Started 2026-03-21. Validate fixed combined strategy on diverse 18-case test suite. Produce ablation study and failure characterization.

### Answers to Scientific Questions
1. **Does 2.35x hold?** Yes. 2.353x ± 0.001 across all 18 cases. Perfectly consistent — independent of content, difficulty, or image count.
2. **Does SSIM ~0.87 hold?** No. Mean SSIM drops to 0.836 (from 0.869) with more diverse inputs. The original 7 cases under-represented hard content types.
3. **Quality distribution?** SSIM > 0.90: 39% (7/18). SSIM > 0.85: 61% (11/18). SSIM < 0.80: 33% (6/18). Bimodal: robust cluster (0.83–0.96) and fragile cluster (0.64–0.80).
4. **Component contribution?** Temporal caching is the better single axis (1.667x, 0.885 SSIM vs CFG-skip 1.600x, 0.851). Combining does NOT compound quality loss — overlapping vulnerability means actual combined SSIM (0.836) far exceeds independent prediction (0.753).
5. **Absolute time scaling?** Savings proportional to image count: 15s (1-img), 27s (2-img), 42s (3-img). 2-image has best quality (0.896 SSIM), likely due to difficulty confound.
6. **What makes a case hard?** Fine-grained visual detail: complex scenes (0.637), style transfer (0.642), material texture (0.665), nature scenes (0.687). Structural/semantic content is robust (objects: 0.947, portraits: 0.911, arrangements: 0.955). Difficulty rating is the strongest predictor: easy=0.943, medium=0.886, hard=0.753.

### Design
- **18 test cases**: 6×1-image, 6×2-image, 6×3-image. Covers: simple objects, landscape, style transfer, text/logo, material/texture, portrait, object placement, scene composition, texture detail, identity-preserving, object replacement, background change, arrangement, multi-portrait, nature scene, interior design, complex scene, abstract. Includes original 5 M5/M6 cases + 2 M8 cases + 11 new cases.
- **4 configurations per case**:
  a. Baseline: full 40-step, full CFG (80 passes)
  b. CFG-skip only: M5 conservative (skip uncond steps 5-34, 50 passes, 1.60x theoretical)
  c. Temporal cache only: M6 conservative (cache steps 10-25, 48 passes, 1.67x theoretical)
  d. Fixed combined: M8 (temporal 10-25 + CFG-skip 5-9,26-34, 34 passes, 2.35x theoretical)
- **Metrics**: Wall-clock time, speedup, SSIM, LPIPS (perceptual distance)
- **Identical seeds** (SEED=42) for all paired comparisons

### Tasks
1. [x] Bootstrap environment: B200 GPU, venv, install LPIPS
2. [x] Write m9_validation.py evaluation script
3. [x] Run 18 cases × 4 configs = 72 inference runs (~40 min on B200)
4. [x] Analyze results: ablation, quality distribution, scaling, failure characterization
5. [x] Update findings.md and progress.md with full analysis
