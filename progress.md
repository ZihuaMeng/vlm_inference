# Progress Log

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
