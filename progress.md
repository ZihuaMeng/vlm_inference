# Progress Log

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
