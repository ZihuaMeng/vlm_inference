# Task Plan

## Milestone 0: Workspace Init + Baseline Recon — COMPLETED

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
