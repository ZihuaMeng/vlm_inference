# Findings

---

# Milestone 4: Redundancy Characterization

## Bottom-Line Summary

**Both axes show strong, exploitable redundancy.** This model is a strong candidate for redundancy-based dynamic compute acceleration.

| Axis | Exploitable? | Strength | Where in Trajectory |
|------|-------------|----------|-------------------|
| **Temporal** (step-to-step) | **Yes** | Strong (cos >0.999 for steps 4–30) | Mid-trajectory plateau; first 3 steps and final step are dynamic |
| **CFG** (cond vs uncond) | **Yes** | Very strong (cos >0.999 for steps 3–38) | After initial divergence (steps 0–2), cond ≈ uncond for 90% of trajectory |

## 1. Instrumentation Design

- **Representations compared**:
  - Transformer output (noise prediction), sliced to noise-latent tokens only (4096 tokens)
  - Image-stream hidden states from 5 sampled blocks (0, 14, 29, 44, 59)
- **Metrics**:
  - Cosine similarity: per-token mean (token-wise cos sim averaged across the 4096 noise-latent tokens)
  - Relative L2 distance: ||a - b|| / ||a|| (global norm ratio)
- **Memory strategy**: Online comparison — store only previous step's conditional outputs (~6 tensors, ~150 MB theoretical). Actual measured VRAM overhead: **0.00 GB** (within measurement noise of the 63.20 GB baseline).
- **Warmup**: Full un-instrumented pass (48.2s) before the profiled run.

## 2. Instrumentation Overhead

| Metric | Value |
|--------|-------|
| **Instrumented loop time** | 47.6s |
| **M3 profiled baseline** | 47.7s |
| **Peak VRAM** | 63.20 GB |
| **VRAM overhead** | +0.00 GB |

Instrumentation is free — the forward-hook-based block capture and online similarity computation add no measurable overhead in time or memory.

## 3. Axis 1: Temporal Redundancy (Step-to-Step Similarity)

### Transformer Output — Full Trajectory

| Step | Cosine Sim | Relative L2 |
|------|-----------|-------------|
| 1 | 0.980547 | 0.201351 |
| 2 | 0.995275 | 0.104198 |
| 3 | 0.996028 | 0.087355 |
| 4 | 0.999529 | 0.031029 |
| 5 | 0.999581 | 0.028747 |
| 6 | 0.999609 | 0.028484 |
| 7 | 0.999734 | 0.023462 |
| 8 | 0.999790 | 0.020772 |
| 9 | 0.999824 | 0.018849 |
| 10 | 0.999827 | 0.018252 |
| 11 | 0.999819 | 0.018634 |
| 12 | 0.999871 | 0.016002 |
| 13 | 0.999893 | 0.014427 |
| 14 | 0.999885 | 0.015360 |
| 15 | 0.999876 | 0.016065 |
| 16 | 0.999883 | 0.015665 |
| 17 | 0.999906 | 0.013601 |
| 18 | 0.999900 | 0.014184 |
| 19 | 0.999860 | 0.016558 |
| 20 | 0.999924 | 0.012306 |
| 21 | 0.999927 | 0.012099 |
| 22 | 0.999924 | 0.012751 |
| 23 | 0.999937 | 0.011231 |
| 24 | 0.999928 | 0.012593 |
| 25 | 0.999929 | 0.012221 |
| 26 | 0.999931 | 0.011963 |
| 27 | 0.999934 | 0.011877 |
| 28 | 0.999913 | 0.013274 |
| 29 | 0.999915 | 0.013351 |
| 30 | 0.999897 | 0.014470 |
| 31 | 0.999871 | 0.016070 |
| 32 | 0.999842 | 0.019486 |
| 33 | 0.999751 | 0.022054 |
| 34 | 0.999710 | 0.024831 |
| 35 | 0.999536 | 0.030944 |
| 36 | 0.999162 | 0.041459 |
| 37 | 0.998125 | 0.061896 |
| 38 | 0.994734 | 0.110461 |
| 39 | 0.816439 | 0.933850 |

### Temporal Summary Statistics

| Metric | Value |
|--------|-------|
| **Overall cos sim** | min=0.816, max=0.9999, mean=0.994 |
| **First 5 steps avg cos** | 0.994192 |
| **Last 5 steps avg cos** | 0.961599 |
| **Peak similarity region** | Steps 20–29 (cos >0.9999) |

### Temporal Trajectory Pattern

Three distinct phases:

1. **Convergence phase (steps 1–4)**: Rapid increase in step-to-step similarity. Cosine sim rises from 0.98 to 0.9995. The model is making large structural changes as the noise pattern takes shape. Relative L2 drops from 0.20 to 0.03.

2. **Plateau phase (steps 5–30)**: Extremely high similarity. All steps have cos >0.9995. Consecutive outputs are near-identical — the model is making only tiny refinements. This is the primary exploitable region: **~26 steps where step-skipping or cached inference could be applied.**

3. **Divergence phase (steps 31–39)**: Similarity gradually decreases. Steps 31–38 show gentle decline (0.9998→0.995). **Step 39 is dramatically different** (cos=0.816, relative L2=0.934) — the final denoising step produces a fundamentally different output from step 38. This final step cannot be approximated.

### Block-Level Temporal Redundancy

| Block | Min Cos | Max Cos | Mean Cos | First 5 Avg | Last 5 Avg |
|-------|---------|---------|----------|-------------|------------|
| 0 (early) | 0.904 | 0.9999 | 0.997 | 0.999 | 0.979 |
| 14 (early-mid) | 0.884 | 0.9998 | 0.996 | 0.994 | 0.974 |
| 29 (mid) | 0.821 | 0.999 | 0.991 | 0.985 | 0.958 |
| 44 (late) | 0.850 | 0.979 | 0.967 | 0.949 | 0.946 |
| 59 (final) | 0.995 | 0.9999 | 0.9998 | 0.9998 | 0.999 |

**Key observation**: Block 44 is the most temporally dynamic (lowest mean similarity, 0.967), while block 59 (the final block) is the most static (0.9998). This suggests:
- The actual denoising dynamics are concentrated in mid-to-late blocks (~block 44), not at the boundary blocks
- The final block (59) acts as a stabilizer/normalizer — its output barely changes across steps even when earlier blocks are changing more
- Early blocks (0, 14) track the overall trajectory pattern closely

## 4. Axis 2: CFG Redundancy (Conditional vs Unconditional)

### Transformer Output — Full Trajectory

| Step | Cosine Sim | Relative L2 |
|------|-----------|-------------|
| 0 | 0.976447 | 0.215197 |
| 1 | 0.968130 | 0.256166 |
| 2 | 0.995404 | 0.096391 |
| 3 | 0.999001 | 0.044258 |
| 4 | 0.999527 | 0.030908 |
| 5 | 0.999714 | 0.023917 |
| 6 | 0.999646 | 0.026683 |
| 7 | 0.999621 | 0.027602 |
| 8 | 0.999707 | 0.024158 |
| 9 | 0.999815 | 0.018875 |
| 10 | 0.999666 | 0.025864 |
| 11 | 0.999866 | 0.016050 |
| 12 | 0.999880 | 0.015308 |
| 13 | 0.999825 | 0.018606 |
| 14 | 0.999820 | 0.018788 |
| 15 | 0.999868 | 0.016053 |
| 16 | 0.999871 | 0.015970 |
| 17 | 0.999887 | 0.014886 |
| 18 | 0.999892 | 0.014489 |
| 19 | 0.999873 | 0.015522 |
| 20 | 0.999910 | 0.013310 |
| 21 | 0.999921 | 0.012402 |
| 22 | 0.999923 | 0.012294 |
| 23 | 0.999918 | 0.012669 |
| 24 | 0.999935 | 0.011336 |
| 25 | 0.999932 | 0.011515 |
| 26 | 0.999937 | 0.011144 |
| 27 | 0.999938 | 0.011046 |
| 28 | 0.999940 | 0.010858 |
| 29 | 0.999938 | 0.011008 |
| 30 | 0.999948 | 0.010089 |
| 31 | 0.999945 | 0.010338 |
| 32 | 0.999944 | 0.010472 |
| 33 | 0.999948 | 0.010060 |
| 34 | 0.999944 | 0.010388 |
| 35 | 0.999935 | 0.011118 |
| 36 | 0.999928 | 0.011722 |
| 37 | 0.999918 | 0.012485 |
| 38 | 0.999874 | 0.015546 |
| 39 | 0.993371 | 0.092711 |

### CFG Summary Statistics

| Metric | Value |
|--------|-------|
| **Overall cos sim** | min=0.968, max=0.9999, mean=0.998 |
| **First 5 steps avg cos** | 0.987702 |
| **Last 5 steps avg cos** | 0.998605 |
| **Peak similarity region** | Steps 24–34 (cos >0.9999) |
| **Trend** | Cond/uncond **converge** over trajectory |

### CFG Trajectory Pattern

1. **Initial divergence (steps 0–2)**: Conditional and unconditional outputs are most different at the start. Step 1 shows peak divergence (cos=0.968, L2=0.256). This is when the prompt conditioning has the most impact.

2. **Rapid convergence (steps 3–5)**: Similarity jumps to >0.999. The cond/uncond gap collapses quickly.

3. **Near-identity plateau (steps 6–38)**: cos >0.999 throughout. The conditional and unconditional transformer outputs are essentially the same. **The unconditional pass is redundant for ~33 of 40 steps.** Since CFG doubles the compute (two transformer passes per step), eliminating the redundant unconditional pass could save ~41% of total inference time (33 uncond passes × 592ms = 19.5s out of 47.6s).

4. **Final step divergence (step 39)**: Slight drop to cos=0.993 — mild, not as dramatic as the temporal divergence.

### Block-Level CFG Redundancy

| Block | Min Cos | Max Cos | Mean Cos | First 5 Avg | Last 5 Avg |
|-------|---------|---------|----------|-------------|------------|
| 0 (early) | 0.99999 | 0.99999 | 0.99999 | 0.99999 | 0.99999 |
| 14 (early-mid) | 0.995 | 0.9995 | 0.997 | 0.995 | 0.999 |
| 29 (mid) | 0.978 | 0.998 | 0.990 | 0.981 | 0.998 |
| 44 (late) | 0.925 | 0.992 | 0.949 | 0.932 | 0.985 |
| 59 (final) | 0.9997 | 0.9999 | 0.9999 | 0.9998 | 0.9999 |

**Key observations**:
- **Block 0**: No CFG divergence at all — cond and uncond are identical at the first block's output. The text conditioning has not yet propagated into the image stream.
- **Block 44**: Maximum CFG divergence (mean cos=0.949). This is where conditional vs unconditional processing diverges most. The CFG "signal" is concentrated here.
- **Block 59**: CFG divergence disappears again. The final block reconverges cond and uncond outputs, similar to its temporal stabilization role.
- **Pattern**: CFG divergence is bell-shaped across blocks — zero at entry, peaks in late-mid blocks (~44), returns to near-zero at exit. This is consistent with the dual-stream attention gradually mixing the text conditioning into the image stream through mid-to-late blocks.

## 5. Cross-Axis Analysis

Block 44 is uniquely interesting: it shows BOTH the most temporal dynamism (lowest step-to-step similarity) AND the most CFG divergence (lowest cond-vs-uncond similarity). This block is where the denoising "action" happens — it is the most prompt-responsive and step-responsive block.

Block 59, conversely, is the most stable on both axes — acting as a normalizer/stabilizer regardless of what happens upstream.

The temporal and CFG patterns are complementary for acceleration:
- **Temporal redundancy** enables step-skipping in the plateau (steps 5–30)
- **CFG redundancy** enables unconditional pass elimination after step 2–3
- **Combined**: A strategy that skips the uncond pass after step 3 AND caches/reuses outputs during the plateau could potentially reduce compute by 50–60%

## 6. Acceleration Implications (Observation Only)

These measurements suggest several acceleration targets (not implemented in this milestone):

1. **CFG schedule**: Run full CFG (both passes) for steps 0–3, then skip the unconditional pass for steps 4–38, resume for step 39. Potential savings: ~33 uncond passes × 592ms = ~19.5s (41% of loop time).

2. **Step caching/reuse**: In the plateau region (steps 5–30), consecutive outputs are >99.9% similar. Step-skipping (run every Nth step, interpolate between) could reduce the 26-step plateau to ~8–13 steps. Potential savings: ~13–18 steps × 1184ms = ~15–21s (32–44% of loop time).

3. **Block-selective caching**: Block 59's output barely changes across steps — it could potentially be cached and reused across multiple steps without re-running its forward pass. Block 44, being the most dynamic, must always be recomputed.

4. **Combined ceiling**: If both CFG skipping and step caching are applied, the theoretical compute reduction is 50–70% of denoising loop time.

## 7. Raw Data Artifacts

- Full JSON: `/blue/yuanyuan.lei/zihua.meng/vlm_inference/profiling/redundancy/redundancy_results.json`
- Script: `/home/zihua.meng/projects/vlm_inference/scripts/m4_redundancy.py`

---

# Milestone 3: Structured Baseline Profiling

## Bottleneck Summary

The 47.7s profiled inference time is dominated by the **denoising loop** (99.3% of wall-clock). Within the loop, each of the 40 steps runs two transformer forward passes (conditional + unconditional for classifier-free guidance), each taking ~592ms. The transformer forward pass is almost entirely spent in 60 dual-stream transformer blocks (99.8%), and within each block, **joint attention is the dominant cost at 75.8%** of block time. The remaining block time is image-stream MLP (12.2%) and modulation/normalization/gating overhead (11.2%). Text-stream MLP is negligible (0.7%) due to the short text sequence. The per-step cost is perfectly uniform across all 40 steps (std=1.4ms on a 1184ms mean) — there is no early-step warmup effect or late-step divergence. Joint attention across the denoising loop accounts for approximately **75% of total wall-clock time** (~35.7s of 47.7s).

## 1. Instrumentation Approach

- **Level 1**: `torch.cuda.synchronize()` + `time.perf_counter()` between pipeline stages
- **Level 2**: Same sync-based timing around each denoising step's sub-operations
- **Level 3**: `torch.cuda.Event` hooks on transformer sub-modules, activated only for sampled steps (0, 19, 39) during the conditional pass
- **Warmup**: One full un-instrumented pipeline pass before the profiled run

## 2. Instrumentation Overhead

| Metric | Value |
|--------|-------|
| **Profiled total** | 47.7s |
| **M2 baseline** | 51.2s |
| **Delta** | -3.5s (-6.8%) |
| **Warmup pass** | 48.3s |

The profiled run is **faster** than the M2 baseline, not slower. This is because:
1. The warmup pass eliminates CUDA kernel JIT compilation and memory allocation overhead
2. The profiled pass omits pipeline framework overhead (progress bar, callback dispatch, `maybe_free_model_hooks`)
3. The M2 baseline was a cold first-run measurement

Instrumentation overhead from `torch.cuda.synchronize()` calls (5 per step × 40 steps = 200 syncs) is negligible. The profiled total is consistent with the baseline.

## 3. Level 1: Pipeline Stage Breakdown

| Stage | Time (s) | Pct of Total |
|-------|----------|-------------|
| **Denoising loop** | **47.380** | **99.3%** |
| Text encoding | 0.141 | 0.3% |
| VAE decoding | 0.076 | 0.2% |
| Image preprocessing | 0.063 | 0.1% |
| VAE encoding | 0.059 | 0.1% |
| Scheduler setup | <0.001 | 0.0% |
| **Total** | **47.719** | **100%** |

The denoising loop is overwhelmingly dominant. All other stages combined account for 0.34s (<1% of total). Text encoding (Qwen2.5-VL-7B, two passes for positive + negative prompts) is the next most expensive at 0.14s but is still negligible.

## 4. Level 2: Per-Step Denoising Dynamics

### Summary Statistics

| Metric | Value |
|--------|-------|
| **Mean step time** | 1184.5ms |
| **Std deviation** | 1.4ms |
| **Min** | 1182.1ms (step 39) |
| **Max** | 1188.6ms (step 20) |
| **Cond pass mean** | 592.2ms |
| **Uncond pass mean** | 592.0ms |
| **Scheduler step mean** | 0.1ms |

### Key Findings

1. **Steps are perfectly uniform.** The coefficient of variation is 0.12% — there is no measurable difference between early, middle, and late denoising steps.
2. **CFG exactly doubles the cost.** Each step runs two identical transformer forward passes (conditional: 592ms, unconditional: 592ms). Without CFG (true_cfg_scale=1), step time would halve to ~592ms, total denoising to ~23.7s.
3. **The scheduler step is negligible** (0.1ms per step). All time is in the transformer.

### Per-Step Timing Table

| Step | Total (ms) | Cond (ms) | Uncond (ms) | Sched (ms) |
|------|-----------|-----------|-------------|-----------|
| 0 | 1182.9 | 591.8 | 590.8 | 0.1 |
| 1 | 1186.5 | 594.1 | 592.1 | 0.1 |
| 2 | 1185.7 | 592.8 | 592.6 | 0.1 |
| 3 | 1186.6 | 594.5 | 591.8 | 0.1 |
| 4 | 1184.5 | 592.5 | 591.7 | 0.1 |
| 5 | 1184.0 | 591.9 | 591.7 | 0.1 |
| 6 | 1187.1 | 594.0 | 592.9 | 0.1 |
| 7 | 1184.3 | 592.2 | 591.9 | 0.1 |
| 8 | 1184.2 | 593.1 | 590.9 | 0.1 |
| 9 | 1184.5 | 592.4 | 591.8 | 0.1 |
| 10 | 1182.2 | 591.5 | 590.5 | 0.1 |
| 11 | 1184.3 | 592.4 | 591.7 | 0.1 |
| 12 | 1183.9 | 591.7 | 592.0 | 0.1 |
| 13 | 1183.6 | 591.9 | 591.5 | 0.1 |
| 14 | 1185.4 | 590.8 | 594.5 | 0.1 |
| 15 | 1183.9 | 591.7 | 591.9 | 0.1 |
| 16 | 1182.4 | 591.2 | 590.9 | 0.1 |
| 17 | 1182.8 | 592.0 | 590.6 | 0.1 |
| 18 | 1184.5 | 592.1 | 592.2 | 0.1 |
| 19 | 1185.9 | 592.5 | 593.1 | 0.1 |
| 20 | 1188.6 | 595.3 | 593.0 | 0.1 |
| 21 | 1184.1 | 591.9 | 591.9 | 0.1 |
| 22 | 1186.5 | 591.8 | 594.4 | 0.1 |
| 23 | 1183.9 | 591.7 | 592.0 | 0.1 |
| 24 | 1185.3 | 591.8 | 593.2 | 0.1 |
| 25 | 1185.4 | 592.6 | 592.5 | 0.1 |
| 26 | 1185.3 | 591.7 | 593.3 | 0.1 |
| 27 | 1184.1 | 591.7 | 592.1 | 0.1 |
| 28 | 1182.6 | 591.8 | 590.6 | 0.1 |
| 29 | 1185.6 | 592.0 | 593.3 | 0.1 |
| 30 | 1185.5 | 592.5 | 592.8 | 0.1 |
| 31 | 1185.3 | 593.3 | 591.7 | 0.1 |
| 32 | 1183.1 | 591.8 | 591.1 | 0.1 |
| 33 | 1185.1 | 592.2 | 592.6 | 0.1 |
| 34 | 1183.4 | 592.6 | 590.6 | 0.1 |
| 35 | 1185.2 | 590.8 | 594.2 | 0.1 |
| 36 | 1182.8 | 591.4 | 591.2 | 0.1 |
| 37 | 1183.0 | 591.3 | 591.5 | 0.1 |
| 38 | 1183.3 | 592.2 | 590.9 | 0.1 |
| 39 | 1182.1 | 591.7 | 590.2 | 0.1 |

## 5. Level 3: Module-Level Breakdown Within Transformer Forward Pass

Profiled for the conditional pass at steps 0, 19, and 39. Results are identical across steps, confirming no step-dependent variation at the module level.

### Transformer Forward: Top-Level Components

| Component | Time (ms) | Pct |
|-----------|----------|-----|
| **Transformer blocks (60)** | **585.8** | **99.8%** |
| Embeddings (timestep + RoPE) | 0.48 | 0.1% |
| Output (AdaLayerNorm + proj) | 0.28 | 0.0% |
| Input projections (img_in, txt_in) | 0.26 | 0.0% |
| **Transformer forward total** | **~587** | |

### Transformer Blocks: Per-Block Statistics

| Metric | Value |
|--------|-------|
| **Mean block time** | 9.76ms |
| **Std deviation** | 0.05ms |
| **Min** | 9.68ms |
| **Max** | 9.87ms |

All 60 blocks take essentially identical time. There is no block-position-dependent cost variation.

### Within-Block Breakdown (Sampled Blocks 0, 29, 59)

| Module | Time (ms) | Pct of Block |
|--------|----------|-------------|
| **Joint attention** | **7.44** | **75.8%** |
| Image MLP (feedforward) | 1.20 | 12.2% |
| Other (modulation, norm, gating) | 1.10 | 11.2% |
| Text MLP (feedforward) | 0.07 | 0.7% |
| **Block total** | **~9.81** | **100%** |

This breakdown is consistent across all sampled blocks (0, 29, 59) and across all sampled steps (0, 19, 39).

### Block Detail by Position

| Block | Attention (ms) | Img MLP (ms) | Txt MLP (ms) | Other (ms) | Total (ms) |
|-------|---------------|-------------|-------------|-----------|-----------|
| 0 | 7.43 | 1.20 | 0.07 | 1.17 | 9.87 |
| 29 | 7.44 | 1.22 | 0.07 | 1.07 | 9.80 |
| 59 | 7.44 | 1.18 | 0.07 | 1.06 | 9.76 |

## 6. Cost Decomposition: Full Chain

| Level | Component | Time | Pct of Total |
|-------|-----------|------|-------------|
| Pipeline | Denoising loop (40 steps) | 47.4s | 99.3% |
| Pipeline | Non-denoising (text enc, VAE, etc.) | 0.3s | 0.7% |
| Step | Per step (2 transformer passes + sched) | 1184ms | — |
| Step | Conditional transformer pass | 592ms | — |
| Step | Unconditional transformer pass | 592ms | — |
| Transformer | 60 transformer blocks | 586ms | 99.8% of pass |
| Block | Joint attention | 7.44ms | 75.8% of block |
| Block | Image MLP | 1.20ms | 12.2% of block |
| Block | Other (mod/norm/gate) | 1.10ms | 11.2% of block |
| **Dominant cost** | **Joint attention across full run** | **~35.7s** | **~75%** |

## 7. Key Takeaways for Acceleration

1. **Joint attention is the singular bottleneck** — 75% of total inference time. Any acceleration strategy must primarily target attention computation.

2. **CFG doubles cost for free** — eliminating the unconditional pass (via guidance distillation or CFG-free training) would halve inference time from ~47.7s to ~23.8s.

3. **All 40 steps are identical** — there is no "easy step" to skip or approximate. Step reduction (distillation, progressive distillation, consistency models) would provide linear speedup.

4. **All 60 blocks are identical** — no block is cheaper or more expensive than another. Block pruning or early exit would provide linear savings proportional to blocks removed.

5. **Text MLP is negligible** (0.7% of block time) due to short text sequence length. The image sequence dominates compute.

6. **Non-denoising stages are irrelevant** — text encoding, VAE encode/decode combined are <1% of total. Optimizing these would yield no perceptible speedup.

7. **VRAM headroom remains large** — peak 63.20 GB of 183 GB (34.5%). Acceleration approaches that trade memory for compute (larger batch sizes, speculative execution, caching) have room to operate.

## 8. Raw Profiling Artifacts

- Full JSON: `/blue/yuanyuan.lei/zihua.meng/vlm_inference/profiling/profiling_results.json`
- Per-step CSV: `/blue/yuanyuan.lei/zihua.meng/vlm_inference/profiling/per_step_timing.csv`
- Output image: `/blue/yuanyuan.lei/zihua.meng/vlm_inference/profiling/m3_profiled_output.png`
- Script: `/home/zihua.meng/projects/vlm_inference/scripts/m3_profile.py`

---

# Milestone 2: B200 Bootstrap + Native-Speed Baseline

## 1. B200 Environment Snapshot

| Item | Value |
|------|-------|
| **GPU** | NVIDIA B200, 1x, 183,359 MiB (191.5 GB) VRAM |
| **GPU Driver** | 570.148.08 |
| **CUDA Version** | 12.8 |
| **Persistence Mode** | On |
| **TDP** | 1000W (idle ~192W) |

## 2. Environment Compatibility

The existing venv from Milestone 1 worked on B200 **without any changes**:

| Package | Version | B200 Compatible? |
|---------|---------|-----------------|
| Python | 3.12.5 | Yes |
| torch | 2.10.0+cu128 | Yes |
| CUDA (torch) | 12.8 | Yes (matches driver) |
| diffusers | 0.38.0.dev0 | Yes |
| transformers | 5.3.0 | Yes |
| accelerate | 1.13.0 | Yes |

No environment adaptation was needed. The same cu128 PyTorch wheel works on both L4 and B200.

## 3. Model Loading: Full bf16, No Offloading

```python
pipeline = QwenImageEditPlusPipeline.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
pipeline.to("cuda")
```

- **Load time**: 42.4s
- **VRAM after load**: 57.76 GB allocated, 57.99 GB reserved
- **No offloading** — entire model (transformer + text_encoder + VAE) resident on GPU
- Only warning: `guidance_scale is passed as 1.0, but ignored` (same benign warning as L4)

## 4. Inference Performance

| Metric | Value |
|--------|-------|
| **Generation time** | 51.2s |
| **Steps** | 40 |
| **Time per step** | ~1.17s/step |
| **Peak VRAM allocated** | 62.22 GB |
| **Peak VRAM reserved** | 63.20 GB |
| **VRAM utilization** | 34.5% of 183 GB |

## 5. Side-by-Side Comparison: L4 vs B200

| Metric | L4 (seq offload) | B200 (full bf16) | Delta |
|--------|-------------------|-------------------|-------|
| **Generation time** | 728s | 51.2s | **14.2x faster** |
| **Peak VRAM** | 3.98 GB | 63.20 GB | 15.9x more |
| **VRAM % used** | 17% of 23 GB | 34.5% of 183 GB | — |
| **Offloading** | Sequential CPU | None | — |
| **Load time** | ~30s | 42.4s | Comparable |

## 6. Output

- **File**: `/blue/yuanyuan.lei/zihua.meng/vlm_inference/outputs/m2_b200_full_bf16.png`
- **Size**: 661 KB, PNG, 1024x1024 RGB
- **Visual**: Red and blue objects side by side on a light marble-like surface. Visually comparable to the L4 output — the model produces coherent multi-image editing results in both configurations.

## 7. Key Takeaways for Future Milestones

1. **B200 eliminates the memory bottleneck.** The full bf16 model uses ~58 GB of 183 GB VRAM, leaving >120 GB headroom for batching, larger resolutions, or additional models.

2. **Generation is now compute-bound, not memory-bound.** At 1.17s/step, the bottleneck is transformer FLOPs, not data movement. This is the correct regime for acceleration work (kernel optimization, distillation, compilation).

3. **~65% VRAM headroom** means there is ample room for:
   - Batch inference (multiple images per pass)
   - Higher resolution inputs
   - torch.compile() workspace
   - Profiling overhead

4. **No environment differences** between L4 and B200. The same venv, same packages, same code — only the `.to("cuda")` vs `enable_sequential_cpu_offload()` call differs.

---

# Milestone 1: Environment Setup + Feasibility Gate

## 1. Installed Package Versions

| Package | Version |
|---------|---------|
| **Python** | 3.12.5 (via `module load python/3.12`) |
| **torch** | 2.10.0+cu128 |
| **CUDA (torch)** | 12.8 |
| **transformers** | 5.3.0 |
| **diffusers** | 0.38.0.dev0 (from git, commit c02c17c6) |
| **Pillow** | 12.0.0 |
| **accelerate** | 1.13.0 |
| **numpy** | 2.3.5 |
| **safetensors** | 0.7.0 |

Note: System Python 3.9.21 was **not compatible** with latest diffusers (requires >=3.10). Used `module load python/3.12` instead.

Venv location: `/blue/yuanyuan.lei/zihua.meng/vlm_inference/venv/`

## 2. Model Download

| Item | Value |
|------|-------|
| **Model** | Qwen/Qwen-Image-Edit-2511 |
| **Location** | `/blue/yuanyuan.lei/zihua.meng/vlm_inference/weights/Qwen-Image-Edit-2511/` |
| **Total size on disk** | 54 GB |
| **Transformer** | 39 GB (5 shards) |
| **Text encoder** | 16 GB (729 weight files) |
| **VAE** | 241 MB |
| **Download time** | ~17 seconds (xet-accelerated) |

## 3. Loading Configuration Attempts

### Strategy 1: Full bf16 `.to('cuda')`
- **Result**: OOM
- **Detail**: Model consumed 21.62 GiB of the 22.04 GiB VRAM, leaving only 49 MiB free. OOM when trying to allocate 108 MiB for inference activations.
- **Conclusion**: Full bf16 is not viable on L4. The model footprint alone nearly saturates VRAM with zero headroom for activations.

### Strategy 2: `enable_model_cpu_offload()`
- **Result**: OOM
- **Detail**: Model-level offloading still moved the entire transformer to GPU during the forward pass. OOM at the same point — 21.99 GiB in use, only 53 MiB free.
- **Conclusion**: Model-level offload is insufficient. The transformer alone (~39GB bf16 weights -> ~19.5GB in memory) fills the 23GB VRAM.

### Strategy 3: `enable_sequential_cpu_offload()` — SUCCESS
- **Result**: Successful forward pass
- **Detail**: Loads model to CPU, moves individual layers to GPU one at a time during forward pass. Peak GPU allocation during inference: ~3.92 GB reserved (0.02 GB allocated by PyTorch). The nvidia-smi snapshot during inference showed peak usage of **~3.98 GB** (3980 MiB).
- **Wall-clock time**: 728.1 seconds (~12.1 minutes) for 40 inference steps
- **Conclusion**: Sequential CPU offload is the only viable strategy for L4 without quantization. Extreme VRAM savings (~4 GB vs 22+ GB) but at the cost of very slow inference due to constant CPU↔GPU data transfers.

### Viable Loading Configuration

```python
from diffusers import QwenImageEditPlusPipeline
import torch

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "path/to/weights", torch_dtype=torch.bfloat16
)
pipeline.enable_sequential_cpu_offload()
```

## 4. Peak VRAM Usage

| Measurement | Value |
|-------------|-------|
| **After load (seq offload)** | 262 MiB (model on CPU) |
| **Peak during inference** | 3,980 MiB (~3.9 GB) |
| **Post-inference (torch reserved)** | 3.92 GB |
| **Post-inference (torch allocated)** | 0.02 GB |

The L4's 23 GB VRAM is massively underutilized with sequential offload — only 17% peak usage. The bottleneck is that no single component (transformer, text_encoder) fits in VRAM entirely.

## 5. nvcc / CUDA Compatibility

- **No nvcc issues at runtime.** All operations used pre-compiled CUDA kernels from the PyTorch wheel (cu128). No JIT compilation was needed.
- The `guidance_scale is passed as 1.0, but ignored` warning is benign — the model is not guidance-distilled.
- `accelerate` was **required** for offloading. Without it, `enable_model_cpu_offload()` and `enable_sequential_cpu_offload()` both raise errors.

## 6. Test Configuration

| Parameter | Value |
|-----------|-------|
| **Input images** | Two 512x512 solid-color images (red, blue) |
| **Prompt** | "A red object and a blue object sitting side by side on a white table." |
| **num_inference_steps** | 40 |
| **true_cfg_scale** | 4.0 |
| **negative_prompt** | " " (single space) |
| **guidance_scale** | 1.0 |
| **generator seed** | 42 |
| **Output resolution** | 1024x1024 |

## 7. Output

- **File**: `/blue/yuanyuan.lei/zihua.meng/vlm_inference/outputs/m1_seq_offload.png`
- **Size**: 586 KB, PNG, 1024x1024 RGB
- **Visual**: Correctly generated a red object and blue object side by side on a white table, demonstrating the model understood the multi-image editing task

## 8. Key Takeaways for Milestone 2

1. **L4 is viable but only with sequential CPU offload**, yielding ~12 min per forward pass (40 steps). This is functionally unusable for production but sufficient for profiling and correctness validation.

2. **The VRAM bottleneck is the transformer** (39 GB on disk, ~19.5 GB in bf16 memory). It alone exceeds the 23 GB VRAM capacity. The text encoder (16 GB on disk, ~8 GB in memory) is the second-largest component.

3. **Quantization is untested** but is the logical next fallback. FP8 or INT8 quantization could halve the transformer footprint to ~10 GB, potentially enabling model-level offload (text_encoder on CPU, transformer on GPU).

4. **B200 (80 GB VRAM) would eliminate the memory bottleneck entirely.** The full bf16 model (~28 GB) would fit with >50 GB headroom for activations, KV cache, and batch processing.

5. **No nvcc or compilation issues.** The pre-compiled PyTorch+CUDA wheels work cleanly on this system.

---

# Milestone 0: Workspace Init + Baseline Recon

## 1. Environment Snapshot

| Item | Value |
|------|-------|
| **GPU** | NVIDIA L4, 1x, 23034 MiB VRAM |
| **GPU Driver** | 570.195.03 (supports CUDA 12.x) |
| **nvcc** | Not available (no CUDA toolkit installed) |
| **Python** | 3.9.21 (`/usr/bin/python3`) |
| **Conda** | Not available |
| **OS** | RHEL 9.5, kernel 5.14.0-503.40.1.el9_5.x86_64 |
| **Home disk** | 40G total, 28G available (31% used) |
| **Blue disk** | 8.0T total, 449G available (95% used) |

## 2. Repo Structure Summary (Relevant Files)

```
Qwen-Image/
├── README.md                          # Main documentation, quickstart, showcases
├── Qwen-Image.md                      # T2I capabilities showcase
├── Qwen-Image-Edit.md                 # Original Edit model showcase (Aug 2025)
├── Qwen-Image-Edit-2509.md            # 2509 variant showcase & capabilities
├── LICENSE                            # Apache 2.0
├── src/examples/
│   ├── edit_demo.py                   # Gradio demo for image editing (single-image)
│   ├── demo.py                        # Gradio demo for T2I (multi-GPU support)
│   ├── generate_w_prompt_enhance.py   # T2I with prompt enhancement
│   └── tools/
│       ├── __init__.py
│       ├── prompt_utils.py            # Prompt rewriting (DashScope API)
│       └── prompt_utils_2512.py       # Prompt enhancement for 2512 variant
```

## 3. Multi-Image Editing Entry Point

**Pipeline class**: `QwenImageEditPlusPipeline` (from `diffusers`)

Both 2509 and 2511 use the same pipeline class. The multi-image editing interface works by passing a list of PIL Images to the `image` parameter:

```python
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
pipeline.to('cuda')

inputs = {
    "image": [image1, image2],        # List of PIL Images (1-3 images)
    "prompt": prompt,
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
output = pipeline(**inputs)
```

**Source evidence**: `README.md:131-163` (2511 quickstart), `README.md:270-303` (2509 quickstart)

The original Qwen-Image-Edit (Aug 2025) uses a different, older pipeline: `QwenImageEditPipeline` (note: no "Plus"), which only supports single-image input. See `src/examples/edit_demo.py:7`.

## 4. 2509 vs 2511 Comparison

### Model Timeline
| Model | Release | Pipeline Class | HuggingFace ID |
|-------|---------|---------------|----------------|
| Qwen-Image-Edit | Aug 18, 2025 | `QwenImageEditPipeline` | `Qwen/Qwen-Image-Edit` |
| Qwen-Image-Edit-2509 | Sep 22, 2025 | `QwenImageEditPlusPipeline` | `Qwen/Qwen-Image-Edit-2509` |
| Qwen-Image-Edit-2511 | Dec 23, 2025 | `QwenImageEditPlusPipeline` | `Qwen/Qwen-Image-Edit-2511` |

### Qwen-Image-Edit-2509 Capabilities
Source: `Qwen-Image-Edit-2509.md`

- **Multi-image editing** (new in 2509): Supports person+person, person+product, person+scene combinations. Optimal with 1-3 input images.
- **Enhanced single-image consistency**: Improved person identity preservation (face, pose), product identity, text editing (font type, color, material).
- **Native ControlNet support**: Depth maps, edge maps, keypoint maps, sketches — all built into the base model.

### Qwen-Image-Edit-2511 Capabilities
Source: `README.md:481-531` (showcase section)

- **Significantly improved character consistency**: Preserves identity and visual characteristics across imaginative edits.
- **Multi-person group photo consistency**: High-fidelity fusion of separate person images into coherent group shots (improvement over 2509's multi-image support).
- **Built-in community LoRA support**: Lighting enhancement, viewpoint generation — no extra tuning needed.
- **Industrial design applications**: Batch product design, material replacement for industrial components.
- **Enhanced geometric reasoning**: Direct generation of auxiliary construction lines for design/annotation.

### Feature Delta: 2511 over 2509
| Capability | 2509 | 2511 |
|------------|------|------|
| Multi-image editing | Yes (1-3 images) | Yes (improved) |
| Single character consistency | Good | Significantly improved |
| Multi-person consistency | Not emphasized | Explicitly improved |
| ControlNet support | Native | Inherited |
| Community LoRAs built-in | No | Yes |
| Industrial design | Not emphasized | Dedicated support |
| Geometric reasoning | Not mentioned | New capability |
| Default inference steps | 40 | 40 |
| Pipeline class | QwenImageEditPlusPipeline | QwenImageEditPlusPipeline |

### Acceleration Ecosystem
Source: `README.md:43-47`

- **LightX2V**: Day-0 acceleration for 2511. Combines diffusion distillation with inference optimizations. Claims 25x reduction in DiT NFEs and 42.55x overall speedup. Supports NVIDIA, Hygon, Metax, Ascend, Cambricon hardware.
- **vLLM-Omni**: Day-0 support for 2511 with long sequence parallelism, cache acceleration, fast kernels.
- **SGLang-Diffusion**: Day-0 support for 2511 (CLI example provided in README).

## 5. Dependency and Environment Risk Inventory

### Declared Dependencies
Source: `README.md:73-78`

1. `transformers>=4.51.3` (Qwen2.5-VL support)
2. `diffusers` — latest from git (`pip install git+https://github.com/huggingface/diffusers`)
3. `torch` with CUDA (bfloat16 required)
4. `PIL`/`Pillow`
5. For prompt enhancement (optional): `dashscope` API + API key

No `requirements.txt`, `pyproject.toml`, or `setup.py` is present in the repo — dependencies are documented only in the README.

### Risk Assessment

| Risk | Severity | Detail |
|------|----------|--------|
| **VRAM constraint** | HIGH | L4 has 23GB. Qwen-Image is a 20B MMDiT model. May not fit in full bf16 without offloading. |
| **No CUDA toolkit** | MEDIUM | nvcc is missing. Some diffusers/torch operations may require compilation. Driver 570.x supports CUDA 12.x but runtime libraries may be absent. |
| **No conda** | MEDIUM | Cannot create isolated environments via conda. Will need `venv` or system-level pip. |
| **Blue storage at 95%** | MEDIUM | Only 449G free on 8T Lustre. Model weights (~40-80GB) will fit but monitor usage. |
| **Python 3.9** | LOW | Should be compatible with all declared deps. transformers 4.51.3 supports 3.8+. |
| **No requirements file** | LOW | Must manually track dep versions. Pin versions in our own environment config. |
| **Prompt enhancement needs DashScope API** | LOW | Optional feature. Can run inference without it. |

## 6. First-Baseline Variant Recommendation

### Recommendation: **Qwen-Image-Edit-2511**

### Rationale

1. **Latest and most capable iteration**: Released Dec 23, 2025 — represents the most mature version of the multi-image editing pipeline. If the goal is to profile and optimize production-relevant inference, 2511 is the production-relevant model.

2. **Same pipeline architecture as 2509**: Both use `QwenImageEditPlusPipeline` with identical API signatures (`image`, `prompt`, `true_cfg_scale`, `negative_prompt`, `num_inference_steps`, `guidance_scale`). Profiling results on 2511 will be structurally applicable to 2509 as well.

3. **Superset of capabilities**: 2511 includes everything from 2509 plus character consistency improvements, multi-person fusion, built-in LoRAs, industrial design support, and geometric reasoning. Profiling the richer model ensures we capture the full computational footprint.

4. **Actively promoted as the default**: The README header links directly to `Qwen/Qwen-Image-Edit-2511` on HuggingFace. The main quickstart section features 2511, with 2509 nested under a "Previous Version" collapsible. This signals 2511 is the canonical version.

5. **Day-0 acceleration ecosystem**: LightX2V, vLLM-Omni, and SGLang-Diffusion all explicitly support 2511. Profiling 2511 as baseline allows direct comparison against these optimized backends later.

6. **Backward comparison path**: If profiling reveals 2511-specific bottlenecks, we can subsequently profile 2509 to isolate which improvements introduced overhead. Starting with the superset makes this straightforward.

### Caveat
The 20B model on a single L4 (23GB) will be tight. We may need to use model offloading (CPU offload or sequential layer loading) or quantization (FP8) for the initial baseline. DiffSynth-Studio claims inference within 4GB VRAM with layer-by-layer offload — worth evaluating if full-precision loading fails.
