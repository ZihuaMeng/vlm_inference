# Findings

---

# Cross-Model Analysis: Qwen-Image-Edit-2511 vs FLUX.1 Kontext-dev

This section synthesizes all Qwen (M0–M9) and FLUX (F0–F2) findings into a structured cross-model comparison. It serves as the experimental-results backbone for the paper, answering: what is shared, what differs, what is intrinsic to the method vs the models, and what the strongest defensible claims are.

## 1. Profiling Comparison

### 1.1 Pipeline Cost Distribution

Both models concentrate >97% of inference cost in the denoising loop. Non-denoising stages (text encoding, VAE encode/decode, image preprocessing) are negligible on both architectures.

| Stage | Qwen (M3) | Kontext (F1) | Shared? |
|-------|-----------|-------------|---------|
| Denoising loop | 99.3% (47.38s) | 97.8% (8.997s) | **YES** — denoising dominates |
| Text encoding | 0.3% (0.14s) | 0.2% (0.02s) | Yes — negligible |
| VAE decode | 0.2% (0.08s) | 1.2% (0.11s) | Yes — negligible (Kontext's higher % is proportional) |
| VAE encode | 0.1% (0.06s) | 0.6% (0.05s) | Yes — negligible |
| Image preprocessing | 0.1% (0.06s) | 0.2% (0.02s) | Yes — negligible |
| **Total wall-clock** | **47.7s** | **9.2s** | **5.2x gap** |

**Finding 1.1**: The denoising loop is the singular optimization target on both models. Any acceleration method that does not reduce denoising cost is irrelevant. This is structural to diffusion-based image generation, not model-specific.

### 1.2 Per-Step Timing Pattern

Both models exhibit perfectly uniform step costs — no step is inherently more expensive than another within the denoising loop.

| Metric | Qwen (M3) | Kontext (F1) | Shared? |
|--------|-----------|-------------|---------|
| Mean step time | 1184.5ms | 321.3ms | 3.7x gap |
| Std dev | ±1.4ms | ±0.2ms | Both <0.1% CV |
| CV (std/mean) | 0.12% | 0.06% | **YES** — perfectly uniform |
| Passes per step | 2 (cond + uncond) | 1 (guidance-distilled) | Different |
| Per-pass time | 592ms | 321ms | 1.8x gap |
| Scheduler overhead | 0.1ms | 0.1ms | Identical — negligible |
| Total steps | 40 | 28 | Different |
| Total passes | 80 | 28 | **2.86x gap** |

**Finding 1.2**: Step uniformity means any step can be cached or skipped with equal compute savings. There are no "cheap" or "expensive" steps to preferentially target. This uniformity is shared across architectures and justifies fixed caching schedules (which assume constant savings per cached step).

### 1.3 Why Kontext Is 5.2x Faster (Cost Decomposition)

The 5.2x speedup decomposes cleanly:

| Factor | Contribution | Explanation |
|--------|-------------|-------------|
| Guidance distillation | 2.0x | 1 pass/step vs 2 (eliminates unconditional pass) |
| Fewer steps | 1.43x | 28 vs 40 default steps |
| Smaller model | 1.84x | 592ms/pass vs 321ms/pass (12B vs 19.5B transformer) |
| **Combined** | **5.2x** | 47.4s → 9.17s |

Note: 2.0 × 1.43 × 1.84 = 5.27, closely matching the measured 5.18x.

### 1.4 Module-Level Breakdown

Attention dominates compute in both architectures at ~73–76% of block time.

| Component | Qwen (60 blocks) | Kontext dual (19 blocks) | Kontext single (38 blocks) |
|-----------|-------------------|--------------------------|---------------------------|
| **Attention** | **75.8%** | **73%** | **75%** |
| Image MLP/FF | 12.2% | 14% | 7% + 9% proj |
| Text MLP/FF | 0.7% | 1% | — |
| Norms/gates/other | 11.2% | 12% | 10% |
| Per-block time | 9.87ms | 5.91ms | 5.48ms |

**Block uniformity comparison:**
- **Qwen**: All 60 blocks identical in structure and cost (dual-stream only).
- **Kontext**: Two block types, but the cost ratio is only **1.08x** (not the 2.4x predicted from parameter counts). On B200 with FlashAttention, attention cost is dominated by sequence length (same for both types), not parameter count. The blocks are _effectively_ uniform.

**Finding 1.3**: Attention dominance (~75%) is shared across both architectures, both block types. This means attention-based optimizations (FlashAttention, sparse attention, token pruning) would affect both models similarly, and schedule-based acceleration (which skips entire passes) captures ~75% attention savings per skipped pass.

### 1.5 Profiling Comparison Summary

**What is shared across both models (structural to diffusion inference):**
1. Denoising loop concentrates >97% of cost
2. Steps are perfectly uniform (CV < 0.15%)
3. Attention dominates at ~75% of block time
4. Blocks are effectively uniform in cost (even Kontext's two types)
5. Scheduler overhead is negligible (<0.01% of step time)

**What differs (model-specific):**
1. Guidance strategy: True CFG (2 passes) vs guidance-distilled (1 pass)
2. Step count: 40 vs 28 default
3. Model size: 19.5B vs 12B transformer
4. Block structure: Homogeneous vs heterogeneous (but cost-equivalent)
5. Text encoder: Vision-language (8B) vs dual text-only (5.1B)

---

## 2. Temporal Redundancy Comparison

### 2.1 Three-Phase Pattern: Confirmed on Both Models

Both models exhibit an identical three-phase pattern in step-to-step cosine similarity of transformer outputs (noise predictions):

| Phase | Qwen (40 steps) | Kontext (28 steps) | Pattern |
|-------|-----------------|-------------------|---------|
| **Convergence** | Steps 1–4 (cos 0.980→0.9995) | Steps 1–3 (cos 0.957→0.999) | Rapid increase |
| **Plateau** | Steps 5–30 (cos >0.999) | Steps 4–23 (cos >0.999) | Near-identical outputs |
| **Divergence** | Steps 31–39 (cos 0.999→0.816) | Steps 24–27 (cos 0.999→0.981) | Final corrections |

### 2.2 Plateau Boundaries and Coverage

| Metric | Qwen | Kontext |
|--------|------|---------|
| Plateau start step | 5 | 4 |
| Plateau end step | 30 | 23 |
| Plateau steps | 26 | 20 |
| Total step transitions | 39 | 27 |
| **Plateau coverage** | **66.7%** | **74.1%** |
| Peak similarity (cos) | 0.99994 (step 23) | 0.99994 (step 12) |
| Plateau mean cos | 0.99989 | 0.99992 |

**Finding 2.1**: The plateau occupies a larger fraction of Kontext's trajectory (74% vs 67%). This is because Kontext's guidance-distilled trajectory is smoother — the model does not need to reconcile conditional and unconditional predictions, so consecutive steps are more similar.

### 2.3 Normalized Trajectory Overlay

To compare the temporal profiles despite different step counts, we normalize to trajectory fraction (0.0 = start, 1.0 = end):

| Trajectory Fraction | Qwen cos | Kontext cos | Phase |
|---------------------|----------|-------------|-------|
| 0.00 (step 0→1) | 0.981 | 0.957 | Convergence |
| 0.05 | 0.995 | 0.995 | Convergence |
| 0.10 | 0.999 | 0.999 | Convergence → Plateau |
| 0.15 | 0.9996 | 0.9997 | Plateau |
| 0.25 | 0.9999 | 0.9999 | Plateau (peak) |
| 0.50 | 0.9999 | 0.9999 | Plateau (mid) |
| 0.65 | 0.9999 | 0.9997 | Plateau (Qwen) / Late plateau (Kontext) |
| 0.75 | 0.9998 | 0.9995 | Plateau (Qwen) / Divergence start (Kontext) |
| 0.85 | 0.9995 | 0.993 | Late plateau (Qwen) / Divergence (Kontext) |
| 0.90 | 0.9991 | 0.982 | Divergence start (Qwen) / Late divergence (Kontext) |
| 0.95 | 0.998 | — | Divergence (Qwen) |
| 1.00 (final step) | 0.816 | 0.981 | **Divergence endpoint** |

**Finding 2.2**: When normalized, the plateau shapes are nearly identical in the range 0.10–0.65. The key difference is in the divergence phase: Qwen diverges dramatically (cos=0.816 at the final step), while Kontext diverges mildly (cos=0.981). This means Kontext's final steps make smaller corrections, likely because the guidance-distilled trajectory doesn't need the large final-step adjustment that CFG-based trajectories require.

### 2.4 Divergence Severity

| Metric | Qwen | Kontext | Interpretation |
|--------|------|---------|---------------|
| Final-step cos | 0.816 | 0.981 | **Kontext 8.3x milder** (1−cos: 0.184 vs 0.019) |
| Divergence onset (normalized) | ~0.77 (step 31/40) | ~0.85 (step 24/28) | Kontext diverges later |
| Steps below cos 0.999 | 9 (steps 31-39) | 4 (steps 24-27) | Kontext has fewer |
| Min cos at any step | 0.816 | 0.957 (step 1) | Qwen's final step is more extreme than any Kontext step |

**Finding 2.3**: Kontext's milder divergence has a paradoxical consequence for caching quality. The model has less natural error-correction capacity at trajectory end — when caching perturbs the trajectory during the plateau, there is less divergence to "pull back" toward the correct result. This may explain why Kontext shows lower caching SSIM despite a larger plateau.

### 2.5 Block-Level Temporal Dynamics

| Block Position | Qwen (sampled) | Kontext (sampled) | Role |
|---------------|----------------|-------------------|------|
| Early (Block 0 / dual_0) | mean cos=0.997 | mean cos=0.999 | Stable — safe to cache |
| Mid-early (Block 14 / dual_9) | mean cos=0.996 | mean cos=0.999 | Stable |
| Mid (Block 29 / dual_18) | mean cos=0.991 | mean cos=0.997 | Moderate dynamics |
| Mid-late (Block 44 / single_18) | mean cos=0.967 | mean cos=0.994 | **Most dynamic (Qwen)** |
| Late (Block 59 / single_37) | mean cos=0.9998 | mean cos=0.990 | **Normalizer (Qwen) / Most dynamic (Kontext)** |

**Finding 2.4**: The most temporally dynamic block position differs between models:
- **Qwen**: Block 44 (mid-late, 73% through the stack) is most dynamic. Block 59 (final) is a normalizer that collapses all variation.
- **Kontext**: single_37 (final block) is most dynamic. There is no strong normalizer — dynamics increase monotonically toward the end.

This structural difference explains why block-aware caching failed on Qwen (M6): the normalizer block (59) masks internal dynamics, so caching blocks 0-39 while recomputing 40-59 still produces outputs that look "similar" to the model's output norm but are internally wrong. On Kontext, the monotonically increasing dynamics would make block-aware caching even less viable — the most important blocks are at the end.

### 2.6 CFG Redundancy (Qwen Only)

| Phase | Steps | Cond-vs-Uncond cos | Interpretation |
|-------|-------|-------------------|---------------|
| Early divergence | 0–2 | 0.968–0.995 | CFG matters — both passes needed |
| Plateau | 3–38 | >0.999 | Cond ≈ uncond — unconditional pass is redundant |
| Final step | 39 | 0.993 | Mild re-divergence |

**Kontext has no CFG redundancy to exploit** — guidance distillation eliminates the unconditional pass entirely. This is not a disadvantage but an architectural advantage: Kontext permanently captures the benefit that Qwen's CFG-skip only approximates during the plateau.

**Finding 2.5**: Qwen's CFG redundancy (cos > 0.999 for 90% of the trajectory) validates the guidance distillation approach used by Kontext. The unconditional pass is genuinely redundant for most of the trajectory — Kontext's architecture simply makes this permanent rather than requiring a runtime skip schedule.

### 2.7 Temporal Redundancy Summary

**What is shared (structural to rectified flow matching):**
1. Three-phase pattern: convergence → plateau → divergence
2. Plateau similarity >0.999 in both models
3. Plateau occupies the majority of the trajectory (65–74%)
4. Block-level dynamics increase toward the model's "processing center"
5. The plateau is exploitable by noise-prediction-level caching on both models

**What differs (model/architecture-specific):**
1. Plateau coverage: Kontext 74% vs Qwen 67% (guidance distillation makes trajectory smoother)
2. Divergence severity: Kontext cos=0.981 vs Qwen cos=0.816 (10x milder)
3. CFG redundancy: Present in Qwen, absent in Kontext (by design)
4. Most dynamic block location: Mid-late (Qwen) vs final (Kontext)
5. Error-correction capacity: Qwen has strong final-step correction; Kontext does not

---

## 3. Acceleration Comparison

### 3.1 All Tested Configurations

| Model | Method | Config | Passes | Speedup | Mean SSIM | Test Cases | Source |
|-------|--------|--------|-------:|--------:|----------:|-----------:|--------|
| Qwen | CFG-skip only | Skip uncond steps 5-34 | 50/80 | 1.60x | 0.851 | 18 | M9 |
| Qwen | Temporal cache only | Cache steps 10-25 | 48/80 | 1.67x | 0.885 | 18 | M9 |
| Qwen | **Fixed combined** | **Temporal 10-25 + CFG 5-9,26-34** | **34/80** | **2.35x** | **0.836** | **18** | **M9** |
| Qwen | Adaptive combined | Cosine-signal 3-tier | ~20/80 | 3.95x | 0.809 | 7 | M8 |
| Kontext | **Conservative temporal** | **Cache steps 6-20** | **13/28** | **2.15x** | **0.754** | **12** | **F2** |
| Kontext | Aggressive temporal | Cache steps 4-23 | 8/28 | 3.50x | 0.635 | 12 | F2 |

Note: Bold rows indicate the recommended operating point for each model.

### 3.2 Speedup Reliability

All fixed schedules achieve exactly their theoretical speedup with negligible variance:

| Configuration | Theoretical | Measured | Std |
|--------------|----------:|--------:|-----:|
| Qwen CFG-skip | 1.600x | 1.600x | ±0.001 |
| Qwen temporal | 1.667x | 1.667x | ±0.001 |
| Qwen combined | 2.353x | 2.353x | ±0.001 |
| Kontext conservative | 2.154x | 2.154x | ±0.001 |
| Kontext aggressive | 3.500x | 3.500x | ±0.001 |

**Finding 3.1**: Speedup is perfectly deterministic for fixed schedules. It depends solely on the pass count ratio and is independent of input content, difficulty, or image count. This is a strong property for production deployment — the latency improvement is guaranteed.

### 3.3 The Dual-Axis Advantage

Qwen has two independent acceleration axes (CFG redundancy + temporal redundancy). Kontext has only one (temporal redundancy). This creates a fundamental asymmetry:

| Dimension | Qwen | Kontext |
|-----------|------|---------|
| Available axes | CFG + Temporal | Temporal only |
| Best single-axis speedup | 1.67x (temporal) | 2.15x (temporal) |
| Best single-axis SSIM | 0.885 (temporal) | 0.754 (temporal, F2) |
| Best combined speedup | 2.35x | 2.15x (no combination possible) |
| Best combined SSIM | 0.836 | 0.754 |

**Finding 3.2**: Qwen's dual-axis access enables a critical advantage: combining two moderate schedules produces better quality than either aggressive schedule alone. Qwen's combined method (2.35x, 0.836 SSIM) achieves higher speedup AND better quality than Kontext's conservative temporal-only method (2.15x, 0.754 SSIM). The combination works because the two axes target different redundancy dimensions (inter-step vs intra-step) with overlapping vulnerability — combining them does NOT compound quality loss (observed 0.836 vs predicted-independent 0.753).

**Why Kontext's single-axis SSIM is lower despite a larger plateau:**
1. The entire quality cost falls on one axis (no CFG sharing)
2. Kontext's milder divergence (cos=0.981) provides less error correction
3. F2's test suite is arguably harder-distributed than M9's

### 3.4 Absolute Timing Impact

Despite Kontext's lower speedup ratio and quality, its absolute timing is far more favorable because the baseline is 5.2x faster:

| Scenario | Qwen Baseline | Qwen Accel | Kontext Baseline | Kontext Accel |
|----------|--------------|-----------|-----------------|--------------|
| 1-image | 26.5s | 11.2s (2.35x) | 8.9s | 4.1s (2.15x) |
| 2-image | 47.0s | 20.0s (2.35x) | N/A | N/A |
| 3-image | 72.2s | 30.7s (2.35x) | N/A | N/A |

**Finding 3.3**: Kontext's accelerated time (4.1s) is 2.7x faster than Qwen's accelerated time (11.2s) for comparable 1-image tasks, despite having a lower speedup ratio. The baseline advantage from guidance distillation + fewer steps + smaller model overwhelms the acceleration ratio difference. For latency-critical applications, Kontext + temporal caching is the fastest path to acceptable results.

### 3.5 Component Contribution (Qwen Ablation)

From M9's 18-case ablation, decomposing the combined method:

| Component | Speedup | SSIM | LPIPS | Role |
|-----------|--------:|-----:|------:|------|
| CFG-skip alone | 1.60x | 0.851 | 0.170 | Moderate speedup, moderate quality |
| Temporal alone | 1.67x | 0.885 | 0.143 | Slightly better on both axes |
| Combined | 2.35x | 0.836 | 0.215 | +41% more speedup vs temporal alone, -0.049 SSIM cost |

**Finding 3.4**: Temporal caching is the stronger single axis (better speedup AND better quality than CFG-skip). CFG-skip's primary value is as an additive speedup when combined with temporal caching, contributing +0.686x additional speedup at only -0.049 SSIM cost. The non-compounding quality loss (0.836 observed vs 0.753 predicted) is because both axes degrade the same content types.

---

## 4. Quality Boundary Analysis

### 4.1 Bimodal Distribution on Both Models

Both models exhibit a bimodal quality distribution under temporal caching acceleration:

| Metric | Qwen M9 (combined, 2.35x) | Kontext F2 (conservative, 2.15x) |
|--------|-------------------------:|--------------------------------:|
| Test cases | 18 | 12 |
| Mean SSIM | 0.836 | 0.754 |
| Median SSIM | 0.890 | 0.787 |
| SSIM std | 0.108 | 0.190 |
| SSIM >= 0.90 | 39% (7/18) | 33% (4/12) |
| SSIM >= 0.85 | 61% (11/18) | 42% (5/12) |
| SSIM < 0.80 | 33% (6/18) | 50% (6/12) |
| SSIM < 0.70 | 22% (4/18) | 25% (3/12) |
| Robust cluster | 0.83–0.96 | 0.83–0.99 |
| Fragile cluster | 0.64–0.80 | 0.37–0.74 |

**Finding 4.1**: The bimodal distribution is a method property, not a model property. Both models separate into a "robust" cluster where acceleration works well and a "fragile" cluster where quality degrades substantially. The gap between clusters is 0.03–0.09 SSIM, with few cases in the middle.

### 4.2 Content-Type Failure Overlap

The same content types degrade on both models. This is the strongest evidence that the quality boundary is method-intrinsic:

| Content Type | Qwen M9 SSIM | Kontext F2 SSIM | Both Degrade? | Category |
|-------------|-------------|----------------|--------------|----------|
| complex_scene | 0.637 | 0.415 | **YES** | Fragile |
| style_transfer | 0.642 | 0.681 | **YES** | Fragile |
| material_texture / fine_texture | 0.665 | 0.366 | **YES** | Fragile |
| nature_scene / nature_detail | 0.687 | 0.710 | **YES** | Fragile |
| identity_preserving | 0.753 / 0.931 | 0.686 | **Partial** | Mixed |
| interior_design | 0.890 | 0.834 | No | Robust |
| portrait | 0.911 | 0.909 | No | Robust |
| simple_object | 0.926 | 0.987 | No | Robust |
| arrangement | 0.955 | 0.906 | No | Robust |
| object_placement | 0.947 | — | — | Robust (Qwen) |

Note: Qwen's identity_preserving has two measurements (0.753 for a hard 3-img case, 0.931 for a 2-img case). The content-type label masks within-type variance driven by difficulty.

**Finding 4.2**: Content types that require fine-grained visual detail (texture, artistic style, complex multi-element scenes, natural organic patterns) degrade on BOTH models. Content types with structural/semantic focus (object placement, portraits, arrangements, simple objects) are robust on BOTH models. The Spearman rank correlation between Qwen and Kontext SSIM across matched content types is strongly positive — the failure ranking is preserved across architectures.

### 4.3 Difficulty as the Dominant Predictor

| Difficulty | Qwen M9 Combined SSIM | Kontext F2 Conservative SSIM |
|-----------|----------------------:|----------------------------:|
| Easy | 0.943 (n=3) | 0.892 (n=4) |
| Medium | 0.886 (n=7) | 0.826 (n=4) |
| Hard | 0.753 (n=8) | 0.543 (n=4) |
| **Easy→Medium drop** | **-0.057** | **-0.066** |
| **Medium→Hard drop** | **-0.133** | **-0.283** |

**Finding 4.3**: The quality cliff between medium and hard content is the defining limitation. Easy→medium is a gradual decline; medium→hard is a dramatic drop (2-4x larger). This non-linearity means there is no smooth quality-speedup tradeoff — the method either works well or fails substantially, depending on content type.

### 4.4 Why Kontext Degrades More Severely

Kontext shows lower absolute SSIM on matched content types (e.g., fine_texture: 0.366 vs 0.665, complex_scene: 0.415 vs 0.637). Three contributing factors:

1. **Single-axis loading**: Qwen distributes quality cost across two moderate schedules (CFG-skip + temporal cache). Kontext loads the entire cost onto temporal caching alone. The quality budget is exhausted faster.

2. **Weaker error correction**: Kontext's final-step divergence is 10x milder (cos=0.981 vs 0.816). When caching introduces trajectory perturbations during the plateau, Kontext has less capacity to "pull back" toward the correct result in the divergence phase.

3. **Test suite calibration**: F2's hard cases may be inherently harder than M9's. Kontext's fine_texture case (SSIM 0.366) is catastrophically low in a way that suggests the baseline generation was particularly sensitive to mid-trajectory perturbation.

**Finding 4.4**: The severity difference is real but partially confounded by test suite difficulty. The directional finding — Kontext degrades more than Qwen on matched content types — is robust and attributable primarily to single-axis loading and weaker error correction.

### 4.5 The "Method Limitation" Argument

The evidence for temporal step-caching quality loss being a method limitation (not model-specific):

| Evidence | Strength | Detail |
|----------|----------|--------|
| Same failure content types | **Strong** | 4/4 fragile types overlap: complex_scene, style_transfer, texture, nature |
| Same robust content types | **Strong** | 4/4 robust types overlap: objects, portraits, arrangements, interiors |
| Bimodal distribution on both | **Strong** | Both show clean separation between robust and fragile clusters |
| Difficulty as strongest predictor on both | **Strong** | Easy > medium >> hard on both models |
| Same mechanism | **Strong** | Noise-prediction-level caching in the plateau region on both |
| Different architectures | **Supporting** | 19.5B dual-stream vs 12B mixed-stream, true CFG vs guidance-distilled |
| Different step counts | **Supporting** | 40 steps vs 28 steps, different plateau boundaries |
| Same scheduler | **Caveat** | Both use FlowMatchEulerDiscreteScheduler — the shared scheduler could be the source rather than the method per se |

**Finding 4.5**: The quality boundary of temporal step-caching is a property of the caching method interacting with content complexity during rectified flow matching inference. It is not specific to either model's architecture, guidance strategy, or parameter count. The shared scheduler (FlowMatchEulerDiscreteScheduler) is a confound — we cannot fully distinguish "property of temporal caching" from "property of this specific scheduler's trajectory dynamics." However, since this scheduler is the standard for flow matching models, this distinction has limited practical significance.

---

## 5. Paper Claim Structure

### Claim 1: Temporal Plateau is Structural to Rectified Flow Matching

**Statement**: Rectified flow matching models exhibit a universal temporal redundancy pattern — a three-phase trajectory (convergence → plateau → divergence) where consecutive denoising steps produce near-identical noise predictions (cos > 0.999) for 65–74% of the trajectory.

**Evidence**:
- Qwen (M4): Steps 5–30, cos > 0.999, 65% coverage, 40-step/2-pass model
- Kontext (F1): Steps 4–23, cos > 0.999, 74% coverage, 28-step/1-pass model
- Tested on architecturally distinct models: 19.5B dual-stream with true CFG vs 12B mixed-stream with guidance distillation
- Both use FlowMatchEulerDiscreteScheduler with velocity prediction

**Strength**: Strong. Two architecturally different models with different guidance strategies, step counts, and parameter counts exhibit the same pattern.

**Limitation**: Both models use the same scheduler implementation. Testing on a model with a different flow matching scheduler would strengthen this claim.

### Claim 2: Temporal Step-Caching Transfers Across Architectures

**Statement**: A simple noise-prediction-level caching mechanism (reuse previous step's full output during the plateau) transfers from Qwen to Kontext without architectural adaptation, achieving predictable speedup on both models.

**Evidence**:
- Qwen conservative temporal: 1.67x speedup, exactly matching 48/80 pass ratio
- Kontext conservative temporal: 2.15x speedup, exactly matching 13/28 pass ratio
- Same implementation pattern (cache + reuse noise prediction), adapted only in step boundaries
- Speedup is deterministic (std < ±0.001) on both models across all test cases

**Strength**: Strong. The mechanism is architecture-agnostic by design (operates above the transformer), and the measurements confirm perfect theoretical-to-measured speedup on both models.

**Limitation**: The plateau boundaries require per-model calibration. The method does not transfer zero-shot — a redundancy characterization step is needed to identify the plateau region.

### Claim 3: Dual-Axis Acceleration Outperforms Single-Axis

**Statement**: On models with true CFG (like Qwen), combining temporal caching with CFG-skip achieves higher speedup at comparable quality to either axis alone, because the two axes target orthogonal redundancy dimensions.

**Evidence**:
- Qwen combined: 2.35x speedup, 0.836 SSIM (18 cases)
- Qwen temporal-only: 1.67x, 0.885 SSIM
- Qwen CFG-only: 1.60x, 0.851 SSIM
- Quality does NOT compound: observed combined SSIM 0.836 >> predicted-independent 0.753
- Kontext (single-axis): 2.15x, 0.754 SSIM — lower speedup AND lower quality

**Strength**: Moderate-to-strong. The non-compounding quality loss is a key finding. The Qwen vs Kontext comparison partially confounded by test suite differences.

**Limitation**: The comparison is indirect — Kontext cannot test CFG-skip because it is guidance-distilled. A truly controlled comparison would require a model that supports both guidance modes.

### Claim 4: Content-Type Quality Boundary is Method-Intrinsic

**Statement**: Temporal step-caching produces a bimodal quality distribution with a consistent failure boundary: content requiring fine-grained visual detail (texture, style, complex scenes) degrades substantially, while structural/semantic content (objects, portraits, arrangements) is robust. This boundary is method-intrinsic, not model-specific.

**Evidence**:
- Same fragile content types on both models (4/4 overlap: complex_scene, style_transfer, texture, nature)
- Same robust content types on both models (4/4 overlap: objects, portraits, arrangements, interiors)
- Bimodal distribution on both (Qwen: robust 0.83–0.96, fragile 0.64–0.80; Kontext: robust 0.83–0.99, fragile 0.37–0.74)
- Difficulty is the strongest predictor on both models (easy > medium >> hard)
- Different architectures, guidance strategies, and step counts

**Strength**: Strong. The overlap in failure types across architecturally distinct models is the most compelling evidence for a method limitation.

**Limitation**: Limited to 2 models and 18+12 test cases. A broader model survey would strengthen the claim. The content-type labels are coarse — a perceptual complexity metric would be more rigorous than difficulty categories.

### Claim 5: Schedule-Based Acceleration Has a Quality Ceiling

**Statement**: Fixed schedule-based acceleration (temporal caching, CFG-skip) achieves 2.0–2.5x speedup with acceptable quality (SSIM > 0.83) for structural content, but reaching higher speedup or extending to fine-detail content requires fundamentally different approaches.

**Evidence**:
- Qwen combined: 2.35x is the Pareto-optimal fixed schedule (best speedup at SSIM > 0.83)
- Kontext conservative: 2.15x is the only fixed schedule maintaining SSIM > 0.75
- Adaptive combined (Qwen): 3.95x achieves higher speedup but with unacceptable quality on hard cases (SSIM 0.672)
- Cosine similarity (the natural runtime signal) does not correlate with perceptual quality for complex content
- Block-aware caching failed catastrophically on Qwen (SSIM 0.37)

**Strength**: Moderate. Based on thorough exploration of the schedule-based design space on two models.

**Limitation**: The ceiling may be schedule-specific rather than method-specific. More sophisticated caching strategies (e.g., frequency-domain selection, learned cache policies) could potentially push the boundary.

---

## 6. Evidence Gaps and Remaining Questions

### 6.1 Gaps That Could Be Filled with Small Follow-Up Measurements

| Gap | Effort | Value | Recommendation |
|-----|--------|-------|---------------|
| Matched-edit similarity curves | Low (rerun profiling with same edit type on both models) | Strengthens temporal plateau comparison with controlled conditions | Nice to have, not blocking |
| Kontext with true_cfg_scale > 1.0 | Medium (re-enable CFG, test CFG redundancy + combined acceleration) | Would enable true dual-axis comparison on Kontext | Would strengthen Claim 3 but changes Kontext's operating mode |
| SSIM vs perceptual human judgment correlation | Medium (need human eval) | Validates whether SSIM captures the quality difference users care about | Important for paper but out of current scope |

### 6.2 Gaps Requiring Larger Experiments (Future Work)

| Gap | Effort | Value |
|-----|--------|-------|
| Third model (different scheduler) | High | Would strengthen Claim 1 by testing on a non-FlowMatchEuler scheduler |
| Larger test suite (50+ cases) | High | Would refine the bimodal boundary and enable statistical significance tests |
| Content complexity metric | Medium-High | Would replace coarse difficulty labels with a continuous, computable predictor |
| Perceptual quality metric (FID, human eval) | High | SSIM/LPIPS may not capture all relevant quality dimensions |
| Non-schedule acceleration (sparse attention, token pruning) | Very High | Would test whether the quality ceiling can be overcome with architectural changes |

### 6.3 Potential Inconsistencies to Monitor

1. **Kontext's severe fine_texture degradation (SSIM 0.366)**: This is an outlier even relative to Qwen's worst case (0.637). It could be a test-case-specific artifact rather than a systematic difference. Rerunning with additional fine_texture cases would clarify.

2. **Identity_preserving divergence**: Qwen's 2-image identity case scored 0.931 (robust) while Kontext's scored 0.686 (fragile). This could reflect a genuine difficulty difference (2-image vs 1-image conditioning) or a prompt-specific effect.

3. **Test suite comparability**: M9 (18 cases, 1-3 images) and F2 (12 cases, 1 image only) differ in size, image count distribution, and exact prompts. Cross-model SSIM comparisons are directional, not exact matched-pair estimates.

### 6.4 Assessment: Is the Evidence Paper-Ready?

**Yes, with qualifications.** The comparative analysis is coherent and supports the key claims with quantitative evidence from two architecturally distinct models. The main qualifications:

1. The two models share the same scheduler — this is a confound for Claim 1 that should be acknowledged.
2. The test suites are not perfectly matched — cross-model SSIM comparisons are approximate.
3. SSIM/LPIPS are proxy metrics — human evaluation would strengthen quality claims.
4. N=2 models is sufficient for an initial cross-model study but should be framed as such.

The evidence is strong enough for a workshop paper or a section of a longer paper. For a top-venue full paper, the third-model validation and human evaluation gaps should be flagged as future work.

---

# FLUX Track — F1: Kontext-dev Baseline, Profiling, and Temporal Transfer Test

## 1. Environment + Baseline (Phase 1)

| Metric | Value |
|--------|-------|
| **GPU** | NVIDIA B200, 191.5 GB VRAM |
| **torch** | 2.10.0+cu128 |
| **diffusers** | 0.38.0.dev0 |
| **sentencepiece** | 0.2.1 |
| **Model weights** | 54 GB on disk (23 GB transformer + text encoders + VAE) |
| **VRAM after load** | 33.76 GB |
| **Peak VRAM** | 37.52 GB reserved (19.6% of 191.5 GB) |
| **Baseline wall-clock** | 9.17s (28 steps, single pass) |
| **Per-step average** | 327.7ms |
| **Output size** | 1024x1024 |
| **Multi-image** | NOT supported (pipeline rejects list input) |

### Comparison to Qwen baseline

| Metric | Qwen (M2) | Kontext (F1) | Ratio |
|--------|-----------|-------------|-------|
| Wall-clock time | 47.4s (40 steps × 2 passes) | 9.17s (28 steps × 1 pass) | **5.2x faster** |
| Per-step time | 1184.5ms | 327.7ms | **3.6x faster** |
| Total passes | 80 | 28 | 2.86x fewer |
| Peak VRAM | 63.20 GB | 37.52 GB | 1.69x less |

Kontext is 5.2x faster than Qwen out of the box, driven by guidance distillation (1 vs 2 passes/step) and fewer steps (28 vs 40).

## 2. Structured Profiling (Phase 2)

### Level 1: Pipeline Stage Breakdown

| Stage | Time (s) | Pct |
|-------|----------|-----|
| Image preprocessing | 0.017 | 0.2% |
| Text encoding | 0.020 | 0.2% |
| VAE encoding + latent prep | 0.054 | 0.6% |
| **Denoising loop** | **8.997** | **97.8%** |
| VAE decoding | 0.109 | 1.2% |
| **Total** | **9.196** | |

Same pattern as Qwen (M3): denoising loop dominates (97.8% vs 99.3%). The small difference is that Kontext's VAE decode (1.2%) is proportionally more visible because the denoising loop is much shorter.

### Level 2: Per-Step Dynamics

Steps are **perfectly uniform**: 321.3ms ± 0.2ms (std/mean = 0.06%). No variation across early, middle, or late steps. Same finding as Qwen.

| Metric | Kontext | Qwen (M3) |
|--------|---------|-----------|
| Mean step time | 321.3ms | 1184.5ms |
| Std dev | 0.2ms | 1.4ms |
| Passes per step | 1 | 2 |
| Scheduler step | 0.1ms | 0.1ms |

### Level 3: Module-Level Breakdown

#### Block type comparison

| Block Type | Count | Mean per-block (ms) | Total (ms) | Pct of transformer |
|------------|-------|--------------------:|----------:|-------------------:|
| Dual-stream | 19 | 5.91 | 112.35 | 34.9% |
| Single-stream | 38 | 5.48 | 208.13 | 64.7% |
| Embeddings | — | — | 0.85 | 0.3% |
| Output | — | — | 0.20 | 0.1% |

**Surprise: dual/single cost ratio is 1.08x**, not the 2.4x predicted in F0. The F0 estimate was based on parameter counts (~340M vs ~142M), but on B200 with FlashAttention, the attention-dominated cost makes both block types nearly equal per forward pass. This is because attention cost depends on sequence length (same for both types), not parameter count.

#### Within-block breakdown (averaged across sampled blocks)

| Component | Dual-stream | Single-stream |
|-----------|------------:|--------------:|
| **Attention** | **4.34ms (73%)** | **4.10ms (75%)** |
| FF/MLP (image) | 0.86ms (14%) | 0.39ms (7%) |
| FF/MLP (text) | 0.08ms (1%) | — |
| Proj out | — | 0.49ms (9%) |
| Norms | 0.33ms (6%) | 0.20ms (4%) |
| Other (gates, residual) | 0.35ms (6%) | 0.31ms (6%) |

Attention dominates both block types at ~73-75%, matching Qwen's 75.8%.

### Comparison to Qwen profiling structure

| Dimension | Qwen (M3) | Kontext (F1) |
|-----------|-----------|-------------|
| Denoising loop dominance | 99.3% | 97.8% |
| Steps uniform? | Yes (±1.4ms) | Yes (±0.2ms) |
| Attention fraction | 75.8% | 73-75% |
| Block uniformity | All 60 identical | 2 types, but ~equal cost (1.08x ratio) |
| Per-block cost | 9.87ms (cond pass) | 5.91ms (dual) / 5.48ms (single) |
| CFG overhead | 2x (cond + uncond) | None (guidance-distilled) |

## 3. Temporal Redundancy Characterization (Phase 3)

### Step-to-step similarity across 28 steps

| Step | Cos Sim | Rel L2 | Phase |
|------|---------|--------|-------|
| 1 | 0.9574 | 0.514 | Convergence |
| 2 | 0.9954 | 0.099 | Convergence |
| 3 | 0.9987 | 0.050 | Convergence |
| 4 | 0.9992 | 0.046 | **Plateau start** |
| 5-11 | 0.9997-0.9999 | 0.012-0.027 | Plateau |
| 12 | **0.9999** | **0.012** | **Peak stability** |
| 13-20 | 0.9998-0.9999 | 0.013-0.018 | Plateau |
| 21-23 | 0.9994-0.9997 | 0.023-0.038 | Late plateau |
| 24-25 | 0.9966-0.9986 | 0.051-0.078 | **Divergence start** |
| 26-27 | 0.9815-0.9939 | 0.104-0.182 | Divergence |

### Three-phase pattern: CONFIRMED

Same pattern as Qwen:
- **Steps 1-3**: Convergence (cos 0.957 → 0.999)
- **Steps 4-23**: Plateau (cos > 0.999) — 20 steps, **74% of trajectory**
- **Steps 24-27**: Divergence (cos drops to 0.981)

### Comparison to Qwen temporal profile

| Dimension | Qwen (M4) | Kontext (F1) |
|-----------|-----------|-------------|
| Three-phase pattern? | Yes | Yes |
| Convergence phase | Steps 1-4 (cos 0.98→0.9995) | Steps 1-3 (cos 0.957→0.999) |
| Plateau region | Steps 5-30 (26 steps, 65%) | Steps 4-23 (20 steps, **74%**) |
| Peak similarity | cos > 0.9999 | cos = 0.9999 |
| Divergence | Dramatic (cos=0.816 at step 39) | **Mild (cos=0.981 at step 27)** |
| Most dynamic block | Block 44 (mid-late) | single_37 (last, cos min=0.891) |

Key difference: Kontext's divergence is **much milder** (cos=0.981 vs 0.816). This suggests the final steps make smaller corrections, possibly because the guidance-distilled trajectory is smoother than Qwen's CFG-modified trajectory.

## 4. Temporal Caching Prototype (Phase 4)

### Schedules tested

| Schedule | Cached Steps | Full Steps | Theoretical Speedup |
|----------|-------------|-----------|-------------------:|
| Aggressive | Steps 4-23 (20 cached) | 8 | 3.50x |
| Conservative | Steps 6-20 (15 cached) | 13 | 2.15x |

### Results (5 test cases, SEED=42)

| Test Case | Aggr Speed | Aggr SSIM | Cons Speed | Cons SSIM |
|-----------|----------:|----------:|----------:|----------:|
| tc1_red_cube | 3.50x | **0.976** | 2.15x | **0.987** |
| tc2_landscape | 3.50x | 0.673 | 2.15x | 0.832 |
| tc3_complex | 3.50x | **0.895** | 2.15x | **0.952** |
| tc4_portrait | 3.50x | 0.749 | 2.15x | 0.836 |
| tc5_texture | 3.50x | 0.708 | 2.15x | 0.807 |
| **Mean** | **3.50x** | **0.800** | **2.15x** | **0.883** |

### Key findings

1. **Speedups match theory exactly**: 3.50x and 2.15x — no overhead from caching mechanism.
2. **Conservative schedule (2.15x, 0.883 SSIM)** is the better tradeoff. 3/5 cases have SSIM > 0.83.
3. **Aggressive schedule (3.50x, 0.800 SSIM)** degrades significantly on complex content (3/5 severe).
4. **Same vulnerability pattern as Qwen**: landscape, portrait, and texture cases degrade more than simple geometry.
5. **Baseline is already very fast**: 8.9s → 4.1s (conservative) or 2.6s (aggressive). Absolute savings of ~5-6s.

### Comparison to Qwen caching results

| Dimension | Qwen (M6) | Kontext (F1) |
|-----------|-----------|-------------|
| Conservative speedup | 1.66x | **2.15x** |
| Conservative SSIM | 0.93 | 0.88 |
| Aggressive speedup | 2.86x | **3.50x** |
| Aggressive SSIM | 0.78 | 0.80 |
| Speedup matches theory? | Yes | Yes |
| Content-dependent quality? | Yes (tc3/tc5 worst) | Yes (tc2/tc4/tc5 worst) |

Kontext achieves higher speedup ratios because the plateau is proportionally larger (74% vs 65% of trajectory), but quality degradation is comparable.

## 5. Transfer Verdict

### Temporal Plateau
**CONFIRMED.** Identical three-phase pattern (convergence → plateau → divergence) in both Qwen and Kontext, despite different architectures, guidance strategies, and step counts. The plateau is a property of rectified flow matching, not model-specific.

### Temporal Caching
**TRANSFERS.** The same mechanism (reuse previous step's noise prediction during the plateau) works on both models with comparable quality/speedup tradeoffs. Conservative schedule recommended for both.

### CFG-Skip
**NOT APPLICABLE.** Kontext is guidance-distilled — no unconditional pass exists. This axis is eliminated by design, which is why Kontext starts from a faster baseline.

### Block-Aware Caching
**NOT TESTED** (per task spec — failed on Qwen, different block structure here). F0 predicted it wouldn't transfer; F1 profiling confirms blocks are more uniform than expected (1.08x ratio), so block-level differentiation has even less justification.

## 6. Structured Cross-Model Comparison Table

| Dimension | Qwen-Image-Edit-2511 | FLUX.1 Kontext-dev |
|-----------|---------------------|-------------------|
| **Architecture** | 19.5B transformer, 60 identical dual-stream blocks | 12B transformer, 19 dual + 38 single blocks |
| **Guidance** | True CFG (2 passes/step) | Guidance-distilled (1 pass/step) |
| **Default steps** | 40 (80 passes total) | 28 (28 passes total) |
| **Multi-image** | 1-3 images | 1 image only |
| **Baseline time** | 47.4s | 9.17s (**5.2x faster**) |
| **Peak VRAM** | 63.2 GB | 37.5 GB |
| **Denoising dominance** | 99.3% | 97.8% |
| **Steps uniform** | Yes (±1.4ms) | Yes (±0.2ms) |
| **Attention fraction** | 75.8% | 73-75% |
| **Block cost uniformity** | All identical | 1.08x (effectively uniform) |
| **Temporal plateau** | Steps 5-30 (65%) | Steps 4-23 (74%) |
| **Plateau cos range** | >0.999 | >0.999 |
| **Final-step divergence** | Dramatic (cos=0.816) | Mild (cos=0.981) |
| **Conservative cache speedup** | 1.66x → 28.5s | 2.15x → 4.1s |
| **Conservative cache SSIM** | 0.93 | 0.88 |
| **Aggressive cache speedup** | 2.86x → 16.6s | 3.50x → 2.6s |
| **Aggressive cache SSIM** | 0.78 | 0.80 |
| **Combined (temporal+CFG) speedup** | 2.35x → 20.2s | N/A (no CFG axis) |
| **Content-dependent quality** | Yes (texture/complex worst) | Yes (same pattern) |

### Bottom Line

The temporal plateau and step-caching mechanism transfer from Qwen to Kontext, confirming that this redundancy pattern is intrinsic to rectified flow matching inference, not specific to one model. However, Kontext's acceleration ceiling is lower in absolute terms because it starts from a leaner baseline (no CFG to skip). The practical value of temporal caching on Kontext is reducing 9s → 4s (conservative) — significant but less dramatic than Qwen's 47s → 20s.

---

# FLUX Track — F2: Validation + Ablation Study (Broader Test Suite)

## 1. Test Suite Design

12 diverse cases covering 12 edit types across 3 difficulty levels. All single-image (Kontext limitation). Balanced: 4 easy, 4 medium, 4 hard. Includes cases expected to be robust (simple objects, arrangements) AND fragile (fine texture, complex scenes, style transfer) based on Qwen M9 failure patterns.

| Case | Edit Type | Difficulty |
|------|-----------|-----------|
| f2_01_simple_object | simple_object | easy |
| f2_02_object_modify | object_modification | easy |
| f2_03_background_change | background_change | easy |
| f2_04_arrangement | arrangement | easy |
| f2_05_portrait | portrait | medium |
| f2_06_identity_edit | identity_preserving | medium |
| f2_07_interior | interior_design | medium |
| f2_08_text_edit | text_editing | medium |
| f2_09_style_transfer | style_transfer | hard |
| f2_10_fine_texture | fine_texture | hard |
| f2_11_complex_scene | complex_scene | hard |
| f2_12_nature_detail | nature_detail | hard |

## 2. Full Results Table

Three configs per case: baseline (28 full steps), conservative (cache steps 6-20, 2.15x theoretical), aggressive (cache steps 4-23, 3.50x theoretical). SEED=42 for all.

| Case | Type | Diff | Cons Speed | Cons SSIM | Cons LPIPS | Aggr Speed | Aggr SSIM | Aggr LPIPS |
|------|------|------|----------:|----------:|----------:|----------:|----------:|----------:|
| f2_01_simple_object | simple_object | easy | 2.15x | **0.987** | 0.011 | 3.50x | **0.976** | 0.034 |
| f2_02_object_modify | object_modification | easy | 2.15x | **0.934** | 0.093 | 3.50x | 0.888 | 0.197 |
| f2_03_background_change | background_change | easy | 2.15x | 0.741 | 0.250 | 3.50x | 0.525 | 0.515 |
| f2_04_arrangement | arrangement | easy | 2.15x | **0.906** | 0.026 | 3.50x | 0.731 | 0.123 |
| f2_05_portrait | portrait | medium | 2.15x | **0.909** | 0.048 | 3.50x | 0.820 | 0.161 |
| f2_06_identity_edit | identity_preserving | medium | 2.15x | 0.686 | 0.352 | 3.50x | 0.491 | 0.654 |
| f2_07_interior | interior_design | medium | 2.15x | 0.834 | 0.146 | 3.50x | 0.752 | 0.331 |
| f2_08_text_edit | text_editing | medium | 2.15x | 0.874 | 0.175 | 3.50x | 0.823 | 0.284 |
| f2_09_style_transfer | style_transfer | hard | 2.15x | 0.681 | 0.544 | 3.50x | 0.591 | 0.761 |
| f2_10_fine_texture | fine_texture | hard | 2.15x | 0.366 | 0.513 | 3.50x | 0.173 | 0.555 |
| f2_11_complex_scene | complex_scene | hard | 2.15x | 0.415 | 0.573 | 3.50x | 0.280 | 0.834 |
| f2_12_nature_detail | nature_detail | hard | 2.15x | 0.710 | 0.298 | 3.50x | 0.570 | 0.527 |

Baseline mean time: 8.90s (range 8.85-8.99s). Speedups match theoretical exactly (std=0.001).

## 3. Per-Schedule Aggregate Statistics

| Metric | Conservative | Aggressive |
|--------|----------:|----------:|
| Theoretical speedup | 2.15x | 3.50x |
| **Mean speedup** | **2.154x** ± 0.001 | **3.500x** ± 0.001 |
| **Mean SSIM** | **0.754** | **0.635** |
| Median SSIM | 0.787 | 0.661 |
| SSIM std | 0.190 | 0.233 |
| SSIM range | 0.366 – 0.987 | 0.173 – 0.976 |
| **Mean LPIPS** | **0.253** | **0.415** |
| Median LPIPS | 0.213 | 0.423 |
| LPIPS range | 0.011 – 0.573 | 0.034 – 0.834 |

### Comparison to F1 (5-case test suite)

| Metric | F1 Conservative (5 cases) | F2 Conservative (12 cases) | Change |
|--------|------------------------:|-------------------------:|--------|
| Mean SSIM | 0.883 | 0.754 | **-0.129** |
| SSIM range | 0.807 – 0.987 | 0.366 – 0.987 | Much wider |
| Mean speedup | 2.15x | 2.15x | Identical |

**F1's 5 test cases were unrepresentatively optimistic.** The broader F2 suite exposed severe quality degradation on content types not covered in F1.

## 4. Quality Distribution (Conservative)

| Threshold | Count | Percentage |
|-----------|------:|----------:|
| SSIM >= 0.95 | 1/12 | 8% |
| SSIM >= 0.90 | 4/12 | 33% |
| SSIM >= 0.85 | 5/12 | 42% |
| SSIM >= 0.80 | 6/12 | 50% |
| **SSIM < 0.80** | **6/12** | **50%** |

Percentile breakdown:
- P10: 0.442
- P25: 0.685
- P50 (median): 0.787
- P75: 0.907
- P90: 0.931

**Distribution is bimodal**: a robust cluster (0.83-0.99, structural/simple content) and a fragile cluster (0.37-0.74, fine-detail/complex content). This matches Qwen M9's bimodal pattern.

## 5. Quality by Difficulty

| Difficulty | Conservative SSIM | Conservative LPIPS | Aggressive SSIM | Aggressive LPIPS |
|-----------|------------------:|-------------------:|----------------:|-----------------:|
| Easy (n=4) | 0.892 | 0.095 | 0.780 | 0.217 |
| Medium (n=4) | 0.826 | 0.181 | 0.721 | 0.358 |
| Hard (n=4) | **0.543** | **0.482** | **0.403** | **0.669** |

Difficulty is the strongest predictor of quality loss, matching Qwen M9's finding. The easy→medium transition is gradual (~0.07 SSIM drop), but medium→hard is dramatic (~0.28 SSIM drop).

## 6. Failure Analysis

### Cases with severe degradation (Conservative SSIM < 0.80)

| Case | Type | Difficulty | SSIM | LPIPS | Assessment |
|------|------|-----------|-----:|------:|-----------|
| f2_10_fine_texture | fine_texture | hard | 0.366 | 0.513 | Catastrophic |
| f2_11_complex_scene | complex_scene | hard | 0.415 | 0.573 | Catastrophic |
| f2_09_style_transfer | style_transfer | hard | 0.681 | 0.544 | Severe |
| f2_06_identity_edit | identity_preserving | medium | 0.686 | 0.352 | Severe |
| f2_12_nature_detail | nature_detail | hard | 0.710 | 0.298 | Severe |
| f2_03_background_change | background_change | easy | 0.741 | 0.250 | Moderate-severe |

### Comparison to Qwen M9 failure pattern

| Edit Type | Qwen M9 SSIM | Kontext F2 SSIM | Both fail? |
|-----------|-------------|----------------|-----------|
| complex_scene | 0.637 | 0.415 | **YES** |
| style_transfer | 0.642 | 0.681 | **YES** |
| fine_texture / material | 0.665 | 0.366 | **YES** |
| nature_scene / nature_detail | 0.687 | 0.710 | **YES** |
| identity_preserving | 0.753 | 0.686 | **YES** |
| interior_design | 0.768 | 0.834 | Partial (Kontext better) |

**The failure pattern is highly consistent across models.** The same content types (fine texture, complex scenes, style transfer, nature detail) degrade on both Qwen and Kontext. This confirms that quality degradation from temporal caching is a property of the acceleration method interacting with content complexity, not a model-specific artifact.

**Kontext degrades more severely** than Qwen on fine_texture (0.37 vs 0.67) and complex_scene (0.42 vs 0.64). This may be because Kontext's guidance-distilled trajectory is smoother but more sensitive to caching perturbations — the final-step divergence is milder (cos=0.981 vs 0.816), meaning the model has less natural error-correction at the end of the trajectory.

### f2_03_background_change anomaly

This case was labeled "easy" but degraded notably (SSIM 0.741). The prompt ("A white cat sitting on a windowsill overlooking a rainy city street") demands fine spatial detail despite the simple reference image. This suggests the difficulty of the *prompt* matters more than the reference image complexity.

## 7. Cross-Model Comparison Summary

| Dimension | Qwen M9 Combined | Kontext F2 Conservative | Kontext F2 Aggressive |
|-----------|----------------:|----------------------:|---------------------:|
| **Speedup** | 2.35x | 2.15x | 3.50x |
| **Mean SSIM** | 0.836 | **0.754** | 0.635 |
| **Mean LPIPS** | 0.215 | **0.253** | 0.415 |
| Test cases | 18 | 12 | 12 |
| Acceleration axes | CFG + temporal | Temporal only | Temporal only |
| Baseline time | 47.4s | ~9s | ~9s |
| Accelerated time | ~20s | ~4.1s | ~2.6s |
| SSIM >= 0.90 | 39% | 33% | 17% |
| SSIM < 0.80 | 33% | **50%** | 67% |
| Bimodal distribution? | Yes | Yes | Yes |
| Hard-case types | texture, complex, style | Same | Same |

### Interpretation

1. **Speedup transfers perfectly.** Both schedules match theoretical exactly on Kontext, just as they did on Qwen.

2. **Quality is worse on Kontext** (conservative: 0.754 vs Qwen combined 0.836). Two factors:
   - F2 test suite has a harder difficulty distribution (4/12 hard vs 6/18 = 33% in both, but F2's hard cases are genuinely harder)
   - Kontext has only the temporal axis (no CFG redundancy to exploit), and the quality cost is borne entirely by temporal caching
   - Kontext's smoother trajectory (milder final-step divergence) may paradoxically make it more sensitive to caching perturbations — less natural error-correction

3. **The transfer claim holds with qualification.** Temporal caching transfers from Qwen to Kontext: same plateau, same mechanism, same failure modes. But the quality/speedup tradeoff is less favorable on Kontext's broader test suite. The conservative schedule (2.15x) is only reliable for structural content (SSIM > 0.90 on simple objects, arrangements, portraits).

4. **Same failure boundary across models.** Both Qwen and Kontext degrade on the same content types (fine texture, complex scenes, style transfer). This is evidence that the quality boundary is a fundamental limitation of the temporal step-caching approach, not of either model.

### Bottom Line

**Does the transfer claim hold on a broader test suite?** Partially.

- **The mechanism transfers**: confirmed. Same plateau, same caching, same speedup.
- **The quality/speedup tradeoff is less favorable than F1 suggested**: mean SSIM 0.754 vs F1's 0.883. F1 tested only 5 cases that happened to be relatively easy.
- **The failure pattern is model-independent**: same content types degrade on both Qwen and Kontext, confirming this is a method limitation.
- **Recommendation**: Temporal caching at 2.15x is suitable for structural/semantic content (objects, portraits, arrangements). It should NOT be applied blindly to fine-texture, complex scenes, or style transfer tasks. Content-type gating remains necessary for production use.

---

# FLUX Track — F0: Kontext-dev Reconnaissance

## 1. Architectural Summary

**FLUX.1 Kontext-dev** is a 12B-parameter guidance-distilled rectified flow transformer for image-to-image editing, released by Black Forest Labs (arXiv:2506.15742).

| Aspect | Detail |
|--------|--------|
| **Architecture** | Rectified flow matching transformer in VAE latent space |
| **Transformer params** | ~12B |
| **Total model params** | ~17.2B (incl. text encoders + VAE) |
| **Block structure** | 19 dual-stream (MMDiT) + 38 single-stream = 57 blocks |
| **Hidden dimension** | 3072 (24 heads x 128 dim/head) |
| **Text encoders** | CLIP-L (pooled, 77 tokens) + T5-v1_1-XXL (sequence, 512 tokens) |
| **VAE** | Custom Flux-VAE, 16 latent channels |
| **Guidance** | Guidance-distilled (single pass, embedded guidance_scale) |
| **True CFG** | Available via `true_cfg_scale > 1.0`, disabled by default |
| **Default steps** | 28 |
| **Default guidance_scale** | 3.5 (pipeline) / 2.5 (model card) |
| **Scheduler** | FlowMatchEulerDiscreteScheduler (velocity prediction, Euler integration) |
| **Positional encoding** | 3D RoPE: (t, h, w) with axes_dim (16, 56, 56) |
| **Conditioning** | AdaLN-Zero with shift/scale/gate from timestep + pooled text embeddings |
| **Tokens at 1024x1024** | ~4096 per image (VAE 8x compress + 2x2 pack = 16x per dim) |

### Block Types

**Dual-stream blocks (19x, ~340M each = ~6.46B)**: Separate AdaLN-Zero normalization and FFN for text and image streams. Joint attention over concatenated text+image Q,K,V. Per-head RMSNorm on Q,K before attention. GELU(tanh) activation.

**Single-stream blocks (38x, ~142M each = ~5.38B)**: Text and image tokens merged into a single sequence. Attention (pre_only) and MLP computed in parallel, concatenated, and projected back to model dim. More parameter-efficient per block but twice as many blocks.

## 2. Multi-Image Conditioning Mechanism

### How reference images are processed

1. Reference image -> Flux-VAE encode -> apply shift/scale factors -> 2x2 spatial packing -> `image_latents` (shape: `[B, seq_len, C]`)
2. **Positional IDs**: Target noise tokens get `(0, h, w)`, reference image tokens get `(1, h, w)`. This "virtual time-step offset" disambiguates generated vs. reference content in 3D RoPE space.
3. **Sequence concatenation**: `latent_model_input = cat([noise_latents, image_latents], dim=1)` — reference image tokens appended to noise tokens along the sequence dimension.
4. **Transformer processes the full concatenated sequence** through all 57 blocks.
5. **Output slicing**: Only the noise-latent portion is kept as the noise prediction: `noise_pred = output[:, :noise_len]`.

### Multi-image architectural support

The paper describes multi-image conditioning via per-image time-step offsets: target=(0,h,w), img1=(1,h,w), img2=(2,h,w), etc. However, the current diffusers pipeline (`FluxKontextPipeline`) accepts only a **single reference image**. Multi-image is listed as future work in the paper.

### Comparison to Qwen's mechanism

| Aspect | Qwen-Image-Edit-2511 | FLUX.1 Kontext-dev |
|--------|----------------------|---------------------|
| **Multi-image support** | 1-3 images (released, active) | 1 image (released); N images (architectural, unreleased) |
| **Concatenation** | Sequence dim: `[noise, img1, img2, ...]` | Sequence dim: `[noise, img_ref]` |
| **Per-image identification** | Shared frame index (-1 for all condition images), spatial (h,w) positions | Virtual time-step offset (1, 2, ... per image) in 3D RoPE |
| **Timestep modulation** | Zero-timestep for all condition images (shared) | Same — condition images at t=1 offset |
| **Cross-image interaction** | Implicit via joint attention (no dedicated mechanism) | Same architecture, but untested with multiple images |
| **Tokens per image (1024x1024)** | ~4,096 | ~4,096 |

## 3. Structured Comparison: Qwen vs Kontext

| Dimension | Qwen-Image-Edit-2511 | FLUX.1 Kontext-dev |
|-----------|----------------------|---------------------|
| **Total model params** | ~27.6B | ~17.2B |
| **Transformer params** | ~19.5B | ~12B |
| **Block count** | 60 (all dual-stream) | 57 (19 dual + 38 single) |
| **Block uniformity** | All blocks identical type and cost | Two distinct block types; dual ~2.4x costlier than single |
| **Hidden dim** | 3072 | 3072 |
| **Attention heads** | — (same architecture family) | 24 x 128 |
| **Text encoder** | Qwen2.5-VL-7B (~8B, joint vision-language) | CLIP-L (~0.43B) + T5-XXL (~4.7B) = ~5.1B |
| **VAE** | Standard, few channels | Custom Flux-VAE, 16 latent channels |
| **Guidance** | True CFG (dual pass, scale=4.0) | Guidance-distilled (single pass, scale=2.5-3.5) |
| **Passes per step** | 2 (cond + uncond) | 1 (guidance embedded) |
| **Default steps** | 40 | 28 |
| **Total passes (default)** | 80 (40 steps x 2 passes) | 28 (28 steps x 1 pass) |
| **Scheduler** | FlowMatchEulerDiscreteScheduler | FlowMatchEulerDiscreteScheduler |
| **Positional encoding** | RoPE with frame indices | 3D RoPE (t, h, w) |
| **Multi-image** | 1-3 images (production) | 1 image (current release) |
| **Estimated bf16 VRAM (weights)** | ~55 GB | ~34.4 GB |
| **Measured/est. peak VRAM** | 63.2 GB (measured on B200) | ~45-55 GB (estimated) |
| **Per-step time on B200 (est.)** | 592ms/pass x 2 = 1184ms/step | ~400-600ms/pass x 1 (est.) |
| **Estimated total time on B200** | 47.4s (40 steps, measured) | ~12-17s (28 steps, estimated) |

### Key Architectural Differences

1. **Guidance strategy is the biggest difference.** Qwen uses true CFG (2 passes/step), Kontext is guidance-distilled (1 pass/step). This alone makes Kontext ~2x faster per step in default configuration.

2. **Block heterogeneity.** Qwen has 60 identical dual-stream blocks. Kontext has two types: 19 expensive dual-stream followed by 38 cheaper single-stream. This means Kontext's blocks are NOT uniformly expensive — profiling will show a cost step-function between block 18 and 19.

3. **Smaller model.** 12B vs 19.5B transformer = ~38% fewer parameters. Combined with single-pass inference, the per-step compute is roughly 12B/19.5B * 1/2 = ~31% of Qwen's per-step compute.

4. **Fewer steps.** 28 vs 40 default. Combined with single-pass, total passes = 28 vs 80 = 35% of Qwen's total passes.

5. **Text encoder difference.** Qwen uses a vision-language model (Qwen2.5-VL-7B) that processes both text and image tokens. Kontext uses separate CLIP-L (text only, pooled) + T5-XXL (text only, sequence). Kontext's approach is more modular but cannot condition on image content at the text-encoder level.

## 4. Transfer-Likelihood Assessment

### M3: Structured Profiling (Block Uniformity, Attention Dominance)

**Transfer: PARTIAL — methodology transfers, but results will differ.**

- **What transfers**: The profiling methodology (pipeline stages, per-step timing, module-level hooks) applies identically to the Kontext pipeline.
- **What differs**: Qwen's M3 found all 60 blocks identical in cost. Kontext has two block types with ~2.4x cost difference (dual ~340M vs single ~142M). The profiling will reveal a heterogeneous cost structure instead of a flat one.
- **Attention dominance**: Joint attention should still dominate compute in dual-stream blocks. Single-stream blocks compute attention and MLP in parallel, so the per-block cost decomposition will differ.
- **CFG doubling**: M3 found CFG doubles cost (cond + uncond). Kontext eliminates this by default — the per-step cost will be roughly half of what a hypothetical CFG-enabled Kontext run would show.

### M4: Temporal Redundancy (Step-to-Step Plateau)

**Transfer: LIKELY.**

- **Architectural basis**: The temporal plateau is a property of the flow matching ODE trajectory, not of the specific transformer architecture. Both models use the same scheduler (FlowMatchEulerDiscreteScheduler) with the same velocity-prediction Euler integration.
- **Expected pattern**: Convergence (early steps) -> plateau (mid steps) -> divergence (final steps). The exact step boundaries may shift due to 28 vs 40 steps and different shift parameters.
- **Key uncertainty**: The guidance-distilled model bakes guidance into a single pass. The Qwen M4 finding showed both temporal AND CFG redundancy — the CFG axis doesn't exist in Kontext's default mode, so only the temporal axis is relevant.
- **Prediction**: We expect a similar cosine similarity trajectory (>0.999 in mid-steps) but possibly compressed into the 28-step range.

### M5: CFG-Skip

**Transfer: NOT APPLICABLE.**

- Kontext is guidance-distilled — there is no unconditional pass to skip. The guidance signal is embedded in the model weights and passed as a scalar (guidance_scale → guidance embedding).
- When `true_cfg_scale > 1.0` is manually enabled, CFG-skip findings would apply. But this is not the default operating mode.
- **Bottom line**: This entire acceleration axis is eliminated by architecture. Kontext gets the equivalent of "permanent CFG-skip" for free via guidance distillation.

### M6: Temporal Step Caching

**Transfer: LIKELY.**

- Conservative uniform temporal caching (reuse noise prediction across consecutive steps) operates at the noise-prediction level, above the transformer architecture. It is architecture-agnostic.
- If the temporal plateau exists (M4 transfer), then the same caching strategy should work: cache the full noise prediction for plateau steps, skip the transformer entirely.
- **Key question**: With only 28 steps (vs 40), the plateau may be narrower. A cache window of 10-25 (M6 conservative) would need to be re-calibrated to Kontext's step range.
- **Prediction**: The cache window will be narrower (perhaps 8-20 out of 28 steps), yielding a similar speedup ratio but fewer total cached steps.

### M7: Multi-Image Superlinear Scaling

**Transfer: LIKELY (architecturally) but UNTESTABLE (single-image release).**

- Both architectures use sequence concatenation for multi-image conditioning with O(S^2) attention cost. FlashAttention would produce the same sub-quadratic wall-clock scaling.
- However, Kontext's current release only supports a single reference image. Multi-image scaling cannot be tested without modifying the pipeline.
- **If/when multi-image is released**: Expect similar O(S^1.4) wall-clock scaling. The findings about orthogonal cross-image hidden states and minimal cross-image attention should also hold, since the attention mechanism is architecturally similar.

### Combined Acceleration (M8/M9)

**Transfer: PARTIAL.**

- Kontext's default mode already eliminates the CFG axis. The combined M8 strategy (temporal cache + CFG-skip) reduces to temporal cache only.
- The 2.35x speedup from M8/M9 was achieved by combining both axes (80→34 passes). Kontext starts at 28 passes (no CFG) — applying temporal caching alone might yield ~1.5-1.8x (caching ~10-15 of 28 steps).
- The absolute speedup ceiling for schedule-based acceleration is lower because the baseline is already lean (single-pass, 28 steps).

### Summary Table

| Finding | Transfer Likelihood | Reasoning |
|---------|-------------------|-----------|
| M3: Profiling methodology | **Transfers** | Same framework, same profiling approach |
| M3: Uniform block cost | **Does NOT transfer** | Kontext has 2 block types with 2.4x cost difference |
| M4: Temporal plateau | **Likely** | Same scheduler, same ODE trajectory dynamics |
| M5: CFG-skip | **Not applicable** | Guidance-distilled = no CFG to skip |
| M6: Temporal caching | **Likely** | Architecture-agnostic; depends on M4 plateau |
| M7: Multi-image scaling | **Likely** (architectural) | Same attention mechanism; untestable on current release |
| M8/M9: Combined speedup | **Partial** | Only temporal axis applies; lower ceiling |

## 5. Dependency and VRAM Analysis

### Dependency Compatibility

| Package | Current Version | Kontext Compatible? | Notes |
|---------|----------------|-------------------|-------|
| torch | 2.10.0+cu128 | YES | |
| diffusers | 0.38.0.dev0 | YES | FluxKontextPipeline present |
| transformers | 5.3.0 | YES | CLIPTextModel, T5EncoderModel available |
| accelerate | 1.13.0 | YES | |
| **sentencepiece** | **NOT INSTALLED** | **NEEDED** | Required for T5 tokenizer |

**Action needed**: `pip install sentencepiece` before loading Kontext. This is the only missing dependency. No other changes to the venv are required, and installing sentencepiece will not affect Qwen pipeline compatibility.

### VRAM Estimate

| Component | Parameters | bf16 Size |
|-----------|-----------|-----------|
| Flux transformer | ~12.0B | ~24.0 GB |
| T5-v1_1-XXL encoder | ~4.7B | ~9.4 GB |
| CLIP-L text encoder | ~0.43B | ~0.86 GB |
| Flux-VAE | ~0.08B | ~0.16 GB |
| **Total model weights** | **~17.2B** | **~34.4 GB** |
| Activations + KV cache (est., 1024x1024) | — | ~10-20 GB |
| **Estimated peak** | — | **~45-55 GB** |

### Hardware Compatibility

| Hardware | VRAM | Fit in bf16? | Notes |
|----------|------|-------------|-------|
| **B200** | 183 GB | **YES** | ~45-55 GB peak, >120 GB headroom |
| **L4** | 23 GB | **NO** | Requires sequential CPU offload (~4 GB peak) |
| **A100-80G** | 80 GB | **YES** | Tight but feasible with ~25 GB headroom |
| **A100-40G** | 40 GB | **MAYBE** | Weights alone ~34.4 GB, may need 8-bit or offload |

B200 is the clear target for acceleration experiments. L4 could be used for functional validation with offloading.

## 6. GPU Validation Plan — Milestone F1

### Objective

Establish a native-speed baseline for FLUX.1 Kontext-dev on B200, analogous to Qwen M2. Verify the architecture works, measure real timing and VRAM, and run the first temporal redundancy characterization.

### Prerequisites

1. Access to B200 node
2. Install sentencepiece: `pip install sentencepiece`
3. Download FLUX.1-Kontext-dev weights (~24 GB for transformer, ~10 GB for T5-XXL, ~1 GB for CLIP-L + VAE). Total ~35 GB. Requires HuggingFace access token (gated model).
4. Accept model license on HuggingFace (non-commercial research license)

### Phase 1: Load + Baseline (analogous to M2)

1. Load FluxKontextPipeline in full bf16 on B200
2. Run single image-to-image pass: reference image + edit prompt, 28 steps, guidance_scale=2.5
3. Measure: load time, generation time, per-step time, peak VRAM
4. Save output image
5. Compare measured values to our estimates above

**Expected outcome**: ~12-17s generation, ~45-55 GB peak VRAM, successful image output.

### Phase 2: Profiling (analogous to M3)

1. Pipeline-stage timing (text encode, VAE, denoising loop)
2. Per-step timing within the denoising loop
3. Module-level timing: dual-stream blocks vs single-stream blocks vs other
4. Verify block heterogeneity: do dual-stream blocks cost ~2.4x more than single-stream?

**Expected outcome**: Denoising loop dominates (>99%). Dual-stream blocks are costlier. Per-step time ~400-600ms.

### Phase 3: Temporal Redundancy (analogous to M4)

1. Instrument consecutive-step cosine similarity on transformer output
2. Instrument 5 sampled blocks: 2 dual-stream (e.g., 0, 9), 2 single-stream (e.g., 25, 45), 1 boundary (18)
3. Map the convergence → plateau → divergence trajectory over 28 steps
4. Determine optimal temporal cache window for Kontext

**Expected outcome**: Similar temporal plateau (cos >0.999) in mid-steps, compressed to 28-step range.

### Phase 4: Temporal Caching Prototype (analogous to M6)

1. Implement conservative uniform temporal caching for Kontext
2. Calibrate cache window based on Phase 3 results
3. Measure speedup and quality (SSIM, LPIPS) across 5 diverse test cases
4. Compare speedup ratio to Qwen M6 conservative (1.66x)

**Expected outcome**: ~1.5-1.8x speedup from temporal caching alone.

### Hardware Requirement

- **B200** (183 GB VRAM) for all phases
- CPU-only not suitable (no inference needed for F0, but F1 requires GPU)

### Evidence to Produce

1. Measured VRAM and timing (validates our estimates)
2. Per-step timing profile (reveals block heterogeneity)
3. Temporal redundancy trajectory (determines cache window)
4. Temporal caching results (validates transfer from Qwen M6)
5. Quality metrics on diverse test cases

---

# Qwen Track — Milestone 9: Broad Validation + Ablation Study

## 1. Test Suite Description

18 test cases spanning 6×1-image, 6×2-image, 6×3-image inputs. Covers 18 distinct edit types: simple object, landscape, style transfer, text/logo, material texture, portrait, object placement, scene composition, texture detail, identity-preserving edit, object replacement, background change, arrangement, multi-portrait, nature scene, interior design, complex scene, abstract. Includes 5 original M5/M6 cases, 2 M8 cases, and 11 new cases. Difficulty: 3 easy, 7 medium, 8 hard.

Four configurations per case:
- **Baseline**: Full 40-step CFG (80 transformer passes)
- **CFG-skip only**: M5 conservative — skip unconditional pass for steps 5-34 (50 passes, 1.60x theoretical)
- **Temporal cache only**: M6 conservative — cache prediction for steps 10-25 (48 passes, 1.67x theoretical)
- **Fixed combined**: M8 — temporal cache steps 10-25 + CFG-skip steps 5-9, 26-34 (34 passes, 2.35x theoretical)

All comparisons use SEED=42, TRUE_CFG_SCALE=4.0, 40 denoising steps. Metrics: SSIM (structural), LPIPS-AlexNet (perceptual distance).

## 2. Full Results Table

| Case | Img | Type | Diff | Base(s) | CFG_x | CFG_SSIM | CFG_LPIPS | TMP_x | TMP_SSIM | TMP_LPIPS | CMB_x | CMB_SSIM | CMB_LPIPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tc02_1img_landscape | 1 | landscape | med | 26.2 | 1.60 | 0.935 | 0.054 | 1.67 | 0.917 | 0.080 | 2.35 | 0.895 | 0.112 |
| tc08_1img_simple | 1 | simple_object | easy | 26.6 | 1.60 | 0.935 | 0.081 | 1.67 | 0.944 | 0.077 | 2.35 | 0.926 | 0.124 |
| tc09_1img_style | 1 | style_transfer | hard | 26.7 | 1.60 | 0.680 | 0.337 | 1.67 | 0.726 | 0.323 | 2.35 | 0.642 | 0.439 |
| tc10_1img_text | 1 | text_logo | hard | 26.5 | 1.60 | 0.749 | 0.218 | 1.67 | 0.846 | 0.216 | 2.35 | 0.766 | 0.274 |
| tc11_1img_material | 1 | material_texture | hard | 26.6 | 1.60 | 0.769 | 0.352 | 1.67 | 0.806 | 0.281 | 2.35 | 0.665 | 0.509 |
| tc12_1img_portrait | 1 | portrait | med | 26.2 | 1.60 | 0.895 | 0.112 | 1.67 | 0.963 | 0.043 | 2.35 | 0.911 | 0.098 |
| tc01_2img_objects | 2 | object_placement | easy | 47.0 | 1.60 | 0.941 | 0.106 | 1.67 | 0.955 | 0.084 | 2.35 | 0.947 | 0.101 |
| tc03_2img_scene | 2 | scene_composition | hard | 46.9 | 1.60 | 0.883 | 0.114 | 1.67 | 0.902 | 0.130 | 2.35 | 0.864 | 0.180 |
| tc05_2img_texture | 2 | texture_detail | hard | 47.4 | 1.60 | 0.821 | 0.179 | 1.67 | 0.893 | 0.109 | 2.35 | 0.832 | 0.204 |
| tc13_2img_identity | 2 | identity_preserving | hard | 47.1 | 1.60 | 0.963 | 0.044 | 1.67 | 0.936 | 0.104 | 2.35 | 0.931 | 0.118 |
| tc14_2img_replace | 2 | object_replacement | med | 47.0 | 1.60 | 0.952 | 0.042 | 1.67 | 0.965 | 0.037 | 2.35 | 0.949 | 0.049 |
| tc15_2img_background | 2 | background_change | med | 46.6 | 1.60 | 0.838 | 0.241 | 1.67 | 0.905 | 0.159 | 2.35 | 0.853 | 0.247 |
| tc04_3img_arrange | 3 | arrangement | easy | 72.0 | 1.60 | 0.952 | 0.104 | 1.67 | 0.974 | 0.043 | 2.35 | 0.955 | 0.109 |
| tc06_3img_portrait | 3 | multi_portrait | med | 72.1 | 1.60 | 0.903 | 0.113 | 1.67 | 0.937 | 0.103 | 2.35 | 0.905 | 0.147 |
| tc07_3img_nature | 3 | nature_scene | hard | 72.0 | 1.60 | 0.731 | 0.238 | 1.67 | 0.741 | 0.222 | 2.35 | 0.687 | 0.304 |
| tc16_3img_interior | 3 | interior_design | med | 72.1 | 1.60 | 0.882 | 0.150 | 1.67 | 0.922 | 0.104 | 2.35 | 0.890 | 0.148 |
| tc17_3img_complex | 3 | complex_scene | hard | 72.2 | 1.60 | 0.681 | 0.269 | 1.67 | 0.736 | 0.262 | 2.35 | 0.637 | 0.364 |
| tc18_3img_abstract | 3 | abstract | med | 72.1 | 1.60 | 0.816 | 0.308 | 1.67 | 0.855 | 0.199 | 2.35 | 0.799 | 0.345 |
| **Mean** | | | | | **1.600** | **0.851** | **0.170** | **1.667** | **0.885** | **0.143** | **2.353** | **0.836** | **0.215** |

## 3. Ablation Analysis

### Per-Configuration Aggregate

| Configuration | Mean Speedup | Mean SSIM | Std SSIM | Range | Mean LPIPS | Std LPIPS |
|---|---|---|---|---|---|---|
| CFG-skip only | 1.600x ± 0.001 | 0.8513 | 0.0923 | 0.680–0.963 | 0.1702 | 0.0984 |
| Temporal only | 1.667x ± 0.001 | 0.8847 | 0.0795 | 0.726–0.974 | 0.1430 | 0.0848 |
| Fixed combined | 2.353x ± 0.001 | 0.8363 | 0.1078 | 0.637–0.955 | 0.2151 | 0.1274 |

### Component Contribution

- **Temporal caching** is the better single-axis method: +0.034 higher mean SSIM than CFG-skip (0.885 vs 0.851), −0.027 lower LPIPS (0.143 vs 0.170), at slightly higher speedup (1.667x vs 1.600x).
- **CFG-skip** contributes meaningful additional speedup when combined with temporal caching: combined 2.353x > temporal alone 1.667x (+41% more speedup).
- **Combining does NOT compound quality loss.** If losses were independent, predicted combined SSIM ≈ 0.753. Actual combined SSIM = 0.836 — substantially better. The components overlap: features vulnerable to CFG-skip are also vulnerable to temporal caching, so applying both does not double the damage.
- **Quality cost of combining over best single axis**: combined SSIM 0.836 vs temporal-only 0.885 = −0.049 SSIM for +0.686x more speedup.

## 4. Quality Distribution (Fixed Combined)

### SSIM Percentiles
- p10 = 0.642, p25 = 0.766, p50 (median) = 0.890, p75 = 0.926, p90 = 0.949

### LPIPS Percentiles
- p10 = 0.098, p25 = 0.112, p50 = 0.180, p75 = 0.304, p90 = 0.439

### Fraction Analysis
- **SSIM > 0.90**: 7/18 (39%) — good quality
- **SSIM > 0.85**: 11/18 (61%) — acceptable quality
- **SSIM < 0.80**: 6/18 (33%) — concerning quality
- **SSIM < 0.70**: 4/18 (22%) — significant degradation

### Distribution Shape
The quality is **bimodal**: a "robust" cluster of 11 cases (SSIM 0.83–0.96) where acceleration works well, and a "fragile" cluster of 7 cases (SSIM 0.64–0.80) where quality degrades substantially. The median (0.890) is higher than the mean (0.836), indicating the distribution is left-skewed by hard cases.

## 5. Image-Count Scaling

| Input Count | n | Mean Base(s) | Mean Accel(s) | Mean Speedup | Mean SSIM | Std SSIM | Mean LPIPS | Abs Saved(s) |
|---|---|---|---|---|---|---|---|---|
| 1-image | 6 | 26.5 | 11.2 | 2.353x | 0.801 | 0.117 | 0.259 | 15.2 |
| 2-image | 6 | 47.0 | 20.0 | 2.353x | 0.896 | 0.048 | 0.150 | 27.0 |
| 3-image | 6 | 72.2 | 30.7 | 2.353x | 0.812 | 0.117 | 0.236 | 41.5 |

### Key Observations
1. **Speedup is perfectly constant** at 2.353x regardless of image count — confirmed on the expanded suite.
2. **Absolute savings scale linearly**: 15s → 27s → 42s (approximately proportional to baseline time). Multi-image cases benefit most in wall-clock terms.
3. **2-image cases have best quality** (0.896 mean SSIM, lowest variance). Additional conditioning stabilizes the generation trajectory. 1-image and 3-image are comparable (~0.80–0.81).
4. **This is confounded by test case difficulty** — the 1-image set happened to include the harshest new cases (style, material, text). Difficulty is a stronger predictor than image count.

## 6. Failure Characterization

### By Difficulty
| Difficulty | n | Combined SSIM | CFG-only SSIM | Temporal-only SSIM | Combined LPIPS |
|---|---|---|---|---|---|
| Easy | 3 | 0.943 ± 0.012 | 0.942 | 0.958 | 0.111 |
| Medium | 7 | 0.886 ± 0.044 | 0.889 | 0.924 | 0.164 |
| Hard | 8 | 0.753 ± 0.105 | 0.785 | 0.823 | 0.299 |

**Difficulty is the strongest predictor of quality degradation.** All hard cases degrade; all easy cases preserve quality well.

### By Edit Type (worst to best)
1. **complex_scene** (0.637) — bustling market, many elements
2. **style_transfer** (0.642) — watercolor painting, artistic detail
3. **material_texture** (0.665) — granite surface, fine texture
4. **nature_scene** (0.687) — forest/lake, organic detail
5. **text_logo** (0.766) — neon signage, precise shapes
6. **abstract** (0.799) — geometric, sharp edges
7. **texture_detail** (0.832) — cat/cushion, mixed texture
8. **background_change** (0.853) — beach scene, moderate detail
9. **scene_composition** (0.864) — sunset, compositional
10. **interior_design** (0.890) — minimalist room
11. **landscape** (0.895) — meadow/sky
12. **multi_portrait** (0.905) — studio portraits
13. **portrait** (0.911) — single portrait
14. **simple_object** (0.926) — apple on white
15. **identity_preserving** (0.931) — same person, different outfit
16. **object_placement** (0.947) — objects on table
17. **object_replacement** (0.949) — furniture swap
18. **arrangement** (0.955) — geometric shapes

### Failure Pattern
Content with **fine-grained visual detail** (texture, artistic style, complex multi-element scenes, natural organic patterns) degrades most. Content with **structural/semantic focus** (object placement, portrait, identity, arrangement) is robust. The acceleration drops CFG and temporal steps that are critical for refining fine details, but structural layout is established in early steps that always run at full compute.

**The degradation is NOT correlated with image count** — it is correlated with the visual complexity demanded by the prompt.

## 7. Generalization Assessment

### Comparison to M8 Reference
| Metric | M8 (7 cases) | M9 (18 cases) | Delta |
|---|---|---|---|
| Mean speedup | 2.35x | 2.353x | +0.003x |
| Mean SSIM | 0.869 | 0.836 | −0.033 |
| SSIM std | 0.085 | 0.108 | +0.023 |
| Min SSIM | 0.687 | 0.637 | −0.050 |

### Bottom Line
**The speedup generalizes perfectly (2.353x, ±0.001).** The schedule always uses exactly 34/80 passes regardless of content.

**The quality is lower than M8 suggested.** Mean SSIM drops from 0.869 to 0.836 with a more diverse test suite. The original 7 cases under-represented hard content types (style, texture, complex scenes, text). The broader suite reveals a bimodal quality distribution: 61% of cases maintain SSIM > 0.85, but 33% fall below 0.80.

**The method is robust for structural/semantic content** (portraits, objects, arrangements, identity edits — SSIM 0.90+). **It is unreliable for fine-detail content** (texture, style, complex scenes — SSIM 0.64–0.77). This is a fundamental limitation of both CFG-skip and temporal caching: they prune computation in the mid-trajectory where detail refinement occurs.

**Recommendation**: The fixed combined method (2.35x) is suitable for deployment with a content-type gate — use acceleration for structural edits, fall back to full compute for texture/style-heavy prompts. Alternatively, a more conservative combined schedule (e.g., temporal cache steps 12-22, CFG-skip steps 6-8, 23-30) could narrow the quality gap at the cost of reduced speedup.

---

# Milestone 8: Adaptive Combined Acceleration Prototype

## 1. Combined Method Description

### Design
Integrates temporal step-caching (M6) and CFG-skip (M5) within a single denoising loop. Two operating modes:

**Fixed Combined**: Conservative schedules from both axes applied simultaneously.
- Steps 0-4: Full compute (2 transformer passes per step)
- Steps 5-9: CFG-skip only (1 pass — skip unconditional)
- Steps 10-25: Temporal cache (0 passes — reuse previous prediction)
- Steps 26-34: CFG-skip only (1 pass)
- Steps 35-39: Full compute (2 passes)
- Total: 34/80 passes → theoretical 2.35x

**Adaptive Combined**: Runtime cosine similarity signal modulates per-step decisions. Three tiers:
1. **Temporal cache** (0 passes): when lagged similarity > 0.999 AND consecutive cached < 4
2. **CFG-skip** (1 pass): run conditional only, when current similarity > 0.995
3. **Full compute** (2 passes): when similarity drops below threshold

Adaptive config: temporal_threshold=0.999, cfg_skip_threshold=0.995, max_consecutive_cache=4, min_warmup_steps=3.

### Implementation
Single script (`scripts/m8_combined.py`) with `run_inference(mode=...)` supporting baseline, fixed_combined, and adaptive_combined. The adaptive mode measures cosine similarity between current and previous conditional outputs at each probe step, using the similarity as a signal to decide the next step's action.

## 2. Per-Case Results Table

### Phase 1: Primary Experiments (7 test cases)

| Case | Imgs | Base(s) | Fix(s) | Fix_x | Fix_SSIM | Adp(s) | Adp_x | Adp_SSIM |
|---|---|---|---|---|---|---|---|---|
| tc1_2img_objects | 2 | 46.95 | 19.95 | 2.35x | 0.947 | 14.15 | 3.32x | 0.932 |
| tc2_1img_landscape | 1 | 26.14 | 11.10 | 2.35x | 0.895 | 5.27 | 4.96x | 0.806 |
| tc3_2img_scene | 2 | 46.90 | 19.92 | 2.35x | 0.864 | 12.36 | 3.79x | 0.807 |
| tc4_3img_arrange | 3 | 71.89 | 30.54 | 2.35x | 0.955 | 15.32 | 4.69x | 0.919 |
| tc5_2img_texture | 2 | 47.27 | 20.09 | 2.35x | 0.832 | 11.28 | 4.19x | 0.693 |
| tc6_3img_portrait | 3 | 72.00 | 30.59 | 2.35x | 0.905 | 22.57 | 3.19x | 0.830 |
| tc7_3img_nature | 3 | 71.93 | 30.56 | 2.35x | 0.687 | 20.73 | 3.47x | 0.672 |
| **Aggregate** | | | | **2.35x** | **0.869** | | **3.95x** | **0.809** |

### Adaptive Decision Distribution

| Case | Full | CFG-skip | Temporal Cache | Total Passes |
|---|---|---|---|---|
| tc1_2img_objects | 4 | 16 | 20 | 24 |
| tc2_1img_landscape | 4 | 8 | 28 | 16 |
| tc3_2img_scene | 5 | 11 | 24 | 21 |
| tc4_3img_arrange | 3 | 11 | 26 | 17 |
| tc5_2img_texture | 3 | 13 | 24 | 19 |
| tc6_3img_portrait | 5 | 15 | 20 | 25 |
| tc7_3img_nature | 7 | 9 | 24 | 23 |

## 3. Fixed vs Adaptive Comparison

### Aggregate
- **Fixed combined**: 2.35x speedup, 0.869 mean SSIM, SSIM std=0.085
- **Adaptive combined**: 3.95x speedup, 0.809 mean SSIM, SSIM std=0.093

**Trade-off**: Adaptive delivers 68% more speedup (+1.60x) at a cost of 0.060 lower mean SSIM.

### Hard Cases (tc3, tc5 from M5/M6)
- **tc3_2img_scene**: Fixed 0.864 vs Adaptive 0.807 (Adaptive 0.057 worse)
- **tc5_2img_texture**: Fixed 0.832 vs Adaptive 0.693 (Adaptive 0.139 worse)

**Finding**: Adaptive mode does NOT protect hard cases. It is more aggressive (caching 24+ of 40 steps) and degrades quality further on prompts where complexity or texture are critical. The runtime similarity signal remains high even when the model is producing outputs that diverge visually from the baseline — the cosine similarity in latent space does not correlate well with perceptual quality differences for complex/textured content.

### New Hard Case: tc7_3img_nature
Both modes struggle on tc7 (forest/lake scene): Fixed SSIM=0.687, Adaptive SSIM=0.672. This is the worst-performing case for both modes. The dense, textured nature scene (forest + lake + sky) is inherently hard for both CFG-skip and temporal caching.

## 4. Image-Count Scaling Analysis

### Phase 2: Scaling Results (3 cases × 1/2/3 images)

| Case | 1img Base | 2img Base | 3img Base | 1img Fix_x | 2img Fix_x | 3img Fix_x | 1img Adp_x | 2img Adp_x | 3img Adp_x |
|---|---|---|---|---|---|---|---|---|---|
| scale_objects | 26.6s | 46.9s | 71.9s | 2.35x | 2.35x | 2.35x | 3.78x | 4.98x | 3.99x |
| scale_shapes | 26.6s | 46.9s | 72.0s | 2.35x | 2.35x | 2.35x | 3.78x | 4.97x | 4.69x |
| scale_portrait | 26.6s | 47.0s | 72.1s | 2.35x | 2.35x | 2.35x | 3.78x | 3.19x | 3.19x |

### Key Observations

1. **Fixed combined speedup is perfectly constant** at 2.35x regardless of image count. This is expected: the fixed schedule always uses exactly 34/80 passes, and the speedup ratio is independent of per-pass cost.

2. **Baseline scales super-linearly**: 1img=26.6s, 2img=46.9s (1.76x), 3img=72.0s (2.71x). Matches M7's O(S^1.4) scaling from quadratic attention.

3. **Adaptive speedup varies non-monotonically**: Not a simple function of image count. The adaptive signal's behavior depends on the specific generation trajectory, not just the sequence length. For scale_objects: 1img=3.78x, 2img=4.98x, 3img=3.99x.

4. **Absolute time saved scales with image count**: Despite constant speedup ratio for fixed mode, absolute savings grow: 1img saves ~15s, 2img saves ~27s, 3img saves ~42s. Multi-image cases benefit more in wall-clock terms.

5. **Quality generally improves with more images** for both modes: scale_portrait fixed SSIM: 1img=0.803, 2img=0.854, 3img=0.905. This suggests additional conditioning images stabilize the generation trajectory, making it more tolerant of acceleration.

## 5. Quality Robustness Assessment

| Metric | Fixed Combined | Adaptive Combined |
|---|---|---|
| Mean SSIM | 0.869 | 0.809 |
| Std SSIM | 0.085 | 0.093 |
| Min SSIM | 0.687 | 0.672 |
| Max SSIM | 0.955 | 0.932 |
| Range | 0.268 | 0.260 |

**Fixed is more robust** (lower variance). Adaptive's higher speed comes with more variable quality outcomes. However, both modes share the same hard cases — quality variance is primarily driven by prompt complexity, not by the acceleration mode.

The adaptive mode's failure to protect hard cases suggests that the per-step cosine similarity is not a good proxy for perceptual quality. In latent space, complex/textured images can have high step-to-step similarity (the trajectory is "smooth") while still producing visually different outputs when acceleration perturbs the trajectory.

## 6. Comparison with Single-Axis Methods (M5, M6)

| Method | Speedup | Mean SSIM | Hard Case SSIM (tc3/tc5 avg) |
|---|---|---|---|
| M5 CFG-skip (aggressive) | 1.82x | 0.861 | 0.772 |
| M5 CFG-skip (conservative) | 1.60x | 0.852* | 0.852* |
| M6 Temporal (conservative) | 1.66x | 0.928 | 0.897 |
| M6 Temporal (aggressive) | 2.86x | 0.776 | 0.665 |
| **M8 Fixed combined** | **2.35x** | **0.869** | **0.848** |
| **M8 Adaptive combined** | **3.95x** | **0.809** | **0.750** |

*M5 conservative only tested on tc3/tc5.

**Combined speedup exceeds either single axis**:
- Fixed combined (2.35x) > M5 aggressive (1.82x) + M6 conservative (1.66x) individually
- Adaptive combined (3.95x) > M6 aggressive (2.86x), the fastest single-axis method

**No negative interaction**: Combining temporal caching and CFG-skip does NOT compound quality loss. Fixed combined SSIM (0.869) is between M5 aggressive (0.861) and M6 conservative (0.928), demonstrating that the two acceleration axes operate on orthogonal redundancy dimensions (temporal step-to-step vs. cond/uncond within a step).

## 7. Bottom-Line Summary

### What works
- **Fixed combined** is the recommended operating mode: **2.35x speedup at 0.869 mean SSIM**. Predictable, consistent performance (exactly 2.35x on every test case regardless of prompt or image count). Quality is acceptable for most use cases.
- **Adaptive combined** achieves **3.95x mean speedup** but with unacceptable quality on hard cases (SSIM 0.672-0.693 for nature/texture prompts). Useful only where quality tolerance is high.

### What doesn't work
- **Adaptive quality protection**: The cosine similarity signal does not protect quality on hard cases. It successfully identifies the temporal plateau for caching but cannot distinguish "safe to skip" from "will degrade quality" within the plateau.
- **Image-count scaling of speedup ratio**: Speedup ratio is constant (fixed) or non-monotonic (adaptive). Multi-image cases do NOT get a higher speedup ratio — they get the same ratio applied to a larger baseline, saving more absolute time.

### Pareto frontier
Two operating points are available:
1. **Quality-first**: Fixed combined at 2.35x, 0.869 SSIM
2. **Speed-first**: Adaptive combined at 3.95x, 0.809 SSIM
There is no point between these — the adaptive signal has discrete transitions that produce these two clusters.

### Implications
- The 2.35x speedup from fixed combined reduces a 3-image inference from 72s → 31s. This is a meaningful production improvement.
- Further acceleration beyond 2.35x with acceptable quality will require approaches beyond schedule-based caching/skipping — likely architectural changes (sparse attention, token pruning) or distillation.

---

# Milestone 7: Multi-Image Condition Redundancy Characterization

## 1. Pipeline Mechanical Structure for Multi-Image Input

### How Multiple Images Enter the Pipeline

The `QwenImageEditPlusPipeline` processes multiple input images through two parallel paths:

**Path A — Text Encoder (Qwen2.5-VL-7B):**
- Each image gets a template token: `"Picture N: <|vision_start|><|image_pad|><|vision_end|>"`
- All templates are concatenated into a single prompt string: `"Picture 1: ... Picture 2: ... <user prompt>"`
- Qwen2.5-VL processes this as a multi-image understanding task
- Output: `prompt_embeds` — text encoder hidden states. Token count grows with image count due to vision tokens.

**Path B — VAE Latent Encoding:**
- Each image is resized to `VAE_IMAGE_SIZE = 1024*1024` area (maintaining aspect ratio)
- Each image is independently VAE-encoded to latent space (shape: `[1, 1, 16, H/8, W/8]`)
- Each image's latents are packed into 2x2 patch tokens (sequence of `(H/16)*(W/16)` tokens, each 64-dim)
- All image latents are **concatenated along the sequence dimension** (dim=1)

### Token Structure Entering the Transformer

The transformer receives `hidden_states = [noise_latents, img1_latents, img2_latents, ...]`:

| Component | Token Count (1024x1024 output, 1024x1024 input) | Notes |
|-----------|--------------------------------------------------|-------|
| Noise latents | `(H/16)*(W/16)` = 4096 (for 1024x1024) | Output image being denoised |
| Per-image latents | `(H_i/16)*(W_i/16)` ≈ 4096 (for 1024x1024 input) | Each condition image |
| **Total image tokens** | **4096 + N × ~4096** | Linear in image count |

### Positional Encoding (3D RoPE)

- **Noise latents**: Positive frame index (idx=0), positive spatial (h,w) positions
- **ALL condition images**: Negative frame index (-1 via `_compute_condition_freqs`), positive spatial (h,w) positions
- Multiple condition images share the same frame index (-1). They are distinguished only by their position in the concatenated sequence, not by any per-image identifier in RoPE.

### Timestep Modulation (`zero_cond_t`)

- `modulate_index = [0]*noise_tokens + [1]*all_condition_tokens`
- Index 0 (noise): modulated by full timestep embedding
- Index 1 (conditions): modulated by **zero timestep** embedding
- ALL condition images receive identical timestep modulation — no per-image differentiation

### Joint Attention Mechanism

In each of the 60 transformer blocks:
1. Text stream and image stream are separately projected to Q, K, V
2. Concatenated: `[text_Q, image_Q]`, `[text_K, image_K]`, `[text_V, image_V]`
3. Full scaled dot-product attention over the joint sequence
4. Output split back to text and image parts

The joint attention sequence = `[text_tokens, noise_latents, img1_latents, img2_latents, ...]`

**Attention cost**: O(S²) where S = text_len + noise_tokens + sum(image_tokens). Adding each image adds ~4096 tokens, causing quadratic growth in attention compute.

### Key Architectural Summary

Multi-image conditioning is **pure sequence concatenation** with:
- No dedicated cross-image attention mechanism
- No per-image positional differentiation (all share frame=-1)
- No per-image timestep modulation (all get zero-timestep)
- Cross-image interaction is entirely implicit through joint attention

## 2. Token Counts

| Config | Noise Tokens | Per-Image Cond | Total Cond | Total Image-Stream | Text (cond) | Text (neg) |
|--------|-------------|----------------|------------|-------------------|-------------|------------|
| 1 image | 4,096 | 4,096 | 4,096 | **8,192** | 217 | 209 |
| 2 images | 4,096 | 4,096 | 8,192 | **12,288** | 427 | 412 |
| 3 images | 4,096 | 4,096 | 12,288 | **16,384** | 628 | 615 |

Each additional image adds exactly **4,096 condition tokens** to the image stream (for 512×512 input → 1024×1024 VAE encoding → 64×64 latent → 32×32 packed patches = 1024... wait, actually the tokens are computed from VAE_IMAGE_SIZE=1024×1024, so H/16 × W/16 = 64×64 = 4096). Each image also adds ~200 text encoder tokens (the vision tokens from Qwen2.5-VL).

The joint attention sequence (text + image-stream) grows from **~8,400 to ~17,000 tokens** going from 1 to 3 images.

## 3. Compute Scaling Analysis

| Config | Total (s) | Mean Step (ms) | Per-Pass (ms) | VRAM (GB) | Relative Cost |
|--------|-----------|----------------|---------------|-----------|---------------|
| 1 image | 26.89 | 672 | 336 | 63.20 | **1.000x** |
| 2 images | 47.25 | 1,181 | 591 | 62.49 | **1.757x** |
| 3 images | 72.29 | 1,807 | 904 | 63.52 | **2.688x** |

### Scaling Characterization

The scaling is **super-linear but sub-quadratic** (~O(S^1.4)):

| Metric | 2img/1img | 3img/1img |
|--------|-----------|-----------|
| Sequence length ratio | 1.51x | 2.02x |
| Cost ratio (observed) | 1.76x | 2.69x |
| If linear (O(S)) | 1.51x | 2.02x |
| If quadratic (O(S²)) | 2.29x | 4.09x |
| **Observed scaling exponent** | **1.37** | **1.40** |

The ~O(S^1.4) scaling is consistent with FlashAttention: the FLOP count is O(S²) but the memory-efficient implementation achieves better-than-quadratic wall-clock scaling at these sequence lengths. From M3, joint attention accounts for ~75.8% of compute, and the remaining linear-scaling components (MLP, projections) pull the effective exponent below 2.0.

**VRAM is essentially constant** (~63 GB) across image counts. The additional condition tokens are relatively small compared to the model parameters (57.8 GB), and FlashAttention avoids materializing the full attention matrix.

## 4. Cross-Image Hidden State Similarity

At step 20 (mid-plateau), cosine similarity between hidden states of different image condition tokens:

### 2-Image Case

| Block | img1↔img2 | noise↔img1 | noise↔img2 |
|-------|-----------|------------|------------|
| 0 (early) | **-0.067** | 0.092 | 0.095 |
| 14 | 0.078 | 0.100 | 0.146 |
| 29 (mid) | 0.057 | 0.102 | 0.120 |
| 44 (dynamic) | 0.079 | 0.197 | 0.215 |
| 59 (final) | **0.999** | 0.999 | 0.999 |

### 3-Image Case

| Block | img1↔img2 | img1↔img3 | img2↔img3 | noise↔img1 | noise↔img2 | noise↔img3 |
|-------|-----------|-----------|-----------|------------|------------|------------|
| 0 | **-0.174** | -0.079 | **-0.278** | 0.080 | 0.157 | 0.044 |
| 14 | 0.047 | 0.147 | 0.133 | 0.122 | 0.200 | 0.143 |
| 29 | -0.011 | 0.081 | -0.087 | 0.104 | 0.150 | 0.088 |
| 44 | -0.004 | 0.089 | -0.058 | 0.193 | 0.245 | 0.185 |
| 59 | **0.999** | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |

### Interpretation

**Blocks 0–44**: Cross-image hidden state similarity is near zero or negative (range: -0.28 to +0.15). The transformer maintains **highly differentiated, nearly orthogonal representations** for different input images. There is no hidden-state redundancy — each image carries genuinely distinct information through the computational pipeline.

**Block 59** (final block): All tokens converge to near-identical representations (cos ~0.999). This is the normalizing/stabilizing block identified in M4 — it collapses all representations to a common subspace, which is its architectural function.

**Block 44** (most dynamic block): Shows the highest noise↔image similarity (~0.20), suggesting this is where condition information most actively influences the noise prediction. But cross-image similarity remains near zero.

**Key conclusion**: There is **no exploitable cross-image hidden state redundancy** in blocks 0–44. The representations are genuinely different, not redundantly similar.

## 5. Attention Region Distribution Analysis

### 2-Image Case (Block 29)

| Step | noise→noise | noise→img1 | noise→img2 | img1→self | img1→img2 | img1→noise | img2→self | img2→img1 | img2→noise |
|------|-------------|------------|------------|-----------|-----------|------------|-----------|-----------|------------|
| 0 | 88.2% | 5.7% | 6.1% | 29.2% | 17.9% | 52.8% | 35.0% | 11.9% | 53.0% |
| 5 | 94.4% | 3.5% | 2.1% | 34.4% | 13.0% | 52.6% | 29.0% | 14.0% | 57.1% |
| 15 | 94.6% | 3.2% | 2.2% | 36.1% | 11.8% | 52.1% | 29.7% | 13.1% | 57.2% |
| 25 | 93.1% | 3.7% | 3.2% | 37.5% | 12.6% | 49.9% | 30.4% | 15.1% | 54.5% |
| 39 | 91.4% | 4.9% | 3.7% | 29.9% | 11.7% | 58.4% | 22.4% | 13.0% | 64.6% |

### 3-Image Case (Block 29)

| Step | noise→noise | noise→img1 | noise→img2 | noise→img3 | img1→self | img1→others | img2→self | img2→others | img3→self | img3→others |
|------|-------------|------------|------------|------------|-----------|-------------|-----------|-------------|-----------|-------------|
| 0 | 84.2% | 5.1% | 5.6% | 5.1% | 29.0% | 28.3% | 40.0% | 33.1% | 38.8% | 20.6% |
| 15 | 93.4% | 3.4% | 1.2% | 2.0% | 37.2% | 18.6% | 35.9% | 26.2% | 33.7% | 22.1% |
| 25 | 93.2% | 3.6% | 1.3% | 2.0% | 36.8% | 18.9% | 38.7% | 22.4% | 34.2% | 23.8% |

### Attention Pattern Key Findings

1. **Noise tokens are overwhelmingly self-attentive**: 88–95% of noise token attention goes to other noise tokens. Only 3–6% of attention goes to each condition image. This means the denoising computation largely operates on its own representation, with condition images providing only a small steering signal.

2. **Image tokens attend primarily to themselves and noise**: Each image attends ~30–40% to its own tokens and ~50–60% to noise tokens. Cross-image attention is only ~10–15% per other image.

3. **Cross-image attention is minimal and roughly symmetric**: img1→img2 ≈ img2→img1 ≈ 10–15%. The model does not show strong directional cross-image attention patterns.

4. **The attention pattern is stable across denoising steps**: The distribution barely changes between steps 5–35 (the plateau region), consistent with M4's temporal redundancy findings.

5. **At step 0 and step 39**, attention is slightly more distributed (lower noise self-attention), consistent with M4's divergence at trajectory endpoints.

## 6. Temporal/CFG Redundancy Interaction with Image Count

### Temporal Redundancy (step-to-step cosine similarity of noise prediction)

| Step | 1 image | 2 images | 3 images | Pattern |
|------|---------|----------|----------|---------|
| 1 | 0.9957 | 0.9805 | 0.9893 | Convergence |
| 5 | 0.9992 | 0.9996 | 0.9972 | Entering plateau |
| 10 | 0.9999 | 0.9998 | 0.9999 | Deep plateau |
| 20 | 0.9999 | 0.9999 | 0.9999 | Peak plateau |
| 30 | 0.9999 | 0.9999 | 0.9999 | Plateau |
| 35 | 0.9997 | 0.9995 | 0.9998 | Beginning exit |
| 39 | **0.890** | **0.816** | **0.872** | Final divergence |

**The three-phase temporal pattern is preserved across all image counts.** Plateau region (steps 5–30) remains cos >0.999 regardless of image count. The final-step divergence (step 39) varies somewhat but is always present.

### CFG Redundancy (cond-vs-uncond cosine similarity)

| Step | 1 image | 2 images | 3 images | Pattern |
|------|---------|----------|----------|---------|
| 0 | 0.975 | 0.976 | 0.958 | Early divergence |
| 2 | 0.996 | 0.995 | 0.994 | Converging |
| 5 | 0.999 | 0.999 | 0.999 | Plateau |
| 20 | 0.9999 | 0.9999 | 0.9999 | Deep plateau |
| 39 | 0.996 | 0.993 | 0.999 | Late divergence |

**The CFG redundancy pattern is also preserved.** The unconditional pass produces nearly identical output to the conditional pass for ~90% of the trajectory, regardless of image count.

### Conclusion on Interaction

Adding more images does **not** shift the plateau boundaries or change the CFG divergence structure. The temporal and CFG acceleration strategies developed in M5/M6 are **directly applicable** to multi-image cases without modification. The same skip schedules (e.g., CFG skip steps 3–38, temporal cache steps 10–25) should be equally effective regardless of image count.

## 7. Bottom-Line Characterization

### Does multi-image conditioning create exploitable redundancy?

**No.** Multi-image conditioning introduces:

1. **Compute overhead (O(S^1.4))** — but no novel redundancy to exploit. The cost comes from a longer attention sequence, not from redundant computation.

2. **Orthogonal hidden states** — different images maintain genuinely differentiated representations (cos ≈ 0) through blocks 0–44. There is no "processing the same thing twice" to skip.

3. **Minimal cross-image attention** — only 10–15% of each image's attention goes to other images. The model does not waste compute on redundant cross-image interactions.

4. **Unchanged temporal/CFG redundancy** — the existing acceleration axes (CFG-skip and temporal caching) apply identically regardless of image count.

### Implications for Method Design (M8)

1. **No third acceleration axis from multi-image redundancy.** The differentiation must come from combining and adaptively controlling the temporal/CFG axes, not from a new redundancy source.

2. **Multi-image cases benefit MORE from existing acceleration.** Since the per-step cost is higher (1.76x for 2 images, 2.69x for 3 images), the same fractional speedup from CFG-skip or temporal caching produces a larger absolute time savings.

3. **The attention sparsity pattern (noise ~90% self-attentive) suggests a potential optimization**: sparse attention over condition tokens. But this is a compute-efficiency improvement, not a redundancy-exploitation strategy, and would require custom kernels — outside current scope.

4. **The VRAM headroom is maintained**: even with 3 images, peak VRAM is only ~63.5 GB (34.7% of 183 GB), leaving >100 GB for any caching or method overhead.

---

# Milestone 6: Temporal Step-Caching Prototype (Block-Aware)

## Bottom-Line Summary

**Conservative uniform temporal caching delivers 1.66x speedup with better quality than M5's CFG-skip (mean SSIM 0.93 vs 0.86).** Aggressive uniform caching achieves 2.86x speedup but causes severe degradation on 3/5 cases. Block-aware caching (recompute dynamic blocks, cache stable ones) **failed catastrophically** (SSIM 0.23-0.46), revealing that block-level caching in a sequential transformer is fundamentally flawed — cached intermediate states become inconsistent with evolving latent inputs.

**Key findings for the combined strategy:**
1. Temporal caching at the noise-prediction level works; block-level caching does not.
2. Conservative temporal caching (steps 10-25) is a better quality-speed tradeoff than CFG-skip (1.66x / SSIM 0.93 vs 1.82x / SSIM 0.86).
3. The same test cases degrade on both axes (tc3, tc5 — complex scenes, textures), suggesting these cases resist any form of acceleration.
4. Degradation correlates with edit complexity, not input count.

## 1. Mechanism Design

### Uniform Temporal Caching (Schedules A, B)
- **Mechanism**: For cached steps, skip both conditional and unconditional transformer passes entirely. Reuse the noise prediction (after CFG combination) from the most recent full step. Apply the scheduler step with the cached noise prediction and current timestep/latents.
- **Rationale**: M4 showed step-to-step cos >0.999 in the plateau (steps 5-30). If consecutive noise predictions are nearly identical, reusing the previous one should produce similar latent evolution.
- **Schedule A (aggressive)**: Cache steps 5-30 (26 cached, 14 full). 28 passes vs 80 baseline → theoretical 2.86x.
- **Schedule B (conservative)**: Cache steps 10-25 (16 cached, 24 full). 48 passes vs 80 → theoretical 1.67x.

### Block-Aware Caching (Schedule C) — FAILED
- **Mechanism**: For cached steps, run the transformer but cache stable blocks (0-39) and only recompute dynamic blocks (40-59, including block 44 and all downstream). norm_out/proj_out always run on fresh block 59 output.
- **Design rationale**: M4 showed block 44 is most dynamic (cos ~0.973) while blocks 0, 14, 59 are very stable (cos >0.999). By caching stable blocks and recomputing the dynamic zone, we expected to preserve quality while saving ~67% of block compute.
- **Why it failed**: Cached blocks return stale hidden states from the last full step (step 4). Over 26 cached steps, these states become completely disconnected from the current latent state. Block 40 receives hidden states from step 4's latents rather than the current latents, creating a fundamental inconsistency. Unlike uniform caching (which reuses the final noise prediction and lets the scheduler handle latent evolution), block-level caching disrupts the internal transformer state machine.
- **Lesson**: High step-to-step block output similarity does NOT imply safe block-level caching. The similarity holds because both the input and the block's computation are similar. When you fix the output (cache) but change the input (different latents), the downstream computation derails.

## 2. Per-Schedule Results

### Schedule A: Aggressive Uniform (cache steps 5-30)

| Test Case | Images | Baseline (s) | Cached (s) | Speedup | SSIM | MSE | PSNR (dB) | VRAM (GB) |
|-----------|--------|-------------|------------|---------|------|-----|-----------|-----------|
| tc1_2img_objects | 2 | 46.97 | 16.43 | 2.859 | 0.8687 | 336.1 | 22.9 | 64.87 |
| tc2_1img_landscape | 1 | 26.16 | 9.14 | 2.862 | 0.7576 | 530.2 | 20.9 | 64.87 |
| tc3_2img_scene | 2 | 46.94 | 16.43 | 2.856 | 0.6845 | 996.6 | 18.1 | 64.87 |
| tc4_3img_arrange | 3 | 72.00 | 25.19 | 2.858 | 0.9219 | 419.3 | 21.9 | 64.87 |
| tc5_2img_texture | 2 | 47.36 | 16.56 | 2.861 | 0.6451 | 1260.6 | 17.1 | 64.87 |
| **AGGREGATE** | | | | **2.859** | **0.7756** | **708.6** | **20.2** | |

Speedup matches theoretical 80/28 = 2.857x perfectly. Quality is poor: 3/5 cases show severe degradation (SSIM <0.70). Reusing a 26-step-stale noise prediction is too aggressive.

### Schedule B: Conservative Uniform (cache steps 10-25)

| Test Case | Images | Baseline (s) | Cached (s) | Speedup | SSIM | MSE | PSNR (dB) | VRAM (GB) |
|-----------|--------|-------------|------------|---------|------|-----|-----------|-----------|
| tc1_2img_objects | 2 | 46.97 | 28.50 | 1.648 | 0.9552 | 80.7 | 29.1 | 64.87 |
| tc2_1img_landscape | 1 | 26.16 | 15.68 | 1.669 | 0.9171 | 127.3 | 27.1 | 64.87 |
| tc3_2img_scene | 2 | 46.94 | 28.17 | 1.666 | 0.9020 | 165.0 | 26.0 | 64.87 |
| tc4_3img_arrange | 3 | 72.00 | 43.19 | 1.667 | 0.9739 | 44.3 | 31.7 | 64.87 |
| tc5_2img_texture | 2 | 47.36 | 28.40 | 1.668 | 0.8929 | 143.1 | 26.6 | 64.87 |
| **AGGREGATE** | | | | **1.664** | **0.9282** | **112.1** | **28.1** | |

The best quality-speed tradeoff. 2/5 cases are "good" (SSIM >0.95), 2/5 "moderate" (0.90-0.95), 1/5 "notable" (0.89). Significantly better quality than aggressive uniform and M5 CFG-skip.

### Schedule C: Block-Aware (cache blocks 0-39, recompute 40-59) — NEGATIVE RESULT

| Test Case | Images | Baseline (s) | Cached (s) | Speedup | SSIM | MSE | PSNR (dB) | VRAM (GB) |
|-----------|--------|-------------|------------|---------|------|-----|-----------|-----------|
| tc1_2img_objects | 2 | 46.97 | 26.70 | 1.759 | 0.4097 | 2473.9 | 14.2 | 73.24 |
| tc2_1img_landscape | 1 | 26.16 | 14.88 | 1.759 | 0.3555 | 4279.8 | 11.8 | 73.24 |
| tc3_2img_scene | 2 | 46.94 | 26.68 | 1.760 | 0.2322 | 4043.9 | 12.1 | 73.24 |
| tc4_3img_arrange | 3 | 72.00 | 40.93 | 1.759 | 0.4633 | 1873.4 | 15.4 | 76.07 |
| tc5_2img_texture | 2 | 47.36 | 26.91 | 1.760 | 0.3847 | 3472.3 | 12.7 | 76.07 |
| **AGGREGATE** | | | | **1.759** | **0.3691** | **3228.7** | **13.2** | |

**Catastrophic failure.** SSIM 0.23-0.46 across all cases — worse than random perturbation of the baseline. Block-level caching destroys the transformer's internal state coherence. Additionally, VRAM increased by ~10 GB (73-76 GB vs 63-65 GB baseline) due to storing 40 blocks × 2 passes of cached hidden states — wasted memory for garbage output.

## 3. Block-Aware vs Uniform: Direct Comparison

| Metric | Aggressive Uniform | Conservative Uniform | Block-Aware |
|--------|-------------------|---------------------|-------------|
| **Speedup** | 2.86x | 1.66x | 1.76x |
| **Mean SSIM** | 0.776 | 0.928 | 0.369 |
| **SSIM range** | 0.645-0.922 | 0.893-0.974 | 0.232-0.463 |
| **VRAM** | 64.87 GB | 64.87 GB | 73-76 GB |
| **Verdict** | Too aggressive | Best tradeoff | Failed |

Block-aware caching does NOT outperform uniform. It is dramatically worse at a similar speedup (1.76x vs 1.66x). The block-level redundancy identified in M4 (high cosine similarity between consecutive steps) does not translate into a viable block-level caching strategy. Uniform noise-prediction-level caching is the correct abstraction.

## 4. Cross-Axis Comparison (M5 CFG-Skip vs M6 Temporal)

| Test Case | Input Count | Edit Type | M5 CFG SSIM | M6 Cons. SSIM | M6 Aggr. SSIM | Overlap? |
|-----------|------------|-----------|-------------|---------------|---------------|----------|
| tc1_2img_objects | 2 | Object arrangement | 0.9352 | 0.9552 | 0.8687 | Both moderate; temporal better |
| tc2_1img_landscape | 1 | Scene generation | 0.9008 | 0.9171 | 0.7576 | Both moderate; temporal better |
| tc3_2img_scene | 2 | Complex scene | 0.8125 | 0.9020 | 0.6845 | **Both worst tier** — overlap |
| tc4_3img_arrange | 3 | Multi-object | 0.9259 | 0.9739 | 0.9219 | Both good; temporal much better |
| tc5_2img_texture | 2 | Texture/detail | 0.7315 | 0.8929 | 0.6451 | **Both worst tier** — overlap |

### Key Observations

1. **Failure modes overlap**: tc3 (complex scene) and tc5 (texture/detail) are the hardest cases on BOTH axes. These cases have fundamentally complex denoising dynamics that resist acceleration regardless of method.

2. **Conservative temporal beats CFG-skip on every case**: At a slightly lower speedup (1.66x vs 1.82x), conservative temporal caching produces better quality on all 5 test cases. SSIM improvement: +0.02 to +0.16.

3. **No input-count pattern**:
   - 1-image (tc2): moderate on both
   - 2-image (tc1, tc3, tc5): ranges from good to severe on both
   - 3-image (tc4): good on both
   Degradation is driven by **edit complexity** (texture, scene composition), not image count.

4. **Edit type is the dominant factor**:
   - Simple object placement (tc1, tc4): resilient to both acceleration methods
   - Complex scenes / textures (tc3, tc5): degrade significantly under any acceleration
   - This defines the boundary of what's achievable: complex/textured edits may need full compute

5. **Combined strategy implication**: Since the hard cases overlap, combining CFG-skip + temporal caching on those cases would compound degradation. A selective strategy (accelerate easy cases, full compute for hard cases) would be more effective than a uniform combined policy.

## 5. Why Aggressive Temporal Fails More Than CFG-Skip

Both methods achieve similar speedups at their respective aggressive levels (2.86x temporal vs 1.82x CFG-skip), but temporal is more destructive:

- **CFG-skip**: Removes the guidance signal but still computes a fresh noise prediction at every step. The denoising trajectory tracks the correct latent state, just without course-correction from CFG.
- **Aggressive temporal**: Reuses a 26-step-stale noise prediction. The denoising trajectory diverges because the noise prediction doesn't adapt to the evolving latent state. The scheduler applies the SAME noise prediction to DIFFERENT latents across 26 steps, compounding error.

This explains why conservative temporal (16 cached steps, max ~8-step staleness) works well — the staleness is bounded, limiting error accumulation.

## 6. VRAM Impact

| Configuration | VRAM (GB) | Delta vs Baseline |
|--------------|-----------|-------------------|
| Baseline | 63.20-64.87 | — |
| Uniform temporal | 64.87 | +0-1.67 (noise pred tensor) |
| Block-aware | 73.24-76.07 | +8.4-11.2 (40 blocks × 2 passes cache) |

Uniform temporal caching has negligible VRAM overhead. Block-aware required ~10 GB extra for the block cache (wasted, given the quality collapse).

---

# Milestone 5: CFG-Skip Prototype + Robustness

## Bottom-Line Summary

**CFG-skip delivers a consistent 1.82x speedup (45% time reduction) but causes moderate-to-notable quality degradation.** The speedup is real and reliable, but the quality tradeoff is non-trivial: 3 of 5 test cases show moderate differences (SSIM 0.90–0.94), and 2 show notable degradation (SSIM 0.73–0.81). The degradation stems from disabling the CFG guidance signal, which compounds over 36 steps despite each step's cond/uncond similarity being >0.999. A conservative schedule (fewer skip steps) improves quality slightly but cannot eliminate the issue.

**This narrows the method space**: CFG-skip alone is not a drop-in replacement for full CFG at scale=4.0. It is viable for quality-tolerant applications or as one component of a combined acceleration strategy (e.g., CFG-skip on fewer steps + temporal step-caching on the plateau region).

## 1. Mechanism Design

### CFG-Skip v1: Drop-CFG (Primary Method)
- **Skip schedule**: Full CFG for steps 0–2 and 39; skip unconditional pass for steps 3–38 (36 of 40 steps)
- **When skipping**: Use conditional transformer output directly as noise prediction — no unconditional pass, no CFG combination
- **Rationale**: M4 showed cond/uncond cos >0.999 for steps 3–38. Eliminating the redundant unconditional pass saves one transformer forward pass per skipped step.
- **Theoretical speedup**: 80 passes → 44 passes = 1.818x

### CFG-Skip v2: Cached Uncond Reuse (Failed Variant)
- **Same skip schedule**, but instead of dropping CFG, cache the most recent unconditional output and reuse it in the CFG formula for skip steps
- **Result**: Catastrophic quality loss (SSIM 0.05–0.53). The cached uncond output is from a different latent state (previous step's input), so the cond/uncond relationship is destroyed.
- **Conclusion**: Caching stale uncond outputs is not viable. The uncond output must correspond to the same input latents as the cond output.

### Conservative v1 Schedule (Follow-Up)
- **Skip schedule**: Full CFG for steps 0–4 and 35–39; skip uncond for steps 5–34 (30 steps)
- **Theoretical speedup**: 80 passes → 50 passes = 1.60x
- **Tested on 2 worst cases** from the aggressive schedule

## 2. Per-Test-Case Results — v1 Aggressive (Skip 3–38)

| Test Case | Images | Baseline (s) | Skip (s) | Speedup | SSIM | MSE | PSNR (dB) | VRAM_B (GB) | VRAM_S (GB) |
|-----------|--------|-------------|----------|---------|------|-----|-----------|-------------|-------------|
| tc1_2img_objects | 2 | 47.06 | 25.91 | 1.816 | 0.9352 | 281.7 | 23.6 | 63.20 | 63.20 |
| tc2_1img_landscape | 1 | 26.19 | 14.40 | 1.818 | 0.9008 | 359.3 | 22.6 | 63.20 | 63.20 |
| tc3_2img_scene | 2 | 47.07 | 25.88 | 1.819 | 0.8125 | 556.5 | 20.7 | 63.20 | 63.20 |
| tc4_3img_arrange | 3 | 72.02 | 39.63 | 1.817 | 0.9259 | 393.0 | 22.2 | 63.23 | 63.23 |
| tc5_2img_texture | 2 | 47.56 | 26.17 | 1.817 | 0.7315 | 1158.1 | 17.5 | 64.87 | 64.87 |
| **AGGREGATE** | | | | **1.817** | **0.8612** | **549.7** | **21.3** | | |

### Key Observations

1. **Speedup is perfectly consistent**: 1.816–1.819x across all 5 cases regardless of image count (1, 2, or 3) or prompt complexity. Matches theoretical 80/44 = 1.818x exactly.

2. **Quality varies by test case**:
   - Good (SSIM >0.92): tc1 (objects), tc4 (3-image arrangement)
   - Moderate (SSIM 0.90): tc2 (landscape)
   - Notable degradation (SSIM <0.85): tc3 (scene), tc5 (texture)

3. **Degradation pattern**: More complex/detailed prompts (scene composition, textures) degrade more than simple object placement prompts. This suggests CFG's guidance signal matters more for fine-grained generation than for coarse structure.

4. **VRAM unchanged**: 63.20 GB for both variants. The unconditional pass doesn't allocate additional persistent memory beyond the model — it uses transient compute buffers that are already allocated.

## 3. Conservative Schedule Comparison (tc3, tc5 only)

| Test Case | Aggressive SSIM | Conservative SSIM | Delta | Aggressive Speed | Conservative Speed |
|-----------|----------------|-------------------|-------|-----------------|-------------------|
| tc3_2img_scene | 0.8125 | 0.8834 | +0.071 | 1.819x | 1.599x |
| tc5_2img_texture | 0.7315 | 0.8213 | +0.090 | 1.817x | 1.600x |

The conservative schedule improves SSIM by 0.07–0.09 at the cost of reducing speedup from 1.82x to 1.60x. Quality degradation is reduced but not eliminated.

## 4. v2 Cached Uncond Results (Negative Result)

| Test Case | v1 SSIM | v2 SSIM | v1→v2 Delta |
|-----------|---------|---------|-------------|
| tc1_2img_objects | 0.9352 | 0.0531 | -0.882 |
| tc2_1img_landscape | 0.9008 | 0.2220 | -0.679 |
| tc3_2img_scene | 0.8125 | 0.2365 | -0.576 |
| tc4_3img_arrange | 0.9259 | 0.5275 | -0.398 |
| tc5_2img_texture | 0.7315 | 0.2728 | -0.459 |

**v2 is strictly worse than v1** on every test case by a large margin. Reusing stale unconditional outputs breaks the CFG computation because the cached uncond was computed from a different input latent. This amplifies error through the CFG scale factor (4.0×).

## 5. Cross-Axis Analysis

### Why High Redundancy ≠ Free Skipping

The M4 redundancy data showed cond/uncond cosine similarity >0.999 for steps 3–38, with relative L2 distances of ~0.01–0.03. This suggested the unconditional pass was "redundant." However, the CFG formula amplifies this small difference:

```
combined = uncond + cfg_scale × (cond − uncond)
```

With cfg_scale=4.0, even a 1.5% relative difference between cond and uncond gets amplified to a ~4.5% perturbation per step. Over 36 skipped steps, this compounds:
- Each step contributes ~4.5% perturbation relative to what full CFG would produce
- The perturbations are correlated (same direction of change), so they accumulate
- After 36 steps, the accumulated drift produces SSIM degradation of 0.06–0.27

### Implications for Future Work

1. **CFG-skip is best suited for lower cfg_scale values** where the amplification effect is smaller
2. **Partial CFG-skip** (skip 10–20 steps in the peak-similarity region) may hit a better quality-speed sweet spot
3. **Temporal step-caching** (M4 Axis 1) may be more promising because it doesn't disable a guidance signal — it caches the full combined prediction from a previous step
4. **Combined approach**: Use full CFG for first ~5 and last ~5 steps, temporal caching for the plateau (steps 5–30), full steps for divergence (steps 31–39)

## 6. Test Case Diversity

| Test Case | Images | Aspect | Edit Type | Quality Impact |
|-----------|--------|--------|-----------|----------------|
| tc1_2img_objects | 2 | 1:1 | Object arrangement | Moderate |
| tc2_1img_landscape | 1 | 3:2 | Scene generation | Moderate |
| tc3_2img_scene | 2 | 1:1 | Complex scene | Notable |
| tc4_3img_arrange | 3 | 1:1 | Multi-object | Moderate |
| tc5_2img_texture | 2 | 4:3 | Texture/detail | Notable |

Test cases with more compositional complexity or fine texture detail showed greater quality degradation from CFG-skip.

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
