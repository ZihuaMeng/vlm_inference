# Results

Full experimental results across both model tracks. For analysis and interpretation, see [findings.md](../findings.md).

---

## M9: Qwen Validation and Ablation (18 cases, combined 2.35x)

### Ablation Table

| Configuration | Speedup | Mean SSIM | Std SSIM | Range | Mean LPIPS |
|---|---|---|---|---|---|
| CFG-skip only | 1.60x | 0.851 | 0.092 | 0.680 - 0.963 | 0.170 |
| Temporal cache only | 1.67x | 0.885 | 0.080 | 0.726 - 0.974 | 0.143 |
| Fixed combined | 2.35x | 0.836 | 0.108 | 0.637 - 0.955 | 0.215 |

### Per-Case Results

| Case | Images | Type | Difficulty | CFG SSIM | Temporal SSIM | Combined SSIM | Combined LPIPS |
|---|---|---|---|---|---|---|---|
| tc04_3img_arrange | 3 | arrangement | easy | 0.952 | 0.974 | 0.955 | 0.109 |
| tc14_2img_replace | 2 | object_replacement | medium | 0.952 | 0.965 | 0.949 | 0.049 |
| tc01_2img_objects | 2 | object_placement | easy | 0.941 | 0.955 | 0.947 | 0.101 |
| tc13_2img_identity | 2 | identity_preserving | hard | 0.963 | 0.936 | 0.931 | 0.118 |
| tc08_1img_simple | 1 | simple_object | easy | 0.935 | 0.944 | 0.926 | 0.124 |
| tc12_1img_portrait | 1 | portrait | medium | 0.895 | 0.963 | 0.911 | 0.098 |
| tc06_3img_portrait | 3 | multi_portrait | medium | 0.903 | 0.937 | 0.905 | 0.147 |
| tc02_1img_landscape | 1 | landscape | medium | 0.935 | 0.917 | 0.895 | 0.112 |
| tc16_3img_interior | 3 | interior_design | medium | 0.882 | 0.922 | 0.890 | 0.148 |
| tc03_2img_scene | 2 | scene_composition | hard | 0.883 | 0.902 | 0.864 | 0.180 |
| tc15_2img_background | 2 | background_change | medium | 0.838 | 0.905 | 0.853 | 0.247 |
| tc05_2img_texture | 2 | texture_detail | hard | 0.821 | 0.893 | 0.832 | 0.204 |
| tc18_3img_abstract | 3 | abstract | medium | 0.816 | 0.855 | 0.799 | 0.345 |
| tc10_1img_text | 1 | text_logo | hard | 0.749 | 0.846 | 0.766 | 0.274 |
| tc07_3img_nature | 3 | nature_scene | hard | 0.731 | 0.741 | 0.687 | 0.304 |
| tc11_1img_material | 1 | material_texture | hard | 0.769 | 0.806 | 0.665 | 0.509 |
| tc09_1img_style | 1 | style_transfer | hard | 0.680 | 0.726 | 0.642 | 0.439 |
| tc17_3img_complex | 3 | complex_scene | hard | 0.681 | 0.736 | 0.637 | 0.364 |
| **Mean** | | | | **0.851** | **0.885** | **0.836** | **0.215** |

### Quality Distribution (Combined 2.35x)

| Threshold | Count | Percentage |
|---|---|---|
| SSIM >= 0.90 | 7/18 | 39% |
| SSIM >= 0.85 | 11/18 | 61% |
| SSIM < 0.80 | 6/18 | 33% |
| SSIM < 0.70 | 4/18 | 22% |

Percentiles: p10 = 0.642, p25 = 0.766, p50 = 0.890, p75 = 0.926, p90 = 0.949

### Quality by Difficulty

| Difficulty | n | Mean SSIM | Std SSIM | Mean LPIPS |
|---|---|---|---|---|
| Easy | 3 | 0.943 | 0.012 | 0.111 |
| Medium | 7 | 0.886 | 0.044 | 0.164 |
| Hard | 8 | 0.753 | 0.105 | 0.299 |

### Image-Count Scaling

| Input Images | Baseline (s) | Accelerated (s) | Time Saved (s) | Mean SSIM |
|---|---|---|---|---|
| 1 | 26.5 | 11.2 | 15.2 | 0.801 |
| 2 | 47.0 | 20.0 | 27.0 | 0.896 |
| 3 | 72.2 | 30.7 | 41.5 | 0.812 |

Speedup is constant at 2.35x regardless of image count. Quality differences across image counts are confounded by test case difficulty.

---

## F2: Kontext Validation (12 cases, temporal caching)

### Full Results

| Case | Type | Difficulty | Cons Speedup | Cons SSIM | Cons LPIPS | Aggr Speedup | Aggr SSIM | Aggr LPIPS |
|---|---|---|---|---|---|---|---|---|
| f2_01_simple_object | simple_object | easy | 2.15x | 0.987 | 0.011 | 3.50x | 0.976 | 0.034 |
| f2_02_object_modify | object_modification | easy | 2.15x | 0.934 | 0.093 | 3.50x | 0.888 | 0.197 |
| f2_03_background_change | background_change | easy | 2.15x | 0.741 | 0.250 | 3.50x | 0.525 | 0.515 |
| f2_04_arrangement | arrangement | easy | 2.15x | 0.906 | 0.026 | 3.50x | 0.731 | 0.123 |
| f2_05_portrait | portrait | medium | 2.15x | 0.909 | 0.048 | 3.50x | 0.820 | 0.161 |
| f2_06_identity_edit | identity_preserving | medium | 2.15x | 0.686 | 0.352 | 3.50x | 0.491 | 0.654 |
| f2_07_interior | interior_design | medium | 2.15x | 0.834 | 0.146 | 3.50x | 0.752 | 0.331 |
| f2_08_text_edit | text_editing | medium | 2.15x | 0.874 | 0.175 | 3.50x | 0.823 | 0.284 |
| f2_09_style_transfer | style_transfer | hard | 2.15x | 0.681 | 0.544 | 3.50x | 0.591 | 0.761 |
| f2_10_fine_texture | fine_texture | hard | 2.15x | 0.366 | 0.513 | 3.50x | 0.173 | 0.555 |
| f2_11_complex_scene | complex_scene | hard | 2.15x | 0.415 | 0.573 | 3.50x | 0.280 | 0.834 |
| f2_12_nature_detail | nature_detail | hard | 2.15x | 0.710 | 0.298 | 3.50x | 0.570 | 0.527 |

Baseline mean time: 8.90s. Speedups match theoretical exactly.

### Aggregate Statistics

| Metric | Conservative (2.15x) | Aggressive (3.50x) |
|---|---|---|
| Mean SSIM | 0.754 | 0.635 |
| Median SSIM | 0.787 | 0.661 |
| SSIM std | 0.190 | 0.233 |
| SSIM range | 0.366 - 0.987 | 0.173 - 0.976 |
| Mean LPIPS | 0.253 | 0.415 |

### Quality Distribution (Conservative 2.15x)

| Threshold | Count | Percentage |
|---|---|---|
| SSIM >= 0.90 | 4/12 | 33% |
| SSIM >= 0.85 | 5/12 | 42% |
| SSIM < 0.80 | 6/12 | 50% |
| SSIM < 0.70 | 3/12 | 25% |

### Quality by Difficulty

| Difficulty | n | Cons SSIM | Cons LPIPS | Aggr SSIM | Aggr LPIPS |
|---|---|---|---|---|---|
| Easy | 4 | 0.892 | 0.095 | 0.780 | 0.217 |
| Medium | 4 | 0.826 | 0.181 | 0.721 | 0.358 |
| Hard | 4 | 0.543 | 0.482 | 0.403 | 0.669 |

---

## F3: Cross-Model Comparison

### Structured Comparison Table

| Dimension | Qwen-Image-Edit-2511 | FLUX.1 Kontext-dev |
|---|---|---|
| Architecture | 19.5B, 60 identical dual-stream blocks | 12B, 19 dual + 38 single blocks |
| Guidance | True CFG (2 passes/step) | Guidance-distilled (1 pass/step) |
| Default steps | 40 (80 passes total) | 28 (28 passes total) |
| Multi-image | 1-3 images | 1 image only |
| Baseline time | 47.4s | 9.17s |
| Peak VRAM | 63.2 GB | 37.5 GB |
| Denoising dominance | 99.3% | 97.8% |
| Attention fraction | 75.8% | 73-75% |
| Temporal plateau | Steps 5-30 (65%) | Steps 4-23 (74%) |
| Plateau cos range | >0.999 | >0.999 |
| Final-step divergence | Dramatic (cos=0.816) | Mild (cos=0.981) |
| Best operating point | 2.35x combined | 2.15x temporal |
| Quality at best point | SSIM 0.836 | SSIM 0.754 |

### Content-Type Failure Overlap

| Content Type | Qwen SSIM | Kontext SSIM | Both degrade? |
|---|---|---|---|
| complex_scene | 0.637 | 0.415 | Yes |
| style_transfer | 0.642 | 0.681 | Yes |
| material/fine_texture | 0.665 | 0.366 | Yes |
| nature | 0.687 | 0.710 | Yes |
| portrait | 0.911 | 0.909 | No (robust) |
| simple_object | 0.926 | 0.987 | No (robust) |
| arrangement | 0.955 | 0.906 | No (robust) |

The same content types degrade on both models. The failure ranking is preserved across architectures.

### Dual-Axis vs. Single-Axis

| | Qwen (dual-axis) | Kontext (single-axis) |
|---|---|---|
| Available axes | CFG + temporal | Temporal only |
| Best single-axis | 1.67x, SSIM 0.885 | 2.15x, SSIM 0.754 |
| Best combined | 2.35x, SSIM 0.836 | 2.15x, SSIM 0.754 |
| Quality cost distribution | Spread across two axes | Concentrated on one axis |

Qwen's combined method achieves higher speedup AND better quality because the two axes target different redundancy dimensions with overlapping vulnerability.

---

## F4: Mechanistic Ablation (KV Reuse vs. Temporal Cache)

### Per-Case Comparison

| Test Case | KV Speedup | KV SSIM | KV LPIPS | TC Speedup | TC SSIM | TC LPIPS |
|---|---|---|---|---|---|---|
| simple_object | 1.016x | 0.990 | 0.008 | 2.155x | 0.987 | 0.011 |
| object_modify | 1.016x | 0.931 | 0.006 | 2.153x | 0.888 | 0.051 |
| portrait | 1.015x | 0.947 | 0.026 | 2.154x | 0.836 | 0.175 |
| arrangement | 1.016x | 0.955 | 0.024 | 2.154x | 0.885 | 0.106 |
| style_transfer | 1.016x | 0.902 | 0.089 | 2.154x | 0.812 | 0.334 |
| fine_texture | 1.015x | 0.927 | 0.041 | 2.154x | 0.807 | 0.171 |
| complex_scene | 1.015x | 0.732 | 0.162 | 2.154x | 0.442 | 0.547 |
| nature_detail | 1.015x | 0.952 | 0.044 | 2.154x | 0.778 | 0.310 |
| **Mean** | **1.015x** | **0.917** | **0.050** | **2.154x** | **0.804** | **0.213** |

KV = condition-token K/V caching only (text + image K/V cached, everything else fresh).
TC = whole-step temporal caching (entire noise prediction reused).
Same plateau window (steps 6-20) for both methods.

### Quality Gap by Content Category

| Subset | KV SSIM | TC SSIM | Gap |
|---|---|---|---|
| Robust (n=4) | 0.955 | 0.899 | +0.056 |
| Fragile (n=4) | 0.878 | 0.710 | +0.168 |
| All (n=8) | 0.917 | 0.804 | +0.112 |

KV caching wins in all 8/8 cases. The quality advantage is largest on fragile content (+0.168 SSIM).

### Source of Quality Loss

| Component | KV Caching | Temporal Caching |
|---|---|---|
| Condition K/V (text + image) | Cached | Stale |
| Noise K/V | Fresh | Stale |
| Q projections | Fresh | Stale |
| Attention matmul | Fresh | Stale |
| FFN | Fresh | Stale |

The +0.112 SSIM gap identifies the quality bottleneck: quality loss comes from stale attention matmuls and FFN, not from stale condition K/V. Condition-token K/V staleness contributes negligibly to degradation.
