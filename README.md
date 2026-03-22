# Accelerated Multi-Image Conditional Editing with Qwen-Image

Inference acceleration for multi-image conditional image editing using Qwen-Image-Edit-2511. Combines temporal step-caching and selective CFG reduction to achieve 2.35x speedup on a 40-step denoising pipeline, validated across 18 diverse test cases covering structural, semantic, and fine-detail content.

## Problem

Multi-image conditional editing with diffusion models is slow. Qwen-Image-Edit generates high-quality edits but requires 80 transformer passes per image (40 denoising steps x 2 passes for classifier-free guidance). On an NVIDIA B200, a single 3-image edit takes ~72 seconds.

## Method

Profiling reveals two sources of redundancy in the denoising trajectory:

1. **CFG redundancy**: Conditional and unconditional transformer outputs are near-identical (cosine similarity > 0.999) for steps 3-38 of 40. The unconditional pass can be skipped for these steps.
2. **Temporal redundancy**: Noise predictions change minimally between consecutive steps during the mid-trajectory plateau (steps 10-25). Cached predictions can be reused.

The combined method applies both simultaneously:

| Steps | Action | Passes per step |
|---|---|---|
| 0-4 | Full compute (conditional + unconditional) | 2 |
| 5-9 | CFG-skip (conditional only) | 1 |
| 10-25 | Temporal cache (reuse previous prediction) | 0 |
| 26-34 | CFG-skip (conditional only) | 1 |
| 35-39 | Full compute | 2 |

Total: 34 of 80 passes executed.

## Key Results

Validated on 18 test cases spanning 1-3 input images, 18 edit types, and easy/medium/hard difficulty levels.

- **Speedup**: 2.35x (constant across all image counts and content types)
- **Mean quality**: SSIM 0.836 vs. baseline (LPIPS 0.215)
- **Quality distribution is bimodal**:
  - 61% of cases maintain SSIM > 0.85 (structural/semantic content)
  - 33% of cases fall below SSIM 0.80 (fine-detail content)
- **Absolute time savings**: 15s (1-image), 27s (2-image), 42s (3-image)

### What works well

Structural and semantic edits — object placement, portraits, identity-preserving edits, arrangements — maintain SSIM 0.90+ under full acceleration. Layout is established in the early steps that always run at full compute.

### What degrades

Fine-detail content — style transfer, material textures, complex multi-element scenes, organic nature patterns — drops to SSIM 0.64-0.77. These edits depend on mid-trajectory refinement steps that the acceleration prunes.

### Ablation

| Configuration | Speedup | Mean SSIM |
|---|---|---|
| CFG-skip only | 1.60x | 0.851 |
| Temporal cache only | 1.67x | 0.885 |
| Combined | 2.35x | 0.836 |

Temporal caching is the better single-axis method. Combining does not compound quality loss — the two axes have overlapping vulnerability patterns.

## Current Stage

This is a validated prototype under active research. The fixed combined schedule (2.35x) represents a confirmed operating point with known quality characteristics. Content-adaptive routing — selecting between full and accelerated compute based on edit complexity — is under investigation.

## Repository Structure

```
scripts/
  m1_feasibility.py     # Environment + loading feasibility
  m3_profile.py         # Three-level denoising profiling
  m4_redundancy.py      # Temporal + CFG redundancy characterization
  m5_cfg_skip.py        # CFG-skip prototype (+ v2 variant)
  m6_temporal_cache.py  # Temporal step-caching prototype
  m7_multi_image.py     # Multi-image conditioning analysis
  m8_combined.py        # Combined acceleration prototype
  m9_validation.py      # 18-case validation + ablation study
docs/
  milestone_summary.md  # One-paragraph summary per milestone (M0-M9)
  results.md            # Full M9 results, tables, and analysis
  examples/             # Curated success/failure image comparisons
configs/                # Configuration files
Qwen-Image/             # Upstream model reference (code only)
```

## Visualizations

### Ablation by Edit Type

![Ablation bar chart](docs/ablation_by_edit_type.png)

### Quality Distribution

![Quality distribution](docs/quality_distribution.png)

### Image-Count Scaling

![Scaling plot](docs/image_count_scaling.png)

## Example Outputs

See [docs/examples/](docs/examples/) for curated baseline vs. accelerated comparisons, including both success cases (SSIM > 0.94) and failure cases (SSIM < 0.67).

## Hardware

All experiments run on NVIDIA B200 (183 GB VRAM). Model loaded in bf16 (57.76 GB), peak VRAM 63.20 GB. No quantization or offloading required.
