# Inference Acceleration for Visual Language Models

Experimental study of inference acceleration across two model families:

- **DiT Track** (completed): Temporal step-caching for flow-matching diffusion pipelines (Qwen-Image-Edit, FLUX Kontext-dev)
- **AR Track** (active): Attention-informed optimization for autoregressive VLM in-context learning (Lumina-mGPT-2.0-Omni)

## DiT Track: Temporal Step-Caching (Completed)

**Models**: Qwen-Image-Edit-2511 (19.5B) and FLUX.1 Kontext-dev (12B)

**Finding**: Both models exhibit a structural temporal plateau during denoising --- consecutive steps produce near-identical noise predictions (cosine similarity > 0.999) for 65-74% of the trajectory. This pattern transfers across architecturally distinct models with different guidance strategies.

**Method**: During the plateau, reuse the previous step's noise prediction instead of running the transformer. On Qwen, combine with CFG-skip (skip unconditional pass when conditional/unconditional outputs converge).

| Model | Method | Speedup | Mean SSIM | Notes |
|---|---|---|---|---|
| Qwen (19.5B) | Combined (temporal + CFG-skip) | 2.35x | 0.836 (n=18) | Two independent axes |
| Kontext (12B) | Temporal only | 2.15x | 0.754 (n=12) | Single axis, guidance-distilled |

Quality is bimodal: structural/semantic edits are robust (SSIM 0.85-0.99), while fine-detail edits (style transfer, texture) degrade under acceleration (SSIM 0.37-0.80). The same content types are fragile on both models.

**Mechanistic insight (F4)**: Quality loss comes from stale attention matmuls and FFN computations, not from stale conditioning K/V. Caching only conditioning K/V yields SSIM 0.917 with negligible speedup (1.015x), localizing the quality bottleneck to recomputation decisions above the K/V level.

**Status**: Experimental phase complete. Milestones M0-M9 (Qwen), F0-F4 (FLUX).

See [docs/results.md](docs/results.md) for full tables. See [docs/examples/](docs/examples/) for curated Qwen comparisons.

## AR Track: Autoregressive VLM In-Context Learning (Active)

**Model**: Lumina-mGPT-2.0-Omni (7B, Chameleon architecture, 32 layers, 32 heads)

**Task**: 1-shot depth estimation via in-context learning (NYU Depth V2, 512x512 dual-panel)

### Bottleneck

AR generation dominates the pipeline at 99.4% of total wall-clock time. Adding a single in-context demo image causes catastrophic scaling:

| Condition | Prompt Tokens | ms/token | Total Time | Slowdown |
|---|---|---|---|---|
| 0-shot | 2,129 | 39.9 | 83.0s | 1.00x |
| 1-shot | 6,293 | 274.5 | 571.2s | **6.88x** |

The 3x increase in prompt length produces a 7x increase in wall-clock time due to quadratic prefill scaling and increased per-token generation cost from the larger KV cache.

### Characterization (AR-1)

Per-head attention profiling across all 32x32 = 1,024 attention heads during generation reveals:

- 33% of total attention goes to demo-image tokens (range 0.7%-82.1% per head)
- Strong layer gradient: shallow layers (0-10) attend minimally to demo tokens, deep layers (22-31) attend heavily
- Head specialization within layers: some heads in the same layer attend 5-80% to demo content

This characterization informed the design of AR-2 and AR-3.

### KV Compression and Sparse Prefill (AR-2, AR-3)

**AR-2** (generation-time KV compression): Zeroing demo KV entries in low-demo-attention heads. Head-selective compression outperforms uniform at matched sparsity. Best config (moderate, 26.5% heads): SSIM 0.692 vs baseline. The intervention triggers AR generation chaos --- token-level divergence is immediate, but head-selective targeting limits perceptual damage.

**AR-3** (prefill-time sparse attention): Masking query-to-demo attention in low-demo heads during prefill. Fundamentally milder than AR-2 because demo KV cache entries are never modified (causal mask ensures demo tokens don't attend to query tokens). Best config (moderate, 26.5% sparsity): SSIM 0.905 --- a +31% improvement over AR-2's equivalent.

Both experiments validate the design principle (attention characterization identifies compressible structure) but achieve no realized wall-clock speedup: prefill is only 0.3% of pipeline cost, and KV compression doesn't reduce per-token generation cost.

### Prefix Caching + Position Encoding Extension (AR-4)

Extends position encoding via Dynamic NTK scaling (factor=2.0) to support 2-3 demo contexts (up to 16,702 tokens, 1.63x beyond the model's 10,240 max_position_embeddings). Implements demo-prefix KV caching: compute demo KV once, save to disk, reload for each query.

| Config | Fresh Pipeline | Cached Pipeline | Observed Speedup |
|---|---|---|---|
| 1-demo (6,293 tok) | 95.5s | 91.4s | 1.05x |
| 2-demo (10,457 tok) | 544.9s | 101.6s | **5.36x** |
| 3-demo (14,621 tok) | 562.2s | 108.5s | **5.18x** |

Prefix caching is essential for multi-demo ICL --- without it, generation time for sequences exceeding max_position_embeddings becomes prohibitive.

**Mechanism caveat**: The observed 5x speedup for 2-3 demos is under investigation. Preliminary analysis shows the gain comes primarily from generation being ~5.5x faster in the cached pipeline, not from avoiding demo re-prefill (which costs <2s). The fresh pipeline's generation becomes pathologically slow (543-561s for 2,081 tokens) when sequences exceed max_position_embeddings, while the cached pipeline generates at near-1-demo speed (~97-101s). This suggests the speedup partially arises from avoiding a DynamicNTKScaling RoPE recomputation pathology rather than purely from prefill amortization. AR-5 will investigate the root cause.

Quality between fresh and cached pipelines: mean SSIM 0.811 across 5 validation queries. Token-level divergence is immediate (inherent to AR generation stochasticity), but both pipelines produce valid depth maps.

**Status**: Active. Mechanism clarification planned (AR-5).

## Repository Structure

```
scripts/
  qwen/                             # Qwen-Image-Edit track (M0-M9)
    m1_feasibility.py                # Environment + loading
    m3_profile.py                    # Three-level denoising profiling
    m4_redundancy.py                 # Temporal + CFG redundancy
    m5_cfg_skip.py                   # CFG-skip prototype
    m6_temporal_cache.py             # Temporal step-caching prototype
    m7_multi_image.py                # Multi-image conditioning analysis
    m8_combined.py                   # Combined acceleration prototype
    m9_validation.py                 # 18-case validation + ablation
  flux/                             # FLUX Kontext-dev track (F0-F4)
    f1_baseline.py                   # Baseline + profiling
    f1_redundancy.py                 # Temporal redundancy characterization
    f1_temporal_cache.py             # Temporal cache transfer test
    f2_validation.py                 # 12-case validation study
    f4_kv_cache_prototype.py         # Conditioning-aware KV reuse
    f4_kv_measurement.py             # K/V stability measurement
  generate_plots.py                 # DiT track visualization
vlm_icl/                            # AR track (Lumina-mGPT)
  run_inference_profiled.py          # Baseline profiling
  run_context_scaling.py             # Demo-count scaling measurement
  prepare_nyu_depth.py               # NYU Depth V2 data preparation
  ar1_attention_characterization.py  # Per-head attention profiling
  ar2_kv_compression.py              # Generation-time KV compression
  ar3_sparse_prefill.py              # Prefill-time sparse attention
  ar4_prefix_cache.py                # Prefix caching + NTK extension
docs/
  results.md                        # Full results tables
  milestone_summary.md              # Per-milestone summaries
  plots/                            # DiT track visualizations
  examples/                         # Curated Qwen comparisons
```

## Hardware

All experiments on NVIDIA B200 (183 GB VRAM). Models loaded in bf16 without quantization.

- **Qwen**: 57.8 GB model, 63.2 GB peak
- **Kontext**: 34.4 GB model, 37.5 GB peak
- **Lumina-mGPT**: 48.7 GB model (0-shot), up to 93.9 GB peak (3-demo cached)
