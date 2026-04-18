# Inference Acceleration for Visual Language Models

Experimental study of inference acceleration across two model families:

- **DiT Track** (completed): Temporal step-caching for flow-matching diffusion pipelines (Qwen-Image-Edit, FLUX Kontext-dev)
- **AR Track** (active): Attention-informed optimization for autoregressive VLM in-context learning (Lumina-mGPT-2.0-Omni)
- **Phase A — Emu3 / understanding-regime probe** (frozen): Emu3-Chat understanding-regime anchor validated; generation-regime (D-2a) currently in PI branch re-judgment after a window-clean negative checkpoint.

## Current project state

For the current branch-state summary (AR-6 frozen; D-1 Emu3 understanding frozen as the clean control baseline; D-2a generation in PI branch re-judgment after a window-clean negative checkpoint; second-model preparation as the upcoming path), see [`docs/current_project_state.md`](docs/current_project_state.md). Experiment tables for the completed tracks remain in [`docs/results.md`](docs/results.md) and [`docs/milestone_summary.md`](docs/milestone_summary.md); repo-hygiene context is in [`docs/storage_and_archive_notes.md`](docs/storage_and_archive_notes.md).

D-2a result wording is intentionally conservative and configuration-bound: on the tested zero-shot prompt shape and current colorization scaffold, the within-window re-probe remained Gate-B fail; this removes the Stage-2 context-overflow confound for this configuration. No Emu3-Gen-wide or Emu3-family-wide inability is claimed.


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

Quality between fresh and cached pipelines: mean SSIM 0.811 across 5 validation queries. Token-level divergence is immediate (inherent to AR generation stochasticity), but both pipelines produce valid depth maps.

### Speedup Mechanism Decomposition (AR-5)

The AR-4 5x speedup is **entirely explained by a DynamicNTK RoPE recomputation pathology** in the HuggingFace transformers implementation. `ChameleonDynamicNTKScalingRotaryEmbedding.forward()` recomputes `inv_freq` on every forward call when `seq_len > max_position_embeddings`, incurring a GPU-to-CPU synchronization (`torch.max(position_ids)`) plus tensor reallocation and `register_buffer()` on every step across all 32 layers --- 66,592 unnecessary sync points for a 2-demo generation.

Controlled decomposition on 2-demo (2,081 generated tokens):

| Config | Gen (s) | s/tok | Total | vs Baseline |
|---|---|---|---|---|
| A. Original DynamicNTK (fresh) | 976.9 | 0.469 | 977.3s | 1.00x |
| B. Pre-computed NTK (fresh) | 81.5 | 0.039 | 81.9s | **11.93x** |
| C. Cached, original DynamicNTK | 98.6 | 0.047 | 100.4s | 9.74x |
| D. Cached + pre-computed NTK | 80.9 | 0.039 | 81.1s | 12.04x |

Pre-computing `inv_freq` at the expected max sequence length (B) eliminates the pathology entirely, achieving 12x speedup without any caching infrastructure. Caching alone (C) provides a 10x practical workaround. Once the RoPE fix is applied, caching adds no further benefit (B ≈ D). The same pattern holds at 3-demo: pre-computed NTK yields 6.8x vs original DynamicNTK.

The fix is trivial: compute the NTK-adjusted base frequency once at model initialization rather than on every forward call. Prefix caching remains useful operationally (warm-start for interactive workflows) but cannot be claimed as a latency optimization for this architecture.

### Understanding Task Generalization (AR-UND)

Tested whether prefix caching generalizes to multi-image understanding tasks with text-only output. The caching mechanism is output-modality agnostic: the DynamicNTK pathology and caching workaround affect all token types equally. However, Lumina-mGPT-2.0 cannot generate coherent text for understanding tasks --- it produces image-mode special tokens regardless of prompt structure. True understanding-task validation requires a model fine-tuned for text generation.

**Status**: Mechanism resolved on Lumina-mGPT-2.0. Chameleon bring-up pending for generalization.

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
  ar5_rope_decomposition.py          # Speedup mechanism decomposition
  ar_und_understanding.py            # Understanding task generalization
  emu3_a1_probe.py                   # Phase A A-1 Emu3-Chat deployment audit
  emu3_a2_gqa_scaling.py             # Phase A A-2 GQA-scaling sweep
  emu3_a3a_mibench_scaling.py        # Phase A A-3a MIBench scaling sweep
  d1_anchor_validation.py            # D-1 Emu3 understanding-freeze anchor validation
  d2a_capability_runner.py           # D-2a Emu3 generation-capability runner (active)
docs/
  current_project_state.md          # Current branch state (tracked)
  storage_and_archive_notes.md      # Repo hygiene and archive policy (tracked)
  results.md                        # Full DiT/AR results tables
  milestone_summary.md              # Per-milestone summaries
  plots/                            # DiT track visualizations
  examples/                         # Curated Qwen comparisons
```

Raw evidence, recon notes, and planning state are held locally in the working tree and are not part of git history by convention. See [`docs/storage_and_archive_notes.md`](docs/storage_and_archive_notes.md) for the tracked-vs-local map.

## Hardware

All experiments on NVIDIA B200 (183 GB VRAM). Models loaded in bf16 without quantization.

- **Qwen**: 57.8 GB model, 63.2 GB peak
- **Kontext**: 34.4 GB model, 37.5 GB peak
- **Lumina-mGPT**: 48.7 GB model (0-shot), up to 93.9 GB peak (3-demo cached)
