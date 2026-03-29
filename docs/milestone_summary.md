# Milestone Summary

## Qwen-Image-Edit Track (M0-M9)

### M0: Workspace Init + Baseline Recon

Set up dual-tier project layout (home for code, blue storage for weights/outputs). Verified compute environment (NVIDIA L4, 23 GB VRAM). Cloned upstream Qwen-Image repo and identified `QwenImageEditPlusPipeline` as the multi-image editing entry point. Selected Qwen-Image-Edit-2511 as the target model.

### M1: Environment Setup + Feasibility Gate

Installed full dependency stack (torch 2.10.0+cu128, diffusers 0.38.0.dev0, transformers 5.3.0). Tested three loading strategies on L4: bf16 direct (OOM), model CPU offload (OOM), sequential CPU offload (success at 728s/pass, 3.98 GB peak VRAM). Confirmed model runs end-to-end. L4 is viable only with sequential offload.

### M2: B200 Bootstrap + Native-Speed Baseline

Migrated to NVIDIA B200 (183 GB VRAM). Loaded full bf16 model directly on GPU (57.76 GB allocated, 63.20 GB peak). Single 2-image forward pass: 51.2s (14.2x faster than L4 sequential offload). Inference is compute-bound on B200, with 120+ GB VRAM headroom.

### M3: Structured Baseline Profiling

Three-level profiling of the denoising loop. L1: denoising = 99.3% of total time. L2: steps perfectly uniform at 1184.5ms each; CFG doubles cost via 2 transformer passes per step. L3: within each pass, 60 transformer blocks = 99.8% of time; joint attention = 75.8% of each block. The dominant bottleneck is joint attention across the denoising loop (~75% of total inference).

### M4: Redundancy Characterization

Instrumented 5 sampled transformer blocks across 40 denoising steps. Temporal redundancy: strong three-phase pattern (converge, plateau at cos > 0.999, diverge). CFG redundancy: conditional and unconditional outputs near-identical for steps 3-38 (cos > 0.999). Combined theoretical ceiling: 50-70% compute reduction via CFG schedule + step caching.

### M5: CFG-Skip Prototype

Implemented CFG-skip (drop unconditional transformer pass at selected steps). Aggressive schedule (skip steps 3-38): 1.82x speedup, mean SSIM 0.86. Conservative schedule (skip steps 5-34): 1.60x speedup, better quality. A cached-unconditional variant (v2) failed catastrophically (SSIM 0.05-0.53). CFG-skip delivers real speedup but quality tradeoff is non-trivial at cfg_scale=4.0.

### M6: Temporal Step-Caching Prototype

Implemented noise-prediction-level temporal caching (reuse previous step's denoised output). Conservative uniform (cache steps 10-25): 1.66x, mean SSIM 0.93 --- best single-axis method. Aggressive uniform (cache steps 5-35): 2.86x, mean SSIM 0.78. Block-aware caching: catastrophic failure (SSIM 0.37). Temporal caching at the noise-prediction level outperforms CFG-skip on quality.

### M7: Multi-Image Conditioning Characterization

Instrumented token counts, compute scaling, cross-image similarity, and attention distribution across 1/2/3-image inputs. Token count scales linearly (+4,096 per image). Compute scales O(S^1.4) due to FlashAttention. Cross-image hidden states are near-orthogonal --- no exploitable redundancy. Temporal and CFG redundancy patterns are identical across image counts. Multi-image cases benefit more from existing acceleration (larger absolute savings).

### M8: Combined Acceleration Prototype

Integrated temporal caching (M6) + CFG-skip (M5) in a single denoising loop. Fixed combined: 2.35x speedup (34/80 passes), mean SSIM 0.869 on 7 cases. Adaptive combined (cosine-similarity-gated): 3.95x mean speedup but more variable quality (0.809 SSIM). Adaptive mode does not protect hard cases --- cosine similarity is a poor proxy for perceptual quality. Fixed combined is the recommended operating point.

### M9: Broad Validation + Ablation Study

Validated fixed combined method on 18 diverse test cases (6 per image count, 18 edit types, easy/medium/hard). Four configurations per case (baseline, CFG-skip only, temporal only, combined). Results: speedup generalizes perfectly at 2.353x. Mean SSIM = 0.836. Quality is bimodal: 61% of cases above SSIM 0.85, 33% below 0.80. Difficulty and visual complexity are the strongest predictors of degradation, not image count.

---

## FLUX Kontext-dev Track (F0-F4)

### F0: Architecture Reconnaissance

Documented FLUX.1 Kontext-dev architecture from source code and paper. Key differences from Qwen: 12B mixed-stream transformer (19 dual-stream + 38 single-stream blocks vs. 60 identical dual-stream), guidance-distilled (1 pass/step, no true CFG by default), 28 default steps, single-image conditioning via VAE-encoded reference concatenated in the sequence dimension. Identified FluxKontextPipeline entry point and confirmed all dependencies available in existing environment.

### F1: Baseline, Profiling, and Temporal Transfer Test

Established Kontext baseline on B200: 9.17s for 28 steps (5.2x faster than Qwen). Profiling confirms the same structural patterns: denoising loop = 97.8%, steps perfectly uniform (321.3ms +/- 0.2ms), attention = 73-75% of block time. Temporal redundancy shows the identical three-phase pattern: plateau at steps 4-23 (74% coverage, cos > 0.999), with milder final divergence (cos=0.981 vs. Qwen's 0.816). Transfer test: temporal caching at conservative (2.15x) and aggressive (3.50x) schedules works without architectural modification. Initial 5-case results: conservative SSIM 0.883 (later shown to be optimistic by broader testing in F2).

### F2: Validation Study (Broader Test Suite)

Expanded to 12 diverse test cases (4 easy, 4 medium, 4 hard). Conservative temporal caching (2.15x): mean SSIM 0.754, bimodal distribution with 50% of cases below SSIM 0.80. Aggressive (3.50x): mean SSIM 0.635. The same content types that degraded on Qwen (complex scenes, style transfer, fine texture, nature detail) also degrade on Kontext, confirming the quality boundary is method-intrinsic. Kontext degrades more severely on matched content types, attributable to single-axis loading and weaker error correction from milder final-step divergence.

### F3: Cross-Model Analysis

Synthesized all Qwen (M0-M9) and FLUX (F0-F2) findings into a structured comparison. Confirmed five shared structural properties (denoising dominance, step uniformity, attention fraction, plateau pattern, bimodal quality distribution) and five model-specific differences (guidance strategy, step count, model size, block structure, divergence severity). Established that Qwen's dual-axis access (CFG + temporal) yields both higher speedup and better quality than Kontext's single-axis (temporal only) because quality cost is distributed across independent redundancy dimensions.

### F4: Conditioning-Aware KV Reuse (Mechanistic Ablation)

Tested whether caching only the conditioning-token K/V projections (text + reference image, 53% of the attention sequence) preserves quality better than whole-step caching. Results: KV reuse achieves SSIM 0.917 (vs. 0.804 for temporal cache) but with negligible speedup (1.015x vs. 2.15x). The K/V projection savings (~11% of attention cost per step) are overwhelmed by the attention matmul, which still runs at full cost. Primary value is as an ablation study: quality loss in temporal caching comes from stale attention matmuls and FFN, not from stale conditioning K/V. This identifies partial attention recomputation as the relevant direction for future quality improvement.
