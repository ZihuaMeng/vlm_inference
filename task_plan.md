# Task Plan

## AR Mainline — AR-4: Demo-Prefix Caching + Position Encoding Extension — COMPLETED

Started 2026-04-04. Extends position encoding for multi-demo generation and implements demo-prefix KV caching for amortized latency reduction.

### Objective
1. Extend position encoding to support 2-3 demo contexts (removing the hard 10,240 token limit).
2. Implement demo-prefix KV caching: compute demo KV once, reuse across queries.
3. Measure amortized per-query latency (cached vs fresh) — first real wall-clock acceleration.
4. Validate generation quality across all conditions.

### Scientific Questions
1. Can Lumina-mGPT's RoPE be extended to 2-3 demo contexts without quality collapse?
2. Does cached demo-KV produce identical/near-identical generation to fresh-computed KV?
3. What is the amortized per-query latency with cached demos vs full-prefill baseline?
4. How does generation quality scale with 1, 2, 3 demos?

### Key Technical Context
- max_position_embeddings = 10,240 (config.json)
- rope_theta = 10,000.0, no rope_scaling configured
- 1-demo prompt = 6,293 tokens, 2-demo ≈ 10,456, 3-demo ≈ 14,619
- ChameleonRotaryEmbedding: no position clamping, allows extrapolation
- Dynamic NTK scaling code exists in transformers but not enabled
- Generation adds up to 2,081 tokens → 2-demo total ≈ 12,537, 3-demo ≈ 16,700

### Phases
1. [x] Phase 1: Position encoding extension — Dynamic NTK scaling (factor=2.0), all 3 configs generate valid images
2. [x] Phase 2: Demo-prefix KV caching — Validated: warm SSIM 0.962-0.975, caching mechanism correct
3. [x] Phase 3: Latency measurement — 5.36x speedup (2-demo), 5.18x (3-demo), generation pathology discovered
4. [x] Phase 4: Quality validation — 5-query mean SSIM 0.811, consistent quality across queries

### Key Results
- **5.36x speedup** for 2-demo, **5.18x** for 3-demo (cached vs fresh pipeline)
- Speedup comes from generation, not prefill: NTK-scaled fresh generation pathologically slow (~544s) vs cached (~97s)
- Prefix caching is essential for multi-demo ICL (not just an optimization)
- Mean SSIM 0.811 across 5 validation queries — quality preserved
- Dynamic NTK scaling (factor=2.0) extends to 16,702 tokens (3-demo + gen)

---

## AR Mainline — AR-3: Generation-Informed Sparse Prefill Prototype — COMPLETED

Started 2026-04-04. Tests whether AR-1's attention characterization can inform a sparse prefill policy that reduces demo-image prefill cost while preserving generation quality.

### Objective
1. Test whether restricting demo-token attention to high-demo heads during prefill reduces prefill computation.
2. Determine whether the resulting KV cache preserves generation quality.
3. Compare to AR-2's generation-time KV compression: does prefill sparsification trigger the same chaos?
4. Map the quality-sparsity tradeoff across conservative/moderate/aggressive configs.
5. Test whether head-selection outperforms uniform at matched sparsity (as it did in AR-2).

### Scientific Questions
1. Does restricting demo-token attention to high-demo heads during prefill reduce prefill latency?
2. Does this sparsified prefill preserve the KV cache quality that generation depends on?
3. How does generation quality compare to full prefill?
4. What fraction of prefill attention computation can be skipped while maintaining quality?
5. Does prefill sparsification trigger the same AR chaos as AR-2's generation-time KV compression?

### Key Design Insight
During causal prefill, demo tokens (positions 0-4163) NEVER attend to query tokens (4164+). Masking query→demo attention for specific heads does NOT change demo KV cache entries — only query positions' KV is affected. This is fundamentally milder than AR-2's demo KV zeroing.

### Sparsity Configurations (from AR-1 data)
- **baseline**: Full prefill (two-phase for fair comparison)
- **conservative**: Skip heads with <5% demo attn (91/1024 heads, 8.9%)
- **moderate**: Skip heads with <15% demo attn (271/1024 heads, 26.5%)
- **aggressive**: Skip heads with <30% demo attn (472/1024 heads, 46.1%)
- **uniform_moderate**: Random 271 heads (matched to moderate, no AR-1 guidance)
- **uniform_aggressive**: Random 472 heads (matched to aggressive)

### Mechanism
Per-head 4D attention masks via forward pre-hooks on decoder layers during prefill. For masked heads, attention from query positions to demo content positions is blocked. Generation runs UNMODIFIED with whatever KV cache the sparse prefill produces.

### Theoretical FLOPs Savings (Prefill)
- Conservative: 4.0% attention / 3.0% total FLOPs
- Moderate: 11.8% attention / 8.9% total FLOPs
- Aggressive: 20.6% attention / 15.5% total FLOPs

### Phases
1. [x] Phase 1: Design — Head classification from AR-1 data, sparsity configs, FLOPs calculation
2. [x] Phase 2: Implementation — ar3_sparse_prefill.py with 6 configs, hook-based 4D masking
3. [x] Phase 3: Evaluation — 3 cases × 6 configs on B200
4. [x] Phase 4: Analysis — Results, AR-2 chaos comparison, wall-clock assessment

### Key Results
- **Best config**: moderate (26.5% sparsity) — SSIM 0.905 mean
- **AR-3 vs AR-2**: SSIM 0.905 vs 0.692 (+31%) — prefill-side intervention far less destructive
- **Head-selective vs uniform**: mixed (+0.036 at moderate, -0.087 at aggressive)
- **Non-monotonic**: moderate > conservative > aggressive (same pattern as AR-2)
- **Prefill latency**: NOT reduced (hooks add overhead); theoretical 8.9% FLOPs savings unrealized
- **Verdict**: Quality results justify pursuing kernel-level sparse prefill (AR-4)

### Deliverables
1. ar3_sparse_prefill.py at PROJECT_ROOT/vlm_icl/
2. ar3_results.json at PROJECT_BLUE_ROOT/vlm_icl/outputs/ar3_sparse_prefill/
3. Per-config generated images at same output directory
4. Findings in findings.md under "AR Mainline — AR-3"

---

## AR Mainline — AR-2: Head-Selective Demo-KV Compression Prototype — COMPLETED

Started 2026-04-04. Training-free, per-head demo-KV eviction guided by AR-1 attention characterization.

### Objective
1. Test whether head-selective KV compression preserves generation quality better than uniform compression at matched retention rates.
2. Determine the optimal retention rate (25%, 50%, 75%) on the cost-quality Pareto frontier.
3. Evaluate demo sub-structure ablation (keep output image, drop input image).

### Scientific Questions
1. Does head-selective compression reduce KV cost while preserving quality?
2. At matched retention rate (50%), does head-selective outperform uniform?
3. What retention rate balances cost and quality?
4. Does demo sub-structure knowledge (input vs output image) help?
5. Is the quality signal (SSIM, token match) strong enough to guide method design?

### Retention Policies (from AR-1 data)
- **head_selective_25**: Top 25% heads by demo-attention (threshold = P75 ≈ 0.506). 256/1024 heads retain demo KV.
- **head_selective_50**: Top 50% heads (threshold = P50 ≈ 0.330). 512/1024 heads retain.
- **head_selective_75**: Top 75% heads (threshold = P25 ≈ 0.144). 768/1024 heads retain.
- **uniform_50**: All heads, random 50% of demo token positions zeroed.
- **substructure_50**: All heads, zero input-image KV (keep output-image KV). ~50% retention by structure.

### Mechanism
Zero out K and V at demo content positions for evicted heads/positions. Structural marker tokens (positions 0,1,2 and 4163, ~5% of attention) always preserved. Two-phase generation: manual prefill → compress cache → continue with model.generate().

### Phases
1. [x] Phase 1: Design — Retention policies from AR-1 per-head data, compression mechanism, two-phase generation approach
2. [x] Phase 2: Implementation — ar2_kv_compression.py with 6 configs, quality metrics, theoretical savings
3. [x] Phase 3: Evaluation — 3 cases × 6 configs on B200 (reduced from 5 due to cold-start speed)
4. [x] Phase 4: Analysis — Results tables, head-selective vs uniform comparison, operating point recommendation

### Key Results
- **AR generation is chaotic**: All compressed configs produce completely different token sequences from baseline (first_diff=1, token_match~2%). Quality cannot be assessed by reference-based pixel metrics.
- **Head-selective > uniform @50%**: SSIM 0.692 vs 0.622 (+0.070), consistent across all 3 cases. Validates AR-1 attention characterization.
- **Best operating point**: head_selective_50 (50% retention, P50 threshold ≈ 0.330, 33.1% theoretical KV reduction)
- **Non-monotonic quality**: 50% > 75% > 25% retention — suggests 50% concentrates attention on most informative heads
- **Substructure does NOT help**: Dropping all input-image KV (0.611) ≈ random position dropout (0.622)
- **No actual speedup**: K/V zeroing doesn't reduce tensor dimensions; would need sparse kernels or physical eviction

### Deliverables
1. ar2_kv_compression.py at PROJECT_ROOT/vlm_icl/
2. ar2_results.json at PROJECT_BLUE_ROOT/vlm_icl/outputs/ar2_kv_compression/
3. Per-config generated images at same output directory
4. Findings in findings.md under "AR Mainline — AR-2"

---

## AR Mainline — AR-1: X-Prompt Positioning + Demo Attention Characterization — COMPLETED

Started 2026-04-04, completed 2026-04-04. Empirical foundation for AR-side acceleration method design.

### Objective
1. Understand X-Prompt's mechanism, identify gaps, define novelty corridor.
2. Characterize how generated image tokens attend to demonstration context during AR generation.

### Phases
1. [x] Phase 1: X-Prompt Deep Inspection — paper analysis (code not released)
2. [x] Phase 2: Attention Characterization — instrument Lumina-mGPT, log per-layer/per-head attention mass on demo tokens during 1-shot generation

### Key Results
- Demo tokens receive **33.18%** of total attention (far from negligible)
- Deep layers (22-31) dedicate >50% attention to demos; shallow layers <20%
- Strong head specialization: top heads >80% demo attention, bottom heads <1%
- Temporal stability: demo attention decreases only 11% from early to late generation
- Demo output image receives 1.31x more attention than input image
- Novelty corridor defined: inference-time KV eviction guided by head-level attention patterns

### Scientific Questions
- How does X-Prompt compress demos? (learned bottleneck tokens + attention masking)
- What does X-Prompt NOT do? (no cross-demo redundancy, no attention analysis, no KV cache optimization, no position encoding solution)
- What fraction of attention goes to demo tokens during AR generation?
- Is demo attention concentrated in specific layers/heads?
- Does demo attention change during generation (early/mid/late)?
- Within demo, is attention on input image vs output image?

### Deliverables
1. X-Prompt analysis + novelty corridor in findings.md
2. Attention characterization tables in findings.md
3. Raw data at PROJECT_BLUE_ROOT/vlm_icl/profiling/
4. Analysis script at PROJECT_ROOT/vlm_icl/

---

## VLM In-Context Generation Profiling — COMPLETED

Started 2026-04-02. New workstream: deploy a multimodal in-context generation model, run depth estimation tasks, and produce a bottleneck profile to assess where our existing DiT-side acceleration methods apply.

### Objective
Determine the bottleneck structure of a multimodal in-context generation pipeline and assess whether our temporal caching / CFG-skip methods target the identified bottleneck.

### Deliverables
1. Latency breakdown (pipeline stage + within-stage decomposition)
2. Acceleration applicability assessment (do our methods target the bottleneck?)
3. Initial evaluation metric recommendations

### Phases
1. [x] Phase 1: Literature Review — Chameleon (blocked), Lumina-mGPT, Painter/SegGPT, Emu3
2. [x] Phase 2: Dataset Preparation — NYUDepthV2, 40 pairs, 10 task instances
3. [x] Phase 3: Model Deployment — Chameleon blocked (image gen disabled), deployed Lumina-mGPT-2.0-Omni
4. [x] Phase 4: Inference — 5 instances completed, all produced depth maps (512×512)
5. [x] Phase 5: Profiling — Stage decomposition + context-length scaling complete
6. [x] Phase 6: Analysis + Meeting Prep — Latency table, acceleration assessment, metric recommendations

### Fallback Rules
- If Chameleon blocks within ~3 hours → pivot to Lumina-mGPT
- If both block → document failures, escalate
- If OOM with 3-image context → try 2-image, then 1-image
- If dataset prep is complex → use 3 hand-picked examples

### Directory Structure
- Scripts: `PROJECT_ROOT/vlm_icl/`
- Weights: `PROJECT_BLUE_ROOT/vlm_icl/weights/`
- Data: `PROJECT_BLUE_ROOT/vlm_icl/data/`
- Outputs: `PROJECT_BLUE_ROOT/vlm_icl/outputs/`
- Profiling: `PROJECT_BLUE_ROOT/vlm_icl/profiling/`

---

## FLUX Track — F4: Conditioning-Aware KV Reuse — COMPLETED

Completed 2026-03-28. See findings.md "FLUX Track — F4" section for full results.

### Objective
Test whether caching the K/V projections of static conditioning tokens (text + reference image) across denoising steps in the plateau region produces meaningful speedup with better quality preservation than whole-step output caching.

### Answers to Scientific Questions
1. **Can condition-token K/V be identified and isolated?** YES. Linear projections are position-independent. Condition tokens occupy fixed positions (text: 512 tokens, image: 4096 tokens, noise: 4096 tokens).
2. **How much do condition-token K/V change across steps?** Image K/V changes 3-5x less than noise K/V (measured by 1-cos). Stable in all blocks except the most-dynamic single-stream block (S18).
3. **Does caching produce measurable speedup?** NO. 1.015x — negligible. K/V projection is ~11% of attention cost; the matmul dominates.
4. **Does it preserve quality better than whole-step caching?** YES, dramatically. SSIM 0.917 vs 0.804 (+0.112), wins 8/8 cases. LPIPS 0.050 vs 0.213 (4.3x better).
5. **Does speedup scale with image count?** Not testable (single image only). Token ratio is constant regardless.

### Key Finding (Ablation Value)
Condition-token K/V staleness contributes negligibly to temporal caching's quality loss. The quality degradation comes from stale attention matmuls and FFN, not stale K/V. This points to partial attention recomputation as the target for future improvement — but it requires custom kernels incompatible with Flash Attention.

### Tasks
1. [x] Phase 1: Code inspection — document token layout, K/V separability, feasibility per block type
2. [x] Phase 2: Measure K/V stability across steps (6 blocks × 28 steps)
3. [x] Phase 3: Implement KV caching prototype (custom attention processor, 57 blocks)
4. [x] Phase 4: Evaluate on 8 F2 cases (4 robust + 4 fragile), compare to temporal caching
5. [x] Update findings.md, progress.md, task_plan.md

---

## Cross-Model Analysis — F3: Cross-Model Comparative Analysis — COMPLETED

Completed 2026-03-28. See findings.md "Cross-Model Analysis" section for full results.

### Objective
Produce the paper's central comparative artifact: a structured cross-model analysis synthesizing all Qwen (M0–M9) and FLUX (F0–F2) findings into a coherent narrative with supporting tables.

### Answers to Scientific Questions
1. **What is the common structure across both models?** Temporal plateau (convergence → plateau → divergence), attention dominance (~75%), denoising loop concentration (>97%), bimodal quality distribution, step uniformity. All are structural to rectified flow matching inference.
2. **What differs, and why?** Guidance strategy (true CFG vs distilled), block structure (homogeneous vs heterogeneous-but-equal-cost), divergence severity (Qwen dramatic cos=0.816 vs Kontext mild cos=0.981), acceleration axes available (2 vs 1).
3. **What does the comparison reveal about method vs models?** Temporal caching is architectural — same plateau pattern on both. Quality boundary is method-intrinsic — same failure content types on both. Dual-axis access (Qwen) improves quality at comparable speedup vs single-axis (Kontext).
4. **What is the strongest defensible claim?** Temporal plateau is structural to rectified flow matching; content-type quality boundary is method-intrinsic (not model-specific).

### Tasks
1. [x] Bootstrap: read all planning files, load M9/F2 data, load redundancy/profiling JSONs
2. [x] Section 1: Profiling comparison (pipeline cost, per-step, module-level, 5.2x decomposition)
3. [x] Section 2: Temporal redundancy comparison (plateau boundaries, normalized overlay, divergence, block-level, CFG)
4. [x] Section 3: Acceleration comparison (all configs, dual-axis advantage, absolute timing, ablation)
5. [x] Section 4: Quality boundary analysis (bimodal distribution, content-type overlap, difficulty predictor, severity explanation)
6. [x] Section 5: Paper claim structure (5 claims with evidence and limitations)
7. [x] Section 6: Evidence gaps and remaining questions
8. [x] Update task_plan.md and progress.md

---

## FLUX Track — F2: Validation + Ablation Study — COMPLETED

Completed 2026-03-28. See findings.md "FLUX Track — F2" section for full results.

### Answers to Scientific Questions
1. **Does the 2.15x conservative speedup hold across diverse inputs?** YES. 2.154x ± 0.001 across all 12 cases. Perfectly consistent — independent of content or difficulty.
2. **Does mean SSIM ~0.88 hold, or does it shift?** It SHIFTS DOWN significantly. Mean SSIM = 0.754 (vs F1's 0.883). F1's 5-case suite was unrepresentatively optimistic.
3. **What is the quality distribution? Is it bimodal?** YES, bimodal. Robust cluster (0.83-0.99, structural content) and fragile cluster (0.37-0.74, fine-detail content). 33% SSIM > 0.90, 50% SSIM < 0.80.
4. **Are failure cases similar to or different from Qwen's?** SAME. Both models degrade on fine_texture, complex_scene, style_transfer, nature_detail, identity_preserving. This is a method limitation, not model-specific.
5. **Does the quality/speedup tradeoff match Qwen's?** WORSE. Kontext conservative (2.15x, 0.754 SSIM) underperforms Qwen combined (2.35x, 0.836 SSIM) because Kontext lacks the CFG axis and bears quality cost entirely on temporal caching.

### Tasks
1. [x] Bootstrap: verify B200, activate venv, confirm FLUX weights and LPIPS
2. [x] Design 12-case test suite (4 easy, 4 medium, 4 hard, 12 edit types)
3. [x] Write f2_validation.py (3 configs × 12 cases, SSIM + LPIPS)
4. [x] Run validation: 36 inference runs on B200
5. [x] Analyze: aggregates, quality distribution, failure analysis, cross-model comparison
6. [x] Update findings.md, progress.md, task_plan.md

---

## FLUX Track — F1: Baseline, Profiling, and Temporal Transfer Test — COMPLETED

Completed 2026-03-28. See findings.md "FLUX Track — F1" section for full results.

### Answers to Scientific Questions
1. **Baseline performance?** 9.17s wall-clock (28 steps, single pass). 5.2x faster than Qwen baseline. Peak VRAM 37.52 GB.
2. **Multi-image support?** No. Pipeline rejects multi-image input. Single reference image only.
3. **Profiling structure?** Denoising loop = 97.8%. Steps perfectly uniform (321.3ms ± 0.2ms). Attention = 73-75% of block time. Dual/single block cost ratio = 1.08x (not 2.4x as predicted).
4. **Temporal plateau?** CONFIRMED. Steps 4-23 (74% of trajectory). Same three-phase pattern as Qwen. Milder divergence (cos=0.981 vs Qwen's 0.816).
5. **Does temporal caching transfer?** YES. Conservative: 2.15x speedup, 0.883 SSIM. Aggressive: 3.50x, 0.800 SSIM. Same content-dependent quality pattern.
6. **Transfer verdict?** Temporal plateau is a property of rectified flow matching, not model-specific. Caching transfers directly.

### Tasks
1. [x] Bootstrap: install sentencepiece, download FLUX Kontext-dev weights (54 GB), verify B200
2. [x] Phase 1: Baseline timing (9.17s, 37.52 GB VRAM, multi-image test → fails)
3. [x] Phase 2: Three-level profiling (pipeline stages, per-step dynamics, module-level breakdown)
4. [x] Phase 3: Temporal redundancy characterization (plateau steps 4-23, 74% of trajectory)
5. [x] Phase 4: Temporal caching prototype (conservative 2.15x/0.883, aggressive 3.50x/0.800)
6. [x] Update findings.md, progress.md, task_plan.md with full results

---

## FLUX Track — F0: Kontext-dev Reconnaissance — COMPLETED

Completed 2026-03-28. See findings.md "FLUX Track — F0" section for full results.

### Scientific Questions Answered
1. **Architecture?** 12B rectified flow transformer. 19 dual-stream + 38 single-stream = 57 blocks. Hidden dim 3072. CLIP-L + T5-XXL text encoders. Guidance-distilled (single pass).
2. **Multi-image conditioning?** Sequence concatenation with per-image time-step offsets in 3D RoPE. Current release: single image only. Multi-image: architectural but unreleased.
3. **Guidance vs Qwen?** Guidance-distilled (1 pass/step) vs Qwen's true CFG (2 passes/step). CFG-skip (M5) is NOT APPLICABLE — the unconditional pass doesn't exist.
4. **Default steps/scheduler?** 28 steps, FlowMatchEulerDiscreteScheduler (same as Qwen). Velocity prediction with Euler integration.
5. **VRAM?** ~34.4 GB weights in bf16, ~45-55 GB peak estimated. Fits on B200 (183 GB). Does NOT fit on L4 (23 GB) without offloading.
6. **Dependencies?** Current venv compatible. Only missing: `sentencepiece` for T5 tokenizer.
7. **Transfer assessment?** Temporal plateau (M4) and temporal caching (M6): LIKELY. CFG-skip (M5): NOT APPLICABLE. Block uniformity (M3): does NOT transfer. Multi-image scaling (M7): likely but untestable.

### Tasks
1. [x] Read FluxKontextPipeline source code
2. [x] Read FluxTransformer2DModel source code (19 dual + 38 single blocks)
3. [x] Read FlowMatchEulerDiscreteScheduler source code
4. [x] Fetch HuggingFace model card and arXiv paper details
5. [x] Produce structured Qwen vs Kontext comparison table
6. [x] Assess transfer likelihood for each Qwen finding (M3-M9)
7. [x] Check dependency compatibility and estimate VRAM
8. [x] Create scripts/flux/ directory with architecture notes
9. [x] Write GPU validation plan for F1
10. [x] Update findings.md, progress.md, task_plan.md

---

## Qwen Track — Milestone 0: Workspace Init + Baseline Recon — COMPLETED

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

## Milestone 5: CFG-Skip Prototype + Robustness — COMPLETED

Completed 2026-03-21. See findings.md for full results.

### Answers to Scientific Questions
1. **What speedup?** 1.82x consistent across all 5 test cases. Matches theoretical 80/44 = 1.818x.
2. **Does it degrade quality?** Yes. SSIM 0.73–0.94 (mean 0.86). 3/5 moderate, 2/5 notable degradation.
3. **Consistent across test cases?** Speedup: perfectly consistent. Quality: varies — complex/textured prompts degrade more.
4. **Simplest effective schedule?** v1 drop-CFG (steps 3–38 skip) is simplest. v2 cached-uncond reuse failed catastrophically. Conservative schedule (steps 5–34 skip) slightly better quality, lower speedup (1.60x).

### Tasks
1. [x] Update task_plan.md with Milestone 5 objectives
2. [x] Activate venv, verify environment, install scikit-image for SSIM
3. [x] Write m5_cfg_skip.py (v1 drop-CFG) + m5_cfg_skip_v2.py (cached uncond reuse)
4. [x] Run experiments: v1 aggressive (5 cases), v2 (5 cases), v1 conservative (2 worst cases)
5. [x] Analyze results, update findings.md and progress.md

## Milestone 6: Temporal Step-Caching Prototype (Block-Aware) — COMPLETED

Started 2026-03-21. Temporal axis acceleration: cache transformer outputs across denoising steps.

### Scientific Questions
1. What speedup does temporal step-caching achieve?
2. Does block-aware caching (recompute dynamic blocks, cache stable) outperform uniform caching?
3. What is the quality cost, and which test cases degrade?
4. Do quality failure modes overlap with M5's CFG-skip failures (same hard cases)?
5. Is degradation correlated with input count or edit complexity?

### Design Decisions (from M4 block-level data)
- **Block 44**: Most dynamic (temporal cos ~0.973 in plateau). Always recompute.
- **Block 59**: Most stable (cos ~0.9999). Safe to cache.
- **Blocks 0, 14**: Very stable (cos >0.9996). Safe to cache.
- **Block 29**: Moderately stable (cos ~0.998). Safe to cache.
- **Dynamic zone**: Blocks 40-59 always recomputed (conservative — includes block 44 and all downstream so fresh output reaches norm_out/proj_out)
- **Cached zone**: Blocks 0-39 (all measured blocks in this range have cos >0.996)

### Three Schedules
- **A: Aggressive uniform** — Cache steps 5-30 (skip entire transformer, reuse prev noise pred). 28 passes vs 80 baseline → ~2.86x theoretical.
- **B: Conservative uniform** — Cache steps 10-25 (narrower window). 48 passes vs 80 → ~1.67x theoretical.
- **C: Block-aware** — Cache steps 5-30, but recompute blocks 40-59 (dynamic zone). Cache blocks 0-39. ~1.7-1.8x theoretical.

### Answers to Scientific Questions
1. **What speedup?** Aggressive uniform: 2.86x. Conservative uniform: 1.66x. Block-aware: 1.76x.
2. **Does block-aware outperform uniform?** No. Block-aware FAILED (SSIM 0.37). Uniform conservative is the best method (SSIM 0.93).
3. **Quality cost?** Conservative uniform: SSIM 0.89-0.97 (mean 0.93). Aggressive: SSIM 0.65-0.92 (mean 0.78).
4. **Overlap with M5 failures?** Yes. tc3 (complex scene) and tc5 (texture) are hardest on both axes.
5. **Input count correlation?** No. Degradation driven by edit complexity, not image count.

### Tasks
1. [x] Update task_plan.md with M6 objectives
2. [x] Write m6_temporal_cache.py
3. [x] Run experiments: baseline + 3 schedules × 5 test cases
4. [x] Analyze results, cross-axis comparison with M5
5. [x] Update findings.md and progress.md

## Milestone 7: Multi-Image Condition Redundancy Characterization — COMPLETED

Started 2026-03-21. Determine whether multi-image conditioning introduces structured redundancy absent in single-image conditioning.

### Scientific Questions
1. How are multiple image conditions represented internally? (Concatenated? Independent? Where does cross-image interaction occur?)
2. How does compute cost scale with input image count? (Linear? Super-linear?)
3. Is there quantifiable redundancy in how the model processes multi-image conditions? (Attention convergence? Hidden state similarity?)
4. Does the temporal or CFG redundancy structure change with more images?
5. Does multi-image conditioning create exploitable redundancy, and if so, what kind?

### Mechanical Structure (from code reading)
- **Text encoder**: Each image gets `"Picture N: <|vision_start|><|image_pad|><|vision_end|>"` in prompt. Qwen2.5-VL processes all images jointly.
- **VAE path**: Each image individually VAE-encoded, packed to 2x2 patch tokens, then all image latents **concatenated along sequence dimension** (dim=1).
- **Transformer input**: `hidden_states = [noise_latents, img1_latents, img2_latents, ...]` (single concatenated sequence).
- **Timestep modulation** (`zero_cond_t`): noise tokens get full timestep modulation (index=0), ALL condition image tokens get zero-timestep modulation (index=1). No per-image differentiation.
- **RoPE**: Output noise tokens get positive frame index. ALL condition images get negative frame index (-1). Condition images share frame position, differ only in spatial (h,w) positions.
- **Joint attention**: Text and image streams separately projected to Q,K,V, then concatenated `[text, image]` for full joint attention. Attention cost is O(n²) in total sequence length.
- **Output**: Only the first `noise_latents.size(1)` tokens of transformer output used as noise prediction.

### Key Architectural Insight
Multi-image conditioning is pure sequence concatenation with no special cross-image mechanism. Adding N images linearly increases sequence length, causing quadratic attention cost scaling. Cross-image interaction is entirely implicit through the joint attention mechanism.

### Answers to Scientific Questions
1. **How are multi-image conditions represented?** Pure sequence concatenation. Each image → 4,096 tokens, all concatenated along dim=1. Shared frame index (-1), shared zero-timestep modulation. No special cross-image mechanism.
2. **How does compute scale?** O(S^1.4): 1img=26.9s, 2img=47.3s (1.76x), 3img=72.3s (2.69x). Super-linear due to quadratic attention, sub-quadratic due to FlashAttention + linear MLP cost.
3. **Is there cross-image redundancy?** No. Hidden states are near-orthogonal (cos ≈ 0) at blocks 0-44. Cross-image attention is only 10-15%. Each image carries genuinely distinct information.
4. **Does temporal/CFG redundancy change with image count?** No. Same three-phase temporal pattern, same CFG plateau. Skip schedules transfer directly.
5. **Bottom line?** No exploitable multi-image redundancy exists. Differentiation must come from adaptive temporal/CFG combination.

### Tasks
1. [x] Inspect pipeline and transformer code for multi-image handling
2. [x] Write m7_multi_image.py instrumentation script
3. [x] Run instrumented passes for 1, 2, 3 images
4. [x] Analyze results: compute scaling, attention, cross-image redundancy, temporal/CFG interaction
5. [x] Update findings.md and progress.md with full analysis

## Milestone 8: Adaptive Combined Acceleration Prototype — COMPLETED

Started 2026-03-21. Combined temporal caching + CFG-skip with adaptive quality-aware control.

### Scientific Questions
1. Does combining temporal caching + CFG-skip achieve higher speedup than either axis alone?
2. Does an adaptive policy (modulating skip/cache decisions based on a runtime signal) improve quality robustness compared to fixed schedules?
3. How does the combined speedup scale across 1, 2, and 3-image inputs?
4. What is the quality/speedup Pareto frontier of the combined method?
5. Can the adaptive policy preserve quality on the hard cases (complex/textured edits)?

### Design
- **Fixed combined**: M6 conservative temporal (cache steps 10-25) + M5 conservative CFG-skip (skip steps 5-9, 26-34). Total: 34/80 passes.
- **Adaptive combined**: Per-step cosine similarity signal. Three tiers: temporal cache (cos>0.999, reuse prev), CFG-skip (cos>0.995, cond only), full compute (low similarity). Max consecutive cache: 4 steps.

### Answers to Scientific Questions
1. **Combined > single axis?** Yes. Fixed combined 2.35x > M5 1.82x and M6 conservative 1.66x individually. Adaptive 3.95x > M6 aggressive 2.86x.
2. **Adaptive improve quality?** No. Adaptive is MORE aggressive and degrades quality MORE on hard cases (tc5 SSIM: fixed 0.832 vs adaptive 0.693). Fixed mode has lower SSIM variance (0.085 vs 0.093).
3. **Image-count scaling?** Fixed: constant 2.35x regardless of image count. Adaptive: variable (3.19-4.98x). Absolute time saved scales with images (15s→27s→42s for 1→2→3 imgs).
4. **Pareto frontier?** Two operating points: quality-first (fixed: 2.35x, 0.869 SSIM) and speed-first (adaptive: 3.95x, 0.809 SSIM).
5. **Adaptive protect hard cases?** No. Cosine similarity in latent space does not correlate with perceptual quality for complex/textured content.

### Tasks
1. [x] Review M5/M6 results, M4 redundancy data, design combined schedules
2. [x] Write m8_combined.py with fixed and adaptive modes
3. [x] Run Phase 1: 7 test cases × 3 modes (21 runs)
4. [x] Run Phase 2: 3 scaling cases × 3 image counts × 3 modes (27 runs)
5. [x] Analyze results, update findings.md and progress.md

## Milestone 9: Broad Validation + Ablation Study — COMPLETED

Started 2026-03-21. Validate fixed combined strategy on diverse 18-case test suite. Produce ablation study and failure characterization.

### Answers to Scientific Questions
1. **Does 2.35x hold?** Yes. 2.353x ± 0.001 across all 18 cases. Perfectly consistent — independent of content, difficulty, or image count.
2. **Does SSIM ~0.87 hold?** No. Mean SSIM drops to 0.836 (from 0.869) with more diverse inputs. The original 7 cases under-represented hard content types.
3. **Quality distribution?** SSIM > 0.90: 39% (7/18). SSIM > 0.85: 61% (11/18). SSIM < 0.80: 33% (6/18). Bimodal: robust cluster (0.83–0.96) and fragile cluster (0.64–0.80).
4. **Component contribution?** Temporal caching is the better single axis (1.667x, 0.885 SSIM vs CFG-skip 1.600x, 0.851). Combining does NOT compound quality loss — overlapping vulnerability means actual combined SSIM (0.836) far exceeds independent prediction (0.753).
5. **Absolute time scaling?** Savings proportional to image count: 15s (1-img), 27s (2-img), 42s (3-img). 2-image has best quality (0.896 SSIM), likely due to difficulty confound.
6. **What makes a case hard?** Fine-grained visual detail: complex scenes (0.637), style transfer (0.642), material texture (0.665), nature scenes (0.687). Structural/semantic content is robust (objects: 0.947, portraits: 0.911, arrangements: 0.955). Difficulty rating is the strongest predictor: easy=0.943, medium=0.886, hard=0.753.

### Design
- **18 test cases**: 6×1-image, 6×2-image, 6×3-image. Covers: simple objects, landscape, style transfer, text/logo, material/texture, portrait, object placement, scene composition, texture detail, identity-preserving, object replacement, background change, arrangement, multi-portrait, nature scene, interior design, complex scene, abstract. Includes original 5 M5/M6 cases + 2 M8 cases + 11 new cases.
- **4 configurations per case**:
  a. Baseline: full 40-step, full CFG (80 passes)
  b. CFG-skip only: M5 conservative (skip uncond steps 5-34, 50 passes, 1.60x theoretical)
  c. Temporal cache only: M6 conservative (cache steps 10-25, 48 passes, 1.67x theoretical)
  d. Fixed combined: M8 (temporal 10-25 + CFG-skip 5-9,26-34, 34 passes, 2.35x theoretical)
- **Metrics**: Wall-clock time, speedup, SSIM, LPIPS (perceptual distance)
- **Identical seeds** (SEED=42) for all paired comparisons

### Tasks
1. [x] Bootstrap environment: B200 GPU, venv, install LPIPS
2. [x] Write m9_validation.py evaluation script
3. [x] Run 18 cases × 4 configs = 72 inference runs (~40 min on B200)
4. [x] Analyze results: ablation, quality distribution, scaling, failure characterization
5. [x] Update findings.md and progress.md with full analysis
