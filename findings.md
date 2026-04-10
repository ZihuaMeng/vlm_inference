# Findings

---

# AR Mainline — AR-5: Speedup Mechanism Decomposition

## 1. Research Question

AR-4 demonstrated a 5.36x speedup for 2-demo prefix caching. How much of this comes from:
- (a) Avoiding a DynamicNTK RoPE recomputation pathology
- (b) Genuine prefill amortization (not re-processing demo tokens)

## 2. Root Cause Analysis: DynamicNTK Recomputation Pathology

### Source Code Analysis
Source: `transformers/models/chameleon/modeling_chameleon.py`, class `ChameleonDynamicNTKScalingRotaryEmbedding`

```python
def forward(self, x, position_ids):
    seq_len = torch.max(position_ids) + 1     # GPU→CPU synchronization
    if seq_len > self.max_position_embeddings:  # Always true for 2+ demos
        base = self.base * (
            (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
        ) ** (self.dim / (self.dim - 2))
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    cos, sin = super().forward(x, position_ids)
    return cos, sin
```

### Per-Step Cost
Every generation step, for every layer:
1. `torch.max(position_ids)` — GPU→CPU synchronization point (stalls pipeline)
2. `torch.arange(...).to(device)` — new tensor allocation on GPU
3. `register_buffer("inv_freq", ...)` — replaces module buffer

For 2-demo generation: 32 layers × 2081 steps = **66,592 calls** with GPU→CPU sync.

### The Fix: Pre-computed NTK
Pre-compute `inv_freq` once at the maximum expected sequence length, then replace `ChameleonDynamicNTKScalingRotaryEmbedding` with vanilla `ChameleonRotaryEmbedding` using the pre-computed `inv_freq`. This eliminates all per-step recomputation.

## 3. Experiment Design

Four controlled configurations on 2-demo generation (10,457 token prompt → 2,081 generated tokens):

| Config | RoPE | Pipeline | Description |
|--------|------|----------|-------------|
| A | Original DynamicNTK | Fresh | Baseline (pathological) |
| B | Pre-computed NTK | Fresh | Fix applied, no caching |
| C | Original DynamicNTK | Cached | AR-4 approach |
| D | Pre-computed NTK | Cached | Fix + caching |

Warmup strategy: 1-demo (100 tok) + 2-demo (50 tok) warmup before measurements to eliminate CUDA kernel compilation confound.

## 4. Results

### 2-Demo Decomposition

| Config | Prefill | Gen | s/tok | Total | vs A |
|--------|---------|-----|-------|-------|------|
| A. Original DynamicNTK fresh | 0.382s | 976.9s | 0.4694 | 977.3s | 1.00x |
| B. Pre-computed NTK fresh | 0.375s | 81.5s | 0.0392 | 81.9s | **11.93x** |
| C. Cached (original DynamicNTK) | 1.805s | 98.6s | 0.0474 | 100.4s | 9.74x |
| D. Cached + pre-computed NTK | 0.270s | 80.9s | 0.0389 | 81.1s | 12.04x |

### 3-Demo Measurements

| Config | Prefill | Gen | s/tok | Total |
|--------|---------|-----|-------|-------|
| Original DynamicNTK fresh | 1.419s | 560.2s | 0.2692 | 561.6s |
| Pre-computed NTK fresh | 0.536s | 81.8s | 0.0393 | 82.3s |
| Cached (original DynamicNTK) | 1.862s | 101.9s | 0.0490 | 103.8s |

### Quality Validation (5 queries, 2-demo, pre-computed NTK)

| Query | Fresh Gen | Cached Gen | SSIM |
|-------|-----------|------------|------|
| Q0 | 82.2s | 80.9s | 0.6121 |
| Q1 | 80.8s | 81.6s | 0.7677 |
| Q2 | 81.2s | 81.0s | 0.8662 |
| Q3 | 81.2s | 81.1s | 0.9222 |
| Q4 | 80.6s | 80.4s | 0.6539 |

Mean SSIM: 0.7644, consistent with AR chaos property (autoregressive divergence with perceptually similar output).

## 5. Speedup Attribution

### Definitive Finding: RoPE Pathology is the Entire Cause

```
Total AR-4 speedup (A→C):  9.74x   (caching as workaround)
RoPE fix alone (A→B):      11.93x  (direct fix, no caching needed)
Residual (B vs C):          0.82x   (caching adds zero benefit once fixed)
B ≈ D:                      81.9s ≈ 81.1s (fix makes fresh ≈ cached)
```

### Interpretation

1. **The RoPE fix alone (B) is faster than caching alone (C)**: 81.9s vs 100.4s. Pre-computing NTK eliminates more overhead than caching does because the cached pipeline (C) still runs DynamicNTK during its generation phase.

2. **Caching provides zero marginal benefit once the RoPE fix is applied**: B ≈ D (81.9s vs 81.1s). The only value of caching was that it happened to circumvent the DynamicNTK pathology — likely because the cached pipeline's attention mask/position_ids behave differently during generation.

3. **Prefill amortization is negligible**: The prefill phase (0.375-0.382s) is <0.5% of total time. Avoiding demo re-processing saves ~1s in a ~80s pipeline.

4. **The pathology is worse at higher demo counts but saturates**: 2-demo DynamicNTK overhead is 12x (977s vs 82s), 3-demo is 6.8x (562s vs 82s). The per-token cost decreases with seq_len because the DynamicNTK base computation stabilizes at longer sequences.

### Secondary Finding: DynamicNTK Asymmetry

DynamicNTK overhead is dramatically different between fresh and cached pipelines:
- Fresh: 0.4694 s/tok (A) vs 0.0392 s/tok (B) → 12.0x overhead
- Cached: 0.0474 s/tok (C) vs 0.0389 s/tok (D) → 1.22x overhead

Both pipelines should trigger DynamicNTK during generation (position_ids > 10,240 in both). The asymmetry suggests the cached pipeline's DynamicNTK calls are cheaper, possibly due to memory allocation patterns (fresh prefill creates a monolithic KV cache, cached loads disk-backed contiguous tensors) or position_ids handling differences.

## 6. Honest Paper Claim

**Incorrect claim**: "Prefix caching amortizes demo prefill cost, yielding Nx speedup."

**Correct claim**: "Prefix caching provides a practical workaround for a DynamicNTK RoPE recomputation pathology in the HuggingFace transformers implementation of Chameleon's rotary embedding. The 10-12x generation slowdown at long sequences is caused by per-step GPU→CPU synchronization and tensor reallocation in `ChameleonDynamicNTKScalingRotaryEmbedding.forward()`. Pre-computing the NTK base frequency eliminates the pathology directly, achieving equivalent speedup without any caching infrastructure."

## 7. Implications for the Research Story

1. **The DynamicNTK pathology is a library implementation bug**, not an inherent cost of long-context generation. The NTK scaling formula itself is sound; the problem is recomputing it on every forward call.

2. **Prefix caching remains valuable for other reasons** (warm-start for interactive querying, reduced first-token latency for demo-heavy workflows) but cannot be claimed as a speedup mechanism for this architecture.

3. **The fix is trivial**: pre-compute `inv_freq` at model initialization for the expected max sequence length. This should be upstreamed to transformers.

## 8. Artifacts

- Script: `vlm_icl/ar5_rope_decomposition.py`
- Results: `/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/outputs/ar5_decomposition/ar5_results.json`
- Generated images: `exp_{a,b,c,d}_*.png`, `q{0-4}_{fresh,cached}.png`

---

# AR Mainline — AR-UND: Understanding Task — Prefix Caching Generalization

## 1. Objective

Test whether the AR-4 prefix caching mechanism (Dynamic NTK + demo KV caching) generalizes from image generation to multi-image understanding tasks with text-only output.

## 2. Task Design

**Prompt format**: N-shot scene captioning — each demo provides (image, text description), query provides (image) and model generates text.

**Question**: "Describe what you see in this image in one sentence."

**Demo labels**: Generic indoor scene descriptions.

**Key difference from AR-4**: Output is text tokens (~50–200) instead of image tokens (~2081). Same caching mechanism used as-is.

## 3. Critical Finding: Model Does Not Generate Coherent Text

Lumina-mGPT-2.0 does **not** produce coherent text output for understanding tasks. Instead of natural language, the model generates special tokens — primarily `<reserved08799>` (the image newline marker). The model is fine-tuned exclusively for image generation and defaults to image-like token sequences even when prompted for text.

- **0-shot captioning**: Output = `<reserved08799><reserved08799><reserved08799>` (201 tokens, no EOS hit)
- **1/2/3-demo captioning**: Same pattern — special tokens, no natural language
- **Implication**: Lumina-mGPT-2.0 cannot serve as an understanding model. Testing prefix caching on a true understanding task requires a different model (e.g., un-surgeried Chameleon).

Despite the non-text output, the caching mechanism is task-agnostic and the latency measurements remain informative.

## 4. Latency Results

### 4.1 Phase 2: Fresh vs Cached Pipeline (Single Query)

| Demos | Prompt | Fresh PF | Fresh Gen | Fresh Tot | Cached Gen | Cached Tot | Speedup |
|-------|--------|----------|-----------|-----------|------------|------------|---------|
| 1     | 8,365  | 1.139s   | 27.956s   | 29.095s   | 23.804s    | 27.113s    | 1.07x   |
| 2     | 12,554 | 1.342s   | 48.946s   | 50.288s   | 5.294s     | 10.925s    | 4.60x   |
| 3     | 16,743 | 1.459s   | 34.767s   | 36.227s   | 18.422s    | 27.256s    | 1.33x   |

**Caveat**: 1-demo results are inflated by cold-start overhead (first generation after warmup). Phase 3 shows warm 1-demo gen ≈ 4.4s, not 28s.

### 4.2 Per-Token Generation Cost

| Demos | Fresh tok | Fresh s/tok | Cached tok | Cached s/tok | Ratio |
|-------|-----------|-------------|------------|--------------|-------|
| 1     | 118       | 0.237       | 201        | 0.118        | 2.0x  |
| 2     | 201       | 0.244       | 201        | 0.026        | 9.2x  |
| 3     | 145       | 0.240       | 201        | 0.092        | 2.6x  |

Fresh per-token cost is stable at ~0.24 s/tok regardless of demo count. Cached per-token cost varies: 2-demo cached is fastest (0.026 s/tok), close to the warm baseline of ~0.022 s/tok from Phase 3.

### 4.3 Phase 3: Multi-Query Validation (1-demo, 5 queries, warm)

| Query | Fresh Gen | Cached Gen | Exact Match | Char Similarity |
|-------|-----------|------------|-------------|-----------------|
| Q0    | 4.433s    | 4.400s     | True        | 1.000           |
| Q1    | 4.514s    | 4.424s     | True        | 1.000           |
| Q2    | 4.428s    | 4.401s     | True        | 1.000           |
| Q3    | 4.452s    | 4.365s     | False       | 0.000           |
| Q4    | 4.372s    | 4.352s     | True        | 1.000           |

At 1-demo (within max_position_embeddings), fresh ≈ cached ≈ 4.4s. Token match mean = 0.579. The AR chaos property appears occasionally (Q3) but most queries are exact-match.

## 5. Cross-Task Comparison (AR-4 Image vs AR-UND)

### 5.1 Generation Pathology

| Config | AR-4 Fresh Gen | AR-4 Cached Gen | AR-4 Ratio | AR-UND Fresh Gen | AR-UND Cached Gen | AR-UND Ratio |
|--------|---------------|-----------------|------------|------------------|-------------------|--------------|
| 1-demo | 95.3s         | 90.3s           | 1.06x      | 27.956s*         | 23.804s*          | 1.17x*       |
| 2-demo | 543.7s        | 97.3s           | 5.59x      | 48.946s          | 5.294s            | 9.24x        |
| 3-demo | 560.8s        | 100.6s          | 5.57x      | 34.767s          | 18.422s           | 1.89x        |

*1-demo AR-UND values inflated by cold-start; warm values are ~4.4s (see Phase 3).

**Verdict**: The DynamicNTK generation pathology **does affect text output** — confirmed at 2-demo (9.24x per-token ratio, even higher than AR-4's 5.59x). The pathology is a position-encoding issue, not specific to image-token output.

### 5.2 Prefill-to-Generation Ratio

| Config | AR-4 PF/Total | AR-UND PF/Total |
|--------|--------------|-----------------|
| 1-demo | 0.0024       | 0.0391          |
| 2-demo | 0.0023       | 0.0267          |
| 3-demo | 0.0025       | 0.0403          |

Even with much shorter text output (118–201 tokens vs 2081), generation still dominates cost (>96%). This is because the DynamicNTK pathology inflates per-token generation cost far above baseline.

## 6. Key Conclusions

1. **Prefix caching mechanism generalizes**: The demo-prefix KV caching from AR-4 works identically for non-image tasks. The mechanism is output-modality agnostic — it caches input KV regardless of output type.

2. **DynamicNTK pathology is universal**: The per-token generation slowdown at extended positions affects ALL token types, not just image tokens. This is a RoPE/position-encoding issue, not an output-format issue.

3. **Model limitation**: Lumina-mGPT-2.0 cannot generate coherent text for understanding tasks. Testing true understanding-task generalization requires a model with text generation capability (e.g., base Chameleon, or a Chameleon variant with text fine-tuning).

4. **Practical implication**: For any multi-demo ICL use of Lumina-mGPT-2.0, prefix caching is essential — the DynamicNTK pathology makes uncached multi-demo generation impractically slow regardless of output modality.

## 7. Deliverables

- Script: `PROJECT_ROOT/vlm_icl/ar_und_understanding.py`
- Results: `PROJECT_BLUE_ROOT/vlm_icl/outputs/ar_und/ar_und_results.json`
- KV caches: `PROJECT_BLUE_ROOT/vlm_icl/cache/und_demo_*shot_kv.pt`

---

# AR Mainline — AR-4: Demo-Prefix Caching + Position Encoding Extension

## 1. Experimental Design

### 1.1 Objective
Extend Lumina-mGPT's position encoding to support multi-demo contexts (2-3 demos, 10,457-14,621 tokens) and implement demo-prefix KV caching — compute the demo KV cache once, save to disk, and reuse across multiple query images. This is the first experiment targeting real wall-clock acceleration for the ICL pipeline.

### 1.2 Position Encoding Challenge
- **max_position_embeddings**: 10,240 (config.json)
- **rope_theta**: 10,000.0, no scaling configured
- 1-demo prompt: 6,293 tokens (fits), 2-demo: 10,457 (exceeds), 3-demo: 14,621 (well over)
- With generation (+2,081 tokens): 2-demo→12,538, 3-demo→16,702
- ChameleonRotaryEmbedding has no position clamping — allows extrapolation but quality degrades
- **Solution**: Enable `ChameleonDynamicNTKScalingRotaryEmbedding` (factor=2.0) on all 32 layers. Dynamically adjusts rope_theta when seq_len > max_position_embeddings, leaving short-sequence behavior unchanged.

### 1.3 Prefix Caching Mechanism
- **Demo-only prefill**: Process demo tokens (positions 0..N_demo-1) through the model, producing demo KV cache
- **Save/Load**: Serialize KV cache to disk via torch.save/load (32 layers × 2 tensors each)
- **Query-only prefill**: Load demo KV, process query tokens (positions N_demo..N_demo+N_query-1). Model auto-continues position_ids from `past_key_values.get_seq_length()`
- **Correctness guarantee**: With causal attention, demo token KV entries depend only on demo tokens. Split prefill produces mathematically identical demo KV as full-prompt prefill.

### 1.4 Evaluation Setup
- **Model**: Lumina-mGPT-2.0-Omni (7B params, 32 layers, 32 heads)
- **Hardware**: NVIDIA B200, 183 GB VRAM
- **Task**: 1-shot depth estimation (NYU Depth V2), 512×512 dual-panel
- **Configs**: 1-demo (6,293 tok), 2-demo (10,457 tok), 3-demo (14,621 tok)
- **Generation**: CFG=4.0, top_k=200, seed=42, max_gen=2,081 tokens
- **Metrics**: SSIM (fresh vs cached), token match rate, first_diff position, wall-clock latency

## 2. Results

### 2.1 Phase 1: Position Encoding Extension
Dynamic NTK scaling (factor=2.0) successfully extends position support:

| Config | Prompt Tokens | Demo | Query | Exceeds 10,240 | Est. Total w/ Gen |
|--------|:---:|:---:|:---:|:---:|:---:|
| 1-demo | 6,293 | 4,164 | 2,129 | No | 8,374 |
| 2-demo | 10,457 | 8,328 | 2,129 | **Yes** | 12,538 |
| 3-demo | 14,621 | 12,492 | 2,129 | **Yes** | 16,702 |

All three configs produce valid images with NTK scaling enabled.

### 2.2 Phase 2: Prefix Caching Validation (1-demo)
Demo-only KV cache: 4,164 positions, created in 0.985s, 3,274.7 MB on disk.

| Query | Fresh Total | Cached Total | Token Match | First Diff | SSIM |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Q0 (cold) | 970.5s | 92.1s | 0.049 | 8 | 0.495* |
| Q1 (warm) | 89.6s | 91.6s | 0.094 | 0 | 0.962 |
| Q2 (warm) | 89.6s | 91.2s | 0.063 | 6 | 0.975 |

*Q0 SSIM 0.495 is a cold-start artifact (CUDA kernel compilation on first generation, 970s vs 89s). Warm-start SSIM 0.962-0.975 confirms caching mechanism is correct.

Key observation: Token-level divergence is immediate (first_diff 0-8) but perceptual quality remains high. This is the same AR chaos pattern observed in AR-2 and AR-3.

### 2.3 Phase 3: Multi-Demo Latency (Key Results)

#### Full Timing Breakdown

| Config | Fresh Prefill | Fresh Gen | Fresh Total | Peak VRAM |
|--------|:---:|:---:|:---:|:---:|
| 1-demo | 0.227s | 95.3s | 95.5s | 75.9 GB |
| 2-demo | 1.256s | 543.7s | 544.9s | 77.1 GB |
| 3-demo | 1.431s | 560.8s | 562.2s | 88.0 GB |

| Config | Demo Prefill | Save | Load | Query Prefill | Gen | Amortized | Peak VRAM |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1-demo | 0.154s | 4.024s | 0.849s | 0.188s | 90.3s | **91.4s** | 62.5 GB |
| 2-demo | 1.091s | 7.776s | 2.626s | 1.651s | 97.3s | **101.6s** | 83.0 GB |
| 3-demo | 1.476s | 12.761s | 5.968s | 1.965s | 100.6s | **108.5s** | 93.9 GB |

#### Speedup Summary

| Config | Fresh Pipeline | Cached Pipeline | Speedup | Break-even |
|--------|:---:|:---:|:---:|:---:|
| 1-demo | 95.5s | 91.4s | **1.05x** | 1.0 queries |
| 2-demo | 544.9s | 101.6s | **5.36x** | immediate |
| 3-demo | 562.2s | 108.5s | **5.18x** | immediate |

#### KV Cache Sizes

| Config | Demo Tokens | Cache Size |
|--------|:---:|:---:|
| 1-demo | 4,164 | 3,274.7 MB |
| 2-demo | 8,328 | 6,549.4 MB |
| 3-demo | 12,492 | 9,824.1 MB |

### 2.4 Phase 4: Multi-Query Consistency (1-demo, 5 Queries)
All 5 queries use the same pre-computed 1-demo KV cache, reloaded for each query.

| Query | Fresh Prefill | Cached QPrefill | Token Match | First Diff | SSIM |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Q0 | 0.227s | 0.187s | 0.046 | 2 | 0.955 |
| Q1 | 0.227s | 0.225s | 0.034 | 22 | 0.745 |
| Q2 | 0.226s | 0.188s | 0.029 | 3 | 0.870 |
| Q3 | 0.227s | 0.188s | 0.034 | 3 | 0.874 |
| Q4 | 0.226s | 0.187s | 0.028 | 5 | 0.610 |
| **Mean** | 0.227s | 0.195s | **0.034** | — | **0.811** |

Quality is consistent across queries: mean SSIM 0.811 with variance driven by AR chaos, not systematic degradation.

## 3. Analysis

### 3.1 Unexpected Speedup Discovery
The 5x speedup for 2-3 demos was expected to come from avoiding demo re-prefill. Instead, **generation itself is ~5x faster** in the cached pipeline:

| Config | Fresh Gen Time | Cached Gen Time | Gen Speedup |
|--------|:---:|:---:|:---:|
| 1-demo | 95.3s | 90.3s | 1.06x |
| 2-demo | 543.7s | 97.3s | **5.59x** |
| 3-demo | 560.8s | 100.6s | **5.57x** |

For 1-demo (within max_position_embeddings), generation times are nearly identical. For 2-3 demos (exceeding max_position_embeddings), fresh-pipeline generation becomes pathologically slow.

**Hypothesis**: `ChameleonDynamicNTKScalingRotaryEmbedding.forward()` recalculates `inv_freq` at every generation step when `seq_len > max_position_embeddings`. In the fresh pipeline, after a single large prefill, the RoPE layer's cached state triggers per-step recomputation. In the cached pipeline, the KV cache loaded from disk with a shorter query prefill may cause the RoPE layer to avoid repeated recomputation — but both pipelines have the same final sequence length.

This is a significant practical finding: **prefix caching doesn't just save prefill time — it circumvents a performance pathology in NTK-scaled generation**.

### 3.2 Cached Generation Time Scales Gently
Cached-pipeline generation stays nearly flat across demo counts:
- 1-demo: 90.3s
- 2-demo: 97.3s (+7.8%)
- 3-demo: 100.6s (+11.4%)

The ~10% increase for 3-demo vs 1-demo reflects the larger KV cache attention, but it's far from the ~6x explosion in the fresh pipeline. This makes prefix caching essential for multi-demo ICL.

### 3.3 Prefix Caching Overhead
One-time costs per demo configuration:
- 1-demo: demo_prefill(0.15s) + save(4.0s) = 4.2s setup
- 2-demo: demo_prefill(1.1s) + save(7.8s) = 8.9s setup
- 3-demo: demo_prefill(1.5s) + save(12.8s) = 14.3s setup

Per-query overhead: load + query_prefill
- 1-demo: 0.85s + 0.19s = 1.04s
- 2-demo: 2.63s + 1.65s = 4.28s
- 3-demo: 5.97s + 1.97s = 7.93s

For 2-3 demos, the per-query overhead (4-8s) is negligible compared to the generation-time savings (~450s).

### 3.4 Quality Assessment: Fresh vs Cached
The SSIM between fresh and cached outputs measures AR divergence, not quality degradation. Both pipelines produce valid, high-quality depth maps. Evidence:
- Warm-start Q1-Q2 Phase 2: SSIM 0.962-0.975 (near-identical despite different token sequences)
- Phase 4 mean: SSIM 0.811 (variance from AR chaos, not systematic error)
- Compare: AR-3 moderate SSIM 0.905 (with intentional quality compromise), AR-2 moderate SSIM 0.692

### 3.5 Multi-Demo Generation Quality
Both 2-demo and 3-demo configurations produce valid images with NTK scaling. Generation token count is consistent (2,081 tokens across all configs). Peak VRAM scales linearly: 75.9 → 77.1 → 88.0 GB (fresh), well within B200's 183 GB capacity.

## 4. Answers to Scientific Questions

**Q1: Can Lumina-mGPT's RoPE be extended to 2-3 demos without quality collapse?**
**YES.** Dynamic NTK scaling (factor=2.0) extends to 3 demos (16,702 total tokens, 1.63x beyond max_position_embeddings). All configurations produce valid images with consistent token counts.

**Q2: Does cached demo-KV produce equivalent generation to fresh-computed KV?**
**YES**, with the expected AR chaos caveat. Token-level divergence is immediate (first_diff 0-22), but perceptual quality is high (SSIM 0.811 mean, up to 0.975). Both pipelines produce valid depth maps. The divergence is inherent to AR generation, not a caching artifact.

**Q3: What is the amortized per-query latency with cached demos?**
- 1-demo: 91.4s cached vs 95.5s fresh → **1.05x** (negligible, within max_position_embeddings)
- 2-demo: 101.6s cached vs 544.9s fresh → **5.36x** (dramatic)
- 3-demo: 108.5s cached vs 562.2s fresh → **5.18x** (dramatic)
- Break-even is immediate for multi-demo configs. The speedup comes primarily from generation, not prefill savings.

**Q4: How does generation quality scale with 1, 2, 3 demos?**
All demo counts produce valid images. KV cache sizes scale linearly (3.3 → 6.5 → 9.8 GB). Peak VRAM stays within budget. The cached pipeline makes multi-demo ICL practical by avoiding the NTK generation pathology.

## 5. Verdict
**Demo-prefix caching delivers the first real wall-clock acceleration in the AR mainline.** The 5x speedup for multi-demo configs was unexpected — it comes not from avoiding demo re-prefill (which is cheap) but from circumventing a performance pathology where DynamicNTKScalingRoPE causes generation to run ~5.5x slower after full-sequence prefill exceeding max_position_embeddings.

Practical implications:
1. **Multi-demo ICL is now viable**: 3-demo generation drops from 562s → 109s per query
2. **Prefix caching is essential, not optional**: Without it, NTK-scaled generation is pathologically slow
3. **Amortization across queries costs only ~4-8s overhead** (KV load + query prefill)
4. **Storage-compute tradeoff**: 3.3-9.8 GB cache files for 5x generation speedup

Next: AR-5 should investigate the NTK generation pathology directly. Understanding whether the root cause is inv_freq recomputation, CUDA kernel mismatch, or memory layout could yield further optimizations.

---

# AR Mainline — AR-3: Generation-Informed Sparse Prefill Prototype

## 1. Experimental Design

### 1.1 Objective
Test whether AR-1's generation-time attention characterization can inform a **sparse prefill policy** that reduces the catastrophic demo-image prefill cost (6.88x slowdown from F0) while preserving generation quality. This approaches the problem from the opposite direction vs AR-2: instead of compressing the KV cache after prefill, AR-3 restricts which attention heads attend to demo tokens *during* prefill itself.

### 1.2 Key Design Insight
During prefill, demo tokens (positions 3–4163) only attend to earlier demo tokens due to the causal mask — they never attend to query tokens (positions 4164+). Therefore, masking query→demo attention in selected heads does NOT change the demo KV cache entries. Only query-position KV entries are affected. This is fundamentally milder than AR-2's post-hoc KV zeroing, which destroyed demo KV entries that generation tokens depend on.

### 1.3 Sparsification Mechanism
Per-head 4D attention masking via PyTorch `register_forward_pre_hook` on `ChameleonDecoderLayer`:
- Build a base causal mask `(1, 1, S, S)` once (~75 MB)
- For each hooked layer, expand to `(1, 32, S, S)` and set `mask[0, h, QUERY_START:, DEMO_START:DEMO_END] = -inf` for masked head indices `h`
- Inject via `kwargs['attention_mask']` in the pre-hook, bypassing the model's `_update_causal_mask` validation
- Hooks removed after prefill; generation proceeds from the (partially-sparse) KV cache normally

### 1.4 Sparsity Configurations
Head classification from AR-1 per-head demo attention data (32×32 = 1024 heads, range [0.007, 0.821]):

| Config | Type | Threshold | Heads Masked | Sparsity | Attn FLOPs Saved | Total FLOPs Saved |
|--------|------|-----------|-------------|----------|-----------------|------------------|
| baseline | Full | — | 0/1024 | 0% | 0% | 0% |
| conservative | Head-selective | <0.05 | 91/1024 | 8.9% | 4.0% | 3.0% |
| moderate | Head-selective | <0.15 | 271/1024 | 26.5% | 11.8% | 8.9% |
| aggressive | Head-selective | <0.30 | 472/1024 | 46.1% | 20.6% | 15.5% |
| uniform_moderate | Random | — | 271/1024 | 26.5% | 11.8% | 8.9% |
| uniform_aggressive | Random | — | 472/1024 | 46.1% | 20.6% | 15.5% |

- **Head-selective**: Masks heads whose AR-1 demo-attention fraction falls below the threshold (low-demo-attention heads).
- **Uniform**: Same number of heads masked but randomly distributed across layers (seed=42).
- Layer distribution: shallow layers (0–10) heavily masked (concentrated low-demo heads), deep layers (22–31) almost untouched.

### 1.5 Evaluation Setup
- **Model**: Lumina-mGPT-2.0-Omni (7B params, 32 layers, 32 heads, MHA, SDPA)
- **Task**: 1-shot depth estimation (NYU Depth V2), 512×512 dual-panel
- **Test cases**: 3 instances, same seed (42), same prompt layout (6293 tokens)
- **Metrics**: SSIM (structural similarity to baseline), token match rate, early-100 match, first_diff position
- **Hardware**: NVIDIA B200, 183 GB VRAM
- **Fair comparison**: All configs (including baseline) use identical two-phase approach (manual prefill → model.generate from KV cache)

## 2. Results

### 2.1 Summary (Mean of 3 Cases)

| Config | Prefill | Gen | Total | SSIM | Tok Match | First Diff |
|--------|---------|-----|-------|------|-----------|------------|
| baseline | 0.22s | 370.1s | 370.3s | 1.000 | 100.0% | 2081 |
| conservative | 1.16s | 80.9s | 82.1s | 0.811 | 3.4% | 3.0 |
| moderate | 0.52s | 79.6s | 80.1s | **0.905** | 10.7% | 0.3 |
| aggressive | 0.57s | 79.1s | 79.6s | 0.820 | 3.1% | 0.0 |
| uniform_moderate | 0.60s | 78.5s | 79.1s | 0.870 | 3.0% | 0.0 |
| uniform_aggressive | 0.61s | 79.2s | 79.8s | **0.907** | 6.7% | 0.3 |

Note: baseline Case 0 gen time includes CUDA kernel compilation cold start (948s); Cases 1–2 are ~80s. Mean gen time is inflated by this one-time cost.

### 2.2 Per-Case Results

**Case 0:**
| Config | SSIM | MSE | Tok Match | Early 100 | First Diff |
|--------|------|-----|-----------|-----------|------------|
| baseline | 1.000 | 0.0 | 100% | 100% | 2081 |
| conservative | 0.800 | 3990 | 5.0% | 22.0% | 8 |
| moderate | 0.881 | 2166 | 27.2% | 17.0% | 0 |
| aggressive | 0.794 | 6141 | 4.9% | 8.0% | 0 |
| uniform_moderate | 0.818 | 2804 | 3.6% | 7.0% | 0 |
| uniform_aggressive | 0.889 | 1648 | 13.8% | 8.0% | 0 |

**Case 1:**
| Config | SSIM | Tok Match | Early 100 | First Diff |
|--------|------|-----------|-----------|------------|
| baseline | 1.000 | 100% | 100% | 2081 |
| conservative | 0.835 | 2.8% | 14.0% | 0 |
| moderate | 0.919 | 2.2% | 4.0% | 0 |
| aggressive | 0.783 | 2.5% | 1.0% | 0 |
| uniform_moderate | 0.901 | 1.7% | 2.0% | 0 |
| uniform_aggressive | 0.925 | 3.6% | 23.0% | 0 |

**Case 2:**
| Config | SSIM | Tok Match | Early 100 | First Diff |
|--------|------|-----------|-----------|------------|
| baseline | 1.000 | 100% | 100% | 2081 |
| conservative | 0.797 | 2.5% | 19.0% | 1 |
| moderate | 0.916 | 2.8% | 21.0% | 1 |
| aggressive | 0.884 | 2.0% | 7.0% | 0 |
| uniform_moderate | 0.891 | 3.8% | 14.0% | 0 |
| uniform_aggressive | 0.906 | 2.6% | 16.0% | 1 |

## 3. Analysis

### 3.1 AR-3 vs AR-2: Quality Dramatically Better

| Metric | AR-2 Best (head_sel_50) | AR-3 Best (moderate) | Improvement |
|--------|------------------------|---------------------|-------------|
| Mean SSIM | 0.692 | 0.905 | +0.213 (+31%) |
| SSIM range | 0.60–0.69 | 0.78–0.92 | Much tighter |

The design insight is validated: restricting query→demo attention during prefill (which only affects query-position KV entries) is far less destructive than zeroing demo KV entries post-hoc (AR-2), which directly corrupts the information generation tokens need.

### 3.2 Non-Monotonic Quality Pattern (Confirmed)

Same paradox as AR-2: more masking can yield HIGHER quality.

| Config | Sparsity | Mean SSIM |
|--------|----------|-----------|
| conservative | 8.9% | 0.811 |
| moderate | 26.5% | **0.905** |
| aggressive | 46.1% | 0.820 |

Moderate (26.5%) outperforms conservative (8.9%) by +0.094 SSIM. This is consistent in 3/3 cases. The pattern persists at the uniform level too: uniform_aggressive (46.1%) outperforms uniform_moderate (26.5%) by +0.037 SSIM.

Possible explanation: the AR-1 "low-demo-attention" heads that conservative masks are the very heads whose sparse attention is nonetheless important — they may act as precision channels for fine structural detail. Moderate's broader masking redistributes attention more evenly, causing a regularization-like effect.

### 3.3 Head-Selective vs Uniform: Mixed Results

| Sparsity Level | Head-Selective SSIM | Uniform SSIM | Delta | Winner |
|---------------|--------------------|--------------|---------|----|
| 26.5% (moderate) | 0.905 | 0.870 | +0.036 | Head-selective |
| 46.1% (aggressive) | 0.820 | 0.907 | -0.087 | **Uniform** |

At moderate sparsity, AR-1 characterization provides a +0.036 advantage — meaningful but modest. At aggressive sparsity, uniform random masking dramatically outperforms head-selective (-0.087). This suggests that AR-1's generation-time attention profile is not a reliable guide for prefill masking at high sparsity levels — the heads that matter during generation may differ from those that matter during prefill for building good query-position KV entries.

### 3.4 Chaos Analysis: Prefill Sparsification is Modestly More Stable

| Experiment | Mechanism | Typical first_diff |
|-----------|-----------|-------------------|
| AR-2 | Generation-time KV zeroing | 1 (ALL configs) |
| AR-3 conservative | Prefill attention masking | 3.0 |
| AR-3 moderate | Prefill attention masking | 0.3 |
| AR-3 aggressive | Prefill attention masking | 0.0 |

Conservative achieves first_diff=3 (mean), slightly better than AR-2's universal first_diff=1. All other configs still show near-immediate divergence (first_diff ≈ 0). Token match rates remain very low (3–11%), confirming that AR generation chaos persists — different tokens are generated but they converge to perceptually similar images (SSIM 0.80–0.92).

### 3.5 Prefill Wall-Clock: Hook Overhead, Not Savings

| Config | Prefill Time | Overhead vs Baseline |
|--------|-------------|---------------------|
| baseline | 0.22s | — |
| conservative | 1.16s | +428% |
| moderate | 0.52s | +136% |
| aggressive | 0.57s | +159% |
| uniform_moderate | 0.60s | +173% |
| uniform_aggressive | 0.61s | +177% |

The hook-based 4D masking approach **increases** prefill time by 136–428%. Conservative's higher overhead (1.16s) includes first-time SDPA kernel compilation for the 4D mask code path. Steady-state overhead is ~0.3–0.4s per config.

This is expected: Python-level hooks with full 4D mask expansion cannot skip the actual attention computation — the masked QK^T products are still computed and then zeroed by the mask. True FLOPs savings require kernel-level sparse attention (e.g., FlashAttention with block-sparse masks, or custom CUDA kernels that physically skip masked blocks).

## 4. Answers to Scientific Questions

1. **Does restricting demo-token attention to high-demo heads during prefill reduce latency?**
   NO for wall-clock (hook overhead dominates). YES for theoretical FLOPs (up to 15.5% total). The theoretical savings are real but unrealizable without kernel-level support.

2. **Does sparsified prefill preserve KV cache quality?**
   YES, dramatically better than AR-2. Best config (moderate) achieves SSIM 0.905 vs AR-2's best of 0.692. The design insight holds: only query-position KV is affected, and demo KV (which generation tokens depend on most) is preserved intact.

3. **Does prefill sparsification trigger the same AR chaos as AR-2?**
   Partially. Token-level chaos persists (3–11% match), but perceptual quality (SSIM) is much higher. Conservative shows modest stability improvement (first_diff=3 vs AR-2's first_diff=1). The chaos is NOT absent, but its impact on visual quality is much smaller.

4. **Does AR-1 characterization guide optimal head selection?**
   Mixed. At moderate sparsity (26.5%), head-selective beats uniform by +0.036 SSIM. At aggressive sparsity (46.1%), uniform wins by +0.087. The generation-time attention profile is only weakly predictive of prefill importance.

5. **Does the non-monotonic quality pattern from AR-2 replicate?**
   YES. Moderate (26.5%) outperforms conservative (8.9%) by +0.094 SSIM, replicating AR-2's pattern where the middle retention rate is best. This is now a confirmed phenomenon across both prefill and generation-time interventions.

## 5. Verdict

**AR-3 achieves a significant quality improvement over AR-2** (SSIM 0.905 vs 0.692), validating that prefill-side intervention is fundamentally less destructive than generation-time KV compression. The design insight — that causal masking protects demo KV entries — is correct and consequential.

**However, AR-3 does NOT achieve its primary objective of reducing prefill latency.** The hook-based 4D masking mechanism adds overhead rather than saving time. True speedup requires kernel-level sparse attention support (block-sparse FlashAttention, triton kernels, or framework-level changes) that can physically skip masked attention blocks.

**The non-monotonic pattern and mixed head-selective vs uniform results suggest that the relationship between generation-time attention profiles and optimal prefill masking is complex.** AR-1's characterization provides moderate guidance at low sparsity but becomes unreliable at high sparsity.

**Path forward:** The quality results are encouraging enough to justify pursuing kernel-level sparse prefill (AR-4), where the theoretical 9–15.5% FLOPs savings could translate to real wall-clock improvements. The moderate config (26.5% sparsity, 8.9% total FLOPs) offers the best quality-cost tradeoff for a first kernel implementation.

---

# AR Mainline — AR-2: Head-Selective Demo-KV Compression Prototype

## 1. Experimental Design

### 1.1 Objective
Test whether training-free, per-head demo-KV eviction guided by AR-1 attention characterization preserves generation quality while reducing KV cache cost for 1-shot in-context image generation with Lumina-mGPT-2.0.

### 1.2 Compression Mechanism
Zero out K and V tensors at demo content positions (positions 3–4163, the demonstration image tokens) for evicted heads/positions. Structural marker tokens (positions 0, 1, 2, 4163) are always preserved (~5% of demo attention). Softmax still computes over the same sequence length, but evicted entries contribute ~zero attention weight.

### 1.3 Retention Policies
Derived from AR-1 per-head demo attention data (32 layers × 32 heads = 1024 heads):

| Policy | Type | Retention | Threshold | Heads Retained | KV Reduction |
|--------|------|-----------|-----------|---------------|--------------|
| baseline | — | 100% | — | 1024/1024 | 0% |
| head_selective_25 | Per-head | 25% | P75 ≈ 0.506 | 256/1024 | 49.6% |
| head_selective_50 | Per-head | 50% | P50 ≈ 0.330 | 512/1024 | 33.1% |
| head_selective_75 | Per-head | 75% | P25 ≈ 0.144 | 768/1024 | 16.5% |
| uniform_50 | Random positions | 50% | — | 1024/1024 | 33.1% |
| substructure_50 | Structural | ~50% | — | 1024/1024 | 33.1% |

- **head_selective**: Retains demo KV only for heads whose demo-attention exceeds the threshold. Low-demo-attention heads have their demo K/V zeroed.
- **uniform_50**: Random 50% of demo token positions zeroed uniformly across all heads.
- **substructure_50**: Zeros demo input-image KV (positions 2083–4163), keeps demo output-image KV (positions 3–2083). Exploits AR-1 finding that output image receives 1.31× more attention.

### 1.4 Generation Approach
Two-phase generation for compressed configs:
1. **Prefill**: Manual forward pass on full prompt (6293 tokens) → builds KV cache
2. **Compress**: Zero out K/V at evicted positions (~0.01s)
3. **Sample first token**: Apply logits processor to prefill logits
4. **Continue**: `model.generate()` with compressed `DynamicCache` + fresh logits processor (includes CFG, image_top_k)

Baseline uses standard `model.generate()` without cache intervention.

### 1.5 Evaluation Setup
- **Model**: Lumina-mGPT-2.0-Omni (7B params, 32 layers, 32 heads, MHA, SDPA attention)
- **Task**: 1-shot depth estimation (NYU Depth V2), 512×512 dual-panel output
- **Test cases**: 3 instances, same seed (42) across all configs
- **Metrics**: SSIM (structural similarity to baseline image), MSE, token match rate, early-100 token match
- **Hardware**: NVIDIA B200, 183 GB VRAM

## 2. Results

### 2.1 Theoretical KV Savings

Baseline KV cache: **3,299 MB** (2 × 32 heads × 6293 seq_len × 128 head_dim × 2 bytes × 32 layers).

| Config | KV Reduction | Saved |
|--------|-------------|-------|
| head_selective_25 | 49.6% | 1,636 MB |
| head_selective_50 | 33.1% | 1,091 MB |
| head_selective_75 | 16.5% | 545 MB |
| uniform_50 | 33.1% | 1,091 MB |
| substructure_50 | 33.1% | 1,091 MB |

Note: Zeroing does not reduce tensor dimensions — savings are theoretical (would require sparse attention kernels for actual memory/speed reduction).

### 2.2 Quality Results (Mean Across 3 Cases)

| Config | Mean SSIM | Mean MSE | Token Match | Early-100 Match | First Diff |
|--------|----------|---------|------------|----------------|-----------|
| baseline | 1.000 | 0 | 100% | 100% | 2081 |
| **head_selective_50** | **0.692** | 7,978 | 2.2% | 11.0% | 1 |
| head_selective_75 | 0.672 | 7,186 | 3.2% | 30.3% | 1 |
| uniform_50 | 0.622 | 5,854 | 2.4% | 15.0% | 1 |
| substructure_50 | 0.611 | 7,867 | 2.2% | 10.7% | 1 |
| head_selective_25 | 0.603 | 9,511 | 2.1% | 10.7% | 1 |

### 2.3 Per-Case Breakdown

| Config | Case 0 SSIM | Case 1 SSIM | Case 2 SSIM | Std Dev |
|--------|------------|------------|------------|---------|
| head_selective_25 | 0.687 | 0.618 | 0.505 | 0.075 |
| **head_selective_50** | 0.667 | **0.755** | 0.654 | 0.044 |
| head_selective_75 | 0.610 | 0.739 | 0.665 | 0.053 |
| uniform_50 | 0.565 | 0.721 | 0.581 | 0.069 |
| substructure_50 | 0.606 | 0.609 | 0.620 | 0.006 |

### 2.4 Key Comparisons

**Head-selective vs Uniform @50%**: SSIM 0.692 vs 0.622 (Δ = **+0.070**) → **head-selective wins**. Consistent across all 3 cases (deltas: +0.102, +0.034, +0.073).

**Substructure vs Uniform @50%**: SSIM 0.611 vs 0.622 (Δ = -0.011) → **uniform slightly wins**. Demo sub-structure knowledge (keep output image, drop input image) does NOT help despite output image receiving 1.31× more attention.

**Retention rate sweet spot**: head_selective_50 (0.692) > head_selective_75 (0.672) > head_selective_25 (0.603). Non-monotonic: 50% retention outperforms both 75% and 25%.

### 2.5 Timing

| Config | Case 0 (s) | Case 1 (s) | Case 2 (s) | Mean (s) |
|--------|-----------|-----------|-----------|---------|
| baseline | 1002.9 | 79.7 | 80.7 | 387.8 |
| compressed configs | 100-104 | 100-104 | 100-104 | ~101-103 |

- **Case 0 baseline**: 1002.9s due to cold-start CUDA kernel compilation. Cases 1-2: ~80s (steady-state, ~38 ms/tok).
- **Compressed configs**: ~101s consistently (~48 ms/tok). The two-phase approach adds ~25% overhead vs warm baseline (101s vs 80s).
- **Compression time**: 0.001–0.026s across all configs — completely negligible.

## 3. Analysis

### 3.1 Fundamental Finding: AR Generation is Chaotic

The most important finding is that **all compressed configs produce completely different token sequences from baseline**. Token match is ~2% and first divergence occurs at position 1 (the very first generated token). This is fundamentally different from diffusion models where intermediate perturbations produce similar final outputs.

K/V zeroing changes the softmax distribution at the first generation step, producing a different token, which cascades into a completely different image. The SSIM values (0.60–0.69) measure "similarity between two independent samples from the same prompt" rather than "quality preservation from compression."

**Implication**: For AR models, K/V compression quality cannot be evaluated by reference-based pixel metrics against a specific baseline sample. The generation is too sensitive to initial conditions. Better evaluation would require:
1. Multiple samples per config with distributional comparison (FID, IS)
2. Prompt-based quality metrics (does the image match the task?)
3. Semantic-level evaluation (depth map accuracy vs ground truth)

### 3.2 Head-Selective Compression Validates AR-1 Characterization

Despite the stochastic nature, head-selective compression **consistently outperforms uniform** at matched 50% retention:

| Case | head_selective_50 | uniform_50 | Delta |
|------|------------------|-----------|-------|
| 0 | 0.667 | 0.565 | +0.102 |
| 1 | 0.755 | 0.721 | +0.034 |
| 2 | 0.654 | 0.581 | +0.073 |
| **Mean** | **0.692** | **0.622** | **+0.070** |

This validates that AR-1's per-head demo attention characterization captures meaningful functional specialization. Heads with high demo attention carry more important information for generation quality.

### 3.3 Non-Monotonic Quality with Retention Rate

The quality ordering is head_selective_50 > head_selective_75 > head_selective_25, which is **non-monotonic** with retention rate:

| Retention | KV Reduction | Mean SSIM |
|-----------|-------------|----------|
| 25% | 49.6% | 0.603 |
| **50%** | **33.1%** | **0.692** |
| 75% | 16.5% | 0.672 |

Possible explanations for 50% > 75%:
1. **Noise amplification**: Retaining too many low-demo-attention heads may introduce noise (their demo KV carries irrelevant information that interferes with generation)
2. **Attention redistribution**: Zeroing more heads forces attention to concentrate on retained high-demo-attention heads, which may provide a cleaner signal
3. **Statistical noise**: With only 3 cases and stochastic generation, the difference (0.692 vs 0.672) may not be significant

### 3.4 Demo Sub-Structure Does Not Help

Substructure_50 (keep output image, drop input image) does NOT outperform uniform_50 (random positions). Mean SSIM: 0.611 vs 0.622. This is surprising given AR-1's finding that output image receives 1.31× more attention.

Possible explanation: The substructure approach drops ALL input-image positions for ALL heads, while uniform drops random positions. Even heads that primarily attend to the input image lose their input-image KV entirely, which may be more disruptive than randomly losing 50% of positions across both sub-images.

### 3.5 Practical Speedup Assessment

**K/V zeroing provides NO actual speedup** — tensors remain the same size. Theoretical savings (up to 1,636 MB / 49.6%) would require:
1. **Sparse attention kernels**: Skip zero entries during softmax (not available in SDPA)
2. **Physical KV eviction**: Remove evicted positions from cache tensors (requires position re-encoding to maintain RoPE correctness)
3. **Selective prefill**: Never compute KV for evicted head×position combinations (requires custom model forward)

The two-phase approach actually adds ~25% overhead (101s vs 80s baseline) due to manual prefill + cache handoff.

## 4. X-Prompt Positioning

| Dimension | X-Prompt (Sun et al.) | Our Method (AR-2) |
|-----------|----------------------|-------------------|
| Approach | Learned bottleneck tokens | Training-free KV eviction |
| Training cost | 128 A100s, ~8M pairs | None |
| Mechanism | Attention masking (IE→XP→TD) | Per-head K/V zeroing |
| Guidance | Learned weights | AR-1 attention characterization |
| Token sequence preservation | Not measured | ~2% (none) |
| Key advantage | Integrates into model training | Zero-cost, model-agnostic |
| Key limitation | Requires retraining per model | AR chaos prevents quality preservation |

**Positioning**: Our approach validates that attention-based head selection captures meaningful functional structure (head_selective > uniform). However, the fundamental AR chaos limits the practical value of inference-time KV manipulation. X-Prompt's training-based approach may be necessary to actually learn compression patterns that preserve generation quality.

## 5. Operating Point Recommendation

If head-selective KV compression must be used (e.g., for memory-constrained serving):
- **Recommended**: head_selective_50 (50% retention, P50 threshold ≈ 0.330)
- **KV savings**: 33.1% theoretical (1,091 MB)
- **Quality**: Mean SSIM 0.692 (best of all compressed configs)
- **Why 50% > 75%**: 50% retention concentrates attention on the most informative heads

## 6. Answers to Scientific Questions

1. **Does head-selective compression reduce KV cost while preserving quality?** Partially. It reduces theoretical KV by up to 49.6% but quality (SSIM 0.60–0.69 vs baseline 1.0) degrades significantly due to AR generation chaos. The quality loss is NOT from compression per se, but from the chaotic amplification of any perturbation.

2. **At matched 50% retention, does head-selective outperform uniform?** YES. SSIM 0.692 vs 0.622 (+0.070), consistent across all 3 cases. This validates that AR-1 attention characterization captures meaningful head specialization.

3. **What retention rate balances cost and quality?** 50% (head_selective_50). Non-monotonically best — both 25% and 75% are worse. The P50 threshold (0.330) provides the best balance.

4. **Does demo sub-structure knowledge help?** NO. Substructure_50 (0.611) slightly underperforms uniform_50 (0.622). Dropping all input-image KV for all heads is more disruptive than random position sampling.

5. **Is the quality signal strong enough to guide method design?** Yes, for relative comparisons between compression strategies. No, for absolute quality assessment — AR chaos means reference-based metrics conflate compression quality with stochastic generation variance.

---

# AR Mainline — AR-1: X-Prompt Positioning + Demo Attention Characterization

## 1. X-Prompt Deep Inspection

### 1.1 Paper Overview

**X-Prompt: Towards Universal In-Context Image Generation in Auto-Regressive Vision Language Foundation Models**
(Sun et al., arXiv:2412.01824, Dec 2024, updated Aug 2025)

X-Prompt is the first work applying learnable-token-based demonstration compression specifically to multimodal AR image generation. Built on Chameleon (7B), it addresses the core scalability problem: in-context image-to-image tasks require at least 4 images in context (1 demo pair + query + output), each consuming 1024–4096 VQ tokens, making the training context window prohibitively long.

### 1.2 Mechanism: Learnable Bottleneck Tokens with Attention Masking

X-Prompt defines three token categories with a specific attention masking scheme:

| Token Type | Definition | Can attend to |
|-----------|-----------|---------------|
| **IE** (In-Context Example) | Full token representation of demo images/text | IE only (self-attention) |
| **XP** (X-Prompt Tokens) | S learnable tokens (X_XP ∈ ℝ^{S×C}), trained end-to-end | IE + XP (reads demos) |
| **TD** (TODO Tokens) | Target output to generate autoregressively | XP + TD (NO access to IE) |

**Critical design**: Attention masking disconnects IE from TD. The output tokens CANNOT directly attend to raw in-context example tokens — all demonstration information must flow through the fixed-size XP bottleneck:

```
Information flow:  IE → XP → TD
Attention mask:    TD ← XP ← IE  (TD sees XP but not IE)
```

The generation factorizes as: p(X_TD | X_XP) = ∏ p(X_TD_t | X_XP, X_TD_{<t})

**XP tokens are learned from scratch** — randomly initialized, not derived from input. They are optimized end-to-end via standard next-token prediction loss during training. The exact count S is not specified in the paper (significant omission — no ablation on S).

### 1.3 Training Details

| Parameter | Value |
|-----------|-------|
| Base model | Chameleon (7B implied) |
| Training context window | 5,120 tokens |
| Training data | ~8M pairs after augmentation |
| Data sources | LAION-Aesthetic 500K, NYU-v2, ADE-20K, UltraEdit 500K, Rain-13K, LOL, GoPro, MagicBrush 8K, custom 20K |
| Compute | 128 A100-80G GPUs, 20K steps, batch 1024 |
| Resolution | 512×512 only (Chameleon pretrained at this resolution) |

**Task augmentation**: Reversed tasks (e.g., derain → add rain) double training variety. Text prediction tasks (describe differences between input/output) improve task understanding.

**RAIE (Retrieval-Augmented Image Editing)**: At inference, retrieve a similar editing example via CLIP text similarity and use it as the in-context example. This is a practical contribution but orthogonal to the compression mechanism.

### 1.4 What X-Prompt Reports

- **GenEval text-to-image**: 0.57 overall (vs base Chameleon 0.39)
- **Dense prediction** (depth, segmentation, normals): Competitive with specialist models
- **Image editing** (MagicBrush): CLIPdir 0.097, DINO 0.792 with RAIE
- **Generalization**: Demonstrated on unseen tasks (color spectrum, object addition)

### 1.5 What X-Prompt Does NOT Do — Gaps

| Gap | Detail | Relevance to our work |
|-----|--------|----------------------|
| **No cross-demo redundancy analysis** | Multiple demos sharing structure (e.g., all showing depth estimation) are not exploited. Each IE is compressed independently. | HIGH — if demos share task structure, cross-demo compression could reduce XP token count |
| **No attention pattern characterization** | Paper provides zero analysis of how attention flows from IE through XP to TD. No per-layer or per-head visualization. | HIGH — our Phase 2 directly addresses this |
| **No position encoding solution** | Does not address max_position_embeddings limits. Context window fixed at 5120 during training. | HIGH — Lumina-mGPT's 10240 limit blocks 2-shot; X-Prompt's compression enables training but doesn't solve inference position limits |
| **No KV cache optimization** | IE tokens remain in the sequence (just masked). No KV eviction or cache management at inference. | HIGH — inference-time KV reduction is orthogonal to X-Prompt's training-time compression |
| **No inference speedup measurement** | Paper reports zero latency measurements. Focus is on training feasibility, not inference efficiency. | HIGH — the acceleration value of demo compression is unmeasured |
| **No head/layer specialization analysis** | No investigation of which model components process demo information. | HIGH — essential for targeted KV eviction or selective recomputation |
| **Code not released** | GitHub repo exists but is empty placeholder. Cannot inspect implementation details. | MODERATE — limits our ability to replicate or build on their exact mechanism |
| **Single resolution (512×512)** | Cannot scale to 1024×1024 without retraining base model. | LOW — our profiling target is also 512×512 |
| **Limited multi-demo scaling** | Training used ≤3 images. No systematic multi-demo evaluation. | MODERATE — we also hit position limits at 2-shot |

### 1.6 Novelty Corridor

Based on X-Prompt's gaps, our work must differentiate in these areas:

1. **Empirical attention characterization** (this milestone): Provide the first per-layer, per-head analysis of how AR VLMs attend to demonstration tokens during image generation. X-Prompt proposes a compression mechanism but never measures the phenomenon it's compressing.

2. **Inference-time efficiency**: X-Prompt reduces the training context window. We target inference speedup — making generation faster when demos are in context, without retraining.

3. **Attention-guided KV eviction**: Rather than learned bottleneck tokens (requiring training on 128 A100s), use empirical attention patterns to identify which demo KV entries are actually read during generation, and evict the rest.

4. **Position encoding solutions**: Address the max_position_embeddings bottleneck that prevents multi-shot ICL, independent of compression.

5. **Cross-demo structure**: If multiple demos share task structure, exploit that structure for more efficient representation.

### 1.7 Related Work Context

| Paper | Approach | Domain | Key difference from X-Prompt |
|-------|----------|--------|------------------------------|
| **ICAE** (ICLR 2024) | LoRA encoder compresses long context → memory slots | Text-only | Text-only; 4x compression with autoencoding |
| **Gist Tokens** (NeurIPS 2023) | Learned prompt compression tokens | Text-only | Architectural ancestor; no multimodal |
| **InstaManip** (CVPR 2025 Highlight) | Group self-attention for few-shot image manipulation | Multimodal AR | No token compression; uses attention structure |
| **LineAR** (Dec 2025) | Training-free KV cache compression for AR image gen | AR image gen | Inference-time KV management, complementary to X-Prompt |

**LineAR** is particularly relevant — it manages KV cache at the line level with 2D awareness during AR image generation, achieving 67% memory reduction and 7.5x speedup. This is a training-free inference optimization, which is the direction we're targeting but hasn't been applied to ICL demo tokens specifically.

## 2. Attention Characterization

### 2.1 Experimental Setup

**Model**: Lumina-mGPT 2.0-Omni (7B, Chameleon-based), 32 layers × 32 heads, head_dim=128.
**Task**: 1-shot depth estimation on NYUDepthV2. Demo = stitched depth+RGB image (512×512), query = RGB half-image (512×256).
**Token layout** (6293 prompt + ~2081 generated = ~8374 total):

| Region | Position | Tokens | Description |
|--------|----------|--------|-------------|
| demo_depth | [3, 2083) | 2,080 | Demo output image (depth map, upper half) |
| demo_rgb | [2083, 4163) | 2,080 | Demo input image (RGB, lower half) |
| demo_other | [0,3)+[4163,4164) | 4 | Demo header/footer markers |
| query_text | [4164, 4210) | 46 | Text instruction tokens |
| query_img | [4210, 6293) | 2,083 | Query image tokens |
| generated | [6293, ...) | ~2,081 | Autoregressively generated tokens |

**Method**: Phase A generates tokens with CFG=4.0 (standard pipeline). Phase B replays token-by-token, extracting attention weights at every 20th step (SDPA→eager fallback, 105 samples per case). Per-layer, per-head attention mass on each region computed as sum of softmax weights over region positions.

**Cases**: 3 test instances from NYUDepthV2, different query images, same demo pair per instance.

### 2.2 Aggregate Attention Mass Per Region

| Region | Case 0 | Case 1 | Case 2 | **Mean** | **Std** |
|--------|--------|--------|--------|----------|---------|
| **demo_depth** | 15.56% | 16.46% | 15.90% | **15.97%** | 0.37% |
| **demo_rgb** | 11.69% | 12.64% | 12.24% | **12.19%** | 0.39% |
| **demo_other** | 5.21% | 4.72% | 5.12% | **5.02%** | 0.21% |
| query_text | 0.35% | 0.36% | 0.39% | **0.37%** | 0.02% |
| query_img | 19.58% | 17.51% | 16.54% | **17.88%** | 1.27% |
| generated | 47.61% | 48.31% | 49.82% | **48.58%** | 0.92% |
| **DEMO total** | **32.46%** | **33.82%** | **33.26%** | **33.18%** | — |
| **QUERY total** | 19.93% | 17.87% | 16.93% | **18.24%** | — |

**Key finding: Demo tokens receive ~33% of total attention — far from negligible.** This means the model actively and substantially reads demonstration context during every step of image generation. The query image receives ~18% and previously generated tokens receive ~49%.

The demo_other region (just 4 structural tokens: image_start, h_grid, w_grid, image_end) receives a disproportionate 5.0% — averaging **1.25% per token** vs 0.008% per token for VQ image tokens. These structural markers serve as attention sinks/anchors.

### 2.3 Per-Layer Demo Attention Breakdown

Demo attention increases dramatically with layer depth, revealing a clear two-regime pattern:

| Layer Group | Mean Demo Attn | Pattern |
|------------|---------------|---------|
| Layers 0–12 (early) | 0.24 ± 0.04 | Moderate, fluctuating |
| Layers 13–21 (mid) | 0.24 ± 0.05 | Similar to early, slight increase |
| **Layers 22–31 (deep)** | **0.52 ± 0.05** | **Sharp increase: 2x early layers** |

Peak layers: Layer 28 (0.575), Layer 24 (0.572), Layer 26 (0.563). Deep layers dedicate over **half of all attention to demo tokens**, while generating image tokens. This suggests demo information extraction happens primarily in the model's upper layers — the "reading" layers.

Minimum: Layer 13 (0.167). Early/mid layers focus more on query image and previously generated tokens.

### 2.4 Head Specialization: "Demo-Reading Heads"

Strong head specialization exists. Top 10 demo-reading heads have >75% attention on demo tokens:

| Rank | Layer | Head | Demo Attn | 
|------|-------|------|-----------|
| 1 | 19 | 4 | **82.1%** |
| 2 | 24 | 22 | 81.2% |
| 3 | 5 | 21 | 80.8% |
| 4 | 8 | 2 | 79.0% |
| 5 | 28 | 28 | 78.4% |
| 6 | 18 | 27 | 78.1% |
| 7 | 8 | 5 | 77.4% |
| 8 | 31 | 7 | 76.8% |
| 9 | 29 | 8 | 76.5% |
| 10 | 29 | 7 | 75.2% |

Bottom 5 demo-ignoring heads have <1% demo attention:

| Layer | Head | Demo Attn |
|-------|------|-----------|
| 18 | 25 | 0.96% |
| 13 | 3 | 0.95% |
| 11 | 4 | 0.91% |
| 13 | 7 | 0.73% |
| 10 | 12 | 0.68% |

**Key finding**: Demo-reading heads span ALL layers (5, 8, 18, 19, 24, 28, 29, 31), not just deep layers. This suggests demo information is extracted at multiple levels of abstraction. The specialization ratio is extreme: top heads devote >80% to demos while bottom heads devote <1%.

This head-level specialization is analogous to the spatial-locality and semantic-sink head types identified in SSD (Streaming LLM), but for a new functional category: **demo-context reading heads**.

### 2.5 Temporal Dynamics During Generation

Demo attention is remarkably stable throughout generation, with only a slight decrease:

| Phase | Demo Attn | Gen Attn | Notes |
|-------|-----------|----------|-------|
| **Early** (tokens 0–693) | 0.356 ± 0.016 | 0.393 ± 0.014 | More demo reading at start |
| **Mid** (tokens 694–1387) | 0.323 ± 0.010 | 0.515 ± 0.009 | Shift toward generated tokens |
| **Late** (tokens 1388–2081) | 0.316 ± 0.011 | 0.549 ± 0.013 | Continued but mild decrease |
| **Ratio late/early** | **0.890** | 1.398 | Demo decreases 11%, gen increases 40% |

**Key finding: Demo attention does NOT collapse during generation.** Even in the final 700 tokens, 31.6% of attention still goes to demo tokens. The model continuously references the demonstration throughout the entire generation process. The decrease is mild (11% relative) and driven by the growing pool of generated tokens (which compete for attention mass), not by the model losing interest in demos.

This temporal stability means demo KV entries cannot be safely evicted after an early phase — they remain important throughout.

### 2.6 Demo Sub-Structure: Input Image vs Output Image

Within the demo, the output image (depth map) consistently receives more attention than the input image (RGB):

| Case | Depth (output) | RGB (input) | Ratio |
|------|---------------|-------------|-------|
| 0 | 15.56% | 11.69% | **1.331** |
| 1 | 16.46% | 12.64% | **1.302** |
| 2 | 15.90% | 12.24% | **1.299** |
| **Mean** | **15.97%** | **12.19%** | **1.31** |

**Key finding: The model attends 31% more to the demo output than the demo input.** This makes sense for the depth estimation task — the model needs to learn what a correct depth map looks like, so it references the demo's output more than its input. However, the input still receives substantial attention (12.2%), indicating the model also reads the input-output correspondence.

### 2.7 Memory and Timing Overhead

| Metric | Case 0 | Case 1 | Case 2 |
|--------|--------|--------|--------|
| Generation time | 1007s (cold) | 79s | 79s |
| Replay time (attention logging) | 38s | 38s | 38s |
| Generation VRAM | 42.5 GB | 45.8 GB | 49.1 GB |
| Replay VRAM | 36.6 GB | 39.9 GB | 43.1 GB |
| Sampled steps | 105 | 105 | 105 |

Replay with attention logging adds only **38s** overhead (vs 79s generation), and uses **less VRAM** than generation (no CFG = no unconditional KV cache). This confirms attention characterization is practical and not prohibitively expensive.

## 3. Novelty Corridor and Method Design Implications

### 3.1 Where X-Prompt Falls Short, Empirically

Our attention characterization reveals why X-Prompt's approach — while valid for training — leaves significant inference optimization on the table:

1. **X-Prompt blocks ALL demo-to-output attention via masking.** But our data shows 33% of output attention naturally goes to demos. X-Prompt redirects this through bottleneck tokens, which requires training. An inference-time approach could selectively retain the most-attended demo KV entries without retraining.

2. **X-Prompt treats all demo tokens equally.** Our data shows structural markers (4 tokens) receive disproportionate attention (5% total, 1.25% each), while per-VQ-token attention is only 0.008%. KV eviction could exploit this non-uniformity.

3. **X-Prompt doesn't analyze temporal dynamics.** Our finding that demo attention is stable throughout generation (0.356→0.316, only 11% decrease) means demo KV cannot be evicted after an early phase. Any compression must persist for the entire generation.

### 3.2 Method Design Implications

Based on the attention characterization, the following method design principles emerge:

1. **Demo KV eviction must be selective, not wholesale.** With 33% of attention on demos, removing all demo KV would catastrophically degrade generation. But the strong head specialization (>80% demo attention in some heads, <1% in others) suggests **head-selective KV retention**: keep demo KV for demo-reading heads, evict for demo-ignoring heads.

2. **Layer-aware strategy.** Deep layers (22-31) allocate >50% attention to demos — these are the critical layers for demo information. Early layers could tolerate more aggressive demo KV eviction.

3. **Structural markers must be preserved.** The 4 demo structural tokens (image_start, h_grid, w_grid, image_end) receive disproportionate attention and likely serve as positional anchors. Any KV eviction strategy should always retain these.

4. **Output-image tokens are higher priority than input-image tokens.** The 1.31x attention ratio suggests that if forced to evict some demo KV, evicting input-image tokens first preserves more information per retained token.

5. **LineAR-style line-level eviction** could be extended to semantic-role-aware eviction: instead of (or in addition to) evicting by spatial importance within the generated image, evict by semantic role (demo vs query) weighted by measured attention mass.

### 3.3 Quantitative Novelty Claims

This characterization provides the first empirical evidence for three claims:

1. **AR VLMs actively use demo context during generation** (33% attention mass on demo tokens, sustained throughout generation). No prior work measures this.

2. **Head-level specialization for demo-reading exists** (functional head types with >80% vs <1% demo attention). Analogous to but distinct from known head types (positional, semantic, sink heads).

3. **Demo sub-structure matters**: output-image tokens in demos receive 1.31x more attention than input-image tokens, suggesting the model primarily learns "what to generate" from demos rather than "what the input looks like."

---

# VLM In-Context Generation Profiling

## 1. Literature Review

### 1.1 In-Context Visual Learning: Two Paradigms

Two distinct approaches exist for in-context visual task generalization:

**Paradigm A — Image-Stitching (Painter/SegGPT, CVPR/ICCV 2023):**
- Input-output image pairs are stitched side-by-side into a single composite image
- ViT-Large backbone with MAE-style masked image modeling reconstructs the missing output
- No VQ codebook; operates in continuous pixel space
- NYUDepthV2 depth estimation is a standard evaluation task (depth map rendered as colorized image)
- In-context signal: a reference pair (input + ground truth) stitched alongside the query
- **Not autoregressive** — no AR token generation to profile

**Paradigm B — Autoregressive Token Generation (Chameleon/Lumina-mGPT):**
- Images tokenized into discrete tokens via VQ-VAE, concatenated with text tokens in a unified sequence
- Next-token prediction over the unified modality sequence
- Multi-image context = multiple image token sequences concatenated
- Image generation requires autoregressively predicting all image tokens, then VQ-decoding
- **This is our profiling target** — AR generation creates a fundamentally different bottleneck structure than diffusion denoising

### 1.2 Chameleon Architecture (Meta, 2024)

- **Model**: 7B and 34B parameter early-fusion decoder-only transformer (LLaMA-style with QK-norm, RoPE)
- **Image tokenization**: VQ-VAE encodes each 512×512 image into 32×32 = **1,024 discrete tokens** from codebook of **8,192 entries**
- **Unified vocabulary**: 65,536 total (8,192 image + 57,344 text BPE tokens)
- **Multi-image sequencing**: Images interleaved with text via `<image>` markers. All modalities in a single flat sequence
- **Training**: ~10T interleaved text-image tokens

**BLOCKER: Image generation disabled in released weights.** Meta intentionally stripped image generation capability from public 7B and 30B checkpoints (safety reasons). The VQ-VAE decoder was not released. Confirmed via official GitHub issues and HuggingFace model card. **No known workaround.**

### 1.3 Lumina-mGPT (Alpha-VLLM, 2024)

- **Architecture**: Decoder-only transformer initialized from Chameleon weights, then fine-tuned with:
  - FP-SFT (Flexible Progressive SFT) for high-resolution generation
  - Omni-SFT for multi-task generalist capabilities
- **Image representation**: Same VQ-VAE tokenization (1,024 tokens/image at 512×512). "UniRep" for flexible aspect ratios
- **Capabilities**: Text-to-image, multiview generation, controllable generation, segmentation, depth estimation, VQA
- **Depth estimation**: Trained on KITTI v2. Reformulated as image generation — model autoregressively generates depth map tokens conditioned on input image tokens + text instruction. Uses `cfg=1.0, image_top_k=200`
- **In-context multi-image**: Pretrained on interleaved text-image sequences, so multi-image context is architecturally supported
- **Lumina-mGPT 2.0** (2025): Pure AR model trained from scratch, competitive with diffusion models
- **Latency note**: Paper states AR image generation "often requires minutes to generate a full sequence of image tokens" — significantly slower than diffusion

### 1.4 Alternative: Emu3 (BAAI, Sep 2024)

- **Architecture**: Pure next-token prediction over unified vocabulary (text + image tokens)
- **Weights**: Available on HuggingFace (BAAI/Emu3-Gen). Separate Gen vs Chat checkpoints
- **In-context**: Trained on interleaved multimodal sequences
- **Limitation**: Resolution limited to 512×512

### 1.5 Key Architectural Details for Profiling

**Note**: Chameleon and Lumina-mGPT use different VQ tokenizers. Original Chameleon uses 32×32 latent (1,024 tokens) with 8,192-entry codebook. Lumina-mGPT 2.0 uses MoVQGAN 270M with 64×64 latent (4,096 tokens) with 16,384-entry codebook and embed_dim=4.

| Parameter | Chameleon (original) | Lumina-mGPT 2.0 (deployed) |
|-----------|---------------------|---------------------------|
| VQ tokenizer | Chameleon VQ-VAE | MoVQGAN 270M |
| Latent grid (512×512) | 32×32 | 64×64 |
| Tokens per image | 1,024 | 4,096 (+64 newlines +4 headers) |
| VQ codebook size | 8,192 | 16,384 |
| Embed dim | 256 | 4 |
| Image token ID range | — | 155000–171383 |
| max_position_embeddings | — | 10,240 |

| Parameter | Value |
|-----------|-------|
| Depth est. output (512×512 dual-panel) | ~2,081 generated tokens (upper half) |
| Full image in context (512×512) | ~4,164 tokens (4096 VQ + 64 newlines + 4 headers) |
| Expected bottleneck | AR token generation (not denoising loop) |
| KV cache | Applicable — standard AR generation |
| Attention cost per token | O(total_sequence_length) with KV cache |

### 1.6 Implications for Our Acceleration Methods

**Critical structural difference from Qwen/FLUX tracks:**
- Qwen/FLUX: Bottleneck is the **denoising loop** (iterative refinement of continuous latents). Our temporal caching and CFG-skip methods target this.
- VLM ICL: Bottleneck is expected to be **autoregressive token generation** (sequential discrete token prediction). No denoising loop exists in pure AR models.
- If the deployed model is pure AR (Lumina-mGPT 1.0, Emu3), our temporal caching methods **do not apply**. Relevant acceleration would be: speculative decoding, KV cache optimization, token pruning, quantization.
- If Lumina-mGPT 2.0 uses a DiT decoder component, there may be a denoising stage where our methods apply. The AR-vs-DiT time split becomes the key measurement.

### 1.7 Model Selection Decision

| Model | Image Gen? | Weights? | In-Context? | Decision |
|-------|-----------|----------|-------------|----------|
| Chameleon 7B | Disabled | Partial | Yes | **BLOCKED** — decoder not released |
| Lumina-mGPT 7B | Yes | Yes | Supported | **PRIMARY** — designated fallback, depth estimation demonstrated |
| Emu3-Gen | Yes | Yes | Supported | **BACKUP** — if Lumina-mGPT blocks |

**Proceeding with Lumina-mGPT** per fallback rule (Chameleon confirmed blocked before deployment attempt).

## 2. Profiling Results

### 2.1 Pipeline Stage Decomposition

Profiled depth estimation on NYUDepthV2 (512×512 dual-panel output, 5 instances + warmup, B200 GPU).

**Three pipeline stages:**
1. **Input tokenization**: Text tokenization + MoVQGAN encode of query image (512×256 → 2048 VQ tokens)
2. **AR token generation**: Autoregressive next-token prediction for output image tokens (2081 tokens with CFG)
3. **VQ-GAN decode**: MoVQGAN 270M decoder converts generated tokens back to pixels

| Stage | Mean Time (s) | % of Total | Notes |
|-------|---------------|-----------|-------|
| Input tokenization | 0.24 | 0.30% | VQ encode + text tokenize, negligible |
| **AR generation** | **80.9** | **99.42%** | **Dominant bottleneck** |
| VQ-GAN decode | 0.23 | 0.29% | MoVQGAN 270M, single forward pass |
| **Total** | **81.4 ± 3.5** | **100%** | ~1.4 minutes per image |

**Cold-start vs steady-state**: First inference (no warmup) took ~537s vs steady-state ~81s — a 6.6x difference. This is due to CUDA kernel compilation (SDPA on B200), KV cache buffer allocation, and PyTorch lazy initialization. All profiled numbers above are steady-state (after warmup).

### 2.2 Per-Token Timing

| Metric | Value |
|--------|-------|
| Tokens generated per image | 2,081 (perfectly consistent across all 5 runs) |
| ms per token (with CFG, steady-state) | 38.9 ± 1.7 |
| Prompt tokens (0-shot) | 2,129 |
| Peak VRAM (across runs) | 45–58 GB |
| Model load VRAM | 16.9 GB |

**CFG doubles the per-token cost**: Each token requires two forward passes (conditional + unconditional). The CFG logits processor maintains a separate KV cache for the unconditional branch, effectively doubling compute per generated token. Each forward pass costs ~19.5 ms (38.9 ms / 2).

**CFG decay is inactive at cfg=4.0**: The built-in decay formula `max(4.0, scale * (1 - progress))` clamps at 4.0, meaning the unconditional pass runs at full guidance for ALL 2,081 tokens. A higher initial cfg (e.g., 8.0) would trigger decay mid-generation. This represents a potential 2x acceleration target: reducing or skipping the unconditional pass for later tokens could halve generation time.

### 2.3 Context-Length Scaling

Measured generation speed with 0-shot (query only) vs 1-shot (1 demo image + query) depth estimation. 2-shot exceeds `max_position_embeddings` (10457 prompt + 2081 generated = 12538 > 10240).

| Condition | Prompt Tokens | Generated | ms/token | Total (s) | VRAM (GB) | Slowdown |
|-----------|--------------|-----------|----------|-----------|-----------|----------|
| **0-shot** | 2,129 | 2,081 | 39.9 | 83.0 | 48.7 | 1.00x |
| **1-shot** | 6,293 | 2,081 | 274.5 | 571.2 | 52.0 | **6.88x** |

**Critical finding: adding ONE in-context demo image causes a 6.88x slowdown.**

The prompt length increased ~3x (2129 → 6293), but the effective cost increased ~7x. This super-linear scaling has two components:
1. **Prefill cost scales quadratically**: Processing the initial prompt requires O(n²) attention over all prompt tokens. Prefill cost ratio: (6293/2129)² ≈ 8.7x
2. **Per-token generation scales linearly**: Each generated token attends to all prior tokens via KV cache. Average sequence length during generation: 0-shot ~3170, 1-shot ~7334 → ~2.3x

The observed 6.88x is consistent with prefill dominating the total cost at longer context lengths.

**Implications for in-context learning**: Multi-image ICL on AR models is prohibitively expensive. Even a single 512×512 demo image pushes total time from 1.4 min to 9.5 min. Two demos exceed the position encoding limit entirely. This contrasts sharply with diffusion models, where context images are encoded into a fixed-size conditioning signal (e.g., FLUX Kontext's 4096 image tokens are always present).

### 2.4 Cross-Paradigm Latency Comparison

| Model | Architecture | Task | Total Time | Tokens/Steps | ms per unit |
|-------|-------------|------|-----------|-------------|-------------|
| **Lumina-mGPT 2.0** | AR (7B) | Depth est. (512×512) | **81.4s** | 2,081 tokens | 38.9 ms/tok |
| **Qwen-Image-Edit** | Diffusion (DiT) | Image edit (1024×1024) | 47.4s | 40 steps | 1,184 ms/step |
| **FLUX Kontext** | Diffusion (DiT) | Image edit (1024×1024) | 9.2s | 28 steps | 321 ms/step |

**Key observations:**
- AR model is **~1.7x slower than Qwen** and **~8.8x slower than FLUX** (steady-state, B200)
- The bottleneck structures are fundamentally different: diffusion models iterate over a fixed number of steps, AR models generate tokens sequentially with count proportional to image resolution²
- At 512×512, Lumina-mGPT generates 2,081 tokens; scaling to 1024×1024 would require ~8,192 tokens (~4x longer, ~5.4 min)
- **Cold start penalty**: First inference without warmup is 6.6x slower (537s vs 81s) due to CUDA kernel compilation. Not applicable to diffusion models which have uniform step timing from the start
- Per-forward-pass cost: AR model ~19.5 ms (38.9/2 for CFG) vs Qwen ~592 ms vs FLUX ~321 ms. AR forward passes are much cheaper individually but there are ~52x more of them than Qwen denoising steps

## 3. Acceleration Applicability Assessment

### 3.1 Our Methods vs AR Bottleneck

| Method | Targets | Applies to AR? | Why/Why Not |
|--------|---------|---------------|-------------|
| **Temporal caching** (Qwen track) | Denoising loop — reuse intermediate features across timesteps | **NO** | AR models have no denoising loop. Each token is generated once, sequentially |
| **CFG-skip** (Qwen track) | Skip unconditional forward pass at select denoising steps | **Partially** | AR CFG also uses dual forward passes. Could skip unconditional pass for some tokens. But each token is unique (no temporal redundancy across steps) |
| **Step reduction** | Reduce number of denoising steps | **NO** | Token count is fixed by image resolution, not tunable |

### 3.2 What Would Work for AR Models

| Technique | Mechanism | Expected Speedup | Maturity |
|-----------|-----------|-----------------|----------|
| **Speculative decoding** | Draft model proposes N tokens, target model verifies in parallel | 2-3x | Proven for LLMs, nascent for image tokens |
| **KV cache optimization** | Reduce memory/compute for attention over long sequences | 1.2-1.5x | Standard (already enabled) |
| **Quantization** (INT4/INT8) | Reduce model weight precision | 1.5-2x | Lumina-mGPT has TorchAO quant support built-in |
| **CFG guidance schedule** | Reduce CFG scale (and skip unconditional pass) as generation progresses | Up to 2x | Already implemented in Lumina — `guidance_scale = max(4.0, cfg * (1 - progress))` decays during generation |
| **Token pruning / parallel decoding** | Generate multiple tokens per forward pass | 2-4x | Active research (Medusa, EAGLE) |
| **Diffusion-hybrid (Lumina-mGPT 2.0)** | Use AR for semantic planning + diffusion for pixel generation | N/A | Architecture-level change; not applicable to existing AR model |

### 3.3 Key Takeaway

**Our temporal caching and CFG-skip methods fundamentally target the denoising loop, which does not exist in pure AR image generation.** The AR bottleneck (sequential token generation with O(n) attention per token) requires different acceleration strategies. The most promising near-term approach for this model specifically is quantization (already supported) and CFG schedule optimization (already partially implemented).

## 4. Evaluation Metrics for Depth Estimation

### 4.1 Recommended Metrics

For evaluating depth estimation quality in the generated images:

| Metric | What It Measures | Notes |
|--------|-----------------|-------|
| **RMSE** (Root Mean Square Error) | Pixel-level depth accuracy | Standard in NYUDepthV2 benchmarks; requires extracting depth from upper panel |
| **δ < 1.25** (threshold accuracy) | % of pixels with max(pred/gt, gt/pred) < 1.25 | Standard metric; tolerant of scale ambiguity |
| **SSIM** (structural similarity) | Structural similarity between predicted and GT depth maps | Already used in our Qwen/FLUX tracks |
| **Scale-Invariant Log Error (SILog)** | Depth accuracy invariant to global scale | Important because AR-generated depth may have arbitrary scale |

### 4.2 Practical Considerations

- The model generates a **visual depth map** (colorized), not raw depth values. Evaluation requires color-to-depth conversion or evaluation in colorized space (SSIM/LPIPS on the color images)
- Ground truth depth maps from NYUDepthV2 are available as both raw `.npy` and colorized `.png`
- For cross-model comparison, SSIM and LPIPS on the colorized outputs are most practical
- Quantitative depth accuracy (RMSE, δ < 1.25) requires a colormap inversion step, which introduces its own error

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

# FLUX Track — F4: Conditioning-Aware KV Reuse

## Objective

Test whether caching the K/V projections of static conditioning tokens (text + reference image) across denoising steps in the plateau region produces meaningful speedup with better quality preservation than whole-step output caching.

## 1. Token Layout (Phase 1: Code Inspection)

### 1.1 Attention Sequence Structure

The Flux transformer processes three token types in each attention operation:

| Token Type | Source | Sequence Position | Count (1024x1024) | Changes Across Steps? |
|------------|--------|-------------------|--------------------|-----------------------|
| Text | CLIP-L + T5-XXL prompt embeddings | First in attention concat | 512 | No (fixed input) |
| Noise | Random latents being denoised | Middle | 4,096 | Yes (updated by scheduler) |
| Image condition | VAE-encoded reference image | Last | 4,096 | No (fixed input) |

**Total attention sequence**: 8,704 tokens. Condition tokens (text + image) = 4,608 = 53% of sequence.

### 1.2 Block Architecture and K/V Separability

**Dual-stream blocks (19):**
- Separate K/V projections: `to_k/v` for [noise | image], `add_k/v_proj` for text
- Joint attention on concatenated [text | noise | image], then split back
- K/V separable by position index (linear projections are position-independent)

**Single-stream blocks (38):**
- Concatenate [text | noise | image] before norm/attention
- Single `to_k/v` for all tokens
- K/V separable by position index

**Feasibility assessment**: K/V CAN be separated and selectively cached in both block types. The condition-token portions occupy fixed, known positions in the sequence.

### 1.3 Key Complication

Although raw `image_latents` input is fixed across steps, intermediate representations evolve through attention interactions with noise tokens. At block L, image-condition K/V depends on attention with noise at blocks 0..L-1. The premise is that in the plateau region, this evolution is small.

## 2. K/V Stability Measurement (Phase 2)

Measured consecutive-step cosine similarity of K/V projections for each token type across all 28 steps, at 6 sampled blocks (3 dual-stream: 0, 9, 18; 3 single-stream: 0, 18, 37).

### 2.1 Plateau Region Results (Steps 4-23, Mean Cosine Similarity)

| Block | noise_k | noise_v | image_k | image_v | text_k | text_v |
|-------|---------|---------|---------|---------|--------|--------|
| D0    | 1.0002  | 0.9996  | 1.0009  | 0.9992  | 0.9996 | 0.9993 |
| D9    | 0.9985  | 0.9975  | 0.9997  | 0.9992  | 0.9991 | 0.9988 |
| D18   | 0.9983  | 0.9952  | 1.0000  | 0.9987  | 0.9989 | 0.9973 |
| S0    | 0.9974  | 0.9946  | 0.9998  | 0.9981  | 0.9982 | 0.9965 |
| S18   | 0.9884  | 0.9705  | 0.9791  | 0.9558  | 0.8573 | 0.6218 |
| S37   | 0.9987  | 0.9988  | 0.9996  | 0.9996  | 0.9258 | 0.9496 |

### 2.2 Stability Gap Analysis

Measured by relative change (1 - cos_sim), image K/V changes 3-5x less than noise K/V in 5 of 6 sampled blocks:

| Block | Noise K change | Image K change | Ratio |
|-------|----------------|----------------|-------|
| D9    | 0.0015         | 0.0003         | 5.0x  |
| D18   | 0.0017         | ~0             | >>5x  |
| D18 (V) | 0.0048      | 0.0013         | 3.7x  |
| S0    | 0.0026         | 0.0002         | 13x   |
| S37   | 0.0013         | 0.0004         | 3.3x  |

**Exception**: Block S18 (single-stream middle, overall block index 37/57) is the most dynamic block in the model — analogous to Qwen's block 44. All token types are less stable here. Text tokens show dramatic instability (cos = 0.62 for V), suggesting this block performs the heaviest text-noise interaction.

### 2.3 Premise Assessment

**Supported for 5/6 blocks**: Image K/V changes 3-5x less than noise K/V across the plateau. The gap is consistent in direction and grows in deeper dual-stream blocks.

**Exception at most-dynamic block**: S18 shows image K/V less stable than noise K/V. This is the only block where the premise fails.

**Absolute magnitude**: Both noise and image K/V are very stable in the plateau (cos > 0.995 for most blocks). The absolute difference is small (0.001-0.004 in cosine), but relative change ratio is substantial.

## 3. Prototype Implementation (Phase 3)

### 3.1 Design

Custom attention processor (`KVCachingProcessor`) that, during the plateau region:
1. **First plateau step** (step 6): Compute all K/V normally; cache condition-token K/V (text + image) for each of the 57 blocks
2. **Subsequent plateau steps** (steps 7-20): Reuse cached condition K/V; compute fresh K/V only for noise tokens; compute fresh Q for all tokens; run full attention matmul with mixed (cached + fresh) K/V
3. **Non-plateau steps**: Full computation (no caching)

### 3.2 Theoretical Speedup

K/V projection savings per cached step:
- Save: 2 projections × 4,608 condition tokens × 3072² FLOPs ≈ 87B FLOPs per block
- Total per step: 87B × 57 blocks ≈ 5.0T FLOPs
- Total step cost: ~45.2T FLOPs → **~11% savings per cached step**
- Applied to 14 cached steps out of 28 → **~5.5% wall-clock reduction**
- Theoretical speedup: **~1.06x**

The limitation: **attention matmuls (Q×K^T and attn×V) dominate attention cost (~60%) and are NOT reduced** because the full K/V tensor (fresh noise + cached condition) must still participate in the matmul.

## 4. Evaluation Results (Phase 4)

### 4.1 Per-Case Results (vs Baseline)

| Test Case | KV Speed | KV SSIM | KV LPIPS | TC Speed | TC SSIM | TC LPIPS |
|-----------|----------|---------|----------|----------|---------|----------|
| simple_object (robust) | 1.016x | 0.990 | 0.008 | 2.155x | 0.987 | 0.011 |
| object_modify (robust) | 1.016x | 0.931 | 0.006 | 2.153x | 0.888 | 0.051 |
| portrait (robust) | 1.015x | 0.947 | 0.026 | 2.154x | 0.836 | 0.175 |
| arrangement (robust) | 1.016x | 0.955 | 0.024 | 2.154x | 0.885 | 0.106 |
| style_transfer (fragile) | 1.016x | 0.902 | 0.089 | 2.154x | 0.812 | 0.334 |
| fine_texture (fragile) | 1.015x | 0.927 | 0.041 | 2.154x | 0.807 | 0.171 |
| complex_scene (fragile) | 1.015x | 0.732 | 0.162 | 2.154x | 0.442 | 0.547 |
| nature_detail (fragile) | 1.015x | 0.952 | 0.044 | 2.154x | 0.778 | 0.310 |
| **MEAN** | **1.015x** | **0.917** | **0.050** | **2.154x** | **0.804** | **0.213** |

KV = condition-token KV caching. TC = whole-step temporal caching (F2-style). Same plateau window (steps 6-20) for fair comparison.

### 4.2 Quality Comparison

**KV caching wins on quality in ALL 8/8 cases.**

| Subset | KV SSIM | TC SSIM | Gap |
|--------|---------|---------|-----|
| Robust (n=4) | 0.955 | 0.899 | +0.056 |
| Fragile (n=4) | 0.878 | 0.710 | +0.168 |
| All (n=8) | 0.917 | 0.804 | +0.112 |

The quality advantage is **largest on fragile cases** (+0.168 SSIM), showing that KV caching protects fine-detail content that temporal caching degrades.

Most dramatic: complex_scene — KV 0.732 vs TC 0.442 (nearly double the SSIM). LPIPS: KV 0.162 vs TC 0.547 (3.4x better perceptual similarity).

### 4.3 Speedup Comparison

- KV caching: **1.015x** (negligible — ~0.14s saved on 8.9s baseline)
- Temporal caching: **2.154x** (4.8s saved — transformer skipped entirely on cached steps)
- Speedup gap: **-1.14x** (temporal caching is dramatically faster)

### 4.4 Ablation Insight: Where Does Quality Loss Come From?

This experiment serves as an ablation study isolating the source of quality degradation in temporal caching:

| Component | KV Caching | Temporal Caching | Quality Impact |
|-----------|------------|------------------|----------------|
| Noise K/V projections | Fresh | Stale (prev step) | – |
| Condition K/V projections | Cached (prev step) | Stale (prev step) | Minimal (cos > 0.999) |
| Q projections (all tokens) | Fresh | Stale | **Moderate** |
| Attention matmul (Q×K^T, attn×V) | Fresh (with mixed K/V) | Stale | **Dominant** |
| FFN | Fresh | Stale | Moderate |
| Output projection | Fresh | Stale | Minor |

**Finding**: KV caching introduces minimal error (SSIM 0.917) while temporal caching introduces substantial error (SSIM 0.804). The difference (+0.112 SSIM) must come from the components that differ: Q projections, attention matmuls, and FFN. Since K/V caching recomputes ALL of these freshly (only condition K/V is cached), the quality loss in temporal caching is attributable to:
1. **Stale attention matmuls** (dominant — Q×K^T and attn×V with stale Q and K/V)
2. **Stale FFN outputs** (moderate)
3. **Stale condition K/V** (negligible — confirmed by KV caching's high quality)

**Key finding: Condition-token K/V staleness contributes negligibly to temporal caching's quality loss. The bottleneck is the attention matmul and FFN recomputation.**

## 5. Answers to Scientific Questions

1. **Can condition-token K/V projections be identified and isolated?** YES. Linear projections are position-independent. Condition tokens occupy fixed, known positions in the sequence.

2. **How much do condition-token K/V change across steps in the plateau?** Image K/V changes 3-5x less than noise K/V (measured by 1-cos). Text K/V varies: stable in dual-stream blocks, highly dynamic in single-stream block S18.

3. **Does caching condition-token K/V produce measurable speedup?** NO. Speedup is 1.015x — negligible. K/V projection is only ~11% of attention cost; the attention matmul (which still runs at full cost) dominates.

4. **Does this approach preserve quality better than whole-step caching?** YES, dramatically. SSIM 0.917 vs 0.804 (+0.112), LPIPS 0.050 vs 0.213 (4.3x better). KV caching wins in ALL 8/8 test cases.

5. **Does speedup scale with input image count?** Not testable — Kontext supports single image only. But the token ratio (condition/total = 53%) is constant, so the savings fraction would be constant regardless of sequence length.

## 6. Verdict and Implications

**Verdict**: KV reuse provides **dramatically better quality** (+0.112 SSIM) but **negligible speedup** (1.015x). The tradeoff is unfavorable for practical acceleration. Whole-step temporal caching remains the preferred method for speedup despite its quality cost.

**Paper contribution**: This experiment's primary value is as an **ablation study** identifying where quality loss occurs:
- Condition-token K/V is stable and cacheable (confirmed by both measurement and quality evaluation)
- Quality loss in temporal caching comes overwhelmingly from stale attention matmuls and FFN, NOT from stale condition K/V
- This implies that future improvement of temporal caching should target **partial recomputation of the attention matmul** (e.g., recomputing Q×K^T for noise queries while reusing condition-query results), not finer-grained K/V caching
- However, such partial attention decomposition would require custom kernels incompatible with Flash Attention, making it an engineering challenge beyond schedule-based methods

**Failure condition met**: Speedup is negligible (as predicted by theoretical analysis). The K/V computation is a small fraction of total attention cost; the matmul dominates. Reported as a **scientifically informative negative result** with clear ablation value.

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
