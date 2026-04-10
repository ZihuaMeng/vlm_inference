"""
AR-UND: Understanding Task Generalization of Prefix Caching

Tests whether the AR-4 prefix caching mechanism (Dynamic NTK + demo KV
caching) generalizes to multi-image understanding tasks (text-only output).

Scientific questions:
  1. Does prefix caching produce wall-clock speedup for understanding tasks?
  2. Does the generation pathology (DynamicNTK slowdown at long sequences)
     also affect text output, or is it specific to long image generation?
  3. Is understanding-task text quality preserved under prefix caching?
  4. How does the prefill-to-generation cost ratio differ for text vs image?
     (Text output is ~50-200 tokens vs ~2081 for images.)
"""

import sys
import os
import json
import time
import numpy as np
import torch
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

# ═══════════════════════════════════════════════
# Path Setup (critical: CWD for MoVQGAN)
# ═══════════════════════════════════════════════
REPO_ROOT = "/home/zihua.meng/projects/vlm_inference/vlm_icl/Lumina-mGPT-2.0"
sys.path.insert(0, os.path.join(REPO_ROOT, "lumina_mgpt"))
sys.path.insert(0, REPO_ROOT)
os.chdir(os.path.join(REPO_ROOT, "lumina_mgpt"))

from data.item_processor import FlexARItemProcessor
from transformers import DynamicCache, GenerationConfig

# ═══════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════
MODEL_PATH = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/weights/Lumina-mGPT-2.0-Omni"
DATA_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/data/nyu_depth_v2"
OUTPUT_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/outputs/ar_und"
CACHE_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/cache"

CFG_SCALE = 4.0
IMAGE_TOP_K = 200
IMAGE_END_TOKEN_ID = 151666
IMAGE_START_TOKEN_ID = 151665
SEED = 42

MAX_GEN_TOKENS = 200   # text output: ~50-200 tokens (vs 2500 for image)
MAX_TOTAL_LENGTH = 20000

# Understanding task configuration
QUESTION = "Describe what you see in this image in one sentence."
DEMO_LABELS = [
    "A room with furniture and natural lighting from a window.",
    "An indoor space with shelving and various objects on display.",
    "A room with a bed and decorative items on the walls.",
]

# AR-4 results for cross-task comparison
AR4_RESULTS = {
    1: {"fresh_prefill": 0.227, "fresh_gen": 95.3, "fresh_total": 95.5,
        "cached_gen": 90.3, "cached_total": 91.4, "speedup": 1.05,
        "n_gen_tokens": 2081, "prompt_len": 6293},
    2: {"fresh_prefill": 1.256, "fresh_gen": 543.7, "fresh_total": 544.9,
        "cached_gen": 97.3, "cached_total": 101.6, "speedup": 5.36,
        "n_gen_tokens": 2081, "prompt_len": 10457},
    3: {"fresh_prefill": 1.431, "fresh_gen": 560.8, "fresh_total": 562.2,
        "cached_gen": 100.6, "cached_total": 108.5, "speedup": 5.18,
        "n_gen_tokens": 2081, "prompt_len": 14621},
}


# ═══════════════════════════════════════════════════════════
# Position Encoding Extension (same as AR-4)
# ═══════════════════════════════════════════════════════════

def enable_dynamic_ntk_scaling(model, scaling_factor=2.0):
    """Replace vanilla RoPE with Dynamic NTK scaling for extended sequences."""
    from transformers.models.chameleon.modeling_chameleon import (
        ChameleonDynamicNTKScalingRotaryEmbedding,
    )
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    replaced = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.rotary_emb = ChameleonDynamicNTKScalingRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=scaling_factor,
        ).to(model.device)
        replaced += 1
    print(f"  Enabled Dynamic NTK scaling (factor={scaling_factor}) on {replaced} layers")
    return replaced


# ═══════════════════════════════════════════════════════════
# KV Cache Utilities (same as AR-4)
# ═══════════════════════════════════════════════════════════

def clone_kv_cache(kv_cache):
    new_cache = DynamicCache()
    for layer_idx in range(len(kv_cache.key_cache)):
        new_cache.update(
            kv_cache.key_cache[layer_idx].clone(),
            kv_cache.value_cache[layer_idx].clone(),
            layer_idx,
        )
    return new_cache


def save_kv_cache(kv_cache, path):
    data = {
        "key_cache": [k.cpu() for k in kv_cache.key_cache],
        "value_cache": [v.cpu() for v in kv_cache.value_cache],
    }
    torch.save(data, path)


def load_kv_cache(path, device):
    data = torch.load(path, map_location=device, weights_only=True)
    cache = DynamicCache()
    for layer_idx in range(len(data["key_cache"])):
        cache.update(
            data["key_cache"][layer_idx].to(device),
            data["value_cache"][layer_idx].to(device),
            layer_idx,
        )
    return cache


# ═══════════════════════════════════════════════════════════
# Prompt Building for Understanding Task
# ═══════════════════════════════════════════════════════════

def build_understanding_prompt(solver, query_path, demo_data, n_demos):
    """Build an N-shot understanding prompt with text-only output.

    Structure per demo: <image> question <sep> answer <sep>
    Query:              <image> question <sep>  → model generates text

    Returns:
        prompt_tensor: (1, seq_len) token IDs
        prompt_len: total number of tokens
        demo_len: tokens in the demo portion (for caching split)
        query_ids: list of query token IDs
        demo_token_ids: list of demo token IDs
    """
    # Set transform for input images (not i2i)
    solver.item_processor.transform = {"<|image|>": solver.item_processor.process_image}

    # Build full conversation (all demos + query)
    conversations = []
    images = []
    for i in range(n_demos):
        conversations.append({"from": "human", "value": f"<|image|> {QUESTION}"})
        conversations.append({"from": "gpt", "value": demo_data[i]["label"]})
        images.append(demo_data[i]["img_path"])

    # Query turn (answer=None → model generates)
    conversations.append({"from": "human", "value": f"<|image|> {QUESTION}"})
    conversations.append({"from": "gpt", "value": None})
    images.append(query_path)

    full_item = {"image": images, "conversations": conversations}
    full_tokens = solver.item_processor.process_item(full_item)

    # Build demo-only prompt to find the split point
    if n_demos > 0:
        demo_convs = conversations[:-2]  # exclude query turn
        demo_item = {"image": images[:-1], "conversations": demo_convs}
        demo_token_ids = solver.item_processor.process_item(demo_item)
        demo_len = len(demo_token_ids)

        # Verify split correctness
        query_ids = full_tokens[demo_len:]
        reconstructed = demo_token_ids + query_ids
        assert reconstructed == full_tokens, (
            f"Prompt split mismatch: demo_len={demo_len}, "
            f"query_len={len(query_ids)}, total={len(full_tokens)}, "
            f"reconstructed={len(reconstructed)}"
        )
    else:
        demo_len = 0
        demo_token_ids = []
        query_ids = full_tokens

    prompt_tensor = torch.tensor(
        full_tokens, dtype=torch.int64, device=solver.model.device
    ).unsqueeze(0)

    return prompt_tensor, len(full_tokens), demo_len, query_ids, demo_token_ids


# ═══════════════════════════════════════════════════════════
# Prefill & Generation
# ═══════════════════════════════════════════════════════════

def fresh_prefill(solver, prompt_tensor):
    """Full prefill on entire prompt. Returns KV cache and last logits."""
    dtype = torch.bfloat16
    cache = DynamicCache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        out = solver.model(prompt_tensor, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    return out.past_key_values, out.logits[:, -1:, :].clone(), t


def demo_only_prefill(solver, demo_token_ids):
    """Prefill ONLY the demo tokens. Returns demo KV cache."""
    dtype = torch.bfloat16
    demo_tensor = torch.tensor(
        demo_token_ids, dtype=torch.int64, device=solver.model.device
    ).unsqueeze(0)
    cache = DynamicCache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        out = solver.model(demo_tensor, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    return out.past_key_values, t


def query_prefill_from_cache(solver, query_ids, demo_kv):
    """Prefill query tokens using a pre-computed demo KV cache."""
    dtype = torch.bfloat16
    query_tensor = torch.tensor(
        query_ids, dtype=torch.int64, device=solver.model.device
    ).unsqueeze(0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        out = solver.model(query_tensor, past_key_values=demo_kv, use_cache=True)
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    return out.past_key_values, out.logits[:, -1:, :].clone(), t


def generate_text_from_cache(solver, full_prompt_tensor, prompt_len,
                             kv_cache, prefill_logits, sep_id):
    """Generate text output from prefill KV cache.

    Adapted from AR-4's generate_from_cache. Uses the same logits processor
    (MultiModalLogitsProcessor is no-op for text, InterleavedTopKLogitsWarper
    applies text_top_k=10, CFG is no-op for text).
    """
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    gen_config = GenerationConfig(
        max_new_tokens=MAX_GEN_TOKENS,
        max_length=MAX_TOTAL_LENGTH,
        temperature=1.0, do_sample=True,
        eos_token_id=[sep_id, IMAGE_END_TOKEN_ID],
    )

    logits_proc = solver.create_logits_processor(
        cfg=CFG_SCALE, image_top_k=IMAGE_TOP_K
    )
    with torch.no_grad():
        processed = logits_proc(full_prompt_tensor, prefill_logits[:, -1, :])
        probs = torch.softmax(processed, dim=-1)
        first_token = torch.multinomial(probs, num_samples=1)
    del prefill_logits

    gen_ids = torch.cat([full_prompt_tensor, first_token], dim=1)
    logits_proc2 = solver.create_logits_processor(
        cfg=CFG_SCALE, image_top_k=IMAGE_TOP_K
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result = solver.model.generate(
            gen_ids, gen_config,
            logits_processor=logits_proc2,
            past_key_values=kv_cache,
        )[0][prompt_len:].tolist()
    torch.cuda.synchronize()
    return result, time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════
# Text Quality Metrics
# ═══════════════════════════════════════════════════════════

def decode_text(solver, tokens, sep_id):
    """Decode generated token IDs to text, stripping special tokens."""
    clean = [t for t in tokens
             if t != IMAGE_START_TOKEN_ID
             and t != IMAGE_END_TOKEN_ID
             and t != sep_id
             and t < 155000]
    return solver.item_processor.tokenizer.decode(clean)


def text_quality(baseline_text, test_text):
    """Compare two text outputs. Returns quality metrics."""
    b = baseline_text.strip()
    t = test_text.strip()
    exact_match = (b == t)

    # Character-level Levenshtein distance
    m, n = len(b), len(t)
    if m == 0 and n == 0:
        return {"exact_match": True, "edit_distance": 0, "char_similarity": 1.0,
                "word_match_rate": 1.0, "baseline_len": 0, "test_len": 0}
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (0 if b[i - 1] == t[j - 1] else 1),
            )
    edit_dist = dp[m][n]
    max_len = max(m, n, 1)
    char_sim = 1.0 - edit_dist / max_len

    # Word-level match
    b_words = b.split()
    t_words = t.split()
    n_words = min(len(b_words), len(t_words))
    if n_words > 0:
        word_match = sum(1 for i in range(n_words) if b_words[i] == t_words[i]) / n_words
    else:
        word_match = 0.0

    return {
        "exact_match": exact_match,
        "edit_distance": edit_dist,
        "char_similarity": float(char_sim),
        "word_match_rate": float(word_match),
        "baseline_len": m,
        "test_len": n,
    }


def token_match_rate(base, test):
    """Token-level comparison of two generated sequences."""
    n = min(len(base), len(test))
    if n == 0:
        return 0.0, 0.0, 0
    matches = sum(1 for i in range(n) if base[i] == test[i])
    first_diff = next((i for i in range(n) if base[i] != test[i]), n)
    n20 = min(20, n)
    early = sum(1 for i in range(n20) if base[i] == test[i]) / n20
    return matches / n, early, first_diff


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("AR-UND: Understanding Task — Prefix Caching Generalization")
    print("=" * 70)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(CACHE_ROOT, exist_ok=True)

    # Load task instances (same data as AR-4)
    with open(os.path.join(DATA_ROOT, "task_instances.json")) as f:
        instances = json.load(f)
    print(f"Task instances: {len(instances)}")

    # Prepare demo data
    inst = instances[0]
    demo_data = []
    for i, dp in enumerate(inst["demo_pairs"][:3]):
        demo_data.append({
            "img_path": dp["rgb_path"],
            "label": DEMO_LABELS[i],
        })
    print(f"Demo images: {[d['img_path'].split('/')[-2] for d in demo_data]}")
    print(f"Demo labels: {[d['label'][:50] for d in demo_data]}")

    # Load model
    print("\nLoading model...")
    from inference_solver import FlexARInferenceSolver
    solver = FlexARInferenceSolver(
        model_path=MODEL_PATH, precision="bf16", target_size=512
    )
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Config: max_position_embeddings={solver.model.config.max_position_embeddings}")

    # Get separator token ID
    sep_token = "<reserved08706>"
    sep_ids = solver.item_processor.tokenizer.encode(sep_token, bos=False, eos=False)
    sep_id = sep_ids[0] if len(sep_ids) == 1 else sep_ids[-1]
    print(f"Separator token: '{sep_token}' → ID {sep_id}")

    # Enable Dynamic NTK scaling (same as AR-4)
    print("\nEnabling Dynamic NTK scaling...")
    enable_dynamic_ntk_scaling(solver.model, scaling_factor=2.0)

    # ─── Phase 0: Text Generation Viability ────────────────

    print("\n" + "=" * 70)
    print("PHASE 0: Text Generation Viability Test (0-shot)")
    print("=" * 70)

    query_path_0 = instances[0]["query"]["rgb_path"]
    print(f"\nQuery image: {query_path_0}")

    # Warmup with a short generation
    print("Warmup (100 tokens)...")
    prompt_tensor_0, prompt_len_0, _, _, _ = build_understanding_prompt(
        solver, query_path_0, demo_data, 0
    )
    print(f"  0-shot prompt length: {prompt_len_0} tokens")

    gc_warmup = GenerationConfig(
        max_new_tokens=100, max_length=MAX_TOTAL_LENGTH,
        temperature=1.0, do_sample=True,
        eos_token_id=[sep_id, IMAGE_END_TOKEN_ID],
    )
    lp_warmup = solver.create_logits_processor(cfg=CFG_SCALE, image_top_k=IMAGE_TOP_K)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        warmup_result = solver.model.generate(
            prompt_tensor_0, gc_warmup, logits_processor=lp_warmup
        )[0][prompt_len_0:].tolist()
    warmup_text = decode_text(solver, warmup_result, sep_id)
    print(f"  Warmup output ({len(warmup_result)} tokens): '{warmup_text[:200]}'")

    # 0-shot full generation
    print("\n0-shot full generation...")
    kv_0, logits_0, prefill_t_0 = fresh_prefill(solver, prompt_tensor_0)
    tokens_0, gen_t_0 = generate_text_from_cache(
        solver, prompt_tensor_0, prompt_len_0, kv_0, logits_0, sep_id
    )
    text_0 = decode_text(solver, tokens_0, sep_id)
    print(f"  Prefill: {prefill_t_0:.3f}s, Generation: {gen_t_0:.3f}s")
    print(f"  Generated {len(tokens_0)} tokens: '{text_0[:300]}'")

    text_is_coherent = len(tokens_0) > 2 and len(text_0.strip()) > 5
    print(f"  Text coherent: {text_is_coherent}")

    phase0_results = {
        "prompt_len": prompt_len_0,
        "prefill_time": prefill_t_0,
        "gen_time": gen_t_0,
        "n_tokens": len(tokens_0),
        "text": text_0[:500],
        "coherent": text_is_coherent,
    }

    if not text_is_coherent:
        print("\n  WARNING: Model produced incoherent text output.")
        print("  Trying simpler question: 'What type of room is this?'")
        # Fallback — try classification
        global QUESTION
        QUESTION = "What type of room is this? Answer in one word."
        prompt_tensor_0b, prompt_len_0b, _, _, _ = build_understanding_prompt(
            solver, query_path_0, demo_data, 0
        )
        kv_0b, logits_0b, _ = fresh_prefill(solver, prompt_tensor_0b)
        tokens_0b, _ = generate_text_from_cache(
            solver, prompt_tensor_0b, prompt_len_0b, kv_0b, logits_0b, sep_id
        )
        text_0b = decode_text(solver, tokens_0b, sep_id)
        print(f"  Fallback output: '{text_0b[:200]}'")
        text_is_coherent = len(tokens_0b) > 0 and len(text_0b.strip()) > 0
        if text_is_coherent:
            DEMO_LABELS[0] = "bedroom"
            DEMO_LABELS[1] = "bathroom"
            DEMO_LABELS[2] = "office"
            for i, d in enumerate(demo_data):
                d["label"] = DEMO_LABELS[i]

    # ─── Phase 1: Multi-Demo Prompt Analysis ──────────────

    print("\n" + "=" * 70)
    print("PHASE 1: Multi-Demo Prompt Analysis")
    print("=" * 70)

    phase1_results = {}
    for n_demos in [1, 2, 3]:
        print(f"\n--- {n_demos}-demo prompt ---")
        prompt_tensor, prompt_len, demo_len, query_ids, demo_token_ids = \
            build_understanding_prompt(solver, query_path_0, demo_data, n_demos)
        seq_len = prompt_tensor.shape[1]
        gen_total = seq_len + MAX_GEN_TOKENS
        exceeds = seq_len > solver.model.config.max_position_embeddings
        print(f"  Prompt tokens: {seq_len} (demo={demo_len}, query={len(query_ids)})")
        print(f"  With generation (~{MAX_GEN_TOKENS} text tokens): ~{gen_total}")
        print(f"  Exceeds max_position_embeddings ({solver.model.config.max_position_embeddings}): "
              f"{'YES' if exceeds else 'no'}")
        phase1_results[n_demos] = {
            "prompt_len": seq_len, "demo_len": demo_len,
            "query_len": len(query_ids), "exceeds_max": exceeds,
            "est_total_with_gen": gen_total,
        }

    # ─── Phase 2: Latency Measurement ─────────────────────

    print("\n" + "=" * 70)
    print("PHASE 2: Multi-Demo Latency Measurement (Fresh vs Cached)")
    print("=" * 70)

    phase2_results = {}

    for n_demos in [1, 2, 3]:
        print(f"\n{'=' * 50}")
        print(f"{n_demos}-demo configuration")
        print(f"{'=' * 50}")

        demo_pairs_n = demo_data[:n_demos]

        prompt_tensor, prompt_len, demo_len, query_ids, demo_token_ids = \
            build_understanding_prompt(solver, query_path_0, demo_pairs_n, n_demos)
        seq_len = prompt_tensor.shape[1]
        print(f"  Prompt: {seq_len} tokens (demo={demo_len}, query={len(query_ids)})")

        # --- Fresh full pipeline ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"\n  [Fresh] Full prefill + text generation...")
        fresh_kv, fresh_logits, fresh_prefill_t = fresh_prefill(solver, prompt_tensor)
        fresh_peak_prefill = torch.cuda.max_memory_allocated() / 1e9
        fresh_tokens, fresh_gen_t = generate_text_from_cache(
            solver, prompt_tensor, prompt_len, fresh_kv, fresh_logits, sep_id
        )
        fresh_total = fresh_prefill_t + fresh_gen_t
        fresh_peak = torch.cuda.max_memory_allocated() / 1e9
        fresh_text = decode_text(solver, fresh_tokens, sep_id)
        print(f"  [Fresh] prefill={fresh_prefill_t:.3f}s gen={fresh_gen_t:.3f}s "
              f"total={fresh_total:.3f}s ({len(fresh_tokens)} tok)")
        print(f"  [Fresh] Text: '{fresh_text[:200]}'")

        # --- Cached pipeline ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"\n  [Cached] Demo prefill + cache + query prefill + generation...")

        # Demo-only prefill
        demo_kv, demo_t = demo_only_prefill(solver, demo_token_ids)
        demo_kv_len = demo_kv.get_seq_length()
        print(f"  [Cached] Demo prefill: {demo_t:.3f}s ({demo_kv_len} positions)")

        # Save to disk
        cache_path_n = os.path.join(CACHE_ROOT, f"und_demo_{n_demos}shot_kv.pt")
        t_save_start = time.perf_counter()
        save_kv_cache(demo_kv, cache_path_n)
        save_t = time.perf_counter() - t_save_start
        cache_mb = os.path.getsize(cache_path_n) / 1e6
        print(f"  [Cached] Save: {save_t:.3f}s ({cache_mb:.1f} MB)")

        # Load from disk
        torch.cuda.synchronize()
        t_load_start = time.perf_counter()
        loaded_kv = load_kv_cache(cache_path_n, solver.model.device)
        torch.cuda.synchronize()
        load_t = time.perf_counter() - t_load_start
        print(f"  [Cached] Load: {load_t:.3f}s")

        # Query prefill from cached demo KV
        full_kv, query_logits, query_t = query_prefill_from_cache(
            solver, query_ids, loaded_kv
        )
        print(f"  [Cached] Query prefill: {query_t:.3f}s")

        # Generate text
        cached_tokens, cached_gen_t = generate_text_from_cache(
            solver, prompt_tensor, prompt_len, full_kv, query_logits, sep_id
        )
        cached_total = load_t + query_t + cached_gen_t
        first_query_total = demo_t + save_t + load_t + query_t + cached_gen_t
        cached_text = decode_text(solver, cached_tokens, sep_id)
        cached_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  [Cached] gen={cached_gen_t:.3f}s")
        print(f"  [Cached] Amortized per-query: load({load_t:.3f}) + qprefill({query_t:.3f}) "
              f"+ gen({cached_gen_t:.3f}) = {cached_total:.3f}s")
        print(f"  [Cached] Text: '{cached_text[:200]}'")

        # Compare outputs
        tok_match, early_match, first_diff = token_match_rate(fresh_tokens, cached_tokens)
        quality = text_quality(fresh_text, cached_text)
        print(f"\n  Comparison (fresh vs cached):")
        print(f"    Token match: {tok_match:.4f} first_diff={first_diff}")
        print(f"    Text exact_match={quality['exact_match']} "
              f"char_similarity={quality['char_similarity']:.4f} "
              f"edit_dist={quality['edit_distance']}")

        # Speedup analysis
        prefill_speedup = fresh_prefill_t / (load_t + query_t) if (load_t + query_t) > 0 else float('inf')
        total_speedup = fresh_total / cached_total if cached_total > 0 else float('inf')
        if fresh_total > cached_total:
            break_even = (demo_t + save_t) / (fresh_total - cached_total)
        else:
            break_even = float('inf')

        # Prefill-to-generation ratio
        prefill_ratio_fresh = fresh_prefill_t / fresh_total if fresh_total > 0 else 0
        prefill_ratio_cached = query_t / cached_total if cached_total > 0 else 0

        print(f"\n  Speedup:")
        print(f"    Total per-query: {fresh_total:.3f}s → {cached_total:.3f}s "
              f"({total_speedup:.2f}x)")
        print(f"    Break-even: {break_even:.1f} queries")
        print(f"    Prefill/total ratio: fresh={prefill_ratio_fresh:.3f} "
              f"cached={prefill_ratio_cached:.3f}")

        phase2_results[n_demos] = {
            "prompt_len": seq_len, "demo_len": demo_len, "query_len": len(query_ids),
            "fresh": {
                "prefill": fresh_prefill_t, "gen": fresh_gen_t,
                "total": fresh_total, "peak_gb": fresh_peak,
                "n_tokens": len(fresh_tokens),
                "text": fresh_text[:500],
            },
            "cached": {
                "demo_prefill": demo_t, "save": save_t,
                "load": load_t, "query_prefill": query_t,
                "gen": cached_gen_t, "amortized_total": cached_total,
                "first_query_total": first_query_total,
                "cache_mb": cache_mb, "peak_gb": cached_peak,
                "n_tokens": len(cached_tokens),
                "text": cached_text[:500],
            },
            "comparison": {
                "token_match": tok_match, "early_20": early_match,
                "first_diff": first_diff, **quality,
            },
            "speedup": {
                "prefill": prefill_speedup, "total": total_speedup,
                "break_even_queries": break_even,
            },
            "prefill_gen_ratio": {
                "fresh_prefill_frac": prefill_ratio_fresh,
                "cached_prefill_frac": prefill_ratio_cached,
            },
        }

    # ─── Phase 3: Multi-Query Validation ──────────────────

    print("\n" + "=" * 70)
    print("PHASE 3: Multi-Query Validation (1-demo, 5 queries)")
    print("=" * 70)

    demo_1shot = demo_data[:1]
    # Build demo KV
    _, _, _, _, demo_ids_1 = build_understanding_prompt(
        solver, instances[0]["query"]["rgb_path"], demo_1shot, 1
    )
    demo_kv_1, _ = demo_only_prefill(solver, demo_ids_1)
    cache_path_validate = os.path.join(CACHE_ROOT, "und_demo_1shot_validate_kv.pt")
    save_kv_cache(demo_kv_1, cache_path_validate)

    phase3_results = []
    n_validate = min(5, len(instances))

    for qi in range(n_validate):
        qinst = instances[qi]
        query_path = qinst["query"]["rgb_path"]

        prompt_tensor, prompt_len, demo_len, query_ids, _ = \
            build_understanding_prompt(solver, query_path, demo_1shot, 1)

        # Fresh
        torch.cuda.empty_cache()
        fresh_kv, fresh_logits, fp_t = fresh_prefill(solver, prompt_tensor)
        fresh_tokens, fg_t = generate_text_from_cache(
            solver, prompt_tensor, prompt_len, fresh_kv, fresh_logits, sep_id
        )
        fresh_text = decode_text(solver, fresh_tokens, sep_id)

        # Cached
        torch.cuda.empty_cache()
        loaded_kv = load_kv_cache(cache_path_validate, solver.model.device)
        full_kv, q_logits, qp_t = query_prefill_from_cache(solver, query_ids, loaded_kv)
        cached_tokens, cg_t = generate_text_from_cache(
            solver, prompt_tensor, prompt_len, full_kv, q_logits, sep_id
        )
        cached_text = decode_text(solver, cached_tokens, sep_id)

        tok_match, early_match, first_diff = token_match_rate(fresh_tokens, cached_tokens)
        quality = text_quality(fresh_text, cached_text)

        print(f"  Q{qi}: {len(fresh_tokens)} tok | "
              f"fresh_pf={fp_t:.3f}s fresh_gen={fg_t:.3f}s | "
              f"cached_qp={qp_t:.3f}s cached_gen={cg_t:.3f}s | "
              f"exact={quality['exact_match']} char_sim={quality['char_similarity']:.4f}")
        print(f"    Fresh:  '{fresh_text[:120]}'")
        print(f"    Cached: '{cached_text[:120]}'")

        phase3_results.append({
            "query_idx": qi,
            "fresh_prefill": fp_t, "fresh_gen": fg_t,
            "cached_query_prefill": qp_t, "cached_gen": cg_t,
            "token_match": tok_match, "first_diff": first_diff,
            "text_quality": quality,
            "fresh_text": fresh_text[:300],
            "cached_text": cached_text[:300],
        })

    # ─── Phase 4: Cross-Task Comparison ───────────────────

    print("\n" + "=" * 70)
    print("PHASE 4: Cross-Task Comparison (AR-4 Image vs AR-UND Text)")
    print("=" * 70)

    print(f"\n{'Task':<10} {'Demos':>5} {'Output':>8} {'FreshPF':>9} {'FreshGen':>10} "
          f"{'FreshTot':>10} {'CachedGen':>10} {'CachedTot':>10} {'Speedup':>8} {'GenTok':>7}")
    print("-" * 98)

    for nd in [1, 2, 3]:
        # AR-4 (image output)
        ar4 = AR4_RESULTS[nd]
        print(f"{'AR-4':<10} {nd:>5} {'image':>8} {ar4['fresh_prefill']:>8.3f}s "
              f"{ar4['fresh_gen']:>9.1f}s {ar4['fresh_total']:>9.1f}s "
              f"{ar4['cached_gen']:>9.1f}s {ar4['cached_total']:>9.1f}s "
              f"{ar4['speedup']:>7.2f}x {ar4['n_gen_tokens']:>7d}")

        # AR-UND (text output)
        if nd in phase2_results:
            r = phase2_results[nd]
            print(f"{'AR-UND':<10} {nd:>5} {'text':>8} {r['fresh']['prefill']:>8.3f}s "
                  f"{r['fresh']['gen']:>9.3f}s {r['fresh']['total']:>9.3f}s "
                  f"{r['cached']['gen']:>9.3f}s {r['cached']['amortized_total']:>9.3f}s "
                  f"{r['speedup']['total']:>7.2f}x {r['fresh']['n_tokens']:>7d}")
        print()

    # Pathology analysis
    print("── Generation Pathology Analysis ──")
    for nd in [1, 2, 3]:
        if nd in phase2_results:
            r = phase2_results[nd]
            ar4 = AR4_RESULTS[nd]
            print(f"  {nd}-demo:")
            print(f"    AR-4  (image): fresh_gen={ar4['fresh_gen']:.1f}s  "
                  f"cached_gen={ar4['cached_gen']:.1f}s  "
                  f"ratio={ar4['fresh_gen']/ar4['cached_gen']:.2f}x")
            cached_gen = r['cached']['gen']
            fresh_gen = r['fresh']['gen']
            ratio = fresh_gen / cached_gen if cached_gen > 0 else float('inf')
            print(f"    AR-UND (text):  fresh_gen={fresh_gen:.3f}s  "
                  f"cached_gen={cached_gen:.3f}s  "
                  f"ratio={ratio:.2f}x")

    # Prefill-to-generation ratio comparison
    print("\n── Prefill-to-Generation Cost Ratio ──")
    for nd in [1, 2, 3]:
        if nd in phase2_results:
            r = phase2_results[nd]
            ar4 = AR4_RESULTS[nd]
            ar4_pf_frac = ar4['fresh_prefill'] / ar4['fresh_total']
            und_pf_frac = r['prefill_gen_ratio']['fresh_prefill_frac']
            print(f"  {nd}-demo: AR-4 prefill/total={ar4_pf_frac:.4f}  "
                  f"AR-UND prefill/total={und_pf_frac:.4f}")

    # ─── Summary ─────────────────────────────────────────

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n── Phase 0: Text Generation Viability ──")
    print(f"  Coherent text output: {phase0_results['coherent']}")
    print(f"  Sample: '{phase0_results['text'][:150]}'")

    print(f"\n── Phase 2: Latency Table ──")
    print(f"{'Demos':>5} {'Prompt':>7} {'Fresh PF':>9} {'Fresh Gen':>10} {'Fresh Tot':>10} "
          f"{'Cached Gen':>10} {'Cached Tot':>10} {'Speedup':>8}")
    print("-" * 70)
    for nd in [1, 2, 3]:
        r = phase2_results[nd]
        print(f"{nd:>5} {r['prompt_len']:>7} {r['fresh']['prefill']:>8.3f}s "
              f"{r['fresh']['gen']:>9.3f}s {r['fresh']['total']:>9.3f}s "
              f"{r['cached']['gen']:>9.3f}s {r['cached']['amortized_total']:>9.3f}s "
              f"{r['speedup']['total']:>7.2f}x")

    print(f"\n── Phase 3: Quality Validation (1-demo, {n_validate} queries) ──")
    exact_matches = sum(1 for r in phase3_results if r["text_quality"]["exact_match"])
    char_sims = [r["text_quality"]["char_similarity"] for r in phase3_results]
    tok_matches = [r["token_match"] for r in phase3_results]
    print(f"  Exact match: {exact_matches}/{n_validate}")
    print(f"  Char similarity: mean={np.mean(char_sims):.4f} "
          f"min={min(char_sims):.4f} max={max(char_sims):.4f}")
    print(f"  Token match: mean={np.mean(tok_matches):.4f}")

    # Pathology verdict
    pathology_present = False
    for nd in [2, 3]:
        if nd in phase2_results:
            r = phase2_results[nd]
            ratio = r['fresh']['gen'] / r['cached']['gen'] if r['cached']['gen'] > 0 else 1.0
            if ratio > 2.0:
                pathology_present = True
                break

    print(f"\n── Key Findings ──")
    print(f"  Generation pathology in text output: "
          f"{'YES — also affects text' if pathology_present else 'NO — text generation unaffected'}")
    if not pathology_present:
        print(f"    → Pathology is specific to LONG output generation, not prefill/position encoding")
        print(f"    → For text output, speedup comes purely from prefill amortization")

    caching_generalizes = all(
        phase2_results[nd]['speedup']['total'] > 0.95  # at least not slower
        for nd in [1, 2, 3]
    )
    quality_preserved = np.mean(char_sims) > 0.8 if char_sims else False

    print(f"  Prefix caching generalizes to understanding: "
          f"{'YES' if caching_generalizes else 'PARTIAL/NO'}")
    print(f"  Text quality preserved under caching: "
          f"{'YES' if quality_preserved else 'NO'} "
          f"(mean char_similarity={np.mean(char_sims):.4f})")

    # ─── Save Results ────────────────────────────────────

    all_results = {
        "experiment": "AR-UND: Understanding Task Generalization of Prefix Caching",
        "task": {
            "question": QUESTION,
            "demo_labels": DEMO_LABELS,
            "output_type": "text",
        },
        "config": {
            "model": "Lumina-mGPT-2.0-Omni",
            "max_position_embeddings": solver.model.config.max_position_embeddings,
            "rope_scaling": "dynamic_ntk_2.0",
            "cfg": CFG_SCALE, "seed": SEED,
            "max_gen_tokens": MAX_GEN_TOKENS,
        },
        "phase0_viability": phase0_results,
        "phase1_prompt_lengths": {str(k): v for k, v in phase1_results.items()},
        "phase2_latency": {str(k): v for k, v in phase2_results.items()},
        "phase3_multi_query": phase3_results,
        "phase4_cross_task": {
            "ar4_image_results": {str(k): v for k, v in AR4_RESULTS.items()},
            "pathology_in_text": pathology_present,
            "caching_generalizes": caching_generalizes,
            "quality_preserved": quality_preserved,
        },
    }
    results_path = os.path.join(OUTPUT_ROOT, "ar_und_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
