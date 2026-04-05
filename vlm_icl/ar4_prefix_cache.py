"""
AR-4: Demo-Prefix Caching + Position Encoding Extension

Goals:
  1. Extend position encoding to support 2-3 demo contexts.
  2. Implement demo-prefix KV caching: compute demo KV once, reuse across queries.
  3. Measure amortized latency: cached vs fresh prefill per-query cost.
  4. Validate generation quality across all conditions.

Key insight: demo tokens are identical across queries sharing the same demo pair.
If we pre-compute and cache the demo-portion KV, subsequent queries only need
to compute the query-specific portion — a fraction of the full prefill cost.
"""

import sys
import os
import json
import time
import copy
import numpy as np
import torch
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

# ══════════════════════════════════════════════��
# Path Setup (critical: CWD for MoVQGAN)
# ═══════════════════════════════════════════════
REPO_ROOT = "/home/zihua.meng/projects/vlm_inference/vlm_icl/Lumina-mGPT-2.0"
sys.path.insert(0, os.path.join(REPO_ROOT, "lumina_mgpt"))
sys.path.insert(0, REPO_ROOT)
os.chdir(os.path.join(REPO_ROOT, "lumina_mgpt"))

from data.item_processor import FlexARItemProcessor, center_crop
from transformers import DynamicCache, GenerationConfig

# ═══════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════
MODEL_PATH = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/weights/Lumina-mGPT-2.0-Omni"
DATA_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/data/nyu_depth_v2"
OUTPUT_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/outputs/ar4_prefix_cache"
CACHE_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/cache"

WIDTH, HEIGHT = 512, 512
CFG_SCALE = 4.0
IMAGE_TOP_K = 200
EOS_TOKEN_ID = 151666
IMAGE_START_TOKEN_ID = 151665
SEED = 42

MAX_GEN_TOKENS = 2500
MAX_TOTAL_LENGTH = 20000  # accommodate 3-demo + generation


# ═══════════════════════════════════════════════════════════
# Position Encoding Extension
# ═══════════════════════════════════════════════════════════

def enable_dynamic_ntk_scaling(model, scaling_factor=2.0):
    """Replace vanilla RoPE with Dynamic NTK scaling for extended sequences.

    Dynamic NTK adjusts base frequency when seq_len > max_position_embeddings,
    leaving short-sequence behavior unchanged.
    """
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
# Prompt Building
# ═══════════════════════════════════════════════════════════

def build_demo_image(demo_rgb_path, demo_depth_path):
    """Stitch depth (top) + RGB (bottom) into a 512x512 demo image."""
    rgb = Image.open(demo_rgb_path)
    depth = Image.open(demo_depth_path)
    rgb_c = center_crop(rgb, crop_size=[WIDTH, HEIGHT // 2])
    depth_c = center_crop(depth, crop_size=[WIDTH, HEIGHT // 2])
    stitched = Image.new("RGB", (WIDTH, HEIGHT))
    stitched.paste(depth_c, (0, 0))
    stitched.paste(rgb_c, (0, HEIGHT // 2))
    return stitched


def build_nshot_prompt(solver, query_path, demo_pairs, n_demos):
    """Build an N-shot depth estimation prompt.

    Returns:
        prompt_tensor: (1, seq_len) token IDs
        prompt_len: total number of tokens
        ref_tokens: reference tokens for image decoding
        demo_len: number of tokens from demo portion only
        query_ids: list of query token IDs (for separate prefill)
        demo_token_ids: list of all demo token IDs
    """
    # Process each demo image
    all_demo_tokens = []
    for i in range(n_demos):
        dp = demo_pairs[i]
        stitched = build_demo_image(dp["rgb_path"], dp["depth_colored_path"])
        tokens = solver.item_processor.process_image(stitched)["input_ids"]
        all_demo_tokens.extend(tokens)

    demo_len = len(all_demo_tokens)

    # Process query (text + image reference)
    img = Image.open(query_path)
    img_c = center_crop(img, crop_size=[WIDTH, HEIGHT // 2])
    cropped_path = os.path.join(OUTPUT_ROOT, "ar4_query_cropped.png")
    img_c.save(cropped_path)

    prompt_text = (
        f"Generate a dual-panel image of {WIDTH}x{HEIGHT} where the <upper half> "
        f"displays a <depth map>, while the <lower half> retains the original image "
        f"for direct visual comparison:\nAn indoor scene"
    )
    conversations = [
        {"from": "human", "value": prompt_text},
        {"from": "gpt", "value": "<|image|>"},
    ]
    item = {"image": [cropped_path], "conversations": conversations}
    solver.item_processor.transform = {
        "<|image|>": solver.item_processor.process_i2i_image
    }
    _prompt = solver.item_processor.process_item(item)

    if IMAGE_START_TOKEN_ID in _prompt:
        _prompt.pop()
        start_index = _prompt.index(IMAGE_START_TOKEN_ID)
        ref_tokens = _prompt[start_index:]
    else:
        ref_tokens = None

    query_ids = []
    for v in _prompt:
        if isinstance(v, int):
            query_ids.append(v)
        else:
            query_ids += v["input_ids"]

    all_ids = all_demo_tokens + query_ids
    prompt_tensor = torch.tensor(
        all_ids, dtype=torch.int64, device=solver.model.device
    ).unsqueeze(0)

    return prompt_tensor, len(all_ids), ref_tokens, demo_len, query_ids, all_demo_tokens


# ═══════════════════════════════════════════════════════════
# KV Cache Utilities
# ═══════════════════════════════════════════════════════════

def clone_kv_cache(kv_cache):
    """Deep-clone a DynamicCache (GPU tensors)."""
    new_cache = DynamicCache()
    for layer_idx in range(len(kv_cache.key_cache)):
        new_cache.update(
            kv_cache.key_cache[layer_idx].clone(),
            kv_cache.value_cache[layer_idx].clone(),
            layer_idx,
        )
    return new_cache


def save_kv_cache(kv_cache, path):
    """Save KV cache to disk."""
    data = {
        "key_cache": [k.cpu() for k in kv_cache.key_cache],
        "value_cache": [v.cpu() for v in kv_cache.value_cache],
    }
    torch.save(data, path)


def load_kv_cache(path, device):
    """Load KV cache from disk into a DynamicCache."""
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
    """Prefill query tokens using a pre-computed demo KV cache.

    The model automatically sets position_ids starting from
    demo_kv.get_seq_length(), so positions continue correctly.
    """
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


def generate_from_cache(solver, full_prompt_tensor, prompt_len, ref_tokens,
                        kv_cache, prefill_logits):
    """Sample first token from prefill logits, then generate from KV cache."""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    gen_config = GenerationConfig(
        max_new_tokens=MAX_GEN_TOKENS,
        max_length=MAX_TOTAL_LENGTH,
        temperature=1.0, do_sample=True, eos_token_id=[EOS_TOKEN_ID],
    )

    logits_proc = solver.create_logits_processor(cfg=CFG_SCALE, image_top_k=IMAGE_TOP_K)
    with torch.no_grad():
        processed = logits_proc(full_prompt_tensor, prefill_logits[:, -1, :])
        probs = torch.softmax(processed, dim=-1)
        first_token = torch.multinomial(probs, num_samples=1)
    del prefill_logits

    gen_ids = torch.cat([full_prompt_tensor, first_token], dim=1)
    logits_proc2 = solver.create_logits_processor(cfg=CFG_SCALE, image_top_k=IMAGE_TOP_K)

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
# Quality Metrics
# ═══════════════════════════════════════════════════════════

def decode_image(solver, ref_tokens, gen_tokens):
    decode_tokens = (list(ref_tokens) + list(gen_tokens)) if ref_tokens else list(gen_tokens)
    _, images = solver.decode_ids(decode_tokens)
    return images[0] if images else None


def compute_quality(baseline_img, test_img):
    from skimage.metrics import structural_similarity as ssim_fn
    b = np.array(baseline_img)
    t = np.array(test_img)
    if b.shape != t.shape:
        test_img = test_img.resize(baseline_img.size, Image.LANCZOS)
        t = np.array(test_img)
    ssim = ssim_fn(b, t, channel_axis=2, data_range=255)
    mse = float(np.mean((b.astype(float) - t.astype(float)) ** 2))
    return {"ssim": float(ssim), "mse": mse}


def token_match_rate(base, test):
    n = min(len(base), len(test))
    if n == 0:
        return 0.0, 0.0, 0
    matches = sum(1 for i in range(n) if base[i] == test[i])
    first_diff = next((i for i in range(n) if base[i] != test[i]), n)
    n100 = min(100, n)
    early = sum(1 for i in range(n100) if base[i] == test[i]) / n100
    return matches / n, early, first_diff


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("AR-4: Demo-Prefix Caching + Position Encoding Extension")
    print("=" * 60)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(CACHE_ROOT, exist_ok=True)

    # Load test data
    with open(os.path.join(DATA_ROOT, "task_instances.json")) as f:
        instances = json.load(f)
    print(f"Task instances: {len(instances)} (each has {len(instances[0]['demo_pairs'])} demo pairs)")

    # Load model
    print("\nLoading model...")
    from inference_solver import FlexARInferenceSolver
    solver = FlexARInferenceSolver(
        model_path=MODEL_PATH, precision="bf16", target_size=512
    )
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Config: max_position_embeddings={solver.model.config.max_position_embeddings}, "
          f"rope_theta={solver.model.config.rope_theta}")

    # ─── Phase 1: Position Encoding Investigation ───────────

    print("\n" + "=" * 60)
    print("PHASE 1: Position Encoding Investigation")
    print("=" * 60)

    inst = instances[0]
    phase1_results = {}

    for n_demos in [1, 2, 3]:
        print(f"\n--- {n_demos}-demo prompt ---")
        prompt_tensor, prompt_len, ref_tokens, demo_len, query_ids, demo_token_ids = \
            build_nshot_prompt(solver, inst["query"]["rgb_path"],
                              inst["demo_pairs"], n_demos)
        seq_len = prompt_tensor.shape[1]
        gen_total = seq_len + 2081  # expected total with generation
        exceeds = seq_len > solver.model.config.max_position_embeddings
        print(f"  Prompt tokens: {seq_len} (demo={demo_len}, query={len(query_ids)})")
        print(f"  With generation: ~{gen_total} tokens")
        print(f"  Exceeds max_position_embeddings ({solver.model.config.max_position_embeddings}): "
              f"{'YES' if exceeds else 'no'}")
        phase1_results[n_demos] = {
            "prompt_len": seq_len, "demo_len": demo_len,
            "query_len": len(query_ids), "exceeds_max": exceeds,
            "est_total_with_gen": gen_total,
        }

    # Test generation at each demo count
    # Enable Dynamic NTK scaling for extended sequences
    print("\nEnabling Dynamic NTK scaling for extended position support...")
    enable_dynamic_ntk_scaling(solver.model, scaling_factor=2.0)

    # Warmup with 1-demo (within original range)
    print("\nWarmup (1-demo, 100 tokens)...")
    prompt_tensor, prompt_len, ref_tokens, demo_len, query_ids, demo_token_ids = \
        build_nshot_prompt(solver, inst["query"]["rgb_path"],
                          inst["demo_pairs"], 1)
    lp = solver.create_logits_processor(cfg=CFG_SCALE, image_top_k=IMAGE_TOP_K)
    gc = GenerationConfig(
        max_new_tokens=100, max_length=MAX_TOTAL_LENGTH,
        temperature=1.0, do_sample=True, eos_token_id=[EOS_TOKEN_ID],
    )
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = solver.model.generate(prompt_tensor, gc, logits_processor=lp)
    print("Warmup complete.")

    # ─── Phase 2: Prefix Caching Validation ─────────────────

    print("\n" + "=" * 60)
    print("PHASE 2: Prefix Caching Validation (1-demo)")
    print("=" * 60)

    # Use multiple queries with the SAME demo to test prefix reuse
    # Instance 0 demo pair 0 as the shared demo
    demo_pair = [inst["demo_pairs"][0]]

    # Queries from instances 0, 1, 2
    query_instances = instances[:3]

    # Step A: Build demo-only KV cache
    print("\nStep A: Compute demo-only KV cache...")
    _, _, _, _, _, demo_token_ids_1 = build_nshot_prompt(
        solver, query_instances[0]["query"]["rgb_path"], demo_pair, 1
    )
    # demo_token_ids_1 is just the demo tokens (same regardless of query)
    demo_kv, demo_prefill_t = demo_only_prefill(solver, demo_token_ids_1)
    demo_kv_len = demo_kv.get_seq_length()
    print(f"  Demo KV cache: {demo_kv_len} positions, prefill time: {demo_prefill_t:.3f}s")

    # Save demo KV to disk
    cache_path = os.path.join(CACHE_ROOT, "demo_1shot_kv.pt")
    save_kv_cache(demo_kv, cache_path)
    cache_size_mb = os.path.getsize(cache_path) / 1e6
    print(f"  Saved to disk: {cache_path} ({cache_size_mb:.1f} MB)")

    # Step B: For each query, compare fresh vs cached generation
    print("\nStep B: Fresh vs Cached generation comparison...")
    phase2_results = []

    for qi, qinst in enumerate(query_instances):
        print(f"\n  Query {qi}:")
        query_path = qinst["query"]["rgb_path"]

        # Build full prompt for fresh generation
        prompt_tensor, prompt_len, ref_tokens, demo_len, query_ids, all_demo_ids = \
            build_nshot_prompt(solver, query_path, demo_pair, 1)
        print(f"    Full prompt: {prompt_len} tokens (demo={demo_len}, query={len(query_ids)})")

        # --- Fresh generation ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fresh_kv, fresh_logits, fresh_prefill_t = fresh_prefill(solver, prompt_tensor)
        fresh_tokens, fresh_gen_t = generate_from_cache(
            solver, prompt_tensor, prompt_len, ref_tokens, fresh_kv, fresh_logits
        )
        fresh_total = fresh_prefill_t + fresh_gen_t
        print(f"    Fresh: prefill={fresh_prefill_t:.3f}s gen={fresh_gen_t:.1f}s "
              f"total={fresh_total:.1f}s ({len(fresh_tokens)} tok)")

        # --- Cached generation ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Load demo KV from disk (simulates real caching scenario)
        torch.cuda.synchronize()
        t_load_start = time.perf_counter()
        cached_demo_kv = load_kv_cache(cache_path, solver.model.device)
        torch.cuda.synchronize()
        cache_load_t = time.perf_counter() - t_load_start

        # Query-only prefill extending from cached demo KV
        full_kv, query_logits, query_prefill_t = query_prefill_from_cache(
            solver, query_ids, cached_demo_kv
        )
        cached_tokens, cached_gen_t = generate_from_cache(
            solver, prompt_tensor, prompt_len, ref_tokens, full_kv, query_logits
        )
        cached_total = cache_load_t + query_prefill_t + cached_gen_t
        print(f"    Cached: load={cache_load_t:.3f}s query_prefill={query_prefill_t:.3f}s "
              f"gen={cached_gen_t:.1f}s total={cached_total:.1f}s ({len(cached_tokens)} tok)")

        # Compare outputs
        match_rate, early_match, first_diff = token_match_rate(fresh_tokens, cached_tokens)
        print(f"    Token match: {match_rate:.3f} early100={early_match:.2f} first_diff={first_diff}")

        # Decode and compare images
        fresh_img = decode_image(solver, ref_tokens, fresh_tokens)
        cached_img = decode_image(solver, ref_tokens, cached_tokens)
        if fresh_img and cached_img:
            quality = compute_quality(fresh_img, cached_img)
            print(f"    SSIM (fresh vs cached): {quality['ssim']:.6f}")
            fresh_img.save(os.path.join(OUTPUT_ROOT, f"1demo_q{qi}_fresh.png"))
            cached_img.save(os.path.join(OUTPUT_ROOT, f"1demo_q{qi}_cached.png"))
        else:
            quality = {"ssim": None, "mse": None}

        phase2_results.append({
            "query_idx": qi,
            "fresh": {"prefill": fresh_prefill_t, "gen": fresh_gen_t,
                      "total": fresh_total, "n_tokens": len(fresh_tokens)},
            "cached": {"load": cache_load_t, "query_prefill": query_prefill_t,
                       "gen": cached_gen_t, "total": cached_total,
                       "n_tokens": len(cached_tokens)},
            "comparison": {"token_match": match_rate, "early_100": early_match,
                           "first_diff": first_diff, **quality},
        })

    # ─── Phase 3: Multi-Demo + Latency Measurement ─────────

    print("\n" + "=" * 60)
    print("PHASE 3: Multi-Demo Latency Measurement")
    print("=" * 60)

    phase3_results = {}

    for n_demos in [1, 2, 3]:
        print(f"\n{'='*40}")
        print(f"{n_demos}-demo configuration")
        print(f"{'='*40}")

        demo_pairs_n = inst["demo_pairs"][:n_demos]

        # Build prompt
        prompt_tensor, prompt_len, ref_tokens, demo_len, query_ids, demo_token_ids = \
            build_nshot_prompt(solver, inst["query"]["rgb_path"],
                              demo_pairs_n, n_demos)
        seq_len = prompt_tensor.shape[1]
        print(f"  Prompt: {seq_len} tokens (demo={demo_len}, query={len(query_ids)})")

        # --- Fresh full pipeline ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"\n  [Fresh] Full prefill + generation...")
        fresh_kv, fresh_logits, fresh_prefill_t = fresh_prefill(solver, prompt_tensor)
        fresh_peak_prefill = torch.cuda.max_memory_allocated() / 1e9
        fresh_tokens, fresh_gen_t = generate_from_cache(
            solver, prompt_tensor, prompt_len, ref_tokens, fresh_kv, fresh_logits
        )
        fresh_total = fresh_prefill_t + fresh_gen_t
        fresh_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  [Fresh] prefill={fresh_prefill_t:.3f}s gen={fresh_gen_t:.1f}s "
              f"total={fresh_total:.1f}s peak={fresh_peak:.1f}GB ({len(fresh_tokens)} tok)")

        fresh_img = decode_image(solver, ref_tokens, fresh_tokens)
        if fresh_img:
            fresh_img.save(os.path.join(OUTPUT_ROOT, f"{n_demos}demo_fresh.png"))

        # --- Cached pipeline ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"\n  [Cached] Demo prefill + cache + query prefill + generation...")

        # Demo-only prefill
        demo_kv, demo_t = demo_only_prefill(solver, demo_token_ids)
        demo_kv_len = demo_kv.get_seq_length()
        print(f"  [Cached] Demo prefill: {demo_t:.3f}s ({demo_kv_len} positions)")

        # Save to disk
        cache_path_n = os.path.join(CACHE_ROOT, f"demo_{n_demos}shot_kv.pt")
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

        # Generate
        cached_tokens, cached_gen_t = generate_from_cache(
            solver, prompt_tensor, prompt_len, ref_tokens, full_kv, query_logits
        )
        cached_total = load_t + query_t + cached_gen_t
        first_query_total = demo_t + save_t + load_t + query_t + cached_gen_t
        print(f"  [Cached] gen={cached_gen_t:.1f}s")
        print(f"  [Cached] Amortized per-query: load({load_t:.3f}) + query_prefill({query_t:.3f}) "
              f"+ gen({cached_gen_t:.1f}) = {cached_total:.1f}s")
        print(f"  [Cached] First-query cost: {first_query_total:.1f}s "
              f"(includes demo_prefill + save)")

        cached_peak = torch.cuda.max_memory_allocated() / 1e9

        cached_img = decode_image(solver, ref_tokens, cached_tokens)
        if cached_img:
            cached_img.save(os.path.join(OUTPUT_ROOT, f"{n_demos}demo_cached.png"))

        # Compare
        match_rate, early_match, first_diff = token_match_rate(fresh_tokens, cached_tokens)
        if fresh_img and cached_img:
            quality = compute_quality(fresh_img, cached_img)
        else:
            quality = {"ssim": None, "mse": None}
        print(f"\n  Comparison (fresh vs cached):")
        print(f"    Token match: {match_rate:.4f} first_diff={first_diff}")
        print(f"    SSIM: {quality['ssim']}")

        # Speedup
        prefill_speedup = fresh_prefill_t / (load_t + query_t) if (load_t + query_t) > 0 else float('inf')
        total_speedup = fresh_total / cached_total if cached_total > 0 else float('inf')
        # Break-even: N queries where N * cached_total < demo_t + save_t + N * cached_total
        # The demo_t + save_t is the fixed overhead. Break-even is:
        # N_fresh * fresh_total = (demo_t + save_t) + N * cached_total
        # At N=1: fresh_total vs first_query_total
        # Break-even N: (demo_t + save_t) / (fresh_total - cached_total) if fresh_total > cached_total
        if fresh_total > cached_total:
            break_even = (demo_t + save_t) / (fresh_total - cached_total)
        else:
            break_even = float('inf')
        print(f"\n  Speedup:")
        print(f"    Prefill: {fresh_prefill_t:.3f}s → {load_t + query_t:.3f}s "
              f"({prefill_speedup:.2f}x)")
        print(f"    Total per-query: {fresh_total:.1f}s → {cached_total:.1f}s "
              f"({total_speedup:.2f}x)")
        print(f"    Break-even: {break_even:.1f} queries")

        phase3_results[n_demos] = {
            "prompt_len": seq_len, "demo_len": demo_len, "query_len": len(query_ids),
            "fresh": {"prefill": fresh_prefill_t, "gen": fresh_gen_t,
                      "total": fresh_total, "peak_gb": fresh_peak,
                      "n_tokens": len(fresh_tokens)},
            "cached": {
                "demo_prefill": demo_t, "save": save_t,
                "load": load_t, "query_prefill": query_t,
                "gen": cached_gen_t, "amortized_total": cached_total,
                "first_query_total": first_query_total,
                "cache_mb": cache_mb, "peak_gb": cached_peak,
                "n_tokens": len(cached_tokens),
            },
            "comparison": {"token_match": match_rate, "early_100": early_match,
                           "first_diff": first_diff, **quality},
            "speedup": {"prefill": prefill_speedup, "total": total_speedup,
                        "break_even_queries": break_even},
        }

    # ─── Phase 4: Multi-Query Validation (1-demo cached) ───

    print("\n" + "=" * 60)
    print("PHASE 4: Multi-Query Validation (1-demo, 5 queries)")
    print("=" * 60)

    demo_pair_0 = [inst["demo_pairs"][0]]
    _, _, _, _, _, demo_ids_0 = build_nshot_prompt(
        solver, instances[0]["query"]["rgb_path"], demo_pair_0, 1
    )

    # Compute and save demo KV
    demo_kv_0, _ = demo_only_prefill(solver, demo_ids_0)
    cache_path_q = os.path.join(CACHE_ROOT, "demo_1shot_validate_kv.pt")
    save_kv_cache(demo_kv_0, cache_path_q)

    phase4_results = []
    n_validate = min(5, len(instances))

    for qi in range(n_validate):
        qinst = instances[qi]
        query_path = qinst["query"]["rgb_path"]

        prompt_tensor, prompt_len, ref_tokens, demo_len, query_ids, _ = \
            build_nshot_prompt(solver, query_path, demo_pair_0, 1)

        # Fresh
        torch.cuda.empty_cache()
        fresh_kv, fresh_logits, fp_t = fresh_prefill(solver, prompt_tensor)
        fresh_tokens, fg_t = generate_from_cache(
            solver, prompt_tensor, prompt_len, ref_tokens, fresh_kv, fresh_logits
        )

        # Cached
        torch.cuda.empty_cache()
        loaded_kv = load_kv_cache(cache_path_q, solver.model.device)
        full_kv, q_logits, qp_t = query_prefill_from_cache(solver, query_ids, loaded_kv)
        cached_tokens, cg_t = generate_from_cache(
            solver, prompt_tensor, prompt_len, ref_tokens, full_kv, q_logits
        )

        match_rate, early_match, first_diff = token_match_rate(fresh_tokens, cached_tokens)
        fresh_img = decode_image(solver, ref_tokens, fresh_tokens)
        cached_img = decode_image(solver, ref_tokens, cached_tokens)
        if fresh_img and cached_img:
            quality = compute_quality(fresh_img, cached_img)
            fresh_img.save(os.path.join(OUTPUT_ROOT, f"validate_q{qi}_fresh.png"))
            cached_img.save(os.path.join(OUTPUT_ROOT, f"validate_q{qi}_cached.png"))
        else:
            quality = {"ssim": None, "mse": None}

        print(f"  Q{qi}: fresh_prefill={fp_t:.3f}s cached_qprefill={qp_t:.3f}s "
              f"tok_match={match_rate:.4f} first_diff={first_diff} "
              f"SSIM={quality['ssim']}")

        phase4_results.append({
            "query_idx": qi,
            "fresh_prefill": fp_t, "fresh_gen": fg_t,
            "cached_query_prefill": qp_t, "cached_gen": cg_t,
            "token_match": match_rate, "first_diff": first_diff,
            "ssim": quality["ssim"],
        })

    # ─── Summary ────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Config':<12s} {'Demos':>5s} {'Prompt':>7s} {'Fresh Prefill':>13s} "
          f"{'Cached(load+qp)':>16s} {'Speedup':>8s} {'SSIM':>8s} {'TokMatch':>9s}")
    print("-" * 80)
    for nd in [1, 2, 3]:
        r = phase3_results[nd]
        f_pf = r["fresh"]["prefill"]
        c_lq = r["cached"]["load"] + r["cached"]["query_prefill"]
        sp = r["speedup"]["prefill"]
        ssim = r["comparison"].get("ssim", "N/A")
        tm = r["comparison"]["token_match"]
        ssim_s = f"{ssim:.4f}" if ssim is not None else "N/A"
        print(f"{'fresh→cache':<12s} {nd:>5d} {r['prompt_len']:>7d} "
              f"{f_pf:>12.3f}s {c_lq:>15.3f}s {sp:>7.2f}x {ssim_s:>8s} {tm:>8.4f}")

    print(f"\n── Amortized Per-Query Cost ──")
    for nd in [1, 2, 3]:
        r = phase3_results[nd]
        print(f"  {nd}-demo: fresh={r['fresh']['total']:.1f}s  "
              f"cached={r['cached']['amortized_total']:.1f}s  "
              f"break-even={r['speedup']['break_even_queries']:.1f} queries")

    print(f"\n── Phase 4: Multi-Query Consistency (1-demo) ──")
    ssims = [r["ssim"] for r in phase4_results if r["ssim"] is not None]
    matches = [r["token_match"] for r in phase4_results]
    diffs = [r["first_diff"] for r in phase4_results]
    if ssims:
        print(f"  SSIM (fresh vs cached): mean={np.mean(ssims):.6f} "
              f"min={min(ssims):.6f} max={max(ssims):.6f}")
    print(f"  Token match: mean={np.mean(matches):.4f}")
    print(f"  First diff: {diffs}")

    # Save results
    all_results = {
        "experiment": "AR-4: Demo-Prefix Caching + Position Encoding Extension",
        "config": {
            "model": "Lumina-mGPT-2.0-Omni",
            "max_position_embeddings": solver.model.config.max_position_embeddings,
            "rope_theta": solver.model.config.rope_theta,
            "rope_scaling": "dynamic_ntk_2.0",
            "cfg": CFG_SCALE, "seed": SEED,
        },
        "phase1_prompt_lengths": phase1_results,
        "phase2_prefix_validation": phase2_results,
        "phase3_latency": {str(k): v for k, v in phase3_results.items()},
        "phase4_multi_query": phase4_results,
    }
    results_path = os.path.join(OUTPUT_ROOT, "ar4_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")
    print(f"Images saved: {OUTPUT_ROOT}/")
    print(f"KV caches saved: {CACHE_ROOT}/")


if __name__ == "__main__":
    main()
