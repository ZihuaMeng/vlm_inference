"""
AR-5: Speedup Mechanism Decomposition

Decomposes the AR-4 5x speedup to determine how much comes from:
  (a) Avoiding DynamicNTK RoPE recomputation pathology
  (b) Genuine prefill amortization
  (c) Other factors

Approach: Run 2-demo fresh pipeline under multiple RoPE configurations
and compare to cached pipeline.

Configs:
  A. original_dnk: DynamicNTK as-is (pathological, baseline)
  B. precomputed_dnk: Pre-compute inv_freq at max seq_len, use vanilla RoPE
  C. cached_pipeline: AR-4 cached pipeline (for comparison)
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
# Path Setup
# ═══════════════════════════════════════════════
REPO_ROOT = "/home/zihua.meng/projects/vlm_inference/vlm_icl/Lumina-mGPT-2.0"
sys.path.insert(0, os.path.join(REPO_ROOT, "lumina_mgpt"))
sys.path.insert(0, REPO_ROOT)
os.chdir(os.path.join(REPO_ROOT, "lumina_mgpt"))

from data.item_processor import FlexARItemProcessor, center_crop
from transformers import DynamicCache, GenerationConfig
from transformers.models.chameleon.modeling_chameleon import (
    ChameleonDynamicNTKScalingRotaryEmbedding,
    ChameleonRotaryEmbedding,
)

# ═══════════════════════════════════════════════
# Constants (same as AR-4)
# ═══════════════════════════════════════════════
MODEL_PATH = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/weights/Lumina-mGPT-2.0-Omni"
DATA_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/data/nyu_depth_v2"
OUTPUT_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/outputs/ar5_decomposition"
CACHE_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/cache"

WIDTH, HEIGHT = 512, 512
CFG_SCALE = 4.0
IMAGE_TOP_K = 200
EOS_TOKEN_ID = 151666
IMAGE_START_TOKEN_ID = 151665
SEED = 42
MAX_GEN_TOKENS = 2500
MAX_TOTAL_LENGTH = 20000


# ═══════════════════════════════════════════════
# RoPE Configuration Functions
# ═══════════════════════════════════════════════

def enable_dynamic_ntk_scaling(model, scaling_factor=2.0):
    """Standard DynamicNTK (same as AR-4). Recomputes inv_freq every step."""
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    replaced = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.rotary_emb = ChameleonDynamicNTKScalingRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=model.config.max_position_embeddings,
            base=model.config.rope_theta,
            scaling_factor=scaling_factor,
        ).to(model.device)
        replaced += 1
    print(f"  [original_dnk] Enabled DynamicNTK (factor={scaling_factor}) on {replaced} layers")


def enable_precomputed_ntk(model, max_seq_len, scaling_factor=2.0):
    """Pre-compute inv_freq for max_seq_len and use vanilla RoPE (no per-step recomputation).

    This eliminates the DynamicNTK recomputation and torch.max() GPU→CPU sync.
    The base frequency is set to the value DynamicNTK would compute at max_seq_len.
    """
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    dim = head_dim
    base = config.rope_theta

    # Compute the adjusted base at max_seq_len (same formula as DynamicNTK)
    adjusted_base = base * (
        (scaling_factor * max_seq_len / config.max_position_embeddings) - (scaling_factor - 1)
    ) ** (dim / (dim - 2))

    inv_freq = 1.0 / (
        adjusted_base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
    )

    replaced = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        new_emb = ChameleonRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_seq_len,
            base=adjusted_base,
        ).to(model.device)
        new_emb.register_buffer("inv_freq", inv_freq.to(model.device), persistent=False)
        attn.rotary_emb = new_emb
        replaced += 1
    print(f"  [precomputed_ntk] Pre-computed NTK (base={adjusted_base:.1f}, "
          f"max_seq={max_seq_len}) on {replaced} layers")


def restore_vanilla_rope(model):
    """Restore original vanilla RoPE (no scaling). For ablation only."""
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    replaced = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.rotary_emb = ChameleonRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        ).to(model.device)
        replaced += 1
    print(f"  [vanilla_rope] Restored vanilla RoPE on {replaced} layers")


# ═══════════════════════════════════════════════
# AR-4 Prompt Building & Utilities (reused)
# ═══════════════════════════════════════════════

def build_demo_image(demo_rgb_path, demo_depth_path):
    rgb = Image.open(demo_rgb_path)
    depth = Image.open(demo_depth_path)
    rgb_c = center_crop(rgb, crop_size=[WIDTH, HEIGHT // 2])
    depth_c = center_crop(depth, crop_size=[WIDTH, HEIGHT // 2])
    stitched = Image.new("RGB", (WIDTH, HEIGHT))
    stitched.paste(depth_c, (0, 0))
    stitched.paste(rgb_c, (0, HEIGHT // 2))
    return stitched


def build_nshot_prompt(solver, query_path, demo_pairs, n_demos):
    all_demo_tokens = []
    for i in range(n_demos):
        dp = demo_pairs[i]
        stitched = build_demo_image(dp["rgb_path"], dp["depth_colored_path"])
        tokens = solver.item_processor.process_image(stitched)["input_ids"]
        all_demo_tokens.extend(tokens)
    demo_len = len(all_demo_tokens)

    img = Image.open(query_path)
    img_c = center_crop(img, crop_size=[WIDTH, HEIGHT // 2])
    cropped_path = os.path.join(OUTPUT_ROOT, "ar5_query_cropped.png")
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


# ═══════════════════════════════════════════════
# Prefill & Generation (same as AR-4)
# ═══════════════════════════════════════════════

def fresh_prefill(solver, prompt_tensor):
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


def decode_image(solver, ref_tokens, gen_tokens):
    decode_tokens = (list(ref_tokens) + list(gen_tokens)) if ref_tokens else list(gen_tokens)
    _, images = solver.decode_ids(decode_tokens)
    return images[0] if images else None


def compute_ssim(baseline_img, test_img):
    from skimage.metrics import structural_similarity as ssim_fn
    b = np.array(baseline_img)
    t = np.array(test_img)
    if b.shape != t.shape:
        test_img = test_img.resize(baseline_img.size, Image.LANCZOS)
        t = np.array(test_img)
    return float(ssim_fn(b, t, channel_axis=2, data_range=255))


# ═══════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════

def main():
    print("=" * 70)
    print("AR-5: Speedup Mechanism Decomposition")
    print("=" * 70)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Load data
    with open(os.path.join(DATA_ROOT, "task_instances.json")) as f:
        instances = json.load(f)
    inst = instances[0]

    # Load model
    print("\nLoading model...")
    from inference_solver import FlexARInferenceSolver
    solver = FlexARInferenceSolver(
        model_path=MODEL_PATH, precision="bf16", target_size=512
    )
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ─── PHASE 1: RoPE Investigation ──────────────────────

    print("\n" + "=" * 70)
    print("PHASE 1: RoPE Recomputation Investigation")
    print("=" * 70)

    # 1a. Document the recomputation behavior
    print("""
Analysis of ChameleonDynamicNTKScalingRotaryEmbedding.forward():

  seq_len = torch.max(position_ids) + 1     ← GPU→CPU sync on every call
  if seq_len > self.max_position_embeddings: ← triggers every gen step for 2+ demos
      base = self.base * ((scaling * seq_len / max_pos) - (scaling - 1)) ** (dim/(dim-2))
      inv_freq = 1.0 / (base ** torch.arange(...).to(device))  ← new GPU tensor every call
      self.register_buffer("inv_freq", inv_freq)                ← buffer replaced every call

Result: For 2-demo generation (seq_len 10457→12538):
  - 32 layers × 2081 steps = 66,592 calls to this code
  - Each call: torch.max() sync + tensor creation + register_buffer
  - Total: thousands of GPU→CPU synchronization points
""")

    # 1b. Warmup with standard DynamicNTK
    print("Setting up DynamicNTK and warming up...")
    enable_dynamic_ntk_scaling(solver.model, scaling_factor=2.0)

    # Build 2-demo prompt
    prompt_tensor, prompt_len, ref_tokens, demo_len, query_ids, demo_token_ids = \
        build_nshot_prompt(solver, inst["query"]["rgb_path"],
                          inst["demo_pairs"], 2)
    max_seq_len = prompt_len + MAX_GEN_TOKENS
    print(f"  2-demo prompt: {prompt_len} tokens (demo={demo_len}, query={len(query_ids)})")
    print(f"  Max expected seq_len: {max_seq_len}")

    # Warmup generation (1-demo, short)
    print("  Warmup (1-demo, 100 tokens)...")
    pt_1d, pl_1d, rt_1d, _, _, _ = build_nshot_prompt(
        solver, inst["query"]["rgb_path"], inst["demo_pairs"], 1
    )
    gc_warmup = GenerationConfig(
        max_new_tokens=100, max_length=MAX_TOTAL_LENGTH,
        temperature=1.0, do_sample=True, eos_token_id=[EOS_TOKEN_ID],
    )
    lp = solver.create_logits_processor(cfg=CFG_SCALE, image_top_k=IMAGE_TOP_K)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = solver.model.generate(pt_1d, gc_warmup, logits_processor=lp)
    print("  Warmup done.")

    # 1c. Additional warmup: run a SHORT 2-demo generation to warm CUDA kernels
    # at the 2-demo sequence length (avoiding cold-start confound from AR-UND)
    print("  Additional warmup (2-demo, 50 tokens)...")
    gc_warmup2 = GenerationConfig(
        max_new_tokens=50, max_length=MAX_TOTAL_LENGTH,
        temperature=1.0, do_sample=True, eos_token_id=[EOS_TOKEN_ID],
    )
    lp2 = solver.create_logits_processor(cfg=CFG_SCALE, image_top_k=IMAGE_TOP_K)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = solver.model.generate(prompt_tensor, gc_warmup2, logits_processor=lp2)
    print("  Additional warmup done.")

    # ─── Experiment A: Original DynamicNTK (fresh 2-demo) ───

    print("\n" + "-" * 50)
    print("Experiment A: ORIGINAL DynamicNTK — Fresh 2-demo")
    print("-" * 50)

    # Re-enable DynamicNTK (warmup may have changed state)
    enable_dynamic_ntk_scaling(solver.model, scaling_factor=2.0)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    kv_a, logits_a, prefill_t_a = fresh_prefill(solver, prompt_tensor)
    tokens_a, gen_t_a = generate_from_cache(
        solver, prompt_tensor, prompt_len, ref_tokens, kv_a, logits_a
    )
    total_a = prefill_t_a + gen_t_a
    per_tok_a = gen_t_a / len(tokens_a) if tokens_a else 0

    img_a = decode_image(solver, ref_tokens, tokens_a)
    if img_a:
        img_a.save(os.path.join(OUTPUT_ROOT, "exp_a_original_dnk.png"))

    print(f"  Prefill: {prefill_t_a:.3f}s")
    print(f"  Generation: {gen_t_a:.1f}s ({len(tokens_a)} tokens, {per_tok_a:.4f} s/tok)")
    print(f"  Total: {total_a:.1f}s")

    # ─── Experiment B: Pre-computed NTK (fresh 2-demo) ────

    print("\n" + "-" * 50)
    print("Experiment B: PRE-COMPUTED NTK — Fresh 2-demo")
    print("(inv_freq computed once at max_seq_len, no per-step recomputation)")
    print("-" * 50)

    enable_precomputed_ntk(solver.model, max_seq_len=max_seq_len, scaling_factor=2.0)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    kv_b, logits_b, prefill_t_b = fresh_prefill(solver, prompt_tensor)
    tokens_b, gen_t_b = generate_from_cache(
        solver, prompt_tensor, prompt_len, ref_tokens, kv_b, logits_b
    )
    total_b = prefill_t_b + gen_t_b
    per_tok_b = gen_t_b / len(tokens_b) if tokens_b else 0

    img_b = decode_image(solver, ref_tokens, tokens_b)
    if img_b:
        img_b.save(os.path.join(OUTPUT_ROOT, "exp_b_precomputed_ntk.png"))

    print(f"  Prefill: {prefill_t_b:.3f}s")
    print(f"  Generation: {gen_t_b:.1f}s ({len(tokens_b)} tokens, {per_tok_b:.4f} s/tok)")
    print(f"  Total: {total_b:.1f}s")

    # ─── Experiment C: Cached pipeline (original DynamicNTK) ─

    print("\n" + "-" * 50)
    print("Experiment C: CACHED PIPELINE — 2-demo (original DynamicNTK)")
    print("-" * 50)

    enable_dynamic_ntk_scaling(solver.model, scaling_factor=2.0)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Demo-only prefill
    demo_kv, demo_t = demo_only_prefill(solver, demo_token_ids)
    print(f"  Demo prefill: {demo_t:.3f}s ({demo_kv.get_seq_length()} positions)")

    # Save/load (to match AR-4's full pipeline)
    cache_path = os.path.join(CACHE_ROOT, "ar5_demo_2shot_kv.pt")
    save_kv_cache(demo_kv, cache_path)
    loaded_kv = load_kv_cache(cache_path, solver.model.device)

    # Query prefill from cache
    full_kv, query_logits, query_t = query_prefill_from_cache(
        solver, query_ids, loaded_kv
    )
    print(f"  Query prefill: {query_t:.3f}s")

    # Generate
    tokens_c, gen_t_c = generate_from_cache(
        solver, prompt_tensor, prompt_len, ref_tokens, full_kv, query_logits
    )
    cached_total = query_t + gen_t_c  # amortized (excluding demo prefill + save/load)
    per_tok_c = gen_t_c / len(tokens_c) if tokens_c else 0

    img_c = decode_image(solver, ref_tokens, tokens_c)
    if img_c:
        img_c.save(os.path.join(OUTPUT_ROOT, "exp_c_cached.png"))

    print(f"  Generation: {gen_t_c:.1f}s ({len(tokens_c)} tokens, {per_tok_c:.4f} s/tok)")
    print(f"  Amortized total (qprefill + gen): {cached_total:.1f}s")

    # ─── Experiment D: Pre-computed NTK + cached pipeline ──

    print("\n" + "-" * 50)
    print("Experiment D: PRE-COMPUTED NTK + CACHED — 2-demo")
    print("-" * 50)

    enable_precomputed_ntk(solver.model, max_seq_len=max_seq_len, scaling_factor=2.0)

    torch.cuda.empty_cache()

    # Re-create demo KV with pre-computed NTK
    demo_kv_d, demo_t_d = demo_only_prefill(solver, demo_token_ids)
    loaded_kv_d = clone_kv_cache(demo_kv_d)

    full_kv_d, query_logits_d, query_t_d = query_prefill_from_cache(
        solver, query_ids, loaded_kv_d
    )
    tokens_d, gen_t_d = generate_from_cache(
        solver, prompt_tensor, prompt_len, ref_tokens, full_kv_d, query_logits_d
    )
    cached_total_d = query_t_d + gen_t_d
    per_tok_d = gen_t_d / len(tokens_d) if tokens_d else 0

    print(f"  Query prefill: {query_t_d:.3f}s")
    print(f"  Generation: {gen_t_d:.1f}s ({len(tokens_d)} tokens, {per_tok_d:.4f} s/tok)")
    print(f"  Amortized total: {cached_total_d:.1f}s")

    # Free Phase 1 KV caches (keep token lists for results JSON)
    del kv_a, kv_b, full_kv, demo_kv_d, full_kv_d
    torch.cuda.empty_cache()

    # ─── PHASE 2: 3-demo measurements ────────────────────

    print("\n" + "=" * 70)
    print("PHASE 2: 3-Demo Measurements")
    print("=" * 70)

    prompt_tensor_3, prompt_len_3, ref_tokens_3, demo_len_3, query_ids_3, demo_token_ids_3 = \
        build_nshot_prompt(solver, inst["query"]["rgb_path"],
                          inst["demo_pairs"], 3)
    max_seq_3 = prompt_len_3 + MAX_GEN_TOKENS
    print(f"  3-demo prompt: {prompt_len_3} tokens")

    # 3-demo original DynamicNTK fresh
    print("\n  [3-demo] Original DynamicNTK — Fresh...")
    enable_dynamic_ntk_scaling(solver.model, scaling_factor=2.0)
    torch.cuda.empty_cache()
    kv_3a, logits_3a, pf_3a = fresh_prefill(solver, prompt_tensor_3)
    tokens_3a, gen_3a = generate_from_cache(
        solver, prompt_tensor_3, prompt_len_3, ref_tokens_3, kv_3a, logits_3a
    )
    per_tok_3a = gen_3a / len(tokens_3a) if tokens_3a else 0
    print(f"    Prefill: {pf_3a:.3f}s  Gen: {gen_3a:.1f}s  "
          f"({len(tokens_3a)} tok, {per_tok_3a:.4f} s/tok)  Total: {pf_3a + gen_3a:.1f}s")

    # 3-demo pre-computed NTK fresh
    print("  [3-demo] Pre-computed NTK — Fresh...")
    enable_precomputed_ntk(solver.model, max_seq_len=max_seq_3, scaling_factor=2.0)
    torch.cuda.empty_cache()
    kv_3b, logits_3b, pf_3b = fresh_prefill(solver, prompt_tensor_3)
    tokens_3b, gen_3b = generate_from_cache(
        solver, prompt_tensor_3, prompt_len_3, ref_tokens_3, kv_3b, logits_3b
    )
    per_tok_3b = gen_3b / len(tokens_3b) if tokens_3b else 0
    print(f"    Prefill: {pf_3b:.3f}s  Gen: {gen_3b:.1f}s  "
          f"({len(tokens_3b)} tok, {per_tok_3b:.4f} s/tok)  Total: {pf_3b + gen_3b:.1f}s")

    # 3-demo cached (original DynamicNTK)
    print("  [3-demo] Cached (original DynamicNTK)...")
    enable_dynamic_ntk_scaling(solver.model, scaling_factor=2.0)
    torch.cuda.empty_cache()
    demo_kv_3, _ = demo_only_prefill(solver, demo_token_ids_3)
    loaded_3 = clone_kv_cache(demo_kv_3)
    full_kv_3, q_logits_3, qt_3 = query_prefill_from_cache(solver, query_ids_3, loaded_3)
    tokens_3c, gen_3c = generate_from_cache(
        solver, prompt_tensor_3, prompt_len_3, ref_tokens_3, full_kv_3, q_logits_3
    )
    per_tok_3c = gen_3c / len(tokens_3c) if tokens_3c else 0
    print(f"    QPrefill: {qt_3:.3f}s  Gen: {gen_3c:.1f}s  "
          f"({len(tokens_3c)} tok, {per_tok_3c:.4f} s/tok)  Total: {qt_3 + gen_3c:.1f}s")

    # Free Phase 2 KV caches (keep token lists for results JSON)
    del kv_3a, logits_3a, kv_3b, logits_3b
    del demo_kv_3, full_kv_3, q_logits_3
    torch.cuda.empty_cache()

    # ─── PHASE 3: Quality Confirmation ────────────────────

    print("\n" + "=" * 70)
    print("PHASE 3: Quality Confirmation (2-demo cached, 5 additional queries)")
    print("=" * 70)

    # Set up cached pipeline with pre-computed NTK for quality validation
    enable_precomputed_ntk(solver.model, max_seq_len=max_seq_len, scaling_factor=2.0)
    demo_pair_2 = inst["demo_pairs"][:2]
    _, _, _, _, _, demo_ids_2 = build_nshot_prompt(
        solver, instances[0]["query"]["rgb_path"], demo_pair_2, 2
    )
    demo_kv_q, _ = demo_only_prefill(solver, demo_ids_2)
    cache_path_q = os.path.join(CACHE_ROOT, "ar5_demo_2shot_validate_kv.pt")
    save_kv_cache(demo_kv_q, cache_path_q)
    del demo_kv_q
    torch.cuda.empty_cache()

    quality_results = []
    n_queries = min(5, len(instances))

    # Run pre-computed NTK fresh for Q0 as quality baseline, then cached for all
    # (Skip original DynamicNTK fresh — takes ~990s/query and OOMs with accumulation)
    enable_precomputed_ntk(solver.model, max_seq_len=max_seq_len, scaling_factor=2.0)

    for qi in range(n_queries):
        qinst = instances[qi]
        query_path = qinst["query"]["rgb_path"]

        pt_q, pl_q, rt_q, dl_q, qi_q, _ = build_nshot_prompt(
            solver, query_path, demo_pair_2, 2
        )

        # Fresh (pre-computed NTK — fast, for quality baseline)
        torch.cuda.empty_cache()
        fkv, flog, fpt = fresh_prefill(solver, pt_q)
        ftok, fgt = generate_from_cache(solver, pt_q, pl_q, rt_q, fkv, flog)
        del fkv, flog
        torch.cuda.empty_cache()

        # Cached (pre-computed NTK)
        lkv = load_kv_cache(cache_path_q, solver.model.device)
        ckv, clog, cpt = query_prefill_from_cache(solver, qi_q, lkv)
        del lkv
        ctok, cgt = generate_from_cache(solver, pt_q, pl_q, rt_q, ckv, clog)
        del ckv, clog
        torch.cuda.empty_cache()

        fimg = decode_image(solver, rt_q, ftok)
        cimg = decode_image(solver, rt_q, ctok)
        ssim_val = compute_ssim(fimg, cimg) if fimg and cimg else None

        if fimg:
            fimg.save(os.path.join(OUTPUT_ROOT, f"q{qi}_fresh.png"))
        if cimg:
            cimg.save(os.path.join(OUTPUT_ROOT, f"q{qi}_cached.png"))

        print(f"  Q{qi}: fresh_gen={fgt:.1f}s cached_gen={cgt:.1f}s SSIM={ssim_val:.4f}" if ssim_val else
              f"  Q{qi}: fresh_gen={fgt:.1f}s cached_gen={cgt:.1f}s SSIM=N/A")
        quality_results.append({
            "query_idx": qi,
            "fresh_gen": fgt, "cached_gen": cgt,
            "ssim": ssim_val,
            "fresh_tokens": len(ftok), "cached_tokens": len(ctok),
        })
        del ftok, ctok, fimg, cimg, pt_q

    # ─── PHASE 4: Speedup Attribution ─────────────────────

    print("\n" + "=" * 70)
    print("PHASE 4: Speedup Attribution")
    print("=" * 70)

    # Summary table
    print(f"\n── 2-Demo Decomposition Table ──")
    print(f"{'Config':<30} {'Prefill':>8} {'Gen':>8} {'s/tok':>8} {'Total':>8} {'vs A':>8}")
    print("-" * 80)
    print(f"{'A. Original DynamicNTK fresh':<30} {prefill_t_a:>7.3f}s {gen_t_a:>7.1f}s "
          f"{per_tok_a:>7.4f}  {total_a:>7.1f}s {'1.00x':>8}")
    print(f"{'B. Pre-computed NTK fresh':<30} {prefill_t_b:>7.3f}s {gen_t_b:>7.1f}s "
          f"{per_tok_b:>7.4f}  {total_b:>7.1f}s {total_a/total_b:>7.2f}x")
    print(f"{'C. Cached (original DynamicNTK)':<30} {query_t:>7.3f}s {gen_t_c:>7.1f}s "
          f"{per_tok_c:>7.4f}  {cached_total:>7.1f}s {total_a/cached_total:>7.2f}x")
    print(f"{'D. Cached + pre-computed NTK':<30} {query_t_d:>7.3f}s {gen_t_d:>7.1f}s "
          f"{per_tok_d:>7.4f}  {cached_total_d:>7.1f}s {total_a/cached_total_d:>7.2f}x")

    print(f"\n── 3-Demo Decomposition Table ──")
    print(f"{'Config':<30} {'Prefill':>8} {'Gen':>8} {'s/tok':>8} {'Total':>8}")
    print("-" * 72)
    print(f"{'Original DynamicNTK fresh':<30} {pf_3a:>7.3f}s {gen_3a:>7.1f}s "
          f"{per_tok_3a:>7.4f}  {pf_3a+gen_3a:>7.1f}s")
    print(f"{'Pre-computed NTK fresh':<30} {pf_3b:>7.3f}s {gen_3b:>7.1f}s "
          f"{per_tok_3b:>7.4f}  {pf_3b+gen_3b:>7.1f}s")
    print(f"{'Cached (original DynamicNTK)':<30} {qt_3:>7.3f}s {gen_3c:>7.1f}s "
          f"{per_tok_3c:>7.4f}  {qt_3+gen_3c:>7.1f}s")

    # Attribution
    rope_fix_speedup = total_a / total_b if total_b > 0 else 0
    cached_speedup = total_a / cached_total if cached_total > 0 else 0

    # If B ≈ C: rope fix explains all speedup → no genuine prefill amortization
    # If B ≈ A: rope fix doesn't help → something else causes pathology
    # If B between A and C: rope fix helps partially

    print(f"\n── Speedup Attribution (2-demo) ──")
    print(f"  Total AR-4 speedup (A→C): {cached_speedup:.2f}x")
    print(f"  RoPE fix alone (A→B):     {rope_fix_speedup:.2f}x")

    if rope_fix_speedup > 1.5:
        residual = cached_speedup / rope_fix_speedup
        print(f"  Residual after RoPE fix:  {residual:.2f}x")
        print(f"\n  Attribution:")
        print(f"    RoPE pathology avoidance: {rope_fix_speedup:.2f}x")
        print(f"    Prefill amortization + other: {residual:.2f}x")
    else:
        print(f"  RoPE fix does NOT explain the speedup.")
        print(f"  The pathology source is elsewhere in the pipeline.")

    # Quality summary
    ssims = [r["ssim"] for r in quality_results if r["ssim"] is not None]
    if ssims:
        print(f"\n── Quality (2-demo, {len(ssims)} queries) ──")
        print(f"  SSIM (fresh vs cached): mean={np.mean(ssims):.4f} "
              f"min={min(ssims):.4f} max={max(ssims):.4f}")

    # Honest paper claim
    print(f"\n── Honest Paper Claim ──")
    if rope_fix_speedup > 2.0:
        print(f"  The {cached_speedup:.1f}x speedup is primarily ({rope_fix_speedup:.1f}x) due to")
        print(f"  avoiding a DynamicNTK RoPE recomputation pathology in the")
        print(f"  transformers library implementation.")
        if cached_speedup / rope_fix_speedup > 1.1:
            print(f"  The remaining {cached_speedup/rope_fix_speedup:.1f}x comes from")
            print(f"  prefill amortization.")
        print(f"  Claim: 'Prefix caching provides a practical workaround for a")
        print(f"  DynamicNTK generation pathology, yielding {cached_speedup:.1f}x speedup.'")
    elif rope_fix_speedup < 1.2:
        print(f"  RoPE recomputation is NOT the bottleneck.")
        print(f"  The {cached_speedup:.1f}x speedup comes from other factors")
        print(f"  (investigate: attention scaling, memory, kernel selection).")
    else:
        print(f"  Mixed: RoPE accounts for ~{rope_fix_speedup:.1f}x of {cached_speedup:.1f}x total.")

    # ─── Save Results ────────────────────────────────────

    all_results = {
        "experiment": "AR-5: Speedup Mechanism Decomposition",
        "phase1_2demo": {
            "A_original_dnk_fresh": {
                "prefill": prefill_t_a, "gen": gen_t_a, "total": total_a,
                "per_tok": per_tok_a, "n_tokens": len(tokens_a),
            },
            "B_precomputed_ntk_fresh": {
                "prefill": prefill_t_b, "gen": gen_t_b, "total": total_b,
                "per_tok": per_tok_b, "n_tokens": len(tokens_b),
            },
            "C_cached_original_dnk": {
                "query_prefill": query_t, "gen": gen_t_c,
                "amortized_total": cached_total,
                "per_tok": per_tok_c, "n_tokens": len(tokens_c),
            },
            "D_cached_precomputed_ntk": {
                "query_prefill": query_t_d, "gen": gen_t_d,
                "amortized_total": cached_total_d,
                "per_tok": per_tok_d, "n_tokens": len(tokens_d),
            },
        },
        "phase2_3demo": {
            "original_fresh": {"prefill": pf_3a, "gen": gen_3a, "total": pf_3a + gen_3a,
                               "per_tok": per_tok_3a, "n_tokens": len(tokens_3a)},
            "precomputed_fresh": {"prefill": pf_3b, "gen": gen_3b, "total": pf_3b + gen_3b,
                                  "per_tok": per_tok_3b, "n_tokens": len(tokens_3b)},
            "cached_original": {"query_prefill": qt_3, "gen": gen_3c,
                                "total": qt_3 + gen_3c,
                                "per_tok": per_tok_3c, "n_tokens": len(tokens_3c)},
        },
        "phase3_quality": quality_results,
        "phase4_attribution": {
            "total_speedup": cached_speedup,
            "rope_fix_speedup": rope_fix_speedup,
            "residual_speedup": cached_speedup / rope_fix_speedup if rope_fix_speedup > 0 else None,
        },
    }
    results_path = os.path.join(OUTPUT_ROOT, "ar5_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
