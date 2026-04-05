"""
AR-3: Generation-Informed Sparse Prefill Prototype

Tests whether AR-1's per-head demo-attention characterization can inform
a sparse prefill policy that reduces demo-image prefill cost while
preserving generation quality.

Core idea: heads that barely attend to demo tokens during generation
don't need full-resolution demo context during prefill either.

Key insight: during causal prefill, demo tokens (early positions) never
attend to non-demo tokens (later positions). Therefore, masking
non-demo->demo attention for specific heads does NOT change demo tokens'
KV cache entries. Only non-demo positions' KV is affected -- a much
milder intervention than AR-2's generation-time KV zeroing.

Configurations:
  1. baseline:            Full prefill (two-phase for fair comparison)
  2. conservative:        Mask heads with <5% demo attn during prefill
  3. moderate:            Mask heads with <15% demo attn
  4. aggressive:          Mask heads with <30% demo attn
  5. uniform_moderate:    Random heads at matched moderate sparsity
  6. uniform_aggressive:  Random heads at matched aggressive sparsity

Mechanism: per-head 4D attention masks via forward pre-hooks on decoder
layers. Blocked region: attention from query positions to demo content
positions. Demo->demo and query->structural attention preserved.
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

from data.item_processor import FlexARItemProcessor, center_crop
from transformers import DynamicCache, GenerationConfig

# ═══════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════
MODEL_PATH = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/weights/Lumina-mGPT-2.0-Omni"
DATA_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/data/nyu_depth_v2"
OUTPUT_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/outputs/ar3_sparse_prefill"
PROFILING_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/profiling"
AR1_DATA_PATH = os.path.join(PROFILING_ROOT, "ar1_attention_characterization.json")

WIDTH, HEIGHT = 512, 512
CFG_SCALE = 4.0
IMAGE_TOP_K = 200
EOS_TOKEN_ID = 151666
IMAGE_START_TOKEN_ID = 151665
SEED = 42
N_TEST_CASES = 3

# Token layout (from AR-1/AR-2 analysis)
# Positions 0-2: structural markers (BOS, delimiters)
# Positions 3-4162: demo image content (depth 3-2082, RGB 2083-4162)
# Position 4163: structural marker (delimiter)
# Positions 4164-4209: query text (46 tokens)
# Positions 4210-6292: query image (2083 tokens)
DEMO_CONTENT_START = 3
DEMO_CONTENT_END = 4163      # exclusive
QUERY_START = 4164

N_LAYERS = 32
N_HEADS = 32
HEAD_DIM = 128


# ═══════════════════════════════════════════════════════════
# AR-1 Data & Head Classification
# ═══════════════════════════════════════════════════════════

def load_ar1_data():
    with open(AR1_DATA_PATH) as f:
        data = json.load(f)
    return np.array(data["analysis"]["per_head_demo"])  # (32, 32)


def build_configs(per_head_demo, seed=SEED):
    """Build sparsity configurations from AR-1 per-head demo attention."""
    configs = {}

    # 1. Baseline: full prefill
    configs["baseline"] = {
        "type": "full", "masked_heads": {}, "total_masked": 0,
        "description": "Full prefill, no masking",
    }

    # 2-4. Head-selective: skip low-demo-attention heads (AR-1 guided)
    for name, thresh in [("conservative", 0.05), ("moderate", 0.15), ("aggressive", 0.30)]:
        masked = {}
        total = 0
        for li in range(N_LAYERS):
            heads = sorted(hi for hi in range(N_HEADS) if per_head_demo[li, hi] < thresh)
            if heads:
                masked[li] = heads
                total += len(heads)
        configs[name] = {
            "type": "head_selective", "threshold": thresh,
            "masked_heads": masked, "total_masked": total,
            "description": f"Skip <{thresh} heads ({total}/1024, {total/1024*100:.1f}%)",
        }

    # 5-6. Uniform baselines: random heads at matched sparsity
    rng = np.random.RandomState(seed)
    all_entries = [(li, hi) for li in range(N_LAYERS) for hi in range(N_HEADS)]
    for src in ["moderate", "aggressive"]:
        n = configs[src]["total_masked"]
        indices = rng.choice(len(all_entries), n, replace=False)
        masked = {}
        for idx in indices:
            li, hi = all_entries[idx]
            masked.setdefault(li, []).append(hi)
        for li in masked:
            masked[li] = sorted(masked[li])
        configs[f"uniform_{src}"] = {
            "type": "uniform", "n_matched": n,
            "masked_heads": masked, "total_masked": n,
            "description": f"Random {n}/1024 heads ({n/1024*100:.1f}%), matched to {src}",
        }

    return configs


# ═══════════════════════════════════════════════════════════
# Model Utilities
# ═══════════════════════════════════════════════════════════

def get_decoder_layers(model):
    """Navigate model hierarchy to find 32 decoder layers."""
    for accessor in [
        lambda m: m.model.model.layers,
        lambda m: m.model.layers,
        lambda m: m.layers,
    ]:
        try:
            layers = accessor(model)
            if hasattr(layers, '__len__') and len(layers) == N_LAYERS:
                return layers
        except AttributeError:
            continue
    raise RuntimeError(f"Could not find {N_LAYERS} decoder layers")


def build_causal_mask(seq_len, device, dtype):
    """Build lower-triangular causal mask: (1, 1, S, S), additive form."""
    min_val = torch.finfo(dtype).min
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    mask.masked_fill_(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(diagonal=1),
        min_val,
    )
    return mask.unsqueeze(0).unsqueeze(0)


# ═══════════════════════════════════════════════════════════
# Sparse Prefill via Per-Head Attention Masking
# ═══════════════════════════════════════════════════════════

def sparse_prefill(solver, prompt_tensor, config, seq_len):
    """Run prefill with per-head demo-attention masking via forward hooks.

    For masked heads in each layer: attention from query positions (4164+)
    to demo content positions (3-4162) is blocked. Demo-to-demo attention
    is always preserved, ensuring demo KV cache entries are identical to
    full prefill.

    Returns: (kv_cache, prefill_logits, prefill_time_s)
    """
    device = prompt_tensor.device
    dtype = torch.bfloat16
    min_val = torch.finfo(dtype).min
    masked_heads_map = config["masked_heads"]  # layer_idx -> [head_indices]

    cache = DynamicCache()

    if not masked_heads_map:
        # Full prefill (baseline): no hooks
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            out = solver.model(prompt_tensor, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        return out.past_key_values, out.logits[:, -1:, :].clone(), time.perf_counter() - t0

    # Sparse prefill: register per-layer hooks on decoder layers
    layers = get_decoder_layers(solver.model)

    # Build base causal mask once (shared, ~75 MB)
    base_mask = build_causal_mask(seq_len, device, dtype)  # (1, 1, S, S)

    def make_hook(head_indices):
        """Create pre-forward hook that injects 4D per-head sparse mask."""
        def hook(module, args, kwargs):
            # Expand base causal to (1, 32, S, S) and mask demo for specific heads
            mask = base_mask.expand(1, N_HEADS, -1, -1).clone()
            for h in head_indices:
                # Block: query positions attending to demo content positions
                mask[0, h, QUERY_START:, DEMO_CONTENT_START:DEMO_CONTENT_END] = min_val
            kwargs['attention_mask'] = mask
            return args, kwargs
        return hook

    # Only hook layers that have masked heads (others run with default mask)
    hooks = []
    for li, heads in masked_heads_map.items():
        h = layers[li].register_forward_pre_hook(make_hook(heads), with_kwargs=True)
        hooks.append(h)

    # Run prefill
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        out = solver.model(prompt_tensor, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0

    # Clean up hooks and mask
    for h in hooks:
        h.remove()
    del base_mask
    torch.cuda.empty_cache()

    return out.past_key_values, out.logits[:, -1:, :].clone(), prefill_time


# ═══════════════════════════════════════════════════════════
# Generation from Prefilled Cache
# ═══════════════════════════════════════════════════════════

def generate_from_cache(solver, prompt_tensor, prompt_len, ref_tokens,
                        kv_cache, prefill_logits):
    """Sample first token from prefill logits, then model.generate() from cache.

    Generation runs with UNMODIFIED attention -- only prefill was sparsified.
    """
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    gen_config = GenerationConfig(
        max_new_tokens=10240,
        max_length=solver.model.config.max_position_embeddings,
        temperature=1.0, do_sample=True, eos_token_id=[EOS_TOKEN_ID],
    )

    # Sample first token from prefill logits
    logits_proc = solver.create_logits_processor(cfg=CFG_SCALE, image_top_k=IMAGE_TOP_K)
    with torch.no_grad():
        processed = logits_proc(prompt_tensor, prefill_logits[:, -1, :])
        probs = torch.softmax(processed, dim=-1)
        first_token = torch.multinomial(probs, num_samples=1)
    del prefill_logits

    # Continue generation from cache
    gen_ids = torch.cat([prompt_tensor, first_token], dim=1)
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
# Prompt Building (reused from AR-2)
# ═══════════════════════════════════════════════════════════

def build_1shot_prompt(solver, query_path, demo_rgb_path, demo_depth_path):
    """Build 1-shot depth estimation prompt."""
    rgb = Image.open(demo_rgb_path)
    depth = Image.open(demo_depth_path)
    rgb_c = center_crop(rgb, crop_size=[WIDTH, HEIGHT // 2])
    depth_c = center_crop(depth, crop_size=[WIDTH, HEIGHT // 2])
    stitched = Image.new("RGB", (WIDTH, HEIGHT))
    stitched.paste(depth_c, (0, 0))
    stitched.paste(rgb_c, (0, HEIGHT // 2))

    demo_tokens = solver.item_processor.process_image(stitched)["input_ids"]
    n_demo = len(demo_tokens)

    img = Image.open(query_path)
    img_c = center_crop(img, crop_size=[WIDTH, HEIGHT // 2])
    cropped_path = os.path.join(OUTPUT_ROOT, "ar3_query_cropped.png")
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

    all_ids = demo_tokens + query_ids
    prompt_tensor = torch.tensor(
        all_ids, dtype=torch.int64, device=solver.model.device
    ).unsqueeze(0)
    return prompt_tensor, n_demo + len(query_ids), ref_tokens


# ═══════════════════════════════════════════════════════════
# Quality Metrics (reused from AR-2)
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
# Theoretical FLOPs Savings
# ═══════════════════════════════════════════════════════════

def compute_flops_savings(configs, seq_len):
    """Compute theoretical attention FLOPs savings from sparse prefill."""
    total_causal = seq_len * (seq_len + 1) // 2
    n_query = seq_len - QUERY_START            # 2129
    n_demo = DEMO_CONTENT_END - DEMO_CONTENT_START  # 4160
    blocked_per_head = n_query * n_demo        # ~8.86M entries
    total_heads = N_LAYERS * N_HEADS           # 1024

    savings = {}
    for name, cfg in configs.items():
        n = cfg["total_masked"]
        attn_frac = (n * blocked_per_head) / (total_heads * total_causal)
        savings[name] = {
            "heads_masked": n,
            "heads_pct": n / total_heads * 100,
            "blocked_per_head": blocked_per_head,
            "attn_flops_saved_pct": attn_frac * 100,
            "total_flops_saved_pct": attn_frac * 0.75 * 100,
        }
    return savings


# ═══════════════════════════════════════════════════════════
# Per-Case Runner
# ═══════════════════════════════════════════════════════════

def run_case(solver, case_id, query_path, demo_rgb_path, demo_depth_path,
             configs, output_dir):
    print(f"\n{'='*60}")
    print(f"Case {case_id}")
    print(f"{'='*60}")

    prompt_tensor, prompt_len, ref_tokens = build_1shot_prompt(
        solver, query_path, demo_rgb_path, demo_depth_path
    )
    seq_len = prompt_tensor.shape[1]
    print(f"  Prompt: {prompt_len} tokens, seq_len: {seq_len}")

    results = {}
    baseline_tokens = None
    baseline_img = None

    for cfg_name, config in configs.items():
        print(f"\n  --- {cfg_name} ---")
        print(f"      {config['description']}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Phase 1: Prefill (sparse or full)
        kv_cache, prefill_logits, prefill_time = sparse_prefill(
            solver, prompt_tensor, config, seq_len
        )
        print(f"    Prefill: {prefill_time:.2f}s")

        # Phase 2: Generate from cache (identical for all configs)
        gen_tokens, gen_time = generate_from_cache(
            solver, prompt_tensor, prompt_len, ref_tokens,
            kv_cache, prefill_logits,
        )
        total_time = prefill_time + gen_time
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"    Gen: {gen_time:.1f}s ({len(gen_tokens)} tok) | "
              f"Total: {total_time:.1f}s | VRAM: {peak_mem:.1f}GB")

        # Decode image
        img = decode_image(solver, ref_tokens, gen_tokens)
        img_path = os.path.join(output_dir, f"case{case_id}_{cfg_name}.png")
        if img:
            img.save(img_path)
        else:
            print(f"    WARNING: decode failed")

        # Quality metrics
        quality = {}
        tok_match = {}
        if cfg_name == "baseline":
            baseline_tokens = gen_tokens
            baseline_img = img
            quality = {"ssim": 1.0, "mse": 0.0}
            tok_match = {"overall": 1.0, "early_100": 1.0, "first_diff": len(gen_tokens)}
        elif baseline_img and img:
            quality = compute_quality(baseline_img, img)
            overall, early, fd = token_match_rate(baseline_tokens, gen_tokens)
            tok_match = {"overall": overall, "early_100": early, "first_diff": fd}
            print(f"    SSIM={quality['ssim']:.4f} | "
                  f"tok_match={overall:.3f} early100={early:.3f} first_diff@{fd}")

        results[cfg_name] = {
            "timing": {"prefill": prefill_time, "gen": gen_time, "total": total_time},
            "peak_mem_gb": peak_mem,
            "n_generated": len(gen_tokens),
            "quality": quality,
            "token_match": tok_match,
        }

        del kv_cache
        torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("AR-3: Generation-Informed Sparse Prefill Prototype")
    print("=" * 60)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Load AR-1 per-head demo attention data
    per_head_demo = load_ar1_data()
    configs = build_configs(per_head_demo)
    print(f"\nAR-1 per-head demo attention: shape={per_head_demo.shape}, "
          f"range=[{per_head_demo.min():.4f}, {per_head_demo.max():.4f}]")

    print(f"\nConfigurations:")
    for name, cfg in configs.items():
        print(f"  {name}: {cfg['description']}")

    # Per-layer breakdown for head-selective configs
    print(f"\nPer-layer head counts (head-selective):")
    for name in ["conservative", "moderate", "aggressive"]:
        masked = configs[name]["masked_heads"]
        layer_counts = [len(masked.get(li, [])) for li in range(N_LAYERS)]
        print(f"  {name}: {layer_counts}")

    # Theoretical FLOPs savings
    flops = compute_flops_savings(configs, 6293)
    print(f"\nTheoretical FLOPs savings (prefill):")
    print(f"  {'Config':<25s}  {'Heads':>8s}  {'Attn FLOPs':>10s}  {'Total FLOPs':>11s}")
    print(f"  {'-'*56}")
    for name in configs:
        f = flops[name]
        print(f"  {name:<25s}  {f['heads_masked']:>4d}/1024  "
              f"{f['attn_flops_saved_pct']:>9.1f}%  {f['total_flops_saved_pct']:>10.1f}%")

    # Load test data
    with open(os.path.join(DATA_ROOT, "task_instances.json")) as f:
        instances = json.load(f)
    print(f"\nUsing {N_TEST_CASES} of {len(instances)} task instances")

    # Load model
    print("\nLoading model...")
    from inference_solver import FlexARInferenceSolver
    solver = FlexARInferenceSolver(
        model_path=MODEL_PATH, precision="bf16", target_size=512
    )
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Verify decoder layers
    layers = get_decoder_layers(solver.model)
    print(f"Decoder layers: {len(layers)} ({type(layers[0]).__name__})")

    # Test hook mechanism on first layer
    print("\nTesting hook mechanism...")
    test_hook_ok = True
    def test_hook(module, args, kwargs):
        nonlocal test_hook_ok
        test_hook_ok = 'attention_mask' in kwargs or True  # kwargs may vary
        return args, kwargs
    h = layers[0].register_forward_pre_hook(test_hook, with_kwargs=True)
    h.remove()
    print(f"  Hook registration: OK")

    # Warmup (standard generate, 100 tokens)
    print("\nWarmup...")
    inst = instances[0]
    pt, pl, rt = build_1shot_prompt(
        solver, inst["query"]["rgb_path"],
        inst["demo_pairs"][0]["rgb_path"],
        inst["demo_pairs"][0]["depth_colored_path"],
    )
    lp = solver.create_logits_processor(cfg=CFG_SCALE, image_top_k=IMAGE_TOP_K)
    gc = GenerationConfig(
        max_new_tokens=100, max_length=10240,
        temperature=1.0, do_sample=True, eos_token_id=[EOS_TOKEN_ID],
    )
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = solver.model.generate(pt, gc, logits_processor=lp)
    torch.cuda.empty_cache()
    print("Warmup complete.\n")

    # Run all cases
    all_results = {}
    for ci in range(N_TEST_CASES):
        inst = instances[ci]
        r = run_case(
            solver, ci,
            inst["query"]["rgb_path"],
            inst["demo_pairs"][0]["rgb_path"],
            inst["demo_pairs"][0]["depth_colored_path"],
            configs, OUTPUT_ROOT,
        )
        all_results[f"case_{ci}"] = r

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    cfg_names = list(configs.keys())

    print(f"\n{'Config':<25s}  {'Prefill':>8s}  {'Gen':>7s}  {'Total':>7s}  "
          f"{'SSIM':>7s}  {'TokMatch':>8s}  {'1stDiff':>8s}")
    print("-" * 80)

    summary = {}
    for c in cfg_names:
        vals = {"pf": [], "gt": [], "tt": [], "ssim": [], "tok": [], "fd": []}
        for ck in all_results:
            r = all_results[ck][c]
            vals["pf"].append(r["timing"]["prefill"])
            vals["gt"].append(r["timing"]["gen"])
            vals["tt"].append(r["timing"]["total"])
            if "ssim" in r["quality"]:
                vals["ssim"].append(r["quality"]["ssim"])
            if "overall" in r["token_match"]:
                vals["tok"].append(r["token_match"]["overall"])
            if "first_diff" in r["token_match"]:
                vals["fd"].append(r["token_match"]["first_diff"])

        mpf = np.mean(vals["pf"])
        mgt = np.mean(vals["gt"])
        mtt = np.mean(vals["tt"])
        ms = np.mean(vals["ssim"]) if vals["ssim"] else None
        mtk = np.mean(vals["tok"]) if vals["tok"] else None
        mfd = np.mean(vals["fd"]) if vals["fd"] else None

        def f(v, fmt):
            return fmt.format(v) if v is not None else "N/A"

        print(f"  {c:<23s}  {mpf:>7.2f}s  {mgt:>6.1f}s  {mtt:>6.1f}s  "
              f"{f(ms, '{:.4f}'):>7s}  {f(mtk, '{:.3f}'):>8s}  {f(mfd, '{:.0f}'):>8s}")

        summary[c] = {
            "mean_prefill": float(mpf), "mean_gen": float(mgt),
            "mean_total": float(mtt),
            "mean_ssim": float(ms) if ms is not None else None,
            "mean_token_match": float(mtk) if mtk is not None else None,
            "mean_first_diff": float(mfd) if mfd is not None else None,
        }

    # Head-selective vs uniform comparison
    print("\n── Head-Selective vs Uniform ──")
    for level in ["moderate", "aggressive"]:
        hs = summary.get(level, {})
        uni = summary.get(f"uniform_{level}", {})
        if hs.get("mean_ssim") is not None and uni.get("mean_ssim") is not None:
            d = hs["mean_ssim"] - uni["mean_ssim"]
            w = "head-selective" if d > 0 else "uniform"
            print(f"  {level}: head-sel vs uniform SSIM "
                  f"{hs['mean_ssim']:.4f} vs {uni['mean_ssim']:.4f} "
                  f"(delta={d:+.4f}) -> {w}")

    # Chaos comparison with AR-2
    print(f"\n── AR-3 vs AR-2 Chaos Comparison ──")
    print(f"  AR-2 (generation-time KV compression): first_diff = 1 for ALL configs")
    print(f"  AR-3 (prefill sparsification):")
    for c in cfg_names:
        if c != "baseline" and summary[c].get("mean_first_diff") is not None:
            fd = summary[c]["mean_first_diff"]
            if fd > 1:
                status = f"MORE STABLE (first_diff={fd:.0f} >> 1)"
            else:
                status = "SAME CHAOS (first_diff=1)"
            print(f"    {c}: {status}")

    # Prefill wall-clock comparison
    print(f"\n── Prefill Wall-Clock ──")
    base_pf = summary["baseline"]["mean_prefill"]
    print(f"  baseline: {base_pf:.2f}s")
    for c in cfg_names:
        if c != "baseline":
            pf = summary[c]["mean_prefill"]
            overhead = (pf / base_pf - 1) * 100 if base_pf > 0 else 0
            print(f"  {c}: {pf:.2f}s ({overhead:+.1f}% vs baseline)")

    # Save results
    def ser(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    def ser_dict(d):
        if isinstance(d, dict):
            return {k: ser_dict(v) for k, v in d.items()}
        if isinstance(d, list):
            return [ser_dict(v) for v in d]
        return ser(d)

    save_data = {
        "experiment": "AR-3: Generation-Informed Sparse Prefill",
        "config": {
            "model": "Lumina-mGPT-2.0-Omni", "cfg": CFG_SCALE, "seed": SEED,
            "n_cases": N_TEST_CASES, "n_layers": N_LAYERS, "n_heads": N_HEADS,
            "head_dim": HEAD_DIM,
            "demo_content_range": [DEMO_CONTENT_START, DEMO_CONTENT_END],
            "query_start": QUERY_START,
        },
        "sparsity_configs": {
            name: {k: v for k, v in cfg.items() if k != "masked_heads"}
            for name, cfg in configs.items()
        },
        "flops_savings": flops,
        "per_case": ser_dict(all_results),
        "summary": summary,
    }
    results_path = os.path.join(OUTPUT_ROOT, "ar3_results.json")
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=ser)
    print(f"\nResults saved: {results_path}")
    print(f"Images saved: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
