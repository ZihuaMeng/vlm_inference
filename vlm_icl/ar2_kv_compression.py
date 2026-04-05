"""
AR-2: Head-Selective Demo-KV Compression Prototype

Tests whether training-free, per-head demo-KV eviction guided by AR-1
attention characterization preserves generation quality while reducing
KV cache size.

Configurations:
  1. baseline:           No compression
  2. head_selective_25:  Top 25% heads by demo-attention retain demo KV
  3. head_selective_50:  Top 50% heads retain
  4. head_selective_75:  Top 75% heads retain
  5. uniform_50:         All heads, random 50% of demo tokens retained
  6. substructure_50:    All heads, keep demo output-image, drop input-image

Mechanism: zero out K and V at demo content positions for evicted heads
via CompressingDynamicCache that hooks into model.generate()'s prefill.
Structural marker tokens (4 tokens, ~5% attention) always preserved.
"""

import sys
import os
import json
import time
import numpy as np
import torch
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = "/home/zihua.meng/projects/vlm_inference/vlm_icl/Lumina-mGPT-2.0"
sys.path.insert(0, os.path.join(REPO_ROOT, "lumina_mgpt"))
sys.path.insert(0, REPO_ROOT)
os.chdir(os.path.join(REPO_ROOT, "lumina_mgpt"))

from data.item_processor import FlexARItemProcessor, center_crop
from transformers import DynamicCache, GenerationConfig

MODEL_PATH = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/weights/Lumina-mGPT-2.0-Omni"
DATA_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/data/nyu_depth_v2"
OUTPUT_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/outputs/ar2_kv_compression"
PROFILING_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/profiling"
AR1_DATA_PATH = os.path.join(PROFILING_ROOT, "ar1_attention_characterization.json")

WIDTH, HEIGHT = 512, 512
CFG_SCALE = 4.0
IMAGE_TOP_K = 200
EOS_TOKEN_ID = 151666
IMAGE_START_TOKEN_ID = 151665
SEED = 42
N_TEST_CASES = 3


# ═══════════════════════════════════════════════════════════
# KV Cache Compression
# ═══════════════════════════════════════════════════════════

def compress_kv_cache(cache, regions, policy, seed=42):
    """Zero out K/V at demo content positions based on policy."""
    if policy is None:
        return 0

    n_layers = len(cache.key_cache)
    demo_start = regions["demo_content_start"]
    demo_end = regions["demo_content_end"]
    n_demo = demo_end - demo_start
    device = cache.key_cache[0].device
    zeroed = 0

    if policy["type"] == "head_selective":
        mask = policy["retain_mask"]  # (32, 32) bool
        for li in range(n_layers):
            k = cache.key_cache[li]
            v = cache.value_cache[li]
            for hi in range(k.shape[1]):
                if not mask[li, hi]:
                    k[0, hi, demo_start:demo_end, :] = 0
                    v[0, hi, demo_start:demo_end, :] = 0
                    zeroed += n_demo

    elif policy["type"] == "uniform":
        rng = np.random.RandomState(seed)
        positions = np.arange(demo_start, demo_end)
        drop = torch.tensor(
            rng.choice(positions, n_demo // 2, replace=False),
            dtype=torch.long, device=device,
        )
        for li in range(n_layers):
            cache.key_cache[li][0, :, drop, :] = 0
            cache.value_cache[li][0, :, drop, :] = 0
        zeroed = (n_demo // 2) * cache.key_cache[0].shape[1] * n_layers

    elif policy["type"] == "substructure":
        rs = regions["demo_rgb_start"]
        re = regions["demo_rgb_end"]
        n_rgb = re - rs
        for li in range(n_layers):
            cache.key_cache[li][0, :, rs:re, :] = 0
            cache.value_cache[li][0, :, rs:re, :] = 0
        zeroed = n_rgb * cache.key_cache[0].shape[1] * n_layers

    return zeroed


# ═══════════════════════════════════════════════════════════
# Retention Policy
# ═══════════════════════════════════════════════════════════

def load_retention_policy():
    data = json.load(open(AR1_DATA_PATH))
    per_head_demo = np.array(data["analysis"]["per_head_demo"])  # (32, 32)
    flat = per_head_demo.flatten()
    p25, p50, p75 = [float(np.percentile(flat, p)) for p in (25, 50, 75)]

    policies = {
        "baseline": None,
        "head_selective_25": {
            "type": "head_selective",
            "retain_mask": per_head_demo >= p75,
            "threshold": p75, "retain_rate": 0.25,
        },
        "head_selective_50": {
            "type": "head_selective",
            "retain_mask": per_head_demo >= p50,
            "threshold": p50, "retain_rate": 0.50,
        },
        "head_selective_75": {
            "type": "head_selective",
            "retain_mask": per_head_demo >= p25,
            "threshold": p25, "retain_rate": 0.75,
        },
        "uniform_50": {"type": "uniform", "retain_rate": 0.50},
        "substructure_50": {"type": "substructure", "retain_rate": 0.50},
    }
    return policies, per_head_demo


# ═══════════════════════════════════════════════════════════
# Prompt Building
# ═══════════════════════════════════════════════════════════

def build_1shot_prompt(solver, query_path, demo_rgb_path, demo_depth_path):
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
    cropped_path = os.path.join(OUTPUT_ROOT, "ar2_query_cropped.png")
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

    header_end = 3
    tokens_per_row = 65
    depth_end = header_end + 32 * tokens_per_row   # 2083
    rgb_end = depth_end + 32 * tokens_per_row       # 4163

    regions = {
        "demo_depth_start": header_end,
        "demo_depth_end": depth_end,
        "demo_rgb_start": depth_end,
        "demo_rgb_end": rgb_end,
        "demo_content_start": header_end,
        "demo_content_end": rgb_end,
        "structural_positions": [0, 1, 2, rgb_end],
        "n_demo": n_demo,
    }

    return prompt_tensor, n_demo + len(query_ids), ref_tokens, regions


# ═══════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════

def generate_with_compression(solver, prompt_tensor, prompt_len, ref_tokens,
                               policy, regions, cfg_scale=4.0, seed=42):
    """Generate using model.generate() with optional KV compression.

    Baseline (policy=None): standard model.generate() without explicit cache.
    Compressed: two-phase approach — prefill manually, compress cache, then
    continue generation with model.generate() passing the compressed cache.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    logits_proc = solver.create_logits_processor(cfg=cfg_scale, image_top_k=IMAGE_TOP_K)
    gen_config = GenerationConfig(
        max_new_tokens=10240,
        max_length=solver.model.config.max_position_embeddings,
        temperature=1.0, do_sample=True, eos_token_id=[EOS_TOKEN_ID],
    )

    if policy is None:
        # Baseline: standard model.generate(), no cache intervention
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = solver.model.generate(
                prompt_tensor, gen_config,
                logits_processor=logits_proc,
            )[0][prompt_len:].tolist()
        torch.cuda.synchronize()
        total_time = time.perf_counter() - t0
        return result, {"total_time": total_time, "compress_time": 0.0,
                        "n_generated": len(result)}

    # Compressed: prefill → compress → continue generation
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Phase 1: Prefill to get KV cache
    cache = DynamicCache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = solver.model(prompt_tensor, past_key_values=cache, use_cache=True)
    cache = outputs.past_key_values
    prefill_logits = outputs.logits[:, -1:, :].clone()
    del outputs

    torch.cuda.synchronize()
    t_prefill = time.perf_counter()

    # Phase 2: Compress demo KV
    compress_kv_cache(cache, regions, policy, seed=seed)
    torch.cuda.synchronize()
    t_compress = time.perf_counter()
    compress_time = t_compress - t_prefill

    # Phase 3: Sample first token (no CFG for first token — prompt is balanced)
    with torch.no_grad():
        processed = logits_proc(prompt_tensor, prefill_logits[:, -1, :])
        probs = torch.softmax(processed, dim=-1)
        first_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
    del prefill_logits

    # Phase 4: Continue with model.generate() from first_token + compressed cache
    # Build input_ids that includes prompt + first_token so the logits processor
    # can correctly detect image_start/end token balance for CFG
    gen_input_ids = torch.cat([prompt_tensor, first_token], dim=1)

    # Fresh logits processor for model.generate() (the first one is consumed)
    logits_proc2 = solver.create_logits_processor(cfg=cfg_scale, image_top_k=IMAGE_TOP_K)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result = solver.model.generate(
            gen_input_ids, gen_config,
            logits_processor=logits_proc2,
            past_key_values=cache,
        )[0][prompt_len:].tolist()

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    return result, {
        "total_time": total_time,
        "compress_time": compress_time,
        "n_generated": len(result),
    }


# ═══════════════════════════════════════════════════════════
# Quality Metrics
# ═══════════════════════════════════════════════════════════

def decode_image(solver, ref_tokens, gen_tokens):
    if ref_tokens is not None:
        decode_tokens = list(ref_tokens) + list(gen_tokens)
    else:
        decode_tokens = list(gen_tokens)
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
# Theoretical Savings
# ═══════════════════════════════════════════════════════════

def compute_theoretical_savings(regions, policies, prompt_len,
                                 n_heads=32, n_layers=32, head_dim=128):
    n_demo = regions["demo_content_end"] - regions["demo_content_start"]
    baseline_kv = 2 * n_heads * prompt_len * head_dim * 2 * n_layers  # bytes

    savings = {}
    for name, pol in policies.items():
        if pol is None:
            savings[name] = {"kv_pct": 0.0, "kv_mb": 0.0, "attn_pct": 0.0}
            continue
        if pol["type"] == "head_selective":
            evicted = n_heads * n_layers * (1 - pol["retain_rate"]) * n_demo
        else:
            evicted = n_heads * n_layers * n_demo * (1 - pol["retain_rate"])
        total = n_heads * n_layers * prompt_len
        pct = evicted / total * 100
        savings[name] = {
            "kv_pct": pct,
            "kv_mb": pct / 100 * baseline_kv / 1e6,
            "attn_pct": pct,
        }
    return savings, baseline_kv / 1e6


# ═══════════════════════════════════════════════════════════
# Per-Case Runner
# ═══════════════════════════════════════════════════════════

def run_case(solver, case_id, query_path, demo_rgb_path, demo_depth_path,
             policies, output_dir):
    print(f"\n{'='*60}")
    print(f"Case {case_id}")
    print(f"{'='*60}")

    prompt_tensor, prompt_len, ref_tokens, regions = build_1shot_prompt(
        solver, query_path, demo_rgb_path, demo_depth_path
    )
    n_demo = regions["demo_content_end"] - regions["demo_content_start"]
    print(f"  Prompt: {prompt_len} tokens, demo content: {n_demo} tokens")

    results = {}
    baseline_tokens = None
    baseline_img = None

    for cfg_name, policy in policies.items():
        print(f"\n  --- {cfg_name} ---")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        gen_tokens, timing = generate_with_compression(
            solver, prompt_tensor, prompt_len, ref_tokens,
            policy, regions, cfg_scale=CFG_SCALE, seed=SEED,
        )
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        print(f"    {timing['n_generated']} tokens | "
              f"total={timing['total_time']:.1f}s "
              f"compress={timing['compress_time']:.4f}s | "
              f"VRAM={peak_mem:.1f}GB")

        # Decode
        img_path = os.path.join(output_dir, f"case{case_id}_{cfg_name}.png")
        img = decode_image(solver, ref_tokens, gen_tokens)
        if img:
            img.save(img_path)
            print(f"    Saved: {img_path}")
        else:
            print(f"    WARNING: decode failed")

        # Quality
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
            "timing": timing,
            "peak_mem_gb": peak_mem,
            "quality": quality,
            "token_match": tok_match,
        }

    return results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("AR-2: Head-Selective Demo-KV Compression Prototype")
    print("=" * 60)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    policies, per_head_demo = load_retention_policy()
    print(f"\nRetention policies:")
    for name, pol in policies.items():
        if pol is None:
            print(f"  {name}: 100% retention")
        elif pol["type"] == "head_selective":
            n = int(pol["retain_mask"].sum())
            print(f"  {name}: thresh={pol['threshold']:.4f}, {n}/1024 heads ({n/1024*100:.1f}%)")
        else:
            print(f"  {name}: {pol['type']}, {pol['retain_rate']*100:.0f}% retain")

    with open(os.path.join(DATA_ROOT, "task_instances.json")) as f:
        instances = json.load(f)
    print(f"\nUsing {N_TEST_CASES} of {len(instances)} task instances")

    print("\nLoading model...")
    from inference_solver import FlexARInferenceSolver
    solver = FlexARInferenceSolver(
        model_path=MODEL_PATH, precision="bf16", target_size=512
    )
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Warmup (standard model.generate, 100 tokens)
    print("\nWarmup...")
    inst = instances[0]
    pt, pl, rt, rg = build_1shot_prompt(
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
            policies, OUTPUT_ROOT,
        )
        all_results[f"case_{ci}"] = r

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    config_names = list(policies.keys())

    # Theoretical savings
    _, pl0, _, rg0 = build_1shot_prompt(
        solver, instances[0]["query"]["rgb_path"],
        instances[0]["demo_pairs"][0]["rgb_path"],
        instances[0]["demo_pairs"][0]["depth_colored_path"],
    )
    theory, baseline_kv_mb = compute_theoretical_savings(rg0, policies, pl0)

    print(f"\nBaseline KV: {baseline_kv_mb:.1f} MB")
    print(f"\n{'Config':<25s}  {'KV ↓':>7s}  {'Saved':>8s}  {'Attn ↓':>7s}")
    print("-" * 55)
    for c in config_names:
        s = theory[c]
        print(f"  {c:<23s}  {s['kv_pct']:>6.1f}%  {s['kv_mb']:>7.1f}MB  {s['attn_pct']:>6.1f}%")

    # Quality table
    print(f"\n{'Config':<25s}  {'Time':>7s}  {'SSIM':>7s}  {'MSE':>8s}  "
          f"{'TokMatch':>8s}  {'Early100':>8s}  {'1stDiff':>8s}")
    print("-" * 85)

    summary = {}
    baseline_times = [all_results[ck]["baseline"]["timing"]["total_time"]
                      for ck in all_results]
    mean_bt = np.mean(baseline_times)

    for c in config_names:
        vals = {"time": [], "ssim": [], "mse": [], "tok": [], "early": [], "fd": []}
        for ck in all_results:
            r = all_results[ck][c]
            vals["time"].append(r["timing"]["total_time"])
            if "ssim" in r["quality"]:
                vals["ssim"].append(r["quality"]["ssim"])
            if "mse" in r["quality"]:
                vals["mse"].append(r["quality"]["mse"])
            if "overall" in r["token_match"]:
                vals["tok"].append(r["token_match"]["overall"])
            if "early_100" in r["token_match"]:
                vals["early"].append(r["token_match"]["early_100"])
            if "first_diff" in r["token_match"]:
                vals["fd"].append(r["token_match"]["first_diff"])

        mt = np.mean(vals["time"])
        ms = np.mean(vals["ssim"]) if vals["ssim"] else None
        mm = np.mean(vals["mse"]) if vals["mse"] else None
        mtk = np.mean(vals["tok"]) if vals["tok"] else None
        me = np.mean(vals["early"]) if vals["early"] else None
        mf = np.mean(vals["fd"]) if vals["fd"] else None

        def f(v, fmt): return fmt.format(v) if v is not None else "N/A"
        print(f"  {c:<23s}  {mt:>6.1f}s  {f(ms,'{:.4f}'):>7s}  {f(mm,'{:.0f}'):>8s}  "
              f"{f(mtk,'{:.3f}'):>8s}  {f(me,'{:.3f}'):>8s}  {f(mf,'{:.0f}'):>8s}")

        summary[c] = {
            "mean_time": float(mt),
            "mean_ssim": float(ms) if ms is not None else None,
            "mean_mse": float(mm) if mm is not None else None,
            "mean_token_match": float(mtk) if mtk is not None else None,
            "mean_early_100": float(me) if me is not None else None,
            "mean_first_diff": float(mf) if mf is not None else None,
        }

    # Key comparisons
    hs50 = summary.get("head_selective_50", {})
    u50 = summary.get("uniform_50", {})
    if hs50.get("mean_ssim") and u50.get("mean_ssim"):
        d = hs50["mean_ssim"] - u50["mean_ssim"]
        w = "head-selective" if d > 0 else "uniform"
        print(f"\n  Head-selective vs Uniform @50%: delta={d:+.4f} → {w} wins")

    sub50 = summary.get("substructure_50", {})
    if sub50.get("mean_ssim") and u50.get("mean_ssim"):
        d = sub50["mean_ssim"] - u50["mean_ssim"]
        w = "substructure" if d > 0 else "uniform"
        print(f"  Substructure vs Uniform @50%:   delta={d:+.4f} → {w} wins")

    # Save
    def ser(o):
        if isinstance(o, (np.floating, np.float32, np.float64)): return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)): return int(o)
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    save_data = {
        "config": {"model": "Lumina-mGPT-2.0-Omni", "cfg": CFG_SCALE, "seed": SEED,
                    "n_cases": N_TEST_CASES, "n_layers": 32, "n_heads": 32, "head_dim": 128},
        "policies": {
            n: {"type": p["type"] if p else "none",
                "retain_rate": p.get("retain_rate", 1.0) if p else 1.0,
                "threshold": p.get("threshold") if p else None}
            for n, p in policies.items()
        },
        "theoretical_savings": theory,
        "baseline_kv_mb": baseline_kv_mb,
        "per_case": {
            ck: {
                cn: {"timing": r["timing"], "peak_mem_gb": r["peak_mem_gb"],
                      "quality": {k: ser(v) for k, v in r["quality"].items()},
                      "token_match": {k: ser(v) for k, v in r["token_match"].items()}}
                for cn, r in cd.items()
            }
            for ck, cd in all_results.items()
        },
        "summary": summary,
    }
    results_path = os.path.join(OUTPUT_ROOT, "ar2_results.json")
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=ser)
    print(f"\nResults saved: {results_path}")
    print(f"Images saved: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
