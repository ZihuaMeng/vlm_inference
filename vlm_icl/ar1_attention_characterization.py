"""
AR-1: Attention Characterization During AR Image Generation

Characterizes how generated image tokens attend to demonstration context
during autoregressive generation with 1-shot in-context learning on
Lumina-mGPT 2.0 (Chameleon-based, 7B, 32 layers, 32 heads).

Approach:
  Phase A: Generate tokens using standard pipeline (model.generate with CFG)
  Phase B: Replay generation token-by-token, logging attention at sampled steps
           (output_attentions=True triggers SDPA→eager fallback for those steps)

Token regions tracked:
  - demo_depth: Upper half of demo image (output/depth map), 2080 tokens
  - demo_rgb:   Lower half of demo image (input/RGB), 2080 tokens
  - demo_other: Demo header + footer, 4 tokens
  - query_text: Text instruction tokens, 46 tokens
  - query_img:  Query image tokens, ~2083 tokens
  - generated:  Previously generated tokens, grows during generation

Scientific questions:
  a. What fraction of attention goes to demo vs query vs generated tokens?
  b. Is demo attention concentrated in specific layers or heads?
  c. Does demo attention change during generation (early/mid/late)?
  d. Within demo, is attention on input image vs output image?
  e. Are there "demo-reading" heads that specialize in demo attention?
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = "/home/zihua.meng/projects/vlm_inference/vlm_icl/Lumina-mGPT-2.0"
sys.path.insert(0, os.path.join(REPO_ROOT, "lumina_mgpt"))
sys.path.insert(0, REPO_ROOT)
os.chdir(os.path.join(REPO_ROOT, "lumina_mgpt"))

import torch
from PIL import Image
from data.item_processor import FlexARItemProcessor, center_crop

MODEL_PATH = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/weights/Lumina-mGPT-2.0-Omni"
DATA_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/data/nyu_depth_v2"
OUTPUT_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/outputs"
PROFILING_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/profiling"

SAMPLE_EVERY = 20  # Log attention every 20th generated token (~104 samples)
WIDTH, HEIGHT = 512, 512


def build_1shot_prompt(solver, query_path, demo_rgb_path, demo_depth_path):
    """Build 1-shot prompt and return tensor + token region boundaries."""
    # Build demo image: depth on top, RGB on bottom (512x512)
    rgb = Image.open(demo_rgb_path)
    depth = Image.open(demo_depth_path)
    rgb_cropped = center_crop(rgb, crop_size=[WIDTH, HEIGHT // 2])
    depth_cropped = center_crop(depth, crop_size=[WIDTH, HEIGHT // 2])
    stitched = Image.new("RGB", (WIDTH, HEIGHT))
    stitched.paste(depth_cropped, (0, 0))
    stitched.paste(rgb_cropped, (0, HEIGHT // 2))

    # Tokenize demo as full image
    demo_result = solver.item_processor.process_image(stitched)
    demo_tokens = demo_result["input_ids"]
    n_demo = len(demo_tokens)

    # Build query prompt
    img = Image.open(query_path)
    img_cropped = center_crop(img, crop_size=[WIDTH, HEIGHT // 2])
    cropped_path = os.path.join(OUTPUT_ROOT, "ar1_query_cropped.png")
    img_cropped.save(cropped_path)

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

    if 151665 in _prompt:
        _prompt.pop()
        start_index = _prompt.index(151665)
        ref_tokens = _prompt[start_index:]
    else:
        ref_tokens = None

    query_ids = []
    for value in _prompt:
        if isinstance(value, int):
            query_ids.append(value)
        else:
            query_ids += value["input_ids"]
    n_query = len(query_ids)

    # Combine: demo + query
    all_ids = demo_tokens + query_ids
    prompt_tensor = torch.tensor(
        all_ids, dtype=torch.int64, device=solver.model.device
    ).unsqueeze(0)

    # Token regions (absolute positions in full sequence)
    # Demo: [start_tok, h_grid, w_grid] + [32 rows depth × 65] + [32 rows RGB × 65] + [end_tok]
    header_end = 3
    half_rows = 32
    tokens_per_row = 65  # 64 VQ + 1 newline
    depth_end = header_end + half_rows * tokens_per_row  # 2083
    rgb_end = depth_end + half_rows * tokens_per_row  # 4163

    # Find query text/image boundary
    ip = solver.item_processor
    img_start_id = ip.token2id(ip.image_start_token)
    query_img_starts = [
        i for i, t in enumerate(query_ids) if t == img_start_id
    ]
    text_end_in_query = query_img_starts[0] if query_img_starts else n_query

    regions = {
        "demo_depth": (header_end, depth_end),  # 2080 tokens
        "demo_rgb": (depth_end, rgb_end),  # 2080 tokens
        "demo_other": (0, header_end, rgb_end, n_demo),  # header + footer (4 tokens)
        "query_text": (n_demo, n_demo + text_end_in_query),  # ~46 tokens
        "query_img": (n_demo + text_end_in_query, n_demo + n_query),  # ~2083 tokens
    }
    prompt_len = n_demo + n_query

    return prompt_tensor, prompt_len, ref_tokens, regions


def generate_tokens(solver, prompt_tensor, prompt_len):
    """Phase A: Generate tokens using standard pipeline with CFG."""
    from transformers import GenerationConfig

    logits_processor = solver.create_logits_processor(cfg=4.0, image_top_k=200)
    generation_config = GenerationConfig(
        max_new_tokens=10240,
        max_length=solver.model.config.max_position_embeddings,
        temperature=1.0,
        top_k=None,
        do_sample=True,
        eos_token_id=[151666],
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result = solver.model.generate(
            prompt_tensor, generation_config, logits_processor=logits_processor
        )[0][prompt_len:].tolist()
    torch.cuda.synchronize()
    gen_time = time.perf_counter() - t0

    return result, gen_time


def replay_with_attention(model, prompt_tensor, generated_tokens, prompt_len, regions):
    """Phase B: Replay generation token-by-token, logging attention at sampled steps.

    At each sampled step, output_attentions=True triggers SDPA→eager fallback,
    giving us attention weights (1, n_heads, 1, kv_len) per layer.
    We compute attention mass per region (sum of weights on region tokens).
    """
    from transformers import DynamicCache

    device = prompt_tensor.device
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_gen = len(generated_tokens)

    # Region names for output
    region_keys = ["demo_depth", "demo_rgb", "demo_other", "query_text", "query_img", "generated"]

    # Storage: per sampled step, per layer → dict of region masses (n_heads,)
    # Final shape: (n_sampled_steps, n_layers, n_heads, n_regions)
    sampled_steps = []
    sampled_step_indices = []

    print(f"  Replay: {n_gen} tokens, sampling every {SAMPLE_EVERY}th...")

    # Phase B.1: Prefill (no attention logging — too large)
    past_key_values = DynamicCache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=prompt_tensor,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
        )
        past_key_values = outputs.past_key_values

    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0
    print(f"  Prefill: {prefill_time:.1f}s")

    # Phase B.2: Replay generated tokens one-by-one
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step_idx in range(n_gen):
        token_id = generated_tokens[step_idx]
        input_ids = torch.tensor(
            [[token_id]], dtype=torch.int64, device=device
        )
        should_log = (step_idx % SAMPLE_EVERY == 0) or (step_idx == n_gen - 1)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=should_log,
            )
            past_key_values = outputs.past_key_values

        if should_log and outputs.attentions is not None:
            # outputs.attentions: tuple of n_layers tensors, each (1, n_heads, 1, kv_len)
            step_data = np.zeros((n_layers, n_heads, len(region_keys)), dtype=np.float32)

            for layer_idx, attn_w in enumerate(outputs.attentions):
                # attn_w: (1, n_heads, 1, kv_len)
                w = attn_w[0, :, 0, :].float()  # (n_heads, kv_len)
                kv_len = w.shape[-1]

                # Compute mass per region
                # demo_depth
                s, e = regions["demo_depth"]
                step_data[layer_idx, :, 0] = w[:, s:e].sum(dim=-1).cpu().numpy()

                # demo_rgb
                s, e = regions["demo_rgb"]
                step_data[layer_idx, :, 1] = w[:, s:e].sum(dim=-1).cpu().numpy()

                # demo_other (non-contiguous: header + footer)
                s1, e1, s2, e2 = regions["demo_other"]
                mass = w[:, s1:e1].sum(dim=-1) + w[:, s2:e2].sum(dim=-1)
                step_data[layer_idx, :, 2] = mass.cpu().numpy()

                # query_text
                s, e = regions["query_text"]
                step_data[layer_idx, :, 3] = w[:, s:e].sum(dim=-1).cpu().numpy()

                # query_img
                s, e = regions["query_img"]
                step_data[layer_idx, :, 4] = w[:, s:e].sum(dim=-1).cpu().numpy()

                # generated (everything after prompt)
                gen_start = prompt_len
                if kv_len > gen_start:
                    step_data[layer_idx, :, 5] = (
                        w[:, gen_start:].sum(dim=-1).cpu().numpy()
                    )

            sampled_steps.append(step_data)
            sampled_step_indices.append(step_idx)

        if (step_idx + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            rate = (step_idx + 1) / elapsed
            eta = (n_gen - step_idx - 1) / rate
            print(
                f"    Step {step_idx + 1}/{n_gen} "
                f"({elapsed:.0f}s elapsed, {eta:.0f}s ETA)"
            )

    torch.cuda.synchronize()
    replay_time = time.perf_counter() - t0
    print(f"  Replay: {replay_time:.1f}s ({n_gen} tokens)")

    # Stack into array: (n_sampled, n_layers, n_heads, n_regions)
    attention_data = np.stack(sampled_steps, axis=0)

    return {
        "attention_masses": attention_data,
        "sampled_steps": sampled_step_indices,
        "region_keys": region_keys,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_generated": n_gen,
        "prompt_len": prompt_len,
        "prefill_time_s": prefill_time,
        "replay_time_s": replay_time,
    }


def run_case(solver, case_id, query_path, demo_rgb_path, demo_depth_path):
    """Run attention characterization for a single test case."""
    print(f"\n{'='*60}")
    print(f"Case {case_id}")
    print(f"{'='*60}")

    # Build prompt
    prompt_tensor, prompt_len, ref_tokens, regions = build_1shot_prompt(
        solver, query_path, demo_rgb_path, demo_depth_path
    )
    print(f"  Prompt: {prompt_len} tokens")
    print(f"  Regions: demo_depth={regions['demo_depth'][1]-regions['demo_depth'][0]}, "
          f"demo_rgb={regions['demo_rgb'][1]-regions['demo_rgb'][0]}, "
          f"query_text={regions['query_text'][1]-regions['query_text'][0]}, "
          f"query_img={regions['query_img'][1]-regions['query_img'][0]}")

    # Phase A: Generate tokens
    print(f"  Phase A: Generating tokens (CFG=4.0)...")
    torch.cuda.reset_peak_memory_stats()
    gen_tokens, gen_time = generate_tokens(solver, prompt_tensor, prompt_len)
    gen_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"    Generated {len(gen_tokens)} tokens in {gen_time:.1f}s "
          f"({gen_time/len(gen_tokens)*1000:.1f} ms/tok), VRAM: {gen_vram:.1f} GB")

    # Decode generated image to verify quality
    if ref_tokens is not None:
        decode_tokens = ref_tokens + gen_tokens
    else:
        decode_tokens = gen_tokens
    text_result, images_result = solver.decode_ids(decode_tokens)
    if images_result:
        out_path = os.path.join(OUTPUT_ROOT, f"ar1_case{case_id}_generated.png")
        images_result[0].save(out_path)
        print(f"    Output saved: {out_path}")

    # Phase B: Replay with attention logging
    print(f"  Phase B: Replaying with attention logging...")
    torch.cuda.reset_peak_memory_stats()
    attn_result = replay_with_attention(
        solver.model, prompt_tensor, gen_tokens, prompt_len, regions
    )
    replay_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"    Replay VRAM: {replay_vram:.1f} GB")
    print(f"    Sampled {len(attn_result['sampled_steps'])} steps")

    # Quick summary stats
    masses = attn_result["attention_masses"]  # (n_sampled, n_layers, n_heads, n_regions)
    region_keys = attn_result["region_keys"]

    # Aggregate: mean across all sampled steps, layers, heads
    mean_per_region = masses.mean(axis=(0, 1, 2))  # (n_regions,)
    print(f"\n  --- Aggregate attention mass per region (mean over all) ---")
    for i, name in enumerate(region_keys):
        print(f"    {name:>15s}: {mean_per_region[i]:.4f} ({mean_per_region[i]*100:.2f}%)")

    return {
        "case_id": case_id,
        "prompt_len": prompt_len,
        "n_generated": len(gen_tokens),
        "gen_time_s": gen_time,
        "gen_vram_gb": gen_vram,
        "replay_vram_gb": replay_vram,
        "regions": {
            k: list(v) for k, v in regions.items()
        },
        "attention_masses": masses,
        "sampled_steps": attn_result["sampled_steps"],
        "region_keys": region_keys,
        "prefill_time_s": attn_result["prefill_time_s"],
        "replay_time_s": attn_result["replay_time_s"],
    }


def analyze_results(all_results):
    """Comprehensive analysis of attention characterization across all cases."""
    print("\n" + "=" * 70)
    print("ATTENTION CHARACTERIZATION ANALYSIS")
    print("=" * 70)

    region_keys = all_results[0]["region_keys"]
    n_layers = all_results[0]["attention_masses"].shape[1]
    n_heads = all_results[0]["attention_masses"].shape[2]

    # Stack all cases: (n_cases, n_sampled, n_layers, n_heads, n_regions)
    # Note: sampled_steps count may differ slightly between cases
    # Use per-case analysis then aggregate

    # 1. AGGREGATE: Mean attention mass per region across all cases
    print("\n1. AGGREGATE ATTENTION MASS PER REGION")
    print("-" * 50)
    all_means = []
    for r in all_results:
        m = r["attention_masses"]  # (n_sampled, n_layers, n_heads, n_regions)
        case_mean = m.mean(axis=(0, 1, 2))  # (n_regions,)
        all_means.append(case_mean)
    grand_mean = np.mean(all_means, axis=0)
    grand_std = np.std(all_means, axis=0)

    # Combined demo = demo_depth + demo_rgb + demo_other
    demo_total_mean = grand_mean[0] + grand_mean[1] + grand_mean[2]
    query_total_mean = grand_mean[3] + grand_mean[4]

    print(f"  {'Region':>20s}  {'Mean':>8s}  {'Std':>8s}")
    for i, name in enumerate(region_keys):
        print(f"  {name:>20s}  {grand_mean[i]:>8.4f}  {grand_std[i]:>8.4f}")
    print(f"  {'--- DEMO total':>20s}  {demo_total_mean:>8.4f}")
    print(f"  {'--- QUERY total':>20s}  {query_total_mean:>8.4f}")

    # 2. PER-LAYER BREAKDOWN: Mean demo attention by layer
    print("\n2. PER-LAYER DEMO ATTENTION (averaged over heads, steps, cases)")
    print("-" * 50)
    # Demo attention = sum of demo_depth + demo_rgb + demo_other
    per_layer_demo = []
    for r in all_results:
        m = r["attention_masses"]
        demo_mass = m[:, :, :, 0] + m[:, :, :, 1] + m[:, :, :, 2]  # (n_sampled, n_layers, n_heads)
        layer_mean = demo_mass.mean(axis=(0, 2))  # (n_layers,)
        per_layer_demo.append(layer_mean)
    per_layer_demo = np.mean(per_layer_demo, axis=0)  # (n_layers,)

    for layer in range(n_layers):
        bar = "#" * int(per_layer_demo[layer] * 200)
        print(f"  Layer {layer:2d}: {per_layer_demo[layer]:.4f}  {bar}")

    # 3. HEAD SPECIALIZATION: Find heads with highest demo attention
    print("\n3. HEAD SPECIALIZATION (top 10 demo-reading heads)")
    print("-" * 50)
    per_head_demo = []
    for r in all_results:
        m = r["attention_masses"]
        demo_mass = m[:, :, :, 0] + m[:, :, :, 1] + m[:, :, :, 2]
        head_mean = demo_mass.mean(axis=0)  # (n_layers, n_heads)
        per_head_demo.append(head_mean)
    per_head_demo = np.mean(per_head_demo, axis=0)  # (n_layers, n_heads)

    # Flatten and sort
    head_list = []
    for l in range(n_layers):
        for h in range(n_heads):
            head_list.append((per_head_demo[l, h], l, h))
    head_list.sort(reverse=True)

    print(f"  {'Layer':>5s}  {'Head':>4s}  {'Demo Attn':>10s}  {'Category'}")
    for mass, l, h in head_list[:10]:
        cat = "STRONG" if mass > 0.1 else "moderate" if mass > 0.05 else "weak"
        print(f"  {l:>5d}  {h:>4d}  {mass:>10.4f}  {cat}")

    # Min demo-attending heads
    print(f"\n  Bottom 5 (lowest demo attention):")
    for mass, l, h in head_list[-5:]:
        print(f"  {l:>5d}  {h:>4d}  {mass:>10.4f}")

    # 4. TEMPORAL DYNAMICS: How demo attention changes during generation
    print("\n4. TEMPORAL DYNAMICS (early / mid / late)")
    print("-" * 50)
    for r in all_results:
        m = r["attention_masses"]  # (n_sampled, n_layers, n_heads, n_regions)
        steps = r["sampled_steps"]
        n_gen = r["n_generated"]
        n_sampled = len(steps)

        # Split into thirds
        third = n_sampled // 3
        early = m[:third]
        mid = m[third:2*third]
        late = m[2*third:]

        demo_early = (early[:, :, :, 0] + early[:, :, :, 1] + early[:, :, :, 2]).mean()
        demo_mid = (mid[:, :, :, 0] + mid[:, :, :, 1] + mid[:, :, :, 2]).mean()
        demo_late = (late[:, :, :, 0] + late[:, :, :, 1] + late[:, :, :, 2]).mean()

        gen_early = early[:, :, :, 5].mean()
        gen_mid = mid[:, :, :, 5].mean()
        gen_late = late[:, :, :, 5].mean()

        print(f"  Case {r['case_id']}:")
        print(f"    Demo attention:  early={demo_early:.4f}  mid={demo_mid:.4f}  late={demo_late:.4f}")
        print(f"    Gen attention:   early={gen_early:.4f}  mid={gen_mid:.4f}  late={gen_late:.4f}")
        print(f"    Ratio late/early (demo): {demo_late/max(demo_early, 1e-8):.3f}")

    # 5. DEMO SUB-STRUCTURE: input image vs output image
    print("\n5. DEMO SUB-STRUCTURE: depth (output) vs RGB (input)")
    print("-" * 50)
    for r in all_results:
        m = r["attention_masses"]
        depth_mean = m[:, :, :, 0].mean()
        rgb_mean = m[:, :, :, 1].mean()
        ratio = depth_mean / max(rgb_mean, 1e-8)
        print(f"  Case {r['case_id']}: depth={depth_mean:.4f}  rgb={rgb_mean:.4f}  ratio={ratio:.3f}")

    # 6. MEMORY OVERHEAD
    print("\n6. MEMORY AND TIMING OVERHEAD")
    print("-" * 50)
    for r in all_results:
        print(f"  Case {r['case_id']}: "
              f"gen={r['gen_time_s']:.0f}s  "
              f"prefill={r['prefill_time_s']:.1f}s  "
              f"replay={r['replay_time_s']:.0f}s  "
              f"gen_VRAM={r['gen_vram_gb']:.1f}GB  "
              f"replay_VRAM={r['replay_vram_gb']:.1f}GB")

    return {
        "grand_mean_per_region": {k: float(v) for k, v in zip(region_keys, grand_mean)},
        "demo_total_mean": float(demo_total_mean),
        "query_total_mean": float(query_total_mean),
        "per_layer_demo": per_layer_demo.tolist(),
        "per_head_demo": per_head_demo.tolist(),
    }


def main():
    print("=" * 60)
    print("AR-1: Attention Characterization During AR Image Generation")
    print("=" * 60)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(PROFILING_ROOT, exist_ok=True)

    # Load task instances
    with open(os.path.join(DATA_ROOT, "task_instances.json")) as f:
        instances = json.load(f)
    print(f"Loaded {len(instances)} task instances")

    # Load model
    print("Loading model...")
    from inference_solver import FlexARInferenceSolver
    solver = FlexARInferenceSolver(
        model_path=MODEL_PATH, precision="bf16", target_size=512
    )
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Warmup (standard 0-shot to compile kernels)
    print("\nWarmup run (0-shot, no attention logging)...")
    query_path = instances[0]["query"]["rgb_path"]
    img = Image.open(query_path)
    img_cropped = center_crop(img, crop_size=[WIDTH, HEIGHT // 2])
    cropped_path = os.path.join(OUTPUT_ROOT, "ar1_warmup.png")
    img_cropped.save(cropped_path)

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
    if 151665 in _prompt:
        _prompt.pop()
    prompt_ids = []
    for v in _prompt:
        if isinstance(v, int):
            prompt_ids.append(v)
        else:
            prompt_ids += v["input_ids"]
    warmup_tensor = torch.tensor(
        prompt_ids, dtype=torch.int64, device=solver.model.device
    ).unsqueeze(0)
    from transformers import GenerationConfig
    gen_cfg = GenerationConfig(
        max_new_tokens=100, max_length=10240,
        temperature=1.0, do_sample=True, eos_token_id=[151666],
    )
    logits_proc = solver.create_logits_processor(cfg=4.0, image_top_k=200)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = solver.model.generate(warmup_tensor, gen_cfg, logits_processor=logits_proc)
    print("Warmup complete.\n")

    # Run 3 test cases (different query images, same demo pair for consistency)
    all_results = []
    for case_idx in range(3):
        inst = instances[case_idx]
        query_path = inst["query"]["rgb_path"]
        demo_rgb = inst["demo_pairs"][0]["rgb_path"]
        demo_depth = inst["demo_pairs"][0]["depth_colored_path"]

        result = run_case(solver, case_idx, query_path, demo_rgb, demo_depth)
        all_results.append(result)

    # Comprehensive analysis
    analysis = analyze_results(all_results)

    # Save results
    save_data = {
        "config": {
            "model": "Lumina-mGPT-2.0-Omni",
            "n_layers": 32,
            "n_heads": 32,
            "sample_every": SAMPLE_EVERY,
            "cfg": 4.0,
            "n_cases": len(all_results),
        },
        "analysis": analysis,
        "per_case": [],
    }
    for r in all_results:
        case_data = {
            "case_id": r["case_id"],
            "prompt_len": r["prompt_len"],
            "n_generated": r["n_generated"],
            "gen_time_s": r["gen_time_s"],
            "replay_time_s": r["replay_time_s"],
            "prefill_time_s": r["prefill_time_s"],
            "gen_vram_gb": r["gen_vram_gb"],
            "replay_vram_gb": r["replay_vram_gb"],
            "regions": r["regions"],
            "sampled_steps": r["sampled_steps"],
            "region_keys": r["region_keys"],
            # attention_masses: (n_sampled, n_layers, n_heads, n_regions)
            # Save as nested list for JSON
            "attention_masses": r["attention_masses"].tolist(),
        }
        save_data["per_case"].append(case_data)

    out_path = os.path.join(PROFILING_ROOT, "ar1_attention_characterization.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n=== AR-1 ATTENTION CHARACTERIZATION COMPLETE ===")


if __name__ == "__main__":
    main()
