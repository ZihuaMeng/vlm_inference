"""
Context-Length Scaling Experiment

Measures how AR generation speed (ms/token) scales with input context length.
Tests 0-shot (query only) vs 1-shot (1 demo image + query) depth estimation.

Key insight: decoder-only attention means each generated token attends to ALL
prior tokens. Longer context → more attention compute per generated token.
With KV cache, incremental cost per token is O(total_sequence_length).
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


def build_prompt_with_context(solver, query_path, demo_paths=None, width=512, height=512):
    """Build prompt tensor with optional in-context demo images prepended.

    Args:
        demo_paths: List of (rgb_path, depth_colored_path) tuples for demos.
                    Each demo is tokenized as a full 512x512 image.
    """
    demo_paths = demo_paths or []

    # Tokenize demo images as full context (using process_image, not i2i)
    context_tokens = []
    for rgb_path, depth_path in demo_paths:
        # Create stitched demo: depth on top, RGB on bottom (512x512 total)
        rgb = Image.open(rgb_path)
        depth = Image.open(depth_path)
        rgb_cropped = center_crop(rgb, crop_size=[width, height // 2])
        depth_cropped = center_crop(depth, crop_size=[width, height // 2])
        stitched = Image.new("RGB", (width, height))
        stitched.paste(depth_cropped, (0, 0))
        stitched.paste(rgb_cropped, (0, height // 2))

        # Tokenize the full stitched image (includes image_start and image_end)
        demo_result = solver.item_processor.process_image(stitched)
        context_tokens.extend(demo_result["input_ids"])

    # Build the query prompt (same as run_inference_profiled.py)
    img = Image.open(query_path)
    img_cropped = center_crop(img, crop_size=[width, height // 2])
    cropped_path = os.path.join(OUTPUT_ROOT, "ctx_scaling_cropped.png")
    img_cropped.save(cropped_path)

    prompt_text = (
        f"Generate a dual-panel image of {width}x{height} where the <upper half> "
        f"displays a <depth map>, while the <lower half> retains the original image "
        f"for direct visual comparison:\nAn indoor scene"
    )

    conversations = [
        {"from": "human", "value": prompt_text},
        {"from": "gpt", "value": "<|image|>"},
    ]
    item = {"image": [cropped_path], "conversations": conversations}
    solver.item_processor.transform = {"<|image|>": solver.item_processor.process_i2i_image}
    _prompt = solver.item_processor.process_item(item)

    if 151665 in _prompt:
        _prompt.pop()
        start_index = _prompt.index(151665)
        ref_tokens = _prompt[start_index:]
    else:
        ref_tokens = None

    # Flatten query prompt
    query_ids = []
    for value in _prompt:
        if isinstance(value, int):
            query_ids.append(value)
        else:
            query_ids += value["input_ids"]

    # Combine: context tokens + query tokens
    all_ids = context_tokens + query_ids
    prompt_tensor = torch.tensor(all_ids, dtype=torch.int64, device=solver.model.device).unsqueeze(0)

    return prompt_tensor, len(all_ids), len(context_tokens), ref_tokens


def run_scaling_trial(solver, prompt_tensor, prompt_len, ref_tokens, trial_name):
    """Run a single AR generation trial and return timing profile."""
    from transformers import GenerationConfig

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    logits_processor = solver.create_logits_processor(cfg=4.0, image_top_k=200)
    generation_config = GenerationConfig(
        max_new_tokens=10240,
        max_length=solver.model.config.max_position_embeddings,
        temperature=1.0,
        top_k=None,
        do_sample=True,
        eos_token_id=[151666],
    )

    # Prefill timing
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        generation_result = solver.model.generate(
            prompt_tensor, generation_config, logits_processor=logits_processor
        )[0][prompt_len:].tolist()
    torch.cuda.synchronize()
    t_total = time.perf_counter() - t0

    n_generated = len(generation_result)
    ms_per_token = (t_total / n_generated * 1000) if n_generated > 0 else 0

    # Decode to get output image
    if ref_tokens is not None:
        generation_result = ref_tokens + generation_result
    text_result, images_result = solver.decode_ids(generation_result)

    profile = {
        "trial": trial_name,
        "prompt_tokens": prompt_len,
        "tokens_generated": n_generated,
        "total_s": t_total,
        "ms_per_token": ms_per_token,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "has_output_image": len(images_result) > 0,
    }

    if images_result:
        out_path = os.path.join(OUTPUT_ROOT, f"ctx_scaling_{trial_name}.png")
        images_result[0].save(out_path)
        profile["output_path"] = out_path

    return profile


def main():
    print("=" * 60)
    print("Context-Length Scaling Experiment")
    print("=" * 60)

    with open(os.path.join(DATA_ROOT, "task_instances.json")) as f:
        instances = json.load(f)

    # Use instance 0 for all trials (same query, different context)
    inst = instances[0]
    query_path = inst["query"]["rgb_path"]
    demo_pairs = [(d["rgb_path"], d["depth_colored_path"]) for d in inst["demo_pairs"]]

    print("Loading model...")
    from inference_solver import FlexARInferenceSolver
    solver = FlexARInferenceSolver(model_path=MODEL_PATH, precision="bf16", target_size=512)
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Warmup
    print("\nWarmup run (0-shot)...")
    pt, pl, cl, rt = build_prompt_with_context(solver, query_path, demo_paths=[])
    _ = run_scaling_trial(solver, pt, pl, rt, "warmup")
    print("Warmup complete.\n")

    # Define conditions
    conditions = [
        ("0-shot", []),
        ("1-shot", [demo_pairs[0]]),
    ]

    # Check if 2-shot fits in max_position_embeddings
    pt_test, pl_test, cl_test, _ = build_prompt_with_context(
        solver, query_path, demo_paths=[demo_pairs[0], demo_pairs[1]]
    )
    estimated_total = pl_test + 2200  # ~2200 generated tokens expected
    print(f"2-shot check: prompt={pl_test} tokens, estimated total={estimated_total}, "
          f"max={solver.model.config.max_position_embeddings}")
    if estimated_total <= solver.model.config.max_position_embeddings:
        conditions.append(("2-shot", [demo_pairs[0], demo_pairs[1]]))
        print("  → 2-shot FITS, will test it")
    else:
        print("  → 2-shot EXCEEDS limit, skipping")

    # Run trials
    results = []
    for name, demos in conditions:
        print(f"\n--- {name} ---")
        pt, pl, cl, rt = build_prompt_with_context(solver, query_path, demo_paths=demos)
        print(f"  Prompt: {pl} tokens ({cl} context + {pl - cl} query)")

        profile = run_scaling_trial(solver, pt, pl, rt, name)
        results.append(profile)

        print(f"  Generated: {profile['tokens_generated']} tokens in {profile['total_s']:.1f}s")
        print(f"  ms/token: {profile['ms_per_token']:.1f}")
        print(f"  Peak VRAM: {profile['peak_vram_gb']:.2f} GB")

    # Save results
    out_path = os.path.join(PROFILING_ROOT, "context_scaling_profiles.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("CONTEXT SCALING SUMMARY")
    print("=" * 60)
    print(f"{'Condition':<12} {'Prompt Toks':>12} {'Gen Toks':>10} {'ms/tok':>8} {'Total(s)':>10} {'VRAM(GB)':>10}")
    print("-" * 62)
    baseline_ms = None
    for r in results:
        if baseline_ms is None:
            baseline_ms = r["ms_per_token"]
            ratio = "1.00x"
        else:
            ratio = f"{r['ms_per_token']/baseline_ms:.2f}x"
        print(f"{r['trial']:<12} {r['prompt_tokens']:>12} {r['tokens_generated']:>10} "
              f"{r['ms_per_token']:>7.1f} {r['total_s']:>10.1f} {r['peak_vram_gb']:>10.2f}  ({ratio})")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
