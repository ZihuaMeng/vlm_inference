"""
VLM ICL Profiled Inference — Phases 4 & 5 Combined

Runs depth estimation on NYUDepthV2 instances with stage-level profiling.
No per-token hooks (they add prohibitive overhead). Instead:
1. Time overall pipeline stages (tokenization, AR generation, VQ decode)
2. Use model's own timing for token count
3. Derive per-token cost from total AR time / token count
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

# Unbuffered output
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

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(PROFILING_ROOT, exist_ok=True)


def run_depth_profiled(solver, instance_id, image_path, width=512, height=512):
    """Run depth estimation with stage-level timing (no per-token hooks)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    profile = {"instance_id": instance_id}

    # Stage 1: Input tokenization
    t0 = time.perf_counter()
    img = Image.open(image_path)
    img_cropped = center_crop(img, crop_size=[width, height // 2])
    cropped_path = os.path.join(OUTPUT_ROOT, f"inst_{instance_id}_cropped.png")
    img_cropped.save(cropped_path)

    prompt = (
        f"Generate a dual-panel image of {width}x{height} where the <upper half> "
        f"displays a <depth map>, while the <lower half> retains the original image "
        f"for direct visual comparison:\nAn indoor scene"
    )

    conversations = [
        {"from": "human", "value": prompt},
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

    prompt_ids = []
    for value in _prompt:
        if isinstance(value, int):
            prompt_ids.append(value)
        else:
            prompt_ids += value["input_ids"]
    prompt_len = len(prompt_ids)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.int64, device=solver.model.device).unsqueeze(0)

    torch.cuda.synchronize()
    t_tokenize = time.perf_counter() - t0
    profile["input_tokenization_s"] = t_tokenize
    profile["prompt_tokens"] = prompt_len

    # Stage 2: AR token generation
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
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        generation_result = solver.model.generate(
            prompt_tensor, generation_config, logits_processor=logits_processor
        )[0][prompt_len:].tolist()
    torch.cuda.synchronize()
    t_ar = time.perf_counter() - t0

    n_generated = len(generation_result)
    profile["ar_generation_s"] = t_ar
    profile["tokens_generated"] = n_generated
    profile["ms_per_token"] = (t_ar / n_generated * 1000) if n_generated > 0 else 0

    # Stage 3: VQ-GAN decode
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if ref_tokens is not None:
        generation_result = ref_tokens + generation_result
    text_result, images_result = solver.decode_ids(generation_result)
    torch.cuda.synchronize()
    t_decode = time.perf_counter() - t0
    profile["vq_decode_s"] = t_decode

    # Save output
    if images_result:
        out_path = os.path.join(OUTPUT_ROOT, f"inst_{instance_id}_depth.png")
        images_result[0].save(out_path)
        profile["output_path"] = out_path
        profile["output_size"] = list(images_result[0].size)

    # Summary
    t_total = t_tokenize + t_ar + t_decode
    profile["total_s"] = t_total
    profile["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    profile["pct_tokenize"] = t_tokenize / t_total * 100
    profile["pct_ar"] = t_ar / t_total * 100
    profile["pct_decode"] = t_decode / t_total * 100

    return profile


def main():
    print("=" * 60)
    print("VLM ICL Profiled Inference — Phases 4 & 5")
    print("=" * 60)

    with open(os.path.join(DATA_ROOT, "task_instances.json")) as f:
        instances = json.load(f)

    print(f"Loaded {len(instances)} task instances")

    # Load model
    print("Loading model...")
    from inference_solver import FlexARInferenceSolver
    solver = FlexARInferenceSolver(
        model_path=MODEL_PATH,
        precision="bf16",
        target_size=512,
    )
    vram_load = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. VRAM: {vram_load:.2f} GB")

    # Warmup (not profiled)
    print("\nWarmup run...")
    query_path = instances[0]["query"]["rgb_path"]
    _ = run_depth_profiled(solver, "warmup", query_path)
    print("Warmup complete.\n")

    # Run 5 instances
    all_profiles = []
    for i, inst in enumerate(instances[:5]):
        query_path = inst["query"]["rgb_path"]
        print(f"Instance {i} (query pair {inst['query']['index']})...")
        profile = run_depth_profiled(solver, i, query_path)
        all_profiles.append(profile)
        print(f"  {profile['total_s']:.1f}s | Tokens: {profile['tokens_generated']} | "
              f"{profile['ms_per_token']:.1f} ms/tok | "
              f"AR: {profile['pct_ar']:.1f}% | VQ: {profile['pct_decode']:.1f}%")

    # Save profiles
    profile_path = os.path.join(PROFILING_ROOT, "depth_estimation_profiles.json")
    with open(profile_path, "w") as f:
        json.dump(all_profiles, f, indent=2, default=str)
    print(f"\nSaved profiles to {profile_path}")

    # Summary
    print("\n" + "=" * 60)
    print("PROFILING SUMMARY")
    print("=" * 60)

    times = [p["total_s"] for p in all_profiles]
    ar_pcts = [p["pct_ar"] for p in all_profiles]
    tok_pcts = [p["pct_tokenize"] for p in all_profiles]
    vq_pcts = [p["pct_decode"] for p in all_profiles]
    ms_per_tok = [p["ms_per_token"] for p in all_profiles]
    n_tokens = [p["tokens_generated"] for p in all_profiles]

    print(f"\nPipeline Stage Decomposition ({len(all_profiles)} runs):")
    print(f"  Input tokenization: {np.mean(tok_pcts):.2f}% ({np.mean([p['input_tokenization_s'] for p in all_profiles]):.2f}s)")
    print(f"  AR generation:      {np.mean(ar_pcts):.2f}% ({np.mean([p['ar_generation_s'] for p in all_profiles]):.1f}s)")
    print(f"  VQ-GAN decode:      {np.mean(vq_pcts):.2f}% ({np.mean([p['vq_decode_s'] for p in all_profiles]):.2f}s)")
    print(f"\nTiming:")
    print(f"  Total:       {np.mean(times):.1f}s ± {np.std(times):.1f}s")
    print(f"  ms/token:    {np.mean(ms_per_tok):.1f} ± {np.std(ms_per_tok):.1f}")
    print(f"  Tokens/run:  {np.mean(n_tokens):.0f}")
    print(f"  Peak VRAM:   {max(p['peak_vram_gb'] for p in all_profiles):.2f} GB")

    # Comparison with diffusion models
    print(f"\n--- Comparison with Diffusion (Qwen/FLUX) ---")
    print(f"  Lumina-mGPT i2i depth: {np.mean(times):.1f}s")
    print(f"  Qwen 2-img edit (B200): 47.4s")
    print(f"  FLUX Kontext (B200):    9.2s")
    print(f"  AR model is ~{np.mean(times)/47.4:.0f}x slower than Qwen, ~{np.mean(times)/9.2:.0f}x slower than FLUX")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
