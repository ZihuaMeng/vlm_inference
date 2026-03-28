#!/usr/bin/env python3
"""F1 Phase 1: FLUX Kontext-dev Baseline Timing

Loads Kontext-dev in full bf16 on B200, runs a single-image editing pass,
records wall-clock time and peak VRAM. Also tests whether multi-image
input is accepted by the pipeline.
"""

import torch
import time
import json
import os
from PIL import Image

# ─── Configuration ───
WEIGHTS = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/weights/FLUX-Kontext-dev"
OUTPUT_DIR = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/outputs/flux"
NUM_STEPS = 28
GUIDANCE_SCALE = 2.5
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Load Pipeline ───
print("Loading FluxKontextPipeline in bf16...")
t_load_start = time.perf_counter()

from diffusers import FluxKontextPipeline

pipeline = FluxKontextPipeline.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
pipeline.to("cuda")

t_load_end = time.perf_counter()
vram_after_load = torch.cuda.memory_allocated() / 1e9
vram_reserved_load = torch.cuda.max_memory_reserved() / 1e9
print(f"Pipeline loaded in {t_load_end - t_load_start:.1f}s")
print(f"VRAM allocated: {vram_after_load:.2f} GB, reserved: {vram_reserved_load:.2f} GB")

# ─── Print Environment Info ───
print(f"\nEnvironment:")
print(f"  torch: {torch.__version__}")
print(f"  CUDA: {torch.version.cuda}")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import diffusers
print(f"  diffusers: {diffusers.__version__}")
import sentencepiece
print(f"  sentencepiece: {sentencepiece.__version__}")

# ─── Prepare Test Image ───
# Simple solid-color reference image (same approach as Qwen baseline)
ref_image = Image.new("RGB", (512, 512), "red")
prompt = "A red cube sitting on a white table with soft lighting."

# ─── Warmup Pass ───
print("\nRunning warmup pass...")
torch.cuda.synchronize()
warmup_start = time.perf_counter()
with torch.no_grad():
    _ = pipeline(
        image=ref_image,
        prompt=prompt,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=torch.Generator("cuda").manual_seed(SEED),
    )
torch.cuda.synchronize()
warmup_time = time.perf_counter() - warmup_start
print(f"Warmup complete: {warmup_time:.1f}s")

# ─── Timed Baseline Pass ───
print("\n" + "=" * 60)
print("BASELINE PASS (single image, 28 steps)")
print("=" * 60)

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t_start = time.perf_counter()

with torch.no_grad():
    result = pipeline(
        image=ref_image,
        prompt=prompt,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=torch.Generator("cuda").manual_seed(SEED),
    )

torch.cuda.synchronize()
t_end = time.perf_counter()

baseline_time = t_end - t_start
peak_vram = torch.cuda.max_memory_reserved() / 1e9
peak_allocated = torch.cuda.max_memory_allocated() / 1e9

output_image = result.images[0]
output_path = os.path.join(OUTPUT_DIR, "f1_baseline_single.png")
output_image.save(output_path)

print(f"\n  Wall-clock time: {baseline_time:.2f}s")
print(f"  Per-step average: {baseline_time / NUM_STEPS * 1000:.1f}ms")
print(f"  Peak VRAM allocated: {peak_allocated:.2f} GB")
print(f"  Peak VRAM reserved: {peak_vram:.2f} GB")
print(f"  Output saved: {output_path}")
print(f"  Output size: {output_image.size}")

# ─── Multi-Image Test ───
print("\n" + "=" * 60)
print("MULTI-IMAGE TEST")
print("=" * 60)

multi_image_works = False
multi_time = None
try:
    ref_image2 = Image.new("RGB", (512, 512), "blue")
    torch.cuda.synchronize()
    t_multi_start = time.perf_counter()
    with torch.no_grad():
        result2 = pipeline(
            image=[ref_image, ref_image2],
            prompt="A red object and a blue object side by side on a white table.",
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(SEED),
        )
    torch.cuda.synchronize()
    multi_time = time.perf_counter() - t_multi_start
    multi_image_works = True
    multi_output = result2.images[0]
    multi_path = os.path.join(OUTPUT_DIR, "f1_baseline_multi.png")
    multi_output.save(multi_path)
    print(f"  Multi-image WORKS. Time: {multi_time:.2f}s")
    print(f"  Output saved: {multi_path}")
except Exception as e:
    print(f"  Multi-image FAILED: {type(e).__name__}: {e}")
    print("  As expected from F0 recon: pipeline accepts single image only.")

# ─── Results Summary ───
results = {
    "environment": {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "diffusers": diffusers.__version__,
        "sentencepiece": sentencepiece.__version__,
    },
    "config": {
        "model": "FLUX.1 Kontext-dev",
        "weights_path": WEIGHTS,
        "num_steps": NUM_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "seed": SEED,
        "dtype": "bfloat16",
    },
    "baseline": {
        "wall_clock_s": round(baseline_time, 2),
        "per_step_ms": round(baseline_time / NUM_STEPS * 1000, 1),
        "vram_after_load_gb": round(vram_after_load, 2),
        "peak_vram_allocated_gb": round(peak_allocated, 2),
        "peak_vram_reserved_gb": round(peak_vram, 2),
        "output_size": list(output_image.size),
        "warmup_time_s": round(warmup_time, 1),
    },
    "multi_image": {
        "works": multi_image_works,
        "time_s": round(multi_time, 2) if multi_time else None,
    },
}

json_path = os.path.join(OUTPUT_DIR, "f1_baseline_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {json_path}")
print("\nDone.")
