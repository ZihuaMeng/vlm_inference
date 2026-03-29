#!/usr/bin/env python3
"""F2: FLUX Kontext-dev Validation + Ablation Study

Validates temporal caching transfer on a diverse 12-case test suite.
Three configurations per case:
  a. Baseline (full 28 steps, no caching)
  b. Conservative (cache steps 6-20, 13 full steps, 2.15x theoretical)
  c. Aggressive (cache steps 4-23, 8 full steps, 3.50x theoretical)

Metrics: wall-clock time, speedup, SSIM, LPIPS.
Identical seeds for paired comparison.
"""

import torch
import time
import json
import os
import csv
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import lpips
import importlib

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════
WEIGHTS = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/weights/FLUX-Kontext-dev"
OUTPUT_DIR = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/outputs/flux/f2_validation"
RESULTS_DIR = "/home/zihua.meng/projects/vlm_inference/scripts/flux"
NUM_STEPS = 28
GUIDANCE_SCALE = 2.5
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  SCHEDULE DEFINITIONS (from F1)
# ═══════════════════════════════════════════════════════════════
SCHEDULES = {
    "baseline": {"cached_steps": set(), "label": "Baseline (full 28 steps)"},
    "conservative": {"cached_steps": set(range(6, 21)), "label": "Conservative (cache steps 6-20)"},
    "aggressive": {"cached_steps": set(range(4, 24)), "label": "Aggressive (cache steps 4-23)"},
}

MODES = ["baseline", "conservative", "aggressive"]

# ═══════════════════════════════════════════════════════════════
#  TEST SUITE — 12 cases covering diverse edit types
# ═══════════════════════════════════════════════════════════════
#
# Design rationale:
# - 4 easy, 4 medium, 4 hard (balanced difficulty)
# - Covers: objects, backgrounds, portraits, style, text, texture,
#   identity-preserving, complex scenes, abstract, interior, landscape, nature
# - Includes cases expected to be easy AND hard based on Qwen M9 failure
#   patterns (complex scenes, textures, style transfer degrade most)
# - Single-image only (Kontext limitation)

TEST_SUITE = [
    # ── EASY (structural/semantic — expect robust) ──
    {
        "name": "f2_01_simple_object",
        "ref_color": "red",
        "ref_size": (512, 512),
        "prompt": "A red cube sitting on a white table with soft lighting.",
        "edit_type": "simple_object",
        "difficulty": "easy",
    },
    {
        "name": "f2_02_object_modify",
        "ref_color": "blue",
        "ref_size": (512, 512),
        "prompt": "A shiny blue sphere on a wooden desk with a warm lamp in the background.",
        "edit_type": "object_modification",
        "difficulty": "easy",
    },
    {
        "name": "f2_03_background_change",
        "ref_color": "lightgray",
        "ref_size": (768, 512),
        "prompt": "A white cat sitting on a windowsill overlooking a rainy city street.",
        "edit_type": "background_change",
        "difficulty": "easy",
    },
    {
        "name": "f2_04_arrangement",
        "ref_color": "green",
        "ref_size": (512, 512),
        "prompt": "Three colorful geometric shapes arranged in a triangle pattern on a gray surface.",
        "edit_type": "arrangement",
        "difficulty": "easy",
    },
    # ── MEDIUM (moderate detail — expect some degradation) ──
    {
        "name": "f2_05_portrait",
        "ref_color": "peachpuff",
        "ref_size": (512, 768),
        "prompt": "A professional headshot portrait of a smiling woman with short brown hair.",
        "edit_type": "portrait",
        "difficulty": "medium",
    },
    {
        "name": "f2_06_identity_edit",
        "ref_color": "lightblue",
        "ref_size": (512, 768),
        "prompt": "A person wearing a bright red jacket standing in front of a brick wall.",
        "edit_type": "identity_preserving",
        "difficulty": "medium",
    },
    {
        "name": "f2_07_interior",
        "ref_color": "ivory",
        "ref_size": (768, 512),
        "prompt": "A modern minimalist living room with a large window, white walls, and wooden furniture.",
        "edit_type": "interior_design",
        "difficulty": "medium",
    },
    {
        "name": "f2_08_text_edit",
        "ref_color": "navy",
        "ref_size": (640, 480),
        "prompt": "A neon sign reading OPEN glowing in pink and blue against a dark brick wall.",
        "edit_type": "text_editing",
        "difficulty": "medium",
    },
    # ── HARD (fine detail / texture / style — expect degradation) ──
    {
        "name": "f2_09_style_transfer",
        "ref_color": "beige",
        "ref_size": (512, 512),
        "prompt": "A watercolor painting of a Japanese garden with cherry blossoms and a stone lantern.",
        "edit_type": "style_transfer",
        "difficulty": "hard",
    },
    {
        "name": "f2_10_fine_texture",
        "ref_color": "cyan",
        "ref_size": (640, 480),
        "prompt": "A close-up of intricate lace fabric with delicate floral embroidery patterns.",
        "edit_type": "fine_texture",
        "difficulty": "hard",
    },
    {
        "name": "f2_11_complex_scene",
        "ref_color": "coral",
        "ref_size": (768, 512),
        "prompt": "A bustling street market with colorful stalls, hanging lanterns, and many people walking between vendors.",
        "edit_type": "complex_scene",
        "difficulty": "hard",
    },
    {
        "name": "f2_12_nature_detail",
        "ref_color": "darkgreen",
        "ref_size": (768, 512),
        "prompt": "A dense forest with tall trees, ferns on the forest floor, and sunlight filtering through the canopy.",
        "edit_type": "nature_detail",
        "difficulty": "hard",
    },
]


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def compute_ssim(img_a, img_b):
    """Compute SSIM between two PIL images."""
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.LANCZOS)
    a = np.array(img_a).astype(np.float64)
    b = np.array(img_b).astype(np.float64)
    return float(ssim(a, b, channel_axis=2, data_range=255.0))


def compute_lpips_distance(img_a, img_b, lpips_fn):
    """Compute LPIPS perceptual distance between two PIL images."""
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.LANCZOS)
    a = torch.from_numpy(np.array(img_a)).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    b = torch.from_numpy(np.array(img_b)).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    with torch.no_grad():
        d = lpips_fn(a.cuda(), b.cuda())
    return float(d.item())


# ═══════════════════════════════════════════════════════════════
#  LOAD PIPELINE
# ═══════════════════════════════════════════════════════════════

print("Loading FLUX Kontext-dev pipeline...")
from diffusers import FluxKontextPipeline

pipeline = FluxKontextPipeline.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
pipeline.to("cuda")
vram_after_load = torch.cuda.memory_allocated() / 1e9
print(f"Pipeline loaded. VRAM: {vram_after_load:.2f} GB")

flux_mod = importlib.import_module("diffusers.pipelines.flux.pipeline_flux_kontext")
calculate_shift = flux_mod.calculate_shift
retrieve_timesteps = flux_mod.retrieve_timesteps
PREFERRED_KONTEXT_RESOLUTIONS = flux_mod.PREFERRED_KONTEXT_RESOLUTIONS

multiple_of = pipeline.vae_scale_factor * 2

# Load LPIPS
print("Loading LPIPS model...")
lpips_fn = lpips.LPIPS(net='alex').cuda()
print("LPIPS ready.")


# ═══════════════════════════════════════════════════════════════
#  INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════

def run_inference(ref_image, prompt, seed, cached_steps=None):
    """Run FLUX Kontext-dev inference with optional temporal caching.

    cached_steps=None or empty set: baseline (no caching)
    cached_steps=set: reuse previous step's noise prediction for these steps
    """
    pipeline._guidance_scale = GUIDANCE_SCALE
    pipeline._joint_attention_kwargs = {}
    pipeline._current_timestep = None
    pipeline._interrupt = False

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        # Image preprocessing
        image_height, image_width = pipeline.image_processor.get_default_height_width(ref_image)
        aspect_ratio = image_width / image_height
        _, image_width, image_height = min(
            (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
        )
        image_width = image_width // multiple_of * multiple_of
        image_height = image_height // multiple_of * multiple_of

        height, width = 1024, 1024
        ar = width / height
        width = round((1024**2 * ar) ** 0.5)
        height = round((1024**2 / ar) ** 0.5)
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        image_resized = pipeline.image_processor.resize(ref_image, image_height, image_width)
        image_preprocessed = pipeline.image_processor.preprocess(image_resized, image_height, image_width)

        # Text encoding
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None,
            device="cuda", num_images_per_prompt=1, max_sequence_length=512,
        )

        # Prepare latents
        num_channels_latents = pipeline.transformer.config.in_channels // 4
        latents, image_latents, latent_ids, image_ids = pipeline.prepare_latents(
            image_preprocessed, 1, num_channels_latents, height, width,
            prompt_embeds.dtype, "cuda",
            torch.Generator("cuda").manual_seed(seed),
        )
        if image_ids is not None:
            latent_ids = torch.cat([latent_ids, image_ids], dim=0)

        noise_len = latents.shape[1]

        # Scheduler
        sigmas = np.linspace(1.0, 1.0 / NUM_STEPS, NUM_STEPS)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            pipeline.scheduler.config.get("base_image_seq_len", 256),
            pipeline.scheduler.config.get("max_image_seq_len", 4096),
            pipeline.scheduler.config.get("base_shift", 0.5),
            pipeline.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            pipeline.scheduler, NUM_STEPS, "cuda", sigmas=sigmas, mu=mu,
        )
        pipeline.scheduler.set_begin_index(0)

        guidance = torch.full([1], GUIDANCE_SCALE, device="cuda", dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])

        # Denoising loop
        prev_noise_pred = None

        torch.cuda.synchronize()
        t_loop_start = time.perf_counter()

        for i, t in enumerate(timesteps):
            is_cached = cached_steps is not None and (i in cached_steps)

            if is_cached and prev_noise_pred is not None:
                noise_pred = prev_noise_pred
            else:
                latent_model_input = torch.cat([latents, image_latents], dim=1)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred_full = pipeline.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=pipeline.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred_full[:, :noise_len]
                prev_noise_pred = noise_pred

            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        torch.cuda.synchronize()
        t_loop_end = time.perf_counter()
        loop_time = t_loop_end - t_loop_start

        # VAE decode
        latents_dec = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
        latents_dec = (latents_dec / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        decoded = pipeline.vae.decode(latents_dec, return_dict=False)[0]
        output_image = pipeline.image_processor.postprocess(decoded, output_type="pil")[0]

    peak_vram = torch.cuda.max_memory_reserved() / 1e9
    return output_image, loop_time, peak_vram


# ═══════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════

# Warmup
print("\nWarmup...")
ref_warmup = Image.new("RGB", (512, 512), "red")
with torch.no_grad():
    _ = pipeline(
        image=ref_warmup, prompt="warmup",
        num_inference_steps=NUM_STEPS, guidance_scale=GUIDANCE_SCALE,
        generator=torch.Generator("cuda").manual_seed(SEED),
    )
print("Warmup done.\n")

# Results storage
all_results = []
baseline_images = {}
RESULTS_FILE = os.path.join(OUTPUT_DIR, "f2_results.json")

print("=" * 80)
print(f"F2 VALIDATION: {len(TEST_SUITE)} cases x {len(MODES)} configs = {len(TEST_SUITE)*len(MODES)} runs")
print("=" * 80)

for tc_idx, tc in enumerate(TEST_SUITE):
    tc_name = tc["name"]
    ref_img = Image.new("RGB", tc["ref_size"], tc["ref_color"])

    print(f"\n{'─'*80}")
    print(f"[{tc_idx+1}/{len(TEST_SUITE)}] {tc_name}  ({tc['edit_type']}, {tc['difficulty']})")
    print(f"  Prompt: {tc['prompt'][:70]}...")
    print(f"{'─'*80}")

    for mode in MODES:
        cached_steps = SCHEDULES[mode]["cached_steps"] if SCHEDULES[mode]["cached_steps"] else None

        print(f"  {mode:<14s} ... ", end="", flush=True)
        img_out, t_loop, vram = run_inference(ref_img, tc["prompt"], SEED, cached_steps)

        # Save image
        img_path = os.path.join(OUTPUT_DIR, f"{tc_name}_{mode}.png")
        img_out.save(img_path)

        result = {
            "test_case": tc_name,
            "edit_type": tc["edit_type"],
            "difficulty": tc["difficulty"],
            "mode": mode,
            "time_s": round(t_loop, 3),
            "vram_gb": round(vram, 2),
        }

        if mode == "baseline":
            baseline_images[tc_name] = img_out
            result["speedup"] = 1.0
            result["ssim"] = 1.0
            result["lpips"] = 0.0
            print(f"time={t_loop:.2f}s, vram={vram:.1f}GB")
        else:
            baseline_t = [r for r in all_results if r["test_case"] == tc_name and r["mode"] == "baseline"][0]["time_s"]
            speedup = baseline_t / t_loop
            ssim_val = compute_ssim(baseline_images[tc_name], img_out)
            lpips_val = compute_lpips_distance(baseline_images[tc_name], img_out, lpips_fn)
            result["speedup"] = round(speedup, 3)
            result["ssim"] = round(ssim_val, 4)
            result["lpips"] = round(lpips_val, 4)
            print(f"time={t_loop:.2f}s, speedup={speedup:.2f}x, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")

        all_results.append(result)

    # Incremental save after each test case
    with open(RESULTS_FILE, "w") as f:
        json.dump({"config": {
            "model": "FLUX.1 Kontext-dev",
            "num_steps": NUM_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "seed": SEED,
            "test_cases": len(TEST_SUITE),
        }, "results": all_results}, f, indent=2)


# ═══════════════════════════════════════════════════════════════
#  ANALYSIS
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("FULL RESULTS TABLE")
print("=" * 80)
print(f"{'Case':<25s} {'Type':<18s} {'Diff':<7s} {'Mode':<14s} {'Time':>6s} {'Speed':>6s} {'SSIM':>7s} {'LPIPS':>7s}")
print("-" * 95)
for r in all_results:
    print(f"{r['test_case']:<25s} {r['edit_type']:<18s} {r['difficulty']:<7s} "
          f"{r['mode']:<14s} {r['time_s']:6.2f} {r['speedup']:6.2f} {r['ssim']:7.4f} {r['lpips']:7.4f}")

# Per-schedule aggregates
print("\n" + "=" * 80)
print("PER-SCHEDULE AGGREGATES")
print("=" * 80)
for mode in MODES:
    mode_results = [r for r in all_results if r["mode"] == mode]
    if mode == "baseline":
        times = [r["time_s"] for r in mode_results]
        print(f"\n  {SCHEDULES[mode]['label']}:")
        print(f"    Mean time: {np.mean(times):.2f}s (range {min(times):.2f}-{max(times):.2f})")
        continue

    speedups = [r["speedup"] for r in mode_results]
    ssims = [r["ssim"] for r in mode_results]
    lpipses = [r["lpips"] for r in mode_results]

    n_cached = len(SCHEDULES[mode]["cached_steps"])
    n_full = NUM_STEPS - n_cached
    theoretical = NUM_STEPS / n_full

    print(f"\n  {SCHEDULES[mode]['label']}:")
    print(f"    Theoretical speedup: {theoretical:.2f}x")
    print(f"    Mean speedup: {np.mean(speedups):.3f}x (std={np.std(speedups):.3f})")
    print(f"    Mean SSIM:    {np.mean(ssims):.4f} (median={np.median(ssims):.4f}, std={np.std(ssims):.4f})")
    print(f"    SSIM range:   {min(ssims):.4f} - {max(ssims):.4f}")
    print(f"    Mean LPIPS:   {np.mean(lpipses):.4f} (median={np.median(lpipses):.4f})")
    print(f"    LPIPS range:  {min(lpipses):.4f} - {max(lpipses):.4f}")

# Quality distribution
print("\n" + "=" * 80)
print("QUALITY DISTRIBUTION (Conservative)")
print("=" * 80)
cons_results = [r for r in all_results if r["mode"] == "conservative"]
cons_ssims = sorted([r["ssim"] for r in cons_results])

thresholds = [0.95, 0.90, 0.85, 0.80]
for t in thresholds:
    n_above = sum(1 for s in cons_ssims if s >= t)
    pct = 100 * n_above / len(cons_ssims)
    print(f"  SSIM >= {t:.2f}: {n_above}/{len(cons_ssims)} ({pct:.0f}%)")
n_below_80 = sum(1 for s in cons_ssims if s < 0.80)
print(f"  SSIM <  0.80: {n_below_80}/{len(cons_ssims)} ({100*n_below_80/len(cons_ssims):.0f}%)")

print(f"\n  Percentiles:")
for p in [10, 25, 50, 75, 90]:
    print(f"    P{p}: {np.percentile(cons_ssims, p):.4f}")

# By difficulty
print("\n" + "=" * 80)
print("QUALITY BY DIFFICULTY")
print("=" * 80)
for mode in ["conservative", "aggressive"]:
    print(f"\n  {SCHEDULES[mode]['label']}:")
    mode_results = [r for r in all_results if r["mode"] == mode]
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in mode_results if r["difficulty"] == diff]
        if diff_results:
            ssims = [r["ssim"] for r in diff_results]
            lpipses = [r["lpips"] for r in diff_results]
            print(f"    {diff:<8s}: mean SSIM={np.mean(ssims):.4f}, mean LPIPS={np.mean(lpipses):.4f}  "
                  f"(n={len(diff_results)}, range={min(ssims):.4f}-{max(ssims):.4f})")

# Failure analysis
print("\n" + "=" * 80)
print("FAILURE ANALYSIS (Conservative, sorted by SSIM)")
print("=" * 80)
cons_sorted = sorted(cons_results, key=lambda r: r["ssim"])
print(f"  {'Case':<25s} {'Type':<18s} {'Diff':<7s} {'SSIM':>7s} {'LPIPS':>7s} {'Assessment'}")
print(f"  {'-'*25} {'-'*18} {'-'*7} {'-'*7} {'-'*7} {'-'*20}")
for r in cons_sorted:
    if r["ssim"] < 0.80:
        label = "SEVERE DEGRADATION"
    elif r["ssim"] < 0.85:
        label = "Notable degradation"
    elif r["ssim"] < 0.90:
        label = "Moderate difference"
    elif r["ssim"] < 0.95:
        label = "Minor difference"
    else:
        label = "Excellent"
    print(f"  {r['test_case']:<25s} {r['edit_type']:<18s} {r['difficulty']:<7s} "
          f"{r['ssim']:7.4f} {r['lpips']:7.4f} {label}")

# Cross-model comparison
print("\n" + "=" * 80)
print("CROSS-MODEL COMPARISON")
print("=" * 80)
cons_ssims = [r["ssim"] for r in cons_results]
cons_lpipses = [r["lpips"] for r in cons_results]
cons_speedups = [r["speedup"] for r in cons_results]

aggr_results = [r for r in all_results if r["mode"] == "aggressive"]
aggr_ssims = [r["ssim"] for r in aggr_results]
aggr_lpipses = [r["lpips"] for r in aggr_results]
aggr_speedups = [r["speedup"] for r in aggr_results]

print(f"\n  {'Metric':<25s} {'Qwen M9 Combined':>18s} {'Kontext F2 Cons':>18s} {'Kontext F2 Aggr':>18s}")
print(f"  {'-'*25} {'-'*18} {'-'*18} {'-'*18}")
print(f"  {'Speedup':<25s} {'2.35x':>18s} {f'{np.mean(cons_speedups):.2f}x':>18s} {f'{np.mean(aggr_speedups):.2f}x':>18s}")
print(f"  {'Mean SSIM':<25s} {'0.836':>18s} {f'{np.mean(cons_ssims):.4f}':>18s} {f'{np.mean(aggr_ssims):.4f}':>18s}")
print(f"  {'Median SSIM':<25s} {'—':>18s} {f'{np.median(cons_ssims):.4f}':>18s} {f'{np.median(aggr_ssims):.4f}':>18s}")
print(f"  {'Mean LPIPS':<25s} {'0.215':>18s} {f'{np.mean(cons_lpipses):.4f}':>18s} {f'{np.mean(aggr_lpipses):.4f}':>18s}")
print(f"  {'Test cases':<25s} {'18':>18s} {'12':>18s} {'12':>18s}")
print(f"  {'Acceleration axes':<25s} {'CFG + temporal':>18s} {'temporal only':>18s} {'temporal only':>18s}")
print(f"  {'Baseline time':<25s} {'47.4s':>18s} {'~9s':>18s} {'~9s':>18s}")

# SSIM > 0.90 and < 0.80 comparison
n_above_90 = sum(1 for s in cons_ssims if s >= 0.90)
n_below_80 = sum(1 for s in cons_ssims if s < 0.80)
print(f"\n  Kontext F2 Conservative distribution:")
print(f"    SSIM >= 0.90: {n_above_90}/{len(cons_ssims)} ({100*n_above_90/len(cons_ssims):.0f}%)")
print(f"    SSIM <  0.80: {n_below_80}/{len(cons_ssims)} ({100*n_below_80/len(cons_ssims):.0f}%)")
print(f"  Qwen M9 Combined distribution:")
print(f"    SSIM >= 0.90: 7/18 (39%)")
print(f"    SSIM <  0.80: 6/18 (33%)")

# Qwen failure pattern comparison
print("\n" + "=" * 80)
print("FAILURE PATTERN COMPARISON (Qwen vs Kontext)")
print("=" * 80)
print("""
  Qwen M9 hard cases (SSIM < 0.80): complex_scene (0.637), style_transfer (0.642),
    material_texture (0.665), nature_scene (0.687), identity_preserving (0.753),
    interior_design (0.768)

  Kontext F2 hard cases (sorted above): see FAILURE ANALYSIS table

  Pattern match assessment:
    - If BOTH models degrade on style/texture/complex → method limitation
    - If Kontext degrades on DIFFERENT cases → model-specific boundary
""")

# Save CSV
csv_path = os.path.join(RESULTS_DIR, "f2_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["test_case", "edit_type", "difficulty", "mode",
                                           "time_s", "speedup", "ssim", "lpips", "vram_gb"])
    writer.writeheader()
    writer.writerows(all_results)
print(f"\nCSV saved to {csv_path}")

# Save final JSON
with open(RESULTS_FILE, "w") as f:
    json.dump({
        "config": {
            "model": "FLUX.1 Kontext-dev",
            "num_steps": NUM_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "seed": SEED,
            "test_cases": len(TEST_SUITE),
            "modes": MODES,
            "schedules": {
                k: {"label": v["label"], "cached_steps": sorted(v["cached_steps"]),
                     "full_steps": NUM_STEPS - len(v["cached_steps"])}
                for k, v in SCHEDULES.items()
            },
        },
        "results": all_results,
        "aggregates": {
            "conservative": {
                "mean_speedup": round(np.mean(cons_speedups), 3),
                "mean_ssim": round(np.mean(cons_ssims), 4),
                "median_ssim": round(float(np.median(cons_ssims)), 4),
                "std_ssim": round(float(np.std(cons_ssims)), 4),
                "mean_lpips": round(np.mean(cons_lpipses), 4),
                "median_lpips": round(float(np.median(cons_lpipses)), 4),
                "ssim_range": [round(min(cons_ssims), 4), round(max(cons_ssims), 4)],
                "lpips_range": [round(min(cons_lpipses), 4), round(max(cons_lpipses), 4)],
            },
            "aggressive": {
                "mean_speedup": round(np.mean(aggr_speedups), 3),
                "mean_ssim": round(np.mean(aggr_ssims), 4),
                "median_ssim": round(float(np.median(aggr_ssims)), 4),
                "std_ssim": round(float(np.std(aggr_ssims)), 4),
                "mean_lpips": round(np.mean(aggr_lpipses), 4),
                "median_lpips": round(float(np.median(aggr_lpipses)), 4),
                "ssim_range": [round(min(aggr_ssims), 4), round(max(aggr_ssims), 4)],
                "lpips_range": [round(min(aggr_lpipses), 4), round(max(aggr_lpipses), 4)],
            },
        },
    }, f, indent=2)
print(f"JSON saved to {RESULTS_FILE}")

print("\n" + "=" * 80)
print("F2 VALIDATION COMPLETE")
print("=" * 80)
