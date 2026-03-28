#!/usr/bin/env python3
"""F1 Phase 4: Temporal Caching Prototype for FLUX Kontext-dev

Tests temporal step-caching on the plateau identified in Phase 3 (steps 4-23).
Since Kontext is guidance-distilled (single pass), caching reuses the previous
step's noise prediction directly — no CFG pass to skip.

Two schedules:
  A. Aggressive: cache steps 4-23 (20 cached, 8 full). Theoretical 28/8 = 3.5x.
  B. Conservative: cache steps 6-20 (15 cached, 13 full). Theoretical 28/13 = 2.15x.

5 test cases with identical seeds, measuring wall-clock time and SSIM vs baseline.
"""

import torch
import time
import json
import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import importlib

# ─── Configuration ───
WEIGHTS = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/weights/FLUX-Kontext-dev"
OUTPUT_DIR = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/outputs/flux"
NUM_STEPS = 28
GUIDANCE_SCALE = 2.5
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

SCHEDULES = [
    {
        "name": "aggressive",
        "label": "Aggressive (cache steps 4-23)",
        "cached_steps": set(range(4, 24)),
    },
    {
        "name": "conservative",
        "label": "Conservative (cache steps 6-20)",
        "cached_steps": set(range(6, 21)),
    },
]

# 5 test cases: varied reference images and prompts
TEST_CASES = [
    ("tc1_red_cube", "red", (512, 512),
     "A red cube sitting on a white table with soft lighting."),
    ("tc2_landscape", "green", (768, 512),
     "A beautiful green meadow under a bright blue sky with fluffy white clouds."),
    ("tc3_complex", "purple", (512, 512),
     "A purple sunset behind mountains with a golden river reflecting the light."),
    ("tc4_portrait", "orange", (512, 768),
     "A portrait of a person wearing an orange scarf in winter."),
    ("tc5_texture", "cyan", (640, 480),
     "A cyan ceramic vase with intricate floral patterns on a wooden shelf."),
]


def compute_ssim(img_a, img_b):
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.LANCZOS)
    a = np.array(img_a).astype(np.float64)
    b = np.array(img_b).astype(np.float64)
    return float(ssim(a, b, channel_axis=2, data_range=255.0))


# ─── Load Pipeline ───
print("Loading pipeline...")
from diffusers import FluxKontextPipeline

pipeline = FluxKontextPipeline.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
pipeline.to("cuda")
print(f"Pipeline loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

flux_mod = importlib.import_module("diffusers.pipelines.flux.pipeline_flux_kontext")
calculate_shift = flux_mod.calculate_shift
retrieve_timesteps = flux_mod.retrieve_timesteps
PREFERRED_KONTEXT_RESOLUTIONS = flux_mod.PREFERRED_KONTEXT_RESOLUTIONS

multiple_of = pipeline.vae_scale_factor * 2


def run_inference(ref_image, prompt, seed, cached_steps=None):
    """Run inference with optional temporal caching.

    cached_steps=None: baseline (no caching)
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


# ═══════════════════════════════════════════
#  MAIN EXPERIMENT
# ═══════════════════════════════════════════

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

# Run baselines
print("=" * 70)
print("BASELINES")
print("=" * 70)
baseline_results = []
baseline_images = {}

for tc_name, color, size, prompt in TEST_CASES:
    print(f"\n  {tc_name} — BASELINE")
    ref_img = Image.new("RGB", size, color)
    img_out, t_loop, vram = run_inference(ref_img, prompt, SEED)
    print(f"    Time: {t_loop:.2f}s, VRAM: {vram:.2f} GB")

    path = os.path.join(OUTPUT_DIR, f"f1_{tc_name}_baseline.png")
    img_out.save(path)
    baseline_images[tc_name] = img_out
    baseline_results.append({
        "test_case": tc_name,
        "baseline_time_s": round(t_loop, 2),
        "vram_gb": round(vram, 2),
    })

# Run each schedule
all_results = {}
for sched in SCHEDULES:
    sched_name = sched["name"]
    print(f"\n{'='*70}")
    print(f"SCHEDULE: {sched['label']}")
    print(f"{'='*70}")

    sched_results = []
    for tc_idx, (tc_name, color, size, prompt) in enumerate(TEST_CASES):
        print(f"\n  {tc_name} — {sched_name}")
        ref_img = Image.new("RGB", size, color)
        img_out, t_loop, vram = run_inference(ref_img, prompt, SEED, sched["cached_steps"])

        ssim_val = compute_ssim(baseline_images[tc_name], img_out)
        baseline_t = baseline_results[tc_idx]["baseline_time_s"]
        speedup = baseline_t / t_loop

        print(f"    Time: {t_loop:.2f}s (speedup {speedup:.2f}x), SSIM: {ssim_val:.4f}")

        path = os.path.join(OUTPUT_DIR, f"f1_{tc_name}_{sched_name}.png")
        img_out.save(path)

        sched_results.append({
            "test_case": tc_name,
            "baseline_time_s": baseline_t,
            "cached_time_s": round(t_loop, 2),
            "speedup": round(speedup, 3),
            "ssim": round(ssim_val, 4),
            "vram_gb": round(vram, 2),
        })

    all_results[sched_name] = sched_results

# ═══════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

for sched in SCHEDULES:
    sched_name = sched["name"]
    results = all_results[sched_name]

    print(f"\n── {sched['label']} ──")
    n_cached = len(sched["cached_steps"])
    n_full = NUM_STEPS - n_cached
    theoretical = NUM_STEPS / n_full
    print(f"   {n_cached} cached + {n_full} full steps. Theoretical speedup: {theoretical:.2f}x")

    print(f"   {'Test Case':<20s} {'Base(s)':>8s} {'Cache(s)':>8s} {'Speedup':>8s} {'SSIM':>7s}")
    print(f"   {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
    for r in results:
        print(f"   {r['test_case']:<20s} {r['baseline_time_s']:8.2f} {r['cached_time_s']:8.2f} "
              f"{r['speedup']:8.3f} {r['ssim']:7.4f}")

    speedups = [r["speedup"] for r in results]
    ssims = [r["ssim"] for r in results]
    print(f"\n   Mean speedup: {np.mean(speedups):.3f}x  "
          f"Mean SSIM: {np.mean(ssims):.4f}  "
          f"SSIM range: {min(ssims):.4f}-{max(ssims):.4f}")

# Cross-schedule comparison
print(f"\n{'='*70}")
print("CROSS-SCHEDULE COMPARISON")
print(f"{'='*70}")
print(f"{'Test Case':<20s} {'Aggr Speed':>10s} {'Aggr SSIM':>10s} "
      f"{'Cons Speed':>10s} {'Cons SSIM':>10s}")
print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for tc_idx in range(len(TEST_CASES)):
    tc_name = TEST_CASES[tc_idx][0]
    a = all_results["aggressive"][tc_idx]
    b = all_results["conservative"][tc_idx]
    print(f"{tc_name:<20s} {a['speedup']:10.3f} {a['ssim']:10.4f} "
          f"{b['speedup']:10.3f} {b['ssim']:10.4f}")

# Quality assessment
print("\nQuality Assessment:")
for sched in SCHEDULES:
    sched_name = sched["name"]
    print(f"\n  {sched['label']}:")
    for r in all_results[sched_name]:
        if r["ssim"] < 0.80:
            label = "SEVERE DEGRADATION"
        elif r["ssim"] < 0.90:
            label = "NOTABLE degradation"
        elif r["ssim"] < 0.95:
            label = "moderate difference"
        else:
            label = "good quality"
        print(f"    {r['test_case']}: SSIM={r['ssim']:.4f} — {label}")

# Transfer verdict
print(f"\n{'='*70}")
print("TRANSFER VERDICT: Temporal Caching from Qwen → Kontext")
print(f"{'='*70}")
mean_aggr_ssim = np.mean([r["ssim"] for r in all_results["aggressive"]])
mean_cons_ssim = np.mean([r["ssim"] for r in all_results["conservative"]])
mean_aggr_speed = np.mean([r["speedup"] for r in all_results["aggressive"]])
mean_cons_speed = np.mean([r["speedup"] for r in all_results["conservative"]])

print(f"\n  Temporal plateau: CONFIRMED (steps 4-23, 74% of trajectory)")
print(f"  Same three-phase pattern as Qwen: convergence → plateau → divergence")
print(f"  Conservative caching: {mean_cons_speed:.2f}x speedup, {mean_cons_ssim:.4f} SSIM")
print(f"  Aggressive caching:   {mean_aggr_speed:.2f}x speedup, {mean_aggr_ssim:.4f} SSIM")
print(f"\n  Temporal caching: {'TRANSFERS' if mean_cons_ssim > 0.85 else 'DOES NOT TRANSFER'}")

# Save results
results_json = {
    "config": {
        "model": "FLUX.1 Kontext-dev",
        "num_steps": NUM_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "seed": SEED,
        "schedules": {
            s["name"]: {
                "label": s["label"],
                "cached_steps": sorted(s["cached_steps"]),
                "full_steps": NUM_STEPS - len(s["cached_steps"]),
                "theoretical_speedup": round(NUM_STEPS / (NUM_STEPS - len(s["cached_steps"])), 2),
            } for s in SCHEDULES
        },
    },
    "baseline": baseline_results,
    "per_schedule": all_results,
    "transfer_verdict": {
        "temporal_plateau": "confirmed",
        "plateau_steps": "4-23",
        "three_phase_pattern": True,
        "conservative_speedup": round(mean_cons_speed, 3),
        "conservative_ssim": round(mean_cons_ssim, 4),
        "aggressive_speedup": round(mean_aggr_speed, 3),
        "aggressive_ssim": round(mean_aggr_ssim, 4),
        "transfers": bool(mean_cons_ssim > 0.85),
    },
}

json_path = os.path.join(OUTPUT_DIR, "f1_temporal_cache_results.json")
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\nResults saved to {json_path}")
print("\nDone.")
