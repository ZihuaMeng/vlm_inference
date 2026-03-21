#!/usr/bin/env python3
"""Milestone 5: CFG-Skip Prototype + Robustness

Implements a minimal CFG-skip mechanism that skips the unconditional
transformer pass for denoising steps where M4 showed cond ≈ uncond
(cos >0.999). For skipped steps, the conditional output is used directly
as the noise prediction (no CFG combination).

Skip schedule (informed by M4 redundancy data):
  - Steps 0–2: Full CFG (cond/uncond divergence, cos 0.968–0.995)
  - Steps 3–38: SKIP unconditional pass (cos >0.999)
  - Step 39: Full CFG (final step divergence, cos=0.993)

For each of 5 test cases, generates:
  (a) Baseline output (40 steps, full CFG every step)
  (b) CFG-skip output (same seed, skip schedule above)
Then measures wall-clock time, SSIM, MSE, PSNR.
"""

import torch
import time
import json
import os
import contextlib
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ─── Configuration ───
WEIGHTS = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/weights/Qwen-Image-Edit-2511"
OUTPUT_DIR = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/outputs/m5_cfg_skip"
NUM_STEPS = 40
TRUE_CFG_SCALE = 4.0
SEED = 42

# Steps where we run full CFG (both cond + uncond passes)
FULL_CFG_STEPS = set(range(0, 3)) | {39}  # steps 0, 1, 2, 39
# All other steps: skip the unconditional pass

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Test Cases ───
# Each: (name, images_spec, prompt, negative_prompt)
# images_spec is a list of (color, size) tuples to create solid test images
TEST_CASES = [
    (
        "tc1_2img_objects",
        [("red", (512, 512)), ("blue", (512, 512))],
        "A red object and a blue object sitting side by side on a white table.",
        " ",
    ),
    (
        "tc2_1img_landscape",
        [("green", (768, 512))],
        "A beautiful green meadow under a bright blue sky with fluffy white clouds.",
        " ",
    ),
    (
        "tc3_2img_scene",
        [("yellow", (512, 512)), ("purple", (512, 512))],
        "A yellow sun setting behind purple mountains with a river in the foreground.",
        " ",
    ),
    (
        "tc4_3img_arrange",
        [("red", (512, 512)), ("green", (512, 512)), ("blue", (512, 512))],
        "Three colorful geometric shapes arranged in a triangle pattern on a gray surface.",
        " ",
    ),
    (
        "tc5_2img_texture",
        [("orange", (640, 480)), ("cyan", (640, 480))],
        "An orange cat sleeping next to a cyan cushion on a wooden floor.",
        " ",
    ),
]


def make_images(specs):
    """Create solid-color PIL images from specs."""
    return [Image.new("RGB", size, color) for color, size in specs]


# ─── Quality Metrics ───
def compute_metrics(img_a, img_b):
    """Compute SSIM, MSE, PSNR between two PIL images."""
    # Resize to same dimensions if needed
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.LANCZOS)

    a = np.array(img_a).astype(np.float64)
    b = np.array(img_b).astype(np.float64)

    # MSE
    mse = np.mean((a - b) ** 2)

    # PSNR
    if mse < 1e-10:
        psnr = float("inf")
    else:
        psnr = 10.0 * np.log10(255.0**2 / mse)

    # SSIM (multichannel)
    ssim_val = ssim(a, b, channel_axis=2, data_range=255.0)

    return {"ssim": float(ssim_val), "mse": float(mse), "psnr": float(psnr)}


# ─── Load Pipeline ───
print("Loading pipeline...")
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
pipeline.to("cuda")
vram_after_load = torch.cuda.memory_allocated() / 1e9
print(f"Pipeline loaded. VRAM: {vram_after_load:.2f} GB")

# ─── Import pipeline helpers ───
import importlib

pm = importlib.import_module(
    "diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus"
)
calculate_dimensions = pm.calculate_dimensions
CONDITION_IMAGE_SIZE = pm.CONDITION_IMAGE_SIZE
VAE_IMAGE_SIZE = pm.VAE_IMAGE_SIZE
calculate_shift = pm.calculate_shift
retrieve_timesteps = pm.retrieve_timesteps


def prepare_inputs(image_list):
    """Prepare all inputs for the denoising loop (text enc, VAE enc, scheduler)."""
    image_size = image_list[-1].size
    calculated_width, calculated_height = calculate_dimensions(
        1024 * 1024, image_size[0] / image_size[1]
    )
    height, width = calculated_height, calculated_width
    multiple_of = pipeline.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    condition_images = []
    vae_images = []
    vae_image_sizes = []
    for img in image_list:
        iw, ih = img.size
        cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, iw / ih)
        vw, vh = calculate_dimensions(VAE_IMAGE_SIZE, iw / ih)
        vae_image_sizes.append((vw, vh))
        condition_images.append(pipeline.image_processor.resize(img, ch, cw))
        vae_images.append(
            pipeline.image_processor.preprocess(img, vh, vw).unsqueeze(2)
        )

    return condition_images, vae_images, vae_image_sizes, height, width


def run_inference(image_list, prompt, negative_prompt, seed, cfg_skip=False):
    """Run a single inference pass. If cfg_skip=True, skip uncond for designated steps.

    Returns (output_image, wall_time_s, peak_vram_gb).
    """
    pipeline._guidance_scale = 1.0
    pipeline._attention_kwargs = {}
    pipeline._current_timestep = None
    pipeline._interrupt = False

    torch.cuda.reset_peak_memory_stats()

    condition_images, vae_images, vae_image_sizes, height, width = prepare_inputs(
        image_list
    )

    with torch.no_grad():
        # Text encoding
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            image=condition_images,
            prompt=prompt,
            prompt_embeds=None,
            prompt_embeds_mask=None,
            device="cuda",
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        negative_prompt_embeds, negative_prompt_embeds_mask = pipeline.encode_prompt(
            image=condition_images,
            prompt=negative_prompt,
            prompt_embeds=None,
            prompt_embeds_mask=None,
            device="cuda",
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

        # VAE encoding / prepare latents
        num_channels_latents = pipeline.transformer.config.in_channels // 4
        generator = torch.Generator("cuda").manual_seed(seed)
        latents, image_latents = pipeline.prepare_latents(
            vae_images,
            1,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            "cuda",
            generator,
            None,
        )
        img_shapes = [
            [
                (
                    1,
                    height // pipeline.vae_scale_factor // 2,
                    width // pipeline.vae_scale_factor // 2,
                ),
                *[
                    (
                        1,
                        vh // pipeline.vae_scale_factor // 2,
                        vw // pipeline.vae_scale_factor // 2,
                    )
                    for vw, vh in vae_image_sizes
                ],
            ]
        ]

        noise_latent_len = latents.size(1)

        # Scheduler setup
        sigmas = np.linspace(1.0, 1 / NUM_STEPS, NUM_STEPS)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            pipeline.scheduler.config.get("base_image_seq_len", 256),
            pipeline.scheduler.config.get("max_image_seq_len", 4096),
            pipeline.scheduler.config.get("base_shift", 0.5),
            pipeline.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            pipeline.scheduler, NUM_STEPS, "cuda", sigmas=sigmas, mu=mu
        )
        pipeline.scheduler.set_begin_index(0)

        guidance = None
        if pipeline.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], 1.0, device="cuda", dtype=torch.float32
            ).expand(latents.shape[0])

        has_cache_ctx = hasattr(pipeline.transformer, "cache_context")

        # ── Denoising Loop ──
        torch.cuda.synchronize()
        t_loop_start = time.perf_counter()

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents, image_latents], dim=1)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # Conditional pass (always run)
            ctx = (
                pipeline.transformer.cache_context("cond")
                if has_cache_ctx
                else contextlib.nullcontext()
            )
            with ctx:
                noise_pred = pipeline.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    attention_kwargs={},
                    return_dict=False,
                )[0]
            cond_pred = noise_pred[:, :noise_latent_len]

            # Decide whether to run unconditional pass
            skip_this_step = cfg_skip and (i not in FULL_CFG_STEPS)

            if skip_this_step:
                # Skip unconditional pass — use conditional output directly
                final_pred = cond_pred
            else:
                # Full CFG: run unconditional pass
                ctx = (
                    pipeline.transformer.cache_context("uncond")
                    if has_cache_ctx
                    else contextlib.nullcontext()
                )
                with ctx:
                    neg_noise_pred = pipeline.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        encoder_hidden_states=negative_prompt_embeds,
                        img_shapes=img_shapes,
                        attention_kwargs={},
                        return_dict=False,
                    )[0]
                uncond_pred = neg_noise_pred[:, :noise_latent_len]

                # Rescaled CFG combination
                comb_pred = uncond_pred + TRUE_CFG_SCALE * (cond_pred - uncond_pred)
                cond_norm = torch.norm(cond_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                final_pred = comb_pred * (cond_norm / noise_norm)

            # Scheduler step
            latents_dtype = latents.dtype
            latents = pipeline.scheduler.step(final_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        torch.cuda.synchronize()
        t_loop_end = time.perf_counter()
        loop_time = t_loop_end - t_loop_start

        # VAE decode
        latents_dec = pipeline._unpack_latents(
            latents, height, width, pipeline.vae_scale_factor
        )
        latents_dec = latents_dec.to(pipeline.vae.dtype)
        latents_mean = (
            torch.tensor(pipeline.vae.config.latents_mean)
            .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents_dec.device, latents_dec.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(pipeline.vae.config.latents_std)
            .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents_dec.device, latents_dec.dtype)
        )
        latents_dec = latents_dec / latents_std + latents_mean
        decoded = pipeline.vae.decode(latents_dec, return_dict=False)[0][:, :, 0]
        output_image = pipeline.image_processor.postprocess(decoded, output_type="pil")[0]

    peak_vram = torch.cuda.max_memory_reserved() / 1e9
    return output_image, loop_time, peak_vram


# ═══════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("WARMUP PASS (tc1, baseline)")
print("=" * 70)
warmup_imgs = make_images(TEST_CASES[0][1])
with torch.no_grad():
    _ = pipeline(
        image=warmup_imgs,
        prompt=TEST_CASES[0][2],
        num_inference_steps=NUM_STEPS,
        true_cfg_scale=TRUE_CFG_SCALE,
        negative_prompt=TEST_CASES[0][3],
        guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(SEED),
    )
print("Warmup done.\n")

results = []
skip_schedule_desc = (
    f"Full CFG: steps {sorted(FULL_CFG_STEPS)}; "
    f"Skip uncond: steps 3–38 ({40 - len(FULL_CFG_STEPS)} steps skipped)"
)

for tc_idx, (tc_name, img_specs, prompt, neg_prompt) in enumerate(TEST_CASES):
    print("=" * 70)
    print(f"TEST CASE {tc_idx + 1}/{len(TEST_CASES)}: {tc_name}")
    print(f"  Images: {len(img_specs)}, Prompt: {prompt[:60]}...")
    print("=" * 70)

    images = make_images(img_specs)

    # ── Baseline ──
    print("  Running BASELINE (full CFG, 40 steps)...")
    img_baseline, time_baseline, vram_baseline = run_inference(
        images, prompt, neg_prompt, SEED, cfg_skip=False
    )
    print(f"    Time: {time_baseline:.2f}s, Peak VRAM: {vram_baseline:.2f} GB")

    # ── CFG-Skip ──
    print("  Running CFG-SKIP variant...")
    img_skip, time_skip, vram_skip = run_inference(
        images, prompt, neg_prompt, SEED, cfg_skip=True
    )
    print(f"    Time: {time_skip:.2f}s, Peak VRAM: {vram_skip:.2f} GB")

    # ── Metrics ──
    speedup = time_baseline / time_skip
    metrics = compute_metrics(img_baseline, img_skip)
    print(f"    Speedup: {speedup:.2f}x")
    print(f"    SSIM: {metrics['ssim']:.4f}, MSE: {metrics['mse']:.2f}, PSNR: {metrics['psnr']:.2f} dB")

    # ── Save Images ──
    baseline_path = os.path.join(OUTPUT_DIR, f"{tc_name}_baseline.png")
    skip_path = os.path.join(OUTPUT_DIR, f"{tc_name}_cfg_skip.png")
    img_baseline.save(baseline_path)
    img_skip.save(skip_path)
    print(f"    Saved: {baseline_path}")
    print(f"    Saved: {skip_path}")

    results.append({
        "test_case": tc_name,
        "num_images": len(img_specs),
        "prompt": prompt,
        "baseline_time_s": round(time_baseline, 2),
        "skip_time_s": round(time_skip, 2),
        "speedup": round(speedup, 3),
        "vram_baseline_gb": round(vram_baseline, 2),
        "vram_skip_gb": round(vram_skip, 2),
        **{k: round(v, 4) if k != "mse" else round(v, 2) for k, v in metrics.items()},
    })
    print()

# ═══════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"\nSkip schedule: {skip_schedule_desc}\n")

# Table header
print(f"{'Test Case':<25s} {'Base(s)':>8s} {'Skip(s)':>8s} {'Speedup':>8s} "
      f"{'SSIM':>7s} {'MSE':>9s} {'PSNR':>8s} {'VRAM_B':>7s} {'VRAM_S':>7s}")
print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*9} {'-'*8} {'-'*7} {'-'*7}")
for r in results:
    psnr_str = f"{r['psnr']:.1f}" if r['psnr'] < 100 else "inf"
    print(f"{r['test_case']:<25s} {r['baseline_time_s']:8.2f} {r['skip_time_s']:8.2f} "
          f"{r['speedup']:8.3f} {r['ssim']:7.4f} {r['mse']:9.2f} {psnr_str:>8s} "
          f"{r['vram_baseline_gb']:7.2f} {r['vram_skip_gb']:7.2f}")

# Aggregates
speedups = [r["speedup"] for r in results]
ssims = [r["ssim"] for r in results]
mses = [r["mse"] for r in results]
psnrs = [r["psnr"] for r in results]

print(f"\n{'AGGREGATE':<25s} {'':>8s} {'':>8s} "
      f"{np.mean(speedups):8.3f} {np.mean(ssims):7.4f} {np.mean(mses):9.2f} "
      f"{np.mean([p for p in psnrs if p < 100]):>8.1f}")
print(f"{'  (range)':<25s} {'':>8s} {'':>8s} "
      f"{min(speedups):.2f}-{max(speedups):.2f} "
      f"{min(ssims):.3f}-{max(ssims):.3f} "
      f"{min(mses):.1f}-{max(mses):.1f}")

# Flag quality issues
print("\nQuality Assessment:")
for r in results:
    if r["ssim"] < 0.90:
        print(f"  ⚠ {r['test_case']}: SSIM={r['ssim']:.4f} — NOTABLE DEGRADATION")
    elif r["ssim"] < 0.95:
        print(f"  ~ {r['test_case']}: SSIM={r['ssim']:.4f} — moderate difference")
    else:
        print(f"  ✓ {r['test_case']}: SSIM={r['ssim']:.4f} — good quality")

# Save results JSON
results_json = {
    "config": {
        "num_steps": NUM_STEPS,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "seed": SEED,
        "full_cfg_steps": sorted(FULL_CFG_STEPS),
        "num_skipped_steps": NUM_STEPS - len(FULL_CFG_STEPS),
        "skip_schedule": skip_schedule_desc,
    },
    "per_test_case": results,
    "aggregate": {
        "mean_speedup": round(float(np.mean(speedups)), 3),
        "min_speedup": round(float(min(speedups)), 3),
        "max_speedup": round(float(max(speedups)), 3),
        "mean_ssim": round(float(np.mean(ssims)), 4),
        "min_ssim": round(float(min(ssims)), 4),
        "max_ssim": round(float(max(ssims)), 4),
        "mean_mse": round(float(np.mean(mses)), 2),
        "mean_psnr": round(float(np.mean([p for p in psnrs if p < 100])), 2),
    },
}

json_path = os.path.join(OUTPUT_DIR, "cfg_skip_results.json")
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\nResults saved to {json_path}")
print("\nDone.")
