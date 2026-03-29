#!/usr/bin/env python3
"""Milestone 5: CFG-Skip v2 — Cached Unconditional Reuse

Improvement over v1: instead of disabling CFG entirely for skip steps,
cache the most recent unconditional output and reuse it in the CFG formula.
This preserves the CFG amplification signal with a 1-step-stale uncond
prediction, which should maintain quality better than dropping CFG.

Same skip schedule: full CFG for steps 0–2 and 39, cache-reuse for 3–38.
Same 5 test cases as v1 for direct comparison.
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

FULL_CFG_STEPS = set(range(0, 3)) | {39}  # steps 0, 1, 2, 39

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Test Cases (same as v1) ───
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
    return [Image.new("RGB", size, color) for color, size in specs]


def compute_metrics(img_a, img_b):
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.LANCZOS)
    a = np.array(img_a).astype(np.float64)
    b = np.array(img_b).astype(np.float64)
    mse = np.mean((a - b) ** 2)
    psnr = 10.0 * np.log10(255.0**2 / mse) if mse > 1e-10 else float("inf")
    ssim_val = ssim(a, b, channel_axis=2, data_range=255.0)
    return {"ssim": float(ssim_val), "mse": float(mse), "psnr": float(psnr)}


# ─── Load Pipeline ───
print("Loading pipeline...")
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
pipeline.to("cuda")
print(f"Pipeline loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

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
    image_size = image_list[-1].size
    calculated_width, calculated_height = calculate_dimensions(
        1024 * 1024, image_size[0] / image_size[1]
    )
    height, width = calculated_height, calculated_width
    multiple_of = pipeline.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    condition_images, vae_images, vae_image_sizes = [], [], []
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


def run_inference(image_list, prompt, negative_prompt, seed, mode="baseline"):
    """Run inference in one of three modes:
      - "baseline": full CFG every step
      - "cfg_skip_v1": skip uncond, use cond directly (no CFG for skip steps)
      - "cfg_skip_v2": skip uncond, reuse cached uncond in CFG formula
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
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            image=condition_images, prompt=prompt,
            prompt_embeds=None, prompt_embeds_mask=None,
            device="cuda", num_images_per_prompt=1, max_sequence_length=512,
        )
        negative_prompt_embeds, negative_prompt_embeds_mask = pipeline.encode_prompt(
            image=condition_images, prompt=negative_prompt,
            prompt_embeds=None, prompt_embeds_mask=None,
            device="cuda", num_images_per_prompt=1, max_sequence_length=512,
        )

        num_channels_latents = pipeline.transformer.config.in_channels // 4
        generator = torch.Generator("cuda").manual_seed(seed)
        latents, image_latents = pipeline.prepare_latents(
            vae_images, 1, num_channels_latents, height, width,
            prompt_embeds.dtype, "cuda", generator, None,
        )
        img_shapes = [
            [
                (1, height // pipeline.vae_scale_factor // 2,
                 width // pipeline.vae_scale_factor // 2),
                *[(1, vh // pipeline.vae_scale_factor // 2,
                   vw // pipeline.vae_scale_factor // 2)
                  for vw, vh in vae_image_sizes],
            ]
        ]

        noise_latent_len = latents.size(1)

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
        cached_uncond = None  # For v2: stores last computed uncond output

        torch.cuda.synchronize()
        t_loop_start = time.perf_counter()

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents, image_latents], dim=1)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # Conditional pass (always)
            ctx = (pipeline.transformer.cache_context("cond")
                   if has_cache_ctx else contextlib.nullcontext())
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

            should_skip = mode != "baseline" and (i not in FULL_CFG_STEPS)

            if should_skip and mode == "cfg_skip_v1":
                # v1: use cond directly, no CFG
                final_pred = cond_pred

            elif should_skip and mode == "cfg_skip_v2" and cached_uncond is not None:
                # v2: reuse cached uncond in CFG formula
                comb_pred = cached_uncond + TRUE_CFG_SCALE * (cond_pred - cached_uncond)
                cond_norm = torch.norm(cond_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                final_pred = comb_pred * (cond_norm / noise_norm)

            else:
                # Full CFG: run unconditional pass
                ctx = (pipeline.transformer.cache_context("uncond")
                       if has_cache_ctx else contextlib.nullcontext())
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
                cached_uncond = uncond_pred.clone()  # Cache for v2

                comb_pred = uncond_pred + TRUE_CFG_SCALE * (cond_pred - uncond_pred)
                cond_norm = torch.norm(cond_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                final_pred = comb_pred * (cond_norm / noise_norm)

            latents_dtype = latents.dtype
            latents = pipeline.scheduler.step(final_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        torch.cuda.synchronize()
        loop_time = time.perf_counter() - t_loop_start

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
            1.0 / torch.tensor(pipeline.vae.config.latents_std)
            .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents_dec.device, latents_dec.dtype)
        )
        latents_dec = latents_dec / latents_std + latents_mean
        decoded = pipeline.vae.decode(latents_dec, return_dict=False)[0][:, :, 0]
        output_image = pipeline.image_processor.postprocess(decoded, output_type="pil")[0]

    peak_vram = torch.cuda.max_memory_reserved() / 1e9
    return output_image, loop_time, peak_vram


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT
# ═══════════════════════════════════════════════════════════════

# Warmup
print("\n" + "=" * 70)
print("WARMUP")
print("=" * 70)
warmup_imgs = make_images(TEST_CASES[0][1])
with torch.no_grad():
    _ = pipeline(
        image=warmup_imgs, prompt=TEST_CASES[0][2],
        num_inference_steps=NUM_STEPS, true_cfg_scale=TRUE_CFG_SCALE,
        negative_prompt=TEST_CASES[0][3], guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(SEED),
    )
print("Warmup done.\n")

results = []

for tc_idx, (tc_name, img_specs, prompt, neg_prompt) in enumerate(TEST_CASES):
    print("=" * 70)
    print(f"TEST CASE {tc_idx + 1}/{len(TEST_CASES)}: {tc_name}")
    print(f"  Images: {len(img_specs)}, Prompt: {prompt[:60]}...")
    print("=" * 70)

    images = make_images(img_specs)

    # Baseline
    print("  [baseline] Running...")
    img_base, t_base, vram_base = run_inference(images, prompt, neg_prompt, SEED, "baseline")
    print(f"    Time: {t_base:.2f}s, VRAM: {vram_base:.2f} GB")

    # v2: cached uncond reuse
    print("  [cfg_skip_v2] Running (cached uncond reuse)...")
    img_v2, t_v2, vram_v2 = run_inference(images, prompt, neg_prompt, SEED, "cfg_skip_v2")
    speedup_v2 = t_base / t_v2
    metrics_v2 = compute_metrics(img_base, img_v2)
    print(f"    Time: {t_v2:.2f}s, Speedup: {speedup_v2:.2f}x")
    print(f"    SSIM: {metrics_v2['ssim']:.4f}, MSE: {metrics_v2['mse']:.2f}, PSNR: {metrics_v2['psnr']:.2f} dB")

    # Save images
    img_base.save(os.path.join(OUTPUT_DIR, f"{tc_name}_baseline.png"))
    img_v2.save(os.path.join(OUTPUT_DIR, f"{tc_name}_cfg_skip_v2.png"))

    results.append({
        "test_case": tc_name,
        "num_images": len(img_specs),
        "prompt": prompt,
        "baseline_time_s": round(t_base, 2),
        "v2_time_s": round(t_v2, 2),
        "v2_speedup": round(speedup_v2, 3),
        "vram_baseline_gb": round(vram_base, 2),
        "vram_v2_gb": round(vram_v2, 2),
        "v2_ssim": round(metrics_v2["ssim"], 4),
        "v2_mse": round(metrics_v2["mse"], 2),
        "v2_psnr": round(metrics_v2["psnr"], 2),
    })
    print()


# ═══════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("RESULTS SUMMARY — CFG-Skip v2 (Cached Uncond Reuse)")
print("=" * 70)
print(f"Schedule: Full CFG steps {sorted(FULL_CFG_STEPS)}, cached uncond reuse for steps 3–38")
print(f"CFG scale: {TRUE_CFG_SCALE}\n")

print(f"{'Test Case':<25s} {'Base(s)':>8s} {'v2(s)':>8s} {'Speed':>7s} "
      f"{'SSIM':>7s} {'MSE':>9s} {'PSNR':>8s}")
print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*9} {'-'*8}")
for r in results:
    psnr_str = f"{r['v2_psnr']:.1f}" if r['v2_psnr'] < 100 else "inf"
    print(f"{r['test_case']:<25s} {r['baseline_time_s']:8.2f} {r['v2_time_s']:8.2f} "
          f"{r['v2_speedup']:7.3f} {r['v2_ssim']:7.4f} {r['v2_mse']:9.2f} {psnr_str:>8s}")

# Aggregates
speedups = [r["v2_speedup"] for r in results]
ssims = [r["v2_ssim"] for r in results]
mses = [r["v2_mse"] for r in results]
psnrs = [r["v2_psnr"] for r in results]

print(f"\n{'AGGREGATE':<25s} {'':>8s} {'':>8s} "
      f"{np.mean(speedups):7.3f} {np.mean(ssims):7.4f} {np.mean(mses):9.2f} "
      f"{np.mean([p for p in psnrs if p < 100]):>8.1f}")
print(f"{'  (range)':<25s} {'':>8s} {'':>8s} "
      f"{min(speedups):.2f}-{max(speedups):.2f} "
      f"{min(ssims):.3f}-{max(ssims):.3f}")

print("\nQuality Assessment:")
for r in results:
    if r["v2_ssim"] < 0.90:
        print(f"  ⚠ {r['test_case']}: SSIM={r['v2_ssim']:.4f} — NOTABLE DEGRADATION")
    elif r["v2_ssim"] < 0.95:
        print(f"  ~ {r['test_case']}: SSIM={r['v2_ssim']:.4f} — moderate difference")
    else:
        print(f"  ✓ {r['test_case']}: SSIM={r['v2_ssim']:.4f} — good quality")

# ─── Comparison with v1 ───
print("\n" + "=" * 70)
print("COMPARISON: v1 (no CFG) vs v2 (cached uncond)")
print("=" * 70)

# Load v1 results
v1_path = os.path.join(OUTPUT_DIR, "cfg_skip_results.json")
if os.path.exists(v1_path):
    with open(v1_path) as f:
        v1_data = json.load(f)

    print(f"\n{'Test Case':<25s} {'v1 SSIM':>8s} {'v2 SSIM':>8s} {'Delta':>8s} {'v1 Speed':>9s} {'v2 Speed':>9s}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")
    for v1_r, v2_r in zip(v1_data["per_test_case"], results):
        delta = v2_r["v2_ssim"] - v1_r["ssim"]
        print(f"{v1_r['test_case']:<25s} {v1_r['ssim']:8.4f} {v2_r['v2_ssim']:8.4f} "
              f"{delta:+8.4f} {v1_r['speedup']:9.3f} {v2_r['v2_speedup']:9.3f}")

# Save results
results_json = {
    "config": {
        "num_steps": NUM_STEPS,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "seed": SEED,
        "full_cfg_steps": sorted(FULL_CFG_STEPS),
        "num_skipped_steps": NUM_STEPS - len(FULL_CFG_STEPS),
        "method": "cached_uncond_reuse",
        "description": "On skip steps, reuse the most recent unconditional output in the CFG formula instead of running a fresh unconditional transformer pass.",
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

json_path = os.path.join(OUTPUT_DIR, "cfg_skip_v2_results.json")
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\nResults saved to {json_path}")
print("\nDone.")
