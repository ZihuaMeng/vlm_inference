#!/usr/bin/env python3
"""Milestone 9: Broad Validation + Ablation Study

Validates the fixed combined acceleration strategy (2.35x, 0.869 SSIM from M8)
on a diverse 18-case test suite. Produces ablation study showing each component's
contribution, quality distribution, and failure characterization.

Four configurations per test case:
  a. Baseline (full 40-step, full CFG) — 80 passes
  b. CFG-skip only (M5 conservative: skip uncond steps 5-34) — 50 passes, 1.60x
  c. Temporal cache only (M6 conservative: cache steps 10-25) — 48 passes, 1.67x
  d. Fixed combined (M8: temporal 10-25 + CFG-skip 5-9,26-34) — 34 passes, 2.35x

Metrics: wall-clock time, speedup, SSIM, LPIPS.
"""

import torch
import torch.nn.functional as F
import time
import json
import os
import sys
import contextlib
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import csv
import lpips

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════
WEIGHTS = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/weights/Qwen-Image-Edit-2511"
OUTPUT_DIR = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/outputs/m9_validation"
NUM_STEPS = 40
TRUE_CFG_SCALE = 4.0
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_FILE = os.path.join(OUTPUT_DIR, "m9_results.json")

# ═══════════════════════════════════════════════════════════════
#  SCHEDULE DEFINITIONS
# ═══════════════════════════════════════════════════════════════

# CFG-skip only (M5 conservative): skip uncond for steps 5-34
# Full CFG at steps 0-4 and 35-39 → 10×2 + 30×1 = 50 passes
CFG_SKIP_ONLY_STEPS = set(range(5, 35))

# Temporal cache only (M6 conservative): cache steps 10-25
# Full compute at steps 0-9 and 26-39 → 24×2 = 48 passes
TEMPORAL_ONLY_STEPS = set(range(10, 26))

# Fixed combined (M8): temporal cache 10-25, CFG-skip 5-9 and 26-34
# Steps 0-4,35-39 full (20) + steps 5-9,26-34 cond-only (14) + steps 10-25 cached (0) = 34
FIXED_TEMPORAL_STEPS = set(range(10, 26))
FIXED_CFG_STEPS = set(range(5, 10)) | set(range(26, 35))

MODES = ["baseline", "cfg_skip_only", "temporal_only", "fixed_combined"]

# ═══════════════════════════════════════════════════════════════
#  TEST SUITE — 18 cases: 6×1-img, 6×2-img, 6×3-img
# ═══════════════════════════════════════════════════════════════

TEST_SUITE = [
    # ── 1-IMAGE CASES ──
    {
        "name": "tc02_1img_landscape",
        "images": [("green", (768, 512))],
        "prompt": "A beautiful green meadow under a bright blue sky with fluffy white clouds.",
        "neg_prompt": " ",
        "edit_type": "landscape",
        "difficulty": "medium",
        "source": "M5/M6",
    },
    {
        "name": "tc08_1img_simple",
        "images": [("white", (512, 512))],
        "prompt": "A single red apple on a plain white background.",
        "neg_prompt": " ",
        "edit_type": "simple_object",
        "difficulty": "easy",
        "source": "new",
    },
    {
        "name": "tc09_1img_style",
        "images": [("beige", (512, 512))],
        "prompt": "A watercolor painting of a Japanese garden with cherry blossoms and a stone lantern.",
        "neg_prompt": " ",
        "edit_type": "style_transfer",
        "difficulty": "hard",
        "source": "new",
    },
    {
        "name": "tc10_1img_text",
        "images": [("navy", (640, 480))],
        "prompt": "A neon sign reading OPEN glowing in pink and blue against a dark brick wall.",
        "neg_prompt": " ",
        "edit_type": "text_logo",
        "difficulty": "hard",
        "source": "new",
    },
    {
        "name": "tc11_1img_material",
        "images": [("gray", (512, 512))],
        "prompt": "A close-up of a rough granite stone surface with visible crystal inclusions.",
        "neg_prompt": " ",
        "edit_type": "material_texture",
        "difficulty": "hard",
        "source": "new",
    },
    {
        "name": "tc12_1img_portrait",
        "images": [("lightblue", (512, 768))],
        "prompt": "A professional headshot portrait of a smiling woman with short brown hair.",
        "neg_prompt": " ",
        "edit_type": "portrait",
        "difficulty": "medium",
        "source": "new",
    },
    # ── 2-IMAGE CASES ──
    {
        "name": "tc01_2img_objects",
        "images": [("red", (512, 512)), ("blue", (512, 512))],
        "prompt": "A red object and a blue object sitting side by side on a white table.",
        "neg_prompt": " ",
        "edit_type": "object_placement",
        "difficulty": "easy",
        "source": "M5/M6",
    },
    {
        "name": "tc03_2img_scene",
        "images": [("yellow", (512, 512)), ("purple", (512, 512))],
        "prompt": "A yellow sun setting behind purple mountains with a river in the foreground.",
        "neg_prompt": " ",
        "edit_type": "scene_composition",
        "difficulty": "hard",
        "source": "M5/M6",
    },
    {
        "name": "tc05_2img_texture",
        "images": [("orange", (640, 480)), ("cyan", (640, 480))],
        "prompt": "An orange cat sleeping next to a cyan cushion on a wooden floor.",
        "neg_prompt": " ",
        "edit_type": "texture_detail",
        "difficulty": "hard",
        "source": "M5/M6",
    },
    {
        "name": "tc13_2img_identity",
        "images": [("peachpuff", (512, 512)), ("lightgreen", (512, 512))],
        "prompt": "The same person wearing a red jacket in the first photo now wearing a blue jacket, same pose and background.",
        "neg_prompt": " ",
        "edit_type": "identity_preserving",
        "difficulty": "hard",
        "source": "new",
    },
    {
        "name": "tc14_2img_replace",
        "images": [("brown", (512, 512)), ("silver", (512, 512))],
        "prompt": "A brown leather couch replaced with a modern silver metallic sofa in the same living room.",
        "neg_prompt": " ",
        "edit_type": "object_replacement",
        "difficulty": "medium",
        "source": "new",
    },
    {
        "name": "tc15_2img_background",
        "images": [("darkblue", (768, 512)), ("sandybrown", (768, 512))],
        "prompt": "A person standing on a beach at sunset with waves in the background.",
        "neg_prompt": " ",
        "edit_type": "background_change",
        "difficulty": "medium",
        "source": "new",
    },
    # ── 3-IMAGE CASES ──
    {
        "name": "tc04_3img_arrange",
        "images": [("red", (512, 512)), ("green", (512, 512)), ("blue", (512, 512))],
        "prompt": "Three colorful geometric shapes arranged in a triangle pattern on a gray surface.",
        "neg_prompt": " ",
        "edit_type": "arrangement",
        "difficulty": "easy",
        "source": "M5/M6",
    },
    {
        "name": "tc06_3img_portrait",
        "images": [("pink", (512, 512)), ("teal", (512, 512)), ("gold", (512, 512))],
        "prompt": "Three people standing in front of different colored walls in a studio setting.",
        "neg_prompt": " ",
        "edit_type": "multi_portrait",
        "difficulty": "medium",
        "source": "M8",
    },
    {
        "name": "tc07_3img_nature",
        "images": [("darkgreen", (512, 512)), ("skyblue", (512, 512)), ("brown", (512, 512))],
        "prompt": "A dense forest with tall trees next to a clear lake under a bright sky.",
        "neg_prompt": " ",
        "edit_type": "nature_scene",
        "difficulty": "hard",
        "source": "M8",
    },
    {
        "name": "tc16_3img_interior",
        "images": [("ivory", (512, 512)), ("maroon", (512, 512)), ("olive", (512, 512))],
        "prompt": "A modern minimalist living room with a large window, white walls, and wooden furniture.",
        "neg_prompt": " ",
        "edit_type": "interior_design",
        "difficulty": "medium",
        "source": "new",
    },
    {
        "name": "tc17_3img_complex",
        "images": [("coral", (640, 480)), ("lavender", (640, 480)), ("khaki", (640, 480))],
        "prompt": "A bustling street market with colorful stalls, hanging lanterns, and many people walking between vendors.",
        "neg_prompt": " ",
        "edit_type": "complex_scene",
        "difficulty": "hard",
        "source": "new",
    },
    {
        "name": "tc18_3img_abstract",
        "images": [("black", (512, 512)), ("white", (512, 512)), ("gray", (512, 512))],
        "prompt": "An abstract geometric composition with overlapping circles and triangles in bold primary colors.",
        "neg_prompt": " ",
        "edit_type": "abstract",
        "difficulty": "medium",
        "source": "new",
    },
]


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def make_images(specs):
    """Create solid-color PIL images from (color, (w,h)) specs."""
    return [Image.new("RGB", size, color) for color, size in specs]


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

print("Loading pipeline...")
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
pipeline.to("cuda")
vram_after_load = torch.cuda.memory_allocated() / 1e9
print(f"Pipeline loaded. VRAM: {vram_after_load:.2f} GB")

import importlib
pm = importlib.import_module(
    "diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus"
)
calculate_dimensions = pm.calculate_dimensions
CONDITION_IMAGE_SIZE = pm.CONDITION_IMAGE_SIZE
VAE_IMAGE_SIZE = pm.VAE_IMAGE_SIZE
calculate_shift = pm.calculate_shift
retrieve_timesteps = pm.retrieve_timesteps


# ═══════════════════════════════════════════════════════════════
#  LPIPS MODEL
# ═══════════════════════════════════════════════════════════════

print("Loading LPIPS model...")
lpips_fn = lpips.LPIPS(net='alex').cuda()
print("LPIPS ready.")


# ═══════════════════════════════════════════════════════════════
#  INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════

def prepare_inputs(image_list):
    """Prepare all inputs for the denoising loop."""
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


def get_step_decision(step, mode, has_prev):
    """Determine the action for a given step and mode."""
    if mode == "baseline":
        return "full"
    elif mode == "cfg_skip_only":
        return "cfg_skip" if step in CFG_SKIP_ONLY_STEPS else "full"
    elif mode == "temporal_only":
        if step in TEMPORAL_ONLY_STEPS and has_prev:
            return "temporal_cache"
        return "full"
    elif mode == "fixed_combined":
        if step in FIXED_TEMPORAL_STEPS and has_prev:
            return "temporal_cache"
        if step in FIXED_CFG_STEPS:
            return "cfg_skip"
        return "full"
    return "full"


def run_inference(image_list, prompt, negative_prompt, seed, mode="baseline"):
    """Run inference with specified acceleration mode.

    Returns: (output_image, loop_time_s, peak_vram_gb, total_passes)
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
            image=condition_images, prompt=prompt,
            prompt_embeds=None, prompt_embeds_mask=None,
            device="cuda", num_images_per_prompt=1, max_sequence_length=512,
        )
        negative_prompt_embeds, negative_prompt_embeds_mask = pipeline.encode_prompt(
            image=condition_images, prompt=negative_prompt,
            prompt_embeds=None, prompt_embeds_mask=None,
            device="cuda", num_images_per_prompt=1, max_sequence_length=512,
        )

        # VAE encoding / prepare latents
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

        prev_final_pred = None
        total_passes = 0

        # Denoising loop
        torch.cuda.synchronize()
        t_loop_start = time.perf_counter()

        for i, t in enumerate(timesteps):
            decision = get_step_decision(i, mode, prev_final_pred is not None)

            if decision == "temporal_cache":
                # Reuse previous final prediction
                final_pred = prev_final_pred

            else:
                # Run conditional pass
                latent_model_input = torch.cat([latents, image_latents], dim=1)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

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
                total_passes += 1

                if decision == "cfg_skip":
                    final_pred = cond_pred
                else:
                    # Full: run unconditional pass
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
                    total_passes += 1

                    # Rescaled CFG
                    comb_pred = uncond_pred + TRUE_CFG_SCALE * (cond_pred - uncond_pred)
                    cond_norm = torch.norm(cond_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    final_pred = comb_pred * (cond_norm / noise_norm)

                prev_final_pred = final_pred

            # Scheduler step
            latents_dtype = latents.dtype
            latents = pipeline.scheduler.step(
                final_pred, t, latents, return_dict=False
            )[0]
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
    return output_image, loop_time, peak_vram, total_passes


# ═══════════════════════════════════════════════════════════════
#  WARMUP
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("WARMUP PASS")
print("=" * 70)
warmup_imgs = make_images(TEST_SUITE[0]["images"])
with torch.no_grad():
    _ = pipeline(
        image=warmup_imgs,
        prompt=TEST_SUITE[0]["prompt"],
        num_inference_steps=NUM_STEPS,
        true_cfg_scale=TRUE_CFG_SCALE,
        negative_prompt=TEST_SUITE[0]["neg_prompt"],
        guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(SEED),
    )
print("Warmup done.\n")


# ═══════════════════════════════════════════════════════════════
#  MAIN EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print(f"EVALUATION: {len(TEST_SUITE)} cases × {len(MODES)} modes = {len(TEST_SUITE)*len(MODES)} runs")
print("=" * 70)

all_results = []
t_eval_start = time.perf_counter()

for tc_idx, tc in enumerate(TEST_SUITE):
    images = make_images(tc["images"])
    n_img = len(tc["images"])
    print(f"\n{'─'*65}")
    print(f"[{tc_idx+1}/{len(TEST_SUITE)}] {tc['name']} "
          f"({n_img} img, {tc['edit_type']}, {tc['difficulty']})")
    print(f"{'─'*65}")

    tc_result = {
        "name": tc["name"],
        "num_images": n_img,
        "edit_type": tc["edit_type"],
        "difficulty": tc["difficulty"],
        "source": tc["source"],
        "prompt": tc["prompt"],
    }
    baseline_img = None

    for mode in MODES:
        print(f"  {mode:<20s}...", end=" ", flush=True)
        img_out, t_loop, vram, passes = run_inference(
            images, tc["prompt"], tc["neg_prompt"], SEED, mode=mode
        )

        if mode == "baseline":
            baseline_img = img_out
            tc_result["baseline_time_s"] = round(t_loop, 2)
            tc_result["baseline_vram_gb"] = round(vram, 2)
            tc_result["baseline_passes"] = passes
            print(f"{t_loop:.2f}s, {passes} passes, {vram:.1f} GB")
        else:
            speedup = tc_result["baseline_time_s"] / t_loop
            ssim_val = compute_ssim(baseline_img, img_out)
            lpips_val = compute_lpips_distance(baseline_img, img_out, lpips_fn)
            tc_result[f"{mode}_time_s"] = round(t_loop, 2)
            tc_result[f"{mode}_speedup"] = round(speedup, 3)
            tc_result[f"{mode}_ssim"] = round(ssim_val, 4)
            tc_result[f"{mode}_lpips"] = round(lpips_val, 4)
            tc_result[f"{mode}_passes"] = passes
            print(f"{t_loop:.2f}s, {speedup:.2f}x, "
                  f"SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}, {passes}p")

        # Save image
        img_path = os.path.join(OUTPUT_DIR, f"{tc['name']}_{mode}.png")
        img_out.save(img_path)

    all_results.append(tc_result)

    # Save incrementally
    with open(RESULTS_FILE, "w") as f:
        json.dump({"results": all_results}, f, indent=2)

    elapsed = time.perf_counter() - t_eval_start
    remaining_cases = len(TEST_SUITE) - (tc_idx + 1)
    avg_per_case = elapsed / (tc_idx + 1)
    print(f"  [Saved. Elapsed: {elapsed:.0f}s, ~{remaining_cases * avg_per_case:.0f}s remaining]")

t_eval_end = time.perf_counter()
print(f"\nTotal evaluation time: {t_eval_end - t_eval_start:.1f}s")


# ═══════════════════════════════════════════════════════════════
#  ANALYSIS
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

accel_modes = ["cfg_skip_only", "temporal_only", "fixed_combined"]

# 1. Full results table
print("\n── 1. Full Results Table ──\n")
header = (f"{'Case':<25s} {'Img':>3s} {'Type':<18s} {'Diff':<6s} "
          f"{'Base':>6s} "
          f"{'CFG_x':>6s} {'CFG_S':>6s} {'CFG_L':>6s} "
          f"{'TMP_x':>6s} {'TMP_S':>6s} {'TMP_L':>6s} "
          f"{'CMB_x':>6s} {'CMB_S':>6s} {'CMB_L':>6s}")
print(header)
print("-" * len(header))
for r in all_results:
    print(f"{r['name']:<25s} {r['num_images']:>3d} {r['edit_type']:<18s} {r['difficulty']:<6s} "
          f"{r['baseline_time_s']:6.1f} "
          f"{r['cfg_skip_only_speedup']:6.2f} {r['cfg_skip_only_ssim']:6.3f} {r['cfg_skip_only_lpips']:6.3f} "
          f"{r['temporal_only_speedup']:6.2f} {r['temporal_only_ssim']:6.3f} {r['temporal_only_lpips']:6.3f} "
          f"{r['fixed_combined_speedup']:6.2f} {r['fixed_combined_ssim']:6.3f} {r['fixed_combined_lpips']:6.3f}")

# 2. Ablation summary
print("\n── 2. Ablation Summary ──\n")
for mode in accel_modes:
    speedups = [r[f"{mode}_speedup"] for r in all_results]
    ssims = [r[f"{mode}_ssim"] for r in all_results]
    lpipss = [r[f"{mode}_lpips"] for r in all_results]
    print(f"  {mode:<20s}  speedup={np.mean(speedups):.3f}x ± {np.std(speedups):.3f}  "
          f"SSIM={np.mean(ssims):.4f} ± {np.std(ssims):.4f} [{min(ssims):.3f}-{max(ssims):.3f}]  "
          f"LPIPS={np.mean(lpipss):.4f} ± {np.std(lpipss):.4f}")

# 3. Quality distribution for fixed_combined
print("\n── 3. Quality Distribution (fixed_combined) ──\n")
ssims_cmb = sorted([r["fixed_combined_ssim"] for r in all_results])
lpips_cmb = sorted([r["fixed_combined_lpips"] for r in all_results])
n = len(ssims_cmb)
print(f"  SSIM percentiles: p10={ssims_cmb[int(n*0.1)]:.3f}  p25={ssims_cmb[int(n*0.25)]:.3f}  "
      f"p50={ssims_cmb[n//2]:.3f}  p75={ssims_cmb[int(n*0.75)]:.3f}  p90={ssims_cmb[int(n*0.9)]:.3f}")
print(f"  LPIPS percentiles: p10={lpips_cmb[int(n*0.1)]:.3f}  p25={lpips_cmb[int(n*0.25)]:.3f}  "
      f"p50={lpips_cmb[n//2]:.3f}  p75={lpips_cmb[int(n*0.75)]:.3f}  p90={lpips_cmb[int(n*0.9)]:.3f}")
above_90 = sum(1 for s in ssims_cmb if s > 0.90)
above_85 = sum(1 for s in ssims_cmb if s > 0.85)
below_80 = sum(1 for s in ssims_cmb if s < 0.80)
print(f"  SSIM > 0.90: {above_90}/{n} ({100*above_90/n:.0f}%)")
print(f"  SSIM > 0.85: {above_85}/{n} ({100*above_85/n:.0f}%)")
print(f"  SSIM < 0.80: {below_80}/{n} ({100*below_80/n:.0f}%)")

# 4. Image-count scaling
print("\n── 4. Image-Count Scaling ──\n")
for ni in [1, 2, 3]:
    subset = [r for r in all_results if r["num_images"] == ni]
    base_times = [r["baseline_time_s"] for r in subset]
    for mode in accel_modes:
        speedups = [r[f"{mode}_speedup"] for r in subset]
        ssims = [r[f"{mode}_ssim"] for r in subset]
        accel_times = [r[f"{mode}_time_s"] for r in subset]
        savings = [r["baseline_time_s"] - r[f"{mode}_time_s"] for r in subset]
        if mode == "fixed_combined":
            print(f"  {ni}-img (n={len(subset)}): base={np.mean(base_times):.1f}s  "
                  f"combined={np.mean(accel_times):.1f}s  speedup={np.mean(speedups):.3f}x  "
                  f"SSIM={np.mean(ssims):.4f}  saved={np.mean(savings):.1f}s")

# 5. Failure characterization: by edit type
print("\n── 5. Failure Characterization (fixed_combined SSIM by edit type) ──\n")
from collections import defaultdict
type_ssims = defaultdict(list)
type_lpips = defaultdict(list)
for r in all_results:
    type_ssims[r["edit_type"]].append(r["fixed_combined_ssim"])
    type_lpips[r["edit_type"]].append(r["fixed_combined_lpips"])
ranked = sorted(type_ssims.items(), key=lambda x: np.mean(x[1]))
for etype, vals in ranked:
    lvals = type_lpips[etype]
    print(f"  {etype:<20s}  SSIM={np.mean(vals):.3f}  LPIPS={np.mean(lvals):.3f}  (n={len(vals)})")

# By difficulty
print("\n── 5b. By difficulty ──\n")
for diff in ["easy", "medium", "hard"]:
    subset = [r for r in all_results if r["difficulty"] == diff]
    if subset:
        ssims = [r["fixed_combined_ssim"] for r in subset]
        lpipss = [r["fixed_combined_lpips"] for r in subset]
        print(f"  {diff:<8s} (n={len(subset)}): SSIM={np.mean(ssims):.4f} ± {np.std(ssims):.4f}  "
              f"LPIPS={np.mean(lpipss):.4f} ± {np.std(lpipss):.4f}")

# 6. Component contribution (ablation)
print("\n── 6. Component Contribution ──\n")
cfg_mean_ssim = np.mean([r["cfg_skip_only_ssim"] for r in all_results])
tmp_mean_ssim = np.mean([r["temporal_only_ssim"] for r in all_results])
cmb_mean_ssim = np.mean([r["fixed_combined_ssim"] for r in all_results])
cfg_mean_spd = np.mean([r["cfg_skip_only_speedup"] for r in all_results])
tmp_mean_spd = np.mean([r["temporal_only_speedup"] for r in all_results])
cmb_mean_spd = np.mean([r["fixed_combined_speedup"] for r in all_results])
print(f"  CFG-skip only:     {cfg_mean_spd:.3f}x speedup, {cfg_mean_ssim:.4f} SSIM")
print(f"  Temporal only:     {tmp_mean_spd:.3f}x speedup, {tmp_mean_ssim:.4f} SSIM")
print(f"  Fixed combined:    {cmb_mean_spd:.3f}x speedup, {cmb_mean_ssim:.4f} SSIM")
print(f"  Combined vs best single axis: +{cmb_mean_spd - max(cfg_mean_spd, tmp_mean_spd):.3f}x speedup, "
      f"{cmb_mean_ssim - max(cfg_mean_ssim, tmp_mean_ssim):.4f} SSIM delta")

# 7. Worst cases
print("\n── 7. Worst Cases (fixed_combined, bottom 5 SSIM) ──\n")
sorted_by_ssim = sorted(all_results, key=lambda r: r["fixed_combined_ssim"])
for r in sorted_by_ssim[:5]:
    print(f"  {r['name']:<25s} SSIM={r['fixed_combined_ssim']:.4f}  "
          f"LPIPS={r['fixed_combined_lpips']:.4f}  "
          f"type={r['edit_type']}, diff={r['difficulty']}, imgs={r['num_images']}")

# 8. Bottom line
print("\n── 8. Bottom Line ──\n")
m8_ssim = 0.869
m8_speedup = 2.35
print(f"  M8 reference:  speedup={m8_speedup:.2f}x, mean SSIM={m8_ssim:.3f} (7 cases)")
print(f"  M9 validated:  speedup={cmb_mean_spd:.3f}x, mean SSIM={cmb_mean_ssim:.4f} ({n} cases)")
ssim_delta = cmb_mean_ssim - m8_ssim
print(f"  Delta:         speedup {'+' if cmb_mean_spd - m8_speedup >= 0 else ''}{cmb_mean_spd - m8_speedup:.3f}x, "
      f"SSIM {'+' if ssim_delta >= 0 else ''}{ssim_delta:.4f}")
if cmb_mean_ssim >= 0.82:
    print(f"  VERDICT: Method generalizes. {cmb_mean_spd:.2f}x speedup with {cmb_mean_ssim:.3f} mean SSIM is validated.")
else:
    print(f"  VERDICT: Quality degradation detected. Mean SSIM {cmb_mean_ssim:.3f} < 0.82 threshold. "
          "Method may need more conservative operating point.")


# ═══════════════════════════════════════════════════════════════
#  SAVE FINAL RESULTS
# ═══════════════════════════════════════════════════════════════

# Add aggregate stats to the JSON
final_json = {
    "config": {
        "num_steps": NUM_STEPS,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "seed": SEED,
        "modes": {
            "baseline": {"passes": 80, "description": "Full 40-step CFG"},
            "cfg_skip_only": {"passes": 50, "description": "M5 conservative: skip uncond steps 5-34"},
            "temporal_only": {"passes": 48, "description": "M6 conservative: cache steps 10-25"},
            "fixed_combined": {"passes": 34, "description": "M8: temporal 10-25 + CFG-skip 5-9,26-34"},
        },
    },
    "results": all_results,
    "aggregate": {},
}
for mode in accel_modes:
    speedups = [r[f"{mode}_speedup"] for r in all_results]
    ssims = [r[f"{mode}_ssim"] for r in all_results]
    lpipss = [r[f"{mode}_lpips"] for r in all_results]
    final_json["aggregate"][mode] = {
        "mean_speedup": round(float(np.mean(speedups)), 3),
        "std_speedup": round(float(np.std(speedups)), 3),
        "mean_ssim": round(float(np.mean(ssims)), 4),
        "std_ssim": round(float(np.std(ssims)), 4),
        "min_ssim": round(float(min(ssims)), 4),
        "max_ssim": round(float(max(ssims)), 4),
        "mean_lpips": round(float(np.mean(lpipss)), 4),
        "std_lpips": round(float(np.std(lpipss)), 4),
    }

with open(RESULTS_FILE, "w") as f:
    json.dump(final_json, f, indent=2)
print(f"\nFinal results saved to {RESULTS_FILE}")

# Also save CSV for easy import
csv_path = os.path.join(OUTPUT_DIR, "m9_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "name", "num_images", "edit_type", "difficulty", "source",
        "baseline_time_s",
        "cfg_skip_only_time_s", "cfg_skip_only_speedup", "cfg_skip_only_ssim", "cfg_skip_only_lpips",
        "temporal_only_time_s", "temporal_only_speedup", "temporal_only_ssim", "temporal_only_lpips",
        "fixed_combined_time_s", "fixed_combined_speedup", "fixed_combined_ssim", "fixed_combined_lpips",
    ])
    for r in all_results:
        writer.writerow([
            r["name"], r["num_images"], r["edit_type"], r["difficulty"], r["source"],
            r["baseline_time_s"],
            r["cfg_skip_only_time_s"], r["cfg_skip_only_speedup"], r["cfg_skip_only_ssim"], r["cfg_skip_only_lpips"],
            r["temporal_only_time_s"], r["temporal_only_speedup"], r["temporal_only_ssim"], r["temporal_only_lpips"],
            r["fixed_combined_time_s"], r["fixed_combined_speedup"], r["fixed_combined_ssim"], r["fixed_combined_lpips"],
        ])
print(f"CSV saved to {csv_path}")

# Count images saved
n_images = len(TEST_SUITE) * len(MODES)
print(f"\n{n_images} output images saved to {OUTPUT_DIR}/")
print("\nDone.")
