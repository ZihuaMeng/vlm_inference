#!/usr/bin/env python3
"""Milestone 8: Adaptive Combined Acceleration Prototype

Combines temporal step-caching (M6) and CFG-skip (M5) into a unified
denoising loop with two operating modes:

  1. Fixed combined: conservative temporal cache (M6: cache steps 10-25)
     + conservative CFG-skip (M5: skip uncond steps 5-9, 26-34).
     Total: 34/80 passes → theoretical 2.35x.

  2. Adaptive combined: per-step cosine similarity signal modulates
     cache/skip decisions at runtime. Three tiers:
       - Temporal cache (reuse prev prediction): when lagged signal > 0.999
       - CFG-skip (skip uncond pass): when current signal > 0.995
       - Full compute: when similarity drops below threshold

Evaluates across 7 test cases + image-count scaling analysis (1/2/3 images).
"""

import torch
import torch.nn.functional as F
import time
import json
import os
import contextlib
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ─── Configuration ───
WEIGHTS = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/weights/Qwen-Image-Edit-2511"
OUTPUT_DIR = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/outputs/m8_combined"
NUM_STEPS = 40
TRUE_CFG_SCALE = 4.0
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Fixed Combined Schedule ───
# From M6 conservative: temporal cache steps 10-25 (skip everything)
# From M5 conservative: CFG-skip steps 5-34 (skip uncond)
# Combined: temporal cache takes precedence where both apply
FIXED_CONFIG = {
    "temporal_cache_steps": set(range(10, 26)),   # 16 steps: reuse prev prediction
    "cfg_skip_steps": set(range(5, 10)) | set(range(26, 35)),  # 14 steps: cond only
    # Remaining: steps 0-4 and 35-39 → full compute (10 steps × 2 passes = 20)
    # Total passes: 0 + 14 + 20 = 34 / 80 baseline → theoretical 2.35x
}

# ─── Adaptive Combined Config ───
ADAPTIVE_CONFIG = {
    "temporal_threshold": 0.999,    # cache if lagged similarity above this
    "cfg_skip_threshold": 0.995,    # skip uncond if current similarity above this
    "max_consecutive_cache": 4,     # force probe after this many cached steps
    "min_warmup_steps": 3,          # always full compute for first N steps
}

# ─── Test Cases ───
# tc1-tc5: identical to M5/M6 for continuity
# tc6-tc7: new 3-image cases for multi-image evaluation
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
    (
        "tc6_3img_portrait",
        [("pink", (512, 512)), ("teal", (512, 512)), ("gold", (512, 512))],
        "Three people standing in front of different colored walls in a studio setting.",
        " ",
    ),
    (
        "tc7_3img_nature",
        [("darkgreen", (512, 512)), ("skyblue", (512, 512)), ("brown", (512, 512))],
        "A dense forest with tall trees next to a clear lake under a bright sky.",
        " ",
    ),
]

# Cases for image-count scaling analysis (run each with 1, 2, 3 images)
SCALING_CASES = [
    (
        "scale_objects",
        [("red", (512, 512)), ("blue", (512, 512)), ("green", (512, 512))],
        "Colorful objects arranged on a white table.",
        " ",
    ),
    (
        "scale_shapes",
        [("red", (512, 512)), ("green", (512, 512)), ("blue", (512, 512))],
        "Three colorful geometric shapes arranged in a triangle pattern on a gray surface.",
        " ",
    ),
    (
        "scale_portrait",
        [("pink", (512, 512)), ("teal", (512, 512)), ("gold", (512, 512))],
        "Three people standing in front of different colored walls in a studio setting.",
        " ",
    ),
]


def make_images(specs):
    """Create solid-color PIL images from specs."""
    return [Image.new("RGB", size, color) for color, size in specs]


def compute_metrics(img_a, img_b):
    """Compute SSIM, MSE, PSNR between two PIL images."""
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.LANCZOS)
    a = np.array(img_a).astype(np.float64)
    b = np.array(img_b).astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse < 1e-10:
        psnr = float("inf")
    else:
        psnr = 10.0 * np.log10(255.0**2 / mse)
    ssim_val = ssim(a, b, channel_axis=2, data_range=255.0)
    return {"ssim": float(ssim_val), "mse": float(mse), "psnr": float(psnr)}


# ─── Load Pipeline ───
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


def run_inference(image_list, prompt, negative_prompt, seed, mode="baseline"):
    """Run inference with specified acceleration mode.

    Modes:
        "baseline":          Full CFG every step (80 passes)
        "fixed_combined":    Fixed schedule from M5+M6 conservative (34 passes)
        "adaptive_combined": Runtime cosine-sim signal modulates decisions

    Returns: (output_image, loop_time_s, peak_vram_gb, step_log)
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

        # ── Adaptive state ──
        prev_cond_output = None
        prev_final_pred = None
        last_similarity = 0.0
        consecutive_cached = 0

        step_log = []  # per-step decision log

        # ── Denoising Loop ──
        torch.cuda.synchronize()
        t_loop_start = time.perf_counter()

        for i, t in enumerate(timesteps):
            decision = None
            similarity = None
            transformer_passes = 0

            # ─── Decide action based on mode ───

            if mode == "baseline":
                decision = "full"

            elif mode == "fixed_combined":
                if i in FIXED_CONFIG["temporal_cache_steps"]:
                    decision = "temporal_cache"
                elif i in FIXED_CONFIG["cfg_skip_steps"]:
                    decision = "cfg_skip"
                else:
                    decision = "full"

            elif mode == "adaptive_combined":
                if i < ADAPTIVE_CONFIG["min_warmup_steps"]:
                    decision = "full"
                elif (prev_final_pred is not None
                      and last_similarity > ADAPTIVE_CONFIG["temporal_threshold"]
                      and consecutive_cached < ADAPTIVE_CONFIG["max_consecutive_cache"]):
                    decision = "temporal_cache"
                else:
                    decision = "probe"  # run cond, decide after

            # ─── Execute decision ───

            if decision == "temporal_cache":
                # Reuse previous final prediction entirely
                final_pred = prev_final_pred
                consecutive_cached += 1

            else:
                # Reset consecutive cache counter
                consecutive_cached = 0

                # Always run conditional pass
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
                transformer_passes += 1

                # Measure cosine similarity for adaptive mode
                if mode == "adaptive_combined" and prev_cond_output is not None:
                    similarity = F.cosine_similarity(
                        cond_pred.reshape(1, -1),
                        prev_cond_output.reshape(1, -1),
                    ).item()

                prev_cond_output = cond_pred.clone()

                # Finalize decision for "probe" mode
                if decision == "probe":
                    if (similarity is not None
                            and similarity > ADAPTIVE_CONFIG["cfg_skip_threshold"]):
                        decision = "cfg_skip"
                    else:
                        decision = "full"

                # Execute CFG-skip or full compute
                if decision == "cfg_skip":
                    final_pred = cond_pred
                else:
                    # Full compute: run unconditional pass
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
                    transformer_passes += 1

                    # Rescaled CFG combination
                    comb_pred = uncond_pred + TRUE_CFG_SCALE * (cond_pred - uncond_pred)
                    cond_norm = torch.norm(cond_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    final_pred = comb_pred * (cond_norm / noise_norm)

                # Update adaptive state
                if similarity is not None:
                    last_similarity = similarity
                prev_final_pred = final_pred

            # Scheduler step (always runs)
            latents_dtype = latents.dtype
            latents = pipeline.scheduler.step(
                final_pred, t, latents, return_dict=False
            )[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            step_log.append({
                "step": i,
                "decision": decision,
                "similarity": round(similarity, 6) if similarity is not None else None,
                "passes": transformer_passes,
            })

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
    return output_image, loop_time, peak_vram, step_log


# ═══════════════════════════════════════════════════════════════
#  WARMUP
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("WARMUP PASS")
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

# ═══════════════════════════════════════════════════════════════
#  PHASE 1: PRIMARY EXPERIMENTS (7 test cases × 3 modes)
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 1: PRIMARY EXPERIMENTS")
print("=" * 70)

MODES = ["baseline", "fixed_combined", "adaptive_combined"]
primary_results = []

for tc_idx, (tc_name, img_specs, prompt, neg_prompt) in enumerate(TEST_CASES):
    images = make_images(img_specs)
    n_img = len(img_specs)
    print(f"\n{'─'*60}")
    print(f"[{tc_idx+1}/{len(TEST_CASES)}] {tc_name} ({n_img} images)")
    print(f"{'─'*60}")

    tc_data = {
        "test_case": tc_name,
        "num_images": n_img,
        "prompt": prompt,
    }
    baseline_img = None

    for mode in MODES:
        print(f"  {mode}...", end=" ", flush=True)
        img_out, t_loop, vram, step_log = run_inference(
            images, prompt, neg_prompt, SEED, mode=mode
        )

        total_passes = sum(s["passes"] for s in step_log)
        decisions = {}
        for s in step_log:
            decisions[s["decision"]] = decisions.get(s["decision"], 0) + 1

        if mode == "baseline":
            baseline_img = img_out
            tc_data["baseline_time_s"] = round(t_loop, 2)
            tc_data["baseline_vram_gb"] = round(vram, 2)
            print(f"{t_loop:.2f}s, {vram:.2f} GB, {total_passes} passes")
        else:
            metrics = compute_metrics(baseline_img, img_out)
            speedup = tc_data["baseline_time_s"] / t_loop
            tc_data[f"{mode}_time_s"] = round(t_loop, 2)
            tc_data[f"{mode}_speedup"] = round(speedup, 3)
            tc_data[f"{mode}_ssim"] = round(metrics["ssim"], 4)
            tc_data[f"{mode}_mse"] = round(metrics["mse"], 2)
            tc_data[f"{mode}_psnr"] = round(metrics["psnr"], 2)
            tc_data[f"{mode}_passes"] = total_passes
            tc_data[f"{mode}_decisions"] = decisions
            print(f"{t_loop:.2f}s, speedup={speedup:.2f}x, "
                  f"SSIM={metrics['ssim']:.4f}, passes={total_passes}, "
                  f"decisions={decisions}")

        # Save image
        img_path = os.path.join(OUTPUT_DIR, f"{tc_name}_{mode}.png")
        img_out.save(img_path)

    primary_results.append(tc_data)

# ═══════════════════════════════════════════════════════════════
#  PHASE 2: IMAGE-COUNT SCALING (3 cases × 3 counts × 3 modes)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: IMAGE-COUNT SCALING")
print("=" * 70)

scaling_results = []

for sc_idx, (sc_name, img_specs, prompt, neg_prompt) in enumerate(SCALING_CASES):
    for n_img in [1, 2, 3]:
        images = make_images(img_specs[:n_img])
        run_label = f"{sc_name}_{n_img}img"
        print(f"\n{'─'*60}")
        print(f"[{sc_idx+1}/{len(SCALING_CASES)}] {run_label}")
        print(f"{'─'*60}")

        sc_data = {
            "case": sc_name,
            "num_images": n_img,
            "label": run_label,
        }
        baseline_img = None

        for mode in MODES:
            print(f"  {mode}...", end=" ", flush=True)
            img_out, t_loop, vram, step_log = run_inference(
                images, prompt, neg_prompt, SEED, mode=mode
            )

            total_passes = sum(s["passes"] for s in step_log)
            decisions = {}
            for s in step_log:
                decisions[s["decision"]] = decisions.get(s["decision"], 0) + 1

            if mode == "baseline":
                baseline_img = img_out
                sc_data["baseline_time_s"] = round(t_loop, 2)
                print(f"{t_loop:.2f}s, {total_passes} passes")
            else:
                metrics = compute_metrics(baseline_img, img_out)
                speedup = sc_data["baseline_time_s"] / t_loop
                sc_data[f"{mode}_time_s"] = round(t_loop, 2)
                sc_data[f"{mode}_speedup"] = round(speedup, 3)
                sc_data[f"{mode}_ssim"] = round(metrics["ssim"], 4)
                sc_data[f"{mode}_passes"] = total_passes
                sc_data[f"{mode}_decisions"] = decisions
                print(f"{t_loop:.2f}s, speedup={speedup:.2f}x, "
                      f"SSIM={metrics['ssim']:.4f}, passes={total_passes}")

            # Save image
            img_path = os.path.join(OUTPUT_DIR, f"{run_label}_{mode}.png")
            img_out.save(img_path)

        scaling_results.append(sc_data)


# ═══════════════════════════════════════════════════════════════
#  SUMMARY AND ANALYSIS
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 1: PRIMARY RESULTS TABLE")
print("=" * 70)

header = (f"{'Case':<25s} {'Imgs':>4s} {'Base(s)':>8s} "
          f"{'Fix(s)':>7s} {'Fix_x':>6s} {'Fix_Q':>6s} "
          f"{'Adp(s)':>7s} {'Adp_x':>6s} {'Adp_Q':>6s}")
print(header)
print("-" * len(header))

for r in primary_results:
    fix_q = f"{r['fixed_combined_ssim']:.3f}"
    adp_q = f"{r['adaptive_combined_ssim']:.3f}"
    print(f"{r['test_case']:<25s} {r['num_images']:>4d} {r['baseline_time_s']:8.2f} "
          f"{r['fixed_combined_time_s']:7.2f} {r['fixed_combined_speedup']:6.2f} {fix_q:>6s} "
          f"{r['adaptive_combined_time_s']:7.2f} {r['adaptive_combined_speedup']:6.2f} {adp_q:>6s}")

# Aggregates
fix_speedups = [r["fixed_combined_speedup"] for r in primary_results]
fix_ssims = [r["fixed_combined_ssim"] for r in primary_results]
adp_speedups = [r["adaptive_combined_speedup"] for r in primary_results]
adp_ssims = [r["adaptive_combined_ssim"] for r in primary_results]

print(f"\n{'Fixed combined:':<25s} mean speedup={np.mean(fix_speedups):.3f}x  "
      f"mean SSIM={np.mean(fix_ssims):.4f}  range={min(fix_ssims):.3f}-{max(fix_ssims):.3f}")
print(f"{'Adaptive combined:':<25s} mean speedup={np.mean(adp_speedups):.3f}x  "
      f"mean SSIM={np.mean(adp_ssims):.4f}  range={min(adp_ssims):.3f}-{max(adp_ssims):.3f}")

# Hard cases analysis
print("\n── Hard Cases (tc3, tc5) ──")
for r in primary_results:
    if r["test_case"] in ("tc3_2img_scene", "tc5_2img_texture"):
        print(f"  {r['test_case']}:")
        print(f"    Fixed:    speedup={r['fixed_combined_speedup']:.2f}x  "
              f"SSIM={r['fixed_combined_ssim']:.4f}")
        print(f"    Adaptive: speedup={r['adaptive_combined_speedup']:.2f}x  "
              f"SSIM={r['adaptive_combined_ssim']:.4f}")

# Quality robustness: variance comparison
print(f"\n── Quality Robustness ──")
print(f"  Fixed SSIM std:    {np.std(fix_ssims):.4f}")
print(f"  Adaptive SSIM std: {np.std(adp_ssims):.4f}")
print(f"  {'Adaptive is more robust' if np.std(adp_ssims) < np.std(fix_ssims) else 'Fixed is more robust'}")

# Adaptive step decisions
print(f"\n── Adaptive Mode Decision Distribution ──")
for r in primary_results:
    d = r.get("adaptive_combined_decisions", {})
    print(f"  {r['test_case']}: {d}")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: IMAGE-COUNT SCALING RESULTS")
print("=" * 70)

header2 = (f"{'Case':<25s} {'Imgs':>4s} {'Base(s)':>8s} "
           f"{'Fix_x':>6s} {'Fix_Q':>6s} "
           f"{'Adp_x':>6s} {'Adp_Q':>6s}")
print(header2)
print("-" * len(header2))

for r in scaling_results:
    fix_q = f"{r['fixed_combined_ssim']:.3f}"
    adp_q = f"{r['adaptive_combined_ssim']:.3f}"
    print(f"{r['label']:<25s} {r['num_images']:>4d} {r['baseline_time_s']:8.2f} "
          f"{r['fixed_combined_speedup']:6.2f} {fix_q:>6s} "
          f"{r['adaptive_combined_speedup']:6.2f} {adp_q:>6s}")

# Scaling analysis per case
print(f"\n── Speedup vs Image Count ──")
for sc_name in ["scale_objects", "scale_shapes", "scale_portrait"]:
    case_data = [r for r in scaling_results if r["case"] == sc_name]
    print(f"\n  {sc_name}:")
    for r in sorted(case_data, key=lambda x: x["num_images"]):
        print(f"    {r['num_images']}img: base={r['baseline_time_s']:.1f}s  "
              f"fixed={r['fixed_combined_speedup']:.3f}x  "
              f"adaptive={r['adaptive_combined_speedup']:.3f}x")

# ═══════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ═══════════════════════════════════════════════════════════════

results_json = {
    "config": {
        "num_steps": NUM_STEPS,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "seed": SEED,
        "fixed_combined": {
            "temporal_cache_steps": sorted(FIXED_CONFIG["temporal_cache_steps"]),
            "cfg_skip_steps": sorted(FIXED_CONFIG["cfg_skip_steps"]),
            "theoretical_passes": 34,
            "description": "M6 conservative temporal (cache 10-25) + M5 conservative CFG-skip (skip 5-9, 26-34)",
        },
        "adaptive_combined": ADAPTIVE_CONFIG,
    },
    "primary_results": primary_results,
    "scaling_results": scaling_results,
    "aggregate": {
        "fixed_combined": {
            "mean_speedup": round(float(np.mean(fix_speedups)), 3),
            "mean_ssim": round(float(np.mean(fix_ssims)), 4),
            "min_ssim": round(float(min(fix_ssims)), 4),
            "max_ssim": round(float(max(fix_ssims)), 4),
            "ssim_std": round(float(np.std(fix_ssims)), 4),
        },
        "adaptive_combined": {
            "mean_speedup": round(float(np.mean(adp_speedups)), 3),
            "mean_ssim": round(float(np.mean(adp_ssims)), 4),
            "min_ssim": round(float(min(adp_ssims)), 4),
            "max_ssim": round(float(max(adp_ssims)), 4),
            "ssim_std": round(float(np.std(adp_ssims)), 4),
        },
    },
}

json_path = os.path.join(OUTPUT_DIR, "m8_combined_results.json")
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\nResults saved to {json_path}")
print("\nDone.")
