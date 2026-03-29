#!/usr/bin/env python3
"""Milestone 6: Temporal Step-Caching Prototype (Block-Aware)

Implements temporal caching for the denoising loop plateau region
identified in M4. Tests three schedules:

  A. Aggressive uniform: steps 5-30 skip transformer entirely, reuse
     previous step's noise prediction. (26 cached / 14 full)
  B. Conservative uniform: steps 10-25 skip transformer entirely.
     (16 cached / 24 full)
  C. Block-aware: steps 5-30, cache stable blocks 0-39, always
     recompute dynamic blocks 40-59 (includes block 44 and all
     downstream so fresh output reaches norm_out/proj_out).

For each of 5 test cases (same as M5, identical seeds), generates
baseline + 3 schedule variants. Measures wall-clock time, SSIM, MSE, PSNR.

Key design choices (from M4 block-level data):
  - Block 44: most dynamic (temporal cos ~0.973 in plateau). Must recompute.
  - Block 59: most stable (cos ~0.9999). Safe to cache, but we recompute it
    anyway because it's downstream of block 44 and needs fresh input for
    the norm_out/proj_out to produce a quality noise prediction.
  - Blocks 0-39: all measured blocks in this range have cos >0.996.
    Safe to cache.
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
OUTPUT_DIR = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/outputs/m6_temporal_cache"
NUM_STEPS = 40
TRUE_CFG_SCALE = 4.0
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Schedule Definitions ───
SCHEDULES = [
    {
        "name": "aggressive_uniform",
        "label": "Aggressive Uniform (cache steps 5-30)",
        "cached_steps": set(range(5, 31)),  # 26 cached steps
        "type": "uniform",
    },
    {
        "name": "conservative_uniform",
        "label": "Conservative Uniform (cache steps 10-25)",
        "cached_steps": set(range(10, 26)),  # 16 cached steps
        "type": "uniform",
    },
    {
        "name": "block_aware",
        "label": "Block-Aware (cache steps 5-30, recompute blocks 40-59)",
        "cached_steps": set(range(5, 31)),  # 26 cached steps
        "type": "block_aware",
        "cached_blocks": set(range(0, 40)),   # blocks 0-39: return cached output
        "dynamic_blocks": set(range(40, 60)), # blocks 40-59: always recompute
    },
]

# ─── Test Cases (identical to M5) ───
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


# ─── Block-Level Cache Manager ───
class BlockCacheManager:
    """Manages per-block output caching for the transformer.

    For block-aware caching:
      - Stable blocks (0-39): return cached output from last full step
      - Dynamic blocks (40-59): always recompute, update their cache
    """

    def __init__(self, transformer, cached_blocks, dynamic_blocks):
        self.cached_blocks = cached_blocks
        self.dynamic_blocks = dynamic_blocks
        self.cache = {}  # {(pass_type, block_idx): (encoder_hs, hidden_hs)}
        self.mode = "store"  # "store" = run all blocks and cache; "cached" = use cache for stable blocks
        self.pass_type = "cond"  # "cond" or "uncond"
        self._original_forwards = {}
        self._install_wrappers(transformer)

    def _install_wrappers(self, transformer):
        """Replace each block's forward with a caching wrapper."""
        for idx, block in enumerate(transformer.transformer_blocks):
            original_forward = block.forward
            self._original_forwards[idx] = original_forward

            def make_wrapper(block_idx, orig_fwd):
                def wrapped_forward(*args, **kwargs):
                    cache_key = (self.pass_type, block_idx)

                    if self.mode == "cached" and block_idx in self.cached_blocks:
                        # Return cached output — skip computation
                        return self.cache[cache_key]

                    # Run the block normally
                    result = orig_fwd(*args, **kwargs)

                    # Store output in cache
                    self.cache[cache_key] = result
                    return result

                return wrapped_forward

            block.forward = make_wrapper(idx, original_forward)

    def set_mode(self, mode, pass_type):
        """Set caching mode and pass type."""
        self.mode = mode
        self.pass_type = pass_type

    def restore_forwards(self, transformer):
        """Restore original forward methods."""
        for idx, block in enumerate(transformer.transformer_blocks):
            if idx in self._original_forwards:
                block.forward = self._original_forwards[idx]
        self._original_forwards.clear()
        self.cache.clear()


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


def run_inference(image_list, prompt, negative_prompt, seed, schedule=None):
    """Run inference with optional temporal caching.

    schedule=None: baseline (no caching)
    schedule=dict: apply temporal caching per schedule definition
    """
    pipeline._guidance_scale = 1.0
    pipeline._attention_kwargs = {}
    pipeline._current_timestep = None
    pipeline._interrupt = False

    torch.cuda.reset_peak_memory_stats()

    condition_images, vae_images, vae_image_sizes, height, width = prepare_inputs(
        image_list
    )

    # Block cache manager (only for block-aware schedule)
    cache_mgr = None
    if schedule and schedule["type"] == "block_aware":
        cache_mgr = BlockCacheManager(
            pipeline.transformer,
            schedule["cached_blocks"],
            schedule["dynamic_blocks"],
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
        prev_final_pred = None  # For uniform temporal caching

        torch.cuda.synchronize()
        t_loop_start = time.perf_counter()

        for i, t in enumerate(timesteps):
            is_cached = schedule is not None and (i in schedule["cached_steps"])

            # ── UNIFORM CACHING: skip transformer entirely ──
            if is_cached and schedule["type"] == "uniform":
                final_pred = prev_final_pred
            else:
                # ── BLOCK-AWARE: set cache mode ──
                if cache_mgr is not None:
                    if is_cached:
                        cache_mgr.set_mode("cached", "cond")
                    else:
                        cache_mgr.set_mode("store", "cond")

                latent_model_input = torch.cat([latents, image_latents], dim=1)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # Conditional pass
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

                # Set uncond cache mode for block-aware
                if cache_mgr is not None:
                    if is_cached:
                        cache_mgr.set_mode("cached", "uncond")
                    else:
                        cache_mgr.set_mode("store", "uncond")

                # Unconditional pass
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

                prev_final_pred = final_pred

            # Scheduler step (always runs, even for cached steps)
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
        output_image = pipeline.image_processor.postprocess(decoded, output_type="pil")[
            0
        ]

    # Clean up block cache manager
    if cache_mgr is not None:
        cache_mgr.restore_forwards(pipeline.transformer)

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

# Storage for all results
all_results = {}  # {schedule_name: [per-test-case results]}
baseline_results = []
baseline_images = {}

# ── Run baselines first ──
print("=" * 70)
print("PHASE 1: BASELINES")
print("=" * 70)

for tc_idx, (tc_name, img_specs, prompt, neg_prompt) in enumerate(TEST_CASES):
    print(f"\n  [{tc_idx+1}/5] {tc_name} — BASELINE")
    images = make_images(img_specs)
    img_out, t_loop, vram = run_inference(images, prompt, neg_prompt, SEED, schedule=None)
    print(f"    Time: {t_loop:.2f}s, VRAM: {vram:.2f} GB")

    baseline_path = os.path.join(OUTPUT_DIR, f"{tc_name}_baseline.png")
    img_out.save(baseline_path)
    baseline_images[tc_name] = img_out
    baseline_results.append({
        "test_case": tc_name,
        "num_images": len(img_specs),
        "baseline_time_s": round(t_loop, 2),
        "vram_gb": round(vram, 2),
    })

print("\nBaseline timings:")
for r in baseline_results:
    print(f"  {r['test_case']}: {r['baseline_time_s']:.2f}s")

# ── Run each schedule ──
for sched in SCHEDULES:
    sched_name = sched["name"]
    sched_label = sched["label"]
    print(f"\n{'='*70}")
    print(f"SCHEDULE: {sched_label}")
    print(f"{'='*70}")

    sched_results = []

    for tc_idx, (tc_name, img_specs, prompt, neg_prompt) in enumerate(TEST_CASES):
        print(f"\n  [{tc_idx+1}/5] {tc_name} — {sched_name}")
        images = make_images(img_specs)

        img_out, t_loop, vram = run_inference(
            images, prompt, neg_prompt, SEED, schedule=sched
        )

        # Quality metrics vs baseline
        metrics = compute_metrics(baseline_images[tc_name], img_out)
        baseline_t = baseline_results[tc_idx]["baseline_time_s"]
        speedup = baseline_t / t_loop

        print(f"    Time: {t_loop:.2f}s (speedup {speedup:.2f}x)")
        print(f"    SSIM: {metrics['ssim']:.4f}, MSE: {metrics['mse']:.1f}, PSNR: {metrics['psnr']:.1f} dB")

        # Save image
        img_path = os.path.join(OUTPUT_DIR, f"{tc_name}_{sched_name}.png")
        img_out.save(img_path)

        sched_results.append({
            "test_case": tc_name,
            "num_images": len(img_specs),
            "baseline_time_s": baseline_t,
            "cached_time_s": round(t_loop, 2),
            "speedup": round(speedup, 3),
            "vram_gb": round(vram, 2),
            **{k: round(v, 4) if k != "mse" else round(v, 1) for k, v in metrics.items()},
        })

    all_results[sched_name] = sched_results

# ═══════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

for sched in SCHEDULES:
    sched_name = sched["name"]
    sched_label = sched["label"]
    results = all_results[sched_name]

    print(f"\n── {sched_label} ──")
    print(f"{'Test Case':<25s} {'Base(s)':>8s} {'Cache(s)':>8s} {'Speedup':>8s} "
          f"{'SSIM':>7s} {'MSE':>9s} {'PSNR':>8s} {'VRAM':>7s}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*9} {'-'*8} {'-'*7}")

    for r in results:
        psnr_str = f"{r['psnr']:.1f}" if r['psnr'] < 100 else "inf"
        print(f"{r['test_case']:<25s} {r['baseline_time_s']:8.2f} {r['cached_time_s']:8.2f} "
              f"{r['speedup']:8.3f} {r['ssim']:7.4f} {r['mse']:9.1f} {psnr_str:>8s} "
              f"{r['vram_gb']:7.2f}")

    speedups = [r["speedup"] for r in results]
    ssims = [r["ssim"] for r in results]
    print(f"\n  Mean speedup: {np.mean(speedups):.3f}x  "
          f"Mean SSIM: {np.mean(ssims):.4f}  "
          f"SSIM range: {min(ssims):.4f}-{max(ssims):.4f}")

# ── Cross-schedule comparison ──
print(f"\n{'='*70}")
print("CROSS-SCHEDULE COMPARISON")
print(f"{'='*70}")
print(f"{'Test Case':<25s} {'Aggr Speed':>10s} {'Aggr SSIM':>10s} "
      f"{'Cons Speed':>10s} {'Cons SSIM':>10s} "
      f"{'Block Speed':>11s} {'Block SSIM':>11s}")
print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*11} {'-'*11}")

for tc_idx, (tc_name, _, _, _) in enumerate(TEST_CASES):
    a = all_results["aggressive_uniform"][tc_idx]
    b = all_results["conservative_uniform"][tc_idx]
    c = all_results["block_aware"][tc_idx]
    print(f"{tc_name:<25s} {a['speedup']:10.3f} {a['ssim']:10.4f} "
          f"{b['speedup']:10.3f} {b['ssim']:10.4f} "
          f"{c['speedup']:11.3f} {c['ssim']:11.4f}")

# ── Quality assessment ──
print("\nQuality Assessment (per schedule):")
for sched in SCHEDULES:
    sched_name = sched["name"]
    results = all_results[sched_name]
    print(f"\n  {sched['label']}:")
    for r in results:
        if r["ssim"] < 0.80:
            label = "SEVERE DEGRADATION"
        elif r["ssim"] < 0.90:
            label = "NOTABLE degradation"
        elif r["ssim"] < 0.95:
            label = "moderate difference"
        else:
            label = "good quality"
        print(f"    {r['test_case']}: SSIM={r['ssim']:.4f} — {label}")

# ── Save results JSON ──
results_json = {
    "config": {
        "num_steps": NUM_STEPS,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "seed": SEED,
        "schedules": {
            s["name"]: {
                "label": s["label"],
                "cached_steps": sorted(s["cached_steps"]),
                "type": s["type"],
                **({"cached_blocks": sorted(s.get("cached_blocks", [])),
                    "dynamic_blocks": sorted(s.get("dynamic_blocks", []))}
                   if s["type"] == "block_aware" else {}),
            }
            for s in SCHEDULES
        },
    },
    "baseline": baseline_results,
    "per_schedule": all_results,
}

json_path = os.path.join(OUTPUT_DIR, "temporal_cache_results.json")
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\nResults saved to {json_path}")
print("\nDone.")
