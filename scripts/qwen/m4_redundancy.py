#!/usr/bin/env python3
"""Milestone 4: Redundancy Characterization

Measures two axes of redundancy in the denoising loop:
  Axis 1 — Temporal: step-to-step similarity of hidden states
  Axis 2 — CFG: conditional-vs-unconditional divergence per step

Representations compared:
  - Transformer output (noise prediction), sliced to noise-latent tokens
  - Image-stream hidden states from 5 sampled blocks (0, 14, 29, 44, 59)

Metrics:
  - Cosine similarity (per-token mean)
  - Relative L2 distance: ||a - b|| / ||a||

Memory strategy: online comparison — store only previous step's cond outputs
(~150 MB), compute similarity, overwrite. No full trajectory accumulation.
"""

import torch
import time
import json
import os
import contextlib
import numpy as np
from PIL import Image
from collections import defaultdict

# ─── Configuration ───
WEIGHTS = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/weights/Qwen-Image-Edit-2511"
OUTPUT_DIR = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/profiling/redundancy"
NUM_STEPS = 40
TRUE_CFG_SCALE = 4.0
SEED = 42
SAMPLED_BLOCKS = [0, 14, 29, 44, 59]  # 5 blocks spanning early to late

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Similarity Metrics ───
def cosine_sim_per_token(a, b):
    """Mean cosine similarity across tokens. a, b: [1, seq_len, dim]."""
    a_flat = a.squeeze(0).float()  # [seq_len, dim]
    b_flat = b.squeeze(0).float()
    cos = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1)  # [seq_len]
    return cos.mean().item()


def relative_l2(a, b):
    """||a - b|| / ||a||. a, b: [1, seq_len, dim]."""
    a_flat = a.float()
    b_flat = b.float()
    diff_norm = torch.norm(a_flat - b_flat).item()
    a_norm = torch.norm(a_flat).item()
    return diff_norm / max(a_norm, 1e-8)


# ─── Block Output Capture ───
class BlockCapture:
    """Captures image-stream hidden states from sampled transformer blocks."""

    def __init__(self):
        self.hooks = []
        self.active = False
        self.captured = {}  # block_idx -> tensor

    def register(self, block_idx, module):
        def hook(mod, inp, out, *, _idx=block_idx):
            if not self.active:
                return
            # out = (encoder_hidden_states, hidden_states)
            # hidden_states = image stream
            self.captured[_idx] = out[1].detach()

        h = module.register_forward_hook(hook)
        self.hooks.append(h)

    def start(self):
        self.active = True
        self.captured = {}

    def stop(self):
        self.active = False

    def get_and_clear(self):
        result = dict(self.captured)
        self.captured = {}
        return result

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Load Pipeline ───
print("Loading pipeline...")
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
pipeline.to("cuda")
vram_after_load = torch.cuda.memory_allocated() / 1e9
print(f"Pipeline loaded. VRAM: {vram_after_load:.2f} GB")

# ─── Prepare Inputs (same as M1–M3) ───
img1 = Image.new("RGB", (512, 512), "red")
img2 = Image.new("RGB", (512, 512), "blue")
prompt = "A red object and a blue object sitting side by side on a white table."
negative_prompt = " "

# ─── Warmup Pass ───
print("\nRunning warmup pass...")
torch.cuda.synchronize()
warmup_start = time.perf_counter()
with torch.no_grad():
    _ = pipeline(
        image=[img1, img2], prompt=prompt,
        num_inference_steps=NUM_STEPS, true_cfg_scale=TRUE_CFG_SCALE,
        negative_prompt=negative_prompt, guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(SEED),
    )
torch.cuda.synchronize()
warmup_time = time.perf_counter() - warmup_start
print(f"Warmup complete: {warmup_time:.1f}s")

# ─── Set up block capture hooks ───
capture = BlockCapture()
for bi in SAMPLED_BLOCKS:
    capture.register(bi, pipeline.transformer.transformer_blocks[bi])

# ─── Import pipeline helpers ───
import importlib
pm = importlib.import_module("diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus")
calculate_dimensions = pm.calculate_dimensions
CONDITION_IMAGE_SIZE = pm.CONDITION_IMAGE_SIZE
VAE_IMAGE_SIZE = pm.VAE_IMAGE_SIZE
calculate_shift = pm.calculate_shift
retrieve_timesteps = pm.retrieve_timesteps

# ─── Instrumented Pass ───
print("\n" + "=" * 60)
print("Starting instrumented redundancy pass...")
print("=" * 60)

# Set pipeline internal state
pipeline._guidance_scale = 1.0
pipeline._attention_kwargs = {}
pipeline._current_timestep = None
pipeline._interrupt = False

torch.cuda.reset_peak_memory_stats()

# Results storage
temporal_transformer = []   # step-to-step similarity of transformer output
temporal_blocks = {bi: [] for bi in SAMPLED_BLOCKS}  # per-block temporal
cfg_transformer = []        # cond-vs-uncond similarity of transformer output
cfg_blocks = {bi: [] for bi in SAMPLED_BLOCKS}       # per-block CFG

with torch.no_grad():
    # ── Prepare inputs (same as M3) ──
    image = [img1, img2]
    image_size = image[-1].size
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
    for img in image:
        iw, ih = img.size
        cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, iw / ih)
        vw, vh = calculate_dimensions(VAE_IMAGE_SIZE, iw / ih)
        vae_image_sizes.append((vw, vh))
        condition_images.append(pipeline.image_processor.resize(img, ch, cw))
        vae_images.append(pipeline.image_processor.preprocess(img, vh, vw).unsqueeze(2))

    # ── Text Encoding ──
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

    # ── VAE Encoding / Prepare Latents ──
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    generator = torch.Generator("cuda").manual_seed(SEED)
    latents, image_latents = pipeline.prepare_latents(
        vae_images, 1, num_channels_latents, height, width,
        prompt_embeds.dtype, "cuda", generator, None,
    )
    img_shapes = [[
        (1, height // pipeline.vae_scale_factor // 2,
         width // pipeline.vae_scale_factor // 2),
        *[(1, vh // pipeline.vae_scale_factor // 2,
           vw // pipeline.vae_scale_factor // 2)
          for vw, vh in vae_image_sizes],
    ]]

    noise_latent_len = latents.size(1)  # number of noise-latent tokens
    print(f"Noise latent tokens: {noise_latent_len}")
    print(f"Total input tokens (latents + image_latents): {latents.size(1) + image_latents.size(1)}")

    # ── Scheduler Setup ──
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

    # Previous step's conditional outputs (for temporal comparison)
    prev_cond_output = None
    prev_cond_blocks = {}

    # ── Denoising Loop ──
    t_loop_start = time.perf_counter()

    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents, image_latents], dim=1)
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        # ── Conditional pass ──
        capture.start()
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
        cond_output = noise_pred[:, :noise_latent_len].detach()
        capture.stop()
        cond_blocks = capture.get_and_clear()

        # ── Unconditional pass ──
        capture.start()
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
        uncond_output = neg_noise_pred[:, :noise_latent_len].detach()
        capture.stop()
        uncond_blocks = capture.get_and_clear()

        # ── Compute Axis 2: CFG redundancy (cond vs uncond) ──
        cfg_cos = cosine_sim_per_token(cond_output, uncond_output)
        cfg_l2 = relative_l2(cond_output, uncond_output)
        cfg_transformer.append({
            "step": i, "cosine_sim": cfg_cos, "relative_l2": cfg_l2
        })

        for bi in SAMPLED_BLOCKS:
            if bi in cond_blocks and bi in uncond_blocks:
                c = cosine_sim_per_token(cond_blocks[bi], uncond_blocks[bi])
                l = relative_l2(cond_blocks[bi], uncond_blocks[bi])
                cfg_blocks[bi].append({
                    "step": i, "cosine_sim": c, "relative_l2": l
                })

        # ── Compute Axis 1: Temporal redundancy (step t vs t-1) ──
        if prev_cond_output is not None:
            temp_cos = cosine_sim_per_token(cond_output, prev_cond_output)
            temp_l2 = relative_l2(cond_output, prev_cond_output)
            temporal_transformer.append({
                "step": i, "cosine_sim": temp_cos, "relative_l2": temp_l2
            })

            for bi in SAMPLED_BLOCKS:
                if bi in cond_blocks and bi in prev_cond_blocks:
                    c = cosine_sim_per_token(cond_blocks[bi], prev_cond_blocks[bi])
                    l = relative_l2(cond_blocks[bi], prev_cond_blocks[bi])
                    temporal_blocks[bi].append({
                        "step": i, "cosine_sim": c, "relative_l2": l
                    })

        # ── Store current as previous ──
        prev_cond_output = cond_output
        prev_cond_blocks = {bi: cond_blocks[bi] for bi in SAMPLED_BLOCKS
                           if bi in cond_blocks}

        # ── Scheduler step (same as normal inference) ──
        noise_pred_full = noise_pred[:, :noise_latent_len]
        neg_pred_full = neg_noise_pred[:, :noise_latent_len]
        comb_pred = neg_pred_full + TRUE_CFG_SCALE * (noise_pred_full - neg_pred_full)
        cond_norm = torch.norm(noise_pred_full, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        final_pred = comb_pred * (cond_norm / noise_norm)

        latents_dtype = latents.dtype
        latents = pipeline.scheduler.step(final_pred, t, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

        # Free uncond tensors
        del uncond_output, uncond_blocks

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Step {i:2d}/{NUM_STEPS}: "
                  f"temporal_cos={temporal_transformer[-1]['cosine_sim']:.6f} "
                  if temporal_transformer and temporal_transformer[-1]['step'] == i
                  else f"  Step {i:2d}/{NUM_STEPS}: (first step, no temporal ref) "
                  f"cfg_cos={cfg_transformer[-1]['cosine_sim']:.6f}")

    torch.cuda.synchronize()
    t_loop_end = time.perf_counter()

capture.remove_hooks()

peak_vram_gb = torch.cuda.max_memory_reserved() / 1e9
loop_time = t_loop_end - t_loop_start

print(f"\nInstrumented loop time: {loop_time:.1f}s")
print(f"Peak VRAM: {peak_vram_gb:.2f} GB (baseline: 63.20 GB)")
print(f"VRAM overhead: {peak_vram_gb - 63.20:+.2f} GB")

# ═════════════════════════════════════════════════════════
#  ANALYSIS
# ═════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("AXIS 1: TEMPORAL REDUNDANCY (step-to-step similarity)")
print("=" * 70)

print(f"\n{'Step':>6s} {'Cos Sim':>10s} {'Rel L2':>10s}")
print(f"{'-'*6} {'-'*10} {'-'*10}")
for entry in temporal_transformer:
    print(f"{entry['step']:6d} {entry['cosine_sim']:10.6f} {entry['relative_l2']:10.6f}")

cos_vals = [e["cosine_sim"] for e in temporal_transformer]
l2_vals = [e["relative_l2"] for e in temporal_transformer]
print(f"\nTransformer output summary:")
print(f"  Cosine sim: min={min(cos_vals):.6f}  max={max(cos_vals):.6f}  "
      f"mean={np.mean(cos_vals):.6f}")
print(f"  Relative L2: min={min(l2_vals):.6f}  max={max(l2_vals):.6f}  "
      f"mean={np.mean(l2_vals):.6f}")

# Check for monotonic convergence
first_5_cos = np.mean(cos_vals[:5])
last_5_cos = np.mean(cos_vals[-5:])
print(f"  First 5 steps avg cos: {first_5_cos:.6f}")
print(f"  Last 5 steps avg cos:  {last_5_cos:.6f}")
print(f"  Trend: {'converging' if last_5_cos > first_5_cos else 'diverging or flat'}")

print(f"\nBlock-level temporal redundancy (cosine sim):")
print(f"{'Block':>8s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'First5':>10s} {'Last5':>10s}")
print(f"{'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for bi in SAMPLED_BLOCKS:
    if temporal_blocks[bi]:
        bcos = [e["cosine_sim"] for e in temporal_blocks[bi]]
        print(f"{bi:8d} {min(bcos):10.6f} {max(bcos):10.6f} {np.mean(bcos):10.6f} "
              f"{np.mean(bcos[:5]):10.6f} {np.mean(bcos[-5:]):10.6f}")

print("\n" + "=" * 70)
print("AXIS 2: CFG REDUNDANCY (conditional vs unconditional)")
print("=" * 70)

print(f"\n{'Step':>6s} {'Cos Sim':>10s} {'Rel L2':>10s}")
print(f"{'-'*6} {'-'*10} {'-'*10}")
for entry in cfg_transformer:
    print(f"{entry['step']:6d} {entry['cosine_sim']:10.6f} {entry['relative_l2']:10.6f}")

cfg_cos_vals = [e["cosine_sim"] for e in cfg_transformer]
cfg_l2_vals = [e["relative_l2"] for e in cfg_transformer]
print(f"\nTransformer output summary:")
print(f"  Cosine sim: min={min(cfg_cos_vals):.6f}  max={max(cfg_cos_vals):.6f}  "
      f"mean={np.mean(cfg_cos_vals):.6f}")
print(f"  Relative L2: min={min(cfg_l2_vals):.6f}  max={max(cfg_l2_vals):.6f}  "
      f"mean={np.mean(cfg_l2_vals):.6f}")

first_5_cfg = np.mean(cfg_cos_vals[:5])
last_5_cfg = np.mean(cfg_cos_vals[-5:])
print(f"  First 5 steps avg cos: {first_5_cfg:.6f}")
print(f"  Last 5 steps avg cos:  {last_5_cfg:.6f}")
print(f"  Trend: cond/uncond {'converge' if last_5_cfg > first_5_cfg else 'diverge'} over trajectory")

print(f"\nBlock-level CFG redundancy (cosine sim):")
print(f"{'Block':>8s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'First5':>10s} {'Last5':>10s}")
print(f"{'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for bi in SAMPLED_BLOCKS:
    if cfg_blocks[bi]:
        bcos = [e["cosine_sim"] for e in cfg_blocks[bi]]
        print(f"{bi:8d} {min(bcos):10.6f} {max(bcos):10.6f} {np.mean(bcos):10.6f} "
              f"{np.mean(bcos[:5]):10.6f} {np.mean(bcos[-5:]):10.6f}")

# ─── Save Raw Data ───
results = {
    "config": {
        "num_steps": NUM_STEPS,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "seed": SEED,
        "sampled_blocks": SAMPLED_BLOCKS,
        "warmup_time_s": round(warmup_time, 1),
        "loop_time_s": round(loop_time, 1),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "vram_overhead_gb": round(peak_vram_gb - 63.20, 2),
    },
    "temporal_transformer": temporal_transformer,
    "temporal_blocks": {str(k): v for k, v in temporal_blocks.items()},
    "cfg_transformer": cfg_transformer,
    "cfg_blocks": {str(k): v for k, v in cfg_blocks.items()},
}

json_path = os.path.join(OUTPUT_DIR, "redundancy_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nRaw data saved to {json_path}")

print("\nDone.")
