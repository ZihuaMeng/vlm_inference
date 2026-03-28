#!/usr/bin/env python3
"""F1 Phase 3: Temporal Redundancy Characterization of FLUX Kontext-dev

Measures step-to-step similarity of transformer outputs across all 28 steps.
Adapted from Qwen M4 methodology. Since Kontext is guidance-distilled
(single pass, no CFG), only the temporal axis is relevant.

Captures:
  - Transformer output (noise prediction) similarity between consecutive steps
  - Hidden states from sampled blocks (5 dual-stream + 5 single-stream)

Metrics: cosine similarity (per-token mean), relative L2 distance
"""

import torch
import time
import json
import os
import numpy as np
from PIL import Image
from collections import defaultdict

# ─── Configuration ───
WEIGHTS = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/weights/FLUX-Kontext-dev"
OUTPUT_DIR = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/profiling/flux"
NUM_STEPS = 28
GUIDANCE_SCALE = 2.5
SEED = 42
# Sample across both block types
SAMPLED_DUAL = [0, 9, 18]         # first, mid, last of 19 dual-stream
SAMPLED_SINGLE = [0, 18, 37]      # first, mid, last of 38 single-stream
SAMPLED_BLOCKS = [(f"dual_{i}", "dual", i) for i in SAMPLED_DUAL] + \
                 [(f"single_{i}", "single", i) for i in SAMPLED_SINGLE]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def cosine_sim_per_token(a, b):
    a_flat = a.squeeze(0).float()
    b_flat = b.squeeze(0).float()
    cos = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1)
    return cos.mean().item()


def relative_l2(a, b):
    a_flat = a.float()
    b_flat = b.float()
    diff_norm = torch.norm(a_flat - b_flat).item()
    a_norm = torch.norm(a_flat).item()
    return diff_norm / max(a_norm, 1e-8)


class BlockCapture:
    """Captures hidden_states (image stream) from sampled blocks.

    Both FluxTransformerBlock and FluxSingleTransformerBlock return
    (encoder_hidden_states, hidden_states) — we capture hidden_states [1].
    """

    def __init__(self):
        self.hooks = []
        self.active = False
        self.captured = {}

    def register(self, key, module):
        def hook(mod, inp, out, *, _key=key):
            if not self.active:
                return
            # Both block types return (encoder_hidden_states, hidden_states)
            self.captured[_key] = out[1].detach()

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
from diffusers import FluxKontextPipeline
import importlib

pipeline = FluxKontextPipeline.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
pipeline.to("cuda")
vram_after_load = torch.cuda.memory_allocated() / 1e9
print(f"Pipeline loaded. VRAM: {vram_after_load:.2f} GB")

flux_mod = importlib.import_module("diffusers.pipelines.flux.pipeline_flux_kontext")
calculate_shift = flux_mod.calculate_shift
retrieve_timesteps = flux_mod.retrieve_timesteps
PREFERRED_KONTEXT_RESOLUTIONS = flux_mod.PREFERRED_KONTEXT_RESOLUTIONS

ref_image = Image.new("RGB", (512, 512), "red")
prompt = "A red cube sitting on a white table with soft lighting."

# ─── Warmup ───
print("\nWarmup pass...")
torch.cuda.synchronize()
warmup_start = time.perf_counter()
with torch.no_grad():
    _ = pipeline(
        image=ref_image, prompt=prompt,
        num_inference_steps=NUM_STEPS, guidance_scale=GUIDANCE_SCALE,
        generator=torch.Generator("cuda").manual_seed(SEED),
    )
torch.cuda.synchronize()
warmup_time = time.perf_counter() - warmup_start
print(f"Warmup: {warmup_time:.1f}s")

# ─── Set up block capture ───
capture = BlockCapture()
for key, btype, idx in SAMPLED_BLOCKS:
    if btype == "dual":
        capture.register(key, pipeline.transformer.transformer_blocks[idx])
    else:
        capture.register(key, pipeline.transformer.single_transformer_blocks[idx])

# ─── Instrumented Pass ───
print("\n" + "=" * 60)
print("Starting instrumented redundancy pass...")
print("=" * 60)

pipeline._guidance_scale = GUIDANCE_SCALE
pipeline._joint_attention_kwargs = {}
pipeline._current_timestep = None
pipeline._interrupt = False

torch.cuda.reset_peak_memory_stats()
multiple_of = pipeline.vae_scale_factor * 2

temporal_transformer = []
temporal_blocks = {key: [] for key, _, _ in SAMPLED_BLOCKS}

with torch.no_grad():
    # Prepare inputs (same as profiling script)
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

    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt, prompt_2=None,
        device="cuda", num_images_per_prompt=1, max_sequence_length=512,
    )

    num_channels_latents = pipeline.transformer.config.in_channels // 4
    latents, image_latents, latent_ids, image_ids = pipeline.prepare_latents(
        image_preprocessed, 1, num_channels_latents, height, width,
        prompt_embeds.dtype, "cuda",
        torch.Generator("cuda").manual_seed(SEED),
    )
    if image_ids is not None:
        latent_ids = torch.cat([latent_ids, image_ids], dim=0)

    noise_len = latents.shape[1]
    print(f"Noise latent tokens: {noise_len}")

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

    prev_output = None
    prev_blocks = {}

    t_loop_start = time.perf_counter()

    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents, image_latents], dim=1)
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        capture.start()
        noise_pred = pipeline.transformer(
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
        # Slice to noise-latent tokens only (exclude reference image tokens)
        curr_output = noise_pred[:, :noise_len].detach()
        capture.stop()
        curr_blocks = capture.get_and_clear()

        # Temporal comparison (step t vs t-1)
        if prev_output is not None:
            temp_cos = cosine_sim_per_token(curr_output, prev_output)
            temp_l2 = relative_l2(curr_output, prev_output)
            temporal_transformer.append({
                "step": i, "cosine_sim": temp_cos, "relative_l2": temp_l2,
            })

            for key, _, _ in SAMPLED_BLOCKS:
                if key in curr_blocks and key in prev_blocks:
                    c = cosine_sim_per_token(curr_blocks[key], prev_blocks[key])
                    l = relative_l2(curr_blocks[key], prev_blocks[key])
                    temporal_blocks[key].append({
                        "step": i, "cosine_sim": c, "relative_l2": l,
                    })

        prev_output = curr_output
        prev_blocks = {key: curr_blocks[key] for key, _, _ in SAMPLED_BLOCKS
                       if key in curr_blocks}

        # Scheduler step
        latents = pipeline.scheduler.step(noise_pred[:, :noise_len], t, latents, return_dict=False)[0]

        if (i + 1) % 7 == 0 or i == 0:
            if temporal_transformer and temporal_transformer[-1]["step"] == i:
                print(f"  Step {i:2d}/{NUM_STEPS}: "
                      f"temporal_cos={temporal_transformer[-1]['cosine_sim']:.6f}")
            else:
                print(f"  Step {i:2d}/{NUM_STEPS}: (first step, no temporal ref)")

    torch.cuda.synchronize()
    t_loop_end = time.perf_counter()

capture.remove_hooks()
peak_vram_gb = torch.cuda.max_memory_reserved() / 1e9
loop_time = t_loop_end - t_loop_start

print(f"\nInstrumented loop time: {loop_time:.1f}s")
print(f"Peak VRAM: {peak_vram_gb:.2f} GB")

# ═══════════════════════════════════════════
#  ANALYSIS
# ═══════════════════════════════════════════

print("\n" + "=" * 70)
print("TEMPORAL REDUNDANCY (step-to-step similarity)")
print("=" * 70)

print(f"\n{'Step':>6s} {'Cos Sim':>10s} {'Rel L2':>10s}")
print(f"{'-'*6} {'-'*10} {'-'*10}")
for entry in temporal_transformer:
    print(f"{entry['step']:6d} {entry['cosine_sim']:10.6f} {entry['relative_l2']:10.6f}")

cos_vals = [e["cosine_sim"] for e in temporal_transformer]
l2_vals = [e["relative_l2"] for e in temporal_transformer]
print(f"\nTransformer output summary:")
print(f"  Cosine sim: min={min(cos_vals):.6f}  max={max(cos_vals):.6f}  mean={np.mean(cos_vals):.6f}")
print(f"  Relative L2: min={min(l2_vals):.6f}  max={max(l2_vals):.6f}  mean={np.mean(l2_vals):.6f}")

# Identify plateau region (cos > threshold)
PLATEAU_THRESHOLD = 0.999
plateau_steps = [e["step"] for e in temporal_transformer if e["cosine_sim"] >= PLATEAU_THRESHOLD]
if plateau_steps:
    print(f"\n  Plateau (cos >= {PLATEAU_THRESHOLD}): steps {min(plateau_steps)}-{max(plateau_steps)} "
          f"({len(plateau_steps)} steps)")
else:
    print(f"\n  No plateau found at cos >= {PLATEAU_THRESHOLD}")

# Three-phase analysis
early_cos = np.mean(cos_vals[:5])
mid_cos = np.mean(cos_vals[len(cos_vals)//3:2*len(cos_vals)//3])
late_cos = np.mean(cos_vals[-5:])
print(f"\n  Three-phase pattern:")
print(f"    Early (first 5):  mean cos = {early_cos:.6f}")
print(f"    Middle (mid 1/3): mean cos = {mid_cos:.6f}")
print(f"    Late (last 5):    mean cos = {late_cos:.6f}")

if early_cos < mid_cos > late_cos:
    print("    Pattern: convergence → plateau → divergence (same as Qwen)")
elif early_cos < mid_cos and mid_cos <= late_cos:
    print("    Pattern: convergence → plateau (no divergence)")
elif early_cos >= mid_cos:
    print("    Pattern: no convergence phase detected")
else:
    print("    Pattern: other")

# Block-level analysis
print(f"\nBlock-level temporal redundancy (cosine sim):")
print(f"{'Block':>12s} {'Type':>8s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'First5':>10s} {'Last5':>10s}")
print(f"{'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for key, btype, idx in SAMPLED_BLOCKS:
    if temporal_blocks[key]:
        bcos = [e["cosine_sim"] for e in temporal_blocks[key]]
        print(f"{key:>12s} {btype:>8s} {min(bcos):10.6f} {max(bcos):10.6f} {np.mean(bcos):10.6f} "
              f"{np.mean(bcos[:5]):10.6f} {np.mean(bcos[-5:]):10.6f}")

# ─── Comparison to Qwen ───
print("\n" + "=" * 70)
print("COMPARISON TO QWEN TEMPORAL PROFILE")
print("=" * 70)
print("""
Qwen (40 steps, 80 passes — M4 findings):
  - Steps 1-4: convergence (cos 0.98 → 0.9995)
  - Steps 5-30: plateau (cos > 0.999)
  - Steps 31-39: divergence (step 39 cos=0.816)
  - Block 44 most dynamic, block 59 most stable

Kontext (28 steps, 28 passes — current findings):""")
print(f"  - Trajectory has {len(cos_vals)} step-transitions")
print(f"  - Min cos: {min(cos_vals):.6f} at step {temporal_transformer[np.argmin(cos_vals)]['step']}")
print(f"  - Max cos: {max(cos_vals):.6f} at step {temporal_transformer[np.argmax(cos_vals)]['step']}")
if plateau_steps:
    print(f"  - Plateau: steps {min(plateau_steps)}-{max(plateau_steps)} ({len(plateau_steps)} steps)")
    plateau_pct = len(plateau_steps) / len(cos_vals) * 100
    print(f"  - Plateau coverage: {plateau_pct:.0f}% of trajectory (Qwen: 65%)")
else:
    print("  - No plateau detected — temporal caching may NOT transfer from Qwen")

# ─── Save ───
results = {
    "config": {
        "num_steps": NUM_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "seed": SEED,
        "sampled_blocks": [(key, btype, idx) for key, btype, idx in SAMPLED_BLOCKS],
        "warmup_time_s": round(warmup_time, 1),
        "loop_time_s": round(loop_time, 1),
        "peak_vram_gb": round(peak_vram_gb, 2),
    },
    "temporal_transformer": temporal_transformer,
    "temporal_blocks": {k: v for k, v in temporal_blocks.items()},
    "plateau_threshold": PLATEAU_THRESHOLD,
    "plateau_steps": plateau_steps,
}

json_path = os.path.join(OUTPUT_DIR, "f1_redundancy_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nRaw data saved to {json_path}")
print("\nDone.")
