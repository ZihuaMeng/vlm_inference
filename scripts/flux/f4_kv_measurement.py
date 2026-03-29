#!/usr/bin/env python3
"""F4 Phase 2: Measure condition-token K/V stability across denoising steps.

Before implementing any KV caching, this script measures how much the K/V
projections for condition tokens (text + image) change across consecutive
steps in the plateau region, compared to noise tokens.

This validates the premise: if condition-token K/V changes as much as
noise-token K/V, selective caching is not justified.

Measures:
  - Per-step cosine similarity of K and V projections for:
    1. Noise tokens (should change — they carry the evolving signal)
    2. Image-condition tokens (premise: should change less)
    3. Text tokens (premise: should change least)
  - Sampled across block types: dual-stream and single-stream
  - Across all 28 denoising steps
"""

import torch
import time
import json
import os
import numpy as np
from PIL import Image
import importlib

# ─── Configuration ───
WEIGHTS = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/weights/FLUX-Kontext-dev"
OUTPUT_DIR = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/outputs/flux/f4_kv_reuse"
NUM_STEPS = 28
GUIDANCE_SCALE = 2.5
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Blocks to sample (indices into transformer_blocks and single_transformer_blocks)
DUAL_BLOCKS = [0, 9, 18]       # first, middle, last of 19 dual-stream
SINGLE_BLOCKS = [0, 18, 37]    # first, middle, last of 38 single-stream

# ─── Load Pipeline ───
print("Loading pipeline...")
from diffusers import FluxKontextPipeline
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor, _get_qkv_projections

pipeline = FluxKontextPipeline.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
pipeline.to("cuda")
print(f"Pipeline loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

flux_mod = importlib.import_module("diffusers.pipelines.flux.pipeline_flux_kontext")
calculate_shift = flux_mod.calculate_shift
retrieve_timesteps = flux_mod.retrieve_timesteps
PREFERRED_KONTEXT_RESOLUTIONS = flux_mod.PREFERRED_KONTEXT_RESOLUTIONS

multiple_of = pipeline.vae_scale_factor * 2


# ─── Instrumented Attention Processor ───
class KVMeasurementProcessor:
    """Replaces FluxAttnProcessor to capture K/V projections per token type."""

    def __init__(self, block_type, block_idx, storage):
        self.block_type = block_type  # "dual" or "single"
        self.block_idx = block_idx
        self.storage = storage
        self._attention_backend = None
        self._parallel_config = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        from diffusers.models.transformers.transformer_flux import (
            _get_qkv_projections, apply_rotary_emb, dispatch_attention_fn,
        )

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            # ── CAPTURE K/V before concatenation ──
            # For dual-stream blocks:
            #   hidden_states K/V = [noise_tokens | image_tokens]
            #   encoder K/V = text_tokens
            if self.storage["active"]:
                step = self.storage["current_step"]
                noise_len = self.storage["noise_len"]
                key_id = f"dual_{self.block_idx}"

                # Separate noise vs image-condition K/V
                noise_k = key[:, :noise_len].detach().float().cpu()
                image_k = key[:, noise_len:].detach().float().cpu()
                text_k = encoder_key.detach().float().cpu()

                noise_v = value[:, :noise_len].detach().float().cpu()
                image_v = value[:, noise_len:].detach().float().cpu()
                text_v = encoder_value.detach().float().cpu()

                self.storage["kv_data"][key_id][step] = {
                    "noise_k": noise_k, "noise_v": noise_v,
                    "image_k": image_k, "image_v": image_v,
                    "text_k": text_k, "text_v": text_v,
                }

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        else:
            # ── Single-stream block ──
            # hidden_states = [text | noise | image] already concatenated
            if self.storage["active"]:
                step = self.storage["current_step"]
                text_len = self.storage["text_len"]
                noise_len = self.storage["noise_len"]
                key_id = f"single_{self.block_idx}"

                text_k = key[:, :text_len].detach().float().cpu()
                noise_k = key[:, text_len:text_len + noise_len].detach().float().cpu()
                image_k = key[:, text_len + noise_len:].detach().float().cpu()

                text_v = value[:, :text_len].detach().float().cpu()
                noise_v = value[:, text_len:text_len + noise_len].detach().float().cpu()
                image_v = value[:, text_len + noise_len:].detach().float().cpu()

                self.storage["kv_data"][key_id][step] = {
                    "noise_k": noise_k, "noise_v": noise_v,
                    "image_k": image_k, "image_v": image_v,
                    "text_k": text_k, "text_v": text_v,
                }

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def install_hooks(transformer, storage):
    """Install measurement processors on sampled blocks."""
    for idx in DUAL_BLOCKS:
        block = transformer.transformer_blocks[idx]
        block.attn.set_processor(KVMeasurementProcessor("dual", idx, storage))
        storage["kv_data"][f"dual_{idx}"] = {}

    for idx in SINGLE_BLOCKS:
        block = transformer.single_transformer_blocks[idx]
        block.attn.set_processor(KVMeasurementProcessor("single", idx, storage))
        storage["kv_data"][f"single_{idx}"] = {}


def restore_processors(transformer):
    """Restore default attention processors."""
    for block in transformer.transformer_blocks:
        block.attn.set_processor(FluxAttnProcessor())
    for block in transformer.single_transformer_blocks:
        block.attn.set_processor(FluxAttnProcessor())


def run_instrumented_inference(ref_image, prompt, storage):
    """Run a full inference pass with K/V measurement hooks active."""
    pipeline._guidance_scale = GUIDANCE_SCALE
    pipeline._joint_attention_kwargs = {}
    pipeline._current_timestep = None
    pipeline._interrupt = False

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
            torch.Generator("cuda").manual_seed(SEED),
        )
        if image_ids is not None:
            latent_ids = torch.cat([latent_ids, image_ids], dim=0)

        noise_len = latents.shape[1]
        text_len = prompt_embeds.shape[1]
        img_cond_len = image_latents.shape[1] if image_latents is not None else 0

        # Record token dimensions
        storage["noise_len"] = noise_len
        storage["text_len"] = text_len
        storage["img_cond_len"] = img_cond_len
        print(f"  Token counts: noise={noise_len}, text={text_len}, image_cond={img_cond_len}")
        print(f"  Total sequence in attention: {text_len + noise_len + img_cond_len}")

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

        # Install hooks
        install_hooks(pipeline.transformer, storage)
        storage["active"] = True

        # Denoising loop
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        for i, t in enumerate(timesteps):
            storage["current_step"] = i

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
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        torch.cuda.synchronize()
        t_end = time.perf_counter()

        storage["active"] = False
        restore_processors(pipeline.transformer)

    print(f"  Instrumented pass: {t_end - t_start:.2f}s")
    return t_end - t_start


def compute_kv_stability(storage):
    """Compute cosine similarity of K/V between consecutive steps."""
    results = {}

    for block_key, step_data in storage["kv_data"].items():
        steps = sorted(step_data.keys())
        block_results = {
            "noise_k_cos": [], "noise_v_cos": [],
            "image_k_cos": [], "image_v_cos": [],
            "text_k_cos": [], "text_v_cos": [],
            "noise_k_l2": [], "noise_v_l2": [],
            "image_k_l2": [], "image_v_l2": [],
            "text_k_l2": [], "text_v_l2": [],
            "steps": [],
        }

        for i in range(len(steps) - 1):
            s0 = steps[i]
            s1 = steps[i + 1]
            d0 = step_data[s0]
            d1 = step_data[s1]

            block_results["steps"].append(f"{s0}->{s1}")

            for token_type in ["noise", "image", "text"]:
                for proj_type in ["k", "v"]:
                    key = f"{token_type}_{proj_type}"
                    t0 = d0[key].flatten().float()
                    t1 = d1[key].flatten().float()

                    cos_sim = float(torch.nn.functional.cosine_similarity(
                        t0.unsqueeze(0), t1.unsqueeze(0)
                    ))
                    rel_l2 = float(torch.norm(t1 - t0) / (torch.norm(t0) + 1e-8))

                    block_results[f"{key}_cos"].append(round(cos_sim, 6))
                    block_results[f"{key}_l2"].append(round(rel_l2, 6))

        results[block_key] = block_results

    return results


# ═══════════════════════════════════════════
#  MAIN
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

# Run measurement
print("=" * 70)
print("K/V STABILITY MEASUREMENT")
print("=" * 70)

ref_img = Image.new("RGB", (512, 512), "red")
prompt = "A red cube sitting on a white table with soft lighting."

storage = {
    "active": False,
    "current_step": 0,
    "noise_len": 0,
    "text_len": 0,
    "img_cond_len": 0,
    "kv_data": {},
}

print(f"\nRunning instrumented pass...")
elapsed = run_instrumented_inference(ref_img, prompt, storage)

# Compute stability metrics
print(f"\nComputing K/V stability...")
stability = compute_kv_stability(storage)

# ═══════════════════════════════════════════
#  ANALYSIS
# ═══════════════════════════════════════════

print(f"\n{'='*70}")
print("K/V STABILITY RESULTS")
print(f"{'='*70}")

# Define plateau region (steps 4-23 from F1)
plateau_start, plateau_end = 4, 23

all_summary = {}

for block_key in sorted(stability.keys()):
    data = stability[block_key]
    print(f"\n── {block_key} ──")

    # Filter to plateau region
    plateau_indices = []
    for i, step_label in enumerate(data["steps"]):
        s0 = int(step_label.split("->")[0])
        if plateau_start <= s0 < plateau_end:
            plateau_indices.append(i)

    block_summary = {}

    for token_type in ["noise", "image", "text"]:
        for proj_type in ["k", "v"]:
            key = f"{token_type}_{proj_type}"
            all_vals = data[f"{key}_cos"]
            plateau_vals = [all_vals[i] for i in plateau_indices]

            mean_all = np.mean(all_vals) if all_vals else 0
            mean_plateau = np.mean(plateau_vals) if plateau_vals else 0
            min_plateau = min(plateau_vals) if plateau_vals else 0

            block_summary[f"{key}_cos_mean_all"] = round(float(mean_all), 6)
            block_summary[f"{key}_cos_mean_plateau"] = round(float(mean_plateau), 6)
            block_summary[f"{key}_cos_min_plateau"] = round(float(min_plateau), 6)

    all_summary[block_key] = block_summary

    # Print per-block summary
    print(f"  {'Token Type':<12s} {'Proj':<5s} {'Plateau Mean Cos':>18s} {'Plateau Min Cos':>16s} {'All-Steps Mean':>16s}")
    print(f"  {'-'*12} {'-'*5} {'-'*18} {'-'*16} {'-'*16}")
    for token_type in ["noise", "image", "text"]:
        for proj_type in ["k", "v"]:
            key = f"{token_type}_{proj_type}"
            s = block_summary
            print(f"  {token_type:<12s} {proj_type:<5s} "
                  f"{s[f'{key}_cos_mean_plateau']:18.6f} "
                  f"{s[f'{key}_cos_min_plateau']:16.6f} "
                  f"{s[f'{key}_cos_mean_all']:16.6f}")

# Cross-block comparison: is there a consistent pattern?
print(f"\n{'='*70}")
print("CROSS-BLOCK SUMMARY: PLATEAU REGION (steps 4-23)")
print(f"{'='*70}")

# Average across all blocks
token_types = ["noise", "image", "text"]
proj_types = ["k", "v"]

print(f"\n  {'Token':<8s} {'Proj':<5s}", end="")
for bk in sorted(all_summary.keys()):
    short = bk.replace("dual_", "D").replace("single_", "S")
    print(f" {short:>8s}", end="")
print(f" {'MEAN':>8s}")

print(f"  {'-'*8} {'-'*5}", end="")
for bk in sorted(all_summary.keys()):
    print(f" {'-'*8}", end="")
print(f" {'-'*8}")

for tt in token_types:
    for pt in proj_types:
        key = f"{tt}_{pt}"
        print(f"  {tt:<8s} {pt:<5s}", end="")
        vals = []
        for bk in sorted(all_summary.keys()):
            v = all_summary[bk][f"{key}_cos_mean_plateau"]
            vals.append(v)
            print(f" {v:8.6f}", end="")
        mean_v = np.mean(vals)
        print(f" {mean_v:8.6f}")

# Print per-step detail for one representative block
rep_block = f"dual_{DUAL_BLOCKS[1]}"
if rep_block in stability:
    data = stability[rep_block]
    print(f"\n{'='*70}")
    print(f"PER-STEP DETAIL: {rep_block}")
    print(f"{'='*70}")
    print(f"  {'Step':<8s} {'noise_k':>9s} {'noise_v':>9s} {'image_k':>9s} {'image_v':>9s} {'text_k':>9s} {'text_v':>9s}")
    print(f"  {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for i, step_label in enumerate(data["steps"]):
        s0 = int(step_label.split("->")[0])
        marker = " *" if plateau_start <= s0 < plateau_end else ""
        print(f"  {step_label:<8s} "
              f"{data['noise_k_cos'][i]:9.6f} {data['noise_v_cos'][i]:9.6f} "
              f"{data['image_k_cos'][i]:9.6f} {data['image_v_cos'][i]:9.6f} "
              f"{data['text_k_cos'][i]:9.6f} {data['text_v_cos'][i]:9.6f}{marker}")

# Verdict
print(f"\n{'='*70}")
print("PREMISE VALIDATION")
print(f"{'='*70}")

# Compute mean plateau cosine for each token type across all blocks
noise_cos = []
image_cos = []
text_cos = []
for bk in all_summary:
    noise_cos.append(all_summary[bk]["noise_k_cos_mean_plateau"])
    noise_cos.append(all_summary[bk]["noise_v_cos_mean_plateau"])
    image_cos.append(all_summary[bk]["image_k_cos_mean_plateau"])
    image_cos.append(all_summary[bk]["image_v_cos_mean_plateau"])
    text_cos.append(all_summary[bk]["text_k_cos_mean_plateau"])
    text_cos.append(all_summary[bk]["text_v_cos_mean_plateau"])

mean_noise = np.mean(noise_cos)
mean_image = np.mean(image_cos)
mean_text = np.mean(text_cos)

print(f"\n  Mean plateau cosine similarity (K+V combined):")
print(f"    Noise tokens:           {mean_noise:.6f}")
print(f"    Image condition tokens: {mean_image:.6f}")
print(f"    Text tokens:            {mean_text:.6f}")

gap_image_noise = mean_image - mean_noise
gap_text_noise = mean_text - mean_noise

print(f"\n  Stability gap (vs noise):")
print(f"    Image - Noise:  {gap_image_noise:+.6f}")
print(f"    Text  - Noise:  {gap_text_noise:+.6f}")

if mean_image > 0.999 and mean_text > 0.999:
    verdict = "STRONG SUPPORT"
    detail = "Condition K/V is near-static (cos > 0.999). Caching is well-justified."
elif mean_image > mean_noise + 0.001:
    verdict = "MODERATE SUPPORT"
    detail = "Condition K/V is more stable than noise K/V. Caching may provide quality advantage."
else:
    verdict = "PREMISE INVALIDATED"
    detail = "Condition K/V changes as much as noise K/V. Selective caching is not justified."

print(f"\n  Verdict: {verdict}")
print(f"  {detail}")

# Save results
save_data = {
    "config": {
        "model": "FLUX.1 Kontext-dev",
        "num_steps": NUM_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "seed": SEED,
        "plateau_region": [plateau_start, plateau_end],
        "dual_blocks_sampled": DUAL_BLOCKS,
        "single_blocks_sampled": SINGLE_BLOCKS,
    },
    "token_counts": {
        "noise": storage["noise_len"],
        "text": storage["text_len"],
        "image_condition": storage["img_cond_len"],
    },
    "per_block_summary": all_summary,
    "per_step_detail": {
        block_key: {
            "steps": data["steps"],
            "noise_k_cos": data["noise_k_cos"],
            "noise_v_cos": data["noise_v_cos"],
            "image_k_cos": data["image_k_cos"],
            "image_v_cos": data["image_v_cos"],
            "text_k_cos": data["text_k_cos"],
            "text_v_cos": data["text_v_cos"],
        }
        for block_key, data in stability.items()
    },
    "verdict": {
        "mean_noise_cos": round(float(mean_noise), 6),
        "mean_image_cos": round(float(mean_image), 6),
        "mean_text_cos": round(float(mean_text), 6),
        "gap_image_noise": round(float(gap_image_noise), 6),
        "gap_text_noise": round(float(gap_text_noise), 6),
        "verdict": verdict,
    },
    "elapsed_s": round(elapsed, 2),
}

json_path = os.path.join(OUTPUT_DIR, "f4_kv_measurement.json")
with open(json_path, "w") as f:
    json.dump(save_data, f, indent=2)
print(f"\nResults saved to {json_path}")

# Free KV data from memory
del storage["kv_data"]
torch.cuda.empty_cache()

print("\nDone.")
