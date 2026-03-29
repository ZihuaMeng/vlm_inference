#!/usr/bin/env python3
"""F4 Phase 3+4: Conditioning-Aware KV Reuse Prototype and Evaluation.

Implements K/V caching for condition tokens (text + image) in the plateau
region. On cached steps, reuses previous step's K/V projections for condition
tokens while freshly computing K/V for noise tokens.

Evaluation on 8 cases from the F2 test suite (4 robust + 4 fragile) with
comparison to F2 whole-step caching results.
"""

import torch
import torch.nn.functional as F_torch
import time
import json
import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import lpips
import importlib
from collections import defaultdict

# ─── Configuration ───
WEIGHTS = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/weights/FLUX-Kontext-dev"
OUTPUT_DIR = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/outputs/flux/f4_kv_reuse"
NUM_STEPS = 28
GUIDANCE_SCALE = 2.5
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# F2 test cases: 4 robust (high SSIM) + 4 fragile (low SSIM) for comparison
TEST_CASES = [
    # Robust cases (F2 conservative SSIM > 0.85)
    ("f2_01_simple_object",  "red",    (512, 512),  "A red cube sitting on a white table with soft lighting."),
    ("f2_02_object_modify",  "blue",   (512, 512),  "A blue sphere floating above a marble pedestal."),
    ("f2_05_portrait",       "orange", (512, 768),  "A portrait of a person wearing an orange scarf in winter."),
    ("f2_04_arrangement",    "green",  (512, 512),  "A green vase with white flowers arranged on a wooden table."),
    # Fragile cases (F2 conservative SSIM < 0.75)
    ("f2_09_style_transfer", "purple", (512, 512),  "Transform this into a watercolor painting style with visible brush strokes."),
    ("f2_10_fine_texture",   "cyan",   (640, 480),  "A cyan ceramic vase with intricate floral patterns on a wooden shelf."),
    ("f2_11_complex_scene",  "yellow", (768, 512),  "A bustling farmers market with colorful produce stalls and people shopping."),
    ("f2_12_nature_detail",  "green",  (768, 512),  "A detailed forest floor with mushrooms, fallen leaves, and morning dew drops."),
]

# Plateau region (from F1 Phase 3)
PLATEAU_START = 6   # conservative start (like F2 conservative schedule)
PLATEAU_END = 20    # conservative end


# ─── Load Pipeline ───
print("Loading pipeline...")
from diffusers import FluxKontextPipeline
from diffusers.models.transformers.transformer_flux import (
    FluxAttnProcessor, _get_qkv_projections, apply_rotary_emb,
)
from diffusers.models.attention_dispatch import dispatch_attention_fn

pipeline = FluxKontextPipeline.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
pipeline.to("cuda")
print(f"Pipeline loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

flux_mod = importlib.import_module("diffusers.pipelines.flux.pipeline_flux_kontext")
calculate_shift = flux_mod.calculate_shift
retrieve_timesteps = flux_mod.retrieve_timesteps
PREFERRED_KONTEXT_RESOLUTIONS = flux_mod.PREFERRED_KONTEXT_RESOLUTIONS

multiple_of = pipeline.vae_scale_factor * 2

# LPIPS
lpips_fn = lpips.LPIPS(net='alex').cuda()


# ─── KV-Caching Attention Processor ───
class KVCachingProcessor:
    """Attention processor that caches condition-token K/V across steps.

    In the plateau region:
      - First plateau step: compute all K/V normally, cache condition K/V
      - Subsequent steps: reuse cached condition K/V, compute fresh noise K/V
    Outside plateau: compute everything normally.

    Dual-stream blocks: hidden_states = [noise | image], encoder = text
    Single-stream blocks: hidden_states = [text | noise | image]
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, block_type, cache_store):
        self.block_type = block_type  # "dual" or "single"
        self.cache_store = cache_store  # shared dict for cache state

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):

        is_caching = self.cache_store.get("caching_active", False)
        is_first_cached_step = self.cache_store.get("is_first_cached_step", False)
        block_id = id(attn)  # unique key per block

        if encoder_hidden_states is not None:
            # ── DUAL-STREAM BLOCK ──
            noise_len = self.cache_store.get("noise_len", hidden_states.shape[1])

            if is_caching and not is_first_cached_step and block_id in self.cache_store["kv_cache"]:
                # Reuse cached condition K/V, compute fresh noise K/V
                cached = self.cache_store["kv_cache"][block_id]

                # Fresh Q for all tokens
                query = attn.to_q(hidden_states)
                query = query.unflatten(-1, (attn.heads, -1))
                query = attn.norm_q(query)

                encoder_query = attn.add_q_proj(encoder_hidden_states)
                encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
                encoder_query = attn.norm_added_q(encoder_query)

                # Fresh K/V for noise tokens only
                noise_hidden = hidden_states[:, :noise_len]
                noise_key = attn.to_k(noise_hidden)
                noise_key = noise_key.unflatten(-1, (attn.heads, -1))
                noise_key = attn.norm_k(noise_key)

                noise_value = attn.to_v(noise_hidden)
                noise_value = noise_value.unflatten(-1, (attn.heads, -1))

                # Reconstruct full K/V: [encoder | noise | image_cond]
                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([cached["encoder_key"], noise_key, cached["image_key"]], dim=1)
                value = torch.cat([cached["encoder_value"], noise_value, cached["image_value"]], dim=1)

            else:
                # Full computation (baseline or first cached step)
                query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
                    attn, hidden_states, encoder_hidden_states
                )

                query = query.unflatten(-1, (attn.heads, -1))
                key = key.unflatten(-1, (attn.heads, -1))
                value = value.unflatten(-1, (attn.heads, -1))

                query = attn.norm_q(query)
                key = attn.norm_k(key)

                encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
                encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
                encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

                encoder_query = attn.norm_added_q(encoder_query)
                encoder_key = attn.norm_added_k(encoder_key)

                # Cache condition K/V if this is the first cached step
                if is_caching and is_first_cached_step:
                    self.cache_store["kv_cache"][block_id] = {
                        "encoder_key": encoder_key.detach(),
                        "encoder_value": encoder_value.detach(),
                        "image_key": key[:, noise_len:].detach(),
                        "image_value": value[:, noise_len:].detach(),
                    }

                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)

            # Apply RoPE and attention (same for all paths)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            hidden_states_out = dispatch_attention_fn(
                query, key, value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states_out = hidden_states_out.flatten(2, 3)
            hidden_states_out = hidden_states_out.to(query.dtype)

            encoder_hidden_states_out, hidden_states_out = hidden_states_out.split_with_sizes(
                [encoder_hidden_states.shape[1],
                 hidden_states_out.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states_out = attn.to_out[0](hidden_states_out.contiguous())
            hidden_states_out = attn.to_out[1](hidden_states_out)
            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out.contiguous())

            return hidden_states_out, encoder_hidden_states_out

        else:
            # ── SINGLE-STREAM BLOCK ──
            text_len = self.cache_store.get("text_len", 0)
            noise_len = self.cache_store.get("noise_len", 0)

            if is_caching and not is_first_cached_step and block_id in self.cache_store["kv_cache"]:
                cached = self.cache_store["kv_cache"][block_id]

                # Fresh Q for all tokens
                query = attn.to_q(hidden_states)
                query = query.unflatten(-1, (attn.heads, -1))
                query = attn.norm_q(query)

                # Fresh K/V for noise tokens only
                noise_start = text_len
                noise_end = text_len + noise_len
                noise_hidden = hidden_states[:, noise_start:noise_end]

                noise_key = attn.to_k(noise_hidden)
                noise_key = noise_key.unflatten(-1, (attn.heads, -1))
                noise_key = attn.norm_k(noise_key)

                noise_value = attn.to_v(noise_hidden)
                noise_value = noise_value.unflatten(-1, (attn.heads, -1))

                # Reconstruct full K/V: [text | noise | image_cond]
                key = torch.cat([cached["text_key"], noise_key, cached["image_key"]], dim=1)
                value = torch.cat([cached["text_value"], noise_value, cached["image_value"]], dim=1)

            else:
                query, key, value, _, _, _ = _get_qkv_projections(
                    attn, hidden_states, None
                )

                query = query.unflatten(-1, (attn.heads, -1))
                key = key.unflatten(-1, (attn.heads, -1))
                value = value.unflatten(-1, (attn.heads, -1))

                query = attn.norm_q(query)
                key = attn.norm_k(key)

                # Cache condition K/V if first cached step
                if is_caching and is_first_cached_step:
                    noise_start = text_len
                    noise_end = text_len + noise_len
                    self.cache_store["kv_cache"][block_id] = {
                        "text_key": key[:, :text_len].detach(),
                        "text_value": value[:, :text_len].detach(),
                        "image_key": key[:, noise_end:].detach(),
                        "image_value": value[:, noise_end:].detach(),
                    }

            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            hidden_states_out = dispatch_attention_fn(
                query, key, value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states_out = hidden_states_out.flatten(2, 3)
            hidden_states_out = hidden_states_out.to(query.dtype)

            return hidden_states_out


def install_kv_caching(transformer, cache_store):
    """Install KV-caching processors on all blocks."""
    for block in transformer.transformer_blocks:
        block.attn.set_processor(KVCachingProcessor("dual", cache_store))
    for block in transformer.single_transformer_blocks:
        block.attn.set_processor(KVCachingProcessor("single", cache_store))


def restore_processors(transformer):
    """Restore default processors."""
    for block in transformer.transformer_blocks:
        block.attn.set_processor(FluxAttnProcessor())
    for block in transformer.single_transformer_blocks:
        block.attn.set_processor(FluxAttnProcessor())


# ─── Inference Functions ───
def prepare_inputs(ref_image, prompt):
    """Prepare all inputs for the denoising loop (shared across modes)."""
    pipeline._guidance_scale = GUIDANCE_SCALE
    pipeline._joint_attention_kwargs = {}
    pipeline._current_timestep = None
    pipeline._interrupt = False

    with torch.no_grad():
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
        text_len = prompt_embeds.shape[1]

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

    return {
        "latents": latents, "image_latents": image_latents,
        "prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids, "latent_ids": latent_ids,
        "timesteps": timesteps, "guidance": guidance,
        "noise_len": noise_len, "text_len": text_len,
        "height": height, "width": width,
    }


def run_baseline(inputs):
    """Run standard inference (no caching)."""
    latents = inputs["latents"].clone()
    image_latents = inputs["image_latents"]
    noise_len = inputs["noise_len"]

    pipeline.scheduler._step_index = None
    pipeline.scheduler.set_begin_index(0)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        for i, t in enumerate(inputs["timesteps"]):
            latent_model_input = torch.cat([latents, image_latents], dim=1)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred_full = pipeline.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=inputs["guidance"],
                pooled_projections=inputs["pooled_prompt_embeds"],
                encoder_hidden_states=inputs["prompt_embeds"],
                txt_ids=inputs["text_ids"],
                img_ids=inputs["latent_ids"],
                joint_attention_kwargs=pipeline.joint_attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred_full[:, :noise_len]
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    # Decode
    with torch.no_grad():
        latents_dec = pipeline._unpack_latents(latents, inputs["height"], inputs["width"],
                                               pipeline.vae_scale_factor)
        latents_dec = (latents_dec / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        decoded = pipeline.vae.decode(latents_dec, return_dict=False)[0]
        output_image = pipeline.image_processor.postprocess(decoded, output_type="pil")[0]

    vram = torch.cuda.max_memory_reserved() / 1e9
    return output_image, t_end - t_start, vram


def run_kv_cached(inputs, cache_store):
    """Run inference with condition-token KV caching in plateau."""
    latents = inputs["latents"].clone()
    image_latents = inputs["image_latents"]
    noise_len = inputs["noise_len"]
    text_len = inputs["text_len"]

    cache_store["noise_len"] = noise_len
    cache_store["text_len"] = text_len
    cache_store["kv_cache"] = {}
    cache_store["caching_active"] = False

    install_kv_caching(pipeline.transformer, cache_store)

    pipeline.scheduler._step_index = None
    pipeline.scheduler.set_begin_index(0)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        for i, t in enumerate(inputs["timesteps"]):
            is_plateau = PLATEAU_START <= i <= PLATEAU_END
            is_first = (i == PLATEAU_START)

            cache_store["caching_active"] = is_plateau
            cache_store["is_first_cached_step"] = is_first

            latent_model_input = torch.cat([latents, image_latents], dim=1)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred_full = pipeline.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=inputs["guidance"],
                pooled_projections=inputs["pooled_prompt_embeds"],
                encoder_hidden_states=inputs["prompt_embeds"],
                txt_ids=inputs["text_ids"],
                img_ids=inputs["latent_ids"],
                joint_attention_kwargs=pipeline.joint_attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred_full[:, :noise_len]
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    restore_processors(pipeline.transformer)
    cache_store["kv_cache"] = {}  # free cache memory

    # Decode
    with torch.no_grad():
        latents_dec = pipeline._unpack_latents(latents, inputs["height"], inputs["width"],
                                               pipeline.vae_scale_factor)
        latents_dec = (latents_dec / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        decoded = pipeline.vae.decode(latents_dec, return_dict=False)[0]
        output_image = pipeline.image_processor.postprocess(decoded, output_type="pil")[0]

    vram = torch.cuda.max_memory_reserved() / 1e9
    return output_image, t_end - t_start, vram


def run_temporal_cached(inputs):
    """Run F2-style whole-step temporal caching (for direct comparison)."""
    latents = inputs["latents"].clone()
    image_latents = inputs["image_latents"]
    noise_len = inputs["noise_len"]
    cached_steps = set(range(PLATEAU_START, PLATEAU_END + 1))

    pipeline.scheduler._step_index = None
    pipeline.scheduler.set_begin_index(0)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    prev_noise_pred = None

    with torch.no_grad():
        for i, t in enumerate(inputs["timesteps"]):
            if i in cached_steps and prev_noise_pred is not None:
                noise_pred = prev_noise_pred
            else:
                latent_model_input = torch.cat([latents, image_latents], dim=1)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred_full = pipeline.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=inputs["guidance"],
                    pooled_projections=inputs["pooled_prompt_embeds"],
                    encoder_hidden_states=inputs["prompt_embeds"],
                    txt_ids=inputs["text_ids"],
                    img_ids=inputs["latent_ids"],
                    joint_attention_kwargs=pipeline.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred_full[:, :noise_len]
                prev_noise_pred = noise_pred

            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    # Decode
    with torch.no_grad():
        latents_dec = pipeline._unpack_latents(latents, inputs["height"], inputs["width"],
                                               pipeline.vae_scale_factor)
        latents_dec = (latents_dec / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        decoded = pipeline.vae.decode(latents_dec, return_dict=False)[0]
        output_image = pipeline.image_processor.postprocess(decoded, output_type="pil")[0]

    vram = torch.cuda.max_memory_reserved() / 1e9
    return output_image, t_end - t_start, vram


def compute_metrics(img_baseline, img_test):
    """Compute SSIM and LPIPS between two PIL images."""
    if img_baseline.size != img_test.size:
        img_test = img_test.resize(img_baseline.size, Image.LANCZOS)

    a = np.array(img_baseline).astype(np.float64)
    b = np.array(img_test).astype(np.float64)
    ssim_val = float(ssim(a, b, channel_axis=2, data_range=255.0))

    # LPIPS
    def to_tensor(img):
        t = torch.from_numpy(np.array(img)).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0) * 2 - 1
        return t.cuda()

    with torch.no_grad():
        lpips_val = float(lpips_fn(to_tensor(img_baseline), to_tensor(img_test)).item())

    return ssim_val, lpips_val


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

# F2 results for comparison (conservative schedule, steps 6-20)
F2_RESULTS = {
    "f2_01_simple_object":  {"ssim": 0.987,  "lpips": 0.0114},
    "f2_02_object_modify":  {"ssim": 0.9338, "lpips": 0.093},
    "f2_05_portrait":       {"ssim": 0.9091, "lpips": 0.0484},
    "f2_04_arrangement":    {"ssim": 0.9058, "lpips": 0.0263},
    "f2_09_style_transfer": {"ssim": 0.6813, "lpips": 0.5436},
    "f2_10_fine_texture":   {"ssim": 0.3657, "lpips": 0.5132},
    "f2_11_complex_scene":  {"ssim": 0.4148, "lpips": 0.5726},
    "f2_12_nature_detail":  {"ssim": 0.7098, "lpips": 0.2976},
}

# Run experiments
print("=" * 70)
print("F4: CONDITIONING-AWARE KV REUSE EVALUATION")
print(f"Plateau region: steps {PLATEAU_START}-{PLATEAU_END}")
print(f"Modes: baseline, kv_cached, temporal_cached")
print("=" * 70)

all_results = []
cache_store = {}

for tc_name, color, size, prompt in TEST_CASES:
    print(f"\n{'─'*50}")
    print(f"  {tc_name}")
    print(f"{'─'*50}")

    ref_img = Image.new("RGB", size, color)
    inputs = prepare_inputs(ref_img, prompt)

    # Baseline
    print(f"  Running baseline...")
    img_base, t_base, vram_base = run_baseline(inputs)
    img_base.save(os.path.join(OUTPUT_DIR, f"{tc_name}_baseline.png"))
    print(f"    Time: {t_base:.2f}s, VRAM: {vram_base:.2f} GB")

    # KV-cached
    print(f"  Running kv_cached...")
    img_kv, t_kv, vram_kv = run_kv_cached(inputs, cache_store)
    img_kv.save(os.path.join(OUTPUT_DIR, f"{tc_name}_kv_cached.png"))
    ssim_kv, lpips_kv = compute_metrics(img_base, img_kv)
    speedup_kv = t_base / t_kv
    print(f"    Time: {t_kv:.2f}s ({speedup_kv:.3f}x), SSIM: {ssim_kv:.4f}, LPIPS: {lpips_kv:.4f}")

    # Temporal cached (F2-style, same plateau window for fair comparison)
    print(f"  Running temporal_cached...")
    img_tc, t_tc, vram_tc = run_temporal_cached(inputs)
    img_tc.save(os.path.join(OUTPUT_DIR, f"{tc_name}_temporal.png"))
    ssim_tc, lpips_tc = compute_metrics(img_base, img_tc)
    speedup_tc = t_base / t_tc
    print(f"    Time: {t_tc:.2f}s ({speedup_tc:.3f}x), SSIM: {ssim_tc:.4f}, LPIPS: {lpips_tc:.4f}")

    # F2 reference values
    f2_ref = F2_RESULTS.get(tc_name, {})

    result = {
        "test_case": tc_name,
        "baseline_time_s": round(t_base, 3),
        "kv_cached": {
            "time_s": round(t_kv, 3),
            "speedup": round(speedup_kv, 3),
            "ssim": round(ssim_kv, 4),
            "lpips": round(lpips_kv, 4),
            "vram_gb": round(vram_kv, 2),
        },
        "temporal_cached": {
            "time_s": round(t_tc, 3),
            "speedup": round(speedup_tc, 3),
            "ssim": round(ssim_tc, 4),
            "lpips": round(lpips_tc, 4),
            "vram_gb": round(vram_tc, 2),
        },
        "f2_reference": f2_ref,
    }
    all_results.append(result)

# ═══════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════

print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")

print(f"\n  {'Test Case':<25s} {'KV Speed':>9s} {'KV SSIM':>8s} {'KV LPIPS':>9s} "
      f"{'TC Speed':>9s} {'TC SSIM':>8s} {'TC LPIPS':>9s}")
print(f"  {'-'*25} {'-'*9} {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*9}")

for r in all_results:
    kv = r["kv_cached"]
    tc = r["temporal_cached"]
    print(f"  {r['test_case']:<25s} "
          f"{kv['speedup']:9.3f} {kv['ssim']:8.4f} {kv['lpips']:9.4f} "
          f"{tc['speedup']:9.3f} {tc['ssim']:8.4f} {tc['lpips']:9.4f}")

# Aggregates
kv_speeds = [r["kv_cached"]["speedup"] for r in all_results]
kv_ssims = [r["kv_cached"]["ssim"] for r in all_results]
kv_lpipss = [r["kv_cached"]["lpips"] for r in all_results]
tc_speeds = [r["temporal_cached"]["speedup"] for r in all_results]
tc_ssims = [r["temporal_cached"]["ssim"] for r in all_results]
tc_lpipss = [r["temporal_cached"]["lpips"] for r in all_results]

print(f"\n  {'MEAN':<25s} "
      f"{np.mean(kv_speeds):9.3f} {np.mean(kv_ssims):8.4f} {np.mean(kv_lpipss):9.4f} "
      f"{np.mean(tc_speeds):9.3f} {np.mean(tc_ssims):8.4f} {np.mean(tc_lpipss):9.4f}")

# Quality comparison
print(f"\n{'='*70}")
print("KV CACHING vs TEMPORAL CACHING (same plateau window)")
print(f"{'='*70}")

kv_better_ssim = sum(1 for r in all_results if r["kv_cached"]["ssim"] > r["temporal_cached"]["ssim"])
tc_better_ssim = sum(1 for r in all_results if r["temporal_cached"]["ssim"] > r["kv_cached"]["ssim"])

print(f"\n  KV caching wins on SSIM: {kv_better_ssim}/{len(all_results)} cases")
print(f"  Temporal caching wins on SSIM: {tc_better_ssim}/{len(all_results)} cases")
print(f"  KV mean speedup: {np.mean(kv_speeds):.3f}x")
print(f"  Temporal mean speedup: {np.mean(tc_speeds):.3f}x")
print(f"  KV mean SSIM: {np.mean(kv_ssims):.4f}")
print(f"  Temporal mean SSIM: {np.mean(tc_ssims):.4f}")
print(f"  SSIM difference (KV - TC): {np.mean(kv_ssims) - np.mean(tc_ssims):+.4f}")

# Robust vs fragile breakdown
robust_cases = all_results[:4]
fragile_cases = all_results[4:]

print(f"\n  Robust cases (n=4):")
print(f"    KV SSIM: {np.mean([r['kv_cached']['ssim'] for r in robust_cases]):.4f}")
print(f"    TC SSIM: {np.mean([r['temporal_cached']['ssim'] for r in robust_cases]):.4f}")

print(f"  Fragile cases (n=4):")
print(f"    KV SSIM: {np.mean([r['kv_cached']['ssim'] for r in fragile_cases]):.4f}")
print(f"    TC SSIM: {np.mean([r['temporal_cached']['ssim'] for r in fragile_cases]):.4f}")

# Verdict
print(f"\n{'='*70}")
print("F4 VERDICT")
print(f"{'='*70}")

speedup_diff = np.mean(kv_speeds) - np.mean(tc_speeds)
ssim_diff = np.mean(kv_ssims) - np.mean(tc_ssims)

if ssim_diff > 0.02 and speedup_diff > -0.1:
    verdict = "KV REUSE SUPERIOR"
    detail = "Better quality with comparable speedup."
elif ssim_diff > 0.005:
    verdict = "KV REUSE MARGINALLY BETTER"
    detail = "Slightly better quality, but difference may not justify complexity."
elif abs(ssim_diff) < 0.005:
    verdict = "NO MEANINGFUL DIFFERENCE"
    detail = "Quality indistinguishable. Whole-step caching is simpler and faster."
elif ssim_diff < -0.02:
    verdict = "TEMPORAL CACHING SUPERIOR"
    detail = "Whole-step caching is both faster AND better quality."
else:
    verdict = "TEMPORAL CACHING PREFERRED"
    detail = "Similar quality but temporal caching is simpler and faster."

print(f"\n  Verdict: {verdict}")
print(f"  {detail}")
print(f"  Speedup gap: {speedup_diff:+.3f}x (KV - Temporal)")
print(f"  SSIM gap:    {ssim_diff:+.4f} (KV - Temporal)")

# Save results
save_data = {
    "config": {
        "model": "FLUX.1 Kontext-dev",
        "num_steps": NUM_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "seed": SEED,
        "plateau_window": [PLATEAU_START, PLATEAU_END],
        "n_cached_steps": PLATEAU_END - PLATEAU_START,
        "n_full_steps": NUM_STEPS - (PLATEAU_END - PLATEAU_START),
    },
    "results": all_results,
    "aggregates": {
        "kv_cached": {
            "mean_speedup": round(float(np.mean(kv_speeds)), 3),
            "mean_ssim": round(float(np.mean(kv_ssims)), 4),
            "mean_lpips": round(float(np.mean(kv_lpipss)), 4),
        },
        "temporal_cached": {
            "mean_speedup": round(float(np.mean(tc_speeds)), 3),
            "mean_ssim": round(float(np.mean(tc_ssims)), 4),
            "mean_lpips": round(float(np.mean(tc_lpipss)), 4),
        },
    },
    "verdict": {
        "label": verdict,
        "detail": detail,
        "speedup_gap": round(float(speedup_diff), 3),
        "ssim_gap": round(float(ssim_diff), 4),
    },
}

json_path = os.path.join(OUTPUT_DIR, "f4_kv_cache_results.json")
with open(json_path, "w") as f:
    json.dump(save_data, f, indent=2)
print(f"\nResults saved to {json_path}")
print("\nDone.")
