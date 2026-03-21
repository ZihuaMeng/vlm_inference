#!/usr/bin/env python3
"""Milestone 7: Multi-Image Condition Redundancy Characterization

Measures how multi-image conditioning affects computation:
  1. Token counts for 1, 2, 3 input images
  2. Per-step and total wall-clock time (compute scaling)
  3. Cross-image hidden state similarity at sampled blocks
  4. Cross-image attention score analysis (region-level)
  5. Temporal/CFG redundancy interaction with image count

Design:
  - For each image count (1, 2, 3): run full instrumented forward pass
  - Capture block outputs and attention Q/K at sampled blocks/steps
  - Compare hidden states and attention patterns across image token regions
"""

import torch
import time
import json
import os
import math
import numpy as np
from PIL import Image
from collections import defaultdict
import importlib

# ─── Configuration ───
WEIGHTS = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/weights/Qwen-Image-Edit-2511"
OUTPUT_DIR = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/profiling/multi_image_conditioning"
NUM_STEPS = 40
TRUE_CFG_SCALE = 4.0
SEED = 42
SAMPLED_BLOCKS = [0, 14, 29, 44, 59]
# Steps to capture attention Q/K (early, mid-plateau, late)
ATTN_CAPTURE_STEPS = [0, 5, 15, 25, 35, 39]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Test Configurations ───
# Each: (name, image_specs, prompt, negative_prompt)
# image_specs: list of (color, size)
CONFIGS = [
    (
        "1img",
        [("red", (512, 512))],
        "A red object sitting on a white table.",
        " ",
    ),
    (
        "2img",
        [("red", (512, 512)), ("blue", (512, 512))],
        "A red object and a blue object sitting side by side on a white table.",
        " ",
    ),
    (
        "3img",
        [("red", (512, 512)), ("green", (512, 512)), ("blue", (512, 512))],
        "A red, green, and blue object arranged on a white table.",
        " ",
    ),
]


def make_images(specs):
    return [Image.new("RGB", size, color) for color, size in specs]


# ─── Similarity Metrics ───
def cosine_sim(a, b):
    """Mean cosine similarity. a, b: [seq, dim] or [1, seq, dim]."""
    a = a.reshape(-1, a.shape[-1]).float()
    b = b.reshape(-1, b.shape[-1]).float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return cos.mean().item()


def relative_l2(a, b):
    a, b = a.float(), b.float()
    return (torch.norm(a - b) / max(torch.norm(a).item(), 1e-8)).item()


# ─── Block Output Capture ───
class BlockCapture:
    """Captures image-stream hidden states from sampled transformer blocks."""
    def __init__(self):
        self.hooks = []
        self.active = False
        self.captured = {}

    def register(self, block_idx, module):
        def hook(mod, inp, out, *, _idx=block_idx):
            if not self.active:
                return
            self.captured[_idx] = out[1].detach()  # image stream
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


# ─── Attention Q/K Capture ───
class AttnCapture:
    """Captures Q and K tensors from the attention processor at sampled blocks.

    Hooks into the attention module's forward to capture Q/K after projection
    and QK-norm but before RoPE and concatenation.
    We capture the image-stream Q and K, which contain noise + condition tokens.
    """
    def __init__(self):
        self.hooks = []
        self.active = False
        self.captured = {}

    def register(self, block_idx, block_module):
        """Hook into the attn sub-module of the block."""
        attn_module = block_module.attn

        def hook(mod, args, kwargs, output, *, _idx=block_idx):
            if not self.active:
                return
            # We need Q and K for the image stream.
            # Instead of modifying the processor, we compute Q/K from inputs.
            # The processor receives hidden_states (image) and encoder_hidden_states (text).
            # We can compute Q/K the same way the processor does.
            hidden_states = args[0] if len(args) > 0 else kwargs.get('hidden_states')
            if hidden_states is None:
                return
            # Compute image Q and K using the attn module's projections
            with torch.no_grad():
                img_q = mod.to_q(hidden_states)
                img_k = mod.to_k(hidden_states)
                # Reshape to [batch, seq, heads, head_dim]
                img_q = img_q.unflatten(-1, (mod.heads, -1))
                img_k = img_k.unflatten(-1, (mod.heads, -1))
                # Apply QK norm
                if mod.norm_q is not None:
                    img_q = mod.norm_q(img_q)
                if mod.norm_k is not None:
                    img_k = mod.norm_k(img_k)
            self.captured[_idx] = {
                'q': img_q.detach(),  # [1, seq, heads, head_dim]
                'k': img_k.detach(),
            }

        h = attn_module.register_forward_hook(hook, with_kwargs=True)
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


# ─── Region-Level Attention Analysis ───
def compute_region_attention(q, k, regions, n_sample=64):
    """Compute attention score matrix between regions.

    Args:
        q: [1, total_seq, heads, head_dim]
        k: [1, total_seq, heads, head_dim]
        regions: dict mapping name -> (start, end) token indices
        n_sample: number of query tokens to sample per region

    Returns:
        dict: {query_region: {key_region: mean_attention_fraction}}
    """
    q = q.squeeze(0).float()  # [seq, heads, head_dim]
    k = k.squeeze(0).float()
    total_seq = k.shape[0]
    n_heads = q.shape[1]
    head_dim = q.shape[2]
    scale = 1.0 / math.sqrt(head_dim)

    result = {}
    for qr_name, (qr_start, qr_end) in regions.items():
        qr_len = qr_end - qr_start
        if qr_len == 0:
            continue

        # Sample query indices
        if qr_len <= n_sample:
            q_indices = list(range(qr_start, qr_end))
        else:
            rng = np.random.RandomState(42)
            q_indices = sorted(rng.choice(range(qr_start, qr_end), n_sample, replace=False))

        # Get sampled Q: [n_sample, heads, head_dim]
        q_sampled = q[q_indices]

        # Compute attention scores: [n_sample, heads, total_seq]
        # q_sampled: [n_sample, heads, head_dim] -> [heads, n_sample, head_dim]
        # k: [total_seq, heads, head_dim] -> [heads, head_dim, total_seq]
        q_perm = q_sampled.permute(1, 0, 2)  # [heads, n_sample, head_dim]
        k_perm = k.permute(1, 2, 0)  # [heads, head_dim, total_seq]
        scores = torch.bmm(q_perm, k_perm) * scale  # [heads, n_sample, total_seq]
        attn_weights = torch.softmax(scores, dim=-1)  # [heads, n_sample, total_seq]

        # Average across heads and sampled queries
        avg_attn = attn_weights.mean(dim=(0, 1))  # [total_seq]

        # Aggregate by region
        region_scores = {}
        for kr_name, (kr_start, kr_end) in regions.items():
            if kr_end - kr_start == 0:
                continue
            region_scores[kr_name] = avg_attn[kr_start:kr_end].sum().item()

        result[qr_name] = region_scores

    return result


# ─── Load Pipeline ───
print("Loading pipeline...")
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
pipeline.to("cuda")
vram_after_load = torch.cuda.memory_allocated() / 1e9
print(f"Pipeline loaded. VRAM: {vram_after_load:.2f} GB")

pm = importlib.import_module("diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus")
calculate_dimensions = pm.calculate_dimensions
CONDITION_IMAGE_SIZE = pm.CONDITION_IMAGE_SIZE
VAE_IMAGE_SIZE = pm.VAE_IMAGE_SIZE
calculate_shift = pm.calculate_shift
retrieve_timesteps = pm.retrieve_timesteps

# ─── Warmup ───
print("\nRunning warmup...")
img_warmup = [Image.new("RGB", (512, 512), "red"), Image.new("RGB", (512, 512), "blue")]
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    _ = pipeline(
        image=img_warmup, prompt="A red and blue object on a table.",
        num_inference_steps=NUM_STEPS, true_cfg_scale=TRUE_CFG_SCALE,
        negative_prompt=" ", guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(SEED),
    )
torch.cuda.synchronize()
print(f"Warmup: {time.perf_counter() - t0:.1f}s")

# ─── Set Up Hooks ───
block_capture = BlockCapture()
attn_capture = AttnCapture()
for bi in SAMPLED_BLOCKS:
    block_capture.register(bi, pipeline.transformer.transformer_blocks[bi])
    attn_capture.register(bi, pipeline.transformer.transformer_blocks[bi])

# ─── Main Loop: For Each Image Count ───
all_results = {}

for config_name, img_specs, prompt, neg_prompt in CONFIGS:
    n_images = len(img_specs)
    print(f"\n{'='*60}")
    print(f"CONFIG: {config_name} ({n_images} image(s))")
    print(f"{'='*60}")

    images = make_images(img_specs)

    # ── Pipeline setup ──
    pipeline._guidance_scale = 1.0
    pipeline._attention_kwargs = {}
    pipeline._current_timestep = None
    pipeline._interrupt = False

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        # ── Preprocess ──
        image_size = images[-1].size
        calc_w, calc_h = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height, width = calc_h, calc_w
        mult = pipeline.vae_scale_factor * 2
        width = width // mult * mult
        height = height // mult * mult

        condition_images = []
        vae_images = []
        vae_image_sizes = []
        for img in images:
            iw, ih = img.size
            cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, iw / ih)
            vw, vh = calculate_dimensions(VAE_IMAGE_SIZE, iw / ih)
            vae_image_sizes.append((vw, vh))
            condition_images.append(pipeline.image_processor.resize(img, ch, cw))
            vae_images.append(pipeline.image_processor.preprocess(img, vh, vw).unsqueeze(2))

        # ── Text Encoding ──
        t_enc_start = time.perf_counter()
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            image=condition_images, prompt=prompt,
            prompt_embeds=None, prompt_embeds_mask=None,
            device="cuda", num_images_per_prompt=1, max_sequence_length=512,
        )
        neg_prompt_embeds, neg_prompt_embeds_mask = pipeline.encode_prompt(
            image=condition_images, prompt=neg_prompt,
            prompt_embeds=None, prompt_embeds_mask=None,
            device="cuda", num_images_per_prompt=1, max_sequence_length=512,
        )
        torch.cuda.synchronize()
        t_enc = time.perf_counter() - t_enc_start

        # ── VAE Encoding ──
        num_ch = pipeline.transformer.config.in_channels // 4
        generator = torch.Generator("cuda").manual_seed(SEED)
        latents, image_latents = pipeline.prepare_latents(
            vae_images, 1, num_ch, height, width,
            prompt_embeds.dtype, "cuda", generator, None,
        )

        # ── Token counts ──
        noise_tokens = latents.shape[1]
        cond_tokens = image_latents.shape[1] if image_latents is not None else 0
        tokens_per_image = cond_tokens // n_images
        text_tokens = prompt_embeds.shape[1]
        neg_text_tokens = neg_prompt_embeds.shape[1]
        total_img_seq = noise_tokens + cond_tokens  # hidden_states length

        # Token region boundaries (within hidden_states, i.e., image stream)
        token_regions = {"noise": (0, noise_tokens)}
        offset = noise_tokens
        for i in range(n_images):
            token_regions[f"img{i+1}"] = (offset, offset + tokens_per_image)
            offset += tokens_per_image

        print(f"  Noise tokens: {noise_tokens}")
        print(f"  Condition tokens: {cond_tokens} ({tokens_per_image} per image × {n_images})")
        print(f"  Total image-stream tokens: {total_img_seq}")
        print(f"  Text tokens (cond): {text_tokens}")
        print(f"  Text tokens (neg): {neg_text_tokens}")
        print(f"  Text encoding time: {t_enc:.2f}s")

        # ── img_shapes ──
        img_shapes = [
            [
                (1, height // pipeline.vae_scale_factor // 2, width // pipeline.vae_scale_factor // 2),
                *[
                    (1, vh // pipeline.vae_scale_factor // 2, vw // pipeline.vae_scale_factor // 2)
                    for vw, vh in vae_image_sizes
                ],
            ]
        ]

        # ── Timesteps ──
        sigmas = np.linspace(1.0, 1 / NUM_STEPS, NUM_STEPS)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(image_seq_len,
            pipeline.scheduler.config.get("base_image_seq_len", 256),
            pipeline.scheduler.config.get("max_image_seq_len", 4096),
            pipeline.scheduler.config.get("base_shift", 0.5),
            pipeline.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            pipeline.scheduler, NUM_STEPS, "cuda", sigmas=sigmas, mu=mu,
        )
        pipeline.scheduler.set_begin_index(0)

        # Model is not guidance-distilled, so guidance must be None
        guidance = None

        # ── Results storage ──
        step_times_cond = []
        step_times_uncond = []
        temporal_results = []        # step-to-step similarity
        cfg_results = []             # cond-vs-uncond similarity
        cross_image_results = []     # per-step cross-image hidden state similarity
        attn_region_results = []     # per-step region-level attention matrix

        prev_cond_output = None
        prev_cond_blocks = None

        # ── Denoising Loop ──
        for step_idx, t in enumerate(timesteps):
            latent_input = latents
            if image_latents is not None:
                latent_input = torch.cat([latents, image_latents], dim=1)

            ts = t.expand(latents.shape[0]).to(latents.dtype)
            capture_attn = (step_idx in ATTN_CAPTURE_STEPS)

            # ── Conditional pass ──
            block_capture.start()
            if capture_attn:
                attn_capture.start()

            torch.cuda.synchronize()
            t_cond_start = time.perf_counter()

            with pipeline.transformer.cache_context("cond"):
                cond_output = pipeline.transformer(
                    hidden_states=latent_input,
                    timestep=ts / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    attention_kwargs={},
                    return_dict=False,
                )[0]

            torch.cuda.synchronize()
            t_cond = time.perf_counter() - t_cond_start
            step_times_cond.append(t_cond)

            block_capture.stop()
            attn_capture.stop()

            cond_noise_pred = cond_output[:, :noise_tokens].detach()
            cond_blocks = block_capture.get_and_clear()
            cond_attn = attn_capture.get_and_clear() if capture_attn else None

            # ── Cross-image hidden state analysis ──
            if n_images >= 2:
                step_cross = {"step": step_idx}
                for bi in SAMPLED_BLOCKS:
                    if bi not in cond_blocks:
                        continue
                    hs = cond_blocks[bi]  # [1, total_img_seq, dim]
                    # Compare each pair of image regions
                    for i in range(n_images):
                        for j in range(i + 1, n_images):
                            ri_start, ri_end = token_regions[f"img{i+1}"]
                            rj_start, rj_end = token_regions[f"img{j+1}"]
                            hi = hs[:, ri_start:ri_end, :]
                            hj = hs[:, rj_start:rj_end, :]
                            cos = cosine_sim(hi, hj)
                            step_cross[f"b{bi}_img{i+1}_img{j+1}_cos"] = cos
                    # Also: noise vs each image
                    noise_hs = hs[:, :noise_tokens, :]
                    for i in range(n_images):
                        ri_start, ri_end = token_regions[f"img{i+1}"]
                        img_hs = hs[:, ri_start:ri_end, :]
                        cos = cosine_sim(noise_hs, img_hs)
                        step_cross[f"b{bi}_noise_img{i+1}_cos"] = cos
                cross_image_results.append(step_cross)

            # ── Attention region analysis ──
            if capture_attn and cond_attn:
                step_attn = {"step": step_idx}
                for bi in SAMPLED_BLOCKS:
                    if bi not in cond_attn:
                        continue
                    qk = cond_attn[bi]
                    region_attn = compute_region_attention(
                        qk['q'], qk['k'], token_regions, n_sample=64,
                    )
                    step_attn[f"b{bi}"] = region_attn
                attn_region_results.append(step_attn)
                # Free attention tensors
                del cond_attn
                torch.cuda.empty_cache()

            # ── Temporal similarity ──
            if prev_cond_output is not None:
                step_temporal = {
                    "step": step_idx,
                    "transformer_cos": cosine_sim(cond_noise_pred, prev_cond_output),
                    "transformer_rl2": relative_l2(cond_noise_pred, prev_cond_output),
                }
                if prev_cond_blocks is not None:
                    for bi in SAMPLED_BLOCKS:
                        if bi in cond_blocks and bi in prev_cond_blocks:
                            step_temporal[f"b{bi}_cos"] = cosine_sim(
                                cond_blocks[bi][:, :noise_tokens, :],
                                prev_cond_blocks[bi][:, :noise_tokens, :],
                            )
                temporal_results.append(step_temporal)

            prev_cond_output = cond_noise_pred
            prev_cond_blocks = {bi: cond_blocks[bi] for bi in SAMPLED_BLOCKS if bi in cond_blocks}

            # ── Unconditional pass ──
            block_capture.start()
            torch.cuda.synchronize()
            t_uncond_start = time.perf_counter()

            with pipeline.transformer.cache_context("uncond"):
                uncond_output = pipeline.transformer(
                    hidden_states=latent_input,
                    timestep=ts / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=neg_prompt_embeds_mask,
                    encoder_hidden_states=neg_prompt_embeds,
                    img_shapes=img_shapes,
                    attention_kwargs={},
                    return_dict=False,
                )[0]

            torch.cuda.synchronize()
            t_uncond = time.perf_counter() - t_uncond_start
            step_times_uncond.append(t_uncond)

            block_capture.stop()
            uncond_noise_pred = uncond_output[:, :noise_tokens].detach()
            uncond_blocks = block_capture.get_and_clear()

            # ── CFG divergence ──
            step_cfg = {
                "step": step_idx,
                "transformer_cos": cosine_sim(cond_noise_pred, uncond_noise_pred),
                "transformer_rl2": relative_l2(cond_noise_pred, uncond_noise_pred),
            }
            for bi in SAMPLED_BLOCKS:
                if bi in cond_blocks and bi in uncond_blocks:
                    step_cfg[f"b{bi}_cos"] = cosine_sim(
                        cond_blocks[bi][:, :noise_tokens, :],
                        uncond_blocks[bi][:, :noise_tokens, :],
                    )
            cfg_results.append(step_cfg)

            # ── CFG combination ──
            comb = uncond_noise_pred + TRUE_CFG_SCALE * (cond_noise_pred - uncond_noise_pred)
            cond_norm = torch.norm(cond_noise_pred, dim=-1, keepdim=True)
            comb_norm = torch.norm(comb, dim=-1, keepdim=True)
            noise_pred = comb * (cond_norm / comb_norm)

            # ── Scheduler step ──
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if step_idx % 10 == 0 or step_idx == NUM_STEPS - 1:
                print(f"  Step {step_idx:2d}: cond {t_cond*1000:.0f}ms, uncond {t_uncond*1000:.0f}ms")

        # ── Summary ──
        peak_vram = torch.cuda.max_memory_reserved() / 1e9
        total_cond = sum(step_times_cond)
        total_uncond = sum(step_times_uncond)
        total_time = total_cond + total_uncond
        mean_step = total_time / NUM_STEPS

        print(f"\n  Total: {total_time:.2f}s ({total_cond:.2f}s cond + {total_uncond:.2f}s uncond)")
        print(f"  Mean step: {mean_step*1000:.1f}ms")
        print(f"  Peak VRAM: {peak_vram:.2f} GB")

        # ── Store results ──
        config_result = {
            "config_name": config_name,
            "n_images": n_images,
            "tokens": {
                "noise": noise_tokens,
                "per_image_condition": tokens_per_image,
                "total_condition": cond_tokens,
                "total_image_stream": total_img_seq,
                "text_cond": text_tokens,
                "text_neg": neg_text_tokens,
            },
            "timing": {
                "text_encoding_s": t_enc,
                "total_cond_s": total_cond,
                "total_uncond_s": total_uncond,
                "total_denoising_s": total_time,
                "mean_step_ms": mean_step * 1000,
                "per_step_cond_ms": [t * 1000 for t in step_times_cond],
                "per_step_uncond_ms": [t * 1000 for t in step_times_uncond],
            },
            "peak_vram_gb": peak_vram,
            "temporal_redundancy": temporal_results,
            "cfg_redundancy": cfg_results,
            "cross_image_similarity": cross_image_results,
            "attention_regions": attn_region_results,
        }
        all_results[config_name] = config_result

# ─── Save Results ───
output_path = os.path.join(OUTPUT_DIR, "m7_results.json")
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {output_path}")

# ─── Print Summary Table ───
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Token counts
print("\n--- Token Counts ---")
print(f"{'Config':>10} | {'Noise':>8} | {'Per-Img':>8} | {'Total Cond':>10} | {'Total Img':>10} | {'Text':>6}")
for cfg_name in ["1img", "2img", "3img"]:
    r = all_results[cfg_name]["tokens"]
    print(f"{cfg_name:>10} | {r['noise']:>8} | {r['per_image_condition']:>8} | {r['total_condition']:>10} | {r['total_image_stream']:>10} | {r['text_cond']:>6}")

# Timing
print("\n--- Compute Scaling ---")
print(f"{'Config':>10} | {'Total (s)':>10} | {'Mean Step (ms)':>14} | {'Cond (s)':>10} | {'Uncond (s)':>10} | {'VRAM (GB)':>10}")
for cfg_name in ["1img", "2img", "3img"]:
    r = all_results[cfg_name]
    t = r["timing"]
    print(f"{cfg_name:>10} | {t['total_denoising_s']:>10.2f} | {t['mean_step_ms']:>14.1f} | {t['total_cond_s']:>10.2f} | {t['total_uncond_s']:>10.2f} | {r['peak_vram_gb']:>10.2f}")

# Scaling ratios
t1 = all_results["1img"]["timing"]["total_denoising_s"]
print(f"\n--- Scaling Relative to 1-Image ---")
for cfg_name in ["1img", "2img", "3img"]:
    tn = all_results[cfg_name]["timing"]["total_denoising_s"]
    print(f"  {cfg_name}: {tn/t1:.3f}x")

# Cross-image similarity (2img and 3img)
print("\n--- Cross-Image Hidden State Similarity (step 20, mid-plateau) ---")
for cfg_name in ["2img", "3img"]:
    cross = all_results[cfg_name]["cross_image_similarity"]
    # Find step closest to 20
    mid_step = min(cross, key=lambda x: abs(x["step"] - 20))
    print(f"\n  {cfg_name} (step {mid_step['step']}):")
    for key, val in sorted(mid_step.items()):
        if key == "step":
            continue
        print(f"    {key}: {val:.6f}")

# Temporal redundancy comparison
print("\n--- Temporal Redundancy Comparison (transformer output cos) ---")
print(f"{'Step':>6}", end="")
for cfg_name in ["1img", "2img", "3img"]:
    print(f" | {cfg_name:>10}", end="")
print()
for step_idx in [1, 5, 10, 15, 20, 25, 30, 35, 39]:
    print(f"{step_idx:>6}", end="")
    for cfg_name in ["1img", "2img", "3img"]:
        temporal = all_results[cfg_name]["temporal_redundancy"]
        match = [t for t in temporal if t["step"] == step_idx]
        if match:
            print(f" | {match[0]['transformer_cos']:>10.6f}", end="")
        else:
            print(f" | {'N/A':>10}", end="")
    print()

# CFG redundancy comparison
print("\n--- CFG Redundancy Comparison (transformer output cos) ---")
print(f"{'Step':>6}", end="")
for cfg_name in ["1img", "2img", "3img"]:
    print(f" | {cfg_name:>10}", end="")
print()
for step_idx in [0, 1, 2, 5, 10, 20, 30, 38, 39]:
    print(f"{step_idx:>6}", end="")
    for cfg_name in ["1img", "2img", "3img"]:
        cfg_data = all_results[cfg_name]["cfg_redundancy"]
        match = [c for c in cfg_data if c["step"] == step_idx]
        if match:
            print(f" | {match[0]['transformer_cos']:>10.6f}", end="")
        else:
            print(f" | {'N/A':>10}", end="")
    print()

# Attention region analysis
print("\n--- Attention Region Distribution (block 29, sampled steps) ---")
for cfg_name in ["2img", "3img"]:
    attn_data = all_results[cfg_name]["attention_regions"]
    if not attn_data:
        continue
    print(f"\n  {cfg_name}:")
    for step_entry in attn_data:
        step = step_entry["step"]
        if f"b29" in step_entry:
            region_matrix = step_entry["b29"]
            print(f"    Step {step}:")
            for qr, scores in sorted(region_matrix.items()):
                scores_str = ", ".join(f"{kr}={v:.4f}" for kr, v in sorted(scores.items()))
                print(f"      {qr} attends to: {scores_str}")

print("\n" + "=" * 80)
print("M7 instrumentation complete.")
print("=" * 80)

# ─── Cleanup ───
block_capture.remove_hooks()
attn_capture.remove_hooks()
