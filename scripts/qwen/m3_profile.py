#!/usr/bin/env python3
"""Milestone 3: Structured Baseline Profiling

Three-level profiling of QwenImageEditPlusPipeline on B200.
  Level 1: Pipeline stage breakdown (text enc, VAE enc, denoising, VAE dec)
  Level 2: Per-step denoising dynamics (all 40 steps)
  Level 3: Module-level breakdown within sampled denoising steps

Uses torch.cuda.synchronize() + time.perf_counter() for wall-clock timing,
and torch.cuda.Event for fine-grained module-level timing within transformer.
"""

import torch
import time
import json
import csv
import os
import contextlib
import numpy as np
from PIL import Image
from collections import defaultdict

# ─── Configuration ───
WEIGHTS = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/weights/Qwen-Image-Edit-2511"
OUTPUT_DIR = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/profiling"
NUM_STEPS = 40
TRUE_CFG_SCALE = 4.0
SEED = 42
SAMPLED_STEPS = [0, 19, 39]  # early, mid, late
SAMPLED_BLOCKS = [0, 29, 59]  # early, mid, late within transformer

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Module-Level Profiler (Level 3) ───
class ModuleProfiler:
    """Records cuda event pairs on registered modules when active."""

    def __init__(self):
        self.records = defaultdict(list)  # name -> [(start_event, end_event), ...]
        self.hooks = []
        self.active = False
        self._pending = {}

    def register(self, name, module):
        def pre_hook(mod, inp, *, _name=name):
            if not self.active:
                return
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._pending[_name] = ev

        def post_hook(mod, inp, out, *, _name=name):
            if not self.active or _name not in self._pending:
                return
            start_ev = self._pending.pop(_name)
            end_ev = torch.cuda.Event(enable_timing=True)
            end_ev.record()
            self.records[_name].append((start_ev, end_ev))

        h1 = module.register_forward_pre_hook(pre_hook)
        h2 = module.register_forward_hook(post_hook)
        self.hooks.extend([h1, h2])

    def get_times_ms(self):
        """Synchronize and compute elapsed times from event pairs."""
        torch.cuda.synchronize()
        result = {}
        for name, pairs in self.records.items():
            result[name] = [s.elapsed_time(e) for s, e in pairs]
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
print(f"Pipeline loaded. VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ─── Prepare Inputs (same as M1/M2) ───
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

# ─── Set up Level 3 profiler ───
profiler = ModuleProfiler()

# Register on transformer top-level modules
for name in ["img_in", "txt_norm", "txt_in", "time_text_embed", "pos_embed", "norm_out", "proj_out"]:
    profiler.register(name, getattr(pipeline.transformer, name))

# Register on each transformer block + sub-modules for sampled blocks
for i, block in enumerate(pipeline.transformer.transformer_blocks):
    profiler.register(f"block_{i}", block)
    if i in SAMPLED_BLOCKS:
        profiler.register(f"block_{i}/attn", block.attn)
        profiler.register(f"block_{i}/img_mlp", block.img_mlp)
        profiler.register(f"block_{i}/txt_mlp", block.txt_mlp)

# ─── Import pipeline helpers ───
import importlib
pm = importlib.import_module("diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus")
calculate_dimensions = pm.calculate_dimensions
CONDITION_IMAGE_SIZE = pm.CONDITION_IMAGE_SIZE
VAE_IMAGE_SIZE = pm.VAE_IMAGE_SIZE
calculate_shift = pm.calculate_shift
retrieve_timesteps = pm.retrieve_timesteps

# ─── Profiled Pass ───
print("\n" + "=" * 60)
print("Starting profiled pass...")
print("=" * 60)

# Set pipeline internal state (normally done in __call__)
pipeline._guidance_scale = 1.0
pipeline._attention_kwargs = {}
pipeline._current_timestep = None
pipeline._interrupt = False

torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    # ── Stage 1: Image Preprocessing (CPU) ──
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    image = [img1, img2]
    image_size = image[-1].size
    calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
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

    torch.cuda.synchronize()
    t_preprocess = time.perf_counter()

    # ── Stage 2: Text Encoding ──
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

    torch.cuda.synchronize()
    t_text_enc = time.perf_counter()

    # ── Stage 3: VAE Encoding / Prepare Latents ──
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    generator = torch.Generator("cuda").manual_seed(SEED)
    latents, image_latents = pipeline.prepare_latents(
        vae_images, 1, num_channels_latents, height, width,
        prompt_embeds.dtype, "cuda", generator, None,
    )
    img_shapes = [[
        (1, height // pipeline.vae_scale_factor // 2, width // pipeline.vae_scale_factor // 2),
        *[(1, vh // pipeline.vae_scale_factor // 2, vw // pipeline.vae_scale_factor // 2)
          for vw, vh in vae_image_sizes],
    ]]

    torch.cuda.synchronize()
    t_vae_enc = time.perf_counter()

    # ── Stage 4: Scheduler Setup ──
    sigmas = np.linspace(1.0, 1 / NUM_STEPS, NUM_STEPS)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(pipeline.scheduler, NUM_STEPS, "cuda", sigmas=sigmas, mu=mu)
    pipeline.scheduler.set_begin_index(0)

    # Handle guidance (guidance_embeds is False for this model → guidance=None)
    guidance = None
    if pipeline.transformer.config.guidance_embeds:
        guidance = torch.full([1], 1.0, device="cuda", dtype=torch.float32).expand(latents.shape[0])

    torch.cuda.synchronize()
    t_sched = time.perf_counter()

    # ── Stage 5: Denoising Loop ──
    step_details = []
    has_cache_ctx = hasattr(pipeline.transformer, "cache_context")

    for i, t in enumerate(timesteps):
        torch.cuda.synchronize()
        step_start = time.perf_counter()

        latent_model_input = torch.cat([latents, image_latents], dim=1)
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        # Activate Level 3 profiler for sampled steps (conditional pass only)
        if i in SAMPLED_STEPS:
            profiler.active = True

        # ── Conditional transformer pass ──
        torch.cuda.synchronize()
        cond_start = time.perf_counter()
        ctx = pipeline.transformer.cache_context("cond") if has_cache_ctx else contextlib.nullcontext()
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
        noise_pred = noise_pred[:, :latents.size(1)]
        torch.cuda.synchronize()
        cond_end = time.perf_counter()

        # Deactivate profiler (only profile conditional pass for Level 3)
        profiler.active = False

        # ── Unconditional transformer pass (CFG) ──
        torch.cuda.synchronize()
        uncond_start = time.perf_counter()
        ctx = pipeline.transformer.cache_context("uncond") if has_cache_ctx else contextlib.nullcontext()
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
        neg_noise_pred = neg_noise_pred[:, :latents.size(1)]
        torch.cuda.synchronize()
        uncond_end = time.perf_counter()

        # ── CFG combination ──
        comb_pred = neg_noise_pred + TRUE_CFG_SCALE * (noise_pred - neg_noise_pred)
        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        noise_pred = comb_pred * (cond_norm / noise_norm)

        # ── Scheduler step ──
        torch.cuda.synchronize()
        sched_start = time.perf_counter()
        latents_dtype = latents.dtype
        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)
        torch.cuda.synchronize()
        sched_end = time.perf_counter()

        step_end = time.perf_counter()

        step_details.append({
            "step": i,
            "cond_ms": (cond_end - cond_start) * 1000,
            "uncond_ms": (uncond_end - uncond_start) * 1000,
            "sched_ms": (sched_end - sched_start) * 1000,
            "total_ms": (step_end - step_start) * 1000,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Step {i:2d}/{NUM_STEPS}: {step_details[-1]['total_ms']:.1f}ms")

    torch.cuda.synchronize()
    t_denoise = time.perf_counter()

    # ── Stage 6: VAE Decoding ──
    latents_dec = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
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
    output_image = pipeline.image_processor.postprocess(decoded, output_type="pil")

    torch.cuda.synchronize()
    t_vae_dec = time.perf_counter()

t_total = t_vae_dec - t0
peak_vram_gb = torch.cuda.max_memory_reserved() / 1e9

# Save output image for verification
output_image[0].save(os.path.join(OUTPUT_DIR, "m3_profiled_output.png"))

# ─── Collect Level 3 data ───
level3_raw = profiler.get_times_ms()
profiler.remove_hooks()

# ═══════════════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════════════

# ─── Level 1: Pipeline Stage Breakdown ───
level1 = {
    "image_preprocess": t_preprocess - t0,
    "text_encoding": t_text_enc - t_preprocess,
    "vae_encoding": t_vae_enc - t_text_enc,
    "scheduler_setup": t_sched - t_vae_enc,
    "denoising_loop": t_denoise - t_sched,
    "vae_decoding": t_vae_dec - t_denoise,
}

print("\n" + "=" * 70)
print("LEVEL 1: PIPELINE STAGE BREAKDOWN")
print("=" * 70)
print(f"  {'Stage':<25s} {'Time (s)':>10s} {'Pct':>8s}")
print(f"  {'-'*25} {'-'*10} {'-'*8}")
for name, val in level1.items():
    pct = val / t_total * 100
    print(f"  {name:<25s} {val:10.3f} {pct:7.1f}%")
print(f"  {'-'*25} {'-'*10} {'-'*8}")
print(f"  {'TOTAL':<25s} {t_total:10.3f}")
print(f"  Peak VRAM: {peak_vram_gb:.2f} GB")
print(f"  Baseline (M2): 51.2s | Overhead: {(t_total - 51.2)/51.2*100:+.1f}%")

# ─── Level 2: Per-Step Denoising Timing ───
step_totals = [s["total_ms"] for s in step_details]
cond_times = [s["cond_ms"] for s in step_details]
uncond_times = [s["uncond_ms"] for s in step_details]
sched_times = [s["sched_ms"] for s in step_details]

print("\n" + "=" * 70)
print("LEVEL 2: PER-STEP DENOISING TIMING")
print("=" * 70)
print(f"  Step total:   mean={np.mean(step_totals):.1f}ms  std={np.std(step_totals):.1f}ms  "
      f"min={np.min(step_totals):.1f}ms  max={np.max(step_totals):.1f}ms")
print(f"  Cond pass:    mean={np.mean(cond_times):.1f}ms  std={np.std(cond_times):.1f}ms")
print(f"  Uncond pass:  mean={np.mean(uncond_times):.1f}ms  std={np.std(uncond_times):.1f}ms")
print(f"  Scheduler:    mean={np.mean(sched_times):.1f}ms  std={np.std(sched_times):.1f}ms")
print(f"\n  {'Step':>6s} {'Total':>9s} {'Cond':>9s} {'Uncond':>9s} {'Sched':>9s}")
print(f"  {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
for s in step_details:
    print(f"  {s['step']:6d} {s['total_ms']:9.1f} {s['cond_ms']:9.1f} {s['uncond_ms']:9.1f} {s['sched_ms']:9.1f}")

# ─── Level 3: Module-Level Breakdown ───
print("\n" + "=" * 70)
print("LEVEL 3: MODULE-LEVEL BREAKDOWN (sampled steps, conditional pass only)")
print("=" * 70)

# Organize Level 3 data by sampled step
# Each module has len(SAMPLED_STEPS) entries: one per sampled step
for step_idx, step_num in enumerate(SAMPLED_STEPS):
    print(f"\n  --- Step {step_num} (conditional pass) ---")

    # Top-level modules
    input_proj_ms = 0
    for name in ["img_in", "txt_norm", "txt_in"]:
        if name in level3_raw and step_idx < len(level3_raw[name]):
            input_proj_ms += level3_raw[name][step_idx]

    embed_ms = 0
    for name in ["time_text_embed", "pos_embed"]:
        if name in level3_raw and step_idx < len(level3_raw[name]):
            embed_ms += level3_raw[name][step_idx]

    output_ms = 0
    for name in ["norm_out", "proj_out"]:
        if name in level3_raw and step_idx < len(level3_raw[name]):
            output_ms += level3_raw[name][step_idx]

    # All 60 blocks
    block_times = []
    for bi in range(60):
        key = f"block_{bi}"
        if key in level3_raw and step_idx < len(level3_raw[key]):
            block_times.append(level3_raw[key][step_idx])
    blocks_total_ms = sum(block_times)

    transformer_total = input_proj_ms + embed_ms + blocks_total_ms + output_ms

    print(f"    {'Component':<30s} {'Time (ms)':>10s} {'Pct':>8s}")
    print(f"    {'-'*30} {'-'*10} {'-'*8}")
    if transformer_total > 0:
        print(f"    {'Input projections':<30s} {input_proj_ms:10.2f} {input_proj_ms/transformer_total*100:7.1f}%")
        print(f"    {'Embeddings (time+rope)':<30s} {embed_ms:10.2f} {embed_ms/transformer_total*100:7.1f}%")
        print(f"    {'Transformer blocks (60)':<30s} {blocks_total_ms:10.2f} {blocks_total_ms/transformer_total*100:7.1f}%")
        print(f"    {'Output (norm+proj)':<30s} {output_ms:10.2f} {output_ms/transformer_total*100:7.1f}%")
        print(f"    {'-'*30} {'-'*10}")
        print(f"    {'Transformer forward total':<30s} {transformer_total:10.2f}")

    # Per-block stats
    if block_times:
        print(f"\n    Block timing (60 blocks): mean={np.mean(block_times):.2f}ms  "
              f"std={np.std(block_times):.2f}ms  min={np.min(block_times):.2f}ms  max={np.max(block_times):.2f}ms")

    # Sampled block internals: attention vs MLP
    print(f"\n    {'Block':<12s} {'Attention':>10s} {'ImgMLP':>10s} {'TxtMLP':>10s} {'Other':>10s} {'Total':>10s}")
    print(f"    {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for bi in SAMPLED_BLOCKS:
        bkey = f"block_{bi}"
        attn_key = f"block_{bi}/attn"
        img_mlp_key = f"block_{bi}/img_mlp"
        txt_mlp_key = f"block_{bi}/txt_mlp"

        bt = level3_raw.get(bkey, [0])[step_idx] if bkey in level3_raw and step_idx < len(level3_raw.get(bkey, [])) else 0
        at = level3_raw.get(attn_key, [0])[step_idx] if attn_key in level3_raw and step_idx < len(level3_raw.get(attn_key, [])) else 0
        im = level3_raw.get(img_mlp_key, [0])[step_idx] if img_mlp_key in level3_raw and step_idx < len(level3_raw.get(img_mlp_key, [])) else 0
        tm = level3_raw.get(txt_mlp_key, [0])[step_idx] if txt_mlp_key in level3_raw and step_idx < len(level3_raw.get(txt_mlp_key, [])) else 0
        other = bt - at - im - tm

        print(f"    Block {bi:<4d} {at:10.2f} {im:10.2f} {tm:10.2f} {other:10.2f} {bt:10.2f}")

    # Aggregate attention vs MLP across all sampled blocks for this step
    if SAMPLED_BLOCKS:
        total_attn = sum(
            level3_raw.get(f"block_{bi}/attn", [0])[step_idx]
            for bi in SAMPLED_BLOCKS
            if f"block_{bi}/attn" in level3_raw and step_idx < len(level3_raw.get(f"block_{bi}/attn", []))
        )
        total_img_mlp = sum(
            level3_raw.get(f"block_{bi}/img_mlp", [0])[step_idx]
            for bi in SAMPLED_BLOCKS
            if f"block_{bi}/img_mlp" in level3_raw and step_idx < len(level3_raw.get(f"block_{bi}/img_mlp", []))
        )
        total_txt_mlp = sum(
            level3_raw.get(f"block_{bi}/txt_mlp", [0])[step_idx]
            for bi in SAMPLED_BLOCKS
            if f"block_{bi}/txt_mlp" in level3_raw and step_idx < len(level3_raw.get(f"block_{bi}/txt_mlp", []))
        )
        sampled_block_total = sum(
            level3_raw.get(f"block_{bi}", [0])[step_idx]
            for bi in SAMPLED_BLOCKS
            if f"block_{bi}" in level3_raw and step_idx < len(level3_raw.get(f"block_{bi}", []))
        )
        total_other = sampled_block_total - total_attn - total_img_mlp - total_txt_mlp

        if sampled_block_total > 0:
            print(f"\n    Sampled block avg breakdown:")
            print(f"      Attention:  {total_attn/len(SAMPLED_BLOCKS):7.2f}ms  ({total_attn/sampled_block_total*100:.1f}%)")
            print(f"      Image MLP:  {total_img_mlp/len(SAMPLED_BLOCKS):7.2f}ms  ({total_img_mlp/sampled_block_total*100:.1f}%)")
            print(f"      Text MLP:   {total_txt_mlp/len(SAMPLED_BLOCKS):7.2f}ms  ({total_txt_mlp/sampled_block_total*100:.1f}%)")
            print(f"      Other:      {total_other/len(SAMPLED_BLOCKS):7.2f}ms  ({total_other/sampled_block_total*100:.1f}%)")

# ─── Save Raw Data ───
# JSON with all results
results = {
    "level1": {k: round(v, 6) for k, v in level1.items()},
    "level1_total_s": round(t_total, 3),
    "level1_baseline_s": 51.2,
    "level1_overhead_pct": round((t_total - 51.2) / 51.2 * 100, 1),
    "peak_vram_gb": round(peak_vram_gb, 2),
    "warmup_time_s": round(warmup_time, 1),
    "level2": step_details,
    "level2_summary": {
        "mean_ms": round(np.mean(step_totals), 1),
        "std_ms": round(np.std(step_totals), 1),
        "min_ms": round(float(np.min(step_totals)), 1),
        "max_ms": round(float(np.max(step_totals)), 1),
    },
    "level3_raw_ms": {k: [round(v, 3) for v in vals] for k, vals in level3_raw.items()},
    "sampled_steps": SAMPLED_STEPS,
    "sampled_blocks": SAMPLED_BLOCKS,
}

json_path = os.path.join(OUTPUT_DIR, "profiling_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nRaw data saved to {json_path}")

# CSV with per-step timing
csv_path = os.path.join(OUTPUT_DIR, "per_step_timing.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["step", "total_ms", "cond_ms", "uncond_ms", "sched_ms"])
    writer.writeheader()
    for s in step_details:
        writer.writerow({k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()})
print(f"Per-step CSV saved to {csv_path}")

print(f"\nOutput image saved to {OUTPUT_DIR}/m3_profiled_output.png")
print("\nDone.")
