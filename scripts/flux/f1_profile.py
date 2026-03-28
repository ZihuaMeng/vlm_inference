#!/usr/bin/env python3
"""F1 Phase 2: Structured Profiling of FLUX Kontext-dev

Three-level profiling adapted from Qwen M3 methodology:
  Level 1: Pipeline stage breakdown (text enc, image preprocess, denoising, VAE dec)
  Level 2: Per-step denoising dynamics (all 28 steps, single pass — no CFG)
  Level 3: Module-level breakdown within sampled denoising steps
           - Dual-stream blocks (19): attention, ff (image), ff_context (text), norms
           - Single-stream blocks (38): attention, proj_mlp (MLP), proj_out, norm
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
SAMPLED_DUAL_BLOCKS = [0, 9, 18]
SAMPLED_SINGLE_BLOCKS = [0, 18, 37]
SAMPLED_STEPS = [0, 13, 27]

os.makedirs(OUTPUT_DIR, exist_ok=True)


class ModuleProfiler:
    def __init__(self):
        self.records = defaultdict(list)
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
from diffusers import FluxKontextPipeline

pipeline = FluxKontextPipeline.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
pipeline.to("cuda")
print(f"Pipeline loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Import module-level helpers used by pipeline.__call__
import importlib
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

# ─── Set up Level 3 profiler ───
profiler = ModuleProfiler()
transformer = pipeline.transformer

for name in ["x_embedder", "context_embedder", "time_text_embed", "pos_embed",
             "norm_out", "proj_out"]:
    profiler.register(name, getattr(transformer, name))

for i, block in enumerate(transformer.transformer_blocks):
    profiler.register(f"dual_{i}", block)
    if i in SAMPLED_DUAL_BLOCKS:
        profiler.register(f"dual_{i}/attn", block.attn)
        profiler.register(f"dual_{i}/ff", block.ff)
        profiler.register(f"dual_{i}/ff_context", block.ff_context)
        profiler.register(f"dual_{i}/norm1", block.norm1)
        profiler.register(f"dual_{i}/norm1_context", block.norm1_context)
        profiler.register(f"dual_{i}/norm2", block.norm2)
        profiler.register(f"dual_{i}/norm2_context", block.norm2_context)

for i, block in enumerate(transformer.single_transformer_blocks):
    profiler.register(f"single_{i}", block)
    if i in SAMPLED_SINGLE_BLOCKS:
        profiler.register(f"single_{i}/attn", block.attn)
        profiler.register(f"single_{i}/proj_mlp", block.proj_mlp)
        profiler.register(f"single_{i}/proj_out", block.proj_out)
        profiler.register(f"single_{i}/norm", block.norm)

# ─── Profiled Pass ───
print("\n" + "=" * 60)
print("Starting profiled pass...")
print("=" * 60)

pipeline._guidance_scale = GUIDANCE_SCALE
pipeline._joint_attention_kwargs = {}
pipeline._current_timestep = None
pipeline._interrupt = False

torch.cuda.reset_peak_memory_stats()
multiple_of = pipeline.vae_scale_factor * 2

with torch.no_grad():
    # ── Stage 1: Image Preprocessing ──
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    image_height, image_width = pipeline.image_processor.get_default_height_width(ref_image)
    aspect_ratio = image_width / image_height
    _, image_width, image_height = min(
        (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
    )
    image_width = image_width // multiple_of * multiple_of
    image_height = image_height // multiple_of * multiple_of

    # Target generation dimensions
    height = 1024
    width = 1024
    ar = width / height
    width = round((1024**2 * ar) ** 0.5)
    height = round((1024**2 / ar) ** 0.5)
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    image_resized = pipeline.image_processor.resize(ref_image, image_height, image_width)
    image_preprocessed = pipeline.image_processor.preprocess(image_resized, image_height, image_width)

    torch.cuda.synchronize()
    t_preprocess = time.perf_counter()

    # ── Stage 2: Text Encoding ──
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt, prompt_2=None,
        device="cuda", num_images_per_prompt=1,
        max_sequence_length=512,
    )

    torch.cuda.synchronize()
    t_text_enc = time.perf_counter()

    # ── Stage 3: Prepare Latents (includes VAE encoding of reference image) ──
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    latents, image_latents, latent_ids, image_ids = pipeline.prepare_latents(
        image_preprocessed,
        1, num_channels_latents, height, width,
        prompt_embeds.dtype, "cuda",
        torch.Generator("cuda").manual_seed(SEED),
    )
    if image_ids is not None:
        latent_ids = torch.cat([latent_ids, image_ids], dim=0)

    torch.cuda.synchronize()
    t_vae_enc = time.perf_counter()

    # ── Stage 4: Scheduler Setup ──
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

    noise_len = latents.shape[1]
    print(f"Noise latent tokens: {noise_len}")
    print(f"Image latent tokens: {image_latents.shape[1]}")
    print(f"Total input tokens: {noise_len + image_latents.shape[1]}")

    torch.cuda.synchronize()
    t_sched = time.perf_counter()

    # ── Stage 5: Denoising Loop ──
    step_details = []

    for i, t in enumerate(timesteps):
        torch.cuda.synchronize()
        step_start = time.perf_counter()

        latent_model_input = torch.cat([latents, image_latents], dim=1)
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        if i in SAMPLED_STEPS:
            profiler.active = True

        torch.cuda.synchronize()
        pass_start = time.perf_counter()
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
        noise_pred = noise_pred[:, :noise_len]
        torch.cuda.synchronize()
        pass_end = time.perf_counter()

        profiler.active = False

        torch.cuda.synchronize()
        sched_start = time.perf_counter()
        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        torch.cuda.synchronize()
        sched_end = time.perf_counter()

        step_details.append({
            "step": i,
            "pass_ms": (pass_end - pass_start) * 1000,
            "sched_ms": (sched_end - sched_start) * 1000,
            "total_ms": (time.perf_counter() - step_start) * 1000,
        })

        if (i + 1) % 7 == 0 or i == 0:
            print(f"  Step {i:2d}/{NUM_STEPS}: {step_details[-1]['total_ms']:.1f}ms "
                  f"(pass={step_details[-1]['pass_ms']:.1f}ms)")

    torch.cuda.synchronize()
    t_denoise = time.perf_counter()

    # ── Stage 6: VAE Decoding ──
    latents_dec = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    latents_dec = (latents_dec / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    decoded = pipeline.vae.decode(latents_dec, return_dict=False)[0]
    output_image = pipeline.image_processor.postprocess(decoded, output_type="pil")

    torch.cuda.synchronize()
    t_vae_dec = time.perf_counter()

t_total = t_vae_dec - t0
peak_vram_gb = torch.cuda.max_memory_reserved() / 1e9

output_image[0].save(os.path.join(OUTPUT_DIR, "f1_profiled_output.png"))

level3_raw = profiler.get_times_ms()
profiler.remove_hooks()

# ═══════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════

level1 = {
    "image_preprocess": t_preprocess - t0,
    "text_encoding": t_text_enc - t_preprocess,
    "vae_encoding_latent_prep": t_vae_enc - t_text_enc,
    "scheduler_setup": t_sched - t_vae_enc,
    "denoising_loop": t_denoise - t_sched,
    "vae_decoding": t_vae_dec - t_denoise,
}

print("\n" + "=" * 70)
print("LEVEL 1: PIPELINE STAGE BREAKDOWN")
print("=" * 70)
print(f"  {'Stage':<30s} {'Time (s)':>10s} {'Pct':>8s}")
print(f"  {'-'*30} {'-'*10} {'-'*8}")
for name, val in level1.items():
    pct = val / t_total * 100
    print(f"  {name:<30s} {val:10.3f} {pct:7.1f}%")
print(f"  {'-'*30} {'-'*10} {'-'*8}")
print(f"  {'TOTAL':<30s} {t_total:10.3f}")
print(f"  Peak VRAM: {peak_vram_gb:.2f} GB")

pass_times = [s["pass_ms"] for s in step_details]
sched_times = [s["sched_ms"] for s in step_details]
total_times = [s["total_ms"] for s in step_details]

print("\n" + "=" * 70)
print("LEVEL 2: PER-STEP DENOISING TIMING (single pass, no CFG)")
print("=" * 70)
print(f"  Step total:    mean={np.mean(total_times):.1f}ms  std={np.std(total_times):.1f}ms  "
      f"min={np.min(total_times):.1f}ms  max={np.max(total_times):.1f}ms")
print(f"  Pass time:     mean={np.mean(pass_times):.1f}ms  std={np.std(pass_times):.1f}ms")
print(f"  Scheduler:     mean={np.mean(sched_times):.1f}ms  std={np.std(sched_times):.1f}ms")
print(f"\n  {'Step':>6s} {'Total':>9s} {'Pass':>9s} {'Sched':>9s}")
print(f"  {'-'*6} {'-'*9} {'-'*9} {'-'*9}")
for s in step_details:
    print(f"  {s['step']:6d} {s['total_ms']:9.1f} {s['pass_ms']:9.1f} {s['sched_ms']:9.1f}")

print("\n" + "=" * 70)
print("LEVEL 3: MODULE-LEVEL BREAKDOWN (sampled steps)")
print("=" * 70)

for step_idx, step_num in enumerate(SAMPLED_STEPS):
    print(f"\n  --- Step {step_num} ---")

    def get_val(key):
        if key in level3_raw and step_idx < len(level3_raw[key]):
            return level3_raw[key][step_idx]
        return 0.0

    embed_ms = sum(get_val(n) for n in ["x_embedder", "context_embedder", "time_text_embed", "pos_embed"])
    output_ms = sum(get_val(n) for n in ["norm_out", "proj_out"])

    dual_times = [get_val(f"dual_{i}") for i in range(19)]
    single_times = [get_val(f"single_{i}") for i in range(38)]
    dual_total = sum(dual_times)
    single_total = sum(single_times)
    blocks_total = dual_total + single_total
    transformer_total = embed_ms + blocks_total + output_ms

    print(f"    {'Component':<35s} {'Time (ms)':>10s} {'Pct':>8s}")
    print(f"    {'-'*35} {'-'*10} {'-'*8}")
    if transformer_total > 0:
        print(f"    {'Embeddings (x/ctx/time/pos)':<35s} {embed_ms:10.2f} {embed_ms/transformer_total*100:7.1f}%")
        print(f"    {'Dual-stream blocks (19)':<35s} {dual_total:10.2f} {dual_total/transformer_total*100:7.1f}%")
        print(f"    {'Single-stream blocks (38)':<35s} {single_total:10.2f} {single_total/transformer_total*100:7.1f}%")
        print(f"    {'Output (norm_out+proj_out)':<35s} {output_ms:10.2f} {output_ms/transformer_total*100:7.1f}%")
        print(f"    {'-'*35} {'-'*10}")
        print(f"    {'Transformer forward total':<35s} {transformer_total:10.2f}")

    if dual_times and any(v > 0 for v in dual_times):
        print(f"\n    Dual-stream (19): mean={np.mean(dual_times):.2f}ms  "
              f"std={np.std(dual_times):.2f}ms")
    if single_times and any(v > 0 for v in single_times):
        print(f"    Single-stream (38): mean={np.mean(single_times):.2f}ms  "
              f"std={np.std(single_times):.2f}ms")
    if dual_times and single_times and np.mean(single_times) > 0:
        print(f"    Dual/Single cost ratio: {np.mean(dual_times)/np.mean(single_times):.2f}x")

    print(f"\n    Sampled Dual-Stream Block Internals:")
    print(f"    {'Block':<12s} {'Attention':>10s} {'FF(img)':>10s} {'FF(txt)':>10s} {'Norms':>10s} {'Other':>10s} {'Total':>10s}")
    print(f"    {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for bi in SAMPLED_DUAL_BLOCKS:
        bt = get_val(f"dual_{bi}")
        at = get_val(f"dual_{bi}/attn")
        ff_i = get_val(f"dual_{bi}/ff")
        ff_c = get_val(f"dual_{bi}/ff_context")
        norms = sum(get_val(f"dual_{bi}/{n}") for n in ["norm1", "norm1_context", "norm2", "norm2_context"])
        other = bt - at - ff_i - ff_c - norms
        print(f"    Dual {bi:<5d} {at:10.2f} {ff_i:10.2f} {ff_c:10.2f} {norms:10.2f} {other:10.2f} {bt:10.2f}")

    print(f"\n    Sampled Single-Stream Block Internals:")
    print(f"    {'Block':<12s} {'Attention':>10s} {'ProjMLP':>10s} {'ProjOut':>10s} {'Norm':>10s} {'Other':>10s} {'Total':>10s}")
    print(f"    {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for bi in SAMPLED_SINGLE_BLOCKS:
        bt = get_val(f"single_{bi}")
        at = get_val(f"single_{bi}/attn")
        pm = get_val(f"single_{bi}/proj_mlp")
        po = get_val(f"single_{bi}/proj_out")
        nm = get_val(f"single_{bi}/norm")
        other = bt - at - pm - po - nm
        print(f"    Sing {bi:<5d} {at:10.2f} {pm:10.2f} {po:10.2f} {nm:10.2f} {other:10.2f} {bt:10.2f}")

# ─── Aggregate attention vs MLP across sampled blocks ───
print("\n    Aggregate Component Breakdown (sampled blocks):")
for step_idx, step_num in enumerate(SAMPLED_STEPS):
    def get_val(key):
        if key in level3_raw and step_idx < len(level3_raw[key]):
            return level3_raw[key][step_idx]
        return 0.0

    # Dual-stream aggregated
    dual_attn = sum(get_val(f"dual_{bi}/attn") for bi in SAMPLED_DUAL_BLOCKS)
    dual_ff = sum(get_val(f"dual_{bi}/ff") for bi in SAMPLED_DUAL_BLOCKS)
    dual_ffc = sum(get_val(f"dual_{bi}/ff_context") for bi in SAMPLED_DUAL_BLOCKS)
    dual_norms = sum(
        sum(get_val(f"dual_{bi}/{n}") for n in ["norm1", "norm1_context", "norm2", "norm2_context"])
        for bi in SAMPLED_DUAL_BLOCKS
    )
    dual_sampled_total = sum(get_val(f"dual_{bi}") for bi in SAMPLED_DUAL_BLOCKS)

    # Single-stream aggregated
    single_attn = sum(get_val(f"single_{bi}/attn") for bi in SAMPLED_SINGLE_BLOCKS)
    single_mlp = sum(get_val(f"single_{bi}/proj_mlp") for bi in SAMPLED_SINGLE_BLOCKS)
    single_proj = sum(get_val(f"single_{bi}/proj_out") for bi in SAMPLED_SINGLE_BLOCKS)
    single_norm = sum(get_val(f"single_{bi}/norm") for bi in SAMPLED_SINGLE_BLOCKS)
    single_sampled_total = sum(get_val(f"single_{bi}") for bi in SAMPLED_SINGLE_BLOCKS)

    if dual_sampled_total + single_sampled_total > 0:
        total = dual_sampled_total + single_sampled_total
        total_attn = dual_attn + single_attn
        print(f"\n    Step {step_num}: Total attention = {total_attn:.1f}ms ({total_attn/total*100:.1f}%)")
        print(f"      Dual: attn={dual_attn:.1f}ms  ff_img={dual_ff:.1f}ms  ff_txt={dual_ffc:.1f}ms  norms={dual_norms:.1f}ms")
        print(f"      Single: attn={single_attn:.1f}ms  proj_mlp={single_mlp:.1f}ms  proj_out={single_proj:.1f}ms  norm={single_norm:.1f}ms")

# ─── Save ───
results = {
    "level1": {k: round(v, 6) for k, v in level1.items()},
    "level1_total_s": round(t_total, 3),
    "peak_vram_gb": round(peak_vram_gb, 2),
    "warmup_time_s": round(warmup_time, 1),
    "level2": step_details,
    "level2_summary": {
        "mean_ms": round(np.mean(total_times), 1),
        "std_ms": round(np.std(total_times), 1),
        "min_ms": round(float(np.min(total_times)), 1),
        "max_ms": round(float(np.max(total_times)), 1),
    },
    "level3_raw_ms": {k: [round(v, 3) for v in vals] for k, vals in level3_raw.items()},
    "sampled_steps": SAMPLED_STEPS,
    "sampled_dual_blocks": SAMPLED_DUAL_BLOCKS,
    "sampled_single_blocks": SAMPLED_SINGLE_BLOCKS,
}

json_path = os.path.join(OUTPUT_DIR, "f1_profiling_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nRaw data saved to {json_path}")
print("\nDone.")
