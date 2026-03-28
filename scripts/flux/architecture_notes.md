# FLUX.1 Kontext-dev — Architecture Notes

Reference: arXiv:2506.15742, HuggingFace model card (black-forest-labs/FLUX.1-Kontext-dev)

## Pipeline Entry Point

```python
from diffusers import FluxKontextPipeline
import torch

pipeline = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
)
pipeline.to("cuda")

result = pipeline(
    image=reference_image,       # Single PIL Image (reference/context)
    prompt="edit instruction",
    guidance_scale=2.5,          # Embedded guidance (distilled)
    num_inference_steps=28,
)
```

## Key Source Files (diffusers 0.38.0.dev0)

| File | Location |
|------|----------|
| Pipeline | `diffusers/pipelines/flux/pipeline_flux_kontext.py` |
| Transformer | `diffusers/models/transformers/transformer_flux.py` |
| Scheduler | `diffusers/schedulers/scheduling_flow_match_euler_discrete.py` |
| Attention | `FluxAttnProcessor` in transformer_flux.py |

## Architecture Overview

- **Type**: Rectified flow transformer (flow matching in VAE latent space)
- **Total transformer params**: ~12B
- **Blocks**: 19 dual-stream (MMDiT) + 38 single-stream = 57 total
- **Hidden dim**: 3072 (24 heads x 128 dim/head)
- **Text encoders**: CLIP-L (pooled, 77 tokens) + T5-v1_1-XXL (sequence, 512 tokens)
- **VAE**: Custom Flux-VAE, 16 latent channels, 8x spatial compression + 2x2 packing = 16x effective
- **Guidance**: Guidance-distilled (single pass). True CFG available via `true_cfg_scale > 1.0` but disabled by default.
- **Default steps**: 28
- **Default guidance_scale**: 3.5 (pipeline) / 2.5 (model card example)

## Dual-Stream Block (FluxTransformerBlock, x19)

Separate normalization and FFN for text and image streams, connected by joint attention.

```
Image stream:  AdaLN-Zero -> Q,K,V projection -+
                                                |-> Joint Attention -> gate -> residual
Text stream:   AdaLN-Zero -> Q,K,V projection -+
                                                    |
Image stream:  LayerNorm -> shift/scale -> FFN (3072->12288->3072) -> gate -> residual
Text stream:   LayerNorm -> shift/scale -> FFN (3072->12288->3072) -> gate -> residual
```

- ~340M params per block, ~6.46B total

## Single-Stream Block (FluxSingleTransformerBlock, x38)

Text and image tokens merged into a single sequence. Attention and MLP computed in parallel.

```
Merged sequence: AdaLN-ZeroSingle -> split -> [Attention(pre_only) | MLP(3072->12288)]
                                              -> concat -> proj_out(15360->3072) -> gate -> residual
```

- ~142M params per block, ~5.38B total

## Image Conditioning (Reference Image)

1. Reference image -> Flux-VAE encode -> shift/scale -> 2x2 pack -> `image_latents`
2. Image positional IDs: `(1, h, w)` vs target noise `(0, h, w)` — virtual time-step offset
3. Concatenate: `latent_model_input = cat([noise_latents, image_latents], dim=1)` (sequence dim)
4. Transformer processes both, but only noise portion of output used: `noise_pred = output[:, :noise_len]`

For 1024x1024: ~4096 tokens per image (same as Qwen)

## Multi-Image Support

- Architecture supports multiple context images via different time-step offsets: target=(0,h,w), img1=(1,h,w), img2=(2,h,w), ...
- Current release: single reference image only (pipeline takes one image)
- Paper notes multi-image as future work

## Scheduler

- FlowMatchEulerDiscreteScheduler (same as Qwen)
- Velocity prediction: `prev_sample = sample + dt * model_output`
- Dynamic shift: mu interpolated between base_shift=0.5 and max_shift=1.15 based on sequence length
- Sigma schedule: linear from 1.0 to 1/N, shifted by mu

## VRAM Estimate (bf16)

| Component | Params | bf16 Size |
|-----------|--------|-----------|
| Transformer | ~12B | ~24 GB |
| T5-XXL encoder | ~4.7B | ~9.4 GB |
| CLIP-L encoder | ~0.43B | ~0.86 GB |
| VAE | ~0.08B | ~0.16 GB |
| **Model weights** | **~17.2B** | **~34.4 GB** |
| Activations (est.) | — | ~10-20 GB |
| **Estimated peak** | — | **~45-55 GB** |

- B200 (183 GB): YES, fits with >120 GB headroom
- L4 (23 GB): NO, requires sequential CPU offload

## Dependency Notes

- All core components available in diffusers 0.38.0.dev0 (our current venv)
- Missing: `sentencepiece` (needed for T5 tokenizer) — must install before loading
- No other package changes needed
