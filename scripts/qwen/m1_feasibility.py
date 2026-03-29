"""Milestone 1: Feasibility test — 2-image forward pass on L4."""
import os
import sys
import time
import torch
from PIL import Image

# Paths
WEIGHTS = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/weights/Qwen-Image-Edit-2511"
OUTPUT_DIR = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def vram_snapshot(label=""):
    """Print current VRAM usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[VRAM {label}] allocated={alloc:.2f}GB  reserved={reserved:.2f}GB")
    os.system("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits")

def create_test_images():
    """Create two simple solid-color test images (512x512)."""
    img1 = Image.new("RGB", (512, 512), color=(255, 0, 0))    # red
    img2 = Image.new("RGB", (512, 512), color=(0, 0, 255))    # blue
    return img1, img2

def run_inference(pipeline, img1, img2):
    """Run a single 2-image forward pass."""
    prompt = "A red object and a blue object sitting side by side on a white table."
    inputs = {
        "image": [img1, img2],
        "prompt": prompt,
        "generator": torch.manual_seed(42),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    vram_snapshot("pre-inference")
    t0 = time.time()
    with torch.inference_mode():
        output = pipeline(**inputs)
    elapsed = time.time() - t0
    vram_snapshot("post-inference")
    print(f"Inference wall-clock: {elapsed:.1f}s")
    return output.images[0], elapsed

# ── Strategy 1: Full bf16 on CUDA ──────────────────────────────────────────
def try_full_bf16():
    print("\n=== Strategy 1: Full bf16 .to('cuda') ===")
    from diffusers import QwenImageEditPlusPipeline
    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            WEIGHTS, torch_dtype=torch.bfloat16
        )
        pipeline.to("cuda")
        pipeline.set_progress_bar_config(disable=None)
        vram_snapshot("after-load")
        img1, img2 = create_test_images()
        out_img, elapsed = run_inference(pipeline, img1, img2)
        out_img.save(os.path.join(OUTPUT_DIR, "m1_bf16.png"))
        print(f"SUCCESS: bf16 — saved to {OUTPUT_DIR}/m1_bf16.png")
        return True, "bf16", elapsed
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM: {e}")
        torch.cuda.empty_cache()
        return False, "bf16_oom", 0
    except Exception as e:
        print(f"ERROR: {e}")
        torch.cuda.empty_cache()
        return False, f"bf16_error: {e}", 0

# ── Strategy 2: model_cpu_offload ──────────────────────────────────────────
def try_cpu_offload():
    print("\n=== Strategy 2: enable_model_cpu_offload ===")
    from diffusers import QwenImageEditPlusPipeline
    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            WEIGHTS, torch_dtype=torch.bfloat16
        )
        pipeline.enable_model_cpu_offload()
        pipeline.set_progress_bar_config(disable=None)
        vram_snapshot("after-load-offload")
        img1, img2 = create_test_images()
        out_img, elapsed = run_inference(pipeline, img1, img2)
        out_img.save(os.path.join(OUTPUT_DIR, "m1_cpu_offload.png"))
        print(f"SUCCESS: cpu_offload — saved to {OUTPUT_DIR}/m1_cpu_offload.png")
        return True, "cpu_offload", elapsed
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM: {e}")
        torch.cuda.empty_cache()
        return False, "cpu_offload_oom", 0
    except Exception as e:
        print(f"ERROR: {e}")
        torch.cuda.empty_cache()
        return False, f"cpu_offload_error: {e}", 0

# ── Strategy 3: sequential_cpu_offload ─────────────────────────────────────
def try_sequential_offload():
    print("\n=== Strategy 3: enable_sequential_cpu_offload ===")
    from diffusers import QwenImageEditPlusPipeline
    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            WEIGHTS, torch_dtype=torch.bfloat16
        )
        pipeline.enable_sequential_cpu_offload()
        pipeline.set_progress_bar_config(disable=None)
        vram_snapshot("after-load-seq-offload")
        img1, img2 = create_test_images()
        out_img, elapsed = run_inference(pipeline, img1, img2)
        out_img.save(os.path.join(OUTPUT_DIR, "m1_seq_offload.png"))
        print(f"SUCCESS: sequential_offload — saved to {OUTPUT_DIR}/m1_seq_offload.png")
        return True, "sequential_offload", elapsed
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM: {e}")
        torch.cuda.empty_cache()
        return False, "seq_offload_oom", 0
    except Exception as e:
        print(f"ERROR: {e}")
        torch.cuda.empty_cache()
        return False, f"seq_offload_error: {e}", 0

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Milestone 1 Feasibility Test — Qwen-Image-Edit-2511")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"torch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print("=" * 60)
    vram_snapshot("baseline")

    results = []

    # Skip bf16 (known OOM: 21.62GB model > 23GB VRAM).
    # Try offload strategies in order; stop on first success.
    results.append(("bf16", False, 0))  # recorded from prior run
    print("Skipping bf16 (known OOM from prior run)")
    for strategy_fn in [try_cpu_offload, try_sequential_offload]:
        ok, label, elapsed = strategy_fn()
        results.append((label, ok, elapsed))
        if ok:
            break
        # Force cleanup between attempts
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    for label, ok, elapsed in results:
        status = "SUCCESS" if ok else "FAILED"
        print(f"  {label}: {status}" + (f" ({elapsed:.1f}s)" if ok else ""))
    print("=" * 60)
