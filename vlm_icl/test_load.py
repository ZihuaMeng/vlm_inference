"""Quick verification that Lumina-mGPT-2.0-Omni loads and generates."""

import sys
import os
import time

# Set up paths
REPO_ROOT = "/home/zihua.meng/projects/vlm_inference/vlm_icl/Lumina-mGPT-2.0"
sys.path.insert(0, os.path.join(REPO_ROOT, "lumina_mgpt"))
sys.path.insert(0, REPO_ROOT)

MODEL_PATH = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/weights/Lumina-mGPT-2.0-Omni"

import torch

# MoVQGAN looks for weights relative to CWD
os.chdir(os.path.join(REPO_ROOT, "lumina_mgpt"))

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\nLoading FlexARInferenceSolver...")
t0 = time.time()
from inference_solver import FlexARInferenceSolver

solver = FlexARInferenceSolver(
    model_path=MODEL_PATH,
    precision="bf16",
    target_size=512,
)
t_load = time.time() - t0
print(f"Model loaded in {t_load:.1f}s")
print(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Test simple text generation
print("\nTest 1: Simple text generation...")
t0 = time.time()
result = solver.generate(
    images=[],
    qas=[["What is depth estimation?", None]],
    max_gen_len=50,
    temperature=0.7,
    logits_processor=solver.create_logits_processor(cfg=1.0, image_top_k=2000),
)
t_gen = time.time() - t0
print(f"Text generation: {t_gen:.2f}s")
print(f"Output: {result[0][:200]}")
print(f"VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Test i2i depth estimation with a single NYU image
print("\nTest 2: Depth estimation (i2i)...")
DATA_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/data/nyu_depth_v2"
test_image = os.path.join(DATA_ROOT, "pair_000", "rgb.png")

from PIL import Image
from data.item_processor import center_crop

# Prepare image: crop to 512 x 256 (half height for i2i dual-panel)
img = Image.open(test_image)
img_cropped = center_crop(img, crop_size=[512, 256])
cropped_path = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/outputs/test_cropped.png"
os.makedirs(os.path.dirname(cropped_path), exist_ok=True)
img_cropped.save(cropped_path)

prompt = "Generate a dual-panel image of 512x512 where the <upper half> displays a <depth map>, while the <lower half> retains the original image for direct visual comparison:\nAn indoor scene"

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
result = solver.generate(
    images=[cropped_path],
    qas=[[prompt, "<|image|>"]],
    max_gen_len=10240,
    temperature=1.0,
    logits_processor=solver.create_logits_processor(cfg=4.0, image_top_k=200),
)
t_gen = time.time() - t0
print(f"Depth generation: {t_gen:.2f}s")
print(f"Generated {len(result[1])} images")
print(f"VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

if result[1]:
    out_path = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/outputs/test_depth.png"
    result[1][0].save(out_path)
    print(f"Saved to {out_path}, size: {result[1][0].size}")

print("\n=== VERIFICATION COMPLETE ===")
