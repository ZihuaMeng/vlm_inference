"""Prepare NYUDepthV2 in-context task instances for VLM ICL profiling.

Structure: 3 demonstration pairs (RGB → depth) + 1 query RGB → model generates depth.
Prepares 10 task instances from the validation split.
"""

import os
import json
from pathlib import Path
from PIL import Image
import numpy as np

DATA_ROOT = "/blue/yuanyuan.lei/zihua.meng/vlm_inference/vlm_icl/data/nyu_depth_v2"
TARGET_SIZE = 512  # Resize to 512x512 for VQ-VAE tokenization

def colorize_depth(depth_map: Image.Image) -> Image.Image:
    """Convert a float32 depth map to a colorized RGB image using plasma colormap."""
    depth_array = np.array(depth_map, dtype=np.float32)
    # Normalize to 0-1
    d_min, d_max = depth_array.min(), depth_array.max()
    if d_max - d_min > 0:
        depth_norm = (depth_array - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_array)
    # Apply plasma-like colormap (simple version without matplotlib dependency)
    # Map depth to RGB: near=warm (red/yellow), far=cool (blue/purple)
    r = np.clip(255 * (1.0 - depth_norm), 0, 255).astype(np.uint8)
    g = np.clip(255 * (1.0 - np.abs(depth_norm - 0.5) * 2), 0, 255).astype(np.uint8)
    b = np.clip(255 * depth_norm, 0, 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb)


def main():
    os.makedirs(DATA_ROOT, exist_ok=True)

    print("Loading NYUDepthV2 validation split...")
    from datasets import load_dataset
    ds = load_dataset("sayakpaul/nyu_depth_v2", split="validation", streaming=True, trust_remote_code=True)

    # Collect 40 diverse pairs (enough for 10 task instances of 4 pairs each)
    pairs = []
    for i, sample in enumerate(ds):
        if i >= 40:
            break
        rgb = sample["image"].convert("RGB").resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
        depth = sample["depth_map"].resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
        depth_colored = colorize_depth(depth)

        # Save images
        pair_dir = os.path.join(DATA_ROOT, f"pair_{i:03d}")
        os.makedirs(pair_dir, exist_ok=True)
        rgb.save(os.path.join(pair_dir, "rgb.png"))
        depth_colored.save(os.path.join(pair_dir, "depth_colored.png"))
        # Also save raw depth as numpy for potential metric computation later
        depth_array = np.array(depth, dtype=np.float32)
        np.save(os.path.join(pair_dir, "depth_raw.npy"), depth_array)

        pairs.append({
            "index": i,
            "rgb_path": os.path.join(pair_dir, "rgb.png"),
            "depth_colored_path": os.path.join(pair_dir, "depth_colored.png"),
            "depth_raw_path": os.path.join(pair_dir, "depth_raw.npy"),
        })
        print(f"  Saved pair {i:03d}")

    print(f"\nSaved {len(pairs)} pairs.")

    # Create 10 task instances
    # Each instance: 3 demo pairs + 1 query pair (query depth used as ground truth)
    instances = []
    for inst_idx in range(10):
        # Use a sliding window: demos = pairs[inst*4 : inst*4+3], query = pairs[inst*4+3]
        base = inst_idx * 4
        if base + 4 > len(pairs):
            print(f"  Not enough pairs for instance {inst_idx}, stopping.")
            break
        demo_indices = [base, base + 1, base + 2]
        query_index = base + 3

        instance = {
            "instance_id": inst_idx,
            "demo_pairs": [pairs[j] for j in demo_indices],
            "query": pairs[query_index],
        }
        instances.append(instance)
        print(f"  Created instance {inst_idx}: demos={demo_indices}, query={query_index}")

    # Save manifest
    manifest_path = os.path.join(DATA_ROOT, "task_instances.json")
    with open(manifest_path, "w") as f:
        json.dump(instances, f, indent=2)
    print(f"\nSaved {len(instances)} task instances to {manifest_path}")

    # Print summary
    print("\n=== Dataset Summary ===")
    print(f"Total pairs: {len(pairs)}")
    print(f"Task instances: {len(instances)}")
    print(f"Structure: 3 demo (RGB→depth) + 1 query (RGB→?)")
    print(f"Image size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Data root: {DATA_ROOT}")


if __name__ == "__main__":
    main()
