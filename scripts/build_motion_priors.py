#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_gray(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def smooth_mask(mask, radius):
    if radius <= 0:
        return mask
    padded = np.pad(mask, radius, mode="edge")
    out = np.zeros_like(mask, dtype=np.float32)
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out += padded[radius + dy:radius + dy + mask.shape[0], radius + dx:radius + dx + mask.shape[1]]
            count += 1
    return out / float(count)


def main():
    parser = argparse.ArgumentParser(description="Build simple temporal-change dynamic mask priors.")
    parser.add_argument("--scene", required=True, help="Scene directory, e.g. data/n3v/cook_spinach")
    parser.add_argument("--images", default="images", help="Image subdirectory inside the scene")
    parser.add_argument("--out", default=None, help="Output prior root. Defaults to <scene>/motion_priors")
    parser.add_argument("--quantile", type=float, default=0.85)
    parser.add_argument("--smooth_radius", type=int, default=2)
    parser.add_argument("--min_value", type=float, default=0.05)
    args = parser.parse_args()

    scene = Path(args.scene)
    image_dir = scene / args.images
    out_root = Path(args.out) if args.out else scene / "motion_priors"
    mask_dir = out_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")

    grays = [load_gray(path) for path in paths]
    for idx, path in enumerate(paths):
        diffs = []
        if idx > 0:
            diffs.append(np.abs(grays[idx] - grays[idx - 1]))
        if idx + 1 < len(grays):
            diffs.append(np.abs(grays[idx] - grays[idx + 1]))
        if diffs:
            score = np.maximum.reduce(diffs)
            score = smooth_mask(score, args.smooth_radius)
            threshold = max(float(np.quantile(score, args.quantile)), args.min_value)
            mask = (score >= threshold).astype(np.float32)
        else:
            mask = np.zeros_like(grays[idx], dtype=np.float32)
        Image.fromarray((mask * 255.0).astype(np.uint8)).save(mask_dir / f"{path.stem}.png")

    print(f"Wrote {len(paths)} masks to {mask_dir}")
    print("Optional dense track-flow caches can be added as motion_priors/track_flows/<image_name>.npy with shape HxWx2.")


if __name__ == "__main__":
    main()
