#!/usr/bin/env python
import argparse
import json
from itertools import groupby
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


def camera_group_key(path):
    stem = path.stem if isinstance(path, Path) else str(path)
    return camera_group_key_from_stem(stem)


def camera_group_key_from_stem(stem):
    if "_" not in stem:
        return stem
    return stem.rsplit("_", 1)[0]


def load_target_names(scene, transforms_file):
    transforms_path = Path(transforms_file)
    if not transforms_path.is_absolute():
        transforms_path = scene / transforms_path
    with transforms_path.open() as handle:
        contents = json.load(handle)
    target_names = set()
    for frame in contents.get("frames", []):
        file_path = frame.get("file_path")
        if file_path:
            target_names.add(Path(file_path).stem)
    if not target_names:
        raise RuntimeError(f"No frame file_path entries found in {transforms_path}")
    return target_names


def write_temporal_masks(paths, mask_dir, quantile, smooth_radius, min_value, target_names=None):
    prev_gray = None
    curr_gray = load_gray(paths[0])
    written = 0

    for idx, path in enumerate(paths):
        next_gray = load_gray(paths[idx + 1]) if idx + 1 < len(paths) else None
        diffs = []
        if prev_gray is not None:
            diffs.append(np.abs(curr_gray - prev_gray))
        if next_gray is not None:
            diffs.append(np.abs(curr_gray - next_gray))
        if diffs:
            score = np.maximum.reduce(diffs)
            score = smooth_mask(score, smooth_radius)
            threshold = max(float(np.quantile(score, quantile)), min_value)
            mask = (score >= threshold).astype(np.float32)
        else:
            mask = np.zeros_like(curr_gray, dtype=np.float32)
        if target_names is None or path.stem in target_names:
            Image.fromarray((mask * 255.0).astype(np.uint8)).save(mask_dir / f"{path.stem}.png")
            written += 1
        prev_gray = curr_gray
        curr_gray = next_gray

    return written


def main():
    parser = argparse.ArgumentParser(description="Build simple temporal-change dynamic mask priors.")
    parser.add_argument("--scene", required=True, help="Scene directory, e.g. data/n3v/cook_spinach")
    parser.add_argument("--images", default="images", help="Image subdirectory inside the scene")
    parser.add_argument("--out", default=None, help="Output prior root. Defaults to <scene>/motion_priors")
    parser.add_argument("--transforms", default=None, help="Optional transforms JSON. Only masks for listed frames are written.")
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

    target_names = load_target_names(scene, args.transforms) if args.transforms else None
    if target_names is not None:
        target_camera_names = {camera_group_key_from_stem(name) for name in target_names}
        paths = [path for path in paths if camera_group_key(path) in target_camera_names]
        if not paths:
            raise RuntimeError(f"No images matching cameras from {args.transforms} found in {image_dir}")

    written = 0
    for camera_name, group_iter in groupby(paths, key=camera_group_key):
        camera_paths = list(group_iter)
        written += write_temporal_masks(
            camera_paths,
            mask_dir,
            args.quantile,
            args.smooth_radius,
            args.min_value,
            target_names,
        )
        print(f"{camera_name}: scanned {len(camera_paths)} frames")

    print(f"Wrote {written} masks to {mask_dir}")
    print("Optional dense track-flow caches can be added as motion_priors/track_flows/<image_name>.npy with shape HxWx2.")


if __name__ == "__main__":
    main()
