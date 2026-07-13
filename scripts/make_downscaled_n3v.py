#!/usr/bin/env python3
import argparse, json, shutil
from pathlib import Path

import numpy as np
from PIL import Image

def scale_transforms(path_in: Path, path_out: Path, s: float):
    with open(path_in, "r") as f:
        data = json.load(f)

    keys_to_scale = ["w", "h", "fl_x", "fl_y", "cx", "cy"]

    def scale_in_obj(obj):
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                if k in keys_to_scale and isinstance(obj[k], (int, float)):
                    new_val = obj[k] * s
                    # Ensure width and height are integers to match physical images
                    if k in ["w", "h"]:
                        obj[k] = float(max(1, int(round(new_val))))
                    else:
                        obj[k] = new_val
                else:
                    scale_in_obj(obj[k])
        elif isinstance(obj, list):
            for it in obj:
                scale_in_obj(it)

    scale_in_obj(data)

    with open(path_out, "w") as f:
        json.dump(data, f, indent=2)

def scale_poses_bounds(path_in: Path, path_out: Path, s: float):
    """
    LLFF-style poses_bounds.npy is typically shape (N, 17) where
    first 15 entries are a 3x5 pose matrix, last 2 are bounds.
    The 3x5 includes H, W, F in the last column (or in the 3rd row depending on convention).
    We’ll scale H, W, and F if we can detect them reliably.
    """
    arr = np.load(path_in)

    # Handle common LLFF layout: pose is 3x5 flattened => (N, 15) + bounds (2) => 17
    if arr.ndim == 2 and arr.shape[1] >= 15:
        poses = arr[:, :15].reshape(-1, 3, 5)

        # Heuristic: H, W, F often live in poses[:, :, 4] (last column): [H, W, F]
        hwf = poses[:, :, 4]
        # If values look like plausible image dimensions and focal (positive, not tiny):
        if np.all(hwf[:, 0] > 1) and np.all(hwf[:, 1] > 1) and np.all(hwf[:, 2] > 1):
            poses[:, 0, 4] *= s  # H
            poses[:, 1, 4] *= s  # W
            poses[:, 2, 4] *= s  # F

        out = arr.copy()
        out[:, :15] = poses.reshape(-1, 15)
        np.save(path_out, out)
    else:
        # Unknown format, just copy
        shutil.copy2(path_in, path_out)

def resize_images(src_dir: Path, dst_dir: Path, scale_factor: float, exts=(".png", ".jpg", ".jpeg")):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            img = Image.open(p)
            w, h = img.size
            nw, nh = max(1, int(round(w * scale_factor))), max(1, int(round(h * scale_factor)))
            img = img.resize((nw, nh), resample=Image.LANCZOS)
            img.save(dst_dir / p.name)
        elif p.is_dir():
            # preserve subfolders if you have them
            resize_images(p, dst_dir / p.name, scale_factor, exts)

def copy_non_image_files(scene_in: Path, scene_out: Path, keep_videos=False):
    scene_out.mkdir(parents=True, exist_ok=True)
    for p in scene_in.iterdir():
        if p.name == "images":
            continue
        if (not keep_videos) and p.suffix.lower() == ".mp4":
            continue
        if p.is_file():
            shutil.copy2(p, scene_out / p.name)
        elif p.is_dir():
            # copy other dirs if any (not images)
            shutil.copytree(p, scene_out / p.name, dirs_exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="e.g. $WORK/proj_adags/data/n3v")
    ap.add_argument("--dst_root", required=True, help="e.g. $WORK/proj_adags/data/n3v_scaled")
    ap.add_argument("--scales", nargs="+", default=["2","4","8"], help="downscale factors (2 means 2x downscale)")
    ap.add_argument("--scenes", nargs="+", required=True, help="scene folder names")
    ap.add_argument("--keep_videos", action="store_true", help="copy mp4s too")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    for ds in args.scales:
        d = int(ds)
        sf = 1.0 / d  # scale factor
        for scene in args.scenes:
            scene_in = src_root / scene
            scene_out = dst_root / f"x{d}" / scene

            if not scene_in.exists():
                raise FileNotFoundError(f"Missing scene: {scene_in}")

            # Copy metadata first
            copy_non_image_files(scene_in, scene_out, keep_videos=args.keep_videos)

            # Resize images
            resize_images(scene_in / "images", scene_out / "images", sf)

            # Scale transforms if present
            for name in ["transforms_train.json", "transforms_test.json"]:
                pin = scene_in / name
                if pin.exists():
                    scale_transforms(pin, scene_out / name, sf)

            # Scale poses_bounds if present
            pb = scene_in / "poses_bounds.npy"
            if pb.exists():
                scale_poses_bounds(pb, scene_out / "poses_bounds.npy", sf)

            print(f"✅ Wrote {scene_out}")

if __name__ == "__main__":
    main()