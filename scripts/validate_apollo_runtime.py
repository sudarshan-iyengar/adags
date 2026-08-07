#!/usr/bin/env python3
"""Validate the ADAGS container and, optionally, its assigned Apollo GPU/data."""

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


PYTHON_IMPORTS = (
    "cv2",
    "determined",
    "imagesize",
    "kornia",
    "lpips",
    "matplotlib",
    "mediapy",
    "numpy",
    "omegaconf",
    "PIL",
    "plyfile",
    "torch",
    "torchmetrics",
    "torchvision",
    "tqdm",
    "wandb",
    "yaml",
)

CUDA_EXTENSIONS = (
    "simple_knn._C",
    "pointops2_cuda",
    "_adags_diff_gaussian_rasterization",
)

COMMANDS = ("colmap", "ffmpeg", "git", "ninja", "nvcc", "tar", "unzip")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-check", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--expected-capability", choices=("7.0", "9.0"))
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--scene", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    report = {"commands": {}, "extensions": {}, "imports": {}}

    for name in PYTHON_IMPORTS:
        module = importlib.import_module(name)
        report["imports"][name] = getattr(module, "__version__", "ok")

    for name in CUDA_EXTENSIONS:
        importlib.import_module(name)
        report["extensions"][name] = "ok"

    for command in COMMANDS:
        executable = shutil.which(command)
        if executable is None:
            raise RuntimeError(f"required command is missing: {command}")
        report["commands"][command] = executable

    import torch

    report["python"] = sys.version.split()[0]
    report["torch"] = torch.__version__
    report["torch_cuda"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()
    report["nvcc"] = subprocess.check_output(("nvcc", "--version"), text=True).splitlines()[-1]

    if args.require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU was required but torch.cuda.is_available() is false")
        capability = ".".join(str(value) for value in torch.cuda.get_device_capability(0))
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_capability"] = capability
        report["gpu_count_visible"] = torch.cuda.device_count()
        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
        if args.expected_capability and capability != args.expected_capability:
            raise RuntimeError(
                f"expected compute capability {args.expected_capability}, found {capability}"
            )

    if args.repo:
        required_repo_paths = (
            args.repo / "main.py",
            args.repo / "configs" / "n3v" / "smoke_apollo.yaml",
            args.repo / "gaussian_renderer",
        )
        missing = [str(path) for path in required_repo_paths if not path.exists()]
        if missing:
            raise RuntimeError(f"repository paths are missing: {missing}")
        sys.path.insert(0, str(args.repo))
        importlib.import_module("main")
        report["repo"] = str(args.repo)
        report["main_import"] = "ok"

    if args.scene:
        required_scene_paths = (
            args.scene / "images",
            args.scene / "points3d.ply",
            args.scene / "transforms_test.json",
            args.scene / "transforms_train.json",
        )
        missing = [str(path) for path in required_scene_paths if not path.exists()]
        if missing:
            raise RuntimeError(f"scene paths are missing: {missing}")
        image_count = sum(1 for path in (args.scene / "images").iterdir() if path.is_file())
        if image_count == 0:
            raise RuntimeError(f"scene contains no images: {args.scene}")
        report["scene"] = str(args.scene)
        report["scene_image_count"] = image_count

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
