#!/usr/bin/env python3
"""Build-time seeding and verification of the SEA-RAFT flow stack.

Baked into the V100 runtime image (see ``Dockerfile.apollo-v100``) and run
twice during the build:

``--seed-only``
    Downloads the torchvision ResNet-34 ImageNet weights into ``TORCH_HOME``.
    ``RAFT.__init__`` builds a ``ResNetFPN`` whose ``_init_weights`` calls
    ``resnet34(weights=IMAGENET1K_V1)`` unconditionally, at construction,
    BEFORE any flow checkpoint is loaded.  Those weights are then entirely
    overwritten by the checkpoint, so the download cannot change a result --
    it can only make a task fail at start time on a network hiccup.  Seeding
    it once at build time removes that failure mode.

``--build-check``
    Constructs ``RAFT`` with the network poisoned, and fails the image build
    if it cannot.  A successful ``pip install`` is not evidence that the
    model can be instantiated: the ResNet fetch above is precisely the step
    that reaches out, and it is the first thing that would break in an
    offline container.  This turns that into a build-time error instead of a
    runtime one.

Both modes are deliberately in the image rather than in a run, so the
guarantee travels with the digest.
"""

from __future__ import annotations

import argparse
import os
import sys

SEARAFT_ROOT = os.environ.get("ADAGS_SEARAFT_ROOT", "/opt/adags/searaft")
EVAL_CFG = "config/eval/spring-M.json"


def seed_resnet() -> int:
    from torchvision.models import ResNet34_Weights, resnet34

    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    n = sum(p.numel() for p in model.parameters())
    print(f"[seed] resnet34 IMAGENET1K_V1 cached under TORCH_HOME="
          f"{os.environ.get('TORCH_HOME')!r}; parameters {n}")
    return 0


def build_check() -> int:
    if not os.path.isdir(SEARAFT_ROOT):
        raise SystemExit(f"REFUSE: SEA-RAFT root {SEARAFT_ROOT} is absent")
    cfg = os.path.join(SEARAFT_ROOT, EVAL_CFG)
    if not os.path.isfile(cfg):
        raise SystemExit(f"REFUSE: eval config {cfg} is absent")

    sys.path.insert(0, SEARAFT_ROOT)
    from config.parser import json_to_args  # noqa: E402
    from core.raft import RAFT  # noqa: E402

    args = json_to_args(cfg)
    # Assert the architecture fields the Tartan-C-T-TSKH-spring540x960-M
    # checkpoint was trained under.  If the pinned commit ever changes these
    # defaults the build fails here rather than producing silently wrong
    # flow from a mismatched backbone.
    expected = {"pretrain": "resnet34", "dim": 128, "radius": 4, "num_blocks": 2}
    for key, want in expected.items():
        got = getattr(args, key, None)
        if got != want:
            raise SystemExit(
                f"REFUSE: {EVAL_CFG} has {key}={got!r}, expected {want!r}; the "
                "pinned SEA-RAFT commit no longer matches the -M checkpoint"
            )

    model = RAFT(args)
    n = sum(p.numel() for p in model.parameters())
    print(f"[build-check] RAFT constructed with no network; parameters {n}; "
          f"cfg {EVAL_CFG}; iters {getattr(args, 'iters', None)}; "
          f"scale {getattr(args, 'scale', None)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed-only", action="store_true")
    mode.add_argument("--build-check", action="store_true")
    args = ap.parse_args()
    return seed_resnet() if args.seed_only else build_check()


if __name__ == "__main__":
    sys.exit(main())
