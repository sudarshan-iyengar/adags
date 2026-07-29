#!/usr/bin/env python3
"""Checkpoint-aligned qualitative lifecycle panels with E1 evidence overlays.

Component 5 of the CSVL-VPL v2 pipeline. For each requested window
``(camera, center_frame)`` and each requested checkpoint this renders the
restored Gaussian model from that *training* camera at frames
``center - frames_around``, ``center``, ``center + frames_around`` and writes a
panel PNG with three rows:

* row 0: ground truth frame (downscaled to the render resolution);
* row 1: the render;
* row 2: the render with a deterministic subsample of projected primitives
  scattered on top, coloured by their ``EvidenceRuntime`` E1 verdict
  (green = near, red = occluded, blue = behind-weak, orange = in-front,
  grey = not-evaluable / uncertain).

When more than one checkpoint is supplied, a per-window contact sheet is also
written with the ground truth on the first row and one render+overlay row per
checkpoint, ordered and labelled by restored training iteration
(before / during / after).

The camera math is a deliberate *minimal* replication of the training loader so
that the rendered view is bit-comparable with what the trainer sees, without
dragging in ``Scene`` and its dataloader machinery:

* ``scene/dataset_readers.py::readCamerasFromTransforms`` (OpenGL -> OpenCV
  axis flip, ``w2c = inv(c2w)``, ``R = w2c[:3,:3].T``, ``T = w2c[:3,3]``,
  top-level ``fl_x/fl_y/cx/cy`` branch with ``FovX = FovY = -1.0``, ``far``
  rule, ``timestamp = frame['time']`` with the ``frame_ratio`` divide);
* ``utils/camera_utils.py::loadCam`` (``resolution_scale = 1.0``,
  ``scale = resolution_scale * ModelParams.resolution``, intrinsics divided by
  ``scale``, ``resolution = (round(w/scale), round(h/scale))``,
  ``meta_only = ModelParams.dataloader``);
* ``gaussian_renderer/__init__.py::render`` is called exactly as in
  ``main.py`` (``render(cam.cuda(), gaussians, pipe, background)`` with
  ``background`` from ``ModelParams.white_background``).

This script needs a GPU: ``gaussian_renderer.render`` and ``GaussianModel``
allocate on ``"cuda"`` unconditionally. Run it as a Slurm job. ``--help`` works
on a login node because every heavy import is deferred into ``main()``.

No scientific claim is derived here; these are qualitative diagnostics.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = "csvl-vpl-v2-qualitative-panel-v1"

# Deterministic overlay subsample seed. Fixed so that the same checkpoint and
# window always draw the same primitives, and so that two lanes with the same
# bank size are visually comparable.
SUBSAMPLE_SEED = 20260730

# Mirrors depth_visibility.evidence_runtime verdict codes. Re-declared locally
# so that --help does not have to import torch; checked against the module at
# runtime in _verify_verdict_codes().
V_NOT_EVALUABLE = 0
V_NEAR = 1
V_OCCLUDED = 2
V_BEHIND_WEAK = 3
V_IN_FRONT = 4
V_UNCERTAIN = 5

VERDICT_NAMES = {
    V_NOT_EVALUABLE: "not_evaluable",
    V_NEAR: "near",
    V_OCCLUDED: "occluded",
    V_BEHIND_WEAK: "behind_weak",
    V_IN_FRONT: "in_front",
    V_UNCERTAIN: "uncertain",
}

VERDICT_COLORS = {
    V_NOT_EVALUABLE: "#9e9e9e",  # grey
    V_NEAR: "#20c020",           # green
    V_OCCLUDED: "#e02020",       # red
    V_BEHIND_WEAK: "#2060e0",    # blue
    V_IN_FRONT: "#ff8c00",       # orange
    V_UNCERTAIN: "#9e9e9e",      # grey (same class as not-evaluable visually)
}

# Painter order: informative classes are drawn last so they survive overplotting
# by the dominant near/not-evaluable population.
VERDICT_DRAW_ORDER = (
    V_NOT_EVALUABLE,
    V_UNCERTAIN,
    V_NEAR,
    V_IN_FRONT,
    V_BEHIND_WEAK,
    V_OCCLUDED,
)

LEGEND_ORDER = (V_NEAR, V_OCCLUDED, V_BEHIND_WEAK, V_IN_FRONT, V_UNCERTAIN, V_NOT_EVALUABLE)

ROW_GT = "GT"
ROW_RENDER = "render"
ROW_OVERLAY = "render + E1"


class RenderPanelError(RuntimeError):
    """Fail-closed error for argument, artifact, or geometry problems."""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="render_lifecycle_qualitative.py",
        description=(
            "Render checkpoint-aligned before/during/after qualitative panels "
            "(GT | render | render+E1-overlay) for CSVL-VPL v2 lifecycle lanes. "
            "Requires a GPU; run as a Slurm job."
        ),
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        metavar="CHKPNT.pth",
        help="One or more trainer checkpoints (.pth). Panels are written per "
        "checkpoint; a contact sheet is added when more than one is given.",
    )
    parser.add_argument(
        "--lane-config",
        required=True,
        metavar="YAML",
        help="Lane yaml under configs/n3v used to train the checkpoints. "
        "Supplies gaussian_dim/time_duration/rot_4d/force_sh_3d, "
        "ModelParams (sh_degree, resolution, white_background, images, "
        "extension, frame_ratio, dataloader) and PipelineParams.",
    )
    parser.add_argument(
        "--source-path",
        required=True,
        metavar="DIR",
        help="N3V scene directory (holds transforms_train.json and images/).",
    )
    parser.add_argument(
        "--evidence-dir",
        required=True,
        metavar="DIR",
        help="Frozen consensus evidence directory written by "
        "scripts/build_evidence_consensus.py.",
    )
    parser.add_argument(
        "--windows",
        required=True,
        metavar="cam05:60,cam08:175",
        help="Comma-separated CAMERA:CENTER_FRAME windows. Cameras must be "
        "training cameras present in transforms_train.json.",
    )
    parser.add_argument(
        "--frames-around",
        type=int,
        default=10,
        metavar="N",
        help="Render center-N, center and center+N (default: 10). 0 renders "
        "only the center frame.",
    )
    parser.add_argument(
        "--max-overlay-points",
        type=int,
        default=30000,
        metavar="K",
        help="Upper bound on scattered primitives per overlay (default: 30000). "
        "The subsample is a fixed-seed permutation, so it is stable across "
        "runs and across checkpoints of equal bank size.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        metavar="DIR",
        help="Output directory for the panel PNGs, contact sheets and summary.json.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        metavar="DEV",
        help="Torch device (default: cuda). The rasterizer and GaussianModel "
        "hardcode CUDA allocations, so only cuda devices are accepted.",
    )
    parser.add_argument(
        "--transforms",
        default="transforms_train.json",
        metavar="NAME",
        help="Transforms file inside --source-path (default: transforms_train.json).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.0,
        metavar="S",
        help="Matplotlib marker size for the overlay scatter (default: 1.0, a "
        "1px ',' marker).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        metavar="DPI",
        help="Figure DPI for the written PNGs (default: 140).",
    )
    return parser


def expand_path(value: str) -> str:
    """Expand ``$WORK``-style variables and ``~`` the way the configs use them."""

    return os.path.abspath(os.path.expanduser(os.path.expandvars(str(value))))


def parse_windows(spec: str) -> list[tuple[str, int]]:
    """Parse ``"cam05:60,cam08:175"`` into ``[("cam05", 60), ("cam08", 175)]``."""

    windows: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for chunk in str(spec).split(","):
        token = chunk.strip()
        if not token:
            continue
        if token.count(":") != 1:
            raise RenderPanelError(
                f"malformed window {token!r}: expected exactly one ':' "
                "as in cam05:60"
            )
        camera, frame_text = token.split(":")
        camera = camera.strip()
        frame_text = frame_text.strip()
        if not camera:
            raise RenderPanelError(f"malformed window {token!r}: empty camera id")
        try:
            frame = int(frame_text)
        except ValueError as exc:
            raise RenderPanelError(
                f"malformed window {token!r}: center frame {frame_text!r} is not an integer"
            ) from exc
        if frame < 0:
            raise RenderPanelError(f"malformed window {token!r}: negative center frame")
        key = (camera, frame)
        if key in seen:
            raise RenderPanelError(f"duplicate window {camera}:{frame}")
        seen.add(key)
        windows.append(key)
    if not windows:
        raise RenderPanelError("--windows parsed to an empty window list")
    return windows


def unique_labels(paths: list[str]) -> list[str]:
    """Stable, collision-free short labels for the checkpoint files."""

    stems = [Path(p).stem for p in paths]
    if len(set(stems)) == len(stems):
        return stems
    labels = []
    for path, stem in zip(paths, stems):
        labels.append(f"{Path(path).parent.name}_{stem}")
    if len(set(labels)) != len(labels):
        labels = [f"{i:02d}_{label}" for i, label in enumerate(labels)]
    return labels


def validate_args(args: argparse.Namespace) -> dict:
    """Fail-closed validation of every path and numeric argument."""

    if not str(args.device).startswith("cuda"):
        raise RenderPanelError(
            f"--device {args.device!r} is not supported: gaussian_renderer.render "
            "and GaussianModel allocate on 'cuda' unconditionally"
        )
    if args.frames_around < 0:
        raise RenderPanelError("--frames-around must be >= 0")
    if args.max_overlay_points < 1:
        raise RenderPanelError("--max-overlay-points must be >= 1")
    if args.point_size <= 0.0:
        raise RenderPanelError("--point-size must be positive")
    if args.dpi < 30:
        raise RenderPanelError("--dpi must be >= 30")

    checkpoints = [expand_path(p) for p in args.checkpoints]
    if len(set(checkpoints)) != len(checkpoints):
        raise RenderPanelError("--checkpoints contains a duplicate path")
    for path in checkpoints:
        if not os.path.isfile(path):
            raise RenderPanelError(f"checkpoint not found: {path}")
        if not path.endswith(".pth"):
            raise RenderPanelError(f"checkpoint is not a .pth file: {path}")

    lane_config = expand_path(args.lane_config)
    if not os.path.isfile(lane_config):
        raise RenderPanelError(f"lane config not found: {lane_config}")

    source_path = expand_path(args.source_path)
    if not os.path.isdir(source_path):
        raise RenderPanelError(f"--source-path is not a directory: {source_path}")
    transforms_path = os.path.join(source_path, args.transforms)
    if not os.path.isfile(transforms_path):
        raise RenderPanelError(f"transforms file not found: {transforms_path}")

    evidence_dir = expand_path(args.evidence_dir)
    if not os.path.isdir(evidence_dir):
        raise RenderPanelError(f"--evidence-dir is not a directory: {evidence_dir}")

    out_dir = expand_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if not os.access(out_dir, os.W_OK):
        raise RenderPanelError(f"--out-dir is not writable: {out_dir}")

    windows = parse_windows(args.windows)

    return {
        "checkpoints": checkpoints,
        "checkpoint_labels": unique_labels(checkpoints),
        "lane_config": lane_config,
        "source_path": source_path,
        "transforms_path": transforms_path,
        "evidence_dir": evidence_dir,
        "out_dir": out_dir,
        "windows": windows,
    }


# ---------------------------------------------------------------------------
# Lane config
# ---------------------------------------------------------------------------


def load_lane_config(path: str) -> dict:
    """Read the lane yaml the way ``main.py`` merges it onto argparse defaults.

    ``main.py`` loads the yaml with OmegaConf and recursively ``setattr``s every
    key onto the parsed args, so unspecified keys keep the argparse defaults
    from ``arguments/__init__.py``. This reproduces that for the subset of keys
    the renderer needs.
    """

    from argparse import ArgumentParser

    from omegaconf import OmegaConf

    from arguments import PipelineParams

    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise RenderPanelError(f"lane config {path} did not parse to a mapping")

    model_section = cfg.get("ModelParams") or {}
    pipeline_section = cfg.get("PipelineParams") or {}
    if not isinstance(model_section, dict) or not isinstance(pipeline_section, dict):
        raise RenderPanelError(
            f"lane config {path} has a non-mapping ModelParams/PipelineParams section"
        )

    # PipelineParams defaults, straight from arguments/__init__.py.
    pipe_parser = ArgumentParser(add_help=False)
    pipe_group = PipelineParams(pipe_parser)
    pipe = pipe_group.extract(pipe_parser.parse_args([]))
    for key, value in pipeline_section.items():
        if not hasattr(pipe, key):
            raise RenderPanelError(
                f"lane config PipelineParams key {key!r} is unknown to arguments.PipelineParams"
            )
        setattr(pipe, key, value)

    resolution = int(model_section.get("resolution", -1))
    if resolution not in (1, 2, 3, 4, 8):
        raise RenderPanelError(
            f"lane config resolution {resolution} is outside the integer-divisor "
            "branch of utils.camera_utils.loadCam; this renderer only mirrors "
            "that branch"
        )

    time_duration = list(cfg.get("time_duration", [-0.5, 0.5]))
    if len(time_duration) != 2:
        raise RenderPanelError("lane config time_duration must have two entries")

    force_sh_3d = bool(cfg.get("force_sh_3d", False))
    eval_shfs_4d = bool(getattr(pipe, "eval_shfs_4d", False))

    return {
        "path": path,
        "name": Path(path).stem,
        "model": {
            # Mirrors scripts/run_phase0_census2.py::load_model's model_cfg keys.
            "sh_degree": int(model_section.get("sh_degree", 3)),
            "gaussian_dim": int(cfg.get("gaussian_dim", 3)),
            "time_duration": [float(v) for v in time_duration],
            "rot_4d": bool(cfg.get("rot_4d", False)),
            "force_sh_3d": force_sh_3d,
            # main.py: sh_degree_t = 2 if pipe.eval_shfs_4d else 0. Only gates
            # oneupSHdegree and the 4D SH channel count (unused when
            # force_sh_3d); active_sh_degree_t itself comes from the checkpoint.
            "sh_degree_t": 2 if eval_shfs_4d else 0,
        },
        "resolution": resolution,
        "white_background": bool(model_section.get("white_background", False)),
        "images": str(model_section.get("images", "images")),
        "extension": str(model_section.get("extension", ".png")),
        "frame_ratio": int(model_section.get("frame_ratio", 1)),
        "dataloader": bool(model_section.get("dataloader", False)),
        "data_device": str(model_section.get("data_device", "cuda")),
        "pipe": pipe,
    }


# ---------------------------------------------------------------------------
# Transforms / cameras (minimal mirror of the training loader)
# ---------------------------------------------------------------------------


def read_transforms(transforms_path: str) -> tuple[dict, dict]:
    """Return ``(contents, index)`` with ``index[(camera, frame)] = (idx, entry)``."""

    with open(transforms_path, "r", encoding="utf-8") as handle:
        contents = json.load(handle)
    frames = contents.get("frames")
    if not frames:
        raise RenderPanelError(f"{transforms_path} declares no frames")

    index: dict[tuple[str, int], tuple[int, dict]] = {}
    for idx, entry in enumerate(frames):
        stem = str(entry["file_path"]).rsplit("/", 1)[-1]
        if "_" not in stem:
            raise RenderPanelError(
                f"cannot split camera/frame out of file_path stem {stem!r}"
            )
        camera, _, frame_text = stem.rpartition("_")
        try:
            frame = int(frame_text)
        except ValueError as exc:
            raise RenderPanelError(
                f"cannot parse a frame index out of file_path stem {stem!r}"
            ) from exc
        key = (camera, frame)
        if key in index:
            raise RenderPanelError(f"duplicate transforms entry for {camera}:{frame}")
        index[key] = (idx, entry)
    return contents, index


def resolve_window_frames(
    windows: list[tuple[str, int]],
    frames_around: int,
    index: dict,
) -> list[dict]:
    """Fail-closed expansion of each window into its concrete frame list."""

    cameras = sorted({cam for cam, _ in index})
    resolved = []
    for camera, center in windows:
        if not any(cam == camera for cam, _ in index):
            raise RenderPanelError(
                f"camera {camera!r} is not a training camera in the transforms "
                f"file; available: {', '.join(cameras)}"
            )
        available = sorted(f for cam, f in index if cam == camera)
        wanted = [center - frames_around, center, center + frames_around]
        frames: list[int] = []
        for frame in wanted:
            if frame in frames:
                continue
            if (camera, frame) not in index:
                raise RenderPanelError(
                    f"window {camera}:{center} with --frames-around "
                    f"{frames_around} needs frame {frame}, which the transforms "
                    f"file does not have for {camera} "
                    f"(available {available[0]}..{available[-1]})"
                )
            frames.append(frame)
        resolved.append({"camera": camera, "center_frame": center, "frames": frames})
    return resolved


def build_camera(
    entry: dict,
    entry_index: int,
    uid: int,
    contents: dict,
    lane: dict,
    source_path: str,
    image_size: tuple[int, int],
    device: str,
):
    """Rebuild one training ``Camera`` for a transforms entry.

    Mirrors ``readCamerasFromTransforms`` (the ``fl_x`` intrinsics branch) and
    ``loadCam`` (the integer-divisor resolution branch, ``meta_only`` path).
    """

    import numpy as np

    from scene.cameras import Camera

    # -- readCamerasFromTransforms ------------------------------------------
    timestamp = entry.get("time", 0.0)
    frame_ratio = lane["frame_ratio"]
    if frame_ratio > 1:
        timestamp = timestamp / frame_ratio
    time_duration = lane["model"]["time_duration"]
    if "time" in entry and (timestamp < time_duration[0] or timestamp > time_duration[1]):
        raise RenderPanelError(
            f"timestamp {timestamp} for {entry['file_path']} falls outside the "
            f"lane time_duration {time_duration}; the trainer would have dropped "
            "this camera"
        )

    cam_name = os.path.join(source_path, entry["file_path"] + lane["extension"])
    c2w = np.array(entry["transform_matrix"])
    c2w[:3, 1:3] *= -1  # OpenGL/Blender -> COLMAP/OpenCV axes
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3, :3])  # stored transposed for the glm CUDA code
    T = w2c[:3, 3]

    image_path = cam_name
    image_name = Path(cam_name).stem
    width, height = image_size

    far = 100
    if "Birthday" in image_path or "Painter" in image_path or "Train" in image_path:
        far = 300

    if all(k in entry for k in ("fl_x", "fl_y", "cx", "cy")):
        source = entry
    elif all(k in contents for k in ("fl_x", "fl_y", "cx", "cy")):
        source = contents
    else:
        raise RenderPanelError(
            "transforms file has no fl_x/fl_y/cx/cy either per frame or at the "
            "top level; this renderer only mirrors the pinhole-intrinsics branch"
        )
    fov_x = fov_y = -1.0
    fl_x = float(source["fl_x"])
    fl_y = float(source["fl_y"])
    cx = float(source["cx"])
    cy = float(source["cy"])

    # -- loadCam (resolution_scale = 1.0, integer divisor branch) -----------
    resolution_scale = 1.0
    scale = resolution_scale * lane["resolution"]
    resolution = (
        round(width / (resolution_scale * lane["resolution"])),
        round(height / (resolution_scale * lane["resolution"])),
    )
    cx = cx / scale
    cy = cy / scale
    fl_x = fl_x / scale
    fl_y = fl_y / scale

    camera = Camera(
        colmap_id=entry_index,
        R=R,
        T=T,
        FoVx=fov_x,
        FoVy=fov_y,
        image=np.empty(0),
        gt_alpha_mask=None,
        image_name=image_name,
        uid=uid,
        data_device=device,
        timestamp=timestamp,
        cx=cx,
        cy=cy,
        fl_x=fl_x,
        fl_y=fl_y,
        depth=None,
        resolution=resolution,
        image_path=image_path,
        meta_only=True,
        cxr=0.0,
        cyr=0.0,
        far=far,
    )
    return camera


def probe_image_size(path: str) -> tuple[int, int]:
    """Original ``(width, height)``, as ``imagesize.get`` gives the trainer."""

    from PIL import Image

    with Image.open(path) as image:
        return int(image.size[0]), int(image.size[1])


def load_gt_image(path: str, resolution: tuple[int, int]):
    """GT frame downscaled exactly like ``utils.general_utils.PILtoTorch``."""

    import numpy as np
    from PIL import Image

    with Image.open(path) as src:
        image = src.convert("RGB").resize(resolution)
        return np.asarray(image, dtype=np.uint8).copy()


# ---------------------------------------------------------------------------
# Model / render
# ---------------------------------------------------------------------------


def load_gaussians(checkpoint_path: str, model_cfg: dict):
    """Restore a GaussianModel the way ``run_phase0_census2.load_model`` does."""

    import torch

    from scene.gaussian_model import GaussianModel

    gaussians = GaussianModel(
        sh_degree=model_cfg["sh_degree"],
        gaussian_dim=model_cfg["gaussian_dim"],
        time_duration=list(model_cfg["time_duration"]),
        rot_4d=model_cfg["rot_4d"],
        force_sh_3d=model_cfg["force_sh_3d"],
        sh_degree_t=model_cfg["sh_degree_t"],
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_params, checkpoint_iteration = payload
    gaussians.restore(model_params, None)
    static_count = (
        int(gaussians.static_xyz.shape[0]) if gaussians.static_xyz is not None else 0
    )
    if static_count != 0:
        # The census aborts here; a renderer can still draw hard-static points
        # via the non-soft-routing branch of gaussian_renderer.render, so this
        # is reported rather than fatal.
        print(
            f"[warn] {checkpoint_path}: {static_count} hard-static points present; "
            "they are rendered through the hard-static branch",
            flush=True,
        )
    return gaussians, int(checkpoint_iteration)


def render_view(camera, gaussians, pipe, background):
    """One ``gaussian_renderer.render`` call, returned as HxWx3 uint8."""

    import numpy as np
    import torch

    from gaussian_renderer import render

    with torch.no_grad():
        out = render(camera.cuda(), gaussians, pipe, background)
        image = out["render"].clamp(0.0, 1.0).detach()
        array = (image.permute(1, 2, 0).cpu().numpy() * 255.0).round()
    return array.astype(np.uint8)


# ---------------------------------------------------------------------------
# Evidence overlay
# ---------------------------------------------------------------------------


def _verify_verdict_codes() -> None:
    """Fail closed if the local verdict constants drift from the runtime."""

    from depth_visibility import evidence_runtime as er

    expected = {
        V_NOT_EVALUABLE: er.VERDICT_NOT_EVALUABLE,
        V_NEAR: er.VERDICT_NEAR,
        V_OCCLUDED: er.VERDICT_OCCLUDED,
        V_BEHIND_WEAK: er.VERDICT_BEHIND_WEAK,
        V_IN_FRONT: er.VERDICT_IN_FRONT,
        V_UNCERTAIN: er.VERDICT_UNCERTAIN,
    }
    for local, remote in expected.items():
        if local != remote:
            raise RenderPanelError(
                "verdict codes in render_lifecycle_qualitative.py disagree with "
                f"depth_visibility.evidence_runtime ({local} != {remote})"
            )
    if er.NUM_VERDICTS != len(VERDICT_NAMES):
        raise RenderPanelError(
            "verdict count drift between the renderer and evidence_runtime"
        )


def load_evidence_geometry(evidence_dir: str):
    """Per-camera ``(intrinsics, w2c)`` f64 arrays from the consensus artifact."""

    import numpy as np

    from depth_visibility.evidence_runtime import GEOMETRY_FILES

    intrinsics = np.load(os.path.join(evidence_dir, GEOMETRY_FILES["intrinsics"]))
    w2c = np.load(os.path.join(evidence_dir, GEOMETRY_FILES["w2c"]))
    return np.asarray(intrinsics, dtype=np.float64), np.asarray(w2c, dtype=np.float64)


def compute_overlay(
    runtime,
    geometry,
    xyz,
    camera_id: str,
    frame: int,
    render_wh: tuple[int, int],
    max_points: int,
):
    """Verdicts for the whole bank plus a deterministic drawable subsample.

    Returns ``(counts, draw)`` where ``counts`` maps verdict name -> count over
    the *full* bank, and ``draw`` maps verdict code -> ``(x, y)`` float32 arrays
    in render-pixel coordinates. Projection uses the consensus intrinsics/w2c
    (so the drawn position and the looked-up verdict come from one geometry)
    scaled from the consensus resolution to the render resolution.
    """

    import numpy as np
    import torch

    intrinsics, w2c = geometry
    camera_slot = runtime.camera_index[camera_id]
    verdicts = runtime.verdicts(xyz, camera_id, frame)
    counts_t = torch.bincount(
        verdicts.to(torch.int64), minlength=len(VERDICT_NAMES)
    ).cpu()
    counts = {VERDICT_NAMES[code]: int(counts_t[code]) for code in VERDICT_NAMES}

    total = int(xyz.shape[0])
    if total == 0:
        return counts, {}

    if total > max_points:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(SUBSAMPLE_SEED)
        picked = torch.randperm(total, generator=generator)[:max_points]
        picked = torch.sort(picked).values.to(xyz.device)
    else:
        picked = torch.arange(total, device=xyz.device)

    points = xyz.index_select(0, picked).detach().to(torch.float64)
    codes = verdicts.index_select(0, picked).to(torch.int64).cpu().numpy()

    rot = torch.as_tensor(
        w2c[camera_slot, :3, :3], dtype=torch.float64, device=points.device
    )
    trans = torch.as_tensor(
        w2c[camera_slot, :3, 3], dtype=torch.float64, device=points.device
    )
    fx = float(intrinsics[camera_slot, 0, 0])
    fy = float(intrinsics[camera_slot, 1, 1])
    cx = float(intrinsics[camera_slot, 0, 2])
    cy = float(intrinsics[camera_slot, 1, 2])

    cam = points @ rot.transpose(0, 1) + trans
    z = cam[:, 2]
    safe_z = torch.where(z.abs() > 1e-12, z, torch.full_like(z, 1e-12))
    u = fx * cam[:, 0] / safe_z + cx
    v = fy * cam[:, 1] / safe_z + cy

    render_w, render_h = render_wh
    scale_x = float(render_w) / float(runtime.width)
    scale_y = float(render_h) / float(runtime.height)
    u = u * scale_x
    v = v * scale_y

    keep = (
        (z > runtime.near_clip)
        & (u >= 0.0)
        & (u < float(render_w))
        & (v >= 0.0)
        & (v < float(render_h))
    )
    keep_np = keep.cpu().numpy()
    u_np = u.to(torch.float32).cpu().numpy()[keep_np]
    v_np = v.to(torch.float32).cpu().numpy()[keep_np]
    codes = codes[keep_np]

    draw: dict[int, tuple] = {}
    for code in VERDICT_DRAW_ORDER:
        selection = codes == code
        if not np.any(selection):
            continue
        draw[code] = (u_np[selection].copy(), v_np[selection].copy())
    return counts, draw


CAPTION_ABBREV = {
    V_NEAR: "near",
    V_OCCLUDED: "occ",
    V_BEHIND_WEAK: "weak",
    V_IN_FRONT: "front",
    V_UNCERTAIN: "unc",
    V_NOT_EVALUABLE: "n/e",
}


def overlay_caption(counts: dict, total: int) -> str:
    """Compact two-line per-frame verdict breakdown over the full bank."""

    if total <= 0:
        return "empty bank"

    def part(code: int) -> str:
        share = 100.0 * counts.get(VERDICT_NAMES[code], 0) / total
        return f"{CAPTION_ABBREV[code]} {share:.1f}%"

    head = "  ".join(part(code) for code in LEGEND_ORDER[:3])
    tail = "  ".join(part(code) for code in LEGEND_ORDER[3:])
    return f"N={total}  {head}\n{tail}"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _init_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _legend_handles():
    from matplotlib.lines import Line2D

    handles = []
    for code in LEGEND_ORDER:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markersize=6,
                markerfacecolor=VERDICT_COLORS[code],
                markeredgecolor="none",
                label=VERDICT_NAMES[code],
            )
        )
    return handles


def _scatter_overlay(ax, draw: dict, point_size: float) -> None:
    for code in VERDICT_DRAW_ORDER:
        if code not in draw:
            continue
        x, y = draw[code]
        ax.scatter(
            x,
            y,
            s=point_size,
            marker=",",
            linewidths=0.0,
            c=VERDICT_COLORS[code],
        )


def _blank_axes(ax, label: str | None = None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#555555")
    if label is not None:
        ax.set_ylabel(label, fontsize=9)


def write_panel(
    plt,
    out_path: str,
    title: str,
    frames: list[int],
    gt_images: dict,
    renders: dict,
    overlays: dict,
    captions: dict,
    evidence_note: str | None,
    point_size: float,
    dpi: int,
) -> None:
    """Three-row (GT | render | render+overlay) panel for one checkpoint/window."""

    ncols = len(frames)
    sample = renders[frames[0]]
    tile_h, tile_w = sample.shape[0], sample.shape[1]
    tile_in_w = 3.6
    tile_in_h = tile_in_w * tile_h / tile_w
    fig, axes = plt.subplots(
        3,
        ncols,
        figsize=(ncols * tile_in_w + 0.9, 3 * tile_in_h + 1.5),
        squeeze=False,
    )
    for col, frame in enumerate(frames):
        axes[0][col].imshow(gt_images[frame], interpolation="nearest")
        axes[0][col].set_title(f"frame {frame}", fontsize=10)
        _blank_axes(axes[0][col], ROW_GT if col == 0 else None)

        axes[1][col].imshow(renders[frame], interpolation="nearest")
        _blank_axes(axes[1][col], ROW_RENDER if col == 0 else None)

        axes[2][col].imshow(renders[frame], interpolation="nearest")
        if evidence_note is None:
            _scatter_overlay(axes[2][col], overlays.get(frame, {}), point_size)
        else:
            axes[2][col].text(
                0.5,
                0.5,
                evidence_note,
                transform=axes[2][col].transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#e02020",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#e02020"},
            )
        # imshow pixel (col, row) centres sit at integer data coordinates, which
        # is exactly the convention of the projected u/v, so pin the limits to
        # the imshow extent and stop the scatter from autoscaling them.
        axes[2][col].set_xlim(-0.5, tile_w - 0.5)
        axes[2][col].set_ylim(tile_h - 0.5, -0.5)
        _blank_axes(axes[2][col], ROW_OVERLAY if col == 0 else None)
        if evidence_note is None:
            axes[2][col].set_xlabel(captions.get(frame, ""), fontsize=6)

    fig.suptitle(title, fontsize=11)
    if evidence_note is None:
        fig.legend(
            handles=_legend_handles(),
            loc="lower center",
            ncol=len(LEGEND_ORDER),
            frameon=False,
            fontsize=8,
        )
    bottom = 0.055 if evidence_note is None else 0.005
    fig.tight_layout(rect=(0.0, bottom, 1.0, 0.965), h_pad=0.8, w_pad=0.5)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def write_contact_sheet(
    plt,
    out_path: str,
    title: str,
    frames: list[int],
    gt_images: dict,
    rows: list[dict],
    evidence_note: str | None,
    point_size: float,
    dpi: int,
) -> None:
    """GT row plus one render+overlay row per checkpoint, ordered by iteration."""

    ncols = len(frames)
    nrows = 1 + len(rows)
    sample = rows[0]["renders"][frames[0]]
    tile_h, tile_w = sample.shape[0], sample.shape[1]
    tile_in_w = 3.0
    tile_in_h = tile_in_w * tile_h / tile_w
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * tile_in_w + 0.9, nrows * tile_in_h + 1.5),
        squeeze=False,
    )
    for col, frame in enumerate(frames):
        axes[0][col].imshow(gt_images[frame], interpolation="nearest")
        axes[0][col].set_title(f"frame {frame}", fontsize=10)
        _blank_axes(axes[0][col], ROW_GT if col == 0 else None)

    for row_idx, row in enumerate(rows, start=1):
        label = f"iter {row['iteration']}\n{row['label']}"
        for col, frame in enumerate(frames):
            ax = axes[row_idx][col]
            ax.imshow(row["renders"][frame], interpolation="nearest")
            if evidence_note is None:
                _scatter_overlay(ax, row["overlays"].get(frame, {}), point_size)
            ax.set_xlim(-0.5, tile_w - 0.5)
            ax.set_ylim(tile_h - 0.5, -0.5)
            _blank_axes(ax, label if col == 0 else None)

    fig.suptitle(title, fontsize=11)
    if evidence_note is None:
        fig.legend(
            handles=_legend_handles(),
            loc="lower center",
            ncol=len(LEGEND_ORDER),
            frameon=False,
            fontsize=8,
        )
    bottom = 0.045 if evidence_note is None else 0.005
    fig.tight_layout(rect=(0.0, bottom, 1.0, 0.965), h_pad=0.8, w_pad=0.5)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved = validate_args(args)

    # Heavy imports start here so that --help works on a CPU login node.
    import torch

    from depth_visibility.evidence_runtime import EvidenceRuntime

    _verify_verdict_codes()

    if not torch.cuda.is_available():
        raise RenderPanelError(
            "no CUDA device is visible; gaussian_renderer.render requires a GPU. "
            "Submit this script as a Slurm GPU job."
        )

    lane = load_lane_config(resolved["lane_config"])
    contents, index = read_transforms(resolved["transforms_path"])
    windows = resolve_window_frames(
        resolved["windows"], args.frames_around, index
    )

    runtime = EvidenceRuntime(
        resolved["evidence_dir"], device=args.device, mode="valid"
    )
    geometry = load_evidence_geometry(resolved["evidence_dir"])
    print(f"[info] evidence: {runtime}", flush=True)

    background = torch.tensor(
        [1, 1, 1] if lane["white_background"] else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    pipe = lane["pipe"]

    # -- cameras, GT frames --------------------------------------------------
    cameras: dict[tuple[str, int], object] = {}
    gt_cache: dict[tuple[str, int], object] = {}
    uid = 0
    for window in windows:
        camera_id = window["camera"]
        for frame in window["frames"]:
            key = (camera_id, frame)
            if key in cameras:
                continue
            entry_index, entry = index[key]
            image_path = os.path.join(
                resolved["source_path"], entry["file_path"] + lane["extension"]
            )
            if not os.path.isfile(image_path):
                raise RenderPanelError(f"ground truth frame not found: {image_path}")
            image_size = probe_image_size(image_path)
            cameras[key] = build_camera(
                entry,
                entry_index,
                uid,
                contents,
                lane,
                resolved["source_path"],
                image_size,
                args.device,
            )
            uid += 1
            gt_cache[key] = load_gt_image(image_path, cameras[key].resolution)

    for window in windows:
        camera_id = window["camera"]
        note = None
        if not runtime.has_camera(camera_id):
            note = f"no E1 evidence for {camera_id}\noverlay skipped"
        else:
            missing = [
                f for f in window["frames"] if runtime.evidence_slot(f) < 0
            ]
            if missing:
                note = (
                    f"frames {missing} absent from the\nevidence artifact; "
                    "overlay skipped"
                )
        window["evidence_note"] = note
        window["has_evidence"] = note is None
        if note is not None:
            print(
                f"[warn] window {camera_id}:{window['center_frame']}: "
                f"{note.replace(chr(10), ' ')}",
                flush=True,
            )

    # -- per-checkpoint rendering -------------------------------------------
    plt = _init_matplotlib()
    out_dir = resolved["out_dir"]
    panels: list[dict] = []
    checkpoint_records: list[dict] = []
    # store[(label, camera, center)] = {"renders": {...}, "overlays": {...}}
    store: dict[tuple[str, str, int], dict] = {}

    for label, checkpoint in zip(resolved["checkpoint_labels"], resolved["checkpoints"]):
        print(f"[info] loading {label}: {checkpoint}", flush=True)
        gaussians, iteration = load_gaussians(checkpoint, lane["model"])
        num_primitives = int(gaussians._xyz.shape[0])
        checkpoint_records.append(
            {
                "label": label,
                "path": checkpoint,
                "iteration": iteration,
                "num_primitives": num_primitives,
            }
        )

        for window in windows:
            camera_id = window["camera"]
            center = window["center_frame"]
            frames = window["frames"]
            renders: dict[int, object] = {}
            overlays: dict[int, dict] = {}
            captions: dict[int, str] = {}
            verdict_counts: dict[str, dict] = {}

            for frame in frames:
                camera = cameras[(camera_id, frame)]
                renders[frame] = render_view(camera, gaussians, pipe, background)
                if window["has_evidence"]:
                    with torch.no_grad():
                        xyz = gaussians.get_dynamic_xyz(camera.timestamp).detach()
                    counts, draw = compute_overlay(
                        runtime,
                        geometry,
                        xyz,
                        camera_id,
                        frame,
                        (renders[frame].shape[1], renders[frame].shape[0]),
                        args.max_overlay_points,
                    )
                    overlays[frame] = draw
                    captions[frame] = overlay_caption(counts, int(xyz.shape[0]))
                    verdict_counts[str(frame)] = counts
                    del xyz

            filename = f"{label}_{camera_id}_{center}.png"
            out_path = os.path.join(out_dir, filename)
            title = (
                f"{label} (iter {iteration}, {num_primitives} primitives) | "
                f"{camera_id} @ frame {center} +-{args.frames_around} | "
                f"lane {lane['name']}"
            )
            write_panel(
                plt,
                out_path,
                title,
                frames,
                {f: gt_cache[(camera_id, f)] for f in frames},
                renders,
                overlays,
                captions,
                window["evidence_note"],
                args.point_size,
                args.dpi,
            )
            print(f"[info] wrote {out_path}", flush=True)
            panels.append(
                {
                    "file": filename,
                    "checkpoint_label": label,
                    "checkpoint_iteration": iteration,
                    "camera": camera_id,
                    "center_frame": center,
                    "frames": frames,
                    "has_evidence": window["has_evidence"],
                    "verdict_counts": verdict_counts,
                }
            )
            if len(resolved["checkpoints"]) > 1:
                store[(label, camera_id, center)] = {
                    "renders": renders,
                    "overlays": overlays,
                }

        del gaussians
        torch.cuda.empty_cache()

    # -- contact sheets ------------------------------------------------------
    contact_sheets: list[dict] = []
    if len(resolved["checkpoints"]) > 1:
        ordered = sorted(
            checkpoint_records, key=lambda rec: (rec["iteration"], rec["label"])
        )
        for window in windows:
            camera_id = window["camera"]
            center = window["center_frame"]
            frames = window["frames"]
            rows = []
            for record in ordered:
                cached = store[(record["label"], camera_id, center)]
                rows.append(
                    {
                        "label": record["label"],
                        "iteration": record["iteration"],
                        "renders": cached["renders"],
                        "overlays": cached["overlays"],
                    }
                )
            filename = f"contact_{camera_id}_{center}.png"
            out_path = os.path.join(out_dir, filename)
            title = (
                f"lifecycle over training | {camera_id} @ frame {center} "
                f"+-{args.frames_around} | lane {lane['name']} | "
                "rows: GT then render+E1 per checkpoint"
            )
            write_contact_sheet(
                plt,
                out_path,
                title,
                frames,
                {f: gt_cache[(camera_id, f)] for f in frames},
                rows,
                window["evidence_note"],
                args.point_size,
                args.dpi,
            )
            print(f"[info] wrote {out_path}", flush=True)
            contact_sheets.append(
                {
                    "file": filename,
                    "camera": camera_id,
                    "center_frame": center,
                    "frames": frames,
                    "checkpoint_order": [rec["label"] for rec in ordered],
                    "iterations": [rec["iteration"] for rec in ordered],
                }
            )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "lane_config": resolved["lane_config"],
        "lane_name": lane["name"],
        "source_path": resolved["source_path"],
        "transforms": args.transforms,
        "evidence_dir": resolved["evidence_dir"],
        "evidence_resolution": [runtime.height, runtime.width],
        "render_resolution": [
            int(cameras[next(iter(cameras))].image_height),
            int(cameras[next(iter(cameras))].image_width),
        ],
        "frames_around": args.frames_around,
        "max_overlay_points": args.max_overlay_points,
        "subsample_seed": SUBSAMPLE_SEED,
        "checkpoints": checkpoint_records,
        "windows": [
            {
                "camera": w["camera"],
                "center_frame": w["center_frame"],
                "frames": w["frames"],
                "has_evidence": w["has_evidence"],
            }
            for w in windows
        ],
        "panels": panels,
        "contact_sheets": contact_sheets,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, sort_keys=True)
    print(f"[info] wrote {summary_path}", flush=True)
    print(
        f"[done] {len(panels)} panels, {len(contact_sheets)} contact sheets in {out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RenderPanelError as error:
        print(f"[error] {error}", file=sys.stderr, flush=True)
        sys.exit(2)
