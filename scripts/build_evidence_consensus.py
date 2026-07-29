#!/usr/bin/env python3
"""Build the CSVL-VPL v2 training-time evidence consensus artifact.

Component 1 of the CSVL-VPL v2 pipeline. Freezes the census-v2 P01 DA3
consensus depth of one scene into a directory of uncompressed ``.npy`` arrays
that ``depth_visibility.evidence_runtime.EvidenceRuntime`` can bulk-load onto
the training device (or memory-map for smoke runs):

    d.npy           fp16  [C, T, H, W]   consensus median depth (NaN if invalid)
    sigma.npy       fp16  [C, T, H, W]   1.4826 * MAD robust sigma
    valid.npy       uint8 [C, T, H, W]   consensus validity mask
    intrinsics.npy  f64   [C, 3, 3]      per-camera processed intrinsics
    w2c.npy         f64   [C, 4, 4]      per-camera aligned world-to-camera
    meta.json                            camera/frame order and provenance

The per-map consensus is *not* reimplemented: ``build_consensus_cache`` from
``scripts/run_phase0_census2.py`` is imported by path and invoked one frame at
a time, so the artifact is bit-identical to the census-v2 evidence while peak
memory stays at a single frame. Camera geometry is taken from the first frame
that carries each camera and every later frame must agree to better than
``--geometry-tolerance``; the rig is static, so any drift means the P01 export
changed under us and the build fails closed.

No RGB, annotations, evaluator masks, checkpoints, or W&B. Run under Slurm on
a CPU node; the orchestrator submits.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from depth_visibility import primitive_census as census  # noqa: E402
from depth_visibility.canonical import sha256_file  # noqa: E402
from depth_visibility.errors import ArtifactError, ContractError, SchemaError  # noqa: E402
from depth_visibility.evidence_runtime import (  # noqa: E402
    CONSENSUS_SCHEMA_VERSION,
    open_consensus_maps,
    write_consensus_geometry,
    write_consensus_meta,
)

CONFIG_SCHEMA = "phase0-census2-config-v1"
REQUIRED_CONFIG_KEYS = (
    "p01_root",
    "frames",
    "excluded_cameras",
    "min_members",
    "confidence_percentile",
    "min_valid_pixel_fraction",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="census-v2 config JSON (schema phase0-census2-config-v1)")
    parser.add_argument("--output-dir", required=True,
                        help="destination consensus directory (created)")
    parser.add_argument("--frame-limit", type=int, default=None,
                        help="smoke only: keep the first N frames; marks the "
                             "artifact non-scientific")
    parser.add_argument("--geometry-tolerance", type=float, default=1e-4,
                        help="maximum permitted per-frame camera geometry drift")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow writing into a non-empty output directory")
    return parser.parse_args(argv)


def load_builder_config(path: str) -> dict[str, Any]:
    """Load the census-v2 config, requiring only the keys this builder uses."""

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("schema_version") != CONFIG_SCHEMA:
        raise SchemaError(f"unexpected census config schema: {raw.get('schema_version')!r}")
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in raw]
    if missing:
        raise SchemaError(f"census config is missing required keys: {missing}")
    config = dict(raw)
    config["p01_root"] = census.expand_work(config["p01_root"])
    return config


def load_census2_module():
    """Import ``scripts/run_phase0_census2.py`` by path (scripts is not a package)."""

    path = REPO_ROOT / "scripts" / "run_phase0_census2.py"
    if not path.is_file():
        raise ArtifactError(f"census-v2 runner not found: {path}")
    spec = importlib.util.spec_from_file_location("_run_phase0_census2", path)
    if spec is None or spec.loader is None:
        raise ArtifactError(f"cannot import census-v2 runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_consensus_cache"):
        raise ArtifactError("census-v2 runner does not expose build_consensus_cache")
    return module


def manifest_map_shape(manifest: dict[str, Any]) -> tuple[int, int]:
    """Height and width shared by every P01 group, or fail closed."""

    shapes = set()
    for group in manifest["groups"]:
        shape = group.get("processed_depth_shape")
        if shape is None or len(shape) != 3:
            raise SchemaError("P01 group is missing a (M, H, W) processed_depth_shape")
        shapes.add((int(shape[1]), int(shape[2])))
    if len(shapes) != 1:
        raise ContractError(f"P01 groups disagree on the map shape: {sorted(shapes)}")
    return shapes.pop()


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    config = load_builder_config(args.config)
    census2 = load_census2_module()

    p01_root = config["p01_root"]
    manifest_path = os.path.join(p01_root, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    index, all_cameras = census.build_p01_index(manifest, p01_root)
    excluded = sorted(set(config["excluded_cameras"]))
    cameras = [camera for camera in all_cameras if camera not in set(excluded)]
    if not cameras:
        raise ContractError("every P01 camera is excluded by the config")

    frame_cfg = config["frames"]
    frames = list(range(int(frame_cfg["start"]), int(frame_cfg["end"]) + 1))
    scientific = True
    if args.frame_limit is not None:
        if args.frame_limit <= 0:
            raise ContractError("--frame-limit must be positive")
        frames = frames[: args.frame_limit]
        scientific = False
    if not frames:
        raise ContractError("no frames selected")

    height, width = manifest_map_shape(manifest)
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise ArtifactError(
            f"output directory is not empty (use --overwrite): {out_dir}"
        )

    num_cameras = len(cameras)
    num_frames = len(frames)
    print(
        f"[evidence] scene={config.get('scene')} cameras={num_cameras} "
        f"(excluded {excluded}) frames={num_frames} hw=({height}, {width}) "
        f"scientific={scientific}",
        flush=True,
    )

    maps = open_consensus_maps(
        out_dir,
        num_cameras=num_cameras,
        num_frames=num_frames,
        height=height,
        width=width,
    )
    intrinsics = np.zeros((num_cameras, 3, 3), dtype=np.float64)
    w2c = np.zeros((num_cameras, 4, 4), dtype=np.float64)
    geometry_seen = np.zeros(num_cameras, dtype=bool)
    geometry_drift = 0.0

    per_camera = {
        camera: {
            "maps": 0,
            "passed": 0,
            "missing_frames": 0,
            "members_min": None,
            "members_max": None,
            "valid_fraction_sum": 0.0,
        }
        for camera in cameras
    }
    map_totals = {"total": 0, "passed": 0}
    min_members = int(config["min_members"])
    min_valid_fraction = float(config["min_valid_pixel_fraction"])

    for t_index, frame in enumerate(frames):
        cache_d, cache_sigma, cache_valid, geometry, stats = census2.build_consensus_cache(
            index, cameras, [frame], config
        )
        map_totals["total"] += int(stats["total"])
        map_totals["passed"] += int(stats["passed"])
        derived_passed = 0
        for col, camera in enumerate(cameras):
            key = (col, frame)
            record = per_camera[camera]
            if key not in cache_d:
                record["missing_frames"] += 1
                continue
            valid = np.asarray(cache_valid[key], dtype=bool)
            maps["d"][col, t_index] = cache_d[key]
            maps["sigma"][col, t_index] = cache_sigma[key]
            maps["valid"][col, t_index] = valid.astype(np.uint8)

            members = len(index[frame][camera])
            valid_fraction = float(valid.mean())
            record["maps"] += 1
            record["valid_fraction_sum"] += valid_fraction
            record["members_min"] = (
                members if record["members_min"] is None
                else min(record["members_min"], members)
            )
            record["members_max"] = (
                members if record["members_max"] is None
                else max(record["members_max"], members)
            )
            if members >= min_members and valid_fraction >= min_valid_fraction:
                record["passed"] += 1
                derived_passed += 1

            frame_w2c, frame_k = geometry[key]
            frame_w2c = np.asarray(frame_w2c, dtype=np.float64)
            frame_k = np.asarray(frame_k, dtype=np.float64)
            if not geometry_seen[col]:
                w2c[col] = frame_w2c
                intrinsics[col] = frame_k
                geometry_seen[col] = True
            else:
                drift = max(
                    float(np.abs(frame_w2c - w2c[col]).max()),
                    float(np.abs(frame_k - intrinsics[col]).max()),
                )
                geometry_drift = max(geometry_drift, drift)
                if drift > args.geometry_tolerance:
                    raise ContractError(
                        f"camera {camera} geometry drifts by {drift:.3e} at frame "
                        f"{frame} (tolerance {args.geometry_tolerance:.1e})"
                    )
        if derived_passed != int(stats["passed"]):
            raise ContractError(
                f"per-camera map pass derivation disagrees with build_consensus_cache "
                f"at frame {frame}: {derived_passed} vs {int(stats['passed'])}"
            )
        if t_index % 10 == 0 or t_index == num_frames - 1:
            elapsed = time.time() - started
            print(
                f"[evidence] frame {frame} ({t_index + 1}/{num_frames}) "
                f"maps={map_totals['total']} passed={map_totals['passed']} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    missing_geometry = [
        cameras[col] for col in range(num_cameras) if not geometry_seen[col]
    ]
    if missing_geometry:
        raise ContractError(f"cameras without any P01 geometry: {missing_geometry}")

    for handle in maps.values():
        handle.flush()
    del maps

    write_consensus_geometry(out_dir, intrinsics, w2c)

    pass_fraction = (
        map_totals["passed"] / map_totals["total"] if map_totals["total"] else 0.0
    )
    per_camera_meta = {}
    for camera, record in per_camera.items():
        maps_written = record["maps"]
        per_camera_meta[camera] = {
            "maps": maps_written,
            "passed": record["passed"],
            "pass_fraction": (record["passed"] / maps_written) if maps_written else 0.0,
            "missing_frames": record["missing_frames"],
            "members_min": record["members_min"],
            "members_max": record["members_max"],
            "mean_valid_fraction": (
                record["valid_fraction_sum"] / maps_written if maps_written else 0.0
            ),
        }

    with open(args.config, "r", encoding="utf-8") as handle:
        config_echo = json.load(handle)

    meta = {
        "schema_version": CONSENSUS_SCHEMA_VERSION,
        "scene": config.get("scene"),
        "scientific": scientific,
        "cameras": cameras,
        "excluded_cameras": excluded,
        "frames": frames,
        "fps": float(frame_cfg.get("fps", 30.0)),
        "height": height,
        "width": width,
        "num_cameras": num_cameras,
        "num_frames": num_frames,
        "p01_root": p01_root,
        "p01_manifest_sha256": sha256_file(manifest_path),
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "config": config_echo,
        "consensus": {
            "min_members": min_members,
            "confidence_percentile": float(config["confidence_percentile"]),
            "min_valid_pixel_fraction": min_valid_fraction,
        },
        "map_stats": {**map_totals, "pass_fraction": pass_fraction},
        "per_camera_map_stats": per_camera_meta,
        "geometry_drift_max": geometry_drift,
        "geometry_tolerance": float(args.geometry_tolerance),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wall_seconds": round(time.time() - started, 1),
    }
    write_consensus_meta(out_dir, meta)

    print(
        f"[evidence] wrote {out_dir} maps={map_totals['total']} "
        f"pass_fraction={pass_fraction:.4f} geometry_drift_max={geometry_drift:.3e} "
        f"wall={meta['wall_seconds']}s",
        flush=True,
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    build(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
