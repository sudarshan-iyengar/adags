#!/usr/bin/env python3
"""Phase 0 primitive-centric evidence-opportunity census runner (CSVL-VPL v2).

Preregistered: research-wiki/operations/phase0-census-preregistration.md and
configs/depth_visibility/phase0_census_v1.json. Reads the sealed P01 DA3
sidecar and one route0 checkpoint; writes census-v1.json and
transitions-sample.json under the configured output root. No RGB,
annotations, evaluator masks, R009 crop pixels, or W&B are read or written.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.canonical import canonical_json_bytes, sha256_file  # noqa: E402
from depth_visibility import primitive_census as census  # noqa: E402

VOLATILE_KEYS = ("timestamp_utc", "slurm_job_id", "absolute_output_root", "wall_seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--frame-limit", type=int, default=None,
                        help="Smoke-test only: restrict to the first K frames. "
                             "The output is marked non-scientific when set.")
    parser.add_argument("--output-root", default=None,
                        help="Override the configured output root (smoke tests).")
    return parser.parse_args()


def canonical_content_hash(payload: dict) -> str:
    import hashlib

    trimmed = {k: v for k, v in payload.items() if k not in VOLATILE_KEYS}
    return hashlib.sha256(canonical_json_bytes(trimmed)).hexdigest()


def load_model(config: dict):
    import torch

    from scene.gaussian_model import GaussianModel

    model_cfg = config["model"]
    gaussians = GaussianModel(
        sh_degree=model_cfg["sh_degree"],
        gaussian_dim=model_cfg["gaussian_dim"],
        time_duration=list(model_cfg["time_duration"]),
        rot_4d=model_cfg["rot_4d"],
        force_sh_3d=model_cfg["force_sh_3d"],
        sh_degree_t=model_cfg["sh_degree_t"],
    )
    payload = torch.load(config["checkpoint_path"], map_location="cpu", weights_only=False)
    model_params, checkpoint_iteration = payload
    gaussians.restore(model_params, None)
    static_count = int(gaussians.static_xyz.shape[0]) if gaussians.static_xyz is not None else 0
    if static_count != 0:
        raise RuntimeError(f"expected zero hard-static points, found {static_count}")
    return gaussians, int(checkpoint_iteration)


def precompute_primitives(gaussians, config: dict, frames: list[int]):
    import torch

    fps = float(config["frames"]["fps"])
    opacity_floor = float(config["opacity_floor"])
    marginal_threshold = float(config["marginal_t_threshold"])

    with torch.no_grad():
        opacity = torch.sigmoid(gaussians._opacity).squeeze(-1)
        opacity_mask = (opacity >= opacity_floor).cpu().numpy()
        positions: list[np.ndarray] = []
        presence: list[np.ndarray] = []
        for frame in frames:
            timestamp = frame / fps
            xyz = gaussians.get_dynamic_xyz(timestamp).cpu().numpy().astype(np.float32)
            marginal = gaussians.get_marginal_t(timestamp).squeeze(-1).cpu().numpy()
            positions.append(xyz)
            presence.append((marginal >= marginal_threshold) & opacity_mask)
    return positions, presence, opacity_mask


def build_consensus_cache(index, cameras, frames, config):
    """Load P01 arrays once per group; return per (camera, frame) consensus maps."""
    min_members = int(config["min_members"])
    conf_pct = float(config["confidence_percentile"])
    min_valid_fraction = float(config["min_valid_pixel_fraction"])

    cam_pos = {cam: i for i, cam in enumerate(cameras)}
    cache_d: dict[tuple[int, int], np.ndarray] = {}
    cache_sigma: dict[tuple[int, int], np.ndarray] = {}
    cache_valid: dict[tuple[int, int], np.ndarray] = {}
    geometry: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    map_stats = {"total": 0, "passed": 0}
    per_map_records = []

    for frame in frames:
        by_camera = index.get(frame, {})
        group_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for camera_id, refs in by_camera.items():
            col = cam_pos[camera_id]
            depth_slices = []
            conf_slices = []
            first_geometry = None
            for ref in refs:
                if ref.depth_path not in group_arrays:
                    group_arrays[ref.depth_path] = (
                        np.load(ref.depth_path),
                        np.load(ref.confidence_path),
                        np.load(ref.aligned_w2c_path),
                        np.load(ref.intrinsics_path),
                    )
                depth_all, conf_all, w2c_all, k_all = group_arrays[ref.depth_path]
                depth_slices.append(depth_all[ref.member_index])
                conf_slices.append(conf_all[ref.member_index])
                if first_geometry is None:
                    first_geometry = (
                        w2c_all[ref.member_index].astype(np.float64),
                        k_all[ref.member_index].astype(np.float64),
                    )
            depth_stack = np.stack(depth_slices, axis=0)
            conf_stack = np.stack(conf_slices, axis=0)
            d_med, sigma, valid, stats = census.consensus_depth(
                depth_stack, conf_stack, min_members=min_members, confidence_percentile=conf_pct
            )
            key = (col, frame)
            cache_d[key] = d_med.astype(np.float16)
            cache_sigma[key] = sigma.astype(np.float16)
            cache_valid[key] = valid
            geometry[key] = first_geometry
            map_stats["total"] += 1
            passed = stats["members"] >= min_members and stats["valid_fraction"] >= min_valid_fraction
            if passed:
                map_stats["passed"] += 1
            per_map_records.append(
                {"camera": cameras[col], "frame": frame, **{k: stats[k] for k in ("members", "valid_fraction", "median_sigma")}}
            )
    return cache_d, cache_sigma, cache_valid, geometry, map_stats, per_map_records


def run_census_pass(positions, presence, cache_d, cache_sigma, cache_valid, geometry,
                    cameras, frames, config, *, frame_assignment=None, tau_rel=None,
                    with_relaxed=False, sample_cap=0):
    num_primitives = positions[0].shape[0]
    num_cameras = len(cameras)
    tau = float(tau_rel if tau_rel is not None else config["margin_tau_rel_primary"])
    kappa = float(config["margin_kappa_sigma"])
    near_clip = float(config["near_clip"])
    min_run = int(config["min_occlusion_run"])

    strict = census.RunTracker(num_primitives, num_cameras, min_run, relaxed=False,
                               sample_cap=sample_cap)
    relaxed = (
        census.RunTracker(num_primitives, num_cameras, min_run, relaxed=True)
        if with_relaxed else None
    )
    evaluable_total = 0
    oww_total = 0
    per_camera_evaluable = np.zeros(num_cameras, dtype=np.int64)
    per_camera_oww = np.zeros(num_cameras, dtype=np.int64)

    for t_index, frame in enumerate(frames):
        xyz = positions[t_index].astype(np.float64)
        present = presence[t_index]
        states = np.zeros((num_primitives, num_cameras), dtype=np.int8)
        for col in range(num_cameras):
            evidence_frame = frame
            if frame_assignment is not None:
                evidence_frame = frames[int(frame_assignment[col, t_index])]
            key = (col, evidence_frame)
            geo_key = (col, frame)
            if key not in cache_d or geo_key not in geometry:
                continue
            w2c, intrinsics = geometry[geo_key]
            d_map = cache_d[key].astype(np.float32)
            sigma_map = cache_sigma[key].astype(np.float32)
            valid_map = cache_valid[key]
            height, width = d_map.shape
            pixels, z, in_view = census.project_points(
                xyz, w2c, intrinsics, height, width, near_clip=near_clip
            )
            states[:, col] = census.classify_states(
                z, pixels, in_view, present, d_map, sigma_map, valid_map,
                tau_rel=tau, kappa=kappa,
            )
        oww = census.occluded_with_witness(states)
        evaluable = states != census.STATE_NOT_EVALUABLE
        evaluable_total += int(evaluable.sum())
        oww_total += int(oww.sum())
        per_camera_evaluable += evaluable.sum(axis=0)
        per_camera_oww += oww.sum(axis=0)
        strict.update(frame, states, oww)
        if relaxed is not None:
            relaxed.update(frame, states, oww)

    per_camera_fraction = {
        cameras[c]: (per_camera_oww[c] / per_camera_evaluable[c]) if per_camera_evaluable[c] else 0.0
        for c in range(num_cameras)
    }
    result = {
        "tau_rel": tau,
        "evaluable_tuples": int(evaluable_total),
        "occluded_with_witness_tuples": int(oww_total),
        "occluded_with_witness_fraction": (oww_total / evaluable_total) if evaluable_total else 0.0,
        "per_camera_occluded_fraction": {k: float(v) for k, v in per_camera_fraction.items()},
        "strict": strict.summary(cameras),
    }
    if relaxed is not None:
        result["relaxed"] = relaxed.summary(cameras)
    return result, strict


def run_consistency(cache_d, cache_sigma, cache_valid, geometry, cameras, frames, config):
    stride = int(config["consistency"]["frame_stride"])
    pixel_stride = int(config["consistency"]["pixel_stride"])
    pair_count = int(config["consistency"]["camera_pair_count"])
    tau = float(config["margin_tau_rel_primary"])
    kappa = float(config["margin_kappa_sigma"])
    near_clip = float(config["near_clip"])

    rng = np.random.default_rng(int(config["shuffle_seed"]))
    num_cameras = len(cameras)
    pairs = set()
    while len(pairs) < pair_count:
        a, b = rng.integers(0, num_cameras, size=2).tolist()
        if a != b:
            pairs.add((a, b))
    totals = {"evaluated": 0, "consistent": 0, "occluded": 0, "conflict": 0}
    for frame in frames[::stride]:
        for a, b in sorted(pairs):
            key_a, key_b = (a, frame), (b, frame)
            if key_a not in cache_d or key_b not in cache_d:
                continue
            w2c_a, k_a = geometry[key_a]
            w2c_b, k_b = geometry[key_b]
            outcome = census.cross_view_consistency(
                cache_d[key_a].astype(np.float32), cache_valid[key_a], w2c_a, k_a,
                cache_d[key_b].astype(np.float32), cache_sigma[key_b].astype(np.float32),
                cache_valid[key_b], w2c_b, k_b,
                pixel_stride=pixel_stride, tau_rel=tau, kappa=kappa, near_clip=near_clip,
            )
            for k in totals:
                totals[k] += outcome[k]
    return totals


def main() -> int:
    args = parse_args()
    started = time.time()
    config = census.load_census_config(args.config)
    if args.output_root:
        config["output_root"] = census.expand_work(args.output_root)

    p01_root = config["p01_root"]
    manifest_path = os.path.join(p01_root, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    index, cameras = census.build_p01_index(manifest, p01_root)

    frame_cfg = config["frames"]
    frames = list(range(int(frame_cfg["start"]), int(frame_cfg["end"]) + 1))
    scientific = True
    if args.frame_limit is not None:
        frames = frames[: args.frame_limit]
        scientific = False

    checkpoint_sha = sha256_file(config["checkpoint_path"])
    manifest_sha = sha256_file(manifest_path)
    config_sha = sha256_file(args.config)

    print(f"[census] cameras={len(cameras)} frames={len(frames)} scientific={scientific}", flush=True)
    gaussians, checkpoint_iteration = load_model(config)
    positions, presence, opacity_mask = precompute_primitives(gaussians, config, frames)
    num_primitives = positions[0].shape[0]
    print(f"[census] primitives={num_primitives} opacity_excluded={int((~opacity_mask).sum())} "
          f"checkpoint_iteration={checkpoint_iteration}", flush=True)

    cache = build_consensus_cache(index, cameras, frames, config)
    cache_d, cache_sigma, cache_valid, geometry, map_stats, per_map_records = cache
    map_pass_fraction = (map_stats["passed"] / map_stats["total"]) if map_stats["total"] else 0.0
    print(f"[census] consensus maps={map_stats['total']} pass_fraction={map_pass_fraction:.4f}", flush=True)

    primary, strict_tracker = run_census_pass(
        positions, presence, cache_d, cache_sigma, cache_valid, geometry, cameras, frames,
        config, with_relaxed=True, sample_cap=int(config["transitions_sample_cap"]),
    )
    print(f"[census] primary strict pairs={primary['strict']['completed_reveal_pairs']}", flush=True)

    assignment = census.shuffled_frame_assignment(
        len(frames), len(cameras), int(config["shuffle_seed"])
    )
    shuffle, _ = run_census_pass(
        positions, presence, cache_d, cache_sigma, cache_valid, geometry, cameras, frames,
        config, frame_assignment=assignment,
    )
    print(f"[census] shuffle strict pairs={shuffle['strict']['completed_reveal_pairs']}", flush=True)

    variants = {}
    for tau in config["margin_tau_rel_variants"]:
        variant, _ = run_census_pass(
            positions, presence, cache_d, cache_sigma, cache_valid, geometry, cameras, frames,
            config, tau_rel=tau,
        )
        variants[str(tau)] = {
            "occluded_with_witness_fraction": variant["occluded_with_witness_fraction"],
            "strict_pairs": variant["strict"]["completed_reveal_pairs"],
        }

    consistency = run_consistency(cache_d, cache_sigma, cache_valid, geometry, cameras, frames, config)

    r009 = {}
    for name, (lo, hi) in config.get("r009_descriptive_windows", {}).items():
        count = sum(
            v for k, v in primary["strict"]["completions_by_frame"].items() if lo <= int(k) <= hi
        )
        r009[name] = {"frame_range": [lo, hi], "completed_reveal_events": count}

    summary_for_floors = {
        "occluded_with_witness_fraction": primary["occluded_with_witness_fraction"],
        "strict": primary["strict"],
        "shuffle": shuffle["strict"],
        "per_camera_occluded_fraction": primary["per_camera_occluded_fraction"],
        "consistency": consistency,
        "consensus_maps": {"pass_fraction": map_pass_fraction},
    }
    floors = census.evaluate_floors(summary_for_floors, config["floors"])

    payload = {
        "schema_version": "phase0-census-v1",
        "scientific": scientific,
        "scene": config["scene"],
        "config_sha256": config_sha,
        "checkpoint_path": config["checkpoint_path"],
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_iteration": checkpoint_iteration,
        "p01_manifest_sha256": manifest_sha,
        "cameras": cameras,
        "frame_count": len(frames),
        "num_primitives": int(num_primitives),
        "opacity_excluded": int((~opacity_mask).sum()),
        "consensus_map_stats": {**map_stats, "pass_fraction": map_pass_fraction},
        "primary": primary,
        "shuffle_control": shuffle,
        "margin_variants_descriptive": variants,
        "cross_view_consistency": consistency,
        "r009_descriptive_overlap": r009,
        "floors": floors,
        "phase0_go": floors["phase0_go"] if scientific else None,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "absolute_output_root": config["output_root"],
        "wall_seconds": round(time.time() - started, 1),
    }
    payload["canonical_scientific_sha256"] = canonical_content_hash(payload)

    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "census-v1.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    with open(output_root / "transitions-sample.json", "w", encoding="utf-8") as handle:
        json.dump({"samples": strict_tracker.samples, "cameras": cameras}, handle)
    with open(output_root / "consensus-map-records.json", "w", encoding="utf-8") as handle:
        json.dump(per_map_records, handle)

    print(f"[census] wrote {output_root / 'census-v1.json'}", flush=True)
    print(f"[census] phase0_go={payload['phase0_go']} wall={payload['wall_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
