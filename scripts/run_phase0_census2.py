#!/usr/bin/env python3
"""Phase 0 census-v2 runner: certification vs carrier discrimination.

Preregistered: research-wiki/operations/phase0-census2-preregistration.md and
configs/depth_visibility/phase0_census2_v1.json. Runs the anchored-hysteresis
certified-reveal rule in a 2x2 (positions x evidence) decomposition on the
primary checkpoint, moving-position cells on comparison checkpoints, with
primitive-quality stratification. No RGB, annotations, evaluator masks, R009
crop pixels, or W&B.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import traceback
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
                        help="Smoke only: restrict frames; output marked non-scientific.")
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_census2_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != "phase0-census2-config-v1":
        raise ValueError(f"unexpected schema: {config.get('schema_version')!r}")
    config["p01_root"] = census.expand_work(config["p01_root"])
    config["output_root"] = census.expand_work(config["output_root"])
    for entry in config["checkpoints"]:
        entry["path"] = census.expand_work(entry["path"])
    return config


def canonical_content_hash(payload: dict) -> str:
    trimmed = {k: v for k, v in payload.items() if k not in VOLATILE_KEYS}
    return hashlib.sha256(canonical_json_bytes(trimmed)).hexdigest()


def load_model(checkpoint_path: str, model_cfg: dict):
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
    static_count = int(gaussians.static_xyz.shape[0]) if gaussians.static_xyz is not None else 0
    if static_count != 0:
        raise RuntimeError(f"expected zero hard-static points, found {static_count}")
    return gaussians, int(checkpoint_iteration)


def precompute_positions(gaussians, config: dict, frames: list[int], *, frozen: bool):
    import torch

    fps = float(config["frames"]["fps"])
    opacity_floor = float(config["opacity_floor"])
    marginal_threshold = float(config["marginal_t_threshold"])
    with torch.no_grad():
        opacity = torch.sigmoid(gaussians._opacity).squeeze(-1)
        opacity_mask = (opacity >= opacity_floor).cpu().numpy()
        frozen_xyz = gaussians._xyz.detach().cpu().numpy().astype(np.float32)
        positions: list[np.ndarray] = []
        presence: list[np.ndarray] = []
        for frame in frames:
            timestamp = frame / fps
            if frozen:
                positions.append(frozen_xyz)
            else:
                positions.append(
                    gaussians.get_dynamic_xyz(timestamp).cpu().numpy().astype(np.float32)
                )
            marginal = gaussians.get_marginal_t(timestamp).squeeze(-1).cpu().numpy()
            presence.append((marginal >= marginal_threshold) & opacity_mask)
    return positions, presence, opacity_mask


def build_consensus_cache(index, cameras, frames, config):
    min_members = int(config["min_members"])
    conf_pct = float(config["confidence_percentile"])
    min_valid_fraction = float(config["min_valid_pixel_fraction"])
    cam_pos = {cam: i for i, cam in enumerate(cameras)}
    cache_d, cache_sigma, cache_valid, geometry = {}, {}, {}, {}
    map_stats = {"total": 0, "passed": 0}
    for frame in frames:
        by_camera = index.get(frame, {})
        group_arrays = {}
        for camera_id, refs in by_camera.items():
            if camera_id not in cam_pos:
                continue
            col = cam_pos[camera_id]
            depth_slices, conf_slices = [], []
            first_geometry = None
            for ref in refs:
                if ref.depth_path not in group_arrays:
                    group_arrays[ref.depth_path] = (
                        np.load(ref.depth_path), np.load(ref.confidence_path),
                        np.load(ref.aligned_w2c_path), np.load(ref.intrinsics_path),
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
                depth_stack, conf_stack, min_members=min_members,
                confidence_percentile=conf_pct,
            )
            key = (col, frame)
            cache_d[key] = d_med.astype(np.float16)
            cache_sigma[key] = sigma.astype(np.float16)
            cache_valid[key] = valid
            geometry[key] = first_geometry
            map_stats["total"] += 1
            if stats["members"] >= min_members and stats["valid_fraction"] >= min_valid_fraction:
                map_stats["passed"] += 1
    return cache_d, cache_sigma, cache_valid, geometry, map_stats


def frame_evidence(positions_t, present_t, cache, geometry, cameras, frame, config,
                   *, evidence_frame_by_camera=None):
    """Per-frame (N, C) near/gap_raw/occ_depth/occ_sigma + witness-applied gap."""
    cache_d, cache_sigma, cache_valid = cache
    num_primitives = positions_t.shape[0]
    num_cameras = len(cameras)
    tau = float(config["margin_tau_rel"])
    kappa = float(config["margin_kappa_sigma"])
    k_gap = float(config["certification"]["k_gap"])
    near_clip = float(config["near_clip"])
    states = np.zeros((num_primitives, num_cameras), dtype=np.int8)
    gap_raw = np.zeros((num_primitives, num_cameras), dtype=bool)
    occ_depth = np.full((num_primitives, num_cameras), np.nan, dtype=np.float32)
    occ_sigma = np.full((num_primitives, num_cameras), np.nan, dtype=np.float32)
    xyz = positions_t.astype(np.float64)
    for col in range(num_cameras):
        evidence_frame = frame
        if evidence_frame_by_camera is not None:
            evidence_frame = int(evidence_frame_by_camera[col])
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
        s, g, od, osg = census.classify_states_v2(
            z, pixels, in_view, present_t, d_map, sigma_map, valid_map,
            tau_rel=tau, kappa=kappa, k_gap=k_gap,
        )
        states[:, col] = s
        gap_raw[:, col] = g
        occ_depth[:, col] = od
        occ_sigma[:, col] = osg
    oww = census.occluded_with_witness(states)
    near = states == census.STATE_NEAR_SURFACE
    gap_witnessed = gap_raw & oww
    return states, near, gap_witnessed, occ_depth, occ_sigma


def baseline_eligibility(positions, presence, cache, geometry, cameras, frames, config):
    cert = config["certification"]
    lo, hi = cert["baseline_window"]
    min_eval = int(cert["baseline_min_evaluable"])
    majority = float(cert["baseline_behind_majority"])
    num_primitives = positions[0].shape[0]
    num_cameras = len(cameras)
    behind = np.zeros((num_primitives, num_cameras), dtype=np.int16)
    evaluable = np.zeros((num_primitives, num_cameras), dtype=np.int16)
    for t_index, frame in enumerate(frames):
        if frame < lo or frame > hi:
            continue
        states, _, _, _, _ = frame_evidence(
            positions[t_index], presence[t_index], cache, geometry, cameras, frame, config
        )
        behind += (states == census.STATE_BEHIND).astype(np.int16)
        evaluable += (states != census.STATE_NOT_EVALUABLE).astype(np.int16)
    static_parallax = (evaluable >= min_eval) & (behind >= majority * evaluable)
    return ~static_parallax, {
        "excluded_pairs": int(static_parallax.sum()),
        "window": [lo, hi],
    }


def run_certification_pass(positions, presence, cache, geometry, cameras, frames, config,
                           *, eligible, frame_assignment=None, collect_quality=False,
                           sample_cap=0):
    cert = config["certification"]
    win_lo, win_hi = cert["certification_window"]
    num_primitives = positions[0].shape[0]
    tracker = census.CertifiedRevealTracker(
        num_primitives, len(cameras),
        anchor_consec=int(cert["anchor_consec"]),
        entry_consec=int(cert["entry_consec"]),
        reveal_consec=int(cert["reveal_consec"]),
        min_gap_occ_frames=int(cert["min_gap_occ_frames"]),
        grace_frames=int(cert["grace_frames"]),
        smooth_rel=float(cert["occluder_smooth_rel"]),
        smooth_kappa=float(cert["occluder_smooth_kappa_sigma"]),
        sample_cap=sample_cap,
        eligible=eligible,
    )
    quality = None
    if collect_quality:
        quality = {
            "evaluable": np.zeros(num_primitives, dtype=np.int64),
            "near": np.zeros(num_primitives, dtype=np.int64),
            "margin_ratio_sum": np.zeros(num_primitives, dtype=np.float64),
        }
    for t_index, frame in enumerate(frames):
        if frame < win_lo or frame > win_hi:
            continue
        assignment = None
        if frame_assignment is not None:
            assignment = frame_assignment[:, t_index]
        states, near, gap_witnessed, occ_depth, occ_sigma = frame_evidence(
            positions[t_index], presence[t_index], cache, geometry, cameras, frame,
            config, evidence_frame_by_camera=(
                None if assignment is None else [frames[int(a)] for a in assignment]
            ),
        )
        tracker.update(frame, near, gap_witnessed, occ_depth, occ_sigma)
        if quality is not None:
            evaluable = states != census.STATE_NOT_EVALUABLE
            quality["evaluable"] += evaluable.sum(axis=1)
            quality["near"] += near.sum(axis=1)
    return tracker, quality


def rho_from(valid_pairs: int, shuffle_pairs: int) -> float:
    if shuffle_pairs <= 0:
        return float("inf") if valid_pairs > 0 else 0.0
    return valid_pairs / shuffle_pairs


def stratified_rho(strata_masks: dict, valid_pairs: np.ndarray, shuffle_pairs: np.ndarray):
    out = {}
    for name, mask in strata_masks.items():
        v = int(valid_pairs[mask].sum())
        s = int(shuffle_pairs[mask].sum())
        r = rho_from(v, s)
        out[name] = {
            "valid_pairs": v,
            "shuffle_pairs": s,
            "rho": r if np.isfinite(r) else "inf",
        }
    return out


def run_consistency(cache, geometry, cameras, frames, config):
    cache_d, cache_sigma, cache_valid = cache
    stride = int(config["consistency"]["frame_stride"])
    pixel_stride = int(config["consistency"]["pixel_stride"])
    pair_count = int(config["consistency"]["camera_pair_count"])
    tau = float(config["margin_tau_rel"])
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
    config = load_census2_config(args.config)
    if args.output_root:
        config["output_root"] = census.expand_work(args.output_root)

    manifest_path = os.path.join(config["p01_root"], "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    index, all_cameras = census.build_p01_index(manifest, config["p01_root"])
    excluded = set(config["excluded_cameras"])
    cameras = [c for c in all_cameras if c not in excluded]

    frame_cfg = config["frames"]
    frames = list(range(int(frame_cfg["start"]), int(frame_cfg["end"]) + 1))
    scientific = True
    if args.frame_limit is not None:
        frames = frames[: args.frame_limit]
        scientific = False

    config_sha = sha256_file(args.config)
    manifest_sha = sha256_file(manifest_path)
    print(f"[census2] cameras={len(cameras)} (excluded {sorted(excluded)}) "
          f"frames={len(frames)} scientific={scientific}", flush=True)

    cache_d, cache_sigma, cache_valid, geometry, map_stats = build_consensus_cache(
        index, cameras, frames, config
    )
    cache = (cache_d, cache_sigma, cache_valid)
    map_pass_fraction = (map_stats["passed"] / map_stats["total"]) if map_stats["total"] else 0.0
    print(f"[census2] consensus maps={map_stats['total']} pass_fraction={map_pass_fraction:.4f}",
          flush=True)
    consistency = run_consistency(cache, geometry, cameras, frames, config)
    assignment = census.shuffled_frame_assignment(
        len(frames), len(cameras), int(config["shuffle_seed"])
    )

    checkpoints_out = {}
    primary_out = None
    for entry in config["checkpoints"]:
        label = entry["label"]
        try:
            gaussians, ckpt_iter = load_model(entry["path"], config["model"])
        except Exception as exc:  # preregistered fallback for allow_restore_failure
            if entry.get("allow_restore_failure"):
                checkpoints_out[label] = {
                    "restored": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"[census2] {label}: restore FAILED (allowed) — skipping", flush=True)
                continue
            traceback.print_exc()
            raise
        ckpt_sha = sha256_file(entry["path"])
        cells = {}
        modes = [("moving", False)]
        if entry["role"] == "primary":
            modes.append(("frozen", True))
        quality = None
        opacity_mask = None
        for mode_name, frozen in modes:
            positions, presence, opacity_mask = precompute_positions(
                gaussians, config, frames, frozen=frozen
            )
            eligible, baseline_stats = baseline_eligibility(
                positions, presence, cache, geometry, cameras, frames, config
            )
            tracker_valid, q = run_certification_pass(
                positions, presence, cache, geometry, cameras, frames, config,
                eligible=eligible,
                collect_quality=(entry["role"] == "primary" and mode_name == "moving"),
                sample_cap=(int(config["transitions_sample_cap"])
                            if entry["role"] == "primary" and mode_name == "moving" else 0),
            )
            if q is not None:
                quality = q
            tracker_shuffle, _ = run_certification_pass(
                positions, presence, cache, geometry, cameras, frames, config,
                eligible=eligible, frame_assignment=assignment,
            )
            v_pairs = int(tracker_valid.certified_pairs.sum())
            s_pairs = int(tracker_shuffle.certified_pairs.sum())
            r = rho_from(v_pairs, s_pairs)
            cells[mode_name] = {
                "baseline": baseline_stats,
                "valid": tracker_valid.summary(cameras),
                "shuffle": tracker_shuffle.summary(cameras),
                "rho": r if np.isfinite(r) else "inf",
            }
            print(f"[census2] {label}/{mode_name}: valid={v_pairs} shuffle={s_pairs} "
                  f"rho={r if np.isfinite(r) else 'inf'}", flush=True)
            if entry["role"] == "primary" and mode_name == "moving":
                primary_out = {
                    "tracker_valid_pairs": tracker_valid.certified_pairs,
                    "tracker_shuffle_pairs": tracker_shuffle.certified_pairs,
                    "samples": tracker_valid.samples,
                    "summary_valid": cells[mode_name]["valid"],
                    "summary_shuffle": cells[mode_name]["shuffle"],
                    "positions_range": None,
                    "gaussians": gaussians,
                }
        # motion range for strata / diagnostics (moving positions)
        positions, _, _ = precompute_positions(gaussians, config, frames, frozen=False)
        frozen_xyz = positions and gaussians._xyz.detach().cpu().numpy().astype(np.float32)
        motion_range = np.zeros(positions[0].shape[0], dtype=np.float32)
        for xyz_t in positions:
            motion_range = np.maximum(
                motion_range, np.linalg.norm(xyz_t - frozen_xyz, axis=1)
            )
        if entry["role"] == "primary" and primary_out is not None:
            primary_out["motion_range"] = motion_range
            primary_out["opacity_mask_excluded"] = int((~opacity_mask).sum())
        checkpoints_out[label] = {
            "restored": True,
            "checkpoint_iteration": ckpt_iter,
            "checkpoint_sha256": ckpt_sha,
            "num_primitives": int(gaussians._xyz.shape[0]),
            "median_motion_range": float(np.median(motion_range)),
            "cells": cells,
        }
        del positions

    if primary_out is None:
        raise RuntimeError("primary checkpoint produced no output")

    # Strata on the primary moving cell
    import torch

    gaussians = primary_out["gaussians"]
    with torch.no_grad():
        opacity = torch.sigmoid(gaussians._opacity).squeeze(-1).cpu().numpy()
        max_scale = torch.exp(gaussians._scaling).max(dim=1).values.cpu().numpy()
    quality = quality or {
        "evaluable": np.ones(opacity.shape[0], dtype=np.int64),
        "near": np.zeros(opacity.shape[0], dtype=np.int64),
    }
    near_stability = quality["near"] / np.maximum(quality["evaluable"], 1)
    motion_range = primary_out["motion_range"]

    strata_masks = {}
    strata_defs = {}
    labels, qs = census.quartile_labels(motion_range)
    strata_defs["motion_range_quartile_edges"] = qs
    for q in range(4):
        strata_masks[f"motion_q{q}"] = labels == q
    labels, qs = census.quartile_labels(near_stability)
    strata_defs["near_stability_quartile_edges"] = qs
    for q in range(4):
        strata_masks[f"near_stability_q{q}"] = labels == q
    split = float(config["strata"]["opacity_split"])
    strata_masks["opacity_high"] = opacity >= split
    strata_masks["opacity_low"] = opacity < split
    labels, qs = census.quartile_labels(max_scale)
    strata_defs["scale_quartile_edges"] = qs
    for q in range(4):
        strata_masks[f"scale_q{q}"] = labels == q

    valid_pair_counts = primary_out["tracker_valid_pairs"].sum(axis=1)
    shuffle_pair_counts = primary_out["tracker_shuffle_pairs"].sum(axis=1)
    strata = stratified_rho(strata_masks, valid_pair_counts, shuffle_pair_counts)

    primary_label = next(e["label"] for e in config["checkpoints"] if e["role"] == "primary")
    primary_cells = checkpoints_out[primary_label]["cells"]
    floors = census.evaluate_census2_floors(
        {
            "valid": primary_cells["moving"]["valid"],
            "shuffle": primary_cells["moving"]["shuffle"],
            "consistency": consistency,
            "consensus_maps": {"pass_fraction": map_pass_fraction},
        },
        config["floors"],
    )

    # Diagnosis D1-D4
    thr = float(config["diagnosis"]["d_rho_threshold"])
    def rho_num(cell):
        r = cell["rho"]
        return float("inf") if r == "inf" else float(r)
    rho_moving = rho_num(primary_cells["moving"])
    rho_frozen = rho_num(primary_cells["frozen"])
    moving_rhos = {
        label: rho_num(data["cells"]["moving"])
        for label, data in checkpoints_out.items()
        if data.get("restored") and "cells" in data
    }
    finite_rhos = [r for r in moving_rhos.values() if np.isfinite(r) and r > 0]
    d4_ratio = (max(finite_rhos) / min(finite_rhos)) if len(finite_rhos) >= 2 else None
    top_strata_rho = max(
        (v["rho"] if v["rho"] != "inf" else float("inf"))
        for k, v in strata.items() if k in ("motion_q0", "near_stability_q3")
    )
    if rho_moving >= thr:
        diagnosis = "D1_rule_failure_confirmed_carrier_adequate"
    elif rho_frozen >= thr:
        if top_strata_rho >= thr:
            diagnosis = "D2_interaction_rule_works_on_quality_strata"
        else:
            diagnosis = "D2_carrier_failure"
    else:
        diagnosis = "D3_residual_evidence_or_rule_failure"

    r009 = {}
    for name, (lo, hi) in config.get("r009_descriptive_windows", {}).items():
        count = sum(
            v for k, v in primary_cells["moving"]["valid"]["completions_by_frame"].items()
            if lo <= int(k) <= hi
        )
        r009[name] = {"frame_range": [lo, hi], "certified_events": count}

    for label in checkpoints_out:
        checkpoints_out[label].pop("gaussians", None)

    payload = {
        "schema_version": "phase0-census2-v1",
        "scientific": scientific,
        "scene": config["scene"],
        "config_sha256": config_sha,
        "p01_manifest_sha256": manifest_sha,
        "cameras": cameras,
        "excluded_cameras": sorted(excluded),
        "frame_count": len(frames),
        "consensus_map_stats": {**map_stats, "pass_fraction": map_pass_fraction},
        "cross_view_consistency": consistency,
        "checkpoints": {
            k: {kk: vv for kk, vv in v.items() if kk != "gaussians"}
            for k, v in checkpoints_out.items()
        },
        "primary_label": primary_label,
        "strata_definitions": strata_defs,
        "stratified_rho_primary_moving": strata,
        "rho_moving_primary": rho_moving if np.isfinite(rho_moving) else "inf",
        "rho_frozen_primary": rho_frozen if np.isfinite(rho_frozen) else "inf",
        "moving_rho_by_checkpoint": {
            k: (v if np.isfinite(v) else "inf") for k, v in moving_rhos.items()
        },
        "d4_checkpoint_dependence_ratio": d4_ratio,
        "diagnosis": diagnosis,
        "r009_descriptive_overlap": r009,
        "floors": floors,
        "census2_go": floors["census2_go"] if scientific else None,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "absolute_output_root": config["output_root"],
        "wall_seconds": round(time.time() - started, 1),
    }
    payload["canonical_scientific_sha256"] = canonical_content_hash(payload)

    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "census2-v1.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    with open(output_root / "transitions-sample.json", "w", encoding="utf-8") as handle:
        json.dump({"samples": primary_out["samples"], "cameras": cameras}, handle)

    print(f"[census2] diagnosis={diagnosis} census2_go={payload['census2_go']} "
          f"wall={payload['wall_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
