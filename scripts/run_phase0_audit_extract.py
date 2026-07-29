#!/usr/bin/env python3
"""Phase 0 blinded visual forensic audit — case extraction (workstream A).

Builds a blinded review package from the census-v2 configuration on the
development scene only: certified events under real and shuffled evidence,
rule-rejected candidates, censored long occlusions, near-threshold flicker
pairs, and controlled synthetic fixtures. Writes per-case JSON + NPZ under the
output root, a provenance key kept in a separate subdirectory, and renders
contact sheets and clips via depth_visibility.audit_render.

Read boundary: training-camera RGB of the development scene (crop references
only; loaded at render time), sealed P01 arrays, the census-v2 checkpoint.
No cam00 RGB, no annotations, no evaluator masks, no R009 crop pixels, no W&B.
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from depth_visibility import primitive_census as census  # noqa: E402
from depth_visibility import audit_render  # noqa: E402
from depth_visibility.canonical import sha256_file  # noqa: E402


def load_census2_module():
    spec = importlib.util.spec_from_file_location(
        "run_phase0_census2", REPO_ROOT / "scripts" / "run_phase0_census2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="census-v2 config JSON")
    parser.add_argument("--census2-artifact", required=True,
                        help="census2-v1 output root (for recorded valid samples)")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


CATEGORY_PLAN = [
    ("real_certified", 8),
    ("shuffle_certified", 8),
    ("rejected_short", 3),
    ("rejected_grace", 3),
    ("censored_long", 6),
    ("near_threshold", 5),
    ("fixture", 3),
]


def stratify_by_length(records, count, rng):
    if not records:
        return []
    records = sorted(records, key=lambda r: r["gap_occ_frames"])
    short = [r for r in records if r["gap_occ_frames"] <= 5]
    medium = [r for r in records if 6 <= r["gap_occ_frames"] <= 12]
    long_ = [r for r in records if r["gap_occ_frames"] >= 13]
    picks = []
    for bucket, want in ((short, count - count // 2 - count // 4),
                         (medium, count // 2), (long_, count // 4)):
        if bucket:
            idx = rng.choice(len(bucket), size=min(want, len(bucket)), replace=False)
            picks.extend(bucket[i] for i in np.atleast_1d(idx))
    pool = [r for r in records if r not in picks]
    while len(picks) < count and pool:
        picks.append(pool.pop(rng.integers(0, len(pool))))
    return picks[:count]


def extract_series(case, positions, presence, cache, geometry, cameras, frames,
                   config, assignment=None):
    """Per-frame series for one (primitive, camera) over the case window."""
    cache_d, cache_sigma, cache_valid = cache
    tau = float(config["margin_tau_rel"])
    kappa = float(config["margin_kappa_sigma"])
    k_gap = float(config["certification"]["k_gap"])
    near_clip = float(config["near_clip"])
    prim = case["primitive"]
    col = case["camera_index"]
    frame_index = {f: i for i, f in enumerate(frames)}
    series = {k: [] for k in ("frame", "evidence_frame", "px", "py", "z", "d",
                              "sigma", "margin", "gap_ratio", "state", "witness")}
    for frame in case["window"]:
        if frame not in frame_index:
            continue
        t_index = frame_index[frame]
        xyz = positions[t_index][prim:prim + 1].astype(np.float64)
        present = presence[t_index][prim:prim + 1]
        evidence_frame = frame
        if assignment is not None:
            evidence_frame = frames[int(assignment[col, t_index])]
        witness = 0
        state_here = census.STATE_NOT_EVALUABLE
        px = py = -1
        z_val = d_val = sig_val = margin_val = gap_ratio = float("nan")
        for cam2 in range(len(cameras)):
            ev_frame2 = frame
            if assignment is not None:
                ev_frame2 = frames[int(assignment[cam2, t_index])]
            key = (cam2, ev_frame2)
            geo_key = (cam2, frame)
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
            states, _, _, _ = census.classify_states_v2(
                z, pixels, in_view, present, d_map, sigma_map, valid_map,
                tau_rel=tau, kappa=kappa, k_gap=k_gap,
            )
            if cam2 == col:
                state_here = int(states[0])
                if in_view[0]:
                    px, py = int(pixels[0, 0]), int(pixels[0, 1])
                    z_val = float(z[0])
                    if valid_map[py, px]:
                        d_val = float(d_map[py, px])
                        sig_val = float(sigma_map[py, px])
                        margin_val = max(tau * d_val, kappa * sig_val)
                        gap_ratio = (z_val - d_val) / margin_val if margin_val > 0 else float("nan")
            elif states[0] == census.STATE_NEAR_SURFACE:
                witness += 1
        series["frame"].append(int(frame))
        series["evidence_frame"].append(int(evidence_frame))
        series["px"].append(px)
        series["py"].append(py)
        series["z"].append(z_val)
        series["d"].append(d_val)
        series["sigma"].append(sig_val)
        series["margin"].append(margin_val)
        series["gap_ratio"].append(gap_ratio)
        series["state"].append(state_here)
        series["witness"].append(witness)
    return series


def crop_stacks(case, series, cache, cameras, half=48):
    cache_d, cache_sigma, _ = cache
    col = case["camera_index"]
    px_values = [p for p in series["px"] if p >= 0]
    py_values = [p for p in series["py"] if p >= 0]
    if not px_values:
        return None, (0, 0)
    cx = int(np.median(px_values))
    cy = int(np.median(py_values))
    d_crops, s_crops = [], []
    for frame, ev_frame in zip(series["frame"], series["evidence_frame"]):
        key = (col, ev_frame)
        if key not in cache_d:
            d_crops.append(np.full((2 * half, 2 * half), np.nan, dtype=np.float16))
            s_crops.append(np.full((2 * half, 2 * half), np.nan, dtype=np.float16))
            continue
        d_map = cache_d[key]
        s_map = cache_sigma[key]
        height, width = d_map.shape
        y0, y1 = max(0, cy - half), min(height, cy + half)
        x0, x1 = max(0, cx - half), min(width, cx + half)
        d_pad = np.full((2 * half, 2 * half), np.nan, dtype=np.float16)
        s_pad = np.full((2 * half, 2 * half), np.nan, dtype=np.float16)
        d_pad[y0 - (cy - half):y1 - (cy - half), x0 - (cx - half):x1 - (cx - half)] = d_map[y0:y1, x0:x1]
        s_pad[y0 - (cy - half):y1 - (cy - half), x0 - (cx - half):x1 - (cx - half)] = s_map[y0:y1, x0:x1]
        d_crops.append(d_pad)
        s_crops.append(s_pad)
    return {"d": np.stack(d_crops), "sigma": np.stack(s_crops)}, (cx, cy)


def make_fixtures(config):
    """Three controlled fixtures with known ground truth, in series format."""
    tau = float(config["margin_tau_rel"])
    fixtures = []
    half = 48

    def base_series(frames):
        return {k: [] for k in ("frame", "evidence_frame", "px", "py", "z", "d",
                                "sigma", "margin", "gap_ratio", "state", "witness")}

    def push(series, frame, z, d, sigma, state, witness):
        margin = max(tau * d, 2.5 * sigma)
        series["frame"].append(frame)
        series["evidence_frame"].append(frame)
        series["px"].append(half)
        series["py"].append(half)
        series["z"].append(z)
        series["d"].append(d)
        series["sigma"].append(sigma)
        series["margin"].append(margin)
        series["gap_ratio"].append((z - d) / margin)
        series["state"].append(state)
        series["witness"].append(witness)

    def plane_crop(depth_value, occluder_depth=None, occluder_x=None, noise=0.0,
                   rng=None):
        d = np.full((2 * half, 2 * half), depth_value, dtype=np.float32)
        if noise and rng is not None:
            d += rng.normal(0.0, noise, d.shape).astype(np.float32)
        rgb = np.full((2 * half, 2 * half, 3), 150, dtype=np.uint8)
        if occluder_depth is not None and occluder_x is not None:
            x0 = int(np.clip(occluder_x, 0, 2 * half - 30))
            d[:, x0:x0 + 30] = occluder_depth
            rgb[:, x0:x0 + 30] = (90, 60, 40)
        return d, rgb

    rng = np.random.default_rng(7)

    # Fixture 1: genuine hide/reveal — occluder sweeps across the marker.
    series = base_series(None)
    d_stack, s_stack, rgb_stack = [], [], []
    for i, frame in enumerate(range(100, 128)):
        if 8 <= i < 18:
            occ_x = 20 + 5 * (i - 8)
            covered = occ_x <= half <= occ_x + 30
            d_val = 2.0 if covered else 5.0
            state = census.STATE_BEHIND if covered else census.STATE_NEAR_SURFACE
            push(series, frame, 5.0, d_val, 0.02, state, 3 if covered else 4)
            d_img, rgb_img = plane_crop(5.0, 2.0, occ_x)
        else:
            push(series, frame, 5.0, 5.0, 0.02, census.STATE_NEAR_SURFACE, 4)
            d_img, rgb_img = plane_crop(5.0)
        d_stack.append(d_img.astype(np.float16))
        s_stack.append(np.full_like(d_img, 0.02, dtype=np.float16))
        rgb_stack.append(rgb_img)
    fixtures.append({
        "category": "fixture", "fixture_truth": "genuine_disocclusion",
        "series": series, "decision_text": "certified reveal (run 8)",
        "crops": {"d": np.stack(d_stack), "sigma": np.stack(s_stack),
                  "rgb": np.stack(rgb_stack)},
    })

    # Fixture 2: static parallax — always behind a nearer static structure.
    series = base_series(None)
    d_stack, s_stack, rgb_stack = [], [], []
    for frame in range(100, 128):
        push(series, frame, 5.0, 4.2, 0.02, census.STATE_BEHIND, 2)
        d_img, rgb_img = plane_crop(4.2)
        d_stack.append(d_img.astype(np.float16))
        s_stack.append(np.full_like(d_img, 0.02, dtype=np.float16))
        rgb_stack.append(rgb_img)
    fixtures.append({
        "category": "fixture", "fixture_truth": "static_parallax",
        "series": series, "decision_text": "no certified reveal within window",
        "crops": {"d": np.stack(d_stack), "sigma": np.stack(s_stack),
                  "rgb": np.stack(rgb_stack)},
    })

    # Fixture 3: margin flicker — evidence depth oscillates around the margin.
    series = base_series(None)
    d_stack, s_stack, rgb_stack = [], [], []
    for i, frame in enumerate(range(100, 128)):
        d_val = 5.0 - 0.18 * ((-1) ** i) - 0.02 * rng.standard_normal()
        state = census.STATE_BEHIND if 5.0 - d_val > max(tau * d_val, 0.05) else (
            census.STATE_NEAR_SURFACE if abs(5.0 - d_val) <= max(tau * d_val, 0.05)
            else census.STATE_IN_FRONT)
        push(series, frame, 5.0, d_val, 0.02, state, 3)
        d_img, rgb_img = plane_crop(d_val, noise=0.03, rng=rng)
        d_stack.append(d_img.astype(np.float16))
        s_stack.append(np.full_like(d_img, 0.02, dtype=np.float16))
        rgb_stack.append(rgb_img)
    fixtures.append({
        "category": "fixture", "fixture_truth": "depth_flicker",
        "series": series, "decision_text": "no certified reveal (interruption budget exceeded)",
        "crops": {"d": np.stack(d_stack), "sigma": np.stack(s_stack),
                  "rgb": np.stack(rgb_stack)},
    })
    return fixtures


def decision_text_for(category, record):
    n = record.get("gap_occ_frames", 0)
    if category in ("real_certified", "shuffle_certified"):
        return f"certified reveal (occluded run of {n} frames, clean reveal)"
    if category == "rejected_short":
        return f"rejected: occluded run of {n} frames below the 3-frame minimum"
    if category == "rejected_grace":
        return f"rejected: interruption budget exceeded after {n} occluded frames"
    if category == "censored_long":
        return f"no decision: occluded run of {n} frames with no certified reveal"
    if category == "near_threshold":
        return "no decision: high state-flip pair (near/behind oscillation)"
    return "n/a"


def main() -> int:
    args = parse_args()
    started = time.time()
    census2 = load_census2_module()
    config = census2.load_census2_config(args.config)
    rng = np.random.default_rng(args.seed)

    manifest_path = os.path.join(config["p01_root"], "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    index, all_cameras = census.build_p01_index(manifest, config["p01_root"])
    excluded = set(config["excluded_cameras"])
    cameras = [c for c in all_cameras if c not in excluded]
    frames = list(range(int(config["frames"]["start"]), int(config["frames"]["end"]) + 1))
    scientific = True
    if args.frame_limit:
        frames = frames[: args.frame_limit]
        scientific = False

    print(f"[audit] cameras={len(cameras)} frames={len(frames)}", flush=True)
    cache_d, cache_sigma, cache_valid, geometry, map_stats = census2.build_consensus_cache(
        index, cameras, frames, config
    )
    cache = (cache_d, cache_sigma, cache_valid)

    primary = next(e for e in config["checkpoints"] if e["role"] == "primary")
    gaussians, ckpt_iter = census2.load_model(primary["path"], config["model"])
    positions, presence, _ = census2.precompute_positions(
        gaussians, config, frames, frozen=False
    )
    eligible, _ = census2.baseline_eligibility(
        positions, presence, cache, geometry, cameras, frames, config
    )
    assignment = census.shuffled_frame_assignment(
        len(frames), len(cameras), int(config["shuffle_seed"])
    )

    cert = config["certification"]
    win_lo, win_hi = cert["certification_window"]
    num_primitives = positions[0].shape[0]

    def make_tracker(diag):
        return census.CertifiedRevealTracker(
            num_primitives, len(cameras),
            anchor_consec=int(cert["anchor_consec"]),
            entry_consec=int(cert["entry_consec"]),
            reveal_consec=int(cert["reveal_consec"]),
            min_gap_occ_frames=int(cert["min_gap_occ_frames"]),
            grace_frames=int(cert["grace_frames"]),
            smooth_rel=float(cert["occluder_smooth_rel"]),
            smooth_kappa=float(cert["occluder_smooth_kappa_sigma"]),
            sample_cap=20000, eligible=eligible, diagnostics=diag,
        )

    # Sweep 1a: valid evidence, diagnostics + flicker toggles.
    tracker_valid = make_tracker(True)
    toggles = np.zeros((num_primitives, len(cameras)), dtype=np.uint16)
    prev_states = None
    for t_index, frame in enumerate(frames):
        if frame < win_lo or frame > win_hi:
            continue
        states, near, gap_w, occ_d, occ_s = census2.frame_evidence(
            positions[t_index], presence[t_index], cache, geometry, cameras, frame, config
        )
        if prev_states is not None:
            flip = ((states == census.STATE_NEAR_SURFACE) & (prev_states == census.STATE_BEHIND)) | \
                   ((states == census.STATE_BEHIND) & (prev_states == census.STATE_NEAR_SURFACE))
            toggles += flip.astype(np.uint16)
        prev_states = states
        tracker_valid.update(frame, near, gap_w, occ_d, occ_s)
    print(f"[audit] valid: certified={tracker_valid.certified_total} "
          f"aborts={tracker_valid.abort_total} short={tracker_valid.short_end_total}", flush=True)

    # Sweep 1b: shuffled evidence with sample recording.
    tracker_shuffle = make_tracker(False)
    for t_index, frame in enumerate(frames):
        if frame < win_lo or frame > win_hi:
            continue
        _, near, gap_w, occ_d, occ_s = census2.frame_evidence(
            positions[t_index], presence[t_index], cache, geometry, cameras, frame,
            config, evidence_frame_by_camera=[frames[int(a)] for a in assignment[:, t_index]],
        )
        tracker_shuffle.update(frame, near, gap_w, occ_d, occ_s)
    print(f"[audit] shuffle: certified={tracker_shuffle.certified_total}", flush=True)

    # Load recorded valid samples from the sealed census-v2 artifact.
    with open(Path(args.census2_artifact) / "transitions-sample.json") as handle:
        recorded = json.load(handle)
    valid_samples = recorded["samples"]

    censored = tracker_valid.censored_long_runs(15)
    censored += [r for r in tracker_valid.abort_records if r["gap_occ_frames"] >= 15]
    grace_aborts = [r for r in tracker_valid.abort_records
                    if 3 <= r["gap_occ_frames"] < 15]
    shorts = tracker_valid.short_end_records

    top_toggle = np.argsort(toggles, axis=None)[::-1][:200]
    near_threshold_records = []
    for flat in top_toggle.tolist():
        p, c = divmod(flat, len(cameras))
        near_threshold_records.append({
            "primitive": int(p), "camera_index": int(c),
            "toggles": int(toggles[p, c]), "gap_occ_frames": 0,
        })

    picks = {
        "real_certified": stratify_by_length(valid_samples, 8, rng),
        "shuffle_certified": stratify_by_length(tracker_shuffle.samples, 8, rng),
        "rejected_short": list(rng.choice(shorts, size=min(3, len(shorts)),
                                          replace=False)) if shorts else [],
        "rejected_grace": list(rng.choice(grace_aborts, size=min(3, len(grace_aborts)),
                                          replace=False)) if grace_aborts else [],
        "censored_long": stratify_by_length(censored, 6, rng) if censored else [],
        "near_threshold": near_threshold_records[:5],
    }

    cases = []
    for category, records in picks.items():
        for record in records:
            record = dict(record)
            if category in ("real_certified", "shuffle_certified"):
                end = record["frame"] if "frame" in record else record["end_frame"]
                start = end - record["gap_occ_frames"] - 6
                window = list(range(max(win_lo, start - 6), min(win_hi, end + 6) + 1))
            elif category in ("rejected_short", "rejected_grace"):
                end = record["frame"]
                start = end - record["gap_occ_frames"] - 6
                window = list(range(max(win_lo, start - 6), min(win_hi, end + 8) + 1))
            elif category == "censored_long":
                if "frame" in record:
                    end = record["frame"]
                    window = list(range(max(win_lo, end - record["gap_occ_frames"] - 6),
                                        min(win_hi, end + 6) + 1))
                else:
                    window = list(range(max(win_lo, win_hi - record["gap_occ_frames"] - 8),
                                        win_hi + 1))
            else:
                window = list(range(110, 150))
            if len(window) > 60:
                window = window[-60:]
            cases.append({
                "category": category, "record": record,
                "primitive": record["primitive"], "camera_index": record["camera_index"],
                "window": window,
            })

    fixtures = make_fixtures(config)

    # Sweep 2: series + crops for each sampled case.
    output_root = Path(census.expand_work(args.output_root))
    (output_root / "cases").mkdir(parents=True, exist_ok=True)
    (output_root / "provenance").mkdir(parents=True, exist_ok=True)
    (output_root / "sheets").mkdir(parents=True, exist_ok=True)
    (output_root / "clips").mkdir(parents=True, exist_ok=True)

    total = len(cases) + len(fixtures)
    order = rng.permutation(total)
    blind_ids = [f"case_{i + 1:02d}" for i in range(total)]
    provenance = {}
    prepared = []

    scale_x = 1352.0 / 504.0
    scale_y = 1014.0 / 378.0
    rgb_root = Path(census.expand_work("$WORK/proj_adags/data/n3v")) / config["scene"] / "images"

    all_items = cases + fixtures
    for slot, item_index in enumerate(order.tolist()):
        item = all_items[item_index]
        blind_id = blind_ids[slot]
        if item.get("category") == "fixture":
            series = item["series"]
            crops = item["crops"]
            center = (48, 48)
            camera_name = "synthetic"
            decision = item["decision_text"]
            provenance[blind_id] = {"category": "fixture",
                                    "fixture_truth": item["fixture_truth"]}
        else:
            assignment_used = assignment if item["category"] == "shuffle_certified" else None
            series = extract_series(item, positions, presence, cache, geometry,
                                    cameras, frames, config, assignment=assignment_used)
            crops, center = crop_stacks(item, series, cache, cameras)
            if crops is None:
                provenance[blind_id] = {"category": item["category"],
                                        "record": item["record"], "skipped": True}
                continue
            camera_name = cameras[item["camera_index"]]
            decision = decision_text_for(item["category"], item["record"])
            provenance[blind_id] = {"category": item["category"], "record": item["record"],
                                    "camera": camera_name}
        case_json = {
            "blind_id": blind_id,
            "camera": camera_name,
            "series": series,
            "decision_text": decision,
            "crop_center_depth_res": list(center),
            "crop_half": 48,
            "rgb_scale": [scale_x, scale_y],
            "rgb_root": str(rgb_root) if camera_name != "synthetic" else None,
        }
        with open(output_root / "cases" / f"{blind_id}.json", "w") as handle:
            json.dump(case_json, handle)
        np.savez_compressed(output_root / "cases" / f"{blind_id}.npz", **crops)
        prepared.append(blind_id)

    with open(output_root / "provenance" / "blinding_key.json", "w") as handle:
        json.dump({
            "seed": args.seed, "scientific": scientific,
            "checkpoint_iteration": ckpt_iter,
            "checkpoint_sha256": sha256_file(primary["path"]),
            "valid_certified_total": tracker_valid.certified_total,
            "shuffle_certified_total": tracker_shuffle.certified_total,
            "abort_total": tracker_valid.abort_total,
            "short_end_total": tracker_valid.short_end_total,
            "censored_candidates": len(censored),
            "cases": provenance,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, handle, indent=1, sort_keys=True)

    print(f"[audit] prepared {len(prepared)} cases; rendering...", flush=True)
    for blind_id in prepared:
        audit_render.render_case(output_root, blind_id)
    print(f"[audit] done wall={round(time.time() - started, 1)}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
