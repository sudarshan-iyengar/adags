"""Phase 0 primitive-centric evidence-opportunity census (CSVL-VPL v2).

Preregistered in research-wiki/operations/phase0-census-preregistration.md and
configs/depth_visibility/phase0_census_v1.json. Counts occlusion-with-witness
states and completed occluded->revealed transitions for checkpointed route0
primitives against P01 DA3 consensus depth. Pure-numpy state logic lives here
so it is unit-testable without torch, CUDA, or the Gaussian model.

State codes per (primitive, camera, frame):
  0 = not evaluable (absent / out of view / invalid pixel)
  1 = near-surface
  2 = behind (occluded candidate)
  3 = in-front
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

STATE_NOT_EVALUABLE = 0
STATE_NEAR_SURFACE = 1
STATE_BEHIND = 2
STATE_IN_FRONT = 3

MAD_TO_SIGMA = 1.4826


def expand_work(path: str) -> str:
    work = os.environ.get("WORK")
    if "$WORK" in path:
        if not work:
            raise ValueError("config path uses $WORK but WORK is not set")
        path = path.replace("$WORK", work)
    return path


def load_census_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != "phase0-census-config-v1":
        raise ValueError(f"unexpected census config schema: {config.get('schema_version')!r}")
    for key in ("checkpoint_path", "p01_root", "output_root"):
        config[key] = expand_work(config[key])
    return config


# ---------------------------------------------------------------------------
# P01 index
# ---------------------------------------------------------------------------

@dataclass
class MemberRef:
    camera_id: str
    member_index: int
    depth_path: str
    confidence_path: str
    aligned_w2c_path: str
    intrinsics_path: str


def build_p01_index(manifest: Mapping[str, Any], p01_root: str) -> tuple[dict[int, dict[str, list[MemberRef]]], list[str]]:
    """Map frame -> camera_id -> member references, plus the sorted camera list."""
    frames: dict[int, dict[str, list[MemberRef]]] = defaultdict(lambda: defaultdict(list))
    cameras: set[str] = set()
    for group in manifest["groups"]:
        frame = int(group["frame"])
        refs = group["array_refs"]
        depth_path = os.path.join(p01_root, refs["depth"]["path"])
        conf_path = os.path.join(p01_root, refs["confidence"]["path"])
        w2c_path = os.path.join(p01_root, refs["aligned_w2c"]["path"])
        k_path = os.path.join(p01_root, refs["processed_intrinsics"]["path"])
        for member_index, camera_id in enumerate(group["member_camera_ids"]):
            cameras.add(camera_id)
            frames[frame][camera_id].append(
                MemberRef(camera_id, member_index, depth_path, conf_path, w2c_path, k_path)
            )
    target = manifest.get("target_camera")
    if target in cameras:
        raise ValueError(f"target camera {target} must never appear in P01 member views")
    return dict(frames), sorted(cameras)


# ---------------------------------------------------------------------------
# Consensus evidence depth
# ---------------------------------------------------------------------------

def consensus_depth(
    depth_stack: np.ndarray,
    confidence_stack: np.ndarray,
    *,
    min_members: int,
    confidence_percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Per-pixel median depth, robust sigma, and validity from member views.

    depth_stack, confidence_stack: (M, H, W) float arrays.
    """
    if depth_stack.ndim != 3 or depth_stack.shape != confidence_stack.shape:
        raise ValueError("depth/confidence stacks must share (M, H, W) shape")
    members = depth_stack.shape[0]
    finite = np.isfinite(depth_stack) & (depth_stack > 0)
    member_count = finite.sum(axis=0)

    depth_masked = np.where(finite, depth_stack, np.nan)
    with np.errstate(all="ignore"):
        d_med = np.nanmedian(depth_masked, axis=0)
        mad = np.nanmedian(np.abs(depth_masked - d_med[None, :, :]), axis=0)
        conf_med = np.nanmedian(np.where(finite, confidence_stack, np.nan), axis=0)
    sigma = MAD_TO_SIGMA * mad

    valid = (member_count >= min_members) & np.isfinite(d_med) & (d_med > 0)
    conf_values = conf_med[valid & np.isfinite(conf_med)]
    if conf_values.size:
        conf_floor = float(np.percentile(conf_values, confidence_percentile))
        valid &= np.isfinite(conf_med) & (conf_med >= conf_floor)
    else:
        conf_floor = float("nan")

    stats = {
        "members": float(members),
        "valid_fraction": float(valid.mean()) if valid.size else 0.0,
        "median_sigma": float(np.median(sigma[valid])) if valid.any() else float("nan"),
        "confidence_floor": conf_floor,
    }
    d_med = np.where(valid, d_med, np.nan)
    sigma = np.where(valid, sigma, np.nan)
    return d_med.astype(np.float32), sigma.astype(np.float32), valid, stats


# ---------------------------------------------------------------------------
# Projection and per-frame state classification
# ---------------------------------------------------------------------------

def project_points(
    xyz: np.ndarray,
    w2c: np.ndarray,
    intrinsics: np.ndarray,
    height: int,
    width: int,
    *,
    near_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points; return integer pixel coords (N,2 as x,y), z, in-view mask."""
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N, 3)")
    rot = w2c[:3, :3]
    trans = w2c[:3, 3]
    cam = xyz @ rot.T + trans[None, :]
    z = cam[:, 2]
    safe_z = np.where(np.abs(z) > 1e-12, z, 1e-12)
    u = intrinsics[0, 0] * cam[:, 0] / safe_z + intrinsics[0, 2]
    v = intrinsics[1, 1] * cam[:, 1] / safe_z + intrinsics[1, 2]
    px = np.rint(u).astype(np.int64)
    py = np.rint(v).astype(np.int64)
    in_view = (z > near_clip) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    pixels = np.stack([px, py], axis=1)
    return pixels, z.astype(np.float64), in_view


def classify_states(
    z: np.ndarray,
    pixels: np.ndarray,
    in_view: np.ndarray,
    present: np.ndarray,
    d_map: np.ndarray,
    sigma_map: np.ndarray,
    valid_map: np.ndarray,
    *,
    tau_rel: float,
    kappa: float,
) -> np.ndarray:
    """Per-primitive int8 state for one (camera, frame)."""
    n = z.shape[0]
    states = np.zeros(n, dtype=np.int8)
    usable = in_view & present
    if not usable.any():
        return states
    px = pixels[usable, 0]
    py = pixels[usable, 1]
    pix_valid = valid_map[py, px]
    idx = np.flatnonzero(usable)[pix_valid]
    if idx.size == 0:
        return states
    px = pixels[idx, 0]
    py = pixels[idx, 1]
    d = d_map[py, px].astype(np.float64)
    sig = sigma_map[py, px].astype(np.float64)
    margin = np.maximum(tau_rel * d, kappa * sig)
    delta = z[idx] - d
    state = np.full(idx.shape[0], STATE_NEAR_SURFACE, dtype=np.int8)
    state[delta > margin] = STATE_BEHIND
    state[delta < -margin] = STATE_IN_FRONT
    states[idx] = state
    return states


def occluded_with_witness(states: np.ndarray) -> np.ndarray:
    """states: (N, C) int8. Occluded-with-witness: behind in c AND near-surface elsewhere."""
    if states.ndim != 2:
        raise ValueError("states must be (N, C)")
    near = states == STATE_NEAR_SURFACE
    near_count = near.sum(axis=1)
    behind = states == STATE_BEHIND
    witness = (near_count[:, None] - near.astype(np.int64)) >= 1
    return behind & witness


# ---------------------------------------------------------------------------
# Temporal run tracking
# ---------------------------------------------------------------------------

@dataclass
class RunTracker:
    """Tracks occlusion runs and completed reveals per (primitive, camera).

    strict: any state other than occluded-with-witness resets the run; a
    completed reveal requires run >= min_run immediately followed by
    near-surface in the same camera.
    relaxed: runs survive plain-behind (witnessless) frames but only
    occluded-with-witness frames count toward min_run.
    """

    num_primitives: int
    num_cameras: int
    min_run: int
    relaxed: bool = False
    sample_cap: int = 0
    run_length: np.ndarray = field(init=False)
    completed_pairs: np.ndarray = field(init=False)
    completed_total: int = field(init=False, default=0)
    end_frames: set = field(init=False, default_factory=set)
    end_cameras: set = field(init=False, default_factory=set)
    per_camera_completions: np.ndarray = field(init=False)
    run_length_histogram: dict = field(init=False, default_factory=dict)
    completions_by_frame: dict = field(init=False, default_factory=dict)
    samples: list = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.run_length = np.zeros((self.num_primitives, self.num_cameras), dtype=np.int32)
        self.completed_pairs = np.zeros((self.num_primitives, self.num_cameras), dtype=bool)
        self.per_camera_completions = np.zeros(self.num_cameras, dtype=np.int64)

    def update(self, frame: int, states: np.ndarray, oww: np.ndarray) -> None:
        near = states == STATE_NEAR_SURFACE
        completed = near & (self.run_length >= self.min_run)
        if completed.any():
            count = int(completed.sum())
            self.completed_total += count
            self.completed_pairs |= completed
            self.completions_by_frame[int(frame)] = (
                self.completions_by_frame.get(int(frame), 0) + count
            )
            self.end_frames.add(frame)
            cams = np.flatnonzero(completed.any(axis=0))
            self.end_cameras.update(int(c) for c in cams)
            self.per_camera_completions += completed.sum(axis=0)
            lengths, length_counts = np.unique(self.run_length[completed], return_counts=True)
            for length, cnt in zip(lengths.tolist(), length_counts.tolist()):
                key = int(length)
                self.run_length_histogram[key] = self.run_length_histogram.get(key, 0) + int(cnt)
            if self.sample_cap and len(self.samples) < self.sample_cap:
                prim_idx, cam_idx = np.nonzero(completed)
                budget = self.sample_cap - len(self.samples)
                for p, c in zip(prim_idx[:budget].tolist(), cam_idx[:budget].tolist()):
                    self.samples.append(
                        {
                            "primitive": int(p),
                            "camera_index": int(c),
                            "end_frame": int(frame),
                            "run_length": int(self.run_length[p, c]),
                        }
                    )
        if self.relaxed:
            keep = oww | (states == STATE_BEHIND)
            self.run_length = np.where(oww, self.run_length + 1, np.where(keep, self.run_length, 0))
        else:
            self.run_length = np.where(oww, self.run_length + 1, 0)

    def summary(self, camera_ids: list[str]) -> dict[str, Any]:
        pair_count = int(self.completed_pairs.sum())
        return {
            "completed_reveal_events": int(self.completed_total),
            "completed_reveal_pairs": pair_count,
            "distinct_end_frames": len(self.end_frames),
            "distinct_end_cameras": len(self.end_cameras),
            "per_camera_completions": {
                camera_ids[c]: int(self.per_camera_completions[c]) for c in range(self.num_cameras)
            },
            "run_length_histogram": {str(k): v for k, v in sorted(self.run_length_histogram.items())},
            "completions_by_frame": {
                str(k): v for k, v in sorted(self.completions_by_frame.items())
            },
        }


# ---------------------------------------------------------------------------
# Cross-view consistency
# ---------------------------------------------------------------------------

def cross_view_consistency(
    d_a: np.ndarray,
    valid_a: np.ndarray,
    w2c_a: np.ndarray,
    k_a: np.ndarray,
    d_b: np.ndarray,
    sigma_b: np.ndarray,
    valid_b: np.ndarray,
    w2c_b: np.ndarray,
    k_b: np.ndarray,
    *,
    pixel_stride: int,
    tau_rel: float,
    kappa: float,
    near_clip: float,
) -> dict[str, float]:
    height, width = d_a.shape
    ys = np.arange(0, height, pixel_stride)
    xs = np.arange(0, width, pixel_stride)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    gy = grid_y.reshape(-1)
    gx = grid_x.reshape(-1)
    sel = valid_a[gy, gx]
    gy, gx = gy[sel], gx[sel]
    if gy.size == 0:
        return {"evaluated": 0, "consistent": 0, "occluded": 0, "conflict": 0}
    depth = d_a[gy, gx].astype(np.float64)
    x_cam = (gx - k_a[0, 2]) / k_a[0, 0] * depth
    y_cam = (gy - k_a[1, 2]) / k_a[1, 1] * depth
    cam_points = np.stack([x_cam, y_cam, depth], axis=1)
    rot_a = w2c_a[:3, :3]
    trans_a = w2c_a[:3, 3]
    world = (cam_points - trans_a[None, :]) @ rot_a
    pixels, z_b, in_view = project_points(world, w2c_b, k_b, height, width, near_clip=near_clip)
    px, py = pixels[in_view, 0], pixels[in_view, 1]
    ok = valid_b[py, px]
    px, py = px[ok], py[ok]
    z_sel = z_b[in_view][ok]
    if z_sel.size == 0:
        return {"evaluated": 0, "consistent": 0, "occluded": 0, "conflict": 0}
    db = d_b[py, px].astype(np.float64)
    sb = sigma_b[py, px].astype(np.float64)
    margin = np.maximum(tau_rel * db, kappa * sb)
    delta = z_sel - db
    consistent = int((np.abs(delta) <= margin).sum())
    occluded = int((delta > margin).sum())
    conflict = int((delta < -margin).sum())
    return {
        "evaluated": int(z_sel.size),
        "consistent": consistent,
        "occluded": occluded,
        "conflict": conflict,
    }


# ---------------------------------------------------------------------------
# Floors
# ---------------------------------------------------------------------------

def evaluate_floors(summary: Mapping[str, Any], floors: Mapping[str, Any]) -> dict[str, Any]:
    occluded_fraction = summary["occluded_with_witness_fraction"]
    strict = summary["strict"]
    shuffle = summary["shuffle"]
    per_camera = summary["per_camera_occluded_fraction"]
    fractions = np.array([v for v in per_camera.values()], dtype=np.float64)
    consistency = summary["consistency"]
    maps = summary["consensus_maps"]

    f1 = occluded_fraction >= floors["f1_min_occluded_fraction"]
    f2 = (
        strict["completed_reveal_pairs"] >= floors["f2_min_reveal_pairs"]
        and strict["distinct_end_frames"] >= floors["f2_min_distinct_end_frames"]
        and strict["distinct_end_cameras"] >= floors["f2_min_distinct_cameras"]
    )
    median_fraction = float(np.median(fractions)) if fractions.size else 0.0
    lo, hi = floors["f3_median_band"]
    f3 = (
        fractions.size > 0
        and lo <= median_fraction <= hi
        and float(fractions.max()) <= floors["f3_max_any_camera"]
    )
    shuffle_pairs = max(shuffle["completed_reveal_pairs"], 0)
    if shuffle_pairs == 0:
        f4 = strict["completed_reveal_pairs"] > 0
        ratio = float("inf") if strict["completed_reveal_pairs"] > 0 else 0.0
    else:
        ratio = strict["completed_reveal_pairs"] / shuffle_pairs
        f4 = ratio >= floors["f4_min_valid_over_shuffle"]
    evaluated = max(consistency["evaluated"], 1)
    conflict_fraction = consistency["conflict"] / evaluated
    f5 = (
        conflict_fraction <= floors["f5_max_conflict_fraction"]
        and maps["pass_fraction"] >= floors["f5_min_map_pass_fraction"]
    )
    verdict = bool(f1 and f2 and f3 and f4 and f5)
    return {
        "f1_abundance": {"pass": bool(f1), "occluded_fraction": occluded_fraction},
        "f2_reveals": {
            "pass": bool(f2),
            "pairs": strict["completed_reveal_pairs"],
            "distinct_end_frames": strict["distinct_end_frames"],
            "distinct_end_cameras": strict["distinct_end_cameras"],
        },
        "f3_non_degeneracy": {
            "pass": bool(f3),
            "median_fraction": median_fraction,
            "max_fraction": float(fractions.max()) if fractions.size else 0.0,
        },
        "f4_control_separation": {
            "pass": bool(f4),
            "valid_pairs": strict["completed_reveal_pairs"],
            "shuffle_pairs": shuffle_pairs,
            "ratio": ratio if np.isfinite(ratio) else "inf",
        },
        "f5_evidence_validity": {
            "pass": bool(f5),
            "conflict_fraction": conflict_fraction,
            "map_pass_fraction": maps["pass_fraction"],
        },
        "phase0_go": verdict,
    }


def shuffled_frame_assignment(num_frames: int, num_cameras: int, seed: int) -> np.ndarray:
    """Per-camera fixed pseudorandom frame permutation; (C, T) int array."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(num_frames) for _ in range(num_cameras)], axis=0)


# ---------------------------------------------------------------------------
# Census-v2: gap-aware classification and certified-reveal state machine
# ---------------------------------------------------------------------------

def classify_states_v2(
    z: np.ndarray,
    pixels: np.ndarray,
    in_view: np.ndarray,
    present: np.ndarray,
    d_map: np.ndarray,
    sigma_map: np.ndarray,
    valid_map: np.ndarray,
    *,
    tau_rel: float,
    kappa: float,
    k_gap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-primitive state plus gap-occlusion evidence for one (camera, frame).

    Returns (states int8, gap_raw bool, occ_depth f32 NaN-filled,
    occ_sigma f32 NaN-filled). gap_raw = BEHIND with (z - d) >= k_gap * margin
    (witness is applied by the caller from the full state matrix).
    """
    n = z.shape[0]
    states = np.zeros(n, dtype=np.int8)
    gap_raw = np.zeros(n, dtype=bool)
    occ_depth = np.full(n, np.nan, dtype=np.float32)
    occ_sigma = np.full(n, np.nan, dtype=np.float32)
    usable = in_view & present
    if not usable.any():
        return states, gap_raw, occ_depth, occ_sigma
    px = pixels[usable, 0]
    py = pixels[usable, 1]
    pix_valid = valid_map[py, px]
    idx = np.flatnonzero(usable)[pix_valid]
    if idx.size == 0:
        return states, gap_raw, occ_depth, occ_sigma
    px = pixels[idx, 0]
    py = pixels[idx, 1]
    d = d_map[py, px].astype(np.float64)
    sig = sigma_map[py, px].astype(np.float64)
    margin = np.maximum(tau_rel * d, kappa * sig)
    delta = z[idx] - d
    state = np.full(idx.shape[0], STATE_NEAR_SURFACE, dtype=np.int8)
    state[delta > margin] = STATE_BEHIND
    state[delta < -margin] = STATE_IN_FRONT
    states[idx] = state
    gap_raw[idx] = delta >= k_gap * margin
    occ_depth[idx] = d
    occ_sigma[idx] = sig
    return states, gap_raw, occ_depth, occ_sigma


@dataclass
class CertifiedRevealTracker:
    """Vectorized anchored-hysteresis certification per (primitive, camera).

    Rule (frozen in the census-v2 preregistration): an anchor of
    `anchor_consec` consecutive near-surface frames, entry after
    `entry_consec` consecutive gap-occluded (witnessed, coherent) frames, a
    run totalling >= `min_gap_occ_frames` gap-occluded frames with at most
    `grace_frames` interruption frames, certification on `reveal_consec`
    consecutive near-surface frames. Occluder-depth coherence: an accepted
    gap frame must satisfy |d - d_prev| <= max(smooth_rel*d,
    smooth_kappa*sigma) against the previous accepted gap frame of the same
    streak. Pairs where `eligible` is False never certify.
    """

    num_primitives: int
    num_cameras: int
    anchor_consec: int = 2
    entry_consec: int = 2
    reveal_consec: int = 2
    min_gap_occ_frames: int = 3
    grace_frames: int = 1
    smooth_rel: float = 0.2
    smooth_kappa: float = 5.0
    sample_cap: int = 0
    eligible: np.ndarray | None = None

    def __post_init__(self) -> None:
        shape = (self.num_primitives, self.num_cameras)
        if self.eligible is None:
            self.eligible = np.ones(shape, dtype=bool)
        if self.eligible.shape != shape:
            raise ValueError("eligible mask shape mismatch")
        self.anchored = np.zeros(shape, dtype=bool)
        self.near_consec = np.zeros(shape, dtype=np.int16)
        self.occ_consec = np.zeros(shape, dtype=np.int16)
        self.in_occ = np.zeros(shape, dtype=bool)
        self.run_total = np.zeros(shape, dtype=np.int16)
        self.grace_left = np.zeros(shape, dtype=np.int8)
        self.last_occ_depth = np.full(shape, np.nan, dtype=np.float32)
        self.certified_pairs = np.zeros(shape, dtype=bool)
        self.certified_total = 0
        self.end_frames: set = set()
        self.per_camera = np.zeros(self.num_cameras, dtype=np.int64)
        self.run_length_histogram: dict = {}
        self.completions_by_frame: dict = {}
        self.samples: list = []

    def update(
        self,
        frame: int,
        near: np.ndarray,
        gap_raw: np.ndarray,
        occ_depth: np.ndarray,
        occ_sigma: np.ndarray,
    ) -> None:
        smooth_ok = np.isnan(self.last_occ_depth) | (
            np.abs(occ_depth - self.last_occ_depth)
            <= np.maximum(self.smooth_rel * occ_depth, self.smooth_kappa * occ_sigma)
        )
        gap_occ = gap_raw & smooth_ok & self.eligible

        prev_near_consec = self.near_consec.copy()
        self.near_consec = np.where(near, self.near_consec + 1, 0).astype(np.int16)
        self.occ_consec = np.where(gap_occ, self.occ_consec + 1, 0).astype(np.int16)

        entering = self.anchored & (~self.in_occ) & (self.occ_consec >= self.entry_consec)
        if entering.any():
            self.in_occ |= entering
            self.run_total[entering] = self.occ_consec[entering]
            self.grace_left[entering] = self.grace_frames
            self.anchored &= ~entering

        active = self.in_occ & ~entering
        continuing = active & gap_occ
        self.run_total[continuing] += 1

        revealing = active & near & (self.near_consec >= self.reveal_consec)
        certify = revealing & (self.run_total >= self.min_gap_occ_frames) & self.eligible
        if certify.any():
            count = int(certify.sum())
            self.certified_total += count
            self.certified_pairs |= certify
            self.end_frames.add(int(frame))
            self.per_camera += certify.sum(axis=0)
            self.completions_by_frame[int(frame)] = (
                self.completions_by_frame.get(int(frame), 0) + count
            )
            lengths, counts = np.unique(self.run_total[certify], return_counts=True)
            for length, cnt in zip(lengths.tolist(), counts.tolist()):
                key = int(length)
                self.run_length_histogram[key] = self.run_length_histogram.get(key, 0) + int(cnt)
            if self.sample_cap and len(self.samples) < self.sample_cap:
                prim_idx, cam_idx = np.nonzero(certify)
                budget = self.sample_cap - len(self.samples)
                for p, c in zip(prim_idx[:budget].tolist(), cam_idx[:budget].tolist()):
                    self.samples.append({
                        "primitive": int(p),
                        "camera_index": int(c),
                        "end_frame": int(frame),
                        "gap_occ_frames": int(self.run_total[p, c]),
                    })

        # Interruption accounting. A near frame below reveal_consec is PENDING
        # (no charge yet): it becomes one interruption only if its streak
        # breaks before reaching reveal_consec. A frame that is neither near
        # nor an accepted gap frame is a hard interruption. Charges beyond the
        # remaining grace abort the run without certification.
        broken_near = active & (~near) & (prev_near_consec >= 1)
        hard_interruption = active & (~near) & (~gap_occ)
        charge = broken_near.astype(np.int8) + hard_interruption.astype(np.int8)
        over_budget = active & (charge > self.grace_left)
        charged = active & (charge > 0) & ~over_budget
        self.grace_left[charged] = (self.grace_left[charged] - charge[charged]).astype(np.int8)
        aborting = over_budget

        ending = revealing | aborting
        if ending.any():
            self.in_occ &= ~ending
            self.run_total[ending] = 0
            self.grace_left[ending] = 0
            self.last_occ_depth[ending] = np.nan

        self.last_occ_depth = np.where(gap_occ, occ_depth, self.last_occ_depth)
        streak_broken = (~self.in_occ) & (~gap_occ)
        self.last_occ_depth[streak_broken] = np.nan

        self.anchored |= self.near_consec >= self.anchor_consec

    def summary(self, camera_ids: list[str]) -> dict[str, Any]:
        return {
            "certified_events": int(self.certified_total),
            "certified_pairs": int(self.certified_pairs.sum()),
            "distinct_end_frames": len(self.end_frames),
            "distinct_end_cameras": int((self.per_camera > 0).sum()),
            "per_camera_certified": {
                camera_ids[c]: int(self.per_camera[c]) for c in range(self.num_cameras)
            },
            "run_length_histogram": {
                str(k): v for k, v in sorted(self.run_length_histogram.items())
            },
            "completions_by_frame": {
                str(k): v for k, v in sorted(self.completions_by_frame.items())
            },
        }


def quartile_labels(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Label each value 0..3 by quartile; returns (labels, [q25, q50, q75])."""
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape[0], dtype=np.int8), [float("nan")] * 3
    qs = [float(np.percentile(values[finite], p)) for p in (25.0, 50.0, 75.0)]
    labels = np.zeros(values.shape[0], dtype=np.int8)
    labels[values > qs[0]] = 1
    labels[values > qs[1]] = 2
    labels[values > qs[2]] = 3
    labels[~finite] = 0
    return labels, qs


def evaluate_census2_floors(summary: Mapping[str, Any], floors: Mapping[str, Any]) -> dict[str, Any]:
    valid = summary["valid"]
    shuffle = summary["shuffle"]
    consistency = summary["consistency"]
    maps = summary["consensus_maps"]

    g1 = (
        valid["certified_pairs"] >= floors["g1_min_certified_pairs"]
        and valid["distinct_end_frames"] >= floors["g1_min_distinct_end_frames"]
        and valid["distinct_end_cameras"] >= floors["g1_min_distinct_cameras"]
    )
    shuffle_pairs = max(int(shuffle["certified_pairs"]), 0)
    if shuffle_pairs == 0:
        rho = float("inf") if valid["certified_pairs"] > 0 else 0.0
        g2 = valid["certified_pairs"] >= floors["g1_min_certified_pairs"]
    else:
        rho = valid["certified_pairs"] / shuffle_pairs
        g2 = rho >= floors["g2_min_valid_over_shuffle"]
    evaluated = max(int(consistency["evaluated"]), 1)
    conflict_fraction = consistency["conflict"] / evaluated
    g3 = (
        conflict_fraction <= floors["g3_max_conflict_fraction"]
        and maps["pass_fraction"] >= floors["g3_min_map_pass_fraction"]
    )
    per_camera = valid["per_camera_certified"]
    total = max(sum(per_camera.values()), 1)
    max_share = max(per_camera.values()) / total if per_camera else 1.0
    g4 = max_share <= floors["g4_max_camera_share"]
    verdict = bool(g1 and g2 and g3 and g4)
    return {
        "g1_certified_abundance": {
            "pass": bool(g1),
            "pairs": valid["certified_pairs"],
            "distinct_end_frames": valid["distinct_end_frames"],
            "distinct_end_cameras": valid["distinct_end_cameras"],
        },
        "g2_control_separation": {
            "pass": bool(g2),
            "valid_pairs": valid["certified_pairs"],
            "shuffle_pairs": shuffle_pairs,
            "rho": rho if np.isfinite(rho) else "inf",
        },
        "g3_evidence_validity": {
            "pass": bool(g3),
            "conflict_fraction": conflict_fraction,
            "map_pass_fraction": maps["pass_fraction"],
        },
        "g4_non_degeneracy": {"pass": bool(g4), "max_camera_share": max_share},
        "census2_go": verdict,
    }
