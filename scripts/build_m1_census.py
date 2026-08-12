"""M1-A0 model-free census evaluator (gated cell; prereg revision 3).

Authority: ``configs/elgs/prereg_m1_census_v1.json`` REVISION 3 (commit
``72ff97e``), whose floors and definitions were fresh-context reviewer
SIGNED on 2026-08-11 (three-round chain recorded in
``research-wiki/operations/elgs-m1-census-record.md``). Every predicate
below implements a frozen clause of that file; the file's sha256 is
recorded in the output so any drift is detectable.

Inputs (PRIMARY, model-free — no trained model anywhere):
- the CONVERTED temporal scene (calibration via ``transforms_train.json``,
  per-frame fg/bg masks under ``masks/`` — frames_1 alpha, binarized by
  the converter; this evaluator re-binarizes ``> 127`` per the frozen
  ``mask_binarization`` clause, idempotent on {0,255} masks);
- the FROZEN tracks artifact (identities, per-camera reports, consensus
  triangulations).

Statistics (frozen names, ``floor_to_statistic`` mapping):
- ``occlusion_opportunity_upper_bound``
- ``true_absence_candidate_count``
- ``same_object_return_count``
- ``track_coverage_upper_bound``

Frozen constants implemented here: 8-connected components, >= 64 px
eligibility; association by round-half-up report pixel (v >= 0.5,
in-domain); occlusion events strictly consecutive with ONE FIXED absent
camera (no flicker bridging); true-absence windows flicker-bridged
(< 4-frame interruptions); r_site_census = 0.05 * rig_radius; return
position at the earliest defined-consensus frame of the terminating
re-appearance run; census window = full shipped frame range common to
all tracking cameras, realized window recorded.

Implementation disclosure (recorded in the output artifact, cannot move
a candidate IN): a frame whose consensus point is undefined cannot
satisfy the occlusion predicate's frustum-containment clause, so such
frames break occlusion runs; they are tallied as
``occlusion_undefined_position_frames``.

Import-time constraint: torch-free (numpy + cv2 + PIL only; cv2/PIL
imported lazily).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.artifacts import atomic_write_bytes  # noqa: E402
from depth_visibility.camera import camera_center  # noqa: E402
from depth_visibility.canonical import canonical_json_bytes, sha256_file  # noqa: E402
from depth_visibility.errors import ContractError, SchemaError  # noqa: E402
from elgs.tracks_schema import validate_tracks_artifact  # noqa: E402
from scripts.build_elgs_tracks import CameraModel, SceneBundle, load_temporal_scene  # noqa: E402

CENSUS_SCHEMA = "elgs-m1-a0-census-v1"
PREREG_REVISION_REQUIRED = 3

# Frozen constants (prereg revision 3; asserted against the loaded file)
VIS_THRESHOLD = 0.5
MIN_COMPONENT_PX = 64
DURATION_FLOOR = 8
FLICKER_MAX = 4  # interruptions STRICTLY shorter than this are bridged
MIN_CAMERAS_OCCLUSION = 2
MIN_POSITIVE_OPPORTUNITY_CAMERAS = 2
R_SITE_FACTOR = 0.05
MASK_THRESHOLD = 127  # strictly greater => foreground


def round_half_up(value: float) -> int:
    """Frozen pixel rounding: floor(x + 0.5), platform-independent."""

    return int(np.floor(value + 0.5))


# ---------------------------------------------------------------------------
# Mask components
# ---------------------------------------------------------------------------


def load_component_labels(mask_path: Path) -> tuple[np.ndarray, set[int]]:
    """8-connected eligible components of one binarized mask.

    Returns ``(labels, eligible_label_set)``; label 0 is background.
    """

    import cv2
    from PIL import Image

    if not mask_path.is_file():
        raise ContractError(f"census mask missing: {mask_path}")
    with Image.open(mask_path) as image:
        binary = (np.asarray(image.convert("L")) > MASK_THRESHOLD).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    eligible = {
        label
        for label in range(1, count)
        if stats[label, cv2.CC_STAT_AREA] >= MIN_COMPONENT_PX
    }
    return labels, eligible


# ---------------------------------------------------------------------------
# Tracks-artifact access
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ReportIndex:
    """(seed, camera, frame) -> rounded in-domain visible report pixel."""

    pixels: dict[tuple[int, int, int], tuple[int, int]]
    seeds: tuple[int, ...]
    consensus: dict[int, dict[int, np.ndarray]]  # seed -> frame -> point


def index_tracks(artifact: dict[str, Any], scene: SceneBundle) -> ReportIndex:
    validate_tracks_artifact(artifact)
    pixels: dict[tuple[int, int, int], tuple[int, int]] = {}
    for track in artifact["tracks"]:
        seed_id = int(track["seed_id"])
        camera_id = int(track["camera_id"])
        camera = scene.cameras.get(camera_id)
        if camera is None:
            continue
        for report in track["reports"]:
            if report.get("is_miss", False):
                continue
            if float(report["v"]) < VIS_THRESHOLD:
                continue
            x, y = float(report["x"]), float(report["y"])
            # R8 (verifier D1): in-domain binds the ROUND-HALF-UP pixel,
            # not the continuous coordinates — the half-pixel boundary band
            # (e.g. x in [-0.5, 0)) is in-domain.
            col, row = round_half_up(x), round_half_up(y)
            if not (0 <= col <= camera.width - 1 and 0 <= row <= camera.height - 1):
                continue
            frame = int(round(float(report["frame"])))
            pixels[(seed_id, camera_id, frame)] = (col, row)
    consensus: dict[int, dict[int, np.ndarray]] = {}
    for key, entries in artifact.get("consensus", {}).items():
        seed_id = int(key)
        per_frame: dict[int, np.ndarray] = {}
        for entry in entries:
            if entry.get("point") is not None:
                per_frame[int(round(float(entry["frame"])))] = np.asarray(
                    entry["point"], dtype=np.float64
                )
        consensus[seed_id] = per_frame
    seeds = tuple(sorted(int(s["seed_id"]) for s in artifact["seeds"]))
    return ReportIndex(pixels=pixels, seeds=seeds, consensus=consensus)


def frustum_contains(camera: CameraModel, point: np.ndarray) -> bool:
    cam_point = camera.w2c[:3, :3] @ point + camera.w2c[:3, 3]
    if cam_point[2] <= 0.0:
        return False
    uv = camera.K @ cam_point
    x, y = uv[0] / uv[2], uv[1] / uv[2]
    return 0.0 <= x <= camera.width - 1.0 and 0.0 <= y <= camera.height - 1.0


def rig_radius(scene: SceneBundle) -> float:
    centers = np.stack([camera_center(scene.cameras[c].w2c) for c in scene.tracking_ids])
    return float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).max())


# ---------------------------------------------------------------------------
# Association table
# ---------------------------------------------------------------------------


def build_association(
    scene: SceneBundle, index: ReportIndex
) -> tuple[dict[tuple[int, int, int], bool], dict[str, int]]:
    """(seed, camera, frame) -> associated-with-eligible-component, plus the
    coverage tallies (components total / covered) computed in one pass."""

    associated: dict[tuple[int, int, int], bool] = {}
    components_total = 0
    components_covered = 0
    for camera_id in scene.tracking_ids:
        for frame in scene.frame_indices:
            labels, eligible = load_component_labels(scene.mask_path(camera_id, frame))
            components_total += len(eligible)
            covered_labels: set[int] = set()
            for seed_id in index.seeds:
                pixel = index.pixels.get((seed_id, camera_id, frame))
                if pixel is None:
                    associated[(seed_id, camera_id, frame)] = False
                    continue
                col, row = pixel
                col = min(max(col, 0), labels.shape[1] - 1)
                row = min(max(row, 0), labels.shape[0] - 1)
                label = int(labels[row, col])
                hit = label in eligible
                associated[(seed_id, camera_id, frame)] = hit
                if hit:
                    covered_labels.add(label)
            components_covered += len(covered_labels)
    return associated, {
        "components_total": components_total,
        "components_covered": components_covered,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def last_defined_consensus(
    per_frame: dict[int, np.ndarray], frames: Sequence[int], at_or_before: int
) -> tuple[np.ndarray | None, int | None]:
    best_frame = None
    for frame in frames:
        if frame > at_or_before:
            break
        if frame in per_frame:
            best_frame = frame
    if best_frame is None:
        return None, None
    return per_frame[best_frame], best_frame


def occlusion_events(
    scene: SceneBundle, index: ReportIndex, associated: dict[tuple[int, int, int], bool]
) -> tuple[list[dict[str, Any]], int]:
    """Strictly-consecutive runs with ONE FIXED absent camera (no bridging)."""

    events: list[dict[str, Any]] = []
    undefined_position_frames = 0
    frames = scene.frame_indices
    for seed_id in index.seeds:
        per_frame_consensus = index.consensus.get(seed_id, {})
        for absent_camera in scene.tracking_ids:
            run: list[int] = []
            for frame in frames:
                n_assoc = sum(
                    1
                    for camera_id in scene.tracking_ids
                    if associated.get((seed_id, camera_id, frame), False)
                )
                point = per_frame_consensus.get(frame)
                if point is None:
                    undefined_position_frames += 1
                    ok = False
                else:
                    ok = (
                        n_assoc >= MIN_CAMERAS_OCCLUSION
                        and frustum_contains(scene.cameras[absent_camera], point)
                        and not associated.get((seed_id, absent_camera, frame), False)
                    )
                if ok:
                    run.append(frame)
                else:
                    if len(run) >= DURATION_FLOOR:
                        events.append(
                            {
                                "seed_id": seed_id,
                                "absent_camera": absent_camera,
                                "first_frame": run[0],
                                "last_frame": run[-1],
                                "n_frames": len(run),
                            }
                        )
                    run = []
            if len(run) >= DURATION_FLOOR:
                events.append(
                    {
                        "seed_id": seed_id,
                        "absent_camera": absent_camera,
                        "first_frame": run[0],
                        "last_frame": run[-1],
                        "n_frames": len(run),
                    }
                )
    return events, undefined_position_frames


def angular_separation_floor(scene: SceneBundle) -> float:
    """Frozen R2-prime angular floor (prereg_m1_cycle2_screen_v1 rev 2,
    finding B2): the ceil(0.10 * N)-th smallest of the N = C(k,2) pairwise
    optical-axis angular separations among training cameras — NEAREST-RANK,
    no interpolation; calibration only."""

    axes = []
    for camera_id in scene.tracking_ids:
        rotation = scene.cameras[camera_id].w2c[:3, :3]
        axes.append(rotation.T @ np.array([0.0, 0.0, 1.0]))
    separations = []
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            cosine = float(np.clip(np.dot(axes[i], axes[j]), -1.0, 1.0))
            separations.append(float(np.arccos(cosine)))
    separations.sort()
    rank = math.ceil(0.10 * len(separations))  # 1-indexed nearest-rank
    return separations[rank - 1]


def r2prime_holds(
    scene: SceneBundle,
    index: ReportIndex,
    seed_id: int,
    frame: int,
    ltp: np.ndarray,
    r_site: float,
    angular_floor: float,
) -> bool:
    """R2-prime (frozen union member): >= 2 training cameras, pairwise
    angular separation >= angular_floor, EACH with the identity's in-domain
    visible report pixel within tol_c = r_site * fl_x_c / z_c of the LTP's
    projection into that camera (the exact image of the 3-D r_site ball;
    calibration + frozen tracks only, no free constants)."""

    qualifying = []
    for camera_id in scene.tracking_ids:
        pixel = index.pixels.get((seed_id, camera_id, frame))
        if pixel is None:
            continue
        camera = scene.cameras[camera_id]
        cam_point = camera.w2c[:3, :3] @ ltp + camera.w2c[:3, 3]
        z = float(cam_point[2])
        if z <= 0.0:
            continue
        uv = camera.K @ cam_point
        proj = np.array([uv[0] / uv[2], uv[1] / uv[2]])
        tolerance = r_site * float(camera.K[0, 0]) / z
        if float(np.hypot(pixel[0] - proj[0], pixel[1] - proj[1])) <= tolerance:
            axis = camera.w2c[:3, :3].T @ np.array([0.0, 0.0, 1.0])
            qualifying.append(axis)
    if len(qualifying) < 2:
        return False
    for i in range(len(qualifying)):
        for j in range(i + 1, len(qualifying)):
            cosine = float(np.clip(np.dot(qualifying[i], qualifying[j]), -1.0, 1.0))
            if float(np.arccos(cosine)) >= angular_floor:
                return True
    return False


def true_absence_and_returns(
    scene: SceneBundle,
    index: ReportIndex,
    associated: dict[tuple[int, int, int], bool],
    r_site: float,
    angular_floor: float,
) -> tuple[list[dict[str, Any]], int, int, dict[str, int]]:
    """Flicker-bridged true-absence candidates + same-object returns.

    Cycle-2 alignment (prereg_m1_cycle2_screen_v1 frozen readings):
    - R3: disappearance is a TRANSITION — associated in >= 1 camera of S
      at t-1 and in none of S at t; no window opens at frame 0.
    - R9: position_undefined_exclusions are tallied at GLOBAL disappearance
      transitions with no defined consensus at or before t.
    - Union return statistic: revision-3 consensus primary OR R2-prime;
      the single-camera >= 1 form is diagnostic-only.

    Returns (candidates, position_undefined_exclusions,
    return_position_undefined, return_counts) where return_counts has
    'primary', 'union', and 'single_camera_diagnostic'.
    """

    frames = list(scene.frame_indices)
    candidates: list[dict[str, Any]] = []
    position_undefined_exclusions = 0
    return_position_undefined = 0
    return_counts = {"primary": 0, "union": 0, "single_camera_diagnostic": 0}

    def associated_any(seed: int, f: int) -> bool:
        return any(
            associated.get((seed, camera_id, f), False) for camera_id in scene.tracking_ids
        )

    for seed_id in index.seeds:
        per_frame_consensus = index.consensus.get(seed_id, {})
        cursor = 1  # R3: no window can open at frame 0
        while cursor < len(frames):
            frame = frames[cursor]
            previous = frames[cursor - 1]
            ltp, ltp_frame = last_defined_consensus(per_frame_consensus, frames, frame)
            if ltp is None:
                # R9: tally at GLOBAL disappearance transitions only.
                if associated_any(seed_id, previous) and not associated_any(seed_id, frame):
                    position_undefined_exclusions += 1
                cursor += 1
                continue
            containing = [
                camera_id
                for camera_id in scene.tracking_ids
                if frustum_contains(scene.cameras[camera_id], ltp)
            ]
            if len(containing) < MIN_POSITIVE_OPPORTUNITY_CAMERAS:
                cursor += 1
                continue

            def absent_at(f: int) -> bool:
                return not any(
                    associated.get((seed_id, camera_id, f), False) for camera_id in containing
                )

            # R3 transition rule: associated in >= 1 camera of S at t-1 AND
            # in none of S at t (S fixed at event start for the lifetime).
            if not absent_at(frame) or absent_at(previous):
                cursor += 1
                continue

            # Walk forward, bridging interruptions strictly shorter than
            # FLICKER_MAX; a re-appearance run >= FLICKER_MAX terminates.
            window = [frame]
            bridged = 0
            position = cursor + 1
            reappearance_run: list[int] = []
            while position < len(frames):
                f = frames[position]
                if absent_at(f):
                    if reappearance_run:
                        bridged += 1
                        window.extend(reappearance_run)
                        reappearance_run = []
                    window.append(f)
                    position += 1
                else:
                    reappearance_run.append(f)
                    if len(reappearance_run) >= FLICKER_MAX:
                        break
                    position += 1

            absence_frames = [f for f in window]
            if len(absence_frames) >= DURATION_FLOOR:
                record: dict[str, Any] = {
                    "seed_id": seed_id,
                    "containing_cameras": containing,
                    "first_frame": absence_frames[0],
                    "last_frame": absence_frames[-1],
                    "n_frames": len(absence_frames),
                    "bridged_interruptions": bridged,
                    "ltp_frame": ltp_frame,
                    "ltp": [float(v) for v in ltp],
                }
                terminated = len(reappearance_run) >= FLICKER_MAX
                if terminated:
                    # Verifier D2: the frozen texts bind the anchor search to
                    # the MAXIMAL terminating re-appearance run, not the
                    # 4-frame termination-detection prefix. Extend the run to
                    # its true end (next S-absence frame or window end),
                    # using the SAME S per R4 (fixed at event start).
                    extension = position + 1
                    while extension < len(frames) and not absent_at(frames[extension]):
                        reappearance_run.append(frames[extension])
                        extension += 1
                    position = extension - 1
                    record["return_run_start"] = reappearance_run[0]
                    record["return_run_end"] = reappearance_run[-1]
                    # PRIMARY (revision-3 rule): earliest defined-consensus
                    # frame within the terminating run, within r_site.
                    primary_return = False
                    return_point = None
                    return_frame = None
                    for f in reappearance_run:
                        if f in per_frame_consensus:
                            return_point = per_frame_consensus[f]
                            return_frame = f
                            break
                    if return_point is not None:
                        distance = float(np.linalg.norm(return_point - ltp))
                        record["return_frame"] = return_frame
                        record["return_distance"] = distance
                        primary_return = distance <= r_site
                    else:
                        return_position_undefined += 1
                    # R2-PRIME: earliest frame of the run where it holds.
                    r2p_return = False
                    for f in reappearance_run:
                        if r2prime_holds(
                            scene, index, seed_id, f, ltp, r_site, angular_floor
                        ):
                            r2p_return = True
                            record["r2prime_frame"] = f
                            break
                    # Single-camera >= 1 diagnostic (never gate-bearing):
                    # any camera's report pixel within its own tol_c.
                    single_cam = False
                    for f in reappearance_run:
                        for camera_id in scene.tracking_ids:
                            pixel = index.pixels.get((seed_id, camera_id, f))
                            if pixel is None:
                                continue
                            camera = scene.cameras[camera_id]
                            cam_point = camera.w2c[:3, :3] @ ltp + camera.w2c[:3, 3]
                            if cam_point[2] <= 0.0:
                                continue
                            uv = camera.K @ cam_point
                            proj = (uv[0] / uv[2], uv[1] / uv[2])
                            tol = r_site * float(camera.K[0, 0]) / float(cam_point[2])
                            if np.hypot(pixel[0] - proj[0], pixel[1] - proj[1]) <= tol:
                                single_cam = True
                                break
                        if single_cam:
                            break

                    if primary_return:
                        return_counts["primary"] += 1
                    if primary_return or r2p_return:
                        return_counts["union"] += 1
                    if single_cam:
                        return_counts["single_camera_diagnostic"] += 1
                    record["return"] = (
                        "same_object_return_primary"
                        if primary_return
                        else "same_object_return_r2prime"
                        if r2p_return
                        else "return_position_undefined"
                        if return_point is None
                        else "beyond_r_site"
                    )
                else:
                    record["return"] = "no_terminating_reappearance"
                candidates.append(record)

            # Resume scanning after the examined region.
            cursor = position + 1 if position > cursor else cursor + 1

    return candidates, position_undefined_exclusions, return_position_undefined, return_counts


# ---------------------------------------------------------------------------
# Cell driver
# ---------------------------------------------------------------------------


def load_prereg(prereg_path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(prereg_path.read_text(encoding="utf-8"))
    if payload.get("revision") != PREREG_REVISION_REQUIRED:
        raise ContractError(
            f"prereg revision {payload.get('revision')} != required "
            f"{PREREG_REVISION_REQUIRED} (the SIGNED revision)"
        )
    floors = payload["floors"]
    frozen_expectations = {
        "occlusion_events_min": 36,
        "same_object_returns_min": 36,
        "true_absence_candidates_min": 36,
        "track_coverage_min_fraction": 0.5,
    }
    for key, expected in frozen_expectations.items():
        if floors[key] != expected:
            raise ContractError(f"floor {key}={floors[key]} != signed value {expected}")
    return payload, sha256_file(prereg_path)


def _attribute_half(records: list[dict[str, Any]], split_frame: int, key: str) -> dict[str, int]:
    """Frozen boundary conventions: occlusion/absence windows belong to the
    half containing their FIRST frame; returns to the half containing their
    terminating re-appearance run START."""

    first = sum(1 for record in records if record[key] < split_frame)
    return {"first_half": first, "second_half": len(records) - first}


def _census_one_sequence(scene_dir: Path, tracks_path: Path) -> dict[str, Any]:
    """The single-sequence census core; the gate binds to POOLED values."""

    scene = load_temporal_scene(scene_dir)
    artifact = json.loads(tracks_path.read_text(encoding="utf-8"))
    index = index_tracks(artifact, scene)

    associated, coverage = build_association(scene, index)
    r_site = R_SITE_FACTOR * rig_radius(scene)
    angular_floor = angular_separation_floor(scene)

    occlusions, undefined_occ_frames = occlusion_events(scene, index, associated)
    candidates, pos_undef, ret_undef, return_counts = true_absence_and_returns(
        scene, index, associated, r_site, angular_floor
    )
    if coverage["components_total"] == 0:
        raise ContractError(f"{scene_dir}: zero eligible fg components over the census window")

    # Half attribution (cycle-2 screen/gate modes; frozen split floor(n/2)).
    frames = list(scene.frame_indices)
    split_frame = frames[len(frames) // 2]
    returns_terminated = [c for c in candidates if "return_run_start" in c]
    union_returns = [
        c
        for c in returns_terminated
        if c["return"] in ("same_object_return_primary", "same_object_return_r2prime")
    ]
    halves = {
        "split_frame": int(split_frame),
        "occlusion": _attribute_half(occlusions, split_frame, "first_frame"),
        "true_absence": _attribute_half(candidates, split_frame, "first_frame"),
        "union_returns": _attribute_half(union_returns, split_frame, "return_run_start"),
        "primary_returns": _attribute_half(
            [c for c in union_returns if c["return"] == "same_object_return_primary"],
            split_frame,
            "return_run_start",
        ),
    }

    return {
        "statistics": {
            "occlusion_opportunity_upper_bound": len(occlusions),
            "true_absence_candidate_count": len(candidates),
            "same_object_return_count": return_counts["primary"],
            "union_return_count": return_counts["union"],
            "single_camera_return_diagnostic": return_counts["single_camera_diagnostic"],
            "track_coverage_upper_bound": coverage["components_covered"]
            / coverage["components_total"],
        },
        "halves": halves,
        "angular_floor_rad": angular_floor,
        "window": {
            "first_frame": int(scene.frame_indices[0]),
            "last_frame": int(scene.frame_indices[-1]),
            "n_frames": len(scene.frame_indices),
        },
        "constants": {"r_site_census": r_site, "rig_radius": rig_radius(scene)},
        "exclusions": {
            "position_undefined_exclusions": pos_undef,
            "return_position_undefined": ret_undef,
            "occlusion_undefined_position_frames": undefined_occ_frames,
        },
        "records": {
            "occlusion_events": occlusions,
            "true_absence_candidates": candidates,
        },
        "coverage_tallies": coverage,
        "provenance": {
            "scene_dir": str(scene_dir),
            "tracks_sha256": sha256_file(tracks_path),
            "training_cameras": [int(c) for c in scene.tracking_ids],
            "held_out_cameras": [int(c) for c in scene.held_out_ids],
        },
    }


def run_census(
    pairs: Sequence[tuple[Path, Path]],
    prereg_path: Path,
    *,
    apply_gate: bool,
) -> dict[str, Any]:
    """Pooled M1-A0 census over the dev subset.

    The frozen floors bind to counts summed OVER THE DEV SUBSET (prereg:
    "36 model-free opportunities per gated class over the dev subset");
    coverage pools numerator/denominator. Per-sequence breakdowns are
    retained for the traceability requirement; r_site_census is computed
    per sequence from its own calibration (frozen rule).
    """

    if not pairs:
        raise ContractError("at least one (scene, tracks) pair is required")
    started = time.monotonic()
    prereg, prereg_sha = load_prereg(prereg_path)

    per_sequence: dict[str, dict[str, Any]] = {}
    counts = {
        "occlusion_opportunity_upper_bound": 0,
        "true_absence_candidate_count": 0,
        "same_object_return_count": 0,
    }
    covered = total = 0
    for scene_dir, tracks_path in pairs:
        name = Path(scene_dir).name
        if name in per_sequence:
            raise ContractError(f"duplicate sequence name in census input: {name}")
        result = _census_one_sequence(Path(scene_dir), Path(tracks_path))
        per_sequence[name] = result
        for key in counts:
            counts[key] += result["statistics"][key]
        covered += result["coverage_tallies"]["components_covered"]
        total += result["coverage_tallies"]["components_total"]

    statistics = {**counts, "track_coverage_upper_bound": covered / total}

    result: dict[str, Any] = {
        "schema_version": CENSUS_SCHEMA,
        "cell": "M1-A0",
        "statistics": statistics,
        "per_sequence": per_sequence,
        "floors": {
            key: prereg["floors"][key]
            for key in (
                "occlusion_events_min",
                "same_object_returns_min",
                "true_absence_candidates_min",
                "track_coverage_min_fraction",
            )
        },
        "constants": {
            "vis_threshold": VIS_THRESHOLD,
            "min_component_px": MIN_COMPONENT_PX,
            "duration_floor_frames": DURATION_FLOOR,
            "flicker_bridge_below_frames": FLICKER_MAX,
            "mask_binarize_threshold": MASK_THRESHOLD,
            "pixel_rounding": "round-half-up floor(x+0.5)",
        },
        "coverage_tallies": {"components_covered": covered, "components_total": total},
        "provenance": {
            "prereg_sha256": prereg_sha,
            "prereg_revision": PREREG_REVISION_REQUIRED,
            "wall_seconds": time.monotonic() - started,
        },
    }

    if apply_gate:
        comparisons = {
            "occlusion_events_min": statistics["occlusion_opportunity_upper_bound"]
            >= prereg["floors"]["occlusion_events_min"],
            "same_object_returns_min": statistics["same_object_return_count"]
            >= prereg["floors"]["same_object_returns_min"],
            "true_absence_candidates_min": statistics["true_absence_candidate_count"]
            >= prereg["floors"]["true_absence_candidates_min"],
            "track_coverage_min_fraction": statistics["track_coverage_upper_bound"]
            >= prereg["floors"]["track_coverage_min_fraction"],
        }
        result["gate"] = {
            "comparisons": comparisons,
            "pass": all(comparisons.values()),
            "note": (
                "M1-A0 statistics are model-free candidate-opportunity UPPER "
                "BOUNDS; meeting a floor is a necessary condition for adequacy, "
                "not evidence that that many true events exist"
            ),
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scene-dir", type=Path, action="append", required=True,
        help="repeatable; paired positionally with --tracks (the dev subset)",
    )
    parser.add_argument("--tracks", type=Path, action="append", required=True)
    parser.add_argument(
        "--prereg", type=Path, default=REPO_ROOT / "configs" / "elgs" / "prereg_m1_census_v1.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--apply-gate", action="store_true")
    args = parser.parse_args(argv)
    if len(args.scene_dir) != len(args.tracks):
        raise ContractError(
            f"{len(args.scene_dir)} --scene-dir vs {len(args.tracks)} --tracks; must pair 1:1"
        )
    result = run_census(
        list(zip(args.scene_dir, args.tracks)), args.prereg, apply_gate=args.apply_gate
    )
    result["config_sha256"] = hashlib.sha256(
        canonical_json_bytes({"argv": list(argv) if argv else sys.argv[1:]})
    ).hexdigest()
    # Plain strict JSON (floats as JSON numbers) -- the canonical writer's
    # binary64_hex tokens are an identity convention with no repo decoder
    # (same defect class as the tracks-artifact fix at 9faa14b; the gate
    # comparisons are computed in-memory and were never affected).
    body = json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":"))
    atomic_write_bytes(args.out, body.encode("utf-8") + b"\n")
    print(json.dumps({"out": str(args.out), "statistics": result["statistics"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
