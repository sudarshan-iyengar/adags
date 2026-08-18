"""Independent fresh-context recomputation of the coverage bounding pair (A1).

Frozen source: ``research-wiki/operations/elgs-coverage-bounding-pair-design.md``
sections 3-5, plus the section-4 sensitivity readings. This module is
written from that frozen text and the primary inputs (converted temporal
scene + frozen tracks artifact) ONLY, per the design's section 8
independent-recomputation charter. It does not import, and was not written
by reading, ``scripts/build_coverage_bounding_pair.py`` or any
``bounding_pair.json`` output.

It reuses ``scripts/build_elgs_tracks.py``'s ``load_temporal_scene`` /
``SceneBundle`` / ``CameraModel`` (generic scene-loading infrastructure, not
the measurement under test) and ``elgs/tracks_schema.py``'s schema
validator. It reproduces ``scripts/build_m1_census.py``'s ``index_tracks``
report-admission filter for reading (i) EXACTLY, because the design
requires that equality as a hard contract check (section 7 #1) and the
filter itself is frozen in ``configs/elgs/prereg_m1_census_v1.json``
(``association_rules.identity_association`` + the round-half-up-pixel
in-domain convention) -- the implementation below derives that filter
independently from those two frozen texts, never by importing
``build_m1_census``'s functions. Component labeling uses
``scipy.ndimage.label`` (8-connectivity), independent of the tested
reducer's ``cv2.connectedComponentsWithStats``.

CLI (repeatable arguments pair 1:1 positionally):
    --scene-dir PATH
    --tracks PATH
    --sequence-name NAME
    --sealed-census PATH
    --out PATH   (single)
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.artifacts import atomic_write_bytes  # noqa: E402
from depth_visibility.errors import ContractError  # noqa: E402
from elgs.tracks_schema import validate_tracks_artifact  # noqa: E402
from scripts.build_elgs_tracks import CameraModel, SceneBundle, load_temporal_scene  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen constants -- derived directly from the design text and the prereg
# it points to (configs/elgs/prereg_m1_census_v1.json), independently of
# scripts/build_m1_census.py's own constant definitions.
# ---------------------------------------------------------------------------

VIS_THRESHOLD = 0.5  # association_rules.identity_association: v >= 0.5
MIN_COMPONENT_PX = 64  # association_rules.component_definition
MASK_BINARIZE_THRESHOLD = 127  # association_rules.mask_binarization (strictly greater)
COVERAGE_FLOOR = 0.5  # design section 5 / prereg track_coverage_min_fraction

_CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)

# Design section 2's two disclosed naming exceptions. Every other sequence
# has no textual naming rule this script can check independently; see the
# "residual definitional freedom" note in the accompanying report.
_KNOWN_NAMING_EXCEPTIONS = {
    "writing_2": {
        "expected_scene_dir_name": "writing_2_screen_w0_239_fix79ae5b7",
        "forbidden_scene_dir_name": "writing_2_screen_w0_239",
    },
    "xylophone": {
        "expected_scene_dir_suffix": "_fix79ae5b7",
    },
}


def round_half_up(value: float) -> int:
    """floor(x + 0.5); platform-independent, per prereg ``pixel_rounding``."""

    return math.floor(float(value) + 0.5)


def load_component_labels(mask_path: Path) -> tuple[np.ndarray, dict[int, int]]:
    """8-connected components of one binarized mask, with per-label pixel counts.

    Returns ``(labels, areas)``: label 0 is background; ``areas`` maps every
    non-background label to its pixel count (NOT pre-filtered by the 64px
    floor -- eligibility filtering happens at the call site, against the
    same denominator rule shared by every reading).
    """

    from PIL import Image

    if not mask_path.is_file():
        raise ContractError(f"mask file missing: {mask_path}")
    with Image.open(mask_path) as image:
        gray = np.asarray(image.convert("L"))
    binary = gray > MASK_BINARIZE_THRESHOLD
    labels, num_labels = ndi.label(binary, structure=_CONNECTIVITY_8)
    if num_labels == 0:
        return labels, {}
    counts = np.bincount(labels.ravel(), minlength=num_labels + 1)
    areas = {label: int(counts[label]) for label in range(1, num_labels + 1)}
    return labels, areas


def project_point(camera: CameraModel, point: np.ndarray) -> tuple[float, float] | None:
    """Pinhole projection through the census's own w2c/K; None if not in front."""

    cam_point = camera.w2c[:3, :3] @ point + camera.w2c[:3, 3]
    z = float(cam_point[2])
    if z <= 0.0:
        return None
    homogeneous = camera.K @ cam_point
    return float(homogeneous[0] / homogeneous[2]), float(homogeneous[1] / homogeneous[2])


@dataclasses.dataclass(frozen=True)
class AnchorLookup:
    status: str  # "ok" | "out_of_domain" | "undefined"
    label: int | None = None


def resolve_anchor(
    camera: CameraModel,
    point: np.ndarray | None,
    labels_array: np.ndarray,
    *,
    transposed: bool,
) -> AnchorLookup:
    """Design section 4 anchor resolution.

    Primary convention: ``labels[row, col]``. Sensitivity (a): the
    transposed ``labels[col, row]``. In both cases the in-domain test
    (0<=col<=W-1, 0<=row<=H-1) is the frozen ``frustum_containment``
    predicate and is unaffected by the indexing convention. The mask array
    need not be square, so the swapped indices can fall outside the
    array's own shape even when in-domain; the design text has no
    resolution for that case, so it is treated the same as
    "cannot be looked up" == out_of_domain (a disclosed residual
    definitional-freedom choice).
    """

    if point is None:
        return AnchorLookup(status="undefined")
    projected = project_point(camera, point)
    if projected is None:
        return AnchorLookup(status="out_of_domain")
    x, y = projected
    col, row = round_half_up(x), round_half_up(y)
    if not (0 <= col <= camera.width - 1 and 0 <= row <= camera.height - 1):
        return AnchorLookup(status="out_of_domain")
    if not transposed:
        return AnchorLookup(status="ok", label=int(labels_array[row, col]))
    if not (0 <= col < labels_array.shape[0] and 0 <= row < labels_array.shape[1]):
        return AnchorLookup(status="out_of_domain")
    return AnchorLookup(status="ok", label=int(labels_array[col, row]))


def _load_tracks_artifact(tracks_path: Path) -> dict[str, Any]:
    payload = json.loads(Path(tracks_path).read_text(encoding="utf-8"))
    validate_tracks_artifact(payload)
    return payload


def _index_reports_and_consensus(
    artifact: dict[str, Any], scene: SceneBundle
) -> tuple[
    dict[tuple[int, int], list[tuple[int, float, float, float]]],
    dict[int, dict[int, np.ndarray]],
    dict[str, int],
]:
    """Own reduction of the tracks artifact.

    Every non-miss report (ANY v, unlike build_m1_census.index_tracks which
    discards v < 0.5 immediately) grouped by (camera, frame); and per-seed
    defined consensus points by frame.
    """

    reports_by_cf: dict[tuple[int, int], list[tuple[int, float, float, float]]] = {}
    total_reports = 0
    miss_reports = 0
    v_nonbinary = 0
    for track in artifact["tracks"]:
        seed_id = int(track["seed_id"])
        camera_id = int(track["camera_id"])
        if camera_id not in scene.cameras:
            continue
        for report in track["reports"]:
            total_reports += 1
            if report.get("is_miss", False):
                miss_reports += 1
                continue
            v = float(report["v"])
            x = float(report["x"])
            y = float(report["y"])
            if v != 0.0 and v != 1.0:
                v_nonbinary += 1
            frame = int(round(float(report["frame"])))
            reports_by_cf.setdefault((camera_id, frame), []).append((seed_id, v, x, y))

    consensus_by_seed: dict[int, dict[int, np.ndarray]] = {}
    for key, entries in artifact.get("consensus", {}).items():
        seed_id = int(key)
        per_frame: dict[int, np.ndarray] = {}
        for entry in entries:
            if entry.get("point") is not None:
                frame = int(round(float(entry["frame"])))
                per_frame[frame] = np.asarray(entry["point"], dtype=np.float64)
        consensus_by_seed[seed_id] = per_frame

    diagnostics = {
        "total_reports": total_reports,
        "miss_reports": miss_reports,
        "v_nonbinary_reports": v_nonbinary,
    }
    return reports_by_cf, consensus_by_seed, diagnostics


def _load_sealed_reading_i(path: Path, sequence_name: str) -> tuple[int, int]:
    """Best-effort reader for a sealed per-sequence M1 census artifact.

    The design does not specify this file's exact top-level layout for a
    single-sequence sealed census; this reader accepts the shapes
    ``scripts/build_m1_census.py``'s ``run_census`` can plausibly emit for
    one sequence: a pooled top-level ``coverage_tallies``, or
    ``per_sequence[<name>].coverage_tallies``. Disclosed as a residual
    definitional freedom in the report.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    per_sequence = payload.get("per_sequence")
    entry = None
    if isinstance(per_sequence, dict):
        entry = per_sequence.get(sequence_name)
        if entry is None and len(per_sequence) == 1:
            entry = next(iter(per_sequence.values()))
    tallies = None
    if isinstance(entry, dict) and "coverage_tallies" in entry:
        tallies = entry["coverage_tallies"]
    elif "coverage_tallies" in payload:
        tallies = payload["coverage_tallies"]
    if tallies is None:
        raise ContractError(
            f"sealed census {path}: could not locate coverage_tallies for "
            f"sequence {sequence_name!r}"
        )
    return int(tallies["components_covered"]), int(tallies["components_total"])


def _check5_conversion_dir_name(sequence_name: str, scene_dir: Path) -> dict[str, Any]:
    rule = _KNOWN_NAMING_EXCEPTIONS.get(sequence_name)
    if rule is None:
        return {
            "pass": None,
            "note": (
                "no named exception in design section 2 for this sequence; "
                "conversion-directory correctness is the caller's responsibility"
            ),
        }
    name = Path(scene_dir).name
    if "expected_scene_dir_name" in rule:
        ok = name == rule["expected_scene_dir_name"] and name != rule.get(
            "forbidden_scene_dir_name"
        )
        return {"pass": ok, "expected": rule["expected_scene_dir_name"], "actual": name}
    suffix = rule["expected_scene_dir_suffix"]
    return {"pass": name.endswith(suffix), "expected_suffix": suffix, "actual": name}


def process_sequence(
    scene_dir: Path, tracks_path: Path, sequence_name: str, sealed_census_path: Path
) -> dict[str, Any]:
    scene = load_temporal_scene(Path(scene_dir))
    artifact = _load_tracks_artifact(Path(tracks_path))
    reports_by_cf, consensus_by_seed, report_diagnostics = _index_reports_and_consensus(
        artifact, scene
    )
    consensus_frames_sorted = {
        seed_id: sorted(per_frame.keys()) for seed_id, per_frame in consensus_by_seed.items()
    }

    def last_defined_at_or_before(seed_id: int, t: int) -> np.ndarray | None:
        frames_sorted = consensus_frames_sorted.get(seed_id, [])
        idx = bisect.bisect_right(frames_sorted, t) - 1
        if idx < 0:
            return None
        return consensus_by_seed[seed_id][frames_sorted[idx]]

    components_total = 0
    covered = {"i": 0, "ii": 0, "iii": 0, "iii_sens_a": 0, "iii_sens_b": 0}
    tallies: dict[str, int] = {
        "unreadable_masks": 0,
        "anchor_undefined_primary": 0,
        "anchor_out_of_domain_primary": 0,
        "anchor_undefined_sens_a": 0,
        "anchor_out_of_domain_sens_a": 0,
        "anchor_undefined_sens_b": 0,
        "anchor_out_of_domain_sens_b": 0,
        "v_zero_candidates_examined": 0,
        "v_zero_admitted_primary": 0,
        "v_zero_admitted_sens_a": 0,
        "v_zero_admitted_sens_b": 0,
    }
    tallies.update(report_diagnostics)

    for camera_id in scene.tracking_ids:
        camera = scene.cameras[camera_id]
        for frame in scene.frame_indices:
            mask_path = scene.mask_path(camera_id, frame)
            try:
                labels_arr, areas = load_component_labels(mask_path)
            except Exception:
                tallies["unreadable_masks"] += 1
                continue
            eligible_labels = {lbl for lbl, area in areas.items() if area >= MIN_COMPONENT_PX}
            components_total += len(eligible_labels)
            if not eligible_labels:
                continue

            local = {
                "i": set(), "ii": set(), "iii": set(), "iii_sens_a": set(), "iii_sens_b": set(),
            }

            for seed_id, v, x, y in reports_by_cf.get((camera_id, frame), []):
                col, row = round_half_up(x), round_half_up(y)
                if not (0 <= col <= camera.width - 1 and 0 <= row <= camera.height - 1):
                    continue
                label = int(labels_arr[row, col])
                if label not in eligible_labels:
                    continue

                if v >= VIS_THRESHOLD:
                    local["i"].add(label)
                    local["ii"].add(label)
                    local["iii"].add(label)
                    local["iii_sens_a"].add(label)
                    local["iii_sens_b"].add(label)
                    continue

                local["ii"].add(label)
                if v != 0.0:
                    continue

                tallies["v_zero_candidates_examined"] += 1
                point_primary = consensus_by_seed.get(seed_id, {}).get(frame)

                anchor_primary = resolve_anchor(camera, point_primary, labels_arr, transposed=False)
                if anchor_primary.status == "undefined":
                    tallies["anchor_undefined_primary"] += 1
                elif anchor_primary.status == "out_of_domain":
                    tallies["anchor_out_of_domain_primary"] += 1
                elif (
                    anchor_primary.label is not None
                    and anchor_primary.label > 0
                    and anchor_primary.label in eligible_labels
                    and anchor_primary.label == label
                ):
                    local["iii"].add(label)
                    tallies["v_zero_admitted_primary"] += 1

                anchor_a = resolve_anchor(camera, point_primary, labels_arr, transposed=True)
                if anchor_a.status == "undefined":
                    tallies["anchor_undefined_sens_a"] += 1
                elif anchor_a.status == "out_of_domain":
                    tallies["anchor_out_of_domain_sens_a"] += 1
                elif (
                    anchor_a.label is not None
                    and anchor_a.label > 0
                    and anchor_a.label in eligible_labels
                    and anchor_a.label == label
                ):
                    local["iii_sens_a"].add(label)
                    tallies["v_zero_admitted_sens_a"] += 1

                point_b = last_defined_at_or_before(seed_id, frame)
                anchor_b = resolve_anchor(camera, point_b, labels_arr, transposed=False)
                if anchor_b.status == "undefined":
                    tallies["anchor_undefined_sens_b"] += 1
                elif anchor_b.status == "out_of_domain":
                    tallies["anchor_out_of_domain_sens_b"] += 1
                elif (
                    anchor_b.label is not None
                    and anchor_b.label > 0
                    and anchor_b.label in eligible_labels
                    and anchor_b.label == label
                ):
                    local["iii_sens_b"].add(label)
                    tallies["v_zero_admitted_sens_b"] += 1

            for key in covered:
                covered[key] += len(local[key])

    if components_total == 0:
        raise ContractError(f"{sequence_name}: zero eligible fg components over the census window")

    coverage = {key: covered[key] / components_total for key in covered}

    sealed_covered, sealed_total = _load_sealed_reading_i(Path(sealed_census_path), sequence_name)
    check1 = {
        "pass": covered["i"] == sealed_covered and components_total == sealed_total,
        "sealed_components_covered": sealed_covered,
        "sealed_components_total": sealed_total,
        "computed_components_covered": covered["i"],
        "computed_components_total": components_total,
    }
    check2 = {"pass": True, "note": "single shared denominator by construction"}
    check3 = {
        "pass": covered["i"] <= covered["iii"] <= covered["ii"],
        "i": coverage["i"],
        "iii": coverage["iii"],
        "ii": coverage["ii"],
    }
    check4 = {"pass": tallies["unreadable_masks"] == 0, "count": tallies["unreadable_masks"]}
    check5 = _check5_conversion_dir_name(sequence_name, Path(scene_dir))

    primary_eligible = coverage["iii"] >= COVERAGE_FLOOR
    sens_a_eligible = coverage["iii_sens_a"] >= COVERAGE_FLOOR
    sens_b_eligible = coverage["iii_sens_b"] >= COVERAGE_FLOOR
    crosses_a = sens_a_eligible != primary_eligible
    crosses_b = sens_b_eligible != primary_eligible
    convention_dependent = bool(crosses_a or crosses_b)

    if coverage["ii"] < COVERAGE_FLOOR:
        cls = "ineligible"
    elif convention_dependent:
        cls = "indeterminate"
    elif coverage["iii"] >= COVERAGE_FLOOR:
        cls = "eligible"
    else:
        cls = "indeterminate"

    checks_pass = [check1["pass"], check3["pass"], check4["pass"]]
    if check5["pass"] is False:
        checks_pass.append(False)
    contract_status = "ok" if all(checks_pass) else "void"

    return {
        "components_total": components_total,
        "readings": {
            "i_frozen": {"components_covered": covered["i"], "coverage": coverage["i"]},
            "ii_any_report": {"components_covered": covered["ii"], "coverage": coverage["ii"]},
            "iii_anchor_agreeing": {
                "components_covered": covered["iii"], "coverage": coverage["iii"],
            },
            "iii_sensitivity_a_transposed_anchor": {
                "components_covered": covered["iii_sens_a"], "coverage": coverage["iii_sens_a"],
            },
            "iii_sensitivity_b_last_defined_anchor": {
                "components_covered": covered["iii_sens_b"], "coverage": coverage["iii_sens_b"],
            },
        },
        "class": cls,
        "convention_dependent": convention_dependent,
        "convention_dependence_detail": {
            "sensitivity_a_crosses_floor": crosses_a,
            "sensitivity_b_crosses_floor": crosses_b,
        },
        "by_monotonicity_would_apply": coverage["i"] >= COVERAGE_FLOOR,
        "tallies": tallies,
        "contract_checks": {
            "check1_reading_i_matches_sealed_census": check1,
            "check2_denominator_identical_across_readings": check2,
            "check3_monotonicity": check3,
            "check4_unreadable_masks_zero": check4,
            "check5_conversion_dir_name": check5,
        },
        "contract_status": contract_status,
        "provenance": {
            "scene_dir": str(scene_dir),
            "tracks_path": str(tracks_path),
            "sealed_census_path": str(sealed_census_path),
            "training_cameras": [int(c) for c in scene.tracking_ids],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independent recomputation of the A1 coverage bounding pair "
            "(design sections 3-5, plus section-4 sensitivities)."
        )
    )
    parser.add_argument("--scene-dir", type=Path, action="append", default=[])
    parser.add_argument("--tracks", type=Path, action="append", default=[])
    parser.add_argument("--sequence-name", type=str, action="append", default=[])
    parser.add_argument("--sealed-census", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    counts = {
        "--scene-dir": len(args.scene_dir),
        "--tracks": len(args.tracks),
        "--sequence-name": len(args.sequence_name),
        "--sealed-census": len(args.sealed_census),
    }
    if len(set(counts.values())) != 1:
        raise ContractError(
            "indep_coverage_recompute: --scene-dir/--tracks/--sequence-name/"
            f"--sealed-census must pair 1:1; got counts {counts}"
        )
    if counts["--scene-dir"] == 0:
        raise ContractError("indep_coverage_recompute: at least one sequence is required")

    sequences: dict[str, Any] = {}
    for scene_dir, tracks_path, name, sealed_path in zip(
        args.scene_dir, args.tracks, args.sequence_name, args.sealed_census
    ):
        if name in sequences:
            raise ContractError(f"duplicate --sequence-name {name!r}")
        sequences[name] = process_sequence(scene_dir, tracks_path, name, sealed_path)

    result = {
        "schema_version": "indep-elgs-coverage-bounding-pair-recompute-v1",
        "design_source": "research-wiki/operations/elgs-coverage-bounding-pair-design.md",
        "constants": {
            "vis_threshold": VIS_THRESHOLD,
            "min_component_px": MIN_COMPONENT_PX,
            "mask_binarize_threshold": MASK_BINARIZE_THRESHOLD,
            "coverage_floor": COVERAGE_FLOOR,
        },
        "sequences": sequences,
    }
    body = (
        json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    atomic_write_bytes(args.out, body)

    summary = {
        "out": str(args.out),
        "sequences": {
            name: {
                "class": seq["class"],
                "contract_status": seq["contract_status"],
                "convention_dependent": seq["convention_dependent"],
                "coverage_i": seq["readings"]["i_frozen"]["coverage"],
                "coverage_ii": seq["readings"]["ii_any_report"]["coverage"],
                "coverage_iii": seq["readings"]["iii_anchor_agreeing"]["coverage"],
            }
            for name, seq in sequences.items()
        },
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
