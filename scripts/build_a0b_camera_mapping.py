"""A0b applicable-camera mapping (D3) — the sealed candidate -> (S_w, audit
triple) map, computed once and hashed BEFORE any auditor sees any frame.

The rule is frozen in ``configs/elgs/prereg_m1_a0b_audit_v1.json`` revision 3
(``applicable_camera_set``) and is not re-derived here:

  S_w    the tracking cameras whose frustum contains the candidate's FROZEN
         anchor, under the frozen prereg_m1_census_v1 frustum_containment
         predicate. Inputs are the sealed anchor (audit_sample_B8 ``ltp``) and
         the frozen calibration ONLY -- never auditor verdicts, tracker
         visibility flags or mask occupancy.
  triple the 3-subset of S_w maximising the MINIMUM pairwise optical-axis
         angular separation; ties broken by the lexicographically smallest
         sorted camera-id tuple. Optical axis of camera c is
         w2c[:3,:3].T @ [0,0,1], the census's own construction.

A window with |S_w| < 3 is INADMISSIBLE: excluded from both the A3 numerator
and its denominator, and named in the report.

The load-bearing contract check is that the recomputed S_w reproduces the
sealed ``containing_cameras`` on every window. A mismatch VOIDS the run: it
would mean this reducer is not evaluating the candidate generator's own rule.

Emitting this mapping is in scope; RUNNING the audit is not. No candidate
imagery is read, rendered or written by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.artifacts import atomic_write_bytes  # noqa: E402
from depth_visibility.camera import camera_center  # noqa: E402
from depth_visibility.canonical import canonical_json_bytes  # noqa: E402
from depth_visibility.errors import ContractError  # noqa: E402
from scripts.build_elgs_tracks import SceneBundle, load_temporal_scene  # noqa: E402
from scripts.build_m1_census import frustum_contains, rig_radius  # noqa: E402

SCHEMA = "elgs-a0b-camera-mapping-v1"

#: The audit triple size, frozen by the preregistration's instrument section.
TRIPLE = 3


def optical_axis(scene: SceneBundle, camera_id: int) -> np.ndarray:
    """Optical axis of camera c = w2c[:3,:3].T @ [0,0,1] -- identical to the
    census's angular_separation_floor construction and to
    build_absence_diagnostic._pair_confirmable."""

    return scene.cameras[camera_id].w2c[:3, :3].T @ np.array([0.0, 0.0, 1.0])


def applicable_cameras(scene: SceneBundle, anchor: np.ndarray) -> tuple[int, ...]:
    """S_w under the frozen frustum_containment predicate, ascending."""

    return tuple(
        sorted(
            camera_id
            for camera_id in scene.tracking_ids
            if frustum_contains(scene.cameras[camera_id], anchor)
        )
    )


def _min_pairwise_separation(axes: dict[int, np.ndarray], triple: tuple[int, ...]) -> float:
    smallest = math.inf
    for a, b in itertools.combinations(triple, 2):
        cosine = float(np.clip(np.dot(axes[a], axes[b]), -1.0, 1.0))
        smallest = min(smallest, float(np.arccos(cosine)))
    return smallest


def select_audit_triple(
    scene: SceneBundle, cameras: Sequence[int]
) -> tuple[tuple[int, ...] | None, float | None, int]:
    """The frozen selection: argmax of the minimum pairwise optical-axis
    separation over 3-subsets of ``cameras``, ties broken by the
    lexicographically smallest sorted camera-id tuple.

    Returns (triple, separation_radians, n_tied). ``n_tied`` counts how many
    triples attained the maximum, so a tie-break is visible rather than silent.
    """

    ordered = sorted(int(c) for c in cameras)
    if len(ordered) < TRIPLE:
        return None, None, 0
    axes = {camera_id: optical_axis(scene, camera_id) for camera_id in ordered}
    # combinations over a sorted list yields lexicographically ascending tuples,
    # so scanning in order and keeping the FIRST maximum already realises the
    # tie-break; the explicit second pass makes that independent of scan order.
    candidates = list(itertools.combinations(ordered, TRIPLE))
    separations = [_min_pairwise_separation(axes, triple) for triple in candidates]
    best = max(separations)
    tied = [triple for triple, value in zip(candidates, separations) if value == best]
    return min(tied), best, len(tied)


def _quantile(values: Sequence[int], q: float) -> float:
    """Nearest-rank quantile on a sorted copy; no interpolation, so the
    disclosure table cannot drift with a numpy default."""

    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def map_windows(
    windows: Sequence[dict[str, Any]], scenes: dict[str, SceneBundle]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    radii = {name: rig_radius(scene) for name, scene in scenes.items()}
    centroids = {
        name: np.stack(
            [camera_center(scene.cameras[c].w2c) for c in scene.tracking_ids]
        ).mean(axis=0)
        for name, scene in scenes.items()
    }
    for window in windows:
        sequence = str(window["sequence"])
        if sequence not in scenes:
            raise ContractError(
                f"audit sample window in sequence {sequence!r} but no --scene-dir "
                f"was given for it; have {sorted(scenes)}"
            )
        scene = scenes[sequence]
        anchor = np.asarray(window["ltp"], dtype=np.float64)
        sealed = tuple(sorted(int(c) for c in window["containing_cameras"]))
        recomputed = applicable_cameras(scene, anchor)
        triple, separation, n_tied = select_audit_triple(scene, recomputed)
        rows.append(
            {
                "sequence": sequence,
                "seed_id": int(window["seed_id"]),
                "first_frame": int(window["first_frame"]),
                "last_frame": int(window["last_frame"]),
                "ltp_frame": int(window["ltp_frame"]),
                "key": f"{sequence}|{int(window['seed_id'])}|{int(window['first_frame'])}",
                "sealed_containing_cameras": list(sealed),
                "recomputed_S_w": list(recomputed),
                "reproduces_sealed": recomputed == sealed,
                "S_w_size": len(recomputed),
                "admissible": len(recomputed) >= TRIPLE,
                "audit_triple": list(triple) if triple is not None else None,
                "min_pairwise_separation_deg": (
                    None if separation is None else math.degrees(separation)
                ),
                "n_triples_tied_at_maximum": n_tied,
                "anchor_distance_over_rig_radius": float(
                    np.linalg.norm(anchor - centroids[sequence]) / radii[sequence]
                ),
            }
        )
    return rows


def disclosure_table(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sizes = [int(row["S_w_size"]) for row in rows]
    histogram: dict[str, int] = {}
    for size in sizes:
        histogram[str(size)] = histogram.get(str(size), 0) + 1
    inadmissible = [row["key"] for row in rows if not row["admissible"]]
    return {
        "S_w_size_distribution": {
            "min": min(sizes) if sizes else None,
            "q1": _quantile(sizes, 0.25),
            "median": _quantile(sizes, 0.50),
            "q3": _quantile(sizes, 0.75),
            "max": max(sizes) if sizes else None,
            "histogram": histogram,
        },
        "n_windows_with_fewer_than_three_applicable_cameras": len(inadmissible),
        "inadmissible_window_keys": inadmissible,
        "n_windows_with_a_broken_tie": sum(
            1 for row in rows if int(row["n_triples_tied_at_maximum"]) > 1
        ),
        "candidate_independent_alternatives_MEASURED_AND_REJECTED": {
            "D1_fixed_triple": (
                "all 26 tracking cameras with the fixed max-separated triple (7, 37, 46) at "
                "116.2 deg: contains all three of a candidate's anchor in only 14 of "
                "E_select's 49 frozen windows (28.6%), and 32 of 73 (43.8%) overall. "
                "AVAILABLE BUT DESTRUCTIVE -- collapses the kill rule's decidability."
            ),
            "D2_predeclared_volume": (
                "cameras containing a ball of radius k * rig_radius at the tracking-camera "
                "centroid: |S| = 13 / 7 / 6 / 2 / 0 at k = 0.10 / 0.15 / 0.20 / 0.25 / 0.30. "
                "Containing the frozen anchors needs k >= 0.69 while retaining 3 cameras "
                "needs k <= 0.20. REFUTED -- the feasible radius set is empty."
            ),
        },
        "data_quality_flags": [
            {
                "key": row["key"],
                "anchor_distance_over_rig_radius": row["anchor_distance_over_rig_radius"],
                "S_w_size": row["S_w_size"],
                "min_pairwise_separation_deg": row["min_pairwise_separation_deg"],
                "note": "anchor outside the camera sphere; report separately, never pool silently",
            }
            for row in rows
            if row["anchor_distance_over_rig_radius"] > 1.0
        ],
    }


def per_sequence_rollup(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in rows:
        block = out.setdefault(
            row["sequence"],
            {"n_windows": 0, "n_admissible": 0, "n_reproduces_sealed": 0,
             "S_w_size_min": None, "S_w_size_max": None,
             "min_pairwise_separation_deg_min": None},
        )
        block["n_windows"] += 1
        block["n_admissible"] += int(bool(row["admissible"]))
        block["n_reproduces_sealed"] += int(bool(row["reproduces_sealed"]))
        size = int(row["S_w_size"])
        block["S_w_size_min"] = size if block["S_w_size_min"] is None else min(block["S_w_size_min"], size)
        block["S_w_size_max"] = size if block["S_w_size_max"] is None else max(block["S_w_size_max"], size)
        separation = row["min_pairwise_separation_deg"]
        if separation is not None:
            current = block["min_pairwise_separation_deg_min"]
            block["min_pairwise_separation_deg_min"] = (
                separation if current is None else min(current, separation)
            )
    return out


def check_contracts(rows: Sequence[dict[str, Any]], expected_n: int) -> dict[str, Any]:
    failures: list[str] = []
    n_reproduces = sum(1 for row in rows if row["reproduces_sealed"])
    if n_reproduces != len(rows):
        offenders = [row["key"] for row in rows if not row["reproduces_sealed"]]
        failures.append(
            f"recomputed S_w reproduces the sealed containing_cameras on only "
            f"{n_reproduces} of {len(rows)} windows; offenders {offenders}"
        )
    if expected_n is not None and len(rows) != expected_n:
        failures.append(f"mapped {len(rows)} windows, audit sample declares {expected_n}")
    for row in rows:
        if row["admissible"] and row["audit_triple"] is None:
            failures.append(f"{row['key']}: admissible but no triple selected")
        if row["audit_triple"] is not None:
            triple = row["audit_triple"]
            if len(set(triple)) != TRIPLE or not set(triple) <= set(row["recomputed_S_w"]):
                failures.append(f"{row['key']}: triple {triple} is not a 3-subset of S_w")
    return {
        "passed": not failures,
        "failures": failures,
        "n_reproduces_sealed": n_reproduces,
        "n_windows": len(rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--diagnostic", type=Path, required=True,
        help="the sealed absence-diagnostic artifact carrying audit_sample_B8",
    )
    parser.add_argument(
        "--scene-dir", type=Path, action="append", required=True,
        help="repeatable; paired positionally with --sequence-name",
    )
    parser.add_argument("--sequence-name", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if len(args.scene_dir) != len(args.sequence_name):
        raise ContractError(
            f"{len(args.scene_dir)} --scene-dir vs {len(args.sequence_name)} "
            "--sequence-name; must pair 1:1"
        )

    diagnostic_path = Path(args.diagnostic)
    diagnostic_sha = hashlib.sha256(diagnostic_path.read_bytes()).hexdigest()
    payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    sample = payload["audit_sample_B8"]
    windows = sample["windows"]

    scenes = {
        name: load_temporal_scene(Path(scene_dir))
        for scene_dir, name in zip(args.scene_dir, args.sequence_name)
    }

    rows = map_windows(windows, scenes)
    contracts = check_contracts(rows, int(sample["n_selected"]))

    result = {
        "schema_version": SCHEMA,
        "cell": "a0b_camera_mapping",
        "diagnostic_only": True,
        "prereg": {
            "authority": "configs/elgs/prereg_m1_a0b_audit_v1.json",
            "revision_required": 3,
            "adopted_definition": "D3_census_per_candidate_frustum",
            "estimand_this_mapping_licenses": (
                "unobservability across the cameras that the frozen candidate generator "
                "geometrically considered applicable, followed by same-identity reappearance."
            ),
            "does_not_establish": [
                "literal physical absence",
                "unobservability across an independently fixed rig-wide camera set",
                "candidate-generator-independent event supply",
            ],
        },
        "source": {
            "diagnostic_path": str(diagnostic_path),
            "diagnostic_sha256": diagnostic_sha,
            "audit_sample_seed": sample["seed"],
            "audit_sample_n_selected": sample["n_selected"],
            "sequences": sorted(scenes),
        },
        "constants": {
            "triple_size": TRIPLE,
            "frustum_rule": (
                "positive camera-frame depth and a pixel inside [0, W-1] x [0, H-1] "
                "(prereg_m1_census_v1 frustum_containment)"
            ),
            "optical_axis": "w2c[:3,:3].T @ [0,0,1]",
            "tie_break": "lexicographically smallest sorted camera-id tuple",
        },
        "per_window": rows,
        "per_sequence": per_sequence_rollup(rows),
        "disclosure": disclosure_table(rows),
        "contract_checks": contracts,
        "void": not contracts["passed"],
    }
    result["config_sha256"] = hashlib.sha256(
        canonical_json_bytes({"argv": list(argv) if argv else sys.argv[1:]})
    ).hexdigest()
    body = json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":"))
    atomic_write_bytes(args.out, body.encode("utf-8") + b"\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "artifact_sha256": hashlib.sha256(
                    Path(args.out).read_bytes()
                ).hexdigest(),
                "void": result["void"],
                "contract_failures": contracts["failures"],
                "n_windows": contracts["n_windows"],
                "n_reproduces_sealed": contracts["n_reproduces_sealed"],
                "n_inadmissible": result["disclosure"][
                    "n_windows_with_fewer_than_three_applicable_cameras"
                ],
                "S_w_size_distribution": result["disclosure"]["S_w_size_distribution"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
