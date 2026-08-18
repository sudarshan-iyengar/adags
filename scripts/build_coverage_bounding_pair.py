"""Coverage bounding pair (A1) — three report-admission readings of the
M1-A0 ``track_coverage_upper_bound`` over ONE fixed denominator.

DIAGNOSTIC ONLY. This reducer changes no eligibility verdict, no floor, no
census figure and no gate. It brackets how far a corrected report-admission
rule could move per-sequence coverage, and it reports the bracket.

Frozen design: ``research-wiki/operations/elgs-coverage-bounding-pair-design.md``.

The three readings (design section 3), all sharing the census statistic's own
denominator (eligible foreground components over tracking cameras x frames):

  (i)   frozen     -- not a miss, in-domain, v >= 0.5   (LOWER bound; must
                      reproduce the sealed census artifact exactly)
  (ii)  any-report -- not a miss, in-domain, any v       (UPPER bound)
  (iii) anchor     -- (i) plus v == 0 reports whose pixel label equals the
                      label of the identity's anchor pixel in the same
                      (camera, frame), that label being eligible (MIDDLE)

Component eligibility, mask binarization, pixel rounding and the projection
convention are taken from the existing census implementation and prereg; no
new component machinery is introduced (design section 4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.artifacts import atomic_write_bytes  # noqa: E402
from depth_visibility.canonical import canonical_json_bytes  # noqa: E402
from depth_visibility.errors import ContractError  # noqa: E402
from elgs.tracks_schema import validate_tracks_artifact  # noqa: E402
from scripts.build_elgs_tracks import SceneBundle, load_temporal_scene  # noqa: E402
from scripts.build_m1_census import (  # noqa: E402
    VIS_THRESHOLD,
    load_component_labels,
    round_half_up,
)

SCHEMA = "elgs-a1-coverage-bounding-pair-v1"

# Reading keys, fixed here so the artifact cannot drift from the design.
R_FROZEN = "i_frozen"
R_ANY = "ii_any_report"
R_ANCHOR = "iii_anchor_agreeing"
R_ANCHOR_T = "iii_anchor_transposed"
R_ANCHOR_LAST = "iii_anchor_last_defined"
READINGS = (R_FROZEN, R_ANY, R_ANCHOR, R_ANCHOR_T, R_ANCHOR_LAST)


def _report_rows(
    artifact: dict[str, Any], scene: SceneBundle
) -> tuple[
    dict[tuple[int, int], list[tuple[int, int, int]]],
    dict[tuple[int, int], list[tuple[int, int, int]]],
    dict[str, int],
]:
    """Split the in-domain, non-miss reports by visibility, in one pass.

    Returns ``(visible, invisible, tallies)`` where each map is
    ``(camera, frame) -> [(seed_id, col, row)]``.

    The VISIBLE side reproduces ``build_m1_census.index_tracks`` exactly,
    including its LAST-WINS behaviour for a repeated
    ``(seed, camera, frame)`` key -- that dict semantics is part of the
    frozen instrument, and reading (i) has to equal the sealed census
    numerator to the integer (design section 7 check 1). The INVISIBLE side
    keeps every report, since readings (ii)/(iii) only ever ADD admitted
    reports to reading (i)'s set, which is what makes them bounds.
    """

    validate_tracks_artifact(artifact)
    visible_last: dict[tuple[int, int, int], tuple[int, int]] = {}
    invisible: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    tallies = {
        "visible_duplicate_keys": 0,
        "invisible_reports": 0,
        "reports_skipped_miss": 0,
        "reports_skipped_out_of_domain": 0,
        "reports_skipped_unknown_camera": 0,
    }
    for track in artifact["tracks"]:
        seed_id = int(track["seed_id"])
        camera_id = int(track["camera_id"])
        camera = scene.cameras.get(camera_id)
        if camera is None:
            tallies["reports_skipped_unknown_camera"] += len(track["reports"])
            continue
        for report in track["reports"]:
            if report.get("is_miss", False):
                tallies["reports_skipped_miss"] += 1
                continue
            col = round_half_up(float(report["x"]))
            row = round_half_up(float(report["y"]))
            if not (0 <= col <= camera.width - 1 and 0 <= row <= camera.height - 1):
                tallies["reports_skipped_out_of_domain"] += 1
                continue
            frame = int(round(float(report["frame"])))
            if float(report["v"]) >= VIS_THRESHOLD:
                key = (seed_id, camera_id, frame)
                if key in visible_last:
                    tallies["visible_duplicate_keys"] += 1
                visible_last[key] = (col, row)
            else:
                tallies["invisible_reports"] += 1
                invisible.setdefault((camera_id, frame), []).append(
                    (seed_id, col, row)
                )
    visible: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for (seed_id, camera_id, frame), (col, row) in visible_last.items():
        visible.setdefault((camera_id, frame), []).append((seed_id, col, row))
    return visible, invisible, tallies


def _consensus(artifact: dict[str, Any]) -> dict[int, dict[int, np.ndarray]]:
    out: dict[int, dict[int, np.ndarray]] = {}
    for key, entries in artifact.get("consensus", {}).items():
        per_frame: dict[int, np.ndarray] = {}
        for entry in entries:
            if entry.get("point") is not None:
                per_frame[int(round(float(entry["frame"])))] = np.asarray(
                    entry["point"], dtype=np.float64
                )
        out[int(key)] = per_frame
    return out


def _last_defined_frames(
    consensus: dict[int, dict[int, np.ndarray]], frames: Sequence[int]
) -> dict[int, dict[int, int]]:
    """seed -> frame -> the latest frame at or before it with a point."""

    out: dict[int, dict[int, int]] = {}
    for seed, per_frame in consensus.items():
        carry: dict[int, int] = {}
        best: int | None = None
        for frame in frames:
            if frame in per_frame:
                best = frame
            if best is not None:
                carry[frame] = best
        out[seed] = carry
    return out


def _project(camera, point: np.ndarray) -> tuple[int, int] | None:
    """Frozen projection: w2c, K, positive depth, round-half-up, in-raster.

    Returns ``(col, row)``; the first projected coordinate is the column
    because ``frustum_containment`` bounds it by ``W - 1`` (design section 4).
    """

    cam_point = camera.w2c[:3, :3] @ point + camera.w2c[:3, 3]
    if cam_point[2] <= 0.0:
        return None
    uv = camera.K @ cam_point
    col = round_half_up(uv[0] / uv[2])
    row = round_half_up(uv[1] / uv[2])
    if not (0 <= col <= camera.width - 1 and 0 <= row <= camera.height - 1):
        return None
    return col, row


def bounding_pair_one_sequence(
    scene_dir: Path, tracks_path: Path, *, sequence: str
) -> dict[str, Any]:
    scene = load_temporal_scene(scene_dir)
    artifact = json.loads(tracks_path.read_text(encoding="utf-8"))
    visible_rows, invisible_rows, report_tallies = _report_rows(artifact, scene)
    consensus = _consensus(artifact)
    last_defined = _last_defined_frames(consensus, scene.frame_indices)

    covered = {key: 0 for key in READINGS}
    total = 0
    tallies = {
        **report_tallies,
        "visible_on_eligible": 0,
        "invisible_on_eligible": 0,
        "invisible_admitted_anchor": 0,
        "invisible_admitted_anchor_transposed": 0,
        "invisible_admitted_anchor_last_defined": 0,
        "anchor_undefined": 0,
        "anchor_out_of_domain": 0,
        # load_component_labels is fail-closed: a missing or undecodable
        # mask raises ContractError rather than being tallied, so this
        # counter can only ever be 0 in a run that completes.
        "unreadable_masks": 0,
    }
    started = time.time()
    for camera_id in scene.tracking_ids:
        camera = scene.cameras[camera_id]
        for frame in scene.frame_indices:
            labels, eligible = load_component_labels(scene.mask_path(camera_id, frame))
            total += len(eligible)
            hits: dict[str, set[int]] = {key: set() for key in READINGS}
            anchor_cache: dict[tuple[int, str], tuple[int, int] | None] = {}
            max_row, max_col = labels.shape[0] - 1, labels.shape[1] - 1
            for seed_id, col, row in visible_rows.get((camera_id, frame), []):
                # the census clamps into the mask raster after its own
                # camera-raster in-domain test; reproduced for exact parity
                label = int(labels[min(max(row, 0), max_row), min(max(col, 0), max_col)])
                if label in eligible:
                    tallies["visible_on_eligible"] += 1
                    for key in READINGS:
                        hits[key].add(label)
            for seed_id, col, row in invisible_rows.get((camera_id, frame), []):
                row_c = min(max(row, 0), max_row)
                col_c = min(max(col, 0), max_col)
                label = int(labels[row_c, col_c])
                if label not in eligible:
                    # an invisible report off every eligible component can
                    # never cover one under any reading
                    continue
                tallies["invisible_on_eligible"] += 1
                # reading (ii) admits an invisible report unconditionally
                hits[R_ANY].add(label)
                for variant, key, tally in (
                    ("at_frame", R_ANCHOR, "invisible_admitted_anchor"),
                    ("at_frame_t", R_ANCHOR_T, "invisible_admitted_anchor_transposed"),
                    (
                        "last_defined",
                        R_ANCHOR_LAST,
                        "invisible_admitted_anchor_last_defined",
                    ),
                ):
                    cache_key = (seed_id, variant)
                    if cache_key in anchor_cache:
                        pixel = anchor_cache[cache_key]
                    else:
                        per_frame = consensus.get(seed_id, {})
                        if variant == "last_defined":
                            source = last_defined.get(seed_id, {}).get(frame)
                            point = per_frame.get(source) if source is not None else None
                        else:
                            point = per_frame.get(frame)
                        if point is None:
                            pixel = None
                            if variant == "at_frame":
                                tallies["anchor_undefined"] += 1
                        else:
                            pixel = _project(camera, point)
                            if pixel is None and variant == "at_frame":
                                tallies["anchor_out_of_domain"] += 1
                        anchor_cache[cache_key] = pixel
                    if pixel is None:
                        continue
                    a_col, a_row = pixel
                    if variant == "at_frame_t":
                        # the transposed sensitivity reading: read the anchor
                        # as labels[col, row]; skip when that is out of range
                        if not (
                            0 <= a_col < labels.shape[0]
                            and 0 <= a_row < labels.shape[1]
                        ):
                            continue
                        anchor_label = int(labels[a_col, a_row])
                    else:
                        anchor_label = int(labels[a_row, a_col])
                    if anchor_label > 0 and anchor_label == label:
                        hits[key].add(label)
                        tallies[tally] += 1
            for key in READINGS:
                covered[key] += len(hits[key])
    elapsed = time.time() - started

    result: dict[str, Any] = {
        "sequence": sequence,
        "scene_dir": str(scene_dir),
        "tracks": str(tracks_path),
        "n_tracking_cameras": len(scene.tracking_ids),
        "first_frame": int(scene.frame_indices[0]),
        "last_frame": int(scene.frame_indices[-1]),
        "n_frames": len(scene.frame_indices),
        "components_total": total,
        "components_covered": {key: covered[key] for key in READINGS},
        "coverage": {
            key: (covered[key] / total if total else None) for key in READINGS
        },
        "tallies": tallies,
        "seconds": elapsed,
    }
    return result


def _check_contracts(
    per_sequence: list[dict[str, Any]], sealed: dict[str, dict[str, int]]
) -> dict[str, Any]:
    """Design section 7. Any failure VOIDS the run."""

    failures: list[str] = []
    for row in per_sequence:
        seq = row["sequence"]
        cov = row["coverage"]
        if row["tallies"]["unreadable_masks"] != 0:
            failures.append(f"{seq}: unreadable masks")
        if not (cov[R_FROZEN] <= cov[R_ANCHOR] <= cov[R_ANY] + 1e-12):
            failures.append(
                f"{seq}: monotonicity violated "
                f"({cov[R_FROZEN]} / {cov[R_ANCHOR]} / {cov[R_ANY]})"
            )
        ref = sealed.get(seq)
        if ref is None:
            failures.append(f"{seq}: no sealed census tally supplied")
            continue
        if row["components_total"] != ref["components_total"]:
            failures.append(
                f"{seq}: denominator {row['components_total']} != sealed "
                f"{ref['components_total']}"
            )
        if row["components_covered"][R_FROZEN] != ref["components_covered"]:
            failures.append(
                f"{seq}: reading (i) numerator "
                f"{row['components_covered'][R_FROZEN]} != sealed "
                f"{ref['components_covered']}"
            )
    return {"passed": not failures, "failures": failures}


def classify(coverage: dict[str, float | None]) -> str:
    """Design section 5, with the section 4 convention-dependence rule."""

    lower = coverage[R_FROZEN]
    upper = coverage[R_ANY]
    middle = coverage[R_ANCHOR]
    sensitivities = [
        coverage.get(R_ANCHOR_T),
        coverage.get(R_ANCHOR_LAST),
    ]
    crossed = [
        s is not None and (s >= 0.5) != (middle >= 0.5) for s in sensitivities
    ]
    if any(crossed):
        return "indeterminate"
    if middle >= 0.5:
        return "eligible"
    if upper < 0.5:
        return "ineligible"
    assert lower <= middle
    return "indeterminate"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scene-dir", type=Path, action="append", required=True,
        help="repeatable; paired positionally with --tracks and --sequence-name",
    )
    parser.add_argument("--tracks", type=Path, action="append", required=True)
    parser.add_argument("--sequence-name", action="append", required=True)
    parser.add_argument(
        "--sealed-census", type=Path, action="append", required=True,
        help="repeatable; the sealed census.json whose reading (i) tallies "
             "this run must reproduce exactly",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    n = len(args.scene_dir)
    if not (len(args.tracks) == len(args.sequence_name) == len(args.sealed_census) == n):
        raise ContractError(
            f"{n} --scene-dir vs {len(args.tracks)} --tracks vs "
            f"{len(args.sequence_name)} --sequence-name vs "
            f"{len(args.sealed_census)} --sealed-census; must pair 1:1"
        )

    sealed: dict[str, dict[str, int]] = {}
    for name, census_path in zip(args.sequence_name, args.sealed_census):
        payload = json.loads(Path(census_path).read_text(encoding="utf-8"))
        tallies = payload["coverage_tallies"]
        sealed[name] = {
            "components_total": int(tallies["components_total"]),
            "components_covered": int(tallies["components_covered"]),
            "census_path": str(census_path),
            "census_sha256": hashlib.sha256(
                Path(census_path).read_bytes()
            ).hexdigest(),
        }

    per_sequence = [
        bounding_pair_one_sequence(scene_dir, tracks, sequence=name)
        for scene_dir, tracks, name in zip(
            args.scene_dir, args.tracks, args.sequence_name
        )
    ]
    for row in per_sequence:
        row["class"] = classify(row["coverage"])

    contracts = _check_contracts(per_sequence, sealed)
    pooled_total = sum(r["components_total"] for r in per_sequence)
    pooled = {
        key: (
            sum(r["components_covered"][key] for r in per_sequence) / pooled_total
            if pooled_total
            else None
        )
        for key in READINGS
    }

    result = {
        "schema_version": SCHEMA,
        "cell": "a1_coverage_bounding_pair",
        "diagnostic_only": True,
        "constants": {
            "visibility_threshold": VIS_THRESHOLD,
            "readings": list(READINGS),
        },
        "per_sequence": per_sequence,
        "pooled_over_computed_sequences": pooled,
        "sealed_census_reference": sealed,
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
                "void": result["void"],
                "contract_failures": contracts["failures"],
                "classes": {r["sequence"]: r["class"] for r in per_sequence},
                "coverage": {r["sequence"]: r["coverage"] for r in per_sequence},
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
