"""Apply the cycle-3 split gate (prereg_m1_cycle3_gate_v1, SIGNED).

Pure arithmetic over SEALED census artifacts (full-window censuses with
per-half attribution, evaluator 1d8f3b0+). Computes the two separately
verdicted preconditions:

- G-R (reactivation; writing_2 alone): unscreened-half union returns
  >= 36 AND unscreened-half coverage >= 0.5. Publishes the per-identity
  decomposition of writing_2's second-half union returns and the strict
  primary count alongside the union (standing loosening disclosure).
- G-OA (occlusion/absence; the three dev sequences): pooled
  unscreened-half occlusion >= 36 AND pooled true-absence >= 36 AND
  per-sequence true-absence >= 12 AND pooled + per-sequence
  unscreened-half coverage >= 0.5.

Each verdict stands alone; each valid FAIL is final for its claim
family on this subset (frozen policy). Output is plain strict JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.artifacts import atomic_write_bytes  # noqa: E402
from depth_visibility.canonical import sha256_file  # noqa: E402
from depth_visibility.errors import ContractError  # noqa: E402

GATE_SCHEMA = "elgs-m1-cycle3-gate-v1"

G_R_FLOORS = {"union_returns_min": 36, "coverage_min": 0.5}
G_OA_FLOORS = {
    "occlusion_events_min_pooled": 36,
    "true_absence_min_pooled": 36,
    "true_absence_min_per_sequence": 12,
    "coverage_min_pooled": 0.5,
    "coverage_min_per_sequence": 0.5,
}
ANCHOR = "writing_2"


def _sequence_block(census: dict[str, Any]) -> dict[str, Any]:
    per_sequence = census["per_sequence"]
    if len(per_sequence) != 1:
        raise ContractError("cycle-3 gate expects single-sequence census artifacts")
    return next(iter(per_sequence.values()))


def _second_half_returns(sequence: dict[str, Any]) -> dict[str, Any]:
    """Second-half union/primary counts + per-identity decomposition from
    the sealed records (returns attributed by terminating-run start)."""

    split = sequence["halves"]["split_frame"]
    union = primary = 0
    per_identity: dict[str, int] = {}
    for candidate in sequence["records"]["true_absence_candidates"]:
        if "return_run_start" not in candidate:
            continue
        if candidate["return_run_start"] < split:
            continue
        if candidate["return"] == "same_object_return_primary":
            primary += 1
            union += 1
        elif candidate["return"] == "same_object_return_r2prime":
            union += 1
        else:
            continue
        key = str(candidate["seed_id"])
        per_identity[key] = per_identity.get(key, 0) + 1
    return {"union": union, "primary": primary, "per_identity": per_identity}


def apply_gate(census_paths: dict[str, Path]) -> dict[str, Any]:
    if ANCHOR not in census_paths:
        raise ContractError(f"anchor census ({ANCHOR}) is required")
    if len(census_paths) != 3:
        raise ContractError("cycle-3 gate expects exactly three dev-sequence censuses")

    sequences: dict[str, dict[str, Any]] = {}
    provenance: dict[str, str] = {}
    for name, path in census_paths.items():
        census = json.loads(Path(path).read_text(encoding="utf-8"))
        sequences[name] = _sequence_block(census)
        provenance[name] = sha256_file(path)

    # --- G-R: writing_2 alone ------------------------------------------------
    anchor = sequences[ANCHOR]
    anchor_returns = _second_half_returns(anchor)
    # Cross-check against the halves attribution the evaluator computed.
    halves_union = anchor["halves"]["union_returns"]["second_half"]
    if anchor_returns["union"] != halves_union:
        raise ContractError(
            f"record-derived second-half union ({anchor_returns['union']}) disagrees "
            f"with the evaluator's halves attribution ({halves_union})"
        )
    anchor_half_cov = anchor["coverage_tallies"]["by_half"]["second_half"]
    anchor_coverage = (
        anchor_half_cov["components_covered"] / anchor_half_cov["components_total"]
    )
    g_r = {
        "floors": G_R_FLOORS,
        "union_returns_second_half": anchor_returns["union"],
        "primary_returns_second_half": anchor_returns["primary"],
        "loosening_disclosure": "the strict cycle-1 primary is reported alongside the union per the standing disclosure",
        "per_identity_decomposition": anchor_returns["per_identity"],
        "coverage_second_half": anchor_coverage,
        "comparisons": {
            "union_returns_min": anchor_returns["union"] >= G_R_FLOORS["union_returns_min"],
            "coverage_min": anchor_coverage >= G_R_FLOORS["coverage_min"],
        },
    }
    g_r["pass"] = all(g_r["comparisons"].values())

    # --- G-OA: pooled over the three ----------------------------------------
    pooled_occlusion = 0
    pooled_absence = 0
    pooled_cov_num = pooled_cov_den = 0
    per_sequence_detail: dict[str, dict[str, Any]] = {}
    per_seq_pass = True
    for name, sequence in sequences.items():
        occl = sequence["halves"]["occlusion"]["second_half"]
        absence = sequence["halves"]["true_absence"]["second_half"]
        cov_half = sequence["coverage_tallies"]["by_half"]["second_half"]
        coverage = cov_half["components_covered"] / cov_half["components_total"]
        pooled_occlusion += occl
        pooled_absence += absence
        pooled_cov_num += cov_half["components_covered"]
        pooled_cov_den += cov_half["components_total"]
        seq_ok = (
            absence >= G_OA_FLOORS["true_absence_min_per_sequence"]
            and coverage >= G_OA_FLOORS["coverage_min_per_sequence"]
        )
        per_seq_pass = per_seq_pass and seq_ok
        per_sequence_detail[name] = {
            "occlusion_second_half": occl,
            "true_absence_second_half": absence,
            "coverage_second_half": coverage,
            "per_sequence_ok": seq_ok,
        }
    pooled_coverage = pooled_cov_num / pooled_cov_den
    g_oa = {
        "floors": G_OA_FLOORS,
        "pooled_occlusion_second_half": pooled_occlusion,
        "pooled_true_absence_second_half": pooled_absence,
        "pooled_coverage_second_half": pooled_coverage,
        "per_sequence": per_sequence_detail,
        "comparisons": {
            "occlusion_events_min_pooled": pooled_occlusion
            >= G_OA_FLOORS["occlusion_events_min_pooled"],
            "true_absence_min_pooled": pooled_absence
            >= G_OA_FLOORS["true_absence_min_pooled"],
            "per_sequence_floors": per_seq_pass,
            "coverage_min_pooled": pooled_coverage >= G_OA_FLOORS["coverage_min_pooled"],
        },
    }
    g_oa["pass"] = all(g_oa["comparisons"].values())

    return {
        "schema_version": GATE_SCHEMA,
        "G_R_reactivation": g_r,
        "G_OA_occlusion_absence": g_oa,
        "necessary_condition_note": (
            "all statistics are model-free candidate-opportunity UPPER BOUNDS; "
            "meeting a floor is a necessary condition for adequacy, never evidence "
            "that that many true events exist"
        ),
        "verdict_note": (
            "each precondition is verdicted separately; each PASS restores the M1 "
            "precondition for ITS claim family; each valid FAIL is final for its "
            "family on this subset; starting M2 remains a separate user approval"
        ),
        "census_sha256": provenance,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--writing-2", type=Path, required=True, help="writing_2 census.json")
    parser.add_argument("--pour-tea", type=Path, required=True)
    parser.add_argument("--tambourine", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = apply_gate(
        {"writing_2": args.writing_2, "pour_tea": args.pour_tea, "tambourine": args.tambourine}
    )
    body = json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":"))
    atomic_write_bytes(args.out, body.encode("utf-8") + b"\n")
    print(
        json.dumps(
            {
                "G_R": result["G_R_reactivation"]["pass"],
                "G_OA": result["G_OA_occlusion_absence"]["pass"],
                "out": str(args.out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
