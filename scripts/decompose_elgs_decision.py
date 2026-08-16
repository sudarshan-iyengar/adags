#!/usr/bin/env python3
"""Decompose a committed EL-GS acceptance decision from its checkpoint.

DIAGNOSTIC ONLY. This reads a checkpoint, reports every decision
quantity that survives into the artifact, and states plainly which ones
do NOT -- it re-interprets nothing.

The question it answers: experiment 78 committed FISSION:219:a5395116
with `n_samples = 8` and `se = 0.0`, and the durable record says the
photometric and evidence contributions are "not separable from the
artifacts". This measures exactly how unseparable they are.

`elgs.acceptance.decide` computes

    delta_render = SNIS(candidate) - SNIS(incumbent)
                 = sum_i w_i d_i / sum_i w_i          (paired, CRN)
    total        = delta_render + exact_deltas + transaction_increment
    accepted    <=> total + k * se < 0

with d_i = loss_candidate_i - loss_incumbent_i the i-th confirmation
unit's paired PHOTOMETRIC delta and w_i the SHARED SNIS weight. So
`delta_render` IS the weight-normalized mean of the eight per-unit
deltas. `elgs.trainer_hooks` persists only `n_samples`, `se` and the
drawn `units` out of the whole `AcceptanceRecord`, so d_i, w_i,
delta_render, exact_deltas and total are discarded at the end of the
round.

Section 3 therefore asks the one question `se` alone CAN answer: the
user's caution is that `se = 0` must not be read as "the photometric
contribution was zero". `se` is the standard deviation of 200 paired
cluster-bootstrap replicates, each a weighted mean of a resampled
multiset of the same d_i. A common NONZERO d survives every weighted
mean too, so it also drives the spread to (nearly) zero -- but not
bit-exactly, because the two arms' ratios are accumulated separately and
the residual rounding varies with the multiset. This section measures
that distinction numerically instead of asserting it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from elgs.acceptance import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    SnisSample,
    paired_cluster_bootstrap_se,
    paired_snis_delta,
)


#: The fields `elgs.acceptance.AcceptanceRecord` carries versus the ones
#: `elgs.trainer_hooks` writes into `round_bookkeeping.committed_decisions`.
_RECORD_FIELDS = (
    "delta_render", "exact_deltas", "transaction_increment",
    "se", "k", "n_samples", "ess", "n_units", "accepted",
)
_PERSISTED_FIELDS = ("n_samples", "se")


def _find_elgs_state(node, depth: int = 0):
    """The `elgs_state` payload, wherever `capture()` put it.

    `GaussianModel.capture()` returns a long tuple whose
    `routing_motion_params` dict carries `elgs_state`, and that payload
    is the dict holding `round_bookkeeping`. Both shapes are accepted so
    a layout change surfaces as "not found" rather than a wrong answer.
    """
    if depth > 4:
        return None
    if isinstance(node, dict):
        if "round_bookkeeping" in node:
            return node
        for key in ("elgs_state",):
            if key in node:
                found = _find_elgs_state(node[key], depth + 1)
                if found is not None:
                    return found
        for value in node.values():
            found = _find_elgs_state(value, depth + 1)
            if found is not None:
                return found
        return None
    if isinstance(node, (list, tuple)):
        for value in node:
            found = _find_elgs_state(value, depth + 1)
            if found is not None:
                return found
    return None


def _read_checkpoint(path: str) -> dict:
    model_params, iteration = torch.load(path, map_location="cpu")
    return {
        "iteration": int(iteration),
        "extras": _find_elgs_state(model_params),
    }


def _se_of_common_delta(common: float, *, seed: int, n_units: int = 8) -> float:
    """SE the bootstrap reports when every per-unit delta equals `common`.

    Weights are deliberately UNEQUAL -- an equal-weight fixture would
    make every weighted mean identical for trivial reasons and prove
    nothing about the real, unequal-weight case.
    """
    samples = [
        SnisSample(
            unit=(index, float(index)),
            nu_density=0.5 + 0.05 * index,
            mix_density=1.0 + 0.01 * index,
            loss_incumbent=0.3 + 0.017 * index,
            loss_candidate=0.3 + 0.017 * index + common,
        )
        for index in range(n_units)
    ]
    return paired_cluster_bootstrap_se(samples, 0.5, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--k", type=float, default=1.0,
        help="the run's elgs_k_se, used only to restate the decision rule",
    )
    args = parser.parse_args()

    loaded = _read_checkpoint(args.checkpoint)
    extras = loaded["extras"] or {}
    bookkeeping = dict(extras.get("round_bookkeeping") or {})
    decisions = list(bookkeeping.get("committed_decisions") or [])

    report = {
        "schema": "elgs-decision-decomposition-v1",
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": loaded["iteration"],
        "rounds_run": bookkeeping.get("rounds_run"),
        "acceptance_record_fields": list(_RECORD_FIELDS),
        "persisted_fields": list(_PERSISTED_FIELDS),
        "discarded_fields": [
            name for name in _RECORD_FIELDS if name not in _PERSISTED_FIELDS
        ],
        "decisions": [],
    }

    # -- 1. what the artifact actually holds ----------------------------
    for decision in decisions:
        entry = {
            "candidate_id": decision.get("candidate_id"),
            "op": decision.get("op"),
            "family_id": decision.get("family_id"),
            "round_index": decision.get("round_index"),
            "iteration": decision.get("iteration"),
            "n_samples": decision.get("n_samples"),
            "se": decision.get("se"),
            "se_repr": repr(decision.get("se")),
            "se_is_exactly_zero": decision.get("se") == 0.0,
            "n_units_drawn": len(decision.get("units") or []),
            "units": [list(u) for u in (decision.get("units") or [])],
            "incumbent_intervals": decision.get("incumbent_intervals"),
        }
        # -- 1b. WHERE IN TIME the confirmation renders happened -------
        #
        # `SlotGrid.draw` returns a CONTIGUOUS slice of the reserved
        # pool, and `setup_elgs` builds that pool by iterating
        # `sorted(by_time)` -- ascending timestamp. So slot (0, 0, 0) is
        # the pool's first `units_per_slot` entries, i.e. the EARLIEST
        # frames. If they all share one timestamp, the eight bootstrap
        # clusters are eight cameras of ONE frame: distinct enough to
        # clear the >= 6 degeneracy rule, but a single instant in time.
        timestamps = sorted({float(u[1]) for u in (decision.get("units") or [])})
        entry["unit_timestamps"] = timestamps
        entry["distinct_unit_timestamps"] = len(timestamps)
        entry["all_units_share_one_timestamp"] = len(timestamps) == 1
        for field in ("delta_render", "exact_deltas", "transaction_increment",
                      "delta_total", "ess", "n_units", "k"):
            if field in decision:
                entry[field] = decision[field]
        # -- 2. what the decision rule then pins down -------------------
        se = float(decision.get("se") or 0.0)
        entry["implied"] = {
            "rule": "accepted <=> delta_render + exact_deltas "
                    "+ transaction_increment + k*se < 0",
            "k": args.k,
            "k_times_se": args.k * se,
            "delta_total_upper_bound": -args.k * se,
            "delta_total_sign": "strictly negative",
            "delta_render_identity":
                "delta_render = sum_i w_i d_i / sum_i w_i over the "
                "SHARED SNIS weights, i.e. the weight-normalized mean of "
                "the eight per-unit paired photometric deltas",
            "per_unit_deltas_recoverable": False,
            "why_not":
                "SnisSample carries loss_incumbent/loss_candidate only in "
                "memory; run_pass keeps the AcceptanceRecord only in the "
                "RoundOutcome; trainer_hooks persists n_samples and se "
                "alone. Nothing writes d_i, w_i, delta_render, "
                "exact_deltas or transaction_increment to the checkpoint, "
                "the trial log, or the tfevents file.",
            "recomputable_at_this_checkpoint": False,
            "why_not_recomputable":
                "the eight paired renders are deterministic given the "
                "model state AT THE ROUND ITERATION; this checkpoint is "
                "later than that state (the fission was applied and "
                "training continued), so a re-render here answers a "
                "different question and must not be reported as this "
                "decision's numbers.",
        }
        report["decisions"].append(entry)

    # -- 3. does se == 0.0 imply the photometric arm contributed 0? -----
    #      Measured, not assumed.
    probe = {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "fixture": "8 units, UNEQUAL SNIS weights, identical per-unit delta",
        "common_delta_to_se": {},
    }
    for common in (0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1):
        values = [
            _se_of_common_delta(common, seed=seed) for seed in (0, 1, 7, 1234)
        ]
        probe["common_delta_to_se"][repr(common)] = {
            "se_values": [repr(v) for v in values],
            "all_exactly_zero": all(v == 0.0 for v in values),
            "max_se": max(values),
        }
    # A spread of per-unit deltas must NOT give se == 0 -- the control
    # that shows the probe can tell the two cases apart at all.
    spread = [
        SnisSample(
            unit=(i, float(i)),
            nu_density=0.5 + 0.05 * i,
            mix_density=1.0 + 0.01 * i,
            loss_incumbent=0.3,
            loss_candidate=0.3 + 0.01 * (i - 4),
        )
        for i in range(8)
    ]
    probe["control_spread_delta"] = {
        "paired_delta": paired_snis_delta(spread, 0.5),
        "se": paired_cluster_bootstrap_se(spread, 0.5, seed=0),
    }
    report["se_zero_probe"] = probe

    interpretation = []
    zero_case = probe["common_delta_to_se"][repr(0.0)]
    nonzero_bitexact = [
        key for key, value in probe["common_delta_to_se"].items()
        if key != repr(0.0) and value["all_exactly_zero"]
    ]
    interpretation.append(
        "se == 0.0 exactly is reproduced when every per-unit paired delta "
        "is exactly zero: " + ("yes" if zero_case["all_exactly_zero"] else "no")
    )
    interpretation.append(
        "common NONZERO per-unit deltas that ALSO give se == 0.0 exactly: "
        + (", ".join(nonzero_bitexact) if nonzero_bitexact else "none of those tested")
    )
    interpretation.append(
        "control (a genuine spread of per-unit deltas) gives se = "
        f"{probe['control_spread_delta']['se']!r}, so the probe "
        "distinguishes the cases."
    )
    interpretation.append(
        "CONCLUSION SCOPE: se == 0.0 rules out any SPREAD across the eight "
        "units. It does NOT by itself fix delta_render, and it says "
        "nothing about exact_deltas. Whether the evidence term or the "
        "photometric term carried the decision is NOT determined by the "
        "artifact."
    )
    report["interpretation"] = interpretation

    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print("DECOMPOSITION_JSON_BEGIN", flush=True)
    print(text, flush=True)
    print("DECOMPOSITION_JSON_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
