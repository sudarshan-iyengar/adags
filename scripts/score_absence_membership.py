#!/usr/bin/env python
"""Score an ESTIMATED episode program against the fixture's TRUTH program.

Both inputs are `adags-episode-program-v2` artifacts in `row_ids` mode
computed on the SAME cloud (the prefix checkpoint), so a row-set comparison
is exact: a row is a member iff its `row_group_ids` entry is >= 0.

Reported, per pair:
  * row-set precision / recall / Jaccard and the four counts;
  * the temporal IoU (in frames) between the estimate's gap and the authored
    gap, plus the offset/onset errors in frames;
  * optional weighting by a per-row weight vector (`--weights path.npy`,
    e.g. activated opacity or rendered contribution) so that the same
    precision/recall are also given weight-wise.

No threshold is tuned here; the script is a reader of frozen artifacts and
refuses programs that were not computed on the same cloud.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCHEMA = "adags-episode-program-v2"


class ContractError(RuntimeError):
    pass


def load_program(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA:
        raise ContractError("%s is not a %s artifact" % (path, SCHEMA))
    if payload.get("membership_mode") != "row_ids":
        raise ContractError("%s must be in row_ids mode" % path)
    return payload


def member_mask(payload):
    ids = np.asarray(payload.get("row_group_ids") or [], dtype=np.int64)
    if ids.size == 0:
        raise ContractError("program carries no row_group_ids column")
    return ids >= 0


def check_same_cloud(a, b):
    ca, cb = a.get("cloud", {}), b.get("cloud", {})
    if int(ca.get("n_rows", -1)) != int(cb.get("n_rows", -2)):
        raise ContractError("programs were computed on clouds of different "
                            "size (%r vs %r)" % (ca.get("n_rows"),
                                                 cb.get("n_rows")))
    if ca.get("xyz_sha256") and cb.get("xyz_sha256") and \
            ca["xyz_sha256"] != cb["xyz_sha256"]:
        raise ContractError("programs were computed on different clouds "
                            "(xyz fingerprints differ)")


def set_scores(est, truth, weights=None):
    est = np.asarray(est, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if est.shape != truth.shape:
        raise ContractError("row sets have different lengths")
    if weights is None:
        w = np.ones(est.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != est.shape:
            raise ContractError("weights length does not match the row sets")
    tp = float(w[est & truth].sum())
    fp = float(w[est & ~truth].sum())
    fn = float(w[~est & truth].sum())
    union = tp + fp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "n_est": float(w[est].sum()), "n_truth": float(w[truth].sum()),
        "precision": (tp / (tp + fp)) if (tp + fp) > 0 else None,
        "recall": (tp / (tp + fn)) if (tp + fn) > 0 else None,
        "jaccard": (tp / union) if union > 0 else None,
    }


def gap_frames(payload):
    groups = payload.get("groups") or []
    if not groups:
        return None
    offset = min(int(g["offset_frame"]) for g in groups)
    onset = max(int(g["onset_frame"]) for g in groups)
    return offset, onset - 1  # inclusive absent frames


def temporal_iou(est_gap, authored_gap):
    """IoU of two INCLUSIVE frame intervals [A, B]; 0 when disjoint."""
    if est_gap is None:
        return 0.0
    a0, a1 = int(est_gap[0]), int(est_gap[1])
    b0, b1 = int(authored_gap[0]), int(authored_gap[1])
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return float(inter) / float(union) if union > 0 else 0.0


def score(est_payload, truth_payload, authored_gap, weights=None):
    check_same_cloud(est_payload, truth_payload)
    est = member_mask(est_payload)
    truth = member_mask(truth_payload)
    out = {
        "rows": set_scores(est, truth),
        "n_rows": int(est.shape[0]),
    }
    if weights is not None:
        out["weighted"] = set_scores(est, truth, weights)
    eg = gap_frames(est_payload)
    out["gap"] = {
        "estimated_absent_frames": list(eg) if eg else None,
        "authored_absent_frames": [int(authored_gap[0]), int(authored_gap[1])],
        "temporal_iou": temporal_iou(eg, authored_gap),
        "offset_error_frames": (eg[0] - int(authored_gap[0])) if eg else None,
        "onset_error_frames": (eg[1] - int(authored_gap[1])) if eg else None,
    }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--authored_gap", nargs=2, type=int, required=True,
                        help="INCLUSIVE authored absent frames A B")
    parser.add_argument("--weights", default="",
                        help="optional .npy per-row weights (same cloud)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    est = load_program(args.estimate)
    truth = load_program(args.truth)
    weights = np.load(args.weights) if args.weights else None
    report = score(est, truth, args.authored_gap, weights)
    report["inputs"] = {"estimate": str(args.estimate),
                        "truth": str(args.truth),
                        "weights": str(args.weights) or None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    r = report["rows"]
    print("rows: est %d truth %d tp %d precision %s recall %s jaccard %s"
          % (r["n_est"], r["n_truth"], r["tp"], r["precision"], r["recall"],
             r["jaccard"]))
    print("gap: est %r authored %r temporal_iou %.4f"
          % (report["gap"]["estimated_absent_frames"],
             report["gap"]["authored_absent_frames"],
             report["gap"]["temporal_iou"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
