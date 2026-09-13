#!/usr/bin/env python
"""The two wave-2 stage-1 preconditions, evaluated from the SETUP only.

Spec v2.0.0 (`research-wiki/operations/absfix-wave2-spec-v2-2026-09-11.md`
section 11.3, thresholds in `configs/n3v/absfix_gate_spec_v2.json` keys
`MEMBERSHIP_PRECONDITION` and `T1_PRECONDITION`). Neither precondition reads a
PSNR, a render or any score: both are statements about the instrument, decided
before anything is measured with it, which is the 2026-08-24 method finding
("every frozen reading rule needs a frozen precondition asserting the mechanism
it reads was actually exercised").

MEMBERSHIP (the S2 arm, `--mode membership`). The S2 row set of one prefix is
compared against the CONSTRUCTION-DERIVED row set of the SAME prefix:

    TP / FP / FN, truth_n, pred_n, precision, recall, size ratio

plus the 2D pixel IoU between the S2 final mask and the construction mask on
one camera at two frames. Pass iff precision >= 0.80, recall >= 0.70, size
ratio in [0.5, 2.0] and BOTH frame IoUs >= 0.70. A failing prefix still trains;
its G-est-mem pair leaves the mechanism-exercised set and counts toward Claim
B's DESIGN_WITHOUT_POWER.

T1 (`--mode t1`). From T1's own emitted program: how many gated groups, what
distinct absent intervals they imply, the union interval, its temporal IoU with
the authored absent window, and the onset/offset errors in frames. Pass iff
exactly one interval, temporal IoU >= 0.80, |onset error| <= 3 and |offset
error| <= 3.

TWO READINGS OF "intervals: 1" ARE REPORTED, NOT SILENTLY CHOSEN. T1 emits one
group per gated voxel cell, and two cells of the same object can carry slightly
different intervals (wave 1 on cut_roasted_beef gated 2 cells, offsets 60/60,
onsets 90/92). `pass` applies the frozen key mechanically to the number of
DISTINCT intervals; `pass_union_reading` applies the same thresholds to the
union interval instead. Both are written, with the counts that produce them, so
the reader decides on the numbers rather than on this script's opinion.

CONVENTIONS, which are the load-bearing part:

* A program's `row_group_ids` is one entry per row of the cloud; `>= 0` means
  the row is a member of a gated group, `-1` means it is not. The two programs
  of one prefix are over the SAME cloud, so the row index is the join key --
  the script refuses two programs of different length.
* An absent interval is `[offset_frame, onset_frame - 1]` INCLUSIVE:
  `offset_frame` is the first ABSENT frame and `onset_frame` the first PRESENT
  frame after the gap. The authored window [60, 89] therefore has onset 90.
  This is the anchor semantics frozen in spec section 11.1; getting it wrong
  shifts every window by one frame, which is the defect the 2026-09-10 Codex
  review caught in prose version 1.1.0.
* Masks are read as `> 0`; a mask file that is missing is a refusal, never an
  empty mask, because an empty mask would score IoU 0.0 and look like a
  measured failure rather than a missing input.

Everything except `main` is importable without torch, without SAM2 and without
a checkpoint (see `tests/test_wave2_preconditions.py`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class Refusal(RuntimeError):
    """An input the precondition must not silently work around."""


# --------------------------------------------------------------------------
# membership
# --------------------------------------------------------------------------
def member_mask(row_group_ids: Sequence[int]) -> np.ndarray:
    """Boolean membership column from a program's `row_group_ids`."""
    arr = np.asarray(list(row_group_ids))
    if arr.ndim != 1:
        raise Refusal("row_group_ids must be one-dimensional, got %r" % (arr.shape,))
    return arr >= 0


def membership_counts(truth_ids: Sequence[int],
                      pred_ids: Sequence[int]) -> Dict[str, float]:
    """TP/FP/FN and the rates the precondition reads.

    `truth` is the construction-derived row set, `pred` the S2 row set, both of
    the same prefix and therefore of the same cloud.
    """
    truth = member_mask(truth_ids)
    pred = member_mask(pred_ids)
    if truth.shape != pred.shape:
        raise Refusal(
            "the two programs cover different clouds: truth has %d rows, "
            "prediction %d" % (truth.size, pred.size))
    tp = int(np.count_nonzero(truth & pred))
    fp = int(np.count_nonzero(~truth & pred))
    fn = int(np.count_nonzero(truth & ~pred))
    truth_n = int(np.count_nonzero(truth))
    pred_n = int(np.count_nonzero(pred))
    return {
        "n_rows": int(truth.size),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "truth_n": truth_n,
        "pred_n": pred_n,
        "precision": (tp / pred_n) if pred_n else 0.0,
        "recall": (tp / truth_n) if truth_n else 0.0,
        "size_ratio": (pred_n / truth_n) if truth_n else float("inf"),
    }


def pixel_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> Dict[str, float]:
    """Pixel intersection, union and IoU of two binary masks."""
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    if a.shape != b.shape:
        raise Refusal("masks disagree on shape: %r vs %r" % (a.shape, b.shape))
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return {
        "pixel_intersection": inter,
        "pixel_union": union,
        "iou": (inter / union) if union else 0.0,
        "area_construction": int(np.count_nonzero(a)),
        "area_s2": int(np.count_nonzero(b)),
    }


def evaluate_membership(counts: Dict[str, float],
                        iou_by_frame: Dict[str, Dict[str, float]],
                        thresholds: Dict) -> Dict:
    """Apply MEMBERSHIP_PRECONDITION; return pass/fail with every reason."""
    p_min = float(thresholds["precision_min"])
    r_min = float(thresholds["recall_min"])
    lo, hi = (float(x) for x in thresholds["size_ratio"])
    iou_min = float(thresholds["mask_iou_cam15_min"])
    reasons: List[str] = []
    if counts["precision"] < p_min:
        reasons.append("precision %.4f < %.2f" % (counts["precision"], p_min))
    if counts["recall"] < r_min:
        reasons.append("recall %.4f < %.2f" % (counts["recall"], r_min))
    if not (lo <= counts["size_ratio"] <= hi):
        reasons.append("size ratio %.4f outside [%.2f, %.2f]"
                       % (counts["size_ratio"], lo, hi))
    for key in sorted(iou_by_frame):
        iou = iou_by_frame[key]["iou"]
        if iou < iou_min:
            reasons.append("%s IoU %.4f < %.2f" % (key, iou, iou_min))
    return {
        "pass": not reasons,
        "reasons": reasons,
        "thresholds": {
            "precision_min": p_min, "recall_min": r_min,
            "size_ratio": [lo, hi], "mask_iou_min": iou_min,
        },
    }


# --------------------------------------------------------------------------
# T1
# --------------------------------------------------------------------------
def program_intervals(program: Dict) -> List[Tuple[int, int]]:
    """INCLUSIVE absent intervals `[offset_frame, onset_frame - 1]` per group.

    Groups T1 abstained on carry `offset_frame`/`onset_frame` `None` and are
    not intervals; they are counted separately by the caller.
    """
    out = []
    for group in program.get("groups", []):
        off = group.get("offset_frame")
        on = group.get("onset_frame")
        if off is None or on is None:
            continue
        out.append((int(off), int(on) - 1))
    return out


def temporal_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """IoU of two INCLUSIVE integer frame intervals."""
    a0, a1 = int(a[0]), int(a[1])
    b0, b1 = int(b[0]), int(b[1])
    if a1 < a0 or b1 < b0:
        raise Refusal("interval runs backwards: %r %r" % (a, b))
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return (inter / union) if union else 0.0


def evaluate_t1(program: Dict, truth_gap: Sequence[int],
                thresholds: Dict) -> Dict:
    """Apply T1_PRECONDITION; report both readings of `intervals: 1`."""
    truth = (int(truth_gap[0]), int(truth_gap[1]))
    intervals = program_intervals(program)
    distinct = sorted(set(intervals))
    n_groups = len(program.get("groups", []))
    iou_min = float(thresholds["temporal_iou_min"])
    on_max = int(thresholds["onset_error_max_frames"])
    off_max = int(thresholds["offset_error_max_frames"])
    want_n = int(thresholds["intervals"])

    per_group = [
        {
            "interval": [a, b],
            "offset_error": a - truth[0],
            "onset_error": (b + 1) - (truth[1] + 1),
            "temporal_iou": temporal_iou((a, b), truth),
        }
        for a, b in intervals
    ]
    if intervals:
        union = (min(a for a, _ in intervals), max(b for _, b in intervals))
        union_block = {
            "interval": [union[0], union[1]],
            "offset_error": union[0] - truth[0],
            "onset_error": (union[1] + 1) - (truth[1] + 1),
            "temporal_iou": temporal_iou(union, truth),
        }
    else:
        union_block = None

    def _reasons(block, n_intervals, label):
        out = []
        if n_intervals != want_n:
            out.append("%s: %d interval(s), the precondition requires %d"
                       % (label, n_intervals, want_n))
        if block is None:
            out.append("%s: no interval was estimated" % label)
            return out
        if block["temporal_iou"] < iou_min:
            out.append("%s: temporal IoU %.4f < %.2f"
                       % (label, block["temporal_iou"], iou_min))
        if abs(block["offset_error"]) > off_max:
            out.append("%s: |offset error| %d > %d"
                       % (label, abs(block["offset_error"]), off_max))
        if abs(block["onset_error"]) > on_max:
            out.append("%s: |onset error| %d > %d"
                       % (label, abs(block["onset_error"]), on_max))
        return out

    # With one distinct interval the union IS that interval, so the two
    # readings differ only in the interval COUNT they are allowed.
    strict_reasons = _reasons(union_block, len(distinct), "distinct-interval reading")
    union_reasons = _reasons(union_block, 1 if union_block else 0, "union reading")

    return {
        "truth_gap_inclusive": [truth[0], truth[1]],
        "truth_onset_frame": truth[1] + 1,
        "n_groups_in_program": n_groups,
        "n_gated_groups": len(intervals),
        "n_distinct_intervals": len(distinct),
        "distinct_intervals": [[a, b] for a, b in distinct],
        "per_group": per_group,
        "union": union_block,
        "thresholds": {
            "intervals": want_n, "temporal_iou_min": iou_min,
            "onset_error_max_frames": on_max, "offset_error_max_frames": off_max,
        },
        "pass": not strict_reasons,
        "reasons": strict_reasons,
        "pass_union_reading": not union_reasons,
        "reasons_union_reading": union_reasons,
        "reading_note": (
            "T1 emits one group per gated voxel cell, so two cells of one "
            "object can carry slightly different intervals. `pass` applies "
            "the frozen `intervals: 1` to the number of DISTINCT intervals; "
            "`pass_union_reading` applies the same thresholds to their union. "
            "Both are reported; neither is chosen here."
        ),
    }


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_mask(root, camera: int, frame: int, subdir: str) -> np.ndarray:
    from PIL import Image
    path = Path(root) / ("cam%02d" % int(camera))
    if subdir:
        path = path / subdir
    path = path / ("%04d.png" % int(frame))
    if not path.exists():
        raise Refusal("no mask at %s" % path)
    with Image.open(path) as handle:
        return np.array(handle.convert("L"))


# --------------------------------------------------------------------------
def run_membership(a) -> Dict:
    spec = read_json(a.spec)
    thresholds = spec["MEMBERSHIP_PRECONDITION"]
    truth_prog = read_json(a.construction_program)
    pred_prog = read_json(a.s2_program)
    counts = membership_counts(truth_prog["row_group_ids"],
                               pred_prog["row_group_ids"])
    iou_by_frame = {}
    for frame in a.iou_frames:
        con = read_mask(a.construction_mask_root, a.iou_camera, frame,
                        a.construction_mask_subdir)
        s2 = read_mask(a.s2_mask_root, a.iou_camera, frame, a.s2_mask_subdir)
        iou_by_frame["cam%02d_f%d" % (int(a.iou_camera), int(frame))] = \
            pixel_iou(con, s2)
    verdict = evaluate_membership(counts, iou_by_frame, thresholds)
    return {
        "tool": "scripts/wave2_preconditions.py",
        "spec": "absfix-wave2-spec-v2-2026-09-11 section 11.3 "
                "(MEMBERSHIP_PRECONDITION)",
        "mode": "membership",
        "scene": a.scene,
        "prefix_seed": a.seed,
        "inputs": {
            "construction_program": str(Path(a.construction_program).resolve()),
            "construction_program_sha256": sha256_file(a.construction_program),
            "s2_program": str(Path(a.s2_program).resolve()),
            "s2_program_sha256": sha256_file(a.s2_program),
            "construction_mask_root": str(Path(a.construction_mask_root).resolve()),
            "s2_mask_root": str(Path(a.s2_mask_root).resolve()),
            "iou_camera": int(a.iou_camera),
            "iou_frames": [int(f) for f in a.iou_frames],
            "spec_file": str(Path(a.spec).resolve()),
            "spec_sha256": sha256_file(a.spec),
        },
        "counts": counts,
        "mask_iou": iou_by_frame,
        "verdict": verdict,
        "on_failure": thresholds.get("on_failure"),
    }


def run_t1(a) -> Dict:
    spec = read_json(a.spec)
    thresholds = spec["T1_PRECONDITION"]
    program = read_json(a.t1_program)
    result = evaluate_t1(program, a.truth_gap, thresholds)
    out = {
        "tool": "scripts/wave2_preconditions.py",
        "spec": "absfix-wave2-spec-v2-2026-09-11 section 11.3 (T1_PRECONDITION)",
        "mode": "t1",
        "scene": a.scene,
        "prefix_seed": a.seed,
        "inputs": {
            "t1_program": str(Path(a.t1_program).resolve()),
            "t1_program_sha256": sha256_file(a.t1_program),
            "spec_file": str(Path(a.spec).resolve()),
            "spec_sha256": sha256_file(a.spec),
        },
        "on_failure": thresholds.get("on_failure"),
    }
    if a.t1_report:
        out["inputs"]["t1_report"] = str(Path(a.t1_report).resolve())
        out["inputs"]["t1_report_sha256"] = sha256_file(a.t1_report)
        report = read_json(a.t1_report)
        out["t1_report_extract"] = {
            "checkpoint": report.get("checkpoint"),
            "source_path": report.get("source_path"),
            "program_sha256": report.get("program_sha256"),
            "anti_leakage": report.get("anti_leakage"),
            "grouping_n_groups": (report.get("grouping") or {}).get("n_groups"),
        }
    out.update(result)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["membership", "t1"])
    p.add_argument("--spec", required=True,
                   help="configs/n3v/absfix_gate_spec_v2.json")
    p.add_argument("--scene", required=True)
    p.add_argument("--seed", type=int, required=True, help="prefix seed")
    p.add_argument("--out", required=True)
    # membership
    p.add_argument("--construction_program", default="")
    p.add_argument("--s2_program", default="")
    p.add_argument("--construction_mask_root", default="")
    p.add_argument("--s2_mask_root", default="")
    p.add_argument("--construction_mask_subdir", default="pseudo_label/object_mask")
    p.add_argument("--s2_mask_subdir", default="pseudo_label/object_mask")
    p.add_argument("--iou_camera", type=int, default=15)
    p.add_argument("--iou_frames", nargs="+", type=int, default=[50, 95])
    # t1
    p.add_argument("--t1_program", default="")
    p.add_argument("--t1_report", default="")
    p.add_argument("--truth_gap", nargs=2, type=int, default=[60, 89],
                   metavar=("FIRST_ABSENT", "LAST_ABSENT"))
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.mode == "membership":
            for name in ("construction_program", "s2_program",
                         "construction_mask_root", "s2_mask_root"):
                if not getattr(a, name):
                    raise Refusal("--%s is required in membership mode" % name)
            doc = run_membership(a)
        else:
            if not a.t1_program:
                raise Refusal("--t1_program is required in t1 mode")
            doc = run_t1(a)
    except Refusal as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(doc, indent=1, sort_keys=True))
    print("[precondition] %s  PASS=%s  sha256 %s"
          % (a.out, doc.get("verdict", doc).get("pass")
             if a.mode == "membership" else doc.get("pass"),
             sha256_file(a.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
