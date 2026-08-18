"""Apply the frozen LRV1 outcome rule to the per-cell event-eval artifacts.

EXPLORATORY. Reads the `lrv1_event_eval.json` written by
`scripts/eval_lrv1_event.py` for each cell and emits the comparison table plus
the section-7 reading from
`research-wiki/operations/lrv1-oracle-headroom-spec-2026-08-19.md`.

It is a reducer over frozen text and primary artifacts: it applies rules that
were fixed before any cell ran, and it does not choose them.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(v, nd=4):
    return "n/a" if v is None else ("%.*f" % (nd, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", required=True, help="A0 lrv1_event_eval.json")
    ap.add_argument("--a1", default=None, help="A1 lrv1_event_eval.json; omit to obtain the A0-only GATE verdict, which the frozen order requires before any A1 number is read")
    ap.add_argument("--a2", default=None, help="A2 lrv1_event_eval.json (if run)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells = {}
    for key, path in (("A0", args.a0), ("A1", args.a1), ("A2", args.a2)):
        if not path:
            continue
        p = Path(path)
        cells[key] = json.loads(p.read_text())
        cells[key]["_artifact"] = {"path": str(p), "sha256": sha256_file(p),
                                   "bytes": p.stat().st_size}

    def ev(cell, region="event_return", field="psnr_pooled"):
        return cells[cell][region].get(field) if cell in cells else None

    lines = []
    lines.append("LRV1 oracle-headroom comparison")
    lines.append("=" * 78)
    lines.append("")
    hdr = "%-22s %10s %10s %10s" % ("region (PSNR pooled)", "A0", "A1", "A2")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for region in ("event_return", "event_episode1", "ghost_gap",
                   "ordinary_return", "ordinary_all", "whole_frame"):
        lines.append("%-22s %10s %10s %10s" % (
            region, fmt(ev("A0", region)), fmt(ev("A1", region)),
            fmt(ev("A2", region))))
    lines.append("")
    lines.append("%-22s %10s %10s %10s" % (
        "whole-frame SSIM",
        fmt(cells.get("A0", {}).get("whole_frame", {}).get("ssim")),
        fmt(cells.get("A1", {}).get("whole_frame", {}).get("ssim")),
        fmt(cells.get("A2", {}).get("whole_frame", {}).get("ssim"))))
    for key in ("primitives", "checkpoint_iteration", "held_out_views_scored"):
        lines.append("%-22s %10s %10s %10s" % (
            key,
            cells.get("A0", {}).get(key, "n/a"),
            cells.get("A1", {}).get(key, "n/a"),
            cells.get("A2", {}).get(key, "n/a")))
    lines.append("")

    # ---- gate items 3 and 6, on A0 only -----------------------------------
    # Fixed before any cell output was read. Items 3 and 6 must NOT be the same
    # predicate: as first written both reduced to `a0_ep1 > a0_ret`, so a 0.01 dB
    # deficit passed both even if A0 reconstructed the surface badly everywhere.
    # Item 6 is now an ABSOLUTE floor on the control's episode-1 quality; item 3
    # is a MINIMUM deficit.
    ITEM6_EPISODE1_FLOOR_DB = 25.0
    ITEM3_MIN_DEFICIT_DB = 1.0
    a0_ret, a0_ep1 = ev("A0"), ev("A0", "event_episode1")
    gate = {"item6_episode1_floor_db": ITEM6_EPISODE1_FLOOR_DB,
            "item3_min_deficit_db": ITEM3_MIN_DEFICIT_DB}
    if a0_ret is not None and a0_ep1 is not None:
        gate["a0_event_return_psnr"] = a0_ret
        gate["a0_event_episode1_psnr"] = a0_ep1
        gate["a0_return_deficit_db"] = a0_ep1 - a0_ret
        gate["item6_reconstructible_in_principle"] = a0_ep1 >= ITEM6_EPISODE1_FLOOR_DB
        gate["item3_control_shows_relevant_error"] = (
            (a0_ep1 - a0_ret) >= ITEM3_MIN_DEFICIT_DB)
        gate["passed"] = bool(gate["item6_reconstructible_in_principle"]
                              and gate["item3_control_shows_relevant_error"])
    lines.append("GATE (A0 alone; thresholds fixed before any cell output was read)")
    lines.append("  event_episode1 %s dB vs event_return %s dB  ->  deficit %s dB"
                 % (fmt(a0_ep1), fmt(a0_ret), fmt(gate.get("a0_return_deficit_db"))))
    lines.append("  item 6  episode1 >= %.1f dB (reconstructible)      : %s"
                 % (ITEM6_EPISODE1_FLOOR_DB,
                    gate.get("item6_reconstructible_in_principle")))
    lines.append("  item 3  deficit   >= %.1f dB (control errs here)   : %s"
                 % (ITEM3_MIN_DEFICIT_DB,
                    gate.get("item3_control_shows_relevant_error")))
    lines.append("  GATE PASSED: %s" % gate.get("passed"))
    if gate.get("passed") is False:
        lines.append("  -> the scene is UNSUITABLE / INCONCLUSIVE for this question; "
                     "no representation verdict follows from any A1 number.")
    lines.append("")

    d1 = (ev("A1") - a0_ret) if (ev("A1") is not None and a0_ret is not None) else None
    d2 = (ev("A2") - a0_ret) if (ev("A2") is not None and a0_ret is not None) else None
    lines.append("DELTAS on the decisive metric (event_return, held-out, return frames)")
    lines.append("  D1 = A1 - A0 = %s dB" % fmt(d1))
    lines.append("  D2 = A2 - A0 = %s dB" % fmt(d2))
    if d1 is not None and d2 is not None:
        lines.append("  A1 - A2      = %s dB   <- the clean timing comparison" % fmt(d1 - d2))
    lines.append("")

    # ---- section 7 reading -------------------------------------------------
    # A bare `d1 > 0` would call +0.0001 dB a win. Without a measured same-arm
    # spread on this scene, anything under the project's standing 0.5 dB
    # decision floor is reported as INDETERMINATE rather than as a direction.
    # The floor is inherited from the matched-triple spec's `max(0.5 dB, S)`; it
    # is not a number chosen to fit an observed result -- it is fixed here while
    # no cell output has been read.
    RESOLUTION_DB = 0.5
    reading = "INDETERMINATE"
    if d1 is not None:
        if abs(d1) < RESOLUTION_DB:
            reading = ("INDETERMINATE: |D1| = %.4f dB is below the %.1f dB "
                       "resolution floor and no same-arm spread was measured on "
                       "this scene. Close, and not resolvable from one run per "
                       "arm -- the next action is a replicate, not a claim."
                       % (abs(d1), RESOLUTION_DB))
        elif d2 is None:
            reading = ("A1 %s A0 by %.4f dB on the event metric; the wrong-time "
                       "control was NOT run, so no timing attribution is available"
                       % ("improves on" if d1 > 0 else "falls below", abs(d1)))
        elif abs(d1 - d2) < RESOLUTION_DB:
            reading = ("A1 and A2 differ by only %.4f dB, below the %.1f dB "
                       "resolution floor: the gain (if any) is NOT attributable "
                       "to correct episode timing"
                       % (abs(d1 - d2), RESOLUTION_DB))
        elif d1 > 0 and d1 - d2 >= RESOLUTION_DB:
            reading = ("A1 improves and exceeds the wrong-time control by "
                       "%.4f dB: consistent with EPISODE-TIMING-SPECIFIC headroom"
                       % (d1 - d2))
        elif d1 <= -RESOLUTION_DB:
            reading = ("A1 falls below A0 by %.4f dB: NO demonstrated episodic "
                       "headroom on this admitted event" % abs(d1))
        else:
            reading = ("mixed: D1 = %.4f, D2 = %.4f dB -- report both, claim "
                       "neither" % (d1, d2))
    lines.append("SECTION 7 READING: " + reading)
    lines.append("")
    lines.append("No same-arm spread was measured on this scene. A difference below "
                 "the %.1f dB floor is NOT resolvable from one run per arm; the "
                 "frozen rule says the honest report in that case is 'close, and no "
                 "spread was measured', and the next action is a replicate rather "
                 "than a claim." % RESOLUTION_DB)
    lines.append("")
    lines.append("Artifacts:")
    for key in ("A0", "A1", "A2"):
        if key in cells:
            a = cells[key]["_artifact"]
            lines.append("  %s  %s  sha256 %s" % (key, a["path"], a["sha256"]))

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "schema_version": "lrv1-comparison-v1",
            "evidence_bearing": False,
            "cells": {k: {kk: vv for kk, vv in v.items()} for k, v in cells.items()},
            "gate": gate,
            "D1_a1_minus_a0_db": d1,
            "D2_a2_minus_a0_db": d2,
            "a1_minus_a2_db": (d1 - d2) if (d1 is not None and d2 is not None) else None,
            "section7_reading": reading,
            "table": text,
        }, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
