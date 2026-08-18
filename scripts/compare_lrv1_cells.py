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
    ap.add_argument("--a1", required=True, help="A1 lrv1_event_eval.json")
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
    a0_ret, a0_ep1 = ev("A0"), ev("A0", "event_episode1")
    gate = {}
    if a0_ret is not None and a0_ep1 is not None:
        gate["a0_event_return_psnr"] = a0_ret
        gate["a0_event_episode1_psnr"] = a0_ep1
        gate["a0_return_deficit_db"] = a0_ep1 - a0_ret
        gate["item6_reconstructible_in_principle"] = a0_ep1 > a0_ret
        gate["item3_control_shows_relevant_error"] = (a0_ep1 - a0_ret) > 0.0
    lines.append("GATE (checked on A0 alone, before any A1 number is read)")
    lines.append("  event_episode1 %s dB vs event_return %s dB  ->  deficit %s dB"
                 % (fmt(a0_ep1), fmt(a0_ret), fmt(gate.get("a0_return_deficit_db"))))
    lines.append("  item 6 reconstructible in principle : %s"
                 % gate.get("item6_reconstructible_in_principle"))
    lines.append("  item 3 control errs on this surface : %s"
                 % gate.get("item3_control_shows_relevant_error"))
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
    reading = "INDETERMINATE"
    if d1 is not None:
        if d2 is None:
            reading = ("A1 %s A0 on the event metric; the wrong-time control was "
                       "NOT run, so no timing attribution is available"
                       % ("improves on" if d1 > 0 else "does not improve on"))
        elif d1 > 0 and d2 <= 0:
            reading = ("A1 improves and A2 does not: consistent with "
                       "EPISODE-TIMING-SPECIFIC headroom")
        elif d1 > 0 and d2 > 0:
            reading = ("both improve: the gain is NOT attributable to correct "
                       "episode timing")
        else:
            reading = ("neither improves: NO demonstrated episodic headroom on "
                       "this admitted event")
    lines.append("SECTION 7 READING: " + reading)
    lines.append("")
    lines.append("No same-arm spread was measured on this scene. A small difference "
                 "is NOT resolvable from one run per arm; the frozen rule says the "
                 "honest report in that case is 'close, and no spread was measured', "
                 "and the next action is a replicate rather than a claim.")
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
