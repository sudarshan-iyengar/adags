#!/usr/bin/env python
"""Write the window package (scripts/eval_event_windows.py) of one real absence
fixture, from the frozen spec v2 (configs/n3v/absfix_gate_spec_v2.json):

  P1  roi:core per-frame masks   [A+3, B-2]   (every frame must be non-empty)
  T   event box                  [B+1, B+2]   (transition frames, reported; rev 2 s5)
  P2  event box                  [B+3, B+10]
  S1  event box                  [B+11, B+20]
  H1  event box                  [A-30, A-3]
  C1  event box                  [CA, CB]
  H2  whole frame                [0, 299]

The event box is the ``<event_prefix>_absence_gap`` bbox of the fixture's
``absfix/absence_event_masks.json`` (spec EVENT_NAME_RULE); GT is the fixture's own
cam00 frames. Nothing here reads a render.
"""
import argparse
import hashlib
import json
import os


def _sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, help="data_derived/absfix/<fixture>")
    ap.add_argument("--scene", required=True, help="spec scene key, e.g. flame_steak")
    ap.add_argument("--spec", default="configs/n3v/absfix_gate_spec_v2.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    spec = json.load(open(a.spec))
    sc = spec["scenes"][a.scene]
    A, B, CA, CB = sc["A"], sc["B"], sc["CA"], sc["CB"]
    ev_name = sc["event_name"]
    masks_json = os.path.join(a.scene_dir, "absfix", "absence_event_masks.json")
    m = json.load(open(masks_json))
    ev = [e for e in m["events"] if e["name"] == ev_name]
    if len(ev) != 1:
        raise SystemExit(f"event {ev_name!r} not found exactly once in {masks_json}")
    W, H = m["raster"]
    pkg = {
        "package_version": "1",
        "scene": os.path.basename(os.path.normpath(a.scene_dir)),
        "camera": "cam00",
        "raster": [W, H],
        "gt": {"dir": os.path.join(os.path.abspath(a.scene_dir), "images"), "pattern": "cam00_%04d.png"},
        "frames": [0, 299],
        "regions": {
            "whole": {"kind": "whole"},
            "event_box": {"kind": "bbox", "bbox": ev[0]["bbox"], "source_event": ev_name},
            "core": {"kind": "mask_dir", "dir": os.path.join(os.path.abspath(a.scene_dir), "absfix", "evaluation_rois", "core"),
                     "pattern": "%05d.png", "threshold": 127},
        },
        "windows": {
            "P1": {"region": "core", "frames": [A + 3, B - 2], "require_nonempty_every_frame": True},
            "T": {"region": "event_box", "frames": [B + 1, B + 2]},
            "P2": {"region": "event_box", "frames": [B + 3, B + 10]},
            "S1": {"region": "event_box", "frames": [B + 11, B + 20]},
            "H1": {"region": "event_box", "frames": [A - 30, A - 3]},
            "C1": {"region": "event_box", "frames": [CA, CB]},
            "H2": {"region": "whole", "frames": [0, 299]},
        },
        "source": {
            "tool": "scripts/make_window_package_absfix.py",
            "spec": os.path.abspath(a.spec), "spec_sha256": _sha(a.spec), "spec_scene": a.scene,
            "anchors": {"A": A, "B": B, "CA": CA, "CB": CB},
            "event_masks": os.path.abspath(masks_json), "event_masks_sha256": _sha(masks_json),
        },
    }
    with open(a.out, "w") as fh:
        json.dump(pkg, fh, indent=1)
    print(f"wrote {a.out}: {pkg['scene']} A={A} B={B} C=[{CA},{CB}] box={ev[0]['bbox']}")


if __name__ == "__main__":
    main()
