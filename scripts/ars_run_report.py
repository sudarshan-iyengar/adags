#!/usr/bin/env python
"""Visual run reports for the absence-return suite (ARS): the renders, ground truth, error maps and numbers of a
trained baseline run, in one folder, without touching the method's code.

The external substrates (Ex4DGS, Multi4D, STG) save cam00 renders in their own layouts and, for Ex4DGS, no ground
truth at all. This tool reads what a run already saved (renders, point clouds, logs) plus the scene's frozen ground
truth and the run's verified score from scripts/eval_event_windows.py (via ars_score.py), and writes:

  run mode   (--run ...)   -> <out>/
      stats.json        the score's window table (PSNR/SSIM/LPIPS per window), whole-frame per-frame PSNR,
                        native metrics, Gaussian counts per bank, training summary, provenance, input hashes
      perframe.png      per-frame PSNR of the whole frame, static background, ordinary region and target/ghost
      montage.jpg       key frames: ground truth | render | |error| x8, plus a zoom on the event region
  scene mode (--scene_overview ...) -> one image per scene comparing every run of every method at the key frames

Nothing here enters a score: the numbers are copied from the verified score file, and images are for looking.
"""
import argparse
import glob
import hashlib
import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw

KEY_FRAMES = [(30, "before the gap", "target"), (75, "mid-gap (target absent)", "ghost"),
              (92, "return", "target"), (150, "late", "target"), (250, "late", "target")]
WINDOWS = ["P1", "P2", "S1", "T", "H1", "C1", "H2", "ORD", "STAT"]
WINDOW_WORDS = {"P1": "gap (ghost region)", "P2": "return (target)", "S1": "settled return (target)",
                "T": "transition (target)", "H1": "before the gap (target)", "C1": "late control (target)",
                "H2": "whole frame", "ORD": "ordinary region", "STAT": "static background"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))


def ply_count(p):
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as f:
        head = f.read(4096).decode("latin-1")
    m = re.search(r"element vertex (\d+)", head)
    return int(m.group(1)) if m else None


def render_layout(method, run):
    """(render dir, printf pattern) for absolute frame f, as ars_score.py reads them."""
    if method == "ex4dgs":
        return os.path.join(run, "model", "test", "itrs_40000", "renders"), "%06d.png"
    if method == "multi4d":
        return os.path.join(run, "final", "iteration_20000", "render"), "%d.png"
    return os.path.join(run, "stg", "renders"), "%05d.png"


def gt_layout(scene_dir):
    pkg = json.load(open(os.path.join(scene_dir, "gt", "window_package.json")))
    d = os.path.normpath(os.path.join(scene_dir, "gt", pkg["gt"]["dir"]))
    return d, pkg["gt"]["pattern"], pkg


def mask(scene_dir, region, f):
    p = os.path.join(scene_dir, "gt", "masks", region, "%05d.png" % f)
    return (np.asarray(Image.open(p).convert("L")) > 127) if os.path.isfile(p) else None


def zoom_box(scene_dir, region, f, shape, pad=40, minside=160):
    m = mask(scene_dir, region, f)
    if m is None or not m.any():
        m = mask(scene_dir, "target", 30)
    if m is None or not m.any():
        return None
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0, minside)
    x0, x1 = int(max(0, cx - side / 2)), int(min(shape[1], cx + side / 2))
    y0, y1 = int(max(0, cy - side / 2)), int(min(shape[0], cy + side / 2))
    return x0, y0, x1, y1


def err_img(r, g, gain=8):
    e = np.abs(r.astype(np.int16) - g.astype(np.int16)).mean(2) * gain
    return np.repeat(np.clip(e, 0, 255).astype(np.uint8)[..., None], 3, 2)


def psnr(r, g):
    m = ((r.astype(np.float64) - g.astype(np.float64)) ** 2).mean() / 255.0 ** 2
    return float("inf") if m == 0 else 10 * np.log10(1 / m)


def label_tile(img, text, w, h):
    t = Image.fromarray(img).resize((w, h), Image.BILINEAR)
    canvas = Image.new("RGB", (w, h + 18), "white")
    canvas.paste(t, (0, 18))
    ImageDraw.Draw(canvas).text((4, 3), text, fill="black")
    return canvas


def stack(rows):
    W = max(r.width for r in rows)
    out = Image.new("RGB", (W, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    return out


def hstack(tiles):
    out = Image.new("RGB", (sum(t.width for t in tiles), max(t.height for t in tiles)), "white")
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width
    return out


# ------------------------------------------------------------------ run mode

def gaussian_counts(method, run):
    if method == "ex4dgs":
        pc = os.path.join(run, "model", "point_cloud")
        return {"iteration_%s" % it: {"static": ply_count(os.path.join(pc, "iteration_%s" % it, "point_cloud.ply")),
                                      "dynamic": ply_count(os.path.join(pc, "iteration_%s" % it,
                                                                        "dynamic_point_cloud.ply"))}
                for it in ("30000", "40000")}
    if method == "multi4d":
        pc = os.path.join(run, "model", "point_cloud", "iteration_20000")
        return {"iteration_20000": {"persistent_foreground": ply_count(os.path.join(pc, "point_cloud.ply")),
                                    "background": ply_count(os.path.join(pc, "point_cloud_second.ply")),
                                    "transient": ply_count(os.path.join(pc, "point_cloud_transient.ply"))}}
    return {"segment_%d" % o: ply_count(os.path.join(run, "colmap_%d" % o, "point_cloud", "iteration_25000",
                                                     "point_cloud.ply")) for o in range(0, 300, 50)}


def native_metrics(method, run):
    if method == "ex4dgs":
        p = os.path.join(run, "model", "mean_metrics.json")
        return json.load(open(p)) if os.path.isfile(p) else None
    if method == "multi4d":
        out = {}
        p = os.path.join(run, "final", "iteration_20000", "render_record.json")
        if os.path.isfile(p):
            r = json.load(open(p))
            out["final_20000_mean_psnr"] = r.get("native_mean_psnr_debug_metrics")
        dm = glob.glob(os.path.join(run, "results", "*", "debug_metrics.json"))
        if dm:
            rows = [json.loads(l) for l in open(dm[0]) if l.strip()]
            if rows:
                pk = max(rows, key=lambda r: r["psnr"])
                out["peak_psnr"] = pk["psnr"]
                out["peak_iteration"] = pk["iteration"]
                out["curve_psnr_every_500"] = [(r["iteration"], round(r["psnr"], 4)) for r in rows]
        return out
    return None


def training_summary(log):
    """Mean of the last 200 training-PSNR readings in an Ex4DGS/STG tqdm log (training views, not held out)."""
    if not log or not os.path.isfile(log):
        return None
    txt = open(log, "rb").read().decode("utf-8", "replace").replace("\r", "\n")
    ps = [float(m) for m in re.findall(r"PSNR=([0-9.]+)", txt)]
    if not ps:
        return None
    tail = ps[-200:]
    return {"source": log, "train_psnr_last200_mean": sum(tail) / len(tail), "n_readings": len(ps)}


def run_mode(a):
    gdir, gpat, pkg = gt_layout(a.scene_dir)
    rd, rpat = render_layout(a.method, a.run)
    score = json.load(open(a.score))
    os.makedirs(a.out, exist_ok=True)
    wins = score["windows"]

    stats = {
        "tool": "scripts/ars_run_report.py", "tool_sha256": sha(os.path.abspath(__file__)),
        "method": a.method, "scene": pkg["scene"], "run": a.run, "label": a.label,
        "score_file": a.score, "score_sha256": sha(a.score),
        "windows": {w: {"words": WINDOW_WORDS.get(w, w), "frames": wins[w]["frames"], "region": wins[w]["region"],
                        "psnr": wins[w]["psnr"].get("psnr"), "ssim": wins[w].get("ssim", {}).get("value"),
                        "lpips": wins[w].get("lpips", {}).get("value")} for w in WINDOWS if w in wins},
        "whole_frame_psnr_per_frame": [x["psnr"] for x in wins["H2"]["per_frame"]],
        "native_metrics": native_metrics(a.method, a.run),
        "gaussian_counts": gaussian_counts(a.method, a.run),
        "training": training_summary(a.train_log),
        "run_checks": score.get("run_checks"),
    }
    prov = os.path.join(a.run, "provenance.txt")
    if os.path.isfile(prov):
        stats["provenance"] = dict(l.rstrip("\n").split("=", 1) for l in open(prov) if "=" in l)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)

    # per-frame curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    for w, c in (("H2", "C0"), ("STAT", "C2"), ("ORD", "C1")):
        pf = wins[w]["per_frame"]
        ax.plot([x["frame"] for x in pf], [x["psnr"] for x in pf], color=c, lw=1, label=WINDOW_WORDS[w])
    for w, c in (("P1", "C3"), ("H1", "C4"), ("P2", "C4"), ("S1", "C4"), ("C1", "C4")):
        pf = wins[w]["per_frame"]
        ax.plot([x["frame"] for x in pf], [x["psnr"] for x in pf], color=c, lw=1.6,
                label={"P1": "gap (ghost region)", "H1": "target (before gap, return, late)"}.get(w))
    ax.axvspan(60, 89, color="orange", alpha=0.12, label="target absent (60-89)")
    ax.set_xlabel("frame"); ax.set_ylabel("PSNR vs frozen GT, cam00 (dB)"); ax.grid(alpha=.3)
    ax.set_title("%s  %s  (whole frame %.2f dB, gap %.2f, return %.2f)" % (a.label, pkg["scene"], wins["H2"]["psnr"]["psnr"],
                 wins["P1"]["psnr"]["psnr"], wins["P2"]["psnr"]["psnr"]))
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "perframe.png"), dpi=110); plt.close(fig)

    # montage
    rows = []
    for f, words, region in KEY_FRAMES:
        g = rgb(os.path.join(gdir, gpat % f)); r = rgb(os.path.join(rd, rpat % f))
        tiles = [label_tile(g, "frame %d, %s: ground truth" % (f, words), 450, 338),
                 label_tile(r, "render  %.2f dB" % psnr(r, g), 450, 338),
                 label_tile(err_img(r, g), "|error| x8", 450, 338)]
        b = zoom_box(a.scene_dir, region, f, g.shape)
        if b:
            x0, y0, x1, y1 = b
            tiles += [label_tile(g[y0:y1, x0:x1], "zoom: ground truth", 300, 300),
                      label_tile(r[y0:y1, x0:x1], "zoom: render", 300, 300)]
        rows.append(hstack(tiles))
    stack(rows).save(os.path.join(a.out, "montage.jpg"), quality=88)
    print("report written to %s" % a.out)


# ------------------------------------------------------------------ scene mode

def scene_mode(a):
    """One image per scene: every run of every method at the key frames, full frame plus event zoom."""
    gdir, gpat, pkg = gt_layout(a.scene_dir)
    cols = json.load(open(a.scene_overview))  # [{"label":..., "method":..., "run":..., "score":...}, ...]
    rows = []
    for f, words, region in KEY_FRAMES[:3] + KEY_FRAMES[3:4]:
        g = rgb(os.path.join(gdir, gpat % f))
        b = zoom_box(a.scene_dir, region, f, g.shape)
        full = [label_tile(g, "frame %d, %s: GT" % (f, words), 338, 254)]
        zoom = [label_tile(g[b[1]:b[3], b[0]:b[2]], "zoom: GT", 338, 254)] if b else []
        for c in cols:
            rd, rpat = render_layout(c["method"], c["run"])
            r = rgb(os.path.join(rd, rpat % f))
            full.append(label_tile(r, "%s  %.2f dB" % (c["label"], psnr(r, g)), 338, 254))
            if b:
                zg, zr = g[b[1]:b[3], b[0]:b[2]], r[b[1]:b[3], b[0]:b[2]]
                zoom.append(label_tile(zr, "%s zoom  %.2f dB" % (c["label"], psnr(zr, zg)), 338, 254))
        rows.append(hstack(full))
        if zoom:
            rows.append(hstack(zoom))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    stack(rows).save(a.out, quality=85)
    print("scene overview written to %s" % a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene_dir", required=True, help="ARS scene folder (holds frames/ and gt/)")
    ap.add_argument("--method", choices=["ex4dgs", "multi4d", "stg"])
    ap.add_argument("--run")
    ap.add_argument("--score", help="the run's verified score JSON")
    ap.add_argument("--label", default="")
    ap.add_argument("--train_log", default=None)
    ap.add_argument("--scene_overview", default=None, help="JSON list of runs to compare (scene mode)")
    ap.add_argument("--out", required=True, help="run mode: report folder; scene mode: output .jpg")
    a = ap.parse_args()
    if a.scene_overview:
        scene_mode(a)
    else:
        if not (a.method and a.run and a.score):
            ap.error("run mode needs --method, --run and --score")
        run_mode(a)


if __name__ == "__main__":
    main()
