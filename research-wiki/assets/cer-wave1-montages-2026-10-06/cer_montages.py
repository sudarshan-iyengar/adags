"""ARS wave 1, CER: visual montages from existing renders (looking material, never a score).

(1) runs_<scene>.jpg: every Ex4DGS base run (the lrdn0066 replacements, the runs of record) and every Ex4DGS+CER run at
    the mid-gap frame, beside the ground truth; each tile labelled with that run's whole-frame and static-background PSNR
    from its verified score.
(2) methods_<scene>.jpg: ground truth, STG (seed 0), Multi4D (run 1), Ex4DGS base and Ex4DGS+CER at the last frame before
    the gap, the middle of the gap and the first return frame; full frames and a zoomed crop around the target (union of
    the scene's own ghost and target masks); each tile labelled with that run's pre-gap, gap or return-window PSNR. For Ex4DGS and
    CER the run shown is the one whose gap-window PSNR is closest to its arm's mean on that scene (named on the tile).
Usage: python cer_montages.py --out <dir>
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W = "/leonardo_work/EUHPC_D36_068/sri/proj_adags"
S = "/leonardo_scratch/large/userexternal/siyengar/ars/wave1"
SC = W + "/agent-control/ars/scores/wave1"
SCENES = ["ARS-L1-A0", "ARS-L2-A0", "ARS-L3-A0", "ARS-L1-G1", "ARS-L2-G1", "ARS-L3-G1"]
NAME = {"L1": "bottle", "L2": "carton", "L3": "mug", "A0": "same-pose", "G1": "pass-through"}
REPL = {("ARS-L1-A0", 1), ("ARS-L2-A0", 3), ("ARS-L2-R2", 1), ("ARS-L2-G1", 1), ("ARS-L3-G1", 2)}


def font(sz):
    for p in ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.isfile(p):
            return ImageFont.truetype(p, sz)
    try:
        return ImageFont.load_default(size=sz)
    except TypeError:
        return ImageFont.load_default()


F, FB = font(22), font(30)


def base_label(sc, r):
    return "ars_w1_%s_ex4dgs_rep%d%s" % (sc, r, "_repl" if (sc, r) in REPL else "")


def score(method, label):
    return json.load(open("%s/%s/%s.json" % (SC, method, label)))


def ps(o, w):
    return o["windows"][w]["psnr"]["psnr"]


def render(method, sc, label, f):
    if method == "ex4dgs":
        return "%s/runs/ex4dgs/%s/model/test/itrs_40000/renders/%06d.png" % (W, label, f)
    if method == "ex4dgs_cer":
        return "%s/runs/ex4dgs_cer/%s/model/test/itrs_40000/renders/%06d.png" % (W, label, f)
    if method == "multi4d":
        return "%s/runs/multi4d/%s/final/iteration_20000/render/%d.png" % (W, label, f)
    if method == "stg":
        return "%s/runs/stg/ars_w1_%s_seed0/stg/renders/%05d.png" % (W, sc, f)
    return "%s/%s/frames/cam00/%04d.png" % (S, sc, f)


def img(p):
    return Image.open(p).convert("RGB")


def tile(im, size, text, sub=None):
    t = im.resize(size, Image.LANCZOS)
    d = ImageDraw.Draw(t)
    d.rectangle([0, 0, size[0], 30 if sub is None else 56], fill=(0, 0, 0))
    d.text((6, 3), text, fill=(255, 255, 255), font=F)
    if sub:
        d.text((6, 29), sub, fill=(255, 220, 120), font=F)
    return t


def crop_box(sc, frames):
    m = None
    for kind, f in frames:
        p = "%s/%s/gt/masks/%s/%05d.png" % (S, sc, kind, f)
        if os.path.isfile(p):
            a = np.asarray(Image.open(p).convert("L")) > 127
            m = a if m is None else (m | a)
    ys, xs = np.nonzero(m)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * 1.6 + 20
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    return (x0, y0, int(min(1352, cx + half)), int(min(1014, cy + half)))


def representative(method, labels):
    v = [ps(score(method, lab), "P1") for lab in labels]
    m = float(np.mean(v))
    return labels[int(np.argmin([abs(x - m) for x in v]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for sc in SCENES:
        lay, ev = sc.split("-")[1], sc.split("-")[2]
        title = "%s, %s (%s)" % (NAME[lay], NAME[ev], sc)
        pkg = json.load(open("%s/%s/gt/window_package.json" % (S, sc)))
        p1, p2 = pkg["windows"]["P1"]["frames"], pkg["windows"]["P2"]["frames"]
        pre, mid, ret = p1[0] - 4, (p1[0] + p1[1]) // 2, p2[0]
        base = [base_label(sc, r) for r in (1, 2, 3)]
        cer = ["ars_w1_%s_cer_rep%d" % (sc, r) for r in (1, 2, 3)]

        # (1) every Ex4DGS base and CER run at the mid-gap frame
        sz = (608, 456)
        g = img(render("gt", sc, None, mid))
        rows = []
        for arm, labs, meth in (("Ex4DGS base", base, "ex4dgs"), ("Ex4DGS+CER", cer, "ex4dgs_cer")):
            row = [tile(g, sz, "ground truth, frame %d" % mid)]
            for lab in labs:
                o = score(meth, lab)
                row.append(tile(img(render(meth, sc, lab, mid)), sz, "%s run %s" % (arm, lab.split("rep")[1]),
                                "whole %.1f dB  static %.1f dB" % (ps(o, "H2"), ps(o, "STAT"))))
            rows.append(row)
        canvas = Image.new("RGB", (sz[0] * 4, 50 + sz[1] * 2), (255, 255, 255))
        ImageDraw.Draw(canvas).text((10, 8), "%s: every Ex4DGS run at mid-gap frame %d" % (title, mid), fill=(0, 0, 0), font=FB)
        for i, row in enumerate(rows):
            for j, t in enumerate(row):
                canvas.paste(t, (j * sz[0], 50 + i * sz[1]))
        canvas.save("%s/runs_%s.jpg" % (a.out, sc), quality=88)

        # (2) methods side by side
        rb, rc = representative("ex4dgs", base), representative("ex4dgs_cer", cer)
        cols = [("ground truth", "gt", None), ("STG (seed 0)", "stg", "ars_w1_%s_stg_seed0" % sc),
                ("Multi4D (run 1)", "multi4d", "ars_w1_%s_multi4d_rep1" % sc),
                ("Ex4DGS base (run %s)" % rb.split("rep")[1], "ex4dgs", rb),
                ("Ex4DGS+CER (run %s)" % rc.split("rep")[1], "ex4dgs_cer", rc)]
        box = crop_box(sc, [("target", pre), ("ghost", mid), ("target", ret)])
        fs, cs = (420, 315), (315, 315)
        frames = [("before the gap, frame %d" % pre, pre, "H1"), ("mid-gap, frame %d" % mid, mid, "P1"),
                  ("return, frame %d" % ret, ret, "P2")]
        rowh = 40 + fs[1] + cs[1]
        canvas = Image.new("RGB", (fs[0] * len(cols), 60 + rowh * len(frames)), (255, 255, 255))
        dc = ImageDraw.Draw(canvas)
        dc.text((10, 10), "%s: methods side by side (zoom = target region from the scene's own masks)" % title,
                fill=(0, 0, 0), font=FB)
        for i, (lab, f, win) in enumerate(frames):
            y = 60 + i * rowh
            wname = {"H1": "pre-gap (target, frames %d-%d)" % tuple(pkg["windows"]["H1"]["frames"]),
                     "P1": "gap", "P2": "return"}[win]
            dc.text((10, y + 6), lab + "   (tile numbers: %s-window PSNR of that run)" % wname, fill=(0, 0, 0), font=F)
            for j, (cname, meth, rlab) in enumerate(cols):
                im = img(render(meth, sc, rlab, f))
                sub = None
                if win and meth != "gt":
                    sub = "%.1f dB" % ps(score(meth if meth != "stg" else "stg", rlab), win)
                canvas.paste(tile(im, fs, cname, sub), (j * fs[0], y + 40))
                z = im.crop(box).resize(cs, Image.LANCZOS)
                canvas.paste(z, (j * fs[0] + (fs[0] - cs[0]) // 2, y + 40 + fs[1]))
        canvas.save("%s/methods_%s.jpg" % (a.out, sc), quality=88)
        print("wrote", sc, "representative runs:", rb, rc, "crop", box, flush=True)


if __name__ == "__main__":
    main()
