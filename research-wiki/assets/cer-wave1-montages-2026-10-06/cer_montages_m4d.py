"""ARS wave 1, CER on Multi4D: visual montages (looking material, never a score). Same-pose scenes only.

(1) m4d_methods_<scene>.jpg: ground truth, STG (seed 0), Multi4D, Multi4D+CER, Ex4DGS, Ex4DGS+CER at the last pre-gap
    target frame, three gap frames and the first return frame; full frame and a zoom on the target region (union of the
    scene's own target and ghost masks). Each tile: that run's window PSNR / SSIM / LPIPS for the window the frame belongs
    to (pre-gap target H1, gap P1 = ghost region, return P2 = target region) and the frame's own PSNR in that window.
    Multi-run arms show the run whose gap-window PSNR is closest to the arm's mean (named on the tile).
(2) m4d_runs_<scene>.jpg: every Multi4D base run and every Multi4D+CER run at the gap frames, zoomed, beside an error map
    (mean absolute RGB error against the ground truth, fixed scale 0 to 0.25 for every tile), with the frame's ghost-
    region PSNR from the score.
(3) m4d_banks_<scene>.jpg: every Multi4D base and CER run at one pre-gap and one gap frame: the full render, the persistent
    banks alone and the transient bank alone (cer_m4d_banks.py output, black where a bank paints nothing), with the
    run's cer_begin / cer_gates shares and the cam00 bank-alone alpha inside the ghost mask.
Usage: python cer_montages_m4d.py --banks <cer_m4d_banks out dir> --out <dir>
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cer_montages as cm  # noqa: E402

W, S = cm.W, cm.S
SCENES = ["ARS-L1-A0", "ARS-L2-A0", "ARS-L3-A0"]
F, FB = cm.F, cm.FB
FS = cm.font(18)


def render(method, sc, label, f):
    if method == "multi4d_cer":
        return "%s/runs/multi4d_cer/%s/final/iteration_20000/render/%d.png" % (W, label, f)
    return cm.render(method, sc, label, f)


def win_of(pkg, f):
    for w in ("P1", "P2", "H1"):
        a, b = pkg["windows"][w]["frames"]
        if a <= f <= b:
            return w
    return None


def frame_psnr(o, w, f):
    for r in o["windows"][w].get("per_frame", []):
        if r["frame"] == f:
            return r["psnr"]
    return None


def wstats(o, w):
    x = o["windows"][w]
    return x["psnr"]["psnr"], x["ssim"]["value"], x["lpips"]["value"]


def errmap(r, g):
    """Mean absolute RGB error, 0..0.25 mapped through a black-red-yellow-white ramp."""
    e = np.abs(np.asarray(r, float) - np.asarray(g, float)).mean(2) / 255.0
    t = np.clip(e / 0.25, 0, 1)
    rgb = np.stack([np.clip(3 * t, 0, 1), np.clip(3 * t - 1, 0, 1), np.clip(3 * t - 2, 0, 1)], 2)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def text_tile(size, lines, colour=(0, 0, 0)):
    t = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(t)
    for i, s in enumerate(lines):
        d.text((8, 6 + i * 24), s, fill=colour, font=FS)
    return t


def methods(sc, pkg, frames, box, out):
    lay = cm.NAME[sc.split("-")[1]]
    base = ["ars_w1_%s_multi4d_rep%d" % (sc, r) for r in (1, 2, 3)]
    m4c = ["ars_w1_%s_m4dcer_rep%d" % (sc, r) for r in (1, 2, 3)]
    exb = [cm.base_label(sc, r) for r in (1, 2, 3)]
    exc = ["ars_w1_%s_cer_rep%d" % (sc, r) for r in (1, 2, 3)]
    rm, rmc = cm.representative("multi4d", base), cm.representative("multi4d_cer", m4c)
    rb, rc = cm.representative("ex4dgs", exb), cm.representative("ex4dgs_cer", exc)
    run = lambda lab: lab.split("rep")[1]
    cols = [("ground truth", "gt", None), ("STG (seed 0)", "stg", "ars_w1_%s_stg_seed0" % sc),
            ("Multi4D (run %s)" % run(rm), "multi4d", rm), ("Multi4D+CER (run %s)" % run(rmc), "multi4d_cer", rmc),
            ("Ex4DGS (run %s)" % run(rb), "ex4dgs", rb), ("Ex4DGS+CER (run %s)" % run(rc), "ex4dgs_cer", rc)]
    fs, cs = (400, 300), (300, 300)
    rowh = 40 + fs[1] + cs[1] + 50
    canvas = Image.new("RGB", (fs[0] * len(cols), 70 + rowh * len(frames)), (255, 255, 255))
    dc = ImageDraw.Draw(canvas)
    dc.text((10, 8), "%s same-pose (%s): methods side by side. Zoom = target region from the scene's own masks." % (lay, sc),
            fill=(0, 0, 0), font=FB)
    dc.text((10, 42), "Tile numbers: that run's window PSNR / SSIM / LPIPS (window named per row) and this frame's PSNR "
            "in the window region. Multi-run arms: the run closest to its arm's mean gap PSNR.", fill=(0, 0, 0), font=FS)
    wname = {"H1": "pre-gap window (target region, frames %d-%d)" % tuple(pkg["windows"]["H1"]["frames"]),
             "P1": "gap window (ghost region, frames %d-%d)" % tuple(pkg["windows"]["P1"]["frames"]),
             "P2": "return window (target region, frames %d-%d)" % tuple(pkg["windows"]["P2"]["frames"])}
    for i, f in enumerate(frames):
        w = win_of(pkg, f)
        y = 70 + i * rowh
        dc.text((10, y + 8), "frame %d: %s" % (f, wname[w]), fill=(0, 0, 0), font=F)
        for j, (cname, meth, lab) in enumerate(cols):
            im = cm.img(render(meth, sc, lab, f))
            canvas.paste(cm.tile(im, fs, cname), (j * fs[0], y + 40))
            canvas.paste(im.crop(box).resize(cs, Image.LANCZOS), (j * fs[0] + (fs[0] - cs[0]) // 2, y + 40 + fs[1]))
            if meth != "gt":
                o = cm.score(meth, lab)
                p, s, l = wstats(o, w)
                fp = frame_psnr(o, w, f)
                dc.text((j * fs[0] + 8, y + 40 + fs[1] + cs[1] + 2), "%.2f dB / %.3f / %.3f" % (p, s, l), fill=(0, 0, 0), font=FS)
                dc.text((j * fs[0] + 8, y + 40 + fs[1] + cs[1] + 24), "frame %d: %s dB" % (f, "%.2f" % fp if fp else "n/a"),
                        fill=(120, 60, 0), font=FS)
    canvas.save("%s/m4d_methods_%s.jpg" % (out, sc), quality=88)
    return {"multi4d": rm, "multi4d_cer": rmc, "ex4dgs": rb, "ex4dgs_cer": rc}


def runs(sc, pkg, gap, box, out):
    lay = cm.NAME[sc.split("-")[1]]
    rows = [("Multi4D run %d" % r, "multi4d", "ars_w1_%s_multi4d_rep%d" % (sc, r)) for r in (1, 2, 3)] + \
           [("Multi4D+CER run %d" % r, "multi4d_cer", "ars_w1_%s_m4dcer_rep%d" % (sc, r)) for r in (1, 2, 3)]
    cs, lw = (280, 280), 230
    canvas = Image.new("RGB", (lw + cs[0] * 2 * len(gap), 110 + (cs[1] + 30) * (len(rows) + 1)), (255, 255, 255))
    dc = ImageDraw.Draw(canvas)
    dc.text((10, 8), "%s same-pose (%s): every Multi4D run, gap frames, zoom on the target region" % (lay, sc),
            fill=(0, 0, 0), font=FB)
    dc.text((10, 44), "Per frame: render | error map (mean absolute RGB error vs ground truth, black 0 to white >= 0.25). "
            "Numbers: this frame's ghost-region PSNR; gap-window PSNR at left.", fill=(0, 0, 0), font=FS)
    gts = {f: cm.img(render("gt", sc, None, f)) for f in gap}
    y = 80
    for k, f in enumerate(gap):
        x = lw + k * 2 * cs[0]
        dc.text((x + 8, y), "frame %d" % f, fill=(0, 0, 0), font=F)
        canvas.paste(gts[f].crop(box).resize(cs, Image.LANCZOS), (x, y + 30))
    dc.text((8, y + 30 + cs[1] // 2), "ground truth", fill=(0, 0, 0), font=F)
    for i, (name, meth, lab) in enumerate(rows):
        yy = y + 30 + (i + 1) * (cs[1] + 30)
        o = cm.score(meth, lab)
        dc.text((8, yy + cs[1] // 2 - 30), name, fill=(0, 0, 0), font=F)
        dc.text((8, yy + cs[1] // 2), "gap %.2f dB" % cm.ps(o, "P1"), fill=(120, 60, 0), font=F)
        for k, f in enumerate(gap):
            x = lw + k * 2 * cs[0]
            r = cm.img(render(meth, sc, lab, f))
            canvas.paste(r.crop(box).resize(cs, Image.LANCZOS), (x, yy))
            canvas.paste(errmap(r.crop(box), gts[f].crop(box)).resize(cs, Image.NEAREST), (x + cs[0], yy))
            fp = frame_psnr(o, "P1", f)
            dc.rectangle([x, yy, x + 120, yy + 26], fill=(0, 0, 0))
            dc.text((x + 4, yy + 2), "%.2f dB" % fp, fill=(255, 255, 255), font=FS)
    canvas.save("%s/m4d_runs_%s.jpg" % (out, sc), quality=88)


def banks(sc, frames, box, bdir, out):
    lay = cm.NAME[sc.split("-")[1]]
    rows = [("Multi4D run %d" % r, "ars_w1_%s_multi4d_rep%d" % (sc, r)) for r in (1, 2, 3)] + \
           [("Multi4D+CER run %d" % r, "ars_w1_%s_m4dcer_rep%d" % (sc, r)) for r in (1, 2, 3)]
    cs, lw = (250, 250), 420
    kinds = [("full", "all banks"), ("persistent", "persistent alone"), ("transient", "transient alone")]
    canvas = Image.new("RGB", (lw + cs[0] * 3 * len(frames), 120 + (cs[1] + 10) * len(rows)), (255, 255, 255))
    dc = ImageDraw.Draw(canvas)
    dc.text((10, 8), "%s same-pose (%s): what each Multi4D bank paints on cam00 (zoom on the target region)" % (lay, sc),
            fill=(0, 0, 0), font=FB)
    dc.text((10, 44), "Re-rendered from the saved model, black background; black = that bank paints nothing there. "
            "Left: shares from cer_begin.json / cer_gates.json (training views) and cam00 alpha in the ghost mask.",
            fill=(0, 0, 0), font=FS)
    for k, f in enumerate(frames):
        for j, (kd, kn) in enumerate(kinds):
            dc.text((lw + (k * 3 + j) * cs[0] + 6, 84), "f%d %s" % (f, kn), fill=(0, 0, 0), font=FS)
    for i, (name, lab) in enumerate(rows):
        yy = 112 + i * (cs[1] + 10)
        rec = json.load(open("%s/%s/banks.json" % (bdir, lab)))
        lines = [name + ("" if rec["faithful_to_saved_render"] else "  (RE-RENDER NOT FAITHFUL)")]
        gf = str(frames[-1])
        al = rec["frames"][gf].get("alpha_in_ghost")
        if al:
            lines += ["cam00 f%s, ghost-mask alpha of each bank alone:" % gf,
                      "  persistent %.2f, transient %.2f" % (al["persistent"], al["transient"])]
        if rec["cer"]:
            mdir = "%s/runs/multi4d_cer/%s/model" % (W, lab)
            b, g = json.load(open(mdir + "/cer_begin.json")), json.load(open(mdir + "/cer_gates.json"))
            sb = b["gap_region_share_by_bank"]
            lines += ["memory rows bg %d, fg %d; residual %d" % (g["memory_rows"]["bg"], g["memory_rows"]["fg"], b["residual_rows"]),
                      "at CER start, gap region (memory off):",
                      "  fg %.2f  bg %.2f  transient %.2f" % (sb["fg"], sb["bg"], sb["tr"]),
                      "at the end: transient %.2f, residual %.4f" % (g["transient_share_gap_region"], g["residual_share_gap"]),
                      "  ordinary persistent %.2f" % (1 - g["transient_share_gap_region"] - g["residual_share_gap"]),
                      "memory share pre-gap %.2f, return %.2f" % (g["memory_share_pregap"], g["memory_share_return"])]
        canvas.paste(text_tile((lw - 10, cs[1]), lines), (0, yy))
        for k, f in enumerate(frames):
            for j, (kd, kn) in enumerate(kinds):
                im = cm.img("%s/%s/%s_%03d.png" % (bdir, lab, kd, f))
                canvas.paste(im.crop(box).resize(cs, Image.LANCZOS), (lw + (k * 3 + j) * cs[0], yy))
    canvas.save("%s/m4d_banks_%s.jpg" % (out, sc), quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--banks", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rec = {}
    for sc in SCENES:
        pkg = json.load(open("%s/%s/gt/window_package.json" % (S, sc)))
        p1, p2 = pkg["windows"]["P1"]["frames"], pkg["windows"]["P2"]["frames"]
        h1 = pkg["windows"]["H1"]["frames"]
        mid = (p1[0] + p1[1]) // 2
        gap = [p1[0], mid, p1[1]]
        frames = [h1[1]] + gap + [p2[0]]
        box = cm.crop_box(sc, [("target", h1[1]), ("ghost", mid), ("target", p2[0])])
        rec[sc] = {"frames": frames, "crop": box, "representatives": methods(sc, pkg, frames, box, a.out)}
        runs(sc, pkg, gap, box, a.out)
        if a.banks:
            banks(sc, [h1[1], mid], box, a.banks, a.out)
        print("wrote", sc, rec[sc], flush=True)
    json.dump(rec, open("%s/m4d_montages.json" % a.out, "w"), indent=1, default=int)


if __name__ == "__main__":
    main()
