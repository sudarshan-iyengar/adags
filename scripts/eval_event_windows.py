#!/usr/bin/env python
"""Common event-window evaluator (ARS fast path E4; conventions frozen in the
absence-return-suite plan rev 2 section 8.3).

One script scores ANY method's cam00 renders, on real fixtures and on ARS scenes
alike, from saved 8-bit PNGs against one ground truth per scene, over the
regions and windows of a *window package* (JSON):

    {
      "package_version": "1",
      "scene": "...", "camera": "cam00", "raster": [W, H],
      "gt": {"dir": "...", "pattern": "cam00_%04d.png"},
      "frames": [first, last],
      "regions": {
        "<name>": {"kind": "whole"}
                | {"kind": "bbox", "bbox": [x0, y0, x1, y1]}          # half-open
                | {"kind": "mask_dir", "dir": "...", "pattern": "%05d.png",
                   "threshold": 127}                                  # mask = value > threshold
      },
      "windows": {
        "<name>": {"region": "<region>", "frames": [lo, hi],          # inclusive
                   "require_nonempty_every_frame": false}
      }
    }

Relative paths resolve against the package file's directory.

Frozen conventions
------------------
* Pixels: renders and GT are read back from their saved 8-bit PNGs, as RGB,
  divided by 255 and clamped to [0, 1]; all arithmetic in float64.
* PSNR: pooled. MSE = sum over frames, region pixels and channels of the squared
  error, divided by 3 x (region pixel-times). PSNR = 10 log10(1 / MSE). Zero MSE
  is emitted as ``mse: 0, psnr: null, exact_match: true``. An empty region is
  emitted as ``"NA"`` with its (zero) count.
* SSIM: 11x11 Gaussian window, sigma 1.5, K1 0.01, K2 0.03, data range 1,
  population (not sample) covariance; computed per channel and averaged over
  RGB; the FULL-FRAME map with zero padding at frame borders (the GraphDECO
  ``ssim`` convention), averaged over the region's pixels; frames pooled by
  pixel weight.
* LPIPS: AlexNet, v0.1, from the ``lpips`` package (never a method's vendored
  copy), inputs mapped to [-1, 1]; computed per frame on the region's bounding
  box padded by 16 px and grown to at least 64x64, clipped to the frame; the
  window value is the mean over frames; frames whose region has fewer than 100
  pixels are skipped and counted. torch is imported only when LPIPS is asked for.
* ``--segment_len N`` additionally splits every window at multiples of N (STG's
  independent 50-frame specialists): values from different segments are reported
  separately and never pooled together.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
from PIL import Image

SSIM_WIN = 11
SSIM_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03
LPIPS_PAD = 16
LPIPS_MIN_SIDE = 64
LPIPS_MIN_REGION_PX = 100
CONVENTIONS = {
    "pixels": "saved 8-bit PNG, RGB, /255, clamped to [0,1], float64",
    "psnr": "pooled: sum sq err over frames x region px x 3 channels / (3 x pixel-times); zero MSE -> psnr null, exact_match true; empty region -> NA",
    "ssim": "gaussian 11x11 sigma 1.5, K1 0.01, K2 0.03, L 1, population covariance, per channel then RGB mean, full-frame map with zero padding, mean over region pixels, frames pooled by pixel weight",
    "lpips": "lpips package, net alex, version 0.1, inputs in [-1,1], per-frame region bbox padded 16 px and grown to >= 64x64, clipped; window mean over frames; frames with region < 100 px skipped",
}


# --------------------------------------------------------------------------
# pixels and regions
# --------------------------------------------------------------------------


def load_rgb(path):
    """8-bit PNG -> float64 (H, W, 3) in [0, 1]."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    return np.clip(a, 0.0, 1.0)


def region_mask(region, frame, shape, base_dir):
    """Boolean (H, W) mask of `region` at absolute `frame`; None if the mask file is absent."""
    kind = region["kind"]
    H, W = shape
    if kind == "whole":
        return np.ones((H, W), dtype=bool)
    if kind == "bbox":
        x0, y0, x1, y1 = region["bbox"]
        m = np.zeros((H, W), dtype=bool)
        m[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
        return m
    if kind == "mask_dir":
        p = os.path.join(_resolve(region["dir"], base_dir), region.get("pattern", "%05d.png") % frame)
        if not os.path.isfile(p):
            return None
        with Image.open(p) as im:
            m = np.asarray(im.convert("L"))
        if m.shape != (H, W):
            raise SystemExit(f"mask {p} has shape {m.shape}, render is {(H, W)}")
        return m > int(region.get("threshold", 127))
    raise SystemExit(f"unknown region kind {kind!r}")


def _resolve(p, base_dir):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def _gauss1d():
    x = np.arange(SSIM_WIN, dtype=np.float64) - SSIM_WIN // 2
    g = np.exp(-(x ** 2) / (2.0 * SSIM_SIGMA ** 2))
    return g / g.sum()


def _blur(img, g):
    """Separable 'same' Gaussian filter with zero padding (== conv2d padding=5)."""
    from scipy.ndimage import correlate1d
    t = correlate1d(img, g, axis=0, mode="constant", cval=0.0)
    return correlate1d(t, g, axis=1, mode="constant", cval=0.0)


def ssim_map(x, y):
    """Full-frame SSIM map, (H, W), mean over the three channels."""
    g = _gauss1d()
    C1 = (SSIM_K1 * 1.0) ** 2
    C2 = (SSIM_K2 * 1.0) ** 2
    acc = np.zeros(x.shape[:2], dtype=np.float64)
    for c in range(x.shape[2]):
        a, b = x[..., c], y[..., c]
        mu_a, mu_b = _blur(a, g), _blur(b, g)
        s_aa = _blur(a * a, g) - mu_a * mu_a
        s_bb = _blur(b * b, g) - mu_b * mu_b
        s_ab = _blur(a * b, g) - mu_a * mu_b
        acc += ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_aa + s_bb + C2))
    return acc / x.shape[2]


def lpips_box(mask):
    """Half-open [x0, y0, x1, y1] LPIPS crop for a region mask (bbox + 16 px, >= 64 px per side, clipped)."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()) - LPIPS_PAD, int(xs.max()) + 1 + LPIPS_PAD
    y0, y1 = int(ys.min()) - LPIPS_PAD, int(ys.max()) + 1 + LPIPS_PAD

    def grow(lo, hi, n):
        need = LPIPS_MIN_SIDE - (hi - lo)
        if need > 0:
            lo -= need // 2
            hi += need - need // 2
        lo, hi = max(0, lo), min(n, hi)
        # after clipping at one border, extend on the other side if room remains
        if hi - lo < LPIPS_MIN_SIDE:
            if lo == 0:
                hi = min(n, LPIPS_MIN_SIDE)
            else:
                lo = max(0, hi - LPIPS_MIN_SIDE)
        return lo, hi

    x0, x1 = grow(x0, x1, W)
    y0, y1 = grow(y0, y1, H)
    return [x0, y0, x1, y1]


class _Lpips:
    def __init__(self, device):
        import torch
        import lpips
        self.torch = torch
        self.device = device
        self.net = lpips.LPIPS(net="alex", version="0.1", verbose=False).to(device).eval()

    def __call__(self, x, y, box):
        x0, y0, x1, y1 = box
        t = self.torch
        a = t.from_numpy(np.ascontiguousarray(x[y0:y1, x0:x1].transpose(2, 0, 1))).float().unsqueeze(0).to(self.device)
        b = t.from_numpy(np.ascontiguousarray(y[y0:y1, x0:x1].transpose(2, 0, 1))).float().unsqueeze(0).to(self.device)
        with t.no_grad():
            return float(self.net(a * 2.0 - 1.0, b * 2.0 - 1.0).item())


def psnr_block(sq_sum, px):
    """Pooled PSNR record from a squared-error sum over `px` pixel-times x 3 channels."""
    if px == 0:
        return {"psnr": "NA", "pixel_count": 0}
    mse = sq_sum / (3.0 * px)
    if mse == 0.0:
        return {"mse": 0.0, "psnr": None, "exact_match": True, "pixel_count": px}
    return {"mse": mse, "psnr": 10.0 * math.log10(1.0 / mse), "exact_match": False, "pixel_count": px}


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _segments(lo, hi, seg):
    out, a = [], lo
    while a <= hi:
        b = min(hi, (a // seg + 1) * seg - 1)
        out.append((a, b))
        a = b + 1
    return out


def evaluate(package, package_dir, renders_dir, render_pattern, metrics, gt_dir=None, gt_pattern=None,
             segment_len=None, device="cpu", windows=None):
    regions = package["regions"]
    wins = package["windows"] if windows is None else {k: package["windows"][k] for k in windows}
    gdir = gt_dir or _resolve(package["gt"]["dir"], package_dir)
    gpat = gt_pattern or package["gt"]["pattern"]
    need = sorted({f for w in wins.values() for f in range(w["frames"][0], w["frames"][1] + 1)})
    lp = _Lpips(device) if "lpips" in metrics else None

    # per (frame, region) accumulators, computed once per frame
    per = {}  # (frame, region) -> dict(px, sq, ssim_sum, lpips, box)
    for f in need:
        rp = os.path.join(renders_dir, render_pattern % f)
        gp = os.path.join(gdir, gpat % f)
        if not os.path.isfile(rp):
            raise SystemExit(f"missing render for frame {f}: {rp}")
        if not os.path.isfile(gp):
            raise SystemExit(f"missing GT for frame {f}: {gp}")
        r, g = load_rgb(rp), load_rgb(gp)
        if r.shape != g.shape:
            raise SystemExit(f"shape mismatch at frame {f}: render {r.shape} vs gt {g.shape}")
        sq = ((r - g) ** 2).sum(axis=2)
        smap = ssim_map(r, g) if "ssim" in metrics else None
        for rn in {w["region"] for w in wins.values() if w["frames"][0] <= f <= w["frames"][1]}:
            m = region_mask(regions[rn], f, sq.shape, package_dir)
            px = 0 if m is None else int(m.sum())
            rec = {"px": px, "sq": float(sq[m].sum()) if px else 0.0}
            if smap is not None and px:
                rec["ssim_sum"] = float(smap[m].sum())
            if lp is not None:
                if px >= LPIPS_MIN_REGION_PX:
                    rec["box"] = lpips_box(m)
                    rec["lpips"] = lp(r, g, rec["box"])
                else:
                    rec["lpips"] = None
            per[(f, rn)] = rec

    def summarise(rn, lo, hi, require_all):
        recs = [(f, per[(f, rn)]) for f in range(lo, hi + 1)]
        px = sum(r["px"] for _, r in recs)
        empty = [f for f, r in recs if r["px"] == 0]
        out = {"region": rn, "frames": [lo, hi], "n_frames": len(recs), "empty_frames": empty,
               "pixel_times": px}
        if require_all and empty:
            out["error"] = f"empty region on frames {empty} (require_nonempty_every_frame)"
        if "psnr" in metrics:
            out["psnr"] = psnr_block(sum(r["sq"] for _, r in recs), px)
        if "ssim" in metrics:
            out["ssim"] = {"value": (sum(r.get("ssim_sum", 0.0) for _, r in recs) / px) if px else "NA",
                           "pixel_count": px}
        if "lpips" in metrics:
            vals = [r["lpips"] for _, r in recs if r.get("lpips") is not None]
            out["lpips"] = {"value": (sum(vals) / len(vals)) if vals else "NA", "n_frames_used": len(vals),
                            "n_frames_skipped_small_region": len(recs) - len(vals)}
        pf = []
        for f, r in recs:
            e = {"frame": f, "px": r["px"]}
            if "psnr" in metrics:
                b = psnr_block(r["sq"], r["px"])
                e["mse"] = b.get("mse"); e["psnr"] = b["psnr"]
            if "ssim" in metrics:
                e["ssim"] = (r["ssim_sum"] / r["px"]) if r["px"] else None
            if "lpips" in metrics:
                e["lpips"] = r.get("lpips"); e["lpips_box"] = r.get("box")
            pf.append(e)
        out["per_frame"] = pf
        return out

    res = {}
    for wn, w in wins.items():
        lo, hi = w["frames"]
        res[wn] = summarise(w["region"], lo, hi, bool(w.get("require_nonempty_every_frame", False)))
        if segment_len:
            segs = _segments(lo, hi, segment_len)
            if len(segs) > 1:
                res[wn]["segments"] = {f"{a}-{b}": {k: v for k, v in summarise(w["region"], a, b, False).items()
                                                    if k != "per_frame"} for a, b in segs}
    return res, gdir, gpat


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--package", required=True)
    ap.add_argument("--renders", required=True, help="directory of cam00 renders")
    ap.add_argument("--render_pattern", default="%05d.png", help="file name of absolute frame f, printf-style")
    ap.add_argument("--gt_dir", default=None, help="override the package GT directory (golden tests only)")
    ap.add_argument("--gt_pattern", default=None)
    ap.add_argument("--metrics", default="psnr,ssim,lpips")
    ap.add_argument("--windows", default=None, help="comma-separated subset of the package windows")
    ap.add_argument("--segment_len", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--method", default=None, help="label recorded in the output")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    metrics = [m.strip() for m in a.metrics.split(",") if m.strip()]
    bad = set(metrics) - {"psnr", "ssim", "lpips"}
    if bad:
        raise SystemExit(f"unknown metrics {sorted(bad)}")
    with open(a.package) as fh:
        pkg = json.load(fh)
    if str(pkg.get("package_version")) != "1":
        raise SystemExit(f"unsupported package_version {pkg.get('package_version')!r}")
    wsel = [w.strip() for w in a.windows.split(",")] if a.windows else None
    res, gdir, gpat = evaluate(pkg, os.path.dirname(os.path.abspath(a.package)), a.renders, a.render_pattern,
                               metrics, a.gt_dir, a.gt_pattern, a.segment_len, a.device, wsel)
    out = {
        "tool": "scripts/eval_event_windows.py",
        "tool_sha256": _sha(os.path.abspath(__file__)),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": a.method,
        "package": os.path.abspath(a.package), "package_sha256": _sha(a.package),
        "scene": pkg.get("scene"),
        "renders": os.path.abspath(a.renders), "render_pattern": a.render_pattern,
        "gt": gdir, "gt_pattern": gpat, "gt_overridden": a.gt_dir is not None,
        "metrics": metrics, "segment_len": a.segment_len,
        "conventions": CONVENTIONS,
        "windows": res,
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    errs = {k: v["error"] for k, v in res.items() if "error" in v}
    for k, v in res.items():
        p = v.get("psnr", {})
        print(f"{k:>4} [{v['frames'][0]},{v['frames'][1]}] {v['region']:>10}: psnr={p.get('psnr')} "
              f"ssim={v.get('ssim', {}).get('value')} lpips={v.get('lpips', {}).get('value')} px={v['pixel_times']}")
    if errs:
        print("ERRORS " + json.dumps(errs), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
