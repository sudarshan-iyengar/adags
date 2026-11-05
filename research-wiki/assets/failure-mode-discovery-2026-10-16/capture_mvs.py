"""Window capture-artefact check 2b: is the held-out view predictable from TRAINING views alone?

Plane-sweep stereo in cam00's frame, using only training cameras: for each depth Z (fronto-parallel planes in cam00),
warp the N nearest training cameras that see the region into cam00 (known LLFF poses), after a per-camera colour
normalisation to the first neighbour (per-channel mean/std matching over the whole-frame overlap at a far depth). Cost = variance across the
warped views, box-filtered over a 9x9 window. Each pixel takes the depth of minimum cost; the prediction is the mean of
the warped views at that depth. cam00's pixels are NEVER used to choose depth. Then region PSNR vs cam00 ground truth,
raw and after a per-channel affine colour fit on the region (the same fit is applied to the method renders).
If this simple stereo prediction beats the 4D methods in the window, the content is recoverable from the training
views and the methods are at fault; if it is as bad, the windows are view-inconsistent (glass, reflections) or a
capture problem.
"""
import argparse, glob, json, os
import numpy as np
import cv2
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True); ap.add_argument("--src", required=True); ap.add_argument("--gt", required=True)
ap.add_argument("--method", action="append", default=[]); ap.add_argument("--regions", required=True)
ap.add_argument("--frames", default="75,210"); ap.add_argument("--n", type=int, default=4); ap.add_argument("--out", required=True)
a = ap.parse_args()
REG = json.loads(a.regions); FR = [int(x) for x in a.frames.split(",")]
W, H = 1352, 1014


def load(p): return np.asarray(Image.open(p).convert("RGB")).astype(np.float64) / 255
def psnr(x, y): return float(-10 * np.log10(max(((x - y) ** 2).mean(), 1e-12)))


def cfit(x, y):
    o = np.empty_like(x)
    for c in range(3):
        A = np.stack([x[..., c].ravel(), np.ones(x[..., c].size)], 1)
        o[..., c] = (A @ np.linalg.lstsq(A, y[..., c].ravel(), rcond=None)[0]).reshape(x.shape[:-1])
    return np.clip(o, 0, 1)


names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.src, "cam*.mp4")))
pb = np.load(os.path.join(a.src, "poses_bounds.npy"))
cams = {}
for i, nm in enumerate(names):
    M = pb[i, :15].reshape(3, 5); hn, wn, fn = M[:, 4]
    cams[nm] = {"R": np.stack([M[:, 1], M[:, 0], -M[:, 2]], 1), "C": M[:, 3].copy(), "f": fn * W / wn}
c0 = cams["cam00"]
K = lambda f: np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
near, far = float(pb[:, 15].min()), float(pb[:, 16].max())
depths = np.geomspace(near, far * 20, 64)
u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
ray0 = np.linalg.inv(K(c0["f"])) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])


def maps(nm, Z):
    c = cams[nm]
    Xw = c0["R"] @ (ray0 * Z) + c0["C"][:, None]
    p = K(c["f"]) @ (c["R"].T @ (Xw - c["C"][:, None]))
    return (p[0] / p[2]).reshape(H, W).astype(np.float32), (p[1] / p[2]).reshape(H, W).astype(np.float32)


def read(nm, want):
    cap = cv2.VideoCapture(os.path.join(a.src, nm + ".mp4")); out = {}
    for t in range(max(want) + 1):
        ok, fr = cap.read(); assert ok
        if t in want:
            out[t] = np.asarray(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((W, H), Image.LANCZOS)).astype(np.float32) / 255
    return out


res = {"scene": a.scene, "rows": []}
for r, (x0, x1, y0, y1) in REG.items():
    # neighbours that see the whole region at a far depth, nearest first
    cand = sorted([nm for nm in names if nm != "cam00"], key=lambda nm: np.linalg.norm(cams[nm]["C"] - c0["C"]))
    nb = []
    for nm in cand:
        mx, my = maps(nm, far)
        sx, sy = mx[y0:y1, x0:x1], my[y0:y1, x0:x1]
        if (sx >= 0).all() and (sx < W - 1).all() and (sy >= 0).all() and (sy < H - 1).all():
            nb.append(nm)
        if len(nb) == a.n: break
    fr = {nm: read(nm, set(FR)) for nm in nb}
    for t in FR:
        g = load(a.gt.format(f=t)); G = g[y0:y1, x0:x1]
        # colour-normalise every neighbour to the first at a far depth (whole-frame overlap)
        ref = None; norm = {}
        for nm in nb:
            mx, my = maps(nm, far)
            wv = cv2.remap(fr[nm][t], mx, my, cv2.INTER_LINEAR, borderValue=(-1, -1, -1))
            if ref is None: ref = wv; norm[nm] = None; continue
            ok = (wv[..., 0] >= 0) & (ref[..., 0] >= 0)
            ks = []
            for c in range(3):   # moment matching (mean, std) over the overlap: robust to misalignment, unlike regression
                a_, b_ = wv[..., c][ok], ref[..., c][ok]
                g_ = b_.std() / max(a_.std(), 1e-6); ks.append((g_, b_.mean() - g_ * a_.mean()))
            norm[nm] = ks
        cost = np.full((len(depths), y1 - y0, x1 - x0), np.inf, np.float32); stack = []
        for zi, Z in enumerate(depths):
            ws = []
            for nm in nb:
                mx, my = maps(nm, Z)
                wv = cv2.remap(fr[nm][t], mx[y0:y1, x0:x1], my[y0:y1, x0:x1], cv2.INTER_LINEAR)
                if norm[nm] is not None:
                    wv = np.stack([wv[..., c] * norm[nm][c][0] + norm[nm][c][1] for c in range(3)], -1)
                ws.append(wv)
            ws = np.stack(ws)
            var = ws.var(0).sum(-1)
            cost[zi] = cv2.boxFilter(var, -1, (9, 9))
            stack.append(ws.mean(0))
        best = cost.argmin(0)
        pred = np.take_along_axis(np.stack(stack), best[None, ..., None].repeat(3, -1), 0)[0].astype(np.float64)
        row = {"region": r, "frame": t, "neighbours": nb, "stereo_pred_psnr": round(psnr(pred, G), 2),
               "stereo_pred_psnr_colourfit": round(psnr(cfit(pred, G), G), 2),
               "median_depth": round(float(np.median(depths[best])), 2)}
        for lab, pat in (m.split("=", 1) for m in a.method):
            R_ = load(pat.format(f=t))[y0:y1, x0:x1]
            row[lab] = [round(psnr(R_, G), 2), round(psnr(cfit(R_, G), G), 2)]
        res["rows"].append(row); print(json.dumps(row), flush=True)
        Image.fromarray((np.concatenate([G, pred, np.abs(G - pred) * 3], 1).clip(0, 1) * 255).astype(np.uint8)).save(
            os.path.join(os.path.dirname(a.out), f"{a.scene}_{r}_stereo_f{t}.png"))
json.dump(res, open(a.out, "w"), indent=1)
