"""Is the N3V window error a capture artefact? Ground truth and camera geometry only, no training.

For one scene, held-out cam00 at 1352x1014 (the Lanczos-halved ground truth that SA4D and Multi4D use), and the window
regions (rectangles, x0 x1 y0 y1):
  1. MOTION. Each cam00 frame vs that camera's own temporal median (300 frames): region PSNR. If the median already
     matches the frames closely, the window content barely moves.
  2. VIEW CONSISTENCY. For sampled frames, the K training cameras nearest to cam00 (by camera centre) are warped into
     cam00 using the known poses (LLFF poses_bounds; OpenCV axes = [col1, col0, -col2]):
       - rotation-only (plane at infinity);
       - a fronto-parallel plane in cam00 swept over depth; the best depth is kept per region and frame.
     Region PSNR raw and after a per-channel affine colour fit on the region (cameras differ in colour response).
     If a warped training view matches the held-out ground truth, the window content is recoverable from training
     views; if not, it is view-dependent (reflection/refraction) or inconsistent across cameras.
  3. SHARPNESS. Variance of the Laplacian (grey) in the region: cam00 ground truth vs each warped neighbour (Lanczos
     interpolation) and vs each neighbour's own unwarped frame in the same rectangle.
For comparison, the same region PSNRs (raw and colour-fitted) of the SA4D and Ex4DGS renders.
"""
import argparse, glob, json, os
import numpy as np
import cv2
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--src", required=True, help="N3V scene dir (camXX.mp4, poses_bounds.npy)")
ap.add_argument("--gt", required=True, help="cam00 GT pattern with {f}")
ap.add_argument("--method", action="append", default=[], help="LABEL=PATTERN")
ap.add_argument("--regions", required=True, help="json: {name: [x0,x1,y0,y1]}")
ap.add_argument("--frames", default="25,75,140,210,275")
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--out", required=True)
a = ap.parse_args()
REG = json.loads(a.regions)
FR = [int(x) for x in a.frames.split(",")]
W, H = 1352, 1014


def load(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float64) / 255


def psnr(x, y):
    return float(-10 * np.log10(max(((x - y) ** 2).mean(), 1e-12)))


def cfit(x, y):
    """affine per-channel fit of x to y (both N x 3)"""
    o = np.empty_like(x)
    for c in range(3):
        A = np.stack([x[:, c], np.ones(len(x))], 1)
        k = np.linalg.lstsq(A, y[:, c], rcond=None)[0]
        o[:, c] = A @ k
    return np.clip(o, 0, 1)


def lapvar(img):
    g = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# cameras
names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.src, "cam*.mp4")))
pb = np.load(os.path.join(a.src, "poses_bounds.npy"))
assert pb.shape[0] == len(names), (pb.shape, names)
cams = {}
for i, n in enumerate(names):
    M = pb[i, :15].reshape(3, 5)
    hn, wn, fn = M[:, 4]
    R = np.stack([M[:, 1], M[:, 0], -M[:, 2]], 1)
    cams[n] = {"R": R, "C": M[:, 3].copy(), "f": fn * W / wn, "near": pb[i, 15], "far": pb[i, 16]}
c0 = cams["cam00"]
nb = sorted([n for n in names if n != "cam00"], key=lambda n: np.linalg.norm(cams[n]["C"] - c0["C"]))[: a.k]
K = lambda f: np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
near = float(min(c["near"] for c in cams.values())); far = float(max(c["far"] for c in cams.values()))
depths = list(np.geomspace(near, far * 20, 48)) + [np.inf]
res = {"scene": a.scene, "neighbours": {n: round(float(np.linalg.norm(cams[n]["C"] - c0["C"])), 4) for n in nb},
       "near_far": [near, far], "regions": REG}

# 1. motion vs temporal median (all 300 GT frames, region crops only)
crops = {r: [] for r in REG}
for t in range(300):
    g = load(a.gt.format(f=t))
    for r, (x0, x1, y0, y1) in REG.items():
        crops[r].append(g[y0:y1, x0:x1].astype(np.float32))
res["motion"] = {}
for r in REG:
    S = np.stack(crops[r]); med = np.median(S, 0)
    ps = [psnr(S[t], med) for t in range(300)]
    res["motion"][r] = {"frame_vs_own_median_psnr_mean": round(float(np.mean(ps)), 2), "min": round(float(np.min(ps)), 2),
                        "temporal_std_mean": round(float(S.std(0).mean()), 4)}
    for lab, pat in (m.split("=", 1) for m in a.method):
        x0, x1, y0, y1 = REG[r]
        pm = [psnr(load(pat.format(f=t))[y0:y1, x0:x1], S[t]) for t in range(0, 300, 10)]
        res["motion"][r][f"{lab}_region_psnr"] = round(float(np.mean(pm)), 2)
    res["motion"][r]["median_as_render_psnr"] = round(float(np.mean([psnr(med, S[t]) for t in range(0, 300, 10)])), 2)
del crops


# 2/3. warps from neighbours at sampled frames
def read_frames(cam, want):
    cap = cv2.VideoCapture(os.path.join(a.src, cam + ".mp4")); out = {}
    for t in range(max(want) + 1):
        ok, fr = cap.read(); assert ok, (cam, t)
        if t in want:
            im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((W, H), Image.LANCZOS)
            out[t] = np.asarray(im).astype(np.float64) / 255
    return out


u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
ray0 = np.linalg.inv(K(c0["f"])) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])   # cam00 rays, z = 1


def warp(img, n, Z):
    c = cams[n]
    if np.isinf(Z):
        Xn = c["R"].T @ (c0["R"] @ ray0)
    else:
        Xw = c0["R"] @ (ray0 * Z) + c0["C"][:, None]
        Xn = c["R"].T @ (Xw - c["C"][:, None])
    p = K(c["f"]) @ Xn
    mx = (p[0] / p[2]).reshape(H, W).astype(np.float32); my = (p[1] / p[2]).reshape(H, W).astype(np.float32)
    return cv2.remap(img.astype(np.float32), mx, my, cv2.INTER_LANCZOS4, borderValue=(0, 0, 0)).astype(np.float64)


res["warp"] = {r: [] for r in REG}
nbf = {n: read_frames(n, set(FR)) for n in nb}
for t in FR:
    g = load(a.gt.format(f=t))
    meth = {lab: load(p.format(f=t)) for lab, p in (m.split("=", 1) for m in a.method)}
    for r, (x0, x1, y0, y1) in REG.items():
        G = g[y0:y1, x0:x1]
        row = {"frame": t, "gt_lapvar": round(lapvar(G), 1)}
        for lab, im in meth.items():
            R_ = im[y0:y1, x0:x1]
            row[lab] = [round(psnr(R_, G), 2), round(psnr(cfit(R_.reshape(-1, 3), G.reshape(-1, 3)).reshape(G.shape), G), 2), round(lapvar(R_), 1)]
        for n in nb:
            best = None
            for Z in depths:
                Wp = warp(nbf[n][t], n, Z)[y0:y1, x0:x1]
                if (Wp.sum(-1) == 0).mean() > 0.02:
                    continue
                pr = psnr(Wp, G)
                if best is None or pr > best[1]:
                    best = (Z, pr, Wp)
            Wi = warp(nbf[n][t], n, np.inf)[y0:y1, x0:x1]
            if best is None:
                row[n] = "out of view"; continue
            Z, pr, Wp = best
            row[n] = {"rot_only_psnr": round(psnr(Wi, G), 2), "plane_best_psnr": round(pr, 2),
                      "plane_best_depth": (None if np.isinf(Z) else round(float(Z), 2)),
                      "plane_best_psnr_colourfit": round(psnr(cfit(Wp.reshape(-1, 3), G.reshape(-1, 3)).reshape(G.shape), G), 2),
                      "warped_lapvar": round(lapvar(Wp), 1), "own_unwarped_lapvar": round(lapvar(nbf[n][t][y0:y1, x0:x1]), 1)}
            if t == FR[len(FR) // 2]:
                Image.fromarray((np.concatenate([G, Wp, np.abs(G - Wp) * 3], 1).clip(0, 1) * 255).astype(np.uint8)).save(
                    os.path.join(os.path.dirname(a.out) or ".", f"{a.scene}_{r}_{n}_f{t}.png"))
        res["warp"][r].append(row)
        print(json.dumps(row), flush=True)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("neighbours", "near_far", "motion")}, indent=1))
