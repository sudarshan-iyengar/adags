"""fmd mono2 montage: per scene, held-out camera CAM at frame F: original | fixture | one column per method render,
plus a second row zoomed on the vacated region (mask bounding box + margin, red outline = vacated mask).
Usage: montage.py OUT.png CAM F LABEL1:prefix1 [LABEL2:prefix2 ...]   (pack = D:/adags-ars/fmd/real/<prefix>_<scene>_rep1)"""
import sys, numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

out, cam, F = sys.argv[1], sys.argv[2], int(sys.argv[3])
meths = [a.split(":") for a in sys.argv[4:]]
R = "D:/adags-ars/fmd/real"
rows = []
for sc in ["cut_roasted_beef", "flame_steak", "sear_steak"]:
    ref = f"{R}/{meths[0][1]}_{sc}_rep1"
    m = np.asarray(Image.open(f"{ref}/mask/{cam}/{F:05d}.png")) > 0
    ys, xs = np.where(m); pad = 60
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad, 1013), max(xs.min() - pad, 0), min(xs.max() + pad, 1351)
    edge = binary_dilation(m, iterations=2) & ~m
    tiles = [("orig", f"{ref}/orig/{cam}/{F:04d}.png"), ("fixture", f"{ref}/fix/{cam}/{F:04d}.png")]
    tiles += [(lab, f"{R}/{p}_{sc}_rep1/render/{cam}/{F:04d}.png") for lab, p in meths]
    full, zoom = [], []
    for lab, p in tiles:
        a = np.asarray(Image.open(p).convert("RGB")).copy()
        z = a[y0:y1, x0:x1].copy(); z[edge[y0:y1, x0:x1]] = [255, 0, 0]
        f = Image.fromarray(a).resize((676, 507), Image.LANCZOS)
        ImageDraw.Draw(f).text((6, 4), f"{sc} {cam} f{F} {lab}", fill=(255, 255, 0))
        zh = 507; zw = int(z.shape[1] * zh / z.shape[0])
        zi = Image.fromarray(z).resize((min(zw, 676), zh), Image.LANCZOS)
        zz = Image.new("RGB", (676, 507)); zz.paste(zi, ((676 - zi.width) // 2, 0))
        ImageDraw.Draw(zz).text((6, 4), f"zoom {lab}", fill=(255, 255, 0))
        full.append(f); zoom.append(zz)
    for row in (full, zoom):
        r = Image.new("RGB", (676 * len(row), 507))
        for i, t in enumerate(row): r.paste(t, (676 * i, 0))
        rows.append(r)
M = Image.new("RGB", (rows[0].width, 507 * len(rows)))
for i, r in enumerate(rows): M.paste(r, (0, 507 * i))
M.save(out); print(out, M.size)
