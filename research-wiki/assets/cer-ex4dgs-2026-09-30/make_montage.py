"""Montage for the principal: CER smoke (T7, f71c1b0, 17,200 it) vs Ex4DGS base run 2 (40,000 it) vs ground truth,
bottle same-pose scene, cam00, frames 59 (last before the gap), 74 (mid-gap), 90 (first return). Full frames with the
gap region outlined, then zooms of the gap region. Descriptive only."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

S = "D:/adags-ars/wave1/ARS-L1-A0"
FR = [59, 74, 90]
ghost = np.zeros((1014, 1352), bool)
for f in range(63, 88):
    ghost |= np.asarray(Image.open("%s/gt/masks/ghost/%05d.png" % (S, f)).convert("L")) > 127
ys, xs = np.nonzero(ghost)
pad = 40
box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, 1351), min(ys.max() + pad, 1013))
zw, zh = box[2] - box[0], box[3] - box[1]
scale = 420 / zh
Z = (int(zw * scale), 420)
FW = int(1352 * 420 / 1014)
edge = ghost & ~np.pad(ghost, 1)[2:, 1:-1] | ghost & ~np.pad(ghost, 1)[:-2, 1:-1] | ghost & ~np.pad(ghost, 1)[1:-1, 2:] | ghost & ~np.pad(ghost, 1)[1:-1, :-2]

def load(kind, f):
    if kind == "gt":
        return Image.open("%s/frames/cam00/%04d.png" % (S, f)).convert("RGB")
    return Image.open("src/%s_%06d.png" % (kind, f)).convert("RGB")

def outlined(im):
    a = np.asarray(im).copy()
    a[edge] = (255, 0, 255)
    return Image.fromarray(a)

cols = [("ground truth", "gt"), ("Ex4DGS base, run 2 (40,000 it)", "base"), ("CER smoke T7, f71c1b0 (17,200 it)", "cer")]
W = 3 * FW + 3 * Z[0] + 7 * 8
H = 60 + 3 * (420 + 34)
out = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(out)
try:
    font = ImageFont.truetype("arial.ttf", 18)
    small = ImageFont.truetype("arial.ttf", 15)
except OSError:
    font = small = ImageFont.load_default()
d.text((8, 8), "Bottle same-pose scene, cam00. Magenta: gap region (union of the ghost masks, frames 63-87). "
       "Left: full frames; right: gap-region zoom. Gap = frames 60-89. Schedules differ (smoke 17,200 vs base 40,000).",
       fill=(0, 0, 0), font=small)
for r, f in enumerate(FR):
    y = 60 + r * (420 + 34)
    label = {59: "frame 59 (last before the gap)", 74: "frame 74 (mid-gap)", 90: "frame 90 (first return)"}[f]
    x = 8
    for c, (name, kind) in enumerate(cols):
        im = load(kind, f)
        out.paste(outlined(im).resize((FW, 420), Image.BILINEAR), (x, y + 24))
        d.text((x, y + 2), "%s - %s" % (name, label), fill=(0, 0, 0), font=small)
        x += FW + 8
    for c, (name, kind) in enumerate(cols):
        im = load(kind, f).crop(box).resize(Z, Image.NEAREST)
        out.paste(im, (x, y + 24))
        d.text((x, y + 2), "zoom: %s" % name.split(",")[0], fill=(0, 0, 0), font=small)
        x += Z[0] + 8
out.save("cer_t7_smoke_vs_base_montage.png")
print(out.size, box)
