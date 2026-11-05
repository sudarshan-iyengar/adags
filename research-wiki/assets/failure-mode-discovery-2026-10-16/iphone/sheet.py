"""DyCheck iPhone contact sheet: per sampled held-out frame, GT | render | |error| (x3, co-visible region only; grey =
outside the co-visibility mask). Usage: sheet.py SCENE_DIR RENDER_SUBDIR OUT.png  (MoSca layout: test_images/,
test_covisible/, <render dir>/ with the same file names)."""
import sys, glob, os, numpy as np
from PIL import Image, ImageDraw
d, rsub, out = sys.argv[1:4]
fns = sorted(os.path.basename(f) for f in glob.glob(f"{d}/test_images/*.png"))
rd = glob.glob(f"{d}/{rsub}")[0]
rows = []
for fn in fns:
    if not os.path.exists(f"{rd}/{fn}"): continue
    g = np.asarray(Image.open(f"{d}/test_images/{fn}").convert("RGB")).astype(np.float32) / 255
    r = np.asarray(Image.open(f"{rd}/{fn}").convert("RGB").resize((g.shape[1], g.shape[0]))).astype(np.float32) / 255
    m = np.asarray(Image.open(f"{d}/test_covisible/{fn}").convert("L")) > 127
    e = np.clip(np.abs(g - r).mean(-1) * 3, 0, 1)
    ev = np.stack([e] * 3, -1); ev[~m] = 0.35
    mp = -10 * np.log10(((g - r) ** 2).mean(-1)[m].mean()) if m.any() else float("nan")
    tiles = [Image.fromarray((x * 255).astype(np.uint8)) for x in (g, r, ev)]
    row = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height)); x = 0
    for t in tiles: row.paste(t, (x, 0)); x += t.width
    ImageDraw.Draw(row).text((4, 4), f"{fn}  masked PSNR {mp:.1f}", fill=(255, 255, 0)); rows.append(row)
cols = 3
W, H = rows[0].width, rows[0].height
S = Image.new("RGB", (W * cols, H * ((len(rows) + cols - 1) // cols)))
for i, r in enumerate(rows): S.paste(r, ((i % cols) * W, (i // cols) * H))
S = S.resize((S.width // 2, S.height // 2), Image.LANCZOS); S.save(out); print(out, S.size, len(rows))
