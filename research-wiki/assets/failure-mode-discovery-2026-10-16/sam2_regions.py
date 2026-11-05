"""Oracle-study region masks (plan page section 30): SAM2.1 (large) video propagation on held-out cam00.

Frames: the native 2704x2028 cam00 frames halved with Lanczos (the scoring ground truth), written as JPEG for SAM2's
video predictor. Objects are seeded by positive clicks at one or more frames (absolute 1352x1014 pixel coordinates),
propagated forward from the earliest seeded frame and backward, and saved as one binary PNG per object per frame:
OUT/<object>/<frame:04d>.png. Overlays at a few frames (OUT/check_<frame>.png) are for visual verification:
a click can select part of an object instead of the whole one.
Usage: sam2_regions.py --native PAT --clicks JSON --out DIR    JSON = {"object": {"30": [[x, y], ...], ...}, ...}
"""
import argparse, json, os
import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2_video_predictor

ap = argparse.ArgumentParser()
ap.add_argument("--native", required=True); ap.add_argument("--clicks", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", required=True); ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()
clicks = json.loads(open(a.clicks).read()) if os.path.exists(a.clicks) else json.loads(a.clicks)
jd = os.path.join(a.out, "jpg"); os.makedirs(jd, exist_ok=True)
for t in range(a.n):
    p = os.path.join(jd, f"{t:05d}.jpg")
    if not os.path.exists(p):
        Image.open(a.native.format(f=t)).convert("RGB").resize((1352, 1014), Image.LANCZOS).save(p, quality=95)
pred = build_sam2_video_predictor(a.cfg, a.ckpt, device="cuda")
names = list(clicks)
masks = {o: np.zeros((a.n, 1014, 1352), bool) for o in names}
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    st = pred.init_state(video_path=jd)
    for oid, o in enumerate(names, 1):
        for f, pts in clicks[o].items():
            pred.add_new_points_or_box(st, frame_idx=int(f), obj_id=oid, points=np.array(pts, np.float32),
                                       labels=np.ones(len(pts), np.int32))
    for rev in (False, True):
        for t, ids, logits in pred.propagate_in_video(st, reverse=rev):
            for k, oid in enumerate(ids):
                masks[names[oid - 1]][t] |= (logits[k, 0] > 0).cpu().numpy()
for o in names:
    d = os.path.join(a.out, o); os.makedirs(d, exist_ok=True)
    for t in range(a.n):
        Image.fromarray((masks[o][t] * 255).astype(np.uint8)).save(os.path.join(d, f"{t:04d}.png"))
    print(o, "mean area px", int(masks[o].sum((1, 2)).mean()), "frames with mask", int((masks[o].sum((1, 2)) > 0).sum()))
cols = [(255, 40, 40), (40, 255, 40), (60, 120, 255), (255, 220, 0)]
for t in (30, 90, 150, 240):
    im = np.asarray(Image.open(os.path.join(jd, f"{t:05d}.jpg"))).astype(np.float32)
    for k, o in enumerate(names):
        m = masks[o][t]; im[m] = 0.45 * im[m] + 0.55 * np.array(cols[k % 4])
    im = Image.fromarray(im.astype(np.uint8)); dr = ImageDraw.Draw(im)
    for k, o in enumerate(names): dr.text((10, 10 + 14 * k), o, fill=cols[k % 4])
    im.save(os.path.join(a.out, f"check_{t:04d}.png"))
json.dump(clicks, open(os.path.join(a.out, "clicks.json"), "w"), indent=1)
