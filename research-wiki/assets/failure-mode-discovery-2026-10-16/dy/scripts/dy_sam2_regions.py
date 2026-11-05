"""fmd dy P0: region masks on the held-out DyCheck iPhone GROUND TRUTH (MoSca's released test_images), SAM2.1 large
video propagation per held-out camera, from positive clicks on ground-truth frames only (no method render is used).
Objects per scene: hands, held (held object), blades, wheel, person (body; needed for ViDAR's dynamic definition).
Per frame name <cam>_<t>: OUT/<scene>/<object>/<name>.png, plus derived masks:
  dynamic  = union of all objects (ViDAR-style dynamic region: used IN PLACE of co-visibility, as ViDAR's -D metrics)
  fast     = union of hands, held, blades, wheel
  static   = co-visibility mask AND NOT dilate(dynamic, 3 px)   (the static remainder)
  band     = co-visibility AND pixels within 4 px (max-norm) of the dynamic-mask boundary: dilate(m,4) AND dilate(NOT m,4)
             (an 8-px-wide silhouette band; dilation/erosion = 9x9 square max-pooling)
  bandfast = the same band around the fast mask
Check overlays: OUT/<scene>/check_<name>.png. Usage: dy_sam2_regions.py DATA_ROOT CLICKS_JSON OUT SCENE"""
import sys, os, json, glob, shutil, numpy as np, torch
from PIL import Image, ImageDraw
import torch.nn.functional as Fnn
from sam2.build_sam import build_sam2_video_predictor
root, cj, out, scene = sys.argv[1:5]
clicks = json.load(open(cj))[scene]
pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", "/leonardo_work/EUHPC_D36_068/sri/proj_adags/models/sam2/sam2.1_hiera_large.pt", device="cuda")
D = f"{root}/{scene}"; O = f"{out}/{scene}"; os.makedirs(O, exist_ok=True)
allobj = sorted({o for c in clicks.values() for o in c})
stats = {}
for cam, objs in clicks.items():
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{D}/test_images/{cam}_*.png"))
    jd = f"{O}/_jpg_{cam}"; shutil.rmtree(jd, ignore_errors=True); os.makedirs(jd)
    for i, n in enumerate(names):
        Image.open(f"{D}/test_images/{n}.png").convert("RGB").save(f"{jd}/{i:05d}.jpg", quality=95)
    H, W = np.asarray(Image.open(f"{D}/test_images/{names[0]}.png")).shape[:2]
    ol = [o for o in allobj if o in objs and any(objs[o].values())]
    M = {o: np.zeros((len(names), H, W), bool) for o in ol}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = pred.init_state(video_path=jd)
        for oid, o in enumerate(ol, 1):
            for fr, pts in objs[o].items():
                if not pts: continue
                pred.add_new_points_or_box(st, frame_idx=names.index(fr), obj_id=oid, points=np.array([q[:2] for q in pts], np.float32),
                                           labels=np.array([q[2] if len(q) > 2 else 1 for q in pts], np.int32))  # [x,y] positive, [x,y,0] negative
        for rev in (False, True):
            for t, ids, logits in pred.propagate_in_video(st, reverse=rev):
                for k, oid in enumerate(ids):
                    M[ol[oid - 1]][t] |= (logits[k, 0] > 0).cpu().numpy()
    shutil.rmtree(jd)
    for kind in ol + ["dynamic", "fast", "static", "band", "bandfast"]: os.makedirs(f"{O}/{kind}", exist_ok=True)
    fr_dyn = []; checks = set(f for o in ol for f in objs[o]) | {names[len(names) // 4], names[3 * len(names) // 4]}
    for i, n in enumerate(names):
        cov = np.asarray(Image.open(f"{D}/test_covisible/{n}.png").convert("L")) > 127
        dyn = np.zeros((H, W), bool); fast = np.zeros((H, W), bool)
        for o in ol:
            m = M[o][i]; dyn |= m
            if o != "person": fast |= m
            Image.fromarray((m * 255).astype(np.uint8)).save(f"{O}/{o}/{n}.png")
        def dil(m, r):
            x = torch.from_numpy(m.astype(np.float32))[None, None]
            return Fnn.max_pool2d(x, 2 * r + 1, stride=1, padding=r)[0, 0].numpy() > 0.5
        def band(m):
            if not m.any(): return np.zeros_like(m)
            return dil(m, 4) & dil(~m, 4) & cov  # within 4 px of both the object and its outside
        static = cov & ~dil(dyn, 3)
        for kind, m in [("dynamic", dyn), ("fast", fast), ("static", static), ("band", band(dyn)), ("bandfast", band(fast))]:
            Image.fromarray((m * 255).astype(np.uint8)).save(f"{O}/{kind}/{n}.png")
        fr_dyn.append(((dyn & cov).sum() / max(cov.sum(), 1), dyn.mean(), fast.mean()))
        if n in checks:
            im = np.asarray(Image.open(f"{D}/test_images/{n}.png").convert("RGB")).astype(np.float32)
            cols = {"hands": (255, 40, 40), "held": (40, 255, 40), "blades": (40, 255, 40), "wheel": (40, 255, 40), "person": (60, 120, 255)}
            for o in ol:
                m = M[o][i]; im[m] = 0.45 * im[m] + 0.55 * np.array(cols.get(o, (255, 220, 0)))
            im = Image.fromarray(im.astype(np.uint8)); dr = ImageDraw.Draw(im)
            for o, pts in ((o, objs[o].get(n, [])) for o in ol):
                for q in pts: x, y = q[:2]; dr.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(255, 255, 0), width=2)
            dr.text((4, 4), f"{scene} {n} " + " ".join(ol), fill=(255, 255, 0)); im.save(f"{O}/check_{n}.png")
    a = np.array(fr_dyn)
    stats[cam] = {"frames": len(names), "objects": ol, "dynamic_share_of_covisible": round(float(a[:, 0].mean()), 4),
                  "dynamic_area": round(float(a[:, 1].mean()), 4), "fast_area": round(float(a[:, 2].mean()), 4),
                  **{f"{o}_area": round(float(M[o].mean()), 4) for o in ol}}
    print(scene, cam, stats[cam], flush=True)
json.dump({"scene": scene, "clicks": clicks, "stats": stats}, open(f"{O}/regions_info.json", "w"), indent=1)
