"""fmd mono2 lane: render a trained Shape of Motion model (checkpoints/last.ckpt, code unmodified) at the held-out cameras.

Held-out cameras come from the sweep tree's camera/camXX_NNNN.json (KNOWN cameras), normalised exactly as
iphone_dataset.py normalises its cameras (w2c @ inv(transfm), translation / scale, scene_norm_dict saved by training).
SceneModel.render(t, w2c, K, (W, H)) with the trainer's defaults (white background, 3DGS). Also writes whole-frame
PSNR vs the fixture frame for every rendered frame.
Usage: render_som.py SCENE DATA_DIR WORK_DIR OUT_DIR
"""
import sys, os, json
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "/leonardo_scratch/large/userexternal/siyengar/fmd/repo/shape-of-motion")
from flow3d.scene_model import SceneModel

scene, data, work, out = sys.argv[1:5]
SRC = f"/leonardo_work/EUHPC_D36_068/sri/proj_adags/fmd/data/mono_real/{scene}/multi4d"
dev = torch.device("cuda")
ck = torch.load(f"{work}/checkpoints/last.ckpt", map_location=dev)
model = SceneModel.init_from_state_dict(ck["model"]).to(dev); model.use_2dgs = False; model.eval()
norm = torch.load(f"{data}/flow3d_preprocessed/cache/scene_norm_dict.pth", map_location="cpu")
transfm, scale = norm["transfm"].float(), float(norm["scale"])
print("epoch", ck.get("epoch"), "step", ck.get("global_step"), "fg", model.fg.num_gaussians,
      "bg", model.bg.num_gaussians if model.bg is not None else 0, "scale", scale, flush=True)
psnr = {}
for c in ["cam00", "cam05", "cam17"]:
    os.makedirs(f"{out}/{c}", exist_ok=True); psnr[c] = {}
    for t in range(300):
        cam = json.load(open(f"{SRC}/camera/{c}_{t:04d}.json"))
        R = np.array(cam["orientation"]); C = np.array(cam["position"]); f = cam["focal_length"]; pp = cam["principal_point"]
        W, H = cam["image_size"]
        w2c = torch.from_numpy(np.block([[R, -R @ C[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]).astype(np.float32))
        w2c = w2c @ torch.linalg.inv(transfm); w2c[:3, 3] /= scale
        K = torch.tensor([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1]], dtype=torch.float32)
        with torch.inference_mode():
            r = model.render(t, w2c[None].to(dev), K[None].to(dev), (W, H))["img"][0].clamp(0, 1).cpu().numpy()
        img = (r * 255).astype(np.uint8)
        Image.fromarray(img).save(f"{out}/{c}/{t:04d}.png")
        gt = np.asarray(Image.open(f"{SRC}/rgb/1x/{c}_{t:04d}.png").convert("RGB")).astype(np.float32) / 255
        psnr[c][t] = float(-10 * np.log10(((img.astype(np.float32) / 255 - gt) ** 2).mean()))
    v = list(psnr[c].values())
    print(c, "whole-frame PSNR vs fixture: all", round(np.mean(v), 2), "40-99", round(np.mean(v[40:100]), 2), flush=True)
json.dump(psnr, open(f"{out}/psnr_wholeframe.json", "w"))
