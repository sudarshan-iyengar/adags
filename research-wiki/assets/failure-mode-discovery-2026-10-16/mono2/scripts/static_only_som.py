"""fmd mono2 mechanism check: render Shape of Motion's background (time-invariant) Gaussians alone vs all, at held-out
cam00 frames 45 and 75, same camera normalisation as render_som.py. Usage: static_only_som.py SCENE DATA WORK OUT"""
import sys, os, json, numpy as np, torch
from PIL import Image
sys.path.insert(0, "/leonardo_scratch/large/userexternal/siyengar/fmd/repo/shape-of-motion")
from flow3d.scene_model import SceneModel
scene, data, work, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True); dev = torch.device("cuda")
SRC = f"/leonardo_work/EUHPC_D36_068/sri/proj_adags/fmd/data/mono_real/{scene}/multi4d"
ck = torch.load(f"{work}/checkpoints/last.ckpt", map_location=dev)
m = SceneModel.init_from_state_dict(ck["model"]).to(dev); m.use_2dgs = False
norm = torch.load(f"{data}/flow3d_preprocessed/cache/scene_norm_dict.pth", map_location="cpu")
nf, N = m.fg.num_gaussians, m.num_gaussians; print("epoch", ck.get("epoch"), "fg", nf, "all", N)
for t in [45, 75]:
    cam = json.load(open(f"{SRC}/camera/cam00_{t:04d}.json"))
    R = np.array(cam["orientation"]); C = np.array(cam["position"]); f = cam["focal_length"]; pp = cam["principal_point"]
    w2c = torch.from_numpy(np.block([[R, -R @ C[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]).astype(np.float32))
    w2c = w2c @ torch.linalg.inv(norm["transfm"].float()); w2c[:3, 3] /= float(norm["scale"])
    K = torch.tensor([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1]], dtype=torch.float32)
    for lab, fm in [("static", torch.arange(N, device=dev) >= nf), ("all", None), ("dynamic", torch.arange(N, device=dev) < nf)]:
        with torch.inference_mode():
            r = m.render(t, w2c[None].to(dev), K[None].to(dev), (1352, 1014), filter_mask=fm)["img"][0].clamp(0, 1).cpu().numpy()
        Image.fromarray((r * 255).astype(np.uint8)).save(f"{out}/cam00_{t:04d}_{lab}.png")
print("done")
