"""fmd mono2 mechanism check: render MoSca's time-invariant static model alone (s_model) vs static + dynamic, at
held-out cam00, frames 45 (bottle present) and 75 (bottle gone), using render_test's own camera alignment (Sim(3) by
the training trajectory) and intrinsics. Usage (MoSca repo root): static_only_mosca.py WS LOGDIR OUT"""
import sys, os, torch, numpy as np, imageio
sys.path.insert(0, os.getcwd())
from mosca_evaluate import *  # MoSca's own loaders, render, alignment
ws, logdir, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True); dev = torch.device("cuda")
cams = MonocularCameras.load_from_ckpt(torch.load(osp.join(logdir, "photometric_cam.pth"))).to(dev); cams.eval()
s_model = StaticGaussian.load_from_ckpt(torch.load(osp.join(logdir, f"photometric_s_model_{GS_BACKEND.lower()}.pth")), device=dev); s_model.eval()
d_model = DynSCFGaussian.load_from_ckpt(torch.load(osp.join(logdir, f"photometric_d_model_{GS_BACKEND.lower()}.pth")), device=dev); d_model.eval()
tr_T, te_T_list, te_tids, te_fns, tr_fov, te_fov, _, te_cxcy = load_iphone_gt_poses(ws, 1)
solved = torch.stack([cams.T_wc(i) for i in range(cams.T)], 0)
al = align_ate_c2b_use_a2b(traj_a=tr_T, traj_b=solved.detach().cpu(), traj_c=te_T_list[0])
H, W = 1014, 1352; focal = 1.0 / np.tan(np.deg2rad(te_fov[0]) / 2.0); L = min(H, W)
K = torch.tensor([[focal * L / 2, 0, W * te_cxcy[0][0]], [0, focal * L / 2, H * te_cxcy[0][1]], [0, 0, 1]], dtype=torch.float32, device=dev)
s_model.to(dev); d_model.to(dev)
print("static n", s_model.N if hasattr(s_model, "N") else "?", "dyn n", d_model.N if hasattr(d_model, "N") else "?")
with torch.no_grad():
    for t in [45, 75]:
        i = te_tids[0].index(t)
        for lab, parts in [("static", [s_model()]), ("all", [s_model(), d_model(t)]), ("dynamic", [d_model(t)])]:
            r = render(parts, H, W, K, T_cw=torch.linalg.inv(al[i]).to(dev))["rgb"].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            imageio.imwrite(f"{out}/cam00_{t:04d}_{lab}.png", (r * 255).astype(np.uint8))
print("done")
