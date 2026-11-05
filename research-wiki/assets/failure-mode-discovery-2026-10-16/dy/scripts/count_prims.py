"""Active primitive counts of a finished MoSca run: static / dynamic Gaussians (opacity > 0.005) and scaffold nodes."""
import sys, os, torch
d = sys.argv[1]
s = torch.load(os.path.join(d, "photometric_s_model_native_add3.pth"), map_location="cpu")
dm = torch.load(os.path.join(d, "photometric_d_model_native_add3.pth"), map_location="cpu")
def act(sd):
    k = [k for k in sd if k.endswith("_opacity")]
    if not k: return None, None
    o = torch.sigmoid(sd[k[0]].float()).flatten(); return int(o.numel()), int((o > 0.005).sum())
ns, nsa = act(s); nd, nda = act(dm)
nodes = [v.shape[1] for k, v in dm.items() if k.endswith("_node_xyz") and hasattr(v, "shape") and v.dim() == 3]
print(f"prims static={ns} static_active={nsa} dynamic={nd} dynamic_active={nda} nodes={nodes[:1]}")
