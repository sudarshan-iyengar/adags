from scene.gaussian_model import GaussianModel
import torch
import matplotlib.pyplot as plt
import numpy as np
from scene import Scene
from arguments import ModelParams, PipelineParams, OptimizationParams
from omegaconf import OmegaConf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config file (YAML)")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
args = parser.parse_args()


# Load configuration
cfg = OmegaConf.load(args.config)
lp = ModelParams(parser)
pp = PipelineParams(parser)
op = OptimizationParams(parser)

def recursive_merge(key, host, target):
    from omegaconf import DictConfig
    if isinstance(host[key], DictConfig):
        for key1 in host[key].keys():
            recursive_merge(key1, host[key], target)
    else:
        setattr(target, key, host[key])


for k in cfg.ModelParams.keys():
    recursive_merge(k, cfg.ModelParams, lp)
for k in cfg.OptimizationParams.keys():
    recursive_merge(k, cfg.OptimizationParams, op)
for k in cfg.PipelineParams.keys():
    recursive_merge(k, cfg.PipelineParams, pp)

# === Build Gaussian model and restore checkpoint ===
gaussians = GaussianModel(lp.sh_degree, gaussian_dim=4)
model_params, iteration = torch.load(args.checkpoint)
gaussians.restore(model_params, None)

# === extract temporal scales ===
scales_t = torch.exp(gaussians._scaling_t).detach().cpu().numpy().flatten()

print(f"Loaded {len(scales_t)} Gaussians.")
print(f"Temporal scale range: {scales_t.min():.4f} - {scales_t.max():.4f}")

# === Plot histogram ===
plt.figure(figsize=(6,4))
plt.hist(scales_t, bins=200, color='skyblue')
plt.xlim(0, 4)
plt.xlabel("Temporal scale")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

