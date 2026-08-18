"""Held-out evaluation of an LRV1 cell, with the event-region metric.

EXPLORATORY. Scores one trained LRV1 cell on the four held-out cameras and
reports, separately:

* whole-frame PSNR / SSIM / LPIPS over all 240 held-out views;
* the DECISIVE event metric -- reconstruction of the newly revealed surface at
  RETURN, over exactly the pixel-times where the ground-truth renderer says the
  event object is the front-most surface in that held-out view, on the return
  frames only. The object is genuinely absent for the whole preceding gap, so
  every one of those pixel-times is newly revealed by construction. The mask is
  renderer ground truth, so it is independent of every compared method's
  predictions, and it was frozen when the scene was built;
* the same event region over EPISODE-1 frames, which is the in-scene
  reconstructibility oracle: if a cell reconstructs this surface well while the
  object has been present for a long time and badly at return, the deficit is
  specific to the return rather than to the content;
* GHOSTING over the absence gap -- error inside the object's episode-1 footprint
  at frames where the ground truth contains no object at all;
* the ordinary (non-event) region, so a gain at the event that costs ordinary
  quality is visible rather than hidden.

Why this exists rather than `scripts/eval_diva360_heldout.py`: that script never
calls `setup_elgs`, so `gaussians.elgs_runtime` stays None and the renderer
silently falls back to the temporal marginal -- it would score an EL-GS cell
under a presence semantics the model was never trained with. This script
restores the EL-GS state explicitly and refuses to continue if a cell that
declares `elgs_enable` fails to bring its runtime up.

PSNR here pools over channels AND pixels (the standard batched form). It is
reported two ways for every region: `psnr_pooled`, a single PSNR from the MSE
over every pixel-time in the region, which is the primary because it is stable
when a frame contributes few pixels; and `psnr_mean_per_frame`, the mean of
per-frame PSNRs, for comparison with conventions that aggregate that way.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from depth_visibility.errors import ContractError  # noqa: E402
from elgs.trainer_hooks import setup_elgs  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402


def _merge_config(args, config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    def recursive_merge(key, host):
        if isinstance(host[key], dict):
            for kk in host[key]:
                recursive_merge(kk, host[key])
        else:
            assert hasattr(args, key), f"unknown config key {key}"
            setattr(args, key, host[key])

    for key in config:
        recursive_merge(key, config)


def psnr_from_mse(mse):
    if mse <= 0:
        return float("inf")
    return float(20.0 * np.log10(1.0) - 10.0 * np.log10(mse))


class Region:
    """Accumulates squared error over a set of pixel-times."""

    def __init__(self):
        self.sq = 0.0
        self.n = 0
        self.per_frame = []

    def add(self, pred, gt, mask):
        """pred/gt are (3, H, W) float tensors in [0,1]; mask is (H, W) bool."""
        if mask is not None:
            if not bool(mask.any()):
                return
            sel = mask.unsqueeze(0).expand_as(pred)
            diff = (pred - gt)[sel]
        else:
            diff = (pred - gt).reshape(-1)
        sq = float((diff ** 2).sum())
        n = int(diff.numel())
        self.sq += sq
        self.n += n
        self.per_frame.append(psnr_from_mse(sq / n))

    def summary(self):
        if self.n == 0:
            return {"pixel_times": 0, "psnr_pooled": None,
                    "psnr_mean_per_frame": None, "frames": 0}
        return {
            # one pixel-time contributes 3 scalars (RGB); report the pixel count
            "pixel_times": self.n // 3,
            "psnr_pooled": psnr_from_mse(self.sq / self.n),
            "psnr_mean_per_frame": float(np.mean(self.per_frame)),
            "frames": len(self.per_frame),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--cell", type=str, required=True)
    parser.add_argument("--lpips-net", type=str, default="alex",
                        choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--torch-home", type=str, default=None)
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    _merge_config(args, args.config)

    run_dir = os.environ.get("ADAGS_RUN_DIR", "").strip()
    if not str(getattr(args, "model_path", "") or "").strip():
        if not run_dir:
            raise ContractError("--model_path is required when ADAGS_RUN_DIR is unset")
        args.model_path = run_dir
    os.makedirs(args.model_path, exist_ok=True)
    out_path = Path(args.out) if args.out else Path(args.model_path) / "lrv1_event_eval.json"

    spec = json.loads((Path(args.source_path) / "event_spec.json").read_text())
    gt_dir = Path(args.source_path) / "gt_identity"
    obj_id = int(spec["event_object"]["id"])
    return_frames = set(spec["return_frames"])
    ep1 = spec["presence_frames"]["episode_1"]
    gap = spec["presence_frames"]["gap"]
    ep1_frames = set(range(ep1[0], ep1[1] + 1))
    gap_frames = set(range(gap[0], gap[1] + 1))

    torch.manual_seed(args.seed)
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(
        dataset.sh_degree, gaussian_dim=args.gaussian_dim,
        time_duration=args.time_duration, rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0,
    )
    scene = Scene(dataset, gaussians, num_pts=args.num_pts,
                  num_pts_ratio=args.num_pts_ratio, time_duration=args.time_duration,
                  shuffle=False)
    scene.opt = opt
    gaussians.training_setup(opt)
    model_params, ckpt_iteration = torch.load(args.start_checkpoint)
    gaussians.restore(model_params, opt)

    # THE point of this script: restore the presence program the cell trained
    # under. Without it the renderer falls back to `get_marginal_t` and every
    # EL-GS number below would describe a model that was never trained.
    state = setup_elgs(gaussians, scene, dataset, opt)
    elgs_declared = bool(getattr(opt, "elgs_enable", False))
    if elgs_declared and getattr(gaussians, "elgs_runtime", None) is None:
        raise ContractError(
            "cell declares elgs_enable but the EL-GS runtime is not live after "
            "restore; every metric would be scored under the temporal marginal")
    elgs_info = {"declared": elgs_declared, "runtime_live": state is not None}
    if state is not None:
        reg = state.runtime.registry
        ks = [reg.get(f).interval.K for f in reg.active_ids()]
        elgs_info.update({
            "families": len(reg.active_ids()),
            "K_histogram": {str(k): ks.count(k) for k in sorted(set(ks))},
            "oracle_families": sum(1 for k in ks if k > 1),
            "oracle_episodes": str(getattr(opt, "elgs_oracle_episodes", "")),
            "rounds_enabled": state.rounds_enabled,
            "a_lr": state.a_lr,
        })

    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")

    hub_dir = Path(args.torch_home) if args.torch_home else Path(args.model_path) / ".torchhub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(hub_dir)
    torch.hub.set_dir(str(hub_dir))
    from lpipsPyTorch.modules.lpips import LPIPS
    lpips_fn = LPIPS(args.lpips_net, "0.1").to("cuda").eval()

    regions = {k: Region() for k in
               ("whole_frame", "event_return", "event_episode1",
                "ghost_gap", "ordinary_return", "ordinary_all")}
    ssims, lpipss = [], []
    per_return_frame = {}
    missing_masks = 0

    # the object's episode-1 footprint per held-out camera: the union over
    # episode-1 frames of the pixels where the object was front-most. Used to
    # localise ghosting during the gap, where ground truth has no object.
    footprint = {}
    for cam in spec["test_cameras"]:
        acc = None
        for f in sorted(ep1_frames):
            p = gt_dir / ("cam%02d_f%03d.npy" % (cam, f))
            if not p.is_file():
                continue
            m = torch.from_numpy(np.load(p) == obj_id)
            acc = m if acc is None else (acc | m)
        footprint[cam] = acc

    with torch.no_grad():
        for item in scene.getTestCameras():
            cam = item[1] if isinstance(item, (tuple, list)) else item
            gt = (item[0] if isinstance(item, (tuple, list))
                  else cam.original_image)[0:3].cuda().float().clamp(0.0, 1.0)
            pred = render(cam.cuda(), gaussians, pipe, background)["render"].clamp(0.0, 1.0)

            name = cam.image_name                      # cam<NN>_f<FFF>
            cam_idx = int(name.split("_")[0][3:])
            frame = int(name.split("_f")[1])

            regions["whole_frame"].add(pred, gt, None)
            ssims.append(float(ssim(pred.unsqueeze(0), gt.unsqueeze(0)).mean()))
            lpipss.append(float(lpips_fn(pred.unsqueeze(0), gt.unsqueeze(0))))

            mask_path = gt_dir / ("cam%02d_f%03d.npy" % (cam_idx, frame))
            if not mask_path.is_file():
                missing_masks += 1
                continue
            ident = torch.from_numpy(np.load(mask_path)).cuda()
            obj = ident == obj_id
            regions["ordinary_all"].add(pred, gt, ~obj)

            if frame in return_frames:
                regions["event_return"].add(pred, gt, obj)
                regions["ordinary_return"].add(pred, gt, ~obj)
                r = Region()
                r.add(pred, gt, obj)
                per_return_frame.setdefault(str(frame), []).append(
                    r.summary()["psnr_pooled"])
            elif frame in ep1_frames:
                regions["event_episode1"].add(pred, gt, obj)
            elif frame in gap_frames:
                fp = footprint.get(cam_idx)
                if fp is not None:
                    # ground truth here is background: any energy the model puts
                    # inside the vacated footprint is ghosting
                    regions["ghost_gap"].add(pred, gt, fp.cuda())

    if missing_masks:
        raise ContractError(
            "%d held-out views had no ground-truth identity buffer; the event "
            "metric would be computed over an incomplete region" % missing_masks)

    result = {
        "schema_version": "lrv1-event-eval-v1",
        "cell": args.cell,
        "evidence_bearing": False,
        "scene_id": spec["scene_id"],
        "checkpoint_iteration": int(ckpt_iteration),
        "checkpoint": str(args.start_checkpoint),
        "config": str(args.config),
        "primitives": int(gaussians.get_xyz.shape[0]),
        "held_out_cameras": spec["test_cameras"],
        "held_out_views_scored": len(ssims),
        "return_frames": spec["return_frames"],
        "elgs": elgs_info,
        "whole_frame": {
            **regions["whole_frame"].summary(),
            "ssim": float(np.mean(ssims)),
            "lpips_%s_01_input" % args.lpips_net: float(np.mean(lpipss)),
        },
        "event_return": regions["event_return"].summary(),
        "event_episode1": regions["event_episode1"].summary(),
        "ghost_gap": regions["ghost_gap"].summary(),
        "ordinary_return": regions["ordinary_return"].summary(),
        "ordinary_all": regions["ordinary_all"].summary(),
        "event_return_psnr_by_frame": {
            k: float(np.mean(v)) for k, v in sorted(per_return_frame.items())
        },
    }
    out_path.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
