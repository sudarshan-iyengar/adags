#!/usr/bin/env python3
"""BOUNDED smoke for the rendered-flow supervision path.

Answers, on real data and with no training, the questions that decide
whether a flow supervision result would mean anything:

  A units      are the targets pixels of their own raster, or normalized?
  B direction  does the stored field map t -> t+1, and with which sign?
  C resize     does rescaling to the training raster scale MAGNITUDES,
               not just the grid?
  D masks      what fraction of pixels is actually supervised?
  E shapes     do rendered and target flow reach the loss at the SAME
               resolution, so `compute_flow_loss`'s value-preserving
               interpolate is a no-op?
  F gradients  does the loss produce NONZERO gradients on BOTH the
               per-Gaussian LoRA coefficients AND the shared basis?

F is the one that cannot be reasoned about from source. The renderer
only computes a real displacement under `enable_rendered_flow`, while
the loss fires on `lambda_track_flow` alone with no check on the other
key, so a configuration with only the weight set trains a constant-zero
prediction toward real targets — gradient-free with respect to motion,
and indistinguishable at the metric level from "flow supervision hurt".
This asserts both keys together and both gradient paths individually.

Run on N3V, which is the only substrate carrying audited flow.

NOT a training run and NOT a supervision result: it establishes only
that the plumbing is correct, which must be true before any F/X cell is
interpretable.

Usage:
  python3 scripts/flow_plumbing_smoke.py \
      --config configs/n3v/fixed_budget_lora_route0_600k.yaml \
      --source_path <n3v scene> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from depth_visibility.errors import ContractError  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.motion_prior_utils import normalize_flow_tensor, resize_flow  # noqa: E402


def _merge_config(args, config_path: str):
    cfg = OmegaConf.load(config_path)

    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for inner in host[key].keys():
                recursive_merge(inner, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])

    for key in cfg.keys():
        recursive_merge(key, cfg)
    return args


def _unpack(data):
    if isinstance(data, (list, tuple)) and len(data) == 2:
        return data[1], data[0]
    return data, None


def warp_by_flow(image: torch.Tensor, flow: torch.Tensor, sign: float) -> torch.Tensor:
    """Sample `image` at p + sign*flow(p). Flow is (2,H,W) in PIXELS."""
    _, h, w = image.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=torch.float32),
        torch.arange(w, device=image.device, dtype=torch.float32),
        indexing="ij",
    )
    x = xs + sign * flow[0]
    y = ys + sign * flow[1]
    gx = (x / max(w - 1, 1)) * 2.0 - 1.0
    gy = (y / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None]
    return F.grid_sample(image[None], grid, mode="bilinear",
                         padding_mode="border", align_corners=True)[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--val", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--n-cameras", type=int, default=4,
                        help="how many training views to probe for A-E")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    _merge_config(args, args.config)

    import os

    run_dir = os.environ.get("ADAGS_RUN_DIR", "").strip()
    if not str(getattr(args, "model_path", "") or "").strip():
        args.model_path = run_dir or "."
    out_dir = Path(args.out) if args.out else Path(args.model_path) / "flow_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # BOTH keys on. The whole point of this smoke.
    opt.enable_rendered_flow = True
    if getattr(opt, "lambda_track_flow", 0.0) <= 0:
        opt.lambda_track_flow = 1.0

    gaussians = GaussianModel(
        dataset.sh_degree, gaussian_dim=args.gaussian_dim,
        time_duration=args.time_duration, rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0,
    )
    scene = Scene(dataset, gaussians, num_pts=args.num_pts,
                  num_pts_ratio=args.num_pts_ratio, time_duration=args.time_duration)
    gaussians.training_setup(opt)
    if not getattr(gaussians, "enable_rendered_flow", False):
        raise ContractError(
            "training_setup did not propagate enable_rendered_flow; the "
            "renderer would emit a zero flow field"
        )
    from utils.motion_prior_utils import MotionPriorCache

    scene.motion_prior_cache = MotionPriorCache(dataset.source_path, opt, device="cuda")

    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    train_cams = scene.getTrainCameras()
    report: dict = {
        "schema": "adags-flow-plumbing-smoke-v1",
        "config": args.config,
        "source_path": str(dataset.source_path),
        "enable_rendered_flow": True,
        "lambda_track_flow": float(opt.lambda_track_flow),
        "motion_track_dt": float(getattr(opt, "motion_track_dt", 0.0)),
        "n_train_cameras": len(train_cams),
        "units_note": (
            "project_points_to_screen returns PIXELS of the training raster "
            "((ndc+1)/2 * (W-1)), so the rendered field is a pixel "
            "displacement over motion_track_dt seconds"
        ),
        "probes": [],
    }

    probed = 0
    for index in range(len(train_cams)):
        if probed >= args.n_cameras:
            break
        cam, gt_image = _unpack(train_cams[index])
        cam = cam.cuda()
        gt = (gt_image.cuda() if gt_image is not None else cam.original_image.cuda())
        gt = gt[:3]
        target, mask = scene.motion_prior_cache.get_track_flow(cam, gt.shape[-2:])
        if target is None:
            continue
        probed += 1
        target = target.cuda()
        raw = scene.motion_prior_cache._load_flow_file(
            scene.motion_prior_cache._find_existing(
                cam.image_name, ("track_flows", "flows", "flow", ""),
                (".npy", ".npz", ".pt", ".pth")))
        raw = normalize_flow_tensor(raw)
        entry = {
            "camera": str(getattr(cam, "image_name", "?")),
            "target_shape_after_resize": list(target.shape),
            "gt_shape": list(gt.shape),
            "A_units": {
                "raw_shape": list(raw.shape) if raw is not None else None,
                "raw_abs_median_px": float(raw.abs().median()) if raw is not None else None,
                "raw_abs_p99_px": float(raw.abs().flatten().kthvalue(
                    max(1, int(0.99 * raw.numel()))).values) if raw is not None else None,
                "resized_abs_median_px": float(target.abs().median()),
                "verdict": None,
            },
            "D_mask": {
                "mask_present": mask is not None,
                "supervised_fraction": float(mask.float().mean()) if mask is not None else None,
            },
        }
        # A: pixels vs normalized. A normalized field cannot exceed ~2.
        if raw is not None:
            entry["A_units"]["verdict"] = (
                "PIXELS" if float(raw.abs().max()) > 4.0 else "AMBIGUOUS/NORMALIZED")

        # C: resize must scale magnitudes, not only the grid
        if raw is not None:
            half = (max(raw.shape[-2] // 2, 2), max(raw.shape[-1] // 2, 2))
            r_half = resize_flow(raw.clone(), half)
            ratio_x = float(r_half[0].abs().mean() / max(float(raw[0].abs().mean()), 1e-9))
            entry["C_resize"] = {
                "half_size": list(half),
                "observed_x_magnitude_ratio": ratio_x,
                "expected_about": 0.5,
                "ok": bool(0.4 < ratio_x < 0.6),
            }

        # B: direction/sign, by RGB warp against the NEXT frame
        # BOUNDED scan by INTEGER index. The dataset is lazily decoded,
        # so walking all of it would decode thousands of images for one
        # probe -- and it must not be SLICED either: a slice reaches the
        # lazy dataset's __getitem__ as a slice object and comes back as
        # a list, which is not a Camera. Entries are camera-major, so the
        # successor is a few steps away; 64 caps the cost.
        nxt = None
        this_cam_dir = str(getattr(cam, "image_path", "")).replace("\\", "/").rsplit("/", 2)[-2:-1]
        for j in range(index + 1, min(index + 65, len(train_cams))):
            ocam, ogt = _unpack(train_cams[j])
            other_dir = str(getattr(ocam, "image_path", "")).replace("\\", "/").rsplit("/", 2)[-2:-1]
            if other_dir == this_cam_dir and float(ocam.timestamp) > float(cam.timestamp):
                nxt = (ocam, ogt)
                break
        if nxt is not None:
            ocam, ogt = nxt
            nimg = (ogt.cuda() if ogt is not None else ocam.cuda().original_image.cuda())[:3]
            if nimg.shape == gt.shape:
                base = float((gt - nimg).abs().mean())
                plus = float((warp_by_flow(nimg, target, +1.0) - gt).abs().mean())
                minus = float((warp_by_flow(nimg, target, -1.0) - gt).abs().mean())
                entry["B_direction"] = {
                    "next_timestamp": float(ocam.timestamp),
                    "unwarped_l1": base,
                    "warp_plus_l1": plus,
                    "warp_minus_l1": minus,
                    "best": "plus" if plus < minus else "minus",
                    "improves_over_unwarped": bool(min(plus, minus) < base),
                    "reading": (
                        "sampling I(t+1) at p+F(p) reconstructing I(t) means F "
                        "is the FORWARD t->t+1 field in this codebase's sign "
                        "convention"),
                }

        # E: shapes at the loss
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, background)
        entry["E_shapes"] = {
            "rendered_flow_shape": list(pkg["flow"].shape),
            "target_shape": list(target.shape),
            "match": list(pkg["flow"].shape[-2:]) == list(target.shape[-2:]),
            "rendered_flow_abs_median": float(pkg["flow"].abs().median()),
            "rendered_flow_all_zero": bool(float(pkg["flow"].abs().max()) == 0.0),
        }
        report["probes"].append(entry)

    if not report["probes"]:
        raise ContractError("no training camera had a flow target; check the flow root")

    # ---------- F: gradients through BOTH LoRA paths ----------
    from main import compute_flow_loss

    cam, gt_image = _unpack(train_cams[0])
    cam = cam.cuda()
    gt = (gt_image.cuda() if gt_image is not None else cam.original_image.cuda())[:3]
    target, mask = scene.motion_prior_cache.get_track_flow(cam, gt.shape[-2:])
    target = target.cuda()
    coeff = gaussians._motion_lora_coeff
    basis = gaussians._motion_lora_basis
    grad_report = {
        "coeff_is_parameter": bool(isinstance(coeff, torch.nn.Parameter)),
        "coeff_requires_grad": bool(getattr(coeff, "requires_grad", False)),
        "basis_is_parameter": bool(isinstance(basis, torch.nn.Parameter)),
        "basis_requires_grad": bool(getattr(basis, "requires_grad", False)),
        "coeff_shape": list(coeff.shape) if coeff is not None and coeff.numel() else None,
        "basis_shape": list(basis.shape) if basis is not None else None,
    }
    for p in (coeff, basis):
        if p is not None and getattr(p, "grad", None) is not None:
            p.grad = None
    pkg = render(cam, gaussians, pipe, background)
    loss = compute_flow_loss(pkg["flow"], target, mask)
    if loss is None:
        raise ContractError("compute_flow_loss returned None on a real target")
    grad_report["flow_loss"] = float(loss)
    loss.backward()
    for name, p in (("coeff", coeff), ("basis", basis)):
        g = getattr(p, "grad", None) if p is not None else None
        grad_report[f"{name}_grad_present"] = g is not None
        grad_report[f"{name}_grad_absmax"] = float(g.abs().max()) if g is not None else 0.0
        grad_report[f"{name}_grad_nonzero_fraction"] = (
            float((g.abs() > 0).float().mean()) if g is not None else 0.0)
    grad_report["BOTH_PATHS_RECEIVE_GRADIENT"] = bool(
        grad_report.get("coeff_grad_absmax", 0.0) > 0.0
        and grad_report.get("basis_grad_absmax", 0.0) > 0.0)
    report["F_gradients"] = grad_report

    # ---------- G: is a zero field a DEGENERATE INIT or a SEVERED PATH?
    # A freshly initialized model can have exactly-zero motion, in which
    # case screen_next == screen_now and the rendered field is legitimately
    # zero. That is indistinguishable at the output from a gradient path
    # that does not connect. Perturbing the coefficients to a definitely
    # nonzero motion separates the two: if the field becomes nonzero AND
    # gradients appear, the zero was the init; if the field becomes
    # nonzero and gradients stay zero, the path is severed and no
    # supervision cell would learn anything.
    perturb = {"attempted": False}
    if coeff is not None and coeff.numel():
        perturb["attempted"] = True
        with torch.no_grad():
            saved = coeff.detach().clone()
            gen = torch.Generator(device=coeff.device).manual_seed(0)
            coeff.copy_(torch.randn(coeff.shape, generator=gen,
                                    device=coeff.device, dtype=coeff.dtype) * 0.1)
        for p in (coeff, basis):
            if p is not None and getattr(p, "grad", None) is not None:
                p.grad = None
        pkg2 = render(cam, gaussians, pipe, background)
        perturb["rendered_flow_absmax"] = float(pkg2["flow"].abs().max())
        perturb["rendered_flow_nonzero"] = perturb["rendered_flow_absmax"] > 0.0
        loss2 = compute_flow_loss(pkg2["flow"], target, mask)
        perturb["flow_loss"] = float(loss2)
        loss2.backward()
        for name, p in (("coeff", coeff), ("basis", basis)):
            g = getattr(p, "grad", None) if p is not None else None
            perturb[f"{name}_grad_absmax"] = float(g.abs().max()) if g is not None else 0.0
        perturb["BOTH_PATHS_RECEIVE_GRADIENT"] = bool(
            perturb.get("coeff_grad_absmax", 0.0) > 0.0
            and perturb.get("basis_grad_absmax", 0.0) > 0.0)
        perturb["verdict"] = (
            "DEGENERATE INIT -- path is live once motion is nonzero"
            if perturb["rendered_flow_nonzero"] and perturb["BOTH_PATHS_RECEIVE_GRADIENT"]
            else ("SEVERED GRADIENT PATH -- field responds but no gradient"
                  if perturb["rendered_flow_nonzero"]
                  else "FIELD DOES NOT RESPOND TO MOTION -- renderer branch not taken"))
        with torch.no_grad():
            coeff.copy_(saved)
    report["G_perturbation"] = perturb

    (out_dir / "flow_plumbing_smoke.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    p0 = report["probes"][0]
    print(f"[flow] units {p0['A_units']['verdict']} "
          f"(raw median {p0['A_units']['raw_abs_median_px']:.3f} px)", flush=True)
    print(f"[flow] resize ok={p0.get('C_resize', {}).get('ok')} "
          f"ratio={p0.get('C_resize', {}).get('observed_x_magnitude_ratio')}", flush=True)
    print(f"[flow] direction {p0.get('B_direction', {}).get('best')} "
          f"improves={p0.get('B_direction', {}).get('improves_over_unwarped')}", flush=True)
    print(f"[flow] shapes match={p0['E_shapes']['match']} "
          f"rendered_all_zero={p0['E_shapes']['rendered_flow_all_zero']}", flush=True)
    print(f"[flow] mask supervised_fraction={p0['D_mask']['supervised_fraction']}", flush=True)
    print(f"[flow] GRADIENTS coeff={grad_report['coeff_grad_absmax']:.3e} "
          f"basis={grad_report['basis_grad_absmax']:.3e} "
          f"BOTH={grad_report['BOTH_PATHS_RECEIVE_GRADIENT']}", flush=True)
    if perturb.get("attempted"):
        print(f"[flow] PERTURBED flow_absmax={perturb['rendered_flow_absmax']:.3e} "
              f"coeff={perturb['coeff_grad_absmax']:.3e} "
              f"basis={perturb['basis_grad_absmax']:.3e}", flush=True)
        print(f"[flow] VERDICT {perturb['verdict']}", flush=True)
    print(f"[flow] -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
