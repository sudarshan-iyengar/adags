#!/usr/bin/env python3
"""POST-HOC mechanism audit of a trained dynamic-Gaussian checkpoint.

Reads a checkpoint and asks WHERE the representation is spending itself
and WHERE its held-out error lives. It changes NO training state: the
model is restored, put in `no_grad`, measured, and discarded.

Motivation: experiment 123 (hull initialization + extended exposure)
reached 23.183 dB and is still improving, but sits ~2.5 dB below the
eight-frame per-frame oracle. Both single-factor interventions are now
measured and neither explains the residual. This audit is the diagnostic
that decides which ONE change to make next, rather than guessing.

--------------------------------------------------------------------
WHAT IS MEASURED, AND AT WHAT GRANULARITY
--------------------------------------------------------------------

Granularity is stated per item because mixing it silently is how a
correlation becomes meaningless.

PER-POINT (N rows, from the checkpoint alone):
  * route probability `get_dynamic_probability`
  * temporal centre `get_t` and temporal scale `get_scaling_t`
  * LoRA coefficient vector norm
  * motion magnitude, speed and acceleration, sampled on the frame grid
  * normalized k-NN motion strain (below)

PER-FRAME (F rows):
  * active-point count -- points whose temporal marginal exceeds a
    declared threshold at that frame
  * mean/percentile motion speed and strain over active points

PER-(CAMERA, FRAME) (C x F rows, held-out cameras only):
  * PSNR, L1, and the same restricted to foreground and to the mask
    boundary band

GLOBAL:
  * LoRA basis and coefficient singular-value spectra
  * capacity/point-generation statistics from the run's own ledger

CORRELATIONS are computed at the FRAME level (error aggregated over
held-out cameras for that frame, against that frame's motion and strain
aggregates). Per-point strain-vs-speed is reported separately. The two
are never pooled.

--------------------------------------------------------------------
NORMALIZED k-NN MOTION STRAIN
--------------------------------------------------------------------

For point i with k nearest neighbours j (fixed at the temporal centre
configuration, so the neighbour SET never changes over time):

    strain_i(t) = median_j  | d_ij(t) - d_ij(0) | / max(d_ij(0), eps)

i.e. the median relative change in neighbour distance. It is
scale-free, so a fast rigid translation contributes ZERO -- which is the
point. High strain means the local neighbourhood is being deformed, and
that is what a single shared low-rank basis with a per-point coefficient
cannot represent well. The median (not the mean) is used so one
runaway neighbour cannot dominate.

Raw per-point, per-frame and per-(camera, frame) rows are written to
disk; every aggregate in the report is recomputed from those rows by
`reduce()`, which is deterministic and takes no arguments beyond them.

Usage:
  python3 scripts/audit_mechanism.py \
      --config configs/elgs/diva360_scissor_bench30_hull_c15k.yaml \
      --source_path <scene> --start_checkpoint <ckpt> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from depth_visibility.errors import ContractError  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402

#: A point counts as ACTIVE at a frame when its temporal marginal is at
#: least this. 0.5 is the half-maximum of the temporal Gaussian, i.e.
#: the point's own temporal FWHM -- a declared constant, not tuned.
ACTIVE_MARGINAL = 0.5
#: Neighbours for the strain statistic. Fixed before running.
STRAIN_K = 8
#: Points subsampled for the O(N*k) strain pass, deterministic stride.
STRAIN_MAX_POINTS = 20000


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


def _spectrum(matrix: torch.Tensor) -> dict:
    """Singular values plus the participation ratio.

    The participation ratio (sum s)^2 / sum s^2 is an EFFECTIVE rank: it
    equals r when r singular values are equal and 1 when one dominates.
    It answers "is the available rank actually being used" without a
    threshold."""
    if matrix.numel() == 0:
        return {"available": False}
    mat = matrix.detach().to(torch.float64).cpu()
    if mat.ndim > 2:
        mat = mat.reshape(mat.shape[0], -1)
    svals = torch.linalg.svdvals(mat)
    s = svals.numpy()
    total = float(s.sum())
    pr = float((total ** 2) / float((s ** 2).sum())) if total > 0 else 0.0
    return {
        "available": True,
        "shape": list(matrix.shape),
        "singular_values": [float(v) for v in s],
        "energy_fraction": [float(v) for v in (s ** 2 / max((s ** 2).sum(), 1e-30))],
        "participation_ratio": pr,
        "max_rank": int(min(mat.shape)),
    }


def _boundary_band(mask: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Pixels within `width` of a foreground/background transition."""
    m = mask.float()[None, None]
    pooled = torch.nn.functional.max_pool2d(m, 2 * width + 1, 1, width)
    eroded = -torch.nn.functional.max_pool2d(-m, 2 * width + 1, 1, width)
    return ((pooled - eroded) > 0).squeeze(0).squeeze(0)


def reduce_rows(points: dict, frames: dict, units: list) -> dict:
    """DETERMINISTIC reducer. Consumes ONLY the raw rows and returns the
    report aggregates. No model, no config, no randomness."""
    out: dict = {}

    def pct(a, qs=(1, 5, 25, 50, 75, 95, 99)):
        if len(a) == 0:
            return {}
        return {f"p{q}": float(np.percentile(a, q)) for q in qs} | {
            "mean": float(np.mean(a)), "min": float(np.min(a)),
            "max": float(np.max(a)), "n": int(len(a)),
        }

    out["route_probability"] = pct(points["route_prob"])
    out["temporal_center"] = pct(points["t_center"])
    out["temporal_scale"] = pct(points["t_scale"])
    out["lora_coeff_norm"] = pct(points["coeff_norm"])
    out["motion_speed_per_point_mean"] = pct(points["speed_mean"])
    out["motion_accel_per_point_mean"] = pct(points["accel_mean"])
    out["motion_displacement_max"] = pct(points["disp_max"])
    strain = points["strain_mean"]
    out["knn_strain_per_point_mean"] = pct(strain[np.isfinite(strain)])

    fi = np.asarray(frames["frame_index"], dtype=float)
    out["per_frame"] = {
        "active_points": pct(np.asarray(frames["active_points"], dtype=float)),
        "mean_speed": pct(np.asarray(frames["mean_speed"], dtype=float)),
        "mean_strain": pct(np.asarray(frames["mean_strain"], dtype=float)),
        "n_frames": int(len(fi)),
    }

    if units:
        arr = {k: np.asarray([u[k] for u in units], dtype=float)
               for k in ("psnr", "l1", "l1_fg", "l1_bnd", "fg_fraction")}
        out["per_unit"] = {k: pct(v) for k, v in arr.items()}
        # frame-level error, averaged over held-out cameras
        by_frame: dict[float, list] = {}
        for u in units:
            by_frame.setdefault(float(u["frame_time"]), []).append(u)
        ftimes = sorted(by_frame)
        err = np.array([np.mean([x["l1"] for x in by_frame[t]]) for t in ftimes])
        psnr = np.array([np.mean([x["psnr"] for x in by_frame[t]]) for t in ftimes])
        ftimes_arr = np.array(ftimes, dtype=float)

        lut_speed = dict(zip(np.asarray(frames["frame_time"], dtype=float),
                             np.asarray(frames["mean_speed"], dtype=float)))
        lut_strain = dict(zip(np.asarray(frames["frame_time"], dtype=float),
                              np.asarray(frames["mean_strain"], dtype=float)))
        lut_active = dict(zip(np.asarray(frames["frame_time"], dtype=float),
                              np.asarray(frames["active_points"], dtype=float)))
        speed = np.array([lut_speed.get(t, np.nan) for t in ftimes_arr])
        strain_f = np.array([lut_strain.get(t, np.nan) for t in ftimes_arr])
        active = np.array([lut_active.get(t, np.nan) for t in ftimes_arr])

        def corr(a, b):
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
                return None
            return float(np.corrcoef(a[m], b[m])[0, 1])

        out["frame_level_correlations"] = {
            "granularity": ("error aggregated over held-out cameras per frame; "
                            "N = number of frames"),
            "n": int(len(ftimes_arr)),
            "l1_vs_mean_speed": corr(err, speed),
            "l1_vs_mean_strain": corr(err, strain_f),
            "l1_vs_active_points": corr(err, active),
            "psnr_vs_mean_speed": corr(psnr, speed),
            "psnr_vs_mean_strain": corr(psnr, strain_f),
        }
        # temporal bins over the frame axis
        bins = np.array_split(np.argsort(ftimes_arr), 5)
        out["error_by_temporal_bin"] = [
            {"bin": i, "n_frames": int(len(b)),
             "t_lo": float(ftimes_arr[b].min()), "t_hi": float(ftimes_arr[b].max()),
             "mean_psnr": float(psnr[b].mean()), "mean_l1": float(err[b].mean())}
            for i, b in enumerate(bins) if len(b)
        ]

    sp = points["speed_mean"]
    st = points["strain_mean"]
    m = np.isfinite(sp) & np.isfinite(st)
    out["point_level_correlations"] = {
        "granularity": "per point, N = number of audited points",
        "n": int(m.sum()),
        "strain_vs_speed": (float(np.corrcoef(sp[m], st[m])[0, 1])
                            if m.sum() > 3 and np.std(sp[m]) > 0 and np.std(st[m]) > 0
                            else None),
    }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start_checkpoint", required=True)
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
    parser.add_argument("--max-frames", type=int, default=0,
                        help="0 = every frame in the scene")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    _merge_config(args, args.config)

    import os

    run_dir = os.environ.get("ADAGS_RUN_DIR", "").strip()
    if not str(getattr(args, "model_path", "") or "").strip():
        args.model_path = run_dir or "."
    out_dir = Path(args.out) if args.out else Path(args.model_path) / "mechanism_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

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
                  num_pts_ratio=args.num_pts_ratio, time_duration=args.time_duration)
    scene.opt = opt
    gaussians.training_setup(opt)
    model_params, iteration = torch.load(args.start_checkpoint)
    gaussians.restore(model_params, opt)
    n_points = int(gaussians.get_xyz.shape[0])
    print(f"[audit] restored iteration {iteration}; {n_points} points", flush=True)

    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    test_cams = scene.getTestCameras()
    if not len(test_cams):
        raise ContractError("no held-out cameras; is ModelParams.eval true?")

    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        # ---------- per-point, checkpoint-only ----------
        xyz0 = gaussians.get_xyz.detach()
        route = gaussians.get_dynamic_probability.detach().reshape(-1)
        t_center = gaussians.get_t.detach().reshape(-1)
        t_scale = gaussians.get_scaling_t.detach().reshape(-1)
        coeff = gaussians.get_motion_lora_coeff.detach()
        basis = gaussians.get_motion_lora_basis
        coeff_norm = (coeff.norm(dim=1) if coeff.numel() else torch.zeros(n_points, device=xyz0.device))

        frame_times = sorted({float(_unpack(d)[0].timestamp) for d in test_cams})
        if args.max_frames and len(frame_times) > args.max_frames:
            idx = np.unique(np.rint(np.linspace(0, len(frame_times) - 1,
                                                args.max_frames)).astype(int))
            frame_times = [frame_times[i] for i in idx]
        print(f"[audit] {len(frame_times)} distinct frame timestamps", flush=True)

        # positions on the frame grid -> displacement, speed, acceleration
        pos = torch.empty((len(frame_times), n_points, 3), dtype=torch.float32,
                          device=xyz0.device)
        active_counts = []
        for i, t in enumerate(frame_times):
            pos[i] = gaussians.get_dynamic_xyz(t).detach()
            marg = gaussians.get_marginal_t(t).detach().reshape(-1)
            active_counts.append(int((marg >= ACTIVE_MARGINAL).sum()))
        dt = np.diff(np.asarray(frame_times, dtype=np.float64))
        dt_t = torch.tensor(dt, dtype=torch.float32, device=xyz0.device)
        vel = (pos[1:] - pos[:-1]) / dt_t.view(-1, 1, 1)
        speed = vel.norm(dim=2)                                   # (F-1, N)
        if speed.shape[0] > 1:
            acc = (vel[1:] - vel[:-1]) / dt_t[1:].view(-1, 1, 1)
            accel = acc.norm(dim=2)
        else:
            accel = torch.zeros_like(speed)
        disp = (pos - pos[0:1]).norm(dim=2)                        # (F, N)

        # ---------- normalized k-NN motion strain ----------
        stride = max(1, n_points // STRAIN_MAX_POINTS)
        sel = torch.arange(0, n_points, stride, device=xyz0.device)[:STRAIN_MAX_POINTS]
        base = xyz0[sel]
        nb = torch.empty((len(sel), STRAIN_K), dtype=torch.long, device=xyz0.device)
        chunk = 2048
        for s in range(0, len(sel), chunk):
            d = torch.cdist(base[s:s + chunk], xyz0)
            d[torch.arange(d.shape[0], device=d.device), sel[s:s + chunk]] = float("inf")
            nb[s:s + chunk] = d.topk(STRAIN_K, largest=False).indices
        d0 = (xyz0[sel][:, None, :] - xyz0[nb]).norm(dim=2).clamp_min(1e-8)
        strain_per_frame = torch.empty((len(frame_times), len(sel)),
                                       dtype=torch.float32, device=xyz0.device)
        for i in range(len(frame_times)):
            dt_i = (pos[i][sel][:, None, :] - pos[i][nb]).norm(dim=2)
            strain_per_frame[i] = ((dt_i - d0).abs() / d0).median(dim=1).values

        # ---------- per-(camera, frame) held-out error ----------
        units = []
        for data in test_cams:
            cam, gt_image = _unpack(data)
            cam = cam.cuda()
            pred = render(cam, gaussians, pipe, background)["render"].clamp(0.0, 1.0)[:3]
            gt = (gt_image.cuda() if gt_image is not None else cam.original_image.cuda())
            gt = gt[:3].clamp(0.0, 1.0)
            err = (pred - gt).abs().mean(dim=0)
            mse = ((pred - gt) ** 2).mean()
            psnr = float(20.0 * torch.log10(1.0 / torch.sqrt(mse.clamp_min(1e-20))))
            fg = (gt.max(dim=0).values > 1e-3)
            bnd = _boundary_band(fg)
            path = str(getattr(cam, "image_path", "") or "").replace("\\", "/")
            parts = [p for p in path.split("/") if p.startswith("cam")]
            units.append({
                "camera": parts[-1] if parts else "?",
                "frame_name": str(getattr(cam, "image_name", "")),
                "frame_time": float(cam.timestamp),
                "psnr": psnr,
                "l1": float(err.mean()),
                "l1_fg": float(err[fg].mean()) if fg.any() else float("nan"),
                "l1_bnd": float(err[bnd].mean()) if bnd.any() else float("nan"),
                "fg_fraction": float(fg.float().mean()),
            })
        print(f"[audit] scored {len(units)} held-out units", flush=True)

    elapsed = time.perf_counter() - started

    # ---------- raw rows ----------
    points_rows = {
        "route_prob": route.cpu().numpy(),
        "t_center": t_center.cpu().numpy(),
        "t_scale": t_scale.cpu().numpy(),
        "coeff_norm": coeff_norm.cpu().numpy(),
        "speed_mean": speed.mean(dim=0).cpu().numpy(),
        "accel_mean": accel.mean(dim=0).cpu().numpy(),
        "disp_max": disp.max(dim=0).values.cpu().numpy(),
        "strain_mean": np.full(n_points, np.nan, dtype=np.float32),
    }
    points_rows["strain_mean"][sel.cpu().numpy()] = (
        strain_per_frame.mean(dim=0).cpu().numpy())
    frames_rows = {
        "frame_index": list(range(len(frame_times))),
        "frame_time": [float(t) for t in frame_times],
        "active_points": active_counts,
        "mean_speed": [0.0] + [float(v) for v in speed.mean(dim=1).cpu().numpy()],
        "mean_strain": [float(v) for v in strain_per_frame.mean(dim=1).cpu().numpy()],
    }
    np.savez_compressed(out_dir / "points_raw.npz", **points_rows)
    (out_dir / "frames_raw.json").write_text(
        json.dumps(frames_rows, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "units_raw.json").write_text(
        json.dumps(units, indent=2, sort_keys=True), encoding="utf-8")

    report = {
        "schema": "adags-mechanism-audit-v1",
        "checkpoint": args.start_checkpoint,
        "checkpoint_iteration": int(iteration),
        "config": args.config,
        "source_path": str(dataset.source_path),
        "n_points": n_points,
        "n_frames_audited": len(frame_times),
        "n_units": len(units),
        "constants": {
            "active_marginal_threshold": ACTIVE_MARGINAL,
            "strain_k": STRAIN_K,
            "strain_points_subsampled_to": int(len(sel)),
            "strain_definition": (
                "median_j |d_ij(t) - d_ij(0)| / d_ij(0) over a neighbour set "
                "FIXED at the temporal-centre configuration; scale-free, so "
                "rigid translation contributes zero"
            ),
            "boundary_band_width_px": 3,
        },
        "lora_basis_spectrum": _spectrum(basis) if basis is not None else {"available": False},
        "lora_coeff_spectrum": _spectrum(coeff),
        "routing_summary": {
            "mean_dynamic_prob": float(route.mean()),
            "percent_near_dynamic": float((route > 0.9).float().mean() * 100.0),
            "percent_near_static": float((route < 0.1).float().mean() * 100.0),
            "percent_uncertain": float(((route >= 0.1) & (route <= 0.9)).float().mean() * 100.0),
        },
        "seconds": elapsed,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "aggregates": reduce_rows(points_rows, frames_rows, units),
    }
    ledger = Path(args.model_path) / "capacity-ledger.json"
    if ledger.is_file():
        report["capacity_ledger"] = json.loads(ledger.read_text(encoding="utf-8"))

    (out_dir / "mechanism_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    agg = report["aggregates"]
    print(f"[audit] route mean {report['routing_summary']['mean_dynamic_prob']:.5f}  "
          f"uncertain {report['routing_summary']['percent_uncertain']:.4f}%", flush=True)
    print(f"[audit] LoRA coeff participation ratio "
          f"{report['lora_coeff_spectrum'].get('participation_ratio')}"
          f" of max rank {report['lora_coeff_spectrum'].get('max_rank')}", flush=True)
    print(f"[audit] LoRA basis participation ratio "
          f"{report['lora_basis_spectrum'].get('participation_ratio')}", flush=True)
    print(f"[audit] frame-level corr: {agg.get('frame_level_correlations')}", flush=True)
    print(f"[audit] point-level corr: {agg.get('point_level_correlations')}", flush=True)
    print(f"[audit] -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
