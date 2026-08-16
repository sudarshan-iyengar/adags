#!/usr/bin/env python3
"""Protocol-complete evaluation on DiVa-360's OFFICIAL held-out cameras.

Exists because the training loop reports PSNR and SSIM but never LPIPS
(`main.py` maps a `test/lpips` summary key and nothing writes it — see
`research-wiki/operations/diva360-protocol-parity-audit.md` mismatch M3),
and because a sweep needs a single final held-out number that is NOT the
quantity it selected on.

**SELECTION DISCIPLINE.** The six cameras this script scores are the
official held-out split shipped in `transforms_test.json`. They are
SEALED during model selection: rank configurations on a development
split drawn from the 35 TRAINING cameras, pick a winner, and only then
run this once on the winner. Running it on every candidate and taking
the best is selection on the held-out set, which is the thing the split
exists to prevent.

Everything about how images are loaded, composited and resized comes
from the run config, so this scores exactly what the config declares —
at `resolution: 1` that is DiVa-360's own 1160x550 calibration space on
a black background.

LPIPS backbone is DECLARED, not defaulted silently: `--lpips-net`
(default `alex`, matching `lpipsPyTorch`'s own default and the 3DGS
lineage most published GS numbers follow). The criterion is built ONCE;
`lpipsPyTorch.lpips` rebuilds it per call, which would reload weights
for every one of the 846 units.

`--official-metrics` ADDITIONALLY scores every unit under DiVa-360's own
conventions, read out of `utils/benchmark.py` in the official repository
(brown-ivl/DiVa360): `cv.PSNR` on uint8 with R=255, scikit-image's
DEFAULT SSIM (uniform 7x7, data_range 255, channel_axis=2) on uint8, and
torchmetrics LPIPS with `net_type='vgg'` on [-1,1] inputs, each averaged
per image. Two of those three differ materially from what this script
already computes, so both sets are reported side by side on the SAME
images and neither is silently substituted for the other.

WHAT THE OFFICIAL-CONVENTION NUMBERS STILL DO NOT ESTABLISH. They close
the METRIC-DEFINITION gap only. DiVa-360's scissor configuration covers
1125 frames at 30 FPS; the materialization scored here is a 141-frame
window, so a published scissor row remains a comparison across a
different temporal extent no matter which metric convention is used.
That boundary is written into the report rather than left to the reader.
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

import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from depth_visibility.errors import ContractError  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.diva360_official_metrics import (  # noqa: E402
    psnr_official,
    ssim_skimage_default,
    to_uint8_hwc,
)
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402


def _merge_config(args, config_path: str):
    """`main.py`'s own recursive OmegaConf merge, verbatim in behaviour."""
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
    """Test items are `(gt_image, camera)` under `dataloader: True`."""
    if isinstance(data, (list, tuple)) and len(data) == 2:
        return data[1], data[0]
    return data, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--lpips-net", type=str, default="alex",
                        choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--official-metrics", action="store_true",
                        help="also score under DiVa-360's own conventions "
                             "(cv.PSNR/uint8, skimage-default SSIM/uint8, "
                             "torchmetrics VGG LPIPS on [-1,1])")
    parser.add_argument("--torch-home", type=str, default=None,
                        help="writable torch.hub cache; defaults to "
                             "<model_path>/.torchhub. The template's shared "
                             "XDG_CACHE_HOME is not writable by this run.")
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
    args = parser.parse_args(sys.argv[1:])
    _merge_config(args, args.config)

    import os

    run_dir = os.environ.get("ADAGS_RUN_DIR", "").strip()
    if not str(getattr(args, "model_path", "") or "").strip():
        if not run_dir:
            raise ContractError("--model_path is required when ADAGS_RUN_DIR is unset")
        args.model_path = run_dir
    out_path = Path(args.out) if args.out else Path(args.model_path) / "heldout_eval.json"
    os.makedirs(args.model_path, exist_ok=True)

    torch.manual_seed(args.seed)
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(
        dataset.sh_degree, gaussian_dim=args.gaussian_dim,
        time_duration=args.time_duration, rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0,
    )
    scene = Scene(
        dataset, gaussians, num_pts=args.num_pts,
        num_pts_ratio=args.num_pts_ratio, time_duration=args.time_duration,
    )
    scene.opt = opt
    gaussians.training_setup(opt)
    model_params, iteration = torch.load(args.start_checkpoint)
    gaussians.restore(model_params, opt)
    print(f"[eval] restored iteration {iteration}; rows {int(gaussians.get_xyz.shape[0])}",
          flush=True)

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )

    # LPIPS pulls two sets of weights through `torch.hub`: torchvision's
    # backbone and richzhang's linear layers. The template exports
    # XDG_CACHE_HOME=/tmp/adags_cache, which a previous container in this
    # image already created under a different owner — so the hub write
    # fails with PermissionError before either download starts
    # (experiment 85). Point the hub at a directory this run owns, and do
    # it BEFORE the first weight load rather than trusting the shared one.
    hub_dir = Path(args.torch_home) if args.torch_home else Path(args.model_path) / ".torchhub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(hub_dir)
    torch.hub.set_dir(str(hub_dir / "hub"))
    print(f"[eval] torch hub dir: {hub_dir}", flush=True)

    from lpipsPyTorch.modules.lpips import LPIPS

    try:
        criterion = LPIPS(args.lpips_net, "0.1").to("cuda").eval()
    except Exception as exc:
        raise ContractError(
            "LPIPS weights could not be obtained "
            f"({exc!r}). Both the torchvision backbone and richzhang's "
            "linear layers are fetched over the network by torch.hub; if "
            "this container has no egress, LPIPS cannot be computed here "
            "and the protocol gap stands until the weights are staged on "
            "/apollo."
        ) from exc

    # DiVa-360's OWN evaluator, built only when asked for. Its VGG
    # backbone is a SECOND set of weights over the network, so it is not
    # paid for by lanes that do not ask for the official numbers.
    official_lpips = None
    if args.official_metrics:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        try:
            official_lpips = (
                LearnedPerceptualImagePatchSimilarity(net_type="vgg").to("cuda").eval()
            )
        except Exception as exc:
            raise ContractError(
                "DiVa-360's official LPIPS (torchmetrics, net_type='vgg') could "
                f"not be constructed ({exc!r}). The VGG backbone is fetched over "
                "the network by torch.hub; without it the official LPIPS cannot "
                "be computed here and the protocol gap stands."
            ) from exc

    test_cams = scene.getTestCameras()
    if not len(test_cams):
        raise ContractError("no held-out cameras; is ModelParams.eval true?")

    per_camera: dict[str, dict] = {}
    psnrs, ssims, lpipss, lpipss01 = [], [], [], []
    off_psnrs, off_ssims, off_lpipss = [], [], []
    frame_names: set[str] = set()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        for data in test_cams:
            cam, gt_image = _unpack(data)
            cam = cam.cuda()
            pred = render(cam, gaussians, pipe, background)["render"].clamp(0.0, 1.0)
            if gt_image is not None:
                gt = gt_image.cuda()
            else:
                gt = cam.original_image.cuda()
            gt = gt[:3].clamp(0.0, 1.0)
            pred = pred[:3]
            if pred.dim() == 3:
                pred_b, gt_b = pred.unsqueeze(0), gt.unsqueeze(0)
            else:
                pred_b, gt_b = pred, gt
            p = float(psnr(pred_b, gt_b).mean())
            s = float(ssim(pred_b, gt_b).mean())
            # TWO LPIPS CONVENTIONS, both reported, neither guessed.
            # lpipsPyTorch's z-score buffers (-.030/-.088/-.188 over
            # .458/.448/.450) are the reference LPIPS ScalingLayer, which
            # the original implementation documents for inputs in
            # [-1, 1]. But 3DGS's shipped `metrics.py` calls the same
            # wrapper with [0, 1] tensors, and a large share of published
            # GS numbers inherit that. The audit could not verify which
            # one DiVa-360's own script uses (there is no paper page in
            # this repository), so BOTH are computed and labelled rather
            # than one being silently chosen.
            l = float(criterion(pred_b * 2.0 - 1.0, gt_b * 2.0 - 1.0).mean())
            l01 = float(criterion(pred_b, gt_b).mean())
            psnrs.append(p)
            ssims.append(s)
            lpipss.append(l)
            lpipss01.append(l01)

            if official_lpips is not None:
                # DiVa-360's evaluator reads PNG FILES for both the
                # ground truth and the prediction, so the faithful
                # pipeline quantizes the render exactly as the PNG write
                # would and derives ALL THREE official metrics from the
                # quantized data -- including LPIPS, which is fed the
                # uint8 values scaled back to [0,1] and then to [-1,1].
                gt_u8 = to_uint8_hwc(gt_b)
                pred_u8 = to_uint8_hwc(pred_b)
                off_psnrs.append(psnr_official(gt_u8, pred_u8))
                off_ssims.append(ssim_skimage_default(gt_u8, pred_u8))
                gt_q = torch.from_numpy(gt_u8).permute(2, 0, 1).unsqueeze(0)
                pred_q = torch.from_numpy(pred_u8).permute(2, 0, 1).unsqueeze(0)
                gt_q = gt_q.to("cuda", dtype=torch.float32).div_(255.0).mul_(2.0).sub_(1.0)
                pred_q = pred_q.to("cuda", dtype=torch.float32).div_(255.0).mul_(2.0).sub_(1.0)
                off_lpipss.append(float(official_lpips(pred_q, gt_q)))
            # `image_name` is Path(file_path).stem -- the FRAME index, not
            # the camera. Grouping on it yields 141 "cameras" (the frame
            # count) instead of 6. The camera lives in the parent
            # directory of image_path (".../undist/cam17/00000000.png").
            path = str(getattr(cam, "image_path", "") or "").replace("\\", "/")
            parts = [p for p in path.split("/") if p.startswith("cam")]
            key = parts[-1] if parts else str(getattr(cam, "image_name", "?"))
            # `image_name` is the eight-digit FRAME index for this
            # converter, which is exactly what the temporal-extent
            # boundary needs counted.
            frame_names.add(str(getattr(cam, "image_name", "")))
            slot = per_camera.setdefault(
                key, {"n": 0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "lpips_01_input": 0.0}
            )
            slot["n"] += 1
            slot["psnr"] += p
            slot["ssim"] += s
            slot["lpips"] += l
            slot["lpips_01_input"] += l01
    elapsed = time.perf_counter() - started

    for slot in per_camera.values():
        n = max(1, slot["n"])
        for k in ("psnr", "ssim", "lpips", "lpips_01_input"):
            slot[k] /= n

    routing = {}
    try:
        dyn = gaussians.get_dynamic_probability.detach().reshape(-1)
        if dyn.numel():
            routing = {
                "mean_dynamic_prob": float(dyn.mean()),
                "percent_near_dynamic": float((dyn > 0.9).float().mean() * 100.0),
                "percent_near_static": float((dyn < 0.1).float().mean() * 100.0),
            }
    except Exception as exc:  # routing column absent on some lanes
        routing = {"unavailable": repr(exc)}

    report = {
        "schema": "diva360-heldout-eval-v1",
        "checkpoint": args.start_checkpoint,
        "checkpoint_iteration": int(iteration),
        "config": args.config,
        "source_path": str(dataset.source_path),
        "resolution_setting": int(dataset.resolution),
        "white_background": bool(dataset.white_background),
        "lpips_net": args.lpips_net,
        "lpips_version": "0.1",
        "split": "official held-out (transforms_test.json)",
        "n_units": len(psnrs),
        "n_cameras": len(per_camera),
        "points": int(gaussians.get_xyz.shape[0]),
        "routing": routing,
        "psnr": sum(psnrs) / len(psnrs),
        "ssim": sum(ssims) / len(ssims),
        "lpips": sum(lpipss) / len(lpipss),
        "lpips_input_range": "[-1,1] (reference LPIPS ScalingLayer convention)",
        "lpips_01_input": sum(lpipss01) / len(lpipss01),
        "lpips_01_input_note": "same criterion fed [0,1], the convention 3DGS metrics.py ships",
        "per_camera": per_camera,
        "seconds": elapsed,
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "n_frames": len(frame_names),
    }

    if official_lpips is not None:
        finite_psnr = [v for v in off_psnrs if v != float("inf")]
        report["official"] = {
            "source": (
                "brown-ivl/DiVa360 utils/benchmark.py -- cv.PSNR(gt, pred) on "
                "uint8 R=255; skimage structural_similarity(gt, pred, "
                "channel_axis=2) on uint8 (uniform 7x7, data_range 255); "
                "torchmetrics LearnedPerceptualImagePatchSimilarity("
                "net_type='vgg') on inputs mapped to [-1,1]; averaged per image"
            ),
            "psnr": sum(finite_psnr) / len(finite_psnr) if finite_psnr else float("inf"),
            "psnr_infinite_units": len(off_psnrs) - len(finite_psnr),
            "ssim": sum(off_ssims) / len(off_ssims),
            "lpips_vgg": sum(off_lpipss) / len(off_lpipss),
            "ssim_implementation": (
                "utils/diva360_official_metrics.ssim_skimage_default -- "
                "scikit-image is NOT installed in this image, so skimage's "
                "defaults are reproduced exactly and pinned against the real "
                "library by tests/test_diva360_official_metrics.py, which runs "
                "wherever skimage exists and SKIPS here"
            ),
            "quantization": (
                "the render is quantized to uint8 as torchvision's save_image "
                "would write it (x*255 + 0.5, truncated, clamped), because the "
                "official evaluator reads PNG files for BOTH images; all three "
                "official metrics derive from the quantized data"
            ),
            "compositing": (
                "the official evaluator applies rgb*alpha + (1-alpha)*bg to gt "
                "AND pred using each image's OWN alpha. Here the ground truth "
                "is already black-composited once by the lazy dataset path "
                "(utils/data_utils.py) and the render is produced against a "
                "black background by the rasterizer, so it is already composited "
                "by construction. The GT's alpha is NEVER applied to the "
                "prediction -- that would leak ground-truth geometry into the "
                "score."
            ),
            "NOT_ESTABLISHED": (
                "PUBLISHED PARITY. This closes the metric-definition gap only. "
                "DiVa-360's scissor configuration covers 1125 frames at 30 FPS "
                f"(official repo objects_scripts/scissor); this scored {len(frame_names)} "
                "frames, so the temporal extent still differs and no published "
                "scissor row may be placed beside these numbers as a "
                "like-for-like comparison."
            ),
            "published_scissor_black_background": {
                "PF_I-NGP": {"psnr": 25.346, "ssim": 0.944, "lpips": 0.076},
                "MixVoxels": {"psnr": 25.090, "ssim": 0.937, "lpips": 0.086},
                "note": (
                    "DiVa-360 CVPR 2024 supplementary Table 5 (= arXiv 2307.16897 "
                    "v2 Table 12). The K-Planes scissor row is WHITE background "
                    "and is deliberately omitted. Reference only -- see "
                    "NOT_ESTABLISHED above."
                ),
            },
        }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[eval] PSNR {report['psnr']:.4f}  SSIM {report['ssim']:.5f}  "
          f"LPIPS[-1,1] {report['lpips']:.5f}  LPIPS[0,1] "
          f"{report['lpips_01_input']:.5f}  over {report['n_units']} units "
          f"/ {report['n_cameras']} cameras ({args.lpips_net})", flush=True)
    print("HELDOUT_JSON_BEGIN", flush=True)
    print(json.dumps(report, sort_keys=True), flush=True)
    print("HELDOUT_JSON_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
