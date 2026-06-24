#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import os
import random
import socket
import subprocess
import sys
import uuid
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
import torch.nn.functional as F

from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import l1_loss, ssim, msssim
from utils.image_utils import psnr, easy_cmap
from utils.general_utils import safe_state, knn
from utils.motion_prior_utils import (
    MotionPriorCache,
    edge_magnitude,
    erode_mask,
    masked_l1,
    masked_psnr,
    normalize_flow_tensor,
    sample_mask_at_points,
)
from utils.mesh_utils import GaussianExtractor
from utils.render_utils import generate_path, create_videos
import torchvision.transforms.functional as TF

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    wandb = None
    WANDB_FOUND = False

DEFAULT_MAX_TRAIN_ITERATIONS = 6000


def identity_collate(x):
    return x


def normalize_dynamic_weight(weight, eps=1e-6):
    if weight is None or weight.numel() == 0:
        return weight
    weight = torch.nan_to_num(weight.detach()).clamp_min(0.0)
    positive = weight[weight > 0]
    if positive.numel() == 0:
        return torch.zeros_like(weight)
    scale = torch.quantile(positive, 0.9).clamp_min(eps)
    return (weight / scale).clamp(0.0, 1.0)


def compute_dynamic_densify_weight(gaussians, viewpoint_cam, dynamic_mask, residual_map=None):
    if dynamic_mask is None or gaussians.gaussian_dim != 4 or gaussians.get_xyz.numel() == 0:
        return None
    with torch.no_grad():
        points = gaussians.get_dynamic_xyz(viewpoint_cam.timestamp).detach()
        mask_weight = sample_mask_at_points(dynamic_mask, points, viewpoint_cam)
        motion_weight = gaussians.get_motion_offset(viewpoint_cam.timestamp).detach().norm(dim=1, keepdim=True)
        motion_weight = normalize_dynamic_weight(motion_weight)
        if residual_map is not None:
            residual_weight = normalize_dynamic_weight(sample_mask_at_points(residual_map, points, viewpoint_cam))
        else:
            residual_weight = torch.zeros_like(motion_weight)
        return (mask_weight * (0.34 + 0.33 * motion_weight + 0.33 * residual_weight)).clamp(0.0, 1.0)


def compute_static_exclusion_loss(gaussians, viewpoint_cam, dynamic_mask, visibility_filter):
    if dynamic_mask is None or gaussians.get_xyz.numel() == 0 or gaussians.get_route_logit.numel() == 0:
        return torch.zeros((), device=gaussians.get_xyz.device)
    with torch.no_grad():
        points = gaussians.get_dynamic_xyz(viewpoint_cam.timestamp).detach()
        mask_weight = sample_mask_at_points(dynamic_mask, points, viewpoint_cam)
        if visibility_filter is not None and visibility_filter.numel() == mask_weight.shape[0]:
            mask_weight = mask_weight * visibility_filter.float().unsqueeze(-1)
        mask_weight = mask_weight * gaussians.get_opacity.detach()
    denom = mask_weight.sum().clamp_min(1e-6)
    return (mask_weight * gaussians.get_static_probability).sum() / denom


def compute_flow_loss(pred_flow, target_flow, flow_mask):
    pred_flow = normalize_flow_tensor(pred_flow)
    target_flow = normalize_flow_tensor(target_flow)
    if pred_flow is None or target_flow is None:
        return None
    if pred_flow.shape[-2:] != target_flow.shape[-2:]:
        pred_flow = F.interpolate(pred_flow[None], size=target_flow.shape[-2:], mode="bilinear", align_corners=False)[0]
    return masked_l1(pred_flow, target_flow, flow_mask)


def scheduled_flow_weight(opt, iteration):
    base_weight = float(getattr(opt, "lambda_track_flow", 0.0))
    if base_weight <= 0.0:
        return 0.0
    start_iter = int(getattr(opt, "track_flow_loss_start_iter", 0))
    if iteration < start_iter:
        return 0.0
    ramp_iters = int(getattr(opt, "track_flow_loss_ramp_iters", 0))
    if ramp_iters <= 0:
        return base_weight
    progress = float(iteration - start_iter + 1) / float(max(ramp_iters, 1))
    return base_weight * max(0.0, min(1.0, progress))


def build_track_flow_loss_mask(track_flow_mask, dynamic_mask, erode_pixels=0):
    if track_flow_mask is None:
        mask = dynamic_mask
    else:
        mask = track_flow_mask
        if dynamic_mask is not None:
            mask = (mask * dynamic_mask).clamp(0.0, 1.0)
    if mask is not None and erode_pixels > 0:
        mask = erode_mask(mask, erode_pixels)
    return mask


def erode_optional_mask(mask, erode_pixels=0):
    erode_pixels = int(erode_pixels)
    if mask is None or erode_pixels <= 0:
        return mask
    return erode_mask(mask, erode_pixels)


def build_motion_supervision_masks(dynamic_mask, opt):
    if dynamic_mask is None:
        return None, None, None
    dynamic_roi_mask = erode_optional_mask(
        dynamic_mask,
        getattr(opt, "dynamic_roi_mask_erode", 0),
    )
    static_exclusion_mask = erode_optional_mask(
        dynamic_mask,
        getattr(opt, "static_exclusion_mask_erode", 0),
    )
    dynamic_densify_mask = erode_optional_mask(
        dynamic_mask,
        getattr(opt, "dynamic_densify_mask_erode", 0),
    )
    return dynamic_roi_mask, static_exclusion_mask, dynamic_densify_mask


def evaluate_motion_prior_test_metrics(scene, pipe, background, prior_cache, clamp_pred=True):
    if prior_cache is None:
        return {}

    test_cams = scene.getTestCameras()
    if len(test_cams) == 0:
        return {}

    dyn_psnrs = []
    static_region_psnrs = []
    static_ghost_scores = []
    dynamic_edge_scores = []
    track_flow_errors = []

    with torch.no_grad():
        for data in test_cams:
            if isinstance(data, (list, tuple)) and len(data) == 2:
                gt_image, cam = data
            else:
                gt_image, cam = None, data

            cam = cam.cuda()
            render_out = render(cam, scene.gaussians, pipe, background)
            pred = render_out["render"]
            if clamp_pred:
                pred = pred.clamp(0.0, 1.0)

            if gt_image is not None:
                gt = gt_image.cuda()
            elif hasattr(cam, "original_image"):
                gt = cam.original_image.cuda()
            elif hasattr(cam, "gt_image"):
                gt = cam.gt_image.cuda()
            else:
                raise ValueError("No ground truth image found for test camera.")

            dyn_mask = prior_cache.get_dynamic_mask(
                cam,
                target_hw=gt.shape[-2:],
                gt_image=gt,
                pred_image=pred,
                allow_residual=False,
            )
            if dyn_mask is not None and dyn_mask.sum() > 1:
                dyn_psnr = masked_psnr(pred, gt, dyn_mask)
                if dyn_psnr is not None:
                    dyn_psnrs.append(dyn_psnr.item())

                static_mask = (1.0 - dyn_mask).clamp(0.0, 1.0)
                if static_mask.sum() > 1:
                    static_psnr = masked_psnr(pred, gt, static_mask)
                    if static_psnr is not None:
                        static_region_psnrs.append(static_psnr.item())

                static_render = render_out.get("render_3d")
                if static_render is not None:
                    static_ghost_scores.append(
                        (static_render.abs().mean(dim=0, keepdim=True) * dyn_mask)
                        .sum()
                        .div(dyn_mask.sum().clamp_min(1e-6))
                        .item()
                    )
                dynamic_edge_scores.append(
                    (edge_magnitude(pred) * dyn_mask)
                    .sum()
                    .div(dyn_mask.sum().clamp_min(1e-6))
                    .item()
                )

            track_flow, track_flow_mask = prior_cache.get_track_flow(cam, gt.shape[-2:])
            if track_flow is not None:
                if track_flow_mask is None:
                    track_flow_mask = dyn_mask
                flow_loss = compute_flow_loss(render_out.get("flow", None), track_flow, track_flow_mask)
                if flow_loss is not None:
                    track_flow_errors.append(flow_loss.item())

    metrics = {}
    if dyn_psnrs:
        metrics["test/dynamic_mask_psnr"] = float(np.mean(dyn_psnrs))
    if static_region_psnrs:
        metrics["test/static_region_psnr"] = float(np.mean(static_region_psnrs))
    if static_ghost_scores:
        metrics["test/static_ghost_score"] = float(np.mean(static_ghost_scores))
    if dynamic_edge_scores:
        metrics["test/dynamic_edge_magnitude"] = float(np.mean(dynamic_edge_scores))
    if track_flow_errors:
        metrics["test/track_flow_l1"] = float(np.mean(track_flow_errors))
    return metrics


def ensure_model_path(args):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID') if os.getenv('OAR_JOB_ID') else str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    return args.model_path


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def get_git_branch():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def get_git_dirty():
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return bool(status)
    except (OSError, subprocess.SubprocessError):
        return None


def get_job_metadata():
    job_id = None
    job_env_var = None
    for env_name in ("OAR_JOB_ID", "SLURM_JOB_ID", "PBS_JOBID", "JOB_ID"):
        env_value = os.getenv(env_name)
        if env_value:
            job_id = env_value
            job_env_var = env_name
            break

    return {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "job_id": job_id,
        "job_id_env": job_env_var,
        "git_branch": get_git_branch(),
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
    }


WANDB_CONFIG_EXCLUDE_KEYS = {
    "config",
    "debug_from",
    "detect_anomaly",
    "experiment_name",
    "exhaust_test",
    "from3dgs",
    "images",
    "loaded_pth",
    "method_family",
    "model_path",
    "quiet",
    "save_iterations",
    "source_path",
    "start_checkpoint",
    "test_iterations",
    "use_wandb",
    "val",
    "budget_label",
}

WANDB_CONFIG_EXCLUDE_PREFIXES = (
    "runtime_",
    "wandb_",
)


def normalize_wandb_value(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


DATASET_GROUP_TAGS = {"n3v", "panopticsports"}
MAX_WANDB_TAG_LENGTH = 64


def infer_scene_name(args):
    source_path = getattr(args, "source_path", None)
    if source_path:
        scene = os.path.basename(os.path.normpath(str(source_path)))
        if scene:
            return scene

    for tag in getattr(args, "wandb_tags", None) or []:
        if tag not in {"train", "eval", "validation"} | DATASET_GROUP_TAGS:
            return tag
    return None


def infer_method_name(args):
    config_path = getattr(args, "config", None)
    if config_path:
        return os.path.splitext(os.path.basename(str(config_path)))[0]

    tags = getattr(args, "wandb_tags", None) or []
    for tag in tags:
        if tag not in {"train", "eval", "validation", infer_scene_name(args)} | DATASET_GROUP_TAGS:
            return tag
    return None


def infer_wandb_job_type(args):
    tags = set(getattr(args, "wandb_tags", None) or [])
    if getattr(args, "val", False) or "eval" in tags or "validation" in tags:
        return "eval"
    return "train"


def resolve_wandb_group(args):
    scene_name = infer_scene_name(args)
    requested_group = getattr(args, "wandb_group", None)
    if scene_name and (requested_group is None or requested_group in {""} | DATASET_GROUP_TAGS):
        return scene_name
    return requested_group or scene_name


def build_wandb_tags(args):
    tags = []

    def add_tag(tag):
        if tag is None:
            return
        tag = str(tag)
        if not tag:
            return
        if len(tag) > MAX_WANDB_TAG_LENGTH:
            tag = tag[:MAX_WANDB_TAG_LENGTH]
        if tag not in tags:
            tags.append(tag)

    for tag in getattr(args, "wandb_tags", None) or []:
        add_tag(tag)

    for tag in (infer_scene_name(args), infer_method_name(args), infer_wandb_job_type(args)):
        add_tag(tag)

    requested_group = getattr(args, "wandb_group", None)
    if requested_group and requested_group != resolve_wandb_group(args):
        add_tag(f"group:{requested_group}")
    return tags


def build_wandb_config(args):
    config = {}
    for key, value in vars(args).items():
        if key in WANDB_CONFIG_EXCLUDE_KEYS:
            continue
        if any(key.startswith(prefix) for prefix in WANDB_CONFIG_EXCLUDE_PREFIXES):
            continue
        config[key] = normalize_wandb_value(value)

    return config


def build_wandb_metadata(args):
    metadata = {
        "metadata/source_path": getattr(args, "source_path", None),
        "metadata/model_path": getattr(args, "model_path", None),
        "metadata/config_path": getattr(args, "config", None),
        "metadata/config_name": infer_method_name(args),
        "metadata/experiment_name": getattr(args, "experiment_name", None),
        "metadata/method_family": getattr(args, "method_family", None),
        "metadata/budget_label": getattr(args, "budget_label", None),
        "metadata/scene": infer_scene_name(args),
        "metadata/run_phase": infer_wandb_job_type(args),
        "metadata/requested_group": getattr(args, "wandb_group", None),
        "metadata/resolved_group": resolve_wandb_group(args),
        "metadata/wandb_run_name": getattr(args, "wandb_run_name", None),
        "metadata/wandb_resume": getattr(args, "wandb_resume", None),
    }

    for key, value in get_job_metadata().items():
        metadata[f"runtime/{key}"] = value
    return metadata


def init_wandb(args):
    if not args.use_wandb or args.wandb_mode == "disabled":
        return None

    if not WANDB_FOUND:
        raise ImportError("Weights & Biases logging requested, but `wandb` is not installed.")

    if args.wandb_mode == "online" and not has_wandb_credentials():
        raise RuntimeError(
            "Weights & Biases online mode requires credentials from WANDB_API_KEY "
            "or `wandb login`. Use `--wandb_mode offline` for dry runs without credentials."
        )

    ensure_model_path(args)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=resolve_wandb_group(args),
        job_type=infer_wandb_job_type(args),
        tags=build_wandb_tags(args),
        mode=args.wandb_mode,
        id=args.wandb_resume,
        resume="allow" if args.wandb_resume else None,
        dir=args.model_path,
        config=build_wandb_config(args),
    )
    run.summary["model_path"] = args.model_path
    run.summary["config_path"] = args.config
    for key, value in build_wandb_metadata(args).items():
        if value is not None:
            run.summary[key] = value
    if args.start_checkpoint:
        run.summary["start_checkpoint"] = args.start_checkpoint
    return run


def has_wandb_credentials():
    if os.getenv("WANDB_API_KEY"):
        return True

    try:
        import netrc

        host = os.getenv("WANDB_BASE_URL", "https://api.wandb.ai")
        host = host.removeprefix("https://").removeprefix("http://").split("/", 1)[0]
        return netrc.netrc().authenticators(host) is not None
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False


def finish_wandb_run(wandb_run, summary_updates=None):
    if wandb_run is None:
        return

    if summary_updates:
        for key, value in summary_updates.items():
            if value is not None:
                wandb_run.summary[key] = value
    wandb_run.finish()


def maybe_wandb_histogram(values):
    if values is None or not WANDB_FOUND:
        return None
    values = values.detach()
    if values.numel() == 0:
        return None
    return wandb.Histogram(values.float().cpu().numpy())


def log_wandb_metrics(wandb_run, metrics, step):
    if wandb_run is None:
        return

    clean_metrics = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if torch.is_tensor(value):
            if value.numel() == 1:
                clean_metrics[key] = value.item()
            else:
                clean_metrics[key] = maybe_wandb_histogram(value)
        else:
            clean_metrics[key] = value

    clean_metrics = {key: value for key, value in clean_metrics.items() if value is not None}
    if clean_metrics:
        wandb_run.log(clean_metrics, step=step)


def summary_scalar(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return value.detach().item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (int, float, bool, str)):
        return value
    return None


def add_scalar_metric(metrics, key, value):
    value = summary_scalar(value)
    if value is not None:
        metrics[key] = value


def tensor_mean_norm(value, dim=-1):
    if value is None or not torch.is_tensor(value) or value.numel() == 0:
        return None
    value = value.detach()
    if value.dim() == 0:
        return value.abs().item()
    if value.dim() == 1:
        return value.norm().item()
    return value.norm(dim=dim).mean().item()


def collect_decomposition_diagnostics(gaussians, opt=None):
    metrics = {}

    total_points = int(gaussians.get_xyz.shape[0])
    static_points = int(gaussians.get_static_xyz.shape[0]) if hasattr(gaussians, "get_static_xyz") else 0
    dynamic_points = max(0, total_points - static_points)
    add_scalar_metric(metrics, "points/total", total_points)
    add_scalar_metric(metrics, "points/static", static_points)
    add_scalar_metric(metrics, "points/dynamic", dynamic_points)
    add_scalar_metric(metrics, "points/hard_static", static_points)
    add_scalar_metric(metrics, "points/hard_dynamic", dynamic_points)
    add_scalar_metric(metrics, "points/hard_static_fraction", static_points / total_points if total_points > 0 else 0.0)

    route_logit = getattr(gaussians, "get_route_logit", None)
    if torch.is_tensor(route_logit) and route_logit.numel() > 0:
        p_dyn = gaussians.get_dynamic_probability.detach().clamp(1e-6, 1.0 - 1e-6)
        route_entropy = (-(p_dyn * torch.log(p_dyn) + (1.0 - p_dyn) * torch.log(1.0 - p_dyn))).mean()
        expected_static = (1.0 - p_dyn).sum().item()
        expected_dynamic = p_dyn.sum().item()
        add_scalar_metric(metrics, "routing/mean_dynamic_prob", p_dyn.mean().item())
        add_scalar_metric(metrics, "routing/entropy", route_entropy.item())
        add_scalar_metric(metrics, "routing/expected_static_points", expected_static)
        add_scalar_metric(metrics, "routing/expected_dynamic_points", expected_dynamic)
        add_scalar_metric(metrics, "routing/expected_static_fraction", expected_static / total_points if total_points > 0 else 0.0)
        add_scalar_metric(metrics, "routing/percent_near_static", (p_dyn < 0.1).float().mean().item() * 100)
        add_scalar_metric(metrics, "routing/percent_near_dynamic", (p_dyn > 0.9).float().mean().item() * 100)
        add_scalar_metric(metrics, "routing/percent_uncertain", ((p_dyn >= 0.1) & (p_dyn <= 0.9)).float().mean().item() * 100)

    lora_coeff = getattr(gaussians, "get_motion_lora_coeff", None)
    if torch.is_tensor(lora_coeff) and lora_coeff.numel() > 0:
        add_scalar_metric(metrics, "motion_lora/coeff_norm_mean", tensor_mean_norm(lora_coeff, dim=1))
        add_scalar_metric(metrics, "motion_lora/basis_norm_mean", tensor_mean_norm(getattr(gaussians, "get_motion_lora_basis", None), dim=-1))

    scaffold_coeff = getattr(gaussians, "get_motion_scaffold_coeff", None)
    if torch.is_tensor(scaffold_coeff) and scaffold_coeff.numel() > 0:
        add_scalar_metric(metrics, "motion_scaffold/node_count", scaffold_coeff.shape[0])
        add_scalar_metric(metrics, "motion_scaffold/coeff_norm_mean", tensor_mean_norm(scaffold_coeff, dim=1))
        add_scalar_metric(metrics, "motion_scaffold/basis_norm_mean", tensor_mean_norm(getattr(gaussians, "get_motion_scaffold_basis", None), dim=-1))
        attach_w = getattr(gaussians, "get_motion_scaffold_attach_w", None)
        if torch.is_tensor(attach_w) and attach_w.numel() > 0:
            attach_w = attach_w.detach().clamp_min(1e-6)
            if attach_w.dim() > 1:
                attach_entropy = (-(attach_w * torch.log(attach_w)).sum(dim=1)).mean().item()
            else:
                attach_entropy = (-(attach_w * torch.log(attach_w)).sum()).item()
            add_scalar_metric(metrics, "motion_scaffold/attach_entropy", attach_entropy)

    hard_static_conversion = bool(getattr(opt, "enable_hard_static_conversion", False)) if opt is not None else False
    if hard_static_conversion:
        if hasattr(gaussians, "_staticness_score") and opt is not None and gaussians._staticness_score.numel() > 0:
            conversion_rate = (gaussians._staticness_score > opt.static_conversion_threshold).float().mean().item() * 100
            add_scalar_metric(metrics, "points/static_conversion_rate", conversion_rate)
        if hasattr(gaussians, "num_static_candidates_last"):
            add_scalar_metric(metrics, "static_conversion/num_candidates", gaussians.num_static_candidates_last)
        if hasattr(gaussians, "num_converted_last"):
            add_scalar_metric(metrics, "static_conversion/num_converted", gaussians.num_converted_last)
            num_candidates = getattr(gaussians, "num_static_candidates_last", 0)
            if num_candidates > 0:
                add_scalar_metric(metrics, "static_conversion/frac_converted", gaussians.num_converted_last / max(1, num_candidates))

    return metrics


def summary_alias_name(metric_name):
    if metric_name.startswith("test/"):
        return metric_name[len("test/"):]
    return metric_name


def prefixed_summary_metrics(prefix, metrics):
    updates = {}
    for key, value in (metrics or {}).items():
        value = summary_scalar(value)
        if value is not None:
            updates[f"{prefix}/{summary_alias_name(key)}"] = value
    return updates


def build_validation_summary_updates(metrics, iteration, include_best=True, include_final=True):
    updates = {}
    if not metrics:
        return updates

    psnr_value = summary_scalar(metrics.get("test/psnr"))
    if include_best:
        updates["best_val_psnr"] = psnr_value
        updates["best_val_iter"] = iteration if psnr_value is not None else None
        updates.update(prefixed_summary_metrics("best_val", metrics))
    if include_final:
        updates["final_psnr"] = psnr_value
        updates["final_val_iter"] = iteration
        updates.update(prefixed_summary_metrics("final", metrics))
    return updates


def normalize_iteration_schedule(values, final_iteration, include_final=True):
    normalized = []
    for value in values or []:
        value = int(value)
        if 1 <= value <= final_iteration:
            normalized.append(value)
    if include_final and final_iteration > 0:
        normalized.append(int(final_iteration))
    return sorted(set(normalized))


def resolve_max_train_iterations():
    raw_value = os.getenv("ADAGS_MAX_ITERATIONS", "").strip()
    if not raw_value:
        return DEFAULT_MAX_TRAIN_ITERATIONS
    try:
        max_iterations = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"ADAGS_MAX_ITERATIONS must be an integer, got {raw_value!r}") from exc
    if max_iterations < 1:
        raise ValueError(f"ADAGS_MAX_ITERATIONS must be positive, got {max_iterations}")
    return max_iterations


def enforce_train_iteration_guard(args):
    if getattr(args, "val", False):
        return
    max_iterations = resolve_max_train_iterations()
    if args.iterations <= max_iterations or os.getenv("ADAGS_ALLOW_LONG_RUNS") == "1":
        return
    raise RuntimeError(
        f"Refusing to train for {args.iterations} iterations; the guarded maximum is {max_iterations}. "
        "Set ADAGS_MAX_ITERATIONS to a larger value or ADAGS_ALLOW_LONG_RUNS=1 for an intentional long run."
    )


def validation(dataset, opt, pipe, checkpoint, gaussian_dim, time_duration, rot_4d, force_sh_3d, num_pts, num_pts_ratio, wandb_run=None):
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    assert checkpoint, "No checkpoint provided for validation"
    scene = Scene(dataset, gaussians, shuffle=False, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration)
    scene.motion_prior_cache = MotionPriorCache(dataset.source_path, opt, device="cuda")

    (model_params, first_iter) = torch.load(checkpoint)
    train_dir = os.path.join(dataset.model_path, 'train', f"ours_{first_iter}")
    test_dir = os.path.join(dataset.model_path, 'test', f"ours_{first_iter}")
    gaussians.restore(model_params, None)
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)

    print("export rendered testing images ...")
    os.makedirs(test_dir, exist_ok=True)
    validation_stats = gaussExtractor.reconstruction(scene.getTestCameras(), test_dir, stage="validation")
    gaussExtractor.export_image(test_dir, mode="validation")
    summary_updates = {
        "validation_checkpoint": checkpoint,
        "validation_output_dir": test_dir,
        "validation_train_dir": train_dir,
    }
    eval_metrics = collect_decomposition_diagnostics(gaussians, opt)
    if validation_stats:
        metric_map = {
            "psnr": "test/psnr",
            "ssim": "test/ssim",
            "lpips": "test/lpips",
            "num_GS": "points/total",
            "static": "points/hard_static",
        }
        for stat_name, metric_name in metric_map.items():
            if stat_name in validation_stats:
                eval_metrics[metric_name] = validation_stats[stat_name]
        if "num_GS" in validation_stats and "static" in validation_stats:
            hard_dynamic = validation_stats["num_GS"] - validation_stats["static"]
            eval_metrics["points/static"] = validation_stats["static"]
            eval_metrics["points/dynamic"] = hard_dynamic
            eval_metrics["points/hard_dynamic"] = hard_dynamic
            eval_metrics["points/hard_static_fraction"] = (
                validation_stats["static"] / validation_stats["num_GS"] if validation_stats["num_GS"] > 0 else 0.0
            )
    eval_metrics.update(evaluate_motion_prior_test_metrics(scene, pipe, background, scene.motion_prior_cache))
    summary_updates.update(build_validation_summary_updates(eval_metrics, first_iter))
    if wandb_run is not None:
        wandb_run.summary["validation_checkpoint"] = checkpoint
        wandb_run.summary["validation_output_dir"] = test_dir
        wandb_run.summary["validation_train_dir"] = train_dir
        log_wandb_metrics(wandb_run, eval_metrics, first_iter)
    return summary_updates


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from,
             gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, batch_size, wandb_run=None):

    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio, time_duration[1] / dataset.frame_ratio]

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    scene = Scene(dataset, gaussians, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration)
    scene.opt = opt
    scene.motion_prior_cache = MotionPriorCache(dataset.source_path, opt, device="cuda")
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    best_psnr = 0.0
    best_val_metrics = None
    best_val_iter = None
    final_val_metrics = None
    final_val_iter = None
    ema_loss_for_log = 0.0
    ema_l1loss_for_log = 0.0
    ema_ssimloss_for_log = 0.0

    lambda_all = [key for key in opt.__dict__.keys() if key.startswith('lambda') and key != 'lambda_dssim']
    for lambda_name in lambda_all:
        vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"] = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if pipe.env_map_res:
        env_map = nn.Parameter(torch.zeros((3, pipe.env_map_res, pipe.env_map_res), dtype=torch.float, device="cuda").requires_grad_(True))
        env_map_optimizer = torch.optim.Adam([env_map], lr=opt.feature_lr, eps=1e-15)
    else:
        env_map = None
        env_map_optimizer = None

    gaussians.env_map = env_map

    training_dataset = scene.getTrainCameras()
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=0 if dataset.dataloader else 0,
                                     collate_fn=identity_collate, drop_last=True)
    iteration = first_iter

    # placeholders for gate losses
    device = "cuda"
    Lsparsity = torch.tensor(0.0, device=device)
    Lmotion_gate = torch.tensor(0.0, device=device)
    Lmotion_reg = torch.tensor(0.0, device=device)
    Ldynamic_roi = torch.tensor(0.0, device=device)
    Lstatic_exclusion = torch.tensor(0.0, device=device)
    Ltrack_flow = torch.tensor(0.0, device=device)
    Lscaffold_smooth = torch.tensor(0.0, device=device)
    Lscaffold_reg = torch.tensor(0.0, device=device)

    motion_gate_quantile = getattr(opt, "motion_gate_quantile", 0.8)
    hard_static_conversion = getattr(opt, "enable_hard_static_conversion", False)
    use_legacy_gate = (
            hard_static_conversion
            or getattr(opt, "lambda_sparsity", 0.0) > 0
            or getattr(opt, "lambda_motion_gate", 0.0) > 0
    )

    while iteration < opt.iterations + 1:
        for batch_data in training_dataloader:
            iteration += 1
            if iteration > opt.iterations:
                break

            iter_start.record()
            gaussians.update_learning_rate(iteration)

            if iteration % opt.sh_increase_interval == 0:
                gaussians.oneupSHdegree()

            if (iteration - 1) == debug_from:
                pipe.debug = True

            # Legacy hard-conversion gate is disabled by default for reversible routing.
            if use_legacy_gate and iteration >= opt.gate_activation_iter:
                gaussians.compute_differentiable_staticness()
            else:
                if gaussians._xyz.shape[0] > 0:
                    gaussians.differentiable_s = torch.zeros((gaussians._xyz.shape[0], 1), device=device, requires_grad=False)

            total_loss = 0.0

            batch_point_grad, batch_visibility_filter, batch_radii = [], [], []
            batch_point_grad_static, batch_visibility_filter_static, batch_radii_static = [], [], []
            batch_dynamic_densify_weight = []
            Ldynamic_roi = torch.tensor(0.0, device=device)
            Lstatic_exclusion = torch.tensor(0.0, device=device)
            Ltrack_flow = torch.tensor(0.0, device=device)
            Lscaffold_smooth = torch.tensor(0.0, device=device)
            Lscaffold_reg = torch.tensor(0.0, device=device)

            static = False
            flow_weight = scheduled_flow_weight(opt, iteration)

            # ================= inner micro-batch loop =================
            for batch_idx in range(batch_size):
                gt_image, viewpoint_cam = batch_data[batch_idx]
                gt_image = gt_image.cuda()
                viewpoint_cam = viewpoint_cam.cuda()

                render_pkg = render(viewpoint_cam, gaussians, pipe, background, render_flow=flow_weight > 0.0)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                depth = render_pkg["depth"]
                alpha = render_pkg["alpha"]
                viewspace_point_tensor_static = render_pkg["viewspace_points_static"]
                visibility_filter_static = render_pkg["visibility_filter_static"]
                radii_static = render_pkg["radii_static"]
                has_hard_static = render_pkg.get("hard_static_count", 0) > 0

                if opt.blur_until_iter > 0 and iteration < opt.blur_until_iter:
                    progress = iteration / float(opt.blur_until_iter)
                    current_sigma = opt.blur_start_sigma * (1.0 - progress) + 0.1 * progress

                    k_size = int(2 * math.ceil(2 * current_sigma)) + 1
                    image_for_loss = TF.gaussian_blur(image, [k_size, k_size], [current_sigma, current_sigma])
                    gt_image_for_loss = TF.gaussian_blur(gt_image, [k_size, k_size], [current_sigma, current_sigma])
                else:
                    image_for_loss = image
                    gt_image_for_loss = gt_image

                # Reconstruction Loss
                Ll1 = l1_loss(image_for_loss, gt_image_for_loss)
                Lssim = 1.0 - ssim(image_for_loss, gt_image_for_loss)
                loss_recon = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
                loss = loss_recon

                dynamic_mask = scene.motion_prior_cache.get_dynamic_mask(
                    viewpoint_cam,
                    target_hw=gt_image.shape[-2:],
                    gt_image=gt_image,
                    pred_image=image,
                    allow_residual=True,
                )
                dynamic_roi_mask, static_exclusion_mask, dynamic_densify_mask = build_motion_supervision_masks(
                    dynamic_mask,
                    opt,
                )

                if dynamic_roi_mask is not None and getattr(opt, "lambda_dynamic_roi", 0.0) > 0:
                    Ldyn = masked_l1(image_for_loss, gt_image_for_loss, dynamic_roi_mask)
                    loss = loss + opt.lambda_dynamic_roi * Ldyn
                    Ldynamic_roi = Ldynamic_roi + Ldyn.detach() / float(batch_size)

                if static_exclusion_mask is not None and getattr(opt, "lambda_static_exclusion", 0.0) > 0:
                    Lstat = compute_static_exclusion_loss(gaussians, viewpoint_cam, static_exclusion_mask, visibility_filter)
                    loss = loss + opt.lambda_static_exclusion * Lstat
                    Lstatic_exclusion = Lstatic_exclusion + Lstat.detach() / float(batch_size)

                if flow_weight > 0:
                    track_flow, track_flow_mask = scene.motion_prior_cache.get_track_flow(viewpoint_cam, gt_image.shape[-2:])
                    if track_flow is not None:
                        track_flow_mask = build_track_flow_loss_mask(
                            track_flow_mask,
                            dynamic_mask,
                            int(getattr(opt, "track_flow_mask_erode", 0)),
                        )
                        Lflow_prior = compute_flow_loss(render_pkg.get("flow", None), track_flow, track_flow_mask)
                        if Lflow_prior is not None:
                            loss = loss + flow_weight * Lflow_prior
                            Ltrack_flow = Ltrack_flow + Lflow_prior.detach() / float(batch_size)

                # opa mask loss
                if opt.lambda_opa_mask > 0:
                    o = alpha.clamp(1e-6, 1 - 1e-6)
                    sky = 1 - viewpoint_cam.gt_alpha_mask
                    Lopa_mask = (-sky * torch.log(1 - o)).mean()
                    loss = loss + opt.lambda_opa_mask * Lopa_mask

                # rigid loss
                if opt.lambda_rigid > 0 and gaussians.gaussian_dim == 4 and gaussians.get_xyz.shape[0] > 0:
                    k = 20
                    xyz_cur = gaussians.get_xyz
                    idx, dist = knn(xyz_cur[None].contiguous().detach(), xyz_cur[None].contiguous().detach(), k)
                    _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                    weight = torch.exp(-100 * dist)
                    vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
                    Lrigid = (weight * vel_dist).sum() / k / xyz_cur.shape[0]
                    loss = loss + opt.lambda_rigid * Lrigid

                if getattr(opt, "enable_motion_aware_densify", False):
                    residual_map = (image.detach() - gt_image.detach()).abs().mean(dim=0, keepdim=True)
                    dyn_weight = compute_dynamic_densify_weight(gaussians, viewpoint_cam, dynamic_densify_mask, residual_map)
                    if dyn_weight is not None:
                        batch_dynamic_densify_weight.append(dyn_weight.squeeze(-1))

                total_loss += loss.item()
                (loss / batch_size).backward(retain_graph=True)

                batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:, :2], dim=-1))
                batch_radii.append(radii)
                batch_visibility_filter.append(visibility_filter)

                if has_hard_static and len(viewspace_point_tensor_static) > 0:
                    static = True
                    batch_point_grad_static.append(torch.norm(viewspace_point_tensor_static.grad[:, :2], dim=-1))
                    batch_radii_static.append(radii_static)
                    batch_visibility_filter_static.append(visibility_filter_static)

            # ================= aggregate grads over micro-batches =================
            if batch_size > 1:
                visibility_count = torch.stack(batch_visibility_filter, 1).sum(1)
                visibility_filter = visibility_count > 0
                radii = torch.stack(batch_radii, 1).max(1)[0]

                batch_viewspace_point_grad = torch.stack(batch_point_grad, 1).sum(1)
                batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)

                if static:
                    visibility_count_static = torch.stack(batch_visibility_filter_static, 1).sum(1)
                    visibility_filter_static = visibility_count_static > 0
                    radii_static = torch.stack(batch_radii_static, 1).max(1)[0]

                    batch_viewspace_point_grad_static = torch.stack(batch_point_grad_static, 1).sum(1)
                    batch_viewspace_point_grad_static[visibility_filter_static] = batch_viewspace_point_grad_static[visibility_filter_static] * batch_size / visibility_count_static[visibility_filter_static]
                    batch_viewspace_point_grad_static = batch_viewspace_point_grad_static.unsqueeze(1)

                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone()[:, 0].detach()
                    batch_t_grad[visibility_filter] = batch_t_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                    batch_t_grad = batch_t_grad.unsqueeze(1)
                dynamic_densify_weight = None
                if batch_dynamic_densify_weight:
                    dynamic_densify_weight = torch.stack(batch_dynamic_densify_weight, 1).max(dim=1).values.unsqueeze(1)
            else:
                visibility_filter = batch_visibility_filter[0]
                radii = batch_radii[0]
                batch_viewspace_point_grad = batch_point_grad[0].unsqueeze(1)
                if static:
                    visibility_filter_static = batch_visibility_filter_static[0]
                    radii_static = batch_radii_static[0]
                    batch_viewspace_point_grad_static = batch_point_grad_static[0].unsqueeze(1)
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone().detach()
                dynamic_densify_weight = batch_dynamic_densify_weight[0].unsqueeze(1) if batch_dynamic_densify_weight else None

            # ================= gate losses (monotonic logistic on log σ_t) =================
            Lsparsity = torch.tensor(0.0, device=device)
            Lmotion_gate = torch.tensor(0.0, device=device)

            if use_legacy_gate and iteration > opt.gate_activation_iter and gaussians._xyz.shape[0] > 0:
                s = gaussians.differentiable_s
                if s is not None and s.numel() > 0:
                    # annealed sparsity toward static (s→1)
                    if opt.gate_warmup_until_iter > opt.gate_activation_iter:
                        anneal = min(1.0, (iteration - opt.gate_activation_iter) / (opt.gate_warmup_until_iter - opt.gate_activation_iter))
                    else:
                        anneal = 1.0
                    lambda_sparsity_eff = opt.lambda_sparsity * anneal
                    Lsparsity = lambda_sparsity_eff * (1.0 - s).mean()

                    # motion magnitude from covariance-induced mean offset (no grad into motion)
                    with torch.no_grad():
                        _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                        motion_mag = velocity.norm(p=2, dim=1, keepdim=True)
                        motion_mag = torch.nan_to_num(motion_mag)
                        if motion_mag.numel() > 0:
                            scale = torch.quantile(motion_mag, motion_gate_quantile).clamp_min(1e-6)
                        else:
                            scale = torch.tensor(1.0, device=device)
                        Lmotion_per_point = motion_mag / scale

                    # penalize s for high-motion gaussians
                    Lmotion_gate = opt.lambda_motion_gate * (s * Lmotion_per_point.detach()).mean()

                    gate_loss = Lsparsity + Lmotion_gate
                    if gate_loss.requires_grad and gate_loss.item() != 0.0:
                        total_loss += gate_loss.item()
                        gate_loss.backward()

            Lmotion_reg = torch.tensor(0.0, device=device)
            if (
                    getattr(opt, "motion_reg_lambda", 0.0) > 0
                    and gaussians.gaussian_dim == 4
            ):
                if getattr(gaussians, "motion_model", "") == "poly" and gaussians.get_motion_v.numel() > 0:
                    Lmotion_reg = opt.motion_reg_lambda * (
                        gaussians.get_motion_v.pow(2).mean()
                        + gaussians.get_motion_a.pow(2).mean()
                    )
                elif (
                        getattr(gaussians, "motion_model", "") == "lora"
                        and gaussians.get_motion_lora_coeff.numel() > 0
                        and gaussians.get_motion_lora_basis is not None
                ):
                    Lmotion_reg = opt.motion_reg_lambda * (
                        gaussians.get_motion_lora_coeff.pow(2).mean()
                        + gaussians.get_motion_lora_basis.pow(2).mean()
                    )
                if Lmotion_reg.requires_grad and Lmotion_reg.item() != 0.0:
                    total_loss += Lmotion_reg.item()
                    Lmotion_reg.backward()

            if (
                    getattr(opt, "lambda_scaffold_smooth", 0.0) > 0
                    and getattr(gaussians, "motion_scaffold_enable", False)
            ):
                Lscaffold_smooth = opt.lambda_scaffold_smooth * gaussians.get_scaffold_smoothness_loss()
                if Lscaffold_smooth.requires_grad and Lscaffold_smooth.item() != 0.0:
                    total_loss += Lscaffold_smooth.item()
                    Lscaffold_smooth.backward()

            if (
                    getattr(opt, "lambda_scaffold_reg", 0.0) > 0
                    and getattr(gaussians, "motion_scaffold_enable", False)
            ):
                Lscaffold_reg = opt.lambda_scaffold_reg * gaussians.get_scaffold_reg_loss()
                if Lscaffold_reg.requires_grad and Lscaffold_reg.item() != 0.0:
                    total_loss += Lscaffold_reg.item()
                    Lscaffold_reg.backward()

            iter_end.record()

            # ================= logging dictionary =================
            loss_dict = {
                "Ll1": Ll1,
                "Lssim": Lssim,
                "Lsparsity": Lsparsity,
                "Lmotion_gate": Lmotion_gate,
                "Lmotion_reg": Lmotion_reg,
                "Ldynamic_roi": Ldynamic_roi,
                "Lstatic_exclusion": Lstatic_exclusion,
                "Ltrack_flow": Ltrack_flow,
                "Lscaffold_smooth": Lscaffold_smooth,
                "Lscaffold_reg": Lscaffold_reg,
            }
            if 'Lrigid' in locals(): loss_dict["Lrigid"] = Lrigid

            with torch.no_grad():
                psnr_for_log = psnr(image, gt_image).mean().double()
                log_wandb_metrics(wandb_run, {"train/psnr": psnr_for_log}, iteration)

                ema_loss_for_log = 0.4 * total_loss + 0.6 * ema_loss_for_log
                ema_l1loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_l1loss_for_log
                ema_ssimloss_for_log = 0.4 * Lssim.item() + 0.6 * ema_ssimloss_for_log

                for lambda_name in lambda_all:
                    if opt.__dict__[lambda_name] > 0:
                        loss_key = f"L{lambda_name.replace('lambda_', '')}"
                        if loss_key in locals():
                            ema_name = f"ema_{lambda_name.replace('lambda_', '')}_for_log"
                            ema_val = vars()[ema_name]
                            vars()[ema_name] = 0.4 * vars()[loss_key].item() + 0.6 * ema_val
                            loss_dict[loss_key] = vars()[loss_key]

                if iteration % 10 == 0:
                    postfix = {
                        "Loss": f"{ema_loss_for_log:.7f}",
                        "PSNR": f"{psnr_for_log:.2f}",
                        "Ll1": f"{ema_l1loss_for_log:.4f}",
                        "Lssim": f"{ema_ssimloss_for_log:.4f}",
                        "points": scene.gaussians.get_xyz.shape[0],
                        "static": scene.gaussians.get_static_xyz.shape[0]
                    }
                    for lambda_name in lambda_all:
                        if opt.__dict__[lambda_name] > 0:
                            key = lambda_name.replace("lambda_", "L")
                            ema_name = f"ema_{lambda_name.replace('lambda_', '')}_for_log"
                            if key in ("Lscaffold_smooth", "Lscaffold_reg"):
                                postfix[key] = f"{vars()[ema_name]:.4e}"
                            else:
                                postfix[key] = f"{vars()[ema_name]:.4f}"
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                eval_metrics = training_report(tb_writer, iteration, Ll1, Lssim, total_loss, l1_loss, iter_start.elapsed_time(iter_end),
                                               testing_iterations, scene, render, (pipe, background), loss_dict, wandb_run)

                if eval_metrics is not None and "test/psnr" in eval_metrics:
                    final_val_metrics = eval_metrics
                    final_val_iter = iteration
                    test_psnr = summary_scalar(eval_metrics.get("test/psnr"))
                    if test_psnr is None:
                        test_psnr = 0.0
                    if test_psnr >= best_psnr:
                        best_psnr = test_psnr
                        best_val_metrics = dict(eval_metrics)
                        best_val_iter = iteration
                        print(f"\n[ITER {iteration}] Saving best checkpoint")
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_best.pth")

                if iteration in saving_iterations:
                    print(f"\n[ITER {iteration}] Saving Gaussians")
                    scene.save(iteration)

                # ================= densification =================
                if iteration < opt.densify_until_iter and (opt.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < opt.densify_until_num_points):
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    if static:
                        gaussians.static_max_radii2D[visibility_filter_static] = torch.max(gaussians.static_max_radii2D[visibility_filter_static], radii_static[visibility_filter_static])

                    if batch_size == 1:
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter,
                                                          batch_t_grad if gaussians.gaussian_dim == 4 else None,
                                                          dynamic_densify_weight)
                    else:
                        gaussians.add_densification_stats_grad(batch_viewspace_point_grad, visibility_filter,
                                                               batch_t_grad if gaussians.gaussian_dim == 4 else None,
                                                               dynamic_densify_weight)
                        if static:
                            gaussians.add_densification_stats_grad_static(batch_viewspace_point_grad_static, visibility_filter_static)

                    if iteration > opt.densify_from_iter:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        if iteration % opt.densification_interval == 0:
                            if hard_static_conversion:
                                gaussians.update_staticness_score()
                            gaussians.densify_and_prune(
                                max_grad=opt.densify_grad_threshold,
                                min_opacity=opt.thresh_opa_prune,
                                extent=scene.cameras_extent,
                                max_screen_size=size_threshold,
                                max_grad_t=opt.densify_grad_t_threshold,
                                static_conversion_threshold=opt.static_conversion_threshold,
                                gate_activation_iter=opt.gate_activation_iter,
                                gate_warmup_until_iter=opt.gate_warmup_until_iter,
                                iteration=iteration,
                                enable_hard_static_conversion=hard_static_conversion,
                                max_total_points=opt.densify_until_num_points,
                            )
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # ================= optimizer step =================
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    if pipe.env_map_res and iteration < pipe.env_optimize_until and env_map_optimizer is not None:
                        env_map_optimizer.step()
                        env_map_optimizer.zero_grad(set_to_none=True)

    final_diagnostics = collect_decomposition_diagnostics(scene.gaussians, opt)
    summary_updates = {
        "best_test_psnr": best_psnr,
        "best_val_psnr": best_psnr if best_val_iter is not None else None,
        "best_val_iter": best_val_iter,
        "final_psnr": summary_scalar((final_val_metrics or {}).get("test/psnr")),
        "final_val_iter": final_val_iter,
        "final_iteration": opt.iterations,
        "final_total_points": scene.gaussians.get_xyz.shape[0],
        "final_static_points": scene.gaussians.get_static_xyz.shape[0] if hasattr(scene.gaussians, 'get_static_xyz') else 0,
        "final_dynamic_points": (
            scene.gaussians.get_xyz.shape[0] - scene.gaussians.get_static_xyz.shape[0]
            if hasattr(scene.gaussians, 'get_static_xyz') else scene.gaussians.get_xyz.shape[0]
        ),
        "model_path": scene.model_path,
        "start_checkpoint": checkpoint,
    }
    if best_val_metrics is not None:
        summary_updates.update(prefixed_summary_metrics("best_val", best_val_metrics))
    final_metrics = dict(final_diagnostics)
    if final_val_metrics is not None:
        final_metrics.update(final_val_metrics)
    summary_updates.update(prefixed_summary_metrics("final", final_metrics))
    return summary_updates


def prepare_output_and_logger(args):
    ensure_model_path(args)

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
        return None


def training_report(tb_writer, iteration, Ll1, Lssim, loss, l1_loss_fn, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, loss_dict=None, wandb_run=None):
    gaussians = scene.gaussians
    opt = getattr(scene, 'opt', None)
    total_points = gaussians.get_xyz.shape[0]
    static_points = gaussians.get_static_xyz.shape[0] if hasattr(gaussians, 'get_static_xyz') else 0
    dynamic_points = total_points - static_points
    hard_static_conversion = bool(getattr(opt, "enable_hard_static_conversion", False)) if opt is not None else False
    histogram_log_interval = max(1, int(getattr(opt, "histogram_log_interval", 1))) if opt is not None else 1
    log_histograms = (
        iteration % histogram_log_interval == 0
        or iteration == 1
        or iteration in testing_iterations
        or iteration == getattr(opt, "iterations", -1)
    )

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', Lssim.item(), iteration)
        tb_writer.add_scalar('train/ssim', 1.0 - Lssim.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', total_points, iteration)
        if log_histograms:
            tb_writer.add_histogram("scene/opacity_histogram", gaussians.get_opacity, iteration)

        tb_writer.add_scalar('points/total', total_points, iteration)
        tb_writer.add_scalar('points/static', static_points, iteration)
        tb_writer.add_scalar('points/dynamic', dynamic_points, iteration)
        tb_writer.add_scalar('points/hard_static', static_points, iteration)
        tb_writer.add_scalar('points/hard_dynamic', dynamic_points, iteration)

        if hard_static_conversion and hasattr(gaussians, '_staticness_score') and opt is not None and gaussians._staticness_score.numel() > 0:
            conversion_rate = (gaussians._staticness_score > opt.static_conversion_threshold).float().mean().item() * 100
            tb_writer.add_scalar('points/static_conversion_rate', conversion_rate, iteration)

        if hard_static_conversion and hasattr(gaussians, 'differentiable_s') and gaussians.differentiable_s is not None and gaussians.differentiable_s.numel() > 0:
            s = gaussians.differentiable_s.detach()
            tb_writer.add_scalar('gate/scalars/mean_s', s.mean().item(), iteration)
            tb_writer.add_scalar('gate/scalars/median_s', s.median().item(), iteration)
            tb_writer.add_scalar('gate/scalars/min_s', s.min().item(), iteration)
            tb_writer.add_scalar('gate/scalars/max_s', s.max().item(), iteration)
            tb_writer.add_scalar('gate/scalars/percent_s>0.9', (s > 0.9).float().mean().item() * 100, iteration)
            tb_writer.add_scalar('gate/scalars/percent_s<0.1', (s < 0.1).float().mean().item() * 100, iteration)
            if log_histograms:
                tb_writer.add_histogram('gate/hist/s_distribution', s, iteration, bins=50)
            if log_histograms and hasattr(gaussians, 'get_t') and gaussians.get_t.numel() > 0:
                ts = gaussians.get_t.detach()
                tb_writer.add_histogram('gate/ts/timestamps_after_gating', ts, iteration, bins=50)
        if gaussians.get_route_logit.numel() > 0:
            p_dyn = gaussians.get_dynamic_probability.detach().clamp(1e-6, 1.0 - 1e-6)
            route_entropy = (-(p_dyn * torch.log(p_dyn) + (1.0 - p_dyn) * torch.log(1.0 - p_dyn))).mean()
            tb_writer.add_scalar('routing/mean_dynamic_prob', p_dyn.mean().item(), iteration)
            tb_writer.add_scalar('routing/entropy', route_entropy.item(), iteration)
            tb_writer.add_scalar('routing/percent_near_static', (p_dyn < 0.1).float().mean().item() * 100, iteration)
            tb_writer.add_scalar('routing/percent_near_dynamic', (p_dyn > 0.9).float().mean().item() * 100, iteration)
            tb_writer.add_scalar('routing/expected_static_points', (1.0 - p_dyn).sum().item(), iteration)
            tb_writer.add_scalar('routing/expected_dynamic_points', p_dyn.sum().item(), iteration)
            tb_writer.add_scalar('routing/percent_uncertain', ((p_dyn >= 0.1) & (p_dyn <= 0.9)).float().mean().item() * 100, iteration)
            if log_histograms:
                tb_writer.add_histogram('routing/dynamic_probability', p_dyn, iteration, bins=50)

        if getattr(gaussians, "motion_model", "") == "lora" and gaussians.get_motion_lora_coeff.numel() > 0:
            tb_writer.add_scalar('motion_lora/coeff_norm_mean', gaussians.get_motion_lora_coeff.detach().norm(dim=1).mean().item(), iteration)
            if gaussians.get_motion_lora_basis is not None:
                tb_writer.add_scalar('motion_lora/basis_norm_mean', gaussians.get_motion_lora_basis.detach().norm(dim=-1).mean().item(), iteration)
                if log_histograms:
                    tb_writer.add_histogram('motion_lora/basis', gaussians.get_motion_lora_basis.detach(), iteration, bins=50)

        if getattr(gaussians, "motion_scaffold_enable", False) and gaussians.get_motion_scaffold_coeff.numel() > 0:
            tb_writer.add_scalar('motion_scaffold/node_count', gaussians.get_motion_scaffold_coeff.shape[0], iteration)
            tb_writer.add_scalar('motion_scaffold/coeff_norm_mean', gaussians.get_motion_scaffold_coeff.detach().norm(dim=1).mean().item(), iteration)
            if gaussians.get_motion_scaffold_basis is not None:
                tb_writer.add_scalar('motion_scaffold/basis_norm_mean', gaussians.get_motion_scaffold_basis.detach().norm(dim=-1).mean().item(), iteration)
            if gaussians.get_motion_scaffold_attach_w.numel() > 0:
                tb_writer.add_scalar('motion_scaffold/attach_entropy', (-(gaussians.get_motion_scaffold_attach_w.detach().clamp_min(1e-6) * torch.log(gaussians.get_motion_scaffold_attach_w.detach().clamp_min(1e-6))).sum(dim=1)).mean().item(), iteration)

        # Static conversion diagnostics (only for explicit hard-conversion ablations)
        if hard_static_conversion and hasattr(gaussians, "num_static_candidates_last"):
            tb_writer.add_scalar('static_conversion/num_candidates',gaussians.num_static_candidates_last,iteration,)
        if hard_static_conversion and hasattr(gaussians, "num_converted_last"):
            tb_writer.add_scalar('static_conversion/num_converted',gaussians.num_converted_last,iteration,)
            if gaussians.num_static_candidates_last > 0:
                frac = gaussians.num_converted_last / max(1, gaussians.num_static_candidates_last)
                tb_writer.add_scalar('static_conversion/frac_converted',frac,iteration,)

        if loss_dict is not None:
            if "Lrigid" in loss_dict: tb_writer.add_scalar('train_loss_patches/rigid_loss', loss_dict['Lrigid'].item(), iteration)
            if "Ldepth" in loss_dict: tb_writer.add_scalar('train_loss_patches/depth_loss', loss_dict['Ldepth'].item(), iteration)
            if "Ltv" in loss_dict: tb_writer.add_scalar('train_loss_patches/tv_loss', loss_dict['Ltv'].item(), iteration)
            if "Lopa" in loss_dict: tb_writer.add_scalar('train_loss_patches/opa_loss', loss_dict['Lopa'].item(), iteration)
            if "Lptsopa" in loss_dict: tb_writer.add_scalar('train_loss_patches/pts_opa_loss', loss_dict['Lptsopa'].item(), iteration)
            if "Lsmooth" in loss_dict: tb_writer.add_scalar('train_loss_patches/smooth_loss', loss_dict['Lsmooth'].item(), iteration)
            if "Llaplacian" in loss_dict: tb_writer.add_scalar('train_loss_patches/laplacian_loss', loss_dict['Llaplacian'].item(), iteration)
            if "Lsparsity" in loss_dict: tb_writer.add_scalar('train_loss_patches/gate_sparsity_loss', loss_dict['Lsparsity'].item(), iteration)
            if "Lmotion_gate" in loss_dict: tb_writer.add_scalar('train_loss_patches/motion_gate_loss', loss_dict['Lmotion_gate'].item(), iteration)
            if "Lmotion_reg" in loss_dict: tb_writer.add_scalar('train_loss_patches/motion_reg_loss', loss_dict['Lmotion_reg'].item(), iteration)
            if "Ldynamic_roi" in loss_dict: tb_writer.add_scalar('train_loss_patches/dynamic_roi_loss', loss_dict['Ldynamic_roi'].item(), iteration)
            if "Lstatic_exclusion" in loss_dict: tb_writer.add_scalar('train_loss_patches/static_exclusion_loss', loss_dict['Lstatic_exclusion'].item(), iteration)
            if "Ltrack_flow" in loss_dict: tb_writer.add_scalar('train_loss_patches/track_flow_loss', loss_dict['Ltrack_flow'].item(), iteration)
            if "Lscaffold_smooth" in loss_dict: tb_writer.add_scalar('train_loss_patches/scaffold_smooth_loss', loss_dict['Lscaffold_smooth'].item(), iteration)
            if "Lscaffold_reg" in loss_dict: tb_writer.add_scalar('train_loss_patches/scaffold_reg_loss', loss_dict['Lscaffold_reg'].item(), iteration)

        tb_writer.add_scalar('gpu/memory_allocated_MB', torch.cuda.memory_allocated() / 1e6, iteration)
        tb_writer.add_scalar('gpu/memory_reserved_MB', torch.cuda.memory_reserved() / 1e6, iteration)

    wandb_metrics = {
        "train/l1_loss": Ll1.item(),
        "train/ssim_loss": Lssim.item(),
        "train/ssim": 1.0 - Lssim.item(),
        "train/total_loss": loss,
        "train/iter_time_ms": elapsed,
        "gpu/memory_allocated_MB": torch.cuda.memory_allocated() / 1e6,
        "gpu/memory_reserved_MB": torch.cuda.memory_reserved() / 1e6,
    }
    wandb_metrics.update(collect_decomposition_diagnostics(gaussians, opt))
    if log_histograms:
        wandb_metrics["hist/scene_opacity"] = gaussians.get_opacity

    if hard_static_conversion and hasattr(gaussians, '_staticness_score') and opt is not None and gaussians._staticness_score.numel() > 0:
        conversion_rate = (gaussians._staticness_score > opt.static_conversion_threshold).float().mean().item() * 100
        wandb_metrics["points/static_conversion_rate"] = conversion_rate

    if hard_static_conversion and hasattr(gaussians, 'differentiable_s') and gaussians.differentiable_s is not None and gaussians.differentiable_s.numel() > 0:
        s = gaussians.differentiable_s.detach()
        wandb_metrics.update({
            "gate/mean_s": s.mean().item(),
            "gate/median_s": s.median().item(),
            "gate/min_s": s.min().item(),
            "gate/max_s": s.max().item(),
            "gate/percent_s_gt_0_9": (s > 0.9).float().mean().item() * 100,
            "gate/percent_s_lt_0_1": (s < 0.1).float().mean().item() * 100,
        })
        if log_histograms:
            wandb_metrics["hist/gate_s_distribution"] = s
        if log_histograms and hasattr(gaussians, 'get_t') and gaussians.get_t.numel() > 0:
            wandb_metrics["hist/gate_timestamps_after_gating"] = gaussians.get_t.detach()

    if gaussians.get_route_logit.numel() > 0:
        p_dyn = gaussians.get_dynamic_probability.detach().clamp(1e-6, 1.0 - 1e-6)
        route_entropy = (-(p_dyn * torch.log(p_dyn) + (1.0 - p_dyn) * torch.log(1.0 - p_dyn))).mean()
        wandb_metrics.update({
            "routing/mean_dynamic_prob": p_dyn.mean().item(),
            "routing/entropy": route_entropy.item(),
            "routing/percent_near_static": (p_dyn < 0.1).float().mean().item() * 100,
            "routing/percent_near_dynamic": (p_dyn > 0.9).float().mean().item() * 100,
            "routing/expected_static_points": (1.0 - p_dyn).sum().item(),
            "routing/expected_dynamic_points": p_dyn.sum().item(),
            "routing/percent_uncertain": ((p_dyn >= 0.1) & (p_dyn <= 0.9)).float().mean().item() * 100,
        })
        if log_histograms:
            wandb_metrics["hist/routing_dynamic_probability"] = p_dyn

    if getattr(gaussians, "motion_model", "") == "lora" and gaussians.get_motion_lora_coeff.numel() > 0:
        wandb_metrics["motion_lora/coeff_norm_mean"] = gaussians.get_motion_lora_coeff.detach().norm(dim=1).mean().item()
        if gaussians.get_motion_lora_basis is not None:
            wandb_metrics["motion_lora/basis_norm_mean"] = gaussians.get_motion_lora_basis.detach().norm(dim=-1).mean().item()
            if log_histograms:
                wandb_metrics["hist/motion_lora_basis"] = gaussians.get_motion_lora_basis.detach()

    if getattr(gaussians, "motion_scaffold_enable", False) and gaussians.get_motion_scaffold_coeff.numel() > 0:
        wandb_metrics["motion_scaffold/node_count"] = gaussians.get_motion_scaffold_coeff.shape[0]
        wandb_metrics["motion_scaffold/coeff_norm_mean"] = gaussians.get_motion_scaffold_coeff.detach().norm(dim=1).mean().item()
        if gaussians.get_motion_scaffold_basis is not None:
            wandb_metrics["motion_scaffold/basis_norm_mean"] = gaussians.get_motion_scaffold_basis.detach().norm(dim=-1).mean().item()
        if gaussians.get_motion_scaffold_attach_w.numel() > 0:
            attach_w = gaussians.get_motion_scaffold_attach_w.detach().clamp_min(1e-6)
            wandb_metrics["motion_scaffold/attach_entropy"] = (-(attach_w * torch.log(attach_w)).sum(dim=1)).mean().item()

    if hard_static_conversion and hasattr(gaussians, "num_static_candidates_last"):
        wandb_metrics["static_conversion/num_candidates"] = gaussians.num_static_candidates_last
    if hard_static_conversion and hasattr(gaussians, "num_converted_last"):
        wandb_metrics["static_conversion/num_converted"] = gaussians.num_converted_last
        if gaussians.num_static_candidates_last > 0:
            wandb_metrics["static_conversion/frac_converted"] = gaussians.num_converted_last / max(1, gaussians.num_static_candidates_last)

    if loss_dict is not None:
        loss_metric_names = {
            "Lrigid": "train/rigid_loss",
            "Ldepth": "train/depth_loss",
            "Ltv": "train/tv_loss",
            "Lopa": "train/opa_loss",
            "Lptsopa": "train/pts_opa_loss",
            "Lsmooth": "train/smooth_loss",
            "Llaplacian": "train/laplacian_loss",
            "Lsparsity": "train/gate_sparsity_loss",
            "Lmotion_gate": "train/motion_gate_loss",
            "Lmotion_reg": "train/motion_reg_loss",
            "Ldynamic_roi": "train/dynamic_roi_loss",
            "Lstatic_exclusion": "train/static_exclusion_loss",
            "Ltrack_flow": "train/track_flow_loss",
            "Lscaffold_smooth": "train/scaffold_smooth_loss",
            "Lscaffold_reg": "train/scaffold_reg_loss",
        }
        for key, metric_name in loss_metric_names.items():
            if key in loss_dict:
                wandb_metrics[metric_name] = loss_dict[key]

    log_wandb_metrics(wandb_run, wandb_metrics, iteration)

    # simple evaluation on test set when requested
    if iteration in testing_iterations:
        (pipe, background) = renderArgs
        test_cams = scene.getTestCameras()
        if len(test_cams) > 0:
            psnrs = []
            ssims = []
            dyn_psnrs = []
            static_region_psnrs = []
            static_ghost_scores = []
            dynamic_edge_scores = []
            track_flow_errors = []
            with torch.no_grad():
                for data in test_cams:
                    # Unpack in case dataset returns (gt_image, cam)
                    if isinstance(data, (list, tuple)) and len(data) == 2:
                        gt_image, cam = data
                    else:
                        gt_image, cam = None, data

                    cam = cam.cuda()
                    render_out = renderFunc(cam, scene.gaussians, pipe, background)
                    pred = render_out["render"]

                    if gt_image is not None: gt = gt_image.cuda()
                    elif hasattr(cam, "original_image"): gt = cam.original_image.cuda()
                    elif hasattr(cam, "gt_image"): gt = cam.gt_image.cuda()
                    else:
                        raise ValueError("No ground truth image found for test camera.")

                    psnrs.append(psnr(pred, gt).mean().item())
                    ssims.append(ssim(pred, gt).mean().item())

                    prior_cache = getattr(scene, "motion_prior_cache", None)
                    if prior_cache is not None:
                        dyn_mask = prior_cache.get_dynamic_mask(
                            cam,
                            target_hw=gt.shape[-2:],
                            gt_image=gt,
                            pred_image=pred,
                            allow_residual=False,
                        )
                        if dyn_mask is not None and dyn_mask.sum() > 1:
                            dyn_psnr = masked_psnr(pred, gt, dyn_mask)
                            if dyn_psnr is not None:
                                dyn_psnrs.append(dyn_psnr.item())
                            static_mask = (1.0 - dyn_mask).clamp(0.0, 1.0)
                            if static_mask.sum() > 1:
                                static_psnr = masked_psnr(pred, gt, static_mask)
                                if static_psnr is not None:
                                    static_region_psnrs.append(static_psnr.item())
                            static_ghost_scores.append((render_out["render_3d"].abs().mean(dim=0, keepdim=True) * dyn_mask).sum().div(dyn_mask.sum().clamp_min(1e-6)).item())
                            dynamic_edge_scores.append((edge_magnitude(pred) * dyn_mask).sum().div(dyn_mask.sum().clamp_min(1e-6)).item())

                        track_flow, track_flow_mask = prior_cache.get_track_flow(cam, gt.shape[-2:])
                        if track_flow is not None:
                            if track_flow_mask is None:
                                track_flow_mask = dyn_mask
                            flow_loss = compute_flow_loss(render_out.get("flow", None), track_flow, track_flow_mask)
                            if flow_loss is not None:
                                track_flow_errors.append(flow_loss.item())

                    # for cam in test_cams:
                    # cam = cam.cuda()
                    # render_out = renderFunc(cam, scene.gaussians, pipe, background)
                    # pred = render_out["render"]
                    # gt = cam.original_image.cuda() if hasattr(cam, "original_image") else cam.gt_image.cuda()
                    # psnrs.append(psnr(pred, gt).mean().item())
            if psnrs:
                test_psnr = float(np.mean(psnrs))
                eval_metrics = {"test/psnr": test_psnr}
                if ssims:
                    eval_metrics["test/ssim"] = float(np.mean(ssims))
                if dyn_psnrs:
                    eval_metrics["test/dynamic_mask_psnr"] = float(np.mean(dyn_psnrs))
                if static_region_psnrs:
                    eval_metrics["test/static_region_psnr"] = float(np.mean(static_region_psnrs))
                if static_ghost_scores:
                    eval_metrics["test/static_ghost_score"] = float(np.mean(static_ghost_scores))
                if dynamic_edge_scores:
                    eval_metrics["test/dynamic_edge_magnitude"] = float(np.mean(dynamic_edge_scores))
                if track_flow_errors:
                    eval_metrics["test/track_flow_l1"] = float(np.mean(track_flow_errors))
                eval_metrics.update(collect_decomposition_diagnostics(gaussians, opt))
                if tb_writer:
                    for metric_name, metric_value in eval_metrics.items():
                        tb_writer.add_scalar(metric_name, metric_value, iteration)
                log_wandb_metrics(wandb_run, eval_metrics, iteration)
                return eval_metrics
    return None


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", type=str)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 6_000, 9_000, 12_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 6_000, 9_000, 10_000, 12_000, 14_000, 15_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--val", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="adags")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", nargs="+", default=None)
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--wandb_resume", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--method_family", type=str, default=None)
    parser.add_argument("--budget_label", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])

    cfg = OmegaConf.load(args.config)

    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys(): recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])

    for k in cfg.keys(): recursive_merge(k, cfg)

    if args.wandb_mode == "disabled":
        args.use_wandb = False

    enforce_train_iteration_guard(args)
    args.save_iterations = normalize_iteration_schedule(args.save_iterations, args.iterations)
    args.test_iterations = normalize_iteration_schedule(args.test_iterations, args.iterations)
    ensure_model_path(args)

    if args.exhaust_test:
        args.test_iterations = normalize_iteration_schedule(
            args.test_iterations + [i for i in range(0, args.iterations, 500)],
            args.iterations,
        )

    setup_seed(args.seed)
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    wandb_run = init_wandb(args)

    summary_updates = None
    try:
        if not args.val:
            summary_updates = training(lp.extract(args), op.extract(args), pp.extract(args),
                                       args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from,
                                       args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio,
                                       args.rot_4d, args.force_sh_3d, args.batch_size, wandb_run)
        else:
            summary_updates = validation(lp.extract(args), op.extract(args), pp.extract(args),
                                         args.start_checkpoint, args.gaussian_dim, args.time_duration,
                                         args.rot_4d, args.force_sh_3d, args.num_pts, args.num_pts_ratio, wandb_run)
            summary_updates["model_path"] = args.model_path
    finally:
        finish_wandb_run(wandb_run, summary_updates)

    print("\nComplete.")
