#!/usr/bin/env python3
"""Paired per-scene W&B analysis for ADAGS/N3V experiments.

This script treats W&B's built-in parameter importance as a secondary view.
The primary analysis unit is a scene-method pair, compared against a baseline
run from the same scene.
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


SCENES = [
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_salmon_1",
    "flame_steak",
    "sear_steak",
]

SUMMARY_METRICS = [
    "test/psnr",
    "test/ssim",
    "test/lpips",
    "test/dynamic_mask_psnr",
    "test/static_ghost_score",
    "test/dynamic_edge_magnitude",
    "test/track_flow_l1",
    "points/total",
    "points/static",
    "points/dynamic",
    "points/hard_static",
    "points/hard_dynamic",
    "routing/mean_dynamic_prob",
    "routing/entropy",
    "routing/expected_static_points",
    "routing/expected_dynamic_points",
    "routing/percent_near_static",
    "routing/percent_near_dynamic",
    "routing/percent_uncertain",
    "motion_lora/coeff_norm_mean",
    "motion_lora/basis_norm_mean",
    "motion_scaffold/node_count",
    "motion_scaffold/coeff_norm_mean",
    "motion_scaffold/basis_norm_mean",
    "motion_scaffold/attach_entropy",
    "static_conversion/num_candidates",
    "static_conversion/num_converted",
    "static_conversion/frac_converted",
]

DEFAULT_HISTORY_KEYS = [
    "train/psnr",
    "train/l1_loss",
    "train/ssim",
    "train/total_loss",
    "test/psnr",
    "test/ssim",
    "points/total",
    "points/hard_static",
    "routing/entropy",
    "routing/expected_static_points",
    "routing/percent_uncertain",
    "motion_lora/coeff_norm_mean",
    "motion_scaffold/coeff_norm_mean",
    "motion_scaffold/attach_entropy",
]

CONFIG_IGNORE_KEYS = {
    "config",
    "source_path",
    "model_path",
    "images",
    "loaded_pth",
    "from3dgs",
    "start_checkpoint",
    "test_iterations",
    "save_iterations",
    "quiet",
    "debug_from",
    "detect_anomaly",
    "use_wandb",
    "val",
    "exhaust_test",
    "wandb_project",
    "wandb_entity",
    "wandb_run_name",
    "wandb_group",
    "wandb_tags",
    "wandb_mode",
    "wandb_resume",
}

CONFIG_IGNORE_PREFIXES = ("runtime_", "wandb_", "_")
METADATA_SENTINEL = "__metadata__"

FACTOR_PRIORITY = [
    "method_family",
    "budget_label",
    "iterations",
    "gaussian_dim",
    "num_pts",
    "num_pts_ratio",
    "batch_size",
    "rot_4d",
    "force_sh_3d",
    "resolution",
    "dataloader",
    "eval_shfs_4d",
    "position_lr_init",
    "position_t_lr_init",
    "position_lr_final",
    "feature_lr",
    "opacity_lr",
    "scaling_lr",
    "rotation_lr",
    "percent_dense",
    "lambda_dssim",
    "densification_interval",
    "opacity_reset_interval",
    "densify_from_iter",
    "densify_until_iter",
    "densify_grad_threshold",
    "densify_grad_t_threshold",
    "densify_until_num_points",
    "enable_hard_static_conversion",
    "static_conversion_threshold",
    "lambda_gate_sparsity",
    "lambda_sparsity",
    "lambda_motion_gate",
    "motion_gate_quantile",
    "enable_soft_routing",
    "route_logit_init",
    "route_lr",
    "motion_model",
    "motion_poly_order",
    "motion_lr_init",
    "motion_reg_lambda",
    "motion_lora_rank",
    "motion_lora_anchors",
    "motion_lora_init_scale",
    "motion_lora_coeff_lr",
    "motion_lora_basis_lr",
    "dynamic_mask_from_residual",
    "dynamic_mask_residual_quantile",
    "dynamic_mask_dilate",
    "lambda_dynamic_roi",
    "lambda_static_exclusion",
    "lambda_track_flow",
    "lambda_scaffold_smooth",
    "lambda_scaffold_reg",
    "motion_scaffold_enable",
    "motion_scaffold_count",
    "motion_scaffold_rank",
    "motion_scaffold_anchors",
    "motion_scaffold_knn",
    "motion_scaffold_init_scale",
    "motion_scaffold_weight_temp",
    "motion_scaffold_coeff_lr",
    "motion_scaffold_basis_lr",
    "enable_rendered_flow",
    "enable_motion_aware_densify",
    "motion_aware_densify_boost",
    "blur_until_iter",
    "blur_start_sigma",
]


def safe_float(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def format_float(value, digits=4):
    value = safe_float(value)
    if value is None:
        return ""
    return ("%%.%df" % digits) % value


def mean_or_none(values):
    clean = [safe_float(v) for v in values]
    clean = [v for v in clean if v is not None]
    return mean(clean) if clean else None


def bootstrap_ci(values, samples, seed):
    clean = [safe_float(v) for v in values]
    clean = [v for v in clean if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1 or samples <= 0:
        return clean[0], clean[0]
    rng = random.Random(seed)
    estimates = []
    n = len(clean)
    for _ in range(samples):
        estimates.append(mean(clean[rng.randrange(n)] for _ in range(n)))
    estimates.sort()
    low_idx = int(0.025 * (samples - 1))
    high_idx = int(0.975 * (samples - 1))
    return estimates[low_idx], estimates[high_idx]


def parse_jsonish(value):
    if value is None:
        return None
    if isinstance(value, (bool, int, float, list, tuple, dict)):
        return value
    text = str(value).strip()
    if text == "":
        return None
    try:
        return json.loads(text)
    except Exception:
        return value


def value_key(value):
    value = parse_jsonish(value)
    if isinstance(value, float):
        return format_float(value, 8)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def strip_timestamp_prefix(run_name):
    match = re.match(r"^\d{8}_\d{6}_(.+)$", run_name or "")
    return match.group(1) if match else (run_name or "")


def infer_scene(run_name, group, tags, summary, config, scenes):
    if group in scenes:
        return group

    for key in ("metadata/scene", "scene"):
        value = summary.get(key)
        if value in scenes:
            return value

    source_path = summary.get("metadata/source_path") or config.get("source_path")
    if source_path and source_path != METADATA_SENTINEL:
        scene = os.path.basename(os.path.normpath(str(source_path)))
        if scene in scenes:
            return scene

    tag_set = set(tags or [])
    for scene in scenes:
        if scene in tag_set:
            return scene

    remainder = strip_timestamp_prefix(run_name)
    matches = [scene for scene in scenes if remainder == scene or remainder.startswith(scene + "_")]
    return max(matches, key=len) if matches else None


def infer_method(run_name, scene, tags, summary, config):
    for key in ("metadata/config_name", "config_name", "method"):
        value = summary.get(key)
        if value and value != METADATA_SENTINEL:
            return str(value)

    config_path = summary.get("metadata/config_path") or config.get("config")
    if config_path and config_path != METADATA_SENTINEL:
        base = os.path.basename(str(config_path))
        if base.endswith(".yaml"):
            return base[:-5]
        if base:
            return os.path.splitext(base)[0]

    remainder = strip_timestamp_prefix(run_name)
    if scene and remainder.startswith(scene + "_"):
        return remainder[len(scene) + 1:]

    ignored = set(["train", "eval", "validation", "n3v", scene])
    for tag in tags or []:
        if tag and tag not in ignored and not str(tag).startswith("group:"):
            return str(tag)

    return remainder or run_name


def is_noisy_config_key(key):
    if key in CONFIG_IGNORE_KEYS:
        return True
    return any(str(key).startswith(prefix) for prefix in CONFIG_IGNORE_PREFIXES)


def clean_config(config):
    clean = {}
    noisy = {}
    for key, value in (config or {}).items():
        if value == METADATA_SENTINEL or is_noisy_config_key(key):
            noisy[key] = value
            continue
        clean[key] = value
    return clean, noisy


def canonical_created_at(row):
    return row.get("created_at") or ""


def canonical_metric(row, metric):
    value = safe_float(row.get(metric))
    return value if value is not None else float("-inf")


def choose_canonical(rows, dedupe):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("scene") and row.get("method"):
            grouped[(row["scene"], row["method"])].append(row)

    canonical = {}
    duplicates = {}
    for key, group in grouped.items():
        if len(group) > 1:
            duplicates[key] = group
        if dedupe == "best-psnr":
            selected = sorted(group, key=lambda r: (canonical_metric(r, "test/psnr"), canonical_created_at(r), r["run_id"]))[-1]
        elif dedupe == "first":
            selected = sorted(group, key=lambda r: (canonical_created_at(r), r["run_id"]))[0]
        else:
            selected = sorted(group, key=lambda r: (canonical_created_at(r), r["run_id"]))[-1]
        canonical[key] = selected
    return canonical, duplicates


def run_url(entity, project, run_id):
    return "https://wandb.ai/%s/%s/runs/%s" % (entity, project, run_id)


def load_local_stats(stats_root, run_id):
    if not stats_root:
        return {}
    patterns = [
        os.path.join(stats_root, run_id, "test", "ours_*", "stats", "validation.json"),
        os.path.join(stats_root, "*", run_id, "test", "ours_*", "stats", "validation.json"),
        os.path.join(stats_root, "*", "*", run_id, "test", "ours_*", "stats", "validation.json"),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    if not matches:
        return {}
    stats_path = sorted(matches)[-1]
    try:
        with open(stats_path, "r") as handle:
            stats = json.load(handle)
    except Exception:
        return {}
    return {
        "test/psnr": stats.get("psnr"),
        "test/ssim": stats.get("ssim"),
        "test/lpips": stats.get("lpips"),
        "points/total": stats.get("num_GS"),
        "points/hard_static": stats.get("static"),
        "local_stats_path": stats_path,
    }


def collect_runs(args):
    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb is not installed. Install the W&B SDK or run in the W&B analysis environment.")

    api = wandb.Api(timeout=args.timeout)
    path = "%s/%s" % (args.entity, args.project)
    filters = {"state": args.state} if args.state else None
    per_page = min(args.limit or 1000, 1000)
    runs = api.runs(path, filters=filters, order=args.order, per_page=per_page, include_sweeps=False)

    rows = []
    for index, run in enumerate(runs):
        if args.limit and index >= args.limit:
            break

        summary = dict(getattr(run, "summary_metrics", None) or getattr(run, "summary", {}) or {})
        config = dict(getattr(run, "config", {}) or {})
        tags = list(getattr(run, "tags", []) or [])
        local_stats = load_local_stats(args.local_stats_root, run.id)

        merged_summary = dict(summary)
        for key, value in local_stats.items():
            if key != "local_stats_path" and merged_summary.get(key) is None:
                merged_summary[key] = value

        clean_cfg, noisy_cfg = clean_config(config)
        scene = infer_scene(run.name, run.group, tags, summary, config, args.scenes)
        method = infer_method(run.name, scene, tags, summary, config)

        row = {
            "run_id": run.id,
            "run_name": run.name,
            "url": run_url(args.entity, args.project, run.id),
            "state": run.state,
            "created_at": getattr(run, "created_at", ""),
            "group": run.group,
            "job_type": getattr(run, "job_type", None),
            "scene": scene,
            "method": method,
            "tags": ",".join(tags),
            "last_history_step": getattr(run, "lastHistoryStep", None),
            "noisy_config_keys": ",".join(sorted(noisy_cfg.keys())),
            "noisy_config_non_sentinel_keys": ",".join(sorted(
                key for key, value in noisy_cfg.items() if value != METADATA_SENTINEL
            )),
            "local_stats_path": local_stats.get("local_stats_path", ""),
        }

        for metric in SUMMARY_METRICS:
            row[metric] = merged_summary.get(metric)

        if row.get("points/hard_static") is not None and row.get("points/hard_dynamic") is None:
            total = safe_float(row.get("points/total"))
            static = safe_float(row.get("points/hard_static"))
            if total is not None and static is not None:
                row["points/hard_dynamic"] = total - static

        for key, value in clean_cfg.items():
            row["config/" + key] = value

        rows.append(row)

    return rows


def metric_at_or_before(series, target_step):
    selected = None
    for step, value in series:
        if step is not None and step <= target_step and value is not None:
            selected = value
    return selected


def auc(series):
    clean = [(s, v) for s, v in series if s is not None and v is not None]
    if len(clean) < 2:
        return None
    total = 0.0
    span = clean[-1][0] - clean[0][0]
    if span <= 0:
        return None
    for (s0, v0), (s1, v1) in zip(clean[:-1], clean[1:]):
        total += 0.5 * (v0 + v1) * (s1 - s0)
    return total / span


def tail_slope(series):
    clean = [(s, v) for s, v in series if s is not None and v is not None]
    if len(clean) < 4:
        return None
    start = int(len(clean) * 0.8)
    tail = clean[start:] if len(clean[start:]) >= 2 else clean[-2:]
    xs = [float(s) for s, _ in tail]
    ys = [float(v) for _, v in tail]
    mx = mean(xs)
    my = mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def summarize_series(series):
    clean = [(s, v) for s, v in series if v is not None]
    if not clean:
        return {}
    values = [v for _, v in clean]
    out = {
        "first": values[0],
        "final": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "delta": values[-1] - values[0],
        "auc": auc(clean),
        "tail_slope": tail_slope(clean),
    }
    for target in (3000, 6000, 9000, 12000, 15000):
        out["at_%d" % target] = metric_at_or_before(clean, target)
    return out


def fetch_metric_series(run, key, page_size):
    series = []
    try:
        for item in run.scan_history(keys=["_step", key], page_size=page_size):
            value = safe_float(item.get(key))
            step = safe_float(item.get("_step"))
            if value is not None:
                series.append((int(step) if step is not None else None, value))
    except Exception:
        return []
    series.sort(key=lambda pair: -1 if pair[0] is None else pair[0])
    return series


def collect_history_features(args, canonical_rows):
    if args.skip_history:
        return []
    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb is not installed. Install the W&B SDK or use --skip-history.")

    api = wandb.Api(timeout=args.timeout)
    path = "%s/%s" % (args.entity, args.project)
    keys = args.history_keys or DEFAULT_HISTORY_KEYS
    rows = []

    for index, run_row in enumerate(canonical_rows):
        if args.history_limit and index >= args.history_limit:
            break
        run = api.run("%s/%s" % (path, run_row["run_id"]))
        out = {
            "run_id": run_row["run_id"],
            "run_name": run_row["run_name"],
            "scene": run_row["scene"],
            "method": run_row["method"],
        }
        for key in keys:
            series = fetch_metric_series(run, key, args.history_page_size)
            summary = summarize_series(series)
            safe_key = key.replace("/", "__")
            out[safe_key + "__count"] = len(series)
            for feature, value in summary.items():
                out[safe_key + "__" + feature] = value
        rows.append(out)
    return rows


def compute_paired_deltas(canonical, baseline_method):
    rows = []
    baselines = {}
    for (scene, method), row in canonical.items():
        if method == baseline_method:
            baselines[scene] = row

    for (scene, method), row in canonical.items():
        baseline = baselines.get(scene)
        out = {
            "scene": scene,
            "method": method,
            "run_id": row["run_id"],
            "run_name": row["run_name"],
            "baseline_method": baseline_method,
            "baseline_run_id": baseline["run_id"] if baseline else "",
            "baseline_run_name": baseline["run_name"] if baseline else "",
            "has_baseline": bool(baseline),
        }
        for metric in ("test/psnr", "test/ssim", "test/lpips", "points/total", "points/hard_static", "points/hard_dynamic"):
            value = safe_float(row.get(metric))
            base_value = safe_float(baseline.get(metric)) if baseline else None
            out[metric] = value
            out["baseline/" + metric] = base_value
            if value is not None and base_value is not None:
                out["delta/" + metric] = value - base_value
            else:
                out["delta/" + metric] = None
        lpips_delta = safe_float(out.get("delta/test/lpips"))
        out["delta/lpips_quality"] = -lpips_delta if lpips_delta is not None else None
        psnr_delta = safe_float(out.get("delta/test/psnr"))
        points = safe_float(row.get("points/total"))
        if psnr_delta is not None and points and points > 0:
            out["delta_psnr_per_million_points"] = psnr_delta / (points / 1000000.0)
        else:
            out["delta_psnr_per_million_points"] = None
        rows.append(out)

    rows.sort(key=lambda r: (r["method"], r["scene"]))
    return rows


def summarize_methods(delta_rows, bootstrap_samples, bootstrap_seed):
    grouped = defaultdict(list)
    for row in delta_rows:
        if row.get("has_baseline"):
            grouped[row["method"]].append(row)

    out_rows = []
    for method, group in grouped.items():
        psnr_values = [r.get("delta/test/psnr") for r in group]
        ssim_values = [r.get("delta/test/ssim") for r in group]
        lpips_quality_values = [r.get("delta/lpips_quality") for r in group]
        psnr_low, psnr_high = bootstrap_ci(psnr_values, bootstrap_samples, bootstrap_seed)
        ssim_low, ssim_high = bootstrap_ci(ssim_values, bootstrap_samples, bootstrap_seed)
        lpips_low, lpips_high = bootstrap_ci(lpips_quality_values, bootstrap_samples, bootstrap_seed)
        out = {
            "method": method,
            "scene_count": len(set(r["scene"] for r in group)),
            "mean_delta_psnr": mean_or_none(psnr_values),
            "ci95_delta_psnr_low": psnr_low,
            "ci95_delta_psnr_high": psnr_high,
            "mean_delta_ssim": mean_or_none(ssim_values),
            "ci95_delta_ssim_low": ssim_low,
            "ci95_delta_ssim_high": ssim_high,
            "mean_delta_lpips": mean_or_none(r.get("delta/test/lpips") for r in group),
            "mean_delta_lpips_quality": mean_or_none(lpips_quality_values),
            "ci95_delta_lpips_quality_low": lpips_low,
            "ci95_delta_lpips_quality_high": lpips_high,
            "mean_test_psnr": mean_or_none(r.get("test/psnr") for r in group),
            "mean_test_ssim": mean_or_none(r.get("test/ssim") for r in group),
            "mean_test_lpips": mean_or_none(r.get("test/lpips") for r in group),
            "mean_points_total": mean_or_none(r.get("points/total") for r in group),
            "mean_delta_psnr_per_million_points": mean_or_none(r.get("delta_psnr_per_million_points") for r in group),
        }
        out_rows.append(out)
    out_rows.sort(key=lambda r: safe_float(r.get("mean_delta_psnr")) if safe_float(r.get("mean_delta_psnr")) is not None else -999, reverse=True)
    return out_rows


def is_scaffold_row(row):
    method = str(row.get("method") or "")
    enabled = str(row.get("config/motion_scaffold_enable") or "").lower()
    return "scaffold" in method or enabled == "true"


def compute_capacity_pairs(canonical, max_ratio, baseline_method):
    by_scene = defaultdict(list)
    for (scene, method), row in canonical.items():
        if scene:
            by_scene[scene].append(row)

    all_pairs = []
    nearest_pairs = []
    for scene, rows in by_scene.items():
        scaffold_rows = [row for row in rows if is_scaffold_row(row)]
        reference_rows = [
            row for row in rows
            if not is_scaffold_row(row) and row.get("method") != baseline_method
        ]
        if not reference_rows:
            continue

        for scaffold in scaffold_rows:
            scaffold_points = safe_float(scaffold.get("points/total"))
            scaffold_psnr = safe_float(scaffold.get("test/psnr"))
            scene_pairs = []
            for reference in reference_rows:
                reference_points = safe_float(reference.get("points/total"))
                reference_psnr = safe_float(reference.get("test/psnr"))
                if not scaffold_points or not reference_points:
                    point_ratio = None
                    point_log_distance = None
                    within_ratio = False
                else:
                    point_ratio = max(scaffold_points, reference_points) / min(scaffold_points, reference_points)
                    point_log_distance = abs(math.log(point_ratio))
                    within_ratio = point_ratio <= max_ratio
                row = {
                    "scene": scene,
                    "scaffold_method": scaffold.get("method"),
                    "reference_method": reference.get("method"),
                    "scaffold_run_id": scaffold.get("run_id"),
                    "reference_run_id": reference.get("run_id"),
                    "scaffold_points": scaffold_points,
                    "reference_points": reference_points,
                    "point_ratio": point_ratio,
                    "within_capacity_ratio": within_ratio,
                    "scaffold_psnr": scaffold_psnr,
                    "reference_psnr": reference_psnr,
                    "scaffold_minus_reference_psnr": (
                        scaffold_psnr - reference_psnr
                        if scaffold_psnr is not None and reference_psnr is not None else None
                    ),
                    "scaffold_ssim": safe_float(scaffold.get("test/ssim")),
                    "reference_ssim": safe_float(reference.get("test/ssim")),
                    "scaffold_minus_reference_ssim": None,
                    "scaffold_lpips": safe_float(scaffold.get("test/lpips")),
                    "reference_lpips": safe_float(reference.get("test/lpips")),
                    "scaffold_minus_reference_lpips": None,
                    "_point_log_distance": point_log_distance,
                }
                if row["scaffold_ssim"] is not None and row["reference_ssim"] is not None:
                    row["scaffold_minus_reference_ssim"] = row["scaffold_ssim"] - row["reference_ssim"]
                if row["scaffold_lpips"] is not None and row["reference_lpips"] is not None:
                    row["scaffold_minus_reference_lpips"] = row["scaffold_lpips"] - row["reference_lpips"]
                    row["scaffold_minus_reference_lpips_quality"] = -row["scaffold_minus_reference_lpips"]
                else:
                    row["scaffold_minus_reference_lpips_quality"] = None
                all_pairs.append(row)
                scene_pairs.append(row)

            valid_pairs = [row for row in scene_pairs if row["_point_log_distance"] is not None]
            if valid_pairs:
                nearest_pairs.append(sorted(valid_pairs, key=lambda r: (r["_point_log_distance"], r["reference_method"]))[0])

    for row in all_pairs + nearest_pairs:
        row.pop("_point_log_distance", None)
    return all_pairs, nearest_pairs


def summarize_capacity_pairs(pairs, nearest_only=False):
    grouped = defaultdict(list)
    for row in pairs:
        grouped[(row["scaffold_method"], row["reference_method"])].append(row)

    out_rows = []
    for (scaffold_method, reference_method), group in grouped.items():
        out_rows.append({
            "scaffold_method": scaffold_method,
            "reference_method": reference_method,
            "scene_count": len(set(r["scene"] for r in group)),
            "mean_point_ratio": mean_or_none(r.get("point_ratio") for r in group),
            "mean_scaffold_minus_reference_psnr": mean_or_none(r.get("scaffold_minus_reference_psnr") for r in group),
            "mean_scaffold_minus_reference_ssim": mean_or_none(r.get("scaffold_minus_reference_ssim") for r in group),
            "mean_scaffold_minus_reference_lpips_quality": mean_or_none(r.get("scaffold_minus_reference_lpips_quality") for r in group),
            "within_capacity_pair_count": sum(1 for r in group if str(r.get("within_capacity_ratio")).lower() == "true" or r.get("within_capacity_ratio") is True),
            "comparison": "nearest_capacity" if nearest_only else "all_pairs",
        })
    out_rows.sort(key=lambda r: (
        safe_float(r.get("mean_scaffold_minus_reference_psnr")) if safe_float(r.get("mean_scaffold_minus_reference_psnr")) is not None else -999,
        -safe_float(r.get("mean_point_ratio")) if safe_float(r.get("mean_point_ratio")) is not None else -999,
    ), reverse=True)
    return out_rows


def summarize_capacity_by_scaffold(pairs, label):
    grouped = defaultdict(list)
    for row in pairs:
        grouped[row["scaffold_method"]].append(row)

    out_rows = []
    for scaffold_method, group in grouped.items():
        out_rows.append({
            "scaffold_method": scaffold_method,
            "comparison": label,
            "scene_count": len(set(r["scene"] for r in group)),
            "pair_count": len(group),
            "mean_point_ratio": mean_or_none(r.get("point_ratio") for r in group),
            "mean_scaffold_minus_reference_psnr": mean_or_none(r.get("scaffold_minus_reference_psnr") for r in group),
            "mean_scaffold_minus_reference_ssim": mean_or_none(r.get("scaffold_minus_reference_ssim") for r in group),
            "mean_scaffold_minus_reference_lpips_quality": mean_or_none(r.get("scaffold_minus_reference_lpips_quality") for r in group),
            "reference_methods": ",".join(sorted(set(r["reference_method"] for r in group))),
        })
    out_rows.sort(
        key=lambda r: safe_float(r.get("mean_scaffold_minus_reference_psnr"))
        if safe_float(r.get("mean_scaffold_minus_reference_psnr")) is not None else -999,
        reverse=True,
    )
    return out_rows


def configured_budget_points(row):
    value = safe_float(row.get("config/densify_until_num_points"))
    if value is not None and value >= 0:
        return int(round(value))

    method = str(row.get("method") or "")
    match = re.search(r"_(\d+)(k|m)$", method)
    if not match:
        return None
    scale = 1000 if match.group(2) == "k" else 1000000
    return int(match.group(1)) * scale


def add_metric_delta(row, left, right, metric, out_name=None, invert=False):
    left_value = safe_float(left.get(metric))
    right_value = safe_float(right.get(metric))
    if out_name is None:
        out_name = metric.replace("/", "_")
    row["scaffold_" + out_name] = left_value
    row["reference_" + out_name] = right_value
    if left_value is not None and right_value is not None:
        delta = left_value - right_value
        row["scaffold_minus_reference_" + out_name] = -delta if invert else delta
    else:
        row["scaffold_minus_reference_" + out_name] = None


def compute_fixed_budget_pairs(canonical, baseline_method):
    by_scene_budget = defaultdict(list)
    for (scene, method), row in canonical.items():
        budget = configured_budget_points(row)
        if scene and budget is not None:
            by_scene_budget[(scene, budget)].append(row)

    pairs = []
    for (scene, budget), rows in by_scene_budget.items():
        scaffold_rows = [row for row in rows if is_scaffold_row(row)]
        reference_rows = [
            row for row in rows
            if not is_scaffold_row(row) and row.get("method") != baseline_method
        ]
        for scaffold in scaffold_rows:
            scaffold_points = safe_float(scaffold.get("points/total"))
            for reference in reference_rows:
                reference_points = safe_float(reference.get("points/total"))
                if scaffold_points and reference_points:
                    point_ratio = max(scaffold_points, reference_points) / min(scaffold_points, reference_points)
                else:
                    point_ratio = None

                row = {
                    "scene": scene,
                    "budget_points": budget,
                    "scaffold_method": scaffold.get("method"),
                    "reference_method": reference.get("method"),
                    "scaffold_run_id": scaffold.get("run_id"),
                    "reference_run_id": reference.get("run_id"),
                    "scaffold_points": scaffold_points,
                    "reference_points": reference_points,
                    "point_ratio": point_ratio,
                }
                add_metric_delta(row, scaffold, reference, "test/psnr", "psnr")
                add_metric_delta(row, scaffold, reference, "test/ssim", "ssim")
                add_metric_delta(row, scaffold, reference, "test/lpips", "lpips_quality", invert=True)
                add_metric_delta(row, scaffold, reference, "test/dynamic_mask_psnr", "dynamic_mask_psnr")
                add_metric_delta(row, scaffold, reference, "test/static_ghost_score", "static_ghost_score", invert=True)
                add_metric_delta(row, scaffold, reference, "routing/entropy", "routing_entropy")
                add_metric_delta(row, scaffold, reference, "routing/expected_static_points", "routing_expected_static_points")
                add_metric_delta(row, scaffold, reference, "routing/percent_uncertain", "routing_percent_uncertain")
                pairs.append(row)

    pairs.sort(key=lambda r: (r["budget_points"], r["scaffold_method"], r["reference_method"], r["scene"]))
    return pairs


def summarize_fixed_budget_pairs(pairs):
    grouped = defaultdict(list)
    for row in pairs:
        grouped[(row["budget_points"], row["scaffold_method"], row["reference_method"])].append(row)

    out_rows = []
    for (budget, scaffold_method, reference_method), group in grouped.items():
        out_rows.append({
            "budget_points": budget,
            "scaffold_method": scaffold_method,
            "reference_method": reference_method,
            "scene_count": len(set(r["scene"] for r in group)),
            "pair_count": len(group),
            "mean_point_ratio": mean_or_none(r.get("point_ratio") for r in group),
            "mean_scaffold_minus_reference_psnr": mean_or_none(r.get("scaffold_minus_reference_psnr") for r in group),
            "mean_scaffold_minus_reference_ssim": mean_or_none(r.get("scaffold_minus_reference_ssim") for r in group),
            "mean_scaffold_minus_reference_lpips_quality": mean_or_none(r.get("scaffold_minus_reference_lpips_quality") for r in group),
            "mean_scaffold_minus_reference_dynamic_mask_psnr": mean_or_none(r.get("scaffold_minus_reference_dynamic_mask_psnr") for r in group),
            "mean_scaffold_minus_reference_static_ghost_score_quality": mean_or_none(r.get("scaffold_minus_reference_static_ghost_score") for r in group),
            "mean_scaffold_minus_reference_routing_entropy": mean_or_none(r.get("scaffold_minus_reference_routing_entropy") for r in group),
            "mean_scaffold_minus_reference_routing_expected_static_points": mean_or_none(r.get("scaffold_minus_reference_routing_expected_static_points") for r in group),
            "mean_scaffold_minus_reference_routing_percent_uncertain": mean_or_none(r.get("scaffold_minus_reference_routing_percent_uncertain") for r in group),
        })
    out_rows.sort(key=lambda r: (
        r["budget_points"],
        safe_float(r.get("mean_scaffold_minus_reference_psnr")) if safe_float(r.get("mean_scaffold_minus_reference_psnr")) is not None else -999,
    ), reverse=True)
    return out_rows


def compute_factor_effects(run_rows, delta_rows, baseline_method, include_baseline):
    delta_by_run = {row["run_id"]: row for row in delta_rows if row.get("has_baseline")}
    factors = sorted(
        [key for key in collect_fieldnames(run_rows) if key.startswith("config/")],
        key=lambda k: (FACTOR_PRIORITY.index(k[7:]) if k[7:] in FACTOR_PRIORITY else 9999, k),
    )
    effect_rows = []
    for factor in factors:
        by_value = defaultdict(list)
        for row in run_rows:
            if not include_baseline and row.get("method") == baseline_method:
                continue
            delta = delta_by_run.get(row["run_id"])
            if not delta:
                continue
            value = row.get(factor)
            if value is None or value == "":
                continue
            by_value[value_key(value)].append(delta)
        if len(by_value) < 2:
            continue
        value_rows = []
        for value, rows in by_value.items():
            value_rows.append({
                "factor": factor[7:],
                "value": value,
                "count": len(rows),
                "mean_delta_psnr": mean_or_none(r.get("delta/test/psnr") for r in rows),
                "mean_delta_ssim": mean_or_none(r.get("delta/test/ssim") for r in rows),
                "mean_delta_lpips_quality": mean_or_none(r.get("delta/lpips_quality") for r in rows),
                "mean_points_total": mean_or_none(r.get("points/total") for r in rows),
            })
        psnr_values = [safe_float(r["mean_delta_psnr"]) for r in value_rows]
        psnr_values = [v for v in psnr_values if v is not None]
        effect_range = max(psnr_values) - min(psnr_values) if len(psnr_values) >= 2 else None
        for row in value_rows:
            row["factor_psnr_range"] = effect_range
            effect_rows.append(row)
    effect_rows.sort(key=lambda r: (safe_float(r.get("factor_psnr_range")) or -1, r["factor"], r["value"]), reverse=True)
    return effect_rows


def history_value(row, metric_key, feature):
    prefix = metric_key.replace("/", "__")
    return safe_float(row.get(prefix + "__" + feature))


def compute_routing_decomposition(canonical_rows, delta_rows, history_rows):
    delta_by_run = {row["run_id"]: row for row in delta_rows}
    history_by_run = {row["run_id"]: row for row in history_rows}
    rows = []

    for run in canonical_rows:
        history = history_by_run.get(run["run_id"], {})
        delta = delta_by_run.get(run["run_id"], {})
        row = {
            "scene": run.get("scene"),
            "method": run.get("method"),
            "run_id": run.get("run_id"),
            "is_scaffold": is_scaffold_row(run),
            "test_psnr": safe_float(run.get("test/psnr")),
            "delta_psnr_vs_default": safe_float(delta.get("delta/test/psnr")),
            "test_ssim": safe_float(run.get("test/ssim")),
            "delta_ssim_vs_default": safe_float(delta.get("delta/test/ssim")),
            "points_total": safe_float(run.get("points/total")),
            "train_psnr_auc": history_value(history, "train/psnr", "auc"),
            "test_psnr_at_3000": history_value(history, "test/psnr", "at_3000"),
            "test_psnr_at_6000": history_value(history, "test/psnr", "at_6000"),
            "test_psnr_at_9000": history_value(history, "test/psnr", "at_9000"),
            "train_total_loss_final": history_value(history, "train/total_loss", "final"),
            "points_total_final": history_value(history, "points/total", "final"),
            "routing_entropy_first": history_value(history, "routing/entropy", "first"),
            "routing_entropy_final": history_value(history, "routing/entropy", "final"),
            "routing_entropy_delta": history_value(history, "routing/entropy", "delta"),
            "routing_entropy_auc": history_value(history, "routing/entropy", "auc"),
            "routing_expected_static_final": history_value(history, "routing/expected_static_points", "final"),
            "routing_expected_static_delta": history_value(history, "routing/expected_static_points", "delta"),
            "routing_percent_uncertain_final": history_value(history, "routing/percent_uncertain", "final"),
            "motion_lora_coeff_norm_final": history_value(history, "motion_lora/coeff_norm_mean", "final"),
            "motion_lora_basis_norm_final": history_value(history, "motion_lora/basis_norm_mean", "final"),
            "motion_scaffold_coeff_norm_final": history_value(history, "motion_scaffold/coeff_norm_mean", "final"),
            "motion_scaffold_attach_entropy_final": history_value(history, "motion_scaffold/attach_entropy", "final"),
            "routing_entropy_count": safe_float(history.get("routing__entropy__count")),
            "routing_expected_static_count": safe_float(history.get("routing__expected_static_points__count")),
            "motion_scaffold_coeff_norm_count": safe_float(history.get("motion_scaffold__coeff_norm_mean__count")),
            "motion_scaffold_attach_entropy_count": safe_float(history.get("motion_scaffold__attach_entropy__count")),
        }
        rows.append(row)
    return rows


def summarize_routing_decomposition(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    out_rows = []
    fields = [
        "delta_psnr_vs_default",
        "points_total",
        "train_psnr_auc",
        "test_psnr_at_3000",
        "test_psnr_at_6000",
        "test_psnr_at_9000",
        "routing_entropy_final",
        "routing_entropy_delta",
        "routing_entropy_auc",
        "routing_expected_static_final",
        "routing_percent_uncertain_final",
        "motion_scaffold_coeff_norm_final",
        "motion_scaffold_attach_entropy_final",
    ]
    for method, group in grouped.items():
        out = {
            "method": method,
            "scene_count": len(set(r["scene"] for r in group)),
            "has_scaffold": any(r.get("is_scaffold") for r in group),
        }
        for field in fields:
            out["mean_" + field] = mean_or_none(r.get(field) for r in group)
        out_rows.append(out)
    out_rows.sort(key=lambda r: safe_float(r.get("mean_delta_psnr_vs_default")) if safe_float(r.get("mean_delta_psnr_vs_default")) is not None else -999, reverse=True)
    return out_rows


def pearson(xs, ys):
    pairs = [(safe_float(x), safe_float(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        return None, len(pairs)
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mx = mean(xs)
    my = mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None, len(pairs)
    corr = sum((x - mx) * (y - my) for x, y in pairs) / (sx * sy)
    return corr, len(pairs)


def compute_routing_correlations(rows):
    target = "delta_psnr_vs_default"
    features = [
        "points_total",
        "train_psnr_auc",
        "test_psnr_at_3000",
        "test_psnr_at_6000",
        "routing_entropy_final",
        "routing_entropy_delta",
        "routing_entropy_auc",
        "routing_expected_static_final",
        "routing_percent_uncertain_final",
        "motion_scaffold_coeff_norm_final",
        "motion_scaffold_attach_entropy_final",
    ]
    out = []
    for feature in features:
        corr, count = pearson([r.get(feature) for r in rows], [r.get(target) for r in rows])
        out.append({
            "target": target,
            "feature": feature,
            "pearson_r": corr,
            "n": count,
        })
    out.sort(key=lambda r: abs(safe_float(r.get("pearson_r")) or 0.0), reverse=True)
    return out


def collect_fieldnames(rows):
    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = collect_fieldnames(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def markdown_table(rows, columns, limit=None):
    selected = rows[:limit] if limit else rows
    if not selected:
        return "_No rows._\n"
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in selected:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = format_float(value, 4)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def build_warnings(run_rows, duplicates, baseline_method, canonical, scenes):
    warnings = []
    notes = []
    missing_scene = [r for r in run_rows if not r.get("scene")]
    missing_method = [r for r in run_rows if not r.get("method")]
    noisy = [r for r in run_rows if r.get("noisy_config_keys")]
    noisy_with_non_sentinel = [r for r in noisy if r.get("noisy_config_non_sentinel_keys")]
    missing_metrics = [r for r in run_rows if safe_float(r.get("test/psnr")) is None]
    baseline_scenes = set(scene for scene, method in canonical.keys() if method == baseline_method)
    missing_baselines = [scene for scene in scenes if scene not in baseline_scenes]

    if missing_scene:
        notes.append("%d fetched runs were outside the selected scene set or could not be mapped." % len(missing_scene))
    if missing_method:
        warnings.append("%d runs could not be mapped to a method." % len(missing_method))
    if noisy_with_non_sentinel:
        warnings.append("%d runs contain non-sentinel noisy config keys; these are excluded from factor analysis." % len(noisy_with_non_sentinel))
    elif noisy:
        notes.append("%d runs contain old metadata config sentinel keys; these are ignored by factor analysis." % len(noisy))
    if missing_metrics:
        warnings.append("%d runs are missing test/psnr in W&B summary/local stats." % len(missing_metrics))
    if missing_baselines:
        warnings.append("Missing baseline method '%s' for scenes: %s." % (baseline_method, ", ".join(missing_baselines)))
    if duplicates:
        warnings.append("%d scene-method groups have duplicate runs; canonical selection was applied." % len(duplicates))
    return warnings, notes


def write_report(
        out_dir,
        args,
        run_rows,
        canonical_rows,
        delta_rows,
        method_rows,
        factor_rows,
        history_rows,
        capacity_nearest_summary,
        capacity_within_summary,
        capacity_nearest_reference_summary,
        fixed_budget_summary,
        routing_summary,
        routing_correlations,
        warnings,
        notes,
):
    report_path = out_dir / "report.md"
    with report_path.open("w") as handle:
        handle.write("# ADAGS W&B Paired Scene Analysis\n\n")
        handle.write("Entity/project: `%s/%s`\n\n" % (args.entity, args.project))
        handle.write("Baseline method: `%s`\n\n" % args.baseline_method)
        handle.write("Dedupe policy: `%s`\n\n" % args.dedupe)

        handle.write("## Dataset\n\n")
        handle.write("- Runs fetched: `%d`\n" % len(run_rows))
        handle.write("- Canonical scene-method runs: `%d`\n" % len(canonical_rows))
        handle.write("- Paired delta rows: `%d`\n" % len(delta_rows))
        handle.write("- History feature rows: `%d`\n\n" % len(history_rows))

        handle.write("## Warnings\n\n")
        if warnings:
            for warning in warnings:
                handle.write("- %s\n" % warning)
        else:
            handle.write("- None\n")
        handle.write("\n")

        handle.write("## Notes\n\n")
        if notes:
            for note in notes:
                handle.write("- %s\n" % note)
        else:
            handle.write("- None\n")
        handle.write("\n")

        handle.write("## Top Methods By Paired PSNR Delta\n\n")
        handle.write(markdown_table(
            method_rows,
            [
                "method",
                "scene_count",
                "mean_delta_psnr",
                "ci95_delta_psnr_low",
                "ci95_delta_psnr_high",
                "mean_delta_ssim",
                "mean_delta_lpips_quality",
                "mean_points_total",
            ],
            limit=20,
        ))
        handle.write("\n")

        handle.write("## Largest Exploratory Factor Ranges\n\n")
        handle.write(markdown_table(
            factor_rows,
            ["factor", "value", "count", "mean_delta_psnr", "mean_delta_ssim", "mean_delta_lpips_quality", "factor_psnr_range"],
            limit=30,
        ))
        handle.write("\n")

        handle.write("## Capacity-Matched Scaffold Checks\n\n")
        handle.write("Nearest non-scaffold reference per scene/scaffold method. Positive PSNR means scaffold is better.\n\n")
        handle.write(markdown_table(
            capacity_nearest_summary,
            [
                "scaffold_method",
                "comparison",
                "scene_count",
                "pair_count",
                "mean_point_ratio",
                "mean_scaffold_minus_reference_psnr",
                "mean_scaffold_minus_reference_ssim",
                "mean_scaffold_minus_reference_lpips_quality",
                "reference_methods",
            ],
            limit=30,
        ))
        handle.write("\n")

        handle.write("Within configured capacity-ratio threshold:\n\n")
        handle.write(markdown_table(
            capacity_within_summary,
            [
                "scaffold_method",
                "comparison",
                "scene_count",
                "pair_count",
                "mean_point_ratio",
                "mean_scaffold_minus_reference_psnr",
                "mean_scaffold_minus_reference_ssim",
                "mean_scaffold_minus_reference_lpips_quality",
                "reference_methods",
            ],
            limit=30,
        ))
        handle.write("\n")

        handle.write("Nearest pairs split by reference method:\n\n")
        handle.write(markdown_table(
            capacity_nearest_reference_summary,
            [
                "scaffold_method",
                "reference_method",
                "scene_count",
                "mean_point_ratio",
                "mean_scaffold_minus_reference_psnr",
                "mean_scaffold_minus_reference_ssim",
                "mean_scaffold_minus_reference_lpips_quality",
            ],
            limit=30,
        ))
        handle.write("\n")

        handle.write("## Exact Fixed-Budget Scaffold Checks\n\n")
        handle.write("Same scene and same configured `densify_until_num_points`. Positive values mean scaffold is better after metric direction is normalized.\n\n")
        handle.write(markdown_table(
            fixed_budget_summary,
            [
                "budget_points",
                "scaffold_method",
                "reference_method",
                "scene_count",
                "pair_count",
                "mean_point_ratio",
                "mean_scaffold_minus_reference_psnr",
                "mean_scaffold_minus_reference_ssim",
                "mean_scaffold_minus_reference_lpips_quality",
                "mean_scaffold_minus_reference_dynamic_mask_psnr",
                "mean_scaffold_minus_reference_static_ghost_score_quality",
                "mean_scaffold_minus_reference_routing_entropy",
            ],
            limit=40,
        ))
        handle.write("\n")

        handle.write("## Routing / Decomposition Summary\n\n")
        handle.write(markdown_table(
            routing_summary,
            [
                "method",
                "scene_count",
                "has_scaffold",
                "mean_delta_psnr_vs_default",
                "mean_points_total",
                "mean_routing_entropy_final",
                "mean_routing_entropy_delta",
                "mean_routing_expected_static_final",
                "mean_motion_scaffold_coeff_norm_final",
                "mean_motion_scaffold_attach_entropy_final",
            ],
            limit=30,
        ))
        handle.write("\n")

        handle.write("## Routing Correlations\n\n")
        handle.write("Exploratory Pearson correlations against `delta_psnr_vs_default`; only rows with available history are used.\n\n")
        handle.write(markdown_table(
            routing_correlations,
            ["feature", "target", "pearson_r", "n"],
            limit=30,
        ))
        handle.write("\n")

        handle.write("## Interpretation Rules\n\n")
        handle.write("- Treat paired deltas as the primary comparison, not raw W&B parameter importance.\n")
        handle.write("- Treat factor ranges as exploratory unless the factor was varied on matched scenes.\n")
        handle.write("- Check point-count and routing diagnostics before claiming scaffold quality gains.\n")
        handle.write("- Prefer final paper claims from methods promoted to all six scenes.\n")


def try_plot(out_dir, delta_rows, method_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ["matplotlib unavailable; skipped plots."]

    messages = []

    top = [r for r in method_rows if r.get("method") != "default"][:15]
    if top:
        plt.figure(figsize=(10, max(4, 0.35 * len(top))))
        labels = [r["method"] for r in top]
        values = [safe_float(r.get("mean_delta_psnr")) or 0.0 for r in top]
        plt.barh(range(len(labels)), values)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Mean paired delta PSNR vs default")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(str(out_dir / "mean_delta_psnr_by_method.png"), dpi=200)
        plt.close()
        messages.append("mean_delta_psnr_by_method.png")

    scatter = []
    for row in delta_rows:
        x = safe_float(row.get("points/total"))
        y = safe_float(row.get("delta/test/psnr"))
        if x is not None and y is not None:
            scatter.append((x, y, row.get("method", "")))
    if scatter:
        plt.figure(figsize=(7, 5))
        plt.scatter([x for x, _, _ in scatter], [y for _, y, _ in scatter], s=24, alpha=0.75)
        plt.xlabel("Final total Gaussians")
        plt.ylabel("Paired delta PSNR vs default")
        plt.tight_layout()
        plt.savefig(str(out_dir / "delta_psnr_vs_points.png"), dpi=200)
        plt.close()
        messages.append("delta_psnr_vs_points.png")

    return messages


def upload_derived_metrics(args, delta_rows):
    if not args.upload_derived:
        return []
    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb is not installed. Cannot upload derived metrics.")

    api = wandb.Api(timeout=args.timeout)
    path = "%s/%s" % (args.entity, args.project)
    uploaded = []
    for row in delta_rows:
        if not row.get("has_baseline"):
            continue
        run = api.run("%s/%s" % (path, row["run_id"]))
        updates = {
            "analysis/baseline_method": args.baseline_method,
            "analysis/baseline_run_id": row.get("baseline_run_id"),
            "analysis/delta_psnr_vs_%s" % args.baseline_method: row.get("delta/test/psnr"),
            "analysis/delta_ssim_vs_%s" % args.baseline_method: row.get("delta/test/ssim"),
            "analysis/delta_lpips_vs_%s" % args.baseline_method: row.get("delta/test/lpips"),
            "analysis/delta_lpips_quality_vs_%s" % args.baseline_method: row.get("delta/lpips_quality"),
            "analysis/delta_psnr_per_million_points": row.get("delta_psnr_per_million_points"),
        }
        for key, value in updates.items():
            if value is not None:
                run.summary[key] = value
        run.update()
        uploaded.append(run.id)
    return uploaded


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Analyze ADAGS W&B runs with paired per-scene deltas.")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"), help="W&B entity. Defaults to WANDB_ENTITY.")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "adags"), help="W&B project.")
    parser.add_argument("--out-dir", default="analysis/wandb_adags", help="Output directory for CSV/report/plots.")
    parser.add_argument("--from-existing-dir", default=None, help="Recompute analysis from an existing analysis directory instead of querying W&B.")
    parser.add_argument("--baseline-method", default="default", help="Method name used as per-scene baseline.")
    parser.add_argument("--scenes", nargs="+", default=SCENES, help="Known scene names.")
    parser.add_argument("--state", default="finished", help="Optional W&B run state filter. Use empty string for all states.")
    parser.add_argument("--order", default="-created_at", help="W&B run order.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum runs to fetch.")
    parser.add_argument("--timeout", type=int, default=120, help="W&B API timeout.")
    parser.add_argument("--dedupe", choices=["latest", "first", "best-psnr"], default="latest", help="Canonical run selection for duplicate scene-method groups.")
    parser.add_argument("--local-stats-root", default=None, help="Optional local runs root for validation.json backfill.")
    parser.add_argument("--skip-history", action="store_true", help="Skip history curve feature extraction.")
    parser.add_argument("--history-limit", type=int, default=None, help="Limit canonical runs used for history feature extraction.")
    parser.add_argument("--history-page-size", type=int, default=1000, help="W&B scan_history page size.")
    parser.add_argument("--history-keys", nargs="+", default=None, help="Metric keys to summarize from history.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000, help="Bootstrap samples for method-level CIs.")
    parser.add_argument("--bootstrap-seed", type=int, default=0, help="Bootstrap RNG seed.")
    parser.add_argument("--capacity-match-ratio", type=float, default=1.5, help="Point-count ratio considered matched capacity.")
    parser.add_argument("--include-baseline-in-factor-effects", action="store_true", help="Include the baseline method in exploratory factor-effect summaries.")
    parser.add_argument("--upload-derived", action="store_true", help="Write derived analysis/* summary metrics back to W&B.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    if not args.entity and not args.from_existing_dir:
        raise SystemExit("Provide --entity or set WANDB_ENTITY.")
    if not args.entity:
        args.entity = "unknown"
    if args.state == "":
        args.state = None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_existing_dir:
        existing_dir = Path(args.from_existing_dir)
        run_rows = read_csv(existing_dir / "runs.csv")
        canonical_rows = read_csv(existing_dir / "canonical_runs.csv")
        canonical = {
            (row.get("scene"), row.get("method")): row
            for row in canonical_rows
            if row.get("scene") and row.get("method")
        }
        duplicates = {}
        delta_rows = compute_paired_deltas(canonical, args.baseline_method)
        history_rows = [] if args.skip_history else read_csv(existing_dir / "history_features.csv")
    else:
        run_rows = collect_runs(args)
        canonical, duplicates = choose_canonical(run_rows, args.dedupe)
        canonical_rows = [row for _, row in sorted(canonical.items(), key=lambda item: (item[0][1], item[0][0]))]
        delta_rows = compute_paired_deltas(canonical, args.baseline_method)
        history_rows = collect_history_features(args, canonical_rows)

    method_rows = summarize_methods(delta_rows, args.bootstrap_samples, args.bootstrap_seed)
    capacity_pairs, capacity_nearest_pairs = compute_capacity_pairs(
        canonical,
        args.capacity_match_ratio,
        args.baseline_method,
    )
    capacity_pair_summary = summarize_capacity_pairs(capacity_pairs)
    capacity_nearest_reference_summary = summarize_capacity_pairs(capacity_nearest_pairs, nearest_only=True)
    capacity_nearest_summary = summarize_capacity_by_scaffold(capacity_nearest_pairs, "nearest_capacity")
    capacity_within_pairs = [
        row for row in capacity_pairs
        if str(row.get("within_capacity_ratio")).lower() == "true" or row.get("within_capacity_ratio") is True
    ]
    capacity_within_summary = summarize_capacity_by_scaffold(
        capacity_within_pairs,
        "within_%sx_capacity" % args.capacity_match_ratio,
    )
    fixed_budget_pairs = compute_fixed_budget_pairs(canonical, args.baseline_method)
    fixed_budget_summary = summarize_fixed_budget_pairs(fixed_budget_pairs)
    factor_rows = compute_factor_effects(
        canonical_rows,
        delta_rows,
        args.baseline_method,
        args.include_baseline_in_factor_effects,
    )
    warnings, notes = build_warnings(run_rows, duplicates, args.baseline_method, canonical, args.scenes)
    routing_rows = compute_routing_decomposition(canonical_rows, delta_rows, history_rows)
    routing_summary = summarize_routing_decomposition(routing_rows)
    routing_correlations = compute_routing_correlations(routing_rows)

    write_csv(out_dir / "runs.csv", run_rows)
    write_csv(out_dir / "canonical_runs.csv", canonical_rows)
    write_csv(out_dir / "paired_deltas.csv", delta_rows)
    write_csv(out_dir / "method_summary.csv", method_rows)
    write_csv(out_dir / "capacity_pairs.csv", capacity_pairs)
    write_csv(out_dir / "capacity_nearest_pairs.csv", capacity_nearest_pairs)
    write_csv(out_dir / "capacity_pair_summary.csv", capacity_pair_summary)
    write_csv(out_dir / "capacity_nearest_summary.csv", capacity_nearest_summary)
    write_csv(out_dir / "capacity_nearest_reference_summary.csv", capacity_nearest_reference_summary)
    write_csv(out_dir / "capacity_within_pairs.csv", capacity_within_pairs)
    write_csv(out_dir / "capacity_within_summary.csv", capacity_within_summary)
    write_csv(out_dir / "fixed_budget_pairs.csv", fixed_budget_pairs)
    write_csv(out_dir / "fixed_budget_summary.csv", fixed_budget_summary)
    write_csv(out_dir / "factor_effects.csv", factor_rows)
    write_csv(out_dir / "history_features.csv", history_rows)
    write_csv(out_dir / "routing_decomposition.csv", routing_rows)
    write_csv(out_dir / "routing_summary.csv", routing_summary)
    write_csv(out_dir / "routing_correlations.csv", routing_correlations)

    write_report(
        out_dir,
        args,
        run_rows,
        canonical_rows,
        delta_rows,
        method_rows,
        factor_rows,
        history_rows,
        capacity_nearest_summary,
        capacity_within_summary,
        capacity_nearest_reference_summary,
        fixed_budget_summary,
        routing_summary,
        routing_correlations,
        warnings,
        notes,
    )
    plot_messages = try_plot(out_dir, delta_rows, method_rows)
    uploaded = upload_derived_metrics(args, delta_rows)

    print("Wrote analysis to %s" % out_dir)
    print("runs=%d canonical=%d paired=%d history=%d" % (len(run_rows), len(canonical_rows), len(delta_rows), len(history_rows)))
    if warnings:
        print("warnings:")
        for warning in warnings:
            print("  - %s" % warning)
    if notes:
        print("notes:")
        for note in notes:
            print("  - %s" % note)
    if plot_messages:
        print("plots:", ", ".join(plot_messages))
    if uploaded:
        print("uploaded derived metrics to %d W&B runs" % len(uploaded))


if __name__ == "__main__":
    main()
