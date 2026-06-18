#!/usr/bin/env python3
"""Checkpoint-aware fixed-budget W&B analysis for ADAGS."""

import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


DEFAULT_METHODS = [
    "lora_route0",
    "lora_route0_dyn",
    "scaffold_lora_route0_noreg",
    "scaffold_lora_route0_reg",
    "scaffold_lora_route0_dyn",
]
METHODS = list(DEFAULT_METHODS)
METHOD_ALIASES = {
    "scaffold_lora_route0": "scaffold_lora_route0_reg",
}
METHOD_LABELS = {
    "lora_route0": "lora",
    "lora_route0_dyn": "lora_dyn",
    "scaffold_lora_route0_noreg": "scaffold_noreg",
    "scaffold_lora_route0_reg": "scaffold_reg",
    "scaffold_lora_route0_dyn": "scaffold_dyn",
}
DEFAULT_BUDGETS = ["400k", "600k", "800k"]
BUDGETS = list(DEFAULT_BUDGETS)
BUDGET_POINTS = {"400k": 400000, "600k": 600000, "800k": 800000}
DEFAULT_SCENES = ["cut_roasted_beef", "flame_steak", "sear_steak"]
SCENES = list(DEFAULT_SCENES)
BASELINE_METHOD = "lora_route0"
PRIMARY_CHECKPOINT = "6000"
REFERENCE_CHECKPOINT = "9000"

SUMMARY_KEYS = [
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
]
HISTORY_KEYS = [
    "train/psnr",
    "train/total_loss",
    "test/psnr",
    "test/ssim",
    "test/lpips",
    "test/dynamic_mask_psnr",
    "test/static_ghost_score",
    "points/total",
    "routing/entropy",
    "routing/expected_static_points",
    "motion_scaffold/coeff_norm_mean",
    "motion_scaffold/attach_entropy",
]
PAIR_DEFS = []


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


def avg(values):
    clean = [safe_float(v) for v in values]
    clean = [v for v in clean if v is not None]
    return mean(clean) if clean else None


def fmt(value, digits=4):
    value = safe_float(value)
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def canonical_method(method):
    if method is None:
        return None
    return METHOD_ALIASES.get(str(method), str(method))


def method_label(method):
    method = canonical_method(method)
    return METHOD_LABELS.get(method, method)


def parse_list(values, default):
    if not values:
        return list(default)
    out = []
    for value in values:
        out.extend(str(value).split())
    return [value for value in out if value]


def configure_scope(args):
    global METHODS, BUDGETS, SCENES, BASELINE_METHOD, PRIMARY_CHECKPOINT, REFERENCE_CHECKPOINT
    METHODS = [canonical_method(m) for m in parse_list(args.methods, DEFAULT_METHODS)]
    METHODS = [m for i, m in enumerate(METHODS) if m and m not in METHODS[:i]]
    BUDGETS = parse_list(args.budgets, DEFAULT_BUDGETS)
    SCENES = parse_list(args.scenes, DEFAULT_SCENES)
    BASELINE_METHOD = canonical_method(args.baseline_method)
    PRIMARY_CHECKPOINT = str(args.primary_checkpoint)
    REFERENCE_CHECKPOINT = str(args.reference_checkpoint)


def active_pair_defs():
    if PAIR_DEFS:
        return PAIR_DEFS
    return [
        (f"{method_label(method)}_minus_{method_label(BASELINE_METHOD)}", method, BASELINE_METHOD)
        for method in METHODS
        if method != BASELINE_METHOD
    ]


def budget_sort_key(budget):
    return BUDGETS.index(budget) if budget in BUDGETS else 999


def method_sort_key(method):
    method = canonical_method(method)
    return METHODS.index(method) if method in METHODS else 999


def metric_col(metric):
    return metric.replace("/", "__")


def hist_col(metric, step):
    return f"{metric_col(metric)}__at_{step}"


def find_base_run_ids(runs_root):
    pattern = os.path.join(
        runs_root,
        "fixed_budget_*",
        "*fixed_budget_*",
        "wandb",
        "offline-run-*",
    )
    ids = set()
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        if not name.startswith("offline-run-"):
            continue
        rest = name[len("offline-run-") :]
        if "-" not in rest:
            continue
        run_id = rest.split("-", 1)[1]
        if run_id.endswith("_eval6000"):
            continue
        if "fixed_budget_" in run_id:
            ids.add(run_id)
    return sorted(ids)


def infer_from_run_id(run_id):
    scene = next((s for s in SCENES if s in run_id), None)
    method_candidates = sorted(set(METHODS) | set(METHOD_ALIASES.keys()), key=len, reverse=True)
    method = next((m for m in method_candidates if f"fixed_budget_{m}_" in run_id), None)
    method = canonical_method(method)
    match = re.search(r"_(400k|600k|800k)(?:_eval6000)?$", run_id)
    budget = match.group(1) if match else None
    return scene, method, budget


def infer_run_checkpoint(run, default_checkpoint):
    summary = dict(getattr(run, "summary_metrics", None) or {})
    for value in (
        summary.get("final_iteration"),
        summary.get("final_val_iter"),
        getattr(run, "lastHistoryStep", None),
        default_checkpoint,
    ):
        value = safe_float(value)
        if value is not None and value > 0:
            return str(int(value))
    return str(default_checkpoint)


def fetch_summary_row(args, run, base_run_id, checkpoint):
    tags = list(getattr(run, "tags", []) or [])
    cfg = dict(getattr(run, "config", {}) or {})
    summary = dict(getattr(run, "summary_metrics", None) or {})
    scene, method, budget = infer_from_run_id(base_run_id)
    method = canonical_method(cfg.get("method_family") or method)
    budget = cfg.get("budget_label") or budget
    scene = getattr(run, "group", None) if getattr(run, "group", None) in SCENES else scene
    row = {
        "base_run_id": base_run_id,
        "run_id": run.id,
        "run_name": getattr(run, "name", run.id),
        "checkpoint": checkpoint,
        "scene": scene,
        "method": method,
        "method_label": method_label(method),
        "budget_label": budget,
        "budget_points": BUDGET_POINTS.get(budget),
        "state": getattr(run, "state", ""),
        "group": getattr(run, "group", ""),
        "tags": ",".join(tags),
        "url": f"https://wandb.ai/{args.entity}/{args.project}/runs/{run.id}",
    }
    for key in SUMMARY_KEYS:
        row[key] = summary.get(key)
    return row


def fetch_series(run, key, page_size):
    series = []
    for item in run.scan_history(keys=["_step", key], page_size=page_size):
        value = safe_float(item.get(key))
        step = safe_float(item.get("_step"))
        if value is not None and step is not None:
            series.append((int(step), value))
    series.sort()
    return series


def value_at_or_before(series, target):
    selected = None
    for step, value in series:
        if step <= target:
            selected = value
    return selected


def collect_training_history(base_rows, run_objects, args):
    rows = []
    sampled = []
    for index, row in enumerate(base_rows, 1):
        run = run_objects[row["base_run_id"]]
        print(f"[{index}/{len(base_rows)}] history {row['scene']} {row['method_label']} {row['budget_label']}", flush=True)
        out = {
            "base_run_id": row["base_run_id"],
            "scene": row["scene"],
            "method": row["method"],
            "method_label": row["method_label"],
            "budget_label": row["budget_label"],
        }
        for key in args.history_keys:
            try:
                series = fetch_series(run, key, args.history_page_size)
            except Exception as exc:
                print(f"history warning: {run.id} {key}: {exc}", file=sys.stderr)
                series = []
            out[f"{metric_col(key)}__count"] = len(series)
            for checkpoint in sorted({int(PRIMARY_CHECKPOINT), int(REFERENCE_CHECKPOINT)}):
                out[hist_col(key, checkpoint)] = value_at_or_before(series, checkpoint)
            if series:
                out[f"{metric_col(key)}__final"] = series[-1][1]
                out[f"{metric_col(key)}__max"] = max(v for _, v in series)
            sampled.extend(sample_series(row, key, series, args.max_plot_points_per_series))
        rows.append(out)
    return rows, sampled


def sample_series(row, key, series, max_points):
    if not series or max_points <= 0:
        return []
    stride = max(1, len(series) // max_points)
    sampled = series[::stride]
    if sampled[-1] != series[-1]:
        sampled.append(series[-1])
    return [
        {
            "base_run_id": row["base_run_id"],
            "scene": row["scene"],
            "method": row["method"],
            "method_label": row["method_label"],
            "budget_label": row["budget_label"],
            "metric": key,
            "step": step,
            "value": value,
        }
        for step, value in sampled
    ]


def attach_history(eval_rows, history_rows):
    history = {r["base_run_id"]: r for r in history_rows}
    out = []
    for row in eval_rows:
        merged = dict(row)
        h = history.get(row["base_run_id"], {})
        step = int(row["checkpoint"])
        for key in HISTORY_KEYS:
            merged[f"history/{key}@{step}"] = h.get(hist_col(key, step))
        out.append(merged)
    return out


def key_cell(row):
    return row["scene"], row["method"], row["budget_label"]


def best_checkpoint_rows(eval_rows):
    grouped = defaultdict(list)
    for row in eval_rows:
        grouped[key_cell(row)].append(row)
    best = []
    for cell, rows in grouped.items():
        ranked = sorted(rows, key=lambda r: safe_float(r.get("test/psnr")) if safe_float(r.get("test/psnr")) is not None else -1e9, reverse=True)
        if not ranked:
            continue
        chosen = dict(ranked[0])
        chosen["selection"] = "best"
        other = ranked[1] if len(ranked) > 1 else {}
        chosen["other_checkpoint"] = other.get("checkpoint", "")
        chosen["other_psnr"] = other.get("test/psnr", "")
        if safe_float(chosen.get("test/psnr")) is not None and safe_float(other.get("test/psnr")) is not None:
            chosen["best_minus_other_psnr"] = safe_float(chosen["test/psnr"]) - safe_float(other["test/psnr"])
        best.append(chosen)
    best.sort(key=lambda r: (r["scene"], method_sort_key(r["method"]), budget_sort_key(r["budget_label"])))
    return best


def checkpoint_gains(eval_rows):
    by_cell_ckpt = {(r["scene"], r["method"], r["budget_label"], r["checkpoint"]): r for r in eval_rows}
    rows = []
    for scene in SCENES:
        for method in METHODS:
            for budget in BUDGETS:
                primary = by_cell_ckpt.get((scene, method, budget, PRIMARY_CHECKPOINT))
                reference = by_cell_ckpt.get((scene, method, budget, REFERENCE_CHECKPOINT))
                if not primary or not reference:
                    continue
                row = {
                    "scene": scene,
                    "method": method,
                    "method_label": method_label(method),
                    "budget_label": budget,
                    "primary_checkpoint": PRIMARY_CHECKPOINT,
                    "reference_checkpoint": REFERENCE_CHECKPOINT,
                    "primary_psnr": primary.get("test/psnr"),
                    "reference_psnr": reference.get("test/psnr"),
                    "primary_ssim": primary.get("test/ssim"),
                    "reference_ssim": reference.get("test/ssim"),
                    "primary_lpips": primary.get("test/lpips"),
                    "reference_lpips": reference.get("test/lpips"),
                    "primary_points": primary.get("points/total"),
                    "reference_points": reference.get("points/total"),
                }
                row["psnr_primary_minus_reference"] = diff(primary, reference, "test/psnr")
                row["ssim_primary_minus_reference"] = diff(primary, reference, "test/ssim")
                lpips_delta = diff(primary, reference, "test/lpips")
                row["lpips_quality_primary_minus_reference"] = -lpips_delta if lpips_delta is not None else None
                primary_delta = safe_float(row["psnr_primary_minus_reference"])
                row["best_checkpoint"] = PRIMARY_CHECKPOINT if primary_delta is not None and primary_delta > 0 else REFERENCE_CHECKPOINT
                row["best_psnr"] = max(safe_float(row["primary_psnr"]) or -1e9, safe_float(row["reference_psnr"]) or -1e9)
                rows.append(row)
    return rows


def diff(left, right, metric):
    lv = safe_float(left.get(metric))
    rv = safe_float(right.get(metric))
    if lv is None or rv is None:
        return None
    return lv - rv


def method_pairs(rows, label):
    by_key = {(r["scene"], r["method"], r["budget_label"]): r for r in rows}
    pairs = []
    for scene in SCENES:
        for budget in BUDGETS:
            for comparison, left_method, right_method in active_pair_defs():
                left = by_key.get((scene, left_method, budget))
                right = by_key.get((scene, right_method, budget))
                if not left or not right:
                    continue
                pair = {
                    "analysis_set": label,
                    "scene": scene,
                    "budget_label": budget,
                    "comparison": comparison,
                    "left_method": left_method,
                    "right_method": right_method,
                    "left_checkpoint": left.get("checkpoint", ""),
                    "right_checkpoint": right.get("checkpoint", ""),
                    "left_run_id": left["run_id"],
                    "right_run_id": right["run_id"],
                    "left_psnr": left.get("test/psnr"),
                    "right_psnr": right.get("test/psnr"),
                    "left_dynamic_mask_psnr": left.get("test/dynamic_mask_psnr"),
                    "right_dynamic_mask_psnr": right.get("test/dynamic_mask_psnr"),
                    "left_static_ghost_score": left.get("test/static_ghost_score"),
                    "right_static_ghost_score": right.get("test/static_ghost_score"),
                    "left_points": left.get("points/total"),
                    "right_points": right.get("points/total"),
                }
                pair["delta_psnr"] = diff(left, right, "test/psnr")
                pair["delta_ssim"] = diff(left, right, "test/ssim")
                pair["delta_dynamic_mask_psnr"] = diff(left, right, "test/dynamic_mask_psnr")
                static_ghost_delta = diff(left, right, "test/static_ghost_score")
                pair["delta_static_ghost_quality"] = -static_ghost_delta if static_ghost_delta is not None else None
                lpips_delta = diff(left, right, "test/lpips")
                pair["delta_lpips_quality"] = -lpips_delta if lpips_delta is not None else None
                pts_l = safe_float(left.get("points/total"))
                pts_r = safe_float(right.get("points/total"))
                pair["delta_points"] = pts_l - pts_r if pts_l is not None and pts_r is not None else None
                if pts_l and pts_r:
                    pair["point_ratio_left_over_right"] = pts_l / pts_r
                    pair["point_ratio_max_over_min"] = max(pts_l, pts_r) / min(pts_l, pts_r)
                pairs.append(pair)
    return pairs


def within_budget_rows(rows, label):
    by_key = {(r["scene"], r["method"], r["budget_label"]): r for r in rows}
    out = []
    for scene in SCENES:
        for method in METHODS:
            for comparison, high, low in [("600k_minus_400k", "600k", "400k"), ("800k_minus_600k", "800k", "600k"), ("800k_minus_400k", "800k", "400k")]:
                left = by_key.get((scene, method, high))
                right = by_key.get((scene, method, low))
                if not left or not right:
                    continue
                row = {
                    "analysis_set": label,
                    "scene": scene,
                    "method": method,
                    "method_label": method_label(method),
                    "comparison": comparison,
                    "high_budget": high,
                    "low_budget": low,
                    "high_checkpoint": left.get("checkpoint", ""),
                    "low_checkpoint": right.get("checkpoint", ""),
                    "delta_psnr": diff(left, right, "test/psnr"),
                    "delta_ssim": diff(left, right, "test/ssim"),
                    "delta_points": diff(left, right, "points/total"),
                }
                lpips_delta = diff(left, right, "test/lpips")
                row["delta_lpips_quality"] = -lpips_delta if lpips_delta is not None else None
                out.append(row)
    return out


def summarize(rows, group_keys, value_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in group_keys)].append(row)
    out = []
    for key, group in grouped.items():
        row = {k: key[i] for i, k in enumerate(group_keys)}
        row["n"] = len(group)
        if "scene" not in group_keys:
            row["scene_count"] = len(set(g.get("scene") for g in group))
        for value_key in value_keys:
            values = [g.get(value_key) for g in group]
            clean = [safe_float(v) for v in values if safe_float(v) is not None]
            row[f"mean_{value_key}"] = avg(clean)
            row[f"positive_{value_key}"] = sum(1 for v in clean if v > 0)
            row[f"negative_{value_key}"] = sum(1 for v in clean if v < 0)
        out.append(row)
    return out


def row_in_scope(row):
    return (
        row.get("scene") in SCENES
        and row.get("method") in METHODS
        and row.get("budget_label") in BUDGETS
    )


def primary_checkpoint_audit(eval_rows):
    primary_rows = [r for r in eval_rows if r.get("checkpoint") == PRIMARY_CHECKPOINT]
    reference_by_cell = {
        (r["scene"], r["method"], r["budget_label"]): r
        for r in eval_rows
        if r.get("checkpoint") == REFERENCE_CHECKPOINT
    }
    baseline_by_scene_budget = {
        (r["scene"], r["budget_label"]): r
        for r in primary_rows
        if r.get("method") == BASELINE_METHOD
    }
    out = []
    for row in sorted(primary_rows, key=lambda r: (r["scene"], method_sort_key(r["method"]), budget_sort_key(r["budget_label"]))):
        baseline = baseline_by_scene_budget.get((row["scene"], row["budget_label"]))
        reference = reference_by_cell.get((row["scene"], row["method"], row["budget_label"]))
        points = safe_float(row.get("points/total"))
        baseline_points = safe_float(baseline.get("points/total")) if baseline else None
        train_psnr = safe_float(row.get(f"history/train/psnr@{PRIMARY_CHECKPOINT}"))
        test_psnr = safe_float(row.get("test/psnr"))
        audit = {
            "scene": row["scene"],
            "method": row["method"],
            "method_label": row["method_label"],
            "budget_label": row["budget_label"],
            "target_points": BUDGET_POINTS.get(row["budget_label"]),
            "realized_points": row.get("points/total"),
            "point_ratio_vs_lora": points / baseline_points if points is not None and baseline_points else None,
            "test_psnr": row.get("test/psnr"),
            "test_ssim": row.get("test/ssim"),
            "test_lpips": row.get("test/lpips"),
            "dynamic_mask_psnr": row.get("test/dynamic_mask_psnr"),
            "static_ghost_score": row.get("test/static_ghost_score"),
            "routing_entropy": row.get("history/routing/entropy@%s" % PRIMARY_CHECKPOINT),
            "routing_expected_static_points": row.get("history/routing/expected_static_points@%s" % PRIMARY_CHECKPOINT),
            "scaffold_coeff_norm": row.get("history/motion_scaffold/coeff_norm_mean@%s" % PRIMARY_CHECKPOINT),
            "scaffold_attach_entropy": row.get("history/motion_scaffold/attach_entropy@%s" % PRIMARY_CHECKPOINT),
            "train_psnr": train_psnr,
            "train_test_gap_psnr": train_psnr - test_psnr if train_psnr is not None and test_psnr is not None else None,
            "reference_checkpoint": REFERENCE_CHECKPOINT if reference else "",
            "reference_psnr": reference.get("test/psnr") if reference else None,
            "checkpoint_gain_psnr_primary_minus_reference": diff(row, reference, "test/psnr") if reference else None,
            "run_id": row["run_id"],
            "base_run_id": row["base_run_id"],
        }
        out.append(audit)
    return out


def build_gate_summary(primary_pairs, gate_budget, psnr_threshold, point_ratio_threshold):
    grouped = defaultdict(list)
    for row in primary_pairs:
        if row.get("right_method") != BASELINE_METHOD:
            continue
        if gate_budget and row.get("budget_label") != gate_budget:
            continue
        grouped[(row["comparison"], row["left_method"], row["budget_label"])].append(row)

    out = []
    for (comparison, method, budget), group in grouped.items():
        mean_delta_psnr = avg(r.get("delta_psnr") for r in group)
        mean_delta_ssim = avg(r.get("delta_ssim") for r in group)
        mean_delta_lpips_quality = avg(r.get("delta_lpips_quality") for r in group)
        mean_delta_dynamic = avg(r.get("delta_dynamic_mask_psnr") for r in group)
        mean_delta_static_ghost_quality = avg(r.get("delta_static_ghost_quality") for r in group)
        point_ratios = [safe_float(r.get("point_ratio_left_over_right")) for r in group]
        point_ratios = [r for r in point_ratios if r is not None]
        max_point_ratio = max(point_ratios) if point_ratios else None
        row = {
            "comparison": comparison,
            "candidate_method": method,
            "candidate_label": method_label(method),
            "baseline_method": BASELINE_METHOD,
            "budget_label": budget,
            "n": len(group),
            "scene_count": len(set(r.get("scene") for r in group if r.get("scene"))),
            "mean_delta_psnr": mean_delta_psnr,
            "positive_delta_psnr": sum(1 for r in group if (safe_float(r.get("delta_psnr")) or 0) > 0),
            "negative_delta_psnr": sum(1 for r in group if (safe_float(r.get("delta_psnr")) or 0) < 0),
            "mean_delta_ssim": mean_delta_ssim,
            "mean_delta_lpips_quality": mean_delta_lpips_quality,
            "mean_delta_dynamic_mask_psnr": mean_delta_dynamic,
            "mean_delta_static_ghost_quality": mean_delta_static_ghost_quality,
            "mean_point_ratio_vs_lora": avg(point_ratios),
            "max_point_ratio_vs_lora": max_point_ratio,
            "psnr_threshold": psnr_threshold,
            "point_ratio_threshold": point_ratio_threshold,
        }
        row["passes_psnr_gate"] = mean_delta_psnr is not None and mean_delta_psnr >= psnr_threshold
        row["passes_ssim_gate"] = mean_delta_ssim is not None and mean_delta_ssim >= 0.0
        row["passes_lpips_gate"] = mean_delta_lpips_quality is not None and mean_delta_lpips_quality >= 0.0
        row["passes_capacity_gate"] = max_point_ratio is not None and max_point_ratio <= point_ratio_threshold
        row["passes_all_gates"] = (
            row["passes_psnr_gate"]
            and row["passes_ssim_gate"]
            and row["passes_lpips_gate"]
            and row["passes_capacity_gate"]
        )
        out.append(row)
    return sorted(out, key=lambda r: (budget_sort_key(r["budget_label"]), method_sort_key(r["candidate_method"])))


def best_winners(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["budget_label"])].append(row)
    out = []
    for (scene, budget), group in grouped.items():
        ranked = sorted(group, key=lambda r: safe_float(r.get("test/psnr")) if safe_float(r.get("test/psnr")) is not None else -1e9, reverse=True)
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else {}
        out.append({
            "scene": scene,
            "budget_label": budget,
            "best_method": best["method"],
            "best_method_label": best["method_label"],
            "best_checkpoint": best["checkpoint"],
            "best_psnr": best.get("test/psnr"),
            "second_method": second.get("method", ""),
            "second_checkpoint": second.get("checkpoint", ""),
            "second_psnr": second.get("test/psnr", ""),
            "psnr_margin": diff(best, second, "test/psnr") if second else None,
        })
    return sorted(out, key=lambda r: (r["scene"], budget_sort_key(r["budget_label"])))


def collect_fieldnames(rows):
    out = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                out.append(key)
                seen.add(key)
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=collect_fieldnames(rows), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def table(rows, cols, limit=None):
    rows = rows[:limit] if limit else rows
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                val = fmt(val)
            vals.append(str("" if val is None else val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def make_plots(out_dir, eval_rows, best_rows, checkpoint_gain_rows):
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/scratch_local/matplotlib-codex")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plot warning: {exc}", file=sys.stderr)
        return []

    created = []
    colors = {
        "lora_route0": "#1f77b4",
        "lora_route0_dyn": "#ff7f0e",
        "scaffold_lora_route0_noreg": "#2ca02c",
        "scaffold_lora_route0_reg": "#9467bd",
        "scaffold_lora_route0_dyn": "#d62728",
    }
    markers = {PRIMARY_CHECKPOINT: "o", REFERENCE_CHECKPOINT: "^"}

    fig, axes = plt.subplots(1, len(SCENES), figsize=(11, 4), sharey=True)
    if len(SCENES) == 1:
        axes = [axes]
    for ax, scene in zip(axes, SCENES):
        for method in METHODS:
            for ckpt in [PRIMARY_CHECKPOINT, REFERENCE_CHECKPOINT]:
                pts = [r for r in eval_rows if r["scene"] == scene and r["method"] == method and r["checkpoint"] == ckpt]
                pts.sort(key=lambda r: budget_sort_key(r["budget_label"]))
                if not pts:
                    continue
                ax.plot(
                    [BUDGET_POINTS[r["budget_label"]] / 1000 for r in pts],
                    [safe_float(r.get("test/psnr")) for r in pts],
                    color=colors.get(method),
                    linestyle="-" if ckpt == PRIMARY_CHECKPOINT else "--",
                    marker=markers.get(ckpt, "o"),
                    label=f"{method_label(method)} ckpt{ckpt}",
                )
        ax.set_title(scene)
        ax.set_xlabel("target budget (k points)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("test PSNR")
    axes[-1].legend(fontsize=6)
    fig.tight_layout()
    path = out_dir / "checkpoint_psnr_by_scene.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path.name)

    fig, axes = plt.subplots(1, len(SCENES), figsize=(11, 4), sharey=True)
    if len(SCENES) == 1:
        axes = [axes]
    for ax, scene in zip(axes, SCENES):
        for method in METHODS:
            pts = [r for r in best_rows if r["scene"] == scene and r["method"] == method]
            pts.sort(key=lambda r: budget_sort_key(r["budget_label"]))
            ax.plot(
                [BUDGET_POINTS[r["budget_label"]] / 1000 for r in pts],
                [safe_float(r.get("test/psnr")) for r in pts],
                color=colors.get(method),
                marker="o",
                label=method_label(method),
            )
        ax.set_title(scene)
        ax.set_xlabel("target budget (k points)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(f"best-of-{PRIMARY_CHECKPOINT}/{REFERENCE_CHECKPOINT} PSNR")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "best_checkpoint_psnr_by_scene.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path.name)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = []
    values = []
    for method in METHODS:
        group = [r for r in checkpoint_gain_rows if r["method"] == method]
        labels.append(method_label(method))
        values.append(avg(r.get("psnr_primary_minus_reference") for r in group))
    ax.bar(labels, values, color=[colors.get(m) for m in METHODS])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"mean PSNR({PRIMARY_CHECKPOINT}) - PSNR({REFERENCE_CHECKPOINT})")
    ax.set_title("Checkpoint gain by method")
    fig.tight_layout()
    path = out_dir / "checkpoint_gain_by_method.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path.name)
    return created


def write_report(
    out_dir,
    eval_rows,
    best_rows,
    gain_rows,
    winners,
    same_ckpt_summary,
    best_pair_summary,
    within_best_summary,
    primary_audit,
    gate_summary,
    plots,
):
    report = out_dir / "report.md"
    wins_primary = sum(1 for r in gain_rows if r.get("best_checkpoint") == PRIMARY_CHECKPOINT)
    wins_reference = sum(1 for r in gain_rows if r.get("best_checkpoint") == REFERENCE_CHECKPOINT)
    by_method = summarize(gain_rows, ["method_label"], ["psnr_primary_minus_reference"])
    best_means = summarize(best_rows, ["method_label", "budget_label"], ["test/psnr", "test/ssim"])
    with report.open("w") as handle:
        handle.write("# Fixed-budget checkpoint-aware W&B analysis\n\n")
        handle.write(f"- Evaluated cells: `{len(gain_rows)}` scene/method/budget cells.\n")
        handle.write(f"- Checkpoints compared: `{PRIMARY_CHECKPOINT}` primary eval runs and `{REFERENCE_CHECKPOINT}` reference/final runs.\n")
        handle.write(f"- `{PRIMARY_CHECKPOINT}` is best in `{wins_primary}` / `{len(gain_rows)}` cells; `{REFERENCE_CHECKPOINT}` is best in `{wins_reference}` / `{len(gain_rows)}` cells.\n")
        handle.write("- Comparisons remain scene-paired; no cross-scene ranking is used for claims.\n\n")

        handle.write("## Best checkpoint by method\n\n")
        handle.write(table(by_method, ["method_label", "n", "mean_psnr_primary_minus_reference", "positive_psnr_primary_minus_reference", "negative_psnr_primary_minus_reference"]))
        handle.write("\n")

        handle.write("## Primary checkpoint audit\n\n")
        handle.write("One row per `(scene, method, budget)` at the primary checkpoint. Point ratios are relative to the LoRA baseline in the same scene and budget.\n\n")
        handle.write(table(primary_audit, ["scene", "method_label", "budget_label", "realized_points", "point_ratio_vs_lora", "test_psnr", "test_ssim", "test_lpips", "dynamic_mask_psnr", "static_ghost_score", "train_test_gap_psnr", "checkpoint_gain_psnr_primary_minus_reference"], limit=30))
        handle.write("\n")

        handle.write("## Mechanism gate summary\n\n")
        handle.write("A candidate passes the exploratory gate only if mean paired PSNR improves by the configured threshold, SSIM/LPIPS do not regress on average, and realized points stay within the capacity threshold.\n\n")
        handle.write(table(gate_summary, ["candidate_label", "budget_label", "n", "scene_count", "mean_delta_psnr", "positive_delta_psnr", "negative_delta_psnr", "mean_delta_ssim", "mean_delta_lpips_quality", "mean_point_ratio_vs_lora", "max_point_ratio_vs_lora", "passes_all_gates"]))
        handle.write("\n")

        handle.write("## Mean best-checkpoint metrics\n\n")
        handle.write(table(best_means, ["method_label", "budget_label", "n", "mean_test/psnr", "mean_test/ssim"]))
        handle.write("\n")

        handle.write("## Scene-wise winners using best checkpoint\n\n")
        handle.write(table(winners, ["scene", "budget_label", "best_method_label", "best_checkpoint", "best_psnr", "second_method", "second_checkpoint", "second_psnr", "psnr_margin"]))
        handle.write("\n")

        handle.write("## Same-budget method contrasts at fixed checkpoint\n\n")
        handle.write("Positive PSNR means the left side of `comparison` is better.\n\n")
        handle.write(table(same_ckpt_summary, ["analysis_set", "comparison", "budget_label", "n", "scene_count", "mean_delta_psnr", "positive_delta_psnr", "negative_delta_psnr", "mean_delta_ssim", "mean_delta_lpips_quality"]))
        handle.write("\n")

        handle.write("## Same-budget method contrasts using each cell's best checkpoint\n\n")
        handle.write(table(best_pair_summary, ["analysis_set", "comparison", "budget_label", "n", "scene_count", "mean_delta_psnr", "positive_delta_psnr", "negative_delta_psnr", "mean_delta_ssim", "mean_delta_lpips_quality"]))
        handle.write("\n")

        handle.write("## Within-method budget response using best checkpoint\n\n")
        handle.write(table(within_best_summary, ["analysis_set", "method_label", "comparison", "n", "scene_count", "mean_delta_psnr", "positive_delta_psnr", "negative_delta_psnr", "mean_delta_points"]))
        handle.write("\n")

        handle.write("## Interpretation\n\n")
        handle.write(f"- Treat `{PRIMARY_CHECKPOINT}` as the primary checkpoint for claims; best-checkpoint rows are exploratory unless a true validation split is introduced.\n")
        if gate_summary:
            passed = [r for r in gate_summary if r.get("passes_all_gates") is True]
            handle.write(f"- Gate pass count: `{len(passed)}` / `{len(gate_summary)}` candidate comparisons.\n")
        handle.write("- Capacity control is based on realized point ratios, not only target point budgets.\n\n")

        handle.write("## Files\n\n")
        for name in [
            "eval_checkpoints.csv",
            "best_checkpoint_per_cell.csv",
            "checkpoint_gains.csv",
            "best_winners.csv",
            "same_checkpoint_method_pairs.csv",
            "same_checkpoint_method_summary.csv",
            "best_checkpoint_method_pairs.csv",
            "best_checkpoint_method_summary.csv",
            "primary_checkpoint_audit.csv",
            "primary_checkpoint_gate_summary.csv",
            "within_method_budget_best_pairs.csv",
            "within_method_budget_best_summary.csv",
            "training_history_features.csv",
            "history_series_sampled.csv",
        ] + plots:
            handle.write(f"- `{name}`\n")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Checkpoint-aware fixed-budget W&B analysis.")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "models-ku-leuven"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "adags"))
    parser.add_argument("--runs-root", default="/leonardo_work/EUHPC_D21_034/proj_adags/runs")
    parser.add_argument("--out-dir", default="analysis/wandb_fixed_budget_checkpoints")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--history-page-size", type=int, default=1000)
    parser.add_argument("--max-plot-points-per-series", type=int, default=300)
    parser.add_argument("--skip-history", action="store_true")
    parser.add_argument("--history-keys", nargs="+", default=HISTORY_KEYS)
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--budgets", nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--baseline-method", default=BASELINE_METHOD)
    parser.add_argument("--primary-checkpoint", default=PRIMARY_CHECKPOINT)
    parser.add_argument("--reference-checkpoint", default=REFERENCE_CHECKPOINT)
    parser.add_argument("--gate-budget", default="600k")
    parser.add_argument("--gate-psnr-threshold", type=float, default=0.2)
    parser.add_argument("--gate-point-ratio-threshold", type=float, default=1.10)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    configure_scope(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import wandb

    api = wandb.Api(timeout=args.timeout)
    path = f"{args.entity}/{args.project}"
    base_ids = find_base_run_ids(args.runs_root)
    print(f"Found {len(base_ids)} base fixed-budget run ids.", flush=True)

    eval_rows = []
    base_rows = []
    run_objects = {}
    missing = []
    for base_id in base_ids:
        try:
            base_run = api.run(f"{path}/{base_id}")
            run_objects[base_id] = base_run
            base_checkpoint = infer_run_checkpoint(base_run, REFERENCE_CHECKPOINT)
            base_row = fetch_summary_row(args, base_run, base_id, base_checkpoint)
            if not row_in_scope(base_row):
                continue
            eval_rows.append(base_row)
            base_rows.append(base_row)
        except Exception as exc:
            missing.append((base_id, repr(exc)))
            continue

        if base_row.get("checkpoint") == PRIMARY_CHECKPOINT:
            continue

        eval6000_id = base_id + "_eval6000"
        try:
            eval6000_run = api.run(f"{path}/{eval6000_id}")
            primary_row = fetch_summary_row(args, eval6000_run, base_id, PRIMARY_CHECKPOINT)
            if row_in_scope(primary_row):
                eval_rows.append(primary_row)
        except Exception as exc:
            missing.append((eval6000_id, repr(exc)))

    if missing:
        print("missing runs:", file=sys.stderr)
        for run_id, err in missing:
            print(f"  {run_id}: {err}", file=sys.stderr)

    if args.skip_history:
        history_rows = []
        sampled_rows = []
    else:
        history_rows, sampled_rows = collect_training_history(base_rows, run_objects, args)
        eval_rows = attach_history(eval_rows, history_rows)

    eval_rows.sort(key=lambda r: (r["scene"], method_sort_key(r["method"]), budget_sort_key(r["budget_label"]), r["checkpoint"]))
    best_rows = best_checkpoint_rows(eval_rows)
    gains = checkpoint_gains(eval_rows)
    winners = best_winners(best_rows)

    same_ckpt_pairs = []
    for ckpt in [PRIMARY_CHECKPOINT, REFERENCE_CHECKPOINT]:
        same_ckpt_pairs.extend(method_pairs([r for r in eval_rows if r["checkpoint"] == ckpt], f"ckpt{ckpt}"))
    same_ckpt_summary = summarize(
        same_ckpt_pairs,
        ["analysis_set", "comparison", "budget_label"],
        ["delta_psnr", "delta_ssim", "delta_lpips_quality", "delta_dynamic_mask_psnr", "delta_static_ghost_quality", "delta_points"],
    )
    same_ckpt_summary.sort(key=lambda r: (r["analysis_set"], r["comparison"], budget_sort_key(r["budget_label"])))

    best_pair_rows = method_pairs(best_rows, "best_checkpoint")
    best_pair_summary = summarize(
        best_pair_rows,
        ["analysis_set", "comparison", "budget_label"],
        ["delta_psnr", "delta_ssim", "delta_lpips_quality", "delta_dynamic_mask_psnr", "delta_static_ghost_quality", "delta_points"],
    )
    best_pair_summary.sort(key=lambda r: (r["comparison"], budget_sort_key(r["budget_label"])))

    within_best = within_budget_rows(best_rows, "best_checkpoint")
    within_best_summary = summarize(within_best, ["analysis_set", "method_label", "comparison"], ["delta_psnr", "delta_ssim", "delta_lpips_quality", "delta_points"])
    within_best_summary.sort(key=lambda r: (r["method_label"], r["comparison"]))

    primary_audit = primary_checkpoint_audit(eval_rows)
    primary_pairs = method_pairs([r for r in eval_rows if r["checkpoint"] == PRIMARY_CHECKPOINT], f"ckpt{PRIMARY_CHECKPOINT}")
    gate_summary = build_gate_summary(
        primary_pairs,
        args.gate_budget,
        args.gate_psnr_threshold,
        args.gate_point_ratio_threshold,
    )
    plots = make_plots(out_dir, eval_rows, best_rows, gains)

    write_csv(out_dir / "eval_checkpoints.csv", eval_rows)
    write_csv(out_dir / "best_checkpoint_per_cell.csv", best_rows)
    write_csv(out_dir / "checkpoint_gains.csv", gains)
    write_csv(out_dir / "best_winners.csv", winners)
    write_csv(out_dir / "same_checkpoint_method_pairs.csv", same_ckpt_pairs)
    write_csv(out_dir / "same_checkpoint_method_summary.csv", same_ckpt_summary)
    write_csv(out_dir / "best_checkpoint_method_pairs.csv", best_pair_rows)
    write_csv(out_dir / "best_checkpoint_method_summary.csv", best_pair_summary)
    write_csv(out_dir / "primary_checkpoint_audit.csv", primary_audit)
    write_csv(out_dir / "primary_checkpoint_gate_summary.csv", gate_summary)
    write_csv(out_dir / "within_method_budget_best_pairs.csv", within_best)
    write_csv(out_dir / "within_method_budget_best_summary.csv", within_best_summary)
    write_csv(out_dir / "training_history_features.csv", history_rows)
    write_csv(out_dir / "history_series_sampled.csv", sampled_rows)
    write_report(
        out_dir,
        eval_rows,
        best_rows,
        gains,
        winners,
        same_ckpt_summary,
        best_pair_summary,
        within_best_summary,
        primary_audit,
        gate_summary,
        plots,
    )

    print(f"Wrote checkpoint-aware analysis to {out_dir}")
    print(f"eval_rows={len(eval_rows)} best_rows={len(best_rows)} checkpoint_gains={len(gains)} missing={len(missing)}")
    if plots:
        print("plots:", ", ".join(plots))


if __name__ == "__main__":
    main()
