#!/usr/bin/env python3
"""Scene-paired fixed-budget W&B analysis for ADAGS experiments."""

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


METHODS = [
    "lora_route0",
    "lora_route0_dyn",
    "scaffold_lora_route0_noreg",
    "scaffold_lora_route0_reg",
    "scaffold_lora_route0_dyn",
]

METHOD_LABELS = {
    "lora_route0": "lora",
    "lora_route0_dyn": "lora_dyn",
    "scaffold_lora_route0_noreg": "scaffold_noreg",
    "scaffold_lora_route0_reg": "scaffold_reg",
    "scaffold_lora_route0_dyn": "scaffold_dyn",
}

BUDGETS = ["400k", "600k", "800k"]
BUDGET_POINTS = {"400k": 400000, "600k": 600000, "800k": 800000}
SCENES = ["cut_roasted_beef", "flame_steak", "sear_steak"]

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
    "routing/entropy",
    "routing/expected_static_points",
    "routing/expected_dynamic_points",
    "routing/percent_uncertain",
    "motion_lora/coeff_norm_mean",
    "motion_lora/basis_norm_mean",
    "motion_scaffold/node_count",
    "motion_scaffold/coeff_norm_mean",
    "motion_scaffold/basis_norm_mean",
    "motion_scaffold/attach_entropy",
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
    "points/hard_static",
    "routing/entropy",
    "routing/expected_static_points",
    "routing/percent_uncertain",
    "motion_lora/coeff_norm_mean",
    "motion_lora/basis_norm_mean",
    "motion_scaffold/coeff_norm_mean",
    "motion_scaffold/basis_norm_mean",
    "motion_scaffold/attach_entropy",
]

CONFIG_KEYS = [
    "method_family",
    "budget_label",
    "iterations",
    "densify_until_num_points",
    "densify_from_iter",
    "densify_until_iter",
    "densification_interval",
    "opacity_reset_interval",
    "position_lr_max_steps",
    "motion_model",
    "motion_lora_rank",
    "motion_lora_anchors",
    "motion_lora_coeff_lr",
    "motion_lora_basis_lr",
    "lambda_motion_gate",
    "lambda_gate_sparsity",
    "enable_soft_routing",
    "route_lr",
    "route_logit_init",
    "motion_scaffold_enable",
    "motion_scaffold_count",
    "motion_scaffold_rank",
    "motion_scaffold_anchors",
    "motion_scaffold_knn",
    "motion_scaffold_init_scale",
    "motion_scaffold_weight_temp",
    "motion_scaffold_coeff_lr",
    "motion_scaffold_basis_lr",
    "lambda_scaffold_smooth",
    "lambda_scaffold_reg",
    "enable_motion_aware_densify",
    "motion_aware_densify_boost",
    "enable_hard_static_conversion",
    "static_conversion_threshold",
]

PAIR_DEFS = [
    ("lora_dyn_minus_lora", "lora_route0_dyn", "lora_route0"),
    ("scaffold_noreg_minus_lora", "scaffold_lora_route0_noreg", "lora_route0"),
    ("scaffold_reg_minus_lora", "scaffold_lora_route0_reg", "lora_route0"),
    ("scaffold_dyn_minus_lora", "scaffold_lora_route0_dyn", "lora_route0"),
    ("scaffold_dyn_minus_noreg", "scaffold_lora_route0_dyn", "scaffold_lora_route0_noreg"),
]

DELTA_METRICS = [
    ("test/psnr", "delta_psnr", 1.0),
    ("test/ssim", "delta_ssim", 1.0),
    ("test/lpips", "delta_lpips_quality", -1.0),
    ("test/dynamic_mask_psnr", "delta_dynamic_mask_psnr", 1.0),
    ("test/static_ghost_score", "delta_static_ghost_quality", -1.0),
    ("points/total", "delta_points_total", 1.0),
    ("points/hard_static", "delta_hard_static", 1.0),
    ("routing/entropy", "delta_routing_entropy", 1.0),
    ("routing/expected_static_points", "delta_expected_static_points", 1.0),
    ("routing/percent_uncertain", "delta_percent_uncertain", 1.0),
    ("motion_scaffold/coeff_norm_mean", "delta_scaffold_coeff_norm", 1.0),
    ("motion_scaffold/attach_entropy", "delta_scaffold_attach_entropy", 1.0),
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


def avg(values):
    clean = [safe_float(v) for v in values]
    clean = [v for v in clean if v is not None]
    return mean(clean) if clean else None


def fmt(value, digits=4):
    value = safe_float(value)
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def metric_key(metric, feature=None):
    key = metric.replace("/", "__")
    return f"{key}__{feature}" if feature else key


def budget_sort_key(budget):
    return BUDGETS.index(budget) if budget in BUDGETS else 999


def method_sort_key(method):
    return METHODS.index(method) if method in METHODS else 999


def find_local_fixed_budget_run_ids(runs_root):
    if not runs_root:
        return []
    pattern = os.path.join(
        runs_root,
        "fixed_budget_*",
        "*fixed_budget_*",
        "wandb",
        "offline-run-*",
    )
    run_ids = set()
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
        run_ids.add(run_id)
    return sorted(run_ids)


def infer_from_run_name(run_name):
    scene = None
    for candidate in SCENES:
        if candidate in run_name:
            scene = candidate
            break

    method = None
    for candidate in sorted(METHODS, key=len, reverse=True):
        if f"fixed_budget_{candidate}_" in run_name:
            method = candidate
            break

    budget = None
    match = re.search(r"_(400k|600k|800k)$", run_name)
    if match:
        budget = match.group(1)
    return scene, method, budget


def infer_metadata(run):
    tags = list(getattr(run, "tags", []) or [])
    tag_set = set(tags)
    config = dict(getattr(run, "config", {}) or {})
    name = getattr(run, "name", "") or getattr(run, "id", "")

    scene = getattr(run, "group", None)
    if scene not in SCENES:
        scene = next((s for s in SCENES if s in tag_set), None)

    method = config.get("method_family")
    if method not in METHODS:
        method = next((t.split(":", 1)[1] for t in tags if t.startswith("method:")), None)

    budget = config.get("budget_label")
    if budget not in BUDGETS:
        budget = next((t.split(":", 1)[1] for t in tags if t.startswith("budget:")), None)

    name_scene, name_method, name_budget = infer_from_run_name(name)
    return {
        "scene": scene or name_scene,
        "method": method or name_method,
        "method_label": METHOD_LABELS.get(method or name_method, method or name_method),
        "budget_label": budget or name_budget,
        "budget_points": BUDGET_POINTS.get(budget or name_budget),
        "tags": ",".join(tags),
    }


def fetch_runs(args):
    import wandb

    api = wandb.Api(timeout=args.timeout)
    path = f"{args.entity}/{args.project}"
    run_ids = []
    if args.run_id_source in ("auto", "local"):
        run_ids = find_local_fixed_budget_run_ids(args.local_runs_root)
    rows = []
    run_objects = {}

    if run_ids:
        print(f"Using {len(run_ids)} fixed-budget run ids from local offline run folders.", flush=True)
        for run_id in run_ids:
            run = api.run(f"{path}/{run_id}")
            meta = infer_metadata(run)
            if meta["scene"] in SCENES and meta["method"] in METHODS and meta["budget_label"] in BUDGETS:
                row = build_run_row(args, run, meta)
                rows.append(row)
                run_objects[run.id] = run
    else:
        print("Local run ids not found; falling back to W&B tag/name filtering.", flush=True)
        runs = api.runs(
            path,
            filters={"state": "finished", "tags": {"$in": ["fixed_budget"]}},
            order="-created_at",
            per_page=100,
            include_sweeps=False,
        )
        for run in runs[:100]:
            meta = infer_metadata(run)
            if meta["scene"] in SCENES and meta["method"] in METHODS and meta["budget_label"] in BUDGETS:
                row = build_run_row(args, run, meta)
                rows.append(row)
                run_objects[run.id] = run

    rows.sort(key=lambda r: (r["scene"], method_sort_key(r["method"]), budget_sort_key(r["budget_label"])))
    return rows, run_objects


def build_run_row(args, run, meta):
    summary = dict(getattr(run, "summary_metrics", None) or {})
    config = dict(getattr(run, "config", {}) or {})
    row = {
        "run_id": run.id,
        "run_name": getattr(run, "name", run.id),
        "url": f"https://wandb.ai/{args.entity}/{args.project}/runs/{run.id}",
        "state": getattr(run, "state", ""),
        "created_at": getattr(run, "created_at", ""),
        "group": getattr(run, "group", ""),
        "last_history_step": getattr(run, "lastHistoryStep", None),
        **meta,
    }
    for key in SUMMARY_KEYS:
        row[key] = summary.get(key)
    for key in CONFIG_KEYS:
        row[f"config/{key}"] = config.get(key)
    return row


def fetch_series(run, key, page_size):
    series = []
    for item in run.scan_history(keys=["_step", key], page_size=page_size):
        value = safe_float(item.get(key))
        step = safe_float(item.get("_step"))
        if value is not None:
            series.append((int(step) if step is not None else None, value))
    series.sort(key=lambda pair: -1 if pair[0] is None else pair[0])
    return series


def value_at_or_before(series, target):
    selected = None
    for step, value in series:
        if step is not None and step <= target:
            selected = value
    return selected


def auc(series):
    clean = [(s, v) for s, v in series if s is not None and v is not None]
    if len(clean) < 2:
        return None
    span = clean[-1][0] - clean[0][0]
    if span <= 0:
        return None
    area = 0.0
    for (s0, v0), (s1, v1) in zip(clean[:-1], clean[1:]):
        area += 0.5 * (v0 + v1) * (s1 - s0)
    return area / span


def tail_slope_per_1k(series):
    clean = [(s, v) for s, v in series if s is not None and v is not None]
    if len(clean) < 4:
        return None
    tail = clean[max(0, int(len(clean) * 0.8)) :]
    if len(tail) < 2:
        tail = clean[-2:]
    xs = [float(s) for s, _ in tail]
    ys = [float(v) for _, v in tail]
    mx = mean(xs)
    my = mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return slope * 1000.0


def summarize_series(series):
    clean = [(s, v) for s, v in series if v is not None]
    if not clean:
        return {"count": 0}
    values = [v for _, v in clean]
    steps = [s for s, _ in clean if s is not None]
    out = {
        "count": len(clean),
        "first": values[0],
        "final": values[-1],
        "min": min(values),
        "max": max(values),
        "delta": values[-1] - values[0],
        "auc": auc(clean),
        "tail_slope_per_1k": tail_slope_per_1k(clean),
    }
    if steps:
        out["first_step"] = steps[0]
        out["final_step"] = steps[-1]
    for target in (3000, 4500, 6000, 7500, 9000):
        out[f"at_{target}"] = value_at_or_before(clean, target)
    return out


def collect_history(rows, run_objects, args):
    history_rows = []
    sampled_rows = []
    for idx, row in enumerate(rows, 1):
        run = run_objects[row["run_id"]]
        print(f"[{idx}/{len(rows)}] history {row['scene']} {row['method_label']} {row['budget_label']}", flush=True)
        hrow = {
            "run_id": row["run_id"],
            "run_name": row["run_name"],
            "scene": row["scene"],
            "method": row["method"],
            "method_label": row["method_label"],
            "budget_label": row["budget_label"],
        }
        for key in args.history_keys:
            try:
                series = fetch_series(run, key, args.history_page_size)
            except Exception as exc:
                print(f"  history warning: {row['run_id']} {key}: {exc}", file=sys.stderr, flush=True)
                series = []
            summary = summarize_series(series)
            safe_key = metric_key(key)
            for feature, value in summary.items():
                hrow[f"{safe_key}__{feature}"] = value
            sampled_rows.extend(sample_series(row, key, series, args.max_plot_points_per_series))
        history_rows.append(hrow)
    return history_rows, sampled_rows


def sample_series(row, key, series, max_points):
    if not series or max_points <= 0:
        return []
    stride = max(1, len(series) // max_points)
    sampled = series[::stride]
    if sampled[-1] != series[-1]:
        sampled.append(series[-1])
    return [
        {
            "run_id": row["run_id"],
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


def index_rows(rows):
    return {(r["scene"], r["method"], r["budget_label"]): r for r in rows}


def delta_value(left, right, metric, sign=1.0):
    lv = safe_float(left.get(metric))
    rv = safe_float(right.get(metric))
    if lv is None or rv is None:
        return None
    return sign * (lv - rv)


def build_budget_response(rows):
    return sorted(rows, key=lambda r: (r["scene"], method_sort_key(r["method"]), budget_sort_key(r["budget_label"])))


def build_within_method_deltas(rows, history_by_run):
    by_key = index_rows(rows)
    out = []
    comparisons = [("600k_minus_400k", "600k", "400k"), ("800k_minus_600k", "800k", "600k"), ("800k_minus_400k", "800k", "400k")]
    for scene in SCENES:
        for method in METHODS:
            for label, high_budget, low_budget in comparisons:
                high = by_key.get((scene, method, high_budget))
                low = by_key.get((scene, method, low_budget))
                if not high or not low:
                    continue
                row = {
                    "scene": scene,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "comparison": label,
                    "high_budget": high_budget,
                    "low_budget": low_budget,
                    "high_run_id": high["run_id"],
                    "low_run_id": low["run_id"],
                }
                for metric, out_name, sign in DELTA_METRICS[:8]:
                    row[out_name] = delta_value(high, low, metric, sign)
                add_history_delta(row, high, low, history_by_run, "train/psnr", "final", "delta_train_psnr_final")
                add_history_delta(row, high, low, history_by_run, "train/total_loss", "final", "delta_train_total_loss_final")
                add_history_delta(row, high, low, history_by_run, "routing/entropy", "final", "delta_routing_entropy_final")
                add_history_delta(row, high, low, history_by_run, "routing/expected_static_points", "final", "delta_expected_static_final")
                out.append(row)
    return out


def add_history_delta(out, left, right, history_by_run, metric, feature, out_name):
    left_value = history_value(history_by_run, left["run_id"], metric, feature)
    right_value = history_value(history_by_run, right["run_id"], metric, feature)
    if left_value is not None and right_value is not None:
        out[out_name] = left_value - right_value
    else:
        out[out_name] = None


def history_value(history_by_run, run_id, metric, feature):
    row = history_by_run.get(run_id, {})
    return safe_float(row.get(metric_key(metric, feature)))


def build_method_pairs(rows):
    by_key = index_rows(rows)
    pairs = []
    for scene in SCENES:
        for budget in BUDGETS:
            for pair_label, left_method, right_method in PAIR_DEFS:
                left = by_key.get((scene, left_method, budget))
                right = by_key.get((scene, right_method, budget))
                if not left or not right:
                    continue
                row = {
                    "scene": scene,
                    "budget_label": budget,
                    "budget_points": BUDGET_POINTS[budget],
                    "comparison": pair_label,
                    "left_method": left_method,
                    "right_method": right_method,
                    "left_run_id": left["run_id"],
                    "right_run_id": right["run_id"],
                }
                for metric, out_name, sign in DELTA_METRICS:
                    row[out_name] = delta_value(left, right, metric, sign)
                    row[f"left_{metric.replace('/', '_')}"] = left.get(metric)
                    row[f"right_{metric.replace('/', '_')}"] = right.get(metric)
                pairs.append(row)
    return pairs


def summarize_group(rows, group_keys, value_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in group_keys)].append(row)
    out = []
    for key, group in groups.items():
        row = {group_key: key[i] for i, group_key in enumerate(group_keys)}
        row["n"] = len(group)
        if "scene" not in group_keys:
            row["scene_count"] = len(set(g.get("scene") for g in group if g.get("scene")))
        for value_key in value_keys:
            values = [g.get(value_key) for g in group]
            row[f"mean_{value_key}"] = avg(values)
            clean = [safe_float(v) for v in values if safe_float(v) is not None]
            row[f"positive_{value_key}"] = sum(1 for v in clean if v > 0)
            row[f"negative_{value_key}"] = sum(1 for v in clean if v < 0)
        out.append(row)
    return out


def build_best_by_scene_budget(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["budget_label"])].append(row)
    out = []
    for (scene, budget), group in grouped.items():
        ranked = sorted(group, key=lambda r: safe_float(r.get("test/psnr")) if safe_float(r.get("test/psnr")) is not None else -1e9, reverse=True)
        if not ranked:
            continue
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        row = {
            "scene": scene,
            "budget_label": budget,
            "best_method": best["method"],
            "best_method_label": best["method_label"],
            "best_psnr": best.get("test/psnr"),
            "best_ssim": best.get("test/ssim"),
            "best_lpips": best.get("test/lpips"),
            "best_run_id": best["run_id"],
        }
        if second:
            row["second_method"] = second["method"]
            row["second_psnr"] = second.get("test/psnr")
            row["psnr_margin_to_second"] = delta_value(best, second, "test/psnr")
        out.append(row)
    return sorted(out, key=lambda r: (r["scene"], budget_sort_key(r["budget_label"])))


def build_config_differences(rows):
    out = []
    for key in CONFIG_KEYS:
        values = defaultdict(set)
        for row in rows:
            label = f"{row['method_label']}_{row['budget_label']}"
            value = row.get(f"config/{key}")
            values[label].add(str(value))
        flattened = {label: "|".join(sorted(v)) for label, v in values.items()}
        if len(set(flattened.values())) <= 1:
            continue
        out.append({"config_key": key, **flattened})
    return out


def build_metric_availability(rows, history_rows):
    out = []
    for key in SUMMARY_KEYS:
        out.append({
            "metric": key,
            "source": "summary",
            "run_count_with_metric": sum(1 for r in rows if safe_float(r.get(key)) is not None),
            "total_runs": len(rows),
        })
    for key in HISTORY_KEYS:
        count_key = metric_key(key, "count")
        out.append({
            "metric": key,
            "source": "history",
            "run_count_with_metric": sum(1 for r in history_rows if safe_float(r.get(count_key))),
            "total_runs": len(history_rows),
        })
    return out


def collect_fieldnames(rows):
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=collect_fieldnames(rows), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def table_lines(rows, columns, limit=None):
    rows = rows[:limit] if limit else rows
    if not rows:
        return ["_No rows._"]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = fmt(val, 4)
            vals.append(str(val if val is not None else ""))
        out.append("| " + " | ".join(vals) + " |")
    return out


def method_budget_means(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["budget_label"])].append(row)
    out = []
    for (method, budget), group in grouped.items():
        out.append({
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "budget_label": budget,
            "n": len(group),
            "mean_psnr": avg(g.get("test/psnr") for g in group),
            "mean_ssim": avg(g.get("test/ssim") for g in group),
            "mean_lpips": avg(g.get("test/lpips") for g in group),
            "mean_points": avg(g.get("points/total") for g in group),
        })
    return sorted(out, key=lambda r: (method_sort_key(r["method"]), budget_sort_key(r["budget_label"])))


def write_report(out_dir, rows, history_rows, within_deltas, within_summary, method_pairs, method_pair_summary, best_rows, config_diffs, availability):
    report = out_dir / "report.md"
    history_by_run = {r["run_id"]: r for r in history_rows}
    means = method_budget_means(rows)

    positive_pairs = [
        r for r in method_pair_summary
        if str(r.get("comparison", "")).endswith("_minus_lora")
    ]
    overfit_rows = [
        r for r in within_deltas
        if safe_float(r.get("delta_train_psnr_final")) is not None
        and safe_float(r.get("delta_psnr")) is not None
        and safe_float(r.get("delta_train_psnr_final")) > 0
        and safe_float(r.get("delta_psnr")) < 0
    ]

    test_steps = sorted(set(
        int(v)
        for h in history_rows
        for v in [safe_float(h.get(metric_key("test/psnr", "final_step")))]
        if v is not None
    ))
    train_steps = sorted(set(
        int(v)
        for h in history_rows
        for v in [safe_float(h.get(metric_key("train/psnr", "final_step")))]
        if v is not None
    ))

    with report.open("w") as handle:
        handle.write("# Fixed-budget W&B analysis\n\n")
        handle.write(f"- Runs analyzed: `{len(rows)}` (`{len(SCENES)}` scenes x `{len(METHODS)}` methods x `{len(BUDGETS)}` budgets).\n")
        handle.write("- Unit of comparison: paired `(scene, budget)` or paired `(scene, method)` only; no cross-scene ranking is used for claims.\n")
        handle.write(f"- Test PSNR final history step(s): `{test_steps}`. Train PSNR final history step(s): `{train_steps}`.\n")
        handle.write("- Methods in scope: `lora_route0`, `lora_route0_dyn`, `scaffold_lora_route0_noreg`, `scaffold_lora_route0_reg`, `scaffold_lora_route0_dyn`.\n\n")

        handle.write("## Mean metrics by method and budget\n\n")
        handle.write("\n".join(table_lines(means, ["method_label", "budget_label", "n", "mean_psnr", "mean_ssim", "mean_lpips", "mean_points"])))
        handle.write("\n\n")

        handle.write("## Scene-wise winners at matched budget\n\n")
        handle.write("\n".join(table_lines(best_rows, ["scene", "budget_label", "best_method_label", "best_psnr", "second_method", "second_psnr", "psnr_margin_to_second"])))
        handle.write("\n\n")

        handle.write("## Same-budget method contrasts\n\n")
        handle.write("Positive `mean_delta_psnr` means the left method in `comparison` is better. `positive_delta_psnr` counts scenes where the delta is positive.\n\n")
        handle.write("\n".join(table_lines(method_pair_summary, ["comparison", "budget_label", "n", "scene_count", "mean_delta_psnr", "positive_delta_psnr", "negative_delta_psnr", "mean_delta_ssim", "mean_delta_lpips_quality", "mean_delta_dynamic_mask_psnr", "mean_delta_static_ghost_quality", "mean_delta_points_total"])))
        handle.write("\n\n")

        handle.write("## Within-method budget response\n\n")
        handle.write("Positive `mean_delta_psnr` means the higher budget improved over the lower budget within the same method and scene.\n\n")
        handle.write("\n".join(table_lines(within_summary, ["method_label", "comparison", "n", "scene_count", "mean_delta_psnr", "positive_delta_psnr", "negative_delta_psnr", "mean_delta_train_psnr_final", "mean_delta_train_total_loss_final"])))
        handle.write("\n\n")

        handle.write("## Curve-derived interpretation\n\n")
        if overfit_rows:
            handle.write("- Cases where capacity increased, final train PSNR increased, but final test PSNR decreased:\n")
            for row in overfit_rows:
                handle.write(
                    f"  - `{row['scene']}` `{row['method_label']}` `{row['comparison']}`: "
                    f"test PSNR {fmt(row['delta_psnr'])}, train PSNR {fmt(row['delta_train_psnr_final'])}, "
                    f"train loss {fmt(row['delta_train_total_loss_final'])}.\n"
                )
        else:
            handle.write("- No within-method budget comparison showed higher final train PSNR with lower final test PSNR.\n")

        for row in positive_pairs:
            handle.write(
                f"- `{row['comparison']}` at `{row['budget_label']}`: mean PSNR delta `{fmt(row.get('mean_delta_psnr'))}` "
                f"across `{row.get('scene_count')}` scenes; sign split +`{row.get('positive_delta_psnr')}`/-`{row.get('negative_delta_psnr')}`.\n"
            )

        scaffold_coeff_key = metric_key("motion_scaffold/coeff_norm_mean", "final")
        attach_key = metric_key("motion_scaffold/attach_entropy", "final")
        scaffold_rows = [
            {
                "method_label": r["method_label"],
                "budget_label": r["budget_label"],
                "scene": r["scene"],
                "coeff": history_by_run.get(r["run_id"], {}).get(scaffold_coeff_key),
                "attach_entropy": history_by_run.get(r["run_id"], {}).get(attach_key),
            }
            for r in rows if r["method"] != "lora_route0"
        ]
        coeff_avg = avg(r["coeff"] for r in scaffold_rows)
        attach_avg = avg(r["attach_entropy"] for r in scaffold_rows)
        handle.write(f"- Scaffold curve metrics are present for scaffold methods. Mean final scaffold coeff norm: `{fmt(coeff_avg)}`; mean final attach entropy: `{fmt(attach_avg)}`.\n")

        unavailable = [r for r in availability if int(r.get("run_count_with_metric", 0)) == 0]
        if unavailable:
            handle.write("- Metrics absent from this fixed-budget set: " + ", ".join(f"`{r['metric']}`/{r['source']}" for r in unavailable[:12]) + ".\n")
        handle.write("\n")

        handle.write("## Config differences that matter for this comparison\n\n")
        handle.write("These are config keys that vary across the fixed-budget cells. Runtime/source-path metadata is not included.\n\n")
        handle.write("\n".join(table_lines(config_diffs, collect_fieldnames(config_diffs)[:10], limit=30)))
        handle.write("\n\n")

        handle.write("## Files\n\n")
        for name in [
            "runs.csv",
            "history_features.csv",
            "history_series_sampled.csv",
            "within_method_budget_deltas.csv",
            "within_method_budget_summary.csv",
            "same_budget_method_pairs.csv",
            "same_budget_method_summary.csv",
            "best_by_scene_budget.csv",
            "config_differences.csv",
            "metric_availability.csv",
        ]:
            handle.write(f"- `{name}`\n")


def make_plots(out_dir, rows, sampled_rows):
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/scratch_local/matplotlib-codex")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plot warning: matplotlib unavailable: {exc}", file=sys.stderr)
        return []

    created = []
    colors = {
        "lora_route0": "#1f77b4",
        "lora_route0_dyn": "#ff7f0e",
        "scaffold_lora_route0_noreg": "#2ca02c",
        "scaffold_lora_route0_reg": "#9467bd",
        "scaffold_lora_route0_dyn": "#d62728",
    }
    markers = {"400k": "o", "600k": "s", "800k": "^"}

    fig, axes = plt.subplots(1, len(SCENES), figsize=(11, 4), sharey=True)
    if len(SCENES) == 1:
        axes = [axes]
    for ax, scene in zip(axes, SCENES):
        for method in METHODS:
            pts = [r for r in rows if r["scene"] == scene and r["method"] == method]
            pts.sort(key=lambda r: budget_sort_key(r["budget_label"]))
            if not pts:
                continue
            ax.plot(
                [BUDGET_POINTS[r["budget_label"]] / 1000 for r in pts],
                [safe_float(r.get("test/psnr")) for r in pts],
                marker="o",
                color=colors.get(method),
                label=METHOD_LABELS.get(method, method),
            )
        ax.set_title(scene)
        ax.set_xlabel("budget target (k points)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("final test PSNR")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "psnr_by_budget_scene_method.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path.name)

    by_key = index_rows(rows)
    fig, axes = plt.subplots(1, len(SCENES), figsize=(11, 4), sharey=True)
    if len(SCENES) == 1:
        axes = [axes]
    for ax, scene in zip(axes, SCENES):
        for method in METHODS[1:]:
            vals = []
            xs = []
            for budget in BUDGETS:
                left = by_key.get((scene, method, budget))
                right = by_key.get((scene, "lora_route0", budget))
                if left and right:
                    xs.append(BUDGET_POINTS[budget] / 1000)
                    vals.append(delta_value(left, right, "test/psnr"))
            ax.plot(xs, vals, marker="o", color=colors.get(method), label=METHOD_LABELS.get(method, method))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(scene)
        ax.set_xlabel("budget target (k points)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("delta PSNR vs lora")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "delta_psnr_vs_lora.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path.name)

    for metric, filename, ylabel in [
        ("test/psnr", "curves_test_psnr.png", "test PSNR"),
        ("train/total_loss", "curves_train_total_loss.png", "train total loss"),
        ("routing/entropy", "curves_routing_entropy.png", "routing entropy"),
    ]:
        metric_rows = [r for r in sampled_rows if r["metric"] == metric]
        if not metric_rows:
            continue
        fig, axes = plt.subplots(1, len(SCENES), figsize=(12, 4), sharey=False)
        if len(SCENES) == 1:
            axes = [axes]
        for ax, scene in zip(axes, SCENES):
            for method in METHODS:
                for budget in BUDGETS:
                    series = [
                        r for r in metric_rows
                        if r["scene"] == scene and r["method"] == method and r["budget_label"] == budget
                    ]
                    if not series:
                        continue
                    series.sort(key=lambda r: safe_float(r["step"]) or -1)
                    ax.plot(
                        [safe_float(r["step"]) for r in series],
                        [safe_float(r["value"]) for r in series],
                        color=colors.get(method),
                        alpha=0.45 + 0.15 * budget_sort_key(budget),
                        marker=markers[budget],
                        markevery=max(1, len(series) // 6),
                        linewidth=1.0,
                        markersize=3,
                        label=f"{METHOD_LABELS.get(method, method)} {budget}",
                    )
            ax.set_title(scene)
            ax.set_xlabel("iteration")
            ax.grid(True, alpha=0.2)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=6, ncol=1)
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path.name)
    return created


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Analyze synced fixed-budget ADAGS W&B runs.")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "models-ku-leuven"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "adags"))
    parser.add_argument("--local-runs-root", default="/leonardo_work/EUHPC_D21_034/proj_adags/runs")
    parser.add_argument("--run-id-source", choices=["auto", "local", "wandb-tag"], default="auto")
    parser.add_argument("--out-dir", default="analysis/wandb_fixed_budget")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--history-page-size", type=int, default=1000)
    parser.add_argument("--max-plot-points-per-series", type=int, default=500)
    parser.add_argument("--skip-history", action="store_true")
    parser.add_argument("--history-keys", nargs="+", default=HISTORY_KEYS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, run_objects = fetch_runs(args)
    if not rows:
        raise SystemExit("No fixed-budget runs found.")
    expected = len(SCENES) * len(METHODS) * len(BUDGETS)
    if len(rows) != expected:
        print(f"warning: expected {expected} scoped runs, found {len(rows)}", file=sys.stderr)

    if args.skip_history:
        history_rows = []
        sampled_rows = []
    else:
        history_rows, sampled_rows = collect_history(rows, run_objects, args)

    history_by_run = {r["run_id"]: r for r in history_rows}
    budget_response = build_budget_response(rows)
    within_deltas = build_within_method_deltas(rows, history_by_run)
    within_summary = summarize_group(
        within_deltas,
        ["method", "method_label", "comparison"],
        ["delta_psnr", "delta_ssim", "delta_lpips_quality", "delta_train_psnr_final", "delta_train_total_loss_final"],
    )
    method_pairs = build_method_pairs(rows)
    method_pair_summary = summarize_group(
        method_pairs,
        ["comparison", "budget_label"],
        [
            "delta_psnr",
            "delta_ssim",
            "delta_lpips_quality",
            "delta_dynamic_mask_psnr",
            "delta_static_ghost_quality",
            "delta_points_total",
            "delta_routing_entropy",
            "delta_expected_static_points",
        ],
    )
    method_pair_summary.sort(key=lambda r: (r["comparison"], budget_sort_key(r["budget_label"])))
    best_rows = build_best_by_scene_budget(rows)
    config_diffs = build_config_differences(rows)
    availability = build_metric_availability(rows, history_rows)

    write_csv(out_dir / "runs.csv", rows)
    write_csv(out_dir / "budget_response.csv", budget_response)
    write_csv(out_dir / "history_features.csv", history_rows)
    write_csv(out_dir / "history_series_sampled.csv", sampled_rows)
    write_csv(out_dir / "within_method_budget_deltas.csv", within_deltas)
    write_csv(out_dir / "within_method_budget_summary.csv", within_summary)
    write_csv(out_dir / "same_budget_method_pairs.csv", method_pairs)
    write_csv(out_dir / "same_budget_method_summary.csv", method_pair_summary)
    write_csv(out_dir / "best_by_scene_budget.csv", best_rows)
    write_csv(out_dir / "config_differences.csv", config_diffs)
    write_csv(out_dir / "metric_availability.csv", availability)
    plots = make_plots(out_dir, rows, sampled_rows)

    write_report(
        out_dir,
        rows,
        history_rows,
        within_deltas,
        within_summary,
        method_pairs,
        method_pair_summary,
        best_rows,
        config_diffs,
        availability,
    )

    print(f"Wrote fixed-budget analysis to {out_dir}")
    print(f"runs={len(rows)} history={len(history_rows)} method_pairs={len(method_pairs)} within_budget_deltas={len(within_deltas)}")
    if plots:
        print("plots:", ", ".join(plots))


if __name__ == "__main__":
    main()
