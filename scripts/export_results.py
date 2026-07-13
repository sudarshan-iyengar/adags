import os
import re
import json
import argparse
from pathlib import Path
from statistics import mean
from collections import defaultdict

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


SCENES = [
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_salmon_1",
    "flame_steak",
    "sear_steak",
]

SCENE_SET = set(SCENES)

# Higher is better?
METRIC_INFO = {
    "psnr":   {"higher": True,  "digits": 2, "latex": r"PSNR $\uparrow$"},
    "ssim":   {"higher": True,  "digits": 4, "latex": r"SSIM $\uparrow$"},
    "lpips":  {"higher": False, "digits": 4, "latex": r"LPIPS $\downarrow$"},
    "num_GS": {"higher": False, "digits": 0, "latex": r"GS Count"},
    "static": {"higher": False, "digits": 4, "latex": r"Static"},
}

DEFAULT_METRICS = ["psnr", "ssim", "lpips", "num_GS", "static"]


def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def latex_escape(s):
    if s is None:
        return ""
    s = str(s)
    return (s.replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("#", r"\#")
            .replace("{", r"\{")
            .replace("}", r"\}"))


def strip_timestamp_prefix(run_name: str) -> str:
    m = re.match(r"^\d{8}_\d{6}_(.+)$", run_name)
    return m.group(1) if m else run_name


def infer_scene(run_name: str, scenes):
    remainder = strip_timestamp_prefix(run_name)
    matches = [scene for scene in scenes if remainder == scene or remainder.startswith(scene + "_")]
    if not matches:
        return None
    return max(matches, key=len)


def discover_methods(base_path: Path):
    return sorted([p.name for p in base_path.iterdir() if p.is_dir()], key=method_sort_key)


def method_sort_key(method: str):
    # Natural sort for scale folders like 1x, 2x, 4x, 8x
    m = re.fullmatch(r"(\d+)x", method)
    if m:
        return (0, int(m.group(1)), method)
    return (1, method)


def matches_substrings(text, substrings):
    if not substrings:
        return True
    return all(s in text for s in substrings)


def matches_regex(text, pattern):
    if pattern is None:
        return True
    return re.search(pattern, text) is not None


def find_validation_files(base_path: Path):
    return list(base_path.glob("*/*/test/*/stats/validation.json"))


def load_results(
        base_path,
        scenes,
        methods=None,
        method_contains=None,
        method_regex=None,
        exclude_methods=None,
        exclude_method_contains=None,
        selected_scenes=None,
        run_contains=None,
        run_regex=None,
):
    base = Path(base_path)
    rows = []

    for stats_path in find_validation_files(base):
        try:
            # runs/<method>/<run_name>/test/<iter>/stats/validation.json
            method = stats_path.parts[-6]
            run_name = stats_path.parts[-5]
            iteration = stats_path.parts[-3]

            if methods is not None and method not in methods:
                continue
            if exclude_methods is not None and method in exclude_methods:
                continue
            if not matches_substrings(method, method_contains):
                continue
            if exclude_method_contains and any(s in method for s in exclude_method_contains):
                continue
            if not matches_regex(method, method_regex):
                continue
            if not matches_substrings(run_name, run_contains):
                continue
            if not matches_regex(run_name, run_regex):
                continue

            scene = infer_scene(run_name, scenes)
            if scene is None:
                # Skip anything that doesn't match one of the six known scenes
                continue

            if selected_scenes is not None and scene not in selected_scenes:
                continue

            with open(stats_path, "r") as f:
                stats = json.load(f)

            row = {
                "method": method,
                "scene": scene,
                "run_name": run_name,
                "iter": iteration,
                "stats_path": str(stats_path),
            }

            for metric in DEFAULT_METRICS:
                row[metric] = safe_float(stats.get(metric))

            rows.append(row)

        except Exception:
            continue

    rows.sort(key=lambda r: (
        method_sort_key(r["method"]),
        SCENES.index(r["scene"]) if r["scene"] in SCENES else 999,
        r["run_name"],
    ))
    return rows


def compute_averages(rows, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    avg_rows = []
    for method, group in grouped.items():
        out = {"method": method, "count": len(group)}
        for metric in metrics:
            vals = [r[metric] for r in group if r[metric] is not None]
            out[metric] = mean(vals) if vals else None
        avg_rows.append(out)

    avg_rows.sort(key=lambda r: method_sort_key(r["method"]))
    return avg_rows


def best_values_per_scene(rows, metrics):
    result = defaultdict(dict)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scene"]].append(row)

    for scene, group in grouped.items():
        for metric in metrics:
            vals = [r[metric] for r in group if r[metric] is not None]
            if not vals:
                result[scene][metric] = None
            else:
                if METRIC_INFO[metric]["higher"]:
                    result[scene][metric] = max(vals)
                else:
                    result[scene][metric] = min(vals)
    return result


def best_values_overall(avg_rows, metrics):
    result = {}
    for metric in metrics:
        vals = [r[metric] for r in avg_rows if r[metric] is not None]
        if not vals:
            result[metric] = None
        else:
            if METRIC_INFO[metric]["higher"]:
                result[metric] = max(vals)
            else:
                result[metric] = min(vals)
    return result


def fmt_metric(metric, value):
    if value is None:
        return "--"
    digits = METRIC_INFO[metric]["digits"]
    return f"{value:.{digits}f}"


def maybe_bold(text, cond):
    return rf"\bfseries {text}" if cond else text


def build_detailed_latex(rows, metrics, caption, label):
    best = best_values_per_scene(rows, metrics)

    metric_cols = []
    for metric in metrics:
        digits = METRIC_INFO[metric]["digits"]
        if digits == 0:
            metric_cols.append(r"S[table-format=7.0, detect-weight]")
        else:
            metric_cols.append(rf"S[table-format=2.{digits}, detect-weight]")

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"    \centering")
    lines.append(rf"    \caption{{{latex_escape(caption)}}}")
    lines.append(rf"    \label{{{latex_escape(label)}}}")
    lines.append(r"    \small")
    lines.append(r"    \begin{tabular}{ll " + " ".join(metric_cols) + "}")
    lines.append(r"        \toprule")

    header = [r"\textbf{Scale/Method}", r"\textbf{Scene}"]
    header += [rf"\textbf{{{METRIC_INFO[m]['latex']}}}" for m in metrics]
    lines.append("        " + " & ".join(header) + r" \\")
    lines.append(r"        \midrule")

    current_method = None
    for row in rows:
        if current_method is not None and row["method"] != current_method:
            lines.append(r"        \midrule")
        current_method = row["method"]

        vals = []
        for metric in metrics:
            text = fmt_metric(metric, row[metric])
            vals.append(maybe_bold(text, row[metric] == best[row["scene"]][metric]))

        lines.append(
            "        "
            + f"{latex_escape(row['method'])} & {latex_escape(row['scene'])} & "
            + " & ".join(vals)
            + r" \\"
        )

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_average_latex(avg_rows, metrics, caption, label):
    best = best_values_overall(avg_rows, metrics)

    metric_cols = []
    for metric in metrics:
        digits = METRIC_INFO[metric]["digits"]
        if digits == 0:
            metric_cols.append(r"S[table-format=7.0, detect-weight]")
        else:
            metric_cols.append(rf"S[table-format=2.{digits}, detect-weight]")

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"    \centering")
    lines.append(rf"    \caption{{{latex_escape(caption)}}}")
    lines.append(rf"    \label{{{latex_escape(label)}}}")
    lines.append(r"    \begin{tabular}{l " + " ".join(metric_cols) + "}")
    lines.append(r"        \toprule")

    header = [r"\textbf{Training Setup}"]
    header += [rf"\textbf{{Mean {METRIC_INFO[m]['latex']}}}" for m in metrics]
    lines.append("        " + " & ".join(header) + r" \\")
    lines.append(r"        \midrule")

    for row in avg_rows:
        vals = []
        for metric in metrics:
            text = fmt_metric(metric, row[metric])
            vals.append(maybe_bold(text, row[metric] == best[metric]))

        lines.append(
            "        "
            + f"{latex_escape(row['method'])} & "
            + " & ".join(vals)
            + r" \\"
        )

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def print_console(rows, avg_rows, metrics):
    if not HAS_TABULATE:
        return

    detailed = []
    for r in rows:
        row = {"Method": r["method"], "Scene": r["scene"]}
        for m in metrics:
            row[m] = round(r[m], METRIC_INFO[m]["digits"]) if r[m] is not None else None
        detailed.append(row)

    summary = []
    for r in avg_rows:
        row = {"Method": r["method"]}
        for m in metrics:
            row[f"mean_{m}"] = round(r[m], METRIC_INFO[m]["digits"]) if r[m] is not None else None
        summary.append(row)

    print("\n=== Detailed Results ===\n")
    print(tabulate(detailed, headers="keys", tablefmt="grid"))
    print("\n=== Average Results ===\n")
    print(tabulate(summary, headers="keys", tablefmt="grid"))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-path",
        type=str,
        default=os.path.expandvars("$WORK/proj_adags/runs/"),
    )

    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--list-scenes", action="store_true")

    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--method-contains", nargs="+", default=None)
    parser.add_argument("--method-regex", type=str, default=None)
    parser.add_argument("--exclude-methods", nargs="+", default=None)
    parser.add_argument("--exclude-method-contains", nargs="+", default=None)

    parser.add_argument("--scenes", nargs="+", default=None)

    parser.add_argument("--run-contains", nargs="+", default=None)
    parser.add_argument("--run-regex", type=str, default=None)

    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, choices=DEFAULT_METRICS)

    parser.add_argument("--print-console", action="store_true")
    parser.add_argument("--print-latex", action="store_true")
    parser.add_argument("--save-detailed-latex", type=str, default=None)
    parser.add_argument("--save-avg-latex", type=str, default=None)

    parser.add_argument(
        "--detailed-caption",
        type=str,
        default="Detailed results across selected methods and scenes. Best values per scene are bolded.",
    )
    parser.add_argument(
        "--detailed-label",
        type=str,
        default="tab:detailed_results",
    )
    parser.add_argument(
        "--avg-caption",
        type=str,
        default="Aggregate mean metrics across selected methods.",
    )
    parser.add_argument(
        "--avg-label",
        type=str,
        default="tab:mean_results",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    base = Path(args.base_path)

    methods_on_disk = discover_methods(base)

    if args.list_methods:
        print("\n".join(methods_on_disk))
        return

    if args.list_scenes:
        print("\n".join(SCENES))
        return

    selected_scenes = args.scenes if args.scenes is not None else SCENES

    rows = load_results(
        base_path=args.base_path,
        scenes=SCENES,
        methods=args.methods,
        method_contains=args.method_contains,
        method_regex=args.method_regex,
        exclude_methods=args.exclude_methods,
        exclude_method_contains=args.exclude_method_contains,
        selected_scenes=selected_scenes,
        run_contains=args.run_contains,
        run_regex=args.run_regex,
    )

    if not rows:
        print("No matching validation.json files found.")
        return

    avg_rows = compute_averages(rows, metrics=args.metrics)

    if args.print_console:
        print_console(rows, avg_rows, metrics=args.metrics)

    detailed_latex = build_detailed_latex(
        rows=rows,
        metrics=args.metrics,
        caption=args.detailed_caption,
        label=args.detailed_label,
    )
    avg_latex = build_average_latex(
        avg_rows=avg_rows,
        metrics=args.metrics,
        caption=args.avg_caption,
        label=args.avg_label,
    )

    if args.print_latex:
        print("\n" + "=" * 100)
        print("DETAILED LATEX TABLE")
        print("=" * 100)
        print(detailed_latex)

        print("\n" + "=" * 100)
        print("AVERAGE LATEX TABLE")
        print("=" * 100)
        print(avg_latex)

    if args.save_detailed_latex:
        with open(args.save_detailed_latex, "w") as f:
            f.write(detailed_latex)
        print(f"Saved detailed LaTeX to {args.save_detailed_latex}")

    if args.save_avg_latex:
        with open(args.save_avg_latex, "w") as f:
            f.write(avg_latex)
        print(f"Saved average LaTeX to {args.save_avg_latex}")


if __name__ == "__main__":
    main()