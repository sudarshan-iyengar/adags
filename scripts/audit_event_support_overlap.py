#!/usr/bin/env python
"""Posthoc diagnostic overlap audit for event-support artifacts.

This script may read frozen R009 windows, so its outputs are diagnostic only.
Do not feed thresholds or selected windows from this audit back into a
non-oracle method configuration.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def area_xyxy(box):
    x0, y0, x1, y1 = [float(v) for v in box]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def intersect_xyxy(a, b):
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    return [max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)]


def iou_xyxy(a, b):
    inter = area_xyxy(intersect_xyxy(a, b))
    union = area_xyxy(a) + area_xyxy(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def temporal_overlap(a0, a1, b0, b1):
    start = max(int(a0), int(b0))
    end = min(int(a1), int(b1))
    if end < start:
        return 0, 0.0
    inter = end - start + 1
    union = max(int(a1), int(b1)) - min(int(a0), int(b0)) + 1
    return inter, inter / max(union, 1)


def frozen_windows(frozen_manifest):
    payload = load_json(frozen_manifest)
    scene_sources = payload.get("scene_sources", {})
    windows = []
    for window in payload.get("windows", []):
        scene = str(window["scene"])
        source = scene_sources.get(scene, {})
        image_size = source.get("image_size_xy") or source.get("mask_image_size_xy") or [0, 0]
        windows.append(
            {
                "window_id": window.get("window_id") or f"{scene}_{window['frame_start']}_{window['frame_end']}",
                "scene": scene,
                "frame_start": int(window["frame_start"]),
                "frame_end": int(window["frame_end"]),
                "crop_xyxy": [int(v) for v in window["crop_xyxy"]],
                "image_size_xy": [int(v) for v in image_size],
            }
        )
    return windows


def audit_box_windows(windows, support_payload):
    support_windows = support_payload.get("windows", [])
    rows = []
    for frozen in windows:
        crop = frozen["crop_xyxy"]
        n_frames = frozen["frame_end"] - frozen["frame_start"] + 1
        crop_area = max(area_xyxy(crop), 1.0)
        per_frame_coverage = []
        best_crop_iou = 0.0
        best_temporal_iou = 0.0
        matched = 0
        for frame_idx in range(frozen["frame_start"], frozen["frame_end"] + 1):
            frame_boxes = []
            for support in support_windows:
                if str(support.get("scene", "")) != frozen["scene"]:
                    continue
                overlap, temp_iou = temporal_overlap(
                    frozen["frame_start"],
                    frozen["frame_end"],
                    support.get("frame_start", -1),
                    support.get("frame_end", -1),
                )
                if overlap <= 0:
                    continue
                box = support.get("crop_xyxy")
                if not box or len(box) != 4:
                    continue
                best_crop_iou = max(best_crop_iou, iou_xyxy(crop, box))
                best_temporal_iou = max(best_temporal_iou, temp_iou)
                if int(support["frame_start"]) <= frame_idx <= int(support["frame_end"]):
                    frame_boxes.append(box)
            if frame_boxes:
                matched += 1
            # The box count is small; approximate union by rasterizing crop-local pixels.
            width = max(int(crop[2] - crop[0]), 1)
            height = max(int(crop[3] - crop[1]), 1)
            canvas = np.zeros((height, width), dtype=bool)
            for box in frame_boxes:
                inter = intersect_xyxy(crop, box)
                x0 = max(0, int(round(inter[0] - crop[0])))
                y0 = max(0, int(round(inter[1] - crop[1])))
                x1 = min(width, int(round(inter[2] - crop[0])))
                y1 = min(height, int(round(inter[3] - crop[1])))
                if x1 > x0 and y1 > y0:
                    canvas[y0:y1, x0:x1] = True
            per_frame_coverage.append(float(canvas.mean()))
        rows.append(
            {
                "window_id": frozen["window_id"],
                "scene": frozen["scene"],
                "frame_start": frozen["frame_start"],
                "frame_end": frozen["frame_end"],
                "crop_xyxy": json.dumps(crop),
                "support_type": "box_windows",
                "support_frame_fraction": matched / max(n_frames, 1),
                "mean_crop_coverage": float(np.mean(per_frame_coverage)) if per_frame_coverage else 0.0,
                "max_crop_coverage": float(np.max(per_frame_coverage)) if per_frame_coverage else 0.0,
                "best_crop_iou": best_crop_iou,
                "best_temporal_iou": best_temporal_iou,
                "notes": "posthoc diagnostic; frozen labels not used by support generator",
            }
        )
    return rows


def read_support_mask(mask_path, image_size_xy):
    with Image.open(mask_path) as image:
        gray = image.convert("L")
        if image_size_xy and list(gray.size) != list(image_size_xy):
            gray = gray.resize(tuple(image_size_xy), Image.Resampling.NEAREST)
        return np.asarray(gray, dtype=np.float32) > 0.0


def audit_support_masks(windows, support_manifest, support_payload):
    base_dir = Path(support_manifest).resolve().parent
    support_frames = support_payload.get("support_frames", [])
    by_scene_frame = {}
    for record in support_frames:
        scene = str(record.get("scene", ""))
        frame_idx = int(record.get("frame_idx", -1))
        raw_path = record.get("support_mask_path")
        if not scene or frame_idx < 0 or not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = base_dir / path
        by_scene_frame.setdefault((scene, frame_idx), []).append(path)

    rows = []
    for frozen in windows:
        crop = frozen["crop_xyxy"]
        x0, y0, x1, y1 = crop
        n_frames = frozen["frame_end"] - frozen["frame_start"] + 1
        coverages = []
        matched = 0
        missing_masks = 0
        for frame_idx in range(frozen["frame_start"], frozen["frame_end"] + 1):
            paths = by_scene_frame.get((frozen["scene"], frame_idx), [])
            if not paths:
                coverages.append(0.0)
                continue
            matched += 1
            frame_mask = None
            for path in paths:
                if not path.exists():
                    missing_masks += 1
                    continue
                mask = read_support_mask(path, frozen["image_size_xy"])
                frame_mask = mask if frame_mask is None else np.logical_or(frame_mask, mask)
            if frame_mask is None:
                coverages.append(0.0)
                continue
            crop_mask = frame_mask[y0:y1, x0:x1]
            coverages.append(float(crop_mask.mean()) if crop_mask.size else 0.0)
        rows.append(
            {
                "window_id": frozen["window_id"],
                "scene": frozen["scene"],
                "frame_start": frozen["frame_start"],
                "frame_end": frozen["frame_end"],
                "crop_xyxy": json.dumps(crop),
                "support_type": "support_masks",
                "support_frame_fraction": matched / max(n_frames, 1),
                "mean_crop_coverage": float(np.mean(coverages)) if coverages else 0.0,
                "max_crop_coverage": float(np.max(coverages)) if coverages else 0.0,
                "best_crop_iou": "",
                "best_temporal_iou": "",
                "missing_mask_files": missing_masks,
                "notes": "posthoc diagnostic; frozen labels not used by support generator",
            }
        )
    return rows


def write_report(path, rows, support_manifest, frozen_manifest):
    mean_coverage = float(np.mean([float(row["mean_crop_coverage"]) for row in rows])) if rows else 0.0
    mean_frame_fraction = float(np.mean([float(row["support_frame_fraction"]) for row in rows])) if rows else 0.0
    lines = [
        "# Event Support Overlap Audit",
        "",
        "Diagnostic only: this audit reads frozen R009 windows after support generation.",
        "It must not be used as test-time support or for threshold tuning.",
        "",
        f"- Frozen manifest: `{frozen_manifest}`",
        f"- Support manifest: `{support_manifest}`",
        f"- Windows: `{len(rows)}`",
        f"- Mean support-frame fraction: `{mean_frame_fraction:.4f}`",
        f"- Mean crop coverage: `{mean_coverage:.6f}`",
        "",
        "| Window | Scene | Support frame frac | Mean crop coverage | Max crop coverage | Best crop IoU | Best temporal IoU |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {window_id} | {scene} | {support_frame_fraction:.4f} | {mean_crop_coverage:.6f} | "
            "{max_crop_coverage:.6f} | {best_crop_iou} | {best_temporal_iou} |".format(
                **{
                    **row,
                    "support_frame_fraction": float(row["support_frame_fraction"]),
                    "mean_crop_coverage": float(row["mean_crop_coverage"]),
                    "max_crop_coverage": float(row["max_crop_coverage"]),
                    "best_crop_iou": row.get("best_crop_iou", ""),
                    "best_temporal_iou": row.get("best_temporal_iou", ""),
                }
            )
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit posthoc overlap between event support and frozen windows.")
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--support-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    frozen = frozen_windows(args.frozen_manifest)
    support_payload = load_json(args.support_manifest)
    if "support_frames" in support_payload:
        rows = audit_support_masks(frozen, args.support_manifest, support_payload)
    elif "windows" in support_payload:
        rows = audit_box_windows(frozen, support_payload)
    else:
        raise ValueError("Unsupported support manifest: expected support_frames or windows")

    out_dir = Path(args.out_dir)
    fieldnames = [
        "window_id",
        "scene",
        "frame_start",
        "frame_end",
        "crop_xyxy",
        "support_type",
        "support_frame_fraction",
        "mean_crop_coverage",
        "max_crop_coverage",
        "best_crop_iou",
        "best_temporal_iou",
        "missing_mask_files",
        "notes",
    ]
    for row in rows:
        row.setdefault("missing_mask_files", 0)
    summary = {
        "frozen_manifest": args.frozen_manifest,
        "support_manifest": args.support_manifest,
        "diagnostic_only": True,
        "n_windows": len(rows),
        "mean_support_frame_fraction": float(np.mean([float(r["support_frame_fraction"]) for r in rows])) if rows else 0.0,
        "mean_crop_coverage": float(np.mean([float(r["mean_crop_coverage"]) for r in rows])) if rows else 0.0,
        "windows": rows,
    }
    write_json(out_dir / "support_overlap_summary.json", summary)
    write_csv(out_dir / "support_overlap_windows.csv", rows, fieldnames)
    write_report(out_dir / "support_overlap_report.md", rows, args.support_manifest, args.frozen_manifest)
    print(f"Wrote support-overlap audit to {out_dir.resolve()}")
    print(f"windows={len(rows)}")
    print(f"mean_support_frame_fraction={summary['mean_support_frame_fraction']:.4f}")
    print(f"mean_crop_coverage={summary['mean_crop_coverage']:.6f}")


if __name__ == "__main__":
    main()
