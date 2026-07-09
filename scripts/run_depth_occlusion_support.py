#!/usr/bin/env python
"""Depth Anything 3 based non-oracle occlusion/reveal support tooling."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hide_reveal_poc import (  # noqa: E402
    IMAGE_SUFFIXES,
    dilate_binary_map,
    edge_map,
    first_indexed_frame,
    frame_index_from_name,
    gray_image,
    image_size,
    index_image_frames,
    index_named_files,
    indexed_gray,
    load_flow_support,
    normalize_positive_map,
    resize_float_map,
    support_tiles_from_mask,
    write_csv,
    write_json,
)


DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def split_words(value: str) -> List[str]:
    return [piece.strip() for piece in value.replace(",", " ").split() if piece.strip()]


def scene_image_dir(scene: str, scene_source: Dict[str, object], data_root: Optional[Path]) -> Path:
    if data_root is not None:
        return data_root / scene / "images"
    if scene_source.get("image_dir"):
        return Path(str(scene_source["image_dir"]))
    mask_dir = scene_source.get("mask_dir")
    if mask_dir:
        mask_path = Path(str(mask_dir))
        if len(mask_path.parents) >= 2:
            return mask_path.parents[1] / "images"
    raise ValueError(f"Cannot infer image directory for scene {scene}; pass --data-root")


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def robust_normalize(arr: np.ndarray, lo_q: float = 0.02, hi_q: float = 0.98) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros(arr.shape, dtype=np.float32)
    finite = arr[finite_mask]
    lo = float(np.quantile(finite, lo_q))
    hi = float(np.quantile(finite, hi_q))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    clean = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo)
    return np.clip((clean - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def depth_visual(depth: np.ndarray) -> np.ndarray:
    norm = robust_normalize(depth)
    return np.clip(np.rint(norm * 255.0), 0, 255).astype(np.uint8)


def prepare_frame_manifest(
    source_manifest: Path,
    out_path: Path,
    scenes: Sequence[str],
    cameras: Sequence[str],
    data_root: Optional[Path],
    frame_stride: int,
    max_frames_per_scene: Optional[int],
) -> Dict[str, object]:
    payload = load_json(source_manifest)
    if not isinstance(payload, dict) or not isinstance(payload.get("scene_sources"), dict):
        raise ValueError(f"Manifest lacks scene_sources: {source_manifest}")
    source_manifest = source_manifest.resolve()
    frame_stride = max(1, int(frame_stride))
    wanted_scenes = set(scenes)
    wanted_cameras = set(cameras)

    frames: List[Dict[str, object]] = []
    scene_reports: List[Dict[str, object]] = []
    for scene, raw_source in payload["scene_sources"].items():
        scene = str(scene)
        if wanted_scenes and scene not in wanted_scenes:
            continue
        if not isinstance(raw_source, dict):
            continue
        image_dir = scene_image_dir(scene, raw_source, data_root)
        frame_start, frame_end = [int(v) for v in raw_source.get("frame_range", [0, 299])]
        image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        selected = []
        camera_counts: Dict[str, int] = {}
        for path in image_paths:
            stem = path.stem
            camera = stem.split("_")[0] if "_" in stem else ""
            if wanted_cameras and camera not in wanted_cameras:
                continue
            frame_idx = frame_index_from_name(stem)
            if frame_idx is None or frame_idx < frame_start or frame_idx > frame_end:
                continue
            if (frame_idx - frame_start) % frame_stride != 0:
                continue
            selected.append((camera, frame_idx, path))
        selected = sorted(selected, key=lambda item: (item[0], item[1], item[2].name))
        if max_frames_per_scene is not None:
            selected = selected[: int(max_frames_per_scene)]
        for camera, frame_idx, path in selected:
            width, height = image_size(path)
            camera_counts[camera] = camera_counts.get(camera, 0) + 1
            frames.append(
                {
                    "scene": scene,
                    "camera": camera,
                    "frame_idx": int(frame_idx),
                    "image_name": path.stem,
                    "image_path": str(path.resolve()),
                    "image_size_xy": [int(width), int(height)],
                }
            )
        scene_reports.append(
            {
                "scene": scene,
                "image_dir": str(image_dir.resolve()),
                "frame_range": [frame_start, frame_end],
                "cameras": sorted(camera_counts),
                "camera_counts": camera_counts,
                "n_frames": len(selected),
            }
        )

    result = {
        "description": "Depth Anything 3 frame manifest for non-oracle depth occlusion support.",
        "generated_by": "prepare-depth-frame-manifest",
        "generated_at_utc": utc_now(),
        "source_manifest": str(source_manifest),
        "source_manifest_usage": "scene_sources_only_for_paths_and_frame_ranges",
        "uses_gt_residual": False,
        "uses_gt_crop_pixels": False,
        "uses_frozen_window_labels": False,
        "selection_parameters": {
            "scenes": sorted(wanted_scenes) if wanted_scenes else "all_scene_sources",
            "cameras": sorted(wanted_cameras) if wanted_cameras else "all_cameras",
            "frame_stride": int(frame_stride),
            "max_frames_per_scene": max_frames_per_scene,
        },
        "scene_reports": scene_reports,
        "frames": frames,
    }
    write_json(out_path, result)
    return result


def import_da3(da3_repo: Optional[Path]):
    if da3_repo is not None:
        repo = da3_repo.resolve()
        for candidate in (repo / "src", repo):
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3


def run_da3_inference(
    frame_manifest: Path,
    out_dir: Path,
    model_dir: str,
    da3_repo: Optional[Path],
    batch_size: int,
    process_res: int,
    process_res_method: str,
    device: str,
    overwrite: bool,
    write_vis: bool,
    max_images: Optional[int],
) -> Dict[str, object]:
    manifest = load_json(frame_manifest)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("frames"), list):
        raise ValueError(f"Frame manifest lacks frames: {frame_manifest}")
    frames = list(manifest["frames"])
    if max_images is not None:
        frames = frames[: int(max_images)]
    depth_root = out_dir / "depth_npz"
    vis_root = out_dir / "depth_vis"
    depth_root.mkdir(parents=True, exist_ok=True)
    if write_vis:
        vis_root.mkdir(parents=True, exist_ok=True)

    DepthAnything3 = import_da3(da3_repo)
    import torch

    model = DepthAnything3.from_pretrained(model_dir)
    model = model.to(device=torch.device(device))
    model.eval()

    output_records: List[Dict[str, object]] = []
    skipped = 0
    written = 0
    batch_size = max(1, int(batch_size))
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        pending = []
        for record in batch:
            scene = str(record["scene"])
            image_name = str(record["image_name"])
            out_path = depth_root / scene / f"{image_name}.npz"
            if out_path.exists() and not overwrite:
                skipped += 1
                output_records.append(
                    {
                        **{k: record[k] for k in ("scene", "camera", "frame_idx", "image_name", "image_path")},
                        "depth_npz_path": str(out_path.relative_to(out_dir)).replace("\\", "/"),
                        "status": "skipped_existing",
                    }
                )
                continue
            pending.append((record, out_path))
        if not pending:
            continue
        images = [str(item[0]["image_path"]) for item in pending]
        prediction = model.inference(
            images,
            process_res=int(process_res),
            process_res_method=str(process_res_method),
        )
        depths = np.asarray(prediction.depth, dtype=np.float32)
        conf = getattr(prediction, "conf", None)
        conf_arr = None if conf is None else np.asarray(conf, dtype=np.float32)
        for local_idx, (record, out_path) in enumerate(pending):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "depth": depths[local_idx].astype(np.float32),
                "scene": str(record["scene"]),
                "camera": str(record["camera"]),
                "frame_idx": int(record["frame_idx"]),
                "image_name": str(record["image_name"]),
                "image_path": str(record["image_path"]),
                "model_dir": str(model_dir),
            }
            if conf_arr is not None:
                payload["conf"] = conf_arr[local_idx].astype(np.float32)
            np.savez_compressed(out_path, **payload)
            if write_vis:
                vis_path = vis_root / str(record["scene"]) / f"{record['image_name']}.png"
                vis_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(depth_visual(depths[local_idx]), mode="L").save(vis_path)
            written += 1
            output_records.append(
                {
                    **{k: record[k] for k in ("scene", "camera", "frame_idx", "image_name", "image_path")},
                    "depth_npz_path": str(out_path.relative_to(out_dir)).replace("\\", "/"),
                    "status": "written",
                }
            )

    depth_manifest = {
        "description": "Depth Anything 3 depth sidecars for non-oracle depth occlusion support.",
        "generated_by": "infer-da3-depth",
        "generated_at_utc": utc_now(),
        "source_frame_manifest": str(frame_manifest.resolve()),
        "model_dir": str(model_dir),
        "da3_repo": str(da3_repo.resolve()) if da3_repo is not None else None,
        "process_res": int(process_res),
        "process_res_method": str(process_res_method),
        "device": str(device),
        "batch_size": int(batch_size),
        "uses_gt_residual": False,
        "uses_gt_crop_pixels": False,
        "uses_frozen_window_labels": False,
        "n_frames_requested": len(frames),
        "n_written": int(written),
        "n_skipped_existing": int(skipped),
        "frames": output_records,
    }
    write_json(out_dir / "da3_depth_manifest.json", depth_manifest)
    return depth_manifest


def load_depth_record(depth_manifest: Dict[str, object], out_dir: Path) -> Dict[Tuple[str, str], Dict[str, object]]:
    base = out_dir
    by_key: Dict[Tuple[str, str], Dict[str, object]] = {}
    for record in depth_manifest.get("frames", []):
        if not isinstance(record, dict):
            continue
        raw_path = record.get("depth_npz_path")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = base / path
        item = dict(record)
        item["depth_npz_abs_path"] = str(path)
        by_key[(str(record.get("scene", "")), str(record.get("image_name", "")))] = item
    return by_key


def read_depth_npz(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with np.load(path, allow_pickle=False) as npz:
        depth = np.asarray(npz["depth"], dtype=np.float32)
        conf = np.asarray(npz["conf"], dtype=np.float32) if "conf" in npz.files else None
    return depth, conf


def build_depth_occlusion_support(
    source_manifest: Path,
    depth_manifest_path: Path,
    out_dir: Path,
    route0_system: str,
    max_components_per_scene: int,
    max_pixel_fraction: float,
    boundary_dilate: int,
    min_component_area: int,
    min_score: float,
    tile_size: int,
    tile_stride: int,
    use_flow: bool,
    fill_component_tiles: bool,
) -> Dict[str, object]:
    source_payload = load_json(source_manifest)
    depth_payload = load_json(depth_manifest_path)
    if not isinstance(source_payload, dict) or not isinstance(source_payload.get("scene_sources"), dict):
        raise ValueError(f"Source manifest lacks scene_sources: {source_manifest}")
    if not isinstance(depth_payload, dict):
        raise ValueError(f"Depth manifest is not a JSON object: {depth_manifest_path}")
    depth_by_key = load_depth_record(depth_payload, depth_manifest_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    support_root = out_dir / "support_masks"
    support_root.mkdir(parents=True, exist_ok=True)

    selected_components: List[Dict[str, object]] = []
    support_frames: List[Dict[str, object]] = []
    scene_reports: List[Dict[str, object]] = []

    for scene, raw_scene_source in source_payload["scene_sources"].items():
        scene = str(scene)
        if not isinstance(raw_scene_source, dict):
            continue
        frame_start, frame_end = [int(v) for v in raw_scene_source.get("frame_range", [0, 299])]
        eval_dir = Path(str(raw_scene_source["route0_eval_dir"]))
        render_index = index_image_frames(eval_dir / "renders")
        static_index = index_image_frames(eval_dir / "static") if (eval_dir / "static").is_dir() else {}
        dynamic_index = index_image_frames(eval_dir / "dynamic") if (eval_dir / "dynamic").is_dir() else {}
        mask_dir = Path(str(raw_scene_source.get("mask_dir", ""))) if raw_scene_source.get("mask_dir") else None
        flow_dir = Path(str(raw_scene_source.get("flow_dir", ""))) if raw_scene_source.get("flow_dir") else None
        mask_by_name = index_named_files(mask_dir, IMAGE_SUFFIXES)
        flow_by_name = index_named_files(flow_dir, {".npz"}) if use_flow else {}
        first_path = first_indexed_frame(render_index)
        if first_path is not None:
            width, height = image_size(first_path)
            target_hw = (height, width)
        else:
            image_size_xy = raw_scene_source.get("image_size_xy", [676, 507])
            target_hw = (int(image_size_xy[1]), int(image_size_xy[0]))

        render_gray_by_frame: Dict[int, np.ndarray] = {}
        for frame_idx, path in render_index.items():
            render_gray_by_frame[int(frame_idx)] = gray_image(path, target_hw=target_hw)

        scene_depth_records = [
            item
            for key, item in depth_by_key.items()
            if key[0] == scene and frame_start <= int(item.get("frame_idx", -1)) <= frame_end
        ]
        scene_depth_records = sorted(scene_depth_records, key=lambda item: (str(item.get("camera", "")), int(item.get("frame_idx", -1))))

        prev_depth_by_camera: Dict[str, np.ndarray] = {}
        scene_components: List[Dict[str, object]] = []
        n_flow_used = 0
        n_depth_loaded = 0
        for record in scene_depth_records:
            image_name = str(record["image_name"])
            camera = str(record.get("camera", ""))
            frame_idx = int(record["frame_idx"])
            depth_path = Path(str(record["depth_npz_abs_path"]))
            if not depth_path.exists():
                continue
            depth, conf = read_depth_npz(depth_path)
            n_depth_loaded += 1
            depth_norm = robust_normalize(resize_float_map(depth, target_hw))
            depth_edge = normalize_positive_map(edge_map(depth_norm))
            prev_depth = prev_depth_by_camera.get(camera)
            if prev_depth is None:
                temporal_depth = np.zeros(target_hw, dtype=np.float32)
            else:
                temporal_depth = normalize_positive_map(np.abs(depth_norm - prev_depth))
            prev_depth_by_camera[camera] = depth_norm

            if conf is None:
                conf_edge = np.zeros(target_hw, dtype=np.float32)
                low_conf = np.zeros(target_hw, dtype=np.float32)
            else:
                conf_norm = robust_normalize(resize_float_map(conf, target_hw))
                conf_edge = normalize_positive_map(edge_map(conf_norm))
                low_conf = np.clip(1.0 - conf_norm, 0.0, 1.0).astype(np.float32)

            mask_gray = gray_image(mask_by_name[image_name], target_hw=target_hw) if image_name in mask_by_name else np.zeros(target_hw, dtype=np.float32)
            mask_boundary = dilate_binary_map(edge_map(mask_gray) > 0.05, max(1, boundary_dilate // 2))
            dynamic_gray = indexed_gray(dynamic_index, frame_idx, target_hw)
            dynamic_boundary = dilate_binary_map(edge_map(dynamic_gray) > 0.02, max(1, boundary_dilate // 2))
            static_gray = indexed_gray(static_index, frame_idx, target_hw)
            render_gray = render_gray_by_frame.get(frame_idx, np.zeros(target_hw, dtype=np.float32))
            static_delta = np.abs(render_gray - static_gray).astype(np.float32) if static_index else np.zeros(target_hw, dtype=np.float32)
            static_delta_boundary = dilate_binary_map(edge_map(static_delta) > 0.02, max(1, boundary_dilate // 2))
            prev_render = render_gray_by_frame.get(frame_idx - 1)
            if prev_render is None:
                flicker_boundary = np.zeros(target_hw, dtype=np.float32)
            else:
                flicker_boundary = dilate_binary_map(edge_map(np.abs(render_gray - prev_render)) > 0.02, max(1, boundary_dilate // 2))

            flow_path = flow_by_name.get(image_name)
            flow_mag, flow_valid, flow_available = load_flow_support(flow_path, target_hw)
            if flow_available:
                n_flow_used += 1
            flow_valid_boundary = dilate_binary_map(edge_map(flow_valid) > 0.05, max(1, boundary_dilate // 2))
            flow_mag_boundary = dilate_binary_map(edge_map(flow_mag) > 0.05, max(1, boundary_dilate // 2))

            occlusion_gate = np.clip(
                np.maximum(mask_gray, mask_boundary)
                + 0.35 * flow_valid_boundary
                + 0.30 * dynamic_boundary
                + 0.20 * static_delta_boundary,
                0.0,
                1.0,
            )
            depth_signal = np.clip(
                0.55 * depth_edge
                + 0.20 * temporal_depth
                + 0.10 * conf_edge
                + 0.05 * low_conf
                + 0.10 * flow_mag_boundary,
                0.0,
                1.0,
            )
            support_prior = np.clip(0.35 + 0.65 * occlusion_gate, 0.0, 1.0)
            score = np.clip(
                depth_signal * support_prior
                + 0.10 * mask_boundary
                + 0.05 * flicker_boundary
                + 0.05 * static_delta_boundary,
                0.0,
                1.0,
            ).astype(np.float32)
            binary = score >= float(min_score)
            max_pixels = max(1, int(float(max_pixel_fraction) * score.size))
            if int(binary.sum()) > max_pixels:
                positive = score[binary]
                threshold = float(np.partition(positive, max(0, positive.size - max_pixels))[max(0, positive.size - max_pixels)])
                binary = score >= threshold
                if int(binary.sum()) > max_pixels:
                    keep_idx = np.argpartition(score.reshape(-1), -max_pixels)[-max_pixels:]
                    limited = np.zeros(score.size, dtype=bool)
                    limited[keep_idx] = True
                    binary = limited.reshape(score.shape)

            components = support_tiles_from_mask(binary, score, min_component_area, tile_size=tile_size, tile_stride=tile_stride)
            components = sorted(components, key=lambda item: (float(item["component_score"]), int(item["area"])), reverse=True)[:4]
            for local_rank, component in enumerate(components):
                component.update(
                    {
                        "scene": scene,
                        "camera": camera,
                        "image_name": image_name,
                        "frame_idx": int(frame_idx),
                        "depth_source": str(depth_path),
                        "mask_source": str(mask_by_name.get(image_name)) if image_name in mask_by_name else None,
                        "flow_source": str(flow_path) if flow_path is not None else None,
                        "local_component_rank": int(local_rank),
                        "depth_edge_mean": float(depth_edge.mean()),
                        "temporal_depth_mean": float(temporal_depth.mean()),
                        "confidence_edge_mean": float(conf_edge.mean()),
                        "low_confidence_mean": float(low_conf.mean()),
                        "occlusion_gate_mean": float(occlusion_gate.mean()),
                        "mask_boundary_mean": float(mask_boundary.mean()),
                        "flow_valid_boundary_mean": float(flow_valid_boundary.mean()),
                        "flow_mag_boundary_mean": float(flow_mag_boundary.mean()),
                        "dynamic_boundary_mean": float(dynamic_boundary.mean()),
                        "static_delta_boundary_mean": float(static_delta_boundary.mean()),
                        "flicker_boundary_mean": float(flicker_boundary.mean()),
                    }
                )
            scene_components.extend(components)

        selected_scene_components = sorted(
            scene_components,
            key=lambda item: (float(item["component_score"]), int(item["area"])),
            reverse=True,
        )[: int(max_components_per_scene)]
        selected_by_image: Dict[str, List[Dict[str, object]]] = {}
        for rank, component in enumerate(selected_scene_components, start=1):
            component_id = f"{scene}_depth_{rank:03d}_{component['image_name']}"
            component["component_id"] = component_id
            selected_by_image.setdefault(str(component["image_name"]), []).append(component)
            selected_components.append({key: value for key, value in component.items() if not str(key).startswith("_")})

        scene_support_dir = support_root / scene
        scene_support_dir.mkdir(parents=True, exist_ok=True)
        max_support_fraction = 0.0
        for image_name, components in sorted(selected_by_image.items()):
            support_mask = np.zeros(target_hw, dtype=bool)
            score_mask = np.zeros(target_hw, dtype=np.float32)
            for component in components:
                if fill_component_tiles:
                    x0, y0, x1, y1 = [int(v) for v in component["bbox_xyxy"]]
                    tile_mask = np.zeros(target_hw, dtype=bool)
                    tile_mask[max(0, y0) : min(target_hw[0], y1), max(0, x0) : min(target_hw[1], x1)] = True
                    ys, xs = np.nonzero(tile_mask)
                else:
                    ys = component["_ys"]
                    xs = component["_xs"]
                support_mask[ys, xs] = True
                score_mask[ys, xs] = np.maximum(score_mask[ys, xs], float(component["component_score"]))
            max_pixels = max(1, int(float(max_pixel_fraction) * support_mask.size))
            if int(support_mask.sum()) > max_pixels:
                flat_score = score_mask.reshape(-1)
                keep_idx = np.argpartition(flat_score, -max_pixels)[-max_pixels:]
                limited = np.zeros(flat_score.shape, dtype=bool)
                limited[keep_idx] = True
                support_mask = limited.reshape(support_mask.shape)
            frame_idx = frame_index_from_name(image_name)
            support_fraction = float(support_mask.mean())
            max_support_fraction = max(max_support_fraction, support_fraction)
            mask_rel = Path("support_masks") / scene / f"{image_name}.png"
            Image.fromarray(support_mask.astype(np.uint8) * 255, mode="L").save(out_dir / mask_rel)
            support_frames.append(
                {
                    "scene": scene,
                    "image_name": image_name,
                    "frame_idx": int(frame_idx) if frame_idx is not None else -1,
                    "support_mask_path": str(mask_rel).replace("\\", "/"),
                    "support_pixel_count": int(support_mask.sum()),
                    "support_pixel_fraction": support_fraction,
                    "component_ids": [str(component["component_id"]) for component in components],
                }
            )

        scene_reports.append(
            {
                "scene": scene,
                "route0_eval_dir": str(eval_dir),
                "n_depth_frames_loaded": int(n_depth_loaded),
                "n_masks_indexed": len(mask_by_name),
                "n_flow_sidecars_indexed": len(flow_by_name),
                "n_flow_sidecars_used": int(n_flow_used),
                "n_raw_components": len(scene_components),
                "n_selected_components": len(selected_scene_components),
                "n_support_frames": len(selected_by_image),
                "max_support_pixel_fraction": float(max_support_fraction),
                "target_hw": [int(target_hw[0]), int(target_hw[1])],
            }
        )

    support_manifest = {
        "description": "Depth Anything 3 non-oracle depth occlusion/reveal support masks.",
        "frames_are_inclusive": True,
        "generated_by": "depth-occlusion-support",
        "generated_at_utc": utc_now(),
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_usage": "scene_sources_only_for_paths_and_frame_ranges",
        "depth_manifest": str(depth_manifest_path.resolve()),
        "depth_model": str(depth_payload.get("model_dir", DEFAULT_MODEL)),
        "route0_system": route0_system,
        "uses_gt_residual": False,
        "uses_gt_crop_pixels": False,
        "uses_frozen_window_labels": False,
        "selection_parameters": {
            "max_components_per_scene": int(max_components_per_scene),
            "max_pixel_fraction": float(max_pixel_fraction),
            "boundary_dilate": int(boundary_dilate),
            "min_component_area": int(min_component_area),
            "min_score": float(min_score),
            "tile_size": int(tile_size),
            "tile_stride": int(tile_stride),
            "use_flow": bool(use_flow),
            "fill_component_tiles": bool(fill_component_tiles),
            "score_weights": {
                "depth_edge": 0.55,
                "temporal_depth_change": 0.20,
                "confidence_edge": 0.10,
                "low_confidence": 0.05,
                "flow_magnitude_boundary": 0.10,
                "mask_boundary_bonus": 0.10,
                "route0_flicker_bonus": 0.05,
                "route0_static_delta_bonus": 0.05,
            },
        },
        "scene_reports": scene_reports,
        "components": selected_components,
        "support_frames": support_frames,
    }
    validation = {
        "ok": bool(support_frames)
        and all(int(report["n_selected_components"]) <= int(max_components_per_scene) for report in scene_reports)
        and all(float(report["max_support_pixel_fraction"]) <= float(max_pixel_fraction) + 1e-9 for report in scene_reports),
        "n_support_frames": len(support_frames),
        "n_components": len(selected_components),
        "errors": [],
        "warnings": [],
    }
    if not support_frames:
        validation["errors"].append("No support frames were generated.")
        validation["ok"] = False
    for report in scene_reports:
        if int(report["n_depth_frames_loaded"]) == 0:
            validation["errors"].append(f"Scene {report['scene']} loaded zero depth frames.")
            validation["ok"] = False
        if int(report["n_selected_components"]) == 0:
            validation["warnings"].append(f"Scene {report['scene']} has zero selected support components.")
        if use_flow and int(report["n_flow_sidecars_used"]) == 0:
            validation["warnings"].append(f"Scene {report['scene']} did not match flow sidecars by image name.")

    metadata = {
        "support_manifest": str((out_dir / "depth_occlusion_support_manifest.json").resolve()),
        "depth_manifest": str(depth_manifest_path.resolve()),
        "source_manifest": str(source_manifest.resolve()),
        "scene_reports": scene_reports,
        "n_components": len(selected_components),
        "n_support_frames": len(support_frames),
        "limitations": [
            "This support artifact is not a rendered Gaussian method result.",
            "Frozen R009 windows are not used to generate support; they may only be used in posthoc overlap diagnostics.",
            "R030 showed oracle support does not rescue the current posthoc micro-densification recipe, so support success should feed training-loop integration rather than support-only posthoc expansion.",
        ],
    }
    write_json(out_dir / "depth_occlusion_support_manifest.json", support_manifest)
    write_json(out_dir / "depth_occlusion_support_metadata.json", metadata)
    write_json(out_dir / "depth_occlusion_support_validation.json", validation)
    write_csv(out_dir / "depth_occlusion_support_components.csv", selected_components)
    write_depth_support_report(out_dir / "depth_occlusion_support_report.md", support_manifest, metadata, validation)
    return {"manifest": support_manifest, "metadata": metadata, "validation": validation}


def write_depth_support_report(
    path: Path,
    support_manifest: Dict[str, object],
    metadata: Dict[str, object],
    validation: Dict[str, object],
) -> None:
    lines = [
        "# Depth Occlusion Support",
        "",
        f"Generated: {support_manifest.get('generated_at_utc')}",
        "",
        "## Scientific Guardrails",
        "",
        f"- Uses GT residual: `{support_manifest.get('uses_gt_residual')}`",
        f"- Uses GT crop pixels: `{support_manifest.get('uses_gt_crop_pixels')}`",
        f"- Uses frozen event-crop labels: `{support_manifest.get('uses_frozen_window_labels')}`",
        f"- Source manifest usage: `{support_manifest.get('source_manifest_usage')}`",
        f"- Depth model: `{support_manifest.get('depth_model')}`",
        "",
        "## Scene Summary",
        "",
        "| Scene | Depth frames | Flow matched | Raw comps | Selected comps | Support frames | Max support frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in metadata.get("scene_reports", []):
        lines.append(
            "| {scene} | {depth} | {flow} | {raw} | {selected} | {frames} | {frac:.6f} |".format(
                scene=report.get("scene"),
                depth=report.get("n_depth_frames_loaded"),
                flow=report.get("n_flow_sidecars_used"),
                raw=report.get("n_raw_components"),
                selected=report.get("n_selected_components"),
                frames=report.get("n_support_frames"),
                frac=float(report.get("max_support_pixel_fraction", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- validation_ok: `{validation.get('ok')}`",
            f"- validation_errors: `{len(validation.get('errors', []))}`",
            f"- validation_warnings: `{len(validation.get('warnings', []))}`",
            "",
            "## Outputs",
            "",
            "- `depth_occlusion_support_manifest.json`",
            "- `depth_occlusion_support_metadata.json`",
            "- `depth_occlusion_support_components.csv`",
            "- `depth_occlusion_support_validation.json`",
            "- `support_masks/`",
        ]
    )
    errors = validation.get("errors", [])
    warnings = validation.get("warnings", [])
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend([f"- {error}" for error in errors])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DA3 depth occlusion support tooling.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-frame-manifest")
    prepare.add_argument("--source-manifest", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--scenes", default="cut_roasted_beef flame_steak sear_steak")
    prepare.add_argument("--cameras", default="cam00")
    prepare.add_argument("--data-root")
    prepare.add_argument("--frame-stride", type=int, default=1)
    prepare.add_argument("--max-frames-per-scene", type=int)

    infer = subparsers.add_parser("infer-da3-depth")
    infer.add_argument("--frame-manifest", required=True)
    infer.add_argument("--out-dir", required=True)
    infer.add_argument("--model-dir", default=DEFAULT_MODEL)
    infer.add_argument("--da3-repo")
    infer.add_argument("--batch-size", type=int, default=4)
    infer.add_argument("--process-res", type=int, default=504)
    infer.add_argument("--process-res-method", default="upper_bound_resize")
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--overwrite", action="store_true")
    infer.add_argument("--write-vis", action="store_true")
    infer.add_argument("--max-images", type=int)

    support = subparsers.add_parser("build-support")
    support.add_argument("--source-manifest", required=True)
    support.add_argument("--depth-manifest", required=True)
    support.add_argument("--out-dir", required=True)
    support.add_argument("--route0-system", default="route0")
    support.add_argument("--max-components-per-scene", type=int, default=36)
    support.add_argument("--max-pixel-fraction", type=float, default=0.03)
    support.add_argument("--boundary-dilate", type=int, default=6)
    support.add_argument("--min-component-area", type=int, default=16)
    support.add_argument("--min-score", type=float, default=0.08)
    support.add_argument("--tile-size", type=int, default=64)
    support.add_argument("--tile-stride", type=int, default=32)
    support.add_argument("--no-flow", action="store_true")
    support.add_argument(
        "--fill-component-tiles",
        action="store_true",
        help="Write selected component tile footprints instead of only the thresholded score pixels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare-frame-manifest":
        result = prepare_frame_manifest(
            source_manifest=Path(args.source_manifest),
            out_path=Path(args.out),
            scenes=split_words(args.scenes),
            cameras=split_words(args.cameras),
            data_root=Path(args.data_root) if args.data_root else None,
            frame_stride=args.frame_stride,
            max_frames_per_scene=args.max_frames_per_scene,
        )
        print(f"wrote_frame_manifest={Path(args.out).resolve()}")
        print(f"frames={len(result['frames'])}")
        print(f"uses_frozen_window_labels={result['uses_frozen_window_labels']}")
    elif args.command == "infer-da3-depth":
        result = run_da3_inference(
            frame_manifest=Path(args.frame_manifest),
            out_dir=Path(args.out_dir),
            model_dir=args.model_dir,
            da3_repo=Path(args.da3_repo) if args.da3_repo else None,
            batch_size=args.batch_size,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            device=args.device,
            overwrite=args.overwrite,
            write_vis=args.write_vis,
            max_images=args.max_images,
        )
        print(f"wrote_depth_manifest={Path(args.out_dir).resolve() / 'da3_depth_manifest.json'}")
        print(f"frames_requested={result['n_frames_requested']}")
        print(f"written={result['n_written']}")
        print(f"skipped_existing={result['n_skipped_existing']}")
    elif args.command == "build-support":
        result = build_depth_occlusion_support(
            source_manifest=Path(args.source_manifest),
            depth_manifest_path=Path(args.depth_manifest),
            out_dir=Path(args.out_dir),
            route0_system=args.route0_system,
            max_components_per_scene=args.max_components_per_scene,
            max_pixel_fraction=args.max_pixel_fraction,
            boundary_dilate=args.boundary_dilate,
            min_component_area=args.min_component_area,
            min_score=args.min_score,
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            use_flow=not args.no_flow,
            fill_component_tiles=args.fill_component_tiles,
        )
        print(f"wrote_support_manifest={Path(args.out_dir).resolve() / 'depth_occlusion_support_manifest.json'}")
        print(f"validation_ok={result['validation']['ok']}")
        print(f"validation_errors={len(result['validation']['errors'])}")
        print(f"support_frames={len(result['manifest']['support_frames'])}")
        if result["validation"]["errors"]:
            for error in result["validation"]["errors"][:8]:
                print(f"ERROR: {error}", file=sys.stderr)
            return 2
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
