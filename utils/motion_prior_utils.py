import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
TENSOR_EXTENSIONS = (".pt", ".pth", ".npy", ".npz")


def _as_float_tensor(value, device):
    if torch.is_tensor(value):
        return value.detach().float().to(device)
    return torch.from_numpy(np.asarray(value)).float().to(device)


def _select_npz_array(npz_obj, keys):
    for key in keys:
        if key in npz_obj:
            return npz_obj[key]
    if "arr_0" in npz_obj:
        return npz_obj["arr_0"]
    first_key = list(npz_obj.keys())[0]
    return npz_obj[first_key]


def _select_pt_array(obj, keys):
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                return obj[key]
        if "arr_0" in obj:
            return obj["arr_0"]
        first_key = list(obj.keys())[0]
        return obj[first_key]
    return obj


def resize_mask(mask, target_hw, dilate=0):
    if mask is None:
        return None
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() == 3 and mask.shape[-1] in (1, 3, 4):
        mask = mask.permute(2, 0, 1)
    if mask.dim() == 3 and mask.shape[0] != 1:
        mask = mask[:1]
    mask = mask.float().clamp(0.0, 1.0)
    if tuple(mask.shape[-2:]) != tuple(target_hw):
        mask = F.interpolate(mask[None], size=target_hw, mode="bilinear", align_corners=False)[0]
    if dilate > 0:
        kernel = int(2 * dilate + 1)
        mask = F.max_pool2d(mask[None], kernel_size=kernel, stride=1, padding=dilate)[0]
    return mask.clamp(0.0, 1.0)


def normalize_flow_tensor(flow):
    if flow is None:
        return None
    if flow.dim() == 4:
        flow = flow[0]
    if flow.dim() != 3:
        return None
    if flow.shape[0] == 2:
        return flow.float()
    if flow.shape[-1] == 2:
        return flow.permute(2, 0, 1).contiguous().float()
    return None


def resize_flow(flow, target_hw):
    flow = normalize_flow_tensor(flow)
    if flow is None:
        return None
    old_h, old_w = flow.shape[-2:]
    new_h, new_w = target_hw
    if (old_h, old_w) == (new_h, new_w):
        return flow
    flow = F.interpolate(flow[None], size=target_hw, mode="bilinear", align_corners=False)[0]
    flow[0] = flow[0] * (float(new_w) / max(float(old_w), 1.0))
    flow[1] = flow[1] * (float(new_h) / max(float(old_h), 1.0))
    return flow


def masked_l1(pred, target, mask, eps=1e-6):
    if mask is None:
        return torch.mean(torch.abs(pred - target))
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(0)
    if mask.shape[-2:] != pred.shape[-2:]:
        mask = resize_mask(mask.squeeze(0), pred.shape[-2:]).unsqueeze(0)
    denom = mask.sum().clamp_min(eps) * pred.shape[-3]
    return (torch.abs(pred - target) * mask).sum() / denom


def masked_psnr(pred, target, mask, eps=1e-6):
    if mask is None or mask.sum() <= eps:
        return None
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    denom = mask.sum().clamp_min(eps) * pred.shape[0]
    mse = ((pred - target).pow(2) * mask).sum() / denom
    return -10.0 * torch.log10(mse.clamp_min(eps))


def edge_magnitude(image):
    gray = image.mean(dim=0, keepdim=True)
    dx = gray[:, :, 1:] - gray[:, :, :-1]
    dy = gray[:, 1:, :] - gray[:, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-8)


def project_points_to_grid(points, camera):
    if points.numel() == 0:
        empty = points.new_empty((0, 2))
        return empty, points.new_empty((0, 1), dtype=torch.bool)
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    hom = torch.cat([points, ones], dim=-1)
    proj = camera.full_proj_transform.to(points.device, dtype=points.dtype)
    clip = hom @ proj
    w = clip[:, 3:4]
    w_safe = torch.where(w.abs() < 1e-6, torch.full_like(w, 1e-6), w)
    ndc = clip[:, :2] / w_safe
    valid = (
        (w.abs() >= 1e-6)
        & (ndc[:, 0:1] >= -1.0)
        & (ndc[:, 0:1] <= 1.0)
        & (ndc[:, 1:2] >= -1.0)
        & (ndc[:, 1:2] <= 1.0)
    )
    return ndc, valid


def project_points_to_screen(points, camera):
    grid, valid = project_points_to_grid(points, camera)
    if grid.numel() == 0:
        return grid, valid
    x = (grid[:, 0:1] + 1.0) * 0.5 * max(float(camera.image_width - 1), 1.0)
    y = (grid[:, 1:2] + 1.0) * 0.5 * max(float(camera.image_height - 1), 1.0)
    return torch.cat([x, y], dim=-1), valid


def sample_mask_at_points(mask, points, camera):
    if mask is None or points.numel() == 0:
        return points.new_zeros((points.shape[0], 1))
    mask = resize_mask(mask, (int(camera.image_height), int(camera.image_width)))
    grid, valid = project_points_to_grid(points, camera)
    if grid.numel() == 0:
        return points.new_zeros((0, 1))
    sample_grid = grid.view(1, 1, -1, 2)
    samples = F.grid_sample(mask[None], sample_grid, mode="bilinear", align_corners=True)[0, 0, 0]
    return samples.unsqueeze(-1) * valid.float()


class MotionPriorCache:
    def __init__(self, source_path, opt, device="cuda"):
        root = getattr(opt, "motion_prior_root", "")
        self.uses_default_root = root in ("", None)
        if self.uses_default_root:
            root = os.path.join(source_path, "motion_priors")
        self.source_path = Path(source_path)
        self.root = Path(root)
        self.device = device
        self.mask_cache = {}
        self.flow_cache = {}
        self.flow_mask_cache = {}
        self.dynamic_mask_from_residual = bool(getattr(opt, "dynamic_mask_from_residual", False))
        self.dynamic_mask_residual_quantile = float(getattr(opt, "dynamic_mask_residual_quantile", 0.85))
        self.dynamic_mask_dilate = int(getattr(opt, "dynamic_mask_dilate", 2))
        self.event_candidate_manifest = str(getattr(opt, "event_candidate_manifest", "") or "")
        self.event_candidate_scene = str(getattr(opt, "event_candidate_scene", "") or Path(source_path).name)
        self.event_candidate_dilate = int(getattr(opt, "event_candidate_dilate", 0))
        self.event_candidates_by_frame = {}
        self.event_candidate_mask_cache = {}
        if self.event_candidate_manifest:
            self.event_candidates_by_frame = self._load_event_candidates(self.event_candidate_manifest)
        self.event_boundary_support_manifest = str(getattr(opt, "event_boundary_support_manifest", "") or "")
        self.event_boundary_scene = str(getattr(opt, "event_boundary_scene", "") or Path(source_path).name)
        self.event_boundary_dilate = int(getattr(opt, "event_boundary_dilate", 0))
        self.event_boundary_replace_dynamic_mask = bool(getattr(opt, "event_boundary_replace_dynamic_mask", False))
        self.event_boundary_frame_fallback = bool(getattr(opt, "event_boundary_frame_fallback", False))
        self.event_boundary_by_image = {}
        self.event_boundary_by_frame = {}
        self.event_boundary_mask_cache = {}
        if self.event_boundary_support_manifest:
            self.event_boundary_by_image, self.event_boundary_by_frame = self._load_event_boundary_support(
                self.event_boundary_support_manifest
            )

    def _load_event_candidates(self, manifest_path):
        path = Path(manifest_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"event_candidate_manifest not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        by_frame = {}
        for window in payload.get("windows", []):
            if str(window.get("scene", "")) != self.event_candidate_scene:
                continue
            crop = [int(value) for value in window.get("crop_xyxy", [])]
            if len(crop) != 4:
                continue
            frame_start = int(window["frame_start"])
            frame_end = int(window["frame_end"])
            for frame_idx in range(frame_start, frame_end + 1):
                by_frame.setdefault(frame_idx, []).append(crop)
        return by_frame

    def _load_event_boundary_support(self, manifest_path):
        path = Path(manifest_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"event_boundary_support_manifest not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        base_dir = path.parent
        by_image = {}
        by_frame = {}
        for record in payload.get("support_frames", []):
            if str(record.get("scene", "")) != self.event_boundary_scene:
                continue
            raw_mask_path = record.get("support_mask_path")
            if not raw_mask_path:
                continue
            mask_path = Path(str(raw_mask_path))
            if not mask_path.is_absolute():
                mask_path = base_dir / mask_path
            item = {
                "path": mask_path,
                "scene": str(record.get("scene", "")),
                "image_name": str(record.get("image_name", "")),
                "frame_idx": int(record.get("frame_idx", -1)),
            }
            if item["image_name"]:
                by_image[item["image_name"]] = item
            if item["frame_idx"] >= 0 and item["frame_idx"] not in by_frame:
                by_frame[item["frame_idx"]] = item
        return by_image, by_frame

    def _frame_index_from_camera(self, camera):
        candidates = [getattr(camera, "image_name", ""), getattr(camera, "image_path", "")]
        for value in candidates:
            stem = Path(str(value)).stem
            matches = re.findall(r"\d+", stem)
            if matches:
                return int(matches[-1])
        return None

    def _event_candidate_mask(self, camera, target_hw):
        if not self.event_candidate_manifest:
            return None
        frame_idx = self._frame_index_from_camera(camera)
        if frame_idx is None:
            return None
        key = (frame_idx, tuple(target_hw))
        if key in self.event_candidate_mask_cache:
            return self.event_candidate_mask_cache[key]
        crops = self.event_candidates_by_frame.get(frame_idx, [])
        if not crops:
            self.event_candidate_mask_cache[key] = None
            return None
        height, width = int(target_hw[0]), int(target_hw[1])
        mask = torch.zeros((1, height, width), dtype=torch.float32, device=self.device)
        for crop in crops:
            x0, y0, x1, y1 = crop
            x0 = max(0, min(width, x0))
            x1 = max(0, min(width, x1))
            y0 = max(0, min(height, y0))
            y1 = max(0, min(height, y1))
            if x1 > x0 and y1 > y0:
                mask[:, y0:y1, x0:x1] = 1.0
        mask = resize_mask(mask, target_hw, self.event_candidate_dilate)
        if mask is not None and mask.sum() <= 1e-6:
            mask = None
        self.event_candidate_mask_cache[key] = mask
        return mask

    def _event_boundary_mask(self, camera, target_hw):
        if not self.event_boundary_support_manifest:
            return None
        image_name = str(getattr(camera, "image_name", ""))
        frame_idx = self._frame_index_from_camera(camera)
        record = self.event_boundary_by_image.get(image_name)
        if record is None and self.event_boundary_frame_fallback and frame_idx is not None:
            record = self.event_boundary_by_frame.get(frame_idx)
        if record is None:
            return None
        key = (image_name, frame_idx, tuple(target_hw), str(record["path"]))
        if key in self.event_boundary_mask_cache:
            return self.event_boundary_mask_cache[key]
        mask = self._load_mask_file(Path(record["path"]))
        mask = resize_mask(mask, target_hw, self.event_boundary_dilate)
        if mask is not None and mask.sum() <= 1e-6:
            mask = None
        self.event_boundary_mask_cache[key] = mask
        return mask

    def _candidate_roots(self):
        yield self.root
        if self.uses_default_root and self.source_path != self.root:
            yield self.source_path

    def _candidate_paths(self, image_name, subdirs, suffixes):
        names = [image_name, Path(image_name).stem]
        for root in self._candidate_roots():
            for subdir in subdirs:
                base_dir = root / subdir if subdir else root
                for name in names:
                    for suffix in suffixes:
                        yield base_dir / f"{name}{suffix}"

    def _find_existing(self, image_name, subdirs, suffixes):
        for path in self._candidate_paths(image_name, subdirs, suffixes):
            if path.exists():
                return path
        return None

    def _find_panoptic_seg_mask(self, camera):
        seg_root = self.source_path / "seg"
        if not seg_root.exists():
            return None

        image_path = getattr(camera, "image_path", None)
        if image_path:
            try:
                rel = Path(image_path).resolve().relative_to((self.source_path / "ims").resolve())
            except ValueError:
                rel = None
            if rel is not None and len(rel.parts) >= 2:
                candidate = seg_root / rel.parts[0] / f"{Path(rel.parts[-1]).stem}.png"
                if candidate.exists():
                    return candidate

        image_name = getattr(camera, "image_name", "")
        match = re.match(r"cam(\d+)_(\d+)$", image_name)
        if match:
            candidate = seg_root / str(int(match.group(1))) / f"{match.group(2)}.png"
            if candidate.exists():
                return candidate
        return None

    def _load_mask_file(self, path):
        if path is None:
            return None
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            with Image.open(path) as image:
                arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            return torch.from_numpy(arr)[None].to(self.device)
        if suffix == ".npy":
            return _as_float_tensor(np.load(path), self.device)
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as npz:
                return _as_float_tensor(_select_npz_array(npz, ("mask", "dynamic_mask", "foreground")), self.device)
        if suffix in (".pt", ".pth"):
            obj = torch.load(path, map_location=self.device)
            return _as_float_tensor(_select_pt_array(obj, ("mask", "dynamic_mask", "foreground")), self.device)
        return None

    def _load_flow_file(self, path):
        if path is None:
            return None
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return normalize_flow_tensor(_as_float_tensor(np.load(path), self.device))
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as npz:
                return normalize_flow_tensor(_as_float_tensor(_select_npz_array(npz, ("flow", "track_flow", "forward_flow")), self.device))
        if suffix in (".pt", ".pth"):
            obj = torch.load(path, map_location=self.device)
            return normalize_flow_tensor(_as_float_tensor(_select_pt_array(obj, ("flow", "track_flow", "forward_flow")), self.device))
        return None

    def _load_flow_mask_from_file(self, path):
        if path is None:
            return None
        suffix = path.suffix.lower()
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as npz:
                for key in ("mask", "flow_mask", "valid", "valid_mask"):
                    if key in npz:
                        return _as_float_tensor(npz[key], self.device)
        if suffix in (".pt", ".pth"):
            obj = torch.load(path, map_location=self.device)
            if isinstance(obj, dict):
                for key in ("mask", "flow_mask", "valid", "valid_mask"):
                    if key in obj:
                        return _as_float_tensor(obj[key], self.device)
        return None

    def get_dynamic_mask(self, camera, target_hw=None, gt_image=None, pred_image=None, allow_residual=True):
        target_hw = tuple(target_hw or (int(camera.image_height), int(camera.image_width)))
        key = (camera.image_name, target_hw)
        if key in self.mask_cache:
            return self.mask_cache[key]

        path = self._find_existing(
            camera.image_name,
            ("masks", "dynamic_masks", "foreground_masks", ""),
            IMAGE_EXTENSIONS + TENSOR_EXTENSIONS,
        )
        if path is None:
            path = self._find_panoptic_seg_mask(camera)
        mask = self._load_mask_file(path)
        loaded_from_file = mask is not None
        if mask is None and allow_residual and self.dynamic_mask_from_residual and gt_image is not None and pred_image is not None:
            with torch.no_grad():
                residual = (pred_image.detach() - gt_image.detach()).abs().mean(dim=0, keepdim=True)
                threshold = torch.quantile(residual.flatten(), self.dynamic_mask_residual_quantile)
                mask = (residual >= threshold).float()

        mask = resize_mask(mask, target_hw, self.dynamic_mask_dilate)
        candidate_mask = self._event_candidate_mask(camera, target_hw)
        if self.event_candidate_manifest:
            if candidate_mask is None:
                mask = None
            elif mask is None:
                mask = candidate_mask
            else:
                mask = (mask * candidate_mask).clamp(0.0, 1.0)
                if mask.sum() <= 1e-6:
                    mask = None
        boundary_mask = self._event_boundary_mask(camera, target_hw)
        if self.event_boundary_support_manifest:
            if boundary_mask is None:
                mask = None
            elif self.event_boundary_replace_dynamic_mask or mask is None:
                mask = boundary_mask
            else:
                mask = (mask * boundary_mask).clamp(0.0, 1.0)
                if mask.sum() <= 1e-6:
                    mask = None
        if loaded_from_file:
            self.mask_cache[key] = mask
        return mask

    def get_track_flow(self, camera, target_hw):
        target_hw = tuple(target_hw)
        key = (camera.image_name, target_hw)
        if key in self.flow_cache:
            return self.flow_cache[key], self.flow_mask_cache.get(key)

        flow_path = self._find_existing(
            camera.image_name,
            ("track_flows", "flows", "flow", ""),
            TENSOR_EXTENSIONS,
        )
        flow = resize_flow(self._load_flow_file(flow_path), target_hw)
        mask = resize_mask(self._load_flow_mask_from_file(flow_path), target_hw, 0)

        mask_path = self._find_existing(
            camera.image_name,
            ("track_flow_masks", "flow_masks", "track_masks", ""),
            IMAGE_EXTENSIONS + TENSOR_EXTENSIONS,
        )
        if mask_path is not None:
            mask = resize_mask(self._load_mask_file(mask_path), target_hw, self.dynamic_mask_dilate)
        elif mask is None and flow is not None:
            mask = torch.isfinite(flow).all(dim=0, keepdim=True).float()
        if flow is not None:
            flow = torch.nan_to_num(flow)

        self.flow_cache[key] = flow
        self.flow_mask_cache[key] = mask
        return flow, mask
