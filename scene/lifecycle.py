"""CSVL-VPL v2 visibility-conditioned primitive lifecycle (Component 3).

The lifecycle manager owns four mechanisms that never touch rendered opacity:

1. protection: update freezing for primitives that every evidence-bearing view
   in the current batch reports as ``OCCLUDED`` (gradient rows are zeroed);
2. exposure: an occlusion-aware densification denominator so that primitives
   are densified per *exposed* observation rather than per rendered view;
3. persistent occlusion memory: an EMA over the per-iteration occluded
   fraction that drives the prune/split veto and the donor veto;
4. birth: budget-neutral, point-count-neutral reassignment of redundant donor
   slots into model-deficit surface positions, always routed through
   :func:`depth_visibility.capacity.apply_point_neutral_transaction`.

The module is torch-only and imports nothing from ``main.py``. The evidence
runtime is consumed duck-typed (``has_camera``/``verdicts``/``backproject``/
``counts_since_reset``/``reset_counts``/``cameras``) so this file never depends
on ``depth_visibility.evidence_runtime`` at import time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from depth_visibility.capacity import (
    apply_point_neutral_transaction,
    build_event_blind_capacity_targets,
    select_event_blind_donors,
)
from depth_visibility.errors import ContractError
from utils.sh_utils import RGB2SH

# Verdict codes. These MUST stay identical to
# depth_visibility.evidence_runtime (Component 2). They are duplicated here on
# purpose so that the lifecycle never imports the evidence module.
VERDICT_NOT_EVALUABLE = 0
VERDICT_NEAR = 1
VERDICT_OCCLUDED = 2
VERDICT_BEHIND_WEAK = 3
VERDICT_IN_FRONT = 4
VERDICT_UNCERTAIN = 5

VERDICT_NAMES = {
    VERDICT_NOT_EVALUABLE: "not_evaluable",
    VERDICT_NEAR: "near",
    VERDICT_OCCLUDED: "occluded",
    VERDICT_BEHIND_WEAK: "behind_weak",
    VERDICT_IN_FRONT: "in_front",
    VERDICT_UNCERTAIN: "uncertain",
}

#: Per-primitive parameters whose gradient rows are frozen by protection.
PROTECTED_PARAMETER_NAMES = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
    "_t",
    "_scaling_t",
    "_route_logit",
    "_motion_lora_coeff",
)

LIFECYCLE_STATE_SCHEMA = "phase9-lifecycle-state-v1"
LIFECYCLE_LEDGER_SCHEMA = "phase9-lifecycle-ledger-v1"

_BIRTH_OPACITY = 0.3
_VOXEL_KEY_OFFSET = 1 << 20
_VOXEL_KEY_LIMIT = (1 << 21) - 1


def _inverse_sigmoid(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ContractError("inverse sigmoid requires a probability in (0, 1)")
    return math.log(value / (1.0 - value))


def _json_default(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


@dataclass
class LifecycleConfig:
    """Frozen lifecycle knobs (mirrored by the trainer's OptimizationParams)."""

    enable: bool = False
    evidence_mode: str = "e1"  # "e1" | "presence" | "off"
    protection: bool = False
    exposure: bool = False
    birth: bool = False
    birth_mode: str = "e2"  # "e2" | "generic"
    occ_exposure_weight: float = 0.0
    weak_exposure_weight: float = 0.5
    protect_ema_beta: float = 0.98
    protect_ema_threshold: float = 0.6
    birth_interval: int = 500
    birth_k: int = 256
    birth_warmup: int = 1500
    birth_until: int = 5500
    birth_min_deficit: int = 64
    deficit_cell_rel: float = 0.01
    birth_view_checks: int = 3
    birth_view_min_near: int = 2
    backproject_stride: int = 6
    backproject_sigma_max: float = 0.12
    ledger_interval: int = 100

    def validate(self) -> "LifecycleConfig":
        if self.evidence_mode not in {"e1", "presence", "off"}:
            raise ContractError(f"unknown lifecycle evidence mode: {self.evidence_mode}")
        if self.birth_mode not in {"e2", "generic"}:
            raise ContractError(f"unknown lifecycle birth mode: {self.birth_mode}")
        if not 0.0 <= float(self.occ_exposure_weight) <= 1.0:
            raise ContractError("occ_exposure_weight must lie in [0, 1]")
        if not 0.0 <= float(self.weak_exposure_weight) <= 1.0:
            raise ContractError("weak_exposure_weight must lie in [0, 1]")
        if not 0.0 <= float(self.protect_ema_beta) < 1.0:
            raise ContractError("protect_ema_beta must lie in [0, 1)")
        if not 0.0 < float(self.protect_ema_threshold) <= 1.0:
            raise ContractError("protect_ema_threshold must lie in (0, 1]")
        if self.birth:
            if int(self.birth_interval) <= 0:
                raise ContractError("birth_interval must be positive when birth is enabled")
            if int(self.birth_k) <= 0:
                raise ContractError("birth_k must be positive when birth is enabled")
            if int(self.birth_warmup) <= 0:
                raise ContractError("birth_warmup must be positive when birth is enabled")
            if int(self.birth_until) < int(self.birth_warmup):
                raise ContractError("birth_until must not precede birth_warmup")
            if int(self.birth_min_deficit) <= 0:
                raise ContractError("birth_min_deficit must be positive")
            if float(self.deficit_cell_rel) <= 0.0:
                raise ContractError("deficit_cell_rel must be positive")
            if int(self.birth_view_checks) <= 0:
                raise ContractError("birth_view_checks must be positive")
            if int(self.birth_view_min_near) <= 0:
                raise ContractError("birth_view_min_near must be positive")
            if int(self.backproject_stride) <= 0:
                raise ContractError("backproject_stride must be positive")
            if float(self.backproject_sigma_max) <= 0.0:
                raise ContractError("backproject_sigma_max must be positive")
        return self

    @classmethod
    def from_namespace(cls, namespace: Any, prefix: str = "lifecycle_") -> "LifecycleConfig":
        """Build a config from a trainer namespace using ``prefix`` + field name."""

        values: dict[str, Any] = {}
        for spec in fields(cls):
            attribute = f"{prefix}{spec.name}"
            if hasattr(namespace, attribute):
                values[spec.name] = getattr(namespace, attribute)
        return cls(**values).validate()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LifecycleManager:
    """Visibility-conditioned protection, exposure, and budget-neutral birth."""

    def __init__(
        self,
        gaussians: Any,
        evidence: Any,
        cfg: LifecycleConfig,
        ledger_path: Optional[str] = None,
        scene_extent: float = 1.0,
        fps: float = 30.0,
    ) -> None:
        self.gaussians = gaussians
        self.evidence = evidence
        self.cfg = cfg.validate()
        self.ledger_path = str(ledger_path) if ledger_path else None
        self.scene_extent = float(scene_extent)
        if float(fps) <= 0.0:
            raise ContractError("lifecycle fps must be positive")
        self.fps = float(fps)

        count = self._model_row_count()
        device = self._device()
        self.occluded_ema = torch.zeros(count, dtype=torch.float32, device=device)
        self.exposure_accum = torch.zeros(count, dtype=torch.float32, device=device)
        self.born_at = torch.zeros(count, dtype=torch.long, device=device)

        self._views: list[tuple[str, int]] = []
        self._verdicts: list[Optional[torch.Tensor]] = []
        self.last_iteration = 0
        self.n_protected_last = 0

        self.births_total = 0
        self.birth_events = 0
        self.birth_skips: dict[str, int] = {}
        self.rows_added_total = 0
        self.rows_pruned_total = 0
        self.protection_calls = 0
        self.protected_rows_total = 0
        self.ledger_records_written = 0
        self.timers = {"observe": 0.0, "protection": 0.0, "exposure": 0.0, "birth": 0.0}

    # ------------------------------------------------------------------
    # model handles
    # ------------------------------------------------------------------
    def _xyz_parameter(self) -> torch.Tensor:
        xyz = getattr(self.gaussians, "_xyz", None)
        if not torch.is_tensor(xyz):
            xyz = getattr(self.gaussians, "get_xyz", None)
        if not torch.is_tensor(xyz):
            raise ContractError("lifecycle requires a Gaussian model exposing _xyz")
        return xyz

    def _model_row_count(self) -> int:
        return int(self._xyz_parameter().shape[0])

    def _device(self) -> torch.device:
        return self._xyz_parameter().device

    def _require_aligned(self, where: str) -> int:
        count = self._model_row_count()
        state = int(self.occluded_ema.shape[0])
        if state != count:
            raise ContractError(
                f"lifecycle state rows ({state}) do not match Gaussian rows ({count}) at {where}; "
                "on_rows_added/on_rows_pruned hooks are not wired"
            )
        return count

    def _invalidate_view_cache(self) -> None:
        self._views = []
        self._verdicts = []

    def _timestamp(self, frame: int) -> float:
        return float(frame) / self.fps

    def _dynamic_xyz(self, frame: int) -> torch.Tensor:
        getter = getattr(self.gaussians, "get_dynamic_xyz", None)
        if getter is None:
            return self._xyz_parameter().detach()
        return getter(self._timestamp(frame)).detach()

    def _has_camera(self, camera_id: str) -> bool:
        if self.evidence is None:
            return False
        checker = getattr(self.evidence, "has_camera", None)
        if checker is None:
            return True
        return bool(checker(camera_id))

    def _evidence_cameras(self) -> list[str]:
        if self.evidence is None:
            return []
        cameras = getattr(self.evidence, "cameras", None)
        if cameras is None:
            return []
        return [str(value) for value in cameras]

    # ------------------------------------------------------------------
    # evidence observation
    # ------------------------------------------------------------------
    def observe_batch(self, views: Sequence[tuple[str, int]], iteration: int) -> int:
        """Cache per-view verdicts for the current batch and update the EMA.

        Returns the number of views that carried evidence.
        """

        start = time.perf_counter()
        try:
            self.last_iteration = int(iteration)
            count = self._require_aligned("observe_batch")
            self._views = [(str(camera), int(frame)) for camera, frame in views]
            self._verdicts = []
            frame_cache: dict[int, torch.Tensor] = {}
            with torch.no_grad():
                for camera, frame in self._views:
                    if not self._has_camera(camera):
                        self._verdicts.append(None)
                        continue
                    if frame not in frame_cache:
                        frame_cache[frame] = self._dynamic_xyz(frame)
                    verdict = self.evidence.verdicts(frame_cache[frame], camera, frame)
                    verdict = torch.as_tensor(verdict).reshape(-1)
                    if int(verdict.shape[0]) != count:
                        raise ContractError(
                            f"evidence returned {int(verdict.shape[0])} verdicts for {count} primitives"
                        )
                    self._verdicts.append(verdict)
                present = [value for value in self._verdicts if value is not None]
                if present:
                    occluded = torch.stack(
                        [(value == VERDICT_OCCLUDED).to(torch.float32) for value in present]
                    ).mean(dim=0)
                    beta = float(self.cfg.protect_ema_beta)
                    self.occluded_ema.mul_(beta).add_(occluded.to(self.occluded_ema.device), alpha=1.0 - beta)
            return len(present)
        finally:
            self.timers["observe"] += time.perf_counter() - start

    def batch_occlusion_mask(self) -> torch.Tensor:
        """All-views AND rule: True where every evidence-bearing view is OCCLUDED."""

        count = int(self.occluded_ema.shape[0])
        mask = torch.zeros(count, dtype=torch.bool, device=self.occluded_ema.device)
        present = [value for value in self._verdicts if value is not None]
        if not present:
            return mask
        mask = present[0] == VERDICT_OCCLUDED
        for value in present[1:]:
            mask = mask & (value == VERDICT_OCCLUDED)
        return mask.to(device=self.occluded_ema.device)

    def apply_protection(self) -> int:
        """Zero the gradient rows of primitives occluded in every evidence view."""

        start = time.perf_counter()
        try:
            self.n_protected_last = 0
            if not (self.cfg.enable and self.cfg.protection):
                return 0
            if not self._verdicts:
                return 0
            count = self._require_aligned("apply_protection")
            mask = self.batch_occlusion_mask()
            protected = int(mask.sum().item())
            self.n_protected_last = protected
            self.protection_calls += 1
            self.protected_rows_total += protected
            if protected == 0:
                return 0
            rows = mask.nonzero(as_tuple=True)[0]
            with torch.no_grad():
                for name in PROTECTED_PARAMETER_NAMES:
                    parameter = getattr(self.gaussians, name, None)
                    if not torch.is_tensor(parameter):
                        continue
                    if parameter.numel() == 0 or int(parameter.shape[0]) != count:
                        continue
                    grad = parameter.grad
                    if grad is None or grad.numel() == 0 or int(grad.shape[0]) != count:
                        continue
                    grad.index_fill_(0, rows.to(grad.device), 0.0)
            return protected
        finally:
            self.timers["protection"] += time.perf_counter() - start

    # ------------------------------------------------------------------
    # exposure
    # ------------------------------------------------------------------
    def _exposure_weights(self, view_index: int) -> torch.Tensor:
        count = int(self.exposure_accum.shape[0])
        device = self.exposure_accum.device
        weights = torch.ones(count, dtype=torch.float32, device=device)
        mode = self.cfg.evidence_mode
        if mode == "off":
            return weights
        if mode == "presence":
            if view_index < 0 or view_index >= len(self._views):
                return weights
            frame = self._views[view_index][1]
            getter = getattr(self.gaussians, "get_marginal_t", None)
            if getter is None:
                return weights
            with torch.no_grad():
                marginal = getter(self._timestamp(frame))
            marginal = torch.as_tensor(marginal).detach().reshape(-1).to(device=device, dtype=torch.float32)
            if int(marginal.shape[0]) != count:
                raise ContractError("marginal_t row count does not match lifecycle state")
            marginal = torch.nan_to_num(marginal, nan=0.0, posinf=1.0, neginf=0.0)
            return marginal.clamp(0.05, 1.0)
        # "e1"
        if view_index < 0 or view_index >= len(self._verdicts):
            return weights
        verdict = self._verdicts[view_index]
        if verdict is None:
            return weights
        verdict = verdict.to(device)
        weights = torch.where(
            verdict == VERDICT_OCCLUDED,
            torch.full_like(weights, float(self.cfg.occ_exposure_weight)),
            weights,
        )
        weights = torch.where(
            verdict == VERDICT_BEHIND_WEAK,
            torch.full_like(weights, float(self.cfg.weak_exposure_weight)),
            weights,
        )
        return weights

    def accumulate_exposure(self, view_index: int, update_filter: torch.Tensor) -> None:
        """Accumulate exposure-weighted observation counts for one rendered view."""

        start = time.perf_counter()
        try:
            if not self.cfg.enable:
                return
            count = self._require_aligned("accumulate_exposure")
            if count == 0:
                return
            weights = self._exposure_weights(view_index)
            filter_tensor = torch.as_tensor(update_filter)
            with torch.no_grad():
                if filter_tensor.dtype == torch.bool:
                    filter_tensor = filter_tensor.reshape(-1).to(self.exposure_accum.device)
                    if int(filter_tensor.shape[0]) != count:
                        raise ContractError("exposure update filter has the wrong row count")
                    self.exposure_accum[filter_tensor] += weights[filter_tensor]
                else:
                    index = filter_tensor.reshape(-1).to(device=self.exposure_accum.device, dtype=torch.long)
                    if index.numel() == 0:
                        return
                    if int(index.max().item()) >= count or int(index.min().item()) < 0:
                        raise ContractError("exposure update index out of range")
                    self.exposure_accum.index_add_(0, index, weights.index_select(0, index))
        finally:
            self.timers["exposure"] += time.perf_counter() - start

    def exposure_denominator(self) -> Optional[torch.Tensor]:
        """Return the (N, 1) exposure denominator, or None when exposure is off."""

        if not (self.cfg.enable and self.cfg.exposure):
            return None
        return self.exposure_accum.clamp(min=1.0).reshape(-1, 1)

    # ------------------------------------------------------------------
    # persistent occlusion memory
    # ------------------------------------------------------------------
    def persistent_occluded_mask(self) -> torch.Tensor:
        """Raw persistent-occlusion mask (EMA > threshold), config-independent."""

        if int(self.occluded_ema.numel()) == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.occluded_ema.device)
        return self.occluded_ema > float(self.cfg.protect_ema_threshold)

    def protected_persistent(self) -> torch.Tensor:
        """Prune/split veto mask handed to the trainer.

        Gated by ``cfg.protection`` so that protection-off control lanes never
        acquire an implicit pruning veto through the shared trainer call site.
        """

        mask = self.persistent_occluded_mask()
        if not (self.cfg.enable and self.cfg.protection):
            return torch.zeros_like(mask)
        return mask

    # ------------------------------------------------------------------
    # birth
    # ------------------------------------------------------------------
    def _voxel_keys(self, xyz: torch.Tensor, cell: float) -> torch.Tensor:
        quantized = torch.floor(xyz.detach().to(torch.float64) / float(cell)).to(torch.int64)
        quantized = quantized + _VOXEL_KEY_OFFSET
        quantized = quantized.clamp(0, _VOXEL_KEY_LIMIT)
        return (quantized[:, 0] << 42) | (quantized[:, 1] << 21) | quantized[:, 2]

    def _first_occurrence(self, keys: torch.Tensor) -> torch.Tensor:
        """Deterministically keep the first candidate index per distinct key."""

        if int(keys.numel()) == 0:
            return torch.zeros(0, dtype=torch.long, device=keys.device)
        unique, inverse = torch.unique(keys, return_inverse=True)
        positions = torch.arange(int(keys.numel()), dtype=torch.long, device=keys.device)
        first = torch.zeros(int(unique.numel()), dtype=torch.long, device=keys.device)
        first.scatter_(0, inverse.flip(0), positions.flip(0))
        return torch.sort(first).values

    def _select_donors(
        self,
        bank: Any,
        k: int,
        iteration: int,
        excluded: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        if "denom" not in bank.accumulators:
            raise ContractError("capacity bank is missing the denom accumulator")
        row_count = int(bank.dynamic_count)
        k = int(min(int(k), row_count))
        if k <= 0:
            return {"selected_indices": [], "abstained": True, "reason": "empty_bank"}

        def _call(requested: int) -> dict[str, Any]:
            return select_event_blind_donors(
                xyz=bank.parameters["_xyz"].detach(),
                scaling_log=bank.parameters["_scaling"].detach(),
                opacity_logit=bank.parameters["_opacity"].detach(),
                denom=bank.accumulators["denom"].detach(),
                generation=bank.generation.detach(),
                stable_ids=bank.stable_ids.detach(),
                current_iteration=int(iteration),
                k=int(requested),
                excluded_mask=excluded,
            )

        selection = _call(k)
        if selection.get("abstained"):
            universe = list(selection.get("base_universe_indices") or [])
            if 0 < len(universe) < k:
                selection = _call(len(universe))
                selection["reduced_k"] = len(universe)
        return selection

    def _median_row(self, parameter: torch.Tensor, k: int) -> torch.Tensor:
        detached = parameter.detach()
        if int(detached.shape[0]) == 0:
            return torch.zeros((k, *detached.shape[1:]), dtype=detached.dtype, device=detached.device)
        median = detached.reshape(int(detached.shape[0]), -1).median(dim=0).values
        return median.reshape(1, *detached.shape[1:]).expand(k, *detached.shape[1:]).clone()

    def _build_birth_target_rows(
        self,
        bank: Any,
        points: torch.Tensor,
        colors: Optional[torch.Tensor],
        frame: int,
        cell: float,
    ) -> dict[str, torch.Tensor]:
        k = int(points.shape[0])
        rows: dict[str, torch.Tensor] = {}
        route_logit_init = getattr(self.gaussians, "route_logit_init", None)
        for name, parameter in bank.parameters.items():
            trailing = tuple(parameter.shape[1:])
            device = parameter.device
            dtype = parameter.dtype
            if name == "_xyz":
                if trailing != (3,):
                    raise ContractError("birth requires a 3D _xyz parameter")
                value = points.to(device=device, dtype=dtype)
            elif name == "_features_dc":
                value = torch.zeros((k, *trailing), dtype=dtype, device=device)
                flat = value.reshape(k, -1)
                if flat.shape[1] < 3:
                    raise ContractError("birth requires at least three DC feature channels")
                if colors is None:
                    sh = torch.full((k, 3), float(RGB2SH(torch.tensor(0.5)).item()))
                else:
                    sh = RGB2SH(colors.clamp(0.0, 1.0))
                flat[:, :3] = sh.to(device=device, dtype=dtype)
            elif name == "_scaling":
                value = torch.full(
                    (k, *trailing), math.log(max(0.5 * float(cell), 1e-8)), dtype=dtype, device=device
                )
            elif name == "_rotation":
                value = torch.zeros((k, *trailing), dtype=dtype, device=device)
                if value.reshape(k, -1).shape[1] < 1:
                    raise ContractError("birth requires a quaternion rotation parameter")
                value.reshape(k, -1)[:, 0] = 1.0
            elif name == "_opacity":
                value = torch.full(
                    (k, *trailing), _inverse_sigmoid(_BIRTH_OPACITY), dtype=dtype, device=device
                )
            elif name == "_t":
                value = torch.full((k, *trailing), self._timestamp(frame), dtype=dtype, device=device)
            elif name == "_scaling_t":
                value = self._median_row(parameter, k)
            elif name == "_route_logit":
                fill = 0.0 if route_logit_init is None else float(route_logit_init)
                value = torch.full((k, *trailing), fill, dtype=dtype, device=device)
            else:
                value = torch.zeros((k, *trailing), dtype=dtype, device=device)
            rows[name] = value
        return rows

    def _sample_colors(
        self, gt_image: Optional[torch.Tensor], pixels: Optional[torch.Tensor], keep: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if gt_image is None or pixels is None:
            return None
        image = gt_image.detach()
        if image.dim() != 3 or int(image.shape[0]) < 3:
            raise ContractError("birth colour sampling requires a (3, H, W) ground-truth image")
        height = int(image.shape[1])
        width = int(image.shape[2])
        selected = pixels.detach().reshape(-1, 2).index_select(0, keep.to(pixels.device))
        selected = selected.to(torch.float32)
        # Backprojected pixels are at the evidence-consensus resolution; the
        # ground-truth image is at training resolution. Rescale before sampling.
        evidence_width = float(getattr(self.evidence, "width", 0) or width)
        evidence_height = float(getattr(self.evidence, "height", 0) or height)
        px = (selected[:, 0] * (width / evidence_width)).round().to(torch.long).clamp(0, width - 1)
        py = (selected[:, 1] * (height / evidence_height)).round().to(torch.long).clamp(0, height - 1)
        return image[:3, py.to(image.device), px.to(image.device)].transpose(0, 1).to(torch.float32)

    def _record_skip(self, record: dict[str, Any]) -> dict[str, Any]:
        reason = str(record.get("reason") or "unknown")
        self.birth_skips[reason] = int(self.birth_skips.get(reason, 0)) + 1
        return record

    def maybe_birth(
        self,
        iteration: int,
        current_view: tuple[str, int],
        gt_image: Optional[torch.Tensor] = None,
        spatial_lr_scale: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Run one budget-neutral birth transaction when the cadence fires."""

        cfg = self.cfg
        iteration = int(iteration)
        self.last_iteration = iteration
        if not (cfg.enable and cfg.birth):
            return None
        if iteration <= 0 or iteration < int(cfg.birth_warmup) or iteration > int(cfg.birth_until):
            return None
        if int(cfg.birth_interval) <= 0 or iteration % int(cfg.birth_interval) != 0:
            return None
        start = time.perf_counter()
        try:
            with torch.no_grad():
                record = self._birth(iteration, current_view, gt_image, spatial_lr_scale)
        finally:
            self.timers["birth"] += time.perf_counter() - start
        self._append_ledger(record)
        return record

    def _birth(
        self,
        iteration: int,
        current_view: tuple[str, int],
        gt_image: Optional[torch.Tensor],
        spatial_lr_scale: Optional[float],
    ) -> dict[str, Any]:
        cfg = self.cfg
        camera_id = str(current_view[0])
        frame = int(current_view[1])
        record: dict[str, Any] = {
            "schema_version": LIFECYCLE_LEDGER_SCHEMA,
            "event": "birth",
            "iteration": iteration,
            "mode": cfg.birth_mode,
            "camera": camera_id,
            "frame": frame,
            "proposals": 0,
            "multiview_pass": 0,
            "deficits": 0,
            "deficit_voxels": 0,
            "requested_k": 0,
            "realized_k": 0,
            "donor_protected_excluded": 0,
            "reason": None,
        }
        count = self._require_aligned("maybe_birth")
        if count == 0:
            record["reason"] = "empty_bank"
            return self._record_skip(record)

        optimizer = getattr(self.gaussians, "optimizer", None)
        if optimizer is None:
            raise ContractError("lifecycle birth requires an initialized optimizer")
        bank = self.gaussians.build_capacity_bank()
        row_count = int(bank.dynamic_count)
        if row_count != count:
            raise ContractError("capacity bank rows do not match lifecycle state rows")

        excluded = self.persistent_occluded_mask().to(device=bank.parameters["_xyz"].device)
        record["donor_protected_excluded"] = int(excluded.sum().item())

        extent = self.scene_extent
        if extent <= 0.0 and spatial_lr_scale:
            extent = float(spatial_lr_scale)
        if extent <= 0.0:
            raise ContractError("lifecycle birth requires a positive scene extent")
        cell = float(cfg.deficit_cell_rel) * extent

        points: Optional[torch.Tensor] = None
        colors: Optional[torch.Tensor] = None
        if cfg.birth_mode == "e2":
            outcome = self._e2_proposals(iteration, camera_id, frame, cell, gt_image, record)
            if outcome is None:
                return self._record_skip(record)
            points, colors = outcome
            requested_k = int(min(int(cfg.birth_k), int(points.shape[0]), row_count))
        else:
            requested_k = int(min(int(cfg.birth_k), row_count))
        record["requested_k"] = requested_k
        if requested_k <= 0:
            record["reason"] = "no_capacity"
            return self._record_skip(record)

        selection = self._select_donors(bank, requested_k, iteration, excluded)
        record["donor_selection_reason"] = selection.get("reason")
        donor_indices = [int(value) for value in (selection.get("selected_indices") or [])]
        if not donor_indices:
            record["reason"] = "no_donors"
            return self._record_skip(record)
        donors = torch.as_tensor(donor_indices, dtype=torch.long, device=bank.parameters["_xyz"].device)
        if int(excluded.numel()) and bool(excluded.index_select(0, donors).any().item()):
            raise ContractError("donor selection returned a persistently occluded row")
        realized_k = int(donors.numel())

        if cfg.birth_mode == "e2":
            assert points is not None
            generator = torch.Generator(device="cpu")
            generator.manual_seed(iteration)
            order = torch.randperm(int(points.shape[0]), generator=generator)[:realized_k]
            order = order.to(points.device)
            chosen = points.index_select(0, order)
            chosen_colors = None if colors is None else colors.index_select(0, order.to(colors.device))
            target_rows = self._build_birth_target_rows(bank, chosen, chosen_colors, frame, cell)
            record["target_policy"] = "e2_model_deficit_backprojection_v1"
        else:
            target_rows, target_metadata = build_event_blind_capacity_targets(
                bank, donors, seed=0, iteration=iteration
            )
            record["target_policy"] = target_metadata["target_policy"]
            record["target_source_indices"] = target_metadata["target_source_indices"]

        transaction = apply_point_neutral_transaction(
            bank, optimizer, donors, target_rows, iteration=iteration, mode="reassign"
        )
        if transaction["budget_before"] != transaction["budget_after"]:
            raise ContractError("lifecycle birth changed the capacity budget")

        self.born_at[donors.to(self.born_at.device)] = int(iteration)
        self.occluded_ema[donors.to(self.occluded_ema.device)] = 0.0
        self.exposure_accum[donors.to(self.exposure_accum.device)] = 0.0
        self.births_total += realized_k
        self.birth_events += 1

        record["realized_k"] = realized_k
        record["donor_indices"] = transaction["donor_indices"]
        record["stable_slot_ids"] = transaction["stable_slot_ids"]
        record["moment_rows_reset"] = transaction["moment_rows_reset"]
        record["budget_before"] = transaction["budget_before"]
        record["budget_after"] = transaction["budget_after"]
        record["reason"] = None
        return record

    def _e2_proposals(
        self,
        iteration: int,
        camera_id: str,
        frame: int,
        cell: float,
        gt_image: Optional[torch.Tensor],
        record: dict[str, Any],
    ) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
        cfg = self.cfg
        if self.evidence is None or not self._has_camera(camera_id):
            record["reason"] = "camera_without_evidence"
            return None
        points, pixels = self.evidence.backproject(
            camera_id, frame, int(cfg.backproject_stride), float(cfg.backproject_sigma_max)
        )
        points = torch.as_tensor(points).detach().reshape(-1, 3)
        pixels = None if pixels is None else torch.as_tensor(pixels).detach().reshape(-1, 2)
        record["proposals"] = int(points.shape[0])
        if int(points.shape[0]) == 0:
            record["reason"] = "empty_backprojection"
            return None

        cameras = [value for value in self._evidence_cameras() if value != camera_id]
        if not cameras:
            record["reason"] = "no_check_cameras"
            return None
        offset = iteration % len(cameras)
        checks = [cameras[(offset + index) % len(cameras)] for index in range(min(int(cfg.birth_view_checks), len(cameras)))]
        near_counts = torch.zeros(int(points.shape[0]), dtype=torch.int32, device=points.device)
        for check_camera in checks:
            verdict = torch.as_tensor(self.evidence.verdicts(points, check_camera, frame)).reshape(-1)
            if int(verdict.shape[0]) != int(points.shape[0]):
                raise ContractError("multiview verdict row count does not match proposal count")
            near_counts += (verdict.to(points.device) == VERDICT_NEAR).to(torch.int32)
        required = int(min(int(cfg.birth_view_min_near), len(checks)))
        record["check_cameras"] = checks
        record["required_near"] = required
        keep = (near_counts >= required).nonzero(as_tuple=True)[0]
        record["multiview_pass"] = int(keep.numel())
        if int(keep.numel()) == 0:
            record["reason"] = "no_multiview_support"
            return None

        candidates = points.index_select(0, keep)
        occupied = self._voxel_keys(self._dynamic_xyz(frame), cell)
        candidate_keys = self._voxel_keys(candidates, cell)
        free = (~torch.isin(candidate_keys, occupied)).nonzero(as_tuple=True)[0]
        record["deficits"] = int(free.numel())
        if int(free.numel()) == 0:
            record["reason"] = "insufficient_deficit"
            return None
        unique_positions = self._first_occurrence(candidate_keys.index_select(0, free))
        free = free.index_select(0, unique_positions.to(free.device))
        record["deficit_voxels"] = int(free.numel())
        if int(free.numel()) < int(cfg.birth_min_deficit):
            record["reason"] = "insufficient_deficit"
            return None

        keep_positions = keep.index_select(0, free)
        colors = self._sample_colors(gt_image, pixels, keep_positions)
        return candidates.index_select(0, free), colors

    # ------------------------------------------------------------------
    # row bookkeeping
    # ------------------------------------------------------------------
    def on_rows_added(self, n_new: int, iteration: Optional[int] = None) -> None:
        """Extend lifecycle state for densified rows and reset exposure."""

        n_new = int(n_new)
        if n_new <= 0:
            return
        created = self.last_iteration if iteration is None else int(iteration)
        device = self.occluded_ema.device
        total = int(self.occluded_ema.shape[0]) + n_new
        self.occluded_ema = torch.cat(
            [self.occluded_ema, torch.zeros(n_new, dtype=torch.float32, device=device)], dim=0
        )
        self.born_at = torch.cat(
            [self.born_at, torch.full((n_new,), int(created), dtype=torch.long, device=device)], dim=0
        )
        # densify round boundary: exposure statistics restart for the whole bank
        self.exposure_accum = torch.zeros(total, dtype=torch.float32, device=device)
        self.rows_added_total += n_new
        self._invalidate_view_cache()
        self._require_aligned("on_rows_added")

    def on_rows_pruned(self, keep_mask: torch.Tensor) -> None:
        """Slice lifecycle state with the surviving-row mask from prune_points."""

        mask = torch.as_tensor(keep_mask).reshape(-1)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        mask = mask.to(self.occluded_ema.device)
        if int(mask.shape[0]) != int(self.occluded_ema.shape[0]):
            raise ContractError(
                f"prune mask rows ({int(mask.shape[0])}) do not match lifecycle state rows "
                f"({int(self.occluded_ema.shape[0])})"
            )
        self.rows_pruned_total += int((~mask).sum().item())
        self.occluded_ema = self.occluded_ema[mask]
        self.exposure_accum = self.exposure_accum[mask]
        self.born_at = self.born_at[mask]
        self._invalidate_view_cache()
        self._require_aligned("on_rows_pruned")

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def _exposure_stats(self) -> dict[str, Optional[float]]:
        if int(self.exposure_accum.numel()) == 0:
            return {"min": None, "median": None, "mean": None, "max": None}
        values = self.exposure_accum.detach().to(torch.float32)
        return {
            "min": float(values.min().item()),
            "median": float(values.median().item()),
            "mean": float(values.mean().item()),
            "max": float(values.max().item()),
        }

    def _verdict_counts(self) -> dict[str, int]:
        if self.evidence is None:
            return {}
        getter = getattr(self.evidence, "counts_since_reset", None)
        if getter is None:
            return {}
        counts = dict(getter() or {})
        reset = getattr(self.evidence, "reset_counts", None)
        if reset is not None:
            reset()
        return {str(key): int(value) for key, value in counts.items()}

    def _append_ledger(self, record: Mapping[str, Any]) -> None:
        if not self.ledger_path:
            return
        path = Path(self.ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(dict(record), sort_keys=True, allow_nan=False, default=_json_default) + "\n"
            )
        self.ledger_records_written += 1

    def log(self, iteration: int, extra: Optional[Mapping[str, Any]] = None) -> Optional[dict[str, Any]]:
        """Append one JSONL lifecycle record at the configured cadence."""

        iteration = int(iteration)
        self.last_iteration = iteration
        interval = int(self.cfg.ledger_interval)
        if interval <= 0 or iteration % interval != 0:
            return None
        persistent = self.persistent_occluded_mask()
        record: dict[str, Any] = {
            "schema_version": LIFECYCLE_LEDGER_SCHEMA,
            "event": "lifecycle",
            "iteration": iteration,
            "evidence_mode": self.cfg.evidence_mode,
            "birth_mode": self.cfg.birth_mode,
            "verdict_counts": self._verdict_counts(),
            "n_protected_last": int(self.n_protected_last),
            "n_persistent_occluded": int(persistent.sum().item()) if persistent.numel() else 0,
            "n_protected_persistent": int(self.protected_persistent().sum().item())
            if persistent.numel()
            else 0,
            "exposure": self._exposure_stats(),
            "births_total": int(self.births_total),
            "birth_events": int(self.birth_events),
            "birth_skips": dict(self.birth_skips),
            "retirements_total": int(self.rows_pruned_total),
            "rows_added_total": int(self.rows_added_total),
            "points": int(self.occluded_ema.shape[0]),
            "timers_s": {key: round(float(value), 6) for key, value in self.timers.items()},
        }
        if extra:
            record.update(dict(extra))
        self._append_ledger(record)
        return record

    def summary(self) -> dict[str, Any]:
        persistent = self.persistent_occluded_mask()
        return {
            "schema_version": LIFECYCLE_STATE_SCHEMA,
            "config": self.cfg.as_dict(),
            "last_iteration": int(self.last_iteration),
            "points": int(self.occluded_ema.shape[0]),
            "births_total": int(self.births_total),
            "birth_events": int(self.birth_events),
            "birth_skips": dict(self.birth_skips),
            "rows_added_total": int(self.rows_added_total),
            "rows_pruned_total": int(self.rows_pruned_total),
            "protection_calls": int(self.protection_calls),
            "protected_rows_total": int(self.protected_rows_total),
            "protected_rows_last": int(self.n_protected_last),
            "n_persistent_occluded": int(persistent.sum().item()) if persistent.numel() else 0,
            "n_protected_persistent": int(self.protected_persistent().sum().item())
            if persistent.numel()
            else 0,
            "exposure": self._exposure_stats(),
            "ledger_records_written": int(self.ledger_records_written),
            "timers_s": {key: round(float(value), 6) for key, value in self.timers.items()},
        }

    # ------------------------------------------------------------------
    # checkpoint state
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LIFECYCLE_STATE_SCHEMA,
            "occluded_ema": self.occluded_ema.detach().cpu(),
            "exposure_accum": self.exposure_accum.detach().cpu(),
            "born_at": self.born_at.detach().cpu(),
            "last_iteration": int(self.last_iteration),
            "counters": {
                "births_total": int(self.births_total),
                "birth_events": int(self.birth_events),
                "birth_skips": dict(self.birth_skips),
                "rows_added_total": int(self.rows_added_total),
                "rows_pruned_total": int(self.rows_pruned_total),
                "protection_calls": int(self.protection_calls),
                "protected_rows_total": int(self.protected_rows_total),
                "ledger_records_written": int(self.ledger_records_written),
            },
            "timers_s": dict(self.timers),
        }

    def load_state_dict(self, state: Optional[Mapping[str, Any]]) -> bool:
        if not state:
            return False
        if state.get("schema_version") != LIFECYCLE_STATE_SCHEMA:
            raise ContractError("unsupported lifecycle state schema")
        count = self._model_row_count()
        device = self._device()
        occluded = torch.as_tensor(state["occluded_ema"]).reshape(-1).to(device=device, dtype=torch.float32)
        exposure = torch.as_tensor(state["exposure_accum"]).reshape(-1).to(device=device, dtype=torch.float32)
        born_at = torch.as_tensor(state["born_at"]).reshape(-1).to(device=device, dtype=torch.long)
        if occluded.shape[0] != count or exposure.shape[0] != count or born_at.shape[0] != count:
            raise ContractError("lifecycle state row count does not match Gaussian rows")
        self.occluded_ema = occluded
        self.exposure_accum = exposure
        self.born_at = born_at
        self.last_iteration = int(state.get("last_iteration", 0))
        counters = dict(state.get("counters") or {})
        self.births_total = int(counters.get("births_total", 0))
        self.birth_events = int(counters.get("birth_events", 0))
        self.birth_skips = {str(key): int(value) for key, value in dict(counters.get("birth_skips") or {}).items()}
        self.rows_added_total = int(counters.get("rows_added_total", 0))
        self.rows_pruned_total = int(counters.get("rows_pruned_total", 0))
        self.protection_calls = int(counters.get("protection_calls", 0))
        self.protected_rows_total = int(counters.get("protected_rows_total", 0))
        self.ledger_records_written = int(counters.get("ledger_records_written", 0))
        timers = dict(state.get("timers_s") or {})
        for key in self.timers:
            self.timers[key] = float(timers.get(key, 0.0))
        self._invalidate_view_cache()
        return True


__all__ = [
    "LIFECYCLE_LEDGER_SCHEMA",
    "LIFECYCLE_STATE_SCHEMA",
    "LifecycleConfig",
    "LifecycleManager",
    "PROTECTED_PARAMETER_NAMES",
    "VERDICT_BEHIND_WEAK",
    "VERDICT_IN_FRONT",
    "VERDICT_NAMES",
    "VERDICT_NEAR",
    "VERDICT_NOT_EVALUABLE",
    "VERDICT_OCCLUDED",
    "VERDICT_UNCERTAIN",
]
