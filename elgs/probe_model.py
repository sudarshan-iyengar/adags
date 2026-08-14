"""ModelProbe: the live-model RenderProbe the observability path needs.

`elgs.probe.AnalyticProbe` is the FROZEN CPU reference (spec §3, the
"front-set compositing" decision recorded in
`configs/elgs/prereg_observability_v1.json`). It is deliberately naive and
takes an explicit splat list. This module supplies the counterpart the
prereg calls for -- "the GPU implementation (spatially pruned gather over
the model's projected Gaussians, torch, no_grad) lands with the trainer
wiring and must match AnalyticProbe on the closed-form fixture to 1e-6".

`tests/test_elgs_probe_model.py` holds that reference-parity test.

QUERY-SOURCE EXCLUSION -- DISCLOSED READING. AnalyticProbe excludes
splats by `source_track`, a tag that exists only in its hand-built
fixtures: no row of a GaussianModel carries a track id, and nothing in
the M0 substrate creates one. The spec's own §3 comment resolves what to
do -- "self-occlusion of the family is MODELED: other family splats still
occlude" -- so exclusion is NARROWER than the family and family-granular
exclusion would contradict it. What remains model-side is the STRICT
FRONT test that `transmittance` already applies: splats at or behind the
query depth never composite, and the tracked surface's own splats sit at
that depth by construction. `exclude_track` is therefore accepted and
recorded but performs no additional row filtering here, and that is the
CONSERVATIVE direction: any residual same-surface splat strictly in front
lowers T, hence lowers q, hence makes the evidence LESS informative, never
more. Row-level track provenance is a category-2 preregistration item.

Stop-gradient: every method runs under `torch.no_grad()`; q is a round
snapshot (spec §3) and never carries gradients into theta or a.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from depth_visibility.errors import ContractError

#: Splats whose Gaussian footprint at the query pixel falls below this
#: contribute a factor within 1e-9 of 1.0; the spatial prune drops them.
#: Chosen so the pruned product matches the unpruned one to well inside
#: the prereg's 1e-6 reference-parity tolerance.
FOOTPRINT_CUTOFF = 1e-7

#: Screen-space footprint of a splat, in units of its projected world
#: scale. The rasterizer composites an anisotropic 2-D covariance; the
#: probe uses the isotropic mean scale, matching ProbeGaussian's own
#: `sigma_px` contract. Disclosed approximation (prereg_observability
#: `approximation_limitation`: "sub-pixel footprint integration is not
#: modeled").
_MIN_SIGMA_PX = 1e-3


@dataclass(frozen=True)
class ProjectedSplats:
    """One camera/frame projection of the whole model, cached per round."""

    center_px: torch.Tensor  # (N, 2)
    depth: torch.Tensor  # (N,)
    alpha: torch.Tensor  # (N,)
    sigma_px: torch.Tensor  # (N,)
    family_ids: torch.Tensor  # (N,)


def _camera_intrinsics(camera) -> tuple[float, float, float, float]:
    """(fl_x, fl_y, cx, cy) for a Camera, from either convention."""
    fl_x = getattr(camera, "fl_x", None)
    fl_y = getattr(camera, "fl_y", None)
    cx = getattr(camera, "cx", None)
    cy = getattr(camera, "cy", None)
    width = int(camera.image_width)
    height = int(camera.image_height)
    if fl_x is None or fl_y is None or float(fl_x) <= 0.0 or float(fl_y) <= 0.0:
        # FoV convention: fl = 0.5 * size / tan(0.5 * fov).
        fovx = float(getattr(camera, "FoVx", 0.0))
        fovy = float(getattr(camera, "FoVy", 0.0))
        if fovx <= 0.0 or fovy <= 0.0:
            raise ContractError(
                "camera carries neither positive fl_x/fl_y nor a positive FoV"
            )
        fl_x = 0.5 * width / math.tan(0.5 * fovx)
        fl_y = 0.5 * height / math.tan(0.5 * fovy)
    if cx is None or cy is None:
        cx, cy = 0.5 * width, 0.5 * height
    return float(fl_x), float(fl_y), float(cx), float(cy)


def project_points(camera, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """World -> (pixel, view-space depth) for an (N, 3) tensor.

    Uses the camera's own `world_view_transform` (row-vector convention,
    already transposed by `scene.cameras.Camera`), so the probe and the
    rasterizer agree on extrinsics by construction.
    """
    if points.dim() != 2 or points.shape[1] != 3:
        raise ContractError("project_points expects an (N, 3) tensor")
    w2c = camera.world_view_transform.to(points.device, points.dtype)
    homogeneous = torch.cat(
        [points, torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)],
        dim=1,
    )
    view = homogeneous @ w2c
    depth = view[:, 2]
    fl_x, fl_y, cx, cy = _camera_intrinsics(camera)
    safe = depth.clamp_min(1e-8)
    u = fl_x * view[:, 0] / safe + cx
    v = fl_y * view[:, 1] / safe + cy
    return torch.stack([u, v], dim=1), depth


class ModelProbe:
    """RenderProbe over a live GaussianModel at a frozen round snapshot.

    Construct once per round-boundary q refresh; it caches one projection
    per (camera_id, frame) and never mutates the model.
    """

    def __init__(
        self,
        gaussians,
        cameras: dict,
        *,
        presence_at,
        frame_to_time: dict | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """`cameras` maps camera_id -> one Camera carrying that view's
        (static-rig, time-invariant) extrinsics and intrinsics;
        `frame_to_time` maps a frame INDEX to the model timestamp to
        evaluate presence at. The Camera's own `timestamp` is NOT used:
        a static rig contributes one Camera per id standing in for every
        frame, so its timestamp names one arbitrary frame and using it
        would evaluate presence at the wrong time for all the others.
        `presence_at(timestamp, present_family)` returns the (N, 1)
        per-row presence multiplier with `present_family` counterfactually
        forced present (or None for the live state)."""
        self._gaussians = gaussians
        self._cameras = dict(cameras)
        self._presence_at = presence_at
        self._frame_to_time = dict(frame_to_time or {})
        self._device = torch.device(device) if device is not None else None
        self._cache: dict[tuple[int, float, int | None], ProjectedSplats] = {}

    def _timestamp_for(self, frame: float, camera) -> float:
        index = int(frame)
        if self._frame_to_time:
            if index not in self._frame_to_time:
                raise ContractError(
                    f"probe has no model timestamp for frame index {index}"
                )
            return float(self._frame_to_time[index])
        return float(getattr(camera, "timestamp", frame))

    # -- projection -------------------------------------------------------

    def _camera(self, camera_id: int):
        if camera_id not in self._cameras:
            raise ContractError(f"probe has no camera for id {camera_id}")
        return self._cameras[camera_id]

    @torch.no_grad()
    def _project(self, camera_id: int, frame: float, present_family: int | None) -> ProjectedSplats:
        key = (int(camera_id), float(frame), present_family)
        if key in self._cache:
            return self._cache[key]
        camera = self._camera(camera_id)
        timestamp = self._timestamp_for(frame, camera)

        xyz = self._gaussians.get_xyz.detach()
        if self._device is not None:
            xyz = xyz.to(self._device)
        center_px, depth = project_points(camera, xyz)

        opacity = self._gaussians.get_opacity.detach().to(xyz.device).reshape(-1)
        presence = self._presence_at(timestamp, present_family)
        presence = presence.detach().to(xyz.device).reshape(-1)
        if presence.shape[0] != opacity.shape[0]:
            raise ContractError(
                f"presence column has {presence.shape[0]} rows but the model "
                f"has {opacity.shape[0]}"
            )
        alpha = (opacity * presence).clamp(0.0, 1.0)

        scaling = self._gaussians.get_scaling.detach().to(xyz.device)
        world_sigma = scaling[:, :3].mean(dim=1) if scaling.shape[1] >= 3 else scaling.reshape(-1)
        fl_x, fl_y, _, _ = _camera_intrinsics(camera)
        focal = 0.5 * (fl_x + fl_y)
        sigma_px = (world_sigma * focal / depth.clamp_min(1e-8)).clamp_min(_MIN_SIGMA_PX)

        family_ids = getattr(self._gaussians, "_elgs_family_ids", None)
        if family_ids is None:
            family_ids = torch.full((xyz.shape[0],), -1, dtype=torch.long)
        family_ids = family_ids.to(xyz.device)

        projected = ProjectedSplats(
            center_px=center_px,
            depth=depth,
            alpha=alpha,
            sigma_px=sigma_px,
            family_ids=family_ids,
        )
        self._cache[key] = projected
        return projected

    # -- RenderProbe ------------------------------------------------------

    @torch.no_grad()
    def transmittance(
        self,
        camera_id: int,
        frame: float,
        pixel: tuple[float, float],
        depth: float,
        *,
        exclude_track: int | None,
        present_family: int | None,
    ) -> float:
        """T just in FRONT of `depth` along the pixel ray (front-set
        compositing, spec §3). `exclude_track` is recorded but performs
        no row filtering -- see the module docstring's disclosed reading."""
        _ = exclude_track
        projected = self._project(camera_id, frame, present_family)
        if projected.depth.numel() == 0:
            return 1.0

        # Strict front set: splats at or behind the query point never
        # composite. This is also what stands in for query-source
        # exclusion model-side.
        front = (projected.depth > 0.0) & (projected.depth < float(depth))
        if not bool(front.any()):
            return 1.0

        px = torch.tensor(
            [float(pixel[0]), float(pixel[1])],
            device=projected.center_px.device,
            dtype=projected.center_px.dtype,
        )
        offset = projected.center_px[front] - px
        sigma = projected.sigma_px[front]
        exponent = -0.5 * (offset * offset).sum(dim=1) / (sigma * sigma)
        footprint = torch.exp(exponent)

        # Spatial prune: keep only splats that can move the product.
        keep = footprint > FOOTPRINT_CUTOFF
        if not bool(keep.any()):
            return 1.0
        contribution = (projected.alpha[front][keep] * footprint[keep]).clamp(0.0, 1.0)
        t_value = float(torch.exp(torch.log1p(-contribution.clamp_max(1.0 - 1e-12)).sum()))
        if not (0.0 <= t_value <= 1.0):
            raise ContractError(
                f"composited transmittance {t_value} outside [0,1]"
            )
        return t_value

    @torch.no_grad()
    def in_frustum(self, camera_id: int, frame: float, point: tuple[float, float, float]) -> bool:
        camera = self._camera(camera_id)
        device = self._gaussians.get_xyz.device
        tensor = torch.tensor([list(point)], device=device, dtype=torch.float32)
        pixel, depth = project_points(camera, tensor)
        if float(depth[0]) <= float(getattr(camera, "znear", 0.01)):
            return False
        u, v = float(pixel[0, 0]), float(pixel[0, 1])
        return 0.0 <= u <= camera.image_width - 1.0 and 0.0 <= v <= camera.image_height - 1.0

    @torch.no_grad()
    def project(self, camera_id: int, point: tuple[float, float, float]):
        """(pixel, depth) for one world point, or None outside the frustum.

        The `sigma_points_for` projector contract in `elgs.evidence_stack`.
        """
        camera = self._camera(camera_id)
        device = self._gaussians.get_xyz.device
        tensor = torch.tensor([list(point)], device=device, dtype=torch.float32)
        pixel, depth = project_points(camera, tensor)
        d = float(depth[0])
        if d <= float(getattr(camera, "znear", 0.01)):
            return None
        u, v = float(pixel[0, 0]), float(pixel[0, 1])
        if not (0.0 <= u <= camera.image_width - 1.0 and 0.0 <= v <= camera.image_height - 1.0):
            return None
        return (u, v), d


__all__ = ["FOOTPRINT_CUTOFF", "ModelProbe", "ProjectedSplats", "project_points"]
