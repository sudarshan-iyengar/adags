"""ModelProbe: the live-model RenderProbe the observability path needs.

`elgs.probe.AnalyticProbe` is the FROZEN CPU reference (spec §3, the
"front-set compositing" decision recorded in
`configs/elgs/prereg_observability_v1.json`). It is deliberately naive and
takes an explicit splat list. This module supplies the counterpart the
prereg calls for -- "the GPU implementation (spatially pruned gather over
the model's projected Gaussians, torch, no_grad) lands with the trainer
wiring and must match AnalyticProbe on the closed-form fixture to 1e-6".

`tests/test_elgs_evidence_wiring.py::ProbeParityTests` holds that
reference-parity test.

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
recorded but performs no additional row filtering here.

DIRECTION OF BIAS IS NOT ESTABLISHED. An earlier revision called this
"the CONSERVATIVE direction" on the grounds that a residual same-surface
splat in front lowers T and hence q. That is only half the account:
`FOOTPRINT_CUTOFF` DROPS occluders, which RAISES T and q, and its bound
is per-splat so the aggregate grows with the kept set. The two effects
run opposite ways and neither has been measured. No conservatism may be
claimed for q before the q-tilde distribution is measured. Row-level
track provenance is a category-2 preregistration item.

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
class ProjectedGeometry:
    """One (camera, frame) projection of the model's DEFORMED positions.

    CORRECTED. An earlier revision cached per CAMERA ONLY, arguing that
    "`get_xyz` is the canonical position and the rig is static, so screen
    geometry does not vary with frame". That is FALSE for the only
    committed EL-GS lane: `scene/gaussian_model.py::get_dynamic_xyz`
    returns `_xyz + lora_motion_offset(t) + scaffold_motion_offset(t)`
    for `gaussian_dim == 4` / `motion_model == "lora"` / `not rot_4d`,
    and `gaussian_renderer/__init__.py` rasterizes exactly that. Screen
    geometry IS frame-dependent, so the key is (camera, frame).

    The cache is bounded by `max_entries` instead: the revision this
    replaces was keyed (camera, frame, present_family), whose |families|
    factor made it hundreds of GB at M0's S1 scale. Dropping the family
    factor is what makes the cache affordable — presence is applied
    separately and never enters the geometry.
    """

    center_px: torch.Tensor  # (N, 2)
    depth: torch.Tensor  # (N,)
    sigma_px: torch.Tensor  # (N,)


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
        pixel_scale: float = 1.0,
        max_entries: int = 64,
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
        if pixel_scale <= 0.0:
            raise ContractError("pixel_scale must be positive")
        # REPORT-RASTER MAPPING. Reports live in the CONVERTED scene's
        # full raster (the tracks builder bounds them by the
        # transforms' declared w/h), while the trainer may load that
        # scene downscaled -- `utils/camera_utils.loadCam` divides every
        # intrinsic by `scale` for `args.resolution != 1`, and the one
        # committed EL-GS config sets `resolution: 4`. Projected pixels
        # are therefore multiplied back into the report raster before
        # they are ever compared with a report position. Without this
        # the two live in rasters differing by 4x, `g_pos` underflows to
        # zero for every report, and the evidence term acquires a
        # CONSTANT SIGN favouring truncation -- an axis-scale artifact
        # that would read as "the tracker says carve these families".
        self._pixel_scale = float(pixel_scale)
        self._device = torch.device(device) if device is not None else None
        self._max_entries = int(max_entries)
        self._geometry: dict[tuple[int, float], ProjectedGeometry] = {}
        self._static_geometry: dict[int, ProjectedGeometry] = {}
        self._presence: dict[float, torch.Tensor] = {}

    def _timestamp_for(self, frame: float, camera) -> float:
        index = int(frame)
        if not self._frame_to_time:
            raise ContractError(
                "probe has no frame->time map; falling back to the Camera's "
                "own timestamp would collapse every frame onto one arbitrary "
                "time and evaluate the deformed geometry at the wrong instant"
            )
        if index not in self._frame_to_time:
            raise ContractError(
                f"probe has no model timestamp for frame index {index}"
            )
        return float(self._frame_to_time[index])

    # -- projection -------------------------------------------------------

    def _camera(self, camera_id: int):
        if camera_id not in self._cameras:
            raise ContractError(f"probe has no camera for id {camera_id}")
        return self._cameras[camera_id]

    @torch.no_grad()
    def _geometry_for(self, camera_id: int, frame: float) -> ProjectedGeometry:
        """Screen geometry for one (camera, frame); family-independent.

        Positions come from `get_dynamic_xyz(timestamp)` — the same call
        `gaussian_renderer` rasterizes — not from the canonical `_xyz`.
        """
        camera = self._camera(camera_id)
        timestamp = self._timestamp_for(frame, camera)
        key = (int(camera_id), float(timestamp))
        if key in self._geometry:
            return self._geometry[key]
        dynamic = getattr(self._gaussians, "get_dynamic_xyz", None)
        xyz = (
            dynamic(timestamp) if callable(dynamic) else self._gaussians.get_xyz
        ).detach()
        if self._device is not None:
            xyz = xyz.to(self._device)
        center_px, depth = project_points(camera, xyz)
        center_px = center_px * self._pixel_scale

        scaling = self._gaussians.get_scaling.detach().to(xyz.device)
        world_sigma = (
            scaling[:, :3].mean(dim=1) if scaling.shape[1] >= 3 else scaling.reshape(-1)
        )
        fl_x, fl_y, _, _ = _camera_intrinsics(camera)
        focal = 0.5 * (fl_x + fl_y) * self._pixel_scale
        sigma_px = (world_sigma * focal / depth.clamp_min(1e-8)).clamp_min(_MIN_SIGMA_PX)

        geometry = ProjectedGeometry(
            center_px=center_px, depth=depth, sigma_px=sigma_px
        )
        if len(self._geometry) >= self._max_entries:
            # Bounded, not unbounded: evict in insertion order. A round
            # walks frames in order, so the evicted entry is the least
            # recently needed. The bound is what keeps the (camera,
            # frame) key affordable.
            self._geometry.pop(next(iter(self._geometry)))
        self._geometry[key] = geometry
        return geometry

    @torch.no_grad()
    def _alpha_for(self, frame: float, camera, present_family: int | None) -> torch.Tensor:
        """Composited per-row opacity at `frame`, family forced present.

        Presence is cached per FRAME for the live state; forcing is a
        masked write on a clone, so the cache never grows with the
        family count.
        """
        timestamp = self._timestamp_for(frame, camera)
        if timestamp not in self._presence:
            column = self._presence_at(timestamp, None).detach().reshape(-1)
            self._presence[timestamp] = column
        presence = self._presence[timestamp]

        opacity = self._gaussians.get_opacity.detach().reshape(-1).to(presence.device)
        # THE RENDERER'S COMPOSITION, read off gaussian_renderer, not
        # assumed. `opacity = base_opacity * dynamic_probability`, then
        # for gaussian_dim == 4:
        #     marginal_t = get_elgs_presence(t)   if EL-GS is active
        #     marginal_t = get_marginal_t(t)      otherwise
        #     opacity = opacity * marginal_t
        # So under EL-GS the presence multiplier REPLACES the temporal
        # marginal — it is never multiplied by it. A previous revision
        # multiplied by `get_marginal_t` as well, calling that a
        # faithfulness repair; because that factor decays to ~0 for
        # temporally distant Gaussians it deleted most occluders, drove
        # T toward 1 and q TOO HIGH — treating the evidence as fully
        # informative exactly where the model says the point is occluded.
        # `dynamic_probability` was simultaneously omitted, which the
        # renderer does apply.
        routing = getattr(self._gaussians, "get_dynamic_probability", None)
        if routing is not None and getattr(routing, "numel", lambda: 0)():
            routing = routing.detach().reshape(-1).to(presence.device)
            if routing.shape == opacity.shape:
                opacity = opacity * routing

        if presence.shape[0] != opacity.shape[0]:
            raise ContractError(
                f"presence column has {presence.shape[0]} rows but the model "
                f"has {opacity.shape[0]}"
            )
        if present_family is not None:
            family_ids = getattr(self._gaussians, "_elgs_family_ids", None)
            if family_ids is not None:
                mask = (family_ids.to(presence.device) == int(present_family))
                presence = presence.clone().masked_fill(mask, 1.0)
        return (opacity * presence).clamp(0.0, 1.0)

    @torch.no_grad()
    def _static_alpha(self) -> torch.Tensor | None:
        """The SECOND opacity column the rasterizer composites.

        Under soft routing (`gaussian_dim == 4` and `enable_soft_routing`
        and a populated routing column -- the committed EL-GS lane)
        `gaussian_renderer` hands the rasterizer a full second copy of
        every row at `means3D_static = pc.get_xyz` (CANONICAL, not
        deformed) with `opacity_static = base_opacity *
        static_probability` and NO presence and NO temporal marginal.
        Those static twins occlude in the rendered image. A probe that
        ignores them reports transmittance too HIGH, hence q too high --
        the same direction as the get_marginal_t defect. Returns None
        when the branch is inactive.
        """
        static_p = getattr(self._gaussians, "get_static_probability", None)
        if static_p is None or not getattr(static_p, "numel", lambda: 0)():
            return None
        routing = getattr(self._gaussians, "get_dynamic_probability", None)
        if routing is None or not getattr(routing, "numel", lambda: 0)():
            return None
        opacity = self._gaussians.get_opacity.detach().reshape(-1)
        static_column = static_p.detach().reshape(-1).to(opacity.device)
        if static_column.shape != opacity.shape:
            return None
        return (opacity * static_column).clamp(0.0, 1.0)

    @torch.no_grad()
    def _static_geometry_for(self, camera_id: int) -> ProjectedGeometry:
        """Canonical-position geometry for the static branch.

        Frame-independent BY CONSTRUCTION here (the renderer uses
        `pc.get_xyz`, not `get_dynamic_xyz`), so a camera-only key is
        correct for this branch specifically.
        """
        key = int(camera_id)
        if key in self._static_geometry:
            return self._static_geometry[key]
        camera = self._camera(camera_id)
        xyz = self._gaussians.get_xyz.detach()
        if self._device is not None:
            xyz = xyz.to(self._device)
        center_px, depth = project_points(camera, xyz)
        center_px = center_px * self._pixel_scale
        scaling = self._gaussians.get_scaling.detach().to(xyz.device)
        world_sigma = (
            scaling[:, :3].mean(dim=1) if scaling.shape[1] >= 3 else scaling.reshape(-1)
        )
        fl_x, fl_y, _, _ = _camera_intrinsics(camera)
        focal = 0.5 * (fl_x + fl_y) * self._pixel_scale
        sigma_px = (world_sigma * focal / depth.clamp_min(1e-8)).clamp_min(_MIN_SIGMA_PX)
        geometry = ProjectedGeometry(
            center_px=center_px, depth=depth, sigma_px=sigma_px
        )
        self._static_geometry[key] = geometry
        return geometry

    def _branch_product(
        self, geometry: ProjectedGeometry, alpha: torch.Tensor,
        pixel: tuple[float, float], depth: float,
    ) -> float:
        """(1 - a_i G_i) product over one branch's front set."""
        front = (geometry.depth > 0.0) & (geometry.depth < float(depth))
        if not bool(front.any()):
            return 1.0
        px = torch.tensor(
            [float(pixel[0]), float(pixel[1])],
            device=geometry.center_px.device,
            dtype=geometry.center_px.dtype,
        )
        offset = geometry.center_px[front] - px
        sigma = geometry.sigma_px[front]
        footprint = torch.exp(-0.5 * (offset * offset).sum(dim=1) / (sigma * sigma))
        keep = footprint > FOOTPRINT_CUTOFF
        if not bool(keep.any()):
            return 1.0
        # `front`/`keep` are derived from the geometry, `alpha` from the
        # presence column, and the two can be built on different devices
        # (the runtime's device is chosen at setup, the model's by the
        # trainer). Aligning here makes the probe correct regardless,
        # rather than relying on every caller to agree — the mismatch
        # that killed experiment 75 at its first real q evaluation.
        alpha = alpha.to(footprint.device)
        contribution = (alpha[front][keep] * footprint[keep]).clamp(0.0, 1.0)
        return float(
            torch.exp(torch.log1p(-contribution.clamp_max(1.0 - 1e-12)).sum())
        )

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
        # Validate the frame FIRST. The early returns below (empty model,
        # empty front set, everything pruned) would otherwise let an
        # unmapped frame index return 1.0 instead of failing closed,
        # because only `_alpha_for` consults the frame->time map.
        self._timestamp_for(frame, self._camera(camera_id))
        geometry = self._geometry_for(camera_id, frame)
        if geometry.depth.numel() == 0:
            return 1.0

        # BOTH branches the rasterizer composites. Strict front set on
        # each: splats at or behind the query point never composite,
        # which is also what stands in for query-source exclusion
        # model-side. The spatial prune's cutoff is PER SPLAT, so its
        # aggregate error grows with the kept-set size, and it RAISES T
        # (drops occluders) -- recorded, not claimed to be conservative.
        alpha = self._alpha_for(frame, self._camera(camera_id), present_family)
        t_value = self._branch_product(geometry, alpha, pixel, depth)

        static_alpha = self._static_alpha()
        if static_alpha is not None:
            t_value *= self._branch_product(
                self._static_geometry_for(camera_id), static_alpha, pixel, depth
            )
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
        pixel = pixel * self._pixel_scale
        if float(depth[0]) <= float(getattr(camera, "znear", 0.01)):
            return False
        u, v = float(pixel[0, 0]), float(pixel[0, 1])
        width, height = self._report_raster(camera)
        return 0.0 <= u <= width - 1.0 and 0.0 <= v <= height - 1.0

    def _report_raster(self, camera) -> tuple[float, float]:
        """The camera's raster in REPORT pixels (bounds must scale with
        the projected pixels, or a downscaled model rejects most of its
        own frustum)."""
        return (
            camera.image_width * self._pixel_scale,
            camera.image_height * self._pixel_scale,
        )

    @torch.no_grad()
    def project(self, camera_id: int, point: tuple[float, float, float]):
        """(pixel, depth) for one world point, or None outside the frustum.

        The `sigma_points_for` projector contract in `elgs.evidence_stack`.
        """
        camera = self._camera(camera_id)
        device = self._gaussians.get_xyz.device
        tensor = torch.tensor([list(point)], device=device, dtype=torch.float32)
        pixel, depth = project_points(camera, tensor)
        pixel = pixel * self._pixel_scale
        d = float(depth[0])
        if d <= float(getattr(camera, "znear", 0.01)):
            return None
        u, v = float(pixel[0, 0]), float(pixel[0, 1])
        width, height = self._report_raster(camera)
        if not (0.0 <= u <= width - 1.0 and 0.0 <= v <= height - 1.0):
            return None
        return (u, v), d


__all__ = ["FOOTPRINT_CUTOFF", "ModelProbe", "ProjectedGeometry", "project_points"]
