"""CCR B1 observation-born packet birth (ccr-method-2026-08-20 §3.1).

Budget-neutral spacetime relocation during training: every ``interval``
iterations inside the birth window, the bottom ``fraction`` of dynamic rows
ranked by (activated opacity x screen radius) are relocated onto high
photometric-error pixels of the current training view, backprojected through
the rendered depth at that view's timestamp. Relocated rows take the site as
``xyz``, the view timestamp as the temporal mean, a narrow temporal scale, the
observed pixel colour as DC, and a reset opacity. Each event stamps a PACKET
id on the cohort's spatial connected components (per-row long column,
-1 = substrate).

The transaction itself is the established point-neutral substrate
(:func:`depth_visibility.capacity.apply_point_neutral_transaction` over
``GaussianModel.build_capacity_bank``), so the row count, the parameter
identities and the Adam moment reset are handled exactly as by the CSVL-VPL
lifecycle birth in :mod:`scene.lifecycle`.

B1-D (``packet_birth_dynamic_mask_donors``, default False) is the one-variable
donor-side variant pre-identified in ccr-ladder-round1-results-2026-08-20 §6.3:
a row is an eligible donor only when its motion-model centre projects inside
the current training view's dynamic mask, so static-support rows are never
harvested. Ranking among eligible rows is the unchanged (opacity x radius)
rule; see :func:`donor_dynamic_mask_eligibility` for the frozen semantics.

B1-F/B1-X (``packet_birth_flow_init``, ``packet_birth_flow_source``, default
False/"correct") is the flow-derived VELOCITY INITIALIZATION variant: relocated
rows are born motionless here, and with the flag on they instead start from a
SEA-RAFT flow-derived world velocity projected into the LoRA motion basis. All
of that logic lives in :mod:`scene.packet_birth_flow`; this module only threads
the two knobs and hands the solved coefficients to
:func:`_build_packet_target_rows`.

Every entry point is a no-op unless ``packet_birth_enable`` is set, so a B0
cell is bit-identical to the substrate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import os
from typing import Any, Optional

import torch

from depth_visibility.capacity import apply_point_neutral_transaction
from depth_visibility.errors import ContractError
from utils.motion_prior_utils import project_points_to_screen
from utils.sh_utils import RGB2SH

PACKET_BIRTH_SCHEMA = "ccr-b1-packet-birth-v1"
PACKET_STATE_FILENAME = "packet_state.pt"

#: A spatial component with fewer rows than this stays substrate (-1).
MIN_PACKET_ROWS = 4
#: Component epsilon = this multiple of the median nearest-neighbour distance
#: over the event's sampled sites.
PACKET_EPSILON_SCALE = 3.0
#: Reset opacity of a relocated row; matches create_from_pcd
#: (scene/gaussian_model.py:1032 `inverse_sigmoid(0.1 * ones(...))`).
_BIRTH_OPACITY = 0.1


def _inverse_sigmoid(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ContractError("inverse sigmoid requires a probability in (0, 1)")
    return math.log(value / (1.0 - value))


def _row_count(gaussians: Any) -> int:
    xyz = getattr(gaussians, "_xyz", None)
    if not torch.is_tensor(xyz):
        raise ContractError("packet birth requires a Gaussian model exposing _xyz")
    return int(xyz.shape[0])


@dataclass
class PacketBirthConfig:
    """Frozen packet-birth knobs (mirrored by the trainer's OptimizationParams)."""

    enable: bool = False
    interval: int = 500
    fraction: float = 0.005
    t_sigma_frames: float = 1.5
    from_iter: int = 1000
    until_iter: int = -1
    #: B1-D. False = B1 exactly (every row is an eligible donor).
    dynamic_mask_donors: bool = False
    #: B1-F. False = B1 exactly (relocated rows are born motionless).
    flow_init: bool = False
    #: B1-F arm ("correct") vs B1-X wrong-flow control ("camera_swapped").
    flow_source: str = "correct"

    def validate(self) -> "PacketBirthConfig":
        if not self.enable:
            return self
        if int(self.interval) <= 0:
            raise ContractError("packet_birth_interval must be positive when enabled")
        if not 0.0 < float(self.fraction) <= 1.0:
            raise ContractError("packet_birth_fraction must lie in (0, 1]")
        if float(self.t_sigma_frames) <= 0.0:
            raise ContractError("packet_birth_t_sigma_frames must be positive")
        if int(self.from_iter) <= 0:
            raise ContractError("packet_birth_from_iter must be positive")
        if int(self.until_iter) < int(self.from_iter):
            raise ContractError("packet_birth_until_iter must not precede packet_birth_from_iter")
        from scene.packet_birth_flow import PACKET_FLOW_SOURCES

        if str(self.flow_source) not in PACKET_FLOW_SOURCES:
            raise ContractError(
                "packet_birth_flow_source must be one of {}; got {!r}".format(
                    ", ".join(PACKET_FLOW_SOURCES), self.flow_source
                )
            )
        return self

    @classmethod
    def from_namespace(
        cls,
        namespace: Any,
        prefix: str = "packet_birth_",
        densify_until_iter: Optional[int] = None,
    ) -> "PacketBirthConfig":
        """Build a config from a trainer namespace using ``prefix`` + field name."""

        values = {}
        for spec in fields(cls):
            attribute = "{}{}".format(prefix, spec.name)
            if hasattr(namespace, attribute):
                values[spec.name] = getattr(namespace, attribute)
        cfg = cls(**values)
        if int(cfg.until_iter) < 0:
            cfg.until_iter = int(densify_until_iter or 0)
        return cfg.validate()

    def as_dict(self) -> dict:
        return asdict(self)


class PacketBirthState:
    """Cadence, packet-id allocation and the per-event log."""

    def __init__(
        self,
        cfg: PacketBirthConfig,
        frame_dt: float,
        flow_assets: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg.validate()
        if float(frame_dt) <= 0.0:
            raise ContractError("packet birth requires a positive frame dt")
        self.frame_dt = float(frame_dt)
        # B1-F flow asset provider; None for B1 and for every B0 cell. See
        # scene.packet_birth_flow.build_flow_assets for the trainer wiring.
        self.flow_assets = flow_assets
        self.events: list = []
        self.next_packet_id = 0
        self.rows_assigned_total = 0
        self.rows_relocated_total = 0
        # B1-D donor-eligibility bookkeeping (0 for every B1 cell).
        self.donors_requested_total = 0
        self.donors_eligible_total = 0
        self.donors_shortfall_total = 0
        self.events_skipped_no_dynamic_mask = 0

    @property
    def sigma_t(self) -> float:
        return float(self.cfg.t_sigma_frames) * self.frame_dt

    def should_fire(self, iteration: int) -> bool:
        cfg = self.cfg
        iteration = int(iteration)
        if not cfg.enable:
            return False
        if iteration < int(cfg.from_iter) or iteration > int(cfg.until_iter):
            return False
        return iteration % int(cfg.interval) == 0

    def summary(self) -> dict:
        return {
            "schema_version": PACKET_BIRTH_SCHEMA,
            "config": self.cfg.as_dict(),
            "frame_dt": self.frame_dt,
            "sigma_t": self.sigma_t,
            "events": len(self.events),
            "packets": int(self.next_packet_id),
            "rows_assigned_total": int(self.rows_assigned_total),
            "rows_relocated_total": int(self.rows_relocated_total),
            # B1-D aggregates; every per-event field is also in the event log.
            "donor_mask": bool(self.cfg.dynamic_mask_donors),
            "donors_requested_total": int(self.donors_requested_total),
            "donors_eligible_total": int(self.donors_eligible_total),
            "donors_shortfall_total": int(self.donors_shortfall_total),
            "events_skipped_no_dynamic_mask": int(self.events_skipped_no_dynamic_mask),
        }


def setup_packet_birth(opt: Any, flow_assets: Optional[Any] = None) -> Optional[PacketBirthState]:
    """Return the packet-birth state, or ``None`` when the operator is off.

    ``flow_assets`` is the B1-F flow provider (
    :func:`scene.packet_birth_flow.build_flow_assets`). It defaults to None so
    a B0/B1/B1-D call site is unchanged; a ``packet_birth_flow_init: true`` run
    without it raises ``ContractError`` at its first birth event.
    """

    if not bool(getattr(opt, "packet_birth_enable", False)):
        return None
    cfg = PacketBirthConfig.from_namespace(
        opt, densify_until_iter=getattr(opt, "densify_until_iter", 0)
    )
    return PacketBirthState(
        cfg,
        frame_dt=float(getattr(opt, "motion_track_dt", 1.0 / 30.0)),
        flow_assets=flow_assets,
    )


def ensure_packet_column(gaussians: Any) -> torch.Tensor:
    """Lazily size ``_packet_ids`` to the row count, filled with -1."""

    count = _row_count(gaussians)
    ids = getattr(gaussians, "_packet_ids", None)
    if ids is None or int(ids.numel()) == 0:
        ids = torch.full((count,), -1, dtype=torch.long)
        gaussians._packet_ids = ids
        return ids
    if int(ids.shape[0]) != count:
        raise ContractError(
            "packet id column rows ({}) do not match Gaussian rows ({}); "
            "the prune/densify maintenance hooks are not wired".format(
                int(ids.shape[0]), count
            )
        )
    return ids


def requested_donor_count(rows: int, fraction: float) -> int:
    """The §3.1 cohort size the ranking rule asks for: ``int(fraction * rows)``."""

    return int(float(fraction) * int(rows))


def select_packet_donors(
    opacity_logit: torch.Tensor,
    max_radii2D: torch.Tensor,
    fraction: float,
    eligible: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Bottom-``fraction`` dynamic rows by activated opacity x screen radius.

    This is the §3.1 relocation score, not
    :func:`depth_visibility.capacity.select_event_blind_donors`, whose
    redundancy-witness and age rails answer the Slice-B question instead.
    ``torch.argsort`` is called without ``stable=`` because that keyword needs
    torch >= 2.0 and the local test environment is 1.12.

    ``eligible`` is the B1-D donor gate: an optional per-row boolean mask.
    ``None`` (the default, and the whole of B1) means every row is eligible and
    the returned cohort is bit-identical to the original rule. When it is
    given, the SAME (opacity x radius) ranking is applied to the eligible rows
    ONLY, and the cohort is the bottom ``min(k, eligible)`` of those -- an
    ineligible row is never a donor, and no fallback to ineligible rows exists.
    """

    count = int(opacity_logit.shape[0])
    k = requested_donor_count(count, fraction)
    if count == 0 or k <= 0:
        return torch.zeros(0, dtype=torch.long, device=opacity_logit.device)
    with torch.no_grad():
        opacity = torch.sigmoid(opacity_logit.detach().reshape(count))
        radii = max_radii2D.detach().reshape(count).to(opacity.dtype).clamp(min=1.0)
        score = opacity * radii
        if eligible is None:
            return torch.argsort(score)[:k]
        rows = eligible.reshape(count).to(score.device).nonzero().reshape(-1)
        if int(rows.numel()) == 0:
            return torch.zeros(0, dtype=torch.long, device=opacity_logit.device)
        order = torch.argsort(score.index_select(0, rows))
        return rows.index_select(0, order)[:k].to(opacity_logit.device)


def donor_dynamic_mask_eligibility(
    gaussians: Any,
    viewpoint_camera: Any,
    dynamic_mask: torch.Tensor,
    timestamp: float,
) -> torch.Tensor:
    """Per-row B1-D donor eligibility: centre inside the view's dynamic mask.

    FROZEN SEMANTICS -- a row is eligible if and only if ALL of the following
    hold, and is ineligible otherwise (fail-closed, no fallback):

    * its centre is the MOTION-MODEL position ``get_dynamic_xyz(timestamp)``
      at the current training timestamp -- the same call the renderer
      rasterizes (gaussian_renderer/__init__.py:200) and the same one
      ``compute_dynamic_densify_weight`` uses (main.py:87) -- NOT the
      canonical ``_xyz``;
    * the centre is IN FRONT of the camera: its view-space z, taken through
      ``world_view_transform`` (the TRANSPOSED world-to-camera matrix,
      scene/cameras.py:67, the same convention as
      :func:`camera_forward_axis`), is finite and strictly positive;
    * the projection is valid and in frame: ``project_points_to_screen``
      (utils/motion_prior_utils.py:143, the projector already used by the
      dynamic-densify weight) reports ``valid`` -- finite ``w`` and NDC inside
      [-1, 1] -- and the NEAREST pixel ``(round(x), round(y))`` lies inside
      ``[0, W) x [0, H)``;
    * ``mask[y, x] > 0.5`` at that nearest pixel.

    ``dynamic_mask`` is the trainer's mask for this view -- ``(1, H, W)``
    float in [0, 1] as produced by ``MotionPriorCache.get_dynamic_mask`` --
    and is reshaped to ``(H, W)`` exactly as the residual gate in
    :func:`_packet_birth` already reshapes it. NEAREST-pixel lookup with a
    strict 0.5 threshold is deliberate: it makes eligibility a hard
    set-membership test rather than the bilinear weight
    ``sample_mask_at_points`` returns.

    Note the half-pixel convention difference that is inherent to the two
    existing utilities: ``project_points_to_screen`` maps NDC onto
    ``[0, W - 1]`` while ``Camera.get_rays`` (used by the backprojection above)
    uses pixel CENTRES. At nearest-pixel resolution the difference is at most
    one pixel and is not corrected here, so that the projector stays the
    tested one.
    """

    getter = getattr(gaussians, "get_dynamic_xyz", None)
    if getter is None:
        raise ContractError(
            "dynamic-mask donor gating requires a model exposing get_dynamic_xyz"
        )
    points = getter(float(timestamp)).detach().reshape(-1, 3)
    count = int(points.shape[0])
    if count == 0:
        return torch.zeros(0, dtype=torch.bool, device=points.device)
    height = int(getattr(viewpoint_camera, "image_height", 0))
    width = int(getattr(viewpoint_camera, "image_width", 0))
    if height <= 0 or width <= 0:
        raise ContractError("dynamic-mask donor gating requires a sized camera")
    mask = dynamic_mask.detach().reshape(height, width).to(points.device).reshape(-1)

    world_view = viewpoint_camera.world_view_transform.to(
        device=points.device, dtype=points.dtype
    )
    ones = torch.ones((count, 1), dtype=points.dtype, device=points.device)
    camera_z = (torch.cat([points, ones], dim=-1) @ world_view)[:, 2]

    screen_xy, valid = project_points_to_screen(points, viewpoint_camera)
    screen_xy = torch.nan_to_num(screen_xy, nan=-1.0, posinf=-1.0, neginf=-1.0)
    x = screen_xy[:, 0].round().long()
    y = screen_xy[:, 1].round().long()
    eligible = (
        valid.reshape(count).to(points.device)
        & torch.isfinite(camera_z)
        & (camera_z > 0)
        & (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )
    index = y.clamp(0, height - 1) * width + x.clamp(0, width - 1)
    return eligible & (mask.index_select(0, index) > 0.5)


def sample_residual_sites(
    residual: torch.Tensor, count: int, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Sample flat pixel indices proportional to residual^2, without replacement."""

    weights = residual.detach().reshape(-1).to(torch.float32).clamp(min=0.0) ** 2
    available = int((weights > 0).sum().item())
    k = min(int(count), available)
    if k <= 0:
        return torch.zeros(0, dtype=torch.long, device=weights.device)
    return torch.multinomial(weights, k, replacement=False, generator=generator)


def camera_forward_axis(viewpoint_camera: Any) -> torch.Tensor:
    """World-frame unit vector along the camera's +z (view) axis.

    ``world_view_transform`` is the transposed world-to-camera matrix
    (scene/cameras.py:67), so ``inv(world_view_transform.T)`` is the ordinary
    camera-to-world matrix and its third column is the view axis in world
    coordinates -- exactly the convention ``Camera.get_rays`` uses when it
    builds directions from ``c2w`` (scene/cameras.py:114).
    """

    c2w = torch.linalg.inv(viewpoint_camera.world_view_transform.transpose(0, 1))
    return c2w[:3, 2]


def backproject_camera_z(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    forward: torch.Tensor,
    camera_z: torch.Tensor,
) -> torch.Tensor:
    """World points for CAMERA-Z depths sampled along unit rays.

    DEPTH SEMANTICS (determined from the rasterizer, not assumed): the CUDA
    forward pass stores ``depths[idx] = p_view.z`` -- the view-space z of the
    primitive centre (diff-gaussian-rasterization/cuda_rasterizer/forward.cu:489)
    -- and accumulates ``D += depths[...] * alpha * T``
    (forward.cu:761), writing it to ``out_depth`` (forward.cu:798). The python
    binding returns that tensor as ``depth`` and ``1 - T`` as ``alpha``
    (gaussian_renderer/diff_gaussian_rasterization.py:174). So the render
    output is an alpha-WEIGHTED, un-normalized CAMERA-Z map: it must be divided
    by alpha to become an expected camera-z, and it is NOT an along-ray
    distance. ``Camera.get_rays`` returns UNIT directions
    (scene/cameras.py:117), so the along-ray distance for a camera-z value
    ``z`` is ``z / dot(ray_d, forward)``.
    """

    if rays_d.dim() != 2 or int(rays_d.shape[1]) != 3:
        raise ContractError("backprojection requires (K, 3) ray directions")
    if int(camera_z.shape[0]) != int(rays_d.shape[0]):
        raise ContractError("backprojection ray and depth counts disagree")
    cosine = (rays_d * forward.reshape(1, 3).to(rays_d.dtype)).sum(dim=1)
    distance = camera_z.reshape(-1).to(rays_d.dtype) / cosine
    return rays_o.reshape(1, 3).to(rays_d.dtype) + rays_d * distance.reshape(-1, 1)


def backproject_pixels(
    viewpoint_camera: Any, depth: torch.Tensor, alpha: torch.Tensor, pixels: torch.Tensor
):
    """Backproject flat pixel indices through the rendered depth.

    Returns ``(points, valid)``; ``valid`` is False where the expected
    camera-z is not strictly positive and finite (a pixel with no accumulated
    alpha renders depth 0 and would otherwise collapse onto the camera centre)
    or where the ray points away from the view axis.
    """

    if float(getattr(viewpoint_camera, "fl_x", -1.0)) <= 0.0:
        raise ContractError(
            "packet birth backprojection requires pinhole intrinsics on the camera"
        )
    rays_o, rays_d = viewpoint_camera.get_rays()
    rays_d = rays_d.reshape(-1, 3).index_select(0, pixels.to(rays_d.device))
    forward = camera_forward_axis(viewpoint_camera).to(rays_d.device)
    depth_flat = depth.detach().reshape(-1).index_select(0, pixels.to(depth.device))
    alpha_flat = alpha.detach().reshape(-1).index_select(0, pixels.to(alpha.device))
    camera_z = depth_flat / alpha_flat.clamp(min=1e-6)
    camera_z = camera_z.to(rays_d.device)
    cosine = (rays_d * forward.reshape(1, 3).to(rays_d.dtype)).sum(dim=1)
    valid = torch.isfinite(camera_z) & (camera_z > 0) & (cosine > 1e-6)
    points = backproject_camera_z(rays_o, rays_d, forward, camera_z)
    return points, valid


def packet_components(
    points: torch.Tensor,
    min_rows: int = MIN_PACKET_ROWS,
    epsilon_scale: float = PACKET_EPSILON_SCALE,
):
    """Label sites by epsilon-graph connected components.

    ``epsilon`` is ``epsilon_scale`` times the median nearest-neighbour
    distance over the sites. Components with at least ``min_rows`` members get
    consecutive local labels 0..P-1; every smaller residue keeps -1.
    """

    count = int(points.shape[0])
    labels = torch.full((count,), -1, dtype=torch.long, device=points.device)
    if count < int(min_rows):
        return labels, 0
    coordinates = points.detach().to(torch.float32)
    distances = torch.cdist(coordinates, coordinates)
    infinity = torch.full_like(distances, float("inf"))
    nearest = torch.where(
        torch.eye(count, dtype=torch.bool, device=distances.device), infinity, distances
    ).min(dim=1).values
    epsilon = float(epsilon_scale) * float(nearest.median().item())
    adjacency = distances <= epsilon
    # Min-label propagation with pointer jumping: the row label of a node is
    # the smallest node index reachable in the epsilon graph, so equal labels
    # mean one connected component.
    component = torch.arange(count, dtype=torch.long, device=distances.device)
    sentinel = component.new_full((count, count), count)
    for _ in range(count):
        candidates = torch.where(adjacency, component.reshape(1, count).expand(count, count), sentinel)
        propagated = candidates.min(dim=1).values
        propagated = propagated.index_select(0, propagated)
        if torch.equal(propagated, component):
            break
        component = propagated
    roots, inverse, counts = torch.unique(component, return_inverse=True, return_counts=True)
    keep = counts >= int(min_rows)
    local = torch.full((int(roots.numel()),), -1, dtype=torch.long, device=distances.device)
    kept = int(keep.sum().item())
    if kept:
        local[keep] = torch.arange(kept, dtype=torch.long, device=distances.device)
    labels = local.index_select(0, inverse)
    return labels, kept


def _median_log_scaling(scaling: torch.Tensor) -> torch.Tensor:
    """Per-axis median log-scaling over the dynamic rows."""

    detached = scaling.detach()
    return detached.median(dim=0).values.reshape(1, *detached.shape[1:])


def _build_packet_target_rows(
    bank: Any,
    donors: torch.Tensor,
    points: torch.Tensor,
    colors: torch.Tensor,
    timestamp: float,
    sigma_t: float,
    route_logit_init: Optional[float],
    motion_lora_coeff: Optional[torch.Tensor] = None,
) -> dict:
    """Target rows for one relocation cohort (§3.1).

    ``motion_lora_coeff`` is the B1-F flow-derived velocity initialization for
    ``_motion_lora_coeff``. ``None`` (B1, B1-D, and every B0 cell) leaves that
    parameter in the generic ``else`` branch below, i.e. exactly zero.
    """

    k = int(points.shape[0])
    rows = {}
    for name, parameter in bank.parameters.items():
        trailing = tuple(parameter.shape[1:])
        device = parameter.device
        dtype = parameter.dtype
        if name == "_xyz":
            if trailing != (3,):
                raise ContractError("packet birth requires a 3D _xyz parameter")
            value = points.to(device=device, dtype=dtype)
        elif name == "_features_dc":
            value = torch.zeros((k, *trailing), dtype=dtype, device=device)
            flat = value.reshape(k, -1)
            if flat.shape[1] < 3:
                raise ContractError("packet birth requires at least three DC feature channels")
            flat[:, :3] = RGB2SH(colors.clamp(0.0, 1.0)).to(device=device, dtype=dtype)
        elif name == "_scaling":
            # Donor scaling is KEPT (the relocated row inherits a trained
            # footprint) but clamped elementwise to the per-axis median
            # log-scaling of the dynamic rows, so an outsized donor cannot
            # blanket the new site. Chosen over a knn re-fit because it needs
            # no new geometry query and no new constant.
            donor_scaling = parameter.detach().index_select(0, donors.to(device))
            value = torch.minimum(donor_scaling, _median_log_scaling(parameter).to(device))
            value = value.to(dtype=dtype)
        elif name == "_rotation":
            value = torch.zeros((k, *trailing), dtype=dtype, device=device)
            if value.reshape(k, -1).shape[1] < 1:
                raise ContractError("packet birth requires a quaternion rotation parameter")
            value.reshape(k, -1)[:, 0] = 1.0
        elif name == "_opacity":
            value = torch.full(
                (k, *trailing), _inverse_sigmoid(_BIRTH_OPACITY), dtype=dtype, device=device
            )
        elif name == "_t":
            value = torch.full((k, *trailing), float(timestamp), dtype=dtype, device=device)
        elif name == "_scaling_t":
            value = torch.full(
                (k, *trailing), math.log(float(sigma_t)), dtype=dtype, device=device
            )
        elif name == "_route_logit":
            fill = 0.0 if route_logit_init is None else float(route_logit_init)
            value = torch.full((k, *trailing), fill, dtype=dtype, device=device)
        elif name == "_motion_lora_coeff" and motion_lora_coeff is not None:
            # B1-F: flow-derived velocity in place of the zero initialization.
            value = motion_lora_coeff.to(device=device, dtype=dtype)
            if tuple(value.shape) != (k, *trailing):
                raise ContractError(
                    "flow-derived motion coefficients have shape {} but the "
                    "parameter needs {}".format(tuple(value.shape), (k, *trailing))
                )
        else:
            # motion coefficients (poly and LoRA) and the staticness score
            value = torch.zeros((k, *trailing), dtype=dtype, device=device)
        rows[name] = value
    return rows


def maybe_packet_birth(
    state: Optional[PacketBirthState],
    gaussians: Any,
    iteration: int,
    viewpoint_camera: Any,
    image: torch.Tensor,
    gt_image: torch.Tensor,
    depth: torch.Tensor,
    alpha: torch.Tensor,
    dynamic_mask: Optional[torch.Tensor] = None,
) -> Optional[dict]:
    """Run one budget-neutral packet-birth event when the cadence fires."""

    if state is None or not state.should_fire(iteration):
        return None
    with torch.no_grad():
        record = _packet_birth(
            state, gaussians, int(iteration), viewpoint_camera, image, gt_image,
            depth, alpha, dynamic_mask,
        )
    state.events.append(record)
    return record


def _packet_birth(
    state: PacketBirthState,
    gaussians: Any,
    iteration: int,
    viewpoint_camera: Any,
    image: torch.Tensor,
    gt_image: torch.Tensor,
    depth: torch.Tensor,
    alpha: torch.Tensor,
    dynamic_mask: Optional[torch.Tensor],
) -> dict:
    timestamp = float(getattr(viewpoint_camera, "timestamp", 0.0))
    donor_mask = bool(state.cfg.dynamic_mask_donors)
    record = {
        "iteration": int(iteration),
        "event_id": len(state.events),
        "donors": 0,
        "packets": 0,
        "rows_assigned": 0,
        "camera": str(getattr(viewpoint_camera, "image_name", "")),
        "timestamp": timestamp,
        "reason": None,
        # B1-D donor-eligibility accounting. ``requested`` is what the
        # fraction rule asks for, ``eligible_donors`` how many rows pass the
        # gate (all rows when the gate is off), ``realized`` how many rows the
        # event actually relocated, and ``shortfall`` the eligibility deficit
        # max(0, requested - eligible_donors) -- never a depth-validity loss.
        "donor_mask": donor_mask,
        "requested": 0,
        "eligible_donors": 0,
        "realized": 0,
        "shortfall": 0,
    }
    # B1-F funnel. Present on EVERY record, including the early-exit paths, so
    # that a run whose flow never applied is detectable from the record alone.
    from scene import packet_birth_flow

    record.update(packet_birth_flow.idle_record_fields(state.cfg))
    optimizer = getattr(gaussians, "optimizer", None)
    if optimizer is None:
        raise ContractError("packet birth requires an initialized optimizer")
    if donor_mask and dynamic_mask is None:
        # Frozen B1-D rule: without a mask there is no eligibility evidence, so
        # the event is skipped whole -- no relocation, no packet id consumed,
        # and not even the lazy packet column is materialized.
        record["reason"] = "no_dynamic_mask"
        state.events_skipped_no_dynamic_mask += 1
        return record
    packet_ids = ensure_packet_column(gaussians)
    bank = gaussians.build_capacity_bank()
    if "max_radii2D" not in bank.accumulators:
        raise ContractError("packet birth requires the max_radii2D accumulator")

    rows = _row_count(gaussians)
    requested = requested_donor_count(rows, state.cfg.fraction)
    if donor_mask:
        eligible = donor_dynamic_mask_eligibility(
            gaussians, viewpoint_camera, dynamic_mask, timestamp
        )
        if int(eligible.numel()) != rows:
            raise ContractError(
                "dynamic-mask donor gate returned {} rows for {} Gaussians".format(
                    int(eligible.numel()), rows
                )
            )
        eligible_count = int(eligible.sum().item())
    else:
        eligible = None
        eligible_count = rows
    record["requested"] = requested
    record["eligible_donors"] = eligible_count
    record["shortfall"] = max(0, requested - eligible_count)
    state.donors_requested_total += requested
    state.donors_eligible_total += eligible_count
    state.donors_shortfall_total += int(record["shortfall"])

    donors = select_packet_donors(
        bank.parameters["_opacity"],
        bank.accumulators["max_radii2D"],
        state.cfg.fraction,
        eligible=eligible,
    )
    if int(donors.numel()) == 0:
        record["reason"] = "no_donors"
        return record

    residual = (image.detach() - gt_image.detach()).abs().mean(dim=0)
    if dynamic_mask is not None:
        residual = residual * dynamic_mask.detach().reshape(residual.shape).to(residual.device)
    pixels = sample_residual_sites(residual, int(donors.numel()))
    if int(pixels.numel()) == 0:
        record["reason"] = "no_residual_sites"
        return record

    points, valid = backproject_pixels(viewpoint_camera, depth, alpha, pixels)
    points = points[valid]
    pixels = pixels.to(valid.device)[valid]
    if int(points.shape[0]) == 0:
        record["reason"] = "no_valid_depth"
        return record
    donors = donors[: int(points.shape[0])]
    points = points[: int(donors.numel())]
    pixels = pixels[: int(donors.numel())]

    colors = gt_image.detach()[:3].reshape(3, -1).index_select(
        1, pixels.to(gt_image.device)
    ).transpose(0, 1).to(torch.float32)

    motion_lora_coeff = None
    if bool(state.cfg.flow_init):
        if "_motion_lora_coeff" not in bank.parameters:
            raise ContractError(
                "packet_birth_flow_init is on but the capacity bank exposes no "
                "_motion_lora_coeff parameter to initialize"
            )
        motion_lora_coeff, flow_fields = packet_birth_flow.flow_motion_lora_coefficients(
            state, gaussians, viewpoint_camera, pixels, points
        )
        record.update(flow_fields)

    target_rows = _build_packet_target_rows(
        bank,
        donors,
        points,
        colors,
        timestamp,
        state.sigma_t,
        getattr(gaussians, "route_logit_init", None),
        motion_lora_coeff=motion_lora_coeff,
    )
    transaction = apply_point_neutral_transaction(
        bank, optimizer, donors, target_rows, iteration=int(iteration), mode="reassign"
    )
    if transaction["budget_before"] != transaction["budget_after"]:
        raise ContractError("packet birth changed the capacity budget")

    labels, packets = packet_components(points)
    labels = labels.to(packet_ids.device)
    assigned = labels >= 0
    donor_rows = donors.to(packet_ids.device)
    packet_ids[donor_rows] = -1
    if packets:
        packet_ids[donor_rows[assigned]] = labels[assigned] + int(state.next_packet_id)
    state.next_packet_id += int(packets)
    rows_assigned = int(assigned.sum().item())
    state.rows_assigned_total += rows_assigned
    state.rows_relocated_total += int(donors.numel())

    record["donors"] = int(donors.numel())
    record["realized"] = int(donors.numel())
    record["packets"] = int(packets)
    record["rows_assigned"] = rows_assigned
    return record


def write_packet_state(model_path: str, gaussians: Any, state: Optional[PacketBirthState]):
    """Persist the packet column and the event log; ``None`` when no packet exists."""

    if state is None or int(state.next_packet_id) <= 0:
        return None
    packet_ids = getattr(gaussians, "_packet_ids", None)
    if packet_ids is None or int(packet_ids.numel()) == 0:
        return None
    path = os.path.join(str(model_path), PACKET_STATE_FILENAME)
    torch.save(
        {
            "schema_version": PACKET_BIRTH_SCHEMA,
            "packet_ids": packet_ids.detach().cpu(),
            "event_log": list(state.events),
            "config": state.cfg.as_dict(),
        },
        path,
    )
    return path


__all__ = [
    "MIN_PACKET_ROWS",
    "PACKET_BIRTH_SCHEMA",
    "PACKET_EPSILON_SCALE",
    "PACKET_STATE_FILENAME",
    "PacketBirthConfig",
    "PacketBirthState",
    "backproject_camera_z",
    "backproject_pixels",
    "camera_forward_axis",
    "donor_dynamic_mask_eligibility",
    "ensure_packet_column",
    "maybe_packet_birth",
    "packet_components",
    "requested_donor_count",
    "sample_residual_sites",
    "select_packet_donors",
    "setup_packet_birth",
    "write_packet_state",
]
