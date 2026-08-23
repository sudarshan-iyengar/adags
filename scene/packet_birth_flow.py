"""B1-F / B1-X flow-derived VELOCITY INITIALIZATION for relocated packet rows.

Birth SITES are already restricted by the dynamic mask
(:func:`scene.packet_birth._packet_birth` multiplies the photometric residual
by that mask before sampling), so a flow-gated site rule would be largely
redundant. The information dense optical flow uniquely adds over a binary
dynamic mask is motion DIRECTION and MAGNITUDE. Relocated rows are born
motionless today -- ``_build_packet_target_rows`` sends every motion
coefficient through its ``else`` branch and writes zeros. This module tests
whether initializing them with a flow-derived velocity helps.

FROZEN PROCEDURE (one packet-birth event, all rows at once)

1. Read the SEA-RAFT forward flow ``f(p) = (u, v)`` and its per-pixel validity
   mask ``m(p)`` for the training camera at the event's frame, through
   ``MotionPriorCache.get_track_flow`` -- direction ``forward_t_to_t_plus_1``,
   within one camera only, PIXELS at the raster the trainer renders.
2. ``m(p)`` False, or ``f(p)`` non-finite -> the row FAILS CLOSED and keeps the
   zero-motion initialization. Counted as ``flow_failed_mask`` /
   ``flow_failed_nonfinite``.
3. ``p' = p + f(p)``; outside the raster -> fail closed, ``flow_failed_bounds``.
4. INPUT-SIDE OUTLIER REJECTION: a site whose flow MAGNITUDE exceeds the
   :data:`FLOW_OUTLIER_QUANTILE` quantile of THAT VIEW's valid flow magnitudes
   fails closed, counted as ``flow_failed_outlier``. The guard lives in PIXELS,
   where the quantity is physical and directly measurable against the rest of
   the same SEA-RAFT field, and it can only REMOVE extreme sites -- it can
   never manufacture an effect.
5. Backproject BOTH ``p`` and ``p'`` through the SAME rendered camera-z sampled
   at ``p``. This is a CONSTANT-DEPTH approximation, deliberately chosen and
   disclosed: no render exists at ``t + 1`` inside the training loop, so the
   scene flow implied by the pixel flow cannot be measured, only the
   image-plane component of it at the depth we do have.
6. ``v = (X1 - X0) / dt_frame`` in world units per TRAINER time unit.
7. Project ``v`` into the LoRA motion basis at probe offset ``dt_p = dt_frame``
   (see :func:`lora_probe_basis` / :func:`solve_lora_coefficients`).
8. The solved coefficients replace the zeros for those rows. A row whose solved
   coefficient or realized displacement is non-finite fails closed, counted
   into ``flow_failed_nonfinite``.

NO OUTPUT-SIDE MAGNITUDE GUARD, BY DESIGN. An earlier version of this spec
clamped each solved coefficient vector's L2 norm to the median coefficient norm
over the model's existing rows, by analogy with the ``_median_log_scaling``
clamp on donor scaling. That analogy is FALSE and the clamp is deleted. A
scaling and the population median of scalings are the same physical quantity,
so clamping one to the other is meaningful; a LoRA COEFFICIENT is not a
physical quantity at all -- its magnitude is fixed by the arbitrary scale of
the learned basis (``motion_lora_init_scale`` 0.01), so the coefficient needed
to express a given physical displacement bears no relation to the population's
coefficient distribution. Measured on the untrained basis, a realistic
one-frame displacement needs a coefficient norm of order 300 against a
population median of order 1, so the clamp shrank every flow-derived velocity
by roughly two orders of magnitude and the cell would have measured the clamp
rather than the flow.

No overshoot guard is needed in displacement space either, because the solve
CANNOT overshoot: the minimum-norm least-squares solution realizes
``c^T B = d @ pinv(B) @ B``, and ``pinv(B) @ B`` is the orthogonal projector
onto ``B``'s row space, so the realized displacement is an orthogonal
projection of the target and its norm is at most ``|d|``, with equality exactly
when the target lies in that row space. ``torch.linalg.pinv``'s built-in
``rcond`` cut-off is the only regularization, and it is kept.

TWO-TIER FAIL-CLOSED SEMANTICS. A missing or unreadable flow ASSET while
``packet_birth_flow_init`` is on is a CONFIGURATION error and raises
:class:`~depth_visibility.errors.ContractError`; it is never silently
tolerated, because a B1-F cell that quietly degrades to B1 would be
indistinguishable from its own control. Per-ROW invalidity (mask False,
non-finite flow, displaced pixel out of raster, flow-magnitude outlier,
non-finite solve) is normal and produces a counted fallback to zero motion.

HELD-OUT CAMERA SAFETY. Every flow read passes
:meth:`PacketFlowAssets.assert_not_holdout` first, and the camera-swapped
control removes the held-out ids from its candidate roster before choosing, so
neither arm can read the held-out camera's flow.

WIRING. ``MotionPriorCache`` lives on the ``Scene``, not on the trainer's
``OptimizationParams``, so the asset provider has to be handed to
``setup_packet_birth``. In ``main.py`` that is one line, after
``scene.motion_prior_cache`` is built and before the state is created::

    packet_birth_state = setup_packet_birth(
        opt,
        flow_assets=build_flow_assets(
            scene.motion_prior_cache,
            train_cameras=scene.getTrainCameras(),
            holdout_cameras=scene.getTestCameras(),
        ),
    )

Without that wiring a ``packet_birth_flow_init: true`` run raises
``ContractError`` at its first birth event rather than running as B1.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Tuple

import torch

from depth_visibility.errors import ContractError
from scene.packet_birth import backproject_camera_z, camera_forward_axis

#: ``packet_birth_flow_source`` domain. ``correct`` reads the training view's
#: own flow; ``camera_swapped`` is the WRONG-FLOW CONTROL.
FLOW_SOURCE_CORRECT = "correct"
FLOW_SOURCE_CAMERA_SWAPPED = "camera_swapped"
PACKET_FLOW_SOURCES = (FLOW_SOURCE_CORRECT, FLOW_SOURCE_CAMERA_SWAPPED)

#: N3V image-name convention ``cam<NN>_<FFFF>`` -- the same pattern
#: ``MotionPriorCache._find_panoptic_seg_mask`` matches
#: (utils/motion_prior_utils.py:346) and the spelling a real CVAT export uses
#: (tests/test_phase9_cvat_annotation.py:53, ``cam00_0170.png``).
CAMERA_NAME_PATTERN = re.compile(r"^cam(\d+)_(\d+)$")

#: A flow-validity mask pixel counts as valid above this. Strict nearest-pixel
#: set membership, the same threshold and rationale as
#: ``donor_dynamic_mask_eligibility`` (scene/packet_birth.py:324); the mask can
#: be fractional because ``resize_mask`` interpolates bilinearly.
FLOW_MASK_THRESHOLD = 0.5

#: ``timestamp == frame_index * frame_dt`` is asserted to this many frames.
FRAME_TIME_TOLERANCE_FRAMES = 0.25

#: Input-side outlier cut. A birth site whose flow magnitude exceeds this
#: quantile of the SAME VIEW's valid flow magnitudes fails closed. The quantile
#: is taken over the whole view (order 1e6 pixels on N3V), not over the handful
#: of birth sites, so the threshold is a stable property of the flow field.
FLOW_OUTLIER_QUANTILE = 0.99

_EPSILON = 1e-12


class _NamedView(object):
    """The only attribute ``MotionPriorCache.get_track_flow`` reads.

    ``get_track_flow`` (utils/motion_prior_utils.py:447-475) keys its cache on
    ``camera.image_name`` and resolves the asset path from it; nothing else on
    the camera is touched. The camera-swapped control therefore reads a
    different camera's flow by handing the provider this proxy, without
    fabricating a geometry-bearing camera object.
    """

    __slots__ = ("image_name",)

    def __init__(self, image_name: str) -> None:
        self.image_name = str(image_name)


def parse_camera_name(image_name: str) -> Tuple[int, int, str, str]:
    """``(camera_id, frame_index, camera_token, frame_token)`` or ContractError."""

    match = CAMERA_NAME_PATTERN.match(str(image_name))
    if match is None:
        raise ContractError(
            "packet-birth flow init needs the cam<NN>_<FFFF> image-name "
            "convention; got {!r}".format(image_name)
        )
    camera_token, frame_token = match.group(1), match.group(2)
    return int(camera_token), int(frame_token), camera_token, frame_token


def _camera_ids(cameras: Iterable[Any]) -> Tuple[int, ...]:
    """Sorted distinct camera ids from camera objects or bare image names.

    FAILS CLOSED on a non-empty roster that yields no usable name. The
    previous revision skipped unnamed entries silently, which turned a
    wiring mistake into an EMPTY roster rather than an error -- and an empty
    holdout roster makes :meth:`PacketFlowAssets.assert_not_holdout` inert,
    i.e. a safety guard that silently protects nothing. Verified in
    production: ``Scene.getTrainCameras()`` returns a
    :class:`utils.data_utils.CameraDataset` whose ``__getitem__`` yields
    ``(image, camera)`` TUPLES, so every entry was skipped. Callers must
    pass the camera objects themselves (the dataset's ``viewpoint_stack``).
    """

    ids = set()
    seen = 0
    for camera in cameras or ():
        seen += 1
        name = camera if isinstance(camera, str) else getattr(camera, "image_name", "")
        if not name:
            raise ContractError(
                "packet-birth flow init needs camera objects exposing "
                "image_name (or bare image-name strings); entry {} of the "
                "roster is a {} with no image_name. Scene.getTrainCameras() "
                "returns a CameraDataset that iterates as (image, camera) "
                "tuples -- pass its .viewpoint_stack instead.".format(
                    seen - 1, type(camera).__name__
                )
            )
        ids.add(parse_camera_name(name)[0])
    if seen and not ids:
        raise ContractError(
            "packet-birth flow init derived an EMPTY camera roster from {} "
            "entries; refusing rather than leaving the held-out guard "
            "inert".format(seen)
        )
    return tuple(sorted(ids))


class PacketFlowAssets(object):
    """Flow provider plus the camera rosters the frozen rules need.

    ``provider`` is any object exposing
    ``get_track_flow(camera, target_hw) -> (flow, mask)`` -- in production the
    trainer's :class:`~utils.motion_prior_utils.MotionPriorCache`.

    ``train_ids`` is the ascending list of TRAINING camera ids with every
    held-out id removed; it is the candidate roster of the camera-swapped
    control. ``holdout_ids`` is the guard set: no flow is ever read for a
    camera id in it.
    """

    def __init__(
        self,
        provider: Any,
        train_cameras: Iterable[Any] = (),
        holdout_cameras: Iterable[Any] = (),
    ) -> None:
        if provider is None or not hasattr(provider, "get_track_flow"):
            raise ContractError(
                "packet-birth flow init requires a provider exposing get_track_flow"
            )
        self.provider = provider
        self.holdout_ids = frozenset(_camera_ids(holdout_cameras))
        self.train_ids = tuple(
            value for value in _camera_ids(train_cameras) if value not in self.holdout_ids
        )

    def assert_not_holdout(self, image_name: str) -> int:
        """Hard held-out guard. Returns the camera id when it is safe to read."""

        camera_id = parse_camera_name(image_name)[0]
        if camera_id in self.holdout_ids:
            raise ContractError(
                "packet-birth flow init refused to read flow for held-out camera "
                "{!r}".format(image_name)
            )
        return camera_id

    def swapped_image_name(self, image_name: str) -> str:
        """FROZEN camera-swap rule for the WRONG-FLOW CONTROL (B1-X).

        Let ``cid`` be the source camera's integer id and ``frame`` its frame
        token. Let ``S`` be :attr:`train_ids` -- the ascending distinct
        TRAINING camera ids with every held-out id already removed. The
        substitute id is the smallest element of ``S`` STRICTLY GREATER than
        ``cid``, or, when no such element exists, the smallest element of ``S``
        (wrap-around). The substitute name is ``cam<sid>_<frame>`` with the
        SOURCE camera token's digit width and the SOURCE frame token verbatim,
        so the frame index is identical by construction and only the camera
        changes.

        The rule is deterministic, never returns the input camera (a roster
        that would force that is a ContractError), and can never select the
        held-out camera because held-out ids are not in ``S``.
        """

        camera_id, _, camera_token, frame_token = parse_camera_name(image_name)
        if not self.train_ids:
            raise ContractError(
                "the camera_swapped flow control needs a training-camera roster; "
                "pass train_cameras to build_flow_assets"
            )
        greater = [value for value in self.train_ids if value > camera_id]
        substitute = greater[0] if greater else self.train_ids[0]
        if substitute == camera_id:
            raise ContractError(
                "the camera_swapped flow control cannot substitute camera {!r}: "
                "the training roster has no other camera".format(image_name)
            )
        return "cam{:0{width}d}_{frame}".format(
            substitute, width=len(camera_token), frame=frame_token
        )

    def resolve_image_name(self, image_name: str, flow_source: str) -> str:
        """The image name whose flow this event reads, guarded against holdout."""

        if flow_source == FLOW_SOURCE_CORRECT:
            resolved = str(image_name)
        elif flow_source == FLOW_SOURCE_CAMERA_SWAPPED:
            self.assert_not_holdout(image_name)
            resolved = self.swapped_image_name(image_name)
        else:
            raise ContractError(
                "unknown packet_birth_flow_source {!r}; allowed: {}".format(
                    flow_source, ", ".join(PACKET_FLOW_SOURCES)
                )
            )
        self.assert_not_holdout(resolved)
        return resolved

    def get_flow(self, image_name: str, target_hw: Tuple[int, int]):
        """``(flow (2, H, W), mask (1, H, W))`` or ContractError.

        A missing or unreadable asset is a CONFIGURATION error, never a silent
        degradation to B1.
        """

        height, width = int(target_hw[0]), int(target_hw[1])
        flow, mask = self.provider.get_track_flow(_NamedView(image_name), (height, width))
        if flow is None:
            raise ContractError(
                "packet_birth_flow_init is on but no readable flow asset exists for "
                "{!r}".format(image_name)
            )
        flow = flow.detach()
        if flow.dim() != 3 or int(flow.shape[0]) != 2:
            raise ContractError(
                "packet-birth flow must be (2, H, W) after "
                "MotionPriorCache.resize_flow; got {}".format(tuple(flow.shape))
            )
        if tuple(flow.shape[-2:]) != (height, width):
            raise ContractError(
                "packet-birth flow resolution {} does not match the render raster "
                "{}".format(tuple(flow.shape[-2:]), (height, width))
            )
        if mask is None:
            # get_track_flow already applies this fallback (motion_prior_utils.py:469);
            # repeated here so a provider that skips it still fails closed per pixel
            # instead of treating every pixel as valid.
            mask = torch.isfinite(flow).all(dim=0, keepdim=True).float()
        mask = mask.detach()
        if tuple(mask.shape[-2:]) != (height, width) or int(mask.numel()) != height * width:
            # A multi-channel mask would make the flat index_select below read
            # the wrong channel's pixels, so it is refused rather than squeezed.
            raise ContractError(
                "packet-birth flow mask must be a single (H, W) plane matching "
                "the render raster {}; got {}".format((height, width), tuple(mask.shape))
            )
        return flow, mask


def build_flow_assets(
    provider: Any,
    train_cameras: Iterable[Any] = (),
    holdout_cameras: Iterable[Any] = (),
) -> PacketFlowAssets:
    """Trainer-side constructor; see the module docstring for the wiring line."""

    return PacketFlowAssets(
        provider, train_cameras=train_cameras, holdout_cameras=holdout_cameras
    )


def assert_frame_time_convention(image_name: str, timestamp: float, frame_dt: float) -> None:
    """Assert the trainer time unit rather than assuming 1/30.

    VERIFIED, not assumed: the N3V reader sets ``timestamp = frame['time']``
    with ``time = frame_index / 30`` (scene/dataset_readers.py:380 and the
    ``time_duration: [0.0, 1.6340]`` window of configs/n3v/ladder_b1_crb.yaml,
    which admits frames 0..49 at 1/30 spacing), and the frozen packet-birth
    state already treats ``opt.motion_track_dt`` (0.0333333333) as ONE FRAME in
    trainer time units when it forms ``sigma_t = t_sigma_frames * frame_dt``
    (scene/packet_birth.py:141, :179). This function checks that invariant
    against the data actually in hand -- the frame index parsed from the image
    name (the ``re.findall(r"\\d+", stem)[-1]`` convention of
    ``MotionPriorCache._frame_index_from_camera``,
    utils/motion_prior_utils.py:250) times ``frame_dt`` must reproduce the
    camera's timestamp -- so a mis-declared ``motion_track_dt`` or a reader
    whose timestamps are not ``frame_index * frame_dt`` stops the run instead
    of silently scaling every initialized velocity.
    """

    frame_dt = float(frame_dt)
    if not (frame_dt > 0.0):
        raise ContractError("packet-birth flow init requires a positive frame dt")
    frame_index = parse_camera_name(image_name)[1]
    if frame_index <= 0:
        # Frame 0 carries no information about the spacing.
        return
    expected = float(frame_index) * frame_dt
    tolerance = FRAME_TIME_TOLERANCE_FRAMES * frame_dt
    if abs(float(timestamp) - expected) > tolerance:
        raise ContractError(
            "packet-birth flow init expects timestamp == frame_index * "
            "motion_track_dt; camera {!r} has timestamp {!r} but frame {} at dt "
            "{!r} implies {!r}".format(
                image_name, float(timestamp), frame_index, frame_dt, expected
            )
        )


def _unnormalized_direction_grid(rays_d_grid: torch.Tensor, forward: torch.Tensor) -> torch.Tensor:
    """The camera's ray-direction field rescaled to view-space z == 1.

    ``Camera.get_rays`` (scene/cameras.py:110-117) builds
    ``pts_view = ((i - cx)/fl_x, (j - cy)/fl_y, 1)`` at pixel CENTRES, rotates
    it into world coordinates and NORMALIZES. The pre-normalization field is
    AFFINE in the pixel coordinates ``(i, j)``; the normalization is what makes
    it non-affine. Dividing the returned unit direction by its component along
    the view axis undoes exactly that -- ``dot(R @ pts_view, R @ e_z) == 1`` for
    the orthonormal camera-to-world rotation -- so the result is the affine
    field again and a bilinear read of it at a FRACTIONAL pixel is exact rather
    than approximate. It is also precisely the vector
    :func:`~scene.packet_birth.backproject_camera_z` reconstructs internally
    when it divides by ``cosine``, so no ray machinery is reimplemented here.
    """

    cosine = (rays_d_grid * forward.reshape(1, 1, 3).to(rays_d_grid.dtype)).sum(
        dim=-1, keepdim=True
    )
    return rays_d_grid / cosine.clamp(min=1e-6)


def _bilinear_directions(
    unnormalized: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Exact bilinear read of the affine direction field at grid index (x, y)."""

    height, width = int(unnormalized.shape[0]), int(unnormalized.shape[1])
    flat = unnormalized.reshape(-1, 3)
    x0 = torch.floor(x).long().clamp(0, width - 1)
    y0 = torch.floor(y).long().clamp(0, height - 1)
    x1 = (x0 + 1).clamp(0, width - 1)
    y1 = (y0 + 1).clamp(0, height - 1)
    wx = (x - x0.to(x.dtype)).clamp(0.0, 1.0).reshape(-1, 1)
    wy = (y - y0.to(y.dtype)).clamp(0.0, 1.0).reshape(-1, 1)
    d00 = flat.index_select(0, y0 * width + x0)
    d10 = flat.index_select(0, y0 * width + x1)
    d01 = flat.index_select(0, y1 * width + x0)
    d11 = flat.index_select(0, y1 * width + x1)
    top = d00 * (1.0 - wx) + d10 * wx
    bottom = d01 * (1.0 - wx) + d11 * wx
    return top * (1.0 - wy) + bottom * wy


def flow_world_velocity(
    viewpoint_camera: Any,
    pixels: torch.Tensor,
    points: torch.Tensor,
    flow: torch.Tensor,
    mask: torch.Tensor,
    frame_dt: float,
):
    """World velocity for each birth site under the constant-depth rule.

    ``pixels`` are the FLAT birth-site indices (``row * W + col``, the layout
    ``Camera.get_rays`` and the residual map share) AFTER the depth-validity
    filter and the donor truncation, so they align one-to-one with ``points``
    -- the sites' already-backprojected world positions ``X0``.

    AXIS ORDER AND UNITS, VERIFIED not assumed: ``get_track_flow`` returns the
    flow through ``normalize_flow_tensor`` (utils/motion_prior_utils.py:62-73),
    which forces channel-first ``(2, H, W)``, and then through ``resize_flow``
    (:76-87), which rescales channel 0 by ``new_W / old_W`` and channel 1 by
    ``new_H / old_H``. Channel 0 is therefore the HORIZONTAL (column, ``u``)
    component and channel 1 the VERTICAL (row, ``v``) component, both in PIXELS
    at the requested ``target_hw`` -- which is the render raster, so no further
    resize or rescale is applied here.

    ``X0`` already encodes the rendered camera-z: because the unnormalized ray
    has view-space z exactly 1, ``dot(X0 - o, forward)`` recovers the very
    camera-z ``backproject_pixels`` used, with no second read of ``depth`` /
    ``alpha`` and no chance of the two disagreeing. ``X1`` is that SAME camera-z
    along the ray through the displaced pixel -- the disclosed constant-depth
    approximation.

    Returns ``(velocity (K, 3), valid (K,), counters)``. Invalid rows carry a
    zero velocity. The counters attribute each failure to exactly one cause in
    the frozen precedence order mask -> non-finite -> out-of-bounds -> outlier.
    """

    device = points.device
    dtype = points.dtype
    count = int(pixels.numel())
    counters = {"mask": 0, "nonfinite": 0, "bounds": 0, "outlier": 0}
    if count == 0:
        return (
            torch.zeros((0, 3), dtype=dtype, device=device),
            torch.zeros((0,), dtype=torch.bool, device=device),
            counters,
        )
    height = int(viewpoint_camera.image_height)
    width = int(viewpoint_camera.image_width)
    if tuple(flow.shape[-2:]) != (height, width):
        raise ContractError("packet-birth flow does not match the camera raster")

    index = pixels.reshape(-1).to(device)
    column = (index % width).to(dtype)
    row = torch.div(index, width, rounding_mode="floor").to(dtype)

    flow_flat = flow.to(device=device, dtype=dtype).reshape(2, -1)
    u = flow_flat[0].index_select(0, index)
    v = flow_flat[1].index_select(0, index)
    mask_flat = mask.to(device=device, dtype=dtype).reshape(-1)
    mask_ok = mask_flat.index_select(0, index) > FLOW_MASK_THRESHOLD
    finite = torch.isfinite(u) & torch.isfinite(v)

    # p' in the get_rays GRID-INDEX frame: grid index k addresses the ray of
    # pixel centre k + 0.5, so adding the pixel-unit flow to the integer index
    # is the same displacement as adding it to the continuous image coordinate.
    # "Outside the raster" is the interpolation domain [0, W-1] x [0, H-1],
    # which is the raster minus the outer half pixel on each side -- the frozen
    # bounds rule, stated here so it is not silently re-derived.
    x_displaced = column + u
    y_displaced = row + v
    in_bounds = (
        (x_displaced >= 0.0)
        & (x_displaced <= float(width - 1))
        & (y_displaced >= 0.0)
        & (y_displaced <= float(height - 1))
    )

    # Input-side outlier cut, in PIXELS, against the SAME view's own flow
    # field: the threshold is the FLOW_OUTLIER_QUANTILE quantile of the
    # magnitudes over every pixel this view's validity mask admits with finite
    # flow. Taking it over the view rather than over the handful of birth sites
    # makes it a property of the SEA-RAFT field, not of the sampling.
    site_magnitude = torch.sqrt(u * u + v * v)
    view_mask = (mask_flat > FLOW_MASK_THRESHOLD) & torch.isfinite(flow_flat).all(dim=0)
    view_magnitude = torch.sqrt(
        flow_flat[0] * flow_flat[0] + flow_flat[1] * flow_flat[1]
    )[view_mask]
    if int(view_magnitude.numel()) > 0:
        threshold = torch.quantile(view_magnitude.to(torch.float32), FLOW_OUTLIER_QUANTILE)
        not_outlier = site_magnitude <= threshold.to(site_magnitude.dtype)
    else:
        # No valid pixel anywhere in the view: every site already fails the mask
        # test below, so there is nothing left for the cut to remove.
        not_outlier = torch.ones_like(mask_ok)

    counters["mask"] = int((~mask_ok).sum().item())
    counters["nonfinite"] = int((mask_ok & ~finite).sum().item())
    counters["bounds"] = int((mask_ok & finite & ~in_bounds).sum().item())
    counters["outlier"] = int(
        (mask_ok & finite & in_bounds & ~not_outlier).sum().item()
    )
    valid = mask_ok & finite & in_bounds & not_outlier

    rays_o, rays_d_grid = viewpoint_camera.get_rays()
    rays_o = rays_o.to(device=device, dtype=dtype)
    rays_d_grid = rays_d_grid.to(device=device, dtype=dtype).reshape(height, width, 3)
    forward = camera_forward_axis(viewpoint_camera).to(device=device, dtype=dtype)
    unnormalized = _unnormalized_direction_grid(rays_d_grid, forward)

    origin = rays_o.reshape(1, 3)
    camera_z = ((points - origin) * forward.reshape(1, 3)).sum(dim=1)

    safe_x = torch.where(valid, x_displaced, column)
    safe_y = torch.where(valid, y_displaced, row)
    displaced_unnormalized = _bilinear_directions(unnormalized, safe_x, safe_y)
    displaced_dir = displaced_unnormalized / displaced_unnormalized.norm(
        dim=1, keepdim=True
    ).clamp(min=_EPSILON)
    displaced_points = backproject_camera_z(rays_o, displaced_dir, forward, camera_z)

    velocity = (displaced_points - points) / float(frame_dt)
    velocity = torch.where(
        valid.reshape(-1, 1) & torch.isfinite(velocity).all(dim=1, keepdim=True),
        velocity,
        torch.zeros_like(velocity),
    )
    valid = valid & torch.isfinite(velocity).all(dim=1)
    return velocity, valid, counters


def lora_probe_basis(gaussians: Any, dt_probe: float) -> torch.Tensor:
    """``(rank, 3)`` centered LoRA basis at the probe offset, ONCE per event.

    ``GaussianModel.get_lora_motion_offset`` (scene/gaussian_model.py:752-769)
    computes ``offset(t) = einsum('nr,nrd->nd', coeff, centered_basis)`` with
    ``centered_basis = basis(t - t_n) - basis(0)``. Every row born by one event
    shares the same temporal mean ``t_n`` (the view timestamp) and the same
    probe offset, and ``_sample_lora_basis`` depends on nothing but that
    relative offset, so the centered basis is IDENTICAL for every row of the
    event and is computed once here.
    """

    if str(getattr(gaussians, "motion_model", "")) != "lora":
        raise ContractError(
            "packet_birth_flow_init is defined for the LoRA motion model; the "
            "model reports motion_model={!r}".format(getattr(gaussians, "motion_model", None))
        )
    sampler = getattr(gaussians, "_sample_lora_basis", None)
    if sampler is None:
        raise ContractError(
            "packet_birth_flow_init requires a model exposing _sample_lora_basis"
        )
    basis = getattr(gaussians, "_motion_lora_basis", None)
    if basis is None or int(basis.numel()) == 0:
        raise ContractError("packet_birth_flow_init requires a materialized LoRA basis")
    relative = basis.new_full((1, 1), float(dt_probe))
    at_probe = sampler(relative)
    at_center = sampler(torch.zeros_like(relative))
    if at_probe is None or at_center is None:
        raise ContractError("the LoRA basis sampler returned no basis")
    centered = (at_probe - at_center).detach().reshape(-1, 3)
    if not bool(torch.isfinite(centered).all()):
        raise ContractError("the centered LoRA probe basis is not finite")
    if float(centered.abs().max().item()) <= 0.0:
        # A degenerate probe cannot represent ANY velocity, so every row would
        # silently keep zero motion -- exactly the vacuous outcome this cell
        # exists to rule out. Configuration error, not a per-row fallback.
        raise ContractError(
            "the centered LoRA probe basis is identically zero at dt={!r}; "
            "flow-derived velocity cannot be represented".format(float(dt_probe))
        )
    return centered


def solve_lora_coefficients(basis: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Least-squares ``min_c || c^T B - displacement ||^2``, ONE solve per event.

    ``B`` is ``(rank, 3)`` and the model's forward map for a single row is
    ``c^T B``, so the linear system is ``B^T c = displacement`` with ``B^T``
    of shape ``(3, rank)``. The minimum-norm least-squares solution is
    ``c = pinv(B^T) @ displacement = pinv(B).T @ displacement``, hence the
    batched form ``coeff = displacement @ pinv(B)``.

    An explicit pseudo-inverse is used rather than ``torch.linalg.lstsq``
    because the system is UNDER-determined (rank 8 unknowns, 3 equations) and
    lstsq's default ``gels`` driver -- the only one available on CUDA -- is
    documented for full-rank OVER-determined systems. ``pinv`` gives the
    minimum-norm solution on both devices and needs one factorization per
    event, which is then applied to every row by a single matmul. Its ``rcond``
    cut-off is the only regularization in the whole path and is left at the
    default.

    NO OVERSHOOT IS POSSIBLE, so no output-side magnitude guard exists. The
    realized displacement is ``c^T B = d @ (pinv(B) @ B)`` and ``pinv(B) @ B``
    is the orthogonal projector onto ``B``'s row space, hence
    ``|realized| <= |d|`` always, with equality exactly when ``d`` lies in that
    row space. When the projector is not the identity -- a rank-deficient probe
    basis -- the shortfall is real and is reported per event as
    ``flow_realized_ratio_mean`` rather than hidden.
    """

    if basis.dim() != 2 or int(basis.shape[1]) != 3:
        raise ContractError("the LoRA probe basis must be (rank, 3)")
    pseudo_inverse = torch.linalg.pinv(basis.to(displacement.dtype))
    return displacement @ pseudo_inverse


def realized_displacement(coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """``(K, 3)`` displacement the model will actually produce at the probe.

    Uses the model's own map, ``einsum('nr,nrd->nd', coeff, centered_basis)``
    (scene/gaussian_model.py:769), with the event's single centered basis
    broadcast over the rows -- so this is the realized displacement, not a
    paraphrase of it.
    """

    count = int(coefficients.shape[0])
    rank = int(basis.shape[0])
    expanded = basis.reshape(1, rank, 3).expand(count, -1, -1)
    return torch.einsum("nr,nrd->nd", coefficients, expanded)


def flow_motion_lora_coefficients(
    state: Any,
    gaussians: Any,
    viewpoint_camera: Any,
    pixels: torch.Tensor,
    points: torch.Tensor,
):
    """One event's flow-derived ``_motion_lora_coeff`` rows plus its funnel.

    Returns ``(coefficients (K, rank), record_fields)``. Rows that failed
    closed carry exactly zero, so the transaction writes the unchanged
    zero-motion initialization for them.
    """

    cfg = state.cfg
    flow_source = str(cfg.flow_source)
    fields = {
        "flow_init": True,
        "flow_source": flow_source,
        "flow_sites_total": int(pixels.numel()),
        "flow_sites_valid": 0,
        "flow_failed_mask": 0,
        "flow_failed_bounds": 0,
        "flow_failed_nonfinite": 0,
        "flow_failed_outlier": 0,
        "flow_mean_speed": 0.0,
        "flow_coeff_norm_mean": 0.0,
        "flow_coeff_norm_max": 0.0,
        "flow_realized_ratio_mean": 0.0,
    }
    assets = getattr(state, "flow_assets", None)
    if assets is None:
        raise ContractError(
            "packet_birth_flow_init is on but no flow asset provider was attached; "
            "pass flow_assets=build_flow_assets(...) to setup_packet_birth"
        )
    image_name = str(getattr(viewpoint_camera, "image_name", ""))
    timestamp = float(getattr(viewpoint_camera, "timestamp", 0.0))
    frame_dt = float(state.frame_dt)
    assert_frame_time_convention(image_name, timestamp, frame_dt)

    height = int(viewpoint_camera.image_height)
    width = int(viewpoint_camera.image_width)
    resolved = assets.resolve_image_name(image_name, flow_source)
    flow, mask = assets.get_flow(resolved, (height, width))

    velocity, valid, counters = flow_world_velocity(
        viewpoint_camera, pixels, points, flow, mask, frame_dt
    )
    fields["flow_failed_mask"] = counters["mask"]
    fields["flow_failed_nonfinite"] = counters["nonfinite"]
    fields["flow_failed_bounds"] = counters["bounds"]
    fields["flow_failed_outlier"] = counters["outlier"]

    # dt_p = one frame: the probe offset at which the flow measurement itself
    # is defined, so the target displacement is exactly the measured one.
    dt_probe = frame_dt
    basis = lora_probe_basis(gaussians, dt_probe)
    rank = int(basis.shape[0])
    target = velocity * dt_probe
    coefficients = solve_lora_coefficients(basis, target)
    realized = realized_displacement(coefficients, basis)

    # Post-solve fail-closed: a non-finite coefficient or realized displacement
    # would poison the transaction's finiteness contract, so the row keeps its
    # zero-motion initialization and joins the non-finite tally.
    solvable = torch.isfinite(coefficients).all(dim=1) & torch.isfinite(realized).all(dim=1)
    fields["flow_failed_nonfinite"] += int((valid & ~solvable).sum().item())
    valid = valid & solvable
    coefficients = torch.where(
        valid.reshape(-1, 1), coefficients, torch.zeros_like(coefficients)
    )
    if int(coefficients.shape[1]) != rank:
        raise ContractError("solved LoRA coefficients have the wrong rank")

    fields["flow_sites_valid"] = int(valid.sum().item())
    if int(valid.sum().item()):
        fields["flow_mean_speed"] = float(velocity.norm(dim=1)[valid].mean().item())
        norms = coefficients.norm(dim=1)[valid]
        fields["flow_coeff_norm_mean"] = float(norms.mean().item())
        fields["flow_coeff_norm_max"] = float(norms.max().item())
        # |realized| / |target| over the valid rows that asked for a nonzero
        # displacement. Far below 1 means the probe basis cannot REPRESENT the
        # requested velocities, so the cell would be measuring the basis rather
        # than the flow; the ratio is reported so that is visible per event
        # instead of being absorbed silently.
        target_norm = target.norm(dim=1)[valid]
        realized_norm = realized.norm(dim=1)[valid]
        requested = target_norm > 0
        if bool(requested.any()):
            fields["flow_realized_ratio_mean"] = float(
                (realized_norm[requested] / target_norm[requested]).mean().item()
            )
    return coefficients, fields


def idle_record_fields(cfg: Any) -> dict:
    """Funnel fields for an event that runs no flow path (B1, or an early exit).

    Present on EVERY birth record so that a run whose flow never applied is
    detectable from the record alone rather than by its absence.
    """

    return {
        "flow_init": bool(getattr(cfg, "flow_init", False)),
        "flow_source": str(getattr(cfg, "flow_source", FLOW_SOURCE_CORRECT)),
        "flow_sites_total": 0,
        "flow_sites_valid": 0,
        "flow_failed_mask": 0,
        "flow_failed_bounds": 0,
        "flow_failed_nonfinite": 0,
        "flow_failed_outlier": 0,
        "flow_mean_speed": 0.0,
        "flow_coeff_norm_mean": 0.0,
        "flow_coeff_norm_max": 0.0,
        "flow_realized_ratio_mean": 0.0,
    }


__all__ = [
    "CAMERA_NAME_PATTERN",
    "FLOW_MASK_THRESHOLD",
    "FLOW_OUTLIER_QUANTILE",
    "FLOW_SOURCE_CAMERA_SWAPPED",
    "FLOW_SOURCE_CORRECT",
    "FRAME_TIME_TOLERANCE_FRAMES",
    "PACKET_FLOW_SOURCES",
    "PacketFlowAssets",
    "assert_frame_time_convention",
    "build_flow_assets",
    "flow_motion_lora_coefficients",
    "flow_world_velocity",
    "idle_record_fields",
    "lora_probe_basis",
    "parse_camera_name",
    "realized_displacement",
    "solve_lora_coefficients",
]
