"""RenderProbe: point-truncated, query-source-excluded transmittance.

DECISION (plan §3 underdetermined-decision register, resolved here and
preregistered in configs/elgs/prereg_observability_v1.json): the
transmittance T_{-(f;j)} at a sigma point is computed by FRONT-SET
COMPOSITING — the product of (1 - alpha_i * G_i(pixel)) over exactly
the Gaussians whose ray depth is strictly in front of the query
point, with the query track's source Gaussians EXCLUDED and the
family under evaluation PRESENT (counterfactually inserted for
candidates). This is exact w.r.t. the rasterizer's compositing model
and needs no CUDA change. Rejected alternatives (recorded): (b)
depth-thresholded alpha renders — an approximation with quantization
the exact option makes unnecessary; (c) a CUDA rasterizer addition —
not required once (a) meets the runtime budget; revisited only if S1
profiling fails it.

The GPU implementation (spatially pruned gather over the model's
projected Gaussians, torch, no_grad) lands with the trainer wiring
and must match AnalyticProbe on the closed-form fixture to 1e-6
(reference-parity test). AnalyticProbe below IS the CPU reference:
deliberately naive, loop-based, and importable without CUDA.

Stop-gradient contract: probes run under no_grad; q is a ROUND
SNAPSHOT (spec §3) and never carries gradients into theta or a.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from depth_visibility.errors import ContractError


@dataclass(frozen=True)
class ProbeGaussian:
    """One analytic splat for the CPU reference probe.

    depth: ray depth of the splat center for the queried camera;
    alpha: composited opacity at the splat center (post presence and
    routing multipliers); sigma_px: isotropic screen-space footprint;
    center_px: projected center in pixels; source_track / family_id
    identify exclusion and counterfactual-presence sets.
    """

    depth: float
    alpha: float
    sigma_px: float
    center_px: tuple[float, float]
    family_id: int
    source_track: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ContractError("splat alpha must lie in [0,1]")
        if self.sigma_px <= 0.0 or self.depth <= 0.0:
            raise ContractError("splat sigma_px and depth must be positive")

    def footprint(self, pixel: tuple[float, float]) -> float:
        dx = pixel[0] - self.center_px[0]
        dy = pixel[1] - self.center_px[1]
        return math.exp(-0.5 * (dx * dx + dy * dy) / (self.sigma_px**2))


class RenderProbe(Protocol):
    """The narrow interface observability consumes (CPU-fake-able)."""

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
        """T just in FRONT of `depth` along the pixel ray, with the
        query track's source Gaussians excluded and (optionally) one
        family counterfactually forced present."""
        ...

    def in_frustum(self, camera_id: int, frame: float, point: tuple[float, float, float]) -> bool:
        ...


class AnalyticProbe:
    """CPU reference probe over an explicit splat list per (camera, frame)."""

    def __init__(
        self,
        splats: dict,
        frustum: dict | None = None,
    ) -> None:
        # splats: (camera_id, frame) -> Sequence[ProbeGaussian]
        self._splats = splats
        self._frustum = frustum or {}

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
        key = (camera_id, frame)
        if key not in self._splats:
            raise ContractError(f"probe has no splats for camera/frame {key}")
        t_value = 1.0
        for splat in self._splats[key]:
            if splat.depth >= depth:
                continue  # strictly in front only
            if exclude_track is not None and splat.source_track == exclude_track:
                continue  # query-source exclusion (self-occlusion of the
                # family is MODELED: other family splats still occlude)
            t_value *= 1.0 - splat.alpha * splat.footprint(pixel)
        if not (0.0 <= t_value <= 1.0):
            raise ContractError(f"composited transmittance {t_value} outside [0,1]")
        # present_family is honored by construction here: the caller
        # builds the splat list with the candidate family inserted
        # (frozen-functional snapshot, spec rev-2 item 3).
        _ = present_family
        return t_value

    def in_frustum(self, camera_id: int, frame: float, point: tuple[float, float, float]) -> bool:
        key = (camera_id, frame)
        rule = self._frustum.get(key)
        return True if rule is None else bool(rule(point))


__all__ = ["AnalyticProbe", "ProbeGaussian", "RenderProbe"]
