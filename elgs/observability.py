"""Counterfactual observability q (spec §3).

q := clip( sum_s omega_s * 1_frustum(s) * T_{-(f;j)}(s) * kappa_res,
           0, 1 )

- sigma points s: DETERMINISTIC NONNEGATIVE weights summing to one
  (preregistered grid scheme; no negative-weight unscented transform);
- T_{-(f;j)}: family-present, query-source-excluded transmittance at
  the sigma point (RenderProbe — plan §3 decision: front-set
  compositing);
- kappa_res in [0,1] by construction (clipped area ratio);
- q_tilde = q * d_u in [0,1] always.

q is a ROUND SNAPSHOT: the round-r objective is written against
q^{(r)}; candidates are scored under the snapshot; refresh happens
only at round boundaries (the clone-invariance audit measures
refresh-boundary drift — invariance across refreshes is NOT claimed).
The snapshot store below freezes after the round's q values are
computed; reads of missing keys and writes after freeze fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from depth_visibility.errors import ContractError

from .probe import RenderProbe

_WEIGHT_SUM_TOL = 1e-9


@dataclass(frozen=True)
class SigmaPoint:
    """One deterministic sigma point: weight, world point, pixel, depth."""

    weight: float
    point: tuple[float, float, float]
    pixel: tuple[float, float]
    depth: float

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ContractError("sigma-point weights must be nonnegative")
        if self.depth <= 0.0:
            raise ContractError("sigma-point depth must be positive")


def validate_sigma_points(points: Sequence[SigmaPoint]) -> None:
    if not points:
        raise ContractError("at least one sigma point is required")
    total = sum(p.weight for p in points)
    if abs(total - 1.0) > _WEIGHT_SUM_TOL:
        raise ContractError(f"sigma-point weights sum to {total}, expected 1")


def compute_q(
    sigma_points: Sequence[SigmaPoint],
    probe: RenderProbe,
    camera_id: int,
    frame: float,
    *,
    exclude_track: int | None,
    present_family: int | None,
    kappa_res: float,
) -> float:
    """The §3 q for one (bridge sigma set, track, camera, frame)."""
    validate_sigma_points(sigma_points)
    if not (0.0 <= kappa_res <= 1.0):
        raise ContractError(f"kappa_res={kappa_res} outside [0,1]")
    total = 0.0
    for point in sigma_points:
        if not probe.in_frustum(camera_id, frame, point.point):
            continue
        t_value = probe.transmittance(
            camera_id,
            frame,
            point.pixel,
            point.depth,
            exclude_track=exclude_track,
            present_family=present_family,
        )
        total += point.weight * t_value * kappa_res
    return min(max(total, 0.0), 1.0)


def q_tilde(q: float, d_u: float) -> float:
    """q_tilde = q * d_u, in [0,1] always."""
    if not (0.0 <= q <= 1.0):
        raise ContractError(f"q={q} outside [0,1]")
    if not (0.0 <= d_u <= 1.0):
        raise ContractError(f"d_u={d_u} outside [0,1]")
    return q * d_u


class QSnapshot:
    """Round-snapshot store for q_tilde values (spec §3 + rev-2 item 3).

    Keys are (bridge_id, track_id, camera_id, frame). The store is
    written during the round-boundary refresh, then FROZEN; candidate
    evaluation reads only. Writing after freeze, double-writing a key,
    or reading a missing key raises — a silent q refresh mid-round is
    exactly the renderer self-exoneration path the snapshot exists to
    close.
    """

    def __init__(self, round_index: int) -> None:
        if round_index < 0:
            raise ContractError("round index must be non-negative")
        self.round_index = round_index
        self._values: dict[tuple[int, int, int, float], float] = {}
        self._frozen = False

    def put(self, bridge_id: int, track_id: int, camera_id: int, frame: float, value: float) -> None:
        if self._frozen:
            raise ContractError("q snapshot is frozen for this round")
        key = (bridge_id, track_id, camera_id, frame)
        if key in self._values:
            raise ContractError(f"q snapshot key {key} written twice")
        if not (0.0 <= value <= 1.0):
            raise ContractError(f"q_tilde={value} outside [0,1]")
        self._values[key] = value

    def freeze(self) -> None:
        if self._frozen:
            raise ContractError("q snapshot already frozen")
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    def get(self, bridge_id: int, track_id: int, camera_id: int, frame: float) -> float:
        key = (bridge_id, track_id, camera_id, frame)
        if key not in self._values:
            raise ContractError(f"q snapshot has no value for {key}")
        return self._values[key]

    def as_bridge_maps(self) -> dict:
        """{bridge_id: {(j, c, t): q_tilde}} for elgs.energy.EvidenceBatch."""
        out: dict[int, dict] = {}
        for (b, j, c, t), value in self._values.items():
            out.setdefault(b, {})[(j, c, t)] = value
        return out


__all__ = [
    "QSnapshot",
    "SigmaPoint",
    "compute_q",
    "q_tilde",
    "validate_sigma_points",
]
