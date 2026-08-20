"""Presence function, plateaus, edge bands, winner lookup.

Substrate semantics (lgs-method via the spec's §1 delegation):
pi_j(t) = S((t - b_j)/w) * S((d_j - t)/w) with S the clamped cubic
smoothstep (0 below 0, 3u^2 - 2u^3 on [0,1], 1 above 1) — EXACT zero
outside [b_j, d_j], exact plateau 1 on [b_j + w, d_j - w], latched (no
mid-episode dip expressible). Episodes of one family are disjoint with
gap >= floor_gap > 2w, so at most one episode is active at any t and
the winner is a unique interval lookup.

Spec §1: z_f(t) = 1 iff t lies in a plateau [b_k + w, d_k - w];
edge bands X_f = {t : |t - nearest edge| < w, strict}.
"""

from __future__ import annotations

import torch

from depth_visibility.errors import ContractError

from .intervals import IntervalRealization


def smoothstep(u: torch.Tensor) -> torch.Tensor:
    """Clamped cubic smoothstep: exact 0 below 0 and exact 1 above 1."""
    clamped = torch.clamp(u, 0.0, 1.0)
    return clamped * clamped * (3.0 - 2.0 * clamped)


def episode_presence(t: torch.Tensor, b: torch.Tensor, d: torch.Tensor, w: float) -> torch.Tensor:
    """pi for each episode at time(s) t; broadcasts t against (K,) b/d."""
    if w <= 0:
        raise ContractError("presence edge half-width w must be positive")
    return smoothstep((t - b) / w) * smoothstep((d - t) / w)


def winner_index(realization: IntervalRealization, t: float) -> int | None:
    """The unique episode k (0-based) with t in [b_k, d_k], else None.

    Uniqueness holds by construction (disjoint episodes, gap floor
    > 2w); this lookup asserts it rather than trusting it.
    """
    inside = [
        k
        for k in range(realization.b.numel())
        if float(realization.b[k]) <= t <= float(realization.d[k])
    ]
    if len(inside) > 1:
        raise ContractError(
            f"winner lookup found {len(inside)} active episodes at t={t}: "
            "episode intervals are not disjoint"
        )
    return inside[0] if inside else None


def family_presence(realization: IntervalRealization, t: float, w: float) -> torch.Tensor:
    """pi_winner(t) as a differentiable scalar (exact 0 with no winner)."""
    k = winner_index(realization, t)
    if k is None:
        return realization.b.new_zeros(())
    t_tensor = realization.b.new_tensor(t)
    return episode_presence(t_tensor, realization.b[k], realization.d[k], w).reshape(())


def plateau_z(realization: IntervalRealization, t: float, w: float) -> bool:
    """z_f(t) = 1 iff t in a plateau [b_k + w, d_k - w] (spec §1)."""
    for k in range(realization.b.numel()):
        if float(realization.b[k]) + w <= t <= float(realization.d[k]) - w:
            return True
    return False


def in_edge_band(realization: IntervalRealization, t: float, w: float) -> bool:
    """t in X_f iff |t - nearest edge| < w, STRICT (spec §1)."""
    edges = torch.cat([realization.b, realization.d])
    nearest = float((edges - t).abs().min())
    return nearest < w


def local_presence_multipliers(
    presence: torch.Tensor, marginal: torch.Tensor, gated: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row temporal multipliers under LOCALIZED presence.

    gated rows (their family carries a K >= 2 episodic program) get the
    episodic presence on BOTH routed branches, so presence 0 is a total
    gate: no contribution through the dynamic branch or the separately
    rendered static twin. Every other row keeps the substrate exactly:
    the ordinary temporal marginal on the dynamic branch and an
    unmodulated static twin.

    presence/marginal are (N, 1); gated is (N, 1) bool. Returns
    (dynamic_multiplier, static_multiplier), both (N, 1).
    """
    if presence.shape != marginal.shape or presence.shape != gated.shape:
        raise ContractError(
            "local presence: presence, marginal and gated must share one "
            f"(N, 1) shape; got {tuple(presence.shape)}, "
            f"{tuple(marginal.shape)}, {tuple(gated.shape)}"
        )
    dynamic = torch.where(gated, presence, marginal)
    static = torch.where(gated, presence, torch.ones_like(presence))
    return dynamic, static


def z_series(realization: IntervalRealization, frames: list[float], w: float) -> list[bool]:
    return [plateau_z(realization, t, w) for t in frames]


def edge_band_series(realization: IntervalRealization, frames: list[float], w: float) -> list[bool]:
    return [in_edge_band(realization, t, w) for t in frames]


__all__ = [
    "edge_band_series",
    "episode_presence",
    "family_presence",
    "in_edge_band",
    "local_presence_multipliers",
    "plateau_z",
    "smoothstep",
    "winner_index",
    "z_series",
]
