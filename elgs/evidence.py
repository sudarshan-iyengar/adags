"""Evidence objects: reports, heads, L1/L0 (spec §2).

Observation space Y = {miss} ⊔ ([v_min, v_max] × D_img), where the
positional coordinate is the RAW report position in the bounded image
domain D_img. Heads (all normalized densities over Y w.r.t. one base
measure; fitted only on calibration scenes, frozen — category-2):

  p_vis(y|b)  = 1_miss·pi_m^v + (1-1_miss)(1-pi_m^v)·g_v(v)·g_pos(y_pos|b)
  p_cens(y)   = 1_miss·pi_m^c + (1-1_miss)(1-pi_m^c)·h_c(v)·(1/|D_img|)
  p_out(y)    analogous to p_cens with its own miss mass and value head

  L1(y|b,q_tilde,r) = r[q_tilde·p_vis + (1-q_tilde)·p_cens] + (1-r)·p_out
  L0(y|r)           = r·p_cens + (1-r)·p_out

Censoring equality: L1 at q_tilde = 0 is IDENTICALLY L0 — algebraic,
not approximate; PROP 1 depends on it.

g_pos is a truncated Gaussian over D_img centered at the
bridge-projected point, normalized over D_img BY CONSTRUCTION (exact
rectangle normalization via erf — no Jacobian issue). The scalar value
heads g_v / h_c / h_o are injected callables whose families and
parameters are category-2 preregistration items.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from depth_visibility.errors import ContractError

ValueHead = Callable[[float], float]


@dataclass(frozen=True)
class Report:
    """One raw report y: a miss token or (v, position) in D_img."""

    is_miss: bool
    v: float = 0.0
    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class EvidenceParams:
    """Frozen head parameters + the §2 floors/caps."""

    pi_miss_vis: float
    pi_miss_cens: float
    pi_miss_out: float
    d_img_x0: float
    d_img_x1: float
    d_img_y0: float
    d_img_y1: float
    g_pos_sigma: float
    v_min: float
    v_max: float
    r_min: float
    h_floor: float
    pi_floor: float
    g_cap: float
    pos_cap: float

    def __post_init__(self) -> None:
        if self.d_img_x1 <= self.d_img_x0 or self.d_img_y1 <= self.d_img_y0:
            raise ContractError("D_img must have positive area")
        if not (0.0 < self.pi_floor < 0.5):
            raise ContractError("pi_floor must lie in (0, 0.5)")
        for name in ("pi_miss_vis", "pi_miss_cens", "pi_miss_out"):
            value = getattr(self, name)
            if not (self.pi_floor <= value <= 1.0 - self.pi_floor):
                raise ContractError(
                    f"{name}={value} outside [pi_floor, 1-pi_floor]"
                )
        if self.h_floor <= 0.0:
            raise ContractError("h_floor must be strictly positive")
        if self.g_pos_sigma <= 0.0:
            raise ContractError("g_pos_sigma must be positive")
        if self.v_max <= self.v_min:
            raise ContractError("value range [v_min, v_max] must be non-empty")
        if not (0.0 < self.r_min <= 1.0):
            raise ContractError("r_min must lie in (0, 1]")
        if self.g_cap <= 0.0 or self.pos_cap <= 0.0:
            raise ContractError("g_cap and pos_cap must be positive")

    @property
    def d_img_area(self) -> float:
        return (self.d_img_x1 - self.d_img_x0) * (self.d_img_y1 - self.d_img_y0)


def _require_report_in_domain(report: Report, params: EvidenceParams) -> None:
    if report.is_miss:
        return
    if not (params.v_min <= report.v <= params.v_max):
        raise ContractError(f"report value v={report.v} outside [v_min, v_max]")
    if not (
        params.d_img_x0 <= report.x <= params.d_img_x1
        and params.d_img_y0 <= report.y <= params.d_img_y1
    ):
        raise ContractError("report position outside D_img")


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def truncated_gaussian_pos_density(
    x: float, y: float, center_x: float, center_y: float, params: EvidenceParams
) -> float:
    """g_pos(y_pos | b): 2-D isotropic Gaussian truncated to D_img,
    normalized over D_img exactly (product of erf-normalized axes)."""
    sigma = params.g_pos_sigma
    zx = _phi((params.d_img_x1 - center_x) / sigma) - _phi((params.d_img_x0 - center_x) / sigma)
    zy = _phi((params.d_img_y1 - center_y) / sigma) - _phi((params.d_img_y0 - center_y) / sigma)
    if zx <= 0.0 or zy <= 0.0:
        raise ContractError("g_pos normalization degenerate: center too far from D_img")
    fx = math.exp(-0.5 * ((x - center_x) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
    fy = math.exp(-0.5 * ((y - center_y) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
    density = (fx / zx) * (fy / zy)
    if density > params.pos_cap:
        raise ContractError(
            f"g_pos={density} exceeds pos_cap={params.pos_cap}: caps are "
            "floors of the preregistered model, not clamp targets"
        )
    return density


def _checked_value_head(head: ValueHead, v: float, floor: float, cap: float, name: str) -> float:
    value = float(head(v))
    if not math.isfinite(value) or value < 0.0:
        raise ContractError(f"{name}(v) must be finite and non-negative")
    if value < floor:
        raise ContractError(f"{name}(v)={value} below its floor {floor}")
    if value > cap:
        raise ContractError(f"{name}(v)={value} exceeds its cap {cap}")
    return value


def p_vis(
    report: Report,
    bridge_center: tuple[float, float],
    g_v: ValueHead,
    params: EvidenceParams,
) -> float:
    _require_report_in_domain(report, params)
    if report.is_miss:
        return params.pi_miss_vis
    g_val = _checked_value_head(g_v, report.v, 0.0, params.g_cap, "g_v")
    g_pos = truncated_gaussian_pos_density(
        report.x, report.y, bridge_center[0], bridge_center[1], params
    )
    return (1.0 - params.pi_miss_vis) * g_val * g_pos


def p_cens(report: Report, h_c: ValueHead, params: EvidenceParams) -> float:
    _require_report_in_domain(report, params)
    if report.is_miss:
        return params.pi_miss_cens
    h_val = _checked_value_head(h_c, report.v, params.h_floor, math.inf, "h_c")
    return (1.0 - params.pi_miss_cens) * h_val / params.d_img_area


def p_out(report: Report, h_o: ValueHead, params: EvidenceParams) -> float:
    """Analogous to p_cens with its own miss mass and value head."""
    _require_report_in_domain(report, params)
    if report.is_miss:
        return params.pi_miss_out
    h_val = _checked_value_head(h_o, report.v, params.h_floor, math.inf, "h_o")
    return (1.0 - params.pi_miss_out) * h_val / params.d_img_area


@dataclass(frozen=True)
class EvidenceHeads:
    """The frozen head bundle used by the energy: params + value heads."""

    params: EvidenceParams
    g_v: ValueHead
    h_c: ValueHead
    h_o: ValueHead


def likelihood_l1(
    report: Report,
    bridge_center: tuple[float, float],
    q_tilde: float,
    r: float,
    heads: EvidenceHeads,
) -> float:
    """L1(y|b, q_tilde, r) = r[q̃·p_vis + (1−q̃)·p_cens] + (1−r)·p_out."""
    if not (0.0 <= q_tilde <= 1.0):
        raise ContractError(f"q_tilde={q_tilde} outside [0,1]")
    if not (heads.params.r_min <= r <= 1.0):
        raise ContractError(f"r={r} outside [r_min, 1]")
    censored = p_cens(report, heads.h_c, heads.params)
    outlier = p_out(report, heads.h_o, heads.params)
    if q_tilde == 0.0:
        # Censoring equality: identical to L0 by construction — the
        # p_vis branch is not evaluated at all, so a censored report
        # can never contribute through the visible head.
        return r * censored + (1.0 - r) * outlier
    visible = p_vis(report, bridge_center, heads.g_v, heads.params)
    return r * (q_tilde * visible + (1.0 - q_tilde) * censored) + (1.0 - r) * outlier


def likelihood_l0(report: Report, r: float, heads: EvidenceHeads) -> float:
    """L0(y|r) = r·p_cens + (1−r)·p_out."""
    if not (heads.params.r_min <= r <= 1.0):
        raise ContractError(f"r={r} outside [r_min, 1]")
    censored = p_cens(report, heads.h_c, heads.params)
    outlier = p_out(report, heads.h_o, heads.params)
    return r * censored + (1.0 - r) * outlier


__all__ = [
    "EvidenceHeads",
    "EvidenceParams",
    "Report",
    "likelihood_l0",
    "likelihood_l1",
    "p_cens",
    "p_out",
    "p_vis",
    "truncated_gaussian_pos_density",
]
