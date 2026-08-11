"""Canonical interval/latch state (formal spec rev 4, §1).

Single source of truth for the family interval state
``(K, latch_pre, latch_post, a)``:

- latch bits exist ONLY on the two outer endpoints (b_1 / d_K); all
  four patterns are admissible; interior latches do not exist;
- ``dim(a) = 2K + 1 - n_lat`` in the canonical coordinate order
  ``[slack_pre iff latch_pre=0], len_1, gap_1, ..., gap_{K-1}, len_K,
  [slack_post iff latch_post=0]``;
- softmax coordinates are strictly positive, so exact boundary contact
  is representable ONLY by a latch bit — a zero slack coordinate does
  not exist in this parameterization and no caller may target one;
- every K-changing operation writes child states through the inverse
  map defined here.

Importable without CUDA extensions (CPU test suite runs on any node).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from depth_visibility.errors import ContractError, SchemaError

INTERVAL_STATE_SCHEMA = "elgs-interval-state-v1"
K_MAX = 4

# Relative tolerance for the Omega-sum identity check in the inverse
# map. Targets are constructed analytically by structural ops, so any
# larger violation is a caller bug, not float noise.
_OMEGA_SUM_RTOL = 1e-6


@dataclass(frozen=True)
class IntervalConfig:
    """Structural interval constants (category-1, preregistered).

    T is the sequence span, w_m the outer margin (b_1 >= -w_m,
    d_K <= T + w_m), w the presence edge half-width, and the floors
    are floor_len = 2w + delta_len, floor_gap = 2w + delta_gap.
    """

    T: float
    w_m: float
    w: float
    floor_len: float
    floor_gap: float

    def __post_init__(self) -> None:
        for name in ("T", "w_m", "w", "floor_len", "floor_gap"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ContractError(f"interval config {name} must be finite")
        if self.T <= 0 or self.w_m < 0 or self.w <= 0:
            raise ContractError("interval config requires T > 0, w_m >= 0, w > 0")
        if self.floor_len <= 0 or self.floor_gap <= 0:
            raise ContractError("interval floors must be strictly positive")

    @property
    def omega(self) -> float:
        return self.T + 2.0 * self.w_m

    def omega_free(self, K: int) -> float:
        """Omega_free = Omega - K*floor_len - (K-1)*floor_gap.

        NEVER depends on the latch pattern: a latched slack is
        identically zero and simply has no coordinate.
        """
        _require_valid_k(K)
        return self.omega - K * self.floor_len - (K - 1) * self.floor_gap

    def require_admissible_k(self, K: int) -> None:
        if self.omega_free(K) <= 0.0:
            raise ContractError(
                f"K={K} inadmissible: omega_free={self.omega_free(K)} <= 0"
            )


@dataclass(frozen=True)
class IntervalState:
    """The canonical state (K, latch_pre, latch_post, a).

    K == 0 is the defined empty program: no latch bits, no vector
    (both stored as None); renders nothing; prune-pending.
    """

    K: int
    latch_pre: bool | None
    latch_post: bool | None
    a: torch.Tensor | None

    def __post_init__(self) -> None:
        _require_valid_k(self.K)
        if self.K == 0:
            if self.latch_pre is not None or self.latch_post is not None:
                raise ContractError("K=0 empty program carries no latch bits")
            if self.a is not None:
                raise ContractError("K=0 empty program carries no vector a")
            return
        if self.latch_pre is None or self.latch_post is None:
            raise ContractError("K>=1 state requires both latch bits")
        if self.a is None or self.a.dim() != 1:
            raise ContractError("interval vector a must be a 1-D tensor")
        expected = expected_dim(self.K, self.latch_pre, self.latch_post)
        if self.a.numel() != expected:
            raise ContractError(
                f"dim(a)={self.a.numel()} != 2K+1-n_lat={expected} "
                f"for K={self.K}, latches=({self.latch_pre},{self.latch_post})"
            )
        if not torch.isfinite(self.a).all():
            raise ContractError("interval vector a contains non-finite values")

    @property
    def n_lat(self) -> int:
        if self.K == 0:
            raise ContractError("K=0 empty program has no latch pattern")
        return int(self.latch_pre) + int(self.latch_post)


@dataclass(frozen=True)
class IntervalRealization:
    """Forward-map output: realized spans and endpoints.

    Latched slacks are exactly 0.0. Endpoints follow by summation:
    b_1 = -w_m + slack_pre, d_k = b_k + len_k, b_{k+1} = d_k + gap_k.
    """

    slack_pre: torch.Tensor
    slack_post: torch.Tensor
    lens: torch.Tensor
    gaps: torch.Tensor
    b: torch.Tensor
    d: torch.Tensor


def _require_valid_k(K: int) -> None:
    if not isinstance(K, int) or isinstance(K, bool):
        raise ContractError(f"K must be an int, got {type(K).__name__}")
    if K < 0 or K > K_MAX:
        raise ContractError(f"K={K} outside the admissible range 0..{K_MAX}")


def expected_dim(K: int, latch_pre: bool, latch_post: bool) -> int:
    """dim(a) = 2K + 1 - n_lat (K >= 1 only)."""
    _require_valid_k(K)
    if K == 0:
        raise ContractError("K=0 empty program has no vector a")
    return 2 * K + 1 - (int(latch_pre) + int(latch_post))


def coordinate_labels(K: int, latch_pre: bool, latch_post: bool) -> tuple[str, ...]:
    """Canonical (serialization) coordinate order."""
    if K == 0:
        raise ContractError("K=0 empty program has no coordinates")
    _require_valid_k(K)
    labels: list[str] = []
    if not latch_pre:
        labels.append("slack_pre")
    for k in range(1, K + 1):
        labels.append(f"len_{k}")
        if k < K:
            labels.append(f"gap_{k}")
    if not latch_post:
        labels.append("slack_post")
    return tuple(labels)


def forward(state: IntervalState, config: IntervalConfig) -> IntervalRealization:
    """Forward map: softmax(a) over exactly the active coordinates.

    unlatched slack = omega_free * sigma; latched slack == 0 (no
    coordinate); len_k = floor_len + omega_free * sigma; gap_k =
    floor_gap + omega_free * sigma. The Omega-sum identity holds for
    every latch pattern, so d_K <= T + w_m identically under any
    optimizer step, with equality iff latch_post = 1.
    """
    if state.K == 0:
        raise ContractError("K=0 empty program has no forward realization")
    config.require_admissible_k(state.K)
    assert state.a is not None and state.latch_pre is not None and state.latch_post is not None
    omega_free = config.omega_free(state.K)
    sigma = torch.softmax(state.a, dim=0)
    spans = omega_free * sigma

    zero = state.a.new_zeros(())
    idx = 0
    if state.latch_pre:
        slack_pre = zero
    else:
        slack_pre = spans[idx]
        idx += 1
    lens: list[torch.Tensor] = []
    gaps: list[torch.Tensor] = []
    for k in range(1, state.K + 1):
        lens.append(config.floor_len + spans[idx])
        idx += 1
        if k < state.K:
            gaps.append(config.floor_gap + spans[idx])
            idx += 1
    if state.latch_post:
        slack_post = zero
    else:
        slack_post = spans[idx]
        idx += 1
    if idx != state.a.numel():
        raise ContractError("forward map consumed a wrong coordinate count")

    lens_t = torch.stack(lens)
    gaps_t = torch.stack(gaps) if gaps else state.a.new_zeros((0,))
    b = []
    d = []
    cursor = -config.w_m + slack_pre
    for k in range(state.K):
        b.append(cursor)
        cursor = cursor + lens_t[k]
        d.append(cursor)
        if k < state.K - 1:
            cursor = cursor + gaps_t[k]
    return IntervalRealization(
        slack_pre=slack_pre,
        slack_post=slack_post,
        lens=lens_t,
        gaps=gaps_t,
        b=torch.stack(b),
        d=torch.stack(d),
    )


def inverse(
    K: int,
    latch_pre: bool,
    latch_post: bool,
    slack_pre: float,
    lens: list[float] | tuple[float, ...],
    gaps: list[float] | tuple[float, ...],
    slack_post: float,
    config: IntervalConfig,
    *,
    dtype: torch.dtype = torch.float32,
) -> IntervalState:
    """Inverse map (used by every structural op to write child states).

    sigma_i = (value_i - floor_i) / omega_free; a_i = log sigma_i -
    max_j log sigma_j (deterministic gauge, max a_i = 0). Targets at
    an exact floor are inadmissible as stated; a target with an outer
    slack exactly 0 must set the corresponding latch instead (passing
    a zero slack for an unlatched side raises).
    """
    _require_valid_k(K)
    if K == 0:
        raise ContractError("K=0 empty program is constructed directly, not inverted")
    config.require_admissible_k(K)
    if len(lens) != K or len(gaps) != K - 1:
        raise ContractError(
            f"inverse map needs {K} lens and {K - 1} gaps, "
            f"got {len(lens)} and {len(gaps)}"
        )
    omega_free = config.omega_free(K)

    values: list[float] = []
    floors: list[float] = []

    def _push(value: float, floor: float, label: str) -> None:
        if not math.isfinite(value):
            raise ContractError(f"inverse target {label} is non-finite")
        if value <= floor:
            raise ContractError(
                f"inverse target {label}={value} at or below its floor {floor}: "
                "exact-floor targets are inadmissible; an exactly-zero outer "
                "slack must set the latch bit instead"
            )
        values.append(value)
        floors.append(floor)

    if latch_pre:
        if slack_pre != 0.0:
            raise ContractError("latched slack_pre must be exactly 0")
    else:
        _push(float(slack_pre), 0.0, "slack_pre")
    for k in range(1, K + 1):
        _push(float(lens[k - 1]), config.floor_len, f"len_{k}")
        if k < K:
            _push(float(gaps[k - 1]), config.floor_gap, f"gap_{k}")
    if latch_post:
        if slack_post != 0.0:
            raise ContractError("latched slack_post must be exactly 0")
    else:
        _push(float(slack_post), 0.0, "slack_post")

    total = sum(v - f for v, f in zip(values, floors))
    if abs(total - omega_free) > _OMEGA_SUM_RTOL * max(omega_free, 1.0):
        raise ContractError(
            f"inverse targets violate the Omega-sum identity: "
            f"sum(value-floor)={total} vs omega_free={omega_free}"
        )

    log_sigma = [math.log((v - f) / omega_free) for v, f in zip(values, floors)]
    gauge = max(log_sigma)
    a = torch.tensor([ls - gauge for ls in log_sigma], dtype=dtype)
    return IntervalState(K=K, latch_pre=latch_pre, latch_post=latch_post, a=a)


def empty_program() -> IntervalState:
    """K=0: renders nothing; prune-pending."""
    return IntervalState(K=0, latch_pre=None, latch_post=None, a=None)


def birth_interval_state(
    t_birth: float,
    config: IntervalConfig,
    *,
    dtype: torch.dtype = torch.float32,
) -> IntervalState:
    """BIRTH interval encoding (spec §5): (latch_pre, latch_post) = (0, 1).

    d_1 = T + w_m EXACT via the terminal latch bit, never a zero slack
    coordinate. a in R^2 via the inverse map with targets
    slack_pre = t_birth + w_m and len_1 = T + w_m - t_birth; admissible
    iff len_1 > floor_len strictly AND t_birth > -w_m (slack_pre > 0).
    """
    len_1 = config.T + config.w_m - t_birth
    slack_pre = t_birth + config.w_m
    if not (len_1 > config.floor_len and t_birth > -config.w_m):
        raise ContractError(
            f"BIRTH at t_birth={t_birth} inadmissible: needs "
            f"len_1={len_1} > floor_len={config.floor_len} and "
            f"t_birth > -w_m={-config.w_m}"
        )
    return inverse(
        K=1,
        latch_pre=False,
        latch_post=True,
        slack_pre=slack_pre,
        lens=[len_1],
        gaps=[],
        slack_post=0.0,
        config=config,
        dtype=dtype,
    )


_DTYPE_NAMES = {torch.float32: "float32", torch.float64: "float64"}
_DTYPES_BY_NAME = {name: dt for dt, name in _DTYPE_NAMES.items()}


def serialize_state(state: IntervalState) -> dict:
    """Canonical serialization: (K, latch_pre, latch_post, a).

    The dtype is persisted so a round-trip is bit-exact: python floats
    are doubles, so float32 AND float64 logits survive serialization
    losslessly as long as the loader restores the original dtype.
    """
    if state.K == 0:
        return {"schema_version": INTERVAL_STATE_SCHEMA, "K": 0}
    assert state.a is not None
    if state.a.dtype not in _DTYPE_NAMES:
        raise ContractError(f"unsupported interval dtype {state.a.dtype}")
    return {
        "schema_version": INTERVAL_STATE_SCHEMA,
        "K": state.K,
        "latch_pre": bool(state.latch_pre),
        "latch_post": bool(state.latch_post),
        "a": [float(v) for v in state.a.tolist()],
        "dtype": _DTYPE_NAMES[state.a.dtype],
    }


def deserialize_state(payload: dict, *, dtype: torch.dtype | None = None) -> IntervalState:
    """Loader with the MANDATORY dimension and validity checks.

    dtype=None restores the persisted dtype (bit-exact round trip);
    passing a dtype overrides it explicitly.
    """
    if not isinstance(payload, dict):
        raise SchemaError("interval-state payload must be a dict")
    if payload.get("schema_version") != INTERVAL_STATE_SCHEMA:
        raise SchemaError(
            f"unsupported interval-state schema: {payload.get('schema_version')!r}"
        )
    K = payload.get("K")
    _require_valid_k(K)
    if K == 0:
        forbidden = {"latch_pre", "latch_post", "a"} & set(payload)
        if forbidden:
            raise SchemaError(f"K=0 payload must not carry {sorted(forbidden)}")
        return empty_program()
    for key in ("latch_pre", "latch_post", "a"):
        if key not in payload:
            raise SchemaError(f"interval-state payload missing {key}")
    latch_pre = payload["latch_pre"]
    latch_post = payload["latch_post"]
    if not isinstance(latch_pre, bool) or not isinstance(latch_post, bool):
        raise SchemaError("latch bits must be booleans")
    raw = payload["a"]
    if not isinstance(raw, list) or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw
    ):
        raise SchemaError("interval vector a must be a list of numbers")
    if dtype is None:
        name = payload.get("dtype", "float32")
        if name not in _DTYPES_BY_NAME:
            raise SchemaError(f"unsupported interval dtype {name!r}")
        dtype = _DTYPES_BY_NAME[name]
    a = torch.tensor([float(v) for v in raw], dtype=dtype)
    # IntervalState.__post_init__ enforces dim(a) = 2K+1-n_lat.
    return IntervalState(K=K, latch_pre=latch_pre, latch_post=latch_post, a=a)


__all__ = [
    "INTERVAL_STATE_SCHEMA",
    "K_MAX",
    "IntervalConfig",
    "IntervalRealization",
    "IntervalState",
    "birth_interval_state",
    "coordinate_labels",
    "deserialize_state",
    "empty_program",
    "expected_dim",
    "forward",
    "inverse",
    "serialize_state",
]
