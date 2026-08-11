"""SNIS acceptance: weights, CRN pairing, bootstrap, slot grid (spec §7).

CANONICAL ESTIMATOR (errata E2): self-normalized importance sampling
R_hat = sum_i a_i*l(x_i) / sum_i a_i with samples from the mixture
m = lambda_u*nu + (1-lambda_u)*pi_D and a_i = min{1/lambda_u,
nu(x_i)/m(x_i)}. The true weight is <= 1/lambda_u ALWAYS, so the clip
is PROVABLY INACTIVE — it is retained as a formal guard and this
module asserts it never fires. R_hat is a ratio estimator: strongly
consistent for the nu-mean as n -> infinity, bounded-weight stable,
and NOT unbiased in general at finite sample size (ratio-estimator
bias O(1/n)); no unbiasedness or exactness of the sampled estimate is
claimed or used anywhere. "Exact" survives only for the non-sampled
closed-form tracker/prior deltas added outside the sampled render
estimate, and for the weights themselves.

COMMON RANDOM NUMBERS: identical {x_i} for incumbent and candidate;
Delta_E_hat is the paired SNIS difference plus the exact deltas
(including the transaction-ledger increment). SE: paired cluster
bootstrap over (camera, frame) units, B=200, the SAME resample
indices for candidate and incumbent, weights renormalized within each
replicate; SE = sd of paired replicate differences; SE undefined =>
reject. Degeneracy (<= 5 clusters) => reject. Accept iff
Delta_E_hat + k*SE < 0.

Sample partitioning: NO hashing — the reserved pool is pre-partitioned
at iteration 0 into an indexed grid of slots (round, pass, rank) with
rank = the deterministic ordering of conflict components (min lineage
ID); assignment is injective by construction; exhaustion => the
candidate is rejected and logged; unused slots discarded; refits
never see confirmation samples (the trainer excludes reserved units —
the grid exposes them). pi_D and lambda_u are FROZEN before any
confirmation draw.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Sequence

from depth_visibility.errors import ContractError

_CLIP_SLACK = 1e-9  # float-tolerance on the provably-inactive clip guard

DEGENERACY_MIN_CLUSTERS = 6  # "<= 5 clusters => reject"
BOOTSTRAP_REPLICATES = 200


@dataclass(frozen=True)
class SnisSample:
    """One confirmation sample x_i under CRN.

    unit: the (camera, frame) cluster the bootstrap resamples;
    nu_density / mix_density: densities of x_i under nu and m;
    loss_incumbent / loss_candidate: l(x_i) for the two arms rendered
    on the SAME x_i (paired by construction).
    """

    unit: tuple[int, float]
    nu_density: float
    mix_density: float
    loss_incumbent: float
    loss_candidate: float

    def __post_init__(self) -> None:
        if self.nu_density < 0.0 or self.mix_density <= 0.0:
            raise ContractError("sample densities must be nu >= 0, m > 0")
        for name in ("loss_incumbent", "loss_candidate"):
            if not math.isfinite(getattr(self, name)):
                raise ContractError(f"{name} must be finite")


def snis_weight(sample: SnisSample, lambda_u: float) -> float:
    """a_i = min{1/lambda_u, nu/m}; the min is a guard that must be
    inactive (m >= lambda_u * nu identically)."""
    if not (0.0 < lambda_u <= 1.0):
        raise ContractError("lambda_u must lie in (0, 1]")
    w_max = 1.0 / lambda_u
    raw = sample.nu_density / sample.mix_density
    if raw > w_max * (1.0 + _CLIP_SLACK):
        raise ContractError(
            f"SNIS weight {raw} exceeds 1/lambda_u={w_max}: the mixture "
            "density is inconsistent with its own lambda_u (the clip is "
            "provably inactive for correct inputs, so this is a bug)"
        )
    return min(w_max, raw)


def _snis_ratio(samples: Sequence[SnisSample], lambda_u: float, arm: str) -> float:
    num = 0.0
    den = 0.0
    for sample in samples:
        weight = snis_weight(sample, lambda_u)
        loss = getattr(sample, f"loss_{arm}")
        num += weight * loss
        den += weight
    if den <= 0.0:
        raise ContractError("SNIS normalization degenerate (zero weight mass)")
    return num / den


def paired_snis_delta(samples: Sequence[SnisSample], lambda_u: float) -> float:
    """Candidate-minus-incumbent paired SNIS difference (shared x_i,
    shared weights — the CRN design)."""
    if not samples:
        raise ContractError("paired SNIS delta needs at least one sample")
    return _snis_ratio(samples, lambda_u, "candidate") - _snis_ratio(
        samples, lambda_u, "incumbent"
    )


def effective_sample_size(samples: Sequence[SnisSample], lambda_u: float) -> float:
    weights = [snis_weight(s, lambda_u) for s in samples]
    total = sum(weights)
    square = sum(w * w for w in weights)
    if square == 0.0:
        raise ContractError("ESS undefined with zero weights")
    return (total * total) / square


def paired_cluster_bootstrap_se(
    samples: Sequence[SnisSample],
    lambda_u: float,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> float:
    """SE of the paired delta: cluster bootstrap over (camera, frame)
    units, SAME resample indices for both arms, per-replicate SNIS
    renormalization; SE = sd of paired replicate differences.

    Degeneracy (<= 5 distinct units) or an undefined SE => reject
    (raised as ContractError; the caller records the rejection).
    """
    by_unit: dict[tuple[int, float], list[SnisSample]] = {}
    for sample in samples:
        by_unit.setdefault(sample.unit, []).append(sample)
    units = sorted(by_unit)
    if len(units) < DEGENERACY_MIN_CLUSTERS:
        raise ContractError(
            f"bootstrap degeneracy: {len(units)} (camera, frame) clusters "
            f"<= 5 — candidate rejected"
        )
    if replicates < 2:
        raise ContractError("bootstrap needs at least 2 replicates")
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(replicates):
        # One index draw, applied to BOTH arms (paired replicate).
        drawn = [units[rng.randrange(len(units))] for _ in units]
        replicate: list[SnisSample] = []
        for unit in drawn:
            replicate.extend(by_unit[unit])
        # Per-replicate renormalization is exactly the self-normalized
        # form applied to the replicate's samples.
        deltas.append(paired_snis_delta(replicate, lambda_u))
    se = statistics.stdev(deltas)
    if not math.isfinite(se):
        raise ContractError("bootstrap SE undefined — candidate rejected")
    return se


@dataclass(frozen=True)
class FrozenSamplerParams:
    """pi_D identity and lambda_u, frozen BEFORE any confirmation draw."""

    lambda_u: float
    pi_d_identity: str
    frozen: bool = False

    def require_frozen(self) -> None:
        if not self.frozen:
            raise ContractError(
                "pi_D / lambda_u must be frozen before any confirmation draw"
            )


@dataclass(frozen=True)
class AcceptanceRecord:
    """Per-decision acceptance evidence for the ledger/logs."""

    delta_render: float
    exact_deltas: float
    transaction_increment: float
    se: float
    k: float
    n_samples: int
    ess: float
    n_units: int
    accepted: bool

    @property
    def delta_total(self) -> float:
        return self.delta_render + self.exact_deltas + self.transaction_increment


def decide(
    samples: Sequence[SnisSample],
    params: FrozenSamplerParams,
    *,
    exact_deltas: float,
    transaction_increment: float,
    k: float,
    bootstrap_seed: int,
) -> AcceptanceRecord:
    """Accept iff Delta_E_hat + k*SE < 0, with Delta_E_hat = paired
    SNIS render delta + exact closed-form deltas + the transaction
    increment. Degeneracy or undefined SE raise (reject paths)."""
    params.require_frozen()
    if k < 0.0:
        raise ContractError("k must be non-negative")
    if not math.isfinite(exact_deltas) or not math.isfinite(transaction_increment):
        raise ContractError("exact deltas must be finite")
    delta_render = paired_snis_delta(samples, params.lambda_u)
    se = paired_cluster_bootstrap_se(samples, params.lambda_u, seed=bootstrap_seed)
    ess = effective_sample_size(samples, params.lambda_u)
    total = delta_render + exact_deltas + transaction_increment
    return AcceptanceRecord(
        delta_render=delta_render,
        exact_deltas=exact_deltas,
        transaction_increment=transaction_increment,
        se=se,
        k=k,
        n_samples=len(samples),
        ess=ess,
        n_units=len({s.unit for s in samples}),
        accepted=(total + k * se) < 0.0,
    )


class SlotExhausted(ContractError):
    """The pre-partitioned grid has no slot for this rank: the
    candidate is rejected and the exhaustion logged."""


@dataclass
class SlotGrid:
    """Iteration-0 pre-partition of the reserved confirmation pool.

    slots_per_pass is the preregistered candidate cap per pass; the
    grid holds rounds x passes x slots_per_pass slots, each an index
    range over the reserved pool. (round, pass, rank) -> slot is
    injective by construction; consumed slots may not be reused;
    unused slots are simply never drawn. reserved_units() exposes the
    pool for the trainer's refit-sampler exclusion.
    """

    n_rounds: int
    n_passes: int
    slots_per_pass: int
    units_per_slot: int
    reserved_pool: tuple[tuple[int, float], ...]
    _consumed: set = field(default_factory=set)

    def __post_init__(self) -> None:
        needed = self.n_rounds * self.n_passes * self.slots_per_pass * self.units_per_slot
        if len(self.reserved_pool) < needed:
            raise ContractError(
                f"reserved pool holds {len(self.reserved_pool)} units, "
                f"grid needs {needed}"
            )
        if len(set(self.reserved_pool)) != len(self.reserved_pool):
            raise ContractError("reserved pool units must be unique")

    def slot_index(self, round_i: int, pass_i: int, rank: int) -> int:
        if not (0 <= round_i < self.n_rounds and 0 <= pass_i < self.n_passes):
            raise ContractError("slot round/pass outside the grid")
        if rank < 0:
            raise ContractError("rank must be non-negative")
        if rank >= self.slots_per_pass:
            raise SlotExhausted(
                f"rank {rank} exceeds the preregistered cap "
                f"{self.slots_per_pass}: candidate rejected (logged)"
            )
        return (round_i * self.n_passes + pass_i) * self.slots_per_pass + rank

    def draw(self, round_i: int, pass_i: int, rank: int) -> tuple[tuple[int, float], ...]:
        index = self.slot_index(round_i, pass_i, rank)
        if index in self._consumed:
            raise ContractError(f"slot {(round_i, pass_i, rank)} already consumed")
        self._consumed.add(index)
        start = index * self.units_per_slot
        return self.reserved_pool[start : start + self.units_per_slot]

    def reserved_units(self) -> frozenset:
        """Every reserved (camera, frame) unit — the trainer's refit
        sampler must never draw these (spec §7)."""
        return frozenset(self.reserved_pool)


def component_rank_order(component_min_lineage_ids: Sequence[int]) -> tuple[int, ...]:
    """Deterministic conflict-component ordering: ascending min lineage
    ID; the position in this order is the slot rank."""
    if len(set(component_min_lineage_ids)) != len(component_min_lineage_ids):
        raise ContractError("component min-lineage ids must be unique")
    return tuple(sorted(component_min_lineage_ids))


def crn_seed(base_seed: int, round_i: int, pass_i: int, rank: int) -> int:
    """Deterministic per-decision CRN seed (no hashing of content —
    pure index arithmetic on the slot coordinates)."""
    return ((base_seed * 1_000_003 + round_i) * 1_000_003 + pass_i) * 1_000_003 + rank


__all__ = [
    "AcceptanceRecord",
    "BOOTSTRAP_REPLICATES",
    "DEGENERACY_MIN_CLUSTERS",
    "FrozenSamplerParams",
    "SlotExhausted",
    "SlotGrid",
    "SnisSample",
    "component_rank_order",
    "crn_seed",
    "decide",
    "effective_sample_size",
    "paired_cluster_bootstrap_se",
    "paired_snis_delta",
    "snis_weight",
]
