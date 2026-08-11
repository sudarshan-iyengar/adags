"""Structural search: screening, conflict graphs, confirmation order.

SUPPORTING MACHINERY, not a contribution (method page): per-lineage
constrained proposal engines inside an approximate global structural
search — conflict graphs over current ∪ bridge footprints,
Gauss-Seidel passes, a priority queue over screening scores. The
M0-binding CONTRACTS (spec §7 + method schedule) implemented here:

- candidate caps are preregistered and enforced (excess candidates
  are screened out AND ITT-logged, never silently dropped);
- conflict components are deterministic; the component rank is the
  ascending min lineage ID (elgs.acceptance.component_rank_order),
  which is also the slot-grid rank;
- ONE confirmation per component per pass;
- the whole pipeline is deterministic under a fixed seed (candidate
  generation itself takes no RNG — determinism comes from sorted
  iteration everywhere).

Screening accumulators follow the substrate: signed per-(family, bin)
sums over fixed-width frame bins; sign-correct removal estimates.
Proposal thresholds and caps arrive as preregistered config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from depth_visibility.errors import ContractError

from .classification import ITTRecord


@dataclass(frozen=True)
class Candidate:
    """One structural-op proposal with its conflict footprint."""

    candidate_id: str
    op: str
    family_ids: tuple[int, ...]
    screen_score: float
    footprint_frames: tuple[float, float]
    footprint_families: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.family_ids:
            raise ContractError("a candidate must name at least one family")
        if self.footprint_frames[1] < self.footprint_frames[0]:
            raise ContractError("candidate footprint must have start <= end")
        if not self.footprint_families:
            raise ContractError(
                "a candidate must declare its footprint families "
                "(current ∪ bridge)"
            )


class ScreeningAccumulator:
    """Signed per-(family, bin) accumulators over fixed-width bins."""

    def __init__(self, bin_width_frames: float) -> None:
        if bin_width_frames <= 0:
            raise ContractError("bin width must be positive")
        self.bin_width = bin_width_frames
        self._sums: dict[tuple[int, int], float] = {}
        self._counts: dict[tuple[int, int], int] = {}

    def add(self, family_id: int, frame: float, signed_value: float) -> None:
        key = (family_id, int(frame // self.bin_width))
        self._sums[key] = self._sums.get(key, 0.0) + signed_value
        self._counts[key] = self._counts.get(key, 0) + 1

    def bins_above(self, threshold: float) -> tuple[tuple[int, int, float], ...]:
        """(family_id, bin_index, sum) for |sum| > threshold, sorted."""
        if threshold < 0:
            raise ContractError("screening threshold must be non-negative")
        hits = [
            (fid, b, s)
            for (fid, b), s in self._sums.items()
            if abs(s) > threshold
        ]
        return tuple(sorted(hits))

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


def conflict_components(
    candidates: Sequence[Candidate],
) -> tuple[tuple[Candidate, ...], ...]:
    """Connected components of the conflict graph: two candidates
    conflict iff their footprint families intersect AND their frame
    footprints overlap. Components are returned ranked by ascending
    min lineage (family) ID — the slot-grid rank order."""
    ids = list(range(len(candidates)))
    parent = {i: i for i in ids}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            lo, hi = (ri, rj) if ri < rj else (rj, ri)
            parent[hi] = lo

    for i in ids:
        for j in ids[i + 1:]:
            a, b = candidates[i], candidates[j]
            families_overlap = set(a.footprint_families) & set(b.footprint_families)
            frames_overlap = (
                a.footprint_frames[0] <= b.footprint_frames[1]
                and b.footprint_frames[0] <= a.footprint_frames[1]
            )
            if families_overlap and frames_overlap:
                union(i, j)

    groups: dict[int, list[Candidate]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(candidates[i])
    ranked = sorted(
        groups.values(),
        key=lambda group: min(min(c.footprint_families) for c in group),
    )
    return tuple(tuple(sorted(g, key=lambda c: c.candidate_id)) for g in ranked)


@dataclass
class PassPlan:
    """One Gauss-Seidel pass: at most one confirmation per component,
    components in min-lineage rank order, capped and ITT-complete."""

    confirmations: tuple[Candidate, ...]
    itt_records: tuple[ITTRecord, ...]
    component_ranks: tuple[int, ...] = field(default_factory=tuple)


def plan_pass(
    candidates: Sequence[Candidate],
    *,
    round_index: int,
    pass_index: int,
    candidate_cap: int,
) -> PassPlan:
    """Select this pass's confirmations.

    Within each conflict component the highest screen score wins
    (ties by candidate_id — deterministic); the rest of the component
    is deferred (ITT: screened out this pass). Components beyond the
    preregistered cap are rejected AND logged (never silently
    dropped) — mirroring the slot grid's exhaustion rule.
    """
    if candidate_cap < 1:
        raise ContractError("candidate cap must be >= 1")
    components = conflict_components(candidates)
    confirmations: list[Candidate] = []
    itt: list[ITTRecord] = []
    ranks: list[int] = []
    for rank, component in enumerate(components):
        winner = max(component, key=lambda c: (c.screen_score, c.candidate_id))
        for candidate in component:
            if candidate is winner:
                continue
            itt.append(
                ITTRecord(
                    candidate_id=candidate.candidate_id,
                    op=candidate.op,
                    family_id=candidate.family_ids[0],
                    round_index=round_index,
                    screened_in=False,
                    committed=False,
                    rejection_reason=f"conflict_deferred_pass_{pass_index}",
                )
            )
        if rank >= candidate_cap:
            itt.append(
                ITTRecord(
                    candidate_id=winner.candidate_id,
                    op=winner.op,
                    family_id=winner.family_ids[0],
                    round_index=round_index,
                    screened_in=False,
                    committed=False,
                    rejection_reason="candidate_cap_exceeded",
                )
            )
            continue
        confirmations.append(winner)
        ranks.append(rank)
    return PassPlan(
        confirmations=tuple(confirmations),
        itt_records=tuple(itt),
        component_ranks=tuple(ranks),
    )


__all__ = [
    "Candidate",
    "PassPlan",
    "ScreeningAccumulator",
    "conflict_components",
    "plan_pass",
]
