"""Family registry: identity, episodes, tau origins, routing pins.

A FAMILY is a clone-descendant set of Gaussian rows sharing ONE
presence program (spec §1) and tied radiance (substrate). This module
owns the CPU-side registry invariants:

- family IDs are monotone; a retired ID is NEVER reused (rev-3 A3);
- MERGE survivor identity: the merged family retains the OLDER
  parent's family ID, birth time, birth site, and lineage key (ties
  broken by lower family ID); the younger parent's ID is retired;
- motion origins tau_j are fixed at episode creation and never
  changed; FISSION children inherit the parent's tau, REACTIVATE
  episodes get a fresh tau (substrate);
- routing pinning: families with K > 1 (equivalently: any gap) are
  pinned dynamic — the route logit is frozen and the pin logged;
- K <= 4 and interval validity are enforced by IntervalState.

Row binding (the per-row family-ID column on GaussianModel) is
integrated by the trainer wiring; this registry only tracks per-family
row counts through the on_rows_* hooks so the mapping stays auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from depth_visibility.errors import ContractError

from .intervals import IntervalState


@dataclass(frozen=True)
class FamilyRecord:
    """Immutable snapshot of one family's identity + interval state."""

    family_id: int
    birth_time: float
    birth_site: tuple[float, float, float]
    lineage_key: str
    interval: IntervalState
    tau: tuple[float, ...]
    retired: bool = False

    def __post_init__(self) -> None:
        if self.family_id < 0:
            raise ContractError("family_id must be non-negative")
        if self.interval.K > 0 and len(self.tau) != self.interval.K:
            raise ContractError(
                f"family {self.family_id}: {len(self.tau)} tau origins for "
                f"K={self.interval.K} episodes"
            )
        if self.interval.K == 0 and self.tau:
            raise ContractError("K=0 empty program carries no tau origins")

    @property
    def routing_pinned(self) -> bool:
        """Pinned dynamic iff K > 1 (episodes are disjoint, so K > 1
        is exactly the any-gap condition of the substrate rule)."""
        return self.interval.K > 1


class FamilyRegistry:
    """Mutable registry of families with fail-closed invariants."""

    def __init__(self) -> None:
        self._records: dict[int, FamilyRecord] = {}
        self._next_family_id = 0
        self._row_counts: dict[int, int] = {}
        self._pin_log: list[dict] = []

    # -- identity ---------------------------------------------------------

    def create_family(
        self,
        *,
        birth_time: float,
        birth_site: tuple[float, float, float],
        lineage_key: str,
        interval: IntervalState,
        tau: tuple[float, ...],
    ) -> FamilyRecord:
        family_id = self._next_family_id
        self._next_family_id += 1
        record = FamilyRecord(
            family_id=family_id,
            birth_time=birth_time,
            birth_site=birth_site,
            lineage_key=lineage_key,
            interval=interval,
            tau=tau,
        )
        self._records[family_id] = record
        self._row_counts[family_id] = 0
        self._log_pin_if_needed(record)
        return record

    def get(self, family_id: int) -> FamilyRecord:
        try:
            return self._records[family_id]
        except KeyError:
            raise ContractError(f"unknown family_id {family_id}") from None

    def require_active(self, family_id: int) -> FamilyRecord:
        record = self.get(family_id)
        if record.retired:
            raise ContractError(f"family {family_id} is retired")
        return record

    def active_ids(self) -> tuple[int, ...]:
        return tuple(sorted(fid for fid, r in self._records.items() if not r.retired))

    @property
    def next_family_id(self) -> int:
        """The never-reused ID watermark."""
        return self._next_family_id

    # -- interval / tau updates (called by structural ops only) -----------

    def replace_interval(
        self, family_id: int, interval: IntervalState, tau: tuple[float, ...]
    ) -> FamilyRecord:
        """Swap a family's presence program (structural op commit path).

        tau entries for episodes that survive must be carried over by
        the caller; this method enforces only count consistency — the
        per-op inheritance rules live in elgs.ops against the
        preregistered transition table.
        """
        record = self.require_active(family_id)
        updated = replace(record, interval=interval, tau=tau)
        self._records[family_id] = updated
        self._log_pin_if_needed(updated)
        return updated

    # -- MERGE survivor convention (rev-3 A3) -----------------------------

    def merge_survivor_of(self, family_a: int, family_b: int) -> tuple[int, int]:
        """(survivor_id, retired_id): older birth time wins, ties by
        lower family ID. Identity fields of the survivor are retained
        as-is; the younger ID is retired and never reused."""
        if family_a == family_b:
            raise ContractError("MERGE requires two distinct families")
        rec_a = self.require_active(family_a)
        rec_b = self.require_active(family_b)
        if (rec_a.birth_time, rec_a.family_id) <= (rec_b.birth_time, rec_b.family_id):
            return rec_a.family_id, rec_b.family_id
        return rec_b.family_id, rec_a.family_id

    def retire_family(self, family_id: int) -> None:
        record = self.require_active(family_id)
        self._records[family_id] = replace(record, retired=True)

    # -- row bookkeeping --------------------------------------------------

    def on_rows_added(self, family_id: int, count: int) -> None:
        self.require_active(family_id)
        if count < 0:
            raise ContractError("row count delta must be non-negative")
        self._row_counts[family_id] += count

    def on_rows_pruned(self, family_id: int, count: int) -> None:
        self.get(family_id)
        if count < 0 or count > self._row_counts[family_id]:
            raise ContractError(
                f"family {family_id}: cannot prune {count} of "
                f"{self._row_counts[family_id]} rows"
            )
        self._row_counts[family_id] -= count

    def row_count(self, family_id: int) -> int:
        self.get(family_id)
        return self._row_counts[family_id]

    def redirect_rows(self, source_family: int, target_family: int) -> int:
        """Move all row ownership from source to target (MERGE commit)."""
        self.get(source_family)
        self.require_active(target_family)
        moved = self._row_counts[source_family]
        self._row_counts[source_family] = 0
        self._row_counts[target_family] += moved
        return moved

    # -- routing pin log --------------------------------------------------

    def _log_pin_if_needed(self, record: FamilyRecord) -> None:
        if record.routing_pinned:
            self._pin_log.append(
                {"family_id": record.family_id, "K": record.interval.K}
            )

    @property
    def pin_log(self) -> tuple[dict, ...]:
        return tuple(self._pin_log)


__all__ = ["FamilyRecord", "FamilyRegistry"]
