"""Transaction ledger H and search-cost accounting (spec §4 + substrate).

C(H) = chi * N_returnbirth + mu * N_merge is an explicit TRANSACTION
LEDGER over the event history H: chi and mu are charged when the
event commits and NEVER refunded (kappa, by contrast, is a state term
refunded automatically when episodes are deleted — it lives in
elgs.energy.PriorTerms). Acceptance always compares the energy
INCLUDING the candidate's transaction increment.

The substrate additionally requires a full search-cost ledger
(micro-render accounting: cumulative candidate renders, accepted /
tried, rasterizer and topology-management time, peak memory,
end-to-end GPU hours) and dual-cap accounting (peak rendered rows
<= row cap AND total stored trainable scalars <= baseline budget),
with K-overflow rejections logged as the representational-capacity
failure metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from depth_visibility.errors import ContractError

LEDGER_EVENT_KINDS = (
    "return_birth",
    "merge",
    "birth",
    "fission",
    "truncate",
    "reactivate",
    "prune",
    "rollback",
    "moment_reset",
    "k_overflow_reject",
    "slot_exhausted_reject",
)


@dataclass(frozen=True)
class LedgerEvent:
    kind: str
    round_index: int
    iteration: int
    family_ids: tuple[int, ...]
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in LEDGER_EVENT_KINDS:
            raise ContractError(f"unknown ledger event kind {self.kind!r}")
        if self.iteration < 0 or self.round_index < 0:
            raise ContractError("ledger event indices must be non-negative")


class TransactionLedger:
    """Append-only event history H. Events are never removed; a
    rejected candidate appends a rollback event rather than erasing
    anything (chi/mu charges attach only to COMMITTED events)."""

    def __init__(self) -> None:
        self._events: list[LedgerEvent] = []

    def append(self, event: LedgerEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> tuple[LedgerEvent, ...]:
        return tuple(self._events)

    @property
    def n_return_birth(self) -> int:
        return sum(1 for e in self._events if e.kind == "return_birth")

    @property
    def n_merge(self) -> int:
        return sum(1 for e in self._events if e.kind == "merge")

    def charge(self, chi: float, mu: float) -> float:
        """C(H) — never refunded."""
        if chi < 0.0 or mu < 0.0:
            raise ContractError("chi and mu must be non-negative")
        return chi * self.n_return_birth + mu * self.n_merge

    def to_state(self) -> dict:
        return {
            "events": [
                {
                    "kind": e.kind,
                    "round_index": e.round_index,
                    "iteration": e.iteration,
                    "family_ids": list(e.family_ids),
                    "detail": dict(e.detail),
                }
                for e in self._events
            ]
        }

    @classmethod
    def from_state(cls, payload: dict) -> "TransactionLedger":
        ledger = cls()
        for entry in payload["events"]:
            ledger.append(
                LedgerEvent(
                    kind=str(entry["kind"]),
                    round_index=int(entry["round_index"]),
                    iteration=int(entry["iteration"]),
                    family_ids=tuple(int(v) for v in entry["family_ids"]),
                    detail=dict(entry.get("detail", {})),
                )
            )
        return ledger


@dataclass
class SearchCostLedger:
    """Micro-render and dual-cap accounting (substrate requirement)."""

    row_cap: int
    scalar_budget: int

    candidate_renders: int = 0
    candidates_tried: int = 0
    candidates_accepted: int = 0
    rasterizer_seconds: float = 0.0
    topology_seconds: float = 0.0
    peak_memory_bytes: int = 0
    gpu_hours: float = 0.0
    peak_rendered_rows: int = 0
    peak_stored_scalars: int = 0
    k_overflow_rejects: int = 0

    def __post_init__(self) -> None:
        if self.row_cap <= 0 or self.scalar_budget <= 0:
            raise ContractError("dual caps must be positive")

    def observe_rows(self, rendered_rows: int, stored_scalars: int) -> None:
        if rendered_rows > self.row_cap:
            raise ContractError(
                f"peak rendered rows {rendered_rows} exceed the row cap {self.row_cap}"
            )
        if stored_scalars > self.scalar_budget:
            raise ContractError(
                f"stored trainable scalars {stored_scalars} exceed the "
                f"baseline budget {self.scalar_budget}"
            )
        self.peak_rendered_rows = max(self.peak_rendered_rows, rendered_rows)
        self.peak_stored_scalars = max(self.peak_stored_scalars, stored_scalars)

    def to_state(self) -> dict:
        return {
            "row_cap": self.row_cap,
            "scalar_budget": self.scalar_budget,
            "candidate_renders": self.candidate_renders,
            "candidates_tried": self.candidates_tried,
            "candidates_accepted": self.candidates_accepted,
            "rasterizer_seconds": self.rasterizer_seconds,
            "topology_seconds": self.topology_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "gpu_hours": self.gpu_hours,
            "peak_rendered_rows": self.peak_rendered_rows,
            "peak_stored_scalars": self.peak_stored_scalars,
            "k_overflow_rejects": self.k_overflow_rejects,
        }

    @classmethod
    def from_state(cls, payload: dict) -> "SearchCostLedger":
        return cls(**{k: payload[k] for k in cls.__dataclass_fields__})  # type: ignore[arg-type]


__all__ = [
    "LEDGER_EVENT_KINDS",
    "LedgerEvent",
    "SearchCostLedger",
    "TransactionLedger",
]
