"""Transaction application with snapshot/rollback (spec §7 rollback rule).

TransactionPlans are PURE — candidate evaluation never mutates state,
so a rejected candidate has nothing to undo by construction. This
module owns the COMMIT path: applying an accepted plan to the CPU
state bundle (registry, binding, ledger, search cost) atomically,
with a full snapshot taken first and restored on ANY mid-apply
failure — the incumbent must never be observable in a half-mutated
state. Row and optimizer-moment mutations on the GaussianModel are
returned as DIRECTIVES for the trainer integration (single-owner
file), not performed here.

PRUNE-family cluster fate (disclosed ruling, carried by the prereg
table's spec_silent cell): binding entries of a retired family are
left untouched — the binding freeze permits no third mutation kind —
and the clusters are inactive DE FACTO because energy iterates only
active families. The inactivity is recorded in the ledger event.
"""

from __future__ import annotations

from dataclasses import dataclass

from depth_visibility.errors import ContractError

from .clusters import BindingTable
from .families import FamilyRegistry
from .ops import TransactionPlan
from .transaction_ledger import LedgerEvent, SearchCostLedger, TransactionLedger


@dataclass
class StateBundle:
    """The CPU-side structural state a commit mutates atomically."""

    registry: FamilyRegistry
    binding: BindingTable
    ledger: TransactionLedger
    search_cost: SearchCostLedger

    def snapshot(self) -> dict:
        return {
            "registry": self.registry.to_state(),
            "binding": self.binding.to_state(),
            "ledger": self.ledger.to_state(),
            "search_cost": self.search_cost.to_state(),
        }

    def restore(self, snapshot: dict) -> None:
        self.registry = FamilyRegistry.from_state(snapshot["registry"])
        self.binding = BindingTable.from_state(snapshot["binding"])
        self.ledger = TransactionLedger.from_state(snapshot["ledger"])
        self.search_cost = SearchCostLedger.from_state(snapshot["search_cost"])


@dataclass(frozen=True)
class CommitDirectives:
    """What the trainer must do to the GaussianModel after a commit."""

    moment_reset_family_ids: tuple[int, ...]
    row_redirect: tuple[int, int] | None  # (from_family, to_family)
    new_family_id: int | None
    retired_family_id: int | None


def apply_plan(bundle: StateBundle, plan: TransactionPlan) -> CommitDirectives:
    """Commit an ACCEPTED plan. Atomic: on any failure the bundle is
    restored to its pre-apply snapshot and the error re-raised."""
    snapshot = bundle.snapshot()
    try:
        new_family_id: int | None = None
        if plan.op == "BIRTH":
            (family_id,) = plan.family_ids
            record = bundle.registry.create_family(
                birth_time=float(plan.detail["t_birth"]),
                birth_site=tuple(plan.detail["birth_site"]),
                lineage_key=str(plan.detail["lineage_key"]),
                interval=plan.child_intervals[family_id],
                tau=plan.child_tau[family_id],
            )
            if record.family_id != family_id:
                raise ContractError(
                    f"BIRTH id drift: planned {family_id}, assigned {record.family_id}"
                )
            new_family_id = record.family_id
        else:
            for family_id in plan.family_ids:
                if plan.retire_family == family_id and plan.op == "PRUNE_FAMILY":
                    continue  # retirement handles it below
                if family_id in plan.child_intervals:
                    child = plan.child_intervals[family_id]
                    if child.K == 0 and plan.op in ("MERGE",):
                        continue  # the retired parent is retired, not emptied
                    bundle.registry.replace_interval(
                        family_id, child, plan.child_tau[family_id]
                    )
        row_redirect = None
        if plan.op == "MERGE":
            assert plan.retire_family is not None and plan.survivor_identity is not None
            moved = bundle.registry.redirect_rows(
                plan.retire_family, plan.survivor_identity
            )
            bundle.binding.redirect_for_merge(
                retired_family=plan.retire_family,
                surviving_family=plan.survivor_identity,
            )
            bundle.registry.retire_family(plan.retire_family)
            row_redirect = (plan.retire_family, plan.survivor_identity)
            _ = moved
        elif plan.op == "PRUNE_FAMILY":
            assert plan.retire_family is not None
            bundle.registry.replace_interval(
                plan.retire_family,
                plan.child_intervals[plan.retire_family],
                plan.child_tau[plan.retire_family],
            )
            bundle.registry.retire_family(plan.retire_family)
        for event in plan.ledger_events:
            bundle.ledger.append(event)
        for family_id in plan.moment_resets:
            bundle.ledger.append(
                LedgerEvent(
                    "moment_reset",
                    plan.ledger_events[0].round_index if plan.ledger_events else 0,
                    plan.ledger_events[0].iteration if plan.ledger_events else 0,
                    (family_id,),
                    {"op": plan.op},
                )
            )
        return CommitDirectives(
            moment_reset_family_ids=plan.moment_resets,
            row_redirect=row_redirect,
            new_family_id=new_family_id,
            retired_family_id=plan.retire_family,
        )
    except Exception:
        bundle.restore(snapshot)
        raise


__all__ = ["CommitDirectives", "StateBundle", "apply_plan"]
