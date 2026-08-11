"""Structural-round driver: plans -> ordering -> acceptance -> commit.

Chains the already-tested pieces for one Gauss-Seidel pass at a round
boundary (method schedule): candidate TransactionPlans (proposed
upstream; PURE) are wrapped as conflict candidates, ordered and
capped by elgs.search.plan_pass, evaluated one component at a time
through the injected paired sample builder (the trainer renders the
candidate counterfactually via presence overrides under CRN; tests
inject synthetic samples), decided by elgs.acceptance.decide
(SNIS + paired bootstrap + degeneracy rejection, transaction
increment included), and committed atomically via
elgs.transactions.apply_plan — with EVERY outcome ITT-logged and the
search-cost ledger updated. A rejected candidate discards its plan
(plans are pure, nothing to roll back); acceptance-machinery
rejections (degeneracy, slot exhaustion, undefined SE) are recorded
rejections, never crashes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from depth_visibility.errors import ContractError

from .acceptance import (
    AcceptanceRecord,
    FrozenSamplerParams,
    SlotExhausted,
    SnisSample,
    crn_seed,
    decide,
)
from .classification import ITTRecord
from .ops import TransactionPlan
from .runtime import ElgsRuntime
from .search import Candidate, plan_pass
from .transactions import StateBundle, apply_plan

# (plan, crn_seed, slot_rank) -> paired samples. The rank is the
# conflict component's deterministic slot-grid rank, so builders that
# draw from the pre-partitioned reserved pool consume exactly their
# slot (spec §7).
SampleBuilder = Callable[[TransactionPlan, int, int], Sequence[SnisSample]]


@dataclass(frozen=True)
class ProposedCandidate:
    """A pure plan plus its screening metadata for conflict ordering."""

    plan: TransactionPlan
    screen_score: float
    footprint_frames: tuple[float, float]
    candidate_id: str | None = None

    def as_search_candidate(self) -> Candidate:
        if self.candidate_id is not None:
            candidate_id = self.candidate_id
        else:
            # Deterministic disambiguation: two proposals on the same
            # family/op differ in their plan detail.
            import hashlib

            detail_key = hashlib.sha256(
                repr(sorted(self.plan.detail.items())).encode("utf-8")
            ).hexdigest()[:8]
            candidate_id = (
                f"{self.plan.op}:{'-'.join(map(str, self.plan.family_ids))}:{detail_key}"
            )
        return Candidate(
            candidate_id=candidate_id,
            op=self.plan.op,
            family_ids=self.plan.family_ids,
            screen_score=self.screen_score,
            footprint_frames=self.footprint_frames,
            footprint_families=self.plan.family_ids,
        )


@dataclass
class RoundOutcome:
    committed: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    itt: list = field(default_factory=list)
    acceptance_records: dict = field(default_factory=dict)


def run_pass(
    proposals: Sequence[ProposedCandidate],
    runtime: ElgsRuntime,
    bundle: StateBundle,
    sample_builder: SampleBuilder,
    sampler_params: FrozenSamplerParams,
    *,
    k_se: float,
    base_seed: int,
    round_index: int,
    pass_index: int,
    iteration: int,
    candidate_cap: int,
    exact_deltas_fn: Callable[[TransactionPlan], float] | None = None,
    transaction_increment_fn: Callable[[TransactionPlan], float] | None = None,
) -> RoundOutcome:
    """One confirmation pass. Deterministic under base_seed."""
    sampler_params.require_frozen()
    by_id = {p.as_search_candidate().candidate_id: p for p in proposals}
    if len(by_id) != len(proposals):
        raise ContractError("duplicate candidate ids in one pass")
    pass_plan = plan_pass(
        [p.as_search_candidate() for p in proposals],
        round_index=round_index,
        pass_index=pass_index,
        candidate_cap=candidate_cap,
    )
    outcome = RoundOutcome(itt=list(pass_plan.itt_records))
    for rank, confirmation in zip(pass_plan.component_ranks, pass_plan.confirmations):
        proposal = by_id[confirmation.candidate_id]
        plan = proposal.plan
        seed = crn_seed(base_seed, round_index, pass_index, rank)
        try:
            samples = sample_builder(plan, seed, rank)
            record: AcceptanceRecord = decide(
                samples,
                sampler_params,
                exact_deltas=(exact_deltas_fn(plan) if exact_deltas_fn else 0.0),
                transaction_increment=(
                    transaction_increment_fn(plan) if transaction_increment_fn else 0.0
                ),
                k=k_se,
                bootstrap_seed=seed,
            )
        except (ContractError, SlotExhausted) as reject_reason:
            outcome.rejected.append(confirmation.candidate_id)
            outcome.itt.append(
                ITTRecord(
                    candidate_id=confirmation.candidate_id,
                    op=plan.op,
                    family_id=plan.family_ids[0],
                    round_index=round_index,
                    screened_in=True,
                    committed=False,
                    rejection_reason=f"acceptance_machinery: {reject_reason}",
                )
            )
            bundle.search_cost.candidates_tried += 1
            continue
        bundle.search_cost.candidates_tried += 1
        outcome.acceptance_records[confirmation.candidate_id] = record
        if record.accepted:
            apply_plan(bundle, plan)
            for family_id in plan.moment_resets:
                runtime.sync_family_from_registry(family_id)
            if plan.retire_family is not None:
                runtime.sync_family_from_registry(plan.retire_family)
            bundle.search_cost.candidates_accepted += 1
            outcome.committed.append(confirmation.candidate_id)
            outcome.itt.append(
                ITTRecord(
                    candidate_id=confirmation.candidate_id,
                    op=plan.op,
                    family_id=plan.family_ids[0],
                    round_index=round_index,
                    screened_in=True,
                    committed=True,
                    decision_class="PENDING-POST-REFIT",
                )
            )
        else:
            outcome.rejected.append(confirmation.candidate_id)
            outcome.itt.append(
                ITTRecord(
                    candidate_id=confirmation.candidate_id,
                    op=plan.op,
                    family_id=plan.family_ids[0],
                    round_index=round_index,
                    screened_in=True,
                    committed=False,
                    rejection_reason=(
                        f"acceptance: delta_total={record.delta_total:.6g} "
                        f"+ k*SE={record.k * record.se:.6g} >= 0"
                    ),
                )
            )
    return outcome


__all__ = ["ProposedCandidate", "RoundOutcome", "run_pass"]
