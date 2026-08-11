"""Decision classification (spec §8) — fixed-path decomposition.

Per COMMITTED decision, evaluated on its own confirmation samples at
POST-REFIT parameters (fixed path, disclosed):

  (a)              = passes with prior AND ledger terms removed
  render-only flag = passes with tracker AND prior removed
  tracker-only flag= passes with render AND prior removed

  EQUIVALENCE-CLASS     = all data terms below preregistered floors
  PRIOR-PIVOTAL         = fails (a)
  DATA-SUPPORTED        = passes (a) AND >= 1 single-term flag
  INTERACTION-SUPPORTED = passes (a), no single-term flag

Precedence exactly in that order: the equivalence-class test is
decided first (with no data signal the other labels are vacuous),
then (a), then the single-term flags. DATA-SUPPORTED is an
OPERATIONAL label, not statistical support — every printed label
carries the mandatory qualifier. ITT logs every screened candidate
(committed or not); risk-coverage runs over the full event inventory.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from depth_visibility.errors import ContractError

MANDATORY_QUALIFIER = "(fixed-path decision decomposition, not statistical support)"


class DecisionClass(Enum):
    DATA_SUPPORTED = "DATA-SUPPORTED"
    PRIOR_PIVOTAL = "PRIOR-PIVOTAL"
    INTERACTION_SUPPORTED = "INTERACTION-SUPPORTED"
    EQUIVALENCE_CLASS = "EQUIVALENCE-CLASS"


@dataclass(frozen=True)
class DecisionFlags:
    """Outcomes of the fixed-path re-evaluations for one decision.

    Each flag answers: does the acceptance test still pass on the
    decision's confirmation samples at post-refit parameters with the
    named terms removed?
    """

    passes_prior_removed: bool          # test (a): prior + ledger removed
    render_only_flag: bool              # tracker AND prior removed
    tracker_only_flag: bool             # render AND prior removed
    all_data_terms_below_floors: bool   # preregistered floors

    at_post_refit: bool = True

    def __post_init__(self) -> None:
        if not self.at_post_refit:
            raise ContractError(
                "classification is defined at POST-REFIT parameters on the "
                "decision's confirmation samples (spec §8); flags computed "
                "elsewhere are not admissible"
            )


def classify(flags: DecisionFlags) -> DecisionClass:
    """The §8 precedence, exactly."""
    if flags.all_data_terms_below_floors:
        return DecisionClass.EQUIVALENCE_CLASS
    if not flags.passes_prior_removed:
        return DecisionClass.PRIOR_PIVOTAL
    if flags.render_only_flag or flags.tracker_only_flag:
        return DecisionClass.DATA_SUPPORTED
    return DecisionClass.INTERACTION_SUPPORTED


def printable_label(decision_class: DecisionClass) -> str:
    """Every printed label carries the mandatory qualifier."""
    return f"{decision_class.value} {MANDATORY_QUALIFIER}"


@dataclass(frozen=True)
class ITTRecord:
    """Intention-to-treat entry: EVERY screened candidate is logged,
    committed or not (spec §8)."""

    candidate_id: str
    op: str
    family_id: int
    round_index: int
    screened_in: bool
    committed: bool
    rejection_reason: str | None = None
    decision_class: str | None = None

    def __post_init__(self) -> None:
        if self.committed and not self.screened_in:
            raise ContractError("a committed candidate must have been screened in")
        if self.committed and self.decision_class is None:
            raise ContractError("a committed candidate requires a decision class")
        if not self.committed and self.decision_class is not None:
            raise ContractError("only committed decisions carry a decision class")
        if not self.committed and self.rejection_reason is None:
            raise ContractError("an uncommitted candidate requires a rejection reason")


def risk_coverage(records: list[ITTRecord]) -> dict:
    """Coverage over the full candidate inventory for the round report."""
    total = len(records)
    committed = [r for r in records if r.committed]
    by_class: dict[str, int] = {}
    for record in committed:
        assert record.decision_class is not None
        by_class[record.decision_class] = by_class.get(record.decision_class, 0) + 1
    return {
        "total_candidates": total,
        "screened_in": sum(1 for r in records if r.screened_in),
        "committed": len(committed),
        "by_class": dict(sorted(by_class.items())),
        "qualifier": MANDATORY_QUALIFIER,
    }


__all__ = [
    "MANDATORY_QUALIFIER",
    "DecisionClass",
    "DecisionFlags",
    "ITTRecord",
    "classify",
    "printable_label",
    "risk_coverage",
]
