"""ElgsRuntime: the trainer-facing glue (CPU-testable core).

GaussianModel and the renderer call THIN methods here so every piece
of EL-GS logic stays importable and testable without CUDA:

- a-logit storage as PER-FAMILY leaf tensors of exactly dim
  2K+1-n_lat (no padding, no masking — exact dimensions by
  construction; a structural op replaces the family's tensor through
  the §1 inverse map and its optimizer moments are reset);
- the per-row presence multiplier pi_winner(t) gathered through the
  per-row family-id column — differentiable w.r.t. the a-logits (the
  only path from L_render into interval endpoints);
- structural-op commit glue: registry/binding/ledger bundle commits
  via elgs.transactions, a-logit row rewrites through the §1 inverse
  map, and per-family moment-reset directives for the optimizer
  (rewritten rows get zeroed Adam moments — logged);
- round bookkeeping (seeding, audit, rounds, refit end) validated
  against the preregistered schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from depth_visibility.errors import ContractError

from .families import FamilyRegistry
from .intervals import IntervalConfig, IntervalRealization, IntervalState, forward
from .presence import episode_presence


@dataclass
class ScheduleAnchors:
    """Category-1 schedule (loaded from prereg, never from lane YAML)."""

    seed_iteration: int
    audit_iteration: int
    round_iterations: tuple[int, ...]
    refit_until: int

    def __post_init__(self) -> None:
        ordered = (
            [self.seed_iteration, self.audit_iteration]
            + list(self.round_iterations)
            + [self.refit_until]
        )
        if any(b <= a for a, b in zip(ordered, ordered[1:])):
            raise ContractError(
                "schedule anchors must be strictly increasing: "
                "seed < audit < rounds... < refit_until"
            )


class ElgsRuntime:
    """Owns the per-family exact-dimension a-logits and presence."""

    def __init__(
        self,
        registry: FamilyRegistry,
        config: IntervalConfig,
        schedule: ScheduleAnchors,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.registry = registry
        self.config = config
        self.schedule = schedule
        self.device = torch.device(device)
        self.dtype = dtype
        self._logits: dict[int, torch.Tensor] = {}
        self._sync_all_from_registry()

    # -- a-logit storage --------------------------------------------------

    def _sync_all_from_registry(self) -> None:
        for family_id in self.registry.active_ids():
            self.sync_family_from_registry(family_id)

    def sync_family_from_registry(self, family_id: int) -> None:
        """(Re)write a family's logit row from its registry interval —
        the structural-op commit path. The caller must reset this
        family's optimizer moments (directive; logged)."""
        record = self.registry.get(family_id)
        if record.retired or record.interval.K == 0:
            self._logits.pop(family_id, None)
            return
        assert record.interval.a is not None
        self._logits[family_id] = (
            record.interval.a.detach().clone().to(self.device, self.dtype).requires_grad_(True)
        )

    def logit_parameters(self) -> dict[int, torch.Tensor]:
        """family_id -> leaf tensor (the trainer registers these in
        the optimizer's `elgs_a` group)."""
        return dict(self._logits)

    def interval_state(self, family_id: int) -> IntervalState:
        """The LIVE state: registry (K, latches) + current logits."""
        record = self.registry.get(family_id)
        if family_id not in self._logits:
            return record.interval
        return IntervalState(
            K=record.interval.K,
            latch_pre=record.interval.latch_pre,
            latch_post=record.interval.latch_post,
            a=self._logits[family_id],
        )

    def realization(self, family_id: int) -> IntervalRealization:
        return forward(self.interval_state(family_id), self.config)

    # -- presence ---------------------------------------------------------

    def presence_multiplier(
        self,
        family_ids: torch.Tensor,
        timestamp: float,
        overrides: dict | None = None,
    ) -> torch.Tensor:
        """(N, 1) pi_winner(timestamp) per row, gathered by family.

        Exact zero for rows of retired/empty families, rows whose
        family has no active episode at t, and unassigned rows (-1);
        differentiable w.r.t. the a-logits of families active at t.
        `overrides` maps family_id -> IntervalState for COUNTERFACTUAL
        candidate evaluation (frozen-functional snapshot: the live
        state is never mutated by scoring a candidate).
        """
        if family_ids.dim() != 1:
            raise ContractError("family_ids must be a 1-D per-row column")
        per_family: dict[int, torch.Tensor] = {}
        zero = torch.zeros((), device=self.device, dtype=self.dtype)
        for family_id in torch.unique(family_ids).tolist():
            family_id = int(family_id)
            if family_id < 0:
                per_family[family_id] = zero
                continue
            if overrides is not None and family_id in overrides:
                candidate = overrides[family_id]
                if candidate.K == 0:
                    per_family[family_id] = zero
                    continue
                r = forward(candidate, self.config)
            else:
                record = self.registry.get(family_id)
                if record.retired or record.interval.K == 0:
                    per_family[family_id] = zero
                    continue
                r = self.realization(family_id)
            # Differentiable winner: episodes are disjoint, so at most
            # one episode presence is nonzero at t — the SUM equals
            # the winner's value while keeping the graph intact.
            t = torch.as_tensor(timestamp, device=self.device, dtype=self.dtype)
            values = episode_presence(t, r.b, r.d, self.config.w)
            if int((values > 0).sum()) > 1:
                raise ContractError(
                    f"family {family_id}: multiple active episodes at "
                    f"t={timestamp} — intervals not disjoint"
                )
            per_family[family_id] = values.sum()
        rows = torch.stack(
            [per_family[int(f)] for f in family_ids.tolist()]
        ).reshape(-1, 1)
        return rows

    def flush_to_registry(self) -> None:
        """Write the LIVE logits back into the registry records
        (checkpoint path: capture serializes the registry, so the
        current optimizer state of `a` must be reflected there)."""
        for family_id, logits in self._logits.items():
            record = self.registry.get(family_id)
            self.registry.replace_interval(
                family_id,
                IntervalState(
                    K=record.interval.K,
                    latch_pre=record.interval.latch_pre,
                    latch_post=record.interval.latch_post,
                    a=logits.detach().clone(),
                ),
                record.tau,
            )

    # -- schedule ---------------------------------------------------------

    def phase_at(self, iteration: int) -> str:
        s = self.schedule
        if iteration < s.seed_iteration:
            return "warmup"
        if iteration < s.audit_iteration:
            return "seeded"
        if iteration <= max(s.round_iterations):
            return "rounds"
        if iteration <= s.refit_until:
            return "refit"
        return "post_refit"

    def is_round_boundary(self, iteration: int) -> bool:
        return iteration in self.schedule.round_iterations


__all__ = ["ElgsRuntime", "ScheduleAnchors"]
