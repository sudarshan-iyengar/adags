"""Structural operations (spec §5 + §1, oracle: prereg transition table).

Every op is PURE here: it validates admissibility, computes the child
interval state THROUGH THE §1 INVERSE MAP, applies latch and tau
inheritance, and emits a TransactionPlan. Mutation happens only in
elgs.transactions (with snapshot/rollback); rows and optimizer
moments are directed, not touched.

K-change target allocation (disclosed ruling, preregistered; derived
from header v8.2 item (1) "allocating the affected spans and
re-normalizing the remaining simplex"): the spans AFFECTED by an op
absorb or release exactly the freed/consumed budget while every
other span keeps its ABSOLUTE value; the child's simplex coordinates
are re-derived from those absolute values against the child's
Omega_free — which IS the renormalization of the remaining simplex.

Latch inheritance is governed by §1 alone: preserved with a preserved
outer endpoint; discarded when the outermost episode on that side is
deleted; cleared by TRUNCATE-shorten of a latched endpoint;
REACTIVATE insertion outside a latched outer endpoint is inadmissible
(zero room); an outer insertion that consumes the whole outer slack
sets that latch (the §1 exact-boundary rule); MERGE takes l_pre/
l_post from the parents owning the extreme endpoints (equivalently,
the Boolean OR — recorded as derived in the table); BIRTH = (0,1);
interior ops never touch latches.

Interior TRUNCATE-delete exists: §5's parenthesis names the terminal
case, but header v8.2 item (4) — untagged, hence normative revision
text — enumerates "interior TRUNCATE-delete removes two adjacent
gaps and creates one" (adjudicated ruling, recorded in the M0 wiki
record and carried by the transition table's spec_tension node).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from depth_visibility.errors import ContractError

from .families import FamilyRecord, FamilyRegistry
from .intervals import (
    IntervalConfig,
    IntervalState,
    birth_interval_state,
    empty_program,
    forward,
    inverse,
)
from .transaction_ledger import LedgerEvent

_EPS = 1e-9


@dataclass(frozen=True)
class TransactionPlan:
    """One structural decision's full mutation directive."""

    op: str
    family_ids: tuple[int, ...]
    child_intervals: dict  # family_id -> IntervalState (or K=0 empty)
    child_tau: dict        # family_id -> tuple of tau origins
    ledger_events: tuple[LedgerEvent, ...]
    moment_resets: tuple[int, ...]  # family ids whose a-logits are rewritten
    binding_redirect: tuple[int, int] | None = None  # (retired, surviving)
    retire_family: int | None = None
    survivor_identity: int | None = None
    detail: dict = field(default_factory=dict)


def _spans(record: FamilyRecord, config: IntervalConfig):
    r = forward(record.interval, config)
    return (
        float(r.slack_pre),
        [float(v) for v in r.lens],
        [float(v) for v in r.gaps],
        float(r.slack_post),
        [float(v) for v in r.b],
        [float(v) for v in r.d],
    )


def _rebuild(
    K: int,
    latch_pre: bool,
    latch_post: bool,
    slack_pre: float,
    lens: list,
    gaps: list,
    slack_post: float,
    config: IntervalConfig,
    dtype: torch.dtype,
) -> IntervalState:
    return inverse(
        K, latch_pre, latch_post, slack_pre, lens, gaps, slack_post,
        config, dtype=dtype,
    )


def plan_birth(
    registry: FamilyRegistry,
    *,
    t_birth: float,
    birth_site: tuple[float, float, float],
    lineage_key: str,
    at_return_site: bool,
    round_index: int,
    iteration: int,
    config: IntervalConfig,
    cap_saturated: bool,
    dtype: torch.dtype = torch.float32,
) -> TransactionPlan:
    """BIRTH (spec §5): K=1, (l_pre, l_post) = (0, 1); ledger chi iff
    at a return site; cap-saturated => INADMISSIBLE this round."""
    if cap_saturated:
        raise ContractError("BIRTH inadmissible this round: cap saturated (logged)")
    interval = birth_interval_state(t_birth, config, dtype=dtype)
    family_id = registry.next_family_id  # assigned at apply time
    events = [LedgerEvent("birth", round_index, iteration, (family_id,))]
    if at_return_site:
        events.append(
            LedgerEvent("return_birth", round_index, iteration, (family_id,))
        )
    return TransactionPlan(
        op="BIRTH",
        family_ids=(family_id,),
        child_intervals={family_id: interval},
        child_tau={family_id: (t_birth,)},
        ledger_events=tuple(events),
        moment_resets=(family_id,),
        detail={
            "t_birth": t_birth,
            "birth_site": birth_site,
            "lineage_key": lineage_key,
        },
    )


def plan_fission(
    registry: FamilyRegistry,
    family_id: int,
    episode_index: int,
    gap_start: float,
    gap_end: float,
    *,
    round_index: int,
    iteration: int,
    config: IntervalConfig,
    dtype: torch.dtype = torch.float32,
) -> TransactionPlan:
    """FISSION(f, k, (g0, g1]): one episode -> two (Delta K = +1);
    children (b_k, g0] and (g1, d_k]; interior op — latches untouched;
    both children inherit the parent's tau (exact coefficient copy)."""
    record = registry.require_active(family_id)
    if record.interval.K >= 4:
        raise ContractError("FISSION inadmissible: K_f = 4")
    if record.interval.K == 0:
        raise ContractError("FISSION on an empty program")
    slack_pre, lens, gaps, slack_post, b, d = _spans(record, config)
    k = episode_index
    if not (0 <= k < record.interval.K):
        raise ContractError(f"episode index {k} out of range")
    # gap ⊆ plateau with margins: (g0, g1] strictly inside the plateau.
    plateau_lo, plateau_hi = b[k] + config.w, d[k] - config.w
    if not (plateau_lo < gap_start < gap_end < plateau_hi):
        raise ContractError(
            "FISSION gap must lie strictly inside the episode plateau"
        )
    len_left = gap_start - b[k]
    len_right = d[k] - gap_end
    new_gap = gap_end - gap_start
    if len_left <= config.floor_len or len_right <= config.floor_len:
        raise ContractError("FISSION children would violate the length floor")
    if new_gap <= config.floor_gap:
        raise ContractError("FISSION gap would violate the gap floor")
    config.require_admissible_k(record.interval.K + 1)
    new_lens = lens[:k] + [len_left, len_right] + lens[k + 1:]
    new_gaps = gaps[:k] + [new_gap] + gaps[k:]
    child = _rebuild(
        record.interval.K + 1,
        bool(record.interval.latch_pre),
        bool(record.interval.latch_post),
        slack_pre, new_lens, new_gaps, slack_post, config, dtype,
    )
    tau = record.tau
    child_tau = tau[:k] + (tau[k], tau[k]) + tau[k + 1:]
    return TransactionPlan(
        op="FISSION",
        family_ids=(family_id,),
        child_intervals={family_id: child},
        child_tau={family_id: child_tau},
        ledger_events=(
            LedgerEvent("fission", round_index, iteration, (family_id,)),
        ),
        moment_resets=(family_id,),
        detail={"episode_index": k, "gap": (gap_start, gap_end)},
    )


def plan_truncate_shorten(
    registry: FamilyRegistry,
    family_id: int,
    episode_index: int,
    side: str,
    new_edge: float,
    *,
    round_index: int,
    iteration: int,
    config: IntervalConfig,
    dtype: torch.dtype = torch.float32,
) -> TransactionPlan:
    """TRUNCATE-shorten (Delta K = 0): one edge moves inward; the
    adjacent span absorbs the freed budget; shortening a latched outer
    endpoint CLEARS that latch (§1)."""
    record = registry.require_active(family_id)
    if record.interval.K == 0:
        raise ContractError("TRUNCATE on an empty program")
    if side not in ("start", "end"):
        raise ContractError("side must be 'start' or 'end'")
    slack_pre, lens, gaps, slack_post, b, d = _spans(record, config)
    k = episode_index
    if not (0 <= k < record.interval.K):
        raise ContractError(f"episode index {k} out of range")
    latch_pre = bool(record.interval.latch_pre)
    latch_post = bool(record.interval.latch_post)
    if side == "start":
        if not (b[k] < new_edge < d[k]):
            raise ContractError("new start edge must move inward")
        freed = new_edge - b[k]
        new_len = lens[k] - freed
        if new_len <= config.floor_len:
            raise ContractError("shortened episode would violate the length floor")
        lens = lens[:k] + [new_len] + lens[k + 1:]
        if k == 0:
            slack_pre += freed
            if latch_pre:
                latch_pre = False  # cleared by shortening a latched endpoint
        else:
            gaps = gaps[: k - 1] + [gaps[k - 1] + freed] + gaps[k:]
    else:
        if not (b[k] < new_edge < d[k]):
            raise ContractError("new end edge must move inward")
        freed = d[k] - new_edge
        new_len = lens[k] - freed
        if new_len <= config.floor_len:
            raise ContractError("shortened episode would violate the length floor")
        lens = lens[:k] + [new_len] + lens[k + 1:]
        if k == record.interval.K - 1:
            slack_post += freed
            if latch_post:
                latch_post = False
        else:
            gaps = gaps[:k] + [gaps[k] + freed] + gaps[k + 1:]
    child = _rebuild(
        record.interval.K, latch_pre, latch_post,
        slack_pre, lens, gaps, slack_post, config, dtype,
    )
    return TransactionPlan(
        op="TRUNCATE_SHORTEN",
        family_ids=(family_id,),
        child_intervals={family_id: child},
        child_tau={family_id: record.tau},
        ledger_events=(
            LedgerEvent("truncate", round_index, iteration, (family_id,),
                        {"mode": "shorten", "side": side, "episode": k}),
        ),
        moment_resets=(family_id,),
    )


def plan_truncate_delete(
    registry: FamilyRegistry,
    family_id: int,
    episode_index: int,
    *,
    round_index: int,
    iteration: int,
    config: IntervalConfig,
    dtype: torch.dtype = torch.float32,
) -> TransactionPlan:
    """TRUNCATE-delete (Delta K = -1; kappa auto-refunded as a state
    term). Terminal: the episode and its single adjacency are absorbed
    into that side's outer slack, DISCARDING that side's latch.
    Interior (header v8.2 item 4): removes two adjacent gaps, creates
    one spanning the freed range. K_f = 1 -> the K=0 empty program."""
    record = registry.require_active(family_id)
    K = record.interval.K
    if K == 0:
        raise ContractError("TRUNCATE-delete on an empty program")
    slack_pre, lens, gaps, slack_post, b, d = _spans(record, config)
    k = episode_index
    if not (0 <= k < K):
        raise ContractError(f"episode index {k} out of range")
    latch_pre = bool(record.interval.latch_pre)
    latch_post = bool(record.interval.latch_post)
    if K == 1:
        child: IntervalState = empty_program()
        new_tau: tuple = ()
    elif k == 0:
        # terminal delete on the pre side: slack_pre + len_0 + gap_0
        # flow into the new outer slack; the pre latch is discarded.
        new_slack_pre = slack_pre + lens[0] + gaps[0]
        child = _rebuild(
            K - 1, False, latch_post,
            new_slack_pre, lens[1:], gaps[1:], slack_post, config, dtype,
        )
        new_tau = record.tau[1:]
        latch_pre = False
    elif k == K - 1:
        new_slack_post = slack_post + lens[-1] + gaps[-1]
        child = _rebuild(
            K - 1, latch_pre, False,
            slack_pre, lens[:-1], gaps[:-1], new_slack_post, config, dtype,
        )
        new_tau = record.tau[:-1]
        latch_post = False
    else:
        merged_gap = gaps[k - 1] + lens[k] + gaps[k]
        child = _rebuild(
            K - 1, latch_pre, latch_post,
            slack_pre,
            lens[:k] + lens[k + 1:],
            gaps[: k - 1] + [merged_gap] + gaps[k + 1:],
            slack_post, config, dtype,
        )
        new_tau = record.tau[:k] + record.tau[k + 1:]
    return TransactionPlan(
        op="TRUNCATE_DELETE",
        family_ids=(family_id,),
        child_intervals={family_id: child},
        child_tau={family_id: new_tau},
        ledger_events=(
            LedgerEvent("truncate", round_index, iteration, (family_id,),
                        {"mode": "delete", "episode": k}),
        ),
        moment_resets=(family_id,),
    )


def return_family_candidates(
    registry: FamilyRegistry,
    site: tuple[float, float, float],
    window: tuple[float, float],
    r_site: float,
) -> tuple[int, ...]:
    """RETURN-FAMILY PREDICATE (deterministic, spec §5): families born
    within radius r_site of the site with birth time inside W; ties ->
    earliest birth, then lowest family ID."""
    if r_site <= 0:
        raise ContractError("r_site must be positive")
    hits = []
    for family_id in registry.active_ids():
        record = registry.get(family_id)
        dist = sum((a - b) ** 2 for a, b in zip(record.birth_site, site)) ** 0.5
        if dist <= r_site and window[0] <= record.birth_time <= window[1]:
            hits.append((record.birth_time, record.family_id))
    return tuple(fid for _, fid in sorted(hits))


def plan_reactivate(
    registry: FamilyRegistry,
    family_id: int,
    insert_start: float,
    insert_end: float,
    tau_fresh: float,
    *,
    return_candidates: tuple[int, ...],
    round_index: int,
    iteration: int,
    config: IntervalConfig,
    dtype: torch.dtype = torch.float32,
) -> TransactionPlan:
    """REACTIVATE(f, e): pre = NO return family at the site AND
    K_f < 4 AND floors fit. Inserts one episode into a gap or an
    unlatched outer slack; insertion outside a latched outer endpoint
    is inadmissible (zero room); consuming an outer slack exactly to
    zero sets that latch (§1 exact-boundary rule). Fresh tau."""
    if return_candidates:
        raise ContractError(
            "REACTIVATE inadmissible: a return family exists at the site "
            "(REACTIVATE/MERGE mutual exclusivity)"
        )
    record = registry.require_active(family_id)
    K = record.interval.K
    if K >= 4:
        raise ContractError("REACTIVATE inadmissible: K_f = 4")
    if K == 0:
        raise ContractError("REACTIVATE on an empty program")
    config.require_admissible_k(K + 1)
    if insert_end <= insert_start:
        raise ContractError("insertion span must be positive")
    new_len = insert_end - insert_start
    if new_len <= config.floor_len:
        raise ContractError("REACTIVATE episode would violate the length floor")
    slack_pre, lens, gaps, slack_post, b, d = _spans(record, config)
    latch_pre = bool(record.interval.latch_pre)
    latch_post = bool(record.interval.latch_post)
    lo_bound = -config.w_m
    hi_bound = config.T + config.w_m

    if insert_end < b[0] - _EPS or abs(insert_end - b[0]) <= _EPS:
        # before the first episode, inside the pre slack
        if latch_pre:
            raise ContractError(
                "REACTIVATE outside a latched outer endpoint is inadmissible"
            )
        new_slack_pre = insert_start - lo_bound
        trailing_gap = b[0] - insert_end
        if trailing_gap <= config.floor_gap:
            raise ContractError("REACTIVATE would violate the gap floor")
        if new_slack_pre < -_EPS:
            raise ContractError("insertion extends beyond the margin")
        set_latch = abs(new_slack_pre) <= _EPS
        child = _rebuild(
            K + 1, set_latch or latch_pre, latch_post,
            0.0 if set_latch else new_slack_pre,
            [new_len] + lens, [trailing_gap] + gaps, slack_post,
            config, dtype,
        )
        new_tau = (tau_fresh,) + record.tau
        position = 0
    elif insert_start > d[-1] + _EPS or abs(insert_start - d[-1]) <= _EPS:
        if latch_post:
            raise ContractError(
                "REACTIVATE outside a latched outer endpoint is inadmissible"
            )
        leading_gap = insert_start - d[-1]
        new_slack_post = hi_bound - insert_end
        if leading_gap <= config.floor_gap:
            raise ContractError("REACTIVATE would violate the gap floor")
        if new_slack_post < -_EPS:
            raise ContractError("insertion extends beyond the margin")
        set_latch = abs(new_slack_post) <= _EPS
        child = _rebuild(
            K + 1, latch_pre, set_latch or latch_post,
            slack_pre, lens + [new_len], gaps + [leading_gap],
            0.0 if set_latch else new_slack_post,
            config, dtype,
        )
        new_tau = record.tau + (tau_fresh,)
        position = K
    else:
        # interior: locate the containing gap
        position = None
        for g in range(K - 1):
            if d[g] < insert_start and insert_end < b[g + 1]:
                position = g + 1
                break
        if position is None:
            raise ContractError(
                "REACTIVATE insertion must lie strictly inside one gap"
            )
        g = position - 1
        left_gap = insert_start - d[g]
        right_gap = b[g + 1] - insert_end
        if left_gap <= config.floor_gap or right_gap <= config.floor_gap:
            raise ContractError("REACTIVATE would violate the gap floor")
        child = _rebuild(
            K + 1, latch_pre, latch_post,
            slack_pre,
            lens[:position] + [new_len] + lens[position:],
            gaps[:g] + [left_gap, right_gap] + gaps[g + 1:],
            slack_post, config, dtype,
        )
        new_tau = record.tau[:position] + (tau_fresh,) + record.tau[position:]
    return TransactionPlan(
        op="REACTIVATE",
        family_ids=(family_id,),
        child_intervals={family_id: child},
        child_tau={family_id: new_tau},
        ledger_events=(
            LedgerEvent("reactivate", round_index, iteration, (family_id,),
                        {"position": position}),
        ),
        moment_resets=(family_id,),
        detail={"insert": (insert_start, insert_end)},
    )


def plan_merge(
    registry: FamilyRegistry,
    family_a: int,
    family_b: int,
    *,
    return_predicate_holds: bool,
    round_index: int,
    iteration: int,
    config: IntervalConfig,
    dtype: torch.dtype = torch.float32,
) -> TransactionPlan:
    """MERGE(f_old, f_new) (spec §5 + rev-3 A3): pre = return-family
    predicate holds; episode unions disjoint with floors; K sum <= 4.
    Survivor = older parent (identity retained); episodes concatenated
    with absolute values kept; gap terms recomputed on the
    concatenated ordered sequence; l_pre/l_post from the parents
    owning the extreme endpoints (== Boolean OR, derived); radiance
    from the OLDER family (directive detail); ledger +mu; younger ID
    retired, bindings redirected."""
    if not return_predicate_holds:
        raise ContractError("MERGE requires the return-family predicate")
    survivor_id, retired_id = registry.merge_survivor_of(family_a, family_b)
    survivor = registry.get(survivor_id)
    retired = registry.get(retired_id)
    K_total = survivor.interval.K + retired.interval.K
    if K_total > 4:
        raise ContractError("MERGE inadmissible: K_old + K_new > 4")
    if survivor.interval.K == 0 or retired.interval.K == 0:
        raise ContractError("MERGE with an empty program")
    config.require_admissible_k(K_total)
    sp_a, lens_a, gaps_a, so_a, b_a, d_a = _spans(survivor, config)
    sp_b, lens_b, gaps_b, so_b, b_b, d_b = _spans(retired, config)
    episodes = sorted(
        [(b, l, tau, "s") for b, l, tau in zip(b_a, lens_a, survivor.tau)]
        + [(b, l, tau, "r") for b, l, tau in zip(b_b, lens_b, retired.tau)]
    )
    # disjointness with floors
    merged_lens = [e[1] for e in episodes]
    merged_tau = tuple(e[2] for e in episodes)
    merged_b = [e[0] for e in episodes]
    merged_d = [b + l for b, l in zip(merged_b, merged_lens)]
    new_gaps = []
    for i in range(len(episodes) - 1):
        gap = merged_b[i + 1] - merged_d[i]
        if gap <= config.floor_gap:
            raise ContractError("MERGE episode unions not disjoint with floors")
        new_gaps.append(gap)
    latch_pre = (bool(survivor.interval.latch_pre) or bool(retired.interval.latch_pre))
    latch_post = (bool(survivor.interval.latch_post) or bool(retired.interval.latch_post))
    slack_pre = 0.0 if latch_pre else merged_b[0] - (-config.w_m)
    slack_post = 0.0 if latch_post else (config.T + config.w_m) - merged_d[-1]
    child = _rebuild(
        K_total, latch_pre, latch_post,
        slack_pre, merged_lens, new_gaps, slack_post, config, dtype,
    )
    return TransactionPlan(
        op="MERGE",
        family_ids=(survivor_id, retired_id),
        child_intervals={survivor_id: child, retired_id: empty_program()},
        child_tau={survivor_id: merged_tau, retired_id: ()},
        ledger_events=(
            LedgerEvent("merge", round_index, iteration, (survivor_id, retired_id)),
        ),
        moment_resets=(survivor_id,),
        binding_redirect=(retired_id, survivor_id),
        retire_family=retired_id,
        survivor_identity=survivor_id,
        detail={"radiance_from": survivor_id},
    )


def plan_prune_episode(
    registry: FamilyRegistry,
    family_id: int,
    episode_index: int,
    *,
    len_at_floor: bool,
    micro_render_confirms: bool,
    round_index: int,
    iteration: int,
    config: IntervalConfig,
    dtype: torch.dtype = torch.float32,
) -> TransactionPlan:
    """PRUNE-episode: len <= floor + delta_tol AND micro-render
    confirmation required; interval mechanics equal TRUNCATE-delete at
    the position (latches per §1's general delete rule)."""
    if not (len_at_floor and micro_render_confirms):
        raise ContractError(
            "PRUNE-episode requires len at floor AND micro-render confirmation"
        )
    plan = plan_truncate_delete(
        registry, family_id, episode_index,
        round_index=round_index, iteration=iteration, config=config, dtype=dtype,
    )
    return TransactionPlan(
        op="PRUNE_EPISODE",
        family_ids=plan.family_ids,
        child_intervals=plan.child_intervals,
        child_tau=plan.child_tau,
        ledger_events=(
            LedgerEvent("prune", round_index, iteration, (family_id,),
                        {"scope": "episode", "episode": episode_index}),
        ),
        moment_resets=plan.moment_resets,
    )


def plan_prune_family(
    registry: FamilyRegistry,
    family_id: int,
    *,
    episodeless_or_unsupported: bool,
    round_index: int,
    iteration: int,
) -> TransactionPlan:
    """PRUNE-family: episodeless (K=0) or lifetime-unsupported; the
    family is retired; its clusters become inactive (recorded)."""
    if not episodeless_or_unsupported:
        raise ContractError(
            "PRUNE-family requires an episodeless or lifetime-unsupported family"
        )
    registry.require_active(family_id)
    return TransactionPlan(
        op="PRUNE_FAMILY",
        family_ids=(family_id,),
        child_intervals={family_id: empty_program()},
        child_tau={family_id: ()},
        ledger_events=(
            LedgerEvent("prune", round_index, iteration, (family_id,),
                        {"scope": "family"}),
        ),
        moment_resets=(),
        retire_family=family_id,
    )


__all__ = [
    "TransactionPlan",
    "plan_birth",
    "plan_fission",
    "plan_merge",
    "plan_prune_episode",
    "plan_prune_family",
    "plan_reactivate",
    "plan_truncate_delete",
    "plan_truncate_shorten",
    "return_family_candidates",
]
