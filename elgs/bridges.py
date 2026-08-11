"""Anchor intervals, evidence windows, bridge families (spec §3, rev-2 item 7).

Anchor intervals: maximal PLATEAU runs whose capped visible-report
count at binding exceeds a preregistered floor. Windows W(f): maximal
spans BETWEEN consecutive anchor intervals; a family with fewer than
two anchors has NO evidence windows (photometric-only; reported in
coverage). Merged families re-derive windows from the union anchor
set at the next round — MERGE candidates evaluated this round use the
union anchor set to derive windows immediately (spec rev-2 item 3).

Bridge family B(W): evidence-independent constructors with
stop-gradient — bridges are built from committed anchor state only,
never from the reports they will score, and no gradient flows from
bridge poses into model parameters. BRIDGE-LATENT SCOPE (v8 fix 3 as
written out in spec §4): each window's Phi carries ONE tempered
bridge aggregation whose bridge index is shared across every segment
inside that window — per-SEGMENT bridge switching is the rejected
chimera-evidence formulation and is structurally impossible here
because elgs.energy.stream_log_likelihoods scores all of a window's
segments under each single bridge before Phi aggregates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from depth_visibility.errors import ContractError


@dataclass(frozen=True)
class AnchorInterval:
    """One anchor: a plateau run with enough capped visible reports."""

    start_frame: float
    end_frame: float
    capped_visible_reports: int

    def __post_init__(self) -> None:
        if self.end_frame < self.start_frame:
            raise ContractError("anchor interval must have start <= end")
        if self.capped_visible_reports < 0:
            raise ContractError("report count must be non-negative")


@dataclass(frozen=True)
class Window:
    """One evidence window: the span between two consecutive anchors."""

    start_frame: float
    end_frame: float

    def __post_init__(self) -> None:
        if self.end_frame <= self.start_frame:
            raise ContractError("window must have positive span")


def find_anchor_intervals(
    frames: Sequence[float],
    plateau_series: Sequence[bool],
    capped_visible_counts: Sequence[int],
    report_floor: int,
) -> tuple[AnchorInterval, ...]:
    """Maximal plateau runs whose TOTAL capped visible-report count at
    binding exceeds the preregistered floor (strictly)."""
    if not (len(frames) == len(plateau_series) == len(capped_visible_counts)):
        raise ContractError("anchor inputs must have equal length")
    if report_floor < 0:
        raise ContractError("report floor must be non-negative")
    anchors: list[AnchorInterval] = []
    run_start: int | None = None
    total = 0
    for i, (plateau, count) in enumerate(zip(plateau_series, capped_visible_counts)):
        if plateau:
            if run_start is None:
                run_start = i
                total = 0
            total += count
        elif run_start is not None:
            if total > report_floor:
                anchors.append(
                    AnchorInterval(frames[run_start], frames[i - 1], total)
                )
            run_start = None
    if run_start is not None and total > report_floor:
        anchors.append(AnchorInterval(frames[run_start], frames[-1], total))
    return tuple(anchors)


def windows_between_anchors(
    anchors: Sequence[AnchorInterval],
) -> tuple[Window, ...]:
    """W(f): maximal spans between consecutive anchor intervals.

    Fewer than two anchors => NO evidence windows (the family is
    photometric-only for the round; callers report it in coverage).
    """
    if len(anchors) < 2:
        return ()
    ordered = sorted(anchors, key=lambda a: a.start_frame)
    for earlier, later in zip(ordered, ordered[1:]):
        if later.start_frame <= earlier.end_frame:
            raise ContractError("anchor intervals overlap: not maximal runs")
    return tuple(
        Window(earlier.end_frame, later.start_frame)
        for earlier, later in zip(ordered, ordered[1:])
    )


@dataclass(frozen=True)
class CoverageEntry:
    """Per-family evidence coverage for the round report."""

    family_id: int
    n_anchors: int
    n_windows: int
    photometric_only: bool


def coverage_entry(family_id: int, anchors: Sequence[AnchorInterval]) -> CoverageEntry:
    windows = windows_between_anchors(anchors)
    return CoverageEntry(
        family_id=family_id,
        n_anchors=len(anchors),
        n_windows=len(windows),
        photometric_only=len(anchors) < 2,
    )


# A bridge constructor maps (window, committed anchor state) to a
# family of bridge hypotheses, identified by integer bridge ids. The
# constructor family and its parameters are preregistered (category-2)
# and EVIDENCE-INDEPENDENT: nothing derived from the reports the
# bridges will score may enter construction. Implementations must
# detach any tensors they read from the scene model (stop-gradient).
BridgeConstructor = Callable[..., dict]


def validate_bridge_ids(bridge_ids: Sequence[int]) -> tuple[int, ...]:
    """A decision's bridge family: non-empty, unique, sorted ids."""
    if not bridge_ids:
        raise ContractError("a bridge family must contain at least one bridge")
    unique = sorted(set(int(b) for b in bridge_ids))
    if len(unique) != len(bridge_ids):
        raise ContractError("bridge ids must be unique")
    return tuple(unique)


__all__ = [
    "AnchorInterval",
    "BridgeConstructor",
    "CoverageEntry",
    "Window",
    "coverage_entry",
    "find_anchor_intervals",
    "validate_bridge_ids",
    "windows_between_anchors",
]
