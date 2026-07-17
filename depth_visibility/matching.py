"""Exact deterministic bipartite matching for annotation and Gate-A scoring.

The method contract forbids epsilon perturbations and floating-point Hungarian
tie behaviour.  Candidate weights are first converted from their exact
binary64 values to integers with a common power-of-two denominator.  A dynamic
program then maximizes the integer sum and, among equal-primary solutions,
selects the lexicographically smallest sorted edge list.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from .errors import ContractError, NonFiniteError


Edge = tuple[str, str]
WeightedEdge = tuple[str, str, float]


def binary64_integer_weight(value: float, denominator_exponent: int | None = None) -> tuple[int, int]:
    """Return ``(integer, exponent)`` for exact ``value=integer/2**exponent``.

    If ``denominator_exponent`` is supplied, the integer is scaled to that
    denominator.  Only finite non-negative binary64 values are admissible.
    """

    if isinstance(value, bool):
        raise ContractError("boolean is not a matching weight")
    number = float(value)
    if not math.isfinite(number):
        raise NonFiniteError("matching weight must be finite")
    if number < 0.0:
        raise ContractError("matching weight must be non-negative")
    numerator, denominator = number.as_integer_ratio()
    if denominator & (denominator - 1):  # defensive: binary64 denominators are powers of two
        raise ContractError("binary64 denominator is not a power of two")
    exponent = denominator.bit_length() - 1
    if denominator_exponent is None:
        return numerator, exponent
    if denominator_exponent < exponent:
        raise ContractError("requested common denominator loses exactness")
    return numerator << (denominator_exponent - exponent), denominator_exponent


def _normalize_edges(edges: Iterable[WeightedEdge]) -> tuple[list[str], list[str], dict[Edge, int], int]:
    parsed: list[tuple[str, str, float, int]] = []
    maximum_exponent = 0
    seen: set[Edge] = set()
    for left, right, weight in edges:
        key = (str(left), str(right))
        if key in seen:
            raise ContractError(f"duplicate candidate edge: {key}")
        seen.add(key)
        _, exponent = binary64_integer_weight(weight)
        maximum_exponent = max(maximum_exponent, exponent)
        parsed.append((key[0], key[1], float(weight), exponent))
    integer_weights: dict[Edge, int] = {}
    for left, right, weight, _ in parsed:
        integer_weights[(left, right)] = binary64_integer_weight(weight, maximum_exponent)[0]
    return (
        sorted({left for left, _, _, _ in parsed}),
        sorted({right for _, right, _, _ in parsed}),
        integer_weights,
        maximum_exponent,
    )


def exact_lexicographic_assignment(
    edges: Iterable[WeightedEdge],
    *,
    edge_prefix: Sequence[str] = (),
    maximum_right_nodes: int = 22,
) -> dict[str, Any]:
    """Solve exact maximum-weight bipartite assignment with the frozen tie rule.

    Assignment cardinality is deliberately not a secondary objective.  An edge
    list is compared with ordinary tuple lexicographic order after prefixing
    every edge (for example with scene/window identifiers).  Empty or zero-
    weight alternatives therefore obey the same explicit rule.  The exact
    dynamic program is intentionally bounded; exceeding the bound fails closed
    instead of substituting an approximate or floating solver.
    """

    left_nodes, right_nodes, weights, exponent = _normalize_edges(edges)
    if len(right_nodes) > maximum_right_nodes:
        raise ContractError(
            f"exact assignment has {len(right_nodes)} right nodes; "
            f"limit is {maximum_right_nodes}; review solver scope"
        )
    right_index = {node: index for index, node in enumerate(right_nodes)}
    candidates = {
        left: sorted(
            (right for (candidate_left, right) in weights if candidate_left == left),
            key=str,
        )
        for left in left_nodes
    }
    prefix = tuple(str(item) for item in edge_prefix)

    def better(
        first: tuple[int, tuple[tuple[str, ...], ...]],
        second: tuple[int, tuple[tuple[str, ...], ...]],
    ) -> tuple[int, tuple[tuple[str, ...], ...]]:
        if first[0] != second[0]:
            return first if first[0] > second[0] else second
        return first if first[1] < second[1] else second

    @lru_cache(maxsize=None)
    def solve(left_index: int, used_mask: int) -> tuple[int, tuple[tuple[str, ...], ...]]:
        if left_index == len(left_nodes):
            return (0, ())
        left = left_nodes[left_index]
        best = solve(left_index + 1, used_mask)
        for right in candidates[left]:
            bit = 1 << right_index[right]
            if used_mask & bit:
                continue
            remainder_weight, remainder_edges = solve(left_index + 1, used_mask | bit)
            edge_key = prefix + (left, right)
            proposal = (weights[(left, right)] + remainder_weight, (edge_key,) + remainder_edges)
            best = better(best, proposal)
        return best

    total_integer_weight, prefixed_edges = solve(0, 0)
    edge_width = len(prefix)
    selected = [(edge[-2], edge[-1]) for edge in prefixed_edges]
    return {
        "assignment": selected,
        "edge_keys": [list(edge) for edge in prefixed_edges],
        "integer_weight": total_integer_weight,
        "denominator_exponent": exponent,
        "binary64_weight_sum": total_integer_weight / float(1 << exponent),
        "left_count": len(left_nodes),
        "right_count": len(right_nodes),
        "candidate_count": len(weights),
        "tie_rule": "maximum_exact_binary64_sum_then_lexicographically_smallest_sorted_edges",
        "prefix_width": edge_width,
    }


def _pair_iou(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    key: str,
    right_id_key: str,
    iou_fn: Callable[[Any, Any], float] | None,
) -> float:
    explicit = f"{key}_iou"
    if explicit in left and isinstance(left[explicit], Mapping):
        value = left[explicit].get(str(right.get(right_id_key)))
        if value is not None:
            return float(value)
    if iou_fn is None:
        raise ContractError(f"missing {explicit} and no IoU callback supplied")
    return float(iou_fn(left.get(key), right.get(key)))


def _match_tracks(
    left_tracks: Sequence[Mapping[str, Any]],
    right_tracks: Sequence[Mapping[str, Any]],
    *,
    left_id_key: str,
    right_id_key: str,
    prefix: Sequence[str],
    target_iou_minimum: float,
    source_iou_minimum: float,
    iou_fn: Callable[[Any, Any], float] | None,
) -> dict[str, Any]:
    left_ids = [str(track[left_id_key]) for track in left_tracks]
    right_ids = [str(track[right_id_key]) for track in right_tracks]
    if len(set(left_ids)) != len(left_ids) or len(set(right_ids)) != len(right_ids):
        raise ContractError("track IDs must be unique on each assignment side")
    candidates: list[WeightedEdge] = []
    diagnostics: list[dict[str, Any]] = []
    for left in left_tracks:
        for right in right_tracks:
            target_iou = _pair_iou(left, right, "target", right_id_key, iou_fn)
            source_iou = _pair_iou(left, right, "source", right_id_key, iou_fn)
            if not (math.isfinite(target_iou) and math.isfinite(source_iou)):
                raise NonFiniteError("track IoUs must be finite")
            if not (0.0 <= target_iou <= 1.0 and 0.0 <= source_iou <= 1.0):
                raise ContractError("track IoUs must lie in [0,1]")
            accepted = target_iou >= target_iou_minimum and source_iou >= source_iou_minimum
            weight = float(0.75 * target_iou + 0.25 * source_iou)
            diagnostics.append(
                {
                    "left_track_id": str(left[left_id_key]),
                    "right_track_id": str(right[right_id_key]),
                    "target_iou": target_iou,
                    "source_iou": source_iou,
                    "weight": weight,
                    "accepted": accepted,
                }
            )
            if accepted:
                candidates.append((str(left[left_id_key]), str(right[right_id_key]), weight))
    result = exact_lexicographic_assignment(candidates, edge_prefix=prefix)
    matched_left = {edge[0] for edge in result["assignment"]}
    matched_right = {edge[1] for edge in result["assignment"]}
    result.update(
        {
            "unmatched_left": sorted(set(left_ids) - matched_left),
            "unmatched_right": sorted(set(right_ids) - matched_right),
            "candidate_diagnostics": sorted(
                diagnostics, key=lambda item: (item["left_track_id"], item["right_track_id"])
            ),
        }
    )
    return result


def match_discoveries(
    window_id: str,
    discoveries_a: Sequence[Mapping[str, Any]],
    discoveries_b: Sequence[Mapping[str, Any]],
    *,
    iou_fn: Callable[[Any, Any], float] | None = None,
    target_iou_minimum: float = 0.30,
    source_iou_minimum: float = 0.30,
) -> dict[str, Any]:
    """Match sealed A/B discoveries within one window."""

    return _match_tracks(
        discoveries_a,
        discoveries_b,
        left_id_key="track_id",
        right_id_key="track_id",
        prefix=(str(window_id),),
        target_iou_minimum=target_iou_minimum,
        source_iou_minimum=source_iou_minimum,
        iou_fn=iou_fn,
    )


def match_predictions(
    scene: str,
    window_id: str,
    predictions: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    *,
    iou_fn: Callable[[Any, Any], float] | None = None,
    target_iou_minimum: float = 0.30,
    source_iou_minimum: float = 0.30,
) -> dict[str, Any]:
    """Match sealed predictions to references with exact primary and tie rules."""

    return _match_tracks(
        predictions,
        references,
        left_id_key="predicted_track_id",
        right_id_key="reference_track_id",
        prefix=(str(scene), str(window_id)),
        target_iou_minimum=target_iou_minimum,
        source_iou_minimum=source_iou_minimum,
        iou_fn=iou_fn,
    )
