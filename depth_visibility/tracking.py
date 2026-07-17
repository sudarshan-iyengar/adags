"""Deterministic temporal association and visibility-state transitions."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .canonical import domain_id


def patch_pair_terms(
    reprojection_error: float,
    cycle_error: float,
    appearance_error: float,
    consensus_3d_error: float,
    *,
    r_scene: float,
) -> dict[str, float]:
    if not math.isfinite(r_scene) or r_scene <= 0:
        raise ValueError("R_scene must be positive and finite")
    raw = {
        "reprojection": float(reprojection_error) / 2.0,
        "cycle": float(cycle_error) / 1.5,
        "appearance": float(appearance_error) / 0.40,
        "consensus_3d": float(consensus_3d_error) / (0.02 * r_scene),
    }
    if not all(math.isfinite(value) and value >= 0 for value in raw.values()):
        raise ValueError("temporal errors must be finite and nonnegative")
    capped_cost = {key: min(2.0, value) for key, value in raw.items()}
    cost = (
        0.4 * capped_cost["reprojection"]
        + 0.3 * capped_cost["cycle"]
        + 0.2 * capped_cost["appearance"]
        + 0.1 * capped_cost["consensus_3d"]
    )
    risk = max(min(1.0, value) for value in raw.values())
    return {**raw, "cost": cost, "risk": risk}


def identity_risk(best_cost: float, second_cost: float | None) -> float:
    best = float(best_cost)
    if not math.isfinite(best) or best < 0:
        raise ValueError("best identity cost must be finite and nonnegative")
    if second_cost is None:
        return 0.0
    second = float(second_cost)
    if not math.isfinite(second) or second < best:
        raise ValueError("second identity cost must be finite and no lower than best")
    ratio = math.inf if best == 0 and second > 0 else (1.0 if best == second == 0 else second / best)
    return min(1.0, 1.2 / ratio)


def _best_and_ratio(
    candidates: list[Mapping[str, Any]], counterpart_key: str
) -> tuple[Mapping[str, Any] | None, float]:
    if not candidates:
        return None, 0.0
    ordered = sorted(candidates, key=lambda item: (float(item["cost"]), str(item[counterpart_key])))
    costs = [float(item["cost"]) for item in ordered]
    if any(not math.isfinite(cost) or cost < 0 for cost in costs):
        raise ValueError("temporal candidate costs must be finite and nonnegative")
    best = ordered[0]
    if len(ordered) == 1:
        return best, math.inf
    first, second = costs[0], costs[1]
    ratio = math.inf if first == 0 and second > 0 else (1.0 if first == second == 0 else second / first)
    return best, ratio


def _validated_patch_candidates(
    candidates: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    required = {
        "source_patch_id",
        "destination_patch_id",
        "cost",
        "risk",
        "flow_manifests",
        "match_tuple",
        "r_scene",
        "centroid_distance",
        "rgb_l2",
        "camera_node_match_counts",
        "valid_flow_cameras",
    }
    eligible: list[Mapping[str, Any]] = []
    for item in candidates:
        missing = required - set(item)
        if missing:
            raise ValueError(f"temporal candidate is missing {sorted(missing)}")
        r_scene = float(item["r_scene"])
        centroid_distance = float(item["centroid_distance"])
        rgb_l2 = float(item["rgb_l2"])
        if (
            not all(math.isfinite(value) for value in (r_scene, centroid_distance, rgb_l2))
            or r_scene <= 0
            or centroid_distance < 0
            or rgb_l2 < 0
        ):
            raise ValueError("temporal candidate geometry/appearance terms violate their domain")
        counts = item["camera_node_match_counts"]
        valid_cameras = {str(value) for value in item["valid_flow_cameras"]}
        if not isinstance(counts, Mapping) or not isinstance(item["valid_flow_cameras"], (list, tuple, set)):
            raise ValueError("temporal camera support must be explicit")
        supported = {
            str(camera)
            for camera, count in counts.items()
            if int(count) >= 3 and str(camera) in valid_cameras
        }
        if (
            centroid_distance <= 0.05 * r_scene
            and rgb_l2 <= 0.20
            and len(supported) >= 2
        ):
            eligible.append(item)
    return eligible


def reciprocal_patch_edges(candidates: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items = _validated_patch_candidates(candidates)
    forward: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    backward: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        required = {
            "source_patch_id",
            "destination_patch_id",
            "cost",
            "risk",
            "flow_manifests",
            "match_tuple",
        }
        if not required.issubset(item):
            raise ValueError(f"temporal candidate is missing {sorted(required - set(item))}")
        if not 0 <= float(item["risk"]) <= 1 or not math.isfinite(float(item["risk"])):
            raise ValueError("temporal candidate risk must be finite in [0,1]")
        forward[str(item["source_patch_id"])].append(item)
        backward[str(item["destination_patch_id"])].append(item)
    forward_best = {key: _best_and_ratio(value, "destination_patch_id") for key, value in forward.items()}
    backward_best = {key: _best_and_ratio(value, "source_patch_id") for key, value in backward.items()}
    accepted: list[dict[str, Any]] = []
    for source_id in sorted(forward_best):
        item, forward_ratio = forward_best[source_id]
        if item is None:
            continue
        destination_id = str(item["destination_patch_id"])
        reverse, backward_ratio = backward_best[destination_id]
        if (
            reverse is not None
            and str(reverse["source_patch_id"]) == source_id
            and float(item["cost"]) <= 1.0
            and forward_ratio >= 1.2
            and backward_ratio >= 1.2
        ):
            payload = {
                "source_patch_id": source_id,
                "destination_patch_id": destination_id,
                "flow_manifests": sorted(str(value) for value in item["flow_manifests"]),
                "match_tuple": item["match_tuple"],
            }
            accepted.append(
                {
                    **item,
                    "source_patch_id": source_id,
                    "destination_patch_id": destination_id,
                    "forward_ratio": forward_ratio,
                    "backward_ratio": backward_ratio,
                    "identity_risk": max(
                        0.0 if math.isinf(forward_ratio) else min(1.0, 1.2 / forward_ratio),
                        0.0 if math.isinf(backward_ratio) else min(1.0, 1.2 / backward_ratio),
                    ),
                    "edge_id": domain_id("csvl-v1/track-edge", payload),
                }
            )
    return accepted


def split_merge_components(candidates: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items = _validated_patch_candidates(candidates)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for item in items:
        source = "s:" + str(item["source_patch_id"])
        destination = "d:" + str(item["destination_patch_id"])
        adjacency[source].add(destination)
        adjacency[destination].add(source)
    components = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        queue: deque[str] = deque([start])
        unseen.remove(start)
        nodes: list[str] = []
        while queue:
            node = queue.popleft()
            nodes.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        sources = sorted(value[2:] for value in nodes if value.startswith("s:"))
        destinations = sorted(value[2:] for value in nodes if value.startswith("d:"))
        ambiguous = any(len(adjacency[node]) > 1 for node in nodes)
        candidate_ids = sorted(
            str(item.get("candidate_id", f'{item["source_patch_id"]}->{item["destination_patch_id"]}'))
            for item in items
            if str(item["source_patch_id"]) in sources
            and str(item["destination_patch_id"]) in destinations
        )
        components.append(
            {
                "sources": sources,
                "destinations": destinations,
                "candidate_ids": candidate_ids,
                "ambiguous": ambiguous,
                "component_id": domain_id("csvl-v1/split-merge", {"candidate_ids": candidate_ids}),
            }
        )
    return components


def propagate_tracks(
    previous_patches: Iterable[Mapping[str, Any]],
    current_patches: Iterable[Mapping[str, Any]],
    accepted_edges: Iterable[Mapping[str, Any]],
    candidate_components: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    previous = {str(item["patch_id"]): item for item in previous_patches}
    current = {str(item["patch_id"]): dict(item) for item in current_patches}
    ambiguous_sources = {
        value
        for component in candidate_components
        if component["ambiguous"]
        for value in component["sources"]
    }
    ambiguous_destinations = {
        value
        for component in candidate_components
        if component["ambiguous"]
        for value in component["destinations"]
    }
    edge_by_destination: dict[str, Mapping[str, Any]] = {}
    for edge in accepted_edges:
        source_id = str(edge["source_patch_id"])
        destination_id = str(edge["destination_patch_id"])
        if source_id in ambiguous_sources or destination_id in ambiguous_destinations:
            continue
        if source_id not in previous or destination_id not in current:
            raise ValueError("accepted edge endpoint is absent from its frame")
        if destination_id in edge_by_destination:
            raise ValueError("multiple accepted edges target one patch")
        edge_by_destination[destination_id] = edge
    output = []
    for patch_id in sorted(current):
        patch = current[patch_id]
        if patch_id in ambiguous_destinations:
            patch.update({"track_id": None, "identity_state": "uncertain_split_merge"})
        elif patch_id in edge_by_destination:
            source_id = str(edge_by_destination[patch_id]["source_patch_id"])
            source = previous[source_id]
            if source.get("track_id") in (None, ""):
                raise ValueError("propagated source patch has no prior track ID")
            patch.update(
                {
                    "track_id": str(source["track_id"]),
                    "identity_state": "propagated",
                    "temporal_edge_id": str(edge_by_destination[patch_id]["edge_id"]),
                }
            )
        else:
            payload = {
                "scene": patch["scene"],
                "scored_target": patch["scored_target"],
                "first_frame": patch["frame"],
                "initial_patch_id": patch_id,
            }
            patch.update({"track_id": domain_id("csvl-v1/track", payload), "identity_state": "new"})
        output.append(patch)
    return output


_DORMANT_ALLOWED = {
    "last_frame",
    "last_visible_projections",
    "linear_rgb",
    "normal",
    "identity_descriptor",
    "physical_ancestry",
    "flow_chain",
}


def advance_dormant_tracks(
    dormant: Mapping[str, Mapping[str, Any]], current_frame: int, *, maximum_frames: int = 5
) -> dict[str, dict[str, Any]]:
    """Retain identity descriptors only; current xyz/order is deliberately absent."""

    if maximum_frames < 0:
        raise ValueError("maximum dormant age cannot be negative")
    output: dict[str, dict[str, Any]] = {}
    for track_id, record in sorted(dormant.items()):
        if "last_frame" not in record:
            raise ValueError("dormant record is missing last_frame")
        prohibited = {"world_point", "xyz", "depth", "z", "order", "current_projection"}
        if prohibited & set(record):
            raise ValueError("dormant record contains prohibited current geometry/order")
        age = int(current_frame) - int(record["last_frame"])
        if 0 < age <= maximum_frames:
            clean = {key: value for key, value in record.items() if key in _DORMANT_ALLOWED}
            clean["track_id"] = str(track_id)
            clean["dormant_age"] = age
            output[str(track_id)] = clean
    return output


def reidentify_track(candidates: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [
        item
        for item in candidates
        if int(item["camera_count"]) >= 2
        and float(item["endpoint_error_pixels"]) <= 2.0
        and float(item["ncc"]) >= 0.60
        and float(item["normal_angle_degrees"]) <= 30.0
        and float(item["rgb_l2"]) <= 0.15
        and float(item["cost"]) <= 1.0
        and bool(item.get("reciprocal", False))
        and bool(item.get("one_to_one", False))
        and bool(item.get("complete_flow_chain", False))
    ]
    best, ratio = _best_and_ratio(eligible, "track_id")
    if best is None or ratio < 1.2:
        return None
    return {
        **best,
        "second_best_ratio": ratio,
        "identity_risk": 0.0 if math.isinf(ratio) else min(1.0, 1.2 / ratio),
        "event": "reappearance",
    }


_STATES = {"visible", "occluded", "out_of_frustum", "invalid", "uncertain"}


def derive_states(previous: str | None, current: str | None) -> str:
    """Return the exact per-camera transition state; gaps are not reveals."""

    if previous not in _STATES or current not in _STATES:
        return "uncertain"
    if previous == "occluded" and current == "visible":
        return "reveal"
    if previous == "visible" and current == "occluded":
        return "hide"
    if previous == current and previous in {"visible", "occluded", "out_of_frustum"}:
        return "none"
    return "uncertain"


def aggregate_track_time(camera_states: Iterable[str], *, hypothesis_exists: bool) -> str:
    states = [str(value) for value in camera_states]
    if any(state not in _STATES for state in states):
        raise ValueError("unknown per-camera observation state")
    if any(state == "visible" for state in states):
        return "observed"
    if not hypothesis_exists:
        return "unobserved"
    return "uncertain"


def transition_risk(previous_region: float, current_region: float, temporal_edge: float) -> float:
    values = [float(previous_region), float(current_region), float(temporal_edge)]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("transition risk terms must be finite in [0,1]")
    return max(values)


__all__ = [
    "advance_dormant_tracks",
    "aggregate_track_time",
    "derive_states",
    "identity_risk",
    "patch_pair_terms",
    "propagate_tracks",
    "reciprocal_patch_edges",
    "reidentify_track",
    "split_merge_components",
    "transition_risk",
]
