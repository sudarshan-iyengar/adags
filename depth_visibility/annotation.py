"""Fail-closed annotation manifests and two-stage reference handling.

This module creates only empty, blinded packet manifests. It never infers or
fills a human field. Completed reference artifacts are accepted only after
role-separation, row-key, provenance, and evaluability checks.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from .canonical import domain_id, sha256_file
from .errors import ContractError, ProvenanceError, SchemaError
from .matching import match_discoveries


WINDOW_SCHEMA = "depth-visibility-annotation-windows-v1"
PACKET_SCHEMA = "phase9-annotation-packet-v1"
LABEL_FREEZE_SCHEMA = "phase9-human-label-freeze-v1"
_STATES = {"visible", "occluded", "out_of_frustum", "unknown"}
_TRANSITIONS = {"reveal", "hide", "none", "unknown"}
_HUMAN_MANIFEST_FIELDS = {
    "discovery_a_sealed_manifest",
    "discovery_b_sealed_manifest",
    "union_roster_manifest",
    "roster_pass_a_manifest",
    "roster_pass_b_manifest",
    "adjudication_manifest",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise SchemaError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_unique_object)


def _require_keys(record: Mapping[str, Any], required: set[str], where: str) -> None:
    missing = required - set(record)
    if missing:
        raise SchemaError(f"{where} missing keys: {sorted(missing)}")


def validate_annotation_windows(
    manifest: Mapping[str, Any],
    *,
    require_initial_empty: bool = True,
) -> dict[str, Any]:
    """Validate the frozen 54-window population and leakage boundaries."""

    _require_keys(
        manifest,
        {
            "schema_version",
            "frames_are_inclusive",
            "window_length_frames",
            "scene_order",
            "windows",
            "r009_exclusion",
            "double_annotation_protocol",
            "shared_boundary_rule",
            "human_fields_status",
        },
        "annotation-window manifest",
    )
    if manifest["schema_version"] != WINDOW_SCHEMA:
        raise SchemaError("wrong annotation-window schema")
    if manifest["frames_are_inclusive"] is not True or int(manifest["window_length_frames"]) != 11:
        raise ContractError("Phase 9 windows must be inclusive 11-frame windows")
    scenes = list(manifest["scene_order"])
    if scenes != ["cut_roasted_beef", "flame_steak", "sear_steak"]:
        raise ContractError("annotation scene order is not frozen")
    windows = list(manifest["windows"])
    if len(windows) != 54:
        raise ContractError("annotation manifest must contain exactly 54 windows")

    ids: set[str] = set()
    per_scene = defaultdict(int)
    previous_by_scene: dict[str, Mapping[str, Any]] = {}
    r009 = list(manifest["r009_exclusion"]["historical_windows"])
    margin = int(manifest["r009_exclusion"]["margin_frames_each_side"])
    for position, window in enumerate(windows):
        _require_keys(
            window,
            {
                "window_id",
                "scene",
                "split",
                "frame_start_inclusive",
                "frame_end_inclusive",
                "assignment",
                "human_fields",
                "r009_disjoint_with_margin",
            },
            f"window[{position}]",
        )
        window_id = str(window["window_id"])
        if window_id in ids:
            raise ContractError(f"duplicate window_id: {window_id}")
        ids.add(window_id)
        scene = str(window["scene"])
        if scene not in scenes:
            raise ContractError(f"unknown annotation scene: {scene}")
        start, end = int(window["frame_start_inclusive"]), int(window["frame_end_inclusive"])
        if start < 0 or end >= 300 or end - start + 1 != 11:
            raise ContractError(f"invalid inclusive frame interval for {window_id}")
        if str(window["split"]) not in {"calibration", "development_test", "transfer"}:
            raise ContractError(f"invalid split for {window_id}")
        roles = list(window["assignment"])
        if [item.get("role") for item in roles] != ["annotator_a", "annotator_b"]:
            raise ContractError(f"{window_id} does not have ordered independent A/B roles")
        identities = [item.get("actual_annotator_id") for item in roles]
        if require_initial_empty and identities != [None, None]:
            raise ContractError(f"{window_id} initial annotator identities must be empty")
        if not require_initial_empty and (None in identities or identities[0] == identities[1]):
            raise ContractError(f"{window_id} annotator identities must be filled and distinct")
        human_fields = window["human_fields"]
        if set(human_fields) != _HUMAN_MANIFEST_FIELDS:
            raise SchemaError(f"{window_id} has incomplete/unknown human manifest bindings")
        if require_initial_empty and any(value is not None for value in human_fields.values()):
            raise ContractError(f"{window_id} initial human bindings must be empty")
        if not require_initial_empty and any(value is None for value in human_fields.values()):
            raise ContractError(f"{window_id} completed human bindings must all be sealed")

        for historical in r009:
            if str(historical["scene"]) != scene:
                continue
            h0 = int(historical["frame_start_inclusive"]) - margin
            h1 = int(historical["frame_end_inclusive"]) + margin
            if max(start, h0) <= min(end, h1):
                raise ContractError(f"{window_id} overlaps R009 plus the frozen margin")
        if window["r009_disjoint_with_margin"] is not True:
            raise ContractError(f"{window_id} lacks the frozen R009-separation assertion")

        previous = previous_by_scene.get(scene)
        if previous is not None and start < int(previous["frame_end_inclusive"]):
            raise ContractError(f"{window_id} overlaps more than a shared boundary")
        previous_by_scene[scene] = window
        per_scene[scene] += 1

    if dict(per_scene) != {scene: 18 for scene in scenes}:
        raise ContractError("annotation manifest must contain 18 windows per scene")
    cut = [window for window in windows if window["scene"] == "cut_roasted_beef"]
    if [window["split"] for window in cut] != ["calibration"] * 7 + ["development_test"] * 11:
        raise ContractError("cut calibration/development split is not frozen")
    for window in windows:
        if window["scene"] != "cut_roasted_beef" and window["split"] != "transfer":
            raise ContractError("transfer scenes must remain locked")
    if require_initial_empty and manifest["human_fields_status"] != "empty":
        raise ContractError("initial annotation manifest must declare empty human fields")
    return {
        "window_count": len(windows),
        "window_ids": sorted(ids),
        "per_scene": dict(sorted(per_scene.items())),
        "r009_separation_verified": True,
        "human_fields_empty": require_initial_empty,
    }


def build_empty_annotation_packet(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str,
    test_camera: str = "cam00",
) -> dict[str, Any]:
    """Build a bounded raw-RGB-only handoff manifest with empty human rows."""

    audit = validate_annotation_windows(manifest, require_initial_empty=True)
    frame_reviews = []
    rgb_population = []
    for window in manifest["windows"]:
        for frame in range(int(window["frame_start_inclusive"]), int(window["frame_end_inclusive"]) + 1):
            frame_reviews.append(
                {
                    "row_key": [str(window["window_id"]), test_camera, frame],
                    "window_id": str(window["window_id"]),
                    "camera_id": test_camera,
                    "frame": frame,
                    "spatial_complete": None,
                    "no_evaluable_visible_rear_surface": None,
                    "unknown_reason": None,
                    "annotator_id": None,
                    "adjudicator_id": None,
                    "evaluable": False,
                }
            )
            rgb_population.append(
                {
                    "window_id": str(window["window_id"]),
                    "scene": str(window["scene"]),
                    "camera_id": test_camera,
                    "frame": frame,
                    "source": "raw_rgb",
                }
            )
    payload = {
        "schema_version": PACKET_SCHEMA,
        "source_window_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "window_count": audit["window_count"],
        "windows": [
            {
                "window_id": str(window["window_id"]),
                "scene": str(window["scene"]),
                "split": str(window["split"]),
                "frame_start_inclusive": int(window["frame_start_inclusive"]),
                "frame_end_inclusive": int(window["frame_end_inclusive"]),
                "assignment": window["assignment"],
                "human_manifest_bindings": window["human_fields"],
            }
            for window in manifest["windows"]
        ],
        "rgb_population": rgb_population,
        "tables": {
            "track_frames": [],
            "ordering_pairs": [],
            "transitions": [],
            "frame_reviews": frame_reviews,
        },
        "human_fields_status": "empty",
        "prediction_fields_present": False,
        "allowed_visual_input": "synchronized_raw_rgb_only",
        "prohibited_visual_inputs": ["csvl", "da3", "flow", "residual", "render", "r009_crop"],
        "r009_separation_verified": True,
    }
    return {**payload, "packet_id": domain_id("csvl-v1/annotation-packet", payload)}


def validate_empty_annotation_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    if packet.get("schema_version") != PACKET_SCHEMA:
        raise SchemaError("wrong annotation-packet schema")
    if packet.get("human_fields_status") != "empty" or packet.get("prediction_fields_present") is not False:
        raise ContractError("annotation packet is not blinded and empty")
    tables = packet.get("tables")
    if not isinstance(tables, Mapping):
        raise SchemaError("annotation packet tables are missing")
    if tables.get("track_frames") or tables.get("ordering_pairs") or tables.get("transitions"):
        raise ContractError("empty packet must not fabricate discovered tracks or labels")
    reviews = tables.get("frame_reviews")
    if not isinstance(reviews, list) or len(reviews) != 54 * 11:
        raise ContractError("empty packet must enumerate every candidate cam00 frame row")
    for row in reviews:
        for key in ("spatial_complete", "no_evaluable_visible_rear_surface", "unknown_reason", "annotator_id", "adjudicator_id"):
            if row.get(key) is not None:
                raise ContractError(f"empty packet populated human field {key}")
        if row.get("evaluable") is not False:
            raise ContractError("empty packet rows must default evaluable=false")
    return {"valid": True, "review_row_count": len(reviews), "human_fields_empty": True}


def _candidate_components(diagnostics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for item in diagnostics:
        if not item["accepted"]:
            continue
        left = "a:" + str(item["left_track_id"])
        right = "b:" + str(item["right_track_id"])
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(adjacency)
    output = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue: deque[str] = deque([start])
        nodes = []
        while queue:
            node = queue.popleft()
            nodes.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        a_ids = sorted(item[2:] for item in nodes if item.startswith("a:"))
        b_ids = sorted(item[2:] for item in nodes if item.startswith("b:"))
        ambiguous = len(a_ids) > 1 or len(b_ids) > 1 or any(len(adjacency[node]) > 1 for node in nodes)
        output.append({"a_track_ids": a_ids, "b_track_ids": b_ids, "ambiguous": ambiguous})
    return output


def build_union_roster(
    window_id: str,
    discoveries_a: Sequence[Mapping[str, Any]],
    discoveries_b: Sequence[Mapping[str, Any]],
    *,
    iou_fn: Callable[[Any, Any], float] | None = None,
) -> dict[str, Any]:
    """Match discoveries while preserving one-sided and ambiguous evidence."""

    result = match_discoveries(window_id, discoveries_a, discoveries_b, iou_fn=iou_fn)
    components = _candidate_components(result["candidate_diagnostics"])
    ambiguous_a = {value for component in components if component["ambiguous"] for value in component["a_track_ids"]}
    ambiguous_b = {value for component in components if component["ambiguous"] for value in component["b_track_ids"]}
    matched = [(a, b) for a, b in result["assignment"] if a not in ambiguous_a and b not in ambiguous_b]
    matched_a = {a for a, _ in matched}
    matched_b = {b for _, b in matched}
    all_a = {str(item["track_id"]) for item in discoveries_a}
    all_b = {str(item["track_id"]) for item in discoveries_b}

    roster = []
    for left, right in sorted(matched):
        identity = {"window_id": str(window_id), "discovery_a": left, "discovery_b": right}
        roster.append({"roster_track_id": domain_id("csvl-v1/annotation-roster-track", identity), **identity, "status": "matched", "requires_adjudication": False})
    for role, values in (("a", sorted(all_a - matched_a - ambiguous_a)), ("b", sorted(all_b - matched_b - ambiguous_b))):
        for local_id in values:
            identity = {"window_id": str(window_id), f"discovery_{role}": local_id}
            roster.append(
                {
                    "roster_track_id": domain_id("csvl-v1/annotation-roster-track", identity),
                    "window_id": str(window_id),
                    "discovery_a": local_id if role == "a" else None,
                    "discovery_b": local_id if role == "b" else None,
                    "status": "one_sided",
                    "missing_role_response": "not_found",
                    "requires_adjudication": False,
                }
            )
    for component in components:
        if component["ambiguous"]:
            identity = {"window_id": str(window_id), "discovery_a": component["a_track_ids"], "discovery_b": component["b_track_ids"]}
            roster.append(
                {
                    "roster_track_id": domain_id("csvl-v1/annotation-roster-ambiguous", identity),
                    **identity,
                    "status": "fragment_merge_unknown",
                    "requires_adjudication": True,
                }
            )
    roster.sort(key=lambda item: str(item["roster_track_id"]))
    payload = {
        "window_id": str(window_id),
        "roster": roster,
        "matcher": {key: result[key] for key in ("integer_weight", "denominator_exponent", "binary64_weight_sum", "tie_rule", "candidate_diagnostics")},
        "ambiguous_components": [component for component in components if component["ambiguous"]],
    }
    return {**payload, "roster_id": domain_id("csvl-v1/annotation-union-roster", payload)}


def assign_window(scene: str, frame: int, windows: Sequence[Mapping[str, Any]]) -> str:
    """Assign an event to exactly one candidate window; earlier wins ties."""

    candidates = [
        window
        for window in windows
        if str(window["scene"]) == str(scene)
        and int(window["frame_start_inclusive"]) <= int(frame) <= int(window["frame_end_inclusive"])
    ]
    if not candidates:
        raise ContractError(f"frame {scene}/{frame} belongs to no candidate window")
    selected = min(candidates, key=lambda window: (int(window["frame_start_inclusive"]), int(window["frame_end_inclusive"]), str(window["window_id"])))
    return str(selected["window_id"])


def validate_human_label_freeze(
    artifact: Mapping[str, Any],
    windows_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate genuine human return structure without interpreting labels."""

    if artifact.get("schema_version") != LABEL_FREEZE_SCHEMA:
        raise SchemaError("wrong human-label-freeze schema")
    validate_annotation_windows(windows_manifest, require_initial_empty=False)
    if artifact.get("evidence_type") != "human_reference":
        raise ProvenanceError("Gate A labels must declare human_reference evidence")
    tables = artifact.get("tables")
    if not isinstance(tables, Mapping):
        raise SchemaError("human label tables are missing")
    expected_tables = {"track_frames", "ordering_pairs", "transitions", "frame_reviews"}
    if set(tables) != expected_tables:
        raise SchemaError("human label tables are incomplete or unknown")

    seen: dict[str, set[tuple[Any, ...]]] = {name: set() for name in expected_tables}
    for row in tables["track_frames"]:
        _require_keys(row, {"window_id", "roster_track_id", "camera_id", "frame", "state", "rear_polygon", "state_aperture", "evaluable", "annotator_a_response", "annotator_b_response", "adjudication"}, "track_frames row")
        key = (row["window_id"], row["roster_track_id"], row["camera_id"], int(row["frame"]))
        if key in seen["track_frames"]:
            raise ContractError(f"duplicate track_frames row: {key}")
        seen["track_frames"].add(key)
        if row["state"] not in _STATES:
            raise ContractError(f"invalid state in {key}")
        if row["evaluable"] is True:
            if row["state"] == "visible" and (not row["rear_polygon"] or row["state_aperture"] != row["rear_polygon"]):
                raise ContractError("visible evaluable rows require polygon=state_aperture")
            if row["state"] == "occluded" and not row["state_aperture"]:
                raise ContractError("occluded evaluable rows require a genuine state aperture")
            if row["state"] in {"unknown", "out_of_frustum"}:
                raise ContractError("unknown/out-of-frustum rows cannot be evaluable")
        if row["annotator_a_response"] is None or row["annotator_b_response"] is None:
            raise ContractError("every roster row requires explicit A and B responses")
        if row["annotator_a_response"] != row["annotator_b_response"] and row["adjudication"] is None and row["evaluable"] is True:
            raise ContractError("unadjudicated disagreement must remain non-evaluable")

    for row in tables["ordering_pairs"]:
        _require_keys(row, {"window_id", "pair_id", "camera_id", "frame", "foreground_track_id", "rear_track_id", "label", "evaluable", "annotator_a_response", "annotator_b_response", "adjudication"}, "ordering_pairs row")
        key = (row["window_id"], row["pair_id"], row["camera_id"], int(row["frame"]))
        if key in seen["ordering_pairs"]:
            raise ContractError(f"duplicate ordering row: {key}")
        seen["ordering_pairs"].add(key)
        if row["label"] not in {"foreground_before_rear", "unknown"}:
            raise ContractError("invalid ordering label")

    for row in tables["transitions"]:
        _require_keys(row, {"window_id", "roster_track_id", "camera_id", "frame_t", "frame_t1", "label", "evaluable", "annotator_a_response", "annotator_b_response", "adjudication"}, "transitions row")
        key = (row["window_id"], row["roster_track_id"], row["camera_id"], int(row["frame_t"]), int(row["frame_t1"]))
        if key in seen["transitions"]:
            raise ContractError(f"duplicate transition row: {key}")
        seen["transitions"].add(key)
        if int(row["frame_t1"]) != int(row["frame_t"]) + 1 or row["label"] not in _TRANSITIONS:
            raise ContractError("invalid transition row")

    for row in tables["frame_reviews"]:
        _require_keys(row, {"window_id", "camera_id", "frame", "spatial_complete", "no_evaluable_visible_rear_surface", "unknown_reason", "annotator_provenance"}, "frame_reviews row")
        key = (row["window_id"], row["camera_id"], int(row["frame"]))
        if key in seen["frame_reviews"]:
            raise ContractError(f"duplicate frame review: {key}")
        seen["frame_reviews"].add(key)
        if row["spatial_complete"] is True and row["unknown_reason"] is not None:
            raise ContractError("spatial-complete frame cannot retain an unresolved unknown")
    return {"valid": True, "row_counts": {key: len(value) for key, value in seen.items()}, "labels_genuine_but_not_scored": True}


__all__ = [
    "LABEL_FREEZE_SCHEMA",
    "PACKET_SCHEMA",
    "WINDOW_SCHEMA",
    "assign_window",
    "build_empty_annotation_packet",
    "build_union_roster",
    "load_json",
    "validate_annotation_windows",
    "validate_empty_annotation_packet",
    "validate_human_label_freeze",
]
