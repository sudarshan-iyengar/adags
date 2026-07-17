"""Streaming visibility-ledger assembly and recursive provenance checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .canonical import domain_id
from .errors import ProvenanceError
from .surfaces import connected_regions, dense_depth_order


def recursive_provenance_check(
    record: Mapping[str, Any],
    *,
    scored_target: str,
    target_image_hashes: Iterable[str] = (),
) -> set[str]:
    """Validate complete physical ancestry recursively and return its union."""

    prohibited_hashes = {str(value) for value in target_image_hashes}
    if any(not value for value in prohibited_hashes):
        raise ProvenanceError("target image hashes must be nonempty strings")
    seen: set[int] = set()

    def visit(value: Any, *, require_ancestry: bool = False) -> set[str]:
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in seen:
                raise ProvenanceError("cyclic provenance structure")
            seen.add(identity)
            if require_ancestry and "physical_ancestry" not in value:
                raise ProvenanceError("provenance dependency is missing physical ancestry")
            ancestry: set[str] = set()
            if "physical_ancestry" in value:
                raw = value["physical_ancestry"]
                if not isinstance(raw, (list, tuple, set)) or not raw:
                    raise ProvenanceError("physical ancestry must be an explicit nonempty sequence")
                ancestry.update(str(item) for item in raw)
                if any(not item for item in ancestry):
                    raise ProvenanceError("physical ancestry cannot contain empty IDs")
            for key, child in value.items():
                if isinstance(child, str) and child in prohibited_hashes:
                    raise ProvenanceError("target image hash leaked into prediction provenance")
                child_requires_ancestry = key in {
                    "dependencies",
                    "children",
                    "members",
                    "source_nodes",
                    "fused_hypotheses",
                    "temporal_edges",
                    "witnesses",
                }
                if key != "physical_ancestry":
                    ancestry.update(visit(child, require_ancestry=child_requires_ancestry))
            seen.remove(identity)
            if str(scored_target) in ancestry:
                raise ProvenanceError("scored target appears in transitive physical ancestry")
            return ancestry
        if isinstance(value, (list, tuple, set)):
            ancestry: set[str] = set()
            for child in value:
                ancestry.update(visit(child, require_ancestry=require_ancestry))
            return ancestry
        if isinstance(value, str) and value in prohibited_hashes:
            raise ProvenanceError("target image hash leaked into prediction provenance")
        return set()

    if "physical_ancestry" not in record:
        raise ProvenanceError("top-level provenance ancestry is missing")
    ancestry = visit(record)
    if not ancestry:
        raise ProvenanceError("prediction provenance has empty physical ancestry")
    return ancestry


def build_target_frame_ledger(
    *,
    scene: str,
    frame: int,
    scored_target: str,
    track_pixels: Mapping[str, Mapping[tuple[int, int], Mapping[str, Any]]],
    visible_witnesses: Mapping[str, set[str]],
    provenance: Mapping[str, Any],
    target_image_hashes: Iterable[str] = (),
) -> dict[str, Any]:
    ancestry = recursive_provenance_check(
        provenance, scored_target=scored_target, target_image_hashes=target_image_hashes
    )
    for track_id, pixels in track_pixels.items():
        for record in pixels.values():
            record_ancestry = {str(value) for value in record["physical_ancestry"]}
            if scored_target in record_ancestry:
                raise ProvenanceError("scored target leaked into rasterized track support")
            if not record_ancestry.issubset(ancestry):
                raise ProvenanceError(f"track {track_id} ancestry is absent from sealed provenance")
    for witnesses in visible_witnesses.values():
        witness_ids = {str(value) for value in witnesses}
        if scored_target in witness_ids or not witness_ids.issubset(ancestry):
            raise ProvenanceError("visible witness violates sealed target-exclusion provenance")
    ordered = dense_depth_order(track_pixels, visible_witnesses)
    regions = connected_regions(ordered)
    payload = {
        "scene": str(scene),
        "frame": int(frame),
        "scored_target": str(scored_target),
        "physical_ancestry": sorted(ancestry),
        "regions": regions,
    }
    return {"ledger_id": domain_id("csvl-v1/target-frame-ledger", payload), **payload}


def build_scene_ledger(frame_ledgers: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    frames = sorted(
        frame_ledgers,
        key=lambda item: (int(item["frame"]), str(item["scored_target"]), str(item["ledger_id"])),
    )
    if not frames:
        raise ValueError("scene ledger requires at least one frame")
    scene = str(frames[0]["scene"])
    if any(str(item["scene"]) != scene for item in frames):
        raise ValueError("scene ledger cannot mix scenes")
    frame_ids = [str(item["ledger_id"]) for item in frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("scene ledger cannot repeat a frame ledger")
    payload = {"scene": scene, "frame_ledger_ids": frame_ids}
    return {
        "scene_ledger_id": domain_id("csvl-v1/scene-ledger", payload),
        **payload,
        "frames": frames,
    }


def seal_scene_ledger(scene_ledger: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "scene": scene_ledger["scene"],
        "scene_ledger_id": scene_ledger["scene_ledger_id"],
        "frame_ledger_ids": scene_ledger["frame_ledger_ids"],
    }
    return {
        **payload,
        "seal_id": domain_id("csvl-v1/scene-ledger-seal", payload),
        "sealed": True,
    }


def freeze_train_sidecars(
    scene_ledger: Mapping[str, Any], *, prohibited_keys: Iterable[str] = ()
) -> dict[str, Any]:
    prohibited = {str(value).lower() for value in prohibited_keys} | {
        "target_rgb",
        "human_labels",
        "evaluator",
        "r009",
    }

    def check(value: Any) -> None:
        if isinstance(value, Mapping):
            hits = []
            for key in value:
                lowered = str(key).lower()
                if any(token == lowered or token in lowered for token in prohibited):
                    hits.append(str(key))
            if hits:
                raise ProvenanceError(f"prohibited train-sidecar fields: {sorted(hits)}")
            for child in value.values():
                check(child)
        elif isinstance(value, (list, tuple, set)):
            for child in value:
                check(child)

    check(scene_ledger)
    payload = {
        "scene": scene_ledger["scene"],
        "scene_ledger_id": scene_ledger["scene_ledger_id"],
        "frame_ledger_ids": scene_ledger["frame_ledger_ids"],
        "prohibited_read_proof": sorted(prohibited),
    }
    return {
        **payload,
        "freeze_id": domain_id("csvl-v1/train-sidecar-freeze", payload),
    }


__all__ = [
    "build_scene_ledger",
    "build_target_frame_ledger",
    "freeze_train_sidecars",
    "recursive_provenance_check",
    "seal_scene_ledger",
]
