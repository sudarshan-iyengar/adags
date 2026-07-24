"""CVAT handoff helpers for Phase 9 human reference annotations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import copy
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Any

from .annotation import (
    LABEL_FREEZE_SCHEMA,
    load_json,
    validate_annotation_windows,
    validate_human_label_freeze,
)
from .canonical import sha256_file
from .errors import ContractError, SchemaError
from .evaluator import write_json_atomic


_HUMAN_MANIFEST_FIELDS = (
    "discovery_a_sealed_manifest",
    "discovery_b_sealed_manifest",
    "union_roster_manifest",
    "roster_pass_a_manifest",
    "roster_pass_b_manifest",
    "adjudication_manifest",
)
_IMAGE_RE = re.compile(r"(?P<camera>cam\d+)_(?P<frame>\d+)\.[^.]+$")
_STATES = {"visible", "occluded", "out_of_frustum", "unknown"}
_POLYGON_MEANINGS = {
    "visible_rear_polygon",
    "occluded_state_aperture",
    "source_visible_rear_polygon",
}
_TRANSITIONS = {"reveal", "hide", "none", "unknown"}
_ORDERING_LABELS = {"foreground_before_rear", "unknown"}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _selected_windows(manifest: Mapping[str, Any], scene: str) -> list[dict[str, Any]]:
    validate_annotation_windows(manifest, require_initial_empty=True)
    windows = [dict(item) for item in manifest["windows"] if str(item["scene"]) == scene]
    if not windows:
        raise ContractError(f"annotation manifest contains no windows for scene {scene!r}")
    return windows


def image_path(scene_root: str | Path, camera_id: str, frame: int) -> Path:
    return Path(scene_root) / "images" / f"{camera_id}_{frame:04d}.png"


def generate_cvat_annotation_templates(
    *,
    packet_manifest_path: str | Path,
    output_dir: str | Path,
    scene: str = "cut_roasted_beef",
    raw_scene_root: str | Path | None = None,
    test_camera: str = "cam00",
) -> dict[str, Any]:
    """Generate CVAT task lists and CSV tables for Phase 9 labeling."""

    packet_path = Path(packet_manifest_path)
    packet = load_json(packet_path)
    windows = [dict(item) for item in packet.get("windows", []) if str(item.get("scene")) == scene]
    if not windows:
        raise ContractError(f"packet manifest contains no windows for scene {scene!r}")

    scene_root = Path(raw_scene_root) if raw_scene_root else Path("/leonardo_work/EUHPC_D21_034/proj_adags/data/n3v") / scene
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    window_rows = []
    task_rows = []
    frame_review_rows = []
    for window in windows:
        start = int(window["frame_start_inclusive"])
        end = int(window["frame_end_inclusive"])
        image_paths = [str(image_path(scene_root, test_camera, frame)) for frame in range(start, end + 1)]
        window_rows.append(
            {
                "window_id": window["window_id"],
                "scene": scene,
                "split": window["split"],
                "frame_start_inclusive": start,
                "frame_end_inclusive": end,
                "test_camera": test_camera,
                "image_paths_semicolon": ";".join(image_paths),
            }
        )
        for role in ("annotator_a", "annotator_b"):
            task_rows.append(
                {
                    "task_name": f"{window['window_id']}__{role}__{test_camera}",
                    "role": role,
                    "window_id": window["window_id"],
                    "camera_id": test_camera,
                    "frame_start_inclusive": start,
                    "frame_end_inclusive": end,
                    "image_paths_semicolon": ";".join(image_paths),
                }
            )
        for frame in range(start, end + 1):
            frame_review_rows.append(
                {
                    "window_id": window["window_id"],
                    "camera_id": test_camera,
                    "frame": frame,
                    "spatial_complete": "",
                    "no_evaluable_visible_rear_surface": "",
                    "unknown_reason": "",
                    "annotator_provenance": "",
                }
            )

    first_pass_fields = [
        "window_id",
        "candidate_track_id",
        "rear_surface_description",
        "foreground_occluder_description",
        "first_visible_frame",
        "occluded_or_revealed_frame",
        "confidence",
        "notes",
    ]
    track_frame_fields = [
        "window_id",
        "roster_track_id",
        "camera_id",
        "frame",
        "state",
        "evaluable",
        "rear_polygon_source_role",
        "rear_polygon_candidate_track_id",
        "rear_polygon_json",
        "state_aperture_source_role",
        "state_aperture_candidate_track_id",
        "state_aperture_json",
        "annotator_a_response",
        "annotator_b_response",
        "adjudication",
    ]
    transition_fields = [
        "window_id",
        "roster_track_id",
        "camera_id",
        "frame_t",
        "frame_t1",
        "label",
        "evaluable",
        "annotator_a_response",
        "annotator_b_response",
        "adjudication",
    ]
    ordering_fields = [
        "window_id",
        "pair_id",
        "camera_id",
        "frame",
        "foreground_track_id",
        "rear_track_id",
        "label",
        "evaluable",
        "annotator_a_response",
        "annotator_b_response",
        "adjudication",
    ]

    _write_csv(destination / "windows.csv", ["window_id", "scene", "split", "frame_start_inclusive", "frame_end_inclusive", "test_camera", "image_paths_semicolon"], window_rows)
    _write_csv(destination / "cvat_tasks.csv", ["task_name", "role", "window_id", "camera_id", "frame_start_inclusive", "frame_end_inclusive", "image_paths_semicolon"], task_rows)
    _write_csv(destination / "first_pass_annotator_a.csv", first_pass_fields, [])
    _write_csv(destination / "first_pass_annotator_b.csv", first_pass_fields, [])
    _write_csv(destination / "frame_reviews_final.csv", ["window_id", "camera_id", "frame", "spatial_complete", "no_evaluable_visible_rear_surface", "unknown_reason", "annotator_provenance"], frame_review_rows)
    _write_csv(destination / "track_frames_final.csv", track_frame_fields, [])
    _write_csv(destination / "transitions_final.csv", transition_fields, [])
    _write_csv(destination / "ordering_pairs_final.csv", ordering_fields, [])

    label_spec = {
        "label": "rear_surface_track",
        "shape": "polygon",
        "recommended_mode": "track",
        "attributes": {
            "candidate_track_id": "text",
            "window_id": "text",
            "camera_id": "text",
            "state": sorted(_STATES),
            "polygon_meaning": sorted(_POLYGON_MEANINGS),
            "foreground_occluder_description": "text",
            "confidence": ["clear", "unclear"],
            "notes": "text",
        },
    }
    write_json_atomic(destination / "cvat_label_schema.json", label_spec)
    _write_text(destination / "README.md", _template_readme(scene=scene, packet_manifest_path=packet_path, test_camera=test_camera))

    return {
        "output_dir": str(destination),
        "scene": scene,
        "window_count": len(windows),
        "task_count": len(task_rows),
        "frame_review_rows": len(frame_review_rows),
    }


def _template_readme(*, scene: str, packet_manifest_path: Path, test_camera: str) -> str:
    return f"""# Phase 9 CVAT Annotation Kit

Scene: `{scene}`
Packet manifest: `{packet_manifest_path}`
Primary camera: `{test_camera}`

Use raw RGB only. Do not open CSVL, DA3 depth, flow, residuals, renders, or old R009 crops while annotating.

Workflow:
1. Use `windows.csv` to see the frozen 11-frame windows.
2. Create one CVAT task per row in `cvat_tasks.csv`.
3. Configure a polygon track label named `rear_surface_track` with the attributes in `cvat_label_schema.json`.
4. Annotator A fills `first_pass_annotator_a.csv`; annotator B independently fills `first_pass_annotator_b.csv`.
5. Export each CVAT task as native CVAT XML and run `extract-cvat` to make polygon CSVs.
6. After matching/adjudication, fill `track_frames_final.csv`, `transitions_final.csv`, `ordering_pairs_final.csv`, and `frame_reviews_final.csv`.
7. Run `assemble-labels` to produce the Phase 9 human label JSON.

Polygon rule:
- `visible_rear_polygon` surrounds only the visible part of the rear surface.
- `occluded_state_aperture` is a small polygon on the foreground occluder where the rear surface is known to lie behind it.
- Never draw an imagined hidden full shape.
"""


def _xml_attributes(element: ET.Element) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for child in element.findall("attribute"):
        name = child.attrib.get("name")
        if name:
            attrs[name] = (child.text or "").strip()
    return attrs


def _parse_image_name(name: str) -> tuple[str | None, int | None]:
    match = _IMAGE_RE.search(Path(name).name)
    if not match:
        return None, None
    return match.group("camera"), int(match.group("frame"))


def _parse_points(points: str) -> list[list[float]]:
    output = []
    for item in points.split(";"):
        if not item.strip():
            continue
        parts = item.split(",")
        if len(parts) != 2:
            raise SchemaError(f"invalid CVAT polygon point: {item!r}")
        output.append([float(parts[0]), float(parts[1])])
    if len(output) < 3:
        raise SchemaError("CVAT polygon has fewer than three points")
    return output


def _polygon_row(
    *,
    role: str,
    source_export: str,
    label: str,
    points: str,
    attributes: Mapping[str, str],
    frame: int,
    camera_id: str,
    window_id: str,
    cvat_track_id: str | None = None,
) -> dict[str, Any]:
    candidate = attributes.get("candidate_track_id") or attributes.get("track_id") or cvat_track_id
    if not candidate:
        raise ContractError("CVAT polygon is missing candidate_track_id")
    state = attributes.get("state", "unknown").strip() or "unknown"
    if state not in _STATES:
        raise ContractError(f"invalid CVAT state {state!r}")
    meaning = attributes.get("polygon_meaning", "").strip()
    if not meaning:
        meaning = "visible_rear_polygon" if state == "visible" else "occluded_state_aperture" if state == "occluded" else "visible_rear_polygon"
    if meaning not in _POLYGON_MEANINGS:
        raise ContractError(f"invalid CVAT polygon_meaning {meaning!r}")
    return {
        "role": role,
        "candidate_track_id": candidate,
        "window_id": attributes.get("window_id") or window_id,
        "camera_id": attributes.get("camera_id") or camera_id,
        "frame": int(attributes.get("frame") or frame),
        "state": state,
        "polygon_meaning": meaning,
        "points_json": json.dumps(_parse_points(points), separators=(",", ":")),
        "confidence": attributes.get("confidence", ""),
        "notes": attributes.get("notes", ""),
        "source_export": source_export,
        "cvat_track_id": cvat_track_id or "",
        "cvat_shape_label": label,
    }


def extract_cvat_polygons(
    *,
    cvat_xml_path: str | Path,
    role: str,
    output_csv: str | Path,
    window_id: str,
    camera_id: str = "cam00",
    frame_start: int | None = None,
    label_name: str = "rear_surface_track",
) -> dict[str, Any]:
    """Extract Phase 9 polygon rows from a native CVAT XML export."""

    source = Path(cvat_xml_path)
    root = ET.parse(source).getroot()
    rows: list[dict[str, Any]] = []

    for image in root.findall("image"):
        image_camera, image_frame = _parse_image_name(image.attrib.get("name", ""))
        inferred_camera = image_camera or camera_id
        if image_frame is None:
            if frame_start is None:
                raise ContractError("image-mode CVAT export needs frame-like image names or --frame-start")
            image_frame = frame_start + int(image.attrib["id"])
        for polygon in image.findall("polygon"):
            if polygon.attrib.get("label") != label_name:
                continue
            attrs = _xml_attributes(polygon)
            rows.append(
                _polygon_row(
                    role=role,
                    source_export=str(source),
                    label=polygon.attrib.get("label", ""),
                    points=polygon.attrib["points"],
                    attributes=attrs,
                    frame=image_frame,
                    camera_id=inferred_camera,
                    window_id=window_id,
                )
            )

    for track in root.findall("track"):
        if track.attrib.get("label") != label_name:
            continue
        track_attrs = _xml_attributes(track)
        track_id = track.attrib.get("id")
        for polygon in track.findall("polygon"):
            if polygon.attrib.get("outside") == "1":
                continue
            if frame_start is None:
                raise ContractError("track-mode CVAT export requires --frame-start")
            attrs = {**track_attrs, **_xml_attributes(polygon)}
            rows.append(
                _polygon_row(
                    role=role,
                    source_export=str(source),
                    label=track.attrib.get("label", ""),
                    points=polygon.attrib["points"],
                    attributes=attrs,
                    frame=frame_start + int(polygon.attrib["frame"]),
                    camera_id=camera_id,
                    window_id=window_id,
                    cvat_track_id=track_id,
                )
            )

    _write_csv(
        Path(output_csv),
        ["role", "candidate_track_id", "window_id", "camera_id", "frame", "state", "polygon_meaning", "points_json", "confidence", "notes", "source_export", "cvat_track_id", "cvat_shape_label"],
        rows,
    )
    return {"source": str(source), "output_csv": str(output_csv), "polygon_rows": len(rows)}


def _parse_bool(value: str, *, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ContractError(f"{field} must be boolean-like, got {value!r}")


def _parse_optional_bool(value: str, *, field: str) -> bool | None:
    if str(value).strip() == "":
        return None
    return _parse_bool(value, field=field)


def _parse_json_or_string(value: str, *, required: bool = False) -> Any:
    text = str(value).strip()
    if not text:
        if required:
            raise ContractError("required response/adjudication field is empty")
        return None
    if text[0] in "[{":
        return json.loads(text)
    return text


def _parse_polygon_json(value: str) -> list[list[float]]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or len(parsed) < 3:
        raise SchemaError("polygon JSON must be a list with at least three points")
    output = []
    for point in parsed:
        if not isinstance(point, list) or len(point) != 2:
            raise SchemaError("polygon JSON points must be [x,y]")
        output.append([float(point[0]), float(point[1])])
    return output


def _polygon_lookup(polygon_csvs: Sequence[str | Path]) -> dict[tuple[str, str, str, str, int, str], list[list[float]]]:
    lookup: dict[tuple[str, str, str, str, int, str], list[list[float]]] = {}
    for path in polygon_csvs:
        for row in _read_csv(path):
            key = (
                row["role"],
                row["candidate_track_id"],
                row["window_id"],
                row["camera_id"],
                int(row["frame"]),
                row["polygon_meaning"],
            )
            if key in lookup:
                raise ContractError(f"duplicate CVAT polygon row for {key}")
            lookup[key] = _parse_polygon_json(row["points_json"])
    return lookup


def _resolve_polygon(
    row: Mapping[str, str],
    lookup: Mapping[tuple[str, str, str, str, int, str], list[list[float]]],
    *,
    json_field: str,
    role_field: str,
    candidate_field: str,
    meaning: str,
) -> list[list[float]]:
    if row.get(json_field, "").strip():
        return _parse_polygon_json(row[json_field])
    role = row.get(role_field, "").strip()
    candidate = row.get(candidate_field, "").strip()
    if not role or not candidate:
        return []
    key = (role, candidate, row["window_id"], row["camera_id"], int(row["frame"]), meaning)
    try:
        return copy.deepcopy(lookup[key])
    except KeyError as exc:
        raise ContractError(f"missing CVAT polygon for {key}") from exc


def make_completed_window_manifest(
    *,
    windows_manifest_path: str | Path,
    output_path: str | Path,
    annotator_a_id: str,
    annotator_b_id: str,
    adjudicator_id: str,
    scene: str,
) -> dict[str, Any]:
    manifest = load_json(windows_manifest_path)
    validate_annotation_windows(manifest, require_initial_empty=True)
    completed = copy.deepcopy(manifest)
    completed["human_fields_status"] = "completed"
    completed["completed_for_scene"] = scene
    completed["completed_annotation_protocol"] = {
        "annotator_a_id": annotator_a_id,
        "annotator_b_id": annotator_b_id,
        "adjudicator_id": adjudicator_id,
    }
    for window in completed["windows"]:
        window["assignment"] = [
            {"role": "annotator_a", "actual_annotator_id": annotator_a_id},
            {"role": "annotator_b", "actual_annotator_id": annotator_b_id},
        ]
        window["human_fields"] = {
            key: {
                "status": "completed" if window["scene"] == scene else "not_applicable_to_scene_label_freeze",
                "scene": scene,
                "window_id": window["window_id"],
                "field": key,
            }
            for key in _HUMAN_MANIFEST_FIELDS
        }
    write_json_atomic(output_path, completed)
    validate_annotation_windows(completed, require_initial_empty=False)
    return completed


def assemble_phase9_labels(
    *,
    windows_manifest_path: str | Path,
    completed_windows_output: str | Path,
    output_json: str | Path,
    scene: str,
    annotator_a_id: str,
    annotator_b_id: str,
    adjudicator_id: str,
    polygon_csvs: Sequence[str | Path],
    track_frames_csv: str | Path,
    transitions_csv: str | Path,
    ordering_pairs_csv: str | Path,
    frame_reviews_csv: str | Path,
) -> dict[str, Any]:
    completed_manifest = make_completed_window_manifest(
        windows_manifest_path=windows_manifest_path,
        output_path=completed_windows_output,
        annotator_a_id=annotator_a_id,
        annotator_b_id=annotator_b_id,
        adjudicator_id=adjudicator_id,
        scene=scene,
    )
    lookup = _polygon_lookup(polygon_csvs)

    track_frames = []
    for row in _read_csv(track_frames_csv):
        if not row.get("window_id", "").strip():
            continue
        state = row["state"].strip()
        if state not in _STATES:
            raise ContractError(f"invalid track state {state!r}")
        rear_polygon = _resolve_polygon(
            row,
            lookup,
            json_field="rear_polygon_json",
            role_field="rear_polygon_source_role",
            candidate_field="rear_polygon_candidate_track_id",
            meaning="visible_rear_polygon",
        )
        aperture = _resolve_polygon(
            row,
            lookup,
            json_field="state_aperture_json",
            role_field="state_aperture_source_role",
            candidate_field="state_aperture_candidate_track_id",
            meaning="occluded_state_aperture" if state == "occluded" else "visible_rear_polygon",
        )
        if state == "visible" and not aperture:
            aperture = copy.deepcopy(rear_polygon)
        track_frames.append(
            {
                "window_id": row["window_id"],
                "roster_track_id": row["roster_track_id"],
                "camera_id": row["camera_id"],
                "frame": int(row["frame"]),
                "state": state,
                "rear_polygon": rear_polygon,
                "state_aperture": aperture,
                "evaluable": _parse_bool(row["evaluable"], field="track_frames.evaluable"),
                "annotator_a_response": _parse_json_or_string(row["annotator_a_response"], required=True),
                "annotator_b_response": _parse_json_or_string(row["annotator_b_response"], required=True),
                "adjudication": _parse_json_or_string(row["adjudication"]),
            }
        )

    transitions = []
    for row in _read_csv(transitions_csv):
        if not row.get("window_id", "").strip():
            continue
        label = row["label"].strip()
        if label not in _TRANSITIONS:
            raise ContractError(f"invalid transition label {label!r}")
        transitions.append(
            {
                "window_id": row["window_id"],
                "roster_track_id": row["roster_track_id"],
                "camera_id": row["camera_id"],
                "frame_t": int(row["frame_t"]),
                "frame_t1": int(row["frame_t1"]),
                "label": label,
                "evaluable": _parse_bool(row["evaluable"], field="transitions.evaluable"),
                "annotator_a_response": _parse_json_or_string(row["annotator_a_response"], required=True),
                "annotator_b_response": _parse_json_or_string(row["annotator_b_response"], required=True),
                "adjudication": _parse_json_or_string(row["adjudication"]),
            }
        )

    ordering_pairs = []
    for row in _read_csv(ordering_pairs_csv):
        if not row.get("window_id", "").strip():
            continue
        label = row["label"].strip()
        if label not in _ORDERING_LABELS:
            raise ContractError(f"invalid ordering label {label!r}")
        ordering_pairs.append(
            {
                "window_id": row["window_id"],
                "pair_id": row["pair_id"],
                "camera_id": row["camera_id"],
                "frame": int(row["frame"]),
                "foreground_track_id": row["foreground_track_id"],
                "rear_track_id": row["rear_track_id"],
                "label": label,
                "evaluable": _parse_bool(row["evaluable"], field="ordering_pairs.evaluable"),
                "annotator_a_response": _parse_json_or_string(row["annotator_a_response"], required=True),
                "annotator_b_response": _parse_json_or_string(row["annotator_b_response"], required=True),
                "adjudication": _parse_json_or_string(row["adjudication"]),
            }
        )

    frame_reviews = []
    for row in _read_csv(frame_reviews_csv):
        if not row.get("window_id", "").strip():
            continue
        provenance = _parse_json_or_string(row.get("annotator_provenance", "")) or {
            "annotator_a_id": annotator_a_id,
            "annotator_b_id": annotator_b_id,
            "adjudicator_id": adjudicator_id,
        }
        frame_reviews.append(
            {
                "window_id": row["window_id"],
                "camera_id": row["camera_id"],
                "frame": int(row["frame"]),
                "spatial_complete": _parse_optional_bool(row["spatial_complete"], field="frame_reviews.spatial_complete"),
                "no_evaluable_visible_rear_surface": _parse_optional_bool(row["no_evaluable_visible_rear_surface"], field="frame_reviews.no_evaluable_visible_rear_surface"),
                "unknown_reason": row["unknown_reason"].strip() or None,
                "annotator_provenance": provenance,
            }
        )

    artifact = {
        "schema_version": LABEL_FREEZE_SCHEMA,
        "evidence_type": "human_reference",
        "scene": scene,
        "source_window_manifest": {
            "path": str(Path(completed_windows_output)),
            "sha256": sha256_file(completed_windows_output),
        },
        "annotator_records": [
            {"role": "annotator_a", "annotator_id": annotator_a_id},
            {"role": "annotator_b", "annotator_id": annotator_b_id},
        ],
        "adjudication_record": {
            "status": "completed",
            "adjudicator_id": adjudicator_id,
        },
        "tables": {
            "track_frames": track_frames,
            "ordering_pairs": ordering_pairs,
            "transitions": transitions,
            "frame_reviews": frame_reviews,
        },
        "provenance": {
            "polygon_csvs": [str(Path(path)) for path in polygon_csvs],
            "track_frames_csv": str(Path(track_frames_csv)),
            "transitions_csv": str(Path(transitions_csv)),
            "ordering_pairs_csv": str(Path(ordering_pairs_csv)),
            "frame_reviews_csv": str(Path(frame_reviews_csv)),
        },
    }
    validate_human_label_freeze(artifact, completed_manifest)
    write_json_atomic(output_json, artifact)
    return artifact


__all__ = [
    "assemble_phase9_labels",
    "extract_cvat_polygons",
    "generate_cvat_annotation_templates",
    "image_path",
    "make_completed_window_manifest",
]
