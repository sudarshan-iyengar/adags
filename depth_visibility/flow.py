"""Fail-closed optical-flow sidecar adaptation and reciprocal matching."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

import cv2
import numpy as np

from .errors import FlowSemanticsError


@dataclasses.dataclass(frozen=True)
class FlowRecord:
    source_camera: str
    target_camera: str
    source_image: str
    target_image: str
    source_frame: int
    target_frame: int
    direction: str
    dt: float
    height: int
    width: int
    units: str
    pixel_centers: str
    sampling: str
    validity_semantics: str
    occlusion_semantics: str
    generator_revision: str
    source_hashes: tuple[str, ...]
    array_hash: str


_REQUIRED = tuple(field.name for field in dataclasses.fields(FlowRecord))
_DIRECTIONS = {"forward_t_to_t_plus_1", "backward_t_to_t_minus_1"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_flow_manifest(manifest: Mapping[str, Any]) -> FlowRecord:
    missing = [key for key in _REQUIRED if key not in manifest]
    unknown = sorted(set(manifest) - set(_REQUIRED) - {"schema_version"})
    if missing or unknown:
        raise FlowSemanticsError(f"flow manifest missing={missing} unknown={unknown}")
    direction = str(manifest["direction"])
    if direction not in _DIRECTIONS:
        raise FlowSemanticsError("ambiguous or unsupported flow direction")
    if str(manifest["source_camera"]) != str(manifest["target_camera"]):
        raise FlowSemanticsError("temporal flow must remain within one physical camera")
    source_frame = int(manifest["source_frame"])
    target_frame = int(manifest["target_frame"])
    step = target_frame - source_frame
    expected_step = 1 if direction == "forward_t_to_t_plus_1" else -1
    if step != expected_step:
        raise FlowSemanticsError("flow frame order contradicts its declared direction")
    dt = float(manifest["dt"])
    if not math.isfinite(dt) or dt == 0 or math.copysign(1.0, dt) != math.copysign(1.0, step):
        raise FlowSemanticsError("flow dt must be finite, nonzero, and agree with frame direction")
    if manifest["units"] != "pixels" or manifest["pixel_centers"] != "integer":
        raise FlowSemanticsError("flow must use integer-center pixel units")
    if manifest["sampling"] != "bilinear_align_corners_false":
        raise FlowSemanticsError("flow sampling convention mismatch")
    if manifest["validity_semantics"] != "true_means_sample_is_valid":
        raise FlowSemanticsError("flow validity semantics are not the admitted convention")
    if manifest["occlusion_semantics"] != "true_means_not_occluded":
        raise FlowSemanticsError("flow occlusion semantics are not the admitted convention")
    if int(manifest["height"]) <= 0 or int(manifest["width"]) <= 0:
        raise FlowSemanticsError("flow dimensions must be positive")
    if not str(manifest["source_image"]) or not str(manifest["target_image"]):
        raise FlowSemanticsError("flow source/target images are mandatory")
    if not str(manifest["generator_revision"]):
        raise FlowSemanticsError("flow generator revision is mandatory")
    source_hashes = manifest["source_hashes"]
    if (
        not isinstance(source_hashes, (list, tuple))
        or not source_hashes
        or any(_SHA256.fullmatch(str(value)) is None for value in source_hashes)
    ):
        raise FlowSemanticsError("lowercase SHA-256 source hashes are mandatory")
    if _SHA256.fullmatch(str(manifest["array_hash"])) is None:
        raise FlowSemanticsError("flow array hash must be lowercase SHA-256")
    values = dict(manifest)
    values["source_camera"] = str(values["source_camera"])
    values["target_camera"] = str(values["target_camera"])
    values["source_image"] = str(values["source_image"])
    values["target_image"] = str(values["target_image"])
    values["source_frame"] = source_frame
    values["target_frame"] = target_frame
    values["direction"] = direction
    values["dt"] = dt
    values["height"] = int(values["height"])
    values["width"] = int(values["width"])
    values["source_hashes"] = tuple(str(value) for value in source_hashes)
    values["array_hash"] = str(values["array_hash"])
    return FlowRecord(**{key: values[key] for key in _REQUIRED})


def adapt_declared_flow(flow: np.ndarray, record: FlowRecord, output_shape: tuple[int, int]) -> np.ndarray:
    """Resize a declared vector field while preserving pixel displacement units."""

    values = np.asarray(flow, dtype=np.float64)
    if values.shape != (record.height, record.width, 2) or not np.all(np.isfinite(values)):
        raise FlowSemanticsError("flow array shape/finiteness does not match its sealed manifest")
    out_h, out_w = output_shape
    if out_h <= 0 or out_w <= 0:
        raise FlowSemanticsError("output shape must be positive")
    resized = cv2.resize(values, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    resized[..., 0] *= out_w / record.width
    resized[..., 1] *= out_h / record.height
    return resized


def bilinear_flow(flow: np.ndarray, x: float, y: float, valid: np.ndarray | None = None) -> np.ndarray | None:
    values = np.asarray(flow, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 2 or not np.all(np.isfinite(values)):
        raise FlowSemanticsError("flow must be a finite HxWx2 array")
    height, width = values.shape[:2]
    if not math.isfinite(x) or not math.isfinite(y) or x < 0 or y < 0 or x > width - 1 or y > height - 1:
        return None
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != (height, width):
            raise FlowSemanticsError("flow validity mask shape mismatch")
        if not all(mask[yy, xx] for yy, xx in ((y0, x0), (y0, x1), (y1, x0), (y1, x1))):
            return None
    dx, dy = x - x0, y - y0
    return (
        values[y0, x0] * (1 - dx) * (1 - dy)
        + values[y0, x1] * dx * (1 - dy)
        + values[y1, x0] * (1 - dx) * dy
        + values[y1, x1] * dx * dy
    )


def _cycle(
    point_xy: tuple[float, float],
    first_flow: np.ndarray,
    reverse_flow: np.ndarray,
    first_valid: np.ndarray | None,
    reverse_valid: np.ndarray | None,
) -> dict[str, Any]:
    x, y = (float(value) for value in point_xy)
    first = bilinear_flow(first_flow, x, y, first_valid)
    if first is None:
        return {"valid": False, "reason": "invalid_forward"}
    destination = np.array([x, y], dtype=np.float64) + first
    reverse = bilinear_flow(reverse_flow, float(destination[0]), float(destination[1]), reverse_valid)
    if reverse is None:
        return {"valid": False, "reason": "invalid_backward"}
    cycle = float(np.linalg.norm(first + reverse))
    return {
        "valid": True,
        "destination": destination,
        "cycle_error": cycle,
        "accepted": cycle <= 1.5,
    }


def forward_backward_cycle(
    point_xy: tuple[float, float],
    forward: np.ndarray,
    backward: np.ndarray,
    *,
    forward_valid: np.ndarray | None = None,
    backward_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    return _cycle(point_xy, forward, backward, forward_valid, backward_valid)


def _xy(node: Mapping[str, Any]) -> np.ndarray:
    point = np.asarray(node["xy"], dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise FlowSemanticsError("temporal node xy must be a finite 2-vector")
    return point


def reciprocal_node_flow_matches(
    source_nodes: Iterable[Mapping[str, Any]],
    destination_nodes: Iterable[Mapping[str, Any]],
    forward: np.ndarray,
    backward: np.ndarray,
    *,
    forward_valid: np.ndarray | None = None,
    backward_valid: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Mutual nearest matching from independently evaluated forward/backward flow."""

    sources = sorted(source_nodes, key=lambda item: str(item["node_id"]))
    destinations = sorted(destination_nodes, key=lambda item: str(item["node_id"]))
    if len({str(item["node_id"]) for item in sources}) != len(sources):
        raise FlowSemanticsError("source node IDs must be unique")
    if len({str(item["node_id"]) for item in destinations}) != len(destinations):
        raise FlowSemanticsError("destination node IDs must be unique")

    forward_choices: dict[str, tuple[Mapping[str, Any], dict[str, Any]]] = {}
    for source in sources:
        source_xy = _xy(source)
        cycle = _cycle(
            tuple(source_xy), forward, backward, forward_valid, backward_valid
        )
        if not cycle.get("accepted"):
            continue
        predicted = np.asarray(cycle["destination"])
        candidates: list[tuple[tuple[float, int, int, str], Mapping[str, Any]]] = []
        for destination in destinations:
            destination_xy = _xy(destination)
            distance = float(np.linalg.norm(predicted - destination_xy))
            if distance <= 2.0:
                x, y = destination_xy
                candidates.append(
                    ((distance, int(y), int(x), str(destination["node_id"])), destination)
                )
        if candidates:
            _, selected = min(candidates, key=lambda item: item[0])
            forward_choices[str(source["node_id"])] = (selected, cycle)

    reverse_choices: dict[str, tuple[Mapping[str, Any], dict[str, Any]]] = {}
    for destination in destinations:
        destination_xy = _xy(destination)
        cycle = _cycle(
            tuple(destination_xy), backward, forward, backward_valid, forward_valid
        )
        if not cycle.get("accepted"):
            continue
        predicted = np.asarray(cycle["destination"])
        candidates: list[tuple[tuple[float, int, int, str], Mapping[str, Any]]] = []
        for source in sources:
            source_xy = _xy(source)
            distance = float(np.linalg.norm(predicted - source_xy))
            if distance <= 2.0:
                x, y = source_xy
                candidates.append(((distance, int(y), int(x), str(source["node_id"])), source))
        if candidates:
            _, selected = min(candidates, key=lambda item: item[0])
            reverse_choices[str(destination["node_id"])] = (selected, cycle)

    matches = []
    for source_id in sorted(forward_choices):
        destination, forward_cycle = forward_choices[source_id]
        destination_id = str(destination["node_id"])
        reverse_choice = reverse_choices.get(destination_id)
        if reverse_choice is not None and str(reverse_choice[0]["node_id"]) == source_id:
            matches.append(
                {
                    "source_id": source_id,
                    "destination_id": destination_id,
                    "cycle_error": max(
                        float(forward_cycle["cycle_error"]),
                        float(reverse_choice[1]["cycle_error"]),
                    ),
                }
            )
    return matches


__all__ = [
    "FlowRecord",
    "adapt_declared_flow",
    "bilinear_flow",
    "forward_backward_cycle",
    "reciprocal_node_flow_matches",
    "validate_flow_manifest",
]
