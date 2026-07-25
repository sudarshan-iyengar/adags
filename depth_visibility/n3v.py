"""Read-only, side-effect-free parsing of calibrated N3V scene metadata."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .camera import (
    camera_center,
    intrinsics_matrix,
    opengl_c2w_to_opencv_w2c,
    validate_calibration,
)
from .canonical import binary64_hex, domain_id, sha256_file
from .errors import ProvenanceError, SchemaError
from .schema import assert_finite_tree


_RECORD_PATTERN = re.compile(r"(?P<camera>cam[0-9]+)_(?P<frame>[0-9]+)$")
CANONICAL_RECORD_ID_DOMAIN = "csvl-v1/n3v-record-index"


@dataclass(frozen=True)
class CameraRecord:
    scene: str
    split: str
    camera_id: str
    frame: int
    time: float
    width: int
    height: int
    K: np.ndarray
    c2w_opengl: np.ndarray
    w2c_opencv: np.ndarray
    metadata_path: Path
    file_stem: str
    image_path: Path | None
    image_sha256: str | None

    @property
    def key(self) -> tuple[str, int, float]:
        return self.camera_id, self.frame, self.time


@dataclass(frozen=True)
class SceneIndex:
    scene: str
    root: Path
    records: Mapping[str, tuple[CameraRecord, ...]]
    source_sha256: Mapping[str, str]
    canonical_record_identity: Mapping[str, str]

    def split(self, name: str) -> tuple[CameraRecord, ...]:
        try:
            return self.records[name]
        except KeyError as exc:
            raise ProvenanceError(f"unknown N3V split: {name}") from exc

    def by_camera_frame(self, split: str) -> Mapping[tuple[str, int], CameraRecord]:
        result: dict[tuple[str, int], CameraRecord] = {}
        for record in self.split(split):
            key = (record.camera_id, record.frame)
            if key in result:
                raise ProvenanceError(f"duplicate N3V camera/frame record: {key}")
            result[key] = record
        return MappingProxyType(result)


def parse_camera_id(value: str) -> str:
    """Extract a canonical camera ID from a frame stem or camera token."""

    if not isinstance(value, str):
        raise SchemaError("camera identifier source must be a string")
    token = Path(value).stem
    if re.fullmatch(r"cam[0-9]+", token):
        return token
    match = _RECORD_PATTERN.search(token)
    if match is None:
        raise SchemaError(f"cannot parse N3V camera ID: {value!r}")
    return match.group("camera")


def parse_frame(value: str) -> int:
    """Extract a nonnegative integer frame from a frame stem."""

    if not isinstance(value, str):
        raise SchemaError("frame identifier source must be a string")
    match = _RECORD_PATTERN.search(Path(value).stem)
    if match is None:
        raise SchemaError(f"cannot parse N3V frame: {value!r}")
    return int(match.group("frame"))


def _load_metadata(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=lambda token: (_ for _ in ()).throw(
                SchemaError(f"nonfinite JSON constant {token} in {path}")
            ))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot read N3V metadata: {path}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("frames"), list):
        raise SchemaError(f"N3V metadata has no frame list: {path}")
    assert_finite_tree(payload)
    return payload


def _resolve_image(stem: Path) -> Path:
    candidates = [stem] if stem.suffix else [stem.with_suffix(ext) for ext in (".png", ".jpg", ".jpeg")]
    existing = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise ProvenanceError(f"expected exactly one image for N3V stem {stem}, found {len(existing)}")
    return existing[0]


def _canonical_record_identity(records: tuple[CameraRecord, ...]) -> str:
    payload = [
        {
            "camera_id": record.camera_id,
            "frame": record.frame,
            "time_binary64": binary64_hex(record.time),
            "file_stem": record.file_stem,
        }
        for record in sorted(records, key=lambda item: (item.camera_id, item.frame, item.time))
    ]
    return domain_id(CANONICAL_RECORD_ID_DOMAIN, payload)


def _parse_split(
    scene: str,
    scene_root: Path,
    split: str,
    metadata_name: str,
    *,
    expose_images: bool,
    hash_images: bool,
) -> tuple[tuple[CameraRecord, ...], str]:
    metadata_path = (scene_root / metadata_name).resolve()
    payload = _load_metadata(metadata_path)
    width = int(payload["w"])
    height = int(payload["h"])
    top_k = intrinsics_matrix(payload["fl_x"], payload["fl_y"], payload["cx"], payload["cy"])
    records: list[CameraRecord] = []
    keys: set[tuple[str, int, float]] = set()
    for raw in payload["frames"]:
        if not isinstance(raw, Mapping):
            raise SchemaError("N3V frame entry must be an object")
        file_stem = str(raw["file_path"])
        camera_id = parse_camera_id(file_stem)
        frame = parse_frame(file_stem)
        time = float(raw["time"])
        key = (camera_id, frame, time)
        if key in keys:
            raise ProvenanceError(f"duplicate N3V camera/frame/time record: {key}")
        keys.add(key)
        K = intrinsics_matrix(
            raw.get("fl_x", top_k[0, 0]), raw.get("fl_y", top_k[1, 1]),
            raw.get("cx", top_k[0, 2]), raw.get("cy", top_k[1, 2]),
        )
        c2w = np.asarray(raw["transform_matrix"], dtype=np.float64)
        w2c = opengl_c2w_to_opencv_w2c(c2w)
        distortion = raw.get("distortion", payload.get("distortion"))
        rolling_shutter = raw.get("rolling_shutter", payload.get("rolling_shutter", False))
        validate_calibration(
            K, w2c, width, height, distortion=distortion,
            rolling_shutter=rolling_shutter,
        )
        resolved_image: Path | None = None
        image_hash: str | None = None
        if expose_images or hash_images:
            resolved_image = _resolve_image(scene_root / file_stem)
            if hash_images:
                image_hash = sha256_file(resolved_image)
        records.append(
            CameraRecord(
                scene=scene, split=split, camera_id=camera_id, frame=frame, time=time,
                width=width, height=height, K=K, c2w_opengl=c2w,
                w2c_opencv=w2c, metadata_path=metadata_path, file_stem=file_stem,
                image_path=resolved_image if expose_images else None,
                image_sha256=image_hash,
            )
        )
    records.sort(key=lambda item: (item.camera_id, item.frame, item.time))
    return tuple(records), sha256_file(metadata_path)


def _validate_synchronization(records: tuple[CameraRecord, ...], tolerance: float) -> None:
    by_frame: dict[int, list[float]] = {}
    for record in records:
        by_frame.setdefault(record.frame, []).append(record.time)
    for frame, times in by_frame.items():
        if max(times) - min(times) > tolerance:
            raise ProvenanceError(f"N3V frame {frame} exceeds timestamp tolerance")


def load_scene_index(
    scene_root: str | Path,
    *,
    scene: str | None = None,
    expose_train_images: bool = True,
    expose_test_images: bool = False,
    hash_train_images: bool = False,
    timestamp_tolerance_seconds: float = 1e-6,
) -> SceneIndex:
    """Parse train/test metadata without trainer, image, Torch, or CUDA side effects.

    Test RGB paths remain absent unless ``expose_test_images`` is explicitly true.
    Train-image hashing is opt-in because full-scene hashing belongs on Slurm.
    """

    root = Path(scene_root).resolve()
    scene_name = root.name if scene is None else scene
    train, train_sha = _parse_split(
        scene_name, root, "train", "transforms_train.json",
        expose_images=expose_train_images, hash_images=hash_train_images,
    )
    test, test_sha = _parse_split(
        scene_name, root, "test", "transforms_test.json",
        expose_images=expose_test_images, hash_images=False,
    )
    train_keys = {(item.camera_id, item.frame) for item in train}
    test_keys = {(item.camera_id, item.frame) for item in test}
    overlap = train_keys & test_keys
    if overlap:
        raise ProvenanceError(f"N3V train/test camera-frame overlap: {sorted(overlap)[:3]}")
    _validate_synchronization(train, timestamp_tolerance_seconds)
    _validate_synchronization(test, timestamp_tolerance_seconds)
    records = MappingProxyType({"train": train, "test": test})
    source_hashes = MappingProxyType({"train": train_sha, "test": test_sha})
    identities = MappingProxyType(
        {"train": _canonical_record_identity(train), "test": _canonical_record_identity(test)}
    )
    return SceneIndex(scene_name, root, records, source_hashes, identities)


def validate_split_binding(
    index: SceneIndex,
    split_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate source hashes/count/cameras and disclose legacy identity status.

    The tracked v1 manifest does not define how ``record_identity_sha256`` was
    encoded. Source-file SHA-256 already binds every record, so this function
    must not falsely claim to reproduce that legacy field. It emits a separately
    named, domain-separated canonical runtime identity instead.
    """

    if split_manifest.get("schema_version") != "n3v-split-v1":
        raise SchemaError("unexpected N3V split manifest schema")
    try:
        scene_entry = split_manifest["scenes"][index.scene]
    except (KeyError, TypeError) as exc:
        raise ProvenanceError(f"scene absent from N3V split manifest: {index.scene}") from exc
    reports: dict[str, Any] = {}
    for split in ("train", "test"):
        expected = scene_entry[split]
        records = index.split(split)
        actual_cameras = sorted({item.camera_id for item in records})
        if int(expected["record_count"]) != len(records):
            raise ProvenanceError(f"{split} record count mismatch")
        if list(expected["camera_ids"]) != actual_cameras:
            raise ProvenanceError(f"{split} camera set mismatch")
        if expected["source_sha256"] != index.source_sha256[split]:
            raise ProvenanceError(f"{split} metadata SHA-256 mismatch")
        expected_source = "transforms_train.json" if split == "train" else "transforms_test.json"
        if expected["source_path"] != expected_source:
            raise ProvenanceError(f"{split} source path mismatch")
        reports[split] = {
            "source_sha256": index.source_sha256[split],
            "record_count": len(records),
            "camera_ids": actual_cameras,
            "canonical_record_identity_domain": CANONICAL_RECORD_ID_DOMAIN,
            "canonical_record_identity_sha256": index.canonical_record_identity[split],
            "legacy_record_identity_sha256": expected.get("record_identity_sha256"),
            "legacy_record_identity_status": "unverified_encoder",
        }
    return MappingProxyType(reports)


def compute_r_scene(records: tuple[CameraRecord, ...] | list[CameraRecord]) -> float:
    """Median camera-center radius over one pose per train camera."""

    by_camera: dict[str, np.ndarray] = {}
    for record in sorted(records, key=lambda item: (item.camera_id, item.frame, item.time)):
        if record.split != "train":
            raise ProvenanceError("R_scene may use transforms-train cameras only")
        by_camera.setdefault(record.camera_id, camera_center(record.w2c_opencv))
    if not by_camera:
        raise ProvenanceError("cannot compute R_scene from no training cameras")
    centers = np.stack([by_camera[key] for key in sorted(by_camera)])
    center_median = np.sort(centers, axis=0)[(len(centers) - 1) // 2]
    distances = np.linalg.norm(centers - center_median[None, :], axis=1)
    radius = float(np.median(distances))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ProvenanceError("R_scene is nonpositive or nonfinite")
    return radius


__all__ = [
    "CANONICAL_RECORD_ID_DOMAIN",
    "CameraRecord",
    "SceneIndex",
    "compute_r_scene",
    "load_scene_index",
    "parse_camera_id",
    "parse_frame",
    "validate_split_binding",
]
