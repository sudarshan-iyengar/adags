"""DiVa-360 -> ADAGS Blender-convention scene schema (M1 census converter).

DiVa-360 (arXiv 2307.16897, CVPR24 Highlight; 53-camera 360-degree surround
rig, 120 fps, fg/bg masks, MIT license -- facts recorded in
``research-wiki/operations/loop2-sweep-2026-08.md`` SF and
``research-wiki/gap_map.md``) ships, per sequence, ``transforms_train.json``
/ ``transforms_test.json`` / ``transforms_val.json`` in the NeRF-synthetic
convention ``scene/dataset_readers.py::readCamerasFromTransforms`` already
reads (dispatch: ``scene/__init__.py`` selects the "Blender" loader whenever
a ``transforms_train.json`` is present, with ``args.extension`` defaulting
to ``".png"`` -- ``arguments/__init__.py:57``).

Direct inspection on Apollo (2026-08-11, ``unlock`` and ``battery``
sequences) established the exact shipped schema:

- Each ``transforms_*.json`` top level is ``{"frames": [...]}`` -- there is
  NO top-level ``camera_angle_x`` or ``fl_x``/``fl_y``/``cx``/``cy``.
- Every frame entry carries a COMPLETE per-frame intrinsics block
  (``camera_angle_x``, ``camera_angle_y``, ``fl_x``, ``fl_y``, ``cx``,
  ``cy``, ``w``, ``h``, plus ``sharpness`` and ``is_fisheye``), so
  ``readCamerasFromTransforms`` always takes its first
  (``'fl_x' in frame``) branch -- the top-level-``camera_angle_x`` fallback
  branch (which would otherwise ``NameError`` on an undefined ``fovx``) is
  never reached.
- Each shipped ``transforms_*.json`` is a SINGLE-INSTANT multiview
  snapshot: every frame entry in one file references the SAME embedded
  8-digit frame index (``battery`` uses frame index 1000, ``unlock`` uses
  frame index 0 -- the "sharpest" per-sequence instant, not frame 0
  universally).
- There is no ``time`` key anywhere in the shipped JSON.
- ``file_path`` is already extension-less and POSIX-relative, e.g.
  ``"undist/cam01/00001000"`` -- exactly the
  ``os.path.join(path, frame["file_path"] + extension)`` contract.

This module defines the frozen conversion schema, its fail-closed
validators, and pure parsing/mapping helpers that
``scripts/diva360_to_blender.py`` uses. It has no filesystem or archive I/O
of its own (that lives in the script) so it stays importable and
unit-testable without Apollo access, and it is stdlib-only (no numpy, no
torch) so it is safe to import from anywhere.

Held-out-camera handling (``configs/elgs/prereg_m1_census_v1.json``
``held_out_camera_rule``: camera id === 0 mod 4): that exclusion binds
tracking/observability INPUTS downstream (tracker seeds, observability
inputs, per the M0/M1 implementation plan S11.1 item 4), never the
converted scene layout itself. ``held_out_camera_ids`` below exists ONLY so
callers can ANNOTATE provenance with the set a downstream consumer would
exclude -- nothing in this module or the converter filters emitted frames
by it; the scene carries every camera the source JSON has.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

from depth_visibility.errors import ContractError, ProvenanceError, SchemaError

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

#: DiVa-360 capture rate (paper fact; research-wiki/gap_map.md Loop-2 update
#: and research-wiki/operations/elgs-m0-m1-implementation-plan.md S11.1).
DEFAULT_FPS = 120.0

#: ``scene/dataset_readers.py::readCamerasFromTransforms`` /
#: ``readNerfSyntheticInfo`` both default ``extension`` to this
#: (``arguments/__init__.py:57``); the shipped ``file_path`` values omit the
#: extension exactly the way the reader expects it appended back.
DEFAULT_EXTENSION = ".png"

#: ``configs/elgs/prereg_m1_census_v1.json`` -> ``scene_selection`` ->
#: ``held_out_camera_rule``: "every camera id congruent to 0 mod 4 is held
#: out from tracking/observability and reserved for evaluation".
HELD_OUT_MODULUS = 4
HELD_OUT_RESIDUE = 0

SCENE_SCHEMA = "diva360-adags-scene-v1"
PROVENANCE_SCHEMA = "diva360-to-blender-provenance-v1"

#: Splits the converter maps 1:1. ``readNerfSyntheticInfo`` always reads
#: ``transforms_train.json`` and (``transforms_test.json`` unless the scene
#: path ends in ``lego``, which no DiVa-360 sequence name does) --
#: ``transforms_val.json`` is shipped but not required by the loader.
REQUIRED_SPLITS = ("train", "test")
OPTIONAL_SPLITS = ("val",)
ALL_SPLITS = REQUIRED_SPLITS + OPTIONAL_SPLITS

_SPLIT_FILENAMES = {
    "train": "transforms_train.json",
    "test": "transforms_test.json",
    "val": "transforms_val.json",
}

#: ``file_path`` stems are 8-digit zero-padded DiVa-360 frame indices, e.g.
#: ``"undist/cam01/00001000"``.
_FRAME_INDEX_RE = re.compile(r"(\d+)$")
_CAMERA_ID_RE = re.compile(r"cam(\d+)", re.IGNORECASE)

_REQUIRED_FRAME_KEYS = ("file_path", "transform_matrix")
_INTRINSIC_KEYS = ("fl_x", "fl_y", "cx", "cy")


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def split_filename(split: str) -> str:
    """The shipped filename for a split name (``"train"``/``"test"``/``"val"``)."""

    if split not in _SPLIT_FILENAMES:
        raise SchemaError(f"unknown DiVa-360 split: {split!r}")
    return _SPLIT_FILENAMES[split]


def frame_relative_path(file_path: str, extension: str = DEFAULT_EXTENSION) -> str:
    """The on-disk relative path a frame entry resolves to.

    Reader contract (``scene/dataset_readers.py::readCamerasFromTransforms``,
    line ~387): ``os.path.join(path, frame["file_path"] + extension)``.
    """

    if not isinstance(file_path, str) or not file_path:
        raise SchemaError("frame file_path must be a non-empty string")
    if "\\" in file_path:
        raise SchemaError(f"frame file_path must use POSIX separators: {file_path!r}")
    if file_path.startswith("/") or ".." in file_path.split("/"):
        raise SchemaError(f"frame file_path must be a safe relative path: {file_path!r}")
    return file_path + extension


def split_top_level_dir(relative_path: str) -> tuple[str, str]:
    """Split ``"top/rest/of/path"`` into ``("top", "rest/of/path")``."""

    parts = relative_path.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise SchemaError(
            f"expected a top-level-directory-relative path: {relative_path!r}"
        )
    return parts[0], parts[1]


def parse_camera_id(text: str) -> int:
    """Extract the integer camera id out of a ``camNN`` path component."""

    match = _CAMERA_ID_RE.search(text)
    if match is None:
        raise SchemaError(f"cannot parse a DiVa-360 camera id out of: {text!r}")
    return int(match.group(1))


def parse_frame_index(file_path: str) -> int:
    """Extract the embedded DiVa-360 frame index from a ``file_path`` stem."""

    stem = file_path.rsplit("/", 1)[-1]
    match = _FRAME_INDEX_RE.search(stem)
    if match is None:
        raise SchemaError(f"cannot parse a DiVa-360 frame index out of: {file_path!r}")
    return int(match.group(1))


def frame_index_to_time(frame_index: int, fps: float) -> float:
    """DiVa frame index -> ADAGS ``time`` convention: ``frame_index / fps``.

    Mirrors ``scripts/n3v2blender.py``'s own convention exactly (there:
    ``'time': int(im...[-4:]) / 30.``) with DiVa-360's frame index and its
    120 fps capture rate (``DEFAULT_FPS``) in place of N3V's frame suffix
    and 30 fps.
    """

    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise SchemaError(f"fps must be a positive number, got {fps!r}")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise SchemaError(f"frame_index must be a non-negative integer, got {frame_index!r}")
    return frame_index / float(fps)


def held_out_camera_ids(camera_ids: Iterable[int]) -> set[int]:
    """The ``prereg_m1_census_v1.json`` ``held_out_camera_rule`` subset.

    ANNOTATION ONLY -- see module docstring. Never used to filter output.
    """

    return {cid for cid in camera_ids if cid % HELD_OUT_MODULUS == HELD_OUT_RESIDUE}


# ---------------------------------------------------------------------------
# Fail-closed validation
# ---------------------------------------------------------------------------


def _validate_transform_matrix(value: Any, *, where: str) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise SchemaError(f"{where}: transform_matrix must be a 4x4 nested list")
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise SchemaError(f"{where}: transform_matrix must be a 4x4 nested list")
        for entry in row:
            if isinstance(entry, bool) or not isinstance(entry, (int, float)):
                raise SchemaError(f"{where}: transform_matrix entries must be numeric")
            if not math.isfinite(float(entry)):
                raise SchemaError(f"{where}: transform_matrix contains a non-finite value")


def validate_frame_entry(
    frame: Mapping[str, Any],
    *,
    index: int,
    extension: str = DEFAULT_EXTENSION,
    allow_time: bool = False,
) -> dict:
    """Fail-closed single-frame validation; returns the frame as a plain dict.

    ``allow_time=False`` (the default) validates a shipped DiVa-360 source
    frame, which must NOT already carry a ``time`` key. ``allow_time=True``
    validates a converted output frame, which MUST carry a finite numeric
    ``time``.
    """

    if not isinstance(frame, Mapping):
        raise SchemaError(f"frame[{index}] must be an object")
    for key in _REQUIRED_FRAME_KEYS:
        if key not in frame:
            raise SchemaError(f"frame[{index}] missing required key {key!r}")
    frame_relative_path(frame["file_path"], extension)  # raises on malformed file_path
    _validate_transform_matrix(frame["transform_matrix"], where=f"frame[{index}]")

    has_full_intrinsics = all(key in frame for key in _INTRINSIC_KEYS)
    if not has_full_intrinsics and "camera_angle_x" not in frame:
        raise SchemaError(
            f"frame[{index}] has neither per-frame fl_x/fl_y/cx/cy nor "
            "camera_angle_x -- readCamerasFromTransforms has no fallback for "
            "a DiVa-360 entry missing both (it would NameError on the "
            "undefined top-level fovx)"
        )

    if allow_time:
        if "time" not in frame:
            raise SchemaError(f"frame[{index}] missing required 'time' key (converted output)")
        time_value = frame["time"]
        if isinstance(time_value, bool) or not isinstance(time_value, (int, float)):
            raise SchemaError(f"frame[{index}] 'time' must be numeric")
        if not math.isfinite(float(time_value)):
            raise SchemaError(f"frame[{index}] 'time' must be finite")
    elif "time" in frame:
        raise SchemaError(
            f"frame[{index}] unexpectedly already has a 'time' key -- DiVa-360 "
            "never ships one; a converter run must not double-stamp a "
            "previously converted frame"
        )
    return dict(frame)


def validate_transforms_payload(
    payload: Any, *, extension: str = DEFAULT_EXTENSION, allow_time: bool = False
) -> dict:
    """Fail-closed validation of one ``transforms_*.json`` payload."""

    if not isinstance(payload, Mapping):
        raise SchemaError("transforms payload must be a JSON object")
    if "frames" not in payload or not isinstance(payload["frames"], list):
        raise SchemaError("transforms payload missing a 'frames' list")
    if not payload["frames"]:
        raise ContractError("transforms payload has zero frames")
    seen_paths: set[str] = set()
    for index, frame in enumerate(payload["frames"]):
        validate_frame_entry(frame, index=index, extension=extension, allow_time=allow_time)
        relative = frame_relative_path(frame["file_path"], extension)
        if relative in seen_paths:
            raise ContractError(f"duplicate frame file_path in transforms payload: {relative}")
        seen_paths.add(relative)
    return dict(payload)


def stamp_frame_time(
    frame: Mapping[str, Any], *, fps: float, extension: str = DEFAULT_EXTENSION, index: int = 0
) -> dict:
    """Return a copy of ``frame`` with the derived ADAGS ``time`` key added.

    See ``frame_index_to_time`` for the mapping decision this applies.
    """

    validated = validate_frame_entry(frame, index=index, extension=extension, allow_time=False)
    frame_index = parse_frame_index(validated["file_path"])
    stamped = dict(validated)
    stamped["time"] = frame_index_to_time(frame_index, fps)
    return stamped


# ---------------------------------------------------------------------------
# MANIFEST.sha256 parsing (acquisition provenance -- see
# research-wiki/operations/elgs-m1-census-record.md "Data acquisition")
# ---------------------------------------------------------------------------


def parse_sha256_manifest(text: str) -> dict[str, str]:
    """Parse a ``sha256sum``-style manifest (``MANIFEST.sha256``).

    Observed format on Apollo (2026-08-11): lines of
    ``"<64-hex-digest>  zips/<sequence>.zip"`` (two spaces; GNU
    ``sha256sum`` text-mode convention). A leading ``*`` (binary-mode
    marker) on the path is tolerated and stripped.
    """

    entries: dict[str, str] = {}
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise SchemaError(
                f"MANIFEST.sha256 line {lineno} is not '<hash>  <path>': {raw_line!r}"
            )
        digest, path = parts
        digest = digest.strip()
        path = path.strip().lstrip("*")
        if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise SchemaError(
                f"MANIFEST.sha256 line {lineno} has a malformed sha256 digest: {digest!r}"
            )
        digest = digest.lower()
        if path in entries and entries[path] != digest:
            raise ContractError(f"MANIFEST.sha256 has conflicting digests for {path!r}")
        entries[path] = digest
    return entries


def lookup_source_sha256(manifest_entries: Mapping[str, str], sequence_name: str) -> str:
    """The upstream zip hash for ``sequence_name`` (``zips/<sequence_name>.zip``)."""

    key = f"zips/{sequence_name}.zip"
    if key not in manifest_entries:
        raise ProvenanceError(
            f"MANIFEST.sha256 has no entry for {key!r} -- cannot establish source "
            "provenance for this sequence"
        )
    return manifest_entries[key]


__all__ = [
    "ALL_SPLITS",
    "DEFAULT_EXTENSION",
    "DEFAULT_FPS",
    "HELD_OUT_MODULUS",
    "HELD_OUT_RESIDUE",
    "OPTIONAL_SPLITS",
    "PROVENANCE_SCHEMA",
    "REQUIRED_SPLITS",
    "SCENE_SCHEMA",
    "frame_index_to_time",
    "frame_relative_path",
    "held_out_camera_ids",
    "lookup_source_sha256",
    "parse_camera_id",
    "parse_frame_index",
    "parse_sha256_manifest",
    "split_filename",
    "split_top_level_dir",
    "stamp_frame_time",
    "validate_frame_entry",
    "validate_transforms_payload",
]
