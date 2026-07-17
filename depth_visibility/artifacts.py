"""Transactional, content-verified scientific artifacts."""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import (
    array_semantic_ref,
    canonical_array,
    canonical_json_bytes,
    sha256_file,
    verify_array_ref,
)
from .errors import ArtifactError, SchemaError


def _fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: os.PathLike[str] | str, data: bytes) -> Path:
    """Atomically replace path with bytes and fsync file and parent."""

    target = Path(path)
    if not isinstance(data, bytes):
        raise SchemaError("atomic scientific payload must be bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_parent(target)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return target


def atomic_write_json(path: os.PathLike[str] | str, payload: Any) -> Path:
    """Write canonical scientific JSON with one trailing newline."""

    return atomic_write_bytes(path, canonical_json_bytes(payload) + b"\n")


def _relative_artifact_path(path: Path, relative_to: Path | None) -> str:
    if relative_to is None:
        return path.name
    try:
        relative = path.resolve().relative_to(relative_to.resolve())
    except ValueError as exc:
        raise ArtifactError(f"artifact escapes declared root: {path}") from exc
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtifactError("artifact path is not a safe relative path")
    return relative.as_posix()


def write_canonical_array(
    path: os.PathLike[str] | str,
    array: np.ndarray,
    semantic_key: str,
    *,
    relative_to: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Write deterministic uncompressed NPY and return semantic and file IDs."""

    target = Path(path)
    normalized, _ = canonical_array(array)
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, normalized, version=(1, 0), allow_pickle=False)
    atomic_write_bytes(target, buffer.getvalue())
    reference = array_semantic_ref(normalized, semantic_key)
    return {
        **reference,
        "path": _relative_artifact_path(
            target, None if relative_to is None else Path(relative_to)
        ),
        "file_format": "npy-v1.0",
        "file_bytes": target.stat().st_size,
        "file_sha256": sha256_file(target),
    }


def load_verified_array(
    path: os.PathLike[str] | str,
    reference: Mapping[str, Any],
    *,
    mmap_mode: str | None = None,
) -> np.ndarray:
    """Load an NPY only after file and semantic identities verify."""

    source = Path(path)
    required_file = {"file_format", "file_bytes", "file_sha256"}
    if not required_file.issubset(reference):
        raise ArtifactError("array file reference is incomplete")
    if reference["file_format"] != "npy-v1.0":
        raise ArtifactError("unsupported scientific array file format")
    if not source.is_file() or source.stat().st_size != int(reference["file_bytes"]):
        raise ArtifactError("array file is absent or has the wrong size")
    if sha256_file(source) != reference["file_sha256"]:
        raise ArtifactError("array file hash mismatch")
    try:
        result = np.load(source, allow_pickle=False, mmap_mode=mmap_mode)
    except Exception as exc:
        raise ArtifactError("array file cannot be decoded safely") from exc
    semantic_keys = (
        "semantic_key", "dtype", "shape", "byte_order",
        "byte_sha256", "semantic_sha256",
    )
    if not all(key in reference for key in semantic_keys):
        raise ArtifactError("array semantic reference is incomplete")
    semantic = {key: reference[key] for key in semantic_keys}
    verify_array_ref(np.asarray(result), semantic)
    return result


def build_inventory(
    root: os.PathLike[str] | str,
    paths: Iterable[os.PathLike[str] | str] | None = None,
    *,
    exclude: Iterable[os.PathLike[str] | str] = (),
) -> list[dict[str, Any]]:
    """Return a stable inventory of regular files rooted under root."""

    base = Path(root).resolve()
    excluded = {Path(item).as_posix() for item in exclude}
    candidates = (
        sorted(path for path in base.rglob("*") if path.is_file())
        if paths is None
        else [base / Path(item) for item in paths]
    )
    inventory: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item.as_posix()):
        try:
            relative = candidate.resolve().relative_to(base).as_posix()
        except ValueError as exc:
            raise ArtifactError(f"inventory path escapes root: {candidate}") from exc
        if relative in excluded:
            continue
        if not candidate.is_file():
            raise ArtifactError(f"inventory member is missing: {relative}")
        inventory.append(
            {
                "path": relative,
                "bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    return inventory


def write_terminal_last(
    terminal_path: os.PathLike[str] | str,
    payload: Mapping[str, Any],
    required_paths: Iterable[os.PathLike[str] | str],
    *,
    overwrite: bool = False,
) -> Path:
    """Verify expected outputs and atomically write terminal envelope last."""

    terminal = Path(terminal_path)
    if terminal.exists() and not overwrite:
        raise ArtifactError(f"terminal envelope already exists: {terminal}")
    missing = [str(Path(item)) for item in required_paths if not Path(item).is_file()]
    if missing:
        raise ArtifactError(f"cannot seal incomplete output; missing: {missing}")
    return atomic_write_json(terminal, dict(payload))


__all__ = [
    "atomic_write_bytes",
    "atomic_write_json",
    "build_inventory",
    "load_verified_array",
    "write_canonical_array",
    "write_terminal_last",
]
