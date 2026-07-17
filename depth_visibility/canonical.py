"""Canonical scientific payload encoding and content identities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .errors import NonFiniteError, SchemaError


ARRAY_ID_DOMAIN = "csvl-v1/array"


def binary64_hex(value: float) -> str:
    """Return the lowercase big-endian IEEE-754 binary64 identity token."""

    if isinstance(value, (bool, np.bool_)):
        raise SchemaError("boolean is not a binary64 value")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchemaError("value cannot be represented as binary64") from exc
    if not math.isfinite(converted):
        raise NonFiniteError("canonical payload contains a nonfinite float")
    return struct.pack(">d", converted).hex()


def _nfc(value: str, *, what: str = "string") -> str:
    normalized = unicodedata.normalize("NFC", value)
    if normalized != value:
        raise SchemaError(f"{what} is not already NFC normalized")
    return value


def canonicalize(value: Any) -> Any:
    """Convert a supported value to the csvl-cjson-v1 JSON value tree.

    Scientific arrays must use an explicit semantic reference. Implicit ndarray
    embedding is rejected because it loses dtype, shape, and byte-order identity.
    """

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return binary64_hex(float(value))
    if isinstance(value, str):
        return _nfc(value)
    if isinstance(value, (bytes, bytearray, memoryview, np.ndarray)):
        raise SchemaError("binary data and ndarrays require an explicit semantic reference")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaError("canonical JSON mapping keys must be strings")
            key = _nfc(key, what="mapping key")
            if key in result:
                raise SchemaError(f"duplicate canonical mapping key: {key!r}")
            result[key] = canonicalize(item)
        return result
    if isinstance(value, Sequence):
        return [canonicalize(item) for item in value]
    raise SchemaError(f"unsupported canonical payload type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a value as compact, sorted-key, NFC UTF-8 canonical JSON."""

    canonical = canonicalize(value)
    return json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def domain_id(domain_separator: str, payload: Any) -> str:
    """Hash domain_separator + NUL + canonical_payload with SHA-256."""

    if not isinstance(domain_separator, str) or not domain_separator:
        raise SchemaError("domain separator must be a nonempty string")
    domain = _nfc(domain_separator, what="domain separator").encode("utf-8")
    return hashlib.sha256(domain + b"\0" + canonical_json_bytes(payload)).hexdigest()


def _hash_reader(reader: BinaryIO, chunk_size: int) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = reader.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Stream a regular file SHA-256."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise SchemaError("chunk_size must be a positive integer")
    source = Path(path)
    if not source.is_file():
        raise SchemaError(f"not a regular file: {source}")
    with source.open("rb") as handle:
        return _hash_reader(handle, chunk_size)


def canonical_array(array: np.ndarray) -> tuple[np.ndarray, str]:
    """Return a finite, C-contiguous, explicitly little-endian array."""

    if not isinstance(array, np.ndarray):
        raise SchemaError("scientific payload arrays require numpy.ndarray")
    if array.dtype.hasobject or array.dtype.fields is not None:
        raise SchemaError("object and structured arrays are not scientific payload arrays")
    if array.dtype.kind not in "biufc":
        raise SchemaError(f"unsupported scientific array dtype: {array.dtype}")
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        raise NonFiniteError("scientific array contains NaN or infinity")
    dtype = array.dtype
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<")
    normalized = np.ascontiguousarray(array, dtype=dtype)
    byte_order = "not_applicable" if dtype.itemsize == 1 else "little"
    return normalized, byte_order


def array_semantic_ref(array: np.ndarray, semantic_key: str) -> dict[str, Any]:
    """Return the explicit identity reference for an in-memory array."""

    if not isinstance(semantic_key, str) or not semantic_key:
        raise SchemaError("array semantic_key must be a nonempty string")
    semantic_key = _nfc(semantic_key, what="array semantic key")
    normalized, byte_order = canonical_array(array)
    byte_sha256 = hashlib.sha256(normalized.tobytes(order="C")).hexdigest()
    identity = {
        "semantic_key": semantic_key,
        "dtype": normalized.dtype.str,
        "shape": [int(size) for size in normalized.shape],
        "byte_order": byte_order,
        "byte_sha256": byte_sha256,
    }
    return {**identity, "semantic_sha256": domain_id(ARRAY_ID_DOMAIN, identity)}


def verify_array_ref(array: np.ndarray, reference: Mapping[str, Any]) -> None:
    """Fail unless an array exactly matches its declared semantic reference."""

    if not isinstance(reference, Mapping):
        raise SchemaError("array reference must be a mapping")
    expected_keys = {
        "semantic_key", "dtype", "shape", "byte_order",
        "byte_sha256", "semantic_sha256",
    }
    if set(reference) != expected_keys:
        raise SchemaError("array reference has missing or unknown keys")
    actual = array_semantic_ref(array, str(reference["semantic_key"]))
    if actual != dict(reference):
        raise SchemaError("array does not match its declared semantic reference")


__all__ = [
    "ARRAY_ID_DOMAIN",
    "array_semantic_ref",
    "binary64_hex",
    "canonical_array",
    "canonical_json_bytes",
    "canonicalize",
    "domain_id",
    "sha256_file",
    "verify_array_ref",
]
