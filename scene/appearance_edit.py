"""Directional donor payload reuse: the pointer-column edit (CCR v3).

The consolidation pass (scripts/consolidate_packets.py) is a
POST-TRAINING structural edit: the deployed B2 state differs from the
trained B1 state by a per-row source pointer column and a mode flag, and
by nothing else (ccr-method-2026-08-20 §3.2). This module owns that
edit: the pure composition rules, the sidecar format, and apply/clear
helpers. It imports nothing from the trainer.

Modes (frozen clause 2 — arms are never mixed within a deployed state):
- "dc":      recipient rows render with the DONOR row's DC / base
             radiance and keep their own higher-order SH (the original
             primary arm, FALSIFIED 2026-08-20);
- "full":    recipient rows render with the donor's full SH (ablation);
- "opacity": recipient rows render with the donor's RAW OPACITY LOGIT
             (the replacement payload under test).

One pointer column serves every mode; exactly one payload is redirected
per deployed state, selected by the mode flag. The appearance modes are
composed by :func:`compose_shared_features` and the opacity mode by
:func:`compose_shared_opacity`; consumers ask
:func:`redirects_features` / :func:`redirects_opacity` which one applies
so that a pointer installed for one payload is an exact passthrough for
the other.
"""

from __future__ import annotations

import hashlib

import torch

from depth_visibility.errors import ContractError

APPEARANCE_EDIT_SCHEMA = "adags-appearance-edit-v1"
#: The SH/appearance arms — the only modes :func:`compose_shared_features`
#: accepts. Unchanged.
APPEARANCE_EDIT_MODES = ("dc", "full")
#: The opacity arm — the only mode :func:`compose_shared_opacity` accepts.
OPACITY_EDIT_MODES = ("opacity",)
#: Every mode a sidecar payload may carry.
EDIT_MODES = APPEARANCE_EDIT_MODES + OPACITY_EDIT_MODES


def redirects_features(mode):
    """True when `mode` redirects spherical-harmonic appearance."""
    return mode in APPEARANCE_EDIT_MODES


def redirects_opacity(mode):
    """True when `mode` redirects the raw opacity logit."""
    return mode in OPACITY_EDIT_MODES


def _validated_pointer_index(source_idx, n_rows, target):
    """The pointer contract shared by every payload.

    `source_idx` is a per-row long tensor; identity rows point at
    themselves. A row with an incoming pointer is never itself a
    recipient (one hop, enforced at build time and re-checked here).
    Returns the index moved onto `target`'s device.
    """
    if source_idx.dim() != 1 or source_idx.shape[0] != n_rows:
        raise ContractError(
            "appearance-edit pointer column must be 1-D with one entry "
            f"per row; got {tuple(source_idx.shape)} for "
            f"{n_rows} rows"
        )
    idx = source_idx.to(target.device)
    redirected = idx != torch.arange(idx.shape[0], device=idx.device)
    if bool(redirected.any()):
        targets = idx[redirected]
        if bool(redirected[targets].any()):
            raise ContractError(
                "appearance-edit pointer chain detected: a donor row "
                "carries an incoming pointer (one-hop invariant violated)"
            )
    return idx


def compose_shared_features(features_dc, features_rest, source_idx, mode):
    """Return (features_dc, features_rest) under the pointer edit.

    Accepts the APPEARANCE modes only; the opacity arm has its own
    composer. Behaviour for "dc"/"full" is unchanged.
    """
    if mode not in APPEARANCE_EDIT_MODES:
        raise ContractError(f"unknown appearance-edit mode: {mode!r}")
    idx = _validated_pointer_index(
        source_idx, features_dc.shape[0], features_dc
    )
    features_dc = features_dc[idx]
    if mode == "full":
        features_rest = features_rest[idx]
    return features_dc, features_rest


def compose_shared_opacity(opacity, source_idx, mode):
    """Return the RAW OPACITY LOGIT column under the pointer edit.

    The redirect is applied before the sigmoid on purpose: the logit is
    the stored parameter, and because the activation is a monotone
    bijection, redirecting the logit and redirecting the activated value
    produce the same rendered opacity.
    """
    if mode not in OPACITY_EDIT_MODES:
        raise ContractError(f"unknown opacity-edit mode: {mode!r}")
    idx = _validated_pointer_index(source_idx, opacity.shape[0], opacity)
    return opacity[idx]


def build_edit_payload(pointer, mode, *, source_checkpoint, funnel=None):
    """The sidecar payload written by the consolidation pass."""
    if mode not in EDIT_MODES:
        raise ContractError(f"unknown appearance-edit mode: {mode!r}")
    pointer = pointer.detach().to("cpu", torch.long)
    return {
        "schema": APPEARANCE_EDIT_SCHEMA,
        "mode": mode,
        "pointer": pointer,
        "num_rows": int(pointer.shape[0]),
        "num_redirected": int(
            (pointer != torch.arange(pointer.shape[0])).sum()
        ),
        "source_checkpoint": str(source_checkpoint),
        "funnel": dict(funnel or {}),
    }


def load_edit_payload(path):
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != APPEARANCE_EDIT_SCHEMA:
        raise ContractError(
            f"unsupported appearance-edit schema: {payload.get('schema')!r}"
        )
    if payload.get("mode") not in EDIT_MODES:
        raise ContractError(
            f"unsupported appearance-edit mode: {payload.get('mode')!r}"
        )
    pointer = payload["pointer"]
    if pointer.shape[0] != payload.get("num_rows"):
        raise ContractError("appearance-edit pointer/num_rows mismatch")
    return payload


def apply_appearance_edit(gaussians, payload):
    """Install the pointer edit on a restored model (fails closed on a
    row-count mismatch — the edit belongs to exactly one checkpoint)."""
    n_rows = gaussians._features_dc.shape[0]
    pointer = payload["pointer"]
    if pointer.shape[0] != n_rows:
        raise ContractError(
            f"appearance edit was built for {pointer.shape[0]} rows but "
            f"the restored model has {n_rows}"
        )
    gaussians._appearance_source_idx = pointer.to(
        gaussians._features_dc.device
    )
    gaussians._appearance_share_mode = payload["mode"]


def clear_appearance_edit(gaussians):
    gaussians._appearance_source_idx = torch.empty(0, dtype=torch.long)
    gaussians._appearance_share_mode = "dc"


#: Every per-row column the invariant hash covers, in hash order. The
#: last four were added 2026-08-23 with the opacity payload: the edit is
#: a pointer REDIRECT and writes no parameter tensor, so an unchanged
#: hash across install -> render -> clear is the proof of that. The
#: earlier list left `_packet_ids`, `_rotation_r`, `_motion_v` and
#: `_motion_a` uncovered, which would have made a future payload that
#: WROTE one of them invisible to the check. Appending changes the hash
#: VALUE of any given state, so a hash recorded before that date is not
#: comparable with one recorded after; the invariant itself (pre == post
#: within one pass) is unaffected.
HASHED_ROW_COLUMNS = (
    "_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
    "_rotation", "_t", "_scaling_t", "_route_logit",
    "_motion_lora_coeff",
    "_packet_ids", "_rotation_r", "_motion_v", "_motion_a",
)


def non_pointer_state_hash(gaussians):
    """SHA-256 over every per-row parameter EXCEPT the pointer/mode
    columns — the frozen v3 invariant check that B2 == B1 outside the
    edit (ccr-method-2026-08-20 §3.2 item 5).

    Because no composer ever writes back, an install/clear cycle must
    leave this digest untouched: that is what proves the deployed state
    differs from the trained state by the pointer column alone. Empty
    columns are skipped, so a lane that never materialized a tensor
    hashes the same as one that has it empty.
    """
    hasher = hashlib.sha256()
    for name in HASHED_ROW_COLUMNS:
        tensor = getattr(gaussians, name, None)
        if tensor is None or (hasattr(tensor, "numel") and tensor.numel() == 0):
            continue
        hasher.update(name.encode())
        hasher.update(tensor.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


__all__ = [
    "APPEARANCE_EDIT_MODES",
    "APPEARANCE_EDIT_SCHEMA",
    "EDIT_MODES",
    "HASHED_ROW_COLUMNS",
    "OPACITY_EDIT_MODES",
    "apply_appearance_edit",
    "build_edit_payload",
    "clear_appearance_edit",
    "compose_shared_features",
    "compose_shared_opacity",
    "load_edit_payload",
    "non_pointer_state_hash",
    "redirects_features",
    "redirects_opacity",
]
