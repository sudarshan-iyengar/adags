"""Directional donor appearance reuse: the pointer-column edit (CCR v3).

The consolidation pass (scripts/consolidate_packets.py) is a
POST-TRAINING structural edit: the deployed B2 state differs from the
trained B1 state by a per-row appearance-source pointer column and a
mode flag, and by nothing else (ccr-method-2026-08-20 §3.2). This
module owns that edit: the pure composition rule, the sidecar format,
and apply/clear helpers. It imports nothing from the trainer.

Modes (frozen clause 2 — arms are never mixed within a deployed state):
- "dc":   recipient rows render with the DONOR row's DC / base radiance
          and keep their own higher-order SH (the primary arm);
- "full": recipient rows render with the donor's full SH (ablation arm).
"""

from __future__ import annotations

import hashlib

import torch

from depth_visibility.errors import ContractError

APPEARANCE_EDIT_SCHEMA = "adags-appearance-edit-v1"
APPEARANCE_EDIT_MODES = ("dc", "full")


def compose_shared_features(features_dc, features_rest, source_idx, mode):
    """Return (features_dc, features_rest) under the pointer edit.

    source_idx is a per-row long tensor; identity rows point at
    themselves. A row with an incoming pointer is never itself a
    recipient (one hop, enforced at build time and re-checked here).
    """
    if mode not in APPEARANCE_EDIT_MODES:
        raise ContractError(f"unknown appearance-edit mode: {mode!r}")
    if source_idx.dim() != 1 or source_idx.shape[0] != features_dc.shape[0]:
        raise ContractError(
            "appearance-edit pointer column must be 1-D with one entry "
            f"per row; got {tuple(source_idx.shape)} for "
            f"{features_dc.shape[0]} rows"
        )
    idx = source_idx.to(features_dc.device)
    redirected = idx != torch.arange(idx.shape[0], device=idx.device)
    if bool(redirected.any()):
        targets = idx[redirected]
        if bool(redirected[targets].any()):
            raise ContractError(
                "appearance-edit pointer chain detected: a donor row "
                "carries an incoming pointer (one-hop invariant violated)"
            )
    features_dc = features_dc[idx]
    if mode == "full":
        features_rest = features_rest[idx]
    return features_dc, features_rest


def build_edit_payload(pointer, mode, *, source_checkpoint, funnel=None):
    """The sidecar payload written by the consolidation pass."""
    if mode not in APPEARANCE_EDIT_MODES:
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
    if payload.get("mode") not in APPEARANCE_EDIT_MODES:
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


def non_pointer_state_hash(gaussians):
    """SHA-256 over every per-row parameter EXCEPT the pointer/mode
    columns — the frozen v3 invariant check that B2 == B1 outside the
    edit (ccr-method-2026-08-20 §3.2 item 5)."""
    hasher = hashlib.sha256()
    for name in (
        "_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
        "_rotation", "_t", "_scaling_t", "_route_logit",
        "_motion_lora_coeff",
    ):
        tensor = getattr(gaussians, name, None)
        if tensor is None or (hasattr(tensor, "numel") and tensor.numel() == 0):
            continue
        hasher.update(name.encode())
        hasher.update(tensor.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


__all__ = [
    "APPEARANCE_EDIT_MODES",
    "APPEARANCE_EDIT_SCHEMA",
    "apply_appearance_edit",
    "build_edit_payload",
    "clear_appearance_edit",
    "compose_shared_features",
    "load_edit_payload",
    "non_pointer_state_hash",
]
