"""EL-GS checkpoint state (schema elgs-state-v1).

Composes the serializable EL-GS components into ONE schema-versioned
dict that rides inside GaussianModel.capture()'s routing/motion
container as `elgs_state`, sibling to `capacity_state` and
`lifecycle_state`. Loading validates fail-closed: schema id, per-
family interval dimensions (delegated to elgs.intervals'
mandatory loader check), watermark sanity, and unknown-key rejection.
A baseline checkpoint simply has no `elgs_state` (None) — the
substrate initializes fresh; a PRESENT but invalid payload is an
explicit rejection, never a silent skip (implementation plan §9).
"""

from __future__ import annotations

from depth_visibility.errors import ContractError, SchemaError

from .clusters import BindingTable
from .families import FamilyRegistry
from .transaction_ledger import SearchCostLedger, TransactionLedger

ELGS_STATE_SCHEMA = "elgs-state-v1"

_TOP_KEYS = {
    "schema_version",
    "registry",
    "binding",
    "ledger",
    "search_cost",
    "sampler",
    "slot_grid",
    "confirmation_refs",
    "moment_reset_log",
    "round_bookkeeping",
    "rng",
    "evidence",
}

# `rng` and `evidence` are OPTIONAL: a photometric-only run (no
# elgs_tracks_dir) writes no evidence block, and every M0-era checkpoint
# predates both keys. Their absence restores a valid photometric-only
# state; a PRESENT but malformed block still fails closed downstream.
_REQUIRED_KEYS = _TOP_KEYS - {"rng", "evidence"}


def build_elgs_state(
    registry: FamilyRegistry,
    binding: BindingTable,
    ledger: TransactionLedger,
    search_cost: SearchCostLedger,
    *,
    sampler: dict,
    slot_grid: dict,
    confirmation_refs: dict,
    moment_reset_log: list,
    round_bookkeeping: dict,
    rng: dict | None = None,
    evidence: dict | None = None,
) -> dict:
    """Assemble the checkpoint payload.

    sampler: {"lambda_u", "pi_d_identity", "frozen"};
    slot_grid: geometry + consumed slot indices + reserved pool;
    confirmation_refs: decision_id -> list of (camera, frame) units
    (required by the post-refit classification pass, spec §8);
    moment_reset_log / round_bookkeeping / rng: as produced by the
    trainer integration.
    """
    for key in ("lambda_u", "pi_d_identity", "frozen"):
        if key not in sampler:
            raise ContractError(f"sampler state missing {key}")
    return {
        "schema_version": ELGS_STATE_SCHEMA,
        "registry": registry.to_state(),
        "binding": binding.to_state(),
        "ledger": ledger.to_state(),
        "search_cost": search_cost.to_state(),
        "sampler": dict(sampler),
        "slot_grid": dict(slot_grid),
        "confirmation_refs": {str(k): list(v) for k, v in confirmation_refs.items()},
        "moment_reset_log": list(moment_reset_log),
        "round_bookkeeping": dict(round_bookkeeping),
        "rng": dict(rng) if rng is not None else {},
        "evidence": dict(evidence) if evidence is not None else {},
    }


def load_elgs_state(payload: dict) -> dict:
    """Validate and materialize the payload. Returns a dict of live
    objects: registry, binding, ledger, search_cost + the passthrough
    sections. Every failure is an explicit rejection."""
    if not isinstance(payload, dict):
        raise SchemaError("elgs_state must be a dict")
    if payload.get("schema_version") != ELGS_STATE_SCHEMA:
        raise SchemaError(
            f"incompatible elgs_state schema: {payload.get('schema_version')!r} "
            f"(expected {ELGS_STATE_SCHEMA}); refusing to load"
        )
    unknown = set(payload) - _TOP_KEYS
    if unknown:
        raise SchemaError(f"elgs_state carries unknown keys {sorted(unknown)}")
    missing = _REQUIRED_KEYS - set(payload)
    if missing:
        raise SchemaError(f"elgs_state missing keys {sorted(missing)}")
    return {
        "registry": FamilyRegistry.from_state(payload["registry"]),
        "binding": BindingTable.from_state(payload["binding"]),
        "ledger": TransactionLedger.from_state(payload["ledger"]),
        "search_cost": SearchCostLedger.from_state(payload["search_cost"]),
        "sampler": dict(payload["sampler"]),
        "slot_grid": dict(payload["slot_grid"]),
        "confirmation_refs": {
            k: [tuple(u) for u in v] for k, v in payload["confirmation_refs"].items()
        },
        "moment_reset_log": list(payload["moment_reset_log"]),
        "round_bookkeeping": dict(payload["round_bookkeeping"]),
        "rng": dict(payload.get("rng", {})),
        "evidence": dict(payload.get("evidence", {})),
    }


def strip_optimizer_group(state_dict: dict, *, name: str = "elgs_a") -> dict:
    """Remove one named param group (and its per-param state) from an
    optimizer state_dict copy.

    The a-logit group is NOT persisted inside the model's optimizer
    state: on restore, training_setup builds the base-group optimizer
    BEFORE setup_elgs exists to re-add the group, so a checkpoint
    carrying the extra group can never load (the S2 resume smoke
    caught exactly this ValueError). The a-logit VALUES persist
    exactly in elgs_state; their Adam moments reset on restore —
    disclosed and mirrored into moment_reset_log by setup.
    """
    groups = list(state_dict.get("param_groups", []))
    kept_groups = [g for g in groups if g.get("name") != name]
    if len(kept_groups) == len(groups):
        return state_dict
    kept_ids = {pid for g in kept_groups for pid in g.get("params", [])}
    return {
        "state": {
            k: v for k, v in state_dict.get("state", {}).items() if k in kept_ids
        },
        "param_groups": kept_groups,
    }


__all__ = [
    "ELGS_STATE_SCHEMA",
    "build_elgs_state",
    "load_elgs_state",
    "strip_optimizer_group",
]
