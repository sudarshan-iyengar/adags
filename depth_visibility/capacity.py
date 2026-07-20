"""Point-neutral Slice B capacity transaction primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .errors import ContractError


@dataclass
class CapacityBank:
    """Optimizer-stable row bank used by the Slice B B00 fixtures."""

    parameters: dict[str, nn.Parameter]
    accumulators: dict[str, torch.Tensor]
    stable_ids: torch.Tensor
    generation: torch.Tensor
    last_reassigned: torch.Tensor
    hard_static_count: int = 0

    @property
    def dynamic_count(self) -> int:
        counts = {int(value.shape[0]) for value in self.parameters.values()}
        counts.update(int(value.shape[0]) for value in self.accumulators.values())
        counts.add(int(self.stable_ids.shape[0]))
        counts.add(int(self.generation.shape[0]))
        counts.add(int(self.last_reassigned.shape[0]))
        if len(counts) != 1:
            raise ContractError(f"capacity bank has inconsistent row counts: {sorted(counts)}")
        return counts.pop()


def capacity_budget(bank: CapacityBank) -> dict[str, int]:
    dynamic = bank.dynamic_count
    hard_static = int(bank.hard_static_count)
    if dynamic < 0 or hard_static < 0:
        raise ContractError("capacity counts must be nonnegative")
    return {"dynamic": dynamic, "hard_static": hard_static, "total": dynamic + hard_static}


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ContractError(f"capacity tensor contains nonfinite values: {name}")


def _as_index(indices: torch.Tensor, *, device: torch.device, row_count: int) -> torch.Tensor:
    result = torch.as_tensor(indices, dtype=torch.long, device=device).reshape(-1)
    if result.numel() == 0:
        raise ContractError("capacity transaction requires at least one donor")
    if torch.unique(result).numel() != result.numel():
        raise ContractError("capacity donor rows must be unique")
    if int(result.min()) < 0 or int(result.max()) >= row_count:
        raise ContractError("capacity donor index out of range")
    return result


def _reset_optimizer_rows(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    rows: torch.Tensor,
) -> int:
    state = optimizer.state.get(parameter, {})
    reset = 0
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(key)
        if torch.is_tensor(value) and value.shape[:1] == parameter.shape[:1]:
            value.index_fill_(0, rows.to(value.device), 0.0)
            reset += int(rows.numel())
    return reset


def apply_point_neutral_transaction(
    bank: CapacityBank,
    optimizer: torch.optim.Optimizer,
    donor_indices: torch.Tensor,
    target_rows: Mapping[str, torch.Tensor] | None,
    *,
    iteration: int,
    mode: str = "reassign",
) -> dict[str, object]:
    """Rewrite donor rows in place while preserving parameter and point counts."""

    if mode not in {"reassign", "null-reset", "no-op"}:
        raise ContractError(f"unknown capacity transaction mode: {mode}")
    before = capacity_budget(bank)
    row_count = bank.dynamic_count
    device = next(iter(bank.parameters.values())).device
    donors = _as_index(donor_indices, device=device, row_count=row_count)
    if mode == "no-op":
        return {
            "mode": mode,
            "requested_k": int(donors.numel()),
            "realized_k": 0,
            "budget_before": before,
            "budget_after": capacity_budget(bank),
            "moment_rows_reset": 0,
        }
    if int(iteration) <= 0:
        raise ContractError("capacity transaction iteration must be positive")
    if mode == "reassign" and target_rows is None:
        raise ContractError("reassign mode requires target rows")
    if mode == "null-reset" and target_rows is not None:
        raise ContractError("null-reset must not receive target rows")

    parameter_identities = {name: id(parameter) for name, parameter in bank.parameters.items()}
    if mode == "reassign":
        assert target_rows is not None
        missing = set(bank.parameters) - set(target_rows)
        extra = set(target_rows) - set(bank.parameters)
        if missing or extra:
            raise ContractError(
                f"capacity target row names mismatch: missing={sorted(missing)} extra={sorted(extra)}"
            )
        with torch.no_grad():
            for name, parameter in bank.parameters.items():
                rows = torch.as_tensor(
                    target_rows[name], dtype=parameter.dtype, device=parameter.device
                )
                if rows.shape != (donors.numel(), *parameter.shape[1:]):
                    raise ContractError(f"target rows for {name} have wrong shape")
                _require_finite(name, rows)
                parameter.data[donors] = rows

    moment_rows_reset = 0
    with torch.no_grad():
        for name, parameter in bank.parameters.items():
            _require_finite(name, parameter.data)
            moment_rows_reset += _reset_optimizer_rows(optimizer, parameter, donors)
        for name, accumulator in bank.accumulators.items():
            if accumulator.shape[:1] != (row_count,):
                raise ContractError(f"accumulator {name} has wrong row count")
            accumulator.index_fill_(0, donors.to(accumulator.device), 0.0)
            _require_finite(name, accumulator)
        bank.generation[donors.to(bank.generation.device)] += 1
        bank.last_reassigned[donors.to(bank.last_reassigned.device)] = int(iteration)

    for name, parameter in bank.parameters.items():
        if id(parameter) != parameter_identities[name]:
            raise ContractError(f"parameter identity changed for {name}")
    after = capacity_budget(bank)
    if before != after:
        raise ContractError(f"capacity budget changed: before={before} after={after}")
    return {
        "mode": mode,
        "requested_k": int(donors.numel()),
        "realized_k": int(donors.numel()),
        "donor_indices": [int(value) for value in donors.cpu().tolist()],
        "stable_slot_ids": [
            int(value) for value in bank.stable_ids[donors.to(bank.stable_ids.device)].cpu().tolist()
        ],
        "budget_before": before,
        "budget_after": after,
        "moment_rows_reset": int(moment_rows_reset),
        "iteration": int(iteration),
    }


def select_event_blind_donors(
    *,
    xyz: torch.Tensor,
    scaling_log: torch.Tensor,
    opacity_logit: torch.Tensor,
    denom: torch.Tensor,
    generation: torch.Tensor,
    stable_ids: torch.Tensor,
    current_iteration: int,
    k: int,
) -> dict[str, object]:
    """Select event-blind redundant low-opacity dynamic slots."""

    tensors = {
        "xyz": xyz,
        "scaling_log": scaling_log,
        "opacity_logit": opacity_logit,
        "denom": denom,
        "generation": generation,
        "stable_ids": stable_ids,
    }
    for name, tensor in tensors.items():
        _require_finite(name, torch.as_tensor(tensor))
    n = int(xyz.shape[0])
    if k <= 0:
        raise ContractError("donor selection K must be positive")
    if any(int(tensor.shape[0]) != n for tensor in tensors.values()):
        raise ContractError("donor selection tensors have inconsistent row counts")
    if k > n:
        raise ContractError("donor selection K exceeds dynamic row count")

    activated_opacity = torch.sigmoid(opacity_logit.reshape(n))
    ages = int(current_iteration) - generation.to(dtype=torch.long).reshape(n)
    bottom_count = int(torch.floor(torch.tensor(0.20 * n)).item())
    if bottom_count <= 0:
        return {"selected_indices": [], "base_universe_indices": [], "abstained": True, "reason": "empty_bottom_opacity_population"}
    order = sorted(range(n), key=lambda idx: (float(activated_opacity[idx]), int(stable_ids[idx])))
    bottom = set(order[:bottom_count])
    scales = torch.exp(scaling_log).reshape(n, -1).amax(dim=1)
    neighbor_counts = []
    base_universe = []
    for idx in sorted(bottom):
        if int(ages[idx]) < 500:
            continue
        distances = torch.linalg.norm(xyz - xyz[idx], dim=1)
        eligible = []
        for other in range(n):
            if other == idx:
                continue
            if (
                torch.isfinite(distances[other])
                and float(distances[other]) <= 2.0 * float(scales[idx])
                and float(activated_opacity[other]) > float(activated_opacity[idx])
            ):
                eligible.append(other)
        if not eligible:
            continue
        neighbor_counts.append((idx, len(eligible)))
        base_universe.append(idx)
    if len(base_universe) < k:
        return {
            "selected_indices": [],
            "base_universe_indices": [int(value) for value in base_universe],
            "requested_k": int(k),
            "realized_k": 0,
            "abstained": True,
            "reason": "fewer_than_k_donors",
        }
    neighbor_count = {idx: count for idx, count in neighbor_counts}
    ranked = sorted(
        base_universe,
        key=lambda idx: (
            float(activated_opacity[idx]),
            float(denom.reshape(n)[idx]) / max(int(ages[idx]), 1),
            -int(neighbor_count[idx]),
            int(stable_ids[idx]),
        ),
    )
    selected = ranked[:k]
    return {
        "selected_indices": [int(value) for value in selected],
        "base_universe_indices": [int(value) for value in base_universe],
        "requested_k": int(k),
        "realized_k": int(k),
        "abstained": False,
    }


__all__ = [
    "CapacityBank",
    "apply_point_neutral_transaction",
    "capacity_budget",
    "select_event_blind_donors",
]
