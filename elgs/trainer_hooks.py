"""Trainer-side EL-GS orchestration (thin main.py surface).

main.py calls exactly three functions here: setup_elgs() once after
model construction/restore, maybe_run_elgs_schedule() once per
iteration, and elgs_summary() for reporting. Everything else is
internal and CPU-importable; the only CUDA touchpoints are the xyz
read at seeding and the injected render callback at rounds.

Structural constants and schedule anchors load from the FROZEN
configs/elgs/prereg_structural_v1.json — never from lane YAML
(preregistration integrity, implementation plan §7). Frame-interval
units convert to model time units through the dataset's inferred
inter-frame dt.

The M0 smoke proposer below is SMOKE-TIER supporting machinery: it
exercises the full mechanical chain (screen -> conflict order ->
CRN paired renders -> SNIS acceptance -> atomic commit/ITT) without
track evidence, which arrives with the M1 census artifacts. It is
never evidence-bearing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch

from depth_visibility.errors import ContractError, SchemaError

from .acceptance import FrozenSamplerParams, SnisSample
from .clusters import BindingTable
from .families import FamilyRegistry
from .intervals import IntervalConfig, inverse
from .ops import plan_fission
from .round_driver import ProposedCandidate, run_pass
from .runtime import ElgsRuntime, ScheduleAnchors
from .state_io import load_elgs_state
from .transaction_ledger import SearchCostLedger, TransactionLedger
from .transactions import StateBundle

STRUCTURAL_PREREG = "prereg_structural_v1.json"


def _unpack_camera(item):
    """Train-dataset items are (gt_image, camera) tuples or bare
    cameras (main.py's convention); return (camera, original item)."""
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return item[1], item
    return item, item


@dataclass
class ElgsTrainerState:
    runtime: ElgsRuntime
    bundle: StateBundle
    config: IntervalConfig
    schedule: ScheduleAnchors
    sampler_params: FrozenSamplerParams
    prereg: dict
    frame_dt: float
    seeded: bool = False
    rounds_run: list = field(default_factory=list)
    a_lr: float = 0.0
    k_se: float = 1.0
    candidate_cap: int = 8
    confirmation_samples: int = 16
    base_seed: int = 0
    logit_group_index: int | None = None


def load_structural_prereg(prereg_dir: str) -> dict:
    path = os.path.join(prereg_dir, STRUCTURAL_PREREG)
    if not os.path.isfile(path):
        raise ContractError(f"structural prereg missing: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "elgs-prereg-structural-v1":
        raise SchemaError(
            f"unsupported structural prereg schema: {payload.get('schema_version')!r}"
        )
    if payload.get("status") != "frozen":
        raise ContractError("structural prereg must be frozen before any run")
    return payload


def infer_frame_dt(timestamps: list[float]) -> float:
    unique = sorted(set(float(t) for t in timestamps))
    if len(unique) < 2:
        raise ContractError("cannot infer frame dt from fewer than two timestamps")
    gaps = [b - a for a, b in zip(unique, unique[1:])]
    dt = min(gaps)
    if dt <= 0:
        raise ContractError("non-increasing timestamps")
    return dt


def build_interval_config(prereg: dict, time_span: float, frame_dt: float) -> IntervalConfig:
    iv = prereg["interval"]
    w = float(iv["w_frame_intervals"]) * frame_dt
    return IntervalConfig(
        T=float(time_span),
        w_m=float(iv["w_m_frame_intervals"]) * frame_dt,
        w=w,
        floor_len=2.0 * w + float(iv["delta_len_frame_intervals"]) * frame_dt,
        floor_gap=2.0 * w + float(iv["delta_gap_frame_intervals"]) * frame_dt,
    )


def setup_elgs(gaussians, scene, dataset, opt) -> ElgsTrainerState | None:
    """Attach the EL-GS runtime, or return None for non-EL-GS lanes."""
    if not bool(getattr(opt, "elgs_enable", False)):
        return None
    if bool(getattr(opt, "lifecycle_enable", False)):
        raise ContractError("elgs_enable and lifecycle_enable are mutually exclusive")

    prereg = load_structural_prereg(str(getattr(opt, "elgs_prereg_dir", "configs/elgs")))
    cameras = [
        _unpack_camera(item)[0] for item in scene.getTrainCameras()
    ]
    timestamps = [float(getattr(c, "timestamp", 0.0)) for c in cameras]
    frame_dt = infer_frame_dt(timestamps)
    time_span = max(timestamps) - min(timestamps)
    config = build_interval_config(prereg, time_span, frame_dt)

    schedule_key = "smoke" if bool(getattr(opt, "elgs_smoke_schedule", False)) else "full"
    sched = prereg["schedule"][schedule_key]
    schedule = ScheduleAnchors(
        seed_iteration=int(sched["seed_iteration"]),
        audit_iteration=int(sched["audit_iteration"]),
        round_iterations=tuple(int(v) for v in sched["round_iterations"]),
        refit_until=int(sched["refit_until"]),
    )

    for key in ("elgs_a_lr", "elgs_k_se", "elgs_lambda_u", "elgs_candidate_cap",
                "elgs_confirmation_samples"):
        if float(getattr(opt, key)) < 0:
            raise ContractError(f"{key} is unset (-1 sentinel); set it in the run config")

    pending = getattr(gaussians, "_pending_elgs_state", None)
    if pending is not None:
        loaded = load_elgs_state(pending)  # present-but-invalid raises here
        registry = loaded["registry"]
        bundle = StateBundle(registry, loaded["binding"], loaded["ledger"],
                             loaded["search_cost"])
        runtime = ElgsRuntime(registry, config, schedule,
                              device=gaussians._xyz.device
                              if gaussians._xyz.numel() else "cpu",
                              dtype=torch.float32)
        row_ids = loaded["round_bookkeeping"].get("row_family_ids")
        if row_ids is not None:
            gaussians._elgs_family_ids = torch.tensor(row_ids, dtype=torch.long)
        seeded = bool(loaded["round_bookkeeping"].get("seeded", True))
        rounds_run = list(loaded["round_bookkeeping"].get("rounds_run", []))
    else:
        registry = FamilyRegistry()
        bundle = StateBundle(
            registry, BindingTable(), TransactionLedger(),
            SearchCostLedger(
                row_cap=int(getattr(opt, "densify_until_num_points", 600000) or 600000),
                scalar_budget=10**10,
            ),
        )
        runtime = ElgsRuntime(registry, config, schedule, dtype=torch.float32)
        seeded = False
        rounds_run = []
    gaussians._pending_elgs_state = None

    sampler = FrozenSamplerParams(
        lambda_u=float(getattr(opt, "elgs_lambda_u")),
        pi_d_identity="uniform-train-units-v1",
        frozen=True,
    )
    state = ElgsTrainerState(
        runtime=runtime,
        bundle=bundle,
        config=config,
        schedule=schedule,
        sampler_params=sampler,
        prereg=prereg,
        frame_dt=frame_dt,
        seeded=seeded,
        rounds_run=rounds_run,
        a_lr=float(getattr(opt, "elgs_a_lr")),
        k_se=float(getattr(opt, "elgs_k_se")),
        candidate_cap=int(getattr(opt, "elgs_candidate_cap")),
        confirmation_samples=int(getattr(opt, "elgs_confirmation_samples")),
        base_seed=int(getattr(opt, "seed", 0) or 0),
    )
    # Family creation happens at SETUP (spanning-then-carve: the
    # initial cloud starts as K=1 spanning families — substrate).
    # The schedule's seed/audit anchors gate the EVIDENCE machinery
    # (track seeds, cluster binding), which needs the M1 artifacts
    # and stays inactive without an elgs_tracks_dir.
    gaussians.elgs_runtime = runtime
    gaussians._elgs_checkpoint_extras = {
        "binding": bundle.binding,
        "ledger": bundle.ledger,
        "search_cost": bundle.search_cost,
        "sampler": {"lambda_u": sampler.lambda_u,
                    "pi_d_identity": sampler.pi_d_identity,
                    "frozen": sampler.frozen},
        "round_bookkeeping": {"seeded": seeded, "rounds_run": rounds_run},
    }
    if not state.seeded:
        seed_families(state, gaussians, iteration=0)
    else:
        _refresh_logit_param_group(state, gaussians)
    print(json.dumps({"elgs_setup": {
        "schedule": schedule_key,
        "frame_dt": frame_dt,
        "time_span": time_span,
        "restored": pending is not None,
        "families": len(registry.active_ids()),
    }}, sort_keys=True))
    return state


def seed_families(state: ElgsTrainerState, gaussians, iteration: int) -> None:
    """Voxel-grid family seeding at seed_iteration (preregistered rule):
    each nonempty cell is one family, K=1 spanning, latch (1,1)."""
    if state.seeded:
        raise ContractError("families already seeded")
    seeding = state.prereg["family_seeding"]
    cells = int(seeding["voxel_grid_cells_per_axis"])
    max_families = int(seeding["max_families"])
    xyz = gaussians._xyz.detach()
    lo = xyz.min(dim=0).values
    span = (xyz.max(dim=0).values - lo).clamp_min(1e-6)
    voxel = ((xyz - lo) / span * cells).clamp(0, cells - 1).long()
    keys = voxel[:, 0] * cells * cells + voxel[:, 1] * cells + voxel[:, 2]
    unique_keys, inverse_map = torch.unique(keys, sorted=True, return_inverse=True)
    if unique_keys.numel() > max_families:
        raise ContractError(
            f"seeding produced {unique_keys.numel()} families above the "
            f"preregistered cap {max_families}"
        )
    spanning = inverse(
        1, True, True, 0.0, [state.config.omega], [], 0.0, state.config,
        dtype=torch.float32,
    )
    family_ids = torch.empty_like(inverse_map)
    for cell_index in range(unique_keys.numel()):
        member_mask = inverse_map == cell_index
        centroid = xyz[member_mask].mean(dim=0)
        record = state.runtime.registry.create_family(
            birth_time=0.0,
            birth_site=tuple(float(v) for v in centroid.tolist()),
            lineage_key=f"seed-cell-{int(unique_keys[cell_index])}",
            interval=spanning,
            tau=(0.0,),
        )
        state.runtime.registry.on_rows_added(record.family_id, int(member_mask.sum()))
        family_ids[member_mask] = record.family_id
    gaussians._elgs_family_ids = family_ids.cpu()
    state.runtime._sync_all_from_registry()
    _refresh_logit_param_group(state, gaussians)
    state.seeded = True
    gaussians._elgs_checkpoint_extras["round_bookkeeping"]["seeded"] = True
    print(json.dumps({"elgs_seeding": {
        "iteration": iteration,
        "families": len(state.runtime.registry.active_ids()),
        "rows": int(family_ids.numel()),
    }}, sort_keys=True))


def _refresh_logit_param_group(state: ElgsTrainerState, gaussians) -> None:
    """(Re)install the a-logit tensors as the `elgs_a` optimizer group.
    Structural ops replace tensors, so the group is rebuilt after each
    sync; fresh tensors start with zero moments (= the reset)."""
    params = list(state.runtime.logit_parameters().values())
    optimizer = gaussians.optimizer
    if state.logit_group_index is not None:
        group = optimizer.param_groups[state.logit_group_index]
        for old in group["params"]:
            optimizer.state.pop(old, None)
        group["params"] = params
    else:
        optimizer.add_param_group({"params": params, "lr": state.a_lr, "name": "elgs_a"})
        state.logit_group_index = len(optimizer.param_groups) - 1


def _propose_smoke_candidates(state: ElgsTrainerState, iteration: int) -> list:
    """SMOKE-TIER proposer: one admissible mid-plateau fission on the
    largest-span family. Deterministic; never evidence-bearing."""
    proposals = []
    best = None
    for family_id in state.runtime.registry.active_ids():
        record = state.runtime.registry.get(family_id)
        if record.interval.K != 1:
            continue
        r = state.runtime.realization(family_id)
        span = float(r.lens[0])
        if best is None or span > best[1]:
            best = (family_id, span)
    if best is None:
        return proposals
    family_id, _ = best
    r = state.runtime.realization(family_id)
    b, d = float(r.b[0]), float(r.d[0])
    mid = 0.5 * (b + d)
    half_gap = 0.5 * (state.config.floor_gap * 1.5)
    try:
        plan = plan_fission(
            state.runtime.registry, family_id, 0, mid - half_gap, mid + half_gap,
            round_index=len(state.rounds_run), iteration=iteration,
            config=state.config, dtype=torch.float32,
        )
        proposals.append(ProposedCandidate(
            plan=plan, screen_score=1.0, footprint_frames=(b, d),
        ))
    except ContractError:
        pass  # inadmissible on this geometry: no candidate this round
    return proposals


def maybe_run_elgs_schedule(state, gaussians, scene, opt, iteration, render_unit_loss) -> None:
    """Once per iteration: seeding, structural rounds, refit-end log.

    render_unit_loss(camera, override) -> float: the trainer's paired
    render evaluator (renders with the candidate presence override
    applied via gaussians._elgs_presence_override and returns the
    photometric loss on that camera).
    """
    if state is None:
        return
    if not state.seeded:
        raise ContractError("EL-GS families must be seeded at setup")
    if not state.runtime.is_round_boundary(iteration):
        return
    if iteration in state.rounds_run:
        return
    round_index = len(state.rounds_run)
    proposals = _propose_smoke_candidates(state, iteration)
    items = list(scene.getTrainCameras())
    n_units = max(6, min(state.confirmation_samples, len(items)))
    ordered = sorted(
        items,
        key=lambda it: (
            float(_unpack_camera(it)[0].timestamp),
            str(getattr(_unpack_camera(it)[0], "image_name", "")),
        ),
    )
    stride = max(1, len(ordered) // n_units)
    unit_items = ordered[::stride][:n_units]

    def sample_builder(plan, seed):
        overrides = dict(plan.child_intervals)
        samples = []
        for index, item in enumerate(unit_items):
            camera, _ = _unpack_camera(item)
            loss_inc = float(render_unit_loss(item, None))
            loss_cand = float(render_unit_loss(item, overrides))
            samples.append(SnisSample(
                unit=(index, float(camera.timestamp)),
                nu_density=1.0,
                mix_density=1.0,
                loss_incumbent=loss_inc,
                loss_candidate=loss_cand,
            ))
        return samples

    outcome = run_pass(
        proposals, state.runtime, state.bundle, sample_builder,
        state.sampler_params, k_se=state.k_se, base_seed=state.base_seed,
        round_index=round_index, pass_index=0, iteration=iteration,
        candidate_cap=state.candidate_cap,
    )
    if outcome.committed:
        _refresh_logit_param_group(state, gaussians)
    state.rounds_run.append(iteration)
    gaussians._elgs_checkpoint_extras["round_bookkeeping"]["rounds_run"] = list(state.rounds_run)
    print(json.dumps({"elgs_round": {
        "iteration": iteration,
        "round_index": round_index,
        "proposals": len(proposals),
        "committed": outcome.committed,
        "rejected": outcome.rejected,
        "families": len(state.runtime.registry.active_ids()),
    }}, sort_keys=True))


def elgs_summary(state) -> dict | None:
    if state is None:
        return None
    return {
        "families": len(state.runtime.registry.active_ids()),
        "rounds_run": list(state.rounds_run),
        "ledger_events": len(state.bundle.ledger.events),
        "candidates_tried": state.bundle.search_cost.candidates_tried,
        "candidates_accepted": state.bundle.search_cost.candidates_accepted,
    }


__all__ = [
    "ElgsTrainerState",
    "build_interval_config",
    "elgs_summary",
    "infer_frame_dt",
    "load_structural_prereg",
    "maybe_run_elgs_schedule",
    "seed_families",
    "setup_elgs",
]
