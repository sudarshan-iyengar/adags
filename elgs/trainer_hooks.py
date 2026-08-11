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
    slot_grid: object = None            # elgs.acceptance.SlotGrid
    reserved_indices: frozenset = frozenset()
    committed_decisions: list = field(default_factory=list)
    post_refit_done: bool = False


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
        delta_tol=float(iv["delta_tol_frame_intervals"]) * frame_dt,
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
        candidate_cap=slots_per_pass,
        confirmation_samples=int(getattr(opt, "elgs_confirmation_samples")),
        base_seed=int(getattr(opt, "seed", 0) or 0),
        slot_grid=state_slot_grid,
        reserved_indices=frozenset(u[0] for u in reserved),
    )
    # Dual caps (substrate): the scalar budget derives from the
    # baseline per-row scalar count at setup times the row cap, with
    # 1.5x headroom for global tensors and the a-logits (recorded).
    # A restored checkpoint keeps its own persisted budget.
    if pending is None:
        rows_now = max(1, int(gaussians._xyz.shape[0]))
        scalars_now = sum(
            p.numel() for g in gaussians.optimizer.param_groups for p in g["params"]
        )
        bundle.search_cost.scalar_budget = int(
            (scalars_now / rows_now) * bundle.search_cost.row_cap * 1.5
        )
    # Reserved confirmation pool (spec §7): every 4th train unit index
    # is reserved at iteration 0 and pre-partitioned into the slot
    # grid; the trainer's refit dataset EXCLUDES these units
    # (filter_elgs_reserved), so refits never see confirmation samples.
    from .acceptance import SlotGrid

    n_units_total = len(cameras)
    reserved = tuple(
        (i, float(getattr(cameras[i], "timestamp", 0.0)))
        for i in range(0, n_units_total, 4)
    )
    n_rounds = len(schedule.round_iterations)
    slots_per_pass = int(getattr(opt, "elgs_candidate_cap"))
    units_per_slot = max(6, len(reserved) // max(1, n_rounds * slots_per_pass))
    needed = n_rounds * slots_per_pass * units_per_slot
    if len(reserved) < needed:
        raise ContractError(
            f"reserved confirmation pool has {len(reserved)} units but the "
            f"slot grid needs {needed}; reduce elgs_candidate_cap or the "
            "schedule for this scene"
        )
    state_slot_grid = SlotGrid(
        n_rounds=n_rounds,
        n_passes=1,
        slots_per_pass=slots_per_pass,
        units_per_slot=units_per_slot,
        reserved_pool=reserved,
    )
    if pending is not None:
        for consumed in loaded["slot_grid"].get("consumed", []):
            state_slot_grid._consumed.add(int(consumed))

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
        "slot_grid": {
            "n_rounds": n_rounds,
            "slots_per_pass": slots_per_pass,
            "units_per_slot": units_per_slot,
            "reserved_indices": [u[0] for u in reserved],
            "consumed": sorted(state_slot_grid._consumed),
        },
        "confirmation_refs": {},
        "moment_reset_log": [],
        "round_bookkeeping": {"seeded": seeded, "rounds_run": rounds_run},
    }
    if not state.seeded:
        seed_families(state, gaussians, iteration=0)
    else:
        _refresh_logit_param_group(state, gaussians)
    _refresh_routing_pins(state, gaussians)
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


def _refresh_logit_param_group(
    state: ElgsTrainerState, gaussians, *, iteration: int = 0, op: str = "setup"
) -> None:
    """(Re)install the a-logit tensors as the `elgs_a` optimizer group.

    Moment resets are SCOPED (§1): only tensors an op actually
    replaced lose their Adam state — unchanged families keep the same
    tensor objects, so their optimizer state survives untouched. Each
    reset is mirrored into the checkpoint's moment_reset_log.
    """
    params = list(state.runtime.logit_parameters().values())
    optimizer = gaussians.optimizer
    if state.logit_group_index is not None:
        group = optimizer.param_groups[state.logit_group_index]
        new_ids = {id(p) for p in params}
        replaced = [old for old in group["params"] if id(old) not in new_ids]
        for old in replaced:
            optimizer.state.pop(old, None)
        if replaced:
            gaussians._elgs_checkpoint_extras["moment_reset_log"].append(
                {"iteration": iteration, "op": op, "tensors_reset": len(replaced)}
            )
        group["params"] = params
    else:
        optimizer.add_param_group({"params": params, "lr": state.a_lr, "name": "elgs_a"})
        state.logit_group_index = len(optimizer.param_groups) - 1


def _propose_smoke_candidates(state: ElgsTrainerState, iteration: int) -> list:
    """SMOKE-TIER proposer: one admissible mid-plateau fission on the
    largest-span family. Deterministic; never evidence-bearing.

    Inadmissibility is REJECTED AND LOGGED, never swallowed (§5
    'cap-saturated => INADMISSIBLE this round ... (logged)'; substrate
    'K-overflow = reject + log')."""
    from .transaction_ledger import LedgerEvent

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
    except ContractError as reason:
        k_related = "K_f = 4" in str(reason) or "K_old + K_new" in str(reason)
        kind = "k_overflow_reject" if k_related else "inadmissible_reject"
        if k_related:
            state.bundle.search_cost.k_overflow_rejects += 1
        state.bundle.ledger.append(
            LedgerEvent(kind, len(state.rounds_run), iteration, (family_id,),
                        {"op": "FISSION", "reason": str(reason)})
        )
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
    if state.runtime.is_round_boundary(iteration) and iteration not in state.rounds_run:
        _run_round(state, gaussians, scene, iteration, render_unit_loss)
    if (
        iteration >= state.schedule.refit_until
        and not state.post_refit_done
        and state.committed_decisions
    ):
        run_post_refit_classification(state, gaussians, scene, iteration, render_unit_loss)


def _run_round(state, gaussians, scene, iteration, render_unit_loss) -> None:
    import time

    round_index = len(state.rounds_run)
    proposals = _propose_smoke_candidates(state, iteration)
    items = list(scene.getTrainCameras())
    incumbent_snapshot = {
        fid: state.runtime.registry.get(fid).interval
        for p in proposals
        for fid in p.plan.family_ids
        if not state.runtime.registry.get(fid).retired
    }

    drawn_units: dict = {}

    def sample_builder(plan, seed, rank):
        # Confirmation units come EXCLUSIVELY from the slot grid's
        # reserved pool (spec §7): the refit dataset excludes them.
        units = state.slot_grid.draw(round_index, 0, rank)
        drawn_units[id(plan)] = [list(u) for u in units]
        overrides = dict(plan.child_intervals)
        samples = []
        started = time.time()
        for index, timestamp in units:
            item = items[index]
            loss_inc = float(render_unit_loss(item, None))
            loss_cand = float(render_unit_loss(item, overrides))
            state.bundle.search_cost.candidate_renders += 2
            samples.append(SnisSample(
                unit=(index, timestamp),
                nu_density=1.0,
                mix_density=1.0,
                loss_incumbent=loss_inc,
                loss_candidate=loss_cand,
            ))
        state.bundle.search_cost.rasterizer_seconds += time.time() - started
        return samples

    outcome = run_pass(
        proposals, state.runtime, state.bundle, sample_builder,
        state.sampler_params, k_se=state.k_se, base_seed=state.base_seed,
        round_index=round_index, pass_index=0, iteration=iteration,
        candidate_cap=state.candidate_cap,
    )
    from .intervals import serialize_state

    by_id = {p.as_search_candidate().candidate_id: p for p in proposals}
    for candidate_id in outcome.committed:
        plan = by_id[candidate_id].plan
        record = outcome.acceptance_records[candidate_id]
        decision = {
            "candidate_id": candidate_id,
            "op": plan.op,
            "family_id": plan.family_ids[0],
            "round_index": round_index,
            "iteration": iteration,
            "incumbent_intervals": {
                str(fid): serialize_state(incumbent_snapshot[fid])
                for fid in plan.family_ids if fid in incumbent_snapshot
            },
            "units": drawn_units.get(id(plan), []),
            "n_samples": record.n_samples,
            "se": record.se,
        }
        state.committed_decisions.append(decision)
        # confirmation_refs feeds the §8 post-refit classification
        # pass and persists in the checkpoint.
        gaussians._elgs_checkpoint_extras["confirmation_refs"][candidate_id] = [
            tuple(u) for u in decision["units"]
        ]
    if outcome.committed:
        _refresh_logit_param_group(state, gaussians, iteration=iteration, op="round_commit")
        _refresh_routing_pins(state, gaussians)

    # Dual-cap accounting (substrate): observed every round.
    rows = int(gaussians._xyz.shape[0])
    scalars = sum(
        p.numel() for g in gaussians.optimizer.param_groups for p in g["params"]
    )
    state.bundle.search_cost.observe_rows(rows, scalars)

    state.rounds_run.append(iteration)
    extras = gaussians._elgs_checkpoint_extras
    extras["round_bookkeeping"]["rounds_run"] = list(state.rounds_run)
    extras["slot_grid"]["consumed"] = sorted(state.slot_grid._consumed)
    print(json.dumps({"elgs_round": {
        "iteration": iteration,
        "round_index": round_index,
        "proposals": len(proposals),
        "committed": outcome.committed,
        "rejected": outcome.rejected,
        "families": len(state.runtime.registry.active_ids()),
        "peak_rows": state.bundle.search_cost.peak_rendered_rows,
        "peak_scalars": state.bundle.search_cost.peak_stored_scalars,
    }}, sort_keys=True))


def filter_elgs_reserved(training_dataset, state):
    """Exclude the reserved confirmation units from the refit
    training data (spec §7: refits never see confirmation samples).
    Returns a plain list the DataLoader consumes identically."""
    if state is None:
        return training_dataset
    return [
        item for index, item in enumerate(training_dataset)
        if index not in state.reserved_indices
    ]


def _refresh_routing_pins(state, gaussians) -> None:
    """Routing pinning (substrate): rows of families with K > 1 get a
    frozen route logit — realized as a gradient mask applied before
    every optimizer step (apply_elgs_routing_pins)."""
    pinned = {
        fid for fid in state.runtime.registry.active_ids()
        if state.runtime.registry.get(fid).routing_pinned
    }
    if pinned:
        ids = gaussians._elgs_family_ids
        mask = torch.zeros(ids.shape[0], dtype=torch.bool)
        for fid in pinned:
            mask |= ids == fid
        gaussians._elgs_routing_pin_mask = mask
    else:
        gaussians._elgs_routing_pin_mask = None


def apply_elgs_routing_pins(gaussians) -> None:
    """Zero the route-logit gradients of pinned rows (called after
    every backward, before optimizer.step — the freeze that makes the
    registry's pin log true in the optimizer)."""
    mask = getattr(gaussians, "_elgs_routing_pin_mask", None)
    if mask is None:
        return
    grad = getattr(gaussians._route_logit, "grad", None)
    if grad is not None and grad.shape[0] == mask.shape[0]:
        grad[mask.to(grad.device)] = 0.0


def run_post_refit_classification(state, gaussians, scene, iteration, render_unit_loss) -> None:
    """§8: per COMMITTED decision, on its own confirmation samples at
    POST-REFIT parameters — fixed-path decomposition. In the smoke
    wired path there is no tracker term: the render-only flag equals
    test (a) and the tracker-only flag is vacuously False (disclosed);
    the data floor uses the preregistered 0.25*SE rule."""
    from .acceptance import paired_snis_delta
    from .classification import DecisionFlags, classify, printable_label
    from .intervals import deserialize_state

    items = list(scene.getTrainCameras())
    results = {}
    for decision in state.committed_decisions:
        overrides = {
            int(fid): deserialize_state(payload)
            for fid, payload in decision["incumbent_intervals"].items()
        }
        samples = []
        for index, timestamp in decision["units"]:
            item = items[int(index)]
            loss_now = float(render_unit_loss(item, None))
            loss_incumbent = float(render_unit_loss(item, overrides))
            samples.append(SnisSample(
                unit=(int(index), float(timestamp)),
                nu_density=1.0,
                mix_density=1.0,
                loss_incumbent=loss_incumbent,
                loss_candidate=loss_now,
            ))
        delta = paired_snis_delta(samples, state.sampler_params.lambda_u)
        se = max(float(decision["se"]), 1e-12)
        passes_a = (delta + state.k_se * se) < 0.0  # prior+ledger removed
        below_floor = abs(delta) < 0.25 * se  # prereg_metrics rule
        flags = DecisionFlags(
            passes_prior_removed=passes_a,
            render_only_flag=passes_a,   # no tracker term in this path
            tracker_only_flag=False,     # vacuous without tracker evidence
            all_data_terms_below_floors=below_floor,
        )
        label = printable_label(classify(flags))
        results[decision["candidate_id"]] = {
            "delta_post_refit": delta,
            "class": label,
        }
    state.post_refit_done = True
    gaussians._elgs_checkpoint_extras["round_bookkeeping"]["post_refit_classification"] = results
    print(json.dumps({"elgs_post_refit_classification": {
        "iteration": iteration,
        "decisions": results,
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
    "apply_elgs_routing_pins",
    "build_interval_config",
    "elgs_summary",
    "filter_elgs_reserved",
    "infer_frame_dt",
    "load_structural_prereg",
    "maybe_run_elgs_schedule",
    "run_post_refit_classification",
    "seed_families",
    "setup_elgs",
]
