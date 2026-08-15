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
    evidence: object = None             # elgs.evidence_stack.EvidenceContext
    evidence_frame_time: dict = field(default_factory=dict)  # frame index -> model t
    evidence_pixel_scale: float = 1.0   # loaded raster -> report raster


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
        # SAME device selection as the restore branch above. A fresh run
        # previously left the runtime on CPU while the model sat on CUDA,
        # so the presence column and the probe's projected geometry lived
        # on different devices and `alpha[front]` raised
        #   "indices should be either on cpu or on the same device as the
        #    indexed tensor"
        # the first time a round actually reached `transmittance`.
        # Experiment 73 never reached it (no windows), so the asymmetry
        # survived until the evidence-aware proposer found a family with
        # windows in experiment 75.
        runtime = ElgsRuntime(
            registry, config, schedule,
            device=gaussians._xyz.device if gaussians._xyz.numel() else "cpu",
            dtype=torch.float32,
        )
        seeded = False
        rounds_run = []
    gaussians._pending_elgs_state = None

    sampler = FrozenSamplerParams(
        lambda_u=float(getattr(opt, "elgs_lambda_u")),
        pi_d_identity="uniform-train-units-v1",
        frozen=True,
    )

    # Reserved confirmation pool (spec §7): every 4th train unit index
    # is reserved at iteration 0 and pre-partitioned into the slot
    # grid; the trainer's refit dataset EXCLUDES these units
    # (filter_elgs_reserved), so refits never see confirmation samples.
    # Built BEFORE the trainer state that references it.
    from .acceptance import SlotGrid

    # Stratified reservation: within each timestamp group (the
    # cameras of one frame, stably ordered), reserve the diagonal
    # (frame_order + camera_order) % 4 == 0 — ~25% of units, rotating
    # across cameras AND spreading over time, so the §7 confirmation
    # measure can never collapse onto one view regardless of the
    # dataset's index ordering.
    n_units_total = len(cameras)
    by_time: dict = {}
    for i in range(n_units_total):
        t = float(getattr(cameras[i], "timestamp", 0.0))
        by_time.setdefault(t, []).append(i)
    reserved_list = []
    reserved_cam_positions = set()
    for f, t in enumerate(sorted(by_time)):
        group = sorted(
            by_time[t], key=lambda i: str(getattr(cameras[i], "image_name", i))
        )
        for c, index in enumerate(group):
            if (f + c) % 4 == 0:
                reserved_list.append((index, t))
                reserved_cam_positions.add(c)
    reserved = tuple(reserved_list)
    if reserved:
        times = [t for _, t in reserved]
        all_times = sorted(by_time)
        span = all_times[-1] - all_times[0] if len(all_times) > 1 else 0.0
        covered = max(times) - min(times)
        if len(reserved_cam_positions) < min(2, max(len(g) for g in by_time.values())):
            raise ContractError(
                "reserved confirmation pool covers fewer than 2 distinct "
                "camera positions — stratification failed for this ordering"
            )
        if span > 0 and covered < 0.5 * span:
            raise ContractError(
                "reserved confirmation pool spans less than half the "
                "sequence time range — stratification failed"
            )
    n_rounds = len(schedule.round_iterations)
    slots_per_pass = int(getattr(opt, "elgs_candidate_cap"))
    # The configured confirmation-sample count IS the slot size (a
    # declared value must never be silently overridden); the pool and
    # the degeneracy rule bound it fail-closed.
    units_per_slot = int(getattr(opt, "elgs_confirmation_samples"))
    if units_per_slot < 6:
        raise ContractError(
            "elgs_confirmation_samples must be >= 6 (bootstrap degeneracy rule)"
        )
    needed = n_rounds * slots_per_pass * units_per_slot
    if len(reserved) < needed:
        raise ContractError(
            f"reserved confirmation pool has {len(reserved)} units but the "
            f"slot grid needs {needed} (= {n_rounds} rounds x "
            f"{slots_per_pass} cap x {units_per_slot} samples); reduce "
            "elgs_candidate_cap or elgs_confirmation_samples for this scene"
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
        confirmation_samples=units_per_slot,
        base_seed=int(getattr(opt, "seed", 0) or 0),
        slot_grid=state_slot_grid,
        reserved_indices=frozenset(u[0] for u in reserved),
    )
    if pending is not None:
        # Stash the persisted evidence block for attach_evidence, which
        # runs after the registry/runtime are live.
        gaussians._restored_elgs_evidence = dict(loaded.get("evidence") or {})
        state.committed_decisions = list(
            loaded["round_bookkeeping"].get("committed_decisions", [])
        )
        state.post_refit_done = bool(
            loaded["round_bookkeeping"].get("post_refit_done", False)
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
        "confirmation_refs": (
            {k: [tuple(u) for u in v] for k, v in loaded["confirmation_refs"].items()}
            if pending is not None else {}
        ),
        "moment_reset_log": (
            list(loaded["moment_reset_log"]) if pending is not None else []
        ),
        "round_bookkeeping": {
            "seeded": seeded,
            "rounds_run": rounds_run,
            "committed_decisions": state.committed_decisions,
            "post_refit_done": state.post_refit_done,
        },
    }
    if not state.seeded:
        seed_families(state, gaussians, iteration=0)
    else:
        _refresh_logit_param_group(state, gaussians)
        # The captured optimizer state strips the elgs_a group (see
        # elgs.state_io.strip_optimizer_group), so restored a-logits
        # start with fresh Adam moments — disclosed and logged.
        gaussians._elgs_checkpoint_extras["moment_reset_log"].append(
            {"iteration": 0, "op": "restore",
             "tensors_reset": len(state.runtime.logit_parameters())}
        )
    _refresh_routing_pins(state, gaussians)
    attach_evidence(state, gaussians, scene, dataset, opt)
    print(json.dumps({"elgs_setup": {
        "schedule": schedule_key,
        "frame_dt": frame_dt,
        "time_span": time_span,
        "restored": pending is not None,
        "families": len(registry.active_ids()),
        "evidence": state.evidence is not None,
    }}, sort_keys=True))
    return state


def attach_evidence(state: ElgsTrainerState, gaussians, scene, dataset, opt) -> None:
    """Bind the sealed tracks artifact, or stay photometric-only.

    No `elgs_tracks_dir` => the run is photometric-only by declaration
    and this is a no-op (the verified M0 path, unchanged). A tracks dir
    that is present but incompatible RAISES: silently degrading a
    declared evidence run to photometric-only would make the two
    indistinguishable in the artifact.
    """
    tracks_dir = str(getattr(opt, "elgs_tracks_dir", "") or "").strip()
    if not tracks_dir:
        return

    from .evidence_stack import (
        build_evidence_context,
        load_evidence_constants,
        resolve_smoke_report_cap,
    )
    from .tracks_loader import load_sealed_tracks

    smoke = bool(getattr(opt, "elgs_smoke_schedule", False))
    for key in ("elgs_beta", "elgs_tau_b", "elgs_c_cap", "elgs_r_site",
                "elgs_binding_threshold"):
        if float(getattr(opt, key)) < 0:
            raise ContractError(f"{key} is unset (-1 sentinel); the evidence path needs it")

    # THE ACTUAL EVIDENCE-BEARING GATE. `elgs_smoke_schedule` is a
    # SCHEDULE switch; it does NOT make a run non-evidence-bearing.
    # scripts/submit_apollo.py derives `evidence_bearing` purely from
    # working-tree cleanliness, and the two flags are independent — so a
    # clean-tree run stamped `evidence_bearing: true` could otherwise set
    # elgs_smoke_schedule and consume unfrozen hand-picked head constants
    # under an evidence-bearing ledger line. The run manifest the
    # submission wrapper writes into the context is the authority, and it
    # is read here.
    if smoke and _manifest_is_evidence_bearing():
        raise ContractError(
            "the run manifest stamps evidence_bearing=true, so the smoke-tier "
            "evidence heads may not be loaded; freeze the heads in "
            "prereg_evidence_heads_v1.json or submit this run as "
            "non-evidence-bearing"
        )

    constants = load_evidence_constants(
        str(getattr(opt, "elgs_prereg_dir", "configs/elgs")), smoke=smoke
    )

    # `source_path` lives on ModelParams (`dataset`), which
    # ModelParams.extract() has already made absolute. It is NOT on the
    # Scene and NOT on OptimizationParams: reading those yielded '' and
    # the sequence check correctly refused the run rather than scoring
    # scissor's tracks against an unidentified scene.
    source_path = str(getattr(dataset, "source_path", "") or "").strip()
    if not source_path:
        raise ContractError(
            "the evidence path needs the scene's source_path to establish "
            "sequence identity, and ModelParams carries none"
        )
    sequence_id = os.path.basename(str(source_path).replace("\\", "/").rstrip("/"))
    cameras = [_unpack_camera(item)[0] for item in scene.getTrainCameras()]
    camera_ids = sorted({_camera_id_of(c) for c in cameras} - {None})
    frame_time = _scene_frame_times(cameras)
    if not frame_time:
        raise ContractError(
            "cannot recover DiVa-360 frame indices from the loaded scene's "
            "camera image paths; the evidence path needs a frame->model-time "
            "map and will not guess one"
        )

    sealed = load_sealed_tracks(
        tracks_dir,
        scene_sequence_id=sequence_id,
        scene_camera_ids=camera_ids,
        scene_frame_indices=sorted(frame_time),
        scene_dir=source_path,
        require_evidence_admissible=not smoke,
    )

    report_raster = _declared_report_raster(source_path)
    if report_raster is None:
        # FAIL CLOSED. Defaulting pixel_scale to 1.0 and skipping the
        # D_img check — as a previous revision did — silently restores
        # the exact pixel-domain mismatch that made the evidence term
        # favour truncation regardless of the data. An evidence run whose
        # report raster cannot be determined must refuse, not guess.
        raise ContractError(
            f"cannot determine the report raster for {source_path}: "
            "transforms_train.json is missing, unreadable, or carries no "
            "frame with both 'w' and 'h'. The evidence path compares report "
            "positions with bridge projections and will not assume they "
            "share a pixel domain."
        )
    # SMOKE-ONLY evaluation bound. Fail-closed on both conditions; the
    # manifest is the authority on evidence-bearing status, exactly as it
    # is for the smoke heads above.
    smoke_cap = resolve_smoke_report_cap(
        int(getattr(opt, "elgs_smoke_max_reports_per_window", 0) or 0),
        smoke_schedule=smoke,
        evidence_bearing=_manifest_is_evidence_bearing(),
    )
    state.evidence = build_evidence_context(
        sealed,
        constants,
        gaussians._xyz.detach(),
        gaussians._elgs_family_ids.to(gaussians._xyz.device),
        r_site=float(getattr(opt, "elgs_r_site")),
        binding_threshold=float(getattr(opt, "elgs_binding_threshold")),
        c_cap=int(getattr(opt, "elgs_c_cap")),
        beta=float(getattr(opt, "elgs_beta")),
        tau_b=float(getattr(opt, "elgs_tau_b")),
        report_raster=report_raster,
    )
    state.evidence.smoke_max_reports_per_window = smoke_cap
    # RESTORE the persisted binding rather than keeping the one just
    # re-derived. A previous revision serialized the binding and claimed
    # a resume restored it; only the write existed, so a resumed run
    # re-bound clusters against drifted primitive positions and could
    # silently bind them to different families than the original run.
    # `pending` is a local of setup_elgs and is NOT in scope here — an
    # earlier revision referenced it, which made every run that set
    # elgs_tracks_dir die with NameError at setup. setup_elgs stashes the
    # restored block on the model instead.
    restored = getattr(gaussians, "_restored_elgs_evidence", None)
    if restored and restored.get("binding"):
        from .clusters import BindingTable

        state.evidence.binding = BindingTable.from_state(restored["binding"])
        print(json.dumps({"elgs_evidence_binding_restored": {
            "clusters": len(restored.get("clusters", [])),
        }}, sort_keys=True))
    # Projected pixels must be mapped back into the REPORT raster before
    # g_pos ever sees them (see elgs.probe_model). The scale comes from
    # the INTRINSICS, not the raster ratio: utils.camera_utils.loadCam
    # divides fl_x by the UNROUNDED `scale` but sets the raster to
    # round(w/scale), round(h/scale). A raster ratio is therefore only
    # approximate -- DiVa-360 at resolution 4 gives round(550/4) = 138
    # and a height ratio of 3.9855 against a width ratio of 4.0, so an
    # axis-agreement test on the rasters rejects the one committed
    # configuration over half a pixel of rounding. fl ratios are exact.
    declared_fl = _declared_focal(source_path)
    loaded_fl = float(getattr(cameras[0], "fl_x", 0.0) or 0.0)
    if declared_fl and loaded_fl > 0.0:
        state.evidence_pixel_scale = declared_fl / loaded_fl
    else:
        loaded_width = float(cameras[0].image_width)
        if loaded_width <= 0:
            raise ContractError("loaded camera reports a non-positive width")
        state.evidence_pixel_scale = report_raster[0] / loaded_width
    # Frame index -> MODEL time, read off the scene's own cameras. NOT
    # derived from the artifact's fps: the reader divides `time` by
    # `frame_ratio`, so 1/fps is the wrong unit whenever frame_ratio > 1
    # and every interval comparison would be silently mis-scaled.
    # `state.frame_dt` (inferred from the scene at setup) is left alone —
    # `build_interval_config` already used it.
    state.evidence_frame_time = frame_time
    gaussians._elgs_checkpoint_extras["evidence"] = state.evidence.to_state()
    print(json.dumps({"elgs_evidence_bound": {
        "tier": constants.tier,
        "clusters": len(state.evidence.clusters),
        "bound": sum(
            1 for c in state.evidence.clusters
            if state.evidence.binding.family_of(c.cluster_id) is not None
        ),
        **sealed.provenance(),
    }}, sort_keys=True))


def _camera_id_of(camera):
    """The DiVa-360 integer camera id behind a Camera, or None.

    `image_name` is the frame stem (`00000000`) and the camera lives in
    the parent directory (`undist/cam07/00000000`), so the id is parsed
    from `image_path`, falling back to `image_name`.
    """
    from depth_visibility.errors import SchemaError
    from elgs import diva360_schema as dschema

    for attribute in ("image_path", "image_name"):
        text = str(getattr(camera, attribute, "") or "")
        try:
            return dschema.parse_camera_id(text)
        except SchemaError:
            continue
    return None


def _scene_frame_times(cameras) -> dict:
    """frame index -> model timestamp, read off the scene's own cameras.

    Fails closed on a frame index that carries two different timestamps:
    that would mean the loaded scene is not the static-rig temporal
    conversion the tracks artifact was built against.
    """
    from depth_visibility.errors import SchemaError
    from elgs import diva360_schema as dschema

    out: dict[int, float] = {}
    for camera in cameras:
        for attribute in ("image_path", "image_name"):
            text = str(getattr(camera, attribute, "") or "")
            try:
                index = dschema.parse_frame_index(text, "")
            except SchemaError:
                continue
            timestamp = float(getattr(camera, "timestamp", 0.0))
            if index in out and abs(out[index] - timestamp) > 1e-9:
                raise ContractError(
                    f"frame index {index} carries two model timestamps "
                    f"({out[index]} and {timestamp}); the scene is not a "
                    "static-rig temporal conversion"
                )
            out[index] = timestamp
            break
    return out


def _manifest_is_evidence_bearing(manifest_filename: str = "elgs_run_manifest.json") -> bool:
    """Whether the submission wrapper stamped this run evidence-bearing.

    The manifest is written O_EXCL into the git-archive context root, so
    in-container it sits in the working directory (the same file
    `scripts.submit_apollo.runtime_assertions` re-verifies). Absent
    manifest => a local/test invocation => False, which keeps the CPU
    suite and any bare `python main.py` runnable.
    """
    path = os.path.join(os.getcwd(), manifest_filename)
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return bool(json.load(handle).get("evidence_bearing", False))
    except (OSError, ValueError):
        # An unreadable manifest is not a licence to load smoke heads.
        return True


def _declared_report_raster(source_path) -> tuple[float, float] | None:
    """The (w, h) the artifact's report positions live in.

    `build_elgs_tracks` bounds reports by the camera width/height it read
    from `transforms_train.json`, i.e. the FULL declared raster — not the
    possibly-downscaled raster the trainer loads.
    """
    path = os.path.join(str(source_path), "transforms_train.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            frames = json.load(handle).get("frames") or []
    except (OSError, ValueError):
        return None
    for frame in frames:
        if "w" in frame and "h" in frame:
            return float(frame["w"]), float(frame["h"])
    return None


def _declared_focal(source_path) -> float | None:
    """The converted scene's declared `fl_x`, in REPORT pixels.

    `loadCam` divides fl_x by the loader's unrounded `scale`, so
    declared_fl_x / camera.fl_x recovers that scale EXACTLY, with no
    dependence on how the raster rounded.
    """
    path = os.path.join(str(source_path), "transforms_train.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            frames = json.load(handle).get("frames") or []
    except (OSError, ValueError):
        return None
    for frame in frames:
        if "fl_x" in frame:
            return float(frame["fl_x"])
    return None


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
    # Eligibility for the smoke operation is unchanged: K == 1.
    eligible = [
        family_id
        for family_id in state.runtime.registry.active_ids()
        if state.runtime.registry.get(family_id).interval.K == 1
    ]
    if not eligible:
        return proposals

    # SMOKE-TIER SELECTION ONLY. Prefer a family that is bound to a
    # cluster AND carries at least one evidence window, so the round
    # actually exercises the evidence path instead of falling through to
    # the insufficient-evidence case. Experiment 73 rejected all three
    # rounds with q_values 0 for exactly this reason: ranking by span
    # alone is evidence-BLIND, every family is K=1 spanning so the span
    # is identical, and the family it landed on had no bound cluster.
    #
    # The preference reads WINDOW AVAILABILITY and family ids ONLY. It
    # never inspects a likelihood, a q value, an evidence-delta sign, or
    # an acceptance outcome, so it cannot select for a preferred result.
    # This is the smoke proposer; the production candidate-generation
    # semantics are untouched.
    preferred = eligible
    if state.evidence is not None:
        with_windows = state.evidence.families_with_windows(eligible)
        if with_windows:
            preferred = list(with_windows)

    # Deterministic: largest span, ties broken by LOWEST family id.
    # Spanning K=1 families all share one span, so the id tie-break is
    # what actually decides and the choice is stable across runs.
    family_id = max(
        preferred,
        key=lambda f: (float(state.runtime.realization(f).lens[0]), -f),
    )
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

    # Planners must see the LIVE interval state (the optimizer has
    # been stepping the a-logits since the last flush); without this,
    # proposals validate against stale spans and commits discard the
    # learned interval movement.
    state.runtime.flush_to_registry()
    round_index = len(state.rounds_run)
    proposals = _propose_smoke_candidates(state, iteration)
    dataset = scene.getTrainCameras()  # indexed lazily, never listed
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
            item = dataset[index]
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

    candidate_families = sorted(
        {int(f) for p in proposals for f in p.plan.family_ids}
    )
    round_evidence = _refresh_round_evidence(
        state, gaussians, scene, round_index, family_ids=candidate_families
    )
    exact_deltas_fn = None
    if round_evidence is not None:
        from .evidence_stack import evidence_exact_delta

        def exact_deltas_fn(plan):
            # A ContractError raised here would be caught by
            # `round_driver.run_pass` and recorded as an ordinary
            # candidate REJECTION, so a systematic evidence-path defect
            # would present as "every candidate rejected" — a
            # publishable-looking null result produced by a bug. Re-raise
            # outside the ContractError hierarchy so it stops the run
            # loudly instead (AGENTS.md: never a silent no-state
            # observation).
            try:
                return evidence_exact_delta(
                    state.evidence, round_evidence, plan,
                    state.runtime.registry, state.config,
                    frame_time=state.evidence_frame_time,
                )
            except ContractError as failure:
                raise RuntimeError(
                    f"EL-GS evidence path failed on a {plan.op} candidate for "
                    f"families {plan.family_ids}: {failure}"
                ) from failure

    outcome = run_pass(
        proposals, state.runtime, state.bundle, sample_builder,
        state.sampler_params, k_se=state.k_se, base_seed=state.base_seed,
        round_index=round_index, pass_index=0, iteration=iteration,
        candidate_cap=state.candidate_cap,
        exact_deltas_fn=exact_deltas_fn,
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


def _refresh_round_evidence(state, gaussians, scene, round_index, family_ids=None):
    """Recompute and freeze the round's q snapshot, or return None.

    Returns None exactly when the run is photometric-only (no evidence
    context), so the caller passes `exact_deltas_fn=None` and the
    acceptance decision is bit-identical to the verified M0 path.
    """
    if state.evidence is None:
        return None

    from .evidence_stack import refresh_round_evidence
    from .probe_model import ModelProbe

    cameras = {}
    for item in scene.getTrainCameras():
        camera = _unpack_camera(item)[0]
        camera_id = _camera_id_of(camera)
        # One Camera per id is enough: the rig is static (the tracks
        # builder asserts it), so extrinsics and intrinsics do not vary
        # across frames and only the presence multiplier is time-varying.
        if camera_id is not None and camera_id not in cameras:
            cameras[camera_id] = camera

    ids = gaussians._elgs_family_ids

    def presence_at(timestamp, present_family):
        """Per-row presence at t, with `present_family` FORCED present.

        This is the spec §3 "family-present" transmittance: q asks what
        the tracker COULD have seen if the family were there, so the
        family under evaluation must not occlude-or-vanish by its own
        current interval. Forcing is applied to a COPY (frozen-functional
        snapshot, spec rev-2 item 3) — the live column is never mutated.
        """
        column = state.runtime.presence_multiplier(ids, float(timestamp))
        if present_family is None:
            return column
        column = column.clone()
        mask = (ids == int(present_family)).reshape(-1, 1).to(column.device)
        return column.masked_fill(mask, 1.0)

    probe = ModelProbe(
        gaussians,
        cameras,
        presence_at=presence_at,
        frame_to_time=state.evidence_frame_time,
        pixel_scale=state.evidence_pixel_scale,
    )
    # SCOPE q TO THE FAMILIES A CANDIDATE ACTUALLY TOUCHES.
    # `evidence_exact_delta` iterates plan.family_ids only, so q computed
    # for any other family is never read. Computing it for every active
    # family is what makes the round cost unbounded: on scissor, 512
    # singleton clusters x ~21 cameras x window frames x 3 bridges x 7
    # sigma points is ~10^6 full-model transmittance passes per round,
    # none of which reaches a decision. Values are IDENTICAL either way
    # (q for family f does not depend on which candidates exist), so this
    # narrows what is computed, never what is compared. The snapshot is
    # still frozen before any candidate is scored.
    active = list(state.runtime.registry.active_ids())
    scoped = sorted(set(active) & {int(f) for f in family_ids}) if family_ids else active
    evidence = refresh_round_evidence(
        state.evidence,
        probe,
        round_index=round_index,
        family_ids=scoped,
    )
    gaussians._elgs_checkpoint_extras["evidence"] = state.evidence.to_state()
    print(json.dumps({"elgs_evidence_round": {
        "round_index": round_index,
        "families_scoped": len(scoped),
        "families_active": len(active),
        "windows": len(evidence.windows),
        "q_values": evidence.n_q_values,
        "families_with_windows": len({w.family_id for w in evidence.windows}),
        "photometric_only_families": sum(
            1 for v in evidence.coverage.values() if v.get("photometric_only")
        ),
        "reports": evidence.report_accounting,
    }}, sort_keys=True))
    return evidence


def filter_elgs_reserved(training_dataset, state):
    """Exclude the reserved confirmation units from the refit
    training data (spec §7: refits never see confirmation samples).

    Index-based Subset — NEVER materializes items, so the dataloader
    path's lazy image decoding is preserved (a CameraDataset decodes
    on __getitem__)."""
    if state is None:
        return training_dataset
    from torch.utils.data import Subset

    kept = [
        index for index in range(len(training_dataset))
        if index not in state.reserved_indices
    ]
    return Subset(training_dataset, kept)


def _refresh_routing_pins(state, gaussians) -> None:
    """Routing pinning (substrate): rows of families with K > 1 get a
    frozen route logit. Only the pinned FAMILY SET is cached — the
    row mask is rebuilt on every application from the live family-id
    column, so densification can never silently unpin rows."""
    gaussians._elgs_pinned_families = {
        fid for fid in state.runtime.registry.active_ids()
        if state.runtime.registry.get(fid).routing_pinned
    }


def apply_elgs_routing_pins(gaussians) -> None:
    """Zero the route-logit gradients of pinned rows (called after
    every backward, before optimizer.step). The mask is derived from
    the CURRENT family-id column each call, so row-count changes
    (densify/prune) are always reflected; a column/grad length
    mismatch is a wiring bug and fails closed."""
    pinned = getattr(gaussians, "_elgs_pinned_families", None)
    if not pinned:
        return
    grad = getattr(gaussians._route_logit, "grad", None)
    if grad is None:
        return
    ids = gaussians._elgs_family_ids
    if ids.shape[0] != grad.shape[0]:
        raise ContractError(
            f"routing-pin mask desync: family column has {ids.shape[0]} rows, "
            f"route-logit grad has {grad.shape[0]}"
        )
    # Vectorized (runs before EVERY optimizer step): O(rows log pinned)
    # via torch.isin, not a per-family python loop.
    mask = torch.isin(ids, torch.tensor(sorted(pinned), dtype=ids.dtype))
    grad[mask.to(grad.device)] = 0.0


def crn_seed_for_decision(state, decision) -> int:
    """Deterministic post-refit bootstrap seed from the decision's
    slot coordinates (index arithmetic, no hashing — spec §7)."""
    from .acceptance import crn_seed

    return crn_seed(
        state.base_seed + 1_000_000,  # disjoint from decision-time seeds
        int(decision["round_index"]),
        0,
        int(decision["iteration"]) % 1_000_003,
    )


def run_post_refit_classification(state, gaussians, scene, iteration, render_unit_loss) -> None:
    """§8: per COMMITTED decision, on its own confirmation samples at
    POST-REFIT parameters — fixed-path decomposition. In the smoke
    wired path there is no tracker term: the render-only flag equals
    test (a) and the tracker-only flag is vacuously False (disclosed);
    the data floor uses the preregistered 0.25*SE rule.

    DISCLOSED ASYMMETRY (fixed path, spec §8's own framing): the
    committed arm kept training through the refit while the restored
    incumbent is frozen at round time, so this comparison is
    structurally biased toward the committed arm — the labels are an
    operational decomposition, never statistical support (the
    mandatory qualifier states this on every printed label)."""
    from .acceptance import paired_cluster_bootstrap_se, paired_snis_delta
    from .classification import DecisionFlags, classify, printable_label
    from .intervals import deserialize_state

    dataset = scene.getTrainCameras()  # indexed lazily, never listed
    results = {}
    for decision in state.committed_decisions:
        overrides = {
            int(fid): deserialize_state(payload)
            for fid, payload in decision["incumbent_intervals"].items()
        }
        samples = []
        for index, timestamp in decision["units"]:
            item = dataset[int(index)]
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
        # §8 evaluates ON the post-refit samples: the SE is
        # re-bootstrapped from them (units_per_slot >= 6 satisfies the
        # degeneracy rule), never reused from decision time.
        se = paired_cluster_bootstrap_se(
            samples, state.sampler_params.lambda_u,
            seed=crn_seed_for_decision(state, decision),
        )
        passes_a = (delta + state.k_se * se) < 0.0  # prior+ledger removed
        below_floor = abs(delta) < 0.25 * max(se, 1e-12)  # prereg_metrics rule
        flags = DecisionFlags(
            passes_prior_removed=passes_a,
            render_only_flag=passes_a,   # no tracker term in this path
            tracker_only_flag=False,     # vacuous without tracker evidence
            all_data_terms_below_floors=below_floor,
        )
        label = printable_label(classify(flags))
        results[decision["candidate_id"]] = {
            "delta_post_refit": delta,
            "se_post_refit": se,
            "class": label,
        }
    state.post_refit_done = True
    bookkeeping = gaussians._elgs_checkpoint_extras["round_bookkeeping"]
    bookkeeping["post_refit_classification"] = results
    bookkeeping["post_refit_done"] = True
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
