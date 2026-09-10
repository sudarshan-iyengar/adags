"""Render-time gated evaluation of a REAL N3V checkpoint (paired gate ON/OFF).

EXPLORATORY. Restores an `ivv_protocol_300f_6k` checkpoint that was trained
WITHOUT EL-GS, seeds a localized episodic-presence program onto the RESTORED
cloud, and renders every held-out frame in a requested absolute frame range
TWICE from the same model: once with the localized gate live and once with the
gate off. The two arms differ in nothing but `pc.elgs_runtime`.

Why this script exists rather than `main.py --val`
--------------------------------------------------
`validation()` (main.py:1035-1105) restores a checkpoint and renders the test
split, but it NEVER calls `setup_elgs`. `gaussians.elgs_runtime` therefore
stays None, the renderer's `elgs_active` is False
(gaussian_renderer/__init__.py:224), and every frame is drawn under the
ordinary temporal marginal. An EL-GS program cannot be applied on that path at
all. `scripts/eval_lrv1_event.py` does call `setup_elgs`, but it is bound to
the synthetic LRV fixtures: it needs `event_spec.json` and a `gt_identity/`
directory that no N3V scene has.

The seeding path, and why the refusal at the top matters
--------------------------------------------------------
`setup_elgs` has two branches (elgs/trainer_hooks.py:155-196). If the restored
checkpoint carries an `elgs_state` payload it takes the RESTORE branch, sets
`seeded=True`, and `seed_families` is never called -- so `--program` would be
read for nothing and the reported gate would be whatever the checkpoint
trained with. This script refuses that case outright. On a checkpoint with no
`elgs_state` (the ivv arms, which train with `elgs_enable` unset) the fresh
branch runs and `seed_families` (:864-1031) binds the program to
`gaussians._xyz` AS RESTORED, which is the trained cloud. That is the whole
point: a `membership_mode: row_ids` program computed on this checkpoint
matches by row count and xyz fingerprint (:777-796), so the gated row set is
exactly the estimated one rather than a re-derivation over a fresh cloud.

The gate-off arm
----------------
Disabling is a two-attribute edit on the model, not a second model:
`elgs_runtime = None` and `_elgs_local_presence = False`. Reading the renderer,
`elgs_active` (:224) becomes False, `_temporal_multiplier` (:232-245) returns
`pc.get_marginal_t(t)` unchanged, and `elgs_static_multiplier` stays None so
the static twin at :352-356 is unmodulated. That is bit-for-bit the ungated
substrate path. `get_elgs_gated_row_mask` recomputes from the runtime on every
call and caches nothing (scene/gaussian_model.py:263-275), so no stale mask can
leak into the off arm; this script asserts that by checking the accessor raises
while the gate is off.

Anti-leakage
------------
The gate is a pure function of `--program`: no tracks artifact
(`elgs_tracks_dir` must be empty, which makes `attach_evidence` a no-op at
trainer_hooks.py:364-366), no structural rounds, and `elgs_a_lr == 0.0` so no
boundary is learned. Held-out camera ids are asserted disjoint from the
training ids. This script itself opens held-out (cam00) ground truth, but only
inside the scoring loop and only after the program has been frozen, seeded and
reported.

Output
------
    <out_dir>/report.json                 precondition + provenance + metrics
    <out_dir>/gated/renders/<FRAME>.png   gate ON,  absolute frame index
    <out_dir>/ungated/renders/<FRAME>.png gate OFF, absolute frame index
    <out_dir>/gt/<FRAME>.png              held-out ground truth

`report.json` is written TWICE: once carrying only the precondition and
provenance, before a single pixel is scored, and once again at the end with
the metrics appended. The precondition therefore exists on disk even if the
scoring loop dies, and it cannot have been chosen after seeing a number.

PSNR pools over channels and pixels (utils.image_utils.psnr on a batched
(1,3,H,W) pair) and renders are clamped to [0,1] -- the same convention
`GaussianExtractor.reconstruction` uses for `--val`
(utils/mesh_utils.py:107-134).
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# INSERT, not append: the admitted image ships a `pointops2` whose `functions`
# subpackage is absent and `utils.general_utils` imports it unconditionally.
# Same reason as scripts/eval_lrv1_event.py:50.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCHEMA_VERSION = "n3v-gated-eval-v1"

# --restore_state: the SECOND mode, added 2026-09-10 after the montage of the
# absence-fixture cells showed a full bottle inside the gap on every gated
# arm. main.py's `validation()` restores a checkpoint and renders WITHOUT
# calling setup_elgs, so every `--val` metric of an EL-GS cell is rendered
# through the ordinary temporal marginal, gate off. In restore mode this
# script takes setup_elgs's RESTORE branch (the checkpoint's own elgs_state),
# PROVES the restored intervals are the supplied --program (the same lineage-
# key match gate_cell_precondition.py uses), and then scores the model with
# its gate on and off exactly as the fresh mode does.


def resolve_seeding_mode(has_pending_state, restore_flag):
    """'fresh' or 'restore', or raise: the two modes never fall through."""
    from depth_visibility.errors import ContractError

    if restore_flag:
        if not has_pending_state:
            raise ContractError(
                "--restore_state was given but the checkpoint carries no "
                "elgs_state; this is an ungated checkpoint, use the fresh mode"
            )
        return "restore"
    if has_pending_state:
        raise ContractError(
            "the checkpoint carries elgs_state; setup_elgs would restore that "
            "program and skip seeding, so --program would have no effect. This "
            "evaluator is for checkpoints trained WITHOUT EL-GS; score an "
            "EL-GS checkpoint with its own state via --restore_state"
        )
    return "fresh"
# Mirrored from elgs.trainer_hooks so the pure-python helpers below stay
# importable without torch. Checked against the real constant at runtime.
EPISODE_PROGRAM_SCHEMA_V2 = "adags-episode-program-v2"


# ---------------------------------------------------------------------------
# PURE HELPERS -- stdlib only, so the tests can exercise the reading rule and
# the precondition arithmetic without torch or CUDA.
# ---------------------------------------------------------------------------


def absolute_frame_from_name(image_name):
    """The absolute frame index encoded in a camera's image name.

    MIRRORS `gaussian_renderer._frame_index_from_camera` (:34-45): the LAST
    run of digits in the file stem. n3v2blender writes `images/cam00_0150`, so
    `cam00_0150` -> 150 and the leading camera number is never mistaken for a
    frame. Returns None when the name carries no digits.
    """
    stem = str(image_name).rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
    matches = re.findall(r"\d+", stem)
    if not matches:
        return None
    return int(matches[-1])


def camera_id_from_name(image_name):
    """The camera number from a `cam<NN>_<FFFF>` name, or None."""
    stem = str(image_name).rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
    match = re.match(r"^cam(\d+)_", stem)
    return int(match.group(1)) if match else None


def select_frame_range(image_names, frame_lo, frame_hi):
    """Positions of the names whose absolute frame lies in [lo, hi], INCLUSIVE.

    Returns `[(position, frame), ...]` sorted by frame then position, so the
    render order is deterministic and independent of the loader's ordering.
    """
    if frame_lo > frame_hi:
        raise ValueError("--frame_range needs A <= B, got %d %d" % (frame_lo, frame_hi))
    picked = []
    for position, name in enumerate(image_names):
        frame = absolute_frame_from_name(name)
        if frame is None:
            continue
        if frame_lo <= frame <= frame_hi:
            picked.append((position, frame))
    picked.sort(key=lambda pair: (pair[1], pair[0]))
    return picked


def render_filename(frame):
    """`<ABSFRAME:05d>.png`. ABSOLUTE, never the window-relative index that
    `utils/mesh_utils.export_image` (:181-202) writes."""
    return "%05d.png" % int(frame)


def program_gaps_seconds(program):
    """`{group_id: [[a, b], ...]}` in model-time seconds, from a v2 program.

    Reads the same `groups[*].gaps` field `_load_episode_program_v2`
    (trainer_hooks.py:727) reads, so the precondition and the seeder are
    quoting one source.
    """
    gaps = {}
    for entry in program.get("groups") or []:
        gaps[int(entry["group"])] = [
            [float(a), float(b)] for a, b in entry["gaps"]
        ]
    return gaps


def frames_inside_gaps(frame_times, gaps_by_group):
    """Frames whose model time falls inside ANY declared gap.

    `frame_times` is `[(frame, timestamp_seconds), ...]`. Returns the sorted
    list of distinct frames. Endpoints are inclusive: a frame exactly on a gap
    boundary is counted, which is the permissive reading -- the stricter
    exact-absence test is done at runtime against the realized presence.
    """
    hit = set()
    for frame, timestamp in frame_times:
        for gaps in gaps_by_group.values():
            for lo, hi in gaps:
                if lo <= float(timestamp) <= hi:
                    hit.add(int(frame))
                    break
    return sorted(hit)


def gap_frames_from_seconds(gaps_by_group, frame_dt):
    """`{group: [[first_frame, last_frame], ...]}` for reporting only.

    The frame indices a gap covers under a uniform `frame_dt`. Reported so a
    reader can compare the program's seconds against the event window in
    frames without redoing the division; nothing is decided on it.
    """
    if frame_dt <= 0:
        raise ValueError("frame_dt must be positive")
    out = {}
    for group, gaps in gaps_by_group.items():
        spans = []
        for lo, hi in gaps:
            first = int(-(-lo // frame_dt)) if lo > 0 else 0  # ceil
            last = int(hi // frame_dt)                        # floor
            spans.append([first, last])
        out[int(group)] = spans
    return out


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_provenance(repo_root):
    """`{commit, dirty}` or `{error}`. Never raises -- provenance is recorded,
    never enforced, by this script."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
            text=True, timeout=30,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True,
            text=True, timeout=30,
        )
        if commit.returncode != 0:
            return {"error": commit.stderr.strip()[:200]}
        return {
            "commit": commit.stdout.strip(),
            "dirty": bool(status.stdout.strip()),
        }
    except Exception as exc:                                   # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


def _merge_config(args, config_path):
    """Identical to scripts/eval_lrv1_event.py:65-78 and
    scripts/estimate_episodes.py:1240 -- an unknown key is a hard failure."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    def recursive_merge(key, host):
        if isinstance(host[key], dict):
            for kk in host[key]:
                recursive_merge(kk, host[key])
        else:
            assert hasattr(args, key), f"unknown config key {key}"
            setattr(args, key, host[key])

    for key in config:
        recursive_merge(key, config)


# ---------------------------------------------------------------------------
# RUNTIME
# ---------------------------------------------------------------------------


def _unpack(item):
    """`(image_or_None, camera)` from a CameraDataset item or a bare camera."""
    if isinstance(item, (tuple, list)):
        return item[0], item[1]
    return None, item


class _gate_disabled:
    """Context manager: the SAME model rendered as if EL-GS were never set up.

    Sets `elgs_runtime = None` and `_elgs_local_presence = False`, which is
    exactly the state a non-EL-GS run leaves the model in
    (scene/gaussian_model.py:237-238). The renderer then takes
    `elgs_active = False` (gaussian_renderer/__init__.py:224), returns
    `pc.get_marginal_t(t)` from `_temporal_multiplier` (:235) and leaves
    `elgs_static_multiplier` None so the static twin (:352-356) is
    unmodulated. Restored on exit, including on exception.
    """

    def __init__(self, gaussians):
        self.gaussians = gaussians
        self._runtime = None
        self._local = None

    def __enter__(self):
        self._runtime = getattr(self.gaussians, "elgs_runtime", None)
        self._local = getattr(self.gaussians, "_elgs_local_presence", False)
        self.gaussians.elgs_runtime = None
        self.gaussians._elgs_local_presence = False
        return self.gaussians

    def __exit__(self, exc_type, exc, tb):
        self.gaussians.elgs_runtime = self._runtime
        self.gaussians._elgs_local_presence = self._local
        return False


def verify_gate_off(gaussians):
    """Prove the off arm reads no cached gate. Returns a dict for the report.

    `get_elgs_gated_row_mask` and `get_elgs_presence` raise when the runtime is
    detached (scene/gaussian_model.py:256-257, :271-272), so a RuntimeError
    here is the evidence that the mask is recomputed per call and that nothing
    stale survives the toggle.
    """
    checks = {
        "elgs_runtime_is_none": getattr(gaussians, "elgs_runtime", None) is None,
        "local_presence_false": not bool(
            getattr(gaussians, "_elgs_local_presence", False)
        ),
        "no_presence_override": getattr(gaussians, "_elgs_presence_override", None)
        is None,
    }
    for accessor in ("get_elgs_gated_row_mask", "get_elgs_presence"):
        try:
            if accessor == "get_elgs_presence":
                getattr(gaussians, accessor)(0.0)
            else:
                getattr(gaussians, accessor)()
            checks[accessor + "_raised"] = False
        except RuntimeError:
            checks[accessor + "_raised"] = True
    return checks


def assert_gate_is_program_only(opt, state, program_path):
    """The gate must be a pure function of `--program`."""
    from depth_visibility.errors import ContractError

    tracks = str(getattr(opt, "elgs_tracks_dir", "") or "").strip()
    if tracks:
        raise ContractError(
            "anti-leakage: elgs_tracks_dir is set to %r, so the gate could "
            "depend on a tracks artifact as well as the program; this "
            "evaluator scores a program-only gate" % tracks
        )
    if bool(getattr(opt, "elgs_rounds_enabled", True)):
        raise ContractError(
            "elgs_rounds_enabled must be false: a structural round would move "
            "the boundaries away from the supplied program"
        )
    if float(getattr(opt, "elgs_a_lr", -1.0)) != 0.0:
        raise ContractError(
            "elgs_a_lr must be exactly 0.0: a nonzero boundary learning rate "
            "means the rendered gate is not the program that was frozen"
        )
    if os.path.abspath(str(state.oracle_episodes)) != os.path.abspath(program_path):
        raise ContractError(
            "the seeded program is %r but --program is %r"
            % (state.oracle_episodes, program_path)
        )
    if not state.local_presence:
        raise ContractError("elgs_local_presence must be true for a localized gate")
    if state.routing_pins_enabled:
        raise ContractError(
            "elgs_routing_pins_enabled must be false (configs/lrv3/a1_local.yaml "
            "defect 4): pinned routing freezes the gated rows' mixture"
        )


def build_precondition(args, dataset, state, gaussians, program, selected,
                       frame_times, frame_dt, torch):
    """Everything the reading rule needs, computed BEFORE any pixel is scored.

    Carries a frozen PRECONDITION, not only a frozen reading rule: it asserts
    that the mechanism the metrics will read was actually exercised -- there
    are gated rows, and at least one scored frame sits where the gate drives
    presence to EXACT zero. Stated about the setup; it cannot see a score.
    """
    from depth_visibility.errors import ContractError

    n_rows = int(gaussians.get_xyz.shape[0])
    gated_mask = gaussians.get_elgs_gated_row_mask()
    gated_rows = int(gated_mask.sum())

    registry = state.runtime.registry
    families = []
    for family_id in registry.active_ids():
        record = registry.get(family_id)
        realization = state.runtime.realization(family_id)
        families.append({
            "family_id": int(family_id),
            "lineage_key": str(record.lineage_key),
            "K": int(record.interval.K),
            "rows": int((gaussians._elgs_family_ids == family_id).sum()),
            "episode_starts_seconds": [float(v) for v in realization.b.tolist()],
            "episode_ends_seconds": [float(v) for v in realization.d.tolist()],
        })

    gaps = program_gaps_seconds(program)
    in_gap = frames_inside_gaps(frame_times, gaps)

    # The STRICT test: frames at which the realized presence is exactly zero on
    # at least one gated row. This is the mechanism (a total gate, elgs/
    # presence.py local_presence_multipliers), not a restatement of the JSON.
    exact_absence, partial = [], []
    with torch.no_grad():
        for frame, timestamp in frame_times:
            presence = gaussians.get_elgs_presence(float(timestamp))
            on_gated = presence[gated_mask]
            if on_gated.numel() == 0:
                continue
            if bool((on_gated == 0).any()):
                exact_absence.append(int(frame))
            elif bool((on_gated < 1.0).any()):
                partial.append(int(frame))

    block = {
        "n_rows": n_rows,
        "gated_rows": gated_rows,
        "gated_row_fraction": (gated_rows / n_rows) if n_rows else 0.0,
        "frame_dt_seconds": float(frame_dt),
        "frame_range": [int(args.frame_range[0]), int(args.frame_range[1])],
        "held_out_frames_in_range": [int(f) for _, f in selected],
        "held_out_frames_in_range_count": len(selected),
        "program": {
            "path": str(args.program),
            "sha256": sha256_file(args.program),
            "schema_version": str(program.get("schema_version")),
            "membership_mode": str(program.get("membership_mode")),
            "cloud": dict(program.get("cloud") or {}),
            "gaps_seconds": {str(k): v for k, v in sorted(gaps.items())},
            "gap_frames_at_frame_dt": {
                str(k): v for k, v in sorted(
                    gap_frames_from_seconds(gaps, frame_dt).items()
                )
            },
        },
        "families": families,
        "frames_in_declared_gap": in_gap,
        "frames_in_declared_gap_count": len(in_gap),
        "frames_with_exact_absence": exact_absence,
        "frames_with_exact_absence_count": len(exact_absence),
        "frames_with_partial_gate": partial,
        "frames_with_partial_gate_count": len(partial),
        "held_out_camera_ids": sorted({
            camera_id_from_name(getattr(c, "image_name", ""))
            for c in dataset["test_stack"]
        } - {None}),
        "train_camera_ids": sorted(dataset["train_camera_ids"]),
        "eval_split_enabled": bool(dataset["eval"]),
    }

    if gated_rows <= 0:
        raise ContractError(
            "PRECONDITION FAILED: the program gates 0 rows on the restored "
            "cloud; every metric below would compare a model with itself"
        )
    if not selected:
        raise ContractError(
            "PRECONDITION FAILED: no held-out frame lies in --frame_range "
            "%s" % (block["frame_range"],)
        )
    if not in_gap:
        raise ContractError(
            "PRECONDITION FAILED: none of the %d held-out frames in "
            "--frame_range falls inside a declared gap; the gate would be "
            "inactive at every scored frame and the comparison would be "
            "vacuous" % len(selected)
        )
    if not exact_absence:
        raise ContractError(
            "PRECONDITION FAILED: %d scored frames lie in a declared gap but "
            "the realized presence is never exactly zero on a gated row. The "
            "declared gap is narrower than the presence edge band, so the "
            "total gate is never exercised" % len(in_gap)
        )
    return block


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    from arguments import ModelParams, OptimizationParams, PipelineParams

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", required=True,
                        help="the GATED config, e.g. configs/n3v/elgs_local_crb300_6k.yaml")
    parser.add_argument("--start_checkpoint", required=True)
    parser.add_argument("--program", required=True,
                        help="adags-episode-program-v2 JSON; overrides the "
                             "config's elgs_oracle_episodes")
    parser.add_argument("--frame_range", nargs=2, type=int, required=True,
                        metavar=("A", "B"),
                        help="absolute held-out frames to render, inclusive")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--restore_state", action="store_true",
                        help="score an EL-GS checkpoint with the gate it "
                             "trained with (setup_elgs restore branch); "
                             "--program must match the restored intervals")
    # Top-level YAML keys, mirroring scripts/eval_lrv1_event.py:135-143. They
    # must exist as attributes or _merge_config's hasattr assert fires.
    parser.add_argument("--gaussian_dim", type=int, default=4)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[0.0, 10.0])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--seed", type=int, default=6666)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    _merge_config(args, args.config)

    import torch

    from depth_visibility.errors import ContractError
    from elgs.trainer_hooks import (
        EPISODE_PROGRAM_SCHEMA_V2 as _SCHEMA_V2,
        infer_frame_dt,
        setup_elgs,
    )
    from gaussian_renderer import render
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from utils.image_utils import psnr
    from utils.render_utils import save_img_u8

    if _SCHEMA_V2 != EPISODE_PROGRAM_SCHEMA_V2:
        raise ContractError(
            "the mirrored v2 schema constant %r no longer matches "
            "elgs.trainer_hooks' %r" % (EPISODE_PROGRAM_SCHEMA_V2, _SCHEMA_V2)
        )

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = Path(args.out_dir)
    (out_dir / "gated" / "renders").mkdir(parents=True, exist_ok=True)
    (out_dir / "ungated" / "renders").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)

    program = json.loads(Path(args.program).read_text(encoding="utf-8"))
    if program.get("schema_version") != EPISODE_PROGRAM_SCHEMA_V2:
        raise ContractError(
            "--program must carry schema %r, got %r"
            % (EPISODE_PROGRAM_SCHEMA_V2, program.get("schema_version"))
        )
    # The seeder reads the program from OptimizationParams, so the override has
    # to happen BEFORE op.extract(args).
    args.elgs_oracle_episodes = str(args.program)

    torch.manual_seed(args.seed)
    dataset_params = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    if not bool(getattr(opt, "elgs_enable", False)):
        raise ContractError(
            "--config must declare elgs_enable: true (use "
            "configs/n3v/elgs_local_crb300_6k.yaml). With it unset setup_elgs "
            "returns None (elgs/trainer_hooks.py:127-128) and this script "
            "would silently score the ungated model twice"
        )
    if not bool(getattr(dataset_params, "eval", False)):
        raise ContractError(
            "ModelParams.eval must be True: with eval False the reader merges "
            "the test split into training and there is no held-out camera to "
            "score"
        )

    if not str(getattr(args, "source_path", "") or "").strip():
        raise ContractError(
            "--source_path is required: configs/n3v/*.yaml supply the scene at "
            "submit time, not in the config"
        )
    if not str(getattr(args, "model_path", "") or "").strip():
        # Scene() writes input.ply and cameras.json into model_path; default it
        # to the output directory rather than the repository root.
        args.model_path = str(out_dir)
        dataset_params.model_path = str(out_dir)
    os.makedirs(dataset_params.model_path, exist_ok=True)

    gaussians = GaussianModel(
        dataset_params.sh_degree, gaussian_dim=args.gaussian_dim,
        time_duration=args.time_duration, rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0,
    )
    scene = Scene(dataset_params, gaussians, num_pts=args.num_pts,
                  num_pts_ratio=args.num_pts_ratio,
                  time_duration=args.time_duration, shuffle=False)
    scene.opt = opt
    # training_setup BEFORE restore: setup_elgs needs a live optimizer (the
    # scalar-budget read at trainer_hooks.py:279-284 and the elgs_a param group
    # at :1044-1049). Same order as scripts/eval_lrv1_event.py:178-180.
    gaussians.training_setup(opt)
    model_params, ckpt_iteration = torch.load(args.start_checkpoint)
    gaussians.restore(model_params, opt)

    # THE REFUSAL. A checkpoint carrying elgs_state sends setup_elgs down the
    # restore branch (trainer_hooks.py:155-169), which sets seeded=True and
    # SKIPS seed_families entirely -- `--program` would be read for nothing and
    # the rendered gate would be the checkpoint's own.
    seeding_mode = resolve_seeding_mode(
        getattr(gaussians, "_pending_elgs_state", None) is not None,
        bool(args.restore_state),
    )

    state = setup_elgs(gaussians, scene, dataset_params, opt)
    if state is None or getattr(gaussians, "elgs_runtime", None) is None:
        raise ContractError("the EL-GS runtime is not live after setup_elgs")
    if not bool(getattr(gaussians, "_elgs_local_presence", False)):
        raise ContractError(
            "_elgs_local_presence is False on the model: the renderer would "
            "apply a GLOBAL presence to every row (gaussian_renderer/"
            "__init__.py:225) instead of the localized gate"
        )
    program_match = None
    if seeding_mode == "restore":
        from scripts.gate_cell_precondition import check_program_matches_runtime

        assert_gate_is_program_only(opt, state, str(state.oracle_episodes))
        program_match = check_program_matches_runtime(state, program, ContractError)
    else:
        assert_gate_is_program_only(opt, state, str(args.program))

    train_stack = scene.train_cameras[1.0]
    test_stack = scene.test_cameras[1.0]
    train_camera_ids = {
        camera_id_from_name(getattr(c, "image_name", "")) for c in train_stack
    } - {None}
    test_camera_ids = {
        camera_id_from_name(getattr(c, "image_name", "")) for c in test_stack
    } - {None}
    overlap = sorted(train_camera_ids & test_camera_ids)
    if overlap:
        raise ContractError(
            "anti-leakage: cameras %s appear in BOTH splits" % (overlap,)
        )
    if not test_stack:
        raise ContractError("the test split is empty; nothing to score")

    frame_dt = infer_frame_dt([float(getattr(c, "timestamp", 0.0))
                               for c in train_stack])
    test_names = [str(getattr(c, "image_name", "")) for c in test_stack]
    selected = select_frame_range(test_names, int(args.frame_range[0]),
                                  int(args.frame_range[1]))
    frame_times = [
        (frame, float(getattr(test_stack[position], "timestamp", 0.0)))
        for position, frame in selected
    ]

    precondition = build_precondition(
        args,
        {"test_stack": test_stack, "train_camera_ids": train_camera_ids,
         "eval": bool(getattr(dataset_params, "eval", False))},
        state, gaussians, program, selected, frame_times, frame_dt, torch,
    )

    provenance = {
        "checkpoint": str(args.start_checkpoint),
        "checkpoint_sha256": sha256_file(args.start_checkpoint),
        "checkpoint_iteration": int(ckpt_iteration),
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "program": str(args.program),
        "program_sha256": precondition["program"]["sha256"],
        "source_path": str(dataset_params.source_path),
        "model_path": str(dataset_params.model_path),
        "out_dir": str(out_dir),
        "git": git_provenance(repo_root),
        "seeding_mode": seeding_mode,
        "program_match": program_match,
        "elgs": {
            "rounds_enabled": state.rounds_enabled,
            "a_lr": state.a_lr,
            "local_presence": state.local_presence,
            "routing_pins_enabled": state.routing_pins_enabled,
            "oracle_episodes": str(state.oracle_episodes),
            "tracks_dir": str(getattr(opt, "elgs_tracks_dir", "") or ""),
            "reserved_units": len(state.reserved_indices),
        },
    }

    report_path = out_dir / "report.json"
    # WRITTEN BEFORE ANY METRIC. If the loop below dies, this file still says
    # what the instrument was and whether its mechanism was engaged.
    report_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "status": "precondition_only",
        "evidence_bearing": False,
        "precondition": precondition,
        "provenance": provenance,
    }, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"precondition": precondition}, sort_keys=True), flush=True)

    background = torch.tensor(
        [1, 1, 1] if dataset_params.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )
    test_ds = scene.getTestCameras()
    gated_mask = gaussians.get_elgs_gated_row_mask()

    per_frame = []
    gate_off_checks = None
    with torch.no_grad():
        for position, frame in selected:
            image, camera = _unpack(test_ds[position])
            if image is None:
                raise ContractError(
                    "held-out view %s carries no ground-truth image" % frame
                )
            gt = image[0:3].cuda().float().clamp(0.0, 1.0)
            timestamp = float(getattr(camera, "timestamp", 0.0))
            cuda_camera = camera.cuda()

            presence = gaussians.get_elgs_presence(timestamp)
            on_gated = presence[gated_mask]
            gated_off_rows = int((on_gated == 0).sum())
            gated_partial_rows = int(((on_gated > 0) & (on_gated < 1.0)).sum())

            gated_render = render(cuda_camera, gaussians, pipe, background)[
                "render"].clamp(0.0, 1.0)

            with _gate_disabled(gaussians):
                if gate_off_checks is None:
                    gate_off_checks = verify_gate_off(gaussians)
                ungated_render = render(cuda_camera, gaussians, pipe, background)[
                    "render"].clamp(0.0, 1.0)

            name = render_filename(frame)
            save_img_u8(gated_render.permute(1, 2, 0).cpu().numpy(),
                        str(out_dir / "gated" / "renders" / name))
            save_img_u8(ungated_render.permute(1, 2, 0).cpu().numpy(),
                        str(out_dir / "ungated" / "renders" / name))
            save_img_u8(gt.permute(1, 2, 0).cpu().numpy(),
                        str(out_dir / "gt" / name))

            psnr_gated = float(psnr(gt.unsqueeze(0), gated_render.unsqueeze(0)))
            psnr_ungated = float(psnr(gt.unsqueeze(0), ungated_render.unsqueeze(0)))
            per_frame.append({
                "frame": int(frame),
                "camera": camera_id_from_name(getattr(camera, "image_name", "")),
                "image_name": str(getattr(camera, "image_name", "")),
                "timestamp": timestamp,
                "psnr_gated": psnr_gated,
                "psnr_ungated": psnr_ungated,
                "psnr_delta": psnr_gated - psnr_ungated,
                "gated_rows_total": int(gated_mask.sum()),
                "gated_rows_presence_zero": gated_off_rows,
                "gated_rows_presence_partial": gated_partial_rows,
                "gated_presence_min": float(on_gated.min()) if on_gated.numel() else None,
                "gated_presence_mean": float(on_gated.mean()) if on_gated.numel() else None,
                "in_declared_gap": int(frame) in set(precondition["frames_in_declared_gap"]),
                "identical_renders": bool(
                    torch.equal(gated_render, ungated_render)
                ),
            })
            del gated_render, ungated_render, gt

    if getattr(gaussians, "elgs_runtime", None) is None:
        raise ContractError(
            "the EL-GS runtime was not restored after the gate-off arm; the "
            "gated numbers above cannot be trusted"
        )

    scored = [row for row in per_frame if row["gated_rows_presence_zero"] > 0]
    summary = {
        "frames_scored": len(per_frame),
        "frames_with_gate_active": len(scored),
        "mean_psnr_gated": (
            sum(r["psnr_gated"] for r in per_frame) / len(per_frame)
        ) if per_frame else None,
        "mean_psnr_ungated": (
            sum(r["psnr_ungated"] for r in per_frame) / len(per_frame)
        ) if per_frame else None,
        "mean_psnr_delta": (
            sum(r["psnr_delta"] for r in per_frame) / len(per_frame)
        ) if per_frame else None,
        "mean_psnr_delta_gate_active_only": (
            sum(r["psnr_delta"] for r in scored) / len(scored)
        ) if scored else None,
        "frames_with_identical_renders": sum(
            1 for r in per_frame if r["identical_renders"]
        ),
    }

    report_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "evidence_bearing": False,
        "precondition": precondition,
        "provenance": provenance,
        "gate_off_verification": gate_off_checks,
        "per_frame": per_frame,
        "summary": summary,
    }, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary": summary,
                      "gate_off_verification": gate_off_checks},
                     indent=1, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
