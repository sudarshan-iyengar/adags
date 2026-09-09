#!/usr/bin/env python
"""Mechanism-exercise PRECONDITION for ONE trained real-data gate cell.

Writes ``<run_dir>/precondition.json``, the file
``scripts/realdata_gate_analysis.py`` reads to decide whether a cell belongs
in the ``mechanism_exercised`` analysis set
(research-wiki/operations/realdata-gating-lane-2026-09-09.md section 6-v2.3
clauses (a)-(d), plus the reserved-unit equality check of defect 16).

This is BOOKKEEPING ABOUT THE SETUP, not a score. Nothing here reads a PSNR,
a render, or any endpoint; the three quantities the analysis gates on are all
statements about which rows the gate holds and when it drives them to exact
zero. That is why reading the HELD-OUT camera's geometry (cam00) is admitted:
clause (c) is defined on the held-out camera because that is the camera the
endpoints are scored on, and only the camera's POSE and INTRINSICS are used --
no ground-truth pixel is opened, no image is decoded, and no render is
performed. The projection is the same pure projection
``scripts/estimate_episodes.py:build_footprints`` uses
(``utils.motion_prior_utils.project_points_to_screen`` on
``gaussians.get_dynamic_xyz(t)``), reused rather than re-derived.

Both arms
---------
``--config`` decides the arm and the script cross-checks it against the log:

* GATED (``elgs_enable: true``, e.g. ``configs/n3v/elgs_local_crb300_12k.yaml``
  and its ``_mis`` twin). The checkpoint carries ``elgs_state``, so
  ``setup_elgs`` takes the RESTORE branch (elgs/trainer_hooks.py:155-169) and
  rebinds ``_elgs_family_ids`` from the checkpoint's ``row_family_ids``. That
  is deliberately the OPPOSITE of ``scripts/eval_n3v_gated.py``, which refuses
  a checkpoint carrying ``elgs_state``: that evaluator seeds a NEW program
  onto an ungated checkpoint, whereas this script must report the program the
  cell ACTUALLY TRAINED WITH. ``--program`` is therefore never allowed to
  define the gate here; it is cross-checked against the restored intervals
  (``program_family_match`` below) and a mismatch is a refusal.
* UNGATED (``elgs_enable`` unset, ``configs/n3v/b0c_crb300_12k_rp.yaml``).
  ``setup_elgs`` returns None (trainer_hooks.py:127-128); every gated counter
  is 0 by construction and ``arm_kind`` is ``"ungated"``.

Where each number comes from -- READ THIS BEFORE TRUSTING reserved_units
------------------------------------------------------------------------
``gated_rows_seeding`` / ``n_rows_seeding``
    The ``{"elgs_seeding": {...}}`` line ``seed_families`` prints
    (elgs/trainer_hooks.py:1003-1031): ``gated_rows`` = ``(family_ids >= 0)``
    at the seeding iteration, ``rows`` = the cloud's row count there. Parsed
    from ``<run_dir>/meta/train.log`` (``scripts/run_leonardo.sh:231`` tees
    stdout+stderr there). For an ungated cell ``gated_rows_seeding`` is 0 and
    ``n_rows_seeding`` is **null**: a non-EL-GS run has no seeding moment at
    all, so the quantity is UNDEFINED rather than merely unmeasured, and
    substituting the initial cloud size would silently compare two different
    things across arms. The reconstructed initial cloud size is reported
    separately as ``n_rows_scene_init``.

``reserved_units`` / ``training_units_total``
    THE LANE SPEC SAYS "printed by both code paths". THAT IS NOT TRUE OF THE
    CODE AS IT STANDS, and it is the one assumption this script had to work
    around. Only the CONTROL path prints a count: ``main.py:1237-1240`` emits
    ``{"elgs_reserved_parity": {"reserved_units": N, "training_units_after":
    M}}``, and ``reserved_indices_for_parity`` returns None whenever
    ``elgs_enable`` is set (trainer_hooks.py:1508-1511), so that line NEVER
    appears in a gated cell's log. The EL-GS path prints
    ``{"elgs_setup": {...}}`` (trainer_hooks.py:344-351), which carries
    schedule/frame_dt/time_span/restored/families/evidence and NO reserved
    count, and ``filter_elgs_reserved`` prints nothing at all.

    So for a gated cell the count is RECOMPUTED, not parsed, from
    ``build_reserved_pool`` -- the single implementation of the reservation
    rule that ``setup_elgs`` and the parity path both call
    (trainer_hooks.py:1440-1490), which depends on nothing but the training
    camera list and reserves the ``(frame_order + camera_order) % 4 == 0``
    diagonal from SORTED groups, so it is independent of dataset ordering and
    of the training seed. ``reserved_units_source`` records which route was
    used per cell. Both routes are always computed when both are available and
    a disagreement is a refusal, so the ungated cell's printed number audits
    the recomputation that the gated cell has to rely on. That is what makes
    the cross-arm equality check of defect 16 meaningful.

``gated_rows_final`` / ``n_rows_final``
    From the checkpoint, restored the way ``scripts/eval_n3v_gated.py``
    restores it (Scene -> ``training_setup(opt)`` -> ``torch.load`` ->
    ``restore`` -> ``setup_elgs``; the optimizer must exist before setup_elgs
    reads its scalar budget). ``gated_rows_final`` counts
    ``_elgs_family_ids >= 0``, exactly as clause (b) states, and is asserted
    equal to the renderer's own ``get_elgs_gated_row_mask()`` count -- a
    difference would mean a K=1 family had been created and the two readings
    of "gated" had drifted apart.

``gated_rows_fbox_frame150``
    Pure projection of the gated rows at frame ``--fbox_frame`` (default 150,
    a pre-occlusion frame per the lane page's headroom table) into the
    held-out camera, counted inside the F box ``--fbox`` (default
    ``664 912 744 976``, x0 y0 x1 y1, INCLUSIVE; 80x64 px). The model's
    positions are taken at the held-out camera's OWN timestamp -- not at a
    hand-computed 150/30 s -- because that is the number the renderer would
    use; the two agree to 1e-12 s on N3V (the reader sets
    ``timestamp = frame_idx / 30.0``, scene/dataset_readers.py:200) and a
    disagreement above half a frame is a refusal. Pixels are rounded exactly
    as ``build_footprints`` rounds them, and rows failing the projector's own
    ``valid`` test (behind the camera or outside NDC) are excluded, so the
    clamp ``build_footprints`` applies afterwards is a no-op here.

``frames_presence_zero``
    Clause (d). Reported from the LIVE RUNTIME wherever there is one: the
    number of integer frames in ``[0, --n_frames)`` at which
    ``get_elgs_presence(t)`` is EXACTLY 0 on at least one gated row. That is
    the mechanism itself (``elgs/presence.py`` -- a family with no active
    episode at t returns an exact zero, and so does an episode endpoint,
    since ``smoothstep(0) == 0``). The frozen program is read independently
    through ``scripts.eval_n3v_gated.program_gaps_seconds`` /
    ``frames_inside_gaps`` and reported as ``program_frames_in_gap``: because
    ``_interval_from_gaps`` (trainer_hooks.py:638-660) places the episode
    edges EXACTLY on the declared gap endpoints, the closed-gap test and the
    exact-zero test are the same test, and the two counts must agree. They are
    compared, not merged, because the realized intervals are float32 while the
    program's seconds are float64, so an endpoint frame can in principle flip;
    ``frames_presence_zero_agrees`` and ``frames_presence_zero_disagreements``
    record it rather than hiding it.

Fail-closed
-----------
Every missing log line, absent checkpoint, empty held-out frame, arm/config
disagreement, program/interval mismatch and reserved-unit disagreement raises
before anything is written. A precondition that quietly reported a zero would
be worse than none: the consumer would read it as "mechanism not exercised"
and drop a cell that was fine, or -- with the fields absent -- fall back to a
default. Nothing is written unless every clause could be computed.

Cost, and where it goes
-----------------------
GPU node, no rendering, a few minutes. This script itself never decodes a
training or held-out image -- it reads ``scene.train_cameras[1.0]`` and
``scene.test_cameras[1.0]`` directly rather than the ``CameraDataset``
wrappers, whose ``__getitem__`` decodes. ``setup_elgs`` DOES iterate
``scene.getTrainCameras()`` to build the reserved pool
(trainer_hooks.py:132-134), so a gated cell pays one decode pass over the
training views exactly as its own training run did; that cost is in the
existing code path, not added here.

Scratch writes
--------------
``Scene()`` copies ``input.ply`` and writes ``cameras.json`` into
``model_path`` (scene/__init__.py:62-74), so ``--model_path`` defaults to
``<run_dir>/precondition_scratch`` rather than to the run directory: this
script must not overwrite a trained cell's own artifacts. ``precondition.json``
is the ONLY file it writes into ``<run_dir>`` itself.

Usage (see the module footer of the delivery note for the frozen CLI)::

    python scripts/gate_cell_precondition.py \\
        --run_dir <runs>/realdata_gate/<run_id> \\
        --config configs/n3v/elgs_local_crb300_12k.yaml \\
        --source_path <root>/cut_roasted_beef \\
        --program <path>/membership_program_spatial.json
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path

# INSERT, not append: the admitted image ships a `pointops2` whose `functions`
# subpackage is absent and `utils.general_utils` imports it unconditionally.
# Same reason as scripts/eval_n3v_gated.py:86.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The sibling evaluator keeps every heavy import inside main(), so these are
# stdlib-only helpers and importing them here costs no torch.
from scripts.eval_n3v_gated import (  # noqa: E402
    absolute_frame_from_name,
    camera_id_from_name,
    frames_inside_gaps,
    git_provenance,
    program_gaps_seconds,
    sha256_file,
)

SCHEMA_VERSION = "gate-cell-precondition-v1"

# The exact field set scripts/realdata_gate_analysis.py reads back
# (SPEC["PRECONDITION_FIELDS"]). Mirrored so this module stays importable
# without numpy; checked against the real SPEC at runtime and in the tests.
PRECONDITION_FIELDS = (
    "gated_rows_seeding",
    "n_rows_seeding",
    "gated_rows_final",
    "n_rows_final",
    "gated_rows_fbox_frame150",
    "frames_presence_zero",
    "reserved_units",
    "training_units_total",
)

# research-wiki/operations/crb300-event-mask-curation-2026-08-23 via the lane
# page section 2: the F box on held-out cam00, x0 y0 x1 y1 inclusive.
DEFAULT_FBOX = (664, 912, 744, 976)
DEFAULT_FBOX_FRAME = 150
DEFAULT_N_FRAMES = 300
DEFAULT_CKPT_ITER = 12000
# Half a frame at 30 fps: the tolerance on "the held-out camera's own
# timestamp is the frame we asked for".
FRAME_TIME_TOL_SECONDS = 1.0 / 60.0
# Seconds. The realized intervals are float32 (elgs/trainer_hooks.py:656-659)
# while the program's gaps are float64, so an exact comparison is not
# available; 1e-4 s is 1/333 of a frame and cannot hide a shifted window (the
# G-mis program moves its gap by 15 frames = 0.5 s).
INTERVAL_MATCH_TOL_SECONDS = 1e-4


# ---------------------------------------------------------------------------
# PURE HELPERS -- stdlib only, so the log reading rule, the box test and the
# frame arithmetic are testable without torch, CUDA or a checkpoint.
# ---------------------------------------------------------------------------


def _match_brace_span(text, start):
    """End index (exclusive) of the JSON object opening at `text[start]`.

    String- and escape-aware, so a brace inside a value cannot end the scan.
    Returns None when the object is unterminated (a truncated log line).
    """
    if start >= len(text) or text[start] != "{":
        raise ValueError("_match_brace_span must start on '{'")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def extract_json_log_objects(text, key):
    """Every ``{"<key>": {...}}`` payload printed into a run log.

    The trainer prints these with ``print(json.dumps({key: {...}},
    sort_keys=True))`` and ``scripts/run_leonardo.sh`` tees stdout AND stderr
    into one file, so a tqdm progress bar can share the physical line. Scanning
    for the literal ``{"<key>":`` opener and brace-matching forward is therefore
    the reading rule, NOT a line-oriented ``json.loads`` -- which would drop
    every line a progress bar touched.

    Returns the list of inner payloads in file order. Raises on a truncated
    object rather than skipping it: a half-written seeding line means the log
    is not the log of a finished run.
    """
    opener = '{"%s":' % key
    found = []
    cursor = 0
    while True:
        index = text.find(opener, cursor)
        if index < 0:
            return found
        end = _match_brace_span(text, index)
        if end is None:
            raise ValueError(
                "log carries a truncated %r object at offset %d; the run did "
                "not finish writing it" % (key, index)
            )
        blob = text[index:end]
        try:
            payload = json.loads(blob)
        except ValueError as exc:
            raise ValueError(
                "log carries an unparseable %r object at offset %d: %s"
                % (key, index, exc)
            )
        found.append(payload[key])
        cursor = end


def read_unique_log_object(text, key, source="the run log"):
    """The single ``{"<key>": {...}}`` payload in `text`, or None.

    ``run_leonardo.sh`` APPENDS to ``meta/<mode>.log`` (``tee -a``), so a
    resubmitted cell that reused its run directory can leave two runs in one
    file. Identical repeats are accepted (the same run re-teed); ANY
    disagreement is a refusal, because silently taking the last one would let
    a cancelled first attempt's numbers stand for the cell that was scored.
    """
    found = extract_json_log_objects(text, key)
    if not found:
        return None
    first = found[0]
    for other in found[1:]:
        if other != first:
            raise ValueError(
                "%s carries %d DIFFERING %r objects; this run directory holds "
                "more than one run and the cell's numbers are ambiguous. "
                "First: %s Later: %s"
                % (source, len(found), key, json.dumps(first, sort_keys=True),
                   json.dumps(other, sort_keys=True))
            )
    return first


def require_int(payload, field, key, source="the run log"):
    """`payload[field]` as an int, or a refusal naming the log line."""
    if field not in payload:
        raise ValueError(
            "%s's %r object has no %r field; this log was written by a "
            "different revision of the trainer than this script reads"
            % (source, key, field)
        )
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "%s's %r.%s is %r, which is not a count" % (source, key, field, value)
        )
    return int(value)


def point_in_box(x, y, box):
    """The F-box membership test: x0 <= x <= x1 and y0 <= y <= y1, INCLUSIVE.

    `box` is (x0, y0, x1, y1) in pixels of the held-out camera.
    """
    x0, y0, x1, y1 = box
    if x1 < x0 or y1 < y0:
        raise ValueError("--fbox needs x0 <= x1 and y0 <= y1, got %r" % (tuple(box),))
    return (x0 <= x <= x1) and (y0 <= y <= y1)


def frame_time_grid(frame_dt, n_frames, timestamps_by_frame=None):
    """``[(frame, seconds), ...]`` for frames ``0 .. n_frames-1``.

    Uses the dataset's OWN timestamp for a frame whenever one is supplied
    (the value the renderer sees), and ``frame * frame_dt`` otherwise, so the
    grid never silently substitutes an idealized clock for the real one.
    """
    if frame_dt <= 0:
        raise ValueError("frame_dt must be positive")
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    supplied = dict(timestamps_by_frame or {})
    return [
        (frame, float(supplied.get(frame, frame * frame_dt)))
        for frame in range(int(n_frames))
    ]


def frames_with_presence_zero(gaps_by_group, frame_dt, n_frames,
                              timestamps_by_frame=None):
    """Frames at which the program drives a gated row's presence to EXACT 0.

    ``_interval_from_gaps`` (elgs/trainer_hooks.py:638-660) builds every
    family's episodes as ``[-w_m, gap_0_lo], [gap_0_hi, gap_1_lo], ...,
    [gap_last_hi, T + w_m]`` -- the episode edges ARE the declared gap
    endpoints. ``elgs/presence.py`` returns an exact zero when no episode
    contains t, and ``smoothstep(0) == 0`` makes the endpoints exact zeros
    too, so "t lies in a closed declared gap" and "presence is exactly 0" are
    the same predicate. The closed-interval reading is therefore the mechanism
    here, not a permissive stand-in for it.

    Reuses ``scripts.eval_n3v_gated.frames_inside_gaps`` so the precondition
    and the sibling evaluator quote one implementation.
    """
    grid = frame_time_grid(frame_dt, n_frames, timestamps_by_frame)
    return frames_inside_gaps(grid, gaps_by_group)


def source_path_from_cfg_args(text):
    """The ``source_path`` recorded in a run's ``cfg_args``, or None.

    ``main.py:1831`` writes ``Namespace(...)`` there. Parsed with ``ast`` --
    never ``eval`` -- and a shape this cannot read returns None rather than
    raising, because ``cfg_args`` is a cross-check, not the input.
    """
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        return None
    call = tree.body
    if not isinstance(call, ast.Call):
        return None
    for keyword in call.keywords:
        if keyword.arg == "source_path" and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            return value if isinstance(value, str) else None
    return None


def assemble_precondition(
    *,
    arm_kind,
    gated_rows_seeding,
    n_rows_seeding,
    gated_rows_final,
    n_rows_final,
    gated_rows_fbox_frame150,
    frames_presence_zero,
    reserved_units,
    training_units_total,
    detail,
    provenance,
):
    """The written object. Pure, so the field set is testable without torch.

    The eight fields ``realdata_gate_analysis.SPEC["PRECONDITION_FIELDS"]``
    reads sit at the TOP LEVEL because ``read_precondition`` reads them there
    (scripts/realdata_gate_analysis.py:393-402); everything else is context.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "arm_kind": arm_kind,
        "gated_rows_seeding": gated_rows_seeding,
        "n_rows_seeding": n_rows_seeding,
        "gated_rows_final": gated_rows_final,
        "n_rows_final": n_rows_final,
        "gated_rows_fbox_frame150": gated_rows_fbox_frame150,
        "frames_presence_zero": frames_presence_zero,
        "reserved_units": reserved_units,
        "training_units_total": training_units_total,
        "detail": detail,
        "provenance": provenance,
    }
    missing = [field for field in PRECONDITION_FIELDS if field not in payload]
    if missing:
        raise ValueError("precondition payload is missing %s" % (missing,))
    return payload


def _merge_config(args, config_path):
    """Identical to scripts/eval_n3v_gated.py:233-250 -- an unknown key is a
    hard failure, so a config written for another entrypoint cannot be read
    half-way."""
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


def _realized_gaps(realization):
    """`[[d_k, b_{k+1}], ...]` -- the absence gaps a realized interval encodes."""
    b = [float(v) for v in realization.b.tolist()]
    d = [float(v) for v in realization.d.tolist()]
    return [[d[k], b[k + 1]] for k in range(len(b) - 1)]


def check_program_matches_runtime(state, program, ContractError):
    """The restored intervals ARE the supplied program, or a refusal.

    ``--program`` never defines the gate on this path (the checkpoint's
    ``elgs_state`` does), so it has to be PROVED to be the same program or the
    provenance block would attribute a cell to a file it never trained with.
    Matching is by lineage key: the v2 seeder names every family
    ``est-group-<group>`` (elgs/trainer_hooks.py:962), so the correspondence
    is exact and does not depend on family-id ordering. Groups the seeder
    skipped because they gated zero rows carry no family and are reported,
    not silently dropped.
    """
    declared = program_gaps_seconds(program)
    registry = state.runtime.registry
    by_key = {}
    for family_id in registry.active_ids():
        record = registry.get(family_id)
        by_key[str(record.lineage_key)] = family_id
    matched, missing, mismatched = {}, [], []
    for group, gaps in sorted(declared.items()):
        key = "est-group-%d" % group
        family_id = by_key.get(key)
        if family_id is None:
            missing.append(group)
            continue
        realized = _realized_gaps(state.runtime.realization(family_id))
        if len(realized) != len(gaps) or any(
            abs(realized[i][0] - gaps[i][0]) > INTERVAL_MATCH_TOL_SECONDS
            or abs(realized[i][1] - gaps[i][1]) > INTERVAL_MATCH_TOL_SECONDS
            for i in range(len(gaps))
        ):
            mismatched.append({
                "group": group, "declared": gaps, "realized": realized,
            })
            continue
        matched[str(group)] = {"family_id": int(family_id), "gaps": realized}
    if mismatched:
        raise ContractError(
            "--program is NOT the program this cell trained with: the "
            "restored episode boundaries differ from the declared gaps by "
            "more than %g s. %s"
            % (INTERVAL_MATCH_TOL_SECONDS, json.dumps(mismatched, sort_keys=True))
        )
    if not matched:
        raise ContractError(
            "no family in the restored checkpoint carries a lineage key "
            "matching any group of --program (groups %s); the supplied "
            "program does not describe this cell"
            % (sorted(declared),)
        )
    return {
        "matched_groups": matched,
        "groups_without_a_family": missing,
        "tolerance_seconds": INTERVAL_MATCH_TOL_SECONDS,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    from arguments import ModelParams, OptimizationParams, PipelineParams

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--run_dir", required=True,
                        help="the trained cell's run directory (holds "
                             "chkpnt<iter>.pth, meta/train.log, cfg_args)")
    parser.add_argument("--config", required=True,
                        help="the config the cell TRAINED with; it decides the arm")
    parser.add_argument("--program", required=True,
                        help="the frozen adags-episode-program-v2 JSON; "
                             "cross-checked against the restored intervals, "
                             "never used to define the gate")
    parser.add_argument("--ckpt_iter", type=int, default=DEFAULT_CKPT_ITER)
    parser.add_argument("--checkpoint", default="",
                        help="override <run_dir>/chkpnt<ckpt_iter>.pth")
    parser.add_argument("--fbox", nargs=4, type=int, default=list(DEFAULT_FBOX),
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="the F box on the held-out camera, INCLUSIVE")
    parser.add_argument("--fbox_frame", type=int, default=DEFAULT_FBOX_FRAME,
                        help="pre-occlusion held-out frame for clause (c)")
    parser.add_argument("--fbox_camera", type=int, default=0,
                        help="expected held-out camera id (cam00)")
    parser.add_argument("--n_frames", type=int, default=DEFAULT_N_FRAMES,
                        help="frames 0..n_frames-1 are tested for clause (d)")
    # Top-level YAML keys, mirroring scripts/eval_n3v_gated.py:487-495. They
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
        EPISODE_PROGRAM_SCHEMA_V2,
        build_reserved_pool,
        infer_frame_dt,
        setup_elgs,
    )
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from utils.motion_prior_utils import project_points_to_screen

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise ContractError("--run_dir %s is not a directory" % run_dir)

    fbox = tuple(int(v) for v in args.fbox)
    point_in_box(fbox[0], fbox[1], fbox)          # validates the box shape

    # ---- the frozen inputs -------------------------------------------------
    checkpoint = (
        Path(args.checkpoint) if str(args.checkpoint).strip()
        else run_dir / ("chkpnt%d.pth" % int(args.ckpt_iter))
    )
    if not checkpoint.is_file():
        raise ContractError(
            "no checkpoint at %s; this cell has not reached iteration %d"
            % (checkpoint, int(args.ckpt_iter))
        )
    log_path = run_dir / "meta" / "train.log"
    if not log_path.is_file():
        raise ContractError(
            "no %s; the seeding and reserved-parity counters are only "
            "available from the training log (scripts/run_leonardo.sh:231 "
            "tees it there) and this precondition cannot be reconstructed "
            "without it" % log_path
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")

    program = json.loads(Path(args.program).read_text(encoding="utf-8"))
    if program.get("schema_version") != EPISODE_PROGRAM_SCHEMA_V2:
        raise ContractError(
            "--program must carry schema %r, got %r"
            % (EPISODE_PROGRAM_SCHEMA_V2, program.get("schema_version"))
        )

    # ---- arm, decided by the config and cross-checked against the log ------
    torch.manual_seed(args.seed)
    dataset_params = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    gated_arm = bool(getattr(opt, "elgs_enable", False))
    arm_kind = "gated" if gated_arm else "ungated"
    seeding = read_unique_log_object(log_text, "elgs_seeding", str(log_path))
    parity = read_unique_log_object(log_text, "elgs_reserved_parity", str(log_path))
    if gated_arm and seeding is None:
        raise ContractError(
            "%s declares elgs_enable but %s carries no {\"elgs_seeding\": ...} "
            "line. Either this log is not this cell's, or the cell resumed "
            "from a checkpoint that had already seeded (seed_families runs "
            "once, at the schedule's seed_iteration). Clause (a) cannot be "
            "reported for it." % (args.config, log_path)
        )
    if not gated_arm and seeding is not None:
        raise ContractError(
            "%s does not declare elgs_enable but %s carries an elgs_seeding "
            "line; the config and the run disagree about the arm"
            % (args.config, log_path)
        )
    if not str(getattr(args, "source_path", "") or "").strip():
        raise ContractError(
            "--source_path is required: configs/n3v/*.yaml supply the scene "
            "at submit time, not in the config"
        )

    cfg_args_path = run_dir / "cfg_args"
    cfg_args_text = (
        cfg_args_path.read_text(encoding="utf-8") if cfg_args_path.is_file() else ""
    )
    recorded_source = source_path_from_cfg_args(cfg_args_text) if cfg_args_text else None
    if recorded_source and os.path.abspath(recorded_source) != os.path.abspath(
        str(args.source_path)
    ):
        raise ContractError(
            "the cell trained on source_path %r (from %s) but --source_path is "
            "%r; the held-out camera geometry of clause (c) would come from a "
            "different scene" % (recorded_source, cfg_args_path, str(args.source_path))
        )

    # Scene writes input.ply and cameras.json into model_path; keep them out of
    # the trained cell's directory.
    if not str(getattr(args, "model_path", "") or "").strip() or os.path.abspath(
        str(args.model_path)
    ) == os.path.abspath(str(run_dir)):
        scratch = run_dir / "precondition_scratch"
        args.model_path = str(scratch)
        dataset_params.model_path = str(scratch)
    os.makedirs(dataset_params.model_path, exist_ok=True)

    # ---- the model, restored exactly as scripts/eval_n3v_gated.py does ----
    gaussians = GaussianModel(
        dataset_params.sh_degree, gaussian_dim=args.gaussian_dim,
        time_duration=args.time_duration, rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0,
    )
    scene = Scene(dataset_params, gaussians, num_pts=args.num_pts,
                  num_pts_ratio=args.num_pts_ratio,
                  time_duration=args.time_duration, shuffle=False)
    scene.opt = opt
    n_rows_scene_init = int(gaussians.get_xyz.shape[0])
    # training_setup BEFORE restore: setup_elgs needs a live optimizer (the
    # scalar-budget read at trainer_hooks.py:279-284).
    gaussians.training_setup(opt)
    model_params, ckpt_iteration = torch.load(str(checkpoint))
    gaussians.restore(model_params, opt)
    n_rows_final = int(gaussians.get_xyz.shape[0])

    if gated_arm and getattr(gaussians, "_pending_elgs_state", None) is None:
        raise ContractError(
            "%s carries NO elgs_state but the config declares elgs_enable: "
            "setup_elgs would take the FRESH branch and re-seed from "
            "--program, so the reported gate would be one this cell never "
            "trained with. Refusing." % checkpoint
        )
    if not gated_arm and getattr(gaussians, "_pending_elgs_state", None) is not None:
        raise ContractError(
            "%s carries an elgs_state payload but the config does not declare "
            "elgs_enable; the checkpoint and the config disagree about the arm"
            % checkpoint
        )

    state = setup_elgs(gaussians, scene, dataset_params, opt)
    if gated_arm and (state is None or getattr(gaussians, "elgs_runtime", None) is None):
        raise ContractError("the EL-GS runtime is not live after setup_elgs")
    if not gated_arm and state is not None:
        raise ContractError(
            "setup_elgs returned a state for a cell whose config does not "
            "declare elgs_enable"
        )

    # ---- the training units, and the reserved diagonal --------------------
    # `scene.train_cameras[1.0]` and NOT `scene.getTrainCameras()`: the latter
    # wraps the same list (in the same order) in a CameraDataset whose
    # __getitem__ DECODES the training image (utils/data_utils.py:19-35), and
    # this pass needs only `timestamp` and `image_name`. Same reason
    # scripts/estimate_episodes.py:859-862 indexes `viewpoint_stack` directly.
    train_stack = scene.train_cameras[1.0]
    training_units_total = int(len(train_stack))
    if training_units_total <= 0:
        raise ContractError("the training split is empty")
    recomputed_reserved = len(build_reserved_pool(train_stack))
    reserved_sources = {}
    if parity is not None:
        logged_reserved = require_int(parity, "reserved_units",
                                      "elgs_reserved_parity", str(log_path))
        logged_after = require_int(parity, "training_units_after",
                                   "elgs_reserved_parity", str(log_path))
        if logged_reserved + logged_after != training_units_total:
            raise ContractError(
                "the elgs_reserved_parity line says %d reserved + %d trained = "
                "%d units, but this scene has %d training units; the log and "
                "the dataset disagree"
                % (logged_reserved, logged_after,
                   logged_reserved + logged_after, training_units_total)
            )
        reserved_sources["log_elgs_reserved_parity"] = logged_reserved
    if state is not None:
        reserved_sources["setup_elgs_state"] = int(len(state.reserved_indices))
    reserved_sources["build_reserved_pool_recomputed"] = int(recomputed_reserved)
    distinct = sorted(set(reserved_sources.values()))
    if len(distinct) != 1:
        raise ContractError(
            "the reserved-unit count disagrees across its sources: %s. The "
            "reservation rule is a pure function of the training camera list "
            "(elgs/trainer_hooks.py build_reserved_pool), so a disagreement "
            "means the log, the checkpoint and the dataset are not the same "
            "run." % json.dumps(reserved_sources, sort_keys=True)
        )
    reserved_units = distinct[0]
    if parity is not None:
        reserved_units_source = "log:elgs_reserved_parity (audited against build_reserved_pool)"
    elif gated_arm:
        reserved_units_source = (
            "recomputed: build_reserved_pool via setup_elgs state.reserved_indices "
            "-- the EL-GS code path prints NO reserved count "
            "(elgs_setup carries none; reserved_indices_for_parity returns None "
            "when elgs_enable is set, so main.py:1237 never fires)"
        )
    else:
        raise ContractError(
            "an ungated cell must print {\"elgs_reserved_parity\": ...} "
            "(main.py:1237-1240); %s carries none, so this cell trained on ALL "
            "%d units and is NOT unit-matched to the gated arm. It cannot be "
            "the comparator." % (log_path, training_units_total)
        )

    # ---- the frame clock, and the held-out camera --------------------------
    frame_dt = infer_frame_dt(
        [float(getattr(camera, "timestamp", 0.0)) for camera in train_stack]
    )
    test_stack = scene.test_cameras[1.0]
    if not test_stack:
        raise ContractError("the test split is empty; clause (c) is unavailable")
    test_camera_ids = sorted({
        camera_id_from_name(getattr(c, "image_name", "")) for c in test_stack
    } - {None})
    train_camera_ids = sorted({
        camera_id_from_name(getattr(camera, "image_name", ""))
        for camera in train_stack
    } - {None})
    overlap = sorted(set(train_camera_ids) & set(test_camera_ids))
    if overlap:
        raise ContractError("cameras %s appear in BOTH splits" % (overlap,))

    timestamps_by_frame = {}
    for camera in test_stack:
        frame = absolute_frame_from_name(getattr(camera, "image_name", ""))
        if frame is not None:
            timestamps_by_frame.setdefault(frame, float(getattr(camera, "timestamp", 0.0)))

    fbox_cameras = [
        camera for camera in test_stack
        if absolute_frame_from_name(getattr(camera, "image_name", ""))
        == int(args.fbox_frame)
    ]
    if len(fbox_cameras) != 1:
        raise ContractError(
            "clause (c) needs exactly ONE held-out view at frame %d; the test "
            "split has %d (held-out cameras %s)"
            % (int(args.fbox_frame), len(fbox_cameras), test_camera_ids)
        )
    fbox_camera = fbox_cameras[0]
    fbox_camera_id = camera_id_from_name(getattr(fbox_camera, "image_name", ""))
    if fbox_camera_id != int(args.fbox_camera):
        raise ContractError(
            "clause (c) expects the held-out camera cam%02d but frame %d is "
            "held out on cam%s" % (int(args.fbox_camera), int(args.fbox_frame),
                                   fbox_camera_id)
        )
    fbox_timestamp = float(getattr(fbox_camera, "timestamp", 0.0))
    nominal = int(args.fbox_frame) * frame_dt
    if abs(fbox_timestamp - nominal) > FRAME_TIME_TOL_SECONDS:
        raise ContractError(
            "the held-out view at frame %d carries timestamp %.9f s but "
            "frame * inferred dt is %.9f s (dt = %.9f); the frame index and "
            "the model clock disagree by more than half a frame"
            % (int(args.fbox_frame), fbox_timestamp, nominal, frame_dt)
        )

    # ---- the gated rows ----------------------------------------------------
    gated_rows_seeding = 0
    n_rows_seeding = None
    seeding_block = None
    if seeding is not None:
        logged_gated = seeding.get("gated_rows")
        if logged_gated is None:
            raise ContractError(
                "%s's elgs_seeding line reports gated_rows = null, which "
                "seed_families writes only when local_presence is FALSE "
                "(elgs/trainer_hooks.py:1028-1029): this cell did not run a "
                "localized gate" % log_path
            )
        gated_rows_seeding = require_int(seeding, "gated_rows", "elgs_seeding",
                                         str(log_path))
        n_rows_seeding = require_int(seeding, "rows", "elgs_seeding", str(log_path))
        seeding_block = {
            "iteration": seeding.get("iteration"),
            "families": seeding.get("families"),
            "unassigned_rows": seeding.get("unassigned_rows"),
            "v2_membership_mode": seeding.get("v2_membership_mode"),
            "v2_group_rows": seeding.get("v2_group_rows"),
            "v2_group_K": seeding.get("v2_group_K"),
            "local_presence": seeding.get("local_presence"),
            "routing_pins_enabled": seeding.get("routing_pins_enabled"),
            "program_schema": seeding.get("program_schema"),
            "oracle_episodes": seeding.get("oracle_episodes"),
        }

    program_match = None
    gated_rows_final = 0
    gated_rows_fbox = 0
    rows_fbox_all = 0
    gated_rows_onscreen = 0
    frames_presence_zero_list = []
    presence_agrees = None
    disagreements = []

    program_gaps = program_gaps_seconds(program)
    program_frames_in_gap = frames_with_presence_zero(
        program_gaps, frame_dt, int(args.n_frames), timestamps_by_frame
    )

    with torch.no_grad():
        # Clause (c) denominator: EVERY row that projects into the box, gated
        # or not. Reported so the numerator is readable.
        points_all = gaussians.get_dynamic_xyz(fbox_timestamp).detach()
        xy, valid = project_points_to_screen(points_all, fbox_camera)
        valid = valid.reshape(-1)
        # Rounded exactly as scripts/estimate_episodes.py:866-867 rounds; the
        # clamp there is a no-op for rows that already pass `valid`.
        xs = xy[:, 0].round().long()
        ys = xy[:, 1].round().long()
        in_box = (
            (xs >= fbox[0]) & (xs <= fbox[2])
            & (ys >= fbox[1]) & (ys <= fbox[3])
        )
        rows_fbox_all = int((valid & in_box).sum())

        if state is not None:
            family_ids = gaussians._elgs_family_ids
            if int(family_ids.numel()) != n_rows_final:
                raise ContractError(
                    "the restored family-id column has %d entries but the "
                    "cloud has %d rows; the checkpoint's row_family_ids and "
                    "its xyz are not the same run"
                    % (int(family_ids.numel()), n_rows_final)
                )
            gated_rows_final = int((family_ids >= 0).sum())
            runtime_mask = gaussians.get_elgs_gated_row_mask().reshape(-1)
            runtime_gated = int(runtime_mask.sum())
            if runtime_gated != gated_rows_final:
                raise ContractError(
                    "clause (b) reads %d gated rows from _elgs_family_ids >= 0 "
                    "but the renderer's own get_elgs_gated_row_mask() reads "
                    "%d. A family with K = 1 exists, so 'gated' means two "
                    "different things in the same cell."
                    % (gated_rows_final, runtime_gated)
                )
            if gated_rows_final <= 0:
                raise ContractError(
                    "the gated arm's checkpoint carries ZERO gated rows at "
                    "iteration %s: every row the program bound was pruned, and "
                    "the cell rendered the ungated substrate throughout"
                    % ckpt_iteration
                )
            program_match = check_program_matches_runtime(
                state, program, ContractError
            )

            gated_mask = runtime_mask.to(valid.device)
            gated_rows_onscreen = int((gated_mask & valid).sum())
            gated_rows_fbox = int((gated_mask & valid & in_box).sum())

            # Clause (d), from the mechanism itself.
            for frame, timestamp in frame_time_grid(
                frame_dt, int(args.n_frames), timestamps_by_frame
            ):
                presence = gaussians.get_elgs_presence(timestamp).reshape(-1)
                on_gated = presence[gated_mask.to(presence.device)]
                if on_gated.numel() and bool((on_gated == 0).any()):
                    frames_presence_zero_list.append(int(frame))
            presence_agrees = frames_presence_zero_list == program_frames_in_gap
            if not presence_agrees:
                a = set(frames_presence_zero_list)
                b = set(program_frames_in_gap)
                disagreements = sorted(a ^ b)

    frames_presence_zero = len(frames_presence_zero_list)

    detail = {
        "run_dir": str(run_dir),
        "arm_kind": arm_kind,
        "gated_rows_seeding_fraction": (
            (gated_rows_seeding / n_rows_seeding) if n_rows_seeding else None
        ),
        "gated_rows_final_fraction": (
            (gated_rows_final / n_rows_final) if n_rows_final else None
        ),
        "n_rows_scene_init": n_rows_scene_init,
        "n_rows_seeding_note": (
            None if seeding is not None else
            "an ungated cell has no seeding moment, so n_rows_seeding is "
            "undefined rather than unmeasured; n_rows_scene_init records the "
            "reconstructed initial cloud size instead"
        ),
        "seeding_log": seeding_block,
        "reserved": {
            "reserved_units": reserved_units,
            "training_units_total": training_units_total,
            "training_units_after_reservation": training_units_total - reserved_units,
            "source": reserved_units_source,
            "sources_compared": reserved_sources,
            "rule": (
                "elgs/trainer_hooks.build_reserved_pool: within each timestamp "
                "group (cameras sorted by image_name), reserve "
                "(frame_order + camera_order) % 4 == 0"
            ),
            "reserved_parity_log_present": parity is not None,
        },
        "fbox": {
            "box_x0_y0_x1_y1_inclusive": list(fbox),
            "frame": int(args.fbox_frame),
            "camera_id": fbox_camera_id,
            "image_name": str(getattr(fbox_camera, "image_name", "")),
            "timestamp_seconds": fbox_timestamp,
            "timestamp_nominal_seconds": nominal,
            "frame_dt_seconds": frame_dt,
            "image_width": int(getattr(fbox_camera, "image_width", 0)),
            "image_height": int(getattr(fbox_camera, "image_height", 0)),
            "gated_rows_in_box": gated_rows_fbox,
            "gated_rows_on_screen": gated_rows_onscreen,
            "all_rows_in_box": rows_fbox_all,
            "projection": (
                "utils.motion_prior_utils.project_points_to_screen on "
                "gaussians.get_dynamic_xyz(camera.timestamp), pixels rounded "
                "as scripts/estimate_episodes.build_footprints rounds them; "
                "rows failing the projector's `valid` test are excluded"
            ),
        },
        "presence": {
            "n_frames_tested": int(args.n_frames),
            "frames_presence_zero": frames_presence_zero_list,
            "program_frames_in_gap": program_frames_in_gap,
            "program_frames_in_gap_count": len(program_frames_in_gap),
            "program_gaps_seconds": {
                str(k): v for k, v in sorted(program_gaps.items())
            },
            "frames_presence_zero_agrees": presence_agrees,
            "frames_presence_zero_disagreements": disagreements,
            "source": (
                "live runtime get_elgs_presence == 0 on a gated row"
                if state is not None else
                "ungated arm: no row carries an episodic program, so no frame "
                "has presence exactly 0"
            ),
        },
        "splits": {
            "train_camera_ids": train_camera_ids,
            "held_out_camera_ids": test_camera_ids,
            "eval_split_enabled": bool(getattr(dataset_params, "eval", False)),
        },
        "program_family_match": program_match,
        "checkpoint_iteration": int(ckpt_iteration),
    }

    provenance = {
        "script": os.path.relpath(os.path.abspath(__file__), repo_root).replace(
            os.sep, "/"),
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(str(checkpoint)),
        "program": str(args.program),
        "program_sha256": sha256_file(str(args.program)),
        "config": str(args.config),
        "config_sha256": sha256_file(str(args.config)),
        "cfg_args": str(cfg_args_path) if cfg_args_text else None,
        "cfg_args_sha256": (
            sha256_file(str(cfg_args_path)) if cfg_args_text else None
        ),
        "train_log": str(log_path),
        "train_log_sha256": sha256_file(str(log_path)),
        "source_path": str(dataset_params.source_path),
        "model_path": str(dataset_params.model_path),
        "torch_version": str(torch.__version__),
        "git": git_provenance(repo_root),
    }

    payload = assemble_precondition(
        arm_kind=arm_kind,
        gated_rows_seeding=gated_rows_seeding,
        n_rows_seeding=n_rows_seeding,
        gated_rows_final=gated_rows_final,
        n_rows_final=n_rows_final,
        gated_rows_fbox_frame150=gated_rows_fbox,
        frames_presence_zero=frames_presence_zero,
        reserved_units=reserved_units,
        training_units_total=training_units_total,
        detail=detail,
        provenance=provenance,
    )

    out_path = run_dir / "precondition.json"
    out_path.write_text(
        json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(
        {field: payload[field] for field in PRECONDITION_FIELDS},
        indent=1, sort_keys=True,
    ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
