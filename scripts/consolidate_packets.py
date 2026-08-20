#!/usr/bin/env python3
"""CCR v3 post-training consolidation pass (ccr-method-2026-08-20 §3.2).

Loads a trained B1 checkpoint plus its packet state, proposes packet
pairs, admits ties sequentially by the preregistered selection rule on
reserved units, applies the ONE frozen joint veto, and writes the
appearance-edit sidecar. The deployed B2 state is byte-identical to B1
outside the pointer/mode columns (hash-verified here).

Control arms (§3.3, all through the IDENTICAL machinery):
  --control none           the certified pass (default)
  --control placebo        full pass, commits suppressed (identity edit)
  --control shuffled       admitted-count-matched RANDOM edges committed
                           without certification (measures arbitrary
                           tying at equal budget; runs the certified
                           pass first to learn the admitted count)
  --control easy_negative  proposal-count-matched descriptor-FARTHEST
                           disjoint pairs through the identical
                           screen/confirm machinery (heuristic
                           specificity control, NOT known negatives)

Frozen constants (any change is a new spec, not a flag):
  K_PROPOSALS_PER_PACKET = 3       proposals per packet
  P_EVALUATED_MAX        = 128     descriptor-screened pairs entering the screen
  SCREEN_UNITS           = 8       stratified training units for the screen
  P_CONFIRM_MAX          = 10      screen survivors entering confirmation
  CONFIRM_UNITS_PER_SIDE = 8       reserved units per episode side (6 floor,
                                   10 cap when topping up to the 16 total)
  CONFIRM_UNITS_TOTAL    = 16      minimum units per decision
  JOINT_VETO_UNITS       = 64      time-STRATIFIED reserved units for the veto
  DISJOINT_FRAMES_MIN    = 2       temporal separation between packet supports
  T_SUPPORT_SIGMA        = 2.0     effective support = t_mean +/- 2 sigma_t
  ADMIT_K_SE             = 3.0     admit iff mean + 3*SE < 0 (a preregistered
                                   selection rule, not an inferential guarantee)

Sequential admission order (frozen §3.2 item 3): ascending descriptor
distance (= descending descriptor score), fixed BEFORE any confirmation
render. The stage-1 screen only FILTERS (mean delta < 0); it never
reorders the admission sequence.
"""

from __future__ import annotations

import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os  # noqa: E402  (after sys.path for parity with sibling scripts)

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from depth_visibility.errors import ContractError  # noqa: E402
from elgs.trainer_hooks import build_reserved_pool  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from scene.appearance_edit import (  # noqa: E402
    build_edit_payload,
    clear_appearance_edit,
    non_pointer_state_hash,
)
from scene.packet_birth import MIN_PACKET_ROWS  # noqa: E402
from utils.general_utils import build_rotation  # noqa: E402

K_PROPOSALS_PER_PACKET = 3
P_EVALUATED_MAX = 128
SCREEN_UNITS = 8
P_CONFIRM_MAX = 10
CONFIRM_UNITS_PER_SIDE = 8
CONFIRM_UNITS_SIDE_MIN = 6
CONFIRM_UNITS_SIDE_CAP = 10
CONFIRM_UNITS_TOTAL = 16
JOINT_VETO_UNITS = 64
DISJOINT_FRAMES_MIN = 2
T_SUPPORT_SIGMA = 2.0
ADMIT_K_SE = 3.0
ROW_MAP_CHUNK = 4096
MEDOID_GATE_SCALE = 2.0


# ---------------------------------------------------------------------------
# packets and descriptors
# ---------------------------------------------------------------------------


def packet_table(gaussians, packet_ids):
    """Per-packet descriptor table from the FROZEN final state.

    `rows` is the full membership; `rows_tieable` applies the §3.2
    per-row medoid-consistency gate (rows whose DC distance to the
    packet medoid exceeds MEDOID_GATE_SCALE x the median distance stay
    independent and are never tied).
    """
    n = gaussians._xyz.shape[0]
    sigma_t = torch.exp(gaussians._scaling_t.detach()).reshape(-1).cpu()
    t_mean = gaussians._t.detach().reshape(-1).cpu()
    dc = gaussians._features_dc.detach().reshape(n, -1).cpu()
    rest = gaussians._features_rest.detach()
    rest_energy = rest.reshape(rest.shape[0], -1).pow(2).mean(1).cpu()
    scale_log = gaussians._scaling.detach().cpu()
    scale = scale_log.exp().mean(1)
    xyz = gaussians._xyz.detach().cpu()
    rot = build_rotation(F.normalize(gaussians._rotation.detach())).cpu()
    min_axis = scale_log.argmin(dim=1)
    normals = rot[torch.arange(n), :, min_axis]
    normals = F.normalize(normals, dim=1)

    table = {}
    for pid in sorted(set(int(p) for p in packet_ids.tolist() if p >= 0)):
        rows = torch.nonzero(packet_ids == pid).reshape(-1)
        if rows.numel() < MIN_PACKET_ROWS:
            continue
        row_dc = dc[rows]
        medoid = row_dc.median(dim=0).values
        med_dist = (row_dc - medoid).norm(dim=1)
        gate = med_dist <= MEDOID_GATE_SCALE * med_dist.median().clamp_min(1e-8)
        rows_tieable = rows[gate]
        if rows_tieable.numel() < MIN_PACKET_ROWS:
            continue
        lo = float((t_mean[rows] - T_SUPPORT_SIGMA * sigma_t[rows]).min())
        hi = float((t_mean[rows] + T_SUPPORT_SIGMA * sigma_t[rows]).max())
        packet_normal = F.normalize(normals[rows].mean(dim=0), dim=0)
        table[pid] = {
            "rows": rows,
            "rows_tieable": rows_tieable,
            "row_dc_tieable": row_dc[gate],
            "dc": row_dc.mean(0),
            "rest_energy": float(rest_energy[rows].mean()),
            "scale": float(scale[rows].mean()),
            "normal": packet_normal,
            "centroid": xyz[rows].mean(0),
            "extent": float((xyz[rows] - xyz[rows].mean(0)).norm(dim=1).mean()),
            "support": (lo, hi),
        }
    return table


def count_observations(table, unit_times):
    """§3.2: the donor is the packet with more TRAINING-VIEW
    observations — counted as training units whose timestamp falls in
    the packet's effective support (not a duration proxy)."""
    for entry in table.values():
        lo, hi = entry["support"]
        entry["observations"] = sum(1 for t in unit_times if lo <= t <= hi)


def descriptor_distance(a, b):
    return (
        float((a["dc"] - b["dc"]).norm())
        + abs(a["rest_energy"] - b["rest_energy"])
        + abs(a["scale"] - b["scale"]) / max(a["scale"] + b["scale"], 1e-6)
        + float((a["centroid"] - b["centroid"]).norm())
        / max(a["extent"] + b["extent"], 1e-6)
        + (1.0 - abs(float(torch.dot(a["normal"], b["normal"]))))
    )


def disjoint_pairs(table, frame_dt):
    out = []
    pids = sorted(table)
    for i, pa in enumerate(pids):
        for pb in pids[i + 1:]:
            a, b = table[pa], table[pb]
            gap = max(b["support"][0] - a["support"][1],
                      a["support"][0] - b["support"][1])
            if gap >= DISJOINT_FRAMES_MIN * frame_dt:
                out.append((descriptor_distance(a, b), pa, pb))
    return out


def propose_pairs(table, frame_dt, *, farthest=False):
    """Mutual-nearest descriptor matching between temporally disjoint
    packets, ordered by ascending descriptor distance (the frozen
    admission order). `farthest=True` is the easy-negative control:
    the same budget of descriptor-FARTHEST disjoint pairs."""
    all_pairs = disjoint_pairs(table, frame_dt)
    if farthest:
        return sorted(all_pairs, reverse=True)[:P_EVALUATED_MAX], len(all_pairs)
    per_packet = {}
    for dist, pa, pb in all_pairs:
        per_packet.setdefault(pa, []).append((dist, pb))
        per_packet.setdefault(pb, []).append((dist, pa))
    for pid in per_packet:
        per_packet[pid] = sorted(per_packet[pid])[:K_PROPOSALS_PER_PACKET]
    seen, pairs = set(), []
    for pa, lst in per_packet.items():
        for dist, pb in lst:
            if any(q == pa for _, q in per_packet.get(pb, [])):
                key = (min(pa, pb), max(pa, pb))
                if key not in seen:
                    seen.add(key)
                    pairs.append((dist, key[0], key[1]))
    pairs.sort()
    return pairs[:P_EVALUATED_MAX], len(all_pairs)


def row_map(table, donor, recipient):
    """recipient-tieable-row -> donor-tieable-row by nearest DC,
    chunked (packets grow by clone inheritance; an unchunked cdist over
    tens of thousands of rows is an OOM risk)."""
    d_rows = table[donor]["rows_tieable"]
    r_rows = table[recipient]["rows_tieable"]
    d_dc = table[donor]["row_dc_tieable"]
    r_dc = table[recipient]["row_dc_tieable"]
    idx_chunks = []
    for start in range(0, r_dc.shape[0], ROW_MAP_CHUNK):
        chunk = r_dc[start:start + ROW_MAP_CHUNK]
        idx_chunks.append(torch.cdist(chunk, d_dc).argmin(dim=1))
    return r_rows, d_rows[torch.cat(idx_chunks)]


# ---------------------------------------------------------------------------
# units and rendering
# ---------------------------------------------------------------------------


def unit_ground_truth_and_camera(unit):
    """CameraDataset items are (gt_image, camera); bare cameras carry
    .image (scene/cameras.py). NOT elgs._unpack_camera's return order."""
    if isinstance(unit, (tuple, list)) and len(unit) == 2:
        return unit[0], unit[1]
    return unit.image, unit


def unit_l1(unit, gaussians, pipe, background):
    gt, cam = unit_ground_truth_and_camera(unit)
    cam = cam.cuda()
    out = render(cam, gaussians, pipe, background)
    return float((out["render"] - gt.cuda()).abs().mean())


def paired_delta(units, gaussians, pipe, background, pointer_current,
                 pointer_candidate, mode):
    """L1(candidate) - L1(current) per unit; back-to-back renders."""
    device = gaussians._features_dc.device
    deltas = []
    for unit in units:
        gaussians._appearance_source_idx = pointer_current.to(device)
        gaussians._appearance_share_mode = mode
        base = unit_l1(unit, gaussians, pipe, background)
        gaussians._appearance_source_idx = pointer_candidate.to(device)
        cand = unit_l1(unit, gaussians, pipe, background)
        deltas.append(cand - base)
    return deltas


def stratified_indices(entries, count):
    """Evenly spread positions over an ordered (index, t) list — the
    veto partition and the screen must span the WHOLE segment. A
    contiguous slice of the time-ordered reserved pool would
    concentrate on one end of the segment and make the §6.1 veto
    vacuous for edges elsewhere in time."""
    n = len(entries)
    if count >= n:
        return list(range(n))
    positions = sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})
    step = 0
    while len(positions) < count:  # collisions from rounding
        step += 1
        extras = [p for p in range(n) if p not in positions]
        positions.append(extras[0])
        positions.sort()
        if step > n:
            break
    return positions[:count]


def pick_confirmation_slot(slot_pool, used_units, lo_d, hi_d, lo_r, hi_r):
    """Disjoint per-decision slot: >= 6 per episode side, >= 16 total
    (§3.2 item 3), topping either side up to 10 when the other is
    short. Returns (side_d, side_r) or None."""
    avail_d = [u for u, t in slot_pool if u not in used_units and lo_d <= t <= hi_d]
    avail_r = [u for u, t in slot_pool
               if u not in used_units and lo_r <= t <= hi_r and u not in avail_d]
    side_d = avail_d[:CONFIRM_UNITS_PER_SIDE]
    side_r = avail_r[:CONFIRM_UNITS_PER_SIDE]
    if len(side_d) + len(side_r) < CONFIRM_UNITS_TOTAL:
        side_d = avail_d[:CONFIRM_UNITS_SIDE_CAP]
        side_r = avail_r[:CONFIRM_UNITS_SIDE_CAP]
        side_d = side_d[:max(CONFIRM_UNITS_TOTAL - len(side_r),
                             CONFIRM_UNITS_PER_SIDE)]
    if (len(side_d) < CONFIRM_UNITS_SIDE_MIN
            or len(side_r) < CONFIRM_UNITS_SIDE_MIN
            or len(side_d) + len(side_r) < CONFIRM_UNITS_TOTAL):
        return None
    return side_d, side_r


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _merge_config(args, config_path):
    """YAML-over-argparse merge, identical to scripts/eval_lrv1_event.py."""
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


def main(argv=None):
    parser = ArgumentParser(description="CCR v3 consolidation pass")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start_checkpoint", required=True)
    parser.add_argument("--packet_state", required=True)
    parser.add_argument("--arm", choices=("dc", "full"), required=True)
    parser.add_argument("--control",
                        choices=("none", "placebo", "shuffled", "easy_negative"),
                        default="none")
    parser.add_argument("--out_edit", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--gaussian_dim", type=int, default=4)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[0.0, 1.6340])
    parser.add_argument("--num_pts", type=int, default=300_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exhaust_test", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    _merge_config(args, args.config)

    # model_path fallback, exactly as scripts/eval_lrv1_event.py: the run
    # dir is where Scene() writes input.ply/cameras.json.
    run_dir = os.environ.get("ADAGS_RUN_DIR", "").strip()
    if not str(getattr(args, "model_path", "") or "").strip():
        if not run_dir:
            raise ContractError("--model_path is required when ADAGS_RUN_DIR is unset")
        args.model_path = run_dir
    os.makedirs(args.model_path, exist_ok=True)
    for out in (args.out_edit, args.out_report):
        Path(out).parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(
        dataset.sh_degree, gaussian_dim=args.gaussian_dim,
        time_duration=args.time_duration, rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0,
    )
    scene = Scene(dataset, gaussians, num_pts=args.num_pts,
                  num_pts_ratio=args.num_pts_ratio,
                  time_duration=args.time_duration, shuffle=False)
    scene.opt = opt
    gaussians.training_setup(opt)
    model_params, _ = torch.load(args.start_checkpoint)
    gaussians.restore(model_params, opt)
    clear_appearance_edit(gaussians)

    state = torch.load(args.packet_state, map_location="cpu")
    packet_ids = state["packet_ids"]
    n_rows = gaussians._xyz.shape[0]
    if packet_ids.shape[0] != n_rows:
        raise ContractError(
            f"packet state has {packet_ids.shape[0]} rows, model has {n_rows}")

    pre_hash = non_pointer_state_hash(gaussians)
    search_t0 = time.perf_counter()

    # Camera METADATA from the raw camera list (never decodes an image);
    # the dataset is indexed only for units that actually render.
    train_units = scene.getTrainCameras()
    cameras_meta = scene.train_cameras[1.0]
    if len(cameras_meta) != len(train_units):
        raise ContractError("camera metadata / dataset length mismatch")
    reserved = build_reserved_pool(cameras_meta)
    if len(reserved) < JOINT_VETO_UNITS + 2 * CONFIRM_UNITS_PER_SIDE:
        raise ContractError(
            f"reserved pool has {len(reserved)} units; the pass refuses to "
            f"run below {JOINT_VETO_UNITS + 2 * CONFIRM_UNITS_PER_SIDE}")
    veto_positions = set(stratified_indices(reserved, JOINT_VETO_UNITS))
    veto_units = [reserved[i][0] for i in sorted(veto_positions)]
    slot_pool = [reserved[i] for i in range(len(reserved))
                 if i not in veto_positions]
    if len(veto_units) < 32:
        raise ContractError("joint-veto partition below the 32-unit floor")

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda")

    table = packet_table(gaussians, packet_ids)
    unit_times = [float(getattr(c, "timestamp", 0.0)) for c in cameras_meta]
    count_observations(table, unit_times)

    frame_dt = float(getattr(opt, "motion_track_dt", 1.0 / 30.0))
    pairs, n_disjoint = propose_pairs(
        table, frame_dt, farthest=(args.control == "easy_negative"))

    funnel = {
        "arm": args.arm,
        "control": args.control,
        "packets": len(table),
        "disjoint_pairs": n_disjoint,
        "descriptor_survivors": len(pairs),
        "screened": 0,
        "screen_survivors": 0,
        "confirmed_attempted": 0,
        "admitted": 0,
        "unconfirmable": 0,
        "joint_veto": None,
    }

    identity = torch.arange(n_rows)
    pointer = identity.clone()
    admitted_edges = []
    locked_rows = torch.zeros(n_rows, dtype=torch.bool)

    # STRATIFIED screen units: the frame ordering on disk is
    # camera-major, so "the first 8 non-reserved indices" would all be
    # one camera in the first ~10 frames and silently zero the funnel
    # for every packet elsewhere in time.
    reserved_set = {u for u, _ in reserved}
    non_reserved = [(i, float(getattr(cameras_meta[i], "timestamp", 0.0)))
                    for i in range(len(cameras_meta)) if i not in reserved_set]
    non_reserved.sort(key=lambda p: (p[1], p[0]))
    screen_ids = [non_reserved[i][0]
                  for i in stratified_indices(non_reserved, SCREEN_UNITS)]
    screen_units = [train_units[i] for i in screen_ids]

    with torch.no_grad():
        # ---- stage 1: cheap screen (FILTERS only; never reorders)
        screened = []
        for dist, pa, pb in pairs:
            donor, recipient = (
                (pa, pb) if table[pa]["observations"] >= table[pb]["observations"]
                else (pb, pa))
            r_rows, d_rows = row_map(table, donor, recipient)
            cand = pointer.clone()
            cand[r_rows] = d_rows
            deltas = paired_delta(screen_units, gaussians, pipe, background,
                                  identity, cand, args.arm)
            mean = sum(deltas) / len(deltas)
            funnel["screened"] += 1
            if mean < 0:
                screened.append((dist, mean, donor, recipient, r_rows, d_rows))
        # admission order: ascending descriptor distance (frozen §3.2),
        # NOT the screen measurement
        screened.sort(key=lambda s: s[0])
        screened = screened[:P_CONFIRM_MAX]
        funnel["screen_survivors"] = len(screened)

        # ---- stage 2: sequential confirmation on disjoint reserved slots
        used_units: set[int] = set()
        for dist, mean_screen, donor, recipient, r_rows, d_rows in screened:
            # one-hop + disjointness BEFORE any unit is consumed: no row
            # of this edge may already participate in an admitted edge
            if bool(locked_rows[r_rows].any()) or bool(locked_rows[d_rows].any()):
                funnel["unconfirmable"] += 1
                continue
            slot = pick_confirmation_slot(
                slot_pool, used_units,
                *table[donor]["support"], *table[recipient]["support"])
            if slot is None:
                funnel["unconfirmable"] += 1
                continue
            side_d, side_r = slot
            funnel["confirmed_attempted"] += 1
            used_units.update(side_d + side_r)
            cand = pointer.clone()
            cand[r_rows] = d_rows
            dd = paired_delta([train_units[u] for u in side_d], gaussians,
                              pipe, background, pointer, cand, args.arm)
            dr = paired_delta([train_units[u] for u in side_r], gaussians,
                              pipe, background, pointer, cand, args.arm)
            all_d = dd + dr
            n = len(all_d)
            mean = sum(all_d) / n
            var = sum((x - mean) ** 2 for x in all_d) / max(n - 1, 1)
            se = (var / n) ** 0.5
            side_ok = (sum(dd) / len(dd) <= 0) and (sum(dr) / len(dr) <= 0)
            admit = mean + ADMIT_K_SE * se < 0 and side_ok
            if admit and args.control != "placebo":
                pointer = cand
                locked_rows[r_rows] = True
                locked_rows[d_rows] = True
            if admit:
                funnel["admitted"] += 1
                admitted_edges.append({
                    "donor": donor, "recipient": recipient,
                    "rows": int(r_rows.numel()),
                    "mean_delta": mean, "se": se,
                    "screen_delta": mean_screen,
                    "descriptor_distance": dist,
                    "units": n,
                })

        # ---- shuffled control: admitted-count-matched RANDOM edges,
        # committed through the identical pointer machinery, no
        # certification (the certified stage above ran with placebo-like
        # suppression? no -- shuffled reuses the admitted COUNT from the
        # certified pass just executed, then replaces the pointer)
        if args.control == "shuffled" and funnel["admitted"] > 0:
            gen = torch.Generator().manual_seed(args.seed + 991)
            pointer = identity.clone()
            locked_rows.zero_()
            shuffled_edges = []
            order = torch.randperm(len(pairs), generator=gen).tolist()
            for k in order:
                if len(shuffled_edges) >= funnel["admitted"]:
                    break
                dist, pa, pb = pairs[k]
                donor, recipient = (
                    (pa, pb)
                    if table[pa]["observations"] >= table[pb]["observations"]
                    else (pb, pa))
                r_rows, d_rows = row_map(table, donor, recipient)
                if bool(locked_rows[r_rows].any()) or bool(locked_rows[d_rows].any()):
                    continue
                pointer[r_rows] = d_rows
                locked_rows[r_rows] = True
                locked_rows[d_rows] = True
                shuffled_edges.append({"donor": donor, "recipient": recipient,
                                       "rows": int(r_rows.numel())})
            funnel["shuffled_edges"] = shuffled_edges

        # ---- the single joint veto (all-or-nothing, one access)
        if funnel["admitted"] > 0 and args.control in ("none", "easy_negative",
                                                       "shuffled"):
            dv = paired_delta([train_units[u] for u in veto_units], gaussians,
                              pipe, background, identity, pointer, args.arm)
            nan_seen = any(x != x for x in dv)
            veto_mean = (sum(dv) / len(dv)) if not nan_seen else float("nan")
            outcome = "PASS"
            if nan_seen:
                outcome = "REJECT_ALL_NAN"  # fail closed: B2 := B1, reported
            elif veto_mean > 0:
                outcome = "REJECT_ALL"
            funnel["joint_veto"] = {
                "mean_delta": None if nan_seen else veto_mean,
                "units": len(dv), "outcome": outcome,
            }
            if outcome != "PASS":
                pointer = identity.clone()
                funnel["admitted_before_veto"] = funnel["admitted"]
                funnel["admitted"] = 0
                admitted_edges = []

    if args.control == "placebo":
        pointer = identity.clone()

    clear_appearance_edit(gaussians)
    post_hash = non_pointer_state_hash(gaussians)
    if pre_hash != post_hash:
        raise ContractError(
            "non-pointer state changed during the pass — the v3 invariant "
            "is violated and the edit must not be trusted")

    funnel["search_seconds"] = round(time.perf_counter() - search_t0, 1)
    redirected = int((pointer != identity).sum())
    funnel["redirected_rows"] = redirected
    funnel["effective_params_removed"] = int(
        redirected * gaussians._features_dc[0].numel()
        + (redirected * gaussians._features_rest[0].numel()
           if args.arm == "full" else 0))

    payload = build_edit_payload(
        pointer, args.arm, source_checkpoint=args.start_checkpoint,
        funnel=funnel)
    torch.save(payload, args.out_edit)
    report = {
        "schema": "ccr-consolidation-report-v1",
        "funnel": funnel,
        "admitted_edges": admitted_edges,
        "state_hash": pre_hash,
        "arm": args.arm,
        "control": args.control,
        "checkpoint": args.start_checkpoint,
        "veto_unit_indices": veto_units,
        "screen_unit_indices": screen_ids,
    }
    Path(args.out_report).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({"ccr_consolidation": funnel}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
