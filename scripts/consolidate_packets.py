#!/usr/bin/env python3
"""CCR v3 post-training consolidation pass (ccr-method-2026-08-20 §3.2).

Loads a trained B1 checkpoint plus its packet state, proposes packet
pairs, admits ties sequentially by the preregistered selection rule on
reserved units, applies the ONE frozen joint veto, and writes the
appearance-edit sidecar. The deployed B2 state is byte-identical to B1
outside the pointer/mode columns (hash-verified here).

Frozen constants (any change is a new spec, not a flag):
  K_PROPOSALS_PER_PACKET = 3       proposals per packet
  P_EVALUATED_MAX        = 128     descriptor-screened pairs entering the screen
  SCREEN_UNITS           = 8       training units for the cheap screen
  P_CONFIRM_MAX          = 10      screen survivors entering confirmation
  CONFIRM_UNITS_PER_SIDE = 8       reserved units per episode side per decision
  JOINT_VETO_UNITS       = 64      held-back reserved units for the single veto
  DISJOINT_FRAMES_MIN    = 2       temporal separation between packet supports
  T_SUPPORT_SIGMA        = 2.0     effective support = t_mean +/- 2 sigma_t
  ADMIT_K_SE             = 3.0     admit iff mean + 3*SE < 0 (selection rule,
                                   not an inferential guarantee)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from argparse import ArgumentParser

import yaml

from arguments import ModelParams, OptimizationParams, PipelineParams
from depth_visibility.errors import ContractError
from elgs.trainer_hooks import _unpack_camera, build_reserved_pool
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.appearance_edit import (
    apply_appearance_edit,
    build_edit_payload,
    clear_appearance_edit,
    non_pointer_state_hash,
)

K_PROPOSALS_PER_PACKET = 3
P_EVALUATED_MAX = 128
SCREEN_UNITS = 8
P_CONFIRM_MAX = 10
CONFIRM_UNITS_PER_SIDE = 8
JOINT_VETO_UNITS = 64
DISJOINT_FRAMES_MIN = 2
T_SUPPORT_SIGMA = 2.0
ADMIT_K_SE = 3.0
MIN_PACKET_ROWS = 4


def packet_table(gaussians, packet_ids):
    """Per-packet descriptor table from the FROZEN final state."""
    table = {}
    sigma_t = torch.exp(gaussians._scaling_t.detach()).reshape(-1).cpu()
    t_mean = gaussians._t.detach().reshape(-1).cpu()
    dc = gaussians._features_dc.detach().reshape(len(t_mean), -1).cpu()
    rest = gaussians._features_rest.detach()
    rest_energy = rest.reshape(rest.shape[0], -1).pow(2).mean(1).cpu()
    scale = gaussians._scaling.detach().exp().mean(1).cpu()
    xyz = gaussians._xyz.detach().cpu()
    for pid in sorted(set(int(p) for p in packet_ids.tolist() if p >= 0)):
        rows = torch.nonzero(packet_ids == pid).reshape(-1)
        if rows.numel() < MIN_PACKET_ROWS:
            continue
        lo = float((t_mean[rows] - T_SUPPORT_SIGMA * sigma_t[rows]).min())
        hi = float((t_mean[rows] + T_SUPPORT_SIGMA * sigma_t[rows]).max())
        table[pid] = {
            "rows": rows,
            "dc": dc[rows].mean(0),
            "rest_energy": float(rest_energy[rows].mean()),
            "scale": float(scale[rows].mean()),
            "centroid": xyz[rows].mean(0),
            "extent": float((xyz[rows] - xyz[rows].mean(0)).norm(dim=1).mean()),
            "support": (lo, hi),
            "observations": rows.numel() * max(hi - lo, 1e-6),
        }
    return table


def descriptor_distance(a, b):
    return (
        float((a["dc"] - b["dc"]).norm())
        + abs(a["rest_energy"] - b["rest_energy"])
        + abs(a["scale"] - b["scale"]) / max(a["scale"], 1e-6)
        + float((a["centroid"] - b["centroid"]).norm()) / max(a["extent"] + b["extent"], 1e-6)
    )


def propose_pairs(table, frame_dt):
    """Mutual-nearest descriptor matching between temporally disjoint
    packets; PRESPECIFIED order: ascending descriptor distance."""
    pids = sorted(table)
    candidates = {}
    for i, pa in enumerate(pids):
        for pb in pids[i + 1:]:
            a, b = table[pa], table[pb]
            gap = max(b["support"][0] - a["support"][1],
                      a["support"][0] - b["support"][1])
            if gap < DISJOINT_FRAMES_MIN * frame_dt:
                continue
            candidates.setdefault(pa, []).append((descriptor_distance(a, b), pb))
            candidates.setdefault(pb, []).append((descriptor_distance(a, b), pa))
    for pid in candidates:
        candidates[pid] = sorted(candidates[pid])[:K_PROPOSALS_PER_PACKET]
    pairs = []
    seen = set()
    for pa, lst in candidates.items():
        for dist, pb in lst:
            # mutual-nearest: pb must also shortlist pa
            if any(q == pa for _, q in candidates.get(pb, [])):
                key = (min(pa, pb), max(pa, pb))
                if key not in seen:
                    seen.add(key)
                    pairs.append((dist, key[0], key[1]))
    pairs.sort()
    return pairs[:P_EVALUATED_MAX]


def row_map(table, donor, recipient):
    """recipient-row -> donor-row by nearest DC descriptor."""
    d_rows = table[donor]["rows"]
    r_rows = table[recipient]["rows"]
    d_dc = table[donor]["_row_dc"]
    r_dc = table[recipient]["_row_dc"]
    idx = torch.cdist(r_dc, d_dc).argmin(dim=1)
    return r_rows, d_rows[idx]


def unit_l1(unit, gaussians, pipe, background):
    gt, cam = _unpack_camera(unit)
    if gt is cam:
        gt = cam.original_image
    cam = cam.cuda()
    out = render(cam, gaussians, pipe, background)
    return float((out["render"] - gt.cuda()).abs().mean())


def paired_delta(units, gaussians, pipe, background, pointer_current, payload_candidate, mode):
    """L1(candidate) - L1(current) per unit; renders back-to-back."""
    deltas = []
    for unit in units:
        gaussians._appearance_source_idx = pointer_current.to(
            gaussians._features_dc.device)
        gaussians._appearance_share_mode = mode
        base = unit_l1(unit, gaussians, pipe, background)
        gaussians._appearance_source_idx = payload_candidate.to(
            gaussians._features_dc.device)
        cand = unit_l1(unit, gaussians, pipe, background)
        deltas.append(cand - base)
    return deltas


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
    parser = ArgumentParser("CCR v3 consolidation pass")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start_checkpoint", required=True)
    parser.add_argument("--packet_state", required=True)
    parser.add_argument("--arm", choices=("dc", "full"), required=True)
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

    train_units = scene.getTrainCameras()
    cameras = [_unpack_camera(train_units[i])[0] for i in range(len(train_units))]
    reserved = build_reserved_pool(cameras)
    if len(reserved) < JOINT_VETO_UNITS + 2 * CONFIRM_UNITS_PER_SIDE:
        raise ContractError("reserved pool too small for the pass")
    veto_units = [u for u, _ in reserved[-JOINT_VETO_UNITS:]]
    slot_pool = [(u, t) for u, t in reserved[:-JOINT_VETO_UNITS]]

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda")

    table = packet_table(gaussians, packet_ids)
    # per-row DC for the row map
    flat_dc = gaussians._features_dc.detach().reshape(n_rows, -1).cpu()
    for pid, entry in table.items():
        entry["_row_dc"] = flat_dc[entry["rows"]]

    pairs = propose_pairs(table, frame_dt=float(getattr(opt, "motion_track_dt", 1.0 / 30.0)))
    funnel = {
        "packets": len(table),
        "pairs_proposed": len(pairs),
        "screened": 0, "confirmed_attempted": 0, "admitted": 0,
        "unconfirmable": 0, "joint_veto": None, "arm": args.arm,
    }

    identity = torch.arange(n_rows)
    pointer = identity.clone()

    with torch.no_grad():
        # ---- stage 1: cheap screen on the first SCREEN_UNITS training units
        screen_ids = [i for i in range(len(train_units))
                      if i not in {u for u, _ in reserved}][:SCREEN_UNITS]
        screen_units = [train_units[i] for i in screen_ids]
        screened = []
        for dist, pa, pb in pairs:
            donor, recipient = (pa, pb) if table[pa]["observations"] >= table[pb]["observations"] else (pb, pa)
            r_rows, d_rows = row_map(table, donor, recipient)
            cand = pointer.clone()
            cand[r_rows] = d_rows
            deltas = paired_delta(screen_units, gaussians, pipe, background,
                                  pointer, cand, args.arm)
            mean = sum(deltas) / len(deltas)
            screened.append((mean, dist, donor, recipient, r_rows, d_rows))
            funnel["screened"] += 1
        screened = [s for s in sorted(screened) if s[0] < 0][:P_CONFIRM_MAX]

        # ---- stage 2: sequential confirmation on disjoint reserved slots
        used_units: set[int] = set()
        admitted_edges = []
        for mean_screen, dist, donor, recipient, r_rows, d_rows in screened:
            lo_d, hi_d = table[donor]["support"]
            lo_r, hi_r = table[recipient]["support"]
            side_d = [u for u, t in slot_pool
                      if u not in used_units and lo_d <= t <= hi_d][:CONFIRM_UNITS_PER_SIDE]
            side_r = [u for u, t in slot_pool
                      if u not in used_units and lo_r <= t <= hi_r][:CONFIRM_UNITS_PER_SIDE]
            if len(side_d) < 6 or len(side_r) < 6:
                funnel["unconfirmable"] += 1
                continue
            funnel["confirmed_attempted"] += 1
            used_units.update(side_d + side_r)
            cand = pointer.clone()
            # one-hop invariants
            if bool((pointer[d_rows] != d_rows).any()) or bool((pointer[r_rows] != r_rows).any()):
                funnel["unconfirmable"] += 1
                continue
            cand[r_rows] = d_rows
            units_d = [train_units[u] for u in side_d]
            units_r = [train_units[u] for u in side_r]
            dd = paired_delta(units_d, gaussians, pipe, background, pointer, cand, args.arm)
            dr = paired_delta(units_r, gaussians, pipe, background, pointer, cand, args.arm)
            all_d = dd + dr
            n = len(all_d)
            mean = sum(all_d) / n
            var = sum((x - mean) ** 2 for x in all_d) / max(n - 1, 1)
            se = (var / n) ** 0.5
            side_ok = (sum(dd) / len(dd) <= 0) and (sum(dr) / len(dr) <= 0)
            if mean + ADMIT_K_SE * se < 0 and side_ok:
                pointer = cand
                funnel["admitted"] += 1
                admitted_edges.append({
                    "donor": donor, "recipient": recipient,
                    "rows": int(r_rows.numel()),
                    "mean_delta": mean, "se": se,
                    "screen_delta": mean_screen, "descriptor_distance": dist,
                })

        # ---- the single joint veto (all-or-nothing)
        if funnel["admitted"] > 0:
            if len(veto_units) < 32:
                raise ContractError("joint-veto partition below the 32-unit floor")
            units_v = [train_units[u] for u in veto_units]
            dv = paired_delta(units_v, gaussians, pipe, background, identity, pointer, args.arm)
            if any(x != x for x in dv):
                raise ContractError("joint veto saw NaN")
            veto_mean = sum(dv) / len(dv)
            funnel["joint_veto"] = {"mean_delta": veto_mean, "units": len(dv)}
            if veto_mean > 0:
                pointer = identity.clone()
                funnel["joint_veto"]["outcome"] = "REJECT_ALL"
                admitted_edges = []
            else:
                funnel["joint_veto"]["outcome"] = "PASS"

    clear_appearance_edit(gaussians)
    post_hash = non_pointer_state_hash(gaussians)
    if pre_hash != post_hash:
        raise ContractError(
            "non-pointer state changed during the pass — the v3 invariant "
            "is violated and the edit must not be trusted")

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
        "checkpoint": args.start_checkpoint,
    }
    Path(args.out_report).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({"ccr_consolidation": funnel}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
