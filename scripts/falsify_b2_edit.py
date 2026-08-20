#!/usr/bin/env python3
"""Lane-4 synthetic falsification of the B2 directional donor DC edit.

EXPLORATORY. Takes the trained LRV3 B1-packets state and asks the one
question the N3V ladder cannot answer: when the edit under test
(`scene.appearance_edit`, "dc" arm — recipient rows render with the
DONOR row's base radiance and keep their own higher-order SH) is applied
to a link that is CORRECT BY AUTHORED CONSTRUCTION, does the CCR
certificate (ccr-method-2026-08-20 §3.2 item 3) admit it, and does the
event region actually improve?

Three links are built from authored ground truth rather than from the
descriptor proposer, so proposal ambiguity is removed from the question:

  L1 oracle-correct   episode-1 rows of the event object -> its return rows
  L2 wrong-identity   descriptor-close rows OFF the object -> the same
                      return rows (a known negative; the fixture is the
                      only place known identity negatives exist, §3.3)
  L3 same-identity    episode-1 rows of the object split by row-index
     no-op            parity, donor half -> recipient half (trained
                      appearance expected near-identical: the vacuity
                      reference)

Row sets come from the authored oracle region and the effective temporal
support mu +/- 2 sigma_t. That support is an OPERATIONAL MATCHING
INTERVAL at a declared cutoff — never the exact support of a Gaussian
temporal lobe, which is unbounded.

The comparative anti-vacuity gate runs, and is PRINTED, before any
reconstruction render: L1's pre-edit DC distance must strictly exceed
L3's, i.e. the oracle pair must be more different than the same-surface
no-op pair. A run where L1 rows already carry L3-level appearance
distances cannot falsify anything, so no delta is computed at all —
`falsification_flow` never calls the measurement function when the gate
returns a verdict.

The certificate RULE is the unchanged CCR one — reserved units from
`elgs.trainer_hooks.build_reserved_pool` (the b1_packets config sets
`elgs_reserved_parity`, so training dropped exactly these units), paired
per-unit photometric deltas, admit iff mean + 3*SE < 0 AND both
per-side means <= 0 — with ONE amended allocation semantics, decided
before any render on this fixture: links SHARE reserved units instead of
consuming disjoint slots (every link is scored against the same identity
base, never an accumulating admitted state, and LRV3's return window
holds only ~12 reserved units), and L3's slot windows are (W1, W1)
where its rows actually render. See the inline SLOT AMENDMENT comment.
Held-out cameras are REPORT-ONLY: the event_return PSNR delta is a
diagnostic, never a certificate input.
"""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402

SCHEMA = "ccr-b2-falsification-v1"

# ---------------------------------------------------------------------------
# Frozen constants (any change is a new spec, not a flag)
# ---------------------------------------------------------------------------

FRAME_DT = 1.0 / 6.0                       # LRV3 fixture: 6 fps, model seconds
WINDOW_EPISODE1 = (0.0, 29.0 / 6.0)        # W1  = [0, 4.8333]  frames 0-29
WINDOW_RETURN = (57.0 / 6.0, 59.0 / 6.0)   # WR  = [9.5, 9.8333] frames 57-59
RETURN_FRAMES = (57, 58, 59)
DONOR_PROBE_T = 2.5                        # mid-episode-1 position probe
RECIPIENT_PROBE_T = 9.6                    # mid-return position probe
DONOR_SUPPORT_UPPER_MAX = 5.0
RECIPIENT_SUPPORT_LOWER_MIN = 9.3
WRONG_IDENTITY_RADIUS_SCALE = 1.75
MIN_DONOR_ROWS = 8
MIN_RECIPIENT_ROWS = 4
# mirrors scripts/consolidate_packets.T_SUPPORT_SIGMA / ADMIT_K_SE /
# ROW_MAP_CHUNK; re-declared rather than imported because that module
# imports the CUDA renderer at import time and these constants must stay
# reachable from CPU tests.
T_SUPPORT_SIGMA = 2.0
ADMIT_K_SE = 3.0
ROW_MAP_CHUNK = 4096
WINDOW_EPS = 1e-9

EDIT_MODE = "dc"                           # the B2 PRIMARY arm, unchanged
LINK_L1 = "L1_oracle_correct"
LINK_L2 = "L2_wrong_identity"
LINK_L3 = "L3_same_identity_noop"
LINK_ORDER = (LINK_L1, LINK_L2, LINK_L3)   # prespecified; slots are consumed
                                           # in exactly this order

VERDICT_INVALID_SETS = "INVALID_SETS"
VERDICT_INVALID_VACUOUS = "INVALID_VACUOUS"
VERDICT_INVALID_SLOTS = "INVALID_SLOTS"
VERDICT_COMPLETED = "COMPLETED"


# ---------------------------------------------------------------------------
# Pure functions: windows, row sets, row maps, gate, report assembly
# ---------------------------------------------------------------------------


def effective_support(t_mean, scaling_t_log):
    """Operational matching interval per row: mu +/- 2 sigma_t, with
    sigma_t = exp(_scaling_t) (the parameter is stored in log scale).

    NOT the exact support of the row's temporal lobe; a declared cutoff.
    """
    t_mean = torch.as_tensor(t_mean).reshape(-1).to(torch.float64)
    sigma_t = torch.exp(torch.as_tensor(scaling_t_log).reshape(-1).to(torch.float64))
    return t_mean - T_SUPPORT_SIGMA * sigma_t, t_mean + T_SUPPORT_SIGMA * sigma_t


def window_intersects(support_lo, support_hi, window):
    lo, hi = float(window[0]), float(window[1])
    return (support_lo <= hi + WINDOW_EPS) & (support_hi >= lo - WINDOW_EPS)


def build_row_sets(position_episode1, position_return, support_lo, support_hi,
                   features_dc, centre, radius):
    """The authored row sets (frozen design item 1).

    All inputs are CPU tensors over the SAME row order:
      position_episode1  (N,3)  get_dynamic_xyz(DONOR_PROBE_T)
      position_return    (N,3)  get_dynamic_xyz(RECIPIENT_PROBE_T)
      support_lo/hi      (N,)   effective_support(...)
      features_dc        (N,C)  flattened DC features

    Returns a dict of LongTensors plus the counts the report carries.
    Spanning rows are excluded from BOTH sets and counted. (With the
    frozen edge conditions the exclusion is structurally implied — a row
    with support_hi <= 5.0 cannot reach WR — so it is defence in depth,
    and it stays explicit because the edge conditions are the part a
    round-2 spec is most likely to relax.)
    """
    centre = torch.as_tensor(centre, dtype=torch.float64).reshape(1, 3)
    radius = float(radius)
    p1 = torch.as_tensor(position_episode1, dtype=torch.float64).reshape(-1, 3)
    pr = torch.as_tensor(position_return, dtype=torch.float64).reshape(-1, 3)
    lo = torch.as_tensor(support_lo, dtype=torch.float64).reshape(-1)
    hi = torch.as_tensor(support_hi, dtype=torch.float64).reshape(-1)
    dc = torch.as_tensor(features_dc, dtype=torch.float64)
    dc = dc.reshape(dc.shape[0], -1)
    n = p1.shape[0]
    if not (pr.shape[0] == lo.shape[0] == hi.shape[0] == dc.shape[0] == n):
        raise ContractError("row-set inputs disagree on the row count")

    d_ep1 = (p1 - centre).norm(dim=1)
    d_ret = (pr - centre).norm(dim=1)
    hits_w1 = window_intersects(lo, hi, WINDOW_EPISODE1)
    hits_wr = window_intersects(lo, hi, WINDOW_RETURN)
    inside_ep1 = d_ep1 <= radius
    inside_ret = d_ret <= radius

    spanning_mask = hits_w1 & hits_wr & (inside_ep1 | inside_ret)
    donor_mask = (inside_ep1 & hits_w1
                  & (hi <= DONOR_SUPPORT_UPPER_MAX) & ~spanning_mask)
    recipient_mask = (inside_ret & hits_wr
                      & (lo >= RECIPIENT_SUPPORT_LOWER_MIN) & ~spanning_mask)
    wrong_pool_mask = ((d_ep1 > radius * WRONG_IDENTITY_RADIUS_SCALE) & hits_w1
                       & (hi <= DONOR_SUPPORT_UPPER_MAX) & ~spanning_mask)

    donor = torch.nonzero(donor_mask).reshape(-1)
    recipient = torch.nonzero(recipient_mask).reshape(-1)
    spanning = torch.nonzero(spanning_mask).reshape(-1)
    wrong_pool = torch.nonzero(wrong_pool_mask).reshape(-1)

    # Dw: descriptor-close but a DIFFERENT surface — ranked by DC
    # distance to the recipient set's DC medoid (median, the same medoid
    # convention consolidate_packets.packet_table uses), truncated to |D|.
    if recipient.numel() > 0 and wrong_pool.numel() > 0 and donor.numel() > 0:
        medoid = dc[recipient].median(dim=0).values
        dist = (dc[wrong_pool] - medoid).norm(dim=1)
        order = torch.argsort(dist)
        wrong = wrong_pool[order][:int(donor.numel())]
    else:
        wrong = torch.zeros(0, dtype=torch.long)

    donor_a, donor_b = split_by_row_parity(donor)
    return {
        "donor": donor,
        "recipient": recipient,
        "wrong": wrong,
        "spanning": spanning,
        "wrong_pool": wrong_pool,
        "donor_a": donor_a,
        "donor_b": donor_b,
    }


def split_by_row_parity(rows):
    """No-op link split: D_a = even row indices, D_b = odd (frozen
    design item 1). Disjoint by construction, and independent of any
    appearance quantity, so the no-op reference cannot be tuned."""
    rows = torch.as_tensor(rows).reshape(-1).to(torch.long)
    return rows[(rows % 2) == 0], rows[(rows % 2) == 1]


def sets_summary(sets):
    return {
        "donor_rows": int(sets["donor"].numel()),
        "recipient_rows": int(sets["recipient"].numel()),
        "wrong_identity_rows": int(sets["wrong"].numel()),
        "wrong_identity_pool_rows": int(sets["wrong_pool"].numel()),
        "spanning_rows_excluded": int(sets["spanning"].numel()),
        "noop_donor_rows": int(sets["donor_a"].numel()),
        "noop_recipient_rows": int(sets["donor_b"].numel()),
        "min_donor_rows": MIN_DONOR_ROWS,
        "min_recipient_rows": MIN_RECIPIENT_ROWS,
        "window_episode1": list(WINDOW_EPISODE1),
        "window_return": list(WINDOW_RETURN),
        "probe_times": [DONOR_PROBE_T, RECIPIENT_PROBE_T],
        "support_sigma_multiple": T_SUPPORT_SIGMA,
        "support_semantics": "operational matching interval at a declared "
                             "cutoff, NOT exact temporal support",
    }


def sets_are_sufficient(sets):
    return (int(sets["donor"].numel()) >= MIN_DONOR_ROWS
            and int(sets["recipient"].numel()) >= MIN_RECIPIENT_ROWS)


def assert_sets_disjoint(sets):
    """No link may make a row both a donor and a recipient (the one-hop
    invariant of scene.appearance_edit), and the no-op split must be a
    genuine partition."""
    pairs = (
        ("donor", "recipient"), ("wrong", "recipient"), ("wrong", "donor"),
        ("donor_a", "donor_b"),
    )
    for left, right in pairs:
        overlap = (set(sets[left].tolist()) & set(sets[right].tolist()))
        if overlap:
            raise ContractError(
                f"row sets {left} and {right} share {len(overlap)} row(s); "
                "the one-hop invariant would be violated")
    return True


def nearest_dc_row_map(features_dc, recipient_rows, donor_rows):
    """recipient row -> donor row by nearest DC, chunked — mirrors
    consolidate_packets.row_map semantics on explicit row sets."""
    dc = torch.as_tensor(features_dc, dtype=torch.float64)
    dc = dc.reshape(dc.shape[0], -1)
    r_rows = torch.as_tensor(recipient_rows).reshape(-1).to(torch.long)
    d_rows = torch.as_tensor(donor_rows).reshape(-1).to(torch.long)
    if r_rows.numel() == 0 or d_rows.numel() == 0:
        raise ContractError("row map needs a non-empty donor and recipient set")
    d_dc = dc[d_rows]
    r_dc = dc[r_rows]
    idx_chunks = []
    for start in range(0, r_dc.shape[0], ROW_MAP_CHUNK):
        chunk = r_dc[start:start + ROW_MAP_CHUNK]
        idx_chunks.append(torch.cdist(chunk, d_dc).argmin(dim=1))
    return r_rows, d_rows[torch.cat(idx_chunks)]


def link_pointer(n_rows, recipient_rows, donor_rows):
    """The pointer column for one link, with the one-hop invariant
    checked on the SETS (no row may be both donor and recipient)."""
    r_rows = torch.as_tensor(recipient_rows).reshape(-1).to(torch.long)
    d_rows = torch.as_tensor(donor_rows).reshape(-1).to(torch.long)
    if r_rows.shape != d_rows.shape:
        raise ContractError("row map is not one-to-one with the recipient set")
    overlap = set(r_rows.tolist()) & set(d_rows.tolist())
    if overlap:
        raise ContractError(
            f"one-hop invariant violated: {len(overlap)} row(s) are both "
            "donor and recipient in the same link")
    pointer = torch.arange(int(n_rows), dtype=torch.long)
    pointer[r_rows] = d_rows
    return pointer


def pre_edit_dc_stats(features_dc, recipient_rows, donor_rows):
    """Anti-vacuity measurement: how different are the mapped pairs
    BEFORE the edit? Computed with no render."""
    dc = torch.as_tensor(features_dc, dtype=torch.float64)
    dc = dc.reshape(dc.shape[0], -1)
    r_rows = torch.as_tensor(recipient_rows).reshape(-1).to(torch.long)
    d_rows = torch.as_tensor(donor_rows).reshape(-1).to(torch.long)
    dist = (dc[r_rows] - dc[d_rows]).norm(dim=1)
    changed = int((r_rows != d_rows).sum())
    return {
        "pre_edit_dc_distance_mean": float(dist.mean()) if dist.numel() else 0.0,
        "pre_edit_dc_distance_max": float(dist.max()) if dist.numel() else 0.0,
        "rows_changed": changed,
        "rows_changed_fraction": (changed / int(r_rows.numel())
                                  if r_rows.numel() else 0.0),
    }


def gate_decision(link_stats):
    """The comparative anti-vacuity gate (frozen design item 3).

    `link_stats` maps link name -> dict with at least
    `pre_edit_dc_distance_mean`, `rows_changed`, `slot_ok`.

    Returns (verdict_or_None, anti_vacuity_block). A non-None verdict
    means NO reconstruction delta may be computed. Precedence:
    INVALID_VACUOUS (the comparison is degenerate — a scientific
    refusal) before INVALID_SLOTS (an infrastructural refusal), so a
    vacuous fixture is never reported as a scheduling problem.
    """
    l1 = link_stats[LINK_L1]
    l3 = link_stats[LINK_L3]
    comparative_ok = bool(float(l1["pre_edit_dc_distance_mean"])
                          > float(l3["pre_edit_dc_distance_mean"]))
    rows_ok = int(l1["rows_changed"]) >= 1
    slots_ok = all(bool(link_stats[name].get("slot_ok")) for name in LINK_ORDER)

    verdict = None
    if not (comparative_ok and rows_ok):
        verdict = VERDICT_INVALID_VACUOUS
    elif not slots_ok:
        verdict = VERDICT_INVALID_SLOTS

    anti = {
        "rule": ("L1 pre-edit mean DC distance STRICTLY > L3's, L1 rows "
                 "changed >= 1, and every link's reserved slot satisfiable"),
        "l1_pre_edit_dc_distance_mean": float(l1["pre_edit_dc_distance_mean"]),
        "l3_pre_edit_dc_distance_mean": float(l3["pre_edit_dc_distance_mean"]),
        "comparative_ok": comparative_ok,
        "l1_rows_changed": int(l1["rows_changed"]),
        "l1_rows_changed_ok": rows_ok,
        "slots_ok": slots_ok,
        "per_link": {
            name: {
                "pre_edit_dc_distance_mean": float(
                    link_stats[name]["pre_edit_dc_distance_mean"]),
                "pre_edit_dc_distance_max": float(
                    link_stats[name]["pre_edit_dc_distance_max"]),
                "rows_changed": int(link_stats[name]["rows_changed"]),
                "rows_changed_fraction": float(
                    link_stats[name]["rows_changed_fraction"]),
                "slot_ok": bool(link_stats[name].get("slot_ok")),
                "slot_units_per_side": link_stats[name].get(
                    "slot_units_per_side"),
                "slot_available_per_side": link_stats[name].get(
                    "slot_available_per_side"),
            }
            for name in LINK_ORDER
        },
        "verdict": verdict,
    }
    return verdict, anti


def _empty_link_entry(name, stat):
    return {
        "name": name,
        "pre_edit_dc_distance_mean": float(stat["pre_edit_dc_distance_mean"]),
        "pre_edit_dc_distance_max": float(stat["pre_edit_dc_distance_max"]),
        "rows_changed": int(stat["rows_changed"]),
        "rows_changed_fraction": float(stat["rows_changed_fraction"]),
        "slot_units_per_side": stat.get("slot_units_per_side"),
        "slot_available_per_side": stat.get("slot_available_per_side"),
        "raw_slot_delta_mean": None,
        "raw_slot_delta_se": None,
        "return_side_delta_mean": None,
        "event_return_psnr_base": None,
        "event_return_psnr_edited": None,
        "event_return_psnr_delta": None,
        "certificate": {
            "stage_reached": "slot" if not stat.get("slot_ok") else None,
            "admitted": False,
            "pooled_mean": None,
            "pooled_se": None,
            "side_means": None,
        },
    }


def falsification_flow(sets_block, link_stats, measure_fn, packet_block=None):
    """Assemble the report. `measure_fn(name, stat) -> dict` is the ONLY
    place a reconstruction render happens, and it is called exclusively
    after the gate returns no verdict — that is the structural guarantee
    that a failed gate cannot be followed by a delta."""
    verdict, anti = gate_decision(link_stats)
    links = []
    for name in LINK_ORDER:
        entry = _empty_link_entry(name, link_stats[name])
        if verdict is None:
            entry.update(measure_fn(name, link_stats[name]))
        links.append(entry)
    report = {
        "schema": SCHEMA,
        "sets": sets_block,
        "anti_vacuity": anti,
        "links": links,
        "verdict": verdict or VERDICT_COMPLETED,
    }
    if packet_block is not None:
        report["packets"] = packet_block
    return report


def invalid_sets_report(sets_block, packet_block=None):
    """The refusal report when the authored sets are too small to ask
    the question at all (|D| < 8 or |R| < 4)."""
    report = {
        "schema": SCHEMA,
        "sets": sets_block,
        "anti_vacuity": None,
        "links": [],
        "verdict": VERDICT_INVALID_SETS,
    }
    if packet_block is not None:
        report["packets"] = packet_block
    return report


def packet_summary(packet_ids, sets):
    """Descriptive only: B1 packet membership of the authored sets. The
    falsification never uses packet ids to build a link."""
    ids = torch.as_tensor(packet_ids).reshape(-1).to(torch.long)
    valid = ids >= 0
    donor = sets["donor"]
    recipient = sets["recipient"]
    return {
        "rows": int(ids.numel()),
        "rows_with_packet": int(valid.sum()),
        "packets": int(len(set(int(p) for p in ids[valid].tolist()))),
        "donor_rows_with_packet": int(valid[donor].sum()) if donor.numel() else 0,
        "recipient_rows_with_packet": (int(valid[recipient].sum())
                                       if recipient.numel() else 0),
        "recipient_packet_ids": sorted(
            set(int(p) for p in ids[recipient][valid[recipient]].tolist())
        )[:32] if recipient.numel() else [],
    }


def mean_se(values):
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / max(n - 1, 1)
    return mean, (var / n) ** 0.5


def certificate_stage(pooled_mean, pooled_se, side_means):
    """The UNCHANGED CCR admission rule (§3.2 item 3): admit iff
    mean + 3*SE < 0 AND every per-side mean <= 0. Returns
    (stage_reached, admitted) with the FAILING stage named."""
    pooled_ok = (pooled_mean + ADMIT_K_SE * pooled_se) < 0
    side_ok = all(m <= 0 for m in side_means)
    if not pooled_ok:
        return "pooled-rule", False
    if not side_ok:
        return "side-rule", False
    return "admitted", True


def psnr_from_mse(mse):
    if mse <= 0:
        return float("inf")
    return float(20.0 * np.log10(1.0) - 10.0 * np.log10(mse))


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


# ---------------------------------------------------------------------------
# Render-dependent helpers (thin; every CUDA import is lazy)
# ---------------------------------------------------------------------------


def set_pointer(gaussians, pointer, mode=EDIT_MODE):
    gaussians._appearance_source_idx = pointer.to(gaussians._features_dc.device)
    gaussians._appearance_share_mode = mode


def event_return_psnr(gaussians, scene, pipe, background, gt_dir, obj_id):
    """Pooled PSNR over the event object's front-most mask at the RETURN
    frames, on the held-out cameras, under the model's CURRENT pointer
    state. Mask loading follows scripts/eval_lrv1_event.py exactly
    (source_path/gt_identity/cam%02d_f%03d.npy, front-most == the event
    object id). REPORT-ONLY: never an input to the certificate."""
    from gaussian_renderer import render  # lazy: CUDA JIT at import time

    sq = 0.0
    count = 0
    frames = 0
    with torch.no_grad():
        for item in scene.getTestCameras():
            cam = item[1] if isinstance(item, (tuple, list)) else item
            name = cam.image_name                       # cam<NN>_f<FFF>
            frame = int(name.split("_f")[1])
            if frame not in RETURN_FRAMES:
                continue
            cam_idx = int(name.split("_")[0][3:])
            mask_path = Path(gt_dir) / ("cam%02d_f%03d.npy" % (cam_idx, frame))
            if not mask_path.is_file():
                raise ContractError(
                    f"held-out view {name} has no ground-truth identity buffer; "
                    "the event metric would be computed over an incomplete region")
            obj = torch.from_numpy(np.load(mask_path)).cuda() == obj_id
            if not bool(obj.any()):
                continue
            gt = (item[0] if isinstance(item, (tuple, list))
                  else cam.original_image)[0:3].cuda().float().clamp(0.0, 1.0)
            pred = render(cam.cuda(), gaussians, pipe,
                          background)["render"].clamp(0.0, 1.0)
            sel = obj.unsqueeze(0).expand_as(pred)
            diff = (pred - gt)[sel]
            sq += float((diff ** 2).sum())
            count += int(diff.numel())
            frames += 1
    if count == 0:
        raise ContractError(
            "no held-out return-frame pixel carried the event object id")
    return psnr_from_mse(sq / count), count // 3, frames


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = ArgumentParser(description="Lane-4 B2 edit falsification (LRV3)")
    from arguments import ModelParams, OptimizationParams, PipelineParams

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start_checkpoint", required=True)
    parser.add_argument("--packet_state", default="")
    parser.add_argument("--oracle_region",
                        default=str(REPO_ROOT / "configs" / "lrv3"
                                    / "oracle_correct.json"))
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--gaussian_dim", type=int, default=4)
    parser.add_argument("--time_duration", nargs=2, type=float,
                        default=[0.0, 10.0])
    parser.add_argument("--num_pts", type=int, default=50_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exhaust_test", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    _merge_config(args, args.config)

    from elgs.trainer_hooks import build_reserved_pool
    from scene import Scene, GaussianModel
    from scene.appearance_edit import clear_appearance_edit
    from scripts.consolidate_packets import paired_delta, pick_confirmation_slot

    run_dir = os.environ.get("ADAGS_RUN_DIR", "").strip()
    if not str(getattr(args, "model_path", "") or "").strip():
        if not run_dir:
            raise ContractError("--model_path is required when ADAGS_RUN_DIR is unset")
        args.model_path = run_dir
    os.makedirs(args.model_path, exist_ok=True)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)

    region = json.loads(Path(args.oracle_region).read_text())["region"]
    if region.get("kind") != "sphere":
        raise ContractError(
            f"oracle region kind {region.get('kind')!r} is not a sphere; the "
            "authored row sets are defined on a sphere only")
    centre, radius = region["centre"], float(region["radius"])

    spec = json.loads((Path(args.source_path) / "event_spec.json").read_text())
    obj_id = int(spec["event_object"]["id"])
    gt_dir = Path(args.source_path) / "gt_identity"
    spec_return = tuple(sorted(int(f) for f in spec["return_frames"]))
    if spec_return != RETURN_FRAMES:
        raise ContractError(
            f"fixture return frames {spec_return} are not the frozen "
            f"{RETURN_FRAMES}; this tool is specified for the LRV3 fixture")

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
    if bool(getattr(opt, "elgs_enable", False)):
        raise ContractError(
            "this falsification is specified on the b1_packets substrate "
            "(elgs_enable false); an EL-GS cell would need its presence "
            "program restored before any render")

    n_rows = int(gaussians._xyz.shape[0])
    with torch.no_grad():
        pos_ep1 = gaussians.get_dynamic_xyz(DONOR_PROBE_T).detach().cpu()
        pos_ret = gaussians.get_dynamic_xyz(RECIPIENT_PROBE_T).detach().cpu()
        dc = gaussians._features_dc.detach().reshape(n_rows, -1).cpu()
        sup_lo, sup_hi = effective_support(
            gaussians._t.detach().cpu(), gaussians._scaling_t.detach().cpu())

    sets = build_row_sets(pos_ep1, pos_ret, sup_lo, sup_hi, dc, centre, radius)
    sets_block = sets_summary(sets)
    sets_block["rows"] = n_rows
    sets_block["oracle_region"] = {"centre": list(centre), "radius": radius,
                                   "source": str(args.oracle_region)}

    packet_block = None
    if str(args.packet_state or "").strip():
        state = torch.load(args.packet_state, map_location="cpu")
        packet_ids = state["packet_ids"]
        if int(packet_ids.shape[0]) != n_rows:
            raise ContractError(
                f"packet state has {int(packet_ids.shape[0])} rows, model has "
                f"{n_rows}")
        packet_block = packet_summary(packet_ids, sets)
        packet_block["schema_version"] = str(state.get("schema_version", ""))

    def _emit(report):
        Path(args.out_report).write_text(
            json.dumps(report, indent=1, sort_keys=True))
        print(json.dumps(report, indent=1, sort_keys=True), flush=True)
        return 0

    if not sets_are_sufficient(sets):
        return _emit(invalid_sets_report(sets_block, packet_block))
    assert_sets_disjoint(sets)

    # ---- links, row maps and pointers (no render)
    link_rows = {
        LINK_L1: (sets["recipient"], sets["donor"]),
        LINK_L2: (sets["recipient"], sets["wrong"]),
        LINK_L3: (sets["donor_b"], sets["donor_a"]),
    }
    link_stats = {}
    for name, (r_set, d_set) in link_rows.items():
        if r_set.numel() == 0 or d_set.numel() == 0:
            raise ContractError(
                f"link {name} has an empty side ({int(r_set.numel())} recipient "
                f"rows, {int(d_set.numel())} donor rows)")
        r_rows, d_rows = nearest_dc_row_map(dc, r_set, d_set)
        stat = dict(pre_edit_dc_stats(dc, r_rows, d_rows))
        stat["pointer"] = link_pointer(n_rows, r_rows, d_rows)
        link_stats[name] = stat

    # ---- reserved pool and per-link disjoint slots (no render)
    train_units = scene.getTrainCameras()
    cameras_meta = scene.train_cameras[1.0]
    if len(cameras_meta) != len(train_units):
        raise ContractError("camera metadata / dataset length mismatch")
    reserved = build_reserved_pool(cameras_meta)
    slot_pool = list(reserved)
    avail_d_all = [u for u, t in slot_pool
                   if WINDOW_EPISODE1[0] <= t <= WINDOW_EPISODE1[1]]
    avail_r_all = [u for u, t in slot_pool
                   if WINDOW_RETURN[0] <= t <= WINDOW_RETURN[1]]
    return_units = list(avail_r_all)

    # SLOT AMENDMENT (2026-08-20, decided by the primary BEFORE any
    # reconstruction render on this fixture; the worker's static
    # feasibility measurement showed the CCR-style disjoint-slot rule is
    # infeasible on LRV3 — only ~12 reserved units exist in WR, so L2/L3
    # would starve after L1's pick and the run would be INVALID_SLOTS by
    # construction). Here every link is measured against the SAME
    # identity base state, never an accumulating admitted state, so the
    # sequential-disjointness rationale of the CCR pass does not apply:
    # links SHARE reserved units, which additionally makes their deltas
    # directly comparable. Per-link windows: L1 and L2 measure donor-side
    # W1 / recipient-side WR; L3's rows all live in episode 1, so its
    # photometric null is measured on (W1, W1) — the frozen (W1, WR)
    # windows would have made it structurally near-null instead of a
    # genuine control. pick_confirmation_slot keeps the two sides of one
    # link internally disjoint.
    link_windows = {
        LINK_L1: (WINDOW_EPISODE1, WINDOW_RETURN),
        LINK_L2: (WINDOW_EPISODE1, WINDOW_RETURN),
        LINK_L3: (WINDOW_EPISODE1, WINDOW_EPISODE1),
    }
    for name in LINK_ORDER:                    # prespecified order
        stat = link_stats[name]
        (d_lo, d_hi), (r_lo, r_hi) = link_windows[name]
        slot = pick_confirmation_slot(
            slot_pool, set(),                  # fresh per link: shared units
            d_lo, d_hi, r_lo, r_hi)
        stat["slot_available_per_side"] = [
            len([u for u, t in slot_pool if d_lo <= t <= d_hi]),
            len([u for u, t in slot_pool if r_lo <= t <= r_hi]),
        ]
        stat["slot_windows"] = [[d_lo, d_hi], [r_lo, r_hi]]
        if slot is None:
            stat["slot"] = None
            stat["slot_ok"] = False
            stat["slot_units_per_side"] = None
            continue
        side_d, side_r = slot
        stat["slot"] = (side_d, side_r)
        stat["slot_ok"] = True
        stat["slot_units_per_side"] = [len(side_d), len(side_r)]

    # ---- the anti-vacuity gate, PRINTED before any reconstruction render
    gate_verdict, anti_preview = gate_decision(link_stats)
    print(json.dumps({"ccr_b2_falsification_gate": {
        "sets": sets_block, "anti_vacuity": anti_preview,
        "reserved_units": len(reserved),
        "reserved_units_in_W1": len(avail_d_all),
        "reserved_units_in_WR": len(avail_r_all),
    }}, sort_keys=True), flush=True)

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda")
    identity = torch.arange(n_rows, dtype=torch.long)
    base_psnr = {"value": None}

    def measure(name, stat):
        """Every render in this tool happens here, and this runs only
        when the gate returned no verdict (falsification_flow)."""
        pointer = stat["pointer"]
        side_d, side_r = stat["slot"]
        with torch.no_grad():
            dd = paired_delta([train_units[u] for u in side_d], gaussians, pipe,
                              background, identity, pointer, EDIT_MODE)
            dr = paired_delta([train_units[u] for u in side_r], gaussians, pipe,
                              background, identity, pointer, EDIT_MODE)
            pooled_mean, pooled_se = mean_se(dd + dr)
            side_means = [sum(dd) / len(dd), sum(dr) / len(dr)]
            stage, admitted = certificate_stage(pooled_mean, pooled_se,
                                                side_means)
            ret = paired_delta([train_units[u] for u in return_units], gaussians,
                               pipe, background, identity, pointer, EDIT_MODE)
            return_mean, _ = mean_se(ret)

            if base_psnr["value"] is None:
                set_pointer(gaussians, identity)
                base_psnr["value"] = event_return_psnr(
                    gaussians, scene, pipe, background, gt_dir, obj_id)
            set_pointer(gaussians, pointer)
            edited = event_return_psnr(gaussians, scene, pipe, background,
                                       gt_dir, obj_id)
            clear_appearance_edit(gaussians)
        base = base_psnr["value"]
        return {
            "raw_slot_delta_mean": pooled_mean,
            "raw_slot_delta_se": pooled_se,
            "return_side_delta_mean": return_mean,
            "return_side_units": len(return_units),
            "event_return_psnr_base": base[0],
            "event_return_psnr_edited": edited[0],
            "event_return_psnr_delta": edited[0] - base[0],
            "event_return_pixels": edited[1],
            "event_return_frames": edited[2],
            "certificate": {
                "stage_reached": stage,
                "admitted": bool(admitted),
                "pooled_mean": pooled_mean,
                "pooled_se": pooled_se,
                "side_means": side_means,
            },
        }

    report = falsification_flow(sets_block, link_stats, measure, packet_block)
    report["checkpoint"] = str(args.start_checkpoint)
    report["config"] = str(args.config)
    report["arm"] = EDIT_MODE
    report["evidence_bearing"] = False
    report["reserved_units"] = len(reserved)
    report["reserved_units_in_W1"] = len(avail_d_all)
    report["reserved_units_in_WR"] = len(avail_r_all)
    if gate_verdict is not None and report["verdict"] != gate_verdict:
        raise ContractError("gate verdict changed between preview and flow")
    return _emit(report)


if __name__ == "__main__":
    raise SystemExit(main())
