#!/usr/bin/env python3
"""Payload headroom screen: how much could ANY payload possibly recover?

EXPLORATORY. Pure state analysis of a trained checkpoint — this tool
performs NO rendering, loads no rasterizer, and computes no
reconstruction delta. It answers one question, before any delta is
inspected: for the oracle-correct identity link, how far is each
candidate payload tensor of the RECIPIENT from its linked DONOR?

That distance is an upper bound on what redirecting the payload could
change. The DC payload was falsified on exactly this fixture with an
oracle-correct link that turned out to be nearly vacuous, so the screen
exists to make the vacuity visible in state space, cheaply, for every
candidate at once.

Row sets are byte-identical to the DC falsification: the same authored
oracle region, the same probe times, the same masks, the same three
links, all IMPORTED from `scripts.falsify_b2_edit` rather than copied.

  L1 oracle-correct   episode-1 rows of the event object -> its return rows
  L2 wrong-identity   descriptor-close rows OFF the object -> the same rows
  L3 same-identity    episode-1 rows split by row-index parity: the
     no-op            vacuity reference

Two spaces are reported wherever the stored parameter is a logit or a
log, because the RENDERED effect depends on the activated value while
the pointer redirects the stored one:

  _opacity     raw logit          and  sigmoid(_opacity)
  _scaling     log std            and  exp(_scaling)
  _scaling_t   log temporal scale and  exp(_scaling_t)
  _rotation    raw quaternion, the normalized quaternion, and the
               GEODESIC ANGLE in degrees (an L2 norm on quaternions is
               not an interpretable distance)

The activations are the MODEL's own (`opacity_activation`,
`scaling_activation`, `rotation_activation`), read off the restored
object — never re-implemented here.

Two row maps are reported for every tensor:

  PRIMARY  `nearest_dc_row_map` — the appearance-derived correspondence
           the recorded DC experiment used, so every number is directly
           comparable to it;
  NATIVE   the same nearest-map in the payload's OWN space, which
           separates "this payload has no headroom" from "the
           appearance-derived correspondence is the wrong
           correspondence for this payload".

Every numeric function in this module is importable and testable
without the renderer: the scene/checkpoint imports are lazy and live
inside `main`.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402
from scripts.falsify_b2_edit import (  # noqa: E402
    DEFAULT_PROTOCOL,
    LINK_L1,
    LINK_L2,
    LINK_L3,
    LINK_ORDER,
    T_SUPPORT_SIGMA,
    assert_sets_disjoint,
    build_row_sets,
    effective_support,
    load_event_spec,
    load_oracle_region,
    nearest_dc_row_map,
    nearest_row_map,
    probe_row_state,
    protocol_block,
    protocol_from_event_spec,
    recipient_prefilter_diagnostic,
    resolve_model_path,
    restore_model_and_scene,
    sets_are_sufficient,
    sets_summary,
    split_by_row_parity,
    validate_protocol,
)

#: Everything the screen reuses from the DC falsification, named here so
#: the provenance is explicit and so the transitively-used helpers
#: (`split_by_row_parity` inside `build_row_sets`, `effective_support`
#: inside `probe_row_state`) are visibly part of the shared surface
#: rather than dead imports.
_REUSED_FROM_FALSIFICATION = (
    build_row_sets, nearest_dc_row_map, nearest_row_map, split_by_row_parity,
    assert_sets_disjoint, effective_support, probe_row_state,
    restore_model_and_scene, resolve_model_path, load_oracle_region,
    load_event_spec, sets_summary, sets_are_sufficient,
    protocol_from_event_spec, validate_protocol, protocol_block,
    recipient_prefilter_diagnostic,
)

SCHEMA = "adags-payload-headroom-v1"

MAP_PRIMARY = "dc_primary"
MAP_NATIVE = "payload_native"
MAP_ORDER = (MAP_PRIMARY, MAP_NATIVE)

SPACE_RAW = "raw"
SPACE_ACTIVATED = "activated"
SPACE_GEODESIC = "geodesic_degrees"

METRIC_L2 = "l2"
METRIC_QUAT = "quaternion_degrees"

#: The FROZEN candidate list, in report order. `_features_dc` is first
#: because it is the known, falsified control every other row is read
#: against.
TENSOR_SPECS = (
    {
        "name": "_features_dc",
        "activation": None,
        "role": "the FALSIFIED DC payload — the control row",
        "raw_label": "_features_dc (degree-0 SH)",
    },
    {
        "name": "_opacity",
        "activation": "opacity_activation",
        "role": "candidate 2: opacity / support-state reuse",
        "raw_label": "_opacity (raw logit)",
        "activated_label": "sigmoid(_opacity)",
    },
    {
        "name": "_scaling_t",
        "activation": "scaling_activation",
        "role": "candidate 2: temporal support width",
        "raw_label": "_scaling_t (log temporal scale)",
        "activated_label": "exp(_scaling_t)",
    },
    {
        "name": "_t",
        "activation": None,
        "role": "candidate 2: temporal centre",
        "raw_label": "_t (temporal mean, model seconds)",
    },
    {
        "name": "_xyz",
        "activation": None,
        "role": "candidate 1: position (vacuous on LRV3 by construction)",
        "raw_label": "_xyz (world position at the temporal mean)",
    },
    {
        "name": "_scaling",
        "activation": "scaling_activation",
        "role": "candidate 1: spatial extent",
        "raw_label": "_scaling (log spatial std)",
        "activated_label": "exp(_scaling)",
    },
    {
        "name": "_rotation",
        "activation": "rotation_activation",
        "role": "candidate 1: orientation",
        "raw_label": "_rotation (raw quaternion)",
        "activated_label": "normalize(_rotation)",
        "geodesic": True,
    },
)


# ---------------------------------------------------------------------------
# Pure numerics — no torch.nn, no renderer, no GaussianModel
# ---------------------------------------------------------------------------


def flatten_rows(values):
    """(N, ...) -> CPU float64 (N, C)."""
    matrix = torch.as_tensor(values)
    matrix = matrix.detach().to("cpu", torch.float64)
    return matrix.reshape(matrix.shape[0], -1)


def quaternion_geodesic_degrees(q_a, q_b):
    """Geodesic angle between unit quaternions, in degrees.

    Uses |<a, b>| so that q and -q — the same rotation — are at distance
    zero, and clamps the dot product before arccos so numerical drift
    past 1 cannot produce NaN. Angle = 2*arccos(|<a, b>|), i.e. the
    rotation angle that carries one frame onto the other, in [0, 180].
    """
    a = torch.as_tensor(q_a).detach().to("cpu", torch.float64)
    b = torch.as_tensor(q_b).detach().to("cpu", torch.float64)
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    if a.shape != b.shape:
        raise ContractError("quaternion pairs disagree on shape")
    if a.shape[1] != 4:
        raise ContractError(
            "geodesic angle needs 4-component quaternions; got "
            "{} components".format(int(a.shape[1])))
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-12)
    dot = (a * b).sum(dim=1).abs().clamp(0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(dot))


def pair_distances(matrix, recipient_rows, source_rows, metric=METRIC_L2):
    """Per-row distance between each mapped (recipient, source) pair."""
    r_rows = torch.as_tensor(recipient_rows).reshape(-1).to(torch.long)
    s_rows = torch.as_tensor(source_rows).reshape(-1).to(torch.long)
    if r_rows.shape != s_rows.shape:
        raise ContractError("row map is not one-to-one with the recipient set")
    if metric == METRIC_QUAT:
        return quaternion_geodesic_degrees(matrix[r_rows], matrix[s_rows])
    if metric != METRIC_L2:
        raise ContractError("unknown distance metric: {!r}".format(metric))
    return (matrix[r_rows] - matrix[s_rows]).norm(dim=1)


def distance_summary(distances):
    """mean / median / max / p95 over the per-row distances.

    `median` and `p95` are both linear-interpolated quantiles, so the
    two are read on one definition. An empty set yields all-None rather
    than a silent zero.
    """
    dist = torch.as_tensor(distances).reshape(-1).to(torch.float64)
    if dist.numel() == 0:
        return {"pairs": 0, "mean": None, "median": None,
                "max": None, "p95": None}
    return {
        "pairs": int(dist.numel()),
        "mean": float(dist.mean()),
        "median": float(torch.quantile(dist, 0.5)),
        "max": float(dist.max()),
        "p95": float(torch.quantile(dist, 0.95)),
    }


def safe_ratio(numerator, denominator, denominator_label="L3 mean distance"):
    """Return (value, reason). `value` is None whenever a finite ratio
    cannot be formed — never inf, never NaN."""
    if numerator is None or denominator is None:
        return None, "a link summary is missing (empty mapped row set)"
    num = float(numerator)
    den = float(denominator)
    if not math.isfinite(num) or not math.isfinite(den):
        return None, "non-finite operand"
    if den == 0.0:
        return None, "{} is exactly zero".format(denominator_label)
    value = num / den
    if not math.isfinite(value):
        return None, "ratio is not finite"
    return value, None


def ratio_block(link_summaries):
    """The two ratios that made the DC result interpretable.

    headroom_ratio       = L1 mean / L3 mean  (DC measured 1.43)
    discrimination_ratio = L2 mean / L3 mean  (DC measured 21.7)
    """
    l1 = link_summaries.get(LINK_L1, {}).get("mean")
    l2 = link_summaries.get(LINK_L2, {}).get("mean")
    l3 = link_summaries.get(LINK_L3, {}).get("mean")
    headroom, headroom_reason = safe_ratio(l1, l3)
    discrimination, discrimination_reason = safe_ratio(l2, l3)
    return {
        "headroom_ratio": headroom,
        "headroom_ratio_reason": headroom_reason,
        "discrimination_ratio": discrimination,
        "discrimination_ratio_reason": discrimination_reason,
        "headroom_ratio_definition": "L1_mean / L3_mean",
        "discrimination_ratio_definition": "L2_mean / L3_mean",
    }


def link_row_pairs(sets):
    """The three authored links, exactly as the DC falsification builds
    them: (recipient_rows, source_rows) per link name."""
    return {
        LINK_L1: (sets["recipient"], sets["donor"]),
        LINK_L2: (sets["recipient"], sets["wrong"]),
        LINK_L3: (sets["donor_b"], sets["donor_a"]),
    }


def build_maps(dc_matrix, native_matrix, sets):
    """Both correspondences for all three links.

    PRIMARY is `nearest_dc_row_map` on `_features_dc`; NATIVE is the
    SAME generalized nearest-map on the payload's own matrix. Links with
    an empty side are recorded as None rather than raising, so one thin
    link cannot suppress the whole screen.
    """
    pairs = link_row_pairs(sets)
    maps = {MAP_PRIMARY: {}, MAP_NATIVE: {}}
    for name in LINK_ORDER:
        r_set, s_set = pairs[name]
        if int(r_set.numel()) == 0 or int(s_set.numel()) == 0:
            maps[MAP_PRIMARY][name] = None
            maps[MAP_NATIVE][name] = None
            continue
        maps[MAP_PRIMARY][name] = nearest_dc_row_map(dc_matrix, r_set, s_set)
        maps[MAP_NATIVE][name] = nearest_row_map(native_matrix, r_set, s_set)
    return maps


def maps_agree(map_a, map_b):
    """True when two row maps select the same source row everywhere."""
    if map_a is None or map_b is None:
        return None
    return bool(torch.equal(map_a[1], map_b[1]))


def analyse_spaces(spaces, maps):
    """The per-(space, map) distance summaries and ratios.

    `spaces` maps a space label -> (matrix, metric, label). Nothing here
    knows what a Gaussian is; the caller supplies already-activated
    matrices.
    """
    out = {}
    for space, (matrix, metric, label) in spaces.items():
        by_map = {}
        for map_name in MAP_ORDER:
            summaries = {}
            for link in LINK_ORDER:
                mapping = maps[map_name].get(link)
                if mapping is None:
                    summaries[link] = distance_summary(
                        torch.zeros(0, dtype=torch.float64))
                    continue
                r_rows, s_rows = mapping
                summaries[link] = distance_summary(
                    pair_distances(matrix, r_rows, s_rows, metric))
            entry = {"links": summaries}
            entry.update(ratio_block(summaries))
            by_map[map_name] = entry
        out[space] = {"label": label, "metric": metric, "maps": by_map}
    return out


def sha256_file(path):
    """(sha256_hex, bytes) for a file, or (None, None) when absent."""
    p = Path(path)
    if not p.is_file():
        return None, None
    hasher = hashlib.sha256()
    size = 0
    with open(str(p), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def format_table(report):
    """Compact human-readable view of the JSON report."""
    lines = []
    lines.append("payload headroom screen ({})".format(report["schema"]))
    lines.append("checkpoint: {}".format(report["checkpoint"]))
    lines.append("oracle region: {}".format(
        report["oracle_region"]["source"]))
    fixture = report.get("protocol", {}).get("fixture")
    if fixture:
        lines.append(
            "fixture: {scene}  return frames {ret}  W1 {w1}  WR {wr}  "
            "probes {probes}  scalars={src}".format(
                scene=fixture["scene_id"], ret=fixture["return_frames"],
                w1=["%.4f" % v for v in fixture["window_episode1"]],
                wr=["%.4f" % v for v in fixture["window_return"]],
                probes=["%.4f" % v for v in fixture["probe_times"]],
                src=fixture["scalars_source"]))
    counts = report["row_sets"]
    lines.append(
        "rows={rows}  |D|={donor_rows}  |R|={recipient_rows}  "
        "|Dw|={wrong_identity_rows}  spanning={spanning_rows_excluded}  "
        "D/R disjoint={disjoint}".format(
            rows=counts.get("rows"),
            donor_rows=counts.get("donor_rows"),
            recipient_rows=counts.get("recipient_rows"),
            wrong_identity_rows=counts.get("wrong_identity_rows"),
            spanning_rows_excluded=counts.get("spanning_rows_excluded"),
            disjoint=report["row_sets_disjoint"]))
    lines.append("")
    header = ("{:<26} {:<16} {:<14} {:>10} {:>10} {:>10} {:>9} {:>9}"
              .format("tensor / space", "map", "metric", "L1 mean",
                      "L2 mean", "L3 mean", "headroom", "discrim"))
    lines.append(header)
    lines.append("-" * len(header))

    def cell(value, digits=6):
        if value is None:
            return "null"
        return "{:.{d}g}".format(value, d=digits)

    for spec in TENSOR_SPECS:
        name = spec["name"]
        block = report["tensors"].get(name)
        if block is None:
            continue
        for space, space_block in block["spaces"].items():
            for map_name in MAP_ORDER:
                entry = space_block["maps"][map_name]
                links = entry["links"]
                lines.append(
                    "{:<26} {:<16} {:<14} {:>10} {:>10} {:>10} {:>9} {:>9}"
                    .format(
                        "{}/{}".format(name, space)[:26],
                        map_name,
                        space_block["metric"],
                        cell(links[LINK_L1]["mean"]),
                        cell(links[LINK_L2]["mean"]),
                        cell(links[LINK_L3]["mean"]),
                        cell(entry["headroom_ratio"], 4),
                        cell(entry["discrimination_ratio"], 4)))
    lines.append("")
    lines.append("headroom_ratio = L1_mean / L3_mean; "
                 "discrimination_ratio = L2_mean / L3_mean")
    lines.append("PRIMARY map = {} (appearance-derived, frozen); "
                 "NATIVE map = {}".format(MAP_PRIMARY, MAP_NATIVE))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main — the only place a checkpoint or a scene is touched
# ---------------------------------------------------------------------------


def build_parser():
    parser = ArgumentParser(
        description="Payload headroom screen on a trained LRV3 checkpoint "
                    "(state analysis only; renders nothing)")
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
    parser.add_argument("--hash_checkpoint", action="store_true",
                        help="also SHA-256 the checkpoint itself (reads the "
                             "whole file; off by default)")
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
    return parser, lp, op, pp


def main(argv=None):
    from scripts.falsify_b2_edit import _merge_config

    parser, lp, op, pp = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    _merge_config(args, args.config)

    resolve_model_path(args)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)

    centre, radius = load_oracle_region(args.oracle_region)
    # Windows, return frames, probes and support edges come from the
    # FIXTURE's own event_spec.json (`validate_protocol` refuses a probe
    # that would land outside the window it is meant to sample), so the
    # screen reads LRV4's single-instant return correctly instead of
    # inheriting LRV3's 9.6 probe.
    _spec, _obj_id, _gt_dir, protocol = load_event_spec(args.source_path)

    gaussians, scene, dataset, opt, pipe = restore_model_and_scene(
        args, lp, op, pp)

    probe = probe_row_state(gaussians, protocol)
    n_rows = probe["n_rows"]
    dc = probe["dc"]
    sets = build_row_sets(probe["pos_ep1"], probe["pos_ret"],
                          probe["sup_lo"], probe["sup_hi"], dc,
                          centre, radius, protocol)
    sets_block = sets_summary(sets, protocol)
    sets_block["rows"] = n_rows

    # REPORT-ONLY. The recipient pre-filter `lo` distribution, before the
    # `lower_min` cut. Selects nothing; its self-check asserts it reads the
    # selector's own pre-image. See lrv4-starved-fixture-result-2026-08-23 S5.
    prefilter = recipient_prefilter_diagnostic(
        probe["sup_lo"], probe["sup_hi"], probe["pos_ret"], centre, radius,
        protocol, recipient_rows=int(sets["recipient"].numel()))

    disjoint = False
    disjoint_error = None
    try:
        disjoint = bool(assert_sets_disjoint(sets))
    except ContractError as exc:                # recorded, never swallowed
        disjoint_error = str(exc)

    tensors = {}
    with torch.no_grad():
        for spec in TENSOR_SPECS:
            name = spec["name"]
            source = getattr(gaussians, name, None)
            if source is None or int(source.numel()) == 0:
                tensors[name] = {
                    "role": spec["role"],
                    "available": False,
                    "reason": "{} is empty on this model".format(name),
                    "spaces": {},
                }
                continue
            if int(source.shape[0]) != n_rows:
                raise ContractError(
                    "{} has {} rows, model has {}".format(
                        name, int(source.shape[0]), n_rows))

            raw = flatten_rows(source)
            spaces = {SPACE_RAW: (raw, METRIC_L2, spec["raw_label"])}
            activated = None
            if spec["activation"] is not None:
                activation = getattr(gaussians, spec["activation"], None)
                if activation is None:
                    raise ContractError(
                        "model exposes no {}".format(spec["activation"]))
                # The MODEL's own activation, applied to the model's own
                # tensor — never re-implemented here.
                activated = flatten_rows(
                    activation(source.detach().to(torch.float64)))
                spaces[SPACE_ACTIVATED] = (
                    activated, METRIC_L2, spec["activated_label"])
            if spec.get("geodesic"):
                spaces[SPACE_GEODESIC] = (
                    activated if activated is not None else raw,
                    METRIC_QUAT,
                    "geodesic angle between normalized quaternions (deg)")

            # A payload's NATIVE space is the activated one where an
            # activation exists — that is the space the renderer reads —
            # and the raw one otherwise.
            native_matrix = activated if activated is not None else raw
            native_space = (SPACE_ACTIVATED if activated is not None
                            else SPACE_RAW)
            maps = build_maps(dc, native_matrix, sets)
            block = {
                "role": spec["role"],
                "available": True,
                "shape": list(source.shape),
                "dtype": str(source.dtype),
                "native_map_space": native_space,
                "maps_agree": {
                    link: maps_agree(maps[MAP_PRIMARY][link],
                                     maps[MAP_NATIVE][link])
                    for link in LINK_ORDER
                },
                "spaces": analyse_spaces(spaces, maps),
            }
            tensors[name] = block

    config_hash, config_bytes = sha256_file(args.config)
    region_hash, region_bytes = sha256_file(args.oracle_region)
    checkpoint_hash, checkpoint_bytes = (None, None)
    if args.hash_checkpoint:
        checkpoint_hash, checkpoint_bytes = sha256_file(args.start_checkpoint)

    packet_block = None
    if str(args.packet_state or "").strip():
        state = torch.load(args.packet_state, map_location="cpu")
        packet_ids = state["packet_ids"]
        if int(packet_ids.shape[0]) != n_rows:
            raise ContractError(
                "packet state has {} rows, model has {}".format(
                    int(packet_ids.shape[0]), n_rows))
        from scripts.falsify_b2_edit import packet_summary

        packet_block = packet_summary(packet_ids, sets)
        packet_block["schema_version"] = str(state.get("schema_version", ""))

    report = {
        "schema": SCHEMA,
        "evidence_bearing": False,
        "renders_performed": 0,
        "checkpoint": str(args.start_checkpoint),
        "config": str(args.config),
        "oracle_region": {"centre": list(centre), "radius": radius,
                          "source": str(args.oracle_region)},
        "input_hashes": {
            "config": {"path": str(args.config), "sha256": config_hash,
                       "bytes": config_bytes},
            "oracle_region": {"path": str(args.oracle_region),
                              "sha256": region_hash, "bytes": region_bytes},
            "start_checkpoint": {
                "path": str(args.start_checkpoint),
                "sha256": checkpoint_hash, "bytes": checkpoint_bytes,
                "hashed": bool(args.hash_checkpoint)},
        },
        "protocol": {
            "links": list(LINK_ORDER),
            "maps": {
                MAP_PRIMARY: "nearest_dc_row_map on _features_dc (frozen, "
                             "identical to the DC falsification)",
                MAP_NATIVE: "nearest_row_map in the payload's own space",
            },
            "row_sets_source": "scripts.falsify_b2_edit.build_row_sets "
                               "(imported, not copied)",
            "fixture": protocol_block(protocol),
        },
        "row_sets": sets_block,
        "row_sets_disjoint": disjoint,
        "row_sets_disjoint_error": disjoint_error,
        "row_sets_sufficient": bool(sets_are_sufficient(sets)),
        "recipient_prefilter": prefilter,
        "tensors": tensors,
    }
    if packet_block is not None:
        report["packets"] = packet_block

    Path(args.out_report).write_text(
        json.dumps(report, indent=1, sort_keys=True))
    print(format_table(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
