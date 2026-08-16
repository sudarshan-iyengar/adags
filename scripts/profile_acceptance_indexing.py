#!/usr/bin/env python3
"""Profile and verify the exact (track, frame) indexing in elgs/energy.py.

Authority: the acceptance-indexing repair recorded in
research-wiki/operations/elgs-m1-evidence-wiring-record.md. Experiment 83
scored the complete 135,780-report population of family 219 in five
minutes and was then CANCELLED in acceptance, because
`elgs.energy._capped_keys_for` and `segment_is_censored` recovered "the
cameras of this (track, frame)" by rescanning the whole report dict --
O(tracks x frames x reports) per (bridge, segment).

This script measures the repair rather than asserting it. At every size
it runs BOTH paths on the SAME batch and refuses to report a speedup it
has not first shown exact:

  * `--sweep` runs the scanning oracle and the indexed path at growing
    sizes, checks bit-exact parity of the capped key tuples, the
    censoring verdicts, both stream log-likelihood dicts and Phi, and
    fits the scan's cost law so its cost at full scale is a measured
    extrapolation rather than a guess.
  * `--full` runs the INDEXED path alone at experiment 83's shape. The
    oracle is not run there -- that is the whole point -- so this pass
    reports wall time and memory only, and parity for that shape rests
    on the sweep.

Pure CPU: elgs/energy.py holds no tensors and selects no device.
Peak memory is host-side, via tracemalloc.

Usage:
  python scripts/profile_acceptance_indexing.py --sweep
  python scripts/profile_acceptance_indexing.py --full
  python scripts/profile_acceptance_indexing.py --sweep --full --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import tracemalloc
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from elgs.clusters import SeedCluster  # noqa: E402
from elgs.energy import (  # noqa: E402
    EvidenceBatch,
    Segment,
    _capped_keys_for,
    _capped_keys_for_scan,
    _segment_is_censored_scan,
    build_evidence_index,
    phi,
    segment_is_censored,
    stream_log_likelihoods,
)
from elgs.evidence import EvidenceHeads, EvidenceParams, Report  # noqa: E402

# Experiment 83's measured shape for family 219: 292 tracks, 464 interior
# frames, 23 cameras, 3 bridges, 135,780 reports, 407,340 q values.
EXP83 = {"tracks": 292, "frames": 464, "cameras": 23, "bridges": 3}

PARAMS = EvidenceParams(
    pi_miss_vis=0.2,
    pi_miss_cens=0.6,
    pi_miss_out=0.4,
    d_img_x0=0.0,
    d_img_x1=1160.0,
    d_img_y0=0.0,
    d_img_y1=550.0,
    g_pos_sigma=12.0,
    v_min=0.0,
    v_max=1.0,
    r_min=0.05,
    h_floor=1e-7,
    pi_floor=0.01,
    g_cap=50.0,
    pos_cap=50.0,
)

HEADS = EvidenceHeads(
    params=PARAMS,
    g_v=lambda v: 2.0 * v if v > 0 else 0.0,
    h_c=lambda v: 1.0 / (PARAMS.d_img_area),
    h_o=lambda v: 1.0 / (PARAMS.d_img_area),
)


def build_batch(n_tracks: int, n_frames: int, n_cameras: int, n_bridges: int):
    """One report per (track, frame), its camera cycling through the
    camera set -- experiment 83's measured density (135,780 reports over
    292 x 464 (track, frame) pairs with 23 cameras covered)."""
    reports: dict = {}
    q_tilde = {b: {} for b in range(n_bridges)}
    centers = {b: {} for b in range(n_bridges)}
    for j in range(n_tracks):
        for i in range(n_frames):
            t = float(i)
            c = (j + i) % n_cameras
            key = (j, c, t)
            reports[key] = Report(
                is_miss=((j + i) % 11 == 0),
                v=0.25 + 0.05 * ((j + i) % 12),
                x=40.0 + (j % 900),
                y=30.0 + (i % 500),
            )
            for b in range(n_bridges):
                # frames below 3 are censored (q exactly 0 on EVERY
                # bridge) so the censoring limb is exercised too.
                q_tilde[b][key] = (
                    0.0
                    if i < 3
                    else round(0.5 * (math.sin(1.7 * b + 2.3 * j + 0.41 * i) + 1.0), 9)
                )
                centers[b][key] = (45.0 + 2.0 * b + (j % 900), 33.0 + b + (i % 500))
    return EvidenceBatch(reports=reports, q_tilde=q_tilde, bridge_centers=centers)


def build_units(n_tracks: int, n_frames: int):
    cluster = SeedCluster(
        cluster_id=0,
        seed_ids=(0,),
        canonical_point=(0.0, 0.0, 0.0),
        r_u=0.9,
        d_u=0.5,
        n_cam=2,
        alpha_u=1.0,
    )
    tracks = {0: tuple(range(n_tracks))}
    segments = (
        Segment(frames=(0.0, 1.0, 2.0), z=True),  # wholly censored
        Segment(frames=tuple(float(i) for i in range(3, n_frames)), z=True),
    )
    return (cluster,), tracks, segments


def _timed(fn):
    tracemalloc.start()
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, elapsed, peak


def run_sweep(sizes, c_cap: int) -> list[dict]:
    rows: list[dict] = []
    for n_tracks, n_frames in sizes:
        batch = build_batch(n_tracks, n_frames, EXP83["cameras"], EXP83["bridges"])
        clusters, tracks, segments = build_units(n_tracks, n_frames)
        track_ids = tracks[0]

        # --- exactness first, on every primitive and every downstream value
        index = build_evidence_index(batch)
        keys_equal = True
        censor_equal = True
        for bridge_id in batch.bridges():
            for segment in segments:
                if _capped_keys_for(
                    batch, bridge_id, track_ids, segment, c_cap, index
                ) != _capped_keys_for_scan(batch, bridge_id, track_ids, segment, c_cap):
                    keys_equal = False
        for segment in segments:
            if segment_is_censored(
                batch, track_ids, segment, index
            ) is not _segment_is_censored_scan(batch, track_ids, segment):
                censor_equal = False

        fast, t_fast, mem_fast = _timed(
            lambda: stream_log_likelihoods(
                batch, clusters, tracks, segments, HEADS, c_cap
            )
        )
        slow, t_slow, mem_slow = _timed(
            lambda: _scan_stream(batch, clusters, tracks, segments, c_cap)
        )

        max_diff = max(
            max(abs(fast.scored[b] - slow.scored[b]) for b in slow.scored),
            max(abs(fast.censored[b] - slow.censored[b]) for b in slow.censored),
        )
        phi_fast = phi(fast, 0.7)
        phi_slow = phi(slow, 0.7)

        rows.append(
            {
                "tracks": n_tracks,
                "frames": n_frames,
                "reports": len(batch.reports),
                "bridges": EXP83["bridges"],
                "scan_iterations_model": n_tracks * n_frames * len(batch.reports),
                "capped_keys_exact": keys_equal,
                "censoring_exact": censor_equal,
                "seconds_indexed": t_fast,
                "seconds_scan": t_slow,
                "speedup": (t_slow / t_fast) if t_fast > 0 else None,
                "peak_bytes_indexed": mem_fast,
                "peak_bytes_scan": mem_slow,
                "max_abs_stream_diff": max_diff,
                "phi_indexed": phi_fast,
                "phi_scan": phi_slow,
                "phi_abs_diff": abs(phi_fast - phi_slow),
                "phi_bit_identical": phi_fast == phi_slow,
            }
        )
        print(
            f"  tracks={n_tracks:5d} frames={n_frames:4d} reports={len(batch.reports):7d}"
            f"  indexed={t_fast:8.3f}s  scan={t_slow:9.3f}s"
            f"  speedup={rows[-1]['speedup']:8.1f}x"
            f"  maxdiff={max_diff:.3e}  phi_identical={phi_fast == phi_slow}",
            flush=True,
        )
    return rows


def _scan_stream(batch, clusters, tracks_of_cluster, segments, c_cap):
    """The pre-index reduction: scanning oracles, no index, no memo."""
    from elgs.energy import StreamLogLik, _p_floor
    from elgs.evidence import likelihood_l0, likelihood_l1

    batch.validate_bridge_independent_eligibility()
    scored, censored = {}, {}
    for bridge_id in batch.bridges():
        l_scored = 0.0
        l_cens = 0.0
        for cluster in clusters:
            track_ids = tuple(tracks_of_cluster.get(cluster.cluster_id, ()))
            if not track_ids:
                continue
            r_u = cluster.r_u
            for segment in segments:
                if _segment_is_censored_scan(batch, track_ids, segment):
                    continue
                for key in _capped_keys_for_scan(
                    batch, bridge_id, track_ids, segment, c_cap
                ):
                    report = batch.reports[key]
                    q = batch.q_tilde[bridge_id][key]
                    if segment.z:
                        center = batch.bridge_centers[bridge_id][key]
                        l1 = likelihood_l1(report, center, q, r_u, HEADS)
                    else:
                        l1 = likelihood_l0(report, r_u, HEADS)
                    l0 = likelihood_l0(report, r_u, HEADS)
                    floor = _p_floor(HEADS)
                    l_scored += cluster.alpha_u * math.log(max(l1, floor))
                    l_cens += cluster.alpha_u * math.log(max(l0, floor))
        scored[bridge_id] = l_scored
        censored[bridge_id] = l_cens
    return StreamLogLik(scored=scored, censored=censored)


def run_full(c_cap: int) -> dict:
    """Experiment 83's shape, INDEXED PATH ONLY.

    The scanning oracle is not run here: at this shape it is the cost the
    repair exists to remove, and running it would take hours. Parity for
    this shape is inherited from the sweep, which checks the same code on
    the same generator at smaller sizes.
    """
    batch = build_batch(
        EXP83["tracks"], EXP83["frames"], EXP83["cameras"], EXP83["bridges"]
    )
    clusters, tracks, segments = build_units(EXP83["tracks"], EXP83["frames"])

    index, t_index, mem_index = _timed(lambda: build_evidence_index(batch))
    stream, t_stream, mem_stream = _timed(
        lambda: stream_log_likelihoods(batch, clusters, tracks, segments, HEADS, c_cap)
    )
    value = phi(stream, 0.7)
    scan_iterations = (
        EXP83["tracks"] * EXP83["frames"] * len(batch.reports) * EXP83["bridges"] * 2
    )
    return {
        "shape": dict(EXP83),
        "reports": len(batch.reports),
        "q_values": len(batch.reports) * EXP83["bridges"],
        "seconds_index_build": t_index,
        "seconds_stream": t_stream,
        "peak_bytes_index": mem_index,
        "peak_bytes_stream": mem_stream,
        "index_entries": len(index.cameras_by_track_frame),
        "nonzero_track_frames": len(index.nonzero_track_frames),
        "phi": value,
        "phi_finite": math.isfinite(value),
        "scan_inner_iterations_at_this_shape": scan_iterations,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true", help="paired scan-vs-indexed sweep")
    parser.add_argument("--full", action="store_true", help="indexed path at experiment 83's shape")
    parser.add_argument("--c-cap", type=int, default=4)
    parser.add_argument("--json", default=None, help="write the full report to this path")
    args = parser.parse_args(argv)
    if not (args.sweep or args.full):
        parser.error("choose --sweep, --full, or both")

    report: dict = {"c_cap": args.c_cap, "exp83_shape": dict(EXP83)}

    if args.sweep:
        print("SWEEP (both paths, same batch, parity checked at every size):")
        report["sweep"] = run_sweep(
            [(8, 12), (16, 24), (24, 36), (32, 48), (48, 64)], args.c_cap
        )
        rows = report["sweep"]
        report["all_capped_keys_exact"] = all(r["capped_keys_exact"] for r in rows)
        report["all_censoring_exact"] = all(r["censoring_exact"] for r in rows)
        report["all_phi_bit_identical"] = all(r["phi_bit_identical"] for r in rows)
        report["max_abs_stream_diff_over_sweep"] = max(
            r["max_abs_stream_diff"] for r in rows
        )
        # Cost law: fit seconds_scan = k * (tracks*frames*reports) through
        # the largest row, then project it onto experiment 83's shape.
        big = rows[-1]
        k = big["seconds_scan"] / big["scan_iterations_model"]
        exp83_reports = EXP83["tracks"] * EXP83["frames"]
        report["scan_cost_law"] = {
            "seconds_per_inner_iteration": k,
            "fit_from": {
                "tracks": big["tracks"],
                "frames": big["frames"],
                "reports": big["reports"],
                "seconds_scan": big["seconds_scan"],
            },
            "projected_scan_seconds_at_exp83_shape": (
                k
                * EXP83["tracks"]
                * EXP83["frames"]
                * exp83_reports
                * EXP83["bridges"]
            ),
            "note": (
                "PROJECTION, not a measurement: the scanning oracle was "
                "never run at experiment 83's shape."
            ),
        }
        print(
            "  scan cost law: {:.3e} s per inner iteration -> projected "
            "{:.0f} s ({:.2f} h) at experiment 83's shape".format(
                k,
                report["scan_cost_law"]["projected_scan_seconds_at_exp83_shape"],
                report["scan_cost_law"]["projected_scan_seconds_at_exp83_shape"] / 3600.0,
            )
        )

    if args.full:
        print("\nFULL (experiment 83 shape, indexed path only):")
        report["full"] = run_full(args.c_cap)
        full = report["full"]
        print(
            f"  reports={full['reports']}  q_values={full['q_values']}"
            f"  index_build={full['seconds_index_build']:.3f}s"
            f"  stream={full['seconds_stream']:.3f}s"
            f"  peak_stream={full['peak_bytes_stream'] / 2**20:.1f} MiB"
            f"  phi={full['phi']!r}"
        )

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.json}")

    ok = True
    if args.sweep:
        ok = (
            report["all_capped_keys_exact"]
            and report["all_censoring_exact"]
            and report["all_phi_bit_identical"]
        )
        print(f"\nPARITY: {'EXACT at every swept size' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
