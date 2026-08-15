#!/usr/bin/env python3
"""Benchmark the q refresh on the EXACT workload a real round evaluates.

Scope: measurement only. This entrypoint never trains, never writes a
checkpoint, never mutates the registry beyond what one round's own q
refresh already does, and never enters an acceptance decision. It exists
because experiment 78 measured ~10 q/s and the uncapped first round of
the same family needs 407,340 q values, and a claim that batching fixes
that has to be MEASURED on the same reports rather than extrapolated.

What it does, in order:

1. rebuilds the run exactly as `main.py` does (same config merge, same
   Scene, same GaussianModel, same `--start_checkpoint` restore, same
   `setup_elgs`, which itself calls `attach_evidence`);
2. asks the SAME smoke proposer the round asks which family to score, so
   the report population is the round's, not a benchmark's;
3. runs the q refresh once through the BATCHED probe surface and once
   through the SCALAR one -- the frozen oracle -- timing both and
   comparing every q value key-by-key;
4. optionally re-runs the batched refresh at several chunk sizes to
   demonstrate the chunk is a resource parameter, not a result;
5. compares the DOWNSTREAM quantities the decision actually consumes
   (per-family Phi and the candidate's `exact_deltas`) between the two;
6. writes one JSON report.

The scalar path is forced by hiding the batched methods from
`resolve_window_requests`'s capability check, so both passes go through
the identical production code with the identical probe object shape.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from depth_visibility.errors import ContractError  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402


def _merge_config(args, config_path: str):
    """`main.py`'s own recursive OmegaConf merge, verbatim in behaviour."""
    cfg = OmegaConf.load(config_path)

    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for inner in host[key].keys():
                recursive_merge(inner, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])

    for key in cfg.keys():
        recursive_merge(key, cfg)
    return args


def _probe_classes(chunk: int):
    """(batched-at-`chunk`, scalar-only) ModelProbe subclasses.

    `elgs.evidence_stack.resolve_window_requests` selects the batched
    resolver only when the probe exposes callable `project_batch` and
    `transmittance_batch`. Setting both to None on a subclass routes the
    identical probe object through the scalar oracle instead, so the two
    timed passes differ in nothing but which resolver runs.
    """
    from elgs.probe_model import ModelProbe

    batched = type(
        "ChunkedModelProbe",
        (ModelProbe,),
        {"__init__": _chunked_init(ModelProbe, chunk)},
    )
    scalar = type(
        "ScalarModelProbe",
        (ModelProbe,),
        {"project_batch": None, "transmittance_batch": None},
    )
    return batched, scalar


def _chunked_init(base, chunk: int):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("query_chunk", chunk)
        base.__init__(self, *args, **kwargs)

    return __init__


def _peak_memory_bytes(device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def _reset_peak_memory(device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_refresh(state, gaussians, scene, family_ids, probe_class, device):
    """One timed q refresh under `probe_class`, returning (evidence, seconds)."""
    import elgs.probe_model as probe_module
    import elgs.trainer_hooks as hooks

    original = probe_module.ModelProbe
    probe_module.ModelProbe = probe_class
    gc.collect()
    _reset_peak_memory(device)
    try:
        _sync(device)
        started = time.perf_counter()
        evidence = hooks._refresh_round_evidence(
            state, gaussians, scene, 0, family_ids=family_ids
        )
        _sync(device)
        elapsed = time.perf_counter() - started
    finally:
        probe_module.ModelProbe = original
    return evidence, elapsed, _peak_memory_bytes(device)


def _q_map(evidence) -> dict:
    return {
        f"{b}|{j}|{c}|{t}": value
        for (b, j, c, t), value in evidence.q_snapshot._values.items()
    }


def _downstream(state, evidence, proposals) -> dict:
    """Per-family Phi and each candidate's exact_deltas under `evidence`."""
    from elgs.evidence_stack import evidence_exact_delta, family_phi

    phis = {}
    for family_id in sorted({w.family_id for w in evidence.windows}):
        phis[str(family_id)] = float(
            family_phi(
                state.evidence, evidence, family_id,
                state.runtime.realization(family_id),
                frame_time=state.evidence_frame_time,
            )
        )
    deltas = {}
    for proposal in proposals:
        plan = proposal.plan
        deltas[str(plan.op) + ":" + ",".join(str(f) for f in plan.family_ids)] = float(
            evidence_exact_delta(
                state.evidence, evidence, plan,
                state.runtime.registry, state.config,
                frame_time=state.evidence_frame_time,
            )
        )
    return {"family_phi": phis, "exact_deltas": deltas}


def _run_parity_tests() -> dict:
    """The batched-vs-scalar parity classes, on THIS device.

    The CPU unit cell runs the same classes but skips their CUDA case
    for want of a device, so GPU parity is only ever demonstrated by a
    cell that has one -- which is this one.
    """
    import unittest

    import tests.test_elgs_evidence_wiring as suite_module

    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        loader.loadTestsFromTestCase(getattr(suite_module, name))
        for name in (
            "BatchedTransmittanceParityTests",
            "BatchedProjectionParityTests",
            "BatchedQParityTests",
            "BatchedWindowResolutionTests",
            "ProbeParityTests",
        )
    )
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    return {
        "ok": result.wasSuccessful(),
        "run": result.testsRun,
        "failures": [str(case) for case, _ in result.failures],
        "errors": [str(case) for case, _ in result.errors],
        "skipped": [str(case) for case, _ in result.skipped],
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _max_difference(left: dict, right: dict) -> dict:
    if set(left) != set(right):
        return {
            "keys_equal": False,
            "only_left": sorted(set(left) - set(right))[:8],
            "only_right": sorted(set(right) - set(left))[:8],
        }
    worst = 0.0
    worst_key = None
    non_finite = 0
    for key, value in left.items():
        other = right[key]
        if not (value == value and other == other):  # NaN
            non_finite += 1
            continue
        difference = abs(value - other)
        if difference > worst:
            worst, worst_key = difference, key
    return {
        "keys_equal": True,
        "n_keys": len(left),
        "max_abs_difference": worst,
        "argmax_key": worst_key,
        "non_finite": non_finite,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument(
        "--out", type=str, default=None,
        help="report path; defaults to $ADAGS_RUN_DIR/q_benchmark.json",
    )
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--val", action="store_true", default=False)
    parser.add_argument(
        "--chunk-sizes", type=str, default="64,256,1024",
        help="comma-separated batched query chunk sizes to time",
    )
    parser.add_argument(
        "--skip-scalar", action="store_true",
        help="skip the scalar oracle pass (only for a workload known too "
             "large to time scalar-side; the parity gate then cannot be met)",
    )
    parser.add_argument(
        "--parity-tests", action="store_true",
        help="run the batched-vs-scalar parity unit tests first, ON THIS "
             "DEVICE. The CPU cell skips the CUDA case by construction, so "
             "this is where GPU parity is actually demonstrated.",
    )
    args = parser.parse_args(sys.argv[1:])
    _merge_config(args, args.config)

    run_dir = os.environ.get("ADAGS_RUN_DIR", "").strip()
    if args.out is None:
        if not run_dir:
            raise ContractError("--out is required when ADAGS_RUN_DIR is unset")
        args.out = os.path.join(run_dir, "q_benchmark.json")
    if not str(getattr(args, "model_path", "") or "").strip():
        if not run_dir:
            raise ContractError("--model_path is required when ADAGS_RUN_DIR is unset")
        args.model_path = run_dir

    parity = None
    if args.parity_tests:
        parity = _run_parity_tests()
        print(f"[bench] parity tests: {parity}", flush=True)
        if not parity["ok"]:
            raise ContractError(f"parity tests failed on this device: {parity}")

    torch.manual_seed(args.seed)
    dataset = lp.extract(args)
    opt = op.extract(args)
    _ = pp.extract(args)

    if not bool(getattr(opt, "elgs_enable", False)):
        raise ContractError("benchmark_elgs_q needs an EL-GS lane (elgs_enable true)")

    os.makedirs(dataset.model_path, exist_ok=True)
    print(f"[bench] building scene from {dataset.source_path}", flush=True)
    gaussians = GaussianModel(
        dataset.sh_degree, gaussian_dim=args.gaussian_dim,
        time_duration=args.time_duration, rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d, sh_degree_t=0,
    )
    scene = Scene(
        dataset, gaussians, num_pts=args.num_pts,
        num_pts_ratio=args.num_pts_ratio, time_duration=args.time_duration,
    )
    scene.opt = opt
    gaussians.training_setup(opt)

    print(f"[bench] restoring {args.start_checkpoint}", flush=True)
    model_params, first_iter = torch.load(args.start_checkpoint)
    gaussians.restore(model_params, opt)
    device = gaussians.get_xyz.device
    print(f"[bench] restored at iteration {first_iter}; device {device}; "
          f"rows {int(gaussians.get_xyz.shape[0])}", flush=True)

    from elgs.trainer_hooks import _propose_smoke_candidates, setup_elgs

    state = setup_elgs(gaussians, scene, dataset, opt)
    if state is None or state.evidence is None:
        raise ContractError("no evidence context; this lane cannot be benchmarked")

    proposals = _propose_smoke_candidates(state, int(first_iter))
    family_ids = sorted({int(f) for p in proposals for f in p.plan.family_ids})
    if not family_ids:
        raise ContractError("the smoke proposer produced no candidate families")
    print(f"[bench] scoped families {family_ids}", flush=True)

    chunks = [int(v) for v in str(args.chunk_sizes).split(",") if v.strip()]
    report = {
        "schema": "elgs-q-benchmark-v1",
        "checkpoint": args.start_checkpoint,
        "checkpoint_iteration": int(first_iter),
        "config": args.config,
        "source_path": str(dataset.source_path),
        "device": str(device),
        "model_rows": int(gaussians.get_xyz.shape[0]),
        "scoped_families": family_ids,
        "smoke_max_reports_per_window": int(
            state.evidence.smoke_max_reports_per_window
        ),
        "parity_tests": parity,
        "passes": [],
    }

    # -- batched passes, one per chunk size -----------------------------
    baseline_q = None
    baseline_downstream = None
    for chunk in chunks:
        batched_class, _ = _probe_classes(chunk)
        evidence, seconds, peak = _run_refresh(
            state, gaussians, scene, family_ids, batched_class, device
        )
        values = _q_map(evidence)
        entry = {
            "mode": "batched",
            "chunk_size": chunk,
            "seconds": seconds,
            "q_values": evidence.n_q_values,
            "q_per_second": (evidence.n_q_values / seconds) if seconds > 0 else None,
            "windows": len(evidence.windows),
            "reports": dict(evidence.report_accounting),
            "peak_gpu_bytes": peak,
        }
        entry["downstream"] = _downstream(state, evidence, proposals)
        if baseline_q is None:
            baseline_q, baseline_downstream = values, entry["downstream"]
        else:
            entry["vs_first_chunk"] = _max_difference(values, baseline_q)
        report["passes"].append(entry)
        print(f"[bench] batched chunk={chunk}: {evidence.n_q_values} q in "
              f"{seconds:.3f}s = {entry['q_per_second']:.2f} q/s, "
              f"peak {peak / 2**20:.1f} MiB", flush=True)

    # -- scalar oracle pass ---------------------------------------------
    if not args.skip_scalar:
        _, scalar_class = _probe_classes(chunks[0])
        evidence, seconds, peak = _run_refresh(
            state, gaussians, scene, family_ids, scalar_class, device
        )
        scalar_values = _q_map(evidence)
        entry = {
            "mode": "scalar",
            "chunk_size": None,
            "seconds": seconds,
            "q_values": evidence.n_q_values,
            "q_per_second": (evidence.n_q_values / seconds) if seconds > 0 else None,
            "windows": len(evidence.windows),
            "reports": dict(evidence.report_accounting),
            "peak_gpu_bytes": peak,
            "downstream": _downstream(state, evidence, proposals),
        }
        entry["vs_batched"] = _max_difference(baseline_q, scalar_values)
        entry["downstream_vs_batched"] = {
            "family_phi": _max_difference(
                baseline_downstream["family_phi"], entry["downstream"]["family_phi"]
            ),
            "exact_deltas": _max_difference(
                baseline_downstream["exact_deltas"],
                entry["downstream"]["exact_deltas"],
            ),
        }
        report["passes"].append(entry)
        print(f"[bench] scalar: {evidence.n_q_values} q in {seconds:.3f}s = "
              f"{entry['q_per_second']:.2f} q/s", flush=True)

        fastest = max(
            p["q_per_second"] for p in report["passes"] if p["mode"] == "batched"
        )
        report["speedup_vs_scalar"] = fastest / entry["q_per_second"]
        report["max_q_difference"] = entry["vs_batched"].get("max_abs_difference")
        print(f"[bench] SPEEDUP {report['speedup_vs_scalar']:.2f}x   "
              f"max |dq| {report['max_q_difference']}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[bench] wrote {out_path}", flush=True)
    print("BENCHMARK_JSON_BEGIN", flush=True)
    print(json.dumps(report, sort_keys=True), flush=True)
    print("BENCHMARK_JSON_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
