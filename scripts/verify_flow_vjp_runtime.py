"""Verify the rendered-flow VJP on a real GPU, against a real image.

The workstation has no GPU and no working torch, so every CUDA test in
`tests/test_flow_backward_vjp.py` SKIPS there. A skip is not a pass. This
entrypoint exists so the flow gradient is executed on an actual DGX/V100
before any F/X supervision cell is allowed to run.

It answers three separate questions and refuses to conflate them:

  1. WHICH BINARY is loaded — its path, size and sha256, and that it is
     not the shared mutable worktree.
  2. WHICH SOURCE the run context carries — a CRLF-normalised content
     hash over the tracked rasterizer sources in the `git archive`
     context, comparable against a value computed on the workstation.
  3. WHETHER THE BINARY ACTUALLY HAS THE PATCH — a functional proof.

Point 3 needs saying plainly: nothing stamps a git commit into the
compiled `.so`, so provenance cannot be read off the binary. What CAN be
established is stronger for this purpose. The pre-repair extension
returned `dL_dflows` EXACTLY zero, because the only flow-gradient code in
`backward.cu` lived in an uninstantiated template. So a nonzero flow
gradient that also matches the independent pure-PyTorch oracle cannot be
produced by the old binary. Points 2 and 3 together close the loop; point
3 alone is what the F/X gate actually depends on.

Exits nonzero on any failure, and fails closed: an exception anywhere is
a failure, never a skip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: The one shared, mutable, historical worktree. Nothing may execute from
#: here, and no extension may be imported from underneath it.
FORBIDDEN_WORKTREE = "/apollo/users/sri/proj_adags/repo/adags"

#: The rasterizer sources whose content decides whether the flow VJP is
#: present. Hashed in this order, after CRLF normalisation.
RASTERIZER_SOURCES = (
    "diff-gaussian-rasterization/cuda_rasterizer/backward.cu",
    "diff-gaussian-rasterization/cuda_rasterizer/backward.h",
    "diff-gaussian-rasterization/cuda_rasterizer/forward.cu",
    "diff-gaussian-rasterization/cuda_rasterizer/forward.h",
    "diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu",
    "diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h",
    "diff-gaussian-rasterization/rasterize_points.cu",
)

#: Test modules executed here. Both are pure verification; neither writes
#: training state.
TEST_MODULES = (
    "tests.test_flow_backward_vjp",
    "tests.test_flow_resize_semantics",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-commit",
        help=(
            "the patched commit this run is supposed to be executing; "
            "required for verification, not for --print-cuda-sha256"
        ),
    )
    parser.add_argument(
        "--expect-cuda-sha256",
        help=(
            "combined CRLF-normalised hash of the rasterizer sources, as "
            "printed by --print-cuda-sha256 on the workstation; omitted "
            "means report-only, which is NOT a verification"
        ),
    )
    parser.add_argument(
        "--print-cuda-sha256",
        action="store_true",
        help="print the combined source hash and exit (workstation use)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="write the JSON report here as well as to stdout",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combined_cuda_sha256(root: Path) -> tuple[str, dict[str, str]]:
    """Hash the rasterizer sources, normalising line endings.

    The Windows checkout stores CRLF and the container LF, so a raw byte
    hash would differ across platforms for identical source. Normalising
    to LF makes the two directly comparable, which is the whole point of
    computing it on both sides.
    """
    per_file: dict[str, str] = {}
    combined = hashlib.sha256()
    for relpath in RASTERIZER_SOURCES:
        path = root / relpath
        if not path.is_file():
            raise FileNotFoundError(f"rasterizer source missing: {relpath}")
        normalised = path.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(normalised).hexdigest()
        per_file[relpath] = digest
        combined.update(relpath.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return combined.hexdigest(), per_file


def describe_binary() -> dict:
    """Locate the loaded extension and describe it. Never guesses."""
    import _adags_diff_gaussian_rasterization as ext

    origin = getattr(ext, "__file__", None)
    if origin is None:
        raise RuntimeError(
            "the extension has no __file__, so its provenance cannot be "
            "established; refusing to report it as verified"
        )
    path = Path(origin).resolve()
    record = {
        "module": ext.__name__,
        "file": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if FORBIDDEN_WORKTREE in str(path):
        raise RuntimeError(
            f"extension imported from the forbidden shared worktree: {path}"
        )
    return record


def describe_environment() -> dict:
    import torch

    record = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "det_experiment_id": os.environ.get("DET_EXPERIMENT_ID"),
        "det_trial_id": os.environ.get("DET_TRIAL_ID"),
    }
    if not record["cuda_available"]:
        raise RuntimeError(
            "no CUDA device: this entrypoint exists precisely to run the "
            "flow gradient on real hardware, so a CPU host is a failure"
        )
    record["device_name"] = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    record["device_capability"] = f"{major}.{minor}"
    return record


def prove_flow_gradient_is_live() -> dict:
    """The functional provenance proof: the old binary returns EXACTLY 0.

    Deliberately independent of the unittest suite, so that a collection
    error or an over-broad skip in the suite cannot let this pass
    silently.

    A zero gradient has SEVERAL possible causes and only one of them is
    "the image lacks the patch". If the scene rasterizes nothing, or the
    forward flow is zero, or the upstream gradient is zero, then dL_dflows
    is zero for reasons that say nothing about the binary. Each of those
    is therefore measured and reported BEFORE any conclusion is drawn, so
    that a scene-construction mistake can never be misattributed to the
    kernel. Every measurement is returned even on failure.
    """
    import torch

    from tests.ref_impls.flow_compositing_reference import flow_vjp
    from tests.test_flow_backward_vjp import TinyScene

    scene = TinyScene(
        means3D=[(0.0, 0.0, -0.5), (0.05, -0.05, 0.5)],
        opacities=[0.6, 0.7],
        flows=[(3.0, -2.0), (-1.5, 2.5)],
    )
    generator = torch.Generator(device="cuda").manual_seed(29)
    upstream = torch.rand(
        (2, scene.height, scene.width), generator=generator, device="cuda"
    )
    flows = scene.flows.clone().requires_grad_(True)
    opacities = scene.opacities.clone().requires_grad_(True)
    out = scene.render(flows=flows, opacities=opacities)

    measured = {
        "radii": [int(value) for value in out["radii"].reshape(-1).tolist()],
        "rasterized_primitives": int((out["radii"].reshape(-1) > 0).sum()),
        "forward_flow_absmax": float(out["flow"].abs().max()),
        "forward_alpha_absmax": float(out["alpha"].abs().max()),
        "forward_render_absmax": float(out["render"].abs().max()),
        "flow_requires_grad": bool(out["flow"].requires_grad),
        "flow_has_grad_fn": out["flow"].grad_fn is not None,
        "upstream_absmax": float(upstream.abs().max()),
    }

    # Ordered so the FIRST failure names the actual cause.
    if measured["rasterized_primitives"] == 0:
        raise RuntimeError(
            f"the probe scene rasterizes nothing (all radii zero): {measured}. "
            "This is a scene-construction fault in the test harness, NOT "
            "evidence about the compiled kernel."
        )
    if measured["forward_flow_absmax"] == 0.0:
        raise RuntimeError(
            f"the forward flow image is exactly zero: {measured}. A zero "
            "gradient is then expected and says nothing about the backward "
            "kernel. Fix the probe scene before drawing any conclusion."
        )
    if not measured["flow_has_grad_fn"]:
        raise RuntimeError(
            f"the flow output is not in the autograd graph: {measured}"
        )

    (out["flow"] * upstream).sum().backward()

    if flows.grad is None:
        raise RuntimeError(f"flow_2d received no gradient at all: {measured}")
    measured["max_abs_grad_flows"] = float(flows.grad.abs().max())
    measured["max_abs_grad_opacities"] = (
        float(opacities.grad.abs().max()) if opacities.grad is not None else None
    )

    if measured["max_abs_grad_flows"] == 0.0:
        raise RuntimeError(
            "dL_dflows is EXACTLY zero while the forward flow is nonzero and "
            f"the upstream gradient is nonzero: {measured}. With those "
            "confounders excluded, the compiled backward kernel is not "
            "writing the flow gradient."
        )

    alphas = scene.per_gaussian_alphas()
    expected, _ = flow_vjp(
        alphas, scene.unified_flows(), upstream, order=scene.depth_order()
    )
    expected = expected[: scene.count]
    scale = max(float(expected.abs().max()), 1.0)
    measured["max_relative_error_vs_oracle"] = (
        float((flows.grad - expected).abs().max()) / scale
    )
    if measured["max_relative_error_vs_oracle"] > 5e-3:
        raise RuntimeError(
            "flow gradient is nonzero but disagrees with the independent "
            f"oracle: {measured}"
        )
    return measured


def diagnose_gradient_routing() -> dict:
    """Where does the gradient actually GO?

    Experiment 133 established that the scene renders, the forward flow is
    nonzero, the upstream gradient is nonzero, and yet BOTH the flow-value
    and the flow-mediated opacity gradients are exactly zero. Every
    argument position from Python through to the kernel has been read and
    matches. So the next thing to measure is routing: does a flow loss
    deposit its gradient on some OTHER input (an off-by-one in the
    autograd Function's returned tuple would do exactly that), and does
    the same machinery work at all for a colour loss?

    The colour loss is the control. If colour moves opacity and flow moves
    nothing anywhere, the defect is specific to the flow path. If a flow
    loss moves the wrong tensor, it is an autograd position bug and the
    kernel is fine.
    """
    import math

    import torch

    import gaussian_renderer.diff_gaussian_rasterization as wrapper
    from tests.test_flow_backward_vjp import TinyScene

    # `num_buckets` decides the backward render grid: BACKWARD::render
    # launches ((B*32)+31)/32 == B blocks, so B == 0 launches NOTHING and,
    # with debug off, the invalid-configuration error is never surfaced.
    # Every render-stage gradient would then be silently zero. That is a
    # measurement, not a guess, so capture it at the boundary.
    captured: dict = {}
    original_module = wrapper._C

    class _Spy:
        """Delegates everything, and records the two counts on the way past.

        The module-level name is rebound rather than an attribute set on
        the extension module itself, because a C extension is not
        obliged to accept attribute assignment.
        """

        def __getattr__(self, name):
            return getattr(original_module, name)

        def rasterize_gaussians(self, *args):
            result = original_module.rasterize_gaussians(*args)
            captured["num_rendered"] = int(result[0])
            captured["num_buckets"] = int(result[1])
            return result

    def build(count):
        if count == 2:
            means3D = [(0.0, 0.0, -0.5), (0.05, -0.05, 0.5)]
            opacities = [0.6, 0.7]
            flows = [(3.0, -2.0), (-1.5, 2.5)]
        else:
            means3D, opacities, flows = [], [], []
            for index in range(count):
                z = -0.6 + 1.2 * index / max(count - 1, 1)
                means3D.append((0.0, 0.0, z))
                opacities.append(0.08)
                flows.append((2.0 * math.cos(0.7 * index), 2.0 * math.sin(0.7 * index)))
        return TinyScene(means3D=means3D, opacities=opacities, flows=flows)

    def run(count, loss_key):
        scene = build(count)
        leaves = {
            "flows": scene.flows.clone().requires_grad_(True),
            "opacities": scene.opacities.clone().requires_grad_(True),
            "means3D": scene.means3D.clone().requires_grad_(True),
        }
        means2D = torch.zeros_like(leaves["means3D"], requires_grad=True)
        captured.clear()
        out = scene.render(
            means3D=leaves["means3D"],
            opacities=leaves["opacities"],
            flows=leaves["flows"],
            means2D=means2D,
        )
        target = out[loss_key]
        (target * torch.ones_like(target)).sum().backward()
        grads = {
            name: (None if leaf.grad is None else float(leaf.grad.abs().max()))
            for name, leaf in leaves.items()
        }
        grads["means2D"] = (
            None if means2D.grad is None else float(means2D.grad.abs().max())
        )
        grads["_output_absmax"] = float(target.abs().max())
        grads["_rasterized"] = int((out["radii"].reshape(-1) > 0).sum())
        grads.update(captured)
        return grads

    report = {}
    wrapper._C = _Spy()
    try:
        for count in (2, 8, 72):
            for loss_key in ("render", "flow"):
                key = f"n{count}_loss_on_{loss_key}"
                try:
                    report[key] = run(count, loss_key)
                except Exception as exc:  # noqa: BLE001 - diagnostic only
                    report[key] = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        wrapper._C = original_module
    return report


def run_test_modules() -> dict:
    """Run the suites and treat a SKIP as a failure, because here it is."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        loader.loadTestsFromName(module) for module in TEST_MODULES
    )
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    record = {
        "modules": list(TEST_MODULES),
        "run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "skipped_detail": [
            {"test": str(test), "reason": reason} for test, reason in result.skipped
        ],
    }
    if result.failures or result.errors:
        raise RuntimeError(
            f"{len(result.failures)} failures and {len(result.errors)} errors"
        )
    # A skip here means the GPU guard did not see a GPU, which contradicts
    # describe_environment(). Reporting that as success is exactly the
    # false-green this whole entrypoint exists to prevent.
    if result.skipped:
        raise RuntimeError(
            f"{len(result.skipped)} tests SKIPPED on a machine that reports "
            "a CUDA device; a skip is not a pass"
        )
    return record


def main() -> int:
    args = parse_args()

    if args.print_cuda_sha256:
        combined, per_file = combined_cuda_sha256(REPO_ROOT)
        print(json.dumps({"combined": combined, "files": per_file}, indent=2))
        return 0

    if not args.expect_commit:
        raise SystemExit("--expect-commit is required when verifying")

    report: dict = {"expect_commit": args.expect_commit, "verified": False}

    combined, per_file = combined_cuda_sha256(REPO_ROOT)
    report["cuda_sources"] = {"combined_sha256": combined, "files": per_file}
    if args.expect_cuda_sha256:
        report["cuda_sources"]["expected"] = args.expect_cuda_sha256
        report["cuda_sources"]["matches"] = combined == args.expect_cuda_sha256
    else:
        report["cuda_sources"]["matches"] = None

    # Each stage records its own result. A stage that raises records the
    # failure and stops the run, but the report is STILL printed: losing
    # the diagnostics to a traceback is how a cheap misdiagnosis turns
    # into an expensive one.
    stages = (
        ("environment", describe_environment),
        ("binary", describe_binary),
        ("gradient_routing", diagnose_gradient_routing),
        ("functional_proof", prove_flow_gradient_is_live),
        ("tests", run_test_modules),
    )
    failure = None
    for name, stage in stages:
        try:
            report[name] = stage()
        except Exception as exc:  # noqa: BLE001 - reported, then re-raised
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}
            failure = f"{name}: {type(exc).__name__}: {exc}"
            break

    if failure is None and args.expect_cuda_sha256 and combined != args.expect_cuda_sha256:
        failure = (
            "the run context's rasterizer sources do not match the expected "
            f"hash: {combined} != {args.expect_cuda_sha256}"
        )

    report["verified"] = failure is None
    if failure is not None:
        report["failure"] = failure

    payload = json.dumps(report, indent=2, sort_keys=True)
    print("\n=== FLOW VJP RUNTIME VERIFICATION ===")
    print(payload)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")

    if failure is not None:
        print(f"\nFAILED: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
