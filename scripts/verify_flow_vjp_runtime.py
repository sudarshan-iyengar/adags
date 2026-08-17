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
    out = scene.render(flows=flows)
    (out["flow"] * upstream).sum().backward()

    if flows.grad is None:
        raise RuntimeError("flow_2d received no gradient at all")
    observed = float(flows.grad.abs().max())
    if observed == 0.0:
        raise RuntimeError(
            "dL_dflows is EXACTLY zero: this is the pre-repair binary. The "
            "image does not contain the flow VJP patch."
        )

    alphas = scene.per_gaussian_alphas()
    expected, _ = flow_vjp(
        alphas, scene.unified_flows(), upstream, order=scene.depth_order()
    )
    expected = expected[: scene.count]
    scale = max(float(expected.abs().max()), 1.0)
    relative_error = float((flows.grad - expected).abs().max()) / scale
    if relative_error > 5e-3:
        raise RuntimeError(
            f"flow gradient is nonzero but disagrees with the independent "
            f"oracle by {relative_error:.3e} relative"
        )
    return {
        "max_abs_grad": observed,
        "max_relative_error_vs_oracle": relative_error,
    }


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

    report["environment"] = describe_environment()
    report["binary"] = describe_binary()
    report["functional_proof"] = prove_flow_gradient_is_live()
    report["tests"] = run_test_modules()

    if args.expect_cuda_sha256 and combined != args.expect_cuda_sha256:
        raise RuntimeError(
            "the run context's rasterizer sources do not match the expected "
            f"hash: {combined} != {args.expect_cuda_sha256}"
        )

    report["verified"] = True
    payload = json.dumps(report, indent=2, sort_keys=True)
    print("\n=== FLOW VJP RUNTIME VERIFICATION ===")
    print(payload)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
