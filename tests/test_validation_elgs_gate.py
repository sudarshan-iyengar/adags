"""`main.py --val` must render an EL-GS checkpoint with its own gate.

The defect this file pins (research-wiki/operations/
absence-fixture-lane-2026-09-10.md §6): `validation()` restored a
checkpoint and rendered it WITHOUT calling `setup_elgs`, so
`gaussians.elgs_runtime` stayed None, the renderer's `elgs_active` was
False (gaussian_renderer/__init__.py:224) and every row went through the
ordinary temporal marginal. Every `--val` metric of every EL-GS cell was
therefore rendered with the presence gate OFF.

Part (a) — unit, no GPU and no data. `validation()` is read out of
`main.py` and executed against stubs, the way
`tests/test_appearance_edit.py` executes the real property bodies: the
text under test is the repository's, only the surroundings are
synthetic. This keeps the check runnable on a workstation where
importing `main` is impossible (it pulls in the compiled CUDA
rasterizer), and it also makes "the runtime is live" observable without
rendering a pixel.

Part (b) — integration, Leonardo only. Given an EL-GS run directory and
the `f_box_profile.json` that `scripts/eval_n3v_gated.py --restore_state`
produced for it, run `main.py --val` and require the two profiles to
agree to four decimals. Skipped cleanly wherever the environment
variables are absent; it needs a GPU, the scene and a checkpoint, so it
must be run through Slurm.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402


def _function_source(path, name):
    """The source lines of a top-level function, from the real file."""
    source = Path(path).read_text(encoding="utf-8")
    lines = source.splitlines()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1:node.end_lineno]) + "\n"
    raise AssertionError("{} not found in {}".format(name, path))


class _StubTensor:
    def __init__(self, value):
        self._value = value

    def sum(self):
        return self._value


class _StubMask:
    """Stands in for `get_elgs_gated_row_mask()`."""

    def sum(self):
        return 7


class _StubTorch(types.SimpleNamespace):
    """Only the three `torch` names `validation()` uses."""

    float32 = "float32"

    def __init__(self, loaded):
        super().__init__()
        self._loaded = loaded

    def tensor(self, value, dtype=None, device=None):
        return _StubTensor(value)

    def load(self, path):
        return self._loaded


class _StubGaussians:
    def __init__(self, carries_elgs_state):
        self.calls = []
        self.elgs_runtime = None
        self._elgs_local_presence = False
        self._carries_elgs_state = carries_elgs_state
        self._pending_elgs_state = None

    def training_setup(self, opt):
        self.calls.append(("training_setup", opt))

    def restore(self, model_params, training_args):
        self.calls.append(("restore", training_args))
        if self._carries_elgs_state:
            self._pending_elgs_state = {"schema": "elgs-state-v1"}

    def get_elgs_gated_row_mask(self):
        return _StubMask()


class _StubExtractor:
    def __init__(self, *args, **kwargs):
        pass

    def reconstruction(self, cameras, out_dir, stage=None):
        return {}

    def export_image(self, out_dir, mode=None):
        return None


class _StubScene:
    def __init__(self, *args, **kwargs):
        self.motion_prior_cache = None

    def getTestCameras(self):
        return []


def _run_validation(*, elgs_enable, carries_elgs_state, setup_elgs):
    """Execute the repository's `validation()` against stubs.

    Returns `(gaussians, setup_calls, model_path)`.
    """
    gaussians = _StubGaussians(carries_elgs_state)
    setup_calls = []

    def _setup_elgs(model, scene, dataset, opt):
        setup_calls.append((model, scene, dataset, opt))
        return setup_elgs(model, scene, dataset, opt)

    hooks = types.ModuleType("elgs.trainer_hooks")
    hooks.setup_elgs = _setup_elgs

    namespace = {
        "torch": _StubTorch(({"model": "params"}, 12000)),
        "os": os,
        "json": json,
        "ContractError": ContractError,
        "GaussianModel": lambda *a, **k: gaussians,
        "Scene": _StubScene,
        "MotionPriorCache": lambda *a, **k: object(),
        "GaussianExtractor": _StubExtractor,
        "render": None,
        "configure_visibility_events_from_opt": lambda *a, **k: None,
        "collect_decomposition_diagnostics": lambda *a, **k: {},
        "evaluate_motion_prior_test_metrics": lambda *a, **k: {},
        "build_validation_summary_updates": lambda *a, **k: {},
        "log_wandb_metrics": lambda *a, **k: None,
    }
    exec(_function_source(REPO_ROOT / "main.py", "validation"),
         namespace)                                      # noqa: S102

    with tempfile.TemporaryDirectory() as model_path:
        dataset = types.SimpleNamespace(
            white_background=False, model_path=model_path,
            sh_degree=3, source_path=model_path,
        )
        opt = types.SimpleNamespace(elgs_enable=elgs_enable)
        pipe = types.SimpleNamespace(eval_shfs_4d=False)
        saved = sys.modules.get("elgs.trainer_hooks")
        sys.modules["elgs.trainer_hooks"] = hooks
        try:
            namespace["validation"](
                dataset, opt, pipe, "chkpnt12000.pth", 4, [0.0, 10.0],
                False, False, 100, 1.0,
            )
        finally:
            if saved is None:
                sys.modules.pop("elgs.trainer_hooks", None)
            else:
                sys.modules["elgs.trainer_hooks"] = saved
    return gaussians, setup_calls


def _attaching_setup_elgs(model, scene, dataset, opt):
    model.elgs_runtime = object()
    model._elgs_local_presence = True
    return types.SimpleNamespace(local_presence=True)


class ValidationGateTests(unittest.TestCase):
    def test_an_elgs_checkpoint_gets_its_runtime_attached(self):
        gaussians, setup_calls = _run_validation(
            elgs_enable=True, carries_elgs_state=True,
            setup_elgs=_attaching_setup_elgs,
        )
        self.assertEqual(len(setup_calls), 1)
        self.assertIsNotNone(gaussians.elgs_runtime)

    def test_the_optimizer_exists_before_setup_elgs_and_restore_sees_opt(self):
        """`setup_elgs`'s restore branch re-installs the `elgs_a`
        parameter group (elgs/trainer_hooks.py:334, :1044-1045), so
        `training_setup` must have run and `restore` must be handed the
        optimization params rather than None."""
        gaussians, _ = _run_validation(
            elgs_enable=True, carries_elgs_state=True,
            setup_elgs=_attaching_setup_elgs,
        )
        names = [name for name, _ in gaussians.calls]
        self.assertEqual(names, ["training_setup", "restore"])
        self.assertIsNotNone(gaussians.calls[1][1])

    def test_a_non_elgs_checkpoint_takes_the_unchanged_path(self):
        gaussians, setup_calls = _run_validation(
            elgs_enable=False, carries_elgs_state=False,
            setup_elgs=_attaching_setup_elgs,
        )
        self.assertEqual(setup_calls, [])
        self.assertIsNone(gaussians.elgs_runtime)
        self.assertEqual(gaussians.calls, [("restore", None)])

    def test_an_elgs_config_on_a_stateless_checkpoint_is_refused(self):
        """Fail closed rather than seed a program out of the config:
        that gate would not be the checkpoint's own."""
        with self.assertRaisesRegex(ContractError, "carries no.*elgs_state"):
            _run_validation(
                elgs_enable=True, carries_elgs_state=False,
                setup_elgs=_attaching_setup_elgs,
            )

    def test_a_runtime_that_does_not_attach_is_refused(self):
        def _no_runtime(model, scene, dataset, opt):
            return None

        with self.assertRaisesRegex(ContractError, "presence gate OFF"):
            _run_validation(
                elgs_enable=True, carries_elgs_state=True,
                setup_elgs=_no_runtime,
            )


ENV_RUN_DIR = "ADAGS_VAL_GATE_RUN_DIR"
ENV_CONFIG = "ADAGS_VAL_GATE_CONFIG"
ENV_SOURCE = "ADAGS_VAL_GATE_SOURCE_PATH"
ENV_MASKS = "ADAGS_VAL_GATE_MASKS"
ENV_CHECKPOINT = "ADAGS_VAL_GATE_CHECKPOINT"
ENV_PROFILE = "ADAGS_VAL_GATE_PROFILE"


def _integration_inputs():
    """The declared integration inputs, or None when unset.

    `ADAGS_VAL_GATE_RUN_DIR`   an EL-GS run directory (holds the checkpoint
                               and, by default, the reference profile)
    `ADAGS_VAL_GATE_CONFIG`    the cell's YAML (the one it trained with)
    `ADAGS_VAL_GATE_SOURCE_PATH`  the scene
    `ADAGS_VAL_GATE_MASKS`     the event-mask manifest the reference profile
                               was produced with
    `ADAGS_VAL_GATE_CHECKPOINT`  optional; defaults to <run_dir>/chkpnt12000.pth
    `ADAGS_VAL_GATE_PROFILE`     optional; defaults to
                               <run_dir>/gated_eval_12000/f_box_profile.json
    """
    required = [ENV_RUN_DIR, ENV_CONFIG, ENV_SOURCE, ENV_MASKS]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        return None
    run_dir = Path(os.environ[ENV_RUN_DIR])
    return {
        "run_dir": run_dir,
        "config": Path(os.environ[ENV_CONFIG]),
        "source_path": Path(os.environ[ENV_SOURCE]),
        "masks": Path(os.environ[ENV_MASKS]),
        "checkpoint": Path(os.environ.get(ENV_CHECKPOINT)
                           or run_dir / "chkpnt12000.pth"),
        "profile": Path(os.environ.get(ENV_PROFILE)
                        or run_dir / "gated_eval_12000" / "f_box_profile.json"),
    }


class ValidationGateProfileIntegrationTests(unittest.TestCase):
    """`--val` must reproduce the gate-ON profile of the same checkpoint.

    Needs a GPU, the scene and the checkpoint: run it on Leonardo through
    Slurm, never on a login node (AGENTS.md "Execution and Slurm").
    """

    def setUp(self):
        self.inputs = _integration_inputs()
        if self.inputs is None:
            raise unittest.SkipTest(
                "set {} to run the --val gate integration check".format(
                    ", ".join([ENV_RUN_DIR, ENV_CONFIG, ENV_SOURCE, ENV_MASKS])))

    def _profile(self, renders, gt, out):
        subprocess.run(
            [sys.executable,
             str(REPO_ROOT / "scripts" / "event_region_frame_profile.py"),
             "--renders", str(renders), "--gt", str(gt),
             "--masks", str(self.inputs["masks"]), "--out", str(out)],
            check=True, cwd=str(REPO_ROOT),
        )
        return json.loads(Path(out).read_text(encoding="utf-8"))

    def test_val_reproduces_the_gated_evaluator_profile(self):
        reference = json.loads(
            self.inputs["profile"].read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as work:
            model_path = Path(work) / "val"
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "main.py"), "--val",
                 "--config", str(self.inputs["config"]),
                 "--source_path", str(self.inputs["source_path"]),
                 "--model_path", str(model_path),
                 "--start_checkpoint", str(self.inputs["checkpoint"])],
                check=True, cwd=str(REPO_ROOT),
            )
            rendered = sorted(model_path.glob("test/ours_*/renders"))
            self.assertTrue(rendered, "--val wrote no renders")
            measured = self._profile(
                rendered[-1], rendered[-1].parent / "gt",
                Path(work) / "val_profile.json",
            )

        self.assertEqual(measured["n_frames"], reference["n_frames"])
        for got, want in zip(measured["whole_frame_psnr"],
                             reference["whole_frame_psnr"]):
            self.assertAlmostEqual(got, want, places=4)
        shared = set(measured["events"]) & set(reference["events"])
        self.assertTrue(shared, "the two profiles share no event")
        for name in sorted(shared):
            for got, want in zip(measured["events"][name]["per_frame_psnr"],
                                 reference["events"][name]["per_frame_psnr"]):
                if got is None or want is None:
                    self.assertIs(got, want)
                    continue
                self.assertAlmostEqual(got, want, places=4)


if __name__ == "__main__":
    unittest.main()
