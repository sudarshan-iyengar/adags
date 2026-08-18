"""The `elgs_a` optimizer group must survive per-point densify/prune.

`elgs/trainer_hooks.py::_refresh_logit_param_group` installs ONE tensor
PER EL-GS FAMILY in a single `elgs_a` param group. Those tensors are
per-family interval logits owned by `ElgsRuntime._logits`; their leading
dimension is dim(a) = 2K + 1 - latches, which has nothing to do with the
number of Gaussians.

Both per-point optimizer mutation paths in `scene/gaussian_model.py`
walk `optimizer.param_groups` and used to assume every non-skiplisted
group is a single per-point tensor, and `elgs_a` was missing from both
skip lists:

  * `cat_tensors_to_optimizer` asserted `len(group["params"]) == 1`
    BEFORE looking the group up in `tensors_dict`, so two or more
    families raised `AssertionError: Group elgs_a has more than one
    param` at the first densification;
  * `_prune_optimizer` had no guard at all and mask-indexed
    `group["params"][0]`, which raises `IndexError` for the real
    dim(a) != N shapes and silently rewrites family 0's tensor (losing
    its identity with `ElgsRuntime._logits`) whenever the two lengths
    happen to coincide.

Every test here uses the REAL trainer setup path and the REAL optimizer
methods; the integration test crosses a REAL `densify_and_prune`.
"""

import unittest

import torch

from depth_visibility.errors import ContractError
from elgs.trainer_hooks import setup_elgs

HAVE_MODEL = False
MODEL_IMPORT_ERROR = ""
try:
    from argparse import ArgumentParser

    import numpy as np

    from arguments import OptimizationParams
    from scene.gaussian_model import NON_PER_POINT_PARAM_GROUPS, GaussianModel
    from utils.graphics_utils import BasicPointCloud

    HAVE_MODEL = True
except Exception as exc:  # pragma: no cover - environment dependent
    MODEL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

GPU_REASON = (
    "needs a CUDA device and the compiled simple-knn extension"
    + (f" ({MODEL_IMPORT_ERROR})" if MODEL_IMPORT_ERROR else "")
)
HAVE_CUDA = torch.cuda.is_available()

POINTS = 48


class _StubCamera:
    def __init__(self, timestamp, name):
        self.timestamp = timestamp
        self.image_name = name


class _CountingDataset:
    def __init__(self, cameras):
        self._cameras = cameras

    def __len__(self):
        return len(self._cameras)

    def __getitem__(self, index):
        return self._cameras[index]


class _StubScene:
    def __init__(self, dataset):
        self._dataset = dataset

    def getTrainCameras(self):
        return self._dataset


def _options():
    parser = ArgumentParser(add_help=False)
    group = OptimizationParams(parser)
    opt = group.extract(parser.parse_args([]))
    opt.motion_model = "lora"
    opt.motion_scaffold_enable = False
    opt.elgs_enable = True
    opt.elgs_smoke_schedule = True
    opt.elgs_a_lr = 0.01
    opt.elgs_k_se = 1.0
    opt.elgs_lambda_u = 1.0
    opt.elgs_candidate_cap = 2
    opt.elgs_confirmation_samples = 6
    opt.elgs_tracks_dir = ""
    opt.densify_until_num_points = 100_000
    return opt


def _build():
    """A real 4D GaussianModel with a real, seeded EL-GS state.

    Voxel-grid seeding over a random cloud yields many families, so the
    `elgs_a` group genuinely holds more than one tensor -- the condition
    that used to trip the assert.
    """
    opt = _options()
    generator = np.random.default_rng(11)
    points = generator.uniform(-0.4, 0.4, size=(POINTS, 3)).astype(np.float32)
    colors = generator.uniform(0.2, 0.8, size=(POINTS, 3)).astype(np.float32)
    cloud = BasicPointCloud(
        points=points,
        colors=colors,
        normals=np.zeros_like(points),
        time=np.zeros((POINTS, 1), dtype=np.float32),
    )
    gaussians = GaussianModel(
        0, gaussian_dim=4, time_duration=[-0.5, 0.5], rot_4d=False, sh_degree_t=0
    )
    gaussians.create_from_pcd(cloud, 1.0)
    gaussians.training_setup(opt)

    cameras = [
        _StubCamera(timestamp=0.25 * (i // 4), name=f"cam{i % 4}_t{i // 4:03d}")
        for i in range(240)
    ]
    scene = _StubScene(_CountingDataset(cameras))
    state = setup_elgs(gaussians, scene, dataset=None, opt=opt)
    _prime_optimizer_state(gaussians)
    return gaussians, state, opt


def _prime_optimizer_state(gaussians):
    """One real Adam step so every param owns exp_avg / exp_avg_sq.

    Without stepping, `optimizer.state` is empty and the paths under
    test take their stateless branch, which would hide a moment reset.
    """
    for group in gaussians.optimizer.param_groups:
        for param in group["params"]:
            param.grad = torch.ones_like(param)
    gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)


def _logit_group(gaussians, state):
    return gaussians.optimizer.param_groups[state.logit_group_index]


def _snapshot_logits(gaussians, state):
    group = _logit_group(gaussians, state)
    snapshot = []
    for param in group["params"]:
        stored = gaussians.optimizer.state[param]
        snapshot.append(
            {
                "param": param,
                "value": param.detach().clone(),
                "exp_avg": stored["exp_avg"],
                "exp_avg_value": stored["exp_avg"].detach().clone(),
                "exp_avg_sq": stored["exp_avg_sq"],
                "exp_avg_sq_value": stored["exp_avg_sq"].detach().clone(),
            }
        )
    return snapshot


class _LogitInvarianceMixin:
    def assert_logits_untouched(self, gaussians, state, snapshot):
        """Identity AND optimizer state, not merely equal values.

        `ElgsRuntime._logits` holds the same objects, so a replaced
        tensor desyncs the runtime from the optimizer even when the
        numbers match.
        """
        group = _logit_group(gaussians, state)
        self.assertEqual(len(group["params"]), len(snapshot))
        runtime_params = list(state.runtime.logit_parameters().values())
        self.assertEqual(
            [id(p) for p in group["params"]], [id(p) for p in runtime_params],
            "family-to-tensor correspondence with ElgsRuntime._logits broke",
        )
        for before, param in zip(snapshot, group["params"]):
            self.assertIs(param, before["param"], "an a-logit tensor was replaced")
            self.assertTrue(torch.equal(param.detach(), before["value"]))
            self.assertIn(param, gaussians.optimizer.state)
            stored = gaussians.optimizer.state[param]
            self.assertIs(stored["exp_avg"], before["exp_avg"], "exp_avg was rebuilt")
            self.assertIs(
                stored["exp_avg_sq"], before["exp_avg_sq"], "exp_avg_sq was rebuilt"
            )
            self.assertTrue(torch.equal(stored["exp_avg"], before["exp_avg_value"]))
            self.assertTrue(
                torch.equal(stored["exp_avg_sq"], before["exp_avg_sq_value"])
            )


@unittest.skipUnless(HAVE_MODEL and HAVE_CUDA, GPU_REASON)
class LogitGroupShapeTests(unittest.TestCase):
    def test_seeding_installs_more_than_one_family_tensor(self):
        """The precondition the two defects needed: >= 2 tensors whose
        leading dimension is NOT the point count."""
        gaussians, state, _ = _build()
        group = _logit_group(gaussians, state)
        self.assertGreaterEqual(len(group["params"]), 2)
        self.assertEqual(group["name"], "elgs_a")
        rows = int(gaussians.get_xyz.shape[0])
        for param in group["params"]:
            self.assertNotEqual(int(param.shape[0]), rows)


@unittest.skipUnless(HAVE_MODEL and HAVE_CUDA, GPU_REASON)
class CatTensorsToOptimizerTests(unittest.TestCase, _LogitInvarianceMixin):
    def test_concatenation_grows_per_point_groups_and_skips_elgs_a(self):
        gaussians, state, _ = _build()
        snapshot = _snapshot_logits(gaussians, state)
        rows = int(gaussians.get_xyz.shape[0])
        extra = 5
        extension = {
            "xyz": torch.zeros((extra, 3), device="cuda"),
            "f_dc": torch.zeros((extra, 1, 3), device="cuda"),
            "f_rest": torch.zeros(
                (extra,) + tuple(gaussians._features_rest.shape[1:]), device="cuda"
            ),
            "opacity": torch.zeros((extra, 1), device="cuda"),
            "scaling": torch.zeros((extra, 3), device="cuda"),
            "rotation": torch.zeros((extra, 4), device="cuda"),
            "t": torch.zeros((extra, 1), device="cuda"),
            "scaling_t": torch.zeros((extra, 1), device="cuda"),
        }
        grown = gaussians.cat_tensors_to_optimizer(extension)

        self.assertNotIn("elgs_a", grown)
        for name, tensor in extension.items():
            self.assertEqual(int(grown[name].shape[0]), rows + extra)
        self.assert_logits_untouched(gaussians, state, snapshot)

    def test_concatenated_group_keeps_its_moments_and_zero_fills_new_rows(self):
        """The per-point regression guard: growing a group must extend
        its Adam moments with zeros, not reset them."""
        gaussians, state, _ = _build()
        xyz_group = next(
            g for g in gaussians.optimizer.param_groups if g["name"] == "xyz"
        )
        before = gaussians.optimizer.state[xyz_group["params"][0]]["exp_avg"].clone()
        rows = int(before.shape[0])
        extra = 3
        grown = gaussians.cat_tensors_to_optimizer(
            {"xyz": torch.zeros((extra, 3), device="cuda")}
        )
        after = gaussians.optimizer.state[grown["xyz"]]["exp_avg"]
        self.assertEqual(int(after.shape[0]), rows + extra)
        self.assertTrue(torch.equal(after[:rows], before))
        self.assertEqual(float(after[rows:].abs().sum()), 0.0)


@unittest.skipUnless(HAVE_MODEL and HAVE_CUDA, GPU_REASON)
class PruneOptimizerTests(unittest.TestCase, _LogitInvarianceMixin):
    def test_pruning_slices_per_point_groups_and_skips_elgs_a(self):
        gaussians, state, _ = _build()
        snapshot = _snapshot_logits(gaussians, state)
        rows = int(gaussians.get_xyz.shape[0])
        mask = torch.ones(rows, dtype=torch.bool, device="cuda")
        mask[::5] = False
        kept = int(mask.sum())

        pruned = gaussians._prune_optimizer(mask)

        self.assertNotIn("elgs_a", pruned)
        for name in ("xyz", "f_dc", "opacity", "scaling", "rotation", "t"):
            self.assertEqual(int(pruned[name].shape[0]), kept)
        self.assert_logits_untouched(gaussians, state, snapshot)

    def test_pruned_group_keeps_the_surviving_rows_of_its_moments(self):
        gaussians, state, _ = _build()
        xyz_group = next(
            g for g in gaussians.optimizer.param_groups if g["name"] == "xyz"
        )
        before = gaussians.optimizer.state[xyz_group["params"][0]]["exp_avg"].clone()
        rows = int(before.shape[0])
        mask = torch.ones(rows, dtype=torch.bool, device="cuda")
        mask[::5] = False

        pruned = gaussians._prune_optimizer(mask)

        after = gaussians.optimizer.state[pruned["xyz"]]["exp_avg"]
        self.assertTrue(torch.equal(after, before[mask]))

    def test_an_unlisted_non_per_point_group_fails_closed(self):
        """`_prune_optimizer` has no per-name work list to fall through
        on, so the next non-per-point group someone adds must raise
        instead of being mask-indexed and silently rewritten."""
        gaussians, state, _ = _build()
        rows = int(gaussians.get_xyz.shape[0])
        gaussians.optimizer.add_param_group(
            {
                "params": [torch.nn.Parameter(torch.zeros(3, device="cuda"))],
                "lr": 0.0,
                "name": "future_global",
            }
        )
        mask = torch.ones(rows, dtype=torch.bool, device="cuda")
        mask[0] = False
        with self.assertRaises(ContractError):
            gaussians._prune_optimizer(mask)

    def test_static_branch_prunes_without_tripping_the_guard(self):
        """The guard sits AFTER the static/dynamic branch filter, so its
        correctness is branch-dependent -- and the static branch is exactly
        where a false positive would break an existing non-EL-GS lane."""

        gaussians, state, _ = _build()
        snapshot = _snapshot_logits(gaussians, state)
        # dynamic2static returns immediately on an empty staticness score, so
        # the fixture must supply one or this test silently measures nothing.
        rows = int(gaussians.get_xyz.shape[0])
        gaussians._staticness_score = torch.ones(
            (rows, 1), device=gaussians.get_xyz.device
        )
        gaussians.t_gradient_accum = torch.zeros(
            (rows, 1), device=gaussians.get_xyz.device
        )
        gaussians.denom = torch.ones((rows, 1), device=gaussians.get_xyz.device)
        gaussians.dynamic2static(0.0)
        static_rows = int(gaussians.static_xyz.shape[0])
        self.assertGreater(
            static_rows, 0, "the static prune branch was never reached"
        )
        mask = torch.zeros(
            static_rows, dtype=torch.bool, device=gaussians.static_xyz.device
        )
        mask[0] = True
        gaussians.prune_static_points(mask)
        self.assertEqual(int(gaussians.static_xyz.shape[0]), static_rows - 1)
        self.assert_logits_untouched(gaussians, state, snapshot)


@unittest.skipUnless(HAVE_MODEL and HAVE_CUDA, GPU_REASON)
class ReplaceTensorToOptimizerTests(unittest.TestCase, _LogitInvarianceMixin):
    def test_replacement_matches_by_name_and_never_reaches_elgs_a(self):
        gaussians, state, _ = _build()
        snapshot = _snapshot_logits(gaussians, state)
        rows = int(gaussians.get_xyz.shape[0])

        replaced = gaussians.replace_tensor_to_optimizer(
            torch.full((rows, 1), -3.0, device="cuda"), "opacity"
        )

        self.assertEqual(sorted(replaced), ["opacity"])
        self.assertEqual(float(replaced["opacity"].detach().mean()), -3.0)
        self.assert_logits_untouched(gaussians, state, snapshot)

    def test_reset_opacity_leaves_the_a_logits_alone(self):
        """The production caller of `replace_tensor_to_optimizer`."""
        gaussians, state, _ = _build()
        snapshot = _snapshot_logits(gaussians, state)
        gaussians.reset_opacity()
        self.assert_logits_untouched(gaussians, state, snapshot)


@unittest.skipUnless(HAVE_MODEL and HAVE_CUDA, GPU_REASON)
class DensifyAndPruneIntegrationTests(unittest.TestCase, _LogitInvarianceMixin):
    """The real thing: the frozen capacity policy fires its first
    densification at iteration 600, and an EL-GS run used to die there."""

    def test_first_densification_survives_with_families_live(self):
        gaussians, state, opt = _build()
        snapshot = _snapshot_logits(gaussians, state)
        families_before = len(state.runtime.registry.active_ids())
        rows_before = int(gaussians.get_xyz.shape[0])

        # Make densification and pruning both actually fire: large
        # accumulated gradients select rows for clone/split, and a few
        # near-zero opacities select rows for pruning.
        with torch.no_grad():
            gaussians.xyz_gradient_accum.fill_(1.0)
            gaussians.denom.fill_(1.0)
            gaussians.max_radii2D.fill_(1.0)
            gaussians._opacity[::7] = -20.0

        gaussians.densify_and_prune(
            max_grad=opt.densify_grad_threshold,
            min_opacity=opt.thresh_opa_prune,
            extent=1.0,
            max_screen_size=None,
            max_grad_t=opt.densify_grad_t_threshold,
            iteration=600,
            max_total_points=opt.densify_until_num_points,
        )

        rows_after = int(gaussians.get_xyz.shape[0])
        self.assertNotEqual(rows_after, rows_before, "densification did not fire")

        # The a-logits are untouched: same objects, same Adam moments.
        self.assert_logits_untouched(gaussians, state, snapshot)
        self.assertEqual(len(state.runtime.registry.active_ids()), families_before)

        # Every per-point group agrees with the new row count, and the
        # EL-GS family column stayed row-aligned with the bank.
        for group in gaussians.optimizer.param_groups:
            if group["name"] in NON_PER_POINT_PARAM_GROUPS:
                continue
            if group["name"].startswith("static"):
                continue
            if len(group["params"]) != 1 or group["params"][0].dim() == 0:
                continue
            self.assertEqual(int(group["params"][0].shape[0]), rows_after, group["name"])
        self.assertEqual(int(gaussians._elgs_family_ids.shape[0]), rows_after)

        # The presence multiplier still resolves every row through the
        # runtime, which is the K=1 capability the group exists to serve.
        presence = state.runtime.presence_multiplier(gaussians._elgs_family_ids, 0.0)
        self.assertEqual(int(presence.shape[0]), rows_after)

    def test_repeated_densification_rounds_stay_stable(self):
        gaussians, state, opt = _build()
        snapshot = _snapshot_logits(gaussians, state)
        for iteration in (600, 700, 800):
            with torch.no_grad():
                gaussians.xyz_gradient_accum.fill_(1.0)
                gaussians.denom.fill_(1.0)
                gaussians.max_radii2D.fill_(1.0)
            gaussians.densify_and_prune(
                max_grad=opt.densify_grad_threshold,
                min_opacity=opt.thresh_opa_prune,
                extent=1.0,
                max_screen_size=None,
                max_grad_t=opt.densify_grad_t_threshold,
                iteration=iteration,
                max_total_points=opt.densify_until_num_points,
            )
            self.assert_logits_untouched(gaussians, state, snapshot)
            self.assertEqual(
                int(gaussians._elgs_family_ids.shape[0]),
                int(gaussians.get_xyz.shape[0]),
            )


if __name__ == "__main__":
    unittest.main()
