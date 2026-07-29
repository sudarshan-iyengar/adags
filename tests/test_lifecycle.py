import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from depth_visibility.capacity import (
    CapacityBank,
    build_event_blind_capacity_targets,
    select_event_blind_donors,
)
from depth_visibility.errors import ContractError
from utils.sh_utils import RGB2SH

# scene/__init__.py imports the CUDA-only pointops2 extension, which cannot be
# loaded on a CPU login node, so the lifecycle module is loaded directly from
# its file. scene.lifecycle itself only needs torch, depth_visibility, and
# utils.sh_utils.
REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("adags_scene_lifecycle", REPO / "scene/lifecycle.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE  # dataclasses resolves annotations through sys.modules
SPEC.loader.exec_module(MODULE)

LifecycleConfig = MODULE.LifecycleConfig
LifecycleManager = MODULE.LifecycleManager
VERDICT_NOT_EVALUABLE = MODULE.VERDICT_NOT_EVALUABLE
VERDICT_NEAR = MODULE.VERDICT_NEAR
VERDICT_OCCLUDED = MODULE.VERDICT_OCCLUDED
VERDICT_BEHIND_WEAK = MODULE.VERDICT_BEHIND_WEAK
VERDICT_IN_FRONT = MODULE.VERDICT_IN_FRONT
VERDICT_UNCERTAIN = MODULE.VERDICT_UNCERTAIN


def _inverse_sigmoid(value):
    return math.log(value / (1.0 - value))


class FakeEvidence:
    """Deterministic stand-in for depth_visibility.evidence_runtime.EvidenceRuntime."""

    def __init__(self, cameras, verdict_map=None, default=VERDICT_NEAR, points=None, pixels=None):
        self.cameras = list(cameras)
        self.verdict_map = dict(verdict_map or {})
        self.default = int(default)
        self.points = points
        self.pixels = pixels
        self.counts = {}
        self.calls = []
        self.backproject_calls = []

    def has_camera(self, camera_id):
        return str(camera_id) in self.cameras

    def verdicts(self, xyz, camera_id, frame):
        rows = int(xyz.shape[0])
        self.calls.append((str(camera_id), int(frame), rows))
        self.counts[str(camera_id)] = int(self.counts.get(str(camera_id), 0)) + 1
        spec = self.verdict_map.get(str(camera_id), self.default)
        if torch.is_tensor(spec) and int(spec.shape[0]) == rows:
            return spec.clone().to(torch.int8)
        fill = self.default if torch.is_tensor(spec) else int(spec)
        return torch.full((rows,), int(fill), dtype=torch.int8)

    def backproject(self, camera_id, frame, stride, sigma_max):
        self.backproject_calls.append((str(camera_id), int(frame), int(stride), float(sigma_max)))
        return self.points, self.pixels

    def counts_since_reset(self):
        return dict(self.counts)

    def reset_counts(self):
        self.counts = {}


class FakeGaussians:
    """Minimal 4D Gaussian stand-in exposing the surface the lifecycle consumes."""

    def __init__(self, n=16, dtype=torch.float32, with_optimizer=True):
        self.dtype = dtype
        self.route_logit_init = 4.0
        pair_index = torch.arange(n, dtype=dtype) // 2
        offset = (torch.arange(n, dtype=dtype) % 2) * 0.05
        xyz = torch.stack(
            [pair_index.to(dtype) + offset, torch.zeros(n, dtype=dtype), torch.zeros(n, dtype=dtype)],
            dim=1,
        )
        low = torch.linspace(-6.0, -0.25, n // 2, dtype=dtype)
        opacity = torch.full((n, 1), 3.0, dtype=dtype)
        opacity[0::2, 0] = low
        specs = {
            "_features_dc": (n, 1, 3),
            "_features_rest": (n, 2, 3),
            "_rotation": (n, 4),
            "_t": (n, 1),
            "_scaling_t": (n, 1),
            "_route_logit": (n, 1),
            "_motion_v": (n, 3),
            "_motion_a": (n, 3),
            "_motion_lora_coeff": (n, 4),
            "_staticness_score": (n, 1),
        }
        torch.manual_seed(7)
        for name, shape in specs.items():
            setattr(self, name, nn.Parameter(torch.full(shape, 0.25, dtype=dtype)))
        self._xyz = nn.Parameter(xyz)
        self._scaling = nn.Parameter(torch.log(torch.full((n, 3), 0.10, dtype=dtype)))
        self._opacity = nn.Parameter(opacity)
        self._scaling_t = nn.Parameter(torch.full((n, 1), 0.5, dtype=dtype))
        self._t = nn.Parameter(torch.full((n, 1), 1.0, dtype=dtype))

        self.xyz_gradient_accum = torch.ones(n, 1, dtype=dtype)
        self.t_gradient_accum = torch.ones(n, 1, dtype=dtype)
        self.denom = torch.ones(n, 1, dtype=dtype)
        self.max_radii2D = torch.ones(n, dtype=dtype)

        self._capacity_stable_ids = torch.arange(1000, 1000 + n, dtype=torch.long)
        self._capacity_generation = torch.zeros(n, dtype=torch.long)
        self._capacity_last_reassigned = torch.zeros(n, dtype=torch.long)

        self.optimizer = None
        if with_optimizer:
            self.optimizer = torch.optim.Adam(list(self._parameters_dict().values()), lr=0.01)
            loss = sum(parameter.square().sum() for parameter in self._parameters_dict().values())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def _parameters_dict(self):
        names = (
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_rotation",
            "_opacity",
            "_t",
            "_scaling_t",
            "_route_logit",
            "_motion_v",
            "_motion_a",
            "_motion_lora_coeff",
            "_staticness_score",
        )
        return {name: getattr(self, name) for name in names}

    def build_capacity_bank(self):
        return CapacityBank(
            parameters=self._parameters_dict(),
            accumulators={
                "xyz_gradient_accum": self.xyz_gradient_accum,
                "t_gradient_accum": self.t_gradient_accum,
                "denom": self.denom,
                "max_radii2D": self.max_radii2D,
            },
            stable_ids=self._capacity_stable_ids,
            generation=self._capacity_generation,
            last_reassigned=self._capacity_last_reassigned,
            hard_static_count=0,
        )

    def get_dynamic_xyz(self, timestamp):
        return self._xyz

    def get_marginal_t(self, timestamp):
        sigma = torch.exp(self._scaling_t.detach())
        return torch.exp(-0.5 * (self._t.detach() - float(timestamp)) ** 2 / sigma)

    def attach_grads(self):
        for parameter in self._parameters_dict().values():
            parameter.grad = torch.ones_like(parameter)

    def grow(self, n_new):
        for name, parameter in self._parameters_dict().items():
            extra = torch.zeros((n_new, *parameter.shape[1:]), dtype=parameter.dtype)
            setattr(self, name, nn.Parameter(torch.cat([parameter.detach(), extra], dim=0)))

    def shrink(self, keep_mask):
        for name, parameter in self._parameters_dict().items():
            setattr(self, name, nn.Parameter(parameter.detach()[keep_mask].clone()))


def _config(**overrides):
    values = dict(
        enable=True,
        evidence_mode="e1",
        protection=True,
        exposure=True,
        birth=False,
        birth_mode="e2",
        protect_ema_beta=0.5,
        protect_ema_threshold=0.6,
        birth_interval=100,
        birth_k=2,
        birth_warmup=100,
        birth_until=5000,
        birth_min_deficit=4,
        deficit_cell_rel=0.01,
        birth_view_checks=2,
        birth_view_min_near=2,
        backproject_stride=3,
        backproject_sigma_max=0.2,
        ledger_interval=10,
    )
    values.update(overrides)
    return LifecycleConfig(**values)


class LifecycleProtectionTests(unittest.TestCase):
    def test_all_views_and_rule_zeroes_only_shared_occluded_rows(self):
        gaussians = FakeGaussians(n=8)
        n = 8
        first = torch.full((n,), VERDICT_NEAR, dtype=torch.int8)
        first[[2, 3, 5]] = VERDICT_OCCLUDED
        second = torch.full((n,), VERDICT_NEAR, dtype=torch.int8)
        second[[3, 5, 6]] = VERDICT_OCCLUDED
        evidence = FakeEvidence(["cam00", "cam01"], {"cam00": first, "cam01": second})
        manager = LifecycleManager(gaussians, evidence, _config(), None, scene_extent=1.0)
        manager.observe_batch([("cam00", 10), ("cam01", 10)], 1)
        gaussians.attach_grads()
        protected = manager.apply_protection()
        self.assertEqual(protected, 2)
        expected = torch.zeros(n, dtype=torch.bool)
        expected[[3, 5]] = True
        self.assertTrue(torch.equal(manager.batch_occlusion_mask(), expected))
        for name in ("_xyz", "_features_dc", "_opacity", "_scaling_t", "_motion_lora_coeff"):
            grad = getattr(gaussians, name).grad
            self.assertTrue(torch.all(grad[expected] == 0.0), name)
            self.assertTrue(torch.all(grad[~expected] == 1.0), name)
        # parameters outside the protected list keep their gradients
        self.assertTrue(torch.all(gaussians._motion_v.grad == 1.0))

    def test_views_without_evidence_are_excluded_from_the_and(self):
        gaussians = FakeGaussians(n=8)
        occluded = torch.full((8,), VERDICT_NEAR, dtype=torch.int8)
        occluded[[1, 2]] = VERDICT_OCCLUDED
        evidence = FakeEvidence(["cam00"], {"cam00": occluded})
        manager = LifecycleManager(gaussians, evidence, _config(), None, scene_extent=1.0)
        used = manager.observe_batch([("cam12", 4), ("cam00", 4)], 1)
        self.assertEqual(used, 1)
        gaussians.attach_grads()
        self.assertEqual(manager.apply_protection(), 2)
        self.assertTrue(torch.all(gaussians._xyz.grad[[1, 2]] == 0.0))
        self.assertTrue(torch.all(gaussians._xyz.grad[[0, 3, 4, 5, 6, 7]] == 1.0))

    def test_no_evidence_view_means_no_protection(self):
        gaussians = FakeGaussians(n=8)
        evidence = FakeEvidence(["cam00"])
        manager = LifecycleManager(gaussians, evidence, _config(), None, scene_extent=1.0)
        manager.observe_batch([("cam12", 4), ("cam19", 4)], 1)
        gaussians.attach_grads()
        self.assertEqual(manager.apply_protection(), 0)
        self.assertTrue(torch.all(gaussians._xyz.grad == 1.0))

    def test_protection_disabled_is_a_no_op(self):
        gaussians = FakeGaussians(n=8)
        occluded = torch.full((8,), VERDICT_OCCLUDED, dtype=torch.int8)
        evidence = FakeEvidence(["cam00"], {"cam00": occluded})
        manager = LifecycleManager(gaussians, evidence, _config(protection=False), None, scene_extent=1.0)
        manager.observe_batch([("cam00", 4)], 1)
        gaussians.attach_grads()
        self.assertEqual(manager.apply_protection(), 0)
        self.assertTrue(torch.all(gaussians._xyz.grad == 1.0))

    def test_state_misalignment_fails_closed(self):
        gaussians = FakeGaussians(n=8)
        manager = LifecycleManager(gaussians, FakeEvidence(["cam00"]), _config(), None, scene_extent=1.0)
        gaussians.grow(3)
        with self.assertRaises(ContractError):
            manager.observe_batch([("cam00", 1)], 1)


class LifecycleEmaTests(unittest.TestCase):
    def test_persistent_occlusion_crosses_threshold_and_is_gated_by_protection(self):
        gaussians = FakeGaussians(n=4)
        verdicts = torch.tensor(
            [VERDICT_OCCLUDED, VERDICT_OCCLUDED, VERDICT_NEAR, VERDICT_NEAR], dtype=torch.int8
        )
        evidence = FakeEvidence(["cam00"], {"cam00": verdicts})
        manager = LifecycleManager(gaussians, evidence, _config(), None, scene_extent=1.0)
        for iteration in range(1, 4):
            manager.observe_batch([("cam00", 1)], iteration)
        # beta = 0.5, three all-occluded updates -> 0.875 > 0.6
        self.assertAlmostEqual(float(manager.occluded_ema[0]), 0.875, places=6)
        self.assertAlmostEqual(float(manager.occluded_ema[2]), 0.0, places=6)
        expected = torch.tensor([True, True, False, False])
        self.assertTrue(torch.equal(manager.persistent_occluded_mask(), expected))
        self.assertTrue(torch.equal(manager.protected_persistent(), expected))

        blind = LifecycleManager(
            FakeGaussians(n=4), evidence, _config(protection=False), None, scene_extent=1.0
        )
        for iteration in range(1, 4):
            blind.observe_batch([("cam00", 1)], iteration)
        self.assertTrue(torch.equal(blind.persistent_occluded_mask(), expected))
        self.assertTrue(torch.equal(blind.protected_persistent(), torch.zeros(4, dtype=torch.bool)))


class LifecycleExposureTests(unittest.TestCase):
    def test_e1_weights_follow_verdict_classes(self):
        gaussians = FakeGaussians(n=6)
        verdicts = torch.tensor(
            [
                VERDICT_OCCLUDED,
                VERDICT_BEHIND_WEAK,
                VERDICT_NEAR,
                VERDICT_UNCERTAIN,
                VERDICT_NOT_EVALUABLE,
                VERDICT_IN_FRONT,
            ],
            dtype=torch.int8,
        )
        evidence = FakeEvidence(["cam00"], {"cam00": verdicts})
        manager = LifecycleManager(
            gaussians,
            evidence,
            _config(occ_exposure_weight=0.0, weak_exposure_weight=0.5),
            None,
            scene_extent=1.0,
        )
        manager.observe_batch([("cam00", 1)], 1)
        manager.accumulate_exposure(0, torch.ones(6, dtype=torch.bool))
        self.assertTrue(
            torch.allclose(manager.exposure_accum, torch.tensor([0.0, 0.5, 1.0, 1.0, 1.0, 1.0]))
        )
        partial = torch.zeros(6, dtype=torch.bool)
        partial[[1, 2]] = True
        manager.accumulate_exposure(0, partial)
        self.assertTrue(
            torch.allclose(manager.exposure_accum, torch.tensor([0.0, 1.0, 2.0, 1.0, 1.0, 1.0]))
        )

    def test_view_without_evidence_uses_unit_weight(self):
        gaussians = FakeGaussians(n=4)
        evidence = FakeEvidence(["cam00"])
        manager = LifecycleManager(gaussians, evidence, _config(), None, scene_extent=1.0)
        manager.observe_batch([("cam12", 1)], 1)
        manager.accumulate_exposure(0, torch.ones(4, dtype=torch.bool))
        self.assertTrue(torch.allclose(manager.exposure_accum, torch.ones(4)))

    def test_presence_mode_uses_clamped_marginal_t(self):
        gaussians = FakeGaussians(n=4)
        manager = LifecycleManager(
            gaussians, FakeEvidence(["cam00"]), _config(evidence_mode="presence"), None, scene_extent=1.0
        )
        manager.observe_batch([("cam00", 300)], 1)
        manager.accumulate_exposure(0, torch.ones(4, dtype=torch.bool))
        expected = gaussians.get_marginal_t(10.0).reshape(-1).clamp(0.05, 1.0)
        self.assertTrue(torch.allclose(manager.exposure_accum, expected))
        self.assertTrue(torch.all(manager.exposure_accum >= 0.05))

    def test_off_mode_counts_plain_observations(self):
        gaussians = FakeGaussians(n=4)
        occluded = torch.full((4,), VERDICT_OCCLUDED, dtype=torch.int8)
        evidence = FakeEvidence(["cam00"], {"cam00": occluded})
        manager = LifecycleManager(
            gaussians, evidence, _config(evidence_mode="off"), None, scene_extent=1.0
        )
        manager.observe_batch([("cam00", 1)], 1)
        manager.accumulate_exposure(0, torch.ones(4, dtype=torch.bool))
        self.assertTrue(torch.allclose(manager.exposure_accum, torch.ones(4)))

    def test_exposure_denominator_shape_clamp_and_gate(self):
        gaussians = FakeGaussians(n=4)
        verdicts = torch.full((4,), VERDICT_OCCLUDED, dtype=torch.int8)
        evidence = FakeEvidence(["cam00"], {"cam00": verdicts})
        manager = LifecycleManager(gaussians, evidence, _config(), None, scene_extent=1.0)
        manager.observe_batch([("cam00", 1)], 1)
        manager.accumulate_exposure(0, torch.ones(4, dtype=torch.bool))
        denominator = manager.exposure_denominator()
        self.assertEqual(tuple(denominator.shape), (4, 1))
        self.assertTrue(torch.all(denominator == 1.0))
        off = LifecycleManager(
            FakeGaussians(n=4), evidence, _config(exposure=False), None, scene_extent=1.0
        )
        self.assertIsNone(off.exposure_denominator())


class LifecycleBirthTests(unittest.TestCase):
    def _evidence_with_points(self, count=8, start=100.0):
        points = torch.stack(
            [
                torch.arange(count, dtype=torch.float32) + start,
                torch.full((count,), 50.0),
                torch.full((count,), 50.0),
            ],
            dim=1,
        )
        # EvidenceRuntime.backproject returns int64 (x, y) pixels
        pixels = torch.stack([torch.arange(count), torch.arange(count)], dim=1).to(torch.int64)
        return FakeEvidence(["cam00", "cam01", "cam02"], points=points, pixels=pixels), points

    def test_e2_birth_is_point_neutral_and_writes_target_rows(self):
        gaussians = FakeGaussians(n=16)
        evidence, points = self._evidence_with_points()
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "lifecycle-ledger.jsonl"
            manager = LifecycleManager(
                gaussians, evidence, _config(birth=True), str(ledger), scene_extent=1.0, fps=30.0
            )
            gt_image = torch.full((3, 8, 8), 0.25)
            before = gaussians.build_capacity_bank().dynamic_count
            median_scaling_t = float(gaussians._scaling_t.detach().median())
            record = manager.maybe_birth(2000, ("cam00", 60), gt_image, 1.0)
            self.assertIsNotNone(record)
            self.assertIsNone(record["reason"])
            self.assertEqual(record["realized_k"], 2)
            self.assertEqual(record["proposals"], 8)
            self.assertEqual(record["multiview_pass"], 8)
            self.assertEqual(record["deficits"], 8)
            self.assertEqual(record["deficit_voxels"], 8)
            self.assertEqual(record["camera"], "cam00")
            self.assertEqual(record["frame"], 60)
            self.assertEqual(record["mode"], "e2")
            donors = record["donor_indices"]
            self.assertEqual(donors, [0, 2])
            self.assertEqual(gaussians.build_capacity_bank().dynamic_count, before)

            cell = 0.01
            for row, donor in enumerate(donors):
                self.assertTrue(
                    torch.any(torch.all(torch.isclose(points, gaussians._xyz[donor]), dim=1))
                )
                self.assertAlmostEqual(
                    float(gaussians._opacity[donor]), _inverse_sigmoid(0.3), places=5
                )
                self.assertTrue(
                    torch.allclose(gaussians._rotation[donor], torch.tensor([1.0, 0.0, 0.0, 0.0]))
                )
                self.assertAlmostEqual(float(gaussians._t[donor]), 60.0 / 30.0, places=6)
                self.assertTrue(
                    torch.allclose(
                        gaussians._scaling[donor],
                        torch.full((3,), math.log(0.5 * cell)),
                        atol=1e-5,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        gaussians._features_dc[donor, 0],
                        torch.full((3,), float(RGB2SH(torch.tensor(0.25)))),
                        atol=1e-5,
                    )
                )
                self.assertTrue(torch.all(gaussians._features_rest[donor] == 0.0))
                self.assertTrue(torch.all(gaussians._motion_lora_coeff[donor] == 0.0))
                self.assertAlmostEqual(float(gaussians._route_logit[donor]), 4.0, places=6)
                self.assertAlmostEqual(float(gaussians._scaling_t[donor]), median_scaling_t, places=6)
            self.assertTrue(torch.any(gaussians._xyz[donors[0]] != gaussians._xyz[donors[1]]))

            donor_tensor = torch.tensor(donors, dtype=torch.long)
            self.assertTrue(torch.all(manager.born_at[donor_tensor] == 2000))
            self.assertTrue(torch.all(manager.occluded_ema[donor_tensor] == 0.0))
            self.assertTrue(torch.all(manager.exposure_accum[donor_tensor] == 0.0))
            state = gaussians.optimizer.state[gaussians._xyz]
            self.assertTrue(torch.all(state["exp_avg"][donor_tensor] == 0.0))
            self.assertTrue(torch.all(state["exp_avg_sq"][donor_tensor] == 0.0))
            self.assertTrue(torch.all(gaussians._capacity_generation[donor_tensor] == 1))
            self.assertEqual(manager.births_total, 2)

            lines = [json.loads(line) for line in ledger.read_text().strip().splitlines()]
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0]["event"], "birth")
            self.assertEqual(lines[0]["realized_k"], 2)

    def test_birth_is_deterministic_for_a_fixed_iteration(self):
        results = []
        for _ in range(2):
            gaussians = FakeGaussians(n=16)
            evidence, _ = self._evidence_with_points(count=32)
            manager = LifecycleManager(
                gaussians, evidence, _config(birth=True), None, scene_extent=1.0
            )
            record = manager.maybe_birth(2000, ("cam00", 60), None, 1.0)
            results.append((record["donor_indices"], gaussians._xyz.detach()[record["donor_indices"]]))
        self.assertEqual(results[0][0], results[1][0])
        self.assertTrue(torch.equal(results[0][1], results[1][1]))

    def test_deficit_candidates_are_deduplicated_per_voxel(self):
        gaussians = FakeGaussians(n=16)
        evidence, _ = self._evidence_with_points()
        # four distinct voxels of size 0.01, two proposals each
        base = torch.tensor([100.0, 100.002, 101.0, 101.002, 102.0, 102.002, 103.0, 103.002])
        evidence.points = torch.stack([base, torch.full((8,), 50.0), torch.full((8,), 50.0)], dim=1)
        manager = LifecycleManager(
            gaussians, evidence, _config(birth=True), None, scene_extent=1.0
        )
        record = manager.maybe_birth(2000, ("cam00", 60), None, 1.0)
        self.assertEqual(record["deficits"], 8)
        self.assertEqual(record["deficit_voxels"], 4)
        self.assertEqual(record["realized_k"], 2)

    def test_birth_cadence_gates(self):
        gaussians = FakeGaussians(n=16)
        evidence, _ = self._evidence_with_points()
        manager = LifecycleManager(
            gaussians, evidence, _config(birth=True), None, scene_extent=1.0
        )
        self.assertIsNone(manager.maybe_birth(50, ("cam00", 60), None, 1.0))  # before warmup
        self.assertIsNone(manager.maybe_birth(2050, ("cam00", 60), None, 1.0))  # off cadence
        self.assertIsNone(manager.maybe_birth(6000, ("cam00", 60), None, 1.0))  # after birth_until
        self.assertEqual(manager.birth_events, 0)
        disabled = LifecycleManager(
            FakeGaussians(n=16), evidence, _config(birth=False), None, scene_extent=1.0
        )
        self.assertIsNone(disabled.maybe_birth(2000, ("cam00", 60), None, 1.0))

    def test_persistently_occluded_rows_are_never_donors(self):
        gaussians = FakeGaussians(n=16)
        evidence, _ = self._evidence_with_points()
        manager = LifecycleManager(
            gaussians, evidence, _config(birth=True), None, scene_extent=1.0
        )
        manager.occluded_ema[0] = 1.0
        record = manager.maybe_birth(2000, ("cam00", 60), None, 1.0)
        self.assertIsNotNone(record)
        self.assertEqual(record["donor_indices"], [2, 4])
        self.assertEqual(record["donor_protected_excluded"], 1)

    def test_protection_off_still_vetoes_persistently_occluded_donors(self):
        gaussians = FakeGaussians(n=16)
        evidence, _ = self._evidence_with_points()
        manager = LifecycleManager(
            gaussians, evidence, _config(birth=True, protection=False), None, scene_extent=1.0
        )
        manager.occluded_ema[0] = 1.0
        record = manager.maybe_birth(2000, ("cam00", 60), None, 1.0)
        self.assertEqual(record["donor_indices"], [2, 4])

    def test_insufficient_deficit_aborts_before_any_transaction(self):
        gaussians = FakeGaussians(n=16)
        evidence, _ = self._evidence_with_points()
        evidence.points = gaussians._xyz.detach().clone()
        evidence.pixels = torch.zeros((16, 2))
        manager = LifecycleManager(
            gaussians, evidence, _config(birth=True), None, scene_extent=1.0
        )
        record = manager.maybe_birth(2000, ("cam00", 60), None, 1.0)
        self.assertEqual(record["reason"], "insufficient_deficit")
        self.assertEqual(record["realized_k"], 0)
        self.assertEqual(record["deficits"], 0)
        self.assertTrue(torch.all(gaussians._capacity_generation == 0))
        self.assertEqual(manager.birth_events, 0)
        self.assertEqual(manager.birth_skips["insufficient_deficit"], 1)

    def test_multiview_check_rejects_unsupported_proposals(self):
        gaussians = FakeGaussians(n=16)
        evidence, _ = self._evidence_with_points()
        evidence.default = VERDICT_IN_FRONT
        manager = LifecycleManager(
            gaussians, evidence, _config(birth=True), None, scene_extent=1.0
        )
        record = manager.maybe_birth(2000, ("cam00", 60), None, 1.0)
        self.assertEqual(record["reason"], "no_multiview_support")
        self.assertEqual(record["multiview_pass"], 0)
        self.assertTrue(torch.all(gaussians._capacity_generation == 0))

    def test_camera_without_evidence_skips_birth(self):
        gaussians = FakeGaussians(n=16)
        evidence, _ = self._evidence_with_points()
        manager = LifecycleManager(
            gaussians, evidence, _config(birth=True), None, scene_extent=1.0
        )
        record = manager.maybe_birth(2000, ("cam12", 60), None, 1.0)
        self.assertEqual(record["reason"], "camera_without_evidence")
        self.assertEqual(manager.birth_events, 0)

    def test_generic_mode_uses_event_blind_targets_at_the_same_cadence(self):
        gaussians = FakeGaussians(n=16)
        evidence, _ = self._evidence_with_points()
        manager = LifecycleManager(
            gaussians, evidence, _config(birth=True, birth_mode="generic"), None, scene_extent=1.0
        )
        bank = gaussians.build_capacity_bank()
        donors = torch.tensor([0, 2], dtype=torch.long)
        expected, metadata = build_event_blind_capacity_targets(bank, donors, seed=0, iteration=2000)
        record = manager.maybe_birth(2000, ("cam00", 60), None, 1.0)
        self.assertEqual(record["mode"], "generic")
        self.assertEqual(record["donor_indices"], [0, 2])
        self.assertEqual(record["target_policy"], metadata["target_policy"])
        self.assertEqual(record["target_source_indices"], metadata["target_source_indices"])
        self.assertEqual(record["proposals"], 0)
        self.assertEqual(gaussians.build_capacity_bank().dynamic_count, 16)
        for name, rows in expected.items():
            self.assertTrue(torch.allclose(getattr(gaussians, name)[donors], rows), name)
        self.assertEqual(evidence.backproject_calls, [])


class LifecycleRowBookkeepingTests(unittest.TestCase):
    def test_rows_added_extends_state_and_resets_exposure(self):
        gaussians = FakeGaussians(n=6)
        evidence = FakeEvidence(["cam00"], {"cam00": torch.full((6,), VERDICT_OCCLUDED, dtype=torch.int8)})
        manager = LifecycleManager(gaussians, evidence, _config(), None, scene_extent=1.0)
        manager.observe_batch([("cam00", 1)], 700)
        manager.accumulate_exposure(0, torch.ones(6, dtype=torch.bool))
        manager.exposure_accum += 3.0
        gaussians.grow(2)
        manager.on_rows_added(2)
        self.assertEqual(int(manager.occluded_ema.shape[0]), 8)
        self.assertEqual(int(manager.exposure_accum.shape[0]), 8)
        self.assertEqual(int(manager.born_at.shape[0]), 8)
        self.assertTrue(torch.all(manager.exposure_accum == 0.0))
        self.assertTrue(torch.all(manager.born_at[6:] == 700))
        self.assertTrue(torch.all(manager.occluded_ema[6:] == 0.0))
        self.assertAlmostEqual(float(manager.occluded_ema[0]), 0.5, places=6)
        self.assertEqual(manager.rows_added_total, 2)

    def test_rows_pruned_slices_every_state_array(self):
        gaussians = FakeGaussians(n=6)
        evidence = FakeEvidence(["cam00"], {"cam00": torch.full((6,), VERDICT_OCCLUDED, dtype=torch.int8)})
        manager = LifecycleManager(gaussians, evidence, _config(), None, scene_extent=1.0)
        manager.observe_batch([("cam00", 1)], 10)
        manager.born_at = torch.arange(6, dtype=torch.long)
        manager.exposure_accum = torch.arange(6, dtype=torch.float32)
        keep = torch.tensor([True, False, True, True, False, True])
        gaussians.shrink(keep)
        manager.on_rows_pruned(keep)
        self.assertEqual(int(manager.occluded_ema.shape[0]), 4)
        self.assertTrue(torch.equal(manager.born_at, torch.tensor([0, 2, 3, 5])))
        self.assertTrue(torch.allclose(manager.exposure_accum, torch.tensor([0.0, 2.0, 3.0, 5.0])))
        self.assertEqual(manager.rows_pruned_total, 2)

    def test_prune_mask_length_mismatch_fails_closed(self):
        gaussians = FakeGaussians(n=6)
        manager = LifecycleManager(gaussians, FakeEvidence(["cam00"]), _config(), None, scene_extent=1.0)
        with self.assertRaises(ContractError):
            manager.on_rows_pruned(torch.ones(5, dtype=torch.bool))


class LifecycleReportingTests(unittest.TestCase):
    def test_state_dict_round_trip(self):
        gaussians = FakeGaussians(n=6)
        evidence = FakeEvidence(["cam00"], {"cam00": torch.full((6,), VERDICT_OCCLUDED, dtype=torch.int8)})
        manager = LifecycleManager(gaussians, evidence, _config(), None, scene_extent=1.0)
        manager.observe_batch([("cam00", 1)], 42)
        manager.accumulate_exposure(0, torch.ones(6, dtype=torch.bool))
        manager.born_at[2] = 17
        state = manager.state_dict()
        self.assertEqual(state["schema_version"], "phase9-lifecycle-state-v1")
        for key in ("occluded_ema", "exposure_accum", "born_at"):
            self.assertEqual(state[key].device.type, "cpu")

        restored = LifecycleManager(gaussians, evidence, _config(), None, scene_extent=1.0)
        self.assertTrue(restored.load_state_dict(state))
        self.assertTrue(torch.equal(restored.occluded_ema, manager.occluded_ema))
        self.assertTrue(torch.equal(restored.exposure_accum, manager.exposure_accum))
        self.assertTrue(torch.equal(restored.born_at, manager.born_at))
        self.assertEqual(restored.last_iteration, 42)
        self.assertFalse(restored.load_state_dict(None))
        with self.assertRaises(ContractError):
            restored.load_state_dict({"schema_version": "bogus"})

    def test_ledger_records_at_interval_only(self):
        gaussians = FakeGaussians(n=6)
        evidence = FakeEvidence(["cam00"], {"cam00": torch.full((6,), VERDICT_OCCLUDED, dtype=torch.int8)})
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "nested" / "lifecycle-ledger.jsonl"
            manager = LifecycleManager(gaussians, evidence, _config(), str(ledger), scene_extent=1.0)
            for iteration in (1, 2, 3):
                manager.observe_batch([("cam00", 1)], iteration)
                manager.apply_protection()
                manager.accumulate_exposure(0, torch.ones(6, dtype=torch.bool))
            self.assertIsNone(manager.log(9))
            record = manager.log(10, {"loss": 0.5})
            self.assertIsNotNone(record)
            self.assertEqual(record["iteration"], 10)
            self.assertEqual(record["n_protected_last"], 6)
            self.assertEqual(record["n_persistent_occluded"], 6)
            self.assertEqual(record["verdict_counts"], {"cam00": 3})
            self.assertEqual(record["exposure"]["min"], 0.0)
            self.assertEqual(record["loss"], 0.5)
            self.assertIn("timers_s", record)
            # counts_since_reset is drained per ledger window
            second = manager.log(20)
            self.assertEqual(second["verdict_counts"], {})
            lines = ledger.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["event"], "lifecycle")

    def test_summary_reports_cumulative_totals(self):
        gaussians = FakeGaussians(n=16)
        points = torch.stack(
            [
                torch.arange(8, dtype=torch.float32) + 100.0,
                torch.full((8,), 50.0),
                torch.full((8,), 50.0),
            ],
            dim=1,
        )
        evidence = FakeEvidence(
            ["cam00", "cam01", "cam02"], points=points, pixels=torch.zeros((8, 2))
        )
        manager = LifecycleManager(gaussians, evidence, _config(birth=True), None, scene_extent=1.0)
        manager.observe_batch([("cam00", 1)], 2000)
        manager.apply_protection()
        manager.maybe_birth(2000, ("cam00", 60), None, 1.0)
        summary = manager.summary()
        self.assertEqual(summary["births_total"], 2)
        self.assertEqual(summary["birth_events"], 1)
        self.assertEqual(summary["points"], 16)
        self.assertEqual(summary["config"]["birth_mode"], "e2")
        self.assertIn("exposure", summary)

    def test_invalid_configuration_fails_closed(self):
        with self.assertRaises(ContractError):
            LifecycleConfig(evidence_mode="bogus").validate()
        with self.assertRaises(ContractError):
            LifecycleConfig(birth_mode="bogus").validate()
        with self.assertRaises(ContractError):
            LifecycleConfig(birth=True, birth_interval=0).validate()


class DonorExclusionMaskTests(unittest.TestCase):
    """Coverage for the backward-compatible excluded_mask kwarg on capacity.py."""

    def _fixture(self, n=16):
        # Redundant pairs 0.05 apart with 0.10 scales: even rows are the
        # low-opacity redundant slots, odd rows are the high-opacity witnesses.
        pair_index = torch.arange(n, dtype=torch.float64) // 2
        offset = (torch.arange(n, dtype=torch.float64) % 2) * 0.05
        xyz = torch.stack([pair_index + offset, torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)], dim=1)
        opacity = torch.full((n,), 3.0, dtype=torch.float64)
        opacity[0::2] = torch.linspace(-6.0, -0.25, n // 2, dtype=torch.float64)
        return {
            "xyz": xyz,
            "scaling_log": torch.log(torch.full((n, 3), 0.10, dtype=torch.float64)),
            "opacity_logit": opacity.reshape(n, 1),
            "denom": torch.arange(1, n + 1, dtype=torch.float64).reshape(n, 1),
            "generation": torch.zeros(n, dtype=torch.long),
            "stable_ids": torch.arange(10, 10 + n, dtype=torch.long),
            "current_iteration": 5001,
        }

    def test_default_none_matches_the_legacy_call(self):
        fixture = self._fixture()
        legacy = select_event_blind_donors(k=2, **fixture)
        explicit = select_event_blind_donors(k=2, excluded_mask=None, **fixture)
        self.assertEqual(legacy, explicit)
        self.assertEqual(legacy["selected_indices"], [0, 2])

    def test_excluded_rows_are_removed_from_the_donor_universe(self):
        fixture = self._fixture()
        excluded = torch.zeros(16, dtype=torch.bool)
        excluded[0] = True
        selected = select_event_blind_donors(k=2, excluded_mask=excluded, **fixture)
        self.assertFalse(selected["abstained"])
        self.assertEqual(selected["selected_indices"], [2, 4])
        self.assertNotIn(0, selected["base_universe_indices"])

    def test_excluding_the_whole_bottom_population_abstains(self):
        fixture = self._fixture()
        excluded = torch.zeros(16, dtype=torch.bool)
        excluded[[0, 2, 4]] = True
        selected = select_event_blind_donors(k=1, excluded_mask=excluded, **fixture)
        self.assertTrue(selected["abstained"])
        self.assertEqual(selected["reason"], "empty_unexcluded_bottom_opacity_population")
        self.assertEqual(selected["selected_indices"], [])

    def test_malformed_exclusion_mask_fails_closed(self):
        fixture = self._fixture()
        with self.assertRaises(ContractError):
            select_event_blind_donors(k=1, excluded_mask=torch.zeros(3, dtype=torch.bool), **fixture)
        with self.assertRaises(ContractError):
            select_event_blind_donors(k=1, excluded_mask=torch.zeros(16, dtype=torch.float32), **fixture)


if __name__ == "__main__":
    unittest.main()
