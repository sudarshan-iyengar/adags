"""Anti-vacuity tests for the corrected LOCALIZED episodic-presence cell
(LRV3 A1-LOCAL), covering the six defect classes the 2026-08-20 code
audit confirmed in the 2026-08-19 A1/A2 wiring:

1. configuration controls the materialized route logits (until the
   main.py ordering repair, create_from_pcd filled them from the
   constructor default 4.0 BEFORE training_setup read the YAML, and
   _ensure_route_and_motion_tensors preserved the correctly sized
   tensor — so YAML 0.0 never took effect on a fresh run);
2. presence zero is a TOTAL gate: zero contribution through the
   dynamic branch AND the separately rendered static twin, at
   representative dynamic probabilities (~0.018, 0.5, ~0.982);
3. non-oracle rows keep the ORDINARY temporal marginal;
4. oracle membership is PER-PRIMITIVE, not per voxel cell;
5. K >= 2 episode parameters are non-degenerate and receive gradients
   under local seeding;
6. routing pins can be disabled, and evaluator restoration preserves
   the localized semantics.
"""

import json
import math
import re
import tempfile
import types
import unittest
from pathlib import Path

import torch

from depth_visibility.errors import ContractError
from elgs.presence import local_presence_multipliers
from elgs.state_io import build_elgs_state
from elgs.trainer_hooks import (
    ORACLE_EPISODES_SCHEMA,
    apply_elgs_routing_pins,
    setup_elgs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _StubCamera:
    def __init__(self, timestamp, name):
        self.timestamp = timestamp
        self.image_name = name


class _StubDataset:
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


def _write_oracle(tmp, *, centre=(0.0, 0.0, 0.0), radius=0.1,
                  gaps=((5.0, 9.0),)):
    payload = {
        "schema_version": ORACLE_EPISODES_SCHEMA,
        "units": "model_time_seconds",
        "region": {"kind": "sphere", "centre": list(centre), "radius": radius},
        "gaps": [list(g) for g in gaps],
    }
    path = Path(tmp) / "oracle.json"
    path.write_text(json.dumps(payload))
    return str(path)


def _stub_opt(oracle_path, *, local=True, pins=True):
    return types.SimpleNamespace(
        elgs_enable=True,
        lifecycle_enable=False,
        elgs_prereg_dir="configs/elgs",
        elgs_smoke_schedule=True,
        elgs_a_lr=0.0,
        elgs_k_se=1.0,
        elgs_lambda_u=1.0,
        elgs_candidate_cap=2,
        elgs_confirmation_samples=6,
        elgs_tracks_dir="",
        elgs_oracle_episodes=oracle_path,
        elgs_local_presence=local,
        elgs_routing_pins_enabled=pins,
        densify_until_num_points=100_000,
        seed=7,
    )


def _mixed_cell_xyz():
    """64 rows with a DELIBERATELY mixed voxel cell.

    Corner anchors pin the bounding box to [-1, 1]^3 (8 cells per axis,
    cell size 0.25). Rows 2-6 sit within 0.05 of the origin (inside the
    radius-0.1 oracle sphere); rows 7-11 sit at x ~= 0.2 — the SAME
    voxel cell as rows 2-6, but outside the sphere. Under per-voxel
    membership all ten join the oracle family; under per-primitive
    membership only the five in-region rows do.
    """
    gen = torch.Generator().manual_seed(11)
    xyz = torch.empty(64, 3)
    xyz[0] = torch.tensor([-1.0, -1.0, -1.0])
    xyz[1] = torch.tensor([1.0, 1.0, 1.0])
    for i in range(5):
        xyz[2 + i] = torch.tensor([0.01 * i, 0.01, 0.01])
    for i in range(5):
        xyz[7 + i] = torch.tensor([0.20 + 0.002 * i, 0.01, 0.01])
    xyz[12:] = torch.rand(52, 3, generator=gen) * 1.2 - 0.6
    # keep the random filler out of the oracle sphere so membership
    # counts are exact
    close = xyz[12:].norm(dim=1) < 0.15
    xyz[12:][close] = torch.tensor([0.5, 0.5, 0.5])
    return xyz


def _stub_gaussians(xyz):
    g = types.SimpleNamespace()
    n = xyz.shape[0]
    g._xyz = xyz.clone()
    g._opacity = torch.zeros(n, 1)
    g._route_logit = torch.zeros(n, 1, requires_grad=True)
    g._pending_elgs_state = None
    dummy = torch.zeros(2, requires_grad=True)
    g.optimizer = torch.optim.Adam([{"params": [dummy], "name": "dummy"}], lr=0.0)
    return g


def _build(*, local=True, pins=True, tmp=None):
    cameras = [
        _StubCamera(timestamp=0.25 * (i // 4), name=f"cam{i % 4}_t{i // 4:03d}")
        for i in range(240)
    ]
    dataset = _StubDataset(cameras)
    scene = _StubScene(dataset)
    gaussians = _stub_gaussians(_mixed_cell_xyz())
    oracle = _write_oracle(tmp)
    opt = _stub_opt(oracle, local=local, pins=pins)
    state = setup_elgs(gaussians, scene, dataset=None, opt=opt)
    return state, gaussians, scene, opt


IN_REGION_ROWS = set(range(2, 7))
SAME_CELL_OUT_ROWS = set(range(7, 12))


class TotalGateCompositionTests(unittest.TestCase):
    """Presence zero must zero the TOTAL routed contribution."""

    def test_presence_zero_is_a_total_gate_at_representative_route_probs(self):
        base_opacity = torch.full((3, 1), 0.9)
        presence = torch.zeros(3, 1)
        marginal = torch.full((3, 1), 0.7)
        gated = torch.ones(3, 1, dtype=torch.bool)
        dyn_mult, static_mult = local_presence_multipliers(
            presence, marginal, gated
        )
        for logit in (-4.0, 0.0, 4.0):  # p_dyn ~ 0.018, 0.5, 0.982
            p = torch.sigmoid(torch.full((3, 1), logit))
            total = base_opacity * p * dyn_mult + base_opacity * (1 - p) * static_mult
            self.assertEqual(float(total.abs().max()), 0.0)

    def test_non_gated_rows_keep_the_substrate_exactly(self):
        presence = torch.zeros(4, 1)
        marginal = torch.tensor([[0.7], [0.2], [1.0], [0.05]])
        gated = torch.zeros(4, 1, dtype=torch.bool)
        dyn_mult, static_mult = local_presence_multipliers(
            presence, marginal, gated
        )
        self.assertTrue(torch.equal(dyn_mult, marginal))
        self.assertTrue(torch.equal(static_mult, torch.ones_like(presence)))

    def test_gated_rows_at_full_presence_render_fully_on_both_branches(self):
        presence = torch.ones(2, 1)
        marginal = torch.full((2, 1), 0.3)
        gated = torch.ones(2, 1, dtype=torch.bool)
        dyn_mult, static_mult = local_presence_multipliers(
            presence, marginal, gated
        )
        self.assertTrue(torch.equal(dyn_mult, torch.ones_like(presence)))
        self.assertTrue(torch.equal(static_mult, torch.ones_like(presence)))

    def test_shape_mismatch_fails_closed(self):
        with self.assertRaises(ContractError):
            local_presence_multipliers(
                torch.zeros(3, 1), torch.zeros(4, 1),
                torch.zeros(3, 1, dtype=torch.bool),
            )


class PerPrimitiveMembershipTests(unittest.TestCase):
    def test_membership_is_per_primitive_not_per_voxel_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, gaussians, _, _ = _build(tmp=tmp)
        ids = gaussians._elgs_family_ids
        assigned = {int(i) for i in torch.nonzero(ids >= 0).reshape(-1).tolist()}
        self.assertEqual(assigned, IN_REGION_ROWS)
        # the anti-vacuity half: the SAME voxel cell holds out-of-region
        # rows, and they must stay unassigned
        for row in SAME_CELL_OUT_ROWS:
            self.assertEqual(int(ids[row]), -1)

    def test_legacy_mode_still_gates_the_whole_cell(self):
        """The old per-voxel semantics are preserved when the flag is
        off — and this is the shape of the defect the local mode fixes."""
        with tempfile.TemporaryDirectory() as tmp:
            state, gaussians, _, _ = _build(local=False, tmp=tmp)
        ids = gaussians._elgs_family_ids
        self.assertTrue(bool((ids >= 0).all()))
        k2 = {
            fid for fid in state.runtime.registry.active_ids()
            if state.runtime.registry.get(fid).interval.K > 1
        }
        gated_rows = {
            int(i) for i in torch.nonzero(
                torch.isin(ids, torch.tensor(sorted(k2)))
            ).reshape(-1).tolist()
        }
        self.assertTrue(IN_REGION_ROWS <= gated_rows)
        self.assertTrue(SAME_CELL_OUT_ROWS <= gated_rows)

    def test_gated_row_mask_matches_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, gaussians, _, _ = _build(tmp=tmp)
        mask = state.runtime.gated_row_mask(gaussians._elgs_family_ids)
        self.assertEqual(mask.shape, (64, 1))
        gated = {int(i) for i in torch.nonzero(mask.reshape(-1)).reshape(-1).tolist()}
        self.assertEqual(gated, IN_REGION_ROWS)

    def test_all_local_families_carry_the_k2_program(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, _, _, _ = _build(tmp=tmp)
        reg = state.runtime.registry
        ks = [reg.get(f).interval.K for f in reg.active_ids()]
        self.assertTrue(ks)
        self.assertTrue(all(k == 2 for k in ks))

    def test_local_presence_without_an_oracle_fails_closed(self):
        cameras = [
            _StubCamera(timestamp=0.25 * (i // 4), name=f"c{i}")
            for i in range(240)
        ]
        scene = _StubScene(_StubDataset(cameras))
        opt = _stub_opt("", local=True)
        with self.assertRaises(ContractError):
            setup_elgs(
                _stub_gaussians(_mixed_cell_xyz()), scene, dataset=None, opt=opt
            )


class NonOracleMarginalTests(unittest.TestCase):
    """Non-gated rows must render through the ordinary marginal even
    though presence_multiplier returns exact 0 for unassigned rows."""

    def test_unassigned_rows_use_the_marginal_not_the_zero_presence(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, gaussians, _, _ = _build(tmp=tmp)
        t_in_gap = 7.0  # inside [5, 9]: gated presence is exactly 0
        presence = state.runtime.presence_multiplier(
            gaussians._elgs_family_ids, t_in_gap
        )
        gated = state.runtime.gated_row_mask(gaussians._elgs_family_ids)
        marginal = torch.rand(64, 1) * 0.9 + 0.05
        dyn_mult, static_mult = local_presence_multipliers(
            presence, marginal, gated
        )
        flat = gated.reshape(-1)
        self.assertTrue(torch.equal(dyn_mult[~flat], marginal[~flat]))
        self.assertEqual(float(dyn_mult[flat].abs().max()), 0.0)
        self.assertEqual(float(static_mult[flat].abs().max()), 0.0)
        self.assertTrue(
            torch.equal(static_mult[~flat], torch.ones_like(marginal[~flat]))
        )


class EpisodeParameterTests(unittest.TestCase):
    def test_k2_logits_are_nondegenerate_and_receive_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, gaussians, _, _ = _build(tmp=tmp)
        params = state.runtime.logit_parameters()
        self.assertTrue(params)
        for a in params.values():
            self.assertEqual(a.numel(), 3)  # dim(a) = 2K+1-n_lat = 3
        # gradient flows from the presence column into the a-logits at
        # an edge-band timestamp (0.25 into episode 2's onset ramp)
        column = state.runtime.presence_multiplier(
            gaussians._elgs_family_ids, 9.25
        )
        column.sum().backward()
        grads = [a.grad for a in state.runtime.logit_parameters().values()]
        self.assertTrue(any(
            g is not None and float(g.abs().max()) > 0.0 for g in grads
        ))


class RoutingPinTests(unittest.TestCase):
    def test_pins_disabled_leaves_no_pinned_family_and_grads_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, gaussians, _, _ = _build(pins=False, tmp=tmp)
        self.assertEqual(getattr(gaussians, "_elgs_pinned_families", None), set())
        grad = torch.ones(64, 1)
        gaussians._route_logit.grad = grad.clone()
        apply_elgs_routing_pins(gaussians)
        self.assertTrue(torch.equal(gaussians._route_logit.grad, grad))

    def test_pins_enabled_still_zero_the_gated_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, gaussians, _, _ = _build(pins=True, tmp=tmp)
        self.assertTrue(gaussians._elgs_pinned_families)
        gaussians._route_logit.grad = torch.ones(64, 1)
        apply_elgs_routing_pins(gaussians)
        flat = state.runtime.gated_row_mask(
            gaussians._elgs_family_ids
        ).reshape(-1)
        self.assertEqual(
            float(gaussians._route_logit.grad[flat].abs().max()), 0.0
        )
        self.assertEqual(
            float(gaussians._route_logit.grad[~flat].min()), 1.0
        )


class RestorationTests(unittest.TestCase):
    def test_restore_preserves_local_semantics_and_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, gaussians, scene, opt = _build(tmp=tmp)
            payload = build_elgs_state(
                state.runtime.registry,
                gaussians._elgs_checkpoint_extras["binding"],
                gaussians._elgs_checkpoint_extras["ledger"],
                gaussians._elgs_checkpoint_extras["search_cost"],
                sampler=gaussians._elgs_checkpoint_extras["sampler"],
                slot_grid=gaussians._elgs_checkpoint_extras["slot_grid"],
                confirmation_refs={},
                moment_reset_log=[],
                round_bookkeeping={
                    "seeded": True,
                    "rounds_run": [],
                    "row_family_ids": gaussians._elgs_family_ids.tolist(),
                },
            )
            fresh = _stub_gaussians(_mixed_cell_xyz())
            fresh._pending_elgs_state = payload
            restored = setup_elgs(fresh, scene, dataset=None, opt=opt)
        self.assertTrue(restored.local_presence)
        self.assertTrue(getattr(fresh, "_elgs_local_presence", False))
        mask = restored.runtime.gated_row_mask(fresh._elgs_family_ids)
        gated = {int(i) for i in torch.nonzero(mask.reshape(-1)).reshape(-1).tolist()}
        self.assertEqual(gated, IN_REGION_ROWS)

    def test_default_path_does_not_declare_local_presence(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, gaussians, _, _ = _build(local=False, tmp=tmp)
        self.assertFalse(state.local_presence)
        self.assertFalse(getattr(gaussians, "_elgs_local_presence", True))


class RouteLogitInitOrderTests(unittest.TestCase):
    """The materialized route logits must come from the CONFIGURED
    value, not the constructor default."""

    def _model(self):
        from scene.gaussian_model import GaussianModel

        g = GaussianModel(
            3, gaussian_dim=4, time_duration=[0.0, 10.0],
            rot_4d=False, force_sh_3d=True,
        )
        g._xyz = torch.randn(16, 3)
        return g

    def test_preserve_semantics_are_the_defect_mechanism(self):
        """Anti-vacuity: materialize at the constructor default first
        (the pre-repair order), then hand training_setup-style ensure
        the YAML value — the tensor is PRESERVED at 4.0. This is
        exactly why YAML 0.0 never took effect before the main.py
        ordering repair."""
        g = self._model()
        g._ensure_route_and_motion_tensors()  # constructor default 4.0
        self.assertEqual(float(g._route_logit.min()), 4.0)
        g.route_logit_init = 0.0
        g._ensure_route_and_motion_tensors(0.0)
        self.assertEqual(float(g._route_logit.min()), 4.0)  # preserved

    def test_configured_value_controls_when_set_before_materialization(self):
        """The repaired order: main.py sets route_logit_init from the
        config BEFORE Scene() triggers materialization."""
        g = self._model()
        g.route_logit_init = 0.0
        g._ensure_route_and_motion_tensors()
        self.assertEqual(float(g._route_logit.abs().max()), 0.0)
        self.assertAlmostEqual(
            float(torch.sigmoid(g._route_logit).mean()), 0.5, places=6
        )

    def test_main_py_sets_the_init_before_scene_construction(self):
        """Source-level regression guard on the ordering repair."""
        source = (REPO_ROOT / "main.py").read_text(encoding="utf-8")
        train_src = source[source.index("def training("):]
        scene_pos = train_src.index("scene = Scene(dataset, gaussians")
        assign = re.search(
            r"gaussians\.route_logit_init\s*=\s*getattr\(opt", train_src
        )
        self.assertIsNotNone(assign)
        self.assertLess(assign.start(), scene_pos)


if __name__ == "__main__":
    unittest.main()
