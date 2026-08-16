"""End-to-end CPU-stub tests for elgs/trainer_hooks.py.

Closes the coverage gap the re-verification named: no prior test
constructed setup_elgs, so a startup-ordering defect passed a green
suite. This exercises the REAL trainer surface on stubs: setup ->
seeding -> reserved filter (lazily!) -> a full structural round with
slot-grid draws -> routing pins -> post-refit classification ->
checkpoint state build.
"""

import math
import types
import unittest

import torch

from depth_visibility.errors import ContractError
from elgs.state_io import build_elgs_state, load_elgs_state
from elgs.trainer_hooks import (
    apply_elgs_routing_pins,
    elgs_summary,
    filter_elgs_reserved,
    maybe_run_elgs_schedule,
    setup_elgs,
)


class _StubCamera:
    def __init__(self, timestamp, name):
        self.timestamp = timestamp
        self.image_name = name


class _CountingDataset:
    """Counts __getitem__ calls so laziness is testable."""

    def __init__(self, cameras):
        self._cameras = cameras
        self.getitem_calls = 0

    def __len__(self):
        return len(self._cameras)

    def __getitem__(self, index):
        self.getitem_calls += 1
        return self._cameras[index]


class _StubScene:
    def __init__(self, dataset):
        self._dataset = dataset

    def getTrainCameras(self):
        return self._dataset


def _stub_opt():
    return types.SimpleNamespace(
        elgs_enable=True,
        lifecycle_enable=False,
        elgs_prereg_dir="configs/elgs",
        elgs_smoke_schedule=True,
        elgs_a_lr=0.01,
        elgs_k_se=1.0,
        elgs_lambda_u=1.0,
        elgs_candidate_cap=2,
        elgs_confirmation_samples=6,
        elgs_tracks_dir="",
        densify_until_num_points=100_000,
        seed=7,
    )


def _stub_gaussians(n_rows=64):
    g = types.SimpleNamespace()
    gen = torch.Generator().manual_seed(3)
    g._xyz = torch.randn(n_rows, 3, generator=gen)
    g._opacity = torch.zeros(n_rows, 1)
    g._route_logit = torch.zeros(n_rows, 1, requires_grad=True)
    g._pending_elgs_state = None
    dummy = torch.zeros(2, requires_grad=True)
    g.optimizer = torch.optim.Adam([{"params": [dummy], "name": "dummy"}], lr=0.0)
    return g


def _build():
    # Camera-fastest ordering with n_cams % 4 == 0: exactly the
    # ordering that collapsed an index-stride reservation onto one
    # camera — the stratified scheme must survive it.
    cameras = [
        _StubCamera(timestamp=0.25 * (i // 4), name=f"cam{i % 4}_t{i // 4:03d}")
        for i in range(240)
    ]
    dataset = _CountingDataset(cameras)
    scene = _StubScene(dataset)
    gaussians = _stub_gaussians()
    state = setup_elgs(gaussians, scene, dataset=None, opt=_stub_opt())
    return state, gaussians, scene, dataset


def _render_loss(item, override):
    # Synthetic paired evaluator: candidates improve the loss.
    return 1.0 if override is None else 0.5


class RuntimeDeviceTests(unittest.TestCase):
    """The fresh path must place the runtime on the MODEL's device.

    A fresh run built `ElgsRuntime` with the default device (cpu) while
    the restore branch used `gaussians._xyz.device`. On CUDA that put the
    presence column and the probe's projected geometry on different
    devices, and the first round that actually reached `transmittance`
    died with

      RuntimeError: indices should be either on cpu or on the same
      device as the indexed tensor (cpu)

    Experiment 73 never reached it (no evidence windows), so the
    asymmetry survived until experiment 75. This pins the invariant
    device-agnostically: whatever device the model is on, the runtime
    and the presence column must agree with it, on BOTH paths.
    """

    def test_fresh_runtime_matches_the_model_device(self):
        state, gaussians, _, _ = _build()
        self.assertEqual(state.runtime.device, gaussians._xyz.device)

    def test_presence_column_lands_on_the_model_device(self):
        state, gaussians, _, _ = _build()
        column = state.runtime.presence_multiplier(
            gaussians._elgs_family_ids, 0.0
        )
        self.assertEqual(column.device, gaussians._xyz.device)

    def test_restored_runtime_matches_the_model_device_too(self):
        state, gaussians, scene, dataset = _build()
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
        fresh = _stub_gaussians()
        fresh._pending_elgs_state = payload
        restored = setup_elgs(fresh, scene, dataset=None, opt=_stub_opt())
        self.assertEqual(restored.runtime.device, fresh._xyz.device)


class SetupTests(unittest.TestCase):
    def test_setup_constructs_and_seeds(self):
        state, gaussians, _, _ = _build()
        self.assertIsNotNone(state)  # the N1 regression guard
        self.assertTrue(state.seeded)
        self.assertGreater(len(state.runtime.registry.active_ids()), 0)
        self.assertEqual(gaussians._elgs_family_ids.shape[0], 64)
        self.assertTrue((gaussians._elgs_family_ids >= 0).all())
        # Slot grid honors the DECLARED confirmation-sample count.
        self.assertEqual(state.slot_grid.units_per_slot, 6)
        self.assertEqual(state.candidate_cap, 2)

    def test_reserved_filter_is_lazy_and_disjoint(self):
        state, _, _, dataset = _build()
        calls_before = dataset.getitem_calls
        filtered = filter_elgs_reserved(dataset, state)
        self.assertEqual(dataset.getitem_calls, calls_before)  # N2 guard
        self.assertEqual(len(filtered), 240 - len(state.reserved_indices))
        # The filtered view never exposes a reserved index.
        kept = set(filtered.indices)
        self.assertFalse(kept & state.reserved_indices)

    def test_reservation_is_stratified_across_cameras_and_time(self):
        """P4 guard: under camera-fastest ordering with n_cams % 4 == 0
        the old index-stride reservation collapsed onto one camera;
        the diagonal scheme must cover multiple cameras and span time."""
        state, _, _, dataset = _build()
        reserved_cams = {i % 4 for i in state.reserved_indices}
        self.assertGreaterEqual(len(reserved_cams), 2)
        reserved_frames = {i // 4 for i in state.reserved_indices}
        self.assertGreaterEqual(max(reserved_frames) - min(reserved_frames), 30)

    def test_pool_too_small_fails_closed(self):
        cameras = [_StubCamera(0.25 * i, f"c{i}") for i in range(40)]
        dataset = _CountingDataset(cameras)
        scene = _StubScene(dataset)
        opt = _stub_opt()
        opt.elgs_confirmation_samples = 16  # needs 3*2*16=96 > 10 reserved
        with self.assertRaises(ContractError):
            setup_elgs(_stub_gaussians(), scene, dataset=None, opt=opt)


class RoundAndPostRefitTests(unittest.TestCase):
    def test_full_round_commit_pins_and_post_refit(self):
        state, gaussians, scene, _ = _build()
        # Smoke schedule rounds: 200, 350, 500; refit_until 600.
        maybe_run_elgs_schedule(state, gaussians, scene, _stub_opt(), 200, _render_loss)
        self.assertEqual(state.rounds_run, [200])
        self.assertEqual(len(state.committed_decisions), 1)
        decision = state.committed_decisions[0]
        self.assertEqual(len(decision["units"]), 6)
        self.assertTrue(decision["incumbent_intervals"])
        # The committed fission pinned its family (K=2 now).
        self.assertTrue(gaussians._elgs_pinned_families)
        # confirmation_refs persisted for the §8 pass.
        refs = gaussians._elgs_checkpoint_extras["confirmation_refs"]
        self.assertIn(decision["candidate_id"], refs)

        # OBSERVABILITY: every term the acceptance rule adds up is
        # persisted, not just n_samples and se. Experiment 78 committed
        # with se = 0.0 and its delta_render / exact_deltas split was
        # then unrecoverable from checkpoint, trial log and tfevents
        # alike, so "the photometric arm was silent" and "the evidence
        # term decided it" could not be told apart after the fact.
        for field in ("delta_render", "exact_deltas", "transaction_increment",
                      "delta_total", "k", "ess", "n_units"):
            self.assertIn(field, decision, f"{field} is not persisted")
            self.assertTrue(
                math.isfinite(float(decision[field])),
                f"{field} persisted non-finite",
            )
        # The rule itself, restated against the persisted numbers: a
        # commit means delta_total + k*se < 0, and delta_total is the
        # sum of the three recorded terms.
        self.assertAlmostEqual(
            decision["delta_total"],
            decision["delta_render"] + decision["exact_deltas"]
            + decision["transaction_increment"],
            places=12,
        )
        self.assertLess(
            decision["delta_total"] + decision["k"] * decision["se"], 0.0
        )

        # Routing-pin application zeroes pinned rows' grads and fails
        # closed on a row-count desync (N4 guard).
        gaussians._route_logit.grad = torch.ones(64, 1)
        apply_elgs_routing_pins(gaussians)
        ids = gaussians._elgs_family_ids
        pinned_rows = torch.zeros(64, dtype=torch.bool)
        for fid in gaussians._elgs_pinned_families:
            pinned_rows |= ids == fid
        self.assertEqual(float(gaussians._route_logit.grad[pinned_rows].abs().sum()), 0.0)
        self.assertGreater(float(gaussians._route_logit.grad[~pinned_rows].abs().sum()), 0.0)
        # Desync guard: a family-id column that no longer matches the
        # grad's row count must fail closed, never silently unpin.
        gaussians._route_logit.grad = torch.ones(64, 1)
        full_column = gaussians._elgs_family_ids
        gaussians._elgs_family_ids = full_column[:-1]
        with self.assertRaises(ContractError):
            apply_elgs_routing_pins(gaussians)
        gaussians._elgs_family_ids = full_column
        gaussians._route_logit.grad = None

        # Post-refit §8 pass at refit_until: labels recorded with a
        # re-bootstrapped SE, and the flag survives the checkpoint.
        maybe_run_elgs_schedule(state, gaussians, scene, _stub_opt(), 600, _render_loss)
        self.assertTrue(state.post_refit_done)
        book = gaussians._elgs_checkpoint_extras["round_bookkeeping"]
        self.assertTrue(book["post_refit_done"])
        results = book["post_refit_classification"]
        self.assertEqual(len(results), 1)
        label = next(iter(results.values()))["class"]
        self.assertIn("(fixed-path decision decomposition", label)

    def test_state_round_trips_through_checkpoint(self):
        state, gaussians, scene, _ = _build()
        maybe_run_elgs_schedule(state, gaussians, scene, _stub_opt(), 200, _render_loss)
        state.runtime.flush_to_registry()
        extras = gaussians._elgs_checkpoint_extras
        payload = build_elgs_state(
            state.runtime.registry, state.bundle.binding, state.bundle.ledger,
            state.bundle.search_cost,
            sampler=extras["sampler"],
            slot_grid=extras["slot_grid"],
            confirmation_refs=extras["confirmation_refs"],
            moment_reset_log=extras["moment_reset_log"],
            round_bookkeeping=extras["round_bookkeeping"],
        )
        loaded = load_elgs_state(payload)
        self.assertEqual(
            loaded["round_bookkeeping"]["rounds_run"], [200]
        )
        self.assertEqual(
            len(loaded["round_bookkeeping"]["committed_decisions"]), 1
        )
        summary = elgs_summary(state)
        self.assertEqual(summary["candidates_accepted"], 1)
        self.assertGreater(summary["ledger_events"], 0)


if __name__ == "__main__":
    unittest.main()
