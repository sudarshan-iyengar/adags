"""Oracle-episode supply, and the presence row-expansion repair it needed.

Both landed together for the EXPLORATORY oracle-headroom lane: the supply path
is what lets a cell be seeded with a fixed K >= 2 program, and the repair is
what makes such a cell affordable (the per-row Python stack it replaces cost
1.04 s per call at 50k rows and 4.13 s at 300k, forward+backward, which alone
exceeded the training budget).
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from depth_visibility.errors import ContractError
from elgs.families import FamilyRegistry
from elgs.intervals import IntervalConfig, inverse
from elgs.presence import episode_presence
from elgs.runtime import ElgsRuntime, ScheduleAnchors
from elgs.trainer_hooks import ORACLE_EPISODES_SCHEMA, load_oracle_episodes

DT = 1.0 / 6.0
T_SPAN = 59.0 / 6.0


def build_config():
    return IntervalConfig(T=T_SPAN, w_m=2 * DT, w=2 * DT,
                          floor_len=5 * DT, floor_gap=5 * DT, delta_tol=0.1 * DT)


def write_spec(tmp, gaps, *, centre=(0.7, 0.1, 0.35), radius=0.2, **overrides):
    payload = {
        "schema_version": ORACLE_EPISODES_SCHEMA,
        "units": "model_time_seconds",
        "region": {"kind": "sphere", "centre": list(centre), "radius": radius},
        "gaps": gaps,
    }
    payload.update(overrides)
    path = Path(tmp) / "oracle.json"
    path.write_text(json.dumps(payload))
    return str(path)


class LoadOracleEpisodesTests(unittest.TestCase):

    def test_one_gap_yields_a_latched_k2_program_at_the_requested_boundaries(self):
        cfg = build_config()
        with tempfile.TemporaryDirectory() as tmp:
            _, interval, tau = load_oracle_episodes(
                write_spec(tmp, [[4.9166667, 8.9166667]]), cfg)
        self.assertEqual(interval.K, 2)
        self.assertTrue(interval.latch_pre and interval.latch_post)
        # dim(a) = 2K + 1 - n_lat = 3, and it is NOT the degenerate K=1 case
        self.assertEqual(interval.a.numel(), 3)
        self.assertEqual(len(tau), 2)
        from elgs.intervals import forward
        r = forward(interval, cfg)
        self.assertAlmostEqual(float(r.b[0]), -cfg.w_m, places=5)
        self.assertAlmostEqual(float(r.d[0]), 4.9166667, places=4)
        self.assertAlmostEqual(float(r.b[1]), 8.9166667, places=4)
        self.assertAlmostEqual(float(r.d[1]), cfg.T + cfg.w_m, places=5)

    def test_softmax_weights_are_non_constant_and_the_logits_take_gradient(self):
        """The exact property the K=1 seeding provably lacks."""
        cfg = build_config()
        with tempfile.TemporaryDirectory() as tmp:
            _, interval, _ = load_oracle_episodes(
                write_spec(tmp, [[4.9166667, 8.9166667]]), cfg)
        weights = torch.softmax(interval.a, dim=0)
        self.assertEqual(weights.numel(), 3)
        self.assertGreater(float(weights.max() - weights.min()), 1e-3)

        a = interval.a.detach().clone().requires_grad_(True)
        from elgs.intervals import IntervalState, forward
        r = forward(IntervalState(K=2, latch_pre=True, latch_post=True, a=a), cfg)
        # a timestamp inside the left edge band of the first episode's end
        episode_presence(torch.tensor(4.9166667 - 0.5 * cfg.w), r.b, r.d, cfg.w).sum().backward()
        self.assertIsNotNone(a.grad)
        self.assertGreater(float(a.grad.abs().max()), 0.0)

    def test_two_gaps_yield_k3(self):
        cfg = build_config()
        with tempfile.TemporaryDirectory() as tmp:
            _, interval, tau = load_oracle_episodes(
                write_spec(tmp, [[2.0, 3.5], [6.0, 7.5]]), cfg)
        self.assertEqual((interval.K, interval.a.numel(), len(tau)), (3, 5, 3))

    def test_a_gap_below_the_preregistered_floor_fails_closed(self):
        cfg = build_config()
        short = 0.5 * cfg.floor_gap
        with tempfile.TemporaryDirectory() as tmp:
            path = write_spec(tmp, [[4.9166667, 4.9166667 + short]])
            with self.assertRaises(Exception):
                load_oracle_episodes(path, cfg)

    def test_non_increasing_boundaries_are_rejected(self):
        cfg = build_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_spec(tmp, [[6.0, 7.5], [2.0, 3.5]])
            with self.assertRaises(ContractError):
                load_oracle_episodes(path, cfg)

    def test_wrong_schema_version_is_rejected(self):
        cfg = build_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_spec(tmp, [[4.9166667, 8.9166667]],
                              schema_version="elgs-oracle-episodes-v99")
            with self.assertRaises(ContractError):
                load_oracle_episodes(path, cfg)

    def test_the_wrong_time_control_is_exactly_dose_matched_to_the_correct_one(self):
        """A2 must differ from A1 in timing and in nothing else."""
        cfg = build_config()
        lo, hi = -cfg.w_m, cfg.T + cfg.w_m
        correct = [[4.9166667, 8.9166667]]
        # reflection of the correct schedule about the window midpoint
        mid = 0.5 * (lo + hi)
        wrong = sorted([[2 * mid - b, 2 * mid - a] for a, b in correct])
        with tempfile.TemporaryDirectory() as tmp:
            _, iv_a, tau_a = load_oracle_episodes(write_spec(tmp, correct), cfg)
        with tempfile.TemporaryDirectory() as tmp:
            _, iv_b, tau_b = load_oracle_episodes(write_spec(tmp, wrong), cfg)

        from elgs.intervals import forward
        ra, rb = forward(iv_a, cfg), forward(iv_b, cfg)
        lens_a = sorted(round(float(d - b), 6) for b, d in zip(ra.b, ra.d))
        lens_b = sorted(round(float(d - b), 6) for b, d in zip(rb.b, rb.d))
        self.assertEqual((iv_a.K, len(tau_a)), (iv_b.K, len(tau_b)))
        self.assertEqual(iv_a.a.numel(), iv_b.a.numel())
        self.assertEqual(lens_a, lens_b)                      # same durations
        self.assertNotEqual([float(x) for x in ra.b], [float(x) for x in rb.b])


class RampPlacementTests(unittest.TestCase):
    """The defect a fresh-context review caught, pinned so it cannot return.

    `dim(a)`, episode count and duration multiset can all be correct while the
    experiment is still broken, because presence does not step -- it ramps over
    the smoothstep half-width `w` on each side of a boundary. Putting a
    boundary at the midpoint between the last absent and first present frame
    (the obvious choice) lands that ramp ON the first evaluated frames: the
    original LRV1 oracle rendered the event object at presence 0.15625 on frame
    54 and 0.84375 on frame 55, two of the six frames the whole comparison
    turns on, while its dose-matched control sat at 1.0 on all six.

    A duration-matched control is NOT automatically a ramp-matched one.
    """

    LRV1_PRESENT_FRAMES = frozenset(list(range(0, 30)) + list(range(54, 60)))
    LRV1_RETURN_FRAMES = tuple(range(54, 60))

    def presence_curve(self, path, cfg):
        from elgs.intervals import forward
        _, interval, _ = load_oracle_episodes(path, cfg)
        r = forward(interval, cfg)
        return [float(episode_presence(torch.tensor(f / 6.0), r.b, r.d, cfg.w).sum())
                for f in range(60)]

    def test_shipped_lrv1_oracles_are_flat_on_every_ground_truth_present_frame(self):
        cfg = build_config()
        root = Path(__file__).resolve().parents[1] / "configs" / "lrv1"
        for name in ("oracle_correct.json", "oracle_wrong.json"):
            path = root / name
            if not path.is_file():           # config not shipped in this checkout
                continue
            curve = self.presence_curve(str(path), cfg)
            for f in self.LRV1_RETURN_FRAMES:
                self.assertAlmostEqual(
                    curve[f], 1.0, places=5,
                    msg=("%s ramps on decisive return frame %d (presence %.5f): a "
                         "boundary sits within w of an evaluated frame" % (name, f, curve[f])))

    def test_the_correct_oracle_is_flat_across_the_whole_present_set(self):
        cfg = build_config()
        path = Path(__file__).resolve().parents[1] / "configs" / "lrv1" / "oracle_correct.json"
        if not path.is_file():
            self.skipTest("configs/lrv1/oracle_correct.json not present")
        curve = self.presence_curve(str(path), cfg)
        ramped = [f for f in sorted(self.LRV1_PRESENT_FRAMES) if curve[f] < 1.0 - 1e-5]
        self.assertEqual(ramped, [], "correct oracle ramps on ground-truth-present frames")
        # and the ramps must land inside the absence gap, where they cost only a
        # secondary diagnostic
        partial = [f for f in range(60) if 1e-6 < curve[f] < 1.0 - 1e-6]
        self.assertTrue(set(partial).isdisjoint(self.LRV1_PRESENT_FRAMES))

    def test_a_midpoint_boundary_is_detected_as_ramping(self):
        """Anti-vacuity: the check must FAIL on the construction it replaced."""
        cfg = build_config()
        with tempfile.TemporaryDirectory() as tmp:
            # the original construction: boundary at the frame-53/54 midpoint
            curve = self.presence_curve(write_spec(tmp, [[29.5 / 6.0, 53.5 / 6.0]]), cfg)
        self.assertLess(curve[54], 0.5, "fixture no longer reproduces the defect")
        self.assertAlmostEqual(curve[54], 0.15625, places=5)
        self.assertAlmostEqual(curve[55], 0.84375, places=5)


class PresenceRowExpansionTests(unittest.TestCase):
    """The repaired row expansion must be a pure speedup: identical values and
    identical gradients versus the per-row Python stack it replaced."""

    @staticmethod
    def reference(runtime, family_ids, timestamp):
        per_family = {}
        zero = torch.zeros((), device=runtime.device, dtype=runtime.dtype)
        for fid in torch.unique(family_ids).tolist():
            fid = int(fid)
            if fid < 0:
                per_family[fid] = zero
                continue
            record = runtime.registry.get(fid)
            if record.retired or record.interval.K == 0:
                per_family[fid] = zero
                continue
            r = runtime.realization(fid)
            t = torch.as_tensor(timestamp, device=runtime.device, dtype=runtime.dtype)
            per_family[fid] = episode_presence(t, r.b, r.d, runtime.config.w).sum()
        return torch.stack(
            [per_family[int(f)] for f in family_ids.tolist()]).reshape(-1, 1)

    def setUp(self):
        cfg = build_config()
        self.cfg = cfg
        omega = (cfg.T + cfg.w_m) - (-cfg.w_m)
        spanning = inverse(1, True, True, 0.0, [omega], [], 0.0, cfg, dtype=torch.float32)
        g0, g1 = 4.9166667, 8.9166667
        two = inverse(2, True, True, 0.0,
                      [g0 + cfg.w_m, (cfg.T + cfg.w_m) - g1], [g1 - g0], 0.0, cfg,
                      dtype=torch.float32)
        registry = FamilyRegistry()
        self.oracle_family = 3
        for i in range(12):
            registry.create_family(
                birth_time=0.0, birth_site=(0.0, 0.0, 0.0), lineage_key="cell-%d" % i,
                interval=(two if i == self.oracle_family else spanning),
                tau=((0.0, 0.0) if i == self.oracle_family else (0.0,)))
        self.runtime = ElgsRuntime(
            registry, cfg,
            ScheduleAnchors(seed_iteration=0, audit_iteration=1,
                            round_iterations=(2,), refit_until=3),
            device="cpu", dtype=torch.float32)

    def test_values_match_the_per_row_stack_including_unassigned_rows(self):
        torch.manual_seed(0)
        ids = torch.randint(-1, 12, (2_000,), dtype=torch.long)
        self.assertTrue(bool((ids < 0).any()), "fixture must exercise unassigned rows")
        for t in (0.0, 4.6, 4.9166667, 6.5, 8.9166667, 9.8333):
            got = self.runtime.presence_multiplier(ids, t)
            self.assertEqual(got.shape, (2_000, 1))
            self.assertTrue(torch.equal(got, self.reference(self.runtime, ids, t)),
                            "row expansion changed values at t=%r" % t)

    def test_rows_are_gathered_by_family_not_by_position(self):
        """Guards the searchsorted mapping: a permuted id column must permute
        the output identically."""
        ids = torch.tensor([5, self.oracle_family, 5, -1, self.oracle_family],
                           dtype=torch.long)
        t = 6.5                                     # inside the oracle absence gap
        got = self.runtime.presence_multiplier(ids, t).reshape(-1)
        self.assertAlmostEqual(float(got[0]), 1.0, places=5)
        self.assertAlmostEqual(float(got[1]), 0.0, places=5)   # absent
        self.assertAlmostEqual(float(got[2]), 1.0, places=5)
        self.assertAlmostEqual(float(got[3]), 0.0, places=5)   # unassigned
        self.assertAlmostEqual(float(got[4]), 0.0, places=5)

    def test_gradients_match_the_per_row_stack(self):
        torch.manual_seed(1)
        ids = torch.randint(0, 12, (2_000,), dtype=torch.long)
        t_edge = 4.9166667 - 0.5 * self.cfg.w
        logit = self.runtime._logits[self.oracle_family]

        logit.grad = None
        self.runtime.presence_multiplier(ids, t_edge).sum().backward()
        new = logit.grad.clone()

        logit.grad = None
        self.reference(self.runtime, ids, t_edge).sum().backward()
        ref = logit.grad.clone()

        self.assertGreater(float(ref.abs().max()), 0.0, "fixture must carry gradient")
        self.assertTrue(torch.equal(new, ref))


if __name__ == "__main__":
    unittest.main()
