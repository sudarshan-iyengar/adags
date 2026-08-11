"""Unit tests for elgs/intervals.py against formal spec rev 4, §1.

CPU only, unittest, no CUDA extensions. Oracles are hand-computed
literals and spec-stated identities, not round-trips alone: the
forward-map fixtures below were derived by hand from the §1 formulas
so a shared misreading of forward+inverse cannot pass silently.

Fixture constants: T=10, w_m=1, w=1, floor_len=2, floor_gap=2
=> Omega = 12; Omega_free(K) = 12 - 2K - 2(K-1) = 14 - 4K.
"""

import math
import unittest

import torch

from depth_visibility.errors import ContractError, SchemaError
from elgs.intervals import (
    INTERVAL_STATE_SCHEMA,
    IntervalConfig,
    IntervalState,
    birth_interval_state,
    coordinate_labels,
    deserialize_state,
    empty_program,
    expected_dim,
    forward,
    inverse,
    serialize_state,
)

CFG = IntervalConfig(T=10.0, w_m=1.0, w=1.0, floor_len=2.0, floor_gap=2.0)

# K=4 needs omega_free = T + 2*w_m - 4*floor_len - 3*floor_gap > 0,
# which T=10 deliberately does not satisfy (that inadmissibility is
# itself tested); the K=1..4 sweeps use this wider span.
CFG_WIDE = IntervalConfig(T=30.0, w_m=1.0, w=1.0, floor_len=2.0, floor_gap=2.0)

ALL_PATTERNS = ((False, False), (True, False), (False, True), (True, True))


def _state(K, latch_pre, latch_post, a_values, dtype=torch.float64):
    return IntervalState(
        K=K,
        latch_pre=latch_pre,
        latch_post=latch_post,
        a=torch.tensor(a_values, dtype=dtype),
    )


class ForwardMapLiteralTests(unittest.TestCase):
    """Hand-computed endpoint values, one fixture per latch pattern."""

    def test_pattern_00_k2_uniform_logits(self):
        # dims = 5: [slack_pre, len_1, gap_1, len_2, slack_post];
        # Omega_free = 6, sigma = 1/5 each => span 1.2 per coordinate.
        r = forward(_state(2, False, False, [0.0] * 5), CFG)
        self.assertAlmostEqual(float(r.slack_pre), 1.2, places=12)
        self.assertAlmostEqual(float(r.lens[0]), 3.2, places=12)
        self.assertAlmostEqual(float(r.gaps[0]), 3.2, places=12)
        self.assertAlmostEqual(float(r.lens[1]), 3.2, places=12)
        self.assertAlmostEqual(float(r.slack_post), 1.2, places=12)
        self.assertAlmostEqual(float(r.b[0]), -1.0 + 1.2, places=12)
        self.assertAlmostEqual(float(r.d[0]), 0.2 + 3.2, places=12)
        self.assertAlmostEqual(float(r.b[1]), 3.4 + 3.2, places=12)
        self.assertAlmostEqual(float(r.d[1]), 6.6 + 3.2, places=12)

    def test_pattern_11_k2_uniform_logits(self):
        # dims = 3: [len_1, gap_1, len_2]; Omega_free = 6, sigma = 1/3
        # => len/gap = 2 + 2 = 4; b_1 = -w_m exactly, d_2 = T + w_m.
        r = forward(_state(2, True, True, [0.0] * 3), CFG)
        self.assertEqual(float(r.slack_pre), 0.0)
        self.assertEqual(float(r.slack_post), 0.0)
        self.assertAlmostEqual(float(r.lens[0]), 4.0, places=12)
        self.assertAlmostEqual(float(r.gaps[0]), 4.0, places=12)
        self.assertAlmostEqual(float(r.b[0]), -1.0, places=12)
        self.assertAlmostEqual(float(r.d[0]), 3.0, places=12)
        self.assertAlmostEqual(float(r.b[1]), 7.0, places=12)
        self.assertAlmostEqual(float(r.d[1]), 11.0, places=12)

    def test_pattern_10_k1_uniform_logits(self):
        # dims = 2: [len_1, slack_post]; Omega_free = 10, sigma = 1/2
        # => len_1 = 7, slack_post = 5; b_1 = -1 latched.
        r = forward(_state(1, True, False, [0.0, 0.0]), CFG)
        self.assertEqual(float(r.slack_pre), 0.0)
        self.assertAlmostEqual(float(r.lens[0]), 7.0, places=12)
        self.assertAlmostEqual(float(r.slack_post), 5.0, places=12)
        self.assertAlmostEqual(float(r.b[0]), -1.0, places=12)
        self.assertAlmostEqual(float(r.d[0]), 6.0, places=12)

    def test_pattern_01_k1_uniform_logits(self):
        # dims = 2: [slack_pre, len_1]; Omega_free = 10, sigma = 1/2
        # => slack_pre = 5, len_1 = 7; d_1 = T + w_m exactly.
        r = forward(_state(1, False, True, [0.0, 0.0]), CFG)
        self.assertAlmostEqual(float(r.slack_pre), 5.0, places=12)
        self.assertAlmostEqual(float(r.lens[0]), 7.0, places=12)
        self.assertEqual(float(r.slack_post), 0.0)
        self.assertAlmostEqual(float(r.b[0]), 4.0, places=12)
        self.assertAlmostEqual(float(r.d[0]), 11.0, places=12)


class DimensionAndOrderTests(unittest.TestCase):
    def test_expected_dims_all_patterns(self):
        for K in range(1, 5):
            for lp, lo in ALL_PATTERNS:
                self.assertEqual(expected_dim(K, lp, lo), 2 * K + 1 - (lp + lo))

    def test_canonical_coordinate_order_literal(self):
        self.assertEqual(
            coordinate_labels(3, False, False),
            ("slack_pre", "len_1", "gap_1", "len_2", "gap_2", "len_3", "slack_post"),
        )
        self.assertEqual(coordinate_labels(1, True, True), ("len_1",))
        self.assertEqual(coordinate_labels(2, True, False), ("len_1", "gap_1", "len_2", "slack_post"))
        self.assertEqual(coordinate_labels(2, False, True), ("slack_pre", "len_1", "gap_1", "len_2"))

    def test_wrong_dimension_rejected(self):
        for K in range(1, 5):
            for lp, lo in ALL_PATTERNS:
                bad = expected_dim(K, lp, lo) + 1
                with self.assertRaises(ContractError):
                    _state(K, lp, lo, [0.0] * bad)

    def test_k_out_of_range_rejected(self):
        with self.assertRaises(ContractError):
            _state(5, False, False, [0.0] * 11)
        with self.assertRaises(ContractError):
            expected_dim(-1, False, False)

    def test_non_finite_vector_rejected(self):
        with self.assertRaises(ContractError):
            _state(1, False, False, [0.0, float("nan"), 0.0])

    def test_structurally_inadmissible_k_rejected_at_forward(self):
        # T=10 config: omega_free(4) = 14 - 16 = -2 <= 0.
        with self.assertRaises(ContractError):
            forward(_state(4, False, False, [0.0] * 9), CFG)
        with self.assertRaises(ContractError):
            CFG.require_admissible_k(4)


class OmegaIdentityTests(unittest.TestCase):
    """slack_pre + sum(len) + sum(gap) + slack_post == Omega, every pattern."""

    def test_omega_sum_identity_random(self):
        gen = torch.Generator().manual_seed(20260811)
        for K in range(1, 5):
            for lp, lo in ALL_PATTERNS:
                for _ in range(25):
                    a = torch.randn(expected_dim(K, lp, lo), generator=gen, dtype=torch.float64) * 3
                    r = forward(IntervalState(K=K, latch_pre=lp, latch_post=lo, a=a), CFG_WIDE)
                    total = float(r.slack_pre + r.lens.sum() + r.gaps.sum() + r.slack_post)
                    self.assertAlmostEqual(total, CFG_WIDE.omega, places=9)

    def test_final_endpoint_bounded_identically(self):
        """d_K <= T + w_m under arbitrary logits; equality iff latch_post."""
        gen = torch.Generator().manual_seed(7)
        for K in range(1, 5):
            for lp, lo in ALL_PATTERNS:
                for _ in range(25):
                    a = torch.randn(expected_dim(K, lp, lo), generator=gen, dtype=torch.float64) * 5
                    r = forward(IntervalState(K=K, latch_pre=lp, latch_post=lo, a=a), CFG_WIDE)
                    d_K = float(r.d[-1])
                    if lo:
                        self.assertAlmostEqual(d_K, CFG_WIDE.T + CFG_WIDE.w_m, places=9)
                    else:
                        self.assertLess(d_K, CFG_WIDE.T + CFG_WIDE.w_m)
                    if lp:
                        self.assertAlmostEqual(float(r.b[0]), -CFG_WIDE.w_m, places=12)
                    else:
                        self.assertGreater(float(r.b[0]), -CFG_WIDE.w_m)

    def test_strict_positivity_of_unlatched_spans(self):
        """Softmax coords strictly positive: slacks > 0, len > floor, gap > floor."""
        gen = torch.Generator().manual_seed(99)
        for K in range(1, 5):
            for lp, lo in ALL_PATTERNS:
                a = torch.randn(expected_dim(K, lp, lo), generator=gen, dtype=torch.float64) * 8
                r = forward(IntervalState(K=K, latch_pre=lp, latch_post=lo, a=a), CFG_WIDE)
                if not lp:
                    self.assertGreater(float(r.slack_pre), 0.0)
                if not lo:
                    self.assertGreater(float(r.slack_post), 0.0)
                self.assertTrue(bool((r.lens > CFG.floor_len).all()))
                if K > 1:
                    self.assertTrue(bool((r.gaps > CFG.floor_gap).all()))


class InverseMapTests(unittest.TestCase):
    def test_gauge_max_logit_exactly_zero(self):
        state = inverse(
            2, False, False,
            slack_pre=0.5, lens=[3.0, 2.5], gaps=[4.0], slack_post=2.0,
            config=CFG, dtype=torch.float64,
        )
        assert state.a is not None
        self.assertEqual(float(state.a.max()), 0.0)

    def test_forward_inverse_round_trip_all_patterns(self):
        gen = torch.Generator().manual_seed(31415)
        for K in range(1, 5):
            for lp, lo in ALL_PATTERNS:
                for _ in range(10):
                    a = torch.randn(expected_dim(K, lp, lo), generator=gen, dtype=torch.float64) * 2
                    src = IntervalState(K=K, latch_pre=lp, latch_post=lo, a=a)
                    r = forward(src, CFG_WIDE)
                    rebuilt = inverse(
                        K, lp, lo,
                        slack_pre=float(r.slack_pre),
                        lens=[float(v) for v in r.lens],
                        gaps=[float(v) for v in r.gaps],
                        slack_post=float(r.slack_post),
                        config=CFG_WIDE,
                        dtype=torch.float64,
                    )
                    r2 = forward(rebuilt, CFG_WIDE)
                    self.assertTrue(torch.allclose(r.lens, r2.lens, atol=1e-9))
                    self.assertTrue(torch.allclose(r.gaps, r2.gaps, atol=1e-9))
                    self.assertTrue(torch.allclose(r.b, r2.b, atol=1e-9))
                    self.assertTrue(torch.allclose(r.d, r2.d, atol=1e-9))

    def test_exact_floor_target_rejected(self):
        with self.assertRaises(ContractError):
            inverse(1, False, False, slack_pre=1.0,
                    lens=[CFG.floor_len], gaps=[], slack_post=9.0, config=CFG)

    def test_zero_unlatched_slack_target_rejected(self):
        # An exactly-zero outer slack must set the latch, never a coordinate.
        with self.assertRaises(ContractError):
            inverse(1, False, False, slack_pre=0.0,
                    lens=[7.0], gaps=[], slack_post=5.0, config=CFG)

    def test_nonzero_latched_slack_rejected(self):
        with self.assertRaises(ContractError):
            inverse(1, True, False, slack_pre=0.5,
                    lens=[7.0], gaps=[], slack_post=4.5, config=CFG)

    def test_omega_sum_violation_rejected(self):
        with self.assertRaises(ContractError):
            inverse(1, False, False, slack_pre=1.0,
                    lens=[5.0], gaps=[], slack_post=1.0, config=CFG)


class BirthEncodingTests(unittest.TestCase):
    def test_birth_literal_fixture(self):
        # t_birth = 3: slack_pre = 4, len_1 = 8, sigma = (0.4, 0.6),
        # a = (log 0.4 - log 0.6, 0) -- terminal latch, dims = 2.
        state = birth_interval_state(3.0, CFG, dtype=torch.float64)
        self.assertEqual(state.K, 1)
        self.assertFalse(state.latch_pre)
        self.assertTrue(state.latch_post)
        assert state.a is not None
        self.assertEqual(state.a.numel(), 2)
        self.assertAlmostEqual(float(state.a[0]), math.log(0.4) - math.log(0.6), places=12)
        self.assertEqual(float(state.a[1]), 0.0)
        r = forward(state, CFG)
        self.assertAlmostEqual(float(r.slack_pre), 4.0, places=9)
        self.assertAlmostEqual(float(r.lens[0]), 8.0, places=9)
        self.assertAlmostEqual(float(r.d[0]), CFG.T + CFG.w_m, places=9)

    def test_birth_admissibility_iff(self):
        with self.assertRaises(ContractError):
            birth_interval_state(-CFG.w_m, CFG)  # slack_pre = 0 exactly
        with self.assertRaises(ContractError):
            birth_interval_state(-CFG.w_m - 0.5, CFG)
        with self.assertRaises(ContractError):
            # len_1 = T + w_m - t_birth <= floor_len
            birth_interval_state(CFG.T + CFG.w_m - CFG.floor_len, CFG)


class EmptyProgramTests(unittest.TestCase):
    def test_empty_program_state(self):
        state = empty_program()
        self.assertEqual(state.K, 0)
        self.assertIsNone(state.latch_pre)
        self.assertIsNone(state.a)
        with self.assertRaises(ContractError):
            forward(state, CFG)
        with self.assertRaises(ContractError):
            expected_dim(0, False, False)

    def test_k0_with_latches_or_vector_rejected(self):
        with self.assertRaises(ContractError):
            IntervalState(K=0, latch_pre=False, latch_post=None, a=None)
        with self.assertRaises(ContractError):
            IntervalState(K=0, latch_pre=None, latch_post=None, a=torch.zeros(1))


class SerializationTests(unittest.TestCase):
    def test_round_trip_all_patterns(self):
        gen = torch.Generator().manual_seed(2718)
        for K in range(1, 5):
            for lp, lo in ALL_PATTERNS:
                a = torch.randn(expected_dim(K, lp, lo), generator=gen)
                src = IntervalState(K=K, latch_pre=lp, latch_post=lo, a=a)
                payload = serialize_state(src)
                self.assertEqual(payload["schema_version"], INTERVAL_STATE_SCHEMA)
                back = deserialize_state(payload)
                assert back.a is not None
                self.assertEqual(back.K, K)
                self.assertEqual(back.latch_pre, lp)
                self.assertEqual(back.latch_post, lo)
                self.assertTrue(torch.equal(back.a, a))

    def test_serialized_order_is_canonical(self):
        # Distinct values per coordinate so a permutation cannot pass.
        state = inverse(
            2, False, False,
            slack_pre=0.5, lens=[2.5, 3.5], gaps=[4.5], slack_post=1.0,
            config=CFG, dtype=torch.float64,
        )
        payload = serialize_state(state)
        labels = coordinate_labels(2, False, False)
        self.assertEqual(len(payload["a"]), len(labels))
        r = forward(deserialize_state(payload, dtype=torch.float64), CFG)
        self.assertAlmostEqual(float(r.slack_pre), 0.5, places=9)
        self.assertAlmostEqual(float(r.lens[0]), 2.5, places=9)
        self.assertAlmostEqual(float(r.gaps[0]), 4.5, places=9)
        self.assertAlmostEqual(float(r.lens[1]), 3.5, places=9)
        self.assertAlmostEqual(float(r.slack_post), 1.0, places=9)

    def test_k0_round_trip(self):
        payload = serialize_state(empty_program())
        self.assertEqual(payload, {"schema_version": INTERVAL_STATE_SCHEMA, "K": 0})
        back = deserialize_state(payload)
        self.assertEqual(back.K, 0)

    def test_loader_rejects_corrupt_payloads(self):
        good = serialize_state(_state(2, False, True, [0.0, 0.1, -0.2, 0.3]))
        bad_dim = dict(good, a=good["a"][:-1])
        with self.assertRaises(ContractError):
            deserialize_state(bad_dim)
        with self.assertRaises(SchemaError):
            deserialize_state(dict(good, schema_version="wrong"))
        with self.assertRaises(SchemaError):
            deserialize_state(dict(good, latch_pre=1))
        with self.assertRaises(SchemaError):
            deserialize_state({"schema_version": INTERVAL_STATE_SCHEMA, "K": 0, "a": []})
        missing = dict(good)
        del missing["a"]
        with self.assertRaises(SchemaError):
            deserialize_state(missing)


class DifferentiabilityTests(unittest.TestCase):
    def test_forward_map_is_differentiable_in_a(self):
        a = torch.zeros(5, dtype=torch.float64, requires_grad=True)
        r = forward(IntervalState(K=2, latch_pre=False, latch_post=False, a=a), CFG)
        (r.d[-1] - r.b[0]).backward()
        assert a.grad is not None
        self.assertTrue(torch.isfinite(a.grad).all())


class PresenceFunctionTests(unittest.TestCase):
    """Substrate pi/z/X/winner semantics (spec §1 + lgs-method).

    Analytic fixture: single episode b=2, d=8, w=1 => support exactly
    [2, 8], plateau [3, 7], S(0.5) = 3*0.25 - 2*0.125 = 0.5.
    """

    def _realization(self, b_d_pairs):
        from elgs.intervals import IntervalRealization

        b = torch.tensor([p[0] for p in b_d_pairs], dtype=torch.float64)
        d = torch.tensor([p[1] for p in b_d_pairs], dtype=torch.float64)
        return IntervalRealization(
            slack_pre=torch.tensor(0.0),
            slack_post=torch.tensor(0.0),
            lens=d - b,
            gaps=b.new_zeros((max(len(b_d_pairs) - 1, 0),)),
            b=b,
            d=d,
        )

    def test_presence_analytic_values(self):
        from elgs.presence import family_presence

        r = self._realization([(2.0, 8.0)])
        self.assertEqual(float(family_presence(r, 1.5, w=1.0)), 0.0)  # exact zero outside
        self.assertAlmostEqual(float(family_presence(r, 2.5, w=1.0)), 0.5, places=12)
        self.assertEqual(float(family_presence(r, 5.0, w=1.0)), 1.0)  # exact plateau
        self.assertEqual(float(family_presence(r, 8.0, w=1.0)), 0.0)  # S(0) = 0 at the edge
        self.assertEqual(float(family_presence(r, 9.0, w=1.0)), 0.0)

    def test_exact_zero_in_gaps(self):
        from elgs.presence import family_presence, winner_index

        r = self._realization([(0.0, 4.0), (9.0, 13.0)])
        for t in (4.5, 6.0, 8.5):
            self.assertIsNone(winner_index(r, t))
            self.assertEqual(float(family_presence(r, t, w=1.0)), 0.0)

    def test_winner_unique_and_boundaries(self):
        from elgs.presence import winner_index

        r = self._realization([(0.0, 4.0), (9.0, 13.0)])
        self.assertEqual(winner_index(r, 0.0), 0)
        self.assertEqual(winner_index(r, 4.0), 0)
        self.assertEqual(winner_index(r, 10.0), 1)
        overlapping = self._realization([(0.0, 5.0), (4.0, 9.0)])
        with self.assertRaises(ContractError):
            winner_index(overlapping, 4.5)

    def test_plateau_z_and_edge_band_strictness(self):
        from elgs.presence import in_edge_band, plateau_z

        r = self._realization([(2.0, 8.0)])
        self.assertTrue(plateau_z(r, 3.0, w=1.0))
        self.assertTrue(plateau_z(r, 7.0, w=1.0))
        self.assertFalse(plateau_z(r, 2.5, w=1.0))
        self.assertTrue(in_edge_band(r, 2.5, w=1.0))
        self.assertFalse(in_edge_band(r, 3.0, w=1.0))  # distance exactly w: strict
        self.assertTrue(in_edge_band(r, 7.5, w=1.0))
        self.assertFalse(in_edge_band(r, 5.0, w=1.0))

    def test_no_mid_episode_dip(self):
        """Latched shape: pi is 1 on the whole plateau, monotone on edges."""
        from elgs.presence import family_presence

        r = self._realization([(2.0, 8.0)])
        values = [float(family_presence(r, t, w=1.0)) for t in
                  [2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 6.0, 7.0]]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(earlier, later + 1e-12)
        self.assertEqual(values[4], 1.0)
        self.assertEqual(min(values[4:]), 1.0)

    def test_presence_differentiable_through_endpoints(self):
        from elgs.presence import episode_presence

        b = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        d = torch.tensor(8.0, dtype=torch.float64, requires_grad=True)
        pi = episode_presence(torch.tensor(2.5, dtype=torch.float64), b, d, w=1.0)
        pi.backward()
        assert b.grad is not None and d.grad is not None
        self.assertNotEqual(float(b.grad), 0.0)  # on the rising edge
        self.assertEqual(float(d.grad), 0.0)  # far side saturated


if __name__ == "__main__":
    unittest.main()
