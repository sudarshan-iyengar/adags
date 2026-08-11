"""Unit tests for elgs/acceptance.py (spec §7, errata E2).

CPU only, unittest. Oracles: the analytic weight bound, a closed-form
nu-mean for the empirical consistency check, exact pairing identities
for the bootstrap, enumerated slot-grid structure, and deliberately
degenerate fixtures. A dedicated fresh-context bootstrap reference
implementation joins tests/ref_impls with the remaining prereg work.
"""

import math
import pathlib
import random
import re
import unittest

from depth_visibility.errors import ContractError
from elgs.acceptance import (
    AcceptanceRecord,
    FrozenSamplerParams,
    SlotExhausted,
    SlotGrid,
    SnisSample,
    component_rank_order,
    crn_seed,
    decide,
    effective_sample_size,
    paired_cluster_bootstrap_se,
    paired_snis_delta,
    snis_weight,
)

LAMBDA_U = 0.5
NU = [0.1] * 10                       # uniform on {0..9}
PI_D = [0.55] + [0.05] * 9            # deliberately mismatched proposal
MIX = [LAMBDA_U * n + (1 - LAMBDA_U) * p for n, p in zip(NU, PI_D)]
NU_MEAN = 4.5                          # E_nu[l] for l(x) = x


def _draw_samples(n, seed, *, delta=0.0):
    """n CRN samples from the mixture; candidate loss = l(x) + delta,
    incumbent loss = l(x), unit = (camera, frame) spread over 8 units."""
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        x = rng.choices(range(10), weights=MIX)[0]
        samples.append(
            SnisSample(
                unit=(i % 4, float((i // 4) % 2)),
                nu_density=NU[x],
                mix_density=MIX[x],
                loss_incumbent=float(x),
                loss_candidate=float(x) + delta,
            )
        )
    return samples


class WeightBoundTests(unittest.TestCase):
    def test_weight_bound_holds_and_clip_inactive(self):
        for x in range(10):
            sample = SnisSample((0, 0.0), NU[x], MIX[x], 0.0, 0.0)
            w = snis_weight(sample, LAMBDA_U)
            self.assertLessEqual(w, 1.0 / LAMBDA_U)
            # The clip is provably inactive: the raw ratio already
            # respects the bound because m >= lambda_u * nu.
            self.assertAlmostEqual(w, NU[x] / MIX[x], places=15)

    def test_inconsistent_mixture_density_raises(self):
        # nu/m > 1/lambda_u is impossible under a correct mixture; a
        # caller feeding such densities has a bug and must hear it.
        bad = SnisSample((0, 0.0), 1.0, 0.3, 0.0, 0.0)
        with self.assertRaises(ContractError):
            snis_weight(bad, 0.5)

    def test_lambda_domain(self):
        sample = SnisSample((0, 0.0), 0.1, 0.1, 0.0, 0.0)
        with self.assertRaises(ContractError):
            snis_weight(sample, 0.0)
        with self.assertRaises(ContractError):
            snis_weight(sample, 1.5)


class ConsistencyTests(unittest.TestCase):
    def test_empirical_bias_shrinks_toward_closed_form_nu_mean(self):
        """SNIS is finite-sample biased but strongly consistent: the
        mean absolute error against the closed-form E_nu[l] = 4.5 must
        shrink with n, and be small at n = 256."""

        def mean_abs_error(n, trials=120):
            total = 0.0
            for t in range(trials):
                samples = _draw_samples(n, seed=1000 + t)
                r_hat = paired_snis_delta(
                    [
                        SnisSample(s.unit, s.nu_density, s.mix_density, 0.0, s.loss_candidate)
                        for s in samples
                    ],
                    LAMBDA_U,
                )
                total += abs(r_hat - NU_MEAN)
            return total / trials

        err_small = mean_abs_error(8)
        err_large = mean_abs_error(256)
        self.assertLess(err_large, err_small)
        self.assertLess(err_large, 0.25)

    def test_crn_determinism_bit_exact(self):
        a = paired_snis_delta(_draw_samples(64, seed=7, delta=-0.3), LAMBDA_U)
        b = paired_snis_delta(_draw_samples(64, seed=7, delta=-0.3), LAMBDA_U)
        self.assertEqual(a, b)

    def test_paired_delta_of_constant_shift_is_exact(self):
        """CRN pairing: candidate = incumbent + c => delta == c exactly
        for ANY sample set (shared weights cancel)."""
        samples = _draw_samples(50, seed=3, delta=-0.75)
        self.assertAlmostEqual(paired_snis_delta(samples, LAMBDA_U), -0.75, places=12)


class BootstrapTests(unittest.TestCase):
    def test_constant_shift_gives_zero_se(self):
        """Every paired replicate's delta equals c, so sd is zero up
        to float rounding of the two SNIS ratios (~1e-16 per
        replicate). An implementation that broke pairing (independent
        resamples per arm) would give SE on the order of the loss
        spread here — about eleven orders of magnitude larger."""
        samples = _draw_samples(48, seed=11, delta=0.4)
        se = paired_cluster_bootstrap_se(samples, LAMBDA_U, seed=99)
        self.assertLess(se, 1e-12)

    def test_se_scales_linearly_with_loss_scale(self):
        base = _draw_samples(48, seed=5)
        varied = [
            SnisSample(s.unit, s.nu_density, s.mix_density,
                       s.loss_incumbent, s.loss_incumbent + (i % 3) * 0.2)
            for i, s in enumerate(base)
        ]
        doubled = [
            SnisSample(s.unit, s.nu_density, s.mix_density,
                       2 * s.loss_incumbent, 2 * (s.loss_incumbent + (i % 3) * 0.2))
            for i, s in enumerate(base)
        ]
        se1 = paired_cluster_bootstrap_se(varied, LAMBDA_U, seed=42)
        se2 = paired_cluster_bootstrap_se(doubled, LAMBDA_U, seed=42)
        self.assertGreater(se1, 0.0)
        self.assertAlmostEqual(se2, 2.0 * se1, places=10)

    def test_degeneracy_rejection_at_five_clusters(self):
        samples = [
            SnisSample((c, 0.0), 0.1, MIX[1], 1.0, 1.5) for c in range(5)
        ]
        with self.assertRaises(ContractError):
            paired_cluster_bootstrap_se(samples, LAMBDA_U, seed=1)
        six = [
            SnisSample((c, 0.0), 0.1, MIX[1], 1.0, 1.0 + 0.1 * c) for c in range(6)
        ]
        self.assertTrue(
            math.isfinite(paired_cluster_bootstrap_se(six, LAMBDA_U, seed=1))
        )

    def test_bootstrap_deterministic_under_seed(self):
        samples = _draw_samples(48, seed=5, delta=0.1)
        varied = [
            SnisSample(s.unit, s.nu_density, s.mix_density,
                       s.loss_incumbent, s.loss_candidate + (i % 5) * 0.1)
            for i, s in enumerate(samples)
        ]
        a = paired_cluster_bootstrap_se(varied, LAMBDA_U, seed=13)
        b = paired_cluster_bootstrap_se(varied, LAMBDA_U, seed=13)
        self.assertEqual(a, b)


class DecisionTests(unittest.TestCase):
    def _params(self, frozen=True):
        return FrozenSamplerParams(lambda_u=LAMBDA_U, pi_d_identity="pi-d-v1", frozen=frozen)

    def test_freeze_required_before_any_confirmation(self):
        samples = _draw_samples(48, seed=2)
        with self.assertRaises(ContractError):
            decide(samples, self._params(frozen=False), exact_deltas=0.0,
                   transaction_increment=0.0, k=1.0, bootstrap_seed=1)

    def test_accept_rule_includes_exact_and_transaction_terms(self):
        samples = _draw_samples(60, seed=8, delta=-0.5)  # render improves
        params = self._params()
        rec = decide(samples, params, exact_deltas=0.0,
                     transaction_increment=0.0, k=1.0, bootstrap_seed=3)
        self.assertIsInstance(rec, AcceptanceRecord)
        self.assertAlmostEqual(rec.delta_render, -0.5, places=10)
        self.assertTrue(rec.accepted)  # SE == 0 for a constant shift
        # A transaction charge can flip the verdict.
        rec2 = decide(samples, params, exact_deltas=0.2,
                      transaction_increment=0.4, k=1.0, bootstrap_seed=3)
        self.assertAlmostEqual(rec2.delta_total, 0.1, places=10)
        self.assertFalse(rec2.accepted)

    def test_record_carries_n_ess_units(self):
        samples = _draw_samples(60, seed=8, delta=-0.5)
        rec = decide(samples, self._params(), exact_deltas=0.0,
                     transaction_increment=0.0, k=2.0, bootstrap_seed=3)
        self.assertEqual(rec.n_samples, 60)
        self.assertEqual(rec.n_units, 8)
        self.assertGreater(rec.ess, 0.0)
        self.assertLessEqual(rec.ess, 60.0)
        self.assertAlmostEqual(
            rec.ess, effective_sample_size(samples, LAMBDA_U), places=12
        )


class SlotGridTests(unittest.TestCase):
    def _grid(self):
        pool = tuple((c, float(t)) for c in range(4) for t in range(6))
        return SlotGrid(n_rounds=2, n_passes=2, slots_per_pass=3,
                        units_per_slot=2, reserved_pool=pool)

    def test_injective_slot_mapping(self):
        grid = self._grid()
        seen = set()
        for r in range(2):
            for p in range(2):
                for k in range(3):
                    idx = grid.slot_index(r, p, k)
                    self.assertNotIn(idx, seen)
                    seen.add(idx)

    def test_draw_disjoint_and_single_use(self):
        grid = self._grid()
        a = grid.draw(0, 0, 0)
        b = grid.draw(0, 0, 1)
        self.assertFalse(set(a) & set(b))
        with self.assertRaises(ContractError):
            grid.draw(0, 0, 0)

    def test_exhaustion_rejects_and_is_typed(self):
        grid = self._grid()
        with self.assertRaises(SlotExhausted):
            grid.slot_index(0, 0, 3)

    def test_reserved_units_exposed_for_sampler_exclusion(self):
        grid = self._grid()
        self.assertEqual(len(grid.reserved_units()), 24)
        self.assertIn((0, 0.0), grid.reserved_units())

    def test_pool_too_small_or_duplicated_rejected(self):
        pool = tuple((0, float(t)) for t in range(5))
        with self.assertRaises(ContractError):
            SlotGrid(2, 2, 3, 2, pool)
        dup = tuple((0, 0.0) for _ in range(24))
        with self.assertRaises(ContractError):
            SlotGrid(2, 2, 3, 2, dup)

    def test_component_order_and_crn_seed_determinism(self):
        self.assertEqual(component_rank_order([9, 2, 5]), (2, 5, 9))
        with self.assertRaises(ContractError):
            component_rank_order([2, 2])
        self.assertEqual(crn_seed(1, 2, 3, 4), crn_seed(1, 2, 3, 4))
        self.assertNotEqual(crn_seed(1, 2, 3, 4), crn_seed(1, 2, 3, 5))


class BootstrapOracleParityTests(unittest.TestCase):
    """elgs/acceptance.py vs the FROZEN fresh-context §7 oracle
    (tests/ref_impls/bootstrap_reference.py). Both transcriptions
    sort unit keys and draw len(units) indices per replicate with
    random.Random(seed).randrange, so the REALIZED delta and SE must
    agree at the same seed (oracle ambiguities A1-A4 are pinned the
    same way on both sides: sample sd ddof=1, sorted units)."""

    @classmethod
    def setUpClass(cls):
        from tests.ref_impls import bootstrap_reference as ref

        cls.ref = ref

    def _pairs(self, n=48, delta_fn=lambda i: -0.5):
        mine, theirs = [], []
        for i in range(n):
            unit = (i % 4, float((i // 4) % 2))
            nu, m = 0.1, LAMBDA_U * 0.1 + (1 - LAMBDA_U) * 0.08
            inc = 1.0 + 0.03 * (i % 7)
            cand = inc + delta_fn(i)
            mine.append(SnisSample(unit, nu, m, inc, cand))
            theirs.append(self.ref.Sample(unit, nu, m, inc, cand))
        return mine, theirs

    def test_weight_and_delta_parity(self):
        mine, theirs = self._pairs()
        for m_s, t_s in zip(mine, theirs):
            self.assertAlmostEqual(
                snis_weight(m_s, LAMBDA_U),
                self.ref.snis_weight(t_s, LAMBDA_U),
                places=15,
            )
        self.assertAlmostEqual(
            paired_snis_delta(mine, LAMBDA_U),
            self.ref.paired_delta(theirs, LAMBDA_U),
            places=13,
        )

    def test_bootstrap_se_parity_at_same_seed(self):
        mine, theirs = self._pairs(delta_fn=lambda i: -0.3 + 0.1 * (i % 5))
        se_mine = paired_cluster_bootstrap_se(mine, LAMBDA_U, seed=17)
        se_ref = self.ref.paired_cluster_bootstrap_se(theirs, LAMBDA_U, seed=17)
        self.assertAlmostEqual(se_mine, se_ref, places=13)
        self.assertGreater(se_mine, 0.0)

    def test_degeneracy_parity(self):
        mine, theirs = self._pairs(n=5)
        # 5 samples over <= 5 units: both sides reject.
        with self.assertRaises(ContractError):
            paired_cluster_bootstrap_se(mine, LAMBDA_U, seed=1)
        with self.assertRaises(ValueError):
            self.ref.paired_cluster_bootstrap_se(theirs, LAMBDA_U, seed=1)

    def test_accept_rule_parity_strictness(self):
        self.assertTrue(self.ref.accept(-1e-15, 0.0, 1.0))
        self.assertFalse(self.ref.accept(0.0, 0.0, 1.0))
        # Mine: decide() realizes the same strict rule; spot-check the
        # boundary via a constant-shift fixture with zero SE.
        samples = _draw_samples(60, seed=8, delta=0.0)
        params = FrozenSamplerParams(lambda_u=LAMBDA_U, pi_d_identity="pi", frozen=True)
        rec = decide(samples, params, exact_deltas=0.0,
                     transaction_increment=0.0, k=1.0, bootstrap_seed=3)
        self.assertFalse(rec.accepted)  # delta_total == 0 is NOT < 0


class EstimatorLanguageTests(unittest.TestCase):
    def test_no_unbiased_or_exact_estimator_claims_in_elgs(self):
        """Errata E2: no code or comment may describe the sampled
        estimate as unbiased or exact. 'unbiased' may appear only in
        an explicit negation on the same line."""
        elgs_dir = pathlib.Path(__file__).resolve().parent.parent / "elgs"
        negation = re.compile(r"\b(not|no|never)\b[^.\n]*\bunbiased", re.IGNORECASE)
        reverse_negation = re.compile(r"\bunbiasedness\b", re.IGNORECASE)
        for path in sorted(elgs_dir.glob("*.py")):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                lowered = line.lower()
                if "unbiased" in lowered:
                    self.assertTrue(
                        negation.search(line) or reverse_negation.search(line),
                        f"{path.name}:{lineno} mentions 'unbiased' without negation: {line!r}",
                    )
                self.assertNotIn("exact estimator", lowered, f"{path.name}:{lineno}")
                self.assertNotIn("estimator exact", lowered, f"{path.name}:{lineno}")
                self.assertNotIn("estimator is exact", lowered, f"{path.name}:{lineno}")


if __name__ == "__main__":
    unittest.main()
