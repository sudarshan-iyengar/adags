"""Unit tests for elgs/evidence.py (spec §2) — heads, L1/L0, equality.

CPU only, unittest. The independently-authored frozen oracle in
tests/ref_impls/likelihood_reference.py is compared against this
implementation by a dedicated parity test class appended once the
oracle lands (oracle-independence rule of the implementation plan).
"""

import math
import unittest

from depth_visibility.errors import ContractError
from elgs.evidence import (
    EvidenceHeads,
    EvidenceParams,
    Report,
    likelihood_l0,
    likelihood_l1,
    p_cens,
    p_out,
    p_vis,
    truncated_gaussian_pos_density,
)

PARAMS = EvidenceParams(
    pi_miss_vis=0.2,
    pi_miss_cens=0.6,
    pi_miss_out=0.4,
    d_img_x0=0.0,
    d_img_x1=10.0,
    d_img_y0=0.0,
    d_img_y1=8.0,
    g_pos_sigma=1.5,
    v_min=0.0,
    v_max=1.0,
    r_min=0.05,
    h_floor=0.01,
    pi_floor=0.01,
    g_cap=50.0,
    pos_cap=50.0,
)

HEADS = EvidenceHeads(
    params=PARAMS,
    g_v=lambda v: 2.0 * v if v > 0 else 0.0,  # normalized on [0,1]
    h_c=lambda v: 1.0,  # uniform on [0,1]
    h_o=lambda v: 1.0,
)

MISS = Report(is_miss=True)
HIT = Report(is_miss=False, v=0.7, x=4.0, y=3.0)
BRIDGE = (4.5, 3.5)


class HeadTests(unittest.TestCase):
    def test_miss_masses(self):
        self.assertEqual(p_vis(MISS, BRIDGE, HEADS.g_v, PARAMS), PARAMS.pi_miss_vis)
        self.assertEqual(p_cens(MISS, HEADS.h_c, PARAMS), PARAMS.pi_miss_cens)
        self.assertEqual(p_out(MISS, HEADS.h_o, PARAMS), PARAMS.pi_miss_out)

    def test_g_pos_normalizes_over_d_img(self):
        # Riemann sum over D_img must be ~1 for a center inside the domain.
        n = 200
        dx = (PARAMS.d_img_x1 - PARAMS.d_img_x0) / n
        dy = (PARAMS.d_img_y1 - PARAMS.d_img_y0) / n
        total = 0.0
        for i in range(n):
            for j in range(n):
                total += truncated_gaussian_pos_density(
                    PARAMS.d_img_x0 + (i + 0.5) * dx,
                    PARAMS.d_img_y0 + (j + 0.5) * dy,
                    BRIDGE[0],
                    BRIDGE[1],
                    PARAMS,
                ) * dx * dy
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_g_pos_normalizes_near_the_boundary_center(self):
        n = 300
        dx = (PARAMS.d_img_x1 - PARAMS.d_img_x0) / n
        dy = (PARAMS.d_img_y1 - PARAMS.d_img_y0) / n
        total = 0.0
        for i in range(n):
            for j in range(n):
                total += truncated_gaussian_pos_density(
                    PARAMS.d_img_x0 + (i + 0.5) * dx,
                    PARAMS.d_img_y0 + (j + 0.5) * dy,
                    0.2,
                    7.9,
                    PARAMS,
                ) * dx * dy
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_positional_censored_density_is_uniform(self):
        value = p_cens(HIT, HEADS.h_c, PARAMS)
        expected = (1.0 - PARAMS.pi_miss_cens) * 1.0 / PARAMS.d_img_area
        self.assertAlmostEqual(value, expected, places=15)

    def test_domain_violations_rejected(self):
        with self.assertRaises(ContractError):
            p_cens(Report(is_miss=False, v=2.0, x=1.0, y=1.0), HEADS.h_c, PARAMS)
        with self.assertRaises(ContractError):
            p_cens(Report(is_miss=False, v=0.5, x=-1.0, y=1.0), HEADS.h_c, PARAMS)

    def test_floor_and_cap_violations_rejected(self):
        with self.assertRaises(ContractError):
            p_cens(HIT, lambda v: 0.0, PARAMS)  # below h_floor
        with self.assertRaises(ContractError):
            p_vis(HIT, BRIDGE, lambda v: 100.0, PARAMS)  # above g_cap
        with self.assertRaises(ContractError):
            EvidenceParams(**{**PARAMS.__dict__, "pi_miss_cens": 0.999})

    def test_far_center_normalization_degenerate_raises(self):
        with self.assertRaises(ContractError):
            truncated_gaussian_pos_density(1.0, 1.0, 1e6, 1e6, PARAMS)


class LikelihoodTests(unittest.TestCase):
    def test_censoring_equality_exact(self):
        """L1 at q_tilde = 0 is IDENTICALLY L0 — exact equality, both
        report kinds (PROP 1 depends on this)."""
        for report in (MISS, HIT):
            l1 = likelihood_l1(report, BRIDGE, 0.0, 0.8, HEADS)
            l0 = likelihood_l0(report, 0.8, HEADS)
            self.assertEqual(l1, l0)

    def test_l1_interpolates_between_cens_and_vis(self):
        vis = p_vis(HIT, BRIDGE, HEADS.g_v, PARAMS)
        cen = p_cens(HIT, HEADS.h_c, PARAMS)
        out = p_out(HIT, HEADS.h_o, PARAMS)
        r = 0.9
        for q in (0.0, 0.25, 0.5, 1.0):
            expected = r * (q * vis + (1 - q) * cen) + (1 - r) * out
            self.assertAlmostEqual(
                likelihood_l1(HIT, BRIDGE, q, r, HEADS), expected, places=15
            )

    def test_l0_formula(self):
        r = 0.7
        expected = r * p_cens(HIT, HEADS.h_c, PARAMS) + (1 - r) * p_out(HIT, HEADS.h_o, PARAMS)
        self.assertAlmostEqual(likelihood_l0(HIT, r, HEADS), expected, places=15)

    def test_out_of_range_arguments_rejected(self):
        with self.assertRaises(ContractError):
            likelihood_l1(HIT, BRIDGE, 1.5, 0.8, HEADS)
        with self.assertRaises(ContractError):
            likelihood_l1(HIT, BRIDGE, 0.5, 0.01, HEADS)  # below r_min
        with self.assertRaises(ContractError):
            likelihood_l0(HIT, 1.2, HEADS)

    def test_likelihoods_strictly_positive(self):
        for report in (MISS, HIT):
            self.assertGreater(likelihood_l0(report, 0.5, HEADS), 0.0)
            self.assertGreater(likelihood_l1(report, BRIDGE, 0.9, 0.5, HEADS), 0.0)


class OracleParityTests(unittest.TestCase):
    """elgs/evidence.py vs the FROZEN fresh-context oracle
    (tests/ref_impls/likelihood_reference.py, authored from spec §2
    alone). Parity is asserted on the valid domain only; the one
    adjudicated divergence — off-domain reports, where the oracle
    returns density 0.0 (its A6 reading) while the pipeline raises
    fail-closed — is a pipeline contract, not a numeric disagreement.
    """

    @classmethod
    def setUpClass(cls):
        from tests.ref_impls import likelihood_reference as ref

        cls.ref = ref
        cls.ref_params = ref.LikelihoodParams(
            pi_m_v=PARAMS.pi_miss_vis,
            pi_m_c=PARAMS.pi_miss_cens,
            pi_m_o=PARAMS.pi_miss_out,
            x0=PARAMS.d_img_x0,
            x1=PARAMS.d_img_x1,
            y0=PARAMS.d_img_y0,
            y1=PARAMS.d_img_y1,
            g_pos_sigma=PARAMS.g_pos_sigma,
            v_min=PARAMS.v_min,
            v_max=PARAMS.v_max,
            r_min=PARAMS.r_min,
            h_floor=PARAMS.h_floor,
            pi_floor=PARAMS.pi_floor,
            g_cap=PARAMS.g_cap,
            pos_cap=PARAMS.pos_cap,
            g_v=HEADS.g_v,
            h_c=HEADS.h_c,
            h_o=HEADS.h_o,
        )

    def _pairs(self):
        reports = [
            (MISS, self.ref.Report(is_miss=True)),
            (HIT, self.ref.Report(is_miss=False, v=HIT.v, pos=(HIT.x, HIT.y))),
            (
                Report(is_miss=False, v=0.05, x=0.3, y=7.8),
                self.ref.Report(is_miss=False, v=0.05, pos=(0.3, 7.8)),
            ),
        ]
        centers = [BRIDGE, (0.5, 0.4), (9.8, 7.9)]
        return reports, centers

    def test_head_parity_on_valid_domain(self):
        reports, centers = self._pairs()
        for mine, theirs in reports:
            self.assertAlmostEqual(
                p_cens(mine, HEADS.h_c, PARAMS),
                self.ref.p_cens(theirs, self.ref_params),
                places=13,
            )
            self.assertAlmostEqual(
                p_out(mine, HEADS.h_o, PARAMS),
                self.ref.p_out(theirs, self.ref_params),
                places=13,
            )
            for center in centers:
                self.assertAlmostEqual(
                    p_vis(mine, center, HEADS.g_v, PARAMS),
                    self.ref.p_vis(theirs, center, self.ref_params),
                    places=13,
                )

    def test_likelihood_parity_grid(self):
        reports, centers = self._pairs()
        for mine, theirs in reports:
            for center in centers:
                for q in (0.0, 0.36, 0.9, 1.0):
                    for r in (PARAMS.r_min, 0.7, 1.0):
                        self.assertAlmostEqual(
                            likelihood_l1(mine, center, q, r, HEADS),
                            self.ref.L1(theirs, center, q, r, self.ref_params),
                            places=13,
                        )
                        self.assertAlmostEqual(
                            likelihood_l0(mine, r, HEADS),
                            self.ref.L0(theirs, r, self.ref_params),
                            places=13,
                        )

    def test_oracle_censoring_gap_is_exact_zero(self):
        reports, _ = self._pairs()
        for _, theirs in reports:
            gap = self.ref.censoring_equality_gap(theirs, BRIDGE, 0.7, self.ref_params)
            self.assertEqual(gap, 0.0)


class SegmentConstructionTests(unittest.TestCase):
    """§4 S(P_f, W): maximal edge-band-free runs, constant z; STRICT
    band boundaries are exercised through elgs.presence upstream."""

    def test_band_and_z_change_split_runs(self):
        from elgs.energy import build_segments

        frames = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        z = [True, True, False, False, True, True, True]
        band = [False, False, False, False, True, False, False]
        segments = build_segments(frames, z, band)
        self.assertEqual(
            [(s.frames, s.z) for s in segments],
            [((0.0, 1.0), True), ((2.0, 3.0), False), ((5.0, 6.0), True)],
        )

    def test_all_banded_yields_no_segments(self):
        from elgs.energy import build_segments

        self.assertEqual(build_segments([0.0, 1.0], [True, True], [True, True]), ())

    def test_length_mismatch_rejected(self):
        from elgs.energy import build_segments

        with self.assertRaises(ContractError):
            build_segments([0.0], [True, False], [False])


class CapOperatorTests(unittest.TestCase):
    def test_top_c_by_q_tilde(self):
        from elgs.energy import cap_select

        q = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.7}
        self.assertEqual(cap_select(q, 2), (1, 3))

    def test_ties_broken_by_ascending_camera_id(self):
        from elgs.energy import cap_select

        q = {4: 0.5, 1: 0.5, 3: 0.5}
        self.assertEqual(cap_select(q, 2), (1, 3))

    def test_all_zero_reduces_to_camera_id_order(self):
        """PROP 1 (i): with q_tilde all zero the selection is by the
        bridge-independent camera-ID tie-break alone."""
        from elgs.energy import cap_select

        q = {7: 0.0, 2: 0.0, 5: 0.0}
        self.assertEqual(cap_select(q, 2), (2, 5))

    def test_fewer_than_cap_takes_all(self):
        from elgs.energy import cap_select

        self.assertEqual(cap_select({3: 0.2}, 4), (3,))

    def test_invalid_cap_rejected(self):
        from elgs.energy import cap_select

        with self.assertRaises(ContractError):
            cap_select({0: 0.1}, 0)


class PhiAndProp1Tests(unittest.TestCase):
    def _batch_and_clusters(self, q_active=0.8):
        from elgs.clusters import SeedCluster
        from elgs.energy import EvidenceBatch

        # One track (j=0), two cameras, frames 0..3; frames 0-1 are the
        # censored segment (q=0 under EVERY bridge), frames 2-3 active.
        reports = {}
        q0, q1, centers0, centers1 = {}, {}, {}, {}
        for t in (0.0, 1.0, 2.0, 3.0):
            for c in (0, 1):
                key = (0, c, t)
                reports[key] = Report(is_miss=(c == 1), v=0.6, x=4.0 + c, y=3.0)
                active = t >= 2.0
                q0[key] = q_active if active else 0.0
                q1[key] = (q_active * 0.5) if active else 0.0
                centers0[key] = (4.2 + c, 3.1)
                centers1[key] = (4.4 + c, 3.3)
        batch = EvidenceBatch(
            reports=reports,
            q_tilde={0: q0, 1: q1},
            bridge_centers={0: centers0, 1: centers1},
        )
        cluster = SeedCluster(0, (0,), (0.0, 0.0, 0.0), 0.9, 1.0, 2, 0.5)
        return batch, (cluster,), {0: (0,)}

    def test_censored_segment_contributes_exactly_zero(self):
        """PROP 1: Phi is bit-identical with and without the censored
        segment, under any re-partitioning of it."""
        from elgs.energy import Segment, phi, stream_log_likelihoods

        batch, clusters, tracks = self._batch_and_clusters()
        active = Segment(frames=(2.0, 3.0), z=True)
        censored_whole = Segment(frames=(0.0, 1.0), z=True)
        censored_split = (
            Segment(frames=(0.0,), z=True),
            Segment(frames=(1.0,), z=False),  # re-partitioned, z flipped
        )
        phi_without = phi(
            stream_log_likelihoods(batch, clusters, tracks, (active,), HEADS, 2), 0.7
        )
        phi_with_whole = phi(
            stream_log_likelihoods(
                batch, clusters, tracks, (censored_whole, active), HEADS, 2
            ),
            0.7,
        )
        phi_with_split = phi(
            stream_log_likelihoods(
                batch, clusters, tracks, censored_split + (active,), HEADS, 2
            ),
            0.7,
        )
        self.assertEqual(phi_without, phi_with_whole)
        self.assertEqual(phi_without, phi_with_split)

    def test_all_censored_gives_phi_exactly_zero(self):
        from elgs.energy import Segment, phi, stream_log_likelihoods

        batch, clusters, tracks = self._batch_and_clusters()
        censored = Segment(frames=(0.0, 1.0), z=True)
        value = phi(
            stream_log_likelihoods(batch, clusters, tracks, (censored,), HEADS, 2), 0.7
        )
        self.assertEqual(value, 0.0)

    def test_hand_computed_single_report_stream(self):
        """l_b = alpha * log L on a minimal fixture, C_cap = 1."""
        from elgs.energy import Segment, stream_log_likelihoods

        batch, clusters, tracks = self._batch_and_clusters()
        segment = Segment(frames=(2.0,), z=True)
        out = stream_log_likelihoods(batch, clusters, tracks, (segment,), HEADS, 1)
        # C_cap=1 selects camera 0 (higher q on both bridges).
        key = (0, 0, 2.0)
        alpha = clusters[0].alpha_u
        for bridge in (0, 1):
            q = batch.q_tilde[bridge][key]
            center = batch.bridge_centers[bridge][key]
            expected_scored = alpha * math.log(
                likelihood_l1(batch.reports[key], center, q, 0.9, HEADS)
            )
            expected_cens = alpha * math.log(likelihood_l0(batch.reports[key], 0.9, HEADS))
            self.assertAlmostEqual(out.scored[bridge], expected_scored, places=12)
            self.assertAlmostEqual(out.censored[bridge], expected_cens, places=12)

    def test_z_zero_segment_cancels_in_phi(self):
        """L_{z=0} IS L0, so an uncensored z=0 segment adds equal mass
        to both streams and cancels in Phi (differences between
        programs arise from their z=1 spans)."""
        from elgs.energy import Segment, phi, stream_log_likelihoods

        batch, clusters, tracks = self._batch_and_clusters()
        active_z1 = Segment(frames=(2.0,), z=True)
        active_z0 = Segment(frames=(3.0,), z=False)
        phi_z1_only = phi(
            stream_log_likelihoods(batch, clusters, tracks, (active_z1,), HEADS, 2), 0.7
        )
        phi_both = phi(
            stream_log_likelihoods(
                batch, clusters, tracks, (active_z1, active_z0), HEADS, 2
            ),
            0.7,
        )
        self.assertAlmostEqual(phi_both, phi_z1_only, places=10)

    def test_constant_shift_cancels(self):
        """F(x + c*1) = F(x) - c => shifting BOTH streams cancels."""
        from elgs.energy import StreamLogLik, phi

        base = StreamLogLik(scored={0: -3.0, 1: -5.0}, censored={0: -4.0, 1: -4.5})
        shifted = StreamLogLik(
            scored={0: -3.0 + 2.5, 1: -5.0 + 2.5},
            censored={0: -4.0 + 2.5, 1: -4.5 + 2.5},
        )
        self.assertAlmostEqual(phi(base, 0.7), phi(shifted, 0.7), places=10)

    def test_phi_error_paths(self):
        from elgs.energy import StreamLogLik, phi

        with self.assertRaises(ContractError):
            phi(StreamLogLik(scored={}, censored={}), 0.7)
        with self.assertRaises(ContractError):
            phi(StreamLogLik(scored={0: 0.0}, censored={1: 0.0}), 0.7)
        with self.assertRaises(ContractError):
            phi(StreamLogLik(scored={0: 0.0}, censored={0: 0.0}), 0.0)


class EnergyAssemblyTests(unittest.TestCase):
    def test_prior_and_ledger_arithmetic(self):
        from elgs.energy import PriorTerms, TransactionCharges, total_energy

        priors = PriorTerms(
            kappa=2.0,
            psi_dur=lambda x: 0.1 * (x - 5.0) ** 2,
            psi_gap=lambda x: 0.2 * (x - 4.0) ** 2,
        )
        state = priors.state_prior(lens=[5.0, 6.0], gaps=[3.0])
        self.assertAlmostEqual(state, 2.0 * 2 + 0.1 * 1.0 + 0.2 * 1.0, places=12)
        charges = TransactionCharges(chi=1.5, mu=2.5)
        self.assertAlmostEqual(charges.ledger_charge(2, 1), 5.5, places=12)
        with self.assertRaises(ContractError):
            charges.ledger_charge(-1, 0)
        e = total_energy(10.0, 0.5, [1.0, -0.5], [state], 5.5)
        self.assertAlmostEqual(e, 10.0 + 0.25 + state + 5.5, places=12)
        with self.assertRaises(ContractError):
            total_energy(0.0, -0.1, [], [], 0.0)


if __name__ == "__main__":
    unittest.main()
