"""Unit tests for elgs/bridges.py and elgs/observability.py (spec §3).

CPU only, unittest. The q oracle is a closed-form 2-3-Gaussian
front-set compositing fixture computed by hand (plan §3 register:
the chosen probe method must match this to float precision).
"""

import math
import unittest

from depth_visibility.errors import ContractError
from elgs.bridges import (
    AnchorInterval,
    coverage_entry,
    find_anchor_intervals,
    validate_bridge_ids,
    windows_between_anchors,
)
from elgs.observability import (
    QSnapshot,
    SigmaPoint,
    compute_q,
    q_tilde,
    validate_sigma_points,
)
from elgs.probe import AnalyticProbe, ProbeGaussian


class AnchorAndWindowTests(unittest.TestCase):
    def test_anchor_detection_with_strict_floor(self):
        frames = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        plateau = [True, True, False, True, True, True, False]
        counts = [3, 2, 0, 1, 1, 1, 0]
        # Run 1 total 5, run 2 total 3; floor 3 is STRICT: run 2 fails.
        anchors = find_anchor_intervals(frames, plateau, counts, report_floor=3)
        self.assertEqual(len(anchors), 1)
        self.assertEqual((anchors[0].start_frame, anchors[0].end_frame), (0.0, 1.0))
        anchors2 = find_anchor_intervals(frames, plateau, counts, report_floor=2)
        self.assertEqual(len(anchors2), 2)
        self.assertEqual((anchors2[1].start_frame, anchors2[1].end_frame), (3.0, 5.0))

    def test_trailing_run_is_closed(self):
        anchors = find_anchor_intervals(
            [0.0, 1.0], [True, True], [2, 2], report_floor=1
        )
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0].end_frame, 1.0)

    def test_windows_between_consecutive_anchors(self):
        anchors = (
            AnchorInterval(0.0, 2.0, 5),
            AnchorInterval(6.0, 8.0, 4),
            AnchorInterval(12.0, 14.0, 6),
        )
        windows = windows_between_anchors(anchors)
        self.assertEqual(
            [(w.start_frame, w.end_frame) for w in windows],
            [(2.0, 6.0), (8.0, 12.0)],
        )

    def test_fewer_than_two_anchors_means_no_windows(self):
        self.assertEqual(windows_between_anchors(()), ())
        self.assertEqual(windows_between_anchors((AnchorInterval(0.0, 2.0, 5),)), ())
        entry = coverage_entry(3, (AnchorInterval(0.0, 2.0, 5),))
        self.assertTrue(entry.photometric_only)
        self.assertEqual(entry.n_windows, 0)

    def test_overlapping_anchors_rejected(self):
        with self.assertRaises(ContractError):
            windows_between_anchors(
                (AnchorInterval(0.0, 5.0, 5), AnchorInterval(4.0, 8.0, 4))
            )

    def test_bridge_id_validation(self):
        self.assertEqual(validate_bridge_ids([3, 1, 2]), (1, 2, 3))
        with self.assertRaises(ContractError):
            validate_bridge_ids([])
        with self.assertRaises(ContractError):
            validate_bridge_ids([1, 1, 2])


def _probe():
    # Closed-form fixture: two occluders in front of depth 10, one behind.
    # At pixel (5,5): G1 factor = 1 - 0.5*1 = 0.5;
    # G2 footprint = exp(-0.5 * (2^2)/(2^2)) = exp(-0.5),
    #   factor = 1 - 0.4*exp(-0.5); G3 at depth 12 never enters.
    splats = {
        (0, 1.0): [
            ProbeGaussian(depth=3.0, alpha=0.5, sigma_px=2.0, center_px=(5.0, 5.0),
                          family_id=1, source_track=7),
            ProbeGaussian(depth=6.0, alpha=0.4, sigma_px=2.0, center_px=(7.0, 5.0),
                          family_id=2),
            ProbeGaussian(depth=12.0, alpha=0.9, sigma_px=2.0, center_px=(5.0, 5.0),
                          family_id=3),
        ]
    }
    return AnalyticProbe(splats)


T_FULL = 0.5 * (1.0 - 0.4 * math.exp(-0.5))   # both occluders
T_NEAR = 0.5                                   # only G1 in front of depth 5
T_EXCL = 1.0 - 0.4 * math.exp(-0.5)            # G1's track excluded


class TransmittanceFixtureTests(unittest.TestCase):
    def test_closed_form_front_set_compositing(self):
        probe = _probe()
        t = probe.transmittance(0, 1.0, (5.0, 5.0), 10.0,
                                exclude_track=None, present_family=None)
        self.assertAlmostEqual(t, T_FULL, places=15)

    def test_strictly_in_front_only(self):
        probe = _probe()
        t = probe.transmittance(0, 1.0, (5.0, 5.0), 5.0,
                                exclude_track=None, present_family=None)
        self.assertAlmostEqual(t, T_NEAR, places=15)
        # A splat exactly AT the query depth does not occlude it.
        t_at = probe.transmittance(0, 1.0, (5.0, 5.0), 6.0,
                                   exclude_track=None, present_family=None)
        self.assertAlmostEqual(t_at, T_NEAR, places=15)

    def test_query_source_exclusion_changes_t(self):
        probe = _probe()
        t = probe.transmittance(0, 1.0, (5.0, 5.0), 10.0,
                                exclude_track=7, present_family=None)
        self.assertAlmostEqual(t, T_EXCL, places=15)
        self.assertNotAlmostEqual(t, T_FULL, places=3)

    def test_unknown_camera_frame_rejected(self):
        with self.assertRaises(ContractError):
            _probe().transmittance(9, 1.0, (0.0, 0.0), 1.0,
                                   exclude_track=None, present_family=None)


class ComputeQTests(unittest.TestCase):
    def test_hand_computed_q(self):
        probe = _probe()
        points = [
            SigmaPoint(weight=0.6, point=(0.0, 0.0, 10.0), pixel=(5.0, 5.0), depth=10.0),
            SigmaPoint(weight=0.4, point=(0.0, 0.0, 5.0), pixel=(5.0, 5.0), depth=5.0),
        ]
        q = compute_q(points, probe, 0, 1.0, exclude_track=None,
                      present_family=None, kappa_res=0.9)
        self.assertAlmostEqual(q, 0.9 * (0.6 * T_FULL + 0.4 * T_NEAR), places=15)

    def test_out_of_frustum_contributes_zero(self):
        splats = _probe()._splats  # same splat set, restrictive frustum
        probe = AnalyticProbe(splats, frustum={(0, 1.0): lambda p: p[2] < 8.0})
        points = [
            SigmaPoint(weight=0.6, point=(0.0, 0.0, 10.0), pixel=(5.0, 5.0), depth=10.0),
            SigmaPoint(weight=0.4, point=(0.0, 0.0, 5.0), pixel=(5.0, 5.0), depth=5.0),
        ]
        q = compute_q(points, probe, 0, 1.0, exclude_track=None,
                      present_family=None, kappa_res=1.0)
        self.assertAlmostEqual(q, 0.4 * T_NEAR, places=15)

    def test_q_clipped_and_validated(self):
        probe = _probe()
        good = [SigmaPoint(1.0, (0.0, 0.0, 1.0), (5.0, 5.0), 1.0)]
        self.assertEqual(
            compute_q(good, probe, 0, 1.0, exclude_track=None,
                      present_family=None, kappa_res=1.0),
            1.0,  # nothing in front of depth 1 => T = 1, kappa = 1
        )
        with self.assertRaises(ContractError):
            compute_q(good, probe, 0, 1.0, exclude_track=None,
                      present_family=None, kappa_res=1.5)
        bad_weights = [
            SigmaPoint(0.6, (0.0, 0.0, 1.0), (5.0, 5.0), 1.0),
            SigmaPoint(0.6, (0.0, 0.0, 1.0), (5.0, 5.0), 1.0),
        ]
        with self.assertRaises(ContractError):
            compute_q(bad_weights, probe, 0, 1.0, exclude_track=None,
                      present_family=None, kappa_res=1.0)
        with self.assertRaises(ContractError):
            validate_sigma_points([])
        with self.assertRaises(ContractError):
            SigmaPoint(-0.1, (0.0, 0.0, 1.0), (0.0, 0.0), 1.0)

    def test_q_tilde_range(self):
        self.assertAlmostEqual(q_tilde(0.5, 0.4), 0.2, places=15)
        with self.assertRaises(ContractError):
            q_tilde(1.2, 0.5)
        with self.assertRaises(ContractError):
            q_tilde(0.5, -0.1)


class QSnapshotTests(unittest.TestCase):
    def test_snapshot_freeze_semantics(self):
        snap = QSnapshot(round_index=1)
        snap.put(0, 7, 0, 2.0, 0.36)
        with self.assertRaises(ContractError):
            snap.put(0, 7, 0, 2.0, 0.5)  # double write
        snap.freeze()
        with self.assertRaises(ContractError):
            snap.put(0, 8, 0, 2.0, 0.1)  # write after freeze
        with self.assertRaises(ContractError):
            snap.freeze()
        self.assertEqual(snap.get(0, 7, 0, 2.0), 0.36)
        with self.assertRaises(ContractError):
            snap.get(1, 7, 0, 2.0)  # missing key fails closed

    def test_snapshot_determinism_and_bridge_maps(self):
        def build():
            snap = QSnapshot(round_index=2)
            snap.put(0, 7, 0, 2.0, 0.36)
            snap.put(1, 7, 0, 2.0, 0.18)
            snap.put(0, 7, 1, 2.0, 0.0)
            snap.freeze()
            return snap.as_bridge_maps()

        first, second = build(), build()
        self.assertEqual(first, second)
        self.assertEqual(first[0][(7, 0, 2.0)], 0.36)
        self.assertEqual(first[1][(7, 0, 2.0)], 0.18)

    def test_out_of_range_value_rejected(self):
        snap = QSnapshot(round_index=0)
        with self.assertRaises(ContractError):
            snap.put(0, 1, 0, 0.0, 1.5)


if __name__ == "__main__":
    unittest.main()
