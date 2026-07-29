"""Unit tests for the census-v2 certified-reveal machinery (CSVL-VPL v2).

Pure numpy; no torch/model/P01. Covers gap-aware classification, the
anchored-hysteresis certification state machine (anchor, entry, grace,
pending-near, abort, smoothness, min-run, eligibility, re-certification),
quartile stratification, and the census-v2 floors.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from depth_visibility import primitive_census as census  # noqa: E402


def simple_intrinsics(f=100.0, cx=32.0, cy=24.0):
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def make_tracker(**overrides):
    defaults = dict(
        num_primitives=1, num_cameras=1,
        anchor_consec=2, entry_consec=2, reveal_consec=2,
        min_gap_occ_frames=3, grace_frames=1,
        smooth_rel=0.2, smooth_kappa=5.0, sample_cap=10,
    )
    defaults.update(overrides)
    return census.CertifiedRevealTracker(**defaults)


NEAR = "near"
GAP = "gap"
OTHER = "other"


def drive(tracker, sequence, occ_depths=None):
    """sequence: list of NEAR/GAP/OTHER tokens for one (primitive, camera).
    occ_depths optionally gives the evidence depth on GAP frames."""
    for frame, token in enumerate(sequence):
        near = np.array([[token == NEAR]])
        gap = np.array([[token == GAP]])
        depth_value = 2.0
        if occ_depths is not None and occ_depths[frame] is not None:
            depth_value = occ_depths[frame]
        occ_depth = np.array([[depth_value if token == GAP else np.nan]], dtype=np.float32)
        occ_sigma = np.array([[0.01]], dtype=np.float32)
        tracker.update(frame, near, gap, occ_depth, occ_sigma)


class ClassifyStatesV2Tests(unittest.TestCase):
    def run_case(self, z_value, depth_value=5.0, k_gap=3.0):
        d = np.full((48, 64), depth_value, dtype=np.float32)
        s = np.full((48, 64), 0.01, dtype=np.float32)
        v = np.ones((48, 64), dtype=bool)
        xyz = np.array([[0.0, 0.0, z_value]])
        pixels, z, in_view = census.project_points(
            xyz, np.eye(4), simple_intrinsics(), 48, 64, near_clip=0.01
        )
        return census.classify_states_v2(
            z, pixels, in_view, np.array([True]), d, s, v,
            tau_rel=0.03, kappa=2.5, k_gap=k_gap,
        )

    def test_gap_requires_large_margin_multiple(self):
        # margin = max(0.03*5, 2.5*0.01) = 0.15; behind at 0.2 but gap needs 0.45
        states, gap, occ_depth, occ_sigma = self.run_case(5.2)
        self.assertEqual(states[0], census.STATE_BEHIND)
        self.assertFalse(gap[0])
        # 0.5 behind exceeds 3x margin
        states, gap, occ_depth, _ = self.run_case(5.5)
        self.assertEqual(states[0], census.STATE_BEHIND)
        self.assertTrue(gap[0])
        self.assertAlmostEqual(float(occ_depth[0]), 5.0, places=5)

    def test_near_has_no_gap(self):
        states, gap, _, _ = self.run_case(5.05)
        self.assertEqual(states[0], census.STATE_NEAR_SURFACE)
        self.assertFalse(gap[0])

    def test_not_evaluable_depth_nan(self):
        d = np.full((48, 64), 5.0, dtype=np.float32)
        s = np.full((48, 64), 0.01, dtype=np.float32)
        v = np.zeros((48, 64), dtype=bool)
        xyz = np.array([[0.0, 0.0, 5.0]])
        pixels, z, in_view = census.project_points(
            xyz, np.eye(4), simple_intrinsics(), 48, 64, near_clip=0.01
        )
        states, gap, occ_depth, _ = census.classify_states_v2(
            z, pixels, in_view, np.array([True]), d, s, v,
            tau_rel=0.03, kappa=2.5, k_gap=3.0,
        )
        self.assertEqual(states[0], census.STATE_NOT_EVALUABLE)
        self.assertTrue(np.isnan(occ_depth[0]))


class CertifiedRevealTrackerTests(unittest.TestCase):
    def test_clean_certification(self):
        tracker = make_tracker()
        drive(tracker, [NEAR, NEAR, GAP, GAP, GAP, GAP, NEAR, NEAR])
        self.assertEqual(tracker.certified_total, 1)
        self.assertTrue(tracker.certified_pairs[0, 0])
        self.assertEqual(tracker.summary(["cam"])["run_length_histogram"], {"4": 1})

    def test_no_anchor_no_certification(self):
        tracker = make_tracker()
        drive(tracker, [GAP, GAP, GAP, GAP, NEAR, NEAR])
        self.assertEqual(tracker.certified_total, 0)

    def test_min_run_not_met(self):
        tracker = make_tracker()
        drive(tracker, [NEAR, NEAR, GAP, GAP, NEAR, NEAR])
        self.assertEqual(tracker.certified_total, 0)

    def test_flicker_never_enters(self):
        tracker = make_tracker()
        drive(tracker, [NEAR, NEAR] + [GAP, NEAR] * 6)
        self.assertEqual(tracker.certified_total, 0)

    def test_post_entry_flicker_aborts(self):
        tracker = make_tracker()
        drive(tracker, [NEAR, NEAR, GAP, GAP, NEAR, GAP, NEAR, GAP, NEAR, NEAR])
        self.assertEqual(tracker.certified_total, 0)

    def test_grace_other_frame_tolerated(self):
        tracker = make_tracker()
        drive(tracker, [NEAR, NEAR, GAP, GAP, OTHER, GAP, GAP, NEAR, NEAR])
        self.assertEqual(tracker.certified_total, 1)
        self.assertEqual(tracker.summary(["cam"])["run_length_histogram"], {"4": 1})

    def test_pending_near_blip_consumes_grace_then_certifies(self):
        tracker = make_tracker()
        drive(tracker, [NEAR, NEAR, GAP, GAP, NEAR, GAP, GAP, NEAR, NEAR])
        self.assertEqual(tracker.certified_total, 1)
        self.assertEqual(tracker.summary(["cam"])["run_length_histogram"], {"4": 1})

    def test_two_interruptions_abort(self):
        tracker = make_tracker()
        drive(tracker, [NEAR, NEAR, GAP, GAP, OTHER, GAP, OTHER, GAP, NEAR, NEAR])
        self.assertEqual(tracker.certified_total, 0)

    def test_near_blip_then_other_aborts(self):
        # blip charge (1) + hard interruption (1) exceeds grace 1
        tracker = make_tracker()
        drive(tracker, [NEAR, NEAR, GAP, GAP, NEAR, OTHER, GAP, GAP, NEAR, NEAR])
        self.assertEqual(tracker.certified_total, 0)

    def test_occluder_depth_jump_breaks_run(self):
        tracker = make_tracker()
        seq = [NEAR, NEAR, GAP, GAP, GAP, GAP, NEAR, NEAR]
        depths = [None, None, 2.0, 2.0, 0.5, 0.5, None, None]
        drive(tracker, seq, occ_depths=depths)
        # frames 4 and 5 fail smoothness vs last accepted depth 2.0:
        # two hard interruptions exceed grace 1 -> abort, no certification.
        self.assertEqual(tracker.certified_total, 0)

    def test_smooth_occluder_drift_allowed(self):
        tracker = make_tracker()
        seq = [NEAR, NEAR, GAP, GAP, GAP, GAP, NEAR, NEAR]
        depths = [None, None, 2.0, 2.1, 2.2, 2.3, None, None]
        drive(tracker, seq, occ_depths=depths)
        self.assertEqual(tracker.certified_total, 1)

    def test_ineligible_pair_never_certifies(self):
        tracker = make_tracker(eligible=np.zeros((1, 1), dtype=bool))
        drive(tracker, [NEAR, NEAR, GAP, GAP, GAP, GAP, NEAR, NEAR])
        self.assertEqual(tracker.certified_total, 0)

    def test_recertification_after_reanchor(self):
        tracker = make_tracker()
        cycle = [NEAR, NEAR, GAP, GAP, GAP, NEAR, NEAR]
        drive(tracker, cycle + cycle)
        self.assertEqual(tracker.certified_total, 2)
        self.assertEqual(int(tracker.certified_pairs.sum()), 1)

    def test_reveal_streak_reanchors(self):
        tracker = make_tracker()
        drive(tracker, [NEAR, NEAR, GAP, GAP, GAP, NEAR, NEAR, GAP, GAP, GAP, NEAR, NEAR])
        self.assertEqual(tracker.certified_total, 2)

    def test_sample_records_run_length(self):
        tracker = make_tracker()
        drive(tracker, [NEAR, NEAR, GAP, GAP, GAP, NEAR, NEAR])
        self.assertEqual(len(tracker.samples), 1)
        self.assertEqual(tracker.samples[0]["gap_occ_frames"], 3)
        self.assertEqual(tracker.samples[0]["end_frame"], 6)


class QuartileAndFloorTests(unittest.TestCase):
    def test_quartile_labels(self):
        values = np.arange(100, dtype=np.float64)
        labels, edges = census.quartile_labels(values)
        self.assertEqual(labels.min(), 0)
        self.assertEqual(labels.max(), 3)
        self.assertEqual(int((labels == 0).sum()), 25)
        self.assertAlmostEqual(edges[1], 49.5)

    def make_summary(self, **overrides):
        summary = {
            "valid": {
                "certified_pairs": 8000,
                "distinct_end_frames": 60,
                "distinct_end_cameras": 12,
                "per_camera_certified": {f"cam{i:02d}": 470 for i in range(17)},
            },
            "shuffle": {"certified_pairs": 500},
            "consistency": {"evaluated": 10000, "consistent": 9500,
                            "occluded": 400, "conflict": 100},
            "consensus_maps": {"pass_fraction": 1.0},
        }
        summary.update(overrides)
        return summary

    def floors(self):
        return {
            "g1_min_certified_pairs": 2000,
            "g1_min_distinct_end_frames": 10,
            "g1_min_distinct_cameras": 8,
            "g2_min_valid_over_shuffle": 3.0,
            "g3_max_conflict_fraction": 0.15,
            "g3_min_map_pass_fraction": 0.90,
            "g4_max_camera_share": 0.60,
        }

    def test_all_pass(self):
        floors = census.evaluate_census2_floors(self.make_summary(), self.floors())
        self.assertTrue(floors["census2_go"])

    def test_g2_fail(self):
        summary = self.make_summary(shuffle={"certified_pairs": 6000})
        floors = census.evaluate_census2_floors(summary, self.floors())
        self.assertFalse(floors["g2_control_separation"]["pass"])
        self.assertFalse(floors["census2_go"])

    def test_g2_zero_shuffle_passes_with_abundance(self):
        summary = self.make_summary(shuffle={"certified_pairs": 0})
        floors = census.evaluate_census2_floors(summary, self.floors())
        self.assertTrue(floors["g2_control_separation"]["pass"])
        self.assertEqual(floors["g2_control_separation"]["rho"], "inf")

    def test_g1_fail(self):
        summary = self.make_summary()
        summary["valid"] = {**summary["valid"], "certified_pairs": 100}
        floors = census.evaluate_census2_floors(summary, self.floors())
        self.assertFalse(floors["g1_certified_abundance"]["pass"])
        self.assertFalse(floors["census2_go"])

    def test_g4_fail_on_camera_concentration(self):
        per_cam = {f"cam{i:02d}": 10 for i in range(17)}
        per_cam["cam00"] = 5000
        summary = self.make_summary()
        summary["valid"] = {**summary["valid"], "per_camera_certified": per_cam}
        floors = census.evaluate_census2_floors(summary, self.floors())
        self.assertFalse(floors["g4_non_degeneracy"]["pass"])
        self.assertFalse(floors["census2_go"])


if __name__ == "__main__":
    unittest.main()
