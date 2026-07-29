"""Unit tests for the Phase 0 primitive-centric evidence census (CSVL-VPL v2).

Pure-numpy fixtures only; no torch, no CUDA, no Gaussian model, no P01 reads.
Covers projection convention, consensus statistics, margin states, witness
logic, strict/relaxed run tracking, shuffle determinism, floors, cross-view
consistency, and a controlled two-layer hide/reveal sequence.
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


def identity_w2c():
    return np.eye(4, dtype=np.float64)


def run_sequence(tracker, sequence):
    """sequence: list of (state, oww) scalars for a single (primitive, camera)."""
    for frame, (state, oww) in enumerate(sequence):
        states = np.array([[state]], dtype=np.int8)
        oww_matrix = np.array([[oww]], dtype=bool)
        tracker.update(frame, states, oww_matrix)


class ConsensusDepthTests(unittest.TestCase):
    def test_median_and_mad(self):
        depth = np.stack([
            np.full((4, 4), 2.0), np.full((4, 4), 2.2), np.full((4, 4), 1.8),
        ])
        conf = np.ones_like(depth)
        d, sigma, valid, stats = census.consensus_depth(
            depth, conf, min_members=3, confidence_percentile=20.0
        )
        self.assertTrue(valid.all())
        self.assertTrue(np.allclose(d[valid], 2.0))
        self.assertTrue(np.allclose(sigma[valid], census.MAD_TO_SIGMA * 0.2, atol=1e-6))
        self.assertEqual(stats["members"], 3)

    def test_member_count_gate(self):
        depth = np.stack([
            np.full((2, 2), 2.0), np.full((2, 2), np.nan), np.full((2, 2), np.nan),
        ])
        conf = np.ones_like(depth)
        _, _, valid, _ = census.consensus_depth(
            depth, conf, min_members=3, confidence_percentile=20.0
        )
        self.assertFalse(valid.any())

    def test_nonpositive_depth_invalid(self):
        depth = np.stack([np.full((2, 2), -1.0)] * 3)
        conf = np.ones_like(depth)
        _, _, valid, _ = census.consensus_depth(
            depth, conf, min_members=3, confidence_percentile=20.0
        )
        self.assertFalse(valid.any())

    def test_confidence_percentile_gate(self):
        depth = np.stack([np.full((1, 10), 2.0)] * 3)
        conf = np.stack([np.linspace(0.0, 1.0, 10)[None, :]] * 3)
        _, _, valid, stats = census.consensus_depth(
            depth, conf, min_members=3, confidence_percentile=50.0
        )
        self.assertLess(int(valid.sum()), 10)
        self.assertTrue(np.isfinite(stats["confidence_floor"]))

    def test_shape_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            census.consensus_depth(
                np.zeros((3, 2, 2)), np.zeros((3, 2, 3)),
                min_members=3, confidence_percentile=20.0,
            )


class ProjectionTests(unittest.TestCase):
    def test_center_projection(self):
        xyz = np.array([[0.0, 0.0, 5.0]])
        pixels, z, in_view = census.project_points(
            xyz, identity_w2c(), simple_intrinsics(), 48, 64, near_clip=0.01
        )
        self.assertTrue(in_view[0])
        self.assertEqual(pixels[0].tolist(), [32, 24])
        self.assertAlmostEqual(float(z[0]), 5.0, places=9)

    def test_offset_projection_sign(self):
        xyz = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])
        pixels, _, in_view = census.project_points(
            xyz, identity_w2c(), simple_intrinsics(), 48, 64, near_clip=0.01
        )
        self.assertTrue(in_view.all())
        self.assertEqual(pixels[0].tolist(), [52, 24])
        self.assertEqual(pixels[1].tolist(), [32, 44])

    def test_behind_camera_excluded(self):
        xyz = np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 0.001]])
        _, _, in_view = census.project_points(
            xyz, identity_w2c(), simple_intrinsics(), 48, 64, near_clip=0.01
        )
        self.assertFalse(in_view.any())

    def test_w2c_translation(self):
        w2c = identity_w2c()
        w2c[2, 3] = 3.0
        xyz = np.array([[0.0, 0.0, 2.0]])
        pixels, z, in_view = census.project_points(
            xyz, w2c, simple_intrinsics(), 48, 64, near_clip=0.01
        )
        self.assertTrue(in_view[0])
        self.assertAlmostEqual(float(z[0]), 5.0, places=9)
        self.assertEqual(pixels[0].tolist(), [32, 24])

    def test_bad_shape_fails_closed(self):
        with self.assertRaises(ValueError):
            census.project_points(
                np.zeros((3,)), identity_w2c(), simple_intrinsics(), 48, 64, near_clip=0.01
            )


class StateClassificationTests(unittest.TestCase):
    def make_maps(self, depth_value=5.0, sigma_value=0.01, shape=(48, 64)):
        d = np.full(shape, depth_value, dtype=np.float32)
        s = np.full(shape, sigma_value, dtype=np.float32)
        v = np.ones(shape, dtype=bool)
        return d, s, v

    def classify(self, z_value, depth_value=5.0, sigma_value=0.01, present=True,
                 valid=True):
        d, s, v = self.make_maps(depth_value, sigma_value)
        if not valid:
            v[:] = False
        xyz = np.array([[0.0, 0.0, z_value]])
        pixels, z, in_view = census.project_points(
            xyz, identity_w2c(), simple_intrinsics(), 48, 64, near_clip=0.01
        )
        return census.classify_states(
            z, pixels, in_view, np.array([present]), d, s, v, tau_rel=0.03, kappa=2.5
        )[0]

    def test_near_surface(self):
        self.assertEqual(self.classify(5.05), census.STATE_NEAR_SURFACE)

    def test_behind(self):
        self.assertEqual(self.classify(5.5), census.STATE_BEHIND)

    def test_in_front(self):
        self.assertEqual(self.classify(4.5), census.STATE_IN_FRONT)

    def test_absent_not_evaluable(self):
        self.assertEqual(self.classify(5.0, present=False), census.STATE_NOT_EVALUABLE)

    def test_invalid_pixel_not_evaluable(self):
        self.assertEqual(self.classify(5.0, valid=False), census.STATE_NOT_EVALUABLE)

    def test_sigma_widens_margin(self):
        self.assertEqual(
            self.classify(6.0, sigma_value=1.0), census.STATE_NEAR_SURFACE
        )


class WitnessTests(unittest.TestCase):
    def test_witness_required(self):
        states = np.array([
            [census.STATE_BEHIND, census.STATE_NEAR_SURFACE],
            [census.STATE_BEHIND, census.STATE_NOT_EVALUABLE],
            [census.STATE_NEAR_SURFACE, census.STATE_NEAR_SURFACE],
        ], dtype=np.int8)
        oww = census.occluded_with_witness(states)
        self.assertTrue(oww[0, 0])
        self.assertFalse(oww[0, 1])
        self.assertFalse(oww[1].any())
        self.assertFalse(oww[2].any())

    def test_own_camera_not_a_witness(self):
        states = np.array([[census.STATE_BEHIND]], dtype=np.int8)
        self.assertFalse(census.occluded_with_witness(states).any())


class RunTrackerTests(unittest.TestCase):
    def test_completed_reveal(self):
        tracker = census.RunTracker(1, 1, min_run=3)
        seq = [(census.STATE_BEHIND, True)] * 3 + [(census.STATE_NEAR_SURFACE, False)]
        run_sequence(tracker, seq)
        self.assertEqual(tracker.completed_total, 1)
        self.assertTrue(tracker.completed_pairs[0, 0])
        self.assertEqual(tracker.summary(["cam"])["run_length_histogram"], {"3": 1})

    def test_short_run_not_completed(self):
        tracker = census.RunTracker(1, 1, min_run=3)
        seq = [(census.STATE_BEHIND, True)] * 2 + [(census.STATE_NEAR_SURFACE, False)]
        run_sequence(tracker, seq)
        self.assertEqual(tracker.completed_total, 0)

    def test_gap_resets_strict(self):
        tracker = census.RunTracker(1, 1, min_run=3)
        seq = [
            (census.STATE_BEHIND, True), (census.STATE_BEHIND, True),
            (census.STATE_NOT_EVALUABLE, False),
            (census.STATE_BEHIND, True), (census.STATE_NEAR_SURFACE, False),
        ]
        run_sequence(tracker, seq)
        self.assertEqual(tracker.completed_total, 0)

    def test_witnessless_behind_resets_strict(self):
        tracker = census.RunTracker(1, 1, min_run=3)
        seq = [
            (census.STATE_BEHIND, True), (census.STATE_BEHIND, True),
            (census.STATE_BEHIND, False),
            (census.STATE_NEAR_SURFACE, False),
        ]
        run_sequence(tracker, seq)
        self.assertEqual(tracker.completed_total, 0)

    def test_relaxed_survives_witnessless_behind(self):
        tracker = census.RunTracker(1, 1, min_run=3, relaxed=True)
        seq = [
            (census.STATE_BEHIND, True), (census.STATE_BEHIND, True),
            (census.STATE_BEHIND, False),
            (census.STATE_BEHIND, True), (census.STATE_NEAR_SURFACE, False),
        ]
        run_sequence(tracker, seq)
        self.assertEqual(tracker.completed_total, 1)

    def test_in_front_resets(self):
        tracker = census.RunTracker(1, 1, min_run=3)
        seq = [
            (census.STATE_BEHIND, True), (census.STATE_BEHIND, True),
            (census.STATE_BEHIND, True),
            (census.STATE_IN_FRONT, False), (census.STATE_NEAR_SURFACE, False),
        ]
        run_sequence(tracker, seq)
        self.assertEqual(tracker.completed_total, 0)

    def test_completions_by_frame_and_samples(self):
        tracker = census.RunTracker(1, 1, min_run=2, sample_cap=10)
        seq = [(census.STATE_BEHIND, True)] * 2 + [(census.STATE_NEAR_SURFACE, False)]
        run_sequence(tracker, seq)
        summary = tracker.summary(["cam"])
        self.assertEqual(summary["completions_by_frame"], {"2": 1})
        self.assertEqual(tracker.samples[0]["end_frame"], 2)
        self.assertEqual(tracker.samples[0]["run_length"], 2)


class TwoLayerHideRevealTests(unittest.TestCase):
    """Controlled fixture: a front plane covers a rear point in camera A for
    four frames while camera B always sees it; the census must count exactly
    one completed reveal for (point, camera A) and none for camera B."""

    def test_end_to_end(self):
        height, width = 48, 64
        intrinsics = simple_intrinsics()
        w2c_a = identity_w2c()
        w2c_b = identity_w2c()
        w2c_b[0, 3] = 0.5  # small lateral offset; rear point stays in view B
        rear_point = np.array([[0.0, 0.0, 5.0]])
        tracker = census.RunTracker(1, 2, min_run=3)

        occluded_frames = {2, 3, 4, 5}
        for frame in range(8):
            states = np.zeros((1, 2), dtype=np.int8)
            for cam_index, w2c in enumerate((w2c_a, w2c_b)):
                d = np.full((height, width), 5.0, dtype=np.float32)
                if cam_index == 0 and frame in occluded_frames:
                    d[:] = 2.0
                s = np.full((height, width), 0.01, dtype=np.float32)
                v = np.ones((height, width), dtype=bool)
                pixels, z, in_view = census.project_points(
                    rear_point, w2c, intrinsics, height, width, near_clip=0.01
                )
                states[:, cam_index] = census.classify_states(
                    z, pixels, in_view, np.array([True]), d, s, v,
                    tau_rel=0.03, kappa=2.5,
                )
            oww = census.occluded_with_witness(states)
            if frame in occluded_frames:
                self.assertEqual(states[0, 0], census.STATE_BEHIND)
                self.assertEqual(states[0, 1], census.STATE_NEAR_SURFACE)
                self.assertTrue(oww[0, 0])
            tracker.update(frame, states, oww)

        self.assertEqual(tracker.completed_total, 1)
        self.assertTrue(tracker.completed_pairs[0, 0])
        self.assertFalse(tracker.completed_pairs[0, 1])
        self.assertEqual(
            tracker.summary(["camA", "camB"])["run_length_histogram"], {"4": 1}
        )


class CrossViewConsistencyTests(unittest.TestCase):
    def test_consistent_plane(self):
        height, width = 48, 64
        d = np.full((height, width), 5.0, dtype=np.float32)
        s = np.full((height, width), 0.01, dtype=np.float32)
        v = np.ones((height, width), dtype=bool)
        outcome = census.cross_view_consistency(
            d, v, identity_w2c(), simple_intrinsics(), d, s, v,
            identity_w2c(), simple_intrinsics(),
            pixel_stride=8, tau_rel=0.03, kappa=2.5, near_clip=0.01,
        )
        self.assertGreater(outcome["evaluated"], 0)
        self.assertEqual(outcome["conflict"], 0)
        self.assertEqual(outcome["consistent"], outcome["evaluated"])

    def test_conflict_detected(self):
        height, width = 48, 64
        d_a = np.full((height, width), 5.0, dtype=np.float32)
        d_b = np.full((height, width), 8.0, dtype=np.float32)
        s = np.full((height, width), 0.01, dtype=np.float32)
        v = np.ones((height, width), dtype=bool)
        outcome = census.cross_view_consistency(
            d_a, v, identity_w2c(), simple_intrinsics(), d_b, s, v,
            identity_w2c(), simple_intrinsics(),
            pixel_stride=8, tau_rel=0.03, kappa=2.5, near_clip=0.01,
        )
        self.assertEqual(outcome["conflict"], outcome["evaluated"])


class ShuffleAndFloorTests(unittest.TestCase):
    def test_shuffle_deterministic_permutation(self):
        a = census.shuffled_frame_assignment(50, 3, seed=20260729)
        b = census.shuffled_frame_assignment(50, 3, seed=20260729)
        self.assertTrue(np.array_equal(a, b))
        for row in a:
            self.assertEqual(sorted(row.tolist()), list(range(50)))
        self.assertFalse(np.array_equal(a, np.tile(np.arange(50), (3, 1))))

    def make_summary(self, **overrides):
        summary = {
            "occluded_with_witness_fraction": 0.02,
            "strict": {
                "completed_reveal_pairs": 10000,
                "distinct_end_frames": 40,
                "distinct_end_cameras": 10,
            },
            "shuffle": {"completed_reveal_pairs": 100},
            "per_camera_occluded_fraction": {f"cam{i:02d}": 0.02 for i in range(19)},
            "consistency": {
                "evaluated": 10000, "consistent": 9000, "occluded": 800, "conflict": 200,
            },
            "consensus_maps": {"pass_fraction": 0.99},
        }
        summary.update(overrides)
        return summary

    def floors_config(self):
        return {
            "f1_min_occluded_fraction": 0.005,
            "f2_min_reveal_pairs": 5000,
            "f2_min_distinct_end_frames": 10,
            "f2_min_distinct_cameras": 5,
            "f3_median_band": [0.001, 0.40],
            "f3_max_any_camera": 0.60,
            "f4_min_valid_over_shuffle": 2.0,
            "f5_max_conflict_fraction": 0.15,
            "f5_min_map_pass_fraction": 0.90,
        }

    def test_all_pass(self):
        floors = census.evaluate_floors(self.make_summary(), self.floors_config())
        self.assertTrue(floors["phase0_go"])
        for key in ("f1_abundance", "f2_reveals", "f3_non_degeneracy",
                    "f4_control_separation", "f5_evidence_validity"):
            self.assertTrue(floors[key]["pass"], key)

    def test_f1_fail(self):
        floors = census.evaluate_floors(
            self.make_summary(occluded_with_witness_fraction=0.001),
            self.floors_config(),
        )
        self.assertFalse(floors["f1_abundance"]["pass"])
        self.assertFalse(floors["phase0_go"])

    def test_f2_fail(self):
        summary = self.make_summary()
        summary["strict"] = {**summary["strict"], "completed_reveal_pairs": 100}
        floors = census.evaluate_floors(summary, self.floors_config())
        self.assertFalse(floors["f2_reveals"]["pass"])
        self.assertFalse(floors["phase0_go"])

    def test_f4_fail_on_weak_separation(self):
        summary = self.make_summary()
        summary["shuffle"] = {"completed_reveal_pairs": 9000}
        floors = census.evaluate_floors(summary, self.floors_config())
        self.assertFalse(floors["f4_control_separation"]["pass"])
        self.assertFalse(floors["phase0_go"])

    def test_f5_fail_on_conflict(self):
        summary = self.make_summary()
        summary["consistency"] = {
            "evaluated": 10000, "consistent": 5000, "occluded": 2000, "conflict": 3000,
        }
        floors = census.evaluate_floors(summary, self.floors_config())
        self.assertFalse(floors["f5_evidence_validity"]["pass"])
        self.assertFalse(floors["phase0_go"])

    def test_degenerate_camera_fails_f3(self):
        fractions = {f"cam{i:02d}": 0.02 for i in range(18)}
        fractions["cam18"] = 0.7
        floors = census.evaluate_floors(
            self.make_summary(per_camera_occluded_fraction=fractions),
            self.floors_config(),
        )
        self.assertFalse(floors["f3_non_degeneracy"]["pass"])
        self.assertFalse(floors["phase0_go"])


class ConfigAndIndexTests(unittest.TestCase):
    def test_index_rejects_target_camera(self):
        manifest = {
            "target_camera": "cam00",
            "groups": [{
                "frame": 0,
                "member_camera_ids": ["cam00", "cam01"],
                "array_refs": {
                    "depth": {"path": "arrays/f0/g0/depth.npy"},
                    "confidence": {"path": "arrays/f0/g0/confidence.npy"},
                    "aligned_w2c": {"path": "arrays/f0/g0/aligned_w2c.npy"},
                    "processed_intrinsics": {"path": "arrays/f0/g0/processed_intrinsics.npy"},
                },
            }],
        }
        with self.assertRaises(ValueError):
            census.build_p01_index(manifest, "/tmp/none")

    def test_index_builds_camera_map(self):
        manifest = {
            "target_camera": "cam00",
            "groups": [{
                "frame": 7,
                "member_camera_ids": ["cam01", "cam02"],
                "array_refs": {
                    "depth": {"path": "arrays/f7/g0/depth.npy"},
                    "confidence": {"path": "arrays/f7/g0/confidence.npy"},
                    "aligned_w2c": {"path": "arrays/f7/g0/aligned_w2c.npy"},
                    "processed_intrinsics": {"path": "arrays/f7/g0/processed_intrinsics.npy"},
                },
            }],
        }
        index, cameras = census.build_p01_index(manifest, "/root")
        self.assertEqual(cameras, ["cam01", "cam02"])
        self.assertEqual(list(index.keys()), [7])
        self.assertEqual(index[7]["cam01"][0].member_index, 0)
        self.assertEqual(index[7]["cam02"][0].member_index, 1)
        self.assertTrue(index[7]["cam01"][0].depth_path.startswith("/root/arrays/f7"))


if __name__ == "__main__":
    unittest.main()
