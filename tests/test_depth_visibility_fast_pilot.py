import unittest

import numpy as np

from depth_visibility.fast_pilot import (
    anchor_duplicate_diagnostic,
    evaluate_frame_geometry,
    temporal_bin_transitions,
)


class FastPilotTests(unittest.TestCase):
    def test_anchor_duplicate_abstains_only_on_inconsistent_pixels(self):
        first = np.ones((16, 16), dtype=np.float64)
        second = np.ones_like(first)
        second[0, 0] = 2.0
        confidence = np.ones_like(first)
        report, aggregate, retained = anchor_duplicate_diagnostic(
            first,
            second,
            confidence,
            confidence,
            relative_limit=0.05,
            calibration_stride=8,
        )
        self.assertEqual(report["rejected_pixel_count"], 1)
        self.assertAlmostEqual(report["retained_fraction"], 255 / 256)
        self.assertFalse(retained[0, 0])
        self.assertTrue(retained[1, 1])
        self.assertEqual(aggregate[0, 0], 1.0)
        self.assertFalse(
            report["heldout_scale_diagnostic"]["applied_to_geometry"]
        )

    def test_consistent_three_camera_plane_has_supported_geometry(self):
        K = np.array([[40.0, 0.0, 15.5], [0.0, 40.0, 11.5], [0.0, 0.0, 1.0]])
        depth = np.full((3, 24, 32), 2.0, dtype=np.float64)
        prediction = {
            "depth": depth,
            "intrinsics": np.repeat(K[None, ...], 3, axis=0),
            "extrinsics": np.repeat(np.eye(4)[None, ...], 3, axis=0),
        }
        report, bins = evaluate_frame_geometry(
            [prediction],
            [("cam01", "cam02", "cam03")],
            target_K=K,
            target_w2c=np.eye(4),
            target_width=32,
            target_height=24,
            stride=8,
            minimum_cameras=3,
            maximum_depth_sigma=2.5,
            target_bin_pixels=8,
        )
        self.assertEqual(report["physical_camera_count"], 3)
        self.assertGreater(report["supported_source_point_count"], 0)
        self.assertGreater(report["target_supported_bin_count"], 0)
        self.assertAlmostEqual(report["cross_view_agreement_fraction_of_valid"], 1.0)
        self.assertEqual(report["target_ordered_multilayer_bin_count"], 0)
        self.assertEqual(len(bins), report["target_supported_bin_count"])

    def test_anchor_mask_applies_on_source_and_target_support(self):
        K = np.array([[40.0, 0.0, 15.5], [0.0, 40.0, 11.5], [0.0, 0.0, 1.0]])
        depth = np.full((3, 24, 32), 2.0, dtype=np.float64)
        prediction = {
            "depth": depth,
            "intrinsics": np.repeat(K[None, ...], 3, axis=0),
            "extrinsics": np.repeat(np.eye(4)[None, ...], 3, axis=0),
        }
        report, bins = evaluate_frame_geometry(
            [prediction],
            [("cam01", "cam02", "cam03")],
            target_K=K,
            target_w2c=np.eye(4),
            target_width=32,
            target_height=24,
            stride=8,
            minimum_cameras=3,
            camera_valid_masks={"cam01": np.zeros((24, 32), dtype=bool)},
        )
        self.assertEqual(report["supported_source_point_count"], 0)
        self.assertEqual(bins, set())

    def test_temporal_bins_report_reveal_and_hide_as_proxy_only(self):
        transitions = temporal_bin_transitions(
            [(125, {(0, 0), (1, 0)}), (126, {(1, 0), (2, 0)})]
        )
        self.assertEqual(transitions[0]["retained_bin_count"], 1)
        self.assertEqual(transitions[0]["newly_supported_bin_count"], 1)
        self.assertEqual(transitions[0]["newly_hidden_bin_count"], 1)
        self.assertIn("not surface-track", transitions[0]["interpretation"])


if __name__ == "__main__":
    unittest.main()
