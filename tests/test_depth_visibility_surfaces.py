import unittest

import numpy as np

from depth_visibility.fixtures import planar_fused_points, two_plane_track_pixels
from depth_visibility.surfaces import (
    build_micro_surfaces,
    connected_regions,
    dense_depth_order,
    estimate_normal,
    orient_normal,
    rasterize_patch,
)


class NormalAndSurfaceTests(unittest.TestCase):
    def test_pca_normal_orientation_and_degeneracy(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [-0.005, -0.005, 0.0],
                [-0.005, 0.0, 0.0],
                [-0.005, 0.005, 0.0],
                [0.0, -0.005, 0.0],
                [0.0, 0.005, 0.0],
                [0.005, -0.005, 0.0],
                [0.005, 0.0, 0.0],
                [0.005, 0.005, 0.0],
            ],
            dtype=np.float64,
        )
        ids = [f"p{index}" for index in range(len(points))]
        centers = [np.array([0.0, 0.0, 1.0])] * len(points)
        normal = estimate_normal(0, points, ids, centers, r_scene=1.0)
        np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-12)

        line = np.column_stack(
            (np.arange(9, dtype=np.float64) * 0.001, np.zeros(9), np.zeros(9))
        )
        self.assertIsNone(estimate_normal(0, line, ids, centers, r_scene=1.0))
        np.testing.assert_array_equal(
            orient_normal(
                np.array([-1.0, 0.0, 0.0]),
                np.zeros(3),
                np.array([0.0, 0.0, 1.0]),
            ),
            [1.0, 0.0, 0.0],
        )

    def test_micro_surface_weighted_color_and_risk(self):
        patches = build_micro_surfaces(
            planar_fused_points(),
            scene_center=np.array([0.0, 0.0, 2.0]),
            r_scene=1.0,
        )
        self.assertEqual(len(patches), 1)
        patch = patches[0]
        np.testing.assert_allclose(patch["linear_rgb"], [0.15, 0.15, 0.15])
        self.assertAlmostEqual(patch["risk"], 0.3)
        self.assertEqual(len(patch["fused_hypothesis_ids"]), 3)
        self.assertFalse(patch["uncertain"])

    def test_rasterization_uses_covariance_and_patch_uncertainty(self):
        patch = {
            "patch_id": "patch",
            "track_id": "track",
            "risk": 0.2,
            "uncertain": False,
            "members": [
                {
                    "fused_id": "f0",
                    "uv": np.array([5.0, 5.0]),
                    "pixel_covariance": np.eye(2),
                    "z": 2.0,
                    "sigma_z": 0.02,
                    "risk": 0.1,
                    "physical_ancestry": ["cam01", "cam02", "cam03"],
                }
            ],
        }
        pixels = rasterize_patch(patch, 12, 12)
        self.assertIn((5, 5), pixels)
        self.assertEqual(pixels[(5, 5)]["risk"], 0.2)
        self.assertFalse(pixels[(5, 5)]["forced_uncertain"])
        patch["uncertain"] = True
        uncertain = rasterize_patch(patch, 12, 12)
        ordered = dense_depth_order({"track": uncertain}, {})
        self.assertTrue(all(layer[0]["state"] == "uncertain" for layer in ordered.values()))


class OrderingAndRegionTests(unittest.TestCase):
    def test_two_plane_order_and_reveal_geometry(self):
        tracks, witnesses = two_plane_track_pixels()
        ordered = dense_depth_order(tracks, witnesses)
        self.assertEqual(len(ordered), 16)
        for layers in ordered.values():
            self.assertEqual([layer["state"] for layer in layers], ["visible", "occluded"])
            self.assertLess(layers[1]["order_risk"], 1.0)
        regions = connected_regions(ordered)
        states = {(region["track_id"], region["state"], region["area"]) for region in regions}
        self.assertIn(("front", "visible", 16), states)
        self.assertIn(("rear", "occluded", 16), states)

        revealed_tracks, revealed_witnesses = two_plane_track_pixels(revealed=True)
        revealed = dense_depth_order(revealed_tracks, revealed_witnesses)
        self.assertEqual(
            {layer["state"] for layers in revealed.values() for layer in layers},
            {"visible"},
        )

    def test_ties_and_invalid_depth_fail_closed(self):
        tracks, witnesses = two_plane_track_pixels()
        for record in tracks["rear"].values():
            record["z"] = 2.01
        ordered = dense_depth_order(tracks, witnesses)
        self.assertTrue(
            all(
                [layer["state"] for layer in layers] == ["uncertain", "uncertain"]
                for layers in ordered.values()
            )
        )
        invalid, invalid_witnesses = two_plane_track_pixels(sign_error=True)
        with self.assertRaises(ValueError):
            dense_depth_order(invalid, invalid_witnesses)

    def test_small_components_become_uncertain(self):
        ordered = {
            (1, 1): [
                {
                    "track_id": "t",
                    "state": "visible",
                    "risk": 0.2,
                }
            ]
        }
        regions = connected_regions(ordered)
        self.assertEqual(regions[0]["state"], "uncertain")
        self.assertEqual(regions[0]["area"], 1)


if __name__ == "__main__":
    unittest.main()
