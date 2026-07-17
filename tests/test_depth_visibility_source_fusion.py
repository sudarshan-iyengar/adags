import math
import unittest

import numpy as np

from depth_visibility.canonical import domain_id
from depth_visibility.fixtures import analytic_camera
from depth_visibility.fusion import (
    deduplicate_fused,
    enforce_source_exclusivity,
    patch_ncc,
    proposal_terms,
    reciprocal_candidates,
    robust_fuse,
)
from depth_visibility.sampling import (
    extract_patch,
    linear_gray,
    ordered_samples,
    regular_samples,
    sobel_magnitude,
    srgb_to_linear,
)
from depth_visibility.source_nodes import (
    aggregate_group_samples,
    build_source_node,
    duplicate_relative_mad,
    node_geometry,
    weighted_median_stable,
)


class SamplingAndSourceNodeTests(unittest.TestCase):
    def test_sampling_is_grid_first_stable_and_reflected(self):
        srgb = np.zeros((17, 17, 3), dtype=np.float64)
        srgb[4:9, 4:9] = 1.0
        linear = srgb_to_linear(srgb)
        self.assertEqual(linear_gray(linear).shape, (17, 17))
        magnitude = sobel_magnitude(linear_gray(linear))
        samples = ordered_samples(magnitude)
        grid = regular_samples(17, 17)
        self.assertEqual(samples[: len(grid)], grid)
        self.assertEqual(len(samples), len(grid) + math.floor(0.25 * len(grid)))
        self.assertEqual(len(samples), len(set(samples)))
        salient = samples[len(grid) :]
        for index, first in enumerate(salient):
            for second in salient[index + 1 :]:
                self.assertGreaterEqual(math.dist(first, second), 8.0)
        patch = extract_patch(np.arange(25).reshape(5, 5), 0, 0)
        self.assertEqual(patch.shape, (5, 5))
        self.assertEqual(patch[2, 2], 0)

    def test_weighted_median_and_zero_relative_mad_policy(self):
        self.assertEqual(weighted_median_stable([3, 1, 2], [0, 0, 0], ["c", "a", "b"]), 2)
        self.assertEqual(weighted_median_stable([1, 2, 3], [1, 10, 1], ["a", "b", "c"]), 2)
        self.assertEqual(duplicate_relative_mad([0.0, 0.0]), 0.0)
        self.assertTrue(math.isinf(duplicate_relative_mad([-1.0, 1.0], center=0.0)))

    @staticmethod
    def _sample(sample_id, depth, confidence=1.0):
        return {
            "sample_id": sample_id,
            "scene": "fixture",
            "frame": 0,
            "time": 0.0,
            "source_camera": "cam01",
            "scored_target": "cam00",
            "y": 2,
            "x": 3,
            "optical_z": depth,
            "confidence": confidence,
            "physical_ancestry": ["cam01", "cam02", "cam03"],
        }

    @staticmethod
    def _aggregate():
        return {
            "scene": "fixture",
            "frame": 0,
            "time": 0.0,
            "source_camera": "cam01",
            "scored_target": "cam00",
            "y": 2,
            "x": 3,
            "covariance": np.eye(3, dtype=np.float64) * 0.01,
        }

    def test_source_node_identity_excludes_record_state(self):
        samples = [self._sample("b", 2.0), self._sample("a", 2.02)]
        node = build_source_node(samples, self._aggregate())
        identity = {
            key: value
            for key, value in node.items()
            if key not in {"node_id", "retained"}
        }
        self.assertEqual(node["node_id"], domain_id("csvl-v1/source-node", identity))
        self.assertTrue(node["retained"])
        self.assertEqual(node["contributing_group_sample_ids"], ["a", "b"])

        accepted = aggregate_group_samples(samples, self._aggregate())
        self.assertTrue(accepted["retained"])
        rejected = aggregate_group_samples(
            [self._sample("a", 1.0), self._sample("b", 2.0), self._sample("c", 3.0)],
            self._aggregate(),
        )
        self.assertFalse(rejected["retained"])
        self.assertEqual(rejected["reason"], "duplicate_relative_mad")
        self.assertEqual(rejected["duplicate_risk"], 1.0)

        contaminated = [dict(samples[0], physical_ancestry=["cam00", "cam01"])]
        with self.assertRaises(ValueError):
            build_source_node(contaminated, self._aggregate())

    def test_optical_z_unprojection_and_covariance_transport(self):
        camera = analytic_camera("cam01")
        result = node_geometry(
            K=camera["K"],
            w2c=camera["w2c"],
            x=15.5,
            y=11.5,
            optical_z=2.0,
            duplicate_depths=[2.0, 2.0],
        )
        np.testing.assert_allclose(result["camera_point"], [0.0, 0.0, 2.0], atol=1e-12)
        np.testing.assert_allclose(result["world_point"], [0.0, 0.0, 2.0], atol=1e-12)
        self.assertEqual(result["sigma_z"], 0.02)
        self.assertTrue(np.all(np.linalg.eigvalsh(result["covariance"]) >= 0))


class FusionTests(unittest.TestCase):
    @staticmethod
    def _node(node_id, camera, xyz, color, confidence=1.0):
        return {
            "node_id": node_id,
            "source_camera": camera,
            "scene": "fixture",
            "frame": 0,
            "scored_target": "cam00",
            "world_point_array": np.array(xyz, dtype=np.float64),
            "covariance_array": np.eye(3, dtype=np.float64) * 0.01,
            "linear_rgb": np.array(color, dtype=np.float64),
            "confidence": confidence,
            "physical_ancestry": [camera],
            "duplicate_risk": 0.1,
            "pair_risk": 0.2,
            "pair_cost": 0.3,
        }

    def test_pair_terms_ncc_and_reciprocal_identity(self):
        self.assertEqual(patch_ncc(np.ones((2, 2)), np.ones((2, 2))), 1.0)
        self.assertEqual(patch_ncc(np.ones((2, 2)), np.full((2, 2), 2.0)), 0.0)
        terms = proposal_terms(1.0, 0.5, 0.2, 0.8)
        self.assertLessEqual(terms["cost"], 1.0)
        with self.assertRaises(ValueError):
            proposal_terms(-1.0, 0.5, 0.2, 0.8)
        forward = {
            "a": [{"cost": 0.2, "camera_id": "cam02", "y": 1, "x": 1, "node_id": "b"}]
        }
        reverse = {
            "b": [{"cost": 0.2, "camera_id": "cam01", "y": 1, "x": 1, "node_id": "a"}]
        }
        self.assertEqual(reciprocal_candidates(forward, reverse), [("a", "b", 0.2)])

    def test_robust_fusion_color_pruning_exclusivity_and_dedup(self):
        nodes = [
            self._node("a", "cam01", [0.0, 0.0, 2.0], [0.1, 0.1, 0.1], 1.0),
            self._node("b", "cam02", [0.001, 0.0, 2.0], [0.5, 0.5, 0.5], 10.0),
            self._node("c", "cam03", [0.0, 0.001, 2.0], [0.9, 0.9, 0.9], 1.0),
            self._node("z", "cam04", [100.0, 100.0, 100.0], [0.2, 0.2, 0.2], 1.0),
        ]
        fused = robust_fuse(nodes, r_scene=1.0, anchor_id="b")
        self.assertEqual(fused["camera_count"], 3)
        self.assertEqual(fused["anchor_id"], "b")
        self.assertIn("cam04", fused["physical_ancestry"])
        self.assertNotIn("z", fused["retained_source_node_ids"])
        np.testing.assert_allclose(fused["linear_rgb"], [0.5, 0.5, 0.5])
        self.assertEqual(fused["fused_id"], robust_fuse(reversed(nodes), r_scene=1.0, anchor_id="b")["fused_id"])

        better = dict(fused, fused_id="first", camera_count=4, median_pair_cost=0.1)
        worse = dict(fused, fused_id="second", camera_count=3, median_pair_cost=0.2)
        self.assertEqual(enforce_source_exclusivity([worse, better]), [better])
        boundary = dict(
            fused,
            fused_id="boundary",
            world_point_array=fused["world_point_array"] + np.array([0.002, 0.0, 0.0]),
        )
        self.assertEqual(len(deduplicate_fused([boundary, fused], r_scene=1.0)), 1)


if __name__ == "__main__":
    unittest.main()
