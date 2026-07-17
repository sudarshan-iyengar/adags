import math
import unittest

import numpy as np

from depth_visibility.errors import ProvenanceError
from depth_visibility.groups import (
    assert_transitive_target_exclusion,
    enumerate_anchor_groups,
    filter_groups_for_target,
    physical_ancestry,
    select_anchor_group,
    validate_group,
)


def camera(camera_id, center, rotation=None):
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
    w2c = np.eye(4)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = -(rotation @ np.asarray(center, dtype=np.float64))
    return {"camera_id": camera_id, "w2c": w2c}


def diverse_cameras(count=10):
    result = {}
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        center = [2.0 * math.cos(angle), 2.0 * math.sin(angle), 0.5 * math.sin(2.0 * angle)]
        camera_id = f"cam{index:02d}"
        result[camera_id] = camera(camera_id, center)
    return result


class GroupTests(unittest.TestCase):
    def test_farthest_point_tie_uses_lower_camera_id(self):
        cameras = {
            "cam00": camera("cam00", [0.0, 0.0, 0.0]),
            "cam01": camera("cam01", [1.0, 0.0, 0.0]),
            "cam02": camera("cam02", [-1.0, 0.0, 0.0]),
        }
        group = select_anchor_group(
            "cam00", cameras, 1.0,
            maximum_cameras=2,
            minimum_second_singular_value_rscene=0.0,
        )
        self.assertEqual(group, ("cam00", "cam01"))

    def test_angle_and_center_eligibility(self):
        yaw_90 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        cameras = {
            "cam00": camera("cam00", [0.0, 0.0, 0.0]),
            "cam01": camera("cam01", [0.001, 0.0, 0.0]),
            "cam02": camera("cam02", [1.0, 0.0, 0.0], yaw_90),
            "cam03": camera("cam03", [1.0, 1.0, 0.0]),
        }
        with self.assertRaises(ProvenanceError):
            select_anchor_group(
                "cam00", cameras, 1.0,
                maximum_cameras=4,
                minimum_second_singular_value_rscene=0.0,
            )

    def test_complete_six_and_diversity_failures(self):
        cameras = diverse_cameras(6)
        valid = validate_group(tuple(cameras), cameras, 1.0)
        self.assertEqual(len(valid), 6)
        with self.assertRaises(ProvenanceError):
            validate_group(tuple(cameras)[:5], cameras, 1.0)
        collinear = {
            f"cam{index:02d}": camera(f"cam{index:02d}", [float(index), 0.0, 0.0])
            for index in range(6)
        }
        with self.assertRaises(ProvenanceError):
            validate_group(tuple(collinear), collinear, 1.0)

    def test_target_filter_recomputes_two_group_support(self):
        cameras = diverse_cameras(10)
        generated = enumerate_anchor_groups(cameras, 1.0)
        self.assertEqual(len(generated), 10)
        for group in generated:
            validate_group(group, cameras, 1.0)
        groups = (
            ("cam00", "cam01", "cam02", "cam03", "cam04", "cam05"),
            ("cam01", "cam02", "cam03", "cam04", "cam05", "cam06"),
            ("cam02", "cam03", "cam04", "cam05", "cam06", "cam07"),
        )
        filtered = filter_groups_for_target(groups, "cam00", cameras, 1.0)
        self.assertTrue(filtered)
        for source, source_groups in filtered.items():
            self.assertNotEqual(source, "cam00")
            self.assertGreaterEqual(len(source_groups), 2)
            for group in source_groups:
                self.assertNotIn("cam00", group)
                validate_group(group, cameras, 1.0)

    def test_transitive_ancestry_and_target_image_exclusion(self):
        clean = {
            "physical_camera_ancestry": ["cam01", "cam02"],
            "parents": [
                {"physical_camera_ancestry": ["cam01"], "image_sha256": "a" * 64},
                {"physical_camera_ancestry": ["cam02"], "image_sha256": "b" * 64},
            ],
            "scored_target_camera": "cam00",
        }
        self.assertEqual(physical_ancestry(clean), frozenset({"cam01", "cam02"}))
        assert_transitive_target_exclusion(clean, "cam00", target_image_sha256="c" * 64)
        contaminated = dict(clean)
        contaminated["physical_camera_ancestry"] = ["cam00", "cam01", "cam02"]
        with self.assertRaises(ProvenanceError):
            assert_transitive_target_exclusion(contaminated, "cam00")
        target_image = dict(clean)
        target_image["parents"] = clean["parents"] + [
            {"physical_camera_ancestry": ["cam03"], "input_image_sha256": "c" * 64}
        ]
        target_image["physical_camera_ancestry"] = ["cam01", "cam02", "cam03"]
        with self.assertRaises(ProvenanceError):
            assert_transitive_target_exclusion(
                target_image, "cam00", target_image_sha256="c" * 64
            )
        with self.assertRaises(ProvenanceError):
            assert_transitive_target_exclusion({"payload": "no ancestry"}, "cam00")

    def test_declared_ancestry_cannot_omit_nested_camera(self):
        malformed = {
            "physical_camera_ancestry": ["cam01"],
            "parents": [{"physical_camera_ancestry": ["cam02"]}],
        }
        with self.assertRaises(ProvenanceError):
            physical_ancestry(malformed)


if __name__ == "__main__":
    unittest.main()
