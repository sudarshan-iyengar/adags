import math
import unittest

import numpy as np

from depth_visibility.camera import (
    camera_center,
    intrinsics_matrix,
    opengl_c2w_to_opencv_w2c,
    processed_image_transform,
    project_world,
    transport_pixel_depth_covariance,
    unproject_optical_z,
    validate_calibration,
)
from depth_visibility.da3_adapter import (
    run_analytic_conformance,
    run_two_group_conformance,
    validate_prediction,
)
from depth_visibility.errors import CameraConventionError, ContractError, NonFiniteError


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.K = intrinsics_matrix(50.0, 52.0, 31.5, 23.5)

    def test_opengl_flip_exactly_once_and_native_projection(self):
        c2w = np.eye(4)
        w2c = opengl_c2w_to_opencv_w2c(c2w)
        np.testing.assert_allclose(w2c, np.diag([1.0, -1.0, -1.0, 1.0]))
        pixel, z = project_world(self.K, w2c, np.array([0.0, 0.0, -2.0]))
        np.testing.assert_allclose(pixel, [31.5, 23.5], atol=1e-12)
        self.assertEqual(z, 2.0)
        recovered = unproject_optical_z(self.K, w2c, pixel, z)
        np.testing.assert_allclose(recovered, [0.0, 0.0, -2.0], atol=1e-12)

    def test_translated_yawed_roundtrip(self):
        angle = math.radians(31.0)
        rotation = np.array(
            [[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0],
             [-math.sin(angle), 0.0, math.cos(angle)]]
        )
        w2c = np.eye(4)
        w2c[:3, :3] = rotation
        w2c[:3, 3] = [0.4, -0.2, 0.1]
        pixels = np.array([[0.0, 0.0], [31.5, 23.5], [63.0, 47.0]])
        z = np.array([1.5, 2.0, 2.5])
        points = unproject_optical_z(self.K, w2c, pixels, z)
        recovered_pixels, recovered_z = project_world(self.K, w2c, points)
        np.testing.assert_allclose(recovered_pixels, pixels, atol=1e-10)
        np.testing.assert_allclose(recovered_z, z, atol=1e-12)
        expected_center = -(rotation.T @ w2c[:3, 3])
        np.testing.assert_allclose(camera_center(w2c), expected_center)

    def test_optical_z_is_not_ray_distance_off_axis(self):
        pixel = np.array([63.0, 47.0])
        point = unproject_optical_z(self.K, np.eye(4), pixel, 2.0)
        self.assertAlmostEqual(point[2], 2.0)
        self.assertGreater(float(np.linalg.norm(point)), 2.0)

    def test_align_corners_false_processed_intrinsics(self):
        transform = processed_image_transform(64, 48, 32, 24)
        expected = np.array([[0.5, 0.0, -0.25], [0.0, 0.5, -0.25], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(transform, expected)
        processed_k = transform @ self.K
        self.assertAlmostEqual(processed_k[0, 0], 25.0)
        self.assertAlmostEqual(processed_k[0, 2], 15.5)
        self.assertAlmostEqual(processed_k[1, 2], 11.5)

    def test_covariance_transport_is_symmetric_psd(self):
        covariance = transport_pixel_depth_covariance(
            self.K, np.eye(4), [31.5, 23.5], 2.0, 0.1
        )
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-15)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(covariance))), -1e-15)
        self.assertAlmostEqual(covariance[2, 2], 0.01)

    def test_fail_closed_camera_metadata_and_nonpositive_z(self):
        validate_calibration(self.K, np.eye(4), 64, 48, distortion=[0.0] * 5)
        with self.assertRaises(CameraConventionError):
            validate_calibration(self.K, np.eye(4), 64, 48, distortion=[0.0, 0.1])
        with self.assertRaises(CameraConventionError):
            validate_calibration(self.K, np.eye(4), 64, 48, rolling_shutter=True)
        with self.assertRaises(CameraConventionError):
            validate_calibration(
                self.K, np.eye(4), 64, 48,
                timestamps=[0.0, 2e-6], timestamp_tolerance_seconds=1e-6,
            )
        bad_k = self.K.copy()
        bad_k[0, 0] = np.nan
        with self.assertRaises(NonFiniteError):
            validate_calibration(bad_k, np.eye(4), 64, 48)
        with self.assertRaises(CameraConventionError):
            project_world(self.K, np.eye(4), [0.0, 0.0, 0.0])
        with self.assertRaises(CameraConventionError):
            unproject_optical_z(self.K, np.eye(4), [0.0, 0.0], -1.0)

    def test_da3_prediction_requires_complete_calibrated_provenance(self):
        complete = {
            "depth": np.ones((6, 4, 5), dtype=np.float32),
            "conf": np.ones((6, 4, 5), dtype=np.float32),
            "intrinsics": np.repeat(self.K[None, ...], 6, axis=0),
            "extrinsics": np.repeat(np.eye(4)[None, ...], 6, axis=0),
            "processed_images": np.zeros((6, 4, 5, 3), dtype=np.float32),
            "is_metric": 0,
            "scale_factor": 1.0,
        }
        validated = validate_prediction(complete, 6)
        self.assertEqual(validated["depth"].shape, (6, 4, 5))
        pinned_shape = dict(complete)
        pinned_shape["extrinsics"] = complete["extrinsics"][:, :3, :]
        normalized = validate_prediction(pinned_shape, 6)
        self.assertEqual(normalized["extrinsics"].shape, (6, 4, 4))
        np.testing.assert_array_equal(
            normalized["extrinsics"][:, 3, :],
            np.repeat(np.array([[0.0, 0.0, 0.0, 1.0]]), 6, axis=0),
        )
        for missing in ("conf", "intrinsics", "extrinsics", "processed_images"):
            incomplete = dict(complete)
            incomplete.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(ContractError):
                validate_prediction(incomplete, 6)

    def test_registered_analytic_conformance(self):
        report = run_analytic_conformance()
        self.assertLessEqual(report["center_corner_pixel_error_maximum"], 1e-10)
        self.assertLessEqual(report["optical_z_error_maximum"], 1e-12)
        self.assertLessEqual(report["translated_yawed_roundtrip_error_maximum"], 1e-9)

    def test_two_group_da3_conformance_repeats_and_compares_anchor(self):
        class FakeModel:
            def inference(self, images, *, extrinsics, intrinsics, **kwargs):
                self.last_kwargs = kwargs
                depth = np.stack([
                    np.ones((4, 5), dtype=np.float64)
                    * (1.0 if image == "cam01" else 2.0)
                    for image in images
                ])
                return {
                    "depth": depth,
                    "conf": np.ones_like(depth),
                    "intrinsics": np.asarray(intrinsics),
                    "extrinsics": np.asarray(extrinsics),
                    "processed_images": np.zeros((6, 4, 5, 3), dtype=np.uint8),
                    "is_metric": False,
                    "scale_factor": 1.0,
                }

        groups = []
        for members in (
            ("cam01", "cam02", "cam03", "cam04", "cam05", "cam06"),
            ("cam01", "cam07", "cam08", "cam09", "cam10", "cam11"),
        ):
            groups.append({
                "member_camera_ids": members,
                "images": list(members),
                "extrinsics_w2c": np.repeat(np.eye(4)[None, ...], 6, axis=0),
                "intrinsics": np.repeat(self.K[None, ...], 6, axis=0),
                "expected_processed_intrinsics": np.repeat(
                    self.K[None, ...], 6, axis=0
                ),
            })
        model = FakeModel()
        report = run_two_group_conformance(
            model, groups, anchor_camera_id="cam01"
        )
        self.assertTrue(report["numerically_repeatable"])
        self.assertEqual(report["anchor_cross_group_relative_mad_maximum"], 0.0)
        self.assertEqual(len(report["groups"]), 2)
        self.assertEqual(model.last_kwargs["ref_view_strategy"], "saddle_balanced")
        self.assertFalse(model.last_kwargs["infer_gs"])
        self.assertFalse(model.last_kwargs["use_ray_pose"])



if __name__ == "__main__":
    unittest.main()
