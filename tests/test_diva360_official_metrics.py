"""Pin `utils/diva360_official_metrics.py` against the real libraries.

The Apollo image has `cv2` but NOT `scikit-image` (probed directly), so
the SSIM under DiVa-360's conventions has to be reimplemented to run
there. That reimplementation is only worth anything if it is checked
against the thing it reproduces, so the skimage parity tests SKIP where
skimage is absent and run where it is present. The workstation's base
environment carries scikit-image 0.24.0, which is where they execute.

A skip is not a pass: `test_skimage_parity_actually_ran_somewhere`
records that plainly rather than letting a silently-skipped suite read
as a green one.
"""

import unittest

import numpy as np

from utils.diva360_official_metrics import (
    SSIM_C1,
    SSIM_C2,
    SSIM_WIN_SIZE,
    psnr_official,
    ssim_skimage_default,
    to_uint8_hwc,
)

try:  # the Apollo image has neither; the workstation base env has both
    from skimage.metrics import structural_similarity as _sk_ssim

    HAVE_SKIMAGE = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_SKIMAGE = False

try:
    import cv2 as _cv2

    HAVE_CV2 = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_CV2 = False


def _pair(seed: int, h: int = 64, w: int = 96):
    rng = np.random.default_rng(seed)
    gt = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    noise = rng.normal(0.0, 18.0, size=gt.shape)
    pred = np.clip(gt.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return gt, pred


class SsimParityTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_SKIMAGE, "scikit-image not installed")
    def test_matches_skimage_defaults_on_random_pairs(self):
        for seed in range(6):
            gt, pred = _pair(seed)
            with self.subTest(seed=seed):
                self.assertAlmostEqual(
                    ssim_skimage_default(gt, pred),
                    float(_sk_ssim(gt, pred, channel_axis=2)),
                    places=10,
                )

    @unittest.skipUnless(HAVE_SKIMAGE, "scikit-image not installed")
    def test_matches_skimage_on_identical_images(self):
        gt, _ = _pair(11)
        self.assertAlmostEqual(
            ssim_skimage_default(gt, gt), float(_sk_ssim(gt, gt, channel_axis=2)), places=10
        )
        self.assertAlmostEqual(ssim_skimage_default(gt, gt), 1.0, places=10)

    @unittest.skipUnless(HAVE_SKIMAGE, "scikit-image not installed")
    def test_matches_skimage_on_degenerate_and_structured_content(self):
        """Flat regions drive the variance terms to zero and are where a
        wrong covariance normalizer or a wrong constant shows up."""
        black = np.zeros((40, 40, 3), dtype=np.uint8)
        white = np.full((40, 40, 3), 255, dtype=np.uint8)
        bars = np.zeros((40, 40, 3), dtype=np.uint8)
        bars[:, ::4] = 255
        matte = np.zeros((40, 40, 3), dtype=np.uint8)
        matte[10:30, 10:30] = 200  # a black-composited object on black
        for name, a, b in (
            ("black/black", black, black),
            ("black/white", black, white),
            ("bars/black", bars, black),
            ("matte/bars", matte, bars),
            ("matte/matte", matte, matte),
        ):
            with self.subTest(case=name):
                self.assertAlmostEqual(
                    ssim_skimage_default(a, b),
                    float(_sk_ssim(a, b, channel_axis=2)),
                    places=10,
                )

    @unittest.skipUnless(HAVE_SKIMAGE, "scikit-image not installed")
    def test_matches_skimage_at_the_official_aspect_ratio(self):
        """1160x550 is DiVa-360's calibration space; a non-square image
        catches an axis swap that square fixtures cannot."""
        gt, pred = _pair(3, h=550, w=1160)
        self.assertAlmostEqual(
            ssim_skimage_default(gt, pred),
            float(_sk_ssim(gt, pred, channel_axis=2)),
            places=10,
        )

    def test_constants_are_the_skimage_defaults(self):
        self.assertEqual(SSIM_WIN_SIZE, 7)
        self.assertAlmostEqual(SSIM_C1, (0.01 * 255.0) ** 2, places=12)
        self.assertAlmostEqual(SSIM_C2, (0.03 * 255.0) ** 2, places=12)

    def test_is_not_the_3dgs_gaussian_ssim(self):
        """Guard against silently reverting to the 11x11 Gaussian window:
        on structured content the two conventions genuinely disagree, so
        an implementation that matched both would be wrong."""
        gt, pred = _pair(5)
        uniform = ssim_skimage_default(gt, pred)
        if HAVE_SKIMAGE:
            gaussian = float(
                _sk_ssim(
                    gt,
                    pred,
                    channel_axis=2,
                    gaussian_weights=True,
                    use_sample_covariance=False,
                    sigma=1.5,
                    data_range=255,
                )
            )
            self.assertNotAlmostEqual(uniform, gaussian, places=4)

    def test_shape_and_size_contracts(self):
        gt, pred = _pair(1)
        with self.assertRaises(ValueError):
            ssim_skimage_default(gt, pred[:, :-1])
        with self.assertRaises(ValueError):
            ssim_skimage_default(gt[:, :, 0], pred[:, :, 0])
        tiny = np.zeros((4, 4, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            ssim_skimage_default(tiny, tiny)


class PsnrTests(unittest.TestCase):
    def test_matches_the_closed_form(self):
        gt, pred = _pair(2)
        diff = gt.astype(np.float64) - pred.astype(np.float64)
        expected = 10.0 * np.log10(255.0**2 / float(np.mean(diff * diff)))
        self.assertAlmostEqual(psnr_official(gt, pred), expected, places=9)

    @unittest.skipUnless(HAVE_CV2, "cv2 not installed")
    def test_matches_cv2_directly(self):
        gt, pred = _pair(7)
        self.assertAlmostEqual(psnr_official(gt, pred), float(_cv2.PSNR(gt, pred)), places=9)

    def test_identical_images_are_infinite_in_every_environment(self):
        """The answer must not depend on whether cv2 is installed.

        This test previously asserted `inf` and passed only because the
        environment it ran in had no cv2; on Apollo, which does,
        `cv2.PSNR` returned 361.202 and it failed. That was a real defect
        in the module rather than in the test — the same input gave two
        different numbers in two environments — so the module now settles
        the case itself."""
        gt, _ = _pair(4)
        self.assertEqual(psnr_official(gt, gt), float("inf"))

    @unittest.skipUnless(HAVE_CV2, "cv2 not installed")
    def test_opencvs_own_zero_mse_guard_is_finite_and_is_not_used(self):
        """Pins the divergence rather than leaving it as folklore: this
        is the value delegation would have produced."""
        gt, _ = _pair(4)
        raw = float(_cv2.PSNR(gt, gt))
        self.assertTrue(raw < float("inf"))
        self.assertGreater(raw, 100.0)
        self.assertNotEqual(psnr_official(gt, gt), raw)

    def test_uint8_domain_differs_from_float_domain(self):
        """The float-domain PSNR this repository used is 20*log10(1/rmse)
        on [0,1] data. That is ALGEBRAICALLY the same number as the uint8
        form when the data are identical -- the real difference is the
        QUANTIZATION of the prediction, which this pins."""
        gt, pred = _pair(8)
        gt_f = gt.astype(np.float64) / 255.0
        pred_f = np.clip(pred.astype(np.float64) / 255.0 + 0.0011, 0.0, 1.0)
        mse_f = float(np.mean((gt_f - pred_f) ** 2))
        float_domain = 10.0 * np.log10(1.0 / mse_f)
        quantized = psnr_official(gt, to_uint8_hwc(pred_f))
        self.assertNotAlmostEqual(float_domain, quantized, places=3)

    def test_shape_mismatch_rejected(self):
        gt, pred = _pair(9)
        with self.assertRaises(ValueError):
            psnr_official(gt, pred[:-1])


class QuantizationTests(unittest.TestCase):
    def test_round_trip_of_exact_uint8_levels_is_lossless(self):
        levels = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
        as_float = levels.astype(np.float64) / 255.0
        np.testing.assert_array_equal(to_uint8_hwc(as_float), levels)

    def test_accepts_chw_hwc_and_batched(self):
        hwc = np.random.default_rng(0).random((12, 20, 3))
        chw = np.transpose(hwc, (2, 0, 1))
        np.testing.assert_array_equal(to_uint8_hwc(hwc), to_uint8_hwc(chw))
        np.testing.assert_array_equal(to_uint8_hwc(hwc), to_uint8_hwc(chw[None]))

    def test_alpha_channel_is_dropped_not_scored(self):
        rgba = np.random.default_rng(1).random((10, 10, 4))
        np.testing.assert_array_equal(to_uint8_hwc(rgba), to_uint8_hwc(rgba[:, :, :3]))

    def test_rounds_half_up_like_torchvision_save_image(self):
        half = np.full((8, 8, 3), 0.5 / 255.0)
        self.assertEqual(int(to_uint8_hwc(half)[0, 0, 0]), 1)

    def test_clamps_out_of_range(self):
        wild = np.array([[[-3.0, 0.5, 9.0]]])
        out = to_uint8_hwc(wild)
        self.assertEqual(list(out[0, 0]), [0, 128, 255])

    def test_rejects_non_image_shapes(self):
        with self.assertRaises(ValueError):
            to_uint8_hwc(np.zeros((5, 5)))
        with self.assertRaises(ValueError):
            to_uint8_hwc(np.zeros((2, 3, 8, 8)))


class EnvironmentDisclosureTests(unittest.TestCase):
    def test_skimage_parity_actually_ran_somewhere(self):
        """A skipped parity test is not a passing one. This records which
        environment the suite ran in, so 'all green' on Apollo is never
        mistaken for 'the skimage reimplementation was verified there'."""
        if not HAVE_SKIMAGE:
            self.skipTest(
                "scikit-image absent (expected on the Apollo image): the SSIM "
                "parity pin did NOT run here and must be run where skimage exists"
            )
        self.assertTrue(HAVE_SKIMAGE)


if __name__ == "__main__":
    unittest.main()
