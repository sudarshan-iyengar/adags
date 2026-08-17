"""Pin the MAGNITUDE semantics of flow resampling.

A flow field is not an image. Resampling it to a different raster must
rescale the vectors as well as the grid: halving the width halves every
horizontal displacement, because a displacement of `d` source pixels IS a
displacement of `d/2` pixels once the raster is half as wide. A bare
`F.interpolate` moves the samples and leaves the numbers alone, so a
prediction resized that way is compared against targets in the wrong
units and the loss is silently wrong by the resize ratio.

`utils/motion_prior_utils.resize_flow` does it correctly.
`main.compute_flow_loss` used to resize the PREDICTION with a bare
`F.interpolate`; these tests pin the corrected behaviour.

Two layers, deliberately:

  * `ResizeFlowMagnitudeTests` and `ComputeFlowLossResizeTests` are the
    numeric pins. They need torch.
  * `ComputeFlowLossSourceTests` is a torch-free source-level guard, so
    that a revert to the magnitude-blind resize is caught even on a
    machine where `main` cannot be imported at all (it pulls in the
    CUDA-only `simple_knn` extension through `scene`).

A skip is not a pass: `EnvironmentDisclosureTests` records which layers
actually executed.
"""

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch

    from utils.motion_prior_utils import resize_flow

    HAVE_TORCH = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_TORCH = False

try:  # main imports scene -> simple_knn._C, which needs a GPU build
    from main import compute_flow_loss

    HAVE_MAIN = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_MAIN = False


def _constant_flow(height, width, dx, dy):
    """(2, H, W) field with a constant displacement everywhere.

    Constant is the right probe: bilinear resampling of a constant field
    is exactly that constant, so anything other than the pure rescale
    factor in the output is the resize's own doing.
    """
    flow = torch.zeros((2, height, width), dtype=torch.float32)
    flow[0] = dx
    flow[1] = dy
    return flow


@unittest.skipUnless(HAVE_TORCH, "torch is not importable in this environment")
class ResizeFlowMagnitudeTests(unittest.TestCase):
    def test_two_times_downsample_halves_the_magnitude(self):
        flow = _constant_flow(32, 48, dx=8.0, dy=-6.0)
        resized = resize_flow(flow.clone(), (16, 24))
        self.assertEqual(tuple(resized.shape), (2, 16, 24))
        self.assertAlmostEqual(float(resized[0].mean()), 4.0, places=5)
        self.assertAlmostEqual(float(resized[1].mean()), -3.0, places=5)
        self.assertAlmostEqual(float(resized[0].std()), 0.0, places=5)
        self.assertAlmostEqual(float(resized[1].std()), 0.0, places=5)

    def test_axes_scale_independently(self):
        # Halve the height only. The vertical component must halve and the
        # horizontal one must not move.
        flow = _constant_flow(32, 48, dx=8.0, dy=-6.0)
        resized = resize_flow(flow.clone(), (16, 48))
        self.assertAlmostEqual(float(resized[0].mean()), 8.0, places=5)
        self.assertAlmostEqual(float(resized[1].mean()), -3.0, places=5)

    def test_two_times_upsample_doubles_the_magnitude(self):
        flow = _constant_flow(16, 24, dx=3.0, dy=5.0)
        resized = resize_flow(flow.clone(), (32, 48))
        self.assertAlmostEqual(float(resized[0].mean()), 6.0, places=5)
        self.assertAlmostEqual(float(resized[1].mean()), 10.0, places=5)

    def test_identity_size_is_a_no_op(self):
        flow = _constant_flow(16, 24, dx=3.0, dy=5.0)
        resized = resize_flow(flow.clone(), (16, 24))
        self.assertTrue(torch.equal(resized, flow))

    def test_a_bare_interpolate_would_not_rescale(self):
        """The failure mode this pin exists for, stated positively."""
        import torch.nn.functional as F

        flow = _constant_flow(32, 48, dx=8.0, dy=-6.0)
        naive = F.interpolate(flow[None], size=(16, 24), mode="bilinear", align_corners=False)[0]
        self.assertAlmostEqual(float(naive[0].mean()), 8.0, places=5)
        correct = resize_flow(flow.clone(), (16, 24))
        self.assertAlmostEqual(float(correct[0].mean()), 4.0, places=5)
        self.assertNotAlmostEqual(float(naive[0].mean()), float(correct[0].mean()), places=3)


@unittest.skipUnless(HAVE_TORCH and HAVE_MAIN, "main is not importable (needs the CUDA build)")
class ComputeFlowLossResizeTests(unittest.TestCase):
    def test_downsampled_prediction_is_compared_in_target_units(self):
        # Prediction lives on a 2x raster and predicts 2x the displacement,
        # which is the SAME physical motion as the target. After a correct
        # resize the loss is zero; under a magnitude-blind resize it is the
        # full residual.
        pred = _constant_flow(32, 48, dx=8.0, dy=-6.0)
        target = _constant_flow(16, 24, dx=4.0, dy=-3.0)
        loss = compute_flow_loss(pred, target, None)
        self.assertIsNotNone(loss)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_magnitude_blind_resize_would_have_failed_this(self):
        pred = _constant_flow(32, 48, dx=8.0, dy=0.0)
        target = _constant_flow(16, 24, dx=4.0, dy=0.0)
        loss = float(compute_flow_loss(pred, target, None))
        # A bare interpolate would leave the prediction at 8.0 against a
        # target of 4.0, i.e. a mean absolute error of 4.0/2 channels = 2.0.
        self.assertLess(loss, 1e-4)

    def test_upsampled_prediction_is_also_rescaled(self):
        pred = _constant_flow(16, 24, dx=3.0, dy=5.0)
        target = _constant_flow(32, 48, dx=6.0, dy=10.0)
        self.assertAlmostEqual(float(compute_flow_loss(pred, target, None)), 0.0, places=5)

    def test_matching_shapes_are_untouched(self):
        pred = _constant_flow(16, 24, dx=3.0, dy=5.0)
        target = _constant_flow(16, 24, dx=3.0, dy=5.0)
        self.assertAlmostEqual(float(compute_flow_loss(pred, target, None)), 0.0, places=6)

    def test_gradient_flows_through_the_resize(self):
        pred = _constant_flow(32, 48, dx=8.0, dy=-6.0).requires_grad_(True)
        target = _constant_flow(16, 24, dx=1.0, dy=1.0)
        loss = compute_flow_loss(pred, target, None)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertGreater(float(pred.grad.abs().max()), 0.0)


class ComputeFlowLossSourceTests(unittest.TestCase):
    """Torch-free guard on `main.compute_flow_loss`'s resize call."""

    @staticmethod
    def _source():
        text = (REPO_ROOT / "main.py").read_text(encoding="utf-8")
        match = re.search(
            r"^def compute_flow_loss\(.*?(?=^def |\Z)", text, re.MULTILINE | re.DOTALL
        )
        if match is None:
            raise AssertionError("compute_flow_loss is no longer defined in main.py")
        return match.group(0)

    def test_uses_the_magnitude_aware_resize(self):
        self.assertIn("resize_flow(", self._source())

    def test_does_not_resize_the_prediction_with_a_bare_interpolate(self):
        self.assertNotIn("F.interpolate", self._source())

    def test_resize_flow_is_imported_by_main(self):
        text = (REPO_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertRegex(text, r"resize_flow")


class EnvironmentDisclosureTests(unittest.TestCase):
    def test_numeric_layers_actually_ran_somewhere(self):
        """A skipped pin is not a passing one."""
        if not HAVE_TORCH:
            self.skipTest(
                "torch absent: only the source-level guard ran here; the numeric "
                "resize pins must be run where torch is installed"
            )
        if not HAVE_MAIN:
            self.skipTest(
                "main not importable (it pulls in the CUDA-only simple_knn "
                "extension through scene): resize_flow was pinned numerically "
                "but compute_flow_loss was covered only by the source guard"
            )
        self.assertTrue(HAVE_TORCH and HAVE_MAIN)


if __name__ == "__main__":
    unittest.main()
