"""PSNR pooling semantics (stg-n3v-protocol-parity-2026-08-19).

`utils.image_utils.psnr` views its input as (B, -1): an unbatched
(3, H, W) image is split BY CHANNEL and the caller's `.mean()` averages
three per-channel PSNRs, which exceeds the pooled PSNR by
10*log10(AM/GM) over per-channel MSEs (always >= 0; measured at
+0.268 dB on a real run). Both ADAGS eval sites now pass (1, 3, H, W).
These tests pin the semantics so the call sites cannot silently regress.
"""

import math
import re
import unittest
from pathlib import Path

import torch

from utils.image_utils import psnr

REPO_ROOT = Path(__file__).resolve().parents[1]


class PsnrPoolingTests(unittest.TestCase):
    def _images(self):
        gen = torch.Generator().manual_seed(5)
        gt = torch.rand(3, 8, 8, generator=gen)
        # deliberately UNEQUAL per-channel error so AM != GM
        pred = gt.clone()
        pred[0] += 0.20
        pred[1] += 0.02
        pred = pred.clamp(0, 1)
        return pred, gt

    def test_batched_input_gives_the_pooled_psnr(self):
        pred, gt = self._images()
        pooled = float(psnr(pred.unsqueeze(0), gt.unsqueeze(0)).mean())
        mse = float(((pred - gt) ** 2).mean())
        expected = 20 * math.log10(1.0 / math.sqrt(mse))
        self.assertAlmostEqual(pooled, expected, places=5)

    def test_unbatched_input_is_channel_split_and_biased_upward(self):
        """Anti-vacuity: the defect shape still exists for unbatched
        input, and its bias is strictly positive when channel MSEs
        differ — which is why the call sites must batch."""
        pred, gt = self._images()
        channel_split = float(psnr(pred, gt).mean())
        pooled = float(psnr(pred.unsqueeze(0), gt.unsqueeze(0)).mean())
        self.assertGreater(channel_split, pooled + 1e-4)

    def test_eval_call_sites_pass_batched_input(self):
        main_src = (REPO_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn(
            "psnr(pred.unsqueeze(0), gt.unsqueeze(0))", main_src
        )
        mesh_src = (REPO_ROOT / "utils" / "mesh_utils.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "psnr(gt_image.unsqueeze(0), rgb.unsqueeze(0))", mesh_src
        )
        # the old unbatched forms must be gone from live (non-comment) code
        def live_hits(source, pattern):
            return [
                line for line in source.splitlines()
                if re.search(pattern, line)
                and not line.lstrip().startswith("#")
            ]

        self.assertEqual(live_hits(main_src, r"psnr\(pred, gt\)"), [])
        self.assertEqual(live_hits(mesh_src, r"psnr\(gt_image, rgb\)"), [])


if __name__ == "__main__":
    unittest.main()
