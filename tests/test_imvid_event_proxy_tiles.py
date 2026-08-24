"""CPU tests for the TILE PASS of scripts/imvid_event_proxy.py.

Run with:
    python -m unittest tests.test_imvid_event_proxy_tiles

No torch, no cv2, no ffmpeg, no media: proxy PNGs are written with the
script's own self-test encoder and read back with its own reader, so the
fixtures exercise the real decode path.

WHY THIS PASS EXISTS. All three signals in ``frame_signals`` are WHOLE-FRAME
MEANS, so the instrument is blind to small objects -- a small object cannot
move a whole-frame mean. Until a region-sensitive pass has run, no zero-event
result from this census is exhaustive. These tests pin the frozen
preconditions P1-P5 of that pass, and pin that the pass is ADDITIVE: with
``--tile-mode`` off the census is exactly what it was.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402
from scripts import imvid_event_proxy as proxy  # noqa: E402

STRIDE = 30
SOURCE_RATE = "60000/1001"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _patch_window(*, height: int, width: int, n_frames: int,
                  plateau: tuple[int, int], patch: int, contrast: float,
                  top_left: tuple[int, int], base: float = 40.0) -> np.ndarray:
    """Flat window with a square patch present over a run of frames."""
    stack = np.full((n_frames, height, width), base, dtype=np.float32)
    y0, x0 = top_left
    lo, hi = plateau
    stack[lo:hi, y0:y0 + patch, x0:x0 + patch] += contrast
    return stack


def _census_args(proxy_root: Path, **overrides) -> argparse.Namespace:
    base = {
        "proxy_root": str(proxy_root),
        "limit_cameras": None,
        "window_frames": 300,
        "k_mad": proxy.DEFAULT_K_MAD,
        "min_amplitude": proxy.DEFAULT_MIN_AMPLITUDE,
        "match_tol_frames": None,
        "min_cameras": 2,
        "emit_signals": False,
        "top": 20,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _write_proxy_root(root: Path, stack: np.ndarray, cameras=("cam00", "cam01")):
    """A minimal but REAL proxy tree: per-camera manifest + src_*.png frames."""
    n_frames, height, width = stack.shape
    for camera in cameras:
        frames_dir = root / camera / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for j in range(n_frames):
            source_frame = j * STRIDE
            plane = np.clip(stack[j], 0, 255).astype(np.uint8)[:, :, None]
            (frames_dir / f"src_{source_frame:06d}.png").write_bytes(
                proxy._encode_png(plane, 0))
        (root / camera / proxy.PROXY_MANIFEST_NAME).write_text(json.dumps({
            "schema": "imvid-event-proxy-camera-v1",
            "camera": camera,
            "proxy_raster": [width, height],
            "mapping": {"stride_frames": STRIDE,
                        "source_rate_exact": SOURCE_RATE},
        }), encoding="utf-8")


#: The declared P1 scale, restated here so the test is readable on its own.
P1_PATCH = 32
P1_CONTRAST = 25.0


class TileGeometryTests(unittest.TestCase):
    """P4: the tile partition covers every pixel exactly once."""

    def _assert_exact_cover(self, height: int, width: int, tile_size: int):
        rows = proxy.tile_edges(height, tile_size)
        cols = proxy.tile_edges(width, tile_size)
        cover = np.zeros((height, width), dtype=np.int32)
        for y0, y1 in rows:
            for x0, x1 in cols:
                cover[y0:y1, x0:x1] += 1
        self.assertEqual(int(cover.min()), 1, f"gap at {width}x{height}")
        self.assertEqual(int(cover.max()), 1, f"double count at {width}x{height}")
        self.assertEqual(sum(y1 - y0 for y0, y1 in rows), height)
        self.assertEqual(sum(x1 - x0 for x0, x1 in cols), width)
        self.assertEqual(rows[0][0], 0)
        self.assertEqual(rows[-1][1], height)

    def test_p4_exact_cover_census_raster(self):
        self._assert_exact_cover(540, 960, proxy.DEFAULT_TILE_SIZE)
        self.assertEqual(len(proxy.tile_edges(540, 60)), 9)
        self.assertEqual(len(proxy.tile_edges(960, 60)), 16)

    def test_p4_exact_cover_non_divisible(self):
        for height, width in ((100, 130), (61, 59), (1, 1), (541, 961),
                              (270, 480)):
            self._assert_exact_cover(height, width, proxy.DEFAULT_TILE_SIZE)

    def test_p4_partial_edge_tile_is_kept_short_not_padded(self):
        spans = proxy.tile_edges(130, 60)
        self.assertEqual(spans, [(0, 60), (60, 120), (120, 130)])
        self.assertEqual(spans[-1][1] - spans[-1][0], 10)

    def test_p4_refusals(self):
        with self.assertRaises(ContractError):
            proxy.tile_edges(100, 0)
        with self.assertRaises(ContractError):
            proxy.tile_edges(0, 60)
        with self.assertRaises(ContractError):
            proxy.tile_edges(100, -3)

    def test_pixel_box_matches_spans(self):
        rows = proxy.tile_edges(130, 60)
        cols = proxy.tile_edges(200, 60)
        self.assertEqual(proxy.tile_pixel_box(0, 0, rows, cols), [0, 0, 60, 60])
        self.assertEqual(proxy.tile_pixel_box(2, 3, rows, cols),
                         [180, 120, 200, 130])


class TileSignalTests(unittest.TestCase):
    """P1, P3, P5 on the signal function itself."""

    def test_p1_tile_detects_declared_scale_and_global_misses(self):
        #  Declared BEFORE any number is read: a 32x32 px patch at 25 grey
        #  levels on the 960x540 census raster, wholly inside one 60 px tile.
        stack = _patch_window(height=540, width=960, n_frames=12,
                              plateau=(4, 8), patch=P1_PATCH,
                              contrast=P1_CONTRAST, top_left=(134, 314))
        source_frames = [n * STRIDE for n in range(12)]
        global_signals = proxy.frame_signals(stack)
        tiled = proxy.tiled_template_signals(stack, proxy.DEFAULT_TILE_SIZE)

        global_value = float(global_signals["template_dist"][5])
        tile_value = float(tiled["tile_max"][5])
        self.assertAlmostEqual(global_value,
                               P1_CONTRAST * P1_PATCH ** 2 / (540 * 960),
                               places=6)
        self.assertAlmostEqual(tile_value,
                               P1_CONTRAST * P1_PATCH ** 2 / 60 ** 2, places=9)
        #  The blindness, quantified: the global value is 40x under its floor.
        self.assertLess(global_value, proxy.DEFAULT_MIN_AMPLITUDE)
        self.assertGreater(tile_value, proxy.DEFAULT_TILE_MIN_AMPLITUDE)
        self.assertAlmostEqual(tile_value / global_value, 144.0, places=3)

        global_events = proxy.detect_changepoints(
            global_signals["template_dist"], source_frames,
            k_mad=proxy.DEFAULT_K_MAD,
            min_amplitude=proxy.DEFAULT_MIN_AMPLITUDE, stride=STRIDE)
        tile_events = proxy.detect_changepoints(
            tiled["tile_max"], source_frames, k_mad=proxy.DEFAULT_K_MAD,
            min_amplitude=proxy.DEFAULT_TILE_MIN_AMPLITUDE, stride=STRIDE)
        self.assertEqual(len(global_events), 0)
        self.assertEqual([e["polarity"] for e in tile_events], ["rise", "fall"])
        self.assertEqual(tile_events[0]["source_frame"], 4 * STRIDE)
        self.assertEqual(tile_events[1]["source_frame"], 8 * STRIDE)
        self.assertEqual([int(v) for v in tiled["tile_argmax"][5]], [2, 5])

    def test_grid_shape_and_argmax_are_consistent(self):
        stack = _patch_window(height=540, width=960, n_frames=12,
                              plateau=(4, 8), patch=P1_PATCH,
                              contrast=P1_CONTRAST, top_left=(134, 314))
        tiled = proxy.tiled_template_signals(stack)
        self.assertEqual(tiled["tile_template_dist"].shape, (12, 9, 16))
        self.assertEqual(tiled["tile_max"].shape, (12,))
        self.assertEqual(tiled["tile_argmax"].shape, (12, 2))
        self.assertEqual(tiled["proxy_raster"], [960, 540])
        grid = tiled["tile_template_dist"]
        for t in range(grid.shape[0]):
            i, j = (int(v) for v in tiled["tile_argmax"][t])
            self.assertAlmostEqual(float(grid[t, i, j]),
                                   float(tiled["tile_max"][t]), places=12)

    def test_p3_finite_and_non_constant_where_change_exists(self):
        stack = _patch_window(height=120, width=180, n_frames=8,
                              plateau=(3, 6), patch=P1_PATCH,
                              contrast=P1_CONTRAST, top_left=(14, 74))
        tiled = proxy.tiled_template_signals(stack)
        self.assertTrue(np.all(np.isfinite(tiled["tile_template_dist"])))
        self.assertTrue(np.all(np.isfinite(tiled["tile_max"])))
        self.assertGreater(float(np.std(tiled["tile_max"])), 0.0)
        self.assertEqual(float(tiled["tile_max"][0]), 0.0)

    def test_p2_flat_and_subfloor_noise_produce_no_candidates(self):
        source_frames = [n * STRIDE for n in range(8)]
        flat = np.full((8, 120, 180), 40.0, dtype=np.float32)
        flat_tiled = proxy.tiled_template_signals(flat)
        self.assertEqual(float(np.max(flat_tiled["tile_template_dist"])), 0.0)
        self.assertEqual(len(proxy.detect_changepoints(
            flat_tiled["tile_max"], source_frames, k_mad=proxy.DEFAULT_K_MAD,
            min_amplitude=proxy.DEFAULT_TILE_MIN_AMPLITUDE, stride=STRIDE)), 0)

        rng = np.random.default_rng(7)
        noisy = (np.full((8, 120, 180), 40.0, dtype=np.float32)
                 + rng.uniform(-0.5, 0.5, size=(8, 120, 180)).astype(np.float32))
        noisy_tiled = proxy.tiled_template_signals(noisy)
        self.assertLess(float(np.max(noisy_tiled["tile_max"])),
                        proxy.DEFAULT_TILE_MIN_AMPLITUDE)
        self.assertEqual(len(proxy.detect_changepoints(
            noisy_tiled["tile_max"], source_frames, k_mad=proxy.DEFAULT_K_MAD,
            min_amplitude=proxy.DEFAULT_TILE_MIN_AMPLITUDE, stride=STRIDE)), 0)

    def test_p5_neuter_replacing_the_max_by_a_mean_destroys_detection(self):
        """FAILS if the load-bearing tile MAXIMUM is silently made a mean."""
        stack = _patch_window(height=540, width=960, n_frames=12,
                              plateau=(4, 8), patch=P1_PATCH,
                              contrast=P1_CONTRAST, top_left=(134, 314))
        source_frames = [n * STRIDE for n in range(12)]
        tiled = proxy.tiled_template_signals(stack)
        grid = tiled["tile_template_dist"]

        self.assertTrue(np.array_equal(tiled["tile_max"], grid.max(axis=(1, 2))),
                        "tile_max must be the MAXIMUM over tiles")
        self.assertFalse(np.allclose(tiled["tile_max"], grid.mean(axis=(1, 2))),
                         "tile_max must not be a mean over tiles")
        #  A tile mean is an area-weighted whole-frame mean again, i.e. the
        #  very blindness this pass removes.
        neutered = grid.mean(axis=(1, 2))
        self.assertAlmostEqual(
            float(neutered[5]),
            float(proxy.frame_signals(stack)["template_dist"][5]), places=6)
        self.assertEqual(len(proxy.detect_changepoints(
            neutered, source_frames, k_mad=proxy.DEFAULT_K_MAD,
            min_amplitude=proxy.DEFAULT_TILE_MIN_AMPLITUDE, stride=STRIDE)), 0)
        self.assertEqual(len(proxy.detect_changepoints(
            tiled["tile_max"], source_frames, k_mad=proxy.DEFAULT_K_MAD,
            min_amplitude=proxy.DEFAULT_TILE_MIN_AMPLITUDE, stride=STRIDE)), 2)

    def test_edge_tile_is_a_mean_over_its_own_pixels_only(self):
        """A patch inside a SHORT edge tile is still measured, unweighted."""
        #  130x70 -> x tiles 0..60, 60..120, 120..130 and y tiles 0..60,
        #  60..70, so tile (1, 2) is a 10x10 corner.
        stack = np.full((8, 70, 130), 40.0, dtype=np.float32)
        stack[3:6, 60:70, 120:130] += 25.0      # fills the 10x10 corner tile
        tiled = proxy.tiled_template_signals(stack)
        self.assertEqual(tiled["tile_template_dist"].shape, (8, 2, 3))
        self.assertEqual([int(v) for v in tiled["tile_argmax"][4]], [1, 2])
        #  Mean over the tile's OWN 10x10 = 100 px, so the full contrast --
        #  the short edge tile is NOT area-weighted back down.
        self.assertAlmostEqual(float(tiled["tile_max"][4]), 25.0, places=9)
        self.assertEqual(proxy.tile_pixel_box(1, 2, tiled["row_spans"],
                                              tiled["col_spans"]),
                         [120, 60, 130, 70])
        #  the neighbouring 60x10 edge tile sees only its own share
        self.assertAlmostEqual(float(tiled["tile_template_dist"][4, 0, 2]),
                               0.0, places=12)

    def test_refuses_a_window_with_fewer_than_two_frames(self):
        with self.assertRaises(ContractError):
            proxy.tiled_template_signals(np.zeros((1, 8, 8), dtype=np.float32))
        with self.assertRaises(ContractError):
            proxy.tiled_template_signals(np.zeros((8, 8), dtype=np.float32))

    def test_frame_signals_is_untouched(self):
        """The global path must be behaviourally byte-identical."""
        stack = np.full((10, 16, 16), 40.0, dtype=np.float32)
        stack[3:7] += 60.0
        signals = proxy.frame_signals(stack)
        self.assertEqual(sorted(signals),
                         ["absdiff_mean", "changed_frac", "template_dist"])
        self.assertEqual(len(signals["template_dist"]), 10)
        self.assertAlmostEqual(float(signals["template_dist"][0]), 0.0, places=6)
        self.assertAlmostEqual(float(signals["template_dist"][4]), 60.0, places=6)
        self.assertAlmostEqual(float(signals["absdiff_mean"][0]), 0.0, places=9)
        self.assertAlmostEqual(float(signals["absdiff_mean"][3]), 60.0, places=6)
        self.assertAlmostEqual(float(signals["changed_frac"][4]), 1.0, places=9)


class TileExplanationTests(unittest.TestCase):
    """Deliverables 3 and 4: the overlay payload and the side-by-side global."""

    def setUp(self):
        self.stack = _patch_window(height=540, width=960, n_frames=12,
                                   plateau=(4, 8), patch=P1_PATCH,
                                   contrast=P1_CONTRAST, top_left=(134, 314))
        self.tiled = proxy.tiled_template_signals(self.stack)
        self.global_signals = proxy.frame_signals(self.stack)
        self.explanation = proxy.tile_candidate_explanation(
            self.tiled, 5, source_frame=150,
            top_n=proxy.DEFAULT_TILE_TOP_N,
            global_template_dist=float(self.global_signals["template_dist"][5]),
            global_template_dist_before=float(
                self.global_signals["template_dist"][3]),
            global_min_amplitude=proxy.DEFAULT_MIN_AMPLITUDE,
            tile_min_amplitude=proxy.DEFAULT_TILE_MIN_AMPLITUDE)

    def test_documented_shape(self):
        expected = {
            "what_this_is", "is_instance_mask", "kind", "source_frame",
            "proxy_index_in_window", "tile_size_px", "proxy_raster",
            "grid_shape", "tile_template_dist_grid", "tile_max",
            "tile_argmax_row_col", "tile_argmax_pixel_box_xyxy", "top_tiles",
            "global_template_dist_at_candidate",
            "global_template_dist_before_candidate",
            "tile_max_over_global_template_dist",
            "global_absolute_floor_grey_levels",
            "tile_absolute_floor_grey_levels",
            "global_pass_would_clear_its_own_floor",
        }
        self.assertEqual(set(self.explanation), expected)
        self.assertEqual(self.explanation["grid_shape"], [9, 16])
        self.assertEqual(len(self.explanation["tile_template_dist_grid"]), 9)
        self.assertTrue(all(len(row) == 16 for row
                            in self.explanation["tile_template_dist_grid"]))
        self.assertEqual(self.explanation["tile_size_px"], 60)
        self.assertEqual(self.explanation["proxy_raster"], [960, 540])

    def test_it_says_it_is_not_an_instance_mask(self):
        self.assertFalse(self.explanation["is_instance_mask"])
        self.assertEqual(self.explanation["kind"],
                         "detector_signal_explanation")
        self.assertIn("NOT AN INSTANCE MASK", self.explanation["what_this_is"])
        self.assertIn("NOT AN INSTANCE MASK",
                      proxy.tiled_template_signals.__doc__)
        self.assertIn("NOT AN INSTANCE MASK",
                      proxy.tile_candidate_explanation.__doc__)
        self.assertTrue(all(t["pixel_box_is_tile_extent_not_object_extent"]
                            for t in self.explanation["top_tiles"]))

    def test_top_tiles_are_ranked_with_boxes(self):
        top = self.explanation["top_tiles"]
        self.assertEqual(len(top), proxy.DEFAULT_TILE_TOP_N)
        values = [t["value"] for t in top]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertEqual([top[0]["tile_row"], top[0]["tile_col"]], [2, 5])
        self.assertEqual(top[0]["pixel_box_xyxy"], [300, 120, 360, 180])
        self.assertEqual(top[0]["n_pixels"], 3600)
        #  the injected patch lies inside the winning box
        x0, y0, x1, y1 = top[0]["pixel_box_xyxy"]
        self.assertTrue(x0 <= 314 and y0 <= 134 and x1 >= 346 and y1 >= 166)
        self.assertEqual(self.explanation["tile_argmax_pixel_box_xyxy"],
                         top[0]["pixel_box_xyxy"])

    def test_global_signal_is_reported_side_by_side(self):
        self.assertAlmostEqual(
            self.explanation["global_template_dist_at_candidate"],
            0.049383, places=5)
        self.assertAlmostEqual(self.explanation["tile_max"], 7.1111, places=4)
        self.assertAlmostEqual(
            self.explanation["tile_max_over_global_template_dist"],
            144.0, places=1)
        self.assertFalse(
            self.explanation["global_pass_would_clear_its_own_floor"])
        self.assertEqual(self.explanation["global_absolute_floor_grey_levels"],
                         proxy.DEFAULT_MIN_AMPLITUDE)
        self.assertEqual(self.explanation["tile_absolute_floor_grey_levels"],
                         proxy.DEFAULT_TILE_MIN_AMPLITUDE)

    def test_is_json_serializable(self):
        blob = json.dumps(self.explanation, sort_keys=True)
        self.assertEqual(json.loads(blob)["grid_shape"], [9, 16])


class CensusTileModeTests(unittest.TestCase):
    """End to end over a real (tiny) proxy tree."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.small_root = root / "small"
        #  small patch: below the GLOBAL floor, above the TILE floor
        _write_proxy_root(cls.small_root,
                          _patch_window(height=120, width=180, n_frames=8,
                                        plateau=(3, 6), patch=P1_PATCH,
                                        contrast=P1_CONTRAST,
                                        top_left=(14, 74)))
        cls.big_root = root / "big"
        #  whole-frame change: the DEFAULT census already sees this one
        big = np.full((8, 120, 180), 40.0, dtype=np.float32)
        big[3:6] += 60.0
        _write_proxy_root(cls.big_root, big)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_tile_mode_off_is_byte_identical_to_the_pre_tile_census(self):
        legacy = proxy.mode_census(_census_args(self.big_root))
        with_flag_off = proxy.mode_census(
            _census_args(self.big_root, tile_mode=False,
                         tile_size=proxy.DEFAULT_TILE_SIZE,
                         tile_min_amplitude=proxy.DEFAULT_TILE_MIN_AMPLITUDE,
                         tile_top_n=proxy.DEFAULT_TILE_TOP_N))
        self.assertEqual(json.dumps(legacy, sort_keys=True, default=str),
                         json.dumps(with_flag_off, sort_keys=True, default=str))
        self.assertNotIn("tile_pass", legacy)
        blob = json.dumps(legacy, sort_keys=True, default=str)
        self.assertNotIn("tile_", blob)
        #  and the default census still finds the whole-frame change
        window = legacy["windows"][0]
        self.assertEqual(window["n_candidates_total"], 4)   # 2 cameras x rise+fall
        self.assertEqual(window["n_clusters_multi_camera"], 2)

    def test_tile_mode_off_misses_the_small_patch(self):
        report = proxy.mode_census(_census_args(self.small_root))
        self.assertEqual(report["windows"][0]["n_candidates_total"], 0)

    def test_tile_mode_on_finds_it_and_explains_where(self):
        report = proxy.mode_census(
            _census_args(self.small_root, tile_mode=True,
                         tile_size=proxy.DEFAULT_TILE_SIZE,
                         tile_min_amplitude=proxy.DEFAULT_TILE_MIN_AMPLITUDE,
                         tile_top_n=4))
        window = report["windows"][0]
        self.assertEqual(window["n_candidates_total"], 4)
        self.assertEqual(window["n_clusters"], 2)
        self.assertEqual(window["n_clusters_multi_camera"], 2)

        tile_pass = report["tile_pass"]
        self.assertTrue(tile_pass["enabled"])
        self.assertEqual(tile_pass["tile_size_px"], 60)
        self.assertEqual(tile_pass["tile_min_amplitude_grey_levels"], 2.0)
        self.assertEqual(tile_pass["tile_grid"], [3, 2])
        self.assertIn("NOT AN INSTANCE MASK", tile_pass["note"])

        explanations = window["tile_explanations"]
        self.assertEqual(sorted(explanations), ["cam00", "cam01"])
        first = explanations["cam00"][0]
        self.assertEqual(first["grid_shape"], [2, 3])
        self.assertEqual(first["tile_argmax_row_col"], [0, 1])
        self.assertEqual(first["tile_argmax_pixel_box_xyxy"], [60, 0, 120, 60])
        self.assertEqual(len(first["top_tiles"]), 4)
        self.assertAlmostEqual(first["tile_max"], 7.1111, places=4)
        #  DELIVERABLE 4: the global signal at the same frame, side by side
        self.assertAlmostEqual(first["global_template_dist_at_candidate"],
                               25.0 * 32 * 32 / (120 * 180), places=4)
        self.assertFalse(first["global_pass_would_clear_its_own_floor"])
        self.assertIn("NOT AN INSTANCE MASK", window["tile_explanations_note"])

        cluster = window["candidate_clusters"][0]
        self.assertEqual(cluster["signal_used_for_detection"], "tile_max")
        self.assertEqual(cluster["per_camera_tile_argmax_row_col"],
                         {"cam00": [0, 1], "cam01": [0, 1]})
        self.assertEqual(sorted(cluster["per_camera_global_template_dist"]),
                         ["cam00", "cam01"])
        json.dumps(report, sort_keys=True)          # must stay serializable

    def test_tile_mode_keeps_the_global_signal_in_per_camera_signals(self):
        report = proxy.mode_census(
            _census_args(self.small_root, tile_mode=True, emit_signals=True,
                         tile_size=proxy.DEFAULT_TILE_SIZE,
                         tile_min_amplitude=proxy.DEFAULT_TILE_MIN_AMPLITUDE,
                         tile_top_n=proxy.DEFAULT_TILE_TOP_N))
        signals = report["windows"][0]["per_camera_signals"]["cam00"]
        for key in ("absdiff_mean", "template_dist", "changed_frac",
                    "threshold", "tile_max", "tile_argmax_row_col",
                    "tile_threshold", "signal_used_for_detection"):
            self.assertIn(key, signals)
        self.assertEqual(signals["signal_used_for_detection"], "tile_max")
        self.assertEqual(len(signals["tile_max"]), 8)
        self.assertLess(max(signals["template_dist"]),
                        proxy.DEFAULT_MIN_AMPLITUDE)
        self.assertGreater(max(signals["tile_max"]),
                           proxy.DEFAULT_TILE_MIN_AMPLITUDE)

    def test_tile_floor_is_honoured_as_an_absolute_gate(self):
        """Raising the tile floor above the measured tile mean rejects it."""
        report = proxy.mode_census(
            _census_args(self.small_root, tile_mode=True,
                         tile_size=proxy.DEFAULT_TILE_SIZE,
                         tile_min_amplitude=20.0,
                         tile_top_n=proxy.DEFAULT_TILE_TOP_N))
        self.assertEqual(report["windows"][0]["n_candidates_total"], 0)


class TileSelfTestModeTests(unittest.TestCase):
    def test_cli_tile_selftest_passes(self):
        self.assertEqual(proxy.main(["--tile-selftest"]), 0)

    def test_cli_self_test_still_passes(self):
        self.assertEqual(proxy.main(["--self-test"]), 0)


if __name__ == "__main__":
    unittest.main()
