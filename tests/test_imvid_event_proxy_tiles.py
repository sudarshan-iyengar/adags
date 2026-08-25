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


def _grid_tiled(grid: np.ndarray, tile_size: int = 60) -> dict:
    """A minimal ``tiled`` payload wrapped around a SYNTHETIC tile grid.

    Only the keys ``per_tile_camera_candidates`` reads, and the spans come
    from the real ``tile_edges`` so the emitted pixel boxes are the real ones.
    Building the grid directly is what lets a test control exactly which tiles
    fire at which sample -- which is the only practical way to exercise the
    48-of-144 cap without a 960x540 fixture.
    """
    n_frames, n_y, n_x = np.asarray(grid).shape
    return {
        "tile_template_dist": np.asarray(grid, dtype=np.float64),
        "row_spans": proxy.tile_edges(n_y * tile_size, tile_size),
        "col_spans": proxy.tile_edges(n_x * tile_size, tile_size),
        "n_tile_y": int(n_y),
        "n_tile_x": int(n_x),
        "tile_size": int(tile_size),
        "proxy_raster": [n_x * tile_size, n_y * tile_size],
    }


def _bumped_grid(n_frames: int, n_y: int, n_x: int, tiles, plateau, *,
                 base: float = 1.0, bump: float = 5.0) -> np.ndarray:
    """Constant grid with a raised plateau in the named tiles.

    Constant off the plateau on purpose: a constant tile series has MAD 0 and
    a threshold equal to its own median, and ``signal > threshold`` is strict,
    so it can never fire. That leaves the FIRING SET exactly equal to
    ``tiles``, which is what the cap and connectivity tests need.
    """
    grid = np.full((n_frames, n_y, n_x), float(base), dtype=np.float64)
    lo, hi = plateau
    for row, col in tiles:
        grid[lo:hi, row, col] += float(bump)
    return grid


#: PER-TILE CENSUS FIXTURE, in INTEGER grey levels because the proxy tree is
#: written as 8-bit PNGs. Structure mirrors the P1-REL precondition: ONE loud
#: distractor tile whose symmetric excursion drives its own median AND its own
#: MAD so high that its threshold exceeds its own maximum, plus a QUIET target
#: spanning three face-adjacent tiles.
PT_RASTER = (120, 180)                  # 2 x 3 tiles at 60 px
PT_FRAMES = 12
PT_BASE = 40
PT_DISTRACTOR_TILE = (1, 0)
PT_DISTRACTOR_SPAN = 42
PT_TARGET_TILES = ((0, 0), (0, 1), (0, 2))
PT_TARGET_PLATEAU = (5, 8)
PT_TARGET_PATCH = 32
PT_TARGET_CONTRAST = 25


def _per_tile_census_stack(seed: int = 11) -> np.ndarray:
    """Loud distractor + quiet 3-tile target, all INTEGER grey levels."""
    height, width = PT_RASTER
    rng = np.random.default_rng(seed)
    stack = (np.full((PT_FRAMES, height, width), float(PT_BASE),
                     dtype=np.float32)
             + rng.integers(0, 3, size=(PT_FRAMES, height, width)
                            ).astype(np.float32))
    row, col = PT_DISTRACTOR_TILE
    for t in range(PT_FRAMES):
        offset = float(round(PT_DISTRACTOR_SPAN * t / (PT_FRAMES - 1)))
        stack[t, row * 60:row * 60 + 60, col * 60:col * 60 + 60] += offset
    lo, hi = PT_TARGET_PLATEAU
    pad = (60 - PT_TARGET_PATCH) // 2
    for row, col in PT_TARGET_TILES:
        y0, x0 = row * 60 + pad, col * 60 + pad
        stack[lo:hi, y0:y0 + PT_TARGET_PATCH,
              x0:x0 + PT_TARGET_PATCH] += float(PT_TARGET_CONTRAST)
    return stack


def _per_tile_args(proxy_root: Path, **overrides) -> argparse.Namespace:
    base = {
        "per_tile_mode": True,
        "per_tile_floor_f": proxy.DEFAULT_PER_TILE_FLOOR_F,
        "per_tile_min_component": proxy.DEFAULT_PER_TILE_MIN_COMPONENT,
        "per_tile_max_firing_tiles": proxy.DEFAULT_PER_TILE_MAX_FIRING_TILES,
    }
    base.update(overrides)
    return _census_args(proxy_root, **base)


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


class PerTileNoiseScaleAndFloorTests(unittest.TestCase):
    """v2 section 3: the absolute floor is declared on NOISE, not on events."""

    def test_noise_scale_is_1_4826_times_the_temporal_mad_per_tile(self):
        #  tile (0,0) series 0,1,2,3,4 -> median 2, |dev| 2,1,0,1,2 -> MAD 1
        grid = np.zeros((5, 1, 2), dtype=np.float64)
        grid[:, 0, 0] = [0.0, 1.0, 2.0, 3.0, 4.0]
        grid[:, 0, 1] = 7.0                       # constant -> MAD 0
        scales = proxy.per_tile_noise_scale(grid)
        self.assertEqual(scales.shape, (1, 2))
        self.assertAlmostEqual(float(scales[0, 0]), 1.4826, places=12)
        self.assertEqual(float(scales[0, 1]), 0.0)

    def test_noise_scale_refuses_a_bad_grid(self):
        with self.assertRaises(ContractError):
            proxy.per_tile_noise_scale(np.zeros((1, 2, 2)))
        with self.assertRaises(ContractError):
            proxy.per_tile_noise_scale(np.zeros((4, 4)))

    def test_floor_is_F_times_the_pooled_median_noise_scale(self):
        scales = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        record = proxy.per_tile_absolute_floor(scales, 3.0)
        self.assertEqual(record["noise_scale_median_grey_levels"], 2.5)
        self.assertEqual(record["floor_grey_levels"], 7.5)
        self.assertEqual(record["n_tile_series"], 4)
        self.assertEqual(record["F"], 3.0)
        self.assertTrue(
            record["F_is_a_declared_judgment_not_derived_from_data"])
        self.assertFalse(record["floor_is_degenerate_zero"])

    def test_the_median_pools_over_all_camera_window_tile_series(self):
        """The rule says 'over all (camera, window, tile)', so it pools."""
        one = proxy.per_tile_absolute_floor([np.array([1.0, 1.0, 1.0])], 3.0)
        pooled = proxy.per_tile_absolute_floor(
            [np.array([1.0, 1.0, 1.0]), np.array([9.0, 9.0, 9.0]),
             np.array([9.0, 9.0, 9.0])], 3.0)
        self.assertEqual(one["noise_scale_median_grey_levels"], 1.0)
        self.assertEqual(pooled["noise_scale_median_grey_levels"], 9.0)
        self.assertEqual(pooled["n_tile_series"], 9)
        self.assertEqual(pooled["n_camera_window_grids"], 3)

    def test_degenerate_and_refused_cases(self):
        zero = proxy.per_tile_absolute_floor([np.zeros(16)], 3.0)
        self.assertEqual(zero["floor_grey_levels"], 0.0)
        self.assertTrue(zero["floor_is_degenerate_zero"])
        self.assertEqual(zero["n_zero_scale_tile_series"], 16)
        with self.assertRaises(ContractError):
            proxy.per_tile_absolute_floor([], 3.0)
        with self.assertRaises(ContractError):
            proxy.per_tile_absolute_floor([np.array([])], 3.0)
        with self.assertRaises(ContractError):
            proxy.per_tile_absolute_floor([np.array([1.0])], 0.0)
        with self.assertRaises(ContractError):
            proxy.per_tile_absolute_floor([np.array([np.nan])], 3.0)

    def test_the_frozen_constants_are_what_the_spec_says(self):
        self.assertEqual(proxy.DEFAULT_PER_TILE_FLOOR_F, 3.0)
        self.assertEqual(proxy.PER_TILE_FLOOR_F_SWEEP, (1.5, 2.0, 3.0, 4.5, 6.0))
        self.assertEqual(proxy.DEFAULT_PER_TILE_MIN_COMPONENT, 3)
        self.assertEqual(proxy.DEFAULT_PER_TILE_MAX_FIRING_TILES, 48)
        #  33% of the 16x9 census grid, and the grid is still 144 tiles.
        self.assertEqual(proxy.DEFAULT_PER_TILE_MAX_FIRING_TILES, 144 // 3)


class PerTileRelativeGateTests(unittest.TestCase):
    """v2 section 2, and the whole reason the reduction had to change."""

    @staticmethod
    def _loud_and_quiet(n_frames=24):
        """One loud tile with a symmetric excursion, one quiet tile with a
        small plateau, and a strictly positive MAD everywhere."""
        rng = np.random.default_rng(3)
        grid = 0.75 + rng.uniform(-0.01, 0.01, size=(n_frames, 2, 4))
        grid[:, 1, 0] += np.linspace(0.0, 43.0, n_frames)     # LOUD
        grid[4:8, 0, 0] += 7.0                                # quiet target
        grid[4:8, 0, 1] += 7.0
        grid[4:8, 0, 2] += 7.0
        return grid

    def test_a_loud_tile_does_not_raise_a_quiet_tiles_threshold(self):
        grid = self._loud_and_quiet()
        without = grid.copy()
        without[:, 1, 0] = grid[:, 1, 3]        # replace the loud tile
        quiet_with = proxy.robust_threshold(grid[:, 0, 1], proxy.DEFAULT_K_MAD)
        quiet_without = proxy.robust_threshold(without[:, 0, 1],
                                               proxy.DEFAULT_K_MAD)
        #  EXACTLY equal: the gate is computed from the tile's own series and
        #  nothing else can enter it.
        self.assertEqual(quiet_with["median"], quiet_without["median"])
        self.assertEqual(quiet_with["mad_scaled"], quiet_without["mad_scaled"])
        self.assertEqual(quiet_with["threshold"], quiet_without["threshold"])

        #  ... while the MAX-over-tiles reduction is moved enormously.
        max_with = proxy.robust_threshold(grid.max(axis=(1, 2)),
                                          proxy.DEFAULT_K_MAD)
        max_without = proxy.robust_threshold(without.max(axis=(1, 2)),
                                             proxy.DEFAULT_K_MAD)
        self.assertGreater(max_with["threshold"],
                           50 * max_without["threshold"])
        self.assertGreater(max_with["threshold"], quiet_with["threshold"] * 20)

    def test_the_max_threshold_exceeds_its_own_signal_maximum(self):
        """The audited cam12 signature: 73.99 against a maximum of 72.19."""
        grid = self._loud_and_quiet()
        reduced = grid.max(axis=(1, 2))
        stats = proxy.robust_threshold(reduced, proxy.DEFAULT_K_MAD)
        self.assertGreater(stats["threshold"], float(np.max(reduced)))
        self.assertEqual(len(proxy.detect_changepoints(
            reduced, [n * STRIDE for n in range(len(reduced))],
            k_mad=proxy.DEFAULT_K_MAD, min_amplitude=0.0, stride=STRIDE)), 0)

    def test_per_tile_finds_the_quiet_target_the_max_reduction_cannot(self):
        grid = self._loud_and_quiet()
        frames = [n * STRIDE for n in range(grid.shape[0])]
        result = proxy.per_tile_camera_candidates(
            _grid_tiled(grid), frames, k_mad=proxy.DEFAULT_K_MAD,
            min_amplitude=0.1, stride=STRIDE)
        self.assertEqual([c["polarity"] for c in result["candidates"]],
                         ["rise", "fall"])
        self.assertEqual([c["source_frame"] for c in result["candidates"]],
                         [4 * STRIDE, 8 * STRIDE])
        self.assertEqual(result["candidates"][0]["component_tiles_row_col"],
                         [[0, 0], [0, 1], [0, 2]])
        #  the loud tile's own gate holds it: it never fires
        self.assertNotIn([1, 0], result["candidates"][0]["firing_tiles_row_col"])


class GridConnectedComponentTests(unittest.TestCase):
    """v2 section 4: FACE-adjacent, and diagonals are not faces."""

    def test_diagonal_tiles_are_not_adjacent(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = mask[1, 1] = mask[2, 2] = mask[3, 3] = True
        components = proxy.grid_connected_components(mask)
        self.assertEqual([len(c) for c in components], [1, 1, 1, 1])

    def test_face_adjacent_runs_and_shapes(self):
        row = np.zeros((3, 5), dtype=bool)
        row[1, 1] = row[1, 2] = row[1, 3] = True
        self.assertEqual(proxy.grid_connected_components(row),
                         [[(1, 1), (1, 2), (1, 3)]])
        column = np.zeros((4, 2), dtype=bool)
        column[0, 1] = column[1, 1] = column[2, 1] = True
        self.assertEqual([len(c) for c in
                          proxy.grid_connected_components(column)], [3])
        elbow = np.zeros((3, 3), dtype=bool)
        elbow[0, 0] = elbow[1, 0] = elbow[1, 1] = True
        self.assertEqual(proxy.grid_connected_components(elbow),
                         [[(0, 0), (1, 0), (1, 1)]])

    def test_two_components_separated_by_one_gap(self):
        mask = np.zeros((1, 7), dtype=bool)
        mask[0, 0] = mask[0, 1] = mask[0, 2] = True
        mask[0, 4] = mask[0, 5] = True
        self.assertEqual([len(c) for c in
                          proxy.grid_connected_components(mask)], [3, 2])

    def test_a_diagonal_chain_never_reaches_the_minimum_component(self):
        """Three scattered false positives must NOT pass as one event."""
        mask = np.zeros((9, 16), dtype=bool)
        for row, col in ((1, 1), (2, 2), (3, 3), (5, 9), (6, 10)):
            mask[row, col] = True
        self.assertTrue(all(len(c) < proxy.DEFAULT_PER_TILE_MIN_COMPONENT
                            for c in proxy.grid_connected_components(mask)))

    def test_wraparound_is_not_adjacency(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = mask[0, 2] = mask[2, 0] = True
        self.assertEqual([len(c) for c in
                          proxy.grid_connected_components(mask)], [1, 1, 1])

    def test_empty_full_and_refusals(self):
        self.assertEqual(proxy.grid_connected_components(
            np.zeros((3, 3), dtype=bool)), [])
        self.assertEqual([len(c) for c in proxy.grid_connected_components(
            np.ones((3, 4), dtype=bool))], [12])
        with self.assertRaises(ContractError):
            proxy.grid_connected_components(np.ones(4, dtype=bool))
        with self.assertRaises(ContractError):
            proxy.grid_connected_components(np.ones((2, 2, 2), dtype=bool))


class PerTileSpatialCoherenceTests(unittest.TestCase):
    """v2 section 4, on the real candidate builder over a 9x16 grid."""

    FRAMES = 10
    PLATEAU = (4, 7)

    def _run(self, tiles, **overrides):
        grid = _bumped_grid(self.FRAMES, 9, 16, tiles, self.PLATEAU)
        frames = [n * STRIDE for n in range(self.FRAMES)]
        kwargs = {"k_mad": proxy.DEFAULT_K_MAD, "min_amplitude": 0.5,
                  "stride": STRIDE}
        kwargs.update(overrides)
        return proxy.per_tile_camera_candidates(_grid_tiled(grid), frames,
                                                **kwargs)

    def test_a_constant_grid_off_the_plateau_fires_only_the_bumped_tiles(self):
        result = self._run([(0, 0), (0, 1), (0, 2)])
        self.assertEqual([c["n_tiles_firing_at_sample"]
                          for c in result["candidates"]], [3, 3])
        self.assertEqual(result["n_tiles_with_events"], 3)

    def test_two_tiles_is_below_the_minimum_component(self):
        result = self._run([(0, 0), (0, 1)])
        self.assertEqual(result["candidates"], [])
        reasons = {r["reason"] for r in result["rejected"]}
        self.assertEqual(reasons,
                         {"component_smaller_than_min_component_tiles"})
        self.assertEqual(result["rejected"][0]["component_size_tiles"], 2)

    def test_three_diagonal_tiles_are_rejected_but_three_in_a_row_are_not(self):
        diagonal = self._run([(0, 0), (1, 1), (2, 2)])
        self.assertEqual(diagonal["candidates"], [])
        self.assertEqual(len(diagonal["rejected"]), 6)   # 3 tiles x rise+fall
        straight = self._run([(0, 0), (1, 0), (2, 0)])
        self.assertEqual(len(straight["candidates"]), 2)
        self.assertEqual(straight["candidates"][0]["component_size_tiles"], 3)

    def test_the_48_of_144_cap_is_exclusive(self):
        forty_eight = [(r, c) for r in range(3) for c in range(16)]
        self.assertEqual(len(forty_eight), 48)
        capped = self._run(forty_eight)
        self.assertEqual(capped["candidates"], [])
        self.assertEqual({r["reason"] for r in capped["rejected"]},
                         {"too_many_tiles_firing_global_change"})
        self.assertEqual(capped["rejected"][0]["n_tiles_firing_at_sample"], 48)

        forty_seven = forty_eight[:-1]
        admitted = self._run(forty_seven)
        self.assertEqual(len(admitted["candidates"]), 2)
        self.assertEqual(admitted["candidates"][0]["n_tiles_firing_at_sample"],
                         47)
        self.assertEqual(admitted["candidates"][0]["component_size_tiles"], 47)

    def test_the_cap_and_the_component_read_the_SAME_firing_set(self):
        """A big global change plus a small real one is still capped out."""
        tiles = ([(r, c) for r in range(3) for c in range(15)]
                 + [(8, 0), (8, 1), (8, 2)])
        self.assertEqual(len(tiles), 48)
        result = self._run(tiles)
        self.assertEqual(result["candidates"], [])

    def test_candidate_carries_the_full_spatial_explanation(self):
        result = self._run([(0, 0), (0, 1), (0, 2)])
        candidate = result["candidates"][0]
        for key in ("component_size_tiles", "n_tiles_firing_at_sample",
                    "n_tiles_in_grid", "component_tiles_row_col",
                    "firing_tiles_row_col", "per_tile_amplitude",
                    "per_tile_threshold", "per_tile_median",
                    "per_tile_mad_scaled", "per_tile_polarity",
                    "tile_pixel_boxes_xyxy", "component_pixel_bbox_xyxy",
                    "max_tile_amplitude"):
            self.assertIn(key, candidate)
        self.assertEqual(candidate["n_tiles_in_grid"], 144)
        self.assertEqual(len(candidate["per_tile_amplitude"]), 3)
        self.assertEqual(candidate["tile_pixel_boxes_xyxy"],
                         [[0, 0, 60, 60], [60, 0, 120, 60], [120, 0, 180, 60]])
        self.assertEqual(candidate["component_pixel_bbox_xyxy"],
                         [0, 0, 180, 60])
        #  the labelling the review gallery must not be able to drop
        self.assertFalse(candidate["is_instance_mask"])
        self.assertEqual(candidate["kind"], "detector_signal_explanation")
        self.assertIn("NOT AN INSTANCE MASK", candidate["what_this_is"])
        self.assertTrue(
            candidate["pixel_boxes_are_tile_extent_not_object_extent"])
        json.dumps(result["candidates"], sort_keys=True)

    def test_amplitude_is_the_mean_over_the_components_own_tiles(self):
        grid = _bumped_grid(self.FRAMES, 9, 16, [], self.PLATEAU)
        lo, hi = self.PLATEAU
        grid[lo:hi, 0, 0] += 2.0
        grid[lo:hi, 0, 1] += 4.0
        grid[lo:hi, 0, 2] += 6.0
        result = proxy.per_tile_camera_candidates(
            _grid_tiled(grid), [n * STRIDE for n in range(self.FRAMES)],
            k_mad=proxy.DEFAULT_K_MAD, min_amplitude=0.5, stride=STRIDE)
        candidate = result["candidates"][0]
        self.assertAlmostEqual(candidate["amplitude"], 4.0, places=9)
        self.assertAlmostEqual(candidate["max_tile_amplitude"], 6.0, places=9)
        self.assertEqual(candidate["per_tile_amplitude"], [2.0, 4.0, 6.0])

    def test_polarity_is_a_recorded_majority_never_an_ordering_constraint(self):
        result = self._run([(0, 0), (0, 1), (0, 2)])
        for candidate in result["candidates"]:
            self.assertEqual(candidate["n_tiles_rise"] +
                             candidate["n_tiles_fall"], 3)
            self.assertIn("polarity_is_tied", candidate)
        self.assertEqual([c["polarity"] for c in result["candidates"]],
                         ["rise", "fall"])

    def test_refusals(self):
        grid = _bumped_grid(self.FRAMES, 9, 16, [(0, 0)], self.PLATEAU)
        frames = [n * STRIDE for n in range(self.FRAMES)]
        with self.assertRaises(ContractError):
            proxy.per_tile_camera_candidates(
                _grid_tiled(grid), frames[:-1], k_mad=3.0, min_amplitude=0.5,
                stride=STRIDE)
        with self.assertRaises(ContractError):
            proxy.per_tile_camera_candidates(
                _grid_tiled(grid), frames, k_mad=3.0, min_amplitude=0.5,
                stride=STRIDE, min_component_tiles=0)
        with self.assertRaises(ContractError):
            proxy.per_tile_camera_candidates(
                _grid_tiled(grid), frames, k_mad=3.0, min_amplitude=0.5,
                stride=STRIDE, max_firing_tiles=0)
        with self.assertRaises(ContractError):
            proxy.per_tile_camera_candidates(
                _grid_tiled(np.zeros((1, 9, 16))), [0], k_mad=3.0,
                min_amplitude=0.5, stride=STRIDE)


class CensusPerTileModeTests(unittest.TestCase):
    """End to end over a real (tiny) proxy tree, in per-tile mode."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.root = root / "pertile"
        _write_proxy_root(cls.root, _per_tile_census_stack())
        #  whole-frame change: the DEFAULT census already sees this one
        cls.big_root = root / "big"
        big = np.full((8, 120, 180), 40.0, dtype=np.float32)
        big[3:6] += 60.0
        _write_proxy_root(cls.big_root, big)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_per_tile_mode_off_is_byte_identical_to_the_pre_v2_census(self):
        legacy = proxy.mode_census(_census_args(self.big_root))
        with_flags_off = proxy.mode_census(
            _census_args(self.big_root, per_tile_mode=False,
                         per_tile_floor_f=proxy.DEFAULT_PER_TILE_FLOOR_F,
                         per_tile_min_component=3,
                         per_tile_max_firing_tiles=48))
        self.assertEqual(json.dumps(legacy, sort_keys=True, default=str),
                         json.dumps(with_flags_off, sort_keys=True,
                                    default=str))
        blob = json.dumps(legacy, sort_keys=True, default=str)
        self.assertNotIn("per_tile", blob)
        self.assertNotIn("tile_", blob)
        #  and the pre-v2 ranking is untouched when the mode is off
        clusters = legacy["windows"][0]["candidate_clusters"]
        self.assertEqual([c["n_cameras_supporting"] for c in clusters],
                         sorted([c["n_cameras_supporting"] for c in clusters],
                                reverse=True))

    def test_tile_mode_and_per_tile_mode_may_not_be_combined(self):
        with self.assertRaises(ContractError):
            proxy.mode_census(_per_tile_args(
                self.root, tile_mode=True, tile_size=60,
                tile_min_amplitude=2.0, tile_top_n=8))

    def test_the_default_census_and_tile_max_both_miss_the_quiet_target(self):
        """The NEUTER's premise, measured rather than asserted."""
        plain = proxy.mode_census(_census_args(self.root))
        self.assertEqual(plain["windows"][0]["n_candidates_total"], 0)
        tiled_report = proxy.mode_census(
            _census_args(self.root, tile_mode=True, tile_size=60,
                         tile_min_amplitude=proxy.DEFAULT_TILE_MIN_AMPLITUDE,
                         tile_top_n=4))
        self.assertEqual(tiled_report["windows"][0]["n_candidates_total"], 0)

    def test_neuter_per_tile_must_not_be_a_max_over_tiles_in_disguise(self):
        """FAILS if the per-tile reduction silently becomes tile_max.

        The fixture is built so the max-over-tiles threshold EXCEEDS its own
        signal maximum -- the audited cam12 signature -- so a fallback to that
        reduction returns a structurally guaranteed ZERO.
        """
        report = proxy.mode_census(_per_tile_args(self.root))
        window = report["windows"][0]
        self.assertGreater(window["n_candidates_total"], 0)

        #  and, on the very same window, the max reduction finds nothing
        stack = _per_tile_census_stack()
        tiled = proxy.tiled_template_signals(stack, 60)
        reduced = tiled["tile_max"]
        stats = proxy.robust_threshold(reduced, proxy.DEFAULT_K_MAD)
        self.assertGreater(stats["threshold"], float(np.max(reduced)))
        self.assertEqual(len(proxy.detect_changepoints(
            reduced, [n * STRIDE for n in range(len(reduced))],
            k_mad=proxy.DEFAULT_K_MAD, min_amplitude=0.0, stride=STRIDE)), 0)
        self.assertEqual(
            report["per_tile_pass"]["signal"].count("INDEPENDENTLY"), 1)

    def test_per_tile_census_finds_the_target_where_it_was_injected(self):
        report = proxy.mode_census(_per_tile_args(self.root))
        window = report["windows"][0]
        self.assertEqual(window["n_candidates_total"], 4)   # 2 cameras x 2
        self.assertEqual(window["n_clusters"], 2)
        self.assertEqual(window["n_clusters_multi_camera"], 2)
        frames = sorted(c["source_frame_median"]
                        for c in window["candidate_clusters"])
        self.assertEqual(frames, [PT_TARGET_PLATEAU[0] * STRIDE,
                                  PT_TARGET_PLATEAU[1] * STRIDE])
        for cluster in window["candidate_clusters"]:
            self.assertEqual(cluster["signal_used_for_detection"],
                             "per_tile_template_dist")
            self.assertEqual(cluster["per_camera_component_size_tiles"],
                             {"cam00": 3, "cam01": 3})
            self.assertEqual(
                cluster["per_camera_component_tiles_row_col"]["cam00"],
                [list(rc) for rc in PT_TARGET_TILES])
            self.assertFalse(cluster["is_instance_mask"])
            self.assertIn("NOT AN INSTANCE MASK",
                          cluster["tile_explanation_note"])
        json.dumps(report, sort_keys=True)

    def test_the_manifest_records_the_measured_floor_and_the_sweep(self):
        report = proxy.mode_census(_per_tile_args(self.root))
        block = report["per_tile_pass"]
        self.assertTrue(block["enabled"])
        self.assertTrue(block["is_primary_reading"])
        self.assertEqual(block["primary_reading_F"], 3.0)
        self.assertEqual(block["tile_size_px"], 60)
        self.assertEqual(block["tile_grid"], [3, 2])
        self.assertIn("NOT AN INSTANCE MASK", block["note"])
        self.assertIn("EXHAUSTIVE RECALL",
                      block["windowing_defect_not_repaired_here"])

        floor = block["absolute_floor"]
        self.assertEqual(floor["F"], 3.0)
        self.assertTrue(floor["F_is_a_declared_judgment_not_derived_from_data"])
        #  THE MEDIAN POOLS OVER (camera, window, tile), exactly as the rule
        #  says: 2 cameras x 2 windows x 6 tiles.
        self.assertEqual(floor["n_tile_series"], 2 * 2 * 6)
        self.assertEqual(floor["n_camera_window_grids"], 2 * 2)
        self.assertAlmostEqual(
            floor["floor_grey_levels"],
            3.0 * floor["noise_scale_median_grey_levels"], places=12)
        self.assertGreater(floor["floor_grey_levels"], 0.0)

        sweep = block["sensitivity_sweep"]
        self.assertEqual([row["F"] for row in sweep],
                         list(proxy.PER_TILE_FLOOR_F_SWEEP))
        self.assertEqual(sum(1 for row in sweep if row["is_primary_reading"]), 1)
        primary = [row for row in sweep if row["is_primary_reading"]][0]
        self.assertEqual(primary["F"], 3.0)
        self.assertEqual(primary["floor_grey_levels"],
                         floor["floor_grey_levels"])
        self.assertEqual(primary["n_per_camera_candidates"],
                         report["windows"][0]["n_candidates_total"])
        for row in sweep:
            self.assertIn("floor_grey_levels", row)
            self.assertIn("n_clusters", row)
            self.assertIn("n_clusters_at_min_cameras", row)
            if not row["is_primary_reading"]:
                self.assertIn("SENSITIVITY PROBE", row["label"])
        floors = [row["floor_grey_levels"] for row in sweep]
        self.assertEqual(floors, sorted(floors))

    def test_a_non_primary_F_is_labelled_as_a_probe(self):
        report = proxy.mode_census(_per_tile_args(self.root,
                                                  per_tile_floor_f=6.0))
        block = report["per_tile_pass"]
        self.assertFalse(block["is_primary_reading"])
        self.assertFalse(block["absolute_floor"]["is_primary_reading"])
        self.assertEqual(block["primary_reading_F"], 3.0)

    def test_a_high_floor_rejects_the_target_as_an_absolute_gate(self):
        report = proxy.mode_census(_per_tile_args(self.root,
                                                  per_tile_floor_f=5000.0))
        floor = report["per_tile_pass"]["absolute_floor"]["floor_grey_levels"]
        #  above the injected target's own tile amplitude, so the ABSOLUTE
        #  gate -- not the relative one, not coherence -- does the rejecting
        self.assertGreater(floor, PT_TARGET_CONTRAST * PT_TARGET_PATCH ** 2
                           / 60 ** 2)
        self.assertEqual(report["windows"][0]["n_candidates_total"], 0)

    def test_a_single_tile_false_positive_is_rejected_by_coherence(self):
        """The coherence gate, doing its job on a REAL background fluke."""
        report = proxy.mode_census(_per_tile_args(self.root))
        window = report["windows"][0]
        #  4 tiles carry crossings: the 3 target tiles plus one background
        #  tile that fluked past its own ~3 sigma gate.
        self.assertEqual(window["per_tile_n_tiles_with_events"],
                         {"cam00": 4, "cam01": 4})
        rejected = window["per_tile_rejected_by_spatial_coherence"]["cam00"]
        self.assertEqual([r["component_size_tiles"] for r in rejected], [1])
        self.assertEqual([r["reason"] for r in rejected],
                         ["component_smaller_than_min_component_tiles"])
        #  ... and it never reaches the candidate list
        for cluster in window["candidate_clusters"]:
            self.assertEqual(cluster["per_camera_component_size_tiles"],
                             {"cam00": 3, "cam01": 3})

    def test_rejections_are_recorded_never_silently_dropped(self):
        report = proxy.mode_census(_per_tile_args(self.root,
                                                  per_tile_min_component=4))
        window = report["windows"][0]
        self.assertEqual(window["n_candidates_total"], 0)
        rejected = window["per_tile_rejected_by_spatial_coherence"]["cam00"]
        self.assertEqual({r["reason"] for r in rejected},
                         {"component_smaller_than_min_component_tiles"})
        #  the 3-tile target is now BELOW the raised minimum, and the fact is
        #  recorded with its measured size rather than dropped
        self.assertIn(3, [r["component_size_tiles"] for r in rejected])
        self.assertIn(1, [r["component_size_tiles"] for r in rejected])

    def test_per_camera_signals_keep_the_global_signal_beside_the_per_tile(self):
        report = proxy.mode_census(_per_tile_args(self.root,
                                                  emit_signals=True))
        signals = report["windows"][0]["per_camera_signals"]["cam00"]
        for key in ("absdiff_mean", "template_dist", "changed_frac",
                    "threshold", "signal_used_for_detection", "n_tiles",
                    "n_tiles_with_events", "n_rejected_by_spatial_coherence",
                    "per_tile_noise_scale_median"):
            self.assertIn(key, signals)
        self.assertEqual(signals["signal_used_for_detection"],
                         "per_tile_template_dist")
        self.assertEqual(signals["n_tiles"], 6)
        self.assertEqual(signals["n_tiles_with_events"], 4)
        self.assertEqual(signals["n_rejected_by_spatial_coherence"], 1)
        #  The whole-frame signal is reported unchanged beside the per-tile
        #  one, and on this fixture it shows the SAME monopolisation defect:
        #  the loud region drives its median AND its MAD so its own threshold
        #  exceeds its own maximum, exactly as measured on cam12.
        self.assertGreater(signals["threshold"]["threshold"],
                           max(signals["template_dist"]))


class PerTileRankingTests(unittest.TestCase):
    """v2 section 8: camera count is RETIRED as the primary order."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "ranking"
        n_frames, height, width = 12, 120, 180
        rng = np.random.default_rng(5)
        base = (np.full((n_frames, height, width), 40.0, dtype=np.float32)
                + rng.integers(0, 3, size=(n_frames, height, width)
                               ).astype(np.float32))
        #  LOW amplitude, seen by ALL THREE cameras, tiles (0,0..2) at j=4
        for col in range(3):
            base[4:6, 14:46, col * 60 + 14:col * 60 + 46] += 6.0
        three_camera_only = base.copy()
        #  HIGH amplitude, seen by TWO cameras only, tiles (1,0..2) at j=8
        two_camera = base.copy()
        for col in range(3):
            two_camera[8:10, 66:114, col * 60 + 6:col * 60 + 54] += 40.0
        _write_proxy_root(cls.root, two_camera, cameras=("cam00", "cam01"))
        _write_proxy_root(cls.root, three_camera_only, cameras=("cam02",))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_clusters_are_ranked_by_mean_amplitude_not_by_camera_count(self):
        report = proxy.mode_census(_per_tile_args(self.root))
        clusters = report["windows"][0]["candidate_clusters"]
        amplitudes = [c["mean_amplitude"] for c in clusters]
        self.assertEqual(amplitudes, sorted(amplitudes, reverse=True))

        top = clusters[0]
        low = [c for c in clusters if c["n_cameras_supporting"] == 3]
        self.assertTrue(low, "the 3-camera cluster must exist")
        #  THE INVERSION: the top-ranked cluster has FEWER supporting cameras
        #  than a lower-ranked one, so camera count demonstrably is not the
        #  sort key.
        self.assertEqual(top["n_cameras_supporting"], 2)
        self.assertGreater(top["mean_amplitude"], low[0]["mean_amplitude"])
        self.assertGreater(clusters.index(low[0]), 0)

    def test_c_min_is_a_corroboration_floor_and_never_filters_the_output(self):
        report = proxy.mode_census(_per_tile_args(self.root, min_cameras=3))
        window = report["windows"][0]
        clusters = window["candidate_clusters"]
        self.assertGreater(len(clusters), window["n_clusters_multi_camera"])
        #  the 2-camera cluster is still emitted, and still ranked first
        self.assertEqual(clusters[0]["n_cameras_supporting"], 2)
        ranking = report["per_tile_pass"]["ranking"]
        self.assertTrue(
            ranking["c_min_is_a_corroboration_floor_never_a_sort_key"])
        self.assertTrue(ranking["camera_count_is_retired_as_the_primary_order"])

    def test_the_default_path_still_ranks_by_camera_count(self):
        """Additive: the pre-v2 ordering is untouched with the mode off."""
        report = proxy.mode_census(_census_args(self.root))
        clusters = report["windows"][0]["candidate_clusters"]
        counts = [c["n_cameras_supporting"] for c in clusters]
        self.assertEqual(counts, sorted(counts, reverse=True))


class TileSelfTestModeTests(unittest.TestCase):
    def test_cli_tile_selftest_passes(self):
        self.assertEqual(proxy.main(["--tile-selftest"]), 0)

    def test_cli_self_test_still_passes(self):
        self.assertEqual(proxy.main(["--self-test"]), 0)

    def test_cli_per_tile_selftest_passes(self):
        self.assertEqual(proxy.main(["--per-tile-selftest"]), 0)


if __name__ == "__main__":
    unittest.main()
