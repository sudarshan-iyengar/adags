"""CPU tests for scripts/imvid_build_gallery.py.

Run with:
    python -m unittest tests.test_imvid_build_gallery

No torch, no cv2, no ffmpeg, no media, no network. The fixtures build a real
proxy tree (per-camera ``MANIFEST.imvid_proxy.json`` plus ``src_*.png``
frames written with the census script's own PNG encoder) and a real census
JSON in the census script's own schema, then run the gallery builder over
them and read the produced HTML back.

WHAT THESE TESTS ARE FOR. The gallery is the artefact a human uses to decide
what is and is not an event, so its failure modes are all failures of
HONESTY rather than of computation:

* a wrong timestamp silently mislabels every picture. The NEUTER test below
  asserts the exact rational-derived value and would FAIL if the builder ever
  assumed 60.0 fps -- at 60000/1001 those differ from the first frame;
* a silent cap turns "40 of 300 candidates" into "the candidates". The drop
  notices are asserted as VISIBLE page text, not merely as manifest fields;
* a spatial overlay is the artefact a reader mistakes for an instance mask,
  so the "not an instance mask" wording is pinned;
* zero candidates is a RESULT. A gallery that errored on it would push the
  reader toward re-running until something appeared;
* THE ORIGINAL PER-TILE DEFECT WAS NOT A CRASH. The gallery was written
  against the v1 ``--tile-mode`` census and then handed a v2
  ``--per-tile-mode`` one. It read ``n_cameras_supporting`` and
  ``tile_explanations`` through ``.get()``, found neither in the shape it
  expected, and DEGRADED SILENTLY: it produced exactly the ranking
  [[operations/imvid-tile-scout-v2-per-tile-2026-08-25]] §8 forbids, with no
  spatial overlay, and reported success. So the tests below assert two things
  a passing build cannot show on its own -- that a schema it does not
  understand is REFUSED rather than rendered, and that the per-tile order is
  amplitude-descending with camera count absent from the sort key entirely.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402
from scripts import imvid_build_gallery as gallery  # noqa: E402
from scripts import imvid_event_proxy as proxy  # noqa: E402

#: ImViD's measured rate. 60000/1001 = 59.94005994..., NOT 60.
SOURCE_RATE = "60000/1001"
STRIDE = 30
RASTER = (120, 68)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _write_proxy_tree(root: Path, cameras, n_proxy_frames: int = 12,
                      raster=RASTER) -> None:
    """A real proxy tree: manifests plus PNG frames named by SOURCE index."""
    width, height = raster
    rng = np.random.default_rng(0)
    for index, camera in enumerate(cameras):
        frames_dir = root / camera / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for j in range(n_proxy_frames):
            source_frame = j * STRIDE
            plane = np.full((height, width), 40.0, dtype=np.float32)
            plane += rng.normal(0.0, 1.0, size=plane.shape).astype(np.float32)
            #  A bright square present over a run of frames, so the pictures
            #  are not uniform grey and a human could actually see something.
            if 4 <= j <= 7:
                y0 = 10 + 3 * index
                plane[y0:y0 + 16, 30:46] += 90.0
            rgb = np.clip(plane, 0, 255).astype(np.uint8)[:, :, None]
            rgb = np.repeat(rgb, 3, axis=2)
            (frames_dir / f"src_{source_frame:06d}.png").write_bytes(
                proxy._encode_png(rgb, 0))
        (root / camera / proxy.PROXY_MANIFEST_NAME).write_text(json.dumps({
            "schema": "imvid-event-proxy-camera-v1",
            "camera": camera,
            "proxy_raster": [width, height],
            "n_output_frames": n_proxy_frames,
            "mapping": {"stride_frames": STRIDE,
                        "source_rate_exact": SOURCE_RATE,
                        "start_frame": 0},
        }, indent=2), encoding="utf-8")
    (root / "MANIFEST.imvid_event_proxy_scene.json").write_text(json.dumps({
        "schema": "imvid-event-proxy-scene-v1",
        "scene": root.name,
        "proxy_raster": [width, height],
        "cameras": {c: {} for c in cameras},
    }, indent=2), encoding="utf-8")


def _tile_explanation(camera: str, source_frame: int, *, raster=RASTER,
                      tile_size: int = 20, hot=(1, 2)) -> dict:
    """A census tile explanation in the real shape ``tile_candidate_explanation``
    emits, built from the real tiling helpers so the boxes are consistent."""
    width, height = raster
    rows = proxy.tile_edges(height, tile_size)
    cols = proxy.tile_edges(width, tile_size)
    grid = np.full((len(rows), len(cols)), 0.4)
    grid[hot[0], hot[1]] = 7.75
    grid[hot[0], min(hot[1] + 1, len(cols) - 1)] = 3.2
    order = np.argsort(grid, axis=None)[::-1][:4]
    top = []
    for flat in order:
        r, c = int(flat) // len(cols), int(flat) % len(cols)
        box = proxy.tile_pixel_box(r, c, rows, cols)
        top.append({
            "tile_row": r, "tile_col": c,
            "value": round(float(grid[r, c]), 4),
            "pixel_box_xyxy": box,
            "pixel_box_is_tile_extent_not_object_extent": True,
            "n_pixels": int((box[3] - box[1]) * (box[2] - box[0])),
        })
    return {
        "what_this_is": proxy.TILE_EXPLANATION_NOTE,
        "is_instance_mask": False,
        "kind": "detector_signal_explanation",
        "source_frame": int(source_frame),
        "proxy_index_in_window": int(source_frame // STRIDE),
        "tile_size_px": tile_size,
        "proxy_raster": [width, height],
        "grid_shape": [len(rows), len(cols)],
        "tile_template_dist_grid": [[round(float(v), 4) for v in row]
                                    for row in grid],
        "tile_max": 7.75,
        "tile_argmax_row_col": [hot[0], hot[1]],
        "tile_argmax_pixel_box_xyxy": proxy.tile_pixel_box(
            hot[0], hot[1], rows, cols),
        "top_tiles": top,
        "global_template_dist_at_candidate": 0.6123,
        "global_template_dist_before_candidate": 0.1044,
        "tile_max_over_global_template_dist": 12.658,
        "global_absolute_floor_grey_levels": 2.0,
        "tile_absolute_floor_grey_levels": 2.0,
        "global_pass_would_clear_its_own_floor": False,
    }


def _census(cameras, candidates, *, tile_mode: bool = True,
            raster=RASTER, window=(0, 330)) -> dict:
    """A census manifest in the real ``imvid-event-proxy-census-v1`` shape.

    ``candidates`` is a list of ``(polarity, {camera: source_frame})``.
    """
    rate = Fraction(SOURCE_RATE)
    step_ms = proxy.frames_to_ms(STRIDE, rate)
    clusters, explanations = [], {c: [] for c in cameras}
    for polarity, members in candidates:
        frames = sorted(members.values())
        spread = frames[-1] - frames[0]
        cluster = {
            "polarity": polarity,
            "n_cameras_supporting": len(members),
            "cameras": sorted(members),
            "per_camera_source_frame": {c: int(f)
                                        for c, f in sorted(members.items())},
            "source_frame_median": int(np.median(frames)),
            "source_frame_min": int(frames[0]),
            "source_frame_max": int(frames[-1]),
            "spread_frames": int(spread),
            "spread_ms": proxy.frames_to_ms(spread, rate),
            "spread_std_frames": float(np.std(frames)),
            "mean_amplitude": 6.5,
            "max_amplitude": 7.75,
            "localization_frames": STRIDE,
            "localization_ms": step_ms,
        }
        if tile_mode:
            cluster.update({
                "signal_used_for_detection": "tile_max",
                "per_camera_tile_argmax_row_col": {c: [1, 2] for c in members},
                "per_camera_global_template_dist": {c: 0.6123 for c in members},
                "tile_explanation_note": proxy.TILE_EXPLANATION_NOTE,
            })
            for camera, frame in members.items():
                explanations[camera].append(
                    _tile_explanation(camera, frame, raster=raster))
        clusters.append(cluster)
    clusters.sort(key=lambda c: (-c["n_cameras_supporting"],
                                 -c["mean_amplitude"],
                                 c["source_frame_median"]))
    window_out = {
        "window_source_frames": [int(window[0]), int(window[1])],
        "n_cameras": len(cameras),
        "n_candidates_total": sum(c["n_cameras_supporting"] for c in clusters),
        "n_clusters": len(clusters),
        "n_clusters_multi_camera": sum(1 for c in clusters
                                       if c["n_cameras_supporting"] >= 2),
        "candidate_clusters": clusters,
        "per_camera_signals": None,
    }
    if tile_mode:
        window_out["tile_explanations"] = explanations
        window_out["tile_explanations_note"] = proxy.TILE_EXPLANATION_NOTE
    report = {
        "schema": "imvid-event-proxy-census-v1",
        "instrument_status": "SCOUTING INSTRUMENT, NOT GROUND TRUTH",
        "disclaimer": proxy.DISCLAIMER,
        "proxy_root": "fixture",
        "n_cameras": len(cameras),
        "cameras": sorted(cameras),
        "mapping": {
            "source_rate_exact": SOURCE_RATE,
            "source_rate_float": float(rate),
            "stride_frames": STRIDE,
            "frame_period_ms": proxy.frames_to_ms(1, rate),
            "proxy_step_ms": step_ms,
            "proxy_raster": list(raster),
        },
        "parameters": {
            "window_frames": 300,
            "k_mad": 3.0,
            "min_amplitude_grey_levels": 2.0,
            "match_tol_frames": STRIDE,
            "match_tol_ms": step_ms,
            "min_cameras_for_multi": 2,
            "signal": "template_dist = mean |I_t - per-pixel temporal median|",
            "luma_weights": list(proxy.LUMA_WEIGHTS),
        },
        "temporal_resolution_note": "fixture note",
        "sync_uncertainty_resolvable_at_this_proxy_rate": False,
        "imvid_stated_sync_uncertainty_ms": [10.0, 20.0],
        "source_frame_range": [0, 330],
        "windows": [window_out],
    }
    if tile_mode:
        report["tile_pass"] = {
            "enabled": True,
            "note": proxy.TILE_EXPLANATION_NOTE,
            "signal": "tile_template_dist",
            "why": "whole-frame means are blind to small objects",
            "tile_size_px": 20,
            "tile_min_amplitude_grey_levels": 2.0,
            "tile_grid": [6, 4],
            "tile_grid_order": "[n_tile_x, n_tile_y]",
            "global_min_amplitude_grey_levels": 2.0,
            "top_tiles_per_candidate": 8,
            "k_mad": 3.0,
        }
    return report


# ---------------------------------------------------------------------------
# the PER-TILE (v2) census fixture
# ---------------------------------------------------------------------------

PER_TILE_SIZE = 20
#: A face-adjacent component of 3 tiles, plus one isolated tile firing at the
#: same sample that the coherence gate did NOT join to it.
PER_TILE_COMPONENT = ((1, 2), (1, 3), (2, 2))
PER_TILE_EXTRA_FIRING = (3, 0)


def _per_tile_grid(raster=RASTER, tile_size: int = PER_TILE_SIZE):
    width, height = raster
    return (proxy.tile_edges(height, tile_size),
            proxy.tile_edges(width, tile_size))


def _per_tile_candidate(camera: str, source_frame: int, polarity: str,
                        amplitude: float, *, raster=RASTER,
                        tile_size: int = PER_TILE_SIZE,
                        component=PER_TILE_COMPONENT) -> dict:
    """One per-camera per-tile candidate in the shape the census emits."""
    rows, cols = _per_tile_grid(raster, tile_size)
    boxes = [proxy.tile_pixel_box(r, c, rows, cols) for r, c in component]
    n = len(component)
    #  amplitudes centred on ``amplitude`` so the component mean IS it
    amps = [round(amplitude + (i - (n - 1) / 2.0) * 0.5, 4) for i in range(n)]
    firing = sorted(set(component) | {PER_TILE_EXTRA_FIRING})
    return {
        "polarity": polarity,
        "polarity_is_tied": False,
        "n_tiles_rise": n if polarity == "rise" else 0,
        "n_tiles_fall": 0 if polarity == "rise" else n,
        "source_frame": int(source_frame),
        "bracket_source_frames": [int(source_frame) - STRIDE,
                                  int(source_frame)],
        "localization_frames": STRIDE,
        "proxy_index_in_window": int(source_frame // STRIDE),
        "amplitude": float(amplitude),
        "max_tile_amplitude": float(max(amps)),
        "excess_over_median": round(float(amplitude) - 0.4, 4),
        "component_size_tiles": n,
        "n_tiles_firing_at_sample": len(firing),
        "n_tiles_in_grid": len(rows) * len(cols),
        "component_tiles_row_col": [[int(r), int(c)] for r, c in component],
        "firing_tiles_row_col": [[int(r), int(c)] for r, c in firing],
        "per_tile_amplitude": amps,
        "per_tile_excess_over_median": [round(v - 0.4, 4) for v in amps],
        "per_tile_polarity": [polarity] * n,
        "per_tile_signal_before": [0.4] * n,
        "per_tile_signal_after": amps,
        "per_tile_threshold": [1.2] * n,
        "per_tile_median": [0.4] * n,
        "per_tile_mad_scaled": [0.26] * n,
        "tile_pixel_boxes_xyxy": boxes,
        "component_pixel_bbox_xyxy": [
            min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)],
        "pixel_boxes_are_tile_extent_not_object_extent": True,
        "is_instance_mask": False,
        "kind": "detector_signal_explanation",
        "what_this_is": proxy.TILE_EXPLANATION_NOTE,
    }


def _per_tile_census(cameras, candidates, *, raster=RASTER,
                     window=(0, 330), floor_f: float = 3.0,
                     noise_median: float = 0.4108,
                     degenerate: bool = False,
                     tile_size: int = PER_TILE_SIZE) -> dict:
    """A census manifest in the real ``--per-tile-mode`` shape.

    ``candidates`` is a list of ``(polarity, {camera: source_frame},
    mean_amplitude)``.
    """
    rate = Fraction(SOURCE_RATE)
    step_ms = proxy.frames_to_ms(STRIDE, rate)
    rows, cols = _per_tile_grid(raster, tile_size)
    per_camera_candidates = {c: [] for c in cameras}
    clusters = []
    for polarity, members, amplitude in candidates:
        frames = sorted(members.values())
        spread = frames[-1] - frames[0]
        events = {}
        for camera, frame in sorted(members.items()):
            event = _per_tile_candidate(camera, frame, polarity, amplitude,
                                        raster=raster, tile_size=tile_size)
            events[camera] = event
            per_camera_candidates[camera].append(event)
        clusters.append({
            "signal_used_for_detection": "per_tile_template_dist",
            "per_camera_component_size_tiles": {
                c: e["component_size_tiles"] for c, e in events.items()},
            "per_camera_n_tiles_firing_at_sample": {
                c: e["n_tiles_firing_at_sample"] for c, e in events.items()},
            "per_camera_component_tiles_row_col": {
                c: e["component_tiles_row_col"] for c, e in events.items()},
            "per_camera_firing_tiles_row_col": {
                c: e["firing_tiles_row_col"] for c, e in events.items()},
            "per_camera_per_tile_amplitude": {
                c: e["per_tile_amplitude"] for c, e in events.items()},
            "per_camera_tile_pixel_boxes_xyxy": {
                c: e["tile_pixel_boxes_xyxy"] for c, e in events.items()},
            "per_camera_component_pixel_bbox_xyxy": {
                c: e["component_pixel_bbox_xyxy"] for c, e in events.items()},
            "pixel_boxes_are_tile_extent_not_object_extent": True,
            "is_instance_mask": False,
            "tile_explanation_note": proxy.TILE_EXPLANATION_NOTE,
            "polarity": polarity,
            "n_cameras_supporting": len(members),
            "cameras": sorted(members),
            "per_camera_source_frame": {c: int(f)
                                        for c, f in sorted(members.items())},
            "source_frame_median": int(np.median(frames)),
            "source_frame_min": int(frames[0]),
            "source_frame_max": int(frames[-1]),
            "spread_frames": int(spread),
            "spread_ms": proxy.frames_to_ms(spread, rate),
            "spread_std_frames": float(np.std(frames)),
            "mean_amplitude": float(amplitude),
            "max_amplitude": float(amplitude) + 0.5,
            "localization_frames": STRIDE,
            "localization_ms": step_ms,
        })
    #  Deliberately NOT sorted the way the gallery must sort: the census
    #  fixture emits census order, and the gallery is what must re-rank.
    clusters.sort(key=lambda c: (-c["mean_amplitude"],
                                 c["source_frame_median"]))
    window_out = {
        "window_source_frames": [int(window[0]), int(window[1])],
        "n_cameras": len(cameras),
        "n_candidates_total": sum(c["n_cameras_supporting"] for c in clusters),
        "n_clusters": len(clusters),
        "n_clusters_multi_camera": sum(1 for c in clusters
                                       if c["n_cameras_supporting"] >= 3),
        "candidate_clusters": clusters,
        "per_camera_signals": None,
        "per_tile_candidates": per_camera_candidates,
        "per_tile_rejected_by_spatial_coherence": {c: [] for c in cameras},
        "per_tile_n_tiles_with_events": {c: 4 for c in cameras},
        "tile_explanations_note": proxy.TILE_EXPLANATION_NOTE,
    }
    median = 0.0 if degenerate else float(noise_median)
    floor = {
        "F": float(floor_f),
        "F_is_a_declared_judgment_not_derived_from_data": True,
        "rule": ("floor = F * median over all (camera, window, tile) of "
                 "1.4826 * MAD_t(S_ij)"),
        "noise_scale_median_grey_levels": median,
        "floor_grey_levels": float(floor_f) * median,
        "n_tile_series": len(rows) * len(cols) * len(cameras),
        "n_camera_window_grids": len(cameras),
        "noise_scale_min_grey_levels": 0.0 if degenerate else 0.21,
        "noise_scale_max_grey_levels": 0.0 if degenerate else 3.44,
        "n_zero_scale_tile_series": (len(rows) * len(cols) * len(cameras)
                                     if degenerate else 2),
        "floor_is_degenerate_zero": bool(degenerate),
        "is_primary_reading": float(floor_f) == 3.0,
    }
    sweep = []
    for f_value in proxy.PER_TILE_FLOOR_F_SWEEP:
        is_primary = float(f_value) == 3.0
        sweep.append({
            "F": float(f_value),
            "is_primary_reading": is_primary,
            "label": ("PRIMARY READING -- this row IS the census" if is_primary
                      else "SENSITIVITY PROBE -- NOT the census, may not be "
                           "reported as one"),
            "floor_grey_levels": float(f_value) * median,
            "n_per_camera_candidates": int(40 / f_value),
            "n_rejected_by_spatial_coherence": int(90 / f_value),
            "n_clusters": int(20 / f_value),
            "n_clusters_at_min_cameras": int(6 / f_value),
            "min_cameras_for_multi": 3,
        })
    report = {
        "schema": "imvid-event-proxy-census-v1",
        "instrument_status": "SCOUTING INSTRUMENT, NOT GROUND TRUTH",
        "disclaimer": proxy.DISCLAIMER,
        "proxy_root": "fixture",
        "n_cameras": len(cameras),
        "cameras": sorted(cameras),
        "mapping": {
            "source_rate_exact": SOURCE_RATE,
            "source_rate_float": float(rate),
            "stride_frames": STRIDE,
            "frame_period_ms": proxy.frames_to_ms(1, rate),
            "proxy_step_ms": step_ms,
            "proxy_raster": list(raster),
        },
        "parameters": {
            "window_frames": 300,
            "k_mad": 3.0,
            "min_amplitude_grey_levels": 2.0,
            "match_tol_frames": STRIDE,
            "match_tol_ms": step_ms,
            "min_cameras_for_multi": 3,
            "signal": "template_dist = mean |I_t - per-pixel temporal median|",
            "luma_weights": list(proxy.LUMA_WEIGHTS),
        },
        "temporal_resolution_note": "fixture note",
        "sync_uncertainty_resolvable_at_this_proxy_rate": False,
        "imvid_stated_sync_uncertainty_ms": [10.0, 20.0],
        "source_frame_range": [0, 330],
        "windows": [window_out],
        "per_tile_pass": {
            "enabled": True,
            "note": proxy.TILE_EXPLANATION_NOTE,
            "is_instance_mask": False,
            "spec": ("research-wiki/operations/"
                     "imvid-tile-scout-v2-per-tile-2026-08-25.md"),
            "supersedes": "the max-over-tiles (tile_max) DETECTION REDUCTION",
            "signal": "tile_template_dist",
            "why": "tile_max is monopolised by the loudest region",
            "tile_size_px": tile_size,
            "tile_grid": [len(cols), len(rows)],
            "tile_grid_order": "[n_tile_x, n_tile_y]",
            "k_mad": 3.0,
            "relative_gate": "per TILE: median_t + k_mad * 1.4826 * MAD_t",
            "absolute_floor": floor,
            "spatial_coherence": {
                "min_component_tiles": proxy.DEFAULT_PER_TILE_MIN_COMPONENT,
                "connectivity": "4 (face-adjacent)",
                "max_firing_tiles_exclusive":
                    proxy.DEFAULT_PER_TILE_MAX_FIRING_TILES,
                "both_bounds_are_declared_judgments": True,
                "bounds_derived_from_a_known_positive_and_known_negative": True,
                "disclosure": "declared bounds, disclosed dependence",
            },
            "ranking": {
                "order": ("mean_amplitude DESCENDING, then "
                          "source_frame_median ascending"),
                "camera_count_is_retired_as_the_primary_order": True,
                "c_min_is_a_corroboration_floor_never_a_sort_key": True,
                "min_cameras_for_multi": 3,
            },
            "primary_reading_F": 3.0,
            "is_primary_reading": float(floor_f) == 3.0,
            "sensitivity_sweep": sweep,
            "sensitivity_sweep_is_supplementary": (
                "THE PRIMARY READING IS F = 3.0 ONLY. Every other point in "
                "this sweep is a sensitivity probe and may not be reported as "
                "the census."),
            "windowing_defect_not_repaired_here": (
                "Windows are non-overlapping and each is templated on its OWN "
                "temporal median. NO CLAIM OF EXHAUSTIVE RECALL MAY BE MADE "
                "WHILE IT STANDS."),
        },
    }
    return report


class _PerTileFixture:
    """tmpdir with a proxy tree and a PER-TILE census file."""

    def __init__(self, cameras, candidates, **census_kwargs):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.proxy_root = self.root / "proxy" / "scene_pertile"
        self.proxy_root.mkdir(parents=True)
        _write_proxy_tree(self.proxy_root, cameras)
        self.census = _per_tile_census(cameras, candidates, **census_kwargs)
        self.census_path = self.root / "census.json"
        self._write()
        self.out = self.root / "gallery"

    def _write(self):
        self.census_path.write_text(
            json.dumps(self.census, indent=2, sort_keys=True),
            encoding="utf-8")

    def mutate(self, fn):
        """Break the census in one specific way, then rewrite it."""
        fn(self.census)
        self._write()

    def build(self, **kwargs):
        defaults = {
            "census_path": self.census_path,
            "proxy_root": self.proxy_root,
            "out_dir": self.out,
            "scene": "scene_pertile",
            "montage_tile_width": 90,
            "focus_tile_width": 140,
            "overlay_width": 200,
        }
        defaults.update(kwargs)
        return gallery.build_gallery(**defaults)

    def html(self) -> str:
        return (self.out / "index.html").read_text(encoding="utf-8")

    def manifest(self) -> dict:
        return json.loads(
            (self.out / gallery.GALLERY_MANIFEST_NAME).read_text(
                encoding="utf-8"))

    def close(self):
        self._tmp.cleanup()


class _Fixture:
    """tmpdir with a proxy tree, a census file, and an output directory."""

    def __init__(self, cameras, candidates, **census_kwargs):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.proxy_root = self.root / "proxy" / "scene_fixture"
        self.proxy_root.mkdir(parents=True)
        _write_proxy_tree(self.proxy_root, cameras)
        self.census_path = self.root / "census.json"
        self.census_path.write_text(
            json.dumps(_census(cameras, candidates, **census_kwargs),
                       indent=2, sort_keys=True), encoding="utf-8")
        self.out = self.root / "gallery"

    def build(self, **kwargs):
        defaults = {
            "census_path": self.census_path,
            "proxy_root": self.proxy_root,
            "out_dir": self.out,
            "scene": "scene_fixture",
            "montage_tile_width": 90,
            "focus_tile_width": 140,
            "overlay_width": 200,
        }
        defaults.update(kwargs)
        return gallery.build_gallery(**defaults)

    def html(self) -> str:
        return (self.out / "index.html").read_text(encoding="utf-8")

    def manifest(self) -> dict:
        return json.loads(
            (self.out / gallery.GALLERY_MANIFEST_NAME).read_text(
                encoding="utf-8"))

    def close(self):
        self._tmp.cleanup()


# ---------------------------------------------------------------------------
# a minimal well-formedness checker (no external HTML library)
# ---------------------------------------------------------------------------

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"}


class _WellFormed(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.tags_seen: set[str] = set()
        self.text_parts: list[str] = []
        self.scripts: dict[str, str] = {}
        self.images: list[str] = []
        self.inputs: list[dict] = []
        self._script_id: str | None = None
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags_seen.add(tag)
        if tag == "img":
            self.images.append(attrs.get("src", ""))
        if tag in ("input", "textarea"):
            self.inputs.append({"tag": tag, **attrs})
        if tag in ("script", "style"):
            self._in_script = True
            self._script_id = attrs.get("id") if tag == "script" else None
            if tag == "script" and attrs.get("src"):
                self.errors.append("external script src")
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.errors.append("external stylesheet link")
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._in_script = False
            self._script_id = None
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append(f"</{tag}> with empty stack")
            return
        if self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes <{self.stack[-1]}>")
            return
        self.stack.pop()

    def handle_data(self, data):
        if self._in_script:
            if self._script_id:
                self.scripts[self._script_id] = (
                    self.scripts.get(self._script_id, "") + data)
            return
        self.text_parts.append(data)

    @property
    def visible_text(self) -> str:
        return "".join(self.text_parts)


def _parse(text: str) -> _WellFormed:
    parser = _WellFormed()
    parser.feed(text)
    parser.close()
    return parser


# ---------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------

class TimestampTests(unittest.TestCase):
    """The load-bearing arithmetic: seconds from the RATIONAL rate."""

    def test_exact_rational_seconds(self):
        rate = Fraction(SOURCE_RATE)
        self.assertEqual(gallery.frame_time_exact(0, rate), Fraction(0))
        self.assertEqual(gallery.frame_time_exact(300, rate),
                         Fraction(300 * 1001, 60000))
        self.assertEqual(gallery.frame_time_exact(300, rate), Fraction(1001, 200))
        self.assertEqual(float(gallery.frame_time_exact(300, rate)), 5.005)

    def test_neuter_an_assumed_60fps_would_fail_this(self):
        #  NEUTER. If the builder ever computed t = n / 60.0 this test fails.
        #  At 60000/1001 the two disagree from the very first frame and the
        #  gap grows without bound: one whole frame per 1001 frames.
        rate = Fraction(SOURCE_RATE)
        for frame in (1, 30, 300, 1001, 15214):
            exact = gallery.frame_time_exact(frame, rate)
            assumed_60 = Fraction(frame, 60)
            assumed_30 = Fraction(frame, 30)
            self.assertNotEqual(exact, assumed_60, f"frame {frame}")
            self.assertNotEqual(exact, assumed_30, f"frame {frame}")
            self.assertEqual(exact, Fraction(frame * 1001, 60000))
        #  The single most quotable one, spelled out.
        self.assertEqual(float(gallery.frame_time_exact(1001, rate)),
                         1001 * 1001 / 60000)
        self.assertNotAlmostEqual(
            float(gallery.frame_time_exact(1001, rate)), 1001 / 60.0, places=6)

    def test_refuses_a_non_positive_rate(self):
        with self.assertRaises(ContractError):
            gallery.frame_time_exact(10, Fraction(0))

    def test_phase_label_marks_the_bracket_as_gap(self):
        #  The bracket is [f - stride, f]. Both ends are GAP: the detector
        #  cannot say where inside them the change happened.
        self.assertEqual(gallery._phase(60, 120, 30), "pre")
        self.assertEqual(gallery._phase(89, 120, 30), "pre")
        self.assertEqual(gallery._phase(90, 120, 30), "gap")
        self.assertEqual(gallery._phase(120, 120, 30), "gap")
        self.assertEqual(gallery._phase(121, 120, 30), "post")
        self.assertEqual(gallery._phase(150, 120, 30), "post")


# ---------------------------------------------------------------------------
# the ordinary build
# ---------------------------------------------------------------------------

class GalleryBuildTests(unittest.TestCase):

    def setUp(self):
        self.fixture = _Fixture(
            ["cam00", "cam01", "cam02", "cam03"],
            [("rise", {"cam00": 120, "cam01": 120, "cam02": 150,
                       "cam03": 120}),
             ("fall", {"cam00": 240, "cam01": 240})])
        self.manifest = self.fixture.build()
        self.text = self.fixture.html()
        self.parsed = _parse(self.text)

    def tearDown(self):
        self.fixture.close()

    def test_html_is_produced_and_parses(self):
        self.assertTrue((self.fixture.out / "index.html").is_file())
        self.assertEqual(self.parsed.errors, [])
        self.assertEqual(self.parsed.stack, [])
        for tag in ("html", "head", "body", "section", "img", "details",
                    "textarea", "button"):
            self.assertIn(tag, self.parsed.tags_seen, tag)

    def test_bundle_is_self_contained(self):
        for needle in ("http://", "https://", "//cdn", "<script src=",
                       '<link rel="stylesheet"'):
            self.assertNotIn(needle, self.text, needle)
        for src in self.parsed.images:
            self.assertTrue(src.startswith("assets/"), src)
            self.assertTrue((self.fixture.out / src).is_file(), src)

    def test_every_image_referenced_exists_and_is_a_jpeg(self):
        self.assertTrue(self.parsed.images)
        for src in self.parsed.images:
            data = (self.fixture.out / src).read_bytes()
            self.assertEqual(data[:2], b"\xff\xd8", src)

    def test_manifest_records_provenance_and_settings(self):
        manifest = self.fixture.manifest()
        self.assertEqual(manifest["schema"], gallery.GALLERY_SCHEMA)
        self.assertEqual(len(manifest["census_sha256"]), 64)
        self.assertEqual(len(manifest["proxy_manifest_sha256"]), 64)
        self.assertEqual(manifest["scene"], "scene_fixture")
        self.assertEqual(manifest["n_candidates_in_census"], 2)
        self.assertEqual(manifest["n_candidates_rendered"], 2)
        self.assertEqual(manifest["source_rate_exact"], SOURCE_RATE)
        self.assertGreater(manifest["n_images"], 0)
        self.assertGreater(manifest["bytes_total"], manifest["bytes_images"])
        self.assertIn("max_candidates", manifest["settings"])
        self.assertIn("jpeg_quality", manifest["settings"])
        self.assertIn("dropped", manifest)

    def test_required_wording_is_present(self):
        text = self.parsed.visible_text
        self.assertIn(gallery.SPREAD_CAVEAT, text)
        self.assertIn("proxy-resolution localization spread — not "
                      "synchronization", text)
        self.assertIn("CANDIDATES, NOT EVENTS", text)
        self.assertIn("not a semantic", text)
        self.assertIn("NOT EVIDENCE OF ABSENCE", text.upper())

    def test_overlay_is_labelled_not_an_instance_mask(self):
        text = self.parsed.visible_text
        self.assertIn("NOT AN INSTANCE MASK", text.upper())
        self.assertIn("not an instance mask", text)
        self.assertIn("NOT PROOF OF IDENTITY", text.upper())
        self.assertIn("TILE EXTENTS, not object extents", text)
        #  the census's own note, carried verbatim
        self.assertIn("EXPLANATION OF THE DETECTOR SIGNAL, NOT AN INSTANCE "
                      "MASK", text)
        self.assertTrue(any(src.endswith("_overlay.jpg")
                            for src in self.parsed.images))

    def test_timestamps_on_the_page_come_from_the_rational_rate(self):
        #  cam02 supports the first candidate at source frame 150.
        rate = Fraction(SOURCE_RATE)
        exact = gallery.frame_time_exact(150, rate)
        self.assertEqual(exact, Fraction(150 * 1001, 60000))
        self.assertIn(f"{exact.numerator}/{exact.denominator}",
                      self.parsed.visible_text)
        self.assertIn(gallery.format_seconds(exact), self.parsed.visible_text)
        #  and the assumed-60 value must NOT be what the page reports for it
        self.assertNotEqual(gallery.format_seconds(exact),
                            "%.4f" % (150 / 60.0))

    def test_candidate_card_carries_support_amplitude_and_bracket(self):
        text = self.parsed.visible_text
        self.assertIn("multi-camera support", text)
        self.assertIn("amplitude / score", text)
        self.assertIn("candidate bracket (source frames)", text)
        self.assertIn("global vs tile signal", text)
        self.assertIn("distinct cameras", text)

    def test_controls_exist_for_every_candidate(self):
        radios = [i for i in self.parsed.inputs
                  if i.get("type") == "radio" and i.get("data-field") == "cls"]
        values = {i["value"] for i in radios}
        self.assertEqual(values, set(gallery.CLASS_CHOICES))
        candidates = {i["data-cand"] for i in radios}
        self.assertEqual(candidates, {"c0001", "c0002"})
        self.assertEqual(len(radios), 2 * len(gallery.CLASS_CHOICES))
        fields = {(i["data-cand"], i["data-field"]) for i in self.parsed.inputs
                  if i.get("data-field") in ("obj", "notes")}
        for cid in ("c0001", "c0002"):
            self.assertIn((cid, "obj"), fields)
            self.assertIn((cid, "notes"), fields)

    def test_both_export_paths_are_present(self):
        self.assertIn('id="download-decisions"', self.text)
        self.assertIn("Download decisions as JSON", self.text)
        self.assertIn('id="decisions-text"', self.text)
        self.assertIn("<textarea", self.text)
        self.assertIn("localStorage", self.text)

    def test_raw_json_provenance_is_embedded_in_details(self):
        self.assertIn("Raw JSON provenance", self.parsed.visible_text)
        self.assertIn("bracket_rule", self.parsed.visible_text)
        self.assertIn("time_rule", self.parsed.visible_text)
        self.assertIn("t = source_frame * denominator / numerator, exact",
                      self.parsed.visible_text)

    def test_decisions_schema_is_what_the_primary_ingests(self):
        payload = json.loads(self.parsed.scripts["decisions-template"])
        self.assertEqual(payload["schema"], gallery.DECISIONS_SCHEMA)
        for key in ("gallery_id", "scene", "census_path", "census_sha256",
                    "proxy_manifest_sha256", "source_rate_exact",
                    "exported_utc", "n_candidates_in_gallery",
                    "n_candidates_in_census", "n_decided", "class_choices",
                    "reading_rule", "decisions"):
            self.assertIn(key, payload, key)
        self.assertEqual(payload["class_choices"], list(gallery.CLASS_CHOICES))
        self.assertEqual(payload["n_candidates_in_gallery"], 2)
        self.assertEqual(payload["source_rate_exact"], SOURCE_RATE)
        self.assertEqual(len(payload["decisions"]), 2)
        row = payload["decisions"][0]
        for key in ("candidate_id", "window_source_frames", "polarity",
                    "source_frame_median", "bracket_source_frames",
                    "t_seconds", "t_seconds_exact", "n_cameras_supporting",
                    "cameras", "reference_camera", "mean_amplitude", "class",
                    "object_of_interest", "boundary_notes"):
            self.assertIn(key, row, key)
        self.assertIsNone(row["class"])
        self.assertEqual(row["object_of_interest"], "")
        self.assertEqual(row["boundary_notes"], "")
        self.assertEqual(row["candidate_id"], "c0001")
        self.assertEqual(row["n_cameras_supporting"], 4)
        #  the timestamp in the ingestible file is the rational one
        rate = Fraction(SOURCE_RATE)
        exact = gallery.frame_time_exact(row["source_frame_median"], rate)
        self.assertEqual(row["t_seconds_exact"],
                         f"{exact.numerator}/{exact.denominator}")
        self.assertEqual(row["t_seconds"], round(float(exact), 6))
        self.assertNotEqual(row["t_seconds"],
                            round(row["source_frame_median"] / 60.0, 6))

    def test_nothing_dropped_is_stated_positively(self):
        self.assertIn("NOTHING WAS DROPPED", self.parsed.visible_text)
        drops = self.manifest["dropped"]
        self.assertEqual(drops["candidates_dropped_by_max_candidates"], 0)
        self.assertEqual(drops["camera_rows_dropped_by_page_cap"], 0)
        self.assertEqual(drops["candidates_dropped_by_size_budget"], 0)


# ---------------------------------------------------------------------------
# caps, pagination, and the drop notices
# ---------------------------------------------------------------------------

class CapAndPaginationTests(unittest.TestCase):

    def test_pagination_splits_cameras_across_montage_pages(self):
        cameras = [f"cam{i:02d}" for i in range(7)]
        fixture = _Fixture(cameras,
                           [("rise", {c: 120 for c in cameras})])
        try:
            manifest = fixture.build(max_cameras_per_candidate=3)
            parsed = _parse(fixture.html())
            pages = sorted(src for src in parsed.images if "_montage_p" in src)
            #  7 cameras at 3 rows per page = 3 pages, and NOTHING dropped
            self.assertEqual(len(pages), 3, pages)
            self.assertIn("assets/c0001_montage_p1.jpg", pages)
            self.assertIn("assets/c0001_montage_p3.jpg", pages)
            self.assertEqual(
                manifest["dropped"]["camera_rows_dropped_by_page_cap"], 0)
            self.assertIn("NOTHING WAS DROPPED", parsed.visible_text)
            for page in (1, 2, 3):
                self.assertIn(f"montage page {page} of 3", parsed.visible_text)
        finally:
            fixture.close()

    def test_page_cap_drops_cameras_and_says_so_visibly(self):
        cameras = [f"cam{i:02d}" for i in range(7)]
        fixture = _Fixture(cameras,
                           [("rise", {c: 120 for c in cameras})])
        try:
            manifest = fixture.build(max_cameras_per_candidate=3,
                                     max_montage_pages=1)
            parsed = _parse(fixture.html())
            text = parsed.visible_text
            drops = manifest["dropped"]
            self.assertEqual(drops["camera_rows_dropped_by_page_cap"], 4)
            self.assertEqual(
                drops["candidates_with_dropped_cameras"][0]["cameras_shown"], 3)
            self.assertIn("SUPPORTING CAMERAS DROPPED FROM SOME MONTAGES", text)
            self.assertIn("4 camera rows were dropped", text)
            self.assertIn("c0001 (4 of 7)", text)
            self.assertNotIn("NOTHING WAS DROPPED", text)
        finally:
            fixture.close()

    def test_max_candidates_drops_and_says_so_visibly(self):
        cameras = ["cam00", "cam01", "cam02"]
        candidates = [("rise", {c: 60 for c in cameras}),
                      ("fall", {c: 120 for c in cameras}),
                      ("rise", {c: 180 for c in cameras}),
                      ("fall", {c: 240 for c in cameras})]
        fixture = _Fixture(cameras, candidates)
        try:
            manifest = fixture.build(max_candidates=2)
            parsed = _parse(fixture.html())
            text = parsed.visible_text
            self.assertEqual(manifest["n_candidates_in_census"], 4)
            self.assertEqual(manifest["n_candidates_rendered"], 2)
            self.assertEqual(
                manifest["dropped"]["candidates_dropped_by_max_candidates"], 2)
            self.assertIn("CANDIDATES DROPPED", text)
            self.assertIn("THIS GALLERY IS NOT THE WHOLE CENSUS", text)
            self.assertIn("The census holds 4 candidate clusters", text)
            self.assertIn("2 were dropped by --max-candidates", text)
            self.assertIn("were NOT reviewed and NOT rejected", text)
            payload = json.loads(parsed.scripts["decisions-template"])
            self.assertEqual(payload["n_candidates_in_gallery"], 2)
            self.assertEqual(payload["n_candidates_in_census"], 4)
        finally:
            fixture.close()

    def test_size_budget_drop_is_reported(self):
        cameras = ["cam00", "cam01"]
        candidates = [("rise", {c: 60 for c in cameras}),
                      ("fall", {c: 120 for c in cameras}),
                      ("rise", {c: 180 for c in cameras})]
        fixture = _Fixture(cameras, candidates)
        try:
            manifest = fixture.build(max_bytes=1)
            parsed = _parse(fixture.html())
            self.assertEqual(manifest["n_candidates_rendered"], 1)
            self.assertEqual(
                manifest["dropped"]["candidates_dropped_by_size_budget"], 2)
            self.assertIn("2 by the bundle size budget", parsed.visible_text)
        finally:
            fixture.close()


# ---------------------------------------------------------------------------
# zero candidates, and a census without the tile pass
# ---------------------------------------------------------------------------

class ZeroCandidateTests(unittest.TestCase):

    def test_zero_candidates_is_a_result_not_an_error(self):
        fixture = _Fixture(["cam00", "cam01"], [])
        try:
            manifest = fixture.build()
            parsed = _parse(fixture.html())
            text = parsed.visible_text
            self.assertEqual(parsed.errors, [])
            self.assertEqual(manifest["n_candidates_in_census"], 0)
            self.assertEqual(manifest["n_candidates_rendered"], 0)
            self.assertEqual(manifest["n_images"], 0)
            self.assertIn("THE CENSUS RETURNED ZERO CANDIDATES", text)
            self.assertIn("Zero is a RESULT", text)
            self.assertIn("NOT evidence that the scene contains no occlusion",
                          text)
            #  the settings and the declared detection scale are still shown
            self.assertIn("Census settings and declared detection scale", text)
            self.assertIn("declared detection scale", text)
            self.assertIn("tile absolute floor", text)
            self.assertIn("k_mad", text)
            self.assertIn("INVISIBLE to this census", text)
            #  still self-contained, still honest about what it is
            for needle in ("http://", "https://", "<script src="):
                self.assertNotIn(needle, fixture.html())
            self.assertIn("CANDIDATES, NOT EVENTS", text)
        finally:
            fixture.close()

    def test_census_without_tile_mode_still_builds_and_says_the_pass_was_off(self):
        cameras = ["cam00", "cam01"]
        fixture = _Fixture(cameras, [("rise", {c: 120 for c in cameras})],
                           tile_mode=False)
        try:
            manifest = fixture.build()
            parsed = _parse(fixture.html())
            text = parsed.visible_text
            self.assertEqual(parsed.errors, [])
            self.assertEqual(manifest["n_candidates_rendered"], 1)
            self.assertIn("OFF — detection ran on WHOLE-FRAME MEANS", text)
            self.assertIn("NO SPATIAL EXPLANATION FOR SOME CANDIDATES", text)
            self.assertIn("No tile explanation is present in the census", text)
            #  the wording still appears even with no overlay drawn
            self.assertIn("NOT AN INSTANCE MASK", text.upper())
            self.assertFalse(any(s.endswith("_overlay.jpg")
                                 for s in parsed.images))
        finally:
            fixture.close()


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

class RefusalTests(unittest.TestCase):

    def test_refuses_a_foreign_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not_a_census.json"
            path.write_text(json.dumps({"schema": "something-else"}),
                            encoding="utf-8")
            with self.assertRaises(ContractError):
                gallery.load_census(path)

    def test_refuses_a_proxy_root_without_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ContractError):
                gallery.load_proxy_tree(Path(tmp))

    def test_refuses_a_tile_grid_that_does_not_match_the_raster(self):
        fixture = _Fixture(["cam00"], [("rise", {"cam00": 120})])
        try:
            cameras = gallery.load_proxy_tree(fixture.proxy_root)
            explanation = _tile_explanation("cam00", 120)
            explanation["tile_size_px"] = 7  # inconsistent with the grid shape
            with self.assertRaises(ContractError):
                gallery.build_overlay(cameras["cam00"], "cam00", explanation,
                                      Fraction(SOURCE_RATE), width=100,
                                      header_lines=[("x", (255, 255, 255))])
        finally:
            fixture.close()


# ---------------------------------------------------------------------------
# schema detection: the defect that made this whole change necessary
# ---------------------------------------------------------------------------

#: amplitude order and camera-count order are EXACTLY REVERSED here, so any
#: test that cannot tell them apart is not testing anything.
_INVERTED = [
    ("rise", {"cam00": 120, "cam01": 120}, 9.0),                    # 2 cams
    ("fall", {"cam00": 210, "cam01": 210, "cam02": 210,
              "cam03": 210}, 3.0),                                  # 4 cams
    ("rise", {"cam00": 60, "cam01": 60, "cam02": 60}, 6.0),         # 3 cams
]
_INVERTED_CAMERAS = ["cam00", "cam01", "cam02", "cam03"]


class SchemaDetectionTests(unittest.TestCase):
    """A schema this gallery does not understand must FAIL, never degrade."""

    def test_detects_each_of_the_three_schemas(self):
        self.assertEqual(
            gallery.detect_census_mode(
                _per_tile_census(["cam00"], [("rise", {"cam00": 120}, 5.0)])),
            gallery.MODE_PER_TILE)
        self.assertEqual(
            gallery.detect_census_mode(
                _census(["cam00"], [("rise", {"cam00": 120})])),
            gallery.MODE_TILE)
        self.assertEqual(
            gallery.detect_census_mode(
                _census(["cam00"], [("rise", {"cam00": 120})],
                        tile_mode=False)),
            gallery.MODE_WHOLE_FRAME)

    def test_refuses_both_passes_enabled(self):
        census = _per_tile_census(["cam00"], [("rise", {"cam00": 120}, 5.0)])
        census["tile_pass"] = {"enabled": True, "tile_size_px": 20}
        with self.assertRaises(ContractError) as raised:
            gallery.detect_census_mode(census)
        self.assertIn("BOTH", str(raised.exception))

    def test_refuses_a_per_tile_manifest_whose_clusters_say_tile_max(self):
        #  THE HISTORICAL SHAPE OF THE DEFECT, inverted: a manifest whose
        #  pass block and whose clusters disagree is reconciled by nobody.
        census = _per_tile_census(["cam00"], [("rise", {"cam00": 120}, 5.0)])
        for cluster in census["windows"][0]["candidate_clusters"]:
            cluster["signal_used_for_detection"] = "tile_max"
        with self.assertRaises(ContractError) as raised:
            gallery.detect_census_mode(census)
        message = str(raised.exception)
        self.assertIn("REFUSED", message)
        self.assertIn("FOUND", message)
        self.assertIn("EXPECTED", message)

    def test_refuses_an_unknown_detection_signal(self):
        census = _per_tile_census(["cam00"], [("rise", {"cam00": 120}, 5.0)])
        census["windows"][0]["candidate_clusters"][0][
            "signal_used_for_detection"] = "wavelet_max_v9"
        with self.assertRaises(ContractError) as raised:
            gallery.detect_census_mode(census)
        self.assertIn("wavelet_max_v9", str(raised.exception))

    def test_refuses_a_per_tile_window_without_per_tile_candidates(self):
        census = _per_tile_census(["cam00"], [("rise", {"cam00": 120}, 5.0)])
        del census["windows"][0]["per_tile_candidates"]
        with self.assertRaises(ContractError) as raised:
            gallery.detect_census_mode(census)
        self.assertIn("per_tile_candidates", str(raised.exception))

    def test_refuses_a_per_tile_cluster_without_its_component(self):
        for field in gallery.PER_TILE_CLUSTER_FIELDS:
            census = _per_tile_census(["cam00"],
                                      [("rise", {"cam00": 120}, 5.0)])
            del census["windows"][0]["candidate_clusters"][0][field]
            with self.assertRaises(ContractError) as raised:
                gallery.detect_census_mode(census)
            self.assertIn(field, str(raised.exception), field)

    def test_refuses_a_per_tile_cluster_without_the_sort_key(self):
        census = _per_tile_census(["cam00"], [("rise", {"cam00": 120}, 5.0)])
        del census["windows"][0]["candidate_clusters"][0]["mean_amplitude"]
        with self.assertRaises(ContractError) as raised:
            gallery.detect_census_mode(census)
        self.assertIn("mean_amplitude", str(raised.exception))
        self.assertIn("FROZEN SORT KEY", str(raised.exception))

    def test_refuses_a_per_tile_pass_missing_its_floor_provenance(self):
        for field in gallery.PER_TILE_FLOOR_FIELDS:
            census = _per_tile_census(["cam00"],
                                      [("rise", {"cam00": 120}, 5.0)])
            del census["per_tile_pass"]["absolute_floor"][field]
            with self.assertRaises(ContractError) as raised:
                gallery.detect_census_mode(census)
            self.assertIn(field, str(raised.exception), field)

    def test_refuses_a_tile_manifest_without_its_explanations(self):
        census = _census(["cam00"], [("rise", {"cam00": 120})])
        del census["windows"][0]["tile_explanations"]
        with self.assertRaises(ContractError) as raised:
            gallery.detect_census_mode(census)
        self.assertIn("tile_explanations", str(raised.exception))

    def test_refuses_a_disabled_pass_block(self):
        census = _census(["cam00"], [("rise", {"cam00": 120})])
        census["tile_pass"]["enabled"] = False
        with self.assertRaises(ContractError):
            gallery.detect_census_mode(census)

    def test_a_broken_per_tile_manifest_is_refused_by_the_whole_build(self):
        """END TO END: no index.html, not a degraded one."""
        fixture = _PerTileFixture(_INVERTED_CAMERAS, _INVERTED)
        try:
            def break_it(census):
                for window in census["windows"]:
                    for cluster in window["candidate_clusters"]:
                        del cluster["per_camera_component_tiles_row_col"]
            fixture.mutate(break_it)
            with self.assertRaises(ContractError):
                fixture.build()
            self.assertFalse((fixture.out / "index.html").exists())
        finally:
            fixture.close()

    def test_a_per_tile_manifest_stripped_of_its_pass_block_is_refused(self):
        """The exact silent-degradation path, made loud.

        Strip ``per_tile_pass`` and the old code would have called this a
        whole-frame census, ranked it by camera support and drawn no overlay.
        """
        fixture = _PerTileFixture(_INVERTED_CAMERAS, _INVERTED)
        try:
            fixture.mutate(lambda census: census.pop("per_tile_pass"))
            with self.assertRaises(ContractError):
                fixture.build()
            self.assertFalse((fixture.out / "index.html").exists())
        finally:
            fixture.close()


# ---------------------------------------------------------------------------
# RANKING: mean amplitude, and camera count is not in the key at all
# ---------------------------------------------------------------------------

class PerTileRankingTests(unittest.TestCase):

    def setUp(self):
        self.fixture = _PerTileFixture(_INVERTED_CAMERAS, _INVERTED)
        self.manifest = self.fixture.build()
        self.parsed = _parse(self.fixture.html())
        self.payload = json.loads(self.parsed.scripts["decisions-template"])

    def tearDown(self):
        self.fixture.close()

    def test_emitted_order_is_amplitude_descending(self):
        amplitudes = [row["mean_amplitude"] for row in self.payload["decisions"]]
        self.assertEqual(amplitudes, [9.0, 6.0, 3.0])
        self.assertEqual(amplitudes, sorted(amplitudes, reverse=True))
        ids = [row["candidate_id"] for row in self.payload["decisions"]]
        self.assertEqual(ids, ["c0001", "c0002", "c0003"])

    def test_neuter_a_camera_count_sort_would_reverse_this(self):
        #  NEUTER. The fixture's amplitude order and camera-count order are
        #  EXACTLY REVERSED, so if the sort key ever silently reverts to
        #  n_cameras_supporting the emitted order becomes [3, 6, 9] and both
        #  assertions below fail.
        counts = [row["n_cameras_supporting"]
                  for row in self.payload["decisions"]]
        self.assertEqual(counts, [2, 3, 4])
        self.assertNotEqual(counts, sorted(counts, reverse=True))
        amplitudes = [row["mean_amplitude"] for row in self.payload["decisions"]]
        self.assertNotEqual(amplitudes, [3.0, 6.0, 9.0])

    def test_neuter_camera_count_is_absent_from_the_sort_key_itself(self):
        #  NEUTER, structural: two rows differing ONLY in
        #  n_cameras_supporting must produce IDENTICAL sort keys. Adding the
        #  count back to the key -- at any position, with any sign -- makes
        #  them differ and fails this.
        key = gallery.RANK_KEY_BY_MODE[gallery.MODE_PER_TILE]
        base = {"cluster": {"mean_amplitude": 5.0, "source_frame_median": 120,
                            "n_cameras_supporting": 2},
                "window_source_frames": [0, 330]}
        many = {"cluster": dict(base["cluster"], n_cameras_supporting=39),
                "window_source_frames": [0, 330]}
        self.assertEqual(key(base), key(many))
        self.assertEqual(key(base), (-5.0, 120, 0))
        self.assertEqual(gallery.RANK_KEY_BY_MODE[gallery.MODE_TILE](base),
                         (-2, -5.0, 120, 0))
        self.assertNotEqual(
            gallery.RANK_KEY_BY_MODE[gallery.MODE_TILE](base),
            gallery.RANK_KEY_BY_MODE[gallery.MODE_TILE](many))

    def test_the_sort_key_refuses_a_missing_amplitude_rather_than_zeroing_it(self):
        key = gallery.RANK_KEY_BY_MODE[gallery.MODE_PER_TILE]
        with self.assertRaises(KeyError):
            key({"cluster": {"source_frame_median": 1},
                 "window_source_frames": [0, 1]})

    def test_the_page_says_what_it_ranked_by_and_that_cameras_do_not(self):
        text = self.parsed.visible_text
        self.assertIn("RANKED BY MEAN AMPLITUDE", text)
        self.assertIn("mean_amplitude DESCENDING", text)
        self.assertIn("n_cameras_supporting IS NOT A SORT KEY", text)
        self.assertIn("CORROBORATION FLOOR ONLY, this count orders nothing",
                      text)
        #  the count is still SHOWN -- it is informative, it just orders nothing
        self.assertIn("distinct cameras", text)
        self.assertIn("cameras (NOT a sort key)", text)

    def test_manifest_records_the_mode_and_the_order(self):
        self.assertEqual(self.manifest["census_detection_mode"],
                         gallery.MODE_PER_TILE)
        self.assertFalse(self.manifest["ranking"]["camera_count_is_a_sort_key"])
        self.assertIn("mean_amplitude DESCENDING",
                      self.manifest["ranking"]["order"])
        self.assertEqual(self.payload["census_detection_mode"],
                         gallery.MODE_PER_TILE)


# ---------------------------------------------------------------------------
# the per-tile spatial overlay: the COMPONENT, not an argmax
# ---------------------------------------------------------------------------

class PerTileOverlayTests(unittest.TestCase):

    def setUp(self):
        self.fixture = _PerTileFixture(_INVERTED_CAMERAS, _INVERTED)
        self.manifest = self.fixture.build()
        self.parsed = _parse(self.fixture.html())

    def tearDown(self):
        self.fixture.close()

    def test_an_overlay_is_drawn_for_every_candidate(self):
        overlays = [s for s in self.parsed.images if s.endswith("_overlay.jpg")]
        self.assertEqual(len(overlays), 3, overlays)
        for src in overlays:
            self.assertTrue((self.fixture.out / src).is_file(), src)
            self.assertEqual(
                (self.fixture.out / src).read_bytes()[:2], b"\xff\xd8", src)

    def test_the_overlay_reports_the_component_not_an_argmax(self):
        text = self.parsed.visible_text
        self.assertIn("firing_connected_component", text)
        self.assertIn("component size (face-adjacent tiles)", text)
        self.assertIn("tiles firing anywhere at this sample", text)
        self.assertIn("component pixel bbox (proxy raster)", text)
        #  the v1 argmax row must NOT appear on a per-tile page
        self.assertNotIn("argmax tile (row, col)", text)

    def test_component_and_firing_counts_reach_the_decisions_file(self):
        payload = json.loads(self.parsed.scripts["decisions-template"])
        for row in payload["decisions"]:
            self.assertEqual(row["component_size_tiles"],
                             len(PER_TILE_COMPONENT))
            self.assertEqual(row["n_tiles_firing_at_sample"],
                             len(PER_TILE_COMPONENT) + 1)

    def test_the_not_an_instance_mask_wording_survives(self):
        text = self.parsed.visible_text
        self.assertIn("NOT AN INSTANCE MASK", text.upper())
        self.assertIn("NOT PROOF OF IDENTITY", text.upper())
        self.assertIn("TILE EXTENTS, not object extents", text)
        self.assertIn("EXPLANATION OF THE DETECTOR SIGNAL, NOT AN INSTANCE "
                      "MASK", text)
        self.assertIn("it bounds TILES, not an object extent", text)

    def test_overlay_refuses_a_box_that_does_not_match_the_tiling(self):
        cameras = gallery.load_proxy_tree(self.fixture.proxy_root)
        window = self.fixture.census["windows"][0]
        cluster = window["candidate_clusters"][0]
        explanation = gallery.per_tile_explanation_for(window, cluster, "cam00")
        explanation["tile_pixel_boxes_xyxy"][0] = [0, 0, 7, 7]
        with self.assertRaises(ContractError) as raised:
            gallery.build_per_tile_overlay(
                cameras["cam00"], "cam00", explanation, Fraction(SOURCE_RATE),
                width=120, tile_size=PER_TILE_SIZE, raster=RASTER,
                header_lines=[("x", (255, 255, 255))])
        self.assertIn("wrong pixels", str(raised.exception))

    def test_overlay_refuses_a_tile_count_that_contradicts_the_raster(self):
        cameras = gallery.load_proxy_tree(self.fixture.proxy_root)
        window = self.fixture.census["windows"][0]
        cluster = window["candidate_clusters"][0]
        explanation = gallery.per_tile_explanation_for(window, cluster, "cam00")
        explanation["n_tiles_in_grid"] = 144
        with self.assertRaises(ContractError):
            gallery.build_per_tile_overlay(
                cameras["cam00"], "cam00", explanation, Fraction(SOURCE_RATE),
                width=120, tile_size=PER_TILE_SIZE, raster=RASTER,
                header_lines=[("x", (255, 255, 255))])

    def test_the_explanation_matches_the_full_per_camera_record(self):
        window = self.fixture.census["windows"][0]
        cluster = window["candidate_clusters"][0]
        explanation = gallery.per_tile_explanation_for(window, cluster, "cam00")
        self.assertEqual(explanation["component_size_tiles"],
                         len(PER_TILE_COMPONENT))
        self.assertIn("detail", explanation)
        self.assertEqual(explanation["detail"]["source_frame"],
                         cluster["per_camera_source_frame"]["cam00"])
        self.assertEqual(explanation["n_tiles_in_grid"],
                         explanation["detail"]["n_tiles_in_grid"])
        self.assertIsNone(
            gallery.per_tile_explanation_for(window, cluster, "camZZ"))


# ---------------------------------------------------------------------------
# the floor provenance on the index page
# ---------------------------------------------------------------------------

class PerTileFloorProvenanceTests(unittest.TestCase):

    def test_the_measured_floor_and_the_sweep_are_on_the_page(self):
        fixture = _PerTileFixture(_INVERTED_CAMERAS, _INVERTED)
        try:
            manifest = fixture.build()
            parsed = _parse(fixture.html())
            text = parsed.visible_text
            floor = fixture.census["per_tile_pass"]["absolute_floor"]
            self.assertIn("Per-tile absolute floor — provenance", text)
            self.assertIn(str(floor["floor_grey_levels"]), text)
            self.assertIn(str(floor["noise_scale_median_grey_levels"]), text)
            self.assertIn("F is a DECLARED JUDGMENT", text)
            self.assertIn("PRIMARY READING", text)
            self.assertIn("Sensitivity sweep over F — SUPPLEMENTARY", text)
            self.assertIn("SENSITIVITY PROBE — NOT the census", text)
            self.assertIn("THE PRIMARY READING IS F = 3.0 ONLY", text)
            #  every sweep point is present and labelled
            for f_value in proxy.PER_TILE_FLOOR_F_SWEEP:
                self.assertIn(str(float(f_value)), text)
            self.assertIn("PRIMARY READING — this row IS the census", text)
            #  the live windowing defect is stated, not hidden
            self.assertIn("NO CLAIM OF EXHAUSTIVE RECALL MAY BE MADE", text)
            self.assertEqual(manifest["per_tile_floor_provenance"]["F"], 3.0)
            self.assertFalse(
                manifest["per_tile_floor_provenance"]["floor_is_degenerate_zero"])
        finally:
            fixture.close()

    def test_a_degenerate_zero_floor_is_a_loud_visible_warning(self):
        fixture = _PerTileFixture(_INVERTED_CAMERAS, _INVERTED,
                                  degenerate=True)
        try:
            manifest = fixture.build()
            parsed = _parse(fixture.html())
            text = parsed.visible_text
            self.assertIn("THE MEASURED PER-TILE NOISE SCALE MEDIAN IS ZERO",
                          text)
            self.assertIn("THE ABSOLUTE GATE IS INERT", text)
            self.assertIn("SCREENS NOTHING", text)
            self.assertTrue(
                manifest["per_tile_floor_provenance"]["floor_is_degenerate_zero"])
        finally:
            fixture.close()

    def test_a_probe_F_is_flagged_as_not_the_census(self):
        fixture = _PerTileFixture(_INVERTED_CAMERAS, _INVERTED, floor_f=6.0)
        try:
            fixture.build()
            text = _parse(fixture.html()).visible_text
            self.assertIn("THIS CENSUS WAS RUN AT A SENSITIVITY PROBE, NOT AT "
                          "THE PRIMARY READING", text)
            self.assertIn("forbids reporting as the census", text)
        finally:
            fixture.close()


# ---------------------------------------------------------------------------
# the FROZEN A / B / C definitions
# ---------------------------------------------------------------------------

class ClassLegendTests(unittest.TestCase):

    def setUp(self):
        self.fixture = _PerTileFixture(_INVERTED_CAMERAS, _INVERTED)
        self.manifest = self.fixture.build()
        self.parsed = _parse(self.fixture.html())

    def tearDown(self):
        self.fixture.close()

    def test_the_frozen_definitions_replace_the_reviewers_own_tiers(self):
        text = self.parsed.visible_text
        self.assertNotIn("A / B / C are the reviewer's own tiers", text)
        self.assertIn(gallery.CLASS_LEGEND_TITLE, text)
        self.assertIn("genuine scene ABSENCE", text)
        self.assertIn("LEFT THE REPRESENTED VOLUME", text)
        self.assertIn("POSITIVE multi-view evidence", text)
        self.assertIn("observed FREE for the duration of the gap", text)
        self.assertIn("FAILURE TO FIND AN OCCLUDER IS NOT EVIDENCE", text)
        self.assertIn("ordinary OCCLUSION", text)
        self.assertIn("still present but hidden", text)
        self.assertIn("RIG- or CAMERA-INDUCED", text)
        self.assertIn("left the frustum", text)
        self.assertIn("exposure / white balance changed", text)
        self.assertIn("applicable camera set itself changed", text)
        self.assertIn("not a real scene change", text)

    def test_the_one_camera_frame_note_and_the_citation_are_present(self):
        text = self.parsed.visible_text
        self.assertIn("A ball thrown out of ONE camera's frame is C for that "
                      "camera", text)
        self.assertIn("left the volume observed by the APPLICABLE CAMERA SET",
                      text)
        self.assertIn("imvid-event-definition-2026-08-24", text)
        self.assertIn("§3", text)

    def test_the_definitions_travel_in_the_decisions_file(self):
        payload = json.loads(self.parsed.scripts["decisions-template"])
        self.assertEqual(set(payload["class_definitions"]),
                         set(gallery.CLASS_CHOICES))
        self.assertIn("imvid-event-definition-2026-08-24",
                      payload["class_definitions_source"])
        self.assertIn("APPLICABLE CAMERA SET", payload["class_frame_note"])
        self.assertEqual(set(self.manifest["class_definitions"]),
                         set(gallery.CLASS_CHOICES))

    def test_the_legacy_tile_gallery_gets_the_frozen_definitions_too(self):
        fixture = _Fixture(["cam00", "cam01"],
                           [("rise", {"cam00": 120, "cam01": 120})])
        try:
            fixture.build()
            text = _parse(fixture.html()).visible_text
            self.assertNotIn("A / B / C are the reviewer's own tiers", text)
            self.assertIn("genuine scene ABSENCE", text)
            self.assertIn("FAILURE TO FIND AN OCCLUDER IS NOT EVIDENCE", text)
        finally:
            fixture.close()


# ---------------------------------------------------------------------------
# everything that already worked must still work in per-tile mode
# ---------------------------------------------------------------------------

class PerTilePreservationTests(unittest.TestCase):

    def setUp(self):
        self.fixture = _PerTileFixture(_INVERTED_CAMERAS, _INVERTED)
        self.manifest = self.fixture.build()
        self.text = self.fixture.html()
        self.parsed = _parse(self.text)

    def tearDown(self):
        self.fixture.close()

    def test_html_parses_and_is_self_contained(self):
        self.assertEqual(self.parsed.errors, [])
        self.assertEqual(self.parsed.stack, [])
        for needle in ("http://", "https://", "//cdn", "<script src=",
                       '<link rel="stylesheet"'):
            self.assertNotIn(needle, self.text, needle)
        for src in self.parsed.images:
            self.assertTrue(src.startswith("assets/"), src)
            self.assertTrue((self.fixture.out / src).is_file(), src)

    def test_timestamps_still_come_from_the_rational_rate(self):
        rate = Fraction(SOURCE_RATE)
        exact = gallery.frame_time_exact(210, rate)
        self.assertEqual(exact, Fraction(210 * 1001, 60000))
        self.assertIn(f"{exact.numerator}/{exact.denominator}",
                      self.parsed.visible_text)
        self.assertNotEqual(gallery.format_seconds(exact),
                            "%.4f" % (210 / 60.0))

    def test_neuter_an_assumed_60fps_would_change_the_decisions_file(self):
        payload = json.loads(self.parsed.scripts["decisions-template"])
        rate = Fraction(SOURCE_RATE)
        for row in payload["decisions"]:
            exact = gallery.frame_time_exact(row["source_frame_median"], rate)
            self.assertEqual(row["t_seconds_exact"],
                             f"{exact.numerator}/{exact.denominator}")
            self.assertEqual(row["t_seconds"], round(float(exact), 6))
            self.assertNotEqual(
                row["t_seconds"],
                round(row["source_frame_median"] / 60.0, 6))

    def test_both_export_paths_survive(self):
        self.assertIn('id="download-decisions"', self.text)
        self.assertIn("Download decisions as JSON", self.text)
        self.assertIn('id="decisions-text"', self.text)
        self.assertIn("<textarea", self.text)
        self.assertIn("localStorage", self.text)

    def test_controls_exist_for_every_per_tile_candidate(self):
        radios = [i for i in self.parsed.inputs
                  if i.get("type") == "radio" and i.get("data-field") == "cls"]
        self.assertEqual({i["data-cand"] for i in radios},
                         {"c0001", "c0002", "c0003"})
        self.assertEqual({i["value"] for i in radios},
                         set(gallery.CLASS_CHOICES))

    def test_caps_still_report_visibly_in_per_tile_mode(self):
        fixture = _PerTileFixture(_INVERTED_CAMERAS, _INVERTED)
        try:
            manifest = fixture.build(max_candidates=1)
            text = _parse(fixture.html()).visible_text
            self.assertEqual(manifest["n_candidates_in_census"], 3)
            self.assertEqual(manifest["n_candidates_rendered"], 1)
            self.assertIn("CANDIDATES DROPPED", text)
            self.assertIn("The census holds 3 candidate clusters", text)
            self.assertIn("2 were dropped by --max-candidates", text)
            #  and the one kept is the LOUDEST, not the most-corroborated
            payload = json.loads(
                _parse(fixture.html()).scripts["decisions-template"])
            self.assertEqual(payload["decisions"][0]["mean_amplitude"], 9.0)
            self.assertEqual(payload["decisions"][0]["n_cameras_supporting"], 2)
        finally:
            fixture.close()

    def test_zero_candidates_in_per_tile_mode_is_still_a_result(self):
        fixture = _PerTileFixture(["cam00", "cam01"], [])
        try:
            manifest = fixture.build()
            parsed = _parse(fixture.html())
            text = parsed.visible_text
            self.assertEqual(parsed.errors, [])
            self.assertEqual(manifest["n_candidates_in_census"], 0)
            self.assertIn("THE CENSUS RETURNED ZERO CANDIDATES", text)
            self.assertIn("Zero is a RESULT", text)
            #  the floor provenance is shown even with nothing to rank
            self.assertIn("Per-tile absolute floor — provenance", text)
            self.assertIn("PER-TILE (v2)", text)
        finally:
            fixture.close()


if __name__ == "__main__":
    unittest.main()
