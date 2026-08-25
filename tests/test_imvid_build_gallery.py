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
  reader toward re-running until something appeared.
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


if __name__ == "__main__":
    unittest.main()
