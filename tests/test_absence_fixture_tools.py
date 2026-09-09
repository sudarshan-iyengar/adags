"""Tests for the counterfactual-absence fixture builder and its verifier.

Two tiers, deliberately separated:

* PURE HELPERS -- the reading rules (frame parsing, ROI construction, bbox
  union, the hardlink-vs-copy decision, difference confinement, the transforms
  time rule) exercised on hand-built arrays where the right answer is known by
  construction, with no scene on disk;
* END-TO-END -- a 3-camera 6-frame synthetic scene is built in ``tmp_path``,
  the builder runs, the verifier passes, and then each of three DISTINCT
  corruptions is injected and the verifier is required to fail on it. A checker
  that has never been shown to fail is not evidence that anything passed.

The synthetic scene uses a 64x48 raster rather than the real 1352x1014, so the
builder is invoked with ``--expect_raster 64 48``. Everything else -- the
naming, the window/margin arithmetic, the mask semantics, the manifest -- is
the production path.
"""

import json
import os
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_absence_fixture_scene as build  # noqa: E402
from scripts import verify_absence_fixture as verify  # noqa: E402


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


class FrameParsingTests(unittest.TestCase):

    def test_parses_camera_and_frame_from_every_spelling(self):
        for name in ("cam03_0147.png", "cam03_0147", "images/cam03_0147",
                     "images\\cam03_0147.png"):
            self.assertEqual(build.parse_image_stem(name), ("cam03", 147), name)

    def test_camera_digits_are_preserved_verbatim(self):
        # "cam00" is a directory key elsewhere in the fixture; stripping the
        # padding to 0 would silently break every lookup.
        self.assertEqual(build.parse_image_stem("cam00_0000.png"), ("cam00", 0))

    def test_rejects_names_that_are_not_cam_frame(self):
        for name in ("points3d.ply", "0147.png", "camXX_0147.png", "cam03.png"):
            self.assertIsNone(build.parse_image_stem(name), name)

    def test_frame_tokens_and_roi_names(self):
        self.assertEqual(build.frame_token(7), "0007")
        self.assertEqual(build.frame_token(7, 5), "00007")
        self.assertEqual(build.roi_filename(7), "00007.png")
        self.assertEqual(build.roi_filename(299), "00299.png")


class MorphologyTests(unittest.TestCase):

    def test_dilation_is_a_square_structuring_element(self):
        m = np.zeros((9, 9), dtype=bool)
        m[4, 4] = True
        grown = build.dilate(m, 2)
        self.assertEqual(int(grown.sum()), 25)
        self.assertTrue(grown[2:7, 2:7].all())

    def test_erosion_is_dilation_of_the_complement(self):
        m = np.zeros((20, 20), dtype=bool)
        m[5:15, 5:15] = True
        eroded = build.erode(m, 2)
        self.assertEqual(int(eroded.sum()), 6 * 6)
        self.assertTrue(eroded[7:13, 7:13].all())

    def test_zero_radius_is_the_identity_and_does_not_alias(self):
        m = np.zeros((5, 5), dtype=bool)
        m[2, 2] = True
        out = build.dilate(m, 0)
        out[0, 0] = True
        self.assertFalse(m[0, 0])

    def test_negative_radius_is_refused(self):
        with self.assertRaises(ValueError):
            build.dilate(np.zeros((3, 3), dtype=bool), -1)


class RoiConstructionTests(unittest.TestCase):

    def setUp(self):
        h = w = 60
        self.construction = np.zeros((h, w), dtype=bool)
        self.construction[20:40, 20:40] = True
        self.silhouette_ids = np.zeros((h, w), dtype=bool)
        self.silhouette_ids[20:40, 20:40] = True
        # the tracker leaking onto the stool, far from anything the editor
        # actually rewrote
        self.silhouette_ids[0:4, 0:4] = True
        self.support = np.zeros((h, w), dtype=bool)
        self.support[20:40, 20:30] = True
        self.rois = build.build_rois(self.silhouette_ids, self.construction, self.support)

    def test_deva_leak_outside_the_dilated_construction_mask_is_clipped(self):
        self.assertFalse(self.rois["object"][0:4, 0:4].any())
        self.assertTrue(self.rois["object"][20:40, 20:40].all())

    def test_core_is_the_supported_silhouette_eroded(self):
        core = self.rois["core"]
        self.assertTrue(core.any())
        # never outside the support
        self.assertFalse((core & ~self.support).any())
        # never outside the silhouette
        self.assertFalse((core & ~self.rois["object"]).any())
        # eroded by exactly CORE_ERODE_PX from the supported region's boundary
        expected = build.erode(self.rois["object"] & self.support, build.CORE_ERODE_PX)
        np.testing.assert_array_equal(core, expected)

    def test_ring_is_the_dilated_silhouette_minus_core_and_is_disjoint_from_it(self):
        ring, core = self.rois["ring"], self.rois["core"]
        self.assertFalse((ring & core).any())
        self.assertTrue(ring[19, 30])  # just outside the silhouette boundary
        grown = build.dilate(self.rois["object"], build.RING_DILATE_PX)
        np.testing.assert_array_equal(ring, grown & ~core)

    def test_mismatched_shapes_are_refused(self):
        with self.assertRaises(ValueError):
            build.build_rois(self.silhouette_ids, self.construction[:-1], self.support)

    def test_empty_silhouette_yields_empty_core_and_object(self):
        empty = np.zeros_like(self.silhouette_ids)
        rois = build.build_rois(empty, self.construction, self.support)
        self.assertFalse(rois["object"].any())
        self.assertFalse(rois["core"].any())


class BboxTests(unittest.TestCase):

    def test_bbox_is_half_open_on_the_far_edge(self):
        m = np.zeros((10, 10), dtype=bool)
        m[2:5, 3:7] = True
        self.assertEqual(build.mask_bbox(m), [3, 2, 7, 5])

    def test_empty_mask_has_no_bbox(self):
        self.assertIsNone(build.mask_bbox(np.zeros((4, 4), dtype=bool)))

    def test_union_skips_none_and_spans_the_rest(self):
        self.assertEqual(build.union_bbox([[3, 2, 7, 5], None, [1, 6, 4, 9]]),
                         [1, 2, 7, 9])
        self.assertIsNone(build.union_bbox([None, None]))

    def test_padding_clamps_to_the_raster(self):
        self.assertEqual(build.pad_bbox([3, 2, 7, 5], 8, 20, 20), [0, 0, 15, 13])
        self.assertEqual(build.pad_bbox([3, 2, 7, 5], 8, 10, 10), [0, 0, 10, 10])


class HardlinkDecisionTests(unittest.TestCase):

    def test_frame_outside_the_window_is_never_copied(self):
        self.assertFalse(build.should_copy_edited(9, (10, 20), __file__))

    def test_frame_inside_the_window_with_an_edited_file_is_copied(self):
        self.assertTrue(build.should_copy_edited(10, (10, 20), __file__))
        self.assertTrue(build.should_copy_edited(20, (10, 20), __file__))

    def test_frame_inside_the_window_without_an_edited_file_is_shared(self):
        self.assertFalse(build.should_copy_edited(15, (10, 20), "no-such-file.png"))


class DifferenceConfinementTests(unittest.TestCase):

    def setUp(self):
        self.a = np.zeros((8, 8, 3), dtype=np.uint8)
        self.mask = np.zeros((8, 8), dtype=bool)
        self.mask[2:6, 2:6] = True

    def test_change_inside_the_mask_is_confined(self):
        b = self.a.copy()
        b[3, 3] = 200
        confined, outside, inside = build.difference_confined(self.a, b, self.mask)
        self.assertTrue(confined)
        self.assertEqual(outside, 0)
        self.assertGreater(inside, 0.0)

    def test_a_single_leaked_pixel_is_caught(self):
        b = self.a.copy()
        b[3, 3] = 200
        b[0, 0, 1] = 1  # a one-count leak in one channel
        confined, outside, _ = build.difference_confined(self.a, b, self.mask)
        self.assertFalse(confined)
        self.assertEqual(outside, 1)

    def test_no_change_reports_zero_mean_inside(self):
        confined, outside, inside = build.difference_confined(self.a, self.a.copy(), self.mask)
        self.assertTrue(confined)
        self.assertEqual(outside, 0)
        self.assertEqual(inside, 0.0)

    def test_shape_mismatches_are_refused(self):
        with self.assertRaises(ValueError):
            build.difference_confined(self.a, self.a[:-1], self.mask)
        with self.assertRaises(ValueError):
            build.difference_confined(self.a, self.a.copy(), self.mask[:-1])


class TransformsTimeTests(unittest.TestCase):

    def test_correct_times_pass(self):
        frames = [{"file_path": "images/cam01_0030", "time": 1.0},
                  {"file_path": "images/cam01_0000", "time": 0.0}]
        self.assertTrue(all(ok for _p, _f, _t, ok in build.frames_have_expected_time(frames)))

    def test_a_wrong_time_is_caught(self):
        frames = [{"file_path": "images/cam01_0030", "time": 1.001}]
        self.assertFalse(build.frames_have_expected_time(frames)[0][3])

    def test_missing_or_unparseable_entries_fail_rather_than_raise(self):
        rows = build.frames_have_expected_time(
            [{"file_path": "images/junk"}, {"time": 1.0}])
        self.assertEqual([r[3] for r in rows], [False, False])


class HoleFractionSummaryTests(unittest.TestCase):

    def test_flat_camera_frame_keys(self):
        payload = {"cam00_0010": 0.1, "cam00_0011": 0.3, "cam01_0010": 0.5}
        summary = build.summarize_hole_fraction(payload)
        self.assertEqual(summary["cam00"]["min"], 0.1)
        self.assertEqual(summary["cam00"]["max"], 0.3)
        self.assertEqual(summary["cam00"]["n"], 2)
        self.assertEqual(summary["cam01"]["median"], 0.5)

    def test_camera_keyed_maps_and_lists(self):
        self.assertEqual(
            build.summarize_hole_fraction({"cam00": {"0010": 0.2, "0011": 0.4}})["cam00"]["median"],
            0.30000000000000004)
        self.assertEqual(
            build.summarize_hole_fraction({"cam00": [0.2, 0.4, 0.6]})["cam00"]["max"], 0.6)

    def test_record_lists(self):
        payload = [{"camera": "cam00", "hole_fraction": 0.25},
                   {"camera": "cam00", "hole_fraction": 0.75}]
        self.assertEqual(build.summarize_hole_fraction(payload)["cam00"]["n"], 2)

    def test_an_unrecognized_shape_is_marked_unparsed_not_summarized(self):
        self.assertTrue(build.summarize_hole_fraction("nonsense")["unparsed"])


class EventManifestTests(unittest.TestCase):

    def test_frozen_window_offsets(self):
        man = build.build_event_manifest(
            "X", [10, 20, 30, 40], (100, 160), (5, 25), (1352, 1014), "note")
        got = {e["name"]: e["frames"] for e in man["events"]}
        self.assertEqual(got["X_absence_gap"], [[103, 158]])
        self.assertEqual(got["X_return_early"], [[163, 170]])
        self.assertEqual(got["X_return_late"], [[171, 180]])
        self.assertEqual(got["X_pre"], [[70, 97]])
        self.assertEqual(got["X_control"], [[5, 25]])

    def test_schema_and_class_fields(self):
        man = build.build_event_manifest(
            "X", [10, 20, 30, 40], (100, 160), (5, 25), (1352, 1014), "note")
        self.assertEqual(man["schema_version"], "ccr-event-ray-masks-v1")
        self.assertEqual(man["camera"], "cam00")
        self.assertEqual(man["raster"], [1352, 1014])
        self.assertEqual(man["window_frames"], [0, 299])
        for event in man["events"]:
            self.assertEqual(event["class"], "counterfactual_absence")
            self.assertEqual(event["confidence"], "authored")
            self.assertEqual(event["bbox"], [10, 20, 30, 40])

    def test_an_empty_bbox_is_refused_rather_than_written(self):
        with self.assertRaises(ValueError):
            build.build_event_manifest("X", None, (100, 160), (5, 25), (1352, 1014), "n")


# ---------------------------------------------------------------------------
# end to end on a synthetic scene
# ---------------------------------------------------------------------------

W, H = 64, 48
CAMERAS = ("cam00", "cam01", "cam15")
N_FRAMES = 6
WINDOW = (1, 4)
MARGIN = 1
DILATE, FEATHER = 2, 1
DEVA_ID = 95
CONSTRUCTION_BOX = (10, 10, 40, 30)   # x0, y0, x1, y1 -> 30 x 20 = 600 px >= 500


def _png(path, arr, mode=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode=mode).save(path)


def _construction_mask():
    m = np.zeros((H, W), dtype=np.uint8)
    x0, y0, x1, y1 = CONSTRUCTION_BOX
    m[y0:y1, x0:x1] = 255
    return m


def _original_frame(cam, frame):
    """Deterministic non-uniform imagery, distinct per camera and frame."""
    yy, xx = np.mgrid[0:H, 0:W]
    seed = (int(cam[3:]) * 17 + frame * 31) % 200
    arr = np.stack([(xx * 3 + seed) % 256, (yy * 5 + seed) % 256,
                    (xx + yy + seed) % 256], axis=2)
    return arr.astype(np.uint8)


def make_synthetic_scene(root):
    """Write an ``<orig>``, an ``<edit>`` tranche and a DEVA cam00 run."""
    root = Path(root)
    orig, edit, deva = root / "orig", root / "edit", root / "deva"
    (orig / "images").mkdir(parents=True)

    construction = _construction_mask()
    con_bool = construction >= 128
    a, b = WINDOW

    train, test = [], []
    for cam in CAMERAS:
        for frame in range(N_FRAMES):
            name = "%s_%s.png" % (cam, build.frame_token(frame))
            _png(orig / "images" / name, _original_frame(cam, frame))
            entry = {"file_path": "images/%s_%s" % (cam, build.frame_token(frame)),
                     "time": frame / 30.0,
                     "transform_matrix": np.eye(4).tolist()}
            (test if cam == "cam00" else train).append(entry)

            # construction masks over the whole margin range (and beyond -- the
            # verifier only requires the margin range)
            _png(edit / "construction_masks" / name, construction, mode="L")
            # support: the left half of the construction box
            support = np.zeros((H, W), dtype=np.uint8)
            support[CONSTRUCTION_BOX[1]:CONSTRUCTION_BOX[3],
                    CONSTRUCTION_BOX[0]:CONSTRUCTION_BOX[0] + 20] = 255
            _png(edit / "support" / name, support, mode="L")
            # visible_object: all-zero inside the window, non-empty outside
            vo = np.zeros((H, W), dtype=np.uint8)
            if not (a <= frame <= b):
                vo[CONSTRUCTION_BOX[1]:CONSTRUCTION_BOX[3],
                   CONSTRUCTION_BOX[0]:CONSTRUCTION_BOX[2]] = 255
            _png(edit / "visible_object" / name, vo, mode="L")

            if a <= frame <= b:
                edited = _original_frame(cam, frame).copy()
                # change ONLY inside the construction mask
                edited[con_bool] = (edited[con_bool].astype(np.int16) + 61) % 256
                _png(edit / "images_edited" / name, edited.astype(np.uint8))

    for tname, frames in (("transforms_train.json", train),
                          ("transforms_test.json", test)):
        (orig / tname).write_text(json.dumps(
            {"camera_angle_x": 0.7, "w": W, "h": H, "frames": frames}), encoding="utf-8")
    (orig / "points3d.ply").write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")

    (edit / "edit_params.json").write_text(json.dumps({
        "ids": [DEVA_ID], "window": list(WINDOW), "margin": MARGIN,
        "dilate": DILATE, "feather": FEATHER, "soft_thresh": 0.5}), encoding="utf-8")
    (edit / "sa4d_provenance.json").write_text(json.dumps(
        {"tool": "sa4d", "checkpoint": "synthetic"}), encoding="utf-8")
    (edit / "hole_fraction.json").write_text(json.dumps(
        {"%s_%s" % (cam, build.frame_token(f)): 0.1 + 0.01 * f
         for cam in CAMERAS for f in range(N_FRAMES)}), encoding="utf-8")

    # DEVA cam00 id maps: the object id inside the construction box, plus a
    # DIFFERENT id leaking onto a far-away "stool" that must be clipped away.
    for frame in range(N_FRAMES):
        ids = np.zeros((H, W), dtype=np.uint8)
        ids[CONSTRUCTION_BOX[1]:CONSTRUCTION_BOX[3],
            CONSTRUCTION_BOX[0]:CONSTRUCTION_BOX[2]] = DEVA_ID
        # same id, in the far corner -- outside the 12-px guard dilation of the
        # construction box, so the builder must clip it away
        ids[H - 3:H, W - 3:W] = DEVA_ID
        _png(deva / "object_mask" / ("%s.png" % build.frame_token(frame)), ids, mode="L")

    return orig, edit, deva


def _hardlinks_work(tmp):
    probe = Path(tmp) / "_link_probe"
    probe.write_bytes(b"x")
    target = Path(tmp) / "_link_probe2"
    try:
        os.link(probe, target)
    except OSError:
        return False
    finally:
        probe.unlink(missing_ok=True)
        Path(target).unlink(missing_ok=True)
    return True


class EndToEndTests(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="absfix_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.orig, self.edit, self.deva = make_synthetic_scene(self.tmp)
        self.out = self.tmp / "derived" / "absfix_scene"
        self.copy_mode = not _hardlinks_work(self.tmp)
        self.assertEqual(0, self._build())
        self.manifest_path = self.out / "MANIFEST.absence_edit.json"

    def _build(self, out=None):
        argv = ["--orig", str(self.orig), "--edit", str(self.edit),
                "--deva_cam00", str(self.deva), "--deva_cam00_ids", str(DEVA_ID),
                "--out", str(out or self.out), "--control_window", "0", "1",
                "--expect_raster", str(W), str(H)]
        if self.copy_mode:
            argv.append("--copy")
        return build.main(argv)

    def _verify(self, montage=None):
        argv = ["--scene_dir", str(self.out), "--orig", str(self.orig),
                "--manifest", str(self.manifest_path), "--sample_fraction", "1.0"]
        if montage:
            argv += ["--montage", str(montage)]
        return verify.main(argv)

    # --- the build ---------------------------------------------------------

    def test_build_produces_the_expected_tree(self):
        self.assertTrue((self.out / "images").is_dir())
        self.assertTrue((self.out / "points3d.ply").is_file())
        self.assertTrue((self.out / "transforms_train.json").is_file())
        self.assertTrue((self.out / "transforms_test.json").is_file())
        self.assertTrue((self.out / "absfix" / "absence_event_masks.json").is_file())
        self.assertEqual(len(list((self.out / "images").glob("*.png"))),
                         len(CAMERAS) * N_FRAMES)

    def test_no_derived_priors_are_carried_over(self):
        for bad in build.FORBIDDEN_DIRS:
            self.assertFalse((self.out / bad).exists(), bad)

    def test_only_windowed_frames_are_copies(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        a, b = WINDOW
        self.assertEqual(manifest["counts"]["edited"], len(CAMERAS) * (b - a + 1))
        if not self.copy_mode:
            self.assertEqual(manifest["counts"]["hardlinked"],
                             len(CAMERAS) * (N_FRAMES - (b - a + 1)))
            outside = self.out / "images" / ("cam01_%s.png" % build.frame_token(0))
            self.assertEqual(os.stat(outside).st_ino,
                             os.stat(self.orig / "images" / outside.name).st_ino)

    def test_rois_cover_the_margin_range_and_clip_the_tracker_leak(self):
        roi = self.out / "absfix" / "evaluation_rois"
        a, b = WINDOW
        for frame in range(a - MARGIN, b + MARGIN + 1):
            for kind in ("core", "ring", "object"):
                self.assertTrue((roi / kind / build.roi_filename(frame)).is_file())
        obj = np.asarray(Image.open(roi / "object" / build.roi_filename(WINDOW[0])))
        self.assertEqual(int(obj[H - 3:H, W - 3:W].sum()), 0)  # the stool leak is gone
        self.assertGreater(int((obj > 0).sum()), 0)
        core = np.asarray(Image.open(roi / "core" / build.roi_filename(WINDOW[0])))
        self.assertGreater(int((core > 0).sum()), 0)

    def test_event_manifest_bbox_covers_the_silhouette(self):
        events = json.loads(
            (self.out / "absfix" / "absence_event_masks.json").read_text(encoding="utf-8"))
        self.assertEqual(events["schema_version"], "ccr-event-ray-masks-v1")
        x0, y0, x1, y1 = events["events"][0]["bbox"]
        cx0, cy0, cx1, cy1 = CONSTRUCTION_BOX
        self.assertLessEqual(x0, cx0)
        self.assertLessEqual(y0, cy0)
        self.assertGreaterEqual(x1, cx1)
        self.assertGreaterEqual(y1, cy1)

    def test_manifest_records_the_evidence_boundary(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertIs(manifest["teacher_rendered_counterfactual"], True)
        self.assertIs(manifest["evidence_bearing"], False)
        self.assertTrue(manifest["raw_tree_untouched"]["unchanged"])
        self.assertEqual(manifest["deva_cam00_ids"], [DEVA_ID])

    def test_margin_range_is_clamped_to_the_frames_that_exist_and_says_so(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        a, b = WINDOW
        self.assertEqual(manifest["margin_range"], [a - MARGIN, b + MARGIN])
        self.assertFalse(manifest["margin_range_clamped"])

        params = json.loads((self.edit / "edit_params.json").read_text(encoding="utf-8"))
        params["margin"] = 3          # requests [-2, 7]; only [0, 5] exist
        (self.edit / "edit_params.json").write_text(json.dumps(params), encoding="utf-8")
        out = self.tmp / "derived" / "clamped"
        self.assertEqual(0, self._build(out=out))
        clamped = json.loads((out / "MANIFEST.absence_edit.json").read_text(encoding="utf-8"))
        self.assertEqual(clamped["margin_range_requested"], [a - 3, b + 3])
        self.assertEqual(clamped["margin_range"], [0, N_FRAMES - 1])
        self.assertTrue(clamped["margin_range_clamped"])

    def test_refuses_to_overwrite_an_existing_output(self):
        with self.assertRaises(SystemExit):
            self._build()

    def test_refuses_a_vacuous_edit_tranche(self):
        shutil.rmtree(self.edit / "images_edited")
        (self.edit / "images_edited").mkdir()
        with self.assertRaises(SystemExit) as ctx:
            self._build(out=self.tmp / "derived" / "vacuous")
        self.assertIn("PRECONDITION FAILED", str(ctx.exception))

    def test_refuses_an_empty_silhouette(self):
        out = self.tmp / "derived" / "empty_ids"
        argv = ["--orig", str(self.orig), "--edit", str(self.edit),
                "--deva_cam00", str(self.deva), "--deva_cam00_ids", "7",
                "--out", str(out), "--control_window", "0", "1",
                "--expect_raster", str(W), str(H)]
        if self.copy_mode:
            argv.append("--copy")
        with self.assertRaises(SystemExit) as ctx:
            build.main(argv)
        self.assertIn("PRECONDITION FAILED", str(ctx.exception))

    def test_refuses_an_unexpected_raster(self):
        with self.assertRaises(SystemExit):
            build.main(["--orig", str(self.orig), "--edit", str(self.edit),
                        "--deva_cam00", str(self.deva), "--deva_cam00_ids", str(DEVA_ID),
                        "--out", str(self.tmp / "derived" / "wrongraster"),
                        "--control_window", "0", "1",
                        "--expect_raster", "1352", "1014"])

    # --- the verifier ------------------------------------------------------

    def test_verify_passes_on_a_clean_build(self):
        self.assertEqual(0, self._verify())

    def test_verify_writes_a_montage(self):
        path = self.tmp / "montage.jpg"
        self.assertEqual(0, self._verify(montage=path))
        self.assertTrue(path.is_file())
        with Image.open(path) as img:
            self.assertEqual(img.format, "JPEG")
            self.assertEqual(img.width, verify.MONTAGE_TILE_WIDTH * 4)

    def test_verify_fails_when_a_frame_outside_the_window_is_corrupted(self):
        target = self.out / "images" / ("cam01_%s.png" % build.frame_token(0))
        target.unlink()                     # break the hardlink, do not follow it
        _png(target, (_original_frame("cam01", 0) + 7).astype(np.uint8))
        self.assertEqual(1, self._verify())

    def test_verify_fails_when_an_edit_leaks_outside_its_construction_mask(self):
        target = self.out / "images" / ("cam01_%s.png" % build.frame_token(WINDOW[0]))
        arr = np.asarray(Image.open(target)).copy()
        arr[0, 0, 0] = (int(arr[0, 0, 0]) + 40) % 256   # outside the mask + slack
        target.unlink()
        _png(target, arr)
        self.assertEqual(1, self._verify())

    def test_verify_fails_when_a_windowed_frame_was_never_edited(self):
        target = self.out / "images" / ("cam01_%s.png" % build.frame_token(WINDOW[0]))
        target.unlink()
        shutil.copyfile(self.orig / "images" / target.name, target)
        self.assertEqual(1, self._verify())

    def test_verify_fails_when_visible_object_is_not_empty_in_the_window(self):
        name = "cam01_%s.png" % build.frame_token(WINDOW[0])
        arr = np.zeros((H, W), dtype=np.uint8)
        arr[5, 5] = 255
        _png(self.edit / "visible_object" / name, arr, mode="L")
        self.assertEqual(1, self._verify())

    def test_verify_fails_when_an_evaluation_roi_is_missing(self):
        (self.out / "absfix" / "evaluation_rois" / "core"
         / build.roi_filename(WINDOW[0])).unlink()
        self.assertEqual(1, self._verify())

    def test_verify_fails_when_a_forbidden_prior_directory_appears(self):
        (self.out / "motion_priors").mkdir()
        self.assertEqual(1, self._verify())

    def test_verify_fails_when_points3d_changes(self):
        (self.out / "points3d.ply").unlink()
        (self.out / "points3d.ply").write_bytes(b"ply\ndifferent\n")
        self.assertEqual(1, self._verify())

    def test_verify_fails_when_cam00_enters_training(self):
        path = self.out / "transforms_train.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["frames"].append({"file_path": "images/cam00_0000", "time": 0.0,
                                  "transform_matrix": np.eye(4).tolist()})
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(1, self._verify())

    def test_verify_fails_when_a_transforms_time_is_wrong(self):
        path = self.out / "transforms_train.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["frames"][0]["time"] = 99.0
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(1, self._verify())

    def test_verify_fails_when_an_image_carries_an_alpha_channel(self):
        name = "cam01_%s.png" % build.frame_token(0)
        target = self.out / "images" / name
        arr = np.asarray(Image.open(target).convert("RGB"))
        rgba = np.dstack([arr, np.full((H, W), 255, dtype=np.uint8)])
        target.unlink()
        _png(target, rgba, mode="RGBA")
        self.assertEqual(1, self._verify())


if __name__ == "__main__":
    unittest.main()
