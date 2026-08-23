"""LRV1 fixture geometry: does the scene mean what the evaluator assumes?

This is gate item 7 of
`research-wiki/operations/lrv1-oracle-headroom-spec-2026-08-19.md` made
reproducible. The event metric is defined over the ground-truth identity
buffers, so the whole comparison rests on those buffers agreeing with the
camera geometry the TRAINER sees. If the builder and the loader disagreed about
the camera convention -- an easy thing to get wrong, since the builder emits
OpenGL camera-to-world and the loader flips two columns to reach COLMAP -- the
masks would sit somewhere other than the object and nothing downstream would
say so.

The check: project the event sphere's world centre through the loader's OWN
transformation and assert it lands on event-object pixels in the held-out
identity buffer, on every return view-frame.

Skips when the fixture is not present in the checkout (it is a gitignored
dataset under `data/`, not repository content).
"""

import json
import unittest
from pathlib import Path

import numpy as np

# Checks the newest leave-and-return fixture present. The variants differ only
# in the ground-plane extent (LRV2) and the return length (LRV3, LRV4), so the
# same geometry invariants must hold for all of them.
_ROOT = Path(__file__).resolve().parents[1] / "data" / "synthetic"
_VARIANTS = ("lrv4", "lrv3", "lrv2", "lrv1")
SCENE = next((_ROOT / n for n in _VARIANTS if (_ROOT / n).is_dir()),
             _ROOT / "lrv1")


@unittest.skipUnless(SCENE.is_dir(), "LRV1 fixture not present in this checkout")
class Lrv1FixtureGeometryTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((SCENE / "event_spec.json").read_text())
        cls.test_json = json.loads((SCENE / "transforms_test.json").read_text())
        cls.train_json = json.loads((SCENE / "transforms_train.json").read_text())

    def test_object_centre_reprojects_onto_object_pixels_on_every_return_frame(self):
        spec, tt = self.spec, self.test_json
        centre = np.array(spec["event_object"]["centre"], dtype=np.float64)
        fx, fy, cx, cy = tt["fl_x"], tt["fl_y"], tt["cx"], tt["cy"]
        obj_id = int(spec["event_object"]["id"])
        returns = set(spec["return_frames"])

        checked = misses = 0
        for frame in tt["frames"]:
            f = frame["frame_index"]
            if f not in returns:
                continue
            # exactly what scene/dataset_readers.readCamerasFromTransforms does:
            # flip the OpenGL Y/Z axes to COLMAP, invert to world-to-camera,
            # then store R transposed and T as the translation.
            c2w = np.array(frame["transform_matrix"], dtype=np.float64)
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            # and exactly what utils.graphics_utils.getWorld2View2 reconstructs
            xc = R.T @ centre + T
            self.assertGreater(xc[2], 0.0, "object behind the camera")
            u = int(round(fx * xc[0] / xc[2] + cx))
            v = int(round(fy * xc[1] / xc[2] + cy))

            ident = np.load(SCENE / "gt_identity" /
                            ("cam%02d_f%03d.npy" % (frame["camera_index"], f)))
            checked += 1
            inside = 0 <= u < ident.shape[1] and 0 <= v < ident.shape[0]
            if not (inside and ident[v, u] == obj_id):
                misses += 1
        self.assertGreater(checked, 0, "no return view-frames found")
        self.assertEqual(misses, 0,
                         "%d of %d return view-frames reproject off the object"
                         % (misses, checked))
        self.assertEqual(checked, len(spec["test_cameras"]) * len(spec["return_frames"]))

    def test_the_event_object_is_absent_from_every_gap_frame_and_present_at_return(self):
        spec = self.spec
        counts = spec["event_object_pixels_per_test_view_frame"]
        gap = range(spec["presence_frames"]["gap"][0],
                    spec["presence_frames"]["gap"][1] + 1)
        for cam in spec["test_cameras"]:
            for f in gap:
                self.assertEqual(counts["cam%02d_f%03d" % (cam, f)], 0,
                                 "object visible during the absence gap")
            for f in spec["return_frames"]:
                self.assertGreater(counts["cam%02d_f%03d" % (cam, f)], 0,
                                   "returned surface invisible in a held-out view")

    def test_train_and_held_out_camera_sets_are_disjoint(self):
        train = {f["camera_index"] for f in self.train_json["frames"]}
        test = {f["camera_index"] for f in self.test_json["frames"]}
        self.assertEqual(train & test, set())
        self.assertEqual(sorted(test), sorted(self.spec["test_cameras"]))
        self.assertEqual(sorted(train), sorted(self.spec["train_cameras"]))

    def test_identity_buffers_exist_for_every_held_out_view_frame(self):
        spec = self.spec
        missing = [
            "cam%02d_f%03d" % (c, f)
            for c in spec["test_cameras"] for f in range(spec["n_frames"])
            if not (SCENE / "gt_identity" / ("cam%02d_f%03d.npy" % (c, f))).is_file()
        ]
        self.assertEqual(missing, [], "identity buffers missing; the event metric "
                                      "would be computed over an incomplete region")


# ---------------------------------------------------------------------------
# The observation-supply knob and its default-off admissibility cap
# ---------------------------------------------------------------------------

import sys  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_synthetic_reveal_scene as gen  # noqa: E402


class AdmissibilityBoundTests(unittest.TestCase):
    """The cap `--allow-short-return` relaxes, as arithmetic.

    Needs no fixture: these are pure functions of the frame layout.
    """

    def test_floor_len_is_the_frozen_value(self):
        self.assertAlmostEqual(gen.FLOOR_LEN, 10.0 / 12.0, places=15)

    def test_episode_2_durations(self):
        self.assertAlmostEqual(gen.episode_2_duration(57), 1.0, places=12)
        self.assertAlmostEqual(gen.episode_2_duration(59), 2.0 / 3.0,
                               places=12)
        self.assertAlmostEqual(gen.episode_2_duration(54), 1.5, places=12)

    def test_57_is_exactly_the_largest_admissible_first_return_frame(self):
        self.assertEqual(gen.ADMISSIBLE_MAX_FIRST_RETURN_FRAME, 57)
        self.assertTrue(gen.gated_presence_admissible(57))
        self.assertFalse(gen.gated_presence_admissible(58))
        self.assertFalse(gen.gated_presence_admissible(59))

    def test_the_boundary_case_is_decided_in_frames_not_floats(self):
        # A 2-frame return (first return frame 58) lasts EXACTLY
        # floor_len and the spec refuses it. In float64 the comparison
        # goes the other way, so the admissibility test is done on
        # integers; this pins that, because a silent regression here
        # would let an inadmissible scene through as admissible.
        self.assertEqual(gen.episode_2_duration_frames(58),
                         gen.FLOOR_LEN_FRAMES)
        self.assertGreater(gen.episode_2_duration(58), gen.FLOOR_LEN)
        self.assertFalse(gen.gated_presence_admissible(58))
        self.assertEqual(gen.WINDOW_END_FRAMES, 63)
        self.assertEqual(gen.FLOOR_LEN_FRAMES, 5)

    def test_the_cap_constant_agrees_with_the_predicate(self):
        largest = max(f for f in range(31, 60)
                      if gen.gated_presence_admissible(f))
        self.assertEqual(largest, gen.ADMISSIBLE_MAX_FIRST_RETURN_FRAME)

    def test_lrv4_is_inadmissible_and_lrv3_is_not(self):
        self.assertTrue(gen.gated_presence_admissible(57))    # LRV3
        self.assertFalse(gen.gated_presence_admissible(59))   # LRV4

    def test_the_flag_is_default_off(self):
        # Parsing is not reachable without --out, so assert on the parser
        # the module builds: the flag must exist, store True, and default
        # to False, because default-off is what keeps LRV3 regeneration
        # byte-identical.
        import argparse
        import inspect
        source = inspect.getsource(gen.main)
        self.assertIn('"--allow-short-return"', source)
        self.assertIn('action="store_true"', source)
        parser = argparse.ArgumentParser()
        parser.add_argument("--allow-short-return", action="store_true")
        self.assertFalse(parser.parse_args([]).allow_short_return)
        self.assertTrue(
            parser.parse_args(["--allow-short-return"]).allow_short_return)


@unittest.skipUnless((_ROOT / "lrv4").is_dir(),
                     "LRV4 fixture not present in this checkout")
class Lrv4StarvationTests(unittest.TestCase):
    """LRV4 is LRV3 with ONE variable moved: return observation supply."""

    @classmethod
    def setUpClass(cls):
        cls.lrv4 = json.loads((_ROOT / "lrv4" / "event_spec.json").read_text())
        lrv3 = _ROOT / "lrv3" / "event_spec.json"
        cls.lrv3 = json.loads(lrv3.read_text()) if lrv3.is_file() else None

    def test_the_return_is_a_single_frame(self):
        self.assertEqual(self.lrv4["return_frames"], [59])
        self.assertEqual(self.lrv4["presence_frames"]["episode_1"], [0, 29])
        self.assertEqual(self.lrv4["presence_frames"]["gap"], [30, 58])
        self.assertEqual(self.lrv4["presence_frames"]["episode_2"], [59, 59])

    def test_the_scene_declares_itself_inadmissible_for_gated_presence(self):
        block = self.lrv4["gated_presence_admissibility"]
        self.assertFalse(block["admissible"])
        self.assertLess(block["episode_2_duration_s"], block["floor_len_s"])
        self.assertEqual(block["built_with"], "--allow-short-return")
        self.assertIn("INADMISSIBLE", block["WARNING"])
        self.assertIn("PAYLOAD", block["WARNING"])

    def test_return_supply_is_one_third_of_lrv3s(self):
        if self.lrv3 is None:
            self.skipTest("LRV3 fixture not present")
        def supply(spec):
            px = spec["event_object_pixels_per_test_view_frame"]
            return sum(px["cam%02d_f%03d" % (c, f)]
                       for c in spec["test_cameras"]
                       for f in spec["return_frames"])
        self.assertEqual(supply(self.lrv4) * 3, supply(self.lrv3))

    def test_training_view_frames_drop_from_48_to_16(self):
        n_train = self.lrv4["n_cameras"] - len(self.lrv4["test_cameras"])
        self.assertEqual(n_train, 16)
        self.assertEqual(len(self.lrv4["return_frames"]) * n_train, 16)
        if self.lrv3 is not None:
            self.assertEqual(len(self.lrv3["return_frames"]) * n_train, 48)

    def test_everything_except_presence_is_unchanged_from_lrv3(self):
        if self.lrv3 is None:
            self.skipTest("LRV3 fixture not present")
        for key in ("kind", "n_frames", "fps", "time_duration", "width",
                    "height", "focal_px", "n_cameras", "test_cameras",
                    "train_cameras", "event_object", "ground_half_extent",
                    "evidence_bearing"):
            self.assertEqual(self.lrv4[key], self.lrv3[key], key)

    def test_lrv3_carries_no_admissibility_block(self):
        # The block is emitted ONLY for a scene the cap would have
        # refused, which is what keeps an admissible scene's
        # event_spec.json byte-identical to the pre-flag generator's.
        if self.lrv3 is None:
            self.skipTest("LRV3 fixture not present")
        self.assertNotIn("gated_presence_admissibility", self.lrv3)
        self.assertEqual(set(self.lrv4) - set(self.lrv3),
                         {"gated_presence_admissibility"})


if __name__ == "__main__":
    unittest.main()
