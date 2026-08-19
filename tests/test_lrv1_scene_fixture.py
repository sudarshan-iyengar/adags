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

# Checks whichever leave-and-return fixture is present. LRV2 differs from
# LRV1 only in the ground-plane extent, so the same geometry invariants must
# hold for both and the later scene is the one that matters.
_ROOT = Path(__file__).resolve().parents[1] / "data" / "synthetic"
SCENE = next((_ROOT / n for n in ("lrv2", "lrv1") if (_ROOT / n).is_dir()),
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


if __name__ == "__main__":
    unittest.main()
