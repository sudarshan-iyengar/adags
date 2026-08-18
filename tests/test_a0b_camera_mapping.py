"""CPU tests for scripts/build_a0b_camera_mapping.py (the A0b D3 mapping).

Run with:
    python -m unittest tests.test_a0b_camera_mapping

Oracle style: a synthetic 12-camera rig whose geometry is known in closed
form, so the frozen selection rule is checked against a hand-derived answer
rather than against the reducer's own output.

Every camera sits at radius 4 looking back at the origin, so its optical axis
is the negated unit direction of its position and every pairwise separation is
computable by hand. `cam00`, `cam04` and `cam08` are held out by the mod-4
rule, leaving the tracking set (1, 2, 3, 5, 6, 7, 9, 10, 11).

Two sub-rigs carry the two halves of the frozen rule, because a single
sub-rig cannot test both:

  * **maximisation** is tested on (1, 2, 3, 5). Axes -x, -y, -z and a
    5-degree perturbation of -x. The four triples have minimum pairwise
    separations 90 / 5 / 5 / 85 degrees, so the winner (1, 2, 3) is UNIQUE
    and no tie-break is involved.
  * **the tie-break** is tested on (2, 3, 9, 11). Axes -y, -z, +x, +z. Every
    pair is 90 degrees apart except (3, 11) at 180, and since the rule takes
    the MINIMUM every one of the four triples scores exactly 90. The tie is
    genuine and four-wide, so the lexicographic tie-break is what decides it.

That second case is the one worth stating: a min-separation rule does not
penalise a back-to-back pair, so exact ties are the normal case on a
symmetric rig rather than an edge case.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402
from scripts import build_a0b_camera_mapping as cm  # noqa: E402
from scripts.build_elgs_tracks import load_temporal_scene  # noqa: E402

N_FRAMES = 2
IMAGE_SIZE = 64
FOCAL = 40.0
CAMERA_RADIUS = 4.0
PERTURBATION_DEG = 5.0

_P = math.radians(PERTURBATION_DEG)
#: unit positions in camera-index order; 0, 4 and 8 are held out by mod-4
POSITIONS = (
    (1.0, 0.0, 0.0),                        # 0  held out
    (1.0, 0.0, 0.0),                        # 1  axis -x
    (0.0, 1.0, 0.0),                        # 2  axis -y
    (0.0, 0.0, 1.0),                        # 3  axis -z
    (0.0, 1.0, 0.0),                        # 4  held out
    (math.cos(_P), math.sin(_P), 0.0),      # 5  axis 5 deg off -x
    (-1.0, 0.0, 0.0),                       # 6  filler
    (0.0, -1.0, 0.0),                       # 7  filler
    (0.0, 0.0, -1.0),                       # 8  held out
    (-1.0, 0.0, 0.0),                       # 9  axis +x
    (0.0, -1.0, 0.0),                       # 10 filler
    (0.0, 0.0, -1.0),                       # 11 axis +z
)
N_CAMERAS = len(POSITIONS)
TRACKING = (1, 2, 3, 5, 6, 7, 9, 10, 11)

MAXIMISATION_SUBRIG = (1, 2, 3, 5)
TIE_SUBRIG = (2, 3, 9, 11)


def _look_at_c2w(position: np.ndarray) -> np.ndarray:
    """Camera at ``position`` looking back at the origin, in the converter's
    own convention (the loader supplies the axis flip)."""

    z_axis = position / np.linalg.norm(position)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, z_axis))) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x_axis, y_axis, z_axis, position
    return c2w


def _write_scene(root: Path) -> Path:
    from PIL import Image

    scene = root / "scene"
    frames = []
    for cam_index, unit in enumerate(POSITIONS):
        position = CAMERA_RADIUS * np.asarray(unit, dtype=np.float64)
        c2w = _look_at_c2w(position)
        for frame_index in range(N_FRAMES):
            frames.append(
                {
                    "file_path": f"undist/cam{cam_index:02d}/{frame_index:08d}",
                    "transform_matrix": c2w.tolist(),
                    "fl_x": FOCAL,
                    "fl_y": FOCAL,
                    "cx": (IMAGE_SIZE - 1) / 2.0,
                    "cy": (IMAGE_SIZE - 1) / 2.0,
                    "w": IMAGE_SIZE,
                    "h": IMAGE_SIZE,
                    "time": frame_index / 120.0,
                }
            )
    scene.mkdir(parents=True)
    (scene / "transforms_train.json").write_text(
        json.dumps({"fps": 120.0, "frames": frames}), encoding="utf-8"
    )
    for cam_index in range(N_CAMERAS):
        mask_dir = scene / "masks" / f"cam{cam_index:02d}"
        mask_dir.mkdir(parents=True)
        image_dir = scene / "undist" / f"cam{cam_index:02d}"
        image_dir.mkdir(parents=True)
        (image_dir / "00000000.png").write_bytes(b"placeholder")
        for frame_index in range(N_FRAMES):
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            mask[20:40, 20:40] = 255
            Image.fromarray(mask).save(mask_dir / f"{frame_index:08d}.png")
    return scene


def _window(sequence: str, anchor, containing, *, seed_id: int = 0, first: int = 0):
    return {
        "sequence": sequence,
        "seed_id": seed_id,
        "first_frame": first,
        "last_frame": first + 8,
        "ltp_frame": first,
        "ltp": list(anchor),
        "containing_cameras": list(containing),
    }


class CameraMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.scene_dir = _write_scene(Path(cls._tmp.name))
        cls.scene = load_temporal_scene(cls.scene_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # -- an independent oracle for the separation, so the test does not
    # -- certify the reducer's geometry with the reducer's own helper
    def _separation_deg(self, a: int, b: int) -> float:
        axis_a = -np.asarray(POSITIONS[a], dtype=np.float64)
        axis_b = -np.asarray(POSITIONS[b], dtype=np.float64)
        cosine = float(
            np.dot(axis_a, axis_b) / (np.linalg.norm(axis_a) * np.linalg.norm(axis_b))
        )
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    def _brute_force(self, cameras):
        scored = [
            (min(self._separation_deg(a, b) for a, b in itertools.combinations(t, 2)), t)
            for t in itertools.combinations(sorted(cameras), 3)
        ]
        best = max(score for score, _ in scored)
        tied = sorted(t for score, t in scored if abs(score - best) < 1e-9)
        return tied[0], best, len(tied)

    # -- the rig is what the docstring claims ---------------------------

    def test_tracking_set_is_the_expected_nine_cameras(self):
        self.assertEqual(self.scene.tracking_ids, TRACKING)

    def test_optical_axes_point_from_each_camera_at_the_origin(self):
        """If this fails, every separation below is meaningless."""

        for camera_id in TRACKING:
            axis = cm.optical_axis(self.scene, camera_id)
            expected = -np.asarray(POSITIONS[camera_id], dtype=np.float64)
            np.testing.assert_allclose(axis, expected, atol=1e-12)

    def test_maximisation_subrig_has_the_hand_derived_separations(self):
        self.assertAlmostEqual(self._separation_deg(1, 5), PERTURBATION_DEG, places=9)
        self.assertAlmostEqual(
            self._separation_deg(2, 5), 90.0 - PERTURBATION_DEG, places=9
        )
        for pair in ((1, 2), (1, 3), (2, 3), (3, 5)):
            self.assertAlmostEqual(self._separation_deg(*pair), 90.0, places=9)

    def test_tie_subrig_has_exactly_one_opposite_pair(self):
        self.assertAlmostEqual(self._separation_deg(3, 11), 180.0, places=9)
        for pair in ((2, 3), (2, 9), (2, 11), (3, 9), (9, 11)):
            self.assertAlmostEqual(self._separation_deg(*pair), 90.0, places=9)

    # -- the frozen selection rule --------------------------------------

    def test_maximisation_picks_the_unique_best_triple(self):
        triple, separation, n_tied = cm.select_audit_triple(self.scene, MAXIMISATION_SUBRIG)
        self.assertEqual(n_tied, 1, "this sub-rig must not tie, or it tests nothing")
        self.assertEqual(triple, (1, 2, 3))
        self.assertAlmostEqual(math.degrees(separation), 90.0, places=9)

    def test_maximisation_rejects_the_near_duplicate_pair(self):
        """(1, 5) are 5 degrees apart, so any triple containing both scores 5
        and the rule must not return one."""

        triple, _, _ = cm.select_audit_triple(self.scene, MAXIMISATION_SUBRIG)
        self.assertFalse({1, 5} <= set(triple))

    def test_tie_break_is_the_lexicographically_smallest(self):
        """All four triples of this sub-rig score exactly 90, so the
        tie-break is what decides. Hand answer: (2, 3, 9)."""

        triple, separation, n_tied = cm.select_audit_triple(self.scene, TIE_SUBRIG)
        self.assertEqual(n_tied, 4)
        self.assertEqual(triple, (2, 3, 9))
        self.assertAlmostEqual(math.degrees(separation), 90.0, places=9)

    def test_selection_agrees_with_an_independent_brute_force(self):
        for cameras in (MAXIMISATION_SUBRIG, TIE_SUBRIG, TRACKING):
            triple, separation, n_tied = cm.select_audit_triple(self.scene, cameras)
            expected_triple, expected_deg, expected_tied = self._brute_force(cameras)
            self.assertEqual(triple, expected_triple, cameras)
            self.assertEqual(n_tied, expected_tied, cameras)
            self.assertAlmostEqual(math.degrees(separation), expected_deg, places=6)

    def test_selection_is_order_independent(self):
        forward = cm.select_audit_triple(self.scene, TRACKING)
        backward = cm.select_audit_triple(self.scene, tuple(reversed(TRACKING)))
        self.assertEqual(forward, backward)

    def test_triple_of_exactly_three_cameras_is_those_three(self):
        subset = (1, 2, 3)
        triple, _, n_tied = cm.select_audit_triple(self.scene, subset)
        self.assertEqual(triple, subset)
        self.assertEqual(n_tied, 1)

    def test_fewer_than_three_cameras_yields_no_triple(self):
        for cameras in ((), (1,), (1, 3)):
            triple, separation, n_tied = cm.select_audit_triple(self.scene, cameras)
            self.assertIsNone(triple)
            self.assertIsNone(separation)
            self.assertEqual(n_tied, 0)

    # -- S_w under the frozen frustum rule ------------------------------

    def test_origin_anchor_is_seen_by_every_tracking_camera(self):
        self.assertEqual(cm.applicable_cameras(self.scene, np.zeros(3)), TRACKING)

    def test_anchor_behind_a_camera_is_excluded_from_that_camera(self):
        """A point twice as far out as cam01 (which sits at +x looking in)
        lies BEHIND cam01, so the positive-depth half of the frozen predicate
        must drop cam01 -- and must keep cam09, which faces it."""

        anchor = np.array([2.0 * CAMERA_RADIUS, 0.0, 0.0])
        cameras = cm.applicable_cameras(self.scene, anchor)
        self.assertNotIn(1, cameras)
        self.assertIn(9, cameras)

    # -- the driver ------------------------------------------------------

    def _run(self, windows, sequence="seq", declared=None):
        with tempfile.TemporaryDirectory() as out_root:
            diagnostic = Path(out_root) / "diagnostic.json"
            diagnostic.write_text(
                json.dumps(
                    {
                        "audit_sample_B8": {
                            "seed": 20260814,
                            "n_selected": len(windows) if declared is None else declared,
                            "windows": windows,
                        }
                    }
                ),
                encoding="utf-8",
            )
            out = Path(out_root) / "mapping.json"
            code = cm.main(
                [
                    "--diagnostic", str(diagnostic),
                    "--scene-dir", str(self.scene_dir),
                    "--sequence-name", sequence,
                    "--out", str(out),
                ]
            )
            self.assertEqual(code, 0)
            return json.loads(out.read_text(encoding="utf-8"))

    def test_contract_passes_when_the_sealed_set_is_reproduced(self):
        result = self._run([_window("seq", (0.0, 0.0, 0.0), TRACKING)])
        self.assertTrue(result["contract_checks"]["passed"], result["contract_checks"])
        self.assertFalse(result["void"])
        row = result["per_window"][0]
        self.assertTrue(row["reproduces_sealed"])
        self.assertEqual(row["S_w_size"], len(TRACKING))
        self.assertEqual(row["audit_triple"], list(self._brute_force(TRACKING)[0]))

    def test_contract_FAILS_and_VOIDS_on_a_sealed_mismatch(self):
        """The load-bearing check: if the recomputed set does not reproduce
        the candidate generator's own sealed set, the run must VOID rather
        than quietly publish a different camera instrument."""

        result = self._run([_window("seq", (0.0, 0.0, 0.0), (1, 2, 3))])
        self.assertFalse(result["contract_checks"]["passed"])
        self.assertTrue(result["void"])
        self.assertEqual(result["contract_checks"]["n_reproduces_sealed"], 0)
        self.assertIn("reproduces the sealed", result["contract_checks"]["failures"][0])

    def test_declared_sample_size_mismatch_voids(self):
        result = self._run([_window("seq", (0.0, 0.0, 0.0), TRACKING)], declared=73)
        self.assertTrue(result["void"])
        self.assertTrue(
            any(
                "audit sample declares 73" in failure
                for failure in result["contract_checks"]["failures"]
            )
        )

    def test_unknown_sequence_raises_rather_than_silently_skipping(self):
        with self.assertRaises(ContractError):
            self._run([_window("some_other_sequence", (0.0, 0.0, 0.0), TRACKING)])

    def test_inadmissible_window_is_named_not_dropped(self):
        """|S_w| < 3 must appear in the artifact as an exclusion carrying its
        key, because the prereg requires every excluded candidate to be
        named rather than silently dropped."""

        far = (0.0, 0.0, 400.0)
        sealed = cm.applicable_cameras(self.scene, np.asarray(far))
        self.assertLess(len(sealed), 3, "fixture no longer produces a small S_w")
        result = self._run([_window("seq", far, sealed)])
        disclosure = result["disclosure"]
        self.assertEqual(
            disclosure["n_windows_with_fewer_than_three_applicable_cameras"], 1
        )
        self.assertEqual(disclosure["inadmissible_window_keys"], ["seq|0|0"])
        self.assertFalse(result["per_window"][0]["admissible"])
        self.assertIsNone(result["per_window"][0]["audit_triple"])
        self.assertTrue(result["contract_checks"]["passed"])

    def test_disclosure_reports_size_distribution_and_alternatives(self):
        result = self._run([_window("seq", (0.0, 0.0, 0.0), TRACKING)])
        disclosure = result["disclosure"]
        distribution = disclosure["S_w_size_distribution"]
        self.assertEqual(distribution["min"], len(TRACKING))
        self.assertEqual(distribution["max"], len(TRACKING))
        self.assertEqual(distribution["histogram"], {str(len(TRACKING)): 1})
        self.assertEqual(
            disclosure["n_windows_with_fewer_than_three_applicable_cameras"], 0
        )
        self.assertIn(
            "D1_fixed_triple",
            disclosure["candidate_independent_alternatives_MEASURED_AND_REJECTED"],
        )

    def test_anchor_outside_the_camera_sphere_is_flagged(self):
        far = (0.0, 0.0, 400.0)
        sealed = cm.applicable_cameras(self.scene, np.asarray(far))
        result = self._run([_window("seq", far, sealed)])
        flags = result["disclosure"]["data_quality_flags"]
        self.assertEqual(len(flags), 1)
        self.assertGreater(flags[0]["anchor_distance_over_rig_radius"], 1.0)

    def test_artifact_records_the_narrowed_estimand_and_its_limits(self):
        result = self._run([_window("seq", (0.0, 0.0, 0.0), TRACKING)])
        prereg = result["prereg"]
        self.assertEqual(prereg["adopted_definition"], "D3_census_per_candidate_frustum")
        self.assertIn(
            "geometrically considered applicable",
            prereg["estimand_this_mapping_licenses"],
        )
        self.assertIn("literal physical absence", prereg["does_not_establish"])

    def test_artifact_is_reproducible(self):
        a = self._run([_window("seq", (0.0, 0.0, 0.0), TRACKING)])
        b = self._run([_window("seq", (0.0, 0.0, 0.0), TRACKING)])
        for payload in (a, b):
            payload.pop("source")
            payload.pop("config_sha256")
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
