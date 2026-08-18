"""CPU tests for scripts/build_coverage_bounding_pair.py (A1 reducer).

Run with:
    python -m unittest tests.test_coverage_bounding_pair

Oracle style: a hand-engineered synthetic scene in which each of the three
readings, and both section-4 sensitivity readings, is separated by
construction. Four cameras (cam00 held out by the mod-4 rule), four frames,
64x64 masks, two eligible components per mask:

  * a horizontal BAR (4 rows x 20 cols = 80 px) centred on the projection of
    the identity's consensus point -- the "anchor" component;
  * an untracked SQUARE (12x12 = 144 px) in a corner.

One seed, one report per (training camera, frame):

  | frame | report        | v | consensus at frame | admitted by            |
  |-------|---------------|---|--------------------|------------------------|
  | 0     | on the bar    | 1 | defined            | (i), (ii), (iii), both sensitivities |
  | 1     | on the bar    | 0 | defined            | (ii), (iii), last-defined |
  | 2     | on the SQUARE | 0 | defined            | (ii) only              |
  | 3     | on the bar    | 0 | NONE               | (ii), last-defined only |

The bar is deliberately a bar rather than a square so that reading the
anchor as ``labels[col, row]`` instead of ``labels[row, col]`` falls off it.
That makes the transposed sensitivity reading discriminating rather than
decorative, and the fixture asserts the geometry that guarantees it.

Hand-derived expectations, 3 training cameras x 4 frames x 2 components:

    components_total            = 24
    (i)   frozen                =  3   (frame 0 only)
    (ii)  any-report            = 12   (every frame's report is on some
                                        eligible component)
    (iii) anchor-agreeing       =  6   (frames 0 and 1)
    (iii) transposed            =  3   (frame 0 only; frame 1's transposed
                                        anchor is background)
    (iii) last-defined anchor   =  9   (frames 0, 1 and 3)
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from elgs.tracks_schema import TRACKS_SCHEMA  # noqa: E402
from scripts import build_coverage_bounding_pair as bp  # noqa: E402
from scripts.build_elgs_tracks import load_temporal_scene  # noqa: E402

N_FRAMES = 4
IMAGE_SIZE = 64
FOCAL = 80.0
CAMERA_RADIUS = 4.0
TRAINING = (1, 2, 3)
# off-axis so the projection is NOT on the image diagonal
CONSENSUS_POINT = (0.0, 0.0, 0.35)
SQUARE_ROWS = slice(2, 14)
SQUARE_COLS = slice(2, 14)


def _look_at_c2w(position: np.ndarray) -> np.ndarray:
    z_axis = position / np.linalg.norm(position)
    up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x_axis, y_axis, z_axis, position
    return c2w


def _independent_projection(camera, point: np.ndarray) -> tuple[int, int]:
    """A 3-line oracle for the frozen projection, written here so the test
    does not certify the reducer's projection with the reducer's own code."""

    cam_point = camera.w2c[:3, :3] @ np.asarray(point, dtype=np.float64) + camera.w2c[:3, 3]
    assert cam_point[2] > 0.0
    col = int(np.floor(camera.K[0, 0] * cam_point[0] / cam_point[2] + camera.K[0, 2] + 0.5))
    row = int(np.floor(camera.K[1, 1] * cam_point[1] / cam_point[2] + camera.K[1, 2] + 0.5))
    return col, row


def _write_transforms(scene: Path) -> None:
    frames = []
    angles = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    for cam_index, angle in enumerate(angles):
        position = CAMERA_RADIUS * np.array([np.cos(angle), np.sin(angle), 0.0])
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


def _write_scene(root: Path) -> tuple[Path, Path, dict[int, tuple[int, int]]]:
    from PIL import Image

    scene = root / "scene"
    _write_transforms(scene)

    # the loader validates the mask tree before any projection can be
    # computed, so lay down the square-only masks first, read the calibration
    # back, then rewrite the masks with the anchor bar in place
    for cam_index in range(4):
        mask_dir = scene / "masks" / f"cam{cam_index:02d}"
        mask_dir.mkdir(parents=True)
        image_dir = scene / "undist" / f"cam{cam_index:02d}"
        image_dir.mkdir(parents=True)
        (image_dir / "00000000.png").write_bytes(b"placeholder")
        for frame_index in range(N_FRAMES):
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            mask[SQUARE_ROWS, SQUARE_COLS] = 255
            Image.fromarray(mask).save(mask_dir / f"{frame_index:08d}.png")

    bundle = load_temporal_scene(scene)
    anchors: dict[int, tuple[int, int]] = {}
    for camera_id in TRAINING:
        camera = bundle.cameras[camera_id]
        anchors[camera_id] = _independent_projection(camera, CONSENSUS_POINT)

    for cam_index in TRAINING:
        mask_dir = scene / "masks" / f"cam{cam_index:02d}"
        col, row = anchors[cam_index]
        for frame_index in range(N_FRAMES):
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            mask[SQUARE_ROWS, SQUARE_COLS] = 255
            mask[row - 2 : row + 2, col - 10 : col + 10] = 255
            Image.fromarray(mask).save(mask_dir / f"{frame_index:08d}.png")

    tracks_path = root / "tracks.json"
    tracks_path.write_text(json.dumps(_build_tracks(anchors)), encoding="utf-8")
    return scene, tracks_path, anchors


def _build_tracks(anchors: dict[int, tuple[int, int]]) -> dict:
    square_col = (SQUARE_COLS.start + SQUARE_COLS.stop) // 2
    square_row = (SQUARE_ROWS.start + SQUARE_ROWS.stop) // 2
    tracks = []
    for track_id, camera_id in enumerate(TRAINING):
        col, row = anchors[camera_id]
        reports = [
            {"frame": 0.0, "is_miss": False, "v": 1.0, "x": float(col), "y": float(row)},
            {"frame": 1.0, "is_miss": False, "v": 0.0, "x": float(col), "y": float(row)},
            {
                "frame": 2.0,
                "is_miss": False,
                "v": 0.0,
                "x": float(square_col),
                "y": float(square_row),
            },
            {"frame": 3.0, "is_miss": False, "v": 0.0, "x": float(col), "y": float(row)},
        ]
        tracks.append(
            {
                "track_id": track_id,
                "seed_id": 0,
                "camera_id": camera_id,
                "reports": reports,
            }
        )
    consensus = [
        {"frame": float(f), "point": list(CONSENSUS_POINT), "n_cam": 3}
        for f in range(3)
    ]
    consensus.append({"frame": 3.0, "point": None, "n_cam": 0})
    return {
        "schema_version": TRACKS_SCHEMA,
        "seeds": [{"seed_id": 0, "point": list(CONSENSUS_POINT), "n_cam": 3}],
        "tracks": tracks,
        "consensus": {"0": consensus},
        "window": {
            "frame_indices": list(range(N_FRAMES)),
            "query_frame": 0,
            "fps": 120.0,
        },
        "manifest": {
            "training_cameras": list(TRAINING),
            "held_out_cameras": [0],
            "tracker_identity": "hand-built-a1-fixture",
            "source_data_sha256": "0" * 64,
        },
    }


class CoverageBoundingPairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.scene_dir, cls.tracks_path, cls.anchors = _write_scene(Path(cls._tmp.name))
        cls.result = bp.bounding_pair_one_sequence(
            cls.scene_dir, cls.tracks_path, sequence="fixture"
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_fixture_geometry_makes_the_transpose_discriminating(self):
        """If the anchor were on the diagonal the transposed reading would be
        vacuously equal to the primary one, so the fixture asserts it is not."""

        for camera_id, (col, row) in self.anchors.items():
            self.assertGreater(
                abs(col - row), 2, f"cam{camera_id:02d} anchor {col},{row} too diagonal"
            )
            self.assertTrue(10 <= col < IMAGE_SIZE - 10)
            self.assertTrue(2 <= row < IMAGE_SIZE - 2)

    def test_reducer_projection_matches_the_independent_oracle(self):
        bundle = load_temporal_scene(self.scene_dir)
        for camera_id in TRAINING:
            camera = bundle.cameras[camera_id]
            self.assertEqual(
                bp._project(camera, np.asarray(CONSENSUS_POINT, dtype=np.float64)),
                self.anchors[camera_id],
            )

    def test_denominator_is_shared_by_every_reading(self):
        self.assertEqual(self.result["components_total"], 24)

    def test_reading_i_is_the_frozen_visible_only_rule(self):
        self.assertEqual(self.result["components_covered"][bp.R_FROZEN], 3)

    def test_reading_ii_admits_every_in_domain_report(self):
        self.assertEqual(self.result["components_covered"][bp.R_ANY], 12)

    def test_reading_iii_admits_anchor_agreeing_invisible_reports_only(self):
        self.assertEqual(self.result["components_covered"][bp.R_ANCHOR], 6)

    def test_transposed_anchor_sensitivity_has_teeth(self):
        self.assertEqual(self.result["components_covered"][bp.R_ANCHOR_T], 3)

    def test_last_defined_anchor_sensitivity_rescues_the_undefined_frame(self):
        self.assertEqual(self.result["components_covered"][bp.R_ANCHOR_LAST], 9)

    def test_anchor_undefined_is_counted_not_admitted(self):
        self.assertEqual(self.result["tallies"]["anchor_undefined"], len(TRAINING))

    def test_monotonicity_holds(self):
        cov = self.result["coverage"]
        self.assertLessEqual(cov[bp.R_FROZEN], cov[bp.R_ANCHOR])
        self.assertLessEqual(cov[bp.R_ANCHOR], cov[bp.R_ANY])

    def test_no_duplicate_visible_keys_in_the_fixture(self):
        self.assertEqual(self.result["tallies"]["visible_duplicate_keys"], 0)


class ContractAndClassificationTest(unittest.TestCase):
    def test_contract_check_catches_a_sealed_mismatch(self):
        row = {
            "sequence": "s",
            "coverage": {
                bp.R_FROZEN: 0.1,
                bp.R_ANCHOR: 0.2,
                bp.R_ANY: 0.3,
                bp.R_ANCHOR_T: 0.1,
                bp.R_ANCHOR_LAST: 0.2,
            },
            "components_total": 100,
            "components_covered": {bp.R_FROZEN: 10},
            "tallies": {"unreadable_masks": 0},
        }
        ok = bp._check_contracts(
            [row], {"s": {"components_total": 100, "components_covered": 10}}
        )
        self.assertTrue(ok["passed"], ok["failures"])
        bad = bp._check_contracts(
            [row], {"s": {"components_total": 100, "components_covered": 11}}
        )
        self.assertFalse(bad["passed"])
        self.assertIn("reading (i) numerator", bad["failures"][0])

    def test_contract_check_catches_a_monotonicity_violation(self):
        row = {
            "sequence": "s",
            "coverage": {
                bp.R_FROZEN: 0.4,
                bp.R_ANCHOR: 0.2,  # below the lower bound: impossible
                bp.R_ANY: 0.3,
                bp.R_ANCHOR_T: 0.1,
                bp.R_ANCHOR_LAST: 0.2,
            },
            "components_total": 100,
            "components_covered": {bp.R_FROZEN: 40},
            "tallies": {"unreadable_masks": 0},
        }
        out = bp._check_contracts(
            [row], {"s": {"components_total": 100, "components_covered": 40}}
        )
        self.assertFalse(out["passed"])
        self.assertIn("monotonicity violated", out["failures"][0])

    def test_classes_follow_the_frozen_rule(self):
        eligible = {
            bp.R_FROZEN: 0.30,
            bp.R_ANCHOR: 0.60,
            bp.R_ANY: 0.80,
            bp.R_ANCHOR_T: 0.55,
            bp.R_ANCHOR_LAST: 0.65,
        }
        self.assertEqual(bp.classify(eligible), "eligible")

        ineligible = {
            bp.R_FROZEN: 0.10,
            bp.R_ANCHOR: 0.20,
            bp.R_ANY: 0.30,
            bp.R_ANCHOR_T: 0.15,
            bp.R_ANCHOR_LAST: 0.25,
        }
        self.assertEqual(bp.classify(ineligible), "ineligible")

        indeterminate = {
            bp.R_FROZEN: 0.10,
            bp.R_ANCHOR: 0.40,
            bp.R_ANY: 0.70,
            bp.R_ANCHOR_T: 0.35,
            bp.R_ANCHOR_LAST: 0.45,
        }
        self.assertEqual(bp.classify(indeterminate), "indeterminate")

    def test_convention_dependence_forces_indeterminate(self):
        # primary says eligible, the transposed sensitivity says not: the
        # frozen rule demotes it rather than picking the convenient reading
        flipped = {
            bp.R_FROZEN: 0.30,
            bp.R_ANCHOR: 0.55,
            bp.R_ANY: 0.80,
            bp.R_ANCHOR_T: 0.45,
            bp.R_ANCHOR_LAST: 0.60,
        }
        self.assertEqual(bp.classify(flipped), "indeterminate")


if __name__ == "__main__":
    unittest.main()
