"""CPU tests for scripts/build_m1_census.py (M1-A0 evaluator; prereg rev 3).

Run with:
    C:/Users/sucar/venvs/elgs-cpu/Scripts/python.exe -m unittest tests.test_elgs_m1_census

Oracle style: a fully hand-computed synthetic scene — 4-camera look-at rig
(cam00 held out by the mod-4 rule), 40 frames of engineered 64x64 masks,
and a hand-built tracks artifact — such that every gated statistic has a
closed-form expected value:
  - ONE flicker-bridged true-absence candidate (frames 10..24, one 2-frame
    bridged interruption at 15..16),
  - ONE same-object return (terminating re-appearance at 25 with consensus
    0.1 world units from the disappearance position; r_site_census ~ 0.211),
  - ONE occlusion event (cam01 loses the component for frames 30..39 = 10
    consecutive frames while cam02/cam03 keep it),
  - coverage EXACTLY 71/191 (every mask frame carries an untracked
    eligible component; the tracked component exists on 17 cam01 frames
    and 27 cam02/cam03 frames, all covered by visible reports).
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

from depth_visibility.errors import ContractError  # noqa: E402
from elgs.tracks_schema import TRACKS_SCHEMA  # noqa: E402
from scripts import build_m1_census as census  # noqa: E402

N_FRAMES = 40
IMAGE_SIZE = 64
FOCAL = 80.0
CAMERA_RADIUS = 4.0

PRESENT = sorted(set(range(0, 10)) | {15, 16} | set(range(25, 40)))
ABSENT_WINDOW = [f for f in range(10, 25) if f not in (15, 16)]
OCCLUDED_CAM01 = set(range(30, 40))  # cam01-only mask dropout


def _look_at_c2w(position: np.ndarray) -> np.ndarray:
    z_axis = position / np.linalg.norm(position)
    up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x_axis, y_axis, z_axis, position
    return c2w


def _seed_present(camera_id: int, frame: int) -> bool:
    if frame not in PRESENT:
        return False
    if camera_id == 1 and frame in OCCLUDED_CAM01:
        return False
    return True


def _write_scene(root: Path, *, return_offset: float) -> tuple[Path, Path]:
    from PIL import Image

    scene = root / "scene"
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

    for cam_index in range(4):
        mask_dir = scene / "masks" / f"cam{cam_index:02d}"
        mask_dir.mkdir(parents=True)
        for frame_index in range(N_FRAMES):
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            if _seed_present(cam_index, frame_index):
                mask[8:20, 8:20] = 255  # tracked component (144 px >= 64)
            mask[40:52, 40:52] = 255  # untracked component, every frame
            Image.fromarray(mask).save(mask_dir / f"{frame_index:08d}.png")
        # placeholder frame images are not read by the census
        image_dir = scene / "undist" / f"cam{cam_index:02d}"
        image_dir.mkdir(parents=True)
        (image_dir / "00000000.png").write_bytes(b"placeholder")

    tracks = _build_tracks_artifact(return_offset=return_offset)
    tracks_path = root / "tracks.json"
    tracks_path.write_text(json.dumps(tracks), encoding="utf-8")
    return scene, tracks_path


def _build_tracks_artifact(*, return_offset: float) -> dict:
    training = [1, 2, 3]
    seeds = [{"seed_id": 0, "point": [0.0, 0.0, 0.0], "n_cam": 3}]
    tracks = []
    track_id = 0
    for camera_id in training:
        reports = []
        for frame in range(N_FRAMES):
            if _seed_present(camera_id, frame):
                reports.append(
                    {"frame": float(frame), "is_miss": False, "v": 0.9, "x": 12.0, "y": 12.0}
                )
            else:
                reports.append({"frame": float(frame), "is_miss": True})
        tracks.append(
            {"track_id": track_id, "seed_id": 0, "camera_id": camera_id, "reports": reports}
        )
        track_id += 1
    consensus = []
    for frame in range(N_FRAMES):
        visible = sum(1 for c in training if _seed_present(c, frame))
        if visible >= 2:
            point = [0.0, 0.0, 0.0]
            if frame >= 25:
                point = [return_offset, 0.0, 0.0]
            consensus.append({"frame": float(frame), "point": point, "n_cam": visible})
        else:
            consensus.append({"frame": float(frame), "point": None, "n_cam": visible})
    return {
        "schema_version": TRACKS_SCHEMA,
        "seeds": seeds,
        "tracks": tracks,
        "consensus": {"0": consensus},
        "window": {"frame_indices": list(range(N_FRAMES)), "query_frame": 0, "fps": 120.0},
        "manifest": {
            "training_cameras": training,
            "held_out_cameras": [0],
            "tracker_identity": "hand-built-census-fixture",
            "source_data_sha256": "0" * 64,
        },
    }


class CensusStatisticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.scene_dir, cls.tracks_path = _write_scene(
            Path(cls._tmp.name), return_offset=0.1
        )
        cls.result = census.run_census(
            cls.scene_dir,
            cls.tracks_path,
            REPO_ROOT / "configs" / "elgs" / "prereg_m1_census_v1.json",
            apply_gate=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_true_absence_candidate(self):
        self.assertEqual(self.result["statistics"]["true_absence_candidate_count"], 1)
        record = self.result["records"]["true_absence_candidates"][0]
        self.assertEqual(record["first_frame"], 10)
        self.assertEqual(record["last_frame"], 24)
        self.assertEqual(record["n_frames"], 15)  # includes the bridged 15..16
        self.assertEqual(record["bridged_interruptions"], 1)
        self.assertEqual(sorted(record["containing_cameras"]), [1, 2, 3])

    def test_same_object_return_within_r_site(self):
        self.assertEqual(self.result["statistics"]["same_object_return_count"], 1)
        record = self.result["records"]["true_absence_candidates"][0]
        self.assertEqual(record["return"], "same_object_return")
        self.assertEqual(record["return_frame"], 25)
        self.assertAlmostEqual(record["return_distance"], 0.1, places=9)
        self.assertGreater(self.result["constants"]["r_site_census"], 0.1)

    def test_occlusion_event(self):
        self.assertEqual(self.result["statistics"]["occlusion_opportunity_upper_bound"], 1)
        event = self.result["records"]["occlusion_events"][0]
        self.assertEqual(event["seed_id"], 0)
        self.assertEqual(event["absent_camera"], 1)
        self.assertEqual(event["first_frame"], 30)
        self.assertEqual(event["last_frame"], 39)
        self.assertEqual(event["n_frames"], 10)

    def test_coverage_hand_derived(self):
        # Hand count: the tracked square exists on 17 cam01 frames (PRESENT
        # minus the 30..39 dropout) and 27 frames on cam02/cam03; the
        # untracked square exists on EVERY frame of every camera. So
        # components_total = (17*2 + 23) + 2*(27*2 + 13) = 191 and
        # components_covered = 17 + 27 + 27 = 71.
        tallies = self.result["coverage_tallies"]
        self.assertEqual(tallies["components_total"], 191)
        self.assertEqual(tallies["components_covered"], 71)
        self.assertAlmostEqual(
            self.result["statistics"]["track_coverage_upper_bound"], 71 / 191, places=12
        )

    def test_gate_application(self):
        gate = self.result["gate"]
        self.assertFalse(gate["comparisons"]["occlusion_events_min"])  # 1 < 36
        self.assertFalse(gate["comparisons"]["true_absence_candidates_min"])
        self.assertFalse(gate["comparisons"]["same_object_returns_min"])
        self.assertFalse(gate["comparisons"]["track_coverage_min_fraction"])  # 71/191 < 0.5
        self.assertFalse(gate["pass"])

    def test_window_and_provenance(self):
        self.assertEqual(self.result["window"], {"first_frame": 0, "last_frame": 39, "n_frames": 40})
        self.assertEqual(self.result["provenance"]["training_cameras"], [1, 2, 3])
        self.assertEqual(self.result["provenance"]["held_out_cameras"], [0])
        self.assertEqual(self.result["provenance"]["prereg_revision"], 3)


class ReturnBeyondRSiteTest(unittest.TestCase):
    def test_beyond_r_site_is_not_a_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir, tracks_path = _write_scene(Path(tmp), return_offset=1.0)
            result = census.run_census(
                scene_dir,
                tracks_path,
                REPO_ROOT / "configs" / "elgs" / "prereg_m1_census_v1.json",
                apply_gate=False,
            )
            self.assertEqual(result["statistics"]["true_absence_candidate_count"], 1)
            self.assertEqual(result["statistics"]["same_object_return_count"], 0)
            record = result["records"]["true_absence_candidates"][0]
            self.assertEqual(record["return"], "beyond_r_site")
            self.assertNotIn("gate", result)


class FrozenGuardTests(unittest.TestCase):
    def test_round_half_up_is_platform_independent(self):
        self.assertEqual(census.round_half_up(12.5), 13)
        self.assertEqual(census.round_half_up(12.49), 12)
        self.assertEqual(census.round_half_up(-0.5), 0)
        self.assertEqual(census.round_half_up(-0.51), -1)

    def test_prereg_revision_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prereg.json"
            payload = json.loads(
                (REPO_ROOT / "configs" / "elgs" / "prereg_m1_census_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            payload["revision"] = 2
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ContractError):
                census.load_prereg(path)

    def test_prereg_floor_tamper_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prereg.json"
            payload = json.loads(
                (REPO_ROOT / "configs" / "elgs" / "prereg_m1_census_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            payload["floors"]["occlusion_events_min"] = 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ContractError):
                census.load_prereg(path)


if __name__ == "__main__":
    unittest.main()
