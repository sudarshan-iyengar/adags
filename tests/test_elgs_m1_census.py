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
            [(cls.scene_dir, cls.tracks_path)],
            REPO_ROOT / "configs" / "elgs" / "prereg_m1_census_v1.json",
            apply_gate=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _sequence(self):
        return self.result["per_sequence"]["scene"]

    def test_true_absence_candidate(self):
        self.assertEqual(self.result["statistics"]["true_absence_candidate_count"], 1)
        record = self._sequence()["records"]["true_absence_candidates"][0]
        self.assertEqual(record["first_frame"], 10)
        self.assertEqual(record["last_frame"], 24)
        self.assertEqual(record["n_frames"], 15)  # includes the bridged 15..16
        self.assertEqual(record["bridged_interruptions"], 1)
        self.assertEqual(sorted(record["containing_cameras"]), [1, 2, 3])

    def test_same_object_return_within_r_site(self):
        self.assertEqual(self.result["statistics"]["same_object_return_count"], 1)
        record = self._sequence()["records"]["true_absence_candidates"][0]
        self.assertEqual(record["return"], "same_object_return_primary")
        self.assertEqual(record["return_frame"], 25)
        self.assertAlmostEqual(record["return_distance"], 0.1, places=9)
        self.assertGreater(self._sequence()["constants"]["r_site_census"], 0.1)

    def test_occlusion_event(self):
        self.assertEqual(self.result["statistics"]["occlusion_opportunity_upper_bound"], 1)
        event = self._sequence()["records"]["occlusion_events"][0]
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
        # Per-half tallies (cycle-3 gate; split_frame = 20): hand count —
        # first half: 3 cams x (12 tracked-present + 20 untracked) = 96
        # total / 36 covered; second half: cam01 5+20, cam02/03 15+20 = 95
        # total / 35 covered.
        by_half = self.result["per_sequence"]["scene"]["coverage_tallies"]["by_half"]
        self.assertEqual(by_half["first_half"], {"components_total": 96, "components_covered": 36})
        self.assertEqual(by_half["second_half"], {"components_total": 95, "components_covered": 35})
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
        self.assertEqual(self._sequence()["window"], {"first_frame": 0, "last_frame": 39, "n_frames": 40})
        self.assertEqual(self._sequence()["provenance"]["training_cameras"], [1, 2, 3])
        self.assertEqual(self._sequence()["provenance"]["held_out_cameras"], [0])
        self.assertEqual(self.result["provenance"]["prereg_revision"], 3)


class ReturnBeyondRSiteTest(unittest.TestCase):
    def test_beyond_r_site_is_not_a_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir, tracks_path = _write_scene(Path(tmp), return_offset=1.0)
            result = census.run_census(
                [(scene_dir, tracks_path)],
                REPO_ROOT / "configs" / "elgs" / "prereg_m1_census_v1.json",
                apply_gate=False,
            )
            self.assertEqual(result["statistics"]["true_absence_candidate_count"], 1)
            self.assertEqual(result["statistics"]["same_object_return_count"], 0)
            record = result["per_sequence"]["scene"]["records"]["true_absence_candidates"][0]
            self.assertEqual(record["return"], "beyond_r_site")
            self.assertNotIn("gate", result)


class Cycle2AlignmentTests(unittest.TestCase):
    """The prereg_m1_cycle2_screen_v1 frozen readings and union predicate."""

    def _run(self, return_offset, mutate_tracks=None, recenter_masks_from=None):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir, tracks_path = _write_scene(Path(tmp), return_offset=return_offset)
            if recenter_masks_from is not None:
                # Redraw the tracked component at the image center for
                # frames >= recenter_masks_from (so center-pixel reports
                # stay associated), in every camera that shows it.
                from PIL import Image

                for cam_index in range(4):
                    for frame_index in range(recenter_masks_from, N_FRAMES):
                        if not _seed_present(cam_index, frame_index):
                            continue
                        mask_path = (
                            scene_dir / "masks" / f"cam{cam_index:02d}" / f"{frame_index:08d}.png"
                        )
                        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
                        mask[26:38, 26:38] = 255  # centered tracked component
                        mask[40:52, 40:52] = 255  # untracked component
                        Image.fromarray(mask).save(mask_path)
            if mutate_tracks is not None:
                payload = json.loads(tracks_path.read_text(encoding="utf-8"))
                mutate_tracks(payload)
                tracks_path.write_text(json.dumps(payload), encoding="utf-8")
            return census.run_census(
                [(scene_dir, tracks_path)],
                REPO_ROOT / "configs" / "elgs" / "prereg_m1_census_v1.json",
                apply_gate=False,
            )["per_sequence"]["scene"]

    def test_r3_transition_rule_blocks_absent_from_start(self):
        # Remove all associations before frame 12 (masks still show the
        # square, but the identity has no visible reports until 12): with
        # the transition rule, absence starting at the sequence start can
        # never open a window at its first ELIGIBLE frame without a
        # preceding associated frame.
        def mutate(payload):
            for track in payload["tracks"]:
                for report in track["reports"]:
                    if report["frame"] < 12:
                        report.clear()
                        report.update({"frame": report.get("frame", 0.0), "is_miss": True})

        # rebuild reports properly (clear/update above loses the frame key
        # ordering); simpler: mark frames < 12 as miss
        def mutate_clean(payload):
            for track in payload["tracks"]:
                new_reports = []
                for report in track["reports"]:
                    if report["frame"] < 12:
                        new_reports.append({"frame": report["frame"], "is_miss": True})
                    else:
                        new_reports.append(report)
                track["reports"] = new_reports

        sequence = self._run(0.1, mutate_clean)
        # The original candidate window (10..24) required presence at 9;
        # with reports gone until 12 there is no associated frame before
        # the absence, so no transition => no candidate for that window.
        starts = [c["first_frame"] for c in sequence["records"]["true_absence_candidates"]]
        self.assertNotIn(10, starts)

    def test_union_r2prime_counts_undefined_consensus_return(self):
        # Break the consensus at the re-appearance frames (>= 25) so the
        # PRIMARY return is undefined, while the per-camera reports remain:
        # the reports sit at pixel (12, 12), which is NOT the projection of
        # the LTP (origin projects to the image center 31.5), so R2-prime
        # must NOT fire either -> return_position_undefined.
        def kill_consensus(payload):
            for entry in payload["consensus"]["0"]:
                if entry["frame"] >= 25:
                    entry["point"] = None

        sequence = self._run(0.1, kill_consensus)
        record = next(
            c for c in sequence["records"]["true_absence_candidates"] if c["first_frame"] == 10
        )
        self.assertEqual(record["return"], "return_position_undefined")
        self.assertEqual(sequence["statistics"]["union_return_count"], 0)

        # Now ALSO move the re-appearance reports onto the LTP's projection
        # (image center) in every camera: R2-prime fires (>= 2 separated
        # cameras within the projected r_site tolerance), so the union
        # counts the return the primary cannot.
        def kill_consensus_and_center_reports(payload):
            kill_consensus(payload)
            for track in payload["tracks"]:
                for report in track["reports"]:
                    if not report["is_miss"] and report["frame"] >= 25:
                        report["x"] = 31.5
                        report["y"] = 31.5

        sequence = self._run(0.1, kill_consensus_and_center_reports, recenter_masks_from=25)
        record = next(
            c for c in sequence["records"]["true_absence_candidates"] if c["first_frame"] == 10
        )
        self.assertEqual(record["return"], "same_object_return_r2prime")
        self.assertEqual(sequence["statistics"]["union_return_count"], 1)
        self.assertEqual(sequence["statistics"]["same_object_return_count"], 0)  # primary strict

    def test_halves_attribution(self):
        sequence = self._run(0.1)
        halves = sequence["halves"]
        self.assertEqual(halves["split_frame"], 20)  # frames[40//2]
        # true-absence candidate first_frame 10 -> first half
        self.assertEqual(halves["true_absence"], {"first_half": 1, "second_half": 0})
        # its terminating re-appearance run starts at 25 -> second half
        self.assertEqual(halves["union_returns"], {"first_half": 0, "second_half": 1})
        # occlusion event first_frame 30 -> second half
        self.assertEqual(halves["occlusion"], {"first_half": 0, "second_half": 1})

    def test_angular_floor_is_calibration_only_and_positive(self):
        sequence = self._run(0.1)
        self.assertGreater(sequence["angular_floor_rad"], 0.0)

    def test_d2_anchor_searches_full_maximal_run(self):
        # Verifier D2 pinning: consensus undefined for the first 6 frames of
        # the terminating run (25..30) and defined only at 31 with a
        # within-r_site point. Under the 4-frame-prefix bug the primary
        # anchor search would end at frame 28 and tally
        # return_position_undefined; the frozen full-run search finds 31.
        def defined_only_late(payload):
            for entry in payload["consensus"]["0"]:
                if 25 <= entry["frame"] <= 30:
                    entry["point"] = None

        sequence = self._run(0.1, defined_only_late)
        record = next(
            c for c in sequence["records"]["true_absence_candidates"] if c["first_frame"] == 10
        )
        self.assertEqual(record["return"], "same_object_return_primary")
        self.assertEqual(record["return_frame"], 31)
        self.assertEqual(record["return_run_start"], 25)
        self.assertEqual(record["return_run_end"], 39)
        self.assertEqual(sequence["statistics"]["same_object_return_count"], 1)

    def test_d1_half_pixel_boundary_band_is_in_domain(self):
        # Verifier D1 pinning: a report at x = -0.3 rounds to pixel 0 (in
        # domain under R8) and must associate/count; x = -0.6 rounds to -1
        # (out of domain) and must be dropped.
        self.assertEqual(census.round_half_up(-0.3), 0)
        self.assertEqual(census.round_half_up(-0.6), -1)

        with tempfile.TemporaryDirectory() as tmp:
            scene_dir, tracks_path = _write_scene(Path(tmp), return_offset=0.1)
            payload = json.loads(tracks_path.read_text(encoding="utf-8"))
            scene = census.load_temporal_scene(scene_dir)
            # in-band: rounds to 0 -> kept; out-of-band: rounds to -1 -> dropped
            payload["tracks"][0]["reports"][0].update({"x": -0.3, "y": 12.0})
            payload["tracks"][1]["reports"][0].update({"x": -0.6, "y": 12.0})
            index = census.index_tracks(payload, scene)
            camera_in = int(payload["tracks"][0]["camera_id"])
            camera_out = int(payload["tracks"][1]["camera_id"])
            self.assertEqual(index.pixels.get((0, camera_in, 0)), (0, 12))
            self.assertNotIn((0, camera_out, 0), index.pixels)


class PooledCensusTest(unittest.TestCase):
    def test_two_sequences_pool_counts_and_coverage(self):
        # Two copies of the same fixture (distinct sequence names): every
        # event count doubles; the coverage FRACTION is invariant; the gate
        # binds to the pooled values (prereg: "over the dev subset").
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            scene_a, tracks_a = _write_scene(Path(tmp_a), return_offset=0.1)
            scene_b, tracks_b = _write_scene(Path(tmp_b), return_offset=0.1)
            renamed = scene_b.parent / "scene_b"
            scene_b.rename(renamed)
            pooled = census.run_census(
                [(scene_a, tracks_a), (renamed, tracks_b)],
                REPO_ROOT / "configs" / "elgs" / "prereg_m1_census_v1.json",
                apply_gate=True,
            )
            stats = pooled["statistics"]
            self.assertEqual(stats["true_absence_candidate_count"], 2)
            self.assertEqual(stats["same_object_return_count"], 2)
            self.assertEqual(stats["occlusion_opportunity_upper_bound"], 2)
            self.assertAlmostEqual(stats["track_coverage_upper_bound"], 71 / 191, places=12)
            self.assertEqual(pooled["coverage_tallies"]["components_total"], 2 * 191)
            self.assertEqual(sorted(pooled["per_sequence"]), ["scene", "scene_b"])
            self.assertFalse(pooled["gate"]["pass"])

    def test_duplicate_sequence_names_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, tracks = _write_scene(Path(tmp), return_offset=0.1)
            with self.assertRaises(ContractError):
                census.run_census(
                    [(scene, tracks), (scene, tracks)],
                    REPO_ROOT / "configs" / "elgs" / "prereg_m1_census_v1.json",
                    apply_gate=False,
                )


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
