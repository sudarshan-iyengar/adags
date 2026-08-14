"""CPU tests for scripts/build_absence_diagnostic.py (prereg revision 5).

Run with:
    C:/Users/sucar/venvs/elgs-cpu/Scripts/python.exe -m unittest tests.test_absence_diagnostic

Oracle style (matching tests/test_elgs_m1_census.py): a fully hand-computed
synthetic scene -- a look-at rig, engineered 64x64 masks, a hand-built tracks
artifact and a hand-built census record -- such that the expected class of
every window is derivable on paper.

FIXTURE A (used by most tests): 4 camera slots, cam00 held out by the mod-4
rule, training = {1, 2, 3} at 90 / 180 / 270 degrees, radius 4, focal 80,
64x64 images, principal point 31.5.

  - the world origin projects to the CONTINUOUS pixel (31.5, 31.5) in every
    training camera at depth z = 4, so the round-half-up anchor pixel is
    (32, 32);
  - r_site_census is fixed at 0.25 in the hand-built census, so
    tol_c = r_site * fl_x / z = 0.25 * 80 / 4 = 5.0 px exactly, and the S2
    ladder gives 1.25 px (0.25x), 5.0 px (1x) and 10.0 px (2x);
  - the three pairwise optical-axis separations are 90, 90 and 180 degrees;
    N = 3 pairs, rank = ceil(0.10 * 3) = 1, so the frozen angular floor is
    90 degrees and EVERY training pair clears it.

FIXTURE B (angular-floor test only): 8 camera slots, cam00/cam04 held out,
training = {1, 2, 3, 5, 6, 7} at 0, 0.5, 45, 90, 180, 270 degrees. N = 15
pairs, rank = ceil(1.5) = 2, and the sorted separations begin 0.5, 44.5, ...
so the frozen floor is 44.5 degrees: the (0, 0.5) pair is BELOW it while the
(0, 45) pair clears it.
"""

from __future__ import annotations

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
from elgs.tracks_schema import TRACKS_SCHEMA  # noqa: E402
from scripts import build_absence_diagnostic as diag  # noqa: E402
from scripts import build_m1_census as census  # noqa: E402

PREREG = REPO_ROOT / "configs" / "elgs" / "prereg_m1_absence_diagnostic_v1.json"

IMAGE_SIZE = 64
FOCAL = 80.0
CAMERA_RADIUS = 4.0
N_FRAMES = 24
R_SITE = 0.25  # => tol_c = 5.0 px exactly in every training camera
TOL = R_SITE * FOCAL / CAMERA_RADIUS
ANCHOR_COL, ANCHOR_ROW = 32, 32  # round_half_up(31.5)

WINDOW_FIRST, WINDOW_LAST = 10, 19
WINDOW_FRAMES = tuple(range(WINDOW_FIRST, WINDOW_LAST + 1))
LTP_FRAME = 9

RIG_A = (0.0, 90.0, 180.0, 270.0)  # cam00 held out; training {1,2,3}
RIG_B = (0.0, 0.0, 0.5, 45.0, 0.0, 90.0, 180.0, 270.0)  # cam00/cam04 held out

# ---------------------------------------------------------------------------
# Mask primitives
# ---------------------------------------------------------------------------

FAR_BLOB = (slice(4, 16), slice(4, 16))  # 144 px, nearest pixel (15,15): ~23 px away
ANCHOR_BLOB = (slice(26, 38), slice(26, 38))  # 144 px, contains (32,32)
NEAR_BLOB = (slice(34, 46), slice(26, 38))  # 144 px, nearest (34,32): 2.5 px from anchor
TINY_BLOB = (slice(30, 35), slice(30, 35))  # 25 px, contains (31,31): 0.707 px from anchor
# NOTE: TINY_BLOB lies INSIDE ANCHOR_BLOB, so drawing both merges them into a
# single 144 px component. Use OFFSET_TINY_BLOB when a separate sub-threshold
# component must coexist with an eligible anchor component.
OFFSET_TINY_BLOB = (slice(48, 53), slice(48, 53))  # 25 px, disjoint from FAR/ANCHOR/NEAR


def blank() -> np.ndarray:
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)


def with_blobs(*regions) -> np.ndarray:
    mask = blank()
    for region in regions:
        mask[region] = 255
    return mask


# ---------------------------------------------------------------------------
# Scene / artifact construction
# ---------------------------------------------------------------------------


def _look_at_c2w(position: np.ndarray) -> np.ndarray:
    z_axis = position / np.linalg.norm(position)
    up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x_axis, y_axis, z_axis, position
    return c2w


def write_scene(
    root: Path,
    mask_fn,
    *,
    rig=RIG_A,
    n_frames: int = N_FRAMES,
    bad_mask: tuple[int, int] | None = None,
) -> Path:
    """Write a converted-scene fixture. ``mask_fn(camera_id, frame) -> uint8``.

    ``bad_mask`` = ``(camera_id, frame)`` writes that mask at half size, which
    the substrate check must reject.
    """

    from PIL import Image

    scene = root / "scene"
    frames = []
    for cam_index, degrees in enumerate(rig):
        angle = math.radians(degrees)
        position = CAMERA_RADIUS * np.array([math.cos(angle), math.sin(angle), 0.0])
        c2w = _look_at_c2w(position)
        for frame_index in range(n_frames):
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

    training = [i for i in range(len(rig)) if i % 4 != 0]
    for cam_index in training:
        mask_dir = scene / "masks" / f"cam{cam_index:02d}"
        mask_dir.mkdir(parents=True)
        image_dir = scene / "undist" / f"cam{cam_index:02d}"
        image_dir.mkdir(parents=True)
        for frame_index in range(n_frames):
            mask = mask_fn(cam_index, frame_index)
            if bad_mask == (cam_index, frame_index):
                mask = mask[: IMAGE_SIZE // 2, : IMAGE_SIZE // 2]
            Image.fromarray(mask).save(mask_dir / f"{frame_index:08d}.png")
        # the substrate check decodes ONE undistorted frame per camera, at the
        # first realized frame index
        Image.fromarray(blank()).save(image_dir / "00000000.png")

    (scene / "diva360_conversion_provenance.json").write_text(
        json.dumps({"archive_selection": {"train": {"archive_path": "frames_1.tar.gz"}}}),
        encoding="utf-8",
    )
    return scene


def write_tracks(
    root: Path,
    report_fn,
    *,
    rig=RIG_A,
    n_frames: int = N_FRAMES,
    seed_id: int = 0,
    consensus_point=(0.0, 0.0, 0.0),
) -> Path:
    """``report_fn(camera_id, frame)`` returns a report dict, or None for a
    miss token, or the sentinel ``NO_TRACK_ROW`` to omit the camera entirely."""

    training = [i for i in range(len(rig)) if i % 4 != 0]
    tracks = []
    track_id = 0
    for camera_id in training:
        if report_fn(camera_id, 0) is NO_TRACK_ROW:
            continue
        reports = []
        for frame in range(n_frames):
            report = report_fn(camera_id, frame)
            if report is None:
                reports.append({"frame": float(frame), "is_miss": True})
            else:
                reports.append({"frame": float(frame), "is_miss": False, **report})
        tracks.append(
            {"track_id": track_id, "seed_id": seed_id, "camera_id": camera_id, "reports": reports}
        )
        track_id += 1
    consensus = [
        {
            "frame": float(frame),
            "point": list(consensus_point),
            "n_cam": len(training),
            "reproj_rms": 0.01,
        }
        for frame in range(n_frames)
    ]
    payload = {
        "schema_version": TRACKS_SCHEMA,
        "seeds": [{"seed_id": seed_id, "point": list(consensus_point), "n_cam": len(training)}],
        "tracks": tracks,
        "consensus": {str(seed_id): consensus},
        "diagnostics": {
            "fb_rms_px": {f"{seed_id}:{c}": 0.5 for c in training},
            "reproj_rms_px": {str(seed_id): 0.25},
            "r_u_mapping": "test fixture",
        },
        "window": {"frame_indices": list(range(n_frames)), "query_frame": 0, "fps": 120.0},
        "manifest": {
            "training_cameras": training,
            "held_out_cameras": [i for i in range(len(rig)) if i % 4 == 0],
            "tracker_identity": "hand-built-absence-diagnostic-fixture",
            "source_data_sha256": "0" * 64,
        },
    }
    path = root / "tracks.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


NO_TRACK_ROW = object()


def write_census(
    root: Path,
    scene_dir: Path,
    *,
    containing_cameras,
    n_frames_record: int | None = None,
    bridged_interruptions: int = 0,
    first_frame: int = WINDOW_FIRST,
    last_frame: int = WINDOW_LAST,
    ltp=(0.0, 0.0, 0.0),
    ltp_frame: int = LTP_FRAME,
    seed_id: int = 0,
    angular_floor: float | None = None,
    r_site: float = R_SITE,
) -> Path:
    record = {
        "seed_id": seed_id,
        "containing_cameras": list(containing_cameras),
        "first_frame": first_frame,
        "last_frame": last_frame,
        "n_frames": (
            last_frame - first_frame + 1 if n_frames_record is None else n_frames_record
        ),
        "bridged_interruptions": bridged_interruptions,
        "ltp_frame": ltp_frame,
        "ltp": list(ltp),
        "return": "no_terminating_reappearance",
    }
    block = {
        "statistics": {"track_coverage_upper_bound": 0.8},
        "coverage_tallies": {"components_total": 100, "components_covered": 80},
        "constants": {"r_site_census": r_site, "rig_radius": r_site / 0.05},
        "window": {"first_frame": 0, "last_frame": N_FRAMES - 1, "n_frames": N_FRAMES},
        "records": {"true_absence_candidates": [record], "occlusion_events": []},
    }
    if angular_floor is not None:
        block["angular_floor_rad"] = angular_floor
    payload = {
        "schema_version": "elgs-m1-a0-census-v1",
        "cell": "M1-A0",
        "per_sequence": {scene_dir.name: block},
    }
    path = root / "census.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Standard report patterns
# ---------------------------------------------------------------------------


def reports_miss_in_window(camera_id: int, frame: int):
    """ASSOCIATED before the window (on the anchor blob), miss inside it."""

    if frame < WINDOW_FIRST:
        return {"v": 0.9, "x": float(ANCHOR_COL), "y": float(ANCHOR_ROW)}
    return None


def reports_all_miss(camera_id: int, frame: int):
    return None


class Harness(unittest.TestCase):
    """Builds a one-window fixture and exposes the evaluated window."""

    def build(
        self,
        mask_fn,
        report_fn=reports_miss_in_window,
        *,
        rig=RIG_A,
        containing=None,
        bad_mask=None,
        with_angular_floor=True,
        **census_kwargs,
    ):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        scene_dir = write_scene(root, mask_fn, rig=rig, bad_mask=bad_mask)
        tracks_path = write_tracks(root, report_fn, rig=rig)
        training = [i for i in range(len(rig)) if i % 4 != 0]
        floor = None
        if with_angular_floor:
            from scripts.build_elgs_tracks import load_temporal_scene

            floor = census.angular_separation_floor(load_temporal_scene(scene_dir))
        census_path = write_census(
            root,
            scene_dir,
            containing_cameras=training if containing is None else containing,
            angular_floor=floor,
            **census_kwargs,
        )
        return scene_dir, tracks_path, census_path

    def evaluate(self, *args, **kwargs):
        scene_dir, tracks_path, census_path = self.build(*args, **kwargs)
        prereg = diag.load_prereg(PREREG)
        result = diag.evaluate_sequence(
            sequence_name=scene_dir.name,
            scene_dir=scene_dir,
            tracks_path=tracks_path,
            census_path=census_path,
            prereg=prereg,
        )
        return result["windows"][0], prereg

    def run_full(self, *args, **kwargs):
        scene_dir, tracks_path, census_path = self.build(*args, **kwargs)
        return diag.run_diagnostic(
            [(scene_dir.name, scene_dir, tracks_path, census_path)], PREREG, panel="primary"
        )

    def classify(self, window, prereg, **overrides):
        reading = diag.primary_reading(prereg)
        if overrides:
            reading = type(reading)(**{**reading.__dict__, **overrides})
        return diag.classify(window, reading, prereg)


# ---------------------------------------------------------------------------
# Step 1: the C1 split
# ---------------------------------------------------------------------------


class Step1Tests(Harness):
    def test_c1a_no_foreground_anywhere_near_the_anchor(self):
        # Far blob only during the window (nearest pixel 23.3 px away, well
        # beyond 2x tol = 10); anchor blob at ltp_frame so the anchor sat on an
        # eligible component in 3/3 of S.
        def masks(camera_id, frame):
            if frame == LTP_FRAME:
                return with_blobs(FAR_BLOB, ANCHOR_BLOB)
            return with_blobs(FAR_BLOB)

        window, prereg = self.evaluate(masks)
        verdict = self.classify(window, prereg)
        self.assertEqual(verdict["class"], diag.CLASS_C1A)
        self.assertFalse(verdict["t2b_any"])
        self.assertEqual(verdict["ltp_on_eligible_component"], 1.0)

    def test_c1b_subthreshold_foreground_only(self):
        # A 25 px blob touching the anchor: ineligible at the primary 64 px
        # floor (so T2 stays False) but T2b fires.
        def masks(camera_id, frame):
            if frame == LTP_FRAME:
                return with_blobs(FAR_BLOB, ANCHOR_BLOB)
            return with_blobs(FAR_BLOB, TINY_BLOB)

        window, prereg = self.evaluate(masks)
        verdict = self.classify(window, prereg)
        self.assertEqual(verdict["class"], diag.CLASS_C1B)
        self.assertTrue(verdict["t2b_any"])

    def test_c1c_anchor_unsupported(self):
        # Nothing near the anchor at any time, and at ltp_frame the anchor
        # pixel is background in every camera -> ltp_on_eligible = 0 < 0.5.
        def masks(camera_id, frame):
            return with_blobs(FAR_BLOB)

        def reports(camera_id, frame):
            # associate on the FAR blob before the window so D2 still has an
            # entry; the anchor itself is never on foreground
            if frame < WINDOW_FIRST:
                return {"v": 0.9, "x": 10.0, "y": 10.0}
            return None

        window, prereg = self.evaluate(masks, reports)
        verdict = self.classify(window, prereg)
        self.assertEqual(verdict["class"], diag.CLASS_C1C)
        self.assertFalse(verdict["t2b_any"])
        self.assertEqual(verdict["ltp_on_eligible_component"], 0.0)

    def test_c1a_becomes_c1b_only_via_subthreshold_foreground(self):
        # Same geometry, one bit different: the discriminator is T2b alone.
        def base(camera_id, frame):
            return with_blobs(FAR_BLOB, ANCHOR_BLOB) if frame == LTP_FRAME else with_blobs(FAR_BLOB)

        def tiny(camera_id, frame):
            if frame == LTP_FRAME:
                return with_blobs(FAR_BLOB, ANCHOR_BLOB)
            return with_blobs(FAR_BLOB, TINY_BLOB)

        a, prereg = self.evaluate(base)
        b, _ = self.evaluate(tiny)
        self.assertEqual(self.classify(a, prereg)["class"], diag.CLASS_C1A)
        self.assertEqual(self.classify(b, prereg)["class"], diag.CLASS_C1B)


# ---------------------------------------------------------------------------
# Step 2: C2 / C3 / the multi-view tightening
# ---------------------------------------------------------------------------


class Step2Tests(Harness):
    @staticmethod
    def _sustained_masks(camera_id, frame):
        return with_blobs(FAR_BLOB, ANCHOR_BLOB)

    def test_c2_sustained_multiview_occupancy_with_miss_tokens(self):
        window, prereg = self.evaluate(self._sustained_masks, reports_miss_in_window)
        verdict = self.classify(window, prereg)
        self.assertEqual(verdict["class"], diag.CLASS_C2)
        self.assertEqual(verdict["m_count"], len(WINDOW_FRAMES))
        self.assertEqual(verdict["c_star_loss_fraction"], 1.0)
        # ties broken by lowest camera id
        self.assertEqual(verdict["c_star"], 1)

    def test_c3_low_visibility_reports(self):
        def reports(camera_id, frame):
            if frame < WINDOW_FIRST:
                return {"v": 0.9, "x": float(ANCHOR_COL), "y": float(ANCHOR_ROW)}
            return {"v": 0.3, "x": float(ANCHOR_COL), "y": float(ANCHOR_ROW)}

        window, prereg = self.evaluate(self._sustained_masks, reports)
        verdict = self.classify(window, prereg)
        self.assertEqual(verdict["class"], diag.CLASS_C3)
        self.assertEqual(verdict["c_star_loss_fraction"], 0.0)
        self.assertEqual(
            verdict["c_star_status_counts"]["LOW_VISIBILITY"], len(WINDOW_FRAMES)
        )

    def test_c3_off_component_reports(self):
        # v = 0.9, in-domain, landing on background (2, 2)
        def reports(camera_id, frame):
            if frame < WINDOW_FIRST:
                return {"v": 0.9, "x": float(ANCHOR_COL), "y": float(ANCHOR_ROW)}
            return {"v": 0.9, "x": 2.0, "y": 2.0}

        window, prereg = self.evaluate(self._sustained_masks, reports)
        verdict = self.classify(window, prereg)
        self.assertEqual(verdict["class"], diag.CLASS_C3)
        self.assertEqual(
            verdict["c_star_status_counts"]["OFF_COMPONENT"], len(WINDOW_FRAMES)
        )

    def test_c5_single_camera_occupancy_is_not_multiview_confirmed(self):
        # Only cam01 covers the anchor. This is the revision-2 tightening.
        def masks(camera_id, frame):
            if camera_id == 1:
                return with_blobs(FAR_BLOB, ANCHOR_BLOB)
            return with_blobs(FAR_BLOB)

        window, prereg = self.evaluate(masks)
        verdict = self.classify(window, prereg)
        self.assertEqual(verdict["class"], diag.CLASS_C5_NS)
        self.assertEqual(verdict["m_count"], 0)
        # ... and under the S7 = 1 sensitivity reading it would have been C2
        relaxed = self.classify(window, prereg, min_cameras=1)
        self.assertEqual(relaxed["class"], diag.CLASS_C2)
        self.assertEqual(relaxed["m_count"], len(WINDOW_FRAMES))

    def test_c5_foreground_within_tolerance_but_never_on_the_anchor_pixel(self):
        # NEAR_BLOB's nearest pixel is 2.5 px from the continuous projection
        # (inside tol = 5) but row 32 is not in the blob, so T1 is False.
        def masks(camera_id, frame):
            return with_blobs(FAR_BLOB, NEAR_BLOB)

        window, prereg = self.evaluate(masks)
        verdict = self.classify(window, prereg)
        self.assertEqual(verdict["class"], diag.CLASS_C5_NS)
        level = prereg.primary_level_index
        radius = diag.RADIUS_INDEX["1x"]
        self.assertTrue(window.t2[radius, level].any())  # step 1 did not fire
        self.assertFalse(window.t1[level].any())  # strict occupancy never held

    def test_never_queried_cameras_cannot_force_c2(self):
        # cam01 and cam02 (90 degrees apart) cover the anchor but have NO track
        # row; only cam03, which does not cover it, was queried.
        def masks(camera_id, frame):
            if camera_id in (1, 2):
                return with_blobs(FAR_BLOB, ANCHOR_BLOB)
            return with_blobs(FAR_BLOB)

        def reports(camera_id, frame):
            if camera_id in (1, 2):
                return NO_TRACK_ROW
            return reports_miss_in_window(camera_id, frame)

        window, prereg = self.evaluate(masks, reports)
        self.assertEqual(sorted(window.queried_S), [3])
        verdict = self.classify(window, prereg)
        self.assertEqual(verdict["class"], diag.CLASS_C5_NI)
        self.assertTrue(verdict["structurally_silent"])
        # the same geometry WITH track rows is C2 -- the only difference is
        # whether the tracker was ever asked
        window2, prereg2 = self.evaluate(masks, reports_miss_in_window)
        self.assertEqual(self.classify(window2, prereg2)["class"], diag.CLASS_C2)

    def test_never_queried_status_is_recorded_separately(self):
        def masks(camera_id, frame):
            return with_blobs(FAR_BLOB, ANCHOR_BLOB)

        def reports(camera_id, frame):
            if camera_id == 3:
                return NO_TRACK_ROW
            return reports_miss_in_window(camera_id, frame)

        window, prereg = self.evaluate(masks, reports)
        level = prereg.primary_level_index
        ci = window.camera_index(3)
        statuses = window.status[level, ci][window.w_mask]
        self.assertTrue((statuses == diag.ST_NEVER_QUERIED).all())
        self.assertNotIn(3, window.queried_S)


class AngularFloorTests(Harness):
    """FIXTURE B: the frozen floor is 44.5 degrees."""

    def _floor(self, scene_dir):
        from scripts.build_elgs_tracks import load_temporal_scene

        return census.angular_separation_floor(load_temporal_scene(scene_dir))

    def test_floor_is_the_hand_derived_44_point_5_degrees(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        scene_dir = write_scene(
            Path(tmp.name), lambda c, f: with_blobs(FAR_BLOB), rig=RIG_B
        )
        self.assertAlmostEqual(math.degrees(self._floor(scene_dir)), 44.5, places=6)

    def test_two_close_cameras_do_not_multiview_confirm(self):
        # cam01 (0 deg) and cam02 (0.5 deg) both satisfy T1; their separation
        # is 0.5 deg < the 44.5 deg floor, so no frame is multi-view-confirmed.
        def masks(camera_id, frame):
            if camera_id in (1, 2):
                return with_blobs(FAR_BLOB, ANCHOR_BLOB)
            return with_blobs(FAR_BLOB)

        window, prereg = self.evaluate(masks, rig=RIG_B)
        level = prereg.primary_level_index
        occupying = [
            window.cams[ci]
            for ci in range(len(window.cams))
            if window.t1[level, ci].any()
        ]
        self.assertEqual(sorted(occupying), [1, 2])  # |A_t| = 2, but unseparated
        verdict = self.classify(window, prereg)
        self.assertEqual(verdict["class"], diag.CLASS_C5_NS)
        self.assertEqual(verdict["m_count"], 0)

    def test_two_separated_cameras_do_multiview_confirm(self):
        # cam01 (0 deg) and cam03 (45 deg): 45 >= 44.5, so the pair qualifies.
        def masks(camera_id, frame):
            if camera_id in (1, 3):
                return with_blobs(FAR_BLOB, ANCHOR_BLOB)
            return with_blobs(FAR_BLOB)

        window, prereg = self.evaluate(masks, rig=RIG_B)
        verdict = self.classify(window, prereg)
        self.assertEqual(verdict["class"], diag.CLASS_C2)
        self.assertEqual(verdict["m_count"], len(WINDOW_FRAMES))


# ---------------------------------------------------------------------------
# W(record), bridged frames and the R6 operative guard
# ---------------------------------------------------------------------------


def _bridged_reports(camera_id, frame):
    """ASSOCIATED before the window AND at frames 14-15 (one bridged run)."""

    if frame < WINDOW_FIRST or frame in (14, 15):
        return {"v": 0.9, "x": float(ANCHOR_COL), "y": float(ANCHOR_ROW)}
    return None


class BridgedFrameTests(Harness):
    @staticmethod
    def _masks(camera_id, frame):
        return with_blobs(FAR_BLOB, ANCHOR_BLOB)

    def test_bridged_window_does_not_abort_and_bridged_frames_leave_the_denominator(self):
        # Hand count: [10, 19] is 10 frames; frames 14 and 15 carry an
        # association in every camera of S, so |W| = 8 and one maximal bridged
        # run exists.
        window, prereg = self.evaluate(
            self._masks, _bridged_reports, bridged_interruptions=1
        )
        self.assertEqual(int(window.w_mask.sum()), 8)
        self.assertEqual(len(window.interval) - int(window.w_mask.sum()), 2)
        w_frames = [f for f, keep in zip(window.interval, window.w_mask) if keep]
        self.assertEqual(w_frames, [10, 11, 12, 13, 16, 17, 18, 19])
        verdict = self.classify(window, prereg)
        self.assertEqual(verdict["w_frames"], 8)  # denominator excludes the bridge
        self.assertEqual(verdict["m_count"], 8)
        self.assertEqual(verdict["class"], diag.CLASS_C2)

    def test_r6_guard_rejects_a_wrong_bridged_interruption_count(self):
        with self.assertRaises(ContractError) as raised:
            self.evaluate(self._masks, _bridged_reports, bridged_interruptions=0)
        self.assertIn("operative_fail_closed_guard_R6", str(raised.exception))
        self.assertIn("maximal bridged runs", str(raised.exception))

    def test_r6_guard_rejects_a_wrong_n_frames(self):
        with self.assertRaises(ContractError) as raised:
            self.evaluate(
                self._masks,
                _bridged_reports,
                bridged_interruptions=1,
                n_frames_record=9,
            )
        self.assertIn("operative_fail_closed_guard_R6", str(raised.exception))

    def test_two_separate_bridged_runs_are_counted_as_two(self):
        def reports(camera_id, frame):
            if frame < WINDOW_FIRST or frame in (12, 16):
                return {"v": 0.9, "x": float(ANCHOR_COL), "y": float(ANCHOR_ROW)}
            return None

        window, _ = self.evaluate(self._masks, reports, bridged_interruptions=2)
        self.assertEqual(int(window.w_mask.sum()), 8)
        with self.assertRaises(ContractError):
            self.evaluate(self._masks, reports, bridged_interruptions=1)

    def test_maximal_runs_helper(self):
        self.assertEqual(diag._maximal_runs([]), 0)
        self.assertEqual(diag._maximal_runs([False, False]), 0)
        self.assertEqual(diag._maximal_runs([True, True, True]), 1)
        self.assertEqual(diag._maximal_runs([True, False, True]), 2)
        self.assertEqual(diag._maximal_runs([False, True, True, False, True]), 2)


class WInvarianceTests(Harness):
    """R4: W is computed once at the primary reading and never moves."""

    def test_w_is_identical_under_every_S5_level(self):
        def masks(camera_id, frame):
            return with_blobs(FAR_BLOB, ANCHOR_BLOB)

        window, prereg = self.evaluate(masks, _bridged_reports, bridged_interruptions=1)
        baseline = int(window.w_mask.sum())
        for level in prereg.s5_component_px:
            verdict = self.classify(window, prereg, component_px=level)
            self.assertEqual(
                verdict["w_frames"], baseline, f"S5 = {level} moved the W denominator"
            )

    def test_reading_induced_associated_is_excluded_from_both_sides(self):
        # The report lands on a 25 px component: OFF_COMPONENT at the primary
        # 64 px floor (so the frame stays in W) but ASSOCIATED at S5 = 16.
        # The anchor must ALSO sit on an eligible component (ANCHOR_BLOB) so
        # the window reaches step 2 rather than falling to C1b. That forces a
        # DISJOINT sub-threshold blob: TINY_BLOB lies inside ANCHOR_BLOB and
        # would merge with it into one eligible 144 px component.
        def masks(camera_id, frame):
            return with_blobs(FAR_BLOB, ANCHOR_BLOB, OFFSET_TINY_BLOB)

        def reports(camera_id, frame):
            if frame < WINDOW_FIRST:
                return {"v": 0.9, "x": float(ANCHOR_COL), "y": float(ANCHOR_ROW)}
            return {"v": 0.9, "x": 50.0, "y": 50.0}  # inside OFFSET_TINY_BLOB

        window, prereg = self.evaluate(masks, reports)
        self.assertEqual(int(window.w_mask.sum()), len(WINDOW_FRAMES))  # W is full
        primary = self.classify(window, prereg)
        self.assertEqual(primary["class"], diag.CLASS_C3)
        self.assertEqual(primary["reading_induced_associated"], 0)
        relaxed = self.classify(window, prereg, component_px=16)
        self.assertEqual(relaxed["w_frames"], len(WINDOW_FRAMES))  # unchanged
        self.assertEqual(relaxed["reading_induced_associated"], len(WINDOW_FRAMES))
        self.assertEqual(relaxed["c_star_denominator"], 0)
        # A6: an emptied denominator yields fraction 0.0, hence C3
        self.assertEqual(relaxed["class"], diag.CLASS_C3)


# ---------------------------------------------------------------------------
# S2 ladder, including the r4 0.25x level
# ---------------------------------------------------------------------------


class S2LadderTests(Harness):
    def test_quarter_tolerance_level_excludes_foreground_at_2_point_5_px(self):
        # NEAR_BLOB is 2.5 px from the continuous projection: inside 1x tol
        # (5.0) and 2x tol (10.0) but outside 0.25x tol (1.25).
        def masks(camera_id, frame):
            if frame == LTP_FRAME:
                return with_blobs(FAR_BLOB, ANCHOR_BLOB)
            return with_blobs(FAR_BLOB, NEAR_BLOB)

        window, prereg = self.evaluate(masks)
        level = prereg.primary_level_index
        self.assertFalse(window.t2[diag.RADIUS_INDEX["0.25x"], level].any())
        self.assertTrue(window.t2[diag.RADIUS_INDEX["1x"], level].any())
        self.assertTrue(window.t2[diag.RADIUS_INDEX["2x"], level].any())
        # primary reading: step 1 does not fire -> C5
        self.assertEqual(self.classify(window, prereg)["class"], diag.CLASS_C5_NS)
        # 0.25x reading: step 1 fires, and nothing sub-threshold is inside
        # 1.25 px either, so the split lands on C1a
        quarter = self.classify(window, prereg, step1=diag.OCC_T2_025, step2=diag.OCC_T1)
        self.assertEqual(quarter["class"], diag.CLASS_C1A)

    def test_t2b_radius_tracks_the_s2_level(self):
        # TINY_BLOB (25 px, sub-threshold) is 0.707 px from the projection, so
        # it is inside every radius on the ladder; NEAR_BLOB is not inside
        # 0.25x. Only the eligibility filter separates T2 from T2b.
        def masks(camera_id, frame):
            if frame == LTP_FRAME:
                return with_blobs(FAR_BLOB, ANCHOR_BLOB)
            return with_blobs(FAR_BLOB, TINY_BLOB)

        window, prereg = self.evaluate(masks)
        for key in ("0.25x", "1x", "2x"):
            self.assertTrue(
                window.t2b[diag.RADIUS_INDEX[key]].any(), f"T2b should fire at {key}"
            )
        quarter = self.classify(window, prereg, step1=diag.OCC_T2_025, step2=diag.OCC_T1)
        self.assertEqual(quarter["class"], diag.CLASS_C1B)

    def test_grid_has_144_cells_in_the_frozen_order(self):
        prereg = diag.load_prereg(PREREG)
        grid = diag.decision_grid(prereg)
        self.assertEqual(len(grid), 144)
        self.assertEqual(
            [(r.step1, r.step2) for r in grid[:1]], [(diag.OCC_T2_025, diag.OCC_T1)]
        )
        primary = diag.primary_reading(prereg)
        self.assertEqual((primary.step1, primary.step2), (diag.OCC_T2, diag.OCC_T1))
        self.assertIn(primary.label(), [r.label() for r in grid])


# ---------------------------------------------------------------------------
# tol_c geometry (the disc predicate)
# ---------------------------------------------------------------------------


class ToleranceGeometryTests(unittest.TestCase):
    """The inclusive '<= tol_c' comparison, exercised on the primitive."""

    def setUp(self):
        self.labels = np.zeros((64, 64), dtype=np.int32)
        self.labels[37, 32] = 1  # exactly 5.0 px below (32, 32)
        self.lut = np.array([False, True])

    def test_component_exactly_at_tol_counts(self):
        disc = diag.Disc(32.0, 32.0, 32, 32, 5.0, 64, 64)
        self.assertTrue(disc.hits(self.labels, self.lut))

    def test_component_beyond_tol_by_one_ulp_does_not_count(self):
        just_under = float(np.nextafter(5.0, 0.0))
        disc = diag.Disc(32.0, 32.0, 32, 32, just_under, 64, 64)
        self.assertFalse(disc.hits(self.labels, self.lut))

    def test_disc_is_measured_from_the_continuous_projection(self):
        # The foreground pixel is (row 37, col 32). From the CONTINUOUS point
        # (31.5, 32.0) it is hypot(5.0, 0.5) = 5.0249 px away, but from the
        # ROUNDED pixel (32, 32) it is exactly 5.0. A tolerance strictly
        # between the two therefore separates the correct predicate from a
        # distance-transform lookup at the rounded pixel, which would wrongly
        # admit it.
        self.assertAlmostEqual(math.hypot(37 - 32.0, 32 - 31.5), 5.024938, places=6)
        self.assertFalse(diag.Disc(31.5, 32.0, 32, 32, 5.01, 64, 64).hits(self.labels, self.lut))
        self.assertTrue(diag.Disc(31.5, 32.0, 32, 32, 5.03, 64, 64).hits(self.labels, self.lut))

    def test_lookup_none_selects_any_foreground(self):
        disc = diag.Disc(32.0, 32.0, 32, 32, 5.0, 64, 64)
        self.assertTrue(disc.hits(self.labels, None))
        ineligible = np.array([False, False])
        self.assertFalse(disc.hits(self.labels, ineligible))

    def test_search_box_is_a_superset_of_the_mandated_disc(self):
        # A17: radius ceil(tol) + 1 so a pixel within tol of the continuous
        # point can never fall outside the box.
        disc = diag.Disc(31.5, 31.5, 32, 32, 5.0, 64, 64)
        self.assertGreaterEqual(32 - disc.r0, int(math.ceil(5.0)) + 1 - 1)
        self.assertGreaterEqual(disc.r1 - 32, int(math.ceil(5.0)))


# ---------------------------------------------------------------------------
# D2 lineage
# ---------------------------------------------------------------------------


class LineageTests(Harness):
    def _verdict(self, window, prereg):
        return window.d2["verdicts"][diag._d2_key_name(diag._d2_keys(prereg)[0])]

    def test_lineage_survives(self):
        def masks(camera_id, frame):
            return with_blobs(FAR_BLOB, ANCHOR_BLOB)

        window, prereg = self.evaluate(masks, reports_miss_in_window)
        self.assertEqual(self._verdict(window, prereg), diag.D2_SURVIVES)
        per_camera = window.d2["cameras"][diag._d2_key_name(diag._d2_keys(prereg)[0])]
        self.assertEqual(per_camera["1"]["entry"], WINDOW_FIRST - 1)
        self.assertTrue(per_camera["1"]["entry_corroborated"])
        self.assertEqual(per_camera["1"]["survived_fraction"], 1.0)

    def test_lineage_ends(self):
        # The anchor component is replaced by a disjoint one from frame 12,
        # so the IoU against X_t collapses to 0 and the lineage ends.
        def masks(camera_id, frame):
            if frame >= 12:
                return with_blobs(FAR_BLOB, (slice(50, 62), slice(50, 62)))
            return with_blobs(FAR_BLOB, ANCHOR_BLOB)

        window, prereg = self.evaluate(masks, reports_miss_in_window)
        self.assertEqual(self._verdict(window, prereg), diag.D2_ENDS)
        per_camera = window.d2["cameras"][diag._d2_key_name(diag._d2_keys(prereg)[0])]
        self.assertEqual(per_camera["1"]["end_frame"], 12)

    def test_lineage_entry_unavailable(self):
        def masks(camera_id, frame):
            return with_blobs(FAR_BLOB, ANCHOR_BLOB)

        window, prereg = self.evaluate(masks, reports_all_miss)
        self.assertEqual(self._verdict(window, prereg), diag.D2_ENTRY_UNAVAILABLE)

    def test_d2_never_alters_the_assigned_class(self):
        # Identical masks; the ONLY difference is whether the tracker ever
        # associated BEFORE the window, which moves the D2 verdict from
        # SURVIVES to ENTRY_UNAVAILABLE and must leave the class at C2.
        def masks(camera_id, frame):
            return with_blobs(FAR_BLOB, ANCHOR_BLOB)

        with_entry, prereg = self.evaluate(masks, reports_miss_in_window)
        without_entry, _ = self.evaluate(masks, reports_all_miss)
        self.assertEqual(self._verdict(with_entry, prereg), diag.D2_SURVIVES)
        self.assertEqual(self._verdict(without_entry, prereg), diag.D2_ENTRY_UNAVAILABLE)
        self.assertEqual(
            self.classify(with_entry, prereg)["class"],
            self.classify(without_entry, prereg)["class"],
        )
        self.assertEqual(self.classify(with_entry, prereg)["class"], diag.CLASS_C2)

    def test_classify_signature_cannot_see_d2(self):
        # Structural guarantee, not a behavioural one: the decision list is a
        # function of (window evidence, reading, prereg) only.
        import inspect

        parameters = list(inspect.signature(diag.classify).parameters)
        self.assertEqual(parameters, ["window", "reading", "prereg"])
        source = inspect.getsource(diag.classify)
        self.assertNotIn("d2", source)


# ---------------------------------------------------------------------------
# Substrate, tautology classes, census cross-checks
# ---------------------------------------------------------------------------


class SubstrateAndFailClosedTests(Harness):
    def test_substrate_mismatch_fails_the_run_closed(self):
        def masks(camera_id, frame):
            return with_blobs(FAR_BLOB, ANCHOR_BLOB)

        with self.assertRaises(ContractError) as raised:
            self.evaluate(masks, bad_mask=(2, 0))
        self.assertIn("substrate_check FAILED CLOSED", str(raised.exception))

    def test_c4_projection_failure_fails_the_run_closed(self):
        # An anchor 3.9 units off-axis leaves cam01's frustum (half-FOV
        # atan(32/80) = 21.8 deg, required 44 deg), yet S asserts containment.
        def masks(camera_id, frame):
            return with_blobs(FAR_BLOB)

        scene_dir, tracks_path, census_path = self.build(
            masks, reports_all_miss, ltp=(3.9, 0.0, 0.0), with_angular_floor=True
        )
        with self.assertRaises(ContractError) as raised:
            diag.run_diagnostic(
                [(scene_dir.name, scene_dir, tracks_path, census_path)], PREREG
            )
        message = str(raised.exception)
        self.assertIn("C4_SUBSTRATE_OR_PROJECTION", message)
        self.assertIn("TAUTOLOGY CHECK", message)

    def test_c4_and_c6_are_empty_on_a_healthy_run(self):
        def masks(camera_id, frame):
            return with_blobs(FAR_BLOB, ANCHOR_BLOB)

        result = self.run_full(masks)
        self.assertEqual(result["pooled"]["classes"][diag.CLASS_C4], 0)
        self.assertEqual(result["pooled"]["classes"][diag.CLASS_C6], 0)

    def test_tampered_angular_floor_in_the_census_fails_closed(self):
        def masks(camera_id, frame):
            return with_blobs(FAR_BLOB, ANCHOR_BLOB)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        scene_dir = write_scene(root, masks)
        tracks_path = write_tracks(root, reports_miss_in_window)
        census_path = write_census(
            root, scene_dir, containing_cameras=[1, 2, 3], angular_floor=0.123456
        )
        with self.assertRaises(ContractError) as raised:
            diag.evaluate_sequence(
                sequence_name=scene_dir.name,
                scene_dir=scene_dir,
                tracks_path=tracks_path,
                census_path=census_path,
                prereg=diag.load_prereg(PREREG),
            )
        self.assertIn("angular floor", str(raised.exception))

    def test_containing_camera_outside_the_scene_fails_closed(self):
        def masks(camera_id, frame):
            return with_blobs(FAR_BLOB, ANCHOR_BLOB)

        with self.assertRaises(ContractError) as raised:
            self.evaluate(masks, containing=[1, 2, 3, 99])
        self.assertIn("not a training camera", str(raised.exception))


# ---------------------------------------------------------------------------
# Prereg guards
# ---------------------------------------------------------------------------


class PreregGuardTests(unittest.TestCase):
    def _mutated(self, mutate):
        payload = json.loads(PREREG.read_text(encoding="utf-8"))
        mutate(payload)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "prereg.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_current_prereg_loads(self):
        prereg = diag.load_prereg(PREREG)
        self.assertEqual(prereg.component_min_px, census.MIN_COMPONENT_PX)
        self.assertEqual(prereg.mask_binarize_threshold, census.MASK_THRESHOLD)
        self.assertEqual(prereg.visibility_threshold, census.VIS_THRESHOLD)
        self.assertEqual(prereg.robustness_supermajority, 0.6667)

    def test_wrong_revision_fails_closed(self):
        path = self._mutated(lambda p: p.__setitem__("revision", 4))
        with self.assertRaises(ContractError) as raised:
            diag.load_prereg(path)
        self.assertIn("revision", str(raised.exception))

    def test_missing_key_fails_closed(self):
        path = self._mutated(lambda p: p.pop("classification"))
        with self.assertRaises(ContractError):
            diag.load_prereg(path)

    def test_missing_frozen_constant_fails_closed(self):
        path = self._mutated(
            lambda p: p["frozen_constants"].pop("ltp_on_eligible_component_min")
        )
        with self.assertRaises(ContractError):
            diag.load_prereg(path)

    def test_tampered_component_floor_fails_closed(self):
        # The reused census primitive carries 64 px internally and cannot
        # honour a different value, so a divergence must stop the run.
        path = self._mutated(lambda p: p["frozen_constants"].__setitem__("component_min_px", 32))
        with self.assertRaises(ContractError) as raised:
            diag.load_prereg(path)
        self.assertIn("MIN_COMPONENT_PX", str(raised.exception))

    def test_tampered_visibility_threshold_fails_closed(self):
        path = self._mutated(
            lambda p: p["frozen_constants"].__setitem__("visibility_threshold", 0.9)
        )
        with self.assertRaises(ContractError):
            diag.load_prereg(path)

    def test_missing_r5_amendment_fails_closed(self):
        path = self._mutated(lambda p: p.pop("amendment_2026_08_14_r5_signoff"))
        with self.assertRaises(ContractError):
            diag.load_prereg(path)

    def test_missing_robustness_supermajority_fails_closed(self):
        path = self._mutated(
            lambda p: p["amendment_2026_08_14_r5_signoff"][
                "R15_OWNER_robustness_grade_replaces_unanimity"
            ].pop("robustness_supermajority")
        )
        with self.assertRaises(ContractError):
            diag.load_prereg(path)

    def test_renamed_class_fails_closed(self):
        def mutate(payload):
            steps = payload["classification"]["ordered_decision_list"]
            payload["classification"]["ordered_decision_list"] = [
                step.replace("C1a_GENUINE_ABSENCE_CORROBORATED", "C1a_RENAMED") for step in steps
            ]

        path = self._mutated(mutate)
        with self.assertRaises(ContractError) as raised:
            diag.load_prereg(path)
        self.assertIn("C1a_GENUINE_ABSENCE_CORROBORATED", str(raised.exception))

    def test_dropped_s2_level_fails_closed(self):
        path = self._mutated(
            lambda p: p["sensitivity_readings"].__setitem__(
                "S2_occupancy_tolerance", "(T2, T1) [primary], (T2, T2)"
            )
        )
        with self.assertRaises(ContractError):
            diag.load_prereg(path)

    def test_never_queried_exemption_must_be_present(self):
        path = self._mutated(
            lambda p: p["per_pair_tests"].__setitem__(
                "R_grouping_for_the_decision_list", "MISS_TOKEN and OUT_OF_DOMAIN form a group."
            )
        )
        with self.assertRaises(ContractError):
            diag.load_prereg(path)


# ---------------------------------------------------------------------------
# Commensurability: the census primitives are REUSED, not reimplemented
# ---------------------------------------------------------------------------


class PrimitiveReuseTests(unittest.TestCase):
    def test_primitives_are_the_census_objects_themselves(self):
        self.assertIs(diag.load_component_labels, census.load_component_labels)
        self.assertIs(diag.frustum_contains, census.frustum_contains)
        self.assertIs(diag.round_half_up, census.round_half_up)
        self.assertIs(diag.index_tracks, census.index_tracks)
        self.assertIs(diag.ReportIndex, census.ReportIndex)
        self.assertIs(diag.rig_radius, census.rig_radius)
        self.assertIs(diag.angular_separation_floor, census.angular_separation_floor)
        self.assertEqual(diag.MASK_THRESHOLD, census.MASK_THRESHOLD)
        self.assertEqual(diag.MIN_COMPONENT_PX, census.MIN_COMPONENT_PX)
        self.assertEqual(diag.VIS_THRESHOLD, census.VIS_THRESHOLD)

    def test_module_defines_no_shadowing_copies(self):
        import inspect

        source = inspect.getsource(diag)
        for name in (
            "def load_component_labels",
            "def frustum_contains",
            "def round_half_up",
            "def index_tracks",
            "def angular_separation_floor",
            "def rig_radius",
        ):
            self.assertNotIn(name, source, f"{name} is reimplemented instead of imported")

    def test_anchor_projection_agrees_with_frustum_contains(self):
        from scripts.build_elgs_tracks import load_temporal_scene

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        scene_dir = write_scene(Path(tmp.name), lambda c, f: with_blobs(FAR_BLOB))
        scene = load_temporal_scene(scene_dir)
        rng = np.random.default_rng(7)
        for _ in range(200):
            point = rng.uniform(-6.0, 6.0, size=3)
            for camera_id in scene.tracking_ids:
                camera = scene.cameras[camera_id]
                x, y, z = diag.anchor_projection(camera, point)
                derived = (
                    z > 0.0
                    and 0.0 <= x <= camera.width - 1.0
                    and 0.0 <= y <= camera.height - 1.0
                )
                self.assertEqual(bool(derived), census.frustum_contains(camera, point))

    def test_mask_frame_eligibility_matches_the_census_primitive(self):
        from PIL import Image

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "m.png"
        # FAR_BLOB (144 px) and TINY_BLOB (25 px) are DISJOINT, so S5 actually
        # discriminates. NOTE: TINY_BLOB lies inside ANCHOR_BLOB, so drawing
        # both would merge them into one 144 px component and the 16 px level
        # would discriminate nothing -- the original form of this test made
        # exactly that mistake.
        Image.fromarray(with_blobs(FAR_BLOB, TINY_BLOB)).save(path)
        frame = diag.MaskFrame(path, (16, 64, 256), 1)
        _, eligible = census.load_component_labels(path)
        self.assertEqual(frame.eligible[1], frozenset(int(v) for v in eligible))
        self.assertEqual(len(frame.eligible[0]), 2)  # 144 and 25 px, both >= 16
        self.assertEqual(len(frame.eligible[1]), 1)  # the 25 px blob drops out at 64
        self.assertEqual(len(frame.eligible[2]), 0)  # nothing reaches 256 px


# ---------------------------------------------------------------------------
# End-to-end output contract
# ---------------------------------------------------------------------------


class OutputContractTests(Harness):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        scene_dir = write_scene(root, lambda c, f: with_blobs(FAR_BLOB, ANCHOR_BLOB))
        tracks_path = write_tracks(root, reports_miss_in_window)
        from scripts.build_elgs_tracks import load_temporal_scene

        floor = census.angular_separation_floor(load_temporal_scene(scene_dir))
        census_path = write_census(
            root, scene_dir, containing_cameras=[1, 2, 3], angular_floor=floor
        )
        cls.result = diag.run_diagnostic(
            [(scene_dir.name, scene_dir, tracks_path, census_path)], PREREG, panel="primary"
        )
        cls.sequence = scene_dir.name

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_required_top_level_sections(self):
        for key in (
            "per_sequence",
            "pooled",
            "pooling_B7_R2",
            "windows",
            "sensitivity_table",
            "sensitivity_descriptive",
            "sensitivity_d2",
            "audit_sample_B8",
            "measurement_closure",
            "relationships",
            "evidence_grading",
            "inputs",
            "provenance",
            "substrate_check",
            "disclosed_readings",
        ):
            self.assertIn(key, self.result)

    def test_sensitivity_table_has_144_cells_with_reading_labels(self):
        table = self.result["sensitivity_table"]
        self.assertEqual(len(table), 144)
        for row in table:
            self.assertIn("reading_label", row)
            self.assertIn("counts", row)
        summary = self.result["sensitivity_summary"]
        self.assertEqual(summary["n_decision_relevant_readings"], 144)
        self.assertIn("robustness_grade", summary)
        self.assertIn("primary_status_under_strict_unanimity", summary)
        self.assertIn("strict_unanimity_differs_from_primary_status", summary)

    def test_measurement_closure_reports_grade_and_unanimity_counterfactual(self):
        closure = self.result["measurement_closure"]
        self.assertIn(
            closure["robustness_grade"],
            (diag.GRADE_UNANIMOUS, diag.GRADE_ROBUST, diag.GRADE_FRAGILE),
        )
        self.assertEqual(closure["binding_pooling"], diag.POOLING_B)
        self.assertIn("primary_status_under_strict_unanimity", closure)
        self.assertIn("primary_status_if_FRAGILE_dominates", closure)
        self.assertIn("strict_unanimity_differs_from_primary_status", closure)
        self.assertEqual(
            closure["conditions"]["pooling_disagreement_scope"],
            "status_2 and status_3 only (R14)",
        )

    def test_input_sha256s_and_prereg_identity_are_recorded(self):
        entry = self.result["inputs"][self.sequence]
        for key in ("transforms_train_sha256", "tracks_sha256", "census_sha256"):
            self.assertRegex(entry[key], r"^[0-9a-f]{64}$")
        self.assertEqual(
            entry["conversion_provenance"]["train_archive_path"], "frames_1.tar.gz"
        )
        self.assertRegex(self.result["provenance"]["prereg_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(self.result["provenance"]["prereg_revision"], 5)
        self.assertGreater(self.result["provenance"]["wall_seconds"], 0.0)

    def test_window_record_carries_the_required_fields(self):
        record = self.result["windows"][0]
        for key in (
            "sequence",
            "seed_id",
            "first_frame",
            "last_frame",
            "W_frames",
            "bridged_associated_frames",
            "n_applicable_cameras",
            "n_queried_cameras",
            "class",
            "terminal_class",
            "anchor_quality",
            "ltp_on_eligible_component",
            "tol_px",
            "tallies",
            "d2",
            "W_seconds",
            "n_frames_seconds",
            "W_stratum",
            "end_truncated",
        ):
            self.assertIn(key, record)
        self.assertEqual(record["n_applicable_cameras"], 3)
        self.assertEqual(record["anchor_quality"]["anchor_staleness"], WINDOW_FIRST - LTP_FRAME)
        self.assertEqual(record["anchor_quality"]["ltp_n_cam"], 3)
        self.assertIsNotNone(record["anchor_quality"]["seed_reproj_rms_px"])
        self.assertEqual(record["anchor_quality"]["fb_rms_px"]["1"], 0.5)

    def test_tolerance_outputs_match_the_hand_derived_tol(self):
        tolerance = self.result["per_sequence"][self.sequence]["tolerance"]
        self.assertAlmostEqual(tolerance["tol_px_median"], TOL, places=9)
        self.assertAlmostEqual(tolerance["tol_px_min"], TOL, places=9)
        self.assertAlmostEqual(tolerance["tol_px_max"], TOL, places=9)
        self.assertEqual(tolerance["median_eligible_component_area"], 144.0)
        self.assertGreater(tolerance["decoded_camera_frame_pairs"], 0)

    def test_audit_sample_is_emitted_and_deterministic(self):
        sample = self.result["audit_sample_B8"]
        self.assertEqual(sample["seed"], 20260814)
        self.assertEqual(sample["n_available"], 1)
        self.assertEqual(sample["n_selected"], 1)
        entry = sample["windows"][0]
        for key in (
            "sequence",
            "seed_id",
            "first_frame",
            "last_frame",
            "W_frames",
            "bridged_associated_frames",
            "class",
            "anchor_staleness",
            "containing_cameras",
            "ltp",
            "ltp_frame",
            "has_bridged_frames",
        ):
            self.assertIn(key, entry)
        self.assertEqual(entry["class"], diag.CLASS_C2)

    def test_pooling_reports_three_ways(self):
        pooling = self.result["pooling_B7_R2"]
        self.assertEqual(pooling["binding"], diag.POOLING_B)
        self.assertIn("a_pooled", pooling)
        self.assertIn("c_leave_scissor_out_pooled", pooling)
        # This fixture has ONE window, so no sequence clears the >= 10-window
        # floor and the BINDING pooling (b) is legitimately undefined. That is
        # the contract: (b) is None rather than silently falling back to (a).
        # The binding-pooling arithmetic itself is exercised directly in
        # BindingPoolingTests below, which does not need a full pipeline run.
        self.assertEqual(len(self.result["windows"]), 1)
        self.assertIsNone(
            pooling["b_unweighted_mean_over_sequences_with_at_least_10_windows"]
        )
        self.assertEqual(pooling["b_sequences"], [])

    def test_tautology_checks_are_recorded_not_raised(self):
        block = self.result["per_sequence"][self.sequence]
        self.assertTrue(block["tautology_checks_W"]["all_passed"])
        self.assertEqual(block["tautology_checks_W"]["failures"], [])

    def test_output_is_plain_strict_json(self):
        body = json.dumps(self.result, allow_nan=False, sort_keys=True, separators=(",", ":"))
        self.assertGreater(len(body), 1000)
        json.loads(body)


class AuditSampleDeterminismTests(Harness):
    def test_two_runs_produce_the_identical_sample(self):
        def masks(camera_id, frame):
            return with_blobs(FAR_BLOB, ANCHOR_BLOB)

        scene_dir, tracks_path, census_path = self.build(masks)
        first = diag.run_diagnostic(
            [(scene_dir.name, scene_dir, tracks_path, census_path)], PREREG
        )["audit_sample_B8"]["windows"]
        second = diag.run_diagnostic(
            [(scene_dir.name, scene_dir, tracks_path, census_path)], PREREG
        )["audit_sample_B8"]["windows"]
        self.assertEqual(first, second)

    def test_terciles_use_nearest_rank_with_ties_to_the_lower_tercile(self):
        # anchor_staleness is a small non-negative integer with heavy ties at
        # 0, which is exactly why nearest-rank is frozen.
        # n = 9, so the frozen rule takes the ceil(9/3) = 3rd and
        # ceil(18/3) = 6th SORTED values (1-indexed), i.e. 0 and 2.
        # An interpolated percentile would have given (0.0, 3.33) and put
        # different windows in different strata -- which is exactly why
        # nearest-rank is frozen.
        cuts = diag._terciles([0, 0, 0, 0, 1, 2, 3, 4, 5])
        self.assertEqual(cuts, (0.0, 2.0))
        self.assertEqual(diag._tercile_label(0, cuts), "staleness_T1")  # ties low
        self.assertEqual(diag._tercile_label(1, cuts), "staleness_T2")
        self.assertEqual(diag._tercile_label(2, cuts), "staleness_T2")  # ties low
        self.assertEqual(diag._tercile_label(3, cuts), "staleness_T3")


class RobustnessGradeTests(unittest.TestCase):
    def test_grade_thresholds(self):
        self.assertEqual(diag.robustness_grade(["X"] * 5, "X", 0.6667)[0], diag.GRADE_UNANIMOUS)
        grade, agreement = diag.robustness_grade(["X"] * 7 + ["Y"] * 3, "X", 0.6667)
        self.assertEqual(grade, diag.GRADE_ROBUST)
        self.assertAlmostEqual(agreement, 0.7)
        self.assertEqual(
            diag.robustness_grade(["X"] * 6 + ["Y"] * 4, "X", 0.6667)[0], diag.GRADE_FRAGILE
        )
        # agreement is measured against the PRIMARY reading's label
        self.assertEqual(
            diag.robustness_grade(["X"] * 9 + ["Y"], "Y", 0.6667)[0], diag.GRADE_FRAGILE
        )


class BindingPoolingTests(unittest.TestCase):
    """Pooling (b) BINDS every status predicate (prereg R2), so it is tested
    directly on Tally rather than only through a full pipeline run.

    The point of B7 is that (a) and (b) can disagree: pooling (a) over 597
    windows is arithmetically scissor alone (343/597 = 57.45%), which is why
    the sequence-unweighted mean binds instead.
    """

    @staticmethod
    def _counts(**overrides):
        counts = diag._empty_counts()
        counts.update(overrides)
        return counts

    def _tally(self, supply, per_sequence):
        pooled = diag._empty_counts()
        for block in per_sequence.values():
            for key, value in block.items():
                pooled[key] += value
        return diag.Tally(
            counts=pooled,
            per_sequence=per_sequence,
            end_truncated_c1a={"pooled": 0, **{name: 0 for name in supply}},
            structurally_silent=0,
            total=sum(supply.values()),
            supply=supply,
            prereg=diag.load_prereg(PREREG),
        )

    def test_only_sequences_at_or_above_the_floor_enter_pooling_b(self):
        tally = self._tally(
            {"big": 10, "small": 2},
            {
                "big": self._counts(**{diag.CLASS_C2: 8, diag.CLASS_C1A: 2}),
                "small": self._counts(**{diag.CLASS_C1A: 2}),
            },
        )
        self.assertEqual(tally.big_sequences(), ["big"])  # floor is >= 10, inclusive
        b = tally.b_fractions()
        self.assertIsNotNone(b)
        self.assertAlmostEqual(b["C2_plus_C3"], 0.8)  # 8/10 in 'big' alone
        # ...and pooling (a) genuinely DISAGREES, which is the whole point.
        self.assertAlmostEqual(tally.a_c23(), 8 / 12)

    def test_pooling_b_is_none_when_no_sequence_clears_the_floor(self):
        tally = self._tally(
            {"a": 9, "b": 9},
            {
                "a": self._counts(**{diag.CLASS_C2: 9}),
                "b": self._counts(**{diag.CLASS_C1A: 9}),
            },
        )
        self.assertEqual(tally.big_sequences(), [])
        self.assertIsNone(tally.b_fractions())  # never silently falls back to (a)

    def test_pooling_b_is_an_unweighted_mean_not_a_window_weighted_one(self):
        # 'huge' dominates by window count but must NOT dominate pooling (b).
        tally = self._tally(
            {"huge": 100, "modest": 10},
            {
                "huge": self._counts(**{diag.CLASS_C2: 100}),
                "modest": self._counts(**{diag.CLASS_C1A: 10}),
            },
        )
        self.assertEqual(tally.big_sequences(), ["huge", "modest"])
        self.assertAlmostEqual(tally.b_fractions()["C2_plus_C3"], 0.5)  # (1.0 + 0.0) / 2
        self.assertAlmostEqual(tally.a_c23(), 100 / 110)  # window-weighted: 0.909


if __name__ == "__main__":
    unittest.main()
