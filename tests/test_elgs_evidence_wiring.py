"""Unit tests for the EL-GS evidence-path wiring (M1).

CPU only, unittest. Covers the surface M0 disclosed as implemented-but-
unwired: sealed-artifact admission (`elgs/tracks_loader.py`), the
assembly that turns an artifact into Phi (`elgs/evidence_stack.py`), the
live-model probe's reference parity against the FROZEN CPU oracle
(`elgs/probe_model.py` vs `elgs/probe.py`), and the checkpoint slot the
evidence block rides in (`elgs/state_io.py`).

The probe-parity test is the one `configs/elgs/prereg_observability_v1.json`
mandates: "the GPU implementation must match elgs.probe.AnalyticProbe on
the closed-form fixture to 1e-6".
"""

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from depth_visibility.canonical import sha256_file
from depth_visibility.errors import ContractError
from elgs.bridges import Window
from elgs.clusters import SeedCluster
from elgs.evidence_stack import (
    bridge_point,
    build_bridge_family,
    build_clusters,
    build_evidence_context,
    derive_seed_evidence,
    family_anchor_series,
    family_windows,
    load_evidence_constants,
    presence_series,
    seed_overlap_edges,
    sigma_points_for,
)
from elgs.probe import AnalyticProbe, ProbeGaussian
from elgs.probe_model import ModelProbe, project_points
from elgs.state_io import build_elgs_state, load_elgs_state
from elgs.tracks_loader import (
    MANIFEST_FILENAME,
    PRIMARY_FILENAME,
    SUPERSEDED_MARKER,
    TRACKS_MANIFEST_SCHEMA,
    TracksIncompatible,
    load_sealed_tracks,
)
from elgs.tracks_schema import TRACKS_SCHEMA

PREREG_DIR = str(Path(__file__).resolve().parents[1] / "configs" / "elgs")

#: The fixture's object is STATIONARY, so every report lands on the same
#: pixel and a correct bridge projects onto it. That agreement is what
#: makes the evidence support presence, and therefore what makes the sign
#: of the evidence term well defined. An earlier fixture let the report
#: drift with the frame while the stub probe projected to a fixed pixel;
#: `g_pos` then decayed with distance and the evidence legitimately
#: opposed presence — a fixture artifact that says nothing about the code.
_REPORT_PX = (100.0, 60.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _artifact(
    *,
    n_seeds: int = 3,
    frames: tuple = (0, 1, 2, 3, 4, 5, 6, 7),
    cameras: tuple = (1, 2, 3),
    absent: tuple = (3, 4),
) -> dict:
    """A valid artifact whose seeds are visible outside `absent` frames.

    The visible/absent shape gives every family two anchor plateaus with
    a genuine window between them, which is what the evidence path needs
    to produce a nonzero Phi.
    """
    seeds = [
        {"seed_id": s, "point": [0.1 * s, 0.0, 1.0], "n_cam": len(cameras)}
        for s in range(n_seeds)
    ]
    tracks = []
    track_id = 0
    for seed in seeds:
        for camera_id in cameras:
            reports = []
            for frame in frames:
                base = {"frame": float(frame), "time": float(frame) / 120.0}
                if frame in absent:
                    # In-domain but low visibility: the exact condition
                    # the absence diagnostic found dominates real data.
                    reports.append({**base, "is_miss": False, "v": 0.2,
                                    "x": _REPORT_PX[0], "y": _REPORT_PX[1]})
                else:
                    reports.append({**base, "is_miss": False, "v": 0.9,
                                    "x": _REPORT_PX[0], "y": _REPORT_PX[1]})
            tracks.append({
                "track_id": track_id, "seed_id": seed["seed_id"],
                "camera_id": camera_id, "reports": reports,
            })
            track_id += 1
    consensus = {
        str(s["seed_id"]): [
            {
                "frame": float(f),
                "point": None if f in absent else [0.1 * s["seed_id"], 0.0, 1.0],
                "n_cam": 0 if f in absent else len(cameras),
                "reproj_rms": 0.4,
            }
            for f in frames
        ]
        for s in seeds
    }
    return {
        "schema_version": TRACKS_SCHEMA,
        "seeds": seeds,
        "tracks": tracks,
        "consensus": consensus,
        "diagnostics": {
            "fb_rms_px": {f"{s['seed_id']}:{cameras[0]}": 0.5 for s in seeds},
            "reproj_rms_px": {str(s["seed_id"]): 0.4 for s in seeds},
            "r_u_mapping": "test",
        },
        "window": {
            "frame_indices": [int(f) for f in frames],
            "query_frame": 0,
            "fps": 120.0,
        },
        "manifest": {
            "training_cameras": [int(c) for c in cameras],
            "held_out_cameras": [0],
            "tracker_identity": json.dumps(
                {"backend": "cotracker3", "evidence_admissible": True}, sort_keys=True
            ),
            "source_data_sha256": "a" * 64,
        },
    }


def _seal(directory: Path, artifact: dict, **manifest_overrides) -> Path:
    """Write a sealed artifact dir the way build_elgs_tracks does."""
    directory.mkdir(parents=True, exist_ok=True)
    primary = directory / PRIMARY_FILENAME
    primary.write_text(
        json.dumps(artifact, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": TRACKS_MANIFEST_SCHEMA,
        "files_sha256": {PRIMARY_FILENAME: sha256_file(primary)},
        "primary": PRIMARY_FILENAME,
        "config_sha256": "c" * 64,
        "tracker_identity": {"backend": "cotracker3", "evidence_admissible": True},
        "scene_dir": "/apollo/users/sri/proj_adags/data/diva360_derived/demo_screen_w0_7",
        "source_sha256": {"value": "a" * 64, "of": "transforms_train.json"},
        "training_cameras": artifact["manifest"]["training_cameras"],
        "held_out_cameras": artifact["manifest"]["held_out_cameras"],
        "window": {
            "frame_indices_first": artifact["window"]["frame_indices"][0],
            "frame_indices_last": artifact["window"]["frame_indices"][-1],
            "n_frames": len(artifact["window"]["frame_indices"]),
            "fps": 120.0,
        },
    }
    manifest.update(manifest_overrides)
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return directory


_SCENE_KWARGS = dict(
    scene_sequence_id="demo_screen_w0_7",
    scene_camera_ids=(1, 2, 3),
    scene_frame_indices=tuple(range(8)),
    scene_source_sha256="a" * 64,
)


class _TempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="elgs-evidence-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. Sealed-artifact admission — every refusal reason, individually
# ---------------------------------------------------------------------------


class TracksAdmissionTests(_TempDirTest):
    def _sealed(self, **overrides) -> Path:
        return _seal(self.tmp / "tracks", _artifact(), **overrides)

    def test_valid_artifact_is_admitted(self):
        sealed = load_sealed_tracks(self._sealed(), **_SCENE_KWARGS)
        self.assertEqual(sealed.sequence_id, "demo_screen_w0_7")
        self.assertTrue(sealed.evidence_admissible)
        self.assertEqual(sealed.training_cameras, (1, 2, 3))
        self.assertEqual(len(sealed.seeds), 3)
        self.assertIn("primary_sha256", sealed.provenance())

    def _assert_reason(self, reason, directory=None, **kwargs):
        merged = dict(_SCENE_KWARGS)
        merged.update(kwargs)
        with self.assertRaises(TracksIncompatible) as caught:
            load_sealed_tracks(directory if directory is not None else self._sealed(), **merged)
        self.assertEqual(caught.exception.reason, reason)

    def test_missing_directory(self):
        self._assert_reason("missing_directory", self.tmp / "absent")

    def test_unsealed_directory(self):
        bare = self.tmp / "bare"
        bare.mkdir()
        (bare / PRIMARY_FILENAME).write_text("{}", encoding="utf-8")
        self._assert_reason("unsealed_directory", bare)

    def test_manifest_schema_mismatch(self):
        self._assert_reason("manifest_schema", self._sealed(schema_version="nope-v9"))

    def test_superseded_marker_refuses(self):
        directory = self._sealed()
        (directory / SUPERSEDED_MARKER).write_text("fix79ae5b7", encoding="utf-8")
        self._assert_reason("superseded", directory)

    def test_superseded_manifest_key_refuses(self):
        self._assert_reason(
            "superseded", self._sealed(superseded_by="writing_2_screen_w0_239_fix79ae5b7")
        )

    def test_hash_mismatch_refuses(self):
        directory = self._sealed()
        # Byte-identical length, different content: only the digest catches it.
        primary = directory / PRIMARY_FILENAME
        text = primary.read_text(encoding="utf-8")
        primary.write_text(text.replace('"v":0.9', '"v":0.8', 1), encoding="utf-8")
        self._assert_reason("hash_mismatch", directory)

    def test_fake_backend_refused_for_evidence_bearing(self):
        directory = self._sealed(
            tracker_identity={"backend": "fake", "evidence_admissible": False}
        )
        self._assert_reason("backend_not_admissible", directory)

    def test_fake_backend_admitted_when_not_evidence_bearing(self):
        directory = self._sealed(
            tracker_identity={"backend": "fake", "evidence_admissible": False}
        )
        sealed = load_sealed_tracks(
            directory, require_evidence_admissible=False, **_SCENE_KWARGS
        )
        self.assertFalse(sealed.evidence_admissible)

    def test_cross_sequence_isolation(self):
        merged = dict(_SCENE_KWARGS)
        merged["scene_sequence_id"] = "poker_screen_w0_109"
        with self.assertRaises(TracksIncompatible) as caught:
            load_sealed_tracks(self._sealed(), **merged)
        self.assertEqual(caught.exception.reason, "sequence_mismatch")

    def test_substrate_mismatch(self):
        merged = dict(_SCENE_KWARGS)
        merged["scene_source_sha256"] = "b" * 64
        with self.assertRaises(TracksIncompatible) as caught:
            load_sealed_tracks(self._sealed(), **merged)
        self.assertEqual(caught.exception.reason, "substrate_mismatch")

    def test_camera_not_in_scene(self):
        merged = dict(_SCENE_KWARGS)
        merged["scene_camera_ids"] = (1, 2)
        with self.assertRaises(TracksIncompatible) as caught:
            load_sealed_tracks(self._sealed(), **merged)
        self.assertEqual(caught.exception.reason, "camera_mismatch")

    def test_frame_range_mismatch(self):
        merged = dict(_SCENE_KWARGS)
        merged["scene_frame_indices"] = (0, 1, 2)
        with self.assertRaises(TracksIncompatible) as caught:
            load_sealed_tracks(self._sealed(), **merged)
        self.assertEqual(caught.exception.reason, "frame_range_mismatch")

    def test_held_out_leak(self):
        artifact = _artifact()
        directory = _seal(
            self.tmp / "leak", artifact, held_out_cameras=[1], training_cameras=[1, 2, 3]
        )
        self._assert_reason("held_out_leak", directory)

    def test_control_arm_is_refused_even_when_promoted_to_primary(self):
        """A control's digest sits in the same files_sha256 map, so
        editing `primary` would otherwise slip a falsification arm past
        every other check with its seal intact."""
        artifact = _artifact()
        artifact["control"] = {"kind": "shift", "shift_frames": 10}
        self._assert_reason("control_artifact", _seal(self.tmp / "ctl", artifact))

    def test_artifact_schema_failure_is_reported_as_such(self):
        artifact = _artifact()
        artifact["seeds"][0]["n_cam"] = 1  # violates the >= 2 camera seed rule
        self._assert_reason("artifact_schema", _seal(self.tmp / "badseed", artifact))


# ---------------------------------------------------------------------------
# 2. Probe reference parity (prereg_observability: match to 1e-6)
# ---------------------------------------------------------------------------


class _StubCamera:
    """Minimal Camera surface the probe reads."""

    def __init__(self, width=64, height=48, fl=50.0):
        self.image_width = width
        self.image_height = height
        self.fl_x = fl
        self.fl_y = fl
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.znear = 0.01
        self.timestamp = 0.0
        # Identity extrinsics in the row-vector convention Camera uses.
        self.world_view_transform = torch.eye(4, dtype=torch.float32)


class _StubGaussians:
    def __init__(self, xyz, opacity, scaling, family_ids):
        self._xyz = xyz
        self._opacity = opacity
        self._scaling = scaling
        self._elgs_family_ids = family_ids

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_scaling(self):
        return self._scaling


class ProbeParityTests(unittest.TestCase):
    def test_model_probe_matches_a_hand_computed_closed_form(self):
        """INDEPENDENT parity: the expected value is computed here from
        the compositing definition, NOT by calling project_points.

        The earlier version of this test built the AnalyticProbe fixture
        with the implementation under test, so the extrinsics convention,
        the intrinsic fallback and the sigma_px derivation all cancelled
        and only the product was compared. This one fixes the geometry by
        hand: identity extrinsics, fl = 50, principal point (32, 24), so
        a splat at (X, Y, Z) lands at (50X/Z + 32, 50Y/Z + 24) with
        sigma_px = mean_scale * 50 / Z.
        """
        camera = _StubCamera()  # 64x48, fl 50, cx 32, cy 24, identity w2c
        # One splat exactly on the query ray at depth 2, one off-axis.
        xyz = torch.tensor([[0.0, 0.0, 2.0], [0.04, 0.0, 2.0]], dtype=torch.float32)
        opacity = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
        scale = 0.02
        scaling = torch.full((2, 3), scale, dtype=torch.float32)
        gaussians = _StubGaussians(
            xyz, opacity, scaling, torch.tensor([1, 1], dtype=torch.long)
        )
        probe = ModelProbe(
            gaussians, {1: camera},
            presence_at=lambda t, f: torch.ones((2, 1), dtype=torch.float32),
        )

        # Hand-computed geometry.
        sigma_px = scale * 50.0 / 2.0                      # 0.5 px
        centre_0 = (32.0, 24.0)                            # on-axis
        centre_1 = (50.0 * 0.04 / 2.0 + 32.0, 24.0)        # 33.0 px
        query = (32.0, 24.0)
        g0 = math.exp(-0.5 * 0.0 / sigma_px ** 2)          # 1.0
        g1 = math.exp(-0.5 * ((query[0] - centre_1[0]) ** 2) / sigma_px ** 2)
        expected = (1.0 - 0.5 * g0) * (1.0 - 0.25 * g1)

        actual = probe.transmittance(
            1, 0.0, query, 3.0, exclude_track=None, present_family=None
        )
        self.assertAlmostEqual(actual, expected, delta=1e-6)
        self.assertAlmostEqual(centre_0[0], 32.0, places=12)  # fixture sanity

    def test_pixel_scale_maps_projections_into_the_report_raster(self):
        """The domain fix: a model loaded at resolution N must report
        pixels in the FULL report raster, not its own downscaled one."""
        full = _StubCamera(width=64, height=48, fl=50.0)
        quarter = _StubCamera(width=16, height=12, fl=12.5)
        quarter.cx, quarter.cy = 8.0, 6.0
        xyz = torch.tensor([[0.04, 0.0, 2.0]], dtype=torch.float32)
        gaussians = _StubGaussians(
            xyz, torch.tensor([[0.5]], dtype=torch.float32),
            torch.full((1, 3), 0.02, dtype=torch.float32),
            torch.tensor([1], dtype=torch.long),
        )
        at_full = ModelProbe(
            gaussians, {1: full}, presence_at=lambda t, f: torch.ones((1, 1))
        ).project(1, (0.04, 0.0, 2.0))
        at_quarter = ModelProbe(
            gaussians, {1: quarter}, presence_at=lambda t, f: torch.ones((1, 1)),
            pixel_scale=4.0,
        ).project(1, (0.04, 0.0, 2.0))
        self.assertIsNotNone(at_full)
        self.assertIsNotNone(at_quarter)
        self.assertAlmostEqual(at_full[0][0], at_quarter[0][0], delta=1e-4)
        self.assertAlmostEqual(at_full[0][1], at_quarter[0][1], delta=1e-4)

    def test_geometry_cache_does_not_grow_with_frames_or_families(self):
        """The cache key was (camera, frame, family), which at S1 scale
        is hundreds of GB. Geometry depends on the camera alone."""
        camera = _StubCamera()
        gaussians = _StubGaussians(
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.5]], dtype=torch.float32),
            torch.full((1, 3), 0.02, dtype=torch.float32),
            torch.tensor([1], dtype=torch.long),
        )
        probe = ModelProbe(
            gaussians, {1: camera}, presence_at=lambda t, f: torch.ones((1, 1)),
            frame_to_time={i: i / 120.0 for i in range(20)},
        )
        for frame in range(20):
            for family in (1, 2, 3, 4, 5):
                probe.transmittance(
                    1, float(frame), (32.0, 24.0), 4.0,
                    exclude_track=None, present_family=family,
                )
        self.assertEqual(len(probe._geometry), 1)
        self.assertEqual(len(probe._presence), 20)

    def test_model_probe_matches_analytic_probe_to_1e6(self):
        camera = _StubCamera()
        # Three splats in front of a query point at depth 4.0.
        xyz = torch.tensor(
            [[0.0, 0.0, 1.0], [0.05, 0.0, 2.0], [0.0, 0.04, 3.0], [0.0, 0.0, 5.0]],
            dtype=torch.float32,
        )
        opacity = torch.tensor([[0.6], [0.4], [0.3], [0.9]], dtype=torch.float32)
        scaling = torch.full((4, 3), 0.02, dtype=torch.float32)
        family_ids = torch.tensor([7, 7, 8, 9], dtype=torch.long)
        gaussians = _StubGaussians(xyz, opacity, scaling, family_ids)

        presence = torch.ones((4, 1), dtype=torch.float32)
        model_probe = ModelProbe(
            gaussians, {1: camera}, presence_at=lambda t, f: presence
        )

        pixel_centers, depths = project_points(camera, xyz)
        focal = 0.5 * (camera.fl_x + camera.fl_y)
        splats = [
            ProbeGaussian(
                depth=float(depths[i]),
                alpha=float(opacity[i, 0]),
                sigma_px=max(float(scaling[i].mean() * focal / depths[i]), 1e-3),
                center_px=(float(pixel_centers[i, 0]), float(pixel_centers[i, 1])),
                family_id=int(family_ids[i]),
            )
            for i in range(xyz.shape[0])
        ]
        analytic = AnalyticProbe({(1, 0.0): splats})

        query_pixel = (float(camera.cx), float(camera.cy))
        for depth in (2.5, 4.0, 6.0):
            expected = analytic.transmittance(
                1, 0.0, query_pixel, depth, exclude_track=None, present_family=None
            )
            actual = model_probe.transmittance(
                1, 0.0, query_pixel, depth, exclude_track=None, present_family=None
            )
            self.assertAlmostEqual(actual, expected, delta=1e-6, msg=f"depth={depth}")

    def test_presence_zero_makes_transmittance_one(self):
        camera = _StubCamera()
        xyz = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        gaussians = _StubGaussians(
            xyz,
            torch.tensor([[0.9]], dtype=torch.float32),
            torch.full((1, 3), 0.02, dtype=torch.float32),
            torch.tensor([3], dtype=torch.long),
        )
        absent = ModelProbe(
            gaussians, {1: camera},
            presence_at=lambda t, f: torch.zeros((1, 1), dtype=torch.float32),
        )
        self.assertAlmostEqual(
            absent.transmittance(
                1, 0.0, (camera.cx, camera.cy), 3.0,
                exclude_track=None, present_family=None,
            ),
            1.0,
            delta=1e-12,
        )

    def test_frame_to_time_is_used_not_camera_timestamp(self):
        """A static rig contributes one Camera per id; presence must be
        evaluated at the FRAME's time, not that Camera's own timestamp."""
        camera = _StubCamera()
        camera.timestamp = 0.0
        seen = []
        gaussians = _StubGaussians(
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.5]], dtype=torch.float32),
            torch.full((1, 3), 0.02, dtype=torch.float32),
            torch.tensor([1], dtype=torch.long),
        )

        def presence_at(timestamp, family):
            seen.append(timestamp)
            return torch.ones((1, 1), dtype=torch.float32)

        probe = ModelProbe(
            gaussians, {1: camera}, presence_at=presence_at,
            frame_to_time={0: 0.0, 5: 0.125},
        )
        probe.transmittance(
            1, 5.0, (camera.cx, camera.cy), 4.0,
            exclude_track=None, present_family=None,
        )
        self.assertEqual(seen, [0.125])

    def test_unknown_frame_fails_closed(self):
        camera = _StubCamera()
        gaussians = _StubGaussians(
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.5]], dtype=torch.float32),
            torch.full((1, 3), 0.02, dtype=torch.float32),
            torch.tensor([1], dtype=torch.long),
        )
        probe = ModelProbe(
            gaussians, {1: camera},
            presence_at=lambda t, f: torch.ones((1, 1)),
            frame_to_time={0: 0.0},
        )
        with self.assertRaises(ContractError):
            probe.transmittance(
                1, 9.0, (0.0, 0.0), 2.0, exclude_track=None, present_family=None
            )


# ---------------------------------------------------------------------------
# 3. Assembly: seeds -> clusters -> anchors/windows -> bridges
# ---------------------------------------------------------------------------


class AssemblyTests(_TempDirTest):
    def setUp(self):
        super().setUp()
        self.constants = load_evidence_constants(PREREG_DIR, smoke=True)
        self.sealed = load_sealed_tracks(
            _seal(self.tmp / "tracks", _artifact()), **_SCENE_KWARGS
        )

    def test_seed_evidence_follows_the_artifact_r_u_mapping(self):
        seeds = derive_seed_evidence(
            self.sealed, r_min=0.05, vis_threshold=0.5, min_cameras=2
        )
        self.assertEqual(len(seeds), 3)
        # min(1 - 0.5/8, 1 - 0.4/8) = min(0.9375, 0.95) = 0.9375
        self.assertAlmostEqual(seeds[0].r_u, 0.9375, places=9)
        self.assertGreater(seeds[0].d_u, 0.0)
        self.assertLessEqual(seeds[0].d_u, 1.0)

    def test_missing_diagnostics_fall_back_to_r_min(self):
        artifact = _artifact()
        artifact["diagnostics"]["fb_rms_px"] = {}
        artifact["diagnostics"]["reproj_rms_px"] = {}
        sealed = load_sealed_tracks(_seal(self.tmp / "nodiag", artifact), **_SCENE_KWARGS)
        seeds = derive_seed_evidence(sealed, r_min=0.05, vis_threshold=0.5, min_cameras=2)
        self.assertTrue(all(s.r_u == 0.05 for s in seeds))

    def test_overlap_radius_controls_cluster_count(self):
        seeds = derive_seed_evidence(
            self.sealed, r_min=0.05, vis_threshold=0.5, min_cameras=2
        )
        # Seeds sit 0.1 apart on x.
        self.assertEqual(len(seed_overlap_edges(seeds, 0.05)), 0)
        self.assertEqual(len(build_clusters(seeds, radius=0.05, constants=self.constants)), 3)
        self.assertEqual(len(build_clusters(seeds, radius=0.5, constants=self.constants)), 1)

    def test_anchor_series_splits_on_the_low_visibility_gap(self):
        track_ids = [int(t["track_id"]) for t in self.sealed.tracks
                     if int(t["seed_id"]) == 0]
        frames, plateau, capped = family_anchor_series(
            self.sealed, track_ids, c_cap=4, vis_threshold=0.5, min_cameras=2
        )
        self.assertEqual(len(frames), 8)
        # Frames 3 and 4 carry v=0.2 for every camera -> not a plateau.
        self.assertEqual(plateau, (True, True, True, False, False, True, True, True))
        self.assertTrue(all(c <= 4 for c in capped))

    def test_windows_appear_between_the_two_anchors(self):
        track_ids = [int(t["track_id"]) for t in self.sealed.tracks
                     if int(t["seed_id"]) == 0]
        anchors, windows = family_windows(
            self.sealed, track_ids, c_cap=4, constants=self.constants
        )
        self.assertEqual(len(anchors), 2)
        self.assertEqual(len(windows), 1)
        self.assertEqual((windows[0].start_frame, windows[0].end_frame), (2.0, 5.0))

    def test_bridge_family_is_three_hypotheses_from_anchor_endpoints(self):
        window = Window(2.0, 5.0)
        family = build_bridge_family(self.sealed, 0, window)
        self.assertEqual(sorted(family), [0, 1, 2])
        # hold-left, linear, hold-right, all at the same consensus point
        # here (the fixture's object does not move), so all three agree.
        for bridge_id in family:
            point = bridge_point(family[bridge_id], window, 3.0)
            self.assertAlmostEqual(point[0], 0.0, places=9)

    def test_sigma_points_renormalize_when_some_fail_to_project(self):
        def project(point):
            # Drop every +x offset.
            return None if point[0] > 0.001 else ((10.0, 10.0), 2.0)

        points = sigma_points_for((0.0, 0.0, 1.0), 0.02, project)
        self.assertTrue(points)
        self.assertAlmostEqual(sum(p.weight for p in points), 1.0, places=12)

    def test_presence_series_requires_an_explicit_frame_time_map(self):
        class _Realization:
            b = [0.0]
            d = [0.02]

        self.assertEqual(
            presence_series(_Realization(), (0.0, 1.0), {0: 0.0, 1: 0.01}),
            (True, True),
        )
        with self.assertRaises(ContractError):
            presence_series(_Realization(), (0.0, 9.0), {0: 0.0})


# ---------------------------------------------------------------------------
# 4. Insufficient-evidence fallback and cross-sequence isolation in context
# ---------------------------------------------------------------------------


class EvidenceContextTests(_TempDirTest):
    def setUp(self):
        super().setUp()
        self.constants = load_evidence_constants(PREREG_DIR, smoke=True)
        self.sealed = load_sealed_tracks(
            _seal(self.tmp / "tracks", _artifact()), **_SCENE_KWARGS
        )

    def _context(self, threshold=1.0, offset=0.0):
        # `offset` moves the primitives away from the seed points so a
        # binding threshold can actually fail; at offset 0 the canonical
        # point coincides with a primitive and every threshold binds.
        positions = torch.tensor(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]], dtype=torch.float32
        ) + torch.tensor([0.0, offset, 0.0], dtype=torch.float32)
        family_ids = torch.tensor([11, 11, 12], dtype=torch.long)
        return build_evidence_context(
            self.sealed, self.constants, positions, family_ids,
            r_site=0.5, binding_threshold=threshold, c_cap=4, beta=0.1, tau_b=1.0,
        )

    def test_clusters_bind_to_families(self):
        context = self._context()
        self.assertTrue(context.clusters)
        bound = [
            context.binding.family_of(c.cluster_id) for c in context.clusters
        ]
        self.assertTrue(any(f is not None for f in bound))

    def test_unbindable_clusters_are_inactive_not_an_error(self):
        context = self._context(threshold=0.01, offset=5.0)
        self.assertTrue(
            all(context.binding.family_of(c.cluster_id) is None
                for c in context.clusters)
        )
        # A family with no bound cluster has no evidence: the fallback.
        self.assertEqual(context.track_ids_of_family(11), ())

    def test_declared_D_img_must_match_the_report_raster(self):
        """The pixel-domain guard. A D_img in one raster and reports in
        another zeroes the visible head for every report and gives the
        evidence term a constant sign."""
        positions = torch.tensor(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]], dtype=torch.float32
        )
        family_ids = torch.tensor([11, 11, 12], dtype=torch.long)
        with self.assertRaises(ContractError) as caught:
            build_evidence_context(
                self.sealed, self.constants, positions, family_ids,
                r_site=0.5, binding_threshold=1.0, c_cap=4, beta=0.1, tau_b=1.0,
                report_raster=(290.0, 137.0),  # the downscaled raster
            )
        self.assertIn("pixel domain", str(caught.exception))
        # The declared raster is accepted.
        build_evidence_context(
            self.sealed, self.constants, positions, family_ids,
            r_site=0.5, binding_threshold=1.0, c_cap=4, beta=0.1, tau_b=1.0,
            report_raster=(1160.0, 550.0),
        )

    def test_context_state_round_trips_through_the_checkpoint_slot(self):
        context = self._context()
        payload = context.to_state()
        self.assertIn("tracks_provenance", payload)
        self.assertEqual(payload["tier"], "smoke")
        self.assertTrue(payload["clusters"])
        # The evidence binding is its OWN table and must persist: the M0
        # elgs_state.binding slot holds a different object.
        self.assertIn("binding", payload)
        self.assertIn("binding", payload["binding"])
        # The evidence block rides inside elgs_state; a run WITHOUT one
        # must still load (photometric-only and every M0-era checkpoint).
        state = build_elgs_state(
            *_empty_bundle(),
            sampler={"lambda_u": 1.0, "pi_d_identity": "u", "frozen": True},
            slot_grid={}, confirmation_refs={}, moment_reset_log=[],
            round_bookkeeping={}, evidence=payload,
        )
        loaded = load_elgs_state(json.loads(json.dumps(state)))
        self.assertEqual(loaded["evidence"]["tier"], "smoke")
        self.assertEqual(
            loaded["evidence"]["tracks_provenance"]["sequence_id"],
            "demo_screen_w0_7",
        )

    def test_checkpoint_without_evidence_block_still_loads(self):
        state = build_elgs_state(
            *_empty_bundle(),
            sampler={"lambda_u": 1.0, "pi_d_identity": "u", "frozen": True},
            slot_grid={}, confirmation_refs={}, moment_reset_log=[],
            round_bookkeeping={},
        )
        del state["evidence"]  # an M0-era checkpoint
        loaded = load_elgs_state(state)
        self.assertEqual(loaded["evidence"], {})


def _empty_bundle():
    """A real, empty (registry, binding, ledger, search_cost) bundle.

    Real objects rather than stubs: a hand-written stub can disagree with
    the live `to_state`/`from_state` contract and pass anyway, which is
    exactly what an M0-era-checkpoint compatibility test must not do.
    """
    from elgs.clusters import BindingTable
    from elgs.families import FamilyRegistry
    from elgs.transaction_ledger import SearchCostLedger, TransactionLedger

    return (
        FamilyRegistry(),
        BindingTable(),
        TransactionLedger(),
        SearchCostLedger(row_cap=600000, scalar_budget=10**10),
    )


# ---------------------------------------------------------------------------
# 5. The heads gate
# ---------------------------------------------------------------------------


class _StubProbe:
    """A probe that projects everything to one pixel with full visibility.

    Enough to exercise the whole q -> Phi chain deterministically without
    a model: `ProbeParityTests` is what checks the real probe's physics.
    """

    def __init__(self, transmittance_value=1.0):
        self._t = float(transmittance_value)

    def project(self, camera_id, point):
        # ON the report pixel: a correct bridge for a stationary object.
        return _REPORT_PX, 2.0

    def in_frustum(self, camera_id, frame, point):
        return True

    def transmittance(self, camera_id, frame, pixel, depth, *, exclude_track, present_family):
        return self._t


class EvidenceDeltaTests(_TempDirTest):
    """The end-to-end claim: a NON-PHOTOMETRIC term reaches acceptance."""

    def setUp(self):
        super().setUp()
        from elgs.families import FamilyRegistry
        from elgs.intervals import IntervalConfig, inverse

        self.constants = load_evidence_constants(PREREG_DIR, smoke=True)
        self.sealed = load_sealed_tracks(
            _seal(self.tmp / "tracks", _artifact()), **_SCENE_KWARGS
        )
        # Frame index -> model time; 8 frames at 120 fps.
        self.frame_time = {i: i / 120.0 for i in range(8)}
        self.config = IntervalConfig(
            T=8 / 120.0, w_m=2 / 120.0, w=0.5 / 120.0,
            floor_len=2.0 * (0.5 / 120.0) + 1 / 120.0,
            floor_gap=2.0 * (0.5 / 120.0) + 1 / 120.0,
            delta_tol=0.1 / 120.0,
        )
        self.registry = FamilyRegistry()
        spanning = inverse(
            1, True, True, 0.0, [self.config.omega], [], 0.0, self.config,
            dtype=torch.float32,
        )
        self.record = self.registry.create_family(
            birth_time=0.0, birth_site=(0.0, 0.0, 1.0),
            lineage_key="test", interval=spanning, tau=(0.0,),
        )
        positions = torch.tensor(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]], dtype=torch.float32
        )
        family_ids = torch.full((3,), self.record.family_id, dtype=torch.long)
        self.context = build_evidence_context(
            self.sealed, self.constants, positions, family_ids,
            r_site=0.5, binding_threshold=1.0, c_cap=4, beta=0.1, tau_b=1.0,
        )

    def _round_evidence(self, probe=None):
        from elgs.evidence_stack import refresh_round_evidence

        return refresh_round_evidence(
            self.context, probe or _StubProbe(),
            round_index=0, family_ids=[self.record.family_id],
        )

    def test_round_refresh_produces_windows_and_q_values(self):
        evidence = self._round_evidence()
        self.assertTrue(evidence.windows, "the fixture must yield >=1 window")
        self.assertGreater(evidence.n_q_values, 0)
        self.assertTrue(evidence.q_snapshot.frozen)
        coverage = evidence.coverage[str(self.record.family_id)]
        self.assertEqual(coverage["n_anchors"], 2)
        self.assertFalse(coverage["photometric_only"])

    def test_q_snapshot_is_frozen_against_mid_round_writes(self):
        evidence = self._round_evidence()
        with self.assertRaises(ContractError):
            evidence.q_snapshot.put(0, 0, 1, 3.0, 0.5)

    def test_phi_is_nonzero_and_absence_differs_from_presence(self):
        from elgs.intervals import forward

        evidence = self._round_evidence()
        present = forward(self.record.interval, self.config)
        phi_present = _family_phi(self.context, evidence, self.record.family_id,
                                  present, self.frame_time)
        phi_absent = _family_phi(self.context, evidence, self.record.family_id,
                                 None, self.frame_time)
        # Absence is the L0 stream on both limbs, so Phi is exactly 0.
        self.assertEqual(phi_absent, 0.0)
        # Presence must actually move the energy: this is the whole point.
        self.assertNotEqual(phi_present, 0.0)
        self.assertTrue(math.isfinite(phi_present))

    def test_exact_delta_is_nonzero_for_a_presence_changing_candidate(self):
        from elgs.evidence_stack import evidence_exact_delta

        evidence = self._round_evidence()
        plan = _Plan(
            family_ids=(self.record.family_id,),
            child_intervals={self.record.family_id: _EMPTY_INTERVAL()},
        )
        delta = evidence_exact_delta(
            self.context, evidence, plan, self.registry, self.config,
            frame_time=self.frame_time,
        )
        self.assertNotEqual(delta, 0.0)
        self.assertTrue(math.isfinite(delta))

    def test_family_without_windows_contributes_exactly_zero(self):
        """INSUFFICIENT-EVIDENCE FALLBACK: the decision reduces to the
        verified M0 photometric comparison, bit for bit."""
        from elgs.evidence_stack import evidence_exact_delta

        evidence = self._round_evidence()
        plan = _Plan(
            family_ids=(4242,),  # a family with no bound clusters
            child_intervals={4242: _EMPTY_INTERVAL()},
        )
        delta = evidence_exact_delta(
            self.context, evidence, plan, self.registry, self.config,
            frame_time=self.frame_time,
        )
        self.assertEqual(delta, 0.0)

    def test_fully_occluded_probe_drives_q_to_zero(self):
        """Censoring: q_tilde == 0 everywhere makes the scored and
        censored streams identical, so PROP-1 gives Phi exactly 0."""
        from elgs.intervals import forward

        evidence = self._round_evidence(probe=_StubProbe(transmittance_value=0.0))
        present = forward(self.record.interval, self.config)
        phi_present = _family_phi(self.context, evidence, self.record.family_id,
                                  present, self.frame_time)
        self.assertEqual(phi_present, 0.0)

    def test_evidence_term_has_the_right_SIGN_for_a_supported_family(self):
        """The behaviour no earlier test covered, and the one whose
        absence let a pixel-domain bug through: the SIGN.

        The fixture's tracks report the object present (v=0.9) either
        side of the gap and the bridge sits on its consensus point, so
        the evidence supports PRESENCE. Deleting the family must
        therefore RAISE the energy — a positive exact_deltas — which
        pushes `delta_total + k*SE < 0` away from acceptance. A sign
        inversion anywhere in phi() or evidence_exact_delta flips this.
        """
        from elgs.evidence_stack import evidence_exact_delta

        evidence = self._round_evidence()
        delta = evidence_exact_delta(
            self.context, evidence,
            _Plan(family_ids=(self.record.family_id,),
                  child_intervals={self.record.family_id: _EMPTY_INTERVAL()}),
            self.registry, self.config, frame_time=self.frame_time,
        )
        self.assertGreater(
            delta, 0.0,
            "deleting a family the tracks support must raise the energy",
        )

    def test_window_endpoints_are_excluded_so_windows_never_share_a_frame(self):
        """Single-frame anchors make consecutive windows touch. With
        inclusive endpoints the shared frame was counted in both windows
        AND its q was reused across two different bridge geometries."""
        from elgs.bridges import AnchorInterval, windows_between_anchors
        from elgs.evidence_stack import _reports_in_window

        anchors = (
            AnchorInterval(0.0, 1.0, 9),
            AnchorInterval(3.0, 3.0, 9),   # single frame
            AnchorInterval(6.0, 7.0, 9),
        )
        windows = windows_between_anchors(anchors)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].end_frame, windows[1].start_frame)

        track_ids = [int(t["track_id"]) for t in self.sealed.tracks]
        params = self.constants.heads.params
        first = set(_reports_in_window(self.sealed, track_ids, windows[0], params))
        second = set(_reports_in_window(self.sealed, track_ids, windows[1], params))
        self.assertEqual(first & second, set(), "windows must not share reports")

    def test_beta_scales_the_delta_linearly(self):
        from elgs.evidence_stack import evidence_exact_delta

        evidence = self._round_evidence()
        plan = _Plan(
            family_ids=(self.record.family_id,),
            child_intervals={self.record.family_id: _EMPTY_INTERVAL()},
        )
        base = evidence_exact_delta(
            self.context, evidence, plan, self.registry, self.config,
            frame_time=self.frame_time,
        )
        self.context.beta = 0.2
        doubled = evidence_exact_delta(
            self.context, evidence, plan, self.registry, self.config,
            frame_time=self.frame_time,
        )
        self.assertAlmostEqual(doubled, 2.0 * base, places=9)

    def test_scoring_a_candidate_never_mutates_the_registry(self):
        """Rollback: candidate evaluation is frozen-functional, so a
        rejected plan leaves the live interval untouched."""
        from elgs.evidence_stack import evidence_exact_delta
        from elgs.intervals import serialize_state

        evidence = self._round_evidence()
        before = serialize_state(self.registry.get(self.record.family_id).interval)
        plan = _Plan(
            family_ids=(self.record.family_id,),
            child_intervals={self.record.family_id: _EMPTY_INTERVAL()},
        )
        evidence_exact_delta(
            self.context, evidence, plan, self.registry, self.config,
            frame_time=self.frame_time,
        )
        after = serialize_state(self.registry.get(self.record.family_id).interval)
        self.assertEqual(before, after)


def _family_phi(context, evidence, family_id, realization, frame_time):
    from elgs.evidence_stack import family_phi

    return family_phi(
        context, evidence, family_id, realization, frame_time=frame_time
    )


def _EMPTY_INTERVAL():
    """The K=0 empty program: no latch bits and no vector.

    `IntervalState.__post_init__` rejects latch bits on a K=0 state
    ("K=0 empty program carries no latch bits"), so a deletion candidate
    must be built exactly this way.
    """
    from elgs.intervals import IntervalState

    return IntervalState(K=0, latch_pre=None, latch_post=None, a=None)


class _Plan:
    """The TransactionPlan surface `evidence_exact_delta` reads."""

    def __init__(self, family_ids, child_intervals):
        self.family_ids = family_ids
        self.child_intervals = child_intervals
        self.op = "TEST"


class HeadsGateTests(unittest.TestCase):
    def test_frozen_path_refuses_while_heads_are_unfrozen(self):
        with self.assertRaises(ContractError) as caught:
            load_evidence_constants(PREREG_DIR, smoke=False)
        message = str(caught.exception)
        self.assertIn("not frozen", message)
        self.assertIn("g_v", message)

    def test_smoke_path_loads_declared_constants(self):
        constants = load_evidence_constants(PREREG_DIR, smoke=True)
        self.assertEqual(constants.tier, "smoke")
        self.assertAlmostEqual(constants.alpha_model(1), 1.0, places=12)
        self.assertLess(constants.alpha_model(4), constants.alpha_model(2))

    def test_g_v_is_monotone_increasing(self):
        """The heads prereg's g_v family carries a monotone-increasing
        requirement, which beta(a,b) satisfies only for b <= 1 (for b > 1
        it peaks interior at (a-1)/(a+b-2)). This test is what caught the
        first draft's b=2."""
        constants = load_evidence_constants(PREREG_DIR, smoke=True)
        values = [constants.heads.g_v(v) for v in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)]
        self.assertEqual(values, sorted(values))
        self.assertLessEqual(max(values), constants.heads.params.g_cap)

    def test_smoke_file_must_declare_itself_non_evidence_bearing(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(PREREG_DIR) / "smoke_evidence_heads_v1.json"
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["evidence_bearing"] = True
            (Path(tmp) / "smoke_evidence_heads_v1.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaises(ContractError):
                load_evidence_constants(tmp, smoke=True)


if __name__ == "__main__":
    unittest.main()
