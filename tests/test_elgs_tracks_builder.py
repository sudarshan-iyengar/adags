"""CPU tests for scripts/build_elgs_tracks.py (spec section 2 evidence; M1).

Run with:
    C:/Users/sucar/venvs/elgs-cpu/Scripts/python.exe -m unittest tests.test_elgs_tracks_builder

Oracle style: a fully analytic synthetic rig — four cameras on a circle
looking at a radius-0.5 sphere at the origin, masks rasterized by exact
ray-sphere intersection, and a FakeTracker that reprojects known rigid
motion exactly — so seed placement, miss-token conversion, consensus
triangulation, and the controls are all checked against closed-form
ground truth, never against the implementation.
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
from elgs.tracks_schema import validate_tracks_artifact  # noqa: E402
from scripts import build_elgs_tracks as builder  # noqa: E402

SPHERE_RADIUS = 0.5
IMAGE_SIZE = 128
FOCAL = 160.0
N_FRAMES = 6
CAMERA_RADIUS = 4.0


def _look_at_c2w(position: np.ndarray) -> np.ndarray:
    """OpenGL c2w: camera -Z looks at the origin, world +Z is up."""

    z_axis = position / np.linalg.norm(position)  # +Z points away from target
    up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x_axis, y_axis, z_axis, position
    return c2w


def _rasterize_sphere_mask(camera: builder.CameraModel, center: np.ndarray) -> np.ndarray:
    """Exact ray-sphere hit test per pixel (the analytic mask oracle)."""

    us, vs = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE), indexing="xy")
    pixels = np.stack([us.ravel(), vs.ravel(), np.ones(us.size)], axis=0).astype(np.float64)
    rays_cam = np.linalg.inv(camera.K) @ pixels
    rotation = camera.w2c[:3, :3]
    origin = -(rotation.T @ camera.w2c[:3, 3])
    rays_world = (rotation.T @ rays_cam).T
    rays_world /= np.linalg.norm(rays_world, axis=1, keepdims=True)
    oc = origin - center
    b = 2.0 * rays_world @ oc
    c = float(oc @ oc) - SPHERE_RADIUS**2
    discriminant = b * b - 4.0 * c
    hit = (discriminant >= 0.0) & ((-b - np.sqrt(np.maximum(discriminant, 0.0))) / 2.0 > 0.0)
    return hit.reshape(IMAGE_SIZE, IMAGE_SIZE)


def _project_exact(camera: builder.CameraModel, point: np.ndarray) -> tuple[float, float, float]:
    cam_point = camera.w2c[:3, :3] @ point + camera.w2c[:3, 3]
    z = float(cam_point[2])
    if z <= 1e-9:
        return -1e6, -1e6, z
    uv = camera.K @ cam_point
    return float(uv[0] / uv[2]), float(uv[1] / uv[2]), z


def _write_synthetic_scene(root: Path) -> Path:
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

    from PIL import Image

    for cam_index, angle in enumerate(angles):
        position = CAMERA_RADIUS * np.array([np.cos(angle), np.sin(angle), 0.0])
        camera = builder.CameraModel(
            camera_id=cam_index,
            name=f"cam{cam_index:02d}",
            K=np.array([[FOCAL, 0, (IMAGE_SIZE - 1) / 2.0], [0, FOCAL, (IMAGE_SIZE - 1) / 2.0], [0, 0, 1.0]]),
            w2c=np.linalg.inv(_look_at_c2w(position) @ np.diag([1.0, -1.0, -1.0, 1.0])),
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
        )
        mask_dir = scene / "masks" / f"cam{cam_index:02d}"
        mask_dir.mkdir(parents=True)
        mask = _rasterize_sphere_mask(camera, np.zeros(3))
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_dir / f"{0:08d}.png")
        image_dir = scene / "undist" / f"cam{cam_index:02d}"
        image_dir.mkdir(parents=True)
        for frame_index in range(N_FRAMES):
            (image_dir / f"{frame_index:08d}.png").write_bytes(b"placeholder")
    return scene


def _rigid_motion_tracker(scene: builder.SceneBundle, seeds, velocity: np.ndarray, vis_value=1.0):
    """FakeTracker motion: exact reprojection of seeds moving rigidly."""

    def motion(image_paths, queries):
        camera_id = builder.dschema.parse_camera_id(str(image_paths[0]))
        camera = scene.cameras[camera_id]
        frame_of_local = [int(Path(p).stem) for p in image_paths]
        matched = []
        for t_local, x, y in queries:
            best, best_distance = None, np.inf
            for seed in seeds:
                point = seed.point + velocity * frame_of_local[int(round(t_local))]
                px, py, z = _project_exact(camera, point)
                distance = (px - x) ** 2 + (py - y) ** 2
                if z > 0 and distance < best_distance:
                    best, best_distance = seed, distance
            matched.append(best)
        xy = np.zeros((len(image_paths), len(queries), 2))
        vis = np.zeros((len(image_paths), len(queries)))
        for column, seed in enumerate(matched):
            for local, frame_index in enumerate(frame_of_local):
                point = seed.point + velocity * frame_index
                px, py, z = _project_exact(camera, point)
                xy[local, column] = (px, py)
                if callable(vis_value):
                    vis[local, column] = vis_value(camera_id, frame_index)
                else:
                    vis[local, column] = vis_value
        return xy, vis

    return motion


class ChunkedTrackTests(unittest.TestCase):
    """Pure chunking logic (experiment 12 CUDA OOM fix), against an exact
    linear-motion oracle: point p moves +1 px/frame in x, tracked exactly
    by a fake per-chunk tracker that continues from whatever query it gets."""

    @staticmethod
    def _exact_linear_tracker(image_paths, queries):
        # frame index encoded as the path stem; position = x0 + (f - f_q)
        frames = [int(Path(p).stem) for p in image_paths]
        xy = np.zeros((len(frames), len(queries), 2))
        vis = np.ones((len(frames), len(queries)))
        for column, (t_local, x, y) in enumerate(np.asarray(queries)):
            f_query = frames[int(round(t_local))]
            for local, frame in enumerate(frames):
                xy[local, column] = (x + (frame - f_query), y)
        return xy, vis

    def _paths(self, n):
        return [Path(f"{i:08d}.png") for i in range(n)]

    def test_short_sequence_is_single_call_passthrough(self):
        calls = []

        def spy(paths, queries):
            calls.append(len(paths))
            return self._exact_linear_tracker(paths, queries)

        queries = np.array([[0.0, 10.0, 5.0]])
        xy, vis = builder.chunked_track(spy, self._paths(50), queries, chunk_len=64, overlap=8)
        self.assertEqual(calls, [50])
        np.testing.assert_allclose(xy[:, 0, 0], 10.0 + np.arange(50))

    def test_long_sequence_chunks_and_stitches_exactly(self):
        calls = []

        def spy(paths, queries):
            calls.append((int(Path(paths[0]).stem), len(paths)))
            return self._exact_linear_tracker(paths, queries)

        queries = np.array([[0.0, 10.0, 5.0], [0.0, 3.0, 4.0]])
        n = 150
        xy, vis = builder.chunked_track(spy, self._paths(n), queries, chunk_len=64, overlap=16)
        # chunks: [0,64), [48,112), [96,150)
        self.assertEqual(calls, [(0, 64), (48, 64), (96, 54)])
        np.testing.assert_allclose(xy[:, 0, 0], 10.0 + np.arange(n))
        np.testing.assert_allclose(xy[:, 1, 0], 3.0 + np.arange(n))
        np.testing.assert_allclose(xy[:, 0, 1], 5.0)
        self.assertTrue((vis == 1.0).all())

    def test_overlap_keeps_earlier_chunk_outputs(self):
        def biased(paths, queries):
            xy, vis = self._exact_linear_tracker(paths, queries)
            # later chunks (starting past frame 0) report a recognizable bias
            if int(Path(paths[0]).stem) > 0:
                xy = xy + 1000.0
            return xy, vis

        queries = np.array([[0.0, 10.0, 5.0]])
        xy, _ = builder.chunked_track(biased, self._paths(100), queries, chunk_len=64, overlap=16)
        # frames [48,64) are covered by both chunk 0 and chunk 1: chunk 0 wins
        self.assertLess(float(xy[63, 0, 0]), 500.0)
        self.assertGreater(float(xy[64, 0, 0]), 500.0)

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ContractError):
            builder.chunked_track(
                self._exact_linear_tracker, self._paths(10), np.zeros((1, 3)), chunk_len=8, overlap=8
            )


class SyntheticSceneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.scene_dir = _write_synthetic_scene(Path(cls._tmp.name))
        cls.cfg = builder.TracksConfig(hull_resolution=32, max_seeds=64, hull_min_observers=3)
        cls.scene = builder.load_temporal_scene(cls.scene_dir)
        cls.seeds = builder.build_hull_seeds(cls.scene, 0, cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # -- scene loading and camera split ------------------------------------

    def test_held_out_rule_mod4(self):
        self.assertEqual(self.scene.held_out_ids, (0,))
        self.assertEqual(self.scene.tracking_ids, (1, 2, 3))

    def test_static_rig_violation_raises(self):
        payload = json.loads((self.scene_dir / "transforms_train.json").read_text(encoding="utf-8"))
        # frames are grouped camera-major: cam01 (a TRACKING camera) starts at
        # index N_FRAMES; perturbing a held-out camera would never be loaded.
        payload["frames"][N_FRAMES + 1]["transform_matrix"][0][3] += 0.01
        broken = self.scene_dir.parent / "broken"
        broken.mkdir(exist_ok=True)
        (broken / "transforms_train.json").write_text(json.dumps(payload), encoding="utf-8")
        (broken / "masks").mkdir(exist_ok=True)
        for name in ("cam00", "cam01", "cam02", "cam03"):
            target = broken / "masks" / name
            target.mkdir(exist_ok=True)
            (target / "00000000.png").write_bytes(
                (self.scene_dir / "masks" / name / "00000000.png").read_bytes()
            )
        with self.assertRaises(ContractError):
            builder.load_temporal_scene(broken)

    # -- seed construction --------------------------------------------------

    def test_hull_seeds_lie_on_sphere_surface(self):
        self.assertGreater(len(self.seeds), 0)
        self.assertLessEqual(len(self.seeds), self.cfg.max_seeds)
        voxel = 2.0 * CAMERA_RADIUS * self.cfg.hull_bounds_scale / (self.cfg.hull_resolution - 1)
        for seed in self.seeds:
            self.assertLess(
                abs(np.linalg.norm(seed.point) - SPHERE_RADIUS),
                3.0 * voxel,
                msg=f"seed {seed.seed_id} at {seed.point} is off the sphere surface",
            )
            self.assertGreaterEqual(seed.n_cam, 2)

    def test_seed_determinism(self):
        again = builder.build_hull_seeds(self.scene, 0, self.cfg)
        self.assertEqual(len(again), len(self.seeds))
        for a, b in zip(again, self.seeds):
            np.testing.assert_array_equal(a.point, b.point)
            self.assertEqual(a.queries, b.queries)

    def test_seed_queries_never_use_held_out_cameras(self):
        for seed in self.seeds:
            self.assertNotIn(0, seed.queries)

    # -- artifact building --------------------------------------------------

    def _build(self, velocity, vis_value=1.0, cfg=None):
        cfg = cfg or self.cfg
        backend = builder.FakeTracker(
            _rigid_motion_tracker(self.scene, self.seeds, np.asarray(velocity), vis_value)
        )
        return builder.build_artifact(self.scene, self.seeds, backend, cfg, 0)

    def test_artifact_validates_and_triangulates_motion(self):
        velocity = np.array([0.05, 0.0, 0.0])
        artifact = self._build(velocity)
        validate_tracks_artifact(artifact)
        checked = 0
        for seed in self.seeds[:5]:
            for entry in artifact["consensus"][str(seed.seed_id)]:
                if entry["point"] is None:
                    continue
                frame = int(entry["frame"])
                expected = seed.point + velocity * frame
                np.testing.assert_allclose(entry["point"], expected, atol=1e-6)
                checked += 1
        self.assertGreater(checked, 0)

    def test_out_of_domain_positions_become_miss_tokens(self):
        artifact = self._build(np.array([1.0, 0.0, 0.0]))
        misses = sum(
            1 for track in artifact["tracks"] for r in track["reports"] if r["is_miss"]
        )
        positional = sum(
            1 for track in artifact["tracks"] for r in track["reports"] if not r["is_miss"]
        )
        self.assertGreater(misses, 0)
        self.assertGreater(positional, 0)
        for track in artifact["tracks"]:
            for report in track["reports"]:
                if not report["is_miss"]:
                    camera = self.scene.cameras[track["camera_id"]]
                    self.assertTrue(0.0 <= report["x"] <= camera.width - 1.0)
                    self.assertTrue(0.0 <= report["y"] <= camera.height - 1.0)

    def test_consensus_unknown_below_two_visible_cameras(self):
        artifact = self._build(
            np.zeros(3), vis_value=lambda camera_id, frame: 1.0 if camera_id == 1 else 0.1
        )
        for entries in artifact["consensus"].values():
            for entry in entries:
                self.assertIsNone(entry["point"])
                self.assertLessEqual(entry["n_cam"], 1)

    def test_fb_diagnostics_zero_for_exact_tracker(self):
        artifact = self._build(np.array([0.05, 0.0, 0.0]))
        fb = artifact["diagnostics"]["fb_rms_px"]
        self.assertGreater(len(fb), 0)
        for value in fb.values():
            self.assertLess(value, 1e-6)

    # -- controls ------------------------------------------------------------

    def test_shift_control_shifts_and_drops(self):
        artifact = self._build(np.array([0.05, 0.0, 0.0]))
        control = builder.make_shift_control(artifact, 2)
        validate_tracks_artifact(control)
        window = set(float(i) for i in artifact["window"]["frame_indices"])
        for original, shifted in zip(artifact["tracks"], control["tracks"]):
            kept = [r for r in original["reports"] if r["frame"] + 2 in window]
            self.assertEqual(len(shifted["reports"]), len(kept))
            for report_original, report_shifted in zip(kept, shifted["reports"]):
                self.assertEqual(report_shifted["frame"], report_original["frame"] + 2)

    def test_shuffle_control_permutes_identity(self):
        artifact = self._build(np.array([0.05, 0.0, 0.0]))
        control = builder.make_shuffle_control(artifact, self.cfg.shuffle_seed)
        validate_tracks_artifact(control)
        mapping = control["control"]["mapping"]
        self.assertEqual(sorted(mapping.keys(), key=int), sorted(set(mapping.keys()), key=int))
        self.assertEqual(
            sorted(int(v) for v in mapping.values()), sorted(int(k) for k in mapping.keys())
        )
        self.assertNotEqual(
            [int(k) for k in mapping.keys()], [int(v) for v in mapping.values()],
            msg="permutation must not be the identity for this seed count",
        )
        for old_id, entries in artifact["consensus"].items():
            new_id = str(mapping[str(old_id)])
            self.assertEqual(control["consensus"][new_id], entries)
        again = builder.make_shuffle_control(artifact, self.cfg.shuffle_seed)
        self.assertEqual(again["control"]["mapping"], mapping)

    # -- frozen dir + CLI ----------------------------------------------------

    def test_frozen_dir_seal_and_duplicate_refusal(self):
        artifact = self._build(np.array([0.05, 0.0, 0.0]))
        backend = builder.FakeTracker()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "artifact"
            builder.write_frozen_artifact_dir(
                out, artifact, self.cfg, self.scene, backend, wall_seconds=1.0
            )
            manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], builder.TRACKS_MANIFEST_SCHEMA)
            self.assertIn("tracks.json", manifest["files_sha256"])
            self.assertIn("tracks_shift.json", manifest["files_sha256"])
            self.assertIn("tracks_shuffle.json", manifest["files_sha256"])
            self.assertEqual(manifest["r_u_mapping"], builder.R_U_MAPPING)
            self.assertEqual(manifest["accounting"]["category"], "preprocessing")
            self.assertFalse(manifest["tracker_identity"]["evidence_admissible"])
            from depth_visibility.canonical import sha256_file

            for name, expected in manifest["files_sha256"].items():
                self.assertEqual(sha256_file(out / name), expected)
            with self.assertRaises(ContractError):
                builder.write_frozen_artifact_dir(
                    out, artifact, self.cfg, self.scene, backend, wall_seconds=1.0
                )

    def test_sealed_artifact_round_trips_plain_floats(self):
        # Regression (experiment 8 / retry 2 defect): the canonical writer
        # hex-encodes floats (binary64_hex identity tokens, no repo decoder),
        # which made the first sealed real artifact numerically unreadable.
        # The sealed files must be PLAIN JSON: numbers come back as floats
        # and a control can be re-derived from the RELOADED artifact.
        artifact = self._build(np.array([0.05, 0.0, 0.0]))
        backend = builder.FakeTracker()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "artifact"
            builder.write_frozen_artifact_dir(
                out, artifact, self.cfg, self.scene, backend, wall_seconds=1.0
            )
            reloaded = json.loads((out / "tracks.json").read_text(encoding="utf-8"))
            report = next(
                r
                for track in reloaded["tracks"]
                for r in track["reports"]
                if not r["is_miss"]
            )
            self.assertIsInstance(report["v"], float)
            self.assertIsInstance(report["x"], float)
            self.assertIsInstance(report["frame"], float)
            point = next(
                e["point"]
                for entries in reloaded["consensus"].values()
                for e in entries
                if e["point"] is not None
            )
            self.assertTrue(all(isinstance(c, float) for c in point))
            rederived = builder.make_shift_control(reloaded, 2)
            self.assertGreater(len(rederived["tracks"]), 0)

    def test_cli_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = builder.main(
                [
                    "--scene-dir", str(self.scene_dir),
                    "--out-dir", str(Path(tmp) / "dry"),
                    "--tracker", "fake",
                    "--hull-resolution", "32",
                    "--max-seeds", "64",
                    "--dry-run",
                ]
            )
            self.assertEqual(code, 0)
            report = json.loads((Path(tmp) / "dry" / "dry_run_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["dry_run"])
            self.assertGreater(report["n_seeds"], 0)

    def test_cli_fake_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sealed"
            code = builder.main(
                [
                    "--scene-dir", str(self.scene_dir),
                    "--out-dir", str(out),
                    "--tracker", "fake",
                    "--hull-resolution", "32",
                    "--max-seeds", "64",
                ]
            )
            self.assertEqual(code, 0)
            artifact = json.loads((out / "tracks.json").read_text(encoding="utf-8"))
            validate_tracks_artifact(artifact)
            self.assertTrue((out / "MANIFEST.json").is_file())


if __name__ == "__main__":
    unittest.main()
