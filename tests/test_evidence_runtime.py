"""Unit tests for the CSVL-VPL v2 evidence runtime and consensus builder.

CPU only, unittest, no GPU and no P01 access. Covers:

* verdict parity against ``primitive_census.classify_states_v2`` on a synthetic
  two-camera consensus directory (near / behind-weak / occluded / in-front /
  not-evaluable), including the census ``gap_raw`` split;
* the ``UNCERTAIN`` material abstention that the census does not have;
* ``time_shift`` mode reading ``(frame + shift) % T``;
* ``backproject`` round-tripping to ``NEAR`` verdicts with (x, y) pixels;
* fail-closed construction guards, counters, and the memory-mapped path;
* ``scripts/build_evidence_consensus.py`` end to end on a synthetic P01 tree,
  including the per-frame camera geometry drift guard.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from depth_visibility import evidence_runtime as er  # noqa: E402
from depth_visibility import primitive_census as census  # noqa: E402
from depth_visibility.errors import ArtifactError, ContractError, SchemaError  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

HEIGHT = 24
WIDTH = 32
FRAMES = 4
FOCAL = 40.0
CX = 16.0
CY = 12.0

TAU_REL = 0.03
KAPPA = 2.5
K_GAP = 3.0
NEAR_CLIP = 0.01

SIGMA_BASE = 0.01
SIGMA_PATCH = 0.5
PATCH_ROWS = slice(16, 20)
PATCH_COLS = slice(20, 26)
INVALID_ROWS = slice(0, 3)
INVALID_COLS = slice(0, 2)


# ---------------------------------------------------------------------------
# Synthetic consensus fixture
# ---------------------------------------------------------------------------


def intrinsics_matrix() -> np.ndarray:
    return np.array(
        [[FOCAL, 0.0, CX], [0.0, FOCAL, CY], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def rotation_y(degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array(
        [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]], dtype=np.float64
    )


def fixture_arrays() -> dict[str, np.ndarray]:
    cameras = ["cam01", "cam05"]
    frames = list(range(FRAMES))
    intrinsics = np.stack([intrinsics_matrix(), intrinsics_matrix()])
    w2c = np.zeros((2, 4, 4), dtype=np.float64)
    w2c[0] = np.eye(4)
    w2c[1, :3, :3] = rotation_y(12.0)
    w2c[1, :3, 3] = np.array([0.35, -0.10, 0.20])
    w2c[1, 3, 3] = 1.0

    rows, cols = np.meshgrid(
        np.arange(HEIGHT, dtype=np.float64),
        np.arange(WIDTH, dtype=np.float64),
        indexing="ij",
    )
    base = 4.0 + 0.004 * cols + 0.003 * rows

    depth = np.zeros((2, FRAMES, HEIGHT, WIDTH), dtype=np.float64)
    sigma = np.full((2, FRAMES, HEIGHT, WIDTH), SIGMA_BASE, dtype=np.float64)
    valid = np.ones((2, FRAMES, HEIGHT, WIDTH), dtype=bool)
    for camera in range(2):
        for frame in range(FRAMES):
            depth[camera, frame] = base + 0.45 * frame + 0.20 * camera

    valid[:, :, INVALID_ROWS, :] = False
    valid[:, :, :, INVALID_COLS] = False
    sigma[:, :, PATCH_ROWS, PATCH_COLS] = SIGMA_PATCH

    depth = np.where(valid, depth, np.nan)
    sigma = np.where(valid, sigma, np.nan)
    return {
        "cameras": cameras,
        "frames": frames,
        "intrinsics": intrinsics,
        "w2c": w2c,
        "d": depth.astype(np.float16),
        "sigma": sigma.astype(np.float16),
        "valid": valid,
    }


def write_fixture(out_dir: Path) -> dict[str, np.ndarray]:
    arrays = fixture_arrays()
    er.write_consensus_directory(
        out_dir,
        d=arrays["d"],
        sigma=arrays["sigma"],
        valid=arrays["valid"],
        intrinsics=arrays["intrinsics"],
        w2c=arrays["w2c"],
        cameras=arrays["cameras"],
        frames=arrays["frames"],
        meta_extra={"scene": "unit_fixture", "scientific": False},
    )
    return arrays


def stored_maps(out_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read back exactly what the runtime reads (fp16 promoted to float32)."""

    depth = np.load(out_dir / "d.npy").astype(np.float32)
    sigma = np.load(out_dir / "sigma.npy").astype(np.float32)
    valid = np.load(out_dir / "valid.npy").astype(bool)
    return depth, sigma, valid


def pixel_ray_point(px: float, py: float, depth: float) -> np.ndarray:
    """World point that lands on pixel (px, py) at ``depth`` for camera 0."""

    return np.array(
        [(px - CX) * depth / FOCAL, (py - CY) * depth / FOCAL, depth],
        dtype=np.float64,
    )


def census_expected(
    xyz: np.ndarray,
    w2c: np.ndarray,
    intrinsics: np.ndarray,
    depth_map: np.ndarray,
    sigma_map: np.ndarray,
    valid_map: np.ndarray,
) -> np.ndarray:
    """Map ``classify_states_v2`` output onto the runtime verdict codes."""

    pixels, z, in_view = census.project_points(
        np.asarray(xyz, dtype=np.float64),
        w2c,
        intrinsics,
        HEIGHT,
        WIDTH,
        near_clip=NEAR_CLIP,
    )
    present = np.ones(xyz.shape[0], dtype=bool)
    states, gap_raw, _, _ = census.classify_states_v2(
        z,
        pixels,
        in_view,
        present,
        depth_map,
        sigma_map,
        valid_map,
        tau_rel=TAU_REL,
        kappa=KAPPA,
        k_gap=K_GAP,
    )
    expected = np.full(states.shape[0], er.VERDICT_NOT_EVALUABLE, dtype=np.int8)
    expected[states == census.STATE_NEAR_SURFACE] = er.VERDICT_NEAR
    expected[states == census.STATE_IN_FRONT] = er.VERDICT_IN_FRONT
    behind = states == census.STATE_BEHIND
    expected[behind & ~gap_raw] = er.VERDICT_BEHIND_WEAK
    expected[behind & gap_raw] = er.VERDICT_OCCLUDED
    return expected


def probe_points() -> np.ndarray:
    """Hand-built points covering every class plus a deterministic cloud."""

    depth_at = 4.0 + 0.004 * 14.0 + 0.003 * 10.0  # camera 0, frame 0, pixel (14, 10)
    margin = max(TAU_REL * depth_at, KAPPA * SIGMA_BASE)
    crafted = [
        pixel_ray_point(14.0, 10.0, depth_at),                      # near
        pixel_ray_point(14.0, 10.0, depth_at + 2.0 * margin),       # behind weak
        pixel_ray_point(14.0, 10.0, depth_at + 6.0 * margin),       # occluded
        pixel_ray_point(14.0, 10.0, depth_at - 2.0 * margin),       # in front
        pixel_ray_point(14.0, 1.0, depth_at),                       # invalid rows
        pixel_ray_point(0.0, 10.0, depth_at),                       # invalid columns
        pixel_ray_point(200.0, 10.0, depth_at),                     # out of view
        np.array([0.0, 0.0, -2.0]),                                 # behind camera
        pixel_ray_point(22.0, 17.0, depth_at + 0.6),                # sigma patch
    ]
    rng = np.random.default_rng(20260730)
    cloud = np.stack(
        [
            rng.uniform(-1.6, 1.6, size=512),
            rng.uniform(-1.1, 1.1, size=512),
            rng.uniform(2.6, 6.4, size=512),
        ],
        axis=1,
    )
    return np.concatenate([np.stack(crafted), cloud], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Runtime tests
# ---------------------------------------------------------------------------


class EvidenceRuntimeFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._tmp.name) / "consensus"
        cls.arrays = write_fixture(cls.dir)
        cls.depth, cls.sigma, cls.valid = stored_maps(cls.dir)
        cls.points = probe_points()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def runtime(self, **overrides):
        kwargs = dict(
            device="cpu",
            tau_rel=TAU_REL,
            kappa=KAPPA,
            k_gap=K_GAP,
            sigma_abstain=0.20,
            near_clip=NEAR_CLIP,
        )
        kwargs.update(overrides)
        return er.EvidenceRuntime(str(self.dir), **kwargs)


class VerdictParityTests(EvidenceRuntimeFixtureTest):
    def test_parity_with_classify_states_v2(self):
        # Abstention disabled so the runtime and the census cover the same
        # class space; the census has no UNCERTAIN state.
        runtime = self.runtime(sigma_abstain=1.0e9)
        seen = set()
        for camera_index, camera in enumerate(self.arrays["cameras"]):
            for frame in self.arrays["frames"]:
                got = runtime.verdicts(
                    torch.from_numpy(self.points), camera, frame
                ).numpy()
                expected = census_expected(
                    self.points,
                    self.arrays["w2c"][camera_index],
                    self.arrays["intrinsics"][camera_index],
                    self.depth[camera_index, frame],
                    self.sigma[camera_index, frame],
                    self.valid[camera_index, frame],
                )
                np.testing.assert_array_equal(got, expected)
                seen.update(int(v) for v in np.unique(got))
        self.assertEqual(
            seen,
            {
                er.VERDICT_NOT_EVALUABLE,
                er.VERDICT_NEAR,
                er.VERDICT_OCCLUDED,
                er.VERDICT_BEHIND_WEAK,
                er.VERDICT_IN_FRONT,
            },
        )

    def test_crafted_classes(self):
        runtime = self.runtime(sigma_abstain=1.0e9)
        got = runtime.verdicts(torch.from_numpy(self.points[:8]), "cam01", 0).numpy()
        self.assertEqual(int(got[0]), er.VERDICT_NEAR)
        self.assertEqual(int(got[1]), er.VERDICT_BEHIND_WEAK)
        self.assertEqual(int(got[2]), er.VERDICT_OCCLUDED)
        self.assertEqual(int(got[3]), er.VERDICT_IN_FRONT)
        for index in range(4, 8):
            self.assertEqual(int(got[index]), er.VERDICT_NOT_EVALUABLE)

    def test_gap_boundary_matches_k_gap(self):
        runtime = self.runtime(sigma_abstain=1.0e9)
        depth_at = float(self.depth[0, 0, 10, 14])
        margin = max(TAU_REL * depth_at, KAPPA * SIGMA_BASE)
        just_weak = pixel_ray_point(14.0, 10.0, depth_at + K_GAP * margin - 1e-3)
        just_gap = pixel_ray_point(14.0, 10.0, depth_at + K_GAP * margin + 1e-3)
        points = torch.from_numpy(np.stack([just_weak, just_gap]).astype(np.float32))
        got = runtime.verdicts(points, "cam01", 0).numpy()
        self.assertEqual(int(got[0]), er.VERDICT_BEHIND_WEAK)
        self.assertEqual(int(got[1]), er.VERDICT_OCCLUDED)


class AbstentionTests(EvidenceRuntimeFixtureTest):
    def test_high_sigma_pixel_abstains(self):
        runtime = self.runtime(sigma_abstain=0.20)
        depth_at = float(self.depth[0, 0, 17, 22])
        inside = pixel_ray_point(22.0, 17.0, depth_at)
        outside = pixel_ray_point(14.0, 10.0, float(self.depth[0, 0, 10, 14]))
        points = torch.from_numpy(np.stack([inside, outside]).astype(np.float32))
        got = runtime.verdicts(points, "cam01", 0).numpy()
        self.assertEqual(int(got[0]), er.VERDICT_UNCERTAIN)
        self.assertEqual(int(got[1]), er.VERDICT_NEAR)

    def test_abstention_overrides_occlusion(self):
        strict = self.runtime(sigma_abstain=0.20)
        permissive = self.runtime(sigma_abstain=1.0e9)
        depth_at = float(self.depth[0, 0, 17, 22])
        margin = max(TAU_REL * depth_at, KAPPA * SIGMA_PATCH)
        occluded = pixel_ray_point(22.0, 17.0, depth_at + 4.0 * margin)
        points = torch.from_numpy(occluded[None, :].astype(np.float32))
        self.assertEqual(
            int(permissive.verdicts(points, "cam01", 0)[0]), er.VERDICT_OCCLUDED
        )
        self.assertEqual(
            int(strict.verdicts(points, "cam01", 0)[0]), er.VERDICT_UNCERTAIN
        )

    def test_abstention_threshold_is_absolute_units(self):
        # sigma_abstain sits between the base sigma and the patch sigma; only
        # the patch abstains, independently of the local depth magnitude.
        runtime = self.runtime(sigma_abstain=0.5 * (SIGMA_BASE + SIGMA_PATCH))
        depth_patch = float(self.depth[0, 0, 17, 22])
        depth_plain = float(self.depth[0, 0, 10, 14])
        points = np.stack(
            [
                pixel_ray_point(22.0, 17.0, depth_patch),
                pixel_ray_point(14.0, 10.0, depth_plain),
            ]
        ).astype(np.float32)
        got = runtime.verdicts(torch.from_numpy(points), "cam01", 0).numpy()
        self.assertEqual(int(got[0]), er.VERDICT_UNCERTAIN)
        self.assertEqual(int(got[1]), er.VERDICT_NEAR)


class TimeShiftTests(EvidenceRuntimeFixtureTest):
    def test_evidence_frame_is_circular_shift(self):
        runtime = self.runtime(mode="time_shift", time_shift=1)
        for frame in self.arrays["frames"]:
            self.assertEqual(
                runtime.evidence_frame(frame, "cam01"), (frame + 1) % FRAMES
            )
            self.assertEqual(
                runtime.evidence_frame(frame, "cam05"),
                runtime.evidence_frame(frame, "cam01"),
            )

    def test_valid_mode_is_identity(self):
        runtime = self.runtime()
        for frame in self.arrays["frames"]:
            self.assertEqual(runtime.evidence_frame(frame, "cam01"), frame)

    def test_shifted_verdicts_equal_shifted_frame(self):
        plain = self.runtime()
        shifted = self.runtime(mode="time_shift", time_shift=1)
        points = torch.from_numpy(self.points)
        differed = False
        for camera in self.arrays["cameras"]:
            for frame in self.arrays["frames"]:
                got = shifted.verdicts(points, camera, frame).numpy()
                reference = plain.verdicts(
                    points, camera, (frame + 1) % FRAMES
                ).numpy()
                np.testing.assert_array_equal(got, reference)
                same_frame = plain.verdicts(points, camera, frame).numpy()
                differed = differed or bool((got != same_frame).any())
        self.assertTrue(differed, "time_shift control did not change any verdict")

    def test_negative_shift_wraps(self):
        runtime = self.runtime(mode="time_shift", time_shift=-1)
        self.assertEqual(runtime.evidence_frame(0, "cam01"), FRAMES - 1)

    def test_no_op_shift_is_rejected(self):
        with self.assertRaises(ContractError):
            self.runtime(mode="time_shift", time_shift=0)
        with self.assertRaises(ContractError):
            self.runtime(mode="time_shift", time_shift=2 * FRAMES)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ContractError):
            self.runtime(mode="shuffle")


class BackprojectTests(EvidenceRuntimeFixtureTest):
    def test_round_trip_is_near(self):
        runtime = self.runtime()
        world, pixels = runtime.backproject("cam01", 1, stride=3, sigma_max=0.1)
        self.assertGreater(int(world.shape[0]), 50)
        self.assertEqual(tuple(world.shape[1:]), (3,))
        self.assertEqual(tuple(pixels.shape[1:]), (2,))
        got = runtime.verdicts(world, "cam01", 1).numpy()
        self.assertTrue((got == er.VERDICT_NEAR).all(), np.unique(got))

    def test_pixels_are_x_then_y(self):
        runtime = self.runtime()
        world, pixels = runtime.backproject("cam01", 0, stride=3, sigma_max=0.1)
        px = pixels[:, 0].numpy()
        py = pixels[:, 1].numpy()
        self.assertLess(int(px.max()), WIDTH)
        self.assertLess(int(py.max()), HEIGHT)
        # camera 0 has an identity extrinsic, so world z is the map depth.
        expected_depth = self.depth[0, 0][py, px]
        np.testing.assert_allclose(
            world[:, 2].numpy(), expected_depth, rtol=1e-3, atol=1e-3
        )
        recovered_x = np.rint(FOCAL * world[:, 0].numpy() / world[:, 2].numpy() + CX)
        np.testing.assert_array_equal(recovered_x.astype(np.int64), px)

    def test_sigma_and_validity_filters(self):
        runtime = self.runtime()
        _, pixels = runtime.backproject("cam01", 0, stride=1, sigma_max=0.1)
        px = pixels[:, 0].numpy()
        py = pixels[:, 1].numpy()
        in_patch = (
            (py >= PATCH_ROWS.start)
            & (py < PATCH_ROWS.stop)
            & (px >= PATCH_COLS.start)
            & (px < PATCH_COLS.stop)
        )
        self.assertEqual(int(in_patch.sum()), 0)
        self.assertEqual(int((py < INVALID_ROWS.stop).sum()), 0)
        self.assertEqual(int((px < INVALID_COLS.stop).sum()), 0)

    def test_high_sigma_max_admits_the_patch(self):
        runtime = self.runtime()
        _, pixels = runtime.backproject("cam01", 0, stride=1, sigma_max=1.0)
        px = pixels[:, 0].numpy()
        py = pixels[:, 1].numpy()
        in_patch = (
            (py >= PATCH_ROWS.start)
            & (py < PATCH_ROWS.stop)
            & (px >= PATCH_COLS.start)
            & (px < PATCH_COLS.stop)
        )
        self.assertGreater(int(in_patch.sum()), 0)

    def test_time_shift_backprojects_shifted_frame(self):
        plain = self.runtime()
        shifted = self.runtime(mode="time_shift", time_shift=1)
        world_shifted, _ = shifted.backproject("cam01", 0, stride=3, sigma_max=0.1)
        world_plain, _ = plain.backproject("cam01", 1, stride=3, sigma_max=0.1)
        np.testing.assert_array_equal(world_shifted.numpy(), world_plain.numpy())

    def test_unknown_camera_or_frame_is_empty(self):
        runtime = self.runtime()
        world, pixels = runtime.backproject("cam99", 0, stride=3, sigma_max=0.1)
        self.assertEqual(int(world.shape[0]), 0)
        self.assertEqual(int(pixels.shape[0]), 0)
        world, pixels = runtime.backproject("cam01", 999, stride=3, sigma_max=0.1)
        self.assertEqual(int(world.shape[0]), 0)

    def test_invalid_arguments(self):
        runtime = self.runtime()
        with self.assertRaises(ContractError):
            runtime.backproject("cam01", 0, stride=0, sigma_max=0.1)
        with self.assertRaises(ContractError):
            runtime.backproject("cam01", 0, stride=3, sigma_max=0.0)


class LedgerAndGuardTests(EvidenceRuntimeFixtureTest):
    def test_has_camera(self):
        runtime = self.runtime()
        self.assertTrue(runtime.has_camera("cam01"))
        self.assertFalse(runtime.has_camera("cam00"))
        self.assertFalse(runtime.has_camera(None))

    def test_missing_camera_is_not_evaluable(self):
        runtime = self.runtime()
        points = torch.from_numpy(self.points)
        got = runtime.verdicts(points, "cam00", 0)
        self.assertTrue((got.numpy() == er.VERDICT_NOT_EVALUABLE).all())
        counts = runtime.counts_since_reset()
        self.assertEqual(counts["missing_camera_calls"], 1)
        self.assertEqual(counts["not_evaluable"], self.points.shape[0])

    def test_missing_frame_is_not_evaluable(self):
        runtime = self.runtime()
        got = runtime.verdicts(torch.from_numpy(self.points), "cam01", 4242)
        self.assertTrue((got.numpy() == er.VERDICT_NOT_EVALUABLE).all())
        self.assertEqual(runtime.evidence_frame(4242, "cam01"), -1)
        self.assertEqual(runtime.counts_since_reset()["missing_frame_calls"], 1)

    def test_counts_match_verdicts_and_reset(self):
        runtime = self.runtime()
        points = torch.from_numpy(self.points)
        total = np.zeros(er.NUM_VERDICTS, dtype=np.int64)
        for frame in self.arrays["frames"]:
            got = runtime.verdicts(points, "cam01", frame).numpy()
            total += np.bincount(got.astype(np.int64), minlength=er.NUM_VERDICTS)
        counts = runtime.counts_since_reset()
        for code, name in enumerate(er.VERDICT_NAMES):
            self.assertEqual(counts[name], int(total[code]), name)
        self.assertEqual(counts["calls"], FRAMES)
        self.assertEqual(counts["points"], FRAMES * self.points.shape[0])
        self.assertEqual(counts["total"], int(total.sum()))
        runtime.reset_counts()
        cleared = runtime.counts_since_reset()
        self.assertEqual(cleared["total"], 0)
        self.assertEqual(cleared["calls"], 0)

    def test_empty_batch(self):
        runtime = self.runtime()
        got = runtime.verdicts(torch.zeros((0, 3)), "cam01", 0)
        self.assertEqual(tuple(got.shape), (0,))
        self.assertEqual(got.dtype, torch.int8)

    def test_bad_shape_is_rejected(self):
        runtime = self.runtime()
        with self.assertRaises(ContractError):
            runtime.verdicts(torch.zeros((5, 4)), "cam01", 0)

    def test_detaches_graph_inputs(self):
        runtime = self.runtime()
        xyz = torch.tensor(self.points, requires_grad=True)
        got = runtime.verdicts(xyz, "cam01", 0)
        self.assertFalse(got.requires_grad)

    def test_memmap_path_matches_preload(self):
        preloaded = self.runtime(preload=True)
        mapped = self.runtime(preload=False)
        points = torch.from_numpy(self.points)
        for camera in self.arrays["cameras"]:
            for frame in self.arrays["frames"]:
                np.testing.assert_array_equal(
                    preloaded.verdicts(points, camera, frame).numpy(),
                    mapped.verdicts(points, camera, frame).numpy(),
                )
        world_a, pixels_a = preloaded.backproject("cam05", 2, stride=2, sigma_max=0.1)
        world_b, pixels_b = mapped.backproject("cam05", 2, stride=2, sigma_max=0.1)
        np.testing.assert_array_equal(world_a.numpy(), world_b.numpy())
        np.testing.assert_array_equal(pixels_a.numpy(), pixels_b.numpy())

    def test_hyperparameter_guards(self):
        for kwargs in (
            {"tau_rel": 0.0},
            {"kappa": -1.0},
            {"k_gap": 0.5},
            {"sigma_abstain": 0.0},
            {"near_clip": 0.0},
        ):
            with self.assertRaises(ContractError):
                self.runtime(**kwargs)

    def test_missing_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ArtifactError):
                er.EvidenceRuntime(str(Path(tmp) / "absent"), device="cpu")
            partial = Path(tmp) / "partial"
            partial.mkdir()
            (partial / "meta.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ArtifactError):
                er.EvidenceRuntime(str(partial), device="cpu")

    def test_schema_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "consensus"
            write_fixture(target)
            meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
            meta["schema_version"] = "something-else-v9"
            (target / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(SchemaError):
                er.EvidenceRuntime(str(target), device="cpu")

    def test_shape_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "consensus"
            write_fixture(target)
            meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
            meta["height"] = HEIGHT + 1
            (target / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(SchemaError):
                er.EvidenceRuntime(str(target), device="cpu")

    def test_public_surface(self):
        runtime = self.runtime()
        self.assertEqual(runtime.cameras, ["cam01", "cam05"])
        self.assertEqual(runtime.camera_index, {"cam01": 0, "cam05": 1})
        self.assertEqual(runtime.num_frames, FRAMES)
        self.assertEqual((runtime.height, runtime.width), (HEIGHT, WIDTH))
        self.assertEqual(
            (er.VERDICT_NOT_EVALUABLE, er.VERDICT_NEAR, er.VERDICT_OCCLUDED,
             er.VERDICT_BEHIND_WEAK, er.VERDICT_IN_FRONT, er.VERDICT_UNCERTAIN),
            (0, 1, 2, 3, 4, 5),
        )


# ---------------------------------------------------------------------------
# Consensus builder tests
# ---------------------------------------------------------------------------


P01_CAMERAS = ["cam01", "cam05", "cam12"]
P01_FRAMES = [0, 1, 2]
P01_HEIGHT = 10
P01_WIDTH = 12
GROUPS_PER_FRAME = 2


def load_builder_module():
    path = REPO_ROOT / "scripts" / "build_evidence_consensus.py"
    spec = importlib.util.spec_from_file_location("_build_evidence_consensus", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def member_depth(camera_index: int, frame: int, group: int) -> np.ndarray:
    rows, cols = np.meshgrid(
        np.arange(P01_HEIGHT, dtype=np.float32),
        np.arange(P01_WIDTH, dtype=np.float32),
        indexing="ij",
    )
    return (
        3.0
        + 0.05 * camera_index
        + 0.25 * frame
        + 0.01 * group
        + 0.02 * cols
        + 0.03 * rows
    ).astype(np.float32)


def member_confidence(camera_index: int) -> np.ndarray:
    conf = np.full((P01_HEIGHT, P01_WIDTH), 0.9, dtype=np.float32)
    conf[0, :] = 0.1  # one low-confidence row, dropped by the percentile floor
    conf[:, 0] = 0.1
    if camera_index == 0:
        conf[5, 5] = 0.1
    return conf


def write_fake_p01(root: Path, *, drift_frame: int | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    intrinsics = np.array(
        [[30.0, 0.0, 6.0], [0.0, 30.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    groups = []
    for frame in P01_FRAMES:
        for group in range(GROUPS_PER_FRAME):
            folder = root / "arrays" / f"frame_{frame:06d}" / f"group_{group:04d}"
            folder.mkdir(parents=True, exist_ok=True)
            members = len(P01_CAMERAS)
            depth = np.stack(
                [member_depth(i, frame, group) for i in range(members)]
            )
            if group == 1:
                depth[2, 2, 2] = 0.0  # a non-finite/invalid member sample
            confidence = np.stack([member_confidence(i) for i in range(members)])
            w2c = np.stack([np.eye(4, dtype=np.float32) for _ in range(members)])
            for i in range(members):
                w2c[i, 0, 3] = 0.1 * i
            if drift_frame is not None and frame == drift_frame:
                w2c[:, 1, 3] += 0.01
            k = np.stack([intrinsics.copy() for _ in range(members)])
            np.save(folder / "depth.npy", depth)
            np.save(folder / "confidence.npy", confidence)
            np.save(folder / "aligned_w2c.npy", w2c)
            np.save(folder / "processed_intrinsics.npy", k)
            relative = f"arrays/frame_{frame:06d}/group_{group:04d}"
            groups.append(
                {
                    "frame": frame,
                    "group_index": len(groups),
                    "member_camera_ids": list(P01_CAMERAS),
                    "processed_depth_shape": [members, P01_HEIGHT, P01_WIDTH],
                    "array_refs": {
                        "depth": {"path": f"{relative}/depth.npy"},
                        "confidence": {"path": f"{relative}/confidence.npy"},
                        "aligned_w2c": {"path": f"{relative}/aligned_w2c.npy"},
                        "processed_intrinsics": {
                            "path": f"{relative}/processed_intrinsics.npy"
                        },
                    },
                }
            )
    manifest = {
        "schema_version": "phase9-p01-da3-v1",
        "scene": "unit_fixture",
        "target_camera": "cam00",
        "frame_count": len(P01_FRAMES),
        "group_count": len(groups),
        "groups": groups,
    }
    with open(root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)


def write_builder_config(path: Path, p01_root: Path) -> None:
    config = {
        "schema_version": "phase0-census2-config-v1",
        "scene": "unit_fixture",
        "p01_root": str(p01_root),
        "output_root": str(p01_root.parent / "out"),
        "frames": {"start": P01_FRAMES[0], "end": P01_FRAMES[-1], "fps": 30.0},
        "excluded_cameras": ["cam12"],
        "min_members": 2,
        "confidence_percentile": 20.0,
        "min_valid_pixel_fraction": 0.5,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)


class ConsensusBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_builder_module()

    def test_builds_artifact_matching_consensus_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p01 = root / "p01"
            write_fake_p01(p01)
            config = root / "config.json"
            write_builder_config(config, p01)
            out = root / "evidence"
            self.builder.main(
                ["--config", str(config), "--output-dir", str(out)]
            )

            meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["schema_version"], er.CONSENSUS_SCHEMA_VERSION)
            self.assertEqual(meta["cameras"], ["cam01", "cam05"])
            self.assertEqual(meta["excluded_cameras"], ["cam12"])
            self.assertEqual(meta["frames"], P01_FRAMES)
            self.assertEqual((meta["height"], meta["width"]), (P01_HEIGHT, P01_WIDTH))
            self.assertTrue(meta["scientific"])
            self.assertEqual(meta["geometry_drift_max"], 0.0)
            self.assertEqual(
                meta["map_stats"]["total"], len(meta["cameras"]) * len(P01_FRAMES)
            )
            for camera in meta["cameras"]:
                stats = meta["per_camera_map_stats"][camera]
                self.assertEqual(stats["maps"], len(P01_FRAMES))
                self.assertEqual(stats["missing_frames"], 0)
                self.assertEqual(stats["members_min"], GROUPS_PER_FRAME)
            self.assertIn("p01_manifest_sha256", meta)
            self.assertIn("config_sha256", meta)
            self.assertEqual(meta["config"]["scene"], "unit_fixture")

            depth = np.load(out / "d.npy")
            sigma = np.load(out / "sigma.npy")
            valid = np.load(out / "valid.npy")
            self.assertEqual(depth.dtype, np.float16)
            self.assertEqual(sigma.dtype, np.float16)
            self.assertEqual(valid.dtype, np.uint8)
            self.assertEqual(
                depth.shape, (2, len(P01_FRAMES), P01_HEIGHT, P01_WIDTH)
            )

            for camera_index, camera in enumerate(meta["cameras"]):
                source_index = P01_CAMERAS.index(camera)
                for t_index, frame in enumerate(P01_FRAMES):
                    stack = np.stack(
                        [
                            member_depth(source_index, frame, group)
                            for group in range(GROUPS_PER_FRAME)
                        ]
                    )
                    if source_index == 2:
                        stack[1, 2, 2] = 0.0
                    conf = np.stack(
                        [member_confidence(source_index)] * GROUPS_PER_FRAME
                    )
                    expect_d, expect_sigma, expect_valid, _ = census.consensus_depth(
                        stack, conf, min_members=2, confidence_percentile=20.0
                    )
                    np.testing.assert_array_equal(
                        valid[camera_index, t_index], expect_valid.astype(np.uint8)
                    )
                    np.testing.assert_array_equal(
                        depth[camera_index, t_index],
                        expect_d.astype(np.float16),
                    )
                    np.testing.assert_array_equal(
                        sigma[camera_index, t_index],
                        expect_sigma.astype(np.float16),
                    )

            intrinsics = np.load(out / "intrinsics.npy")
            w2c = np.load(out / "w2c.npy")
            self.assertEqual(intrinsics.dtype, np.float64)
            self.assertEqual(w2c.dtype, np.float64)
            np.testing.assert_allclose(w2c[0, 0, 3], 0.0)
            np.testing.assert_allclose(w2c[1, 0, 3], 0.1, rtol=1e-6)

            runtime = er.EvidenceRuntime(str(out), device="cpu")
            self.assertEqual(runtime.cameras, ["cam01", "cam05"])
            verdicts = runtime.verdicts(
                torch.zeros((4, 3), dtype=torch.float32), "cam01", 0
            )
            self.assertEqual(tuple(verdicts.shape), (4,))

    def test_frame_limit_marks_non_scientific(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p01 = root / "p01"
            write_fake_p01(p01)
            config = root / "config.json"
            write_builder_config(config, p01)
            out = root / "evidence"
            self.builder.main(
                ["--config", str(config), "--output-dir", str(out), "--frame-limit", "2"]
            )
            meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
            self.assertFalse(meta["scientific"])
            self.assertEqual(meta["frames"], P01_FRAMES[:2])
            self.assertEqual(np.load(out / "d.npy").shape[1], 2)

    def test_geometry_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p01 = root / "p01"
            write_fake_p01(p01, drift_frame=2)
            config = root / "config.json"
            write_builder_config(config, p01)
            out = root / "evidence"
            with self.assertRaises(ContractError):
                self.builder.main(
                    ["--config", str(config), "--output-dir", str(out)]
                )

    def test_non_empty_output_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p01 = root / "p01"
            write_fake_p01(p01)
            config = root / "config.json"
            write_builder_config(config, p01)
            out = root / "evidence"
            out.mkdir()
            (out / "stale.npy").write_bytes(b"0")
            with self.assertRaises(ArtifactError):
                self.builder.main(["--config", str(config), "--output-dir", str(out)])
            self.builder.main(
                ["--config", str(config), "--output-dir", str(out), "--overwrite"]
            )
            self.assertTrue((out / "meta.json").is_file())

    def test_bad_config_schema_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            with open(config, "w", encoding="utf-8") as handle:
                json.dump({"schema_version": "nope"}, handle)
            with self.assertRaises(SchemaError):
                self.builder.main(
                    ["--config", str(config), "--output-dir", str(root / "out")]
                )

    def test_real_census2_config_parses(self):
        config = REPO_ROOT / "configs" / "depth_visibility" / "phase0_census2_v1.json"
        parsed = self.builder.load_builder_config(str(config))
        self.assertEqual(parsed["excluded_cameras"], ["cam12", "cam19"])
        self.assertEqual(parsed["min_members"], 3)
        self.assertEqual(parsed["confidence_percentile"], 20.0)
        self.assertNotIn("$WORK", parsed["p01_root"])


if __name__ == "__main__":
    unittest.main()
