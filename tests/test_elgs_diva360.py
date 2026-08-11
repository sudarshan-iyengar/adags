"""Unit tests for scripts/diva360_to_blender.py + elgs/diva360_schema.py.

CPU-only, no Apollo/Determined access, no torch import anywhere in this
file or its targets. Every test builds a miniature synthetic DiVa-360
sequence directory (tempfile) rather than touching the real Apollo data.

Run with:
    C:/Users/sucar/venvs/elgs-cpu/Scripts/python.exe -m unittest tests.test_elgs_diva360
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ArtifactError, ContractError, ProvenanceError, SchemaError  # noqa: E402
from elgs import diva360_schema as schema  # noqa: E402
from scripts import diva360_to_blender as converter  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

FPS = 120.0
FRAME_INDEX = 100
TRAIN_CAMERA_IDS = (0, 1, 2, 3)  # cam00 and cam04-style ids exercise held-out (mod 4)
TEST_CAMERA_IDS = (4,)


def _identity_transform(tx: float, ty: float, tz: float) -> list:
    return [
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, ty],
        [0.0, 0.0, 1.0, tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _make_frame(camera_id: int, frame_index: int = FRAME_INDEX, *, top_dir: str = "undist") -> dict:
    return {
        "file_path": f"{top_dir}/cam{camera_id:02d}/{frame_index:08d}",
        "sharpness": 100.0 + camera_id,
        "transform_matrix": _identity_transform(float(camera_id), 0.0, 3.0 + camera_id * 0.1),
        "camera_angle_x": 1.2,
        "camera_angle_y": 0.68,
        "fl_x": 800.0,
        "fl_y": 795.0,
        "is_fisheye": False,
        "cx": 580.0,
        "cy": 275.0,
        "w": 1160,
        "h": 550,
        "aabb_scale": 4,
    }


def _payload(camera_ids, frame_index: int = FRAME_INDEX, *, top_dir: str = "undist") -> dict:
    return {"frames": [_make_frame(cid, frame_index, top_dir=top_dir) for cid in camera_ids]}


def _add_tar_member(tar: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(content)
    tar.addfile(info, io.BytesIO(content))


def _write_frame_archive(
    path: Path, camera_ids, frame_index: int = FRAME_INDEX, *, top_dir: str = "frames_1"
) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for cid in camera_ids:
            name = f"{top_dir}/cam{cid:02d}/{frame_index:08d}.png"
            _add_tar_member(tar, name, f"fake-png-bytes-cam{cid}".encode("ascii"))


def _write_manifest(path: Path, sequence_name: str, digest: str = "a" * 64) -> None:
    path.write_text(f"{digest}  zips/{sequence_name}.zip\n", encoding="utf-8")


class Diva360FixtureCase(unittest.TestCase):
    """Builds one miniature valid DiVa-360 sequence directory per test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.data_root = self.root / "diva360"
        self.data_root.mkdir()
        self.sequence_name = "unlock_fixture"
        self.sequence_dir = self.data_root / self.sequence_name
        self.sequence_dir.mkdir()
        self.output_dir = self.root / "converted" / self.sequence_name

        (self.sequence_dir / "transforms_train.json").write_text(
            json.dumps(_payload(TRAIN_CAMERA_IDS)), encoding="utf-8"
        )
        (self.sequence_dir / "transforms_test.json").write_text(
            json.dumps(_payload(TEST_CAMERA_IDS)), encoding="utf-8"
        )
        _write_frame_archive(
            self.sequence_dir / "frames_1.tar.gz", TRAIN_CAMERA_IDS + TEST_CAMERA_IDS
        )
        # Decoy archive: same .png extension and per-camera shape, but a
        # DIFFERENT embedded frame index -- must never be selected.
        _write_frame_archive(
            self.sequence_dir / "segmented_gt.tar.gz",
            TRAIN_CAMERA_IDS + TEST_CAMERA_IDS,
            frame_index=FRAME_INDEX + 637,
            top_dir="segmented_gt",
        )
        _write_manifest(self.data_root / "MANIFEST.sha256", self.sequence_name)

    def build_plan(self, **overrides):
        kwargs = dict(
            sequence_dir=self.sequence_dir,
            output_dir=self.output_dir,
            num_random_points=64,
            seed=1,
        )
        kwargs.update(overrides)
        return converter.build_plan(**kwargs)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class DiscoveryTests(Diva360FixtureCase):
    def test_selects_the_matching_archive_over_the_decoy(self):
        plan, _ = self.build_plan()
        self.assertEqual(plan.archive_selection["train"].archive_path, "frames_1.tar.gz")
        self.assertEqual(plan.archive_selection["test"].archive_path, "frames_1.tar.gz")
        self.assertEqual(plan.archive_selection["train"].top_level_dir, "frames_1")
        self.assertEqual(plan.scene_top_dir, "undist")
        self.assertIn("segmented_gt.tar.gz", plan.candidate_archives)
        self.assertIn("frames_1.tar.gz", plan.candidate_archives)

    def test_never_assumes_archive_names(self):
        # Rename both archives to arbitrary names; discovery must still work
        # because it matches on content, not filename.
        (self.sequence_dir / "frames_1.tar.gz").rename(self.sequence_dir / "blob_a.tar.gz")
        (self.sequence_dir / "segmented_gt.tar.gz").rename(self.sequence_dir / "blob_b.tar.gz")
        plan, _ = self.build_plan()
        self.assertEqual(plan.archive_selection["train"].archive_path, "blob_a.tar.gz")

    def test_ambiguous_archive_match_is_rejected(self):
        # A second archive with the SAME wanted content makes discovery
        # ambiguous -- must fail closed rather than pick one arbitrarily.
        _write_frame_archive(
            self.sequence_dir / "frames_1_dup.tar.gz", TRAIN_CAMERA_IDS + TEST_CAMERA_IDS
        )
        with self.assertRaises(ContractError):
            self.build_plan()

    def test_no_archive_contains_every_frame_is_rejected(self):
        (self.sequence_dir / "frames_1.tar.gz").unlink()
        _write_frame_archive(
            self.sequence_dir / "frames_1.tar.gz", TRAIN_CAMERA_IDS  # missing cam04 (test split)
        )
        with self.assertRaises(ContractError):
            self.build_plan()


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


class DryRunTests(Diva360FixtureCase):
    def test_dry_run_prints_plan_and_touches_nothing(self):
        buffer = io.StringIO()
        argv = [
            "--sequence-dir", str(self.sequence_dir),
            "--output-dir", str(self.output_dir),
            "--num-random-points", "16",
            "--dry-run",
        ]
        with contextlib.redirect_stdout(buffer):
            exit_code = converter.main(argv)
        self.assertEqual(exit_code, 0)
        self.assertFalse(self.output_dir.exists())

        printed = json.loads(buffer.getvalue())
        self.assertEqual(printed["sequence_name"], self.sequence_name)
        self.assertEqual(printed["scene_top_dir"], "undist")
        splits = {entry["split"]: entry for entry in printed["splits"]}
        self.assertEqual(sorted(splits["train"]["camera_ids"]), list(TRAIN_CAMERA_IDS))
        self.assertEqual(splits["train"]["selected_archive"], "frames_1.tar.gz")
        self.assertEqual(sorted(printed["held_out_camera_ids_downstream_only"]), [0, 4])


# ---------------------------------------------------------------------------
# Full conversion, timestamp mapping, held-out non-exclusion
# ---------------------------------------------------------------------------


class ConversionTests(Diva360FixtureCase):
    def test_conversion_writes_expected_layout(self):
        plan, payloads = self.build_plan()
        result = converter.execute_plan(plan, payloads)

        train_path = Path(result["transforms"]["train"])
        test_path = Path(result["transforms"]["test"])
        self.assertTrue(train_path.is_file())
        self.assertTrue(test_path.is_file())

        # Relocated frame bytes land at the transforms-referenced path,
        # renamed from the archive's own top-level dir ("frames_1") to the
        # transforms JSON's top-level dir ("undist").
        for cid in TRAIN_CAMERA_IDS:
            relocated = self.output_dir / "undist" / f"cam{cid:02d}" / f"{FRAME_INDEX:08d}.png"
            self.assertTrue(relocated.is_file(), relocated)
            self.assertEqual(relocated.read_bytes(), f"fake-png-bytes-cam{cid}".encode("ascii"))

        ply_path = Path(result["points3d_ply"])
        self.assertTrue(ply_path.is_file())
        header_lines = ply_path.read_text(encoding="utf-8").splitlines()[:3]
        self.assertEqual(header_lines[0], "ply")
        self.assertEqual(header_lines[2], "element vertex 64")

        provenance_path = Path(result["provenance"])
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        self.assertEqual(provenance["schema_version"], schema.PROVENANCE_SCHEMA)
        self.assertEqual(provenance["source_sha256"], "a" * 64)
        self.assertEqual(provenance["sequence_name"], self.sequence_name)
        self.assertIn("argv", provenance["converter"])

    def test_timestamp_mapping_is_frame_index_over_fps(self):
        plan, payloads = self.build_plan(fps=FPS)
        result = converter.execute_plan(plan, payloads)
        train_payload = json.loads(Path(result["transforms"]["train"]).read_text(encoding="utf-8"))
        expected_time = FRAME_INDEX / FPS
        for frame in train_payload["frames"]:
            self.assertAlmostEqual(frame["time"], expected_time)
        # And directly against the pure schema helper both ways round.
        self.assertAlmostEqual(
            schema.frame_index_to_time(FRAME_INDEX, FPS), expected_time
        )
        self.assertEqual(schema.parse_frame_index(f"undist/cam00/{FRAME_INDEX:08d}"), FRAME_INDEX)

    def test_custom_fps_changes_the_stamped_time(self):
        plan, payloads = self.build_plan(fps=30.0)
        result = converter.execute_plan(plan, payloads)
        train_payload = json.loads(Path(result["transforms"]["train"]).read_text(encoding="utf-8"))
        for frame in train_payload["frames"]:
            self.assertAlmostEqual(frame["time"], FRAME_INDEX / 30.0)

    def test_held_out_cameras_are_not_excluded_from_the_scene(self):
        # cam00 and cam04 both satisfy the prereg's "id === 0 mod 4"
        # held-out rule. That rule is downstream-tracking-only per
        # configs/elgs/prereg_m1_census_v1.json -- the converter must still
        # emit them.
        plan, payloads = self.build_plan()
        self.assertEqual(set(plan.held_out_camera_ids), {0, 4})

        result = converter.execute_plan(plan, payloads)
        train_payload = json.loads(Path(result["transforms"]["train"]).read_text(encoding="utf-8"))
        test_payload = json.loads(Path(result["transforms"]["test"]).read_text(encoding="utf-8"))
        train_cameras = {schema.parse_camera_id(f["file_path"]) for f in train_payload["frames"]}
        test_cameras = {schema.parse_camera_id(f["file_path"]) for f in test_payload["frames"]}
        self.assertIn(0, train_cameras)  # held-out camera present in train output
        self.assertIn(4, test_cameras)  # held-out camera present in test output
        self.assertEqual(train_cameras, set(TRAIN_CAMERA_IDS))
        self.assertEqual(test_cameras, set(TEST_CAMERA_IDS))

    def test_rerun_without_overwrite_is_fail_closed(self):
        plan, payloads = self.build_plan()
        converter.execute_plan(plan, payloads)
        plan2, payloads2 = self.build_plan()
        with self.assertRaises(ArtifactError):
            converter.execute_plan(plan2, payloads2)

    def test_rerun_with_overwrite_succeeds(self):
        plan, payloads = self.build_plan()
        converter.execute_plan(plan, payloads)
        plan2, payloads2 = self.build_plan()
        result = converter.execute_plan(plan2, payloads2, overwrite=True)
        self.assertTrue(Path(result["provenance"]).is_file())


# ---------------------------------------------------------------------------
# Fail-closed schema surprises
# ---------------------------------------------------------------------------


class FailClosedTests(Diva360FixtureCase):
    def test_missing_required_split_is_rejected(self):
        (self.sequence_dir / "transforms_test.json").unlink()
        with self.assertRaises(SchemaError):
            self.build_plan()

    def test_frame_missing_transform_matrix_is_rejected(self):
        payload = _payload(TRAIN_CAMERA_IDS)
        del payload["frames"][0]["transform_matrix"]
        (self.sequence_dir / "transforms_train.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SchemaError):
            self.build_plan()

    def test_non_finite_transform_matrix_is_rejected(self):
        payload = _payload(TRAIN_CAMERA_IDS)
        payload["frames"][0]["transform_matrix"][0][0] = float("nan")
        (self.sequence_dir / "transforms_train.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SchemaError):
            self.build_plan()

    def test_frame_missing_all_intrinsics_is_rejected(self):
        payload = _payload(TRAIN_CAMERA_IDS)
        for key in ("fl_x", "fl_y", "cx", "cy", "camera_angle_x"):
            payload["frames"][0].pop(key, None)
        (self.sequence_dir / "transforms_train.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SchemaError):
            self.build_plan()

    def test_preexisting_time_key_in_source_is_rejected(self):
        payload = _payload(TRAIN_CAMERA_IDS)
        payload["frames"][0]["time"] = 0.0
        (self.sequence_dir / "transforms_train.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SchemaError):
            self.build_plan()

    def test_duplicate_file_path_is_rejected(self):
        payload = _payload(TRAIN_CAMERA_IDS)
        payload["frames"].append(dict(payload["frames"][0]))
        (self.sequence_dir / "transforms_train.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ContractError):
            self.build_plan()

    def test_missing_manifest_is_rejected(self):
        (self.data_root / "MANIFEST.sha256").unlink()
        with self.assertRaises(ProvenanceError):
            self.build_plan()

    def test_manifest_missing_sequence_entry_is_rejected(self):
        _write_manifest(self.data_root / "MANIFEST.sha256", "some_other_sequence")
        with self.assertRaises(ProvenanceError):
            self.build_plan()

    def test_malformed_manifest_line_is_rejected(self):
        (self.data_root / "MANIFEST.sha256").write_text("not-a-valid-manifest-line\n", encoding="utf-8")
        with self.assertRaises(SchemaError):
            self.build_plan()

    def test_no_archives_present_is_rejected(self):
        (self.sequence_dir / "frames_1.tar.gz").unlink()
        (self.sequence_dir / "segmented_gt.tar.gz").unlink()
        with self.assertRaises(ContractError):
            self.build_plan()

    def test_disagreeing_top_level_dirs_across_splits_is_rejected(self):
        payload = _payload(TEST_CAMERA_IDS, top_dir="undist_other")
        (self.sequence_dir / "transforms_test.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ContractError):
            self.build_plan()


# ---------------------------------------------------------------------------
# Pure schema.py helpers
# ---------------------------------------------------------------------------


class SchemaHelperTests(unittest.TestCase):
    def test_frame_index_to_time(self):
        self.assertAlmostEqual(schema.frame_index_to_time(120, 120.0), 1.0)
        self.assertAlmostEqual(schema.frame_index_to_time(0, 120.0), 0.0)
        with self.assertRaises(SchemaError):
            schema.frame_index_to_time(-1, 120.0)
        with self.assertRaises(SchemaError):
            schema.frame_index_to_time(10, 0.0)

    def test_held_out_camera_ids_is_mod_four(self):
        ids = range(0, 12)
        self.assertEqual(schema.held_out_camera_ids(ids), {0, 4, 8})

    def test_parse_camera_id_and_frame_index(self):
        self.assertEqual(schema.parse_camera_id("undist/cam23/00001000.png"), 23)
        self.assertEqual(schema.parse_frame_index("undist/cam23/00001000"), 1000)
        with self.assertRaises(SchemaError):
            schema.parse_camera_id("undist/nocam/00001000.png")

    def test_split_top_level_dir(self):
        self.assertEqual(schema.split_top_level_dir("undist/cam01/x.png"), ("undist", "cam01/x.png"))
        with self.assertRaises(SchemaError):
            schema.split_top_level_dir("no_slash_here")

    def test_frame_relative_path_rejects_unsafe_paths(self):
        self.assertEqual(schema.frame_relative_path("undist/cam01/00000000"), "undist/cam01/00000000.png")
        with self.assertRaises(SchemaError):
            schema.frame_relative_path("../escape/cam01/00000000")
        with self.assertRaises(SchemaError):
            schema.frame_relative_path("undist\\cam01\\00000000")

    def test_parse_sha256_manifest_happy_path(self):
        digest = "b" * 64
        text = f"{digest}  zips/battery.zip\n{digest}  zips/unlock.zip\n"
        entries = schema.parse_sha256_manifest(text)
        self.assertEqual(entries["zips/battery.zip"], digest)
        self.assertEqual(schema.lookup_source_sha256(entries, "battery"), digest)

    def test_parse_sha256_manifest_rejects_malformed_digest(self):
        with self.assertRaises(SchemaError):
            schema.parse_sha256_manifest("not-hex  zips/battery.zip\n")

    def test_lookup_source_sha256_missing_entry(self):
        with self.assertRaises(ProvenanceError):
            schema.lookup_source_sha256({}, "battery")

    def test_stamp_frame_time_rejects_double_stamp(self):
        frame = _make_frame(0)
        stamped = schema.stamp_frame_time(frame, fps=FPS)
        self.assertIn("time", stamped)
        with self.assertRaises(SchemaError):
            schema.stamp_frame_time(stamped, fps=FPS)


# ---------------------------------------------------------------------------
# points3d.ply synthesis
# ---------------------------------------------------------------------------


class PointCloudSynthesisTests(unittest.TestCase):
    def test_frustum_union_bounding_box_is_finite_and_nonempty(self):
        frames = [_make_frame(cid) for cid in TRAIN_CAMERA_IDS]
        bbox_min, bbox_max = converter.frustum_union_bounding_box(frames)
        self.assertTrue((bbox_max > bbox_min).all())

    def test_frustum_union_bounding_box_rejects_empty_input(self):
        with self.assertRaises(ContractError):
            converter.frustum_union_bounding_box([])

    def test_sample_random_points_are_inside_the_box(self):
        frames = [_make_frame(cid) for cid in TRAIN_CAMERA_IDS]
        bbox_min, bbox_max = converter.frustum_union_bounding_box(frames)
        points, colors = converter.sample_random_points(bbox_min, bbox_max, 128, seed=7)
        self.assertEqual(points.shape, (128, 3))
        self.assertEqual(colors.shape, (128, 3))
        self.assertTrue((points >= bbox_min).all())
        self.assertTrue((points <= bbox_max).all())

    def test_write_points3d_ply_round_trips_vertex_count(self):
        frames = [_make_frame(cid) for cid in TRAIN_CAMERA_IDS]
        bbox_min, bbox_max = converter.frustum_union_bounding_box(frames)
        points, colors = converter.sample_random_points(bbox_min, bbox_max, 10, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            ply_path = Path(tmp) / "points3d.ply"
            converter.write_points3d_ply(ply_path, points, colors)
            lines = ply_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "ply")
            self.assertIn("element vertex 10", lines)
            self.assertEqual(lines[-1].count(" "), 8)  # x y z nx ny nz r g b


if __name__ == "__main__":
    unittest.main()
