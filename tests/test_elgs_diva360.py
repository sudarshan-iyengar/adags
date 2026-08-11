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


# --- --window mode fixtures -------------------------------------------------

WINDOW_START = 0
WINDOW_END = 60
WINDOW_STRIDE = 20
WINDOW_INDICES = tuple(range(WINDOW_START, WINDOW_END + 1, WINDOW_STRIDE))  # (0, 20, 40, 60)


def _photographic_png_bytes() -> bytes:
    """A tiny genuinely-decodable RGB PNG with >2 distinct grayscale values
    (the frame-vs-mask content probe classifies it as photographic)."""

    from PIL import Image

    image = Image.new("RGB", (4, 4))
    image.putdata([(16 * i, 8 * i, 4 * i) for i in range(16)])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _binary_mask_png_bytes() -> bytes:
    """A tiny genuinely-decodable {0,255} mask PNG (probe: binary mask)."""

    from PIL import Image

    image = Image.new("L", (4, 4))
    image.putdata([255 if i % 2 else 0 for i in range(16)])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_multi_index_archive(
    path: Path,
    camera_frame_indices: dict,
    *,
    top_dir: str,
    extension: str = ".png",
    content_prefix: str = "fake-bytes",
    content_bytes: bytes | None = None,
) -> None:
    """camera_frame_indices: {camera_id: [frame_index, ...]}.

    ``content_bytes``, when given, is written verbatim for EVERY member
    (used to make members genuinely decodable for the content probe);
    otherwise members carry unique ascii placeholders.
    """

    with tarfile.open(path, "w:gz") as tar:
        for cid, indices in camera_frame_indices.items():
            for idx in indices:
                name = f"{top_dir}/cam{cid:02d}/{idx:08d}{extension}"
                payload = (
                    content_bytes
                    if content_bytes is not None
                    else f"{content_prefix}-cam{cid}-{idx}".encode("ascii")
                )
                _add_tar_member(tar, name, payload)


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


class WindowFixtureCase(Diva360FixtureCase):
    """Adds a temporal frame archive + segmented_ngp mask archive on top of
    the base single-instant fixture, for --window mode coverage."""

    ALL_CAMERA_IDS = TRAIN_CAMERA_IDS + TEST_CAMERA_IDS

    def setUp(self) -> None:
        super().setUp()
        # Full window coverage for every camera, PLUS the original
        # single-instant index -- proves --window discovery doesn't
        # depend on the archive being window-only, and that the same
        # sequence dir still supports single-instant mode unchanged.
        # Members are genuinely decodable photographic PNGs because the
        # real mask archive below MIRRORS the frame layout (as on real
        # Apollo data, det task 72bf9fd6), so frame-source discovery must
        # disambiguate by decoded pixel content, not by path matching.
        frame_indices = {
            cid: [FRAME_INDEX, *WINDOW_INDICES] for cid in self.ALL_CAMERA_IDS
        }
        (self.sequence_dir / "frames_1.tar.gz").unlink()
        _write_multi_index_archive(
            self.sequence_dir / "frames_1.tar.gz",
            frame_indices,
            top_dir="frames_1",
            content_bytes=_photographic_png_bytes(),
        )
        # The real per-frame fg/bg masks the census consumes (per the M1
        # census record's "INPUT MAPPING" note) -- full coverage of the
        # window AND the single-instant index, mirroring real data.
        mask_indices = {
            cid: [FRAME_INDEX, *WINDOW_INDICES] for cid in self.ALL_CAMERA_IDS
        }
        _write_multi_index_archive(
            self.sequence_dir / "segmented_ngp.tar.gz",
            mask_indices,
            top_dir="segmented_ngp",
            content_bytes=_binary_mask_png_bytes(),
        )

    def build_window_plan(self, start=WINDOW_START, end=WINDOW_END, stride=WINDOW_STRIDE, **overrides):
        kwargs = dict(
            sequence_dir=self.sequence_dir,
            output_dir=self.output_dir,
            num_random_points=64,
            seed=1,
            window=(start, end, stride),
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

    def test_default_mode_has_no_window_plan(self):
        # Single-instant mode (the default, no --window) must carry no
        # window state at all -- the plan/provenance shape is unchanged.
        plan, payloads = self.build_plan()
        self.assertIsNone(plan.window)
        self.assertIsNone(plan.window_splits)
        result = converter.execute_plan(plan, payloads)
        provenance = json.loads(Path(result["provenance"]).read_text(encoding="utf-8"))
        self.assertNotIn("window", provenance)
        self.assertNotIn("masks_dir", result)


# ---------------------------------------------------------------------------
# --window mode: temporal cross of static calibration with frame indices
# ---------------------------------------------------------------------------


class WindowModeTests(WindowFixtureCase):
    def test_window_emission_counts_and_layout(self):
        plan, payloads = self.build_window_plan()
        self.assertEqual(plan.window.start, WINDOW_START)
        self.assertEqual(plan.window.end, WINDOW_END)
        self.assertEqual(plan.window.stride, WINDOW_STRIDE)
        for split, ws in plan.window_splits.items():
            for cam_id, paths in ws.camera_relative_paths.items():
                self.assertEqual(len(paths), len(WINDOW_INDICES), (split, cam_id))

        result = converter.execute_plan(plan, payloads)
        train_payload = json.loads(Path(result["transforms"]["train"]).read_text(encoding="utf-8"))
        test_payload = json.loads(Path(result["transforms"]["test"]).read_text(encoding="utf-8"))
        self.assertEqual(len(train_payload["frames"]), len(TRAIN_CAMERA_IDS) * len(WINDOW_INDICES))
        self.assertEqual(len(test_payload["frames"]), len(TEST_CAMERA_IDS) * len(WINDOW_INDICES))

        # Every (camera, index) pair in the window is present exactly once,
        # and every emitted frame reuses that camera's static pose.
        seen = set()
        for frame in train_payload["frames"] + test_payload["frames"]:
            cam_id = schema.parse_camera_id(frame["file_path"])
            frame_index = schema.parse_frame_index(frame["file_path"])
            self.assertIn(frame_index, WINDOW_INDICES)
            key = (cam_id, frame_index)
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertEqual(
                frame["transform_matrix"],
                _make_frame(cam_id)["transform_matrix"],
                "window frame must reuse the camera's static single-instant pose",
            )
        self.assertEqual(len(seen), (len(TRAIN_CAMERA_IDS) + len(TEST_CAMERA_IDS)) * len(WINDOW_INDICES))

        # Relocated frame bytes land under the SAME scene_top_dir ("undist")
        # single-instant mode uses -- not a window-specific directory.
        relocated = self.output_dir / "undist" / "cam00" / f"{WINDOW_INDICES[0]:08d}.png"
        self.assertTrue(relocated.is_file())
        self.assertEqual(relocated.read_bytes(), _photographic_png_bytes())

    def test_window_time_values(self):
        plan, payloads = self.build_window_plan()
        result = converter.execute_plan(plan, payloads)
        train_payload = json.loads(Path(result["transforms"]["train"]).read_text(encoding="utf-8"))
        by_index = {schema.parse_frame_index(f["file_path"]): f["time"] for f in train_payload["frames"]}
        for index in WINDOW_INDICES:
            self.assertAlmostEqual(by_index[index], index / FPS)

    def test_window_mask_extraction(self):
        plan, payloads = self.build_window_plan()
        result = converter.execute_plan(plan, payloads)
        masks_dir = Path(result["masks_dir"])
        self.assertTrue(masks_dir.is_dir())
        for cam_id in self.ALL_CAMERA_IDS:
            for index in WINDOW_INDICES:
                mask_path = masks_dir / f"cam{cam_id:02d}" / f"{index:08d}.png"
                self.assertTrue(mask_path.is_file(), mask_path)
                self.assertEqual(mask_path.read_bytes(), _binary_mask_png_bytes())
        expected_mask_count = len(self.ALL_CAMERA_IDS) * len(WINDOW_INDICES)
        provenance = json.loads(Path(result["provenance"]).read_text(encoding="utf-8"))
        total_masks = sum(s["mask_count"] for s in provenance["window"]["splits"].values())
        self.assertEqual(total_masks, expected_mask_count)

    def test_window_provenance_gains_bounds_and_counts(self):
        plan, payloads = self.build_window_plan()
        result = converter.execute_plan(plan, payloads)
        provenance = json.loads(Path(result["provenance"]).read_text(encoding="utf-8"))
        window = provenance["window"]
        self.assertEqual(window["start"], WINDOW_START)
        self.assertEqual(window["end"], WINDOW_END)
        self.assertEqual(window["stride"], WINDOW_STRIDE)
        train_split = window["splits"]["train"]
        self.assertEqual(train_split["total_frame_count"], len(TRAIN_CAMERA_IDS) * len(WINDOW_INDICES))
        for cam_id in TRAIN_CAMERA_IDS:
            self.assertEqual(train_split["per_camera_frame_counts"][str(cam_id)], len(WINDOW_INDICES))
        self.assertEqual(train_split["mask_count"], len(TRAIN_CAMERA_IDS) * len(WINDOW_INDICES))

    def test_window_dry_run_touches_nothing(self):
        buffer = io.StringIO()
        argv = [
            "--sequence-dir", str(self.sequence_dir),
            "--output-dir", str(self.output_dir),
            "--num-random-points", "8",
            "--window", str(WINDOW_START), str(WINDOW_END),
            "--stride", str(WINDOW_STRIDE),
            "--dry-run",
        ]
        with contextlib.redirect_stdout(buffer):
            exit_code = converter.main(argv)
        self.assertEqual(exit_code, 0)
        self.assertFalse(self.output_dir.exists())
        printed = json.loads(buffer.getvalue())
        self.assertEqual(printed["window"]["start"], WINDOW_START)
        self.assertEqual(
            printed["window"]["splits"]["train"]["total_frame_count"],
            len(TRAIN_CAMERA_IDS) * len(WINDOW_INDICES),
        )

    def test_window_cli_end_to_end(self):
        buffer = io.StringIO()
        argv = [
            "--sequence-dir", str(self.sequence_dir),
            "--output-dir", str(self.output_dir),
            "--num-random-points", "8",
            "--window", str(WINDOW_START), str(WINDOW_END),
            "--stride", str(WINDOW_STRIDE),
        ]
        with contextlib.redirect_stdout(buffer):
            exit_code = converter.main(argv)
        self.assertEqual(exit_code, 0)
        printed = json.loads(buffer.getvalue())
        self.assertTrue(Path(printed["masks_dir"]).is_dir())

    def test_default_stride_is_one(self):
        plan, _ = self.build_window_plan(start=0, end=2, stride=1)
        for ws in plan.window_splits.values():
            for paths in ws.camera_relative_paths.values():
                indices = sorted(int(p.rsplit("/", 1)[-1].split(".")[0]) for p in paths)
                # Only frame 0 is available at stride 1 within [0, 2] for
                # this fixture (indices 1 and 2 were never written) --
                # partial per-camera coverage is expected and NOT a
                # failure as long as it's non-empty.
                self.assertEqual(indices, [0])


class WindowFailClosedTests(WindowFixtureCase):
    def test_camera_with_zero_window_frames_is_rejected(self):
        # cam02 (a train camera) gets NO entries at all in the requested
        # window -- every other camera keeps full coverage.
        frame_indices = {
            cid: [FRAME_INDEX, *WINDOW_INDICES] for cid in self.ALL_CAMERA_IDS if cid != 2
        }
        frame_indices[2] = [FRAME_INDEX]  # present at the single instant only
        (self.sequence_dir / "frames_1.tar.gz").unlink()
        _write_multi_index_archive(
            self.sequence_dir / "frames_1.tar.gz", frame_indices, top_dir="frames_1"
        )
        with self.assertRaisesRegex(ContractError, r"\[2\]|\b2\b"):
            self.build_window_plan()

    def test_missing_mask_for_a_kept_frame_is_rejected_with_itemized_list(self):
        # frames_1 has full window coverage, but segmented_ngp is missing
        # exactly one (camera, index) pair the frame archive kept.
        mask_indices = {cid: list(WINDOW_INDICES) for cid in self.ALL_CAMERA_IDS}
        mask_indices[0] = [i for i in WINDOW_INDICES if i != WINDOW_INDICES[-1]]
        (self.sequence_dir / "segmented_ngp.tar.gz").unlink()
        _write_multi_index_archive(
            self.sequence_dir / "segmented_ngp.tar.gz", mask_indices, top_dir="segmented_ngp"
        )
        expected_missing = f"cam00/{WINDOW_INDICES[-1]:08d}.png"
        with self.assertRaisesRegex(ContractError, "expected member.*not found"):
            self.build_window_plan()
        # Confirm the itemized message really names the missing pair.
        try:
            self.build_window_plan()
            self.fail("expected ContractError")
        except ContractError as exc:
            self.assertIn(expected_missing, str(exc))

    def test_ambiguous_mask_archive_is_rejected(self):
        # A second archive, distinct from frames_1, that ALSO plausibly
        # holds the requested masks -- must fail closed rather than guess.
        mask_indices = {cid: list(WINDOW_INDICES) for cid in self.ALL_CAMERA_IDS}
        _write_multi_index_archive(
            self.sequence_dir / "segmented_ngp_dup.tar.gz", mask_indices, top_dir="segmented_ngp_dup"
        )
        with self.assertRaises(ContractError):
            self.build_window_plan()

    def test_no_mask_candidates_after_excluding_frame_archive_is_rejected(self):
        # Remove every archive except the one already used as the frame
        # source -- nothing is left to plausibly serve as the mask source.
        (self.sequence_dir / "segmented_gt.tar.gz").unlink()
        (self.sequence_dir / "segmented_ngp.tar.gz").unlink()
        with self.assertRaises(ContractError):
            self.build_window_plan()

    def test_invalid_window_bounds_are_rejected(self):
        with self.assertRaises(SchemaError):
            self.build_window_plan(start=10, end=5, stride=1)
        with self.assertRaises(SchemaError):
            self.build_window_plan(start=0, end=10, stride=0)


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

    def test_window_indices_is_inclusive_both_ends(self):
        self.assertEqual(schema.window_indices(0, 60, 20), (0, 20, 40, 60))
        self.assertEqual(schema.window_indices(5, 5, 1), (5,))
        self.assertEqual(schema.window_indices(0, 10, 3), (0, 3, 6, 9))

    def test_window_indices_rejects_bad_bounds(self):
        with self.assertRaises(SchemaError):
            schema.window_indices(-1, 10)
        with self.assertRaises(SchemaError):
            schema.window_indices(10, 5)
        with self.assertRaises(SchemaError):
            schema.window_indices(0, 10, 0)
        with self.assertRaises(SchemaError):
            schema.window_indices(0, 10, -2)

    def test_camera_path_template(self):
        self.assertEqual(
            schema.camera_path_template("undist/cam01/00001000"), ("undist", "cam01", 8)
        )
        with self.assertRaises(SchemaError):
            schema.camera_path_template("undist/cam01/notdigits")
        with self.assertRaises(SchemaError):
            schema.camera_path_template("no_slash_here")

    def test_window_file_path(self):
        self.assertEqual(
            schema.window_file_path("undist", "cam01", 8, 60), "undist/cam01/00000060"
        )
        with self.assertRaises(SchemaError):
            schema.window_file_path("undist", "cam01", 8, -1)
        with self.assertRaises(SchemaError):
            schema.window_file_path("undist", "cam01", 2, 1000)  # overflows 2-digit width


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


# ---------------------------------------------------------------------------
# Real-schema convention: shipped file_path INCLUDES the extension
# (measured on Apollo unlock 2026-08-11, det task 85424662:
# "undist/cam01/00000000.png") while the ADAGS reader contract requires
# extension-LESS converted output. These tests pin the normalization fix.
# ---------------------------------------------------------------------------


def _append_extension_to_transforms(sequence_dir: Path, extension: str = ".png") -> None:
    for name in ("transforms_train.json", "transforms_test.json"):
        path = sequence_dir / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        for frame in payload["frames"]:
            frame["file_path"] = frame["file_path"] + extension
        path.write_text(json.dumps(payload), encoding="utf-8")


class RealSchemaHelperTests(unittest.TestCase):
    def test_normalize_strips_extension_and_passes_canonical(self):
        self.assertEqual(
            schema.normalize_source_file_path("undist/cam01/00000000.png"),
            "undist/cam01/00000000",
        )
        self.assertEqual(
            schema.normalize_source_file_path("undist/cam01/00000000"),
            "undist/cam01/00000000",
        )

    def test_frame_relative_path_identical_for_both_conventions(self):
        self.assertEqual(
            schema.frame_relative_path("undist/cam01/00000000.png"),
            schema.frame_relative_path("undist/cam01/00000000"),
        )
        self.assertEqual(
            schema.frame_relative_path("undist/cam01/00000000.png"),
            "undist/cam01/00000000.png",
        )

    def test_parse_frame_index_both_conventions(self):
        self.assertEqual(schema.parse_frame_index("undist/cam01/00000123.png"), 123)
        self.assertEqual(schema.parse_frame_index("undist/cam01/00000123"), 123)

    def test_camera_path_template_both_conventions(self):
        self.assertEqual(
            schema.camera_path_template("undist/cam01/00000000.png"), ("undist", "cam01", 8)
        )
        self.assertEqual(
            schema.camera_path_template("undist/cam01/00000000"), ("undist", "cam01", 8)
        )

    def test_stamp_frame_time_emits_extension_less_output(self):
        frame = _make_frame(1)
        frame["file_path"] = frame["file_path"] + ".png"
        stamped = schema.stamp_frame_time(frame, fps=FPS)
        self.assertEqual(stamped["file_path"], _make_frame(1)["file_path"])
        self.assertAlmostEqual(stamped["time"], FRAME_INDEX / FPS)


class MirroredMaskAmbiguityTests(WindowFixtureCase):
    """Real-data condition (det task 72bf9fd6): the per-frame mask archive
    mirrors the frame archive's member layout exactly, so BOTH fully cover
    the single-instant referenced paths and the content probe must decide."""

    def test_single_instant_discovery_resolves_via_content_probe(self):
        plan, _ = self.build_plan()
        self.assertEqual(plan.archive_selection["train"].archive_path, "frames_1.tar.gz")
        self.assertEqual(plan.archive_selection["test"].archive_path, "frames_1.tar.gz")

    def test_window_frame_discovery_resolves_via_content_probe(self):
        plan, _ = self.build_window_plan()
        self.assertEqual(plan.archive_selection["train"].archive_path, "frames_1.tar.gz")
        for window_split in plan.window_splits.values():
            self.assertEqual(window_split.mask_archive.archive_path, "segmented_ngp.tar.gz")

    def test_sparse_gt_hit_disambiguated_by_full_coverage(self):
        # segmented_gt holding ONE window index (a real sparse-GT overlap)
        # hits the mask probe but cannot cover the window; segmented_ngp
        # must still be selected, by full coverage.
        (self.sequence_dir / "segmented_gt.tar.gz").unlink()
        _write_multi_index_archive(
            self.sequence_dir / "segmented_gt.tar.gz",
            {cid: [WINDOW_INDICES[0]] for cid in self.ALL_CAMERA_IDS},
            top_dir="segmented_gt",
            content_bytes=_binary_mask_png_bytes(),
        )
        plan, _ = self.build_window_plan()
        for window_split in plan.window_splits.values():
            self.assertEqual(window_split.mask_archive.archive_path, "segmented_ngp.tar.gz")


class RealSchemaSingleInstantTests(Diva360FixtureCase):
    def setUp(self) -> None:
        super().setUp()
        _append_extension_to_transforms(self.sequence_dir)

    def test_discovery_and_conversion_with_extension_included_source(self):
        plan, payloads = self.build_plan()
        self.assertEqual(plan.archive_selection["train"].archive_path, "frames_1.tar.gz")
        result = converter.execute_plan(plan, payloads)
        train_payload = json.loads(Path(result["transforms"]["train"]).read_text(encoding="utf-8"))
        for frame in train_payload["frames"]:
            self.assertFalse(
                frame["file_path"].endswith(".png"),
                msg=f"converted output must be extension-less: {frame['file_path']!r}",
            )
        for cid in TRAIN_CAMERA_IDS:
            relocated = self.output_dir / "undist" / f"cam{cid:02d}" / f"{FRAME_INDEX:08d}.png"
            self.assertTrue(relocated.is_file(), relocated)


class RealSchemaWindowTests(WindowFixtureCase):
    def setUp(self) -> None:
        super().setUp()
        _append_extension_to_transforms(self.sequence_dir)

    def test_window_mode_with_extension_included_source(self):
        plan, payloads = self.build_window_plan()
        result = converter.execute_plan(plan, payloads)
        train_payload = json.loads(Path(result["transforms"]["train"]).read_text(encoding="utf-8"))
        emitted = {frame["file_path"] for frame in train_payload["frames"]}
        for cid in TRAIN_CAMERA_IDS:
            for idx in WINDOW_INDICES:
                self.assertIn(f"undist/cam{cid:02d}/{idx:08d}", emitted)
        for frame in train_payload["frames"]:
            self.assertFalse(frame["file_path"].endswith(".png"))
        for cid in TRAIN_CAMERA_IDS:
            for idx in WINDOW_INDICES:
                mask = self.output_dir / "masks" / f"cam{cid:02d}" / f"{idx:08d}.png"
                self.assertTrue(mask.is_file(), mask)


if __name__ == "__main__":
    unittest.main()
