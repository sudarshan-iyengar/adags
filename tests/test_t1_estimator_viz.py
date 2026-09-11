"""Unit tests for scripts/t1_estimator_viz.py.

Everything here runs on a synthetic scene and a synthetic report: a 2-cell
grid holding three candidate groups, one of them gated, and a pinhole camera
whose projection is hand-checkable. No torch, no CUDA, no data.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO_ROOT, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import t1_estimator_viz as viz  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic fixture
# ---------------------------------------------------------------------------

CELLS = 2
LO = [0.0, 0.0, 0.0]
SPAN = [2.0, 2.0, 2.0]
WIDTH = 200
HEIGHT = 200
FOCAL = 200.0
PRINCIPAL = 100.0
#: camera centre; with the Blender->OpenCV flip this looks down -Z in world
#: space, so every point of the grid box is in front of it.
CENTRE = [1.0, 1.0, 5.0]

#: (cell key, number of rows) for the cells the synthetic cloud populates.
#: Key 2 is deliberately below the row floor, so it must be dropped.
CELL_ROWS = [(0, 5), (4, 7), (7, 6), (2, 2)]
MIN_ROWS = 4
#: after the floor, kept keys ascend to 0, 4, 7 -> groups 0, 1, 2.
EXPECTED_KEPT_KEYS = [0, 4, 7]
EXPECTED_ROWS = [5, 7, 6]

FRAMES = [4, 6, 8]
TRUTH = [5, 7]


def cell_centre_points(key, count):
    """`count` points jittered inside the cell `key`, all strictly interior."""
    low, high = viz.cell_box(key, CELLS, LO, SPAN)
    centre = 0.5 * (np.asarray(low) + np.asarray(high))
    offsets = np.linspace(-0.2, 0.2, count)
    points = np.tile(centre, (count, 1))
    points[:, 0] += offsets
    return points


def synthetic_cloud():
    return np.concatenate([cell_centre_points(key, count)
                           for key, count in CELL_ROWS],
                          axis=0).astype(np.float32)


def synthetic_decisions():
    """Three groups: 0 abstains on contrast, 1 is gated, 2 abstains on shape."""
    return [
        {
            "group": 0, "rows": EXPECTED_ROWS[0], "gated": False,
            "offset_frame": None, "onset_frame": None, "gap_seconds": None,
            "abstain_reason": "contrast", "agreeing_cameras": 0,
            "per_camera": {},
            "contrast": {"n_high": 40, "n_low": 40, "separation": 0.0013,
                         "within_mode_scale": 0.0004, "midpoint": 0.06,
                         "range": 0.0035},
        },
        {
            "group": 1, "rows": EXPECTED_ROWS[1], "gated": True,
            "offset_frame": 5, "onset_frame": 8, "gap_seconds": [0.1, 0.25],
            "abstain_reason": None, "agreeing_cameras": 3,
            "per_camera": {
                "1": {"offset_frame": 5, "onset_frame": 8, "reason": None},
                "6": {"offset_frame": 5, "onset_frame": 8, "reason": None},
                "11": {"offset_frame": 6, "onset_frame": 8, "reason": None},
                "16": {"offset_frame": None, "onset_frame": None,
                       "reason": "no_interior_gap"},
            },
            "contrast": {"n_high": 50, "n_low": 30, "separation": 0.069,
                         "within_mode_scale": 0.0029, "midpoint": 0.027,
                         "range": 0.103},
        },
        {
            "group": 2, "rows": EXPECTED_ROWS[2], "gated": False,
            "offset_frame": None, "onset_frame": None, "gap_seconds": None,
            "abstain_reason": "no_interior_gap", "agreeing_cameras": 0,
            "per_camera": {},
            "contrast": {"n_high": 10, "n_low": 70, "separation": 0.01,
                         "within_mode_scale": 0.002, "midpoint": 0.03,
                         "range": 0.02},
        },
    ]


def synthetic_report(n_rows):
    decisions = synthetic_decisions()
    return {
        "schema": viz.REPORT_SCHEMA,
        "checkpoint": "synthetic",
        "config": "synthetic",
        "source_path": "synthetic",
        "program_sha256": "0" * 64,
        "program": {
            "schema": viz.REPORT_SCHEMA + "/program",
            "n_groups": len(decisions),
            "groups": [{k: v for k, v in d.items() if k != "per_camera"}
                       for d in decisions],
        },
        "grouping": {
            "method": "voxel_grid_over_cloud_percentile_box",
            "cells_per_axis": CELLS, "grid_percentile": [1.0, 99.0],
            "grid_lo": LO, "grid_span": SPAN, "min_group_rows": MIN_ROWS,
            "n_groups": len(decisions), "n_rows": int(n_rows),
        },
        "sampling": {
            "train_camera_ids_available": [1, 6, 11, 16],
            "train_camera_ids_used": [1, 6, 11, 16],
            "n_frames": 12, "frame_range": [2, 10], "coarse_stride": 2,
            "coarse_frames": [2, 4, 6, 8, 10],
            "fine_frames": [5, 7],
            "evaluated_frames": [2, 4, 5, 6, 7, 8, 10],
        },
        "decision_rule": {},
        "render_counts": {"base_renders": 1, "ablated_renders": 1},
        "diagnostics": {"decisions": decisions},
        "timing": {"estimation_seconds": 1.0, "seconds_per_render": 0.5},
        "anti_leakage": {},
        "scoring": {"skipped": True, "reason": "synthetic"},
    }


def synthetic_program():
    return {
        "schema_version": "adags-episode-program-v2",
        "membership_mode": "row_ids",
        "spatial": {"kind": "voxel_grid", "cells_per_axis": CELLS,
                    "lo": LO, "span": SPAN,
                    "group_cell_keys": {"1": [EXPECTED_KEPT_KEYS[1]]}},
        "groups": [{"group": 1, "offset_frame": 5, "onset_frame": 8}],
        "cloud": {"n_rows": 20},
    }


def synthetic_transforms():
    matrix = np.eye(4)
    matrix[:3, 3] = CENTRE
    return {
        "w": float(WIDTH), "h": float(HEIGHT),
        "fl_x": FOCAL, "fl_y": FOCAL, "cx": PRINCIPAL, "cy": PRINCIPAL,
        "frames": [{"file_path": "images/cam00_%04d" % frame,
                    "time": frame / 30.0,
                    "transform_matrix": matrix.tolist()}
                   for frame in range(12)],
    }


@pytest.fixture()
def scene(tmp_path):
    """A complete synthetic input set on disk; returns the paths."""
    from PIL import Image

    root = tmp_path / "scene"
    images = root / "images"
    images.mkdir(parents=True)
    (root / "transforms_test.json").write_text(json.dumps(synthetic_transforms()))
    (root / "transforms_train.json").write_text(json.dumps({
        "w": float(WIDTH), "h": float(HEIGHT), "fl_x": FOCAL, "fl_y": FOCAL,
        "cx": PRINCIPAL, "cy": PRINCIPAL, "frames": []}))
    rng = np.random.default_rng(0)
    for frame in range(12):
        pixels = rng.integers(40, 200, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(images / ("cam00_%04d.png" % frame))

    cloud = synthetic_cloud()
    xyz_path = tmp_path / "xyz.npy"
    np.save(xyz_path, cloud)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(synthetic_report(cloud.shape[0])))
    program_path = tmp_path / "program.json"
    program_path.write_text(json.dumps(synthetic_program()))
    return {
        "root": root, "xyz": xyz_path, "report": report_path,
        "program": program_path, "out": tmp_path / "out.jpg", "tmp": tmp_path,
    }


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------


def test_camera_projects_known_points_to_expected_pixels(scene):
    camera = viz.load_camera(scene["root"], "cam00")
    assert camera.width == WIDTH and camera.height == HEIGHT
    assert camera.fl_x == pytest.approx(FOCAL)
    assert camera.cx == pytest.approx(PRINCIPAL)

    # With c2w = I translated to CENTRE and the Blender->OpenCV flip, the
    # camera frame is x_cam = X - 1, y_cam = 1 - Y, z_cam = 5 - Z; the pixel
    # is (f * x_cam / z_cam + c) * (W - 1) / W (see the module docstring).
    points = np.array([
        [1.0, 1.0, 0.0],    # on the optical axis, depth 5
        [1.5, 1.0, 0.0],    # x_cam = 0.5 -> 200 * 0.1 + 100 = 120
        [1.0, 1.5, 0.0],    # y_cam = -0.5 -> 200 * -0.1 + 100 = 80
    ])
    xy, valid, w = camera.project(points)
    assert bool(np.all(valid))
    assert w == pytest.approx([5.0, 5.0, 5.0])
    expected = np.array([[100.0, 100.0], [120.0, 100.0], [100.0, 80.0]])
    expected *= (WIDTH - 1) / float(WIDTH)
    assert xy == pytest.approx(expected, abs=1e-9)


def test_points_behind_the_camera_have_a_negative_w(scene):
    """`valid` mirrors the estimator's own rule, which does NOT test w > 0.

    `project_points_to_grid` (utils/motion_prior_utils.py:133-139) accepts any
    |w| >= 1e-6 whose NDC lands inside [-1, 1], so a point directly behind the
    camera on the optical axis passes it. That quirk is reproduced here rather
    than silently fixed; `cell_polygon` is what refuses such a box.
    """
    camera = viz.load_camera(scene["root"], "cam00")
    _xy, valid, w = camera.project(np.array([[1.0, 1.0, 20.0]]))
    assert w[0] < 0.0
    assert bool(valid[0])
    # a cell wholly behind the camera has no bounded projection
    assert viz.cell_polygon(camera, 0, CELLS, [0.0, 0.0, 10.0], SPAN) is None


def test_cell_polygon_of_a_known_box(scene):
    camera = viz.load_camera(scene["root"], "cam00")
    # Key 7 is the cell [1,2] x [1,2] x [1,2]; its nearest face sits at
    # z_cam = 3, its far face at z_cam = 4, so the hull is the near face.
    polygon = viz.cell_polygon(camera, 7, CELLS, LO, SPAN)
    assert polygon is not None and len(polygon) >= 3
    scale = (WIDTH - 1) / float(WIDTH)
    # x_cam spans [0, 1] at z_cam = 3 -> pixels 100 and 200/3 + 100
    assert polygon[:, 0].min() == pytest.approx(100.0 * scale, abs=1e-6)
    assert polygon[:, 0].max() == pytest.approx(
        (FOCAL * 1.0 / 3.0 + PRINCIPAL) * scale, abs=1e-6)
    # y_cam spans [-1, 0] at z_cam = 3 -> pixels 100 - 200/3 and 100
    assert polygon[:, 1].max() == pytest.approx(100.0 * scale, abs=1e-6)
    assert polygon[:, 1].min() == pytest.approx(
        (PRINCIPAL - FOCAL * 1.0 / 3.0) * scale, abs=1e-6)


def test_cell_box_decodes_the_key():
    low, high = viz.cell_box(4, CELLS, LO, SPAN)
    assert list(low) == [1.0, 0.0, 0.0]
    assert list(high) == [2.0, 1.0, 1.0]
    low, high = viz.cell_box(7, CELLS, LO, SPAN)
    assert list(low) == [1.0, 1.0, 1.0]
    assert list(high) == [2.0, 2.0, 2.0]


def test_convex_hull_of_a_square():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
                       [0.5, 0.5]])
    hull = viz.convex_hull(points)
    assert len(hull) == 4
    assert {tuple(p) for p in hull.tolist()} == {
        (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}


# ---------------------------------------------------------------------------
# grouping and the cross-check
# ---------------------------------------------------------------------------


def test_build_voxel_groups_mirrors_the_estimator():
    cloud = synthetic_cloud()
    labels, kept_keys = viz.build_voxel_groups(cloud, CELLS, LO, SPAN, MIN_ROWS)
    assert list(kept_keys) == EXPECTED_KEPT_KEYS
    counts = np.bincount(labels[labels >= 0], minlength=len(kept_keys))
    assert list(counts) == EXPECTED_ROWS
    # the 2-row cell is below the floor and must be substrate
    assert int((labels < 0).sum()) == 2


def test_rows_outside_the_box_are_substrate():
    cloud = np.concatenate([synthetic_cloud(),
                            np.array([[9.0, 9.0, 9.0]] * 6, dtype=np.float32)])
    labels, kept_keys = viz.build_voxel_groups(cloud, CELLS, LO, SPAN, MIN_ROWS)
    assert list(kept_keys) == EXPECTED_KEPT_KEYS
    assert int((labels < 0).sum()) == 2 + 6


def test_verify_grouping_passes_on_consistent_input():
    cloud = synthetic_cloud()
    report = synthetic_report(cloud.shape[0])
    labels, kept_keys = viz.build_voxel_groups(cloud, CELLS, LO, SPAN, MIN_ROWS)
    ok, message = viz.verify_grouping(report, synthetic_program(), labels,
                                      kept_keys)
    assert ok, message
    assert "OK" in message


def test_verify_grouping_fails_on_a_row_count_mismatch():
    cloud = synthetic_cloud()
    report = synthetic_report(cloud.shape[0])
    report["diagnostics"]["decisions"][1]["rows"] = 99
    labels, kept_keys = viz.build_voxel_groups(cloud, CELLS, LO, SPAN, MIN_ROWS)
    ok, message = viz.verify_grouping(report, None, labels, kept_keys)
    assert not ok
    assert "row count" in message


def test_verify_grouping_fails_on_a_cell_key_mismatch():
    cloud = synthetic_cloud()
    report = synthetic_report(cloud.shape[0])
    program = synthetic_program()
    program["spatial"]["group_cell_keys"] = {"1": [6]}
    labels, kept_keys = viz.build_voxel_groups(cloud, CELLS, LO, SPAN, MIN_ROWS)
    ok, message = viz.verify_grouping(report, program, labels, kept_keys)
    assert not ok
    assert "cell key" in message


# ---------------------------------------------------------------------------
# outcome colouring
# ---------------------------------------------------------------------------


def test_outcome_classes_and_counts():
    decisions = synthetic_decisions()
    assert viz.outcome_of(decisions[0]) == "contrast"
    assert viz.outcome_of(decisions[1]) == viz.GATED
    assert viz.outcome_of(decisions[2]) == "no_interior_gap"
    counts = viz.outcome_counts(decisions)
    assert counts[viz.GATED] == 1
    assert counts["contrast"] == 1
    assert counts["no_interior_gap"] == 1
    assert counts["camera_disagreement"] == 0
    assert counts["empty_footprint"] == 0
    assert counts[viz.OTHER] == 0


def test_unknown_abstention_reason_falls_into_other():
    record = {"group": 0, "gated": False, "abstain_reason": "inadmissible_interval"}
    assert viz.outcome_of(record) == viz.OTHER
    record = {"group": 0, "gated": False, "abstain_reason": None}
    assert viz.outcome_of(record) == viz.OTHER


def test_every_outcome_class_has_a_distinct_colour():
    colours = [viz.OUTCOME_COLOURS[name] for name in viz.OUTCOME_ORDER]
    assert len(set(colours)) == len(colours)
    assert viz.OUTCOME_COLOURS[viz.GATED] == "#d62728"


# ---------------------------------------------------------------------------
# the no-series path
# ---------------------------------------------------------------------------


def test_report_has_series_is_false_for_this_schema():
    report = synthetic_report(20)
    assert viz.report_has_series(report) is False


def test_report_has_series_detects_a_future_series():
    report = synthetic_report(20)
    report["diagnostics"]["decisions"][1]["series"] = [0.1, 0.2]
    assert viz.report_has_series(report) is True


def test_no_series_warning_names_the_schema_and_the_writer():
    assert viz.REPORT_SCHEMA in viz.NO_SERIES_WARNING
    assert "estimate['series']" in viz.NO_SERIES_WARNING
    assert "estimate_episodes.py" in viz.NO_SERIES_WARNING


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def test_main_writes_an_image_and_warns_about_the_missing_series(scene, capsys):
    out = viz.main([
        "--report", str(scene["report"]),
        "--program", str(scene["program"]),
        "--scene", str(scene["root"]),
        "--xyz_npy", str(scene["xyz"]),
        "--frames", *[str(f) for f in FRAMES],
        "--truth", str(TRUTH[0]), str(TRUTH[1]),
        "--truth_label", "authored gap",
        "--out", str(scene["out"]),
        "--dpi", "60",
    ])
    captured = capsys.readouterr().out
    assert "WARNING" in captured and "NO per-frame series" in captured
    assert "grouping cross-check OK" in captured
    assert os.path.exists(out)
    assert os.path.getsize(out) > 5000
    from PIL import Image
    with Image.open(out) as image:
        assert image.size[0] > image.size[1]


def test_main_auto_selects_frames_spanning_the_window(scene):
    out = viz.main([
        "--report", str(scene["report"]),
        "--scene", str(scene["root"]),
        "--xyz_npy", str(scene["xyz"]),
        "--truth", "4", "7", "--pad", "2", "--n_frames", "4",
        "--out", str(scene["tmp"] / "auto.jpg"),
        "--dpi", "60",
    ])
    assert os.path.exists(out)
    # linspace(2, 9, 4) = 2, 4.333, 6.667, 9, rounded to nearest frame
    assert viz.auto_frames([4, 7], 4, 2) == [2, 4, 7, 9]


def test_main_refuses_without_a_cloud(scene):
    with pytest.raises(viz.VizError, match="xyz_npy"):
        viz.main([
            "--report", str(scene["report"]), "--scene", str(scene["root"]),
            "--frames", "4", "6", "--out", str(scene["tmp"] / "x.jpg"),
        ])


def test_main_refuses_a_grouping_mismatch_unless_allowed(scene):
    report = json.loads(scene["report"].read_text())
    report["diagnostics"]["decisions"][0]["rows"] = 123
    path = scene["tmp"] / "bad_report.json"
    path.write_text(json.dumps(report))
    argv = ["--report", str(path), "--scene", str(scene["root"]),
            "--xyz_npy", str(scene["xyz"]), "--frames", "4", "6",
            "--truth", "5", "7", "--dpi", "60",
            "--out", str(scene["tmp"] / "bad.jpg")]
    with pytest.raises(viz.VizError, match="cross-check FAILED"):
        viz.main(argv)
    out = viz.main(argv + ["--allow_row_mismatch"])
    assert os.path.exists(out)


def test_main_refuses_a_cloud_of_the_wrong_size(scene):
    other = scene["tmp"] / "other.npy"
    np.save(other, synthetic_cloud()[:-1])
    with pytest.raises(viz.VizError, match="rows but the report"):
        viz.main(["--report", str(scene["report"]), "--scene", str(scene["root"]),
                  "--xyz_npy", str(other), "--frames", "4",
                  "--out", str(scene["tmp"] / "x.jpg")])


def test_cli_subprocess_runs_end_to_end(scene):
    script = os.path.join(REPO_ROOT,
                          "scripts", "t1_estimator_viz.py")
    out = scene["tmp"] / "cli.jpg"
    result = subprocess.run(
        [sys.executable, script,
         "--report", str(scene["report"]), "--program", str(scene["program"]),
         "--scene", str(scene["root"]), "--xyz_npy", str(scene["xyz"]),
         "--frames", "4", "6", "8", "--truth", "5", "7",
         "--out", str(out), "--dpi", "60"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert out.exists() and out.stat().st_size > 5000
    assert "NO per-frame series" in result.stdout


def test_module_imports_without_torch():
    """The module must never pull torch in at import time."""
    script = os.path.join(REPO_ROOT,
                          "scripts", "t1_estimator_viz.py")
    code = (
        "import sys, runpy, importlib.util;"
        "sys.modules['torch'] = None;"
        "spec = importlib.util.spec_from_file_location('viz', %r);"
        "module = importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(module);"
        "print('ok', module.REPORT_SCHEMA)" % script
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok adags-episode-estimate-v1" in result.stdout
