"""CPU tests for scripts/event_region_frame_profile.py.

Run with:
    python -m pytest tests/test_event_region_frame_profile.py -q

Two things are under test. The first is that the ``--roi_dir`` per-frame-mask
mode reads the masks it claims to read and pools exactly the masked pixels. The
second is a byte-identity guard: the bounding-box output of the CURRENT script
must equal, byte for byte, the output of the version that produced the frozen
profiles, on the same inputs. That reference is pinned by CONTENT -- a git blob
id whose sha256 is asserted here -- so it cannot drift when this branch moves.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import event_region_frame_profile as erfp  # noqa: E402

# The bounding-box reference: the blob as committed in 4b3e9b7, pinned by its
# own sha256 so that a later commit of this file cannot silently become the
# reference it is being compared against.
REFERENCE_BLOB = "8fa186fe71e38a8235ba070ae608ecbddb338083"
REFERENCE_SHA256 = "63050ccdeaf943fed69b093ed877d776f5823d1b23a7e8efb9b5dc9e6805428a"

N_FRAMES = 6
HEIGHT, WIDTH = 6, 8


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def build_images(root):
    """gt is black everywhere; render differs by a per-frame level on a patch.

    Frame f differs from gt by (10 + 10*f)/255 on rows 1..3, cols 2..5.
    """
    renders = Path(root) / "renders"
    gt = Path(root) / "gt"
    levels = {}
    for f in range(N_FRAMES):
        g = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        r = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        level = 10 + 10 * f
        r[1:4, 2:6, :] = level
        levels[f] = level / 255.0
        _write_png(gt / f"{f:05d}.png", g)
        _write_png(renders / f"{f:05d}.png", r)
    return str(renders), str(gt), levels


def build_masks_manifest(root):
    path = Path(root) / "masks.json"
    path.write_text(
        json.dumps(
            {
                "raster": [WIDTH, HEIGHT],
                "events": [
                    {
                        "name": "E_patch",
                        "bbox": [2, 1, 6, 4],
                        "frames": [[1, 3]],
                        "class": "dynamic_content",
                    }
                ],
            }
        )
    )
    return str(path)


def build_roi_dir(root):
    """core/ covers 6 patch pixels; ring/ is empty on frames 0 and 5."""
    roi = Path(root) / "roi"
    for f in range(N_FRAMES):
        core = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        core[1:3, 2:5] = 255                      # 6 pixels, all inside the patch
        _write_png(roi / "core" / f"{f:05d}.png", core)
        ring = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        if f not in (0, 5):
            ring[4, 0:4] = 255                    # 4 pixels, all OUTSIDE the patch
        _write_png(roi / "ring" / f"{f:05d}.png", ring)
    return str(roi)


def run_profile(argv):
    old = sys.argv
    sys.argv = ["event_region_frame_profile.py"] + argv
    try:
        erfp.main()
    finally:
        sys.argv = old


# ---------------------------------------------------------------------------
# 1. Byte identity of the bounding-box default
# ---------------------------------------------------------------------------


def _reference_module(tmp_path):
    try:
        blob = subprocess.run(
            ["git", "cat-file", "blob", REFERENCE_BLOB],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        pytest.skip(f"reference blob {REFERENCE_BLOB} unavailable: {exc}")
    assert hashlib.sha256(blob).hexdigest() == REFERENCE_SHA256, (
        "the pinned reference blob does not hash to the recorded sha256"
    )
    path = Path(tmp_path) / "reference_erfp.py"
    path.write_bytes(blob)
    spec = importlib.util.spec_from_file_location("reference_erfp", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bbox_only_output_is_byte_identical_to_the_reference(tmp_path, capsys):
    renders, gt, _ = build_images(tmp_path)
    masks = build_masks_manifest(tmp_path)
    reference = _reference_module(tmp_path)

    old_out = tmp_path / "old.json"
    new_out = tmp_path / "new.json"
    argv = ["--renders", renders, "--gt", gt, "--masks", masks]

    saved = sys.argv
    try:
        sys.argv = ["ref"] + argv + ["--out", str(old_out)]
        reference.main()
    finally:
        sys.argv = saved
    run_profile(argv + ["--out", str(new_out)])
    capsys.readouterr()

    assert new_out.read_bytes() == old_out.read_bytes()
    assert (tmp_path / "new.csv").read_bytes() == (tmp_path / "old.csv").read_bytes()


# ---------------------------------------------------------------------------
# 2. Per-frame ROI masks
# ---------------------------------------------------------------------------


def test_roi_psnr_pools_exactly_the_masked_pixels(tmp_path, capsys):
    renders, gt, levels = build_images(tmp_path)
    roi = build_roi_dir(tmp_path)
    out = tmp_path / "p.json"
    run_profile(["--renders", renders, "--gt", gt, "--roi_dir", roi, "--out", str(out)])
    capsys.readouterr()
    data = json.loads(out.read_text())

    core = data["events"]["roi:core"]
    assert core["kind"] == "per_frame_mask"
    assert core["pixels_per_frame"] == [6] * N_FRAMES
    for f in range(N_FRAMES):
        # every core pixel is inside the patch, so the MSE is level^2 exactly
        expected = float("inf") if levels[f] == 0.0 else 10.0 * math.log10(1.0 / levels[f] ** 2)
        assert core["per_frame_psnr"][f] == pytest.approx(expected, rel=1e-5)

    ring = data["events"]["roi:ring"]
    # the ring never overlaps the patch -> a perfect match -> infinite PSNR
    assert ring["pixels_per_frame"] == [0, 4, 4, 4, 4, 0]
    assert ring["per_frame_psnr"][0] is None and ring["per_frame_psnr"][5] is None
    assert ring["per_frame_psnr"][1] == float("inf")
    assert ring["n_frames_with_mask"] == 4


def test_roi_psnr_is_a_true_partial_overlap_pool(tmp_path, capsys):
    """A mask straddling the patch pools both parts, not just the bright one."""
    renders, gt, levels = build_images(tmp_path)
    roi = Path(tmp_path) / "roi2"
    for f in range(N_FRAMES):
        m = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        m[1, 2:4] = 255      # 2 pixels inside the patch
        m[5, 0:2] = 255      # 2 pixels outside it
        _write_png(roi / "half" / f"{f:05d}.png", m)
    out = tmp_path / "p.json"
    run_profile(["--renders", renders, "--gt", gt, "--roi_dir", str(roi), "--out", str(out)])
    capsys.readouterr()
    half = json.loads(out.read_text())["events"]["roi:half"]
    assert half["pixels_per_frame"] == [4] * N_FRAMES
    for f in range(1, N_FRAMES):
        mse = 0.5 * levels[f] ** 2          # half the masked pixels differ
        assert half["per_frame_psnr"][f] == pytest.approx(
            10.0 * math.log10(1.0 / mse), rel=1e-5
        )


def test_missing_mask_file_reads_as_an_empty_mask(tmp_path, capsys):
    renders, gt, _ = build_images(tmp_path)
    roi = build_roi_dir(tmp_path)
    (Path(roi) / "core" / "00002.png").unlink()
    out = tmp_path / "p.json"
    run_profile(["--renders", renders, "--gt", gt, "--roi_dir", roi, "--out", str(out)])
    capsys.readouterr()
    core = json.loads(out.read_text())["events"]["roi:core"]
    assert core["pixels_per_frame"][2] == 0
    assert core["per_frame_psnr"][2] is None
    assert core["per_frame_psnr"][1] is not None


def test_roi_and_bbox_events_coexist_and_reach_the_csv(tmp_path, capsys):
    renders, gt, _ = build_images(tmp_path)
    masks = build_masks_manifest(tmp_path)
    roi = build_roi_dir(tmp_path)
    out = tmp_path / "p.json"
    run_profile(
        ["--renders", renders, "--gt", gt, "--masks", masks, "--roi_dir", roi,
         "--out", str(out)]
    )
    capsys.readouterr()
    data = json.loads(out.read_text())
    assert set(data["events"]) == {"E_patch", "roi:core", "roi:ring"}
    assert data["events"]["E_patch"]["pooled_psnr_in_window"] is not None
    assert "kind" not in data["events"]["E_patch"]
    assert data["roi_dir"] == str(Path(roi).resolve())

    rows = (tmp_path / "p.csv").read_text().splitlines()
    assert rows[0].split(",") == ["frame", "whole_frame", "E_patch", "roi:core", "roi:ring"]
    # frame 0 has an empty ring mask -> an empty CSV cell, never a fabricated 0
    assert rows[1].split(",")[4] == ""
    assert rows[2].split(",")[4] != ""


def test_masks_is_optional_only_when_roi_dir_is_given(tmp_path, capsys):
    renders, gt, _ = build_images(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        run_profile(["--renders", renders, "--gt", gt, "--out", str(tmp_path / "p.json")])
    capsys.readouterr()
    assert "at least one of" in str(excinfo.value)


def test_a_mask_at_the_wrong_resolution_is_refused(tmp_path, capsys):
    renders, gt, _ = build_images(tmp_path)
    roi = Path(tmp_path) / "roi3"
    for f in range(N_FRAMES):
        _write_png(roi / "bad" / f"{f:05d}.png", np.zeros((HEIGHT + 2, WIDTH), dtype=np.uint8))
    with pytest.raises(SystemExit) as excinfo:
        run_profile(
            ["--renders", renders, "--gt", gt, "--roi_dir", str(roi),
             "--out", str(tmp_path / "p.json")]
        )
    capsys.readouterr()
    assert "shape mismatch" in str(excinfo.value)


def test_frame_index_comes_from_the_filename_digits():
    assert erfp._frame_index("00042.png") == 42
    assert erfp._frame_index("frame_00007.png") == 7
