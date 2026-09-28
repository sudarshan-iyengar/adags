"""CPU tests for scripts/eval_event_windows.py (the common event-window evaluator).

Run with:
    python -m pytest tests/test_eval_event_windows.py -q

Analytic tests fix the frozen conventions (pooled PSNR, exact-match and empty-region
records, SSIM identity/constant-image values and its zero-padded full-frame map,
the LPIPS crop rule, STG segment splitting). A reproduction test runs the frozen
reference chain -- scripts/event_region_frame_profile.py per-frame PSNRs, pooled
by scripts/realdata_gate_analysis.py -- on a synthetic sequence and requires the
new evaluator's P1 (per-frame mask) and P2 (bbox) PSNR to agree. The LPIPS value
itself needs torch and the lpips package and is only exercised where they exist.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_event_windows as eew  # noqa: E402

H, W = 40, 48


def _png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path)


def _package(root, regions, windows, frames=(0, 9)):
    pkg = {"package_version": "1", "scene": "synthetic", "camera": "cam00", "raster": [W, H],
           "gt": {"dir": "gt", "pattern": "cam00_%04d.png"}, "frames": list(frames),
           "regions": regions, "windows": windows}
    p = Path(root) / "pkg.json"
    p.write_text(json.dumps(pkg))
    return p, pkg


def _run(tmp_path, pkg_path, metrics="psnr,ssim", **kw):
    out = tmp_path / "out.json"
    argv = ["--package", str(pkg_path), "--renders", str(tmp_path / "renders"), "--metrics", metrics, "--out", str(out)]
    for k, v in kw.items():
        argv += [f"--{k}", str(v)]
    rc = eew.main(argv)
    return rc, json.loads(out.read_text())


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------


def test_psnr_constant_error_analytic(tmp_path):
    for f in range(3):
        _png(tmp_path / "gt" / f"cam00_{f:04d}.png", np.zeros((H, W, 3)))
        _png(tmp_path / "renders" / f"{f:05d}.png", np.full((H, W, 3), 51))
    p, _ = _package(tmp_path, {"box": {"kind": "bbox", "bbox": [4, 5, 14, 25]}},
                    {"P": {"region": "box", "frames": [0, 2]}})
    rc, out = _run(tmp_path, p, metrics="psnr")
    w = out["windows"]["P"]
    assert rc == 0
    assert w["pixel_times"] == 3 * 10 * 20
    assert w["psnr"]["mse"] == pytest.approx(0.04, abs=1e-15)
    assert w["psnr"]["psnr"] == pytest.approx(-10 * math.log10(0.04), abs=1e-12)


def test_psnr_pools_squared_error_by_pixel_times(tmp_path):
    # frame 0: 10 px at error 10/255 ; frame 1: 30 px at error 40/255 (per-frame masks)
    sizes, levels = {0: 10, 1: 30}, {0: 10, 1: 40}
    for f in (0, 1):
        _png(tmp_path / "gt" / f"cam00_{f:04d}.png", np.zeros((H, W, 3)))
        _png(tmp_path / "renders" / f"{f:05d}.png", np.full((H, W, 3), levels[f]))
        m = np.zeros((H, W)); m.flat[: sizes[f]] = 255
        _png(tmp_path / "masks" / f"{f:05d}.png", m)
    p, _ = _package(tmp_path, {"m": {"kind": "mask_dir", "dir": "masks", "pattern": "%05d.png"}},
                    {"P": {"region": "m", "frames": [0, 1]}})
    _, out = _run(tmp_path, p, metrics="psnr")
    mse = (10 * (10 / 255) ** 2 + 30 * (40 / 255) ** 2) / 40
    assert out["windows"]["P"]["psnr"]["mse"] == pytest.approx(mse, rel=1e-12)
    assert out["windows"]["P"]["pixel_times"] == 40
    assert [e["px"] for e in out["windows"]["P"]["per_frame"]] == [10, 30]


def test_exact_match_and_empty_region_records(tmp_path):
    img = np.random.default_rng(0).integers(0, 256, (H, W, 3))
    _png(tmp_path / "gt" / "cam00_0000.png", img)
    _png(tmp_path / "renders" / "00000.png", img)
    (tmp_path / "masks").mkdir()  # no mask file for frame 0 -> empty region
    p, _ = _package(tmp_path, {"whole": {"kind": "whole"},
                               "m": {"kind": "mask_dir", "dir": "masks"}},
                    {"A": {"region": "whole", "frames": [0, 0]}, "B": {"region": "m", "frames": [0, 0]}})
    rc, out = _run(tmp_path, p)
    a, b = out["windows"]["A"], out["windows"]["B"]
    assert a["psnr"] == {"mse": 0.0, "psnr": None, "exact_match": True, "pixel_count": H * W}
    assert a["ssim"]["value"] == pytest.approx(1.0, abs=1e-12)
    assert b["psnr"] == {"psnr": "NA", "pixel_count": 0} and b["ssim"]["value"] == "NA"
    assert b["empty_frames"] == [0] and rc == 0


def test_require_nonempty_every_frame_fails_closed(tmp_path):
    for f in (0, 1):
        _png(tmp_path / "gt" / f"cam00_{f:04d}.png", np.zeros((H, W, 3)))
        _png(tmp_path / "renders" / f"{f:05d}.png", np.zeros((H, W, 3)))
    m = np.zeros((H, W)); m[:5, :5] = 255
    _png(tmp_path / "masks" / "00000.png", m)  # frame 1 has no mask
    p, _ = _package(tmp_path, {"m": {"kind": "mask_dir", "dir": "masks"}},
                    {"P1": {"region": "m", "frames": [0, 1], "require_nonempty_every_frame": True}})
    rc, out = _run(tmp_path, p, metrics="psnr")
    assert rc == 2 and "error" in out["windows"]["P1"] and out["windows"]["P1"]["empty_frames"] == [1]


def test_mask_threshold_matches_reference_rule(tmp_path):
    # the reference (event_region_frame_profile._load_roi_mask) uses value > 127
    m = np.zeros((H, W)); m[0, 0] = 127; m[0, 1] = 128; m[0, 2] = 255
    _png(tmp_path / "masks" / "00000.png", m)
    got = eew.region_mask({"kind": "mask_dir", "dir": "masks"}, 0, (H, W), str(tmp_path))
    assert got.sum() == 2 and not got[0, 0] and got[0, 1] and got[0, 2]


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


def _ssim_direct(x, y):
    """Independent reference: 2-D convolution with the outer-product window, zero padding."""
    from scipy.signal import convolve2d
    g = eew._gauss1d(); w2 = np.outer(g, g)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    acc = 0
    for c in range(3):
        a, b = x[..., c], y[..., c]
        f = lambda z: convolve2d(z, w2, mode="same", boundary="fill", fillvalue=0.0)  # noqa: E731
        ma, mb = f(a), f(b)
        saa, sbb, sab = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
        acc = acc + ((2 * ma * mb + C1) * (2 * sab + C2)) / ((ma ** 2 + mb ** 2 + C1) * (saa + sbb + C2))
    return acc / 3


def test_ssim_map_matches_direct_2d_convolution():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 256, (H, W, 3)) / 255.0
    y = np.clip(x + rng.normal(0, 0.05, x.shape), 0, 1)
    np.testing.assert_allclose(eew.ssim_map(x, y), _ssim_direct(x, y), rtol=0, atol=1e-12)


def test_ssim_constant_images_interior_analytic():
    c1, c2 = 0.2, 0.6
    x, y = np.full((H, W, 3), c1), np.full((H, W, 3), c2)
    s = eew.ssim_map(x, y)
    want = (2 * c1 * c2 + 0.01 ** 2) / (c1 ** 2 + c2 ** 2 + 0.01 ** 2)  # sigma terms vanish away from borders
    assert s[10:30, 10:38] == pytest.approx(np.full((20, 28), want), abs=1e-12)
    assert abs(s[0, 0] - want) > 1e-3  # the zero padding reaches the border, by convention


def test_gaussian_window_is_normalised_and_symmetric():
    g = eew._gauss1d()
    assert g.shape == (11,) and g.sum() == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_allclose(g, g[::-1], atol=0)
    assert g[5] == g.max()


# ---------------------------------------------------------------------------
# LPIPS crop rule and segments
# ---------------------------------------------------------------------------


def test_lpips_box_rule():
    m = np.zeros((200, 300), dtype=bool)
    m[100:104, 150:153] = True                       # tiny region -> 64x64 around it
    x0, y0, x1, y1 = eew.lpips_box(m)
    assert (x1 - x0, y1 - y0) == (64, 64) and x0 <= 150 and x1 >= 153 and y0 <= 100 and y1 >= 104
    m = np.zeros((200, 300), dtype=bool); m[50:150, 40:200] = True   # large region -> bbox + 16
    assert eew.lpips_box(m) == [24, 34, 216, 166]
    m = np.zeros((200, 300), dtype=bool); m[0:3, 0:3] = True          # corner -> clipped, still 64
    assert eew.lpips_box(m) == [0, 0, 64, 64]
    m = np.zeros((200, 300), dtype=bool); m[197:200, 297:300] = True  # far corner
    assert eew.lpips_box(m) == [236, 136, 300, 200]


def test_segments_split_at_multiples():
    assert eew._segments(45, 60, 50) == [(45, 49), (50, 60)]
    assert eew._segments(63, 87, 50) == [(63, 87)]
    assert eew._segments(0, 299, 50)[-1] == (250, 299) and len(eew._segments(0, 299, 50)) == 6


def test_segment_len_reports_per_segment(tmp_path):
    for f in range(45, 56):
        _png(tmp_path / "gt" / f"cam00_{f:04d}.png", np.zeros((H, W, 3)))
        _png(tmp_path / "renders" / f"{f:05d}.png", np.full((H, W, 3), 10 if f < 50 else 30))
    p, _ = _package(tmp_path, {"whole": {"kind": "whole"}}, {"S": {"region": "whole", "frames": [45, 55]}})
    _, out = _run(tmp_path, p, metrics="psnr", segment_len=50)
    seg = out["windows"]["S"]["segments"]
    assert set(seg) == {"45-49", "50-55"}
    assert seg["45-49"]["psnr"]["mse"] == pytest.approx((10 / 255) ** 2, rel=1e-12)
    assert seg["50-55"]["psnr"]["mse"] == pytest.approx((30 / 255) ** 2, rel=1e-12)


# ---------------------------------------------------------------------------
# reproduction of the frozen reference chain
# ---------------------------------------------------------------------------


def test_reproduces_event_region_profile_and_reducer_pooling(tmp_path):
    from scripts import event_region_frame_profile as erfp
    from scripts import realdata_gate_analysis as rga
    rng = np.random.default_rng(7)
    frames = range(20)
    box = [10, 8, 30, 28]
    for f in frames:
        g = rng.integers(0, 256, (H, W, 3))
        r = np.clip(g + rng.integers(-25, 26, g.shape), 0, 255)
        _png(tmp_path / "gt" / f"{f:05d}.png", g)
        _png(tmp_path / "renders" / f"{f:05d}.png", r)
        m = np.zeros((H, W)); m[5 + f % 4: 20 + f % 7, 6: 18 + f] = 255
        _png(tmp_path / "rois" / "core" / f"{f:05d}.png", m)
    masks = {"raster": [W, H], "events": [{"name": "EV", "bbox": box, "frames": [[12, 17]]}]}
    (tmp_path / "masks.json").write_text(json.dumps(masks))
    ref_out = tmp_path / "ref.json"
    argv = ["x", "--renders", str(tmp_path / "renders"), "--gt", str(tmp_path / "gt"),
            "--masks", str(tmp_path / "masks.json"), "--roi_dir", str(tmp_path / "rois"), "--out", str(ref_out)]
    old = sys.argv
    try:
        sys.argv = argv
        erfp.main()
    finally:
        sys.argv = old
    ref = json.loads(ref_out.read_text())
    idx = {f: i for i, f in enumerate(ref["frames"])}
    core = ref["events"]["roi:core"]
    p1_ref = rga.pooled_psnr_weighted([core["per_frame_psnr"][idx[f]] for f in range(3, 11)],
                                      [core["pixels_per_frame"][idx[f]] for f in range(3, 11)])
    p2_ref = rga.pooled_psnr([ref["events"]["EV"]["per_frame_psnr"][idx[f]] for f in range(12, 18)])
    h2_ref = rga.pooled_psnr(ref["whole_frame_psnr"])

    pkg = {"package_version": "1", "gt": {"dir": "gt", "pattern": "%05d.png"}, "frames": [0, 19],
           "regions": {"core": {"kind": "mask_dir", "dir": "rois/core"}, "box": {"kind": "bbox", "bbox": box},
                       "whole": {"kind": "whole"}},
           "windows": {"P1": {"region": "core", "frames": [3, 10], "require_nonempty_every_frame": True},
                       "P2": {"region": "box", "frames": [12, 17]}, "H2": {"region": "whole", "frames": [0, 19]}}}
    (tmp_path / "pkg.json").write_text(json.dumps(pkg))
    rc, out = _run(tmp_path, tmp_path / "pkg.json", metrics="psnr")
    got = {k: out["windows"][k]["psnr"]["psnr"] for k in ("P1", "P2", "H2")}
    # the reference computes in float32; the new evaluator in float64
    assert rc == 0
    assert got["P1"] == pytest.approx(p1_ref, abs=1e-4)
    assert got["P2"] == pytest.approx(p2_ref, abs=1e-4)
    assert got["H2"] == pytest.approx(h2_ref, abs=1e-4)


# ---------------------------------------------------------------------------
# LPIPS (only where torch + lpips are installed, i.e. envs/adags on Leonardo)
# ---------------------------------------------------------------------------


def test_lpips_identity_is_zero_and_window_skips_small_regions(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("lpips")
    rng = np.random.default_rng(3)
    for f in (0, 1):
        img = rng.integers(0, 256, (96, 128, 3))
        _png(tmp_path / "gt" / f"cam00_{f:04d}.png", img)
        _png(tmp_path / "renders" / f"{f:05d}.png", img if f == 0 else np.clip(img + 40, 0, 255))
    m = np.zeros((96, 128)); m[10:30, 10:30] = 255
    _png(tmp_path / "masks" / "00000.png", m)
    m = np.zeros((96, 128)); m[10:15, 10:15] = 255      # 25 px < 100 -> skipped
    _png(tmp_path / "masks" / "00001.png", m)
    p, _ = _package(tmp_path, {"m": {"kind": "mask_dir", "dir": "masks"}, "whole": {"kind": "whole"}},
                    {"A": {"region": "m", "frames": [0, 1]}, "B": {"region": "whole", "frames": [0, 1]}})
    rc, out = _run(tmp_path, p, metrics="lpips")
    a, b = out["windows"]["A"]["lpips"], out["windows"]["B"]["lpips"]
    assert a["n_frames_used"] == 1 and a["n_frames_skipped_small_region"] == 1
    assert a["value"] == pytest.approx(0.0, abs=1e-6)
    assert b["n_frames_used"] == 2 and b["value"] > 0.0
