"""Unit tests for scripts/build_absence_suite_scene.py (ARS renderer, fast-path E5)."""
import copy
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import build_absence_suite_scene as B  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
SPEC_G1 = os.path.join(REPO, "configs", "ars", "wave1", "ARS-L1-G1.json")


def _ray(o, d):
    d = np.asarray(d, dtype=np.float64)
    return np.asarray(o, dtype=np.float64), (d / np.linalg.norm(d))[None]


def test_isect_analytic():
    o, d = _ray([0, 0, -5], [0, 0, 1])
    assert B.isect({"type": "sphere", "r": 1.0}, o, d)[0] == pytest.approx(4.0)
    assert B.isect({"type": "box", "half": [1, 2, 0.5]}, o, d)[0] == pytest.approx(4.5)
    # cylinder side (axis y, y in [0, 2]) and top cap from above
    o, d = _ray([0, 1, -5], [0, 0, 1])
    assert B.isect({"type": "cylinder", "r": 0.5, "h": 2.0}, o, d)[0] == pytest.approx(4.5)
    o, d = _ray([0, 5, 0], [0, -1, 0])
    assert B.isect({"type": "cylinder", "r": 0.5, "h": 2.0}, o, d)[0] == pytest.approx(3.0)
    # misses
    o, d = _ray([0, 3, -5], [0, 0, 1])
    assert np.isinf(B.isect({"type": "cylinder", "r": 0.5, "h": 2.0}, o, d)[0])
    assert np.isinf(B.isect({"type": "sphere", "r": 1.0}, o, d)[0])
    # plane rectangle, both sides
    o, d = _ray([0.2, 2, 0.3], [0, -1, 0])
    assert B.isect({"type": "plane", "half": [1, 1]}, o, d)[0] == pytest.approx(2.0)
    o, d = _ray([0.2, -2, 0.3], [0, 1, 0])
    assert B.isect({"type": "plane", "half": [1, 1]}, o, d)[0] == pytest.approx(2.0)
    o, d = _ray([3, 2, 0], [0, -1, 0])
    assert np.isinf(B.isect({"type": "plane", "half": [1, 1]}, o, d)[0])


def test_pinhole_convention():
    cam = {"name": "t", "R": np.eye(3), "C": np.zeros(3), "f": 100.0, "W": 64, "H": 48}
    us, vs = np.array([0, 10, 63]), np.array([0, 20, 47])
    d = B.cam_rays(cam, us, vs, B.OFFS[B.CENTRE:])[:, 0]
    uv, z = B.project(cam, 5.0 * d)
    assert np.allclose(uv[:, 0], us + 0.5) and np.allclose(uv[:, 1], vs + 0.5)
    assert (z > 0).all()


@pytest.mark.parametrize("shape", [{"type": "sphere", "r": 0.7}, {"type": "box", "half": [0.3, 0.8, 0.5]},
                                   {"type": "cylinder", "r": 0.4, "h": 1.3}])
def test_sdf_samples_on_surface(shape):
    s = B.surface_samples(shape, 0.05)
    assert np.abs(B.sdf_local(shape, s)).max() < 1e-9
    inner = np.array([[0.0, 0.5, 0.0]]) if shape["type"] == "cylinder" else np.zeros((1, 3))
    assert B.sdf_local(shape, inner)[0] < 0


def test_yaw_facing_and_upright():
    R = B.r_upright(37.0)
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    assert np.allclose(R @ np.array([0, 1, 0]), B.UP)
    yaw = B._yaw_facing([6.0, 7.8, 11.6], [0.4, -1.3, -0.2])
    got = B.r_upright(yaw) @ np.array([0, 0, 1.0])
    d = np.array([0.4 - 6.0, 0.0, -0.2 - 11.6])
    assert got @ (d / np.linalg.norm(d)) > 0.99999


def test_roll_is_rolling_without_slipping():
    r = 0.7
    m = {"type": "roll", "frames": [10, 40], "from": [0.0, 7.1, 11.0], "to": [5.0, 7.1, 11.0], "radius": r, "yaw_deg": 0.0}
    f, h = 25.0, 1e-4
    R0, t0 = B.root_pose(m, f)
    contact_local = R0.T @ (-r * B.UP)            # material point at the contact
    R1, t1 = B.root_pose(m, f + h)
    v = ((R1 @ contact_local + t1) - (R0 @ contact_local + t0)) / h
    speed = np.linalg.norm((t1 - t0) / h)
    assert speed > 0.1 and np.linalg.norm(v) < 1e-3 * speed


def test_noise_is_frozen():
    """A change here changes every ARS scene's pixels."""
    p = np.array([[0.1, 0.2, 0.3], [-5.5, 2.25, 7.75], [123.4, -56.7, 8.9]])
    v = B.fbm(p, 1234, 5)
    assert (v >= 0).all() and (v <= 1).all()
    assert B.sha256_bytes(v.tobytes())[:16] == "42834b141c58a5e0"


def _tiny(spec, W=338, H=254):
    s = copy.deepcopy(spec)
    s["rig"]["raster"] = [W, H]
    return s


@pytest.mark.skipif(not os.path.isfile(SPEC_G1), reason="wave-1 spec not generated")
def test_optimised_render_equals_brute_force():
    spec = _tiny(json.load(open(SPEC_G1)))
    cams = B.load_rig(spec, REPO)
    for ci, f in ((0, 75), (5, 0)):
        rr = B.CameraRenderer(spec, cams[ci])
        rgb, ident, v6, _ = rr.frame(f, v6=True)
        rgb_b, id_b = B.render_brute(spec, cams[ci], f)
        assert (rgb == rgb_b).all() and (ident == id_b).all()
        assert v6["outside_support_px"] == 0


def test_v10_detects_overlap_and_floating():
    tex = {"type": "noise", "seed": 1, "c0": [0.5] * 3, "c1": [0.6] * 3, "scale": 1.0}
    base = {
        "clearance": 0.05, "v10_spacing": 0.02,
        "support_contacts": [["a", "table"], ["b", "table"]], "support_surfaces": {"table": 0.0},
        "objects": [
            {"name": "table", "role": "static", "parts": [{"id": 10, "shape": {"type": "box", "half": [5, 0.25, 5]}, "tex": tex}],
             "motion": {"type": "static", "pos": [0, 0.25, 0], "R": np.eye(3).tolist()}},
            {"name": "a", "role": "static", "parts": [{"id": 20, "shape": {"type": "cylinder", "r": 0.3, "h": 1.0}, "tex": tex}],
             "motion": {"type": "static", "pos": [0, 0, 0], "yaw_deg": 0}},
            {"name": "b", "role": "static", "parts": [{"id": 21, "shape": {"type": "cylinder", "r": 0.3, "h": 1.0}, "tex": tex}],
             "motion": {"type": "static", "pos": [1.0, 0, 0], "yaw_deg": 0}},
        ],
    }
    assert B.check_v10(base)["pass"]
    bad = copy.deepcopy(base)
    bad["objects"][2]["motion"]["pos"] = [0.5, 0, 0]          # overlapping by 0.1
    assert not B.check_v10(bad)["pass"]
    near = copy.deepcopy(base)
    near["objects"][2]["motion"]["pos"] = [0.63, 0, 0]         # 0.03 apart < clearance 0.05
    assert not B.check_v10(near)["pass"]
    ok = copy.deepcopy(base)
    ok["objects"][2]["motion"]["pos"] = [0.7, 0, 0]            # 0.10 apart: certified >= 0.10 - 0.02
    assert B.check_v10(ok)["pass"]
    flo = copy.deepcopy(base)
    flo["objects"][2]["motion"]["pos"] = [1.0, -0.01, 0]      # floats 0.01 above the table
    r = B.check_v10(flo)
    assert not r["pass"] and any(f["kind"] == "support" for f in r["failures"])
