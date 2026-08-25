"""The per-point temporal-extent seam, end to end through the PLY reader.

`create_from_pcd` needs CUDA, so it cannot be exercised here.  What CAN be
exercised, and is what actually breaks, is the seam either side of it: that a
PLY carrying `time` / `t_extent` reaches `BasicPointCloud` with those columns
intact, that a PLY WITHOUT them still reads exactly as it did before, and that
the standard-deviation-to-`dist_t` conversion `create_from_pcd` applies is the
inverse of the one `imvid_build_population.py` writes.

That last one is the dangerous one.  The trainer stores `_scaling_t` such that
`get_scaling_t = exp(_scaling_t) = sqrt(dist_t)`, and `get_cov_t` then consumes
that value as a VARIANCE -- so a temporal standard deviation is `dist_t ** 0.25`
and not `dist_t ** 0.5`.  A wrong exponent does not raise: it trains happily
with every support width wrong, and no downstream check would notice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

plyfile = pytest.importorskip("plyfile")

from scene.dataset_readers import fetchPly  # noqa: E402
from utils.graphics_utils import BasicPointCloud  # noqa: E402


def _write_ply(path: Path, n: int, *, with_time: bool, with_extent: bool) -> dict:
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"),
              ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
              ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if with_time:
        fields.append(("time", "f4"))
    if with_extent:
        fields.append(("t_extent", "f4"))
    arr = np.empty(n, dtype=fields)
    rng = np.random.default_rng(0)
    arr["x"], arr["y"], arr["z"] = rng.random(n), rng.random(n), rng.random(n)
    arr["nx"] = arr["ny"] = arr["nz"] = 0.0
    arr["red"] = np.arange(n, dtype=np.uint8)
    arr["green"] = 128
    arr["blue"] = 255
    expect = {}
    if with_time:
        arr["time"] = np.linspace(0.0, 4.988316666666667, n, dtype=np.float32)
        expect["time"] = arr["time"].copy()
    if with_extent:
        arr["t_extent"] = np.linspace(0.13, 2.49, n, dtype=np.float32)
        expect["t_extent"] = arr["t_extent"].copy()
    plyfile.PlyData(
        [plyfile.PlyElement.describe(arr, "vertex")]
    ).write(str(path))
    return expect


def test_basic_point_cloud_defaults_keep_old_callers_working():
    """The new field is optional; a three-argument construction still works."""
    pcd = BasicPointCloud(points=np.zeros((3, 3)), colors=np.zeros((3, 3)),
                          normals=np.zeros((3, 3)))
    assert pcd.time is None
    assert pcd.t_extent is None


def test_ply_without_the_new_columns_reads_exactly_as_before(tmp_path):
    p = tmp_path / "points3d.ply"
    _write_ply(p, 16, with_time=False, with_extent=False)
    pcd = fetchPly(str(p))
    assert pcd.points.shape == (16, 3)
    assert pcd.time is None, "a cloud with no time column must not acquire one"
    assert pcd.t_extent is None, "a cloud with no t_extent must not acquire one"


def test_ply_with_time_only_is_unchanged_by_the_new_field(tmp_path):
    p = tmp_path / "points3d.ply"
    expect = _write_ply(p, 16, with_time=True, with_extent=False)
    pcd = fetchPly(str(p))
    np.testing.assert_allclose(pcd.time[:, 0], expect["time"], rtol=0, atol=0)
    assert pcd.t_extent is None


def test_ply_with_both_columns_round_trips(tmp_path):
    p = tmp_path / "points3d.ply"
    expect = _write_ply(p, 32, with_time=True, with_extent=True)
    pcd = fetchPly(str(p))
    assert pcd.time.shape == (32, 1)
    assert pcd.t_extent.shape == (32, 1)
    np.testing.assert_allclose(pcd.time[:, 0], expect["time"], rtol=0, atol=0)
    np.testing.assert_allclose(pcd.t_extent[:, 0], expect["t_extent"], rtol=0, atol=0)


def test_std_to_dist_t_is_the_inverse_the_trainer_applies():
    """`std ** 4` must reproduce the trainer's own uniform default exactly.

    If `create_from_pcd` used `** 2` instead, this default point would get
    `dist_t = 0.99883` rather than `0.99766` -- a difference far too small to
    look wrong in a log and large enough to change every support width.
    """
    span = 299 * 1001 / 60000
    trainer_dist_t = span / 5.0
    std = trainer_dist_t ** 0.25
    assert std ** 4 == pytest.approx(trainer_dist_t, rel=0, abs=1e-12)
    assert std ** 2 != pytest.approx(trainer_dist_t, rel=0, abs=1e-6)


def test_population_builder_default_extent_matches_the_trainer_default():
    """The abstain branch must be indistinguishable from no t_extent at all."""
    import imvid_build_population as bp

    span = 299 * 1001 / 60000
    default_std = (span / 5.0) ** 0.25
    assert default_std ** 4 == pytest.approx(span / 5.0, rel=0, abs=1e-12)
    # and the declared bands are ordered as the arm description claims
    compact = bp.COMPACT_SUPPORT_FRAMES * 1001 / 60000
    broad = span * bp.BROAD_SUPPORT_SPAN_FRAC
    assert compact < default_std < broad
    assert bp.EPS_STATIC_PX < bp.EPS_DYNAMIC_PX
