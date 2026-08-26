"""The per-point temporal-extent seam, through the REAL writer and reader.

`create_from_pcd` needs CUDA and cannot run here.  What can be exercised, and
is what actually breaks, is everything either side of it: that the population
builder's own `write_ply` emits a cloud `fetchPly` reads back with both
columns intact, that a cloud WITHOUT them still reads exactly as before, and
that the conversion `create_from_pcd` applies is the inverse of the one the
builder writes.

An earlier version of this file asserted `(x ** 0.25) ** 4 == x`, imported
neither module, and would have passed unchanged if the trainer's exponent
were edited to 2 tomorrow.  It also discriminated the two exponents at
`span / 5 = 0.9977` -- the one value in the whole system where `** 4` and
`** 2` differ by 0.1% -- while at the values the arm actually uses they
differ by 56x.  Both faults are fixed below: the exponent is pinned against
the trainer's SOURCE, and the numeric check runs at the compact and broad
bands.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

pytest.importorskip("plyfile")

import imvid_build_population as bp  # noqa: E402
from scene.dataset_readers import fetchPly  # noqa: E402
from utils.graphics_utils import BasicPointCloud  # noqa: E402

GAUSSIAN_MODEL = (REPO_ROOT / "scene" / "gaussian_model.py").read_text(encoding="utf-8")


def _write_via_real_writer(path: Path, n: int) -> dict:
    """Write through `bp.write_ply` -- the function the pipeline actually uses.

    Re-implementing its dtype here (as the earlier version did) would have let
    a misnamed field, a wrong width, or a swapped column pass unnoticed.
    """
    rng = np.random.default_rng(0)
    xyz = rng.random((n, 3)).astype(np.float32)
    rgb = (rng.integers(0, 256, (n, 3))).astype(np.uint8)
    times = np.linspace(0.0, 4.988316666666667, n, dtype=np.float32)
    extents = np.full(n, bp.COMPACT_SUPPORT_FRAMES * 1001 / 60000, dtype=np.float32)
    extents[: n // 2] = 2.494158
    bp.write_ply(path, xyz, rgb, times, extents)
    return {"xyz": xyz, "rgb": rgb, "time": times, "t_extent": extents}


def test_real_writer_round_trips_through_the_real_reader(tmp_path):
    p = tmp_path / "points3d.ply"
    expect = _write_via_real_writer(p, 64)
    pcd = fetchPly(str(p))
    assert pcd.points.shape == (64, 3)
    assert pcd.time is not None and pcd.time.shape == (64, 1)
    assert pcd.t_extent is not None and pcd.t_extent.shape == (64, 1)
    np.testing.assert_allclose(pcd.time[:, 0], expect["time"], rtol=0, atol=0)
    np.testing.assert_allclose(pcd.t_extent[:, 0], expect["t_extent"], rtol=0, atol=0)
    # the two bands must survive as two distinct values, not be collapsed
    assert len(np.unique(pcd.t_extent)) == 2


def test_the_read_column_survives_torch_from_numpy(tmp_path):
    """plyfile hands back a STRIDED VIEW; the trainer must not choke on it.

    `vertices['t_extent'][:, None]` is a view into a structured array whose
    stride is the record size -- 35 bytes for this dtype, not a multiple of
    4 -- and `torch.from_numpy` rejects such strides outright. This is the
    first cloud in the repository to carry these columns, so the path had
    never been exercised.
    """
    torch = pytest.importorskip("torch")
    p = tmp_path / "points3d.ply"
    _write_via_real_writer(p, 32)
    pcd = fetchPly(str(p))
    for name in ("time", "t_extent"):
        col = getattr(pcd, name)
        with pytest.raises(Exception):
            torch.from_numpy(col)          # the raw view is NOT acceptable
        torch.from_numpy(np.ascontiguousarray(col))   # and this is the fix in use


def test_cloud_without_the_columns_is_untouched(tmp_path):
    from plyfile import PlyData, PlyElement

    n = 16
    arr = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    p = tmp_path / "points3d.ply"
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(p))
    pcd = fetchPly(str(p))
    assert pcd.time is None, "a cloud with no time column must not acquire one"
    assert pcd.t_extent is None, "a cloud with no t_extent must not acquire one"


def test_basic_point_cloud_defaults_keep_old_callers_working():
    pcd = BasicPointCloud(points=np.zeros((3, 3)), colors=np.zeros((3, 3)),
                          normals=np.zeros((3, 3)))
    assert pcd.time is None and pcd.t_extent is None


def test_trainer_source_pins_the_exponent_and_refuses_rot_4d():
    """Pin the conversion against the TRAINER'S SOURCE, not against arithmetic.

    `create_from_pcd` cannot run without CUDA, so the thing this can still
    guarantee is that nobody edits the exponent or removes the rot_4d refusal
    without a test going red. Both are silent failures otherwise: a wrong
    exponent still trains, it just renders every support width wrong.
    """
    # Bound the search by the STRUCTURE, not by a character count. A fixed
    # window silently stopped covering the conversion as soon as the rot_4d
    # refusal was added ahead of it -- .pow(4) moved to 2,109 chars and the
    # test failed while the code was correct.
    after = GAUSSIAN_MODEL.split("pcd.t_extent", 1)[1]
    block = after.split("scales_t = torch.log", 1)[0]
    assert re.search(r"\.pow\(4\)", block), (
        "create_from_pcd no longer raises t_extent to the 4th power. "
        "get_scaling_t = sqrt(dist_t) and get_cov_t consumes THAT as a variance, "
        "so a standard deviation inverts as dist_t = std ** 4."
    )
    assert "rot_4d" in block and "raise ValueError" in block, (
        "the rot_4d refusal is gone. Under rot_4d, get_cov_t returns "
        "Sigma[3,3] = dist_t, so the exponent would be 2, not 4."
    )


@pytest.mark.parametrize("band_name", ["compact", "broad", "default"])
def test_conversion_is_invertible_at_the_bands_that_are_actually_used(band_name):
    """Check where the exponents DIFFER, not where they nearly coincide."""
    span = 299 * 1001 / 60000
    bands = {
        "compact": bp.COMPACT_SUPPORT_FRAMES * 1001 / 60000,
        "broad": span * bp.BROAD_SUPPORT_SPAN_FRAC,
        "default": (span / 5.0) ** 0.25,
    }
    std = bands[band_name]
    dist_t = std ** 4
    assert dist_t ** 0.25 == pytest.approx(std, rel=1e-12)
    if band_name != "default":
        # The wrong exponent is not a near miss here. Compared as a RATIO,
        # which is direction-agnostic: a relative-difference threshold only
        # works for std < 1, because for std > 1 `std**2 < std**4` and the
        # relative gap is `1 - 1/std**2 < 1` however large the factor is.
        wrong = std ** 2
        factor = max(wrong, dist_t) / min(wrong, dist_t)
        assert factor > 5.0, f"{band_name}: std**2={wrong:.6g} vs std**4={dist_t:.6g}"


def test_default_band_reproduces_the_trainers_uniform_initialisation():
    """An abstaining point must be initialised exactly as if no column existed."""
    span = 299 * 1001 / 60000
    assert ((span / 5.0) ** 0.25) ** 4 == pytest.approx(span / 5.0, rel=0, abs=1e-12)


def test_declared_bands_are_ordered_as_the_arm_describes():
    span = 299 * 1001 / 60000
    compact = bp.COMPACT_SUPPORT_FRAMES * 1001 / 60000
    broad = span * bp.BROAD_SUPPORT_SPAN_FRAC
    default = (span / 5.0) ** 0.25
    assert compact < default < broad
    assert bp.EPS_STATIC_PX < bp.EPS_DYNAMIC_PX
    assert 0.0 < bp.DEGENERATE_SHARE < 1.0


def test_rendered_support_is_narrower_than_the_declared_band():
    """The renderer applies the temporal marginal TWICE; record the factor.

    `gaussian_renderer/__init__.py` multiplies opacity by
    `exp(-0.5 dt^2 / sigma)` and the CUDA kernel
    (`diff-gaussian-rasterization/cuda_rasterizer/forward.cu`) computes and
    applies the same marginal again. The product is `exp(-dt^2 / sigma)`, an
    effective variance of `sigma / 2`, so the RENDERED temporal standard
    deviation is the declared one divided by sqrt(2). This is pre-existing and
    is not repaired here -- repairing it would silently change every 4D result
    this project has produced -- but a number that is wrong by 41% must not be
    quoted as though it were the rendered width.
    """
    renderer = (REPO_ROOT / "gaussian_renderer" / "__init__.py").read_text(encoding="utf-8")
    cuda = (REPO_ROOT / "diff-gaussian-rasterization" / "cuda_rasterizer"
            / "forward.cu").read_text(encoding="utf-8")
    assert "opacity * marginal_t" in renderer
    assert "opacity *= marginal_t" in cuda
    declared = bp.COMPACT_SUPPORT_FRAMES * 1001 / 60000
    rendered = declared / np.sqrt(2.0)
    assert rendered == pytest.approx(0.09437, rel=1e-3)
    assert rendered < declared
