"""Tests for the pure helpers of ``scripts/sa4d_absence_edit_render.py``.

Only the numpy-only, torch-free helpers are exercised. Everything in that
script that touches SA4D, CUDA or the filesystem is deliberately out of scope:
the authoring workstation has no torch, and a test that mocked the rasterizer
would be testing the mock.

The load-bearing properties are:

* ``alpha_from_two_renders`` inverts constant-background compositing EXACTLY;
* ``gaussian_feather`` of a saturated mask stays saturated, so a feathered
  full-frame mask still fully replaces the frame;
* ``apply_subset_filter`` maps a filter result that is indexed WITHIN the kept
  rows back onto the full row space -- the one indexing step in the pipeline
  that is silent when wrong;
* ``majority_rowset`` rejects an exact tie.
"""

import importlib.util
import math
import os
import sys

import numpy as np
import pytest

_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "scripts", "sa4d_absence_edit_render.py")
_SPEC = importlib.util.spec_from_file_location("sa4d_absence_edit_render", _SCRIPT)
sae = importlib.util.module_from_spec(_SPEC)
sys.modules["sa4d_absence_edit_render"] = sae
_SPEC.loader.exec_module(sae)


# ---------------------------------------------------------------- frame lists


def test_parse_frame_list_ranges_are_inclusive():
    assert sae.parse_frame_list("25-105,215-275")[:3] == [25, 26, 27]
    parsed = sae.parse_frame_list("25-105,215-275")
    assert parsed[-1] == 275
    assert len(parsed) == (105 - 25 + 1) + (275 - 215 + 1)


def test_parse_frame_list_mixes_singles_and_sorts_and_dedupes():
    assert sae.parse_frame_list("9, 3,5-7, 5") == [3, 5, 6, 7, 9]


@pytest.mark.parametrize("spec", ["", "  ,  ", "10-4", "1-2-3", "-5"])
def test_parse_frame_list_rejects_bad_input(spec):
    with pytest.raises(ValueError):
        sae.parse_frame_list(spec)


# --------------------------------------------------------------------- alpha


def test_alpha_from_two_renders_is_exact_for_constant_backgrounds():
    rng = np.random.default_rng(0)
    alpha = rng.random((4, 6))
    colour = rng.random((3, 4, 6)) * alpha  # premultiplied
    white = colour + (1.0 - alpha) * 1.0
    black = colour + (1.0 - alpha) * 0.0
    recovered = sae.alpha_from_two_renders(white, black)
    assert np.allclose(recovered, alpha, atol=1e-12)


def test_alpha_from_two_renders_saturates_at_the_ends():
    opaque_w = np.full((3, 2, 2), 0.4)
    opaque_b = np.full((3, 2, 2), 0.4)
    assert np.allclose(sae.alpha_from_two_renders(opaque_w, opaque_b), 1.0)
    empty_w = np.ones((3, 2, 2))
    empty_b = np.zeros((3, 2, 2))
    assert np.allclose(sae.alpha_from_two_renders(empty_w, empty_b), 0.0)


def test_alpha_from_two_renders_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        sae.alpha_from_two_renders(np.zeros((3, 2, 2)), np.zeros((3, 2, 3)))


def test_alpha_honours_the_channel_axis():
    hwc_white = np.ones((2, 2, 3))
    hwc_black = np.zeros((2, 2, 3))
    assert sae.alpha_from_two_renders(hwc_white, hwc_black, channel_axis=2).shape == (2, 2)


# ------------------------------------------------------------ mask machinery


def test_dilate_binary_grows_a_point_into_a_disk():
    mask = np.zeros((11, 11), dtype=bool)
    mask[5, 5] = True
    grown = sae.dilate_binary(mask, 2)
    assert grown[5, 5] and grown[5, 7] and grown[3, 5]
    assert not grown[3, 3]          # (2, 2) is outside radius 2
    assert not grown[5, 8]
    assert grown.sum() == 13        # the exact disk of radius 2


def test_dilate_binary_zero_radius_is_a_copy_not_an_alias():
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    out = sae.dilate_binary(mask, 0)
    out[0, 0] = True
    assert not mask[0, 0]


def test_dilate_binary_clips_at_the_border():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    grown = sae.dilate_binary(mask, 1)
    assert grown[0, 1] and grown[1, 0] and grown[0, 0]
    assert grown.sum() == 3


def test_gaussian_feather_preserves_a_saturated_mask():
    full = np.ones((9, 9))
    assert np.allclose(sae.gaussian_feather(full, 2.0), 1.0)


def test_gaussian_feather_conserves_nothing_at_the_edges_but_stays_bounded():
    mask = np.zeros((21, 21))
    mask[8:13, 8:13] = 1.0
    blurred = sae.gaussian_feather(mask, 2.0)
    assert 0.0 <= blurred.min() and blurred.max() <= 1.0
    assert blurred[10, 10] > blurred[10, 14] > 0.0   # falls off outward
    assert blurred[10, 14] > mask[10, 14]            # the edge really feathers


def test_gaussian_feather_with_zero_sigma_is_a_no_op():
    mask = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.array_equal(sae.gaussian_feather(mask, 0.0), mask)


def test_edit_alpha_binarises_then_dilates_then_feathers():
    alpha = np.zeros((15, 15))
    alpha[7, 7] = 0.9
    alpha[7, 9] = 0.2                      # below threshold, must be dropped
    weight = sae.edit_alpha(alpha, threshold=0.5, dilate=2, feather=0.0)
    assert weight[7, 7] == 1.0
    assert weight[7, 5] == 1.0             # dilated
    assert weight[7, 11] == 0.0            # the 0.2 pixel never entered


# ---------------------------------------------------------------- compositing


def test_composite_alpha_endpoints_are_the_two_sources():
    real = np.zeros((2, 2, 3))
    replacement = np.ones((2, 2, 3))
    keep_real = sae.composite_alpha(real, replacement, np.zeros((2, 2)))
    take_replacement = sae.composite_alpha(real, replacement, np.ones((2, 2)))
    assert np.array_equal(keep_real, np.zeros((2, 2, 3), dtype=np.uint8))
    assert np.array_equal(take_replacement, np.full((2, 2, 3), 255, dtype=np.uint8))


def test_composite_alpha_blends_linearly():
    real = np.zeros((1, 1, 3))
    replacement = np.ones((1, 1, 3))
    blended = sae.composite_alpha(real, replacement, np.array([[0.5]]))
    assert blended.dtype == np.uint8
    assert blended.tolist() == [[[128, 128, 128]]]


def test_composite_alpha_rejects_a_mismatched_alpha():
    with pytest.raises(ValueError):
        sae.composite_alpha(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), np.zeros((3, 3)))


def test_to_uint8_map_clips_out_of_range_values():
    out = sae.to_uint8_map(np.array([-1.0, 0.0, 0.5, 1.0, 2.0]))
    assert out.tolist() == [0, 0, 128, 255, 255]


# ------------------------------------------------------------------- support


def test_hole_fraction_counts_only_inside_the_object_mask():
    alpha_obj = np.array([[0.9, 0.9], [0.1, 0.1]])
    alpha_bg = np.array([[0.9, 0.2], [0.0, 0.0]])
    result = sae.hole_fraction(alpha_obj, alpha_bg)
    assert result["object_px"] == 2
    assert result["hole_px"] == 1
    assert result["hole_fraction"] == 0.5


def test_hole_fraction_of_an_empty_object_mask_is_undefined_not_zero():
    result = sae.hole_fraction(np.zeros((3, 3)), np.zeros((3, 3)))
    assert result == {"object_px": 0, "hole_px": 0, "hole_fraction": None}


# ------------------------------------------------------------- mask geometry


def test_mask_bbox_centroid_reports_inclusive_bounds():
    mask = np.zeros((5, 7), dtype=bool)
    mask[1, 2] = True
    mask[3, 4] = True
    geometry = sae.mask_bbox_centroid(mask)
    assert geometry["count"] == 2
    assert geometry["bbox"] == [2, 1, 4, 3]
    assert geometry["centroid"] == [3.0, 2.0]


def test_mask_bbox_centroid_of_an_empty_mask():
    assert sae.mask_bbox_centroid(np.zeros((3, 3), dtype=bool)) == {
        "count": 0, "bbox": None, "centroid": None}


def test_centroid_velocity_divides_by_the_frame_gap():
    velocity = sae.centroid_velocity([0.0, 0.0], [6.0, 8.0], 2)
    assert velocity["dx_per_frame"] == 3.0
    assert velocity["dy_per_frame"] == 4.0
    assert math.isclose(velocity["speed_px_per_frame"], 5.0)
    assert velocity["frame_gap"] == 2


@pytest.mark.parametrize("previous,current,gap", [
    (None, [1.0, 1.0], 1),
    ([1.0, 1.0], None, 1),
    ([1.0, 1.0], [2.0, 2.0], 0),
])
def test_centroid_velocity_is_none_without_two_endpoints_and_a_gap(previous, current, gap):
    assert sae.centroid_velocity(previous, current, gap) is None


# -------------------------------------------------------------- row-set logic


def test_majority_rowset_needs_a_strict_majority():
    table = np.array([
        [True, True, False, True],
        [True, False, False, True],
        [False, False, False, True],
    ])
    # column counts 2, 1, 0, 3 out of T = 3
    assert sae.majority_rowset(table).tolist() == [True, False, False, True]


def test_majority_rowset_rejects_an_exact_tie():
    table = np.array([[True, False], [False, False]])   # column 0 is 1 of 2
    assert sae.majority_rowset(table).tolist() == [False, False]


def test_majority_rowset_rejects_a_degenerate_table():
    with pytest.raises(ValueError):
        sae.majority_rowset(np.zeros((0, 4), dtype=bool))
    with pytest.raises(ValueError):
        sae.majority_rowset(np.zeros(4, dtype=bool))


def test_jaccard_basic_and_empty_cases():
    left = np.array([True, True, False, False])
    right = np.array([True, False, True, False])
    assert sae.jaccard(left, right) == pytest.approx(1.0 / 3.0)
    assert sae.jaccard(left, left) == 1.0
    assert sae.jaccard(np.zeros(4, bool), np.zeros(4, bool)) == 1.0


def test_apply_subset_filter_maps_kept_indices_back_to_full_row_space():
    keep = np.array([False, True, True, False, True])
    # sub_keep is indexed within the three kept rows (1, 2, 4): drop the middle.
    result = sae.apply_subset_filter(keep, np.array([True, False, True]))
    assert result.tolist() == [False, True, False, False, True]


def test_apply_subset_filter_does_not_mutate_its_input():
    keep = np.array([True, True, False])
    sae.apply_subset_filter(keep, np.array([False, False]))
    assert keep.tolist() == [True, True, False]


def test_rowset_summary_reports_the_three_sensitivity_sizes():
    hard = np.array([True, False, False, False])
    soft_low = np.array([False, True, False, False])
    soft_high = np.array([False, True, True, False])
    assert sae.rowset_summary(hard, soft_low, soft_high) == {
        "argmax_only": 1,
        "argmax_or_soft_low": 2,
        "argmax_or_soft_high": 3,
    }


# ------------------------------------------------------------------- naming


def test_camera_frame_from_image_path_handles_both_separators():
    assert sae.camera_frame_from_image_path(
        "/data/dynerf/cut_roasted_beef/cam15/images/0060.png") == ("cam15", 60)
    assert sae.camera_frame_from_image_path(
        r"C:\data\cut_roasted_beef\cam00\images\0299.png") == ("cam00", 299)


@pytest.mark.parametrize("path", [
    "/data/cam15/frames/0060.png",     # not an `images` directory
    "0060.png",                        # too short to carry a camera
])
def test_camera_frame_from_image_path_refuses_an_unexpected_layout(path):
    with pytest.raises(ValueError):
        sae.camera_frame_from_image_path(path)


def test_frame_name_zero_pads():
    assert sae.frame_name("cam8", 7) == "cam8_0007"
    assert sae.frame_name("cam00", 299) == "cam00_0299"


def test_sha256_file_returns_none_for_a_missing_file(tmp_path):
    assert sae.sha256_file(str(tmp_path / "nope")) is None
    target = tmp_path / "x.bin"
    target.write_bytes(b"abc")
    assert sae.sha256_file(str(target)) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


# ---------------------------------------------------------------- CLI wiring


def test_parser_rejects_preview_without_cameras():
    with pytest.raises(SystemExit):
        sae.main([
            "--mode", "preview", "--model_path", "m", "--cam_view", "cam15",
            "--ids", "95", "--out", "o",
        ])


def test_parser_rejects_build_without_a_window():
    with pytest.raises(SystemExit):
        sae.main([
            "--mode", "build", "--model_path", "m", "--cam_view", "cam15",
            "--ids", "95", "--out", "o", "--real_images", "r",
        ])


def test_parser_defaults_match_the_documented_cli():
    args = sae.build_parser().parse_args([
        "--mode", "build", "--model_path", "m", "--cam_view", "cam15",
        "--ids", "95", "--out", "o", "--real_images", "r", "--window", "40", "80",
    ])
    assert args.iteration == 14000
    assert args.margin == 20
    assert args.dilate == 6
    assert args.feather == 2.0
    assert args.soft_thresh == sae.SENSITIVITY_SOFT_LOW
