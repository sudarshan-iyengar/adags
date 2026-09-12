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


def test_centroid_velocity_is_a_bare_float_not_a_dict():
    velocity = sae.centroid_velocity([0.0, 0.0], [6.0, 8.0], 2)
    assert isinstance(velocity, float)
    assert not isinstance(velocity, dict)
    assert math.isclose(velocity, 5.0)


def test_centroid_velocity_detail_carries_the_signed_components():
    detail = sae.centroid_velocity_detail([0.0, 0.0], [6.0, 8.0], 2)
    assert detail["dx_per_frame"] == 3.0
    assert detail["dy_per_frame"] == 4.0
    assert math.isclose(detail["speed_px_per_frame"], 5.0)
    assert detail["frame_gap"] == 2


def test_centroid_velocity_and_its_detail_agree_on_the_speed():
    previous, current, gap = [3.0, -1.0], [-4.5, 6.0], 3
    assert sae.centroid_velocity(previous, current, gap) == pytest.approx(
        sae.centroid_velocity_detail(previous, current, gap)["speed_px_per_frame"])


@pytest.mark.parametrize("previous,current,gap", [
    (None, [1.0, 1.0], 1),
    ([1.0, 1.0], None, 1),
    ([1.0, 1.0], [2.0, 2.0], 0),
])
def test_centroid_velocity_is_none_without_two_endpoints_and_a_gap(previous, current, gap):
    assert sae.centroid_velocity(previous, current, gap) is None
    assert sae.centroid_velocity_detail(previous, current, gap) is None


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


# ------------------------------------------------------- narrowing: scale (2)


def test_scale_distribution_reports_the_four_order_statistics():
    stats = sae.scale_distribution(np.arange(1.0, 101.0))
    assert stats["count"] == 100
    assert stats["median"] == pytest.approx(50.5)
    assert stats["p90"] == pytest.approx(90.1)
    assert stats["p99"] == pytest.approx(99.01)
    assert stats["max"] == 100.0


def test_scale_distribution_of_an_empty_set_is_undefined_not_zero():
    assert sae.scale_distribution(np.zeros(0)) == {
        "count": 0, "median": None, "p90": None, "p99": None, "max": None}


def test_scale_max_keep_drops_only_above_the_factor_times_median():
    values = np.array([1.0, 1.0, 2.0, 3.0, 40.0])      # median 2.0
    keep, stats = sae.scale_max_keep(values, 1.5)      # threshold 3.0
    assert keep.tolist() == [True, True, True, True, False]
    assert stats["threshold"] == pytest.approx(3.0)
    assert stats["factor"] == 1.5
    assert stats["dropped"] == 1
    assert stats["median"] == 2.0


def test_scale_max_keep_is_inclusive_at_the_threshold():
    values = np.array([1.0, 2.0, 3.0])                 # median 2.0, threshold 4.0
    keep, _ = sae.scale_max_keep(values, 2.0)
    assert keep.all()


def test_scale_max_keep_with_no_factor_keeps_everything_but_still_reports():
    values = np.array([1.0, 5.0, 900.0])
    keep, stats = sae.scale_max_keep(values, None)
    assert keep.all()
    assert stats["factor"] is None and stats["threshold"] is None
    assert stats["dropped"] == 0
    assert stats["max"] == 900.0                       # the distribution is still on record


def test_scale_max_keep_rejects_a_non_positive_factor():
    with pytest.raises(ValueError):
        sae.scale_max_keep(np.array([1.0]), 0.0)


# --------------------------------------------------------- narrowing: box (3)


def test_percentile_box_is_per_axis_and_pad_is_a_fraction_of_the_extent():
    points = np.stack([
        np.arange(0.0, 101.0),                 # x: 0..100
        np.arange(0.0, 101.0) * 2.0,           # y: 0..200
        np.zeros(101),                         # z: degenerate
    ], axis=1)
    low, high = sae.percentile_box(points, 10.0, pad=0.0)
    assert low == pytest.approx([10.0, 20.0, 0.0])
    assert high == pytest.approx([90.0, 180.0, 0.0])
    low_pad, high_pad = sae.percentile_box(points, 10.0, pad=0.5)
    assert low_pad == pytest.approx([-30.0, -60.0, 0.0])   # 0.5 * extent per side
    assert high_pad == pytest.approx([130.0, 260.0, 0.0])


def test_percentile_box_with_zero_percentile_is_the_bounding_box():
    points = np.array([[-1.0, 0.0, 3.0], [5.0, 2.0, -4.0]])
    low, high = sae.percentile_box(points, 0.0, pad=0.0)
    assert low == pytest.approx([-1.0, 0.0, -4.0])
    assert high == pytest.approx([5.0, 2.0, 3.0])


@pytest.mark.parametrize("percentile", [-1.0, 50.0, 60.0])
def test_percentile_box_rejects_a_percentile_outside_zero_to_fifty(percentile):
    with pytest.raises(ValueError):
        sae.percentile_box(np.zeros((4, 3)), percentile)


def test_percentile_box_refuses_an_empty_point_set():
    with pytest.raises(ValueError):
        sae.percentile_box(np.zeros((0, 3)), 5.0)


def test_inside_box_is_inclusive_and_needs_all_three_axes():
    points = np.array([
        [0.0, 0.0, 0.0],     # interior
        [1.0, 1.0, 1.0],     # exactly on the upper corner -> inclusive
        [1.5, 0.0, 0.0],     # outside on x only
        [0.0, 0.0, -2.0],    # outside on z only
    ])
    keep = sae.inside_box(points, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    assert keep.tolist() == [True, True, False, False]


def test_the_percentile_box_drops_a_far_outlier_a_scale_filter_would_miss():
    core = np.random.default_rng(0).normal(scale=0.01, size=(200, 3))
    points = np.vstack([core, np.array([[8.0, 8.0, 8.0]])])
    low, high = sae.percentile_box(points, 1.0, pad=0.1)
    keep = sae.inside_box(points, low, high)
    assert not keep[-1]
    assert keep[:-1].mean() > 0.95


# -------------------------------------------------- narrowing: projection (4)


def _identity4():
    return np.eye(4, dtype=np.float64)


def test_project_points_uses_the_rasterizer_ndc_to_pixel_form():
    # full_proj = I -> clip = [x, y, z, 1]; ndc = (x, y).
    xy, valid = sae.project_points(
        np.array([[0.0, 0.0, 1.0]]), _identity4(), _identity4(), width=100, height=50)
    # ndc2Pix(0, S) = ((0 + 1) * S - 1) * 0.5
    assert xy[0, 0] == pytest.approx(49.5)
    assert xy[0, 1] == pytest.approx(24.5)
    assert valid.tolist() == [True]
    # and NOT the (ndc + 1) * 0.5 * (S - 1) form used elsewhere in ADAGS
    assert xy[0, 0] != pytest.approx(49.5 + 0.5)


def test_project_points_treats_the_matrix_as_a_row_vector_multiply():
    # A translation living in the LAST ROW is what `hom @ proj` picks up; if the
    # convention were `proj @ hom` this term would be ignored.
    proj = _identity4()
    proj[3, 0] = 2.0
    xy, _ = sae.project_points(
        np.array([[0.0, 0.0, 1.0]]), proj, _identity4(), width=100, height=100)
    assert xy[0, 0] == pytest.approx(((2.0 + 1.0) * 100 - 1.0) * 0.5)


def test_project_points_applies_the_perspective_divide():
    proj = _identity4()
    proj[2, 3] = 1.0        # w = z
    proj[3, 3] = 0.0
    xy, valid = sae.project_points(
        np.array([[0.5, 0.0, 2.0]]), proj, _identity4(), width=100, height=100)
    assert valid.tolist() == [True]
    assert xy[0, 0] == pytest.approx(((0.25 + 1.0) * 100 - 1.0) * 0.5, abs=1e-3)


def test_project_points_rejects_points_behind_the_near_clip():
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.1], [0.0, 0.0, -3.0]])
    _, valid = sae.project_points(points, _identity4(), _identity4(), 64, 64)
    assert valid.tolist() == [True, False, False]


def test_project_points_without_a_view_matrix_skips_the_depth_test():
    points = np.array([[0.0, 0.0, -3.0]])
    _, valid = sae.project_points(points, _identity4(), None, 64, 64)
    assert valid.tolist() == [True]


def test_project_points_rejects_a_bad_matrix_shape():
    with pytest.raises(ValueError):
        sae.project_points(np.zeros((1, 3)), np.eye(3), None, 10, 10)


# ------------------------------------------------- narrowing: the vote (4)


def test_sample_id_map_reads_the_nearest_pixel_and_bounds_checks():
    ids = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint8)     # H = 2, W = 3
    xy = np.array([
        [1.0, 0.0],      # -> ids[0, 1] = 1
        [2.4, 1.0],      # rounds to column 2 -> ids[1, 2] = 2
        [-0.6, 0.0],     # rounds to -1 -> out of bounds
        [3.0, 0.0],      # out of bounds on the right
        [0.0, 0.0],      # ids[0, 0] = 0, in bounds but unlabelled
    ])
    valid = np.ones(5, dtype=bool)
    sampled, hit = sae.sample_id_map(ids, xy, valid)
    assert hit.tolist() == [True, True, False, False, True]
    assert sampled.tolist() == [1, 2, 0, 0, 0]


def test_sample_id_map_never_reads_a_row_the_caller_already_invalidated():
    ids = np.array([[7, 7], [7, 7]], dtype=np.uint8)
    sampled, hit = sae.sample_id_map(ids, np.array([[0.0, 0.0], [1.0, 1.0]]),
                                     np.array([True, False]))
    assert hit.tolist() == [True, False]
    assert sampled.tolist() == [7, 0]


def test_sample_id_map_survives_non_finite_projections():
    ids = np.zeros((2, 2), dtype=np.uint8)
    sampled, hit = sae.sample_id_map(
        ids, np.array([[np.nan, 0.0], [np.inf, np.inf]]), np.array([True, True]))
    assert hit.tolist() == [False, False]
    assert sampled.tolist() == [0, 0]


def test_harmonise_id_picks_the_modal_non_zero_id():
    ids = np.array([0, 0, 0, 0, 9, 9, 9, 4, 4])
    hit = np.ones(9, dtype=bool)
    chosen, counts = sae.harmonise_id(ids, hit)
    assert chosen == 9                       # 0 is unlabelled and never wins
    assert counts == {0: 4, 4: 2, 9: 3}      # the zero count is still reported


def test_harmonise_id_breaks_a_tie_towards_the_smaller_id():
    chosen, _ = sae.harmonise_id(np.array([12, 12, 5, 5]), np.ones(4, dtype=bool))
    assert chosen == 5


def test_harmonise_id_returns_none_when_only_unlabelled_pixels_were_hit():
    chosen, counts = sae.harmonise_id(np.array([0, 0, 0]), np.ones(3, dtype=bool))
    assert chosen is None
    assert counts == {0: 3}


def test_harmonise_id_ignores_rows_that_did_not_land_in_the_image():
    chosen, counts = sae.harmonise_id(np.array([3, 3, 8, 8, 8]),
                                      np.array([True, True, False, False, False]))
    assert chosen == 3
    assert counts == {3: 2}


def test_consistency_vote_counts_cameras_and_applies_the_floor():
    hits = np.array([
        [True,  True,  False, False],
        [True,  True,  False, True],
        [True,  False, False, False],
    ])
    counts = sae.consistency_counts(hits)
    assert counts.tolist() == [3, 2, 0, 1]
    assert sae.consistency_keep(counts, 2).tolist() == [True, True, False, False]
    assert sae.consistency_keep(counts, 1).tolist() == [True, True, False, True]
    assert sae.count_histogram(counts) == {"0": 1, "1": 1, "2": 1, "3": 1}


def test_frame_survivors_requires_every_anchor_frame_by_default():
    per_frame = np.array([
        [True, True, False],
        [True, False, False],
    ])
    assert sae.frame_survivors(per_frame, 1.0).tolist() == [True, False, False]
    assert sae.frame_survivors(per_frame, 0.5).tolist() == [True, True, False]


def test_frame_survivors_full_fraction_is_not_defeated_by_float_division():
    per_frame = np.ones((3, 2), dtype=bool)            # 3 / 3 is not exact in binary
    assert sae.frame_survivors(per_frame, 1.0).all()


def test_frame_survivors_rejects_a_degenerate_input():
    with pytest.raises(ValueError):
        sae.frame_survivors(np.zeros((0, 4), dtype=bool), 1.0)
    with pytest.raises(ValueError):
        sae.frame_survivors(np.ones((2, 2), dtype=bool), 0.0)


def test_the_consistency_vote_end_to_end_on_synthetic_projections_and_id_maps():
    """Three cameras, six rows: four on the object, one neighbour, one runaway.

    Each camera sees the object under a DIFFERENT DEVA id -- ids are per-camera
    -- so the harmonisation is what makes the vote possible at all.
    """
    height, width = 40, 60
    object_xy = np.array([[20.0, 20.0], [21.0, 20.0], [20.0, 21.0], [21.0, 21.0]])
    neighbour_xy = np.array([[45.0, 10.0]])
    runaway_xy = np.array([[1000.0, 1000.0]])

    camera_ids = [7, 95, 2]
    hits = np.zeros((3, 6), dtype=bool)
    chosen_per_camera = []
    for slot, deva_id in enumerate(camera_ids):
        id_map = np.zeros((height, width), dtype=np.uint8)
        id_map[18:24, 18:24] = deva_id            # the object
        id_map[8:13, 43:48] = 200                 # an unrelated object
        xy = np.vstack([object_xy, neighbour_xy, runaway_xy])
        if slot == 2:
            # camera 2 sees only three of the four object rows on the object
            xy = xy.copy()
            xy[3] = [2.0, 2.0]
        sampled, hit = sae.sample_id_map(id_map, xy, np.ones(6, dtype=bool))
        chosen, _ = sae.harmonise_id(sampled, hit)
        chosen_per_camera.append(chosen)
        hits[slot] = hit & (sampled == chosen)

    assert chosen_per_camera == camera_ids
    counts = sae.consistency_counts(hits)
    assert counts.tolist() == [3, 3, 3, 2, 0, 0]
    assert sae.consistency_keep(counts, 3).tolist() == [True, True, True, False, False, False]
    assert sae.consistency_keep(counts, 2).tolist() == [True, True, True, True, False, False]
    # the runaway never landed in any image, so it is unanimously excluded
    assert counts[5] == 0


# ------------------------------------- narrowing: the visible-only depth test


def test_camera_depth_is_the_row_vector_z_the_near_clip_test_uses():
    view = _identity4()
    depth = sae.camera_depth(np.array([[0.0, 0.0, 3.0], [1.0, 2.0, -4.0]]), view)
    assert depth.tolist() == [3.0, -4.0]
    # a translation in the LAST ROW is what `hom @ view` picks up
    view[3, 2] = 10.0
    assert sae.camera_depth(np.array([[0.0, 0.0, 3.0]]), view).tolist() == [13.0]


def test_camera_depth_agrees_with_the_near_clip_limb_of_project_points():
    view = _identity4()
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.1], [0.0, 0.0, -3.0]])
    _, valid = sae.project_points(points, _identity4(), view, 64, 64)
    assert valid.tolist() == (sae.camera_depth(points, view) > sae.NEAR_CLIP_Z).tolist()


def test_camera_depth_rejects_bad_shapes():
    with pytest.raises(ValueError):
        sae.camera_depth(np.zeros((2, 2)), _identity4())
    with pytest.raises(ValueError):
        sae.camera_depth(np.zeros((2, 3)), np.eye(3))


def test_depth_consistent_keeps_only_rows_near_the_median_of_the_first_pass():
    # Four rows on the surface at depth ~5, one occluder at 2 that the FIRST pass
    # wrongly called consistent, one row behind at 9, one the first pass rejected.
    depth = np.array([5.0, 5.1, 4.9, 5.05, 2.0, 9.0, 5.0])
    first = np.array([True, True, True, True, True, True, False])
    keep, stats = sae.depth_consistent(depth, first, 0.15)
    assert keep.tolist() == [True, True, True, True, False, False, False]
    assert stats["evaluated"] is True
    assert stats["median_depth"] == pytest.approx(5.025)
    assert stats["reference_rows"] == 6
    assert stats["dropped"] == 2
    assert stats["kept"] == 4


def test_depth_consistent_can_only_shrink_the_first_pass():
    rng = np.random.default_rng(11)
    depth = rng.normal(size=64)
    first = rng.random(64) > 0.5
    keep, _ = sae.depth_consistent(depth, first, 0.3)
    assert (keep & ~first).sum() == 0


def test_depth_consistent_is_two_sided_about_the_median():
    depth = np.array([0.0, 0.0, 0.0, 0.5, -0.5])
    first = np.ones(5, dtype=bool)
    keep, stats = sae.depth_consistent(depth, first, 0.2)
    assert stats["median_depth"] == pytest.approx(0.0)
    assert keep.tolist() == [True, True, True, False, False]   # in front drops too


def test_depth_consistent_is_inclusive_at_the_slack():
    keep, _ = sae.depth_consistent(np.array([0.0, 0.0, 0.15, -0.15]),
                                   np.ones(4, dtype=bool), 0.15)
    assert keep.all()


def test_depth_consistent_without_a_reference_row_does_not_invent_one():
    """No first-pass row means no surface to measure: nothing is dropped."""
    keep, stats = sae.depth_consistent(np.array([1.0, 2.0]), np.zeros(2, dtype=bool), 0.1)
    assert keep.tolist() == [False, False]
    assert stats["evaluated"] is False
    assert stats["median_depth"] is None
    assert stats["dropped"] == 0


def test_depth_consistent_drops_non_finite_depths_and_never_medians_them():
    depth = np.array([4.0, 4.0, np.nan, np.inf])
    first = np.ones(4, dtype=bool)
    keep, stats = sae.depth_consistent(depth, first, 0.5)
    assert keep.tolist() == [True, True, False, False]
    assert stats["median_depth"] == pytest.approx(4.0)
    assert stats["reference_rows"] == 2
    assert stats["dropped"] == 2


def test_depth_consistent_rejects_a_non_positive_slack_and_a_shape_mismatch():
    with pytest.raises(ValueError):
        sae.depth_consistent(np.zeros(3), np.ones(3, dtype=bool), 0.0)
    with pytest.raises(ValueError):
        sae.depth_consistent(np.zeros(3), np.ones(4, dtype=bool), 0.1)


def test_the_two_pass_vote_removes_a_row_that_hides_behind_the_object():
    """Two cameras, four rows. One row projects ONTO the object in both cameras
    but sits well behind it, which is exactly what the single-pass vote cannot
    see: without the depth test it survives a 2-of-2 vote, and with it, it does
    not."""
    height, width = 30, 30
    # rows: 3 on the object surface, 1 behind it but on the same pixels
    xy = np.array([[15.0, 15.0], [16.0, 15.0], [15.0, 16.0], [15.5, 15.5]])
    depths = np.array([5.0, 5.02, 4.98, 8.0])

    hits_single = np.zeros((2, 4), dtype=bool)
    hits_two_pass = np.zeros((2, 4), dtype=bool)
    for slot, deva_id in enumerate([7, 95]):
        id_map = np.zeros((height, width), dtype=np.uint8)
        id_map[13:19, 13:19] = deva_id
        sampled, hit = sae.sample_id_map(id_map, xy, np.ones(4, dtype=bool))
        chosen, _ = sae.harmonise_id(sampled, hit)
        first = hit & (sampled == chosen)
        hits_single[slot] = first
        hits_two_pass[slot], stats = sae.depth_consistent(depths, first, 0.15)
        assert stats["evaluated"] is True

    assert sae.consistency_keep(sae.consistency_counts(hits_single), 2).tolist() == [
        True, True, True, True]
    assert sae.consistency_keep(sae.consistency_counts(hits_two_pass), 2).tolist() == [
        True, True, True, False]


# ------------------------------------------------------------- filter ordering


def test_filter_order_puts_the_mask_vote_first_only_for_base_rows_all():
    assert sae.filter_order("ids") == [
        "argmax_only", "scale_max_factor", "box_percentile", "mask_consistency"]
    assert sae.filter_order("all") == [
        "mask_consistency", "scale_max_factor", "box_percentile"]
    # `all` drops argmax_only entirely: it names a limb of the identity selection
    assert "argmax_only" not in sae.filter_order("all")


def test_modal_value_ignores_missing_entries_and_breaks_ties_low():
    assert sae.modal_value([5, 5, None, 9]) == 5
    assert sae.modal_value([9, 3]) == 3
    assert sae.modal_value([None, None]) is None


# -------------------------------------------------- preview mask-fit reports


def test_alpha_threshold_counts_uses_the_three_fixed_thresholds():
    alpha = np.array([[0.0, 0.3], [0.6, 0.9]])
    assert sae.alpha_threshold_counts(alpha) == {"0.25": 3, "0.5": 2, "0.75": 1}


def test_mask_fit_measures_both_directions_of_the_overlap():
    alpha_obj = np.zeros((4, 4))
    alpha_obj[1:3, 1:3] = 1.0            # 4 object pixels
    ids = np.zeros((4, 4), dtype=np.uint8)
    ids[1:3, 1:2] = 95                   # 2 pixels of the DEVA object, both inside
    ids[3, 3] = 95                       # 1 pixel outside the render's mask
    fit = sae.mask_fit(alpha_obj, ids, 95)
    assert fit["object_px"] == 4
    assert fit["inside_px"] == 2
    assert fit["inside_fraction"] == 0.5
    assert fit["deva_id_px"] == 3
    assert fit["recall_of_deva_id"] == pytest.approx(2.0 / 3.0)


def test_mask_fit_without_a_harmonised_id_reports_nothing_rather_than_zero():
    fit = sae.mask_fit(np.ones((2, 2)), np.zeros((2, 2), dtype=np.uint8), None)
    assert fit["deva_id"] is None
    assert fit["inside_fraction"] is None


def test_mask_fit_of_an_empty_object_mask_is_undefined_not_zero():
    fit = sae.mask_fit(np.zeros((2, 2)), np.full((2, 2), 95, dtype=np.uint8), 95)
    assert fit["object_px"] == 0
    assert fit["inside_fraction"] is None
    assert fit["deva_id_px"] == 4


def test_mask_fit_rejects_a_mismatched_id_map():
    with pytest.raises(ValueError):
        sae.mask_fit(np.zeros((2, 2)), np.zeros((3, 3), dtype=np.uint8), 1)


# --------------------------------------------- edit region: DEVA silhouette


def test_silhouette_unions_every_listed_id():
    ids = np.array([
        [0, 1, 1, 2],
        [0, 1, 3, 2],
        [4, 4, 3, 0],
    ], dtype=np.uint8)
    single = sae.silhouette_from_id_map(ids, [1])
    assert single.sum() == 3
    both = sae.silhouette_from_id_map(ids, [1, 2])
    assert both.sum() == 5
    assert both[0, 3] and both[1, 3] and both[0, 1]
    assert not both[2, 0]                      # id 4 was not asked for
    # the union is exactly the two singles, and order does not matter
    assert np.array_equal(both, single | sae.silhouette_from_id_map(ids, [2]))
    assert np.array_equal(both, sae.silhouette_from_id_map(ids, [2, 1]))


def test_silhouette_of_an_absent_id_is_empty_not_an_error():
    ids = np.array([[0, 7], [7, 0]], dtype=np.uint8)
    assert sae.silhouette_from_id_map(ids, [95]).sum() == 0


def test_silhouette_refuses_the_unlabelled_id_and_an_empty_id_list():
    ids = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        sae.silhouette_from_id_map(ids, [0])
    with pytest.raises(ValueError):
        sae.silhouette_from_id_map(ids, [95, 0])
    with pytest.raises(ValueError):
        sae.silhouette_from_id_map(ids, [])


def test_silhouette_rejects_a_non_2d_id_map():
    with pytest.raises(ValueError):
        sae.silhouette_from_id_map(np.zeros((2, 2, 3), dtype=np.uint8), [1])


# ------------------------------------------------------- the alpha guard


def test_alpha_guard_off_returns_the_silhouette_untouched():
    sil = np.zeros((5, 5), dtype=bool)
    sil[2, 2] = True
    alpha = np.zeros((5, 5))                   # alpha covers NOTHING
    kept, stats = sae.guard_silhouette(sil, alpha, 0)
    assert np.array_equal(kept, sil)           # guard 0 is OFF, not "radius 0"
    assert stats["enabled"] is False
    assert stats["dropped_px"] == 0
    assert stats["guard_px_count"] is None


def test_alpha_guard_intersects_with_the_dilated_alpha():
    sil = np.zeros((11, 11), dtype=bool)
    sil[5, 5] = True                            # inside the alpha
    sil[5, 8] = True                            # 3 px away: inside a radius-3 guard
    sil[0, 0] = True                            # a leak far from the alpha
    alpha = np.zeros((11, 11))
    alpha[5, 5] = 0.9
    kept, stats = sae.guard_silhouette(sil, alpha, 3)
    assert kept[5, 5] and kept[5, 8]
    assert not kept[0, 0]
    assert stats["enabled"] is True
    assert stats["silhouette_px"] == 3 and stats["kept_px"] == 2 and stats["dropped_px"] == 1
    assert stats["guard_px_count"] == int(sae.dilate_binary(alpha > 0.5, 3).sum())


def test_alpha_guard_can_only_shrink_the_silhouette():
    rng = np.random.default_rng(3)
    sil = rng.random((12, 12)) > 0.5
    alpha = rng.random((12, 12))
    kept, _ = sae.guard_silhouette(sil, alpha, 2)
    assert (kept & ~sil).sum() == 0            # never adds a pixel DEVA did not label


def test_alpha_guard_rejects_a_mismatched_alpha():
    with pytest.raises(ValueError):
        sae.guard_silhouette(np.zeros((4, 4), dtype=bool), np.zeros((5, 5)), 1)


# -------------------------------------------------- support ON the silhouette


def test_silhouette_report_measures_support_and_holes_on_S_not_on_the_alpha():
    sil = np.array([[True, True], [True, False]])
    # alpha_bg: two supported pixels, one hole, and one pixel OUTSIDE S that
    # would change the answer if the report leaked past the silhouette.
    alpha_bg = np.array([[0.8, 0.6], [0.1, 0.0]])
    alpha_obj = np.array([[0.9, 0.2], [0.9, 1.0]])
    report = sae.silhouette_report(sil, alpha_obj, alpha_bg)
    assert report["silhouette_px"] == 3
    assert report["hole_px"] == 1
    assert report["hole_fraction"] == pytest.approx(1.0 / 3.0)
    assert report["support_inside_silhouette"] == pytest.approx((0.8 + 0.6 + 0.1) / 3.0)
    # 2 of the 3 silhouette pixels are covered by the object alpha
    assert report["alpha_cover_of_silhouette"] == pytest.approx(2.0 / 3.0)


def test_silhouette_report_disagrees_with_the_alpha_based_hole_number():
    """The whole point of the flag: the two regions give different answers."""
    sil = np.zeros((4, 4), dtype=bool)
    sil[0, :] = True                     # the segmenter's row
    alpha_obj = np.zeros((4, 4))
    alpha_obj[:, 0] = 1.0                # the smear: a different column
    alpha_bg = np.zeros((4, 4))
    alpha_bg[0, :] = 1.0                 # support exists exactly under S
    on_silhouette = sae.silhouette_report(sil, alpha_obj, alpha_bg)
    on_alpha = sae.hole_fraction(alpha_obj, alpha_bg)
    assert on_silhouette["hole_fraction"] == 0.0
    assert on_alpha["hole_fraction"] == pytest.approx(0.75)


def test_silhouette_report_of_an_empty_silhouette_is_undefined_not_zero():
    report = sae.silhouette_report(np.zeros((3, 3), dtype=bool), np.ones((3, 3)),
                                   np.ones((3, 3)))
    assert report["silhouette_px"] == 0
    assert report["hole_fraction"] is None
    assert report["support_inside_silhouette"] is None
    assert report["alpha_cover_of_silhouette"] is None


def test_silhouette_report_rejects_mismatched_maps():
    with pytest.raises(ValueError):
        sae.silhouette_report(np.zeros((2, 2), dtype=bool), np.zeros((2, 2)), np.zeros((3, 3)))


# ------------------------------------------ the composite region derived from S


def test_edit_alpha_from_mask_dilates_then_feathers_a_binary_region():
    sil = np.zeros((15, 15), dtype=bool)
    sil[7, 7] = True
    weight = sae.edit_alpha_from_mask(sil, dilate=2, feather=0.0)
    assert weight[7, 7] == 1.0
    assert weight[7, 5] == 1.0                     # dilated
    assert weight[7, 4] == 0.0
    assert weight.sum() == 13                      # the exact disk of radius 2


def test_edit_alpha_from_mask_does_not_rethreshold_the_silhouette():
    """S is already binary: every labelled pixel must survive, none must be added."""
    sil = np.zeros((7, 7), dtype=bool)
    sil[3, 3] = True
    weight = sae.edit_alpha_from_mask(sil, dilate=0, feather=0.0)
    assert weight[3, 3] == 1.0
    assert weight.sum() == 1.0


def test_edit_alpha_delegates_to_the_mask_form_so_alpha_mode_is_unchanged():
    alpha = np.zeros((13, 13))
    alpha[6, 6] = 0.9
    alpha[6, 9] = 0.2                               # below threshold
    assert np.array_equal(
        sae.edit_alpha(alpha, 0.5, 3, 1.5),
        sae.edit_alpha_from_mask(sae.binarise(alpha, 0.5), 3, 1.5))


def test_the_composite_region_from_S_replaces_exactly_the_silhouette():
    sil = np.zeros((5, 5), dtype=bool)
    sil[2, 1:4] = True
    weight = sae.edit_alpha_from_mask(sil, dilate=0, feather=0.0)
    real = np.zeros((5, 5, 3))
    replacement = np.ones((5, 5, 3))
    edited = sae.composite_alpha(real, replacement, weight)
    assert edited[2, 2].tolist() == [255, 255, 255]     # inside S -> background render
    assert edited[0, 0].tolist() == [0, 0, 0]           # outside S -> the real image


# --------------------------------------------------------- silhouette outline


def test_mask_outline_is_the_inner_boundary():
    sil = np.zeros((7, 7), dtype=bool)
    sil[2:5, 2:5] = True
    outline = sae.mask_outline(sil)
    assert (outline & ~sil).sum() == 0          # never outside the silhouette
    assert not outline[3, 3]                    # the interior pixel is not an edge
    assert outline.sum() == 8                   # the 3x3 ring


def test_overlay_outline_paints_only_the_outline():
    image = np.zeros((3, 3, 3))
    edge = np.zeros((3, 3), dtype=bool)
    edge[1, 1] = True
    out = sae.overlay_outline(image, edge, colour=(1.0, 0.0, 0.0))
    assert out[1, 1].tolist() == [255, 0, 0]
    assert out[0, 0].tolist() == [0, 0, 0]


def test_overlay_outline_rejects_a_mismatched_outline():
    with pytest.raises(ValueError):
        sae.overlay_outline(np.zeros((2, 2, 3)), np.zeros((3, 3), dtype=bool))


# ------------------------------------------------------ per-camera DEVA ids


def test_parse_ids_by_camera_accepts_scalars_and_lists():
    parsed = sae.parse_ids_by_camera('{"cam00": [125], "cam15": 95, "cam07": [3, 3, 1]}')
    assert parsed == {"cam00": [125], "cam15": [95], "cam07": [1, 3]}


def test_parse_ids_by_camera_reads_a_file_as_well_as_inline_json(tmp_path):
    target = tmp_path / "ids.json"
    target.write_text('{"cam00": [125]}', encoding="utf-8")
    assert sae.parse_ids_by_camera(str(target)) == {"cam00": [125]}


@pytest.mark.parametrize("text", [
    "not json",
    "[95]",                       # not an object
    '{"cam00": []}',              # no ids
    '{"cam00": [0]}',             # DEVA's unlabelled id
    '{"cam00": ["95"]}',          # not an integer
    '{"cam00": [1.5]}',
])
def test_parse_ids_by_camera_refuses_a_mapping_that_cannot_mean_anything(text):
    with pytest.raises(ValueError):
        sae.parse_ids_by_camera(text)


def test_resolve_deva_ids_prefers_the_explicit_mapping():
    assert sae.resolve_deva_ids({"cam15": [95]}, "cam15", {"cam15": 7}) == [95]


def test_resolve_deva_ids_falls_back_to_the_harmonised_id():
    assert sae.resolve_deva_ids({"cam00": [125]}, "cam07", {"cam07": 42}) == [42]


def test_resolve_deva_ids_refuses_a_camera_with_no_id_at_all():
    # cam00 is held out, so it is never harmonised and must be given explicitly.
    with pytest.raises(ValueError):
        sae.resolve_deva_ids({"cam15": [95]}, "cam00", {"cam15": 95})
    with pytest.raises(ValueError):
        sae.resolve_deva_ids({}, "cam07", None)


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


def test_every_narrowing_knob_is_off_by_default():
    args = sae.build_parser().parse_args([
        "--mode", "preview", "--model_path", "m", "--cam_view", "cam15",
        "--ids", "95", "--out", "o", "--cameras", "0", "--frames", "30",
    ])
    assert args.argmax_only is False
    assert args.scale_max_factor is None
    assert args.box_percentile is None
    assert args.mask_consistency is None
    assert args.mask_min_cams is None
    assert args.mask_frames is None
    # the two knobs that DO carry a default
    assert args.box_pad == 0.1
    assert args.mask_min_frames == 1.0
    sae._validate_filter_args(args)          # the default CLI must validate


def _narrowing_args(**overrides):
    argv = [
        "--mode", "preview", "--model_path", "m", "--cam_view", "cam15",
        "--ids", "95", "--out", "o", "--cameras", "0", "--frames", "30",
    ]
    args = sae.build_parser().parse_args(argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_mask_consistency_requires_its_two_companions():
    with pytest.raises(SystemExit):
        sae._validate_filter_args(_narrowing_args(mask_consistency="d", mask_frames="30"))
    with pytest.raises(SystemExit):
        sae._validate_filter_args(_narrowing_args(mask_consistency="d", mask_min_cams=3))
    sae._validate_filter_args(
        _narrowing_args(mask_consistency="d", mask_min_cams=3, mask_frames="30,60"))


def test_the_mask_companions_are_refused_without_a_deva_root():
    with pytest.raises(SystemExit):
        sae._validate_filter_args(_narrowing_args(mask_min_cams=3))
    with pytest.raises(SystemExit):
        sae._validate_filter_args(_narrowing_args(mask_frames="30"))


@pytest.mark.parametrize("overrides", [
    {"scale_max_factor": 0.0},
    {"scale_max_factor": -1.0},
    {"box_percentile": 50.0},
    {"box_percentile": -0.5},
    {"box_pad": -0.1},
    {"mask_min_frames": 0.0},
    {"mask_min_frames": 1.5},
    {"mask_consistency": "d", "mask_min_cams": 0, "mask_frames": "30"},
    {"mask_consistency": "d", "mask_min_cams": 3, "mask_frames": "40-30"},
])
def test_validate_filter_args_rejects_meaningless_knobs(overrides):
    with pytest.raises(SystemExit):
        sae._validate_filter_args(_narrowing_args(**overrides))


def test_filter_config_records_the_knobs_verbatim_and_their_order():
    config = sae._filter_config(_narrowing_args(
        argmax_only=True, scale_max_factor=3.0, box_percentile=2.0, box_pad=0.25,
        mask_consistency="/deva", mask_min_cams=4, mask_frames="30,60",
        mask_min_frames=0.5))
    assert config["argmax_only"] is True
    assert config["scale_max_factor"] == 3.0
    assert config["box_percentile"] == 2.0
    assert config["box_pad"] == 0.25
    assert config["mask_consistency_root"] == "/deva"
    assert config["mask_min_cams"] == 4
    assert config["mask_frames"] == "30,60"
    assert config["mask_min_frames"] == 0.5
    assert config["order"] == [
        "argmax_only", "scale_max_factor", "box_percentile", "mask_consistency"]


def test_filter_config_of_a_default_run_carries_no_mask_min_cams():
    config = sae._filter_config(_narrowing_args())
    assert config["mask_consistency_root"] is None
    assert config["mask_min_cams"] is None
    assert config["scale_max_factor"] is None


# ------------------------------------------ CLI: --base_rows and the depth test


def test_the_base_row_set_and_the_depth_test_are_off_by_default():
    args = _narrowing_args()
    assert args.base_rows == "ids"
    assert args.mask_visible_only is False
    assert args.mask_depth_slack == sae.MASK_DEPTH_SLACK_DEFAULT == 0.15
    sae._validate_filter_args(args)                     # the default CLI must validate
    config = sae._filter_config(args)
    assert config["base_rows"] == "ids"
    assert config["mask_visible_only"] is False
    assert config["order"] == sae.filter_order("ids")   # unchanged by the new flags


def test_base_rows_all_is_refused_without_the_mask_vote_that_defines_it():
    with pytest.raises(SystemExit):
        sae._validate_filter_args(_narrowing_args(base_rows="all"))


def test_base_rows_all_refuses_argmax_only_rather_than_ignoring_it():
    with pytest.raises(SystemExit):
        sae._validate_filter_args(_narrowing_args(
            base_rows="all", argmax_only=True,
            mask_consistency="/deva", mask_min_cams=8, mask_frames="40,60"))


def test_base_rows_all_with_the_mask_vote_validates_and_reorders_the_filters():
    args = _narrowing_args(base_rows="all", mask_consistency="/deva",
                           mask_min_cams=8, mask_frames="40,60,89,240",
                           mask_visible_only=True)
    sae._validate_filter_args(args)
    config = sae._filter_config(args)
    assert config["base_rows"] == "all"
    assert config["order"][0] == "mask_consistency"
    assert config["order"] == ["mask_consistency", "scale_max_factor", "box_percentile"]
    assert config["mask_visible_only"] is True
    assert config["mask_depth_slack"] == 0.15
    assert config["mask_min_cams"] == 8
    assert config["mask_frames"] == "40,60,89,240"


def test_the_depth_test_is_refused_without_the_vote_it_is_a_second_pass_over():
    with pytest.raises(SystemExit):
        sae._validate_filter_args(_narrowing_args(mask_visible_only=True))


def test_a_tuned_depth_slack_is_refused_rather_than_silently_ignored():
    with pytest.raises(SystemExit):
        sae._validate_filter_args(_narrowing_args(mask_depth_slack=0.4))
    # ... and accepted once the test it belongs to is switched on
    sae._validate_filter_args(_narrowing_args(
        mask_visible_only=True, mask_depth_slack=0.4,
        mask_consistency="/deva", mask_min_cams=8, mask_frames="40"))


@pytest.mark.parametrize("slack", [0.0, -0.1])
def test_a_non_positive_depth_slack_is_refused(slack):
    with pytest.raises(SystemExit):
        sae._validate_filter_args(_narrowing_args(
            mask_visible_only=True, mask_depth_slack=slack,
            mask_consistency="/deva", mask_min_cams=8, mask_frames="40"))


# ------------------------------------------------------ CLI: the edit region


def test_the_edit_region_defaults_to_the_old_alpha_behaviour():
    args = _narrowing_args()
    assert args.edit_region == "alpha"
    assert args.deva_root is None
    assert args.deva_ids_by_camera is None
    assert args.alpha_guard_px == 0
    sae._validate_edit_region_args(args)          # the default CLI must validate


def test_deva_mode_needs_a_root_and_a_source_of_ids():
    with pytest.raises(SystemExit):
        sae._validate_edit_region_args(_narrowing_args(edit_region="deva"))
    with pytest.raises(SystemExit):
        # a root, but nothing that can name an id
        sae._validate_edit_region_args(_narrowing_args(edit_region="deva", deva_root="/deva"))
    sae._validate_edit_region_args(_narrowing_args(
        edit_region="deva", deva_root="/deva", deva_ids_by_camera='{"cam00": [125]}'))
    # harmonisation alone is an acceptable id source
    sae._validate_edit_region_args(_narrowing_args(
        edit_region="deva", deva_root="/deva", mask_consistency="/deva"))


def test_deva_knobs_are_refused_rather_than_ignored_in_alpha_mode():
    for overrides in ({"deva_root": "/deva"},
                      {"deva_ids_by_camera": '{"cam00": [125]}'},
                      {"alpha_guard_px": 4}):
        with pytest.raises(SystemExit):
            sae._validate_edit_region_args(_narrowing_args(**overrides))


def test_a_malformed_id_mapping_is_rejected_before_any_gpu_work():
    with pytest.raises(SystemExit):
        sae._validate_edit_region_args(_narrowing_args(
            edit_region="deva", deva_root="/deva", deva_ids_by_camera='{"cam00": [0]}'))


def test_a_negative_alpha_guard_is_refused():
    with pytest.raises(SystemExit):
        sae._validate_edit_region_args(_narrowing_args(
            edit_region="deva", deva_root="/deva",
            deva_ids_by_camera='{"cam00": [125]}', alpha_guard_px=-1))


def test_edit_region_config_records_the_mode_the_ids_and_their_source():
    args = _narrowing_args(edit_region="deva", deva_root="/deva", alpha_guard_px=3,
                           deva_ids_by_camera='{"cam00": [125]}')
    config = sae._edit_region_config(
        args,
        {"cam00": [125]},
        {"cam00": [125], "cam07": [42]},
        {"cam00": {"0030": 51234}},
    )
    assert config["mode"] == "deva"
    assert config["deva_root"] == "/deva"
    assert config["alpha_guard_px"] == 3
    assert config["ids_by_camera"] == {"cam00": [125], "cam07": [42]}
    assert config["ids_source"] == {"cam00": "explicit",
                                    "cam07": "mask_consistency_harmonised"}
    assert config["silhouette_px"] == {"cam00": {"0030": 51234}}


def test_edit_region_config_of_an_alpha_run_carries_no_silhouette_keys():
    config = sae._edit_region_config(_narrowing_args(), {}, {}, {})
    assert config["mode"] == "alpha"
    assert "silhouette_px" not in config
    assert "ids_by_camera" not in config


# --- the build's post-vote IQR box (2026-09-12) --------------------------------
# The box dropped the bottle cap's rows on flame_steak (240 of 2,230 voted rows)
# while the preview, which has no box, was cap-free. The knob must default to
# the wave-1 constant and must disable the box at <= 0.


def test_hull_outlier_factor_defaults_to_the_wave1_constant():
    args = sae.build_parser().parse_args(["--mode", "build"] + _minimal_build_argv())
    assert args.hull_outlier_factor is None
    assert sae.effective_hull_outlier_factor(args) == sae.IQR_OUTLIER_FACTOR == 1.0


def test_hull_outlier_factor_zero_disables_the_box_and_is_recorded():
    args = sae.build_parser().parse_args(
        ["--mode", "build", "--hull_outlier_factor", "0"] + _minimal_build_argv()
    )
    assert sae.effective_hull_outlier_factor(args) == 0.0
    assert not (sae.effective_hull_outlier_factor(args) > 0)
    assert sae._filter_config(args)["hull_outlier_factor"] == 0.0


def test_hull_outlier_factor_explicit_value_is_used():
    args = sae.build_parser().parse_args(
        ["--mode", "build", "--hull_outlier_factor", "2.5"] + _minimal_build_argv()
    )
    assert sae.effective_hull_outlier_factor(args) == 2.5


def _minimal_build_argv():
    """The required options other than --mode, with dummies; nothing here
    touches the filesystem or the GPU (parse_args only)."""
    return ["--model_path", "x", "--cam_view", "x", "--ids", "1", "--out", "x"]
