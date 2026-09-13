"""Torch-free unit tests for `scripts/wave2_preconditions.py` on synthetic input.

The precondition scripts of this lane decide whether an instrument was
exercised at all, so a defect in one of them is silent by construction: it
reports a pass, the arm enters the mechanism-exercised set, and the number it
carries is meaningless. These tests fix the arithmetic and, more importantly,
the two CONVENTIONS that a reader cannot check from a JSON: `row_group_ids >= 0`
means member, and an absent interval is `[offset_frame, onset_frame - 1]`
inclusive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import wave2_preconditions as W  # noqa: E402


MEMBERSHIP_THRESHOLDS = {
    "precision_min": 0.80,
    "recall_min": 0.70,
    "size_ratio": [0.5, 2.0],
    "mask_iou_cam15_min": 0.70,
}
T1_THRESHOLDS = {
    "intervals": 1,
    "temporal_iou_min": 0.80,
    "onset_error_max_frames": 3,
    "offset_error_max_frames": 3,
}


# --------------------------------------------------------------------------
# membership counts
# --------------------------------------------------------------------------
def test_member_mask_treats_minus_one_as_non_member():
    mask = W.member_mask([-1, 0, 3, -1, 7])
    assert list(mask) == [False, True, True, False, True]


def test_membership_counts_perfect_agreement():
    ids = [-1, 1, 1, -1, 2]
    counts = W.membership_counts(ids, ids)
    assert counts["TP"] == 3 and counts["FP"] == 0 and counts["FN"] == 0
    assert counts["truth_n"] == 3 and counts["pred_n"] == 3
    assert counts["precision"] == 1.0 and counts["recall"] == 1.0
    assert counts["size_ratio"] == 1.0
    assert counts["n_rows"] == 5


def test_membership_counts_partial_overlap():
    #        rows:   0   1   2   3   4   5
    truth = [1, 1, 1, 1, -1, -1]          # 4 members
    pred = [1, 1, 1, -1, 1, -1]           # 4 members, 3 of them correct
    counts = W.membership_counts(truth, pred)
    assert (counts["TP"], counts["FP"], counts["FN"]) == (3, 1, 1)
    assert counts["truth_n"] == 4 and counts["pred_n"] == 4
    assert counts["precision"] == pytest.approx(0.75)
    assert counts["recall"] == pytest.approx(0.75)
    assert counts["size_ratio"] == pytest.approx(1.0)


def test_membership_counts_empty_prediction_is_zero_not_a_crash():
    counts = W.membership_counts([1, 1, -1], [-1, -1, -1])
    assert counts["pred_n"] == 0
    assert counts["precision"] == 0.0
    assert counts["recall"] == 0.0
    assert counts["size_ratio"] == 0.0


def test_membership_counts_refuses_different_clouds():
    with pytest.raises(W.Refusal):
        W.membership_counts([1, 1, -1], [1, -1])


# --------------------------------------------------------------------------
# pixel IoU
# --------------------------------------------------------------------------
def test_pixel_iou_half_overlap():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    a[0:4, 0:5] = 255          # 20 px
    b[2:6, 0:5] = 255          # 20 px, 10 px shared
    out = W.pixel_iou(a, b)
    assert out["pixel_intersection"] == 10
    assert out["pixel_union"] == 30
    assert out["iou"] == pytest.approx(10 / 30)
    assert out["area_construction"] == 20 and out["area_s2"] == 20


def test_pixel_iou_identical_masks_is_one():
    a = np.zeros((8, 8), dtype=np.uint8)
    a[1:5, 1:5] = 7            # any non-zero value counts as mask
    assert W.pixel_iou(a, a)["iou"] == 1.0


def test_pixel_iou_two_empty_masks_is_zero_not_nan():
    z = np.zeros((4, 4), dtype=np.uint8)
    out = W.pixel_iou(z, z)
    assert out["pixel_union"] == 0 and out["iou"] == 0.0


def test_pixel_iou_refuses_shape_mismatch():
    with pytest.raises(W.Refusal):
        W.pixel_iou(np.zeros((4, 4)), np.zeros((4, 5)))


# --------------------------------------------------------------------------
# membership verdict
# --------------------------------------------------------------------------
def _iou_block(*values):
    return {"cam15_f%d" % (50 + i * 45): {"iou": v, "pixel_intersection": 1,
                                          "pixel_union": 1}
            for i, v in enumerate(values)}


def test_evaluate_membership_passes_when_every_threshold_is_met():
    counts = {"precision": 0.90, "recall": 0.80, "size_ratio": 1.1}
    out = W.evaluate_membership(counts, _iou_block(0.80, 0.75),
                                MEMBERSHIP_THRESHOLDS)
    assert out["pass"] is True and out["reasons"] == []


def test_evaluate_membership_fails_on_one_low_frame_iou_only():
    counts = {"precision": 0.95, "recall": 0.95, "size_ratio": 1.0}
    out = W.evaluate_membership(counts, _iou_block(0.85, 0.60),
                                MEMBERSHIP_THRESHOLDS)
    assert out["pass"] is False
    assert len(out["reasons"]) == 1 and "cam15_f95" in out["reasons"][0]


def test_evaluate_membership_collects_every_reason():
    counts = {"precision": 0.50, "recall": 0.40, "size_ratio": 3.0}
    out = W.evaluate_membership(counts, _iou_block(0.10, 0.20),
                                MEMBERSHIP_THRESHOLDS)
    assert out["pass"] is False and len(out["reasons"]) == 5


def test_evaluate_membership_size_ratio_bounds_are_inclusive():
    for ratio in (0.5, 2.0):
        counts = {"precision": 0.9, "recall": 0.9, "size_ratio": ratio}
        assert W.evaluate_membership(counts, _iou_block(0.9, 0.9),
                                     MEMBERSHIP_THRESHOLDS)["pass"] is True


# --------------------------------------------------------------------------
# T1 intervals
# --------------------------------------------------------------------------
def test_program_intervals_uses_onset_minus_one_as_the_last_absent_frame():
    prog = {"groups": [{"offset_frame": 60, "onset_frame": 90}]}
    assert W.program_intervals(prog) == [(60, 89)]


def test_program_intervals_skips_abstained_groups():
    prog = {"groups": [
        {"offset_frame": None, "onset_frame": None, "abstain_reason": "contrast"},
        {"offset_frame": 60, "onset_frame": 90},
    ]}
    assert W.program_intervals(prog) == [(60, 89)]


def test_temporal_iou_identical_and_disjoint():
    assert W.temporal_iou((60, 89), (60, 89)) == 1.0
    assert W.temporal_iou((0, 10), (20, 30)) == 0.0


def test_temporal_iou_off_by_two_frames():
    # [60,91] against [60,89]: intersection 30, union 32
    assert W.temporal_iou((60, 91), (60, 89)) == pytest.approx(30 / 32)


def test_temporal_iou_refuses_a_backwards_interval():
    with pytest.raises(W.Refusal):
        W.temporal_iou((10, 5), (0, 10))


# --------------------------------------------------------------------------
# T1 verdict
# --------------------------------------------------------------------------
def test_evaluate_t1_exact_single_interval_passes_both_readings():
    prog = {"groups": [{"offset_frame": 60, "onset_frame": 90}]}
    out = W.evaluate_t1(prog, [60, 89], T1_THRESHOLDS)
    assert out["pass"] is True and out["pass_union_reading"] is True
    assert out["n_distinct_intervals"] == 1 and out["n_gated_groups"] == 1
    assert out["union"]["offset_error"] == 0 and out["union"]["onset_error"] == 0
    assert out["union"]["temporal_iou"] == 1.0
    assert out["truth_onset_frame"] == 90


def test_evaluate_t1_two_cells_of_one_object_split_the_two_readings():
    # the wave-1 cut_roasted_beef shape: offsets 60/60, onsets 90/92
    prog = {"groups": [{"offset_frame": 60, "onset_frame": 90},
                       {"offset_frame": 60, "onset_frame": 92}]}
    out = W.evaluate_t1(prog, [60, 89], T1_THRESHOLDS)
    assert out["n_gated_groups"] == 2 and out["n_distinct_intervals"] == 2
    assert out["pass"] is False                  # the frozen `intervals: 1`
    assert out["pass_union_reading"] is True     # union [60,91] is within 3 frames
    assert out["union"]["interval"] == [60, 91]
    assert out["union"]["onset_error"] == 2
    assert out["union"]["temporal_iou"] == pytest.approx(30 / 32)


def test_evaluate_t1_duplicate_identical_groups_are_one_distinct_interval():
    prog = {"groups": [{"offset_frame": 60, "onset_frame": 90},
                       {"offset_frame": 60, "onset_frame": 90}]}
    out = W.evaluate_t1(prog, [60, 89], T1_THRESHOLDS)
    assert out["n_gated_groups"] == 2 and out["n_distinct_intervals"] == 1
    assert out["pass"] is True


def test_evaluate_t1_large_timing_error_fails_both_readings():
    prog = {"groups": [{"offset_frame": 70, "onset_frame": 100}]}
    out = W.evaluate_t1(prog, [60, 89], T1_THRESHOLDS)
    assert out["pass"] is False and out["pass_union_reading"] is False
    assert any("offset error" in r for r in out["reasons"])


def test_evaluate_t1_no_gated_group_is_a_fail_not_a_crash():
    prog = {"groups": [{"offset_frame": None, "onset_frame": None}]}
    out = W.evaluate_t1(prog, [60, 89], T1_THRESHOLDS)
    assert out["pass"] is False and out["union"] is None
    assert any("no interval" in r for r in out["reasons"])
