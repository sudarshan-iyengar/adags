"""Tests for scripts/membership_supervisability.py (schema v2).

The module under test imports torch only inside functions, so every test here
runs on a workstation without torch, cv2 or a GPU. The one test that genuinely
needs torch is skipped rather than faked.

WHAT CHANGED, AND WHY THESE TESTS DID
-------------------------------------
Version 1 of the instrument read ``_features_dc.grad`` after backpropagating
``image.sum()``. That probe was refuted: under soft routing the dynamic branch
and the static twin read the SAME ``pc.get_features``, the rasterizer returns
both ``grad_sh`` and ``grad_sh_static``, and autograd SUMS them -- so a row
culled by the temporal marginal (zero dynamic weight, nonzero static weight)
was reported as supervisable. The probe could not fire for the condition it
existed to detect. Version 2 probes the CARRIER instead: ``flow_2d``, which is
dynamic-only, carries no background term and runs through no SH evaluation.

Several tests are written as NEUTER tests: they fail if the load-bearing line
they guard is bypassed. Those are marked in their docstrings, because a test
that still passes after its mechanism is removed is worse than no test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "membership_supervisability.py"


def _load_module():
    """Import by path so the test does not depend on `scripts` being a package."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "membership_supervisability_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ms = _load_module()

LRV3_DIR = REPO_ROOT / "data" / "synthetic" / "lrv3"

LRV3_SPEC = {
    "presence_frames": {"episode_1": [0, 29], "gap": [30, 56], "episode_2": [57, 59]},
    "return_frames": [57, 58, 59],
    "event_object": {"id": 100, "centre": [0.7, 0.1, 0.35], "radius": 0.2},
    "train_cameras": [0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19],
    "test_cameras": [2, 7, 12, 17],
}

#: A 12-row, 3-class vote fixture. Rows 0-3 are truly the event object; the
#: last row received no compositing weight at all.
CLASS_IDS = [-1, 0, 100]
SCORES = ([[0.1, 0.2, 5.0]] * 4        # event rows: class 100 dominates
          + [[7.0, 0.3, 0.1]] * 4      # background rows
          + [[0.2, 9.0, 0.1]] * 3      # ground rows
          + [[0.0, 0.0, 0.0]])         # a row with no evidence at all
# P10's partition identity: sum_k w_in_mask_k == w_total.
W_TOTAL = [float(sum(row)) for row in SCORES]
TRUTH = [True] * 4 + [False] * 8


# ---------------------------------------------------------------------------
# the sphere test -- the ONE place the oracle enters, and only after every
# weight already exists
# ---------------------------------------------------------------------------


def test_sphere_boundary_is_inclusive():
    """`<=`, not `<`. Exactly representable coordinates so float subtraction
    is not what is being measured."""
    flags = ms.in_sphere_flags(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, -0.5, 0.0]],
        [0.0, 0.0, 0.0], 0.5)
    assert flags.tolist() == [True, True, True]


def test_sphere_excludes_points_outside():
    flags = ms.in_sphere_flags(
        [[0.5, 0.5, 0.0], [10.0, 0.0, 0.0]], [0.0, 0.0, 0.0], 0.5)
    assert flags.tolist() == [False, False]


def test_sphere_uses_the_offset_centre():
    centre = LRV3_SPEC["event_object"]["centre"]
    radius = LRV3_SPEC["event_object"]["radius"]
    flags = ms.in_sphere_flags(
        [[0.7, 0.1, 0.35], [0.7, 0.1, 0.475], [0.7, 0.1, 0.6], [0.0, 0.0, 0.0]],
        centre, radius)
    assert flags.tolist() == [True, True, False, False]


def test_sphere_refuses_a_negative_radius():
    with pytest.raises(ms.ContractError):
        ms.in_sphere_flags([[0.0, 0.0, 0.0]], [0.0, 0.0, 0.0], -1.0)


def test_sphere_radius_zero_admits_only_the_exact_centre():
    flags = ms.in_sphere_flags(
        [[0.7, 0.1, 0.35], [0.7, 0.1, 0.36]],
        LRV3_SPEC["event_object"]["centre"], 0.0)
    assert flags.tolist() == [True, False]


def test_sphere_refuses_points_that_are_not_three_dimensional():
    with pytest.raises(ms.ContractError):
        ms.in_sphere_flags([[0.0, 0.0]], [0.0, 0.0, 0.0], 1.0)


# ---------------------------------------------------------------------------
# answer 1a -- the ceiling as a CURVE over the FROZEN absolute grid
# ---------------------------------------------------------------------------


WEIGHTS = [0.0, 0.0, 1e-9, 1e-4, 1e-2, 0.5, 2.0]


def test_the_absolute_e_min_grid_is_the_frozen_six():
    """The grid is a commitment, not a tuning knob. If it drifts, every ceiling
    number stops being comparable to the ones already on the record."""
    assert ms.CEILING_E_MIN_GRID == (0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0)


def test_ceiling_strict_positive_is_the_exact_predicate():
    curve = ms.ceiling_curve(WEIGHTS)
    strict = curve["strict_positive"]
    assert strict["predicate"] == "w_total > 0"
    assert strict["n_at_or_above"] == 5
    assert strict["achievable_recall_ceiling"] == pytest.approx(5.0 / 7.0)


def test_ceiling_e_min_zero_entry_reads_strictly_not_as_a_no_op():
    """``w >= 0`` is true of every row including the dead ones, so the 0.0 grid
    entry must read the STRICT predicate or the curve would open at 1.0 and
    say nothing."""
    curve = ms.ceiling_curve(WEIGHTS)
    zero_entry = [e for e in curve["curve"] if e["e_min"] == 0.0][0]
    assert zero_entry["predicate"] == "w_total > 0"
    assert zero_entry["n_at_or_above"] == 5


def test_ceiling_curve_arithmetic_at_a_declared_cut():
    entry = [e for e in ms.ceiling_curve(WEIGHTS)["curve"] if e["e_min"] == 1e-2][0]
    assert entry["n_at_or_above"] == 3
    assert entry["achievable_recall_ceiling"] == pytest.approx(3.0 / 7.0)


def test_ceiling_counts_are_monotone_and_partition_the_target():
    curve = ms.ceiling_curve(WEIGHTS)["curve"]
    counts = [e["n_at_or_above"] for e in curve]
    assert all(b <= a for a, b in zip(counts, counts[1:]))
    assert all(e["n_at_or_above"] + e["n_below"] == 7 for e in curve)


def test_ceiling_reports_a_curve_not_a_scalar():
    assert len(ms.ceiling_curve(WEIGHTS)["curve"]) == len(ms.CEILING_E_MIN_GRID) > 1


def test_ceiling_on_an_empty_target_returns_none_not_a_fabricated_ratio():
    """A ratio without its n is not a measurement (LRV4's near-miss)."""
    curve = ms.ceiling_curve([])
    assert curve["n_target_rows"] == 0
    assert curve["strict_positive"]["achievable_recall_ceiling"] is None
    assert all(e["achievable_recall_ceiling"] is None for e in curve["curve"])


def test_ceiling_on_an_all_zero_target_reads_zero_not_none():
    """A starved target is a MEASUREMENT of zero, not an absent measurement.
    The distinction is the whole difference between 'no row can be supervised'
    and 'there were no rows'."""
    curve = ms.ceiling_curve([0.0] * 84)
    assert curve["n_target_rows"] == 84
    assert curve["strict_positive"]["achievable_recall_ceiling"] == 0.0


def test_ceiling_refuses_a_negative_weight():
    with pytest.raises(ms.ContractError):
        ms.ceiling_curve([-1e-30])


# ---------------------------------------------------------------------------
# answer 1b -- the quantile limb, whose cuts come from ALL rows
# ---------------------------------------------------------------------------


#: 100 cloud rows, 60 of them dead. Its q50 is 0.0.
ALL_ROWS = [0.0] * 60 + [float(v) for v in range(1, 41)]
#: A better-supervised subset. Its OWN q50 is 33.0, which is the contrast that
#: makes the leakage test below discriminating.
TARGET_SUBSET = [30.0, 36.0, 40.0, 0.0]


def test_the_quantile_grid_is_the_frozen_five():
    assert ms.CEILING_QUANTILE_GRID == (0.50, 0.75, 0.90, 0.95, 0.99)


def test_quantile_e_min_points_returns_one_cut_per_declared_q():
    points = ms.quantile_e_min_points(ALL_ROWS)
    assert [p["q"] for p in points] == list(ms.CEILING_QUANTILE_GRID)
    assert all(b["e_min"] >= a["e_min"] for a, b in zip(points, points[1:]))


def test_quantile_cuts_come_from_all_rows_and_not_from_the_target():
    """NEUTER. This is the anti-leakage test for answer 1.

    The cloud is 60% dead so its q50 is 0.0 and every target row clears it. The
    TARGET's own q50 is 33.0 and only two rows clear that. Reading 4 is
    positive evidence that the cut came from the cloud; reading 2 would mean
    ``quantile_e_min_points`` had been handed the in-sphere rows and the
    ceiling had become partly a restatement of the oracle.
    """
    points = ms.quantile_e_min_points(ALL_ROWS)
    assert points[0]["e_min"] == 0.0

    from_cloud = ms.ceiling_curve(TARGET_SUBSET, ms.CEILING_E_MIN_GRID, points)
    assert from_cloud["quantile_curve"][0]["n_at_or_above"] == 4

    leaked = ms.quantile_e_min_points(TARGET_SUBSET)
    assert leaked[0]["e_min"] == pytest.approx(33.0)
    from_target = ms.ceiling_curve(TARGET_SUBSET, ms.CEILING_E_MIN_GRID, leaked)
    assert from_target["quantile_curve"][0]["n_at_or_above"] == 2


def test_quantile_limb_entries_carry_their_q_and_their_provenance():
    curve = ms.ceiling_curve(TARGET_SUBSET, ms.CEILING_E_MIN_GRID,
                             ms.quantile_e_min_points(ALL_ROWS))
    assert len(curve["quantile_curve"]) == len(ms.CEILING_QUANTILE_GRID)
    for entry in curve["quantile_curve"]:
        assert "q" in entry
        assert "ALL rows" in entry["e_min_source"]
        assert entry["n_at_or_above"] + entry["n_below"] == len(TARGET_SUBSET)
    assert "never over" in curve["quantile_curve_provenance"]


def test_the_quantile_limb_is_reported_alongside_the_absolute_grid_not_instead():
    curve = ms.ceiling_curve(TARGET_SUBSET, ms.CEILING_E_MIN_GRID,
                             ms.quantile_e_min_points(ALL_ROWS))
    assert len(curve["curve"]) == len(ms.CEILING_E_MIN_GRID)
    assert len(curve["quantile_curve"]) == len(ms.CEILING_QUANTILE_GRID)


def test_omitting_the_quantile_points_leaves_the_limb_empty_never_faked():
    assert ms.ceiling_curve(TARGET_SUBSET)["quantile_curve"] == []


def test_quantile_e_min_points_on_an_empty_cloud_fabricates_no_cut():
    assert ms.quantile_e_min_points([]) == []


def test_quantile_e_min_points_refuses_a_negative_weight():
    with pytest.raises(ms.ContractError):
        ms.quantile_e_min_points([1.0, -1.0])


def test_quantile_e_min_points_refuses_a_q_outside_the_unit_interval():
    with pytest.raises(ms.ContractError):
        ms.quantile_e_min_points(ALL_ROWS, (1.5,))


# ---------------------------------------------------------------------------
# answer 2 -- the zero-parameter vote at its FROZEN operating point
# ---------------------------------------------------------------------------


def test_the_operating_point_is_frozen_at_one_half():
    assert ms.VOTE_TAU == 0.50


def test_the_tau_grid_is_the_frozen_six():
    assert [round(t, 4) for t in ms.VOTE_TAU_GRID] == [
        0.0, 0.25, 0.5, 0.6667, 0.75, 0.9]


def test_the_vote_assigns_the_argmax_class():
    assigned, _ = ms.argmax_vote(CLASS_IDS, SCORES, W_TOTAL)
    assert assigned[:4].tolist() == [100] * 4
    assert assigned[4:8].tolist() == [-1] * 4
    assert assigned[8:11].tolist() == [0] * 3


def test_the_vote_is_perfect_on_a_separable_fixture():
    assigned, _ = ms.argmax_vote(CLASS_IDS, SCORES, W_TOTAL)
    metrics = ms.precision_recall(assigned == 100, TRUTH)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert ms.clears_reference(metrics) is True


def test_a_row_with_no_weight_is_INELIGIBLE_and_abstains():
    """It abstains on eligibility (``w_total > 0``), not on tau -- so it cannot
    be handed a class by tie-break even at tau = 0."""
    assigned, stats = ms.argmax_vote(CLASS_IDS, SCORES, W_TOTAL)
    assert int(assigned[-1]) == ms.ABSTAIN_CLASS
    assert stats["n_abstained_ineligible"] == 1
    assert stats["n_abstained_below_tau"] == 0
    at_zero, zero_stats = ms.argmax_vote(CLASS_IDS, SCORES, W_TOTAL, tau=0.0)
    assert int(at_zero[-1]) == ms.ABSTAIN_CLASS
    assert zero_stats["n_assigned"] == 11


def test_the_vote_reports_its_tau_and_its_eligibility_counts():
    _, stats = ms.argmax_vote(CLASS_IDS, SCORES, W_TOTAL)
    assert stats["tau"] == ms.VOTE_TAU
    assert stats["n_rows"] == 12
    assert stats["n_eligible"] == 11
    assert stats["n_assigned"] == 11
    assert stats["n_abstained"] == 1
    assert stats["n_assigned_with_zero_best"] == 0


#: Three rows, each with ``w_total = 1.0``, whose best-class SHARE straddles the
#: frozen 0.50: 0.80, 0.45 and 0.34.
STRADDLE_SCORES = [[0.80, 0.15, 0.05],
                   [0.45, 0.35, 0.20],
                   [0.34, 0.33, 0.33]]
STRADDLE_TOTAL = [1.0, 1.0, 1.0]


def test_the_tau_rule_actually_bites():
    """NEUTER. If the ``best >= tau * w_total`` line were dropped and the vote
    fell back to bare argmax, all three rows would be assigned at every tau and
    the sweep would be flat at 3. This asserts the shape, not merely that a
    number came back."""
    assigned_by_tau = [
        ms.argmax_vote([-1, 0, 100], STRADDLE_SCORES, STRADDLE_TOTAL,
                       tau=t)[1]["n_assigned"]
        for t in ms.VOTE_TAU_GRID]
    assert assigned_by_tau == [3, 3, 1, 1, 1, 0]


def test_at_the_frozen_point_only_the_dominant_row_is_assigned():
    assigned, stats = ms.argmax_vote([-1, 0, 100], STRADDLE_SCORES, STRADDLE_TOTAL)
    assert assigned.tolist() == [-1, ms.ABSTAIN_CLASS, ms.ABSTAIN_CLASS]
    assert stats["n_assigned"] == 1
    assert stats["n_abstained_below_tau"] == 2
    assert stats["n_abstained_ineligible"] == 0


def test_assignments_are_monotone_in_tau():
    counts = [ms.argmax_vote([-1, 0, 100], STRADDLE_SCORES, STRADDLE_TOTAL,
                             tau=t)[1]["n_assigned"]
              for t in ms.VOTE_TAU_GRID]
    assert all(b <= a for a, b in zip(counts, counts[1:]))


def test_the_tau_rule_is_scale_free_in_the_rows_own_supply():
    """It is a SHARE, so a faintly supervised row is judged on the same terms
    as a strongly supervised one. The magnitude question is answered separately
    by the ceiling curve, and conflating the two would make the vote a second,
    undeclared magnitude threshold."""
    baseline, _ = ms.argmax_vote([-1, 0, 100], STRADDLE_SCORES, STRADDLE_TOTAL)
    shrunk, _ = ms.argmax_vote(
        [-1, 0, 100],
        [[v * 1e-9 for v in row] for row in STRADDLE_SCORES],
        [1e-9, 1e-9, 1e-9])
    assert shrunk.tolist() == baseline.tolist()


def test_the_vote_refuses_duplicate_class_ids():
    with pytest.raises(ms.ContractError):
        ms.argmax_vote([-1, -1], [[1.0, 2.0]], [3.0])


def test_the_vote_refuses_a_class_colliding_with_the_abstain_sentinel():
    with pytest.raises(ms.ContractError):
        ms.argmax_vote([ms.ABSTAIN_CLASS, 0], [[1.0, 2.0]], [3.0])


def test_the_vote_refuses_a_negative_class_score():
    with pytest.raises(ms.ContractError):
        ms.argmax_vote(CLASS_IDS, [[1.0, -1.0, 0.0]], [0.0])


def test_the_vote_refuses_a_negative_w_total():
    with pytest.raises(ms.ContractError):
        ms.argmax_vote(CLASS_IDS, [[1.0, 2.0, 3.0]], [-6.0])


def test_the_vote_refuses_a_w_total_of_the_wrong_length():
    with pytest.raises(ms.ContractError):
        ms.argmax_vote(CLASS_IDS, [[1.0, 2.0, 3.0]], [6.0, 6.0])


def test_the_vote_refuses_a_scores_matrix_of_the_wrong_width():
    with pytest.raises(ms.ContractError):
        ms.argmax_vote(CLASS_IDS, [[1.0, 2.0]], [3.0])


@pytest.mark.parametrize("tau", [-0.1, 1.1])
def test_the_vote_refuses_a_tau_outside_the_unit_interval(tau):
    with pytest.raises(ms.ContractError):
        ms.argmax_vote(CLASS_IDS, [[1.0, 2.0, 3.0]], [6.0], tau=tau)


def test_the_vote_needs_at_least_one_class():
    with pytest.raises(ms.ContractError):
        ms.argmax_vote([], np.zeros((3, 0)), [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# the tau sweep is CEILING INFORMATION, never the score
# ---------------------------------------------------------------------------


def test_the_tau_curve_is_labelled_ceiling_information():
    curve = ms.vote_tau_curve([-1, 0, 100], STRADDLE_SCORES, STRADDLE_TOTAL,
                              [True, True, True], -1)
    assert "CEILING INFORMATION ONLY" in curve["reading"]
    assert curve["frozen_operating_point"] == ms.VOTE_TAU


def test_the_tau_curve_flags_exactly_one_frozen_operating_point():
    """So a reader can never mistake the sweep's best row for the result."""
    curve = ms.vote_tau_curve([-1, 0, 100], STRADDLE_SCORES, STRADDLE_TOTAL,
                              [True, True, True], -1)
    flagged = [e for e in curve["curve"] if e["is_frozen_operating_point"]]
    assert len(flagged) == 1
    assert flagged[0]["tau"] == ms.VOTE_TAU


def test_the_tau_curve_has_one_entry_per_declared_tau():
    curve = ms.vote_tau_curve([-1, 0, 100], STRADDLE_SCORES, STRADDLE_TOTAL,
                              [True, True, True], -1)
    assert [e["tau"] for e in curve["curve"]] == list(ms.VOTE_TAU_GRID)


def test_the_tau_curve_recall_falls_as_tau_rises():
    curve = ms.vote_tau_curve([-1, 0, 100], STRADDLE_SCORES, STRADDLE_TOTAL,
                              [True, True, True], -1)
    recalls = [e["recall"] for e in curve["curve"]]
    assert recalls == [1.0, 1.0, pytest.approx(1 / 3), pytest.approx(1 / 3),
                       pytest.approx(1 / 3), 0.0]


# ---------------------------------------------------------------------------
# NEUTER (a): a vote computed from w_total instead of w_in_mask
# ---------------------------------------------------------------------------


def _neutered_scores():
    """The class information DISCARDED: every column carries the row's
    ``w_total``. This is what the vote degrades to if ``w_in_mask`` is replaced
    by the total the ceiling is computed from."""
    return [[value] * len(CLASS_IDS) for value in W_TOTAL]


def test_NEUTER_a_a_w_total_vote_collapses_precision_to_the_base_rate():
    """With the event class FIRST every eligible row ties and falls to the
    tie-break, so every one of them is called 'event'."""
    assigned, _ = ms.argmax_vote([100, -1, 0], _neutered_scores(), W_TOTAL)
    metrics = ms.precision_recall(assigned == 100, TRUTH)
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == pytest.approx(4.0 / 11.0)
    assert metrics["precision"] < 0.4
    assert ms.clears_reference(metrics) is False


def test_NEUTER_a_a_w_total_vote_with_the_event_class_last_collapses_recall():
    assigned, _ = ms.argmax_vote([-1, 0, 100], _neutered_scores(), W_TOTAL)
    metrics = ms.precision_recall(assigned == 100, TRUTH)
    assert metrics["recall"] == 0.0
    assert metrics["n_predicted_positive"] == 0
    assert metrics["precision"] is None
    assert ms.clears_reference(metrics) is not True


def test_NEUTER_a_the_tau_rule_cannot_rescue_a_w_total_vote():
    """Its best score IS w_total, so ``best >= tau * w_total`` holds for every
    tau <= 1 and the whole decision is the tie-break. The frozen operating
    point is a real guard against a WRONG class, not against a discarded one --
    which is why the collapse has to be detected by precision, above."""
    _, stats = ms.argmax_vote([100, -1, 0], _neutered_scores(), W_TOTAL)
    assert stats["n_assigned"] == 11
    assert stats["n_abstained_below_tau"] == 0
    assert stats["n_tied"] == 11


def test_NEUTER_a_the_correct_w_in_mask_vote_is_unaffected():
    """The control limb: the same 12 rows scored correctly are still perfect,
    so the collapse above is attributable to the neutering and not to the
    fixture."""
    assigned, _ = ms.argmax_vote(CLASS_IDS, SCORES, W_TOTAL)
    metrics = ms.precision_recall(assigned == 100, TRUTH)
    assert (metrics["precision"], metrics["recall"]) == (1.0, 1.0)


# ---------------------------------------------------------------------------
# NEUTER (b): flow_2d bound to a NON-LEAF, so no gradient ever arrives
# ---------------------------------------------------------------------------


class _Stand:
    """Duck-typed tensor stand-in, so the binding check is exercised here."""

    def __init__(self, shape, requires_grad, is_leaf):
        self.shape = shape
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf


def test_a_real_leaf_of_the_right_shape_is_admitted():
    assert ms.check_flow_binding(_Stand((10, 2), True, True), 10) is True


@pytest.mark.parametrize("label,stand", [
    ("wrong channel count", _Stand((10, 3), True, True)),
    ("wrong row count", _Stand((9, 2), True, True)),
    ("requires_grad False", _Stand((10, 2), False, True)),
    ("NON-LEAF", _Stand((10, 2), True, False)),
])
def test_NEUTER_b_the_binding_check_refuses_a_flow_that_cannot_receive_gradient(
        label, stand):
    with pytest.raises(ms.ContractError):
        ms.check_flow_binding(stand, 10)


def test_NEUTER_b_a_non_leaf_binding_starves_the_pass_and_P1_fires():
    """A non-leaf ``flow_2d`` receives no gradient, so every ``w_total`` reads
    zero. The instrument must REFUSE rather than report a clean ceiling of
    0.0 -- an all-zero read is indistinguishable from a real null unless the
    precondition names the cause."""
    _, failures = ms.evaluate_preconditions(
        **ms._healthy_precondition_kwargs(n_rows_nonzero_w_total=0))
    assert failures == ["P1_render_ran"]


def test_NEUTER_b_a_starved_run_abstains_on_every_row():
    assigned, stats = ms.argmax_vote(CLASS_IDS, [[0.0, 0.0, 0.0]] * 5, [0.0] * 5)
    assert set(assigned.tolist()) == {ms.ABSTAIN_CLASS}
    assert stats["n_abstained_ineligible"] == 5
    assert stats["n_assigned"] == 0


def test_NEUTER_b_a_starved_tau_curve_is_flat_at_zero_recall():
    curve = ms.vote_tau_curve(CLASS_IDS, [[0.0, 0.0, 0.0]] * 5, [0.0] * 5,
                              [True] * 5, 100)
    assert [e["recall"] for e in curve["curve"]] == [0.0] * len(ms.VOTE_TAU_GRID)
    assert [e["n_assigned"] for e in curve["curve"]] == [0] * len(ms.VOTE_TAU_GRID)


# ---------------------------------------------------------------------------
# precision / recall
# ---------------------------------------------------------------------------


def test_precision_recall_confusion_cells_add_up():
    metrics = ms.precision_recall([True, True, False, False],
                                  [True, False, True, False])
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (1, 1, 1, 1)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_precision_recall_returns_none_rather_than_zero_for_an_empty_denominator():
    metrics = ms.precision_recall([False, False], [False, False])
    assert metrics["precision"] is None and metrics["recall"] is None


def test_precision_recall_distinguishes_zero_recall_from_undefined_precision():
    metrics = ms.precision_recall([False, False], [True, False])
    assert metrics["recall"] == 0.0
    assert metrics["precision"] is None


def test_precision_recall_refuses_a_length_mismatch():
    with pytest.raises(ms.ContractError):
        ms.precision_recall([True], [True, False])


def test_clears_reference_is_a_conjunction_of_both_commissioned_numbers():
    assert ms.VOTE_REFERENCE == {"precision": 0.80, "recall": 0.90}
    assert ms.clears_reference({"precision": 0.80, "recall": 0.90}) is True
    assert ms.clears_reference({"precision": 0.99, "recall": 0.89}) is False
    assert ms.clears_reference({"precision": 0.79, "recall": 0.99}) is False
    assert ms.clears_reference({"precision": None, "recall": 1.0}) is None


# ---------------------------------------------------------------------------
# answer 3 -- the per-cell breakdown over the recorded 8^3 grid
# ---------------------------------------------------------------------------


def test_the_named_cells_are_the_two_that_produced_the_recall_cap():
    assert ms.NAMED_CELL_KEYS == (420, 429)


def test_cell_keys_decode_to_their_grid_indices():
    assert ms.decode_cell_key(420) == (6, 4, 4)
    assert ms.decode_cell_key(429) == (6, 5, 5)
    assert ms.decode_cell_key(0) == (0, 0, 0)
    assert ms.decode_cell_key(511) == (7, 7, 7)


def test_voxel_keys_encode_ix_iy_iz_and_clamp_the_max_corner():
    unit = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
                    dtype=np.float32)
    lo, span, keys = ms.voxel_keys(unit)
    assert keys.tolist() == [0, 511, 292]
    assert lo.tolist() == [0.0, 0.0, 0.0]
    assert span.tolist() == [1.0, 1.0, 1.0]


def test_voxel_keys_uses_a_supplied_absolute_grid_rather_than_the_points_bounds():
    """A recorded episode program carries the ABSOLUTE world-space grid. Using
    the cloud's own bounds instead would silently renumber every cell, so 420
    and 429 would no longer name the cells the record refers to."""
    unit = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    _, _, keys = ms.voxel_keys(unit, lo=[0.0, 0.0, 0.0], span=[2.0, 2.0, 2.0])
    assert keys.tolist() == [0, 4 * 64 + 4 * 8 + 4]


def test_voxel_keys_clamps_a_point_below_the_supplied_box_into_cell_zero():
    keys = ms.voxel_keys(np.array([[-5.0, -5.0, -5.0]], dtype=np.float32),
                         lo=[0.0, 0.0, 0.0], span=[1.0, 1.0, 1.0])[2]
    assert int(keys[0]) == 0


def test_voxel_keys_refuses_a_degenerate_supplied_span():
    unit = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ms.ContractError):
        ms.voxel_keys(unit, lo=[0.0, 0.0, 0.0], span=[0.0, 1.0, 1.0])


def test_voxel_keys_refuses_fewer_than_one_cell_per_axis():
    with pytest.raises(ms.ContractError):
        ms.voxel_keys(np.zeros((1, 3), dtype=np.float32), cells_per_axis=0)


def test_restricted_metrics_reports_the_cells_own_n_and_its_own_ceiling():
    cell = [True] * 4 + [False] * 8
    assigned, _ = ms.argmax_vote(CLASS_IDS, SCORES, W_TOTAL)
    block = ms.restricted_metrics(cell, W_TOTAL, assigned == 100, TRUTH)
    assert block["n_rows_in_restriction"] == 4
    assert block["n_target_rows_in_restriction"] == 4
    assert block["ceiling"]["n_target_rows"] == 4
    assert block["vote"]["precision"] == 1.0
    assert block["vote"]["recall"] == 1.0


def test_restricted_metrics_on_an_empty_cell_reports_none_not_a_fabricated_ratio():
    assigned, _ = ms.argmax_vote(CLASS_IDS, SCORES, W_TOTAL)
    block = ms.restricted_metrics([False] * 12, W_TOTAL, assigned == 100, TRUTH)
    assert block["n_rows_in_restriction"] == 0
    assert block["ceiling"]["strict_positive"]["achievable_recall_ceiling"] is None
    assert block["vote"]["precision"] is None


def test_every_cell_is_read_against_the_same_cloud_wide_cuts():
    """Per-cell quantiles would be cuts derived from each cell's own rows, and
    cells read against different cuts are not comparable to each other."""
    points = ms.quantile_e_min_points(ALL_ROWS)
    assigned, _ = ms.argmax_vote(CLASS_IDS, SCORES, W_TOTAL)
    for cell in ([True] * 4 + [False] * 8, [False] * 8 + [True] * 4):
        block = ms.restricted_metrics(cell, W_TOTAL, assigned == 100, TRUTH,
                                      quantile_points=points)
        assert ([e["e_min"] for e in block["ceiling"]["quantile_curve"]]
                == [p["e_min"] for p in points])


def test_restricted_metrics_refuses_ragged_inputs():
    with pytest.raises(ms.ContractError):
        ms.restricted_metrics([True], W_TOTAL, [True] * 12, TRUTH)


# ---------------------------------------------------------------------------
# answer 4 -- the static-sphere control, read from the generator
# ---------------------------------------------------------------------------


def test_the_static_control_geometry_comes_from_the_generator_not_a_literal():
    constants = ms.generator_identity_constants()
    assert constants["event_object_id"] == 100
    assert constants["source"] == "scripts/build_synthetic_reveal_scene.py"
    assert len(constants["static_spheres"]) == 3
    assert [s["id"] for s in constants["static_spheres"]] == [1, 2, 3]
    assert constants["static_sphere_id_rule"] == "positional: ids.append(idx + 1)"


def test_no_static_sphere_id_collides_with_the_event_object_id():
    """They share the identity buffers, so a collision would silently merge the
    control into the target."""
    constants = ms.generator_identity_constants()
    assert constants["event_object_id"] not in {
        s["id"] for s in constants["static_spheres"]}
    assert constants["background_id"] not in {
        s["id"] for s in constants["static_spheres"]}


def test_select_static_control_returns_a_usable_sphere():
    chosen = ms.select_static_control(ms.generator_identity_constants(), 0)
    assert chosen["id"] == 1
    assert chosen["radius"] > 0.0
    assert len(chosen["centre"]) == 3


@pytest.mark.parametrize("index", [-1, 3, 99])
def test_select_static_control_refuses_an_out_of_range_index(index):
    with pytest.raises(ms.ContractError):
        ms.select_static_control(ms.generator_identity_constants(), index)


def test_select_static_control_refuses_a_generator_with_no_static_spheres():
    with pytest.raises(ms.ContractError):
        ms.select_static_control({"static_spheres": []}, 0)


# ---------------------------------------------------------------------------
# answer 5 -- the static-twin share is a DISTRIBUTION, never a threshold
# ---------------------------------------------------------------------------


def test_the_share_quantiles_are_q50_q90_q95_q99_and_the_max():
    assert ms.SHARE_QUANTILES == (0.50, 0.90, 0.95, 0.99, 1.0)


def test_quantiles_of_interpolates_linearly_and_keys_by_percent():
    out = ms.quantiles_of([0.0, 0.25, 0.5, 0.75, 1.0], ms.SHARE_QUANTILES)
    assert out["p050"] == 0.5
    assert out["p100"] == 1.0
    assert set(out) == {"p050", "p090", "p095", "p099", "p100"}


def test_quantiles_of_an_empty_input_are_all_none():
    out = ms.quantiles_of([], ms.SHARE_QUANTILES)
    assert all(v is None for v in out.values())


def test_quantiles_of_refuses_a_q_outside_the_unit_interval():
    with pytest.raises(ms.ContractError):
        ms.quantiles_of([0.0, 1.0], (1.5,))


def test_weight_distribution_counts_the_exact_zeros_separately():
    """A row at exactly zero is qualitatively different from a small one: it
    received no gradient at all. Pooling them into a quantile would hide the
    only thing the ceiling turns on."""
    dist = ms.weight_distribution(WEIGHTS)
    assert dist["n"] == 7
    assert dist["n_exactly_zero"] == 2
    assert dist["min"] == 0.0
    assert dist["max"] == 2.0
    assert dist["quantiles"]["p050"] == 1e-4


def test_weight_distribution_reports_every_declared_quantile():
    dist = ms.weight_distribution(WEIGHTS)
    for q in ms.WEIGHT_QUANTILES:
        assert ("p%03d" % int(round(q * 100))) in dist["quantiles"]


def test_weight_distribution_on_an_empty_input_is_all_none():
    dist = ms.weight_distribution([])
    assert dist["n"] == 0
    assert dist["min"] is None and dist["max"] is None
    assert dist["quantiles"]["p050"] is None


# ---------------------------------------------------------------------------
# the frozen preconditions -- statements about the SETUP only
# ---------------------------------------------------------------------------


def test_a_healthy_setup_passes_every_precondition():
    block, failures = ms.evaluate_preconditions(**ms._healthy_precondition_kwargs())
    assert failures == []
    assert len([k for k in block if k != "detail"]) == 15


@pytest.mark.parametrize("key,override", [
    ("P1_render_ran", {"n_rows_nonzero_w_total": 0}),
    ("P2_rows_in_event_target_positive", {"n_rows_in_event_target": 0}),
    ("P3_rows_in_static_target_positive", {"n_rows_in_static_target": 0}),
    ("P4_frame_set_within_presence",
     {"frame_set_ok": False, "offending_frames": [40]}),
    ("P5_any_view_rendered_nonzero", {"n_views_nonzero_image": 0}),
    ("P6_cameras_disjoint_from_test", {"camera_ids_used": [0, 2]}),
    ("P6_cameras_disjoint_from_test", {"cameras_are_train_objects": False}),
    ("P6_cameras_disjoint_from_test", {"test_camera_ids": []}),
    ("P7_row_count_matches_checkpoint", {"n_rows_checkpoint": 999}),
    ("P8_topology_invariant", {"n_rows_after_pass": 999}),
    ("P9_flow_leaf_bound_every_view", {"n_rasterizer_calls": 527}),
    ("P9_flow_leaf_bound_every_view",
     {"n_rasterizer_calls": 0, "n_expected_rasterizer_calls": 0}),
    ("P10_mask_partition_consistent", {"mask_partition_ok": False}),
    ("P11_backward_repeatable", {"backward_repeat_bitwise_identical": False}),
    ("P12_static_branch_shares_features", {"static_branch_shares_features": False}),
    ("P13_identity_masks_complete", {"identity_masks_complete": False}),
    ("P14_camera_mask_supply",
     {"cameras_below_mask_floor": [13, 14], "min_camera_mask_px": 1000}),
    ("P15_fingerprint_as_expected", {"expect_wrong_fingerprint": True}),
])
def test_each_precondition_fires_when_violated(key, override):
    _, failures = ms.evaluate_preconditions(
        **ms._healthy_precondition_kwargs(**override))
    assert failures == [key]


def test_P6_refuses_an_empty_held_out_roster_rather_than_passing_vacuously():
    """NEUTER. A guard that can degrade silently to 'protects nothing' is worse
    than no guard: with ``test_cameras`` empty the disjointness test is
    trivially true and would report a pass forever."""
    block, failures = ms.evaluate_preconditions(
        **ms._healthy_precondition_kwargs(test_camera_ids=[]))
    assert failures == ["P6_cameras_disjoint_from_test"]
    assert block["detail"]["test_camera_ids"] == []


def test_P9_fires_when_no_render_was_intercepted_at_all():
    """Zero calls matching zero expected calls is not a pass."""
    _, failures = ms.evaluate_preconditions(**ms._healthy_precondition_kwargs(
        n_rasterizer_calls=0, n_expected_rasterizer_calls=0))
    assert failures == ["P9_flow_leaf_bound_every_view"]


def test_a_matching_expected_fingerprint_passes_and_records_that_it_was_asked():
    block, failures = ms.evaluate_preconditions(
        **ms._healthy_precondition_kwargs(expect_matching_fingerprint=True))
    assert failures == []
    assert block["detail"]["fingerprint_check_requested"] is True


def test_without_an_expected_fingerprint_the_check_is_recorded_as_not_requested():
    block, failures = ms.evaluate_preconditions(**ms._healthy_precondition_kwargs())
    assert failures == []
    assert block["detail"]["fingerprint_check_requested"] is False


def test_the_precondition_block_records_the_detail_a_reader_needs():
    block, _ = ms.evaluate_preconditions(**ms._healthy_precondition_kwargs())
    detail = block["detail"]
    for key in ("n_rows_nonzero_w_total", "n_rows_in_event_target",
                "n_rows_in_static_target", "camera_ids_used", "test_camera_ids",
                "n_rasterizer_calls_intercepted", "n_rows_after_pass",
                "cameras_below_mask_floor", "fingerprint_measured"):
        assert key in detail


def test_preconditions_are_evaluated_from_the_SETUP_only():
    """NEUTER. A precondition that reads a score can leak the outcome, and a
    rule that can leak the outcome is not a precondition. This asserts on the
    SIGNATURE, so it fails if a scored quantity is ever threaded in."""
    import inspect

    names = set(inspect.signature(ms.evaluate_preconditions).parameters)
    forbidden = {"ceiling", "recall", "precision", "w_total", "w_in_mask",
                 "weights", "vote", "share", "curve", "tau"}
    assert names & forbidden == set()


# ---------------------------------------------------------------------------
# anti-leakage
# ---------------------------------------------------------------------------


def test_a_held_out_camera_is_refused():
    with pytest.raises(ms.LeakageError):
        ms.assert_cameras_are_training(
            [0, 1, 2], LRV3_SPEC["train_cameras"], LRV3_SPEC["test_cameras"])


def test_a_camera_that_is_not_a_declared_training_camera_is_refused():
    with pytest.raises(ms.LeakageError):
        ms.assert_cameras_are_training(
            [0, 1, 42], LRV3_SPEC["train_cameras"], LRV3_SPEC["test_cameras"])


def test_the_full_training_roster_is_admitted():
    assert ms.assert_cameras_are_training(
        LRV3_SPEC["train_cameras"], LRV3_SPEC["train_cameras"],
        LRV3_SPEC["test_cameras"]) is True


@pytest.mark.parametrize("held_out", LRV3_SPEC["test_cameras"])
def test_every_held_out_camera_is_refused_individually(held_out):
    with pytest.raises(ms.LeakageError):
        ms.assert_cameras_are_training(
            [0, held_out], LRV3_SPEC["train_cameras"], LRV3_SPEC["test_cameras"])


def test_an_empty_held_out_roster_is_refused():
    with pytest.raises(ms.ContractError):
        ms.assert_cameras_are_training([0], LRV3_SPEC["train_cameras"], [])


def test_an_empty_training_roster_is_refused():
    with pytest.raises(ms.ContractError):
        ms.assert_cameras_are_training([0], [], LRV3_SPEC["test_cameras"])


def test_the_held_out_identity_directory_is_refused_by_name():
    """``gt_identity/`` means HELD-OUT ONLY in this repository. Reading
    supervision masks from it would be a leak, so it refuses by construction
    rather than by convention."""
    with pytest.raises(ms.LeakageError):
        ms.resolve_identity_dir(LRV3_DIR, LRV3_DIR / "gt_identity")


def test_a_missing_identity_directory_is_refused():
    with pytest.raises(ms.ContractError):
        ms.resolve_identity_dir(LRV3_DIR, LRV3_DIR / "no_such_dir")


def test_camera_and_frame_ids_parse_from_the_fixture_naming():
    assert ms.camera_id_of_name("cam07_f012") == 7
    assert ms.frame_index_of_name("cam19_f059") == 59
    assert ms.frame_index_of_name("cam19") is None


@pytest.mark.parametrize("name", ["", None, "camera_seven"])
def test_an_unparseable_camera_name_is_refused(name):
    with pytest.raises(ms.ContractError):
        ms.camera_id_of_name(name)


# ---------------------------------------------------------------------------
# frame handling
# ---------------------------------------------------------------------------


def test_default_frame_set_is_episode_one_plus_the_return():
    assert ms.default_frame_set(LRV3_SPEC) == list(range(0, 30)) + [57, 58, 59]


def test_default_frame_set_follows_a_different_fixture():
    """Nothing is hardcoded: LRV4's one-frame return yields LRV4's frames."""
    assert ms.default_frame_set({
        "presence_frames": {"episode_1": [0, 29], "episode_2": [59, 59]},
        "return_frames": [59]}) == list(range(0, 30)) + [59]


def test_a_presence_pair_is_read_as_inclusive_endpoints():
    assert ms.expand_inclusive_pair([0, 3], "x") == [0, 1, 2, 3]


@pytest.mark.parametrize("pair", [[3, 0], [0], [0, 1, 2], [-1, 3]])
def test_a_malformed_presence_pair_is_refused(pair):
    with pytest.raises(ms.ContractError):
        ms.expand_inclusive_pair(pair, "x")


def test_presence_windows_exclude_the_gap():
    windows = ms.presence_windows(LRV3_SPEC)
    assert (0, 29) in windows and (57, 59) in windows
    assert (30, 56) not in windows


def test_frames_within_presence_flags_gap_frames():
    windows = ms.presence_windows(LRV3_SPEC)
    ok, bad = ms.frames_within_presence([29, 30, 40, 57], windows)
    assert not ok and bad == [30, 40]


def test_the_default_frame_set_lies_entirely_within_presence():
    ok, bad = ms.frames_within_presence(
        ms.default_frame_set(LRV3_SPEC), ms.presence_windows(LRV3_SPEC))
    assert ok and bad == []


def test_parse_frame_spec_handles_ranges_and_singletons():
    assert ms.parse_frame_spec("0-2,57,58") == [0, 1, 2, 57, 58]
    assert ms.parse_frame_spec("") is None


def test_parse_frame_spec_refuses_a_backwards_range():
    with pytest.raises(ms.ContractError):
        ms.parse_frame_spec("9-2")


def test_a_spec_with_no_episode_one_is_refused():
    with pytest.raises(ms.ContractError):
        ms.default_frame_set({"presence_frames": {"episode_2": [57, 59]},
                              "return_frames": [57]})


def test_a_spec_with_no_return_frames_is_refused():
    with pytest.raises(ms.ContractError):
        ms.default_frame_set({"presence_frames": {"episode_1": [0, 29]},
                              "return_frames": []})


def test_sphere_from_spec_reads_the_fixture_geometry():
    centre, radius = ms.sphere_from_spec(LRV3_SPEC)
    assert centre == [0.7, 0.1, 0.35]
    assert radius == 0.2


def test_sphere_from_spec_refuses_a_degenerate_object():
    with pytest.raises(ms.ContractError):
        ms.sphere_from_spec({"event_object": {"centre": [0, 0, 0], "radius": 0.0}})


# ---------------------------------------------------------------------------
# the per-camera supply floor -- REPORT ONLY by default
# ---------------------------------------------------------------------------


SUPPLY = {0: 8005, 1: 8201, 13: 316, 14: 16}


def test_the_default_floor_of_zero_flags_nobody():
    """This script does not choose a scientific floor. Default 0 means every
    camera's count is reported and nothing is refused."""
    assert ms.cameras_below_supply_floor(SUPPLY, 0) == []


def test_a_declared_floor_flags_the_starved_cameras():
    assert ms.cameras_below_supply_floor(SUPPLY, 1000) == [13, 14]


def test_a_floor_above_every_camera_flags_every_camera():
    assert ms.cameras_below_supply_floor(SUPPLY, 100000) == [0, 1, 13, 14]


def test_a_negative_floor_is_refused():
    with pytest.raises(ms.ContractError):
        ms.cameras_below_supply_floor(SUPPLY, -1)


# ---------------------------------------------------------------------------
# the identity census, on the REAL fixture. numpy only, so it runs here.
# ---------------------------------------------------------------------------


def _require_fixture():
    if not (LRV3_DIR / "train_identity").is_dir():
        pytest.skip("data/synthetic/lrv3/train_identity is not present here")


def test_the_identity_buffers_carry_far_fewer_distinct_images_than_files():
    """THE EFFECTIVE SUPERVISION n. 960 files carry only 32 distinct images by
    content -- 2 per camera -- so the supervision available to a membership head
    is 16 distinct masks, not 528 observations. A count of files would overstate
    it by more than an order of magnitude."""
    _require_fixture()
    identity_dir = ms.resolve_identity_dir(LRV3_DIR, "")
    spec = json.loads((LRV3_DIR / "event_spec.json").read_text(encoding="utf-8"))
    frames = ms.default_frame_set(spec)
    wanted = [(c, f) for c in spec["train_cameras"] for f in frames]

    census, by_view, by_digest = ms.identity_census(identity_dir, wanted,
                                                    min(frames), 100)
    assert census["n_files"] == 960
    assert census["n_distinct_buffers_by_content"] == 32
    assert census["n_cameras_in_dir"] == 16
    assert set(census["distinct_buffers_per_camera"].values()) == {2}
    assert len(by_digest) == 32
    assert len(by_view) == 960
    assert "DISTINCT" in census["effective_supervision_note"]


def test_the_census_finds_every_requested_view_and_one_consistent_shape():
    _require_fixture()
    identity_dir = ms.resolve_identity_dir(LRV3_DIR, "")
    spec = json.loads((LRV3_DIR / "event_spec.json").read_text(encoding="utf-8"))
    frames = ms.default_frame_set(spec)
    wanted = [(c, f) for c in spec["train_cameras"] for f in frames]

    census, _, _ = ms.identity_census(identity_dir, wanted, min(frames), 100)
    assert len(wanted) == 528
    assert census["missing_requested_views"] == []
    assert tuple(census["shape"]) == (300, 400)
    assert census["dtype"] == "int16"
    assert census["class_ids_present"] == [-1, 0, 1, 2, 3, 100]


def test_the_per_camera_event_supply_is_reported_for_every_camera_and_is_uneven():
    """PER-CAMERA, NOT POOLED. The pooled 71,625 event pixels look ample; the
    per-camera view is a 500x spread from 16 to 8,201, with half the cameras
    below the median-ish 4,603. A pooled floor would hide that entirely."""
    _require_fixture()
    identity_dir = ms.resolve_identity_dir(LRV3_DIR, "")
    spec = json.loads((LRV3_DIR / "event_spec.json").read_text(encoding="utf-8"))
    frames = ms.default_frame_set(spec)
    wanted = [(c, f) for c in spec["train_cameras"] for f in frames]

    census, _, _ = ms.identity_census(identity_dir, wanted, min(frames), 100)
    counts = {int(k): int(v) for k, v in
              census["event_pixels_per_camera_at_reference_frame"].items()}
    assert sorted(counts) == spec["train_cameras"]
    assert len(counts) == 16
    assert min(counts.values()) == 16 and counts[14] == 16
    assert max(counts.values()) == 8201 and counts[1] == 8201
    assert sum(counts.values()) == 71625
    assert sum(1 for v in counts.values() if v < 4603) == 8
    assert ms.cameras_below_supply_floor(counts, 0) == []


def test_the_census_refuses_a_directory_with_no_identity_buffers(tmp_path):
    with pytest.raises(ms.ContractError):
        ms.identity_census(tmp_path, [], 0, 100)


# ---------------------------------------------------------------------------
# the grid definition and its provenance
# ---------------------------------------------------------------------------


def test_an_absent_episode_program_recomputes_the_grid_and_SAYS_SO():
    """The report must never present a recomputed grid as the recorded one:
    cell 420 in a recomputed grid is not cell 420 in the record."""
    lo, span, cells, provenance = ms.load_grid_definition("")
    assert lo is None and span is None
    assert cells == ms.DEFAULT_CELLS_PER_AXIS == 8
    assert provenance["grid_source"] == "recomputed_from_cloud_bounds"
    assert provenance["program_path"] is None


def test_a_recorded_episode_program_supplies_the_absolute_grid(tmp_path):
    program = tmp_path / "program.json"
    program.write_text(json.dumps({
        "spatial": {"kind": "voxel_grid", "cells_per_axis": 8,
                    "lo": [-1.0, -1.0, -1.0], "span": [2.0, 2.0, 2.0]},
        "cloud": {"xyz_sha256": "f" * 64, "n_rows": 10648},
    }), encoding="utf-8")

    lo, span, cells, provenance = ms.load_grid_definition(str(program))
    assert lo == [-1.0, -1.0, -1.0]
    assert span == [2.0, 2.0, 2.0]
    assert cells == 8
    assert provenance["grid_source"] == "episode_program"
    assert provenance["program_cloud_rows"] == 10648
    assert len(provenance["program_sha256"]) == 64


def test_a_program_without_a_voxel_grid_is_refused(tmp_path):
    program = tmp_path / "program.json"
    program.write_text(json.dumps({"spatial": {"kind": "something_else"}}),
                       encoding="utf-8")
    with pytest.raises(ms.ContractError):
        ms.load_grid_definition(str(program))


def test_a_missing_program_path_is_refused_rather_than_silently_recomputed():
    with pytest.raises(ms.ContractError):
        ms.load_grid_definition("no/such/program.json")


def test_discover_episode_program_ignores_json_without_a_voxel_grid(tmp_path):
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "b.json").write_text("not json at all", encoding="utf-8")
    assert ms.discover_episode_program(str(tmp_path)) is None
    (tmp_path / "c.json").write_text(json.dumps({
        "spatial": {"kind": "voxel_grid", "lo": [0, 0, 0], "span": [1, 1, 1]}}),
        encoding="utf-8")
    assert ms.discover_episode_program(str(tmp_path)).endswith("c.json")


# ---------------------------------------------------------------------------
# the on-disk fixture agrees with the constants used above
# ---------------------------------------------------------------------------


def test_on_disk_lrv3_spec_matches_the_values_these_tests_assume():
    path = LRV3_DIR / "event_spec.json"
    if not path.is_file():
        pytest.skip("data/synthetic/lrv3/event_spec.json is not present here")
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert spec["test_cameras"] == LRV3_SPEC["test_cameras"]
    assert spec["train_cameras"] == LRV3_SPEC["train_cameras"]
    assert len(spec["train_cameras"]) == 16
    assert spec["event_object"] == LRV3_SPEC["event_object"]
    assert ms.default_frame_set(spec) == list(range(0, 30)) + [57, 58, 59]
    ok, bad = ms.frames_within_presence(
        ms.default_frame_set(spec), ms.presence_windows(spec))
    assert ok and bad == []


# ---------------------------------------------------------------------------
# the self-test is itself a test
# ---------------------------------------------------------------------------


def test_self_test_passes_without_torch():
    assert ms.self_test() == 0


def test_the_module_is_importable_and_declares_the_v2_schema():
    """The whole point of the lazy torch imports: this file must be usable on a
    workstation with neither torch nor cv2."""
    assert ms.SCHEMA == "membership-supervisability-v2"


def test_the_module_docstring_records_why_the_v1_probe_was_replaced():
    """The refutation is load-bearing context for every number this instrument
    produces; losing it would let the v1 probe be reintroduced as a
    simplification."""
    doc = ms.__doc__
    assert "flow_2d" in doc
    assert "_features_dc" in doc
    assert "static" in doc


# ---------------------------------------------------------------------------
# torch-only
# ---------------------------------------------------------------------------


def test_torch_sphere_expression_matches_the_pure_python_twin():
    """The canonical `(xyz - centre).norm(dim=1) <= radius` and the pure twin
    must agree. Skipped where torch is unavailable rather than faked."""
    torch = pytest.importorskip("torch")

    xyz = torch.tensor([[0.7, 0.1, 0.35],
                        [0.7, 0.1, 0.475],
                        [0.7, 0.1, 0.6],
                        [0.0, 0.0, 0.0]], dtype=torch.float32)
    centre = LRV3_SPEC["event_object"]["centre"]
    radius = LRV3_SPEC["event_object"]["radius"]
    centre_t = torch.tensor(centre, dtype=xyz.dtype)
    torch_flags = ((xyz - centre_t).norm(dim=1) <= float(radius)).tolist()
    pure_flags = ms.in_sphere_flags(xyz.tolist(), centre, radius)
    assert torch_flags == pure_flags


# ---------------------------------------------------------------------------
# v4 -- the erosion / dilation kernels
#
# `research-wiki/operations/lrv3-mask-noise-and-shuffle-v4-2026-08-25.md` §3.
# Written with 4-connectivity numpy shifts and no scipy dependency, so `k`
# iterations erode by an L1 ball of radius `k`.
# ---------------------------------------------------------------------------


def _box(shape, top, left, height, width):
    mask = np.zeros(shape, dtype=bool)
    mask[top:top + height, left:left + width] = True
    return mask


def test_erosion_of_a_3x3_square_leaves_only_its_centre():
    """Hand computed: with a plus-shaped structuring element the only pixel of
    a 3x3 square whose four neighbours are all inside is the centre."""
    square = _box((9, 9), 3, 3, 3, 3)
    eroded = ms.erode_mask(square, 1)
    assert int(eroded.sum()) == 1
    assert bool(eroded[4, 4])
    assert int(ms.erode_mask(square, 2).sum()) == 0


def test_erosion_of_a_rectangle_removes_one_ring_per_iteration():
    """Hand computed: a W x H rectangle eroded k times is (W-2k) x (H-2k),
    because the L1 ball of radius k reaches exactly k pixels along each axis."""
    rect = _box((40, 40), 10, 10, 11, 9)          # 11 rows x 9 columns
    assert int(rect.sum()) == 99
    assert int(ms.erode_mask(rect, 1).sum()) == 9 * 7
    assert int(ms.erode_mask(rect, 2).sum()) == 7 * 5
    assert int(ms.erode_mask(rect, 4).sum()) == 3 * 1
    assert int(ms.erode_mask(rect, 5).sum()) == 0


def test_dilation_of_one_pixel_is_the_l1_ball():
    """Hand computed: |L1 ball of radius r| = 2r^2 + 2r + 1."""
    dot = np.zeros((25, 25), dtype=bool)
    dot[12, 12] = True
    for radius in (1, 2, 4, 8):
        assert (int(ms.dilate_mask(dot, radius).sum())
                == 2 * radius ** 2 + 2 * radius + 1)


def test_dilation_of_a_rectangle_adds_one_ring_per_iteration():
    rect = _box((40, 40), 10, 10, 5, 5)
    grown = ms.dilate_mask(rect, 1)
    assert int(grown.sum()) == 25 + 4 * 5          # the four edge strips
    assert bool(grown[9, 12]) and bool(grown[14, 12])
    assert not bool(grown[9, 9])                   # the corner is L1 distance 2


def test_the_kernels_are_the_identity_at_k_zero():
    mask = _box((12, 12), 3, 3, 4, 4)
    assert ms.erode_mask(mask, 0).tolist() == mask.tolist()
    assert ms.dilate_mask(mask, 0).tolist() == mask.tolist()


@pytest.mark.parametrize("kernel", ["erode_mask", "dilate_mask"])
def test_a_negative_k_is_refused(kernel):
    with pytest.raises(ms.ContractError):
        getattr(ms, kernel)(_box((8, 8), 2, 2, 3, 3), -1)


@pytest.mark.parametrize("kernel", ["erode_mask", "dilate_mask"])
def test_a_non_2d_mask_is_refused(kernel):
    with pytest.raises(ms.ContractError):
        getattr(ms, kernel)(np.zeros((4, 4, 4), dtype=bool), 1)


def test_off_image_is_background_so_erosion_eats_a_border_touching_mask():
    """The declared convention. Stated as a test rather than a comment because
    the alternative (pad with the border value) is equally common and would
    change every reported area on a mask that touches an edge."""
    touching = np.zeros((10, 10), dtype=bool)
    touching[0:2, 0:5] = True
    # Under the padded convention the three interior top-row pixels survive;
    # under this one every pixel loses a neighbour and the mask is annihilated.
    assert int(ms.erode_mask(touching, 1).sum()) == 0
    interior = np.zeros((10, 10), dtype=bool)
    interior[4:6, 2:7] = True
    assert int(ms.erode_mask(interior, 1).sum()) == 0


# ---------------------------------------------------------------------------
# v4 -- THE TWO ANNIHILATION POINTS the spec declares in advance (§1, §4 N2)
# ---------------------------------------------------------------------------


def test_erosion_k1_annihilates_a_16_pixel_mask():
    """`cam14` supplies 16 px and the spec declares it destroyed at k=1. A
    16-px mask at that scale is one pixel wide somewhere, and a 1-px-wide
    structure cannot survive a single 4-connectivity erosion."""
    thin = np.zeros((40, 40), dtype=bool)
    thin[10, 10:26] = True
    assert int(thin.sum()) == 16
    assert int(ms.erode_mask(thin, 1).sum()) == 0


def test_erosion_k8_annihilates_a_316_pixel_mask_that_survives_k4():
    """`cam13` supplies 316 px and the spec declares it destroyed at k=8 and
    NOT at k=1. The k=4 limb is the load-bearing half: a mask that died early
    would make the declared k=8 count right for the wrong reason."""
    slab = _box((60, 60), 10, 10, 21, 15)          # 21 x 15 = 315
    slab[31, 10] = True                            # one more, for 316
    assert int(slab.sum()) == 316
    assert int(ms.erode_mask(slab, 1).sum()) > 0
    assert int(ms.erode_mask(slab, 4).sum()) > 0
    assert int(ms.erode_mask(slab, 7).sum()) > 0
    assert int(ms.erode_mask(slab, 8).sum()) == 0


def test_the_spec_declares_one_annihilation_at_k1_and_two_at_k8():
    assert ms.EROSION_ANNIHILATION_EXPECTED == {1: 1, 8: 2}


def test_the_real_fixture_reproduces_the_declared_annihilation_counts():
    """N2 on the REAL masks. The counts are what the spec's §1 input table
    recorded; a different erosion convention would move them."""
    _require_fixture()
    identity_dir = ms.resolve_identity_dir(LRV3_DIR, "")
    emptied = {1: [], 2: [], 4: [], 8: []}
    for camera in LRV3_SPEC["train_cameras"]:
        buffer = np.load(str(identity_dir / ("cam%02d_f000.npy" % camera)))
        obj = buffer == 100
        assert obj.any()
        for k in (1, 2, 4, 8):
            if not ms.erode_mask(obj, k).any():
                emptied[k].append(camera)
    assert emptied[1] == [14]
    assert emptied[8] == [13, 14]
    assert ms.annihilation_block(1, emptied[1])["passed"] is True
    assert ms.annihilation_block(8, emptied[8])["passed"] is True


def test_the_real_fixture_reproduces_the_measured_per_camera_areas():
    """The three cameras the spec's §1 table is anchored on. These numbers are
    what the sweep's N1 block reports, so pinning them here means a kernel
    change cannot silently move the reported sensitivity curve."""
    _require_fixture()
    identity_dir = ms.resolve_identity_dir(LRV3_DIR, "")
    expected = {
        0: (8005, 7721, 7441, 6893, 5851),
        13: (316, 226, 147, 35, 0),
        14: (16, 0, 0, 0, 0),
    }
    for camera, wanted in expected.items():
        obj = np.load(str(identity_dir / ("cam%02d_f000.npy" % camera))) == 100
        measured = (int(obj.sum()),) + tuple(
            int(ms.erode_mask(obj, k).sum()) for k in (1, 2, 4, 8))
        assert measured == wanted, "cam%02d measured %r" % (camera, measured)


# ---------------------------------------------------------------------------
# v4 -- the nearest-non-object relabelling, which keeps every perturbed buffer
# a PARTITION (which is what P10 reads)
# ---------------------------------------------------------------------------


def test_an_enclosed_object_pixel_takes_its_neighbours_class():
    surrounded = np.array([[7, 7, 7], [7, 9, 7], [7, 7, 7]], dtype=np.int16)
    assert ms.nearest_non_object_labels(surrounded, 9).tolist() == [[7] * 3] * 3


def test_a_tie_resolves_in_the_declared_neighbour_order():
    """Three of the four neighbours say 1 and one says 2; the declared order
    puts 'up' first, so the answer is 1 and it is not numpy's to choose."""
    split = np.array([[1, 1, 2], [1, 9, 2], [1, 1, 2]], dtype=np.int16)
    assert int(ms.nearest_non_object_labels(split, 9)[1, 1]) == 1
    assert ms.NEIGHBOUR_ORDER[0] == (-1, 0)


def test_a_deep_object_interior_is_still_reached_by_the_wavefront():
    scene = np.zeros((21, 21), dtype=np.int16)
    scene[4:17, 4:17] = 100
    filled = ms.nearest_non_object_labels(scene, 100)
    assert int((filled == 100).sum()) == 0
    assert set(np.unique(filled).tolist()) == {0}


def test_a_buffer_with_no_object_is_returned_unchanged():
    scene = np.array([[1, 2], [3, 4]], dtype=np.int16)
    assert ms.nearest_non_object_labels(scene, 100).tolist() == scene.tolist()


def test_an_all_object_buffer_is_refused_rather_than_invented():
    with pytest.raises(ms.ContractError):
        ms.nearest_non_object_labels(np.full((4, 4), 100, dtype=np.int16), 100)


# ---------------------------------------------------------------------------
# v4 -- the four families
# ---------------------------------------------------------------------------


def _scene():
    scene = np.zeros((21, 21), dtype=np.int16)
    scene[:, :10] = 1
    scene[8:14, 8:14] = 100
    return scene


def test_erosion_shrinks_the_object_and_leaves_a_partition():
    scene = _scene()
    fallback = ms.nearest_non_object_labels(scene, 100)
    eroded = ms.apply_erosion(scene, 1, 100, fallback)
    assert int((eroded == 100).sum()) == int(ms.erode_mask(scene == 100, 1).sum())
    assert int((eroded == 100).sum()) < int((scene == 100).sum())
    assert set(np.unique(eroded).tolist()) <= set(np.unique(scene).tolist())
    assert eroded.size == scene.size


def test_dilation_grows_the_object_and_leaves_a_partition():
    scene = _scene()
    dilated = ms.apply_dilation(scene, 2, 100)
    assert int((dilated == 100).sum()) == int(ms.dilate_mask(scene == 100, 2).sum())
    assert int((dilated == 100).sum()) > int((scene == 100).sum())
    assert dilated.size == scene.size


def test_a_missing_camera_loses_the_object_entirely():
    scene = _scene()
    fallback = ms.nearest_non_object_labels(scene, 100)
    gone = ms.apply_missing_camera(scene, 100, fallback)
    assert int((gone == 100).sum()) == 0
    assert set(np.unique(gone).tolist()) <= {0, 1}
    assert gone.size == scene.size


def test_identity_switch_preserves_the_total_pixel_count():
    """It RELABELS; it does not delete. The buffer keeps its size, every pixel
    keeps a class that existed in the clean buffer, and whatever the object
    loses the other classes gain -- which is what keeps P10's partition
    identity true at every swept point."""
    scene = _scene()
    fallback = ms.nearest_non_object_labels(scene, 100)
    uniform = ms.switch_uniform(scene, 100, ms.NOISE_SEED_DECLARED, "a" * 64)
    clean_object = int((scene == 100).sum())
    clean_other = int((scene != 100).sum())
    for fraction in ms.NOISE_MAGNITUDES["identity-switch"]:
        switched = ms.apply_identity_switch(scene, fraction, 100, fallback, uniform)
        assert switched.size == scene.size
        assert set(np.unique(switched).tolist()) <= set(np.unique(scene).tolist())
        lost = clean_object - int((switched == 100).sum())
        gained = int((switched != 100).sum()) - clean_other
        assert lost == gained
        assert (int((switched == 100).sum()) + int((switched != 100).sum())
                == scene.size)


def test_identity_switch_draws_are_nested_across_the_magnitudes():
    """5% is a subset of 10% is a subset of 25% is a subset of 50%, so the
    curve reads as one worsening mask rather than four unrelated draws."""
    scene = _scene()
    fallback = ms.nearest_non_object_labels(scene, 100)
    uniform = ms.switch_uniform(scene, 100, ms.NOISE_SEED_DECLARED, "a" * 64)
    kept = [set(map(tuple, np.argwhere(
        ms.apply_identity_switch(scene, f, 100, fallback, uniform) == 100)))
        for f in ms.NOISE_MAGNITUDES["identity-switch"]]
    for smaller, larger in zip(kept, kept[1:]):
        assert larger <= smaller


def test_identity_switch_never_selects_a_non_object_pixel():
    scene = _scene()
    uniform = ms.switch_uniform(scene, 100, ms.NOISE_SEED_DECLARED, "b" * 64)
    assert bool((uniform[scene != 100] == 1.0).all())


@pytest.mark.parametrize("fraction", [-0.01, 1.01])
def test_an_out_of_range_switch_fraction_is_refused(fraction):
    scene = _scene()
    fallback = ms.nearest_non_object_labels(scene, 100)
    uniform = ms.switch_uniform(scene, 100, 0, "c" * 64)
    with pytest.raises(ms.ContractError):
        ms.apply_identity_switch(scene, fraction, 100, fallback, uniform)


# ---------------------------------------------------------------------------
# v4 -- missing CAMERAS (never missing frames: §3's recorded warning)
# ---------------------------------------------------------------------------


def test_missing_cameras_drops_exactly_the_declared_count():
    roster = LRV3_SPEC["train_cameras"]
    for n in ms.NOISE_MAGNITUDES["missing-cameras"]:
        dropped = ms.missing_camera_selection(roster, n)
        assert len(dropped) == n
        assert len(set(dropped)) == n
        assert set(dropped) <= set(roster)


def test_the_missing_camera_drop_sets_are_nested_and_deterministic():
    roster = LRV3_SPEC["train_cameras"]
    drops = {n: ms.missing_camera_selection(roster, n) for n in (1, 2, 4, 8)}
    assert set(drops[1]) <= set(drops[2]) <= set(drops[4]) <= set(drops[8])
    assert ms.missing_camera_selection(roster, 4) == drops[4]


def test_dropping_zero_cameras_drops_nobody():
    assert ms.missing_camera_selection(LRV3_SPEC["train_cameras"], 0) == []


def test_dropping_more_cameras_than_exist_is_refused():
    with pytest.raises(ms.ContractError):
        ms.missing_camera_selection(LRV3_SPEC["train_cameras"], 17)


def test_a_negative_drop_count_is_refused():
    with pytest.raises(ms.ContractError):
        ms.missing_camera_selection(LRV3_SPEC["train_cameras"], -1)


# ---------------------------------------------------------------------------
# v4 N3 -- the derangement
# ---------------------------------------------------------------------------


def test_the_shuffle_is_a_derangement_of_the_training_roster():
    roster = LRV3_SPEC["train_cameras"]
    mapping = ms.camera_shuffle_permutation(roster, ms.SHUFFLE_SEED_DECLARED)
    assert sorted(mapping) == sorted(roster)
    assert sorted(mapping.values()) == sorted(roster)
    assert all(int(k) != int(v) for k, v in mapping.items())
    assert ms.assert_derangement(mapping) is True


def test_the_shuffle_is_deterministic_in_its_seed():
    roster = LRV3_SPEC["train_cameras"]
    first = ms.camera_shuffle_permutation(roster, 0)
    assert ms.camera_shuffle_permutation(roster, 0) == first


def test_a_fixed_point_fails_the_derangement_assertion():
    """NEUTER: this is the whole content of N3. A shuffle that let one camera
    keep its own mask would leak real supervision into the control."""
    with pytest.raises(ms.PreconditionError):
        ms.assert_derangement({0: 0, 1: 3, 3: 1})


def test_a_non_permutation_fails_the_derangement_assertion():
    with pytest.raises(ms.PreconditionError):
        ms.assert_derangement({0: 1, 1: 5})


def test_a_single_camera_cannot_be_deranged_and_is_refused():
    with pytest.raises(ms.ContractError):
        ms.camera_shuffle_permutation([4], 0)


def test_the_declared_shuffle_seed_and_precision_bar_are_the_specs():
    assert ms.SHUFFLE_SEED_DECLARED == 0
    assert ms.SHUFFLE_PRECISION_BAR == 0.30
    assert ms.CHANCE_PRECISION == 0.071


# ---------------------------------------------------------------------------
# v4 N1 -- the noise must actually BITE
# ---------------------------------------------------------------------------


def test_N1_passes_when_the_noise_changed_pixels_and_reports_the_area_change():
    block = ms.noise_bite_block("erosion:1", {0: 8005, 14: 16},
                                {0: 7721, 14: 0}, 105666, 528 * 120000)
    assert block["passed"] is True
    assert block["n_pixel_labels_changed"] == 105666
    per_camera = block["per_camera_object_area"]
    assert per_camera["0"]["delta_px"] == -284
    assert per_camera["0"]["retained_fraction"] == pytest.approx(7721 / 8005.0)
    assert per_camera["14"]["emptied"] is True
    assert per_camera["0"]["emptied"] is False
    assert block["n_cameras_with_area_change"] == 2


def test_N1_FIRES_when_a_noise_level_is_a_no_op():
    """NEUTER: without this the instrument would return the CLEAN score under
    a noise label, which is the vacuity failure this project keeps catching."""
    block = ms.noise_bite_block("erosion:1", {0: 8005}, {0: 8005}, 0, 63360000)
    assert block["passed"] is False
    assert block["n_pixel_labels_changed"] == 0
    assert block["n_cameras_with_area_change"] == 0
    assert block["per_camera_object_area"]["0"]["delta_px"] == 0


def test_N1_fires_on_an_unchanged_area_that_nonetheless_moved_labels():
    """The check is on the LABELS, not on the area: an erosion and a dilation
    of equal size would leave the count alone while changing the mask."""
    block = ms.noise_bite_block("identity-switch:0.05", {0: 8005}, {0: 8005},
                                17, 63360000)
    assert block["passed"] is True
    assert block["per_camera_object_area"]["0"]["delta_px"] == 0


def test_N1_reports_an_undefined_retained_fraction_rather_than_dividing_by_zero():
    block = ms.noise_bite_block("erosion:8", {14: 0}, {14: 0}, 5, 10)
    assert block["per_camera_object_area"]["14"]["retained_fraction"] is None
    assert block["per_camera_object_area"]["14"]["emptied"] is False


# ---------------------------------------------------------------------------
# v4 N2 -- annihilation is reported, and checked where the spec declared it
# ---------------------------------------------------------------------------


def test_N2_passes_at_the_declared_counts_and_fails_otherwise():
    assert ms.annihilation_block(1, [14])["passed"] is True
    assert ms.annihilation_block(1, [])["passed"] is False
    assert ms.annihilation_block(1, [13, 14])["passed"] is False
    assert ms.annihilation_block(8, [13, 14])["passed"] is True
    assert ms.annihilation_block(8, [14])["passed"] is False


def test_N2_is_report_only_where_the_spec_declared_no_count():
    for k in (2, 4):
        block = ms.annihilation_block(k, [14])
        assert block["expected_by_spec"] is None
        assert block["passed"] is True


def test_N2_names_the_emptied_cameras_rather_than_only_counting_them():
    block = ms.annihilation_block(8, [14, 13])
    assert block["cameras_emptied"] == [13, 14]
    assert block["n_cameras_emptied"] == 2


# ---------------------------------------------------------------------------
# v4 -- the degradation point
# ---------------------------------------------------------------------------


CROSSING_CURVE = [
    {"magnitude": 1, "precision": 0.99, "recall": 0.98},
    {"magnitude": 2, "precision": 0.95, "recall": 0.95},
    {"magnitude": 4, "precision": 0.70, "recall": 0.93},
    {"magnitude": 8, "precision": 0.40, "recall": 0.50},
]


def test_the_degradation_point_is_the_smallest_failing_magnitude():
    assert ms.degradation_point(CROSSING_CURVE) == 4.0


def test_the_degradation_point_sorts_by_magnitude_rather_than_input_order():
    assert ms.degradation_point(list(reversed(CROSSING_CURVE))) == 4.0


def test_a_family_that_never_crosses_reports_the_literal_string():
    """§3: report it plainly and do NOT extend the range to find one. The
    string is reproduced verbatim, never paraphrased."""
    holds = [{"magnitude": m, "precision": 0.95, "recall": 0.95}
             for m in (1, 2, 4, 8)]
    assert ms.degradation_point(holds) == "no crossing within the swept range"
    assert ms.NO_CROSSING_TEXT == "no crossing within the swept range"


def test_recall_alone_can_trip_the_degradation_point():
    assert ms.degradation_point(
        [{"magnitude": 1, "precision": 1.0, "recall": 0.89}]) == 1.0


def test_precision_alone_can_trip_the_degradation_point():
    assert ms.degradation_point(
        [{"magnitude": 1, "precision": 0.79, "recall": 1.0}]) == 1.0


def test_an_undefined_metric_does_not_hold_the_gate():
    """An EMPTY selection has not met a 0.80 precision bar; it has failed to
    produce a number. Reading None as a pass would be vacuous."""
    assert ms.gate_holds({"precision": None, "recall": None}) is False
    assert ms.degradation_point(
        [{"magnitude": 1, "precision": None, "recall": None}]) == 1.0


def test_the_gate_the_degradation_point_reads_is_v3s_standing_pair():
    assert ms.VOTE_REFERENCE == {"precision": 0.80, "recall": 0.90}
    assert ms.gate_holds({"precision": 0.80, "recall": 0.90}) is True
    assert ms.gate_holds({"precision": 0.7999, "recall": 1.0}) is False


def test_an_empty_curve_has_no_degradation_point():
    with pytest.raises(ms.ContractError):
        ms.degradation_point([])


# ---------------------------------------------------------------------------
# v4 -- the frozen point grid, its naming, and the default being EXACTLY v3
# ---------------------------------------------------------------------------


def test_the_frozen_magnitudes_are_the_specs():
    assert ms.NOISE_MAGNITUDES == {
        "erosion": (1, 2, 4, 8),
        "dilation": (1, 2, 4, 8),
        "missing-cameras": (1, 2, 4, 8),
        "identity-switch": (0.05, 0.10, 0.25, 0.50),
    }


def test_the_sweep_is_the_clean_reference_plus_sixteen_points():
    points = ms.noise_point_keys()
    assert points[0] == ms.CLEAN_POINT
    assert len(points) == 17
    assert len(set(points)) == 17


def test_point_keys_name_their_family_and_magnitude():
    assert ms.point_key_text("erosion", 1) == "erosion:1"
    assert ms.point_key_text("identity-switch", 0.05) == "identity-switch:0.05"
    assert ms.point_key_text("clean", None) == "clean"


def test_the_default_request_is_exactly_v3():
    """ADDITIVITY: with none of the v4 flags given the measured point list is
    the clean reference alone, which is the v3 measurement."""
    assert ms.requested_points() == [ms.CLEAN_POINT]


def test_the_sweep_and_shuffle_flags_compose():
    assert len(ms.requested_points(sweep=True)) == 17
    assert len(ms.requested_points(sweep=True, shuffle_seed=0)) == 18
    assert ms.requested_points(shuffle_seed=0)[-1] == (ms.SHUFFLE_FAMILY, 0)
    assert ms.requested_points(family="dilation", magnitude=4) == [
        ms.CLEAN_POINT, ("dilation", 4)]


def test_a_single_family_point_is_not_duplicated_inside_a_sweep():
    points = ms.requested_points(sweep=True, family="erosion", magnitude=2)
    assert len(points) == 17
    assert len(set(points)) == 17


@pytest.mark.parametrize("magnitude", [3, 16, 0.2, 0.0])
def test_an_off_grid_magnitude_is_refused(magnitude):
    """§5 forbids extending a sweep range to find a crossing, so an off-grid
    magnitude is a contract error rather than a new point."""
    with pytest.raises(ms.ContractError):
        ms.canonical_magnitude("erosion", magnitude)


def test_a_frozen_magnitude_is_canonicalized_off_the_float_cli_value():
    assert ms.canonical_magnitude("erosion", 1.0) == 1
    assert ms.canonical_magnitude("identity-switch", 0.05) == 0.05


def test_a_family_without_a_magnitude_is_refused():
    with pytest.raises(ms.ContractError):
        ms.canonical_magnitude("erosion", None)


def test_a_magnitude_without_a_family_is_refused():
    with pytest.raises(ms.ContractError):
        ms.requested_points(magnitude=4)


def test_an_unknown_family_is_refused():
    with pytest.raises(ms.ContractError):
        ms.canonical_magnitude("blur", 1)


# ---------------------------------------------------------------------------
# v4 -- the P10/P11 tolerance RE-DECLARATION must be auditable in the output
# ---------------------------------------------------------------------------


def test_the_tolerance_redeclaration_records_both_the_old_and_the_new_value():
    """§4's final bullet: the ORIGINAL values and the reason are recorded so
    the relaxation is auditable and is never mistaken for a response to an
    unfavourable score."""
    block = ms.TOLERANCE_REDECLARATION
    p10 = block["P10_mask_partition_consistent"]
    p11 = block["P11_backward_repeatable"]
    assert p10["old_tolerance"] == 1e-6 == ms.P10_TOLERANCE_V3
    assert p10["new_tolerance"] == 1e-5 == ms.P10_TOLERANCE
    assert p11["old_rule"] == "bitwise identity" == ms.P11_RULE_V3
    assert p11["new_tolerance_absolute"] == 1e-4 == ms.P11_TOLERANCE


def test_the_tolerance_redeclaration_records_the_v3_measurements_and_grounds():
    block = ms.TOLERANCE_REDECLARATION
    assert block["P10_mask_partition_consistent"][
        "v3_measured_relative_deviation"] == pytest.approx(1.0692913292587036e-06)
    assert block["P11_backward_repeatable"][
        "v3_measured_max_abs_difference"] == 2.0 ** -16
    assert block["P10_mask_partition_consistent"]["v3_outcome"] == "FAILED"
    assert block["P11_backward_repeatable"]["v3_outcome"] == "FAILED"
    assert "PLATFORM" in block["grounds"]
    assert "atomicAdd" in block["reason"]
    assert block["declared_in"].startswith(ms.V4_SPEC_PAGE)


def test_the_redeclared_tolerances_admit_the_v3_measurements_and_were_needed():
    """Both halves matter: the relaxation must be large enough to admit what
    v3 measured, and the ORIGINAL must have been small enough to reject it --
    otherwise the re-declaration would be cosmetic."""
    assert 1.0692913292587036e-06 <= ms.P10_TOLERANCE
    assert 1.0692913292587036e-06 > ms.P10_TOLERANCE_V3
    assert 2.0 ** -16 <= ms.P11_TOLERANCE


def test_the_tolerance_block_is_json_serialisable_for_the_report():
    """It travels in every report; a non-serialisable value would lose the
    audit trail at write time rather than at review time."""
    round_tripped = json.loads(json.dumps(ms.TOLERANCE_REDECLARATION,
                                          sort_keys=True))
    assert round_tripped == ms.TOLERANCE_REDECLARATION


# ---------------------------------------------------------------------------
# v4 -- the perturbation table, on the REAL fixture
# ---------------------------------------------------------------------------


def _real_table(sweep=True, shuffle_seed=0):
    identity_dir = ms.resolve_identity_dir(LRV3_DIR, "")
    spec = json.loads((LRV3_DIR / "event_spec.json").read_text(encoding="utf-8"))
    frames = ms.default_frame_set(spec)
    views = [(c, f) for c in spec["train_cameras"] for f in frames]
    census, by_view, by_digest = ms.identity_census(identity_dir, views,
                                                    min(frames), 100)
    points = ms.requested_points(sweep=sweep, shuffle_seed=shuffle_seed)
    table = ms.PerturbationTable(points, by_view, by_digest, 100,
                                 spec["train_cameras"], noise_seed=0,
                                 shuffle_seed=shuffle_seed)
    return table, points, views, census


def test_the_table_reproduces_the_clean_buffers_untouched():
    _require_fixture()
    table, _, views, _ = _real_table()
    for camera, frame in views[:40]:
        key = table.buffer_key(ms.CLEAN_POINT, camera, frame)
        assert key == table.clean_digest(camera, frame)
    changed, compared = table.label_change_count(ms.CLEAN_POINT, views)
    assert changed == 0
    assert compared == len(views) * 300 * 400


def test_every_perturbed_point_measurably_bites_on_the_real_fixture():
    """N1 on the real masks, at every one of the sixteen frozen points."""
    _require_fixture()
    table, points, views, _ = _real_table()
    for point in points:
        if point == ms.CLEAN_POINT:
            continue
        changed, _ = table.label_change_count(point, views)
        assert changed > 0, "%s changed nothing" % (ms.point_key_text(*point),)


def test_the_perturbed_buffers_stay_partitions_of_the_clean_class_set():
    """P10 reads the partition, so a perturbation that invented a class or
    left a pixel unlabelled would be reported as a mechanism failure."""
    _require_fixture()
    table, points, _, census = _real_table()
    clean_classes = set(census["class_ids_present"])
    for point in points:
        buffer = table.buffer(table.buffer_key(point, 0, 0))
        assert tuple(buffer.shape) == tuple(census["shape"])
        assert set(np.unique(buffer).tolist()) <= clean_classes


def test_the_shuffled_buffer_is_another_cameras_clean_buffer():
    _require_fixture()
    table, _, _, _ = _real_table()
    point = (ms.SHUFFLE_FAMILY, 0)
    for camera in table.camera_ids:
        donor = table.shuffle_map[camera]
        assert donor != camera
        assert table.buffer_key(point, camera, 0) == table.clean_digest(donor, 0)


def test_the_real_erosion_areas_fall_monotonically_with_k():
    _require_fixture()
    table, _, _, _ = _real_table()
    areas = [table.per_camera_areas(("erosion", k), 0)
             for k in ms.NOISE_MAGNITUDES["erosion"]]
    for camera in table.camera_ids:
        series = [a[camera] for a in areas]
        assert all(b <= a for a, b in zip(series, series[1:]))


def test_the_real_dilation_areas_rise_monotonically_with_k():
    _require_fixture()
    table, _, _, _ = _real_table()
    areas = [table.per_camera_areas(("dilation", k), 0)
             for k in ms.NOISE_MAGNITUDES["dilation"]]
    for camera in table.camera_ids:
        series = [a[camera] for a in areas]
        assert all(b >= a for a, b in zip(series, series[1:]))


def test_a_dropped_camera_supplies_no_object_and_the_others_are_untouched():
    _require_fixture()
    table, _, _, _ = _real_table()
    clean = table.per_camera_areas(ms.CLEAN_POINT, 0)
    for n in ms.NOISE_MAGNITUDES["missing-cameras"]:
        dropped = table.dropped_cameras[n]
        assert len(dropped) == n
        areas = table.per_camera_areas(("missing-cameras", n), 0)
        for camera in table.camera_ids:
            if camera in dropped:
                assert areas[camera] == 0
            else:
                assert areas[camera] == clean[camera]


def test_the_table_refuses_a_point_list_without_the_clean_reference():
    _require_fixture()
    identity_dir = ms.resolve_identity_dir(LRV3_DIR, "")
    spec = json.loads((LRV3_DIR / "event_spec.json").read_text(encoding="utf-8"))
    frames = ms.default_frame_set(spec)
    views = [(c, f) for c in spec["train_cameras"] for f in frames]
    _, by_view, by_digest = ms.identity_census(identity_dir, views,
                                               min(frames), 100)
    with pytest.raises(ms.ContractError):
        ms.PerturbationTable([("erosion", 1)], by_view, by_digest, 100,
                             spec["train_cameras"])


def test_the_table_refuses_a_shuffle_lookup_with_no_seed():
    _require_fixture()
    table, _, _, _ = _real_table(sweep=False, shuffle_seed=None)
    assert table.shuffle_map is None
    with pytest.raises(ms.ContractError):
        table.buffer_key((ms.SHUFFLE_FAMILY, 0), 0, 0)


def test_the_class_mask_dedupe_is_by_content_so_equal_masks_share_a_digest():
    """The saving that makes a seventeen-point sweep cost one render pass: two
    points whose buffers agree on a class must resolve to the same digest, and
    two that disagree must not."""
    _require_fixture()
    table, _, _, _ = _real_table()
    clean_key = table.buffer_key(ms.CLEAN_POINT, 0, 0)
    missing_key = table.buffer_key(("missing-cameras", 1), 0, 0)
    eroded_key = table.buffer_key(("erosion", 1), 0, 0)
    assert missing_key == clean_key          # cam00 is not among the dropped
    assert table.mask_digest(clean_key, 100) == table.mask_digest(missing_key, 100)
    assert table.mask_digest(clean_key, 100) != table.mask_digest(eroded_key, 100)
