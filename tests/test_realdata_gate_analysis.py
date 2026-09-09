"""CPU tests for scripts/realdata_gate_analysis.py (the frozen real-data gate analysis).

Run with:
    python -m pytest tests/test_realdata_gate_analysis.py -q

Every fixture is a synthetic ``f_box_profile.json`` written to a temp dir with
closed-form per-window PSNRs, so each pooled endpoint, each contrast and each
verdict branch has a hand-checkable expected value. The script's own pure-Python
Student-t is cross-checked against scipy where scipy is installed.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import realdata_gate_analysis as rga  # noqa: E402

EVENT = rga.SPEC["event_name"]
N_FRAMES = rga.SPEC["expected_n_frames"]

try:
    from scipy import stats as sps  # noqa: E402

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - exercised only where scipy is absent
    HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _fill(base, segments):
    """A 300-long per-frame array: `base` everywhere, overwritten by segments."""
    values = [float(base)] * N_FRAMES
    for lo, hi, value in segments:
        for f in range(lo, hi + 1):
            values[f] = float(value)
    return values


def write_profile(run_dir, event_values, whole_values):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_frames": N_FRAMES,
        "frames": list(range(N_FRAMES)),
        "whole_frame_psnr": whole_values,
        "events": {
            EVENT: {
                "bbox": [10, 20, 110, 120],
                "frames": [[190, 209]],
                "class": "dynamic_content",
                "per_frame_psnr": event_values,
                "pooled_psnr_in_window": None,
                "pixel_times_in_window": 100 * 100 * 20,
            },
            "A_hand_press_reveal": {
                "bbox": [0, 0, 10, 10],
                "frames": [[0, 9]],
                "per_frame_psnr": [30.0] * N_FRAMES,
            },
        },
    }
    with open(run_dir / rga.SPEC["profile_filename"], "w") as fh:
        json.dump(payload, fh)
    return run_dir


def make_cell(
    root,
    arm,
    seed,
    *,
    p1,
    p2,
    s1_tail,
    h1,
    c1,
    whole,
    base=30.0,
    precondition=None,
    final_points=None,
):
    """One synthetic cell whose pooled endpoints equal the values passed in."""
    run_dir = Path(root) / f"{arm}_seed{seed}"
    event_values = _fill(
        base,
        [
            (100, 117, h1),
            (118, 147, c1),
            (148, 157, h1),
            (158, 187, p1),
            (190, 199, p2),
            (200, 209, s1_tail),
        ],
    )
    write_profile(run_dir, event_values, [float(whole)] * N_FRAMES)
    if precondition is not None:
        with open(run_dir / rga.SPEC["precondition_filename"], "w") as fh:
            json.dump(precondition, fh)
    if final_points is not None:
        with open(run_dir / rga.SPEC["summary_filename"], "w") as fh:
            json.dump({"summary": {"best_val/points/total": final_points}}, fh)
    return {"arm": arm, "seed": seed, "run_dir": str(run_dir), "status": "complete"}


GOOD_PRECONDITION = {
    "gated_rows_seeding": 4200,
    "n_rows_seeding": 200000,
    "gated_rows_final": 5000,
    "n_rows_final": 900000,
    "gated_rows_fbox_frame150": 800,
    "frames_presence_zero": 27,
    "reserved_units": 16,
    "training_units_total": 5700,
}

SMALL_JITTER = [-0.015, -0.005, 0.005, 0.015]
BIG_JITTER = [-1.5, -0.5, 0.5, 1.5]


def build_manifest(
    tmp_path,
    *,
    g_p1_offset=1.0,
    gmis_p1_offset=0.0,
    gmis_c1_offset=-2.0,
    p1_jitter=SMALL_JITTER,
    g_preconditions=None,
    gmis_preconditions=None,
    n_seeds=4,
):
    """A 3-arm manifest; only the knobs a verdict branch needs are exposed."""
    cells = []
    for i in range(n_seeds):
        j = SMALL_JITTER[i % len(SMALL_JITTER)]
        pj = p1_jitter[i % len(p1_jitter)]
        common = dict(p2=26.0 + j, s1_tail=25.0 + j, h1=31.0 + j, whole=33.0 + j)
        cells.append(
            make_cell(tmp_path, "U", i, p1=24.0 + pj, c1=32.0 + j, **common)
        )
        pre_g = GOOD_PRECONDITION if g_preconditions is None else g_preconditions[i]
        cells.append(
            make_cell(
                tmp_path,
                "G",
                i,
                p1=24.0 + g_p1_offset + pj,
                c1=32.0 + j,
                precondition=pre_g,
                final_points=900000 + i,
                **common,
            )
        )
        pre_m = GOOD_PRECONDITION if gmis_preconditions is None else gmis_preconditions[i]
        cells.append(
            make_cell(
                tmp_path,
                "GMIS",
                i,
                p1=24.0 + gmis_p1_offset + pj,
                c1=32.0 + gmis_c1_offset + j,
                precondition=pre_m,
                **common,
            )
        )
    path = Path(tmp_path) / "manifest.json"
    with open(path, "w") as fh:
        json.dump({"cells": cells}, fh)
    return path


# ---------------------------------------------------------------------------
# 1. Pooling arithmetic
# ---------------------------------------------------------------------------


def test_pooled_psnr_matches_direct_mse():
    mses = np.array([1e-3, 4e-4, 2.5e-3, 7e-4, 1.1e-3])
    psnrs = -10.0 * np.log10(mses)
    expected = -10.0 * math.log10(float(mses.mean()))
    assert rga.pooled_psnr(psnrs) == pytest.approx(expected, abs=1e-12)


def test_pooled_psnr_of_constant_window_is_that_constant():
    assert rga.pooled_psnr([27.5] * 30) == pytest.approx(27.5, abs=1e-12)


def test_pooled_psnr_is_dominated_by_the_worst_frame():
    # Pooling MSEs is not averaging dB: one bad frame drags the pool down.
    pooled = rga.pooled_psnr([40.0] * 29 + [10.0])
    assert pooled < 25.0
    assert pooled == pytest.approx(-10.0 * math.log10((29 * 1e-4 + 1e-1) / 30), abs=1e-12)


def test_endpoint_extraction_uses_the_frozen_windows(tmp_path):
    cell = rga.load_cell(
        make_cell(tmp_path, "U", 0, p1=21.0, p2=22.0, s1_tail=23.0, h1=24.0,
                  c1=25.0, whole=26.0, base=40.0)
    )
    e = cell["endpoints"]
    assert e["P1"] == pytest.approx(21.0, abs=1e-9)     # 158..187
    assert e["P2"] == pytest.approx(22.0, abs=1e-9)     # 190..199
    assert e["H2"] == pytest.approx(26.0, abs=1e-9)     # whole frame
    assert e["C1"] == pytest.approx(25.0, abs=1e-9)     # 118..147
    # S1 = 190..209 pools P2's ten frames with s1_tail's ten
    assert e["S1"] == pytest.approx(
        -10.0 * math.log10((10 * 10 ** -2.2 + 10 * 10 ** -2.3) / 20), abs=1e-9
    )
    # H1 = 100..157 is 28 frames of h1 and 30 of c1
    assert e["H1"] == pytest.approx(
        -10.0 * math.log10((28 * 10 ** -2.4 + 30 * 10 ** -2.5) / 58), abs=1e-9
    )


def test_missing_window_frame_is_an_error(tmp_path):
    run_dir = write_profile(tmp_path / "short", [30.0] * 100, [30.0] * 100)
    payload = json.loads((run_dir / "f_box_profile.json").read_text())
    payload["frames"] = list(range(100))
    (run_dir / "f_box_profile.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing"):
        rga.load_cell({"arm": "U", "seed": 0, "run_dir": str(run_dir), "status": "complete"})


# ---------------------------------------------------------------------------
# 2. Student-t and Welch
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_SCIPY, reason="scipy not installed")
def test_t_distribution_matches_scipy():
    for df in (1.0, 2.5, 6.0, 11.37, 40.0, 300.0):
        for t in (-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0):
            assert rga.t_cdf(t, df) == pytest.approx(sps.t.cdf(t, df), abs=1e-12)
            assert rga.t_two_sided_p(abs(t), df) == pytest.approx(
                2.0 * sps.t.sf(abs(t), df), abs=1e-12
            )
        for q in (0.0125, 0.025, 0.5, 0.9, 0.975, 0.9875):
            assert rga.t_ppf(q, df) == pytest.approx(sps.t.ppf(q, df), abs=1e-9)


def test_welch_ci_against_a_hand_computed_case():
    a = [1.0, 2.0, 3.0, 4.0]     # mean 2.5, var 5/3
    b = [2.0, 3.0, 4.0, 5.0]     # mean 3.5, var 5/3
    res = rga.welch(a, b)
    assert res["diff"] == pytest.approx(-1.0, abs=1e-12)
    assert res["se"] == pytest.approx(math.sqrt((5.0 / 3.0) / 4.0 * 2.0), abs=1e-12)
    # equal variances and equal n -> Welch df collapses to n1 + n2 - 2 = 6
    assert res["df"] == pytest.approx(6.0, abs=1e-9)
    tcrit = rga.t_ppf(0.9875, 6.0)
    assert tcrit == pytest.approx(2.9687, abs=1e-3)
    assert res["ci_low"] == pytest.approx(-1.0 - tcrit * res["se"], abs=1e-12)
    assert res["ci_high"] == pytest.approx(-1.0 + tcrit * res["se"], abs=1e-12)
    assert res["conf"] == 0.975


@pytest.mark.skipif(not HAVE_SCIPY, reason="scipy not installed")
def test_welch_p_matches_scipy_ttest_ind():
    a = [24.9, 25.4, 25.1, 26.0, 25.3]
    b = [24.1, 24.6, 23.9, 24.4]
    res = rga.welch(a, b)
    ref = sps.ttest_ind(a, b, equal_var=False)
    assert res["t"] == pytest.approx(float(ref.statistic), abs=1e-10)
    assert res["p"] == pytest.approx(float(ref.pvalue), abs=1e-12)


def test_welch_zero_variance_rule():
    res = rga.welch([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
    assert res["se"] == 0.0
    assert res["ci_low"] == 0.0 and res["ci_high"] == 0.0
    assert res["p"] == 1.0
    res2 = rga.welch([6.0, 6.0, 6.0], [5.0, 5.0, 5.0])
    assert res2["ci_low"] == 1.0 and res2["ci_high"] == 1.0
    assert res2["p"] == 0.0


def test_welch_insufficient_n_is_flagged_not_faked():
    res = rga.welch([1.0], [2.0, 3.0])
    assert res["insufficient_n"] is True
    assert res["ci_low"] is None and res["p"] is None


# ---------------------------------------------------------------------------
# 3. Placebo splits
# ---------------------------------------------------------------------------


def test_balanced_split_count_is_35_for_eight_cells():
    splits = rga.balanced_splits(8)
    assert len(splits) == 35
    # every split is balanced, disjoint, and covers all eight indices
    for a, b in splits:
        assert len(a) == 4 and len(b) == 4
        assert set(a).isdisjoint(b)
        assert sorted(a + b) == list(range(8))
    # mirrors are de-duplicated: no split's complement appears as another split
    seen = {tuple(sorted(a)) for a, _ in splits}
    for a, b in splits:
        assert tuple(sorted(b)) not in seen


def test_balanced_split_counts_for_other_sizes():
    assert len(rga.balanced_splits(4)) == 3      # C(3,1)
    assert len(rga.balanced_splits(6)) == 10     # C(5,2)
    assert rga.balanced_splits(5) == []          # odd -> undefined
    assert rga.balanced_splits(0) == []


def test_placebo_quantile_is_symmetric_and_frozen():
    dist = rga.placebo_distribution([1.0, 2.0, 3.0, 4.0])
    assert dist["n_splits"] == 3
    # half-mean differences: {1,2}-{3,4} = -2, {1,3}-{2,4} = -1, {1,4}-{2,3} = 0
    assert sorted(abs(d) for d in dist["diffs"]) == [0.0, 1.0, 2.0]
    assert dist["quantile"] == 0.95
    assert dist["abs_quantile"] == pytest.approx(
        float(np.percentile([0.0, 1.0, 2.0], 95.0)), abs=1e-12
    )


def test_placebo_needs_an_even_number_of_u_cells():
    dist = rga.placebo_distribution([1.0, 2.0, 3.0])
    assert dist["n_splits"] == 0 and dist["abs_quantile"] is None


# ---------------------------------------------------------------------------
# 4. Sizing
# ---------------------------------------------------------------------------


def test_sizing_reproduces_53_per_arm_at_the_replicate_floor():
    s_p, n2 = rga.sizing_n2(0.4945, 0.4945)
    assert s_p == pytest.approx(0.4945, abs=1e-12)
    assert n2 == 53
    assert n2 == math.ceil(2 * 9.5049 * (0.4945 / 0.30) ** 2 + 2.2414 ** 2 / 4)


def test_sizing_pools_the_two_arm_sds_and_rounds_up():
    s_p, n2 = rga.sizing_n2(0.40, 0.60)
    assert s_p == pytest.approx(math.sqrt((0.16 + 0.36) / 2.0), abs=1e-12)
    assert n2 == math.ceil(2 * 9.5049 * (s_p / 0.30) ** 2 + 2.2414 ** 2 / 4)
    # a sample size must round UP, never to nearest
    tiny_sp, tiny_n2 = rga.sizing_n2(1e-9, 1e-9)
    assert tiny_n2 == 2  # ceil(0 + 1.2559...) == 2


def test_sizing_needs_both_arm_sds():
    assert rga.sizing_n2(None, 0.5) == (None, None)


# ---------------------------------------------------------------------------
# 5. TOST
# ---------------------------------------------------------------------------


def test_tost_passes_for_a_tight_null_and_fails_for_a_wide_one():
    tight = rga.tost(rga.welch([25.01, 25.02, 25.00, 25.03], [25.00, 25.01, 25.02, 25.00]))
    assert tight["passes"] is True
    wide = rga.tost(rga.welch([25.0, 26.5, 23.5, 25.2], [25.1, 23.9, 26.4, 24.8]))
    assert wide["passes"] is False
    assert wide["margin"] == 0.30 and wide["alpha"] == 0.025


# ---------------------------------------------------------------------------
# 6. Verdict branches
# ---------------------------------------------------------------------------


def _verdict(report, key="P1", analysis_set="itt"):
    return report["analysis"][analysis_set]["verdicts"][key]["verdict"]


def test_verdict_benefit(tmp_path):
    report = rga.run(str(build_manifest(tmp_path, g_p1_offset=1.0, gmis_p1_offset=0.0)))
    block = report["analysis"]["itt"]
    assert block["positive_control"]["valid"] is True
    assert block["harm_guard"]["passes"] is True
    assert block["g_vs_u"]["P1"]["diff"] == pytest.approx(1.0, abs=1e-6)
    assert _verdict(report) == "BENEFIT"
    assert report["headline"]["P1"]["itt"] == "BENEFIT"


def test_verdict_benefit_not_separable(tmp_path):
    # GMIS gains as much as G on P1 -> the correctness contrast cannot separate them
    report = rga.run(str(build_manifest(tmp_path, g_p1_offset=1.0, gmis_p1_offset=1.0)))
    block = report["analysis"]["itt"]
    assert block["positive_control"]["valid"] is True
    assert block["verdicts"]["P1"]["correctness_separable"] is False
    assert _verdict(report) == "BENEFIT_NOT_SEPARABLE"


def test_verdict_harm(tmp_path):
    report = rga.run(str(build_manifest(tmp_path, g_p1_offset=-1.0)))
    assert report["analysis"]["itt"]["verdicts"]["P1"]["ci_below_zero"] is True
    assert _verdict(report) == "HARM"


def test_verdict_equivalent(tmp_path):
    report = rga.run(str(build_manifest(tmp_path, g_p1_offset=0.01)))
    block = report["analysis"]["itt"]
    assert block["equivalence"]["P1"]["passes"] is True
    assert block["verdicts"]["P1"]["ci_above_zero"] is False
    assert _verdict(report) == "EQUIVALENT"


def test_verdict_no_detected_difference(tmp_path):
    report = rga.run(
        str(build_manifest(tmp_path, g_p1_offset=0.1, p1_jitter=BIG_JITTER)),
        last_wave=False,
    )
    block = report["analysis"]["itt"]
    assert block["equivalence"]["P1"]["passes"] is False
    assert block["verdicts"]["P1"]["ci_above_zero"] is False
    assert block["sizing"]["decision"] == "feasibility_stop"
    assert _verdict(report) == "NO_DETECTED_DIFFERENCE"


def test_verdict_feasibility_stop_only_on_the_last_wave(tmp_path):
    manifest = str(build_manifest(tmp_path, g_p1_offset=0.1, p1_jitter=BIG_JITTER))
    assert _verdict(rga.run(manifest, last_wave=False)) == "NO_DETECTED_DIFFERENCE"
    assert _verdict(rga.run(manifest, last_wave=True)) == "FEASIBILITY_STOP"


def test_verdict_design_without_power_overrides_everything(tmp_path):
    # the positive control produces no effect -> the design cannot detect anything
    report = rga.run(
        str(build_manifest(tmp_path, g_p1_offset=1.0, gmis_c1_offset=0.0)),
        last_wave=True,
    )
    block = report["analysis"]["itt"]
    assert block["positive_control"]["valid"] is False
    assert block["verdicts"]["P1"]["ci_above_zero"] is True  # would have been BENEFIT
    assert _verdict(report) == "DESIGN_WITHOUT_POWER"


def test_control_needs_both_direction_and_magnitude(tmp_path):
    # a real but sub-threshold control effect (-0.5 dB) is still INVALID
    report = rga.run(str(build_manifest(tmp_path, gmis_c1_offset=-0.5)))
    ctl = report["analysis"]["itt"]["positive_control"]
    assert ctl["result"]["ci_high"] < 0.0
    assert ctl["valid"] is False
    assert _verdict(report) == "DESIGN_WITHOUT_POWER"


def test_harm_guard_fails_when_g_loses_the_harm_region(tmp_path):
    cells = []
    for i in range(4):
        j = SMALL_JITTER[i]
        common = dict(p2=26.0 + j, s1_tail=25.0 + j, whole=33.0 + j)
        cells.append(make_cell(tmp_path, "U", i, p1=24.0 + j, h1=31.0 + j,
                               c1=32.0 + j, **common))
        cells.append(make_cell(tmp_path, "G", i, p1=25.0 + j, h1=30.0 + j,
                               c1=32.0 + j, precondition=GOOD_PRECONDITION, **common))
        cells.append(make_cell(tmp_path, "GMIS", i, p1=24.0 + j, h1=31.0 + j,
                               c1=30.0 + j, precondition=GOOD_PRECONDITION, **common))
    path = Path(tmp_path) / "harm_manifest.json"
    path.write_text(json.dumps({"cells": cells}))
    report = rga.run(str(path))
    block = report["analysis"]["itt"]
    assert block["harm_guard"]["per_endpoint"]["H1"]["passes"] is False
    assert block["harm_guard"]["passes"] is False
    # the P1 gain is real and the control is valid, but the harm guard blocks BENEFIT
    assert block["verdicts"]["P1"]["ci_above_zero"] is True
    assert _verdict(report) == "NO_DETECTED_DIFFERENCE"


def test_every_verdict_label_is_declared_in_the_spec(tmp_path):
    reached = set()
    reached.add(_verdict(rga.run(str(build_manifest(tmp_path / "a", g_p1_offset=1.0)))))
    reached.add(_verdict(rga.run(str(build_manifest(tmp_path / "b", g_p1_offset=1.0,
                                                    gmis_p1_offset=1.0)))))
    reached.add(_verdict(rga.run(str(build_manifest(tmp_path / "c", g_p1_offset=-1.0)))))
    reached.add(_verdict(rga.run(str(build_manifest(tmp_path / "d", g_p1_offset=0.01)))))
    reached.add(_verdict(rga.run(str(build_manifest(tmp_path / "e", g_p1_offset=0.1,
                                                    p1_jitter=BIG_JITTER)))))
    reached.add(_verdict(rga.run(str(build_manifest(tmp_path / "f", g_p1_offset=0.1,
                                                    p1_jitter=BIG_JITTER)),
                                 last_wave=True)))
    reached.add(_verdict(rga.run(str(build_manifest(tmp_path / "g", gmis_c1_offset=0.0)))))
    assert reached == set(rga.SPEC["VERDICTS"])


# ---------------------------------------------------------------------------
# 7. Analysis sets, preconditions, bookkeeping
# ---------------------------------------------------------------------------


def test_itt_and_mechanism_exercised_sets_separate(tmp_path):
    bad = dict(GOOD_PRECONDITION, gated_rows_final=10)
    report = rga.run(
        str(build_manifest(tmp_path, g_preconditions=[GOOD_PRECONDITION] * 3 + [bad]))
    )
    assert report["analysis"]["itt"]["n_per_arm"]["G"] == 4
    assert report["analysis"]["mechanism_exercised"]["n_per_arm"]["G"] == 3
    assert report["analysis"]["itt"]["n_per_arm"]["U"] == 4
    assert report["analysis"]["mechanism_exercised"]["n_per_arm"]["U"] == 4
    excluded = [c for c in report["cells"] if not c["mechanism_exercised"]]
    assert len(excluded) == 1 and excluded[0]["arm"] == "G"
    assert "gated_rows_final" in excluded[0]["mechanism_reason"]
    assert report["headline"]["P1"]["itt"] == "BENEFIT"
    assert report["headline"]["P1"]["mechanism_exercised"] in rga.SPEC["VERDICTS"]


def test_each_precondition_criterion_can_exclude_a_cell():
    assert rga.mechanism_exercised("U", None)[0] is True
    assert rga.mechanism_exercised("G", None)[0] is False
    assert rga.mechanism_exercised("G", GOOD_PRECONDITION)[0] is True
    for field, bad_value in (
        ("gated_rows_final", rga.MIN_GATED_SURVIVING - 1),
        ("gated_rows_fbox_frame150", rga.MIN_GATED_FBOX - 1),
        ("frames_presence_zero", 0),
    ):
        passes, reason = rga.mechanism_exercised("G", dict(GOOD_PRECONDITION, **{field: bad_value}))
        assert passes is False and field in reason


def test_g_cell_without_precondition_is_excluded_from_me(tmp_path):
    report = rga.run(str(build_manifest(tmp_path, g_preconditions=[None] * 4)))
    assert report["analysis"]["mechanism_exercised"]["n_per_arm"]["G"] == 0
    assert report["analysis"]["itt"]["n_per_arm"]["G"] == 4


def test_failed_cells_are_listed_and_skipped(tmp_path):
    manifest_path = build_manifest(tmp_path)
    payload = json.loads(Path(manifest_path).read_text())
    payload["cells"].append(
        {"arm": "G", "seed": 99, "run_dir": str(tmp_path / "never_ran"), "status": "failed"}
    )
    Path(manifest_path).write_text(json.dumps(payload))
    report = rga.run(str(manifest_path))
    assert report["n_cells"] == 13 and report["n_complete"] == 12
    assert report["failed_cells"] == [
        {"arm": "G", "seed": 99, "run_dir": str(tmp_path / "never_ran")}
    ]
    assert report["analysis"]["itt"]["n_per_arm"]["G"] == 4


def test_missing_profile_for_a_complete_cell_is_an_error(tmp_path):
    manifest_path = build_manifest(tmp_path)
    payload = json.loads(Path(manifest_path).read_text())
    payload["cells"].append(
        {"arm": "G", "seed": 98, "run_dir": str(tmp_path / "absent"), "status": "complete"}
    )
    Path(manifest_path).write_text(json.dumps(payload))
    with pytest.raises(FileNotFoundError, match="complete"):
        rga.run(str(manifest_path))


def test_unknown_arm_and_status_are_rejected(tmp_path):
    entry = make_cell(tmp_path, "U", 0, p1=24.0, p2=26.0, s1_tail=25.0, h1=31.0,
                      c1=32.0, whole=33.0)
    with pytest.raises(ValueError, match="unknown arm"):
        rga.load_cell(dict(entry, arm="X"))
    with pytest.raises(ValueError, match="unknown status"):
        rga.load_cell(dict(entry, status="running"))
    with pytest.raises(ValueError, match="missing required field"):
        rga.load_cell({"arm": "U", "seed": 0, "run_dir": entry["run_dir"]})


def test_reserved_unit_check_reports_n_over_N_and_blocks_on_disagreement(tmp_path):
    report = rga.run(str(build_manifest(tmp_path)))
    ru = report["reserved_unit_check"]
    assert ru["reported_as"] == "8/12"   # only G and GMIS carry precondition.json
    assert ru["consistent"] is True
    assert ru["reserved_units"] == 16
    assert ru["reserved_fraction"] == "16/5700"
    assert report["blocking_errors"] == []

    other = dict(GOOD_PRECONDITION, reserved_units=8)
    bad = rga.run(
        str(build_manifest(tmp_path / "mismatch",
                           g_preconditions=[GOOD_PRECONDITION] * 3 + [other]))
    )
    assert bad["reserved_unit_check"]["consistent"] is False
    assert bad["blocking_errors"] and "reserved_units" in bad["blocking_errors"][0]


def test_final_points_are_read_and_reported(tmp_path):
    report = rga.run(str(build_manifest(tmp_path)))
    g_cells = [c for c in report["cells"] if c["arm"] == "G"]
    assert [c["final_points"] for c in g_cells] == [900000, 900001, 900002, 900003]
    assert [c["final_points"] for c in report["cells"] if c["arm"] == "U"] == [None] * 4
    cap = report["analysis"]["itt"]["capacity_descriptive"]
    assert cap["G"]["n"] == 4
    assert cap["G"]["mean_final_points"] == pytest.approx(900001.5)
    assert cap["U"]["n"] == 0 and cap["U"]["mean_final_points"] is None


def test_final_points_alternate_key_and_tolerance(tmp_path):
    run_dir = tmp_path / "alt"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"final": {"points": {"total": 42}}}))
    assert rga.read_final_points(str(run_dir)) == 42
    (run_dir / "summary.json").write_text("{not json")
    assert rga.read_final_points(str(run_dir)) is None
    assert rga.read_final_points(str(tmp_path / "nonexistent")) is None


def test_short_profile_warns_but_does_not_block(tmp_path):
    manifest_path = build_manifest(tmp_path)
    payload = json.loads(Path(manifest_path).read_text())
    victim = Path(payload["cells"][0]["run_dir"]) / "f_box_profile.json"
    profile = json.loads(victim.read_text())
    for key in ("frames", "whole_frame_psnr"):
        profile[key] = profile[key][:250]
    profile["events"][EVENT]["per_frame_psnr"] = \
        profile["events"][EVENT]["per_frame_psnr"][:250]
    victim.write_text(json.dumps(profile))
    report = rga.run(str(manifest_path))
    assert any("250 frames" in w for w in report["warnings"])
    assert report["blocking_errors"] == []


# ---------------------------------------------------------------------------
# 8. CLI surface
# ---------------------------------------------------------------------------


def test_print_spec_emits_the_frozen_constants(capsys):
    assert rga.main(["--print-spec"]) == 0
    spec = json.loads(capsys.readouterr().out)
    assert spec["event_name"] == EVENT
    assert spec["CONTROL_MIN_EFFECT_DB"] == 1.0
    assert spec["PLACEBO_QUANTILE"] == 0.95
    assert spec["HARM_MARGIN_DB"] == 0.15
    assert spec["MIN_GATED_SURVIVING"] == 1000
    assert spec["MIN_GATED_FBOX"] == 100
    assert spec["SIZING"]["DELTA"] == 0.30
    assert spec["SIZING"]["N2_CAP"] == 60
    assert spec["endpoints"]["P1"]["frames_inclusive"] == [158, 187]
    assert spec["endpoints"]["P2"]["frames_inclusive"] == [190, 199]
    assert spec["endpoints"]["S1"]["frames_inclusive"] == [190, 209]
    assert spec["endpoints"]["H1"]["frames_inclusive"] == [100, 157]
    assert spec["endpoints"]["C1"]["frames_inclusive"] == [118, 147]


def test_cli_writes_json_and_prints_a_markdown_table(tmp_path, capsys):
    manifest_path = build_manifest(tmp_path)
    out = tmp_path / "gate.json"
    assert rga.main(["--manifest", str(manifest_path), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "| arm | seed | status |" in printed
    assert "BENEFIT" in printed
    saved = json.loads(out.read_text())
    assert saved["spec"]["spec_version"] == rga.SPEC["spec_version"]
    assert saved["headline"]["P1"]["itt"] == "BENEFIT"


def test_cli_exit_code_2_on_a_blocking_error(tmp_path, capsys):
    other = dict(GOOD_PRECONDITION, reserved_units=8)
    manifest_path = build_manifest(
        tmp_path, g_preconditions=[GOOD_PRECONDITION] * 3 + [other]
    )
    assert rga.main(["--manifest", str(manifest_path)]) == 2
    assert "BLOCKING" in capsys.readouterr().err


def test_analysis_is_deterministic(tmp_path):
    manifest_path = str(build_manifest(tmp_path))
    first = json.dumps(rga.run(manifest_path), sort_keys=True)
    second = json.dumps(rga.run(manifest_path), sort_keys=True)
    assert first == second
