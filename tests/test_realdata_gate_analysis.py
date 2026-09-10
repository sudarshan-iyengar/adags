"""CPU tests for scripts/realdata_gate_analysis.py (the frozen real-data gate analysis).

Run with:
    python -m pytest tests/test_realdata_gate_analysis.py -q

Every fixture is a synthetic ``f_box_profile.json`` written to a temp dir with
closed-form per-window PSNRs, so each pooled endpoint, each contrast and each
verdict branch has a hand-checkable expected value. The script's own pure-Python
Student-t is cross-checked against scipy where scipy is installed.
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


def write_profile(run_dir, event_values, whole_values, extra_events=None):
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
    payload["events"].update(extra_events or {})
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
    extra_events=None,
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
    write_profile(
        run_dir, event_values, [float(whole)] * N_FRAMES, extra_events=extra_events
    )
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


# ---------------------------------------------------------------------------
# 9. The default path is byte-identical to the frozen 1.0.0 implementation
#
# The reference is pinned by CONTENT: a git blob id whose sha256 is asserted
# below. A later commit of scripts/realdata_gate_analysis.py therefore cannot
# turn this test into a comparison of the new code against itself.
# ---------------------------------------------------------------------------

REFERENCE_BLOB = "8f5103ac103feadc7cf4287c5dadf917f4a25c1c"
REFERENCE_SHA256 = "a6650b607ad8dc4adcece388dfb9c58189a2fbca1587dc374d72655e35bf5e00"


def _frozen_reference(tmp_path):
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
    path = Path(tmp_path) / "reference_rga.py"
    path.write_bytes(blob)
    spec = importlib.util.spec_from_file_location("reference_rga", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"g_p1_offset": -1.0},
        {"g_p1_offset": 0.01},
        {"g_p1_offset": 0.1, "p1_jitter": BIG_JITTER},
        {"gmis_c1_offset": 0.0},
    ],
)
def test_default_path_is_byte_identical_to_the_frozen_reference(tmp_path, kwargs):
    reference = _frozen_reference(tmp_path)
    manifest_path = str(build_manifest(tmp_path / "cells", **kwargs))
    for last_wave in (False, True):
        old = reference.run(manifest_path, last_wave=last_wave)
        new = rga.run(manifest_path, last_wave=last_wave)
        assert json.dumps(new, indent=1, sort_keys=True) == json.dumps(
            old, indent=1, sort_keys=True
        )
        assert rga.markdown_report(new) == reference.markdown_report(old)


def test_the_frozen_spec_dict_itself_is_unchanged(tmp_path):
    reference = _frozen_reference(tmp_path)
    assert json.dumps(rga.SPEC, indent=1, sort_keys=True) == json.dumps(
        reference.SPEC, indent=1, sort_keys=True
    )
    assert rga.SPEC["spec_version"] == "1.0.0"


# ---------------------------------------------------------------------------
# 10. --spec: deep merge, expression resolution, sha256
# ---------------------------------------------------------------------------


def write_spec(path, **overrides):
    payload = {
        "spec_id": "test_instance",
        "A": 100,
        "B": 110,
        "CA": 40,
        "CB": 69,
        "endpoints": {
            "P1": {"event": "roi:core", "frames": ["A+1", "B-1"]},
            "P2": ["B+3", "B+10"],
            "S1": ["B+11", "B+20"],
            "H1": ["A-30", "A-3"],
            "H2": "whole_frame",
            "C1": ["CA", "CB"],
        },
    }
    payload.update(overrides)
    Path(path).write_text(json.dumps(payload))
    return str(path)


def test_spec_deep_merges_rather_than_replacing(tmp_path):
    path = write_spec(tmp_path / "spec.json", SIZING={"DELTA": 0.5})
    merged, digest = rga.load_spec(path)
    # the overridden leaf changes; its untouched siblings survive
    assert merged["SIZING"]["DELTA"] == 0.5
    assert merged["SIZING"]["K"] == rga.SPEC["SIZING"]["K"]
    assert merged["SIZING"]["N2_CAP"] == rga.SPEC["SIZING"]["N2_CAP"]
    # untouched top-level entries survive
    assert merged["HARM_MARGIN_DB"] == rga.SPEC["HARM_MARGIN_DB"]
    # the endpoint keeps its default label and role while its window moves
    assert merged["endpoints"]["P2"]["label"] == "return_clean"
    assert merged["endpoints"]["P2"]["role"] == "primary"
    # the frozen SPEC is untouched by the merge
    assert rga.SPEC["SIZING"]["DELTA"] == 0.30
    assert rga.SPEC["endpoints"]["P1"]["frames_inclusive"] == [158, 187]
    assert digest == hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert merged["spec_sha256"] == digest


def test_spec_version_bumps_only_under_spec(tmp_path):
    merged, _ = rga.load_spec(write_spec(tmp_path / "spec.json"))
    assert merged["spec_version"] == "1.1.0"
    assert rga.SPEC["spec_version"] == "1.0.0"
    pinned, _ = rga.load_spec(write_spec(tmp_path / "pinned.json", spec_version="9.9.9"))
    assert pinned["spec_version"] == "9.9.9"


def test_window_expressions_resolve_against_the_anchors(tmp_path):
    merged, _ = rga.load_spec(write_spec(tmp_path / "spec.json"))
    ep = merged["endpoints"]
    assert ep["P1"]["frames_inclusive"] == [101, 109]     # A+1 .. B-1
    assert ep["P2"]["frames_inclusive"] == [113, 120]     # B+3 .. B+10
    assert ep["S1"]["frames_inclusive"] == [121, 130]
    assert ep["H1"]["frames_inclusive"] == [70, 97]       # A-30 .. A-3
    assert ep["C1"]["frames_inclusive"] == [40, 69]       # CA .. CB
    assert ep["H2"]["source"] == "whole_frame"
    assert ep["P1"]["event_name"] == "roi:core"
    assert merged["unresolved_endpoints"] == []
    assert merged["anchors"] == {"A": 100, "B": 110, "CA": 40, "CB": 69}


def test_unresolved_anchors_block_the_run_instead_of_falling_back(tmp_path):
    path = write_spec(tmp_path / "spec.json", A=None, B=None)
    merged, _ = rga.load_spec(path)
    assert merged["unresolved_endpoints"] == ["H1", "P1", "P2", "S1"]
    # the default window must NOT survive as a silent fallback
    assert merged["endpoints"]["P1"]["frames_inclusive"] is None
    manifest = str(build_manifest(tmp_path / "cells"))
    with pytest.raises(ValueError, match="null anchors"):
        rga.run(manifest, spec=merged)


def test_window_edge_forms_and_their_errors():
    anchors = {"A": 100, "B": 110, "CA": 40, "CB": 69}
    assert rga._resolve_token("A+3", anchors) == 103
    assert rga._resolve_token("B-2", anchors) == 108
    assert rga._resolve_token("CA", anchors) == 40
    assert rga._resolve_token("CB+1", anchors) == 70
    assert rga._resolve_token(158, anchors) == 158
    assert rga._resolve_token("158", anchors) == 158
    assert rga._resolve_token("A", {"A": None}) is None
    with pytest.raises(ValueError, match="neither an integer nor an offset"):
        rga._resolve_token("Q+1", anchors)


def test_endpoint_shorthands_expand():
    assert rga.normalise_endpoint(["B+3", "B+10"]) == {
        "frames": ["B+3", "B+10"], "source": "event"
    }
    assert rga.normalise_endpoint("whole_frame")["source"] == "whole_frame"
    out = rga.normalise_endpoint({"event": "roi:core", "frames": ["A", "B"]})
    assert out["event_name"] == "roi:core" and out["source"] == "event"
    with pytest.raises(ValueError, match="exactly two entries"):
        rga.normalise_endpoint([1, 2, 3])


def test_the_shipped_spec_instance_is_frozen_and_loadable():
    path = REPO_ROOT / "configs" / "n3v" / "absfix_gate_spec_v1.json"
    merged, digest = rga.load_spec(str(path))
    assert merged["spec_version"] == "1.2.0"
    assert merged["PAIRED_MIN_EFFECT_DB"] == 0.5
    assert merged["arms"] == ["U", "G", "GEST", "GMIS", "GWRONGMEM", "GONES"]
    assert merged["endpoints"]["P1"]["event_name"] == "roi:core"
    # v1.2.0 froze the anchors, so every relative window resolves
    assert {k: merged["anchors"][k] for k in ("A", "B", "CA", "CB")} == {
        "A": 60, "B": 89, "CA": 230, "CB": 259}
    assert merged["unresolved_endpoints"] == []
    assert "PLACEBO" in merged and "diagnostic" in merged["PLACEBO"]
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    # a null anchor still leaves its windows unresolved (the placeholder path)
    blank = rga.resolve_spec(dict(merged, A=None, B=None, CA=None, CB=None))
    assert blank["unresolved_endpoints"] == ["C1", "H1", "P1", "P2", "S1"]


def test_backwards_window_is_rejected(tmp_path):
    path = write_spec(tmp_path / "spec.json", A=200)
    with pytest.raises(ValueError, match="empty range"):
        rga.load_spec(path)


# ---------------------------------------------------------------------------
# 11. Per-frame-mask (roi:) endpoints
# ---------------------------------------------------------------------------


def roi_event(values, pixels):
    """A `roi:core` event covering all 300 frames."""
    per_frame = [None] * N_FRAMES
    per_pixels = [0] * N_FRAMES
    for frame, value in values.items():
        per_frame[frame] = value
    for frame, count in pixels.items():
        per_pixels[frame] = count
    return {
        "roi:core": {
            "kind": "per_frame_mask",
            "roi_name": "core",
            "per_frame_psnr": per_frame,
            "pixels_per_frame": per_pixels,
            "n_frames_with_mask": sum(1 for p in per_pixels if p > 0),
        }
    }


def _weighted_pool(values, weights):
    mse = [10.0 ** (-v / 10.0) for v in values]
    total = sum(m * w for m, w in zip(mse, weights))
    return -10.0 * math.log10(total / sum(weights))


def test_per_frame_mask_pooling_is_pixel_weighted(tmp_path):
    values = {101: 20.0, 102: 30.0}
    pixels = {101: 900, 102: 100}
    spec_path = write_spec(tmp_path / "spec.json", B=104)   # P1 = A+1 .. B-1 = 101..103
    spec, _ = rga.load_spec(spec_path)
    spec["endpoints"]["P1"]["frames_inclusive"] = [101, 102]
    entry = make_cell(
        tmp_path, "U", 0, p1=24.0, p2=26.0, s1_tail=25.0, h1=31.0, c1=32.0,
        whole=33.0, extra_events=roi_event(values, pixels),
    )
    cell = rga.load_cell(entry, spec=spec)
    expected = _weighted_pool([20.0, 30.0], [900, 100])
    assert cell["endpoints"]["P1"] == pytest.approx(expected, abs=1e-12)
    # the unweighted pool is a materially different number, so the weighting is
    # doing real work rather than being a decorative field
    assert abs(rga.pooled_psnr([20.0, 30.0]) - expected) > 1.0
    # the other endpoints still come from the bounding-box event
    assert cell["endpoints"]["H2"] == pytest.approx(33.0, abs=1e-9)


def test_empty_mask_frames_carry_zero_weight(tmp_path):
    values = {101: 20.0, 102: None, 103: 30.0}
    pixels = {101: 500, 102: 0, 103: 500}
    spec, _ = rga.load_spec(write_spec(tmp_path / "spec.json", B=104))
    entry = make_cell(
        tmp_path, "U", 0, p1=24.0, p2=26.0, s1_tail=25.0, h1=31.0, c1=32.0,
        whole=33.0, extra_events=roi_event(values, pixels),
    )
    cell = rga.load_cell(entry, spec=spec)
    assert cell["endpoints"]["P1"] == pytest.approx(
        _weighted_pool([20.0, 30.0], [500, 500]), abs=1e-12
    )


def test_a_window_of_only_empty_masks_is_an_error(tmp_path):
    spec, _ = rga.load_spec(write_spec(tmp_path / "spec.json", B=104))
    entry = make_cell(
        tmp_path, "U", 0, p1=24.0, p2=26.0, s1_tail=25.0, h1=31.0, c1=32.0,
        whole=33.0, extra_events=roi_event({}, {}),
    )
    with pytest.raises(ValueError, match="every mask is empty"):
        rga.load_cell(entry, spec=spec)


def test_pooled_psnr_weighted_reduces_to_the_unweighted_pool():
    vals = [24.0, 27.5, 31.0, 22.25]
    assert rga.pooled_psnr_weighted(vals, [7.0] * 4) == pytest.approx(
        rga.pooled_psnr(vals), abs=1e-12
    )


def test_a_missing_roi_event_is_named_in_the_error(tmp_path):
    spec, _ = rga.load_spec(write_spec(tmp_path / "spec.json"))
    entry = make_cell(tmp_path, "U", 0, p1=24.0, p2=26.0, s1_tail=25.0, h1=31.0,
                      c1=32.0, whole=33.0)
    with pytest.raises(ValueError, match="roi:core"):
        rga.load_cell(entry, spec=spec)


# ---------------------------------------------------------------------------
# 12. --paired: within-prefix contrasts
# ---------------------------------------------------------------------------


def build_paired_manifest(tmp_path, p1_values, name="paired.json"):
    """p1_values: {arm: {prefix: P1}}; every other endpoint is held constant."""
    root = Path(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    cells = []
    for arm, per_prefix in p1_values.items():
        for prefix, p1 in sorted(per_prefix.items()):
            entry = make_cell(
                root, arm, prefix, p1=p1, p2=26.0, s1_tail=25.0, h1=31.0,
                c1=32.0, whole=33.0,
                precondition=None if arm == "U" else GOOD_PRECONDITION,
            )
            entry["prefix"] = prefix
            cells.append(entry)
    path = root / name
    path.write_text(json.dumps({"cells": cells}))
    return str(path)


U_BY_PREFIX = {0: 24.0, 1: 24.5, 2: 23.5, 3: 24.2}


def _shift(base, delta):
    return {p: v + delta for p, v in base.items()}


def paired_case(tmp_path, *, g=1.0, gones=0.1, gmis=0.05, gwrong=-0.5, gest=None,
                prefixes=None, name="paired.json"):
    base = U_BY_PREFIX if prefixes is None else {p: U_BY_PREFIX[p] for p in prefixes}
    values = {"U": dict(base), "G": _shift(base, g)}
    if gones is not None:
        values["GONES"] = _shift(base, gones)
    if gmis is not None:
        values["GMIS"] = _shift(base, gmis)
    if gwrong is not None:
        values["GWRONGMEM"] = _shift(base, gwrong)
    if gest is not None:
        values["GEST"] = _shift(base, gest)
    return build_paired_manifest(tmp_path, values, name=name)


def test_paired_contrasts_are_within_prefix(tmp_path):
    report = rga.run(paired_case(tmp_path), paired=True)
    block = report["analysis"]["itt"]
    assert report["paired"] is True
    assert block["kind"] == "paired"
    assert block["prefixes"] == [0, 1, 2, 3]
    assert block["complete_pairs"] == [0, 1, 2, 3]
    gu = block["contrasts"]["G-U"]["P1"]
    assert [p["prefix"] for p in gu["pairs"]] == [0, 1, 2, 3]
    for pair in gu["pairs"]:
        assert pair["diff"] == pytest.approx(1.0, abs=1e-9)
        assert pair["b"] == pytest.approx(U_BY_PREFIX[pair["prefix"]], abs=1e-9)
    assert gu["n_pairs"] == 4
    assert gu["median"] == pytest.approx(1.0, abs=1e-9)
    assert gu["min"] == pytest.approx(1.0, abs=1e-9)
    assert gu["max"] == pytest.approx(1.0, abs=1e-9)
    assert gu["n_positive"] == 4 and gu["sign_consistent"] is True
    # the contrast set is the declared one, restricted to the arms present
    assert set(block["contrasts"]) == {
        "G-U", "GMIS-U", "GWRONGMEM-U", "GONES-U", "G-GMIS", "G-GWRONGMEM"
    }


def test_paired_pairing_beats_the_unpaired_spread(tmp_path):
    """The U values differ by up to 1 dB across prefixes; the pairs do not."""
    report = rga.run(paired_case(tmp_path), paired=True)
    gu = report["analysis"]["itt"]["contrasts"]["G-U"]["P1"]
    assert gu["sd"] == pytest.approx(0.0, abs=1e-9)
    u_values = [c["endpoints"]["P1"] for c in report["cells"] if c["arm"] == "U"]
    assert float(np.std(u_values, ddof=1)) > 0.3


def test_paired_sign_consistency_counts_the_majority_sign(tmp_path):
    values = {
        "U": dict(U_BY_PREFIX),
        "G": {0: 25.0, 1: 25.5, 2: 23.0, 3: 25.2},   # +1, +1, -0.5, +1
    }
    report = rga.run(build_paired_manifest(tmp_path, values), paired=True)
    gu = report["analysis"]["itt"]["contrasts"]["G-U"]["P1"]
    assert gu["n_positive"] == 3 and gu["n_negative"] == 1
    assert gu["n_same_sign"] == 3
    assert gu["sign_consistent"] is False
    assert gu["median"] == pytest.approx(1.0, abs=1e-9)
    assert gu["min"] == pytest.approx(-0.5, abs=1e-9)


def test_paired_t_interval_matches_the_hand_formula(tmp_path):
    values = {
        "U": dict(U_BY_PREFIX),
        "G": {0: 25.0, 1: 25.3, 2: 24.7, 3: 25.4},   # +1.0, +0.8, +1.2, +1.2
    }
    report = rga.run(build_paired_manifest(tmp_path, values), paired=True)
    gu = report["analysis"]["itt"]["contrasts"]["G-U"]["P1"]
    diffs = [p["diff"] for p in gu["pairs"]]
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    se = sd / math.sqrt(len(diffs))
    tcrit = rga.t_ppf(0.9875, 3.0)
    assert gu["mean"] == pytest.approx(mean, abs=1e-9)
    assert gu["sd"] == pytest.approx(sd, abs=1e-9)
    assert gu["df"] == 3.0
    assert gu["ci_low"] == pytest.approx(mean - tcrit * se, abs=1e-9)
    assert gu["ci_high"] == pytest.approx(mean + tcrit * se, abs=1e-9)
    # below six pairs the interval is labelled descriptive, and no p-value exists
    assert gu["descriptive"] is True
    assert "p" not in gu


def test_paired_interval_is_labelled_reported_at_six_pairs():
    six = rga.paired_summary([1.0, 1.1, 0.9, 1.2, 1.0, 1.05])
    assert six["n_pairs"] == 6 and six["descriptive"] is False
    five = rga.paired_summary([1.0, 1.1, 0.9, 1.2, 1.0])
    assert five["descriptive"] is True


def test_gones_sets_the_within_prefix_replicate_floor(tmp_path):
    report = rga.run(paired_case(tmp_path, gones=0.1), paired=True)
    block = report["analysis"]["itt"]
    floor = block["replicate_floor"]["P1"]
    assert floor["contrast"] == "|U - GONES|"
    assert floor["source"] == "GONES"
    assert floor["n"] == 4
    for prefix in (0, 1, 2, 3):
        assert floor["per_prefix"][prefix] == pytest.approx(0.1, abs=1e-9)
    assert floor["median"] == pytest.approx(0.1, abs=1e-9)


def test_paired_claim_conditions_met(tmp_path):
    report = rga.run(paired_case(tmp_path), paired=True)
    block = report["analysis"]["itt"]
    verdict = block["verdicts"]["P1"]
    assert verdict["every_pair_exceeds_floor"] is True
    assert verdict["sham_contrasts_clean"] is True
    assert verdict["floor_source"] == "GONES"
    assert verdict["verdict"] == "CLAIM_CONDITIONS_MET"
    assert block["claim"]["verdict"] == "CLAIM_CONDITIONS_MET"
    assert report["headline"]["P1"]["itt"] == "CLAIM_CONDITIONS_MET"


def test_a_sham_arm_over_the_floor_denies_the_claim(tmp_path):
    report = rga.run(paired_case(tmp_path, gmis=0.8), paired=True)
    verdict = report["analysis"]["itt"]["verdicts"]["P1"]
    assert verdict["every_pair_exceeds_floor"] is True   # G still clears the floor
    assert verdict["sham_contrasts_clean"] is False
    assert verdict["sham"]["GMIS-U"]["n_exceeding"] == 4
    assert verdict["verdict"] == "NOT_MET"


def test_a_wrong_membership_arm_over_the_floor_denies_the_claim(tmp_path):
    report = rga.run(paired_case(tmp_path, gwrong=0.9), paired=True)
    verdict = report["analysis"]["itt"]["verdicts"]["P1"]
    assert verdict["sham"]["GWRONGMEM-U"]["n_exceeding"] == 4
    assert verdict["verdict"] == "NOT_MET"


def test_a_sham_arm_below_the_floor_in_the_other_direction_is_clean(tmp_path):
    # -0.9 dB is far larger than the floor in magnitude but the wrong direction
    report = rga.run(paired_case(tmp_path, gmis=-0.9), paired=True)
    verdict = report["analysis"]["itt"]["verdicts"]["P1"]
    assert verdict["sham"]["GMIS-U"]["n_exceeding"] == 0
    assert verdict["verdict"] == "CLAIM_CONDITIONS_MET"


def test_one_pair_below_the_floor_denies_the_claim(tmp_path):
    values = {
        "U": dict(U_BY_PREFIX),
        "G": {0: 25.0, 1: 25.5, 2: 24.5, 3: 24.25},   # the last pair gains 0.05
        "GONES": _shift(U_BY_PREFIX, 0.1),
    }
    report = rga.run(build_paired_manifest(tmp_path, values), paired=True)
    verdict = report["analysis"]["itt"]["verdicts"]["P1"]
    assert [p["exceeds_floor"] for p in verdict["per_pair"]] == [True, True, True, False]
    assert verdict["every_pair_exceeds_floor"] is False
    assert verdict["verdict"] == "NOT_MET"


def test_without_gones_the_floor_falls_back_to_the_frozen_minimum(tmp_path):
    assert rga.PAIRED_MIN_EFFECT_DB == 0.5
    small = rga.run(paired_case(tmp_path / "a", g=0.3, gones=None), paired=True)
    verdict = small["analysis"]["itt"]["verdicts"]["P1"]
    assert verdict["floor_source"] == "PAIRED_MIN_EFFECT_DB"
    assert all(p["floor"] == 0.5 for p in verdict["per_pair"])
    assert verdict["verdict"] == "NOT_MET"
    big = rga.run(paired_case(tmp_path / "b", g=1.0, gones=None), paired=True)
    assert big["analysis"]["itt"]["verdicts"]["P1"]["verdict"] == "CLAIM_CONDITIONS_MET"


def test_fewer_than_three_pairs_is_design_without_power(tmp_path):
    report = rga.run(paired_case(tmp_path, prefixes=[0, 1]), paired=True)
    block = report["analysis"]["itt"]
    assert block["n_complete_pairs"] == 2
    assert block["verdicts"]["P1"]["verdict"] == "DESIGN_WITHOUT_POWER"
    # exactly three pairs is enough
    three = rga.run(paired_case(tmp_path / "three", prefixes=[0, 1, 2]), paired=True)
    assert three["analysis"]["itt"]["verdicts"]["P1"]["verdict"] == "CLAIM_CONDITIONS_MET"


def test_an_arm_missing_at_one_prefix_drops_only_that_pair(tmp_path):
    manifest = paired_case(tmp_path)
    payload = json.loads(Path(manifest).read_text())
    payload["cells"] = [
        c for c in payload["cells"] if not (c["arm"] == "G" and c["prefix"] == 2)
    ]
    Path(manifest).write_text(json.dumps(payload))
    block = rga.run(manifest, paired=True)["analysis"]["itt"]
    assert block["complete_pairs"] == [0, 1, 3]
    assert block["contrasts"]["G-U"]["P1"]["n_pairs"] == 3
    assert block["contrasts"]["GMIS-U"]["P1"]["n_pairs"] == 4


def test_paired_mode_requires_a_prefix_on_every_cell(tmp_path):
    manifest = str(build_manifest(tmp_path))     # the unpaired 3-arm fixture
    with pytest.raises(ValueError, match="no 'prefix'"):
        rga.run(manifest, paired=True)


def test_paired_mode_accepts_the_extended_arm_set(tmp_path):
    report = rga.run(paired_case(tmp_path, gest=0.4), paired=True)
    block = report["analysis"]["itt"]
    assert set(block["arms_present"]) == {"U", "G", "GEST", "GMIS", "GWRONGMEM", "GONES"}
    assert block["contrasts"]["GEST-U"]["P1"]["median"] == pytest.approx(0.4, abs=1e-9)
    assert block["contrasts"]["G-GEST"]["P1"]["median"] == pytest.approx(0.6, abs=1e-9)
    # the unpaired analysis still refuses an arm it does not know
    with pytest.raises(ValueError, match="unknown arm"):
        rga.run(paired_case(tmp_path / "unpaired", gest=0.4, name="m.json"))


def test_paired_sizing_uses_the_paired_sd(tmp_path):
    values = {
        "U": dict(U_BY_PREFIX),
        "G": {0: 25.0, 1: 25.3, 2: 24.7, 3: 25.4},
    }
    report = rga.run(build_paired_manifest(tmp_path, values), paired=True)
    block = report["analysis"]["itt"]
    sd = block["contrasts"]["G-U"]["P1"]["sd"]
    item = block["sizing"]["per_endpoint"]["P1"]
    assert item["sd_paired"] == pytest.approx(sd, abs=1e-12)
    assert item["delta_db"] == 0.30
    assert item["n2_pairs"] == math.ceil(9.5049 * (sd / 0.30) ** 2 + 2.2414 ** 2 / 4)
    # the paired rule takes ONE sample of differences, so it is the unpaired
    # per-arm formula without the factor of two
    assert rga.paired_n2(0.4945) == math.ceil(
        9.5049 * (0.4945 / 0.30) ** 2 + 2.2414 ** 2 / 4
    )
    assert rga.paired_n2(None) is None


def test_paired_mode_reports_the_placebo_as_a_diagnostic_only(tmp_path):
    block = rga.run(paired_case(tmp_path), paired=True)["analysis"]["itt"]
    assert "diagnostic" in block["placebo_role"]
    assert block["placebo"]["P1"]["n_splits"] == 3
    # the placebo is not an input to any verdict, and no verdict carries a test
    assert not any("placebo" in k for k in block["verdicts"]["P1"])
    # the prose rule is exempt: it says what the paired mode does NOT do
    dumped = json.dumps(
        {k: {f: v for f, v in item.items() if f != "rule"}
         for k, item in block["verdicts"].items()}
    )
    for banned in ("placebo", "tost", "p_value", "p_lower", "equivalen"):
        assert banned not in dumped.lower()
    assert "equivalence" not in block and "positive_control" not in block


def test_paired_mode_emits_no_equivalence_or_p_value_language(tmp_path):
    report = rga.run(paired_case(tmp_path), paired=True)
    md = rga.markdown_report(report)
    assert "PAIRED" in md and "DESCRIPTIVE" in md
    assert "| contrast | endpoint | n pairs |" in md
    assert "CLAIM_CONDITIONS_MET" in md
    for banned in ("TOST", "Equivalence", "p_lower", "p_upper"):
        assert banned not in md
    assert "tost" not in json.dumps(report["analysis"]["itt"]).lower()


def test_paired_mode_keeps_the_two_analysis_sets(tmp_path):
    manifest = paired_case(tmp_path)
    payload = json.loads(Path(manifest).read_text())
    for cell in payload["cells"]:
        if cell["arm"] == "G" and cell["prefix"] == 3:
            bad = dict(GOOD_PRECONDITION, gated_rows_final=10)
            (Path(cell["run_dir"]) / "precondition.json").write_text(json.dumps(bad))
    report = rga.run(manifest, paired=True)
    assert report["analysis"]["itt"]["n_complete_pairs"] == 4
    assert report["analysis"]["mechanism_exercised"]["n_complete_pairs"] == 3
    assert report["headline"]["P1"]["mechanism_exercised"] == "CLAIM_CONDITIONS_MET"


# ---------------------------------------------------------------------------
# 13. CLI surface for the two new flags
# ---------------------------------------------------------------------------


def test_print_spec_with_spec_prints_the_merged_resolved_spec(tmp_path, capsys):
    path = write_spec(tmp_path / "spec.json")
    assert rga.main(["--print-spec", "--spec", path]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["spec_version"] == "1.1.0"
    assert printed["spec_id"] == "test_instance"
    assert printed["endpoints"]["P1"]["frames_inclusive"] == [101, 109]
    assert printed["endpoints"]["P1"]["event_name"] == "roi:core"
    assert printed["spec_sha256"] == hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert printed["spec_file"] == str(Path(path).resolve())
    # the frozen default is still what --print-spec alone prints
    assert rga.main(["--print-spec"]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(
        json.dumps(rga.SPEC, sort_keys=True)
    )


def test_cli_paired_writes_json_and_a_markdown_table(tmp_path, capsys):
    manifest = paired_case(tmp_path)
    out = tmp_path / "paired.json"
    assert rga.main(["--manifest", manifest, "--paired", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "PAIRED" in printed and "| prefix | arm | seed |" in printed
    saved = json.loads(out.read_text())
    assert saved["paired"] is True
    assert saved["spec"]["spec_version"] == "1.0.0"     # no --spec was given
    assert saved["headline"]["P1"]["itt"] == "CLAIM_CONDITIONS_MET"
    assert [c["prefix"] for c in saved["cells"] if c["arm"] == "U"] == [0, 1, 2, 3]


def test_the_record_carries_the_merged_spec_and_its_sha256(tmp_path):
    path = write_spec(tmp_path / "spec.json")
    spec, digest = rga.load_spec(path)
    spec["endpoints"]["P1"] = dict(rga.SPEC["endpoints"]["P1"])   # back to the bbox event
    manifest = str(build_manifest(tmp_path / "cells"))
    report = rga.run(manifest, spec=spec)
    assert report["spec"]["spec_version"] == "1.1.0"
    assert report["spec_sha256"] == digest
    assert report["spec_file"] == str(Path(path).resolve())
    assert report["spec"]["endpoints"]["C1"]["frames_inclusive"] == [40, 69]
    # a default run carries neither key
    plain = rga.run(manifest)
    assert "spec_sha256" not in plain and "paired" not in plain


# ---------------------------------------------------------------------------
# 13. spec keys added for the absence fixture (defaults above stay untouched)
# ---------------------------------------------------------------------------


def _fixture_spec(tmp_path, **extra):
    """A paired spec instance on the default windows with the fixture's keys."""
    payload = {
        "spec_id": "fixture_test",
        "PAIRED_FLOOR_SOURCE": "fixed",
        "PAIRED_MIN_PAIRS": 4,
        "PAIRED_REQUIRE_SHAMS": True,
        "PAIRED_REQUIRE_GEST": True,
        "PAIRED_CLAIM_ENDPOINT_ONLY": True,
        "PAIRED_CLAIM_SET": "mechanism_exercised",
        "PAIRED_SIZING_STATUS": "informational",
    }
    payload.update(extra)
    path = Path(tmp_path) / "fixture_spec.json"
    path.write_text(json.dumps(payload))
    return rga.load_spec(str(path))[0]


def test_fixed_floor_ignores_the_gones_spread(tmp_path):
    # GONES sits 0.9 dB from U (a code-path cost); under a fixed floor the claim
    # still reads against 0.5 dB and GONES is only reported descriptively
    spec = _fixture_spec(tmp_path)
    report = rga.run(paired_case(tmp_path, g=1.0, gones=-0.9, gest=0.9),
                     spec=spec, paired=True)
    block = report["analysis"]["mechanism_exercised"]
    verdict = block["verdicts"]["P1"]
    assert verdict["verdict"] == "CLAIM_CONDITIONS_MET"
    assert all(p["floor"] == 0.5 and p["floor_source"] == "PAIRED_MIN_EFFECT_DB"
               for p in verdict["per_pair"])
    assert block["replicate_floor"]["P1"]["claim_floor"] == "fixed"
    assert block["descriptive_floors"]["GONES"]["P1"]["median"] == pytest.approx(0.9)
    assert block["descriptive_floors"]["GMIS"]["P1"]["median"] == pytest.approx(0.05)
    assert report["operative_set"] == "mechanism_exercised"
    assert report["headline"]["P1"]["operative"] == "CLAIM_CONDITIONS_MET"
    # only P1 carries a verdict; sizing is informational and P1-only
    assert block["verdicts"]["P2"]["verdict"] == "DESCRIPTIVE_ONLY"
    assert block["sizing"]["status"] == "informational"
    assert list(block["sizing"]["per_endpoint"]) == ["P1"]


def test_missing_sham_or_gest_pairs_deny_power(tmp_path):
    spec = _fixture_spec(tmp_path)
    # no GEST arm at all
    report = rga.run(paired_case(tmp_path, g=1.0), spec=spec, paired=True)
    verdict = report["analysis"]["mechanism_exercised"]["verdicts"]["P1"]
    assert verdict["verdict"] == "DESIGN_WITHOUT_POWER"
    assert verdict["missing_control_pairs"] == {"GEST-U": [0, 1, 2, 3]}
    # GMIS present on three prefixes only
    values = {
        "U": dict(U_BY_PREFIX), "G": _shift(U_BY_PREFIX, 1.0),
        "GEST": _shift(U_BY_PREFIX, 0.9), "GWRONGMEM": _shift(U_BY_PREFIX, -0.5),
        "GONES": _shift(U_BY_PREFIX, 0.1),
        "GMIS": {p: v + 0.05 for p, v in U_BY_PREFIX.items() if p != 3},
    }
    report = rga.run(build_paired_manifest(tmp_path, values, name="m2.json"),
                     spec=spec, paired=True)
    verdict = report["analysis"]["mechanism_exercised"]["verdicts"]["P1"]
    assert verdict["verdict"] == "DESIGN_WITHOUT_POWER"
    assert verdict["missing_control_pairs"] == {"GMIS-U": [3]}
    # three complete pairs are below PAIRED_MIN_PAIRS = 4
    report = rga.run(paired_case(tmp_path, g=1.0, gest=0.9, prefixes=[0, 1, 2],
                                 name="m3.json"), spec=spec, paired=True)
    assert report["analysis"]["mechanism_exercised"]["verdicts"]["P1"]["verdict"] \
        == "DESIGN_WITHOUT_POWER"


def test_operative_verdict_reads_the_mechanism_set_not_itt(tmp_path):
    spec = _fixture_spec(tmp_path)
    manifest = paired_case(tmp_path, g=1.0, gest=0.9)
    cells = json.loads(Path(manifest).read_text())["cells"]
    # strip the precondition from one G cell: ITT still "met", the operative set is not
    for c in cells:
        if c["arm"] == "G" and c["prefix"] == 2:
            (Path(c["run_dir"]) / rga.SPEC["precondition_filename"]).unlink()
    report = rga.run(manifest, spec=spec, paired=True)
    assert report["headline"]["P1"]["itt"] == "CLAIM_CONDITIONS_MET"
    assert report["headline"]["P1"]["operative"] == "DESIGN_WITHOUT_POWER"


def test_require_all_frames_refuses_a_gap_in_the_core_masks(tmp_path):
    spec, _ = rga.load_spec(write_spec(
        tmp_path / "spec.json", B=104, PER_FRAME_MASK_REQUIRE_ALL_FRAMES=True))
    entry = make_cell(
        tmp_path, "U", 0, p1=24.0, p2=26.0, s1_tail=25.0, h1=31.0, c1=32.0,
        whole=33.0, extra_events=roi_event({101: 20.0, 103: 30.0},
                                           {101: 500, 102: 0, 103: 500}),
    )
    with pytest.raises(ValueError, match=r"frames \[102\]"):
        rga.load_cell(entry, spec=spec)
    # the default keeps the drop-empty-frames rule
    plain, _ = rga.load_spec(write_spec(tmp_path / "spec2.json", B=104))
    assert rga.load_cell(entry, spec=plain)["endpoints"]["P1"] == pytest.approx(
        _weighted_pool([20.0, 30.0], [500, 500]), abs=1e-12)


def test_per_frame_mask_diagnostics_count_frames_and_infinities(tmp_path):
    spec, _ = rga.load_spec(write_spec(tmp_path / "spec.json", B=105))  # P1 = 101..104
    pixels = {f: 100 for f in range(101, 105)}
    u = make_cell(tmp_path, "U", 0, p1=24.0, p2=26.0, s1_tail=25.0, h1=31.0, c1=32.0,
                  whole=33.0, extra_events=roi_event(
                      {101: 20.0, 102: 20.0, 103: 20.0, 104: 40.0}, pixels))
    g = make_cell(tmp_path, "G", 0, p1=24.0, p2=26.0, s1_tail=25.0, h1=31.0, c1=32.0,
                  whole=33.0, precondition=GOOD_PRECONDITION, extra_events=roi_event(
                      {101: float("inf"), 102: float("inf"), 103: 30.0, 104: 35.0}, pixels))
    cells = []
    for entry in (u, g):
        entry["prefix"] = 0
        cells.append(rga.load_cell(entry, spec=spec))
    diag = rga.per_frame_mask_diagnostics(cells, "P1", spec=spec)
    pair = diag["contrasts"]["G-U"][0]
    assert pair["n_frames"] == 4
    assert pair["frames_a_lower_mse"] == 3 and pair["fraction_a_lower_mse"] == 0.75
    assert pair["n_inf_a"] == 2 and pair["n_inf_b"] == 0
    assert pair["max_psnr_b"] == 40.0 and math.isinf(pair["max_psnr_a"])
    assert pair["pooled_mse_diff_a_minus_b"] < 0


def test_mechanism_window_and_box_checks(tmp_path):
    base = dict(GOOD_PRECONDITION)
    spec = _fixture_spec(tmp_path, MECHANISM_ZERO_FRAMES_WINDOW=[63, 87],
                         MECHANISM_ZERO_FRAMES_EXEMPT_ARMS=["GMIS", "GONES"],
                         MECHANISM_FBOX=[964, 748, 1034, 952], MECHANISM_FBOX_FRAME=75)
    # without the detail block the list is absent -> fails closed
    ok, reason = rga.mechanism_exercised("G", base, spec=spec)
    assert not ok and "zero-presence frame list" in reason
    detailed = dict(base, frames_presence_zero_list=list(range(230, 257)),
                    fbox_frame=75, fbox_box=[964, 748, 1034, 952])
    ok, reason = rga.mechanism_exercised("G", detailed, spec=spec)
    assert not ok and "inside [63, 87]: 0" in reason
    ok, reason = rga.mechanism_exercised("GMIS", detailed, spec=spec)   # exempt
    assert ok
    good = dict(detailed, frames_presence_zero_list=list(range(63, 90)))
    assert rga.mechanism_exercised("G", good, spec=spec)[0]
    ok, reason = rga.mechanism_exercised("GWRONGMEM", good, spec=spec)
    assert ok and "code path" in reason
    wrong_frame = dict(good, fbox_frame=150)
    ok, reason = rga.mechanism_exercised("G", wrong_frame, spec=spec)
    assert not ok and "frame 150" in reason
    wrong_box = dict(good, fbox_box=[664, 912, 744, 976])
    assert not rga.mechanism_exercised("G", wrong_box, spec=spec)[0]


def test_read_precondition_lifts_the_detail_fields(tmp_path):
    run_dir = Path(tmp_path) / "cell"
    run_dir.mkdir()
    payload = dict(GOOD_PRECONDITION, arm_kind="elgs", detail={
        "presence": {"frames_presence_zero": [63, 64, 65]},
        "fbox": {"frame": 75, "box_x0_y0_x1_y1_inclusive": [1, 2, 3, 4]},
    })
    (run_dir / rga.SPEC["precondition_filename"]).write_text(json.dumps(payload))
    pre = rga.read_precondition(str(run_dir))
    assert pre["frames_presence_zero_list"] == [63, 64, 65]
    assert pre["fbox_frame"] == 75 and pre["fbox_box"] == [1, 2, 3, 4]
    assert pre["arm_kind"] == "elgs"


def test_reserved_units_required_blocks_on_a_silent_cell(tmp_path):
    spec = _fixture_spec(tmp_path, RESERVED_UNITS_REQUIRED=True)
    report = rga.run(paired_case(tmp_path, g=1.0, gest=0.9), spec=spec, paired=True)
    # U cells carry no precondition file in the synthetic manifest
    assert any("RESERVED_UNITS_REQUIRED" in msg for msg in report["blocking_errors"])
    assert report["headline"]["P1"]["operative"] == "BLOCKED"


def test_the_shipped_fixture_spec_is_frozen_and_resolves():
    spec, digest = rga.load_spec("configs/n3v/absfix_gate_spec_v1.json")
    assert spec["spec_version"] == "1.2.0"
    assert spec["unresolved_endpoints"] == []
    assert spec["endpoints"]["P1"]["frames_inclusive"] == [63, 87]
    assert spec["endpoints"]["P2"]["frames_inclusive"] == [92, 99]
    assert spec["endpoints"]["S1"]["frames_inclusive"] == [100, 109]
    assert spec["endpoints"]["H1"]["frames_inclusive"] == [30, 57]
    assert spec["endpoints"]["C1"]["frames_inclusive"] == [230, 259]
    assert spec["event_name"] == "BOTTLE_absence_gap"
    assert spec["PAIRED_FLOOR_SOURCE"] == "fixed" and spec["PAIRED_MIN_PAIRS"] == 4
    assert spec["PAIRED_CLAIM_SET"] == "mechanism_exercised"
    assert spec["MECHANISM_FBOX"] == [964, 748, 1034, 952]


def test_paired_markdown_renders_under_the_fixture_keys(tmp_path, capsys):
    spec_path = Path(tmp_path) / "fixture_spec.json"
    spec_path.write_text(json.dumps({
        "spec_id": "fixture_md", "PAIRED_FLOOR_SOURCE": "fixed",
        "PAIRED_MIN_PAIRS": 4, "PAIRED_REQUIRE_SHAMS": True,
        "PAIRED_REQUIRE_GEST": True, "PAIRED_CLAIM_ENDPOINT_ONLY": True,
        "PAIRED_CLAIM_SET": "mechanism_exercised",
    }))
    manifest = paired_case(tmp_path, g=1.0, gest=0.9)
    out = Path(tmp_path) / "out.json"
    rc = rga.main(["--manifest", manifest, "--spec", str(spec_path), "--paired",
                   "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0
    assert "| P2 | - | - | **DESCRIPTIVE_ONLY** |" in text
    assert "descriptive floors" in text
    assert "OPERATIVE (mechanism_exercised) **CLAIM_CONDITIONS_MET**" in text
