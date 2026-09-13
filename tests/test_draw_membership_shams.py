"""CPU tests for scripts/draw_membership_shams.py.

A synthetic 400-row cloud stands in for a prefix: 40 rows are the truth set,
another 60 are locally eligible, and the contribution column is chosen so the
L draw's +-10% band is reachable in one case and unreachable in another.
Nothing here loads torch, a checkpoint or a scene.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import draw_membership_shams as dms  # noqa: E402

N_ROWS = 400
TRUTH_ROWS = list(range(0, 40))
LOCAL_ROWS = list(range(100, 160))
GAP = (60, 89)


def truth_program(n_rows=N_ROWS, truth_rows=TRUTH_ROWS, group=1):
    column = [-1] * n_rows
    for row in truth_rows:
        column[row] = group
    return {
        "schema_version": "adags-episode-program-v2",
        "units": "model_time_seconds",
        "membership_mode": "row_ids",
        "frame_dt": 0.0333333,
        "presence_edge_half_width_w": 0.0083,
        "cloud": {"n_rows": n_rows, "xyz_sha256": "a" * 64},
        "spatial": {"kind": "voxel_grid", "cells_per_axis": 16,
                    "lo": [0.0, 0.0, 0.0], "span": [1.0, 1.0, 1.0],
                    "group_cell_keys": {"1": [7]}},
        "groups": [{"group": group, "gaps": [[1.9, 2.96]],
                    "offset_frame": GAP[0], "onset_frame": GAP[1] + 1,
                    "rows_at_estimation": len(truth_rows)}],
        "source": {"instrument": "test"},
        "row_group_ids": column,
    }


def contribution(local_scale=1.0, n_rows=N_ROWS):
    """Truth rows carry 1.0 each; local rows carry `local_scale` each."""
    values = np.zeros(n_rows, dtype=np.float64)
    values[TRUTH_ROWS] = 1.0
    values[LOCAL_ROWS] = float(local_scale)
    return values


def eligible(n_rows=N_ROWS, rows=None):
    mask = np.zeros(n_rows, dtype=bool)
    mask[LOCAL_ROWS if rows is None else rows] = True
    return mask


def draws(**kwargs):
    kwargs.setdefault("truth_program", truth_program())
    kwargs.setdefault("contribution", contribution())
    kwargs.setdefault("eligible", eligible())
    kwargs.setdefault("prefix_seed", 0)
    return dms.draw_all(
        kwargs["truth_program"], kwargs["contribution"], kwargs["eligible"],
        kwargs["prefix_seed"],
        draw_index=kwargs.get("draw_index", dms.DEFAULT_DRAW_INDEX),
        tolerance=kwargs.get("tolerance", dms.DEFAULT_MASS_TOLERANCE),
    )


def rows_of(program):
    column = np.asarray(program["row_group_ids"])
    return np.flatnonzero(column >= 0)


# ---------------------------------------------------------------------------
# Zero overlap and count match
# ---------------------------------------------------------------------------


def test_every_draw_has_zero_overlap_with_the_truth_set():
    for arm, (program, _, counts) in draws().items():
        assert counts["overlap_n"] == 0, arm
        assert not set(rows_of(program)) & set(TRUTH_ROWS), arm


@pytest.mark.parametrize("arm", [dms.DRAW_A, dms.DRAW_B])
def test_the_uniform_draws_are_count_matched(arm):
    program, _, counts = draws()[arm]
    assert counts["draw_n"] == counts["truth_n"] == len(TRUTH_ROWS)
    assert len(rows_of(program)) == len(TRUTH_ROWS)


def test_the_two_uniform_draws_are_independent():
    out = draws()
    a = set(rows_of(out[dms.DRAW_A][0]))
    b = set(rows_of(out[dms.DRAW_B][0]))
    assert a != b
    assert out[dms.DRAW_A][2]["seed"] != out[dms.DRAW_B][2]["seed"]


def test_the_seeds_follow_the_spec_rule():
    out = draws(prefix_seed=3)
    assert out[dms.DRAW_A][2]["seed"] == 1003
    assert out[dms.DRAW_B][2]["seed"] == 2003
    assert out[dms.DRAW_L][2]["seed"] == 1003


def test_a_cloud_with_too_few_rows_outside_the_truth_set_is_refused():
    program = truth_program(n_rows=60, truth_rows=list(range(0, 45)))
    with pytest.raises(dms.DrawRefused, match="count-matched draw needs"):
        dms.draw_all(program, np.ones(60), np.zeros(60, dtype=bool), 0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_rows():
    first = draws(prefix_seed=7)
    second = draws(prefix_seed=7)
    for arm in dms.DRAWS:
        assert list(rows_of(first[arm][0])) == list(rows_of(second[arm][0]))
        assert first[arm][2]["row_ids_sha256"] == second[arm][2]["row_ids_sha256"]
        assert first[arm][1] == second[arm][1]


def test_a_different_prefix_seed_gives_different_rows():
    a = draws(prefix_seed=0)[dms.DRAW_A]
    b = draws(prefix_seed=1)[dms.DRAW_A]
    assert list(rows_of(a[0])) != list(rows_of(b[0]))


def test_the_local_draw_does_not_depend_on_the_seed():
    """L is greedy, not random; its seed is recorded, never consumed."""
    a = draws(prefix_seed=0)[dms.DRAW_L]
    b = draws(prefix_seed=11)[dms.DRAW_L]
    assert list(rows_of(a[0])) == list(rows_of(b[0]))


# ---------------------------------------------------------------------------
# The local contribution-matched draw
# ---------------------------------------------------------------------------


def test_the_local_draw_lands_inside_the_mass_band():
    program, _, counts = draws()[dms.DRAW_L]
    target = counts["contribution_truth"]
    assert 0.9 * target <= counts["contribution_draw"] <= 1.1 * target
    assert set(rows_of(program)) <= set(LOCAL_ROWS)


def test_the_local_draw_only_takes_locally_eligible_rows():
    program, _, _ = draws()[dms.DRAW_L]
    assert set(rows_of(program)) <= set(LOCAL_ROWS)


def test_heavier_local_rows_mean_fewer_of_them():
    """Matching mass, not count: 2x the contribution needs half the rows."""
    light = draws(contribution=contribution(1.0))[dms.DRAW_L]
    heavy = draws(contribution=contribution(2.0))[dms.DRAW_L]
    assert len(rows_of(heavy[0])) < len(rows_of(light[0]))
    for out in (light, heavy):
        assert 0.9 <= out[2]["contribution_draw"] / out[2]["contribution_truth"] \
            <= 1.1


def test_the_draw_is_greedy_by_descending_contribution():
    values = contribution()
    for i, row in enumerate(LOCAL_ROWS):
        values[row] = 1.0 + 0.01 * i          # strictly increasing along the list
    program, _, _ = draws(contribution=values)[dms.DRAW_L]
    taken = set(rows_of(program))
    # the walk must be a suffix of LOCAL_ROWS ordered by weight, i.e. the
    # heaviest rows are taken and the lightest are left
    heaviest = LOCAL_ROWS[len(LOCAL_ROWS) - len(taken):]
    assert taken == set(heaviest)
    assert LOCAL_ROWS[0] not in taken and LOCAL_ROWS[-1] in taken


def test_an_unreachable_band_is_refused():
    """40 eligible rows of 0.1 carry 4.0 against a target of 40."""
    values = contribution(0.1)
    with pytest.raises(dms.DrawRefused, match="unreachable"):
        dms.draw_all(truth_program(), values, eligible(), 0)


def test_no_eligible_row_is_refused():
    with pytest.raises(dms.DrawRefused, match="locally eligible"):
        dms.draw_all(truth_program(), contribution(),
                     np.zeros(N_ROWS, dtype=bool), 0)


def test_a_truth_set_with_no_contribution_mass_is_refused():
    values = np.zeros(N_ROWS)
    values[LOCAL_ROWS] = 1.0
    with pytest.raises(dms.DrawRefused, match="no contribution mass"):
        dms.draw_all(truth_program(), values, eligible(), 0)


def test_one_heavy_row_is_skipped_rather_than_stranding_the_draw():
    """A row above the upper edge must not end the walk."""
    values = contribution()
    values[LOCAL_ROWS[0]] = 1000.0
    program, _, counts = draws(contribution=values)[dms.DRAW_L]
    assert LOCAL_ROWS[0] not in set(rows_of(program))
    assert counts["contribution_draw"] <= 1.1 * counts["contribution_truth"]


def test_a_tighter_tolerance_still_lands_inside_its_own_band():
    _, _, counts = draws(tolerance=0.02)[dms.DRAW_L]
    target = counts["contribution_truth"]
    assert 0.98 * target <= counts["contribution_draw"] <= 1.02 * target


# ---------------------------------------------------------------------------
# The sidecar and the emitted program
# ---------------------------------------------------------------------------


def test_the_sidecar_counts_are_integers():
    for arm, (_, _, counts) in draws().items():
        for key in ("truth_n", "draw_n", "overlap_n", "eligible_n", "seed"):
            assert isinstance(counts[key], int), (arm, key)
            assert not isinstance(counts[key], bool), (arm, key)
        assert isinstance(counts["contribution_truth"], float)
        assert isinstance(counts["contribution_draw"], float)
        assert len(counts["row_ids_sha256"]) == 64


def test_the_sham_program_differs_from_the_truth_program_only_in_its_rows():
    base = truth_program()
    program, _, _ = dms.draw_all(
        base, contribution(), eligible(), 0)[dms.DRAW_A]
    for key in sorted(set(base) | set(program)):
        if key in ("row_group_ids", "groups", "source"):
            continue
        assert program[key] == base[key], key
    assert program["groups"][0]["gaps"] == base["groups"][0]["gaps"]
    assert program["groups"][0]["offset_frame"] == GAP[0]
    assert program["groups"][0]["onset_frame"] == GAP[1] + 1
    assert program["groups"][0]["rows_at_estimation"] == len(TRUTH_ROWS)
    assert program["source"]["instrument"] == "test"
    assert program["source"]["wrongmem"]["arm"] == dms.DRAW_A


def test_the_row_column_uses_the_truth_programs_group_id():
    base = truth_program(group=4)
    program, _, _ = dms.draw_all(
        base, contribution(), eligible(), 0)[dms.DRAW_L]
    column = np.asarray(program["row_group_ids"])
    assert set(column[column >= 0].tolist()) == {4}


# ---------------------------------------------------------------------------
# Input reading and the CLI
# ---------------------------------------------------------------------------


def test_a_program_that_is_not_row_ids_is_refused(tmp_path):
    payload = truth_program()
    payload["membership_mode"] = "spatial_voxel"
    path = tmp_path / "p.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(dms.DrawRefused, match="membership_mode"):
        dms.load_program(str(path))


def test_a_mismatched_row_count_is_refused(tmp_path):
    payload = truth_program()
    payload["cloud"]["n_rows"] = 399
    path = tmp_path / "p.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(dms.DrawRefused, match="cloud.n_rows"):
        dms.load_program(str(path))


def test_the_gap_is_checked_against_the_program(tmp_path):
    program = truth_program()
    dms.check_gap(program, GAP)
    with pytest.raises(dms.DrawRefused, match="wrong program"):
        dms.check_gap(program, (60, 91))


def test_a_negative_contribution_is_refused(tmp_path):
    path = tmp_path / "w.npz"
    values = contribution()
    values[200] = -1.0
    np.savez(path, w_in=values)
    with pytest.raises(dms.DrawRefused, match="negative"):
        dms.load_contribution(str(path), N_ROWS)


def test_a_short_contribution_column_is_refused(tmp_path):
    path = tmp_path / "w.npy"
    np.save(path, np.ones(10))
    with pytest.raises(dms.DrawRefused, match="10 entries"):
        dms.load_contribution(str(path), N_ROWS)


@pytest.mark.parametrize("suffix", [".npz", ".npy", ".json"])
def test_the_three_input_formats_agree(tmp_path, suffix):
    values = contribution()
    path = tmp_path / ("w" + suffix)
    if suffix == ".npz":
        np.savez(path, w_in=values)
    elif suffix == ".npy":
        np.save(path, values)
    else:
        path.write_text(json.dumps({"contribution": values.tolist()}),
                        encoding="utf-8")
    assert np.array_equal(dms.load_contribution(str(path), N_ROWS), values)


def test_an_npz_without_a_known_key_is_refused(tmp_path):
    path = tmp_path / "w.npz"
    np.savez(path, something_else=contribution())
    with pytest.raises(dms.DrawRefused, match="none of"):
        dms.load_contribution(str(path), N_ROWS)


def test_the_cli_writes_six_files_and_exits_zero(tmp_path, capsys):
    program_path = tmp_path / "program_oracle.json"
    program_path.write_text(json.dumps(truth_program()), encoding="utf-8")
    weights = tmp_path / "w.npz"
    np.savez(weights, w_in=contribution())
    elig = tmp_path / "e.npy"
    np.save(elig, eligible())
    out = tmp_path / "draws"
    code = dms.main([
        "--truth_program", str(program_path),
        "--contribution", str(weights),
        "--local_eligible", str(elig),
        "--out_dir", str(out),
        "--prefix_seed", "2",
        "--gap", str(GAP[0]), str(GAP[1]),
    ])
    assert code == 0
    for arm in dms.DRAWS:
        stem = arm.lower()
        counts = json.loads(
            (out / ("counts_%s.json" % stem)).read_text(encoding="utf-8"))
        assert counts["overlap_n"] == 0
        program = json.loads(
            (out / ("program_%s.json" % stem)).read_text(encoding="utf-8"))
        assert program["membership_mode"] == "row_ids"
    assert "GWRONGMEM_L" in capsys.readouterr().out


def test_the_cli_refuses_with_exit_code_two(tmp_path, capsys):
    program_path = tmp_path / "program_oracle.json"
    program_path.write_text(json.dumps(truth_program()), encoding="utf-8")
    weights = tmp_path / "w.npz"
    np.savez(weights, w_in=contribution(0.1))      # band unreachable
    elig = tmp_path / "e.npy"
    np.save(elig, eligible())
    out = tmp_path / "draws"
    code = dms.main([
        "--truth_program", str(program_path),
        "--contribution", str(weights),
        "--local_eligible", str(elig),
        "--out_dir", str(out),
        "--prefix_seed", "0",
    ])
    assert code == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not out.exists(), "a refused draw must write nothing"


def test_the_module_does_not_import_torch():
    import importlib

    before = "torch" in sys.modules
    importlib.reload(dms)
    assert ("torch" in sys.modules) == before
