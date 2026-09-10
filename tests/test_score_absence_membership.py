import json

import numpy as np
import pytest

from scripts.score_absence_membership import (ContractError, check_same_cloud,
                                              gap_frames, load_program,
                                              main, score, set_scores,
                                              temporal_iou)


def _program(ids, offset, onset, sha="abc"):
    return {
        "schema_version": "adags-episode-program-v2",
        "membership_mode": "row_ids",
        "cloud": {"n_rows": len(ids), "xyz_sha256": sha},
        "row_group_ids": list(ids),
        "groups": [{"group": 1, "offset_frame": offset, "onset_frame": onset,
                    "gaps": [[0.0, 1.0]], "rows_at_estimation": 1}],
    }


def test_set_scores_counts():
    est = [1, 1, 0, 0, 1]
    truth = [1, 0, 1, 0, 1]
    s = set_scores(est, truth)
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 1)
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(2 / 3)
    assert s["jaccard"] == pytest.approx(0.5)


def test_set_scores_weighted_and_empty():
    est = [1, 0]
    truth = [0, 0]
    s = set_scores(est, truth, weights=[0.5, 2.0])
    assert s["precision"] == 0.0 and s["recall"] is None and s["jaccard"] == 0.0
    with pytest.raises(ContractError):
        set_scores([1, 0], [1, 0], weights=[1.0])


def test_temporal_iou_inclusive():
    assert temporal_iou((60, 89), (60, 89)) == 1.0
    assert temporal_iou((59, 88), (60, 89)) == pytest.approx(29 / 31)
    assert temporal_iou((100, 120), (60, 89)) == 0.0
    assert temporal_iou(None, (60, 89)) == 0.0


def test_gap_frames_spans_groups():
    p = _program([1, -1], 60, 90)
    p["groups"].append({"group": 2, "offset_frame": 58, "onset_frame": 85})
    assert gap_frames(p) == (58, 89)
    assert gap_frames({"groups": []}) is None


def test_same_cloud_refusals():
    a = _program([1, -1], 60, 90, sha="a")
    b = _program([1, -1], 60, 90, sha="b")
    with pytest.raises(ContractError):
        check_same_cloud(a, b)
    c = _program([1, -1, -1], 60, 90, sha="a")
    with pytest.raises(ContractError):
        check_same_cloud(a, c)


def test_score_end_to_end(tmp_path):
    est = _program([1, 1, -1, -1], 59, 90)
    truth = _program([1, -1, 1, -1], 60, 90)
    r = score(est, truth, (60, 89))
    assert r["rows"]["precision"] == 0.5 and r["rows"]["recall"] == 0.5
    assert r["gap"]["estimated_absent_frames"] == [59, 89]
    assert r["gap"]["offset_error_frames"] == -1
    assert r["gap"]["temporal_iou"] == pytest.approx(30 / 31)
    assert r["gap"]["method"].startswith("envelope")
    assert r["gap"]["per_group"][0]["absent_frames"] == [59, 89]
    pe, pt, out = tmp_path / "e.json", tmp_path / "t.json", tmp_path / "s.json"
    pe.write_text(json.dumps(est))
    pt.write_text(json.dumps(truth))
    assert main(["--estimate", str(pe), "--truth", str(pt),
                 "--authored_gap", "60", "89", "--out", str(out)]) == 0
    saved = json.loads(out.read_text())
    assert saved["rows"]["tp"] == 1
    w = tmp_path / "w.npy"
    np.save(w, np.array([2.0, 1.0, 1.0, 1.0]))
    assert main(["--estimate", str(pe), "--truth", str(pt), "--weights",
                 str(w), "--authored_gap", "60", "89", "--out", str(out)]) == 0
    saved = json.loads(out.read_text())
    assert saved["weighted"]["precision"] == pytest.approx(2 / 3)


def test_load_program_refuses_spatial(tmp_path):
    p = _program([1], 60, 90)
    p["membership_mode"] = "spatial_voxel"
    f = tmp_path / "p.json"
    f.write_text(json.dumps(p))
    with pytest.raises(ContractError):
        load_program(f)
