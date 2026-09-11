"""Spec v2.0.0 reducer tests for scripts/realdata_gate_analysis.py.

Two halves.

**Half one is the regression that comes first** (section 1): the wave-1
v1.2.0 analysis, replayed from the 24 cells' real `f_box_profile.json`,
`precondition.json` and `summary.json` under `tests/fixtures/absfix_wave1/`,
must reproduce the frozen `wave1_paired.json` that
`runs/realdata/absfix/analysis_wave1/` holds on Leonardo. Byte identity of
the file itself is impossible -- the record embeds the absolute
`run_dir`/`manifest`/`spec_file` paths of the machine that produced it -- so
the comparison is of the parsed record with exactly those path fields
replaced, and nothing else. Everything numeric, every verdict and every
reason string must match.

**Half two** (sections 2 onwards) exercises the v2 reducer on synthetic
scenes, prefixes, profiles and sidecars built in the test: the claim rules of
spec v2.0.0 section 11.5, the two instrument preconditions, the zero-overlap
assertion on the sham row sets, the per-cell precondition thresholds, scene
precedence, and the fail-closed behaviour that must produce
DESIGN_WITHOUT_POWER rather than a silent skip.

The fixture profiles are the real ones, trimmed to the two events the v1.2.0
endpoints read (`BOTTLE_absence_gap` and `roi:core`) and to the precondition
and summary fields the analysis reads; no other field of those files is
touched.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import realdata_gate_analysis as rga  # noqa: E402

WAVE1 = Path(__file__).resolve().parent / "fixtures" / "absfix_wave1"
SPEC_V1 = REPO_ROOT / "configs" / "n3v" / "absfix_gate_spec_v1.json"
SPEC_V2 = REPO_ROOT / "configs" / "n3v" / "absfix_gate_spec_v2.json"

# The absolute paths the frozen record carries from the machine that wrote it.
# These, and only these, are excluded from the comparison.
PATH_FIELDS = ("manifest", "spec_file", "run_dir")


# ---------------------------------------------------------------------------
# 1. The wave-1 v1.2.0 record is reproduced from the real profiles
# ---------------------------------------------------------------------------


def _strip_paths(node):
    """A deep copy with every absolute-path field replaced by its basename."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key in PATH_FIELDS and isinstance(value, str):
                out[key] = Path(value.replace("\\", "/")).name
            else:
                out[key] = _strip_paths(value)
        return out
    if isinstance(node, list):
        return [_strip_paths(v) for v in node]
    return node


def _wave1_manifest(tmp_path):
    """The wave-1 manifest with `run_dir` pointing into the fixture tree."""
    manifest = json.loads((WAVE1 / "manifest.json").read_text(encoding="utf-8"))
    for cell in manifest["cells"]:
        cell["run_dir"] = str(WAVE1 / "cells" / cell["run_dir"])
    path = Path(tmp_path) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return str(path)


# The only slack in the comparison. Leonardo's numpy and this machine's do not
# round `np.mean` over 300 doubles to the same last bit, so a handful of pooled
# PSNRs differ in the twelfth significant figure (measured worst relative
# difference on this fixture: 5.1e-12). Every non-float value -- every verdict,
# reason string, count, prefix list and key -- is compared exactly.
FLOAT_RTOL = 1e-9


def _assert_same(new, old, path="record"):
    if isinstance(old, dict):
        assert isinstance(new, dict), path
        assert sorted(new, key=str) == sorted(old, key=str), path
        for key in sorted(old, key=str):
            _assert_same(new[key], old[key], "%s/%s" % (path, key))
    elif isinstance(old, list):
        assert isinstance(new, list) and len(new) == len(old), path
        for i, (a, b) in enumerate(zip(new, old)):
            _assert_same(a, b, "%s[%d]" % (path, i))
    elif isinstance(old, float) and isinstance(new, (int, float)):
        assert new == pytest.approx(old, rel=FLOAT_RTOL, abs=1e-12), path
    else:
        assert new == old, path


@pytest.mark.skipif(not WAVE1.is_dir(), reason="wave-1 fixture not present")
def test_wave1_v1_2_0_record_is_reproduced(tmp_path):
    """The frozen wave-1 record, recomputed from the 24 cells' own profiles.

    Compared after a json round trip, because the frozen file is the
    serialised form (its `per_prefix` keys are JSON strings, not the ints the
    in-memory record carries) and after replacing the three absolute-path
    fields, which name the machine that wrote it.
    """
    frozen = json.loads((WAVE1 / "wave1_paired.json").read_text(encoding="utf-8"))
    spec, _ = rga.load_spec(str(SPEC_V1))
    report = rga.run(_wave1_manifest(tmp_path), wave=1, spec=spec, paired=True)
    replayed = json.loads(json.dumps(report, indent=1))
    _assert_same(_strip_paths(replayed), _strip_paths(frozen))


@pytest.mark.skipif(not WAVE1.is_dir(), reason="wave-1 fixture not present")
def test_the_wave1_comparison_would_notice_a_changed_number(tmp_path):
    """The tolerance above is not wide enough to hide a real change."""
    frozen = json.loads((WAVE1 / "wave1_paired.json").read_text(encoding="utf-8"))
    spec, _ = rga.load_spec(str(SPEC_V1))
    report = json.loads(json.dumps(
        rga.run(_wave1_manifest(tmp_path), wave=1, spec=spec, paired=True)))
    report["analysis"]["itt"]["contrasts"]["G-U"]["P1"]["median"] += 1e-6
    with pytest.raises(AssertionError):
        _assert_same(_strip_paths(report), _strip_paths(frozen))


@pytest.mark.skipif(not WAVE1.is_dir(), reason="wave-1 fixture not present")
def test_wave1_record_still_reads_not_met_on_the_operative_set(tmp_path):
    """The headline of the frozen record, restated so a regression is named."""
    spec, _ = rga.load_spec(str(SPEC_V1))
    report = rga.run(_wave1_manifest(tmp_path), wave=1, spec=spec, paired=True)
    assert report["operative_set"] == "mechanism_exercised"
    assert report["headline"]["P1"]["operative"] == "NOT_MET"
    assert report["analysis"]["mechanism_exercised"]["n_complete_pairs"] == 4


def test_a_spec_without_the_v2_keys_takes_the_v1_path():
    spec, _ = rga.load_spec(str(SPEC_V1))
    assert rga.is_spec_v2(spec) is False
    assert rga.is_spec_v2(rga.SPEC) is False


def test_the_shipped_v2_spec_instance_is_loadable_and_is_v2():
    spec, _ = rga.load_spec(str(SPEC_V2))
    assert spec["spec_version"] == "2.0.0"
    assert rga.is_spec_v2(spec) is True
    assert spec["PAIRED_CLAIM_ENDPOINTS"] == ["P1", "P2"]
    assert spec["WRONGMEM_OVERLAP_MAX"] == 0
    assert "GESTMEM" in spec["arms"] and "GWRONGMEM_L" in spec["arms"]


# ---------------------------------------------------------------------------
# 2. Synthetic v2 fixture builders
# ---------------------------------------------------------------------------

N_FRAMES = 300
EVENT = "BOTTLE_absence_gap"
CORE = "roi:core"

ARMS = [
    "U", "G", "GEST", "GESTMEM", "GMIS",
    "GWRONGMEM_A", "GWRONGMEM_B", "GWRONGMEM_L", "GONES",
]
GATED_ARMS = [a for a in ARMS if a != "U"]
SCENES = ("flame_steak", "sear_steak")
PREFIXES = (0, 1, 2, 3)

# Offsets in dB added to the U level, per arm. G clears the 0.5 floor against
# every sham on both endpoints; every sham sits at or below U.
DEFAULT_OFFSETS = {
    "U": 0.0, "G": 2.0, "GEST": 1.9, "GESTMEM": 1.8, "GMIS": 0.1,
    "GWRONGMEM_A": 0.2, "GWRONGMEM_B": 0.15, "GWRONGMEM_L": 0.3, "GONES": -0.4,
}


def _series(value):
    return [float(value)] * N_FRAMES


def write_v2_profile(run_dir, level):
    """A profile whose every endpoint pools to `level` dB."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_frames": N_FRAMES,
        "frames": list(range(N_FRAMES)),
        "whole_frame_psnr": _series(level),
        "events": {
            EVENT: {
                "bbox": [964, 748, 1035, 953],
                "per_frame_psnr": _series(level),
            },
            CORE: {
                "kind": "per_frame_mask",
                "per_frame_psnr": _series(level),
                "pixels_per_frame": [1000] * N_FRAMES,
            },
        },
    }
    (run_dir / "f_box_profile.json").write_text(
        json.dumps(payload), encoding="utf-8")


def good_precondition(arm, *, fbox, fbox_frame, zero_frames):
    return {
        "gated_rows_seeding": 3000,
        "n_rows_seeding": 500000,
        "gated_rows_final": 4000,
        "n_rows_final": 600000,
        "gated_rows_fbox_frame150": 900,
        "frames_presence_zero": len(zero_frames),
        "reserved_units": 1425,
        "training_units_total": 5700,
        "arm_kind": "gated" if arm != "U" else "ungated",
        "program_match": True,
        "detail": {
            "presence": {"frames_presence_zero": list(zero_frames)},
            "fbox": {
                "box_x0_y0_x1_y1_inclusive": list(fbox),
                "frame": int(fbox_frame),
            },
        },
    }


def u_precondition():
    return {
        "gated_rows_seeding": 0,
        "n_rows_seeding": 500000,
        "gated_rows_final": 0,
        "n_rows_final": 600000,
        "frames_presence_zero": 0,
        "reserved_units": 1425,
        "training_units_total": 5700,
        "arm_kind": "ungated",
    }


def membership_sidecar(*, tp=90, fp=10, fn=20, truth_n=110, pred_n=100,
                       iou_num=80, iou_den=100):
    return {
        "TP": tp, "FP": fp, "FN": fn,
        "truth_n": truth_n, "pred_n": pred_n,
        "mask_iou_cam15": {
            "50": {"pixel_intersection": iou_num, "pixel_union": iou_den},
            "95": {"pixel_intersection": iou_num, "pixel_union": iou_den},
        },
    }


def t1_sidecar(intervals=((60, 89),)):
    return {"intervals": [list(iv) for iv in intervals]}


def program_sidecar(overlap_n=0, truth_n=1700, draw_n=1700):
    return {
        "truth_n": truth_n, "draw_n": draw_n, "overlap_n": overlap_n,
        "row_ids_sha256": "0" * 64,
    }


def build_v2_case(
    tmp_path,
    *,
    scenes=SCENES,
    prefixes=PREFIXES,
    arms=ARMS,
    offsets=None,
    drop_cells=(),
    membership=None,
    t1=None,
    programs=None,
    preconditions=None,
    calibration=True,
):
    """A manifest + fixture tree for `scenes` x `prefixes` x `arms`.

    `drop_cells` is a set of (scene, prefix, arm) to omit.
    `membership`/`t1`/`programs`/`preconditions` are {(scene, prefix): payload}
    overrides; a payload of None omits that sidecar entirely.
    """
    offsets = dict(DEFAULT_OFFSETS if offsets is None else offsets)
    root = Path(tmp_path) / "cells"
    cells = []
    sidecars = {}
    all_scenes = list(scenes) + (["cut_roasted_beef"] if calibration else [])
    for scene in all_scenes:
        sidecars[scene] = {}
        for prefix in prefixes:
            for arm in arms:
                if (scene, prefix, arm) in drop_cells:
                    continue
                run_dir = root / scene / ("p%d" % prefix) / arm
                write_v2_profile(run_dir, 30.0 + offsets.get(arm, 0.0))
                key = (scene, prefix)
                if preconditions is not None and key in preconditions:
                    pre = preconditions[key].get(arm, "default")
                else:
                    pre = "default"
                if pre == "default":
                    pre = (
                        u_precondition() if arm == "U"
                        else good_precondition(
                            arm, fbox=[964, 748, 1034, 952],
                            fbox_frame=(
                                244 if arm == "GMIS"
                                else 291 if arm == "GONES" else 75),
                            zero_frames=(
                                list(range(230, 260)) if arm == "GMIS"
                                else list(range(286, 298)) if arm == "GONES"
                                else list(range(61, 89))),
                        )
                    )
                if pre is not None:
                    (run_dir / "precondition.json").write_text(
                        json.dumps(pre), encoding="utf-8")
                cells.append({
                    "arm": arm, "seed": prefix, "prefix": prefix,
                    "scene": scene, "run_dir": str(run_dir),
                    "status": "complete",
                })
            entry = {}
            side_dir = root / scene / ("p%d" % prefix)
            key = (scene, prefix)
            m = membership_sidecar() if membership is None else membership.get(
                key, membership_sidecar())
            if m is not None:
                path = side_dir / "membership.json"
                path.write_text(json.dumps(m), encoding="utf-8")
                entry["membership"] = str(path)
            t = t1_sidecar() if t1 is None else t1.get(key, t1_sidecar())
            if t is not None:
                path = side_dir / "t1.json"
                path.write_text(json.dumps(t), encoding="utf-8")
                entry["t1"] = str(path)
            progs = {} if programs is None else programs.get(key, {})
            entry["programs"] = {}
            for arm in ("GWRONGMEM_A", "GWRONGMEM_B", "GWRONGMEM_L"):
                payload = progs.get(arm, program_sidecar())
                if payload is None:
                    continue
                path = side_dir / ("program_%s.json" % arm)
                path.write_text(json.dumps(payload), encoding="utf-8")
                entry["programs"][arm] = str(path)
            sidecars[scene][str(prefix)] = entry
    manifest = {"cells": cells, "sidecars": sidecars}
    path = Path(tmp_path) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return str(path)


def _shipped_v2_spec():
    """The shipped v2 instance, plus the one key it does not carry.

    `configs/n3v/absfix_gate_spec_v2.json` names `roi:core` for P1 but never
    names the bounding-box event the other five endpoints read, so the merge
    inherits the frozen SPEC's `F_blade_over_beef_reveal`, which no wave-2
    profile contains. The file is frozen (its sha256 is recorded in section 12
    of the spec page) and is not edited here; the test supplies the name, and
    the missing key is reported as a freeze-list gap.
    """
    spec, _ = rga.load_spec(str(SPEC_V2))
    return rga.deep_merge(spec, {"event_name": EVENT})


def v2_spec(**overrides):
    """The shipped v2 instance with the two confirmatory scenes filled in.

    The shipped file leaves the confirmatory anchors and boxes "pending"; a
    reducer test needs them resolved, so the test fills exactly the fields
    the freeze list says are filled before a scene trains.
    """
    filled = copy.deepcopy(_shipped_v2_spec())
    for scene in SCENES:
        block = filled["scenes"][scene]
        block["CA"] = 230
        block["CB"] = 259
        block["MECHANISM_FBOX"] = [964, 748, 1034, 952]
        block["MECHANISM_FBOX_FRAME"] = 75
    filled = rga.deep_merge(filled, overrides)
    return rga.resolve_spec(filled)


def run_v2(manifest, spec=None):
    return rga.run(manifest, wave=2, spec=spec or v2_spec(), paired=True)


def _claim(report, name, scene=None):
    block = report["v2"]["claims"][name]
    if scene is None:
        return block["verdict"]
    return block["per_scene"][scene]["verdict"]


# ---------------------------------------------------------------------------
# 3. The pass branch, and each required contrast failing on its own
# ---------------------------------------------------------------------------


def test_the_pass_branch_meets_both_claims(tmp_path):
    report = run_v2(build_v2_case(tmp_path))
    assert _claim(report, "CLAIM_A") == "CLAIM_CONDITIONS_MET"
    assert _claim(report, "CLAIM_B") == "CLAIM_CONDITIONS_MET"
    for scene in SCENES:
        assert _claim(report, "CLAIM_A", scene) == "CLAIM_CONDITIONS_MET"
        assert _claim(report, "CLAIM_B", scene) == "CLAIM_CONDITIONS_MET"


@pytest.mark.parametrize("sham", ["GWRONGMEM_A", "GWRONGMEM_B",
                                  "GWRONGMEM_L", "GMIS"])
def test_each_required_sham_contrast_can_deny_claim_a(tmp_path, sham):
    offsets = dict(DEFAULT_OFFSETS)
    offsets[sham] = offsets["G"] - 0.4          # inside the 0.5 dB floor
    report = run_v2(build_v2_case(tmp_path, offsets=offsets))
    assert _claim(report, "CLAIM_A") == "NOT_MET"
    contrast = "G-%s" % sham
    for scene in SCENES:
        detail = report["v2"]["claims"]["CLAIM_A"]["per_scene"][scene]
        assert detail["contrasts"][contrast]["P1"]["all_pairs_over_floor"] is False


def test_the_g_minus_u_contrast_can_deny_claim_a(tmp_path):
    offsets = dict(DEFAULT_OFFSETS, G=0.4)
    report = run_v2(build_v2_case(tmp_path, offsets=offsets))
    assert _claim(report, "CLAIM_A") == "NOT_MET"


def test_a_contrast_may_fail_on_p2_alone(tmp_path):
    """Both endpoints are claim-bearing, so P2 alone denies the claim."""
    manifest = build_v2_case(tmp_path)
    spec = v2_spec()
    # push G down on P2 only, by rewriting its P2 window on one scene
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    for cell in data["cells"]:
        if cell["arm"] == "G" and cell["scene"] == "flame_steak":
            path = Path(cell["run_dir"]) / "f_box_profile.json"
            profile = json.loads(path.read_text(encoding="utf-8"))
            for frame in range(92, 100):
                profile["events"][EVENT]["per_frame_psnr"][frame] = 30.0
            path.write_text(json.dumps(profile), encoding="utf-8")
    report = rga.run(manifest, wave=2, spec=spec, paired=True)
    assert _claim(report, "CLAIM_A", "flame_steak") == "NOT_MET"
    assert _claim(report, "CLAIM_A", "sear_steak") == "CLAIM_CONDITIONS_MET"
    assert _claim(report, "CLAIM_A") == "PARTIAL"


# ---------------------------------------------------------------------------
# 4. The floor is strict: exactly +0.5 dB does not clear it
# ---------------------------------------------------------------------------


def test_exactly_the_floor_does_not_clear_it(tmp_path):
    offsets = dict(DEFAULT_OFFSETS)
    offsets["GWRONGMEM_A"] = offsets["G"] - 0.5
    report = run_v2(build_v2_case(tmp_path, offsets=offsets))
    pair = (report["v2"]["claims"]["CLAIM_A"]["per_scene"]["flame_steak"]
            ["contrasts"]["G-GWRONGMEM_A"]["P1"]["pairs"][0])
    assert pair["diff"] == pytest.approx(0.5, abs=1e-9)
    assert pair["over_floor"] is False
    assert _claim(report, "CLAIM_A") == "NOT_MET"


def test_just_over_the_floor_clears_it(tmp_path):
    offsets = dict(DEFAULT_OFFSETS)
    offsets["GWRONGMEM_A"] = offsets["G"] - 0.5001
    report = run_v2(build_v2_case(tmp_path, offsets=offsets))
    assert _claim(report, "CLAIM_A") == "CLAIM_CONDITIONS_MET"


# ---------------------------------------------------------------------------
# 5. Harm guards
# ---------------------------------------------------------------------------


def test_a_harm_guard_breach_on_one_pair_denies_the_claim(tmp_path):
    """H1/H2 admit 0.15 dB of `U - G`; one pair over it is enough."""
    manifest = build_v2_case(tmp_path)
    for cell in json.loads(Path(manifest).read_text(encoding="utf-8"))["cells"]:
        if cell["arm"] == "G" and cell["scene"] == "sear_steak" \
                and cell["prefix"] == 2:
            path = Path(cell["run_dir"]) / "f_box_profile.json"
            profile = json.loads(path.read_text(encoding="utf-8"))
            for frame in range(30, 58):          # H1 = [A-30, A-3]
                profile["events"][EVENT]["per_frame_psnr"][frame] = 29.5
            path.write_text(json.dumps(profile), encoding="utf-8")
    report = run_v2(manifest)
    guard = (report["v2"]["claims"]["CLAIM_A"]["per_scene"]["sear_steak"]
             ["harm_guards"]["H1"])
    assert guard["passes"] is False
    assert guard["offenders"][0]["prefix"] == "sear_steak:2"
    assert _claim(report, "CLAIM_A", "sear_steak") == "NOT_MET"


def test_the_control_window_guard_has_its_own_ceiling(tmp_path):
    """C1 admits 0.35 dB, more than H1/H2's 0.15."""
    manifest = build_v2_case(tmp_path)
    for cell in json.loads(Path(manifest).read_text(encoding="utf-8"))["cells"]:
        if cell["arm"] == "G":
            path = Path(cell["run_dir"]) / "f_box_profile.json"
            profile = json.loads(path.read_text(encoding="utf-8"))
            for frame in range(230, 260):        # C1
                profile["events"][EVENT]["per_frame_psnr"][frame] = 30.0 + 1.75
            path.write_text(json.dumps(profile), encoding="utf-8")
    report = run_v2(manifest)                    # U - G = -0.25 <= 0.35
    assert (report["v2"]["claims"]["CLAIM_A"]["per_scene"]["flame_steak"]
            ["harm_guards"]["C1"]["passes"]) is True
    assert _claim(report, "CLAIM_A") == "CLAIM_CONDITIONS_MET"


# ---------------------------------------------------------------------------
# 6. Fail-closed: missing arms, sidecars, malformed programs
# ---------------------------------------------------------------------------


def test_a_missing_gwrongmem_b_pair_is_design_without_power(tmp_path):
    manifest = build_v2_case(
        tmp_path, drop_cells={("flame_steak", 1, "GWRONGMEM_B")})
    report = run_v2(manifest)
    scene = report["v2"]["claims"]["CLAIM_A"]["per_scene"]["flame_steak"]
    assert scene["verdict"] == "DESIGN_WITHOUT_POWER"
    assert any("GWRONGMEM_B" in r for r in scene["reasons"])
    assert _claim(report, "CLAIM_A") == "DESIGN_WITHOUT_POWER"


def test_an_arm_absent_from_the_whole_scene_is_design_without_power(tmp_path):
    manifest = build_v2_case(
        tmp_path,
        drop_cells={(s, p, "GWRONGMEM_L") for s in SCENES for p in PREFIXES})
    report = run_v2(manifest)
    assert _claim(report, "CLAIM_A") == "DESIGN_WITHOUT_POWER"


def test_fewer_than_four_prefixes_is_design_without_power(tmp_path):
    report = run_v2(build_v2_case(tmp_path, prefixes=(0, 1, 2)))
    scene = report["v2"]["claims"]["CLAIM_A"]["per_scene"]["flame_steak"]
    assert scene["verdict"] == "DESIGN_WITHOUT_POWER"
    assert any("4" in r for r in scene["reasons"])


def test_a_missing_membership_sidecar_is_design_without_power_for_claim_b(tmp_path):
    manifest = build_v2_case(
        tmp_path, membership={("flame_steak", 2): None})
    report = run_v2(manifest)
    assert _claim(report, "CLAIM_B", "flame_steak") == "DESIGN_WITHOUT_POWER"
    assert _claim(report, "CLAIM_A", "flame_steak") == "CLAIM_CONDITIONS_MET"


def test_a_missing_t1_sidecar_is_design_without_power_for_the_est_legs(tmp_path):
    report = run_v2(build_v2_case(tmp_path, t1={("sear_steak", 0): None}))
    assert _claim(report, "CLAIM_B", "sear_steak") == "DESIGN_WITHOUT_POWER"
    assert _claim(report, "CLAIM_A", "sear_steak") == "CLAIM_CONDITIONS_MET"


def test_a_malformed_program_sidecar_is_refused(tmp_path):
    report = run_v2(build_v2_case(
        tmp_path,
        programs={("flame_steak", 0): {"GWRONGMEM_A": {"truth_n": 1700}}}))
    scene = report["v2"]["claims"]["CLAIM_A"]["per_scene"]["flame_steak"]
    assert scene["verdict"] == "DESIGN_WITHOUT_POWER"
    assert any("overlap_n" in r for r in scene["reasons"])


def test_a_missing_program_sidecar_is_refused(tmp_path):
    report = run_v2(build_v2_case(
        tmp_path, programs={("sear_steak", 3): {"GWRONGMEM_L": None}}))
    scene = report["v2"]["claims"]["CLAIM_A"]["per_scene"]["sear_steak"]
    assert scene["verdict"] == "DESIGN_WITHOUT_POWER"


def test_a_manifest_cell_without_a_scene_is_refused(tmp_path):
    manifest = build_v2_case(tmp_path)
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    del data["cells"][0]["scene"]
    Path(manifest).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="scene"):
        run_v2(manifest)


def test_a_cell_naming_an_unknown_scene_is_refused(tmp_path):
    manifest = build_v2_case(tmp_path)
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    data["cells"][0]["scene"] = "coffee_martini"
    Path(manifest).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="coffee_martini"):
        run_v2(manifest)


def test_a_confirmatory_scene_absent_from_the_manifest_is_dwp(tmp_path):
    report = run_v2(build_v2_case(tmp_path, scenes=("flame_steak",)))
    assert _claim(report, "CLAIM_A", "sear_steak") == "DESIGN_WITHOUT_POWER"
    assert _claim(report, "CLAIM_A") == "DESIGN_WITHOUT_POWER"


def test_a_scene_whose_anchors_are_still_pending_is_dwp(tmp_path):
    """The shipped instance leaves CA/CB pending; that must fail closed."""
    report = rga.run(build_v2_case(tmp_path), wave=2,
                     spec=rga.resolve_spec(_shipped_v2_spec()), paired=True)
    scene = report["v2"]["scenes"]["flame_steak"]
    assert scene["admitted"] is False
    assert any("pending" in r for r in scene["reasons"])
    assert _claim(report, "CLAIM_A") == "DESIGN_WITHOUT_POWER"


# ---------------------------------------------------------------------------
# 7. The zero-overlap assertion
# ---------------------------------------------------------------------------


def test_overlap_of_one_row_refuses_the_scene(tmp_path):
    report = run_v2(build_v2_case(
        tmp_path,
        programs={("flame_steak", 0): {
            "GWRONGMEM_A": program_sidecar(overlap_n=1)}}))
    scene = report["v2"]["claims"]["CLAIM_A"]["per_scene"]["flame_steak"]
    assert scene["verdict"] == "DESIGN_WITHOUT_POWER"
    assert any("overlap" in r for r in scene["reasons"])
    row = report["v2"]["scenes"]["flame_steak"]["prefixes"]["0"]["programs"]
    assert row["GWRONGMEM_A"]["overlap_n"] == 1
    assert row["GWRONGMEM_A"]["passes"] is False


def test_zero_overlap_passes_and_is_recorded_with_its_counts(tmp_path):
    report = run_v2(build_v2_case(tmp_path))
    row = (report["v2"]["scenes"]["flame_steak"]["prefixes"]["0"]
           ["programs"]["GWRONGMEM_B"])
    assert row["overlap_n"] == 0 and row["passes"] is True
    assert row["count_match"]["numerator"] == row["count_match"]["denominator"]


# ---------------------------------------------------------------------------
# 8. The two instrument preconditions
# ---------------------------------------------------------------------------


def test_membership_precondition_failure_hits_claim_b_only(tmp_path):
    bad = membership_sidecar(tp=60, fp=40, fn=20, truth_n=80, pred_n=100)
    report = run_v2(build_v2_case(tmp_path, membership={("flame_steak", 1): bad}))
    block = (report["v2"]["scenes"]["flame_steak"]["prefixes"]["1"]
             ["membership"])
    assert block["precision"]["numerator"] == 60
    assert block["precision"]["denominator"] == 100
    assert block["precision"]["value"] == pytest.approx(0.6)
    assert block["passes"] is False
    assert _claim(report, "CLAIM_B", "flame_steak") == "DESIGN_WITHOUT_POWER"
    assert _claim(report, "CLAIM_A", "flame_steak") == "CLAIM_CONDITIONS_MET"


@pytest.mark.parametrize("payload,field", [
    (membership_sidecar(tp=60, fp=40, fn=20), "precision"),
    (membership_sidecar(tp=60, fp=5, fn=40, truth_n=100), "recall"),
    (membership_sidecar(pred_n=400), "size_ratio"),
    (membership_sidecar(iou_num=50, iou_den=100), "mask_iou_cam15"),
])
def test_each_membership_criterion_can_fail_on_its_own(tmp_path, payload, field):
    report = run_v2(build_v2_case(tmp_path, membership={("sear_steak", 0): payload}))
    block = report["v2"]["scenes"]["sear_steak"]["prefixes"]["0"]["membership"]
    assert block["passes"] is False
    assert any(field in r for r in block["reasons"])


def test_non_integer_membership_counts_are_refused(tmp_path):
    payload = membership_sidecar()
    payload["TP"] = 90.5
    report = run_v2(build_v2_case(tmp_path, membership={("sear_steak", 0): payload}))
    block = report["v2"]["scenes"]["sear_steak"]["prefixes"]["0"]["membership"]
    assert block["passes"] is False
    assert any("integer" in r for r in block["reasons"])


def test_t1_precondition_failure_takes_the_est_legs_out(tmp_path):
    report = run_v2(build_v2_case(
        tmp_path, t1={("flame_steak", 2): t1_sidecar(((60, 89), (200, 210)))}))
    block = report["v2"]["scenes"]["flame_steak"]["prefixes"]["2"]["t1"]
    assert block["passes"] is False
    assert block["interval_count"] == 2
    assert _claim(report, "CLAIM_B", "flame_steak") == "DESIGN_WITHOUT_POWER"
    assert _claim(report, "CLAIM_A", "flame_steak") == "CLAIM_CONDITIONS_MET"


@pytest.mark.parametrize("intervals,field", [
    (((55, 89),), "onset"),
    (((60, 95),), "offset"),
    (((60, 62),), "temporal_iou"),
])
def test_each_t1_criterion_can_fail_on_its_own(tmp_path, intervals, field):
    report = run_v2(build_v2_case(
        tmp_path, t1={("sear_steak", 1): t1_sidecar(intervals)}))
    block = report["v2"]["scenes"]["sear_steak"]["prefixes"]["1"]["t1"]
    assert block["passes"] is False
    assert any(field in r for r in block["reasons"])


def test_t1_iou_carries_its_numerator_and_denominator(tmp_path):
    report = run_v2(build_v2_case(tmp_path))
    block = report["v2"]["scenes"]["flame_steak"]["prefixes"]["0"]["t1"]
    assert block["temporal_iou"]["numerator"] == 30
    assert block["temporal_iou"]["denominator"] == 30
    assert block["temporal_iou"]["value"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 9. The per-cell precondition
# ---------------------------------------------------------------------------


def test_a_cell_below_the_gated_row_floor_leaves_the_mechanism_set(tmp_path):
    pre = good_precondition("G", fbox=[964, 748, 1034, 952], fbox_frame=75,
                            zero_frames=list(range(61, 89)))
    pre["gated_rows_final"] = 999
    report = run_v2(build_v2_case(
        tmp_path, preconditions={("flame_steak", 0): {"G": pre}}))
    scene = report["v2"]["claims"]["CLAIM_A"]["per_scene"]["flame_steak"]
    assert scene["verdict"] == "DESIGN_WITHOUT_POWER"
    cell = [c for c in report["cells"]
            if c["arm"] == "G" and c["prefix"] == "flame_steak:0"][0]
    assert cell["mechanism_exercised"] is False
    assert "999 < 1000" in cell["mechanism_reason"]


def test_a_cell_below_the_fbox_floor_leaves_the_mechanism_set(tmp_path):
    pre = good_precondition("G", fbox=[964, 748, 1034, 952], fbox_frame=75,
                            zero_frames=list(range(61, 89)))
    pre["gated_rows_fbox_frame150"] = 99
    report = run_v2(build_v2_case(
        tmp_path, preconditions={("sear_steak", 3): {"G": pre}}))
    cell = [c for c in report["cells"]
            if c["arm"] == "G" and c["prefix"] == "sear_steak:3"][0]
    assert cell["mechanism_exercised"] is False
    assert "99 < 100" in cell["mechanism_reason"]


def test_a_cell_with_no_zero_presence_frame_in_the_gap_is_excluded(tmp_path):
    pre = good_precondition("G", fbox=[964, 748, 1034, 952], fbox_frame=75,
                            zero_frames=[200, 201])
    report = run_v2(build_v2_case(
        tmp_path, preconditions={("sear_steak", 1): {"G": pre}}))
    cell = [c for c in report["cells"]
            if c["arm"] == "G" and c["prefix"] == "sear_steak:1"][0]
    assert cell["mechanism_exercised"] is False
    assert "zero-presence" in cell["mechanism_reason"]


def test_program_match_is_required_when_the_spec_says_so(tmp_path):
    pre = good_precondition("G", fbox=[964, 748, 1034, 952], fbox_frame=75,
                            zero_frames=list(range(61, 89)))
    del pre["program_match"]
    report = run_v2(build_v2_case(
        tmp_path, preconditions={("flame_steak", 2): {"G": pre}}))
    cell = [c for c in report["cells"]
            if c["arm"] == "G" and c["prefix"] == "flame_steak:2"][0]
    assert cell["mechanism_exercised"] is False
    assert "program_match" in cell["mechanism_reason"]


def test_unequal_reserved_units_within_a_prefix_drop_that_prefix(tmp_path):
    pre = good_precondition("G", fbox=[964, 748, 1034, 952], fbox_frame=75,
                            zero_frames=list(range(61, 89)))
    pre["reserved_units"] = 1400
    report = run_v2(build_v2_case(
        tmp_path, preconditions={("flame_steak", 0): {"G": pre}}))
    scene = report["v2"]["claims"]["CLAIM_A"]["per_scene"]["flame_steak"]
    assert scene["verdict"] == "DESIGN_WITHOUT_POWER"
    assert any("reserved" in r for r in scene["reasons"])


def test_the_timing_shams_are_exempt_from_the_gap_window_test(tmp_path):
    """GMIS and GONES gate elsewhere by design; the window test would be
    a contradiction, and the exemption must be stated in the record."""
    report = run_v2(build_v2_case(tmp_path))
    cell = [c for c in report["cells"]
            if c["arm"] == "GMIS" and c["prefix"] == "flame_steak:0"][0]
    assert cell["mechanism_exercised"] is True
    assert "own gap" in cell["mechanism_reason"]


# ---------------------------------------------------------------------------
# 10. Scenes: precedence, calibration, descriptive contrasts
# ---------------------------------------------------------------------------


def test_one_confirmatory_scene_met_is_partial(tmp_path):
    manifest = build_v2_case(tmp_path)
    for cell in json.loads(Path(manifest).read_text(encoding="utf-8"))["cells"]:
        if cell["arm"] == "G" and cell["scene"] == "sear_steak":
            write_v2_profile(cell["run_dir"], 30.0 + 0.2)
    report = run_v2(manifest)
    assert _claim(report, "CLAIM_A", "flame_steak") == "CLAIM_CONDITIONS_MET"
    assert _claim(report, "CLAIM_A", "sear_steak") == "NOT_MET"
    assert _claim(report, "CLAIM_A") == "PARTIAL"


def test_neither_scene_met_is_not_met(tmp_path):
    offsets = dict(DEFAULT_OFFSETS, G=0.1)
    report = run_v2(build_v2_case(tmp_path, offsets=offsets))
    assert _claim(report, "CLAIM_A") == "NOT_MET"


def test_dwp_on_one_scene_outranks_met_on_the_other(tmp_path):
    report = run_v2(build_v2_case(
        tmp_path, drop_cells={("sear_steak", 0, "GMIS")}))
    assert _claim(report, "CLAIM_A", "flame_steak") == "CLAIM_CONDITIONS_MET"
    assert _claim(report, "CLAIM_A", "sear_steak") == "DESIGN_WITHOUT_POWER"
    assert _claim(report, "CLAIM_A") == "DESIGN_WITHOUT_POWER"
    assert report["v2"]["claims"]["CLAIM_A"]["precedence"][0] == \
        "DESIGN_WITHOUT_POWER"


def test_the_calibration_scene_is_reported_but_never_in_the_verdict(tmp_path):
    """cut_roasted_beef fails here; the confirmatory verdict is untouched."""
    manifest = build_v2_case(tmp_path)
    for cell in json.loads(Path(manifest).read_text(encoding="utf-8"))["cells"]:
        if cell["arm"] == "G" and cell["scene"] == "cut_roasted_beef":
            write_v2_profile(cell["run_dir"], 30.0)
    report = run_v2(manifest)
    assert _claim(report, "CLAIM_A") == "CLAIM_CONDITIONS_MET"
    calib = report["v2"]["claims"]["CLAIM_A"]["calibration"]["cut_roasted_beef"]
    assert calib["verdict"] == "NOT_MET"
    assert calib["role"] == "CALIBRATION"
    assert "cut_roasted_beef" not in report["v2"]["claims"]["CLAIM_A"]["per_scene"]


def test_the_descriptive_contrasts_are_reported_per_scene(tmp_path):
    report = run_v2(build_v2_case(tmp_path))
    desc = report["v2"]["scenes"]["flame_steak"]["descriptive"]
    for name in ("GWRONGMEM_A-U", "GWRONGMEM_B-U", "GWRONGMEM_L-U",
                 "G-GEST", "G-GESTMEM", "GONES-U"):
        assert name in desc
        assert desc[name]["P1"]["n_pairs"] == 4
    assert desc["GONES-U"]["P1"]["median"] == pytest.approx(-0.4, abs=1e-9)


def test_every_ratio_in_the_v2_record_carries_its_numerator_and_denominator(tmp_path):
    report = run_v2(build_v2_case(tmp_path))
    seen = []

    def walk(node, path):
        if isinstance(node, dict):
            if "value" in node and ("numerator" in node or "denominator" in node):
                assert isinstance(node["numerator"], int), path
                assert isinstance(node["denominator"], int), path
                seen.append(path)
            for key, value in node.items():
                walk(value, path + "/" + str(key))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, "%s[%d]" % (path, i))

    walk(report["v2"], "v2")
    assert len(seen) > 20


def test_the_markdown_report_carries_a_v2_section(tmp_path):
    report = run_v2(build_v2_case(tmp_path))
    text = rga.markdown_report(report)
    assert "## Spec v2 scene verdicts" in text
    assert "CLAIM_A" in text and "CLAIM_B" in text
    assert "flame_steak" in text and "sear_steak" in text
    assert "CALIBRATION" in text


def test_the_v2_record_names_the_pairs_it_used(tmp_path):
    report = run_v2(build_v2_case(tmp_path))
    block = (report["v2"]["claims"]["CLAIM_A"]["per_scene"]["flame_steak"]
             ["contrasts"]["G-U"]["P1"])
    assert block["n_pairs"]["numerator"] == 4
    assert block["n_pairs"]["denominator"] == 4
    assert [p["prefix"] for p in block["pairs"]] == [
        "flame_steak:0", "flame_steak:1", "flame_steak:2", "flame_steak:3"]
    assert all(p["diff"] == pytest.approx(2.0, abs=1e-9) for p in block["pairs"])


def test_v2_does_not_disturb_the_v1_paired_verdict_machinery(tmp_path):
    report = run_v2(build_v2_case(tmp_path))
    assert report["paired"] is True
    assert "itt" in report["analysis"] and "mechanism_exercised" in report["analysis"]
    assert math.isfinite(
        report["analysis"]["itt"]["contrasts"]["G-U"]["P1"]["median"])
