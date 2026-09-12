"""Rule-level tests for the S2 (SAM2) propagation driver.

These exercise the parts of `scripts/s2_sam2_propagate.py` that decide
whether a cell runs at all - click validation, the camera-drop rule, the
>= 12 camera rule and the manifest's field set - and they deliberately need
neither torch nor SAM2, so the frozen rules of spec v2.0.0 sections 11.3 and
13.1 can be checked on a workstation before any GPU is asked for.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "s2_sam2_propagate", REPO_ROOT / "scripts" / "s2_sam2_propagate.py"
)
s2 = importlib.util.module_from_spec(_SPEC)
sys.modules["s2_sam2_propagate"] = s2
_SPEC.loader.exec_module(s2)


def _filled(cams=("cam01", "cam02")):
    return {c: {"f50": [100, 200], "f92": [110, 210]} for c in cams}


# --------------------------------------------------------------------------
# clicks
# --------------------------------------------------------------------------
def test_filled_clicks_round_trip(tmp_path):
    p = tmp_path / "clicks.json"
    p.write_text(json.dumps({"_instructions": "ignored", "crb": _filled()}))
    out = s2.load_clicks(p, "crb")
    assert out == {
        "cam01": {"f50": [100, 200], "f92": [110, 210]},
        "cam02": {"f50": [100, 200], "f92": [110, 210]},
    }


def test_null_click_is_refused(tmp_path):
    """An unfilled template must refuse, never default."""
    table = _filled()
    table["cam02"]["f92"] = None
    p = tmp_path / "clicks.json"
    p.write_text(json.dumps({"crb": table}))
    with pytest.raises(s2.Refusal, match="null"):
        s2.load_clicks(p, "crb")


def test_all_null_template_is_refused(tmp_path):
    p = tmp_path / "clicks.json"
    p.write_text(
        json.dumps({"crb": {"cam01": {"f50": None, "f92": None}}})
    )
    with pytest.raises(s2.Refusal):
        s2.load_clicks(p, "crb")


def test_the_shipped_template_is_still_unfilled_and_refused():
    """The template in the repo must not silently become usable input."""
    template = REPO_ROOT / "research-wiki" / "assets" / "absfix-s2-clicks-template.json"
    if not template.exists():
        pytest.skip("click template not present in this checkout")
    with pytest.raises(s2.Refusal):
        s2.load_clicks(template, "flame_steak")


def test_missing_seed_key_is_refused():
    with pytest.raises(s2.Refusal, match="missing click"):
        s2.validate_clicks({"cam01": {"f50": [1, 2]}}, "crb")


def test_malformed_click_is_refused():
    with pytest.raises(s2.Refusal, match=r"must be \[x, y\]"):
        s2.validate_clicks({"cam01": {"f50": [1, 2, 3], "f92": [1, 2]}}, "crb")
    with pytest.raises(s2.Refusal, match="not a number"):
        s2.validate_clicks({"cam01": {"f50": ["a", 2], "f92": [1, 2]}}, "crb")
    with pytest.raises(s2.Refusal, match="negative"):
        s2.validate_clicks({"cam01": {"f50": [-1, 2], "f92": [1, 2]}}, "crb")


def test_unknown_scene_is_refused(tmp_path):
    p = tmp_path / "clicks.json"
    p.write_text(json.dumps({"crb": _filled()}))
    with pytest.raises(s2.Refusal, match="no entry for scene"):
        s2.load_clicks(p, "flame_steak")


def test_construction_mask_path_is_refused(tmp_path):
    with pytest.raises(s2.Refusal, match="DEVA/SA4D"):
        s2._refuse_deva_path(tmp_path / "cam01" / "pseudo_label" / "object_mask", "x")
    with pytest.raises(s2.Refusal):
        s2._refuse_deva_path(tmp_path / "repo" / "sa4d" / "data", "x")
    with pytest.raises(s2.Refusal):
        s2._refuse_deva_path(tmp_path / "DEVA_masks" / "cam01", "x")


def test_ordinary_derived_scene_path_is_accepted(tmp_path):
    """The component rule must not fire on an unrelated path."""
    s2._refuse_deva_path(tmp_path / "data_derived" / "absfix" / "crb", "x")
    s2._refuse_deva_path("/w/runs/realdata/s2_smoke/out", "x")


# --------------------------------------------------------------------------
# the camera-drop rule
# --------------------------------------------------------------------------
def test_drop_rule_keeps_a_uniform_cohort():
    areas = {f"cam{i:02d}": {"f50": 4000 + 10 * i, "f92": 3800 + 10 * i} for i in range(1, 16)}
    kept, dropped, reasons, medians = s2.apply_camera_drop_rule(areas)
    assert dropped == []
    assert len(kept) == 15
    assert reasons == {}
    assert medians["f50"] == 4080.0


def test_drop_rule_drops_small_and_huge_at_either_seed():
    areas = {f"cam{i:02d}": {"f50": 4000, "f92": 4000} for i in range(1, 14)}
    areas["cam20"] = {"f50": 120, "f92": 4000}     # below the 500 px floor at f50
    areas["cam21"] = {"f50": 4000, "f92": 90000}   # above 5x median at f92
    areas["cam22"] = {"f50": 4000, "f92": 4000}    # fine
    kept, dropped, reasons, medians = s2.apply_camera_drop_rule(areas)
    assert dropped == ["cam20", "cam21"]
    assert "cam22" in kept and "cam20" not in kept
    assert "area 120 < 500 px" in reasons["cam20"][0]
    assert "5.0x median" in reasons["cam21"][0]
    assert medians["f50"] == 4000.0


def test_drop_rule_is_applied_at_each_seed_separately():
    """Spec 13.1: a camera dropped at EITHER seed frame is dropped."""
    areas = {f"cam{i:02d}": {"f50": 1000, "f92": 1000} for i in range(1, 13)}
    areas["cam13"] = {"f50": 1000, "f92": 200}
    kept, dropped, reasons, _ = s2.apply_camera_drop_rule(areas)
    assert dropped == ["cam13"]
    assert reasons["cam13"] == ["f92: area 200 < 500 px"]
    assert len(kept) == 12


def test_drop_rule_thresholds_are_strict_inequalities():
    areas = {f"cam{i:02d}": {"f50": 1000, "f92": 1000} for i in range(1, 12)}
    areas["cam90"] = {"f50": 500, "f92": 5000}  # exactly at floor and exactly 5x median
    kept, dropped, _, medians = s2.apply_camera_drop_rule(areas)
    assert medians["f50"] == 1000.0
    assert dropped == []
    assert "cam90" in kept


def test_drop_rule_custom_thresholds():
    areas = {f"cam{i:02d}": {"f50": 1000, "f92": 1000} for i in range(1, 12)}
    areas["cam90"] = {"f50": 900, "f92": 1000}
    _, dropped, _, _ = s2.apply_camera_drop_rule(areas, min_area=950)
    assert dropped == ["cam90"]
    _, dropped2, _, _ = s2.apply_camera_drop_rule(areas, median_factor=1.0)
    assert dropped2 == []


def test_drop_rule_refuses_empty_and_incomplete_input():
    with pytest.raises(s2.Refusal):
        s2.apply_camera_drop_rule({})
    with pytest.raises(s2.Refusal, match="no area at seed"):
        s2.apply_camera_drop_rule({"cam01": {"f50": 1000}})


# --------------------------------------------------------------------------
# the >= 12 camera rule
# --------------------------------------------------------------------------
def test_twelve_cameras_is_enough():
    s2.check_enough_cameras([f"cam{i:02d}" for i in range(1, 13)], 12)


def test_eleven_cameras_is_refused():
    with pytest.raises(s2.Refusal, match=">= 12"):
        s2.check_enough_cameras([f"cam{i:02d}" for i in range(1, 12)], 12)


def test_drop_rule_and_camera_count_compose():
    """Two drops out of thirteen leaves eleven, which the vote refuses."""
    areas = {f"cam{i:02d}": {"f50": 3000, "f92": 3000} for i in range(1, 14)}
    areas["cam01"]["f50"] = 10
    areas["cam02"]["f92"] = 10
    kept, dropped, _, _ = s2.apply_camera_drop_rule(areas)
    assert sorted(dropped) == ["cam01", "cam02"]
    assert len(kept) == 11
    with pytest.raises(s2.Refusal):
        s2.check_enough_cameras(kept, 12)


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
def _manifest():
    clicks = _filled()
    return s2.build_manifest(
        scene="crb",
        scene_root="/w/data_derived/absfix/crb",
        raster=[1352, 1014],
        frame_digits=4,
        frames_a=list(range(50, 110)),
        frames_b=list(range(92, 100)),
        checkpoint="/w/models/sam2/sam2.1_hiera_large.pt",
        checkpoint_sha256="0" * 64,
        config="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_commit="2b90b9f5ceec907a1c18123530e92e794ad901a4",
        clicks_file="/w/clicks.json",
        clicks_file_sha256="1" * 64,
        clicks={"crb": clicks},
        cameras_requested=["cam01", "cam02"],
        per_camera={"cam01": {"area_f50": 4000, "area_f92": 3900}},
        seed_area_medians={"f50": 4000.0, "f92": 3900.0},
        min_area=500,
        median_factor=5.0,
        dropped_cameras=["cam03"],
        drop_reasons={"cam03": ["f50: area 10 < 500 px"]},
        cameras_kept=["cam01", "cam02"],
        min_cameras=12,
        frame_source={"cam01": {"0050.png": "a", "0092.png": "b"}},
        mask_sha256={"cam01/0050.png": "2" * 64},
        environment={"torch": "2.5.1+cu121"},
    )


def test_manifest_has_every_required_field():
    m = _manifest()
    for field in s2.MANIFEST_FIELDS:
        assert field in m, f"manifest is missing {field}"


def test_manifest_records_provenance_and_rule_outcome():
    m = _manifest()
    assert m["checkpoint_sha256"] == "0" * 64
    assert m["config"] == "configs/sam2.1/sam2.1_hiera_l.yaml"
    assert m["sam2_commit"] == "2b90b9f5ceec907a1c18123530e92e794ad901a4"
    assert m["clicks"]["crb"]["cam01"]["f50"] == [100, 200]  # verbatim
    assert m["dropped_cameras"] == ["cam03"]
    assert m["n_cameras_kept"] == 2
    assert m["min_cameras"] == 12
    assert m["frames_a"] == [50, 109]
    assert m["frames_b"] == [92, 99]
    assert m["mask_sha256"]["cam01/0050.png"] == "2" * 64
    assert m["tool"] == "scripts/s2_sam2_propagate.py"


def test_manifest_is_json_serialisable():
    json.dumps(_manifest())


# --------------------------------------------------------------------------
# scene layout and camera selection
# --------------------------------------------------------------------------
def _scene(tmp_path, cams=("cam00", "cam01", "cam02"), frames=range(48, 56)):
    images = tmp_path / "images"
    images.mkdir(parents=True)
    for cam in cams:
        for f in frames:
            (images / f"{cam}_{f:04d}.png").write_bytes(b"")
    return tmp_path


def test_discover_scene_and_training_cameras(tmp_path):
    root = _scene(tmp_path)
    images, index, digits = s2.discover_scene(root)
    assert digits == 4
    assert sorted(index) == ["cam00", "cam01", "cam02"]
    assert s2.training_cameras(index) == ["cam01", "cam02"]  # cam00 held out
    assert s2.training_cameras(index, ["cam02"]) == ["cam02"]


def test_requesting_the_held_out_camera_is_refused(tmp_path):
    root = _scene(tmp_path)
    _, index, _ = s2.discover_scene(root)
    with pytest.raises(s2.Refusal, match="not training cameras"):
        s2.training_cameras(index, ["cam00"])


def test_scene_without_images_dir_is_refused(tmp_path):
    with pytest.raises(s2.Refusal, match="no images/"):
        s2.discover_scene(tmp_path)


def test_frame_name_matches_the_deva_naming():
    assert s2.frame_name(50, 4) == "0050.png"
    assert s2.frame_name(109, 4) == "0109.png"


def test_frame_range_and_backwards_refusal():
    assert s2.frame_range([92, 99]) == list(range(92, 100))
    with pytest.raises(s2.Refusal, match="backwards"):
        s2.frame_range([99, 92])


def test_sha256_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert s2.sha256_file(p) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
