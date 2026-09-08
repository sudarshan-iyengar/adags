"""Tests for scripts/realdata_membership_vote.py.

The module under test imports torch only inside functions, so everything here
runs on a workstation without torch, CUDA or a checkpoint. The handful of
tests that genuinely need torch use `pytest.importorskip` and are SKIPPED
rather than faked -- and two of them are the load-bearing ones: they check
that the emitted programs parse through the REAL consumer
(`elgs.trainer_hooks._load_episode_program_v2`) and that this module's numpy
voxel-key arithmetic is byte-for-byte the torch arithmetic
`resolve_v2_membership` reapplies at seeding.

Several tests are NEUTER tests: they fail if the line they guard is removed.
They say so in their docstrings, because a test that still passes after its
mechanism is deleted is worse than no test.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "realdata_membership_vote.py"


def _load_module():
    """Import by path so the test does not depend on `scripts` being a package."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "realdata_membership_vote_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rmv = _load_module()

from depth_visibility.errors import ContractError  # noqa: E402


# ---------------------------------------------------------------------------
# 1. specifications
# ---------------------------------------------------------------------------


def test_parse_int_ranges_expands_and_dedupes():
    assert rmv.parse_int_ranges("150-156,195-205") == list(range(150, 157)) + list(
        range(195, 206))
    assert rmv.parse_int_ranges("3, 1 ,3,2") == [1, 2, 3]
    assert rmv.parse_int_ranges("7") == [7]


def test_parse_int_ranges_refuses_descending_and_empty():
    with pytest.raises(ContractError):
        rmv.parse_int_ranges("10-3")
    with pytest.raises(ContractError):
        rmv.parse_int_ranges("   ")


def test_parse_camera_spec_all_and_subset():
    available = [1, 2, 3, 5]
    assert rmv.parse_camera_spec("all", available) == [1, 2, 3, 5]
    assert rmv.parse_camera_spec("ALL", available) == [1, 2, 3, 5]
    assert rmv.parse_camera_spec("5,1", available) == [1, 5]


def test_parse_camera_spec_refuses_a_camera_outside_the_train_split():
    with pytest.raises(ContractError):
        rmv.parse_camera_spec("0,1", [1, 2])


# ---------------------------------------------------------------------------
# 2. the anti-leakage contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [
    "/data/n3v/cut_roasted_beef/cam00/images/0150.png",
    "/sa4d/data/dynerf/cut_roasted_beef/cam00/pseudo_label/object_mask/0150.png",
    "/runs/x/gt_identity/cam03_f012.npy",
    "/runs/x/train_identity/cam03_f012.npy",
    "/data/synthetic/lrv3/event_spec.json",
    "configs/lrv3/oracle_correct.json",
    r"C:\runs\CAM00\images\0000.PNG",
])
def test_forbidden_paths_are_refused(path):
    assert rmv.is_forbidden_path(path) is True


@pytest.mark.parametrize("path", [
    "/data/n3v/cut_roasted_beef/cam01/images/0150.png",
    "/sa4d/data/dynerf/cut_roasted_beef/cam20/pseudo_label/object_mask/0195.png",
    "configs/n3v/ivv_protocol_300f_6k.yaml",
    "/runs/x/chkpnt6000.pth",
])
def test_allowed_paths_pass(path):
    assert rmv.is_forbidden_path(path) is False


def test_held_out_camera_filter_follows_the_declared_ids():
    """NEUTER: fails if the held-out ids stop reaching the path filter."""
    path = "/masks/cam07/pseudo_label/object_mask/0150.png"
    assert rmv.is_forbidden_path(path, held_out_cameras=(0,)) is False
    assert rmv.is_forbidden_path(path, held_out_cameras=(7,)) is True


def test_guard_blocks_forbidden_open_and_restores_builtins(tmp_path):
    """NEUTER: fails if `builtins.open` is not actually patched, or not restored."""
    original = builtins.open
    guard = rmv.HeldOutGuard(held_out_cameras=(0,))
    good = tmp_path / "cam01.txt"
    good.write_text("ok", encoding="utf-8")
    with guard:
        assert builtins.open is not original
        with open(good, "r", encoding="utf-8") as handle:
            assert handle.read() == "ok"
        with pytest.raises(rmv.LeakageError):
            open(str(tmp_path / "cam00.txt"), "r")
    assert builtins.open is original


def test_guard_disables_get_test_cameras_and_restores_it():
    class FakeScene:
        def getTestCameras(self):
            return ["held-out"]

    scene = FakeScene()
    guard = rmv.HeldOutGuard(scene=scene)
    with guard:
        with pytest.raises(rmv.LeakageError):
            scene.getTestCameras()
    # restored: the original bound method answers again (the guard rebinds it
    # on the instance, exactly as estimate_episodes.LeakageGuard does).
    assert scene.getTestCameras() == ["held-out"]
    assert guard.checks["get_test_cameras_disabled"] is True


def test_guard_train_only_is_identity_not_equality():
    """NEUTER: an equal-but-distinct camera must NOT pass.

    `Scene.getTrainCameras` copies the list but not the Camera objects, so an
    object drawn from the test split can only be caught by object identity.
    """
    class Cam:
        def __init__(self, name):
            self.image_name = name

        def __eq__(self, other):
            return getattr(other, "image_name", None) == self.image_name

        def __hash__(self):
            return hash(self.image_name)

    stack = [Cam("cam01_0150")]
    guard = rmv.HeldOutGuard()
    guard.assert_train_only(stack, stack)
    with pytest.raises(rmv.LeakageError):
        guard.assert_train_only([Cam("cam01_0150")], stack)


def test_guard_refuses_a_held_out_camera_id():
    guard = rmv.HeldOutGuard(held_out_cameras=(0,))
    guard.assert_no_held_out_ids([1, 2, 20])
    with pytest.raises(rmv.LeakageError):
        guard.assert_no_held_out_ids([0, 1])


def test_guard_refuses_a_nonempty_event_manifest():
    class Opt:
        event_candidate_manifest = "some/manifest.json"
        event_boundary_support_manifest = ""

    guard = rmv.HeldOutGuard(opt=Opt())
    with pytest.raises(rmv.LeakageError):
        guard.assert_manifests_empty()


# ---------------------------------------------------------------------------
# 3. DEVA id harmonization
# ---------------------------------------------------------------------------


def _toy_view():
    """A 4x6 id map with three ids, and a contribution map covering id 2 only.

    ids:                          S > 0.5 (marked #):
      0 0 0 0 0 0                   . . . . . .
      0 2 2 2 3 3                   . # # # . .
      0 2 2 2 3 3                   . # # . . .
      1 1 0 0 0 0                   . . . . . .

    id 2 has 6 pixels, 5 hot  -> fraction 0.8333  -> chosen at 0.5
    id 3 has 4 pixels, 0 hot  -> fraction 0.0     -> rejected
    id 1 has 2 pixels, 0 hot  -> rejected, and below a min-pixel bar of 4
    id 0 is the unlabelled sentinel and is never considered
    """
    id_map = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 3, 3],
        [0, 2, 2, 2, 3, 3],
        [1, 1, 0, 0, 0, 0],
    ], dtype=np.int64)
    s_map = np.zeros((4, 6), dtype=np.float32)
    s_map[1, 1:4] = 0.9
    s_map[2, 1:3] = 0.9
    return id_map, s_map


def test_choose_ids_picks_the_covered_id_only():
    id_map, s_map = _toy_view()
    chosen, stats = rmv.choose_ids_for_view(id_map, s_map, s_thresh=0.5,
                                            id_overlap=0.5, min_id_pixels=1)
    assert chosen == [2]
    by_id = {row["id"]: row for row in stats}
    assert set(by_id) == {1, 2, 3}
    assert by_id[2]["pixels"] == 6 and by_id[2]["pixels_over_s_thresh"] == 5
    assert by_id[2]["fraction"] == pytest.approx(5.0 / 6.0)
    assert by_id[3]["fraction"] == 0.0


def test_choose_ids_respects_the_overlap_bar():
    id_map, s_map = _toy_view()
    assert rmv.choose_ids_for_view(id_map, s_map, 0.5, 0.9, 1)[0] == []
    assert rmv.choose_ids_for_view(id_map, s_map, 0.5, 0.83, 1)[0] == [2]


def test_choose_ids_respects_the_min_pixel_bar():
    """NEUTER: with the bar removed a 2-pixel speck could be chosen."""
    id_map, s_map = _toy_view()
    s_map = s_map.copy()
    s_map[3, 0:2] = 0.9          # make id 1 (2 pixels) fully covered
    assert 1 in rmv.choose_ids_for_view(id_map, s_map, 0.5, 0.5, 1)[0]
    assert 1 not in rmv.choose_ids_for_view(id_map, s_map, 0.5, 0.5, 4)[0]


def test_choose_ids_thresholds_s_strictly():
    id_map = np.array([[5, 5]], dtype=np.int64)
    s_map = np.array([[0.5, 0.5]], dtype=np.float32)
    assert rmv.choose_ids_for_view(id_map, s_map, 0.5, 0.5, 1)[0] == []
    s_map = np.array([[0.51, 0.51]], dtype=np.float32)
    assert rmv.choose_ids_for_view(id_map, s_map, 0.5, 0.5, 1)[0] == [5]


def test_choose_ids_refuses_a_shape_mismatch():
    with pytest.raises(ContractError):
        rmv.choose_ids_for_view(np.zeros((2, 2), dtype=np.int64),
                                np.zeros((3, 2), dtype=np.float32),
                                0.5, 0.5, 1)


def test_aggregate_chosen_ids_pools_across_anchor_frames():
    per_frame = [[2, 4], [2], [2, 7]]
    keep, counts = rmv.aggregate_chosen_ids(per_frame, id_min_frames=1)
    assert keep == [2, 4, 7]
    assert counts == {"2": 3, "4": 1, "7": 1}
    assert rmv.aggregate_chosen_ids(per_frame, id_min_frames=2)[0] == [2]
    assert rmv.aggregate_chosen_ids(per_frame, id_min_frames=4)[0] == []


def test_mask_from_ids_is_the_union_and_empty_for_no_ids():
    id_map, _ = _toy_view()
    mask = rmv.mask_from_ids(id_map, [2, 3])
    assert int(mask.sum()) == 10
    assert mask.dtype == bool
    assert int(rmv.mask_from_ids(id_map, []).sum()) == 0


# ---------------------------------------------------------------------------
# 4. the vote
# ---------------------------------------------------------------------------


def test_vote_members_abstentions_and_the_tau_boundary():
    w_in = np.array([9.0, 1.0, 0.0, 5.0, 0.0])
    w_out = np.array([1.0, 9.0, 0.0, 5.0, 3.0])
    members, share, stats = rmv.membership_vote(w_in, w_out, tau=0.5)
    # row 0 clearly in; row 1 clearly out; row 2 has NO weight at all and
    # abstains; row 3 sits exactly at tau; row 4 is observed but all outside.
    assert members.tolist() == [True, False, False, True, False]
    assert share.tolist() == [0.9, 0.1, 0.0, 0.5, 0.0]
    assert stats["n_eligible"] == 4
    assert stats["n_abstained_ineligible"] == 1
    assert stats["n_members"] == 2
    assert stats["n_members_strict_gt_tau"] == 1
    assert stats["n_rows_exactly_at_tau"] == 1
    assert stats["e_min"] == 0.0


def test_vote_zero_weight_row_never_becomes_a_member_even_at_tau_zero():
    """NEUTER: fails if eligibility stops being `w_in + w_out > 0`.

    At tau = 0 the share test alone admits everything, so only the strict
    e_min = 0 eligibility keeps an unobserved row out.
    """
    members, _, stats = rmv.membership_vote([0.0, 1.0], [0.0, 0.0], tau=0.0)
    assert members.tolist() == [False, True]
    assert stats["n_abstained_ineligible"] == 1


def test_vote_refuses_negative_weights_and_a_bad_tau():
    with pytest.raises(ContractError):
        rmv.membership_vote([-1.0], [1.0])
    with pytest.raises(ContractError):
        rmv.membership_vote([1.0], [-1.0])
    with pytest.raises(ContractError):
        rmv.membership_vote([1.0], [1.0], tau=1.5)
    with pytest.raises(ContractError):
        rmv.membership_vote([1.0, 2.0], [1.0])


def test_jaccard():
    a = np.array([True, True, False, False])
    b = np.array([True, False, False, True])
    assert rmv.jaccard(a, b) == pytest.approx(1.0 / 3.0)
    assert rmv.jaccard(a, a) == 1.0
    assert rmv.jaccard(np.zeros(3, bool), np.zeros(3, bool)) == 1.0


# ---------------------------------------------------------------------------
# 5. the fine grid
# ---------------------------------------------------------------------------


def test_padded_bbox_pads_both_sides_by_a_fraction_of_the_extent():
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 4.0, 2.0]])
    lo, span = rmv.padded_bbox(pts, 0.05)
    assert lo == pytest.approx([-0.5, -0.2, -0.1])
    assert span == pytest.approx([11.0, 4.4, 2.2])
    assert bool(np.all(rmv.voxel_inside(pts, lo, span)))


def test_padded_bbox_gives_a_degenerate_axis_a_positive_span():
    """`resolve_v2_membership` refuses a non-positive span, so this must not
    emit one even when every member shares a coordinate."""
    pts = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    lo, span = rmv.padded_bbox(pts, 0.05)
    assert bool(np.all(span > 0))
    assert bool(np.all(rmv.voxel_inside(pts, lo, span)))


def test_padded_bbox_refuses_an_empty_member_set():
    with pytest.raises(ContractError):
        rmv.padded_bbox(np.zeros((0, 3)), 0.05)


def test_voxel_keys_match_the_documented_formula_and_clamp():
    lo = np.array([0.0, 0.0, 0.0])
    span = np.array([1.0, 1.0, 1.0])
    cells = 4
    pts = np.array([
        [0.0, 0.0, 0.0],        # cell (0,0,0)
        [0.99, 0.0, 0.0],       # cell (3,0,0)
        [0.0, 0.5, 0.26],       # cell (0,2,1)
        [1.0, 1.0, 1.0],        # clamped to (3,3,3)
        [-5.0, -5.0, -5.0],     # clamped to (0,0,0)
    ])
    keys = rmv.voxel_keys(pts, lo, span, cells)
    assert keys.tolist() == [0, 3 * 16, 2 * 4 + 1, 3 * 16 + 3 * 4 + 3, 0]


def test_voxel_keys_are_unique_per_cell_at_the_fine_resolution():
    rng = np.random.default_rng(0)
    pts = rng.random((500, 3)).astype(np.float32)
    keys = rmv.voxel_keys(pts, np.zeros(3), np.ones(3), 64)
    manual = np.clip((pts * np.float32(64)), 0, 63).astype(np.int64)
    expected = manual[:, 0] * 64 * 64 + manual[:, 1] * 64 + manual[:, 2]
    assert keys.tolist() == expected.tolist()
    assert int(keys.max()) < 64 ** 3


def test_voxel_keys_are_computed_in_float32_like_the_seeding_grid():
    """NEUTER: fails if the arithmetic drifts back to float64.

    `resolve_v2_membership` casts lo/span to the cloud's float32, so a value
    that lands on a cell boundary in float64 but not in float32 would be
    gated differently at seeding than it was voted here.
    """
    # 0.1 + 0.2 is representable differently in the two precisions; scaled by
    # a large cell count the truncated cell index can differ.
    lo = np.array([0.0, 0.0, 0.0])
    span = np.array([1.0, 1.0, 1.0])
    pts = np.array([[np.float64(np.float32(0.7)), 0.0, 0.0]])
    f32 = rmv.voxel_keys(pts, lo, span, 64)
    manual = int(np.clip(np.float32(0.7) * np.float32(64), 0, 63))
    assert int(f32[0]) == manual * 64 * 64


def test_frame_index_of_name():
    assert rmv.frame_index_of_name("cam07_0150") == 150
    assert rmv.frame_index_of_name("cam00_0000") == 0
    assert rmv.frame_index_of_name("cam07") is None
    assert rmv.frame_index_of_name("") is None


def test_frame_index_refuses_a_timestamp_name_disagreement():
    """NEUTER: without this the renders would be paired with the wrong masks.

    The DEVA mask filenames follow the image NAME's 4-digit index, while the
    frame index used for lookup comes from the timestamp.
    """
    class Cam:
        def __init__(self, name, timestamp):
            self.image_name = name
            self.timestamp = timestamp

    frame, checked = rmv.frame_index_of(Cam("cam07_0150", 150.0 / 30.0), 1.0 / 30.0)
    assert (frame, checked) == (150, True)
    frame, checked = rmv.frame_index_of(Cam("cam07", 150.0 / 30.0), 1.0 / 30.0)
    assert (frame, checked) == (150, False)
    with pytest.raises(ContractError):
        # a halved frame_ratio, or any other timestamp rescaling
        rmv.frame_index_of(Cam("cam07_0150", 75.0 / 30.0), 1.0 / 30.0)


def test_voxel_keys_refuse_a_nonpositive_span():
    with pytest.raises(ContractError):
        rmv.voxel_keys(np.zeros((1, 3)), np.zeros(3), np.array([1.0, 0.0, 1.0]), 8)


def test_voxel_inside_excludes_rows_outside_the_originating_box():
    lo = np.zeros(3)
    span = np.ones(3)
    pts = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [-0.01, 0.5, 0.5],
                    [0.0, 1.0, 0.0]])
    assert rmv.voxel_inside(pts, lo, span).tolist() == [True, False, False, True]


# ---------------------------------------------------------------------------
# 6. the gap convention and the fingerprint
# ---------------------------------------------------------------------------


def test_inset_gap_seconds_matches_the_frozen_convention():
    # LRV3's recorded program: offset 30, onset 57, dt = 1/6, w = 2*dt.
    dt = 1.0 / 6.0
    w = 2.0 * dt
    start, end = rmv.inset_gap_seconds(30, 57, dt, w)
    assert start == pytest.approx(29 * dt + w)
    assert end == pytest.approx(57 * dt - w)
    assert start == pytest.approx(31.0 / 6.0)
    assert end == pytest.approx(55.0 / 6.0)


def test_inset_gap_seconds_on_the_n3v_event_window():
    """`--gap_frames 158 187` means absent 158..187 inclusive, so the last
    present frame is 157 and the first present frame after the gap is 188."""
    dt = 1.0 / 30.0
    w = 2.0 * dt
    start, end = rmv.inset_gap_seconds(158, 188, dt, w)
    assert start == pytest.approx(159.0 / 30.0)
    assert end == pytest.approx(186.0 / 30.0)
    assert end > start


def test_cloud_fingerprint_is_sha256_of_the_float32_bytes():
    xyz = np.arange(12, dtype=np.float32).reshape(4, 3)
    expected = hashlib.sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest()
    assert rmv.cloud_fingerprint_from_array(xyz) == expected
    # float64 input must be cast, not hashed as-is.
    assert rmv.cloud_fingerprint_from_array(xyz.astype(np.float64)) == expected


# ---------------------------------------------------------------------------
# 7. program shape
# ---------------------------------------------------------------------------


def _program(mode, n_rows=6, members=(0, 1, 4)):
    column = [rmv.EMITTED_GROUP_ID if i in set(members) else -1
              for i in range(n_rows)]
    return rmv.build_v2_program(
        mode, column, n_rows, "deadbeef" * 8,
        lo=[0.0, 0.0, 0.0], span=[1.0, 1.0, 1.0], cells_per_axis=64,
        group_cell_keys=[7, 9, 11],
        gap_seconds=(5.3, 6.2), offset_frame=158, onset_frame=188,
        rows_at_estimation=len(members), w=2.0 / 30.0, frame_dt=1.0 / 30.0,
        source={"checkpoint": "chkpnt6000.pth"})


def test_row_ids_program_shape():
    payload, digest = _program("row_ids")
    assert payload["schema_version"] == "adags-episode-program-v2"
    assert payload["units"] == "model_time_seconds"
    assert payload["membership_mode"] == "row_ids"
    assert payload["cloud"]["n_rows"] == 6
    assert len(payload["row_group_ids"]) == 6
    assert payload["row_group_ids"] == [1, 1, -1, -1, 1, -1]
    assert payload["groups"][0]["group"] == 1
    assert payload["groups"][0]["gaps"] == [[5.3, 6.2]]
    assert payload["groups"][0]["offset_frame"] == 158
    assert payload["groups"][0]["onset_frame"] == 188
    assert payload["groups"][0]["rows_at_estimation"] == 3
    # the spatial block is written in BOTH modes
    assert payload["spatial"]["kind"] == "voxel_grid"
    assert payload["spatial"]["cells_per_axis"] == 64
    assert payload["spatial"]["group_cell_keys"] == {"1": [7, 9, 11]}
    assert len(digest) == 64


def test_spatial_program_carries_no_row_column():
    payload, _ = _program("spatial_voxel")
    assert payload["membership_mode"] == "spatial_voxel"
    assert "row_group_ids" not in payload
    assert payload["spatial"]["lo"] == [0.0, 0.0, 0.0]
    assert payload["spatial"]["span"] == [1.0, 1.0, 1.0]


def test_program_hash_is_content_addressed():
    a, ha = _program("row_ids")
    b, hb = _program("row_ids")
    assert ha == hb
    _, hc = _program("row_ids", members=(0, 1))
    assert hc != ha
    assert ha == hashlib.sha256(
        json.dumps(a, sort_keys=True).encode("utf-8")).hexdigest()


def test_program_refuses_an_unknown_mode_and_a_short_row_column():
    with pytest.raises(ContractError):
        rmv.build_v2_program(
            "voxel", [1], 1, "x", [0, 0, 0], [1, 1, 1], 8, [0], (0.1, 0.2),
            1, 2, 1, 0.1, 0.1)
    with pytest.raises(ContractError):
        rmv.build_v2_program(
            "row_ids", [1, -1], 5, "x", [0, 0, 0], [1, 1, 1], 8, [0],
            (0.1, 0.2), 1, 2, 1, 0.1, 0.1)


# ---------------------------------------------------------------------------
# 8. seed resolution
# ---------------------------------------------------------------------------


class _Args:
    def __init__(self, **kwargs):
        self.seed_program = ""
        self.seed_groups = ""
        self.seed_bbox3d = None
        for key, value in kwargs.items():
            setattr(self, key, value)


XYZ = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])


def test_seed_from_bbox_is_recorded_as_authored():
    seed, prov = rmv.resolve_seed_rows(
        _Args(seed_bbox3d=[-0.5, -0.5, -0.5, 2.0, 2.0, 2.0]), XYZ, "fp")
    assert seed.tolist() == [True, True, False]
    assert prov["kind"] == "seed_bbox3d"
    assert prov["authored"] is True
    assert prov["seed_rows"] == 2


def test_seed_requires_exactly_one_source():
    with pytest.raises(ContractError):
        rmv.resolve_seed_rows(_Args(), XYZ, "fp")
    with pytest.raises(ContractError):
        rmv.resolve_seed_rows(
            _Args(seed_program="p.json", seed_bbox3d=[0, 0, 0, 1, 1, 1]),
            XYZ, "fp")


def test_seed_bbox_refuses_a_degenerate_box_and_an_empty_selection():
    with pytest.raises(ContractError):
        rmv.resolve_seed_rows(_Args(seed_bbox3d=[1, 0, 0, 0, 1, 1]), XYZ, "fp")
    with pytest.raises(ContractError):
        rmv.resolve_seed_rows(
            _Args(seed_bbox3d=[100, 100, 100, 101, 101, 101]), XYZ, "fp")


def _write_seed_program(tmp_path, **overrides):
    payload = {
        "schema_version": "adags-episode-program-v2",
        "membership_mode": "row_ids",
        "cloud": {"n_rows": 3, "xyz_sha256": "fp"},
        "row_group_ids": [7, -1, 8],
        "groups": [{"group": 7}, {"group": 8}],
    }
    payload.update(overrides)
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_seed_from_program_selects_the_gated_rows(tmp_path):
    path = _write_seed_program(tmp_path)
    seed, prov = rmv.resolve_seed_rows(_Args(seed_program=path), XYZ, "fp")
    assert seed.tolist() == [True, False, True]
    assert prov["kind"] == "seed_program"
    assert prov["authored"] is False
    assert prov["groups_used"] == [7, 8]
    assert len(prov["program_sha256"]) == 64


def test_seed_program_group_subset(tmp_path):
    path = _write_seed_program(tmp_path)
    seed, prov = rmv.resolve_seed_rows(
        _Args(seed_program=path, seed_groups="8"), XYZ, "fp")
    assert seed.tolist() == [False, False, True]
    assert prov["groups_used"] == [8]
    with pytest.raises(ContractError):
        rmv.resolve_seed_rows(_Args(seed_program=path, seed_groups="9"),
                              XYZ, "fp")


def test_seed_program_fails_closed_on_a_fingerprint_or_row_count_mismatch(tmp_path):
    """NEUTER: fails if either binding check is dropped.

    A seed column bound to the wrong cloud would silently select the wrong
    primitives, and every downstream number would be about other rows.
    """
    path = _write_seed_program(tmp_path)
    with pytest.raises(ContractError):
        rmv.resolve_seed_rows(_Args(seed_program=path), XYZ, "OTHER")
    other = _write_seed_program(
        tmp_path, cloud={"n_rows": 9, "xyz_sha256": "fp"})
    with pytest.raises(ContractError):
        rmv.resolve_seed_rows(_Args(seed_program=other), XYZ, "fp")


def test_seed_program_refuses_a_spatial_program_or_a_foreign_schema(tmp_path):
    spatial = _write_seed_program(tmp_path, membership_mode="spatial_voxel")
    with pytest.raises(ContractError):
        rmv.resolve_seed_rows(_Args(seed_program=spatial), XYZ, "fp")
    foreign = _write_seed_program(tmp_path, schema_version="something-else")
    with pytest.raises(ContractError):
        rmv.resolve_seed_rows(_Args(seed_program=foreign), XYZ, "fp")


# ---------------------------------------------------------------------------
# 9. torch-dependent: the real consumer, and grid equivalence
# ---------------------------------------------------------------------------


def _interval_config(torch_mod):
    from elgs.intervals import IntervalConfig

    dt = 1.0 / 30.0
    w = 2.0 * dt
    return IntervalConfig(
        T=299 * dt, w_m=2.0 * dt, w=w,
        floor_len=2.0 * w + dt, floor_gap=2.0 * w + dt, delta_tol=0.1 * dt)


@pytest.mark.parametrize("mode", ["row_ids", "spatial_voxel"])
def test_emitted_program_parses_through_the_real_consumer(mode):
    """The load-bearing validation: both programs must satisfy
    `_load_episode_program_v2`, which is the function `seed_families` calls."""
    torch = pytest.importorskip("torch")
    from elgs.trainer_hooks import _load_episode_program_v2

    dt = 1.0 / 30.0
    payload, _ = rmv.build_v2_program(
        mode, [1, -1, 1], 3, "a" * 64, lo=[0.0, 0.0, 0.0], span=[1.0, 1.0, 1.0],
        cells_per_axis=64, group_cell_keys=[0, 5],
        gap_seconds=rmv.inset_gap_seconds(158, 188, dt, 2.0 * dt),
        offset_frame=158, onset_frame=188, rows_at_estimation=2,
        w=2.0 * dt, frame_dt=dt)
    loaded = _load_episode_program_v2(payload, _interval_config(torch))
    assert loaded.membership_mode == mode
    assert sorted(loaded.intervals) == [rmv.EMITTED_GROUP_ID]


def test_numpy_voxel_keys_equal_resolve_v2_membership_arithmetic():
    """NEUTER: fails if this module's grid drifts from the seeding grid.

    `resolve_v2_membership` reapplies the grid in torch at seeding time. If
    the numpy keys emitted here disagree by even one cell, the program gates
    different rows than the ones voted for.
    """
    torch = pytest.importorskip("torch")
    from elgs.trainer_hooks import _load_episode_program_v2, resolve_v2_membership

    rng = np.random.default_rng(7)
    xyz = rng.random((200, 3)).astype(np.float32) * 3.0 - 1.0
    members = np.zeros(200, dtype=bool)
    members[:40] = True
    lo, span = rmv.padded_bbox(xyz[members].astype(np.float64), 0.05)
    keys = rmv.voxel_keys(xyz.astype(np.float64), lo, span, 64)
    member_keys = sorted(set(int(k) for k in keys[members]))
    inside = rmv.voxel_inside(xyz.astype(np.float64), lo, span)
    expected = np.logical_and(
        inside, np.isin(keys, np.asarray(member_keys, dtype=np.int64)))

    dt = 1.0 / 30.0
    payload, _ = rmv.build_v2_program(
        "spatial_voxel", None, 200, "b" * 64, lo=lo, span=span,
        cells_per_axis=64, group_cell_keys=member_keys,
        gap_seconds=rmv.inset_gap_seconds(158, 188, dt, 2.0 * dt),
        offset_frame=158, onset_frame=188, rows_at_estimation=int(members.sum()),
        w=2.0 * dt, frame_dt=dt)
    program = _load_episode_program_v2(payload, _interval_config(torch))
    column = resolve_v2_membership(program, torch.from_numpy(xyz))
    got = (column.cpu().numpy() == rmv.EMITTED_GROUP_ID)
    assert got.tolist() == expected.tolist()
    # every voted member is gated; the extra rows are the cell-quantization cost
    assert bool(np.all(got[members]))


def test_row_ids_program_fails_closed_on_a_foreign_cloud():
    """NEUTER: this is the property that makes `row_ids` safe to emit.

    A fresh `create_from_pcd` run never reproduces the trained cloud, so the
    row column must refuse rather than bind to the wrong primitives.
    """
    torch = pytest.importorskip("torch")
    from elgs.trainer_hooks import _load_episode_program_v2, resolve_v2_membership

    xyz = np.zeros((3, 3), dtype=np.float32)
    fingerprint = rmv.cloud_fingerprint_from_array(xyz)
    dt = 1.0 / 30.0
    payload, _ = rmv.build_v2_program(
        "row_ids", [1, -1, 1], 3, fingerprint, lo=[-1.0, -1.0, -1.0],
        span=[2.0, 2.0, 2.0], cells_per_axis=64, group_cell_keys=[0],
        gap_seconds=rmv.inset_gap_seconds(158, 188, dt, 2.0 * dt),
        offset_frame=158, onset_frame=188, rows_at_estimation=2,
        w=2.0 * dt, frame_dt=dt)
    program = _load_episode_program_v2(payload, _interval_config(torch))
    column = resolve_v2_membership(program, torch.from_numpy(xyz))
    assert column.cpu().numpy().tolist() == [1, -1, 1]
    with pytest.raises(Exception):
        resolve_v2_membership(program, torch.ones((3, 3), dtype=torch.float32))
    with pytest.raises(Exception):
        resolve_v2_membership(program, torch.zeros((4, 3), dtype=torch.float32))


# ---------------------------------------------------------------------------
# 10. the CLI declares what the config merge needs
# ---------------------------------------------------------------------------


def test_parser_accepts_every_top_level_key_of_the_target_config():
    """NEUTER: `_merge_config` asserts every YAML key exists on `args`, so a
    missing scene-shape flag would only surface at submit time on Leonardo."""
    yaml = pytest.importorskip("yaml")
    parser, _, _, _ = rmv.build_parser()
    args = parser.parse_args([
        "--config", "x", "--start_checkpoint", "y", "--out_report", "z",
        "--mask_root", "m"])
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "n3v" / "ivv_protocol_300f_6k.yaml")
        .read_text(encoding="utf-8"))

    def walk(host):
        for key, value in host.items():
            if isinstance(value, dict):
                walk(value)
            else:
                assert hasattr(args, key), "parser is missing %s" % key

    walk(config)
