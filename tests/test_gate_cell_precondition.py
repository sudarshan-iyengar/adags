"""CPU tests for scripts/gate_cell_precondition.py.

No torch, no CUDA, no checkpoint. The script keeps every heavy import inside
`main()`, so its log reading rule, its box test, its frame arithmetic and the
shape of the object it writes are all exercised here directly.

What is deliberately NOT tested: that `setup_elgs` takes the restore branch on
a real gated checkpoint, that the projection matches a render, and that
`build_reserved_pool` returns the count the trainer used. All three need a GPU
and a trained cell; the script asserts them at runtime instead (the
`_pending_elgs_state` arm check, the `get_elgs_gated_row_mask` cross-read, and
the three-way reserved-unit comparison).

Run with:
    python -m pytest tests/test_gate_cell_precondition.py -q
"""

import ast
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "gate_cell_precondition.py"

from scripts import gate_cell_precondition as GP  # noqa: E402
from scripts import realdata_gate_analysis as RGA  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic run logs. Both carry tqdm noise on the same physical line as the
# JSON, because scripts/run_leonardo.sh tees stdout AND stderr into one file.
# ---------------------------------------------------------------------------

GATED_SEEDING = {
    "families": 3,
    "gated_rows": 4317,
    "iteration": 3000,
    "local_presence": True,
    "oracle_episodes": "configs/n3v/crb300_program_spatial.json",
    "oracle_K": None,
    "oracle_families": 3,
    "oracle_rows": 0,
    "program_schema": "adags-episode-program-v2",
    "rows": 412_233,
    "routing_pins_enabled": False,
    "unassigned_rows": 407_916,
    "v2_group_K": {"7": 2},
    "v2_group_rows": {"7": 4317},
    "v2_membership_mode": "spatial_voxel",
}


def gated_log(seeding=None):
    payload = json.dumps({"elgs_seeding": seeding or GATED_SEEDING},
                         sort_keys=True)
    return "\n".join([
        "Loading Training Cameras",
        json.dumps({"elgs_setup": {"families": 3, "frame_dt": 0.03333,
                                   "restored": False}}, sort_keys=True),
        "Training progress:  25%|##5    | 3000/12000 [10:11<30:33]" + payload,
        "Training progress: 100%|#######| 12000/12000",
    ]) + "\n"


def ungated_log(reserved=1425, after=4275):
    payload = json.dumps(
        {"elgs_reserved_parity": {"reserved_units": reserved,
                                  "training_units_after": after}},
        sort_keys=True,
    )
    return "\n".join([
        "Loading Training Cameras",
        payload,
        "Training progress: 100%|#######| 12000/12000",
    ]) + "\n"


class LogParsing(unittest.TestCase):
    def test_gated_arm_seeding_line(self):
        seeding = GP.read_unique_log_object(gated_log(), "elgs_seeding")
        self.assertIsNotNone(seeding)
        self.assertEqual(
            GP.require_int(seeding, "gated_rows", "elgs_seeding"), 4317
        )
        self.assertEqual(GP.require_int(seeding, "rows", "elgs_seeding"), 412_233)
        # A gated log carries no parity line: reserved_indices_for_parity
        # returns None whenever elgs_enable is set, so main.py:1237 never fires.
        self.assertIsNone(
            GP.read_unique_log_object(gated_log(), "elgs_reserved_parity")
        )

    def test_ungated_arm_parity_line(self):
        parity = GP.read_unique_log_object(ungated_log(), "elgs_reserved_parity")
        self.assertEqual(
            GP.require_int(parity, "reserved_units", "elgs_reserved_parity"), 1425
        )
        self.assertEqual(
            GP.require_int(parity, "training_units_after", "elgs_reserved_parity"),
            4275,
        )
        # The ungated arm never seeds.
        self.assertIsNone(
            GP.read_unique_log_object(ungated_log(), "elgs_seeding")
        )

    def test_json_survives_a_progress_bar_on_the_same_line(self):
        found = GP.extract_json_log_objects(gated_log(), "elgs_seeding")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["gated_rows"], 4317)

    def test_braces_inside_a_string_do_not_end_the_object(self):
        note = 'a } brace, a \\" quote and a { brace'
        text = "noise" + json.dumps(
            {"elgs_seeding": {"gated_rows": 5, "note": note}}
        ) + "trailing }"
        found = GP.extract_json_log_objects(text, "elgs_seeding")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["gated_rows"], 5)

    def test_identical_repeats_are_accepted(self):
        text = gated_log() + gated_log()
        seeding = GP.read_unique_log_object(text, "elgs_seeding")
        self.assertEqual(seeding["gated_rows"], 4317)

    def test_two_differing_runs_in_one_log_are_a_refusal(self):
        other = dict(GATED_SEEDING, gated_rows=11)
        text = gated_log() + gated_log(other)
        with self.assertRaises(ValueError) as caught:
            GP.read_unique_log_object(text, "elgs_seeding", "meta/train.log")
        self.assertIn("more than one run", str(caught.exception))

    def test_truncated_object_is_a_refusal(self):
        text = '{"elgs_seeding": {"gated_rows": 4317, '
        with self.assertRaises(ValueError) as caught:
            GP.extract_json_log_objects(text, "elgs_seeding")
        self.assertIn("truncated", str(caught.exception))

    def test_missing_field_names_the_line(self):
        with self.assertRaises(ValueError) as caught:
            GP.require_int({"rows": 3}, "gated_rows", "elgs_seeding", "x.log")
        self.assertIn("gated_rows", str(caught.exception))
        self.assertIn("elgs_seeding", str(caught.exception))


class BoxTest(unittest.TestCase):
    BOX = GP.DEFAULT_FBOX  # 664, 912, 744, 976

    def test_corners_are_inside(self):
        for x, y in ((664, 912), (744, 976), (664, 976), (744, 912)):
            self.assertTrue(GP.point_in_box(x, y, self.BOX), (x, y))

    def test_one_pixel_outside_each_edge(self):
        for x, y in ((663, 940), (745, 940), (700, 911), (700, 977)):
            self.assertFalse(GP.point_in_box(x, y, self.BOX), (x, y))

    def test_interior(self):
        self.assertTrue(GP.point_in_box(700, 940, self.BOX))

    def test_degenerate_box_is_rejected(self):
        with self.assertRaises(ValueError):
            GP.point_in_box(0, 0, (744, 912, 664, 976))

    def test_box_matches_the_frozen_f_box(self):
        # research-wiki/operations/realdata-gating-lane-2026-09-09.md section 2
        self.assertEqual(tuple(GP.DEFAULT_FBOX), (664, 912, 744, 976))
        x0, y0, x1, y1 = GP.DEFAULT_FBOX
        self.assertEqual((x1 - x0, y1 - y0), (80, 64))


class PresenceZeroFrames(unittest.TestCase):
    """Clause (d) on the lane's own window."""

    DT = 1.0 / 30.0

    def test_the_ghost_window(self):
        gaps = {7: [[158 / 30.0, 187 / 30.0]]}
        frames = GP.frames_with_presence_zero(gaps, self.DT, 300)
        self.assertEqual(frames[0], 158)
        self.assertEqual(frames[-1], 187)
        self.assertEqual(len(frames), 30)

    def test_endpoints_are_included_because_smoothstep_zero_is_zero(self):
        gaps = {0: [[10 / 30.0, 12 / 30.0]]}
        self.assertEqual(
            GP.frames_with_presence_zero(gaps, self.DT, 300), [10, 11, 12]
        )

    def test_two_groups_union_without_double_counting(self):
        gaps = {0: [[10 / 30.0, 12 / 30.0]], 1: [[11 / 30.0, 13 / 30.0]]}
        self.assertEqual(
            GP.frames_with_presence_zero(gaps, self.DT, 300), [10, 11, 12, 13]
        )

    def test_the_mis_specified_window(self):
        # G-mis gates [118, 147] -- same duration, wrong time.
        gaps = {7: [[118 / 30.0, 147 / 30.0]]}
        frames = GP.frames_with_presence_zero(gaps, self.DT, 300)
        self.assertEqual((frames[0], frames[-1], len(frames)), (118, 147, 30))

    def test_supplied_timestamps_win_over_the_idealized_clock(self):
        gaps = {0: [[1.0, 2.0]]}
        # frame 5's real timestamp lands inside the gap although 5*dt does not
        supplied = {5: 1.5}
        self.assertEqual(
            GP.frames_with_presence_zero(gaps, self.DT, 10, supplied), [5]
        )

    def test_the_grid_uses_real_timestamps_where_given(self):
        grid = GP.frame_time_grid(self.DT, 3, {1: 99.0})
        self.assertEqual(grid[0], (0, 0.0))
        self.assertEqual(grid[1], (1, 99.0))
        self.assertAlmostEqual(grid[2][1], 2 * self.DT)

    def test_a_program_the_window_never_reaches(self):
        gaps = {0: [[400.0, 401.0]]}
        self.assertEqual(GP.frames_with_presence_zero(gaps, self.DT, 300), [])

    def test_bad_clock_is_rejected(self):
        with self.assertRaises(ValueError):
            GP.frame_time_grid(0.0, 300)
        with self.assertRaises(ValueError):
            GP.frame_time_grid(self.DT, 0)


class ConsumerContract(unittest.TestCase):
    """The written object must be exactly what realdata_gate_analysis reads."""

    def test_field_set_matches_the_frozen_spec(self):
        self.assertEqual(
            tuple(RGA.SPEC["PRECONDITION_FIELDS"]), GP.PRECONDITION_FIELDS
        )

    def _payload(self, **overrides):
        base = dict(
            arm_kind="gated",
            gated_rows_seeding=4317,
            n_rows_seeding=412_233,
            gated_rows_final=4576,
            n_rows_final=598_112,
            gated_rows_fbox_frame150=1_204,
            frames_presence_zero=30,
            reserved_units=1425,
            training_units_total=5700,
            detail={},
            provenance={},
        )
        base.update(overrides)
        return GP.assemble_precondition(**base)

    def test_every_spec_field_is_top_level_and_survives_a_round_trip(self):
        payload = self._payload()
        loaded = json.loads(json.dumps(payload))
        for field in RGA.SPEC["PRECONDITION_FIELDS"]:
            self.assertIn(field, loaded, field)

    def test_read_precondition_recovers_all_eight_fields(self, ):
        import tempfile

        payload = self._payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / RGA.SPEC["precondition_filename"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            read = RGA.read_precondition(tmp)
        self.assertEqual(
            sorted(read), sorted(RGA.SPEC["PRECONDITION_FIELDS"])
        )
        self.assertEqual(read["gated_rows_final"], 4576)
        self.assertEqual(read["reserved_units"], 1425)

    def test_a_healthy_gated_cell_passes_mechanism_exercised(self):
        payload = self._payload()
        passes, reason = RGA.mechanism_exercised("G", payload)
        self.assertTrue(passes, reason)

    def test_each_frozen_minimum_can_fail_the_cell(self):
        cases = {
            "gated_rows_final": RGA.SPEC["MIN_GATED_SURVIVING"] - 1,
            "gated_rows_fbox_frame150": RGA.SPEC["MIN_GATED_FBOX"] - 1,
            "frames_presence_zero": RGA.SPEC["MIN_FRAMES_PRESENCE_ZERO"] - 1,
        }
        for field, value in cases.items():
            passes, reason = RGA.mechanism_exercised("G", self._payload(**{field: value}))
            self.assertFalse(passes, field)
            self.assertIn(field, reason)

    def test_an_ungated_cell_writes_zeros_and_a_null_seeding_count(self):
        payload = self._payload(
            arm_kind="ungated", gated_rows_seeding=0, n_rows_seeding=None,
            gated_rows_final=0, gated_rows_fbox_frame150=0,
            frames_presence_zero=0,
        )
        # U passes by construction and the analysis never reads the counters.
        passes, _ = RGA.mechanism_exercised("U", payload)
        self.assertTrue(passes)
        self.assertIsNone(payload["n_rows_seeding"])
        self.assertEqual(payload["reserved_units"], 1425)

    def test_reserved_unit_check_sees_the_pair(self):
        cells = [
            {"arm": "G", "seed": 0, "precondition": self._payload()},
            {"arm": "U", "seed": 0,
             "precondition": self._payload(arm_kind="ungated")},
        ]
        out = RGA.reserved_unit_check(cells)
        self.assertTrue(out["consistent"], out.get("message"))
        self.assertEqual(out["reserved_units"], 1425)
        self.assertEqual(out["training_units_total"], 5700)

    def test_a_disagreeing_reserved_count_is_caught_by_the_consumer(self):
        cells = [
            {"arm": "G", "seed": 0, "precondition": self._payload()},
            {"arm": "U", "seed": 0,
             "precondition": self._payload(reserved_units=0)},
        ]
        out = RGA.reserved_unit_check(cells)
        self.assertFalse(out["consistent"])

    def test_a_missing_field_is_a_refusal_not_a_silent_null(self):
        with self.assertRaises(TypeError):
            GP.assemble_precondition(  # noqa: F841 - missing reserved_units
                arm_kind="gated", gated_rows_seeding=1, n_rows_seeding=2,
                gated_rows_final=3, n_rows_final=4,
                gated_rows_fbox_frame150=5, frames_presence_zero=6,
                training_units_total=7, detail={}, provenance={},
            )


class CfgArgs(unittest.TestCase):
    def test_source_path_is_recovered(self):
        text = (
            "Namespace(sh_degree=3, source_path='/work/data/n3v/"
            "cut_roasted_beef', model_path='/work/runs/x', eval=True)"
        )
        self.assertEqual(
            GP.source_path_from_cfg_args(text),
            "/work/data/n3v/cut_roasted_beef",
        )

    def test_unreadable_cfg_args_returns_none_rather_than_raising(self):
        self.assertIsNone(GP.source_path_from_cfg_args("not a namespace ("))
        self.assertIsNone(GP.source_path_from_cfg_args("Namespace(eval=True)"))
        self.assertIsNone(GP.source_path_from_cfg_args("42"))


class StaticShape(unittest.TestCase):
    """The pure helpers must stay importable without torch."""

    def _tree(self):
        return ast.parse(SCRIPT.read_text(encoding="utf-8"))

    def test_no_heavy_import_at_module_level(self):
        heavy = {"torch", "numpy", "scene", "gaussian_renderer", "arguments",
                 "elgs", "utils", "yaml", "depth_visibility"}
        for node in self._tree().body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name.split(".")[0], heavy, alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                self.assertNotIn(node.module.split(".")[0], heavy, node.module)

    def test_the_script_is_syntactically_valid_and_has_a_main(self):
        names = {
            node.name for node in self._tree().body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("main", names)
        for helper in ("extract_json_log_objects", "read_unique_log_object",
                       "point_in_box", "frames_with_presence_zero",
                       "assemble_precondition"):
            self.assertIn(helper, names, helper)

    def test_it_writes_exactly_one_file_into_the_run_dir(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('run_dir / "precondition.json"', source)
        self.assertIn('run_dir / "precondition_scratch"', source)


if __name__ == "__main__":
    unittest.main()
