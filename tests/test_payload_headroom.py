"""CPU tests for scripts/payload_headroom.py (the payload headroom screen).

Run with:
    python -m unittest tests.test_payload_headroom
    python -m pytest tests/test_payload_headroom.py

No renderer, no CUDA, no checkpoint: every numeric in the screen lives
in a module-level pure function, and the scene/checkpoint imports are
lazy inside `main`, so the arithmetic that the scientific conclusion
rests on — the distance summaries, the raw-vs-activated readings, the
quaternion geodesic angle, the two ratios and their divide-by-zero
handling, and the two row maps — is auditable here on hand-built
tensors whose expected values are closed-form.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402
from scripts import falsify_b2_edit as fb  # noqa: E402
from scripts import payload_headroom as ph  # noqa: E402


def _sets(donor, recipient, wrong, donor_a, donor_b):
    return {
        "donor": torch.tensor(donor, dtype=torch.long),
        "recipient": torch.tensor(recipient, dtype=torch.long),
        "wrong": torch.tensor(wrong, dtype=torch.long),
        "donor_a": torch.tensor(donor_a, dtype=torch.long),
        "donor_b": torch.tensor(donor_b, dtype=torch.long),
    }


#: 6 rows: donors {0,1}, recipients {2,3}, wrong-identity {4,5}.
#: The parity split of the donor set is donor_a {0} / donor_b {1}.
SETS = _sets([0, 1], [2, 3], [4, 5], [0], [1])

#: DC values chosen so every mapped pair has an integral distance:
#:   L1  10->1 (9), 11->1 (10)   mean 9.5
#:   L2  10->4 (90), 11->4 (89)  mean 89.5
#:   L3   1->0 (1)               mean 1.0
DC = torch.tensor([[0.0], [1.0], [10.0], [11.0], [100.0], [101.0]],
                  dtype=torch.float64)


class FlattenAndDistanceTests(unittest.TestCase):
    def test_flatten_rows_is_cpu_float64_and_row_major(self):
        values = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
        flat = ph.flatten_rows(values)
        self.assertEqual(flat.dtype, torch.float64)
        self.assertEqual(tuple(flat.shape), (3, 4))
        self.assertEqual(flat[1].tolist(), [4.0, 5.0, 6.0, 7.0])

    def test_pair_distances_is_the_l2_norm_over_trailing_dims(self):
        values = torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]],
                              dtype=torch.float64)
        dist = ph.pair_distances(values, torch.tensor([1, 2]),
                                 torch.tensor([0, 0]))
        self.assertEqual([round(float(x), 9) for x in dist], [5.0, 10.0])

    def test_pair_distances_on_one_dimensional_values_is_the_absolute_gap(self):
        values = torch.tensor([[2.0], [-3.0]], dtype=torch.float64)
        dist = ph.pair_distances(values, torch.tensor([0]), torch.tensor([1]))
        self.assertAlmostEqual(float(dist[0]), 5.0, places=12)

    def test_pair_distances_refuses_a_map_that_is_not_one_to_one(self):
        values = torch.zeros(4, 1, dtype=torch.float64)
        with self.assertRaises(ContractError):
            ph.pair_distances(values, torch.tensor([0, 1]), torch.tensor([2]))

    def test_pair_distances_refuses_an_unknown_metric(self):
        values = torch.zeros(2, 1, dtype=torch.float64)
        with self.assertRaises(ContractError):
            ph.pair_distances(values, torch.tensor([0]), torch.tensor([1]),
                              "cosine")


class DistanceSummaryTests(unittest.TestCase):
    def test_summary_of_a_known_set(self):
        summary = ph.distance_summary(torch.tensor([9.0, 10.0]))
        self.assertEqual(summary["pairs"], 2)
        self.assertAlmostEqual(summary["mean"], 9.5, places=12)
        self.assertAlmostEqual(summary["median"], 9.5, places=12)
        self.assertAlmostEqual(summary["max"], 10.0, places=12)
        # linear-interpolated quantile: 9 + 0.95 * (10 - 9)
        self.assertAlmostEqual(summary["p95"], 9.95, places=12)

    def test_summary_of_four_values(self):
        summary = ph.distance_summary(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.assertAlmostEqual(summary["mean"], 2.5, places=12)
        self.assertAlmostEqual(summary["median"], 2.5, places=12)
        self.assertAlmostEqual(summary["max"], 4.0, places=12)
        self.assertAlmostEqual(summary["p95"], 3.85, places=12)

    def test_empty_summary_is_all_none_not_zero(self):
        summary = ph.distance_summary(torch.zeros(0))
        self.assertEqual(summary["pairs"], 0)
        for key in ("mean", "median", "max", "p95"):
            self.assertIsNone(summary[key])


class QuaternionGeodesicTests(unittest.TestCase):
    def test_identical_rotations_are_zero_degrees(self):
        q = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
        angle = ph.quaternion_geodesic_degrees(q, q)
        self.assertAlmostEqual(float(angle[0]), 0.0, places=6)

    def test_ninety_degree_rotation_about_x(self):
        half = math.sqrt(0.5)
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        rot90 = torch.tensor([[half, half, 0.0, 0.0]], dtype=torch.float64)
        angle = ph.quaternion_geodesic_degrees(identity, rot90)
        self.assertAlmostEqual(float(angle[0]), 90.0, places=6)

    def test_one_hundred_and_eighty_degree_rotation(self):
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        rot180 = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
        angle = ph.quaternion_geodesic_degrees(identity, rot180)
        self.assertAlmostEqual(float(angle[0]), 180.0, places=6)

    def test_double_cover_q_and_minus_q_are_the_same_rotation(self):
        q = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float64)
        angle = ph.quaternion_geodesic_degrees(q, -q)
        self.assertAlmostEqual(float(angle[0]), 0.0, places=6)

    def test_unnormalized_inputs_are_normalized_first(self):
        half = math.sqrt(0.5)
        identity = torch.tensor([[3.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        rot90 = torch.tensor([[half * 7, half * 7, 0.0, 0.0]],
                             dtype=torch.float64)
        angle = ph.quaternion_geodesic_degrees(identity, rot90)
        self.assertAlmostEqual(float(angle[0]), 90.0, places=6)

    def test_geodesic_metric_is_reachable_through_pair_distances(self):
        half = math.sqrt(0.5)
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                              [half, half, 0.0, 0.0]], dtype=torch.float64)
        dist = ph.pair_distances(quats, torch.tensor([1]), torch.tensor([0]),
                                 ph.METRIC_QUAT)
        self.assertAlmostEqual(float(dist[0]), 90.0, places=6)

    def test_non_quaternion_input_fails_closed(self):
        values = torch.zeros(2, 3, dtype=torch.float64)
        with self.assertRaises(ContractError):
            ph.quaternion_geodesic_degrees(values, values)


class SafeRatioTests(unittest.TestCase):
    def test_an_ordinary_ratio(self):
        value, reason = ph.safe_ratio(0.0464, 0.0325)
        self.assertIsNone(reason)
        self.assertAlmostEqual(value, 0.0464 / 0.0325, places=12)

    def test_zero_denominator_is_null_with_a_reason(self):
        value, reason = ph.safe_ratio(1.0, 0.0)
        self.assertIsNone(value)
        self.assertIn("exactly zero", reason)

    def test_missing_operand_is_null_with_a_reason(self):
        value, reason = ph.safe_ratio(None, 1.0)
        self.assertIsNone(value)
        self.assertIn("missing", reason)
        value, reason = ph.safe_ratio(1.0, None)
        self.assertIsNone(value)
        self.assertIn("missing", reason)

    def test_non_finite_operand_is_null_with_a_reason(self):
        for bad in (float("inf"), float("nan"), float("-inf")):
            value, reason = ph.safe_ratio(bad, 1.0)
            self.assertIsNone(value)
            self.assertEqual(reason, "non-finite operand")
            value, reason = ph.safe_ratio(1.0, bad)
            self.assertIsNone(value)
            self.assertEqual(reason, "non-finite operand")

    def test_the_result_is_never_inf_or_nan(self):
        cases = [(1.0, 0.0), (0.0, 0.0), (float("nan"), 0.0),
                 (1e308, 1e-308), (None, None)]
        for numerator, denominator in cases:
            value, reason = ph.safe_ratio(numerator, denominator)
            if value is None:
                self.assertIsInstance(reason, str)
                self.assertTrue(reason)
            else:
                self.assertTrue(math.isfinite(value))


class RatioBlockTests(unittest.TestCase):
    def test_the_two_documented_ratios(self):
        summaries = {
            fb.LINK_L1: {"mean": 0.0464},
            fb.LINK_L2: {"mean": 0.706},
            fb.LINK_L3: {"mean": 0.0325},
        }
        block = ph.ratio_block(summaries)
        # the recorded DC readings: 1.43 and 21.7
        self.assertAlmostEqual(block["headroom_ratio"], 1.4276923, places=6)
        self.assertAlmostEqual(block["discrimination_ratio"], 21.7230769,
                               places=6)
        self.assertIsNone(block["headroom_ratio_reason"])
        self.assertIsNone(block["discrimination_ratio_reason"])
        self.assertEqual(block["headroom_ratio_definition"],
                         "L1_mean / L3_mean")
        self.assertEqual(block["discrimination_ratio_definition"],
                         "L2_mean / L3_mean")

    def test_a_zero_no_op_reference_yields_null_ratios_with_reasons(self):
        summaries = {
            fb.LINK_L1: {"mean": 0.5},
            fb.LINK_L2: {"mean": 0.9},
            fb.LINK_L3: {"mean": 0.0},
        }
        block = ph.ratio_block(summaries)
        self.assertIsNone(block["headroom_ratio"])
        self.assertIsNone(block["discrimination_ratio"])
        self.assertIn("exactly zero", block["headroom_ratio_reason"])
        self.assertIn("exactly zero", block["discrimination_ratio_reason"])

    def test_an_empty_link_yields_null_ratios_with_reasons(self):
        summaries = {
            fb.LINK_L1: ph.distance_summary(torch.zeros(0)),
            fb.LINK_L2: {"mean": 0.9},
            fb.LINK_L3: {"mean": 0.1},
        }
        block = ph.ratio_block(summaries)
        self.assertIsNone(block["headroom_ratio"])
        self.assertIn("missing", block["headroom_ratio_reason"])
        self.assertIsNotNone(block["discrimination_ratio"])


class RowMapTests(unittest.TestCase):
    def test_link_row_pairs_match_the_falsification_links(self):
        pairs = ph.link_row_pairs(SETS)
        self.assertEqual(sorted(pairs), sorted(fb.LINK_ORDER))
        self.assertEqual(pairs[fb.LINK_L1][0].tolist(), [2, 3])
        self.assertEqual(pairs[fb.LINK_L1][1].tolist(), [0, 1])
        self.assertEqual(pairs[fb.LINK_L2][1].tolist(), [4, 5])
        self.assertEqual(pairs[fb.LINK_L3][0].tolist(), [1])
        self.assertEqual(pairs[fb.LINK_L3][1].tolist(), [0])

    def test_the_primary_map_is_the_frozen_dc_map(self):
        maps = ph.build_maps(DC, DC, SETS)
        self.assertEqual(maps[ph.MAP_PRIMARY][fb.LINK_L1][1].tolist(), [1, 1])
        self.assertEqual(maps[ph.MAP_PRIMARY][fb.LINK_L2][1].tolist(), [4, 4])
        self.assertEqual(maps[ph.MAP_PRIMARY][fb.LINK_L3][1].tolist(), [0])

    def test_the_native_map_differs_when_the_two_spaces_disagree(self):
        # DC puts donor 1 nearest to both recipients; in the payload's own
        # space donor 0 is nearest instead, so the correspondences differ.
        native = torch.tensor([[10.5], [99.0], [10.0], [11.0],
                               [100.0], [101.0]], dtype=torch.float64)
        maps = ph.build_maps(DC, native, SETS)
        self.assertEqual(maps[ph.MAP_PRIMARY][fb.LINK_L1][1].tolist(), [1, 1])
        self.assertEqual(maps[ph.MAP_NATIVE][fb.LINK_L1][1].tolist(), [0, 0])
        self.assertFalse(ph.maps_agree(maps[ph.MAP_PRIMARY][fb.LINK_L1],
                                       maps[ph.MAP_NATIVE][fb.LINK_L1]))
        self.assertTrue(ph.maps_agree(maps[ph.MAP_PRIMARY][fb.LINK_L3],
                                      maps[ph.MAP_NATIVE][fb.LINK_L3]))

    def test_both_maps_are_always_reported(self):
        native = torch.tensor([[10.5], [99.0], [10.0], [11.0],
                               [100.0], [101.0]], dtype=torch.float64)
        maps = ph.build_maps(DC, native, SETS)
        self.assertEqual(sorted(maps), sorted(ph.MAP_ORDER))
        for map_name in ph.MAP_ORDER:
            self.assertEqual(sorted(maps[map_name]), sorted(fb.LINK_ORDER))

    def test_an_empty_side_is_recorded_as_none_not_raised(self):
        thin = _sets([0, 1], [2, 3], [], [0], [1])
        maps = ph.build_maps(DC, DC, thin)
        self.assertIsNone(maps[ph.MAP_PRIMARY][fb.LINK_L2])
        self.assertIsNone(maps[ph.MAP_NATIVE][fb.LINK_L2])
        self.assertIsNone(ph.maps_agree(None, maps[ph.MAP_NATIVE][fb.LINK_L1]))
        self.assertIsNotNone(maps[ph.MAP_PRIMARY][fb.LINK_L1])


class AnalyseSpacesTests(unittest.TestCase):
    def test_raw_and_activated_readings_are_both_correct(self):
        # A logit column whose activated distances are NOT proportional to
        # its raw distances: the whole reason both spaces are reported.
        raw = torch.tensor([[0.0], [4.0], [1.0], [5.0], [-4.0], [-5.0]],
                           dtype=torch.float64)
        activated = torch.sigmoid(raw)
        maps = ph.build_maps(DC, activated, SETS)
        spaces = {
            ph.SPACE_RAW: (raw, ph.METRIC_L2, "_opacity (raw logit)"),
            ph.SPACE_ACTIVATED: (activated, ph.METRIC_L2,
                                 "sigmoid(_opacity)"),
        }
        result = ph.analyse_spaces(spaces, maps)

        # PRIMARY (DC) map: L1 sends both recipients to donor row 1.
        raw_l1 = result[ph.SPACE_RAW]["maps"][ph.MAP_PRIMARY]["links"][
            fb.LINK_L1]
        self.assertAlmostEqual(raw_l1["mean"], (3.0 + 1.0) / 2.0, places=12)
        self.assertAlmostEqual(raw_l1["max"], 3.0, places=12)

        act_l1 = result[ph.SPACE_ACTIVATED]["maps"][ph.MAP_PRIMARY]["links"][
            fb.LINK_L1]
        expected = [
            float(abs(torch.sigmoid(torch.tensor(1.0, dtype=torch.float64))
                      - torch.sigmoid(torch.tensor(4.0,
                                                   dtype=torch.float64)))),
            float(abs(torch.sigmoid(torch.tensor(5.0, dtype=torch.float64))
                      - torch.sigmoid(torch.tensor(4.0,
                                                   dtype=torch.float64)))),
        ]
        self.assertAlmostEqual(act_l1["mean"], sum(expected) / 2.0, places=12)
        self.assertAlmostEqual(act_l1["max"], max(expected), places=12)

        # The two spaces genuinely disagree; reporting only one would hide it.
        self.assertNotAlmostEqual(raw_l1["mean"], act_l1["mean"], places=3)

    def test_every_space_reports_both_maps_and_both_ratios(self):
        raw = torch.tensor([[0.0], [4.0], [1.0], [5.0], [-4.0], [-5.0]],
                           dtype=torch.float64)
        maps = ph.build_maps(DC, raw, SETS)
        result = ph.analyse_spaces(
            {ph.SPACE_RAW: (raw, ph.METRIC_L2, "raw")}, maps)
        entry = result[ph.SPACE_RAW]
        self.assertEqual(entry["metric"], ph.METRIC_L2)
        self.assertEqual(sorted(entry["maps"]), sorted(ph.MAP_ORDER))
        for map_name in ph.MAP_ORDER:
            block = entry["maps"][map_name]
            self.assertEqual(sorted(block["links"]), sorted(fb.LINK_ORDER))
            for key in ("headroom_ratio", "discrimination_ratio",
                        "headroom_ratio_reason",
                        "discrimination_ratio_reason"):
                self.assertIn(key, block)

    def test_ratios_follow_the_measured_distances(self):
        maps = ph.build_maps(DC, DC, SETS)
        result = ph.analyse_spaces(
            {ph.SPACE_RAW: (DC, ph.METRIC_L2, "dc")}, maps)
        block = result[ph.SPACE_RAW]["maps"][ph.MAP_PRIMARY]
        links = block["links"]
        self.assertAlmostEqual(links[fb.LINK_L1]["mean"], 9.5, places=12)
        self.assertAlmostEqual(links[fb.LINK_L2]["mean"], 89.5, places=12)
        self.assertAlmostEqual(links[fb.LINK_L3]["mean"], 1.0, places=12)
        self.assertAlmostEqual(block["headroom_ratio"], 9.5, places=12)
        self.assertAlmostEqual(block["discrimination_ratio"], 89.5, places=12)

    def test_a_vacuous_link_set_reports_null_ratios_not_nan(self):
        flat = torch.zeros(6, 1, dtype=torch.float64)
        maps = ph.build_maps(flat, flat, SETS)
        result = ph.analyse_spaces(
            {ph.SPACE_RAW: (flat, ph.METRIC_L2, "flat")}, maps)
        block = result[ph.SPACE_RAW]["maps"][ph.MAP_PRIMARY]
        self.assertAlmostEqual(block["links"][fb.LINK_L3]["mean"], 0.0,
                               places=12)
        self.assertIsNone(block["headroom_ratio"])
        self.assertIn("exactly zero", block["headroom_ratio_reason"])


class SpecTableTests(unittest.TestCase):
    def test_the_frozen_candidate_list(self):
        names = [spec["name"] for spec in ph.TENSOR_SPECS]
        self.assertEqual(names, ["_features_dc", "_opacity", "_scaling_t",
                                 "_t", "_xyz", "_scaling", "_rotation"])

    def test_only_the_logit_and_log_columns_declare_an_activation(self):
        activations = {spec["name"]: spec["activation"]
                       for spec in ph.TENSOR_SPECS}
        self.assertEqual(activations["_opacity"], "opacity_activation")
        self.assertEqual(activations["_scaling"], "scaling_activation")
        self.assertEqual(activations["_scaling_t"], "scaling_activation")
        self.assertEqual(activations["_rotation"], "rotation_activation")
        self.assertIsNone(activations["_features_dc"])
        self.assertIsNone(activations["_t"])
        self.assertIsNone(activations["_xyz"])

    def test_every_spec_names_its_raw_label_and_activated_label(self):
        for spec in ph.TENSOR_SPECS:
            self.assertIn("raw_label", spec)
            if spec["activation"] is not None:
                self.assertIn("activated_label", spec)

    def test_only_rotation_declares_the_geodesic_space(self):
        geodesic = [spec["name"] for spec in ph.TENSOR_SPECS
                    if spec.get("geodesic")]
        self.assertEqual(geodesic, ["_rotation"])

    def test_the_screen_reuses_the_falsification_surface(self):
        self.assertIs(ph.build_row_sets, fb.build_row_sets)
        self.assertIs(ph.nearest_dc_row_map, fb.nearest_dc_row_map)
        self.assertIs(ph.nearest_row_map, fb.nearest_row_map)
        self.assertIs(ph.split_by_row_parity, fb.split_by_row_parity)
        self.assertIs(ph.assert_sets_disjoint, fb.assert_sets_disjoint)
        self.assertIs(ph.effective_support, fb.effective_support)
        self.assertIs(ph.restore_model_and_scene, fb.restore_model_and_scene)
        self.assertIs(ph.probe_row_state, fb.probe_row_state)

    def test_the_screen_reuses_the_fixture_protocol(self):
        # The screen must not carry its own copy of the windows/probes:
        # pointing it at LRV4 has to move the recipient probe onto LRV4's
        # single return instant, and that only happens if it derives the
        # protocol through the same code the falsification does.
        self.assertIs(ph.protocol_from_event_spec, fb.protocol_from_event_spec)
        self.assertIs(ph.validate_protocol, fb.validate_protocol)
        self.assertIs(ph.protocol_block, fb.protocol_block)
        self.assertIs(ph.DEFAULT_PROTOCOL, fb.DEFAULT_PROTOCOL)


class ReportRenderingTests(unittest.TestCase):
    @staticmethod
    def _minimal_report():
        maps = ph.build_maps(DC, DC, SETS)
        spaces = ph.analyse_spaces(
            {ph.SPACE_RAW: (DC, ph.METRIC_L2, "dc")}, maps)
        return {
            "schema": ph.SCHEMA,
            "checkpoint": "/apollo/run/chkpnt6000.pth",
            "oracle_region": {"source": "configs/lrv3/oracle_correct.json"},
            "row_sets": {"rows": 6, "donor_rows": 2, "recipient_rows": 2,
                         "wrong_identity_rows": 2,
                         "spanning_rows_excluded": 0},
            "row_sets_disjoint": True,
            "tensors": {
                "_features_dc": {"role": "control", "spaces": spaces},
            },
        }

    def test_table_lists_every_reported_map(self):
        text = ph.format_table(self._minimal_report())
        self.assertIn(ph.SCHEMA, text)
        self.assertIn(ph.MAP_PRIMARY, text)
        self.assertIn(ph.MAP_NATIVE, text)
        self.assertIn("headroom", text)
        self.assertIn("_features_dc/raw", text)

    def test_table_prints_null_rather_than_nan(self):
        flat = torch.zeros(6, 1, dtype=torch.float64)
        maps = ph.build_maps(flat, flat, SETS)
        report = self._minimal_report()
        report["tensors"]["_features_dc"]["spaces"] = ph.analyse_spaces(
            {ph.SPACE_RAW: (flat, ph.METRIC_L2, "flat")}, maps)
        text = ph.format_table(report)
        self.assertIn("null", text)
        self.assertNotIn("nan", text.lower())
        self.assertNotIn("inf", text.lower())

    def test_unavailable_tensors_are_skipped_by_the_table(self):
        report = self._minimal_report()
        report["tensors"]["_rotation_r"] = {"available": False, "spaces": {}}
        text = ph.format_table(report)
        self.assertNotIn("_rotation_r/", text)

    def test_the_table_names_the_fixture_so_a_report_cannot_be_misread(self):
        report = self._minimal_report()
        lrv4 = fb.protocol_from_event_spec({
            "scene_id": "LRV4", "kind": "synthetic_leave_and_return",
            "fps": 6.0,
            "presence_frames": {"episode_1": [0, 29], "gap": [30, 58],
                                "episode_2": [59, 59]},
            "return_frames": [59],
        })
        report["protocol"] = {"fixture": ph.protocol_block(lrv4)}
        text = ph.format_table(report)
        self.assertIn("LRV4", text)
        self.assertIn("[59]", text)
        self.assertIn("9.8333", text)
        self.assertIn("derived", text)

    def test_a_report_without_a_fixture_block_still_renders(self):
        # The fixture line is additive: an older report that predates it
        # must still render rather than raising a KeyError.
        text = ph.format_table(self._minimal_report())
        self.assertNotIn("fixture:", text)
        self.assertIn(ph.MAP_PRIMARY, text)


class InputHashTests(unittest.TestCase):
    def test_sha256_of_a_known_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_bytes(b"payload: opacity\n")
            digest, size = ph.sha256_file(path)
        self.assertEqual(
            digest,
            "6d696fbe83d2f7d77499619b86eae49d9a1302709800d0d640155ea06416164e")
        self.assertEqual(size, len(b"payload: opacity\n"))

    def test_sha256_is_stable_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.txt"
            b = Path(tmp) / "b.txt"
            c = Path(tmp) / "c.txt"
            a.write_text("same")
            b.write_text("same")
            c.write_text("different")
            self.assertEqual(ph.sha256_file(a)[0], ph.sha256_file(b)[0])
            self.assertNotEqual(ph.sha256_file(a)[0], ph.sha256_file(c)[0])

    def test_missing_file_is_none_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            digest, size = ph.sha256_file(Path(tmp) / "absent.json")
        self.assertIsNone(digest)
        self.assertIsNone(size)


if __name__ == "__main__":
    unittest.main()
