"""CPU tests for scripts/falsify_b2_edit.py (Lane-4 B2 falsification).

Run with:
    conda run -n adags python -m unittest tests.test_falsify_b2_edit

No renderer, no CUDA: the module under test keeps every CUDA-touching
import inside `main`/the render helpers, so the pure surface — row-set
construction, the spanning exclusion, the comparative anti-vacuity gate,
the no-op split, and report assembly — is exercised on hand-built rows
whose expected membership is closed-form.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402
from scripts import falsify_b2_edit as fb  # noqa: E402

CENTRE = (0.7, 0.1, 0.35)
RADIUS = 0.2


def _rows(specs):
    """Build (pos_ep1, pos_ret, sup_lo, sup_hi, dc) from row specs.

    Each spec is (d_ep1, d_ret, sup_lo, sup_hi, dc_value): the distances
    are radial offsets from the authored centre along +x, so membership
    in the sphere is exactly `distance <= RADIUS`.
    """
    centre = torch.tensor(CENTRE, dtype=torch.float64)
    p1, pr, lo, hi, dc = [], [], [], [], []
    for d1, dr, slo, shi, value in specs:
        p1.append(centre + torch.tensor([d1, 0.0, 0.0], dtype=torch.float64))
        pr.append(centre + torch.tensor([dr, 0.0, 0.0], dtype=torch.float64))
        lo.append(slo)
        hi.append(shi)
        dc.append([value, 0.0, 0.0])
    return (torch.stack(p1), torch.stack(pr),
            torch.tensor(lo, dtype=torch.float64),
            torch.tensor(hi, dtype=torch.float64),
            torch.tensor(dc, dtype=torch.float64))


class RowSetTests(unittest.TestCase):
    def test_branches_are_hit_exactly(self):
        specs = [
            # 0: donor - inside sphere at t=2.5, support inside W1, hi <= 5.0
            (0.05, 0.05, 1.0, 3.0, 0.10),
            # 1: donor
            (0.10, 0.10, 0.5, 4.0, 0.12),
            # 2: recipient - inside sphere at t=9.6, support in WR, lo >= 9.3
            (0.90, 0.05, 9.4, 9.9, 0.80),
            # 3: recipient
            (0.90, 0.15, 9.35, 10.2, 0.82),
            # 4: wrong identity - far outside the sphere, episode-1 support
            (1.20, 1.20, 1.0, 3.0, 0.81),
            # 5: wrong identity, DC farther from the recipient medoid
            (2.00, 2.00, 0.0, 2.0, 0.30),
            # 6: inside the sphere at t=2.5 but support upper edge past 5.0
            (0.05, 0.05, 1.0, 6.0, 0.14),
            # 7: near the sphere but inside radius*1.75 -> neither D nor Dw
            (0.30, 0.30, 1.0, 3.0, 0.15),
            # 8: return-side position in the sphere but support lower edge
            #    below 9.3 -> not a recipient
            (0.90, 0.05, 9.0, 9.9, 0.83),
        ]
        sets = fb.build_row_sets(*_rows(specs), CENTRE, RADIUS)
        self.assertEqual(sets["donor"].tolist(), [0, 1])
        self.assertEqual(sets["recipient"].tolist(), [2, 3])
        self.assertEqual(sets["spanning"].tolist(), [])
        # Dw pool is rows 4 and 5; ranked by DC distance to the recipient
        # medoid (median of 0.80/0.82 = 0.81) and truncated to |D| = 2
        self.assertEqual(sets["wrong_pool"].tolist(), [4, 5])
        self.assertEqual(sets["wrong"].tolist(), [4, 5])
        self.assertNotIn(6, sets["donor"].tolist())
        self.assertNotIn(7, sets["donor"].tolist())
        self.assertNotIn(7, sets["wrong_pool"].tolist())
        self.assertNotIn(8, sets["recipient"].tolist())

    def test_wrong_identity_is_ranked_and_truncated_to_donor_count(self):
        specs = [
            (0.05, 0.05, 1.0, 3.0, 0.10),          # donor
            (0.90, 0.05, 9.4, 9.9, 0.80),          # recipient (medoid 0.80)
            (1.50, 1.50, 1.0, 3.0, 0.50),          # Dw, distance 0.30
            (1.50, 1.50, 1.0, 3.0, 0.79),          # Dw, distance 0.01 (closest)
            (1.50, 1.50, 1.0, 3.0, 0.20),          # Dw, distance 0.60
        ]
        sets = fb.build_row_sets(*_rows(specs), CENTRE, RADIUS)
        self.assertEqual(sets["donor"].tolist(), [0])
        self.assertEqual(sets["wrong_pool"].tolist(), [2, 3, 4])
        self.assertEqual(sets["wrong"].tolist(), [3])   # |D| = 1, DC-closest

    def test_spanning_rows_are_excluded_and_counted(self):
        specs = [
            (0.05, 0.05, 1.0, 3.0, 0.10),          # donor
            (0.10, 0.10, 0.5, 4.0, 0.12),          # donor
            (0.90, 0.05, 9.4, 9.9, 0.80),          # recipient
            # spanning: one wide support covering W1 and WR, in the sphere
            # at BOTH probe times
            (0.05, 0.05, 0.5, 9.9, 0.40),
            # spanning: in the sphere only at the return probe
            (0.90, 0.05, 2.0, 9.7, 0.42),
        ]
        sets = fb.build_row_sets(*_rows(specs), CENTRE, RADIUS)
        self.assertEqual(sets["spanning"].tolist(), [3, 4])
        for row in (3, 4):
            self.assertNotIn(row, sets["donor"].tolist())
            self.assertNotIn(row, sets["recipient"].tolist())
            self.assertNotIn(row, sets["wrong_pool"].tolist())
        self.assertEqual(fb.sets_summary(sets)["spanning_rows_excluded"], 2)

    def test_effective_support_uses_two_sigma_of_exp_scaling_t(self):
        t_mean = torch.tensor([[2.0], [9.6]])
        scaling_t = torch.log(torch.tensor([[0.5], [0.1]]))
        lo, hi = fb.effective_support(t_mean, scaling_t)
        self.assertAlmostEqual(float(lo[0]), 1.0, places=6)
        self.assertAlmostEqual(float(hi[0]), 3.0, places=6)
        self.assertAlmostEqual(float(lo[1]), 9.4, places=6)
        self.assertAlmostEqual(float(hi[1]), 9.8, places=6)


class NoOpSplitTests(unittest.TestCase):
    def test_parity_split_is_disjoint_and_covers_the_donor_set(self):
        donor = torch.tensor([3, 4, 7, 10, 11, 12])
        a, b = fb.split_by_row_parity(donor)
        self.assertEqual(a.tolist(), [4, 10, 12])
        self.assertEqual(b.tolist(), [3, 7, 11])
        self.assertEqual(set(a.tolist()) & set(b.tolist()), set())
        self.assertEqual(sorted(a.tolist() + b.tolist()), donor.tolist())

    def test_one_hop_assertion_rejects_overlapping_sides(self):
        with self.assertRaises(ContractError):
            fb.link_pointer(8, torch.tensor([1, 2]), torch.tensor([2, 3]))
        pointer = fb.link_pointer(8, torch.tensor([1, 2]), torch.tensor([5, 6]))
        self.assertEqual(pointer.tolist(), [0, 5, 6, 3, 4, 5, 6, 7])

    def test_assert_sets_disjoint_flags_a_shared_row(self):
        sets = {
            "donor": torch.tensor([0, 1]),
            "recipient": torch.tensor([2, 3]),
            "wrong": torch.tensor([4, 5]),
            "donor_a": torch.tensor([0]),
            "donor_b": torch.tensor([1]),
        }
        self.assertTrue(fb.assert_sets_disjoint(sets))
        sets["wrong"] = torch.tensor([3, 5])       # shares row 3 with R
        with self.assertRaises(ContractError):
            fb.assert_sets_disjoint(sets)


def _stat(mean, *, rows_changed=4, slot_ok=True, maximum=None):
    return {
        "pre_edit_dc_distance_mean": mean,
        "pre_edit_dc_distance_max": mean if maximum is None else maximum,
        "rows_changed": rows_changed,
        "rows_changed_fraction": 1.0,
        "slot_ok": slot_ok,
        "slot_units_per_side": [8, 8] if slot_ok else None,
        "slot_available_per_side": [40, 12],
    }


def _link_stats(l1_mean, l3_mean, **kwargs):
    return {
        fb.LINK_L1: _stat(l1_mean, **kwargs),
        fb.LINK_L2: _stat(0.5, **kwargs),
        fb.LINK_L3: _stat(l3_mean, **kwargs),
    }


class GateTests(unittest.TestCase):
    def test_gate_passes_when_l1_is_strictly_more_distant(self):
        verdict, anti = fb.gate_decision(_link_stats(0.40, 0.02))
        self.assertIsNone(verdict)
        self.assertTrue(anti["comparative_ok"])
        self.assertTrue(anti["slots_ok"])

    def test_vacuous_gate_blocks_every_delta(self):
        """L1 no more distant than L3: the comparison cannot falsify
        anything, so the measurement function must never run."""
        calls = []

        def measure(name, stat):            # sentinel: must stay uncalled
            calls.append(name)
            raise AssertionError(
                "a reconstruction delta was computed after a failed gate")

        report = fb.falsification_flow(
            {"donor_rows": 12}, _link_stats(0.02, 0.02), measure)
        self.assertEqual(report["verdict"], "INVALID_VACUOUS")
        self.assertEqual(calls, [])
        for link in report["links"]:
            self.assertIsNone(link["raw_slot_delta_mean"])
            self.assertIsNone(link["event_return_psnr_delta"])
            self.assertFalse(link["certificate"]["admitted"])

    def test_equal_distances_are_vacuous_strictly_greater_is_required(self):
        verdict, _ = fb.gate_decision(_link_stats(0.10, 0.10))
        self.assertEqual(verdict, "INVALID_VACUOUS")
        verdict, _ = fb.gate_decision(_link_stats(0.10 + 1e-9, 0.10))
        self.assertIsNone(verdict)

    def test_zero_rows_changed_is_vacuous(self):
        verdict, _ = fb.gate_decision(_link_stats(0.40, 0.02, rows_changed=0))
        self.assertEqual(verdict, "INVALID_VACUOUS")

    def test_unsatisfiable_slot_yields_invalid_slots_and_no_delta(self):
        stats = _link_stats(0.40, 0.02)
        stats[fb.LINK_L3]["slot_ok"] = False
        stats[fb.LINK_L3]["slot_units_per_side"] = None

        def measure(name, stat):
            raise AssertionError("delta computed despite an unsatisfiable slot")

        report = fb.falsification_flow({"donor_rows": 12}, stats, measure)
        self.assertEqual(report["verdict"], "INVALID_SLOTS")
        l3 = [link for link in report["links"] if link["name"] == fb.LINK_L3][0]
        self.assertEqual(l3["certificate"]["stage_reached"], "slot")
        self.assertEqual(l3["slot_available_per_side"], [40, 12])

    def test_vacuity_takes_precedence_over_slots(self):
        stats = _link_stats(0.02, 0.02)
        stats[fb.LINK_L2]["slot_ok"] = False
        verdict, _ = fb.gate_decision(stats)
        self.assertEqual(verdict, "INVALID_VACUOUS")


class CertificateRuleTests(unittest.TestCase):
    def test_admission_needs_the_pooled_and_both_side_rules(self):
        self.assertEqual(
            fb.certificate_stage(-1.0, 0.1, [-1.0, -1.0]), ("admitted", True))
        self.assertEqual(
            fb.certificate_stage(-0.1, 0.1, [-1.0, -1.0])[0], "pooled-rule")
        self.assertEqual(
            fb.certificate_stage(-1.0, 0.1, [-2.0, 0.5])[0], "side-rule")

    def test_mean_se(self):
        mean, se = fb.mean_se([1.0, 3.0])
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(se, 1.0)
        self.assertEqual(fb.mean_se([]), (None, None))


class ReportSchemaTests(unittest.TestCase):
    REQUIRED_LINK_KEYS = {
        "name", "pre_edit_dc_distance_mean", "pre_edit_dc_distance_max",
        "rows_changed", "slot_units_per_side", "raw_slot_delta_mean",
        "raw_slot_delta_se", "return_side_delta_mean",
        "event_return_psnr_base", "event_return_psnr_edited",
        "event_return_psnr_delta", "certificate",
    }
    REQUIRED_CERT_KEYS = {"stage_reached", "admitted", "pooled_mean",
                          "pooled_se", "side_means"}

    def test_completed_report_carries_every_documented_key(self):
        def measure(name, stat):
            return {
                "raw_slot_delta_mean": -0.01,
                "raw_slot_delta_se": 0.001,
                "return_side_delta_mean": -0.02,
                "event_return_psnr_base": 30.0,
                "event_return_psnr_edited": 31.5,
                "event_return_psnr_delta": 1.5,
                "certificate": {
                    "stage_reached": "admitted", "admitted": True,
                    "pooled_mean": -0.01, "pooled_se": 0.001,
                    "side_means": [-0.01, -0.01],
                },
            }

        report = fb.falsification_flow(
            {"donor_rows": 12}, _link_stats(0.4, 0.02), measure,
            packet_block={"packets": 3})
        self.assertEqual(report["schema"], "ccr-b2-falsification-v1")
        self.assertEqual(set(report) >= {"schema", "sets", "anti_vacuity",
                                         "links", "verdict"}, True)
        self.assertEqual(report["verdict"], "COMPLETED")
        self.assertEqual(report["packets"], {"packets": 3})
        self.assertEqual([link["name"] for link in report["links"]],
                         list(fb.LINK_ORDER))
        for link in report["links"]:
            self.assertTrue(self.REQUIRED_LINK_KEYS <= set(link))
            self.assertTrue(self.REQUIRED_CERT_KEYS <= set(link["certificate"]))
            self.assertEqual(link["event_return_psnr_delta"], 1.5)

    def test_invalid_sets_report_shape(self):
        report = fb.invalid_sets_report({"donor_rows": 3, "recipient_rows": 1})
        self.assertEqual(report["verdict"], "INVALID_SETS")
        self.assertEqual(report["links"], [])
        self.assertIsNone(report["anti_vacuity"])
        self.assertEqual(report["schema"], "ccr-b2-falsification-v1")

    def test_sets_sufficiency_thresholds(self):
        def sets(n_donor, n_recipient):
            return {"donor": torch.arange(n_donor),
                    "recipient": torch.arange(n_recipient)}

        self.assertFalse(fb.sets_are_sufficient(sets(7, 10)))
        self.assertFalse(fb.sets_are_sufficient(sets(10, 3)))
        self.assertTrue(fb.sets_are_sufficient(sets(8, 4)))


class RowMapTests(unittest.TestCase):
    def test_nearest_dc_map_and_pre_edit_stats(self):
        dc = torch.tensor([[0.0], [1.0], [0.9], [0.1]], dtype=torch.float64)
        r_rows, d_rows = fb.nearest_dc_row_map(
            dc, torch.tensor([2, 3]), torch.tensor([0, 1]))
        self.assertEqual(r_rows.tolist(), [2, 3])
        self.assertEqual(d_rows.tolist(), [1, 0])      # 0.9 -> 1.0, 0.1 -> 0.0
        stats = fb.pre_edit_dc_stats(dc, r_rows, d_rows)
        self.assertAlmostEqual(stats["pre_edit_dc_distance_mean"], 0.1, places=6)
        self.assertEqual(stats["rows_changed"], 2)
        self.assertEqual(stats["rows_changed_fraction"], 1.0)

    def test_row_map_refuses_an_empty_side(self):
        dc = torch.zeros(4, 1, dtype=torch.float64)
        with self.assertRaises(ContractError):
            fb.nearest_dc_row_map(dc, torch.tensor([], dtype=torch.long),
                                  torch.tensor([0]))


class PacketSummaryTests(unittest.TestCase):
    def test_packet_membership_is_descriptive_only(self):
        sets = {"donor": torch.tensor([0, 1]),
                "recipient": torch.tensor([2, 3])}
        ids = torch.tensor([-1, 5, 5, 7])
        block = fb.packet_summary(ids, sets)
        self.assertEqual(block["rows"], 4)
        self.assertEqual(block["rows_with_packet"], 3)
        self.assertEqual(block["packets"], 2)
        self.assertEqual(block["donor_rows_with_packet"], 1)
        self.assertEqual(block["recipient_rows_with_packet"], 2)
        self.assertEqual(block["recipient_packet_ids"], [5, 7])


if __name__ == "__main__":
    unittest.main()
