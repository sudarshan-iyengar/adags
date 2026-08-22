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

import hashlib
import json
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


# ---------------------------------------------------------------------------
# --payload dc must reproduce the recorded behaviour EXACTLY
# ---------------------------------------------------------------------------

#: An 18-row scenario that exercises every branch the DC arm takes:
#: 8 donors, 4 recipients, 6 wrong-identity rows, no spanning rows.
#: (d_ep1, d_ret, sup_lo, sup_hi, dc_value)
_DC_SPECS = (
    [(0.02 + 0.01 * i, 0.02 + 0.01 * i, 0.5 + 0.1 * i, 2.0 + 0.2 * i,
      0.10 + 0.01 * i) for i in range(8)]
    + [(0.90, 0.03 + 0.02 * i, 9.35 + 0.01 * i, 9.85 + 0.02 * i,
        0.80 + 0.015 * i) for i in range(4)]
    + [(1.10 + 0.2 * i, 1.10 + 0.2 * i, 0.4 + 0.1 * i, 2.5 + 0.1 * i,
        0.30 + 0.09 * i) for i in range(6)]
)

#: sha256 of the canonical fingerprint below, captured from the
#: implementation as it stood BEFORE the payload generalization
#: (commit a5fb0e0 + no local edits). Stable across torch 1.12/py3.7 and
#: torch 2.6/py3.12. If a change to the DC arm moves this digest, the
#: recorded 2026-08-20 falsification is no longer reproducible and the
#: change must be reverted or re-frozen as a NEW spec.
_DC_FINGERPRINT_SHA256 = (
    "f5d8bdb6870725970e475c92713b0a5a87e63ecf964ea04614df9faa4fecce59"
)

_DC_EXPECTED_SETS = {
    "donor": [0, 1, 2, 3, 4, 5, 6, 7],
    "donor_a": [0, 2, 4, 6],
    "donor_b": [1, 3, 5, 7],
    "recipient": [8, 9, 10, 11],
    "spanning": [],
    "wrong": [17, 16, 15, 14, 13, 12],
    "wrong_pool": [12, 13, 14, 15, 16, 17],
}

_DC_EXPECTED_POINTERS = {
    fb.LINK_L1: [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 12, 13, 14, 15, 16, 17],
    fb.LINK_L2: [0, 1, 2, 3, 4, 5, 6, 7, 17, 17, 17, 17,
                 12, 13, 14, 15, 16, 17],
    fb.LINK_L3: [0, 0, 2, 2, 4, 6, 6, 6, 8, 9, 10, 11,
                 12, 13, 14, 15, 16, 17],
}


def _dc_measure(name, stat):
    """A deterministic stand-in for the render-backed measurement, so the
    report can be assembled and fingerprinted with no GPU."""
    seed = float(len(name))
    return {
        "raw_slot_delta_mean": -0.001 * seed,
        "raw_slot_delta_se": 0.0001 * seed,
        "return_side_delta_mean": -0.002 * seed,
        "return_side_units": 12,
        "event_return_psnr_base": 27.0,
        "event_return_psnr_edited": 27.0 + 0.01 * seed,
        "event_return_psnr_delta": 0.01 * seed,
        "event_return_pixels": 1234,
        "event_return_frames": 12,
        "certificate": {
            "stage_reached": "admitted", "admitted": True,
            "pooled_mean": -0.001 * seed, "pooled_se": 0.0001 * seed,
            "side_means": [-0.001 * seed, -0.002 * seed],
        },
    }


def _dc_pipeline(payload=None):
    """Run the no-render half of `main` for one payload.

    `payload=None` is the DC arm exactly as `main` runs it (it threads
    None so `pre_edit_stats` emits the legacy keys only).
    """
    pos1, posr, lo, hi, dc = _rows(_DC_SPECS)
    sets = fb.build_row_sets(pos1, posr, lo, hi, dc, CENTRE, RADIUS)
    fb.assert_sets_disjoint(sets)
    n_rows = int(dc.shape[0])

    link_rows = {
        fb.LINK_L1: (sets["recipient"], sets["donor"]),
        fb.LINK_L2: (sets["recipient"], sets["wrong"]),
        fb.LINK_L3: (sets["donor_b"], sets["donor_a"]),
    }
    link_stats = {}
    pointers = {}
    for name, (r_set, d_set) in link_rows.items():
        r_rows, d_rows = fb.nearest_dc_row_map(dc, r_set, d_set)
        stat = dict(fb.pre_edit_stats(dc, r_rows, d_rows, payload))
        pointers[name] = fb.link_pointer(n_rows, r_rows, d_rows).tolist()
        stat["slot_ok"] = True
        stat["slot_units_per_side"] = [8, 8]
        stat["slot_available_per_side"] = [40, 12]
        link_stats[name] = stat

    sets_block = fb.sets_summary(sets)
    sets_block["rows"] = n_rows
    report = fb.falsification_flow(sets_block, link_stats, _dc_measure,
                                   {"packets": 2})
    return {
        "sets": {k: v.tolist() for k, v in sets.items()},
        "pointers": pointers,
        "report": report,
        "sufficient": fb.sets_are_sufficient(sets),
    }


class PayloadDcIdentityTests(unittest.TestCase):
    """The hard requirement: `--payload dc` is the recorded arm."""

    def test_dc_row_sets_are_unchanged(self):
        self.assertEqual(_dc_pipeline()["sets"], _DC_EXPECTED_SETS)

    def test_dc_pointers_are_unchanged(self):
        self.assertEqual(_dc_pipeline()["pointers"], _DC_EXPECTED_POINTERS)

    def test_dc_report_keys_are_unchanged(self):
        report = _dc_pipeline()["report"]
        self.assertEqual(report["verdict"], "COMPLETED")
        for link in report["links"]:
            for key in link:
                self.assertFalse(
                    key.startswith("pre_edit_distance"),
                    "the DC arm must not gain payload-neutral keys")
                self.assertNotEqual(key, "payload")
                self.assertNotEqual(key, "payload_tensor")
        self.assertNotIn("payload", report["anti_vacuity"])
        self.assertNotIn("payload_tensor", report["anti_vacuity"])
        self.assertNotIn("l1_pre_edit_distance_mean", report["anti_vacuity"])

    def test_dc_fingerprint_is_byte_identical_to_the_recorded_arm(self):
        text = json.dumps(_dc_pipeline(), indent=1, sort_keys=True)
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),
                         _DC_FINGERPRINT_SHA256)

    def test_default_payload_is_dc_and_mode_matches(self):
        self.assertEqual(fb.PAYLOAD_DC, "dc")
        self.assertEqual(fb.PAYLOADS[0], fb.PAYLOAD_DC)
        self.assertEqual(fb.PAYLOAD_MODE[fb.PAYLOAD_DC], fb.EDIT_MODE)
        self.assertEqual(fb.PAYLOAD_TENSOR[fb.PAYLOAD_DC], "_features_dc")
        self.assertEqual(fb.PAYLOAD_TENSOR[fb.PAYLOAD_OPACITY], "_opacity")
        self.assertEqual(fb.PAYLOAD_MODE[fb.PAYLOAD_OPACITY], "opacity")

    def test_pre_edit_dc_stats_wrapper_is_the_unannotated_call(self):
        dc = torch.tensor([[0.0], [1.0], [0.9], [0.1]], dtype=torch.float64)
        r_rows, d_rows = fb.nearest_dc_row_map(
            dc, torch.tensor([2, 3]), torch.tensor([0, 1]))
        self.assertEqual(fb.pre_edit_dc_stats(dc, r_rows, d_rows),
                         fb.pre_edit_stats(dc, r_rows, d_rows, None))


class PayloadGeneralizationTests(unittest.TestCase):
    def test_non_dc_payload_adds_neutral_keys_without_dropping_legacy(self):
        result = _dc_pipeline(fb.PAYLOAD_OPACITY)
        for link in result["report"]["links"]:
            self.assertIn("pre_edit_dc_distance_mean", link)
            self.assertIn("pre_edit_distance_mean", link)
            self.assertEqual(link["pre_edit_distance_mean"],
                             link["pre_edit_dc_distance_mean"])
            self.assertEqual(link["payload"], fb.PAYLOAD_OPACITY)
            self.assertEqual(link["payload_tensor"], "_opacity")
        anti = result["report"]["anti_vacuity"]
        self.assertEqual(anti["payload"], fb.PAYLOAD_OPACITY)
        self.assertEqual(anti["l1_pre_edit_distance_mean"],
                         anti["l1_pre_edit_dc_distance_mean"])
        self.assertEqual(anti["l3_pre_edit_distance_mean"],
                         anti["l3_pre_edit_dc_distance_mean"])

    def test_gate_verdict_is_identical_across_payload_annotation(self):
        plain = _dc_pipeline()["report"]
        annotated = _dc_pipeline(fb.PAYLOAD_OPACITY)["report"]
        self.assertEqual(plain["verdict"], annotated["verdict"])
        self.assertEqual(plain["anti_vacuity"]["comparative_ok"],
                         annotated["anti_vacuity"]["comparative_ok"])

    def test_stat_distance_readers_fall_back_to_the_legacy_keys(self):
        legacy = {"pre_edit_dc_distance_mean": 0.25,
                  "pre_edit_dc_distance_max": 0.5}
        self.assertEqual(fb.stat_distance_mean(legacy), 0.25)
        self.assertEqual(fb.stat_distance_max(legacy), 0.5)
        neutral = dict(legacy)
        neutral["pre_edit_distance_mean"] = 0.75
        neutral["pre_edit_distance_max"] = 0.9
        self.assertEqual(fb.stat_distance_mean(neutral), 0.75)
        self.assertEqual(fb.stat_distance_max(neutral), 0.9)

    def test_nearest_row_map_is_the_generalized_nearest_dc_row_map(self):
        dc = torch.tensor([[0.0], [1.0], [0.9], [0.1]], dtype=torch.float64)
        self.assertEqual(
            [t.tolist() for t in fb.nearest_dc_row_map(
                dc, torch.tensor([2, 3]), torch.tensor([0, 1]))],
            [t.tolist() for t in fb.nearest_row_map(
                dc, torch.tensor([2, 3]), torch.tensor([0, 1]))])

    def test_nearest_row_map_follows_the_space_it_is_given(self):
        # DC ranks donor 0 closest to both recipients; opacity ranks
        # donor 1 closest, so the two correspondences must disagree.
        dc = torch.tensor([[0.0], [5.0], [0.1], [0.2]], dtype=torch.float64)
        opacity = torch.tensor([[9.0], [0.0], [0.1], [0.2]],
                               dtype=torch.float64)
        recipients = torch.tensor([2, 3])
        donors = torch.tensor([0, 1])
        _, by_dc = fb.nearest_row_map(dc, recipients, donors)
        _, by_opacity = fb.nearest_row_map(opacity, recipients, donors)
        self.assertEqual(by_dc.tolist(), [0, 0])
        self.assertEqual(by_opacity.tolist(), [1, 1])

    def test_pre_edit_stats_measures_the_tensor_it_is_handed(self):
        values = torch.tensor([[0.0], [1.0], [4.0], [6.0]],
                              dtype=torch.float64)
        stat = fb.pre_edit_stats(values, torch.tensor([2, 3]),
                                 torch.tensor([0, 1]), fb.PAYLOAD_OPACITY)
        self.assertAlmostEqual(stat["pre_edit_distance_mean"], 4.5, places=9)
        self.assertAlmostEqual(stat["pre_edit_distance_max"], 5.0, places=9)
        self.assertEqual(stat["rows_changed"], 2)
        self.assertEqual(stat["payload_tensor"], "_opacity")

    def test_payload_values_refuses_an_unknown_payload(self):
        import types

        model = types.SimpleNamespace(
            _xyz=torch.zeros(3, 3), _opacity=torch.zeros(3, 1))
        with self.assertRaises(ContractError):
            fb.payload_values(model, "geometry")

    def test_payload_values_refuses_an_empty_or_mismatched_tensor(self):
        import types

        empty = types.SimpleNamespace(
            _xyz=torch.zeros(3, 3), _opacity=torch.empty(0))
        with self.assertRaises(ContractError):
            fb.payload_values(empty, fb.PAYLOAD_OPACITY)
        short = types.SimpleNamespace(
            _xyz=torch.zeros(3, 3), _opacity=torch.zeros(2, 1))
        with self.assertRaises(ContractError):
            fb.payload_values(short, fb.PAYLOAD_OPACITY)

    def test_payload_values_returns_the_raw_stored_column(self):
        import types

        model = types.SimpleNamespace(
            _xyz=torch.zeros(3, 3),
            _opacity=torch.tensor([[-2.0], [0.0], [3.0]]))
        values = fb.payload_values(model, fb.PAYLOAD_OPACITY)
        self.assertEqual(values.dtype, torch.float64)
        self.assertEqual(values.shape, (3, 1))
        self.assertEqual(values.reshape(-1).tolist(), [-2.0, 0.0, 3.0])


if __name__ == "__main__":
    unittest.main()
