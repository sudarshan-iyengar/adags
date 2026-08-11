"""Unit tests for elgs/state_io.py, transaction_ledger.py, and the
registry/binding serialization round-trips (implementation plan §9).

CPU only, unittest. Oracles: deterministic fixtures, round-trip
equality on live-object behavior (not just dict equality), and
deliberately corrupted payloads for the explicit-rejection paths.
"""

import unittest

import torch

from depth_visibility.errors import ContractError, SchemaError
from elgs.clusters import BindingTable
from elgs.families import FamilyRegistry
from elgs.intervals import IntervalState, empty_program
from elgs.state_io import ELGS_STATE_SCHEMA, build_elgs_state, load_elgs_state
from elgs.transaction_ledger import (
    LedgerEvent,
    SearchCostLedger,
    TransactionLedger,
)


def _registry():
    reg = FamilyRegistry()
    reg.create_family(
        birth_time=0.0, birth_site=(0.0, 0.0, 0.0), lineage_key="root",
        interval=IntervalState(K=2, latch_pre=True, latch_post=False,
                               a=torch.tensor([0.1, -0.2, 0.3, 0.0])),
        tau=(0.0, 5.0),
    )
    rec = reg.create_family(
        birth_time=3.0, birth_site=(1.0, 2.0, 3.0), lineage_key="late",
        interval=IntervalState(K=1, latch_pre=False, latch_post=True,
                               a=torch.tensor([0.0, 1.0])),
        tau=(4.0,),
    )
    reg.on_rows_added(0, 7)
    reg.on_rows_added(rec.family_id, 3)
    reg.retire_family(rec.family_id)
    return reg


def _binding():
    table = BindingTable()
    table.bind(0, 0)
    table.bind(1, None)
    table.freeze_audited()
    table.bind_late_birth(2, 1)
    return table


def _ledger():
    ledger = TransactionLedger()
    ledger.append(LedgerEvent("birth", 0, 2500, (0,)))
    ledger.append(LedgerEvent("return_birth", 1, 3000, (1,), {"site": [1, 2, 3]}))
    ledger.append(LedgerEvent("merge", 1, 3000, (0, 1)))
    ledger.append(LedgerEvent("rollback", 1, 3000, (1,), {"reason": "rejected"}))
    return ledger


def _search_cost():
    ledger = SearchCostLedger(row_cap=600_000, scalar_budget=10_000_000)
    ledger.observe_rows(500_000, 9_000_000)
    ledger.candidates_tried = 4
    ledger.candidates_accepted = 1
    return ledger


def _state():
    return build_elgs_state(
        _registry(), _binding(), _ledger(), _search_cost(),
        sampler={"lambda_u": 0.5, "pi_d_identity": "pi-d-v1", "frozen": True},
        slot_grid={"n_rounds": 3, "n_passes": 2, "slots_per_pass": 4,
                   "units_per_slot": 8, "consumed": [0, 5],
                   "reserved_pool": [[0, 0.0], [1, 2.0]]},
        confirmation_refs={"decision-1": [(0, 2.0), (1, 3.0)]},
        moment_reset_log=[{"family_id": 0, "iteration": 3000, "op": "fission"}],
        round_bookkeeping={"round_index": 1, "last_refresh_iteration": 3000},
        rng={"python": "0xabc"},
    )


class LedgerTests(unittest.TestCase):
    def test_append_only_counts_and_charge(self):
        ledger = _ledger()
        self.assertEqual(ledger.n_return_birth, 1)
        self.assertEqual(ledger.n_merge, 1)
        self.assertAlmostEqual(ledger.charge(chi=1.5, mu=2.0), 3.5, places=12)
        with self.assertRaises(ContractError):
            ledger.charge(-1.0, 0.0)
        with self.assertRaises(ContractError):
            LedgerEvent("not-a-kind", 0, 0, ())

    def test_ledger_round_trip(self):
        restored = TransactionLedger.from_state(_ledger().to_state())
        self.assertEqual(restored.n_return_birth, 1)
        self.assertEqual(restored.n_merge, 1)
        self.assertEqual(len(restored.events), 4)
        self.assertEqual(restored.events[1].detail, {"site": [1, 2, 3]})

    def test_search_cost_caps_fail_closed(self):
        ledger = _search_cost()
        with self.assertRaises(ContractError):
            ledger.observe_rows(600_001, 0)
        with self.assertRaises(ContractError):
            ledger.observe_rows(0, 10_000_001)
        restored = SearchCostLedger.from_state(ledger.to_state())
        self.assertEqual(restored.peak_rendered_rows, 500_000)
        self.assertEqual(restored.candidates_tried, 4)


class RegistryRoundTripTests(unittest.TestCase):
    def test_registry_round_trip_preserves_behavior(self):
        source = _registry()
        restored = FamilyRegistry.from_state(source.to_state())
        self.assertEqual(restored.active_ids(), source.active_ids())
        self.assertEqual(restored.next_family_id, source.next_family_id)
        self.assertEqual(restored.row_count(0), 7)
        rec = restored.get(0)
        self.assertEqual(rec.interval.K, 2)
        self.assertTrue(rec.interval.latch_pre)
        self.assertEqual(rec.tau, (0.0, 5.0))
        self.assertTrue(restored.get(1).retired)
        with self.assertRaises(ContractError):
            restored.require_active(1)
        # The watermark survives: a new family gets id 2, not a reuse.
        new = restored.create_family(
            birth_time=9.0, birth_site=(0.0, 0.0, 0.0), lineage_key="post",
            interval=empty_program(), tau=(),
        )
        self.assertEqual(new.family_id, 2)

    def test_watermark_regression_rejected(self):
        state = _registry().to_state()
        state["next_family_id"] = 1  # below max existing id 1? equal -> reuse
        with self.assertRaises(ContractError):
            FamilyRegistry.from_state(state)

    def test_duplicate_family_rejected(self):
        state = _registry().to_state()
        state["records"].append(dict(state["records"][0]))
        with self.assertRaises(ContractError):
            FamilyRegistry.from_state(state)


class BindingRoundTripTests(unittest.TestCase):
    def test_binding_round_trip_preserves_freeze_and_log(self):
        source = _binding()
        restored = BindingTable.from_state(source.to_state())
        self.assertTrue(restored.frozen)
        self.assertEqual(restored.family_of(0), 0)
        self.assertIsNone(restored.family_of(1))
        self.assertEqual(restored.family_of(2), 1)
        self.assertEqual(
            [e["kind"] for e in restored.mutation_log], ["late_birth_bind"]
        )
        with self.assertRaises(ContractError):
            restored.bind(0, 5)  # still frozen after restore


class ElgsStateTests(unittest.TestCase):
    def test_full_state_round_trip(self):
        loaded = load_elgs_state(_state())
        self.assertEqual(loaded["registry"].row_count(0), 7)
        self.assertTrue(loaded["binding"].frozen)
        self.assertEqual(loaded["ledger"].n_merge, 1)
        self.assertEqual(loaded["search_cost"].peak_rendered_rows, 500_000)
        self.assertEqual(loaded["sampler"]["lambda_u"], 0.5)
        self.assertEqual(loaded["confirmation_refs"]["decision-1"], [(0, 2.0), (1, 3.0)])
        self.assertEqual(loaded["round_bookkeeping"]["round_index"], 1)

    def test_incompatible_schema_rejected_explicitly(self):
        payload = _state()
        payload["schema_version"] = "elgs-state-v0"
        with self.assertRaises(SchemaError):
            load_elgs_state(payload)

    def test_unknown_and_missing_keys_rejected(self):
        payload = _state()
        payload["surprise"] = 1
        with self.assertRaises(SchemaError):
            load_elgs_state(payload)
        payload = _state()
        del payload["ledger"]
        with self.assertRaises(SchemaError):
            load_elgs_state(payload)

    def test_corrupted_interval_dimension_rejected_at_load(self):
        payload = _state()
        payload["registry"]["records"][0]["interval"]["a"] = [0.1, 0.2]  # wrong dim
        with self.assertRaises(ContractError):
            load_elgs_state(payload)

    def test_sampler_state_validated_at_build(self):
        with self.assertRaises(ContractError):
            build_elgs_state(
                _registry(), _binding(), _ledger(), _search_cost(),
                sampler={"lambda_u": 0.5},  # missing keys
                slot_grid={}, confirmation_refs={}, moment_reset_log=[],
                round_bookkeeping={},
            )

    def test_schema_constant(self):
        self.assertEqual(ELGS_STATE_SCHEMA, "elgs-state-v1")
        self.assertEqual(_state()["schema_version"], ELGS_STATE_SCHEMA)


if __name__ == "__main__":
    unittest.main()
