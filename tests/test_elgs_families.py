"""Unit tests for elgs/families.py registry invariants.

CPU only, unittest. Covers: ID watermark (retired IDs never reused),
MERGE survivor convention (older birth time, ties by lower ID),
tau-count consistency, routing-pin predicate + log, row bookkeeping.
"""

import unittest

import torch

from depth_visibility.errors import ContractError
from elgs.families import FamilyRecord, FamilyRegistry
from elgs.intervals import IntervalState, empty_program


def _interval(K, latch_pre=False, latch_post=False):
    if K == 0:
        return empty_program()
    dim = 2 * K + 1 - (int(latch_pre) + int(latch_post))
    return IntervalState(
        K=K, latch_pre=latch_pre, latch_post=latch_post,
        a=torch.zeros(dim, dtype=torch.float64),
    )


def _create(reg, birth_time, K=1, tau=None):
    return reg.create_family(
        birth_time=birth_time,
        birth_site=(0.0, 0.0, 0.0),
        lineage_key=f"lk-{birth_time}",
        interval=_interval(K),
        tau=tau if tau is not None else tuple(float(k) for k in range(K)),
    )


class IdentityWatermarkTests(unittest.TestCase):
    def test_ids_monotone_and_never_reused(self):
        reg = FamilyRegistry()
        first = _create(reg, 0.0)
        second = _create(reg, 1.0)
        self.assertEqual((first.family_id, second.family_id), (0, 1))
        reg.retire_family(first.family_id)
        third = _create(reg, 2.0)
        self.assertEqual(third.family_id, 2)
        self.assertEqual(reg.next_family_id, 3)
        self.assertNotIn(first.family_id, reg.active_ids())

    def test_retired_family_rejects_mutation(self):
        reg = FamilyRegistry()
        rec = _create(reg, 0.0)
        reg.retire_family(rec.family_id)
        with self.assertRaises(ContractError):
            reg.require_active(rec.family_id)
        with self.assertRaises(ContractError):
            reg.retire_family(rec.family_id)
        with self.assertRaises(ContractError):
            reg.replace_interval(rec.family_id, _interval(1), (0.0,))


class MergeSurvivorTests(unittest.TestCase):
    def test_older_birth_time_survives(self):
        reg = FamilyRegistry()
        older = _create(reg, 1.0)
        younger = _create(reg, 5.0)
        self.assertEqual(
            reg.merge_survivor_of(younger.family_id, older.family_id),
            (older.family_id, younger.family_id),
        )

    def test_birth_time_tie_broken_by_lower_id(self):
        reg = FamilyRegistry()
        first = _create(reg, 3.0)
        second = _create(reg, 3.0)
        self.assertEqual(
            reg.merge_survivor_of(second.family_id, first.family_id),
            (first.family_id, second.family_id),
        )

    def test_merge_with_self_rejected(self):
        reg = FamilyRegistry()
        rec = _create(reg, 0.0)
        with self.assertRaises(ContractError):
            reg.merge_survivor_of(rec.family_id, rec.family_id)


class TauAndPinTests(unittest.TestCase):
    def test_tau_count_must_match_k(self):
        reg = FamilyRegistry()
        with self.assertRaises(ContractError):
            reg.create_family(
                birth_time=0.0, birth_site=(0.0, 0.0, 0.0), lineage_key="x",
                interval=_interval(2), tau=(0.0,),
            )
        with self.assertRaises(ContractError):
            FamilyRecord(
                family_id=0, birth_time=0.0, birth_site=(0.0, 0.0, 0.0),
                lineage_key="x", interval=empty_program(), tau=(1.0,),
            )

    def test_tau_is_immutable(self):
        reg = FamilyRegistry()
        rec = _create(reg, 0.0, K=2)
        with self.assertRaises((AttributeError, TypeError)):
            rec.tau = (9.0, 9.0)  # type: ignore[misc]

    def test_routing_pin_predicate_and_log(self):
        reg = FamilyRegistry()
        single = _create(reg, 0.0, K=1)
        multi = _create(reg, 1.0, K=2)
        self.assertFalse(single.routing_pinned)
        self.assertTrue(multi.routing_pinned)
        pinned_ids = {entry["family_id"] for entry in reg.pin_log}
        self.assertEqual(pinned_ids, {multi.family_id})
        # A K-change into multi-episode form must append a pin entry.
        reg.replace_interval(single.family_id, _interval(2), (0.0, 1.0))
        pinned_ids = {entry["family_id"] for entry in reg.pin_log}
        self.assertEqual(pinned_ids, {single.family_id, multi.family_id})


class RowBookkeepingTests(unittest.TestCase):
    def test_row_counts_and_merge_redirection(self):
        reg = FamilyRegistry()
        a = _create(reg, 0.0)
        b = _create(reg, 1.0)
        reg.on_rows_added(a.family_id, 5)
        reg.on_rows_added(b.family_id, 3)
        reg.on_rows_pruned(b.family_id, 1)
        self.assertEqual(reg.row_count(a.family_id), 5)
        self.assertEqual(reg.row_count(b.family_id), 2)
        moved = reg.redirect_rows(b.family_id, a.family_id)
        self.assertEqual(moved, 2)
        self.assertEqual(reg.row_count(a.family_id), 7)
        self.assertEqual(reg.row_count(b.family_id), 0)

    def test_overprune_rejected(self):
        reg = FamilyRegistry()
        a = _create(reg, 0.0)
        reg.on_rows_added(a.family_id, 2)
        with self.assertRaises(ContractError):
            reg.on_rows_pruned(a.family_id, 3)


if __name__ == "__main__":
    unittest.main()
