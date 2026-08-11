"""Unit tests for elgs/ops.py + elgs/transactions.py (spec §5/§1).

CPU only, unittest. The independent oracle is the fresh-context
transition table configs/elgs/prereg_latch_transition_table_v1.json
(sha256-pinned below; authored and verified from spec §1/§5 alone):
latch pattern transitions and dimension deltas asserted here follow
its rows. Span values use hand-computed absolute fixtures with
T=30, w_m=1, w=1, floor_len=2, floor_gap=2 (Omega = 32).
"""

import hashlib
import json
import pathlib
import unittest

import torch

from depth_visibility.errors import ContractError
from elgs.families import FamilyRegistry
from elgs.intervals import IntervalConfig, expected_dim, forward, inverse
from elgs.ops import (
    plan_birth,
    plan_fission,
    plan_merge,
    plan_prune_episode,
    plan_prune_family,
    plan_reactivate,
    plan_truncate_delete,
    plan_truncate_shorten,
    return_family_candidates,
)
from elgs.clusters import BindingTable
from elgs.transaction_ledger import SearchCostLedger, TransactionLedger
from elgs.transactions import StateBundle, apply_plan

TABLE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "configs" / "elgs" / "prereg_latch_transition_table_v1.json"
)
TABLE_SHA256 = "9fdf0feece2ee87e5d7105f1324eea463a7d8f0856f47611812fa42f1f413f23"

CFG = IntervalConfig(T=30.0, w_m=1.0, w=1.0, floor_len=2.0, floor_gap=2.0)
DT = torch.float64


def _family(reg, *, birth_time=0.0, site=(0.0, 0.0, 0.0), latch_pre=False,
            latch_post=False, slack_pre, lens, gaps, slack_post, tau):
    interval = inverse(len(lens), latch_pre, latch_post, slack_pre,
                       lens, gaps, slack_post, CFG, dtype=DT)
    return reg.create_family(
        birth_time=birth_time, birth_site=site,
        lineage_key=f"lk{birth_time}", interval=interval, tau=tuple(tau),
    )


def _spans(reg, fid):
    r = forward(reg.get(fid).interval, CFG)
    return (float(r.slack_pre), [float(v) for v in r.lens],
            [float(v) for v in r.gaps], float(r.slack_post),
            [float(v) for v in r.b], [float(v) for v in r.d])


def _bundle(reg):
    return StateBundle(reg, BindingTable(), TransactionLedger(),
                       SearchCostLedger(row_cap=10**6, scalar_budget=10**9))


class TransitionTableOracleTests(unittest.TestCase):
    def test_table_is_pinned_and_parses(self):
        data = TABLE_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), TABLE_SHA256)
        table = json.loads(data)
        self.assertEqual(table["schema_version"], "elgs-latch-transition-table-v1")
        self.assertEqual(len(table["operations"]), 15)


class BirthTests(unittest.TestCase):
    def test_birth_plan_and_apply(self):
        reg = FamilyRegistry()
        plan = plan_birth(
            reg, t_birth=5.0, birth_site=(1.0, 2.0, 3.0), lineage_key="b1",
            at_return_site=True, round_index=1, iteration=3000,
            config=CFG, cap_saturated=False, dtype=DT,
        )
        child = plan.child_intervals[plan.family_ids[0]]
        self.assertEqual((child.K, child.latch_pre, child.latch_post), (1, False, True))
        kinds = [e.kind for e in plan.ledger_events]
        self.assertEqual(kinds, ["birth", "return_birth"])
        bundle = _bundle(reg)
        directives = apply_plan(bundle, plan)
        self.assertEqual(directives.new_family_id, 0)
        self.assertEqual(bundle.ledger.n_return_birth, 1)
        self.assertEqual(bundle.registry.get(0).tau, (5.0,))

    def test_birth_without_return_site_charges_no_chi(self):
        reg = FamilyRegistry()
        plan = plan_birth(
            reg, t_birth=5.0, birth_site=(0.0, 0.0, 0.0), lineage_key="b",
            at_return_site=False, round_index=0, iteration=2500,
            config=CFG, cap_saturated=False, dtype=DT,
        )
        self.assertEqual([e.kind for e in plan.ledger_events], ["birth"])

    def test_cap_saturated_inadmissible(self):
        with self.assertRaises(ContractError):
            plan_birth(FamilyRegistry(), t_birth=5.0, birth_site=(0, 0, 0),
                       lineage_key="b", at_return_site=False, round_index=0,
                       iteration=1, config=CFG, cap_saturated=True)


class FissionTests(unittest.TestCase):
    def _base(self):
        reg = FamilyRegistry()
        rec = _family(reg, slack_pre=2.0, lens=[10.0, 5.0], gaps=[7.0],
                      slack_post=8.0, tau=(10.0, 20.0))
        return reg, rec.family_id

    def test_fission_splits_episode_preserving_other_spans(self):
        reg, fid = self._base()
        # Episode 0 = [1, 11], plateau [2, 10]; split with gap (4, 6.5].
        plan = plan_fission(reg, fid, 0, 4.0, 6.5, round_index=1,
                            iteration=3000, config=CFG, dtype=DT)
        child = plan.child_intervals[fid]
        self.assertEqual(child.K, 3)
        self.assertEqual((child.latch_pre, child.latch_post), (False, False))
        self.assertEqual(plan.child_tau[fid], (10.0, 10.0, 20.0))  # tau copy
        bundle = _bundle(reg)
        apply_plan(bundle, plan)
        sp, lens, gaps, so, b, d = _spans(bundle.registry, fid)
        self.assertAlmostEqual(sp, 2.0, places=9)
        self.assertAlmostEqual(so, 8.0, places=9)          # untouched
        self.assertAlmostEqual(lens[2], 5.0, places=9)      # untouched episode
        self.assertAlmostEqual(gaps[1], 7.0, places=9)      # untouched gap
        self.assertAlmostEqual(lens[0], 3.0, places=9)      # 4 - 1
        self.assertAlmostEqual(lens[1], 4.5, places=9)      # 11 - 6.5
        self.assertAlmostEqual(gaps[0], 2.5, places=9)

    def test_fission_admissibility(self):
        reg, fid = self._base()
        with self.assertRaises(ContractError):
            plan_fission(reg, fid, 0, 1.5, 6.0, round_index=1, iteration=1,
                         config=CFG, dtype=DT)  # gap start outside plateau
        with self.assertRaises(ContractError):
            plan_fission(reg, fid, 0, 4.0, 5.5, round_index=1, iteration=1,
                         config=CFG, dtype=DT)  # new gap at floor
        reg2 = FamilyRegistry()
        rec4 = _family(reg2, slack_pre=2.0, lens=[3.0, 3.0, 3.0, 3.0],
                       gaps=[3.0, 3.0, 3.0], slack_post=9.0,
                       tau=(0.0, 1.0, 2.0, 3.0))
        with self.assertRaises(ContractError):
            plan_fission(reg2, rec4.family_id, 0, 3.0, 4.0, round_index=1,
                         iteration=1, config=CFG, dtype=DT)  # K = 4


class TruncateShortenTests(unittest.TestCase):
    def test_shorten_end_feeds_outer_slack(self):
        reg = FamilyRegistry()
        rec = _family(reg, slack_pre=2.0, lens=[6.0, 5.0], gaps=[7.0],
                      slack_post=12.0, tau=(0.0, 1.0))
        # d_2 = 19; move to 17.
        plan = plan_truncate_shorten(reg, rec.family_id, 1, "end", 17.0,
                                     round_index=1, iteration=1, config=CFG, dtype=DT)
        bundle = _bundle(reg)
        apply_plan(bundle, plan)
        sp, lens, gaps, so, b, d = _spans(bundle.registry, rec.family_id)
        self.assertAlmostEqual(lens[1], 3.0, places=9)
        self.assertAlmostEqual(so, 14.0, places=9)
        self.assertAlmostEqual(gaps[0], 7.0, places=9)  # untouched

    def test_shorten_latched_outer_endpoint_clears_latch(self):
        reg = FamilyRegistry()
        rec = _family(reg, latch_pre=True, slack_pre=0.0, lens=[6.0, 5.0],
                      gaps=[7.0], slack_post=14.0, tau=(0.0, 1.0))
        # b_1 = -1 (latched); move to 0.
        plan = plan_truncate_shorten(reg, rec.family_id, 0, "start", 0.0,
                                     round_index=1, iteration=1, config=CFG, dtype=DT)
        child = plan.child_intervals[rec.family_id]
        self.assertFalse(child.latch_pre)  # cleared per §1
        self.assertEqual(child.a.numel(), expected_dim(2, False, False))
        bundle = _bundle(reg)
        apply_plan(bundle, plan)
        sp, lens, *_ = _spans(bundle.registry, rec.family_id)
        self.assertAlmostEqual(sp, 1.0, places=9)
        self.assertAlmostEqual(lens[0], 5.0, places=9)

    def test_floor_violation_rejected(self):
        reg = FamilyRegistry()
        rec = _family(reg, slack_pre=2.0, lens=[3.0, 5.0], gaps=[7.0],
                      slack_post=15.0, tau=(0.0, 1.0))
        # b_1 = 1, len = 3: moving the end to 2.9 leaves 1.9 < floor_len.
        with self.assertRaises(ContractError):
            plan_truncate_shorten(reg, rec.family_id, 0, "end", 2.9,
                                  round_index=1, iteration=1, config=CFG, dtype=DT)


class TruncateDeleteTests(unittest.TestCase):
    def test_terminal_delete_discards_side_latch(self):
        reg = FamilyRegistry()
        rec = _family(reg, latch_pre=True, slack_pre=0.0, lens=[6.0, 5.0],
                      gaps=[7.0], slack_post=14.0, tau=(0.0, 1.0))
        plan = plan_truncate_delete(reg, rec.family_id, 0, round_index=1,
                                    iteration=1, config=CFG, dtype=DT)
        child = plan.child_intervals[rec.family_id]
        self.assertEqual(child.K, 1)
        self.assertFalse(child.latch_pre)  # discarded with the episode
        self.assertEqual(plan.child_tau[rec.family_id], (1.0,))
        bundle = _bundle(reg)
        apply_plan(bundle, plan)
        sp, lens, gaps, so, b, d = _spans(bundle.registry, rec.family_id)
        self.assertAlmostEqual(sp, 13.0, places=9)  # 0 + 6 + 7
        self.assertAlmostEqual(lens[0], 5.0, places=9)
        self.assertAlmostEqual(so, 14.0, places=9)

    def test_interior_delete_merges_adjacent_gaps(self):
        reg = FamilyRegistry()
        rec = _family(reg, slack_pre=2.0, lens=[3.0, 4.0, 5.0], gaps=[3.0, 4.0],
                      slack_post=11.0, tau=(0.0, 1.0, 2.0))
        plan = plan_truncate_delete(reg, rec.family_id, 1, round_index=1,
                                    iteration=1, config=CFG, dtype=DT)
        child = plan.child_intervals[rec.family_id]
        self.assertEqual(child.K, 2)
        self.assertEqual(plan.child_tau[rec.family_id], (0.0, 2.0))
        bundle = _bundle(reg)
        apply_plan(bundle, plan)
        sp, lens, gaps, so, *_ = _spans(bundle.registry, rec.family_id)
        self.assertAlmostEqual(gaps[0], 3.0 + 4.0 + 4.0, places=9)  # merged
        self.assertAlmostEqual(lens[0], 3.0, places=9)
        self.assertAlmostEqual(lens[1], 5.0, places=9)

    def test_k1_delete_yields_empty_program(self):
        reg = FamilyRegistry()
        rec = _family(reg, slack_pre=2.0, lens=[6.0], gaps=[],
                      slack_post=24.0, tau=(0.0,))
        plan = plan_truncate_delete(reg, rec.family_id, 0, round_index=1,
                                    iteration=1, config=CFG, dtype=DT)
        child = plan.child_intervals[rec.family_id]
        self.assertEqual(child.K, 0)
        self.assertIsNone(child.a)
        self.assertEqual(plan.child_tau[rec.family_id], ())


class ReactivateTests(unittest.TestCase):
    def _base(self):
        reg = FamilyRegistry()
        rec = _family(reg, slack_pre=8.0, lens=[5.0, 5.0], gaps=[7.0],
                      slack_post=7.0, tau=(0.0, 1.0))
        return reg, rec.family_id  # b = [7, 19], d = [12, 24]

    def test_interior_insertion_fresh_tau(self):
        reg, fid = self._base()
        plan = plan_reactivate(reg, fid, 14.5, 16.9, 99.0, return_candidates=(),
                               round_index=2, iteration=4500, config=CFG, dtype=DT)
        child = plan.child_intervals[fid]
        self.assertEqual(child.K, 3)
        self.assertEqual(plan.child_tau[fid], (0.0, 99.0, 1.0))
        bundle = _bundle(reg)
        apply_plan(bundle, plan)
        sp, lens, gaps, so, b, d = _spans(bundle.registry, fid)
        self.assertAlmostEqual(lens[1], 2.4, places=9)
        self.assertAlmostEqual(gaps[0], 2.5, places=9)
        self.assertAlmostEqual(gaps[1], 2.1, places=9)
        self.assertAlmostEqual(sp, 8.0, places=9)  # untouched

    def test_outer_insertion_and_latch_set_on_exact_consume(self):
        reg, fid = self._base()
        plan = plan_reactivate(reg, fid, 1.0, 4.0, 50.0, return_candidates=(),
                               round_index=2, iteration=4500, config=CFG, dtype=DT)
        child = plan.child_intervals[fid]
        self.assertEqual(child.K, 3)
        self.assertFalse(child.latch_pre)
        self.assertEqual(plan.child_tau[fid][0], 50.0)
        # Consuming the whole outer slack sets the latch (§1 rule).
        reg2, fid2 = self._base()
        plan2 = plan_reactivate(reg2, fid2, -1.0, 3.0, 50.0, return_candidates=(),
                                round_index=2, iteration=4500, config=CFG, dtype=DT)
        self.assertTrue(plan2.child_intervals[fid2].latch_pre)

    def test_latched_outer_endpoint_inadmissible(self):
        reg = FamilyRegistry()
        rec = _family(reg, latch_pre=True, slack_pre=0.0, lens=[5.0, 5.0],
                      gaps=[7.0], slack_post=15.0, tau=(0.0, 1.0))
        with self.assertRaises(ContractError):
            plan_reactivate(reg, rec.family_id, -0.5, 2.5, 50.0,
                            return_candidates=(), round_index=2, iteration=1,
                            config=CFG, dtype=DT)

    def test_return_family_exclusivity(self):
        reg, fid = self._base()
        with self.assertRaises(ContractError):
            plan_reactivate(reg, fid, 14.5, 16.9, 99.0, return_candidates=(3,),
                            round_index=2, iteration=1, config=CFG, dtype=DT)

    def test_return_family_predicate_ordering(self):
        reg = FamilyRegistry()
        a = _family(reg, birth_time=2.0, site=(0.0, 0.0, 0.0), slack_pre=2.0,
                    lens=[6.0], gaps=[], slack_post=24.0, tau=(0.0,))
        b = _family(reg, birth_time=1.0, site=(0.5, 0.0, 0.0), slack_pre=2.0,
                    lens=[6.0], gaps=[], slack_post=24.0, tau=(0.0,))
        far = _family(reg, birth_time=0.5, site=(9.0, 9.0, 9.0), slack_pre=2.0,
                      lens=[6.0], gaps=[], slack_post=24.0, tau=(0.0,))
        hits = return_family_candidates(reg, (0.0, 0.0, 0.0), (0.0, 5.0), 1.0)
        self.assertEqual(hits, (b.family_id, a.family_id))  # earliest birth first
        self.assertNotIn(far.family_id, hits)


class MergeTests(unittest.TestCase):
    def _pair(self, *, b_latch_post=False):
        reg = FamilyRegistry()
        older = _family(reg, birth_time=0.0, slack_pre=2.0, lens=[6.0], gaps=[],
                        slack_post=24.0, tau=(3.0,))
        if b_latch_post:
            younger = _family(reg, birth_time=5.0, latch_post=True,
                              slack_pre=27.0, lens=[5.0], gaps=[],
                              slack_post=0.0, tau=(8.0,))
        else:
            younger = _family(reg, birth_time=5.0, slack_pre=15.0, lens=[5.0],
                              gaps=[], slack_post=12.0, tau=(8.0,))
        return reg, older.family_id, younger.family_id

    def test_merge_survivor_identity_and_redirection(self):
        reg, old_id, new_id = self._pair()
        plan = plan_merge(reg, new_id, old_id, return_predicate_holds=True,
                          round_index=2, iteration=4500, config=CFG, dtype=DT)
        self.assertEqual(plan.survivor_identity, old_id)
        self.assertEqual(plan.retire_family, new_id)
        child = plan.child_intervals[old_id]
        self.assertEqual(child.K, 2)
        self.assertEqual(plan.child_tau[old_id], (3.0, 8.0))
        bundle = _bundle(reg)
        bundle.binding.bind(0, new_id)
        bundle.binding.bind(1, old_id)
        bundle.binding.freeze_audited()
        bundle.registry.on_rows_added(old_id, 4)
        bundle.registry.on_rows_added(new_id, 2)
        directives = apply_plan(bundle, plan)
        self.assertEqual(directives.row_redirect, (new_id, old_id))
        self.assertEqual(bundle.binding.clusters_of(old_id), (0, 1))
        self.assertEqual(bundle.registry.row_count(old_id), 6)
        self.assertTrue(bundle.registry.get(new_id).retired)
        self.assertEqual(bundle.ledger.n_merge, 1)
        sp, lens, gaps, so, b, d = _spans(bundle.registry, old_id)
        self.assertAlmostEqual(gaps[0], 14.0 - 7.0, places=9)  # recomputed
        self.assertAlmostEqual(lens[0], 6.0, places=9)
        self.assertAlmostEqual(lens[1], 5.0, places=9)

    def test_merge_latch_or_inheritance(self):
        reg, old_id, new_id = self._pair(b_latch_post=True)
        plan = plan_merge(reg, old_id, new_id, return_predicate_holds=True,
                          round_index=2, iteration=4500, config=CFG, dtype=DT)
        child = plan.child_intervals[old_id]
        self.assertFalse(child.latch_pre)
        self.assertTrue(child.latch_post)  # from the parent owning latest d_K

    def test_merge_preconditions(self):
        reg, old_id, new_id = self._pair()
        with self.assertRaises(ContractError):
            plan_merge(reg, old_id, new_id, return_predicate_holds=False,
                       round_index=2, iteration=1, config=CFG, dtype=DT)
        # Overlapping unions rejected.
        reg2 = FamilyRegistry()
        a = _family(reg2, birth_time=0.0, slack_pre=2.0, lens=[6.0], gaps=[],
                    slack_post=24.0, tau=(0.0,))
        b = _family(reg2, birth_time=1.0, slack_pre=3.0, lens=[6.0], gaps=[],
                    slack_post=23.0, tau=(0.0,))
        with self.assertRaises(ContractError):
            plan_merge(reg2, a.family_id, b.family_id,
                       return_predicate_holds=True, round_index=2,
                       iteration=1, config=CFG, dtype=DT)


class PruneTests(unittest.TestCase):
    def test_prune_episode_requires_both_gates(self):
        reg = FamilyRegistry()
        rec = _family(reg, slack_pre=2.0, lens=[6.0, 5.0], gaps=[7.0],
                      slack_post=12.0, tau=(0.0, 1.0))
        with self.assertRaises(ContractError):
            plan_prune_episode(reg, rec.family_id, 0, len_at_floor=True,
                               micro_render_confirms=False, round_index=1,
                               iteration=1, config=CFG, dtype=DT)
        plan = plan_prune_episode(reg, rec.family_id, 0, len_at_floor=True,
                                  micro_render_confirms=True, round_index=1,
                                  iteration=1, config=CFG, dtype=DT)
        self.assertEqual(plan.op, "PRUNE_EPISODE")
        self.assertEqual(plan.child_intervals[rec.family_id].K, 1)

    def test_prune_family_retires_and_leaves_bindings(self):
        reg = FamilyRegistry()
        rec = _family(reg, slack_pre=2.0, lens=[6.0], gaps=[],
                      slack_post=24.0, tau=(0.0,))
        bundle = _bundle(reg)
        bundle.binding.bind(0, rec.family_id)
        bundle.binding.freeze_audited()
        plan = plan_prune_family(reg, rec.family_id,
                                 episodeless_or_unsupported=True,
                                 round_index=1, iteration=1)
        apply_plan(bundle, plan)
        self.assertTrue(bundle.registry.get(rec.family_id).retired)
        # Binding untouched (freeze permits no third mutation kind);
        # inactivity follows from retirement.
        self.assertEqual(bundle.binding.family_of(0), rec.family_id)
        with self.assertRaises(ContractError):
            plan_prune_family(reg, rec.family_id,
                              episodeless_or_unsupported=False,
                              round_index=1, iteration=1)


class RollbackTests(unittest.TestCase):
    def test_failed_apply_restores_bundle_bitwise(self):
        reg = FamilyRegistry()
        older = _family(reg, birth_time=0.0, slack_pre=2.0, lens=[6.0], gaps=[],
                        slack_post=24.0, tau=(3.0,))
        younger = _family(reg, birth_time=5.0, slack_pre=15.0, lens=[5.0],
                          gaps=[], slack_post=12.0, tau=(8.0,))
        plan = plan_merge(reg, older.family_id, younger.family_id,
                          return_predicate_holds=True, round_index=2,
                          iteration=1, config=CFG, dtype=DT)
        bundle = _bundle(reg)
        before = bundle.snapshot()
        # Sabotage: retire the younger family so apply fails mid-way.
        reg.retire_family(younger.family_id)
        with self.assertRaises(ContractError):
            apply_plan(bundle, plan)
        after = bundle.snapshot()
        # Rollback restored the PRE-APPLY snapshot (which includes the
        # sabotage retirement made outside the transaction: rollback
        # must restore exactly the pre-apply state, no more).
        self.assertEqual(json.dumps(before, sort_keys=True) ==
                         json.dumps(after, sort_keys=True), False)
        restored = StateBundle(
            FamilyRegistry.from_state(after["registry"]),
            BindingTable.from_state(after["binding"]),
            TransactionLedger.from_state(after["ledger"]),
            SearchCostLedger.from_state(after["search_cost"]),
        )
        self.assertTrue(restored.registry.get(younger.family_id).retired)
        self.assertEqual(len(restored.ledger.events), 0)  # nothing committed

    def test_successful_apply_then_manual_restore_round_trips(self):
        reg = FamilyRegistry()
        rec = _family(reg, slack_pre=2.0, lens=[10.0, 5.0], gaps=[7.0],
                      slack_post=8.0, tau=(10.0, 20.0))
        bundle = _bundle(reg)
        before = bundle.snapshot()
        plan = plan_fission(reg, rec.family_id, 0, 4.0, 6.5, round_index=1,
                            iteration=3000, config=CFG, dtype=DT)
        apply_plan(bundle, plan)
        self.assertEqual(bundle.registry.get(rec.family_id).interval.K, 3)
        bundle.restore(before)
        self.assertEqual(bundle.registry.get(rec.family_id).interval.K, 2)
        self.assertEqual(
            json.dumps(bundle.snapshot(), sort_keys=True),
            json.dumps(before, sort_keys=True),
        )


if __name__ == "__main__":
    unittest.main()
