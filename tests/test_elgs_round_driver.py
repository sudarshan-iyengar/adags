"""Unit tests for elgs/round_driver.py (one confirmation pass end to end).

CPU only, unittest. Synthetic sample builders stand in for renders;
oracles: state equality for rejected candidates, registry/runtime
sync for committed ones, ITT inventory completeness, and machinery
rejections (degeneracy) recorded as rejections.
"""

import json
import unittest

import torch

from elgs.acceptance import FrozenSamplerParams, SnisSample
from elgs.clusters import BindingTable
from elgs.families import FamilyRegistry
from elgs.intervals import IntervalConfig, inverse
from elgs.ops import plan_fission
from elgs.round_driver import ProposedCandidate, run_pass
from elgs.runtime import ElgsRuntime, ScheduleAnchors
from elgs.transaction_ledger import SearchCostLedger, TransactionLedger
from elgs.transactions import StateBundle

CFG = IntervalConfig(T=30.0, w_m=1.0, w=1.0, floor_len=2.0, floor_gap=2.0)
SCHEDULE = ScheduleAnchors(2500, 2800, (3000, 4500, 6000), 10000)
PARAMS = FrozenSamplerParams(lambda_u=0.5, pi_d_identity="pi", frozen=True)


def _setup():
    reg = FamilyRegistry()
    reg.create_family(
        birth_time=0.0, birth_site=(0.0, 0.0, 0.0), lineage_key="f0",
        interval=inverse(1, False, False, 2.0, [24.0], [], 6.0, CFG,
                         dtype=torch.float64),
        tau=(0.0,),
    )
    reg.create_family(
        birth_time=0.0, birth_site=(5.0, 0.0, 0.0), lineage_key="f1",
        interval=inverse(2, False, False, 2.0, [10.0, 5.0], [7.0], 8.0, CFG,
                         dtype=torch.float64),
        tau=(0.0, 18.0),
    )
    runtime = ElgsRuntime(reg, CFG, SCHEDULE, dtype=torch.float64)
    bundle = StateBundle(reg, BindingTable(), TransactionLedger(),
                         SearchCostLedger(row_cap=10**6, scalar_budget=10**9))
    return reg, runtime, bundle


def _samples(delta, n=48):
    return [
        SnisSample(
            unit=(i % 4, float((i // 4) % 2)),
            nu_density=0.1,
            mix_density=0.1,
            loss_incumbent=1.0,
            loss_candidate=1.0 + delta,
        )
        for i in range(n)
    ]


def _proposal(reg, family_id, score):
    plan = plan_fission(reg, family_id, 0, 4.0, 6.5, round_index=0,
                        iteration=3000, config=CFG, dtype=torch.float64)
    return ProposedCandidate(plan=plan, screen_score=score,
                             footprint_frames=(0.0, 30.0))


class RunPassTests(unittest.TestCase):
    def test_accepted_candidate_commits_and_syncs(self):
        reg, runtime, bundle = _setup()
        proposal = _proposal(reg, 1, 1.0)
        outcome = run_pass(
            [proposal], runtime, bundle,
            sample_builder=lambda plan, seed, rank: _samples(-0.5),
            sampler_params=PARAMS, k_se=1.0, base_seed=7,
            round_index=0, pass_index=0, iteration=3000, candidate_cap=4,
        )
        self.assertEqual(len(outcome.committed), 1)
        self.assertEqual(reg.get(1).interval.K, 3)
        family_ids = torch.tensor([1])
        # t = 5 falls in the committed fission gap (4, 6.5) => exact 0.
        self.assertEqual(
            float(runtime.presence_multiplier(family_ids, 5.0)[0]), 0.0
        )
        self.assertEqual(bundle.search_cost.candidates_accepted, 1)
        record = outcome.acceptance_records[outcome.committed[0]]
        self.assertTrue(record.accepted)
        self.assertEqual(record.n_samples, 48)

    def test_rejected_candidate_leaves_state_bitwise_unchanged(self):
        reg, runtime, bundle = _setup()
        before = bundle.snapshot()
        proposal = _proposal(reg, 1, 1.0)
        outcome = run_pass(
            [proposal], runtime, bundle,
            sample_builder=lambda plan, seed, rank: _samples(+0.5),  # worsens
            sampler_params=PARAMS, k_se=1.0, base_seed=7,
            round_index=0, pass_index=0, iteration=3000, candidate_cap=4,
        )
        self.assertEqual(outcome.committed, [])
        self.assertEqual(len(outcome.rejected), 1)
        # Scientific state (registry/binding/ledger) is bitwise
        # unchanged; the search-cost ledger legitimately records the
        # tried candidate (accounting, not scientific state).
        after = bundle.snapshot()
        for key in ("registry", "binding", "ledger"):
            self.assertEqual(
                json.dumps(after[key], sort_keys=True),
                json.dumps(before[key], sort_keys=True),
            )
        self.assertEqual(bundle.search_cost.candidates_tried, 1)
        self.assertEqual(bundle.search_cost.candidates_accepted, 0)
        self.assertEqual(reg.get(1).interval.K, 2)
        reasons = [r.rejection_reason for r in outcome.itt if not r.committed]
        self.assertTrue(any("acceptance:" in (r or "") for r in reasons))

    def test_degeneracy_rejection_is_recorded_not_raised(self):
        reg, runtime, bundle = _setup()
        proposal = _proposal(reg, 1, 1.0)
        outcome = run_pass(
            [proposal], runtime, bundle,
            sample_builder=lambda plan, seed, rank: _samples(-0.5, n=4),  # 4 units
            sampler_params=PARAMS, k_se=1.0, base_seed=7,
            round_index=0, pass_index=0, iteration=3000, candidate_cap=4,
        )
        self.assertEqual(outcome.committed, [])
        reasons = [r.rejection_reason for r in outcome.itt]
        self.assertTrue(any("acceptance_machinery" in (r or "") for r in reasons))
        self.assertEqual(bundle.search_cost.candidates_tried, 1)
        self.assertEqual(reg.get(1).interval.K, 2)

    def test_conflict_defers_lower_score_and_itt_is_complete(self):
        reg, runtime, bundle = _setup()
        high = _proposal(reg, 1, 5.0)
        # Same family, same episode, different split point: conflicts
        # with `high` (shared family + overlapping frames) and checks
        # the candidate-id disambiguation by plan detail.
        low = ProposedCandidate(
            plan=plan_fission(reg, 1, 0, 4.5, 7.0, round_index=0,
                              iteration=3000, config=CFG, dtype=torch.float64),
            screen_score=1.0,
            footprint_frames=(0.0, 30.0),
        )
        outcome = run_pass(
            [low, high], runtime, bundle,
            sample_builder=lambda plan, seed, rank: _samples(-0.5),
            sampler_params=PARAMS, k_se=1.0, base_seed=7,
            round_index=0, pass_index=0, iteration=3000, candidate_cap=4,
        )
        self.assertEqual(len(outcome.committed), 1)  # one per component
        all_ids = {r.candidate_id for r in outcome.itt}
        self.assertEqual(len(all_ids), 2)  # winner + deferred both logged

    def test_unfrozen_sampler_refused(self):
        reg, runtime, bundle = _setup()
        with self.assertRaises(Exception):
            run_pass(
                [_proposal(reg, 1, 1.0)], runtime, bundle,
                sample_builder=lambda plan, seed, rank: _samples(-0.5),
                sampler_params=FrozenSamplerParams(0.5, "pi", frozen=False),
                k_se=1.0, base_seed=7, round_index=0, pass_index=0,
                iteration=3000, candidate_cap=4,
            )


if __name__ == "__main__":
    unittest.main()
