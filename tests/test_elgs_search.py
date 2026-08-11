"""Unit tests for elgs/search.py (M0 contracts: determinism, caps,
component ordering, one-confirmation-per-component, ITT completeness).
CPU only, unittest."""

import unittest

from depth_visibility.errors import ContractError
from elgs.search import (
    Candidate,
    ScreeningAccumulator,
    conflict_components,
    plan_pass,
)


def _cand(cid, families, score, frames=(0.0, 10.0), footprint=None):
    return Candidate(
        candidate_id=cid,
        op="FISSION",
        family_ids=tuple(families),
        screen_score=score,
        footprint_frames=frames,
        footprint_families=tuple(footprint if footprint is not None else families),
    )


class ScreeningTests(unittest.TestCase):
    def test_signed_bins_and_threshold(self):
        acc = ScreeningAccumulator(bin_width_frames=5.0)
        acc.add(1, 2.0, 0.4)
        acc.add(1, 4.0, 0.3)
        acc.add(1, 7.0, -0.9)
        acc.add(2, 2.0, 0.1)
        hits = acc.bins_above(0.5)
        self.assertEqual(hits, ((1, 0, 0.7), (1, 1, -0.9)))
        acc.reset()
        self.assertEqual(acc.bins_above(0.0), ())

    def test_invalid_inputs(self):
        with self.assertRaises(ContractError):
            ScreeningAccumulator(0.0)
        with self.assertRaises(ContractError):
            ScreeningAccumulator(5.0).bins_above(-1.0)


class ConflictComponentTests(unittest.TestCase):
    def test_components_require_family_and_frame_overlap(self):
        a = _cand("a", [1], 1.0, frames=(0.0, 5.0))
        b = _cand("b", [1], 2.0, frames=(4.0, 9.0))     # conflicts with a
        c = _cand("c", [1], 3.0, frames=(20.0, 25.0))   # same family, no frames
        d = _cand("d", [7], 4.0, frames=(0.0, 5.0))     # no family overlap
        components = conflict_components([a, b, c, d])
        as_sets = [tuple(x.candidate_id for x in comp) for comp in components]
        self.assertIn(("a", "b"), as_sets)
        self.assertIn(("c",), as_sets)
        self.assertIn(("d",), as_sets)

    def test_rank_by_min_lineage_id(self):
        low = _cand("x", [9], 1.0, footprint=[2, 9])
        high = _cand("y", [5], 1.0, footprint=[5])
        components = conflict_components([high, low])
        self.assertEqual(components[0][0].candidate_id, "x")  # min family 2
        self.assertEqual(components[1][0].candidate_id, "y")

    def test_bridge_footprint_creates_conflict(self):
        # Footprint families include BRIDGE families beyond the op's own.
        a = _cand("a", [1], 1.0, footprint=[1, 3])
        b = _cand("b", [2], 2.0, footprint=[2, 3])
        components = conflict_components([a, b])
        self.assertEqual(len(components), 1)


class PassPlanTests(unittest.TestCase):
    def test_one_confirmation_per_component_highest_score_wins(self):
        a = _cand("a", [1], 1.0)
        b = _cand("b", [1], 5.0)
        c = _cand("c", [8], 2.0)
        plan = plan_pass([a, b, c], round_index=0, pass_index=0, candidate_cap=4)
        self.assertEqual([x.candidate_id for x in plan.confirmations], ["b", "c"])
        deferred = [r for r in plan.itt_records
                    if r.rejection_reason == "conflict_deferred_pass_0"]
        self.assertEqual([r.candidate_id for r in deferred], ["a"])
        self.assertEqual(plan.component_ranks, (0, 1))

    def test_cap_rejects_and_logs_never_silently_drops(self):
        cands = [_cand(f"c{i}", [i + 1], float(i)) for i in range(4)]
        plan = plan_pass(cands, round_index=1, pass_index=0, candidate_cap=2)
        self.assertEqual(len(plan.confirmations), 2)
        capped = [r for r in plan.itt_records
                  if r.rejection_reason == "candidate_cap_exceeded"]
        self.assertEqual(len(capped), 2)
        # Inventory completeness: every candidate is either confirmed
        # or ITT-logged.
        seen = {c.candidate_id for c in plan.confirmations}
        seen |= {r.candidate_id for r in plan.itt_records}
        self.assertEqual(seen, {c.candidate_id for c in cands})

    def test_determinism_including_ties(self):
        cands = [
            _cand("t2", [4], 1.0),
            _cand("t1", [4], 1.0),  # tie: candidate_id decides
            _cand("z", [2], 0.5),
        ]
        first = plan_pass(cands, round_index=0, pass_index=0, candidate_cap=4)
        second = plan_pass(list(reversed(cands)), round_index=0, pass_index=0,
                           candidate_cap=4)
        self.assertEqual(
            [c.candidate_id for c in first.confirmations],
            [c.candidate_id for c in second.confirmations],
        )
        self.assertEqual([c.candidate_id for c in first.confirmations], ["z", "t2"])

    def test_invalid_cap_and_candidate(self):
        with self.assertRaises(ContractError):
            plan_pass([], round_index=0, pass_index=0, candidate_cap=0)
        with self.assertRaises(ContractError):
            _cand("bad", [], 0.0)
        with self.assertRaises(ContractError):
            Candidate("b2", "BIRTH", (1,), 0.0, (5.0, 1.0), (1,))


if __name__ == "__main__":
    unittest.main()
