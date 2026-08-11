"""CPU integration tests (implementation plan §8.6 I1/I3/I4 core).

Exercises the runtime glue the trainer and renderer call: per-row
presence gather with exact-zero absence, gradient flow from a render
surrogate into the a-logits (the only differentiable path into
interval endpoints), pure-plan/commit/rollback flow with runtime
sync + moment-reset directives, and the checkpoint round trip
(flush -> build_elgs_state -> load -> identical presence).
"""

import unittest

import torch

from depth_visibility.errors import ContractError
from elgs.clusters import BindingTable
from elgs.families import FamilyRegistry
from elgs.intervals import IntervalConfig, inverse
from elgs.ops import plan_fission
from elgs.runtime import ElgsRuntime, ScheduleAnchors
from elgs.state_io import build_elgs_state, load_elgs_state
from elgs.transaction_ledger import SearchCostLedger, TransactionLedger
from elgs.transactions import StateBundle, apply_plan

CFG = IntervalConfig(T=30.0, w_m=1.0, w=1.0, floor_len=2.0, floor_gap=2.0)
SCHEDULE = ScheduleAnchors(
    seed_iteration=2500,
    audit_iteration=2800,
    round_iterations=(3000, 4500, 6000),
    refit_until=10000,
)


def _setup():
    reg = FamilyRegistry()
    # Family 0: K=1 spanning-ish [1, 25]; family 1: K=2 with a gap.
    reg.create_family(
        birth_time=0.0, birth_site=(0.0, 0.0, 0.0), lineage_key="span",
        interval=inverse(1, False, False, 2.0, [24.0], [], 6.0, CFG,
                         dtype=torch.float64),
        tau=(0.0,),
    )
    reg.create_family(
        birth_time=0.0, birth_site=(1.0, 0.0, 0.0), lineage_key="gap",
        interval=inverse(2, False, False, 2.0, [10.0, 5.0], [7.0], 8.0, CFG,
                         dtype=torch.float64),
        tau=(0.0, 18.0),
    )
    runtime = ElgsRuntime(reg, CFG, SCHEDULE, dtype=torch.float64)
    # Rows: 3 of family 0, 2 of family 1.
    family_ids = torch.tensor([0, 0, 0, 1, 1])
    reg.on_rows_added(0, 3)
    reg.on_rows_added(1, 2)
    return reg, runtime, family_ids


class PresenceIntegrationTests(unittest.TestCase):
    def test_gather_plateau_and_exact_zero_in_gap(self):
        _, runtime, family_ids = _setup()
        # Family 1 episodes: [1, 11] and [18, 23]; gap (11, 18).
        presence_mid_gap = runtime.presence_multiplier(family_ids, 14.0)
        self.assertEqual(presence_mid_gap.shape, (5, 1))
        self.assertEqual(float(presence_mid_gap[3]), 0.0)  # exact zero
        self.assertEqual(float(presence_mid_gap[4]), 0.0)
        self.assertEqual(float(presence_mid_gap[0]), 1.0)  # family 0 plateau
        presence_plateau = runtime.presence_multiplier(family_ids, 5.0)
        self.assertEqual(float(presence_plateau[3]), 1.0)
        self.assertEqual(float(presence_plateau[0]), 1.0)

    def test_retired_and_empty_families_render_nothing(self):
        reg, runtime, family_ids = _setup()
        reg.retire_family(1)
        runtime.sync_family_from_registry(1)
        presence = runtime.presence_multiplier(family_ids, 5.0)
        self.assertEqual(float(presence[3]), 0.0)
        self.assertEqual(float(presence[4]), 0.0)

    def test_gradient_flows_into_a_logits_on_edges(self):
        _, runtime, family_ids = _setup()
        # t = 11 sits on family 1's first falling edge (d=11 => pi=0
        # exactly AT d, so probe just inside: t = 10.5 has pi in (0,1)).
        presence = runtime.presence_multiplier(family_ids, 10.5)
        loss = presence.sum()
        loss.backward()
        grads = {fid: t.grad for fid, t in runtime.logit_parameters().items()}
        assert grads[1] is not None
        self.assertTrue(torch.isfinite(grads[1]).all())
        self.assertGreater(float(grads[1].abs().sum()), 0.0)

    def test_plateau_gradient_is_zero_but_graph_intact(self):
        _, runtime, family_ids = _setup()
        presence = runtime.presence_multiplier(family_ids, 5.0)
        presence.sum().backward()
        grad = runtime.logit_parameters()[0].grad
        assert grad is not None
        self.assertEqual(float(grad.abs().sum()), 0.0)  # saturated plateau


class StructuralCommitIntegrationTests(unittest.TestCase):
    def test_plan_is_pure_then_commit_syncs_runtime(self):
        reg, runtime, family_ids = _setup()
        bundle = StateBundle(reg, BindingTable(), TransactionLedger(),
                             SearchCostLedger(row_cap=10**6, scalar_budget=10**9))
        before_presence = float(
            runtime.presence_multiplier(family_ids, 5.0)[3]
        )
        plan = plan_fission(reg, 1, 0, 4.0, 6.5, round_index=0,
                            iteration=3000, config=CFG, dtype=torch.float64)
        # Purity: building the plan changed nothing.
        self.assertEqual(
            float(runtime.presence_multiplier(family_ids, 5.0)[3]),
            before_presence,
        )
        self.assertEqual(reg.get(1).interval.K, 2)
        directives = apply_plan(bundle, plan)
        self.assertEqual(directives.moment_reset_family_ids, (1,))
        runtime.sync_family_from_registry(1)
        # t = 5 now falls inside the new fission gap (4, 6.5).
        self.assertEqual(float(runtime.presence_multiplier(family_ids, 5.0)[3]), 0.0)
        # Moment reset is ledgered.
        kinds = [e.kind for e in bundle.ledger.events]
        self.assertIn("moment_reset", kinds)

    def test_schedule_phases(self):
        _, runtime, _ = _setup()
        self.assertEqual(runtime.phase_at(1000), "warmup")
        self.assertEqual(runtime.phase_at(2600), "seeded")
        self.assertEqual(runtime.phase_at(4500), "rounds")
        self.assertTrue(runtime.is_round_boundary(4500))
        self.assertFalse(runtime.is_round_boundary(4501))
        self.assertEqual(runtime.phase_at(8000), "refit")
        self.assertEqual(runtime.phase_at(10001), "post_refit")
        with self.assertRaises(ContractError):
            ScheduleAnchors(3000, 2800, (3000,), 10000)


class CheckpointIntegrationTests(unittest.TestCase):
    def test_flush_capture_reload_reproduces_presence(self):
        reg, runtime, family_ids = _setup()
        # Perturb the live logits as an optimizer step would.
        with torch.no_grad():
            runtime.logit_parameters()[1].add_(
                torch.tensor([0.3, -0.2, 0.1, 0.05, -0.1], dtype=torch.float64)
            )
        reference = runtime.presence_multiplier(family_ids, 10.6)
        runtime.flush_to_registry()
        state = build_elgs_state(
            reg, BindingTable(), TransactionLedger(),
            SearchCostLedger(row_cap=10**6, scalar_budget=10**9),
            sampler={"lambda_u": 0.5, "pi_d_identity": "pi", "frozen": True},
            slot_grid={}, confirmation_refs={}, moment_reset_log=[],
            round_bookkeeping={"round_index": 0},
        )
        loaded = load_elgs_state(state)
        runtime2 = ElgsRuntime(loaded["registry"], CFG, SCHEDULE,
                               dtype=torch.float64)
        loaded["registry"].on_rows_added(0, 3)  # row counts restored via state
        restored = runtime2.presence_multiplier(family_ids, 10.6)
        self.assertTrue(torch.equal(reference.detach(), restored.detach()))


if __name__ == "__main__":
    unittest.main()
