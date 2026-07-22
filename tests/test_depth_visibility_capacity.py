import unittest

import torch
from torch import nn

from depth_visibility.capacity import (
    CapacityBank,
    apply_point_neutral_transaction,
    build_event_blind_capacity_targets,
    capacity_budget,
    select_event_blind_donors,
)
from depth_visibility.errors import ContractError


def _make_bank(n=8):
    torch.manual_seed(3)
    specs = {
        "_xyz": (n, 3),
        "_features_dc": (n, 1, 3),
        "_features_rest": (n, 2, 3),
        "_scaling": (n, 3),
        "_rotation": (n, 4),
        "_opacity": (n, 1),
        "_t": (n, 1),
        "_scaling_t": (n, 1),
        "_route_logit": (n, 1),
        "_motion_v": (n, 3),
        "_motion_a": (n, 3),
        "_motion_lora_coeff": (n, 4),
        "_staticness_score": (n, 1),
    }
    parameters = {
        name: nn.Parameter(torch.randn(*shape, dtype=torch.float64))
        for name, shape in specs.items()
    }
    accumulators = {
        "xyz_gradient_accum": torch.ones(n, 1, dtype=torch.float64),
        "t_gradient_accum": torch.ones(n, 1, dtype=torch.float64) * 2,
        "denom": torch.ones(n, 1, dtype=torch.float64) * 3,
        "max_radii2D": torch.ones(n, dtype=torch.float64) * 4,
    }
    return CapacityBank(
        parameters=parameters,
        accumulators=accumulators,
        stable_ids=torch.arange(100, 100 + n, dtype=torch.long),
        generation=torch.zeros(n, dtype=torch.long),
        last_reassigned=torch.zeros(n, dtype=torch.long),
        hard_static_count=2,
    )


def _make_optimizer(bank):
    optimizer = torch.optim.Adam(list(bank.parameters.values()), lr=0.01, amsgrad=True)
    loss = sum(parameter.square().sum() for parameter in bank.parameters.values())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


class CapacityTransactionTests(unittest.TestCase):
    def test_budget_counts_dynamic_and_hard_static(self):
        bank = _make_bank()
        self.assertEqual(capacity_budget(bank), {"dynamic": 8, "hard_static": 2, "total": 10})

    def test_reassignment_preserves_parameter_identity_budget_and_survivors(self):
        bank = _make_bank()
        optimizer = _make_optimizer(bank)
        donors = torch.tensor([1, 6])
        identities = {name: id(parameter) for name, parameter in bank.parameters.items()}
        survivor_before = {name: parameter.detach()[0].clone() for name, parameter in bank.parameters.items()}
        target_rows = {
            name: torch.full((len(donors), *parameter.shape[1:]), 7.0, dtype=parameter.dtype)
            for name, parameter in bank.parameters.items()
        }
        before_moments = {
            name: optimizer.state[parameter]["exp_avg"].detach()[0].clone()
            for name, parameter in bank.parameters.items()
        }
        result = apply_point_neutral_transaction(
            bank, optimizer, donors, target_rows, iteration=5001
        )
        self.assertEqual(result["realized_k"], 2)
        self.assertEqual(result["budget_before"], result["budget_after"])
        self.assertEqual(capacity_budget(bank), {"dynamic": 8, "hard_static": 2, "total": 10})
        for name, parameter in bank.parameters.items():
            self.assertEqual(id(parameter), identities[name])
            self.assertTrue(torch.equal(parameter.detach()[0], survivor_before[name]))
            self.assertTrue(torch.all(parameter.detach()[donors] == 7.0))
            self.assertTrue(torch.equal(optimizer.state[parameter]["exp_avg"][0], before_moments[name]))
            self.assertTrue(torch.all(optimizer.state[parameter]["exp_avg"][donors] == 0.0))
            self.assertTrue(torch.all(optimizer.state[parameter]["exp_avg_sq"][donors] == 0.0))
            self.assertTrue(torch.all(optimizer.state[parameter]["max_exp_avg_sq"][donors] == 0.0))
        self.assertTrue(torch.all(bank.accumulators["denom"][donors] == 0.0))
        self.assertTrue(torch.all(bank.generation[donors] == 1))
        self.assertTrue(torch.all(bank.last_reassigned[donors] == 5001))

    def test_null_reset_leaves_values_but_resets_moments(self):
        bank = _make_bank()
        optimizer = _make_optimizer(bank)
        donors = torch.tensor([2])
        before = {name: parameter.detach().clone() for name, parameter in bank.parameters.items()}
        result = apply_point_neutral_transaction(
            bank, optimizer, donors, None, iteration=5001, mode="null-reset"
        )
        self.assertEqual(result["mode"], "null-reset")
        for name, parameter in bank.parameters.items():
            self.assertTrue(torch.equal(parameter.detach(), before[name]))
            self.assertTrue(torch.all(optimizer.state[parameter]["exp_avg"][donors] == 0.0))

    def test_invalid_target_shape_fails_closed(self):
        bank = _make_bank()
        optimizer = _make_optimizer(bank)
        target_rows = {name: parameter.detach()[:1].clone() for name, parameter in bank.parameters.items()}
        target_rows["_xyz"] = torch.ones(2, 3, dtype=torch.float64)
        with self.assertRaises(ContractError):
            apply_point_neutral_transaction(
                bank, optimizer, torch.tensor([0]), target_rows, iteration=5001
            )

    def test_event_blind_capacity_targets_clone_non_donor_rows_deterministically(self):
        bank = _make_bank()
        donors = torch.tensor([1, 6])
        targets, metadata = build_event_blind_capacity_targets(
            bank, donors, seed=11, iteration=5001
        )
        self.assertEqual(metadata["target_policy"], "event_blind_existing_dynamic_row_clone_v1")
        self.assertEqual(len(metadata["target_source_indices"]), 2)
        self.assertTrue(set(metadata["target_source_indices"]).isdisjoint({1, 6}))
        again, again_meta = build_event_blind_capacity_targets(
            bank, donors, seed=11, iteration=5001
        )
        self.assertEqual(metadata["target_source_indices"], again_meta["target_source_indices"])
        for name in bank.parameters:
            self.assertTrue(torch.equal(targets[name], again[name]))
            expected = bank.parameters[name].detach()[metadata["target_source_indices"]]
            self.assertTrue(torch.equal(targets[name], expected))

    def test_capacity_only_reassignment_preserves_budget_with_generic_targets(self):
        bank = _make_bank()
        optimizer = _make_optimizer(bank)
        donors = torch.tensor([0])
        targets, _ = build_event_blind_capacity_targets(
            bank, donors, seed=3, iteration=5001
        )
        result = apply_point_neutral_transaction(
            bank, optimizer, donors, targets, iteration=5001, mode="reassign"
        )
        self.assertEqual(result["budget_before"], result["budget_after"])
        self.assertEqual(result["realized_k"], 1)
        self.assertTrue(torch.all(optimizer.state[bank.parameters["_xyz"]]["exp_avg"][donors] == 0.0))

    def test_select_event_blind_donor_uses_low_opacity_redundant_old_slot(self):
        n = 8
        xyz = torch.tensor(
            [
                [0.00, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [1.00, 0.0, 0.0],
                [1.05, 0.0, 0.0],
                [2.00, 0.0, 0.0],
                [2.05, 0.0, 0.0],
                [3.00, 0.0, 0.0],
                [3.05, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        scaling_log = torch.log(torch.full((n, 3), 0.10, dtype=torch.float64))
        opacity_logit = torch.tensor([-4.0, 2.0, -1.0, 2.0, -0.5, 2.0, 0.0, 2.0], dtype=torch.float64)
        denom = torch.arange(1, n + 1, dtype=torch.float64).reshape(n, 1)
        generation = torch.zeros(n, dtype=torch.long)
        stable_ids = torch.arange(10, 10 + n, dtype=torch.long)
        selected = select_event_blind_donors(
            xyz=xyz,
            scaling_log=scaling_log,
            opacity_logit=opacity_logit,
            denom=denom,
            generation=generation,
            stable_ids=stable_ids,
            current_iteration=5001,
            k=1,
        )
        self.assertFalse(selected["abstained"])
        self.assertEqual(selected["selected_indices"], [0])


if __name__ == "__main__":
    unittest.main()
