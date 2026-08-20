"""Pointer-column appearance edit (CCR v3): composition, sidecar,
one-hop invariant, and the deployed-state-equals-tested-state property.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from depth_visibility.errors import ContractError
from scene.appearance_edit import (
    apply_appearance_edit,
    build_edit_payload,
    clear_appearance_edit,
    compose_shared_features,
    load_edit_payload,
    non_pointer_state_hash,
)


def _features(n=6, seed=3):
    gen = torch.Generator().manual_seed(seed)
    dc = torch.rand(n, 1, 3, generator=gen)
    rest = torch.rand(n, 15, 3, generator=gen)
    return dc, rest


class ComposeTests(unittest.TestCase):
    def test_identity_pointer_is_a_no_op(self):
        dc, rest = _features()
        idx = torch.arange(6)
        out_dc, out_rest = compose_shared_features(dc, rest, idx, "dc")
        self.assertTrue(torch.equal(out_dc, dc))
        self.assertTrue(torch.equal(out_rest, rest))

    def test_dc_mode_redirects_dc_only(self):
        dc, rest = _features()
        idx = torch.arange(6)
        idx[4] = 1  # row 4 reuses row 1's base radiance
        out_dc, out_rest = compose_shared_features(dc, rest, idx, "dc")
        self.assertTrue(torch.equal(out_dc[4], dc[1]))
        self.assertTrue(torch.equal(out_rest[4], rest[4]))  # own SH kept
        self.assertTrue(torch.equal(out_dc[:4], dc[:4]))

    def test_full_mode_redirects_both(self):
        dc, rest = _features()
        idx = torch.arange(6)
        idx[4] = 1
        out_dc, out_rest = compose_shared_features(dc, rest, idx, "full")
        self.assertTrue(torch.equal(out_dc[4], dc[1]))
        self.assertTrue(torch.equal(out_rest[4], rest[1]))

    def test_one_hop_invariant_fails_closed(self):
        dc, rest = _features()
        idx = torch.arange(6)
        idx[4] = 1
        idx[1] = 0  # row 1 is a donor AND a recipient -> chain
        with self.assertRaises(ContractError):
            compose_shared_features(dc, rest, idx, "dc")

    def test_bad_mode_and_bad_shape_fail_closed(self):
        dc, rest = _features()
        with self.assertRaises(ContractError):
            compose_shared_features(dc, rest, torch.arange(6), "chroma")
        with self.assertRaises(ContractError):
            compose_shared_features(dc, rest, torch.arange(5), "dc")


class SidecarTests(unittest.TestCase):
    def test_roundtrip_and_apply(self):
        pointer = torch.arange(6)
        pointer[5] = 2
        payload = build_edit_payload(
            pointer, "dc", source_checkpoint="chkpnt_best.pth",
            funnel={"evaluated": 3, "admitted": 1},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edit.pt"
            torch.save(payload, path)
            loaded = load_edit_payload(path)
        self.assertEqual(loaded["num_redirected"], 1)
        self.assertEqual(loaded["funnel"]["admitted"], 1)

        import types

        g = types.SimpleNamespace(
            _features_dc=torch.zeros(6, 1, 3),
        )
        apply_appearance_edit(g, loaded)
        self.assertTrue(torch.equal(g._appearance_source_idx, pointer))
        self.assertEqual(g._appearance_share_mode, "dc")

    def test_apply_refuses_row_count_mismatch(self):
        import types

        payload = build_edit_payload(
            torch.arange(6), "dc", source_checkpoint="x"
        )
        g = types.SimpleNamespace(_features_dc=torch.zeros(7, 1, 3))
        with self.assertRaises(ContractError):
            apply_appearance_edit(g, payload)


class InvariantHashTests(unittest.TestCase):
    def test_hash_ignores_pointer_and_mode_but_not_parameters(self):
        import types

        def model():
            gen = torch.Generator().manual_seed(9)
            return types.SimpleNamespace(
                _xyz=torch.rand(4, 3, generator=gen),
                _features_dc=torch.rand(4, 1, 3, generator=gen),
                _features_rest=torch.rand(4, 15, 3, generator=gen),
                _opacity=torch.rand(4, 1, generator=gen),
                _scaling=torch.rand(4, 3, generator=gen),
                _rotation=torch.rand(4, 4, generator=gen),
                _t=torch.rand(4, 1, generator=gen),
                _scaling_t=torch.rand(4, 1, generator=gen),
                _route_logit=torch.rand(4, 1, generator=gen),
                _motion_lora_coeff=torch.rand(4, 8, generator=gen),
            )

        a = model()
        b = model()
        b._appearance_source_idx = torch.tensor([0, 0, 2, 3])
        b._appearance_share_mode = "full"
        self.assertEqual(non_pointer_state_hash(a), non_pointer_state_hash(b))
        b._features_dc = b._features_dc + 1e-6
        self.assertNotEqual(
            non_pointer_state_hash(a), non_pointer_state_hash(b)
        )

    def test_clear_restores_identity_semantics(self):
        import types

        g = types.SimpleNamespace(_features_dc=torch.zeros(3, 1, 3))
        apply_appearance_edit(
            g, build_edit_payload(torch.tensor([0, 0, 2]), "dc",
                                  source_checkpoint="x"),
        )
        clear_appearance_edit(g)
        self.assertEqual(g._appearance_source_idx.numel(), 0)


if __name__ == "__main__":
    unittest.main()
