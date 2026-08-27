"""Phase-alignment ablation: which time the SHARED LoRA basis is indexed at.

`motion_lora_time_reference` selects the one variable under test:

  "primitive"      basis read at t - t_i (default; every historical run)
  "global_matched" basis read at absolute time over the SAME half-table
                   window a mid-sequence primitive sees -- isolates phase
  "global"         basis read at absolute time over the WHOLE table (DynMF)

The load-bearing test here is the ANTI-VACUITY one: if every primitive shares
a temporal centre the arms must COINCIDE, because then there is no phase to
align and the ablation would be measuring nothing. A run whose t_i have
collapsed cannot answer the question, so `temporal_centre_dispersion` exists
to be asserted on before any score is read.
"""

import unittest

import torch

from scene.gaussian_model import GaussianModel


def _model(time_reference, times, rank=4, anchors=32, duration=(0.0, 10.0)):
    g = GaussianModel(0, gaussian_dim=4, time_duration=list(duration))
    n = times.shape[0]
    g.motion_model = "lora"
    g.motion_lora_rank = rank
    g.motion_lora_anchors = anchors
    g.motion_lora_time_reference = time_reference
    g._xyz = torch.zeros(n, 3)
    g._t = times.view(n, 1)
    g._motion_lora_coeff = torch.arange(n * rank, dtype=torch.float32).reshape(n, rank) * 0.01
    gen = torch.Generator().manual_seed(0)
    g._motion_lora_basis = torch.rand(rank, anchors, 3, generator=gen)
    return g


def _dispersed(n=64, duration=(0.0, 10.0)):
    return torch.linspace(duration[0], duration[1], n)


class LoraTimeReferenceTests(unittest.TestCase):
    def test_arms_differ_when_temporal_centres_are_dispersed(self):
        times = _dispersed()
        prim = _model("primitive", times).get_lora_motion_offset(3.0)
        glob = _model("global", times).get_lora_motion_offset(3.0)
        matched = _model("global_matched", times).get_lora_motion_offset(3.0)
        self.assertGreater((prim - glob).abs().max().item(), 1e-4)
        self.assertGreater((prim - matched).abs().max().item(), 1e-4)

    def test_anti_vacuity_arms_coincide_when_every_centre_is_identical(self):
        """The precondition the ablation rests on.

        With every t_i at the sequence midpoint there is no per-primitive
        phase, so "primitive" and "global_matched" must agree EXACTLY. If this
        ever fails the two arms differ by something other than phase and the
        ablation is confounded.
        """
        mid = 0.5 * (0.0 + 10.0)
        times = torch.full((32,), mid)
        prim = _model("primitive", times).get_lora_motion_offset(7.25)
        matched = _model("global_matched", times).get_lora_motion_offset(7.25)
        torch.testing.assert_close(prim, matched, rtol=0, atol=1e-6)

    def test_global_arm_alone_is_not_a_phase_control(self):
        """Anti-vacuity for the CONTROL: "global" still differs from
        "primitive" at identical centres, because it also doubles the
        addressed anchor window. That is why "global_matched" exists and why
        "global" must never be read as isolating phase."""
        times = torch.full((32,), 5.0)
        prim = _model("primitive", times).get_lora_motion_offset(7.25)
        glob = _model("global", times).get_lora_motion_offset(7.25)
        self.assertGreater((prim - glob).abs().max().item(), 1e-4)

    def test_primitive_arm_addresses_at_most_half_the_anchor_table(self):
        """The structural fact the matched control is built around."""
        anchors = 32
        duration = 10.0
        for t_i in (0.0, 2.5, 5.0, 7.5, 10.0):
            us = []
            for t in torch.linspace(0.0, duration, 101):
                us.append(float(((t - t_i) / duration + 1.0) * 0.5))
            span = (max(us) - min(us)) * (anchors - 1)
            self.assertLessEqual(span, (anchors - 1) / 2 + 1e-6)

    def test_default_is_primitive_so_historical_runs_are_unchanged(self):
        from arguments import OptimizationParams
        from argparse import ArgumentParser

        params = OptimizationParams(ArgumentParser())
        self.assertEqual(params.motion_lora_time_reference, "primitive")
        self.assertEqual(
            GaussianModel(0, gaussian_dim=4).motion_lora_time_reference, "primitive"
        )

    def test_time_reference_survives_capture_restore(self):
        g = _model("global_matched", _dispersed())
        params = g.capture()
        blob = params[-1] if isinstance(params[-1], dict) else None
        found = None
        for entry in params:
            if isinstance(entry, dict) and "motion_lora_time_reference" in entry:
                found = entry["motion_lora_time_reference"]
        self.assertEqual(found, "global_matched",
                         "capture() must carry the arm or a branch-from-checkpoint "
                         "silently reverts to the default arm")


def temporal_centre_dispersion(gaussians, duration):
    """Fraction of the sequence spanned by the middle 90% of t_i.

    The ablation's PRECONDITION: near zero means the centres have collapsed,
    the arms coincide by construction, and no score from that run may be read
    as evidence about phase alignment.
    """
    t = gaussians.get_t.detach().flatten()
    lo = torch.quantile(t, 0.05)
    hi = torch.quantile(t, 0.95)
    return float((hi - lo) / max(duration, 1e-6))


if __name__ == "__main__":
    unittest.main()
