"""Keep the background out of the colour opacity gradient TWICE OVER.

`PerGaussianRenderCUDA` reconstructs the remainder of a pixel as
`-pixel_colors + sampled_ar`, and in this fork `pixel_colors` is a plain
copy of `out_color`, into which the forward has already folded
`T_final * bg`. That reconstruction is therefore the COMPLETE remainder,
background included, and the single `-(1 / (1 - alpha)) * (-ar[ch])` term
is the whole of the background's contribution to `d(out) / d(alpha)`.

The kernel used to add

    dL_dalpha += (-T_final / (1 - alpha)) * (bg . dL_dpixel)

on top of that, which is upstream taming-3dgs's line -- correct THERE,
because its forward writes a background-free `pixel_colors`, and a double
count here. It is invisible under a black background, which is what N3V
and DiVa-360 use and what every configuration in `configs/` sets, so the
defect had no numerical effect on any recorded result. It is worth
several tens of percent of the opacity gradient under any other.

These tests check the colour VJP against two independent references:

  * `tests/ref_impls/colour_compositing_reference.py`, a pure-PyTorch
    implementation written from the compositing equation, not from the
    CUDA; and
  * central finite differences of the CUDA forward itself.

`SensitivityTests` is what stops the rest from being vacuous: it models
the deleted line and shows that reinstating it would move these numbers
far past the tolerances used below. Without it, a suite that passes
proves nothing about a background it never made matter.

Measured on a V100 by recompiling the kernel four ways in one job and
importing each build ahead of the image's, on one three-Gaussian scene:

    variant                      dL_dopacity   white-bg opacity vs
                                     (absmax)  finite differences
    background double counted, no guard 48.884  169% / 167% / 234% off
    background counted once,   no guard 17.900  4.9% / 3.6% / within tol

Both rows are with the `max_contrib` guard REMOVED. With it present both
variants gave exactly 0.0 and were indistinguishable, which is the point
of the paragraph below.

The black-background column is bit-for-bit identical between the two,
which is the reason no recorded ADAGS result is affected: every
configuration under `configs/` sets `white_background: False`.

These tests also depend on the removal of the `max_contrib` early return
in the same kernel. While that guard was present it compared against
uninitialised device memory, and when that memory was zero the backward
returned before doing anything -- every gradient exactly 0.0, for every
loss and every background, so nothing here could distinguish any kernel
from any other.

EVERYTHING HERE NEEDS A GPU, for the reasons set out at the top of
`tests/test_flow_backward_vjp.py`, whose `TinyScene` harness and skip
guards this module reuses.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_flow_backward_vjp import (  # noqa: E402
    GPU_REASON,
    HAVE_RASTERIZER,
    HAVE_TORCH,
    TinyScene,
)

if HAVE_TORCH:
    import torch

    from tests.ref_impls.colour_compositing_reference import (
        composite_colour,
        colour_vjp,
    )

    # Generic chain rule from per-pixel alpha down to per-primitive
    # opacity; it knows nothing about flow, so it is reused as-is.
    from tests.ref_impls.flow_compositing_reference import dL_dopacity_from_dL_dalpha

WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)

#: Deliberately moderate. High opacities collapse `T_final` toward zero,
#: which is exactly the regime in which a background defect hides: the
#: term it corrupts is proportional to `T_final`.
OPACITIES = [0.35, 0.45, 0.4]
COLOURS = [(0.7, 0.2, 0.4), (0.3, 0.6, 0.5), (0.5, 0.5, 0.8)]
MEANS = [(-0.15, 0.05, -0.5), (0.05, -0.1, 0.0), (0.2, 0.15, 0.5)]


def _max_rel_error(actual, expected, floor=1.0):
    scale = max(float(expected.abs().max()), floor)
    return float((actual - expected).abs().max()) / scale


def _scene(background):
    return TinyScene(
        means3D=MEANS,
        opacities=OPACITIES,
        flows=[(0.0, 0.0)] * len(MEANS),
        colors=COLOURS,
        background=background,
    )


def _colours_tensor(scene):
    return torch.tensor(COLOURS, dtype=torch.float32, device=scene.device)


def _upstream(scene, seed):
    """Strictly positive, so `bg . dL_dpixel` cannot cancel to nothing.

    A zero-mean upstream would let the defect average away across
    channels and make the whole comparison meaningless.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand((3, scene.height, scene.width), generator=generator, device="cuda")


def _backward(scene, upstream):
    means3D = scene.means3D.clone().requires_grad_(True)
    opacities = scene.opacities.clone().requires_grad_(True)
    means2D = torch.zeros_like(means3D, requires_grad=True)
    out = scene.render(means3D=means3D, opacities=opacities, means2D=means2D)
    (out["render"] * upstream).sum().backward()
    return {
        "opacities": opacities.grad,
        "means3D": means3D.grad,
        "means2D": means2D.grad,
        "render": out["render"].detach(),
    }


def _reference(scene, upstream, background):
    """Correct per-primitive opacity and colour gradients for this scene."""
    alphas = scene.per_gaussian_alphas()
    colours = _colours_tensor(scene)
    dL_dcolours, dL_dalphas = colour_vjp(
        alphas, colours, background, upstream, order=scene.depth_order()
    )
    dL_dopacities = dL_dopacity_from_dL_dalpha(
        dL_dalphas, alphas, scene.unified_opacities()
    )
    return dL_dcolours, dL_dalphas, dL_dopacities, alphas


def _double_counted_dL_dalphas(dL_dalphas, alphas, background, upstream, scene):
    """Model of the deleted line, used ONLY to size the defect.

    Reproduces `dL_dalpha += (-T_final / (1 - alpha)) * (bg . dL_dpixel)`
    added on top of an already-complete reconstruction. The kernel applies
    it only where a primitive actually contributes; here it is applied
    everywhere, which is harmless because the caller chains through
    `dL_dopacity_from_dL_dalpha`, and that weights every pixel by
    `alphas[i]` -- zero wherever the primitive was culled.
    """
    _, transmittance = composite_colour(
        alphas, _colours_tensor(scene), background, order=scene.depth_order()
    )
    background = torch.as_tensor(background, dtype=alphas.dtype, device=alphas.device)
    bg_dot_dpixel = sum(
        background[channel] * upstream[channel] for channel in range(upstream.shape[0])
    )
    corrupted = dL_dalphas.clone()
    for index in range(alphas.shape[0]):
        corrupted[index] = corrupted[index] + (
            -transmittance / (1.0 - alphas[index])
        ) * bg_dot_dpixel
    return corrupted


def _central_difference(scene, upstream, field, index, channel, step):
    base = getattr(scene, field)
    plus = base.clone()
    minus = base.clone()
    plus[index, channel] += step
    minus[index, channel] -= step
    with torch.no_grad():
        loss_plus = float((scene.render(**{field: plus})["render"] * upstream).sum())
        loss_minus = float((scene.render(**{field: minus})["render"] * upstream).sum())
    return (loss_plus - loss_minus) / (2.0 * step)


@unittest.skipUnless(HAVE_RASTERIZER, GPU_REASON)
class SceneValidityTests(unittest.TestCase):
    """Guard the harness before trusting anything it reports."""

    def test_transmittance_survives_the_whole_stack(self):
        """A closed pixel shows no background, so it cannot test one.

        Also confirms the oracle's no-early-termination assumption: the
        kernel abandons a pixel once `T < 1e-4` and the reference does
        not model that.
        """
        scene = _scene(WHITE)
        alphas = scene.per_gaussian_alphas()
        _, transmittance = composite_colour(
            alphas, _colours_tensor(scene), WHITE, order=scene.depth_order()
        )
        self.assertGreater(float(transmittance.min()), 1e-4)
        covered = transmittance[alphas.max(dim=0).values > 1.0 / 255.0]
        self.assertGreater(
            float(covered.max()),
            0.1,
            "every covered pixel is nearly opaque, so T_final * bg is "
            "negligible there and a background defect would not show",
        )

    def test_no_alpha_reaches_the_renderer_ceiling(self):
        """`dL_dopacity_from_dL_dalpha` divides G out; a clamp breaks it."""
        scene = _scene(WHITE)
        self.assertLess(float(scene.per_gaussian_alphas().max()), 0.98)

    def test_the_background_actually_reaches_the_image(self):
        scene_white = _scene(WHITE)
        scene_black = _scene(BLACK)
        with torch.no_grad():
            white = scene_white.render()["render"]
            black = scene_black.render()["render"]
        self.assertGreater(float((white - black).abs().max()), 0.5)


@unittest.skipUnless(HAVE_RASTERIZER, GPU_REASON)
class SensitivityTests(unittest.TestCase):
    """Would reinstating the deleted line fail the tests below?"""

    def test_double_counting_the_background_moves_the_gradient_a_lot(self):
        scene = _scene(WHITE)
        upstream = _upstream(scene, seed=41)
        _, dL_dalphas, correct, alphas = _reference(scene, upstream, WHITE)
        corrupted = dL_dopacity_from_dL_dalpha(
            _double_counted_dL_dalphas(dL_dalphas, alphas, WHITE, upstream, scene),
            alphas,
            scene.unified_opacities(),
        )
        relative = _max_rel_error(corrupted, correct)
        self.assertGreater(
            relative,
            5e-2,
            "the double count would be smaller than the tolerances used by "
            "the tests below, so those tests could not detect it and this "
            "scene needs redesigning",
        )

    def test_double_counting_is_exactly_zero_on_a_black_background(self):
        """Why no recorded ADAGS result is affected.

        Every configuration in `configs/` sets `white_background: False`,
        and `bg . dL_dpixel` is identically zero for `bg == 0`.
        """
        scene = _scene(BLACK)
        upstream = _upstream(scene, seed=43)
        _, dL_dalphas, correct, alphas = _reference(scene, upstream, BLACK)
        corrupted = dL_dopacity_from_dL_dalpha(
            _double_counted_dL_dalphas(dL_dalphas, alphas, BLACK, upstream, scene),
            alphas,
            scene.unified_opacities(),
        )
        self.assertEqual(float((corrupted - correct).abs().max()), 0.0)


@unittest.skipUnless(HAVE_RASTERIZER, GPU_REASON)
class ColourVjpAgainstTheReferenceTests(unittest.TestCase):
    def test_forward_matches_the_reference_on_a_white_background(self):
        scene = _scene(WHITE)
        alphas = scene.per_gaussian_alphas()
        expected, _ = composite_colour(
            alphas, _colours_tensor(scene), WHITE, order=scene.depth_order()
        )
        with torch.no_grad():
            rendered = scene.render()["render"]
        self.assertLess(_max_rel_error(rendered, expected), 2e-3)

    def test_opacity_gradient_matches_the_reference_on_a_white_background(self):
        scene = _scene(WHITE)
        upstream = _upstream(scene, seed=47)
        _, _, expected, _ = _reference(scene, upstream, WHITE)
        actual = _backward(scene, upstream)["opacities"].reshape(-1)
        self.assertLess(_max_rel_error(actual, expected), 5e-3)

    def test_opacity_gradient_matches_the_reference_on_a_black_background(self):
        """The control. Measured to pass with the double count in place too,
        which is what makes the white-background failure attributable to the
        background rather than to anything else in the colour path."""
        scene = _scene(BLACK)
        upstream = _upstream(scene, seed=53)
        _, _, expected, _ = _reference(scene, upstream, BLACK)
        actual = _backward(scene, upstream)["opacities"].reshape(-1)
        self.assertLess(_max_rel_error(actual, expected), 5e-3)

    def test_colour_gradient_matches_the_reference_on_a_white_background(self):
        """`dL_dcolors` is `weight * upstream` and never saw the defect.

        Pinned anyway: a repair that fixed the alpha path by disturbing
        the colour path would not be a repair.
        """
        scene = _scene(WHITE)
        upstream = _upstream(scene, seed=59)
        expected, _, _, _ = _reference(scene, upstream, WHITE)

        shs = scene.shs.clone().requires_grad_(True)
        original_shs = scene.shs
        try:
            scene.shs = shs
            out = scene.render()
            (out["render"] * upstream).sum().backward()
        finally:
            scene.shs = original_shs
        self.assertIsNotNone(shs.grad)
        # SH degree 0 is a constant scale on the decoded colour, so the
        # colour gradient is recoverable without re-deriving the basis.
        sh_c0 = 0.28209479177387814
        actual = shs.grad.reshape(scene.count, 3) * (1.0 / sh_c0)
        self.assertLess(_max_rel_error(actual, expected), 5e-3)


def _finite_difference_error(background, field, axes, step, seed):
    """Vector relative error between the analytic gradient and its secant.

    Normalised by the largest finite difference in the set rather than
    per component, so a component that happens to sit near zero cannot
    dominate the ratio.
    """
    scene = _scene(background)
    upstream = _upstream(scene, seed)
    analytic = _backward(scene, upstream)[field]
    numeric, exact = [], []
    for index in range(scene.count):
        for axis in axes:
            numeric.append(_central_difference(scene, upstream, field, index, axis, step))
            exact.append(float(analytic[index, axis]))
    numeric = torch.tensor(numeric)
    exact = torch.tensor(exact)
    scale = float(numeric.abs().max())
    return {
        "relative_error": float((numeric - exact).abs().max()) / max(scale, 1e-6),
        "largest_finite_difference": scale,
        "numeric": numeric.tolist(),
        "analytic": exact.tolist(),
    }


@unittest.skipUnless(HAVE_RASTERIZER, GPU_REASON)
class ColourVjpAgainstFiniteDifferencesTests(unittest.TestCase):
    """Oracle-independent: perturb the forward and watch the loss move.

    Absolute agreement is deliberately NOT the pin. Central differences
    of this renderer carry a few percent of systematic error that has
    nothing to do with the background: perturbing an opacity slides alpha
    across the `alpha < 1/255` contribution floor at the edge of the
    footprint, so the forward is not smooth in the perturbed parameter
    and the secant is biased. Measured on a V100 against a BLACK
    background, where the background term is provably a no-op, that bias
    is 2.2%, 5.4% and 5.0% on the three primitives.

    A 2% absolute pin would therefore fail on a correct kernel, and
    loosening it to 10% until it went green would be tuning the
    instrument to the answer. So the pin is COMPARATIVE: the same scene
    is finite-differenced on black and on white, and white must agree no
    worse than black does. That isolates exactly the disputed quantity --
    the background's share of d(out)/d(alpha) -- and calibrates itself
    against whatever the renderer's inherent secant bias happens to be.

    The margin is not marginal. With the background double counted, the
    same white-background measurement is off by 169%, 167% and 234%
    while black is bit-for-bit unchanged.
    """

    OPACITY_STEP = 5e-3
    GEOMETRY_STEP = 2e-3

    #: How much worse white may agree than black before this is a defect.
    TOLERANCE_FACTOR = 2.0
    #: Absolute floor, so a scene whose black error is ~0 stays testable.
    TOLERANCE_FLOOR = 0.02
    #: If black itself is this bad the scene is not a usable instrument.
    BLACK_SANITY = 0.15

    def _compare(self, field, axes, step, black_sanity, seed):
        # The SAME seed for both, deliberately. `_upstream` depends only on
        # the seed and the frame size, which the two scenes share, so black
        # and white are finite-differenced against an IDENTICAL upstream
        # gradient and therefore against the same scalar loss. That is the
        # whole basis of the comparison below: the renderer's secant bias is
        # only common-mode, and therefore only cancels, if the two
        # measurements differ in the background and in nothing else.
        black = _finite_difference_error(BLACK, field, axes, step, seed)
        white = _finite_difference_error(WHITE, field, axes, step, seed)

        self.assertGreater(
            black["largest_finite_difference"],
            1.0,
            f"finite differences of {field} are at the float32 noise floor "
            f"({black}), so this comparison would pass for a wrong gradient",
        )
        self.assertLess(
            black["relative_error"],
            black_sanity,
            f"the BLACK-background baseline is itself unreliable ({black}); "
            "the white-vs-black comparison means nothing until it is fixed",
        )
        self.assertLessEqual(
            white["relative_error"],
            max(self.TOLERANCE_FACTOR * black["relative_error"], self.TOLERANCE_FLOOR),
            f"the white background degrades {field} agreement well beyond the "
            f"renderer's background-independent secant bias.\n"
            f"  black: {black}\n  white: {white}",
        )

    def test_opacity_agreement_is_no_worse_on_white_than_on_black(self):
        self._compare("opacities", [0], self.OPACITY_STEP, self.BLACK_SANITY, 67)

    def test_the_geometry_path_is_exercised_under_a_white_background(self):
        """The defect reached more than opacity, but geometry cannot be
        finite-differenced here, and this test does not pretend otherwise.

        `dL_dG = con_o.w * dL_dalpha` feeds the 2D mean and the conic, so
        the corrupted alpha gradient corrupted the geometry gradients and
        the densification signal built on `means2D.grad` with them --
        measured on a V100 as `dL_dmeans3D` 3.668 correct against 10.017
        double-counted.

        There is no finite-difference pin on that, because the instrument
        will not carry one: on a BLACK background, where both the
        background term and the `max_contrib` guard are provably inert,
        the projected-geometry secant still disagrees with the analytic
        gradient by 46% (4.167 against 2.238 on one component). Moving a
        mean re-quantises the radius and the tile touch list, so the
        forward is badly non-smooth in it. The pre-existing
        `test_projected_geometry_gradient_matches_finite_differences` in
        the flow suite fails the same way, including one sign flip.

        Nothing is lost by not pinning it. Opacity and geometry are
        formed from the SAME per-pixel `dL_dalpha` accumulator, which
        `ColourVjpAgainstTheReferenceTests` pins against the oracle at
        5e-3; everything downstream of it is shared machinery this change
        does not touch. So this test claims only what it can support:
        that a white-background colour loss reaches the geometry path at
        all, with finite values.
        """
        scene = _scene(WHITE)
        grads = _backward(scene, _upstream(scene, seed=71))
        for name in ("means3D", "means2D"):
            with self.subTest(parameter=name):
                self.assertIsNotNone(grads[name])
                self.assertTrue(bool(torch.isfinite(grads[name]).all()))
                self.assertGreater(float(grads[name][:, :2].abs().max()), 0.0)


class EnvironmentDisclosureTests(unittest.TestCase):
    def test_the_gpu_pins_actually_ran_somewhere(self):
        """A skipped background pin is not a passing one."""
        if not HAVE_TORCH:
            self.skipTest("torch is not importable in this environment")
        if not HAVE_RASTERIZER:
            self.skipTest(
                "no CUDA device or no compiled rasterizer: the colour "
                "background pins did NOT run here and must be run on a GPU "
                f"node ({GPU_REASON})"
            )
        self.assertTrue(HAVE_RASTERIZER)


if __name__ == "__main__":
    unittest.main()
