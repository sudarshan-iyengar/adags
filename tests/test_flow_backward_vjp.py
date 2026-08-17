"""Pin the rendered-flow forward AND its backward pass in the rasterizer.

`render_pkg["flow"]` is a genuine alpha-composited channel, but until the
repair these tests accompany, its gradient was structurally zero: the only
flow-gradient code in `backward.cu` lived in an uninstantiated template,
so `dL_dflows` came back exactly as it was allocated. A flow supervision
term therefore trained nothing, while looking for all the world like a
supervision result that failed.

These tests check the repaired path against two independent references:

  * `tests/ref_impls/flow_compositing_reference.py`, a pure-PyTorch
    implementation written from the compositing equation, not from the
    CUDA; and
  * central finite differences of the CUDA forward itself.

The per-primitive alpha fields the oracle consumes are obtained by
rendering each Gaussian ALONE and reading the rasterizer's alpha channel.
That is deliberate: alpha is not the quantity under test, and taking it
from the renderer avoids re-deriving the EWA projection (whose own
correctness these tests make no claim about). It is valid only while no
pixel hits the `test_T < 1e-4` early-out, which the scenes below stay
well clear of.

EVERYTHING HERE NEEDS A GPU. The rasterizer has no CPU path, and
importing the wrapper on a machine without the prebuilt extension would
trigger a JIT compile, so the import is not even attempted unless CUDA is
present. A skip is not a pass: `EnvironmentDisclosureTests` records that.
"""

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import numpy as np
    import torch

    HAVE_TORCH = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_TORCH = False

HAVE_CUDA = bool(HAVE_TORCH and torch.cuda.is_available())

RASTERIZER_IMPORT_ERROR = ""
HAVE_RASTERIZER = False
if HAVE_CUDA:
    try:
        from gaussian_renderer.diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
        from utils.graphics_utils import getProjectionMatrix, getWorld2View2

        HAVE_RASTERIZER = True
    except Exception as exc:  # pragma: no cover - environment dependent
        RASTERIZER_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

HAVE_MODEL = False
MODEL_IMPORT_ERROR = ""
if HAVE_RASTERIZER:
    try:
        from argparse import ArgumentParser

        from arguments import OptimizationParams, PipelineParams
        from gaussian_renderer import render as render_gaussians
        from scene.cameras import MiniCam
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import BasicPointCloud

        HAVE_MODEL = True
    except Exception as exc:  # pragma: no cover - environment dependent
        MODEL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

if HAVE_TORCH:
    from tests.ref_impls.flow_compositing_reference import (
        composite_flow,
        dL_dopacity_from_dL_dalpha,
        flow_vjp,
    )

GPU_REASON = (
    "needs a CUDA device and the compiled diff-gaussian-rasterization extension"
    + (f" ({RASTERIZER_IMPORT_ERROR})" if RASTERIZER_IMPORT_ERROR else "")
)

SH_C0 = 0.28209479177387814
OFFSCREEN = (500.0, 500.0, 0.0)


def _sh_for_color(color):
    """DC-only SH coefficients that decode to `color` at degree 0."""
    return [(component - 0.5) / SH_C0 for component in color]


def _camera_matrices(distance=4.0, fov_degrees=60.0, device="cuda"):
    """World-to-view and full projection for a camera on the -z axis.

    Identity rotation, translation (0, 0, distance): the camera sits at
    (0, 0, -distance) and looks down +z, so view-space depth is world z
    plus `distance`. Both matrices are transposed exactly the way
    `scene/cameras.py` transposes them, because the rasterizer reads them
    in that layout.
    """
    fov = math.radians(fov_degrees)
    rotation = np.eye(3)
    translation = np.array([0.0, 0.0, float(distance)])
    world_view = torch.tensor(getWorld2View2(rotation, translation)).transpose(0, 1).to(device)
    projection = (
        getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov, fovY=fov)
        .transpose(0, 1)
        .to(device)
    )
    full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return world_view, full_proj, fov


class TinyScene:
    """A handful of Gaussians, one camera, everything explicit.

    Dynamic primitives carry a 2D flow vector; static ones (the second
    bank the rasterizer composites through the SAME transmittance) do
    not. Indices 0..N-1 are dynamic, N..N+M-1 are static, which is the
    ordering the oracle and the depth sort both use.
    """

    def __init__(
        self,
        means3D,
        opacities,
        flows,
        colors=None,
        scales=None,
        static_means3D=None,
        static_opacities=None,
        static_colors=None,
        static_scales=None,
        width=32,
        height=32,
        distance=4.0,
        fov_degrees=60.0,
        background=(0.0, 0.0, 0.0),
    ):
        device = "cuda"
        self.device = device
        self.width = int(width)
        self.height = int(height)

        self.means3D = self._tensor(means3D, (-1, 3))
        self.count = self.means3D.shape[0]
        self.opacities = self._tensor(opacities, (self.count, 1))
        self.flows = self._tensor(flows, (self.count, 2))
        self.shs = self._shs(colors, self.count)
        self.scales = self._scales(scales, self.count)
        self.rotations = self._rotations(self.count)

        # 4D fields are unused at gaussian_dim == 3, but are handed over as
        # real device tensors so that no kernel can ever dereference a host
        # pointer if a future branch starts reading them.
        self.ts = torch.zeros((self.count, 1), dtype=torch.float32, device=device)
        self.scales_t = torch.ones((self.count, 1), dtype=torch.float32, device=device)
        self.rotations_r = self._rotations(self.count)

        if static_means3D is None:
            self.static_count = 0
            # Mirrors `GaussianModel.get_static_xyz` on a model with no
            # static bank: a 1-D empty tensor, whose gradient shape the
            # extension matches exactly.
            self.static_means3D = torch.empty(0, dtype=torch.float32, device=device)
            self.static_opacities = torch.empty(0, dtype=torch.float32, device=device)
            self.static_shs = torch.zeros((0, 1, 3), dtype=torch.float32, device=device)
            self.static_scales = torch.empty(0, dtype=torch.float32, device=device)
            self.static_rotations = torch.empty(0, dtype=torch.float32, device=device)
        else:
            self.static_means3D = self._tensor(static_means3D, (-1, 3))
            self.static_count = self.static_means3D.shape[0]
            self.static_opacities = self._tensor(static_opacities, (self.static_count, 1))
            self.static_shs = self._shs(static_colors, self.static_count)
            self.static_scales = self._scales(static_scales, self.static_count)
            self.static_rotations = self._rotations(self.static_count)

        world_view, full_proj, fov = _camera_matrices(distance, fov_degrees, device)
        self.world_view = world_view
        self.settings = GaussianRasterizationSettings(
            image_height=self.height,
            image_width=self.width,
            tanfovx=math.tan(fov * 0.5),
            tanfovy=math.tan(fov * 0.5),
            bg=torch.tensor(background, dtype=torch.float32, device=device),
            scale_modifier=1.0,
            viewmatrix=world_view,
            projmatrix=full_proj,
            sh_degree=0,
            sh_degree_t=0,
            campos=world_view.inverse()[3, :3],
            timestamp=0.0,
            time_duration=1.0,
            rot_4d=False,
            gaussian_dim=3,
            force_sh_3d=False,
            prefiltered=False,
            opa_threshold=0.0,
            debug=False,
        )

    # ------------------------------------------------------------ builders

    def _tensor(self, values, shape):
        return torch.tensor(values, dtype=torch.float32, device=self.device).reshape(*shape)

    def _shs(self, colors, count):
        if colors is None:
            colors = [(0.6, 0.4, 0.3)] * count
        return torch.tensor(
            [[_sh_for_color(color)] for color in colors],
            dtype=torch.float32,
            device=self.device,
        ).reshape(count, 1, 3)

    def _scales(self, scales, count):
        if scales is None:
            scales = [(0.25, 0.25, 0.25)] * count
        return self._tensor(scales, (count, 3))

    def _rotations(self, count):
        rotations = torch.zeros((count, 4), dtype=torch.float32, device=self.device)
        if count:
            rotations[:, 0] = 1.0
        return rotations

    # --------------------------------------------------------------- utils

    def _depths_of(self, means3D):
        if means3D.numel() == 0:
            return torch.zeros((0,), device=self.device)
        homogeneous = torch.cat(
            [means3D, torch.ones((means3D.shape[0], 1), device=self.device)], dim=-1
        )
        return (homogeneous @ self.world_view)[:, 2]

    def view_depths(self):
        """Dynamic depths followed by static depths, the oracle's index space."""
        return torch.cat(
            [self._depths_of(self.means3D), self._depths_of(self.static_means3D.reshape(-1, 3))]
        )

    def depth_order(self):
        """Front-to-back index order, which is what the tiler sorts by."""
        return torch.argsort(self.view_depths()).tolist()

    def unified_flows(self):
        """(N + M, 2): the static bank carries no flow of its own."""
        zeros = torch.zeros((self.static_count, 2), dtype=torch.float32, device=self.device)
        return torch.cat([self.flows, zeros], dim=0)

    def unified_opacities(self):
        return torch.cat(
            [self.opacities.reshape(-1), self.static_opacities.reshape(-1)], dim=0
        )

    # ----------------------------------------------------------- rendering

    def render(
        self,
        means3D=None,
        opacities=None,
        flows=None,
        opacities_static=None,
        subset=None,
        subset_static=None,
        means2D=None,
        offscreen_dynamic=False,
    ):
        """Rasterize, optionally only a subset of the primitives.

        Overrides are accepted so a caller can hand in leaf tensors that
        require grad, or perturbed copies for finite differences.
        """
        means3D = self.means3D if means3D is None else means3D
        opacities = self.opacities if opacities is None else opacities
        flows = self.flows if flows is None else flows
        shs, scales, rotations = self.shs, self.scales, self.rotations
        ts, scales_t, rotations_r = self.ts, self.scales_t, self.rotations_r

        if subset is not None:
            index = torch.tensor(subset, dtype=torch.long, device=self.device)
            means3D = means3D[index]
            opacities = opacities[index]
            flows = flows[index]
            shs = shs[index]
            scales = scales[index]
            rotations = rotations[index]
            ts = ts[index]
            scales_t = scales_t[index]
            rotations_r = rotations_r[index]

        if offscreen_dynamic:
            # A placeholder dynamic primitive parked far outside the frustum:
            # the rasterizer needs P >= 1, but this one touches no tile.
            means3D = torch.tensor(
                [OFFSCREEN] * means3D.shape[0], dtype=torch.float32, device=self.device
            )

        static_means3D = self.static_means3D
        static_opacities = (
            self.static_opacities if opacities_static is None else opacities_static
        )
        static_shs = self.static_shs
        static_scales = self.static_scales
        static_rotations = self.static_rotations
        if subset_static is not None:
            index = torch.tensor(subset_static, dtype=torch.long, device=self.device)
            static_means3D = static_means3D.reshape(-1, 3)[index]
            static_opacities = static_opacities.reshape(-1, 1)[index]
            static_shs = static_shs[index]
            static_scales = static_scales.reshape(-1, 3)[index]
            static_rotations = static_rotations.reshape(-1, 4)[index]
            if index.numel() == 0:
                static_means3D = torch.empty(0, dtype=torch.float32, device=self.device)
                static_opacities = torch.empty(0, dtype=torch.float32, device=self.device)
                static_scales = torch.empty(0, dtype=torch.float32, device=self.device)
                static_rotations = torch.empty(0, dtype=torch.float32, device=self.device)

        if means2D is None:
            means2D = torch.zeros_like(means3D, requires_grad=True)

        rasterizer = GaussianRasterizer(raster_settings=self.settings)
        (
            color,
            radii,
            depth,
            alpha,
            flow,
            covs_com,
            radii_static,
            color_4d,
            color_3d,
            invdepth,
        ) = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            flow_2d=flows,
            opacities=opacities,
            ts=ts,
            scales=scales,
            scales_t=scales_t,
            rotations=rotations,
            rotations_r=rotations_r,
            cov3D_precomp=None,
            means3D_static=static_means3D,
            means2D_static=torch.zeros_like(static_means3D),
            shs_static=static_shs,
            opacities_static=static_opacities,
            scales_static=static_scales,
            rotations_static=static_rotations,
        )
        return {
            "render": color,
            "radii": radii,
            "depth": depth,
            "alpha": alpha,
            "flow": flow,
            "means2D": means2D,
        }

    def per_gaussian_alphas(self):
        """(N + M, H, W) alpha field of each primitive rendered on its own."""
        fields = []
        with torch.no_grad():
            for index in range(self.count):
                out = self.render(subset=[index], subset_static=[])
                fields.append(out["alpha"].reshape(self.height, self.width).clone())
            for index in range(self.static_count):
                out = self.render(
                    subset=[0], subset_static=[index], offscreen_dynamic=True
                )
                fields.append(out["alpha"].reshape(self.height, self.width).clone())
        return torch.stack(fields, dim=0)

    def reference_flow(self):
        alphas = self.per_gaussian_alphas()
        image, _ = composite_flow(alphas, self.unified_flows(), order=self.depth_order())
        return image, alphas


def _max_rel_error(actual, expected, floor=1.0):
    scale = max(float(expected.abs().max()), floor)
    return float((actual - expected).abs().max()) / scale


@unittest.skipUnless(HAVE_RASTERIZER, GPU_REASON)
class FlowForwardParityTests(unittest.TestCase):
    def test_single_gaussian_matches_the_reference(self):
        scene = TinyScene(
            means3D=[(0.0, 0.0, 0.0)],
            opacities=[0.8],
            flows=[(2.5, -1.5)],
        )
        rendered = scene.render()["flow"]
        expected, _ = scene.reference_flow()
        self.assertEqual(tuple(rendered.shape), (2, scene.height, scene.width))
        self.assertLess(_max_rel_error(rendered, expected), 1e-4)

    def test_overlapping_gaussians_at_different_depths(self):
        # Two Gaussians on the same ray, opposite flow directions. Getting
        # the occlusion order wrong flips the sign of the composited field.
        scene = TinyScene(
            means3D=[(0.0, 0.0, -0.5), (0.0, 0.0, 0.5)],
            opacities=[0.7, 0.7],
            flows=[(3.0, 0.0), (-3.0, 0.0)],
        )
        self.assertEqual(scene.depth_order(), [0, 1])
        rendered = scene.render()["flow"]
        expected, alphas = scene.reference_flow()
        self.assertLess(_max_rel_error(rendered, expected), 1e-4)

        # The near Gaussian must dominate the centre: it is unoccluded, the
        # far one is attenuated by (1 - alpha_near).
        centre = (scene.height // 2, scene.width // 2)
        self.assertGreater(float(rendered[0][centre]), 0.0)

        swapped, _ = composite_flow(alphas, scene.unified_flows(), order=[1, 0])
        self.assertGreater(
            float((swapped - expected).abs().max()),
            1e-3,
            "the scene is not order-sensitive, so it cannot test ordering",
        )

    def test_three_gaussians_matches_the_reference(self):
        scene = TinyScene(
            means3D=[(-0.2, 0.1, -0.6), (0.0, 0.0, 0.0), (0.15, -0.1, 0.6)],
            opacities=[0.55, 0.65, 0.75],
            flows=[(1.0, 2.0), (-2.0, 0.5), (0.25, -1.75)],
        )
        rendered = scene.render()["flow"]
        expected, _ = scene.reference_flow()
        self.assertLess(_max_rel_error(rendered, expected), 2e-4)


@unittest.skipUnless(HAVE_RASTERIZER, GPU_REASON)
class FlowDirectionTests(unittest.TestCase):
    def _single(self, dx, dy, opacity=0.8):
        scene = TinyScene(
            means3D=[(0.0, 0.0, 0.0)],
            opacities=[opacity],
            flows=[(dx, dy)],
        )
        out = scene.render()
        centre = (scene.height // 2, scene.width // 2)
        alpha = float(out["alpha"].reshape(scene.height, scene.width)[centre])
        return out["flow"], alpha, centre

    def test_positive_x_motion_renders_positive_x_flow(self):
        flow, alpha, centre = self._single(4.0, 0.0)
        self.assertGreater(alpha, 0.1)
        self.assertAlmostEqual(float(flow[0][centre]), 4.0 * alpha, delta=1e-3)
        self.assertAlmostEqual(float(flow[1][centre]), 0.0, delta=1e-5)
        self.assertGreater(float(flow[0].max()), 0.0)
        self.assertGreaterEqual(float(flow[0].min()), 0.0)

    def test_negative_x_motion_renders_negative_x_flow(self):
        flow, alpha, centre = self._single(-4.0, 0.0)
        self.assertAlmostEqual(float(flow[0][centre]), -4.0 * alpha, delta=1e-3)
        self.assertLess(float(flow[0].min()), 0.0)
        self.assertLessEqual(float(flow[0].max()), 0.0)

    def test_the_two_directions_are_exact_negations(self):
        plus, _, _ = self._single(4.0, 1.0)
        minus, _, _ = self._single(-4.0, -1.0)
        self.assertLess(float((plus + minus).abs().max()), 1e-5)

    def test_channels_are_independent(self):
        flow, alpha, centre = self._single(0.0, 3.0)
        self.assertAlmostEqual(float(flow[0].abs().max()), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(flow[1][centre]), 3.0 * alpha, delta=1e-3)


@unittest.skipUnless(HAVE_RASTERIZER, GPU_REASON)
class FlowBackwardVjpTests(unittest.TestCase):
    """The heart of it: does a flow loss produce the right gradients?"""

    def _scene(self):
        return TinyScene(
            means3D=[(-0.15, 0.05, -0.5), (0.05, -0.1, 0.0), (0.2, 0.15, 0.5)],
            opacities=[0.6, 0.7, 0.65],
            flows=[(2.0, -1.0), (-1.5, 2.5), (0.75, 0.5)],
        )

    def _upstream(self, scene, seed=7):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        return (
            torch.rand((2, scene.height, scene.width), generator=generator, device="cuda")
            - 0.5
        )

    def _backward(self, scene, upstream):
        means3D = scene.means3D.clone().requires_grad_(True)
        opacities = scene.opacities.clone().requires_grad_(True)
        flows = scene.flows.clone().requires_grad_(True)
        means2D = torch.zeros_like(means3D, requires_grad=True)
        out = scene.render(
            means3D=means3D, opacities=opacities, flows=flows, means2D=means2D
        )
        loss = (out["flow"] * upstream).sum()
        loss.backward()
        return {
            "loss": float(loss),
            "flows": flows.grad,
            "opacities": opacities.grad,
            "means3D": means3D.grad,
            "means2D": means2D.grad,
        }

    def test_flow_gradient_is_not_structurally_zero(self):
        scene = self._scene()
        grads = self._backward(scene, self._upstream(scene))
        self.assertIsNotNone(grads["flows"])
        self.assertGreater(
            float(grads["flows"].abs().max()),
            0.0,
            "dL_dflows came back all zero: the flow backward is not wired in",
        )

    def test_flow_gradient_matches_the_reference(self):
        scene = self._scene()
        upstream = self._upstream(scene)
        grads = self._backward(scene, upstream)
        alphas = scene.per_gaussian_alphas()
        expected, _ = flow_vjp(
            alphas, scene.unified_flows(), upstream, order=scene.depth_order()
        )
        self.assertLess(_max_rel_error(grads["flows"], expected[: scene.count]), 2e-3)

    def test_opacity_gradient_matches_the_reference(self):
        scene = self._scene()
        upstream = self._upstream(scene)
        grads = self._backward(scene, upstream)
        alphas = scene.per_gaussian_alphas()
        _, dL_dalphas = flow_vjp(
            alphas, scene.unified_flows(), upstream, order=scene.depth_order()
        )
        expected = dL_dopacity_from_dL_dalpha(
            dL_dalphas, alphas, scene.unified_opacities()
        )
        actual = grads["opacities"].reshape(-1)
        self.assertLess(_max_rel_error(actual, expected[: scene.count]), 5e-3)

    def test_flow_gradient_matches_finite_differences(self):
        scene = self._scene()
        upstream = self._upstream(scene)
        analytic = self._backward(scene, upstream)["flows"]
        # The composited flow is exactly linear in the flow values, so a
        # central difference here is not an approximation at all.
        step = 1e-2
        with torch.no_grad():
            for index in range(scene.count):
                for channel in range(2):
                    numeric = self._central_difference(
                        scene, upstream, "flows", index, channel, step
                    )
                    with self.subTest(index=index, channel=channel):
                        self.assertAlmostEqual(
                            numeric,
                            float(analytic[index, channel]),
                            delta=max(2e-3 * abs(numeric), 2e-3),
                        )

    def test_opacity_gradient_matches_finite_differences(self):
        scene = self._scene()
        upstream = self._upstream(scene)
        analytic = self._backward(scene, upstream)["opacities"].reshape(-1)
        step = 5e-3
        with torch.no_grad():
            for index in range(scene.count):
                numeric = self._central_difference(
                    scene, upstream, "opacities", index, 0, step
                )
                with self.subTest(index=index):
                    self.assertAlmostEqual(
                        numeric,
                        float(analytic[index]),
                        delta=max(2e-2 * abs(numeric), 5e-3),
                    )

    def test_projected_geometry_gradient_matches_finite_differences(self):
        """Perturbing the 3D mean moves the splat on screen, so the flow
        loss must respond, and the analytic mean gradient must predict it."""
        scene = self._scene()
        upstream = self._upstream(scene)
        analytic = self._backward(scene, upstream)["means3D"]
        step = 2e-3
        with torch.no_grad():
            for index in range(scene.count):
                for axis in range(2):  # lateral axes: the on-screen motion
                    numeric = self._central_difference(
                        scene, upstream, "means3D", index, axis, step
                    )
                    with self.subTest(index=index, axis=axis):
                        self.assertAlmostEqual(
                            numeric,
                            float(analytic[index, axis]),
                            delta=max(5e-2 * abs(numeric), 5e-2),
                        )

    def test_screenspace_mean2D_gradient_is_populated(self):
        """`means2D.grad` is the densification signal. A flow-only loss must
        reach it, otherwise flow supervision cannot steer the point budget."""
        scene = self._scene()
        grads = self._backward(scene, self._upstream(scene))
        self.assertIsNotNone(grads["means2D"])
        self.assertTrue(bool(torch.isfinite(grads["means2D"]).all()))
        self.assertGreater(float(grads["means2D"][:, :2].abs().max()), 0.0)

    def _central_difference(self, scene, upstream, field, index, channel, step):
        base = getattr(scene, field)
        plus = base.clone()
        minus = base.clone()
        plus[index, channel] += step
        minus[index, channel] -= step
        loss_plus = float((scene.render(**{field: plus})["flow"] * upstream).sum())
        loss_minus = float((scene.render(**{field: minus})["flow"] * upstream).sum())
        return (loss_plus - loss_minus) / (2.0 * step)


@unittest.skipUnless(HAVE_RASTERIZER, GPU_REASON)
class StaticGaussianFlowCouplingTests(unittest.TestCase):
    """A primitive with no flow of its own still shapes the flow image.

    Its alpha attenuates everything behind it, so raising it changes the
    composited flow. That is why the VJP's suffix term must be applied to
    STATIC Gaussians too, even though `f == 0` kills their direct term and
    they have no `dL_dflows` slot to write into.
    """

    def _scene(self):
        # One static occluder in front, two dynamic flow carriers behind.
        return TinyScene(
            means3D=[(0.0, 0.0, 0.3), (0.05, 0.05, 0.8)],
            opacities=[0.7, 0.6],
            flows=[(3.0, -2.0), (-1.0, 1.5)],
            static_means3D=[(0.0, 0.0, -0.4)],
            static_opacities=[0.5],
            static_colors=[(0.4, 0.4, 0.4)],
        )

    def test_static_occluder_is_actually_in_front(self):
        scene = self._scene()
        self.assertEqual(scene.depth_order(), [2, 0, 1])

    def test_forward_matches_the_reference_with_a_static_bank(self):
        scene = self._scene()
        rendered = scene.render()["flow"]
        expected, _ = scene.reference_flow()
        self.assertLess(_max_rel_error(rendered, expected), 2e-4)

    def test_static_occluder_receives_a_flow_mediated_opacity_gradient(self):
        scene = self._scene()
        generator = torch.Generator(device="cuda").manual_seed(11)
        upstream = torch.rand(
            (2, scene.height, scene.width), generator=generator, device="cuda"
        )
        opacities_static = scene.static_opacities.clone().requires_grad_(True)
        flows = scene.flows.clone().requires_grad_(True)
        out = scene.render(flows=flows, opacities_static=opacities_static)
        (out["flow"] * upstream).sum().backward()

        self.assertIsNotNone(opacities_static.grad)
        self.assertGreater(
            float(opacities_static.grad.abs().max()),
            0.0,
            "the static occluder got no opacity gradient from the flow loss",
        )

        alphas = scene.per_gaussian_alphas()
        _, dL_dalphas = flow_vjp(
            alphas, scene.unified_flows(), upstream, order=scene.depth_order()
        )
        expected = dL_dopacity_from_dL_dalpha(
            dL_dalphas, alphas, scene.unified_opacities()
        )
        self.assertLess(
            _max_rel_error(
                opacities_static.grad.reshape(-1), expected[scene.count :]
            ),
            5e-3,
        )

    def test_a_flow_free_dynamic_occluder_also_gets_the_suffix_term(self):
        """Same algebra without the static bank: f == 0 must not zero it."""
        scene = TinyScene(
            means3D=[(0.0, 0.0, -0.5), (0.0, 0.0, 0.5)],
            opacities=[0.5, 0.8],
            flows=[(0.0, 0.0), (3.0, -2.0)],  # the near one carries NO flow
        )
        generator = torch.Generator(device="cuda").manual_seed(13)
        upstream = torch.rand(
            (2, scene.height, scene.width), generator=generator, device="cuda"
        )
        opacities = scene.opacities.clone().requires_grad_(True)
        flows = scene.flows.clone().requires_grad_(True)
        out = scene.render(opacities=opacities, flows=flows)
        (out["flow"] * upstream).sum().backward()

        self.assertGreater(float(opacities.grad[0].abs()), 0.0)
        alphas = scene.per_gaussian_alphas()
        _, dL_dalphas = flow_vjp(
            alphas, scene.unified_flows(), upstream, order=scene.depth_order()
        )
        expected = dL_dopacity_from_dL_dalpha(
            dL_dalphas, alphas, scene.unified_opacities()
        )
        self.assertLess(_max_rel_error(opacities.grad.reshape(-1), expected), 5e-3)
        # Its own flow-value gradient is the ordinary weight * upstream, not
        # zero, even though its flow happens to be zero.
        self.assertGreater(float(flows.grad[0].abs().max()), 0.0)


@unittest.skipUnless(HAVE_RASTERIZER, GPU_REASON)
class CheckpointBoundaryTests(unittest.TestCase):
    """Cross the 32-stride checkpoint, which is the novel part of the VJP.

    The forward writes `sampled_arflow` every 32nd step of the tile's batch
    loop (`forward.cu:709-713`, guarded by `j % 32 == 0`), and the backward
    reconstructs the suffix sum from it as
    `-pixel_flows + sampled_arflow[bucket]`. Every other test in this file
    uses a handful of primitives, so the whole reconstruction runs inside
    bucket 0 with `sampled_arflow == 0` and a wrong bucket stride, a wrong
    channel stride or a checkpoint taken one batch late would all still
    pass. These scenes put >32 primitives in every tile so the boundary is
    crossed two or three times per pixel.

    The scene is built to stay inside the oracle's modelling assumptions:
    `composite_flow` implements no early termination, whereas the kernel
    stops a pixel once `T < 1e-4` (`forward.cu:746`). Per-primitive opacity
    is therefore held low enough that `T` survives the whole stack, and
    `test_the_scene_is_a_valid_model_of_the_kernel` asserts that rather
    than assuming it.
    """

    CONTRIBUTORS = 72
    OPACITY = 0.08

    def _scene(self):
        means3D, opacities, flows = [], [], []
        for index in range(self.CONTRIBUTORS):
            # Strictly increasing depth: the tile sort is then unambiguous
            # and the oracle's `depth_order()` cannot disagree with the
            # kernel's over a tie.
            z = -0.6 + 1.2 * index / (self.CONTRIBUTORS - 1)
            means3D.append((0.0, 0.0, z))
            opacities.append(self.OPACITY)
            # Distinct per primitive, so a suffix sum taken from the wrong
            # bucket lands on visibly wrong values instead of averaging out.
            angle = 0.7 * index
            flows.append((2.0 * math.cos(angle), 2.0 * math.sin(angle)))
        return TinyScene(means3D=means3D, opacities=opacities, flows=flows)

    def test_the_scene_is_a_valid_model_of_the_kernel(self):
        """Guards the test itself: boundary crossed, no early termination."""
        scene = self._scene()
        alphas = scene.per_gaussian_alphas()

        # Contributors are the primitives the kernel does not cull at
        # `alpha < 1/255`. `per_gaussian_alphas` renders each one through
        # the same kernel, so that cull is already baked into `alphas`.
        contributors = (alphas > 1.0 / 255.0).sum(dim=0)
        self.assertGreater(
            int(contributors.max()),
            32,
            "scene never crosses a checkpoint boundary, so it tests nothing",
        )

        _, transmittance = composite_flow(
            alphas, scene.unified_flows(), order=scene.depth_order()
        )
        self.assertGreater(
            float(transmittance.min()),
            1e-4,
            "a pixel terminated early, which the reference oracle does not "
            "model, so any parity failure here would be the test's fault",
        )

    def test_forward_matches_the_reference_across_buckets(self):
        scene = self._scene()
        rendered = scene.render()["flow"]
        expected, _ = scene.reference_flow()
        self.assertLess(_max_rel_error(rendered, expected), 2e-3)

    def test_flow_gradient_matches_the_reference_across_buckets(self):
        scene = self._scene()
        generator = torch.Generator(device="cuda").manual_seed(17)
        upstream = torch.rand(
            (2, scene.height, scene.width), generator=generator, device="cuda"
        )
        flows = scene.flows.clone().requires_grad_(True)
        out = scene.render(flows=flows)
        (out["flow"] * upstream).sum().backward()

        alphas = scene.per_gaussian_alphas()
        expected, _ = flow_vjp(
            alphas, scene.unified_flows(), upstream, order=scene.depth_order()
        )
        self.assertLess(_max_rel_error(flows.grad, expected[: scene.count]), 2e-2)

    def test_opacity_gradient_matches_the_reference_across_buckets(self):
        """The suffix term, which is what the checkpoint reconstructs.

        `dL_dflows` needs no `arflow` at all — it is `weight * upstream`.
        Only the alpha path reads the reconstructed suffix, so this is the
        test that a bucket-indexing defect actually fails.
        """
        scene = self._scene()
        generator = torch.Generator(device="cuda").manual_seed(19)
        upstream = torch.rand(
            (2, scene.height, scene.width), generator=generator, device="cuda"
        )
        opacities = scene.opacities.clone().requires_grad_(True)
        out = scene.render(opacities=opacities)
        (out["flow"] * upstream).sum().backward()

        alphas = scene.per_gaussian_alphas()
        _, dL_dalphas = flow_vjp(
            alphas, scene.unified_flows(), upstream, order=scene.depth_order()
        )
        expected = dL_dopacity_from_dL_dalpha(
            dL_dalphas, alphas, scene.unified_opacities()
        )
        self.assertLess(_max_rel_error(opacities.grad.reshape(-1), expected), 2e-2)

    def test_opacity_gradient_past_the_boundary_matches_central_differences(self):
        """Independent of the oracle: perturb a primitive in a later bucket.

        Index 50 is reached after the checkpoint at stride 32, so its
        gradient is computed from a reconstructed `arflow`, not from the
        zero-initialised bucket 0.
        """
        scene = self._scene()
        index = 50
        self.assertGreater(index, 32)
        generator = torch.Generator(device="cuda").manual_seed(23)
        upstream = torch.rand(
            (2, scene.height, scene.width), generator=generator, device="cuda"
        )

        opacities = scene.opacities.clone().requires_grad_(True)
        out = scene.render(opacities=opacities)
        (out["flow"] * upstream).sum().backward()
        analytic = float(opacities.grad[index, 0])

        step = 5e-3
        plus = scene.opacities.clone()
        minus = scene.opacities.clone()
        plus[index, 0] += step
        minus[index, 0] -= step
        with torch.no_grad():
            loss_plus = float((scene.render(opacities=plus)["flow"] * upstream).sum())
            loss_minus = float((scene.render(opacities=minus)["flow"] * upstream).sum())
        numeric = (loss_plus - loss_minus) / (2.0 * step)

        self.assertGreater(
            abs(numeric),
            1e-3,
            "central difference is at the float32 noise floor, so this "
            "comparison would pass for a zero gradient too",
        )
        self.assertLess(abs(analytic - numeric) / abs(numeric), 5e-2)


@unittest.skipUnless(HAVE_RASTERIZER, GPU_REASON)
class DegenerateSceneTests(unittest.TestCase):
    def test_empty_scene_renders_zero_flow(self):
        device = "cuda"
        scene = TinyScene(
            means3D=[(0.0, 0.0, 0.0)],
            opacities=[0.5],
            flows=[(1.0, 1.0)],
        )
        with torch.no_grad():
            scene.means3D = torch.zeros((0, 3), dtype=torch.float32, device=device)
            scene.opacities = torch.zeros((0, 1), dtype=torch.float32, device=device)
            scene.flows = torch.zeros((0, 2), dtype=torch.float32, device=device)
            scene.shs = torch.zeros((0, 1, 3), dtype=torch.float32, device=device)
            scene.scales = torch.zeros((0, 3), dtype=torch.float32, device=device)
            scene.rotations = torch.zeros((0, 4), dtype=torch.float32, device=device)
            scene.ts = torch.zeros((0, 1), dtype=torch.float32, device=device)
            scene.scales_t = torch.zeros((0, 1), dtype=torch.float32, device=device)
            scene.rotations_r = torch.zeros((0, 4), dtype=torch.float32, device=device)
            scene.count = 0
            out = scene.render()
        self.assertEqual(float(out["flow"].abs().max()), 0.0)
        self.assertEqual(float(out["render"].abs().max()), 0.0)

    def test_offscreen_gaussian_contributes_no_flow(self):
        # An on-screen companion keeps at least one bucket alive; the point
        # is that the off-screen primitive neither writes flow nor collects
        # a flow gradient.
        scene = TinyScene(
            means3D=[OFFSCREEN, (0.0, 0.0, 0.0)],
            opacities=[0.9, 0.6],
            flows=[(5.0, 5.0), (0.0, 0.0)],
        )
        flows = scene.flows.clone().requires_grad_(True)
        out = scene.render(flows=flows)
        self.assertEqual(float(out["flow"].abs().max()), 0.0)
        out["flow"].sum().backward()
        self.assertEqual(float(flows.grad[0].abs().max()), 0.0)
        self.assertGreater(float(flows.grad[1].abs().max()), 0.0)

    def test_alpha_below_the_cull_threshold_contributes_nothing(self):
        # 1/255 is the renderer's contribution floor; 0.002 * G never
        # reaches it, so this Gaussian is skipped at every pixel.
        scene = TinyScene(
            means3D=[(0.0, 0.0, 0.0)],
            opacities=[0.002],
            flows=[(5.0, -5.0)],
        )
        flows = scene.flows.clone().requires_grad_(True)
        out = scene.render(flows=flows)
        self.assertLess(float(out["alpha"].max()), 1.0 / 255.0)
        self.assertEqual(float(out["flow"].abs().max()), 0.0)
        out["flow"].sum().backward()
        self.assertEqual(float(flows.grad.abs().max()), 0.0)

    def test_high_opacity_is_clamped_at_the_renderer_ceiling(self):
        # alpha = min(0.99, opacity * G). The backward deliberately ignores
        # the clamp when converting dL_dalpha to dL_dG -- a PRE-EXISTING
        # approximation the flow term inherits unchanged. So this pins the
        # forward exactly and only demands finiteness of the gradients.
        # The wide scale keeps G ~ 1 near the mean, so the ceiling is the
        # thing that binds rather than the footprint falloff.
        scene = TinyScene(
            means3D=[(0.0, 0.0, 0.0)],
            opacities=[1.0],
            flows=[(4.0, 0.0)],
            scales=[(1.0, 1.0, 1.0)],
        )
        flows = scene.flows.clone().requires_grad_(True)
        opacities = scene.opacities.clone().requires_grad_(True)
        out = scene.render(flows=flows, opacities=opacities)
        alpha_map = out["alpha"].reshape(-1)
        peak = int(torch.argmax(alpha_map))
        alpha = float(alpha_map[peak])
        self.assertLessEqual(alpha, 0.99 + 1e-5)
        self.assertAlmostEqual(alpha, 0.99, delta=1e-3)
        self.assertAlmostEqual(
            float(out["flow"][0].reshape(-1)[peak]), 4.0 * alpha, delta=5e-3
        )
        out["flow"].sum().backward()
        self.assertTrue(bool(torch.isfinite(flows.grad).all()))
        self.assertTrue(bool(torch.isfinite(opacities.grad).all()))
        self.assertGreater(float(flows.grad.abs().max()), 0.0)


@unittest.skipUnless(HAVE_RASTERIZER, GPU_REASON)
class DisabledFlowEquivalenceTests(unittest.TestCase):
    """With flow disabled the renderer must behave exactly as before.

    `gaussian_renderer.render` sets `flow_2d = zeros` whenever
    `enable_rendered_flow` is off, so the pre-patch behaviour is the
    zero-flow case: identical colour, depth and alpha, and identical
    gradients on every parameter that is not the flow itself.
    """

    def _scene(self, flows):
        return TinyScene(
            means3D=[(-0.15, 0.05, -0.5), (0.05, -0.1, 0.0), (0.2, 0.15, 0.5)],
            opacities=[0.6, 0.7, 0.65],
            flows=flows,
            colors=[(0.7, 0.2, 0.4), (0.3, 0.6, 0.5), (0.5, 0.5, 0.8)],
        )

    def _colour_backward(self, scene):
        means3D = scene.means3D.clone().requires_grad_(True)
        opacities = scene.opacities.clone().requires_grad_(True)
        out = scene.render(means3D=means3D, opacities=opacities)
        generator = torch.Generator(device="cuda").manual_seed(3)
        weights = torch.rand(out["render"].shape, generator=generator, device="cuda")
        (out["render"] * weights).sum().backward()
        return out, means3D.grad, opacities.grad

    def test_zero_flow_renders_a_zero_flow_image(self):
        scene = self._scene([(0.0, 0.0)] * 3)
        self.assertEqual(float(scene.render()["flow"].abs().max()), 0.0)

    def test_colour_depth_alpha_are_bit_identical_regardless_of_flow(self):
        disabled = self._scene([(0.0, 0.0)] * 3)
        enabled = self._scene([(2.0, -1.0), (-1.5, 2.5), (0.75, 0.5)])
        with torch.no_grad():
            left = disabled.render()
            right = enabled.render()
        for key in ("render", "depth", "alpha"):
            with self.subTest(output=key):
                self.assertTrue(
                    torch.equal(left[key], right[key]),
                    f"the flow channel perturbed the {key} output",
                )

    def test_non_flow_gradients_are_bit_identical_regardless_of_flow(self):
        disabled = self._scene([(0.0, 0.0)] * 3)
        enabled = self._scene([(2.0, -1.0), (-1.5, 2.5), (0.75, 0.5)])
        _, left_means, left_opacities = self._colour_backward(disabled)
        _, right_means, right_opacities = self._colour_backward(enabled)
        self.assertTrue(torch.equal(left_means, right_means))
        self.assertTrue(torch.equal(left_opacities, right_opacities))

    def test_a_flow_loss_on_a_zero_flow_field_moves_nothing(self):
        """The disabled path must stay gradient-free with respect to the
        geometry: with every flow value zero, the flow term contributes
        exactly zero to dL_dalpha, so opacity and mean gradients vanish."""
        scene = self._scene([(0.0, 0.0)] * 3)
        means3D = scene.means3D.clone().requires_grad_(True)
        opacities = scene.opacities.clone().requires_grad_(True)
        out = scene.render(means3D=means3D, opacities=opacities)
        generator = torch.Generator(device="cuda").manual_seed(5)
        upstream = torch.rand(out["flow"].shape, generator=generator, device="cuda")
        (out["flow"] * upstream).sum().backward()
        self.assertEqual(float(opacities.grad.abs().max()), 0.0)
        self.assertEqual(float(means3D.grad.abs().max()), 0.0)


@unittest.skipUnless(
    HAVE_MODEL,
    "needs the full model stack (scene + gaussian_renderer + CUDA extensions)"
    + (f" ({MODEL_IMPORT_ERROR})" if MODEL_IMPORT_ERROR else ""),
)
class MotionLoraFlowGradientTests(unittest.TestCase):
    """A flow loss must reach the motion parameters through `render()`.

    This is the end the whole repair exists for. The chain is

        _motion_lora_coeff / _motion_lora_basis
          -> get_dynamic_xyz(t) and get_dynamic_xyz(t + motion_track_dt)
          -> project_points_to_screen -> flow_2d
          -> rasterized flow image -> loss

    Two traps make a naive version of this test vacuous, and both are
    handled below: the LoRA coefficients are initialised to exactly zero
    (so the BASIS gradient is zero however well the renderer works), and
    a camera timestamp equal to every Gaussian's `_t` makes the sampled
    basis interval empty (so the COEFF gradient is zero too).

    The disabled-flow companion test is what makes the positive result
    mean something: with `enable_rendered_flow=False` the rendered field
    is identically zero for ANY geometry, so a flow loss must produce
    exactly zero on both parameters. Before the repair BOTH cases were
    zero, because the flow backward was a kernel that never launched.
    """

    WIDTH = 48
    HEIGHT = 48
    COUNT = 24
    TIMESTAMP = 0.05

    def _options(self):
        parser = ArgumentParser(add_help=False)
        opt_group = OptimizationParams(parser)
        pipe_group = PipelineParams(parser)
        parsed = parser.parse_args([])
        opt = opt_group.extract(parsed)
        pipe = pipe_group.extract(parsed)
        opt.motion_model = "lora"
        opt.motion_scaffold_enable = False
        pipe.compute_cov3D_python = False
        pipe.convert_SHs_python = False
        pipe.env_map_res = 0
        pipe.debug = False
        return opt, pipe

    def _build(self, enable_rendered_flow):
        opt, pipe = self._options()
        opt.enable_rendered_flow = bool(enable_rendered_flow)

        generator = np.random.default_rng(19)
        points = generator.uniform(-0.4, 0.4, size=(self.COUNT, 3)).astype(np.float32)
        colors = generator.uniform(0.2, 0.8, size=(self.COUNT, 3)).astype(np.float32)
        cloud = BasicPointCloud(
            points=points,
            colors=colors,
            normals=np.zeros_like(points),
            time=np.zeros((self.COUNT, 1), dtype=np.float32),
        )

        gaussians = GaussianModel(
            0, gaussian_dim=4, time_duration=[-0.5, 0.5], rot_4d=False, sh_degree_t=0
        )
        gaussians.create_from_pcd(cloud, 1.0)
        gaussians.training_setup(opt)

        self.assertEqual(bool(gaussians.enable_rendered_flow), bool(enable_rendered_flow))
        self.assertIsNotNone(
            gaussians._motion_lora_basis, "training_setup did not allocate the LoRA basis"
        )

        # Trap 1: the coefficients ship as exact zeros, which zeroes the
        # basis gradient no matter what the renderer does.
        with torch.no_grad():
            torch_generator = torch.Generator(device=gaussians._motion_lora_coeff.device)
            torch_generator.manual_seed(23)
            gaussians._motion_lora_coeff.copy_(
                torch.randn(
                    gaussians._motion_lora_coeff.shape,
                    generator=torch_generator,
                    device=gaussians._motion_lora_coeff.device,
                )
                * 0.1
            )

        world_view, full_proj, fov = _camera_matrices()
        camera = MiniCam(
            self.WIDTH, self.HEIGHT, fov, fov, 0.01, 100.0, world_view, full_proj
        )
        # Trap 2: a timestamp equal to every Gaussian's `_t` collapses the
        # sampled basis interval and zeroes the coefficient gradient.
        camera.timestamp = self.TIMESTAMP
        return gaussians, camera, pipe

    def _flow_loss_grads(self, enable_rendered_flow):
        gaussians, camera, pipe = self._build(enable_rendered_flow)
        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        for parameter in (gaussians._motion_lora_coeff, gaussians._motion_lora_basis):
            parameter.grad = None
        package = render_gaussians(camera, gaussians, pipe, background)
        flow = package["flow"]
        generator = torch.Generator(device="cuda").manual_seed(29)
        upstream = torch.rand(flow.shape, generator=generator, device="cuda")
        (flow * upstream).sum().backward()
        return {
            "flow_absmax": float(flow.abs().max()),
            "coeff": gaussians._motion_lora_coeff.grad,
            "basis": gaussians._motion_lora_basis.grad,
        }

    def test_flow_loss_reaches_the_lora_coefficients(self):
        result = self._flow_loss_grads(True)
        self.assertGreater(result["flow_absmax"], 0.0, "the rendered flow field is all zero")
        self.assertIsNotNone(result["coeff"])
        self.assertGreater(
            float(result["coeff"].abs().max()),
            0.0,
            "no gradient reached _motion_lora_coeff through the rendered flow",
        )

    def test_flow_loss_reaches_the_lora_basis(self):
        result = self._flow_loss_grads(True)
        self.assertIsNotNone(result["basis"])
        self.assertGreater(
            float(result["basis"].abs().max()),
            0.0,
            "no gradient reached _motion_lora_basis through the rendered flow",
        )

    def test_disabled_flow_leaves_the_motion_parameters_untouched(self):
        result = self._flow_loss_grads(False)
        self.assertEqual(result["flow_absmax"], 0.0)
        for name in ("coeff", "basis"):
            gradient = result[name]
            with self.subTest(parameter=name):
                if gradient is not None:
                    self.assertEqual(float(gradient.abs().max()), 0.0)


class EnvironmentDisclosureTests(unittest.TestCase):
    def test_the_gpu_pins_actually_ran_somewhere(self):
        """A skipped rasterizer pin is not a passing one."""
        if not HAVE_TORCH:
            self.skipTest("torch is not importable in this environment")
        if not HAVE_CUDA:
            self.skipTest(
                "no CUDA device on this runner: the flow forward/backward pins "
                "did NOT run here and must be run on a GPU node"
            )
        if not HAVE_RASTERIZER:
            self.skipTest(
                "CUDA is present but the rasterizer did not import "
                f"({RASTERIZER_IMPORT_ERROR}): the flow pins did NOT run"
            )
        self.assertTrue(HAVE_RASTERIZER)

    def test_the_full_model_stack_pins_actually_ran_somewhere(self):
        if not HAVE_RASTERIZER:
            self.skipTest("no rasterizer: see test_the_gpu_pins_actually_ran_somewhere")
        if not HAVE_MODEL:
            self.skipTest(
                "the model stack did not import "
                f"({MODEL_IMPORT_ERROR}): the LoRA flow-gradient pins did NOT run"
            )
        self.assertTrue(HAVE_MODEL)


if __name__ == "__main__":
    unittest.main()
