"""Anti-vacuity tests for the B1-F/B1-X flow-derived velocity initialization
(:mod:`scene.packet_birth_flow`).

The properties under test are the ones this cell can silently get wrong:

1. FLAG OFF IS BIT IDENTICAL. With ``packet_birth_flow_init`` False the target
   rows are ``torch.equal`` to the ones the pre-change implementation (read
   straight out of ``git show HEAD``) produces, and a whole seeded event leaves
   every parameter identical. This is the load-bearing test: B1-F is only
   interpretable against a B1 that the change did not move.
2. The two knobs validate and round-trip through the trainer namespace.
3. The recovered world velocity matches the analytic pinhole answer for a known
   camera, a known constant depth and a known flow field.
4. The solved LoRA coefficients reproduce the target displacement through the
   SAME einsum ``GaussianModel.get_lora_motion_offset`` uses.
5. The solve can never OVERSHOOT: the realized displacement is an orthogonal
   projection of the target, so its norm never exceeds the target's. This is
   why there is no output-side magnitude guard, and
   ``flow_realized_ratio_mean`` reports 1.0 for a full-rank probe basis and
   strictly less for a rank-deficient one.
6. Per-row invalidity fails closed to zero motion and is attributed to the
   right counter, including the input-side flow-magnitude outlier cut.
7. A missing flow ASSET with the flag on raises ``ContractError`` -- it never
   degrades quietly to B1.
8. The held-out camera is refused, and the camera-swapped rule cannot reach it.
9. The camera-swapped rule is deterministic and never returns its input.
"""

import importlib.util
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from torch import nn

from depth_visibility.capacity import CapacityBank
from depth_visibility.errors import ContractError
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.packet_birth import (
    PacketBirthConfig,
    PacketBirthState,
    _build_packet_target_rows,
    backproject_pixels,
    maybe_packet_birth,
    setup_packet_birth,
)
from scene.packet_birth_flow import (
    FLOW_OUTLIER_QUANTILE,
    _camera_ids,
    FLOW_SOURCE_CAMERA_SWAPPED,
    FLOW_SOURCE_CORRECT,
    assert_frame_time_convention,
    build_flow_assets,
    flow_world_velocity,
    lora_probe_basis,
    parse_camera_name,
    realized_displacement,
    solve_lora_coefficients,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

ROWS = 40
DONOR_FRACTION = 0.25
IMAGE_SIZE = 8
FOCAL = 8.0
PRINCIPAL = 4.0
SITE_DEPTH = 5.0
FRAME_DT = 0.0333333333
FRAME_INDEX = 37
TIMESTAMP = FRAME_INDEX * FRAME_DT
TIME_DURATION = (0.0, 1.6340)
LORA_RANK = 8
LORA_ANCHORS = 32
LORA_INIT_SCALE = 0.01

TRAIN_CAMERAS = ("cam01_0037", "cam03_0037", "cam05_0037", "cam07_0037")
HOLDOUT_CAMERAS = ("cam00_0037",)
EVENT_CAMERA = "cam03_0037"

#: ``(column, row)`` pairs, the spelling the B1 event tests use.
#: A corner block, so the out-of-raster branch is reachable by a small flow.
BLOCK_PIXELS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
#: Ten INTERIOR sites: a 2x5 patch far enough from every border that a
#: few-pixel flow displaces all of them inside the raster, so an integration
#: event's funnel is total == valid rather than total == valid + border losses.
INTERIOR_PIXELS = [(column, row) for column in (1, 2) for row in (1, 2, 3, 4, 5)]
EVENT_PIXELS = INTERIOR_PIXELS

GT_CHANNEL_VALUES = (0.4, 0.5, 0.6)


class _StubCamera(object):
    """Duck-typed camera that runs the REAL ``Camera.get_rays``."""

    get_rays = Camera.get_rays

    def __init__(self, image_name=EVENT_CAMERA, timestamp=TIMESTAMP, centre=(0.0, 0.0, 0.0)):
        self.image_name = image_name
        self.timestamp = timestamp
        self.image_height = IMAGE_SIZE
        self.image_width = IMAGE_SIZE
        self.cx = PRINCIPAL
        self.cy = PRINCIPAL
        self.fl_x = FOCAL
        self.fl_y = FOCAL
        self.data_device = torch.device("cpu")
        centre_tensor = torch.tensor(centre, dtype=torch.float32)
        world_to_camera = torch.eye(4)
        world_to_camera[:3, 3] = -centre_tensor
        # scene/cameras.py stores the TRANSPOSED world-to-camera matrix.
        self.world_view_transform = world_to_camera.transpose(0, 1)
        self.camera_center = centre_tensor
        projection = torch.zeros(4, 4)
        span = float(IMAGE_SIZE - 1)
        projection[0, 0] = 2.0 * FOCAL / span
        projection[0, 2] = 2.0 * PRINCIPAL / span - 1.0
        projection[1, 1] = 2.0 * FOCAL / span
        projection[1, 2] = 2.0 * PRINCIPAL / span - 1.0
        projection[2, 2] = 1.0
        projection[3, 2] = 1.0
        self.projection_matrix = projection.transpose(0, 1)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix


def _lora_basis():
    """The basis ``_ensure_route_and_motion_tensors`` builds
    (scene/gaussian_model.py:1266-1272), reproduced so the stub exercises the
    REAL sampler on a REAL basis shape."""

    anchor_grid = torch.linspace(-1.0, 1.0, LORA_ANCHORS)
    basis = torch.zeros((LORA_RANK, LORA_ANCHORS, 3))
    for rank_idx in range(LORA_RANK):
        freq = rank_idx // 6 + 1
        phase = (
            torch.sin(anchor_grid * torch.pi * freq)
            if rank_idx % 2 == 0
            else torch.cos(anchor_grid * torch.pi * freq) - 1.0
        )
        basis[rank_idx, :, rank_idx % 3] = phase * LORA_INIT_SCALE
    return basis


class _StubGaussians(object):
    """Minimal model exposing the capacity substrate and the REAL LoRA sampler."""

    # The genuine implementations, so the probe basis and the offset map under
    # test are the ones the trainer actually evaluates.
    _sample_lora_basis = GaussianModel._sample_lora_basis
    get_lora_motion_offset = GaussianModel.get_lora_motion_offset

    def __init__(self, rows=ROWS):
        generator = torch.Generator().manual_seed(3)
        self._xyz = nn.Parameter(torch.rand(rows, 3, generator=generator))
        self._features_dc = nn.Parameter(torch.rand(rows, 1, 3, generator=generator))
        self._features_rest = nn.Parameter(torch.rand(rows, 15, 3, generator=generator))
        self._scaling = nn.Parameter(torch.full((rows, 3), -2.0))
        self._rotation = nn.Parameter(torch.rand(rows, 4, generator=generator))
        self._opacity = nn.Parameter(torch.arange(rows, dtype=torch.float32).reshape(-1, 1))
        self._t = nn.Parameter(torch.zeros(rows, 1))
        self._scaling_t = nn.Parameter(torch.full((rows, 1), -1.0))
        self._route_logit = nn.Parameter(torch.zeros(rows, 1))
        self._motion_lora_coeff = nn.Parameter(
            torch.rand(rows, LORA_RANK, generator=generator)
        )
        self._motion_lora_basis = nn.Parameter(_lora_basis())
        self.max_radii2D = torch.zeros(rows)
        self.denom = torch.ones(rows, 1)
        self.xyz_gradient_accum = torch.ones(rows, 1)
        self.t_gradient_accum = torch.ones(rows, 1)
        self._capacity_stable_ids = torch.arange(rows, dtype=torch.long)
        self._capacity_generation = torch.zeros(rows, dtype=torch.long)
        self._capacity_last_reassigned = torch.zeros(rows, dtype=torch.long)
        self.route_logit_init = 4.0
        self._packet_ids = torch.empty(0, dtype=torch.long)
        self.gaussian_dim = 4
        self.rot_4d = False
        self.motion_model = "lora"
        self.motion_lora_rank = LORA_RANK
        self.motion_lora_anchors = LORA_ANCHORS
        self.time_duration = TIME_DURATION
        self.optimizer = torch.optim.Adam(
            [{"params": [value], "name": name} for name, value in self._named_parameters()],
            lr=0.0,
        )
        for _, parameter in self._named_parameters():
            parameter.grad = torch.ones_like(parameter)
        self.optimizer.step()
        for state in self.optimizer.state.values():
            state["exp_avg"].fill_(1.0)
            state["exp_avg_sq"].fill_(1.0)

    @property
    def get_t(self):
        return self._t

    @property
    def get_xyz(self):
        return self._xyz

    def get_dynamic_xyz(self, timestamp):
        return self._xyz.detach() + self.get_lora_motion_offset(timestamp).detach()

    def _named_parameters(self):
        return [
            ("_xyz", self._xyz),
            ("_features_dc", self._features_dc),
            ("_features_rest", self._features_rest),
            ("_scaling", self._scaling),
            ("_rotation", self._rotation),
            ("_opacity", self._opacity),
            ("_t", self._t),
            ("_scaling_t", self._scaling_t),
            ("_route_logit", self._route_logit),
            ("_motion_lora_coeff", self._motion_lora_coeff),
        ]

    def build_capacity_bank(self):
        return CapacityBank(
            parameters=dict(self._named_parameters()),
            accumulators={
                "xyz_gradient_accum": self.xyz_gradient_accum,
                "t_gradient_accum": self.t_gradient_accum,
                "denom": self.denom,
                "max_radii2D": self.max_radii2D,
            },
            stable_ids=self._capacity_stable_ids,
            generation=self._capacity_generation,
            last_reassigned=self._capacity_last_reassigned,
        )


class _StubFlowProvider(object):
    """Stands in for ``MotionPriorCache``; records every name it was asked for."""

    def __init__(self, entries):
        self.entries = dict(entries)
        self.calls = []

    def get_track_flow(self, camera, target_hw):
        self.calls.append((camera.image_name, tuple(target_hw)))
        entry = self.entries.get(camera.image_name)
        if entry is None:
            return None, None
        return entry


def _uniform_flow(u_value, v_value):
    """``(2, H, W)`` flow with channel 0 = horizontal, channel 1 = vertical."""

    flow = torch.zeros(2, IMAGE_SIZE, IMAGE_SIZE)
    flow[0] = float(u_value)
    flow[1] = float(v_value)
    mask = torch.ones(1, IMAGE_SIZE, IMAGE_SIZE)
    return flow, mask


def _stub_opt(**overrides):
    values = dict(
        packet_birth_enable=True,
        packet_birth_interval=1,
        packet_birth_fraction=DONOR_FRACTION,
        packet_birth_t_sigma_frames=1.5,
        packet_birth_from_iter=1,
        packet_birth_until_iter=-1,
        packet_birth_flow_init=False,
        packet_birth_flow_source=FLOW_SOURCE_CORRECT,
        densify_until_iter=10,
        motion_track_dt=FRAME_DT,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _event_inputs(pixels=EVENT_PIXELS, alpha_value=1.0):
    image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
    gt_image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
    for col, row in pixels:
        for channel, value in enumerate(GT_CHANNEL_VALUES):
            gt_image[channel, row, col] = value
    alpha = torch.full((1, IMAGE_SIZE, IMAGE_SIZE), float(alpha_value))
    depth = alpha * SITE_DEPTH
    return image, gt_image, depth, alpha


def _flat(pixels):
    return torch.tensor([row * IMAGE_SIZE + col for col, row in pixels], dtype=torch.long)


def _assets(entries, train=TRAIN_CAMERAS, holdout=HOLDOUT_CAMERAS):
    return build_flow_assets(
        _StubFlowProvider(entries), train_cameras=train, holdout_cameras=holdout
    )


#: The last commit BEFORE flow-derived velocity initialization landed, i.e.
#: the implementation the flag-off claim is about. This is deliberately a
#: FIXED sha rather than ``HEAD``: pinning to HEAD made the test
#: self-invalidating -- the moment the flow work was committed, the
#: "baseline" became the post-change code and the non-vacuity assertion
#: (that the baseline REJECTS the new keyword) started failing. A bit-identity
#: test whose reference moves with the branch cannot prove anything.
FLAG_OFF_BASELINE_COMMIT = "f666dd7"


def _load_head_baseline():
    """Import ``scene/packet_birth.py`` exactly as committed at the baseline.

    The flag-off claim is a claim about the PREVIOUS implementation, so the
    comparison reads that implementation rather than a paraphrase of it.
    """

    result = subprocess.run(
        ["git", "show", "%s:scene/packet_birth.py" % FLAG_OFF_BASELINE_COMMIT],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None
    directory = tempfile.mkdtemp(prefix="packet_birth_head_")
    path = Path(directory) / "packet_birth_head.py"
    path.write_bytes(result.stdout)
    spec = importlib.util.spec_from_file_location("packet_birth_head", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["packet_birth_head"] = module
    spec.loader.exec_module(module)
    return module


class FlagOffBitIdentityTests(unittest.TestCase):
    """Property 1 -- the load-bearing one."""

    def _target_row_inputs(self):
        gaussians = _StubGaussians()
        bank = gaussians.build_capacity_bank()
        donors = torch.arange(6, dtype=torch.long)
        points = torch.rand(6, 3, generator=torch.Generator().manual_seed(11))
        colors = torch.rand(6, 3, generator=torch.Generator().manual_seed(12))
        return bank, donors, points, colors

    def test_target_rows_match_the_head_implementation_exactly(self):
        baseline = _load_head_baseline()
        if baseline is None:
            self.skipTest("git show %s:scene/packet_birth.py is unavailable" % FLAG_OFF_BASELINE_COMMIT)
        bank, donors, points, colors = self._target_row_inputs()
        # Non-vacuity: the baseline really is the PRE-change implementation, so
        # it cannot know the new keyword. If HEAD ever carries this change the
        # comparison would silently compare the new code with itself.
        with self.assertRaises(TypeError):
            baseline._build_packet_target_rows(
                bank, donors, points, colors, TIMESTAMP, 0.05, 4.0,
                motion_lora_coeff=torch.zeros(6, LORA_RANK),
            )
        arguments = (bank, donors, points, colors, TIMESTAMP, 0.05, 4.0)
        expected = baseline._build_packet_target_rows(*arguments)
        produced = _build_packet_target_rows(*arguments)
        self.assertEqual(sorted(expected), sorted(produced))
        for name in expected:
            self.assertTrue(torch.equal(produced[name], expected[name]), name)

    def test_a_whole_flag_off_event_matches_the_head_implementation(self):
        baseline = _load_head_baseline()
        if baseline is None:
            self.skipTest("git show %s:scene/packet_birth.py is unavailable" % FLAG_OFF_BASELINE_COMMIT)
        image, gt_image, depth, alpha = _event_inputs()

        def run(module, opt):
            torch.manual_seed(17)
            gaussians = _StubGaussians()
            state = module.setup_packet_birth(opt)
            record = module.maybe_packet_birth(
                state, gaussians, 3, _StubCamera(), image, gt_image, depth, alpha, None
            )
            return gaussians, record

        head_gaussians, head_record = run(baseline, _stub_opt())
        current_gaussians, current_record = run(
            sys.modules["scene.packet_birth"], _stub_opt()
        )
        for name, value in current_gaussians._named_parameters():
            self.assertTrue(
                torch.equal(value.detach(), dict(head_gaussians._named_parameters())[name].detach()),
                name,
            )
        trimmed = {
            key: value for key, value in current_record.items() if not key.startswith("flow_")
        }
        self.assertEqual(trimmed, head_record)
        self.assertEqual(current_record["flow_init"], False)
        self.assertEqual(current_record["flow_sites_total"], 0)

    def test_flag_off_leaves_every_motion_coefficient_at_exactly_zero(self):
        torch.manual_seed(17)
        gaussians = _StubGaussians()
        image, gt_image, depth, alpha = _event_inputs()
        record = maybe_packet_birth(
            setup_packet_birth(_stub_opt()),
            gaussians,
            3,
            _StubCamera(),
            image,
            gt_image,
            depth,
            alpha,
            None,
        )
        relocated = torch.arange(record["donors"], dtype=torch.long)
        self.assertGreater(int(relocated.numel()), 0)
        self.assertTrue(
            torch.equal(
                gaussians._motion_lora_coeff.detach()[relocated],
                torch.zeros(int(relocated.numel()), LORA_RANK),
            )
        )
        self.assertFalse(record["flow_init"])


class ConfigurationTests(unittest.TestCase):
    """Property 2."""

    def test_defaults_reproduce_b1(self):
        cfg = PacketBirthConfig(enable=True, from_iter=1, until_iter=10).validate()
        self.assertFalse(cfg.flow_init)
        self.assertEqual(cfg.flow_source, FLOW_SOURCE_CORRECT)

    def test_an_unknown_flow_source_is_refused(self):
        with self.assertRaises(ContractError):
            PacketBirthConfig(
                enable=True,
                from_iter=1,
                until_iter=10,
                flow_init=True,
                flow_source="swapped",
            ).validate()

    def test_flags_round_trip_through_the_namespace_and_back(self):
        opt = _stub_opt(
            packet_birth_flow_init=True,
            packet_birth_flow_source=FLOW_SOURCE_CAMERA_SWAPPED,
        )
        cfg = PacketBirthConfig.from_namespace(opt, densify_until_iter=10)
        self.assertTrue(cfg.flow_init)
        self.assertEqual(cfg.flow_source, FLOW_SOURCE_CAMERA_SWAPPED)
        payload = cfg.as_dict()
        self.assertEqual(payload["flow_init"], True)
        self.assertEqual(payload["flow_source"], FLOW_SOURCE_CAMERA_SWAPPED)
        self.assertEqual(PacketBirthConfig(**payload).validate().as_dict(), payload)

    def test_setup_carries_the_flow_assets_onto_the_state(self):
        assets = _assets({})
        state = setup_packet_birth(_stub_opt(), flow_assets=assets)
        self.assertIs(state.flow_assets, assets)
        self.assertIsNone(setup_packet_birth(_stub_opt()).flow_assets)


class VelocityRecoveryTests(unittest.TestCase):
    """Property 3."""

    def test_uniform_flow_recovers_the_analytic_world_velocity(self):
        camera = _StubCamera()
        pixels = _flat(BLOCK_PIXELS)
        _, _, depth, alpha = _event_inputs()
        points, valid = backproject_pixels(camera, depth, alpha, pixels)
        self.assertTrue(bool(valid.all()))
        flow, mask = _uniform_flow(1.5, 0.75)
        velocity, ok, counters = flow_world_velocity(
            camera, pixels, points, flow, mask, FRAME_DT
        )
        self.assertTrue(bool(ok.all()))
        self.assertEqual(counters, {"mask": 0, "nonfinite": 0, "bounds": 0, "outlier": 0})
        # Pinhole, identity rotation, camera at the origin, constant camera-z Z:
        # X = Z * ((i - cx)/f, (j - cy)/f, 1), so a pixel displacement (u, v)
        # moves the world point by Z/f * (u, v, 0).
        expected = torch.tensor(
            [
                SITE_DEPTH * 1.5 / FOCAL / FRAME_DT,
                SITE_DEPTH * 0.75 / FOCAL / FRAME_DT,
                0.0,
            ]
        )
        self.assertTrue(
            torch.allclose(velocity, expected.reshape(1, 3).expand_as(velocity), atol=1e-4)
        )

    def test_a_translated_camera_gives_the_same_velocity(self):
        camera = _StubCamera(centre=(2.0, -1.0, 3.0))
        pixels = _flat(BLOCK_PIXELS)
        _, _, depth, alpha = _event_inputs()
        points, _ = backproject_pixels(camera, depth, alpha, pixels)
        flow, mask = _uniform_flow(2.0, 0.0)
        velocity, ok, _ = flow_world_velocity(camera, pixels, points, flow, mask, FRAME_DT)
        self.assertTrue(bool(ok.all()))
        expected = torch.tensor([SITE_DEPTH * 2.0 / FOCAL / FRAME_DT, 0.0, 0.0])
        self.assertTrue(
            torch.allclose(velocity, expected.reshape(1, 3).expand_as(velocity), atol=1e-4)
        )

    def test_zero_flow_produces_zero_velocity(self):
        camera = _StubCamera()
        pixels = _flat(BLOCK_PIXELS)
        _, _, depth, alpha = _event_inputs()
        points, _ = backproject_pixels(camera, depth, alpha, pixels)
        flow, mask = _uniform_flow(0.0, 0.0)
        velocity, ok, _ = flow_world_velocity(camera, pixels, points, flow, mask, FRAME_DT)
        self.assertTrue(bool(ok.all()))
        # float32 round-trip error on a ~5-unit world point, divided by a 1/30
        # frame, is a few 1e-6 -- the tolerance is that, not slack for a bias.
        self.assertTrue(torch.allclose(velocity, torch.zeros_like(velocity), atol=1e-4))


class TimeUnitTests(unittest.TestCase):
    """The asserted trainer time unit, not an assumed 1/30."""

    def test_the_frame_time_convention_holds_for_the_n3v_grid(self):
        assert_frame_time_convention(EVENT_CAMERA, TIMESTAMP, FRAME_DT)

    def test_a_mis_scaled_frame_dt_is_refused(self):
        with self.assertRaises(ContractError):
            assert_frame_time_convention(EVENT_CAMERA, TIMESTAMP, 2.0 * FRAME_DT)

    def test_frame_zero_carries_no_spacing_information(self):
        assert_frame_time_convention("cam03_0000", 0.0, FRAME_DT)


class LoraProjectionTests(unittest.TestCase):
    """Property 4."""

    def test_solved_coefficients_reproduce_the_target_displacement(self):
        gaussians = _StubGaussians()
        basis = lora_probe_basis(gaussians, FRAME_DT)
        self.assertEqual(tuple(basis.shape), (LORA_RANK, 3))
        target = torch.tensor(
            [[0.01, -0.02, 0.03], [-0.004, 0.005, -0.006], [0.0, 0.0, 0.0]]
        )
        coefficients = solve_lora_coefficients(basis, target)
        self.assertEqual(tuple(coefficients.shape), (3, LORA_RANK))
        # The model's own map: gaussian_model.py:769.
        produced = torch.einsum(
            "nr,nrd->nd", coefficients, basis.reshape(1, LORA_RANK, 3).expand(3, -1, -1)
        )
        self.assertTrue(torch.allclose(produced, target, atol=1e-6))

    def test_the_offset_is_reproduced_through_the_real_model_path(self):
        gaussians = _StubGaussians()
        basis = lora_probe_basis(gaussians, FRAME_DT)
        target = torch.full((ROWS, 3), 0.0)
        target[:] = torch.tensor([0.008, -0.011, 0.02])
        with torch.no_grad():
            gaussians._motion_lora_coeff.copy_(solve_lora_coefficients(basis, target))
            gaussians._t.fill_(TIMESTAMP)
            offset = gaussians.get_lora_motion_offset(TIMESTAMP + FRAME_DT)
        self.assertTrue(torch.allclose(offset, target, atol=1e-6))

    def test_a_degenerate_probe_basis_is_a_contract_error(self):
        gaussians = _StubGaussians()
        with torch.no_grad():
            gaussians._motion_lora_basis.zero_()
        with self.assertRaises(ContractError):
            lora_probe_basis(gaussians, FRAME_DT)

    def test_a_non_lora_motion_model_is_refused(self):
        gaussians = _StubGaussians()
        gaussians.motion_model = "poly"
        with self.assertRaises(ContractError):
            lora_probe_basis(gaussians, FRAME_DT)


class NoOvershootTests(unittest.TestCase):
    """Property 5 -- why no output-side magnitude guard exists.

    The realized displacement is ``d @ (pinv(B) @ B)`` and ``pinv(B) @ B`` is
    the orthogonal projector onto ``B``'s row space, so an overshoot is not
    merely unlikely, it is impossible. What CAN happen is a shortfall, when the
    row space does not contain the target -- and that is exactly what
    ``flow_realized_ratio_mean`` is for.
    """

    def _rank_deficient_basis(self):
        """A probe basis whose row space is the yz-plane, so any target with an
        x component cannot be realized in full."""

        gaussians = _StubGaussians()
        with torch.no_grad():
            gaussians._motion_lora_basis[:, :, 0] = 0.0
        return lora_probe_basis(gaussians, FRAME_DT)

    def test_the_realized_displacement_never_exceeds_the_target(self):
        generator = torch.Generator().manual_seed(29)
        target = 4.0 * (torch.rand(64, 3, generator=generator) - 0.5)
        for basis in (lora_probe_basis(_StubGaussians(), FRAME_DT), self._rank_deficient_basis()):
            realized = realized_displacement(solve_lora_coefficients(basis, target), basis)
            overshoot = realized.norm(dim=1) - target.norm(dim=1)
            self.assertLessEqual(float(overshoot.max()), 1e-5)

    def test_a_full_rank_basis_realizes_the_target_exactly(self):
        basis = lora_probe_basis(_StubGaussians(), FRAME_DT)
        self.assertEqual(int(torch.linalg.matrix_rank(basis)), 3)
        target = torch.tensor([[0.01, -0.02, 0.03], [-0.004, 0.005, -0.006]])
        realized = realized_displacement(solve_lora_coefficients(basis, target), basis)
        ratio = realized.norm(dim=1) / target.norm(dim=1)
        self.assertTrue(torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4))

    def test_a_rank_deficient_basis_falls_short_by_the_projection(self):
        basis = self._rank_deficient_basis()
        self.assertEqual(int(torch.linalg.matrix_rank(basis)), 2)
        target = torch.tensor([[0.02, 0.01, 0.0]])
        realized = realized_displacement(solve_lora_coefficients(basis, target), basis)
        # The x component is unrepresentable, so only (0, 0.01, 0) survives.
        self.assertTrue(
            torch.allclose(realized, torch.tensor([[0.0, 0.01, 0.0]]), atol=1e-6)
        )
        ratio = float(realized.norm() / target.norm())
        self.assertLess(ratio, 1.0)
        self.assertAlmostEqual(ratio, 0.01 / float(torch.tensor([0.02, 0.01]).norm()), places=5)


class PerPixelFailClosedTests(unittest.TestCase):
    """Property 6."""

    def _run(self, flow, mask):
        camera = _StubCamera()
        pixels = _flat(BLOCK_PIXELS)
        _, _, depth, alpha = _event_inputs()
        points, _ = backproject_pixels(camera, depth, alpha, pixels)
        return flow_world_velocity(camera, pixels, points, flow, mask, FRAME_DT), points

    def test_a_masked_out_pixel_keeps_zero_motion(self):
        flow, mask = _uniform_flow(1.0, 1.0)
        mask[0, 0, 0] = 0.0  # (row 0, col 0) == BLOCK_PIXELS[0]
        (velocity, valid, counters), _ = self._run(flow, mask)
        self.assertEqual(counters["mask"], 1)
        self.assertEqual(counters["nonfinite"], 0)
        self.assertEqual(counters["bounds"], 0)
        self.assertFalse(bool(valid[0]))
        self.assertTrue(torch.equal(velocity[0], torch.zeros(3)))
        self.assertTrue(bool(valid[1:].all()))

    def test_a_non_finite_pixel_keeps_zero_motion(self):
        flow, mask = _uniform_flow(1.0, 1.0)
        # flow is indexed [channel, row, column]; BLOCK_PIXELS is (column, row).
        flow[0, 1, 0] = float("nan")  # (row 1, col 0) == BLOCK_PIXELS[1]
        flow[1, 2, 1] = float("inf")  # (row 2, col 1) == BLOCK_PIXELS[5]
        (velocity, valid, counters), _ = self._run(flow, mask)
        self.assertEqual(counters["nonfinite"], 2)
        self.assertEqual(counters["mask"], 0)
        self.assertEqual(counters["bounds"], 0)
        self.assertFalse(bool(valid[1]))
        self.assertFalse(bool(valid[5]))
        self.assertTrue(torch.equal(velocity[1], torch.zeros(3)))
        self.assertTrue(torch.equal(velocity[5], torch.zeros(3)))

    def test_an_out_of_raster_displacement_keeps_zero_motion(self):
        flow, mask = _uniform_flow(0.0, 0.0)
        flow[0, 0, 0] = -1.0  # (row 0, col 0) == BLOCK_PIXELS[0]: column -> -1
        flow[1, 1, 1] = float(IMAGE_SIZE)  # (row 1, col 1) == BLOCK_PIXELS[4]
        (velocity, valid, counters), _ = self._run(flow, mask)
        self.assertEqual(counters["bounds"], 2)
        self.assertEqual(counters["mask"], 0)
        self.assertEqual(counters["nonfinite"], 0)
        self.assertFalse(bool(valid[0]))
        self.assertFalse(bool(valid[4]))
        self.assertTrue(torch.equal(velocity[0], torch.zeros(3)))
        self.assertTrue(torch.equal(velocity[4], torch.zeros(3)))

    def test_failures_are_attributed_to_exactly_one_cause(self):
        flow, mask = _uniform_flow(0.0, 0.0)
        mask[0, 0, 0] = 0.0
        flow[0, 0, 0] = float("nan")
        (_, valid, counters), _ = self._run(flow, mask)
        self.assertEqual(counters["mask"], 1)
        self.assertEqual(counters["nonfinite"], 0)
        self.assertEqual(counters["bounds"], 0)
        self.assertEqual(counters["outlier"], 0)
        self.assertEqual(int((~valid).sum()), 1)

    def test_a_flow_magnitude_outlier_is_rejected(self):
        # 63 pixels at magnitude 1 and one site at 3: the 0.99 quantile lands
        # at ~1.74, so only the extreme site is cut.
        flow, mask = _uniform_flow(1.0, 0.0)
        flow[0, 0, 0] = 3.0  # (row 0, col 0) == BLOCK_PIXELS[0], still in raster
        (velocity, valid, counters), _ = self._run(flow, mask)
        self.assertEqual(counters["outlier"], 1)
        self.assertEqual(counters["mask"], 0)
        self.assertEqual(counters["nonfinite"], 0)
        self.assertEqual(counters["bounds"], 0)
        self.assertFalse(bool(valid[0]))
        self.assertTrue(torch.equal(velocity[0], torch.zeros(3)))
        self.assertTrue(bool(valid[1:].all()))

    def test_a_uniform_field_rejects_nothing(self):
        # The cut is `magnitude > quantile`, so a field whose magnitudes are all
        # equal has every site AT the quantile and loses none of them. The guard
        # can only ever remove extremes.
        flow, mask = _uniform_flow(1.0, 1.0)
        (_, valid, counters), _ = self._run(flow, mask)
        self.assertEqual(counters["outlier"], 0)
        self.assertTrue(bool(valid.all()))

    def test_the_quantile_is_taken_over_the_view_not_over_the_sites(self):
        # Rows 4-7 (32 pixels, none of them birth sites) carry magnitude 10, so
        # the VIEW's 0.99 quantile is 10 and the magnitude-3 site survives.
        # Taken over the six sites alone the quantile would be ~2.9 and that
        # same site would be cut, so this discriminates the two rules.
        flow, mask = _uniform_flow(1.0, 0.0)
        flow[0, 4:, :] = 10.0
        flow[0, 0, 0] = 3.0  # (row 0, col 0) == BLOCK_PIXELS[0]
        (_, valid, counters), _ = self._run(flow, mask)
        self.assertEqual(counters["outlier"], 0)
        self.assertTrue(bool(valid.all()))


class MissingAssetTests(unittest.TestCase):
    """Property 7 -- an asset failure is loud, never a quiet fallback to B1."""

    def _event(self, state):
        image, gt_image, depth, alpha = _event_inputs()
        torch.manual_seed(17)
        return maybe_packet_birth(
            state,
            _StubGaussians(),
            3,
            _StubCamera(),
            image,
            gt_image,
            depth,
            alpha,
            None,
        )

    def test_an_unreadable_flow_asset_raises(self):
        state = setup_packet_birth(
            _stub_opt(packet_birth_flow_init=True), flow_assets=_assets({})
        )
        with self.assertRaises(ContractError):
            self._event(state)

    def test_a_missing_provider_raises(self):
        state = setup_packet_birth(_stub_opt(packet_birth_flow_init=True))
        with self.assertRaises(ContractError):
            self._event(state)

    def test_a_provider_without_get_track_flow_is_refused_at_construction(self):
        with self.assertRaises(ContractError):
            build_flow_assets(object())

    def test_a_wrong_resolution_flow_asset_raises(self):
        flow = torch.zeros(2, IMAGE_SIZE + 1, IMAGE_SIZE)
        mask = torch.ones(1, IMAGE_SIZE + 1, IMAGE_SIZE)
        assets = _assets({EVENT_CAMERA: (flow, mask)})
        with self.assertRaises(ContractError):
            assets.get_flow(EVENT_CAMERA, (IMAGE_SIZE, IMAGE_SIZE))


class HeldOutCameraTests(unittest.TestCase):
    """Property 8 -- the hard requirement."""

    def test_reading_the_held_out_camera_is_refused(self):
        assets = _assets({})
        with self.assertRaises(ContractError):
            assets.assert_not_holdout("cam00_0037")
        with self.assertRaises(ContractError):
            assets.resolve_image_name("cam00_0037", FLOW_SOURCE_CORRECT)

    def test_the_held_out_camera_is_not_in_the_swap_roster(self):
        assets = _assets({})
        self.assertEqual(assets.holdout_ids, frozenset({0}))
        self.assertEqual(assets.train_ids, (1, 3, 5, 7))
        self.assertNotIn(0, assets.train_ids)

    def test_the_swap_never_selects_the_held_out_camera(self):
        # cam07 is the highest training id, so the wrap-around is the case that
        # could reach cam00 if held-out ids were not removed from the roster.
        assets = build_flow_assets(
            _StubFlowProvider({}),
            train_cameras=TRAIN_CAMERAS + HOLDOUT_CAMERAS,
            holdout_cameras=HOLDOUT_CAMERAS,
        )
        for name in TRAIN_CAMERAS:
            substitute = assets.swapped_image_name(name)
            self.assertNotEqual(parse_camera_name(substitute)[0], 0)
        self.assertEqual(assets.swapped_image_name("cam07_0037"), "cam01_0037")

    def test_the_event_reads_only_the_resolved_training_camera(self):
        flow, mask = _uniform_flow(1.0, 0.0)
        provider = _StubFlowProvider({"cam05_0037": (flow, mask)})
        assets = build_flow_assets(
            provider, train_cameras=TRAIN_CAMERAS, holdout_cameras=HOLDOUT_CAMERAS
        )
        state = setup_packet_birth(
            _stub_opt(
                packet_birth_flow_init=True,
                packet_birth_flow_source=FLOW_SOURCE_CAMERA_SWAPPED,
            ),
            flow_assets=assets,
        )
        image, gt_image, depth, alpha = _event_inputs()
        torch.manual_seed(17)
        record = maybe_packet_birth(
            state, _StubGaussians(), 3, _StubCamera(), image, gt_image, depth, alpha, None
        )
        self.assertEqual([name for name, _ in provider.calls], ["cam05_0037"])
        self.assertEqual(record["flow_source"], FLOW_SOURCE_CAMERA_SWAPPED)


class CameraSwapRuleTests(unittest.TestCase):
    """Property 9."""

    def test_the_rule_is_deterministic_and_never_returns_its_input(self):
        assets = _assets({})
        for name in TRAIN_CAMERAS:
            first = assets.swapped_image_name(name)
            self.assertEqual(first, assets.swapped_image_name(name))
            self.assertNotEqual(first, name)

    def test_the_rule_advances_in_sorted_order_and_wraps(self):
        assets = _assets({})
        self.assertEqual(assets.swapped_image_name("cam01_0037"), "cam03_0037")
        self.assertEqual(assets.swapped_image_name("cam03_0037"), "cam05_0037")
        self.assertEqual(assets.swapped_image_name("cam05_0037"), "cam07_0037")
        self.assertEqual(assets.swapped_image_name("cam07_0037"), "cam01_0037")

    def test_the_frame_index_is_preserved_verbatim(self):
        assets = _assets({})
        self.assertEqual(assets.swapped_image_name("cam03_0129"), "cam05_0129")

    def test_a_single_camera_roster_is_refused(self):
        assets = build_flow_assets(
            _StubFlowProvider({}), train_cameras=("cam03_0037",), holdout_cameras=()
        )
        with self.assertRaises(ContractError):
            assets.swapped_image_name("cam03_0037")

    def test_an_empty_roster_is_refused(self):
        assets = build_flow_assets(_StubFlowProvider({}))
        with self.assertRaises(ContractError):
            assets.swapped_image_name("cam03_0037")

    def test_an_unparseable_image_name_is_refused(self):
        with self.assertRaises(ContractError):
            parse_camera_name("frame_0037")


class EventIntegrationTests(unittest.TestCase):
    """The funnel the record has to make auditable."""

    def _run(self, flow, mask, kill_basis_axis=None):
        provider = _StubFlowProvider({EVENT_CAMERA: (flow, mask)})
        assets = build_flow_assets(
            provider, train_cameras=TRAIN_CAMERAS, holdout_cameras=HOLDOUT_CAMERAS
        )
        state = setup_packet_birth(
            _stub_opt(packet_birth_flow_init=True), flow_assets=assets
        )
        gaussians = _StubGaussians()
        if kill_basis_axis is not None:
            with torch.no_grad():
                gaussians._motion_lora_basis[:, :, kill_basis_axis] = 0.0
        image, gt_image, depth, alpha = _event_inputs()
        torch.manual_seed(17)
        record = maybe_packet_birth(
            state, gaussians, 3, _StubCamera(), image, gt_image, depth, alpha, None
        )
        return gaussians, record

    def test_the_record_reports_a_full_funnel(self):
        flow, mask = _uniform_flow(0.5, 0.25)
        gaussians, record = self._run(flow, mask)
        self.assertTrue(record["flow_init"])
        self.assertEqual(record["flow_source"], FLOW_SOURCE_CORRECT)
        self.assertEqual(record["flow_sites_total"], record["donors"])
        self.assertEqual(record["flow_sites_valid"], record["donors"])
        self.assertEqual(record["flow_failed_mask"], 0)
        self.assertEqual(record["flow_failed_bounds"], 0)
        self.assertEqual(record["flow_failed_nonfinite"], 0)
        self.assertEqual(record["flow_failed_outlier"], 0)
        expected_speed = SITE_DEPTH / FOCAL / FRAME_DT * float(
            torch.tensor([0.5, 0.25]).norm()
        )
        self.assertAlmostEqual(record["flow_mean_speed"], expected_speed, places=3)
        # Full-rank probe basis: the requested displacement is realized whole.
        self.assertAlmostEqual(record["flow_realized_ratio_mean"], 1.0, places=4)
        self.assertGreater(record["flow_coeff_norm_mean"], 0.0)
        self.assertGreaterEqual(
            record["flow_coeff_norm_max"], record["flow_coeff_norm_mean"]
        )

    def test_relocated_rows_receive_non_zero_motion_coefficients(self):
        flow, mask = _uniform_flow(0.5, 0.25)
        gaussians, record = self._run(flow, mask)
        relocated = torch.arange(record["donors"], dtype=torch.long)
        rows = gaussians._motion_lora_coeff.detach()[relocated]
        self.assertGreater(float(rows.abs().max()), 0.0)
        # No output-side guard exists any more, so the written norms ARE the
        # solved norms -- nothing between the solve and the transaction.
        self.assertAlmostEqual(
            float(rows.norm(dim=1).mean()), record["flow_coeff_norm_mean"], places=4
        )

    def test_an_outlier_site_is_rejected_at_the_event_level(self):
        flow, mask = _uniform_flow(0.5, 0.0)
        flow[0, 1, 1] = 4.0  # (row 1, col 1) == INTERIOR_PIXELS[0], in raster
        gaussians, record = self._run(flow, mask)
        self.assertEqual(record["flow_failed_outlier"], 1)
        self.assertEqual(record["flow_failed_mask"], 0)
        self.assertEqual(record["flow_failed_bounds"], 0)
        self.assertEqual(record["flow_failed_nonfinite"], 0)
        self.assertEqual(record["flow_sites_valid"], record["flow_sites_total"] - 1)

    def test_a_rank_deficient_basis_shows_up_in_the_realized_ratio(self):
        # The probe basis loses its x axis, so only the y component of a
        # (0.5, 0.25) pixel flow can be represented. The event still runs; the
        # shortfall is REPORTED rather than hidden.
        flow, mask = _uniform_flow(0.5, 0.25)
        gaussians, record = self._run(flow, mask, kill_basis_axis=0)
        self.assertEqual(record["flow_sites_valid"], record["donors"])
        expected = 0.25 / float(torch.tensor([0.5, 0.25]).norm())
        self.assertLess(record["flow_realized_ratio_mean"], 1.0)
        self.assertAlmostEqual(record["flow_realized_ratio_mean"], expected, places=4)

    def test_masked_out_flow_leaves_the_rows_motionless_and_says_so(self):
        flow, mask = _uniform_flow(1.0, 1.0)
        mask.zero_()
        gaussians, record = self._run(flow, mask)
        relocated = torch.arange(record["donors"], dtype=torch.long)
        self.assertEqual(record["flow_sites_valid"], 0)
        self.assertEqual(record["flow_failed_mask"], record["flow_sites_total"])
        self.assertEqual(record["flow_mean_speed"], 0.0)
        self.assertEqual(record["flow_coeff_norm_mean"], 0.0)
        self.assertEqual(record["flow_coeff_norm_max"], 0.0)
        self.assertEqual(record["flow_realized_ratio_mean"], 0.0)
        self.assertTrue(
            torch.equal(
                gaussians._motion_lora_coeff.detach()[relocated],
                torch.zeros(int(relocated.numel()), LORA_RANK),
            )
        )


if __name__ == "__main__":
    unittest.main()


class RosterFailClosedTests(unittest.TestCase):
    """The roster builder must never degrade silently to an EMPTY set.

    Regression for experiments 241/242: ``Scene.getTrainCameras()`` returns a
    ``CameraDataset`` whose ``__getitem__`` yields ``(image, camera)`` tuples
    (utils/data_utils.py:17-35), so iterating it handed ``_camera_ids``
    objects with no ``image_name``. The old code SKIPPED those, producing an
    empty roster -- which fails closed for the camera-swapped control but
    leaves ``assert_not_holdout`` INERT for the correct arm. A safety guard
    that silently protects nothing is worse than no guard.
    """

    class _Cam(object):
        def __init__(self, name):
            self.image_name = name

    def test_camera_objects_and_bare_strings_both_work(self):
        self.assertEqual(
            _camera_ids(
                [self._Cam("cam03_0007"), self._Cam("cam11_0002")]
            ),
            (3, 11),
        )
        self.assertEqual(
            _camera_ids(["cam00_0000", "cam05_0001"]), (0, 5)
        )

    def test_an_empty_roster_is_still_empty(self):
        self.assertEqual(_camera_ids([]), ())
        self.assertEqual(_camera_ids(None), ())

    def test_the_production_dataset_tuple_shape_now_raises(self):
        with self.assertRaises(ContractError) as caught:
            _camera_ids([(object(), self._Cam("cam03_0007"))])
        self.assertIn("viewpoint_stack", str(caught.exception))

    def test_a_non_empty_roster_can_never_yield_an_empty_id_set(self):
        with self.assertRaises(ContractError):
            _camera_ids([object(), object()])
