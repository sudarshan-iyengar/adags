"""Anti-vacuity tests for CCR B1 observation-born packet birth
(research-wiki/operations/ccr-method-2026-08-20.md §3.1).

The properties under test are the ones the operator can silently get wrong:

1. disabled is a true no-op -- no state, no packet column, no parameter touch;
2. one event relocates EXACTLY the selected donor rows and nothing else;
3. the transaction is budget neutral (row counts and Adam moment shapes);
4. packet ids go only to spatial components of at least MIN_PACKET_ROWS rows,
   every smaller residue stays substrate (-1), and the column is maintained
   across prune and densification;
5. the backprojection matches an analytically known pinhole answer, including
   the alpha normalization of the rendered CAMERA-Z depth;
6. the temporal mean and temporal scale of a relocated row are the event's,
   not the donor's.

The B1-D donor gate (``packet_birth_dynamic_mask_donors``) adds four more:

7. the flag OFF path is bit-identical to the pre-gate operator, mask present
   or not;
8. with the flag ON, exactly the rows whose motion-model centre projects into
   a masked pixel are eligible -- behind-camera and out-of-frame centres are
   not -- and the ranking among them is the unchanged score rule;
9. fewer eligible rows than requested relocates all eligible ones, records the
   shortfall and never falls back to an ineligible row;
10. a missing mask skips the event whole (no relocation, no packet id).
"""

import math
import types
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from depth_visibility.capacity import CapacityBank
from depth_visibility.errors import ContractError
from scene.cameras import Camera
from scene.packet_birth import (
    MIN_PACKET_ROWS,
    PacketBirthConfig,
    PacketBirthState,
    backproject_camera_z,
    backproject_pixels,
    camera_forward_axis,
    donor_dynamic_mask_eligibility,
    ensure_packet_column,
    maybe_packet_birth,
    packet_components,
    requested_donor_count,
    sample_residual_sites,
    select_packet_donors,
    setup_packet_birth,
)
from utils.sh_utils import RGB2SH

REPO_ROOT = Path(__file__).resolve().parents[1]

ROWS = 40
DONOR_FRACTION = 0.25
IMAGE_SIZE = 8
FOCAL = 8.0
PRINCIPAL = 4.0
SITE_DEPTH = 5.0
FRAME_DT = 0.0333333333
TIMESTAMP = 1.25

#: six adjacent pixels -> one packet; three adjacent -> below MIN_PACKET_ROWS;
#: one isolated pixel -> its own singleton residue.
BLOCK_PIXELS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
SMALL_PIXELS = [(6, 6), (6, 7), (7, 6)]
LONE_PIXELS = [(0, 7)]
EVENT_PIXELS = BLOCK_PIXELS + SMALL_PIXELS + LONE_PIXELS

GT_CHANNEL_VALUES = (0.4, 0.5, 0.6)


class _StubCamera:
    """Duck-typed camera that runs the REAL Camera.get_rays."""

    get_rays = Camera.get_rays

    def __init__(self, centre=(0.0, 0.0, 0.0)):
        self.image_name = "cam00_0000"
        self.timestamp = TIMESTAMP
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
        # The same pinhole, expressed the way project_points_to_screen reads
        # it: clip = hom_world @ full_proj_transform, w = camera z, and
        # pixel = (ndc + 1) / 2 * (size - 1) recovers FOCAL * X / Z + PRINCIPAL.
        projection = torch.zeros(4, 4)
        span = float(IMAGE_SIZE - 1)
        projection[0, 0] = 2.0 * FOCAL / span
        projection[0, 2] = 2.0 * PRINCIPAL / span - 1.0
        projection[1, 1] = 2.0 * FOCAL / span
        projection[1, 2] = 2.0 * PRINCIPAL / span - 1.0
        projection[2, 2] = 1.0
        projection[3, 2] = 1.0
        # scene/cameras.py stores the TRANSPOSED projection too.
        self.projection_matrix = projection.transpose(0, 1)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix


def _expected_site(col, row, depth=SITE_DEPTH, centre=(0.0, 0.0, 0.0)):
    """Analytic pinhole backprojection for the identity-rotation stub."""

    x = (col + 0.5 - PRINCIPAL) / FOCAL
    y = (row + 0.5 - PRINCIPAL) / FOCAL
    return torch.tensor(centre, dtype=torch.float32) + depth * torch.tensor([x, y, 1.0])


def _donor_point(col, row, depth=SITE_DEPTH, centre=(0.0, 0.0, 0.0)):
    """World point that projects EXACTLY onto integer screen pixel ``(col, row)``.

    Unlike :func:`_expected_site` (which uses ``Camera.get_rays``' pixel-CENTRE
    convention) this inverts ``project_points_to_screen``, so ``round()`` on
    the projected coordinate is unambiguous.
    """

    x = (col - PRINCIPAL) * depth / FOCAL
    y = (row - PRINCIPAL) * depth / FOCAL
    return torch.tensor(centre, dtype=torch.float32) + torch.tensor([x, y, float(depth)])


def _block_mask():
    """(1, H, W) dynamic mask that is 1 exactly on ``BLOCK_PIXELS``."""

    mask = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
    for col, row in BLOCK_PIXELS:
        mask[0, row, col] = 1.0
    return mask


class _StubGaussians:
    """Minimal model exposing the point-neutral capacity substrate on CPU."""

    def __init__(self, rows=ROWS):
        generator = torch.Generator().manual_seed(3)
        self._xyz = nn.Parameter(torch.rand(rows, 3, generator=generator))
        self._features_dc = nn.Parameter(torch.rand(rows, 1, 3, generator=generator))
        self._features_rest = nn.Parameter(torch.rand(rows, 15, 3, generator=generator))
        self._scaling = nn.Parameter(torch.full((rows, 3), -2.0))
        self._rotation = nn.Parameter(torch.rand(rows, 4, generator=generator))
        # ascending opacity logits with a unit screen radius: the donor score
        # is monotone in the row index, so the donors are rows 0..k-1.
        self._opacity = nn.Parameter(torch.arange(rows, dtype=torch.float32).reshape(-1, 1))
        self._t = nn.Parameter(torch.zeros(rows, 1))
        self._scaling_t = nn.Parameter(torch.full((rows, 1), -1.0))
        self._route_logit = nn.Parameter(torch.zeros(rows, 1))
        self._motion_lora_coeff = nn.Parameter(torch.rand(rows, 8, generator=generator))
        self.max_radii2D = torch.zeros(rows)
        self.denom = torch.ones(rows, 1)
        self.xyz_gradient_accum = torch.ones(rows, 1)
        self.t_gradient_accum = torch.ones(rows, 1)
        self._capacity_stable_ids = torch.arange(rows, dtype=torch.long)
        self._capacity_generation = torch.zeros(rows, dtype=torch.long)
        self._capacity_last_reassigned = torch.zeros(rows, dtype=torch.long)
        self.route_logit_init = 4.0
        self._packet_ids = torch.empty(0, dtype=torch.long)
        # Stands in for GaussianModel's LoRA/poly motion offset so that
        # get_dynamic_xyz(t) is genuinely time-dependent (zero by default).
        self.motion_velocity = torch.zeros(rows, 3)
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

    def get_dynamic_xyz(self, timestamp):
        return self._xyz.detach() + float(timestamp) * self.motion_velocity

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


def _stub_opt(**overrides):
    values = dict(
        packet_birth_enable=True,
        packet_birth_interval=1,
        packet_birth_fraction=DONOR_FRACTION,
        packet_birth_t_sigma_frames=1.5,
        packet_birth_from_iter=1,
        packet_birth_until_iter=-1,
        densify_until_iter=10,
        motion_track_dt=FRAME_DT,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _event_inputs(pixels=EVENT_PIXELS, alpha_value=1.0):
    """Render tensors whose residual is nonzero at EXACTLY ``pixels``.

    ``torch.multinomial`` without replacement therefore has to select that
    pixel set, which makes the event's site set deterministic.
    """

    image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
    gt_image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
    for col, row in pixels:
        for channel, value in enumerate(GT_CHANNEL_VALUES):
            gt_image[channel, row, col] = value
    alpha = torch.full((1, IMAGE_SIZE, IMAGE_SIZE), float(alpha_value))
    depth = alpha * SITE_DEPTH
    return image, gt_image, depth, alpha


class DisabledPathTests(unittest.TestCase):
    def test_disabled_builds_no_state_and_materializes_no_column(self):
        self.assertIsNone(setup_packet_birth(_stub_opt(packet_birth_enable=False)))

    def test_disabled_event_is_a_complete_no_op(self):
        gaussians = _StubGaussians()
        before = {name: value.detach().clone() for name, value in gaussians._named_parameters()}
        image, gt_image, depth, alpha = _event_inputs()
        record = maybe_packet_birth(
            None, gaussians, 3, _StubCamera(), image, gt_image, depth, alpha, None
        )
        self.assertIsNone(record)
        self.assertEqual(int(gaussians._packet_ids.numel()), 0)
        for name, value in gaussians._named_parameters():
            self.assertTrue(torch.equal(value.detach(), before[name]), name)

    def test_out_of_window_iterations_never_fire(self):
        state = setup_packet_birth(_stub_opt(packet_birth_from_iter=4, packet_birth_interval=2))
        self.assertFalse(state.should_fire(3))
        self.assertFalse(state.should_fire(5))
        self.assertTrue(state.should_fire(4))
        self.assertFalse(state.should_fire(12))  # until_iter = densify_until_iter = 10


class DonorSelectionTests(unittest.TestCase):
    def test_bottom_fraction_by_opacity_times_radius(self):
        opacity = torch.tensor([[3.0], [-3.0], [0.0], [1.0]])
        radii = torch.tensor([0.0, 0.0, 0.0, 0.0])  # clamped to 1
        donors = select_packet_donors(opacity, radii, 0.5)
        self.assertEqual(donors.tolist(), [1, 2])

    def test_screen_radius_participates_in_the_ranking(self):
        opacity = torch.tensor([[3.0], [-3.0], [0.0], [1.0]])
        # a large radius promotes the otherwise-lowest-opacity row out of the
        # donor set, so the score is not opacity alone
        radii = torch.tensor([1.0, 40.0, 1.0, 1.0])
        donors = select_packet_donors(opacity, radii, 0.5)
        self.assertEqual(donors.tolist(), [2, 3])

    def test_zero_fraction_selects_nothing(self):
        opacity = torch.zeros(10, 1)
        self.assertEqual(int(select_packet_donors(opacity, torch.zeros(10), 0.0).numel()), 0)


class ResidualSamplingTests(unittest.TestCase):
    def test_sampling_is_confined_to_nonzero_residual(self):
        residual = torch.zeros(4, 4)
        residual[1, 2] = 0.5
        residual[3, 0] = 0.25
        pixels = sample_residual_sites(residual, 8)
        self.assertEqual(sorted(pixels.tolist()), [1 * 4 + 2, 3 * 4 + 0])

    def test_empty_residual_samples_nothing(self):
        self.assertEqual(int(sample_residual_sites(torch.zeros(4, 4), 3).numel()), 0)


class BackprojectionTests(unittest.TestCase):
    def test_forward_axis_of_the_identity_extrinsic(self):
        axis = camera_forward_axis(_StubCamera())
        self.assertTrue(torch.allclose(axis, torch.tensor([0.0, 0.0, 1.0]), atol=1e-6))

    def test_camera_z_backprojection_matches_the_analytic_pinhole_point(self):
        camera = _StubCamera()
        _, _, depth, alpha = _event_inputs()
        pixels = torch.tensor([row * IMAGE_SIZE + col for col, row in EVENT_PIXELS])
        points, valid = backproject_pixels(camera, depth, alpha, pixels)
        self.assertTrue(bool(valid.all()))
        for index, (col, row) in enumerate(EVENT_PIXELS):
            self.assertTrue(
                torch.allclose(points[index], _expected_site(col, row), atol=1e-5),
                "pixel {} gave {}".format((col, row), points[index]),
            )

    def test_translated_camera_shifts_every_site_by_the_centre(self):
        centre = (1.0, -2.0, 3.0)
        camera = _StubCamera(centre=centre)
        _, _, depth, alpha = _event_inputs()
        pixels = torch.tensor([2 * IMAGE_SIZE + 5])
        points, valid = backproject_pixels(camera, depth, alpha, pixels)
        self.assertTrue(bool(valid.all()))
        self.assertTrue(
            torch.allclose(points[0], _expected_site(5, 2, centre=centre), atol=1e-5)
        )

    def test_rendered_depth_is_normalized_by_alpha(self):
        """The rasterizer writes sum(z * alpha * T), not an expected z."""

        camera = _StubCamera()
        _, _, depth, alpha = _event_inputs(alpha_value=0.5)
        pixels = torch.tensor([2 * IMAGE_SIZE + 5])
        points, valid = backproject_pixels(camera, depth, alpha, pixels)
        self.assertTrue(bool(valid.all()))
        self.assertTrue(torch.allclose(points[0], _expected_site(5, 2), atol=1e-5))

    def test_uncovered_pixels_are_rejected(self):
        camera = _StubCamera()
        alpha = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        depth = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        pixels = torch.tensor([0, 5])
        _, valid = backproject_pixels(camera, depth, alpha, pixels)
        self.assertFalse(bool(valid.any()))

    def test_along_ray_conversion_uses_the_view_axis_cosine(self):
        rays_d = torch.tensor([[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]])
        forward = torch.tensor([0.0, 0.0, 1.0])
        points = backproject_camera_z(
            torch.zeros(3), rays_d, forward, torch.tensor([2.0, 2.0])
        )
        # the oblique ray must reach the SAME camera-z plane, further along
        self.assertTrue(torch.allclose(points[0], torch.tensor([0.0, 0.0, 2.0]), atol=1e-6))
        self.assertTrue(torch.allclose(points[1], torch.tensor([1.5, 0.0, 2.0]), atol=1e-6))

    def test_missing_intrinsics_fail_closed(self):
        camera = _StubCamera()
        camera.fl_x = -1.0
        _, _, depth, alpha = _event_inputs()
        with self.assertRaises(ContractError):
            backproject_pixels(camera, depth, alpha, torch.tensor([0]))


class PacketComponentTests(unittest.TestCase):
    def _sites(self, pixels):
        return torch.stack([_expected_site(col, row) for col, row in pixels])

    def test_only_components_of_at_least_four_rows_become_packets(self):
        labels, packets = packet_components(self._sites(EVENT_PIXELS))
        self.assertEqual(packets, 1)
        block = labels[: len(BLOCK_PIXELS)]
        residue = labels[len(BLOCK_PIXELS):]
        self.assertEqual(sorted(set(block.tolist())), [0])
        self.assertEqual(set(residue.tolist()), {-1})

    def test_two_separated_clusters_become_two_packets(self):
        pixels = BLOCK_PIXELS + [(6, 6), (6, 7), (7, 6), (7, 7)]
        labels, packets = packet_components(self._sites(pixels))
        self.assertEqual(packets, 2)
        self.assertEqual(sorted(set(labels[:6].tolist())), [0])
        self.assertEqual(sorted(set(labels[6:].tolist())), [1])

    def test_a_cohort_below_the_minimum_yields_no_packet(self):
        labels, packets = packet_components(self._sites(SMALL_PIXELS))
        self.assertEqual(packets, 0)
        self.assertEqual(set(labels.tolist()), {-1})
        self.assertEqual(len(SMALL_PIXELS), MIN_PACKET_ROWS - 1)


class PacketColumnTests(unittest.TestCase):
    def test_column_is_materialized_lazily_at_minus_one(self):
        gaussians = _StubGaussians()
        ids = ensure_packet_column(gaussians)
        self.assertEqual(ids.dtype, torch.long)
        self.assertEqual(ids.device.type, "cpu")
        self.assertEqual(int(ids.shape[0]), ROWS)
        self.assertEqual(set(ids.tolist()), {-1})

    def test_a_misaligned_column_fails_closed(self):
        gaussians = _StubGaussians()
        gaussians._packet_ids = torch.full((ROWS - 1,), -1, dtype=torch.long)
        with self.assertRaises(ContractError):
            ensure_packet_column(gaussians)


class PacketBirthEventTests(unittest.TestCase):
    def _run_event(self, dynamic_mask=None):
        gaussians = _StubGaussians()
        before = {name: value.detach().clone() for name, value in gaussians._named_parameters()}
        state = setup_packet_birth(_stub_opt())
        image, gt_image, depth, alpha = _event_inputs()
        record = maybe_packet_birth(
            state, gaussians, 3, _StubCamera(), image, gt_image, depth, alpha, dynamic_mask
        )
        return gaussians, state, record, before

    def test_event_relocates_exactly_the_donor_rows(self):
        gaussians, _, record, before = self._run_event()
        donors = int(DONOR_FRACTION * ROWS)
        self.assertEqual(record["donors"], donors)
        self.assertEqual(record["camera"], "cam00_0000")
        self.assertEqual(record["timestamp"], TIMESTAMP)
        for name, value in gaussians._named_parameters():
            self.assertTrue(
                torch.equal(value.detach()[donors:], before[name][donors:]),
                "non-donor rows of {} were touched".format(name),
            )
        self.assertFalse(torch.equal(gaussians._xyz.detach()[:donors], before["_xyz"][:donors]))

    def test_relocated_rows_land_on_the_sampled_sites(self):
        gaussians, _, record, _ = self._run_event()
        donors = record["donors"]
        expected = torch.stack([_expected_site(col, row) for col, row in EVENT_PIXELS])
        placed = gaussians._xyz.detach()[:donors]
        distances = torch.cdist(placed, expected)
        self.assertLess(float(distances.min(dim=1).values.max()), 1e-5)
        # every site is used exactly once (the cohort is a permutation)
        self.assertEqual(
            sorted(distances.argmin(dim=1).tolist()), list(range(len(EVENT_PIXELS)))
        )

    def test_budget_is_neutral(self):
        gaussians, _, _, before = self._run_event()
        for name, value in gaussians._named_parameters():
            self.assertEqual(value.shape, before[name].shape, name)
        self.assertEqual(int(gaussians._packet_ids.shape[0]), ROWS)
        self.assertEqual(int(gaussians.max_radii2D.shape[0]), ROWS)

    def test_packet_ids_reach_only_the_qualifying_component(self):
        gaussians, state, record, _ = self._run_event()
        donors = record["donors"]
        ids = gaussians._packet_ids
        self.assertEqual(record["packets"], 1)
        self.assertEqual(record["rows_assigned"], len(BLOCK_PIXELS))
        self.assertEqual(int((ids == 0).sum()), len(BLOCK_PIXELS))
        self.assertEqual(int((ids >= 0).sum()), len(BLOCK_PIXELS))
        self.assertEqual(set(ids[donors:].tolist()), {-1})
        self.assertEqual(state.next_packet_id, 1)

    def test_temporal_mean_and_scale_come_from_the_event(self):
        gaussians, state, record, _ = self._run_event()
        donors = record["donors"]
        self.assertTrue(
            torch.allclose(gaussians._t.detach()[:donors], torch.full((donors, 1), TIMESTAMP))
        )
        expected_log_sigma = math.log(1.5 * FRAME_DT)
        self.assertTrue(
            torch.allclose(
                gaussians._scaling_t.detach()[:donors],
                torch.full((donors, 1), expected_log_sigma),
            )
        )
        self.assertAlmostEqual(state.sigma_t, 1.5 * FRAME_DT, places=9)

    def test_appearance_opacity_rotation_and_routing_are_reset(self):
        gaussians, _, record, before = self._run_event()
        donors = record["donors"]
        expected_dc = RGB2SH(torch.tensor(GT_CHANNEL_VALUES))
        self.assertTrue(
            torch.allclose(
                gaussians._features_dc.detach()[:donors, 0, :],
                expected_dc.reshape(1, 3).expand(donors, 3),
                atol=1e-6,
            )
        )
        self.assertEqual(float(gaussians._features_rest.detach()[:donors].abs().max()), 0.0)
        self.assertTrue(
            torch.allclose(
                torch.sigmoid(gaussians._opacity.detach()[:donors]),
                torch.full((donors, 1), 0.1),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                gaussians._rotation.detach()[:donors],
                torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 4).expand(donors, 4),
            )
        )
        self.assertTrue(
            torch.allclose(
                gaussians._route_logit.detach()[:donors], torch.full((donors, 1), 4.0)
            )
        )
        self.assertEqual(
            float(gaussians._motion_lora_coeff.detach()[:donors].abs().max()), 0.0
        )
        # the donor's own footprint is kept, clamped to the dynamic median
        self.assertTrue(
            torch.allclose(gaussians._scaling.detach()[:donors], before["_scaling"][:donors])
        )

    def test_adam_moments_are_reset_for_relocated_rows_only(self):
        gaussians, _, record, _ = self._run_event()
        donors = record["donors"]
        for _, parameter in gaussians._named_parameters():
            moment = gaussians.optimizer.state[parameter]["exp_avg"]
            self.assertEqual(float(moment[:donors].abs().max()), 0.0)
            self.assertEqual(float(moment[donors:].abs().min()), 1.0)

    def test_the_dynamic_mask_restricts_the_sites_and_the_cohort(self):
        mask = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        for col, row in BLOCK_PIXELS:
            mask[0, row, col] = 1.0
        gaussians, _, record, before = self._run_event(dynamic_mask=mask)
        self.assertEqual(record["donors"], len(BLOCK_PIXELS))
        self.assertEqual(record["rows_assigned"], len(BLOCK_PIXELS))
        # rows beyond the masked cohort keep their donor state untouched
        self.assertTrue(
            torch.equal(
                gaussians._xyz.detach()[len(BLOCK_PIXELS):],
                before["_xyz"][len(BLOCK_PIXELS):],
            )
        )

    def test_a_blank_residual_reports_a_skip_and_touches_nothing(self):
        gaussians = _StubGaussians()
        before = gaussians._xyz.detach().clone()
        state = setup_packet_birth(_stub_opt())
        image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        alpha = torch.ones(1, IMAGE_SIZE, IMAGE_SIZE)
        record = maybe_packet_birth(
            state, gaussians, 3, _StubCamera(), image, image.clone(),
            alpha * SITE_DEPTH, alpha, None,
        )
        self.assertEqual(record["reason"], "no_residual_sites")
        self.assertEqual(record["donors"], 0)
        self.assertTrue(torch.equal(gaussians._xyz.detach(), before))

    def test_a_second_event_allocates_fresh_consecutive_packet_ids(self):
        gaussians, state, _, _ = self._run_event()
        image, gt_image, depth, alpha = _event_inputs()
        # relocate a disjoint donor cohort by making the first cohort the
        # highest-scoring rows
        with torch.no_grad():
            gaussians._opacity[:10] = 100.0
        record = maybe_packet_birth(
            state, gaussians, 4, _StubCamera(), image, gt_image, depth, alpha, None
        )
        self.assertEqual(record["event_id"], 1)
        self.assertEqual(record["packets"], 1)
        self.assertEqual(state.next_packet_id, 2)
        self.assertEqual(int((gaussians._packet_ids == 1).sum()), len(BLOCK_PIXELS))


class ConfigTests(unittest.TestCase):
    def test_negative_until_resolves_to_densify_until_iter(self):
        cfg = PacketBirthConfig.from_namespace(_stub_opt(), densify_until_iter=4321)
        self.assertEqual(cfg.until_iter, 4321)

    def test_invalid_configurations_fail_closed(self):
        with self.assertRaises(ContractError):
            PacketBirthConfig(enable=True, interval=0).validate()
        with self.assertRaises(ContractError):
            PacketBirthConfig(enable=True, fraction=0.0, until_iter=6000).validate()
        with self.assertRaises(ContractError):
            PacketBirthConfig(enable=True, from_iter=100, until_iter=50).validate()

    def test_state_requires_a_positive_frame_dt(self):
        cfg = PacketBirthConfig.from_namespace(_stub_opt(), densify_until_iter=10)
        with self.assertRaises(ContractError):
            PacketBirthState(cfg, frame_dt=0.0)


class RowMaintenanceSourceTests(unittest.TestCase):
    """The packet column must be maintained wherever _elgs_family_ids is.

    ``densification_postfix`` allocates its accumulators with a hard-coded
    ``device="cuda"``, so the appended-row path cannot execute in a CPU test;
    the prune path is exercised for real below, and both are guarded here at
    source level.
    """

    def _source(self):
        return (REPO_ROOT / "scene" / "gaussian_model.py").read_text(encoding="utf-8")

    def _function(self, name):
        source = self._source()
        start = source.index("def {}(".format(name))
        remainder = source[start + 1:]
        end = remainder.index("\n    def ")
        return remainder[:end]

    def test_prune_points_slices_the_packet_column(self):
        body = self._function("prune_points")
        self.assertIn("self._elgs_family_ids = self._elgs_family_ids[", body)
        self.assertIn("if self._packet_ids.numel() > 0:", body)
        self.assertIn("self._packet_ids = self._packet_ids[valid_points_mask", body)

    def test_densification_postfix_inherits_packet_ids_from_the_source_rows(self):
        body = self._function("densification_postfix")
        self.assertIn("if self._packet_ids.numel() > 0:", body)
        self.assertIn("elgs_source_indices.to(self._packet_ids.device)", body)
        self.assertIn("torch.cat([self._packet_ids, inherited_packets]", body)
        # the -1 default for rows appended outside a source context
        self.assertIn("dtype=self._packet_ids.dtype", body)

    def test_the_column_is_declared_next_to_the_family_id_column(self):
        source = self._source()
        family = source.index("self._elgs_family_ids = torch.empty(0, dtype=torch.long)")
        packet = source.index("self._packet_ids = torch.empty(0, dtype=torch.long)")
        self.assertLess(family, packet)


class _CpuPruneModel:
    """Runs the REAL GaussianModel.prune_points against CPU tensors."""

    from scene.gaussian_model import GaussianModel

    prune_points = GaussianModel.prune_points
    _prune_capacity_state = GaussianModel._prune_capacity_state
    _ensure_capacity_state = GaussianModel._ensure_capacity_state
    _capacity_device = GaussianModel._capacity_device
    get_xyz = property(lambda self: self._xyz)

    def __init__(self, rows=6):
        self.gaussian_dim = 4
        self.rot_4d = False
        self.lifecycle = None
        self.elgs_runtime = None
        self._xyz = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
        self._features_dc = torch.zeros(rows, 1, 3)
        self._features_rest = torch.zeros(rows, 15, 3)
        self._opacity = torch.zeros(rows, 1)
        self._scaling = torch.zeros(rows, 3)
        self._rotation = torch.zeros(rows, 4)
        self._t = torch.zeros(rows, 1)
        self._scaling_t = torch.zeros(rows, 1)
        self._staticness_score = torch.empty(0)
        self._motion_scaffold_attach_idx = torch.empty(0, dtype=torch.long)
        self._motion_scaffold_attach_w = torch.empty(0)
        self.xyz_gradient_accum = torch.zeros(rows, 1)
        self.t_gradient_accum = torch.zeros(rows, 1)
        self.denom = torch.zeros(rows, 1)
        self.max_radii2D = torch.zeros(rows)
        self._capacity_stable_ids = torch.arange(rows, dtype=torch.long)
        self._capacity_generation = torch.zeros(rows, dtype=torch.long)
        self._capacity_last_reassigned = torch.zeros(rows, dtype=torch.long)
        self._capacity_next_stable_id = rows
        self._packet_ids = torch.empty(0, dtype=torch.long)

    def _prune_optimizer(self, mask, static=False):
        return {
            "xyz": self._xyz[mask],
            "f_dc": self._features_dc[mask],
            "f_rest": self._features_rest[mask],
            "opacity": self._opacity[mask],
            "scaling": self._scaling[mask],
            "rotation": self._rotation[mask],
            "t": self._t[mask],
            "scaling_t": self._scaling_t[mask],
        }


class PruneMaintenanceTests(unittest.TestCase):
    def test_prune_slices_the_packet_column_consistently(self):
        model = _CpuPruneModel(rows=6)
        model._packet_ids = torch.tensor([-1, 0, 0, -1, 1, 1], dtype=torch.long)
        model.prune_points(torch.tensor([False, True, False, False, True, False]))
        self.assertEqual(int(model._xyz.shape[0]), 4)
        self.assertEqual(model._packet_ids.tolist(), [-1, 0, -1, 1])

    def test_prune_leaves_an_empty_packet_column_alone(self):
        model = _CpuPruneModel(rows=6)
        model.prune_points(torch.tensor([False, True, False, False, True, False]))
        self.assertEqual(int(model._packet_ids.numel()), 0)


INSIDE_PIXEL = (1, 1)  # (col, row); a member of BLOCK_PIXELS
OUTSIDE_PIXEL = (5, 5)  # inside the frame, outside the block mask


class DonorEligibilityTests(unittest.TestCase):
    """The B1-D gate itself: which rows the dynamic mask admits."""

    def _model(self, points):
        gaussians = _StubGaussians(rows=len(points))
        with torch.no_grad():
            gaussians._xyz.copy_(torch.stack(points))
        return gaussians

    def test_only_centres_inside_a_masked_pixel_are_eligible(self):
        gaussians = self._model(
            [_donor_point(*INSIDE_PIXEL), _donor_point(*OUTSIDE_PIXEL), _donor_point(0, 2)]
        )
        eligible = donor_dynamic_mask_eligibility(
            gaussians, _StubCamera(), _block_mask(), TIMESTAMP
        )
        # (1, 1) and (0, 2) are BLOCK_PIXELS; (5, 5) is not.
        self.assertEqual(eligible.tolist(), [True, False, True])

    def test_centres_behind_the_camera_are_ineligible(self):
        """The mirrored point projects onto the SAME pixel with w < 0."""

        behind = _donor_point(*INSIDE_PIXEL, depth=-SITE_DEPTH)
        gaussians = self._model([_donor_point(*INSIDE_PIXEL), behind])
        eligible = donor_dynamic_mask_eligibility(
            gaussians, _StubCamera(), _block_mask(), TIMESTAMP
        )
        self.assertEqual(eligible.tolist(), [True, False])

    def test_centres_outside_the_frame_are_ineligible(self):
        gaussians = self._model(
            [_donor_point(*INSIDE_PIXEL), _donor_point(40, 1), _donor_point(1, -30)]
        )
        eligible = donor_dynamic_mask_eligibility(
            gaussians, _StubCamera(), _block_mask(), TIMESTAMP
        )
        self.assertEqual(eligible.tolist(), [True, False, False])

    def test_eligibility_follows_the_motion_model_not_the_canonical_xyz(self):
        outside = _donor_point(*OUTSIDE_PIXEL)
        inside = _donor_point(*INSIDE_PIXEL)
        gaussians = self._model([outside, inside])
        with torch.no_grad():
            # row 0 drifts INTO the mask by TIMESTAMP; row 1 drifts OUT of it.
            gaussians.motion_velocity[0] = (inside - outside) / TIMESTAMP
            gaussians.motion_velocity[1] = (outside - inside) / TIMESTAMP
        camera, mask = _StubCamera(), _block_mask()
        self.assertEqual(
            donor_dynamic_mask_eligibility(gaussians, camera, mask, TIMESTAMP).tolist(),
            [True, False],
        )
        # at t = 0 the canonical positions decide, and the verdict flips
        self.assertEqual(
            donor_dynamic_mask_eligibility(gaussians, camera, mask, 0.0).tolist(),
            [False, True],
        )

    def test_the_threshold_is_strictly_above_one_half(self):
        gaussians = self._model([_donor_point(*INSIDE_PIXEL)])
        col, row = INSIDE_PIXEL
        camera = _StubCamera()
        mask = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        mask[0, row, col] = 0.5
        self.assertEqual(
            donor_dynamic_mask_eligibility(gaussians, camera, mask, TIMESTAMP).tolist(),
            [False],
        )
        mask[0, row, col] = 0.51
        self.assertEqual(
            donor_dynamic_mask_eligibility(gaussians, camera, mask, TIMESTAMP).tolist(),
            [True],
        )

    def test_a_model_without_a_motion_position_fails_closed(self):
        with self.assertRaises(ContractError):
            donor_dynamic_mask_eligibility(
                types.SimpleNamespace(), _StubCamera(), _block_mask(), TIMESTAMP
            )


class EligibleDonorRankingTests(unittest.TestCase):
    """The ranking rule is unchanged; the gate only removes candidates."""

    OPACITY = torch.tensor([[3.0], [-3.0], [0.0], [1.0], [-1.0], [2.0]])
    RADII = torch.ones(6)

    def test_no_eligibility_mask_reproduces_the_original_cohort(self):
        expected = select_packet_donors(self.OPACITY, self.RADII, 0.5)
        explicit = select_packet_donors(self.OPACITY, self.RADII, 0.5, eligible=None)
        self.assertTrue(torch.equal(expected, explicit))
        self.assertEqual(expected.tolist(), [1, 4, 2])

    def test_ranking_among_eligible_rows_is_the_same_score_rule(self):
        eligible = torch.tensor([True, False, True, True, False, True])
        donors = select_packet_donors(self.OPACITY, self.RADII, 0.5, eligible=eligible)
        # ineligible rows 1 and 4 (the two lowest scores) are skipped entirely
        self.assertEqual(donors.tolist(), [2, 3, 5])

    def test_fewer_eligible_than_requested_returns_all_eligible(self):
        eligible = torch.tensor([False, False, True, True, False, False])
        donors = select_packet_donors(self.OPACITY, self.RADII, 0.5, eligible=eligible)
        self.assertEqual(requested_donor_count(6, 0.5), 3)
        self.assertEqual(donors.tolist(), [2, 3])

    def test_no_eligible_row_selects_nothing(self):
        donors = select_packet_donors(
            self.OPACITY, self.RADII, 0.5, eligible=torch.zeros(6, dtype=torch.bool)
        )
        self.assertEqual(int(donors.numel()), 0)


class DynamicMaskDonorEventTests(unittest.TestCase):
    """End-to-end events with the B1-D flag on and off."""

    def _run(self, placements, dynamic_mask, flag=True, iteration=3):
        """``placements``: {row -> (col, row_px)} screen targets; rest OUTSIDE."""

        gaussians = _StubGaussians()
        with torch.no_grad():
            gaussians._xyz.copy_(
                _donor_point(*OUTSIDE_PIXEL).reshape(1, 3).expand(ROWS, 3)
            )
            for index, pixel in placements.items():
                gaussians._xyz[index] = _donor_point(*pixel)
        before = {name: value.detach().clone() for name, value in gaussians._named_parameters()}
        state = setup_packet_birth(
            _stub_opt(packet_birth_dynamic_mask_donors=flag)
        )
        image, gt_image, depth, alpha = _event_inputs()
        record = maybe_packet_birth(
            state, gaussians, iteration, _StubCamera(), image, gt_image,
            depth, alpha, dynamic_mask,
        )
        return gaussians, state, record, before

    def _moved_rows(self, gaussians, before):
        changed = (gaussians._xyz.detach() != before["_xyz"]).any(dim=1)
        return changed.nonzero().reshape(-1).tolist()

    # ---- (a) flag off ------------------------------------------------------

    def test_flag_off_with_a_mask_present_keeps_the_pre_change_donor_set(self):
        gaussians, state, record, before = self._run({}, _block_mask(), flag=False)
        self.assertFalse(record["donor_mask"])
        # the pre-change rule is bottom-fraction over ALL rows: 0..9, then
        # truncated to the six masked residual sites -> rows 0..5
        self.assertEqual(self._moved_rows(gaussians, before), list(range(len(BLOCK_PIXELS))))
        self.assertEqual(record["donors"], len(BLOCK_PIXELS))
        self.assertEqual(record["realized"], len(BLOCK_PIXELS))
        self.assertEqual(record["requested"], int(DONOR_FRACTION * ROWS))
        self.assertEqual(record["eligible_donors"], ROWS)
        self.assertEqual(record["shortfall"], 0)
        self.assertEqual(state.summary()["donors_shortfall_total"], 0)
        self.assertFalse(state.summary()["donor_mask"])

    def test_flag_off_ignores_the_mask_when_ranking_donors(self):
        """Every row sits OUTSIDE the mask; the flag-off cohort is unaffected."""

        gaussians, _, record, before = self._run({}, _block_mask(), flag=False)
        self.assertEqual(record["donors"], len(BLOCK_PIXELS))
        self.assertEqual(self._moved_rows(gaussians, before), list(range(len(BLOCK_PIXELS))))

    # ---- (b) flag on -------------------------------------------------------

    def test_flag_on_relocates_only_eligible_rows_in_score_order(self):
        placements = {index: INSIDE_PIXEL for index in list(range(2, 12)) + list(range(20, 26))}
        gaussians, state, record, before = self._run(placements, _block_mask())
        self.assertTrue(record["donor_mask"])
        self.assertEqual(record["eligible_donors"], 16)
        self.assertEqual(record["requested"], 10)
        self.assertEqual(record["shortfall"], 0)
        # the ten lowest-scoring ELIGIBLE rows are 2..11 (score is monotone in
        # the row index); the six masked residual sites then truncate to 2..7
        self.assertEqual(self._moved_rows(gaussians, before), list(range(2, 8)))
        self.assertEqual(record["realized"], len(BLOCK_PIXELS))
        self.assertEqual(record["donors"], len(BLOCK_PIXELS))
        # rows 0 and 1 outrank every eligible row but are never harvested
        for name, value in gaussians._named_parameters():
            self.assertTrue(torch.equal(value.detach()[:2], before[name][:2]), name)
        summary = state.summary()
        self.assertEqual(summary["donors_requested_total"], 10)
        self.assertEqual(summary["donors_eligible_total"], 16)
        self.assertTrue(summary["donor_mask"])

    def test_flag_on_with_no_eligible_row_reports_no_donors_and_touches_nothing(self):
        gaussians, _, record, before = self._run({}, _block_mask())
        self.assertEqual(record["reason"], "no_donors")
        self.assertEqual(record["eligible_donors"], 0)
        self.assertEqual(record["shortfall"], 10)
        self.assertEqual(record["realized"], 0)
        for name, value in gaussians._named_parameters():
            self.assertTrue(torch.equal(value.detach(), before[name]), name)

    # ---- (c) shortfall -----------------------------------------------------

    def test_shortfall_relocates_every_eligible_row_and_no_other(self):
        eligible_rows = [0, 1, 2, 3]
        placements = {index: INSIDE_PIXEL for index in eligible_rows}
        gaussians, state, record, before = self._run(placements, _block_mask())
        self.assertEqual(record["requested"], 10)
        self.assertEqual(record["eligible_donors"], len(eligible_rows))
        self.assertEqual(record["shortfall"], 10 - len(eligible_rows))
        self.assertEqual(record["realized"], len(eligible_rows))
        self.assertEqual(record["donors"], len(eligible_rows))
        self.assertEqual(self._moved_rows(gaussians, before), eligible_rows)
        for name, value in gaussians._named_parameters():
            self.assertTrue(
                torch.equal(value.detach()[len(eligible_rows):], before[name][len(eligible_rows):]),
                "an ineligible row of {} was touched".format(name),
            )
        self.assertEqual(state.summary()["donors_shortfall_total"], 6)

    # ---- (d) missing mask --------------------------------------------------

    def test_a_missing_mask_skips_the_event_whole(self):
        gaussians, state, record, before = self._run({0: INSIDE_PIXEL}, None)
        self.assertEqual(record["reason"], "no_dynamic_mask")
        self.assertEqual(record["donors"], 0)
        self.assertEqual(record["realized"], 0)
        self.assertEqual(record["packets"], 0)
        self.assertEqual(state.next_packet_id, 0)
        self.assertEqual(state.rows_relocated_total, 0)
        # not even the lazy packet column is materialized
        self.assertEqual(int(gaussians._packet_ids.numel()), 0)
        for name, value in gaussians._named_parameters():
            self.assertTrue(torch.equal(value.detach(), before[name]), name)
        for _, parameter in gaussians._named_parameters():
            moment = gaussians.optimizer.state[parameter]["exp_avg"]
            self.assertEqual(float(moment.abs().min()), 1.0)
        self.assertEqual(state.summary()["events_skipped_no_dynamic_mask"], 1)
        self.assertEqual(len(state.events), 1)

    def test_flag_off_with_no_mask_still_runs_the_event(self):
        gaussians, _, record, before = self._run({}, None, flag=False)
        self.assertIsNone(record["reason"])
        self.assertEqual(record["donors"], len(EVENT_PIXELS))
        self.assertEqual(self._moved_rows(gaussians, before), list(range(len(EVENT_PIXELS))))


class B1DConfigTests(unittest.TestCase):
    CONFIG_DIR = REPO_ROOT / "configs" / "n3v"

    def _flat(self, name):
        config = OmegaConf.load(self.CONFIG_DIR / name)
        flat = {}

        def walk(node):
            for key in node.keys():
                value = node[key]
                if hasattr(value, "keys"):
                    walk(value)
                else:
                    flat[key] = value

        walk(config)
        return flat

    def test_b1d_is_b1_plus_exactly_one_flag(self):
        b1 = self._flat("ladder_b1_crb.yaml")
        b1d = self._flat("ladder_b1d_crb.yaml")
        self.assertEqual(set(b1d) - set(b1), {"packet_birth_dynamic_mask_donors"})
        self.assertEqual(set(b1) - set(b1d), set())
        self.assertEqual(
            {key for key in b1 if b1[key] != b1d[key]}, set()
        )
        self.assertIs(b1d["packet_birth_dynamic_mask_donors"], True)

    def test_the_flag_reaches_the_packet_birth_config(self):
        from argparse import ArgumentParser

        from arguments import OptimizationParams

        parser = ArgumentParser()
        OptimizationParams(parser)
        defaults = parser.parse_args([])
        self.assertFalse(defaults.packet_birth_dynamic_mask_donors)
        defaults.packet_birth_enable = True
        defaults.packet_birth_dynamic_mask_donors = True
        cfg = PacketBirthConfig.from_namespace(defaults, densify_until_iter=4000)
        self.assertTrue(cfg.dynamic_mask_donors)
        self.assertIn("dynamic_mask_donors", cfg.as_dict())


if __name__ == "__main__":
    unittest.main()
