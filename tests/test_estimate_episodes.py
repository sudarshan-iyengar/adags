"""Unit tests for the phase-T1 episode-boundary estimator.

Everything here runs on CPU with no compiled CUDA extension: the estimator's
decision rule, leakage guard, grouping and ablation wiring are deliberately
importable without `scene` / `gaussian_renderer` (see the note on import
weight in scripts/estimate_episodes.py). Tests that genuinely need the
rasterizer live in tests/test_estimate_episodes_render.py.

The six classes below cover, in order:

1. the anti-leakage contract (forbidden paths, getTestCameras, manifests,
   train-split camera identity);
2. the frozen decision rule on synthetic series (exact crossing, flat-series
   abstention, camera disagreement, inadmissible interval);
3. the boundary inset by exactly w;
4. MAD / hysteresis arithmetic on hand-computed inputs;
5. deterministic grouping with sub-threshold groups dropped;
6. the STRUCTURAL guarantee that no ground-truth object is reachable from the
   estimation entry point's arguments.
"""

import inspect
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from depth_visibility.errors import ContractError  # noqa: E402
from elgs.intervals import IntervalConfig  # noqa: E402
from elgs.presence import episode_presence, local_presence_multipliers  # noqa: E402
from elgs.intervals import forward  # noqa: E402
from elgs.runtime import ScheduleAnchors  # noqa: E402

import estimate_episodes as ee  # noqa: E402


# LRV3 numbers, rederived from the substrate rather than read from any oracle
# file: 60 frames at 6 fps, and the preregistered interval constants
# (w = w_m = 2 frame intervals, delta_len = delta_gap = 1 frame interval).
FRAME_DT = 1.0 / 6.0
N_FRAMES = 60


def lrv3_config():
    w = 2.0 * FRAME_DT
    return IntervalConfig(
        T=(N_FRAMES - 1) * FRAME_DT,
        w_m=2.0 * FRAME_DT,
        w=w,
        floor_len=2.0 * w + FRAME_DT,
        floor_gap=2.0 * w + FRAME_DT,
        delta_tol=0.1 * FRAME_DT,
    )


def _string_literal(node):
    """The str value of an AST string node, across Python versions.

    3.8+ parses string literals to `ast.Constant`; 3.7 -- the version in the
    admitted image -- still produces the deprecated `ast.Str`. Handling only
    `ast.Constant` would make the structural test silently vacuous there.
    """
    import ast

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if sys.version_info < (3, 8):  # ast.Str only; touching it later warns
        legacy = getattr(ast, "Str", None)
        if legacy is not None and isinstance(node, legacy):
            return node.s
    return None


def _scene_importable():
    """True only where the compiled `pointops2_cuda` extension exists."""
    try:
        import scene.packet_birth  # noqa: F401
    except Exception:
        return False
    return True


SCENE_IMPORTABLE = _scene_importable()


def step_series(present_high=1.0, absent_low=0.0, gap=(30, 56)):
    """E_G(t): high while present, low over the authored gap, high on return."""
    frames = list(range(N_FRAMES))
    values = [absent_low if gap[0] <= f <= gap[1] else present_high for f in frames]
    return frames, values


class TestAntiLeakage(unittest.TestCase):
    def test_forbidden_path_predicate(self):
        for path in (
            "/apollo/users/sri/proj_adags/data/synthetic/lrv3/gt_identity/cam02_f000.npy",
            r"D:\adags\data\synthetic\lrv3\gt_identity\cam07_f030.npy",
            "data/synthetic/lrv3/event_spec.json",
            "configs/lrv3/oracle_correct.json",
            "configs/lrv3/oracle_shift2.json",
            "configs/lrv3/oracle_wrong.json",
        ):
            self.assertTrue(ee.is_forbidden_path(path), path)
        for path in (
            "data/synthetic/lrv3/train/cam00_f000.png",
            "configs/lrv3/a0.yaml",
            "configs/elgs/prereg_structural_v1.json",
            "runs/elgs/x/chkpnt6000.pth",
        ):
            self.assertFalse(ee.is_forbidden_path(path), path)

    def test_open_guard_blocks_ground_truth_and_allows_training_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt_dir = Path(tmp) / "gt_identity"
            gt_dir.mkdir()
            forbidden = gt_dir / "cam02_f000.npy"
            forbidden.write_bytes(b"x")
            spec = Path(tmp) / "event_spec.json"
            spec.write_text("{}")
            allowed = Path(tmp) / "train_cam00_f000.png"
            allowed.write_bytes(b"x")

            guard = ee.LeakageGuard()
            with guard:
                with self.assertRaises(ee.LeakageError):
                    open(str(forbidden), "rb")
                with self.assertRaises(ee.LeakageError):
                    open(str(spec), "r")
                with open(str(allowed), "rb") as handle:
                    self.assertEqual(handle.read(), b"x")
            # the guard restores the real open on exit
            with open(str(forbidden), "rb") as handle:
                self.assertEqual(handle.read(), b"x")

    def test_get_test_cameras_is_disabled_during_estimation(self):
        scene = types.SimpleNamespace(getTestCameras=lambda: ["heldout"])
        guard = ee.LeakageGuard(scene=scene)
        with guard:
            with self.assertRaises(ee.LeakageError):
                scene.getTestCameras()
        self.assertEqual(scene.getTestCameras(), ["heldout"])

    def test_manifest_guards_fire(self):
        for key in ("event_candidate_manifest", "event_boundary_support_manifest"):
            opt = types.SimpleNamespace(
                event_candidate_manifest="", event_boundary_support_manifest="")
            setattr(opt, key, "some/event/manifest.json")
            guard = ee.LeakageGuard(opt=opt)
            with self.assertRaises(ee.LeakageError):
                guard.assert_manifests_empty()
        clean = types.SimpleNamespace(
            event_candidate_manifest="", event_boundary_support_manifest="")
        self.assertTrue(ee.LeakageGuard(opt=clean).assert_manifests_empty() is None)

    def test_train_split_identity_check_rejects_a_foreign_camera(self):
        train = [types.SimpleNamespace(image_name="cam00_f000"),
                 types.SimpleNamespace(image_name="cam01_f000")]
        held_out = types.SimpleNamespace(image_name="cam02_f000")
        guard = ee.LeakageGuard()
        guard.assert_train_only(train, train)
        with self.assertRaises(ee.LeakageError):
            guard.assert_train_only(train + [held_out], train)
        # equality is not enough: an identical-looking copy must still fail
        twin = types.SimpleNamespace(image_name="cam00_f000")
        with self.assertRaises(ee.LeakageError):
            guard.assert_train_only([twin], train)


class TestDecisionRule(unittest.TestCase):
    def test_clean_step_series_yields_the_exact_crossing_frames(self):
        frames, values = step_series()
        offset, onset, reason = ee.detect_gap(frames, values)
        self.assertIsNone(reason)
        # offset = first ABSENT frame, onset = first PRESENT frame after the gap
        self.assertEqual(offset, 30)
        self.assertEqual(onset, 57)

    def test_flat_series_abstains_via_the_contrast_test(self):
        frames = list(range(N_FRAMES))
        values = [0.7] * N_FRAMES
        self.assertFalse(ee.contrast_ok(values))
        offset, onset, reason = ee.detect_gap(frames, values)
        self.assertEqual(reason, ee.ABSTAIN_CONTRAST)
        self.assertIsNone(offset)
        self.assertIsNone(onset)

    def test_a_series_with_no_interior_gap_abstains_on_shape(self):
        """Disappears and never returns: not expressible as an interior gap.

        The absent tail is deliberately SHORT. A balanced 50/50 step has MAD =
        half its range and is rejected by the contrast test before the shape
        test is ever reached -- see
        `test_balanced_step_is_rejected_by_the_frozen_contrast_test`.
        """
        frames = list(range(N_FRAMES))
        values = [1.0 if f < 50 else 0.0 for f in frames]
        self.assertTrue(ee.contrast_ok(values))
        offset, onset, reason = ee.detect_gap(frames, values)
        self.assertEqual(reason, ee.ABSTAIN_SHAPE)

    def test_balanced_clean_step_is_admitted(self):
        """The case the SUPERSEDED whole-series-MAD rule could not express.

        A 30/30 step has whole-series MAD = range/2, so the old rule demanded
        range >= 2*range and abstained. The within-mode rule sees two fully
        populated modes with zero spread inside each, i.e. an exact step, and
        admits at ANY balance.
        """
        frames = list(range(N_FRAMES))
        for split in (5, 15, 30, 45, 55):
            values = [1.0 if f < split else 0.0 for f in frames]
            ok, detail = ee.contrast_statistics(values)
            self.assertTrue(ok, "split %d" % split)
            self.assertEqual(detail["n_high"], split)
            self.assertEqual(detail["n_low"], N_FRAMES - split)
            self.assertEqual(detail["within_mode_scale"], 0.0)
            self.assertEqual(detail["separation"], 1.0)

        # and the authored LRV3 shape still admits
        _, authored = step_series()
        self.assertTrue(ee.contrast_ok(authored))

    def test_rule_is_invariant_to_the_present_absent_ratio(self):
        """Scale/shift invariance and ratio invariance, both load-bearing.

        E_G for a static group is a positive constant plus noise; the constant
        must not change the verdict, or the test would depend on how bright a
        group is rather than on whether it varies.
        """
        frames = list(range(N_FRAMES))
        base = [1.0 if f < 40 else 0.0 for f in frames]
        shifted = [v + 17.0 for v in base]
        scaled = [v * 1e-4 for v in base]
        for series in (base, shifted, scaled):
            self.assertTrue(ee.contrast_ok(series))
        self.assertEqual(ee.detect_gap(frames, base)[2],
                         ee.detect_gap(frames, shifted)[2])

    def test_camera_disagreement_abstains(self):
        """Four cameras, each seeing the offset in a different place.

        Only the agreement rule can reject this: every individual series is a
        clean step and passes contrast on its own.
        """
        camera_offsets = {0: 30, 1: 36, 2: 42, 3: 48}
        values = {}
        for cam, off in camera_offsets.items():
            _, series = step_series(gap=(off, 56))
            for frame, value in enumerate(series):
                values[(0, cam, frame)] = value
        agreeing = self._agreement_count(values, list(camera_offsets), pooled=(30, 57))
        self.assertLess(agreeing, ee.MIN_AGREEING_CAMERAS)

        aligned = {}
        for cam in camera_offsets:
            _, series = step_series(gap=(30, 56))
            for frame, value in enumerate(series):
                aligned[(0, cam, frame)] = value
        agreeing = self._agreement_count(aligned, list(camera_offsets), pooled=(30, 57))
        self.assertGreaterEqual(agreeing, ee.MIN_AGREEING_CAMERAS)

    def _agreement_count(self, values, cameras, pooled):
        frames = list(range(N_FRAMES))
        agreeing = 0
        for cam in cameras:
            series = [values[(0, cam, f)] for f in frames]
            off, on, reason = ee.detect_gap(frames, series)
            if reason is not None:
                continue
            if (abs(off - pooled[0]) <= ee.AGREEMENT_TOLERANCE_FRAMES
                    and abs(on - pooled[1]) <= ee.AGREEMENT_TOLERANCE_FRAMES):
                agreeing += 1
        return agreeing

    def test_inadmissible_interval_raises_so_the_caller_can_abstain(self):
        config = lrv3_config()
        # floor_len is 5 frames; a return at frame 59 leaves episode 2 far too
        # short, so the interval is inexpressible and must fail closed.
        gap_start, gap_end = ee.inset_gap_seconds(30, 59, FRAME_DT, config.w)
        with self.assertRaises(ContractError):
            ee.build_gap_interval(gap_start, gap_end, config)
        # the authored-shaped boundary IS admissible, so the test above is
        # rejecting the interval and not the machinery
        ok_start, ok_end = ee.inset_gap_seconds(30, 57, FRAME_DT, config.w)
        self.assertIsNotNone(ee.build_gap_interval(ok_start, ok_end, config))

    def test_boundary_density_rule_forbids_interpolated_crossings(self):
        coarse = [0, 4, 8, 12]
        self.assertFalse(ee.boundary_is_dense(coarse, 8))
        self.assertTrue(ee.boundary_is_dense(coarse + [7], 8))


class TestBoundaryInset(unittest.TestCase):
    def test_gap_is_inset_by_exactly_w(self):
        config = lrv3_config()
        w = config.w
        offset, onset = 30, 57
        gap_start, gap_end = ee.inset_gap_seconds(offset, onset, FRAME_DT, w)
        # inset by exactly one edge half-width from the last/first PRESENT frame
        last_present_t = (offset - 1) * FRAME_DT
        first_present_t = onset * FRAME_DT
        self.assertAlmostEqual(gap_start - last_present_t, w, places=12)
        self.assertAlmostEqual(first_present_t - gap_end, w, places=12)

    def test_inset_puts_the_whole_ramp_inside_the_absence_gap(self):
        """The point of the inset: presence is exactly 1.0 on every truly
        present frame and the smoothstep ramp lands only on absent frames."""
        config = lrv3_config()
        gap_start, gap_end = ee.inset_gap_seconds(30, 57, FRAME_DT, config.w)
        state = ee.build_gap_interval(gap_start, gap_end, config)
        realization = forward(state, config)

        def presence(frame):
            t = torch.tensor(frame * FRAME_DT, dtype=torch.float32)
            return float(episode_presence(t, realization.b, realization.d,
                                          config.w).sum())

        for frame in list(range(0, 30)) + [57, 58, 59]:
            self.assertAlmostEqual(presence(frame), 1.0, places=4,
                                   msg="present frame %d" % frame)
        for frame in range(31, 56):
            self.assertAlmostEqual(presence(frame), 0.0, places=6,
                                   msg="absent frame %d" % frame)
        # the two designed ramp frames sit at 0.5 and are BOTH absent frames
        for frame in (30, 56):
            self.assertAlmostEqual(presence(frame), 0.5, places=4)


class TestMadAndHysteresisArithmetic(unittest.TestCase):
    def test_median_by_hand(self):
        self.assertEqual(ee.median([3.0, 1.0, 2.0]), 2.0)
        self.assertEqual(ee.median([4.0, 1.0, 3.0, 2.0]), 2.5)
        with self.assertRaises(ContractError):
            ee.median([])

    def test_mad_by_hand(self):
        # median 3; deviations [2,1,0,1,2]; median of those is 1
        self.assertEqual(ee.mad([1.0, 2.0, 3.0, 4.0, 5.0]), 1.0)
        # constant series: every deviation is 0
        self.assertEqual(ee.mad([2.0, 2.0, 2.0]), 0.0)

    def test_split_modes_by_hand(self):
        high, low, midpoint = ee.split_modes([0.0, 3.0, 6.0, 7.0, 10.0, 13.0])
        self.assertEqual(midpoint, 6.5)          # (13 + 0) / 2
        self.assertEqual(high, [7.0, 10.0, 13.0])
        self.assertEqual(low, [0.0, 3.0, 6.0])

    def test_contrast_test_by_hand(self):
        # exact step, both modes populated, zero within-mode spread -> admit
        ok, detail = ee.contrast_statistics([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
        self.assertTrue(ok)
        self.assertEqual(detail["within_mode_scale"], 0.0)

        # m = 6; H mean 11, L mean 1 -> sep 10; MAD(H) = MAD(L) = 1 -> s = 1
        # 10 >= 4*1 : admit
        ok, detail = ee.contrast_statistics([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        self.assertTrue(ok)
        self.assertEqual(detail["separation"], 10.0)
        self.assertEqual(detail["within_mode_scale"], 1.0)

        # m = 9; H mean 11, L mean 7 -> sep 4; s = 1; 4 >= 4*1 exactly: admit
        ok, detail = ee.contrast_statistics([6.0, 7.0, 8.0, 10.0, 11.0, 12.0])
        self.assertTrue(ok)
        self.assertEqual(detail["separation"], 4.0)
        self.assertEqual(detail["within_mode_scale"], 1.0)

        # m = 6.5; H mean 10, L mean 3 -> sep 7; MAD(H) = MAD(L) = 3 -> s = 3
        # 7 < 4*3 = 12 : reject
        ok, detail = ee.contrast_statistics([0.0, 3.0, 6.0, 7.0, 10.0, 13.0])
        self.assertFalse(ok)
        self.assertEqual(detail["separation"], 7.0)
        self.assertEqual(detail["within_mode_scale"], 3.0)

        # m = 3; H = {4, 5} has only 2 members -> below MIN_MODE_SAMPLES
        ok, detail = ee.contrast_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertFalse(ok)
        self.assertEqual(detail["n_high"], 2)
        self.assertIsNone(detail["separation"])

        # a flat series has an EMPTY high mode: the old degenerate MAD == 0
        # case is now excluded structurally, not by a patched precondition
        ok, detail = ee.contrast_statistics([5.0, 5.0, 5.0])
        self.assertFalse(ok)
        self.assertEqual(detail["n_high"], 0)
        self.assertFalse(ee.contrast_ok([]))

    def test_linear_ramp_sits_exactly_on_the_threshold(self):
        """DOCUMENTED knife-edge, pinned so it cannot drift silently.

        For any linearly spaced series the two modes are uniform, so the
        separation is half the range and each within-mode MAD is a quarter of
        it: the ratio is EXACTLY 4.0 and the inclusive `>=` admits. A smoothly
        varying, non-episodic group therefore passes the contrast test and is
        rejected only downstream, by the shape test.
        """
        ramp = [i / float(N_FRAMES - 1) for i in range(N_FRAMES)]
        ok, detail = ee.contrast_statistics(ramp)
        self.assertTrue(ok)
        ratio = detail["separation"] / detail["within_mode_scale"]
        self.assertAlmostEqual(ratio, 4.0, places=6)
        # but it has no interior gap, so the pipeline still abstains
        self.assertEqual(ee.detect_gap(list(range(N_FRAMES)), ramp)[2],
                         ee.ABSTAIN_SHAPE)

    def test_hysteresis_band_by_hand(self):
        values = [0.0, 10.0]
        low, high = ee.hysteresis_band(values)
        # midpoint 5, half range 5, band = 0.25*5 = 1.25
        self.assertAlmostEqual(low, 3.75, places=12)
        self.assertAlmostEqual(high, 6.25, places=12)

    def test_hysteresis_ignores_samples_inside_the_band(self):
        """A mid-band excursion must not create a spurious transition."""
        frames = [0, 1, 2, 3, 4, 5, 6, 7]
        values = [10.0, 10.0, 5.0, 10.0, 10.0, 0.0, 0.0, 10.0]
        transitions = ee.hysteresis_transitions(frames, values)
        self.assertEqual(transitions,
                         [("high", "low", 5), ("low", "high", 7)])

    def test_transitions_report_the_first_frame_on_the_new_side(self):
        frames, values = step_series()
        transitions = ee.hysteresis_transitions(frames, values)
        self.assertEqual([t[2] for t in transitions], [30, 57])

    def test_mismatched_lengths_fail_closed(self):
        with self.assertRaises(ContractError):
            ee.hysteresis_transitions([0, 1], [1.0])


class TestNoiseRejection(unittest.TestCase):
    """What the estimator does to a STATIC group.

    E_G for a group that contributes constantly is a positive constant plus
    per-camera noise. Both the separation and the within-mode scale are
    linear in the noise amplitude and the constant cancels in the separation,
    so a static group is exactly the pure-noise case in ratio terms whatever
    its brightness. These tests pin the MEASURED behaviour, including a
    weakness: see `test_pure_noise_contrast_margin_is_thin`.
    """

    TRIALS = 2000
    CAMERAS = 4
    SEED = 20260823

    def _noise_series(self, rng):
        return [rng.gauss(0.0, 1.0) for _ in range(N_FRAMES)]

    def test_pure_noise_contrast_margin_is_thin(self):
        """MEASURED: the contrast test alone admits ~30% of pure noise.

        The rule was amended expecting noise to land near a ratio of 2-3 and
        so be rejected at 4. Measurement says otherwise. Splitting at the
        midpoint of the RANGE (not the median) puts the two half-means near
        +-0.8 sigma, so the separation is about 1.6 sigma, while the MAD
        within a half-normal is about 0.43 sigma -- a ratio near 3.7, just
        under the threshold. The contrast test therefore does much less work
        than intended; the shape test is what actually rejects noise (see
        `test_pure_noise_is_rejected_end_to_end`).
        """
        import random

        rng = random.Random(self.SEED)
        ratios = []
        admitted = 0
        for _ in range(self.TRIALS):
            values = self._noise_series(rng)
            ok, detail = ee.contrast_statistics(values)
            admitted += int(ok)
            scale = detail["within_mode_scale"]
            if scale:
                ratios.append(detail["separation"] / scale)
        rate = admitted / float(self.TRIALS)
        self.assertGreater(len(ratios), self.TRIALS * 0.9)
        median_ratio = ee.median(ratios)
        # the margin is thin but the median does sit below the threshold
        self.assertLess(median_ratio, ee.CONTRAST_MAD_MULTIPLE)
        self.assertGreater(median_ratio, 3.0)
        # pinned band around the 100k-trial measurement of 0.3021
        self.assertGreater(rate, 0.24)
        self.assertLess(rate, 0.37)

    def test_pure_noise_is_rejected_end_to_end(self):
        """The full chain rejects noise: 0 survivors in 100k trials measured.

        Contrast admits ~30%, the shape test cuts that to ~0.02% (a noise
        series almost never produces exactly one high-low-high pair under
        hysteresis), and the 3-of-4 camera agreement removes the remainder.
        """
        import random

        rng = random.Random(self.SEED + 1)
        frames = list(range(N_FRAMES))
        survived_contrast = 0
        survived_shape = 0
        gated = 0
        for _ in range(self.TRIALS):
            per_camera = [self._noise_series(rng) for _ in range(self.CAMERAS)]
            pooled = [sum(cam[f] for cam in per_camera) / float(self.CAMERAS)
                      for f in frames]
            if not ee.contrast_ok(pooled):
                continue
            survived_contrast += 1
            offset, onset, reason = ee.detect_gap(frames, pooled)
            if reason is not None:
                continue
            survived_shape += 1
            agreeing = 0
            for cam in per_camera:
                cam_offset, cam_onset, cam_reason = ee.detect_gap(frames, cam)
                if cam_reason is not None:
                    continue
                if (abs(cam_offset - offset) <= ee.AGREEMENT_TOLERANCE_FRAMES
                        and abs(cam_onset - onset) <= ee.AGREEMENT_TOLERANCE_FRAMES):
                    agreeing += 1
            if agreeing >= ee.MIN_AGREEING_CAMERAS:
                gated += 1
        self.assertGreater(survived_contrast, 0, "contrast should admit some noise")
        self.assertLess(survived_shape, self.TRIALS * 0.01,
                        "the shape test is what rejects noise")
        self.assertEqual(gated, 0, "no static group may be gated on noise alone")

    def test_a_real_step_survives_moderate_noise(self):
        """Specificity check: the rule is not simply rejecting everything."""
        import random

        rng = random.Random(self.SEED + 2)
        frames = list(range(N_FRAMES))
        admitted = 0
        for _ in range(200):
            _, clean = step_series()
            noisy = [v + rng.gauss(0.0, 0.1) for v in clean]
            admitted += int(ee.contrast_ok(noisy))
        self.assertEqual(admitted, 200)


class TestGrouping(unittest.TestCase):
    def _cloud(self):
        """Two dense clusters plus a 3-row residue, inside a pinned bbox."""
        torch.manual_seed(0)
        corners = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        a = torch.full((20, 3), -0.9) + torch.rand(20, 3) * 0.01
        b = torch.full((20, 3), 0.9) - torch.rand(20, 3) * 0.01
        residue = torch.tensor([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0],
                                [0.0, 0.001, 0.0]])
        return torch.cat([corners, a, b, residue], dim=0)

    def test_grouping_is_deterministic(self):
        cloud = self._cloud()
        first, n_first = ee.build_voxel_groups(cloud)
        second, n_second = ee.build_voxel_groups(cloud.clone())
        self.assertEqual(n_first, n_second)
        self.assertTrue(torch.equal(first, second))

    def test_sub_threshold_groups_are_dropped_to_substrate(self):
        cloud = self._cloud()
        labels, n_groups = ee.build_voxel_groups(cloud, min_rows=4)
        # the 3-row residue at the origin is below the floor
        self.assertTrue(bool((labels[-3:] == -1).all()))
        # the two 20-row clusters survive as distinct groups
        self.assertNotEqual(int(labels[2]), -1)
        self.assertNotEqual(int(labels[22]), -1)
        self.assertNotEqual(int(labels[2]), int(labels[22]))
        # every surviving label is a compact index into [0, n_groups)
        kept = labels[labels >= 0]
        self.assertEqual(int(kept.max()) + 1, n_groups)
        for group in range(n_groups):
            self.assertGreaterEqual(int((labels == group).sum()), 4)

    def test_lowering_the_floor_admits_the_residue(self):
        cloud = self._cloud()
        _, high = ee.build_voxel_groups(cloud, min_rows=4)
        labels, low = ee.build_voxel_groups(cloud, min_rows=2)
        self.assertGreater(low, high)
        self.assertNotEqual(int(labels[-1]), -1)

    def test_empty_cloud_yields_no_groups(self):
        labels, n_groups = ee.build_voxel_groups(torch.empty(0, 3))
        self.assertEqual(n_groups, 0)
        self.assertEqual(labels.numel(), 0)


class TestAblationWiring(unittest.TestCase):
    """The ablation must be a TOTAL gate on the group and a no-op elsewhere."""

    def _runtime(self, labels, n_groups):
        gaussians = types.SimpleNamespace(_xyz=torch.zeros(labels.numel(), 3))
        config = lrv3_config()
        schedule = ScheduleAnchors(seed_iteration=2500, audit_iteration=2800,
                                   round_iterations=(3000, 4500, 6000),
                                   refit_until=10000)
        ablation = ee.AblationRuntime(gaussians, labels, n_groups, config, schedule)
        return gaussians, ablation

    def test_attached_group_is_totally_gated_and_others_are_untouched(self):
        labels = torch.tensor([-1, 0, 0, 1, 1, -1])
        gaussians, ablation = self._runtime(labels, 2)
        ablation.attach(0)

        runtime = gaussians.elgs_runtime
        ids = gaussians._elgs_family_ids
        presence = runtime.presence_multiplier(
            ids, 3.0, overrides=gaussians._elgs_presence_override)
        gated = runtime.gated_row_mask(ids)
        marginal = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]])
        dynamic, static = local_presence_multipliers(presence, marginal, gated)

        self.assertTrue(gaussians._elgs_local_presence)
        self.assertEqual(gated.flatten().tolist(),
                         [False, True, True, False, False, False])
        # group 0's rows: zero through BOTH branches
        self.assertEqual(dynamic.flatten().tolist()[1:3], [0.0, 0.0])
        self.assertEqual(static.flatten().tolist()[1:3], [0.0, 0.0])
        # every other row: the ordinary marginal and an unmodulated static twin
        for index in (0, 3, 4, 5):
            self.assertAlmostEqual(float(dynamic[index]),
                                   float(marginal[index]), places=6)
            self.assertEqual(float(static[index]), 1.0)

    def test_detach_restores_the_pure_substrate(self):
        labels = torch.tensor([-1, 0, 0, 1, 1, -1])
        gaussians, ablation = self._runtime(labels, 2)
        ablation.attach(1)
        ablation.detach()
        self.assertIsNone(gaussians.elgs_runtime)
        self.assertFalse(gaussians._elgs_local_presence)
        self.assertIsNone(gaussians._elgs_presence_override)

    def test_each_group_gates_only_its_own_rows(self):
        labels = torch.tensor([-1, 0, 0, 1, 1, -1])
        gaussians, ablation = self._runtime(labels, 2)
        for group, expected in ((0, [1, 2]), (1, [3, 4])):
            ablation.attach(group)
            gated = gaussians.elgs_runtime.gated_row_mask(gaussians._elgs_family_ids)
            self.assertEqual(gated.flatten().nonzero().flatten().tolist(), expected)

    def test_marker_interval_is_never_the_rendered_program(self):
        """The K=2 marker only makes the family gateable; the override decides."""
        config = lrv3_config()
        marker = ee.marker_interval(config)
        self.assertEqual(marker.K, 2)
        self.assertEqual(ee.AblationRuntime.EMPTY.K, 0)


class TestStructuralSeparation(unittest.TestCase):
    """No ground-truth object may be reachable from the estimation stage."""

    #: names that would indicate a scoring input leaking into estimation
    FORBIDDEN_PARAMETERS = (
        "scene", "source_path", "spec", "event_spec", "gt", "ground_truth",
        "identity", "gt_identity", "oracle", "test_cameras", "presence_frames",
        "event_object", "guard",
    )

    def test_estimation_signature_carries_no_ground_truth(self):
        names = set(inspect.signature(ee.estimate_episode_program).parameters)
        for forbidden in self.FORBIDDEN_PARAMETERS:
            self.assertNotIn(forbidden, names,
                             "%r is reachable from the estimation stage" % forbidden)
        self.assertEqual(
            names,
            {"gaussians", "dataset", "views", "frame_dt", "interval_config",
             "schedule", "pipe", "background", "height", "width",
             "coarse_stride", "n_frames", "verbose"},
        )

    def test_scoring_is_the_only_stage_that_takes_the_source_path(self):
        names = set(inspect.signature(ee.score_program).parameters)
        self.assertIn("source_path", names)

    #: Every function that runs inside the estimation stage. `main` installs
    #: the guard around exactly this call graph.
    ESTIMATION_STAGE_FUNCTIONS = (
        "estimate_episode_program", "_pooled_series", "_decide_group",
        "measure_series", "build_footprints", "build_voxel_groups",
        "voxel_grid", "freeze_program", "build_v2_program",
        "_render_error", "_dilate",
    )

    #: Attributes an estimation-stage function may never touch.
    FORBIDDEN_ATTRIBUTES = ("getTestCameras", "source_path", "presence_frames",
                            "event_object")

    def test_estimation_code_never_names_a_ground_truth_artifact(self):
        """AST check: no forbidden path literal or attribute in the stage.

        A text scan is not good enough -- the module docstring and the guard
        predicate legitimately NAME the forbidden inputs. This walks only the
        executable bodies of the estimation-stage functions, with docstrings
        stripped, so it tests what the code does rather than what it says.
        """
        import ast

        source = (REPO_ROOT / "scripts" / "estimate_episodes.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        found = {}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in self.ESTIMATION_STAGE_FUNCTIONS:
                continue
            found[node.name] = True
            body = list(node.body)
            if (body and isinstance(body[0], ast.Expr)
                    and _string_literal(body[0].value) is not None):
                body = body[1:]  # drop the docstring
            for statement in body:
                for child in ast.walk(statement):
                    literal = _string_literal(child)
                    if literal is not None:
                        self.assertFalse(
                            ee.is_forbidden_path(literal),
                            "%s contains a forbidden path literal %r"
                            % (node.name, literal))
                    if isinstance(child, ast.Attribute):
                        self.assertNotIn(
                            child.attr, self.FORBIDDEN_ATTRIBUTES,
                            "%s touches %r" % (node.name, child.attr))
                    if isinstance(child, ast.Name):
                        self.assertNotEqual(
                            child.id, "score_program",
                            "%s calls the scoring stage" % node.name)
        missing = [name for name in self.ESTIMATION_STAGE_FUNCTIONS
                   if name not in found]
        self.assertEqual(missing, [],
                         "estimation-stage functions not found: %s" % missing)

    def test_freeze_happens_before_scoring_in_main(self):
        source = (REPO_ROOT / "scripts" / "estimate_episodes.py").read_text(
            encoding="utf-8")
        main_body = source.split("def main(")[-1]
        freeze_at = main_body.index("freeze_program(")
        score_at = main_body.index("score_program(")
        self.assertLess(freeze_at, score_at,
                        "the program must be frozen and hashed before scoring")


class TestProgramFreezing(unittest.TestCase):
    def test_program_hash_is_stable_and_content_sensitive(self):
        config = lrv3_config()
        estimate = {
            "n_groups": 1,
            "decisions": [{
                "group": 0, "rows": 84, "gated": True, "offset_frame": 30,
                "onset_frame": 57, "gap_seconds": [5.0, 9.0],
                "abstain_reason": None, "agreeing_cameras": 4,
            }],
        }
        first, hash_a = ee.freeze_program(estimate, FRAME_DT, config)
        _, hash_b = ee.freeze_program(estimate, FRAME_DT, config)
        self.assertEqual(hash_a, hash_b)
        self.assertEqual(first["schema"], ee.REPORT_SCHEMA + "/program")
        estimate["decisions"][0]["onset_frame"] = 58
        _, hash_c = ee.freeze_program(estimate, FRAME_DT, config)
        self.assertNotEqual(hash_a, hash_c)

    def test_frozen_program_is_json_serialisable(self):
        config = lrv3_config()
        estimate = {"n_groups": 0, "decisions": []}
        program, _ = ee.freeze_program(estimate, FRAME_DT, config)
        self.assertIsInstance(json.dumps(program, sort_keys=True), str)


class TestSchemaV1ByteIdentity(unittest.TestCase):
    """Phase T2 must not perturb the v1 supply path by a single value.

    `tests/test_elgs_oracle_episodes.py` (not owned by this change) is the
    behavioural regression suite and still passes untouched; this pins the
    EXACT returned triple for the shipped `configs/lrv3/oracle_correct.json`
    so a future edit to the shared `_interval_from_gaps` helper cannot drift
    it silently.
    """

    #: captured from the loader BEFORE schema v2 existed
    GOLDEN_REGION = {"kind": "sphere", "centre": [0.7, 0.1, 0.35], "radius": 0.2}
    GOLDEN_A = (0.0, -0.3877655267715454, -3.332204580307007)
    GOLDEN_TAU = (0.0, 0.0)
    GOLDEN_B = (-0.3333333432674408, 9.166666030883789)
    GOLDEN_D = (5.166666507720947, 10.166666030883789)

    def test_oracle_correct_returns_the_unchanged_triple(self):
        from elgs.intervals import forward
        from elgs.trainer_hooks import load_oracle_episodes

        path = REPO_ROOT / "configs" / "lrv3" / "oracle_correct.json"
        if not path.is_file():
            self.skipTest("configs/lrv3/oracle_correct.json not present")
        config = lrv3_config()
        region, interval, tau = load_oracle_episodes(str(path), config)

        self.assertEqual(region, self.GOLDEN_REGION)
        self.assertEqual(interval.K, 2)
        self.assertEqual((interval.latch_pre, interval.latch_post), (True, True))
        self.assertEqual(interval.a.dtype, torch.float32)
        self.assertEqual(interval.a.numel(), 3)
        for got, want in zip(interval.a.tolist(), self.GOLDEN_A):
            self.assertEqual(float(got), want)
        self.assertEqual(tau, self.GOLDEN_TAU)

        realization = forward(interval, config)
        for got, want in zip(realization.b.tolist(), self.GOLDEN_B):
            self.assertEqual(float(got), want)
        for got, want in zip(realization.d.tolist(), self.GOLDEN_D):
            self.assertEqual(float(got), want)

    def test_v1_still_fails_closed_on_an_inadmissible_gap(self):
        from elgs.trainer_hooks import ORACLE_EPISODES_SCHEMA, load_oracle_episodes

        config = lrv3_config()
        payload = {
            "schema_version": ORACLE_EPISODES_SCHEMA,
            "units": "model_time_seconds",
            "region": {"kind": "sphere", "centre": [0.0, 0.0, 0.0], "radius": 0.2},
            "gaps": [[5.0, 5.0 + 0.5 * config.floor_gap]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oracle.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(Exception):
                load_oracle_episodes(str(path), config)


def _v2_payload(groups, *, membership_mode="row_ids", row_group_ids=None,
                cloud=None, spatial=None):
    from elgs.trainer_hooks import EPISODE_PROGRAM_SCHEMA_V2

    payload = {
        "schema_version": EPISODE_PROGRAM_SCHEMA_V2,
        "units": "model_time_seconds",
        "membership_mode": membership_mode,
        "groups": groups,
    }
    if cloud is not None:
        payload["cloud"] = cloud
    if spatial is not None:
        payload["spatial"] = spatial
    if row_group_ids is not None:
        payload["row_group_ids"] = list(row_group_ids)
    return payload


def _write_v2(tmp, payload, name="program_v2.json"):
    path = Path(tmp) / name
    path.write_text(json.dumps(payload))
    return str(path)


def _stub_gaussians_for_seeding(xyz):
    gaussians = types.SimpleNamespace()
    gaussians._xyz = xyz
    gaussians._elgs_family_ids = torch.empty(0, dtype=torch.long)
    gaussians._elgs_checkpoint_extras = {
        "round_bookkeeping": {}, "moment_reset_log": []}
    gaussians.optimizer = torch.optim.Adam(
        [torch.zeros(1, requires_grad=True)], lr=1e-3)
    return gaussians


def _seed_state(config, program_path, local_presence=True):
    from elgs.families import FamilyRegistry
    from elgs.runtime import ElgsRuntime, ScheduleAnchors
    from elgs.trainer_hooks import ElgsTrainerState, load_structural_prereg

    prereg = load_structural_prereg(str(REPO_ROOT / "configs" / "elgs"))
    sched = prereg["schedule"]["full"]
    schedule = ScheduleAnchors(
        seed_iteration=int(sched["seed_iteration"]),
        audit_iteration=int(sched["audit_iteration"]),
        round_iterations=tuple(int(v) for v in sched["round_iterations"]),
        refit_until=int(sched["refit_until"]),
    )
    registry = FamilyRegistry()
    runtime = ElgsRuntime(registry, config, schedule, device="cpu",
                          dtype=torch.float32)
    return ElgsTrainerState(
        runtime=runtime, bundle=None, config=config, schedule=schedule,
        sampler_params=None, prereg=prereg, frame_dt=FRAME_DT,
        oracle_episodes=str(program_path), local_presence=local_presence,
        a_lr=0.0,
    )


class TestSchemaV2Loading(unittest.TestCase):
    GAP_A = [[5.166666666666666, 9.166666666666666]]
    GAP_B = [[2.0, 4.0]]

    def test_two_groups_carry_their_own_intervals(self):
        from elgs.trainer_hooks import load_oracle_episodes

        config = lrv3_config()
        payload = _v2_payload(
            [{"group": 0, "gaps": self.GAP_A}, {"group": 3, "gaps": self.GAP_B}],
            membership_mode="spatial_voxel",
            spatial={"kind": "voxel_grid", "cells_per_axis": 2,
                     "lo": [0.0, 0.0, 0.0], "span": [1.0, 1.0, 1.0],
                     "group_cell_keys": {"0": [0], "3": [7]}},
        )
        with tempfile.TemporaryDirectory() as tmp:
            program = load_oracle_episodes(_write_v2(tmp, payload), config)
        self.assertEqual(sorted(program.intervals), [0, 3])
        self.assertEqual(program.intervals[0].K, 2)
        self.assertEqual(program.intervals[3].K, 2)
        # different gaps must produce genuinely different logit vectors
        self.assertFalse(torch.equal(program.intervals[0].a, program.intervals[3].a))
        self.assertEqual(program.taus[0], (0.0, 0.0))

    def test_heterogeneous_k_is_accepted(self):
        from elgs.trainer_hooks import load_oracle_episodes

        config = lrv3_config()
        payload = _v2_payload(
            [{"group": 0, "gaps": self.GAP_A},
             {"group": 1, "gaps": [[2.0, 3.5], [6.0, 7.5]]}],
            membership_mode="spatial_voxel",
            spatial={"kind": "voxel_grid", "cells_per_axis": 2,
                     "lo": [0.0, 0.0, 0.0], "span": [1.0, 1.0, 1.0],
                     "group_cell_keys": {"0": [0], "1": [7]}},
        )
        with tempfile.TemporaryDirectory() as tmp:
            program = load_oracle_episodes(_write_v2(tmp, payload), config)
        self.assertEqual(program.intervals[0].K, 2)
        self.assertEqual(program.intervals[1].K, 3)
        self.assertEqual(len(program.taus[1]), 3)

    def test_inadmissible_v2_interval_raises_rather_than_being_repaired(self):
        from elgs.trainer_hooks import load_oracle_episodes

        config = lrv3_config()
        short = 0.5 * config.floor_gap
        payload = _v2_payload(
            [{"group": 0, "gaps": [[5.0, 5.0 + short]]}],
            membership_mode="spatial_voxel",
            spatial={"kind": "voxel_grid", "cells_per_axis": 2,
                     "lo": [0.0, 0.0, 0.0], "span": [1.0, 1.0, 1.0],
                     "group_cell_keys": {"0": [0]}},
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ContractError):
                load_oracle_episodes(_write_v2(tmp, payload), config)

    def test_malformed_programs_fail_closed(self):
        from elgs.trainer_hooks import load_oracle_episodes

        config = lrv3_config()
        spatial = {"kind": "voxel_grid", "cells_per_axis": 2,
                   "lo": [0.0, 0.0, 0.0], "span": [1.0, 1.0, 1.0],
                   "group_cell_keys": {"0": [0]}}
        bad = [
            _v2_payload([], membership_mode="spatial_voxel", spatial=spatial),
            _v2_payload([{"group": 0, "gaps": self.GAP_A}],
                        membership_mode="nonsense", spatial=spatial),
            _v2_payload([{"group": 0, "gaps": self.GAP_A},
                         {"group": 0, "gaps": self.GAP_B}],
                        membership_mode="spatial_voxel", spatial=spatial),
            _v2_payload([{"group": -1, "gaps": self.GAP_A}],
                        membership_mode="spatial_voxel", spatial=spatial),
            # row_ids without the cloud binding
            _v2_payload([{"group": 0, "gaps": self.GAP_A}],
                        membership_mode="row_ids", row_group_ids=[0, -1]),
            # spatial_voxel without a spatial block
            _v2_payload([{"group": 0, "gaps": self.GAP_A}],
                        membership_mode="spatial_voxel"),
        ]
        for index, payload in enumerate(bad):
            payload.setdefault("units", "model_time_seconds")
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ContractError, msg="payload %d" % index):
                    load_oracle_episodes(_write_v2(tmp, payload), config)

    def test_wrong_units_are_rejected(self):
        from elgs.trainer_hooks import load_oracle_episodes

        config = lrv3_config()
        payload = _v2_payload([{"group": 0, "gaps": self.GAP_A}],
                              membership_mode="spatial_voxel",
                              spatial={"kind": "voxel_grid", "cells_per_axis": 2,
                                       "lo": [0.0] * 3, "span": [1.0] * 3,
                                       "group_cell_keys": {"0": [0]}})
        payload["units"] = "frames"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ContractError):
                load_oracle_episodes(_write_v2(tmp, payload), config)


class TestSchemaV2Membership(unittest.TestCase):
    """`resolve_v2_membership` on both modes, including the fail-closed path."""

    CLOUD = torch.tensor([[0.1, 0.1, 0.1],     # cell 0
                          [0.2, 0.2, 0.2],     # cell 0
                          [0.9, 0.9, 0.9],     # cell 7
                          [0.9, 0.1, 0.1],     # cell 4
                          [0.3, 0.3, 0.3]])    # cell 0
    SPATIAL = {"kind": "voxel_grid", "cells_per_axis": 2,
               "lo": [0.0, 0.0, 0.0], "span": [1.0, 1.0, 1.0],
               "group_cell_keys": {"0": [0], "3": [7]}}
    GROUPS = [{"group": 0, "gaps": [[5.166666666666666, 9.166666666666666]]},
              {"group": 3, "gaps": [[2.0, 4.0]]}]

    def _load(self, tmp, **kwargs):
        from elgs.trainer_hooks import load_oracle_episodes

        payload = _v2_payload(self.GROUPS, **kwargs)
        return load_oracle_episodes(_write_v2(tmp, payload), lrv3_config())

    def test_spatial_membership_assigns_by_cell_key(self):
        from elgs.trainer_hooks import resolve_v2_membership

        with tempfile.TemporaryDirectory() as tmp:
            program = self._load(tmp, membership_mode="spatial_voxel",
                                 spatial=self.SPATIAL)
        column = resolve_v2_membership(program, self.CLOUD)
        # rows 0,1,4 are in cell 0 -> group 0; row 2 in cell 7 -> group 3;
        # row 3 is in cell 4, which no group claims -> ungated
        self.assertEqual(column.tolist(), [0, 0, 3, -1, 0])

    def test_spatial_membership_leaves_rows_outside_the_bbox_ungated(self):
        """Without this, clamping would sweep distant rows into an edge cell."""
        from elgs.trainer_hooks import resolve_v2_membership

        cloud = torch.tensor([[0.1, 0.1, 0.1],      # inside, cell 0
                              [-5.0, -5.0, -5.0],   # far outside, clamps to 0
                              [9.0, 9.0, 9.0]])     # far outside, clamps to 7
        with tempfile.TemporaryDirectory() as tmp:
            program = self._load(tmp, membership_mode="spatial_voxel",
                                 spatial=self.SPATIAL)
        column = resolve_v2_membership(program, cloud)
        self.assertEqual(column.tolist(), [0, -1, -1])

    def test_row_ids_membership_is_used_verbatim_when_the_cloud_matches(self):
        from elgs.trainer_hooks import cloud_fingerprint, resolve_v2_membership

        ids = [0, 0, 3, -1, 0]
        cloud = {"n_rows": 5, "xyz_sha256": cloud_fingerprint(self.CLOUD)}
        with tempfile.TemporaryDirectory() as tmp:
            program = self._load(tmp, membership_mode="row_ids",
                                 row_group_ids=ids, cloud=cloud,
                                 spatial=self.SPATIAL)
        self.assertEqual(resolve_v2_membership(program, self.CLOUD).tolist(), ids)

    def test_row_ids_fails_closed_on_a_different_row_count(self):
        from elgs.trainer_hooks import cloud_fingerprint, resolve_v2_membership

        cloud = {"n_rows": 5, "xyz_sha256": cloud_fingerprint(self.CLOUD)}
        with tempfile.TemporaryDirectory() as tmp:
            program = self._load(tmp, membership_mode="row_ids",
                                 row_group_ids=[0, 0, 3, -1, 0], cloud=cloud,
                                 spatial=self.SPATIAL)
        fresh = torch.rand(50, 3)
        with self.assertRaises(ContractError) as caught:
            resolve_v2_membership(program, fresh)
        self.assertIn("spatial_voxel", str(caught.exception))

    def test_row_ids_fails_closed_on_a_different_cloud_of_the_same_size(self):
        """The row-count check alone would pass; the fingerprint must not."""
        from elgs.trainer_hooks import cloud_fingerprint, resolve_v2_membership

        cloud = {"n_rows": 5, "xyz_sha256": cloud_fingerprint(self.CLOUD)}
        with tempfile.TemporaryDirectory() as tmp:
            program = self._load(tmp, membership_mode="row_ids",
                                 row_group_ids=[0, 0, 3, -1, 0], cloud=cloud,
                                 spatial=self.SPATIAL)
        moved = self.CLOUD.clone()
        moved[0, 0] += 1e-3
        with self.assertRaises(ContractError):
            resolve_v2_membership(program, moved)

    def test_membership_naming_an_intervalless_group_fails_closed(self):
        from elgs.trainer_hooks import cloud_fingerprint, resolve_v2_membership

        cloud = {"n_rows": 5, "xyz_sha256": cloud_fingerprint(self.CLOUD)}
        with tempfile.TemporaryDirectory() as tmp:
            program = self._load(tmp, membership_mode="row_ids",
                                 row_group_ids=[0, 0, 3, 9, 0], cloud=cloud,
                                 spatial=self.SPATIAL)
        with self.assertRaises(ContractError):
            resolve_v2_membership(program, self.CLOUD)


class TestSchemaV2Seeding(unittest.TestCase):
    """`seed_families` on a computed program: one family per gated group."""

    CLOUD = TestSchemaV2Membership.CLOUD
    SPATIAL = TestSchemaV2Membership.SPATIAL
    GROUPS = TestSchemaV2Membership.GROUPS

    def _seed(self, tmp, local_presence=True, **kwargs):
        from elgs.trainer_hooks import seed_families

        payload = _v2_payload(self.GROUPS, **kwargs)
        path = _write_v2(tmp, payload)
        config = lrv3_config()
        state = _seed_state(config, path, local_presence=local_presence)
        gaussians = _stub_gaussians_for_seeding(self.CLOUD)
        seed_families(state, gaussians, iteration=0)
        return state, gaussians

    def test_two_groups_seed_two_families_with_distinct_interval_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, gaussians = self._seed(tmp, membership_mode="spatial_voxel",
                                          spatial=self.SPATIAL)
        registry = state.runtime.registry
        active = registry.active_ids()
        self.assertEqual(len(active), 2)
        first, second = (registry.get(f) for f in active)
        self.assertEqual(first.interval.K, 2)
        self.assertEqual(second.interval.K, 2)
        # the two families carry genuinely DIFFERENT programs
        self.assertFalse(torch.equal(first.interval.a, second.interval.a))
        self.assertEqual({first.lineage_key, second.lineage_key},
                         {"est-group-0", "est-group-3"})

    def test_ungated_rows_keep_family_id_minus_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, gaussians = self._seed(tmp, membership_mode="spatial_voxel",
                                      spatial=self.SPATIAL)
        ids = gaussians._elgs_family_ids
        self.assertEqual(int(ids.numel()), 5)
        # row 3 sits in an unclaimed cell and must stay substrate
        self.assertEqual(int(ids[3]), -1)
        self.assertEqual(int((ids < 0).sum()), 1)
        # the three cell-0 rows share one family, distinct from row 2's
        self.assertEqual(int(ids[0]), int(ids[1]))
        self.assertEqual(int(ids[0]), int(ids[4]))
        self.assertNotEqual(int(ids[0]), int(ids[2]))

    def test_gated_rows_are_gated_and_ungated_rows_are_not(self):
        """The renderer-facing consequence: only gated rows lose the marginal."""
        with tempfile.TemporaryDirectory() as tmp:
            state, gaussians = self._seed(tmp, membership_mode="spatial_voxel",
                                          spatial=self.SPATIAL)
        gated = state.runtime.gated_row_mask(gaussians._elgs_family_ids)
        self.assertEqual(gated.flatten().tolist(),
                         [True, True, True, False, True])

    def test_v2_requires_local_presence(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ContractError):
                self._seed(tmp, local_presence=False,
                           membership_mode="spatial_voxel", spatial=self.SPATIAL)

    def test_a_program_that_gates_nothing_here_fails_closed(self):
        """A silent no-op would look like a successful run that did nothing."""
        spatial = dict(self.SPATIAL)
        spatial["group_cell_keys"] = {"0": [1], "3": [2]}   # cells nothing occupies
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ContractError) as caught:
                self._seed(tmp, membership_mode="spatial_voxel", spatial=spatial)
        self.assertIn("no-op", str(caught.exception))


class TestV2Emission(unittest.TestCase):
    """The emitted artifact must round-trip and recompute identically."""

    def _cloud(self):
        torch.manual_seed(3)
        corners = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        a = torch.full((20, 3), -0.9) + torch.rand(20, 3) * 0.01
        b = torch.full((20, 3), 0.9) - torch.rand(20, 3) * 0.01
        return torch.cat([corners, a, b], dim=0)

    def _decisions(self, labels):
        groups = sorted(set(int(v) for v in labels.tolist()) - {-1})
        self.assertGreaterEqual(len(groups), 2)
        gaps = {groups[0]: [5.166666666666666, 9.166666666666666],
                groups[1]: [2.0, 4.0]}
        decisions = []
        for group in groups:
            gated = group in gaps
            decisions.append({
                "group": group,
                "rows": int((labels == group).sum()),
                "gated": gated,
                "offset_frame": 30 if gated else None,
                "onset_frame": 57 if gated else None,
                "gap_seconds": gaps.get(group),
                "abstain_reason": None if gated else ee.ABSTAIN_CONTRAST,
                "agreeing_cameras": 4 if gated else 0,
            })
        return decisions, groups[:2]

    def test_emitted_program_round_trips_through_the_loader(self):
        from elgs.trainer_hooks import load_oracle_episodes

        cloud = self._cloud()
        labels, _ = ee.build_voxel_groups(cloud)
        decisions, gated_ids = self._decisions(labels)
        config = lrv3_config()
        for mode in ("row_ids", "spatial_voxel"):
            program, digest = ee.build_v2_program(
                decisions, labels, cloud, FRAME_DT, config, mode)
            self.assertEqual(len(digest), 64)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "p.json"
                path.write_text(json.dumps(program, sort_keys=True))
                loaded = load_oracle_episodes(str(path), config)
            self.assertEqual(sorted(loaded.intervals), sorted(gated_ids))
            self.assertEqual(loaded.membership_mode, mode)
            # the spatial block is emitted in BOTH modes so one artifact can be
            # re-declared for either run protocol
            self.assertEqual(loaded.spatial["kind"], "voxel_grid")

    def test_both_membership_modes_agree_on_the_originating_cloud(self):
        """The determinism proof for the recomputation route.

        `spatial_voxel` recomputes membership from the stored absolute grid.
        On the cloud the program was emitted from it must reproduce the
        explicit `row_ids` column exactly, or the recomputation route would be
        binding different primitives than the estimate measured.
        """
        from elgs.trainer_hooks import load_oracle_episodes, resolve_v2_membership

        cloud = self._cloud()
        labels, _ = ee.build_voxel_groups(cloud)
        decisions, _ = self._decisions(labels)
        config = lrv3_config()
        columns = {}
        for mode in ("row_ids", "spatial_voxel"):
            program, _ = ee.build_v2_program(
                decisions, labels, cloud, FRAME_DT, config, mode)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "p.json"
                path.write_text(json.dumps(program, sort_keys=True))
                loaded = load_oracle_episodes(str(path), config)
            columns[mode] = resolve_v2_membership(loaded, cloud).tolist()
        self.assertEqual(columns["row_ids"], columns["spatial_voxel"])
        self.assertGreater(sum(1 for v in columns["row_ids"] if v >= 0), 0)

    def test_grouping_and_emission_are_deterministic(self):
        cloud = self._cloud()
        config = lrv3_config()
        first_labels, _ = ee.build_voxel_groups(cloud)
        second_labels, _ = ee.build_voxel_groups(cloud.clone())
        self.assertTrue(torch.equal(first_labels, second_labels))
        decisions, _ = self._decisions(first_labels)
        _, hash_a = ee.build_v2_program(decisions, first_labels, cloud,
                                        FRAME_DT, config, "row_ids")
        _, hash_b = ee.build_v2_program(decisions, second_labels, cloud.clone(),
                                        FRAME_DT, config, "row_ids")
        self.assertEqual(hash_a, hash_b)

    def test_emission_refuses_when_nothing_was_gated(self):
        cloud = self._cloud()
        labels, _ = ee.build_voxel_groups(cloud)
        decisions, _ = self._decisions(labels)
        for record in decisions:
            record["gated"] = False
        with self.assertRaises(ContractError):
            ee.build_v2_program(decisions, labels, cloud, FRAME_DT,
                                lrv3_config(), "row_ids")

    def test_row_ids_column_marks_only_gated_groups(self):
        cloud = self._cloud()
        labels, n_groups = ee.build_voxel_groups(cloud)
        decisions, gated_ids = self._decisions(labels)
        program, _ = ee.build_v2_program(decisions, labels, cloud, FRAME_DT,
                                         lrv3_config(), "row_ids")
        column = program["row_group_ids"]
        self.assertEqual(len(column), int(cloud.shape[0]))
        self.assertEqual(set(v for v in column if v >= 0), set(gated_ids))


@unittest.skipUnless(
    SCENE_IMPORTABLE,
    "requires the compiled pointops2_cuda extension (admitted image only)")
class TestInsideTheAdmittedImage(unittest.TestCase):
    """Interface checks against the modules the estimator imports lazily.

    These are CONTAINER-ONLY: importing `scene` pulls `pointops2_cuda`, which
    cannot be built on a workstation. They verify the surfaces the estimation
    stage depends on, not numerical behaviour -- an end-to-end measurement
    needs a trained checkpoint and is the run itself, not a unit test.
    """

    def test_min_group_rows_is_single_sourced(self):
        from scene.packet_birth import MIN_PACKET_ROWS

        self.assertEqual(ee.MIN_GROUP_ROWS, MIN_PACKET_ROWS)
        self.assertTrue(ee.assert_min_group_rows_single_sourced())

    def test_lazy_imports_resolve(self):
        from gaussian_renderer import render
        from utils.motion_prior_utils import project_points_to_screen
        from scene.gaussian_model import GaussianModel

        self.assertTrue(callable(render))
        self.assertTrue(callable(project_points_to_screen))
        for attribute in ("get_dynamic_xyz", "get_marginal_t",
                          "get_elgs_presence", "get_elgs_gated_row_mask"):
            self.assertTrue(hasattr(GaussianModel, attribute), attribute)

    def test_model_exposes_the_attributes_the_ablation_sets(self):
        from scene.gaussian_model import GaussianModel

        source = inspect.getsource(GaussianModel.__init__)
        for attribute in ("_elgs_family_ids", "elgs_runtime"):
            self.assertIn(attribute, source)


if __name__ == "__main__":
    unittest.main()
