import pytest
"""Static admission of scripts/eval_n3v_gated.py and its gated N3V config.

CPU only. The script keeps every heavy import (torch, scene, renderer) inside
`main()`, so its reading rule and its precondition arithmetic are importable
and testable here without CUDA. The two torch-dependent behaviours -- the
gate-off toggle and the accessor-raises proof -- run against a stub model, so
they exercise the script's own logic rather than the rasterizer.

What is deliberately NOT tested here: that `setup_elgs` seeds correctly on a
restored cloud, and that the renderer's `elgs_active` branch behaves as read.
Both need a GPU and a real checkpoint; they are asserted by the script at
runtime instead (`verify_gate_off`, the `_pending_elgs_state` refusal, and the
precondition block).
"""

import ast
import importlib.util
import pathlib
import sys
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "eval_n3v_gated.py"
GATED_CONFIG = ROOT / "configs" / "n3v" / "elgs_local_crb300_6k.yaml"
UNGATED_CONFIG = ROOT / "configs" / "n3v" / "ivv_protocol_300f_6k.yaml"
A1_LOCAL = ROOT / "configs" / "lrv3" / "a1_local.yaml"


def _load_module():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("eval_n3v_gated", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EV = _load_module()


def _param_names():
    """Every attribute ModelParams/PipelineParams/OptimizationParams define,
    read statically (mirrors tests/test_elgs_configs.py:26-37)."""
    tree = ast.parse((ROOT / "arguments" / "__init__.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in (
            "ModelParams", "PipelineParams", "OptimizationParams"
        ):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Store):
                    if isinstance(sub.value, ast.Name) and sub.value.id == "self":
                        names.add(sub.attr.lstrip("_"))
    return names


def _script_top_level_keys():
    """The `--flag` names eval_n3v_gated.py adds itself, read statically."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if arg.value.startswith("--"):
                            keys.add(arg.value[2:])
    return keys


class FrameNameTests(unittest.TestCase):

    def test_n3v_name_yields_the_frame_not_the_camera(self):
        self.assertEqual(EV.absolute_frame_from_name("cam00_0150"), 150)
        self.assertEqual(EV.camera_id_from_name("cam00_0150"), 0)

    def test_paths_and_extensions_are_stripped(self):
        self.assertEqual(
            EV.absolute_frame_from_name("images/cam13_0299.png"), 299)
        self.assertEqual(
            EV.absolute_frame_from_name("a\\b\\cam13_0299.png"), 299)
        self.assertEqual(EV.camera_id_from_name("images/cam13_0299.png"), 13)

    def test_no_digits_is_none(self):
        self.assertIsNone(EV.absolute_frame_from_name("background"))
        self.assertIsNone(EV.camera_id_from_name("r_0001"))

    def test_matches_the_renderer_rule_statically(self):
        """`_frame_index_from_camera` takes `matches[-1]` of `findall(r"\\d+")`
        on the stem (gaussian_renderer/__init__.py:39-44). If that rule ever
        changes, this test is where the mirror is caught."""
        source = (ROOT / "gaussian_renderer" / "__init__.py").read_text(
            encoding="utf-8")
        self.assertIn('re.findall(r"\\d+", stem)', source)
        self.assertIn("return int(matches[-1])", source)


class FrameRangeTests(unittest.TestCase):

    NAMES = ["cam00_%04d" % f for f in (139, 140, 158, 187, 190, 209, 230, 231)]

    def test_range_is_inclusive_at_both_ends(self):
        picked = EV.select_frame_range(self.NAMES, 140, 230)
        self.assertEqual([f for _, f in picked],
                         [140, 158, 187, 190, 209, 230])

    def test_positions_index_the_original_list(self):
        picked = EV.select_frame_range(self.NAMES, 140, 230)
        for position, frame in picked:
            self.assertEqual(EV.absolute_frame_from_name(self.NAMES[position]),
                             frame)

    def test_output_is_sorted_by_frame_regardless_of_loader_order(self):
        shuffled = list(reversed(self.NAMES))
        picked = EV.select_frame_range(shuffled, 140, 230)
        self.assertEqual([f for _, f in picked],
                         sorted(f for _, f in picked))

    def test_names_without_digits_are_skipped_not_crashed(self):
        picked = EV.select_frame_range(["background", "cam00_0150"], 140, 230)
        self.assertEqual(picked, [(1, 150)])

    def test_inverted_range_is_refused(self):
        with self.assertRaises(ValueError):
            EV.select_frame_range(self.NAMES, 230, 140)

    def test_empty_selection_is_empty_not_an_error(self):
        self.assertEqual(EV.select_frame_range(self.NAMES, 300, 400), [])


class FilenameTests(unittest.TestCase):

    def test_absolute_frame_is_zero_padded_to_five(self):
        self.assertEqual(EV.render_filename(150), "00150.png")
        self.assertEqual(EV.render_filename(0), "00000.png")
        self.assertEqual(EV.render_filename(299), "00299.png")

    def test_it_is_the_absolute_frame_not_a_window_index(self):
        """`utils/mesh_utils.export_image` names files by loop position, so a
        140-230 window would write 00000..00090 there. This must not."""
        picked = EV.select_frame_range(["cam00_%04d" % f for f in (140, 158)],
                                       140, 230)
        self.assertEqual([EV.render_filename(f) for _, f in picked],
                         ["00140.png", "00158.png"])


class PreconditionArithmeticTests(unittest.TestCase):

    # one group, absent over model-time [5.2667, 6.3333] == frames 158..190 at
    # 1/30 s, which is the curated cut_roasted_beef occlusion window
    PROGRAM = {
        "schema_version": "adags-episode-program-v2",
        "units": "model_time_seconds",
        "membership_mode": "row_ids",
        "groups": [{"group": 3, "gaps": [[158.0 / 30.0, 190.0 / 30.0]]}],
    }

    def test_gaps_are_read_from_the_groups_block(self):
        gaps = EV.program_gaps_seconds(self.PROGRAM)
        self.assertEqual(list(gaps), [3])
        self.assertAlmostEqual(gaps[3][0][0], 158.0 / 30.0)

    def test_a_program_with_no_groups_yields_no_gaps(self):
        self.assertEqual(EV.program_gaps_seconds({"groups": []}), {})
        self.assertEqual(EV.program_gaps_seconds({}), {})

    def test_frames_inside_gaps_counts_only_the_gap_frames(self):
        gaps = EV.program_gaps_seconds(self.PROGRAM)
        frame_times = [(f, f / 30.0) for f in range(140, 231)]
        inside = EV.frames_inside_gaps(frame_times, gaps)
        self.assertEqual(inside[0], 158)
        self.assertEqual(inside[-1], 190)
        self.assertEqual(len(inside), 33)

    def test_a_range_disjoint_from_every_gap_counts_zero(self):
        """The vacuity case the script refuses on: the gate would never fire
        at any scored frame."""
        gaps = EV.program_gaps_seconds(self.PROGRAM)
        frame_times = [(f, f / 30.0) for f in range(0, 100)]
        self.assertEqual(EV.frames_inside_gaps(frame_times, gaps), [])

    def test_gap_endpoints_are_inclusive(self):
        gaps = {0: [[1.0, 2.0]]}
        self.assertEqual(
            EV.frames_inside_gaps([(30, 1.0), (60, 2.0), (61, 2.05)], gaps),
            [30, 60])

    def test_multiple_groups_union_their_gaps(self):
        gaps = {0: [[1.0, 2.0]], 1: [[5.0, 6.0]]}
        frame_times = [(30, 1.0), (100, 3.3), (150, 5.0)]
        self.assertEqual(EV.frames_inside_gaps(frame_times, gaps), [30, 150])

    def test_gap_frames_from_seconds_rounds_inward(self):
        spans = EV.gap_frames_from_seconds({3: [[5.28, 6.32]]}, 1.0 / 30.0)
        self.assertEqual(spans[3], [[159, 189]])

    def test_gap_frames_refuses_a_nonpositive_dt(self):
        with self.assertRaises(ValueError):
            EV.gap_frames_from_seconds({0: [[1.0, 2.0]]}, 0.0)


class GateOffTests(unittest.TestCase):
    """The off arm is a two-attribute edit, and it must be exactly reversible."""

    class _Stub:
        """Mirrors the accessors in scene/gaussian_model.py:249-275: both raise
        RuntimeError when the runtime is detached, which is what makes 'no
        cached mask leaks' checkable rather than assumed."""

        def __init__(self):
            self.elgs_runtime = object()
            self._elgs_local_presence = True
            self.calls = 0

        def get_elgs_gated_row_mask(self):
            if self.elgs_runtime is None:
                raise RuntimeError("EL-GS runtime is not attached")
            self.calls += 1
            return "mask"

        def get_elgs_presence(self, timestamp):
            if self.elgs_runtime is None:
                raise RuntimeError("EL-GS runtime is not attached")
            return "presence"

    def test_the_toggle_detaches_and_restores(self):
        model = self._Stub()
        runtime = model.elgs_runtime
        with EV._gate_disabled(model):
            self.assertIsNone(model.elgs_runtime)
            self.assertFalse(model._elgs_local_presence)
        self.assertIs(model.elgs_runtime, runtime)
        self.assertTrue(model._elgs_local_presence)

    def test_it_restores_on_exception(self):
        model = self._Stub()
        runtime = model.elgs_runtime
        with self.assertRaises(KeyError):
            with EV._gate_disabled(model):
                raise KeyError("boom")
        self.assertIs(model.elgs_runtime, runtime)
        self.assertTrue(model._elgs_local_presence)

    def test_verify_gate_off_proves_the_accessors_refuse(self):
        model = self._Stub()
        with EV._gate_disabled(model):
            checks = EV.verify_gate_off(model)
        self.assertTrue(checks["elgs_runtime_is_none"])
        self.assertTrue(checks["local_presence_false"])
        self.assertTrue(checks["no_presence_override"])
        self.assertTrue(checks["get_elgs_gated_row_mask_raised"])
        self.assertTrue(checks["get_elgs_presence_raised"])

    def test_verify_gate_off_fails_loudly_if_the_gate_is_still_live(self):
        """A regression guard: if a future edit forgets to null the runtime,
        this must not silently report a clean off arm."""
        model = self._Stub()
        checks = EV.verify_gate_off(model)
        self.assertFalse(checks["elgs_runtime_is_none"])
        self.assertFalse(checks["get_elgs_gated_row_mask_raised"])


class RendererContractTests(unittest.TestCase):
    """The off arm's correctness rests on three lines of the renderer. Assert
    they still say what the script's docstring claims they say."""

    SOURCE = (ROOT / "gaussian_renderer" / "__init__.py").read_text(encoding="utf-8")

    def test_elgs_active_is_driven_by_the_runtime_attribute(self):
        self.assertIn(
            'elgs_active = getattr(pc, "elgs_runtime", None) is not None',
            self.SOURCE)

    def test_elgs_local_is_driven_by_the_local_presence_attribute(self):
        self.assertIn('_elgs_local_presence', self.SOURCE)

    def test_the_inactive_branch_returns_the_plain_marginal(self):
        self.assertIn("if not elgs_active:", self.SOURCE)
        self.assertIn(
            "return pc.get_marginal_t(viewpoint_camera.timestamp)", self.SOURCE)

    def test_the_static_twin_is_only_touched_when_a_multiplier_exists(self):
        self.assertIn("if elgs_static_multiplier is not None:", self.SOURCE)


class ScriptSurfaceTests(unittest.TestCase):

    def test_it_parses(self):
        ast.parse(SCRIPT.read_text(encoding="utf-8"))

    def test_the_required_flags_exist(self):
        keys = _script_top_level_keys()
        for flag in ("config", "start_checkpoint", "program", "frame_range",
                     "out_dir", "gaussian_dim", "time_duration", "num_pts",
                     "num_pts_ratio", "rot_4d", "force_sh_3d", "batch_size",
                     "exhaust_test"):
            self.assertIn(flag, keys, flag)

    def test_every_top_level_yaml_key_has_a_home(self):
        """`_merge_config` asserts `hasattr(args, key)`, so a top-level YAML key
        the script does not declare would abort the run at merge time."""
        declared = _script_top_level_keys() | _param_names()
        for config in (GATED_CONFIG, UNGATED_CONFIG):
            document = yaml.safe_load(config.read_text(encoding="utf-8"))
            for key, value in document.items():
                if isinstance(value, dict):
                    for sub in value:
                        self.assertIn(sub, declared, "%s: %s" % (config.name, sub))
                else:
                    self.assertIn(key, declared, "%s: %s" % (config.name, key))

    def test_heavy_imports_stay_out_of_module_scope(self):
        """If torch reaches module scope these tests stop running on a machine
        without CUDA, and the static gate this file provides is lost."""
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        top_level = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.add(node.module.split(".")[0])
        for banned in ("torch", "yaml", "numpy", "scene", "gaussian_renderer",
                       "arguments", "elgs", "utils", "depth_visibility"):
            self.assertNotIn(banned, top_level, banned)

    def test_it_refuses_a_checkpoint_that_carries_elgs_state(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('getattr(gaussians, "_pending_elgs_state", None) is not None',
                      source)

    def test_it_overrides_the_program_before_extracting_optimization_params(self):
        """Order is load-bearing: `op.extract(args)` copies the attribute, so an
        override after it would seed the config's program, not `--program`."""
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertLess(source.index("args.elgs_oracle_episodes = str(args.program)"),
                        source.index("opt = op.extract(args)"))

    def test_the_precondition_is_written_before_the_render_loop(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertLess(source.index('"status": "precondition_only"'),
                        source.index("for position, frame in selected:"))


class GatedConfigTests(unittest.TestCase):

    GATED = yaml.safe_load(GATED_CONFIG.read_text(encoding="utf-8"))
    UNGATED = yaml.safe_load(UNGATED_CONFIG.read_text(encoding="utf-8"))
    A1 = yaml.safe_load(A1_LOCAL.read_text(encoding="utf-8"))

    def test_every_key_exists_on_the_argparse_surface(self):
        names = _param_names()
        for section in ("ModelParams", "PipelineParams", "OptimizationParams"):
            for key in self.GATED[section]:
                self.assertIn(key, names, key)

    def test_the_two_configs_differ_only_in_the_elgs_block(self):
        gated_opt = dict(self.GATED["OptimizationParams"])
        ungated_opt = dict(self.UNGATED["OptimizationParams"])
        added = set(gated_opt) - set(ungated_opt)
        self.assertTrue(added)
        self.assertTrue(all(k.startswith("elgs_") for k in added), sorted(added))
        for key in ungated_opt:
            self.assertEqual(gated_opt[key], ungated_opt[key], key)
        self.assertEqual(self.GATED["ModelParams"], self.UNGATED["ModelParams"])
        self.assertEqual(self.GATED["PipelineParams"], self.UNGATED["PipelineParams"])
        for key in self.UNGATED:
            if not isinstance(self.UNGATED[key], dict):
                self.assertEqual(self.GATED[key], self.UNGATED[key], key)

    def test_the_elgs_values_match_a1_local_except_the_program(self):
        gated = {k: v for k, v in self.GATED["OptimizationParams"].items()
                 if k.startswith("elgs_")}
        a1 = {k: v for k, v in self.A1["OptimizationParams"].items()
              if k.startswith("elgs_")}
        self.assertEqual(set(gated), set(a1))
        for key in a1:
            if key == "elgs_oracle_episodes":
                continue
            self.assertEqual(gated[key], a1[key], key)

    def test_the_localized_gate_preconditions_hold(self):
        opt = self.GATED["OptimizationParams"]
        self.assertIs(opt["elgs_enable"], True)
        self.assertIs(opt["elgs_local_presence"], True)
        self.assertIs(opt["elgs_rounds_enabled"], False)
        self.assertIs(opt["elgs_routing_pins_enabled"], False)
        self.assertEqual(opt["elgs_a_lr"], 0.0)
        self.assertEqual(opt["elgs_tracks_dir"], "")
        self.assertTrue(opt["elgs_oracle_episodes"])

    def test_no_numeric_sentinel_survives(self):
        """`setup_elgs` refuses -1 on five keys (trainer_hooks.py:150-153) and
        `attach_evidence` on five more (:376-379)."""
        opt = self.GATED["OptimizationParams"]
        for key in ("elgs_a_lr", "elgs_k_se", "elgs_lambda_u",
                    "elgs_candidate_cap", "elgs_confirmation_samples",
                    "elgs_beta", "elgs_tau_b", "elgs_c_cap", "elgs_r_site",
                    "elgs_binding_threshold", "elgs_kappa", "elgs_chi",
                    "elgs_mu"):
            self.assertGreaterEqual(opt[key], 0, key)

    def test_confirmation_samples_clears_the_bootstrap_floor(self):
        self.assertGreaterEqual(
            self.GATED["OptimizationParams"]["elgs_confirmation_samples"], 6)

    def test_the_slot_grid_fits_the_scene(self):
        """`len(reserved) >= n_rounds * candidate_cap * confirmation_samples`
        (trainer_hooks.py:222-232). 19 train cameras x 300 frames, ~25%
        reserved by the `(f + c) % 4 == 0` diagonal (:1471)."""
        import json as _json

        prereg = _json.loads(
            (ROOT / "configs" / "elgs" / "prereg_structural_v1.json").read_text(
                encoding="utf-8"))
        n_rounds = len(prereg["schedule"]["full"]["round_iterations"])
        opt = self.GATED["OptimizationParams"]
        needed = n_rounds * opt["elgs_candidate_cap"] * opt["elgs_confirmation_samples"]
        self.assertEqual(needed, 48)
        self.assertLess(needed, 19 * 300 // 4)

    def test_reserved_parity_is_a_noop_on_this_arm(self):
        """The documented decision: false, because
        `reserved_indices_for_parity` returns None whenever elgs_enable is set
        (trainer_hooks.py:1510-1511). The header must also say the comparator
        is therefore NOT unit-matched as written."""
        self.assertIs(
            self.GATED["OptimizationParams"]["elgs_reserved_parity"], False)
        source = (ROOT / "elgs" / "trainer_hooks.py").read_text(encoding="utf-8")
        self.assertIn('if bool(getattr(opt, "elgs_enable", False)):\n        return None',
                      source)
        header = GATED_CONFIG.read_text(encoding="utf-8")
        self.assertIn("ivv_protocol_300f_6k.yaml", header)
        self.assertIn("elgs_reserved_parity", header)
        self.assertIn("~75%", header)

    def test_the_protocol_keys_are_the_ivv_ones(self):
        opt = self.GATED["OptimizationParams"]
        self.assertEqual(opt["iterations"], 6_000)
        self.assertEqual(opt["densify_until_num_points"], 600000)
        self.assertEqual(opt["route_logit_init"], 4.0)
        self.assertEqual(self.GATED["ModelParams"]["resolution"], 1)
        self.assertIs(self.GATED["ModelParams"]["eval"], True)
        self.assertEqual(self.GATED["time_duration"], [0.0, 10.0])
        self.assertEqual(self.GATED["batch_size"], 2)


class TorchBackedTests(unittest.TestCase):
    """Guarded: these need torch but not CUDA."""

    def test_presence_zero_is_what_the_total_gate_produces(self):
        torch = __import__("pytest").importorskip("torch")
        from elgs.presence import local_presence_multipliers

        presence = torch.tensor([[0.0], [1.0], [0.5]])
        marginal = torch.tensor([[0.9], [0.9], [0.9]])
        gated = torch.tensor([[True], [True], [False]])
        dynamic, static = local_presence_multipliers(presence, marginal, gated)
        # gated row 0 is switched off on BOTH branches -- exact absence
        self.assertEqual(float(dynamic[0]), 0.0)
        self.assertEqual(float(static[0]), 0.0)
        # the ungated row keeps the substrate exactly
        self.assertEqual(float(dynamic[2]), 0.9)
        self.assertEqual(float(static[2]), 1.0)

    def test_the_mirrored_schema_constant_is_current(self):
        __import__("pytest").importorskip("torch")
        from elgs.trainer_hooks import EPISODE_PROGRAM_SCHEMA_V2

        self.assertEqual(EV.EPISODE_PROGRAM_SCHEMA_V2, EPISODE_PROGRAM_SCHEMA_V2)

    def test_infer_frame_dt_agrees_with_the_n3v_thirtieth(self):
        __import__("pytest").importorskip("torch")
        from elgs.trainer_hooks import infer_frame_dt

        self.assertAlmostEqual(
            infer_frame_dt([f / 30.0 for f in range(300)]), 1.0 / 30.0, places=9)


if __name__ == "__main__":
    unittest.main()


def test_resolve_seeding_mode_never_falls_through():
    from depth_visibility.errors import ContractError
    from scripts.eval_n3v_gated import resolve_seeding_mode

    assert resolve_seeding_mode(False, False) == "fresh"
    assert resolve_seeding_mode(True, True) == "restore"
    with pytest.raises(ContractError, match="carries elgs_state"):
        resolve_seeding_mode(True, False)
    with pytest.raises(ContractError, match="carries no elgs_state"):
        resolve_seeding_mode(False, True)
