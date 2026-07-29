"""Static admission tests for the CSVL-VPL v2 lane configs.

Two failure modes are caught here, both before any Slurm time is spent:

1. a lane yaml key that ``main.py``'s ``recursive_merge`` cannot set, which
   fails a bare ``assert hasattr(args, key)`` several minutes into a job; and
2. a silently mislabeled lane (the L1/L2/L3/L4/L5 flag matrix is the entire
   experimental design, so a wrong flag invalidates the comparison rather than
   crashing).

The test never imports ``main`` or ``scene`` (both pull in compiled CUDA
extensions that do not load on a login node). The argparse surface is rebuilt
from ``arguments`` exactly as ``main.py`` builds it, and the extra top-level
options are read out of ``main.py`` with ``ast`` so this file cannot drift from
the real parser.
"""

import ast
import importlib.util
import sys
import unittest
from argparse import ArgumentParser
from dataclasses import fields as dataclass_fields
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from arguments import ModelParams, OptimizationParams, PipelineParams

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "n3v"
MAIN_PATH = REPO_ROOT / "main.py"
LIFECYCLE_PATH = REPO_ROOT / "scene" / "lifecycle.py"

BASE_CONFIG = "fixed_budget_lora_route0_filemask_residual_600k.yaml"
EVIDENCE_DIR = "$WORK/proj_adags/runs/phase9-depth-visibility-capacity/evidence-consensus-cut-v1"

#: The frozen lane matrix. ``None`` means "the lane must not mention the key".
LANE_MATRIX = {
    "lane_l0_route0.yaml": {
        "lifecycle_enable": None,
        "lifecycle_protection": None,
        "lifecycle_exposure": None,
        "lifecycle_birth": None,
        "lifecycle_birth_mode": None,
        "lifecycle_evidence_mode": None,
        "lifecycle_time_shift": None,
    },
    "lane_l1_internal.yaml": {
        "lifecycle_enable": True,
        "lifecycle_protection": True,
        "lifecycle_exposure": True,
        "lifecycle_birth": False,
        "lifecycle_birth_mode": "e2",
        "lifecycle_evidence_mode": "e1",
        "lifecycle_time_shift": 0,
    },
    "lane_l2_presence_vad.yaml": {
        "lifecycle_enable": True,
        "lifecycle_protection": False,
        "lifecycle_exposure": True,
        "lifecycle_birth": False,
        "lifecycle_birth_mode": "e2",
        "lifecycle_evidence_mode": "presence",
        "lifecycle_time_shift": 0,
    },
    "lane_l3_full.yaml": {
        "lifecycle_enable": True,
        "lifecycle_protection": True,
        "lifecycle_exposure": True,
        "lifecycle_birth": True,
        "lifecycle_birth_mode": "e2",
        "lifecycle_evidence_mode": "e1",
        "lifecycle_time_shift": 0,
    },
    "lane_l4_generic_capacity.yaml": {
        "lifecycle_enable": True,
        "lifecycle_protection": False,
        "lifecycle_exposure": False,
        "lifecycle_birth": True,
        "lifecycle_birth_mode": "generic",
        "lifecycle_evidence_mode": "e1",
        "lifecycle_time_shift": 0,
    },
    "lane_l5_shifted.yaml": {
        "lifecycle_enable": True,
        "lifecycle_protection": True,
        "lifecycle_exposure": True,
        "lifecycle_birth": True,
        "lifecycle_birth_mode": "e2",
        "lifecycle_evidence_mode": "e1",
        "lifecycle_time_shift": 101,
    },
    "lane_smoke.yaml": {
        "lifecycle_enable": True,
        "lifecycle_protection": True,
        "lifecycle_exposure": True,
        "lifecycle_birth": True,
        "lifecycle_birth_mode": "e2",
        "lifecycle_evidence_mode": "e1",
        "lifecycle_time_shift": 0,
    },
}

LIFECYCLE_LANES = tuple(
    name for name, flags in LANE_MATRIX.items() if flags["lifecycle_enable"] is True
)


def main_top_level_option_dests():
    """Destination names of every ``parser.add_argument`` call in ``main.py``.

    Read statically so the test uses the real parser surface without importing
    ``main`` (which imports the CUDA rasterizer).
    """

    tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"), filename=str(MAIN_PATH))
    dests = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                option = argument.value
                if option.startswith("--"):
                    dests.add(option[2:].replace("-", "_"))
    return dests


def build_trainer_namespace():
    """Rebuild ``main.py``'s parsed namespace (defaults only)."""

    parser = ArgumentParser(description="Training script parameters")
    ModelParams(parser)
    OptimizationParams(parser)
    PipelineParams(parser)
    args = parser.parse_args([])
    for dest in main_top_level_option_dests():
        if not hasattr(args, dest):
            setattr(args, dest, None)
    return args


def merge_like_main(config, args):
    """Replay ``main.py``'s ``recursive_merge`` and collect unmergeable keys."""

    missing = []

    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            if not hasattr(args, key):
                missing.append(key)
                return
            setattr(args, key, host[key])

    for key in config.keys():
        recursive_merge(key, config)
    return missing


def flatten_config(config, out=None):
    out = {} if out is None else out
    for key in config.keys():
        value = config[key]
        if isinstance(value, DictConfig):
            flatten_config(value, out)
        else:
            out[key] = value
    return out


def load_lifecycle_module():
    """Load ``scene/lifecycle.py`` standalone (never imports the ``scene`` package)."""

    name = "_adags_lifecycle_standalone"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, LIFECYCLE_PATH)
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__].
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


class LaneConfigTests(unittest.TestCase):
    def test_every_lane_exists(self):
        for name in LANE_MATRIX:
            self.assertTrue((CONFIG_DIR / name).is_file(), f"missing lane config {name}")

    def test_lane_keys_are_recursive_merge_compatible(self):
        for name in LANE_MATRIX:
            with self.subTest(lane=name):
                config = OmegaConf.load(CONFIG_DIR / name)
                args = build_trainer_namespace()
                missing = merge_like_main(config, args)
                self.assertEqual(
                    missing,
                    [],
                    f"{name}: keys absent from the argparse namespace "
                    f"(main.py would fail `assert hasattr(args, key)`): {missing}",
                )

    def test_base_config_is_still_recursive_merge_compatible(self):
        config = OmegaConf.load(CONFIG_DIR / BASE_CONFIG)
        missing = merge_like_main(config, build_trainer_namespace())
        self.assertEqual(missing, [])

    def test_lane_flag_matrix(self):
        for name, expected in LANE_MATRIX.items():
            with self.subTest(lane=name):
                flat = flatten_config(OmegaConf.load(CONFIG_DIR / name))
                for key, value in expected.items():
                    if value is None:
                        self.assertNotIn(
                            key, flat, f"{name}: must not set {key} (lifecycle-free lane)"
                        )
                    else:
                        self.assertIn(key, flat, f"{name}: must set {key} explicitly")
                        self.assertEqual(
                            flat[key], value, f"{name}: {key} is mislabeled"
                        )

    def test_l0_has_no_lifecycle_keys_at_all(self):
        flat = flatten_config(OmegaConf.load(CONFIG_DIR / "lane_l0_route0.yaml"))
        lifecycle_keys = sorted(key for key in flat if str(key).startswith("lifecycle_"))
        self.assertEqual(lifecycle_keys, [])

    def test_l0_body_matches_the_frozen_base_config(self):
        base = OmegaConf.load(CONFIG_DIR / BASE_CONFIG)
        lane = OmegaConf.load(CONFIG_DIR / "lane_l0_route0.yaml")
        self.assertEqual(
            OmegaConf.to_container(base, resolve=True),
            OmegaConf.to_container(lane, resolve=True),
        )

    def test_lifecycle_lanes_share_the_base_non_lifecycle_semantics(self):
        base = OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / BASE_CONFIG), resolve=True)
        for name in LIFECYCLE_LANES:
            with self.subTest(lane=name):
                lane = OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / name), resolve=True)
                for section, values in base.items():
                    if not isinstance(values, dict):
                        self.assertEqual(lane.get(section), values, f"{name}: {section} drifted")
                        continue
                    for key, value in values.items():
                        if name == "lane_smoke.yaml" and key == "iterations":
                            continue
                        self.assertEqual(
                            lane[section].get(key), value, f"{name}: {section}.{key} drifted"
                        )

    def test_lifecycle_lanes_pin_the_evidence_artifact_and_abstention(self):
        for name in LIFECYCLE_LANES:
            with self.subTest(lane=name):
                flat = flatten_config(OmegaConf.load(CONFIG_DIR / name))
                self.assertEqual(flat["lifecycle_evidence_dir"], EVIDENCE_DIR)
                # ADDENDUM A2: the 0.20 code default is inert against the real
                # sigma distribution; lanes must abstain at 0.10.
                self.assertEqual(flat["lifecycle_sigma_abstain"], 0.10)
                self.assertIn("lifecycle_evidence_preload", flat)

    def test_smoke_lane_is_short_and_births_three_times(self):
        flat = flatten_config(OmegaConf.load(CONFIG_DIR / "lane_smoke.yaml"))
        self.assertEqual(flat["iterations"], 600)
        self.assertEqual(flat["lifecycle_birth_warmup"], 200)
        self.assertEqual(flat["lifecycle_birth_interval"], 200)
        self.assertEqual(flat["lifecycle_birth_until"], 600)
        births = [
            iteration
            for iteration in range(1, flat["iterations"] + 1)
            if flat["lifecycle_birth_warmup"] <= iteration <= flat["lifecycle_birth_until"]
            and iteration % flat["lifecycle_birth_interval"] == 0
        ]
        self.assertEqual(births, [200, 400, 600])

    def test_l5_is_l3_plus_a_time_shift(self):
        l3 = flatten_config(OmegaConf.load(CONFIG_DIR / "lane_l3_full.yaml"))
        l5 = flatten_config(OmegaConf.load(CONFIG_DIR / "lane_l5_shifted.yaml"))
        self.assertEqual(set(l3), set(l5))
        differing = {key for key in l3 if l3[key] != l5[key]}
        self.assertEqual(differing, {"lifecycle_time_shift"})
        self.assertEqual(l5["lifecycle_time_shift"], 101)

    def test_l4_matches_l3_except_the_control_flags(self):
        l3 = flatten_config(OmegaConf.load(CONFIG_DIR / "lane_l3_full.yaml"))
        l4 = flatten_config(OmegaConf.load(CONFIG_DIR / "lane_l4_generic_capacity.yaml"))
        differing = {key for key in set(l3) | set(l4) if l3.get(key) != l4.get(key)}
        self.assertEqual(
            differing,
            {"lifecycle_protection", "lifecycle_exposure", "lifecycle_birth_mode"},
        )


class OptimizationParamsLifecycleSurfaceTests(unittest.TestCase):
    def test_every_lifecycle_config_field_has_an_optimization_param(self):
        lifecycle = load_lifecycle_module()
        args = build_trainer_namespace()
        for spec in dataclass_fields(lifecycle.LifecycleConfig):
            attribute = f"lifecycle_{spec.name}"
            self.assertTrue(
                hasattr(args, attribute),
                f"LifecycleConfig.{spec.name} has no OptimizationParams mirror "
                f"({attribute}); from_namespace would silently use the dataclass default",
            )

    def test_defaults_round_trip_into_a_disabled_valid_config(self):
        lifecycle = load_lifecycle_module()
        cfg = lifecycle.LifecycleConfig.from_namespace(build_trainer_namespace())
        self.assertFalse(cfg.enable)
        self.assertFalse(cfg.protection)
        self.assertFalse(cfg.exposure)
        self.assertFalse(cfg.birth)
        self.assertEqual(cfg.evidence_mode, "e1")
        self.assertEqual(cfg.birth_mode, "e2")

    def test_each_lane_builds_a_valid_lifecycle_config(self):
        lifecycle = load_lifecycle_module()
        for name, expected in LANE_MATRIX.items():
            with self.subTest(lane=name):
                args = build_trainer_namespace()
                self.assertEqual(merge_like_main(OmegaConf.load(CONFIG_DIR / name), args), [])
                cfg = lifecycle.LifecycleConfig.from_namespace(args)
                self.assertEqual(cfg.enable, bool(expected["lifecycle_enable"]))
                self.assertEqual(cfg.protection, bool(expected["lifecycle_protection"]))
                self.assertEqual(cfg.exposure, bool(expected["lifecycle_exposure"]))
                self.assertEqual(cfg.birth, bool(expected["lifecycle_birth"]))
                if expected["lifecycle_enable"]:
                    self.assertEqual(cfg.birth_mode, expected["lifecycle_birth_mode"])
                    self.assertEqual(cfg.evidence_mode, expected["lifecycle_evidence_mode"])

    def test_evidence_runtime_knobs_are_exposed(self):
        args = build_trainer_namespace()
        for attribute in (
            "lifecycle_evidence_dir",
            "lifecycle_time_shift",
            "lifecycle_tau_rel",
            "lifecycle_kappa",
            "lifecycle_k_gap",
            "lifecycle_sigma_abstain",
            "lifecycle_evidence_preload",
        ):
            self.assertTrue(hasattr(args, attribute), attribute)
        self.assertEqual(args.lifecycle_tau_rel, 0.03)
        self.assertEqual(args.lifecycle_kappa, 2.5)
        self.assertEqual(args.lifecycle_k_gap, 3.0)
        self.assertEqual(args.lifecycle_sigma_abstain, 0.20)
        self.assertTrue(args.lifecycle_evidence_preload)


if __name__ == "__main__":
    unittest.main()
