"""Static admission of EL-GS configs (mirrors tests/test_lane_configs.py).

CPU only, no CUDA imports: the argparse surface is rebuilt from the
`arguments` package (import-safe) and main.py's extra top-level keys
are read via ast, so a YAML key that recursive_merge cannot set fails
HERE, not minutes into an Apollo job. Also validates the prereg
transition table's presence/schema and the smoke config's
non-evidence posture.
"""

import ast
import json
import pathlib
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
SMOKE = ROOT / "configs" / "elgs" / "smoke_elgs.yaml"
TABLE = ROOT / "configs" / "elgs" / "prereg_latch_transition_table_v1.json"


def _optimization_param_names():
    import sys

    sys.path.insert(0, str(ROOT))
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


def _main_top_level_keys():
    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
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


class SmokeConfigAdmissionTests(unittest.TestCase):
    def test_every_key_is_recursive_merge_compatible(self):
        config = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
        known = _optimization_param_names() | _main_top_level_keys()
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key in value:
                    self.assertIn(
                        sub_key, known,
                        f"{key}.{sub_key} would fail recursive_merge's assert",
                    )
            else:
                self.assertIn(key, known, f"top-level {key} unknown to main.py")

    def test_elgs_enabled_with_prereg_dir(self):
        config = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
        opt = config["OptimizationParams"]
        self.assertTrue(opt["elgs_enable"])
        self.assertEqual(opt["elgs_prereg_dir"], "configs/elgs")
        self.assertTrue(opt["elgs_smoke_schedule"])

    def test_category3_values_present_for_smoke(self):
        # The smoke sets every dev hyperparameter explicitly (no -1
        # sentinels may reach a run: the wrapper refuses unset values
        # on evidence-bearing runs and the smoke should not rely on
        # sentinel passthrough either).
        config = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
        opt = config["OptimizationParams"]
        for key, value in opt.items():
            if key.startswith("elgs_") and isinstance(value, (int, float)):
                self.assertNotEqual(value, -1, f"{key} left at the unset sentinel")


class PreregArtifactTests(unittest.TestCase):
    def test_transition_table_present_and_schema_tagged(self):
        table = json.loads(TABLE.read_text(encoding="utf-8"))
        self.assertEqual(table["schema_version"], "elgs-latch-transition-table-v1")
        self.assertEqual(len(table["operations"]), 15)

    def test_elgs_block_documented_as_non_tunable_for_category1(self):
        text = (ROOT / "arguments" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("Category-1 structural constants", text)
        self.assertIn("elgs_prereg_dir", text)


if __name__ == "__main__":
    unittest.main()
