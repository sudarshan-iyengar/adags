import importlib.util
import inspect
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from depth_visibility.evaluator import resolved_python_argv, validate_execution_manifest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase9_execution_script", REPO / "scripts/run_phase9_depth_visibility.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExecutionBindingTests(unittest.TestCase):
    def test_pinned_da3_processed_size(self):
        self.assertEqual(MODULE._processed_size(1352, 1014), (504, 378))
        self.assertEqual(MODULE._processed_size(64, 48), (504, 378))

    def test_pinned_da3_intrinsics_use_row_scaling_without_half_pixel_offset(self):
        intrinsic = np.array(
            [[1000.0, 0.0, 675.5], [0.0, 900.0, 506.5], [0.0, 0.0, 1.0]]
        )
        actual = MODULE._pinned_da3_processed_intrinsics(
            intrinsic, 1352, 1014, 504, 378
        )
        expected = intrinsic.copy()
        expected[0, :] *= 504.0 / 1352.0
        expected[1, :] *= 378.0 / 1014.0
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)

    def test_resolved_argv_preserves_registered_scene(self):
        argv = [
            "python", "scripts/run_phase9_depth_visibility.py", "da3-conformance",
            "--run-id", "P9-A03", "--execution-manifest", "/tmp/resolved.json",
            "--scene", "cut_roasted_beef",
        ]
        manifest = {
            "run_id": "P9-A03",
            "action": "da3-conformance",
            "resolved_argv": argv,
        }
        self.assertEqual(
            resolved_python_argv(
                manifest,
                run_id="P9-A03",
                action="da3-conformance",
                launcher_path="scripts/run_phase9_depth_visibility.py",
                execution_manifest_path="/tmp/resolved.json",
            ),
            tuple(argv),
        )

    def test_execution_manifest_rejects_rehashed_scene_drift(self):
        argv = [
            "python", "scripts/run_phase9_depth_visibility.py", "da3-conformance",
            "--run-id", "P9-A03", "--execution-manifest", "/tmp/resolved.json",
            "--scene", "cut_roasted_beef",
        ]
        entry = {
            "run_id": "P9-A03",
            "action": "da3-conformance",
            "command": {
                "launcher_path": "scripts/run_phase9_depth_visibility.py",
                "argv_template": list(argv),
            },
            "expected_outputs": [
                {"path": "/tmp/terminal.json", "schema": "phase9-terminal-manifest-v1", "required": True}
            ],
            "input_artifacts": [],
            "external_inputs": [],
        }
        manifest = {
            "schema_version": "phase9-resolved-execution-v1",
            "run_id": "P9-A03",
            "action": "da3-conformance",
            "resolved_argv": list(argv),
            "resolved_argv_sha256": hashlib.sha256(
                json.dumps(argv, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            ).hexdigest(),
            "launcher_sha256": "0" * 64,
            "implementation": {},
            "configuration": {},
            "expected_outputs": list(entry["expected_outputs"]),
            "input_artifacts": [],
            "external_inputs": [],
            "input_binding_status": "unresolved",
        }
        validate_execution_manifest(
            manifest,
            run_id="P9-A03",
            action="da3-conformance",
            run_entry=entry,
            require_resolved=False,
        )
        manifest["resolved_argv"][-1] = "flame_steak"
        manifest["resolved_argv_sha256"] = hashlib.sha256(
            json.dumps(manifest["resolved_argv"], separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        with self.assertRaises(MODULE.ProvenanceError):
            validate_execution_manifest(
                manifest,
                run_id="P9-A03",
                action="da3-conformance",
                run_entry=entry,
                require_resolved=False,
            )

    def test_bound_a02_authority_rejects_mutated_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority_path = root / "da3.json"
            authority_path.write_text(
                json.dumps({"schema_version": "phase9-da3-authority-v1"}) + "\n",
                encoding="utf-8",
            )
            authority_sha = hashlib.sha256(authority_path.read_bytes()).hexdigest()
            terminal_path = root / "terminal.json"
            terminal = {
                "schema_version": "phase9-terminal-manifest-v1",
                "run_id": "P9-A02-DA3-WEIGHT-SHA",
                "action": "hash-da3",
                "status": "completed",
                "exit_code": 0,
                "produced_artifacts": [{
                    "path": str(authority_path),
                    "schema": "phase9-da3-authority-v1",
                    "producer_run_id": "P9-A02-DA3-WEIGHT-SHA",
                    "sha256": authority_sha,
                }],
            }
            terminal_path.write_text(json.dumps(terminal) + "\n", encoding="utf-8")
            terminal_sha = hashlib.sha256(terminal_path.read_bytes()).hexdigest()
            matrix_path = root / "matrix.json"
            matrix_path.write_text(
                json.dumps({"runs": []}) + "\n", encoding="utf-8"
            )
            execution = {"input_artifacts": [{
                "path": str(terminal_path),
                "producer_run_id": "P9-A02-DA3-WEIGHT-SHA",
                "schema": "phase9-terminal-manifest-v1",
                "sha256": terminal_sha,
                "status": "resolved_exact_before_submission",
            }]}
            bound_path, _, observed_terminal_sha = MODULE._bound_a02_authority(
                execution, matrix_path
            )
            self.assertEqual(bound_path, authority_path.resolve())
            self.assertEqual(observed_terminal_sha, terminal_sha)
            authority_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(MODULE.ProvenanceError):
                MODULE._bound_a02_authority(execution, matrix_path)

    def test_group_rule_selects_lower_repeated_anchor_and_rejects_cam00(self):
        records = [
            SimpleNamespace(camera_id=f"cam{index:02d}")
            for index in range(1, 12)
        ]
        groups = (
            ("cam02", "cam03", "cam04", "cam05", "cam06", "cam07"),
            ("cam01", "cam02", "cam03", "cam04", "cam05", "cam06"),
            ("cam01", "cam07", "cam08", "cam09", "cam10", "cam11"),
        )
        config = {"grouping": {
            "maximum_cameras": 6,
            "maximum_optical_axis_angle_degrees": 75,
            "minimum_center_distance_rscene": 0.02,
            "minimum_second_singular_value_rscene": 0.01,
        }}
        with mock.patch.object(MODULE, "enumerate_anchor_groups", return_value=groups):
            anchor, selected = MODULE._select_conformance_groups(
                records, 1.0, config
            )
        self.assertEqual(anchor, "cam01")
        self.assertEqual(selected, (
            ("cam01", "cam02", "cam03", "cam04", "cam05", "cam06"),
            ("cam01", "cam07", "cam08", "cam09", "cam10", "cam11"),
        ))
        with self.assertRaises(MODULE.ProvenanceError):
            MODULE._select_conformance_groups(
                records + [SimpleNamespace(camera_id="cam00")], 1.0, config
            )
    def test_geometry_input_check_rejects_cross_group_anchor_scale_drift(self):
        K = np.repeat(np.eye(3)[None, ...], 6, axis=0)
        first = {
            "depth": np.ones((6, 4, 5), dtype=np.float64),
            "intrinsics": K,
        }
        second = {
            "depth": np.full((6, 4, 5), 2.0, dtype=np.float64),
            "intrinsics": K,
        }
        groups = [
            {
                "member_camera_ids": [
                    "cam01", "cam02", "cam03", "cam04", "cam05", "cam06"
                ],
                "expected_processed_intrinsics": K,
            },
            {
                "member_camera_ids": [
                    "cam01", "cam07", "cam08", "cam09", "cam10", "cam11"
                ],
                "expected_processed_intrinsics": K,
            },
        ]
        report = MODULE._geometry_input_check(
            [first, second], groups, anchor_camera_id="cam01"
        )
        self.assertGreater(
            report["anchor_cross_group_relative_mad_maximum"], 0.05
        )
        self.assertEqual(
            report["processed_k_corner_error_maximum_pixels"], 0.0
        )

    def test_freeze_uses_the_explicit_matrix_argument(self):
        source = inspect.getsource(MODULE.action_freeze_implementation)
        self.assertIn("Path(args.matrix).resolve()", source)
        self.assertIn("_json(matrix_path)", source)
        self.assertNotIn("_json(DEFAULT_MATRIX)", source)

    def test_job_wrapper_executes_resolved_argv_not_reconstructed_cli(self):
        wrapper = (REPO / "scripts/run_phase9_depth_visibility_job.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('"${RESOLVED_ARGV[@]:1}"', wrapper)
        self.assertNotIn('scripts/run_phase9_depth_visibility.py   "$ACTION"', wrapper)


if __name__ == "__main__":
    unittest.main()
