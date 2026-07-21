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

    def test_full_scene_group_rule_keeps_all_unique_groups_and_rejects_cam00(self):
        records = [
            SimpleNamespace(camera_id=f"cam{index:02d}")
            for index in range(1, 9)
        ]
        groups = (
            ("cam02", "cam03", "cam04", "cam05", "cam06", "cam07"),
            ("cam01", "cam02", "cam03", "cam04", "cam05", "cam06"),
            ("cam02", "cam03", "cam04", "cam05", "cam06", "cam07"),
        )
        config = {"grouping": {
            "maximum_cameras": 6,
            "maximum_optical_axis_angle_degrees": 75,
            "minimum_center_distance_rscene": 0.02,
            "minimum_second_singular_value_rscene": 0.01,
        }}
        with mock.patch.object(MODULE, "enumerate_anchor_groups", return_value=groups):
            selected = MODULE._select_full_scene_groups(records, 1.0, config)
        self.assertEqual(selected, tuple(sorted(set(groups))))
        with self.assertRaises(MODULE.ProvenanceError):
            MODULE._select_full_scene_groups(
                records + [SimpleNamespace(camera_id="cam00")], 1.0, config
            )

    def test_da3_group_sidecar_writes_arrays_and_requires_train_hashes(self):
        K = np.repeat(np.eye(3)[None, ...], 6, axis=0)
        w2c = np.repeat(np.eye(4)[None, ...], 6, axis=0)
        group_input = {
            "member_camera_ids": [f"cam{index:02d}" for index in range(1, 7)],
            "intrinsics": K,
            "extrinsics_w2c": w2c,
            "expected_processed_intrinsics": K,
            "source_records": [
                {
                    "camera_id": f"cam{index:02d}",
                    "image_path": f"/data/cam{index:02d}.png",
                    "image_sha256": f"{index}" * 64,
                    "file_stem": f"cam{index:02d}_000000",
                    "time": 0.0,
                }
                for index in range(1, 7)
            ],
        }
        prediction = {
            "depth": np.ones((6, 4, 5), dtype=np.float64),
            "confidence": np.ones((6, 4, 5), dtype=np.float64),
            "intrinsics": K,
            "extrinsics": w2c,
            "processed_images": np.zeros((6, 4, 5, 3), dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            record = MODULE._write_da3_group_sidecar(
                sidecar_root=Path(directory), scene="cut_roasted_beef", frame=125,
                group_index=0, target_camera="cam00", group_input=group_input,
                prediction=prediction,
            )
            self.assertEqual(record["physical_ancestry"], group_input["member_camera_ids"])
            self.assertEqual(record["processed_depth_shape"], [6, 4, 5])
            self.assertEqual(record["processed_k_corner_error_maximum_pixels"], 0.0)
            depth_path = Path(directory) / record["array_refs"]["depth"]["path"]
            self.assertTrue(depth_path.is_file())
            np.testing.assert_array_equal(np.load(depth_path, allow_pickle=False), prediction["depth"])
            bad = {**group_input, "source_records": [dict(group_input["source_records"][0])]}
            bad["source_records"][0]["image_sha256"] = None
            with self.assertRaises(MODULE.ProvenanceError):
                MODULE._write_da3_group_sidecar(
                    sidecar_root=Path(directory), scene="cut_roasted_beef", frame=125,
                    group_index=1, target_camera="cam00", group_input=bad,
                    prediction=prediction,
                )


    def test_flow_record_builder_seals_forward_source_frame_npz(self):
        source = SimpleNamespace(
            camera_id="cam01", file_stem="images/cam01_0000", frame=0,
            time=0.0, width=3, height=2, image_sha256="a" * 64,
        )
        target = SimpleNamespace(
            camera_id="cam01", file_stem="images/cam01_0001", frame=1,
            time=0.1, width=3, height=2, image_sha256="b" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            flow_path = Path(directory) / "cam01_0000.npz"
            np.savez_compressed(
                flow_path,
                flow=np.zeros((2, 3, 2), dtype=np.float32),
                mask=np.ones((2, 3), dtype=bool),
            )
            record = MODULE._build_flow_record(
                scene="cut_roasted_beef", flow_path=flow_path,
                source_record=source, target_record=target,
                direction="forward_t_to_t_plus_1", generator_revision="c" * 64,
            )
            self.assertEqual(record["source_frame"], 0)
            self.assertEqual(record["target_frame"], 1)
            self.assertEqual(record["direction"], "forward_t_to_t_plus_1")
            self.assertEqual(record["valid_pixel_fraction"], 1.0)
            self.assertEqual(record["array_hashes"]["npz_sha256"], MODULE.sha256_file(flow_path))
            with self.assertRaises(MODULE.ContractError):
                MODULE._build_flow_record(
                    scene="cut_roasted_beef", flow_path=flow_path,
                    source_record=source, target_record=target,
                    direction="backward_t_to_t_minus_1", generator_revision="c" * 64,
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


    def test_p03_terminal_artifact_binding_rejects_mutated_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "manifest.json"
            artifact_path.write_text(
                json.dumps({
                    "schema_version": "phase9-da3-sidecar-v1",
                    "run_id": MODULE.P01_DA3_SIDECAR_RUN_ID,
                }) + "\n",
                encoding="utf-8",
            )
            artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            terminal = {
                "produced_artifacts": [{
                    "path": str(artifact_path),
                    "schema": "phase9-da3-sidecar-v1",
                    "producer_run_id": MODULE.P01_DA3_SIDECAR_RUN_ID,
                    "sha256": artifact_sha,
                }],
            }
            path, payload, observed_sha = MODULE._bound_terminal_json_artifact(
                terminal,
                producer_run_id=MODULE.P01_DA3_SIDECAR_RUN_ID,
                schema="phase9-da3-sidecar-v1",
            )
            self.assertEqual(path, artifact_path.resolve())
            self.assertEqual(payload["run_id"], MODULE.P01_DA3_SIDECAR_RUN_ID)
            self.assertEqual(observed_sha, artifact_sha)
            artifact_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(MODULE.ProvenanceError):
                MODULE._bound_terminal_json_artifact(
                    terminal,
                    producer_run_id=MODULE.P01_DA3_SIDECAR_RUN_ID,
                    schema="phase9-da3-sidecar-v1",
                )

    def test_p03_anchor_group_selection_uses_repeated_lower_camera(self):
        groups = [
            {"member_camera_ids": ["cam02", "cam03", "cam04"]},
            {"member_camera_ids": ["cam01", "cam02", "cam05"]},
            {"member_camera_ids": ["cam01", "cam06", "cam07"]},
        ]
        anchor, selected = MODULE._select_sidecar_anchor_groups(groups)
        self.assertEqual(anchor, "cam01")
        self.assertEqual(
            [tuple(item["member_camera_ids"]) for item in selected],
            [("cam01", "cam02", "cam05"), ("cam01", "cam06", "cam07")],
        )
        with self.assertRaises(MODULE.ProvenanceError):
            MODULE._select_sidecar_anchor_groups([
                {"member_camera_ids": ["cam00", "cam01", "cam02"]},
            ])

    def test_p03_group_prediction_loads_verified_sidecar_arrays(self):
        K = np.repeat(np.eye(3)[None, ...], 3, axis=0)
        w2c = np.repeat(np.eye(4)[None, ...], 3, axis=0)
        group_input = {
            "member_camera_ids": ["cam01", "cam02", "cam03"],
            "intrinsics": K,
            "extrinsics_w2c": w2c,
            "expected_processed_intrinsics": K,
            "source_records": [
                {
                    "camera_id": f"cam{index:02d}",
                    "image_path": f"/data/cam{index:02d}.png",
                    "image_sha256": f"{index}" * 64,
                    "file_stem": f"cam{index:02d}_000000",
                    "time": 0.0,
                }
                for index in range(1, 4)
            ],
        }
        prediction = {
            "depth": np.ones((3, 4, 5), dtype=np.float64),
            "confidence": np.ones((3, 4, 5), dtype=np.float64),
            "intrinsics": K,
            "extrinsics": w2c,
            "processed_images": np.zeros((3, 4, 5, 3), dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = MODULE._write_da3_group_sidecar(
                sidecar_root=root,
                scene="cut_roasted_beef",
                frame=0,
                group_index=0,
                target_camera="cam00",
                group_input=group_input,
                prediction=prediction,
            )
            loaded_prediction, loaded_group = MODULE._load_p01_group_prediction(root, record)
            np.testing.assert_array_equal(loaded_prediction["depth"], prediction["depth"])
            self.assertEqual(loaded_group["member_camera_ids"], group_input["member_camera_ids"])
            depth_path = root / record["array_refs"]["depth"]["path"]
            np.save(depth_path, np.zeros((3, 4, 5), dtype=np.float64))
            with self.assertRaises(Exception):
                MODULE._load_p01_group_prediction(root, record)

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


class SliceBPreparedIntegrationTests(unittest.TestCase):
    def test_main_uses_cli_seed_after_logger_setup(self):
        source = (REPO / "main.py").read_text(encoding="utf-8")
        self.assertIn("setup_seed(args.seed)", source)
        self.assertIn("safe_state(args.quiet, seed=args.seed)", source)
        self.assertNotIn("safe_state(args.quiet)\n", source)

    def test_main_counts_dynamic_and_hard_static_as_total_budget(self):
        source = (REPO / "main.py").read_text(encoding="utf-8")
        self.assertIn("dynamic_points = int(gaussians.get_xyz.shape[0])", source)
        self.assertIn("total_points = dynamic_points + static_points", source)
        self.assertIn('"num_GS": "points/dynamic"', source)
        self.assertIn("hard_total = hard_dynamic + hard_static", source)
        self.assertNotIn("dynamic_points = total_points - static_points", source)

    def test_gaussian_model_persists_capacity_state_in_routing_dict(self):
        source = (REPO / "scene/gaussian_model.py").read_text(encoding="utf-8")
        self.assertIn('"capacity_state": self._capture_capacity_state()', source)
        self.assertIn('self._restore_capacity_state(routing_motion_params.get("capacity_state"))', source)
        self.assertIn("def build_capacity_bank(self):", source)
        self.assertIn("created_iteration=iteration", source)

    def test_main_has_guarded_slice_b_transaction_hook(self):
        source = (REPO / "main.py").read_text(encoding="utf-8")
        self.assertIn("def maybe_apply_slice_b_capacity_transaction", source)
        self.assertIn("validate_slice_b_capacity_configuration(opt)", source)
        self.assertIn("gaussians.update_learning_rate(iteration)", source)
        self.assertIn("maybe_apply_slice_b_capacity_transaction(gaussians, opt, iteration)", source)
        self.assertIn("write_slice_b_capacity_ledger(scene.model_path, scene.gaussians, opt)", source)
        self.assertIn("write_local_run_summary(args.model_path, summary_updates)", source)

    def test_phase9_launcher_registers_train_action(self):
        source = (REPO / "scripts/run_phase9_depth_visibility.py").read_text(encoding="utf-8")
        self.assertIn("def action_train", source)
        self.assertIn("fixed_budget_lora_route0_filemask_residual_600k.yaml", source)
        self.assertIn('"train": lambda: action_train(entry, args, execution)', source)
        self.assertIn("phase9-training-metrics-v1", source)
        self.assertIn("phase9-capacity-ledger-v1", source)


if __name__ == "__main__":
    unittest.main()
