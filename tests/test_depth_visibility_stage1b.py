import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from depth_visibility.association_audit import (
    _chain_score,
    _rotate_valid_chain,
    _sample_chain,
    build_ablation_candidate_sets,
    build_audit_artifact,
    canonical_scientific_hash,
    control_definitions,
    matched_control_diagnostics,
    metric_definitions,
    validate_stage1b_config,
)
from depth_visibility.canonical import sha256_file
from depth_visibility.errors import SchemaError
from depth_visibility.surface_tracks import P02FlowStore, validate_stage1_config


STAGE1_CONFIG = Path("configs/depth_visibility/csvl_vpl_stage1_v1.json")
STAGE1B_CONFIG = Path("configs/depth_visibility/csvl_vpl_stage1b_v1.json")


def stage1_config():
    return validate_stage1_config(json.loads(STAGE1_CONFIG.read_text(encoding="utf-8")))


def stage1b_config():
    return validate_stage1b_config(json.loads(STAGE1B_CONFIG.read_text(encoding="utf-8")))


class Stage1BFlowFixture:
    def _record(self, root, camera, frame, vector, *, mask_fraction=1.0):
        flow = np.zeros((12, 12, 2), dtype=np.float32)
        flow[...] = np.asarray(vector, dtype=np.float32)
        valid = np.ones((12, 12), dtype=bool)
        if mask_fraction < 1.0:
            valid[:, -1] = False
        path = root / f"{camera}_{frame:04d}.npz"
        np.savez(path, flow=flow, mask=valid)
        return {
            "schema_version": "depth-visibility-flow-schema-v1",
            "scene": "fixture",
            "source_camera": camera,
            "target_camera": camera,
            "source_frame": frame,
            "target_frame": frame + 1,
            "source_image": f"images/{camera}_{frame:04d}",
            "target_image": f"images/{camera}_{frame + 1:04d}",
            "source_width": 12,
            "source_height": 12,
            "target_width": 12,
            "target_height": 12,
            "direction": "forward_t_to_t_plus_1",
            "dt_seconds": 1 / 30,
            "units": "pixels_at_source_resolution",
            "coordinate_convention": "integer_pixel_centers",
            "sampling": "bilinear_align_corners_false",
            "validity_semantics": "true_means_sample_is_valid",
            "occlusion_semantics": "true_means_not_occluded",
            "generator_name": "fixture",
            "generator_revision": "fixture-revision",
            "flow_npz_path": str(path),
            "flow_key": "flow",
            "valid_key": "mask",
            "flow_dtype": "float32",
            "valid_dtype": "bool",
            "valid_pixel_fraction": float(np.mean(valid)),
            "source_hashes": ["a" * 64, "b" * 64],
            "array_hashes": {
                "flow_contiguous_sha256": hashlib.sha256(np.ascontiguousarray(flow).tobytes()).hexdigest(),
                "mask_contiguous_sha256": hashlib.sha256(np.ascontiguousarray(valid).tobytes()).hexdigest(),
                "npz_sha256": sha256_file(path),
            },
        }

    def store(self, root):
        records = []
        vectors = {
            "cam01": [(1.0, 0.0), (3.0, 0.0), (5.0, 0.0)],
            "cam02": [(0.0, 2.0), (0.0, 4.0), (0.0, 6.0)],
        }
        for camera, values in vectors.items():
            for frame, vector in enumerate(values):
                records.append(self._record(root, camera, frame, vector))
        manifest = {
            "schema_version": "phase9-flow-manifest-v1",
            "direction": "forward_t_to_t_plus_1",
            "cam00_rgb_opened": False,
            "label_dependent_gate_a": "not_evaluable",
            "record_count": len(records),
            "expected_record_count": len(records),
            "camera_ids": ["cam01", "cam02"],
            "records": records,
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return P02FlowStore(manifest, manifest_path=path, manifest_sha256=sha256_file(path))


class Stage1BControlSemanticsTests(unittest.TestCase, Stage1BFlowFixture):
    def test_exact_config_and_control_disclosures_are_frozen(self):
        config = stage1b_config()
        self.assertEqual(config["stage1_config_sha256"], sha256_file(STAGE1_CONFIG))
        definitions = control_definitions()
        self.assertIn("not a sealed backward", definitions["reversed_flow"]["semantic_disclosure"])
        self.assertIn("only flow array", definitions["camera_swap"]["fixed_problem"])
        changed = copy.deepcopy(config)
        changed["matched_controls"]["direction_rotation_degrees"] = 45
        with self.assertRaises(SchemaError):
            validate_stage1b_config(changed)

    def test_direction_time_and_camera_bindings_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            source = np.asarray([4.0, 4.0])
            valid = _sample_chain(
                store, sample_camera="cam01", source_frame=0, target_frame=1,
                source_xy=source,
            )
            reversed_flow = _sample_chain(
                store, sample_camera="cam01", source_frame=0, target_frame=1,
                source_xy=source, vector_transform="negate",
            )
            camera_swap = _sample_chain(
                store, sample_camera="cam02", source_frame=0, target_frame=1,
                source_xy=source,
            )
            temporal_offset = _sample_chain(
                store, sample_camera="cam01", source_frame=0, target_frame=1,
                source_xy=source, frame_offset=1,
            )
            np.testing.assert_allclose(valid["destination_xy"], [5.0, 4.0])
            np.testing.assert_allclose(reversed_flow["destination_xy"], [3.0, 4.0])
            np.testing.assert_allclose(camera_swap["destination_xy"], [4.0, 6.0])
            np.testing.assert_allclose(temporal_offset["destination_xy"], [7.0, 4.0])
            self.assertEqual(reversed_flow["steps"][0]["sample_source_frame"], 0)
            self.assertEqual(reversed_flow["steps"][0]["flow_direction"], "forward_t_to_t_plus_1")
            self.assertEqual(camera_swap["steps"][0]["sample_camera"], "cam02")
            self.assertEqual(temporal_offset["steps"][0]["sample_source_frame"], 1)
            self.assertEqual(valid["steps"][0]["source_resolution"], [12, 12])
            self.assertEqual(valid["steps"][0]["scale_or_resize"], "none_native_resolution")

    def test_fixed_reference_epe_and_direction_matched_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            source = np.asarray([4.0, 4.0])
            valid = _sample_chain(
                store, sample_camera="cam01", source_frame=0, target_frame=1,
                source_xy=source,
            )
            swapped = _sample_chain(
                store, sample_camera="cam02", source_frame=0, target_frame=1,
                source_xy=source,
            )
            destination = np.asarray([5.0, 4.0])
            valid_score = _chain_score(
                valid, destination_xy=destination, tolerance=2.0,
                required_vector=np.asarray([1.0, 0.0]), valid_chain=valid,
            )
            swapped_score = _chain_score(
                swapped, destination_xy=destination, tolerance=2.0,
                required_vector=np.asarray([1.0, 0.0]), valid_chain=valid,
            )
            self.assertEqual(valid_score["endpoint_error_pixels"], 0.0)
            self.assertAlmostEqual(swapped_score["endpoint_error_pixels"], np.sqrt(5.0))
            self.assertAlmostEqual(
                swapped_score["valid_flow_reference_disagreement_pixels"], np.sqrt(5.0)
            )
            rotated = _rotate_valid_chain(valid, source)
            self.assertAlmostEqual(rotated["chain_magnitude_pixels"], valid["chain_magnitude_pixels"])
            self.assertNotEqual(rotated["destination_xy"], valid["destination_xy"])
            self.assertFalse(metric_definitions()["stage1_reported_epe"]["self_evaluation"])


class Stage1BAblationAndHashTests(unittest.TestCase):
    def candidate(self):
        return {
            "candidate_id": "candidate",
            "source_observation_id": "source",
            "destination_observation_id": "destination",
            "source_frame": 0,
            "destination_frame": 1,
            "frame_gap": 1,
            "common_camera_count": 2,
            "valid_camera_count": 2,
            "camera_evidence": [],
            "flow_record_ids": ["flow-0"],
            "endpoint_error_pixels_median": 1.0,
            "endpoint_error_pixels_maximum": 1.0,
            "normalized_endpoint_error_median": 0.25,
            "world_displacement": 0.1,
            "world_displacement_rscene_per_frame": 0.01,
            "cost": 0.28,
            "association_risk": 0.25,
            "association_confidence": 0.75,
            "admitted": True,
            "mode": "valid",
        }

    def test_score_decomposition_keeps_candidate_universe_fixed(self):
        row = {
            "edge_key": "source->destination",
            "geometry_cost": 0.4,
            "zero_flow_normalized_error_median": 0.5,
            "zero_flow_reprojection_error_pixels_median": 2.0,
            "control_scores": {"valid": {
                "normalized_endpoint_error_median": 0.25,
                "association_risk": 0.25,
                "flow_record_ids": ["flow-0"],
                "flow_magnitude_normalized_error_median": 0.1,
                "flow_magnitude_absolute_error_pixels_median": 0.4,
            }},
        }
        variants = build_ablation_candidate_sets([row], [self.candidate()], stage1_config=stage1_config())
        self.assertTrue(all(len(values) == 1 for values in variants.values()))
        self.assertEqual(variants["full_current_score"][0]["candidate_id"], "candidate")
        self.assertEqual(variants["full_current_score"][0]["cost"], 0.28)
        self.assertEqual(variants["geometry_p03_only"][0]["flow_record_ids"], [])
        self.assertEqual(variants["geometry_plus_camera_without_flow"][0]["flow_record_ids"], [])
        self.assertNotEqual(
            variants["geometry_plus_camera_without_flow"][0]["cost"],
            variants["full_current_score"][0]["cost"],
        )

    def test_canonical_hash_excludes_only_declared_runtime_metadata(self):
        scientific = {"candidate": {"edge": "a->b", "risk": 0.2}}
        first = build_audit_artifact(
            scientific,
            runtime_metadata={"timestamp_utc": "one", "slurm_job_id": "1", "absolute_output_root": "/one"},
            config=stage1b_config(),
        )
        second = build_audit_artifact(
            copy.deepcopy(scientific),
            runtime_metadata={"timestamp_utc": "two", "slurm_job_id": "2", "absolute_output_root": "/two"},
            config=stage1b_config(),
        )
        self.assertEqual(first["scientific_content_hash"], second["scientific_content_hash"])
        self.assertEqual(first["scientific_content_hash"], canonical_scientific_hash(scientific))
        self.assertNotEqual(first["runtime_metadata"], second["runtime_metadata"])

    def test_temporal_offsets_are_reported_at_chain_and_step_levels_separately(self):
        row = {
            "control_scores": {control: None for control in (
                "valid",
                "reversed_flow",
                "camera_swap",
                "temporal_offset",
                "corrupted_flow",
                "direction_rotated_matched",
                "camera_swap_matched",
                "temporal_offset_matched",
            )},
            "camera_evidence": [{
                "controls": {
                    "temporal_offset_matched": {
                        "flow_chain": {
                            "chain_magnitude_pixels": 2.0,
                            "camera_baseline_world": None,
                            "matched_temporal_offset_frames": 2,
                            "steps": [
                                {
                                    "manifest_valid_pixel_fraction": 0.9,
                                    "boundary_distance_pixels": 4.0,
                                    "sample_camera": "cam01",
                                    "sample_source_frame": 12,
                                    "logical_source_frame": 10,
                                },
                                {
                                    "manifest_valid_pixel_fraction": 0.8,
                                    "boundary_distance_pixels": 3.0,
                                    "sample_camera": "cam01",
                                    "sample_source_frame": 13,
                                    "logical_source_frame": 11,
                                },
                            ],
                        }
                    }
                }
            }],
        }
        before = copy.deepcopy(row)
        diagnostics = matched_control_diagnostics([row])["temporal_offset_matched"]
        self.assertEqual(diagnostics["selected_chain_time_offset_frames"]["count"], 1)
        self.assertEqual(diagnostics["selected_chain_time_offset_frames"]["mean"], 2.0)
        self.assertEqual(diagnostics["record_step_time_offset_frames"]["count"], 2)
        self.assertEqual(diagnostics["record_step_time_offset_frames"]["mean"], 2.0)
        self.assertNotIn("record_time_offset_frames", diagnostics)
        self.assertEqual(row, before)


if __name__ == "__main__":
    unittest.main()
