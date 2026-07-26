import copy
import json
import unittest
from pathlib import Path

from depth_visibility.errors import ProvenanceError, SchemaError
from depth_visibility.interval_admission import (
    FROZEN_SELECTION_RULE,
    assert_label_free_read_path,
    build_interval_artifact,
    canonical_scientific_hash,
    scan_intervals,
    validate_candidate_universe,
    validate_config,
)


CONFIG = Path("configs/depth_visibility/csvl_vpl_stage1c_c0_v1.json")


def config():
    return validate_config(json.loads(CONFIG.read_text(encoding="utf-8")))


def observation(identifier, frame, order):
    return {
        "observation_id": identifier,
        "frame": frame,
        "depth_order": order,
    }


def camera_evidence(camera, vector):
    return {
        "camera_id": camera,
        "source_calibration_id": f"{camera}-source-calibration",
        "destination_calibration_id": f"{camera}-destination-calibration",
        "required_projected_displacement_xy": [2.0, 0.0],
        "quantization_aware_tolerance_pixels": 8.0,
        "controls": {
            "valid": {
                "flow_chain": {
                    "record_ids": [f"{camera}-flow-record"],
                    "chain_displacement_xy": list(vector),
                    "steps": [{
                        "manifest_valid_pixel_fraction": 0.9,
                        "boundary_distance_pixels": 4.0,
                    }],
                }
            }
        },
    }


def candidate(source, destination, source_frame, destination_frame, source_order, destination_order, suffix):
    return {
        "edge_key": f"{source}->{destination}",
        "source_observation_id": source,
        "destination_observation_id": destination,
        "source_p03_hypothesis_id": f"p03-source-{suffix}",
        "destination_p03_hypothesis_id": f"p03-destination-{suffix}",
        "source_frame": source_frame,
        "destination_frame": destination_frame,
        "frame_gap": destination_frame - source_frame,
        "source_depth_order": source_order,
        "destination_depth_order": destination_order,
        "order_transition": f"{source_order}->{destination_order}",
        "camera_evidence": [
            camera_evidence("cam01", [2.0, 0.0]),
            camera_evidence("cam02", [0.0, 2.0]),
        ],
    }


class Stage1CIntervalAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.observations = [
            observation("source", 10, "rear"),
            observation("front-destination", 12, "front"),
            observation("rear-destination", 13, "rear"),
        ]
        self.cross = candidate(
            "source", "front-destination", 10, 12, "rear", "front", "cross"
        )
        self.same = candidate(
            "source", "rear-destination", 10, 13, "rear", "rear", "same"
        )

    def test_frozen_config_rejects_any_rule_change(self):
        frozen = config()
        self.assertEqual(frozen["selection_rule"], FROZEN_SELECTION_RULE)
        changed = copy.deepcopy(frozen)
        changed["selection_rule"]["window_stride_frames"] = 1
        with self.assertRaises(SchemaError):
            validate_config(changed)

    def test_cross_order_and_ambiguity_admit_one_development_window(self):
        result = scan_intervals(
            self.observations,
            [self.cross, self.same],
            frame_range=(0, 44),
        )
        self.assertTrue(result["gate_c0_admitted"])
        self.assertEqual(len(result["selected_intervals"]), 1)
        selected = result["selected_intervals"][0]
        self.assertEqual(selected["cross_order_candidate_count"], 1)
        self.assertEqual(selected["multi_candidate_source_count"], 1)
        self.assertEqual(selected["complete_candidate_provenance_fraction"], 1.0)
        self.assertEqual(selected["flow_directional_diversity"]["occupied_octant_count"], 2)

    def test_zero_cross_order_is_inadmissible_for_reveal(self):
        same_only = copy.deepcopy(self.cross)
        same_only["destination_depth_order"] = "rear"
        same_only["order_transition"] = "rear->rear"
        result = scan_intervals(
            self.observations,
            [same_only, self.same],
            frame_range=(0, 44),
        )
        self.assertFalse(result["gate_c0_admitted"])
        populated = [value for value in result["all_windows"] if value["candidate_count"]]
        self.assertTrue(populated)
        self.assertTrue(all(
            "zero_front_rear_cross_order_candidates" in value["rejection_reasons"]
            for value in populated
        ))

    def test_tail_window_is_deterministic_and_no_candidate_geometry_is_emitted(self):
        result = scan_intervals(
            self.observations,
            [self.cross, self.same],
            frame_range=(0, 50),
        )
        bounds = [(value["start_frame"], value["end_frame"]) for value in result["all_windows"]]
        self.assertEqual(bounds, [(0, 29), (15, 44), (21, 50)])
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn("world_xyz", serialized)
        self.assertNotIn("median_optical_z", serialized)

    def test_candidate_universe_mismatch_fails_closed(self):
        stage1_candidates = [{
            "source_observation_id": "source",
            "destination_observation_id": "front-destination",
        }]
        validate_candidate_universe(self.observations, stage1_candidates, [self.cross])
        mismatched = copy.deepcopy(self.cross)
        mismatched["destination_observation_id"] = "missing"
        with self.assertRaises(ProvenanceError):
            validate_candidate_universe(self.observations, stage1_candidates, [mismatched])

    def test_prohibited_read_path_is_rejected(self):
        assert_label_free_read_path("sealed/stage1-ledger.json")
        with self.assertRaises(ProvenanceError):
            assert_label_free_read_path("scene/annotations/reveal-mask.json")
        with self.assertRaises(ProvenanceError):
            assert_label_free_read_path("scene/images/cam00/frame.png")

    def test_canonical_repeat_excludes_only_declared_runtime_metadata(self):
        scientific = {
            "selection": scan_intervals(
                self.observations,
                [self.cross, self.same],
                frame_range=(0, 44),
            )
        }
        first = build_interval_artifact(
            scientific,
            runtime_metadata={"timestamp_utc": "one", "slurm_job_id": "1", "absolute_output_root": "/one"},
            config=config(),
        )
        second = build_interval_artifact(
            copy.deepcopy(scientific),
            runtime_metadata={"timestamp_utc": "two", "slurm_job_id": "2", "absolute_output_root": "/two"},
            config=config(),
        )
        self.assertEqual(first["scientific_content_hash"], second["scientific_content_hash"])
        self.assertEqual(first["scientific_content_hash"], canonical_scientific_hash(scientific))
        self.assertNotEqual(first["runtime_metadata"], second["runtime_metadata"])


if __name__ == "__main__":
    unittest.main()
