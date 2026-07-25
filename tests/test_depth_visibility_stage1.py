import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from depth_visibility.canonical import sha256_file
from depth_visibility.errors import ArtifactError, FlowSemanticsError, ProvenanceError
from depth_visibility.flow import validate_p02_flow_record
from depth_visibility.surface_tracks import (
    P02FlowStore,
    associate_observations,
    build_stage1_ledger,
    candidate_evidence,
    canonical_scientific_hash,
    extract_p03_observations,
    validate_no_fabricated_hidden_geometry,
    validate_stage1_config,
)


CONFIG_PATH = Path("configs/depth_visibility/csvl_vpl_stage1_v1.json")


def config():
    return validate_stage1_config(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


def observation(name, frame, *, order="front", cameras=("cam01", "cam02"), risk=0.1):
    ordinal = 0 if order == "front" else 1
    return {
        "observation_id": name,
        "scene": "fixture",
        "frame": frame,
        "time": frame / 30.0,
        "target_camera": "cam00",
        "target_bin": [4, 5],
        "target_bin_pixels": 8,
        "target_pixel_center": [35.5, 43.5],
        "layer_ordinal": ordinal,
        "depth_order": order,
        "target_visibility_state": "visible" if order == "front" else "occluded",
        "median_optical_z": 5.0 + 0.1 * ordinal,
        "world_xyz": [0.1 * frame, 0.0, 5.0 + 0.1 * ordinal],
        "geometry_status": "directly_observed_p03_evidence_unprojected_at_bin_center",
        "observation_risk": risk,
        "uncertainty": {"p03_median_risk": risk},
        "physical_camera_ancestry": list(cameras),
        "camera_time_ancestry": [
            {"camera_id": camera, "frame": frame, "source": "P03_observed_geometry"}
            for camera in cameras
        ],
        "source_observations": {
            "p03_hypothesis_id": f"p03-{name}",
            "p03_frame_ledger_id": f"frame-{frame}",
            "source_da3_group_ids": ["group-a", "group-b"],
            "sample_count": 3,
            "physical_camera_count": len(cameras),
        },
        "order_evidence": {
            "layer_ordinal": ordinal,
            "depth_order": "front_to_rear",
            "order_pairs": [{"front_layer_ordinal": 0, "rear_layer_ordinal": 1, "relative_depth_gap": 0.1}],
            "provenance": "P03_ordered_multilayer_bin",
        },
    }


def edge(source, destination, *, cost=0.2, confidence=0.8, frame_gap=None):
    gap = destination[1] - source[1] if frame_gap is None else frame_gap
    return {
        "candidate_id": f"edge-{source[0]}-{destination[0]}",
        "source_observation_id": source[0],
        "destination_observation_id": destination[0],
        "source_frame": source[1],
        "destination_frame": destination[1],
        "frame_gap": gap,
        "common_camera_count": 2,
        "valid_camera_count": 2,
        "camera_evidence": [],
        "flow_record_ids": [f"flow-{source[1]}"],
        "endpoint_error_pixels_median": cost,
        "endpoint_error_pixels_maximum": cost,
        "normalized_endpoint_error_median": cost,
        "world_displacement": 0.1,
        "world_displacement_rscene_per_frame": 0.01,
        "cost": cost,
        "association_risk": 1.0 - confidence,
        "association_confidence": confidence,
        "admitted": cost <= 1.0,
        "mode": "valid",
    }


class Stage1LifecycleTests(unittest.TestCase):
    def test_two_layer_hide_reveal_has_stable_hypothesis_tracks(self):
        observations = [
            observation("front-0", 0, order="front"),
            observation("rear-0", 0, order="rear"),
            observation("front-1", 1, order="rear"),
            observation("rear-1", 1, order="front"),
        ]
        candidates = [
            edge(("front-0", 0), ("front-1", 1), cost=0.1),
            edge(("rear-0", 0), ("rear-1", 1), cost=0.1),
        ]
        result = associate_observations(observations, candidates, config=config(), frame_range=(0, 1))
        self.assertEqual(result["abstained_observation_count"], 0)
        self.assertEqual(len(result["tracks"]), 2)
        rear_track = next(
            track for track in result["tracks"]
            if track["records"][0]["observation_id"] == "rear-0"
        )
        self.assertEqual(
            [record["track_id"] for record in rear_track["records"]],
            [rear_track["track_id"], rear_track["track_id"]],
        )
        self.assertIn("revealed", rear_track["records"][1]["visibility_events"])
        self.assertEqual(rear_track["records"][0]["state"], "occluded")
        self.assertEqual(rear_track["records"][1]["state"], "visible")

    def test_disappearance_bounded_dormancy_and_reappearance(self):
        observations = [observation("a0", 0), observation("a3", 3)]
        candidates = [edge(("a0", 0), ("a3", 3), cost=0.1)]
        result = associate_observations(observations, candidates, config=config(), frame_range=(0, 3))
        self.assertEqual(len(result["tracks"]), 1)
        records = result["tracks"][0]["records"]
        self.assertEqual([record["state"] for record in records], ["visible", "dormant", "dormant", "visible"])
        self.assertEqual(records[1]["uncertainty"]["dormancy_age"], 1)
        self.assertEqual(records[2]["uncertainty"]["dormancy_age"], 2)
        self.assertIn("reappeared", records[3]["visibility_events"])
        validate_no_fabricated_hidden_geometry(result)

    def test_split_ambiguity_abstains_without_new_identity(self):
        observations = [observation("a0", 0), observation("a1", 1), observation("b1", 1)]
        candidates = [
            edge(("a0", 0), ("a1", 1), cost=0.1),
            edge(("a0", 0), ("b1", 1), cost=0.2),
        ]
        result = associate_observations(observations, candidates, config=config(), frame_range=(0, 1))
        self.assertEqual(result["abstained_observation_count"], 2)
        self.assertTrue(all(value["track_id"] is None for value in result["abstentions"]))
        self.assertTrue(all("split" in value["abstention_reason"] for value in result["abstentions"]))
        self.assertEqual(len(result["tracks"]), 1)

    def test_merge_ambiguity_abstains_without_arbitrary_winner(self):
        observations = [observation("a0", 0), observation("b0", 0), observation("c1", 1)]
        candidates = [
            edge(("a0", 0), ("c1", 1), cost=0.1),
            edge(("b0", 0), ("c1", 1), cost=0.1),
        ]
        result = associate_observations(observations, candidates, config=config(), frame_range=(0, 1))
        self.assertEqual(result["abstained_observation_count"], 1)
        self.assertIn("merge", result["abstentions"][0]["abstention_reason"])
        self.assertIsNone(result["abstentions"][0]["track_id"])

    def test_dormancy_never_fabricates_hidden_xyz_or_order(self):
        result = associate_observations([observation("a0", 0)], [], config=config(), frame_range=(0, 2))
        validate_no_fabricated_hidden_geometry(result)
        dormant = result["tracks"][0]["records"][1:]
        self.assertTrue(all(record["geometry"]["world_xyz"] is None for record in dormant))
        self.assertTrue(all(record["depth_order"] == "unknown_not_observed" for record in dormant))


class Stage1ProvenanceTests(unittest.TestCase):
    def p03_fixture(self):
        layer = lambda ordinal, cameras: {
            "layer_ordinal": ordinal,
            "depth_order": "front_to_rear",
            "median_optical_z": 5.0 + ordinal,
            "median_risk": 0.2 + 0.1 * ordinal,
            "physical_camera_count": len(cameras),
            "physical_cameras": list(cameras),
            "sample_count": len(cameras),
        }
        return {
            "schema_version": "phase9-csvl-ledger-v1",
            "scene": "fixture",
            "target_camera": "cam00",
            "cam00_rgb_opened": False,
            "label_dependent_gate_a": "not_evaluable",
            "evidence_boundary": {
                "human_labels": "not_consumed",
                "temporal_identity_status": "not_propagated_in_p03_v7",
            },
            "frames": [{
                "frame": 0,
                "frame_ledger_id": "frame-ledger",
                "ordered_multilayer_bins": [{
                    "csvl_hypothesis_id": "p03-hypothesis",
                    "target_bin": [1, 2],
                    "target_bin_pixels": 8,
                    "physical_ancestry": ["cam01", "cam02", "cam03", "cam04"],
                    "source_da3_group_ids": ["group-a", "group-b"],
                    "order_pairs": [{
                        "front_layer_ordinal": 0,
                        "rear_layer_ordinal": 1,
                        "relative_depth_gap": 0.2,
                    }],
                    "layers": [
                        layer(0, ("cam01", "cam02")),
                        layer(1, ("cam03", "cam04")),
                    ],
                }],
            }],
        }

    def target_records(self):
        return {("cam00", 0): SimpleNamespace(
            K=np.eye(3), w2c_opencv=np.eye(4), time=0.0, image_path=None
        )}

    def test_complete_permitted_camera_ancestry_and_prohibited_read_enforcement(self):
        observations = extract_p03_observations(
            self.p03_fixture(), target_records=self.target_records(), config=config()
        )
        self.assertEqual(len(observations), 2)
        front = next(value for value in observations if value["layer_ordinal"] == 0)
        self.assertEqual(front["physical_camera_ancestry"], ["cam01", "cam02"])
        self.assertTrue(all(value["camera_id"] != "cam00" for value in front["camera_time_ancestry"]))
        contaminated = self.p03_fixture()
        contaminated["cam00_rgb_opened"] = True
        with self.assertRaises(ProvenanceError):
            extract_p03_observations(contaminated, target_records=self.target_records(), config=config())
        contaminated = self.p03_fixture()
        contaminated["frames"][0]["ordered_multilayer_bins"][0]["layers"][0]["physical_cameras"].append("cam00")
        with self.assertRaises(ProvenanceError):
            extract_p03_observations(contaminated, target_records=self.target_records(), config=config())

    def test_z_sign_or_depth_order_inversion_fails_closed(self):
        inverted = self.p03_fixture()
        inverted["frames"][0]["ordered_multilayer_bins"][0]["layers"][0]["median_optical_z"] = -5.0
        with self.assertRaises(ProvenanceError):
            extract_p03_observations(inverted, target_records=self.target_records(), config=config())
        inverted = self.p03_fixture()
        layers = inverted["frames"][0]["ordered_multilayer_bins"][0]["layers"]
        layers[0]["median_optical_z"], layers[1]["median_optical_z"] = 7.0, 5.0
        with self.assertRaises(ProvenanceError):
            extract_p03_observations(inverted, target_records=self.target_records(), config=config())

    def test_exact_repeat_canonical_hash_excludes_run_metadata(self):
        scientific = {"tracks": [{"track_id": "algorithmic", "risk": 0.25}], "inputs": {"sha256": "a" * 64}}
        first = build_stage1_ledger(
            scientific,
            runtime_metadata={"timestamp_utc": "one", "slurm_job_id": "1", "absolute_output_root": "/one"},
            config=config(),
        )
        second = build_stage1_ledger(
            copy.deepcopy(scientific),
            runtime_metadata={"timestamp_utc": "two", "slurm_job_id": "2", "absolute_output_root": "/two"},
            config=config(),
        )
        self.assertEqual(first["scientific_content_hash"], second["scientific_content_hash"])
        self.assertEqual(first["scientific_content_hash"], canonical_scientific_hash(scientific))
        self.assertNotEqual(first["runtime_metadata"], second["runtime_metadata"])


class Stage1FlowNegativeTests(unittest.TestCase):
    def _record(self, root, camera, frame, dx):
        flow = np.zeros((8, 8, 2), dtype=np.float32)
        flow[..., 0] = dx
        valid = np.ones((8, 8), dtype=bool)
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
            "source_width": 8,
            "source_height": 8,
            "target_width": 8,
            "target_height": 8,
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
            "valid_pixel_fraction": 1.0,
            "source_hashes": ["a" * 64, "b" * 64],
            "array_hashes": {
                "flow_contiguous_sha256": hashlib.sha256(np.ascontiguousarray(flow).tobytes()).hexdigest(),
                "mask_contiguous_sha256": hashlib.sha256(np.ascontiguousarray(valid).tobytes()).hexdigest(),
                "npz_sha256": sha256_file(path),
            },
        }

    def _store(self, root):
        records = [
            self._record(root, "cam01", 0, 1.0),
            self._record(root, "cam01", 1, 3.0),
            self._record(root, "cam02", 0, 4.0),
            self._record(root, "cam02", 1, 5.0),
            self._record(root, "cam03", 0, 7.0),
            self._record(root, "cam03", 1, 8.0),
        ]
        manifest = {
            "schema_version": "phase9-flow-manifest-v1",
            "direction": "forward_t_to_t_plus_1",
            "cam00_rgb_opened": False,
            "label_dependent_gate_a": "not_evaluable",
            "record_count": 6,
            "expected_record_count": 6,
            "camera_ids": ["cam01", "cam02", "cam03"],
            "records": records,
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return P02FlowStore(manifest, manifest_path=path, manifest_sha256=sha256_file(path)), records

    def test_valid_vs_reversed_corrupted_camera_swap_and_temporal_offset(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, _ = self._store(Path(temporary))
            point = np.array([2.0, 2.0])
            valid = store.sample_chain("cam01", 0, 1, point, mode="valid")
            reversed_flow = store.sample_chain("cam01", 0, 1, point, mode="reversed_flow")
            corrupted = store.sample_chain("cam01", 0, 1, point, mode="corrupted_flow")
            swapped = store.sample_chain("cam01", 0, 1, point, mode="camera_swap")
            offset = store.sample_chain("cam01", 0, 1, point, mode="temporal_offset")
            np.testing.assert_allclose(valid["destination_xy"], [3.0, 2.0])
            np.testing.assert_allclose(reversed_flow["destination_xy"], [1.0, 2.0])
            self.assertGreater(np.linalg.norm(np.asarray(corrupted["destination_xy"]) - [3.0, 2.0]), 10.0)
            np.testing.assert_allclose(swapped["destination_xy"], [6.0, 2.0])
            np.testing.assert_allclose(offset["destination_xy"], [5.0, 2.0])
            self.assertNotEqual(valid["destination_xy"], reversed_flow["destination_xy"])

    def test_invalid_controls_increase_risk_or_break_association(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, _ = self._store(Path(temporary))
            source = observation("source", 0)
            destination = observation("destination", 1)
            for value, world in ((source, [10.0, 10.0, 5.0]), (destination, [15.0, 10.0, 5.0])):
                value["world_xyz"] = world
                value["median_optical_z"] = 5.0
                value["target_pixel_center"] = [2.0 if value is source else 3.0, 2.0]
                value["target_bin_pixels"] = 1
            camera = lambda: SimpleNamespace(
                K=np.eye(3), w2c_opencv=np.eye(4), width=8, height=8
            )
            train = {
                (name, frame): camera()
                for name in ("cam01", "cam02")
                for frame in (0, 1)
            }
            target = {("cam00", frame): camera() for frame in (0, 1)}
            rows = {
                mode: candidate_evidence(
                    source,
                    destination,
                    train_records=train,
                    target_records=target,
                    flow_store=store,
                    r_scene=200.0,
                    config=config(),
                    mode=mode,
                )
                for mode in (
                    "valid", "corrupted_flow", "reversed_flow",
                    "camera_swap", "temporal_offset",
                )
            }
            self.assertIsNotNone(rows["valid"])
            self.assertTrue(rows["valid"]["admitted"])
            for mode in ("corrupted_flow", "reversed_flow", "camera_swap", "temporal_offset"):
                self.assertIsNotNone(rows[mode])
                self.assertTrue(
                    not rows[mode]["admitted"]
                    or rows[mode]["association_risk"] > rows["valid"]["association_risk"]
                    or rows[mode]["association_confidence"] < rows["valid"]["association_confidence"]
                )

    def test_missing_mismatched_corrupted_and_reversed_records_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store, records = self._store(root)
            path = Path(records[0]["flow_npz_path"])
            path.write_bytes(b"corrupted")
            with self.assertRaises(ArtifactError):
                store.sample_chain("cam01", 0, 1, np.array([2.0, 2.0]))
            invalid = copy.deepcopy(records[1])
            invalid["direction"] = "backward_t_to_t_minus_1"
            with self.assertRaises(FlowSemanticsError):
                validate_p02_flow_record(invalid)
            missing = copy.deepcopy(records[1])
            missing.pop("array_hashes")
            with self.assertRaises(FlowSemanticsError):
                validate_p02_flow_record(missing)
            offset = copy.deepcopy(records[1])
            offset["source_frame"] = 9
            with self.assertRaises(FlowSemanticsError):
                validate_p02_flow_record(offset)


if __name__ == "__main__":
    unittest.main()
