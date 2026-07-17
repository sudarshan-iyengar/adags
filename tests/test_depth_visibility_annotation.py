from __future__ import annotations

import copy
from pathlib import Path
import unittest

from depth_visibility.annotation import (
    assign_window,
    build_empty_annotation_packet,
    build_union_roster,
    load_json,
    validate_annotation_windows,
    validate_empty_annotation_packet,
    validate_human_label_freeze,
)
from depth_visibility.errors import ContractError
from depth_visibility.matching import exact_lexicographic_assignment, match_predictions


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_PATH = ROOT / "configs/depth_visibility/annotation_windows_v1.json"


class AnnotationContractTests(unittest.TestCase):
    def test_frozen_manifest_and_empty_packet(self) -> None:
        manifest = load_json(WINDOWS_PATH)
        audit = validate_annotation_windows(manifest)
        self.assertEqual(audit["window_count"], 54)
        self.assertEqual(audit["per_scene"], {
            "cut_roasted_beef": 18,
            "flame_steak": 18,
            "sear_steak": 18,
        })
        packet = build_empty_annotation_packet(
            manifest,
            manifest_path=str(WINDOWS_PATH),
        )
        packet_audit = validate_empty_annotation_packet(packet)
        self.assertEqual(packet_audit["review_row_count"], 594)
        self.assertTrue(packet_audit["human_fields_empty"])

    def test_shared_boundary_belongs_to_earlier_window(self) -> None:
        windows = load_json(WINDOWS_PATH)["windows"]
        scene_windows = [item for item in windows if item["scene"] == "cut_roasted_beef"]
        first, second = scene_windows[0], scene_windows[1]
        self.assertEqual(first["frame_end_inclusive"], second["frame_start_inclusive"])
        self.assertEqual(
            assign_window("cut_roasted_beef", first["frame_end_inclusive"], windows),
            first["window_id"],
        )

    def test_exact_assignment_uses_lexicographic_tie_without_epsilon(self) -> None:
        result = exact_lexicographic_assignment(
            [
                ("a0", "b0", 0.5),
                ("a0", "b1", 0.5),
                ("a1", "b0", 0.5),
                ("a1", "b1", 0.5),
            ],
            edge_prefix=("window",),
        )
        self.assertEqual(result["assignment"], [("a0", "b0"), ("a1", "b1")])
        self.assertEqual(result["integer_weight"], 2)

    def test_prediction_explicit_iou_map_uses_reference_id(self) -> None:
        result = match_predictions(
            "scene",
            "window",
            [
                {
                    "predicted_track_id": "p0",
                    "target_iou": {"r0": 0.8},
                    "source_iou": {"r0": 0.4},
                }
            ],
            [{"reference_track_id": "r0"}],
        )
        self.assertEqual(result["assignment"], [("p0", "r0")])

    def test_union_roster_preserves_one_sided_discovery(self) -> None:
        roster = build_union_roster(
            "w0",
            [
                {"track_id": "a0", "target_iou": {"b0": 0.9}, "source_iou": {"b0": 0.9}},
                {"track_id": "a1", "target_iou": {"b0": 0.0}, "source_iou": {"b0": 0.0}},
            ],
            [{"track_id": "b0"}],
        )
        self.assertEqual(sorted(item["status"] for item in roster["roster"]), ["matched", "one_sided"])
        one_sided = next(item for item in roster["roster"] if item["status"] == "one_sided")
        self.assertEqual(one_sided["missing_role_response"], "not_found")

    def test_fragment_merge_component_goes_to_adjudication(self) -> None:
        roster = build_union_roster(
            "w0",
            [
                {"track_id": "a0", "target_iou": {"b0": 0.8}, "source_iou": {"b0": 0.8}},
                {"track_id": "a1", "target_iou": {"b0": 0.7}, "source_iou": {"b0": 0.7}},
            ],
            [{"track_id": "b0"}],
        )
        self.assertEqual(len(roster["roster"]), 1)
        self.assertEqual(roster["roster"][0]["status"], "fragment_merge_unknown")
        self.assertTrue(roster["roster"][0]["requires_adjudication"])

    def test_completed_labels_require_filled_distinct_roles(self) -> None:
        manifest = load_json(WINDOWS_PATH)
        artifact = {
            "schema_version": "phase9-human-label-freeze-v1",
            "evidence_type": "human_reference",
            "tables": {
                "track_frames": [],
                "ordering_pairs": [],
                "transitions": [],
                "frame_reviews": [],
            },
        }
        with self.assertRaises(ContractError):
            validate_human_label_freeze(artifact, manifest)

    def test_r009_overlap_fails_even_if_boolean_claims_disjoint(self) -> None:
        manifest = load_json(WINDOWS_PATH)
        corrupt = copy.deepcopy(manifest)
        corrupt["windows"][0]["frame_start_inclusive"] = 85
        corrupt["windows"][0]["frame_end_inclusive"] = 95
        with self.assertRaises(ContractError):
            validate_annotation_windows(corrupt)


if __name__ == "__main__":
    unittest.main()
