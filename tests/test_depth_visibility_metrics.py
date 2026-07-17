from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from depth_visibility.baselines import (
    choose_spatial_baseline,
    match_selected_fraction,
    validate_baseline_registry,
)
from depth_visibility.evaluator import decide_gate_a, load_schema_bundle, validate_named_artifact
from depth_visibility.metrics import (
    aggregate_spatial_frames,
    equal_mass_ece,
    event_window_metrics,
    flow_relative_flicker,
    isotonic_pav,
    ordering_metrics,
    pooled_masked_psnr,
    relative_error_reduction,
    spatial_frame_counts,
)


ROOT = Path(__file__).resolve().parents[1]


class MetricContractTests(unittest.TestCase):
    def test_semantic_union_unmatched_prediction_is_false_positive(self) -> None:
        prediction = np.zeros((16, 16), dtype=bool)
        reference = np.zeros((16, 16), dtype=bool)
        prediction[1:5, 1:5] = True
        reference[10:14, 10:14] = True
        frame = spatial_frame_counts(prediction, reference, spatial_complete=True)
        window = aggregate_spatial_frames([frame])
        self.assertEqual(window["region_counts"]["fp"], 16)
        self.assertEqual(window["region_counts"]["fn"], 16)
        self.assertEqual(window["region"]["iou"], 0.0)

    def test_empty_empty_spatial_frame_is_true_negative_not_macro_unit(self) -> None:
        empty = np.zeros((8, 8), dtype=bool)
        frame = spatial_frame_counts(empty, empty, spatial_complete=True)
        window = aggregate_spatial_frames([frame])
        self.assertTrue(window["true_negative"])
        self.assertFalse(window["contributing"])

    def test_false_positive_only_event_window_contributes_zero(self) -> None:
        result = event_window_metrics(
            [{"event_id": "p", "type": "reveal", "frame": 5}],
            [],
        )
        self.assertTrue(result["contributing"])
        self.assertEqual(result["f1"], 0.0)
        self.assertEqual(result["fp"], 1)

    def test_transition_matching_stable_tie(self) -> None:
        result = event_window_metrics(
            [
                {"event_id": "p1", "type": "reveal", "frame": 9},
                {"event_id": "p0", "type": "reveal", "frame": 11},
            ],
            [{"event_id": "r0", "type": "reveal", "frame": 10}],
        )
        self.assertEqual(result["matches"][0]["predicted_event_id"], "p1")
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))

    def test_ordering_duplicates_reversed_orientation(self) -> None:
        result = ordering_metrics([
            {"score": 2.0},
            {"score": 1.0},
            {"score": None, "abstained": True},
        ])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["auroc"], 1.0)
        self.assertAlmostEqual(result["coverage"], 2 / 3)

    def test_threshold_fraction_tie_uses_higher_threshold(self) -> None:
        result = match_selected_fraction([0.0, 1.0], 0.25, baseline_id="R031")
        self.assertEqual(result["threshold"], "positive_infinity")
        self.assertEqual(result["selected_fraction"], 0.0)

    def test_spatial_baseline_tie_uses_lower_id(self) -> None:
        winner = choose_spatial_baseline([
            {"baseline_id": "R031", "boundary_f1_4px": 0.5, "region_iou": 0.4},
            {"baseline_id": "R032", "boundary_f1_4px": 0.5, "region_iou": 0.4},
            {"baseline_id": "R033", "boundary_f1_4px": 0.4, "region_iou": 0.9},
        ])
        self.assertEqual(winner["winner"], "R031")

    def test_pav_is_monotone_and_ece_is_stable(self) -> None:
        fit = isotonic_pav(
            [0.1, 0.2, 0.3, 0.4],
            [0, 1, 0, 1],
            stable_ids=["a", "b", "c", "d"],
        )
        self.assertEqual(fit["fitted_probability"], [0.0, 0.5, 0.5, 1.0])
        score = equal_mass_ece(fit["fitted_probability"], [0, 1, 0, 1], ["a", "b", "c", "d"], bins=2)
        self.assertGreaterEqual(score["ece"], 0.0)
        self.assertLessEqual(score["ece"], 1.0)

    def test_exact_zero_psnr_is_finite_120_db(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.float64)
        mask = np.ones((4, 4), dtype=bool)
        result = pooled_masked_psnr([image], [image], [mask])
        self.assertTrue(result["exact_zero_mse"])
        self.assertEqual(result["psnr_db"], 120.0)

    def test_flow_relative_flicker_zero_for_matched_temporal_residual(self) -> None:
        previous = np.zeros((4, 4, 3), dtype=np.float64)
        current = np.full((4, 4, 3), 0.25, dtype=np.float64)
        flow = np.zeros((4, 4, 2), dtype=np.float64)
        result = flow_relative_flicker(
            current,
            previous,
            current,
            previous,
            flow,
            np.ones((4, 4), dtype=bool),
            np.ones((4, 4), dtype=bool),
            np.ones((4, 4), dtype=bool),
            np.zeros((4, 4), dtype=np.float64),
        )
        self.assertEqual(result["l1"], 0.0)

    def test_zero_baseline_relative_reduction_is_fail_closed(self) -> None:
        self.assertEqual(relative_error_reduction(0.0, 0.0), 0.0)
        self.assertEqual(relative_error_reduction(0.0, 0.1), -float("inf"))

    def test_missing_human_labels_are_not_evaluable(self) -> None:
        config = json.loads((ROOT / "configs/depth_visibility/csvl_isr_v1.json").read_text())
        score = {
            "schema_version": "phase9-gate-a-score-v1",
            "evidence_type": "label_free_diagnostic",
        }
        decision = decide_gate_a(score, config, tier="engineering")
        self.assertEqual(decision["status"], "not_evaluable")

    def test_schema_bundle_validates_terminal_and_registry_hashes(self) -> None:
        bundle = load_schema_bundle(ROOT / "configs/depth_visibility/phase9_schema_bundle_v1.json")
        terminal = {
            "schema_version": "phase9-terminal-manifest-v1",
            "run_id": "r",
            "action": "static",
            "status": "completed",
            "exit_code": 0,
            "execution_manifest_sha256": "0" * 64,
            "produced_artifacts": [],
            "scientific_payload": None,
            "failure": None,
            "terminal_id": "1" * 64,
        }
        validate_named_artifact(terminal, "phase9-terminal-manifest-v1", bundle)
        registry = json.loads((ROOT / "configs/depth_visibility/r031_baselines_v1.json").read_text())
        audit = validate_baseline_registry(registry, repo_root=ROOT, verify_present_files=True)
        self.assertTrue(audit["valid"])


if __name__ == "__main__":
    unittest.main()
