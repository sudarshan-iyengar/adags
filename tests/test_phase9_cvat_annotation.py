from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from depth_visibility.annotation import load_json, validate_human_label_freeze
from depth_visibility.cvat_annotation import (
    assemble_phase9_labels,
    extract_cvat_polygons,
    generate_cvat_annotation_templates,
)


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_PATH = ROOT / "configs/depth_visibility/annotation_windows_v1.json"
PACKET_PATH = Path("/leonardo_work/EUHPC_D21_034/proj_adags/runs/phase9-depth-visibility-capacity/cycle-v10/annotation/cut_roasted_beef/packet-manifest.json")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class Phase9CvatAnnotationTests(unittest.TestCase):
    def test_generate_templates_for_cut_windows(self) -> None:
        if not PACKET_PATH.exists():
            self.skipTest("cycle-v10 packet manifest is not present")
        with tempfile.TemporaryDirectory() as tmp:
            summary = generate_cvat_annotation_templates(
                packet_manifest_path=PACKET_PATH,
                output_dir=tmp,
                scene="cut_roasted_beef",
                raw_scene_root="/tmp/cut_roasted_beef",
            )
            self.assertEqual(summary["window_count"], 18)
            self.assertEqual(summary["task_count"], 36)
            self.assertTrue((Path(tmp) / "cvat_tasks.csv").is_file())
            self.assertTrue((Path(tmp) / "track_frames_final.csv").is_file())
            self.assertTrue((Path(tmp) / "README.md").is_file())

    def test_extract_cvat_image_polygons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xml_path = root / "cvat.xml"
            xml_path.write_text(
                """<annotations>
  <image id="0" name="cam00_0170.png" width="1352" height="1014">
    <polygon label="rear_surface_track" points="1,2;3,4;5,6">
      <attribute name="candidate_track_id">cut_w07_A_t001</attribute>
      <attribute name="state">visible</attribute>
      <attribute name="polygon_meaning">visible_rear_polygon</attribute>
      <attribute name="confidence">clear</attribute>
    </polygon>
  </image>
</annotations>
""",
                encoding="utf-8",
            )
            csv_path = root / "polygons.csv"
            summary = extract_cvat_polygons(
                cvat_xml_path=xml_path,
                role="annotator_a",
                output_csv=csv_path,
                window_id="cut_roasted_beef__w07__f170_180",
            )
            self.assertEqual(summary["polygon_rows"], 1)
            rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
            self.assertEqual(rows[0]["frame"], "170")
            self.assertEqual(rows[0]["camera_id"], "cam00")
            self.assertEqual(json.loads(rows[0]["points_json"]), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def test_assemble_labels_validates_against_freeze_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            polygon_csv = root / "polygons.csv"
            _write_csv(
                polygon_csv,
                ["role", "candidate_track_id", "window_id", "camera_id", "frame", "state", "polygon_meaning", "points_json", "confidence", "notes", "source_export", "cvat_track_id", "cvat_shape_label"],
                [
                    {
                        "role": "annotator_a",
                        "candidate_track_id": "cut_w07_A_t001",
                        "window_id": "cut_roasted_beef__w07__f170_180",
                        "camera_id": "cam00",
                        "frame": 170,
                        "state": "visible",
                        "polygon_meaning": "visible_rear_polygon",
                        "points_json": "[[1,2],[3,4],[5,6]]",
                    }
                ],
            )
            track_frames_csv = root / "track_frames.csv"
            _write_csv(
                track_frames_csv,
                [
                    "window_id",
                    "roster_track_id",
                    "camera_id",
                    "frame",
                    "state",
                    "evaluable",
                    "rear_polygon_source_role",
                    "rear_polygon_candidate_track_id",
                    "rear_polygon_json",
                    "state_aperture_source_role",
                    "state_aperture_candidate_track_id",
                    "state_aperture_json",
                    "annotator_a_response",
                    "annotator_b_response",
                    "adjudication",
                ],
                [
                    {
                        "window_id": "cut_roasted_beef__w07__f170_180",
                        "roster_track_id": "roster_track_001",
                        "camera_id": "cam00",
                        "frame": 170,
                        "state": "visible",
                        "evaluable": "true",
                        "rear_polygon_source_role": "annotator_a",
                        "rear_polygon_candidate_track_id": "cut_w07_A_t001",
                        "state_aperture_source_role": "annotator_a",
                        "state_aperture_candidate_track_id": "cut_w07_A_t001",
                        "annotator_a_response": "visible",
                        "annotator_b_response": "visible",
                    }
                ],
            )
            transitions_csv = root / "transitions.csv"
            _write_csv(
                transitions_csv,
                ["window_id", "roster_track_id", "camera_id", "frame_t", "frame_t1", "label", "evaluable", "annotator_a_response", "annotator_b_response", "adjudication"],
                [
                    {
                        "window_id": "cut_roasted_beef__w07__f170_180",
                        "roster_track_id": "roster_track_001",
                        "camera_id": "cam00",
                        "frame_t": 170,
                        "frame_t1": 171,
                        "label": "none",
                        "evaluable": "false",
                        "annotator_a_response": "none",
                        "annotator_b_response": "none",
                    }
                ],
            )
            ordering_csv = root / "ordering.csv"
            _write_csv(
                ordering_csv,
                ["window_id", "pair_id", "camera_id", "frame", "foreground_track_id", "rear_track_id", "label", "evaluable", "annotator_a_response", "annotator_b_response", "adjudication"],
                [
                    {
                        "window_id": "cut_roasted_beef__w07__f170_180",
                        "pair_id": "pair_001",
                        "camera_id": "cam00",
                        "frame": 170,
                        "foreground_track_id": "foreground_001",
                        "rear_track_id": "roster_track_001",
                        "label": "foreground_before_rear",
                        "evaluable": "false",
                        "annotator_a_response": "foreground_before_rear",
                        "annotator_b_response": "foreground_before_rear",
                    }
                ],
            )
            reviews_csv = root / "frame_reviews.csv"
            _write_csv(
                reviews_csv,
                ["window_id", "camera_id", "frame", "spatial_complete", "no_evaluable_visible_rear_surface", "unknown_reason", "annotator_provenance"],
                [
                    {
                        "window_id": "cut_roasted_beef__w07__f170_180",
                        "camera_id": "cam00",
                        "frame": 170,
                        "spatial_complete": "false",
                        "no_evaluable_visible_rear_surface": "false",
                        "unknown_reason": "unit-test partial review",
                    }
                ],
            )
            completed_windows = root / "completed_windows.json"
            labels = root / "labels.json"
            artifact = assemble_phase9_labels(
                windows_manifest_path=WINDOWS_PATH,
                completed_windows_output=completed_windows,
                output_json=labels,
                scene="cut_roasted_beef",
                annotator_a_id="annotator-a",
                annotator_b_id="annotator-b",
                adjudicator_id="adjudicator",
                polygon_csvs=[polygon_csv],
                track_frames_csv=track_frames_csv,
                transitions_csv=transitions_csv,
                ordering_pairs_csv=ordering_csv,
                frame_reviews_csv=reviews_csv,
            )
            self.assertTrue(labels.is_file())
            audit = validate_human_label_freeze(artifact, load_json(completed_windows))
            self.assertEqual(audit["row_counts"]["track_frames"], 1)


if __name__ == "__main__":
    unittest.main()
