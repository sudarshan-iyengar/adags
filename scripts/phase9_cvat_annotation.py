#!/usr/bin/env python3
"""Prepare and convert CVAT annotations for Phase 9 human labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depth_visibility.cvat_annotation import (  # noqa: E402
    assemble_phase9_labels,
    extract_cvat_polygons,
    generate_cvat_annotation_templates,
)


DEFAULT_PACKET = Path("/leonardo_work/EUHPC_D21_034/proj_adags/runs/phase9-depth-visibility-capacity/cycle-v10/annotation/cut_roasted_beef/packet-manifest.json")
DEFAULT_WINDOWS = ROOT / "configs/depth_visibility/annotation_windows_v1.json"
DEFAULT_RAW_ROOT = Path("/leonardo_work/EUHPC_D21_034/proj_adags/data/n3v/cut_roasted_beef")


def _print_summary(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    make = subparsers.add_parser("make-templates", help="Create CVAT task lists and Phase 9 CSV templates")
    make.add_argument("--packet-manifest", default=str(DEFAULT_PACKET))
    make.add_argument("--output-dir", required=True)
    make.add_argument("--scene", default="cut_roasted_beef")
    make.add_argument("--raw-scene-root", default=str(DEFAULT_RAW_ROOT))
    make.add_argument("--test-camera", default="cam00")

    extract = subparsers.add_parser("extract-cvat", help="Extract polygon rows from a native CVAT XML export")
    extract.add_argument("--cvat-xml", required=True)
    extract.add_argument("--role", required=True, choices=["annotator_a", "annotator_b", "adjudication"])
    extract.add_argument("--output-csv", required=True)
    extract.add_argument("--window-id", required=True)
    extract.add_argument("--camera-id", default="cam00")
    extract.add_argument("--frame-start", type=int, default=None)
    extract.add_argument("--label-name", default="rear_surface_track")

    assemble = subparsers.add_parser("assemble-labels", help="Assemble validated Phase 9 label JSON from CSV tables")
    assemble.add_argument("--windows-manifest", default=str(DEFAULT_WINDOWS))
    assemble.add_argument("--completed-windows-output", required=True)
    assemble.add_argument("--output-json", required=True)
    assemble.add_argument("--scene", default="cut_roasted_beef")
    assemble.add_argument("--annotator-a-id", required=True)
    assemble.add_argument("--annotator-b-id", required=True)
    assemble.add_argument("--adjudicator-id", required=True)
    assemble.add_argument("--polygon-csv", action="append", default=[])
    assemble.add_argument("--track-frames-csv", required=True)
    assemble.add_argument("--transitions-csv", required=True)
    assemble.add_argument("--ordering-pairs-csv", required=True)
    assemble.add_argument("--frame-reviews-csv", required=True)

    args = parser.parse_args(argv)
    if args.command == "make-templates":
        _print_summary(
            generate_cvat_annotation_templates(
                packet_manifest_path=args.packet_manifest,
                output_dir=args.output_dir,
                scene=args.scene,
                raw_scene_root=args.raw_scene_root,
                test_camera=args.test_camera,
            )
        )
        return 0
    if args.command == "extract-cvat":
        _print_summary(
            extract_cvat_polygons(
                cvat_xml_path=args.cvat_xml,
                role=args.role,
                output_csv=args.output_csv,
                window_id=args.window_id,
                camera_id=args.camera_id,
                frame_start=args.frame_start,
                label_name=args.label_name,
            )
        )
        return 0
    if args.command == "assemble-labels":
        artifact = assemble_phase9_labels(
            windows_manifest_path=args.windows_manifest,
            completed_windows_output=args.completed_windows_output,
            output_json=args.output_json,
            scene=args.scene,
            annotator_a_id=args.annotator_a_id,
            annotator_b_id=args.annotator_b_id,
            adjudicator_id=args.adjudicator_id,
            polygon_csvs=args.polygon_csv,
            track_frames_csv=args.track_frames_csv,
            transitions_csv=args.transitions_csv,
            ordering_pairs_csv=args.ordering_pairs_csv,
            frame_reviews_csv=args.frame_reviews_csv,
        )
        _print_summary(
            {
                "output_json": args.output_json,
                "completed_windows_output": args.completed_windows_output,
                "row_counts": {key: len(value) for key, value in artifact["tables"].items()},
            }
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
