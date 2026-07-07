#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hide_reveal_poc import (
    FrozenHideRevealParams,
    augment_real_manifest_system,
    create_real_manifest_from_eval,
    derive_real_poc_render_folders,
    discover_event_boundary_support,
    discover_nonoracle_event_candidates,
    direct_window_spec,
    evaluate_real_manifest,
    load_window_specs,
    render_actual_hide_reveal_real_windows,
    run_synthetic_poc,
    validate_real_manifest,
    write_json,
    write_real_manifest_template,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run proof-of-concept hide/reveal experiment tooling.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic", help="Run synthetic margin and matched-lifespan PoC checks.")
    synthetic.add_argument("--out-dir", default="refine-logs/hide_reveal_poc/synthetic")
    synthetic.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    synthetic.add_argument("--clips-per-type", type=int, default=8)
    synthetic.add_argument("--c-min", type=float, default=0.55)
    synthetic.add_argument("--m-event", type=float, default=0.02)
    synthetic.add_argument("--lambda-id", type=float, default=1.0)
    synthetic.add_argument("--lambda-static", type=float, default=0.5)
    synthetic.add_argument("--lambda-budget", type=float, default=0.05)

    template = subparsers.add_parser("write-real-template", help="Write a predeclaration template for real windows.")
    template.add_argument("--out", default="refine-logs/hide_reveal_real_windows_template.json")

    from_eval = subparsers.add_parser(
        "real-manifest-from-eval",
        help="Create a real-window manifest skeleton from ADAGS eval folders with renders/ and gt/.",
    )
    from_eval.add_argument(
        "--eval-root",
        action="append",
        required=True,
        help="Eval folder containing renders/ and gt/, or a parent directory to scan. Repeatable.",
    )
    from_eval.add_argument("--out", default="refine-logs/hide_reveal_real_windows.json")
    from_eval.add_argument("--system-name", default="route0", help="System name for discovered eval folders.")
    from_eval.add_argument("--scene", help="Override inferred scene name, useful when passing one eval folder.")
    from_eval.add_argument("--num-windows", type=int, default=6)
    from_eval.add_argument("--window-length", type=int, default=16)
    from_eval.add_argument("--crop-xyxy", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    from_eval.add_argument("--frame-start", type=int, help="Direct CLI window start frame, inclusive.")
    from_eval.add_argument("--frame-end", type=int, help="Direct CLI window end frame, inclusive.")
    from_eval.add_argument("--occluder", default="TBD_PREDECLARE")
    from_eval.add_argument("--notes", default="Review and freeze this candidate window before scoring.")
    from_eval.add_argument("--windows-json", help="JSON manifest/window list with user-predeclared windows.")
    from_eval.add_argument("--windows-csv", help="CSV with scene, frame_start, frame_end, crop_xyxy or x0/y0/x1/y1.")
    from_eval.add_argument("--max-depth", type=int, default=5)
    from_eval.add_argument("--skip-validate", action="store_true")

    augment = subparsers.add_parser(
        "augment-real-manifest-system",
        help="Add a rendered system from eval folders to existing frozen real windows.",
    )
    augment.add_argument("--manifest", required=True)
    augment.add_argument(
        "--eval-root",
        action="append",
        required=True,
        help="Eval folder containing renders/ and gt/, or a parent directory to scan. Repeatable.",
    )
    augment.add_argument("--system-name", required=True)
    augment.add_argument("--out", required=True)
    augment.add_argument(
        "--merge-manifest",
        action="append",
        help="Optional manifest whose systems should be merged by matching window_id before adding the new system.",
    )
    augment.add_argument("--max-depth", type=int, default=5)
    augment.add_argument("--skip-validate", action="store_true")

    validate = subparsers.add_parser(
        "validate-real-manifest",
        help="Validate real-window manifest paths, frame coverage, and crop bounds.",
    )
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--require-system", action="append", help="System that must be present in every window.")
    validate.add_argument("--out", help="Optional path to write validation JSON.")
    validate.add_argument("--quiet", action="store_true")

    real = subparsers.add_parser("real-eval", help="Evaluate predeclared real event windows from rendered frames.")
    real.add_argument("--manifest", required=True)
    real.add_argument("--out-dir", default="refine-logs/hide_reveal_poc/real")
    real.add_argument("--compute-lpips", action="store_true")

    derive = subparsers.add_parser(
        "derive-real-renders",
        help="Create R012/R013 derived PoC render folders from route0 frames and a frozen real-window manifest.",
    )
    derive.add_argument("--manifest", required=True)
    derive.add_argument("--out-dir", default="refine-logs/hide_reveal_poc/derived_real_renders")
    derive.add_argument(
        "--route0-eval",
        help="Optional eval folder with renders/ and gt/ to use as route0 for every manifest window.",
    )
    derive.add_argument("--route0-system", default="route0")
    derive.add_argument("--hide-reveal-strength", type=float, default=1.0)
    derive.add_argument("--matched-lifespan-strength", type=float, default=0.35)
    derive.add_argument("--event-beta", type=float, default=1.0)
    derive.add_argument("--feather-px", type=int, default=8)
    derive.add_argument("--overwrite", action="store_true")
    derive.add_argument(
        "--run-eval",
        action="store_true",
        help="Immediately run real-eval on the derived manifest after writing render folders.",
    )
    derive.add_argument("--eval-out-dir", default="refine-logs/hide_reveal_poc/real")

    actual = subparsers.add_parser(
        "actual-real-renders",
        help="Render checkpoint-backed R017 runtime hide/reveal outputs for frozen real windows.",
    )
    actual.add_argument("--manifest", required=True)
    actual.add_argument("--out-dir", default="refine-logs/hide_reveal_poc/r017_actual_real_renders")
    actual.add_argument("--eval-out-dir", default="refine-logs/hide_reveal_poc/r017_actual_real_eval")
    actual.add_argument("--residual-manifest", help="Manifest containing residual_uncertainty baseline paths.")
    actual.add_argument("--matched-manifest", help="Manifest containing matched_lifespan baseline paths.")
    actual.add_argument("--route0-system", default="route0")
    actual.add_argument("--actual-system", default="actual_hide_reveal")
    actual.add_argument("--opacity-attenuation", type=float, default=0.95)
    actual.add_argument("--dynamic-probability-min", type=float, default=0.55)
    actual.add_argument("--event-beta", type=float, default=1.0)
    actual.add_argument("--overwrite", action="store_true")
    actual.add_argument("--compute-lpips", action="store_true")

    candidates = subparsers.add_parser(
        "nonoracle-candidates",
        help="Discover non-oracle event-support candidate boxes without frozen crop labels.",
    )
    candidates.add_argument("--manifest", required=True)
    candidates.add_argument("--out-dir", default="refine-logs/hide_reveal_poc/r018_nonoracle_candidates")
    candidates.add_argument("--route0-system", default="route0")
    candidates.add_argument("--window-length", type=int, default=16)
    candidates.add_argument("--temporal-stride", type=int, default=4)
    candidates.add_argument("--tile-size", type=int, default=160)
    candidates.add_argument("--tile-stride", type=int, default=80)
    candidates.add_argument("--top-k-per-scene", type=int, default=8)
    candidates.add_argument("--crop-iou-threshold", type=float, default=0.5)
    candidates.add_argument("--temporal-iou-threshold", type=float, default=0.5)

    boundary = subparsers.add_parser(
        "event-boundary-support",
        help="Build M2 non-oracle occlusion-boundary support masks without frozen crop labels.",
    )
    boundary.add_argument("--manifest", required=True)
    boundary.add_argument("--out-dir", default="refine-logs/hide_reveal_poc/r026_m2_boundary_support")
    boundary.add_argument("--route0-system", default="route0")
    boundary.add_argument("--max-components-per-scene", type=int, default=36)
    boundary.add_argument("--max-pixel-fraction", type=float, default=0.03)
    boundary.add_argument("--boundary-dilate", type=int, default=6)
    boundary.add_argument("--min-component-area", type=int, default=16)
    boundary.add_argument("--min-score", type=float, default=0.05)
    boundary.add_argument("--no-flow", action="store_true", help="Do not use flow sidecars even if present.")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "synthetic":
        params = FrozenHideRevealParams(
            c_min=args.c_min,
            m_event=args.m_event,
            lambda_id=args.lambda_id,
            lambda_static=args.lambda_static,
            lambda_budget=args.lambda_budget,
        )
        summary = run_synthetic_poc(args.seeds, args.clips_per_type, Path(args.out_dir), params=params)
        heldout = summary["summary"]["heldout"]
        stop_go = summary["stop_go"]
        print(f"Wrote synthetic PoC outputs to {Path(args.out_dir).resolve()}")
        print(f"heldout margin_auc={heldout.get('margin_auc')}")
        print(f"heldout candidate_recall={heldout.get('candidate_recall')}")
        print(f"heldout identity_reconnection_accuracy={heldout.get('identity_reconnection_accuracy')}")
        print(
            "heldout matched_lifespan_identity_reconnection_accuracy="
            f"{heldout.get('matched_lifespan_identity_reconnection_accuracy')}"
        )
        print(
            "heldout no_identity_identity_reconnection_accuracy="
            f"{heldout.get('no_identity_identity_reconnection_accuracy')}"
        )
        print(f"proceed_to_real_windows={stop_go.get('proceed_to_real_windows')}")
    elif args.command == "write-real-template":
        out = Path(args.out)
        write_real_manifest_template(out)
        print(f"Wrote real-window template to {out.resolve()}")
    elif args.command == "real-manifest-from-eval":
        window_specs = load_window_specs(
            Path(args.windows_json) if args.windows_json else None,
            Path(args.windows_csv) if args.windows_csv else None,
        )
        direct_spec = direct_window_spec(
            args.scene,
            args.frame_start,
            args.frame_end,
            args.crop_xyxy,
            args.occluder,
            args.notes,
        )
        if direct_spec is not None:
            window_specs.append(direct_spec)
        result = create_real_manifest_from_eval(
            [Path(path) for path in args.eval_root],
            Path(args.out),
            system_name=args.system_name,
            scene=args.scene,
            num_windows=args.num_windows,
            window_length=args.window_length,
            crop_xyxy=args.crop_xyxy,
            occluder=args.occluder,
            notes=args.notes,
            window_specs=window_specs,
            max_depth=args.max_depth,
        )
        out = Path(result["out_path"])
        print(f"Wrote real-window manifest to {out.resolve()}")
        print(f"discovered_eval_runs={len(result['eval_runs'])}")
        print(f"windows={len(result['manifest']['windows'])}")
        if not args.skip_validate:
            validation = validate_real_manifest(out, require_systems=[args.system_name])
            print(f"validation_ok={validation['ok']}")
            print(f"validation_errors={len(validation['errors'])}")
            print(f"validation_warnings={len(validation['warnings'])}")
            if validation["errors"]:
                for error in validation["errors"][:8]:
                    print(f"ERROR: {error}", file=sys.stderr)
                return 2
    elif args.command == "augment-real-manifest-system":
        result = augment_real_manifest_system(
            manifest_path=Path(args.manifest),
            search_roots=[Path(path) for path in args.eval_root],
            out_path=Path(args.out),
            system_name=args.system_name,
            merge_manifest_paths=[Path(path) for path in args.merge_manifest or []],
            max_depth=args.max_depth,
            validate=not args.skip_validate,
        )
        print(f"Wrote augmented real manifest to {Path(result['out_path']).resolve()}")
        print(f"systems_added={args.system_name}")
        print(f"eval_runs={len(result['eval_runs'])}")
        if result["validation"] is not None:
            print(f"validation_ok={result['validation']['ok']}")
            print(f"validation_errors={len(result['validation']['errors'])}")
            print(f"validation_warnings={len(result['validation']['warnings'])}")
            if result["validation"]["errors"]:
                for error in result["validation"]["errors"][:8]:
                    print(f"ERROR: {error}", file=sys.stderr)
                return 2
    elif args.command == "validate-real-manifest":
        required = args.require_system if args.require_system is not None else ["route0"]
        validation = validate_real_manifest(Path(args.manifest), require_systems=required)
        if args.out:
            write_json(Path(args.out), validation)
        if not args.quiet:
            print(f"validation_ok={validation['ok']}")
            print(f"windows={validation['n_windows']}")
            print(f"errors={len(validation['errors'])}")
            print(f"warnings={len(validation['warnings'])}")
            for warning in validation["warnings"][:8]:
                print(f"WARNING: {warning}")
            for error in validation["errors"][:8]:
                print(f"ERROR: {error}", file=sys.stderr)
        return 0 if validation["ok"] else 2
    elif args.command == "real-eval":
        payload = evaluate_real_manifest(Path(args.manifest), Path(args.out_dir), compute_lpips=args.compute_lpips)
        print(f"Wrote real event-window outputs to {Path(args.out_dir).resolve()}")
        print(f"systems={', '.join(payload['summary'].keys())}")
    elif args.command == "derive-real-renders":
        result = derive_real_poc_render_folders(
            manifest_path=Path(args.manifest),
            out_dir=Path(args.out_dir),
            route0_eval_dir=Path(args.route0_eval) if args.route0_eval else None,
            route0_system=args.route0_system,
            hide_reveal_strength=args.hide_reveal_strength,
            matched_lifespan_strength=args.matched_lifespan_strength,
            event_beta=args.event_beta,
            feather_px=args.feather_px,
            overwrite=args.overwrite,
        )
        print(f"Wrote derived PoC render folders under {Path(args.out_dir).resolve()}")
        print(f"derived_manifest={Path(result['manifest_path']).resolve()}")
        print(f"metadata={Path(result['metadata_path']).resolve()}")
        print(f"validation_ok={result['validation']['ok']}")
        print(f"validation_errors={len(result['validation']['errors'])}")
        if result["validation"]["errors"]:
            for error in result["validation"]["errors"][:8]:
                print(f"ERROR: {error}", file=sys.stderr)
            return 2
        if args.run_eval:
            payload = evaluate_real_manifest(
                Path(result["manifest_path"]),
                Path(args.eval_out_dir),
                compute_lpips=False,
            )
            print(f"Wrote real event-window outputs to {Path(args.eval_out_dir).resolve()}")
            print(f"systems={', '.join(payload['summary'].keys())}")
    elif args.command == "actual-real-renders":
        result = render_actual_hide_reveal_real_windows(
            manifest_path=Path(args.manifest),
            out_dir=Path(args.out_dir),
            residual_manifest_path=Path(args.residual_manifest) if args.residual_manifest else None,
            matched_manifest_path=Path(args.matched_manifest) if args.matched_manifest else None,
            route0_system=args.route0_system,
            actual_system=args.actual_system,
            opacity_attenuation=args.opacity_attenuation,
            dynamic_probability_min=args.dynamic_probability_min,
            event_beta=args.event_beta,
            overwrite=args.overwrite,
            run_eval=True,
            eval_out_dir=Path(args.eval_out_dir),
            compute_lpips=args.compute_lpips,
        )
        print(f"Wrote actual hide/reveal render folders under {Path(args.out_dir).resolve()}")
        print(f"actual_manifest={Path(result['manifest_path']).resolve()}")
        print(f"metadata={Path(result['metadata_path']).resolve()}")
        print(f"validation_ok={result['validation']['ok']}")
        print(f"validation_errors={len(result['validation']['errors'])}")
        if result["validation"]["errors"]:
            for error in result["validation"]["errors"][:8]:
                print(f"ERROR: {error}", file=sys.stderr)
            return 2
        if result.get("eval"):
            print(f"Wrote real event-window outputs to {Path(args.eval_out_dir).resolve()}")
            print(f"systems={', '.join(result['eval']['summary'].keys())}")
    elif args.command == "nonoracle-candidates":
        result = discover_nonoracle_event_candidates(
            manifest_path=Path(args.manifest),
            out_dir=Path(args.out_dir),
            route0_system=args.route0_system,
            window_length=args.window_length,
            temporal_stride=args.temporal_stride,
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            top_k_per_scene=args.top_k_per_scene,
            crop_iou_threshold=args.crop_iou_threshold,
            temporal_iou_threshold=args.temporal_iou_threshold,
        )
        print(f"Wrote non-oracle candidate outputs to {Path(args.out_dir).resolve()}")
        print(f"candidate_manifest={Path(result['manifest_path']).resolve()}")
        print(f"metadata={Path(result['metadata_path']).resolve()}")
        print(f"validation_ok={result['validation']['ok']}")
        print(f"validation_errors={len(result['validation']['errors'])}")
        print(f"candidates={len(result['manifest']['windows'])}")
        if result["validation"]["errors"]:
            for error in result["validation"]["errors"][:8]:
                print(f"ERROR: {error}", file=sys.stderr)
            return 2
    elif args.command == "event-boundary-support":
        result = discover_event_boundary_support(
            manifest_path=Path(args.manifest),
            out_dir=Path(args.out_dir),
            route0_system=args.route0_system,
            max_components_per_scene=args.max_components_per_scene,
            max_pixel_fraction=args.max_pixel_fraction,
            boundary_dilate=args.boundary_dilate,
            min_component_area=args.min_component_area,
            min_score=args.min_score,
            use_flow=not args.no_flow,
        )
        print(f"Wrote M2 event-boundary support outputs to {Path(args.out_dir).resolve()}")
        print(f"support_manifest={Path(result['manifest_path']).resolve()}")
        print(f"metadata={Path(result['metadata_path']).resolve()}")
        print(f"validation_ok={result['validation']['ok']}")
        print(f"validation_errors={len(result['validation']['errors'])}")
        print(f"support_frames={len(result['manifest']['support_frames'])}")
        if result["validation"]["errors"]:
            for error in result["validation"]["errors"][:8]:
                print(f"ERROR: {error}", file=sys.stderr)
            return 2
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
