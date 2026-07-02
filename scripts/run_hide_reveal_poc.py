#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hide_reveal_poc import (
    FrozenHideRevealParams,
    create_real_manifest_from_eval,
    direct_window_spec,
    evaluate_real_manifest,
    load_window_specs,
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
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
