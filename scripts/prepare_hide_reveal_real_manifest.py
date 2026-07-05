#!/usr/bin/env python
"""Prepare and validate real-window manifests for the hide/reveal PoC."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hide_reveal_poc import (
    create_real_manifest_from_eval,
    direct_window_spec,
    load_window_specs,
    validate_real_manifest,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create or validate the R009 predeclared real-window manifest from "
            "ADAGS eval folders containing renders/ and gt/."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a manifest skeleton from one or more eval roots.")
    create.add_argument(
        "--eval-root",
        action="append",
        required=True,
        help="Eval folder containing renders/ and gt/, or a parent directory to scan. Repeatable.",
    )
    create.add_argument("--out", default="refine-logs/hide_reveal_real_windows.json")
    create.add_argument("--system-name", default="route0")
    create.add_argument("--scene", help="Override inferred scene name when passing a single eval folder.")
    create.add_argument("--num-windows", type=int, default=6)
    create.add_argument("--window-length", type=int, default=16)
    create.add_argument("--crop-xyxy", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    create.add_argument("--frame-start", type=int, help="Direct window start frame, inclusive.")
    create.add_argument("--frame-end", type=int, help="Direct window end frame, inclusive.")
    create.add_argument("--occluder", default="TBD_PREDECLARE")
    create.add_argument("--notes", default="Review and freeze this candidate window before scoring.")
    create.add_argument("--windows-json", help="JSON manifest/window list with user-predeclared windows.")
    create.add_argument("--windows-csv", help="CSV with scene, frame_start, frame_end and crop coordinates.")
    create.add_argument("--max-depth", type=int, default=5)
    create.add_argument("--skip-validate", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate paths, frame coverage, and crop bounds.")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--require-system", action="append", help="System that must exist in every window.")
    validate.add_argument("--out", help="Optional validation JSON path.")
    validate.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def create_manifest(args: argparse.Namespace) -> int:
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

    if args.skip_validate:
        return 0
    validation = validate_real_manifest(out, require_systems=[args.system_name])
    print(f"validation_ok={validation['ok']}")
    print(f"validation_errors={len(validation['errors'])}")
    print(f"validation_warnings={len(validation['warnings'])}")
    for warning in validation["warnings"][:8]:
        print(f"WARNING: {warning}")
    for error in validation["errors"][:8]:
        print(f"ERROR: {error}", file=sys.stderr)
    return 0 if validation["ok"] else 2


def validate_manifest(args: argparse.Namespace) -> int:
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


def main() -> int:
    args = parse_args()
    if args.command == "create":
        return create_manifest(args)
    if args.command == "validate":
        return validate_manifest(args)
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
