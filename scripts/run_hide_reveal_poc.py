#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hide_reveal_poc import (
    FrozenHideRevealParams,
    evaluate_real_manifest,
    run_synthetic_poc,
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
    elif args.command == "real-eval":
        payload = evaluate_real_manifest(Path(args.manifest), Path(args.out_dir), compute_lpips=args.compute_lpips)
        print(f"Wrote real event-window outputs to {Path(args.out_dir).resolve()}")
        print(f"systems={', '.join(payload['summary'].keys())}")
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
