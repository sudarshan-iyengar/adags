#!/usr/bin/env python3
"""Determined (`det`) status/monitoring CLI for ADAGS Apollo runs.

Authority: Determined-status utility task spec (2026-08-14), which also
required this module to become the single caller-facing entrypoint for the
read-only `det` queries `elgs.determined` implements: wrapper resolution
(`resolve`), single-shot status (`experiment`, `task`), bounded polling
(`watch`), and a live smoke check (`selftest`).

Every subcommand is a thin `cmd_*` wrapper -- the only functions in this
file that print -- over the pure, independently-testable functions in
`elgs.determined` (mirroring `scripts/submit_apollo.py`'s
`cmd_*`-orchestrates-pure-functions style). All output is JSON on stdout.

Style: fail-closed. Every user-facing failure raises a
`depth_visibility.errors.ContractError` (or a subclass); `main()` catches it
at the top level, prints `ERROR: ...` to stderr, and exits 2 -- the same
convention `scripts/submit_apollo.py` uses.

This CLI never creates, kills, or otherwise mutates a Determined
experiment/task: every subcommand routes only to `det version`, `det
experiment describe --json`, or `det task list --json`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402
from elgs import determined  # noqa: E402


# ---------------------------------------------------------------------------
# JSON printing helper
# ---------------------------------------------------------------------------


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _as_dict(value: Any) -> Any:
    """`dataclasses.asdict`, tolerant of plain dicts already passed in."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


def cmd_resolve(args: argparse.Namespace) -> int:
    command = determined.resolve_det_command(args.det_path)
    _print_json(_as_dict(command))
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    command = determined.resolve_det_command(args.det_path)
    status = determined.experiment_status(
        command, args.experiment_id, timeout_seconds=args.timeout_seconds
    )
    _print_json(_as_dict(status))
    return 0


def cmd_task(args: argparse.Namespace) -> int:
    command = determined.resolve_det_command(args.det_path)
    status = determined.task_status(command, args.task_id, timeout_seconds=args.timeout_seconds)
    _print_json(_as_dict(status))
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    command = determined.resolve_det_command(args.det_path)
    result = determined.poll_until_terminal(
        command,
        args.kind,
        args.id,
        interval_seconds=args.interval_seconds,
        max_polls=args.max_polls,
        tolerate_transient=args.tolerate_transient,
        timeout_seconds=args.timeout_seconds,
    )
    _print_json(_as_dict(result))
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Resolve the wrapper, run `det version`, and assert parseable
    client+master versions. Fails loudly (raises `ContractError`, exit 2
    via `main()`) at the first thing that is wrong, rather than printing a
    partial report and exiting 0."""

    report: dict[str, Any] = {}

    command = determined.resolve_det_command(args.det_path)
    report["command"] = _as_dict(command)

    invocation = determined.run_det(command, ["version"], timeout_seconds=args.timeout_seconds)
    determined.check_invocation(invocation)
    report["invocation"] = _as_dict(invocation)

    versions = determined.parse_version_output(invocation.stdout)
    report["versions"] = versions
    report["ok"] = True

    _print_json(report)
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def _add_det_path_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--det-path",
        default=None,
        help="explicit path to the det CLI (default: resolve via PATH)",
    )


def _add_timeout_arg(parser: argparse.ArgumentParser, *, default: float = 120.0) -> None:
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=default,
        help=f"per-invocation subprocess timeout in seconds (default: {default})",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="det_monitor.py",
        description="Read-only Determined (det) status/monitoring CLI for ADAGS Apollo runs.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    resolve = subparsers.add_parser("resolve", help="resolve and classify the det CLI; print as JSON")
    _add_det_path_arg(resolve)
    resolve.set_defaults(func=cmd_resolve)

    experiment = subparsers.add_parser(
        "experiment", help="`det experiment describe --json` -> parsed DetStatus"
    )
    _add_det_path_arg(experiment)
    _add_timeout_arg(experiment)
    experiment.add_argument("--experiment-id", required=True, help="numeric Determined experiment id")
    experiment.set_defaults(func=cmd_experiment)

    task = subparsers.add_parser("task", help="`det task list --json` -> parsed DetStatus for one task")
    _add_det_path_arg(task)
    _add_timeout_arg(task)
    task.add_argument("--task-id", required=True, help="Determined task id (UUID-shaped)")
    task.set_defaults(func=cmd_task)

    watch = subparsers.add_parser(
        "watch", help="bounded polling loop until a terminal state or --max-polls is exhausted"
    )
    _add_det_path_arg(watch)
    _add_timeout_arg(watch)
    watch.add_argument("--kind", required=True, choices=("experiment", "task"))
    watch.add_argument("--id", required=True, help="experiment id or task id, matching --kind")
    watch.add_argument("--interval-seconds", required=True, type=float, help="sleep between polls")
    watch.add_argument("--max-polls", required=True, type=int, help="hard cap on the number of polls")
    watch.add_argument(
        "--tolerate-transient",
        type=int,
        default=0,
        help="number of invocation/parse/unknown-state failures to tolerate as transient (default: 0)",
    )
    watch.set_defaults(func=cmd_watch)

    selftest = subparsers.add_parser(
        "selftest", help="resolve the wrapper, run `det version`, assert parseable versions"
    )
    _add_det_path_arg(selftest)
    _add_timeout_arg(selftest)
    selftest.set_defaults(func=cmd_selftest)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
