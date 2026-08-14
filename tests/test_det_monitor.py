"""Unit tests for elgs/determined.py and scripts/det_monitor.py's pure surface.

Run with:
    C:/Users/sucar/anaconda3/python.exe -m unittest tests.test_det_monitor

CPU-only, no live Determined cluster needed: every test injects a fake
`runner` / `which` / `sleeper` (dependency injection, per elgs.determined's
own signatures) rather than shelling out to a real `det` or monkeypatching
module globals. A handful of literal JSON/YAML fixtures below were captured
verbatim (truncated where noted) from a real probe of the live cluster
(Determined CLI 0.38.1 / master 0.38.0) on 2026-08-14, so the parsing tests
exercise the ACTUAL shapes those subcommands return, not assumed ones.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import (  # noqa: E402
    ContractError,
    DetInvocationError,
    DetParseError,
    DetUnknownStateError,
)
from elgs import determined  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------


#: The literal stderr string `det` was VERIFIED to write on EVERY invocation
#: (success or failure) on this machine -- a version-skew warning, not a
#: failure signal.
REAL_STDERR_WARNING = (
    "Master version 0.38.0 is less than CLI version 0.38.1. Consider upgrading the master."
)

#: `det experiment describe 14 --json` real shape, VERIFIED 2026-08-14:
#: a JSON ARRAY with one object whose "experiment" sub-object carries
#: "state": "STATE_COMPLETED" (irrelevant keys omitted for brevity).
REAL_EXPERIMENT_DESCRIBE_LIST_SHAPE = json.dumps(
    [{"config": {}, "experiment": {"id": 14, "state": "STATE_COMPLETED"}, "jobSummary": {}}]
)

#: `det task list --json` real shape, VERIFIED 2026-08-14: a JSON OBJECT
#: keyed by "<taskId>.<allocationNumber>", records carrying a "taskId"
#: field but NO "state" field at all.
REAL_TASK_LIST_SHAPE = json.dumps(
    {
        "4fe6d93e-113c-4038-a79d-0ac8c9ae16bb.1": {
            "taskId": "4fe6d93e-113c-4038-a79d-0ac8c9ae16bb",
            "name": "JupyterLab (optionally-elegant-sawfish)",
            "resourcePool": "hopper",
        },
        "c03e2932-a054-491b-9893-126afee8122f.1": {
            "taskId": "c03e2932-a054-491b-9893-126afee8122f",
            "name": "Command (firstly-working-pup)",
            "resourcePool": "hopper",
        },
    }
)

#: `det version` real output, VERIFIED 2026-08-14: YAML, not JSON.
REAL_VERSION_OUTPUT = """client:
  version: 0.38.1
master:
  cluster_id: 7285267c-bd65-4d31-91ff-9ea5da7a8542
  cluster_name: ''
  master_id: 970461bd-ae45-47fa-9b4a-69e22e134c89
  sso_providers: null
  telemetry:
    enabled: true
    otel_enabled: false
    otel_endpoint: ''
    segment_key: Ryn8Uh9BYKJ4m9irA3MCzxcfHuB3CaaF
  version: 0.38.0
master_address: http://determined.intern.denayer.be:8080
"""


class _Completed:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _ScriptedRunner:
    """Fake `runner` callable: returns (or raises) one scripted response per
    call, in order, and records every call's argv+kwargs for inspection."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        if not self._responses:
            raise AssertionError(
                f"_ScriptedRunner called more times ({len(self.calls)}) than scripted "
                f"({len(self.calls) - 1} responses provided)"
            )
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _RecordingSleeper:
    def __init__(self):
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _fake_command(kind: str = "executable") -> determined.DetCommand:
    return determined.DetCommand(
        resolved_path="det", kind=kind, interpreter=None, argv_prefix=("det",)
    )


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


# ---------------------------------------------------------------------------
# resolve_det_command
# ---------------------------------------------------------------------------


class ResolveDetCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="det-monitor-resolve-")
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_exe_suffix_classified_executable_by_suffix_alone(self):
        # Empty file: classification must succeed purely from the .exe
        # suffix, without needing valid PE bytes -- matches the spec's
        # documented check order (suffix first).
        path = self.dir / "det.exe"
        _write(path, b"")
        command = determined.resolve_det_command(which=lambda name: str(path))
        self.assertEqual(command.kind, "executable")
        self.assertIsNone(command.interpreter)
        self.assertEqual(command.argv_prefix, (str(path),))
        self.assertEqual(command.resolved_path, str(path))

    def test_bat_and_cmd_suffixes_also_classified_executable(self):
        for suffix in (".bat", ".cmd", ".com"):
            path = self.dir / f"det{suffix}"
            _write(path, b"")
            command = determined.resolve_det_command(which=lambda name, p=path: str(p))
            self.assertEqual(command.kind, "executable", msg=suffix)

    def test_pe_magic_bytes_classified_executable(self):
        path = self.dir / "det"  # extensionless, matches the real machine's... no wrapper case
        _write(path, b"MZ" + b"\x90\x00" * 30)
        command = determined.resolve_det_command(which=lambda name: str(path))
        self.assertEqual(command.kind, "executable")
        self.assertEqual(command.argv_prefix, (str(path),))

    def test_elf_magic_bytes_classified_executable(self):
        path = self.dir / "det"
        _write(path, b"\x7fELF" + b"\x00" * 30)
        command = determined.resolve_det_command(which=lambda name: str(path))
        self.assertEqual(command.kind, "executable")

    def test_python_shebang_extensionless_classified_python_script(self):
        path = self.dir / "det"  # extensionless -- the cross-platform case the spec requires
        _write(path, b"#!/usr/bin/env python3\nimport sys\nfrom determined.cli import main\n")
        command = determined.resolve_det_command(which=lambda name: str(path))
        self.assertEqual(command.kind, "python-script")
        self.assertEqual(command.interpreter, sys.executable)
        self.assertEqual(command.argv_prefix, (sys.executable, str(path)))

    def test_shebang_python_match_is_case_insensitive(self):
        path = self.dir / "det"
        _write(path, b"#!/usr/bin/env PYTHON3\n")
        command = determined.resolve_det_command(which=lambda name: str(path))
        self.assertEqual(command.kind, "python-script")

    def test_dot_py_suffix_without_shebang_classified_python_script(self):
        path = self.dir / "det.py"
        _write(path, b"import sys\nfrom determined.cli import main\n")
        command = determined.resolve_det_command(which=lambda name: str(path))
        self.assertEqual(command.kind, "python-script")
        self.assertEqual(command.interpreter, sys.executable)
        self.assertEqual(command.argv_prefix, (sys.executable, str(path)))

    def test_garbage_binary_file_raises_contract_error(self):
        path = self.dir / "det"
        _write(path, b"\x00\x01\x02\x03garbage-not-a-known-format\xff\xfe")
        with self.assertRaises(ContractError):
            determined.resolve_det_command(which=lambda name: str(path))

    def test_shebang_without_python_raises_contract_error(self):
        path = self.dir / "det"
        _write(path, b"#!/bin/sh\necho hi\n")
        with self.assertRaises(ContractError):
            determined.resolve_det_command(which=lambda name: str(path))

    def test_which_returning_none_raises_contract_error_naming_path(self):
        with self.assertRaises(ContractError) as ctx:
            determined.resolve_det_command(which=lambda name: None)
        self.assertIn("PATH", str(ctx.exception))

    def test_explicit_path_that_does_not_exist_raises(self):
        missing = self.dir / "does-not-exist"
        with self.assertRaises(ContractError):
            determined.resolve_det_command(str(missing))

    def test_explicit_path_overrides_which(self):
        path = self.dir / "det.exe"
        _write(path, b"")

        def _boom(name):
            raise AssertionError("which() must not be called when explicit_path is given")

        command = determined.resolve_det_command(str(path), which=_boom)
        self.assertEqual(command.resolved_path, str(path))


# ---------------------------------------------------------------------------
# run_det / check_invocation
# ---------------------------------------------------------------------------


class RunDetTests(unittest.TestCase):
    def test_successful_invocation_builds_correct_argv_and_fields(self):
        runner = _ScriptedRunner([_Completed(returncode=0, stdout="ok\n", stderr="")])
        command = _fake_command()
        inv = determined.run_det(command, ["version"], runner=runner)
        self.assertEqual(inv.argv, ("det", "version"))
        self.assertEqual(inv.returncode, 0)
        self.assertEqual(inv.stdout, "ok\n")
        self.assertGreaterEqual(inv.duration_seconds, 0.0)
        # never shell=True
        self.assertEqual(runner.calls[0][1].get("shell"), False)
        self.assertEqual(runner.calls[0][0], ["det", "version"])

    def test_python_script_command_prefixes_interpreter(self):
        command = determined.DetCommand(
            resolved_path="/x/det", kind="python-script", interpreter=sys.executable,
            argv_prefix=(sys.executable, "/x/det"),
        )
        runner = _ScriptedRunner([_Completed(returncode=0)])
        determined.run_det(command, ["task", "list", "--json"], runner=runner)
        self.assertEqual(runner.calls[0][0], [sys.executable, "/x/det", "task", "list", "--json"])

    def test_stderr_warning_present_rc_zero_is_not_a_failure(self):
        runner = _ScriptedRunner(
            [_Completed(returncode=0, stdout=REAL_EXPERIMENT_DESCRIBE_LIST_SHAPE, stderr=REAL_STDERR_WARNING)]
        )
        command = _fake_command()
        inv = determined.run_det(command, ["experiment", "describe", "14", "--json"], runner=runner)
        determined.check_invocation(inv)  # must not raise
        self.assertEqual(inv.stderr, REAL_STDERR_WARNING)

    def test_oserror_from_runner_becomes_det_invocation_error(self):
        runner = _ScriptedRunner([OSError("no such file or directory")])
        command = _fake_command()
        with self.assertRaises(DetInvocationError) as ctx:
            determined.run_det(command, ["version"], runner=runner)
        self.assertIn("det", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, OSError)

    def test_timeout_expired_from_runner_becomes_det_invocation_error(self):
        timeout_exc = subprocess.TimeoutExpired(cmd=["det", "version"], timeout=5)
        runner = _ScriptedRunner([timeout_exc])
        command = _fake_command()
        with self.assertRaises(DetInvocationError) as ctx:
            determined.run_det(command, ["version"], runner=runner, timeout_seconds=5)
        self.assertIsInstance(ctx.exception.__cause__, subprocess.TimeoutExpired)

    def test_none_timeout_passed_through_as_none(self):
        captured = {}

        def runner(argv, **kwargs):
            captured.update(kwargs)
            return _Completed(returncode=0)

        determined.run_det(_fake_command(), ["experiment", "logs", "1", "-f"], runner=runner, timeout_seconds=None)
        self.assertIsNone(captured["timeout"])

    def test_check_invocation_nonzero_rc_message_has_rc_and_stderr_and_stdout(self):
        inv = determined.DetInvocation(
            argv=("det", "experiment", "describe", "999999", "--json"),
            returncode=1,
            stdout="",
            stderr=REAL_STDERR_WARNING + "\nFailed to describe experiment: experiment '999999' not found",
            duration_seconds=0.01,
        )
        with self.assertRaises(DetInvocationError) as ctx:
            determined.check_invocation(inv)
        message = str(ctx.exception)
        self.assertIn("1", message)
        self.assertIn("not found", message)

    def test_check_invocation_zero_rc_does_not_raise(self):
        inv = determined.DetInvocation(
            argv=("det",), returncode=0, stdout="x", stderr="y", duration_seconds=0.01
        )
        determined.check_invocation(inv)  # must not raise


# ---------------------------------------------------------------------------
# classify_state
# ---------------------------------------------------------------------------


class ClassifyStateTests(unittest.TestCase):
    def test_every_declared_terminal_state_classifies_terminal(self):
        for state in determined.TERMINAL_STATES:
            self.assertEqual(determined.classify_state(state), "terminal", msg=state)

    def test_every_declared_nonterminal_state_classifies_nonterminal(self):
        for state in determined.NONTERMINAL_STATES:
            self.assertEqual(determined.classify_state(state), "nonterminal", msg=state)

    def test_state_prefix_normalisation(self):
        self.assertEqual(determined.classify_state("COMPLETED"), "terminal")
        self.assertEqual(determined.classify_state("STATE_COMPLETED"), "terminal")
        self.assertEqual(determined.classify_state("RUNNING"), "nonterminal")
        self.assertEqual(determined.classify_state("STATE_RUNNING"), "nonterminal")

    def test_case_insensitivity(self):
        self.assertEqual(determined.classify_state("completed"), "terminal")
        self.assertEqual(determined.classify_state("Completed"), "terminal")
        self.assertEqual(determined.classify_state("state_running"), "nonterminal")

    def test_unknown_state_raises_det_unknown_state_error(self):
        with self.assertRaises(DetUnknownStateError):
            determined.classify_state("SOMETHING_MADE_UP")

    def test_empty_or_non_string_state_raises(self):
        with self.assertRaises(DetUnknownStateError):
            determined.classify_state("")
        with self.assertRaises(DetUnknownStateError):
            determined.classify_state(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_experiment_state
# ---------------------------------------------------------------------------


class ParseExperimentStateTests(unittest.TestCase):
    def test_real_live_list_wrapped_shape(self):
        # VERIFIED live shape: a JSON array containing one object.
        state = determined.parse_experiment_state(REAL_EXPERIMENT_DESCRIBE_LIST_SHAPE)
        self.assertEqual(state, "STATE_COMPLETED")

    def test_bare_dict_with_experiment_state(self):
        payload = json.dumps({"experiment": {"state": "STATE_RUNNING"}})
        self.assertEqual(determined.parse_experiment_state(payload), "STATE_RUNNING")

    def test_bare_dict_with_top_level_state(self):
        payload = json.dumps({"state": "COMPLETED"})
        self.assertEqual(determined.parse_experiment_state(payload), "COMPLETED")

    def test_experiment_state_preferred_over_top_level_when_both_present(self):
        payload = json.dumps({"experiment": {"state": "STATE_RUNNING"}, "state": "COMPLETED"})
        self.assertEqual(determined.parse_experiment_state(payload), "STATE_RUNNING")

    def test_not_json_raises_det_parse_error(self):
        with self.assertRaises(DetParseError):
            determined.parse_experiment_state("not json at all")

    def test_json_without_state_raises(self):
        with self.assertRaises(DetParseError):
            determined.parse_experiment_state(json.dumps({"experiment": {"id": 14}}))

    def test_json_state_null_raises(self):
        with self.assertRaises(DetParseError):
            determined.parse_experiment_state(json.dumps({"state": None}))

    def test_json_state_empty_string_raises(self):
        with self.assertRaises(DetParseError):
            determined.parse_experiment_state(json.dumps({"state": ""}))

    def test_empty_list_raises(self):
        with self.assertRaises(DetParseError):
            determined.parse_experiment_state("[]")

    def test_non_object_non_array_json_raises(self):
        with self.assertRaises(DetParseError):
            determined.parse_experiment_state("42")


# ---------------------------------------------------------------------------
# parse_task_state
# ---------------------------------------------------------------------------


class ParseTaskStateTests(unittest.TestCase):
    def test_real_live_shape_no_state_field_returns_synthetic_active(self):
        state = determined.parse_task_state(
            REAL_TASK_LIST_SHAPE, "4fe6d93e-113c-4038-a79d-0ac8c9ae16bb"
        )
        self.assertEqual(state, "ACTIVE")

    def test_second_record_in_dict_shape_also_resolves(self):
        state = determined.parse_task_state(
            REAL_TASK_LIST_SHAPE, "c03e2932-a054-491b-9893-126afee8122f"
        )
        self.assertEqual(state, "ACTIVE")

    def test_list_of_records_shape_tolerated(self):
        payload = json.dumps([{"taskId": "abc12345", "name": "x"}])
        self.assertEqual(determined.parse_task_state(payload, "abc12345"), "ACTIVE")

    def test_record_with_explicit_state_field_returned_verbatim(self):
        payload = json.dumps({"k.1": {"taskId": "abc12345", "state": "STATE_RUNNING"}})
        self.assertEqual(determined.parse_task_state(payload, "abc12345"), "STATE_RUNNING")

    def test_no_matching_record_raises_det_parse_error(self):
        with self.assertRaises(DetParseError):
            determined.parse_task_state(REAL_TASK_LIST_SHAPE, "0000000000000000000000000000000000")

    def test_not_json_raises(self):
        with self.assertRaises(DetParseError):
            determined.parse_task_state("not json", "abc12345")

    def test_neither_dict_nor_list_raises(self):
        with self.assertRaises(DetParseError):
            determined.parse_task_state("42", "abc12345")


# ---------------------------------------------------------------------------
# parse_version_output
# ---------------------------------------------------------------------------


class ParseVersionOutputTests(unittest.TestCase):
    def test_real_live_yaml_output(self):
        versions = determined.parse_version_output(REAL_VERSION_OUTPUT)
        self.assertEqual(versions["client_version"], "0.38.1")
        self.assertEqual(versions["master_version"], "0.38.0")

    def test_not_yaml_mapping_raises(self):
        with self.assertRaises(DetParseError):
            determined.parse_version_output("- just\n- a\n- list\n")

    def test_missing_versions_raises(self):
        with self.assertRaises(DetParseError):
            determined.parse_version_output("client:\n  name: x\n")


# ---------------------------------------------------------------------------
# validate_experiment_id / validate_task_id
# ---------------------------------------------------------------------------


class ValidateIdentifierTests(unittest.TestCase):
    _SHARED_BAD_VALUES = ("", "abc", "12; rm -rf /", "../../etc/passwd")

    def test_valid_experiment_id_passes_through(self):
        self.assertEqual(determined.validate_experiment_id("14"), "14")
        self.assertEqual(determined.validate_experiment_id(14), "14")

    def test_valid_task_id_passes_through(self):
        tid = "4fe6d93e-113c-4038-a79d-0ac8c9ae16bb"
        self.assertEqual(determined.validate_task_id(tid), tid)

    def test_shared_bad_values_rejected_by_both_validators(self):
        for bad in self._SHARED_BAD_VALUES:
            with self.assertRaises(ContractError, msg=f"experiment_id accepted {bad!r}"):
                determined.validate_experiment_id(bad)
            with self.assertRaises(ContractError, msg=f"task_id accepted {bad!r}"):
                determined.validate_task_id(bad)

    def test_200_char_string_rejected_by_task_id_length_bound(self):
        overlong = ("a1b2c3-" * 30)[:200]
        self.assertEqual(len(overlong), 200)
        with self.assertRaises(ContractError):
            determined.validate_task_id(overlong)

    def test_200_char_non_digit_string_rejected_by_experiment_id(self):
        overlong = ("a1b2c3-" * 30)[:200]
        with self.assertRaises(ContractError):
            determined.validate_experiment_id(overlong)

    def test_experiment_id_has_no_length_cap_for_all_digit_strings(self):
        # Documents a deliberate asymmetry: ^[0-9]+$ has no length bound,
        # unlike task id's {7,63} bound. A very long numeric string is a
        # syntactically valid experiment id even though it's an unlikely one.
        long_numeric = "1" * 200
        self.assertEqual(determined.validate_experiment_id(long_numeric), long_numeric)

    def test_task_id_too_short_rejected(self):
        with self.assertRaises(ContractError):
            determined.validate_task_id("abc123")  # 6 chars, min is 8

    def test_task_id_bad_leading_character_rejected(self):
        with self.assertRaises(ContractError):
            determined.validate_task_id("-abc12345")


# ---------------------------------------------------------------------------
# experiment_status / task_status
# ---------------------------------------------------------------------------


class ExperimentStatusTests(unittest.TestCase):
    def test_success_builds_det_status(self):
        runner = _ScriptedRunner(
            [_Completed(returncode=0, stdout=REAL_EXPERIMENT_DESCRIBE_LIST_SHAPE, stderr=REAL_STDERR_WARNING)]
        )
        status = determined.experiment_status(_fake_command(), 14, runner=runner)
        self.assertEqual(status.kind, "experiment")
        self.assertEqual(status.identifier, "14")
        self.assertEqual(status.state, "STATE_COMPLETED")
        self.assertTrue(status.terminal)
        self.assertEqual(runner.calls[0][0], ["det", "experiment", "describe", "14", "--json"])

    def test_invalid_experiment_id_never_reaches_runner(self):
        def _boom(argv, **kwargs):
            raise AssertionError("runner must not be called for an invalid id")

        with self.assertRaises(ContractError):
            determined.experiment_status(_fake_command(), "abc", runner=_boom)

    def test_nonzero_rc_raises_det_invocation_error(self):
        runner = _ScriptedRunner([_Completed(returncode=1, stdout="", stderr="boom")])
        with self.assertRaises(DetInvocationError):
            determined.experiment_status(_fake_command(), 14, runner=runner)


class TaskStatusTests(unittest.TestCase):
    def test_success_builds_det_status(self):
        runner = _ScriptedRunner([_Completed(returncode=0, stdout=REAL_TASK_LIST_SHAPE, stderr=REAL_STDERR_WARNING)])
        status = determined.task_status(
            _fake_command(), "4fe6d93e-113c-4038-a79d-0ac8c9ae16bb", runner=runner
        )
        self.assertEqual(status.kind, "task")
        self.assertEqual(status.state, "ACTIVE")
        self.assertFalse(status.terminal)
        self.assertEqual(runner.calls[0][0], ["det", "task", "list", "--json"])


# ---------------------------------------------------------------------------
# poll_until_terminal
# ---------------------------------------------------------------------------


def _experiment_response(state: str) -> _Completed:
    payload = json.dumps([{"experiment": {"state": state}}])
    return _Completed(returncode=0, stdout=payload, stderr=REAL_STDERR_WARNING)


class PollUntilTerminalTests(unittest.TestCase):
    def test_reaches_terminal_and_stops_early(self):
        runner = _ScriptedRunner(
            [
                _experiment_response("STATE_RUNNING"),
                _experiment_response("STATE_RUNNING"),
                _experiment_response("STATE_COMPLETED"),
            ]
        )
        sleeper = _RecordingSleeper()
        result = determined.poll_until_terminal(
            _fake_command(), "experiment", 14,
            interval_seconds=10, max_polls=5, sleeper=sleeper, runner=runner,
        )
        self.assertTrue(result.reached_terminal)
        self.assertFalse(result.exhausted)
        self.assertEqual(len(result.observations), 3)
        self.assertEqual(runner.calls.__len__(), 3)  # never over-polls past terminal
        self.assertEqual(sleeper.calls, [10, 10])  # between polls only: before poll 1 and poll 2

        obs = result.observations
        self.assertFalse(obs[0].changed)  # first successful observation
        self.assertFalse(obs[1].changed)  # RUNNING -> RUNNING
        self.assertTrue(obs[2].changed)  # RUNNING -> COMPLETED
        self.assertTrue(obs[2].status.terminal)
        self.assertIs(result.final_status, obs[2].status)

    def test_exhausts_max_polls_without_terminal(self):
        runner = _ScriptedRunner([_experiment_response("STATE_RUNNING") for _ in range(3)])
        sleeper = _RecordingSleeper()
        result = determined.poll_until_terminal(
            _fake_command(), "experiment", 14,
            interval_seconds=5, max_polls=3, sleeper=sleeper, runner=runner,
        )
        self.assertFalse(result.reached_terminal)
        self.assertTrue(result.exhausted)
        self.assertEqual(len(result.observations), 3)
        self.assertEqual(sleeper.calls, [5, 5])
        self.assertIsNotNone(result.final_status)
        self.assertEqual(result.final_status.state, "STATE_RUNNING")

    def test_changed_flag_across_multiple_transitions(self):
        runner = _ScriptedRunner(
            [
                _experiment_response("STATE_QUEUED"),
                _experiment_response("STATE_RUNNING"),
                _experiment_response("STATE_RUNNING"),
                _experiment_response("STATE_COMPLETED"),
            ]
        )
        result = determined.poll_until_terminal(
            _fake_command(), "experiment", 14,
            interval_seconds=0, max_polls=10, sleeper=lambda s: None, runner=runner,
        )
        changed_flags = [obs.changed for obs in result.observations]
        self.assertEqual(changed_flags, [False, True, False, True])

    def test_failed_observation_with_zero_tolerance_raises(self):
        runner = _ScriptedRunner([_Completed(returncode=0, stdout="not json", stderr="")])
        with self.assertRaises(DetParseError):
            determined.poll_until_terminal(
                _fake_command(), "experiment", 14,
                interval_seconds=1, max_polls=3, tolerate_transient=0,
                sleeper=lambda s: None, runner=runner,
            )

    def test_failed_observation_with_tolerance_is_recorded_not_raised(self):
        runner = _ScriptedRunner(
            [
                _Completed(returncode=0, stdout="not json", stderr=""),  # poll 0: transient failure
                _experiment_response("STATE_RUNNING"),  # poll 1: first successful observation
                _experiment_response("STATE_COMPLETED"),  # poll 2: transition + terminal
            ]
        )
        sleeper = _RecordingSleeper()
        result = determined.poll_until_terminal(
            _fake_command(), "experiment", 14,
            interval_seconds=2, max_polls=5, tolerate_transient=1,
            sleeper=sleeper, runner=runner,
        )
        self.assertEqual(len(result.observations), 3)

        failed_obs = result.observations[0]
        self.assertIsNone(failed_obs.status)
        self.assertIsNotNone(failed_obs.failure)
        self.assertFalse(failed_obs.changed)

        first_success = result.observations[1]
        self.assertIsNotNone(first_success.status)
        self.assertIsNone(first_success.failure)
        # "unchanged" must never be conflated with "the previous poll failed":
        # this is the FIRST successful observation, so changed is False on
        # its own terms (nothing successful to compare against), not because
        # it matched the failed poll before it.
        self.assertFalse(first_success.changed)
        self.assertEqual(first_success.status.state, "STATE_RUNNING")

        final_obs = result.observations[2]
        self.assertTrue(final_obs.changed)
        self.assertTrue(final_obs.status.terminal)
        self.assertTrue(result.reached_terminal)

        # sleeper still called between every poll, including around the
        # failed one -- it consumed a poll slot like any other.
        self.assertEqual(sleeper.calls, [2, 2])

    def test_exhausting_transient_budget_then_raises(self):
        runner = _ScriptedRunner(
            [
                _Completed(returncode=0, stdout="not json", stderr=""),  # poll 0: tolerated
                _Completed(returncode=0, stdout="not json", stderr=""),  # poll 1: budget exhausted -> raises
            ]
        )
        with self.assertRaises(DetParseError):
            determined.poll_until_terminal(
                _fake_command(), "experiment", 14,
                interval_seconds=0, max_polls=5, tolerate_transient=1,
                sleeper=lambda s: None, runner=runner,
            )

    def test_unknown_kind_raises_contract_error(self):
        with self.assertRaises(ContractError):
            determined.poll_until_terminal(
                _fake_command(), "notarealkind", "1",
                interval_seconds=1, max_polls=1, sleeper=lambda s: None,
                runner=_ScriptedRunner([]),
            )

    def test_sleeper_never_called_when_terminal_on_first_poll(self):
        runner = _ScriptedRunner([_experiment_response("STATE_COMPLETED")])
        sleeper = _RecordingSleeper()
        result = determined.poll_until_terminal(
            _fake_command(), "experiment", 14,
            interval_seconds=99, max_polls=2, sleeper=sleeper, runner=runner,
        )
        self.assertTrue(result.reached_terminal)
        self.assertEqual(sleeper.calls, [])


if __name__ == "__main__":
    unittest.main()
