"""Every key `GaussianModel.capture()` writes must be read back by `restore()`.

This is a SOURCE-INSPECTION test and imports no torch, so it runs anywhere --
including a workstation where the rest of `tests/test_packet_birth.py` cannot
even be collected. That matters: the defect it exists to prevent was found by
reading code, not by a failing test, precisely because the tests that would
have caught it could not run.

The defect: `_packet_ids` was maintained by the prune and densify paths but
was absent from `capture()`, so a branch-from-checkpoint on a B1 arm silently
lost the packet-id column that `scripts/consolidate_packets.py` and the B2
lane consume. Nothing raised; the column simply came back empty.

This test generalizes that to the whole routing/motion payload rather than
pinning the one field, so the next field added to `capture()` without a
matching `restore()` fails here instead of silently losing state at the next
checkpoint branch.

Deliberately asymmetric: it requires captured => restored, NOT the converse.
`restore()` legitimately reads keys that older checkpoints may lack, and those
`.get` defaults are the backward-compatibility path.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "scene" / "gaussian_model.py"

#: Keys that capture() writes and restore() deliberately does NOT read back,
#: each with the reason. A new entry here needs a stated justification -- the
#: point of the test is that silence is not an option.
CAPTURED_BUT_NOT_RESTORED: dict[str, str] = {}


def _function_body(source: str, name: str) -> str:
    """Return the source text of a top-level method, by indentation."""

    match = re.search(rf"\n    def {re.escape(name)}\(", source)
    if match is None:
        raise AssertionError(f"could not locate `def {name}(` in {SOURCE}")
    start = match.start()
    following = re.search(r"\n    def [A-Za-z_]", source[start + 10 :])
    end = len(source) if following is None else start + 10 + following.start()
    return source[start:end]


class CaptureRestoreSymmetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8")
        self.capture = _function_body(self.source, "capture")
        self.restore = _function_body(self.source, "restore")

    def test_every_captured_routing_motion_key_is_restored(self) -> None:
        captured = set(re.findall(r'"([a-z0-9_]+)":\s*self\.', self.capture))
        # Any quoted key appearing anywhere in restore() counts as read back.
        # Deliberately broad: restore() reads some payloads through NESTED
        # dicts rather than `routing_motion_params.get(...)` -- `logit_a` and
        # `logit_b` arrive via `gate_params["logit_a"]`. A narrower pattern
        # reported those two as lost when they are restored correctly, i.e. it
        # produced a false positive on sound code. The failure this test
        # guards is a key that is captured and then mentioned NOWHERE in
        # restore(), which this still catches.
        restored = set(re.findall(r'"([a-z0-9_]+)"', self.restore))
        self.assertTrue(captured, "parsed zero captured keys -- the parser is broken, not the code")
        self.assertTrue(restored, "parsed zero restored keys -- the parser is broken, not the code")

        missing = {k for k in captured if k not in restored} - set(CAPTURED_BUT_NOT_RESTORED)
        self.assertEqual(
            missing,
            set(),
            "capture() writes these keys but restore() never reads them, so a "
            "branch-from-checkpoint loses them SILENTLY: "
            f"{sorted(missing)}. Either restore them, or add each to "
            "CAPTURED_BUT_NOT_RESTORED with a stated reason.",
        )

    def test_packet_ids_specifically_round_trips(self) -> None:
        """The field whose absence motivated this file. Pinned by name."""

        self.assertIn(
            '"packet_ids": self._packet_ids',
            self.capture,
            "capture() must serialize _packet_ids; without it a B1 arm's packet "
            "column is lost across a checkpoint branch",
        )
        self.assertRegex(
            self.restore,
            r'self\._packet_ids = routing_motion_params\.get\(\s*"packet_ids"',
            "restore() must read packet_ids back",
        )

    def test_packet_ids_default_matches_its_initialiser(self) -> None:
        """An old checkpoint must restore to exactly what __init__ builds."""

        init = re.search(
            r"self\._packet_ids = torch\.empty\(0, dtype=torch\.long\)", self.source
        )
        self.assertIsNotNone(init, "expected _packet_ids to be initialised as an empty long tensor")
        self.assertRegex(
            self.restore,
            r'"packet_ids", torch\.empty\(0, dtype=torch\.long\)',
            "the restore default must match __init__ exactly, so a checkpoint "
            "written before this key existed restores to the pre-repair state "
            "rather than to something new",
        )

    def test_pre_repair_checkpoint_fails_closed(self) -> None:
        """A missing key must REFUSE, not silently restore an empty column.

        Both consumers of `_packet_ids` are guarded by `numel() > 0`
        (`prune_points` and the clone/split append), so an empty column is
        skipped without complaint. That makes the backward-compatibility
        default a silent reproduction of the original defect for any
        packet-birth run resuming a pre-repair checkpoint. This project
        already carries the rule as method: a guard that can degrade to
        "protects nothing" is worse than no guard.
        """

        self.assertIn(
            "_require_packet_ids_on_restore",
            self.restore,
            "restore() must fail closed when packet birth is enabled and the "
            "checkpoint carries no packet_ids",
        )
        self.assertIn(
            "checkpoint carries no `packet_ids`",
            self.restore,
            "the mismatch must RAISE with an actionable message, not warn",
        )
        self.assertIn(
            "raise ValueError(",
            self.restore,
            "the fail-closed path must raise",
        )
        guards = self.source.count("if self._packet_ids.numel() > 0:")
        self.assertGreaterEqual(
            guards,
            2,
            "the numel guards are the reason silence is possible; if they are "
            "removed this test's premise changed and it should be revisited",
        )

    def test_the_guard_is_actually_armed_by_a_caller(self) -> None:
        """A guard nothing switches on is the defect it protects against.

        `restore()` reads the flag via `getattr(..., False)`, so it is inert
        unless a call site sets it. This asserts main.py arms it, and arms it
        conditionally on packet birth so runs that do not use packet birth
        can still load old checkpoints.
        """

        main = (REPO_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn(
            "_require_packet_ids_on_restore",
            main,
            "the guard is never armed, so it protects nothing",
        )
        self.assertIn(
            'getattr(opt, "packet_birth_enable", False)',
            main,
            "the guard must be armed CONDITIONALLY on packet birth, so runs "
            "without it keep loading pre-repair checkpoints",
        )
        arm = main.index("_require_packet_ids_on_restore")
        call = main.index("gaussians.restore(model_params, opt)")
        self.assertLess(arm, call, "the flag must be set BEFORE restore() runs")


if __name__ == "__main__":
    unittest.main()
