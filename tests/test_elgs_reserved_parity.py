"""The reserved-unit parity switch: a control cell must drop EXACTLY the
units an EL-GS cell drops.

Run with:
    python -m unittest tests.test_elgs_reserved_parity

Why this exists: `elgs/trainer_hooks.py` reserves a stratified ~25% of
training units for the spec-§7 confirmation measure and `filter_elgs_reserved`
removes them from the refit dataset — but only when EL-GS state exists. A bare
temporal-substrate control therefore trained on 100% of the units while an
EL-GS cell trained on ~75%, handing the control a third more data on a
benchmark whose open question is held-out generalization
(research-wiki/operations/elgs-matched-triple-readiness-2026-08-18.md §2).

The decisive test below is `test_both_paths_drop_identical_indices`. It does
NOT construct full EL-GS state — that needs a prereg, a schedule, a registry,
a CUDA-shaped Gaussian model and a slot grid — so instead it asserts the
property that makes drift impossible: `setup_elgs` stores exactly
`build_reserved_pool(cameras)`'s indices in `ElgsTrainerState.reserved_indices`,
and the parity path returns exactly the same frozenset from the same function.
That is verified two ways: by reading the one shared producer (there is only
one implementation of the `% 4` rule in the module, asserted here), and by
comparing the parity path's output against `filter_elgs_reserved`'s behaviour
on a hand-built state carrying that same pool.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402
from elgs import trainer_hooks as th  # noqa: E402


class FakeCamera:
    def __init__(self, timestamp: float, image_name: str):
        self.timestamp = timestamp
        self.image_name = image_name


class FakeScene:
    def __init__(self, cameras):
        self._cameras = list(cameras)

    def getTrainCameras(self):
        return list(self._cameras)


class FakeOpt:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_cameras(n_frames: int = 8, n_cameras: int = 8):
    """n_frames timestamps x n_cameras views, in a deliberately awkward index
    order so a rule that depended on raw index order would fail."""

    cameras = []
    for frame in range(n_frames):
        for cam in range(n_cameras):
            cameras.append(
                FakeCamera(
                    timestamp=frame / 30.0,
                    image_name=f"cam{(cam * 3) % n_cameras:02d}",
                )
            )
    return cameras


class SharedReservationTest(unittest.TestCase):
    def test_only_one_implementation_of_the_rule_exists(self):
        """If the `% 4` diagonal appeared twice in the module the two paths
        could drift; the refactor's whole point is that it appears once."""

        source = inspect.getsource(th)
        self.assertEqual(
            source.count("(f + c) % 4 == 0"),
            1,
            "the reservation rule must have exactly one implementation",
        )
        self.assertIn("reserved = build_reserved_pool(cameras)", source)

    def test_reserves_about_a_quarter(self):
        cameras = make_cameras()
        reserved = th.build_reserved_pool(cameras)
        self.assertEqual(len(reserved), len(cameras) // 4)

    def test_stratifies_over_cameras_and_time(self):
        cameras = make_cameras()
        reserved = th.build_reserved_pool(cameras)
        times = {t for _, t in reserved}
        self.assertGreater(len(times), 1)
        all_times = sorted({c.timestamp for c in cameras})
        span = all_times[-1] - all_times[0]
        covered = max(times) - min(times)
        self.assertGreaterEqual(covered, 0.5 * span)

    def test_camera_position_guard_fires(self):
        """One timestamp, three cameras: the diagonal selects only position 0
        while the data offers three, so the >= 2 distinct-positions guard
        must refuse. (With only ONE camera per timestamp the guard correctly
        does NOT fire -- it demands two positions only when two exist.)"""

        cameras = [FakeCamera(0.0, f"cam{c:02d}") for c in range(3)]
        with self.assertRaises(ContractError) as caught:
            th.build_reserved_pool(cameras)
        self.assertIn("camera positions", str(caught.exception))

    def test_time_span_guard_fires(self):
        """With three cameras per timestamp the diagonal skips every frame
        with f == 1 (mod 4) -- including the LAST frame here -- so putting a
        large gap before it leaves the reserved pool covering far less than
        half the sequence's time range."""

        cameras = []
        for t in (0.0, 1.0, 2.0, 3.0, 4.0, 1000.0):
            for c in range(3):
                cameras.append(FakeCamera(t, f"cam{c:02d}"))
        with self.assertRaises(ContractError) as caught:
            th.build_reserved_pool(cameras)
        self.assertIn("half the", str(caught.exception))

    def test_single_camera_per_timestamp_is_allowed(self):
        """The camera guard is min(2, largest group), so a genuinely
        single-view sequence is not refused. Pinned so a future tightening of
        the guard is a deliberate change and not a silent one."""

        cameras = [FakeCamera(f / 30.0, "cam00") for f in range(8)]
        self.assertEqual(len(th.build_reserved_pool(cameras)), 2)

    def test_deterministic(self):
        cameras = make_cameras()
        self.assertEqual(
            th.build_reserved_pool(cameras), th.build_reserved_pool(cameras)
        )


class ParityPathTest(unittest.TestCase):
    def setUp(self):
        self.cameras = make_cameras()
        self.scene = FakeScene(self.cameras)
        self.pool = th.build_reserved_pool(self.cameras)
        self.expected = frozenset(index for index, _ in self.pool)

    def test_flag_defaults_to_off(self):
        from arguments import OptimizationParams
        from argparse import ArgumentParser

        params = OptimizationParams(ArgumentParser())
        self.assertFalse(params.elgs_reserved_parity)

    def test_no_op_when_flag_is_off(self):
        opt = FakeOpt(elgs_reserved_parity=False, elgs_enable=False)
        self.assertIsNone(th.reserved_indices_for_parity(self.scene, opt))

    def test_no_op_when_the_attribute_is_absent_entirely(self):
        """Older configs have no such key; the call must not raise."""

        self.assertIsNone(th.reserved_indices_for_parity(self.scene, FakeOpt()))

    def test_parity_on_control_returns_the_shared_reservation(self):
        opt = FakeOpt(elgs_reserved_parity=True, elgs_enable=False)
        self.assertEqual(th.reserved_indices_for_parity(self.scene, opt), self.expected)

    def test_parity_is_a_no_op_when_elgs_is_enabled(self):
        """EL-GS already drops them through ElgsTrainerState; a second filter
        would double-drop and break the match in the other direction."""

        opt = FakeOpt(elgs_reserved_parity=True, elgs_enable=True)
        self.assertIsNone(th.reserved_indices_for_parity(self.scene, opt))

    def test_both_paths_drop_identical_indices(self):
        """THE decisive assertion: the control path and the EL-GS path remove
        the same dataset positions.

        The EL-GS side is exercised through `filter_elgs_reserved` on a state
        carrying the shared pool's indices -- which is exactly what
        `setup_elgs` stores (`reserved_indices=frozenset(u[0] for u in
        reserved)` over `reserved = build_reserved_pool(cameras)`).
        """

        dataset = list(range(len(self.cameras)))

        class FakeState:
            reserved_indices = self.expected

        elgs_side = th.filter_elgs_reserved(dataset, FakeState())
        control_side = th.filter_reserved_indices(
            dataset,
            th.reserved_indices_for_parity(
                self.scene, FakeOpt(elgs_reserved_parity=True, elgs_enable=False)
            ),
        )
        elgs_kept = sorted(elgs_side.indices)
        control_kept = sorted(control_side.indices)
        self.assertEqual(elgs_kept, control_kept)
        dropped = sorted(set(dataset) - set(elgs_kept))
        self.assertEqual(dropped, sorted(self.expected))
        self.assertEqual(len(elgs_kept), len(dataset) - len(dataset) // 4)

    def test_setup_elgs_stores_the_shared_pool_unchanged(self):
        """Guards the link the decisive test relies on: setup_elgs must derive
        reserved_indices from build_reserved_pool and nothing else."""

        source = inspect.getsource(th.setup_elgs)
        self.assertIn("reserved = build_reserved_pool(cameras)", source)
        self.assertIn("reserved_indices=frozenset(u[0] for u in reserved)", source)

    def test_filter_reserved_indices_is_a_no_op_on_empty(self):
        dataset = list(range(10))
        self.assertIs(th.filter_reserved_indices(dataset, None), dataset)
        self.assertIs(th.filter_reserved_indices(dataset, frozenset()), dataset)


if __name__ == "__main__":
    unittest.main()
