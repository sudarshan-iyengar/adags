"""`scripts/emit_training_identity.py`: does the emitter run the SAME ray-trace
that produced the frozen held-out buffers, and does it refuse everything it
promises to refuse?

The emitter re-renders the fixture's front-most identity buffers for the
sixteen TRAINING cameras. It writes them into `train_identity/`, never into
`gt_identity/`, because that directory name carries a held-out-only meaning
that existing consumers read off the name itself.

The load-bearing check is the emitter's own `--self-test`: it re-renders
HELD-OUT (camera, frame) pairs and asserts byte identity against the frozen
`gt_identity/` files. That is a precondition on the MECHANISM, not a score, so
these tests spend most of their effort on the question "would the self-test
still fail if the mechanism were neutered?" -- a frozen reading rule that
cannot detect its own neutering is the failure mode this project has already
paid for twice.

The neuter this specifically defends against is real and easy to write: both
`render()` and `event_present_at()` read MUTABLE generator globals whose
import-time defaults do NOT match LRV3 (ground half extent 3.0 vs 1.3, first
return frame 54 vs 57). An emitter that forgot to bind them would produce
plausible-looking buffers that disagree with the held-out ones.

No GPU. Fixture-dependent tests skip when the fixture is absent.
"""

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_synthetic_reveal_scene as gen  # noqa: E402
from scripts import emit_training_identity as emitter  # noqa: E402

SCENE = REPO_ROOT / "data" / "synthetic" / "lrv3"
HAVE_SCENE = (SCENE / "event_spec.json").is_file() and (SCENE / "gt_identity").is_dir()


# ---------------------------------------------------------------------------
# refusals and the emission plan -- no fixture needed
# ---------------------------------------------------------------------------

class RefusalTests(unittest.TestCase):

    def test_refuses_gt_identity_as_a_destination(self):
        for candidate in ("gt_identity", "GT_Identity", "gt_identity/"):
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / candidate.rstrip("/")
                with self.assertRaises(emitter.EmitError) as ctx:
                    emitter.resolve_out_dir(Path(tmp), dest)
                self.assertIn("HELD-OUT", str(ctx.exception))

    def test_a_nested_gt_identity_is_refused_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "a" / "b" / "gt_identity"
            with self.assertRaises(emitter.EmitError):
                emitter.resolve_out_dir(Path(tmp), dest)

    def test_the_default_destination_is_train_identity(self):
        out = emitter.resolve_out_dir(Path("/fixture"), None)
        self.assertEqual(out.name, "train_identity")
        self.assertNotEqual(out.name, "gt_identity")

    def test_training_and_held_out_camera_sets_partition_the_rig(self):
        train = emitter.training_cameras()
        held = emitter.held_out_cameras()
        self.assertEqual(sorted(train + held), list(range(gen.N_CAMERAS)))
        self.assertEqual(set(train) & set(held), set())
        self.assertEqual(held, [2, 7, 12, 17])
        self.assertEqual(len(train), 16)


class PlanTests(unittest.TestCase):
    """`planned_outputs` IS the emission loop's iteration order, not a parallel
    description of it, so asserting on it asserts on what gets written."""

    SPEC = {"n_frames": gen.N_FRAMES}

    def test_lrv3_plans_exactly_16x60_files(self):
        plan = emitter.planned_outputs(self.SPEC)
        self.assertEqual(len(plan), 16 * 60)
        self.assertEqual(len(plan), 960)
        self.assertEqual(len({name for _, _, name in plan}), 960)

    def test_the_plan_never_contains_a_held_out_camera(self):
        plan = emitter.planned_outputs(self.SPEC)
        held = set(emitter.held_out_cameras())
        self.assertEqual({c for c, _, _ in plan} & held, set())

    def test_asking_for_a_held_out_camera_is_refused(self):
        for cam in emitter.held_out_cameras():
            with self.assertRaises(emitter.EmitError) as ctx:
                emitter.planned_outputs(self.SPEC, cameras=[cam])
            self.assertIn("not training cameras", str(ctx.exception))
        with self.assertRaises(emitter.EmitError):
            emitter.planned_outputs(self.SPEC, cameras=[0, 2])

    def test_filenames_match_the_frozen_convention(self):
        plan = emitter.planned_outputs(self.SPEC, cameras=[0], frames=[0, 7, 59])
        self.assertEqual([n for _, _, n in plan],
                         ["cam00_f000.npy", "cam00_f007.npy", "cam00_f059.npy"])

    def test_out_of_range_frames_are_refused(self):
        with self.assertRaises(emitter.EmitError):
            emitter.planned_outputs(self.SPEC, frames=[60])
        with self.assertRaises(emitter.EmitError):
            emitter.planned_outputs(self.SPEC, frames=[-1])


# ---------------------------------------------------------------------------
# the load-bearing precondition, and whether it survives neutering
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAVE_SCENE, "LRV3 fixture not present in this checkout")
class SelfTestTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spec = emitter.load_spec(SCENE)

    def test_the_self_test_passes_on_the_real_fixture(self):
        verified = emitter.run_self_test(SCENE, self.spec, verbose=False)
        self.assertGreaterEqual(len(verified), emitter.MIN_SELF_TEST_PAIRS)
        for cam, _, _ in verified:
            self.assertIn(cam, emitter.held_out_cameras())

    def test_the_pairs_exercise_both_branches_of_render(self):
        pairs = emitter.default_self_test_pairs(self.spec)
        with emitter.fixture_scene_globals(self.spec):
            flags = [gen.event_present_at(f) for _, f in pairs]
        self.assertIn(True, flags)
        self.assertIn(False, flags)

    # -- neuter 1: the ray-trace itself -------------------------------------

    def test_NEUTERED_render_is_detected(self):
        """Flip a single pixel of the identity buffer; byte identity must go."""
        real_render = gen.render

        def perturbed(c2w, event_present):
            rgb, identity, depth = real_render(c2w, event_present)
            identity = identity.copy()
            identity[0, 0] = identity[0, 0] + 1
            return rgb, identity, depth

        with _patched(gen, "render", perturbed):
            with self.assertRaises(emitter.SelfTestError) as ctx:
                emitter.run_self_test(SCENE, self.spec, verbose=False)
        msg = str(ctx.exception)
        self.assertIn("SELF-TEST FAILED", msg)
        self.assertIn("differing pixels: 1", msg)

    # -- neuter 2: the dtype ------------------------------------------------

    def test_NEUTERED_dtype_is_detected(self):
        """int32 instead of int16 would produce a file every consumer misreads."""
        with _patched(emitter, "IDENTITY_DTYPE", np.int32):
            with self.assertRaises(emitter.SelfTestError) as ctx:
                emitter.run_self_test(SCENE, self.spec, verbose=False)
        msg = str(ctx.exception)
        self.assertIn("SELF-TEST FAILED", msg)
        self.assertIn("int32", msg)
        self.assertIn("int16", msg)

    # -- neuter 3: the mutable-global binding, the defect this really guards --

    def test_NEUTERED_scene_global_binding_is_detected(self):
        """The realistic mistake: forget that render() reads module globals.

        With the binding removed the emitter would silently use the LRV1
        defaults -- ground half extent 3.0 against the fixture's 1.3, and a
        first return frame of 54 against 57.
        """
        self.assertNotEqual(gen.GROUND_HALF_EXTENT,
                            self.spec["ground_half_extent"])
        self.assertNotEqual(list(gen.EPISODE_2_FRAMES),
                            self.spec["presence_frames"]["episode_2"])

        @contextlib.contextmanager
        def no_binding(spec):
            yield

        with _patched(emitter, "fixture_scene_globals", no_binding):
            with self.assertRaises(emitter.SelfTestError) as ctx:
                emitter.run_self_test(SCENE, self.spec, verbose=False)
        self.assertIn("SELF-TEST FAILED", str(ctx.exception))

    def test_the_binding_is_restored_after_use(self):
        before = (gen.GROUND_HALF_EXTENT, gen.EPISODE_1_FRAMES,
                  gen.GAP_FRAMES, gen.EPISODE_2_FRAMES)
        with emitter.fixture_scene_globals(self.spec):
            self.assertEqual(gen.GROUND_HALF_EXTENT,
                             self.spec["ground_half_extent"])
        self.assertEqual((gen.GROUND_HALF_EXTENT, gen.EPISODE_1_FRAMES,
                          gen.GAP_FRAMES, gen.EPISODE_2_FRAMES), before)

    # -- neuter 4: the precondition ON the precondition ----------------------

    def test_pairs_that_never_render_the_event_are_refused(self):
        gap = self.spec["presence_frames"]["gap"]
        pairs = [(2, gap[0]), (7, gap[0] + 1), (12, gap[0] + 2)]
        with self.assertRaises(emitter.SelfTestError) as ctx:
            emitter.run_self_test(SCENE, self.spec, pairs=pairs, verbose=False)
        self.assertIn("both branches", str(ctx.exception))

    def test_too_few_pairs_is_refused(self):
        with self.assertRaises(emitter.SelfTestError):
            emitter.run_self_test(SCENE, self.spec, pairs=[(2, 0)], verbose=False)

    def test_a_non_held_out_pair_is_refused(self):
        pairs = [(0, 0), (1, 40), (3, 57)]
        with self.assertRaises(emitter.SelfTestError) as ctx:
            emitter.run_self_test(SCENE, self.spec, pairs=pairs, verbose=False)
        self.assertIn("non-held-out", str(ctx.exception))

    def test_a_fixture_with_no_frozen_buffers_cannot_witness_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "event_spec.json").write_text(
                (SCENE / "event_spec.json").read_text())
            with self.assertRaises(emitter.SelfTestError) as ctx:
                emitter.run_self_test(Path(tmp), self.spec, verbose=False)
            self.assertIn("refusing", str(ctx.exception))


# ---------------------------------------------------------------------------
# emission: manifest fidelity, fail-closed behaviour, atomicity
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAVE_SCENE, "LRV3 fixture not present in this checkout")
class EmissionTests(unittest.TestCase):
    """Small partial emissions only: one camera, two frames. The full 960-file
    run is the caller's decision, and the plan is asserted separately."""

    CAMS = [0]
    FRAMES = [29, 57]

    @classmethod
    def setUpClass(cls):
        cls.spec = emitter.load_spec(SCENE)

    def _emit(self, out, **kw):
        kw.setdefault("cameras", self.CAMS)
        kw.setdefault("frames", self.FRAMES)
        kw.setdefault("verbose", False)
        return emitter.emit(SCENE, out_dir=out, **kw)

    def test_manifest_sha256s_match_the_files_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "train_identity"
            manifest = self._emit(out)
            written = sorted(p.name for p in out.glob("*.npy"))
            self.assertEqual(written, sorted(manifest["files"]))
            self.assertEqual(manifest["file_count"], len(written))
            for name, digest in manifest["files"].items():
                self.assertEqual(emitter.sha256_file(out / name), digest, name)

    def test_the_manifest_records_what_it_must(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "train_identity"
            self._emit(out)
            manifest = json.loads((out / emitter.MANIFEST_NAME).read_text())
            self.assertEqual(manifest["schema"], emitter.MANIFEST_SCHEMA)
            self.assertEqual(manifest["scene_id"], "LRV3")
            self.assertTrue(manifest["fixture"].endswith("lrv3"))
            self.assertEqual(manifest["held_out_cameras_excluded"], [2, 7, 12, 17])
            self.assertEqual(manifest["train_cameras"], self.CAMS)
            self.assertEqual(manifest["dtype"], "int16")
            self.assertTrue(manifest["oracle"])
            self.assertFalse(manifest["estimated"])
            self.assertFalse(manifest["curated"])
            self.assertIn("ORACLE", manifest["provenance"])
            self.assertFalse(manifest["complete"])       # limits were in effect
            self.assertEqual(
                manifest["generator_module"]["sha256"],
                emitter.sha256_file(REPO_ROOT / "scripts"
                                    / "build_synthetic_reveal_scene.py"))
            self.assertTrue(manifest["self_test"]["passed"])
            self.assertGreaterEqual(len(manifest["self_test"]["pairs"]),
                                    emitter.MIN_SELF_TEST_PAIRS)
            for pair in manifest["self_test"]["pairs"]:
                self.assertIn(pair["camera"], emitter.held_out_cameras())

    def test_the_buffers_are_int16_and_carry_the_event_id_only_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "train_identity"
            self._emit(out, frames=[40, 57])            # gap frame, return frame
            gap = np.load(out / "cam00_f040.npy")
            ret = np.load(out / "cam00_f057.npy")
            self.assertEqual(gap.dtype, np.dtype(np.int16))
            self.assertEqual(ret.dtype, np.dtype(np.int16))
            self.assertEqual(gap.shape, (gen.HEIGHT, gen.WIDTH))
            self.assertEqual(int((gap == gen.EVENT_OBJECT_ID).sum()), 0)
            self.assertGreater(int((ret == gen.EVENT_OBJECT_ID).sum()), 0)

    def test_no_held_out_camera_file_is_ever_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "train_identity"
            self._emit(out)
            for name in os.listdir(str(out)):
                if name.endswith(".npy"):
                    self.assertNotIn(int(name[3:5]), emitter.held_out_cameras())

    def test_emission_into_gt_identity_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(emitter.EmitError):
                self._emit(Path(tmp) / "gt_identity")
            self.assertFalse((Path(tmp) / "gt_identity").exists())

    def test_conflicting_prior_output_is_refused_and_force_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "train_identity"
            self._emit(out)
            victim = out / "cam00_f029.npy"
            np.save(str(victim), np.zeros((gen.HEIGHT, gen.WIDTH), np.int16))
            with self.assertRaises(emitter.EmitError) as ctx:
                self._emit(out)
            self.assertIn("DIFFERENT", str(ctx.exception))
            # the corrupted file is untouched by the refused run
            self.assertEqual(int(np.load(str(victim)).sum()), 0)
            self._emit(out, force=True)
            self.assertGreater(int((np.load(str(victim)) != 0).sum()), 0)

    def test_identical_prior_output_is_not_a_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "train_identity"
            first = self._emit(out)
            second = self._emit(out)               # must not raise
            self.assertEqual(first["files"], second["files"])

    def test_a_failing_self_test_blocks_emission_entirely(self):
        real_render = gen.render

        def perturbed(c2w, event_present):
            rgb, identity, depth = real_render(c2w, event_present)
            identity = identity.copy()
            identity[1, 1] = identity[1, 1] + 7
            return rgb, identity, depth

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "train_identity"
            with _patched(gen, "render", perturbed):
                with self.assertRaises(emitter.SelfTestError):
                    self._emit(out)
            self.assertFalse(out.exists(), "emitted despite a failed precondition")

    def test_dry_run_writes_nothing_but_still_runs_the_precondition(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "train_identity"
            manifest = self._emit(out, dry_run=True)
            self.assertTrue(manifest["dry_run"])
            self.assertEqual(manifest["file_count"], len(self.FRAMES))
            self.assertTrue(manifest["self_test"]["passed"])
            self.assertFalse(out.exists())

    def test_no_temp_files_survive_a_completed_emission(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "train_identity"
            self._emit(out)
            leftovers = [p.name for p in out.iterdir()
                         if p.name.endswith(".tmp")]
            self.assertEqual(leftovers, [])


@unittest.skipUnless(HAVE_SCENE, "LRV3 fixture not present in this checkout")
class CliTests(unittest.TestCase):

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "emit_training_identity.py")]
            + list(args),
            cwd=str(REPO_ROOT), capture_output=True, text=True)

    def test_self_test_exits_zero_on_the_real_fixture(self):
        proc = self._run("--fixture", str(SCENE), "--self-test")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("SELF-TEST PASSED", proc.stdout)

    def test_gt_identity_destination_exits_non_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._run("--fixture", str(SCENE),
                             "--out-dir", str(Path(tmp) / "gt_identity"),
                             "--limit-cameras", "0", "--limit-frames", "0")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("REFUSED", proc.stderr)


# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patched(module, name, value):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


if __name__ == "__main__":
    unittest.main()
