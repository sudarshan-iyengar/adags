"""Emit per-pixel front-most identity buffers for the TRAINING cameras of an
existing synthetic leave-and-return fixture.

What this does
--------------
``scripts/build_synthetic_reveal_scene.py`` already computes a front-most
``identity`` buffer for EVERY camera while it emits images -- it uses one on
every camera to count the event object's visible pixels -- but it SAVES the
buffer only for the four held-out cameras, into ``gt_identity/``. This script
re-runs that same ray-trace, by IMPORTING the generator rather than
reimplementing it, and writes the buffers for the sixteen TRAINING cameras.

Nothing that already exists is touched. The generator is imported read-only,
and the output goes into a NEW directory.

Why ``train_identity/`` is a separate directory from ``gt_identity/``
---------------------------------------------------------------------
``gt_identity/`` has a HELD-OUT-ONLY meaning that consumers already rely on.
``scripts/estimate_episodes.py`` installs a runtime leakage guard whose
forbidden-path test is the substring ``gt_identity``, and the fixture's own
tests enumerate ``gt_identity/`` expecting exactly ``len(test_cameras) *
n_frames`` files. Adding training-camera buffers to that directory would
silently change what every existing consumer sees -- an evaluator iterating
the directory would start scoring training views, and the file-count invariant
would stop meaning what it means today. So the emission goes somewhere else,
and writing into a directory whose final component is ``gt_identity`` is
REFUSED by construction.

These masks are ORACLE masks
----------------------------
They are ray-traced from the authored world, not curated by a human and not
estimated from images. They are ground truth in exactly the sense
``gt_identity/`` is, and they are just as forbidden to any estimation stage
that must not see ground truth. The leakage guard in
``scripts/estimate_episodes.py`` keys on the substring ``gt_identity`` and
therefore does NOT block this directory; a consumer that must stay
oracle-blind has to exclude ``train_identity/`` itself. The manifest states
this in the artifact as well as here.

The load-bearing precondition
-----------------------------
Before any buffer is written, ``--self-test`` re-renders several HELD-OUT
(camera, frame) pairs through the imported ``render()`` and asserts the
resulting ``int16`` ``.npy`` payload is BYTE-IDENTICAL to the frozen file
already in ``gt_identity/``. That is a statement about the MECHANISM -- "the
ray-trace this emitter runs is the one that produced the frozen buffers" --
evaluated before anything is emitted, so it cannot be read as a favourable
outcome. It matters because ``render()`` and ``event_present_at()`` read
MUTABLE module globals (``GROUND_HALF_EXTENT``, ``EPISODE_1_FRAMES``,
``GAP_FRAMES``, ``EPISODE_2_FRAMES``) that the generator's ``main()`` sets from
its command line and that import-time defaults do NOT match for LRV3: the
module default ground half extent is 3.0 against the fixture's 1.3, and the
default first return frame is 54 against the fixture's 57. An emitter that
forgot to bind those from ``event_spec.json`` would produce a plausible-looking
buffer for every training camera that disagreed with the held-out ones about
the ground plane and about three frames of event presence, and nothing
downstream would say so. The self-test fails loudly on exactly that.

The self-test also carries its OWN precondition: the pairs must exercise both
the event-present and the event-absent branch of ``render()``, so a pair list
that never renders the event object cannot deliver a clean pass.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_synthetic_reveal_scene as gen  # noqa: E402

#: Schema string stamped into the manifest.
MANIFEST_SCHEMA = "adags.train_identity.v1"
MANIFEST_NAME = "MANIFEST.train_identity.json"

#: Default output directory name, relative to the fixture root.
OUTPUT_DIRNAME = "train_identity"

#: The directory name this emitter refuses to write into, ever.
FORBIDDEN_DIRNAME = "gt_identity"

#: Buffer dtype. Must match what the generator saves for the held-out views
#: (`identity.astype(np.int16)`); the self-test compares raw `.npy` bytes, so
#: a dtype change is caught rather than silently accepted.
IDENTITY_DTYPE = np.int16

#: Minimum number of held-out (camera, frame) pairs the self-test must check.
MIN_SELF_TEST_PAIRS = 3

FILENAME_FMT = "cam{:02d}_f{:03d}.npy"


class EmitError(RuntimeError):
    """Refusal: a precondition on the emission itself failed."""


class SelfTestError(RuntimeError):
    """The emitter's ray-trace does not reproduce the frozen buffers."""


# ---------------------------------------------------------------------------
# fixture reading and the mutable-global binding
# ---------------------------------------------------------------------------

def load_spec(fixture):
    """Read ``event_spec.json`` and refuse a fixture the generator cannot
    reproduce."""
    fixture = Path(fixture)
    spec_path = fixture / "event_spec.json"
    if not spec_path.is_file():
        raise EmitError("no event_spec.json under %s" % fixture)
    spec = json.loads(spec_path.read_text())

    frozen = (
        ("n_frames", gen.N_FRAMES),
        ("n_cameras", gen.N_CAMERAS),
        ("width", gen.WIDTH),
        ("height", gen.HEIGHT),
        ("focal_px", gen.FOCAL),
    )
    for key, want in frozen:
        if spec.get(key) != want:
            raise EmitError(
                "fixture %s disagrees with the generator on %r: fixture %r, "
                "generator %r -- this emitter can only reproduce a scene the "
                "IMPORTED module still describes"
                % (fixture, key, spec.get(key), want))
    if list(spec.get("test_cameras", [])) != list(gen.TEST_CAMERAS):
        raise EmitError(
            "fixture held-out cameras %r differ from the generator's %r"
            % (spec.get("test_cameras"), list(gen.TEST_CAMERAS)))
    if int(spec.get("event_object", {}).get("id", -999)) != gen.EVENT_OBJECT_ID:
        raise EmitError("fixture event object id differs from the generator's")
    if list(spec.get("train_cameras", [])) != training_cameras():
        raise EmitError(
            "fixture training cameras %r differ from the generator's %r"
            % (spec.get("train_cameras"), training_cameras()))
    return spec


def training_cameras():
    """Every camera that is NOT held out. The only source of cameras this
    emitter will write, so a held-out camera cannot be emitted."""
    held_out = set(gen.TEST_CAMERAS)
    return [c for c in range(gen.N_CAMERAS) if c not in held_out]


def held_out_cameras():
    return list(gen.TEST_CAMERAS)


@contextlib.contextmanager
def fixture_scene_globals(spec):
    """Bind the generator's MUTABLE module globals to this fixture's values.

    ``render()`` reads ``GROUND_HALF_EXTENT`` and ``event_present_at()`` reads
    the episode frame ranges. The generator's ``main()`` sets both from its
    command line; import-time defaults are the LRV1 defaults and do not match
    LRV3. Restored on exit so importing this module cannot perturb anything
    else in the process.
    """
    presence = spec["presence_frames"]
    saved = (gen.GROUND_HALF_EXTENT, gen.EPISODE_1_FRAMES, gen.GAP_FRAMES,
             gen.EPISODE_2_FRAMES)
    gen.GROUND_HALF_EXTENT = float(spec["ground_half_extent"])
    gen.EPISODE_1_FRAMES = tuple(presence["episode_1"])
    gen.GAP_FRAMES = tuple(presence["gap"])
    gen.EPISODE_2_FRAMES = tuple(presence["episode_2"])
    try:
        yield
    finally:
        (gen.GROUND_HALF_EXTENT, gen.EPISODE_1_FRAMES, gen.GAP_FRAMES,
         gen.EPISODE_2_FRAMES) = saved


def render_identity(pose, frame):
    """The buffer this emitter writes, as an array. Single code path: the
    self-test and the emission both go through here."""
    _, identity, _ = gen.render(pose, gen.event_present_at(frame))
    return np.asarray(identity).astype(IDENTITY_DTYPE)


def npy_bytes(array):
    """The exact bytes ``np.save`` would write for ``array``."""
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# the load-bearing self-test
# ---------------------------------------------------------------------------

def default_self_test_pairs(spec):
    """Held-out (camera, frame) pairs spanning presence, absence and return.

    Deterministic, and deliberately not tunable from the command line beyond a
    count: the point of the check is that it always exercises both branches of
    ``render()``.
    """
    presence = spec["presence_frames"]
    ep1, gap, ep2 = presence["episode_1"], presence["gap"], presence["episode_2"]
    frames = [
        ep1[0],                       # first present frame
        ep1[1],                       # last frame before the gap
        gap[0],                       # first absent frame
        (gap[0] + gap[1]) // 2,       # mid gap
        ep2[0],                       # first return frame
        ep2[1],                       # last frame
    ]
    cams = held_out_cameras()
    return [(cams[i % len(cams)], int(f)) for i, f in enumerate(frames)]


def run_self_test(fixture, spec, pairs=None, verbose=True):
    """Assert this emitter's ray-trace reproduces the frozen held-out buffers
    BYTE FOR BYTE. Raises :class:`SelfTestError` on any difference.

    Returns the list of verified ``(camera, frame, sha256)`` triples.
    """
    fixture = Path(fixture)
    gt = fixture / FORBIDDEN_DIRNAME
    if not gt.is_dir():
        raise SelfTestError(
            "%s has no %s/ directory, so the emitter's ray-trace cannot be "
            "checked against anything frozen; refusing" % (fixture, FORBIDDEN_DIRNAME))

    pairs = list(default_self_test_pairs(spec) if pairs is None else pairs)
    if len(pairs) < MIN_SELF_TEST_PAIRS:
        raise SelfTestError(
            "self-test needs at least %d held-out pairs, got %d"
            % (MIN_SELF_TEST_PAIRS, len(pairs)))
    held_out = set(held_out_cameras())
    bad = [c for c, _ in pairs if c not in held_out]
    if bad:
        raise SelfTestError(
            "self-test pairs name non-held-out cameras %r; only frozen "
            "held-out buffers can witness path identity" % sorted(set(bad)))

    verified = []
    with fixture_scene_globals(spec):
        # PRECONDITION on the check itself: the pairs must exercise BOTH
        # branches of render(). A pair list that never renders the event
        # object would pass while leaving the event-presence path untested.
        present = [f for _, f in pairs if gen.event_present_at(f)]
        absent = [f for _, f in pairs if not gen.event_present_at(f)]
        if not present or not absent:
            raise SelfTestError(
                "self-test pairs do not exercise both branches of render(): "
                "%d event-present, %d event-absent" % (len(present), len(absent)))

        poses = gen.camera_poses()
        for cam, frame in pairs:
            name = FILENAME_FMT.format(cam, frame)
            ref_path = gt / name
            if not ref_path.is_file():
                raise SelfTestError("frozen buffer missing: %s" % ref_path)
            reference = ref_path.read_bytes()
            produced = npy_bytes(render_identity(poses[cam], frame))
            if produced != reference:
                raise SelfTestError(_mismatch_report(ref_path, reference, produced))
            verified.append((cam, frame, sha256_bytes(produced)))
            if verbose:
                print("  self-test OK  %s  (event %s)  sha256 %s"
                      % (name, "present" if gen.event_present_at(frame) else "absent",
                         sha256_bytes(produced)[:16]))
    return verified


def _mismatch_report(ref_path, reference, produced):
    lines = [
        "SELF-TEST FAILED: this emitter's ray-trace does NOT reproduce the "
        "frozen held-out buffer.",
        "  frozen file : %s" % ref_path,
        "  frozen bytes: %d (sha256 %s)" % (len(reference), sha256_bytes(reference)),
        "  produced    : %d (sha256 %s)" % (len(produced), sha256_bytes(produced)),
    ]
    try:
        ref_arr = np.load(io.BytesIO(reference), allow_pickle=False)
        got_arr = np.load(io.BytesIO(produced), allow_pickle=False)
        lines.append("  frozen array: dtype=%s shape=%s"
                     % (ref_arr.dtype, ref_arr.shape))
        lines.append("  produced arr: dtype=%s shape=%s"
                     % (got_arr.dtype, got_arr.shape))
        if ref_arr.shape == got_arr.shape:
            diff = int((ref_arr != got_arr).sum())
            lines.append("  differing pixels: %d of %d" % (diff, ref_arr.size))
    except Exception as exc:                      # pragma: no cover - defensive
        lines.append("  (could not parse one of the payloads: %s)" % exc)
    lines.append(
        "  The emitted training buffers would disagree with the held-out ones. "
        "Refusing to write anything.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# emission
# ---------------------------------------------------------------------------

def resolve_out_dir(fixture, out_dir):
    out = Path(fixture) / OUTPUT_DIRNAME if out_dir is None else Path(out_dir)
    if out.name.strip().lower() == FORBIDDEN_DIRNAME:
        raise EmitError(
            "refusing to write into %r: that directory name means HELD-OUT "
            "ONLY. Existing consumers (the leakage guard in "
            "scripts/estimate_episodes.py, and the fixture's file-count "
            "invariants) read that meaning off the directory name, and adding "
            "training-camera buffers would silently change what they see. "
            "Use %r." % (str(out), OUTPUT_DIRNAME))
    return out


def planned_outputs(spec, cameras=None, frames=None):
    """The exact ordered list of ``(camera, frame, filename)`` this emitter
    will write. The emission loop iterates THIS list, so it is not a parallel
    description of the plan -- it is the plan.
    """
    train = training_cameras()
    if cameras is None:
        cams = list(train)
    else:
        allowed = set(train)
        cams = [int(c) for c in cameras]
        rogue = sorted({c for c in cams if c not in allowed})
        if rogue:
            raise EmitError(
                "cameras %r are not training cameras of this fixture (held-out "
                "cameras are %r); this emitter never writes a held-out camera "
                "into %s/" % (rogue, held_out_cameras(), OUTPUT_DIRNAME))
        cams = [c for c in train if c in set(cams)]
    frs = list(range(int(spec["n_frames"]))) if frames is None else [int(f) for f in frames]
    for f in frs:
        if not 0 <= f < int(spec["n_frames"]):
            raise EmitError("frame %d is outside [0, %d)" % (f, spec["n_frames"]))
    return [(c, f, FILENAME_FMT.format(c, f)) for c in cams for f in frs]


def _atomic_save(path, array):
    """Write ``array`` to ``path`` via a temp file + rename, so a killed run
    cannot leave a truncated ``.npy`` behind."""
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent),
                                    prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with open(str(tmp), "wb") as fh:
            np.save(fh, array)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp), str(path))
        tmp = None
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def emit(fixture, out_dir=None, cameras=None, frames=None, force=False,
         dry_run=False, verbose=True):
    """Run the precondition, then emit. Returns the manifest dict."""
    fixture = Path(fixture).resolve()
    spec = load_spec(fixture)
    out = resolve_out_dir(fixture, out_dir)

    if verbose:
        print("fixture   : %s" % fixture)
        print("scene_id  : %s" % spec.get("scene_id"))
        print("out dir   : %s" % out)
        print("self-test : re-rendering held-out buffers before emitting")
    verified = run_self_test(fixture, spec, verbose=verbose)

    plan = planned_outputs(spec, cameras=cameras, frames=frames)
    complete = cameras is None and frames is None
    if verbose:
        print("plan      : %d files (%d cameras x %d frames)%s"
              % (len(plan), len({c for c, _, _ in plan}),
                 len({f for _, f, _ in plan}),
                 "" if complete else "  [PARTIAL: limits in effect]"))

    # Fail closed on conflicting prior output BEFORE writing anything.
    if out.is_dir() and not force:
        conflicts = [n for _, _, n in plan if (out / n).is_file()]
        if conflicts:
            with fixture_scene_globals(spec):
                poses = gen.camera_poses()
                differing = []
                for cam, frame, name in plan:
                    dest = out / name
                    if not dest.is_file():
                        continue
                    if dest.read_bytes() != npy_bytes(render_identity(poses[cam], frame)):
                        differing.append(name)
                        if len(differing) >= 8:
                            break
            if differing:
                raise EmitError(
                    "%d of %d planned files already exist in %s with DIFFERENT "
                    "content (e.g. %s). Refusing to overwrite; pass --force if "
                    "that is intended."
                    % (len(differing), len(plan), out, ", ".join(differing[:8])))

    files = {}
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
    held_out = set(held_out_cameras())
    with fixture_scene_globals(spec):
        poses = gen.camera_poses()
        for i, (cam, frame, name) in enumerate(plan):
            # Belt and braces: the plan cannot contain a held-out camera, and
            # this asserts it at the moment of writing anyway.
            if cam in held_out:
                raise EmitError("refusing to write held-out camera %d" % cam)
            array = render_identity(poses[cam], frame)
            if array.dtype != IDENTITY_DTYPE:
                raise EmitError("identity buffer dtype drifted to %s" % array.dtype)
            payload = npy_bytes(array)
            files[name] = sha256_bytes(payload)
            if not dry_run:
                _atomic_save(out / name, array)
            if verbose and (i + 1) % 100 == 0:
                print("  %d/%d" % (i + 1, len(plan)))

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "provenance": (
            "ORACLE ray-traced front-most identity buffers, produced by "
            "re-running scripts/build_synthetic_reveal_scene.render() on the "
            "TRAINING cameras of this fixture. They are NOT curated by a human "
            "and NOT estimated from images: they are ground truth in exactly "
            "the sense gt_identity/ is."),
        "oracle": True,
        "estimated": False,
        "curated": False,
        "leakage_note": (
            "As forbidden to any oracle-blind estimation stage as gt_identity/. "
            "The runtime guard in scripts/estimate_episodes.py keys on the "
            "substring 'gt_identity' and therefore does NOT block this "
            "directory; a consumer that must stay oracle-blind has to exclude "
            "%s/ explicitly." % OUTPUT_DIRNAME),
        "fixture": str(fixture).replace("\\", "/"),
        "scene_id": spec.get("scene_id"),
        "out_dir": str(out).replace("\\", "/"),
        "train_cameras": sorted({c for c, _, _ in plan}),
        "held_out_cameras_excluded": held_out_cameras(),
        "n_frames": len({f for _, f, _ in plan}),
        "frames": sorted({f for _, f, _ in plan}),
        "dtype": np.dtype(IDENTITY_DTYPE).name,
        "file_count": len(files),
        "files": files,
        "complete": bool(complete),
        "limits": {
            "cameras": None if cameras is None else sorted(int(c) for c in cameras),
            "frames": None if frames is None else sorted(int(f) for f in frames),
        },
        "generator_module": {
            "path": "scripts/build_synthetic_reveal_scene.py",
            "sha256": sha256_file(Path(gen.__file__)),
        },
        "emitter_module": {
            "path": "scripts/emit_training_identity.py",
            "sha256": sha256_file(Path(__file__)),
        },
        "self_test": {
            "description": (
                "Re-rendered these HELD-OUT (camera, frame) pairs through the "
                "imported render() and asserted the int16 .npy payload is "
                "byte-identical to the frozen file in gt_identity/. Run BEFORE "
                "any emission, as a precondition on the mechanism."),
            "pairs": [{"camera": c, "frame": f, "sha256": h} for c, f, h in verified],
            "passed": True,
        },
        "dry_run": bool(dry_run),
    }
    if not dry_run:
        manifest_path = out / MANIFEST_NAME
        tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=1))
        os.replace(str(tmp), str(manifest_path))
        if verbose:
            print("wrote %d buffers + %s" % (len(files), manifest_path))
    elif verbose:
        print("DRY RUN: nothing written (%d files planned)" % len(files))
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _int_list(text):
    if text is None:
        return None
    return [int(x) for x in str(text).replace(",", " ").split()]


def build_parser():
    ap = argparse.ArgumentParser(
        description="Emit training-camera front-most identity buffers for a "
                    "synthetic leave-and-return fixture.")
    ap.add_argument("--fixture", required=True,
                    help="fixture root, e.g. data/synthetic/lrv3")
    ap.add_argument("--out-dir", default=None,
                    help="output directory; default <fixture>/%s. A path whose "
                         "final component is %r is REFUSED."
                         % (OUTPUT_DIRNAME, FORBIDDEN_DIRNAME))
    ap.add_argument("--self-test", action="store_true",
                    help="run ONLY the byte-identity precondition against the "
                         "frozen held-out buffers and exit")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing output whose content differs")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the precondition and the render loop but write "
                         "nothing")
    ap.add_argument("--limit-cameras", default=None,
                    help="comma-separated subset of TRAINING cameras")
    ap.add_argument("--limit-frames", default=None,
                    help="comma-separated subset of frames")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    fixture = Path(args.fixture)
    try:
        if args.self_test:
            spec = load_spec(fixture)
            print("fixture   : %s" % fixture.resolve())
            print("scene_id  : %s" % spec.get("scene_id"))
            print("ground_half_extent %.4f  presence %s"
                  % (spec["ground_half_extent"], spec["presence_frames"]))
            verified = run_self_test(fixture, spec, verbose=True)
            print("SELF-TEST PASSED: %d held-out (camera, frame) pairs "
                  "reproduce their frozen gt_identity buffers byte for byte."
                  % len(verified))
            return 0
        emit(fixture,
             out_dir=args.out_dir,
             cameras=_int_list(args.limit_cameras),
             frames=_int_list(args.limit_frames),
             force=args.force,
             dry_run=args.dry_run)
        return 0
    except (EmitError, SelfTestError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
