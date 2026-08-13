#!/usr/bin/env python3
"""DiVa-360 -> ADAGS Blender-convention scene converter (M1, DRAFT).

Authority: ``research-wiki/operations/elgs-m0-m1-implementation-plan.md``
S6 item 21 ("DiVa-360 support: new ``scripts/diva360_to_blender.py``
converter -> existing ``transforms_train.json`` Blender-branch loader; no
``scene/`` reader changes unless converter proves impossible") and S11.1
item 5 ("Preprocessing: ... converter -> existing Blender-branch layout
(``transforms_train/test.json``, ``points3d.ply``) so ``scene/`` is
untouched"). Schema findings that shaped this converter are recorded in
``research-wiki/operations/elgs-m1-census-record.md`` and re-verified by
direct Apollo inspection on 2026-08-11 (see ``elgs/diva360_schema.py``'s
module docstring for the exact findings).

What this script does, per sequence directory
(``/apollo/users/sri/proj_adags/data/diva360/<sequence>/`` on Apollo):

1. Discovers and validates the shipped ``transforms_train.json`` /
   ``transforms_test.json`` / (optional) ``transforms_val.json``.
2. Discovers the sequence's frame tarball(s) (``*.tar`` / ``*.tar.gz``) --
   by CONTENT, never by name: the winning archive is whichever one's
   members (after stripping ITS OWN top-level directory) cover every
   ``(camera, frame-index)`` relative path a split's frames reference.
   This is necessary because DiVa-360 sequences ship inconsistent tarball
   names/formats (``battery`` has ``image.tar`` uncompressed + fisheye
   masks in ``.tar``, others ship ``.tar.gz`` -- see the census record) and
   because the ``segmented_gt``/``segmented_ngp`` archives share the same
   ``.png`` extension and per-camera directory shape but a DIFFERENT
   embedded frame index, so extension/name matching alone is ambiguous.
3. Un-tars exactly the referenced frames into the layout the transforms
   JSON references (whatever top-level directory ``file_path`` declares,
   e.g. ``undist/`` -- never hardcoded).
4. Writes ``transforms_train.json`` / ``transforms_test.json`` /
   ``transforms_val.json`` in the exact convention
   ``scene/dataset_readers.py::readCamerasFromTransforms`` expects,
   stamping the ADAGS ``time`` key DiVa-360 never ships (mapping decision:
   ``time = embedded_frame_index / fps``; see
   ``elgs.diva360_schema.frame_index_to_time``).
5. Synthesizes ``points3d.ply`` (DiVa-360 ships no COLMAP sparse
   reconstruction): points are sampled uniformly inside the union of every
   converted camera's frustum corners at a near/far depth derived from the
   camera-ring radius -- documented as a coarse smoke-test volume, not a
   content-aware bound (see ``frustum_union_bounding_box``).
6. Writes an immutable provenance JSON: source sha256 (looked up in
   ``MANIFEST.sha256`` -- the acquisition manifest recorded in
   ``research-wiki/operations/elgs-m1-census-record.md``), converter argv,
   git commit/branch/dirty state, and the archive/camera selections made.

``--dry-run`` performs every read-only discovery/validation step (steps 1-2
above) and prints the resulting plan as JSON without touching the
filesystem (no directory creation, no extraction, no writes).

``--window START END [--stride N]`` (owner decision D-M1-1, recorded in
``research-wiki/operations/elgs-m1-census-record.md``): activates GENUINE
TEMPORAL mode instead of the default single-instant scene. The DiVa-360 rig
is static, so each camera's ``transform_matrix``/intrinsics in the shipped
single-instant split are valid for EVERY frame, not just the one instant
the JSON happens to reference. For each camera already in a split (train or
test membership is unchanged -- it still follows the shipped files) and
each frame index ``i`` in ``[START, END]`` inclusive, stepped by
``--stride`` (default 1), this mode emits one frame entry reusing that
camera's pose/intrinsics with ``file_path`` pointing at the per-index image
and ``"time": i / fps``. Only ``(camera, index)`` pairs whose image
actually exists in the discovered frame archive are emitted (``frames_1``
does not necessarily hold every camera or every index); a camera that
contributes zero frames to the requested window is a fail-closed
``ContractError``, never a silently smaller scene.

Masks are DERIVED, not extracted from a separate archive (real-data finding,
det tasks 405a7a8d/c662fadf on ``unlock``): ``frames_1``'s members are RGBA
at exactly the transforms' declared calibration resolution, and their ALPHA
channel already IS the per-frame fg matte (continuous alpha, 99.9%
concentrated at the extremes) -- ``segmented_ngp``/``segmented_gt`` carry
the same segmentation but in the DIFFERENT, ORIGINAL pre-undistortion pixel
space (measured 1280x720 vs. the calibration space's 1160x550), so they are
never extracted by this converter. For every extracted window frame, this
mode opens the PNG, takes its alpha channel, binarizes it strictly ``> 127``
into an L-mode ``{0, 255}`` mask, and writes it to ``masks/camNN/<8-digit>``
under the SAME filename the frame used -- one mask per extracted
``(camera, index)`` pair. A frame with no alpha channel is a fail-closed
``ContractError`` naming the file, never a silently blank mask.

Fail-closed: every schema/contract surprise raises a
``depth_visibility.errors.ContractError`` (or a typed subclass -- see
``elgs/diva360_schema.py``); nothing here silently degrades or guesses.

Runtime constraint (Apollo Determined container): stdlib + numpy + Pillow
(PIL). Never import torch (or anything that imports torch) at module level.
``depth_visibility.errors``, ``depth_visibility.canonical``,
``depth_visibility.artifacts``, and ``elgs.diva360_schema`` are all
verified torch-free at import time (see their own module docstrings /
``depth_visibility/__init__.py``'s own no-Torch guarantee) and are the only
module-level, non-stdlib, non-numpy imports here. PIL is a legitimate
dependency now (baked into the Apollo images and present in the CPU test
venv) for two things -- the archive-selection resolution probe and window
mode's alpha-to-mask derivation -- but every PIL (and ``io``) import stays
LOCAL to the function that needs it, so importing this module never touches
PIL and module import stays exactly as cheap as before.

Known performance characteristic (documented, not a defect): tar is a
sequential format with no central index, so content-addressed archive
discovery and extraction stream through a candidate archive until every
wanted member is found or the archive is exhausted. For a real multi-GB
DiVa-360 tarball this can take minutes; it is a one-time per-sequence cost
within the M1 census's "preprocessing: reproducibility accounting only, no
hard ceiling" allowance (``configs/elgs/prereg_m1_census_v1.json``
``ceilings``), not something this script tries to index around.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: --window mode provenance/plan note.
#:
#: 2026-08-13: these were previously two unconditional constants asserting
#: that masks came from ``frames_1`` and that ``segmented_ngp`` was "deliberately
#: NOT extracted". They were emitted verbatim regardless of what the selector
#: actually chose, so for the three defective conversions the sealed provenance
#: CONTRADICTED its own ``archive_selection`` field and hid the substrate defect
#: from three cycles of integrity audits. Provenance must report measurements,
#: never assumptions -- ``window_mask_provenance`` below is computed from the
#: actual selection. See
#: ``research-wiki/operations/elgs-substrate-defect-2026-08-13.md``.
WINDOW_MASK_NOTE = (
    "masks are derived from the alpha channel of the ACTUALLY SELECTED frame "
    "archive recorded in frame_source below; the selector enforces an "
    "unconditional decoded-vs-declared resolution postcondition"
)


def window_mask_provenance(
    archive_selection: Mapping[str, ArchiveMatch],
    declared_sizes: Mapping[str, tuple[int, int]] | None = None,
) -> dict:
    """Measured (not assumed) frame/mask substrate record, per split."""

    declared_values = sorted({tuple(v) for v in (declared_sizes or {}).values()})
    return {
        "mask_source": (
            "alpha channel of the selected frame archive, binarized >127"
        ),
        "mask_note": WINDOW_MASK_NOTE,
        "frame_source": {
            split: {
                "archive_path": match.archive_path,
                "member_prefix": match.top_level_dir,
                "decoded_size": list(match.decoded_size) if match.decoded_size else None,
            }
            for split, match in sorted(archive_selection.items())
        },
        "declared_sizes_observed": [list(v) for v in declared_values],
    }

from depth_visibility.artifacts import atomic_write_json_immutable  # noqa: E402
from depth_visibility.canonical import sha256_file  # noqa: E402
from depth_visibility.errors import ArtifactError, ContractError, ProvenanceError, SchemaError  # noqa: E402
from elgs import diva360_schema as schema  # noqa: E402


# ---------------------------------------------------------------------------
# Discovery data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitPlan:
    split: str
    source_path: str  # transforms filename
    frame_count: int
    camera_ids: tuple[int, ...]
    wanted_relative_paths: tuple[str, ...]  # e.g. "cam01/00001000.png"


@dataclass(frozen=True)
class ArchiveMatch:
    archive_path: str  # relative to sequence_dir
    top_level_dir: str  # this archive's own internal member prefix (may be
    # MULTI-component, e.g. "dynamic_data/frames_1" -- some DiVa-360
    # sequences nest frames_1 one level deeper than others)
    decoded_size: tuple[int, int] | None = None  # measured (w, h) of the
    # probe member; recorded in provenance so the substrate is auditable


@dataclass(frozen=True)
class WindowSpec:
    start: int
    end: int
    stride: int


@dataclass(frozen=True)
class WindowSplitPlan:
    """Per-split ``--window`` discovery result.

    ``camera_relative_paths``: camera id -> sorted tuple of "rest" paths
    (archive-top-dir-stripped, e.g. ``"cam01/00000060.png"``) confirmed
    present in the split's frame archive within the requested window.
    Every camera key is guaranteed non-empty (build_plan fails closed
    otherwise). Masks are no longer discovered as a separate archive --
    they are DERIVED at execution time from each extracted frame's own
    alpha channel (see the module docstring's ``--window`` section), so
    there is no mask-archive field here to plan.
    """

    split: str
    camera_relative_paths: dict[int, tuple[str, ...]]


@dataclass(frozen=True)
class ConversionPlan:
    sequence_dir: str
    sequence_name: str
    output_dir: str
    fps: float
    extension: str
    num_random_points: int
    seed: int
    scene_top_dir: str
    splits: tuple[SplitPlan, ...]
    archive_selection: dict[str, ArchiveMatch]
    manifest_path: str
    source_sha256: str
    held_out_camera_ids: tuple[int, ...]
    candidate_archives: tuple[str, ...]
    window: WindowSpec | None = None
    window_splits: dict[str, WindowSplitPlan] | None = None
    declared_sizes: dict[str, tuple[int, int]] | None = None


# ---------------------------------------------------------------------------
# Pure discovery / planning (read-only: JSON + tar member peeking)
# ---------------------------------------------------------------------------


def discover_transform_files(sequence_dir: Path) -> dict[str, Path]:
    """Locate the shipped transforms files; required splits must exist."""

    found: dict[str, Path] = {}
    for split in schema.ALL_SPLITS:
        candidate = sequence_dir / schema.split_filename(split)
        if candidate.is_file():
            found[split] = candidate
    missing_required = [s for s in schema.REQUIRED_SPLITS if s not in found]
    if missing_required:
        raise SchemaError(
            f"{sequence_dir}: missing required transforms file(s) for split(s) "
            f"{missing_required}"
        )
    return found


def load_transforms_payload(path: Path, *, extension: str) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path}: not valid JSON: {exc}") from exc
    return schema.validate_transforms_payload(raw, extension=extension)


def build_split_plan(split: str, path: Path, *, extension: str) -> tuple[SplitPlan, dict]:
    payload = load_transforms_payload(path, extension=extension)
    camera_ids: list[int] = []
    wanted: list[str] = []
    for frame in payload["frames"]:
        relative = schema.frame_relative_path(frame["file_path"], extension)
        _, rest = schema.split_top_level_dir(relative)
        wanted.append(rest)
        camera_ids.append(schema.parse_camera_id(rest))
    plan = SplitPlan(
        split=split,
        source_path=path.name,
        frame_count=len(payload["frames"]),
        camera_ids=tuple(sorted(set(camera_ids))),
        wanted_relative_paths=tuple(wanted),
    )
    return plan, payload


def discover_candidate_archives(sequence_dir: Path) -> list[Path]:
    candidates = sorted(
        p
        for p in sequence_dir.iterdir()
        if p.is_file() and (p.name.endswith(".tar") or p.name.endswith(".tar.gz"))
    )
    if not candidates:
        raise ContractError(f"{sequence_dir}: no .tar / .tar.gz archives found")
    return candidates


def _iter_tar_relative_paths(archive_path: Path):
    """Yield every regular-file member name in ``archive_path``, streaming.

    Uses auto-detecting stream mode (``r:*``) rather than trusting the
    archive's own filename extension, per this script's discovery
    philosophy.
    """

    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar:
            if member.isfile():
                yield member.name


def _probe_member_size(
    archive_path: Path, top_level_dir: str, rest: str
) -> tuple[int, int] | None:
    """Decode ONE member and return its ``(width, height)``, or ``None`` if
    the member is absent or undecodable."""

    import io

    from PIL import Image

    member_name = f"{top_level_dir}/{rest}"
    with tarfile.open(archive_path, "r:*") as tar:
        try:
            handle = tar.extractfile(member_name)
        except KeyError:
            return None
        if handle is None:
            return None
        payload = handle.read()
    try:
        with Image.open(io.BytesIO(payload)) as image:
            return image.size
    except Exception:
        return None


def strip_archive_prefix(member_name: str, prefix: str) -> str | None:
    """Strip ``prefix + "/"`` off ``member_name``; ``None`` if it does not
    match. Unlike ``schema.split_top_level_dir`` this handles MULTI-component
    prefixes (see ``ArchiveMatch.top_level_dir``)."""

    head = prefix + "/"
    if not member_name.startswith(head):
        return None
    return member_name[len(head) :] or None


def _discover_member_prefix(
    archive_path: Path, wanted: set[str]
) -> tuple[str, set[str]] | None:
    """Find the unique member prefix under which ``archive_path`` carries the
    wanted relative paths.

    DiVa-360 ships the SAME logical archive at different nesting depths
    across sequences (measured 2026-08-13: ``pour_tea`` ->
    ``frames_1/camNN/...`` but ``writing_2`` -> ``dynamic_data/frames_1/
    camNN/...``). Stripping a FIXED single component therefore silently
    fails to match on the deeper layout, which is what caused the
    2026-08-13 substrate defect
    (``research-wiki/operations/elgs-substrate-defect-2026-08-13.md``).

    Each wanted path has a known component count, so a member's prefix is
    whatever precedes its last ``k`` components -- an O(1) test per member,
    no per-wanted scan. Returns ``(prefix, found)`` or ``None`` if nothing
    matched. Fails closed if members match under MORE THAN ONE prefix.
    """

    depths = {path.count("/") + 1 for path in wanted}
    prefixes: dict[str, set[str]] = {}
    for member_name in _iter_tar_relative_paths(archive_path):
        parts = member_name.split("/")
        for depth in depths:
            if len(parts) <= depth:
                continue
            rest = "/".join(parts[-depth:])
            if rest in wanted:
                prefixes.setdefault("/".join(parts[:-depth]), set()).add(rest)
        if any(found == wanted for found in prefixes.values()):
            break
    if not prefixes:
        return None
    if len(prefixes) > 1:
        raise ContractError(
            f"{archive_path}: wanted frame paths match under more than one "
            f"member prefix {sorted(prefixes)} -- cannot resolve a unique "
            "archive layout"
        )
    prefix, found = next(iter(prefixes.items()))
    return prefix, found


def select_archive_for_split(
    candidates: Sequence[Path],
    wanted_relative_paths: Sequence[str],
    declared_sizes: Mapping[str, tuple[int, int]] | None = None,
) -> ArchiveMatch:
    """Content-addressed discovery of the archive that has every wanted frame.

    Matches purely on content (member path with the archive's own member
    prefix stripped) -- never on the archive's filename or an assumed
    internal directory name, and never assuming a FIXED prefix depth.
    Scanning a candidate stops as soon as every wanted path has been found.

    THE RESOLUTION POSTCONDITION IS UNCONDITIONAL (2026-08-13). The
    selected archive's decoded ``(width, height)`` MUST equal the declared
    ``(w, h)``, whether one candidate covered or several. Before
    2026-08-13 the check ran only on the disambiguation branch, so a
    single covering candidate was accepted undecoded -- that hole selected
    1280x720 ``segmented_ngp`` imagery against a 1160x550 calibration for
    ``writing_2`` and ``xylophone``. A declared size that is unavailable
    for the probe frame is now itself a fail-closed condition: an
    unverifiable substrate is not an acceptable substrate.

    When MORE than one candidate fully covers the wanted set (real DiVa-360:
    the per-frame mask archive (``segmented_ngp``) MIRRORS the frame
    archive's exact member layout -- measured on Apollo ``unlock``, det
    tasks 405a7a8d/c662fadf), candidates are disambiguated by a RESOLUTION
    probe, not content classification: DiVa-360's calibration
    (``transforms_*.json`` ``fl_x``/``fl_y``/``cx``/``cy``/``w``/``h``) is
    fit in the UNDISTORTED calibration space ``frames_1`` ships (measured
    1160x550, exactly the transforms' declared ``w``/``h``); the mask
    archives ship the ORIGINAL pre-undistortion space instead (measured
    1280x720) -- a resolution the declared intrinsics could never have been
    computed against. So: decode one probe member from each covering
    candidate and keep only the candidate(s) whose decoded ``(width,
    height)`` equals the declared ``(w, h)`` for that exact probe frame
    (``declared_sizes``, keyed by the same "rest" relative path); exactly
    one candidate must remain, else this fails closed. (A prior
    content-based probe -- classifying decoded pixel values as
    binary-mask-like -- was tried and failed on real data: the mask
    archive's RGB channel is zeroed only OUTSIDE the foreground and carries
    real photographic color INSIDE it, so luminance-based classification
    could not tell it apart from ``frames_1``'s own photographic RGB; the
    real, exact, a-priori signal lives in the declared resolution, not in
    pixel values.)
    """

    wanted = set(wanted_relative_paths)
    if not wanted:
        raise ContractError("cannot select an archive for zero wanted frame paths")
    covering: list[ArchiveMatch] = []
    for archive_path in candidates:
        discovered = _discover_member_prefix(archive_path, wanted)
        if discovered is None:
            continue
        prefix, found = discovered
        if found == wanted:
            covering.append(
                ArchiveMatch(archive_path=archive_path.name, top_level_dir=prefix)
            )
    if not covering:
        raise ContractError(
            f"no candidate archive contains every referenced frame path "
            f"({len(wanted)} wanted); candidates: {[c.name for c in candidates]}"
        )

    by_name = {c.name: c for c in candidates}
    probe_rest = sorted(wanted)[0]
    declared = (declared_sizes or {}).get(probe_rest)
    if declared is None:
        # Fail-closed even for a single covering candidate: without a
        # declared size the substrate cannot be verified at all.
        raise ContractError(
            f"cannot verify the frame substrate: no declared (w, h) is available "
            f"for the probe frame {probe_rest!r}; covering candidate(s) "
            f"{sorted(m.archive_path for m in covering)}"
        )

    if len(covering) > 1:
        matching = [
            match
            for match in covering
            if _probe_member_size(by_name[match.archive_path], match.top_level_dir, probe_rest)
            == declared
        ]
        if len(matching) != 1:
            raise ContractError(
                "ambiguous frame source: "
                f"{sorted(m.archive_path for m in covering)} all contain every "
                f"referenced frame; the resolution probe (declared {declared} for "
                f"{probe_rest!r}) left {sorted(m.archive_path for m in matching)} "
                "candidate(s) -- cannot discover a unique archive"
            )
        selected = matching[0]
    else:
        selected = covering[0]

    # UNCONDITIONAL POSTCONDITION -- runs on the single-covering path too.
    decoded = _probe_member_size(
        by_name[selected.archive_path], selected.top_level_dir, probe_rest
    )
    if decoded != declared:
        raise ContractError(
            f"frame-substrate mismatch: selected archive "
            f"{selected.archive_path!r} (member prefix "
            f"{selected.top_level_dir!r}) decodes {probe_rest!r} at {decoded}, "
            f"but the calibration declares {declared}. The declared intrinsics "
            "could not have been computed against that pixel space -- refusing "
            "to convert. (2026-08-13 substrate defect; see "
            "research-wiki/operations/elgs-substrate-defect-2026-08-13.md)"
        )
    return ArchiveMatch(
        archive_path=selected.archive_path,
        top_level_dir=selected.top_level_dir,
        decoded_size=decoded,
    )


# ---------------------------------------------------------------------------
# --window mode discovery (owner decision D-M1-1)
# ---------------------------------------------------------------------------


def discover_window_frames_for_split(
    frame_archive_path: Path,
    frame_archive_top_level_dir: str,
    camera_templates: Mapping[int, tuple[str, int]],
    window: WindowSpec,
    extension: str,
) -> dict[int, list[str]]:
    """Stream the frame archive once; keep exactly the requested ``(camera,
    index)`` window pairs that actually exist.

    ``camera_templates``: camera id -> ``(camera_dir, index_width)`` (from
    ``elgs.diva360_schema.camera_path_template`` applied to that camera's
    single-instant source frame). Tolerant by design: a camera missing some
    -- or even all -- of its requested indices does not raise here; callers
    must check for empty per-camera lists themselves (a camera with zero
    hits is the fail-closed condition, not a partial-coverage one).
    """

    indices = schema.window_indices(window.start, window.end, window.stride)
    requested: dict[str, int] = {}
    for camera_id, (camera_dir, index_width) in camera_templates.items():
        for frame_index in indices:
            # Build via schema.window_file_path (reuses its frame_index
            # range validation) using the ARCHIVE's own top-level dir as a
            # scratch prefix, then strip it back off -- "rest" is the
            # archive-top-dir-stripped, extension-INCLUDED relative path
            # that member names compare against.
            # Build "rest" (prefix-stripped, extension-included) directly.
            # A sentinel prefix keeps window_file_path's index-width
            # validation without assuming the archive prefix is ONE
            # component (it may be multi-component -- see ArchiveMatch).
            full_path = schema.window_file_path(
                "\x00sentinel", camera_dir, index_width, frame_index
            )
            _, camera_rest = schema.split_top_level_dir(full_path)
            rest = camera_rest + extension
            requested[rest] = camera_id

    found: dict[int, list[str]] = {camera_id: [] for camera_id in camera_templates}
    remaining = set(requested)
    with tarfile.open(frame_archive_path, "r:*") as tar:
        for member in tar:
            if not member.isfile() or not remaining:
                continue
            rest = strip_archive_prefix(member.name, frame_archive_top_level_dir)
            if rest is None or rest not in remaining:
                continue
            found[requested[rest]].append(rest)
            remaining.discard(rest)
            if not remaining:
                break
    for paths in found.values():
        paths.sort()
    return found


def build_plan(
    sequence_dir: Path,
    output_dir: Path,
    *,
    sequence_name: str | None = None,
    manifest_path: Path | None = None,
    fps: float = schema.DEFAULT_FPS,
    extension: str = schema.DEFAULT_EXTENSION,
    num_random_points: int = 20000,
    seed: int = 0,
    window: tuple[int, int, int] | None = None,
) -> tuple[ConversionPlan, dict[str, dict]]:
    """Read-only: validates the source and discovers frame (and, in
    ``--window`` mode, mask) archives.

    ``window``, if given, is ``(start, end, stride)`` -- see the module
    docstring's ``--window`` section for the exact semantics. Returns
    ``(plan, {split: source_payload})``. Performs no filesystem writes --
    safe to call for ``--dry-run``, including full mask-completeness
    verification in window mode.
    """

    sequence_dir = sequence_dir.resolve()
    if not sequence_dir.is_dir():
        raise SchemaError(f"sequence_dir does not exist or is not a directory: {sequence_dir}")
    name = sequence_name or sequence_dir.name

    transform_files = discover_transform_files(sequence_dir)
    payloads: dict[str, dict] = {}
    split_plans: list[SplitPlan] = []
    scene_top_dirs: set[str] = set()
    declared_sizes: dict[str, tuple[int, int]] = {}
    for split, path in transform_files.items():
        split_plan, payload = build_split_plan(split, path, extension=extension)
        payloads[split] = payload
        split_plans.append(split_plan)
        for frame in payload["frames"]:
            relative = schema.frame_relative_path(frame["file_path"], extension)
            top, rest = schema.split_top_level_dir(relative)
            scene_top_dirs.add(top)
            if "w" in frame and "h" in frame:
                declared_sizes[rest] = (int(frame["w"]), int(frame["h"]))
    if len(scene_top_dirs) != 1:
        raise ContractError(
            f"{sequence_dir}: transforms files disagree on the frame "
            f"top-level directory: {sorted(scene_top_dirs)}"
        )
    scene_top_dir = next(iter(scene_top_dirs))

    candidates = discover_candidate_archives(sequence_dir)
    archive_selection: dict[str, ArchiveMatch] = {}
    for split_plan in split_plans:
        archive_selection[split_plan.split] = select_archive_for_split(
            candidates, split_plan.wanted_relative_paths, declared_sizes
        )

    resolved_manifest_path = manifest_path or (sequence_dir.parent / "MANIFEST.sha256")
    if not resolved_manifest_path.is_file():
        raise ProvenanceError(f"manifest not found: {resolved_manifest_path}")
    manifest_entries = schema.parse_sha256_manifest(
        resolved_manifest_path.read_text(encoding="utf-8")
    )
    source_sha256 = schema.lookup_source_sha256(manifest_entries, name)

    all_camera_ids = sorted({cid for plan in split_plans for cid in plan.camera_ids})
    held_out = tuple(sorted(schema.held_out_camera_ids(all_camera_ids)))

    window_spec: WindowSpec | None = None
    window_splits: dict[str, WindowSplitPlan] | None = None
    if window is not None:
        start, end, stride = window
        schema.window_indices(start, end, stride)  # validates start/end/stride
        window_spec = WindowSpec(start=start, end=end, stride=stride)
        window_splits = {}
        for split_plan in split_plans:
            payload = payloads[split_plan.split]
            camera_templates: dict[int, tuple[str, int]] = {}
            for frame in payload["frames"]:
                relative = schema.frame_relative_path(frame["file_path"], extension)
                _, rest = schema.split_top_level_dir(relative)
                camera_id = schema.parse_camera_id(rest)
                _, camera_dir, index_width = schema.camera_path_template(frame["file_path"])
                camera_templates[camera_id] = (camera_dir, index_width)

            frame_archive = archive_selection[split_plan.split]
            found = discover_window_frames_for_split(
                sequence_dir / frame_archive.archive_path,
                frame_archive.top_level_dir,
                camera_templates,
                window_spec,
                extension,
            )
            empty_cameras = sorted(cid for cid, paths in found.items() if not paths)
            if empty_cameras:
                raise ContractError(
                    f"{split_plan.split}: camera id(s) {empty_cameras} have zero frames "
                    f"in window [{start}, {end}] stride {stride} inside "
                    f"{frame_archive.archive_path!r}"
                )

            # No mask-archive discovery: masks are DERIVED at execution
            # time from each extracted frame's own alpha channel (see the
            # module docstring's --window section).
            window_splits[split_plan.split] = WindowSplitPlan(
                split=split_plan.split,
                camera_relative_paths={cid: tuple(paths) for cid, paths in found.items()},
            )

    plan = ConversionPlan(
        sequence_dir=str(sequence_dir),
        sequence_name=name,
        output_dir=str(output_dir),
        fps=fps,
        extension=extension,
        num_random_points=num_random_points,
        seed=seed,
        scene_top_dir=scene_top_dir,
        splits=tuple(split_plans),
        archive_selection=archive_selection,
        manifest_path=str(resolved_manifest_path),
        source_sha256=source_sha256,
        held_out_camera_ids=held_out,
        candidate_archives=tuple(c.name for c in candidates),
        window=window_spec,
        window_splits=window_splits,
        declared_sizes=declared_sizes,
    )
    return plan, payloads


def plan_to_json(plan: ConversionPlan) -> dict:
    payload = {
        "sequence_dir": plan.sequence_dir,
        "sequence_name": plan.sequence_name,
        "output_dir": plan.output_dir,
        "fps": plan.fps,
        "extension": plan.extension,
        "num_random_points": plan.num_random_points,
        "seed": plan.seed,
        "scene_top_dir": plan.scene_top_dir,
        "manifest_path": plan.manifest_path,
        "source_sha256": plan.source_sha256,
        "candidate_archives": list(plan.candidate_archives),
        "held_out_camera_ids_downstream_only": list(plan.held_out_camera_ids),
        "splits": [
            {
                "split": s.split,
                "source_path": s.source_path,
                "frame_count": s.frame_count,
                "camera_ids": list(s.camera_ids),
                "selected_archive": plan.archive_selection[s.split].archive_path,
                "selected_archive_top_level_dir": plan.archive_selection[s.split].top_level_dir,
            }
            for s in plan.splits
        ],
    }
    if plan.window is not None and plan.window_splits is not None:
        payload["window"] = {
            "start": plan.window.start,
            "end": plan.window.end,
            "stride": plan.window.stride,
            "masks_dir": "masks",
            **window_mask_provenance(plan.archive_selection, plan.declared_sizes),
            "splits": {
                split: {
                    "per_camera_frame_counts": {
                        str(cid): len(paths) for cid, paths in ws.camera_relative_paths.items()
                    },
                    "total_frame_count": sum(len(p) for p in ws.camera_relative_paths.values()),
                    # One derived mask per kept frame -- see mask_source above.
                    "mask_count": sum(len(p) for p in ws.camera_relative_paths.values()),
                }
                for split, ws in plan.window_splits.items()
            },
        }
    return payload


# ---------------------------------------------------------------------------
# Extraction (the only step that writes frame image bytes)
# ---------------------------------------------------------------------------


def extract_wanted_members(
    archive_path: Path,
    archive_top_level_dir: str,
    wanted_relative_paths: Sequence[str],
    destination_root: Path,
    *,
    scene_top_dir: str,
) -> list[Path]:
    """Extract exactly the wanted members into the layout the transforms
    JSON references (``destination_root/scene_top_dir/<relative-path>``)."""

    wanted = set(wanted_relative_paths)
    written: list[Path] = []
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            rest = strip_archive_prefix(member.name, archive_top_level_dir)
            if rest is None or rest not in wanted:
                continue
            target = destination_root / scene_top_dir / rest
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ArtifactError(
                    f"{archive_path}: {member.name} is not a regular extractable file"
                )
            with source, open(target, "wb") as handle:
                handle.write(source.read())
            written.append(target)
            wanted.discard(rest)
            if not wanted:
                break
    if wanted:
        raise ContractError(
            f"{archive_path}: expected member(s) not found during extraction: "
            f"{sorted(wanted)}"
        )
    return written


# ---------------------------------------------------------------------------
# --window mask derivation (frames_1's own alpha channel -- see the module
# docstring's --window section; segmented_ngp/segmented_gt are never used)
# ---------------------------------------------------------------------------


def derive_mask_from_frame(frame_path: Path, mask_path: Path, *, threshold: int = 127) -> None:
    """Derive a binary fg/bg mask from an already-extracted frame's own
    alpha channel: L-mode, ``{0, 255}``, strictly ``> threshold``.

    Real-data finding (Apollo ``unlock``, det tasks 405a7a8d/c662fadf):
    ``frames_1``'s alpha channel IS the per-frame fg matte, already in the
    calibration-aligned pixel space -- continuous alpha, 99.9% concentrated
    at the extremes, same per-frame fg fraction as ``segmented_ngp``'s
    binary alpha at the (wrong, original) resolution. A frame with no
    alpha channel at all is a fail-closed ``ContractError`` naming the
    file, never a silently blank/opaque mask (``Image.convert("RGBA")``
    would otherwise fabricate one).
    """

    from PIL import Image

    with Image.open(frame_path) as image:
        if "A" not in image.getbands():
            raise ContractError(
                f"{frame_path}: extracted frame has no alpha channel -- cannot derive "
                "a mask from it (frames_1's alpha channel is the per-frame fg matte "
                "this converter relies on for --window masks)"
            )
        alpha = image.getchannel("A")
        binary = alpha.point(lambda value: 255 if value > threshold else 0)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    binary.save(mask_path, format="PNG")


# ---------------------------------------------------------------------------
# points3d.ply synthesis (DiVa-360 ships no COLMAP sparse reconstruction)
# ---------------------------------------------------------------------------


def _frame_fov(frame: Mapping[str, Any]) -> tuple[float, float]:
    if "camera_angle_x" in frame and "camera_angle_y" in frame:
        return float(frame["camera_angle_x"]), float(frame["camera_angle_y"])
    fl_x, fl_y, w, h = frame["fl_x"], frame["fl_y"], frame["w"], frame["h"]
    fov_x = 2.0 * math.atan(float(w) / (2.0 * float(fl_x)))
    fov_y = 2.0 * math.atan(float(h) / (2.0 * float(fl_y)))
    return fov_x, fov_y


def frustum_union_bounding_box(
    frames: Sequence[Mapping[str, Any]],
    *,
    near_fraction: float = 0.05,
    far_fraction: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned union of every camera frustum's near/far corners.

    DiVa-360 ships no scene-content geometry, so this approximates "the
    volume the rig is plausibly looking at" from calibration alone: near
    and far are fractions of the camera-ring radius (the same quantity
    ``scene/dataset_readers.py::getNerfppNorm`` computes), and the box is
    the union of each camera's four near-plane corners, four far-plane
    corners, and its own center. This is a coarse smoke-test volume, not a
    content-aware bound -- acceptable for the M1 census smoke per the task
    scope, explicitly NOT a claim-grade initialization.
    """

    if not frames:
        raise ContractError("cannot compute a frustum bounding box from zero frames")
    centers = np.stack(
        [np.asarray(frame["transform_matrix"], dtype=np.float64)[:3, 3] for frame in frames],
        axis=0,
    )
    mean_center = centers.mean(axis=0)
    radius = float(np.linalg.norm(centers - mean_center, axis=1).max())
    radius = max(radius, 1e-3)
    near = radius * near_fraction
    far = radius * far_fraction

    corners = []
    for frame in frames:
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        rotation = c2w[:3, :3]
        center = c2w[:3, 3]
        fov_x, fov_y = _frame_fov(frame)
        for depth in (near, far):
            half_w = depth * math.tan(fov_x / 2.0)
            half_h = depth * math.tan(fov_y / 2.0)
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    # Blender/NeRF-synthetic convention: camera looks down
                    # local -Z (the shipped transform_matrix, before the
                    # reader's own R/T axis flip, is in this convention).
                    local = np.array([sx * half_w, sy * half_h, -depth])
                    corners.append(center + rotation @ local)
        corners.append(center)
    corners_arr = np.stack(corners, axis=0)
    if not np.all(np.isfinite(corners_arr)):
        raise ContractError("frustum corner computation produced non-finite coordinates")
    return corners_arr.min(axis=0), corners_arr.max(axis=0)


def sample_random_points(
    bbox_min: np.ndarray, bbox_max: np.ndarray, num_points: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(num_points, bool) or not isinstance(num_points, int) or num_points <= 0:
        raise SchemaError(f"num_points must be a positive integer, got {num_points!r}")
    rng = np.random.default_rng(seed)
    span = bbox_max - bbox_min
    points = bbox_min + rng.random((num_points, 3)) * span
    colors = rng.integers(0, 256, size=(num_points, 3), dtype=np.uint8)
    return points.astype(np.float64), colors


def write_points3d_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Minimal dependency-free ASCII PLY writer.

    Emits the ``x,y,z,nx,ny,nz,red,green,blue`` vertex schema
    ``scene/dataset_readers.py::storePly``/``fetchPly`` use, without
    depending on the optional ``plyfile`` package (this script is
    stdlib+numpy only).
    """

    if points.shape[0] != colors.shape[0]:
        raise SchemaError("points and colors must have the same length")
    if points.shape[0] == 0:
        raise ContractError("refusing to write an empty points3d.ply")
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    body_lines = [
        f"{x:.6f} {y:.6f} {z:.6f} 0 0 0 {int(r)} {int(g)} {int(b)}"
        for (x, y, z), (r, g, b) in zip(points.tolist(), colors.tolist())
    ]
    path.write_text("\n".join(header + body_lines) + "\n", encoding="utf-8")


def build_window_frame_entry(
    source_frame: Mapping[str, Any],
    *,
    frame_index: int,
    fps: float,
    top_dir: str,
    camera_dir: str,
    index_width: int,
    extension: str,
) -> dict:
    """One temporal frame entry: ``source_frame``'s pose/intrinsics (the rig
    is static, so they hold for every index) + a per-index ``file_path`` +
    ``time = frame_index / fps``.

    ``sharpness`` (if present) is carried through unchanged from the
    single-instant source frame -- it is informational only
    (``scene/dataset_readers.py`` never reads it) and DiVa-360 does not
    ship a per-arbitrary-index sharpness value to substitute instead.
    """

    entry = dict(source_frame)
    entry.pop("time", None)
    entry["file_path"] = schema.window_file_path(top_dir, camera_dir, index_width, frame_index)
    entry["time"] = schema.frame_index_to_time(frame_index, fps)
    return entry


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_info() -> dict:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception:
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--short")
    return {"commit": commit, "branch": branch, "dirty": (bool(status) if status is not None else None)}


def build_provenance(
    plan: ConversionPlan, written_transforms: Mapping[str, Path], ply_path: Path
) -> dict:
    output_files: dict[str, dict] = {}
    for split, path in written_transforms.items():
        output_files[split] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    output_files["points3d_ply"] = {
        "path": ply_path.name,
        "sha256": sha256_file(ply_path),
        "bytes": ply_path.stat().st_size,
    }
    payload = {
        "schema_version": schema.PROVENANCE_SCHEMA,
        "sequence_name": plan.sequence_name,
        "sequence_dir": plan.sequence_dir,
        "source_manifest_path": plan.manifest_path,
        "source_sha256": plan.source_sha256,
        "converter": {
            "script": "scripts/diva360_to_blender.py",
            "argv": sys.argv[1:],
            **_git_info(),
        },
        "fps": plan.fps,
        "extension": plan.extension,
        "num_random_points": plan.num_random_points,
        "seed": plan.seed,
        "scene_top_dir": plan.scene_top_dir,
        "archive_selection": {
            split: {"archive_path": match.archive_path, "top_level_dir": match.top_level_dir}
            for split, match in plan.archive_selection.items()
        },
        "held_out_camera_ids_downstream_only": list(plan.held_out_camera_ids),
        "output_files": output_files,
    }
    if plan.window is not None and plan.window_splits is not None:
        payload["window"] = {
            "start": plan.window.start,
            "end": plan.window.end,
            "stride": plan.window.stride,
            "masks_dir": "masks",
            **window_mask_provenance(plan.archive_selection, plan.declared_sizes),
            "splits": {
                split: {
                    "per_camera_frame_counts": {
                        str(cid): len(paths) for cid, paths in ws.camera_relative_paths.items()
                    },
                    "total_frame_count": sum(len(p) for p in ws.camera_relative_paths.values()),
                    # One derived mask per kept frame -- see mask_source above.
                    "mask_count": sum(len(p) for p in ws.camera_relative_paths.values()),
                }
                for split, ws in plan.window_splits.items()
            },
        }
    return payload


# ---------------------------------------------------------------------------
# Execution (writes)
# ---------------------------------------------------------------------------


def execute_plan(
    plan: ConversionPlan, payloads: Mapping[str, dict], *, overwrite: bool = False
) -> dict:
    sequence_dir = Path(plan.sequence_dir)
    output_dir = Path(plan.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise ArtifactError(
            f"output_dir is non-empty (pass --overwrite to replace it): {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if plan.window is not None:
        # Single-instant mode below is left completely untouched by
        # --window -- this dispatches to an entirely separate execution
        # path rather than threading conditionals through the loop below.
        return _execute_windowed_plan(plan, payloads, sequence_dir, output_dir, overwrite=overwrite)

    written_transforms: dict[str, Path] = {}
    all_stamped_frames: list[dict] = []
    for split_plan in plan.splits:
        payload = payloads[split_plan.split]
        stamped_frames = [
            schema.stamp_frame_time(frame, fps=plan.fps, extension=plan.extension, index=i)
            for i, frame in enumerate(payload["frames"])
        ]
        out_payload = dict(payload)
        out_payload["frames"] = stamped_frames
        schema.validate_transforms_payload(out_payload, extension=plan.extension, allow_time=True)

        dest = output_dir / schema.split_filename(split_plan.split)
        dest.write_text(
            json.dumps(out_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        written_transforms[split_plan.split] = dest
        all_stamped_frames.extend(stamped_frames)

        archive_match = plan.archive_selection[split_plan.split]
        extract_wanted_members(
            sequence_dir / archive_match.archive_path,
            archive_match.top_level_dir,
            split_plan.wanted_relative_paths,
            output_dir,
            scene_top_dir=plan.scene_top_dir,
        )

    bbox_min, bbox_max = frustum_union_bounding_box(all_stamped_frames)
    points, colors = sample_random_points(
        bbox_min, bbox_max, plan.num_random_points, seed=plan.seed
    )
    ply_path = output_dir / "points3d.ply"
    write_points3d_ply(ply_path, points, colors)

    provenance = build_provenance(plan, written_transforms, ply_path)
    provenance_path = output_dir / "diva360_conversion_provenance.json"
    if provenance_path.exists():
        if not overwrite:
            raise ArtifactError(f"refusing to overwrite existing provenance: {provenance_path}")
        provenance_path.unlink()
    atomic_write_json_immutable(provenance_path, provenance)

    return {
        "output_dir": str(output_dir),
        "transforms": {split: str(path) for split, path in written_transforms.items()},
        "points3d_ply": str(ply_path),
        "provenance": str(provenance_path),
    }


def _execute_windowed_plan(
    plan: ConversionPlan,
    payloads: Mapping[str, dict],
    sequence_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> dict:
    """The ``--window`` execution path: crosses each camera's static
    single-instant calibration with the requested frame-index window,
    extracts the matching frames, DERIVES a mask per frame from its own
    alpha channel, then proceeds exactly like the single-instant path for
    ``points3d.ply`` and provenance."""

    assert plan.window is not None and plan.window_splits is not None
    written_transforms: dict[str, Path] = {}
    all_stamped_frames: list[dict] = []

    for split_plan in plan.splits:
        payload = payloads[split_plan.split]
        window_split = plan.window_splits[split_plan.split]
        camera_by_id: dict[int, dict] = {}
        for frame in payload["frames"]:
            relative = schema.frame_relative_path(frame["file_path"], plan.extension)
            _, rest = schema.split_top_level_dir(relative)
            camera_by_id[schema.parse_camera_id(rest)] = frame

        stamped_frames: list[dict] = []
        for camera_id in sorted(window_split.camera_relative_paths):
            source_frame = camera_by_id[camera_id]
            top_dir, camera_dir, index_width = schema.camera_path_template(source_frame["file_path"])
            for relative in window_split.camera_relative_paths[camera_id]:
                stem = relative.rsplit("/", 1)[-1]
                if plan.extension and stem.endswith(plan.extension):
                    stem = stem[: -len(plan.extension)]
                frame_index = int(stem)
                stamped_frames.append(
                    build_window_frame_entry(
                        source_frame,
                        frame_index=frame_index,
                        fps=plan.fps,
                        top_dir=top_dir,
                        camera_dir=camera_dir,
                        index_width=index_width,
                        extension=plan.extension,
                    )
                )

        out_payload = {"frames": stamped_frames}
        schema.validate_transforms_payload(out_payload, extension=plan.extension, allow_time=True)

        dest = output_dir / schema.split_filename(split_plan.split)
        dest.write_text(json.dumps(out_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        written_transforms[split_plan.split] = dest
        all_stamped_frames.extend(stamped_frames)

        frame_archive = plan.archive_selection[split_plan.split]
        flattened = tuple(
            sorted(p for paths in window_split.camera_relative_paths.values() for p in paths)
        )
        written_frames = extract_wanted_members(
            sequence_dir / frame_archive.archive_path,
            frame_archive.top_level_dir,
            flattened,
            output_dir,
            scene_top_dir=plan.scene_top_dir,
        )

        scene_root = output_dir / plan.scene_top_dir
        for frame_path in written_frames:
            mask_path = output_dir / "masks" / frame_path.relative_to(scene_root)
            derive_mask_from_frame(frame_path, mask_path)

    bbox_min, bbox_max = frustum_union_bounding_box(all_stamped_frames)
    points, colors = sample_random_points(bbox_min, bbox_max, plan.num_random_points, seed=plan.seed)
    ply_path = output_dir / "points3d.ply"
    write_points3d_ply(ply_path, points, colors)

    provenance = build_provenance(plan, written_transforms, ply_path)
    provenance_path = output_dir / "diva360_conversion_provenance.json"
    if provenance_path.exists():
        if not overwrite:
            raise ArtifactError(f"refusing to overwrite existing provenance: {provenance_path}")
        provenance_path.unlink()
    atomic_write_json_immutable(provenance_path, provenance)

    return {
        "output_dir": str(output_dir),
        "transforms": {split: str(path) for split, path in written_transforms.items()},
        "points3d_ply": str(ply_path),
        "provenance": str(provenance_path),
        "masks_dir": str(output_dir / "masks"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence-dir", required=True, help="DiVa-360 raw sequence directory")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination derived-data directory (must NOT be the read-only raw tree)",
    )
    parser.add_argument("--sequence-name", default=None, help="Defaults to sequence-dir's basename")
    parser.add_argument(
        "--manifest", default=None, help="Defaults to <sequence-dir>/../MANIFEST.sha256"
    )
    parser.add_argument("--fps", type=float, default=schema.DEFAULT_FPS)
    parser.add_argument("--extension", default=schema.DEFAULT_EXTENSION)
    parser.add_argument("--num-random-points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output-dir")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only discover and print the plan; no writes"
    )
    parser.add_argument(
        "--window",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help=(
            "Activate temporal mode (owner decision D-M1-1): emit one frame per "
            "(camera, index) for index in [START, END] inclusive, stepped by "
            "--stride, crossing each camera's static single-instant calibration "
            "with the requested frame indices, plus a mask per frame derived "
            "from its own alpha channel under masks/. Default: single-instant "
            "mode (unchanged)."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Window-mode frame-index step (default 1). Ignored without --window.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    window = (args.window[0], args.window[1], args.stride) if args.window is not None else None
    plan, payloads = build_plan(
        Path(args.sequence_dir),
        Path(args.output_dir),
        sequence_name=args.sequence_name,
        manifest_path=Path(args.manifest) if args.manifest else None,
        fps=args.fps,
        extension=args.extension,
        num_random_points=args.num_random_points,
        seed=args.seed,
        window=window,
    )
    if args.dry_run:
        print(json.dumps(plan_to_json(plan), indent=2))
        return 0
    result = execute_plan(plan, payloads, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ContractError as exc:
        print(f"FATAL (schema/contract violation): {exc}", file=sys.stderr)
        sys.exit(1)
