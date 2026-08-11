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

Fail-closed: every schema/contract surprise raises a
``depth_visibility.errors.ContractError`` (or a typed subclass -- see
``elgs/diva360_schema.py``); nothing here silently degrades or guesses.

Runtime constraint (Apollo Determined container): stdlib + numpy only.
Never import torch (or anything that imports torch) at module level.
``depth_visibility.errors``, ``depth_visibility.canonical``,
``depth_visibility.artifacts``, and ``elgs.diva360_schema`` are all
verified torch-free at import time (see their own module docstrings /
``depth_visibility/__init__.py``'s own no-Torch guarantee) and are the only
non-stdlib, non-numpy imports here.

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
    top_level_dir: str  # this archive's own internal top-level directory


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


def select_archive_for_split(
    candidates: Sequence[Path], wanted_relative_paths: Sequence[str]
) -> ArchiveMatch:
    """Content-addressed discovery of the archive that has every wanted frame.

    Matches purely on content (member path with the archive's own
    top-level directory stripped) -- never on the archive's filename or an
    assumed internal directory name. Scanning a candidate stops as soon as
    every wanted path has been found in it.
    """

    wanted = set(wanted_relative_paths)
    if not wanted:
        raise ContractError("cannot select an archive for zero wanted frame paths")
    best: ArchiveMatch | None = None
    for archive_path in candidates:
        found: set[str] = set()
        top_dirs: set[str] = set()
        for member_name in _iter_tar_relative_paths(archive_path):
            try:
                top, rest = schema.split_top_level_dir(member_name)
            except SchemaError:
                continue
            top_dirs.add(top)
            if rest in wanted:
                found.add(rest)
            if found == wanted:
                break
        if found == wanted:
            if len(top_dirs) != 1:
                raise ContractError(
                    f"{archive_path}: members span more than one top-level "
                    f"directory in the scanned prefix: {sorted(top_dirs)}"
                )
            match = ArchiveMatch(
                archive_path=archive_path.name, top_level_dir=next(iter(top_dirs))
            )
            if best is not None:
                raise ContractError(
                    f"ambiguous frame source: both {best.archive_path!r} and "
                    f"{archive_path.name!r} contain every referenced frame -- "
                    "cannot discover a unique archive"
                )
            best = match
    if best is None:
        raise ContractError(
            f"no candidate archive contains every referenced frame path "
            f"({len(wanted)} wanted); candidates: {[c.name for c in candidates]}"
        )
    return best


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
) -> tuple[ConversionPlan, dict[str, dict]]:
    """Read-only: validates the source and discovers frame archives.

    Returns ``(plan, {split: source_payload})``. Performs no filesystem
    writes -- safe to call for ``--dry-run``.
    """

    sequence_dir = sequence_dir.resolve()
    if not sequence_dir.is_dir():
        raise SchemaError(f"sequence_dir does not exist or is not a directory: {sequence_dir}")
    name = sequence_name or sequence_dir.name

    transform_files = discover_transform_files(sequence_dir)
    payloads: dict[str, dict] = {}
    split_plans: list[SplitPlan] = []
    scene_top_dirs: set[str] = set()
    for split, path in transform_files.items():
        split_plan, payload = build_split_plan(split, path, extension=extension)
        payloads[split] = payload
        split_plans.append(split_plan)
        for frame in payload["frames"]:
            relative = schema.frame_relative_path(frame["file_path"], extension)
            top, _ = schema.split_top_level_dir(relative)
            scene_top_dirs.add(top)
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
            candidates, split_plan.wanted_relative_paths
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
    )
    return plan, payloads


def plan_to_json(plan: ConversionPlan) -> dict:
    return {
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
            try:
                top, rest = schema.split_top_level_dir(member.name)
            except SchemaError:
                continue
            if top != archive_top_level_dir or rest not in wanted:
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
    return {
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan, payloads = build_plan(
        Path(args.sequence_dir),
        Path(args.output_dir),
        sequence_name=args.sequence_name,
        manifest_path=Path(args.manifest) if args.manifest else None,
        fps=args.fps,
        extension=args.extension,
        num_random_points=args.num_random_points,
        seed=args.seed,
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
