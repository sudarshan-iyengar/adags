"""Sealed tracks-artifact loading for the live round path (spec §2 evidence).

`scripts/build_elgs_tracks.py` SEALS a frozen artifact directory
(`tracks.json` + controls + an immutable canonical `MANIFEST.json`).
This module is the consumer side: it discovers that directory, validates
it against the scene the trainer actually loaded, and refuses — never
degrades — on any incompatibility.

Everything here is FAIL-CLOSED by design. The failure mode this guards
against is a run that silently scores one sequence's geometry against
another sequence's tracks, or against a superseded artifact from the
substrate defect era (`research-wiki/operations/elgs-substrate-defect-2026-08-13.md`).
Both would produce evidence that looks well-formed and means nothing, so
every mismatch raises :class:`TracksIncompatible` with the specific
reason rather than falling back to photometric-only.

Checks performed, in order (each has its own refusal reason so the
tests can assert them individually):

  1. directory / `MANIFEST.json` / primary file present;
  2. manifest schema id;
  3. the primary file's SHA-256 matches the manifest's `files_sha256`;
  4. artifact schema (delegated to `elgs.tracks_schema`);
  5. not superseded (a `SUPERSEDED` marker file, or a manifest
     `superseded_by` key, retires an artifact permanently);
  6. tracker backend evidence-admissibility (the `fake` backend stamps
     `evidence_admissible: false` and may only be consumed by a run that
     has declared itself non-evidence-bearing);
  7. sequence identity: the manifest's `scene_dir` basename must equal
     the scene the trainer loaded (CROSS-SEQUENCE ISOLATION);
  8. calibration/substrate identity: the manifest's `source_sha256`
     must match the scene's own `transforms_train.json` digest;
  9. camera sets: every artifact training camera must exist in the
     scene, and no held-out camera may appear;
 10. frame range: the artifact's window must lie inside the frames the
     scene actually loaded.

Torch-free and filesystem-only, so the CPU suite exercises it directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from depth_visibility.canonical import sha256_file
from depth_visibility.errors import ContractError

from .tracks_schema import validate_tracks_artifact

TRACKS_MANIFEST_SCHEMA = "elgs-tracks-manifest-v1"

#: Written by `write_frozen_artifact_dir`; the artifact the evidence path
#: consumes is always the PRIMARY, never a control (the shift/shuffle
#: controls are M1-D falsification arms, not evidence).
PRIMARY_FILENAME = "tracks.json"
MANIFEST_FILENAME = "MANIFEST.json"
SUPERSEDED_MARKER = "SUPERSEDED"


class TracksIncompatible(ContractError):
    """A sealed artifact may not be consumed by this run.

    Carries a stable ``reason`` slug so tests and logs can distinguish
    the ten refusal paths without string-matching the message.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"[{reason}] {message}")
        self.reason = reason


@dataclass(frozen=True)
class SealedTracks:
    """One validated artifact, bound to the scene that will consume it."""

    directory: Path
    artifact: dict
    manifest: dict
    sequence_id: str
    training_cameras: tuple[int, ...]
    held_out_cameras: tuple[int, ...]
    frame_indices: tuple[int, ...]
    fps: float
    evidence_admissible: bool
    primary_sha256: str

    @property
    def seeds(self) -> list:
        return self.artifact["seeds"]

    @property
    def tracks(self) -> list:
        return self.artifact["tracks"]

    @property
    def consensus(self) -> dict:
        return self.artifact.get("consensus", {})

    def provenance(self) -> dict:
        """The identity block a run manifest / summary should record."""
        return {
            "directory": str(self.directory),
            "sequence_id": self.sequence_id,
            "primary_sha256": self.primary_sha256,
            "config_sha256": self.manifest.get("config_sha256"),
            "tracker_identity": self.manifest.get("tracker_identity"),
            "evidence_admissible": self.evidence_admissible,
            "n_seeds": len(self.seeds),
            "n_tracks": len(self.tracks),
            "frame_first": self.frame_indices[0] if self.frame_indices else None,
            "frame_last": self.frame_indices[-1] if self.frame_indices else None,
        }


def _read_json(path: Path, reason: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TracksIncompatible(reason, f"cannot read {path}: {exc}") from exc


def _tracker_identity(manifest: Mapping[str, Any]) -> dict:
    """`tracker_identity` is a dict in MANIFEST.json and a JSON STRING in
    the artifact's own manifest block (`build_artifact` json.dumps-es it);
    accept either so the check does not depend on which one is read."""
    identity = manifest.get("tracker_identity")
    if isinstance(identity, str):
        try:
            identity = json.loads(identity)
        except json.JSONDecodeError:
            return {}
    return identity if isinstance(identity, dict) else {}


def load_sealed_tracks(
    tracks_dir: str | Path,
    *,
    scene_sequence_id: str,
    scene_camera_ids: Sequence[int],
    scene_frame_indices: Sequence[int] | None = None,
    scene_source_sha256: str | None = None,
    require_evidence_admissible: bool = True,
) -> SealedTracks:
    """Load and validate one sealed artifact directory, or refuse.

    ``scene_sequence_id`` is the basename of the converted scene
    directory the trainer loaded; ``scene_camera_ids`` every camera id
    present in that scene; ``scene_frame_indices`` the DiVa-360 frame
    indices the scene actually carries (``None`` skips check 10 for a
    scene whose indices are not recoverable); ``scene_source_sha256``
    the scene's own ``transforms_train.json`` digest (``None`` skips
    check 8). ``require_evidence_admissible=False`` is only valid for a
    run the submission wrapper has stamped ``evidence_bearing: false``.
    """

    directory = Path(tracks_dir)
    if not directory.is_dir():
        raise TracksIncompatible(
            "missing_directory", f"tracks directory does not exist: {directory}"
        )

    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise TracksIncompatible(
            "unsealed_directory",
            f"{directory} has no {MANIFEST_FILENAME}: an unsealed tracks "
            "directory is never admissible evidence",
        )
    manifest = _read_json(manifest_path, "unreadable_manifest")

    if manifest.get("schema_version") != TRACKS_MANIFEST_SCHEMA:
        raise TracksIncompatible(
            "manifest_schema",
            f"unsupported tracks manifest schema "
            f"{manifest.get('schema_version')!r} (expected {TRACKS_MANIFEST_SCHEMA})",
        )

    # 5. Supersession is checked BEFORE the expensive hash+parse: a
    # superseded artifact is retired regardless of its integrity.
    if (directory / SUPERSEDED_MARKER).exists() or manifest.get("superseded_by"):
        raise TracksIncompatible(
            "superseded",
            f"{directory} is marked superseded "
            f"({manifest.get('superseded_by') or SUPERSEDED_MARKER!r}); "
            "superseded artifacts are preserved, never consumed",
        )

    primary_name = str(manifest.get("primary", PRIMARY_FILENAME))
    primary_path = directory / primary_name
    if not primary_path.is_file():
        raise TracksIncompatible(
            "missing_primary", f"primary artifact missing: {primary_path}"
        )

    # 3. Integrity: the sealed digest is the artifact's identity.
    declared = (manifest.get("files_sha256") or {}).get(primary_name)
    if not declared:
        raise TracksIncompatible(
            "unhashed_primary",
            f"manifest files_sha256 has no entry for {primary_name!r}",
        )
    actual = sha256_file(primary_path)
    if actual != declared:
        raise TracksIncompatible(
            "hash_mismatch",
            f"{primary_path} sha256 {actual} != sealed {declared}: the "
            "artifact has been modified since it was sealed",
        )

    artifact = _read_json(primary_path, "unreadable_primary")
    try:
        validate_tracks_artifact(artifact)
    except ContractError as exc:
        raise TracksIncompatible("artifact_schema", str(exc)) from exc

    # A CONTROL is never evidence. `make_shift_control` /
    # `make_shuffle_control` stamp `control: {"kind": ...}` and their
    # digests sit in the SAME `files_sha256` map, so editing the
    # manifest's `primary` field would otherwise promote a falsification
    # arm past every other check with its seal intact.
    control = artifact.get("control")
    if control:
        raise TracksIncompatible(
            "control_artifact",
            f"{primary_path} is the {control.get('kind')!r} CONTROL arm, not "
            "the primary artifact; controls are falsification arms and are "
            "never consumed as evidence",
        )

    # 6. Backend admissibility.
    identity = _tracker_identity(manifest)
    admissible = bool(identity.get("evidence_admissible", False))
    if require_evidence_admissible and not admissible:
        raise TracksIncompatible(
            "backend_not_admissible",
            f"tracker backend {identity.get('backend')!r} stamps "
            "evidence_admissible=false; only an explicitly "
            "non-evidence-bearing run may consume it",
        )

    # 7. Cross-sequence isolation.
    manifest_scene = manifest.get("scene_dir")
    if not manifest_scene:
        raise TracksIncompatible(
            "unidentified_sequence", "manifest carries no scene_dir"
        )
    sequence_id = Path(str(manifest_scene).replace("\\", "/")).name
    if sequence_id != scene_sequence_id:
        raise TracksIncompatible(
            "sequence_mismatch",
            f"artifact was built for sequence {sequence_id!r} but the "
            f"trainer loaded {scene_sequence_id!r}: refusing to score one "
            "sequence's geometry against another's tracks",
        )

    # 8. Calibration / substrate identity.
    if scene_source_sha256 is not None:
        recorded = manifest.get("source_sha256")
        recorded_value = (
            recorded.get("value") if isinstance(recorded, Mapping) else recorded
        )
        if recorded_value != scene_source_sha256:
            raise TracksIncompatible(
                "substrate_mismatch",
                f"artifact source_sha256 {recorded_value!r} != scene "
                f"{scene_source_sha256!r}: the tracks were built on a "
                "different conversion of this sequence",
            )

    # 9. Camera sets.
    training = tuple(int(c) for c in manifest.get("training_cameras", ()))
    held_out = tuple(int(c) for c in manifest.get("held_out_cameras", ()))
    if not training:
        raise TracksIncompatible(
            "no_training_cameras", "manifest declares no training cameras"
        )
    scene_ids = {int(c) for c in scene_camera_ids}
    unknown = sorted(set(training) - scene_ids)
    if unknown:
        raise TracksIncompatible(
            "camera_mismatch",
            f"artifact training cameras {unknown} are absent from the "
            "loaded scene",
        )
    leaked = sorted(set(held_out) & set(training))
    if leaked:
        raise TracksIncompatible(
            "held_out_leak",
            f"cameras {leaked} are declared both held-out and training",
        )

    # 10. Frame range.
    window = manifest.get("window") or {}
    artifact_window = artifact.get("window") or {}
    frame_indices = tuple(int(i) for i in artifact_window.get("frame_indices", ()))
    if not frame_indices:
        raise TracksIncompatible(
            "no_frame_window", "artifact carries no window.frame_indices"
        )
    if scene_frame_indices is not None:
        scene_frames = {int(i) for i in scene_frame_indices}
        outside = sorted(set(frame_indices) - scene_frames)
        if outside:
            raise TracksIncompatible(
                "frame_range_mismatch",
                f"{len(outside)} artifact frames (e.g. {outside[:5]}) are "
                "absent from the loaded scene's frame set",
            )

    fps = float(artifact_window.get("fps") or window.get("fps") or 0.0)
    if fps <= 0.0:
        raise TracksIncompatible("no_fps", "artifact window declares no positive fps")

    return SealedTracks(
        directory=directory,
        artifact=artifact,
        manifest=manifest,
        sequence_id=sequence_id,
        training_cameras=training,
        held_out_cameras=held_out,
        frame_indices=frame_indices,
        fps=fps,
        evidence_admissible=admissible,
        primary_sha256=actual,
    )


__all__ = [
    "MANIFEST_FILENAME",
    "PRIMARY_FILENAME",
    "SUPERSEDED_MARKER",
    "TRACKS_MANIFEST_SCHEMA",
    "SealedTracks",
    "TracksIncompatible",
    "load_sealed_tracks",
]
