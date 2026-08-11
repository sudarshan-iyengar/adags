"""Frozen tracks artifact schema + synthetic fixture (spec §2 evidence).

The evidence artifact EL-GS consumes is FROZEN before training: per-
camera track reports from 3D surface seeds visible in >= 2 training
cameras, identity by common seed, robust consensus triangulation
(method page). This module defines the artifact schema, its fail-
closed validator, and a synthetic fixture builder — the B0 "tracker-
pipeline dry run" exercises schema construction and validation on the
fixture; the real CoTracker3-class builder script lands with M1 and
must emit exactly this schema.

Held-out cameras NEVER appear in the artifact (validator-enforced
against the manifest's declared held-out list).
"""

from __future__ import annotations

from depth_visibility.errors import ContractError, SchemaError

TRACKS_SCHEMA = "elgs-tracks-artifact-v1"


def validate_tracks_artifact(payload: dict) -> dict:
    """Fail-closed validation; returns the payload on success."""
    if not isinstance(payload, dict):
        raise SchemaError("tracks artifact must be a dict")
    if payload.get("schema_version") != TRACKS_SCHEMA:
        raise SchemaError(
            f"unsupported tracks schema: {payload.get('schema_version')!r}"
        )
    for key in ("seeds", "tracks", "manifest"):
        if key not in payload:
            raise SchemaError(f"tracks artifact missing {key}")
    manifest = payload["manifest"]
    for key in ("training_cameras", "held_out_cameras", "tracker_identity", "source_data_sha256"):
        if key not in manifest:
            raise SchemaError(f"tracks manifest missing {key}")
    held_out = set(manifest["held_out_cameras"])
    training = set(manifest["training_cameras"])
    if held_out & training:
        raise ContractError("held-out cameras overlap training cameras")

    seed_ids = set()
    for seed in payload["seeds"]:
        for key in ("seed_id", "point", "n_cam"):
            if key not in seed:
                raise SchemaError(f"seed record missing {key}")
        if seed["seed_id"] in seed_ids:
            raise ContractError(f"duplicate seed_id {seed['seed_id']}")
        seed_ids.add(seed["seed_id"])
        if int(seed["n_cam"]) < 2:
            raise ContractError(
                f"seed {seed['seed_id']}: seeds require visibility in >= 2 "
                "training cameras (spec evidence rule)"
            )

    track_ids = set()
    for track in payload["tracks"]:
        for key in ("track_id", "seed_id", "camera_id", "reports"):
            if key not in track:
                raise SchemaError(f"track record missing {key}")
        if track["track_id"] in track_ids:
            raise ContractError(f"duplicate track_id {track['track_id']}")
        track_ids.add(track["track_id"])
        if track["seed_id"] not in seed_ids:
            raise ContractError(
                f"track {track['track_id']} references unknown seed "
                f"{track['seed_id']} (identity = common-seed membership)"
            )
        if track["camera_id"] in held_out:
            raise ContractError(
                f"track {track['track_id']} uses held-out camera "
                f"{track['camera_id']}: held-out cameras never enter tracking"
            )
        if track["camera_id"] not in training:
            raise ContractError(
                f"track {track['track_id']} camera {track['camera_id']} "
                "not in the declared training set"
            )
        for report in track["reports"]:
            if "frame" not in report:
                raise SchemaError("report missing frame")
            if not report.get("is_miss", False):
                for key in ("v", "x", "y"):
                    if key not in report:
                        raise SchemaError(f"positional report missing {key}")
    return payload


def build_synthetic_fixture(*, n_seeds: int = 4, n_frames: int = 6) -> dict:
    """A deterministic valid artifact for the B0 dry run."""
    training = [0, 1, 2, 3]
    held_out = [4]
    seeds = [
        {"seed_id": s, "point": [float(s), 0.0, 1.0], "n_cam": 2 + (s % 2)}
        for s in range(n_seeds)
    ]
    tracks = []
    track_id = 0
    for seed in seeds:
        for camera_id in training[: seed["n_cam"]]:
            reports = []
            for frame in range(n_frames):
                if (frame + seed["seed_id"]) % 5 == 4:
                    reports.append({"frame": float(frame), "is_miss": True})
                else:
                    reports.append({
                        "frame": float(frame),
                        "is_miss": False,
                        "v": 0.5 + 0.05 * (frame % 3),
                        "x": 10.0 * seed["seed_id"] + frame,
                        "y": 5.0 + camera_id,
                    })
            tracks.append({
                "track_id": track_id,
                "seed_id": seed["seed_id"],
                "camera_id": camera_id,
                "reports": reports,
            })
            track_id += 1
    return {
        "schema_version": TRACKS_SCHEMA,
        "seeds": seeds,
        "tracks": tracks,
        "manifest": {
            "training_cameras": training,
            "held_out_cameras": held_out,
            "tracker_identity": "synthetic-fixture-v1",
            "source_data_sha256": "0" * 64,
        },
    }


__all__ = ["TRACKS_SCHEMA", "build_synthetic_fixture", "validate_tracks_artifact"]
