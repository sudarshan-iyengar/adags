"""Build the frozen EL-GS multi-view tracks artifact (spec section 2; M1).

Authority: implementation plan section 6 module 20 and section 11.1 item 6;
method page evidence definition ("per-camera CoTracker3-class queries from 3D
surface seeds visible >= 2 training cameras; identity = common-seed
membership; robust consensus triangulation; < 2 cameras => 3D unknown");
artifact schema = ``elgs/tracks_schema.py`` (``elgs-tracks-artifact-v1``,
fail-closed validated before anything is written).

Pipeline (all constants disclosed in the emitted manifest):

1. Load the CONVERTED temporal scene (output of ``diva360_to_blender.py
   --window``): per-camera static-rig calibration from ``transforms_train``
   (rig staticity asserted), per-frame images, per-frame fg/bg masks under
   ``masks/``. Held-out cameras (id congruent 0 mod 4 per
   ``prereg_m1_census_v1.json``, codified in ``elgs.diva360_schema``) are
   structurally excluded: no query, track, or triangulation ever sees them.
2. Seed construction (MODEL-FREE, deterministic; the artifact-level decision
   the heads prereg defers to the artifact manifest, same treatment as the
   r_u mapping): visual-hull carving on a voxel grid over the rig volume
   from the query-frame fg masks; surface voxels (6-neighborhood erosion)
   are the 3D surface seeds; deterministic stride subsampling to the seed
   budget; ``n_cam`` = number of tracking cameras where the seed projects
   in-bounds and mask-positive; seeds require ``n_cam >= 2``.
3. Per-camera queries: the seed's query-frame projection in every tracking
   camera where it is in-bounds and mask-positive.
4. Tracking through a backend interface (``cotracker3`` on GPU, ``fake``
   for plumbing smokes; the backend identity is stamped in the manifest and
   only the cotracker3 backend is admissible for evidence-bearing use).
5. Reports: per (seed, camera, frame) either a positional report
   ``{frame, time, v, x, y}`` or a miss token. Out-of-domain tracker
   positions are converted to miss tokens here (heads prereg: the pipeline
   fail-closes the off-domain case; the evidence domain never sees them).
6. Robust consensus triangulation per (seed, frame): IRLS DLT with Huber
   reweighting over cameras with visible reports; < 2 visible cameras =>
   3D unknown (null).
7. Reliability diagnostics: forward-backward re-query consistency per
   (seed, camera) and consensus reprojection RMS per seed. The r_u mapping
   is declared and FROZEN in the manifest (heads prereg: "exact mapping
   frozen with the tracker artifact manifest").
8. Controls (M1-D): time-shift (+``shift_frames``, out-of-window reports
   dropped) and seed-identity shuffle (fixed RNG seed) derived from the
   frozen primary artifact; each schema-validated and hash-tied to the
   primary in the manifest.

Import-time constraint: torch-free (torch loads lazily inside the
cotracker3 backend only), so the CPU test suite imports this module
directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.artifacts import atomic_write_bytes, atomic_write_json, atomic_write_json_immutable  # noqa: E402
from depth_visibility.camera import camera_center, intrinsics_matrix, opengl_c2w_to_opencv_w2c  # noqa: E402
from depth_visibility.canonical import canonical_json_bytes, sha256_file  # noqa: E402
from depth_visibility.errors import ContractError, SchemaError  # noqa: E402
from elgs import diva360_schema as dschema  # noqa: E402
from elgs.tracks_schema import TRACKS_SCHEMA, validate_tracks_artifact  # noqa: E402

TRACKS_MANIFEST_SCHEMA = "elgs-tracks-manifest-v1"
MASK_BINARIZE_THRESHOLD = 127  # grayscale value strictly above => foreground
R_U_MAPPING = (
    "r_u = clip(min(1 - fb_rms_px / 8.0, 1 - reproj_rms_px / 8.0), 0.05, 1.0); "
    "fb_rms_px per (seed, camera) from the forward-backward re-query check, "
    "reproj_rms_px per seed from consensus triangulation; missing diagnostics => r_min"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TracksConfig:
    query_frames: tuple[int, ...] = (0,)
    hull_resolution: int = 96
    hull_bounds_scale: float = 0.5
    hull_min_observers: int = 3
    hull_mask_agreement: float = 0.9
    max_seeds: int = 512
    min_seed_cameras: int = 2
    vis_threshold: float = 0.5
    triangulation_iters: int = 3
    huber_px: float = 4.0
    fb_check: bool = True
    shift_frames: int = 10
    shuffle_seed: int = 20260811

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["query_frames"] = list(self.query_frames)
        return payload


# ---------------------------------------------------------------------------
# Scene loading (converter --window output)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CameraModel:
    camera_id: int
    name: str
    K: np.ndarray
    w2c: np.ndarray
    width: int
    height: int


@dataclasses.dataclass(frozen=True)
class SceneBundle:
    scene_dir: Path
    masks_root: Path
    cameras: dict[int, CameraModel]
    tracking_ids: tuple[int, ...]
    held_out_ids: tuple[int, ...]
    frame_indices: tuple[int, ...]
    fps: float
    image_template: dict[int, tuple[str, str, int]]
    mask_index_width: dict[int, int]
    source_sha256: str
    source_sha_of: str

    def image_path(self, camera_id: int, frame_index: int) -> Path:
        top, cam_dir, width = self.image_template[camera_id]
        relative = dschema.window_file_path(top, cam_dir, width, frame_index)
        return self.scene_dir / (relative + dschema.DEFAULT_EXTENSION)

    def mask_path(self, camera_id: int, frame_index: int) -> Path:
        name = self.cameras[camera_id].name
        width = self.mask_index_width[camera_id]
        return self.masks_root / name / f"{frame_index:0{width}d}.png"


def _static_rig_matrix(entries: Sequence[Mapping[str, Any]], *, name: str) -> np.ndarray:
    matrices = [np.asarray(e["transform_matrix"], dtype=np.float64) for e in entries]
    for other in matrices[1:]:
        if not np.allclose(other, matrices[0], atol=1e-9, rtol=0.0):
            raise ContractError(
                f"camera {name}: transform_matrix varies across frames; the tracks "
                "builder requires the static-rig temporal conversion"
            )
    return matrices[0]


def _camera_intrinsics(entries: Sequence[Mapping[str, Any]], *, name: str) -> tuple[np.ndarray, int, int]:
    keys = ("fl_x", "fl_y", "cx", "cy", "w", "h")
    first = entries[0]
    for key in keys:
        if key not in first:
            raise SchemaError(f"camera {name}: frame entry missing intrinsic key {key!r}")
    values = tuple(float(first[key]) for key in keys)
    for entry in entries[1:]:
        entry_values = tuple(float(entry[key]) for key in keys)
        if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9) for a, b in zip(values, entry_values)):
            raise ContractError(f"camera {name}: intrinsics vary across frames")
    fl_x, fl_y, cx, cy, w, h = values
    return intrinsics_matrix(fl_x, fl_y, cx, cy), int(round(w)), int(round(h))


def _detect_index_width(directory: Path, *, what: str) -> int:
    if not directory.is_dir():
        raise ContractError(f"{what}: directory missing: {directory}")
    for child in sorted(directory.iterdir()):
        if child.suffix.lower() == ".png" and child.stem.isdigit():
            return len(child.stem)
    raise ContractError(f"{what}: no numeric .png files in {directory}")


def load_temporal_scene(scene_dir: Path, masks_root: Path | None = None) -> SceneBundle:
    scene_dir = Path(scene_dir)
    masks_root = Path(masks_root) if masks_root is not None else scene_dir / "masks"
    transforms_path = scene_dir / dschema.split_filename("train")
    if not transforms_path.is_file():
        raise ContractError(f"missing {transforms_path}")
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise SchemaError("transforms_train has no frames")

    per_camera: dict[int, list[dict[str, Any]]] = {}
    for frame in frames:
        camera_id = dschema.parse_camera_id(str(frame["file_path"]))
        per_camera.setdefault(camera_id, []).append(frame)

    fps = float(payload.get("fps", dschema.DEFAULT_FPS))
    all_ids = sorted(per_camera)
    held_out = sorted(dschema.held_out_camera_ids(all_ids))
    tracking = tuple(cid for cid in all_ids if cid not in set(held_out))
    if len(tracking) < 2:
        raise ContractError(
            f"only {len(tracking)} tracking cameras after the held-out rule; >= 2 required"
        )

    cameras: dict[int, CameraModel] = {}
    image_template: dict[int, tuple[str, str, int]] = {}
    mask_index_width: dict[int, int] = {}
    index_sets: list[set[int]] = []
    for camera_id in tracking:
        entries = per_camera[camera_id]
        match = dschema._CAMERA_ID_RE.search(str(entries[0]["file_path"]))
        name = match.group(0) if match else f"cam{camera_id:02d}"
        c2w = _static_rig_matrix(entries, name=name)
        K, width, height = _camera_intrinsics(entries, name=name)
        cameras[camera_id] = CameraModel(
            camera_id=camera_id,
            name=name,
            K=K,
            w2c=opengl_c2w_to_opencv_w2c(c2w),
            width=width,
            height=height,
        )
        image_template[camera_id] = dschema.camera_path_template(str(entries[0]["file_path"]))
        mask_index_width[camera_id] = _detect_index_width(
            masks_root / name, what=f"masks for {name}"
        )
        index_sets.append({dschema.parse_frame_index(str(e["file_path"])) for e in entries})

    common = sorted(set.intersection(*index_sets))
    if len(common) < 2:
        raise ContractError("tracking cameras share fewer than 2 common frame indices")

    provenance = scene_dir / "provenance.json"
    if provenance.is_file():
        source_sha, source_of = sha256_file(provenance), "provenance.json"
    else:
        source_sha, source_of = sha256_file(transforms_path), dschema.split_filename("train")

    return SceneBundle(
        scene_dir=scene_dir,
        masks_root=masks_root,
        cameras=cameras,
        tracking_ids=tracking,
        held_out_ids=tuple(held_out),
        frame_indices=tuple(common),
        fps=fps,
        image_template=image_template,
        mask_index_width=mask_index_width,
        source_sha256=source_sha,
        source_sha_of=source_of,
    )


def load_mask(path: Path) -> np.ndarray:
    from PIL import Image

    if not path.is_file():
        raise ContractError(f"mask missing: {path}")
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    return array > MASK_BINARIZE_THRESHOLD


# ---------------------------------------------------------------------------
# Projection helpers (behind-camera tolerated, unlike camera.project_world)
# ---------------------------------------------------------------------------


def _project_batch(camera: CameraModel, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (uv (N,2), valid (N,)) with valid = z > 0 and inside the image."""

    cam_points = points @ camera.w2c[:3, :3].T + camera.w2c[:3, 3]
    z = cam_points[:, 2]
    front = z > 1e-9
    uv = np.zeros((len(points), 2))
    if front.any():
        homogeneous = cam_points[front] @ camera.K.T
        uv[front] = homogeneous[:, :2] / homogeneous[:, 2:3]
    inside = (
        front
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= camera.width - 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= camera.height - 1.0)
    )
    return uv, inside


def _mask_positive(mask: np.ndarray, uv: np.ndarray, inside: np.ndarray) -> np.ndarray:
    positive = np.zeros(len(uv), dtype=bool)
    if inside.any():
        cols = np.clip(np.rint(uv[inside, 0]).astype(int), 0, mask.shape[1] - 1)
        rows = np.clip(np.rint(uv[inside, 1]).astype(int), 0, mask.shape[0] - 1)
        positive[inside] = mask[rows, cols]
    return positive


# ---------------------------------------------------------------------------
# Seed construction: visual-hull surface voxels (model-free, deterministic)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Seed:
    seed_id: int
    point: np.ndarray
    n_cam: int
    queries: dict[int, tuple[float, float]]  # camera_id -> query-frame pixel


def build_hull_seeds(
    scene: SceneBundle,
    query_index: int,
    cfg: TracksConfig,
    *,
    mask_loader: Callable[[Path], np.ndarray] = load_mask,
) -> list[Seed]:
    if query_index not in scene.frame_indices:
        raise ContractError(f"query frame {query_index} not in the common frame set")

    centers = np.stack([camera_center(scene.cameras[cid].w2c) for cid in scene.tracking_ids])
    centroid = centers.mean(axis=0)
    radius = float(np.linalg.norm(centers - centroid, axis=1).max())
    if radius <= 0.0:
        raise ContractError("degenerate rig: all camera centers coincide")
    half = radius * cfg.hull_bounds_scale

    res = cfg.hull_resolution
    axis = np.linspace(-half, half, res)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    voxels = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1) + centroid

    observers = np.zeros(len(voxels), dtype=np.int32)
    positives = np.zeros(len(voxels), dtype=np.int32)
    masks: dict[int, np.ndarray] = {}
    for camera_id in scene.tracking_ids:
        camera = scene.cameras[camera_id]
        masks[camera_id] = mask_loader(scene.mask_path(camera_id, query_index))
        uv, inside = _project_batch(camera, voxels)
        observers += inside
        positives += _mask_positive(masks[camera_id], uv, inside)

    agreement_floor = np.ceil(cfg.hull_mask_agreement * observers).astype(np.int32)
    in_hull = (observers >= cfg.hull_min_observers) & (positives >= agreement_floor) & (positives > 0)
    if not in_hull.any():
        raise ContractError(
            f"visual hull empty at query frame {query_index}: masks and calibration "
            "produced no mutually consistent foreground volume"
        )

    hull = in_hull.reshape(res, res, res)
    eroded = hull.copy()
    padded = np.pad(hull, 1, mode="constant", constant_values=False)
    for offset in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        shifted = padded[
            1 + offset[0] : 1 + offset[0] + res,
            1 + offset[1] : 1 + offset[1] + res,
            1 + offset[2] : 1 + offset[2] + res,
        ]
        eroded &= shifted
    surface = hull & ~eroded

    surface_indices = np.flatnonzero(surface.ravel())
    if len(surface_indices) > cfg.max_seeds:
        stride_positions = np.linspace(0, len(surface_indices) - 1, cfg.max_seeds)
        surface_indices = surface_indices[np.unique(stride_positions.astype(int))]

    seeds: list[Seed] = []
    seed_points = voxels[surface_indices]
    per_camera_uv: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    per_camera_positive: dict[int, np.ndarray] = {}
    for camera_id in scene.tracking_ids:
        uv, inside = _project_batch(scene.cameras[camera_id], seed_points)
        per_camera_uv[camera_id] = (uv, inside)
        per_camera_positive[camera_id] = _mask_positive(masks[camera_id], uv, inside)

    for row, point in enumerate(seed_points):
        queries: dict[int, tuple[float, float]] = {}
        for camera_id in scene.tracking_ids:
            uv, inside = per_camera_uv[camera_id]
            if inside[row] and per_camera_positive[camera_id][row]:
                queries[camera_id] = (float(uv[row, 0]), float(uv[row, 1]))
        if len(queries) >= cfg.min_seed_cameras:
            seeds.append(
                Seed(seed_id=len(seeds), point=point.copy(), n_cam=len(queries), queries=queries)
            )
    if not seeds:
        raise ContractError("no hull surface voxel is mask-positive in >= 2 tracking cameras")
    return seeds


# ---------------------------------------------------------------------------
# Tracker backends
# ---------------------------------------------------------------------------


class FakeTracker:
    """Plumbing-smoke backend: reprojects a caller-supplied motion model.

    Never admissible for evidence-bearing artifacts (the manifest stamps the
    backend name; the census submission path checks it). The default motion
    model holds every query fixed with full visibility.
    """

    name = "fake"

    def __init__(
        self,
        motion: Callable[[Sequence[Path], np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self._motion = motion

    @property
    def identity(self) -> dict[str, Any]:
        return {"backend": self.name, "evidence_admissible": False}

    def track(self, image_paths: Sequence[Path], queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._motion is not None:
            xy, vis = self._motion(image_paths, queries)
        else:
            xy = np.repeat(np.asarray(queries)[None, :, 1:3], len(image_paths), axis=0)
            vis = np.ones((len(image_paths), len(queries)))
        return np.asarray(xy, dtype=np.float64), np.asarray(vis, dtype=np.float64)


def chunked_track(
    track_fn: Callable[[Sequence[Path], np.ndarray], tuple[np.ndarray, np.ndarray]],
    image_paths: Sequence[Path],
    queries: np.ndarray,
    *,
    chunk_len: int,
    overlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic temporal chunking with overlap re-query stitching.

    The offline tracker's memory is O(T); battery's 1,519 frames exceed one
    H100 (experiment 12 CUDA OOM: 69.7 GiB allocated + 7.27 GiB request)
    while <= 512 frames fit (flip_book, experiment 11). Sequences with
    T <= chunk_len take the single-call path UNCHANGED (bit-identical to
    the unchunked backend); longer sequences are processed in overlapping
    chunks, each chunk's queries re-anchored at the chunk's first frame
    using the PREVIOUS chunk's tracked positions at that same global frame
    (which lies inside the previous chunk's tail, overlap >= 1). Overlapped
    frames keep the EARLIER chunk's outputs (first-writer wins,
    deterministic). Disclosed in the backend identity and therefore in the
    sealed artifact manifest.
    """

    n_frames, n_queries = len(image_paths), len(queries)
    if chunk_len <= overlap:
        raise ContractError("chunk_len must exceed overlap")
    if n_frames <= chunk_len:
        return track_fn(image_paths, queries)

    xy = np.zeros((n_frames, n_queries, 2))
    vis = np.zeros((n_frames, n_queries))
    written = np.zeros(n_frames, dtype=bool)
    stride = chunk_len - overlap
    start = 0
    previous_queries = np.asarray(queries, dtype=np.float64)
    while start < n_frames:
        stop = min(start + chunk_len, n_frames)
        chunk_paths = image_paths[start:stop]
        if start == 0:
            chunk_queries = previous_queries.copy()
        else:
            # Re-anchor every query at the chunk's first frame using the
            # previous chunk's tracked position at that global frame.
            chunk_queries = np.zeros((n_queries, 3))
            chunk_queries[:, 1:] = anchor_positions
            # t_local = 0: all queries anchored at the chunk's first frame.
        chunk_xy, chunk_vis = track_fn(chunk_paths, chunk_queries)
        expected = (stop - start, n_queries)
        if chunk_xy.shape != (*expected, 2) or chunk_vis.shape != expected:
            raise ContractError(
                f"chunked track_fn returned shapes {chunk_xy.shape}/{chunk_vis.shape}, "
                f"expected {(*expected, 2)}/{expected}"
            )
        for local in range(stop - start):
            frame = start + local
            if not written[frame]:
                xy[frame] = chunk_xy[local]
                vis[frame] = chunk_vis[local]
                written[frame] = True
        if stop >= n_frames:
            break
        next_start = start + stride
        # The next chunk's first frame lies inside this chunk (overlap >= 1).
        anchor_positions = chunk_xy[next_start - start].copy()
        start = next_start
    return xy, vis


class CoTracker3Backend:
    """Offline CoTracker3 (commit-pinned in the apollo-h100-v2 image)."""

    name = "cotracker3"

    def __init__(
        self,
        weights: Path,
        device: str = "cuda",
        window_len: int = 60,
        chunk_len: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        import torch
        from cotracker.predictor import CoTrackerPredictor

        self._torch = torch
        self.weights = Path(weights)
        self.device = device
        self.window_len = window_len
        self.chunk_len = chunk_len
        self.chunk_overlap = chunk_overlap
        self.weights_sha256 = sha256_file(self.weights)
        self.predictor = CoTrackerPredictor(
            checkpoint=str(self.weights), v2=False, offline=True, window_len=window_len
        ).to(device)

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "evidence_admissible": True,
            "weights_path": str(self.weights),
            "weights_sha256": self.weights_sha256,
            "constructor": {"v2": False, "offline": True, "window_len": self.window_len},
            "temporal_chunking": {
                "chunk_len": self.chunk_len,
                "chunk_overlap": self.chunk_overlap,
                "applies_when": f"T > {self.chunk_len}",
                "stitching": "overlap re-query at chunk first frame; first-writer-wins overlap outputs",
            },
            "device": self.device,
        }

    def _load_video(self, image_paths: Sequence[Path]):
        from PIL import Image

        frames = []
        for path in image_paths:
            with Image.open(path) as image:
                frames.append(np.asarray(image.convert("RGB"), dtype=np.float32))
        video = self._torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None]
        return video.to(self.device)

    def _track_single_chunk(
        self, image_paths: Sequence[Path], queries: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        video = self._load_video(image_paths)
        query_tensor = self._torch.from_numpy(np.asarray(queries, dtype=np.float32))[None].to(self.device)
        with self._torch.no_grad():
            pred_tracks, pred_visibility = self.predictor(video, queries=query_tensor)
        xy = pred_tracks[0].detach().cpu().numpy().astype(np.float64)
        vis = pred_visibility[0].detach().cpu().numpy().astype(np.float64)
        del video, query_tensor, pred_tracks, pred_visibility
        self._torch.cuda.empty_cache()
        return xy, np.clip(vis, 0.0, 1.0)

    def track(self, image_paths: Sequence[Path], queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return chunked_track(
            self._track_single_chunk,
            image_paths,
            queries,
            chunk_len=self.chunk_len,
            overlap=self.chunk_overlap,
        )


# ---------------------------------------------------------------------------
# Tracking, reports, triangulation
# ---------------------------------------------------------------------------


def _run_camera_tracks(
    scene: SceneBundle,
    camera_id: int,
    seeds: Sequence[Seed],
    query_local: int,
    backend: Any,
    cfg: TracksConfig,
) -> dict[str, Any]:
    seed_rows = [seed for seed in seeds if camera_id in seed.queries]
    if not seed_rows:
        return {"seed_ids": [], "xy": None, "vis": None, "fb_rms": {}}
    image_paths = [scene.image_path(camera_id, idx) for idx in scene.frame_indices]
    for path in image_paths:
        if not path.is_file():
            raise ContractError(f"frame image missing: {path}")
    queries = np.array(
        [[float(query_local), *seed.queries[camera_id]] for seed in seed_rows], dtype=np.float64
    )
    xy, vis = backend.track(image_paths, queries)
    n_frames = len(image_paths)
    if xy.shape != (n_frames, len(seed_rows), 2) or vis.shape != (n_frames, len(seed_rows)):
        raise ContractError(
            f"backend returned shapes {xy.shape}/{vis.shape}, expected "
            f"({n_frames}, {len(seed_rows)}, 2)/({n_frames}, {len(seed_rows)})"
        )

    fb_rms: dict[int, float] = {}
    if cfg.fb_check:
        back_queries = np.array(
            [[0.0, xy[-1, col, 0], xy[-1, col, 1]] for col in range(len(seed_rows))],
            dtype=np.float64,
        )
        xy_back, _ = backend.track(list(reversed(image_paths)), back_queries)
        for col, seed in enumerate(seed_rows):
            forward = xy[:, col]
            backward = xy_back[::-1, col]
            fb_rms[seed.seed_id] = float(
                np.sqrt(np.mean(np.sum((forward - backward) ** 2, axis=1)))
            )

    return {"seed_ids": [s.seed_id for s in seed_rows], "xy": xy, "vis": vis, "fb_rms": fb_rms}


def _triangulate(
    observations: Sequence[tuple[CameraModel, tuple[float, float]]],
    cfg: TracksConfig,
) -> tuple[np.ndarray, float] | None:
    if len(observations) < 2:
        return None
    if cfg.triangulation_iters < 1:
        raise ContractError("triangulation_iters must be >= 1")
    projections = [camera.K @ camera.w2c[:3, :] for camera, _ in observations]
    weights = np.ones(len(observations))
    point = None
    for _ in range(cfg.triangulation_iters):
        rows = []
        for weight, (proj, (_, uv)) in zip(weights, zip(projections, observations)):
            x, y = uv
            rows.append(weight * (x * proj[2] - proj[0]))
            rows.append(weight * (y * proj[2] - proj[1]))
        matrix = np.stack(rows)
        _, _, vt = np.linalg.svd(matrix)
        homogeneous = vt[-1]
        if abs(homogeneous[3]) < 1e-12:
            return None
        point = homogeneous[:3] / homogeneous[3]
        residuals = []
        for proj, (_, uv) in zip(projections, observations):
            projected = proj @ np.append(point, 1.0)
            if projected[2] <= 1e-9:
                residuals.append(np.inf)
                continue
            residuals.append(
                float(np.hypot(projected[0] / projected[2] - uv[0], projected[1] / projected[2] - uv[1]))
            )
        weights = np.array(
            [1.0 if r <= cfg.huber_px else (cfg.huber_px / r if np.isfinite(r) else 0.0) for r in residuals]
        )
        if weights.sum() < 1.0:
            return None
    finite = [r for r in residuals if np.isfinite(r)]
    if point is None or not finite:
        return None
    return point, float(np.sqrt(np.mean(np.square(finite))))


def build_artifact(
    scene: SceneBundle,
    seeds: Sequence[Seed],
    backend: Any,
    cfg: TracksConfig,
    query_index: int,
) -> dict[str, Any]:
    query_local = scene.frame_indices.index(query_index)
    camera_runs = {
        camera_id: _run_camera_tracks(scene, camera_id, seeds, query_local, backend, cfg)
        for camera_id in scene.tracking_ids
    }

    tracks: list[dict[str, Any]] = []
    positional: dict[tuple[int, int], dict[int, tuple[float, float]]] = {}
    fb_diagnostics: dict[str, float] = {}
    track_id = 0
    for camera_id in scene.tracking_ids:
        run = camera_runs[camera_id]
        camera = scene.cameras[camera_id]
        for col, seed_id in enumerate(run["seed_ids"]):
            reports = []
            for local, frame_index in enumerate(scene.frame_indices):
                x, y = float(run["xy"][local, col, 0]), float(run["xy"][local, col, 1])
                v = float(run["vis"][local, col])
                in_domain = 0.0 <= x <= camera.width - 1.0 and 0.0 <= y <= camera.height - 1.0
                base = {
                    "frame": float(frame_index),
                    "time": dschema.frame_index_to_time(frame_index, scene.fps),
                }
                if in_domain:
                    reports.append({**base, "is_miss": False, "v": v, "x": x, "y": y})
                    if v >= cfg.vis_threshold:
                        positional.setdefault((seed_id, frame_index), {})[camera_id] = (x, y)
                else:
                    reports.append({**base, "is_miss": True})
            tracks.append(
                {"track_id": track_id, "seed_id": seed_id, "camera_id": camera_id, "reports": reports}
            )
            track_id += 1
            if seed_id in run["fb_rms"]:
                fb_diagnostics[f"{seed_id}:{camera_id}"] = run["fb_rms"][seed_id]

    consensus: dict[str, list[dict[str, Any]]] = {}
    reproj_by_seed: dict[int, list[float]] = {}
    for seed in seeds:
        entries = []
        for frame_index in scene.frame_indices:
            observed = positional.get((seed.seed_id, frame_index), {})
            observations = [
                (scene.cameras[camera_id], uv) for camera_id, uv in sorted(observed.items())
            ]
            solved = _triangulate(observations, cfg)
            if solved is None:
                entries.append({"frame": float(frame_index), "point": None, "n_cam": len(observed)})
            else:
                point, rms = solved
                entries.append(
                    {
                        "frame": float(frame_index),
                        "point": [float(c) for c in point],
                        "n_cam": len(observed),
                        "reproj_rms": rms,
                    }
                )
                reproj_by_seed.setdefault(seed.seed_id, []).append(rms)
        consensus[str(seed.seed_id)] = entries

    artifact = {
        "schema_version": TRACKS_SCHEMA,
        "seeds": [
            {"seed_id": s.seed_id, "point": [float(c) for c in s.point], "n_cam": s.n_cam}
            for s in seeds
        ],
        "tracks": tracks,
        "consensus": consensus,
        "diagnostics": {
            "fb_rms_px": fb_diagnostics,
            "reproj_rms_px": {
                str(sid): float(np.mean(values)) for sid, values in sorted(reproj_by_seed.items())
            },
            "r_u_mapping": R_U_MAPPING,
        },
        "window": {
            "frame_indices": [int(i) for i in scene.frame_indices],
            "query_frame": int(query_index),
            "fps": scene.fps,
        },
        "manifest": {
            "training_cameras": [int(c) for c in scene.tracking_ids],
            "held_out_cameras": [int(c) for c in scene.held_out_ids],
            "tracker_identity": json.dumps(backend.identity, sort_keys=True),
            "source_data_sha256": scene.source_sha256,
        },
    }
    return validate_tracks_artifact(artifact)


# ---------------------------------------------------------------------------
# Controls (M1-D)
# ---------------------------------------------------------------------------


def make_shift_control(artifact: dict[str, Any], shift_frames: int) -> dict[str, Any]:
    window = set(float(i) for i in artifact["window"]["frame_indices"])
    fps = float(artifact["window"]["fps"])
    control = json.loads(json.dumps(artifact))
    for track in control["tracks"]:
        shifted = []
        for report in track["reports"]:
            frame = report["frame"] + shift_frames
            if frame in window:
                shifted.append(
                    {**report, "frame": frame, "time": dschema.frame_index_to_time(int(frame), fps)}
                )
        track["reports"] = shifted
    for entries in control["consensus"].values():
        for entry in entries:
            entry["frame"] = entry["frame"] + shift_frames
        entries[:] = [e for e in entries if e["frame"] in window]
    control["control"] = {"kind": "shift", "shift_frames": shift_frames}
    return validate_tracks_artifact(control)


def make_shuffle_control(artifact: dict[str, Any], shuffle_seed: int) -> dict[str, Any]:
    control = json.loads(json.dumps(artifact))
    seed_ids = [s["seed_id"] for s in control["seeds"]]
    rng = np.random.default_rng(shuffle_seed)
    permuted = [seed_ids[i] for i in rng.permutation(len(seed_ids))]
    mapping = dict(zip(seed_ids, permuted))
    for track in control["tracks"]:
        track["seed_id"] = mapping[track["seed_id"]]
    control["consensus"] = {
        str(mapping[int(key)]): value for key, value in control["consensus"].items()
    }
    control["control"] = {
        "kind": "shuffle",
        "shuffle_seed": shuffle_seed,
        "mapping": {str(key): int(value) for key, value in mapping.items()},
    }
    return validate_tracks_artifact(control)


# ---------------------------------------------------------------------------
# Frozen-dir emission
# ---------------------------------------------------------------------------


def write_frozen_artifact_dir(
    out_dir: Path,
    artifact: dict[str, Any],
    cfg: TracksConfig,
    scene: SceneBundle,
    backend: Any,
    *,
    wall_seconds: float,
    controls: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "MANIFEST.json").exists():
        raise ContractError(f"frozen artifact dir already sealed: {out_dir}")

    # The artifact and its controls are written as PLAIN strict JSON (floats
    # as standard JSON numbers, exact binary64 round-trip via Python's
    # shortest-repr float encoding) -- NOT via the canonical writer, whose
    # binary64_hex float tokens are an identity/hashing convention with no
    # repo decoder and would make the artifact numerically unreadable to the
    # census evaluator and the independent recomputation worker (defect
    # caught on the first sealed real artifact, experiment 8 / retry 2).
    # The MANIFEST stays canonical: it is identity-bearing metadata.
    def _write_plain_json(path: Path, payload: dict[str, Any]) -> Path:
        body = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        return atomic_write_bytes(path, body.encode("utf-8") + b"\n")

    files: dict[str, str] = {}
    primary = _write_plain_json(out_dir / "tracks.json", artifact)
    files["tracks.json"] = sha256_file(primary)
    if controls:
        shift = make_shift_control(artifact, cfg.shift_frames)
        shuffle = make_shuffle_control(artifact, cfg.shuffle_seed)
        files["tracks_shift.json"] = sha256_file(_write_plain_json(out_dir / "tracks_shift.json", shift))
        files["tracks_shuffle.json"] = sha256_file(
            _write_plain_json(out_dir / "tracks_shuffle.json", shuffle)
        )

    manifest = {
        "schema_version": TRACKS_MANIFEST_SCHEMA,
        "files_sha256": files,
        "primary": "tracks.json",
        "controls_of_primary_sha256": files["tracks.json"],
        "config": cfg.as_dict(),
        "config_sha256": hashlib.sha256(canonical_json_bytes(cfg.as_dict())).hexdigest(),
        "seed_constructor": {
            "kind": "visual-hull-surface-voxels",
            "mask_binarize_threshold": MASK_BINARIZE_THRESHOLD,
            "disclosure": (
                "deterministic model-free carving from query-frame fg masks + static "
                "calibration; surface = 6-neighborhood hull boundary; stride subsample"
            ),
        },
        "r_u_mapping": R_U_MAPPING,
        "tracker_identity": backend.identity,
        "scene_dir": str(scene.scene_dir),
        "source_sha256": {"value": scene.source_sha256, "of": scene.source_sha_of},
        "training_cameras": [int(c) for c in scene.tracking_ids],
        "held_out_cameras": [int(c) for c in scene.held_out_ids],
        "window": {
            "frame_indices_first": int(scene.frame_indices[0]),
            "frame_indices_last": int(scene.frame_indices[-1]),
            "n_frames": len(scene.frame_indices),
            "fps": scene.fps,
        },
        "accounting": {
            "category": "preprocessing",
            "wall_seconds": wall_seconds,
            "note": "reproducibility accounting only; no ceiling applies (plan 11.1 item 6)",
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json_immutable(out_dir / "MANIFEST.json", manifest)
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--masks-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--query-frame", type=int, action="append", default=None)
    parser.add_argument("--tracker", choices=("cotracker3", "fake"), default="cotracker3")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seeds", type=int, default=TracksConfig.max_seeds)
    parser.add_argument("--hull-resolution", type=int, default=TracksConfig.hull_resolution)
    parser.add_argument("--shift-frames", type=int, default=TracksConfig.shift_frames)
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--no-fb-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="stop after seeds+queries; no tracker")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    query_frames = tuple(args.query_frame) if args.query_frame else (0,)
    if len(query_frames) != 1:
        raise ContractError("exactly one query frame per artifact in v1 (one seeding instant)")
    cfg = TracksConfig(
        query_frames=query_frames,
        max_seeds=args.max_seeds,
        hull_resolution=args.hull_resolution,
        shift_frames=args.shift_frames,
        fb_check=not args.no_fb_check,
    )
    started = time.monotonic()
    scene = load_temporal_scene(args.scene_dir, args.masks_root)
    query_index = query_frames[0] if query_frames[0] in scene.frame_indices else scene.frame_indices[0]
    seeds = build_hull_seeds(scene, query_index, cfg)
    print(
        json.dumps(
            {
                "tracking_cameras": list(scene.tracking_ids),
                "held_out_cameras": list(scene.held_out_ids),
                "n_frames": len(scene.frame_indices),
                "query_frame": query_index,
                "n_seeds": len(seeds),
                "queries_per_camera": {
                    str(cid): sum(1 for s in seeds if cid in s.queries) for cid in scene.tracking_ids
                },
            }
        )
    )
    if args.dry_run:
        atomic_write_json(
            Path(args.out_dir) / "dry_run_report.json",
            {
                "dry_run": True,
                "n_seeds": len(seeds),
                "query_frame": query_index,
                "n_frames": len(scene.frame_indices),
            },
        )
        return 0

    if args.tracker == "cotracker3":
        if args.weights is None:
            raise ContractError("--weights required for the cotracker3 backend")
        backend: Any = CoTracker3Backend(args.weights, device=args.device)
    else:
        backend = FakeTracker()

    artifact = build_artifact(scene, seeds, backend, cfg, query_index)
    write_frozen_artifact_dir(
        args.out_dir,
        artifact,
        cfg,
        scene,
        backend,
        wall_seconds=time.monotonic() - started,
        controls=not args.no_controls,
    )
    print(json.dumps({"sealed": str(args.out_dir), "n_tracks": len(artifact["tracks"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
