"""Training-time E1 external reprojection visibility evidence (CSVL-VPL v2).

Component 2 of the CSVL-VPL v2 pipeline. ``EvidenceRuntime`` answers, for a
batch of primitive world positions and one ``(camera, frame)`` view, whether
each primitive sits near the externally observed consensus surface, is hidden
behind it, sits in front of it, must be abstained on, or cannot be evaluated
at all.

The evidence itself is the frozen offline consensus artifact written by
``scripts/build_evidence_consensus.py``: a directory of uncompressed ``.npy``
files (``d``/``sigma`` fp16 ``[C, T, H, W]``, ``valid`` uint8 ``[C, T, H, W]``,
``intrinsics`` f64 ``[C, 3, 3]``, ``w2c`` f64 ``[C, 4, 4]``, plus
``meta.json``). Nothing here reads RGB, annotations, evaluator masks, or the
Gaussian model.

Verdict semantics mirror ``depth_visibility.primitive_census.classify_states_v2``
so that the training-time evidence and the Phase 0 census cannot silently
diverge:

* same projection convention ``x_cam = R x_world + t`` with nearest-pixel
  rounding and the same in-view/near-clip/valid-pixel gating;
* same margin rule ``margin = max(tau_rel * d, kappa * sigma)``;
* same gap rule ``z - d >= k_gap * margin`` (census ``gap_raw``).

The census ``STATE_BEHIND`` class is split here into ``VERDICT_OCCLUDED`` (the
census gap class) and ``VERDICT_BEHIND_WEAK`` (behind but inside the gap
multiple), and one class is added that the census does not have:
``VERDICT_UNCERTAIN``, a material abstention triggered by an absolute-units
consensus sigma above ``sigma_abstain``. The blinded forensic audit found
sigma blowups on glass and specular regions; abstaining beats emitting a
confident wrong verdict there.

There is deliberately no witness logic at runtime. Multi-view agreement is the
consumer's job (the lifecycle manager applies an all-batch-views AND rule).

Hot-path rules: torch only, device agnostic (CPU in tests, CUDA in training),
fp16 map storage, float64 projection (nearest-pixel rounding is discontinuous,
so the projection is kept in the census dtype), float32 depth comparison after
the gather.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .errors import ArtifactError, CameraConventionError, ContractError, SchemaError

# ---------------------------------------------------------------------------
# Artifact layout
# ---------------------------------------------------------------------------

CONSENSUS_SCHEMA_VERSION = "csvl-vpl-v2-evidence-consensus-v1"

MAP_FILES = {"d": "d.npy", "sigma": "sigma.npy", "valid": "valid.npy"}
GEOMETRY_FILES = {"intrinsics": "intrinsics.npy", "w2c": "w2c.npy"}
META_FILE = "meta.json"
ARRAY_FILES = {**MAP_FILES, **GEOMETRY_FILES}

MAP_DTYPES = {"d": np.float16, "sigma": np.float16, "valid": np.uint8}

# ---------------------------------------------------------------------------
# Verdict codes
# ---------------------------------------------------------------------------

VERDICT_NOT_EVALUABLE = 0
VERDICT_NEAR = 1
VERDICT_OCCLUDED = 2
VERDICT_BEHIND_WEAK = 3
VERDICT_IN_FRONT = 4
VERDICT_UNCERTAIN = 5

NUM_VERDICTS = 6

VERDICT_NAMES = (
    "not_evaluable",
    "near",
    "occluded",
    "behind_weak",
    "in_front",
    "uncertain",
)

MODE_VALID = "valid"
MODE_TIME_SHIFT = "time_shift"
MODES = (MODE_VALID, MODE_TIME_SHIFT)


# ---------------------------------------------------------------------------
# Artifact writing (offline builder and fixtures share this layout)
# ---------------------------------------------------------------------------


def open_consensus_maps(
    out_dir: str | Path,
    *,
    num_cameras: int,
    num_frames: int,
    height: int,
    width: int,
) -> dict[str, np.memmap]:
    """Create the three writable ``[C, T, H, W]`` map memmaps in ``out_dir``.

    Uncompressed ``.npy`` so the training-time runtime can memory-map or bulk
    load them; the offline builder fills them one frame at a time.
    """

    from numpy.lib.format import open_memmap

    for name, value in (
        ("num_cameras", num_cameras),
        ("num_frames", num_frames),
        ("height", height),
        ("width", width),
    ):
        if int(value) <= 0:
            raise ContractError(f"consensus artifact {name} must be positive")
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    shape = (int(num_cameras), int(num_frames), int(height), int(width))
    maps: dict[str, np.memmap] = {}
    for key, filename in MAP_FILES.items():
        maps[key] = open_memmap(
            target / filename, mode="w+", dtype=MAP_DTYPES[key], shape=shape
        )
    return maps


def write_consensus_geometry(
    out_dir: str | Path, intrinsics: np.ndarray, w2c: np.ndarray
) -> None:
    """Write per-camera ``intrinsics`` (C,3,3) and ``w2c`` (C,4,4) as f64."""

    k = np.asarray(intrinsics, dtype=np.float64)
    m = np.asarray(w2c, dtype=np.float64)
    if k.ndim != 3 or k.shape[1:] != (3, 3):
        raise ContractError("intrinsics must be (C, 3, 3)")
    if m.ndim != 3 or m.shape[1:] != (4, 4):
        raise ContractError("w2c must be (C, 4, 4)")
    if k.shape[0] != m.shape[0]:
        raise ContractError("intrinsics and w2c disagree on the camera count")
    if not np.isfinite(k).all() or not np.isfinite(m).all():
        raise ContractError("consensus geometry contains nonfinite values")
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / GEOMETRY_FILES["intrinsics"], k)
    np.save(target / GEOMETRY_FILES["w2c"], m)


def write_consensus_meta(out_dir: str | Path, meta: Mapping[str, Any]) -> dict[str, Any]:
    """Write ``meta.json``, injecting and validating the schema version."""

    payload = dict(meta)
    payload.setdefault("schema_version", CONSENSUS_SCHEMA_VERSION)
    if payload["schema_version"] != CONSENSUS_SCHEMA_VERSION:
        raise SchemaError(
            f"unexpected consensus schema: {payload['schema_version']!r}"
        )
    for key in ("cameras", "frames", "height", "width"):
        if key not in payload:
            raise SchemaError(f"consensus meta is missing required key: {key}")
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    with open(target / META_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    return payload


def write_consensus_directory(
    out_dir: str | Path,
    *,
    d: np.ndarray,
    sigma: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    cameras: Sequence[str],
    frames: Sequence[int],
    meta_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a complete in-memory consensus directory (fixtures and smokes)."""

    d_arr = np.asarray(d, dtype=np.float16)
    sigma_arr = np.asarray(sigma, dtype=np.float16)
    valid_arr = np.asarray(valid).astype(np.uint8)
    if d_arr.ndim != 4 or d_arr.shape != sigma_arr.shape or d_arr.shape != valid_arr.shape:
        raise ContractError("d, sigma and valid must share one (C, T, H, W) shape")
    num_cameras, num_frames, height, width = (int(v) for v in d_arr.shape)
    if len(cameras) != num_cameras:
        raise ContractError("camera list length does not match the map array")
    if len(frames) != num_frames:
        raise ContractError("frame list length does not match the map array")
    maps = open_consensus_maps(
        out_dir,
        num_cameras=num_cameras,
        num_frames=num_frames,
        height=height,
        width=width,
    )
    maps["d"][:] = d_arr
    maps["sigma"][:] = sigma_arr
    maps["valid"][:] = valid_arr
    for handle in maps.values():
        handle.flush()
    write_consensus_geometry(out_dir, intrinsics, w2c)
    meta: dict[str, Any] = {
        "schema_version": CONSENSUS_SCHEMA_VERSION,
        "cameras": [str(c) for c in cameras],
        "frames": [int(f) for f in frames],
        "height": height,
        "width": width,
        "num_cameras": num_cameras,
        "num_frames": num_frames,
    }
    if meta_extra:
        meta.update(dict(meta_extra))
    return write_consensus_meta(out_dir, meta)


def load_consensus_meta(consensus_dir: str | Path) -> dict[str, Any]:
    """Read and schema-check ``meta.json`` from a consensus directory."""

    path = Path(consensus_dir) / META_FILE
    if not path.is_file():
        raise ArtifactError(f"consensus artifact is missing {META_FILE}: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("schema_version") != CONSENSUS_SCHEMA_VERSION:
        raise SchemaError(
            f"unexpected consensus schema: {meta.get('schema_version')!r}"
        )
    for key in ("cameras", "frames", "height", "width"):
        if key not in meta:
            raise SchemaError(f"consensus meta is missing required key: {key}")
    return meta


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class EvidenceRuntime:
    """Per-primitive E1 verdicts against the frozen consensus depth artifact.

    Parameters
    ----------
    consensus_dir:
        Directory written by ``scripts/build_evidence_consensus.py``.
    device:
        Torch device for every returned tensor and (when ``preload``) for the
        map storage.
    mode:
        ``"valid"`` uses the evidence of the requested frame. ``"time_shift"``
        is the preregistered negative control: every camera reads the evidence
        of ``(frame + time_shift) % T`` instead. The shift is circular and
        identical across cameras, so per-camera geometry, per-camera evidence
        statistics, and the temporal marginal distribution are all preserved;
        only the frame-to-evidence correspondence is broken. This replaces the
        per-camera frame *shuffle* control used in census-v2, which was shown
        to be pathological (it destroys temporal coherence and therefore
        cannot certify anything, making the ratio uninformative).
    time_shift:
        Circular shift in frames. Must not be a multiple of ``T`` in
        ``time_shift`` mode, otherwise the control would silently be a no-op.
    tau_rel, kappa:
        Margin rule ``margin = max(tau_rel * d, kappa * sigma)``.
    k_gap:
        Gap multiple separating ``VERDICT_OCCLUDED`` from
        ``VERDICT_BEHIND_WEAK``.
    sigma_abstain:
        Absolute-units (scene metric, same units as depth) consensus sigma
        above which the runtime abstains with ``VERDICT_UNCERTAIN``.
    near_clip:
        Minimum camera-space z for a point to be evaluable.
    preload:
        Copy every map onto ``device`` at construction (training default).
        When false the maps stay memory-mapped on host and per-view slices are
        materialised lazily with a small cache (debug/smoke path).
    """

    def __init__(
        self,
        consensus_dir: str,
        device: str | torch.device = "cuda",
        mode: str = MODE_VALID,
        time_shift: int = 0,
        tau_rel: float = 0.03,
        kappa: float = 2.5,
        k_gap: float = 3.0,
        sigma_abstain: float = 0.20,
        near_clip: float = 0.01,
        preload: bool = True,
    ) -> None:
        root = Path(consensus_dir)
        if not root.is_dir():
            raise ArtifactError(f"consensus directory not found: {root}")
        for filename in ARRAY_FILES.values():
            if not (root / filename).is_file():
                raise ArtifactError(f"consensus artifact is missing {filename}: {root}")

        if mode not in MODES:
            raise ContractError(f"unknown evidence mode: {mode!r} (expected {MODES})")
        tau_rel = float(tau_rel)
        kappa = float(kappa)
        k_gap = float(k_gap)
        sigma_abstain = float(sigma_abstain)
        near_clip = float(near_clip)
        if tau_rel <= 0.0:
            raise ContractError("tau_rel must be positive")
        if kappa < 0.0:
            raise ContractError("kappa must be nonnegative")
        if k_gap < 1.0:
            raise ContractError("k_gap must be >= 1 so the gap class implies behind")
        if sigma_abstain <= 0.0:
            raise ContractError("sigma_abstain must be positive")
        if near_clip <= 0.0:
            raise ContractError("near_clip must be positive")

        self.consensus_dir = str(root)
        self.device = torch.device(device)
        self.mode = mode
        self.tau_rel = tau_rel
        self.kappa = kappa
        self.k_gap = k_gap
        self.sigma_abstain = sigma_abstain
        self.near_clip = near_clip
        self.preload = bool(preload)

        meta = load_consensus_meta(root)
        self.meta = meta
        self.cameras: list[str] = [str(c) for c in meta["cameras"]]
        self.frames: list[int] = [int(f) for f in meta["frames"]]
        self.height = int(meta["height"])
        self.width = int(meta["width"])
        self.num_cameras = len(self.cameras)
        self.num_frames = len(self.frames)
        if self.num_cameras == 0 or self.num_frames == 0:
            raise SchemaError("consensus artifact declares no cameras or no frames")
        if len(set(self.cameras)) != self.num_cameras:
            raise SchemaError("consensus artifact has duplicate camera ids")
        if len(set(self.frames)) != self.num_frames:
            raise SchemaError("consensus artifact has duplicate frame ids")
        self.camera_index: dict[str, int] = {
            camera: i for i, camera in enumerate(self.cameras)
        }
        self._frame_slot: dict[int, int] = {
            frame: i for i, frame in enumerate(self.frames)
        }

        self.time_shift = int(time_shift)
        if self.mode == MODE_TIME_SHIFT and self.time_shift % self.num_frames == 0:
            raise ContractError(
                "time_shift mode with a shift that is a multiple of the frame "
                "count is a silent no-op control"
            )

        self._load_geometry(root)
        self._load_maps(root)

        self._slice_cache: "OrderedDict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._slice_cache_capacity = 64
        self._grid_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        self._verdict_scalars = {
            code: torch.tensor(code, dtype=torch.int8, device=self.device)
            for code in range(NUM_VERDICTS)
        }
        self._counts = torch.zeros(NUM_VERDICTS, dtype=torch.int64, device=self.device)
        self._calls = 0
        self._points = 0
        self._missing_camera_calls = 0
        self._missing_frame_calls = 0

    # -- construction helpers ------------------------------------------------

    def _load_geometry(self, root: Path) -> None:
        intrinsics = np.load(root / GEOMETRY_FILES["intrinsics"])
        w2c = np.load(root / GEOMETRY_FILES["w2c"])
        if intrinsics.shape != (self.num_cameras, 3, 3):
            raise SchemaError(
                f"intrinsics shape {intrinsics.shape} does not match "
                f"{self.num_cameras} cameras"
            )
        if w2c.shape != (self.num_cameras, 4, 4):
            raise SchemaError(
                f"w2c shape {w2c.shape} does not match {self.num_cameras} cameras"
            )
        if not np.isfinite(intrinsics).all() or not np.isfinite(w2c).all():
            raise ContractError("consensus geometry contains nonfinite values")
        bottom = w2c[:, 3, :]
        expected = np.zeros_like(bottom)
        expected[:, 3] = 1.0
        if not np.allclose(bottom, expected, atol=1e-6):
            raise CameraConventionError(
                "w2c bottom row must be [0, 0, 0, 1] (x_cam = R x_world + t)"
            )
        if not (intrinsics[:, 0, 0] > 0).all() or not (intrinsics[:, 1, 1] > 0).all():
            raise CameraConventionError("intrinsics must have positive focal lengths")

        k = torch.as_tensor(intrinsics, dtype=torch.float64).to(self.device)
        m = torch.as_tensor(w2c, dtype=torch.float64).to(self.device)
        self._intrinsics = k
        self._w2c = m
        self._rot = m[:, :3, :3].contiguous()
        self._trans = m[:, :3, 3].contiguous()
        self._fx = k[:, 0, 0].contiguous()
        self._fy = k[:, 1, 1].contiguous()
        self._cx = k[:, 0, 2].contiguous()
        self._cy = k[:, 1, 2].contiguous()

    def _load_maps(self, root: Path) -> None:
        shape = (self.num_cameras, self.num_frames, self.height, self.width)
        self._maps_np: dict[str, np.ndarray] = {}
        self._maps_gpu: dict[str, torch.Tensor] = {}
        pixels = self.height * self.width
        for key, filename in MAP_FILES.items():
            array = np.load(root / filename, mmap_mode="r")
            if tuple(array.shape) != shape:
                raise SchemaError(
                    f"{filename} shape {tuple(array.shape)} does not match meta {shape}"
                )
            if array.dtype != np.dtype(MAP_DTYPES[key]):
                raise SchemaError(
                    f"{filename} dtype {array.dtype} is not {np.dtype(MAP_DTYPES[key])}"
                )
            if self.preload:
                # np.array copies the memory map into a writable host buffer;
                # torch.from_numpy rejects read-only memory maps.
                host = np.array(array, order="C")
                if key == "valid":
                    host = host.view(np.bool_)
                tensor = torch.from_numpy(host).to(self.device)
                self._maps_gpu[key] = tensor.reshape(
                    self.num_cameras, self.num_frames, pixels
                )
                del host
            else:
                self._maps_np[key] = array

    # -- public surface ------------------------------------------------------

    def has_camera(self, camera_id: Any) -> bool:
        """True when the consensus artifact carries evidence for a camera."""

        return isinstance(camera_id, str) and camera_id in self.camera_index

    def evidence_frame(self, frame: int, camera_id: Any = None) -> int:
        """Frame whose evidence is read for a requested ``frame``.

        ``valid`` mode returns ``frame`` itself. ``time_shift`` mode returns
        the circularly shifted frame; the shift is identical for every camera,
        so ``camera_id`` is accepted for interface symmetry only. Returns
        ``-1`` when the requested frame has no slot in the artifact.
        """

        slot = self._frame_slot.get(int(frame), -1)
        if slot < 0:
            return -1
        if self.mode == MODE_VALID:
            return int(self.frames[slot])
        return int(self.frames[(slot + self.time_shift) % self.num_frames])

    def evidence_slot(self, frame: int) -> int:
        """Array time index actually read for ``frame`` (``-1`` when absent)."""

        slot = self._frame_slot.get(int(frame), -1)
        if slot < 0:
            return -1
        if self.mode == MODE_VALID:
            return slot
        return (slot + self.time_shift) % self.num_frames

    @torch.no_grad()
    def verdicts(self, xyz: torch.Tensor, camera_id: Any, frame: int) -> torch.Tensor:
        """Per-primitive int8 verdict codes for one ``(camera, frame)`` view.

        ``xyz`` is ``(N, 3)`` world positions (any device/dtype; detached and
        upcast internally). The result is ``(N,)`` int8 on ``self.device``.
        """

        points = torch.as_tensor(xyz)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ContractError("verdicts expects an (N, 3) world position tensor")
        count = int(points.shape[0])
        self._calls += 1
        self._points += count
        out = torch.full(
            (count,), VERDICT_NOT_EVALUABLE, dtype=torch.int8, device=self.device
        )
        if count == 0:
            return out
        camera = self.camera_index.get(camera_id) if isinstance(camera_id, str) else None
        if camera is None:
            self._missing_camera_calls += 1
            self._counts[VERDICT_NOT_EVALUABLE] += count
            return out
        slot = self.evidence_slot(frame)
        if slot < 0:
            self._missing_frame_calls += 1
            self._counts[VERDICT_NOT_EVALUABLE] += count
            return out

        world = points.detach().to(device=self.device, dtype=torch.float64)
        cam = world @ self._rot[camera].transpose(0, 1) + self._trans[camera]
        z = cam[:, 2]
        safe_z = torch.where(z.abs() > 1e-12, z, torch.full_like(z, 1e-12))
        u = self._fx[camera] * cam[:, 0] / safe_z + self._cx[camera]
        v = self._fy[camera] * cam[:, 1] / safe_z + self._cy[camera]
        px = torch.round(u).to(torch.int64)
        py = torch.round(v).to(torch.int64)
        in_view = (
            (z > self.near_clip)
            & (px >= 0)
            & (px < self.width)
            & (py >= 0)
            & (py < self.height)
        )
        flat = torch.where(in_view, py * self.width + px, torch.zeros_like(px))

        d_map, sigma_map, valid_map = self._view_maps(camera, slot)
        pixel_valid = valid_map[flat]
        if pixel_valid.dtype != torch.bool:
            pixel_valid = pixel_valid > 0
        depth = d_map[flat].to(torch.float32)
        sigma = sigma_map[flat].to(torch.float32)

        finite_depth = torch.isfinite(depth) & (depth > 0)
        evaluable = in_view & pixel_valid & finite_depth
        abstain = (~torch.isfinite(sigma)) | (sigma > self.sigma_abstain)
        sigma = torch.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        margin = torch.maximum(depth * self.tau_rel, sigma * self.kappa)
        gap = margin * self.k_gap
        delta = z.to(torch.float32) - depth

        scalars = self._verdict_scalars
        out = torch.where(evaluable & (delta.abs() <= margin), scalars[VERDICT_NEAR], out)
        out = torch.where(evaluable & (delta < -margin), scalars[VERDICT_IN_FRONT], out)
        out = torch.where(
            evaluable & (delta > margin) & (delta < gap),
            scalars[VERDICT_BEHIND_WEAK],
            out,
        )
        out = torch.where(evaluable & (delta >= gap), scalars[VERDICT_OCCLUDED], out)
        out = torch.where(evaluable & abstain, scalars[VERDICT_UNCERTAIN], out)

        self._counts += torch.bincount(out.to(torch.int64), minlength=NUM_VERDICTS)
        return out

    @torch.no_grad()
    def backproject(
        self, camera_id: Any, frame: int, stride: int, sigma_max: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lift confident consensus pixels of one view into world space.

        Returns ``(world, pixels)`` where ``world`` is ``(P, 3)`` float32 world
        positions and ``pixels`` is ``(P, 2)`` int64 **(x, y) = (column, row)**
        in *consensus map* resolution, matching the ordering of
        ``primitive_census.project_points``. Pixels are taken on a
        ``stride``-spaced grid and kept when the consensus is valid, the depth
        is finite and positive, and ``sigma <= sigma_max``.
        """

        stride = int(stride)
        sigma_max = float(sigma_max)
        if stride < 1:
            raise ContractError("backproject stride must be >= 1")
        if sigma_max <= 0.0:
            raise ContractError("backproject sigma_max must be positive")
        empty = (
            torch.zeros((0, 3), dtype=torch.float32, device=self.device),
            torch.zeros((0, 2), dtype=torch.int64, device=self.device),
        )
        camera = self.camera_index.get(camera_id) if isinstance(camera_id, str) else None
        if camera is None:
            return empty
        slot = self.evidence_slot(frame)
        if slot < 0:
            return empty

        flat, grid_x, grid_y = self._pixel_grid(stride)
        d_map, sigma_map, valid_map = self._view_maps(camera, slot)
        pixel_valid = valid_map[flat]
        if pixel_valid.dtype != torch.bool:
            pixel_valid = pixel_valid > 0
        depth = d_map[flat].to(torch.float32)
        sigma = sigma_map[flat].to(torch.float32)
        keep = (
            pixel_valid
            & torch.isfinite(depth)
            & (depth > 0)
            & torch.isfinite(sigma)
            & (sigma <= sigma_max)
        )
        index = keep.nonzero(as_tuple=False).reshape(-1)
        if int(index.numel()) == 0:
            return empty

        x_pix = grid_x[index]
        y_pix = grid_y[index]
        depth_sel = depth[index].to(torch.float64)
        x_cam = (x_pix - self._cx[camera]) / self._fx[camera] * depth_sel
        y_cam = (y_pix - self._cy[camera]) / self._fy[camera] * depth_sel
        cam_points = torch.stack([x_cam, y_cam, depth_sel], dim=1)
        world = (cam_points - self._trans[camera]) @ self._rot[camera]
        pixels = torch.stack(
            [x_pix.to(torch.int64), y_pix.to(torch.int64)], dim=1
        )
        return world.to(torch.float32), pixels

    def counts_since_reset(self) -> dict[str, int]:
        """Verdict histogram and call bookkeeping since the last reset."""

        counts = [int(v) for v in self._counts.tolist()]
        result = {name: counts[code] for code, name in enumerate(VERDICT_NAMES)}
        result["total"] = int(sum(counts))
        result["calls"] = int(self._calls)
        result["points"] = int(self._points)
        result["missing_camera_calls"] = int(self._missing_camera_calls)
        result["missing_frame_calls"] = int(self._missing_frame_calls)
        return result

    def reset_counts(self) -> None:
        """Zero the verdict histogram and call bookkeeping."""

        self._counts.zero_()
        self._calls = 0
        self._points = 0
        self._missing_camera_calls = 0
        self._missing_frame_calls = 0

    # -- internals -----------------------------------------------------------

    def _view_maps(
        self, camera: int, slot: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flat ``(H*W,)`` d/sigma/valid tensors for one (camera, slot)."""

        if self.preload:
            return (
                self._maps_gpu["d"][camera, slot],
                self._maps_gpu["sigma"][camera, slot],
                self._maps_gpu["valid"][camera, slot],
            )
        key = (camera, slot)
        cached = self._slice_cache.get(key)
        if cached is not None:
            self._slice_cache.move_to_end(key)
            return cached
        tensors = []
        for name in ("d", "sigma", "valid"):
            block = np.array(
                self._maps_np[name][camera, slot], order="C"
            ).reshape(-1)
            if name == "valid":
                block = block.view(np.bool_)
            tensors.append(torch.from_numpy(block).to(self.device))
        value = (tensors[0], tensors[1], tensors[2])
        self._slice_cache[key] = value
        if len(self._slice_cache) > self._slice_cache_capacity:
            self._slice_cache.popitem(last=False)
        return value

    def _pixel_grid(
        self, stride: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cached flat indices and float64 (x, y) coordinates of a stride grid."""

        cached = self._grid_cache.get(stride)
        if cached is not None:
            return cached
        rows = torch.arange(0, self.height, stride, device=self.device, dtype=torch.int64)
        cols = torch.arange(0, self.width, stride, device=self.device, dtype=torch.int64)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
        grid_y = grid_y.reshape(-1)
        grid_x = grid_x.reshape(-1)
        flat = grid_y * self.width + grid_x
        value = (flat, grid_x.to(torch.float64), grid_y.to(torch.float64))
        self._grid_cache[stride] = value
        return value

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"EvidenceRuntime(dir={self.consensus_dir!r}, cameras={self.num_cameras}, "
            f"frames={self.num_frames}, hw=({self.height}, {self.width}), "
            f"mode={self.mode!r}, time_shift={self.time_shift}, device={self.device})"
        )


__all__ = [
    "ARRAY_FILES",
    "CONSENSUS_SCHEMA_VERSION",
    "EvidenceRuntime",
    "GEOMETRY_FILES",
    "MAP_DTYPES",
    "MAP_FILES",
    "META_FILE",
    "MODES",
    "MODE_TIME_SHIFT",
    "MODE_VALID",
    "NUM_VERDICTS",
    "VERDICT_BEHIND_WEAK",
    "VERDICT_IN_FRONT",
    "VERDICT_NAMES",
    "VERDICT_NEAR",
    "VERDICT_NOT_EVALUABLE",
    "VERDICT_OCCLUDED",
    "VERDICT_UNCERTAIN",
    "load_consensus_meta",
    "open_consensus_maps",
    "write_consensus_directory",
    "write_consensus_geometry",
    "write_consensus_meta",
]
