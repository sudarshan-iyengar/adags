#!/usr/bin/env python
"""S2: SAM2 video propagation over the training cameras of a DERIVED scene.

This is the S2 membership instrument of spec v2.0.0 (`research-wiki/
operations/absfix-wave2-spec-v2-2026-09-11.md` section 11.3, as amended by
section 13.1). It produces, per training camera, a binary mask per frame of
the wine bottle, from TWO user-placed positive clicks and nothing else:

    run A   one positive click at frame 50, propagated forward 50 -> 109
    run B   one positive click at frame 92, propagated forward 92 -> 99

Run B RE-SEEDS the return window with a click the user placed after the
object came back, so on the overlap 92..99 run B is authoritative and
overwrites run A. Run A's overlap masks are not discarded: they are kept
under ``<out>/_seed_a_overlap/camNN/`` so the effect of the re-seed stays
auditable. ``manifest.json`` records, per frame, which run produced the mask
that survives in ``<out>/camNN/``.

WHY THE INSTRUMENT LOOKS LIKE THIS, and what it must never do:

* The construction masks of the absence fixture ARE the DEVA silhouettes, so
  an estimator that reads a DEVA / SA4D mask would be reading the answer. The
  script therefore refuses any path that looks like a DEVA or SA4D mask tree
  (`_refuse_deva_path`), and it opens nothing but ``<scene_root>/images/
  camNN_FFFF.png``, the checkpoint, the config and the clicks file.
* The clicks are the user's and are placed ONCE, from the raw frames, with no
  retries. An unfilled template (any ``null`` click) is a refusal, not a
  default: the script exits 2 before loading SAM2.
* The checkpoint sha256, the config name and the SAM2 code commit are written
  into the manifest, and the spec requires them recorded BEFORE inference.
  They are collected up front and appear in the manifest whatever happens.

THE CAMERA-DROP RULE (section 13.1) is applied at EACH seed frame separately:
a camera whose mask area at that seed frame is below ``--min_area`` px, or
above ``--median_factor`` times the median area over cameras AT THAT SEED
FRAME, is dropped; a camera dropped at either seed frame is dropped from the
vote and listed. The vote needs ``--min_cameras`` (12 by the spec) cameras
left, and the script exits 2 when fewer remain.

Masks are uint8 0/255 PNGs at the scene raster, named ``FFFF.png`` with the
same zero-padded frame numbering the DEVA ``object_mask`` trees use, so the
membership vote reads an S2 tree exactly the way it reads a construction one.

Everything except `run_propagation` is importable without torch or SAM2, so
the rule logic is testable on a workstation (see
``tests/test_s2_sam2_propagate.py``).

RUNNING IT ON LEONARDO, and the one trap measured there. Use the ``envs/sam2``
venv through ``agent-control/s2/s2_env.sh``, which mirrors
``exp_index/leonardo_env.sh``. Its ``LD_PRELOAD`` of gcc 12.2.0's libstdc++ is
load-bearing and NOT redundant with ``LD_LIBRARY_PATH``: the spack python
carries a DT_RPATH into gcc-runtime-8.5.0, DT_RPATH beats ``LD_LIBRARY_PATH``,
and without the preload ``sam2/_C.so`` fails to load, whereupon SAM2 prints a
warning, SKIPS its mask post-processing and carries on - a silent degradation,
observed in job 57383444. A job that runs this script should import
``sam2._C`` first and refuse to continue if that import fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TOOL = "scripts/s2_sam2_propagate.py"
SPEC = "absfix-wave2-spec-v2-2026-09-11 sections 11.3 and 13.1"

#: Path COMPONENTS that mark a path as belonging to a DEVA / SA4D mask tree.
#: The S2 instrument must be independent of the construction masks, and those
#: masks are the DEVA silhouettes, so reading one would make the estimate
#: circular. Matching is per component, not by substring: an ordinary path
#: such as ``.../test_deva_path/...`` must not be refused, while
#: ``.../cam01/pseudo_label/object_mask/`` must.
FORBIDDEN_PATH_COMPONENTS = ("pseudo_label", "object_mask", "sa4d")
#: A component starting with this prefix (``deva``, ``deva_masks``, ...) is a
#: DEVA tree too.
FORBIDDEN_COMPONENT_PREFIX = "deva"

#: Held-out camera of the N3V protocol; never a training camera.
HELD_OUT_CAMERA = "cam00"

CAMERA_RE = re.compile(r"^(cam\d+)_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


class Refusal(RuntimeError):
    """A frozen precondition of the instrument is not satisfied.

    Raised instead of proceeding with a default. `main` turns it into exit
    code 2, which is the spec's "this cell does not run" signal.
    """


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
def _refuse_deva_path(path: os.PathLike | str, what: str) -> Path:
    """Refuse a path that lies in a DEVA / SA4D mask tree."""
    p = Path(path)
    for raw in p.parts:
        part = raw.lower()
        if part in FORBIDDEN_PATH_COMPONENTS or part.startswith(
            FORBIDDEN_COMPONENT_PREFIX
        ):
            raise Refusal(
                f"{what} {p} looks like a DEVA/SA4D mask tree (component "
                f"{raw!r}); S2 must be independent of the construction masks"
            )
    return p


def sha256_file(path: os.PathLike | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# clicks
# --------------------------------------------------------------------------
def load_clicks(clicks_path: os.PathLike | str, scene: str) -> Dict[str, Dict[str, List[int]]]:
    """Load and validate the click table for one scene.

    The file is the template at ``research-wiki/assets/
    absfix-s2-clicks-template.json``::

        {"<scene>": {"camNN": {"f50": [x, y], "f92": [x, y]}}}

    Keys beginning with ``_`` are documentation and are ignored. Any missing
    or ``null`` click is a refusal: an unfilled template must never silently
    become a default click.
    """
    p = _refuse_deva_path(clicks_path, "clicks file")
    with open(p, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise Refusal(f"clicks file {p} is not a JSON object")
    if scene not in raw:
        raise Refusal(
            f"clicks file {p} has no entry for scene {scene!r}; present: "
            + ", ".join(sorted(k for k in raw if not k.startswith("_")))
        )
    table = raw[scene]
    if not isinstance(table, dict) or not table:
        raise Refusal(f"clicks for scene {scene!r} is not a non-empty object")
    return validate_clicks(table, scene)


def validate_clicks(table: dict, scene: str) -> Dict[str, Dict[str, List[int]]]:
    """Validate a per-camera click table; raise `Refusal` on any defect."""
    out: Dict[str, Dict[str, List[int]]] = {}
    for cam in sorted(k for k in table if not str(k).startswith("_")):
        entry = table[cam]
        if not isinstance(entry, dict):
            raise Refusal(f"{scene}/{cam}: click entry is not an object")
        cam_out: Dict[str, List[int]] = {}
        for key in ("f50", "f92"):
            if key not in entry:
                raise Refusal(f"{scene}/{cam}: missing click {key!r}")
            click = entry[key]
            if click is None:
                raise Refusal(
                    f"{scene}/{cam}: click {key!r} is null - the click "
                    "template is unfilled; the clicks are placed by the user "
                    "from the raw frames before any inference (spec 13.1)"
                )
            if (
                not isinstance(click, (list, tuple))
                or len(click) != 2
                or isinstance(click, str)
            ):
                raise Refusal(
                    f"{scene}/{cam}: click {key!r} must be [x, y], got {click!r}"
                )
            xy = []
            for v in click:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise Refusal(
                        f"{scene}/{cam}: click {key!r} coordinate {v!r} is not a number"
                    )
                if v < 0:
                    raise Refusal(
                        f"{scene}/{cam}: click {key!r} coordinate {v!r} is negative"
                    )
                xy.append(int(round(float(v))))
            cam_out[key] = xy
        out[cam] = cam_out
    if not out:
        raise Refusal(f"clicks for scene {scene!r} name no camera")
    return out


# --------------------------------------------------------------------------
# scene layout
# --------------------------------------------------------------------------
def discover_scene(scene_root: os.PathLike | str) -> Tuple[Path, Dict[str, Dict[int, Path]], int]:
    """Index ``<scene_root>/images/camNN_FFFF.png``.

    Returns the images directory, ``{camera: {frame: path}}`` and the
    zero-pad width used by the scene's own file names (4 on the derived
    absence-fixture scenes, which is also the DEVA ``object_mask`` width).
    """
    root = _refuse_deva_path(scene_root, "scene root")
    images = root / "images"
    if not images.is_dir():
        raise Refusal(f"scene root {root} has no images/ directory")
    index: Dict[str, Dict[int, Path]] = {}
    widths: set[int] = set()
    for name in os.listdir(images):
        m = CAMERA_RE.match(name)
        if not m:
            continue
        cam, digits = m.group(1), m.group(2)
        widths.add(len(digits))
        index.setdefault(cam, {})[int(digits)] = images / name
    if not index:
        raise Refusal(f"no camNN_FFFF.png frames under {images}")
    if len(widths) != 1:
        raise Refusal(f"{images} mixes frame-number widths {sorted(widths)}")
    return images, index, widths.pop()


def training_cameras(
    index: Dict[str, Dict[int, Path]],
    requested: Optional[Sequence[str]] = None,
) -> List[str]:
    """Training cameras of the scene: every camera present except cam00.

    The camera list comes from the scene, not from a hardcoded cam01..cam20:
    the N3V rig has a camera missing on these scenes, and a hardcoded list
    would silently ask for frames that do not exist.
    """
    present = sorted(c for c in index if c != HELD_OUT_CAMERA)
    if requested is None:
        return present
    missing = [c for c in requested if c not in present]
    if missing:
        raise Refusal(
            f"requested camera(s) {missing} are not training cameras of this "
            f"scene; present: {present}"
        )
    return list(requested)


def frame_name(frame: int, width: int) -> str:
    return f"{frame:0{width}d}.png"


def frame_range(bounds: Sequence[int]) -> List[int]:
    start, end = int(bounds[0]), int(bounds[1])
    if end < start:
        raise Refusal(f"frame range {bounds} runs backwards")
    return list(range(start, end + 1))


# --------------------------------------------------------------------------
# the camera-drop rule (spec 11.3 as amended by 13.1)
# --------------------------------------------------------------------------
def apply_camera_drop_rule(
    areas: Dict[str, Dict[str, int]],
    min_area: int = 500,
    median_factor: float = 5.0,
) -> Tuple[List[str], List[str], Dict[str, List[str]], Dict[str, float]]:
    """Apply the drop rule at EACH seed frame separately.

    ``areas`` is ``{camera: {"f50": area, "f92": area}}``. A camera is dropped
    when, at either seed frame, its mask area is ``< min_area`` px or
    ``> median_factor`` times the median area over cameras at that same seed
    frame. Returns (kept, dropped, reasons, medians).
    """
    if not areas:
        raise Refusal("the drop rule was given no camera areas")
    seeds = ("f50", "f92")
    medians: Dict[str, float] = {}
    for seed in seeds:
        values = []
        for cam, per_seed in areas.items():
            if seed not in per_seed:
                raise Refusal(f"camera {cam} has no area at seed {seed}")
            values.append(int(per_seed[seed]))
        medians[seed] = float(statistics.median(values))

    reasons: Dict[str, List[str]] = {}
    for cam in sorted(areas):
        cam_reasons: List[str] = []
        for seed in seeds:
            area = int(areas[cam][seed])
            if area < min_area:
                cam_reasons.append(
                    f"{seed}: area {area} < {min_area} px"
                )
            limit = median_factor * medians[seed]
            if area > limit:
                cam_reasons.append(
                    f"{seed}: area {area} > {median_factor}x median "
                    f"{medians[seed]:.1f} = {limit:.1f} px"
                )
        if cam_reasons:
            reasons[cam] = cam_reasons

    dropped = sorted(reasons)
    kept = [c for c in sorted(areas) if c not in reasons]
    return kept, dropped, reasons, medians


def check_enough_cameras(kept: Sequence[str], min_cameras: int) -> None:
    if len(kept) < min_cameras:
        raise Refusal(
            f"{len(kept)} camera(s) remain after the drop rule, the vote "
            f"requires >= {min_cameras} (spec 13.1)"
        )


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
#: Every key `build_manifest` guarantees. The test asserts on this list, so a
#: field removed from the manifest breaks a test rather than a downstream read.
MANIFEST_FIELDS = (
    "tool",
    "spec",
    "created_utc",
    "scene",
    "scene_root",
    "raster",
    "frame_digits",
    "frames_a",
    "frames_b",
    "checkpoint",
    "checkpoint_sha256",
    "config",
    "sam2_commit",
    "clicks_file",
    "clicks_file_sha256",
    "clicks",
    "cameras_requested",
    "per_camera",
    "seed_area_medians",
    "min_area",
    "median_factor",
    "dropped_cameras",
    "drop_reasons",
    "cameras_kept",
    "n_cameras_kept",
    "min_cameras",
    "frame_source",
    "mask_sha256",
    "environment",
)


def build_manifest(
    *,
    scene: str,
    scene_root: str,
    raster: Optional[Sequence[int]],
    frame_digits: int,
    frames_a: Sequence[int],
    frames_b: Sequence[int],
    checkpoint: str,
    checkpoint_sha256: str,
    config: str,
    sam2_commit: str,
    clicks_file: str,
    clicks_file_sha256: str,
    clicks: dict,
    cameras_requested: Sequence[str],
    per_camera: dict,
    seed_area_medians: dict,
    min_area: int,
    median_factor: float,
    dropped_cameras: Sequence[str],
    drop_reasons: dict,
    cameras_kept: Sequence[str],
    min_cameras: int,
    frame_source: dict,
    mask_sha256: dict,
    environment: Optional[dict] = None,
) -> dict:
    """Assemble the manifest. ``clicks`` is stored verbatim, as recorded."""
    return {
        "tool": TOOL,
        "spec": SPEC,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scene": scene,
        "scene_root": str(scene_root),
        "raster": list(raster) if raster is not None else None,
        "frame_digits": int(frame_digits),
        "frames_a": [int(frames_a[0]), int(frames_a[-1])] if len(frames_a) else [],
        "frames_b": [int(frames_b[0]), int(frames_b[-1])] if len(frames_b) else [],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "config": config,
        "sam2_commit": sam2_commit,
        "clicks_file": str(clicks_file),
        "clicks_file_sha256": clicks_file_sha256,
        "clicks": clicks,
        "cameras_requested": list(cameras_requested),
        "per_camera": per_camera,
        "seed_area_medians": seed_area_medians,
        "min_area": int(min_area),
        "median_factor": float(median_factor),
        "dropped_cameras": list(dropped_cameras),
        "drop_reasons": drop_reasons,
        "cameras_kept": list(cameras_kept),
        "n_cameras_kept": len(cameras_kept),
        "min_cameras": int(min_cameras),
        "frame_source": frame_source,
        "mask_sha256": mask_sha256,
        "environment": environment or {},
    }


# --------------------------------------------------------------------------
# SAM2 (imported lazily: everything above is testable without torch)
# --------------------------------------------------------------------------
def _resolve_sam2_commit(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    try:
        import subprocess

        import sam2  # noqa: F401

        repo = Path(sam2.__file__).resolve().parent.parent
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        head = out.stdout.strip()
        if out.returncode == 0 and head:
            dirty = subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            return head + ("-dirty" if dirty else "")
    except Exception:  # pragma: no cover - diagnostic only
        pass
    raise Refusal(
        "could not determine the SAM2 code commit; pass --sam2_commit "
        "explicitly (the spec requires it recorded before inference)"
    )


def _stage_frames(paths: Sequence[Path], stage_dir: Path) -> None:
    """Stage frames as ``<i>.jpg`` for SAM2's video loader.

    SAM2's ``load_video_frames`` accepts a directory whose entries are named
    ``<int>.jpg`` and decodes them with PIL, which sniffs the file header
    rather than trusting the extension. Symlinking the scene's PNGs under
    integer ``.jpg`` names therefore feeds SAM2 the ORIGINAL pixels, with no
    JPEG re-encode between the fixture and the estimator.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(paths):
        dst = stage_dir / f"{i}.jpg"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def run_propagation(args) -> int:
    """Run both propagations for every requested camera and write the tree."""
    import numpy as np  # noqa: WPS433 - lazy on purpose
    import torch
    from PIL import Image

    from sam2.build_sam import build_sam2_video_predictor

    checkpoint = _refuse_deva_path(args.checkpoint, "checkpoint")
    if not checkpoint.is_file():
        raise Refusal(f"checkpoint {checkpoint} does not exist")
    out_dir = _refuse_deva_path(args.out, "output directory")

    # Recorded BEFORE any inference (spec 11.3).
    ckpt_sha = sha256_file(checkpoint)
    clicks_sha = sha256_file(args.clicks)
    sam2_commit = _resolve_sam2_commit(args.sam2_commit)

    clicks = load_clicks(args.clicks, args.scene)
    images_dir, index, digits = discover_scene(args.scene_root)
    if args.frame_digits:
        digits = int(args.frame_digits)
    cameras = training_cameras(index, args.cameras)
    missing_clicks = [c for c in cameras if c not in clicks]
    if missing_clicks:
        raise Refusal(
            f"no click recorded for camera(s) {missing_clicks} of scene "
            f"{args.scene}"
        )

    frames_a = frame_range(args.frames_a)
    frames_b = frame_range(args.frames_b)
    seed_a, seed_b = frames_a[0], frames_b[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = build_sam2_video_predictor(args.config, str(checkpoint), device=device)

    out_dir.mkdir(parents=True, exist_ok=True)
    overlap_dir = out_dir / "_seed_a_overlap"
    stage_root = Path(args.stage_dir) if args.stage_dir else Path(
        tempfile.mkdtemp(prefix="s2_stage_")
    )

    areas: Dict[str, Dict[str, int]] = {}
    per_camera: Dict[str, dict] = {}
    frame_source: Dict[str, Dict[str, str]] = {}
    raster: Optional[List[int]] = None

    for cam in cameras:
        cam_out = out_dir / cam
        cam_out.mkdir(parents=True, exist_ok=True)
        cam_src: Dict[str, str] = {}
        cam_areas: Dict[str, int] = {}
        t0 = time.time()

        overlap = set(frames_a) & set(frames_b)
        for label, frames, seed, click_key in (
            ("a", frames_a, seed_a, "f50"),
            ("b", frames_b, seed_b, "f92"),
        ):
            missing = [f for f in frames if f not in index[cam]]
            if missing:
                raise Refusal(
                    f"{cam}: scene has no frame(s) {missing[:5]} "
                    f"(needed for run {label})"
                )
            paths = [index[cam][f] for f in frames]
            stage = stage_root / cam / label
            _stage_frames(paths, stage)

            state = predictor.init_state(video_path=str(stage))
            predictor.reset_state(state)
            x, y = clicks[cam][click_key]
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=np.array([[x, y]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )
            for rel_idx, _obj_ids, logits in predictor.propagate_in_video(state):
                frame = frames[rel_idx]
                mask = (logits[0] > 0.0).squeeze().detach().cpu().numpy()
                arr = (mask.astype(np.uint8)) * 255
                if raster is None:
                    raster = [int(arr.shape[1]), int(arr.shape[0])]
                name = frame_name(frame, digits)
                if label == "a" and frame in overlap:
                    # Run B re-seeds this frame and overwrites it below; keep
                    # run A's version so the re-seed stays auditable.
                    keep = overlap_dir / cam
                    keep.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(arr, mode="L").save(keep / name)
                else:
                    Image.fromarray(arr, mode="L").save(cam_out / name)
                    cam_src[name] = label
                if frame == seed:
                    cam_areas[click_key] = int(mask.sum())
            del state

        if "f50" not in cam_areas or "f92" not in cam_areas:
            raise Refusal(f"{cam}: SAM2 returned no mask at a seed frame")
        areas[cam] = cam_areas
        per_camera[cam] = {
            "click_f50": clicks[cam]["f50"],
            "click_f92": clicks[cam]["f92"],
            "area_f50": cam_areas["f50"],
            "area_f92": cam_areas["f92"],
            "n_masks": len(cam_src),
            "seconds": round(time.time() - t0, 2),
        }
        frame_source[cam] = cam_src
        print(
            f"[s2] {cam} area(f{seed_a})={cam_areas['f50']} "
            f"area(f{seed_b})={cam_areas['f92']} masks={len(cam_src)} "
            f"{per_camera[cam]['seconds']}s",
            flush=True,
        )

    kept, dropped, reasons, medians = apply_camera_drop_rule(
        areas, min_area=args.min_area, median_factor=args.median_factor
    )
    for cam in per_camera:
        per_camera[cam]["dropped"] = cam in dropped
        per_camera[cam]["drop_reasons"] = reasons.get(cam, [])

    mask_sha: Dict[str, str] = {}
    for path in sorted(out_dir.rglob("*.png")):
        mask_sha[path.relative_to(out_dir).as_posix()] = sha256_file(path)

    manifest = build_manifest(
        scene=args.scene,
        scene_root=str(Path(args.scene_root).resolve()),
        raster=raster,
        frame_digits=digits,
        frames_a=frames_a,
        frames_b=frames_b,
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=ckpt_sha,
        config=args.config,
        sam2_commit=sam2_commit,
        clicks_file=str(Path(args.clicks).resolve()),
        clicks_file_sha256=clicks_sha,
        clicks={args.scene: clicks},
        cameras_requested=cameras,
        per_camera=per_camera,
        seed_area_medians=medians,
        min_area=args.min_area,
        median_factor=args.median_factor,
        dropped_cameras=dropped,
        drop_reasons=reasons,
        cameras_kept=kept,
        min_cameras=args.min_cameras,
        frame_source=frame_source,
        mask_sha256=mask_sha,
        environment={
            "torch": torch.__version__,
            "device": str(device),
            "python": sys.version.split()[0],
            "argv": sys.argv[1:],
        },
    )
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=False)
        fh.write("\n")
    print(f"[s2] wrote {manifest_path}", flush=True)
    print(
        f"[s2] kept {len(kept)} camera(s), dropped {dropped or 'none'}",
        flush=True,
    )

    if args.stage_dir is None:
        shutil.rmtree(stage_root, ignore_errors=True)

    check_enough_cameras(kept, args.min_cameras)
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scene_root", required=True, help="derived scene root (holds images/)")
    p.add_argument("--scene", required=True, help="scene key inside the clicks JSON")
    p.add_argument("--clicks", required=True, help="clicks JSON (spec 13.1 format)")
    p.add_argument("--out", required=True, help="output mask tree")
    p.add_argument("--checkpoint", required=True, help="SAM 2.1 checkpoint .pt")
    p.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 hydra config name",
    )
    p.add_argument(
        "--frames-a",
        dest="frames_a",
        nargs=2,
        type=int,
        default=[50, 109],
        metavar=("START", "END"),
        help="run A: forward propagation seeded by the frame-50 click",
    )
    p.add_argument(
        "--frames-b",
        dest="frames_b",
        nargs=2,
        type=int,
        default=[92, 99],
        metavar=("START", "END"),
        help="run B: forward propagation seeded by the frame-92 click",
    )
    p.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="restrict to these training cameras (default: every camera of "
        "the scene except cam00)",
    )
    p.add_argument("--min_area", type=int, default=500, help="drop rule floor, px")
    p.add_argument(
        "--median_factor", type=float, default=5.0, help="drop rule ceiling, x median"
    )
    p.add_argument(
        "--min_cameras",
        type=int,
        default=12,
        help="cameras the vote requires after the drop rule; exit 2 below it",
    )
    p.add_argument(
        "--frame_digits",
        type=int,
        default=0,
        help="zero-pad width of the output mask names (default: the scene's own)",
    )
    p.add_argument(
        "--sam2_commit",
        default=None,
        help="SAM2 code commit (default: read from the installed package's git repo)",
    )
    p.add_argument(
        "--stage_dir",
        default=None,
        help="where to stage frames for SAM2's loader (default: a temp dir, removed)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_propagation(args)
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
