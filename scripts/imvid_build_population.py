#!/usr/bin/env python3
"""Build the NF and FG initial Gaussian populations for an ImViD window.

Consumes the per-frame training-view clouds written by
`scripts/imvid_framewise_init.py` and, for FG, the SEA-RAFT fields written by
`scripts/imvid_flow_searaft.py`.  Emits a `points3d.ply` carrying a per-point
`time` column, which `scene/dataset_readers.py:137-140` already reads into
`BasicPointCloud.time` and `scene/gaussian_model.py:1098` already consumes as
the per-primitive temporal centre.

THE TWO ARMS

  NF  the 300 framewise clouds concatenated, every point keeping the
      timestamp of the frame it was triangulated in.  No flow, no
      classification, no static deduplication.  Static content therefore
      appears once per frame; that duplication is a RECORDED PROPERTY of the
      arm, not a defect to be quietly removed.

  FG  the same candidate geometry, classified per point as static / dynamic
      / ABSTAIN by SEA-RAFT flow, then:
        static   one copy, from the reference frame, with broad temporal
                 support -- the paper's "initialize static geometry once
                 from the reference frame";
        dynamic  kept at its own observation timestamp with COMPACT
                 temporal support;
        abstain  kept at its own timestamp with the trainer's DEFAULT
                 support -- neither collapsed into static nor compacted as
                 dynamic.  Abstention is preserved rather than forced,
                 because forcing an ambiguous point is a decision the
                 evidence does not support and it would be invisible
                 afterwards.

STATIC IS TAKEN FROM THE REFERENCE FRAME, NOT DEDUPLICATED.  Spatial
deduplication needs a distance threshold, a threshold is a tuned parameter,
and a tuned parameter makes the initializer a fitted object rather than a
frozen one -- the reasoning `scripts/imvid_build_initialization.py` already
records for the three-frame union.  Taking the reference frame's static
points is threshold-free.  Its disclosed cost is that static geometry which
becomes visible only later in the window is dropped from the static set;
such points are still present as whatever the classifier calls them in the
frames they were triangulated in.

THE THRESHOLD IS IN MEASURED PIXELS, NOT THE PAPER'S NUMBER.  ImViD reports
`epsilon_f = 0.1` and never states its units; 0.1 px at 2656x1494 and 0.1
normalized by image width differ by a factor of ~2,656, and the paper's own
classifier ("dynamic if the sampled flow exceeds a threshold in ANY view")
is a maximum over 38 views, which is extremely permissive.  Transplanting
the number would be guessing.  The thresholds here are absolute, declared in
full-raster pixels, and stated with the reason for their value.

Degenerate classification is an INVALID RUN, not a result: see
`--require-nondegenerate`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imvid_verify_pinhole as vp  # noqa: E402
from imvid_to_blender import (  # noqa: E402
    ContractError,
    camera_token,
    derive_output_camera,
    parse_cameras_txt,
    parse_images_txt,
)

#: Below this many pixels of inter-frame motion, in FULL-raster pixels, a
#: point is called static.  Half a pixel is the scale at which a dense flow
#: field at this raster stops carrying a trustworthy displacement at all, so
#: it is the natural floor rather than a fitted one.
EPS_STATIC_PX = 0.5

#: At or above this, a point is called dynamic.  1.5 px is three times the
#: static floor: a gap wide enough that a point cannot be pushed across it by
#: flow noise, which is what creates a genuine abstention band instead of a
#: single knife-edge threshold.
EPS_DYNAMIC_PX = 1.5

#: Temporal support is carried in the PLY as a per-point temporal STANDARD
#: DEVIATION in seconds.  The trainer stores `_scaling_t` such that
#: `get_scaling_t = exp(_scaling_t) = sqrt(dist_t)`, and `get_cov_t` consumes
#: that as a VARIANCE (`scene/gaussian_model.py:946-958`) -- so the standard
#: deviation is `dist_t ** 0.25`, and the trainer's own uniform default
#: `dist_t = span / 5` corresponds to a std of `(span / 5) ** 0.25`.  Writing a
#: standard deviation rather than the raw `dist_t` keeps the column readable;
#: the exponent lives in one place, `create_from_pcd`.
#: "Broad" is half the window, so a static primitive's support covers it.
BROAD_SUPPORT_SPAN_FRAC = 0.5
#: "Compact" is eight frames, ~0.133 s -- narrow enough that a dynamic
#: primitive is supported near its own observation and not across the window.
COMPACT_SUPPORT_FRAMES = 8.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frame_clouds(root: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in sorted(root.glob("frame_*.npz")):
        d = np.load(p)
        out[int(d["frame"])] = {
            "xyz": d["xyz"].astype(np.float64),
            "rgb": d["rgb"].astype(np.uint8),
            "time": float(d["time"]),
            "path": p,
        }
    if not out:
        raise ContractError(f"no frame_*.npz clouds under {root}")
    return out


def camera_geometry(model_dir: Path, scale: float, exclude: tuple[str, ...]):
    """World-to-camera rotations, translations and one K per camera name."""
    cameras = parse_cameras_txt((model_dir / "cameras.txt").read_text(encoding="utf-8"))
    images = parse_images_txt((model_dir / "images.txt").read_text(encoding="utf-8"))
    geom = {}
    for name, entry in images.items():
        cam = camera_token(name)
        if cam in exclude:
            continue
        out = derive_output_camera(cameras[entry["camera_id"]], scale)
        geom[cam] = {
            # vp.qvec2rotmat is the same world-to-camera rotation the
            # converter uses to build every pose, so a projection here lands
            # where the trainer's camera looks.
            "R": vp.qvec2rotmat(entry["qvec"]),
            "t": np.asarray(entry["tvec"], dtype=np.float64),
            "K": np.asarray(out["K_new"], dtype=np.float64),
            "w": int(out["width"]),
            "h": int(out["height"]),
        }
    if not geom:
        raise ContractError("no training cameras left after exclusion")
    return geom


def max_flow_per_point(xyz: np.ndarray, geom: dict, flow_root: Path, frame: int,
                       last_pair: int, opened: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Max flow magnitude over training views, and how many views saw a point.

    Flow is stored on a HALF raster with FULL-raster magnitudes, so the
    projected full-raster pixel is halved to index it while the sampled value
    needs no conversion.
    """
    best = np.zeros(xyz.shape[0], dtype=np.float32)
    seen = np.zeros(xyz.shape[0], dtype=np.int32)
    src = min(frame, last_pair)
    for cam, g in geom.items():
        fp = flow_root / f"{cam}_{src:06d}.npz"
        if not fp.is_file():
            raise ContractError(f"flow field {fp} is absent")
        with np.load(fp) as z:
            flow = z["flow"].astype(np.float32)
        opened.append(str(fp))

        pc = xyz @ g["R"].T + g["t"]
        z_ok = pc[:, 2] > 1e-6
        with np.errstate(divide="ignore", invalid="ignore"):
            u = g["K"][0, 0] * pc[:, 0] / pc[:, 2] + g["K"][0, 2]
            v = g["K"][1, 1] * pc[:, 1] / pc[:, 2] + g["K"][1, 2]
        fh, fw = flow.shape[0], flow.shape[1]
        # half raster -> the projected full-raster pixel indexes at u/2, v/2
        iu = np.floor(u * fw / g["w"]).astype(np.int64)
        iv = np.floor(v * fh / g["h"]).astype(np.int64)
        ok = z_ok & (iu >= 0) & (iu < fw) & (iv >= 0) & (iv < fh)
        if not ok.any():
            continue
        mag = np.linalg.norm(flow[iv[ok], iu[ok]].astype(np.float32), axis=-1)
        best[ok] = np.maximum(best[ok], mag)
        seen[ok] += 1
    return best, seen


def write_ply(path: Path, xyz, rgb, times, extents) -> None:
    """Write the cloud with `time` and `t_extent` vertex properties.

    `storePly` in `scene/dataset_readers.py:143-147` fixes its dtype to
    xyz/normals/rgb and cannot carry a time column, so the PLY is written
    here.  Normals are emitted as zeros because `fetchPly` reads them and
    `create_from_pcd` never uses them; omitting them would change nothing but
    would make the file differ from every other cloud in the project.
    """
    from plyfile import PlyData, PlyElement

    n = xyz.shape[0]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1"),
             ("time", "f4"), ("t_extent", "f4")]
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["nx"] = arr["ny"] = arr["nz"] = 0.0
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    arr["time"] = times
    arr["t_extent"] = extents
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("nf", "fg"), help="which population to build")
    ap.add_argument("--clouds-root", help="per-frame npz clouds from imvid_framewise_init")
    ap.add_argument("--model", help="SUPPLIED calibration dir (poses + cameras)")
    ap.add_argument("--flow-root", default=None, help="SEA-RAFT fields (FG only)")
    ap.add_argument("--out-ply", help="destination points3d.ply")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--fps-rational", default="60000/1001")
    ap.add_argument("--reference-frame", type=int, default=0)
    ap.add_argument("--exclude-cameras", default="cam00")
    ap.add_argument("--eps-static-px", type=float, default=EPS_STATIC_PX)
    ap.add_argument("--eps-dynamic-px", type=float, default=EPS_DYNAMIC_PX)
    ap.add_argument("--max-points", type=int, default=0,
                    help="0 = no cap. If set, a fixed-seed WITHOUT-replacement "
                         "subsample is applied and both counts are recorded")
    ap.add_argument("--subsample-seed", type=int, default=0)
    ap.add_argument("--require-nondegenerate", action="store_true", default=True)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()
    for required in ("arm", "clouds_root", "model", "out_ply"):
        if not getattr(args, required):
            raise ContractError(f"--{required.replace('_', '-')} is required")
    if args.arm == "fg" and not args.flow_root:
        raise ContractError("--flow-root is required for the FG arm")
    if args.eps_static_px >= args.eps_dynamic_px:
        raise ContractError("--eps-static-px must be strictly below --eps-dynamic-px; "
                            "equal thresholds destroy the abstention band")

    clouds = load_frame_clouds(Path(args.clouds_root))
    frames = sorted(clouds)
    num, den = (int(x) for x in args.fps_rational.split("/"))
    span = (len(frames) - 1) * den / num
    broad = float(span * BROAD_SUPPORT_SPAN_FRAC)
    compact = float(COMPACT_SUPPORT_FRAMES * den / num)
    # Reproduces the trainer's uniform default EXACTLY when expressed as a
    # standard deviation, so an abstaining point is initialized identically to
    # how it would have been with no t_extent column at all.
    default_extent = float((span / 5.0) ** 0.25)

    exclude = tuple(c.strip() for c in args.exclude_cameras.split(",") if c.strip())
    opened: list[str] = []
    stats = {"static": 0, "dynamic": 0, "abstain": 0}
    xs, cs, ts, es = [], [], [], []

    if args.arm == "nf":
        for f in frames:
            c = clouds[f]
            xs.append(c["xyz"])
            cs.append(c["rgb"])
            ts.append(np.full(c["xyz"].shape[0], c["time"], dtype=np.float64))
            es.append(np.full(c["xyz"].shape[0], default_extent, dtype=np.float64))
            opened.append(str(c["path"]))
        per_frame_static_sum = None
        mag_summary = None
    else:
        geom = camera_geometry(Path(args.model), args.scale, exclude)
        flow_root = Path(args.flow_root)
        last_pair = max(frames) - 1
        mags_all = []
        per_frame_static_sum = 0
        for f in frames:
            c = clouds[f]
            best, seen = max_flow_per_point(c["xyz"], geom, flow_root, f, last_pair, opened)
            mags_all.append(best)
            is_static = best <= args.eps_static_px
            is_dynamic = best >= args.eps_dynamic_px
            is_abstain = ~is_static & ~is_dynamic
            # A point no camera saw has no evidence; calling it static would
            # manufacture evidence from an absence, so it abstains.
            unseen = seen == 0
            is_static = is_static & ~unseen
            is_dynamic = is_dynamic & ~unseen
            is_abstain = is_abstain | unseen

            per_frame_static_sum += int(is_static.sum())
            stats["dynamic"] += int(is_dynamic.sum())
            stats["abstain"] += int(is_abstain.sum())

            if is_dynamic.any():
                xs.append(c["xyz"][is_dynamic]); cs.append(c["rgb"][is_dynamic])
                ts.append(np.full(int(is_dynamic.sum()), c["time"]))
                es.append(np.full(int(is_dynamic.sum()), compact))
            if is_abstain.any():
                xs.append(c["xyz"][is_abstain]); cs.append(c["rgb"][is_abstain])
                ts.append(np.full(int(is_abstain.sum()), c["time"]))
                es.append(np.full(int(is_abstain.sum()), default_extent))
            if f == args.reference_frame and is_static.any():
                xs.append(c["xyz"][is_static]); cs.append(c["rgb"][is_static])
                ts.append(np.full(int(is_static.sum()), c["time"]))
                es.append(np.full(int(is_static.sum()), broad))
                stats["static"] = int(is_static.sum())
            opened.append(str(c["path"]))
        allm = np.concatenate(mags_all)
        mag_summary = {
            "mean": float(allm.mean()), "median": float(np.median(allm)),
            "p90": float(np.percentile(allm, 90)), "p99": float(np.percentile(allm, 99)),
            "max": float(allm.max()),
        }

    xyz = np.concatenate(xs); rgb = np.concatenate(cs)
    times = np.concatenate(ts); extents = np.concatenate(es)
    raw_count = int(xyz.shape[0])

    capped = False
    if args.max_points and raw_count > args.max_points:
        # WITHOUT replacement, unlike the reader's own subsample at
        # scene/dataset_readers.py:498 which uses np.random.randint and so
        # returns duplicates. Fixed seed, recorded.
        idx = np.random.default_rng(args.subsample_seed).choice(
            raw_count, size=args.max_points, replace=False)
        idx.sort()
        xyz, rgb, times, extents = xyz[idx], rgb[idx], times[idx], extents[idx]
        capped = True

    leaked = sorted({c for c in exclude for p in opened if f"/{c}_" in p.replace("\\", "/")})
    if leaked:
        raise ContractError(f"held-out camera(s) {leaked} were READ; this run is void")

    if args.arm == "fg" and args.require_nondegenerate:
        classified = stats["static"] + stats["dynamic"] + stats["abstain"]
        degenerate = [k for k, v in stats.items() if v == classified and classified > 0]
        if classified == 0:
            raise ContractError("no candidate observation was classified; the FG "
                                "mechanism did not engage and this run is INVALID")
        if degenerate:
            raise ContractError(
                f"classification is degenerate -- every point is {degenerate[0]!r}. "
                "The precondition that the mechanism was exercised has FAILED; "
                f"measured flow magnitudes {mag_summary}. This run is INVALID and "
                "the thresholds or the units must be re-derived, not the reading."
            )
        if per_frame_static_sum is not None and stats["static"] >= per_frame_static_sum:
            raise ContractError(
                f"static deduplication did not occur: kept {stats['static']} against "
                f"a per-frame static sum of {per_frame_static_sum}"
            )
        if len(np.unique(times)) < 2:
            raise ContractError("every point carries the same timestamp; dynamic "
                                "geometry did not receive frame-local support")

    out_ply = Path(args.out_ply)
    write_ply(out_ply, xyz.astype(np.float32), rgb, times.astype(np.float32),
              extents.astype(np.float32))

    manifest = {
        "schema": "imvid-init-population-v1",
        "arm": args.arm,
        "clouds_root": args.clouds_root,
        "flow_root": args.flow_root,
        "reference_frame": args.reference_frame,
        "excluded_cameras": list(exclude),
        "frames": len(frames),
        "raw_points": raw_count,
        "written_points": int(xyz.shape[0]),
        "capped": capped,
        "max_points": args.max_points,
        "subsample_seed": args.subsample_seed,
        "distinct_timestamps": int(len(np.unique(times))),
        "distinct_t_extents": sorted(float(x) for x in np.unique(extents)),
        "classification": stats if args.arm == "fg" else None,
        "per_frame_static_sum": per_frame_static_sum,
        "static_duplication_reduction": (
            None if per_frame_static_sum in (None, 0)
            else round(1.0 - stats["static"] / per_frame_static_sum, 6)),
        "flow_magnitude_px": mag_summary,
        "thresholds_px": {"static_at_or_below": args.eps_static_px,
                          "dynamic_at_or_above": args.eps_dynamic_px},
        "support_seconds": {"broad": broad, "compact": compact, "default": default_extent},
        "flow_assets_opened": sum(1 for p in opened if p.endswith(".npz") and "frame_" not in p),
        "out_ply": str(out_ply),
        "out_ply_sha256": sha256_file(out_ply),
        "out_ply_bytes": out_ply.stat().st_size,
    }
    if args.manifest:
        mp = Path(args.manifest)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _check(name: str, ok: bool, detail) -> dict:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return {"name": name, "ok": bool(ok), "detail": detail}


def run_self_test() -> int:
    results = []
    results.append(_check("abstention_band_is_nonempty",
                          EPS_STATIC_PX < EPS_DYNAMIC_PX,
                          f"[{EPS_STATIC_PX}, {EPS_DYNAMIC_PX})"))

    mag = np.array([0.0, 0.4, 0.5, 0.9, 1.5, 4.0])
    static = mag <= EPS_STATIC_PX
    dynamic = mag >= EPS_DYNAMIC_PX
    abstain = ~static & ~dynamic
    results.append(_check("three_way_partition_is_exhaustive_and_disjoint",
                          bool(np.all(static.astype(int) + dynamic.astype(int)
                                      + abstain.astype(int) == 1)),
                          {"static": int(static.sum()), "dynamic": int(dynamic.sum()),
                           "abstain": int(abstain.sum())}))
    results.append(_check("boundary_values_land_where_declared",
                          bool(static[2] and dynamic[4] and abstain[3]),
                          "0.5->static, 1.5->dynamic, 0.9->abstain"))

    num, den, n = 60000, 1001, 300
    span = (n - 1) * den / num
    compact = COMPACT_SUPPORT_FRAMES * den / num
    results.append(_check("compact_support_is_far_below_broad",
                          compact < span * BROAD_SUPPORT_SPAN_FRAC / 10,
                          f"compact={compact:.4f}s broad={span * BROAD_SUPPORT_SPAN_FRAC:.4f}s"))
    default_std = (span / 5.0) ** 0.25
    results.append(_check("default_std_reproduces_trainer_dist_t_exactly",
                          abs(default_std ** 4 - span / 5.0) < 1e-12,
                          f"std={default_std:.10f} -> dist_t={default_std ** 4:.10f} "
                          f"vs span/5={span / 5.0:.10f}"))
    results.append(_check("compact_std_is_much_narrower_than_default",
                          compact < default_std / 5.0,
                          f"compact={compact:.4f}s default={default_std:.4f}s"))

    rng = np.random.default_rng(0)
    idx = rng.choice(1000, size=100, replace=False)
    results.append(_check("subsample_is_without_replacement",
                          len(set(idx.tolist())) == 100, "100 distinct of 100"))

    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ContractError as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        sys.exit(2)
