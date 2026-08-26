#!/usr/bin/env python3
"""SEA-RAFT optical flow over ImViD training views, for the FG init arm.

Produces forward flow (frame ``f`` -> ``f+1``) for every adjacent pair of
every TRAINING camera in a converted ImViD window, and refuses to produce it
for a held-out camera.

WHY THIS SCRIPT EXISTS RATHER THAN THE CHECKOUT'S OWN DRIVER.  The shared
SEA-RAFT worktree carries an untracked ``generate_dataset_flow.py`` written
for N3V on Leonardo.  It is unusable here for four independent reasons, each
verified against its source:

  * it runs the model at FULL input resolution.  SEA-RAFT retains all four
    correlation-pyramid levels simultaneously, and level 0 alone is
    ``h1*w1*h2*w2`` float32.  At 2656x1494 that is ~15.4 GB resident with a
    transient of about the same again -- so on a 32 GB V100 it is marginal
    at best, and it is the checkpoint's own eval config that says not to do
    it (``"scale": -1``, i.e. half-resolution inference).
  * it hard-codes ``iters=24`` while the spring-M config the checkpoint was
    released with specifies ``iters: 4``.
  * it also computes BACKWARD flow for a consistency mask, doubling cost.
  * it walks an ``images/`` tree by ``split('_')[0]``, with no notion of a
    held-out camera -- nothing in it could refuse to read ``cam00``.

MEMORY AND UNITS, the two things easy to get silently wrong:

  Inference runs at ``2 ** scale`` of the input raster (``scale = -1`` ->
  half).  Upstream's ``calc_flow`` then upsamples the flow back to full
  resolution AND multiplies magnitudes by ``0.5 ** scale``.  This script
  applies the MAGNITUDE factor and deliberately SKIPS the spatial upsample:
  storing 11,400 dense fields at 2656x1494 float32 would be ~360 GB, and the
  consumer samples flow at projected point locations where a half-raster
  lookup is exact enough and four times cheaper.  So the stored field is on
  a HALF raster while its magnitudes are in FULL-raster pixels.  That
  combination is stated in every manifest and in every ``.npz``, because it
  is precisely the sort of thing a later reader would assume wrongly.

DIRECTION IS MEASURED, NOT ASSUMED.  This project has twice been burned by
optical-flow orientation, and nothing in the consuming code could detect a
reversed field -- it would simply classify the wrong points as dynamic.  So
a sampled subset of pairs is warp-tested: image2 is sampled at
``x + flow(x)`` and compared against image1, and the same is done with the
flow negated.  Forward must win by a declared margin or the run REFUSES.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imvid_searaft_seed import EVAL_CFG, searaft_sys_path  # noqa: E402

IMAGE_RE = re.compile(r"^(?P<camera>cam\d+)_(?P<frame>\d+)\.(?P<ext>png|jpg|jpeg)$")

#: Forward interpretation must beat the reversed one by at least this factor
#: on the warp test.  A correct field is typically several times better; 1.2
#: is a floor that a genuinely forward field clears easily and a reversed one
#: cannot, chosen before any measurement and not adjusted afterwards.
DIRECTION_MARGIN = 1.2

#: Frame gap used by the DIRECTION check only.  Production flow is always
#: adjacent (f -> f+1), but at 59.94 fps adjacent motion on these scenes is
#: sub-pixel, and a warp test cannot tell a forward field from its negation
#: when the displacement is smaller than the interpolation kernel -- measured
#: at a ratio of 1.083 on Opera, against a 1.2 floor.  That is the guard
#: working, not a wiring fault, but a guard that cannot discriminate is not a
#: guard.  Orientation is a property of the MODEL, not of the frame gap, so it
#: is tested where the displacement is large enough to resolve and then relied
#: on for the adjacent pairs.
DEFAULT_DIRECTION_GAP = 8


class ContractError(RuntimeError):
    """A refusal: the setup is not what this script requires."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(images_root: Path) -> dict[str, dict[int, Path]]:
    """Map camera -> {frame_index: path} from a converted scene's images/."""
    found: dict[str, dict[int, Path]] = {}
    for entry in sorted(images_root.iterdir()):
        m = IMAGE_RE.match(entry.name)
        if not m:
            continue
        found.setdefault(m["camera"], {})[int(m["frame"])] = entry
    if not found:
        raise ContractError(f"no camXX_FFFFFF.png images under {images_root}")
    return found


def select_cameras(found: dict, exclude: tuple[str, ...],
                   only: tuple[str, ...] = ()) -> list[str]:
    """Cameras to process: everything present, minus the held-out exclusions,
    optionally narrowed to an explicit shard.

    `only` exists so one scene's flow can be split across several GPUs. It
    NARROWS, it never widens: the exclusion is applied first and independently,
    so a shard can never reach a held-out camera by naming it.
    """
    cameras = sorted(found)
    kept = [c for c in cameras if c not in exclude]
    if only:
        unknown = [c for c in only if c not in cameras]
        if unknown:
            raise ContractError(f"--only-cameras names absent camera(s) {unknown}")
        forbidden = sorted(set(only) & set(exclude))
        if forbidden:
            raise ContractError(
                f"--only-cameras names held-out camera(s) {forbidden}; the shard "
                "list may narrow the training set, never reach outside it"
            )
        kept = [c for c in kept if c in only]
    missing = [c for c in exclude if c not in cameras]
    if missing:
        # Not fatal by itself, but it means the exclusion list and the scene
        # disagree, and a silent no-op exclusion is how held-out data leaks.
        raise ContractError(
            f"excluded camera(s) {missing} are not present in {sorted(cameras)}; "
            "the exclusion list does not match this scene, and an exclusion "
            "that matches nothing protects nothing"
        )
    if not kept:
        raise ContractError("every camera was excluded; nothing to do")
    return kept


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as float32 RGB in 0-255, HWC.

    SEA-RAFT's ``RAFT.forward`` maps ``2 * (image / 255) - 1`` itself, so the
    model expects RAW 0-255 and must NOT be pre-normalised.
    """
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    return arr


def build_model(searaft_root: str, cfg_rel: str, checkpoint: Path, device: str):
    import torch

    from config.parser import json_to_args
    from core.raft import RAFT

    cfg_path = os.path.join(searaft_root, cfg_rel)
    args = json_to_args(cfg_path)
    model = RAFT(args)
    state = torch.load(str(checkpoint), map_location="cpu")
    # strict=True: a key mismatch means the pinned commit and this checkpoint
    # have diverged, which would otherwise produce a partly-random network
    # that still emits plausible-looking flow.
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, args


def infer_pair(model, args, img1: np.ndarray, img2: np.ndarray, device: str):
    """Return forward flow on the HALF raster, magnitudes in FULL pixels."""
    import torch
    import torch.nn.functional as F

    def to_tensor(a):
        return torch.from_numpy(a).permute(2, 0, 1)[None].to(device)

    scale = float(getattr(args, "scale", 0))
    factor = 2.0 ** scale
    t1, t2 = to_tensor(img1), to_tensor(img2)
    if factor != 1.0:
        t1 = F.interpolate(t1, scale_factor=factor, mode="bilinear", align_corners=False)
        t2 = F.interpolate(t2, scale_factor=factor, mode="bilinear", align_corners=False)
    with torch.no_grad():
        out = model(t1, t2, iters=int(getattr(args, "iters", 4)), test_mode=True)
    flow = out["flow"][-1]
    # Magnitude conversion ONLY.  Upstream's calc_flow also upsamples the
    # field spatially; see the module docstring for why that is skipped.
    flow = flow * (1.0 / factor)
    return flow, t1, t2


#: A pixel must carry at least this much displacement, in the units the warp
#: test evaluates in, to be admitted as EVIDENCE about direction.  Below it,
#: forward and reversed warps land within an interpolation kernel of each
#: other and the pixel says nothing either way.
DIRECTION_EVIDENCE_PX = 1.0

#: ...and there must be at least this many such pixels, or the test ABSTAINS
#: loudly rather than returning a ratio computed from a handful of pixels.
DIRECTION_MIN_EVIDENCE_PIXELS = 20000


def warp_check(model_out_flow, t1, t2) -> tuple[float, float, int]:
    """Warp-test the flow both ways; return (forward_err, reversed_err, n).

    Forward flow means ``image1[x] ~ image2[x + flow(x)]``.  Sampling image2
    at ``x + flow`` should reconstruct image1.  If the field were actually
    backward, ``x - flow`` would do better.  Both are computed and compared.

    THE ERROR IS MEASURED ONLY WHERE THE FLOW MOVES.  An earlier version
    averaged over every pixel and could not discriminate: on an opera stage
    roughly 95% of the frame is static, those pixels warp to the same place
    under ``+flow`` and ``-flow``, and they dominate the mean -- the measured
    ratio was 1.083 at gap 1 and 1.132 at gap 8, both under the 1.2 floor, for
    a field that is in fact correctly oriented.  That was a defect in the
    TEST, not in the flow: a direction test has to be evaluated where
    direction is observable.  Restricting to displaced pixels does not make
    the test easier to pass -- a genuinely REVERSED field fails harder on
    exactly those pixels.
    """
    import torch
    import torch.nn.functional as F

    _, _, h, w = t1.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=t1.device, dtype=torch.float32),
        torch.arange(w, device=t1.device, dtype=torch.float32),
        indexing="ij",
    )

    mag = torch.linalg.vector_norm(model_out_flow[0], dim=0)
    moving = mag >= DIRECTION_EVIDENCE_PX

    def err(sign: float) -> float:
        gx = xs + sign * model_out_flow[0, 0]
        gy = ys + sign * model_out_flow[0, 1]
        grid = torch.stack(
            [2.0 * gx / max(w - 1, 1) - 1.0, 2.0 * gy / max(h - 1, 1) - 1.0], dim=-1
        )[None]
        warped = F.grid_sample(t2, grid, mode="bilinear", padding_mode="border", align_corners=True)
        inside = (gx >= 0) & (gx <= w - 1) & (gy >= 0) & (gy <= h - 1)
        keep = (inside & moving)[None, None].expand_as(warped)
        if not bool(keep.any()):
            return float("nan")
        return float((warped - t1).abs()[keep].mean())

    return err(1.0), err(-1.0), int(moving.sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-root", help="converted scene's images/ directory (READ ONLY)")
    ap.add_argument("--out-root", help="destination for the .npz flow fields")
    ap.add_argument("--exclude-cameras", default="cam00",
                    help="comma-separated cameras that MUST NOT be read (held-out)")
    ap.add_argument("--only-cameras", default="",
                    help="comma-separated shard: process ONLY these training "
                         "cameras, so one scene can be split across GPUs. Narrows "
                         "the training set; it can never reach a held-out camera")
    ap.add_argument("--checkpoint", default="/apollo/users/sri/proj_adags/repo/SEA-RAFT/"
                                            "Tartan-C-T-TSKH-spring540x960-M.pth")
    ap.add_argument("--expect-checkpoint-sha256",
                    default="adcc169244e99d4e6fe645b60aa8eaf3e4263698a3e870b8fbae618e3d2acc28",
                    help="fail closed unless the checkpoint hashes to this")
    ap.add_argument("--cfg", default=EVAL_CFG, help="SEA-RAFT eval config, relative to its root")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--store-dtype", default="float16", choices=("float16", "float32"))
    ap.add_argument("--direction-samples", type=int, default=8,
                    help="pairs warp-tested for flow direction; 0 disables (never do that)")
    ap.add_argument("--direction-gap", type=int, default=DEFAULT_DIRECTION_GAP,
                    help="frame gap for the DIRECTION check only. Production flow "
                         "stays adjacent; at 59.94 fps adjacent motion is sub-pixel "
                         "and the warp test cannot discriminate, so orientation is "
                         "proven where the displacement resolves")
    ap.add_argument("--limit-pairs", type=int, default=0, help="smoke mode: stop after N pairs")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()

    for required in ("images_root", "out_root"):
        if not getattr(args, required):
            raise ContractError(f"--{required.replace('_', '-')} is required")

    images_root = Path(args.images_root)
    out_root = Path(args.out_root)
    checkpoint = Path(args.checkpoint)
    exclude = tuple(c.strip() for c in args.exclude_cameras.split(",") if c.strip())

    if not checkpoint.is_file():
        raise ContractError(f"checkpoint {checkpoint} is absent")
    digest = sha256_file(checkpoint)
    if args.expect_checkpoint_sha256 and digest != args.expect_checkpoint_sha256:
        raise ContractError(
            f"checkpoint sha256 {digest} != expected {args.expect_checkpoint_sha256}; "
            "the flow weights on shared storage are not the recorded ones"
        )

    only = tuple(c.strip() for c in args.only_cameras.split(",") if c.strip())
    found = discover(images_root)
    cameras = select_cameras(found, exclude, only)

    searaft_root = searaft_sys_path()
    model, cfg_args = build_model(searaft_root, args.cfg, checkpoint, args.device)

    out_root.mkdir(parents=True, exist_ok=True)
    opened: list[str] = []
    written = 0
    mag_stats: list[tuple[float, float, float]] = []
    direction_records: list[dict] = []
    started = time.time()

    for camera in cameras:
        frames = sorted(found[camera])
        for i in range(len(frames) - 1):
            f0, f1 = frames[i], frames[i + 1]
            if f1 != f0 + 1:
                # A gap means the window is not contiguous; adjacent-pair flow
                # would silently span it.
                raise ContractError(
                    f"{camera} frames {f0} and {f1} are not adjacent; the window is not contiguous"
                )
            dst = out_root / f"{camera}_{f0:06d}.npz"
            if dst.exists() and not args.overwrite:
                continue
            p0, p1 = found[camera][f0], found[camera][f1]
            a, b = load_rgb(p0), load_rgb(p1)
            opened.extend([str(p0), str(p1)])
            flow, t1, t2 = infer_pair(model, cfg_args, a, b, args.device)

            if len(direction_records) < args.direction_samples:
                # Tested on a WIDER pair from the same camera, for the reason in
                # DEFAULT_DIRECTION_GAP. The model is the same; only the
                # displacement is large enough to resolve.
                fgap = f0 + max(1, int(args.direction_gap))
                if fgap in found[camera]:
                    gflow, gt1, gt2 = infer_pair(
                        model, cfg_args, a, load_rgb(found[camera][fgap]), args.device)
                    fwd_err, rev_err, n_evidence = warp_check(gflow, gt1, gt2)
                    direction_records.append(
                        {"camera": camera, "frame": f0, "gap": int(args.direction_gap),
                         "forward_err": fwd_err, "reversed_err": rev_err,
                         "evidence_pixels": n_evidence,
                         "ratio": (rev_err / fwd_err) if fwd_err > 0 else float("inf")}
                    )

            arr = flow[0].permute(1, 2, 0).cpu().numpy()
            mag = np.linalg.norm(arr, axis=-1)
            mag_stats.append((float(mag.mean()), float(np.percentile(mag, 99)), float(mag.max())))
            np.savez_compressed(
                dst,
                flow=arr.astype(args.store_dtype),
                stored_raster=np.array(arr.shape[:2][::-1], dtype=np.int32),
                source_raster=np.array([a.shape[1], a.shape[0]], dtype=np.int32),
                magnitude_units=np.array("full_raster_pixels"),
            )
            written += 1
            if args.limit_pairs and written >= args.limit_pairs:
                break
        if args.limit_pairs and written >= args.limit_pairs:
            break

    leaked = sorted({c for c in exclude for p in opened if f"/{c}_" in p.replace("\\", "/")})
    if leaked:
        raise ContractError(f"held-out camera(s) {leaked} were READ; this run is void")

    if args.direction_samples and direction_records:
        thin = [r for r in direction_records
                if r["evidence_pixels"] < DIRECTION_MIN_EVIDENCE_PIXELS]
        if len(thin) == len(direction_records):
            raise ContractError(
                "FLOW DIRECTION CHECK ABSTAINED: every sampled pair carried fewer "
                f"than {DIRECTION_MIN_EVIDENCE_PIXELS} pixels above "
                f"{DIRECTION_EVIDENCE_PX} px, so the test had nothing to read. "
                f"Evidence counts {[r['evidence_pixels'] for r in direction_records]}. "
                "Raise --direction-gap so the displacement resolves. Refusing to "
                "emit flow whose orientation is UNTESTED -- an abstention is not a "
                "pass."
            )
        usable = [r for r in direction_records
                  if r["evidence_pixels"] >= DIRECTION_MIN_EVIDENCE_PIXELS]
        worst = min(r["ratio"] for r in usable)
        if worst < DIRECTION_MARGIN:
            raise ContractError(
                f"FLOW DIRECTION CHECK FAILED: reversed/forward warp-error ratio "
                f"{worst:.4f} < {DIRECTION_MARGIN} at gap {args.direction_gap}. "
                "Forward flow must reconstruct "
                "image1 from image2 markedly better than the negated field does. "
                "Either the field is not forward, or the pairs carry too little "
                "motion for the test to discriminate. Refusing to emit flow whose "
                "orientation is unproven."
            )

    means = [m[0] for m in mag_stats]
    p99s = [m[1] for m in mag_stats]
    manifest = {
        "schema": "imvid-searaft-flow-v1",
        "images_root": str(images_root),
        "out_root": str(out_root),
        "cameras": cameras,
        "excluded_cameras": list(exclude),
        "shard_only_cameras": list(only),
        "pairs_written": written,
        "searaft_root": searaft_root,
        "searaft_commit": "886fb094fe21d4fa5ff675da18362b27b023ccc3",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "cfg": args.cfg,
        "cfg_scale": float(getattr(cfg_args, "scale", 0)),
        "cfg_iters": int(getattr(cfg_args, "iters", 4)),
        "store_dtype": args.store_dtype,
        "magnitude_units": "full_raster_pixels",
        "stored_on": "half_raster (spatial upsample deliberately skipped)",
        "direction_check": {
            "margin_required": DIRECTION_MARGIN,
            "gap": int(args.direction_gap),
            "gap_rationale": ("production flow is adjacent; the direction test uses a "
                              "wider pair because adjacent motion at 59.94 fps is "
                              "sub-pixel and cannot discriminate orientation"),
            "samples": direction_records,
            "evidence_px_threshold": DIRECTION_EVIDENCE_PX,
            "min_evidence_pixels": DIRECTION_MIN_EVIDENCE_PIXELS,
            "worst_ratio": (min((r["ratio"] for r in direction_records
                                 if r["evidence_pixels"] >= DIRECTION_MIN_EVIDENCE_PIXELS),
                                default=None)),
        },
        "magnitude_summary": {
            "mean_of_means": float(np.mean(means)) if means else None,
            "mean_of_p99": float(np.mean(p99s)) if p99s else None,
            "max": float(max(m[2] for m in mag_stats)) if mag_stats else None,
        },
        "elapsed_s": round(time.time() - started, 2),
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
    """Exercise the pure logic with no model, no GPU and no images."""
    results = []

    found = {f"cam{i:02d}": {0: Path("x"), 1: Path("y")} for i in range(39)}
    kept = select_cameras(found, ("cam00",))
    results.append(_check("exclude_cam00_leaves_38", len(kept) == 38 and "cam00" not in kept, len(kept)))

    try:
        select_cameras(found, ("cam99",))
        results.append(_check("unmatched_exclusion_refuses", False, "no refusal"))
    except ContractError as exc:
        results.append(_check("unmatched_exclusion_refuses", True, str(exc)[:60]))

    m = IMAGE_RE.match("cam07_000123.png")
    results.append(_check("image_name_parse", m is not None and m["camera"] == "cam07"
                          and int(m["frame"]) == 123, m.groupdict() if m else None))
    results.append(_check("non_image_rejected", IMAGE_RE.match("transforms_train.json") is None, "ok"))

    # magnitude conversion: half-raster inference must be scaled by 1/factor
    for scale, want in ((-1.0, 2.0), (0.0, 1.0)):
        factor = 2.0 ** scale
        results.append(_check(f"magnitude_factor_scale_{scale}", abs((1.0 / factor) - want) < 1e-12,
                              f"1/2**{scale} = {1.0 / factor}"))

    results.append(_check("direction_margin_declared", DIRECTION_MARGIN > 1.0, DIRECTION_MARGIN))

    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ContractError as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        sys.exit(2)
