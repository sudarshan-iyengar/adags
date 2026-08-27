#!/usr/bin/env python3
"""Convert a decoded Google Immersive scene into the Blender-branch layout.

The ADAGS `Scene` loader dispatches on disk contents, first match wins:
`sparse/` -> Colmap, `train_meta.json`+`test_meta.json` -> PanopticSports,
`transforms_train.json` -> Blender. Immersive has to become the third, so this
writes `transforms_train.json` / `transforms_test.json`, `images/`, and
`points3d.ply`, and REFUSES to leave a `sparse/` directory under the output
root because that would silently win the dispatch.

WHAT THIS IS NOT
----------------
STG does not train Immersive in pinhole space. It keeps the fisheye images and
warps the RENDER into fisheye at loss time through a per-camera inverse flow
map. This converter produces a pinhole scene, which is a DIFFERENT METHOD, and
numbers from it are NOT comparable to STG's or the ImViD paper's published
Immersive figures. That is recorded here rather than in a footnote because the
comparison is the thing a reader will reach for.

THE PROJECTION MODEL is the publisher's own, from the deepview_video_dataset
README:

    r     = hypot(x, y);   theta = atan2(r, z)
    d     = 1 + theta^2 * (k1 + theta^2 * k2)
    (u,v) = (theta/r * x * d, theta/r * y * d)      then K @ (u, v, 1)

    R = Rotation.from_rotvec(view['orientation']).as_matrix()
    t = -R @ position                      # position is the camera CENTRE

which is equidistant fisheye with a radial polynomial in theta -- i.e.
`cv2.fisheye` (Kannala-Brandt) with k3 = k4 = 0. `--check-model` asserts the
two agree, so the OpenCV path is validated against the publisher's own code
rather than assumed equivalent to it.

FOCAL SCALE. A pinhole camera at the fisheye's focal length sees far less of
the world (r = f*tan(theta) vs r = f*theta), so the undistorted intrinsics need
a focal BELOW the fisheye focal or the frame is a narrow crop. STG uses 0.5 on
its undistorted path; that is the default here and it is a declared parameter,
not a hidden constant.

The one discipline carried verbatim from scripts/imvid_to_blender.py: the
`K_new` handed to `cv2.fisheye.initUndistortRectifyMap` is the SAME object
whose floats are written into `transforms_*.json`. Pixels and intrinsics cannot
disagree because there is only one of it.

Usage:
  python3 scripts/immersive_to_blender.py --frames-root <decoded> \\
      --archive <scene>.zip --out <scene_pinhole> --scale 0.5 --fps-rational 30/1
  python3 scripts/immersive_to_blender.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402
from scripts.immersive_decode_frames import read_models  # noqa: E402

HELD_OUT = ("camera_0001",)          # STG's Immersive split; ImViD says Camera 1
POINTCLOUD_BASENAME = "points3d.ply"
POSE_ROUNDTRIP_TOL = 1e-9


# --------------------------------------------------------------------------
# camera model
# --------------------------------------------------------------------------
def publisher_project(point_cam: np.ndarray, k: np.ndarray) -> np.ndarray:
    """The README's fisheye_to_perspective, transcribed. Reference only."""
    x, y, z = float(point_cam[0]), float(point_cam[1]), float(point_cam[2])
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(r, z)
    r2 = theta * theta
    d = 1.0 + r2 * (k[0] + r2 * k[1])
    if r == 0.0:
        return np.array([0.0, 0.0, 1.0])
    return np.array([theta / r * x * d, theta / r * y * d, 1.0])


def view_to_camera(view: dict) -> dict:
    """models.json entry -> K, D, R, t. `position` is the camera CENTRE."""
    from scipy.spatial.transform import Rotation

    R = Rotation.from_rotvec(np.asarray(view["orientation"], float)).as_matrix()
    C = np.asarray(view["position"], float).reshape(3)
    t = -R @ C
    f = float(view["focal_length"])
    par = float(view.get("pixel_aspect_ratio", 1.0))
    cx, cy = (float(v) for v in view["principal_point"])
    K = np.array([[f, 0.0, cx], [0.0, f * par, cy], [0.0, 0.0, 1.0]])
    rd = [float(v) for v in view["radial_distortion"]]
    if len(rd) < 2:
        raise ContractError(f"{view['name']}: radial_distortion has <2 terms")
    if len(rd) > 2 and any(abs(v) > 0 for v in rd[2:]):
        raise ContractError(
            f"{view['name']}: radial_distortion carries non-zero terms beyond "
            f"k1,k2 ({rd}); cv2.fisheye k3/k4 would have to be populated"
        )
    D = np.array([rd[0], rd[1], 0.0, 0.0])
    return {"name": view["name"], "K": K, "D": D, "R": R, "t": t, "C": C,
            "width": int(view["width"]), "height": int(view["height"])}


def new_intrinsics(cam: dict, scale: float, focal_scale: float) -> tuple[np.ndarray, int, int]:
    """The undistorted pinhole K, and the raster it belongs to.

    `scale` is the resolution factor (0.5 == the 2x downsample the ImViD paper
    applies to Immersive). Principal point uses the pixel-centre convention
    (c + 0.5) * s - 0.5 rather than a naive c * s.
    """
    w_out = int(round(cam["width"] * scale))
    h_out = int(round(cam["height"] * scale))
    f_new = cam["K"][0, 0] * focal_scale * scale
    fy_new = cam["K"][1, 1] * focal_scale * scale
    cx_new = (cam["K"][0, 2] + 0.5) * scale - 0.5
    cy_new = (cam["K"][1, 2] + 0.5) * scale - 0.5
    K_new = np.array([[f_new, 0.0, cx_new], [0.0, fy_new, cy_new], [0.0, 0.0, 1.0]])
    return K_new, w_out, h_out


def c2w_blender(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """world->camera (R, t) as a Blender/OpenGL camera-to-world 4x4.

    The reader does `c2w[:3,1:3] *= -1` then inverts, so columns 1 and 2 are
    pre-negated here and the round trip is asserted by the caller.
    """
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    c2w[:3, 1:3] *= -1
    return c2w


def invert_blender(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exactly what scene/dataset_readers.py:readCamerasFromTransforms does."""
    m = np.array(c2w, dtype=float, copy=True)
    m[:3, 1:3] *= -1
    w2c = np.linalg.inv(m)
    return w2c[:3, :3], w2c[:3, 3]


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
def self_test() -> int:
    """Runs with no data. Every check below has failed for someone before."""
    from scipy.spatial.transform import Rotation
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
        if not cond:
            ok = False

    view = {"name": "camera_0001",
            "position": [0.003311238794086777, 0.000126384684934784, 0.42421246053630285],
            "orientation": [-0.03295944563072383, 0.02964606619481096, -0.03337751007195316],
            "focal_length": 1111.3638715594666, "pixel_aspect_ratio": 1.0,
            "principal_point": [1284.7296468363463, 925.4651609728676],
            "height": 1920.0, "width": 2560.0,
            "radial_distortion": [0.0970525284520715, -0.01708587111009977, 0.0],
            "projection_type": "fisheye"}

    # 1. The publisher's own worked example, from the README verbatim.
    cam = view_to_camera(view)
    Rm = Rotation.from_rotvec(np.asarray(view["orientation"])).as_matrix()
    extr = np.concatenate((Rm, -Rm @ np.asarray(view["position"])[:, None]), axis=1)
    local = extr @ np.array([0.5, 0.5, 10.0, 1.0])
    px = cam["K"] @ publisher_project(local, cam["D"][:2])
    expected = np.array([1377.85525, 1017.61440])
    check("publisher worked example reproduces [1377.855, 1017.614]",
          np.allclose(px[:2], expected, atol=2e-2), f"got {px[:2]}")

    # 2. Our (R, t) must equal the README's own extrinsics construction.
    check("extrinsics match the README construction",
          np.allclose(np.c_[cam["R"], cam["t"]], extr, atol=1e-12))

    # 3. cv2.fisheye must AGREE with the publisher's model, not merely resemble
    #    it. If it does not, the undistortion is wrong everywhere at once.
    try:
        import cv2
        pts = []
        for dx in (-0.4, -0.1, 0.0, 0.2, 0.5):
            for dy in (-0.3, 0.0, 0.35):
                pts.append([dx, dy, 1.0])
        pts = np.asarray(pts, float)
        theirs = np.array([(cam["K"] @ publisher_project(p, cam["D"][:2]))[:2] for p in pts])
        ours = cv2.fisheye.projectPoints(
            pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
            cam["K"], cam["D"])[0].reshape(-1, 2)
        err = float(np.abs(theirs - ours).max())
        check("cv2.fisheye agrees with the publisher model (<1e-6 px)",
              err < 1e-6, f"max |diff| = {err:.3e} px")
    except ImportError:
        print("  SKIP  cv2 not importable here; the agreement check runs on the cluster")

    # 4. Pose round trip through the reader's exact arithmetic.
    c2w = c2w_blender(cam["R"], cam["t"])
    R_back, t_back = invert_blender(c2w)
    check("pose survives the reader's inverse",
          np.allclose(R_back, cam["R"], atol=POSE_ROUNDTRIP_TOL)
          and np.allclose(t_back, cam["t"], atol=POSE_ROUNDTRIP_TOL))

    # 5. ANTI-VACUITY. The round trip must FAIL on each corruption, or it is
    #    not testing anything. This is the check that catches a converter that
    #    "passes" while writing transposed rotations.
    Rt = cam["R"].T
    bad_c2w = c2w_blender(Rt, cam["t"])
    R_bad, _ = invert_blender(bad_c2w)
    check("round trip DETECTS a transposed rotation",
          not np.allclose(R_bad, cam["R"], atol=1e-6))
    c2w_noflip = np.eye(4)
    c2w_noflip[:3, :3] = cam["R"].T
    c2w_noflip[:3, 3] = -cam["R"].T @ cam["t"]     # forgot the axis flip
    R_nf, t_nf = invert_blender(c2w_noflip)
    check("round trip DETECTS a missing OpenGL axis flip",
          not (np.allclose(R_nf, cam["R"], atol=1e-6)
               and np.allclose(t_nf, cam["t"], atol=1e-6)))
    c2w_cw = c2w_blender(cam["R"], -cam["R"] @ cam["C"] * -1.0)
    R_cw, t_cw = invert_blender(c2w_cw)
    check("round trip DETECTS a sign-flipped translation",
          not np.allclose(t_cw, cam["t"], atol=1e-6))

    # 6. Principal point convention, and the raster it belongs to.
    K_new, w, h = new_intrinsics(cam, 0.5, 0.5)
    check("2x downsample gives 1280x960", (w, h) == (1280, 960), f"got {w}x{h}")
    check("focal is scaled by BOTH focal_scale and resolution",
          abs(K_new[0, 0] - 1111.3638715594666 * 0.5 * 0.5) < 1e-9)
    check("principal point uses the pixel-centre convention",
          abs(K_new[0, 2] - ((1284.7296468363463 + 0.5) * 0.5 - 0.5)) < 1e-12)

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# point cloud
# --------------------------------------------------------------------------
def triangulate_cloud(cams: list[dict], K_news: dict, images: dict,
                      max_points: int, min_views: int = 3,
                      reproj_px: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """SIFT + pairwise triangulation on ONE frame, using the KNOWN poses.

    The Blender reader fabricates a uniform random cloud when points3d.ply is
    absent, without erroring -- the DiVa-360 silent-initialisation failure. So
    a real cloud is produced here and its size is asserted by the caller.
    """
    import cv2

    sift = cv2.SIFT_create(nfeatures=4000)
    feats = {}
    for c in cams:
        img = cv2.imread(str(images[c["name"]]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ContractError(f"could not read {images[c['name']]}")
        kp, des = sift.detectAndCompute(img, None)
        if des is None or len(kp) < 20:
            continue
        feats[c["name"]] = (np.array([k.pt for k in kp], np.float32), des)

    if len(feats) < 2:
        raise ContractError("fewer than two cameras produced SIFT features")

    centres = np.array([c["C"] for c in cams])
    names = [c["name"] for c in cams]
    by_name = {c["name"]: c for c in cams}
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    P = {}
    for c in cams:
        Kn = K_news[c["name"]]
        P[c["name"]] = Kn @ np.hstack([c["R"], c["t"].reshape(3, 1)])

    pts, cols = [], []
    colour_cache = {}
    for i, name_a in enumerate(names):
        if name_a not in feats:
            continue
        d = np.linalg.norm(centres - centres[i], axis=1)
        order = np.argsort(d)[1:4]                    # three nearest neighbours
        for j in order:
            name_b = names[j]
            if name_b not in feats or name_b <= name_a:
                continue
            (pa, da), (pb, db) = feats[name_a], feats[name_b]
            raw = matcher.knnMatch(da, db, k=2)
            good = [m for m, n in (r for r in raw if len(r) == 2)
                    if m.distance < 0.75 * n.distance]
            if len(good) < 8:
                continue
            ia = np.array([m.queryIdx for m in good])
            ib = np.array([m.trainIdx for m in good])
            xa, xb = pa[ia].T, pb[ib].T
            X = cv2.triangulatePoints(P[name_a], P[name_b], xa, xb)
            X = (X[:3] / np.where(np.abs(X[3]) < 1e-12, np.nan, X[3])).T
            keep = np.isfinite(X).all(axis=1)
            for nm, x2 in ((name_a, xa), (name_b, xb)):
                cam = by_name[nm]
                Xc = (cam["R"] @ X.T).T + cam["t"]
                keep &= Xc[:, 2] > 1e-6                       # cheirality
                proj = (P[nm] @ np.c_[X, np.ones(len(X))].T)
                proj = (proj[:2] / np.where(np.abs(proj[2]) < 1e-12, np.nan, proj[2])).T
                keep &= np.nan_to_num(
                    np.linalg.norm(proj - x2.T, axis=1), nan=1e9) < reproj_px
            if not keep.any():
                continue
            if name_a not in colour_cache:
                colour_cache[name_a] = cv2.imread(str(images[name_a]), cv2.IMREAD_COLOR)
            rgb_img = colour_cache[name_a]
            uv = np.round(xa.T[keep]).astype(int)
            uv[:, 0] = np.clip(uv[:, 0], 0, rgb_img.shape[1] - 1)
            uv[:, 1] = np.clip(uv[:, 1], 0, rgb_img.shape[0] - 1)
            cols.append(rgb_img[uv[:, 1], uv[:, 0]][:, ::-1])   # BGR -> RGB
            pts.append(X[keep])

    if not pts:
        raise ContractError("triangulation produced no points")
    pts = np.concatenate(pts).astype(np.float64)
    cols = np.concatenate(cols).astype(np.uint8)

    # Drop far outliers: the rig is a sub-metre dome, so anything decades away
    # is a mismatch, not geometry.
    centre = np.median(pts, axis=0)
    rad = np.linalg.norm(pts - centre, axis=1)
    keep = rad < np.percentile(rad, 98.0)
    pts, cols = pts[keep], cols[keep]

    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    return pts, cols


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> str:
    from plyfile import PlyData, PlyElement
    n = len(xyz)
    arr = np.empty(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames-root", type=Path)
    ap.add_argument("--archive", type=Path, help="scene zip, for models.json")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="resolution factor; 0.5 is the ImViD 2x downsample")
    ap.add_argument("--focal-scale", type=float, default=0.5,
                    help="pinhole focal / fisheye focal; STG uses 0.5 undistorted")
    ap.add_argument("--fps-rational", default=None,
                    help="REQUIRED, e.g. 30/1. Decimals are refused.")
    ap.add_argument("--frames", nargs=2, type=int, default=[0, 50])
    ap.add_argument("--max-points", type=int, default=200_000)
    ap.add_argument("--expect-points-min", type=int, default=5_000)
    ap.add_argument("--check-model", action="store_true",
                    help="assert cv2.fisheye == the publisher model, then exit")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    for req in ("frames_root", "archive", "out"):
        if getattr(args, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required")
    if not args.fps_rational:
        ap.error("--fps-rational is required (e.g. 30/1); decimals are refused")
    if "/" not in args.fps_rational:
        raise ContractError("--fps-rational must be NUM/DEN, not a decimal")
    num, den = (int(v) for v in args.fps_rational.split("/"))
    fps = num / den

    import cv2
    from PIL import Image

    views, _ = read_models(args.archive)
    cams = [view_to_camera(v) for v in views]
    K_news, rasters = {}, {}
    for c in cams:
        K_new, w, h = new_intrinsics(c, args.scale, args.focal_scale)
        K_news[c["name"]] = K_new
        rasters[c["name"]] = (w, h)

    if args.check_model:
        return self_test()

    start, end = args.frames
    out_images = args.out / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    # ---- undistort every frame -------------------------------------------
    frame0 = {}
    for idx, c in enumerate(cams, 1):
        name = c["name"]
        w, h = rasters[name]
        # ONE K_new object: it maps the pixels AND it is written to JSON.
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(
            c["K"], c["D"], np.eye(3), K_news[name], (w, h), cv2.CV_32FC1)
        src_dir = args.frames_root / name
        for f in range(start, end):
            src = src_dir / f"{f:06d}.png"
            if not src.exists():
                raise ContractError(f"missing decoded frame {src}")
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                raise ContractError(f"unreadable {src}")
            if (img.shape[1], img.shape[0]) != (c["width"], c["height"]):
                raise ContractError(
                    f"{src}: {img.shape[1]}x{img.shape[0]} != models.json "
                    f"{c['width']}x{c['height']}")
            dst = cv2.remap(img, m1, m2, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
            op = out_images / f"{name}_{f:06d}.png"
            cv2.imwrite(str(op), dst)
            if f == start:
                frame0[name] = op
        print(f"  [{idx:2d}/{len(cams)}] {name}: undistorted {end - start} frames",
              flush=True)

    # ---- point cloud ------------------------------------------------------
    train_cams = [c for c in cams if c["name"] not in HELD_OUT]
    xyz, rgb = triangulate_cloud(train_cams, K_news, frame0, args.max_points)
    if len(xyz) < args.expect_points_min:
        raise ContractError(
            f"triangulated only {len(xyz)} points (< {args.expect_points_min}). "
            "Refusing: the Blender reader substitutes a RANDOM uniform cloud "
            "for a missing/!unusable points3d.ply without erroring.")
    ply_sha = write_ply(args.out / POINTCLOUD_BASENAME, xyz, rgb)
    print(f"points3d.ply: {len(xyz)} points  sha256 {ply_sha[:16]}")

    # ---- transforms -------------------------------------------------------
    def frames_for(cam_list):
        out = []
        for c in cam_list:
            name = c["name"]
            w, h = rasters[name]
            Kn = K_news[name]
            c2w = c2w_blender(c["R"], c["t"])
            R_back, t_back = invert_blender(c2w)
            if not (np.allclose(R_back, c["R"], atol=POSE_ROUNDTRIP_TOL)
                    and np.allclose(t_back, c["t"], atol=POSE_ROUNDTRIP_TOL)):
                raise ContractError(f"{name}: pose round trip failed")
            for f in range(start, end):
                out.append({
                    "file_path": f"images/{name}_{f:06d}",
                    "time": (f - start) / fps,
                    "fl_x": float(Kn[0, 0]), "fl_y": float(Kn[1, 1]),
                    "cx": float(Kn[0, 2]), "cy": float(Kn[1, 2]),
                    "w": w, "h": h,
                    "transform_matrix": c2w.tolist(),
                })
        return out

    test_cams = [c for c in cams if c["name"] in HELD_OUT]
    if not test_cams:
        raise ContractError(f"held-out camera(s) {HELD_OUT} absent from models.json")
    for tag, cam_list in (("train", train_cams), ("test", test_cams)):
        payload = {"camera_model": "PINHOLE", "frames": frames_for(cam_list)}
        (args.out / f"transforms_{tag}.json").write_text(
            json.dumps(payload, indent=1), encoding="utf-8")

    # ---- refusals ---------------------------------------------------------
    if (args.out / "sparse").exists():
        raise ContractError(
            "a `sparse/` directory exists under the output root; it would win "
            "the loader's dispatch and route this scene to the Colmap branch")
    with Image.open(frame0[cams[0]["name"]]) as im:
        if im.mode not in ("RGB", "L"):
            raise ContractError(f"unexpected image mode {im.mode}; alpha is not handled")

    manifest = {
        "out": str(args.out), "archive": str(args.archive),
        "scale": args.scale, "focal_scale": args.focal_scale,
        "fps_rational": args.fps_rational, "frames": [start, end],
        "cameras_total": len(cams), "cameras_train": len(train_cams),
        "held_out": list(HELD_OUT), "raster": list(rasters[cams[0]["name"]]),
        "points": int(len(xyz)), "points3d_sha256": ply_sha,
        "comparability": ("PINHOLE PORT -- not comparable to published STG or "
                          "ImViD Immersive numbers, which train in fisheye space"),
    }
    mp = args.manifest or (args.out / "convert_manifest.json")
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
