"""fmd mono2 lane: Shape of Motion (iPhone/DyCheck route, code unmodified) input tree from a real monocular sweep.

Reads the Multi4D/Nerfies sweep tree (fmd/data/mono_real/<scene>/multi4d) and writes a tree in the layout that
flow3d/data/iphone_dataset.py reads with --data.camera-type original (the KNOWN cameras, camera/*.json):
  rgb/1x/mono_NNNN.png            RGBA copies of the sweep frames (alpha = 255: every pixel valid)
  camera/mono_NNNN.json           copied unchanged (orientation = world-to-camera R, position = centre, OpenCV axes)
  splits/train.json, val.json     train = the 300 sweep frames (time_id = frame index); val empty (held-out cameras
                                  are rendered after training by render_som.py, not through the trainer)
  extra.json                      fps 30
  flow3d_preprocessed/track_anything/1x/*.png   moving-content masks (ADAPTER, see below)
  flow3d_preprocessed/depth_anything/1x/*.png   Depth Anything v1 large disparity, SoM's own compute_depth.py
  flow3d_preprocessed/aligned_depth_anything_colmap/1x/*.npy  disparity aligned with SoM's own scale/shift formula
                                  (align_monodepth_with_colmap), but against the sweep's starting cloud (points.npy,
                                  COLMAP dense of frame 0 over 4 sweep cameras) projected through the KNOWN cameras
                                  instead of a COLMAP sparse model (ADAPTER)
  flow3d_preprocessed/rgb_preproc/ RGB symlinks for the trackers (the RGBA copies would give TAPIR 4 channels)
Mask ADAPTER (in place of Track-Anything clicks): every N3V camera is static, so frame t (taken by sweep camera c(t))
is masked where |fixture_c(t) - temporal median of camera c's fixture footage| > 20/255 on the max channel, then
opened (3 px) and dilated (7 px). This marks the cook, hands, pans and flames, and also the vacated surface during the
gap (it differs from the median), i.e. it is the method-favourable choice.
Usage: prep_som.py SCENE OUT  (stage 'tree' on CPU; stage 'depth' needs the GPU)
"""
import sys, os, json, glob, hashlib, shutil
import numpy as np
from PIL import Image
from scipy.ndimage import binary_opening, binary_dilation

scene, out, stage = sys.argv[1], sys.argv[2], sys.argv[3]
SRC = f"/leonardo_work/EUHPC_D36_068/sri/proj_adags/fmd/data/mono_real/{scene}/multi4d"
SOM = "/leonardo_scratch/large/userexternal/siyengar/fmd/repo/shape-of-motion"
man = json.load(open(f"{SRC}/MANIFEST.json"))
T = 300
names = [f"mono_{t:04d}" for t in range(T)]
per = man["frames_per_camera"]; sweep = man["sweep_order"]
seq = [sweep[min(int(t / per), len(sweep) - 1)] for t in range(T)]
for t in range(T):  # the symlink must point at the camera the formula names
    tgt = os.path.realpath(f"{SRC}/rgb/1x/{names[t]}.png")
    assert seq[t] in tgt and f"{t:04d}.png" in os.path.basename(tgt), (t, tgt, seq[t])
P = f"{out}/flow3d_preprocessed"


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]; return x * x + y * y <= r * r


if stage == "tree":
    if os.path.exists(out): sys.exit(f"REFUSE {out} exists")
    for d in ["rgb/1x", "camera", "splits", f"{P}/track_anything/1x", f"{P}/rgb_preproc"]:
        os.makedirs(d if d.startswith("/") else f"{out}/{d}")
    for n in names:
        im = np.asarray(Image.open(f"{SRC}/rgb/1x/{n}.png").convert("RGB"))
        Image.fromarray(np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])).save(f"{out}/rgb/1x/{n}.png")
        os.symlink(os.path.realpath(f"{SRC}/rgb/1x/{n}.png"), f"{P}/rgb_preproc/{n}.png")
        shutil.copy(f"{SRC}/camera/{n}.json", f"{out}/camera/{n}.json")
    json.dump({"frame_names": names, "time_ids": list(range(T))}, open(f"{out}/splits/train.json", "w"))
    json.dump({"frame_names": [], "time_ids": []}, open(f"{out}/splits/val.json", "w"))
    json.dump({"fps": 30.0}, open(f"{out}/extra.json", "w"))
    fx = man["fixture"]; frac = {}
    for c in sorted(set(seq)):
        vid = [np.asarray(Image.open(f"{fx}/{c}/images/{t:04d}.png").convert("RGB")) for t in range(0, T, 2)]
        med = np.median(np.stack(vid), 0).astype(np.float32); del vid
        for t in [t for t in range(T) if seq[t] == c]:
            f = np.asarray(Image.open(f"{fx}/{c}/images/{t:04d}.png").convert("RGB")).astype(np.float32)
            m = np.abs(f - med).max(-1) > 20
            m = binary_dilation(binary_opening(m, disk(3)), disk(7))
            frac[names[t]] = float(m.mean())
            Image.fromarray((m * 255).astype(np.uint8)).save(f"{P}/track_anything/1x/{names[t]}.png")
        print(c, "mask frac", round(np.mean([frac[names[t]] for t in range(T) if seq[t] == c]), 4), flush=True)
    json.dump({"scene": scene, "src": SRC, "seq": seq, "mask_frac": frac,
               "mask_rule": "|fixture - per-camera temporal median (stride 2)| > 20 max-channel; open r3; dilate r7"},
              open(f"{out}/MANIFEST_som.json", "w"), indent=1)
    print("tree done; mean mask frac", np.mean(list(frac.values())))

if stage == "depth":
    sys.path.insert(0, f"{SOM}/preproc")
    import cv2
    from compute_depth import save_disp_from_dir, UINT16_MAX  # SoM's own Depth Anything route, unmodified
    save_disp_from_dir("depth-anything", f"{P}/rgb_preproc", f"{P}/depth_anything/1x", "mono_*")
    pts = np.load(f"{SRC}/points.npy").astype(np.float64)
    os.makedirs(f"{P}/aligned_depth_anything_colmap/1x", exist_ok=True)
    log = {}
    for n in names:
        cam = json.load(open(f"{out}/camera/{n}.json"))
        R = np.array(cam["orientation"]); C = np.array(cam["position"]); f = cam["focal_length"]; pp = cam["principal_point"]
        W, H = cam["image_size"]
        Xc = (pts - C) @ R.T; z = Xc[:, 2]; ok = z > 1e-3
        u = f * Xc[ok, 0] / z[ok] + pp[0]; v = f * Xc[ok, 1] / z[ok] + pp[1]; z = z[ok]
        ib = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1); u, v, z = u[ib], v[ib], z[ib]
        # z-buffer on 3-px cells: keep the nearest point per cell (visible surface)
        cell = (np.floor(v / 3).astype(np.int64) * (W // 3 + 1) + np.floor(u / 3).astype(np.int64))
        o = np.lexsort((z, cell)); first = np.ones(len(o), bool); first[1:] = cell[o][1:] != cell[o][:-1]
        sel = o[first]; u, v, z = u[sel], v[sel], z[sel]
        fg = np.asarray(Image.open(f"{P}/track_anything/1x/{n}.png")) > 0
        keep = ~fg[np.round(v).astype(int), np.round(u).astype(int)]; u, v, z = u[keep], v[keep], z[keep]
        mono_disp_map = cv2.imread(f"{P}/depth_anything/1x/{n}.png", cv2.IMREAD_UNCHANGED) / UINT16_MAX
        # --- SoM align_monodepth_with_colmap formula, verbatim from here ---
        colmap_disp = 1.0 / np.clip(z, a_min=1e-6, a_max=1e6)
        mono_disp = cv2.remap(mono_disp_map, np.stack([u, v], -1)[None].astype(np.float32), None, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)[0]
        ms_colmap_disp = colmap_disp - np.median(colmap_disp) + 1e-8
        ms_mono_disp = mono_disp - np.median(mono_disp) + 1e-8
        scale = np.median(ms_colmap_disp / ms_mono_disp)
        shift = np.median(colmap_disp - scale * mono_disp)
        mono_disp_aligned = scale * mono_disp_map + shift
        min_thre = min(1e-6, np.quantile(mono_disp_aligned, 0.01))
        mono_disp_aligned[mono_disp_aligned < min_thre] = 0.0
        np.save(f"{P}/aligned_depth_anything_colmap/1x/{n}.npy", mono_disp_aligned)
        # sanity: relative depth error of the aligned map at the cloud points
        dz = 1.0 / np.clip(scale * mono_disp + shift, 1e-6, None)
        log[n] = {"npts": int(len(z)), "scale": float(scale), "shift": float(shift),
                  "med_rel_err": float(np.median(np.abs(dz - z) / z))}
    json.dump(log, open(f"{out}/depth_align_log.json", "w"), indent=1)
    e = [l["med_rel_err"] for l in log.values()]; npt = [l["npts"] for l in log.values()]
    print("depth align: median rel err", np.median(e), "p90", np.quantile(e, 0.9), "pts min/med", min(npt), np.median(npt))
