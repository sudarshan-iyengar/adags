"""fmd mono2 lane: MoSca (iPhone/DyCheck route, code unmodified) workspace from a real monocular sweep.

Writes <ws>/ with
  images/mono_NNNN.png        sweep frames downsampled 2x to 676x507 (DyCheck iPhone protocol trains at 2x; MoSca loads
                              every buffer of all frames onto the GPU, ~22 GB at full resolution)
  cameras/0_NNNNN.json        KNOWN training cameras (Nerfies JSON, MoSca's iphone_helpers reader): focal and principal
                              point halved, image_size [676, 507], position / S
  test_cameras/{1,2,3}_NNNNN.json  held-out cam00 / cam05 / cam17 at full resolution (1352x1014), position / S
  test_images/{1,2,3}_NNNNN.png    symlinks to the held-out fixture frames (read only by MoSca's own end-of-run
                              evaluation, which does a pose-only test-time alignment against them)
  sensor_depth/mono_NNNN.npz  key 'dep': depth = 1 / (the Shape of Motion aligned Depth Anything disparity), i.e. the
                              same depth as the SoM run, aligned to the KNOWN cameras' scale, nearest-downsampled, / S
ADAPTERS: (1) world scale S = 8: N3V world units are ~8x metres (median depth of the starting cloud from cam00 = 16
units; the cook stands ~2 m from the rig), and MoSca's published iPhone config fixes sizes in metres
(mosca_unit_world 0.02, gs_radius_max 0.03) for metric sensor depth. Cameras and depth are divided by the same S, so
geometry is unchanged up to scale. (2) 'sensor' depth = aligned monocular depth (the iPhone config reads LiDAR depth).
Usage: prep_mosca.py SCENE SOM_DATA_DIR WS
"""
import sys, os, json
import numpy as np, cv2
from PIL import Image

scene, som, ws = sys.argv[1:4]
S = 8.0
SRC = f"/leonardo_work/EUHPC_D36_068/sri/proj_adags/fmd/data/mono_real/{scene}/multi4d"
if os.path.exists(ws): sys.exit(f"REFUSE {ws} exists")
for d in ["images", "cameras", "test_cameras", "test_images", "sensor_depth"]: os.makedirs(f"{ws}/{d}")
T = 300; W2, H2 = 676, 507


def cam(c, half):
    j = json.load(open(f"{SRC}/camera/{c}.json"))
    j["position"] = (np.array(j["position"]) / S).tolist()
    if half:
        j["focal_length"] /= 2; j["principal_point"] = [p / 2 for p in j["principal_point"]]; j["image_size"] = [W2, H2]
    return j


stats = []
for t in range(T):
    n = f"mono_{t:04d}"
    im = Image.open(f"{SRC}/rgb/1x/{n}.png").convert("RGB")
    im.resize((W2, H2), Image.LANCZOS).save(f"{ws}/images/{n}.png")
    json.dump(cam(n, True), open(f"{ws}/cameras/0_{t:05d}.json", "w"))
    disp = np.load(f"{som}/flow3d_preprocessed/aligned_depth_anything_colmap/1x/{n}.npy")
    dep = np.where(disp > 1e-6, 1.0 / np.maximum(disp, 1e-6), 0.0) / S
    dep = cv2.resize(dep.astype(np.float32), (W2, H2), interpolation=cv2.INTER_NEAREST)
    np.savez_compressed(f"{ws}/sensor_depth/{n}.npz", dep=dep)
    stats.append(np.median(dep[dep > 0]))
for k, c in [(1, "cam00"), (2, "cam05"), (3, "cam17")]:
    for t in range(T):
        json.dump(cam(f"{c}_{t:04d}", False), open(f"{ws}/test_cameras/{k}_{t:05d}.json", "w"))
        os.symlink(os.path.realpath(f"{SRC}/rgb/1x/{c}_{t:04d}.png"), f"{ws}/test_images/{k}_{t:05d}.png")
json.dump({"scene": scene, "S": S, "train_res": [W2, H2], "test_res": [1352, 1014], "test_ids": {"1": "cam00", "2": "cam05", "3": "cam17"},
           "depth_src": f"{som}/flow3d_preprocessed/aligned_depth_anything_colmap/1x", "median_depth_m": float(np.median(stats))},
          open(f"{ws}/MANIFEST_mosca.json", "w"), indent=1)
print("mosca ws done; median scene depth (scaled units)", np.median(stats))
