# ARS scene spec format (`ars_spec_version: "ars-1"`)

This document defines, independently of any renderer, what an ARS scene spec JSON
(`configs/ars/wave1/ARS-<layout>-<condition>.json`) describes. A renderer or verifier that implements
this document must reproduce the scene's geometry and its **front-most identity** at every pixel centre.
Textures and shading are described only as far as they are data in the spec; identity does not depend on
them.

## 1. World, time, rig

- **World frame**: right-handed, units arbitrary ("scene units"). **+y points DOWN** (gravity). The world
  up vector is `UP = (0, -1, 0)`.
- **Time**: frame `f` in `0 .. n_frames-1` (`time.n_frames = 300`), time `t = f / time.fps` seconds
  (`fps = 30`). Motions may be evaluated at fractional frames.
- **Rig**: `rig.poses_bounds` is an LLFF `poses_bounds.npy` (N x 17, float64); its sha256 is `rig.sha256`.
  Row `i` is camera `rig.camera_names[i]`. Reshape the first 15 values to a 3 x 5 matrix `M`:
  - `M[:, 0]` = camera "down" axis in world, `M[:, 1]` = camera "right" axis, `M[:, 2]` = camera "back"
    axis (the camera looks along `-M[:, 2]`), `M[:, 3]` = camera centre `C`, `M[:, 4]` = `(h, w, f)` of
    the native images.
  - Camera-to-world rotation with columns (right, down, forward): `R = [M[:,1], M[:,0], -M[:,2]]`.
- **Raster**: `rig.raster = [W, H]` = `[1352, 1014]`. Focal in raster pixels `fr = f * W / w`.
- **Pinhole, no distortion.** Principal point `(W/2, H/2)`. Pixel `(u, v)` (column `u` in `0..W-1`,
  row `v` in `0..H-1`) covers continuous image coordinates `[u, u+1) x [v, v+1)`; its **centre** is
  `(u + 0.5, v + 0.5)`. The centre ray direction in world is proportional to
  `x * R[:,0] + y * R[:,1] + 1 * R[:,2]` with `x = (u + 0.5 - W/2) / fr`, `y = (v + 0.5 - H/2) / fr`,
  starting at `C`.
- `rig.held_out` lists the evaluation camera(s) (`cam00`).

## 2. Objects and parts

`objects` is a list. Each object has:

- `name` (unique), `role` in {`static`, `ordinary`, `target`, `occupant`};
- `presence` (optional): list of inclusive `[first, last]` frame intervals. **Outside them the object is
  not in the world** (it neither renders nor occludes). If absent, the object is always present.
- `motion`: the object's root pose `(R_root, t_root)` as a function of frame (section 3);
- `parts`: list of parts. Each part has an integer `id` (the identity value it writes), a `shape`
  (section 4), a `tex` (appearance, not needed for identity), optional `local`
  (`{"pos": [x,y,z], "R": 3x3}`; defaults: zero vector, identity) and optional `joints` (list).

### Part pose

Start with `Rl = local.R`, `tl = local.pos`. For each joint in `joints` **taken in reverse order**
(last listed first):

```
angle = radians(offset_deg + amp_deg * sin(2*pi*t / period_s + radians(phase_deg)))
Rj    = rotation about unit(axis) by angle (right-hand rule, Rodrigues)
Rl    = Rj @ Rl
tl    = Rj @ (tl - pivot) + pivot
```

(`pivot` and `axis` are in the object's rest frame.) Then the part's world pose is
`R_part = R_root @ Rl`, `t_part = R_root @ tl + t_root`; a local point `p` maps to `R_part @ p + t_part`.

## 3. Motions (root pose)

Helpers:
- `rot(axis, a)`: rotation about unit `axis` by angle `a` (radians), right-hand rule.
- `R_UP = diag(1, -1, -1)` (maps object-local +y onto world up).
- `upright(yaw_deg) = rot(UP, radians(yaw_deg)) @ R_UP`. Object-local +y is up, object-local +z maps to
  `(sin yaw, 0, -cos yaw)` (toward the cameras at yaw 0).
- `smoothstep(u) = u*u*(3 - 2u)` for `u` clamped to `[0, 1]`.

| `type` | root pose at frame `f` (time `t = f / fps`) |
|---|---|
| `static` | `R = motion.R` if given, else `upright(yaw_deg)`; `t = pos` |
| `keyed` | the key whose inclusive `frames` interval contains `f`: `R = upright(key.yaw_deg)`, `t = key.pos`. Frames outside every key never occur while the object is present. |
| `slide` | `s = smoothstep((f - a) / (b - a))` with `[a, b] = frames`; `t = from + (to - from) * s`; `R = upright(yaw_deg)` |
| `roll` | as `slide`, then `R = rot(cross(UP, unit(to - from)), |to - from| * s / radius) @ upright(yaw_deg)` (rolling without slipping) |
| `spin` | `R = upright(yaw0_deg + 360 * t / period_s)`; `t = pos` |
| `sway` | `t = pos + amp * sin(2*pi*t / period_s + radians(phase_deg)) * dir`; `R = upright(yaw_deg)` |

## 4. Shapes (in the part's local frame)

| `type` | definition |
|---|---|
| `sphere` | centre at the origin, radius `r` |
| `box` | axis-aligned box centred at the origin, half extents `half = [hx, hy, hz]` |
| `cylinder` | **closed** (capped) circular cylinder, axis local +y, radius `r`, from `y = 0` (base) to `y = h` (top) |
| `plane` | the rectangle `y = 0`, `|x| <= half[0]`, `|z| <= half[1]` (zero thickness, visible from both sides). Planes are used only for the room (floor, walls, ceiling); for collision checks each is a half-space with the free side at local `y > 0`. |

## 5. Front-most identity (the ground truth a verifier must reproduce)

For a camera and frame, the identity of pixel `(u, v)` is the `id` of the part hit **first** (smallest
positive ray parameter) by the pixel-centre ray among all parts of all objects present at that frame.
Hits at ray parameter `<= 1e-9` are ignored. Exact ties are measure-zero and may be broken arbitrarily.
Every pixel-centre ray hits something (the room is closed). Several parts may share one `id`.

The rendered colour image additionally averages four sub-pixel rays at offsets `(0.25|0.75, 0.25|0.75)`
(box filter) and uses Lambertian + ambient shading with `render.light_dir` / `render.ambient`; no cast
shadows. These are not needed for identity.

## 6. Clock, windows, regions (for reference)

- `clock.A`, `clock.B`: the target is removed from the world over frames `[A, B]` (instant conditions).
- `windows.<name> = {region, frames: [lo, hi], require_nonempty_every_frame?}`, evaluated in the held-out
  camera.
- Regions:
  - `ghost`: pixels whose centre ray hits a target part at the target's pose of frame `A - 1`, ignoring
    every other object.
  - `target`: front-most identity is a target part.
  - `vacated`: `ghost` minus the target-alone projection at the current pose.
  - `occupant`: `ghost` intersected with front-most identity being an occupant part.
  - `ordinary`: front-most identity of role `ordinary`.
  - `static_complement`: front-most identity of role `static`, minus a 2-px dilation of the union of all
    target and occupant silhouettes over the sequence.
  - `whole`: every pixel.

## 7. Scene outputs

- `frames/<camera>/<frame:04d>.png`: 8-bit RGB images.
- `gt/identity/<camera>.npz`: one entry `f<frame:04d>` per frame, an int16 `H x W` array of the section 5
  identity. In a partial render, the file is `<camera>_partial.npz`.
- `event_spec.json`: a byte copy of the spec.
- `rig/poses_bounds.npy`: a byte copy of the rig.
- `MANIFEST.json`: the sha256 of every file.
