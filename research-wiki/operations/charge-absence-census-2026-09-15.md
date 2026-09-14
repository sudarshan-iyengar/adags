---
title: "Charge dataset (arXiv 2512.13639) — public-release check and full-multiview absence-and-return census (2026-09-14/15): NEGATIVE at scene level on 6 of 8 scenes"
date: 2026-09-15
evidence_bearing: false
---

# Charge absence-and-return census (2026-09-14/15)

Answer to the question recorded in spec v2.0.0 §13.21 ("does Charge
contain a scene-level object or character that leaves every camera and
returns?"): **NO on the evidence recoverable from the released
metadata**, on 6 of the 8 scenes; 2 scenes were not censused (download
budget). Read-only work by two bounded subagents on Windows; nothing on
Leonardo and nothing in the repository was touched. Source memo (session
scratchpad, may be lost): `charge_census_memo.md` with scripts
`download_charge_masks.py`, `census_charge_absence.py`,
`peak_size_check.py`, `probe_repo_sizes2.py`, `probe_camdirs.py`,
`census_scene_full.py` and per-scene `census_<scene>_full.json`.

## 1. Release, licence, size (verified)

* Public on Hugging Face, org `charge-benchmark`, 14 dataset repos
  (Dense / Sparse-Mono / camera-info / one optical-flow repo); index repo
  `charge-benchmark/Charge` with a `downloader.py`. `gated: false` on
  every repo checked; anonymous downloads succeeded.
* Licence CC BY 4.0 (project page statement; `license:cc-by-4.0` tag
  present on the index repo and on 3 of the 8 per-scene Dense repos, the
  other 5 carry no licence tag — a tagging gap, not evidence of a
  different licence).
* Size: no published total; `Charge-050_0130` all modalities = 283.0 GB
  (RGB 8.1, depth 49.1, segmentation 1.2, normal 28.4, dyn-mask 0.2,
  flow 98.1); `Charge-060_0100` ≈ 158 GB. Paper: 8 scenes, 25 + 16 dense /
  9 + 10 sparse / 4 + 16 mono cameras per scene, 185,600 frames, 2048×858.
* Masks: `frame_XXXX_segmentation.png`, uint16 single-channel, one
  arbitrary id per Blender mesh object (0 = background) with a scene-level
  `segmentation.json` id → mesh name; per-MESH-PART, not per character (the
  robot is hundreds of `GEO-*` sub-meshes; no id is "the whole robot").

## 2. Census method

Per scene, Dense task only (25 train + 16 test cameras, the same 41-camera
grid on every scene checked), segmentation PNGs only (0.6–2.1 GB per
scene). For every instance id: a per-frame boolean "seen by ≥ 1 of the 41
cameras"; an EVENT = a maximal run of ≥ 5 consecutive frames absent from
every camera, present in the frame before and after. Peak footprint =
the id's pixel count summed over cameras at its largest frame. Grouped
test (second pass): ids grouped by name prefix (rule a: before the first
`-`, `.` or `_`; rule b: first 4 characters), the union mask's visibility
tested the same way (exact, since each pixel carries one id).

## 3. Results

| scene | cameras | frames | ids | id-level events (ids hit) | largest event footprint (px) | group-level events |
|---|---:|---|---:|---|---:|---:|
| 050_0130 | 41 | 416–508 (93) | 286 | 26 (16) | 306 at the boundary; every large id present 93/93 | 0 |
| 010_0050 | 41 | 404–784 (381) | 1,250 | 170 (73) | 113,663 (`GEO-car_jack_custom_base_plate`) | 0 |
| 050_0160 | 41 | 404–680 (277) | 796 | 119 (72) | 33,798 (`GEO-index.3`) | 0 |
| 060_0100 | 41 | 404–568 (165) | 841 | 131 (84) | 77,074 (`GEO-einar_tire_knee`) | 0 |
| 060_0130 | 41 | 404–700 (297) | 524 | 54 (32) | 51,331 (`GEO-arm_base.L.002`) | 0 |
| 070_0123 | 41 | 404–524 (121) | 327 | 11 (6) | 8,164 (`GEO-electrical_wires_hand.mesh`) | 0 |
| 020_0020 | 41 (metadata) | 404–736 | — | NOT CENSUSED (Dense segmentation 4.3 GB > 3 GB cap) | — | — |
| 040_0040 | 41 (metadata) | 404–1056 | — | NOT CENSUSED (3.46 GB across two repos > cap; a train-only partial census would drop 16 test cameras and was refused) | — | — |

Every id-level event is a single named sub-mesh of a rig, character or
prop (bolts, finger segments, joints, batteries, a knee pad, a sweater
mesh, a jack base plate) whose gap is consistent with self-occlusion
during articulated motion; the largest ids in every scene (floor, walls,
shell panels, up to ~9.7 M px) are present in every frame. Under both
grouping rules almost every id falls into one `GEO`/`GEO-` supergroup
(96–100 % of ids; peak 61.7–72.0 M px) that never leaves all cameras, so
the grouping rules cannot separate a character from props or background
on this naming convention, and no group of any size has an event.

## 4. Verdict and its boundary

**No scene-level object or character that leaves every camera and
returns was found in 6 of 8 Charge scenes**, at the mesh level or under
name-prefix grouping. This is a negative about recoverability from the
released metadata, not a proof about the rendered frames: the
per-character identity lives in the Blender scene graph and is not
exposed in `segmentation.json`, so a whole-character exit in which some
sub-mesh stays visible cannot be distinguished from self-occlusion with
this data alone. For the paper (thesis claim 3, "no public ≥ 8-camera
capture supplies absence-and-return"), Charge can be cited as checked:
per-mesh-part segmentation, 6 scenes censused, zero scene-level events,
2 scenes not censused.

## 5. Not checked, and one method note

Sparse and Mono tasks; scenes 020_0020 and 040_0040; any RGB, depth,
normal or flow file; the supplementary per-scene table (arXiv carries no
ancillary files; the CVF PDF returned 403); the mask-generation
mechanism (Cryptomatte vs custom pass, inferred only). Method note
carried: two concurrent Hugging Face downloads failed mid-way with a
storage-backend 429 while reporting exit code 0 through a shell pipe;
completeness was established by counting files against the
pre-download metadata manifest, never by the exit code.
