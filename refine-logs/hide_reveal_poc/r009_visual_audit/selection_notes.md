# R009 Real-Window Selection Notes

Frozen at: 2026-07-06T00:20:08Z

## Policy

- Selected before any real-window scoring.
- Used sparse GT/render visual audit only.
- Did not use LPIPS, PSNR, flicker, static ghost, or hide/reveal scores to choose windows.
- Chose five 16-frame inclusive windows, within the R009 cap of 4-6 windows.
- Excluded `sear_steak_w02` because the spoon/steak geometry was too static relative to `sear_steak_w01`.

## Selected Windows

| Window | Frames | Crop | Reason |
| --- | --- | --- | --- |
| `cut_roasted_beef_hand_tongs_meat_095_110` | 95-110 | `[245, 315, 455, 470]` | Tongs and knife occlude/reveal sliced meat on the cutting board. |
| `cut_roasted_beef_hand_knife_meat_140_155` | 140-155 | `[245, 315, 455, 470]` | Hand/knife sweep over sliced meat with visible reveal. |
| `flame_steak_torch_pan_155_170` | 155-170 | `[250, 300, 445, 460]` | Torch flame covers steak/pan region and changes rapidly. |
| `flame_steak_torch_sweep_195_210` | 195-210 | `[250, 300, 445, 460]` | Torch flame sweeps across and reveals the steak/pan region. |
| `sear_steak_spoon_pan_220_235` | 220-235 | `[245, 320, 450, 470]` | Spoon crosses over the steak surface with enough reveal motion. |

## Source Availability

- Exact route0 evals are complete for `cut_roasted_beef`, `flame_steak`, and `sear_steak`: 300 `renders`, 300 `gt`, 300 `static`, and 300 `dynamic` frames per scene at 676x507.
- Scene masks exist at `/leonardo_work/EUHPC_D21_034/proj_adags/data/n3v/<scene>/motion_priors/masks`: 300 PNGs per scene at 1352x1014.
- Flow sidecars exist at `/leonardo_work/EUHPC_D21_034/proj_adags/data/n3v/<scene>/flow`: `cut_roasted_beef` has 5980 NPZs; `flame_steak` and `sear_steak` have 6279 NPZs each. Sample NPZ keys are `flow.npy` and `mask.npy`.
- No track or confidence sidecar was found under the exact route0 evals or the scene data sidecars during R009 discovery.

## Evidence Artifacts

- Broad scene sheets: `contact_sheets/`.
- Cropped candidate sheets: `candidate_crop_sheets/`.
- Canonical frozen manifest: `refine-logs/hide_reveal_real_windows.json`.
