# N3V baseline registry (2026-08-20)

Operational record, static curation tier: every row traces to a tracked
file; nothing is inferred. Produced by a bounded read-only curation
worker during the 2026-08-20 block and reviewed by the primary before
promotion. Two of its recommendations were EXECUTED the same block and
are recorded here as done rather than pending:

- the PSNR pooling repair landed (commit `7ac4238`, tests
  `tests/test_psnr_pooling.py`), and
- the `--val` pass over experiment 181's checkpoint ran as experiment
  194: pooled+clamped **33.5050 dB / SSIM 0.95934 / LPIPS-alex(norm)
  0.08136** at 1352x1014 -- see
  [[stg-n3v-protocol-parity-2026-08-19]] Appendix C.

The registry text below is otherwise as curated.

---
# Lane A canonical baseline registry â€” ADAGS N3V historical runs

Read-only curation pass, 2026-08-20, over `D:\adags` @ branch
`apollo/csvl-vpl-v2-exploratory`. Every row below is traced to a tracked file
in this repository; anything not traceable is marked **UNRECOVERABLE FROM
REPO** rather than inferred.

---

## 0. Protocol constants that apply to EVERY N3V row

Verified directly in source this pass:

| item | value | source |
|---|---|---|
| on-disk raster | **1352x1014** (native 2704x2028 halved once) | `scripts/n3v2blender.py:273` (`img.resize((w//2, h//2))`); confirmed by opening `data/cut_roasted_beef/images/cam00_0000.png` -> `(1352, 1014)` |
| loader rescale | `resolution = round(orig_w/(scale*args.resolution))`, no auto-cap for values in `[1,2,3,4,8]` | `utils/camera_utils.py:19-23` |
| **eval raster at `resolution: 2`** | **676x507** | composition of the two above |
| **eval raster at `resolution: 1`** | **1352x1014** | same |
| frame extraction | `ffmpeg -i <video> -t 10 -start_number 0` -> 300 PNGs/cam | `scripts/n3v2blender.py:266` (STG uses OpenCV â€” pixel-level equality **never byte-verified**) |
| cameras present | 20 (`cam00..cam20`, **`cam04` absent**) | `ls data/cut_roasted_beef` |
| held-out camera | **`cam00`**, chosen POSITIONALLY (`if i == 0: test_frames`), 19 train cams | `scripts/n3v2blender.py:341-349`; N3V loads via `readNerfSyntheticInfo` -> `transforms_train/test.json` |
| timestamp | `t = frame_index / 30.0`; `time_duration: [0.0, 10.0]` = frames 0-299 | `scripts/n3v2blender.py:344` |
| initialization | `data/<scene>/points3d.ply`, **366,366 points** (frame-0 COLMAP + dense MVS, no time channel) | ply header read this pass; `scene/dataset_readers.py:481-499` |
| `num_pts: 300_0000` | = 3,000,000, a SUBSAMPLE CAP that **never binds** (366,366 < 3e6) | `dataset_readers.py:497` |
| `num_pts_ratio: 1.0`, `num_extra_pts: 0` | no random point injection | all configs |
| seed | `main.py --seed`, default **6666**; `setup_seed` sets torch/cuda/np/random | `main.py:2070, 2041-2045, 2112` |
| **PSNR (both eval sites)** | **channel-split** â€” `psnr()` does `.view(img.shape[0], -1).mean(1)` on an unbatched `(3,H,W)`, returns 3 per-channel PSNRs, caller `.mean()`s them. Bias `10*log10(AM/GM) >= 0`, **measured 0.268 dB** on the LRV1 A0 run | `utils/image_utils.py:17-19`; call sites `main.py:1979`, `utils/mesh_utils.py:106` |
| SSIM | `utils.loss_utils.ssim`, 3DGS Gaussian-window variant | `main.py:1980` |
| **LPIPS** | torchmetrics `LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)` â€” i.e. inputs in [0,1] mapped internally to [-1,1]. STG/3DGS use raw [0,1] and the measured gap on identical images is **0.14685 vs 0.12398** | `utils/mesh_utils.py:65-66`; gap from [[diva360-protocol-parity-audit]] |
| LPIPS availability | **only on the `--val` pass**, never in the training-time test eval | `main.py:1949-2040` computes psnr/ssim only |

### Two DIFFERENT eval paths produce differently-named numbers

| path | trigger | clamp? | metrics emitted |
|---|---|---|---|
| **training-time test eval** | `--test_iterations` during training; best snapshot written as `best_val/*` | **NO clamp** on `pred` | `test/psnr`, `test/ssim`, `test/dynamic_mask_psnr`, `test/static_region_psnr`, `test/static_ghost_score`, `test/dynamic_edge_magnitude`, `test/track_flow_l1` (`main.py:2016-2031`) |
| **`--val` pass** (`main.py:1034` `validation()`) | second `main.py` invocation with `--val --start_checkpoint chkpntN.pth` | **YES**, `torch.clamp(rgb,0,1)` | `psnr`, `ssim`, **`lpips`**, `num_GS`, `static` (`utils/mesh_utils.py:56-120`) + `dynamic_mask_psnr`/`static_region_psnr` re-derived by `evaluate_motion_prior_test_metrics` |

**This matters for Lane A**: the survey CSV carries `mean_lpips` AND
`mean_num_GS`, which only the `--val` path produces together. Exact match
confirms it: CSV `csvl_vpl_v2_exploratory` = `34.4785 / 0.9622 / 0.0500 /
593657` is byte-for-byte lane **L5** of the `--val` table in
[[operations/csvl-vpl-v2-exploratory-round1-results]]. So **the survey is
`--val`-protocol (clamped)**, while **experiment 181's 33.5210 is
`best_val/psnr` from the unclamped training-time path** â€” an unmatched axis on
top of raster/batch.

---

## 1. Registry â€” top methods by mean PSNR

All rows: `gaussian_dim: 4`, `rot_4d: False`, `force_sh_3d: True`,
`sh_degree: 3`, `eval_shfs_4d: True`, `enable_soft_routing: true`,
`motion_model: "lora"`, `lambda_dssim: 0.2`, `densify_from_iter: 500`,
`densification_interval: 100`, `opacity_reset_interval: 30000` (i.e. never
fires inside 6k/15k), `percent_dense: 0.01`, unless noted.

| # | method (CSV) | PSNR / scenes | config file | res | **eval raster** | iters | batch | pt cap | time_duration (frames) | route_logit_init | family switches |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `scaffold_lora_route0_dyn_densify` | 34.587 / 1 (sear_steak) | `configs/n3v/scaffold_lora_route0_dyn_densify.yaml` | 2 | **676x507** | 9,000 | 4 | **-1 (uncapped)**; `densify_until_iter 6000` | [0,10] = 0-299 | 0.0 | scaffold ON (512 nodes, r8/a32, knn4), `enable_motion_aware_densify: true` boost 1.5, `lambda_scaffold_smooth 1e-4`, `lambda_scaffold_reg 1e-6`, blur OFF |
| 2 | `fixed_budget_scaffold_lora_route0_noreg_800k` | 34.482 / 2 (crb, flame_steak) | `configs/n3v/fixed_budget_scaffold_lora_route0_noreg_800k.yaml` | 2 | **676x507** | 6,000 | 4 | 800,000 | [0,10] | 0.0 | scaffold ON, **both scaffold lambdas 0.0**, motion-aware densify OFF, blur OFF |
| 3 | `csvl_vpl_v2_exploratory` | 34.4785 / 1 (crb) | **`configs/n3v/lane_l5_shifted.yaml`** â€” see Â§3, this row is the **time-shifted misaligned-evidence CONTROL (L5)**, not a baseline | 2 | **676x507** | 6,000 | 4 | 600,000 | [0,10] | 0.0 | full E1/E2 lifecycle with deliberately WRONG-TIME evidence |
| 4 | `visibility_event_smooth_control_6000` | 34.335 / 3 | `configs/n3v/visibility_event_smooth_control_6000.yaml` | 2 | **676x507** | 6,000 | 4 | 600,000 | [0,10] | 0.0 | frozen R020 event manifest (`event_candidate_manifest`, dilate 4), motion-aware densify boost 2.0 + `use_residual: true` |
| 5 | `..._filemask_residual_flow_static_anchor_rendergate_600k` | 34.245 / 2 | **NO CONFIG IN REPO** | â€” | â€” | â€” | â€” | â€” | â€” | â€” | UNRECOVERABLE FROM REPO |
| 6 | `..._filemask_residual_flow_coremask_ramp_rendergate_600k` | 34.206 / 3 | **NO CONFIG IN REPO** | â€” | â€” | â€” | â€” | â€” | â€” | â€” | UNRECOVERABLE FROM REPO |
| 7 | `fixed_budget_lora_route0_800k` | 34.180 / 2 | `configs/n3v/fixed_budget_lora_route0_800k.yaml` | 2 | **676x507** | 6,000 | 4 | 800,000 | [0,10] | 0.0 | scaffold OFF, motion-aware densify OFF, blur OFF, dyn-mask-from-residual ON |
| 8 | `phase2_lora_r16_a32_600k` | 34.143 / 3 | **NO CONFIG IN REPO** (nearest sibling `lora_r16_a32.yaml` is 15k/uncapped/blur, NOT 600k) | â€” | â€” | â€” | â€” | â€” | â€” | â€” | UNRECOVERABLE FROM REPO |
| 9 | `phase2_lora_r8_a32_coeff00032_basis00004_600k` | 34.133 / 3 | **NO CONFIG IN REPO** | â€” | â€” | â€” | â€” | â€” | â€” | â€” | UNRECOVERABLE FROM REPO |
| 10 | `..._filemask_residual_flow_boundary_ring_rendergate_600k` | 34.128 / 3 | **NO CONFIG IN REPO** | â€” | â€” | â€” | â€” | â€” | â€” | â€” | UNRECOVERABLE FROM REPO |
| 11 | `fixed_budget_lora_route0_dyn_600k` | 34.119 / 3 | `configs/n3v/fixed_budget_lora_route0_dyn_600k.yaml` | 2 | **676x507** | 6,000 | 4 | 600,000 | [0,10] | 0.0 | scaffold OFF, **motion-aware densify ON** |
| 12 | `visibility_event_train_6000` | 34.038 / 3 | `configs/n3v/visibility_event_train_6000.yaml` | 2 | **676x507** | 6,000 | 4 | 600,000 | [0,10] | 0.0 | visibility-event opacity attenuation lane |
| 13 | **`fixed_budget_lora_route0_600k`** | 34.025 / 3 | **`configs/n3v/fixed_budget_lora_route0_600k.yaml`** (SHA-256 `b7372f3fâ€¦6eb3f`, the *declared* canonical N3V baseline) | 2 | **676x507** | 6,000 | 4 | 600,000 | [0,10] | 0.0 | plain LoRA route0; scaffold OFF, densify OFF, blur OFF |
| â€” | `fixed_budget_lora_route0_filemask_residual_600k` (== `lane_l0_route0`) | 33.722 / 3 | `configs/n3v/fixed_budget_lora_route0_filemask_residual_600k.yaml`; `lane_l0_route0.yaml` differs **only in comments** | 2 | **676x507** | 6,000 | 4 | 600,000 | [0,10] | 0.0 | identical switch set to #13; file-mask + residual fallback |
| 34 | **`lora_route0_dynmask`** (6-scene leader) | **32.762 / 6** | `configs/n3v/lora_route0_dynmask.yaml` | 2 | **676x507** | **15,000** | 4 | **-1 (uncapped)**; `densify_until_iter 12000` | [0,10] | 0.0 | **`blur_until_iter: 9000`, `blur_start_sigma: 8.0` (blur curriculum ON)**; scaffold OFF; dyn-mask ROI 0.5 / static-excl 0.02 |
| 35 | `lora_r8_a32_route0` | 32.730 / 6 | `configs/n3v/lora_r8_a32_route0.yaml` | 2 | **676x507** | 15,000 | 4 | -1 | [0,10] | 0.0 | blur 9000; no dynamic-mask ROI keys |
| 40 | `scaffold_lora_route0_dyn_densify_ptbudget` | 32.524 / 6 | `configs/n3v/scaffold_lora_route0_dyn_densify_ptbudget.yaml` | 2 | **676x507** | 6,000 | 4 | 800,000 | [0,10] | 0.0 | scaffold ON + motion-aware densify ON, blur OFF |
| â€” | **`b1_stg_matched_crb` (exp 181)** | **33.5210** (channel-split) / 1 (crb) | `configs/n3v/b1_stg_matched_cut_roasted_beef.yaml` | **1** | **1352x1014** | 6,000 | **2** | 600,000 | **[0, 1.6340] = frames 0-49** | 0.0 | identical to `lane_l0_route0` otherwise (verified by diff) |

### `b1_stg_matched_cut_roasted_beef.yaml` vs `lane_l0_route0.yaml` â€” exact diff

Comment-stripped `diff` run this pass returns **exactly three lines**:

```
time_duration: [0.0, 10.0]  ->  [0.0, 1.6340]
batch_size: 4               ->  2
resolution: 2               ->  1
```

Nothing else differs. This is the cleanest historical-to-matched mapping in
the repo.

---

## 2. Provenance recoverable for experiment 181 (the 181 protocol)

From `agent-control/elgs-apollo/claims/b1_stg_matched_crb__r2.json` and
`experiment-ledger.jsonl` (line 121):

| field | value |
|---|---|
| experiment id | **181** (retry 2; r0=166 cancelled, r1=169 OOM) |
| commit | **`456f4d6d6fb1ebe3962231e0b8fa338224873a90`** |
| **seed** | **0** (encoded in `run_dir` per `_build_run_id`: `..._b1_stg_matched_crb_0_456f4d6`) |
| pool | `dgx` (V100-SXM2 32 GB) |
| run_dir | `/apollo/users/sri/proj_adags/runs/elgs/20260819T020504Z_b1_stg_matched_crb_0_456f4d6` |
| `evidence_bearing` | **false** (exploratory) |
| eval schedule | `--test_iterations 6000` only (stated in the parity page Appendix A: "the first and only evaluation would have run at 6000") |
| result | `best_val/psnr` 33.5210, ssim 0.95934, `dynamic_mask_psnr` 25.3561, `static_region_psnr` 33.6781, 599,342 primitives, `best_val_iter` 6000 |
| **no LPIPS** | consistent with the training-time path; **no `--val` pass was run** |
| wall time | 2h43m; r1 at batch 4 OOM'd at iter 3510 with 28.98 GiB reserved on 31.74 GiB |

Image digest for exp 181 is recorded as "the admitted digest" in the handover
but the literal `sha256:` string is **UNRECOVERABLE FROM REPO** (the ledger
line does not carry it).

## 3. Provenance for the CSVL-VPL v2 exploratory lanes (Leonardo/Slurm)

From `research-wiki/operations/csvl-vpl-v2-exploratory-round1-results.md` and
`scripts/submit_exploratory_lane.sh`:

- branch `csvl-vpl-v2-exploratory`; contract at `932b32b`, corrected `e584ea3`
- **seed 0** (`SEED="${SEED:-0}"` in the launcher), scene `cut_roasted_beef`
- `--save_iterations 1000 3000 6000`, `--test_iterations 1000 2000 3000 4500 6000`
- then a second invocation `--val --start_checkpoint <run>/chkpnt6000.pth`
- Slurm job ids (verified against sacct in the wiki page): L0 `50896779`,
  L1 `50896788`, L2 `50896801`, L3 `50896810`, L4 `50896816`, L5 `50896823`
- `--val` results table (676x507, clamped, channel-split PSNR, LPIPS alex/normalize=True):

| Lane | config | PSNR | SSIM | LPIPS | Gaussians |
|---|---|---:|---:|---:|---:|
| L0 baseline | `lane_l0_route0.yaml` | **34.366** | 0.9605 | 0.0524 | 541,662 |
| L1 | `lane_l1_internal.yaml` | 34.231 | 0.9613 | 0.0498 | 592,209 |
| L2 | `lane_l2_presence_vad.yaml` | 34.399 | 0.9628 | 0.0497 | 599,571 |
| L3 | `lane_l3_full.yaml` | 34.306 | 0.9608 | 0.0517 | 591,774 |
| L4 | `lane_l4_generic_capacity.yaml` | 34.020 | 0.9613 | 0.0514 | 593,336 |
| **L5 (shifted/wrong-time control)** | `lane_l5_shifted.yaml` | **34.479** | 0.9622 | 0.0500 | 593,657 |

The CSV row `csvl_vpl_v2_exploratory` (34.4785 / 0.9622 / 0.0500 / 593657) is
**L5**. Peak CUDA at 676x507 / batch 4 was **12.8-12.9 GB**.

## 4. Dynamic-mask / static-region metrics â€” do they exist?

**Yes, but not in the survey CSV and not in any local artifact.**

- The metric keys exist and are computed whenever a `MotionPriorCache` yields a
  dynamic mask: `test/dynamic_mask_psnr`, `test/static_region_psnr`,
  `test/static_ghost_score`, `test/dynamic_edge_magnitude`
  (`main.py:1982-2007`, `main.py:158-205`). All top configs set
  `dynamic_mask_from_residual: true`, so they were produced.
- They are enumerated as harvestable **W&B summary keys** in
  `scripts/analyze_fixed_budget_wandb.py:36-58` â€” so for the whole
  `fixed_budget_*` family they live in W&B, which is a network resource and
  was **not** queried (prohibited this pass).
- `runs-metrics-survey-2026-08-19-full.csv` carries only
  psnr/ssim/lpips/num_GS. **No dyn/static column exists in it.**
- Locally recorded values exist only where a wiki page transcribed them:
  - CSVL-VPL v2 round 1, `training_report` protocol, all six lanes:
    static-region PSNR L0 34.827 / L1 34.693 / L2 34.848 / L3 34.768 /
    L4 34.437 / L5 34.949; dynamic-mask PSNR L0 25.320 / L1 25.713 / L2 26.037
    / L3 25.752 / L4 25.655 / L5 25.816; static ghost L1 0.0950 best.
  - Experiment 181: dyn 25.3561, static-region 33.6781.
- `find` over the repo returns **no** `training_report`/`stats` JSON for any
  N3V run. Only `refine-logs/**/real_event_window_summary.json` (event-crop
  protocol) exist locally.

---

## 5. What CANNOT be recovered from this repository

1. **Five of the top-10 CSV method names have no config and no mention
   anywhere in the working tree**: `phase2_lora_r16_a32_600k`,
   `phase2_lora_r8_a32_coeff00032_basis00004_600k`, and the three
   `..._rendergate_600k` variants (`static_anchor`, `coremask_ramp`,
   `boundary_ring`), plus `..._softborder_rendergate_600k`. A repo-wide grep
   hits only the CSV itself and the git pack â€” i.e. they exist in history but
   not on this branch. **UNRECOVERABLE FROM REPO as configs.** (They may be
   recoverable from git history; that was not attempted, being outside a
   read-only-file scope.)
2. **The survey generator.** No script in `scripts/` or anywhere produces
   `runs-metrics-survey-2026-08-19-full.csv`; `grep` for the filename returns
   nothing. Which W&B summary key each `mean_*` column came from, and at which
   iteration, is **inferred** (from the exact L5 match) rather than verified.
3. **Commit, seed, job id, and W&B run id for every non-CSVL-VPL, non-Apollo
   N3V run** â€” i.e. rows 1, 2, 4, 7, 11, 12, 13, 34, 35, 40. The
   `agent-control/elgs-apollo/` ledger covers only the ELGS/DiVa360/LRV/B1
   Apollo cells. `refine-logs/*.tsv` carry job ids for the event-refinement
   chains only. Two run-id sets survive in
   `research-wiki/baselines/phase7-next-phase-baseline.md`:
   `20260701_012706/012711/012714_*_fixed_budget_lora_route0_600k` and
   `20260619_184247_*_fixed_budget_lora_route0_filemask_residual_600k` â€” but
   with **no commit and no seed**.
4. **Image digest for exp 181** (ledger records commit and pool only).
5. **Whether ADAGS's ffmpeg-extracted PNGs are pixel-identical to STG's
   OpenCV-extracted ones.** The parity page flags this as a one-off byte-diff
   that was never performed.
6. **Any measured seed spread** at any of these configurations. Every recorded
   number is single-seed.

---

## 6. Cross-protocol number collisions to avoid

Two irreconcilable "route0 baseline" numbers exist for the *same* config
family and were never reconciled (parity page Â§6 says so explicitly):

- **34.025 dB** (3-scene mean, `fixed_budget_lora_route0_600k`, survey CSV,
  full-frame `--val` at 676x507)
- **30.5021 dB** (3-scene "route0" mean quoted in
  `research-wiki/baselines/phase7-next-phase-baseline.md` and the R036/R037
  gate) â€” this is the **event-window crop** protocol from
  `refine-logs/hide_reveal_poc/`, not full-frame.
- **34.366 dB** (L0, single scene `cut_roasted_beef`)
- **34.37 dB** cited in `sota-sweep-2026-08` as "inside the competitive band"
  vs STG's 33.52 â€” **retracted** by
  `research-wiki/operations/stg-n3v-protocol-parity-2026-08-19.md` Â§1.

Do not mix these in one table.

---

## 7. Recommendation

**`configs/n3v/b1_stg_matched_cut_roasted_beef.yaml` is already the correct
matched-raster cell, and its parent `configs/n3v/lane_l0_route0.yaml` (==
`fixed_budget_lora_route0_filemask_residual_600k.yaml`, == the declared
canonical `fixed_budget_lora_route0_600k` switch set) is the strongest
CREDIBLE substrate family.** Rationale:

- It is the only config family whose historical number, matched-raster
  number, per-lane controls, seed, commit, job ids, and activation checks are
  ALL recoverable from tracked files.
- The four rows above it in the CSV are each disqualified as a *baseline*:
  - `csvl_vpl_v2_exploratory` (34.4785) **is the wrong-time evidence control
    L5** â€” a deliberately misaligned lane. Reporting it as a baseline would be
    a serious error.
  - `scaffold_lora_route0_dyn_densify` (34.587) is **one scene only
    (sear_steak)**, uncapped Gaussians (1.24 M final â€” 2x the 600k family),
    and 9,000 iterations. Not budget-comparable and not measured on
    `cut_roasted_beef`.
  - `fixed_budget_scaffold_lora_route0_noreg_800k` (34.482) and
    `fixed_budget_lora_route0_800k` (34.180) sit at an **800k cap**, a
    different budget from every 600k row and from the 181 protocol.
  - `visibility_event_smooth_control_6000` (34.335) depends on a frozen R020
    event manifest under `refine-logs/`, i.e. an external artifact.
- `lora_route0_dynmask`, the 6-scene leader, is a **different budget
  entirely**: 15,000 iterations, uncapped Gaussians, and a **Gaussian blur
  curriculum to iteration 9,000**. It is not a drop-in substitute for the
  6k/600k family and the blur curriculum is on the project's own
  "failed or weak ideas to preserve" list.

**If a stronger-substrate full-raster run is wanted**, the minimal credible
escalations from the B1 cell, in order of defensibility:

1. `resolution: 1` + `batch_size: 4` (ADAGS's own published batch) â€” **needs
   >32 GB**; this is exactly what OOM'd at 3510/6000 on V100. Requires
   `hopper` (H100 80 GB).
2. `resolution: 1`, `batch_size: 2`, cap raised 600k -> 800k, matching the
   `fixed_budget_*_800k` family â€” but memory scales with points too and this
   was never attempted at full raster.
3. Add a `--val` pass so LPIPS and clamped PSNR exist and the number is on the
   same protocol as every historical CSV row.

**Before any of these, two one-line source fixes are free and change the
headline**: pool PSNR correctly (`.unsqueeze(0)` at `main.py:1979` and
`utils/mesh_utils.py:106`, worth **-0.268 dB**) and report LPIPS under both
input conventions.

