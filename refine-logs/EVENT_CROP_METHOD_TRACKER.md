# Event-Crop Method Tracker

Generated: 2026-07-07

## Current Phase

Phase 2: define the non-oracle target and choose the first method candidate.

The R017 runtime opacity gate is closed as a failed actual-method check. Future methods must not use the frozen R009 event crops as test-time method inputs.

## Source Metrics

| Role | System | Mean PSNR | Mean L1/proxy-LPIPS | Mean flicker | Mean static ghost | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | route0 | 30.5021 | 0.0148316 | 0.00799083 | 0.127333 | `refine-logs/hide_reveal_poc/r010_route0_real_eval/real_event_window_summary.json` |
| baseline | matched_lifespan | 29.8181 | 0.0163546 | 0.00795601 | 0.127333 | `refine-logs/hide_reveal_poc/r012_r013_derived_real_eval/real_event_window_summary.json` |
| baseline | residual_uncertainty | 30.0734 | 0.0165723 | 0.00803902 | 0.145702 | `refine-logs/hide_reveal_poc/r011_residual_uncertainty_real_eval/real_event_window_summary.json` |
| upper_bound | derived oracle hide_reveal | 41.7149 | 0.00266536 | 0.00168586 | 0.127333 | `refine-logs/hide_reveal_poc/r012_r013_derived_real_eval/real_event_window_summary.json` |
| failed actual | R017 actual_hide_reveal | 19.3667 | 0.0761056 | 0.0162899 | 0.152789 | `refine-logs/hide_reveal_poc/r017_actual_real_eval/real_event_window_summary.json` |

Oracle upper-bound deltas versus route0:
- PSNR: `+11.2128`
- L1/proxy-LPIPS: `-0.0121662`
- Flicker: `-0.0063050`
- Static ghost: `0.0`

## Predeclared Evaluation Protocol

Evaluation split:
- Frozen R009 windows in `refine-logs/hide_reveal_real_windows.json`.
- Crops are evaluation-only and must not be consumed by the method as test-time event support.

Primary metrics:
- Per-window crop PSNR, higher is better.
- Per-window crop L1/proxy-LPIPS, lower is better.
- Per-window crop flicker, lower is better.
- Per-window static ghost score, lower is better.
- Accepted event count and false/broad event count if the method produces event candidates.
- Qualitative crop strips using the same five frozen windows.

Currently unavailable metrics:
- Learned LPIPS: unavailable unless sidecar weights are made available without compute-node network downloads.
- Confident-track identity switches: unavailable because R009 discovery found no track-confidence sidecars.

Strict PASS gate for a method:
- The method is checkpoint-backed or newly trained Gaussian-rendered output, not GT crop compositing.
- The method does not use R009 frozen event crops as test-time support.
- At least 3/5 frozen windows improve versus route0, matched_lifespan, and residual_uncertainty on both PSNR and L1/proxy-LPIPS.
- At least 3/5 frozen windows do not worsen static ghost versus route0.
- Mean PSNR improves over route0 by at least `+0.5 dB` and mean L1/proxy-LPIPS improves over route0 by at least `-0.001`.
- The method recovers at least 25% of the oracle upper bound on either mean PSNR improvement or mean L1/proxy-LPIPS reduction:
  - PSNR fraction: `(method_psnr - 30.5021) / 11.2128`
  - L1 fraction: `(0.0148316 - method_l1) / 0.0121662`

FAIL gate:
- A complete method run worsens mean PSNR and L1/proxy-LPIPS versus route0, or passes fewer than 3/5 windows, after logs and outputs are valid.
- A method requires oracle event-crop labels at test time to obtain its gain.
- A method cannot produce comparable Gaussian-rendered folders and only produces image composites.

## Candidate Methods

| Candidate | Mechanism | Why It Might Recover Oracle-Like Fix | Cost | Failure Modes | Required Data | Status |
| --- | --- | --- | --- | --- | --- | --- |
| M1 non-oracle residual-component local refinement | Detect candidate event supports from high residual, dynamic masks, flow validity/disagreement, and local flicker in training/eval render diagnostics; select connected components without R009 crop labels; locally optimize a small set of Gaussian color/opacity/visibility parameters under a fixed budget; render normally. | R013 says the error is local and large. R017 failed by subtracting content; M1 can add/refine local appearance while preserving route0 elsewhere. | Medium. Reuses route0 checkpoints and renderer; needs candidate-map generation, local optimization CLI, Slurm wrapper, metadata. | Detector selects easy residuals or wrong regions; local fitting overfits observed view; static ghost rises; output becomes crop-like if support leaks from evaluation labels. | Route0 renders/GT/static/dynamic, masks, flow, checkpoints. | R021 timed out before checkpoints; R022 replacement train jobs pending. |
| M2 occlusion-boundary gated micro-densification | Use dynamic-mask boundaries and flow occlusion/disocclusion cues to seed a small event-local set of Gaussians, with strict budget and no use of frozen crop labels. | The oracle fix may require new or sharper local capacity, not just opacity changes. | Medium-high. Requires densification/update path and budget accounting. | Novelty pressure from visibility-aware densification; may become known densification rather than identity event; may fragment identity. | Masks, flow, checkpoints, training images. | Backup. |
| M3 temporal inconsistency event proposal plus conservative visibility gate | Build non-oracle event candidates from route0 render flicker/residual over time; apply a much narrower, component-local gate than R017 and log selected Gaussian counts. | R017 selected too many Gaussians. A component-local proposal may avoid broad content deletion. | Low-medium. Reuses R017 renderer hook with non-oracle support maps. | Still only removes content; likely cannot synthesize revealed texture; may repeat R017 failure with smaller damage. | Route0 renders/GT for candidate construction, masks. | Backup or diagnostic. |
| M4 identity-aware reveal matching with tracks/features | If reliable track/feature sidecars can be generated, commit hide/reveal only when hidden identity evidence reconnects across the event; train/refine selected carriers. | Aligns with original novelty boundary against lifespan-only gating. | High. Track sidecars were absent in R009; generation may be a separate project. | Blocked by unavailable tracks; noisy tracks around occlusions; high implementation burden. | Confident tracks/features, checkpoints, masks/flow. | Deferred. |

## Selected First Candidate

Tentative first candidate: M1 non-oracle residual-component local refinement.

Reason:
- It directly addresses the R017 failure mode: opacity attenuation removed/dimmed content but did not synthesize the hidden/revealed surface.
- It is non-oracle if candidate components are generated from residual/mask/flow cues without reading R009 crop labels.
- It can produce actual Gaussian-rendered output folders if implemented as checkpoint-backed local refinement.
- It has a clear fail interpretation: if local Gaussian refinement cannot beat route0 on the frozen windows, the oracle crop fix may require stronger geometry/identity machinery rather than simple local updates.

Before full local-refinement implementation:
- Inspect existing train/resume/render code for the smallest local-refinement entry point.
- Verify candidate-map inputs exist on HPC for the three scenes with the R018 dry-run output.
- Use the A1 dry-run output to choose whether M1 has plausible event support before writing any checkpoint-updating code.
- Record the exact command and output directory before submitting any Slurm job.

## Attempt Log

| Attempt | Candidate | Commit | Job ID | Output | Verdict | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Recovery | Evidence/state recovery | `0b18166bec6d1a2d371764c70bdcf53b23319a5e` | n/a | `refine-logs/EVENT_CROP_FIX_EVIDENCE.md`, `research-wiki/event-crop-fix.md` | COMPLETE | Preserved R001-R017 evidence, oracle upper bound, R017 failure, frozen-window paths, and predeclared non-oracle evaluation gate. |
| A0 | R017 runtime opacity gate | `f5d43539aee500051f2a4c5eeca5420293b636f1` | `48760029`, `48760448` | `refine-logs/hide_reveal_poc/r017_actual_real_eval/` | FAIL | Checkpoint-backed renderer output, no GT pixels, but 0/5 windows passed and all mean metrics worsened. |
| A1 | Non-oracle event candidate discovery dry-run | `0bf0967483e622af5cb6ac81de2b3f09060c33d9` | local smoke | `refine-logs/hide_reveal_poc/local_smoke/nonoracle_candidates/out/` | SMOKE PASS | New `nonoracle-candidates` stage produced 3 valid smoke candidates with `validation_ok=True`, `uses_gt_residual=false`, and `uses_frozen_window_labels=false`; not yet a Gaussian-rendered method result. |
| R018 | A1 candidate discovery on real scene sources | `db467a328b6b0e02482eb0e36de24b27b850b907` | `48763378` | `refine-logs/hide_reveal_poc/r018_nonoracle_candidates/` | DETECTOR FAIL | Slurm job completed and validation passed with 24 candidates, but posthoc frozen-overlap audit covered 0/5 windows; selected boxes clustered on top image bands with zero motion-mask support. |
| A2 | Motion-supported candidate discovery dry-run | `f69034be1ca32ddcd24756d945ead467d59e3c24` | local smoke | `refine-logs/hide_reveal_poc/local_smoke/nonoracle_candidates/out_motion_supported/` | SMOKE PASS | Revised detector multiplies dynamic/static-delta/flicker evidence by motion-mask support when masks are available; local smoke still selects the synthetic moving-square support. |
| R019 | A2 motion-supported candidate discovery | `a46c79678150881f6c3fc50e074e7a4de100a9bc` | `48763799` | `refine-logs/hide_reveal_poc/r019_motion_supported_nonoracle_candidates/` | PARTIAL | Validation passed with 24 candidates and top-band artifact removed; posthoc frozen-overlap audit covered 2/5 windows, so M1 support remains insufficient. |
| R020 | A2 high-recall motion-supported candidate pool | `a46c79678150881f6c3fc50e074e7a4de100a9bc` | `48764048` | `refine-logs/hide_reveal_poc/r020_high_recall_motion_supported_nonoracle_candidates/` | SUPPORT POOL | Validation passed with 72 candidates using `HIDE_REVEAL_NONORACLE_TOP_K_PER_SCENE=24`; posthoc frozen-overlap audit covered 3/5 windows. This is candidate support only, not a rendered-method pass. |
| R021 | Candidate-local refinement train jobs | `5a3ded1d338dc7534317434907835c0de4da0e73` | `48764715`, `48764716`, `48764718` | `refine-logs/event_candidate_refine_train_jobs_20260707_033133.tsv` | RUNNING / MONITOR BLOCKED | Train jobs submitted for cut_roasted_beef, flame_steak, and sear_steak; startup logs looked normal, but later SSH auth rejection blocked monitoring and collection. Eval/scoring not yet run. |
| R021b | Leonardo runner extension-dir fix | `7d848fc` | n/a | `scripts/run_leonardo.sh` | COMPLETE | Added per-job `TORCH_EXTENSIONS_DIR`, `TORCH_CUDA_ARCH_LIST=8.0`, and `MAX_JOBS` after R021 timed out at shared PyTorch extension startup without checkpoints. |
| R021c | Real-manifest augmentation helper | `657f0cd201dda6c84c0b4442e53380e4d837f1ad` | n/a | `scripts/run_hide_reveal_poc.py augment-real-manifest-system` | COMPLETE | Adds reusable command to attach future refined eval folders to frozen windows and merge baseline manifests before scoring. |
| R022 | Candidate-local refinement replacement train jobs | `657f0cd201dda6c84c0b4442e53380e4d837f1ad` | `48796168`, `48796170`, `48796174` | `refine-logs/event_candidate_refine_train_jobs_20260707_110953.tsv` | PENDING | Replacement train jobs submitted with per-job torch extension dirs and `TIME=02:00:00`; pending on scheduler priority at last poll. |
