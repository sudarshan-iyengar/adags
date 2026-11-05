## Verdict

MATERIAL DEFECT

## Confirmed defects

1. **MAJOR — Moving-surface masks fail the self-visibility invariant.** The [self-check output](/D:/adags-ars/fmd/census_v3/smoke/summary.json:20) contains **21 before-only + 408 before-and-after = 429 false remembered pixel-frames**, although the evaluation camera is itself a training camera. The [code](/D:/adags-ars/fmd/obs_census_v3.py:243) assigns the nearest sample’s entire history. My direct ray check reproduced a mismatch at frame 0, cam10, pixel (404,924): the actual point is visible; its nearest sample is invisible.

   **Fix:** use exact surface-point histories for scored points, or exclude uncertain assignments after class-specific validation and sampling-convergence checks. “99.6% seen now” does not establish precision for rare remembered classes. Keep moving masks out of claim-bearing scores meanwhile.

2. **MAJOR — The common cut does not put every main subject in its published protocol.** “[Each at its own published setting](/D:/adags/research-wiki/operations/failure-mode-discovery-plan-2026-10-16.md:421)” conflicts with the four-camera main block. GC-4DGS’s paper specifies **three training videos and center-camera evaluation** ([§V-A](https://arxiv.org/html/2511.23044v1)). Its own-setting runs are budgeted only on synthetic scenes; Multi4D’s monocular configuration is already acknowledged as a transfer.

   **Fix:** label common-cut GC and Multi4D runs as protocol variants. Add GC’s native three-view real runs if claiming failure under its published protocol; otherwise restrict that claim to 4C4D. A different camera placement alone does not prove a method is outside its intended sparse-view regime.

3. **MAJOR — The advertised event-level inference is not implemented in census v3.** [Section 11](/D:/adags/research-wiki/operations/failure-mode-discovery-plan-2026-10-16.md:397) requires event clustering and ten independent events per age bin. The [output code](/D:/adags-ars/fmd/obs_census_v3.py:262) exports counts and label/age/reference images, without event identities or clustering. The reproduced **2,118 pixel episodes** cannot establish independent-event support.

   **Fix:** compute event components, eligible exposure histories, and counts per age bin before funding the full grid.

## Suspected issues

- **MAJOR — Starting-cloud leakage remains untested.** The training-only rule is an intention, not provenance evidence. 4C4D’s [loader](https://github.com/yangzf-1023/4C4D/blob/main/scene/dataset_readers.py#L271-L280) reuses an existing `points3D.ply`; rebuilding `points3D.bin` alone does not replace it. Settle this with manifests of the actual consumed clouds and depth files, their source cameras, and timestamps. No actual leakage is established here.

- **MAJOR — Control subtraction may retain geometry confounding.** “[General out-of-regime degradation hits both sets](/D:/adags/research-wiki/operations/failure-mode-discovery-plan-2026-10-16.md:448)” does not establish equal effects. Settle this with the promised matching, pre-trends, and interventions implemented for the new main subjects.

## Unverified claims

- The ~66 GPU-hour estimate, successful environment setup, preprocessing repairs, and runtime compatibility remain unmeasured. No submitted-run artifacts establish them.

## Missing

**Execution recommendation: GO WITH CHANGES**, in order: validate masks and event support; freeze per-run input/protocol manifests; then run one end-to-end smoke per main method before the full block.

Enforce the fixed-final-checkpoint policy in executable render commands. GC’s [published script](https://github.com/liyw420/GC-4DGS/blob/main/scripts/pre_dynerf/run_train_eval.sh#L84-L87) explicitly loads `chkpnt_best.pth`; the written policy alone does not prevent that selection leak.