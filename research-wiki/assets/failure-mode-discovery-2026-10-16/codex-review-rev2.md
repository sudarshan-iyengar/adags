## Verdict

MATERIAL DEFECT

The original narrow/wide-arc and label-conflation issues are resolved: [label code](/D:/adags-ars/fmd/obs_census.py:107) separates all five classes and counts remembered episodes correctly. Reaggregating the [CSV](/D:/adags/research-wiki/assets/failure-mode-discovery-2026-10-16/census_v2_summary.csv) reproduces the wide-arc row: **0.848% median, 1.677% maximum, 21,784 episodes ≥10 frames**. The automatic fallback is removed in §4.2, but commitment timing remains inconsistent.

## Confirmed defects

1. **CRITICAL — The held-render counterfactual cannot support its binary verdict.** “[Forgetting = good … bad … Never learned = bad at both](/D:/adags-ars/fmd/review/request_rev2.md:169)” overinterprets two scene-time queries of a final offline model. They establish time-dependent rendering quality, not whether training learned and subsequently lost a representation; incorrect foreground depth can produce the same pattern.

   There is also a concrete visibility failure: recomputing `ARS-L1-A0`, `c3a`, `cam00` with the supplied code finds **114 of 7,535 remembered pixel-frames** whose last training-visible time is occluded in the held-out camera. The code updates `last` from training visibility alone ([line 127](/D:/adags-ars/fmd/obs_census.py:127)). That historical render depicts another surface.

   **Fix:** require reference-time visibility in the evaluation camera, verify appearance invariance, and use temporal-support/foreground-depth interventions before naming the mechanism. Treat the held render as a conditional diagnostic.

2. **MAJOR — The three-camera “threshold” is unsupported.** “[It opens at 3 or fewer](/D:/adags-ars/fmd/review/request_rev2.md:95)” excludes four- and five-camera settings, but the CSV contains only **2, 3, 6, 19, 21** training cameras. Pixel shares also cannot establish that averaged error is “dominated by hallucination error.”

   **Fix:** test four/five cameras with multiple controlled subsets; measure classwise contributions to total squared error. Until then, report the tested configurations without a threshold claim.

3. **MAJOR — Required checks follow commitment.** §4.3 requires controls “[before the commit](/D:/adags-ars/fmd/review/request_rev2.md:277),” while the [schedule](/D:/adags-ars/fmd/review/request_rev2.md:321) puts replication, kill-argument and novelty checking on 24–28 October for the already “committed failure.” The unconditional “[dense failure is the paper](/D:/adags-ars/fmd/review/request_rev2.md:138)” also survives.

   **Fix:** make 24 October provisional selection; commit by 28 October only after all gates pass. Explicitly subordinate every fallback sentence to those gates.

4. **MAJOR — STG’s proposed protocol conflicts with “own published schedules.”** The [baseline table](/D:/adags-ars/fmd/review/request_rev2.md:300) specifies one model over the sequence. The [STG implementation](https://github.com/oppo-us-research/SpacetimeGaussians#training) documents 50-frame training; [Temporally Aware Densification, supplementary](https://arxiv.org/html/2606.23212v1) explicitly identifies STG’s 50-frame segment protocol.

   **Fix:** retain the published segmented baseline and identify a whole-sequence variant separately; account for segment boundaries in visibility history.

## Suspected issues

- **MAJOR — Pseudoreplication and age confounding.** Episodes are per sampled pixel, so thousands can represent neighboring pixels in a few occlusion events. Different age bins can contain different surfaces and events. Settle this with spatial/event-cluster bootstrapping and within-surface age comparisons.
- **MAJOR — Real-label precision remains unknown.** Synthetic validation plus one visual example cannot establish 90% precision on real footage. Require independently audited real histories, stratified by class and gap duration, with uncertainty and retained coverage.

## Unverified claims

- **MAJOR:** Remote inventory, allocation, runtimes, bank masses and gate gains were not independently verified from primary artifacts.
- **MAJOR:** Exhaustive novelty remains unestablished. PersistGS’s five-camera withheld-view evaluation is confirmed, but does not settle novelty.
- CVPR’s **10 November registration / 16 November submission** dates are confirmed by the [official page](https://cvpr.thecvf.com/Conferences/2027/Dates).

## Missing

- **MAJOR:** An explicit persistent/frozen-background control and intervention budget. [PersistGS §3](https://arxiv.org/html/2606.03479v1) already freezes reconstructed representations; the proposed freezing fix needs a concrete distinction.
- **MINOR:** Numerical definitions of “good,” “bad,” minimum exposure, and minimum independent events per age bin before promotion.