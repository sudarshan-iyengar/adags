## Venue facts (verified 2026-09-21 from the CFP page)

NeurIPS 2026 style; double-blind via OpenReview; long papers up to 8 pages;
non-archival window Sep 29 – Oct 29 (portal "will be open soon"); concurrent
submission permitted if other venues' policies allow. Ambiguity: "(Reference and
Appendix are not included)" is attached to the 4-page extended-abstract line;
confirm it also applies to 8-page papers.

## The story the evidence points to (a hypothesis, not a title)

What exists: temporal opacity (STG) renders the authored gap 3.4–7.5 dB better
than our gate; our gate returns 1.3–2.5 dB better than STG on 7/8 prefixes
(unpaired); on a real occlusion the gate is inert. Missing: what published methods
do on these fixtures. If they also render the gap well but return poorly, the
paper is: (1) an absence-and-return benchmark, (2) the finding that published
methods fail on the return, (3) a presence mechanism that improves a published
method on its own code, (4) a post-training demonstration of which primitives come
back. That is the "beats the benchmark" story. The first two weeks test whether it
holds.

## Menu decisions

Do:
1. Baselines on all three fixtures at their published settings, one evaluator:
   STG on cut_roasted_beef (flame/sear done), Ex4DGS and Multi4D (~1.2 / 2.7
   GPU-h per run, n=3 per scene). Converters are simpler than they look: only
   frames 60–89 are edited, so frame 0 and its point cloud are unchanged — Ex4DGS's
   `colmap_0` from the reproduction is reusable on crb. Multi4D needs `fused.ply`
   per scene; crb exists; flame/sear unverified.
2. Presence notch on Ex4DGS — base vs base+notch, 6 runs per arm per scene
   (~43 GPU-h). Ordered lifespan-minus-notch form, q=0 exact nesting.
3. Contribution-mass attribution via Ex4DGS's auxiliary flow-VJP channel —
   which primitives paint the object before and after the gap. Evaluation code
   only; the "demonstrate after training" item.
4. Honesty conditions: independent SAM 2 masks, every P1/P2/S1 frame
   human-audited, frozen reducer re-run unchanged.

Stretch only if Gate 1 shows Multi4D returns badly: gate on Multi4D's persistent
foreground bank.

Skip: Mask-Benchmark return scoring (segmentation not needed for this story),
Multi4D semantic stage, persistent roots / lineage column, lineage sketches,
physical-event census.

## A risk found while reading

Ex4DGS's STATIC bank has no temporal opacity at all
(`get_opacity` = `sigmoid(_opacity)`, `c_gaussian_model.py:355-357`), and
static→dynamic promotion is triggered by motion (`‖_xyz_disp‖`), not by
disappearance. The wine bottle is static, so in Ex4DGS it probably ghosts
through the gap from the static bank. The notch must therefore also go on the
static bank's opacity (plain sigmoid in Python, no CUDA). Adding it only to the
dynamic bank would be a vacuous treatment. Multi4D more likely rebuilds the
bottle from new transient Gaussians on return (respawn). Which failure each
baseline shows is itself a result; attribution measures it.

## Compute and gates

~100–150 GPU-h total (~1% of allocation); wall-clock is the constraint.

| gate | date | condition |
|---|---|---|
| 1 | Sep 26 | Baseline fixture results in. If published methods already return well (P2 ≥ our G), the paper stays benchmark+diagnostic and the notch becomes secondary. |
| 2 | Oct 3 | Notch comparison frozen before runs, with a non-vacuity precondition (notch acts on the bottle's Gaussians); results read only after per-arm montages. |
| 3 | Oct 8 | Results in; pick the title; full draft. |
| — | Oct 9–15 | Adversarial review; claim and citation audits. Done. |

## Open decisions for the principal

1. Does the review-gated-diff control model also cover EL-GS code?
2. Accept the menu above?
3. Who does the ~129-frame SAM 2 mask audit?
4. Certificate renewals during Sep 23–26 baseline runs.
