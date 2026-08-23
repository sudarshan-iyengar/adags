# DECISION — redesign the measurement protocol first; no membership
# instrument is admitted (2026-08-23, block 3)

Inputs: [[same-code-replicate-floor-spec-2026-08-23]] RESULT,
[[lrv3-membership-candidates-result-2026-08-23]], and the bounded
occupancy check below. Supersedes the recommendation in
[[paper-path-decision-2026-08-23b]] §4 only in *ordering*; the
representation-first paper path is unchanged.

## 1. THE RECOMMENDATION: option 5 — redesign the measurement protocol first

Not chosen for the membership lane's sake. Chosen because the block
measured that **the N3V 50-frame protocol cannot resolve any effect the
ladder was built to measure**, and every real-data claim in the paper
depends on that channel.

Why each alternative is not selected:

| option | why not |
|---|---|
| 1 repair the instrument (A) | A1 0.0667/0.7143 and A2 0.0667/0.8810 fail both frozen floors on the cloud they bind to. |
| 2 same-cloud `row_ids` (B) | B alone is 0.9688/0.3298 — clears precision, fails recall outright. |
| 3 A + B | 0.9375/0.8088. Fails the recall floor, and the shortfall is **structural**: two cells with no absence signal hold 19.12% of the object and cap recall at exactly 0.8088. |
| 4 segmentation-derived identity | Not ready, and structurally disadvantaged here — see §3. A large new dependency, which this block is forbidden to add. |
| 6 proceed unchanged | Wrong: this block invalidated the measurement basis for the real-data half of the paper. "No change" is not an available reading of the evidence. |
| 7 pause the membership lane | Understates the outcome. The lane produced a clean decisive negative *and* a named near-miss with a future spec; and the fixture lane's own noise still permits membership work. |

**Scope, stated because it is easy to over-apply.** Option 5 halts **N3V
utility scaling** — the B0/B1/B2 ladder, the event-union deltas, the
deferred 300-frame comparison. It does **not** halt LRV3 membership work,
which uses a different fixture, frame count, camera set and evaluator.

## 2. The single next experiment, and it is not a mechanism cell

**Establish a resolvable N3V endpoint before running any further N3V
comparison.** The failure is variance, not effect size, so the redesign
must attack variance. Four levers, none yet costed, to be compared in a
frozen spec:

1. **more replicates per arm** — the honest floor scales as
   `R / sqrt(n)`; at the measured 0.4945 dB, resolving a 0.1 dB effect
   needs roughly 25 replicates per arm, which is likely unaffordable and
   should be priced before it is assumed;
2. **a longer schedule** past the densification churn that amplifies the
   divergence 1400× at iteration 500;
3. **a lower-variance endpoint** — pooled over more held-out views or
   more frames, rather than 50 frames of one camera;
4. **paired evaluation on identical densification trajectories** —
   forcing the arms to share densification decisions so the intervention
   is the only difference, which attacks the measured cause directly and
   is the most promising of the four.

**Do not run another mechanism comparison at the current protocol.** Its
result would be unreadable in advance.

## 3. Bounded occupancy check — Q8 answered first, as required

**Q8, and it is decisive: none of the seven operates on a 4D
representation with per-primitive temporal support.** Gaussian Grouping
(paper limitation verbatim: *"currently limited to the static 3D
scene"*), SAGA, SA3D/SA3D-GS, Feature-3DGS, OmniSeg3D and Click-Gaussian
are static 3DGS. CubifyGS is not 4D either — its primitive is
`{μ, Σ, α, c}` and time enters only as object-list membership plus a
per-**object** existence scalar.

**What specifically breaks, verified against THIS repo rather than
paraphrased — with one correction to the worker's account.** Every method
listed learns identity by α-blending a per-primitive feature through the
rasterizer. Here `opacity = opacity * marginal_t` runs unconditionally
for `gaussian_dim == 4` (`gaussian_renderer/__init__.py:257,268`), so a
row outside its temporal support renders at ≈0 opacity and its
α-blended identity feature receives **vanishing gradient** at that frame.

*The correction:* the harder mechanism — `mask = marginal_t[:,0] > 0.05`
with rows sliced out of `means2D`/`means3D`/`ts`/`shs` entirely
(`:321`) — is gated on `pipe.compute_cov3D_python`, and **every
configuration in use sets that False** (verified across
`configs/n3v/ladder_b1_crb.yaml` and all `configs/lrv3/*.yaml`). So rows
are gradient-starved, not excluded. Same direction, weaker form, and the
distinction matters for anyone costing a repair.

Either way the consequence stands: **the rows with the shortest
`_scaling_t` — exactly the event rows a gate must catch — are the least
supervised**, which biases any lifted-segmentation instrument toward the
partial-gating regime this project measures at −2.47 dB.

**A second break that tuning cannot fix.** In SAGA, OmniSeg3D-GS and
Click-Gaussian the contrastive loss relates pixels **within one rendered
frame**. Two primitives whose supports are disjoint — pre-absence and
post-return — never co-render, so no positive pull is ever applied
between them. **The cross-episode link the gate requires has no
supervisory path at all.**

**A third, mechanical.** Gaussian Grouping's 3D regularizer `loss_cls_3d`
does `torch.cdist` over `_xyz` **only**, so on a 4D cloud it forces
spatially co-located but temporally disjoint rows to share identity.
ADAGS already exposes `get_xyzt` (`scene/gaussian_model.py:606-607`,
verified), so a 4D variant is a one-line change — the cheapest concrete
repair the check found.

### The measurement finding, and it is the strongest part

**No method on the list reports per-primitive precision, recall or IoU.
Not one** — all report rendered-2D mask mIoU, and 2D mIoU does not bound
per-primitive recall in either direction, because α-compositing lets a
few high-opacity front rows paint a correct mask while most of the
object's rows stay unlabelled. PointGauss had to build a per-Gaussian
annotated dataset precisely because of this, and scores **69.4–82.5% 3D
IoU on static rigid desktop objects**. The frozen bar (P ≥ 0.80,
R ≥ 0.90) implies **IoU ≥ 0.735**, so the only purpose-built
3D-primitive segmenter in the literature **straddles our bar on the
easiest possible content**.

### Would adopting it collapse novelty?

**Not the CCR claim** — the load-bearing words are trial-render-certified,
appearance-only, per-primitive tying with exact restoration; a membership
instrument is upstream supporting machinery, and CubifyGS remains the
nearest foil at object granularity.

**But it WOULD collapse a presence/gating framing.** "Assign
per-primitive identity, then suppress those primitives" is exactly
Gaussian Grouping's removal path and SA4D's 4D removal. **The only
unoccupied part is that the suppression is time-windowed and the window
is INFERRED** — which is precisely the part this project has
demonstrated (exact boundaries) and the part membership failure is
blocking.

### Strongest candidate if option 4 is ever revisited

TRASE/SADG's tracker-free within-frame contrastive per-primitive feature
— densification-correct and exercised live, with no tracker to lose the
object across an authored gap — plus a 4D `(xyz, t)` variant of
`loss_cls_3d`. Preferable to Gaussian Grouping and SA4D because DEVA is
the component that collapses across absence.

**Strongest objection:** every shipped decision rule in this literature
is an uncalibrated threshold that fails either silently closed
(Feature-3DGS's zero-init NaN, SA4D's monotonically subtractive cascade)
or catastrophically open — SAGA and SA3D-GS both contain
`if count_nonzero(mask) == 0: mask = ~mask`, i.e. **an empty selection
selects the whole cloud**. Under a measured ordering where partial gating
(24.67) is worse than no gating (27.14), a silent recall hole is an
active hazard, not a missing feature.

### Recorded corrections and one unresolved item

Click-Gaussian has **no official code** (404); CubifyGS has **no code**;
SA3D-GS is a branch, not a repository. **Unresolved and flagged as the
highest-value follow-up:** *Spacetime Gaussian Grouping* (EUVIP 2024 /
EURASIP JIVP 2025) appears to port Gaussian Grouping's 16-d identity
encoding onto an STG backbone — the same per-primitive temporal-support
substrate family as ADAGS. **Full text not obtained** (paywalled); only
abstract-level evidence. It must be read in full before any claim that
time-windowed per-primitive suppression is unoccupied.

## 4. Permitted and forbidden

**Permitted.** The N3V 50-frame protocol cannot resolve its recorded
deltas. No membership instrument tested clears the frozen gate.
Precision is a cloud-binding problem and is solved by same-cloud
binding; recall is the binding variable and is capped structurally. No
listed segmentation method operates on a 4D per-primitive-support
substrate, and none reports per-primitive precision/recall.

**Forbidden.** That any recorded ladder number is wrong (none is
retracted — they are unresolvable, not incorrect). That non-oracle
membership is impossible. That hull completion works (unscored under a
frozen spec, post-hoc in origin). That lifted segmentation cannot work
on a 4D substrate — it has not been tried, and the objection is a
predicted hazard, not a measurement. That time-windowed per-primitive
suppression is unoccupied, until Spacetime Gaussian Grouping is read in
full.
