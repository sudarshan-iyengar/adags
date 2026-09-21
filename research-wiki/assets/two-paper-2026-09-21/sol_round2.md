## Attestation

(a) Can I read files? **YES.**

(b) Verbatim line 99 of [train.py:99](/D:/adags/refine-logs/review-2026-09-21/src/ex4dgs/train.py:99):

```python
                if custom_cam != None:
```

(c) Fable marks **one** of the 19 demands as REJECT: **#17**, marked “REJECT in part.” Demands #2, #5, #13, and #16 are marked MODIFY, not REJECT. See [fable_round2.md:66–88](/D:/adags/refine-logs/two-paper-2026-09-21/fable_round2.md:66).

# Final verdict

Fable’s revision is substantially better and now constitutes a credible two-paper programme. Paper W is ready to be developed as a scoped benchmark/diagnostic paper. Paper C remains high-risk, but the revision now contains real scientific gates rather than a schedule for manufacturing a submission.

My remaining corrections are:

- Audit every primary scored frame in W’s independent-mask pass, not every fifth frame.
- Treat October 5 as requiring two viable event candidates across two scenes, with one fully certified; require both fully certified by October 12.
- Make dormant M2 rows invisible to every non-render pathway, not merely exact-zero in rendered opacity.
- Remove C’s W-derived photometric table entirely, including from the supplement.
- Permit a full W arXiv during CVPR review only after the supplementary deadline and only if the physical/re-ID boundary is genuinely clean.

# 1. Adjudication of every disagreement

## Demand #2 — independent masks

**Split decision: Fable is right about re-profiling stored renders without retraining, but its human-audit density is inadequate.**

Cam00 DEVA was held out from SA4D training, so the current ROI is less circular than my first review implied. A second mask family remains necessary because construction and evaluation still share segmentation ancestry. Fable proposes one-click SAM 2 plus human audit every fifth frame ([fable_round2.md:70–71, 147–161](/D:/adags/refine-logs/two-paper-2026-09-21/fable_round2.md:70)).

Final call:

- Generate SAM 2 masks independently.
- Human-audit **every frame** in P1, P2 and S1.
- Human-audit every fifth frame only in H1/C1.
- Correct all discovered errors, then re-run the frozen reducer unchanged.
- Report DEVA and SAM 2 results side by side.
- If mask choice reverses the sign of G–U or G–sham on any scene-level primary, that becomes a central result and W must not claim robustness.

That is sufficient independent adjudication for W; a third full annotation project is unnecessary.

## Demand #5 — promote the real-occlusion null

**Fable is right.**

The null cannot carry a paper alone. Combining it with the authored-absence positive into one two-sided contribution is stronger and scientifically cleaner: identical mechanism, opposite empirical roles. The revised W-A formulation is good ([fable_round2.md:124–134](/D:/adags/refine-logs/two-paper-2026-09-21/fable_round2.md:124)).

Final call: retain Fable’s combined W-A contribution and make it Figure 1.

## Demand #13 — physical-event prerequisite and zero-acquisition path

**Fable is right about the prerequisite and right to pursue the zero-acquisition path. It is wrong only if the path is treated as likely to succeed rather than as a cheap, decisive census.**

The existing route is real:

- all-camera DEVA masks already exist for flame/sear;
- the remaining N3V scenes can be processed cheaply;
- Panoptic Sports provides six 31-camera, 150-frame sequences, although four cameras currently lack `seg/` and mask instance specificity is unverified ([orchestrator_notes_r3.md:7–16](/D:/adags/refine-logs/two-paper-2026-09-21/orchestrator_notes_r3.md:7), [fable_round2.md:232–243](/D:/adags/refine-logs/two-paper-2026-09-21/fable_round2.md:232)).

But two certified events by October 5 are not a high-probability assumption. Thirty-one dome cameras make true all-view invisibility rare, and the Dynamic 3D Gaussians authors describe their supplied segmentation as crude foreground/background subtraction with noticeable errors—not certification-grade instance masks ([official repository](https://github.com/JonathonLuiten/Dynamic3DGaussians)).

Final call:

- **October 5:** two surviving candidates in two scenes, one fully certified, the second with complete all-camera imagery/masks and a named adjudication path.
- **October 12:** two fully certified events.
- Failure at either date sends C to ICCV.

## Demand #16 — overlap estimate

**Fable is substantively right, but “10–13%” is false precision.**

The redrawn boundary is defensible because the methods, headline metrics and main evidence are now different: W is EL-GS plus authored photometric diagnostics; C is Multi4D roots plus physical semantic/re-ID evidence ([fable_round2.md:263–281](/D:/adags/refine-logs/two-paper-2026-09-21/fable_round2.md:263)).

I would describe residual overlap as **approximately 10–15%**, not defend individual tenths of pages. Page accounting cannot determine substantive similarity.

Cut two more shared elements:

- Delete C5’s W-derived photometric event-window table entirely, including from C’s supplement.
- Delete the STG observation from C; it has no role in the persistent-root paper.

C can cite W once for the authored fixture provenance and use an authored fixture for a semantic mechanism ablation. It should not reproduce W’s metric or result.

Final call: boundary accepted **only if C’s main evidence is physical/re-ID and W’s photometric result does not reappear in C**.

## Demand #17 — full W arXiv

**Fable is right on historical policy; my February-only rule was overconservative. Fable’s proposed November 17 date is nevertheless badly chosen.**

The 2026 guidelines explicitly exempt arXiv from the publication definition ([CVPR 2026 Author Guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines)). However, the 2027 call says very similar contemporaneous work receives additional scrutiny and that influenced work should be discussed; the detailed 2027 Author Guidelines remain unavailable ([CVPR 2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)).

Posting one day after C’s main deadline but before its November 23 supplementary deadline creates avoidable citation, disclosure and version-management awkwardness.

Final call:

- Do **not** automatically post on November 17.
- If the 2027 guidelines retain the arXiv exemption, C’s main tables are physical/re-ID, and the final overlap audit passes, post W on **November 24**, after C’s supplementary package is frozen.
- C should cite the accepted four-page W abstract in the third person and explicitly state the non-overlap.
- If C contains authored-fixture headline evidence, or the 2027 policy changes, hold full W until **February 26**.

Reviewer perception alone is not enough to suppress a cleanly separate technical report for three months. But the boundary must be real, not asserted.

## M2’s remaining defect

Fable’s M2 is close, but exact-zero rendered contribution is not by itself a complete termination.

A dormant pre-gap row must also be excluded after return from:

- FG→TR donor selection;
- densification and cloning;
- KNN semantic smoothing;
- neighbourhood/regularisation losses;
- contribution statistics;
- any optimizer update driven by post-return frames.

Deletion is not mandatory. A tombstoned row is acceptable if it is frozen and excluded from every relevant index and operator. The shared HexPlane may remain shared if the paper explicitly says M2 isolates **root-specific memory**, not every possible source of cross-time information. The critical source behaviour remains that the lift copies FG state while leaving its donor alive ([multi4d-code-map:108–150](/D:/adags/research-wiki/operations/multi4d-code-map-2026-09-19.md:108)).

# 2. Revised scores and odds

## Paper W

- **Review score:** **7/10, weak accept**
- **Confidence:** **4/5**

My odds:

- PhysWorldAI abstract: **65–75%**
- Full CVPR 4D workshop paper: **55–65%**
- Appropriate smaller archival venue: **50–60%**

I essentially accept Fable’s odds. The abstract is slightly stronger than its 60–70% estimate because the new wording directly addresses the most obvious rejection. The full paper remains capped by authored evidence, frozen `NOT_MET`, and the fact that STG is already stronger in the gap under a confounded comparison. The revised binding wording is sound ([fable_round2.md:136–155](/D:/adags/refine-logs/two-paper-2026-09-21/fable_round2.md:136)).

## Paper C

Two different scores matter:

- **Evidence today:** **2/10, reject**, confidence **5/5**
- **If every revised gate genuinely passes:** **6/10, borderline/weak accept**, confidence **3/5**

My odds:

- CVPR after all gates pass: **30–40%**
- ICCV with 3–5 physical events and the mature package: **40–55%**
- Probability that all CVPR gates pass by October 12: **15–25%**

I am slightly more optimistic than Fable about acceptance **conditional on the package existing**, because valid physical events, M2, Cutie, TRASE and the public benchmark together would make a serious paper. I am less optimistic that the package will exist by October 12: physical-event supply, semantic reproduction and leak-free M2 are three independent failure opportunities. Fable’s present estimates are at [fable_round2.md:249–259](/D:/adags/refine-logs/two-paper-2026-09-21/fable_round2.md:249).

# 3. Physical-event path

## Is the path credible?

It is a credible way to obtain an answer quickly. It is not yet a credible promise of two events.

The order should be:

1. Existing all-camera flame/sear masks.
2. DEVA on the remaining N3V scenes.
3. Panoptic Sports proxy-camera scout followed by an all-31-camera audit.
4. Stop and choose ICCV if these fail.

Panoptic Sports is the right place for a cheap scout because it contains fast human/object motion and already has 31 calibrated views over 150 frames. It is not the right place to assume success: the dome’s coverage makes all-camera disappearance uncommon, the clip is only about five seconds, four views lack supplied `seg/`, and the supplied masks may be foreground rather than instance-specific. The original D3DGS setup uses 31 cameras and 150 timesteps ([paper](https://dynamic3dgaussians.github.io/paper.pdf)); the local inventory is documented at [orchestrator_notes_r3.md:7–16](/D:/adags/refine-logs/two-paper-2026-09-21/orchestrator_notes_r3.md:7).

Events that would count include:

- A ball leaving every camera frustum for at least one second and the same ball returning.
- A box being carried entirely out of the captured volume and returned.
- A player leaving all views and returning, if the complete person—not a limb—is the target.
- A ball or object physically hidden from all cameras and later revealed, if depth/geometry or exhaustive view review establishes all-view occlusion.

Events that do not count:

- A ball missed by the foreground segmenter.
- An object absent from the 27 supplied masks but visible in one of the four held-out cameras.
- A crop/resolution failure.
- A hand, limb or racket leaving view if the claimed identity is the whole person.
- An object invisible in cam00 but visible elsewhere.
- A cut, dropped frame or sequence boundary.
- Two events from the same captured scene presented as two-scene evidence.

## Certification checklist

An event is admitted only if every item passes:

### Capture and continuity

- [ ] Intrinsics, extrinsics and timestamps are present and internally consistent.
- [ ] The sequence is continuous: no edit, cut, dropped interval or camera blackout spanning the gap.
- [ ] Every recorded camera is inspected. For Panoptic Sports this means all 31, not only the 27 with supplied masks.
- [ ] The event and thresholds are frozen and hashed before any M1 result is examined.

### Pre-gap and return visibility

- [ ] The target is visible in at least two cameras for at least 10 consecutive frames before the gap.
- [ ] It is visible in at least two cameras for at least 10 consecutive frames after return.
- [ ] In each anchor interval, its mask occupies at least 100 pixels and 0.05% of the image in two cameras.
- [ ] The designated held-out scoring camera has usable target GT for at least five return frames.

### Full-multiview gap

- [ ] The gap lasts at least **30 consecutive synchronized frames or one second, whichever is longer**.
- [ ] Candidate generation finds no more than 10 target pixels in any camera/frame during the gap.
- [ ] Final human adjudication of raw RGB—not masks alone—finds no visible target support in any camera during the gap.
- [ ] A second segmentation method or annotator independently agrees.
- [ ] All ambiguous edge slivers and partially visible frames are removed from the gap; the remaining interval must still satisfy the duration rule.
- [ ] The event is labelled specifically as `exit/out-of-frustum`, `out-of-capture-volume`, or `all-view occlusion`; these classes are not pooled without stratified reporting.

### Identity and difficulty

- [ ] Two annotators independently agree that the post-gap object is the same physical instance.
- [ ] The sequence contains at least one visually confusing distractor during or after return if “re-ID” is claimed.
- [ ] If multiple visually identical balls/boxes occur and identity cannot be externally established, the event may score segmentation but not re-ID.
- [ ] The pre-gap selection is frozen; there is no post-return re-query.

### Ground truth and leakage

- [ ] Target-specific masks exist for all scored return frames.
- [ ] Post-gap GT is evaluation-only and is inaccessible to reconstruction, semantic distillation and selection.
- [ ] Missing predictions and shape/camera mismatches fail closed.
- [ ] Two admitted events come from two different captured scenes.

That is strict enough for a reviewer to accept “certified full-multiview observation absence.” It does not justify saying the object physically ceased to exist.

# 4. Mask-Benchmark on fixture returns

**Yes. Scoring the fixture return window with original-scene Mask-Benchmark GT is legitimate, conditional on exact camera/frame/identity mapping.**

The reason is simple: frames outside `[60,89]` are byte-identical to the originals. The GT for frames 92–99 therefore remains ground truth for the edited fixture. There is no contamination from the teacher-rendered gap after frame 89. The orchestrator correctly narrowed the problem to gap scoring ([orchestrator_notes_r3.md:28–37](/D:/adags/refine-logs/two-paper-2026-09-21/orchestrator_notes_r3.md:28)).

The catches are important:

1. **Camera identity:** 300 masks are consistent with one 300-frame camera, but do not prove it is cam00.
2. **Resolution/transform:** no silent resize is acceptable. The exact crop, resize and camera mapping must be recorded and asserted.
3. **Mask meaning:** if Neu3D-Mask is semantic/dynamic-foreground GT rather than the specific removed instance, it cannot directly score target re-ID, and its gap GT is not simply empty.
4. **Target correspondence:** the benchmark label must correspond to the authored target, not another dynamic object.
5. **No training leakage:** Mask-Benchmark return masks may score C but must not enter censored semantic training, selection or hyperparameter choice.
6. **One-camera limitation:** it evaluates the return on the held-out view; it does not certify all-camera absence.
7. **Gap evaluation remains separate:** use target-specific empty masks only if the target identity is independently established. Do not replace a multi-object semantic mask with all-zero.

Fable’s §3.3 item 8 is correct on return scoring but too broad when it says the gap GT is simply empty ([fable_round2.md:196–205](/D:/adags/refine-logs/two-paper-2026-09-21/fable_round2.md:196)).

# 5. Decisive recommendation for C

A CVPR paper without physical events is defensible **in principle**, but it would be a different paper:

> Persistent Root Semantics for 4D Gaussian Segmentation

Its headline would need to be a substantial, consistent improvement over Multi4D and TRASE on standard public benchmarks, with temporal stability and identity/tracking as secondary evidence. Object permanence could no longer be the title claim.

That requires considerably more than Fable’s present C3 safety condition. I would require, before allowing that pivot:

- all five Neu3D-Mask scenes;
- at least +3 mean mIoU over shipped M0;
- positive improvement on at least four of five scenes;
- a meaningful advantage over rerun or published TRASE under matched inputs;
- matched capacity and M2/control evidence;
- temporal consistency or tracking improvement, not only framewise mIoU;
- ideally a second public benchmark.

Nothing currently suggests that result. The v0.3 expectation is only 0–2 points, and semantic training has not yet been run. “M1 ≥ M0 − 1” is a safety result, not “beat the benchmark” ([fable_round2.md:181–190](/D:/adags/refine-logs/two-paper-2026-09-21/fable_round2.md:181)).

My decisive call is therefore:

> **No two certified physical events by October 12 means C goes to ICCV 2027. Do not submit an authored-only object-permanence C to CVPR.**

The benchmark-only alternative should be reconsidered only if a large standard-benchmark win has already appeared by then. I would not plan on that rescue.

# 6. The action and the failure most likely to decide the programme

The single most important action this week is:

> **Run the all-camera physical-event census immediately and produce human-reviewable synchronized contact sheets for every surviving flame/sear, remaining-N3V and Panoptic Sports candidate before implementing the full M1.**

It is cheap, eliminates the largest uncertainty, and determines whether C is a CVPR or ICCV project.

The single thing most likely to cost the principal one paper is:

> **Failing to obey the October 5/12 stop rule—continuing to engineer and scale an authored-only C after the physical or M2 gate has failed.**

That would consume the calendar, yield a predictable CVPR rejection, and risk starving W’s independent-mask audit and submission.

# 7. Final go/no-go conditions

## Paper W

**GO for PhysWorldAI and full workshop paper only if:**

- [ ] Title and abstract say authored counterfactual; no “genuine,” “physical absence discovery,” or general real-world claim.
- [ ] Frozen `NOT_MET` appears before the 12/12 descriptive result.
- [ ] Every P1/P2/S1 frame is independently SAM-2-plus-human audited.
- [ ] Second-mask re-scoring preserves the basic G–U direction; any changed sham verdict is disclosed.
- [ ] Prefixes are reported as runs nested in three scenes.
- [ ] STG-crb is complete and qualified, or STG is removed from the abstract.
- [ ] Failed estimator admissions are rows, not footnotes.
- [ ] The fixtures, masks, manifests, reducer and failure conditions are releasable under the applicable licences.
- [ ] The four-page submission stays within the PhysWorldAI format.

**NO-GO / reframe as a narrower negative-results paper if:**

- independent masks reverse the primary G–U direction on a scene;
- the release cannot support independent reproduction;
- the abstract again treats the frozen claim as met.

## Paper C — CVPR 2027

**GO only if all hold:**

- [ ] Gate A: fail-closed semantic training/evaluation works; Mask-Benchmark camera, frame, resolution and target mapping are exact.
- [ ] M0 semantics reproduce within a frozen tolerance.
- [ ] Persistent FG roots make measurable pre-gap and erroneous gap contributions; the mechanism is non-vacuous.
- [ ] M2 has disjoint root/optimizer state and excludes old roots from rendering, lift, KNN, regularisation, densification and post-return optimization.
- [ ] Two physical events in two scenes are fully certified by October 12.
- [ ] Each event has a distractor if re-ID is claimed.
- [ ] M1 beats both M0 and M2 by at least +5 J&F with positive direction in at least 5/6 paired units.
- [ ] Root/descendant return contribution improves by at least 15 percentage points and survives causal cohort ablation.
- [ ] Gap false positives do not increase.
- [ ] Cutie does not match or beat M1 on both return J&F and swap rate.
- [ ] The five-scene Neu3D-Mask table is fail-closed and at least competitive with M0/TRASE.
- [ ] C contains no W photometric result or STG analysis.
- [ ] Main tables are physical/re-ID; authored fixtures are controlled secondary evidence only.

**NO-GO for CVPR, move to ICCV if any of these occur:**

- fewer than two certified physical events;
- M1 ≈ M2;
- no persistent-root mechanism signal;
- Cutie erases the headline advantage;
- semantic reproduction or lineage integrity remains uncertain;
- the result requires M3, M4 and several other mechanisms bundled together to win.

**ICCV target condition:** three-to-five certified physical events, mature M2, Cutie/TRASE comparisons, six-seed confirmation where variance requires, and the public benchmark table.

## FINAL RECOMMENDATION

Adopt Fable round 2 with the corrections above. This week, census and visually certify physical-event candidates before substantial C implementation; in parallel, finish W’s independent-mask re-scoring and scoped four-page abstract. Treat W as the dependable paper and submit it to PhysWorldAI, then a 2027 4D workshop. Give C a real CVPR attempt only if two physical events, non-vacuous persistent roots, leak-free M2, Cutie resistance and paired gains all exist by October 12; otherwise move it immediately to ICCV 2027. Permit the full W arXiv on November 24—not November 17—only after C’s supplementary package proves a physical/re-ID boundary and the live 2027 policy retains the arXiv exemption.