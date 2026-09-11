---
title: "Kill-argument review of paper thesis v1 (2026-09-11) — FAIL, four unresolved critical points"
date: 2026-09-11
evidence_bearing: false
---

# Kill-argument review of [[paper-thesis-v1-2026-09-11]] (2026-09-11)

Protocol: the ARIS `kill-argument` skill (two fresh cross-model threads:
a ~200-word rejection memo, then an independent adjudication of the
memo against the paper text and its cited evidence; the verdict is
computed from the per-point counts by the skill's table, never by the
adjudicator). Reviewer model `gpt-5.6-sol`, reasoning `ultra`, fresh
`mcp__codex__codex` threads, no `codex-reply`. Audited input:
`operations/paper-thesis-v1-2026-09-11.md`, sha256
`bca481060297722dcd634d610c3d9571b5b0f524e72b06fa9f14c91b17c0fb51`
(the page as it stood before the amendments recorded below).

Run notes, recorded honestly: the read-only Codex sandbox cannot launch
on this Windows install (`orchestrator_helper_launch_failed`), so the
first attack thread (`01a08dcb…`) could not be confirmed to have read
the files and its memo misattributes the NOT_MET verdict to the
2026-09-09 lane; the first adjudication thread (`01a08dd7…`) refused
to rule without file access. Both threads were re-run with the thesis
and the §7B evidence pasted inline and the sandbox helper bypassed; the
adjudicator confirmed it read the files from disk. The memo and ruling
below are from those threads (`01a08de1…`, `01a08de2…`). Traces under
`.aris/traces/kill-argument/2026-09-11_run01/` (ignored path).

**Verdict: FAIL** (`reason_code: unresolved_critical`) — 7 rejection
points; 0 answered by the current text, 3 partially answered, 4 still
unresolved, of which 3 are critical.

## 1. Attack memo (verbatim)

> Reject. The central claim is not tested under the conditions implied
> by the title. The only positive real-footage experiment is a
> teacher-rendered counterfactual: SA4D removes a hand-selected bottle
> using DEVA silhouettes, and those same construction masks define the
> oracle primitive membership (§3–§5). Thus the fixture's target pixels,
> timing, and membership favor the gated arm by construction; it is not
> evidence that per-primitive multi-interval presence can be discovered
> in physical scenes. The purported estimated-membership arm is
> explicitly vacuous because its rows coincide with truth by
> construction (§4), while independent membership estimation fails
> catastrophically on LRV3 (precision 0.0446, recall 0.1786, −2.469 dB
> return). Even the timing estimator does not identify absence: it
> reports the real hand/knife occlusion as an "absent" interval (§2),
> proving that training-view evidence recovers invisibility, not scene
> exit—the distinction on which the paper is premised. The headline
> "exact absence" is also false for the primary fixture: §7B states
> that no frame in any arm reaches zero error. Most decisively, the only
> frozen, preregistered fixture verdict is NOT_MET because random
> membership improves P1 beyond the allowed sham floor, showing that
> part of the claimed mechanism is generic row deletion. With one
> authored scene, four correlated prefixes, no inferential statistics,
> no physical absence event, and no trained
> PersistGS/SharpTimeGS/FreeTimeGS/CubifyGS baseline, this is an oracle
> diagnostic around known temporal-opacity machinery, not evidence for
> the claimed method or scope.

## 2. Adjudication (per point, verbatim from the second thread)

### P_1: Physical-scene scope mismatch — partially_answered, critical
**Attack claim**: the only positive real-footage result is an authored
counterfactual, so the title-level claim about discovering presence
intervals in physical scenes is untested.
**Evidence**: thesis §§0–1 makes general scene-level claims, while §5
and the fixture page's opening and §7B reading 5 call the fixture
`evidence_bearing: false`, teacher-rendered, hand-selected, and "not
evidence about the physical world." P2/S1 use real unedited frames, but
the absence and P1 target are authored. Scarcity justifies the
diagnostic; it does not sustain the present title or physical-scene
scope. **The ImViD citation does not supply a physical-positive rescue:
its source page permits only two unclassified "occlude-and-return"
events and records 5,936 frames, whereas thesis §3 calls them
absence-and-return events in a 15,215-frame take.**
**Fix**: evaluate an independently adjudicated physical full-multiview
absence-and-return event, or retitle and restrict the central claim to
an oracle counterfactual diagnostic.

### P_2: Construction-dependent oracle and metric — partially_answered, major
**Attack claim**: the editor's silhouettes determine the scored pixels,
authored timing, and oracle membership, making the comparison
favourable by construction.
**Evidence**: fixture §0 and §§1.3–2 confirm the hand-selected object
and window, the DEVA-derived P1 ROI, and the construction-mask
membership vote; §5 confirms truth and estimated masks coincide. The
memo wrongly equates row sets: SA4D removes 1,720 rows from ITS model,
while separate rendered-contribution votes select 6,813–7,004 EL-GS
rows. Common data across arms and negative timing shams support a
conditional representation effect, but mask-level circularity remains.
**Fix**: run a cross-fitted fixture using independent models or
annotators for construction, gate membership, and evaluation, with
object and window frozen before training.

### P_3: Independent membership is unvalidated — still_unresolved, critical
**Attack claim**: G-est contains no independent membership estimate,
while the only independent membership method has extremely low
precision and recall and degrades return quality.
**Evidence**: thesis §§4–5 and fixture §5 expressly say G-est's
membership equals oracle truth by construction; it tests estimated
timing only, so "purported independent arm" is a misattribution the
paper itself admits. Nevertheless the LRV3 diagnostic and the T2 result
verify 0.0446 precision, 0.1786 recall and −2.469 dB; that refutes the
spatial-partition estimator, and no successful alternative exists.
**Fix**: admit a construction-independent membership arm only after it
passes a frozen per-row precision/recall threshold, then compare it with
oracle, random-membership, and ungated training.

### P_4: Timing detects invisibility, not absence — partially_answered, critical
**Attack claim**: T1 labels the hand/knife occlusion absent,
demonstrating that its training-view signal detects invisibility rather
than physical scene exit.
**Evidence**: thesis §0 already concedes that photometric evidence
cannot distinguish LEFT from HIDDEN. The T1 spec defines the estimator
from the loss change under group ablation with no semantic exit cue;
the 2026-09-09 lane §5.1 explicitly calls the event an occlusion. The
memo overstates that source's interpretation, but thesis §2
nevertheless presents it as an "absence window" with zero false
activations.
**Fix**: rename T1 a contribution-gap or visibility-gap estimator and
declare absence-versus-occlusion classification out of scope unless a
matched physical-exit/occlusion test establishes it.

### P_5: "Exact absence" is false for the fixture — still_unresolved, major
**Attack claim**: the headline describes exact absence, although the
primary gate-faithful fixture produces nonzero error on every frame.
**Evidence**: thesis §§0–1 says the authored real-footage fixture
"yields exact absence"; fixture §7B states that no frame in any arm
reaches zero error. The mechanism predicate establishes exact-zero
primitive PRESENCE, not exact reconstruction of the teacher background.
Literal zero-error rendering appears only in synthetic LRV3, on 21 of
27 gap frames. The memo is factually correct; the text conflates two
meanings.
**Fix**: replace the fixture claim with "exact-zero presence gate and
reduced ghost-core error", reserving "zero-error absence" for the
qualifying synthetic frames.

### P_6: Frozen mechanism verdict failed — still_unresolved, critical
**Attack claim**: the preregistered verdict is NOT_MET because
count-matched random deletion exceeded the sham floor, demonstrating a
generic-deletion component.
**Evidence**: fixture §7B reports G-wrongmem − U at +0.65 dB, above the
frozen +0.5 dB floor in three of four pairs; reading 2 attributes part
of the gain to generic deletion. The random set contains ~1% truth
overlap, so it is not perfectly irrelevant. Thesis §§4–5 reports the
+3.78/+2.32 dB membership-specific contrast and correctly identifies it
as a post-result re-reading requiring a new spec; it cannot reverse the
frozen verdict.
**Fix**: freeze G-oracle − G-wrongmem as the primary on fresh scenes and
use multiple count-matched random sets constrained to zero truth
overlap.

### P_7: Insufficient replication and comparison — still_unresolved, major
**Attack claim**: one scene, four related prefixes, descriptive
statistics only, and no trained external baseline cannot establish
generality or competitiveness.
**Evidence**: fixture §0 and §7B specify one scene, four seeded
prefixes, descriptive statistics at four pairs. "Four correlated
prefixes" is imprecise (seeds are distinct; arms within a seed share a
prefix by design), but all pairs share the scene, event, teacher and
masks, so they are not independent scene-level evidence. Thesis §5
admits no trained external baseline; §7 defers both. Requiring all four
named methods is overbroad (inputs and tasks differ), but the absence of
any closest compatible trained baseline remains.
**Fix**: run a preregistered multi-scene evaluation using scenes as the
generalisation unit and train at least one compatible lifespan baseline
on the identical protocol.

## 3. Net assessment (verbatim)

> The current source would not survive a senior area-chair read as a
> method paper under its present title and claims. It supports a
> narrower result: with authored timing and membership, training with an
> exact-zero gate improves a teacher-rendered counterfactual and its
> real-frame return, while correct timing matters. It does not
> demonstrate physical absence discovery, independent membership,
> semantic absence detection, or exact fixture reconstruction. The sole
> frozen fixture verdict is negative, and the evidence remains
> one-scene, descriptive, and baseline-free. The paper itself
> effectively concedes this in §7 by stating what must be added before
> the claims can be made.

Top action items, in the adjudicator's order: (1) a physical
full-multiview absence-and-return result with construction-independent
membership passing a frozen precision/recall precondition; (2) on fresh
scenes, preregister the membership-specific contrast, zero-overlap
random controls, scene-level analysis and one compatible trained
baseline; (3) narrow the title and claims now: T1 estimates visibility
gaps, and the fixture demonstrates exact-zero gating plus reduced error,
not "exact absence".

## 4. Unresolved list and where each goes (the primary's disposition)

| point | disposition in wave 2 (spec v2.0.0) or in the thesis |
|---|---|
| P_1 physical scope | OUT OF SCOPE for wave 2, stated explicitly: the paper claims an authored-absence result on real footage and reports the real-occlusion null; physical absence discovery is not claimed. The ImViD sentence in the thesis is CORRECTED (see the amendment on the thesis page). Title to be narrowed at paper-plan time. |
| P_2 construction dependence | PRECONDITION + ARM: evaluation ROIs come from the held-out camera's segmenter (never used in construction); the estimated-membership arm uses a segmenter independent of the construction masks; object and window frozen before any prefix trains. |
| P_3 independent membership | ARM G-est-mem with a frozen per-row precision/recall precondition against the construction masks, declared before any score; failing the precondition reports that leg as DESIGN_WITHOUT_POWER, never relaxes it. |
| P_4 T1 semantics | RENAMED in the spec and thesis: T1 is a visibility-gap (contribution-gap) estimator; absence-vs-occlusion classification is declared out of scope; the real-occlusion null is the evidence that the gate is inert on occlusion. |
| P_5 exact absence | WORDING RULE in the spec: "exact-zero presence" is the mechanism predicate; "zero-error frames" is reported as a count and never claimed for the fixture; amended on the thesis page. |
| P_6 generic-deletion share | PRIMARY CONTRAST changed to G-oracle − G-wrongmem; two count-matched random sets per prefix with ZERO truth overlap; G − U secondary; scene-level pattern rule. |
| P_7 replication and baseline | THREE SCENES with the claim required in every scene; one compatible trained lifespan baseline arm on the identical protocol (FreeTimeGS-style temporal opacity), reported as NOT RUN if its code cannot be admitted, never silently dropped. |
