# Follow-up scientific direction — LRV3 membership instrument

## What is and is not refuted

The timing result is not in question. The inferred gap boundaries match the oracle to approximately `1e-14`.

The refuted component is narrower:

> Spatial-partition membership at the tested cell size, applied to the fresh
> sparse initialization cloud on LRV3, achieved approximately 4.5% precision
> and 18% coverage when it bound.

This does not show that non-oracle membership is impossible. It shows that a spatial rule derived from a trained, densified cloud may fail catastrophically when rebound to the compositionally different sparse cloud used by a fresh run.

Any future membership instrument must therefore be evaluated on the exact cloud to which it will bind, before reconstruction training.

## Verification discipline

Do not treat any number quoted in this prompt as established. Every figure here originates from a prior block's records or from an unverified side-channel derivation — including the `2 × 2 × 2` cell decode and the four-cell instrument matrix below. Re-derive each from primary artifacts before it influences a decision. Where a figure cannot be re-derived, say so plainly rather than carrying it forward.

## Immediate order

First verify the live state of experiments 261/262/263 and their stated purpose as the same-code replicate-floor measurement. Treat the reported IDs and running state as navigation, not truth.

If they are still running, monitor them through the tracked Determined tooling while performing the read-only and artifact work below. If terminal, verify provenance, artifacts, hashes, and calculations before interpreting them.

Apply the interpretation table already frozen in [[same-code-replicate-floor-spec-2026-08-23]] §4 exactly as written. Do not move a threshold after reading the numbers. Record the outcome durably.

### Scope of that gate

261/262/263 measure the replicate floor on the N3V 50-frame protocol. That floor gates the N3V utility lane — the B0/B1/B2 ladder, the event-union deltas, and the deferred 300-frame comparison. If it fires, halt N3V utility scaling and record that the measurement protocol must be redesigned first.

It does not gate the LRV3 membership lane, which is a different fixture, a different frame count and camera set, and a different evaluator, with its own measured noise: 0.09–0.17 dB same-arm spread on the LRV3 family against effects of 1.05–2.47 dB. Do not halt LRV3 membership work on the N3V floor result.

Carry one LRV3-specific noise fact forward: the A-est arm's two replicates measured approximately 0.703 dB apart, 4–8× the family's usual spread. Partially-gated configurations are unstable. Every new LRV3 arm needs at least two replicates and must report its own measured spread rather than inheriting the family figure.

## Membership diagnosis

Verify from primary artifacts whether the reported grid pattern is correct:

| Status | Cells |
|---|---|
| Accepted | 364, 365 |
| Rejected | 356, 357, 420, 421, 428, 429 |
| Decoded pattern | One contiguous `2 × 2 × 2` block over `x ∈ {5,6}`, `y ∈ {4,5}`, `z ∈ {4,5}` |

Do not assume this arithmetic proves the mechanism. Confirm from the artifact:
- each cell's decoded coordinates and adjacency relations;
- estimated onset and offset per cell;
- camera-support count per cell;
- accepted/rejected status and recorded abstention reason;
- foreground/background composition on both the trained cloud and the fresh
  initialization cloud; and
- per-row precision, recall, and coverage against the authored sphere, on both
  clouds.

## The gate — freeze it before scoring anything

Write the precision and recall floors, with justification, into a frozen specification before computing any repaired instrument's score. This ordering is what made the previous diagnostic decisive and it is not optional.

Anchor the floors to the measured ordering on `event_return`: fully gated 28.19 > ungated 27.14 >> partially gated 24.67. There is no measurement at intermediate recall, so any interpolated floor would be invented — declare the recall floor as a judgment and label it as one.

Recommended, and to be adopted or replaced explicitly before scoring:

- recall ≥ 0.90 (partial gating is the known failure mode and no intermediate
  recall has been shown safe);
- precision ≥ 0.80 (the oracle is 1.00); and
- a non-degeneracy condition: the instrument must not reduce to the oracle
  sphere test or consume held-out cameras.

## Candidate repairs

The following four measurements are on the record from the prior block.
Re-verify them; do not rediscover them.

| Membership rule | Precision | Recall |
|---|---:|---:|
| 2 cells, fresh cloud — the configuration that failed | ~4.5% | ~17.9% |
| A alone — 8 cells, fresh cloud | ~5.7% | 100% |
| B alone — 2 cells, trained-cloud rows | ~96.9% | ~33.0% |
| A + B — 8 cells, trained-cloud rows | ~94.0% | 100% |

Neither candidate alone is expected to clear a sensible gate: A alone fails on precision, B alone leaves roughly two-thirds of the object ungated.

Evaluate them separately anyway, for a specific reason. The T2 failure confounded low precision with low recall. B alone isolates them — high
precision at partial recall. If B alone still fails, recall is decisively the binding variable, and that is a real finding worth having.

The A+B combination is pre-authorized on the stated justification that it is the only configuration reaching oracle-comparable membership. It does not require a further “neither alone passed” demonstration.

### Candidate A — boundary-conditioned spatial coherence

Test a pre-registrable rule in which a rejected cell may join an accepted component only when it:

- is spatially adjacent to an accepted cell;
- reports the same inferred onset and offset; and
- meets a separately frozen minimum camera-support condition.

This is a joint-evidence rule keyed on boundary agreement, not a threshold relaxation. It is not permission to lower thresholds until the set becomes non-empty. Freeze the rule and its false-positive control before reading any membership score.

#### Required false-positive control

T1's reported properties were zero false activations and 99.52% abstention. Candidate A relaxes acceptance and puts both at risk. Run the coherence rule over all 417 groups and report the full confusion — newly accepted object cells and newly accepted non-object cells. If false activations rise above zero, that is a headline regression and must be reported as one, not absorbed into a summary.

### Candidate B — bind membership to the cloud on which it was inferred

Audit the existing `membership_mode: "row_ids"` path and its cloud-fingerprint protection. Determine whether membership can be inferred and applied to the same trained cloud, avoiding transfer across the densification boundary. Verify the reported trained-cloud figures.

Treat this as a scientific mechanism change, not an engineering repair. It changes the method from:

> infer spatial regions on one model and retrain a fresh model from scratch
to:

> train a model, detect the episode, and apply episodic structure to that same model.

Handle these consequences explicitly:

1. The recorded comparator set becomes invalid. The A1-LOCAL +1.05 dB result
   trained from scratch and cannot be compared to a continue-from-checkpoint
   protocol. No-gate and oracle arms must be regenerated on the same starting
   checkpoint. Budget for it: roughly three arms × two replicates ≈ 6 cells at
   ~0.87 slot-h each ≈ 5.2 slot-h on LRV3.
2. There is a real scientific risk, and it should be pre-identified rather
   than discovered. A model already trained through the gap has learned to
   explain it with a smeared temporal marginal. Whether it re-optimizes cleanly
   once gated is an open question. State in advance what outcome would indicate
   that it does not.
3. Specify the continuation schedule and whether it counts against the
   iteration cap.

Select the simplest instrument that clears the frozen gate. Do not retrain until membership is scored on the exact binding cloud and passes that gate.

## Trackers, segmentation, and geometry models

Do not implement a large new dependency during this block.

Assess candidate families against the actual requirement:

> Assign an object/episode label to primitives in the cloud that will receive the episodic gate, with precision and recall high enough that partial gating is not worse than no gating.

Treat the following as working hypotheses unless primary evidence overturns them:

- Point trackers such as CoTracker or TAPIR may help propagate a 2D mask or association, but do not directly solve primitive membership.
- Motion segmentation is a poor fit for LRV3, because the object is unchanged in pose, colour, and texture before and after the authored absence.
- VGGT, DUSt3R, and MASt3R primarily improve geometry and correspondence and do not directly provide persistent instance membership.
- Segmentation lifted into 3D is the strongest general-purpose alternative:
  supervise per-primitive identity features with multi-view 2D masks and inherit those features through clone, split, prune, and densification.
- Object-level continuity methods may solve membership by moving to whole-object granularity, but risk overlapping the nearest-neighbour novelty territory around CubifyGS.

## Bounded occupancy and implementation check

Do not start an open-ended research-refine cycle.

Perform a bounded check against primary papers and official code for Gaussian Grouping, SAGA, SA3D, Feature-3DGS, OmniSeg3D, Click-Gaussian, CubifyGS, and optionally one tracker-assisted 3D grouping method if directly relevant.

Answer only:

1. How is per-primitive identity represented?
2. How is it supervised across views and time?
3. How is identity preserved through cloning, splitting, pruning, and densification?
4. Does it operate on the initialization cloud, the trained cloud, or both?
5. Can it abstain?
6. What minimum component could be reused as disclosed supporting machinery?
7. Would adopting it materially collapse the project's novelty toward an
   existing method?
8. Does it operate on a dynamic/4D representation with per-primitive temporal
   support, or only on static 3DGS — and, if static, what specifically breaks
   when moved to a 4D representation?

Question 8 is likely decisive for applicability and should be answered first.
Use primary papers and official repositories, and distinguish paper claims from code-verified behaviour.

Run a larger research-refine step only if the artifact audit and this bounded check leave a specific unresolved design question. Do not use research-refine merely to generate more candidate methods.

## Budget and execution

- Any new training cell is capped at 6,000 iterations. No 12k or 18k training is authorized.
- State a slot-hour ceiling for the block and report measured occupancy against it.
- Do not spend GPU time when an artifact-only or CPU diagnostic answers thquestion. The membership instrument scoring is CPU-only and must precede any training.
- Do not submit training merely because an implementation is available.

## Decision required at the end

Return exactly one recommendation:

1. repair the current membership instrument (Candidate A);
2. change to same-cloud `row_ids` binding (Candidate B);
3. adopt the A+B combination;
4. adopt a minimal segmentation-derived identity feature;
5. redesign the measurement protocol first;
6. proceed with the fixture lane as it stands, carrying authored membership as
   a quantified declared limitation — the currently recorded recommendation in
   [[paper-path-decision-2026-08-23b]]; or
7. pause the membership lane.

Lead with the evidence supporting that choice. Include:

- verified status, provenance, and result of 261/262/263, and the frozen interpretation table as applied;
- exact trained-cloud versus fresh-cloud membership metrics, re-derived;
- Candidate A and Candidate B instrument results against the pre-frozen gate, including the full false-activation confusion;
- whether any retraining was scientifically admitted, and if not, why;
- novelty and occupancy findings, with question 8 answered per method;
- Git branch, local/remote SHA, divergence, commits, and protected files;
- experiment IDs, retry indices, pools, image digests, costs, and states;
- permitted and forbidden conclusions;
- unresolved blockers and exact continuation commands for anything still running;
- the smallest next experiment, if one is justified; and
- a concise, self-contained handover.

Continue following the standing Git, Apollo/Determined, delegation, ARIS, verification, and handover rules from the main prompt.
