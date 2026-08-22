# FROZEN — replacement consolidation payload: headroom screen and the
# opacity payload falsification (2026-08-23)

Status: **FROZEN before any cell output.** EXPLORATORY,
`evidence_bearing: false`. Supersedes nothing; it is the successor
experiment to [[b2-edit-falsification-2026-08-20]], whose frozen rule
("stop scaling the current B2 DC appearance operator to N3V") stands
and is not reopened. Fixture facts that bound this lane:
[[lrv3-fixture-hazards-2026-08-23]].

## 1. The question

> Is any information carried across a KNOWN, oracle-correct identity
> link materially useful, once proposal ambiguity is removed?

The DC appearance payload answered "no" on this fixture. This lane asks
whether a different payload answers differently, and — more
importantly — establishes WHY, in a form that generalizes.

## 2. Geometry is DROPPED before any compute, on verified grounds

The candidate ordering handed to this lane put geometry/pose first.
**It is eliminated by inspection.** `EVENT_SPHERE_CENTRE` and
`EVENT_SPHERE_RADIUS` are module constants applied identically wherever
the event object is present, with no time argument
(`scripts/build_synthetic_reveal_scene.py:82-83`, used at `:194` and
`:221-223`). The object returns at the IDENTICAL world pose. An
oracle-correct geometry transfer on LRV3 is therefore an **identity
transform — vacuous by fixture construction**, and would produce a
guaranteed, uninformative null.

Secondary and independent: there is **no per-packet or per-episode
frame anywhere in the executable code**. `_packet_ids` is an integer
column only; `elgs/ops.py` carries the episode-local machinery as
`"M1-gated (episode-local pose/motion tensors not yet present)"` and
its gauge-transport rule as an unimplemented string. "Transfer geometry
into episode-local coordinates" therefore names a coordinate system that
would have to be invented, stored and validated first — and on this
fixture it would be the identity map when finished.

## 3. The headroom screen (runs FIRST, no rendering)

Pure state analysis of experiment 209's `chkpnt6000.pth` (the LRV3 B1
substrate). Row sets are reused BYTE-IDENTICALLY from the DC experiment
— same oracle region file, same probe times, same masks — so every
number is directly comparable to the recorded DC figures.

Three links, exactly as before: **L1** oracle-correct (recipient←donor),
**L2** wrong-identity, **L3** same-identity no-op (parity split).

Candidate tensors screened together: `_features_dc` (the known,
falsified control), `_opacity`, `_scaling_t`, `_t`, `_xyz`, `_scaling`,
`_rotation`.

Reported per tensor per link: mean, median, p95 and max of the per-row
distance, in **both** the raw stored space and the **activated** space
where the parameter is a logit or a log (`sigmoid(_opacity)`,
`exp(_scaling)`, `exp(_scaling_t)`), because the rendered effect depends
on the activated value. `_rotation` additionally reports the geodesic
angle in degrees, since an L2 norm on quaternions is not interpretable.

Two ratios, the pair that made the DC result legible:

* `headroom_ratio = L1_mean / L3_mean` — DC measured **0.0464 / 0.0325 = 1.43**
* `discrimination_ratio = L2_mean / L3_mean` — DC measured **0.706 / 0.0325 = 21.7**

Both row maps are reported: the existing nearest-DC map (PRIMARY, for
comparability with the recorded DC result) and a payload-native nearest
map (SECONDARY). The pair separates "this payload has no headroom" from
"the appearance-derived correspondence is the wrong correspondence for
this payload".

**Screening rule, frozen before output.** A payload is PREFERRED for the
edit test iff `headroom_ratio ≥ 2.0` AND `discrimination_ratio ≥ 5.0`.
The 2.0 demands strictly more headroom than the payload already
falsified at 1.43; the 5.0 demands that the quantity actually
discriminates identity at all, and is permissive against DC's 21.7.
**This is a SCREENING rule that selects which payload to test. It is not
a scientific gate and licenses no conclusion by itself.**

## 4. The payload edit test (runs REGARDLESS of the screen)

`_opacity` is the payload taken to an actual edit, on two verified
engineering grounds: `get_opacity` is read exactly ONCE in the render
path and that single read feeds BOTH the dynamic branch and the
soft-routing static twin, so one redirect is complete and no second
insertion point can drift; and it is the cheapest change with no
coupling into motion, geometry or SH. The redirect targets the RAW
LOGIT before activation.

**The edit runs even if the screen shows no headroom.** The block
requires an oracle-controlled falsification of a replacement payload,
and a measured reconstruction delta against a measured headroom bound is
exactly what made the DC null interpretable rather than merely
discouraging. If the screen is negative, that is reported as the
prediction the edit then tests.

Controls, all mandatory, all through the identical machinery:
oracle-correct same-identity link; wrong-identity link; no-op/placebo
link; the unchanged B1 checkpoint as base; identical reserved units and
evaluator; exact restoration after rejected edits; and a
`non_pointer_state_hash` unchanged across install→clear, proving no
parameter tensor was written.

## 5. Payload promotion gate, frozen before output

A payload advances only if ALL hold:

1. the oracle-correct edit has NEGATIVE reserved loss under the
   unchanged frozen certificate (`mean + 3·SE < 0`, per-side ≤ 0);
2. held-out `event_return` improvement **≥ +0.5 dB** (the floor already
   frozen for this fixture family and used by the localized-presence
   cell);
3. the wrong-identity edit is REJECTED by the certificate and is
   visibly harmful or non-beneficial;
4. the no-op edit is NUMERICAL ZERO;
5. ordinary-region degradation within the frozen non-harm bound;
6. the qualitative return artifact improves on the diagnosed failure.

**If no payload passes: record that consolidation currently has no
useful payload, and recommend the representation-only pivot. Do NOT
relax the certificate, do NOT add payload components, and do NOT
proceed to an N3V B2 after reading a negative.**

## 6. What the two possible outcomes will license

* **Screen negative and edit null** — the strongest available reading,
  and the one the fixture analysis predicts: LRV3's return is identical
  in pose, colour and texture and is observed by 48 training
  view-frames, so the recipient rows are wrong about NOTHING and no
  payload can recover anything. That would generalize the DC
  falsification from "the wrong payload" to **"the wrong fixture"**, and
  the actionable next step becomes an observation-STARVED fixture rather
  than another payload.
* **Screen positive for some tensor** — headroom exists in a quantity
  the recipient did not learn, and the edit measures whether the
  certificate can convert it into reconstruction.

Either outcome is a result. Neither licenses any N3V claim.

## 7. Deliberately NOT done in this block

An observation-starved fixture variant is NOT built before the screen
returns. Designing the starvation without knowing WHICH tensor is
starved, and by how much, would be designing blind; the screen's
per-tensor numbers are precisely the input that design needs. Recorded
so that the omission reads as sequencing, not oversight.
