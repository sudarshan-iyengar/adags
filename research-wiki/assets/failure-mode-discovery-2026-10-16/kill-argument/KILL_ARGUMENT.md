# Kill Argument Report: "observed absence is not represented"

**Date**: 2026-10-19
**Reviewer model**: gpt-6.1-sol, effort high (the principal's standing Codex setting), two fresh `codex exec` threads
(attack session 01a152d2-c1f1-7c20-9b2e-77ff9fd8d4b2; the adjudication session id is in adjudication_trace.log), web
search on.
**Input**: kill_dossier.md (claim + plan-page sections 17-24). There is no LaTeX; the skill was adapted to a claim
dossier.
**Verdict**: **FAIL** (`reason_code: unresolved_critical`). Two still_unresolved points are critical: P_2 (conversion-off
ghosts contradict "no notion of existence") and P_4 (no matched temporal-opacity comparison).

## Attack memo (verbatim)
See attack_memo.md. Thrust: Spacetime Gaussians and RetimeGS already give primitives temporal support. The
conversion-off arm of Hybrid 3D-4DGS ghosts without any static Gaussians. No temporal-opacity method was run under the
same protocol, and the fix is not built. So the step from "ghosting" to "a missing representation that needs a new
gate" is unsupported.

## Adjudication (summary; full text in adjudication_memo.md)
| Point | Verdict | Severity |
|---|---|---|
| P_1 Temporal existence already representable (STG, RetimeGS) | partially_answered | critical |
| P_2 Conversion-off ghosts contradict the causal story | still_unresolved | critical |
| P_3 The two interventions do not test the proposed mechanism | partially_answered | major |
| P_4 No matched temporal-opacity comparison | still_unresolved | critical |
| P_5 The fix is not built | partially_answered | critical |
| P_6 The benchmark shows only selected implementation failures | partially_answered | major |
| P_7 (the principal's argument, adjudicated here in the main session) A few synthetic scenes are not enough for a top-tier paper; the failure must be demonstrable on real data | still_unresolved for the broad claim; partially answered for Multi4D monocular | critical |

Net assessment (adjudicator): the measured ghosting is supported. The inference from ghosting to a missing
representational capacity and a necessary new gate is not. The attack kills that inference, not the direction.

## Evidence obtained after the attack (plan page section 25)
- P_4 is now partly answered. Spacetime Gaussians (STG), synthetic suite, 20 cameras, existing runs (three repeats per
  scene): ghost minus ring -11.8 to -18.5 dB. In absolute terms the ghost reads 30-35 dB: a faint residue against an
  unrealistically clean 46-49 dB synthetic surround.
- P_7 cuts against the broad claim. STG at 20 cameras on the three REAL absence fixtures (published 50-frame-chunk
  protocol, two seeds each): ghost minus ring only -0.7 to -3.4 dB. The render is closer to the departed bottle in just
  0-14 % of vacated pixels (Multi4D monocular: 70-99 %). The 20-camera residue is not demonstrable on real footage for
  STG.

## Top action items
1. Narrow the claim to what holds on real data: the failure to learn an observed absence in a static/dynamic
   decomposition at low view counts or monocular (Multi4D monocular: real 3/3). Drop "every SOTA method" and "no
   notion of existence".
2. Real-data-first evidence. Filmed leave-and-return captures with true held-out ground truth. Other monocular methods
   with a static background (Shape of Motion, MoSca) on the real sweeps. The 4-camera tier on the real fixtures.
3. A matched temporal-window baseline (STG one-model and chunked; a RetimeGS-style short-tail ablation) against the
   proposed gate, with present/return quality preserved.
