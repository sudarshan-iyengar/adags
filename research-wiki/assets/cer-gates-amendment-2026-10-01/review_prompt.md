# Review request: gate amendment to the CER specs on Ex4DGS and Multi4D (2026-10-01, before any real CER run)

You are an independent reviewer. Your working directory holds two clean clones on branch `cer`: `ex4dgs/` at commit
5b82ae7 and `multi4d/` at commit 00357aa, the exact code the amendment refers to. Read `ex4dgs/scene/cer.py` and
`multi4d/scene/cer.py` yourself (functions `gates`, `mass`, `begin`); do not trust the amendment's line numbers.

This review covers ONLY the amendment below (Ex4DGS spec §13 and Multi4D spec §14), not the already-reviewed specs or code.

Check, and score 1-10 each with reasons:
1. Faithfulness: does the amendment implement the principal's decisions (a), (b), (c) quoted below, and nothing more? Flag any
   place where it adds a rule the principal did not take, or reads an ambiguous decision one way without saying so.
2. "No substrate code change": does each `cer_gates.json` really store every field the amended preconditions and the
   descriptive tables need, under the names given? On Multi4D, do `residual_share_gap` and `transient_share_gap_region` really
   share one denominator so that their sum is meaningful? Is leaving the stale `pass` field in the file (never read, recorded
   beside the recomputed verdict) safe, or is there a path where it would be read as the verdict (scorer, worker, reports)?
3. How the reducer applies the preconditions (Ex4DGS 13.4, reused by Multi4D 14.4): scene-level design-without-power when any
   run fails; both arms leave rules 1-3; an event type with a design-without-power scene is reported design-without-power
   because the per-layout clause cannot be checked. Is this consistent with the frozen rule 4 ("a failure is reported as
   DESIGN_WITHOUT_POWER for that scene, never dropped") and rules 1 and 3? Is there a simpler or more faithful reading?
4. Multi4D gap-content precondition (14.2) at 0.20 on residual + transient share, stated together with a forecast of
   0.88-0.96 and a smoke-run value of 0.025. Is a precondition the right reading of decision (b), given that (a) lists the
   preconditions that stay? What does the precondition protect against, if anything, on a memory-gate arm?
5. Anything missing that the reducer will need to read the amended rules unambiguously; anything that should be cut.

List findings as severity (fatal / major / minor), the claim, what the code or record actually says (file:line), and the
concrete fix. Say plainly whether the amendment is ready to freeze after fixes.

Context you cannot read from this directory:

================ PLAN-LOG RECORD: the principal's decisions of 2026-10-01 (later), verbatim ================
### 2026-10-01 (later): principal's decisions on the contribution gates and on Multi4D (recorded before any real CER run)

Glossary: memory gate = the memory cohort switched off inside the gap; residual copies = the gap-only copies of the
donors; gap content = whatever paints the gap region during the gap.

1. **(a) The residual share becomes descriptive, not a design-without-power gate (amendment, proposed before any real
   CER run).** The share of the gap-region mass carried by the residual copies is reported as the attribution between
   the memory gate and the residual copies; it no longer decides whether a scene is scored. The preconditions that stay
   (plan §4 rule 4): memory share ≥ 0.50 of the target mass before the gap and after the return, and the exact zeros
   (memory mass in the gap, residual mass outside it). **Stated with it:** the residual shares of the shortened smoke
   runs (integration tests, not evidence) were already seen, 0.06–0.10 on Ex4DGS (0.10 and 0.09 with the rev-2
   snapshot birth, 0.06 with amendment (a)) and 0.024 on Multi4D. This supersedes the 2026-10-01 freeze of the residual
   gate at 0.20 as a precondition; the reducer's 0.10 / 0.20 / 0.30 table stays, as a description.
2. **(b) Multi4D is treated as a memory-gate arm:** the exact memory gate on its persistent banks, with Multi4D's native
   transient bank as the gap content. On Multi4D the gap-content gate counts residual or transient rows (the census and
   forecast show the transient bank paints 0.88–0.96 of the gap region once the memory rows are off).
3. **(c) Exploratory Multi4D work** (for example transient donors, or gating the transient bank) comes only after the
   confirmatory design is frozen, and is labelled exploratory.

Not yet done (open items in [cer-handoff-2026-10-01](cer-handoff-2026-10-01.md)): writing (a) and (b) into both CER specs,
the code that reports instead of gating, and the reducer.

================ PLAN §4 (rules frozen at S0, 2026-09-28, as amended), verbatim ================
**Frozen rules (as amended; nothing below changes after this line without a new dated entry).**

1. **Causal proof, CER against Ex4DGS base.** Six endpoints: the gap window and the return window, for each of
   same-pose, moved and pass-through. Pooled over the three layouts, 9 runs against 9; two-way fixed effects (arm +
   layout); Holm correction over the 6 endpoints. An event type passes when (a) its gap improvement is
   Holm-significant, (b) the one-sided 95% lower bound of CER − base on its return window exceeds −0.25 dB, and
   (c) *(amendment 2026-09-24)* CER's mean gap improvement over base is positive in each of the three layouts
   separately. Causal proof: at least 2 of the 3 event types pass, one of them same-pose.
2. **Against the benchmarks (descriptive).** Per scene, CER (mean of 3 runs) Pareto-dominates a benchmark when it is
   ≥ that benchmark on both gap and return windows. Counts out of 9 are reported against STG, Multi4D and the Ex4DGS
   base mean. Benchmark proof: CER dominates both STG and Multi4D on ≥ 6 of 9 scenes, including at least one of each
   event type.
3. **Do-no-harm.** Whole-frame and ordinary-region PSNR, CER − base, one-sided 95% lower bound > −0.2 dB, pooled.
4. **Preconditions.** A method arm is scored on a scene only if it passes that scene's contribution gates; a failure
   is reported as DESIGN_WITHOUT_POWER for that scene, never dropped.
5. **Scope** *(amendment 2026-09-24)*: repeated trainings measure training noise on these scenes; every claim holds
   "on these nine synthetic scenes". Per-layout differences are reported for every endpoint. The real fixtures are
   always reported separately and never pooled with synthetic scenes; any other combination is written here before
   its runs start.
6. **Multi4D port census threshold (§4b gate 2):** persistent foreground + background carry ≥ 50% of the
   target-region contribution mass over the pre-gap frames, in every layout, on the wave-1 Multi4D same-pose runs.
7. **Independent identity check, strengthened criterion (2026-09-24):** a scene passes iff away-from-edge agreement
   ≥ 0.999 on every pair and pooled, and overall agreement ≥ 0.9999 pooled. Measured on wave 1: 9/9 at 100.0000%.
8. **Dense-initialisation variant (wave 2, not part of the proof):** precondition and robustness criterion exactly as
   written in the 2026-09-24 evening entry.
9. **COLMAP criterion A3:** unchanged; the wave-1 miss (3 of 2,700 STG frames, mug-layout gap frames) stands as the

================ PLAN §4b (Multi4D port; gate 1 dropped by the principal on 2026-09-30), verbatim ================
## 4b. Conditional wave-2 item: port CER to Multi4D's persistent bank (added 2026-09-23, principal)

**Why.** Ex4DGS was chosen for speed and engineering safety, not as the better model:

- a single Python opacity junction;
- ≈ 1.2 vs 2.7 GPU-h per run;
- an evaluation stack validated to 0.0016 dB.

The record's history of that choice is in
[multi-interval review round 2](ex4dgs-multi-interval-review-round2-2026-09-21.md)
§"substrate reversal" and [sol_round1.md:397](../assets/neurips-plan-sol-2026-09-21/sol_round1.md).

Multi4D reproduced its paper (3/3 inside the band), is quieter (0.3445 dB range at
n = 3) and newer (ECCV 2026). A port shows that the mechanism is
**substrate-independent**, and it answers the objection that we picked the weaker
substrate. Sol listed a successful Multi4D port as one of the moves from 7/10 to
8/10 (sol_round1.md:469).

**Entry gate.** All three conditions must hold. They are checked in order, and
they concern the setup, not a score:

1. **The Ex4DGS causal proof passes** at the Mon Sep 28 readout (§4 rule 1).
2. **Multi4D's persistent bank actually paints the target (census, non-vacuity).**
   On the wave-1 Multi4D base runs of A0, persistent FG + BG must carry
   **≥ 50% of the target-region contribution mass** over the pre-gap frames,
   **in every layout**.
   - If the transient bank paints the target, gating the persistent bank is
     vacuous. That outcome is reported as a finding about Multi4D's architecture,
     not as a port failure.
   - The 50% threshold is **proposed here and frozen at S0**.
3. **The port diff reaches the principal by Thu Oct 1.** If it does not, the port
   is deferred past the data freeze, to the ideal plan and the camera-ready.

**Design, reusing the Multi4D code map §5; verify first at the pinned commit `c483e82`:**

- **Gate all three render paths:** standalone `gaussian_renderer/__init__.py:198`,
  mask render `:314` and hybrid `:468`.
- **`:314` is the recorded silent bypass.** A test renders through each path and
  asserts identical gating.
- **The transient bank is not gated.** Its temporal cull is in CUDA
  (`forward.cu:281-285`).
- **The gap-only residual cohort goes in the persistent PyTorch path**, since the
  transient bank cannot host it.
- **Lifecycle:**
  - densify/prune and the FG→TR lift are disabled for the residual cohort during
    its stage, because the lift **copies rows and leaves the donor alive**
    (query pack, 2026-09-21);
  - every run is one uninterrupted job, because resume restores the FG branch only.

**Tests before any training:**

- `q = 0` gives byte-identical images to unmodified Multi4D, on all three render paths.
- The contribution non-vacuity gate passes.
- The residual is exactly zero outside the gap.
- A checkpoint round-trip reproduces the model.
- The diff is **review-gated**, the same control model as the Ex4DGS diff.

**Runs:**

- **Confirmatory:** CER-Multi4D vs Multi4D base on **A0 across three layouts**, 3 vs
  3 per layout. The base reuses the wave-1 run plus 2 new runs. The model is the
  same pooled blocked one as §4 rule 1, over P1 and P2 with Holm correction.
- **Descriptive, only if the queue allows:** CER-Multi4D ×1 on R2 and G1 per layout.
- **Checkpoint policy:** Multi4D **final iteration** is primary, for both arms.

**Cost:**

- ≈ 2–3 engineering days.
- **≈ 40 GPU-h** for the core: 6 new base runs and 9 CER runs at 2.7 GPU-h each.
- **≈ 57 GPU-h** with the descriptive R2 and G1 runs.

**Head start at zero GPU, taken regardless of the proof:**

- On Sun 27, while CER-Ex4DGS trains, verify the code-map attachment points on
  Leonardo at `c483e82`.
- Write the FG/TR census, and run it on the wave-1 Multi4D A0 models.

The census is informative on its own terms: it shows which bank paints an object
that leaves and returns.


================ EX4DGS SPEC §6 (the gates being amended), verbatim ================
## 6. The contribution gates (plan §4 rule 4; rev 2 §6), computed at the end of training

Written by `train.py` after the last iteration to `<model_path>/cer_gates.json`, on the final
model with `q = 1`, training cameras only (the gates read no cam00 pixel, so they cannot leak the
scored outcome). Thresholds declared here, before any CER output exists:

| gate (rev 2 §6 wording) | measurement | pass | kind |
|---|---|---|---|
| memory active before the gap | memory share of target-mask mass over the pre-gap window | ≥ 0.50 | evidence |
| memory active after the return | memory share of target-mask mass over the return window | ≥ 0.50 | evidence |
| memory suppressed inside the gap | memory mass over the gap window, every pixel | exactly 0 | guard (implied by §1 once T3 passes) |
| residual active inside the gap | residual share of gap-region mass over the gap window | ≥ 0.20 | evidence |
| residual zero outside the gap | residual mass over the pre-gap and return windows, every pixel | exactly 0 | guard (implied by §1) |
| both banks covered | memory rows and pre-gap memory mass per bank | reported | the factor acts on the concatenated vector by construction; T3 checks a memory row in each bank |

`gates` also asserts that every evaluated opacity is finite on every frame (`check_finite`): a NaN opacity times a 0 factor is NaN,
and `min(0.99f, NaN)` renders it at alpha 0.99 (`forward.cu:379`), while `prune_nan_points`
checks positions only (`:1229-1241`).

The birth-time numbers (memory share on the ungated model, memory rows per bank, donor count,
donor share of the gap region on the gated model) go to `cer_begin.json` at 30,000. The 0.50
memory threshold is the one frozen for the Multi4D census (S0 rule 6); 0.20 is Sol's round-1
figure. **A failure of the after-return gate on a same-pose scene** can mean the base already
represented the return with a second lineage (a row whose plateau starts at B + 1 has no pre-gap
mass and is not memory); it is recorded as that, not as a CER defect. On moved-return scenes the
memory rows may not follow the target to its new pose, in which case that gate fails there.

**How the gates reach the proof.** Every planned CER run is scored in full; the scorer copies the
gate verdict into the score file. Plan §4 rule 4 is applied by the reducer: a scene whose CER runs
fail a gate is reported DESIGN_WITHOUT_POWER in the proof, and its numbers stay on the record. No
run is withheld from scoring and none is dropped.


================ MULTI4D SPEC §6 and §13 (being amended), verbatim ================
## 6. Gates

As on Ex4DGS (spec §6), computed at the end of training on the in-memory model, training cameras only, written to
`<model_path>/cer_gates.json`: memory share of target mass ≥ 0.50 before the gap and after the return; residual share of
the gap region ≥ 0.20 in the gap; memory mass in the gap and residual mass outside it exactly 0 (guards); the transient
bank's share of the gap region and of the target mass in the gap reported. `begin` and `gates` assert that every
foreground and background opacity is finite (`min(0.99f, NaN)` renders a NaN at alpha 0.99, `forward.cu:651`; Multi4D
has no NaN prune). Every run is scored; plan §4 rule 4 acts in the reducer.

## 13. Rev 3 (2026-10-01): after the code reviews and the principal's decisions (appended; the sections above are rev 2)

- **§8 is superseded on one point: there is now one CUDA change, and it never touches training.** The upstream hybrid
  forward never writes `max_contrib`, which the launched backward's early exit reads (§3), and in a fresh process the
  instrument measured exactly 0 mass against 1.37e6. Principal's option (a): a block in `forward.cu` compiled **only**
  under `-DCER_WRITE_MAX_CONTRIB` writes the per-tile maximum of `last_contributor` (commit 33215f1, its own commit);
  `scene/cer.py` builds a second extension from the same sources with that macro, under the name
  `diff_gaussian_rasterization_hybrid_cer`, and `instrument_kernel()` swaps it into the hybrid wrapper for the
  duration of `mass()` only (c1e6045). The training and evaluation renders use the unmodified kernel in both arms. The
  per-render conservation assertion stays. Both code reviewers verified the CUDA change (right quantity for the early
  exit, index-consistent, sync-safe; without the macro the training build is the upstream kernel). The upstream
  defect's effect on the baselines is not measured now (principal: later, diagnostic only).
- **The reload refusal lives in the foreground and background `load_ply`** (7d8c5e2), not in the scene's load path: the
  semantic scripts load directory checkpoints by calling `load_ply` directly. This edits two model classes by two lines
  each, contrary to "no model class is edited" in §2.1.
- **§6:** the transient bank's share of the gap region is reported; "of the target mass in the gap" is withdrawn (the
  target is absent from the world in the gap, so its mask is empty there).
- **§7:** the forecast is selected as `FORECAST`; unknown test names are rejected; T3's gradient checks run on the
  instrument build; T4b (new) runs `mass()` in a fresh process and reports, descriptively, whether the training build
  breaks conservation there; T5 also requires the refusal at the direct `load_ply`; T6's regex and sentinel check use
  the full `LIFECYCLE` list (three names added); T7 requires one fine-evaluation series and 300 saved renders.
- **Launcher:** builds both extensions before training starts (a build problem fails at t = 0) and records the
  `forward.cu` hash.
- Residual gate primary threshold 0.20 frozen (plan log 2026-10-01); 0.10 / 0.20 / 0.30 reported descriptively.


================ SMOKE-RUN OUTPUTS (integration tests, not evidence) ================
Ex4DGS T7 on 5b82ae7 (17,200 iterations, CER start 13,000), cer_gates.json:
{"memory_share_pregap": 0.931618056219939, "memory_share_return": 0.9178570820600961, "memory_mass_gap": 0.0,
 "residual_share_gap": 0.060433129773706754, "residual_mass_outside_gap": 0.0, "memory_rows_static": 387,
 "memory_rows_dynamic": 527, "memory_mass_pregap_static": 1603889.73, "memory_mass_pregap_dynamic": 4257566.58,
 "residual_rows_static": 8, "residual_rows_dynamic": 80, "residual_mass_gap_region_static": 14446.97,
 "residual_mass_gap_region_dynamic": 175653.34, "pass": false}
Multi4D T7 on 4261b1a (code identical to 00357aa in scene/cer.py; lift 4,000, CER start 4,500, end 5,000):
cer_begin.json: {"iteration": 4500, "memory_rows": {"fg": 5, "bg": 3119}, "memory_share_pregap": 0.9425, "residual_rows": 99,
 "gap_region_share_by_bank": {"fg": 0.4501, "bg": 0.5485, "tr": 0.0014},
 "majority_donor_share_by_bank": {"fg": 0.3829, "bg": 0.1142, "tr": 0.0010}}
cer_gates.json: {"memory_share_pregap": 0.9437, "memory_share_return": 0.9437, "memory_mass_gap": 0.0,
 "residual_share_gap": 0.0239, "residual_mass_outside_gap": 0.0, "transient_share_gap_region": 0.00087,
 "memory_rows": {"fg": 5, "bg": 3119}, "rows_at_end": {"fg": 12602, "bg": 120824, "tr": 36105}, "pass": false}

Scorer (agent-control/ars/ars_score.py, method ex4dgs_cer): copies cer_gates.json into the score file under a key; it
does not gate scoring on it. No reducer exists yet; it will be written after this amendment is frozen.

================ THE AMENDMENT UNDER REVIEW: Ex4DGS spec §13 ================
## 13. Gate amendment (2026-10-01, the principal's decision (a); appended, before any real CER run)

Applies the plan-log entry "2026-10-01 (later): principal's decisions on the contribution gates and on Multi4D", item
(a). It supersedes §6 row 4 ("residual active inside the gap ≥ 0.20, evidence"), §11 item 4 and amendment (a) §A5's "gate
thresholds unchanged (residual 0.20 frozen)" **as preconditions**. It also supersedes the plan-log freeze of 2026-10-01
item 1 in the same sense. Everything else in this spec and in amendment (a) stands. The Multi4D spec §14 is the companion
for Multi4D.

**13.1 Preconditions (plan §4 rule 4) on Ex4DGS, complete list.** A CER run meets the preconditions when all four hold,
read from its `cer_gates.json`:

| precondition | field | rule |
|---|---|---|
| memory active before the gap | `memory_share_pregap` | ≥ 0.50 |
| memory active after the return | `memory_share_return` | ≥ 0.50 |
| memory suppressed inside the gap | `memory_mass_gap` | exactly 0.0 |
| residual zero outside the gap | `residual_mass_outside_gap` | exactly 0.0 |

Finite evaluated opacity is asserted in `begin` and `gates`; a run that fails it stops before `cer_gates.json` exists
and is reported as a failed run, never scored.

**13.2 The residual share is descriptive.** `residual_share_gap` (the residual copies' share of the gap-region mass over
the gap window, training cameras, final model) is reported for every run as the split of the gap content between the
residual copies and the ordinary rows. The memory rows carry exactly zero there, so the remainder is ordinary rows. It
decides nothing about scoring. It is a description of which rows paint the gap, not a causal attribution of the gap
improvement. That attribution needs the memory-gate-only ablation (wave 2, §10). The reducer reports, per run and per
scene, the share and the verdict "share ≥ 0.10 / 0.20 / 0.30" as a descriptive table. **Stated with it:** the
shortened smoke runs (integration tests, not evidence) already showed shares 0.10 and 0.09 with the rev-2 snapshot birth
and 0.06 with amendment (a).

**13.3 No substrate code change.** `cer_gates.json` written by commit `5b82ae7` already stores every field in 13.1 and
13.2 (`scene/cer.py:159-167`). Its `pass` field (`:168-170`) still applies the superseded rule (it requires
`residual_share_gap ≥ 0.2`). That field is superseded: the reducer never reads it. The reducer recomputes the verdict
from the four fields with the thresholds above and records both values, so the superseded field stays visible. The
reviewed commit `5b82ae7` and the scorer pin therefore stay as they are. The CER runs are submitted on `5b82ae7`.

**13.4 How the reducer applies the preconditions** (rule 4: "a failure is reported as DESIGN_WITHOUT_POWER for that
scene, never dropped"):

- **Per scene.** A scene meets the preconditions when every one of its CER runs does. If any run fails, the scene is
  DESIGN_WITHOUT_POWER.
- **Excluded from inference.** Both arms of a scene that is DESIGN_WITHOUT_POWER leave the inferential rules: the
  pooled model of rule 1, the counts of rule 2 and the bound of rule 3. All of that scene's numbers are reported in a
  separate table with the failing fields.
- **Consequence for an event type.** If one of an event type's three scenes is DESIGN_WITHOUT_POWER, the clause "mean
  gap improvement positive in each of the three layouts" cannot be checked. That event type is then reported
  DESIGN_WITHOUT_POWER, not passed or failed. Its pooled estimate over the remaining layouts is reported descriptively.
- **Missing scores.** The reducer refuses to run when any planned score is missing: 27 CER runs and the base runs the
  principal designates for the proof (see 13.5).

**13.5 Not decided here.** Which Ex4DGS base runs enter the proof is the principal's decision of 2026-09-29, still open:
the frozen originals, or the replacements for the runs trained on `lrdn0066`. The reducer takes that choice as an
explicit argument with no default. The run list is recorded in its output.

================ THE AMENDMENT UNDER REVIEW: Multi4D spec §14 ================
## 14. Gate amendment (2026-10-01, the principal's decisions (a), (b), (c); appended, before any real CER run)

Applies the plan-log entry "2026-10-01 (later): principal's decisions on the contribution gates and on Multi4D". It
supersedes §6's "residual share of the gap region ≥ 0.20 in the gap" and the last bullet of §13 **as preconditions**.
Everything else in rev 2 and §13 stands. The Ex4DGS spec §13 is the companion; its 13.3–13.4 (no code change; how the
reducer applies the preconditions) hold here with the Multi4D fields below.

**14.1 What the Multi4D arm is.** CER-Multi4D is a **memory-gate arm**: the exact memory gate on the persistent foreground
and background banks, with Multi4D's own transient bank as the content that paints the gap. The background residual
copies are still born by the unchanged code (§4.2). The forecast is 6 / 0 / 1 donors on the three same-pose scenes. In
reports this arm is called "memory gate on Multi4D". It is never described as residual copies painting the gap.

**14.2 Preconditions on Multi4D, complete list**, read from `cer_gates.json`:

| precondition | field(s) | rule |
|---|---|---|
| memory active before the gap | `memory_share_pregap` | ≥ 0.50 |
| memory active after the return | `memory_share_return` | ≥ 0.50 |
| memory suppressed inside the gap | `memory_mass_gap` | exactly 0.0 |
| residual zero outside the gap | `residual_mass_outside_gap` | exactly 0.0 |
| gap content from residual or transient rows | `residual_share_gap + transient_share_gap_region` | ≥ 0.20 |

The two shares in the last row have one denominator: the gap-region mass over the gap window of every row in all three
banks (`scene/cer.py:211`, `:213`, both divided by `whole(gap_in)`). Their sum is therefore the share of rows that are
either residual or transient. 0.20 is the threshold of the gate it replaces. The memory denominators also include the
transient bank (`whole`, `:204-206`).

**Stated with it: values already seen.**

- **Forecast.** At birth on the full-schedule wave-1 base models, loaded at 20,000 as a stand-in for 16,000, the
  transient bank paints 0.88–0.96 of the gap region once the memory rows are off.
- **Smoke run (integration test, not evidence).** The shortened run T7 (`cer_t7_m4d_r4`: lift at 4,000, CER start at
  4,500, end at 5,000) shows the opposite. At birth the gap region is painted by foreground 0.45, background 0.55,
  transient 0.001. At the end the residual share is 0.024 and the transient share 0.0009. It would fail this
  precondition. The two disagree, and the shortened schedule gives the transient bank only 500 lift iterations before
  CER starts.

**14.3 Descriptive.** The following are reported per run and per scene, each with the 0.10 / 0.20 / 0.30 table. None
of them decides scoring.

- the residual share (the split between memory gate and residual copies, as on Ex4DGS);
- the transient share;
- their sum;
- the share of ordinary persistent rows, which is one minus the sum.

**14.4 Confirmatory design (plan §4b, as amended 2026-09-30), and how the reducer reads it.**

- **Runs.** CER-Multi4D ×3 against the wave-1 Multi4D base runs 1–3, on the three same-pose scenes.
- **Model.** Pooled over the layouts, 9 against 9, with arm and layout fixed effects.
- **Endpoints.** The gap window and the return window, Holm over these 2.
- **Pass.** The same-pose condition passes under plan §4 rule 1's three clauses: the gap improvement is
  Holm-significant; the one-sided 95% lower bound on the return window exceeds −0.25 dB; the mean gap improvement is
  positive in each layout.
- **Preconditions.** 14.2, applied per scene as in Ex4DGS 13.4.
- **Readings.** Multi4D's final iteration 20,000, scored as the base runs are.
- **Descriptive runs.** Moved-return and pass-through runs (§4b, "only if the queue allows") stay descriptive.

**14.5 Exploratory work (decision (c)).** Transient donors, gating the transient bank and any other Multi4D variant
start only after this confirmatory design is frozen, meaning after the reviews of this amendment and the principal's
acceptance. Their run ids, the ledger and every report label them exploratory. They are never pooled with the
confirmatory runs.

**14.6 No substrate code change.** The Multi4D `cer_gates.json` at `00357aa` stores every field in 14.2 and 14.3. Its
`pass` field (`:216-218`) applies the superseded rule. As on Ex4DGS, that field is superseded, never read, and recorded
beside the recomputed verdict.
