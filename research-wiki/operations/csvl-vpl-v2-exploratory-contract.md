# CSVL-VPL v2 Exploratory Experiment Contract (workstream B)

Date: 2026-07-30
Branch: `csvl-vpl-v2-exploratory` (from the `csvl-vpl-v2-phase0` tip; the census
branch stays unchanged)
Status: exploratory tier. Nothing here is Gate A, Gate B, Phase 0 success, or a
disocclusion claim. Both census verdicts ([[operations/phase0-census-result]],
[[operations/phase0-census2-result]]) stand unchanged.
Authorization: user message of 2026-07-30 (workstream B) — full trainings on
approved development scenes, bounded Slurm/GPU, causal controls, matched
resource accounting, instrumented activation.

## Scene and evidence scope

- Scene: `cut_roasted_beef` only for this round. Sealed P01 DA3 evidence exists
  only for this scene; producing cook_spinach evidence is a multi-GPU-day
  preprocessing job recorded as future work. This narrows "development scenes"
  to the primary one — a disclosed limitation of every number below.
- Evidence artifact: consensus depth/sigma/validity per (camera, frame) built
  once from sealed P01 (17 cameras; cam12/cam19 excluded as in census-v2),
  hashes recorded in the artifact meta. cam00 is never read.
- R009 is untouched: not for tuning, selection, evaluation, or reporting here.

## Base configuration and baseline anchor

All lanes derive from `configs/n3v/fixed_budget_lora_route0_filemask_residual_600k.yaml`
(the Slice-B/phase9 baseline family): 6000 iterations, 600k point cap,
resolution 2, LoRA route0, seed 0. Historical anchor in the `--val` protocol on
cut: PSNR 34.25 / SSIM 0.9610 / LPIPS 0.0518 at 6000 (run 20260619_184247,
`test/ours_6000/stats/validation.json`). The 35 dB aspiration is judged against
the in-harness L0 rerun, with the historical anchor as context. The
`training_report` protocol (cam00 test view metrics incl. dynamic-mask and
static-region PSNR) is reported alongside; the two protocols are never mixed.

## Lane matrix (round 1: six lanes, seed 0, full 6000-iteration trainings)

| Lane | Config | Mechanism | Question it isolates |
|---|---|---|---|
| L0 | `lane_l0_route0` | base, lifecycle off | in-harness baseline |
| L1 | `lane_l1_internal` | E1 protection + occlusion-aware exposure normalization; no births | do the internal limbs alone help? |
| L2 | `lane_l2_presence_vad` | presence-weighted exposure only (TAD-GS-style control); no protection, no births | is occlusion-awareness needed, or does any presence reweighting do it? |
| L3 | `lane_l3_full` | full automatic lifecycle: protection + exposure + E2 birth with budget-matched retirement + reveal/unfreeze | the headline mechanism |
| L4 | `lane_l4_generic_capacity` | event-blind births (B01-style targets) at identical cadence/K/donor rule; protection/exposure off | is L3's effect just capacity churn? |
| L5 | `lane_l5_shifted` | L3 with per-camera circular time-shift (+101 frames) of the evidence | is L3's effect evidence-alignment-specific? |

Control-design note (workstream A lesson): the misaligned-evidence control is a
circular TIME-SHIFT, not a frame shuffle. The audit
([[operations/phase0-audit-result]]) showed frame-shuffling manufactures
pseudo-events for transition-triggered logic; a time-shift keeps evidence
temporally coherent while destroying alignment, so it inherits no such
artifact. The training mechanism consumes instantaneous per-view verdicts (no
duration/grace certification), so the census failure mode of losing long
ragged events does not apply to the mechanism itself; material-aware
abstention (sigma threshold) is built into E1 per the glass-flicker finding.

Redundancy judgment: no separate "reveal-only" or "protection-only vs
exposure-only" split in round 1 — L1 bundles the internal limbs; if L1 shows a
signal, a targeted split is a legitimate round-2 addition. No second seed in
round 1; the effect-size floor below compensates conservatively.

Mechanism definitions as implemented (binding for interpretation):

- **Protection** (L1/L3/L5) = per-iteration gradient freezing of primitives
  whose E1 verdict is OCCLUDED in every evidence-bearing view of the batch,
  PLUS persistent-occlusion (EMA > 0.6) vetoes on pruning, split selection,
  split parent-removal, donor selection, and the densification score (a
  frozen row may not spend clone/split budget — this also guards the
  exposure denominator's clamp(min=1) from inflating fully hidden rows).
- **Exposure normalization** = the densification gradient denominator becomes
  the per-primitive sum of per-view exposure weights (E1 mode: occluded 0.0,
  behind-weak 0.5, else 1.0; presence mode: marginal_t clamped [0.05,1]),
  normalized by batch size so it reduces EXACTLY to the baseline denominator
  under all-ones weights (verified numerically) — the L2/L0 contrast is
  therefore pure weighting.
- **Birth** picks the first evidence-bearing view of the batch; colours are
  sampled from that view's ground-truth image with consensus-to-image pixel
  rescaling; donors are the event-blind rule with the persistent-occlusion
  veto applied identically in E2 and generic modes (the L3/L4 contrast is
  target construction only).
- Evidence artifact: `evidence-consensus-cut-v1` (built by job 50882303 from
  sealed P01; 17 cameras, 300 frames, 5100/5100 maps passing, zero geometry
  drift; hashes in its meta.json). cam00/cam12/cam19 contribute no evidence
  by construction.

## Schedules, budgets, resources

- Iterations 6000; densification per base config; births (L3/L4/L5): K=256
  every 500 iterations in [1500, 5500] (max 9 events, <=2304 reassigned rows),
  strictly point-neutral through the B01 transaction (hard postcondition).
- Checkpoints saved at 1000/3000/6000; test_iterations 1000/2000/3000/4500/6000.
- Per-lane: 1 A100, <=6 h wall. Round-1 cap: <=30 GPU-h photographic
  (6 lanes + smoke + evals + qualitative renders).
- **Result-conditioned iteration cap: at most 3 rounds total** (this round
  included). Uniform bug fixes applied identically to every lane do not count
  as rounds; mechanism changes in response to a mechanism's own result do, are
  recorded per round (what/why/motivating observation), and either rerun the
  affected matrix or are reported as a separate iteration — never spliced.

## Activation diagnostics (a result may not hide behind "it didn't activate")

Every lane writes `lifecycle-ledger.jsonl` (every 100 iterations) and summary
totals. Round-1 validity requirements:

- L1/L3/L5: nonzero occluded verdicts and nonzero protected counts by iter
  2000; exposure denominator differing from raw view counts for >=1% of
  primitives; else the lane is reported "mechanism inactive — invalid", not as
  a null.
- L3/L4/L5: >=1 birth event with realized_k > 0 (or the ledger's explicit
  skip reasons reported verbatim); accepted/rejected proposal counts, donor
  protection exclusions, and post-transaction budget equality logged.
- L2: presence-weight distribution logged and distinct from L0's implicit
  counts.
- All: E1 verdict histograms (incl. abstention), per-component wall-time,
  peak CUDA memory, final/realized point counts.

## Interpretations (pre-declared)

Noise band: |dPSNR| < 0.05 dB or |dLPIPS| < 1% relative — differences inside it
are nulls (heuristic band from B01-scale experience; single seed — stated
limitation).

- **Mechanism-consistent signal**: L3 > L0, L3 > L4, L3 > L5 outside the band
  on val PSNR or LPIPS, static-region PSNR within -0.05 dB of L0, activation
  valid -> justifies moving toward annotated event evaluation. Does NOT
  support: any disocclusion claim (needs annotated events + the full causal
  matrix per the objective), any Gate A/B statement, any claim beyond
  cut_roasted_beef at seed 0.
- **Internal-only signal**: L1 > L0 outside band with L3 ~ L1 -> protection/
  exposure help; births inert. Round-2 candidate: birth redesign or split of
  internal limbs.
- **Presence-equivalent**: L2 ~ L1 within band while both > L0 -> occlusion
  awareness adds nothing over presence reweighting; the E1 evidence layer is
  not earning its complexity — report as a negative for the evidence coupling.
- **Capacity-equivalent**: L4 ~ L3 > L0 -> churn/capacity effect, not
  evidence; no visibility attribution permitted.
- **Alignment-insensitive**: L5 ~ L3 -> gains not evidence-alignment-specific;
  no visibility attribution permitted.
- **Null across lanes**: all within band -> the mechanism family does not move
  dev-tier global metrics at this scale; honest null, preserved; decide
  between event-level evaluation (metrics may be insensitive) and redesign.
- **Harm**: any lane < L0 beyond band on val PSNR or static harm beyond
  -0.05 dB -> record as negative with the lane's activation evidence;
  diagnose before any round-2 change.
- 35 dB: crossing it in any lane is reported in context (protocol, resources);
  it is an aspiration, not a SOTA claim and not a pass/fail gate.

## Deliverables

Per lane: summary.json + lifecycle ledger + `--val` renders/stats (PSNR, SSIM,
real LPIPS) + tensorboard events + checkpoint-aligned qualitative panels
(3 windows x checkpoints 1000/3000/6000) with evidence/lifecycle overlays for
L0, L3, L5 minimum. A results wiki page records all lanes, activation
verdicts, the iteration history, and negative results at their evidence tier.
