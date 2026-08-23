# Handover — block 2026-08-23 (self-contained)

Untracked by design. Durable content lives in `research-wiki/`; this file
is the operational bridge to the next block.

---

## 0. Repository state

| item | value |
|---|---|
| branch | `apollo/csvl-vpl-v2-exploratory` |
| local HEAD | `8a8610eacabe4a098a9691f875c31b6bec334471` (17 commits this block) |
| origin | identical to local; every commit pushed |
| divergence | none |

**Protected / untouched throughout:** `agent-control/`,
`research-wiki/deep-dive-prompt.txt`, `research-wiki/run-deep-dive.ps1`,
the two Obsidian paste images, `supervisor-brief-2026-08-20.md`,
`sync 21-08-2026.md`, the seven earlier `overnight-handover*.md`,
`runs-metrics-survey-2026-08-19-full.csv`,
`experiment_71_trial_71_logs.txt`. No force-push, no history rewrite, no
destructive cleanup, no reused claims.

---

## 1. What this block decided (read `operations/block-2026-08-23-decisions.md` first)

1. **Non-oracle episode boundaries are recoverable EXACTLY** —
   experiment 235, 0 frames of error on both boundaries, zero false
   activations, 99.52% abstention, 0.188 slot-h. The single largest
   scientific risk has its first positive.
2. **No consolidation payload has headroom on LRV3**, and the
   oracle-correct opacity edit is actively harmful (−1.19 dB). Per the
   frozen rule, **the representation-only pivot is the recorded
   recommendation.**
3. **The development scene contains essentially ONE clean
   occlude-and-return event on dynamic content in 300 frames.**
4. The 300-frame B0-R vs B1 comparison is **frozen and deferred** on
   budget arithmetic decided before any result was read.
5. Flow-as-birth-prior is implemented, its asset direction verified
   empirically, and its six-cell screen is **running**.
6. Two instrument defects found: `--seed` never reached the trainer, and
   `scripts/` sits outside the execution-closure set.

---

## 2. Experiments this block

| exp | cell | pool | state | cost (slot-h) |
|---:|---|---|---|---:|
| 233 | `lrv3_payload_headroom` | dgx | COMPLETED | 0.007 |
| 234 | `ladder_b1f_preflight` | hopper | COMPLETED | 0.42 |
| 235 | `lrv3_episode_estimate_t1` | dgx | COMPLETED | 0.188 |
| 236 | `lrv3_falsify_opacity` | dgx | COMPLETED | 0.009 |
| 237 | `ladder_b1_hop_s0` (plain B1) | hopper | see §3 | ~1.9 proj |
| 238 | `ladder_b1_hop_s1` (plain B1) | hopper | see §3 | ~1.9 proj |
| 239 | `ladder_b1f_crb_s0` (correct flow) | hopper | see §3 | ~1.9 proj |
| 240 | `ladder_b1f_crb_s1` (correct flow) | hopper | see §3 | ~1.9 proj |
| 241 | `ladder_b1x_crb_s0` (wrong flow) | hopper | see §3 | ~1.9 proj |
| 242 | `ladder_b1x_crb_s1` (wrong flow) | hopper | see §3 | ~1.9 proj |
| 243 | `lrv3_episode_program_v2` | dgx | COMPLETED | ~0.2 |
| 244 | `lrv3_falsify_opacity_l4` | dgx | COMPLETED | <0.02 |
| 245 | `lrv3_a_est_r0` (phase T2) | dgx | see §3 | ~1.5 proj |
| 246 | `lrv3_a_est_r1` (phase T2) | dgx | see §3 | ~1.5 proj |
| 247 | `lrv4_b1_packets` (starved fixture) | dgx | COMPLETED | ~1.5 |
| 248 | `lrv4_payload_headroom` | dgx | COMPLETED — **INVALID (n=1)** | <0.02 |

All at `--exploratory` (`evidence_bearing: false`). Claims consumed at
retry 0; **never reuse a consumed claim index.**

Images: dgx = `sudarshaniyengar/adags@sha256:70a28e3d46a4768d595bda67328ddaacd18ae3af98ae863fc3f7303c159bb683`;
hopper = `sudarshaniyengar/adags@sha256:0d5771688c9b6580f70133f813b7a4110bd5c967920afe3c5fd1856bb098800e`.

**Measured spend on terminal cells: 0.624 slot-h. Projected for the
running set: ≈11.6. Against the 24 slot-hour block ceiling that leaves
roughly 11.8 unspent.**

---

## 3. Monitoring and continuation

```bash
python scripts/det_monitor.py experiment --experiment-id 237
```

Poll every submitted id (237-243). Terminal states are
`STATE_COMPLETED` / `STATE_CANCELED` / `STATE_ERROR`. Logs:

```bash
python scripts/submit_apollo.py logs --experiment-id 237
```

On Windows set `$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"`
first or the log decode raises `UnicodeDecodeError`. A local log stream
exiting 255 after ~10 minutes is benign; the remote task continues.

**Run PowerShell, never Git Bash** — Git Bash rewrites `/apollo/...`
arguments.

### 3a. Flow screen (237-242) — when terminal

Evaluate each with `main.py --val` on `chkpnt6000.pth`, then score event
regions with `scripts/event_ray_metrics.py` against the FROZEN
`configs/n3v/ladder_event_masks_crb0_49.json`. The gates are frozen in
`operations/b1f-flow-birth-prior-spec-2026-08-23.md` §5. **The decisive
comparison is B1-F vs B1-X**: if correct flow does not beat wrong flow on
the event endpoint, the result is UNATTRIBUTABLE and the flow birth prior
is rejected regardless of how B1-F compares to plain B1.

The established evaluation path, matching how the ladder was scored:
submit a `main.py --val` cell per arm on its `chkpnt6000.pth` (it writes
renders and gt under `<run_dir>/test/ours_6000/`), `rclone copy` those
two directories to `data/synthetic/ladder_eval/<arm>_renders` and
`.../gt`, then run the CPU-only scorer locally:

```bash
python scripts/event_ray_metrics.py --renders_dir data/synthetic/ladder_eval/b1fs0_renders --gt_dir data/synthetic/ladder_eval/gt --masks configs/n3v/ladder_event_masks_crb0_49.json --out data/synthetic/ladder_eval/b1fs0_event.json
```

`scripts/event_ray_metrics.py` is NOT in `ALLOWED_ENTRYPOINT_SCRIPTS` and
does not need to be — it runs on the workstation. Budget ~60 MB of render
pull per arm; that is a deliberate transfer, not a routine one. Sanity
check: the existing `data/synthetic/ladder_eval/b1s0_event.json`
reproduces the recorded B1 s0 event union of 32.0299.

Read Appendix C before interpreting: the 0-49 masks are now known to
score mostly static pixels, so the screen carries a raised false-negative
risk. A null is a pre-registered likely outcome (Appendix A: p50 flow
0.06 px), and the wrong-flow control is what makes a null terminal for
the zero-acquisition BIRTH prior rather than leaving it open.

### 3b. Phase T2 (A-est) — the next thing to launch

Experiment 243 emits `estimated_program_v2.json` into experiment 184's
run dir
(`/apollo/users/sri/proj_adags/runs/elgs/20260820T002949Z_lrv3_a0_prime_0_b7952b0`).
Then, in order:

1. `rclone copy` that file to `configs/lrv3/estimated_program_v2.json`.
2. **Commit and push it** — code reaches the container only through
   `git archive <commit>`, so an uncommitted program file makes the run
   fail on a missing path.
3. Submit two A-est cells:

```bash
python scripts/submit_apollo.py submit --cell lrv3_a_est_r0 --pool dgx --image-ref sudarshaniyengar/adags@sha256:70a28e3d46a4768d595bda67328ddaacd18ae3af98ae863fc3f7303c159bb683 --config configs/lrv3/a1_est.yaml --seed 0 --retry 0 --exploratory --projected-gpu-hours 2 --extra-arg=--test_iterations --extra-arg 6000
```

Second replicate: `--cell lrv3_a_est_r1 --seed 1`.

**Two deliberate omissions, both verified against experiment 185's
manifest — do not "fix" them:**

* **No `--source_path`.** `configs/lrv3/a1_est.yaml` inherits
  `source_path: "/apollo/users/sri/proj_adags/data/synthetic/lrv3"` from
  `a1_local.yaml:54`, and A1-LOCAL passed no such extra-arg.
* **No `--extra-arg=--seed`.** A0′ (experiment 184) and A1-LOCAL
  (experiment 185) both ran at `main.py`'s DEFAULT seed 6666, because the
  wrapper's `--seed` never reached the trainer. Passing an explicit seed
  here would make A-est **not seed-matched to its own comparators**.
  These two cells are therefore **REPLICATES at seed 6666, not two
  seeds** — label them that way. Their spread is comparable to the
  recorded LRV3 same-arm run spread of 0.09-0.17 dB. The wrapper's
  `--seed 0`/`--seed 1` distinguishes the run ids and claims only.

This is the opposite of the choice made for the flow screen, and
deliberately so: that screen trains its OWN comparators, so it can and
does use genuinely different seeds; this one reuses recorded comparators,
so it must match their seed.

Score with `scripts/eval_lrv1_event.py` — **NOT** `main.py --val`, whose
PSNR convention moved after A0′/A1-LOCAL were recorded. Comparator
validity against experiments 184/185 is verified in
`operations/nonoracle-episode-timing-result-2026-08-23.md` §6.

**Watch the first lines of the run for `v2_group_rows`.** Recall shrinks
when the estimate is reapplied to the 50k init cloud; if that count is
very small the cell is UNDERPOWERED rather than negative, and those two
outcomes are not distinguishable from `event_return` alone. Seeding
raises rather than silently no-opping if the program gates zero rows.

### 3c. Work left with a worker mid-flight

A bounded worker was building the **LRV4 observation-starved fixture**
(1-frame return, `--allow-short-return`, scene id LRV4) and adding a
**within-recipient permutation control (L4)** to
`scripts/falsify_b2_edit.py`. If `scripts/build_synthetic_reveal_scene.py`,
`scripts/falsify_b2_edit.py`, `scripts/payload_headroom.py`,
`tests/test_falsify_b2_edit.py`, `tests/test_lrv1_scene_fixture.py` or
`configs/lrv4/` are dirty, that work did not land — **review it before
committing, and re-run the LRV3 byte-identity regression tests**, since
that file's fixture constants were being made fixture-driven.

---

## 4. What the next block should do first

**L4 already ran (experiment 244) and it overturned the attribution** —
see `operations/payload-headroom-result-2026-08-23.md` §7. Remaining, in
order:

1. **LRV4 RAN and returned an INVALID INSTRUMENT (experiment 248) — see
   `operations/lrv4-starved-fixture-result-2026-08-23.md`.** The screen
   found ONE recipient row (`row_sets_sufficient: false`), so the
   mechanism claim is UNTESTED, not refuted. **Do not read the 4.995 DC
   headroom ratio in that report — it is a one-pair statistic.** The
   next step is the PURE DIAGNOSTIC specified in §5 of that page: report
   the distribution of the support lower bound `lo` over rows whose
   support intersects the return window, BEFORE any `lower_min` cut. It
   changes no decision rule and separates "threshold artifact" from "a
   one-frame return leaves nothing localized to transfer into". **Do NOT
   simply re-run LRV4 with LRV3's 9.3 scalar** — choosing a selection
   threshold because it yields a non-empty set, after seeing that the
   principled one does not, is the post-hoc adjustment this project
   forbids.
2. **Score A-est (245/246)** with `scripts/eval_lrv1_event.py`. The
   estimated program's gaps are numerically identical to the oracle's, so
   **this isolates MEMBERSHIP alone.** Watch `v2_group_rows` in the first
   lines of each run — see §3b on underpowered-versus-negative.
3. **Score the flow screen (237-242)** per §3a; B1-F vs B1-X is the
   decisive contrast.

### 3d. LRV4 headroom screen — after experiment 247 completes

With `$L4RUN` = experiment 247's run dir
(`/apollo/users/sri/proj_adags/runs/elgs/20260823T001524Z_lrv4_b1_packets_0_bbf1c4f`),
submit `scripts/payload_headroom.py` (already allowlisted) with
`--config configs/lrv4/b1_packets.yaml`,
`--oracle_region configs/lrv4/oracle_correct.json`,
`--start_checkpoint $L4RUN/chkpnt6000.pth`,
`--packet_state $L4RUN/packet_state.pt`, `--model_path $L4RUN`,
`--out_report $L4RUN/payload_headroom_report.json`, plus
`--gaussian_dim 4 --time_duration 0.0 10.0 --num_pts 50000
--force_sh_3d`.

**Integrity check on the output:** the report's `protocol.fixture` block
must read `LRV4 · return frames [59] · WR [9.8333, 9.8333] · probes
[2.5000, 9.8333] · scalars=derived`. **If it says `9.6000`, the fixture
was misread and the run must be discarded** — 9.6 is LRV3's frozen probe
and falls inside LRV4's absence gap.

The comparison that matters is LRV4's headroom ratios against LRV3's
recorded ones (`_features_dc` 1.429, `_opacity` 1.223 activated,
`_xyz` 1.092, …). LRV4's return supply is 18,978 held-out pixel-times
against LRV3's 56,934 — exactly one third, by construction.

---

## 5. Traps that will bite if forgotten

* **`--seed` does not reach `main.py` unless passed as
  `--extra-arg=--seed --extra-arg N`.** Every historical ladder cell ran
  at 6666. A cell that passes an explicit seed is NOT seed-matched to a
  historical one.
* **Run-to-run variation at this protocol is ≈0.27 dB.** Do not
  transport the 3.3e-4 dB reproducibility figure here; it was measured
  in a different configuration.
* **Never edit `configs/lrv3/*.yaml` comments.** The `a0`/`a1`/`a2`
  headers carry LRV2 timing and are 3 frames WRONG for LRV3 (true gap
  30-56, return 57-59). They are corrected in
  `operations/lrv3-fixture-hazards-2026-08-23.md` and must be left alone
  in the files, because YAML is hashed as raw bytes and a comment edit
  changes the config's content hash.
* **A `--dry-run` consumes a retry index.** Never delete or reuse a
  consumed claim.
* **`scripts/` is outside the execution-closure set.** A dirty entrypoint
  script will not block a submission that uses it; the container runs the
  committed version silently.
* **The 0-49 event masks are not confirmed by ground truth** and score
  mostly static pixels.
* **`_t` is not a payload.** Transferring it deletes the return.

---

## 6. Forbidden conclusions

* Nothing here licenses any N3V claim about consolidation. No N3V B2
  follows a negative — that is the frozen rule.
* The timing result licenses NO reconstruction claim. Nothing was
  retrained.
* The −1.19 dB may not be attributed to identity until L4 runs.
* "No payload has headroom" covers degree-0 SH only — `_features_rest`
  was never screened — and covers ONE payload FORM: per-row value
  replacement. Joint, residual and capacity-freeing payloads are
  unmeasured.
* No SOTA claim, no cross-scene claim, no claim from a single seed.

---

## 7. Commits this block (all pushed; branch `apollo/csvl-vpl-v2-exploratory`)

```
8a8610e Guard the headroom screen against misreading a fixture
9ad1f38 Freeze the LRV4 prediction and decision rule before any result exists
645d6a1 Propagate the L4 attribution reversal into the block record and query pack
2795d07 L4 permutation control OVERTURNS the opacity attribution
bbf1c4f Commit the estimated v2 episode program from experiment 243
662b3e2 LRV4 observation-starved fixture and the L4 permutation control
1d86995 Block 2026-08-23 decisions
72205fb Record an execution-closure gap found while submitting
8840f36 B1-F preflight: mechanism healthy, six-cell screen launched
789595a Phase T2: make an ESTIMATED episode program trainable
30d83f0 Record: the submission wrapper's --seed never reached the trainer
52b705b Query pack: 2026-08-23 block headlines
cf7bff3 Gap map: timing inference / consolidation payload
d38b72d Record adversarial-review corrections to the payload result
54a455c Verify phase T2 comparator validity
8cccf9a RESULT: non-oracle episode boundaries recovered EXACTLY
ca5c108 Freeze the flow preflight launch rule before reading its funnel
1eb4756 RESULT: no replacement payload has oracle-correct headroom on LRV3
f666dd7 Freeze the 300-frame event masks and record the curation findings
dd52e0e Freeze Lane P and Lane F specs
b4e5c16 Block opening records: schedule amendment, live state, LRV3 hazards
```

Base of the block: `a5fb0e0`.

## 8. Durable pages written this block

`operations/`: `block-2026-08-23-schedule-amendment`,
`block-2026-08-23-live-state-and-budget`,
`block-2026-08-23-decisions`, `lrv3-fixture-hazards-2026-08-23`,
`payload-headroom-spec-2026-08-23`,
`payload-headroom-result-2026-08-23`,
`nonoracle-episode-timing-spec-2026-08-23`,
`nonoracle-episode-timing-result-2026-08-23`,
`b1f-flow-birth-prior-spec-2026-08-23`,
`b1f-preflight-result-2026-08-23`, `crb300-b0r-b1-spec-2026-08-23`,
`crb300-event-mask-curation-2026-08-23`,
`seed-threading-defect-2026-08-23`,
`lrv4-starved-fixture-spec-2026-08-23`.
Plus `query_pack.md` and `gap_map.md` updated.

## 9. Workers used, and what each was reviewed on

Six bounded workers, all read-only or implementation-scoped, none
permitted to stage/commit/push/submit. Every result was verified by the
primary against a tool result before it entered the record: the payload
and timing audits (file:line claims spot-checked in source), the mask
curator (mask file structurally validated and run end-to-end through the
real scorer), the flow implementer (138 tests re-run independently, diff
read hunk by hunk), the T2 implementer (v1 byte-identity golden test and
13 untouched oracle tests re-run), the LRV4/L4 implementer (160 tests
re-run, fixture byte counts verified against the upload), and a
FRESH-CONTEXT adversarial reviewer of the payload result whose findings
were accepted and recorded rather than absorbed.

Three worker reports corrected the primary's own frozen specifications
before any output was read — the flow magnitude guard, the timing
contrast rule, and the T2 membership mode. All three are recorded as
amendments with their reasons.
