# Block 2026-08-23 — verified live state, comparator validity, budget plan

Operational record. Every row below was verified from a tool result in
this block, not inherited from a handover. Schedule ceiling:
[[block-2026-08-23-schedule-amendment]].

## 1. Repository state (verified)

| item | value |
|---|---|
| branch | `apollo/csvl-vpl-v2-exploratory` |
| local HEAD | `a5fb0e0129cbda0b4822a2f259f36ececde834a7` |
| `origin/apollo/csvl-vpl-v2-exploratory` | `a5fb0e0129cbda0b4822a2f259f36ececde834a7` |
| divergence | none (identical) |
| working tree | clean inside the execution set; untracked user/transient files only |

Untracked and DELIBERATELY UNTOUCHED: `agent-control/`,
`research-wiki/deep-dive-prompt.txt`, `research-wiki/run-deep-dive.ps1`,
the two Obsidian paste images, `supervisor-brief-2026-08-20.md`,
`sync 21-08-2026.md`, the seven `overnight-handover*.md`,
`runs-metrics-survey-2026-08-19-full.csv`,
`experiment_71_trial_71_logs.txt`.

## 2. Experiment ledger and cluster (verified)

* Ledger `agent-control/elgs-apollo/experiment-ledger.jsonl` ends at
  **experiment 232** (`b0c_uncap_eval36k`). Probed live: 229, 230, 231,
  232 all `STATE_COMPLETED`; **233 and 234 do not exist**. Next free
  experiment id is **233**. Nothing of this project is running.
* `det` reachable; CLI 0.38.1 vs master 0.38.0 skew warning on stderr is
  expected and is not a failure signal.
* **Pool capacity CHANGED from the recorded value.** `det slot list`
  shows `hopper` = **3 × NVIDIA H100 PCIe, all FREE**, and `dgx` =
  **2 × Tesla V100-SXM2-32GB on a single agent**, slot 0 occupied by a
  foreign `Command (strictly-pleased-coyote)` task registered
  2026-08-22T13:04:54Z, slot 1 free. The execution-authority page
  records dgx at 8 slots (2026-08-11 probe). **dgx is now 2 slots.**
  The foreign command task is NOT this project's and was not cancelled.

## 3. Admitted images (verified from the admission record)

| pool | image | digest |
|---|---|---|
| `dgx` (V100) | repaired V100 | `sha256:70a28e3d46a4768d595bda67328ddaacd18ae3af98ae863fc3f7303c159bb683` |
| `hopper` (H100) | `apollo-h100-88ee245` | `sha256:0d5771688c9b6580f70133f813b7a4110bd5c967920afe3c5fd1856bb098800e` |

## 4. Comparator validity for the ladder B1 arms — VERIFIED, not assumed

The B1-D page verified an EMPTY training-path diff against the ladder
comparator commits *at that time*. That is no longer true: since
`22b2dd6`/`a798949`, `scene/packet_birth.py` has gained **179 lines**
and `arguments/__init__.py` **4 lines** (the B1-D donor-mask feature).
Re-verified line by line this block:

* `arguments/__init__.py`: purely additive, one field
  `packet_birth_dynamic_mask_donors = False`.
* `scene/packet_birth.py`: exactly **4 lines removed**, all of them
  provably behaviour-preserving when the flag is off —
  (a) `k = int(float(fraction) * count)` replaced by
  `requested_donor_count(count, fraction)`, whose body is
  `int(float(fraction) * int(rows))` with `count` already an `int`
  (exactly equal); (b) `return torch.argsort(score)[:k]` now sits behind
  `if eligible is None:` — the identical statement on the B1 path;
  (c) a defaulted parameter added to `select_packet_donors`; (d) the
  call site passes `eligible=eligible`, which is `None` whenever
  `dynamic_mask_donors` is False.
* The added `donor_dynamic_mask_eligibility` path is entered ONLY when
  the flag is True, and consumes **no RNG**, so the global RNG stream
  that `sample_residual_sites`'s `torch.multinomial` draws from is
  unchanged on the B1 path.

**Conclusion: reuse of ladder B1 (experiments 197/200) as a plain-B1
comparator is VALID with respect to code drift**, provided any new
flag added this block is likewise proven flag-off bit-identical.
A POOL confound is a separate matter — see §5.

Independently corroborating: the route-init replicate check measured
0.018 dB across ~10 intervening commits
([[route-init-screen-2026-08-20]] Rule 1).

## 5. The pool confound this block must not create

Ladder B0/B1/B1-D all ran on **dgx**. Only **1 dgx slot** is now free
while **3 hopper slots** are free. Running new flow arms on hopper and
comparing them to dgx-trained plain-B1 would confound the flow variable
with a pool/GPU-architecture change — impermissible under the frozen
one-variable discipline, and a pool switch is required to be a new
ledger entry, never silent.

Therefore, if the flow screen runs, **every arm it compares runs on the
same pool**, including a freshly trained plain-B1 on that pool. The
decisive `B1-F vs B1-X` contrast is internally pool-consistent by
construction in either case.

## 6. Budget plan (ceiling 24 GPU slot-hours, from the amendment)

Measured unit costs from the prior blocks: 50-frame 6k training cell
≈ 2.6 slot-h on dgx (ladder 195-206); `--val` eval ≈ 0.15 slot-h; LRV3
fixture 6k training ≈ 1.1 slot-h; a consolidation/falsification pass
≈ 0.15 slot-h; 300-frame capped training measured 15.9 slot-h at 36,000
iterations ≈ **1.585 s/it averaged**.

**The conditional 12k 300-frame promotion is DEFERRED, decided on
arithmetic before any lane result was read.** At 12,000 iterations one
300-frame capped cell costs ≈ 4.7-5.3 slot-h (the point cap saturates
by iteration 6,000, so per-iteration cost is near-constant after that).
The required design is 2 arms × 2 paired seeds = **4 cells ≈ 19-21
slot-h**, i.e. 79-88% of the entire block ceiling, before evaluation.
That displaces the payload and timing lanes, which are the actual paper
blockers, and promotion condition 3 forbids exactly that. Reducing to
one seed is forbidden by amendment rule 8. **The comparison therefore
defers whole; its specification is frozen this block so it is
launch-ready under a future budget.**

Planned allocation of the 24 slot-h ceiling, priority-ordered:

| lane | content | projected slot-h |
|---|---|---:|
| P | payload headroom diagnostic + starved-fixture build/train + payload edit with controls | ≈ 2-3 |
| T | non-oracle episode-timing estimator + comparison cells | ≈ 4-6 |
| F | flow-assisted B1 screen, pool-consistent arms | ≈ 7-11 |
| M | 300-frame mask + packet schedule freeze | **0 (CPU/read-only)** |

Lane F's required outcome is satisfiable either by execution OR by an
implementation-complete, admitted launch packet; it yields GPU
precedence to P and T if the ceiling binds.

## 7. Workstation test environment — measured, and it constrains verification

The project's own `adags` conda environment is BROKEN on this
workstation (`numpy` fails its C-extension import), and the default
`python` has no torch at all. The only usable test interpreter found is
`C:\Users\sucar\anaconda3\envs\DVS-Voltmeter\python.exe`
(torch 2.6.0+cpu, numpy 2.2.3); `pytest`, `omegaconf`, `pyyaml`,
`plyfile` and `pillow` were installed into it this block purely as test
tooling. No repository file and no other environment was changed.

**`pointops2_cuda` is a compiled CUDA extension that cannot be built
here.** Anything importing `scene.gaussian_model`, `scene.cameras`, or
`scene/__init__.py` transitively pulls it in and fails at COLLECTION.
Nine test files are affected, including the pre-existing
`tests/test_packet_birth.py` — so the "14 CPU tests" recorded for B1-D
were never runnable on this workstation and must have been executed
elsewhere.

**Measured baseline before any change of this block** (whole suite minus
the nine CUDA-blocked files): **954 passed, 44 failed, 74 skipped, 39
errors**. Those failures and errors are PRE-EXISTING and environmental;
they are recorded here so that any increase attributable to this
block's changes is detectable. Per-file baselines that matter:
`tests/test_falsify_b2_edit.py` = **21 passed** (matching the "21 CPU
tests" recorded with the B2 falsification).

Consequence for verification: unit tests whose logic is pure arithmetic
are required to be written so they run locally, and end-to-end
validation of anything touching the renderer or the Gaussian model is
done by a bounded ADMITTED-IMAGE PREFLIGHT rather than by container
pytest. The precedent is `configs/n3v/ladder_b1_preflight.yaml` at
1,200 iterations — enough to cross iteration 1,000, where the first
packet-birth event fires.

## 8. Submission plan and pool assignment

LRV3-derived cells run on **dgx** with the admitted V100 image, matching
the pool and image of the checkpoints they read (experiments 184 and
209) and of the recorded DC falsification (experiment 213), so the new
payload numbers stay comparable to the falsified DC numbers. dgx has one
free slot, which is sufficient because these cells are minutes long and
run serially. **hopper**'s three free slots are reserved for the flow
screen, whose arms must all share one pool.

| order | cell | pool | projected slot-h |
|---:|---|---|---:|
| 1 | payload headroom screen (exp 209 checkpoint) | dgx | ≈ 0.2 |
| 2 | opacity payload edit + L1/L2/L3 controls | dgx | ≈ 0.3 |
| 3 | non-oracle episode-timing screen (exp 184 checkpoint) | dgx | ≈ 0.3 |
| 4 | flow preflight, 1,200 iterations | hopper | ≈ 0.35 |
| 5 | flow screen, 6 cells (conditional) | hopper | ≈ 11 |

## 9. An execution-closure GAP, found while submitting (pre-existing)

`check_execution_closure` refuses a submission when anything under
`EXECUTION_DIRS` (`elgs`, `scene`, `gaussian_renderer`, `utils`,
`arguments`, `depth_visibility`) or `EXECUTION_FILES` (`main.py`,
`scripts/submit_apollo.py`, `det_exp_apollo.yaml`) plus the named config
is dirty. It worked exactly as intended this block — it correctly blocked
the flow screen while `elgs/trainer_hooks.py` was mid-edit.

**But `scripts/` is not in either set, apart from `submit_apollo.py`.**
Every other allowed entrypoint — `scripts/falsify_b2_edit.py`,
`scripts/payload_headroom.py`, `scripts/estimate_episodes.py`,
`scripts/consolidate_packets.py`, and the rest of
`ALLOWED_ENTRYPOINT_SCRIPTS` — executes in the container from the
`git archive <commit>` snapshot, yet a DIRTY working copy of one of them
does not block a submission that uses it as the entrypoint.

The consequence is not a wrong result but a silently misleading one: the
container would run the COMMITTED version while the operator is looking
at edited source, and nothing would say so. That is exactly the failure
the closure check exists to prevent for `main.py`.

**Not repaired this block** — changing the closure set mid-block would
alter the admission behaviour of every cell already submitted, and the
repair deserves its own review. Recorded here so the next block can
decide deliberately. **Mitigation used in the meantime: every entrypoint
script this block executed was committed and pushed before its cell was
submitted, and each run's manifest records the commit that actually
ran.** The three `scripts/` files left dirty at the end of this block
(the LRV4 fixture work) were never the entrypoint of any submitted cell.
