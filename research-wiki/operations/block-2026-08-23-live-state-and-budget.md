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
