# OPERATIONAL AMENDMENT — 2026-08-23 block schedule and budget ceiling

Append-only. Recorded BEFORE any new result of this block was read, and
before any cell of this block was frozen or submitted. This is a
**user-directed schedule change**, not a reinterpretation of prior
evidence.

## 1. What is superseded, and what is not

[[block-2026-08-20b-decisions]] decision 5 proposed the next 300-frame
paired comparison (B0-R vs B1, 2 paired seeds) on a **NEW frozen
18,000-iteration schedule** (densify 500→12,000, lr horizon 18,000, cap
600k, endpoint `chkpnt18000`, ≈35 H100 slot-h), justified by the
measured twin convergence curves in
[[b0c-canonical-300f-results-2026-08-21]] Appendix B.

**That proposal remains HISTORICAL and its evidential basis is
untouched.** The measured curves stand exactly as recorded: both the
capped and uncapped 300-frame arms peak at ~12,000 iterations
(~4.2 presentations/unit) and then lose PSNR to densification churn
plus late train-view overfit, with the correction that the decline is
NOT caused by the 600k cap.

**The 18k schedule is NOT AUTHORIZED for this block.** No inference in
this amendment revises the curves, the peak location, the churn
attribution, or the capacity comparison. Only the authorization
changes.

## 2. The binding schedule ceiling for this block

By direct user instruction, every experiment frozen or submitted in
the 2026-08-23 block obeys:

1. **No experiment may exceed 12,000 training iterations.**
2. **6,000 iterations is the DEFAULT** for synthetic fixture training,
   50-frame N3V screens, and engineering/mechanism-admission cells.
3. **12,000 is an absolute ceiling, not a default.**
4. A 12k run may be submitted only after its scientific question,
   matched controls, gates, expected cost, and *unique information
   beyond the 6k screen* are all frozen.
5. **No 18k, 24k, 30k or 36k training continuation is authorized.**
6. **No "settle" continuation beyond 12k.**
7. **Hard ceiling on new GPU work this block: 24 total GPU slot-hours**
   across Apollo pools.
8. A required two-seed comparison may NOT be reduced to one seed to fit
   the budget. Defer the whole comparison instead.
9. The budget is a ceiling, not a target.

## 3. Consequence for the deferred 300-frame comparison

The B0-R vs B1 300-frame question is not withdrawn and not answered.
If it is promoted in this block it runs at a **12,000-iteration frozen
primary endpoint** with the 6,000 checkpoint additionally saved and
evaluated as an early **descriptive screen** (never a second primary
endpoint), densification confined to the frozen ≤12k boundary, and no
continuation phase. Four cells (2 arms × 2 paired seeds) must fit the
remaining budget or the comparison defers whole.

Any future block wishing to run the 18k design must re-freeze it as a
new specification under its own authorization; this amendment does not
pre-authorize it.

## 4. Evidence label

Operational/authorization record. Contains no scientific measurement
and licenses no scientific claim.
