## Verdict

SOUND WITH QUALIFICATIONS

Reviewed on my own judgment; no ARIS skill was referenced or used.

## Confirmed defects

- **The queue-count argument is too absolute.** The prompt says “per-job priority does not depend on how many are queued” and “backfill benefits from many candidates” (lines 23–24). Slurm’s [resource-limit documentation](https://slurm.schedmd.com/resource_limits.html) explicitly allows limits on pending jobs “able to accrue age priority.” Its [configuration reference](https://slurm.schedmd.com/slurm.conf.html) also limits jobs **examined** through `bf_max_job_user`, `bf_max_job_user_part`, and `bf_max_job_test`. **Fix:** say queued-job count has no direct standard multifactor term, but accrual and backfill limits can make additional candidates ineffective. Leonardo’s actual settings are needed.

- **“The one lever without resubmission” is false.** [Slurm’s `scontrol` reference](https://slurm.schedmd.com/scontrol.html) supports pending-job updates including `QOS`, memory requirements, and `TimeMin`. [The `sbatch` reference](https://slurm.schedmd.com/sbatch.html) explains that a minimum time permits backfill to shorten the allocation when advantageous. **Fix:** describe reducing `TimeLimit` as one option; assess other updates against permissions and measured requirements. `TimeMin` still exposes the run to a shorter timeout.

## Suspected issues

- **The proposed shorter limits lack demonstrated runtime coverage.** [The Ex4DGS submission script](/D:/adags-cer/agent/submit_cer_ex4dgs.sh:46) describes “base 1:10 + begin/gates 10-15 min”; [the Multi4D script](/D:/adags-cer/agent/submit_cer_multi4d.sh:56) describes a base estimate plus overhead and a 1.3 multiplier. These are insufficient to establish safe limits across all scenes, repetitions, startup, and rendering. Full-job elapsed measurements would settle this.

- **The debug-job limit has conflicting official sources.** CINECA’s [Leonardo guide](https://docs.hpc.cineca.it/hpc/leonardo.html) says “Max 2 job running and/or pending,” while its [QoS update notice](https://www.hpc.cineca.it/newsletter/leonardo-updated-limits-of-qos_dbg-and-lrd_all_serial-qos/) says the maximum was reduced to one. Live QoS configuration would settle it; neither source establishes that debug is usable for these long jobs.

## Unverified claims

- The 19:27 states, priorities, weights, node counts, rank, 400-job statistics, September 23 timing, and previous immediate starts lack raw scheduler/accounting evidence in the files examined. [The plan log](/D:/adags/research-wiki/operations/ars-fast-path-plan-2026-09-23.md:1314) records only that all 36 were pending at submission.
- Lower-priority starts being caused by backfill remains an inference.
- Resume refusal is supported: both training implementations contain “CER runs are one uninterrupted job” assertions ([Ex4DGS](/D:/adags-cer/ex4dgs/train.py:65), [Multi4D](/D:/adags-cer/multi4d/train.py:291)).

## Missing

- **Account-wide concurrency:** CINECA documents lprod’s “8 nodes / 32 GPUs” maximum per project account. Its current consumption matters. A displayed `Priority` reason does not exclude another blocker: [Slurm documents that only the encountered reason is displayed](https://slurm.schedmd.com/job_reason_codes.html).
- **Concrete morning action:** leave all 36 queued now. If still pending tomorrow, inspect backfill limits, account/QoS usage, reservations, and [start estimates](https://slurm.schedmd.com/squeue.html), then ask support about the identified bottleneck. Do not shorten limits solely because another night elapsed.
- **Exact update syntax:** any eventual change should use `01:40:00` or `03:15:00`; `scontrol` interprets two-field times as minutes:seconds.