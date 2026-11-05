# Question: 36 pending Slurm jobs on a saturated cluster — leave them, cancel and resubmit one at a time, or something else?

Give your own independent recommendation, with reasons. Use web search on Slurm documentation (priority/multifactor,
backfill scheduling, scontrol update of pending jobs, per-user backfill limits) where it settles a point. Say plainly what
you would do now and what you would do if they still had not started by tomorrow morning. Be concrete and short.

Facts (CINECA Leonardo, partition boost_usr_prod, account euhpc_d36_068, QOS boost_qos_lprod):
- 36 one-GPU jobs submitted 2026-10-01 16:54: 27 jobs with --time=02:00:00 (measured run ~1:25: base 1:10 + ~15 min
  extra), 9 jobs with --time=03:40:00 (measured run ~2:36-2:46). cpus=8, mem 64G / 120G, gres/gpu=1, --exclude one node.
- At 19:27 all 36 are PENDING, reason Priority. No dependency, no QOS/account limit reason.
- Priority now ~132,180 (QOS 120000, fairshare 11251 [fairshare factor 0.45], age ~900 and growing, job size 17).
  sprio weights: AGE 20000, FAIRSHARE 25000, JOBSIZE 10000000, QOS 300000. PriorityFavorSmall=no.
- Partition: 3,246 nodes allocated, 18 idle; 11,366 jobs pending, ~1,860 running. Our rank ~8,690 of 11,366.
- Of the 400 most recently started jobs in the partition: priority min 130,710, median 142,846, max 266,124 (some
  lower than ours start, presumably via backfill).
- Earlier on 2026-09-23 on the same machine: three 1-GPU 12 h jobs sat at rank ~9,425 and would not start; cutting
  walltime 12 h -> 8 h got all three backfilled within ~13 min. boost_qos_bprod refuses small jobs (min 65 nodes);
  normal and lprod QOS have the same priority; boost_qos_dbg allows 2 jobs, 30 min cap (jobs need 1.5-3 h).
- On other days identical jobs started within seconds of submission (lighter load).
- Constraints: a run that hits its time limit is lost (resume is unusable for both codes). A cancelled job that had
  already started may leave a run directory that the launcher then refuses (needs a new run id). The principal prefers
  slow starts to cancel/resubmit, and asked for your view before any change.
- My current recommendation: do not cancel; keep all 36 queued (resubmitting resets age; per-job priority does not
  depend on how many are queued; backfill benefits from many candidates). The one lever without resubmission is lowering
  the time limit of pending jobs in place (scontrol update JobId=... TimeLimit=...): 2:00 -> ~1:40 and 3:40 -> ~3:15,
  but I would not do it yet because the margin shrinks and a timeout loses the run.

Question: is that right? What would you do?
