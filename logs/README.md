# Scheduler logs

Slurm stdout and stderr belong in this directory, with the scheduler job ID in
each filename through `%j`.

Raw `*.out` and `*.err` files are ignored and must not be committed.
Record durable job IDs, exact configurations, summarized metrics, failures,
decisions, and claim boundaries in maintained manifests or `research-wiki/`.
