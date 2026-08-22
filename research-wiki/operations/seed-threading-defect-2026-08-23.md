# INSTRUMENT DEFECT — the submission wrapper's `--seed` never reached
# the trainer (2026-08-23)

Append-only. Affects the LABEL of every "multi-seed" claim in this
repository's Apollo era. **It retracts no measured number.**

## 1. The defect, verified in source

`main.py` accepts `--seed` (default **6666**, `main.py:2144`) and applies
it at `main.py:2190` via `setup_seed`, which sets
`torch.manual_seed`, `torch.cuda.manual_seed_all`, `np.random.seed` and
`random.seed` (`main.py:2115-2119`).

`scripts/submit_apollo.py` builds the trainer's argv in
`_build_entrypoint_args`:

```python
if entrypoint_script == "main.py":
    tokens = ["--config", config_path, "--model_path", run_dir, *extra_args]
```

**`--seed` is not in that list.** The wrapper's `--seed` argument feeds
only the run manifest, the generated run id, and the ledger line.

Verified for the recorded ladder cells: `ladder_b1_crb_s1`'s manifest
records `seed: 1`, and its full wrapper argv contains
`--extra-arg=--source_path`, `--extra-arg=--test_iterations` and nothing
else. Neither `configs/n3v/ladder_b0_crb.yaml` nor
`configs/n3v/ladder_b1_crb.yaml` contains a `seed` key, so no config
merge supplied one either.

**Conclusion: every ladder cell — B0 s0/s1, B1 s0/s1, B1-D s0/s1 —
trained at seed 6666.** The `_s0`/`_s1` suffixes distinguish run
identities, not random seeds.

## 2. What this does and does not change

**Does NOT change:** any measured value. B0 read 34.0742 and 33.8068;
those numbers stand, as do every paired delta computed from them. The
runs genuinely differed.

**DOES change the label.** The quantity repeatedly cited as "the
measured B0 SEED spread, ±0.28 dB" is not a seed spread. It is
**run-to-run variation at a FIXED seed** — a reproducibility spread. Any
text describing the ladder as "two paired seeds" is describing two
replicates at one seed.

**The scientific force of the comparisons is largely preserved, and in
one respect improved.** A paired delta between two arms that differ in
one flag, each run twice, still measures the operator against the
instrument's own run-to-run noise — which is exactly the comparison the
gates used. Replicates at a fixed seed arguably bound that noise more
tightly than different seeds would. What is lost is the ability to claim
robustness ACROSS seeds, which no recorded result should now assert.

## 3. The tension this exposes, and it is unresolved

[[renderer-integrity-admission-2026-08-18]] Appendix C records three runs
of the repaired kernel **at fixed seed agreeing to 3.3e-4 dB** of
held-out PSNR. If the ladder cells were also at a fixed seed, on the same
repaired kernel, with a verified byte-identical training path, they
should have agreed to a similar tolerance. **They differ by 0.267 dB —
roughly 800× that figure.**

Both measurements stand; they were taken in different configurations.
The plausible mechanism, recorded as INFERENCE and not verified here:
this protocol runs 600k primitives with active densification, so
float-order nondeterminism in the backward accumulation can flip a
densification threshold, which changes the point set, which compounds
over thousands of iterations. The DiVa-360 configuration behind the
3.3e-4 figure did not have that amplifier in the same form.

**The actionable consequence is a floor, not an explanation: at the
50-frame N3V protocol with densification active, run-to-run variation is
≈0.27 dB, and no effect smaller than that is resolvable by two runs.**
The reproducibility figure of 3.3e-4 dB must NOT be transported to this
protocol.

## 4. The repair, applied from this block forward

New training cells pass the seed explicitly:
`--extra-arg=--seed --extra-arg <N>`, which lands in `main.py`'s argv
through the `*extra_args` splat above. The 2026-08-23 flow screen is the
first family to use it, so its two arms per cell are genuinely different
seeds.

**Consequence for comparator reuse, and it is a real cost:** a cell that
passes an explicit seed is NOT seed-matched to a historical cell that
silently used 6666. This is precisely why the flow screen trains its own
plain-B1 comparator on the same pool rather than reusing experiments
197/200 — a decision taken for the pool confound, which now turns out to
have been necessary for the seed reason as well.

`scripts/submit_apollo.py` is deliberately NOT modified to inject the
seed automatically. Doing so would silently change the meaning of every
future submission that omits it, and would make new cells
non-reproducible against the historical family without any signal. The
explicit `--extra-arg` form makes the seed visible in the argv the
manifest records.
