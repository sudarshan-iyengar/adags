# OPERATIONAL RESULT — the ImViD bulk transfer is rate-limited PER-IP, and
# ~62 GiB is what one host got before it tripped (2026-08-24)

EXPLORATORY, zero GPU slots throughout. Records a measured operational
constraint on acquiring the 1.10 TiB anonymous ImViD release, so a future
block does not rediscover it.

## 1. What was acquired before the limit

| | |
|---|---:|
| verified complete files | **21** (15 `scene1_opera`, 6 `scene5_rendition`) |
| bytes landed | **62.149 GiB** of 1,099.96 GiB (**5.7%**) |
| partial files left behind | **0** |
| stale locks left behind | **0** |
| corrupt or HTML-masquerading files written | **0** |

Every one of the 21 is byte-count verified against the frozen inventory
and SHA-256 recorded in the append-only manifest. One was independently
re-hashed on Apollo and matched.

## 2. What happened, measured

Two zero-slot workers ran from 22:57. At **23:21:33** and **23:21:37** —
four seconds apart — both received an **HTML body** (`content-type:
text/html; charset=utf-8`) where file bytes were expected, and both
stopped cleanly. Aggregate throughput to that point was **~42 MB/s
sustained for ~24 minutes**.

**The fail-closed design did its job.** An HTML body on a bulk endpoint is
treated as a refusal and never as data, so no 6 KB HTML file was written
under an `.mp4` name, no partial was deleted, and every completed byte
survived.

## 3. The diagnosis — it is PER-IP, and that is measured, not assumed

At **23:44**, twenty-three minutes after the trip, a single 1-byte Range
probe was issued **from a different host** (the workstation) against
`scene1_opera/cam20.mp4` — a file that had never been downloaded:

```
HTTP/1.1 206 Partial Content
Content-Type: video/mp4
```

At **23:56**, thirty-five minutes after the trip, the same request **from
Apollo** still returned HTML.

So:

* the release is **intact and still world-readable** — nothing about the
  data or the folder changed;
* the limit is **not per-file** (an untouched file was refused to Apollo
  and served to the workstation);
* the limit is **not account-wide** (there is no account — this is
  anonymous access);
* the limit is **tied to the requesting host/IP**.

**One 1-byte probe was used for this diagnosis, not a retry loop.** The
acquisition rules forbid hammering Drive, and characterizing a limit is
not the same as testing whether it has lifted.

## 4. A decision of mine that the measurement corrected

Partway through I added a **second** concurrent worker to improve
throughput, on the reading that the directive permits "one or two
simultaneous large files".

**It bought ~13% and probably cost the transfer.** Aggregate went from
~37 MB/s (one worker) to ~42 MB/s (two), and the two tripped the limit
within four seconds of each other. The acquisition rules state that
parallel Drive transfers "increase quota and throttling risk without
improving scientific throughput"; the measurement agrees with the warning
and my judgement was the weaker one. **The resume runs a single worker.**

## 5. What changed in the tool

`scripts/fetch_imvid_release.py` 1.0.0 → **1.2.0**:

* **self-healing quota backoff.** A refusal no longer exits; it waits on
  an escalating schedule — **900, 1800, 3600, 3600, 7200 s = 4.75 h of
  total patience** — and resumes from the recorded offset. Escalating
  rather than fixed because the reset horizon is unknown and a fixed short
  retry is exactly the hammering the rules forbid.
* **`--sleep-between-files`**, to lower the sustained request rate
  deliberately. The resume uses 30 s, taking ~37 MB/s to ~26 MB/s.
* **stale-lock recovery** (1.1.0): a lock older than 2 h is stolen. Found
  by reasoning rather than by failure — the lock is released in a
  `finally`, which `SIGKILL` skips, so a killed worker would have blocked
  one file **permanently and silently**, and the transfer would have
  reported success with a hole in it.

## 6. What is NOT established, and the open question that matters

**Whether the limit is a RATE (bytes per unit time) or a VOLUME cap
(bytes per day) is not determined**, and it decides how long the full
release takes:

* if it is a **rate** limit, the paced single worker should stay under it
  and the remaining ~1.04 TiB completes in roughly a day of wall-clock;
* if it is a **daily volume** cap near the observed ~62 GiB, then **no
  pacing helps**, the ceiling is ~62 GiB per host per day, and the full
  release needs **~18 host-days** — or a different acquisition route.

The backoff distinguishes these for free: if the first 900 s retry
succeeds it is a short rate limit; if all five escalating attempts fail
across 4.75 h it is a long-horizon cap. **Do not conclude which it is from
this page — read the resume log.**

**Explicitly NOT done, and it is forbidden by the block directive:**
routing bulk bytes through the workstation. The workstation can still
reach Drive, but the directive states the workstation must not store or
stage the dataset, so that capability is recorded as a diagnostic fact and
not used as a transfer route.

## 7. Consequence for the science

**None of the completed science depends on the missing bytes.** The
calibration gate, the sparse initialization and the loader admission all
run on the 300-frame sample already on Apollo, and all three passed. What
the shortfall blocks is the **full-take** work: the fixed-rig test on a
complete take (which per the frozen event definition cannot be replaced by
metadata), the event census, and any ImViD training.

`scene1_opera` is **15 of 39** cameras complete, so it is not yet usable
even for a partial census — a census needs multi-camera support by
construction, and a 15-camera subset would bias which candidates can reach
the `C_min = 3` bar in a way that depends on which cameras happened to
download first.

---

## THE BACKOFF ANSWERED §6'S QUESTION (2026-08-24, append-only)

Section 6 left one question open and named the evidence that would settle
it: *"if it is a rate limit, the paced single worker should stay under it
... if it is a daily volume cap near the observed ~62 GiB, then no pacing
helps."* It said to read the resume log rather than assume.

**The log now reads, and it is not a short rate limit.**

| attempt | time (UTC) | outcome | next backoff |
|---:|---|---|---:|
| resume | 23:56 | refused (HTML) | 900 s |
| 1/5 | 00:11 | refused | 1800 s |
| 2/5 | 00:41 | refused | 3600 s |
| 3/5 | 01:41 | refused | 3600 s |
| 4/5 | 02:41 | refused | 7200 s |
| 5/5 | ~04:41 | *pending at handover* | — |

**Five refusals across 3 h 20 min of elapsed time since the 23:21 trip**,
with a single paced worker at ~26 MB/s and inter-file pauses. A burst-rate
limit would have cleared inside the first 15-minute wait; none of five
escalating waits cleared it.

**The supported reading: a long-horizon cap, most plausibly ~24 h, on the
order of the ~62 GiB already taken.** Stated as the supported reading, not
as a measured constant — this design can distinguish *short rate limit*
from *not a short rate limit*, and it cannot measure the cap's exact size
or reset period.

**What follows for cost, and it is the §6 branch that now applies:** no
pacing helps within a day, the ceiling is on the order of **~62 GiB per
host per day**, and the remaining ~1.04 TiB implies roughly **17 further
host-days** on a single host — or a different acquisition route.

**Every completed byte is preserved and the transfer remains correctly
resumable.** 21 files verified, zero partials, zero stale locks. The
downloader skips completed files by byte count, resumes partials by
`Range`, and steals locks older than 2 h, so a later relaunch of the exact
same command continues without any manual reconciliation.

**Not established, and not to be inferred from this:** the cap's exact
size, its reset period, whether it is per-IP or per-subnet, and whether a
different host would fare better. The only cross-host evidence is a single
1-byte probe, which shows the *release* is readable elsewhere — it does
not show that another host could sustain a bulk transfer.
