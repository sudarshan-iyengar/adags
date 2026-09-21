# Orchestrator notes on Sol round 1 (2026-09-21)

Independent checks on the review in `sol_round1.md` before it went back to
Fable. Attestation passed: Sol quoted `interpolations.py:58` verbatim and gave
Paper W's title exactly.

## Verified — Sol was right

1. **TRASE is public.** https://github.com/yunjinli/TRASE — MIT licence, with
   training code, rendering and evaluation code, Docker, and dataset-preparation
   scripts. Fable's "TRASE code availability UNVERIFIED" is resolved: **it is
   runnable.** Caveat: its README says the Mask-Benchmarks themselves were to be
   "released later" (see 2).

2. **Mask-Benchmark exists and is small.**
   https://huggingface.co/datasets/yunjinli/Mask-Benchmark — public, **not
   gated**, **~985 MB**, CC-BY-NC 4.0, five subsets including **Neu3D (1.5k
   rows)**. This resolves Fable's "Mask-Benchmark size UNVERIFIED".
   **Three precision issues it raises, which the plan must resolve:**
   * **Which Neu3D scenes it covers is not listed on the card.** If it does not
     cover `cut_roasted_beef`, the proposed Gate-A step "reproduce the published
     crb semantic number" is impossible as written.
   * **Images are 536×536 JPEG**, against Multi4D renders at 1352×1014. The
     mapping between them (crop? resize? which camera?) is unknown and is
     exactly the kind of silent mismatch that has cost this project before.
   * **Mask-Benchmark GT is for the ORIGINAL scenes, not the authored
     fixtures.** In the fixtures the object is removed for frames 60–89, so
     Mask-Benchmark cannot score return-censored J&F on them. The fixtures need
     their own masks — the project has DEVA and SAM 2 masks, and Sol requires
     *independently adjudicated* ones.

## Not verified by me

3. **Sol's ImViD claim** ("only Opera and Puppy are fully available and their
   structural inventory is complete") is cited from **Codex's own memory**
   (`MEMORY.md:181-192` in its store), not from any file in this workspace. Sol
   itself marked the Class-A event as unconfirmed. The workspace record
   (query_pack 2026-08-24 §4) says ImViD acquisition is rate-limited at ~62 GiB
   / 5.7%, 21 files verified complete, and the full Opera take is 15,215
   frames. **Treat ImViD event availability as unverified** until someone lists
   the files on Leonardo or Apollo.

4. **CVM 2027** (Oct 23 abstract / Oct 26 paper, up to 14 pages) — cited by Sol
   with a URL; not re-fetched, because Sol recommends against it anyway.
