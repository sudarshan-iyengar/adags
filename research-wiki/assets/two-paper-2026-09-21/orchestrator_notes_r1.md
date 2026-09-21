# Orchestrator notes on Fable round 1 (2026-09-21)

Independent checks made by the orchestrator (Claude Opus 5) on Fable's round-1
plan before it went to review. Read these alongside `fable_round1.md`.

## Verified — Fable was right

1. **The dual-submission rule, verbatim.** Fetched
   https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines independently. All
   four sentences Fable quoted match word for word:
   * "A publication, for the purposes of this policy, is defined to be a written
     work longer than four pages (excluding references) that was submitted for
     review by peers for either acceptance or rejection, and, after review, was
     accepted."
   * "arXiv preprints and university technical reports are _not_ considered as
     publications."
   * "peer-reviewed workshop papers _are_ considered as publications if their
     length is more than four pages (excluding references), even if they do not
     appear in a proceedings."
   * "...(defined as having 20 percent or more overlap)..."
   So **non-archival status does not exempt a >4-page workshop paper.** The
   2027 Author Guidelines are still HTTP 404, so the 2027 text remains
   unverified; the 2026 text is confirmed.

2. **PhysWorldAI exists as described.** Fetched its CFP independently. "Physical
   World AI: Geometry, Characteristics, and Multimodal Sensing", NeurIPS 2026,
   Atlanta. **Non-archival window Sept 29 – Oct 29, 2026, notification Nov 09,
   2026.** "Long papers up to 8 pages and extended abstracts up to 4 pages
   (Reference and Appendix are not included)." Topics include "3D/4D
   reconstruction ... physically grounded view synthesis." Note: its **archival**
   track closed Sep 09, 2026 and is not available.

## Corrected — Fable was wrong, and so was the brief

3. **The GPU-hour conversion.** The brief said "~124k GPU-hours"; Fable
   corrected it to "≈35k GPU-h at ≈3.5 local-h per GPU-h". **Both are wrong.**
   Measured today from Slurm accounting, not from a page:
   * every GPU job since 2026-09-18 22:00: 17 jobs, **20.91 GPU-h**,
     **167.26 billing-h**, ratio **8.00** (a 1-GPU, 8-CPU job bills
     `billing=8`);
   * `saldo` consumption rose 19,927 → 20,094 = **167 local-h** over the same
     window — matching the billing-hours exactly, so saldo's local-hours are
     billing units.
   * **1 GPU-h = 8 local-h. Remaining 123,906 local-h ≈ 15,500 GPU-h.**
   Still roughly 40× the ~400 GPU-h both papers need, so compute remains
   non-binding — but every budget sentence must use ≈15.5k GPU-h.

## Carried from earlier rounds

4. **The previous reviewer's upstream GitHub line numbers were wrong.** In the
   round-2 review of plan v0.3, citations such as `train_semantic.py#L1407` and
   `#L2079` pointed past the end of a ~550-line file, although the substance was
   right. Its local-file citations were exact. Re-locate any upstream line
   number in the pinned checkout before relying on it.
