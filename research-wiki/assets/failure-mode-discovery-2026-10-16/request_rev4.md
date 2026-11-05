# Short review request, round 3: failure-mode discovery plan, sections 12-14 (ADAGS, 2026-10-17)

You reviewed rev 1 and rev 2 of this plan (D:\adags\research-wiki\assets\failure-mode-discovery-2026-10-16\codex-review-rev1.md
and codex-review-rev2.md). The full current page is D:\adags\research-wiki\operations\failure-mode-discovery-plan-2026-10-16.md
(read it; sections 11-14 are new since your round 2). Section 12 (revision 4, the principal's decision) moves the claim
onto methods PUBLISHED for few cameras (4C4D at its published N3V cut cam01/10/13/20; GC-4DGS; Multi4D's own
low-view config arguments/nerfds.py), with Multi4D dense, Ex4DGS and STG as labelled reference rows, and costs a
~66 GPU-h first block that the principal has now approved, conditional on this short check. Section 13 records
day-1 code reading and census results (incl. GC-4DGS choosing views by position and scoring cam00 only; both new
methods choosing the best checkpoint by test PSNR; census v3 at D:\adags-ars\fmd\obs_census_v3.py).

Under 600 words, CRITICAL / MAJOR / MINOR per point:
1. Does revision 4 resolve the out-of-regime objection without creating a new one (e.g. the common 4C4D cut putting
   GC-4DGS and Multi4D-nerfds out of their own settings; GC-4DGS's own 4-view set having ~0 memory class)?
2. Anything in the first block (section 12 table) that should change BEFORE the first submission: wrong runs, missing
   control, a run that cannot answer its question, a scoring leak (test-PSNR checkpoint selection), start-cloud
   leakage from held-out views.
3. Is census v3's design sound enough to generate scoring masks (nearest-sample labels for moving surfaces, 99.6 %
   self-check, non-unique part ids)? Spot-check one thing in the code if you can.
4. Verdict: GO / GO WITH CHANGES / STOP, with the changes in order.
