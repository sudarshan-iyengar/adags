# RESULT — Spacetime Gaussian Grouping read IN FULL; it does NOT occupy
# the inferred-window cell, but it DOES correct one recorded premise
# (2026-08-24)

EXPLORATORY, read-only, **zero GPU-hours**. Closes the item
[[membership-occupancy-and-decision-2026-08-23c]] §3 flagged as "the
highest-value follow-up", where the paper was recorded as paywalled and
abstract-only.

## 1. Access — the paywall was not the obstacle it looked like

| source | depth |
|---|---|
| **EURASIP JIVP 2026**, `10.1186/s13640-025-00684-1` | **FULL TEXT, CC-BY open access** — §1-8, Eqs. 1-18, Tables 1-4 |
| PhD thesis, Bangning Wei, *Towards Semantic Editing of Volumetric Video*, Univ. Rennes, 2025-10-09 (`theses.hal.science/tel-05420716`) | **FULL TEXT**; Ch. 5 is the SGG chapter. Used only to recover Algorithm 1, which is an IMAGE in the journal HTML |
| **EUVIP 2024**, `10.1109/EUVIP61797.2024.10772975` | **abstract only; full text definitively unreachable** — HAL `hal-05033484` is a bibliographic notice with `files_s: null`; OpenAlex `W4406264439` Closed with both `pdf_url: null`; Semantic Scholar `CLOSED` |
| arXiv | **none** — API returns `totalResults 0` |
| code | **NONE FOUND**. GitHub search `total_count: 0`; no code-availability statement; thesis has zero occurrences of "github". Data "available from the corresponding author on reasonable request" |

**The paywalled record was wrong about the journal version, and right
about EUVIP.** The JIVP article is CC-BY; the plain URL 303-redirects to
Springer's IdP, which is what made it look gated. The 6-page EUVIP
version is genuinely unobtainable, so the EUVIP-vs-journal delta cannot
be checked by any route — recorded as a residual unknown, though the
journal is the extended superset and its abstract *drops* rather than
adds claims.

**citationCount: 0** on both versions, from both Semantic Scholar and
OpenAlex.

## 2. The mechanism, as read

Primary-verified by the primary agent against the cached full text
(quotes below are from it, not paraphrased by a worker):

* **Identity** — a **time-independent** 16-d learned vector `o_i` per
  Gaussian: *"time-independent Object Identity Encoding 16-bit vector
  `o_i` to each 3D Gaussian i"*. ("16-bit" is the authors' slip,
  inherited from Gaussian Grouping; Table 1 gives `o_i ∈ R^16`.)
  Decoded by a linear layer to `K+1` channels, `K = 200`.
* **Backbone** — genuinely Spacetime Gaussians, and **it does carry
  per-primitive temporal support**: *"temporal center, denoted by
  `μ_i^τ`, represents the timestamp at which the STG is most visible"*,
  plus a temporal scaling factor governing the *"effective duration of
  the STG"*, giving `trbf(t) = exp(−s_i^τ|t − μ_i^τ|)` and
  `σ_i(t) = σ_i^s · trbf(t)`.
* **Supervision** — purely 2D. SAM on frame 0 of a centre camera → DEVA
  propagates across views → DEVA propagates across time → cross-entropy
  on the α-composited identity render, plus Gaussian Grouping's kNN
  `loss_cls_3d` ported unchanged (k=5, m=1000).
* **Metrics** — *"The evaluation metrics utilized are the mean
  Intersection over Union (mIoU) and mean Accuracy (mAcc), calculated on
  the sequence of testing camera view."* Tables 3-4 carry exactly
  `mIoU | mAcc | Time | Size`. **The string "PSNR" occurs ZERO times in
  the full text** — there is no reconstruction metric at all. §7 concedes
  no ablation was performed.
* **Abstention** — none per primitive. There is a 2D background/unlabeled
  *pixel* channel; every Gaussian unconditionally carries an `o_i`, with
  no `-1`, no threshold, no decline-to-assign.
* **Suppression** — **not implemented in any form.** The abstract's
  "select and edit specific objects in a 4D manner" is not delivered: §6
  is identity colouring, selection and highlighting in a viewer, and §7
  concedes *"users cannot interact with all objects in the scene."* The
  parent Gaussian Grouping does remove — by deleting the object's
  Gaussians, **globally over the whole sequence** — and SGG does not port
  even that.
* **Inference** — nothing is inferred. `μ_i^τ` and `s_i^τ` are free
  photometric parameters optimized by the image loss; identity labels are
  supplied by SAM+DEVA. There is no absence detector, no changepoint, no
  boundary estimate, and no quantity naming an interval of non-presence.

## 3. VERDICT — DOES NOT OCCUPY

The cell ADAGS is protecting is **time-windowed per-primitive suppression
where the window is INFERRED**. SGG occupies **none of the three
load-bearing words**: there is no suppression, identity is explicitly
time-*independent* so nothing is time-windowed, and nothing is inferred.

**The forbidden-claim entry in
[[membership-occupancy-and-decision-2026-08-23c]] §4 — "that
time-windowed per-primitive suppression is unoccupied, until Spacetime
Gaussian Grouping is read in full" — is now DISCHARGED.** It has been
read in full and it does not occupy the cell.

## 4. ONE RECORDED PREMISE IS CORRECTED (append-only, not rewritten)

[[membership-occupancy-and-decision-2026-08-23c]] §3 opens: *"Q8, and it
is decisive: none of the seven operates on a 4D representation with
per-primitive temporal support."*

**That sentence was true of the seven methods it enumerated and is FALSE
as a general statement about this literature.** Spacetime Gaussian
Grouping is per-primitive 16-d identity on a `(μ^τ, s^τ)` substrate —
the same representational family as ADAGS. The correction is narrow and
it does not propagate: every consequence drawn in that section survives,
and one is *strengthened*.

**Strengthened:** "no method on the list reports per-primitive precision,
recall or IoU. Not one." SGG is now an (n+1)th method reporting only
rendered-2D mIoU/mAcc — and it is the one method that shares our
substrate, so it is the most probative instance available.

**Survives unchanged, and now with primary support:** §3's "first break"
applies to SGG verbatim. Its identity render weights `o_i` by the
*time-dependent* opacity, so the rows with the shortest temporal support
— exactly the event rows a gate must catch — are the least supervised.
And §3's "third break" (the kNN regularizer distances over position
only, with no temporal term) is Gaussian Grouping's `loss_cls_3d` ported
**unchanged**, so it too applies verbatim.

**§3's "second break" does NOT apply, and this is the one genuinely new
mechanism.** That break says a within-frame contrastive loss can never
pull together two primitives whose supports are disjoint, because they
never co-render. SGG's loss is **absolute-label cross-entropy against a
DEVA-propagated instance ID**, not a within-frame contrastive term — so a
primitive alive only before the gap and one alive only after are each
pushed toward the *same* global label. **A cross-episode supervisory path
therefore does exist in this family.** Two qualifications bound it: the
link is only as good as DEVA, and the authors name
disappearance-reappearance as precisely where their pipeline fails
(*"their performance remains suboptimal in specific scenarios, such as
occlusion, disappearance-reappearance"* — future work).

## 5. What adopting its machinery would and would not cost

Adopting SGG wholesale would buy a label-based cross-episode supervisory
path — the thing the contrastive family structurally lacks — and would
cost nothing in novelty on: inferring the window; time-windowed
suppression as a representation operation (unreachable from a
time-independent `o_i`, and exact-zero absence is unreachable from an RBF
that decays but never vanishes); multi-episode presence and reactivation;
per-primitive membership as a *measured* quantity; and abstention.

Given the measured ordering — partially gated 24.67 << not gated 27.14 <
fully gated 28.19 — SGG's total-assignment design, with no way to
decline, is an **active hazard** on this project's substrate rather than
a missing convenience.

## 6. Permitted and forbidden

**Permitted.** Spacetime Gaussian Grouping has been read in full from an
open-access primary source. It does not occupy time-windowed
per-primitive suppression with an inferred window. It reports no
per-primitive precision/recall/IoU and no reconstruction metric of any
kind. It does operate on a per-primitive-temporal-support 4D substrate,
which corrects one recorded sentence.

**Forbidden.** That the whole occupancy question is closed — **SA4D is
now the highest-value remaining read**, because it advertises a
*temporal* (time-varying) identity feature field, which is nearer the
time-windowed cell than SGG is; its full text was NOT obtained here. That
SGG's cross-episode label path is known to work across a genuine absence
— the authors themselves report that regime as failing. That the EUVIP
version contains nothing further — it is unreadable, not read.

## 7. Provenance

Full text obtained and cached by a bounded read-only worker; the primary
agent independently re-verified the load-bearing quotations against that
cached text — the temporal-centre and effective-duration wording, the
time-independent identity wording, the verbatim metrics sentence, and the
zero-occurrence PSNR count. Figures 1-9 were not read (images). The
worker downloaded the open-access thesis PDF (10.8 MB) to a scratchpad
directory and flagged the download; no repository file was touched.
