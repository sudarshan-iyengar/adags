# Evidence heads and unhoused constants — freeze packet (2026-08-18)

**Outcome: the freeze CANNOT be executed tonight, and the reason is not
neglect.** Every head's authorized calibration data is defined
relationally — "DiVa-360 calibration sequences ONLY, disjoint from
dev/locked/held-out" — and since M1 failed there is no admitted sequence
set to be disjoint FROM. Naming a calibration sequence today would
silently invent the development boundary that the still-open event-supply
decision is supposed to set. This page therefore does what the situation
allows: it inventories every item, states precisely what blocks each, and
records the two decisions that unblock the rest.

Reads [[elgs-method]], [[elgs-experiment-plan]],
[[elgs-m1-evidence-wiring-record]], [[elgs-absence-diagnostic-result]],
[[elgs-m2-oncomponent-split-design]].

## 1. The reassuring finding first

`elgs/evidence_stack.py:170-196` already **refuses** to build heads from
the frozen prereg while any required entry carries
`"status": "unfrozen"`, and accepts the declared smoke constants ONLY
when `elgs_smoke_schedule` is set. So the failure mode this packet exists
to prevent — an evidence-bearing run quietly consuming smoke-tier
midpoints as if they were fitted — **is already blocked in code, fail
closed.** Verified by reading the gate, not from its docstring.

`configs/elgs/smoke_evidence_heads_v1.json` is likewise honest about
itself: it declares `"tier": "smoke"`, `"evidence_bearing": false`,
"NOT a preregistration and NOT a freeze", records that its values are
range midpoints "chosen WITHOUT looking at any DiVa-360 report
distribution, so they carry no information from the data and cannot bias
a later fit", and carries its own `unfrozen_items_still_outstanding`
list. The audit's concern that these constants are loose is correct about
their STATUS and wrong about their containment.

## 2. Inventory

`prereg_evidence_heads_v1.json` already fixes family, allowed range,
authorized data, criterion, sensitivity analysis and freeze point for
every head. What is missing is the fit, not the specification.

| item | owner | structural or calibratable | prereg home | blocker |
|---|---|---|---|---|
| `g_v` | `elgs/evidence.py:145` | calibratable — beta(a,b), a∈[1,8], b∈[0.5,4], MLE on calibration visible-report values | `prereg_evidence_heads_v1` ✔ | calibration sequences unnamed |
| `h_c` | `elgs/evidence.py:158` | calibratable — uniform or 8-bin histogram, MLE on censored-condition reports | ✔ | same |
| `h_o` | `elgs/evidence.py:166` | structural — uniform on `[v_min, v_max]`, no free parameter | ✔ | freezes trivially WITH the others; no data needed |
| `pi_miss` | `elgs/evidence.py` | calibratable — empirical miss rates per condition, `[0.01, 0.99]` | ✔ | calibration sequences unnamed |
| `g_pos_sigma` | `elgs/evidence.py:57` | calibratable — MLE of report-to-bridge residuals, `[0.5, 8]` px | ✔ | same |
| `r_u` | `elgs/clusters.py:43` | consumed verbatim from the tracks artifact manifest | ✔ (`reliability.r_u`) | freezes with a tracks artifact, i.e. with the route decision |
| `rho` | `elgs/evidence_stack.py:119` | calibratable — `alpha_u = n_cam^(-rho)`, `[0.5, 1.0]` | **`prereg_observability_v1` ✔** (contra the audit, which listed it as unhoused — verified: 2 occurrences) | empirical variance inflation unmeasured |
| `anchor_report_floor` | `elgs/evidence_stack.py:120` | **structural** (a search-admissibility floor, not an estimate) | **NONE — 0 occurrences in either prereg** | needs a home; see §3 |
| `plateau_seed_fraction` | `elgs/evidence_stack.py:122` | **structural**, user-directed 2026-08-14 | **NONE — 0 occurrences in either prereg** | needs a home; see §3 |
| report-population bound | `elgs/evidence_stack.py:480` | structural cap | none | needs a home; see §3 |
| confirmation-unit time collapse | `elgs/trainer_hooks.py:948` | observability defect | n/a | see §4 |

## 3. The three genuinely unhoused constants

`anchor_report_floor` (8) and `plateau_seed_fraction` (0.5) have **no
prereg home at all** — confirmed by direct search of both prereg files,
0 occurrences each. The smoke file says so itself, in those words.

They are NOT of the same kind as the heads, and the distinction decides
how to house them:

* the heads are **estimates** — they have a criterion (MLE), authorized
  data, and a sensitivity analysis, and they are wrong if fitted badly;
* these two are **structural search parameters** — they define which
  anchor intervals are admissible at all. There is no likelihood to
  maximise. Fitting them to data would be selecting the search on the
  outcome, which is the failure this project has already recorded three
  times.

`plateau_seed_fraction` in particular was **user-directed on 2026-08-14
after the previous predicate was measured INERT**: a pooled seed count
`>= 2` was true at every frame on scissor (one cluster held 512 seeds of
10,995 tracks), giving one anchor and zero windows on the sequence the
census scored with 343 true-absence windows. A fraction is
scale-invariant and cannot saturate. That is a principled structural
correction, not a tuned value.

**Recommendation, for the user to accept or reject:** house all three in
a new `prereg_structural_search_v1.json` as **structural constants frozen
by disclosure** — value, the reasoning that produced it, the range over
which a sensitivity sweep will be reported, and an explicit statement
that they were NOT fitted to any outcome. That is honest and executable
today. The alternative — declaring them calibratable — needs named
calibration data and would make the structural search outcome-dependent.

**This page does not create that file.** Housing a constant is a
preregistration act with claim consequences, and the packet's job is to
put the decision in front of the user, not to take it.

## 4. Confirmation-slot time collapse

`elgs/trainer_hooks.py:948` emits
`"unit_timestamps": sorted({float(u[1]) for u in decision["units"]})`,
and it reported `[0.0]` — every confirmation unit carried the same
timestamp, so the confirmation slot sampled no temporal extent. This is a
**measurement defect in §7's observability**, not a wrong accumulator:
the field is diagnostic output, and the acceptance arithmetic does not
read it.

It matters because `se = 0` on every unit, reported across the evidence
runs, is exactly what a degenerate single-timestamp sample would produce
— so the standard error currently cannot distinguish "the estimator is
precise" from "the sample has one point". Until that is resolved, **no
acceptance decision's `se` may be cited as evidence of anything.**

Repair is a prereg amendment, not a code fix in isolation, because the
confirmation-unit construction is part of the frozen §7 measure. Deferred
to that amendment; recorded here so it is not lost.

## 5. What unblocks what

| decision | unblocks |
|---|---|
| **Event-supply route** (tranche-2 / re-track / R3 descope / synthetic) — open since 2026-08-12 | names the admitted sequence set, hence dev/locked/held-out, hence which sequences are legal calibration data, hence `g_v`, `h_c`, `h_o`, `pi_miss`, `g_pos_sigma`, `r_u` |
| **Structural-vs-calibratable ruling** on the three unhoused constants (§3) | `anchor_report_floor`, `plateau_seed_fraction`, report bound |
| **§7 prereg amendment** | the confirmation-slot measure, and whether `se` is ever citable |

`rho` is the one item blocked by neither: it is housed and calibratable,
and its criterion is an empirical variance inflation of per-camera report
agreement, which could in principle be measured on any admitted sequence.
It still cannot be frozen before the route decision names one.

## 6. What was NOT done, and why

No head was frozen. No value was chosen. No calibration sequence was
named. No prereg file was created or edited.

Freezing a head tonight would mean picking calibration sequences from the
23 tranche-1 DiVa-360 sequences with no admitted development boundary to
be disjoint from — which is precisely the act that would make a later
claim unfalsifiable, and precisely the act the freeze discipline exists
to prevent. The correct output of this lane, given an open route
decision, is this packet.
