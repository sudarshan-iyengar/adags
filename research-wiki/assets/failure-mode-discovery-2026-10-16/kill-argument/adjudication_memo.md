**The attack defeats the current claim of a missing representational capability and a necessary new remedy. It does not invalidate the reported ghosting or kill the research direction.** I inspected the named documents, scorers, census CSVs and montage; I did not recompute training results from checkpoints.

### Point P_1: Temporal existence is already represented in prior art

**Attack claim**: Spacetime Gaussians and RetimeGS already provide temporal opacity for appearing and disappearing content, undermining novelty of a generic per-primitive presence gate.

**Verdict**: partially_answered

**Evidence (or lack of)**: The dossier explicitly acknowledges these designs and calls its proposed gate close to temporal boundaries ([kill_dossier.md:209–230](/D:/adags/research-wiki/assets/failure-mode-discovery-2026-10-16/kill_dossier.md:209)). [STG Eq. 7](https://arxiv.org/html/2312.16812v2#S4.SS1) uses a single Gaussian temporal weight; [RetimeGS Eqs. 3–5](https://arxiv.org/html/2603.13783v1#S3.SS2) uses a single bounded window with sigmoid tails. **Acknowledgment does not resolve novelty.** The headline’s “no notion of when a primitive exists” is unsustainable.

Both provide one connected region of effective temporal support per primitive, with nonzero tails. Encoding present–absent–present **through opacity** therefore requires two windows, supplied by separate primitives/groups or a richer gate. Neither method requires one primitive to represent the object throughout. Single-window support alone establishes no system-level incapacity.

**Severity if unresolved**: critical

**If unresolved, recommended fix**: Replace the missing-existence claim with a precise claim about failure to learn observed absence, and specify what the proposed mechanism adds beyond existing temporal windows and primitive duplication.

### Point P_2: Conversion-off contradicts the general causal inference

**Attack claim**: Substantial ghosting without static Gaussians demonstrates failure despite existing temporal support, rather than absence of a temporal-existence mechanism.

**Verdict**: still_unresolved

**Evidence (or lack of)**: Conversion-off retains −13.3/−11.0/−11.7 dB deficits; the dossier explicitly reports zero static Gaussians and attributes the ghost to temporal tails ([kill_dossier.md:248–270](/D:/adags/research-wiki/assets/failure-mode-discovery-2026-10-16/kill_dossier.md:248)). This establishes that static conversion is unnecessary for the observed failure. It establishes neither representational impossibility nor a causal tail diagnosis: no replacement-kernel experiment is reported.

**Severity if unresolved**: critical

**If unresolved, recommended fix**: On Hybrid 3D-4DGS conversion-off, repeat the three published-regime dense-camera fixtures with a short-tailed temporal kernel; persistent deficits would refute “tails alone explain it,” while their removal would refute the claimed missing-capability diagnosis.

### Point P_3: Negative controls do not test the proposed mechanism

**Attack claim**: Enabling transient densification and deleting seed points leave temporal support unchanged, so their failures cannot establish that a new presence mechanism is necessary.

**Verdict**: partially_answered

**Evidence (or lack of)**: Densification-on fails substantially in two scenes but improves carton; the text lists other configuration differences ([kill_dossier.md:147–155](/D:/adags/research-wiki/assets/failure-mode-discovery-2026-10-16/kill_dossier.md:147)). Seed deletion also leaves substantial deficits ([kill_dossier.md:298–313](/D:/adags/research-wiki/assets/failure-mode-discovery-2026-10-16/kill_dossier.md:298)). These are useful, intentional negative controls. Their narrow interpretation is sustainable; treating them as identification of a missing representation is not.

**Severity if unresolved**: major

**If unresolved, recommended fix**: In Multi4D’s synthetic monocular setting, directly compare the unchanged model, an existing temporal-window gate and the proposed gate; the proposed gate must remove ghosting while preserving present/return reconstruction to establish a mechanism-specific benefit.

### Point P_4: The decisive temporal-support comparison is missing

**Attack claim**: No matched evaluation establishes whether existing short-tailed temporal-support methods already solve the measured absence problem.

**Verdict**: still_unresolved

**Evidence (or lack of)**: The dossier itself requests STG/RetimeGS under the same protocol, but supplies no such absence comparison ([kill_dossier.md:225–230](/D:/adags/research-wiki/assets/failure-mode-discovery-2026-10-16/kill_dossier.md:225)). The full page’s STG results concern first-return frames, not the decisive absence endpoint ([plan.md:204–223](/D:/adags/research-wiki/operations/failure-mode-discovery-plan-2026-10-16.md:204)). Existing Ex4DGS rows do not substitute for testing the cited short-tailed alternative.

The web check qualifies the demanded comparison:

- **RetimeGS implementation:** no public training/rendering implementation was found. Its public [GitHub repository](https://github.com/William-WANG2/RetimeGS) contains the project website, not the reconstruction system.
- **RetimeGS evaluation:** its reported experiments use dense capture—32/60-camera datasets and all-camera Neural3DV tests—not monocular or a few-camera ablation. Low temporal sampling is not low camera count. [Paper §§4.1, C.7](https://arxiv.org/html/2603.13783v1).
- **STG evaluation:** the original paper considers multiview input and identifies monocular adaptation as future work. However, a later [monocular benchmark, §3.3 and Tables 2–7](https://arxiv.org/html/2412.04457), evaluates STG variants in a common implementation on five datasets, including NeRF-DS and iPhone. “STG has never been evaluated monocularly” would be false.

Thus, RetimeGS is architectural counterevidence, **not demonstrated success on this protocol**; its unavailable implementation does not excuse omitting every executable temporal-support control.

**Severity if unresolved**: critical

**If unresolved, recommended fix**: Evaluate released STG and a clearly labelled RetimeGS-style short-tail ablation on the same dense and monocular absence/return fixtures; persistent ghosting in these controls, removed by the proposed mechanism without present/return loss, would answer the missing-comparison objection.

### Point P_5: The proposed remedy has no current effectiveness evidence

**Attack claim**: An explicitly unbuilt presence gate cannot establish either successful correction or necessity relative to existing alternatives.

**Verdict**: partially_answered

**Evidence (or lack of)**: “NOT YET BUILT” is explicit ([kill_dossier.md:18](/D:/adags/research-wiki/assets/failure-mode-discovery-2026-10-16/kill_dossier.md:18)). The full page mentions older memory-gate gains, but labels those effect sizes as historical and not rerun ([plan.md:198–200](/D:/adags/research-wiki/operations/failure-mode-discovery-plan-2026-10-16.md:198), [326–327](/D:/adags/research-wiki/operations/failure-mode-discovery-plan-2026-10-16.md:326)). Deferring implementation is sustainable during discovery; presenting the remedy as established or necessary is not.

**Severity if unresolved**: critical

**If unresolved, recommended fix**: Implement the proposed Multi4D gate and compare it against an existing temporal-window gate on identical synthetic monocular inputs, showing repeatable absence improvement with preserved present, return and surrounding-background quality.

### Point P_6: The benchmark demonstrates nothing beyond selected implementation failures

**Attack claim**: Because the relevant capability already exists architecturally, the benchmark cannot support a substantive contribution beyond exposing selected implementations to an event class.

**Verdict**: partially_answered

**Evidence (or lack of)**: The current evidence establishes more than an unexplained visual defect: training observes 83–100% of the vacated synthetic surface, with substantial local deficits ([kill_dossier.md:164–189](/D:/adags/research-wiki/assets/failure-mode-discovery-2026-10-16/kill_dossier.md:164)). The real-image montage corroborates retained bottles; the scorer measures bottle similarity separately. However, those absences are authored composites ([kill_dossier.md:275–296](/D:/adags/research-wiki/assets/failure-mode-discovery-2026-10-16/kill_dossier.md:275)), and benchmark novelty remains unestablished.

The narrower empirical position is sustainable. Existing representational capacity does not make evaluation of learned failures worthless; it does prevent that evaluation from proving a missing capability.

**Severity if unresolved**: major

**If unresolved, recommended fix**: Establish the benchmark’s discriminating value with existing temporal-support baselines and a filmed removal/return sequence, reporting absence and return separately and showing that whole-frame rankings conceal materially different event performance.

## Summary

Total rejection points: 6

- answered_by_current_text: 0
- partially_answered: 4
- still_unresolved: 2

## Net assessment

**The current headline and diagnosis would not survive a senior area-chair read.** Measured ghosting is supported; the inference from ghosting to absent representational capacity, novel diagnosis and necessary gate is not. The attack kills that inference, not the direction. At discovery stage, the 24 October decision should depend on a discriminating temporal-support comparison. The 16 November submission needs either a validated method advantage or a narrower empirical contribution with established evaluation novelty.

## Top action items

1. **Run the missing comparison:** STG plus a RetimeGS-style short-tail control on identical absence/return fixtures, starting with the dense setting.
2. **Test the mechanism directly:** proposed gate versus existing temporal windows, with repeat runs and present/return quality checks.
3. **Rewrite the claim now:** report failure to learn observed absence; remove assertions that dynamic splatting lacks temporal existence or necessarily requires a new presence gate.