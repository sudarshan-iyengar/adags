# -*- coding: utf-8 -*-
"""Frozen independent ORACLE for EL-GS v8.3 formal spec section 7 ("Acceptance").

SOLE SOURCE OF TRUTH FOR THIS FILE
----------------------------------
`research-wiki/operations/elgs-v8-formal-spec.md`, section
"## 7. Acceptance (heuristic; SNIS estimator - consistent, finite-sample biased)",
plus the untagged normative revision text of header item (6) ("Acceptance
estimator: ... bootstrap: paired cluster resampling of (camera,frame) units
with the SAME resample indices for candidate and incumbent, B=200, SE = sd of
paired replicate differences; degeneracy (<=5 clusters) => reject; ...").

This module was written with NO knowledge of any EL-GS implementation, any
implementation plan, or any existing test. It is a deliberately naive,
straight-line, numpy-only transcription of the spec prose: loops instead of
vectorization, no caching, no shortcuts, no optimization. Readability against
the spec sentence-by-sentence is the only design goal.

Section 7 header note that governs the whole module:

    "CANONICAL ESTIMATOR (self-normalized importance sampling; supersedes
    every 'ordinary/unnormalized', 'unbiased', or 'estimator exact'
    description of the sampled render estimate in earlier revisions)"

and the guarantee statement:

    "GUARANTEES (all that is claimed): R-hat is a ratio estimator - strongly
    consistent for the nu-mean E_nu[l] as n -> infinity and bounded-weight
    stable, but NOT unbiased in general at finite sample size
    (ratio-estimator bias O(1/n); it vanishes only in degenerate cases,
    e.g. lambda_u = 1 or constant l); no unbiasedness or exactness of the
    sampled estimate is claimed or used anywhere - acceptance is a
    preregistered heuristic (section 9), and the paired CRN design cancels
    the shared normalization noise between candidate and incumbent without
    eliminating the bias."

Section 7 also explicitly REJECTS the unnormalized alternative as the adopted
estimator:

    "(The unnormalized alternative (1/n)*Sum_i a_i*l(x_i) would be unbiased
    for the normalized nu but is NOT the adopted estimator; SNIS is adopted
    for weight-noise cancellation under CRN.)"

Unicode note: spec quotes are transliterated to ASCII inside docstrings and
messages (R-hat for R with circumflex, nu for the Greek letter, l for script
ell, etc.) so that the self-check prints cleanly on a cp1252 console. No
mathematical content is altered by the transliteration.
"""

import math
import random
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Sample",
    "snis_weight",
    "clip_active",
    "snis_ratio",
    "paired_delta",
    "paired_cluster_bootstrap_se",
    "delta_e_total",
    "accept",
    "BOOTSTRAP_REPLICATES",
    "MIN_CLUSTERS_EXCLUSIVE",
]


# Section 7 / header item (6): "B=200 replicates".
BOOTSTRAP_REPLICATES = 200

# Header item (6): "degeneracy (<=5 clusters) => reject". Five clusters is
# degenerate; six is the smallest admissible count.
MIN_CLUSTERS_EXCLUSIVE = 5


# ---------------------------------------------------------------------------
# 1. Sample representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One drawn render sample x_i, carrying both arms under common random numbers.

    Section 7 fixes the sampling law:

        "R-hat = Sum_i a_i*l(x_i) / Sum_i a_i with samples x_i drawn from the
        mixture m = lambda_u*nu + (1-lambda_u)*pi_D (0 < lambda_u <= 1)"

    and section 4 fixes nu:

        "per-pixel mean over the evaluation measure nu = uniform over
        (training-camera, frame) pairs x pixels; no masks."

    Fields
    ------
    unit_key:
        The bootstrap cluster identity of this sample. Section 7:
        "SE: cluster bootstrap over (camera, frame) units". `unit_key` IS the
        (camera, frame) pair; all samples sharing a `unit_key` are resampled
        together as one unit.
    nu_density:
        nu(x_i), the target-measure density at the drawn sample.
    m_density:
        m(x_i) = lambda_u*nu(x_i) + (1-lambda_u)*pi_D(x_i), the proposal
        (mixture) density at the drawn sample. Supplied directly rather than
        reconstructed, so that this oracle transcribes only the estimator and
        not the sampler.
    loss_incumbent:
        l(x_i) evaluated at the incumbent program/parameters.
    loss_candidate:
        l(x_i) evaluated at the candidate program/parameters, on the SAME x_i.
        Section 7: "COMMON RANDOM NUMBERS: identical {x_i} for incumbent and
        candidate".
    """

    unit_key: tuple
    nu_density: float
    m_density: float
    loss_incumbent: float
    loss_candidate: float


def _arm_loss(sample: Sample, arm: str) -> float:
    """Select l(x_i) for the named arm. Only the two spec arms exist."""
    if arm == "incumbent":
        return float(sample.loss_incumbent)
    if arm == "candidate":
        return float(sample.loss_candidate)
    raise ValueError(
        "arm must be 'incumbent' or 'candidate' (section 7 names exactly two "
        "arms); got %r" % (arm,)
    )


def _check_lambda_u(lambda_u: float) -> float:
    """Section 7 admissible range: '(0 < lambda_u <= 1)'."""
    lam = float(lambda_u)
    if not (lam > 0.0) or not (lam <= 1.0):
        raise ValueError(
            "lambda_u must satisfy 0 < lambda_u <= 1 (section 7); got %r"
            % (lambda_u,)
        )
    return lam


# ---------------------------------------------------------------------------
# 2. Self-normalized importance weight
# ---------------------------------------------------------------------------


def snis_weight(sample: Sample, lambda_u: float) -> float:
    """Return a_i for one sample.

    Transcribes section 7 verbatim:

        "a_i = min{ w_max, nu(x_i)/m(x_i) } with w_max := 1/lambda_u - the
        true weight is <= 1/lambda_u ALWAYS, so clipping is PROVABLY INACTIVE
        (retained as a formal guard)."

    Why the clip is provably inactive, per the same sentence: the proposal is
    the mixture m = lambda_u*nu + (1-lambda_u)*pi_D with pi_D >= 0, hence
    m(x) >= lambda_u*nu(x) pointwise, hence nu(x)/m(x) <= 1/lambda_u
    everywhere the ratio is defined. The `min` therefore never binds for a
    well-formed mixture; it is transcribed anyway because section 7 keeps it
    as a formal guard. Use `clip_active` to assert this in tests: a clip that
    DOES bind is evidence that the supplied `m_density` is not a valid
    lambda_u-mixture of the supplied `nu_density`, not evidence about the
    estimator.

    The weight is arm-independent by construction (it depends only on the
    sampling densities), which is precisely what makes the paired CRN
    difference of section 7 share one weight sequence across both arms.
    """
    lam = _check_lambda_u(lambda_u)
    w_max = 1.0 / lam
    m = float(sample.m_density)
    if not (m > 0.0):
        raise ValueError(
            "m(x_i) must be strictly positive for the weight ratio to be "
            "defined; got m_density=%r for unit %r" % (m, sample.unit_key)
        )
    nu = float(sample.nu_density)
    if nu < 0.0:
        raise ValueError(
            "nu(x_i) must be nonnegative; got nu_density=%r for unit %r"
            % (nu, sample.unit_key)
        )
    true_weight = nu / m
    return float(min(w_max, true_weight))


def clip_active(sample: Sample, lambda_u: float) -> bool:
    """Diagnostic: did the 'PROVABLY INACTIVE' guard actually bind?

    Section 7 states the clip never binds for a valid mixture proposal. This
    helper exists so a caller can verify that claim on real draws; it is not
    part of the estimator.
    """
    lam = _check_lambda_u(lambda_u)
    m = float(sample.m_density)
    if not (m > 0.0):
        raise ValueError(
            "m(x_i) must be strictly positive; got m_density=%r for unit %r"
            % (m, sample.unit_key)
        )
    return bool((float(sample.nu_density) / m) > (1.0 / lam))


# ---------------------------------------------------------------------------
# 3. Self-normalized ratio estimator
# ---------------------------------------------------------------------------


def snis_ratio(samples: list, lambda_u: float, arm: str) -> float:
    """Return R-hat for the named arm.

    Transcribes section 7:

        "R-hat = Sum_i a_i*l(x_i) / Sum_i a_i"

    Naive two-accumulator loop, exactly as written: numerator = sum of
    a_i*l(x_i), denominator = sum of a_i, then divide. The division IS the
    self-normalization; there is no separate normalization step anywhere else
    in this module.
    """
    if len(samples) == 0:
        raise ValueError(
            "snis_ratio requires at least one sample (Sum_i a_i would be an "
            "empty sum, leaving R-hat undefined)"
        )
    numerator = np.float64(0.0)
    denominator = np.float64(0.0)
    for sample in samples:
        a_i = np.float64(snis_weight(sample, lambda_u))
        l_i = np.float64(_arm_loss(sample, arm))
        numerator = numerator + a_i * l_i
        denominator = denominator + a_i
    if not (float(denominator) > 0.0):
        raise ValueError(
            "Sum_i a_i is not strictly positive (%r); R-hat is undefined"
            % (float(denominator),)
        )
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# 4. Paired (common random numbers) difference
# ---------------------------------------------------------------------------


def paired_delta(samples: list, lambda_u: float) -> float:
    """Return the paired SNIS render difference, candidate minus incumbent.

    Transcribes section 7:

        "COMMON RANDOM NUMBERS: identical {x_i} for incumbent and candidate;
        Delta-E-hat is the paired SNIS difference plus the exact deltas."

    The pairing is structural, not statistical: BOTH arms are evaluated on the
    same `samples` sequence, hence on the same weights a_i and the same
    denominator Sum_i a_i. Section 7 states the purpose:

        "the paired CRN design cancels the shared normalization noise between
        candidate and incumbent without eliminating the bias."

    Sign convention: candidate minus incumbent. This is forced by the
    acceptance rule "Accept iff Delta-E-hat + k*SE < 0" together with section
    4's energy semantics - a candidate is accepted when it LOWERS the energy,
    so a negative delta must mean candidate below incumbent.

    NOTE this returns ONLY the sampled render part. Section 7:

        "Exact (non-sampled) tracker and prior deltas - computed in closed
        form, not estimated - are added outside the sampled render estimate."

    Use `delta_e_total` to combine.
    """
    r_candidate = snis_ratio(samples, lambda_u, "candidate")
    r_incumbent = snis_ratio(samples, lambda_u, "incumbent")
    return float(r_candidate - r_incumbent)


# ---------------------------------------------------------------------------
# 5. Paired cluster bootstrap standard error
# ---------------------------------------------------------------------------


def _group_by_unit(samples: list) -> tuple:
    """Group samples into (camera, frame) units and return (sorted keys, map).

    Section 7 says "cluster bootstrap over (camera, frame) units"; header item
    (6) says "paired cluster resampling of (camera,frame) units". Grouping is
    therefore by `unit_key` exactly. Within-unit sample order is the input
    order (deterministic); the unit list is sorted so the resample indices are
    reproducible from `seed` alone.
    """
    buckets = {}
    for sample in samples:
        key = sample.unit_key
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(sample)
    try:
        keys = sorted(buckets.keys())
    except TypeError:
        # Heterogeneous key types are not orderable; fall back to a stable
        # textual order so determinism is preserved. The spec does not define
        # a unit ordering (see the AMBIGUITY notes at the bottom of this file).
        keys = sorted(buckets.keys(), key=repr)
    return list(keys), buckets


def paired_cluster_bootstrap_se(
    samples: list,
    lambda_u: float,
    seed: int,
    B: int = BOOTSTRAP_REPLICATES,
) -> float:
    """Return SE = sd of paired replicate differences.

    Transcribes section 7:

        "SE: cluster bootstrap over (camera, frame) units, B=200 replicates,
        weights renormalized within each replicate (the renormalization is
        exactly the self-normalized form applied per replicate); SE = sd of
        paired replicate differences; SE undefined => reject."

    and header item (6):

        "bootstrap: paired cluster resampling of (camera,frame) units with the
        SAME resample indices for candidate and incumbent, B=200, SE = sd of
        paired replicate differences; degeneracy (<=5 clusters) => reject"

    Transcription decisions, each tied to one clause:

    * "cluster ... over (camera, frame) units" - the resampling atom is the
      unit, never the individual sample. Drawing a unit pulls in ALL of its
      samples, in input order.
    * "paired ... with the SAME resample indices for candidate and incumbent"
      - one index list per replicate, applied to both arms. The replicate is
      materialized once as a list of Samples, and each Sample carries both
      arms, so the arms cannot desynchronize by construction.
    * "weights renormalized within each replicate (the renormalization is
      exactly the self-normalized form applied per replicate)" - the replicate
      difference is `paired_delta` applied to the replicate multiset, whose
      denominator Sum a_i is recomputed over the resampled multiset. Nothing
      is carried over from the full-sample fit. A unit drawn twice contributes
      its weights twice to that denominator.
    * "B=200 replicates" - the default.
    * "degeneracy (<=5 clusters) => reject" - raises ValueError at five or
      fewer distinct units, before any draw.
    * "SE undefined => reject" - raises ValueError when the standard deviation
      cannot be formed (fewer than two replicates, or a non-finite result).

    Both rejection clauses are transcribed as ValueError rather than as a
    returned sentinel: "reject" is a decision about the CANDIDATE, and the
    caller must not be able to slide a sentinel SE into `accept` and have the
    comparison silently succeed.

    Draw structure: `random.Random(seed)`, and for each replicate one index
    list of length len(units), each index drawn uniformly with replacement
    from range(len(units)) over the SORTED unit list. Replicates are drawn in
    order 0..B-1 from that single generator, so the whole bootstrap is a pure
    function of (samples, lambda_u, seed, B).

    Standard deviation: the SAMPLE standard deviation (n-1 denominator, i.e.
    numpy ddof=1) over the B replicate differences.
    """
    lam = _check_lambda_u(lambda_u)
    if len(samples) == 0:
        raise ValueError("paired_cluster_bootstrap_se requires at least one sample")
    if int(B) < 1:
        raise ValueError("B must be >= 1 (section 7 preregisters B=200); got %r" % (B,))

    unit_keys, buckets = _group_by_unit(samples)
    n_units = len(unit_keys)

    # header item (6): "degeneracy (<=5 clusters) => reject"
    if n_units <= MIN_CLUSTERS_EXCLUSIVE:
        raise ValueError(
            "REJECT (degeneracy): %d distinct (camera, frame) clusters; the "
            "spec rejects at <=%d clusters"
            % (n_units, MIN_CLUSTERS_EXCLUSIVE)
        )

    rng = random.Random(seed)

    replicate_deltas = []
    for _replicate in range(int(B)):
        # ONE index list per replicate; the SAME indices serve both arms.
        indices = []
        for _draw in range(n_units):
            indices.append(rng.randrange(n_units))

        replicate_samples = []
        for idx in indices:
            for sample in buckets[unit_keys[idx]]:
                replicate_samples.append(sample)

        # "weights renormalized within each replicate (the renormalization is
        # exactly the self-normalized form applied per replicate)" - this call
        # recomputes Sum a_i over the replicate multiset for both arms.
        replicate_deltas.append(paired_delta(replicate_samples, lam))

    if len(replicate_deltas) < 2:
        raise ValueError(
            "REJECT (SE undefined): %d replicate difference(s); the sample "
            "standard deviation needs at least 2" % (len(replicate_deltas),)
        )

    deltas = np.asarray(replicate_deltas, dtype=np.float64)
    if not np.all(np.isfinite(deltas)):
        raise ValueError(
            "REJECT (SE undefined): non-finite replicate difference(s) in the "
            "bootstrap"
        )

    se = float(np.std(deltas, ddof=1))  # SAMPLE sd (n-1 denominator)
    if not math.isfinite(se):
        raise ValueError("REJECT (SE undefined): standard deviation is not finite")
    return se


# ---------------------------------------------------------------------------
# 6. Total delta and the acceptance rule
# ---------------------------------------------------------------------------


def delta_e_total(paired_render_delta: float, *exact_deltas: float) -> float:
    """Assemble Delta-E-hat from the sampled part and the closed-form parts.

    Section 7:

        "Exact (non-sampled) tracker and prior deltas - computed in closed
        form, not estimated - are added outside the sampled render estimate.
        COMMON RANDOM NUMBERS: identical {x_i} for incumbent and candidate;
        Delta-E-hat is the paired SNIS difference plus the exact deltas."

        "Accept iff Delta-E-hat + k*SE < 0 (Delta-E-hat includes the
        transaction increment)."

    The transaction increment (section 4: "C(H) = chi*N_returnbirth(H) +
    mu*N_merge(H)") is one of the exact deltas and must be passed in by the
    caller; this oracle does not know the ledger.
    """
    total = float(paired_render_delta)
    for term in exact_deltas:
        total = total + float(term)
    return total


def accept(delta_total: float, se: float, k: float) -> bool:
    """Return the acceptance decision.

    Transcribes section 7 verbatim:

        "Accept iff Delta-E-hat + k*SE < 0 (Delta-E-hat includes the
        transaction increment)."

    Strict inequality, as written: equality is a reject.

    `delta_total` must already include the exact deltas and the transaction
    increment (see `delta_e_total`); this function does not add them.

    Non-finite inputs compare False under Python's IEEE semantics, which is
    the correct direction for section 7's two "=> reject" clauses. The SE
    producer in this module raises instead of returning a non-finite value, so
    a NaN reaching here means the caller bypassed
    `paired_cluster_bootstrap_se`.
    """
    return bool((float(delta_total) + float(k) * float(se)) < 0.0)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _synthetic_samples(
    n_units: int,
    per_unit: int,
    lambda_u: float,
    candidate_shift,
) -> list:
    """Build a deterministic synthetic sample set.

    `m_density` is constructed as the genuine mixture
    m = lambda_u*nu + (1-lambda_u)*pi_D with pi_D > 0, so that the section-7
    bound nu/m <= 1/lambda_u holds by construction and the clip is inactive -
    exactly the situation the spec describes.

    `candidate_shift(unit_index, sample_index, loss_incumbent)` returns the
    additive candidate-minus-incumbent shift at that sample.
    """
    samples = []
    for u in range(n_units):
        for s in range(per_unit):
            nu = 0.5 + 0.25 * ((u * 7 + s * 3) % 5)          # in [0.5, 1.5]
            pi_d = 0.2 + 0.15 * ((u * 5 + s * 2) % 4)        # in [0.2, 0.65]
            m = lambda_u * nu + (1.0 - lambda_u) * pi_d
            loss_inc = 1.0 + 0.1 * ((u * 3 + s) % 6) + 0.01 * s
            loss_cand = loss_inc + candidate_shift(u, s, loss_inc)
            samples.append(
                Sample(
                    unit_key=("cam%02d" % (u % 4), 100 + u),
                    nu_density=float(nu),
                    m_density=float(m),
                    loss_incumbent=float(loss_inc),
                    loss_candidate=float(loss_cand),
                )
            )
    return samples


def _self_check() -> None:
    lambda_u = 0.4
    k = 1.0

    print("=== EL-GS section-7 bootstrap reference: self-check ===")
    print("lambda_u = %r   w_max = 1/lambda_u = %r" % (lambda_u, 1.0 / lambda_u))

    # --- Case A: constant candidate shift of -0.5 over 8 units x 6 = 48 samples
    # Unit keys must be distinct per unit for the cluster count to be 8, so the
    # synthetic builder keys on (cam, 100+u) with u unique.
    samples_a = _synthetic_samples(8, 6, lambda_u, lambda u, s, li: -0.5)
    units_a = sorted({sm.unit_key for sm in samples_a})
    print()
    print("[A] constant candidate shift = -0.5")
    print("    n_samples = %d   n_units = %d" % (len(samples_a), len(units_a)))

    n_clipped = sum(1 for sm in samples_a if clip_active(sm, lambda_u))
    print("    clip active on %d/%d samples (spec: PROVABLY INACTIVE)"
          % (n_clipped, len(samples_a)))

    r_inc = snis_ratio(samples_a, lambda_u, "incumbent")
    r_can = snis_ratio(samples_a, lambda_u, "candidate")
    delta_a = paired_delta(samples_a, lambda_u)
    print("    R_hat incumbent = %.15f" % r_inc)
    print("    R_hat candidate = %.15f" % r_can)
    print("    paired_delta    = %.15f   (expect -0.5)" % delta_a)
    print("    |delta + 0.5|   = %.3e   (expect <= 1e-12)" % abs(delta_a + 0.5))
    assert abs(delta_a + 0.5) <= 1e-12, "constant shift must transcribe exactly"

    se_a = paired_cluster_bootstrap_se(samples_a, lambda_u, seed=20260811)
    print("    bootstrap SE    = %.3e   (expect ~0 for a constant shift)" % se_a)
    assert se_a <= 1e-12, "constant shift must give a ~zero SE"
    print("    accept(delta, SE, k=%r) = %r" % (k, accept(delta_a, se_a, k)))

    # --- Case B: varied candidate shift -> strictly positive SE
    def varied(u, s, li):
        return -0.5 + 0.30 * math.sin(1.7 * u + 0.4 * s) + 0.05 * (u % 3)

    samples_b = _synthetic_samples(8, 6, lambda_u, varied)
    delta_b = paired_delta(samples_b, lambda_u)
    se_b = paired_cluster_bootstrap_se(samples_b, lambda_u, seed=20260811)
    print()
    print("[B] varied candidate shift")
    print("    n_samples = %d   n_units = %d"
          % (len(samples_b), len({sm.unit_key for sm in samples_b})))
    print("    paired_delta    = %.15f" % delta_b)
    print("    bootstrap SE    = %.15f   (expect > 0)" % se_b)
    assert se_b > 0.0, "varied losses must give a strictly positive SE"
    se_b_again = paired_cluster_bootstrap_se(samples_b, lambda_u, seed=20260811)
    print("    same seed reproduces SE: %r" % (se_b_again == se_b,))
    assert se_b_again == se_b, "bootstrap must be a pure function of the seed"
    se_b_other = paired_cluster_bootstrap_se(samples_b, lambda_u, seed=7)
    print("    different seed -> SE = %.15f" % se_b_other)
    total_b = delta_e_total(delta_b, 0.02)  # 0.02 stands in for exact + ledger
    print("    delta_E_total (with +0.02 exact/ledger) = %.15f" % total_b)
    print("    accept(total, SE, k=%r) = %r" % (k, accept(total_b, se_b, k)))
    print("    accept(total, SE, k=%r) = %r" % (10.0, accept(total_b, se_b, 10.0)))

    # --- Case C: degeneracy at 5 clusters
    samples_c = _synthetic_samples(5, 6, lambda_u, lambda u, s, li: -0.5)
    print()
    print("[C] degeneracy: %d distinct units (spec: '<=5 clusters => reject')"
          % (len({sm.unit_key for sm in samples_c}),))
    try:
        paired_cluster_bootstrap_se(samples_c, lambda_u, seed=1)
    except ValueError as exc:
        print("    ValueError raised: %s" % (exc,))
    else:
        raise AssertionError("5 clusters must be rejected as degenerate")

    samples_d = _synthetic_samples(6, 6, lambda_u, varied)
    se_d = paired_cluster_bootstrap_se(samples_d, lambda_u, seed=1)
    print("    6 distinct units accepted; SE = %.15f" % se_d)

    # --- Case D: acceptance boundary is strict
    print()
    print("[D] acceptance rule 'Delta + k*SE < 0' is strict")
    print("    accept(-1e-15, 0.0, 1.0) = %r" % accept(-1e-15, 0.0, 1.0))
    print("    accept( 0.0,   0.0, 1.0) = %r" % accept(0.0, 0.0, 1.0))
    assert accept(-1e-15, 0.0, 1.0) is True
    assert accept(0.0, 0.0, 1.0) is False

    print()
    print("=== self-check PASSED ===")


if __name__ == "__main__":
    _self_check()


# ---------------------------------------------------------------------------
# AMBIGUITIES FLAGGED AGAINST SECTION 7 (recorded, not silently resolved)
# ---------------------------------------------------------------------------
# A1. "SE = sd of paired replicate differences" does not say whether "sd" is
#     the sample (n-1) or population (n) standard deviation. This oracle uses
#     the SAMPLE sd (ddof=1). At B=200 the two differ by a factor
#     sqrt(200/199) ~ 1.0025, which is below any plausible acceptance-decision
#     tolerance but is NOT bit-identical; a comparison against an
#     implementation must fix this convention explicitly.
# A2. The resample size is not stated. This oracle draws len(units) unit
#     indices per replicate (the standard nonparametric cluster bootstrap),
#     which is the only reading consistent with "paired cluster resampling of
#     (camera,frame) units".
# A3. No unit ORDERING is preregistered for the bootstrap. Section 7
#     preregisters a deterministic ordering only for a different object
#     ("deterministic component ordering = (min lineage ID in component)").
#     This oracle sorts the unit keys (repr-order fallback for unorderable
#     mixed key types). Any implementation using a different unit order will
#     produce a different-but-equally-valid SE at the same seed; only the
#     distribution, not the realized value, is spec-determined.
# A4. The RNG is not preregistered. This oracle uses python
#     `random.Random(seed).randrange(n_units)`, one index list per replicate,
#     replicates drawn in order from a single generator. An implementation
#     using numpy's Generator will not reproduce these exact indices.
# A5. "SE undefined => reject" is not given an operational test. This oracle
#     treats "fewer than two replicate differences" and "any non-finite
#     replicate difference or non-finite sd" as undefined. It does NOT treat
#     SE == 0 as undefined: a degenerate-but-defined zero SE (e.g. the
#     constant-shift case above) is a legitimate value under the acceptance
#     rule.
# A6. "degeneracy (<=5 clusters) => reject" appears only in the untagged
#     normative revision text of header item (6), not in the section-7 body,
#     and does not say whether the count is over the full sample or per
#     replicate. This oracle counts DISTINCT units in the full sample, once,
#     before drawing. Note the consequence: a replicate can still draw fewer
#     than six distinct units, and that replicate is kept.
# A7. The sign convention for the paired difference is not stated directly.
#     candidate-minus-incumbent is inferred from "Accept iff Delta-E-hat +
#     k*SE < 0" plus section 4's energy semantics (accept when energy drops).
# A8. Section 7 does not say whether the exact (tracker/prior/ledger) deltas
#     participate in the bootstrap. Read literally they are "computed in
#     closed form, not estimated", so they carry no sampling variance and are
#     excluded from the replicate differences here; SE is the sd of the
#     RENDER-part difference only, while Delta-E-hat includes everything.
# A9. Section 7 does not fix k. It is a required caller argument here.
#
# ---------------------------------------------------------------------------
# REFERENCE_SHA_NOTE
# ---------------------------------------------------------------------------
# This file is a FROZEN ORACLE. It was transcribed from section 7 of
# `research-wiki/operations/elgs-v8-formal-spec.md` (plus the untagged
# normative header item (6)) in a fresh context with no knowledge of any
# implementation.
#
# Freezing rules:
#   * This file MUST NEVER be edited to make an implementation's numbers
#     match. Any such edit destroys its value as an independent oracle.
#   * When this oracle and an implementation disagree, the disagreement is
#     adjudicated AGAINST THE SPEC TEXT, not against either program. The spec
#     sentence quoted in the relevant docstring is the arbiter.
#   * If adjudication shows the SPEC is what changed, revise the spec first,
#     then re-transcribe this file from the revised spec in a fresh context,
#     and record the new sha256 below alongside the spec revision number.
#   * If adjudication shows this ORACLE mis-transcribed a spec sentence, fix
#     the transcription and quote the sentence that proves the fix - never
#     adjust a constant or tolerance to close a numeric gap.
#   * Pin integrity by recording the sha256 of this file in the test that
#     consumes it, so that any edit is surfaced as a deliberate act.
#
# Transcribed against: elgs-v8-formal-spec.md revision 4 (2026-08-11),
# section 7 "Acceptance (heuristic; SNIS estimator - consistent,
# finite-sample biased)" and header item (6).
