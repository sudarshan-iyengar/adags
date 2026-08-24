#!/usr/bin/env python3
"""Compute the FROZEN statistics of the N3V run-level variance study.

Authority: ``research-wiki/operations/n3v-variance-study-spec-2026-08-24.md``.
This script implements that spec's section 5 arithmetic and NOTHING ELSE. It
exists so the numbers are produced by an auditable, re-runnable tool rather
than by hand at read time, and so the estimators the spec FORBIDS cannot be
computed by accident.

What it computes (spec section 5):

  * sample mean and sample standard deviation with the ``n - 1`` denominator;
  * a confidence interval for sigma from the chi-square distribution, with
    the normality assumption stated in the output rather than buried;
  * median and MAD, reported descriptively alongside and never as the
    inferential quantity;
  * the pre-registered within-run contrast ``union - complement``;
  * the sequential stopping rule of section 6, evaluated mechanically.

What it REFUSES to compute, and why (spec section 5.1):

  * any bootstrap over pixels or frames presented as run-level uncertainty —
    frames within a run share the run and are not independent trained
    models; on the n=3 data a frame bootstrap reads about 6x too small;
  * ``range / sqrt(n)`` — not an estimator of anything; the unbiased
    range-based estimator is ``range / d2(n)``, and that ``d2(3) = 1.693``
    happens to sit near ``sqrt(3)`` is a coincidence that does not survive
    to n=6;
  * any p-value. The study is ESTIMATION, not hypothesis testing, which is
    the reason its sequential rule needs no alpha adjustment — and that
    reasoning binds the reporting too.

Chi-square quantiles are embedded rather than imported: the spec needs
exactly df in {2, 5, 8} (n in {3, 6, 9}), scipy is not a guaranteed
dependency of the Apollo image, and three constants are easier to audit
than a dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

#: Two-sided 95% chi-square quantiles, keyed by degrees of freedom.
#: (lower, upper) = (chi2_{0.025, df}, chi2_{0.975, df}).
CHI2_95: dict[int, tuple[float, float]] = {
    2: (0.05064, 7.37776),
    5: (0.83121, 12.83250),
    8: (2.17973, 17.53455),
}

#: The spec's frozen decision constants.
DELTA_STAR_DB = 0.30
SIGMA_DECISION_DB = 0.1672
Z_ALPHA_2 = 1.959964
Z_BETA = 0.841621


def sigma_ci(s: float, n: int, *, confidence: str = "95%") -> tuple[float, float]:
    """Chi-square confidence interval for sigma. Assumes normality."""

    df = n - 1
    if df not in CHI2_95:
        raise ValueError(
            f"no embedded chi-square quantiles for df={df} (have {sorted(CHI2_95)}); "
            "the frozen spec only ever needs n in {3, 6, 9}"
        )
    lo_q, hi_q = CHI2_95[df]
    return s * math.sqrt(df / hi_q), s * math.sqrt(df / lo_q)


def mad(values: list[float]) -> float:
    """Median absolute deviation. Descriptive companion, never inferential."""

    centre = statistics.median(values)
    return statistics.median([abs(v - centre) for v in values])


def replicates_per_arm(sigma: float, delta: float) -> int:
    """Two-sample per-arm n for 80% power at alpha=0.05 two-sided.

    Normal approximation plus the customary small-sample correction; the
    spec reports the corrected figure, so the correction is applied here
    rather than left to the reader.
    """

    if delta <= 0:
        raise ValueError("delta must be positive")
    n = 2.0 * (Z_ALPHA_2 + Z_BETA) ** 2 * (sigma / delta) ** 2
    return int(math.ceil(n)) + 2


def describe(name: str, values: list[float]) -> dict:
    n = len(values)
    s = statistics.stdev(values) if n > 1 else float("nan")
    out = {
        "endpoint": name,
        "n": n,
        "values": values,
        "mean": statistics.mean(values),
        "sd_n_minus_1": s,
        "spread_max_minus_min": max(values) - min(values),
        "median": statistics.median(values),
        "mad": mad(values),
    }
    if n > 1 and (n - 1) in CHI2_95:
        lo, hi = sigma_ci(s, n)
        out["sigma_ci_95"] = [lo, hi]
        out["sigma_ci_ratio"] = hi / lo
        out["sigma_ci_assumes"] = (
            "normality of the RUN-LEVEL endpoint; n cannot verify it and this "
            "is disclosed rather than tested-and-passed"
        )
        out["replicates_per_arm_at_delta_star"] = replicates_per_arm(s, DELTA_STAR_DB)
        out["replicates_per_arm_at_sigma_upper_limit"] = replicates_per_arm(hi, DELTA_STAR_DB)
    return out


def stopping_rule(sd_contrast: float, n: int) -> dict:
    """Spec section 6, evaluated mechanically rather than by eye."""

    if n < 6:
        return {"applies": False, "reason": f"the rule is evaluated at n=6; n={n}"}
    lo, hi = sigma_ci(sd_contrast, n)
    straddles = lo <= SIGMA_DECISION_DB <= hi
    return {
        "applies": True,
        "sigma_decision_db": SIGMA_DECISION_DB,
        "contrast_sigma_ci_95": [lo, hi],
        "ci_straddles_sigma_decision": straddles,
        "extend_to_n9": straddles,
        "note": (
            "ESTIMATION, not hypothesis testing: no alpha adjustment is required "
            "and NO p-value may be attached to this study's output"
        ),
    }


def cmd_analyse(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    union = [float(x) for x in payload["all_events_union"]]
    complement = [float(x) for x in payload["complement"]]
    if len(union) != len(complement):
        raise ValueError(
            f"union has {len(union)} values but complement has {len(complement)}; "
            "the contrast is a WITHIN-RUN difference and needs them paired"
        )
    contrast = [u - c for u, c in zip(union, complement)]

    results = {
        "spec": "research-wiki/operations/n3v-variance-study-spec-2026-08-24.md",
        "delta_star_db": DELTA_STAR_DB,
        "cohort": payload.get("cohort", "unspecified"),
        "endpoints": [
            describe("all_events_union (PRIMARY)", union),
            describe("complement (control)", complement),
            describe("union - complement (CO-PRIMARY, pre-registered prediction)", contrast),
        ],
        "forbidden_and_not_computed": [
            "pixel or frame bootstrap presented as run-level uncertainty",
            "range / sqrt(n)",
            "any p-value (this is estimation, not testing)",
        ],
    }
    for extra in ("whole_frame", "pooled_clamped_psnr"):
        if extra in payload:
            results["endpoints"].append(describe(f"{extra} (control)", [float(x) for x in payload[extra]]))

    sd_contrast = statistics.stdev(contrast) if len(contrast) > 1 else float("nan")
    results["stopping_rule"] = stopping_rule(sd_contrast, len(contrast))

    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.out}", file=sys.stderr)
    return 0


def cmd_self_test(_args: argparse.Namespace) -> int:
    """Reproduce the spec's own tabulated numbers from the recorded n=3 data."""

    checks: list[tuple[str, bool, str]] = []

    union = [31.4059, 31.8043, 31.9004]
    complement = [32.8229, 32.9124, 33.3037]
    contrast = [u - c for u, c in zip(union, complement)]

    s_u = statistics.stdev(union)
    checks.append(("recorded n=3 union sd is 0.262198", abs(s_u - 0.262198) < 5e-7, f"{s_u:.6f}"))

    s_c = statistics.stdev(contrast)
    checks.append(("recorded n=3 contrast sd is 0.174523", abs(s_c - 0.174523) < 5e-7, f"{s_c:.6f}"))

    red = 100.0 * (1 - s_c / s_u)
    checks.append(("contrast reduces sd by 33.4%", abs(red - 33.4) < 0.1, f"{red:.1f}%"))

    lo, hi = sigma_ci(s_u, 3)
    checks.append(("n=3 sigma CI is [0.1365, 1.6478]",
                   abs(lo - 0.1365) < 5e-4 and abs(hi - 1.6478) < 5e-4, f"[{lo:.4f}, {hi:.4f}]"))
    checks.append(("n=3 CI ratio is 12.07x", abs(hi / lo - 12.07) < 0.02, f"{hi/lo:.2f}x"))

    for n, want_width, want_ratio in ((6, 1.83, 3.93), (9, 1.24, 2.84)):
        lo_f, hi_f = sigma_ci(1.0, n)
        checks.append((f"n={n} CI width is {want_width}*s", abs((hi_f - lo_f) - want_width) < 0.01,
                       f"{hi_f - lo_f:.2f}*s"))
        checks.append((f"n={n} CI ratio is {want_ratio}x", abs(hi_f / lo_f - want_ratio) < 0.01,
                       f"{hi_f/lo_f:.2f}x"))

    checks.append(("union endpoint needs 14 replicates/arm at delta*=0.30",
                   replicates_per_arm(s_u, DELTA_STAR_DB) == 14,
                   str(replicates_per_arm(s_u, DELTA_STAR_DB))))
    # 8, not the 7 the spec first tabulated. The raw figure is 5.3127; a
    # SAMPLE SIZE must round UP, and round(5.3127) = 5 would plan for fewer
    # replicates than the power calculation requires. Corrected append-only
    # in the spec rather than matched here.
    checks.append(("contrast endpoint needs 8 replicates/arm at delta*=0.30 (spec said 7)",
                   replicates_per_arm(s_c, DELTA_STAR_DB) == 8,
                   str(replicates_per_arm(s_c, DELTA_STAR_DB))))

    # The 2.8%-for-296x observation that near-falsifies the pool-more-pixels lever.
    whole = [32.8172, 32.9081, 33.2981]
    drop = 100.0 * (1 - (max(whole) - min(whole)) / (max(union) - min(union)))
    checks.append(("296x the pixel-times reduces the spread by only ~2.8%",
                   abs(drop - 2.8) < 0.2, f"{drop:.1f}%"))

    failed = 0
    for name, ok, got in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  -- {got}")
        failed += (not ok)
    print(f"\nSELF-TEST {'PASSED' if failed == 0 else 'FAILED'}: "
          f"{len(checks) - failed}/{len(checks)} checks")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="n3v_variance_analysis.py",
        description="Frozen statistics for the N3V run-level variance study.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("analyse", help="compute the frozen statistics from an endpoint JSON")
    a.add_argument("--inputs", required=True,
                   help='JSON with {"all_events_union": [...], "complement": [...], ...}')
    a.add_argument("--out", default=None)
    a.set_defaults(func=cmd_analyse)
    t = sub.add_parser("self-test", help="reproduce the spec's tabulated numbers; needs no data")
    t.set_defaults(func=cmd_self_test)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
