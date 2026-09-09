#!/usr/bin/env python
"""Pre-registered analysis for the real-data (N3V `cut_roasted_beef`) gate comparison.

Two treatment arms plus a positive-control arm:

    U      ungated baseline
    G      gated (the intervention under test)
    GMIS   gated with a deliberately mis-specified window (positive control)

Everything this script decides is fixed by the frozen ``SPEC`` dictionary below.
No threshold, window, level, margin, or verdict rule is read from the data, the
command line, or the environment, so the sha256 of this file can be recorded
before any gated cell exists and the analysis is fully determined in advance.
``--print-spec`` dumps ``SPEC`` as JSON for that record.

Inputs are the per-frame profiles written by ``scripts/event_region_frame_profile.py``
(``<run_dir>/f_box_profile.json``), an optional ``<run_dir>/precondition.json``
carrying the mechanism-exercised counters, and an optional ``<run_dir>/summary.json``
for the descriptive final point count.

The Student-t distribution is implemented here in pure Python (regularized
incomplete beta + bisection) so that the frozen numbers do not depend on whether
scipy is installed on the machine that runs the analysis.
"""

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np

# --------------------------------------------------------------------------
# FROZEN SPECIFICATION
# --------------------------------------------------------------------------

SPEC = {
    "spec_id": "realdata_gate_analysis",
    "spec_version": "1.0.0",
    "scene": "cut_roasted_beef",
    "event_name": "F_blade_over_beef_reveal",
    "arms": ["U", "G", "GMIS"],
    "arm_roles": {
        "U": "ungated baseline",
        "G": "gated intervention",
        "GMIS": "gated with mis-specified window (positive control)",
    },
    "profile_filename": "f_box_profile.json",
    "precondition_filename": "precondition.json",
    "summary_filename": "summary.json",
    "final_points_keys": ["best_val/points/total", "final/points/total"],
    "expected_n_frames": 300,
    "pooling_rule": "pooled_psnr = -10*log10(mean_over_frames(10**(-psnr_f/10)))",
    "pooling_validity": "every frame contributes the same box pixel count",
    "endpoints": {
        "P1": {
            "label": "ghost",
            "source": "event",
            "frames_inclusive": [158, 187],
            "role": "primary",
        },
        "P2": {
            "label": "return_clean",
            "source": "event",
            "frames_inclusive": [190, 199],
            "role": "primary",
        },
        "S1": {
            "label": "return_curated",
            "source": "event",
            "frames_inclusive": [190, 209],
            "role": "secondary",
        },
        "H1": {
            "label": "harm_region",
            "source": "event",
            "frames_inclusive": [100, 157],
            "role": "harm_guard",
        },
        "H2": {
            "label": "harm_whole",
            "source": "whole_frame",
            "frames_inclusive": None,
            "role": "harm_guard",
        },
        "C1": {
            "label": "control_window",
            "source": "event",
            "frames_inclusive": [118, 147],
            "role": "positive_control",
        },
        "CAP": {
            "label": "final_points",
            "source": "summary",
            "frames_inclusive": None,
            "role": "descriptive",
        },
    },
    "primary_endpoints": ["P1", "P2"],
    "secondary_endpoints": ["S1"],
    "harm_endpoints": ["H1", "H2"],
    "control_endpoint": "C1",
    "descriptive_endpoints": ["CAP"],
    "psnr_endpoints": ["P1", "P2", "S1", "H1", "H2", "C1"],
    "ALPHA_PRIMARY": 0.025,
    "MULTIPLICITY": "Bonferroni over P1 and P2 (2 primaries at 0.05 family-wise)",
    "CI_CONF": 0.975,
    "CI_LEVEL_NOTE": "two-sided 97.5% CI, i.e. t quantile at 1 - 0.025/2 = 0.9875",
    "TEST": "Welch two-sample t, two-sided, Welch-Satterthwaite df",
    "SECONDARY_AND_HARM_LEVEL": 0.975,
    "CONTROL_MIN_EFFECT_DB": 1.0,
    "CONTROL_RULE": (
        "valid iff the 97.5% CI of (GMIS - U) on C1 lies entirely below 0 "
        "AND the point estimate is <= -CONTROL_MIN_EFFECT_DB"
    ),
    "PLACEBO_QUANTILE": 0.95,
    "PLACEBO_RULE": (
        "over all balanced splits of the U cells into two equal halves "
        "(mirror splits de-duplicated), the difference of half-means; "
        "report the PLACEBO_QUANTILE quantile of the absolute differences"
    ),
    "PLACEBO_PERCENTILE_METHOD": "linear (numpy.percentile default)",
    "HARM_MARGIN_DB": 0.15,
    "HARM_RULE": (
        "FAIL if on H1 or H2 the 97.5% CI upper bound of (U - G) exceeds HARM_MARGIN_DB"
    ),
    "HARM_GUARD_ON_INSUFFICIENT_N": "fail",
    "SIZING": {
        "method": "internal pilot, Wittes-Brittain blinded within-arm pooled sd",
        "s_p": "sqrt((s_U^2 + s_G^2)/2) on each primary",
        "formula": "n2 = ceil(2*k*(s_p/DELTA)^2 + z_a^2/4)",
        "DELTA": 0.30,
        "K": 9.5049,
        "K_NOTE": "k = (z_{alpha/2} + z_beta)^2 with alpha = 0.025 and power 0.80",
        "Z_ALPHA_HALF": 2.2414,
        "Z_BETA": 0.8416,
        "Z_A": 2.2414,
        "POWER": 0.80,
        "N2_CAP": 60,
        "decision": (
            "n2_max <= current n per arm -> wave1_final; "
            "n2_max <= N2_CAP -> run_wave2_to_n2; else feasibility_stop"
        ),
        "current_n_per_arm": "min(n_U, n_G) in the analysis set",
    },
    "TOST": {
        "alpha": 0.025,
        "margin_db": 0.30,
        "endpoints": ["P1", "P2"],
        "rule": "pass iff both one-sided p < alpha (equivalently the 95% CI lies inside +-margin)",
        "zero_variance_rule": "with se = 0, TOST passes iff |diff| < margin",
    },
    "ZERO_VARIANCE_RULE": (
        "with se = 0 the CI is the degenerate interval [diff, diff]; "
        "p = 1.0 if diff == 0 else 0.0"
    ),
    "VERDICTS": [
        "BENEFIT",
        "BENEFIT_NOT_SEPARABLE",
        "HARM",
        "EQUIVALENT",
        "NO_DETECTED_DIFFERENCE",
        "DESIGN_WITHOUT_POWER",
        "FEASIBILITY_STOP",
    ],
    "VERDICT_RULE": (
        "DESIGN_WITHOUT_POWER overrides everything when the positive control is invalid; "
        "FEASIBILITY_STOP then overrides when sizing says feasibility_stop and the current "
        "wave is the last; otherwise, in order: "
        "BENEFIT (CI entirely > 0 AND |G-U| > placebo q95 AND control valid AND harm guard "
        "passes AND correctness-contrast CI entirely > 0); "
        "BENEFIT_NOT_SEPARABLE (same without the correctness contrast); "
        "HARM (CI entirely < 0); EQUIVALENT (TOST passes); NO_DETECTED_DIFFERENCE"
    ),
    "ANALYSIS_SETS": ["itt", "mechanism_exercised"],
    "PRIMARY_ANALYSIS_SET": "itt",
    "MIN_GATED_SURVIVING": 1000,
    "MIN_GATED_FBOX": 100,
    "MIN_FRAMES_PRESENCE_ZERO": 1,
    "MECHANISM_EXERCISED_RULE": (
        "U cells always pass; a G or GMIS cell passes iff precondition.json exists and "
        "gated_rows_final >= MIN_GATED_SURVIVING and "
        "gated_rows_fbox_frame150 >= MIN_GATED_FBOX and "
        "frames_presence_zero >= MIN_FRAMES_PRESENCE_ZERO"
    ),
    "PRECONDITION_FIELDS": [
        "gated_rows_seeding",
        "n_rows_seeding",
        "gated_rows_final",
        "n_rows_final",
        "gated_rows_fbox_frame150",
        "frames_presence_zero",
        "reserved_units",
        "training_units_total",
    ],
    "RESERVED_UNIT_RULE": (
        "every cell that reports reserved_units must report the same count; "
        "an inconsistency is a blocking error"
    ),
    "t_distribution": "implemented in-file (regularized incomplete beta + bisection); scipy not required",
}

CONTROL_MIN_EFFECT_DB = SPEC["CONTROL_MIN_EFFECT_DB"]
PLACEBO_QUANTILE = SPEC["PLACEBO_QUANTILE"]
HARM_MARGIN_DB = SPEC["HARM_MARGIN_DB"]
N2_CAP = SPEC["SIZING"]["N2_CAP"]
DELTA = SPEC["SIZING"]["DELTA"]
K_SIZING = SPEC["SIZING"]["K"]
Z_A = SPEC["SIZING"]["Z_A"]
TOST_MARGIN_DB = SPEC["TOST"]["margin_db"]
TOST_ALPHA = SPEC["TOST"]["alpha"]
CI_CONF = SPEC["CI_CONF"]
MIN_GATED_SURVIVING = SPEC["MIN_GATED_SURVIVING"]
MIN_GATED_FBOX = SPEC["MIN_GATED_FBOX"]
MIN_FRAMES_PRESENCE_ZERO = SPEC["MIN_FRAMES_PRESENCE_ZERO"]


# --------------------------------------------------------------------------
# Student-t distribution, scipy-free
# --------------------------------------------------------------------------


def _betacf(a, b, x, itmax=400, eps=3.0e-16, fpmin=1.0e-300):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc_reg(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """P(T > t) for Student-t with df degrees of freedom."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    x = df / (df + t * t)
    tail = 0.5 * betainc_reg(0.5 * df, 0.5, x)
    return tail if t > 0 else 1.0 - tail


def t_cdf(t, df):
    return 1.0 - t_sf(t, df)


def t_two_sided_p(t, df):
    """Two-sided p-value for a t statistic."""
    if math.isinf(t):
        return 0.0
    x = df / (df + t * t)
    return betainc_reg(0.5 * df, 0.5, x)


def t_ppf(q, df):
    """Quantile function for Student-t, by bisection on the cdf."""
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must lie strictly inside (0, 1)")
    if abs(q - 0.5) < 1e-15:
        return 0.0
    lo, hi = -1.0, 1.0
    while t_cdf(lo, df) > q:
        lo *= 2.0
        if lo < -1e12:
            break
    while t_cdf(hi, df) < q:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Pooling and endpoint extraction
# --------------------------------------------------------------------------


def pooled_psnr(psnr_values):
    """Pool per-frame PSNRs by averaging the underlying MSEs.

    Valid because every frame contributes the same box pixel count.
    """
    arr = np.asarray(list(psnr_values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot pool an empty frame set")
    mse = np.power(10.0, -arr / 10.0)
    mean_mse = float(mse.mean())
    if mean_mse <= 0.0:
        return float("inf")
    return float(-10.0 * math.log10(mean_mse))


def select_window(frames, per_frame, lo, hi, what):
    """Values of `per_frame` for absolute frames lo..hi inclusive."""
    index = {f: i for i, f in enumerate(frames)}
    missing = [f for f in range(lo, hi + 1) if f not in index]
    if missing:
        raise ValueError(
            f"{what}: profile is missing {len(missing)} frame(s) in the window "
            f"[{lo}, {hi}]; first missing frame is {missing[0]}"
        )
    return [per_frame[index[f]] for f in range(lo, hi + 1)]


def _lookup_path(root, key):
    if not isinstance(root, dict):
        return None
    if key in root:
        return root[key]
    cur = root
    for part in key.split("/"):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def read_final_points(run_dir):
    path = os.path.join(run_dir, SPEC["summary_filename"])
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (ValueError, OSError):
        return None
    roots = []
    if isinstance(data, dict):
        if isinstance(data.get("summary"), dict):
            roots.append(data["summary"])
        roots.append(data)
    for root in roots:
        for key in SPEC["final_points_keys"]:
            value = _lookup_path(root, key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return int(value)
    return None


def read_precondition(run_dir):
    path = os.path.join(run_dir, SPEC["precondition_filename"])
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return {k: data.get(k) for k in SPEC["PRECONDITION_FIELDS"] if k in data}


def mechanism_exercised(arm, precondition):
    """(passes, reason) for the mechanism-exercised analysis set."""
    if arm == "U":
        return True, "U arm passes by construction"
    if precondition is None:
        return False, f"no {SPEC['precondition_filename']}"
    failures = []
    surviving = precondition.get("gated_rows_final")
    fbox = precondition.get("gated_rows_fbox_frame150")
    zeros = precondition.get("frames_presence_zero")
    if surviving is None or surviving < MIN_GATED_SURVIVING:
        failures.append(f"gated_rows_final={surviving} < {MIN_GATED_SURVIVING}")
    if fbox is None or fbox < MIN_GATED_FBOX:
        failures.append(f"gated_rows_fbox_frame150={fbox} < {MIN_GATED_FBOX}")
    if zeros is None or zeros < MIN_FRAMES_PRESENCE_ZERO:
        failures.append(f"frames_presence_zero={zeros} < {MIN_FRAMES_PRESENCE_ZERO}")
    if failures:
        return False, "; ".join(failures)
    return True, "precondition satisfied"


def load_cell(entry):
    """Read one manifest entry into a cell record with all endpoint values."""
    for field in ("arm", "seed", "run_dir", "status"):
        if field not in entry:
            raise ValueError(f"manifest cell is missing required field '{field}': {entry}")
    arm = entry["arm"]
    if arm not in SPEC["arms"]:
        raise ValueError(f"unknown arm {arm!r}; expected one of {SPEC['arms']}")
    status = entry["status"]
    if status not in ("complete", "failed"):
        raise ValueError(f"unknown status {status!r} for {arm}/seed {entry['seed']}")
    run_dir = entry["run_dir"]
    cell = {
        "arm": arm,
        "seed": entry["seed"],
        "run_dir": run_dir,
        "status": status,
        "endpoints": {},
        "final_points": None,
        "precondition": None,
        "mechanism_exercised": False,
        "mechanism_reason": "cell not complete",
        "n_frames": None,
    }
    if status == "failed":
        return cell

    profile_path = os.path.join(run_dir, SPEC["profile_filename"])
    if not os.path.isfile(profile_path):
        raise FileNotFoundError(
            f"cell {arm}/seed {entry['seed']} is marked 'complete' but "
            f"{profile_path} does not exist"
        )
    with open(profile_path) as fh:
        profile = json.load(fh)
    for key in ("frames", "whole_frame_psnr", "events"):
        if key not in profile:
            raise ValueError(f"{profile_path}: missing key '{key}'")
    frames = [int(f) for f in profile["frames"]]
    whole = [float(v) for v in profile["whole_frame_psnr"]]
    if len(frames) != len(whole):
        raise ValueError(
            f"{profile_path}: frames ({len(frames)}) and whole_frame_psnr "
            f"({len(whole)}) have different lengths"
        )
    event_name = SPEC["event_name"]
    if event_name not in profile["events"]:
        raise ValueError(
            f"{profile_path}: event {event_name!r} not found; "
            f"available: {sorted(profile['events'])}"
        )
    event = profile["events"][event_name]
    if "per_frame_psnr" not in event:
        raise ValueError(f"{profile_path}: event {event_name!r} has no 'per_frame_psnr'")
    per_frame = [float(v) for v in event["per_frame_psnr"]]
    if len(per_frame) != len(frames):
        raise ValueError(
            f"{profile_path}: event per_frame_psnr ({len(per_frame)}) does not align "
            f"with frames ({len(frames)})"
        )
    cell["n_frames"] = len(frames)
    cell["bbox"] = event.get("bbox")

    for key, spec in SPEC["endpoints"].items():
        if spec["source"] == "event":
            lo, hi = spec["frames_inclusive"]
            what = f"cell {arm}/seed {entry['seed']} endpoint {key}"
            cell["endpoints"][key] = pooled_psnr(
                select_window(frames, per_frame, lo, hi, what)
            )
        elif spec["source"] == "whole_frame":
            cell["endpoints"][key] = pooled_psnr(whole)

    cell["final_points"] = read_final_points(run_dir)
    cell["endpoints"]["CAP"] = (
        float(cell["final_points"]) if cell["final_points"] is not None else None
    )
    cell["precondition"] = read_precondition(run_dir)
    passes, reason = mechanism_exercised(arm, cell["precondition"])
    cell["mechanism_exercised"] = passes
    cell["mechanism_reason"] = reason
    return cell


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def welch(a, b, conf=CI_CONF):
    """Welch two-sample t on (mean(a) - mean(b)), two-sided, at confidence `conf`."""
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    n1, n2 = len(a), len(b)
    out = {
        "n1": n1,
        "n2": n2,
        "mean1": float(np.mean(a)) if n1 else None,
        "mean2": float(np.mean(b)) if n2 else None,
        "sd1": float(np.std(a, ddof=1)) if n1 >= 2 else None,
        "sd2": float(np.std(b, ddof=1)) if n2 >= 2 else None,
        "diff": None,
        "se": None,
        "df": None,
        "t": None,
        "p": None,
        "ci_low": None,
        "ci_high": None,
        "conf": conf,
        "insufficient_n": n1 < 2 or n2 < 2,
    }
    if out["insufficient_n"]:
        if n1 and n2:
            out["diff"] = out["mean1"] - out["mean2"]
        return out
    v1 = out["sd1"] ** 2
    v2 = out["sd2"] ** 2
    diff = out["mean1"] - out["mean2"]
    out["diff"] = diff
    se2 = v1 / n1 + v2 / n2
    if se2 <= 0.0:
        out["se"] = 0.0
        out["df"] = float(n1 + n2 - 2)
        out["t"] = None
        out["p"] = 1.0 if diff == 0.0 else 0.0
        out["ci_low"] = diff
        out["ci_high"] = diff
        out["zero_variance"] = True
        return out
    se = math.sqrt(se2)
    df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    tstat = diff / se
    tcrit = t_ppf(1.0 - (1.0 - conf) / 2.0, df)
    out.update(
        {
            "se": se,
            "df": df,
            "t": tstat,
            "p": t_two_sided_p(abs(tstat), df),
            "ci_low": diff - tcrit * se,
            "ci_high": diff + tcrit * se,
            "t_crit": tcrit,
            "zero_variance": False,
        }
    )
    return out


def ci_entirely_above_zero(res):
    return (
        not res["insufficient_n"]
        and res["ci_low"] is not None
        and res["ci_low"] > 0.0
    )


def ci_entirely_below_zero(res):
    return (
        not res["insufficient_n"]
        and res["ci_high"] is not None
        and res["ci_high"] < 0.0
    )


def balanced_splits(n):
    """All balanced two-half splits of range(n), mirror duplicates removed.

    For n = 8 this returns C(7, 3) = 35 splits.
    """
    if n < 2 or n % 2 != 0:
        return []
    half = n // 2
    splits = []
    for combo in itertools.combinations(range(n), half):
        if combo[0] != 0:  # de-duplicate mirror splits by pinning index 0 to side A
            continue
        other = tuple(i for i in range(n) if i not in combo)
        splits.append((combo, other))
    return splits


def placebo_distribution(values):
    """Half-mean differences over all balanced splits, and the frozen quantile."""
    vals = [float(v) for v in values]
    splits = balanced_splits(len(vals))
    if not splits:
        return {
            "n_cells": len(vals),
            "n_splits": 0,
            "quantile": PLACEBO_QUANTILE,
            "abs_quantile": None,
            "reason": "balanced splits require an even number of U cells (>= 2)",
        }
    diffs = [
        float(np.mean([vals[i] for i in a]) - np.mean([vals[i] for i in b]))
        for a, b in splits
    ]
    abs_diffs = [abs(d) for d in diffs]
    return {
        "n_cells": len(vals),
        "n_splits": len(splits),
        "quantile": PLACEBO_QUANTILE,
        "abs_quantile": float(np.percentile(abs_diffs, 100.0 * PLACEBO_QUANTILE)),
        "max_abs": float(max(abs_diffs)),
        "diffs": diffs,
    }


def sizing_n2(sd_u, sd_g):
    """Wittes-Brittain internal-pilot re-estimated per-arm sample size."""
    if sd_u is None or sd_g is None:
        return None, None
    s_p = math.sqrt((sd_u ** 2 + sd_g ** 2) / 2.0)
    n2 = math.ceil(2.0 * K_SIZING * (s_p / DELTA) ** 2 + Z_A ** 2 / 4.0)
    return s_p, int(n2)


def tost(res, margin=TOST_MARGIN_DB, alpha=TOST_ALPHA):
    """Two one-sided tests for equivalence within +-margin."""
    out = {"margin": margin, "alpha": alpha, "p_lower": None, "p_upper": None,
           "passes": False, "ci_low": None, "ci_high": None}
    if res["insufficient_n"] or res["diff"] is None:
        out["reason"] = "insufficient n"
        return out
    diff = res["diff"]
    if res["se"] == 0.0:
        out["passes"] = abs(diff) < margin
        out["ci_low"] = diff
        out["ci_high"] = diff
        out["reason"] = "zero variance rule"
        return out
    se, df = res["se"], res["df"]
    t_lower = (diff + margin) / se   # H01: diff <= -margin
    t_upper = (diff - margin) / se   # H02: diff >= +margin
    p_lower = t_sf(t_lower, df)
    p_upper = t_cdf(t_upper, df)
    tcrit = t_ppf(1.0 - alpha, df)
    out.update(
        {
            "p_lower": p_lower,
            "p_upper": p_upper,
            "passes": bool(p_lower < alpha and p_upper < alpha),
            "ci_low": diff - tcrit * se,
            "ci_high": diff + tcrit * se,
            "ci_conf": 1.0 - 2.0 * alpha,
        }
    )
    return out


# --------------------------------------------------------------------------
# Analysis over one cell set
# --------------------------------------------------------------------------


def _arm_values(cells, arm, endpoint):
    return [
        c["endpoints"][endpoint]
        for c in cells
        if c["arm"] == arm and c["endpoints"].get(endpoint) is not None
    ]


def analyse(cells, set_name, last_wave):
    """Run the whole frozen analysis over one set of complete cells."""
    arms = {a: [c for c in cells if c["arm"] == a] for a in SPEC["arms"]}
    n_per_arm = {a: len(v) for a, v in arms.items()}

    # 1. G vs U on P1, P2, S1, H1, H2
    g_vs_u = {}
    for key in ["P1", "P2", "S1", "H1", "H2"]:
        g_vs_u[key] = welch(_arm_values(cells, "G", key), _arm_values(cells, "U", key))

    # 2. Positive control: GMIS vs U on C1
    control = welch(_arm_values(cells, "GMIS", "C1"), _arm_values(cells, "U", "C1"))
    control_valid = bool(
        ci_entirely_below_zero(control)
        and control["diff"] is not None
        and control["diff"] <= -CONTROL_MIN_EFFECT_DB
    )
    control_block = {
        "endpoint": "C1",
        "contrast": "GMIS - U",
        "result": control,
        "min_effect_db": CONTROL_MIN_EFFECT_DB,
        "valid": control_valid,
        "reason": (
            "valid"
            if control_valid
            else "CI not entirely below 0 and/or point estimate above -%.2f dB"
            % CONTROL_MIN_EFFECT_DB
        ),
    }

    # 3. Correctness contrast: G vs GMIS on the primaries
    correctness = {}
    for key in SPEC["primary_endpoints"]:
        correctness[key] = welch(
            _arm_values(cells, "G", key), _arm_values(cells, "GMIS", key)
        )

    # 4. Placebo distribution over balanced splits of the U cells
    placebo = {
        key: placebo_distribution(_arm_values(cells, "U", key))
        for key in SPEC["psnr_endpoints"]
    }

    # 5. Harm guard on (U - G)
    harm = {}
    harm_pass = True
    for key in SPEC["harm_endpoints"]:
        res = welch(_arm_values(cells, "U", key), _arm_values(cells, "G", key))
        if res["insufficient_n"]:
            ok = SPEC["HARM_GUARD_ON_INSUFFICIENT_N"] != "fail"
            reason = "insufficient n -> fail-closed"
        else:
            ok = res["ci_high"] <= HARM_MARGIN_DB
            reason = "ci_high %.4f vs margin %.2f" % (res["ci_high"], HARM_MARGIN_DB)
        harm[key] = {"contrast": "U - G", "result": res, "passes": bool(ok), "reason": reason}
        harm_pass = harm_pass and bool(ok)

    # 6. Sizing
    current_n = min(n_per_arm["U"], n_per_arm["G"]) if n_per_arm["G"] else 0
    sizing = {"current_n_per_arm": current_n, "per_endpoint": {}, "n2_max": None}
    n2_values = []
    for key in SPEC["primary_endpoints"]:
        res = g_vs_u[key]
        s_p, n2 = sizing_n2(res["sd2"], res["sd1"])  # sd2 = U arm, sd1 = G arm
        sizing["per_endpoint"][key] = {"s_p": s_p, "n2": n2, "sd_U": res["sd2"], "sd_G": res["sd1"]}
        if n2 is not None:
            n2_values.append(n2)
    if n2_values:
        sizing["n2_max"] = max(n2_values)
        if sizing["n2_max"] <= current_n:
            sizing["decision"] = "wave1_final"
        elif sizing["n2_max"] <= N2_CAP:
            sizing["decision"] = "run_wave2_to_n2"
        else:
            sizing["decision"] = "feasibility_stop"
    else:
        sizing["decision"] = "not_estimable"

    # 7. Equivalence
    equivalence = {key: tost(g_vs_u[key]) for key in SPEC["primary_endpoints"]}

    # 8. Verdicts
    verdicts = {}
    for key in SPEC["primary_endpoints"]:
        res = g_vs_u[key]
        pl = placebo[key]["abs_quantile"]
        above_placebo = (
            res["diff"] is not None
            and pl is not None
            and abs(res["diff"]) > pl
        )
        base_benefit = (
            ci_entirely_above_zero(res) and above_placebo and control_valid and harm_pass
        )
        separable = ci_entirely_above_zero(correctness[key])
        if not control_valid:
            verdict = "DESIGN_WITHOUT_POWER"
        elif sizing["decision"] == "feasibility_stop" and last_wave:
            verdict = "FEASIBILITY_STOP"
        elif base_benefit and separable:
            verdict = "BENEFIT"
        elif base_benefit:
            verdict = "BENEFIT_NOT_SEPARABLE"
        elif ci_entirely_below_zero(res):
            verdict = "HARM"
        elif equivalence[key]["passes"]:
            verdict = "EQUIVALENT"
        else:
            verdict = "NO_DETECTED_DIFFERENCE"
        verdicts[key] = {
            "verdict": verdict,
            "ci_above_zero": ci_entirely_above_zero(res),
            "ci_below_zero": ci_entirely_below_zero(res),
            "above_placebo_q95": bool(above_placebo),
            "placebo_abs_q95": pl,
            "control_valid": control_valid,
            "harm_guard_passes": harm_pass,
            "correctness_separable": bool(separable),
            "tost_passes": equivalence[key]["passes"],
        }

    return {
        "set": set_name,
        "n_per_arm": n_per_arm,
        "cells": [
            {"arm": c["arm"], "seed": c["seed"], "run_dir": c["run_dir"]} for c in cells
        ],
        "g_vs_u": g_vs_u,
        "positive_control": control_block,
        "correctness_contrast": correctness,
        "placebo": placebo,
        "harm_guard": {"per_endpoint": harm, "passes": harm_pass, "margin_db": HARM_MARGIN_DB},
        "sizing": sizing,
        "equivalence": equivalence,
        "verdicts": verdicts,
        "capacity_descriptive": {
            a: {
                "n": len(_arm_values(cells, a, "CAP")),
                "mean_final_points": (
                    float(np.mean(_arm_values(cells, a, "CAP")))
                    if _arm_values(cells, a, "CAP")
                    else None
                ),
            }
            for a in SPEC["arms"]
        },
    }


def reserved_unit_check(cells):
    reported = [
        (c, c["precondition"]["reserved_units"])
        for c in cells
        if c.get("precondition") and c["precondition"].get("reserved_units") is not None
    ]
    out = {
        "n_reporting": len(reported),
        "n_cells": len(cells),
        "reported_as": "%d/%d" % (len(reported), len(cells)),
        "values": {},
        "consistent": True,
        "reserved_units": None,
        "training_units_total": None,
    }
    for cell, value in reported:
        out["values"]["%s/seed%s" % (cell["arm"], cell["seed"])] = value
    distinct = sorted({v for _, v in reported})
    if len(distinct) > 1:
        out["consistent"] = False
        out["message"] = (
            "cells disagree on reserved_units: %s" % distinct
        )
    elif distinct:
        out["reserved_units"] = distinct[0]
        totals = sorted(
            {
                c["precondition"]["training_units_total"]
                for c in cells
                if c.get("precondition")
                and c["precondition"].get("training_units_total") is not None
            }
        )
        if len(totals) == 1:
            out["training_units_total"] = totals[0]
            out["reserved_fraction"] = "%d/%d" % (distinct[0], totals[0])
        elif len(totals) > 1:
            out["consistent"] = False
            out["message"] = "cells disagree on training_units_total: %s" % totals
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.{digits}f}"
    return str(value)


def _ci(res):
    if res["insufficient_n"] or res["ci_low"] is None:
        return "-"
    return "[%s, %s]" % (_fmt(res["ci_low"]), _fmt(res["ci_high"]))


def markdown_report(report):
    lines = []
    lines.append("# Real-data gate analysis (%s, %s)" % (SPEC["scene"], SPEC["event_name"]))
    lines.append("")
    lines.append("spec %s v%s | wave %s%s" % (
        SPEC["spec_id"], SPEC["spec_version"], report["wave"],
        " (LAST WAVE)" if report["last_wave"] else "",
    ))
    lines.append("")

    lines.append("## Cells")
    lines.append("")
    header = ["arm", "seed", "status", "ME", "P1 ghost", "P2 return", "S1 curated",
              "H1 region", "H2 whole", "C1 control", "points"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for c in report["cells"]:
        if c["status"] != "complete":
            lines.append("| %s | %s | %s | - | - | - | - | - | - | - | - |"
                         % (c["arm"], c["seed"], c["status"]))
            continue
        e = c["endpoints"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            c["arm"], c["seed"], c["status"], "yes" if c["mechanism_exercised"] else "no",
            _fmt(e.get("P1")), _fmt(e.get("P2")), _fmt(e.get("S1")),
            _fmt(e.get("H1")), _fmt(e.get("H2")), _fmt(e.get("C1")),
            _fmt(c.get("final_points"), 0),
        ))
    lines.append("")

    for set_name in SPEC["ANALYSIS_SETS"]:
        block = report["analysis"].get(set_name)
        if block is None:
            continue
        lines.append("## Analysis set: %s (n per arm: U=%d, G=%d, GMIS=%d)" % (
            set_name, block["n_per_arm"]["U"], block["n_per_arm"]["G"],
            block["n_per_arm"]["GMIS"],
        ))
        lines.append("")
        lines.append("### G - U")
        lines.append("")
        lines.append("| endpoint | role | mean U | mean G | diff (G-U) | 97.5% CI | df | p | placebo q95 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for key in ["P1", "P2", "S1", "H1", "H2"]:
            res = block["g_vs_u"][key]
            pl = block["placebo"][key]["abs_quantile"]
            lines.append("| %s %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                key, SPEC["endpoints"][key]["label"], SPEC["endpoints"][key]["role"],
                _fmt(res["mean2"]), _fmt(res["mean1"]), _fmt(res["diff"]), _ci(res),
                _fmt(res["df"], 2), _fmt(res["p"], 5), _fmt(pl),
            ))
        lines.append("")

        ctl = block["positive_control"]["result"]
        lines.append("### Positive control (GMIS - U on C1): %s" % (
            "VALID" if block["positive_control"]["valid"] else "INVALID"))
        lines.append("")
        lines.append("diff %s, 97.5%% CI %s, required <= -%.2f dB and CI entirely below 0"
                     % (_fmt(ctl["diff"]), _ci(ctl), CONTROL_MIN_EFFECT_DB))
        lines.append("")

        lines.append("### Correctness contrast (G - GMIS)")
        lines.append("")
        lines.append("| endpoint | diff | 97.5% CI | separable |")
        lines.append("|---|---|---|---|")
        for key in SPEC["primary_endpoints"]:
            res = block["correctness_contrast"][key]
            lines.append("| %s | %s | %s | %s |" % (
                key, _fmt(res["diff"]), _ci(res),
                "yes" if ci_entirely_above_zero(res) else "no",
            ))
        lines.append("")

        lines.append("### Harm guard (U - G, fail if CI upper > %.2f dB): %s" % (
            HARM_MARGIN_DB, "PASS" if block["harm_guard"]["passes"] else "FAIL"))
        lines.append("")
        lines.append("| endpoint | diff (U-G) | 97.5% CI | passes |")
        lines.append("|---|---|---|---|")
        for key in SPEC["harm_endpoints"]:
            item = block["harm_guard"]["per_endpoint"][key]
            lines.append("| %s | %s | %s | %s |" % (
                key, _fmt(item["result"]["diff"]), _ci(item["result"]),
                "yes" if item["passes"] else "no",
            ))
        lines.append("")

        sz = block["sizing"]
        lines.append("### Sizing (internal pilot, DELTA=%.2f dB, cap %d)" % (DELTA, N2_CAP))
        lines.append("")
        lines.append("| endpoint | sd U | sd G | s_p | n2 per arm |")
        lines.append("|---|---|---|---|---|")
        for key in SPEC["primary_endpoints"]:
            item = sz["per_endpoint"][key]
            lines.append("| %s | %s | %s | %s | %s |" % (
                key, _fmt(item["sd_U"]), _fmt(item["sd_G"]), _fmt(item["s_p"]),
                _fmt(item["n2"], 0),
            ))
        lines.append("")
        lines.append("n2_max %s, current n per arm %d -> **%s**" % (
            _fmt(sz["n2_max"], 0), sz["current_n_per_arm"], sz["decision"]))
        lines.append("")

        lines.append("### Equivalence (TOST, +-%.2f dB at alpha %.3f)" % (
            TOST_MARGIN_DB, TOST_ALPHA))
        lines.append("")
        lines.append("| endpoint | p_lower | p_upper | passes |")
        lines.append("|---|---|---|---|")
        for key in SPEC["primary_endpoints"]:
            eq = block["equivalence"][key]
            lines.append("| %s | %s | %s | %s |" % (
                key, _fmt(eq["p_lower"], 5), _fmt(eq["p_upper"], 5),
                "yes" if eq["passes"] else "no",
            ))
        lines.append("")

        lines.append("### Verdicts")
        lines.append("")
        lines.append("| primary | verdict |")
        lines.append("|---|---|")
        for key in SPEC["primary_endpoints"]:
            lines.append("| %s %s | **%s** |" % (
                key, SPEC["endpoints"][key]["label"], block["verdicts"][key]["verdict"]))
        lines.append("")

    lines.append("## Reserved-unit check")
    lines.append("")
    ru = report["reserved_unit_check"]
    lines.append("reporting %s cells; consistent: %s; reserved/total: %s" % (
        ru["reported_as"], ru["consistent"], ru.get("reserved_fraction", "-")))
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    for key in SPEC["primary_endpoints"]:
        itt = report["analysis"]["itt"]["verdicts"][key]["verdict"]
        me_block = report["analysis"].get("mechanism_exercised")
        me = me_block["verdicts"][key]["verdict"] if me_block else "-"
        lines.append("- %s (%s): ITT **%s**; mechanism-exercised %s"
                     % (key, SPEC["endpoints"][key]["label"], itt, me))
    if report["blocking_errors"]:
        lines.append("")
        lines.append("## BLOCKING ERRORS")
        lines.append("")
        for msg in report["blocking_errors"]:
            lines.append("- %s" % msg)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run(manifest_path, last_wave=False, wave=1):
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    if "cells" not in manifest:
        raise ValueError(f"{manifest_path}: missing 'cells'")
    cells = [load_cell(entry) for entry in manifest["cells"]]

    blocking = []
    warnings = []
    seen = set()
    for c in cells:
        key = (c["arm"], c["seed"])
        if key in seen:
            warnings.append("duplicate (arm, seed) %s/%s" % key)
        seen.add(key)
        if c["status"] == "complete" and c["n_frames"] != SPEC["expected_n_frames"]:
            warnings.append(
                "%s/seed%s: profile has %s frames, expected %d"
                % (c["arm"], c["seed"], c["n_frames"], SPEC["expected_n_frames"])
            )

    complete = [c for c in cells if c["status"] == "complete"]
    failed = [c for c in cells if c["status"] == "failed"]

    ru = reserved_unit_check(complete)
    if not ru["consistent"]:
        blocking.append(ru.get("message", "reserved-unit check failed"))

    itt = analyse(complete, "itt", last_wave)
    me_cells = [c for c in complete if c["mechanism_exercised"]]
    me = analyse(me_cells, "mechanism_exercised", last_wave)

    report = {
        "spec": SPEC,
        "manifest": os.path.abspath(manifest_path),
        "wave": wave,
        "last_wave": bool(last_wave),
        "n_cells": len(cells),
        "n_complete": len(complete),
        "failed_cells": [
            {"arm": c["arm"], "seed": c["seed"], "run_dir": c["run_dir"]} for c in failed
        ],
        "cells": [
            {
                "arm": c["arm"],
                "seed": c["seed"],
                "run_dir": c["run_dir"],
                "status": c["status"],
                "n_frames": c["n_frames"],
                "endpoints": c["endpoints"],
                "final_points": c["final_points"],
                "precondition": c["precondition"],
                "mechanism_exercised": c["mechanism_exercised"],
                "mechanism_reason": c["mechanism_reason"],
            }
            for c in cells
        ],
        "analysis": {"itt": itt, "mechanism_exercised": me},
        "reserved_unit_check": ru,
        "warnings": warnings,
        "blocking_errors": blocking,
        "headline": {
            key: {
                "itt": itt["verdicts"][key]["verdict"],
                "mechanism_exercised": me["verdicts"][key]["verdict"],
            }
            for key in SPEC["primary_endpoints"]
        },
    }
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", help="json manifest of cells")
    ap.add_argument("--out", help="output json path")
    ap.add_argument("--wave", type=int, default=1, help="wave index (reported only)")
    ap.add_argument(
        "--last-wave",
        action="store_true",
        help="this wave is the last; enables the FEASIBILITY_STOP override",
    )
    ap.add_argument("--print-spec", action="store_true", help="dump the frozen SPEC and exit")
    args = ap.parse_args(argv)

    if args.print_spec:
        print(json.dumps(SPEC, indent=1, sort_keys=True))
        return 0
    if not args.manifest:
        ap.error("--manifest is required unless --print-spec is given")

    report = run(args.manifest, last_wave=args.last_wave, wave=args.wave)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
    print(markdown_report(report))
    if args.out:
        print("\nwrote %s" % os.path.abspath(args.out))
    for msg in report["warnings"]:
        print("warning: %s" % msg, file=sys.stderr)
    if report["blocking_errors"]:
        for msg in report["blocking_errors"]:
            print("BLOCKING: %s" % msg, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
