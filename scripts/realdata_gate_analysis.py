#!/usr/bin/env python
"""Pre-registered analysis for the real-data (N3V `cut_roasted_beef`) gate comparison.

Two treatment arms plus a positive-control arm:

    U      ungated baseline
    G      gated (the intervention under test)
    GMIS   gated with a deliberately mis-specified window (positive control)

Everything this script decides is fixed by the frozen ``SPEC`` dictionary below.
No threshold, window, level, margin, or verdict rule is read from the data or the
environment, so the sha256 of this file can be recorded before any gated cell
exists and the analysis is fully determined in advance.
``--print-spec`` dumps ``SPEC`` as JSON for that record.

Two later additions never touch that default. ``--spec <json>`` deep-merges a
spec instance onto a copy of ``SPEC`` for a DIFFERENT experiment: it may point an
endpoint at another event (including a per-frame-mask event named ``roi:<name>``
written by ``scripts/event_region_frame_profile.py --roi_dir``) and may express
its windows relative to the anchors ``A``, ``B``, ``CA``, ``CB`` (``"A+3"``,
``"B-2"``). A merged spec reports ``spec_version`` 1.1.0, carries the spec file's
sha256, and is written into the record in full; without ``--spec`` the output is
byte-identical to the frozen 1.0.0 analysis. A ``roi:`` event is pooled by
pixel-WEIGHTED MSE, because a per-frame mask does not contribute a constant pixel
count and the unweighted pooling the bounding-box endpoints use is invalid there.

``--paired`` switches to a within-prefix design: every manifest cell carries a
``prefix`` and the arms are compared pair by pair inside it. That analysis is
DESCRIPTIVE by construction -- per-pair differences, their median and range,
sign consistency, the mean paired difference with a paired-t interval, and the
within-prefix replicate floor measured by the ``GONES`` arm. It emits no
p-value-based verdict and no equivalence test; its verdict says only whether the
pre-declared claim conditions are met.

Inputs are the per-frame profiles written by ``scripts/event_region_frame_profile.py``
(``<run_dir>/f_box_profile.json``), an optional ``<run_dir>/precondition.json``
carrying the mechanism-exercised counters, and an optional ``<run_dir>/summary.json``
for the descriptive final point count.

The Student-t distribution is implemented here in pure Python (regularized
incomplete beta + bisection) so that the frozen numbers do not depend on whether
scipy is installed on the machine that runs the analysis.
"""

import argparse
import copy
import hashlib
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
# SPEC INSTANCES (--spec) and the paired design (--paired)
#
# Nothing below is read unless the corresponding flag is given; with neither
# flag the analysis is exactly the frozen 1.0.0 one above.
# --------------------------------------------------------------------------

SPEC_VERSION_WITH_OVERRIDE = "1.1.0"
ANCHOR_KEYS = ("CA", "CB", "A", "B")
ROI_EVENT_PREFIX = "roi:"
ROI_EVENT_KIND = "per_frame_mask"

PAIRED_ARMS = ["U", "G", "GEST", "GMIS", "GWRONGMEM", "GONES"]
PAIRED_CONTRASTS = [
    ("G", "U"),
    ("GEST", "U"),
    ("GMIS", "U"),
    ("GWRONGMEM", "U"),
    ("GONES", "U"),
    ("G", "GEST"),
    ("G", "GMIS"),
    ("G", "GWRONGMEM"),
]
PAIRED_SHAM_CONTRASTS = ["GMIS-U", "GWRONGMEM-U"]
PAIRED_FLOOR_ARM = "GONES"
PAIRED_MIN_EFFECT_DB = 0.5
PAIRED_MIN_PAIRS = 3
PAIRED_DESCRIPTIVE_BELOW_N = 6
PAIRED_CLAIM_ENDPOINT = "P1"
PAIRED_VERDICTS = ["CLAIM_CONDITIONS_MET", "NOT_MET", "DESIGN_WITHOUT_POWER"]
PAIRED_CLAIM_RULE = (
    "CLAIM_CONDITIONS_MET iff every G-U pair on the claim endpoint exceeds its "
    "within-prefix replicate floor (|U - GONES| for that prefix, or "
    "PAIRED_MIN_EFFECT_DB where no GONES cell exists) AND no GMIS-U or "
    "GWRONGMEM-U pair exceeds the same floor in the same direction; "
    "DESIGN_WITHOUT_POWER when fewer than PAIRED_MIN_PAIRS complete U/G pairs "
    "exist; NOT_MET otherwise. The verdict is descriptive: no p-value, no "
    "equivalence test, and the placebo split is a within-run diagnostic only."
)
PAIRED_SIZING_RULE = (
    "n2_pairs = ceil(K*(sd_paired/DELTA)^2 + Z_A^2/4) on the differences "
    "themselves (one sample, not two), at the spec's DELTA"
)
MASK_POOLING_RULE = (
    "an event carrying kind='per_frame_mask' is pooled by MSE weighted by its "
    "pixels_per_frame; frames with an empty mask carry zero weight and are "
    "dropped, so the unweighted constant-pixel-count pooling is never applied "
    "to a per-frame-mask endpoint"
)


def _spec(spec=None):
    """The active spec: the frozen SPEC unless a merged instance is supplied."""
    return SPEC if spec is None else spec


def deep_merge(base, override):
    """Recursively merge `override` onto a deep copy of `base`."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def normalise_endpoint(value):
    """Expand the endpoint shorthands a spec file may use.

    ``["B+3", "B+10"]``            -> a window on the default event
    ``"whole_frame"``              -> the whole-frame source
    ``{"event": "roi:core", ...}`` -> a window on the named event
    """
    if isinstance(value, dict):
        out = dict(value)
    elif isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(
                "an endpoint window shorthand must have exactly two entries, got %r"
                % (value,)
            )
        out = {"frames": list(value)}
    elif value == "whole_frame":
        out = {"source": "whole_frame", "frames": None, "frames_inclusive": None}
    else:
        raise ValueError("cannot interpret endpoint specification %r" % (value,))
    if "event" in out:
        out["event_name"] = out["event"]
        out["source"] = "event"
    if out.get("frames") is not None:
        out.setdefault("source", "event")
    return out


def _resolve_token(token, anchors):
    """One window edge: an int, an int literal, or ``<anchor>[+-]<offset>``.

    Returns None when the referenced anchor is still null in the spec file.
    """
    if isinstance(token, bool):
        raise ValueError("a window edge may not be a boolean")
    if isinstance(token, int):
        return int(token)
    if not isinstance(token, str):
        raise ValueError("cannot interpret window edge %r" % (token,))
    text = token.strip()
    for name in ANCHOR_KEYS:
        if text == name:
            base = anchors.get(name)
            return None if base is None else int(base)
        if text.startswith(name + "+") or text.startswith(name + "-"):
            base = anchors.get(name)
            return None if base is None else int(base) + int(text[len(name):])
    try:
        return int(text)
    except ValueError:
        raise ValueError(
            "window edge %r is neither an integer nor an offset from %s"
            % (token, ", ".join(ANCHOR_KEYS))
        )


def resolve_spec(spec):
    """Resolve every endpoint window expression against the spec's anchors."""
    anchors = {k: spec.get(k) for k in ANCHOR_KEYS}
    endpoints = {}
    unresolved = []
    for key, entry in spec["endpoints"].items():
        entry = dict(entry)
        expr = entry.get("frames")
        if expr is not None:
            if not isinstance(expr, (list, tuple)) or len(expr) != 2:
                raise ValueError(
                    "endpoint %s: 'frames' must be a two-entry window, got %r"
                    % (key, expr)
                )
            lo = _resolve_token(expr[0], anchors)
            hi = _resolve_token(expr[1], anchors)
            if lo is None or hi is None:
                unresolved.append(key)
                entry["frames_inclusive"] = None
            else:
                if hi < lo:
                    raise ValueError(
                        "endpoint %s: window [%d, %d] resolves to an empty range"
                        % (key, lo, hi)
                    )
                entry["frames_inclusive"] = [lo, hi]
        endpoints[key] = entry
    out = dict(spec)
    out["endpoints"] = endpoints
    out["anchors"] = anchors
    out["unresolved_endpoints"] = sorted(unresolved)
    return out


def load_spec(path):
    """Deep-merge a spec instance onto SPEC; return (merged spec, sha256)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    digest = hashlib.sha256(raw).hexdigest()
    override = json.loads(raw.decode("utf-8"))
    if not isinstance(override, dict):
        raise ValueError("%s: expected a JSON object at the top level" % path)
    override = dict(override)
    if "endpoints" in override:
        if not isinstance(override["endpoints"], dict):
            raise ValueError("%s: 'endpoints' must be a JSON object" % path)
        override["endpoints"] = {
            key: normalise_endpoint(value)
            for key, value in override["endpoints"].items()
        }
    merged = deep_merge(SPEC, override)
    if "spec_version" not in override:
        merged["spec_version"] = SPEC_VERSION_WITH_OVERRIDE
    merged = resolve_spec(merged)
    merged["spec_file"] = os.path.abspath(path)
    merged["spec_sha256"] = digest
    merged["MASK_POOLING_RULE"] = merged.get("MASK_POOLING_RULE", MASK_POOLING_RULE)
    return merged, digest


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


def pooled_psnr_weighted(psnr_values, weights):
    """Pool per-frame PSNRs weighting each frame by its own pixel count.

    Required for a per-frame-mask event, where the frames do NOT contribute the
    same pixel count. Frames with no weight (an empty mask) are dropped.
    """
    vals, wts = [], []
    for value, weight in zip(psnr_values, weights):
        if value is None or weight is None or float(weight) <= 0.0:
            continue
        vals.append(float(value))
        wts.append(float(weight))
    if not vals:
        raise ValueError("cannot pool a window in which every mask is empty")
    arr = np.asarray(vals, dtype=np.float64)
    wt = np.asarray(wts, dtype=np.float64)
    mse = np.power(10.0, -arr / 10.0)
    mean_mse = float((mse * wt).sum() / wt.sum())
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


def read_final_points(run_dir, spec=None):
    S = _spec(spec)
    path = os.path.join(run_dir, S["summary_filename"])
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
        for key in S["final_points_keys"]:
            value = _lookup_path(root, key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return int(value)
    return None


def read_precondition(run_dir, spec=None):
    S = _spec(spec)
    path = os.path.join(run_dir, S["precondition_filename"])
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    out = {k: data.get(k) for k in S["PRECONDITION_FIELDS"] if k in data}
    detail = data.get("detail") if isinstance(data.get("detail"), dict) else {}
    presence = (
        detail.get("presence") if isinstance(detail.get("presence"), dict) else {}
    )
    fbox = detail.get("fbox") if isinstance(detail.get("fbox"), dict) else {}
    if isinstance(presence.get("frames_presence_zero"), list):
        out["frames_presence_zero_list"] = [
            int(f) for f in presence["frames_presence_zero"]
        ]
    if fbox:
        out["fbox_frame"] = fbox.get("frame")
        out["fbox_box"] = fbox.get("box_x0_y0_x1_y1_inclusive")
    if data.get("arm_kind") is not None:
        out["arm_kind"] = data.get("arm_kind")
    return out


def mechanism_exercised(arm, precondition, spec=None):
    """(passes, reason) for the mechanism-exercised analysis set."""
    S = _spec(spec)
    if arm == "U":
        return True, "U arm passes by construction"
    if precondition is None:
        return False, f"no {S['precondition_filename']}"
    min_surviving = S["MIN_GATED_SURVIVING"]
    min_fbox = S["MIN_GATED_FBOX"]
    min_zeros = S["MIN_FRAMES_PRESENCE_ZERO"]
    failures = []
    surviving = precondition.get("gated_rows_final")
    fbox = precondition.get("gated_rows_fbox_frame150")
    zeros = precondition.get("frames_presence_zero")
    if surviving is None or surviving < min_surviving:
        failures.append(f"gated_rows_final={surviving} < {min_surviving}")
    if fbox is None or fbox < min_fbox:
        failures.append(f"gated_rows_fbox_frame150={fbox} < {min_fbox}")
    if zeros is None or zeros < min_zeros:
        failures.append(f"frames_presence_zero={zeros} < {min_zeros}")
    window = S.get("MECHANISM_ZERO_FRAMES_WINDOW")
    if window and arm not in S.get("MECHANISM_ZERO_FRAMES_EXEMPT_ARMS", []):
        zl = precondition.get("frames_presence_zero_list")
        if zl is None:
            failures.append("precondition carries no zero-presence frame list")
        else:
            inside = [f for f in zl if int(window[0]) <= int(f) <= int(window[1])]
            if len(inside) < min_zeros:
                failures.append(
                    "zero-presence frames inside %r: %d < %d"
                    % (list(window), len(inside), min_zeros)
                )
    want_frame = S.get("MECHANISM_FBOX_FRAME")
    if want_frame is not None:
        got = precondition.get("fbox_frame")
        if got is None or int(got) != int(want_frame):
            failures.append(
                "fbox measured at frame %r, spec requires %r" % (got, want_frame)
            )
    want_box = S.get("MECHANISM_FBOX")
    if want_box is not None:
        got = precondition.get("fbox_box")
        if got is None or [int(v) for v in got] != [int(v) for v in want_box]:
            failures.append("fbox %r differs from the spec box %r" % (got, list(want_box)))
    if failures:
        return False, "; ".join(failures)
    if arm in ("GWRONGMEM", "GONES"):
        return True, "gate code path exercised (not the target membership/window)"
    return True, "precondition satisfied"


def load_cell(entry, spec=None, arms=None, require_prefix=False):
    """Read one manifest entry into a cell record with all endpoint values."""
    S = _spec(spec)
    for field in ("arm", "seed", "run_dir", "status"):
        if field not in entry:
            raise ValueError(f"manifest cell is missing required field '{field}': {entry}")
    arm = entry["arm"]
    allowed = S["arms"] if arms is None else arms
    if arm not in allowed:
        raise ValueError(f"unknown arm {arm!r}; expected one of {allowed}")
    if require_prefix and entry.get("prefix") is None:
        raise ValueError(
            f"paired mode: cell {arm}/seed {entry['seed']} carries no 'prefix'"
        )
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
    if entry.get("prefix") is not None:
        cell["prefix"] = entry["prefix"]
    if status == "failed":
        return cell

    profile_path = os.path.join(run_dir, S["profile_filename"])
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
    wanted = sorted(
        {
            ep.get("event_name", S["event_name"])
            for ep in S["endpoints"].values()
            if ep.get("source") == "event"
        }
    )
    loaded = {}
    for event_name in wanted:
        if event_name not in profile["events"]:
            raise ValueError(
                f"{profile_path}: event {event_name!r} not found; "
                f"available: {sorted(profile['events'])}"
            )
        event = profile["events"][event_name]
        if "per_frame_psnr" not in event:
            raise ValueError(f"{profile_path}: event {event_name!r} has no 'per_frame_psnr'")
        per_frame = [None if v is None else float(v) for v in event["per_frame_psnr"]]
        if len(per_frame) != len(frames):
            raise ValueError(
                f"{profile_path}: event per_frame_psnr ({len(per_frame)}) does not align "
                f"with frames ({len(frames)})"
            )
        loaded[event_name] = (event, per_frame)
    cell["n_frames"] = len(frames)
    default_event = loaded.get(S["event_name"])
    cell["bbox"] = default_event[0].get("bbox") if default_event else None

    for key, ep in S["endpoints"].items():
        if ep["source"] == "event":
            if ep.get("frames_inclusive") is None:
                raise ValueError(
                    f"endpoint {key} has no resolved frame window; set the spec anchors"
                )
            lo, hi = ep["frames_inclusive"]
            what = f"cell {arm}/seed {entry['seed']} endpoint {key}"
            event, per_frame = loaded[ep.get("event_name", S["event_name"])]
            if event.get("kind") == ROI_EVENT_KIND:
                weights = event.get("pixels_per_frame")
                if weights is None or len(weights) != len(frames):
                    raise ValueError(
                        f"{profile_path}: per-frame-mask event "
                        f"{ep.get('event_name')!r} has no aligned 'pixels_per_frame'"
                    )
                window_psnr = select_window(frames, per_frame, lo, hi, what)
                window_w = select_window(frames, weights, lo, hi, what)
                if S.get("PER_FRAME_MASK_REQUIRE_ALL_FRAMES", False):
                    empty = [
                        f for f, w in zip(range(lo, hi + 1), window_w)
                        if w is None or float(w) <= 0.0
                    ]
                    if empty:
                        raise ValueError(
                            f"{what}: per-frame-mask event {ep.get('event_name')!r} "
                            f"has an empty or missing mask on frames {empty}; "
                            "PER_FRAME_MASK_REQUIRE_ALL_FRAMES forbids pooling a subset"
                        )
                cell.setdefault("per_frame", {})[key] = {
                    "frames": list(range(lo, hi + 1)),
                    "psnr": window_psnr,
                    "weights": window_w,
                }
                cell["endpoints"][key] = pooled_psnr_weighted(window_psnr, window_w)
            else:
                cell["endpoints"][key] = pooled_psnr(
                    select_window(frames, per_frame, lo, hi, what)
                )
        elif ep["source"] == "whole_frame":
            cell["endpoints"][key] = pooled_psnr(whole)

    cell["final_points"] = read_final_points(run_dir, spec=S)
    cell["endpoints"]["CAP"] = (
        float(cell["final_points"]) if cell["final_points"] is not None else None
    )
    cell["precondition"] = read_precondition(run_dir, spec=S)
    passes, reason = mechanism_exercised(arm, cell["precondition"], spec=S)
    cell["mechanism_exercised"] = passes
    cell["mechanism_reason"] = reason
    return cell


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def welch(a, b, conf=None, spec=None):
    """Welch two-sample t on (mean(a) - mean(b)), two-sided, at confidence `conf`."""
    conf = _spec(spec)["CI_CONF"] if conf is None else conf
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


def placebo_distribution(values, spec=None):
    """Half-mean differences over all balanced splits, and the frozen quantile."""
    quantile = _spec(spec)["PLACEBO_QUANTILE"]
    vals = [float(v) for v in values]
    splits = balanced_splits(len(vals))
    if not splits:
        return {
            "n_cells": len(vals),
            "n_splits": 0,
            "quantile": quantile,
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
        "quantile": quantile,
        "abs_quantile": float(np.percentile(abs_diffs, 100.0 * quantile)),
        "max_abs": float(max(abs_diffs)),
        "diffs": diffs,
    }


def sizing_n2(sd_u, sd_g, spec=None):
    """Wittes-Brittain internal-pilot re-estimated per-arm sample size."""
    if sd_u is None or sd_g is None:
        return None, None
    sizing = _spec(spec)["SIZING"]
    s_p = math.sqrt((sd_u ** 2 + sd_g ** 2) / 2.0)
    n2 = math.ceil(
        2.0 * sizing["K"] * (s_p / sizing["DELTA"]) ** 2 + sizing["Z_A"] ** 2 / 4.0
    )
    return s_p, int(n2)


def paired_n2(sd_paired, spec=None):
    """Per-PAIR sample size from the sd of the paired differences themselves."""
    if sd_paired is None:
        return None
    sizing = _spec(spec)["SIZING"]
    return int(
        math.ceil(
            sizing["K"] * (sd_paired / sizing["DELTA"]) ** 2 + sizing["Z_A"] ** 2 / 4.0
        )
    )


def tost(res, margin=None, alpha=None, spec=None):
    """Two one-sided tests for equivalence within +-margin."""
    S = _spec(spec)
    margin = S["TOST"]["margin_db"] if margin is None else margin
    alpha = S["TOST"]["alpha"] if alpha is None else alpha
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


def analyse(cells, set_name, last_wave, spec=None):
    """Run the whole frozen analysis over one set of complete cells."""
    S = _spec(spec)
    control_min_effect_db = S["CONTROL_MIN_EFFECT_DB"]
    harm_margin_db = S["HARM_MARGIN_DB"]
    n2_cap = S["SIZING"]["N2_CAP"]
    arms = {a: [c for c in cells if c["arm"] == a] for a in S["arms"]}
    n_per_arm = {a: len(v) for a, v in arms.items()}

    # 1. G vs U on P1, P2, S1, H1, H2
    g_vs_u = {}
    for key in ["P1", "P2", "S1", "H1", "H2"]:
        g_vs_u[key] = welch(
            _arm_values(cells, "G", key), _arm_values(cells, "U", key), spec=S
        )

    # 2. Positive control: GMIS vs U on C1
    control = welch(
        _arm_values(cells, "GMIS", "C1"), _arm_values(cells, "U", "C1"), spec=S
    )
    control_valid = bool(
        ci_entirely_below_zero(control)
        and control["diff"] is not None
        and control["diff"] <= -control_min_effect_db
    )
    control_block = {
        "endpoint": "C1",
        "contrast": "GMIS - U",
        "result": control,
        "min_effect_db": control_min_effect_db,
        "valid": control_valid,
        "reason": (
            "valid"
            if control_valid
            else "CI not entirely below 0 and/or point estimate above -%.2f dB"
            % control_min_effect_db
        ),
    }

    # 3. Correctness contrast: G vs GMIS on the primaries
    correctness = {}
    for key in S["primary_endpoints"]:
        correctness[key] = welch(
            _arm_values(cells, "G", key), _arm_values(cells, "GMIS", key), spec=S
        )

    # 4. Placebo distribution over balanced splits of the U cells
    placebo = {
        key: placebo_distribution(_arm_values(cells, "U", key), spec=S)
        for key in S["psnr_endpoints"]
    }

    # 5. Harm guard on (U - G)
    harm = {}
    harm_pass = True
    for key in S["harm_endpoints"]:
        res = welch(
            _arm_values(cells, "U", key), _arm_values(cells, "G", key), spec=S
        )
        if res["insufficient_n"]:
            ok = S["HARM_GUARD_ON_INSUFFICIENT_N"] != "fail"
            reason = "insufficient n -> fail-closed"
        else:
            ok = res["ci_high"] <= harm_margin_db
            reason = "ci_high %.4f vs margin %.2f" % (res["ci_high"], harm_margin_db)
        harm[key] = {"contrast": "U - G", "result": res, "passes": bool(ok), "reason": reason}
        harm_pass = harm_pass and bool(ok)

    # 6. Sizing
    current_n = min(n_per_arm["U"], n_per_arm["G"]) if n_per_arm["G"] else 0
    sizing = {"current_n_per_arm": current_n, "per_endpoint": {}, "n2_max": None}
    n2_values = []
    for key in S["primary_endpoints"]:
        res = g_vs_u[key]
        s_p, n2 = sizing_n2(res["sd2"], res["sd1"], spec=S)  # sd2 = U arm, sd1 = G arm
        sizing["per_endpoint"][key] = {"s_p": s_p, "n2": n2, "sd_U": res["sd2"], "sd_G": res["sd1"]}
        if n2 is not None:
            n2_values.append(n2)
    if n2_values:
        sizing["n2_max"] = max(n2_values)
        if sizing["n2_max"] <= current_n:
            sizing["decision"] = "wave1_final"
        elif sizing["n2_max"] <= n2_cap:
            sizing["decision"] = "run_wave2_to_n2"
        else:
            sizing["decision"] = "feasibility_stop"
    else:
        sizing["decision"] = "not_estimable"

    # 7. Equivalence
    equivalence = {key: tost(g_vs_u[key], spec=S) for key in S["primary_endpoints"]}

    # 8. Verdicts
    verdicts = {}
    for key in S["primary_endpoints"]:
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
        "harm_guard": {"per_endpoint": harm, "passes": harm_pass, "margin_db": harm_margin_db},
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
            for a in S["arms"]
        },
    }


# --------------------------------------------------------------------------
# Paired (within-prefix) analysis -- DESCRIPTIVE, no p-value verdict
# --------------------------------------------------------------------------


def paired_summary(diffs, conf=None, spec=None):
    """Descriptive summary of one set of paired differences, plus a paired-t CI."""
    S = _spec(spec)
    conf = S["CI_CONF"] if conf is None else conf
    vals = [float(d) for d in diffs]
    n = len(vals)
    out = {
        "n_pairs": n,
        "mean": float(np.mean(vals)) if n else None,
        "median": float(np.median(vals)) if n else None,
        "min": float(min(vals)) if n else None,
        "max": float(max(vals)) if n else None,
        "sd": float(np.std(vals, ddof=1)) if n >= 2 else None,
        "se": None,
        "df": None,
        "ci_low": None,
        "ci_high": None,
        "conf": conf,
        "n_positive": sum(1 for d in vals if d > 0.0),
        "n_negative": sum(1 for d in vals if d < 0.0),
        "n_zero": sum(1 for d in vals if d == 0.0),
        "insufficient_n": n < 2,
        "descriptive": n < PAIRED_DESCRIPTIVE_BELOW_N,
        "descriptive_reason": (
            "fewer than %d pairs: the interval is reported, not tested"
            % PAIRED_DESCRIPTIVE_BELOW_N
            if n < PAIRED_DESCRIPTIVE_BELOW_N
            else "paired mode reports intervals descriptively"
        ),
    }
    out["n_same_sign"] = max(out["n_positive"], out["n_negative"])
    out["sign_consistent"] = bool(n >= 1 and out["n_same_sign"] == n)
    if n >= 2:
        se = out["sd"] / math.sqrt(n)
        out["se"] = se
        out["df"] = float(n - 1)
        if se == 0.0:
            out["ci_low"] = out["mean"]
            out["ci_high"] = out["mean"]
            out["zero_variance"] = True
        else:
            tcrit = t_ppf(1.0 - (1.0 - conf) / 2.0, n - 1)
            out["t_crit"] = tcrit
            out["ci_low"] = out["mean"] - tcrit * se
            out["ci_high"] = out["mean"] + tcrit * se
            out["zero_variance"] = False
    return out


def _prefix_index(cells, arm):
    """{prefix: cell} for one arm, plus the prefixes that appear more than once."""
    index = {}
    duplicates = []
    for cell in cells:
        if cell["arm"] != arm or cell.get("prefix") is None:
            continue
        if cell["prefix"] in index:
            duplicates.append(cell["prefix"])
            continue
        index[cell["prefix"]] = cell
    return index, sorted(set(duplicates))


def _pairs(cells, arm_a, arm_b, endpoint):
    """Per-prefix (arm_a - arm_b) differences on one endpoint."""
    idx_a, _ = _prefix_index(cells, arm_a)
    idx_b, _ = _prefix_index(cells, arm_b)
    out = []
    for prefix in sorted(set(idx_a) & set(idx_b)):
        va = idx_a[prefix]["endpoints"].get(endpoint)
        vb = idx_b[prefix]["endpoints"].get(endpoint)
        if va is None or vb is None:
            continue
        out.append(
            {
                "prefix": prefix,
                "a": float(va),
                "b": float(vb),
                "diff": float(va) - float(vb),
            }
        )
    return out


def replicate_floor(cells, endpoint, spec=None, arm=None):
    """Within-prefix replicate floor |U - <arm>|, per prefix.

    `arm` defaults to PAIRED_FLOOR_ARM (GONES). A spec may instead declare
    `PAIRED_FLOOR_SOURCE: "fixed"`, in which case the claim floor is the
    constant PAIRED_MIN_EFFECT_DB for every prefix and the arm-derived
    spreads are reported descriptively only (see `descriptive_floors`).
    """
    S = _spec(spec)
    fallback = S.get("PAIRED_MIN_EFFECT_DB", PAIRED_MIN_EFFECT_DB)
    source_kind = S.get("PAIRED_FLOOR_SOURCE", PAIRED_FLOOR_ARM)
    arm = arm or (PAIRED_FLOOR_ARM if source_kind == "fixed" else source_kind)
    pairs = _pairs(cells, "U", arm, endpoint)
    per_prefix = {p["prefix"]: abs(p["diff"]) for p in pairs}
    values = sorted(per_prefix.values())
    block = {
        "contrast": "|U - %s|" % arm,
        "arm": arm,
        "per_prefix": per_prefix,
        "n": len(values),
        "median": float(np.median(values)) if values else None,
        "max": float(max(values)) if values else None,
        "fallback_db": fallback,
        "source": arm if values else "PAIRED_MIN_EFFECT_DB",
    }
    if source_kind == "fixed":
        block["claim_floor"] = "fixed"
        block["source"] = "PAIRED_MIN_EFFECT_DB"
    return block


def _floor_for(floor_block, prefix):
    if floor_block.get("claim_floor") == "fixed":
        return float(floor_block["fallback_db"]), "PAIRED_MIN_EFFECT_DB"
    value = floor_block["per_prefix"].get(prefix)
    if value is None:
        return float(floor_block["fallback_db"]), "PAIRED_MIN_EFFECT_DB"
    return float(value), floor_block.get("arm", PAIRED_FLOOR_ARM)


def _mse_from_psnr(value):
    if value is None:
        return None
    if math.isinf(value):
        return 0.0
    return float(10.0 ** (-float(value) / 10.0))


def per_frame_mask_diagnostics(cells, endpoint, spec=None):
    """Descriptive per-frame readings for a per-frame-mask claim endpoint.

    For every G-U, GEST-U and G-GEST pair on `endpoint` (when the cells carry
    the per-frame series, i.e. the endpoint reads a `per_frame_mask` event):
    the fraction of window frames on which the first arm's masked MSE is lower,
    the count of frames with an exactly-zero masked error (infinite PSNR) per
    arm, the per-arm maximum per-frame PSNR, and the pooled MSE difference.
    Pure reporting; no verdict reads it.
    """
    S = _spec(spec)
    ep = S["endpoints"].get(endpoint)
    if not ep or ep.get("source") != "event" or ep.get("frames_inclusive") is None:
        return None
    out = {"endpoint": endpoint, "contrasts": {}}
    for arm_a, arm_b in (("G", "U"), ("GEST", "U"), ("G", "GEST")):
        idx_a, _ = _prefix_index(cells, arm_a)
        idx_b, _ = _prefix_index(cells, arm_b)
        pairs = []
        for prefix in sorted(set(idx_a) & set(idx_b)):
            sa = (idx_a[prefix].get("per_frame") or {}).get(endpoint)
            sb = (idx_b[prefix].get("per_frame") or {}).get(endpoint)
            if not sa or not sb:
                continue
            wa = sa["weights"]
            ma = [_mse_from_psnr(v) for v in sa["psnr"]]
            mb = [_mse_from_psnr(v) for v in sb["psnr"]]
            valid = [
                i for i, w in enumerate(wa)
                if w and ma[i] is not None and mb[i] is not None
            ]
            if not valid:
                continue
            lower = sum(1 for i in valid if ma[i] < mb[i])
            wsum = sum(wa[i] for i in valid)
            pooled_a = sum(ma[i] * wa[i] for i in valid) / wsum
            pooled_b = sum(mb[i] * wa[i] for i in valid) / wsum
            pairs.append({
                "prefix": prefix,
                "n_frames": len(valid),
                "frames_a_lower_mse": lower,
                "fraction_a_lower_mse": lower / float(len(valid)),
                "n_inf_a": sum(1 for i in valid if math.isinf(sa["psnr"][i])),
                "n_inf_b": sum(1 for i in valid if math.isinf(sb["psnr"][i])),
                "max_psnr_a": max(sa["psnr"][i] for i in valid),
                "max_psnr_b": max(sb["psnr"][i] for i in valid),
                "pooled_mse_a": pooled_a,
                "pooled_mse_b": pooled_b,
                "pooled_mse_diff_a_minus_b": pooled_a - pooled_b,
            })
        if pairs:
            out["contrasts"]["%s-%s" % (arm_a, arm_b)] = pairs
    return out


def analyse_paired(cells, set_name, spec=None):
    """Within-prefix comparison of every arm pair. Descriptive by construction."""
    S = _spec(spec)
    min_pairs = S.get("PAIRED_MIN_PAIRS", PAIRED_MIN_PAIRS)
    claim_endpoint = S.get("PAIRED_CLAIM_ENDPOINT", PAIRED_CLAIM_ENDPOINT)
    claim_rule = S.get("PAIRED_CLAIM_RULE", PAIRED_CLAIM_RULE)
    arms = [a for a in PAIRED_ARMS if any(c["arm"] == a for c in cells)]
    endpoints = [k for k in S["psnr_endpoints"] if k in S["endpoints"]]
    n_per_arm = {a: sum(1 for c in cells if c["arm"] == a) for a in arms}
    prefixes = sorted({c["prefix"] for c in cells if c.get("prefix") is not None})

    duplicates = {}
    for arm in arms:
        _, dupes = _prefix_index(cells, arm)
        if dupes:
            duplicates[arm] = dupes

    contrasts = {}
    for arm_a, arm_b in PAIRED_CONTRASTS:
        if arm_a not in arms or arm_b not in arms:
            continue
        name = "%s-%s" % (arm_a, arm_b)
        block = {}
        for key in endpoints:
            pairs = _pairs(cells, arm_a, arm_b, key)
            entry = paired_summary([p["diff"] for p in pairs], spec=S)
            entry["pairs"] = pairs
            entry["prefixes"] = [p["prefix"] for p in pairs]
            block[key] = entry
        contrasts[name] = block

    floors = {key: replicate_floor(cells, key, spec=S) for key in endpoints}
    descriptive_floors = {
        arm: {key: replicate_floor(cells, key, spec=S, arm=arm) for key in endpoints}
        for arm in ("GONES", "GMIS")
        if arm in arms
    }
    claim_only = bool(S.get("PAIRED_CLAIM_ENDPOINT_ONLY", False))
    require_shams = bool(S.get("PAIRED_REQUIRE_SHAMS", False))
    require_gest = bool(S.get("PAIRED_REQUIRE_GEST", False))

    # Placebo: within-run diagnostic only, never a verdict input in paired mode.
    placebo = {
        key: placebo_distribution(_arm_values(cells, "U", key), spec=S)
        for key in endpoints
    }
    placebo_role = "within-run diagnostic only; not an input to the paired verdict"

    complete_pairs = sorted(
        set(_prefix_index(cells, "U")[0]) & set(_prefix_index(cells, "G")[0])
    )
    underpowered = len(complete_pairs) < min_pairs

    sizing = {
        "per_endpoint": {},
        "n_pairs_current": len(complete_pairs),
        "status": S.get("PAIRED_SIZING_STATUS", "operative"),
    }
    sizing_keys = (
        [claim_endpoint] if claim_only and claim_endpoint in S["primary_endpoints"]
        else S["primary_endpoints"]
    )
    for key in sizing_keys:
        entry = contrasts.get("G-U", {}).get(key)
        sd = entry["sd"] if entry else None
        sizing["per_endpoint"][key] = {
            "sd_paired": sd,
            "n2_pairs": paired_n2(sd, spec=S),
            "delta_db": S["SIZING"]["DELTA"],
        }
    n2_values = [
        v["n2_pairs"] for v in sizing["per_endpoint"].values() if v["n2_pairs"] is not None
    ]
    sizing["n2_pairs_max"] = max(n2_values) if n2_values else None
    if sizing["n2_pairs_max"] is None:
        sizing["decision"] = "not_estimable"
    elif sizing["n2_pairs_max"] <= len(complete_pairs):
        sizing["decision"] = "wave1_final"
    elif sizing["n2_pairs_max"] <= S["SIZING"]["N2_CAP"]:
        sizing["decision"] = "run_wave2_to_n2"
    else:
        sizing["decision"] = "feasibility_stop"

    verdicts = {}
    for key in S["primary_endpoints"]:
        if key not in endpoints:
            continue
        if claim_only and key != claim_endpoint:
            verdicts[key] = {
                "verdict": "DESCRIPTIVE_ONLY",
                "endpoint": key,
                "reason": "PAIRED_CLAIM_ENDPOINT_ONLY: only %s carries a verdict"
                % claim_endpoint,
                "n_complete_pairs": len(complete_pairs),
            }
            continue
        floor_block = floors[key]
        gu = contrasts.get("G-U", {}).get(key, {}).get("pairs", [])
        per_pair = []
        for pair in gu:
            floor, source = _floor_for(floor_block, pair["prefix"])
            per_pair.append(
                {
                    "prefix": pair["prefix"],
                    "diff": pair["diff"],
                    "floor": floor,
                    "floor_source": source,
                    "exceeds_floor": bool(pair["diff"] > floor),
                }
            )
        all_exceed = bool(per_pair) and all(p["exceeds_floor"] for p in per_pair)
        sham = {}
        sham_clean = True
        for name in PAIRED_SHAM_CONTRASTS:
            entries = contrasts.get(name, {}).get(key, {}).get("pairs", [])
            offenders = []
            for pair in entries:
                floor, source = _floor_for(floor_block, pair["prefix"])
                if pair["diff"] > floor:
                    offenders.append(
                        {"prefix": pair["prefix"], "diff": pair["diff"], "floor": floor}
                    )
            sham[name] = {
                "n_pairs": len(entries),
                "n_exceeding": len(offenders),
                "offenders": offenders,
            }
            sham_clean = sham_clean and not offenders
        gu_prefixes = [p["prefix"] for p in gu]
        missing_controls = {}
        if require_shams:
            for name in PAIRED_SHAM_CONTRASTS:
                have = {
                    p["prefix"]
                    for p in contrasts.get(name, {}).get(key, {}).get("pairs", [])
                }
                lacking = [p for p in gu_prefixes if p not in have]
                if lacking:
                    missing_controls[name] = lacking
        if require_gest:
            have = {
                p["prefix"]
                for p in contrasts.get("GEST-U", {}).get(key, {}).get("pairs", [])
            }
            lacking = [p for p in gu_prefixes if p not in have]
            if lacking:
                missing_controls["GEST-U"] = lacking
        if underpowered:
            verdict = "DESIGN_WITHOUT_POWER"
        elif missing_controls:
            verdict = "DESIGN_WITHOUT_POWER"
        elif all_exceed and sham_clean:
            verdict = "CLAIM_CONDITIONS_MET"
        else:
            verdict = "NOT_MET"
        verdicts[key] = {
            "verdict": verdict,
            "endpoint": key,
            "per_pair": per_pair,
            "every_pair_exceeds_floor": all_exceed,
            "sham_contrasts_clean": bool(sham_clean),
            "sham": sham,
            "missing_control_pairs": missing_controls,
            "floor_source": floor_block["source"],
            "n_complete_pairs": len(complete_pairs),
            "rule": claim_rule,
        }

    return {
        "set": set_name,
        "kind": "paired",
        "arms_present": n_per_arm,
        "prefixes": prefixes,
        "duplicate_prefixes": duplicates,
        "complete_pairs": complete_pairs,
        "n_complete_pairs": len(complete_pairs),
        "contrasts": contrasts,
        "replicate_floor": floors,
        "descriptive_floors": descriptive_floors,
        "per_frame_mask_diagnostics": per_frame_mask_diagnostics(
            cells, claim_endpoint, spec=S
        ),
        "placebo": placebo,
        "placebo_role": placebo_role,
        "sizing": sizing,
        "verdicts": verdicts,
        "claim": {
            "endpoint": claim_endpoint,
            "verdict": (
                verdicts[claim_endpoint]["verdict"]
                if claim_endpoint in verdicts
                else "DESIGN_WITHOUT_POWER"
            ),
            "min_effect_db": S.get("PAIRED_MIN_EFFECT_DB", PAIRED_MIN_EFFECT_DB),
            "min_pairs": min_pairs,
            "rule": claim_rule,
        },
        "capacity_descriptive": {
            a: {
                "n": len(_arm_values(cells, a, "CAP")),
                "mean_final_points": (
                    float(np.mean(_arm_values(cells, a, "CAP")))
                    if _arm_values(cells, a, "CAP")
                    else None
                ),
            }
            for a in arms
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
    SPEC_R = report.get("spec", SPEC)
    control_min_effect_db = SPEC_R["CONTROL_MIN_EFFECT_DB"]
    harm_margin_db = SPEC_R["HARM_MARGIN_DB"]
    delta_db = SPEC_R["SIZING"]["DELTA"]
    n2_cap = SPEC_R["SIZING"]["N2_CAP"]
    tost_margin_db = SPEC_R["TOST"]["margin_db"]
    tost_alpha = SPEC_R["TOST"]["alpha"]
    if report.get("paired"):
        return markdown_report_paired(report)
    lines = []
    lines.append("# Real-data gate analysis (%s, %s)" % (SPEC_R["scene"], SPEC_R["event_name"]))
    lines.append("")
    lines.append("spec %s v%s | wave %s%s" % (
        SPEC_R["spec_id"], SPEC_R["spec_version"], report["wave"],
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

    for set_name in SPEC_R["ANALYSIS_SETS"]:
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
                key, SPEC_R["endpoints"][key]["label"], SPEC_R["endpoints"][key]["role"],
                _fmt(res["mean2"]), _fmt(res["mean1"]), _fmt(res["diff"]), _ci(res),
                _fmt(res["df"], 2), _fmt(res["p"], 5), _fmt(pl),
            ))
        lines.append("")

        ctl = block["positive_control"]["result"]
        lines.append("### Positive control (GMIS - U on C1): %s" % (
            "VALID" if block["positive_control"]["valid"] else "INVALID"))
        lines.append("")
        lines.append("diff %s, 97.5%% CI %s, required <= -%.2f dB and CI entirely below 0"
                     % (_fmt(ctl["diff"]), _ci(ctl), control_min_effect_db))
        lines.append("")

        lines.append("### Correctness contrast (G - GMIS)")
        lines.append("")
        lines.append("| endpoint | diff | 97.5% CI | separable |")
        lines.append("|---|---|---|---|")
        for key in SPEC_R["primary_endpoints"]:
            res = block["correctness_contrast"][key]
            lines.append("| %s | %s | %s | %s |" % (
                key, _fmt(res["diff"]), _ci(res),
                "yes" if ci_entirely_above_zero(res) else "no",
            ))
        lines.append("")

        lines.append("### Harm guard (U - G, fail if CI upper > %.2f dB): %s" % (
            harm_margin_db, "PASS" if block["harm_guard"]["passes"] else "FAIL"))
        lines.append("")
        lines.append("| endpoint | diff (U-G) | 97.5% CI | passes |")
        lines.append("|---|---|---|---|")
        for key in SPEC_R["harm_endpoints"]:
            item = block["harm_guard"]["per_endpoint"][key]
            lines.append("| %s | %s | %s | %s |" % (
                key, _fmt(item["result"]["diff"]), _ci(item["result"]),
                "yes" if item["passes"] else "no",
            ))
        lines.append("")

        sz = block["sizing"]
        lines.append("### Sizing (internal pilot, DELTA=%.2f dB, cap %d)" % (delta_db, n2_cap))
        lines.append("")
        lines.append("| endpoint | sd U | sd G | s_p | n2 per arm |")
        lines.append("|---|---|---|---|---|")
        for key in SPEC_R["primary_endpoints"]:
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
            tost_margin_db, tost_alpha))
        lines.append("")
        lines.append("| endpoint | p_lower | p_upper | passes |")
        lines.append("|---|---|---|---|")
        for key in SPEC_R["primary_endpoints"]:
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
        for key in SPEC_R["primary_endpoints"]:
            lines.append("| %s %s | **%s** |" % (
                key, SPEC_R["endpoints"][key]["label"], block["verdicts"][key]["verdict"]))
        lines.append("")

    lines.append("## Reserved-unit check")
    lines.append("")
    ru = report["reserved_unit_check"]
    lines.append("reporting %s cells; consistent: %s; reserved/total: %s" % (
        ru["reported_as"], ru["consistent"], ru.get("reserved_fraction", "-")))
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    for key in SPEC_R["primary_endpoints"]:
        itt = report["analysis"]["itt"]["verdicts"][key]["verdict"]
        me_block = report["analysis"].get("mechanism_exercised")
        me = me_block["verdicts"][key]["verdict"] if me_block else "-"
        lines.append("- %s (%s): ITT **%s**; mechanism-exercised %s"
                     % (key, SPEC_R["endpoints"][key]["label"], itt, me))
    if report["blocking_errors"]:
        lines.append("")
        lines.append("## BLOCKING ERRORS")
        lines.append("")
        for msg in report["blocking_errors"]:
            lines.append("- %s" % msg)
    return "\n".join(lines)


def _paired_ci(entry):
    if entry["ci_low"] is None:
        return "-"
    return "[%s, %s]" % (_fmt(entry["ci_low"]), _fmt(entry["ci_high"]))


def markdown_report_paired(report):
    S = report.get("spec", SPEC)
    lines = []
    lines.append("# Real-data gate analysis, PAIRED (%s)" % S["scene"])
    lines.append("")
    lines.append("spec %s v%s | wave %s%s | within-prefix, DESCRIPTIVE "
                 "(no p-value, no equivalence test)" % (
                     S["spec_id"], S["spec_version"], report["wave"],
                     " (LAST WAVE)" if report["last_wave"] else ""))
    if report.get("spec_sha256"):
        lines.append("")
        lines.append("spec file %s sha256 %s" % (S.get("spec_file"), report["spec_sha256"]))
    lines.append("")

    lines.append("## Cells")
    lines.append("")
    endpoint_keys = [k for k in S["psnr_endpoints"] if k in S["endpoints"]]
    header = ["prefix", "arm", "seed", "status", "ME"] + endpoint_keys + ["points"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for c in sorted(report["cells"], key=lambda c: (str(c.get("prefix")), c["arm"])):
        if c["status"] != "complete":
            lines.append("| %s | %s | %s | %s |" % (
                c.get("prefix"), c["arm"], c["seed"],
                " | ".join([c["status"]] + ["-"] * (len(endpoint_keys) + 2)),
            ))
            continue
        e = c["endpoints"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            c.get("prefix"), c["arm"], c["seed"], c["status"],
            "yes" if c["mechanism_exercised"] else "no",
            " | ".join(_fmt(e.get(k)) for k in endpoint_keys),
            _fmt(c.get("final_points"), 0),
        ))
    lines.append("")

    for set_name in S["ANALYSIS_SETS"]:
        block = report["analysis"].get(set_name)
        if block is None:
            continue
        lines.append("## Analysis set: %s (%d complete U/G pairs; arms %s)" % (
            set_name, block["n_complete_pairs"],
            ", ".join("%s=%d" % (a, n) for a, n in block["arms_present"].items())))
        lines.append("")
        lines.append("### Paired contrasts")
        lines.append("")
        lines.append("| contrast | endpoint | n pairs | median | min | max | same-sign | "
                     "mean | 97.5% CI | note |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for name, per_endpoint in block["contrasts"].items():
            for key in endpoint_keys:
                entry = per_endpoint[key]
                lines.append("| %s | %s | %d | %s | %s | %s | %d/%d | %s | %s | %s |" % (
                    name, key, entry["n_pairs"], _fmt(entry["median"]),
                    _fmt(entry["min"]), _fmt(entry["max"]),
                    entry["n_same_sign"], entry["n_pairs"], _fmt(entry["mean"]),
                    _paired_ci(entry),
                    "descriptive" if entry["descriptive"] else "reported",
                ))
        lines.append("")

        lines.append("### Within-prefix replicate floor (|U - %s|)" % PAIRED_FLOOR_ARM)
        lines.append("")
        lines.append("| endpoint | n | median | max | source | fallback |")
        lines.append("|---|---|---|---|---|---|")
        for key in endpoint_keys:
            fl = block["replicate_floor"][key]
            lines.append("| %s | %d | %s | %s | %s | %s |" % (
                key, fl["n"], _fmt(fl["median"]), _fmt(fl["max"]), fl["source"],
                _fmt(fl["fallback_db"], 2)))
        lines.append("")

        lines.append("### Paired sizing (DELTA=%.2f dB, cap %d pairs)" % (
            S["SIZING"]["DELTA"], S["SIZING"]["N2_CAP"]))
        lines.append("")
        lines.append("| endpoint | sd paired | n2 pairs |")
        lines.append("|---|---|---|")
        for key, item in block["sizing"]["per_endpoint"].items():
            lines.append("| %s | %s | %s |" % (
                key, _fmt(item["sd_paired"]), _fmt(item["n2_pairs"], 0)))
        lines.append("")
        lines.append("n2_pairs_max %s, current %d pairs -> **%s**" % (
            _fmt(block["sizing"]["n2_pairs_max"], 0),
            block["sizing"]["n_pairs_current"], block["sizing"]["decision"]))
        lines.append("")

        lines.append("### Claim conditions (descriptive)")
        lines.append("")
        lines.append("| endpoint | every G-U pair over floor | sham contrasts clean | verdict |")
        lines.append("|---|---|---|---|")
        for key, item in block["verdicts"].items():
            lines.append("| %s | %s | %s | **%s** |" % (
                key,
                "yes" if item["every_pair_exceeds_floor"] else "no",
                "yes" if item["sham_contrasts_clean"] else "no",
                item["verdict"],
            ))
        lines.append("")
        lines.append("placebo split: %s" % block["placebo_role"])
        lines.append("")

    lines.append("## Reserved-unit check")
    lines.append("")
    ru = report["reserved_unit_check"]
    lines.append("reporting %s cells; consistent: %s; reserved/total: %s" % (
        ru["reported_as"], ru["consistent"], ru.get("reserved_fraction", "-")))
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    for key in sorted(report["headline"]):
        row = report["headline"][key]
        lines.append("- %s: ITT **%s**; mechanism-exercised %s"
                     % (key, row["itt"], row["mechanism_exercised"]))
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


def run(manifest_path, last_wave=False, wave=1, spec=None, paired=False):
    S = _spec(spec)
    if S.get("unresolved_endpoints"):
        raise ValueError(
            "spec endpoints %s still reference null anchors; set %s in the spec file"
            % (", ".join(S["unresolved_endpoints"]), "/".join(ANCHOR_KEYS))
        )
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    if "cells" not in manifest:
        raise ValueError(f"{manifest_path}: missing 'cells'")
    allowed_arms = (
        sorted(set(S["arms"]) | set(PAIRED_ARMS)) if paired else None
    )
    cells = [
        load_cell(entry, spec=S, arms=allowed_arms, require_prefix=paired)
        for entry in manifest["cells"]
    ]

    blocking = []
    warnings = []
    seen = set()
    for c in cells:
        key = (c["arm"], c["seed"])
        if key in seen:
            warnings.append("duplicate (arm, seed) %s/%s" % key)
        seen.add(key)
        if c["status"] == "complete" and c["n_frames"] != S["expected_n_frames"]:
            warnings.append(
                "%s/seed%s: profile has %s frames, expected %d"
                % (c["arm"], c["seed"], c["n_frames"], S["expected_n_frames"])
            )

    complete = [c for c in cells if c["status"] == "complete"]
    failed = [c for c in cells if c["status"] == "failed"]

    ru = reserved_unit_check(complete)
    if not ru["consistent"]:
        blocking.append(ru.get("message", "reserved-unit check failed"))
    if S.get("RESERVED_UNITS_REQUIRED", False):
        lacking = [
            "%s/seed%s" % (c["arm"], c["seed"]) for c in complete
            if not c.get("precondition")
            or c["precondition"].get("reserved_units") is None
            or c["precondition"].get("training_units_total") is None
        ]
        if lacking:
            blocking.append(
                "RESERVED_UNITS_REQUIRED: cells without reserved_units/"
                "training_units_total: %s" % ", ".join(lacking)
            )

    me_cells = [c for c in complete if c["mechanism_exercised"]]
    if paired:
        itt = analyse_paired(complete, "itt", spec=S)
        me = analyse_paired(me_cells, "mechanism_exercised", spec=S)
        for arm, dupes in itt["duplicate_prefixes"].items():
            warnings.append("arm %s has more than one cell at prefix(es) %s" % (arm, dupes))
    else:
        itt = analyse(complete, "itt", last_wave, spec=S)
        me = analyse(me_cells, "mechanism_exercised", last_wave, spec=S)

    report = {
        "spec": S,
        "manifest": os.path.abspath(manifest_path),
        "wave": wave,
        "last_wave": bool(last_wave),
        "n_cells": len(cells),
        "n_complete": len(complete),
        "failed_cells": [
            {"arm": c["arm"], "seed": c["seed"], "run_dir": c["run_dir"]} for c in failed
        ],
        "cells": [
            dict(
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
                },
                **({"prefix": c["prefix"]} if c.get("prefix") is not None else {}),
            )
            for c in cells
        ],
        "analysis": {"itt": itt, "mechanism_exercised": me},
        "reserved_unit_check": ru,
        "warnings": warnings,
        "blocking_errors": blocking,
        "headline": {
            key: {
                "itt": itt["verdicts"][key]["verdict"],
                "mechanism_exercised": (
                    me["verdicts"][key]["verdict"]
                    if key in me["verdicts"]
                    else "DESIGN_WITHOUT_POWER"
                ),
            }
            for key in S["primary_endpoints"]
            if key in itt["verdicts"]
        },
    }
    if paired:
        report["paired"] = True
        report["design"] = "within-prefix paired, descriptive"
        operative = S.get("PAIRED_CLAIM_SET", "itt")
        report["operative_set"] = operative
        for row in report["headline"].values():
            row["operative"] = row.get(operative, "DESIGN_WITHOUT_POWER")
        if blocking:
            for row in report["headline"].values():
                row["operative"] = "BLOCKED"
    if S.get("spec_sha256"):
        report["spec_file"] = S.get("spec_file")
        report["spec_sha256"] = S["spec_sha256"]
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
    ap.add_argument(
        "--spec",
        help="json spec instance deep-merged onto a copy of SPEC (reports spec_version %s)"
        % SPEC_VERSION_WITH_OVERRIDE,
    )
    ap.add_argument(
        "--paired",
        action="store_true",
        help="within-prefix paired analysis; every cell must carry a 'prefix'",
    )
    args = ap.parse_args(argv)

    spec = None
    if args.spec:
        spec, _ = load_spec(args.spec)

    if args.print_spec:
        print(json.dumps(_spec(spec), indent=1, sort_keys=True))
        return 0
    if not args.manifest:
        ap.error("--manifest is required unless --print-spec is given")

    report = run(
        args.manifest,
        last_wave=args.last_wave,
        wave=args.wave,
        spec=spec,
        paired=args.paired,
    )
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
