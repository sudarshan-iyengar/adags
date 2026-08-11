"""Frozen reference (ORACLE) transcription of EL-GS v8 formal spec, section 2.

SOURCE OF TRUTH
---------------
`research-wiki/operations/elgs-v8-formal-spec.md`, section "## 2. Evidence
objects" (lines 198-220 of that file at the time of transcription).

This module is a deliberately naive, straight-line, numpy-only transcription of
the section-2 likelihood objects.  It exists so that a SEPARATELY written
implementation can be diffed against an independent reading of the spec text.
It is not optimized, not vectorized, and not intended to be imported by the
implementation.  Loops over reports are the intended style.

The transcribed section-2 text, verbatim, is:

    Observation space Y = {miss} u ([v_min,v_max] x D_img), where the
    positional coordinate is the RAW report position in the image domain
    D_img (bounded). Heads (all normalized densities over Y w.r.t. one base
    measure; fitted only on calibration scenes, frozen):
    p_vis(y|b) = 1_miss*pi_m^v + (1-1_miss)(1-pi_m^v)*g_v(v)*g_pos(y_pos|b),
    with g_pos = a truncated-Gaussian density over D_img centered at the
    bridge-projected point (normalized over D_img for every b BY CONSTRUCTION
    -- no Jacobian issue);
    p_cens(y) = 1_miss*pi_m^c + (1-1_miss)(1-pi_m^c)*h_c(v)*(1/|D_img|);
    p_out analogous. FLOORS/CAPS: h_{c,o} >= h_floor > 0;
    pi_m^{c,o} in [pi_floor, 1-pi_floor]; g_v <= g_cap; g_pos <= pos_cap;
    |D_img| > 0. r_u in [r_min,1]; d_u in [0,1].
    L1(y|b,q~,r) = r[q~*p_vis + (1-q~)*p_cens] + (1-r)*p_out;
    L0(y|r) = r*p_cens + (1-r)*p_out. Censoring equality at q~=0: identical.

(The ASCII rendering above is for terminal-safe reading; the original Unicode
lines are quoted inside the individual function docstrings.)

BASE MEASURE (transcription note, not a spec deviation)
-------------------------------------------------------
Section 2 says the heads are "all normalized densities over Y w.r.t. one base
measure".  Y = {miss} disjoint-union ([v_min,v_max] x D_img), so the natural
single base measure is (counting measure on the singleton {miss}) + (Lebesgue
measure on [v_min,v_max] x D_img).  Consequently the value returned for a MISS
report is a probability MASS and the value returned for a positional report is
a probability DENSITY (units 1/(value * area)).  They are returned by the same
function because they are two branches of one density w.r.t. the one mixed base
measure.  Every function below returns that mixed-measure value.

AMBIGUITIES FLAGGED (see the per-function docstrings for detail)
----------------------------------------------------------------
A1  g_v, h_c, h_o families are NOT fixed by section 2 -> caller-supplied
    callables carried on `LikelihoodParams`.
A2  The covariance of the "truncated-Gaussian" g_pos is not stated -> this
    reference uses an isotropic, axis-aligned sigma^2 * I.
A3  "p_out analogous" is not written out -> transcribed as the p_cens form
    with pi_m^o and h_o.
A4  Section 2 bounds pi_m^{c,o} only; pi_m^v is bounded here too.
A5  Floors/caps are stated as constraints, not as clamps -> this reference
    RAISES on violation rather than clamping.
A6  Reports outside Y (v outside [v_min,v_max], pos outside D_img) are not
    discussed -> this reference returns 0.0 for the non-miss branch.
A7  "bridge-projected point" -> this reference takes the ALREADY-projected
    2-D image-domain point as input; the projection operator is not part of
    section 2.
"""

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# 1. Observation space Y
# ---------------------------------------------------------------------------


@dataclass
class Report:
    """One element y of the section-2 observation space Y.

    Transcribes, verbatim from section 2:

        "Observation space Y = {miss} ⊔ ([v_min,v_max] × D_img), where
        the positional coordinate is the RAW report position in the image
        domain D_img (bounded)."

    Fields
    ------
    is_miss : bool
        True  -> y is the atom `miss` (the {miss} summand of Y).  `v` and
                 `pos` are then unused and may be NaN.
        False -> y lies in [v_min, v_max] x D_img and BOTH `v` and `pos` are
                 meaningful.
    v : float
        The scalar report value; section 2 constrains it to [v_min, v_max]
        but does not otherwise name it.
    pos : tuple of (float, float)
        The RAW report position (x, y) in the bounded image domain D_img.
        Section 2 is explicit that this is the raw position, not a
        normalized / whitened / residual coordinate.

    AMBIGUITY A6: section 2 does not say what happens for a non-miss report
    whose v or pos falls outside Y.  Such a point is simply not in Y, so this
    reference assigns it density 0.0 in every head (see `_g_v_value`,
    `g_pos_density`, `_h_value`).  An implementation that instead raises, or
    that clamps into Y, is a different reading and must be adjudicated
    against the spec text.
    """

    is_miss: bool
    v: float = float("nan")
    pos: tuple = (float("nan"), float("nan"))


# ---------------------------------------------------------------------------
# 2. Parameters, floors and caps
# ---------------------------------------------------------------------------


@dataclass
class LikelihoodParams:
    """Section-2 head parameters together with the section-2 floors and caps.

    Transcribes, verbatim from section 2:

        "FLOORS/CAPS: h_{c,o} ≥ h_floor > 0;
        π_m^{c,o} ∈ [π_floor, 1−π_floor]; g_v ≤ g_cap; g_pos ≤ pos_cap;
        |D_img| > 0. r_u ∈ [r_min,1]; d_u ∈ [0,1]."

    Section 2 fixes NO numeric values for any floor or cap; every value here
    is caller-supplied.  Nothing in this file invents a default for
    pi_floor, h_floor, g_cap, pos_cap, r_min or g_pos_sigma.

    AMBIGUITY A1: section 2 says the heads are "fitted only on calibration
    scenes, frozen" but does not fix the family of g_v, h_c or h_o.  They are
    therefore caller-supplied callables:
        g_v(v) -> float, a density in v on [v_min, v_max];
        h_c(v) -> float, a density in v on [v_min, v_max];
        h_o(v) -> float, a density in v on [v_min, v_max].
    Section 2 does require them to be normalized densities over Y w.r.t. the
    one base measure; with the uniform positional factor 1/|D_img| that means
    h_c and h_o must integrate to 1 over [v_min, v_max].  This reference does
    NOT verify normalization (it is a property of the caller's fitted head),
    it only enforces the stated floors and caps.

    AMBIGUITY A4: section 2 writes the interval constraint for
    "π_m^{c,o}" only -- pi_m^v is not named in the FLOORS/CAPS clause.  This
    reference applies the same [pi_floor, 1-pi_floor] interval to pi_m^v as
    well, because pi_m^v is a miss probability of the same kind and an
    unfloored pi_m^v would break the same likelihood-ratio boundedness that
    the clause exists to guarantee.  This is an EXTENSION of the literal
    text and is flagged as such; an implementation that leaves pi_m^v
    unconstrained is a defensible alternative reading.

    AMBIGUITY A5: the clause states floors and caps as CONSTRAINTS on the
    frozen fitted heads, not as runtime clamps.  This reference therefore
    RAISES ValueError when an evaluated head violates them
    (`enforce_head_bounds=True`).  An implementation that clamps instead of
    raising produces different numbers at the boundary and must be
    adjudicated against the spec text.  Set `enforce_head_bounds=False` to
    observe the raw transcription without the guard.
    """

    # miss probabilities of the three heads
    pi_m_v: float
    pi_m_c: float
    pi_m_o: float

    # D_img = the bounded image domain, taken here as the rectangle
    # [x0, x1] x [y0, y1].  Section 2 says only "the image domain D_img
    # (bounded)"; a rectangle is the standard image domain and is what makes
    # the erf normalization below exact.
    x0: float
    x1: float
    y0: float
    y1: float

    # isotropic standard deviation of g_pos (AMBIGUITY A2)
    g_pos_sigma: float

    # value-coordinate bounds of Y
    v_min: float
    v_max: float

    # r_u ∈ [r_min, 1]
    r_min: float

    # section-2 floors and caps
    h_floor: float
    pi_floor: float
    g_cap: float
    pos_cap: float

    # caller-supplied head families (AMBIGUITY A1)
    g_v: object = None
    h_c: object = None
    h_o: object = None

    # AMBIGUITY A5 switch
    enforce_head_bounds: bool = True

    # -- derived -----------------------------------------------------------

    def d_img_area(self):
        """|D_img| for the rectangular image domain: (x1 - x0) * (y1 - y0)."""
        return float((self.x1 - self.x0) * (self.y1 - self.y0))

    # -- validation --------------------------------------------------------

    def validate(self, r=None, d=None):
        """Check every section-2 floor/cap constraint that is checkable statically.

        Transcribes, verbatim from section 2:

            "FLOORS/CAPS: h_{c,o} ≥ h_floor > 0;
            π_m^{c,o} ∈ [π_floor, 1−π_floor]; g_v ≤ g_cap; g_pos ≤ pos_cap;
            |D_img| > 0. r_u ∈ [r_min,1]; d_u ∈ [0,1]."

        Statically checkable here: h_floor > 0; the pi_m interval; |D_img| > 0;
        and, when supplied, r in [r_min, 1] and d in [0, 1].

        The constraints "h_{c,o} >= h_floor", "g_v <= g_cap" and
        "g_pos <= pos_cap" are properties of EVALUATED head values, so they
        are enforced at evaluation time by `_h_value`, `_g_v_value` and
        `g_pos_density` respectively (see AMBIGUITY A5).

        Raises ValueError on the first violation found.
        """
        # h_floor > 0 is stated explicitly ("h_{c,o} >= h_floor > 0").
        if not (self.h_floor > 0.0):
            raise ValueError("section 2 requires h_floor > 0; got %r" % (self.h_floor,))

        # pi_floor must define a non-empty interval [pi_floor, 1 - pi_floor].
        if not (0.0 <= self.pi_floor <= 0.5):
            raise ValueError(
                "pi_floor must lie in [0, 0.5] for [pi_floor, 1-pi_floor] to be "
                "non-empty; got %r" % (self.pi_floor,)
            )

        # pi_m^{c,o} in [pi_floor, 1 - pi_floor]  (literal section-2 text),
        # plus pi_m^v under EXTENSION A4.
        for name, value in (
            ("pi_m_c", self.pi_m_c),
            ("pi_m_o", self.pi_m_o),
            ("pi_m_v", self.pi_m_v),  # AMBIGUITY A4: extension beyond literal text
        ):
            if not (self.pi_floor <= value <= 1.0 - self.pi_floor):
                raise ValueError(
                    "section 2 requires %s in [pi_floor, 1-pi_floor] = [%r, %r]; got %r"
                    % (name, self.pi_floor, 1.0 - self.pi_floor, value)
                )

        # |D_img| > 0
        if not (self.d_img_area() > 0.0):
            raise ValueError(
                "section 2 requires |D_img| > 0; got area %r for rectangle "
                "[%r,%r] x [%r,%r]" % (self.d_img_area(), self.x0, self.x1, self.y0, self.y1)
            )

        # caps must be positive to be meaningful upper bounds on densities
        if not (self.g_cap > 0.0):
            raise ValueError("g_cap must be > 0; got %r" % (self.g_cap,))
        if not (self.pos_cap > 0.0):
            raise ValueError("pos_cap must be > 0; got %r" % (self.pos_cap,))

        # value-coordinate interval [v_min, v_max] must be non-degenerate
        if not (self.v_min < self.v_max):
            raise ValueError(
                "Y requires a non-degenerate [v_min, v_max]; got [%r, %r]"
                % (self.v_min, self.v_max)
            )

        # g_pos sigma must be positive for the truncated Gaussian to exist
        if not (self.g_pos_sigma > 0.0):
            raise ValueError("g_pos_sigma must be > 0; got %r" % (self.g_pos_sigma,))

        # r_min must itself be a valid lower end of [r_min, 1]
        if not (0.0 <= self.r_min <= 1.0):
            raise ValueError("r_min must lie in [0, 1]; got %r" % (self.r_min,))

        if r is not None:
            self.validate_r(r)
        if d is not None:
            self.validate_d(d)

        return True

    def validate_r(self, r):
        """Check "r_u ∈ [r_min,1]" (verbatim section-2 fragment)."""
        if not (self.r_min <= r <= 1.0):
            raise ValueError(
                "section 2 requires r_u in [r_min, 1] = [%r, 1]; got %r" % (self.r_min, r)
            )
        return True

    def validate_d(self, d):
        """Check "d_u ∈ [0,1]" (verbatim section-2 fragment)."""
        if not (0.0 <= d <= 1.0):
            raise ValueError("section 2 requires d_u in [0, 1]; got %r" % (d,))
        return True

    def validate_q_tilde(self, q_tilde):
        """Check q~ in [0, 1].

        Section 2 itself only carries q~ as an argument of L1; the range is
        pinned in section 3 ("q̃ = q·d_u ∈ [0,1] always"), which section 2's
        L1 line depends on.  Included because L1 below takes q~ directly.
        """
        if not (0.0 <= q_tilde <= 1.0):
            raise ValueError("q~ must lie in [0, 1]; got %r" % (q_tilde,))
        return True


# ---------------------------------------------------------------------------
# 3. Head building blocks
# ---------------------------------------------------------------------------


def _in_v_range(v, params):
    """True iff v lies in the [v_min, v_max] factor of Y."""
    return bool(params.v_min <= v <= params.v_max)


def _in_d_img(pos, params):
    """True iff pos lies in the bounded rectangular image domain D_img."""
    x, y = pos
    return bool(params.x0 <= x <= params.x1 and params.y0 <= y <= params.y1)


def _std_normal_cdf(z):
    """Standard normal CDF via math.erf (exact, closed form -- no Monte Carlo)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _gaussian_interval_mass(mu, sigma, lo, hi):
    """Mass of N(mu, sigma^2) on [lo, hi], computed with erf."""
    return _std_normal_cdf((hi - mu) / sigma) - _std_normal_cdf((lo - mu) / sigma)


def g_pos_density(pos, bridge_point, params):
    """g_pos(y_pos | b): truncated-Gaussian positional density over D_img.

    Transcribes, verbatim from section 2:

        "with g_pos = a truncated-Gaussian density over D_img centered at the
        bridge-projected point (normalized over D_img for every b BY
        CONSTRUCTION — no Jacobian issue)"

    and enforces the cap fragment "g_pos ≤ pos_cap".

    Construction
    ------------
    Untruncated density is the isotropic 2-D Gaussian
    N(bridge_point, sigma^2 * I).  Truncation to the rectangle D_img divides
    by the exact rectangle mass, which for the separable isotropic Gaussian
    factorizes as Zx * Zy with

        Zx = Phi((x1-mx)/sigma) - Phi((x0-mx)/sigma)
        Zy = Phi((y1-my)/sigma) - Phi((y0-my)/sigma)

    each evaluated with math.erf.  This is the exact 2-D Gaussian integral
    over the rectangle -- deliberately NOT a Monte-Carlo estimate.  The
    result integrates to exactly 1 over D_img for every b, which is the
    "normalized over D_img for every b BY CONSTRUCTION" clause.

    AMBIGUITY A2: section 2 says "truncated-Gaussian" without giving a
    covariance.  This reference uses an isotropic, axis-aligned covariance
    sigma^2 * I with sigma = params.g_pos_sigma.  A per-bridge anisotropic
    covariance (e.g. propagated from the sigma-point bridge) is a defensible
    alternative reading of the same words and would change every number here;
    it must be adjudicated against the spec text, not against this file.

    AMBIGUITY A7: `bridge_point` is the ALREADY image-projected bridge point
    (x, y) in D_img.  Section 2 names it "the bridge-projected point" but the
    projection operator itself is defined outside section 2, so it is an
    input here.

    AMBIGUITY A6: positions outside D_img are outside Y; this reference
    returns 0.0 (which is also what "truncated to D_img" means).
    """
    if not _in_d_img(pos, params):
        return 0.0

    x, y = float(pos[0]), float(pos[1])
    mx, my = float(bridge_point[0]), float(bridge_point[1])
    sigma = float(params.g_pos_sigma)

    # untruncated isotropic 2-D Gaussian density at (x, y)
    norm_1d = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    phi_x = norm_1d * np.exp(-0.5 * ((x - mx) / sigma) ** 2)
    phi_y = norm_1d * np.exp(-0.5 * ((y - my) / sigma) ** 2)
    untruncated = float(phi_x * phi_y)

    # exact rectangle mass (erf), factorized because the density is separable
    z_x = _gaussian_interval_mass(mx, sigma, params.x0, params.x1)
    z_y = _gaussian_interval_mass(my, sigma, params.y0, params.y1)
    z = z_x * z_y
    if not (z > 0.0):
        raise ValueError(
            "truncated-Gaussian normalizer over D_img underflowed to %r; the "
            "bridge-projected point %r is too far outside D_img relative to "
            "sigma=%r for a normalized density to be representable"
            % (z, (mx, my), sigma)
        )

    value = untruncated / z

    # section 2: "g_pos <= pos_cap"
    if params.enforce_head_bounds and value > params.pos_cap:
        raise ValueError(
            "section 2 requires g_pos <= pos_cap = %r; got %r at pos=%r, b=%r"
            % (params.pos_cap, value, pos, bridge_point)
        )
    return value


def _g_v_value(v, params):
    """Evaluate the caller-supplied g_v and enforce "g_v ≤ g_cap" (section 2).

    AMBIGUITY A1: section 2 does not fix the family of g_v; it is supplied by
    the caller on `params.g_v`.  Section 2 gives g_v an UPPER cap (g_cap) but,
    unlike h_{c,o}, no lower floor -- that asymmetry is transcribed literally.
    """
    if params.g_v is None:
        raise ValueError("params.g_v is required (section 2 does not fix its family)")
    if not _in_v_range(v, params):
        return 0.0  # AMBIGUITY A6
    value = float(params.g_v(v))
    if params.enforce_head_bounds and value > params.g_cap:
        raise ValueError(
            "section 2 requires g_v <= g_cap = %r; got %r at v=%r"
            % (params.g_cap, value, v)
        )
    return value


def _h_value(v, head, name, params):
    """Evaluate a caller-supplied h_c / h_o and enforce "h_{c,o} ≥ h_floor > 0".

    AMBIGUITY A1: section 2 does not fix the family of h_c or h_o.
    Section 2 gives h_{c,o} a LOWER floor but no explicit upper cap; that
    asymmetry (mirror image of g_v's) is transcribed literally.

    AMBIGUITY A6: v outside [v_min, v_max] is outside Y, so 0.0 is returned
    WITHOUT applying the floor -- the floor constrains the fitted head on its
    support, not the density of points that are not in the observation space.
    """
    if head is None:
        raise ValueError("params.%s is required (section 2 does not fix its family)" % name)
    if not _in_v_range(v, params):
        return 0.0
    value = float(head(v))
    if params.enforce_head_bounds and value < params.h_floor:
        raise ValueError(
            "section 2 requires %s >= h_floor = %r; got %r at v=%r"
            % (name, params.h_floor, value, v)
        )
    return value


# ---------------------------------------------------------------------------
# 4. The three heads
# ---------------------------------------------------------------------------


def p_vis(report, bridge_point, params):
    """p_vis(y|b) -- the visible head.

    Transcribes, verbatim from section 2:

        "p_vis(y|b) = 1_miss·π_m^v + (1−1_miss)(1−π_m^v)·g_v(v)·g_pos(y_pos|b),"

    1_miss is the indicator of the {miss} atom of Y, so the two branches are
    mutually exclusive and this is a straight two-branch evaluation.
    """
    if report.is_miss:
        # 1_miss = 1 -> the whole expression collapses to pi_m^v.
        return float(params.pi_m_v)
    # 1_miss = 0 -> (1 - pi_m^v) * g_v(v) * g_pos(y_pos | b)
    g_v_val = _g_v_value(report.v, params)
    g_pos_val = g_pos_density(report.pos, bridge_point, params)
    return float((1.0 - params.pi_m_v) * g_v_val * g_pos_val)


def p_cens(report, params):
    """p_cens(y) -- the censored head.

    Transcribes, verbatim from section 2:

        "p_cens(y) = 1_miss·π_m^c + (1−1_miss)(1−π_m^c)·h_c(v)·(1/|D_img|);"

    Note the positional factor is the UNIFORM density 1/|D_img| -- it does not
    depend on the bridge point b at all.  That b-independence is exactly what
    makes the section-2 censoring equality hold (see `censoring_equality_gap`).
    """
    if report.is_miss:
        return float(params.pi_m_c)
    h_c_val = _h_value(report.v, params.h_c, "h_c", params)
    uniform_pos = 1.0 / params.d_img_area()
    # AMBIGUITY A6: a position outside D_img is outside Y -> density 0.
    if not _in_d_img(report.pos, params):
        return 0.0
    return float((1.0 - params.pi_m_c) * h_c_val * uniform_pos)


def p_out(report, params):
    """p_out(y) -- the outlier head.

    Transcribes, verbatim from section 2:

        "p_out analogous."

    AMBIGUITY A3 (transcription choice, load-bearing).  Section 2 writes
    p_out only as "analogous", immediately after the fully written-out
    p_cens line.  This reference reads "analogous" as: the SAME structural
    form as p_cens, with its own miss probability pi_m^o and its own value
    head h_o, and the same uniform positional factor 1/|D_img|:

        p_out(y) = 1_miss * pi_m^o
                 + (1 - 1_miss)(1 - pi_m^o) * h_o(v) * (1/|D_img|)

    Evidence inside section 2 for pairing p_out with p_cens rather than with
    p_vis: the FLOORS/CAPS clause groups them as "h_{c,o} ≥ h_floor > 0" and
    "π_m^{c,o} ∈ [π_floor, 1−π_floor]" -- i.e. section 2 asserts that p_out
    has an h-type value head (h_o) and a floored miss probability, exactly
    like p_cens and unlike p_vis (which has a g_v/g_pos pair with caps).
    A reading in which p_out were positionally centred on the bridge point
    would also make p_out depend on b, which would break the section-2
    censoring equality's b-independence of L0.  This reading is therefore
    strongly constrained but is NOT literally written out in section 2 and
    must be adjudicated against the spec text if an implementation differs.
    """
    if report.is_miss:
        return float(params.pi_m_o)
    h_o_val = _h_value(report.v, params.h_o, "h_o", params)
    uniform_pos = 1.0 / params.d_img_area()
    if not _in_d_img(report.pos, params):
        return 0.0  # AMBIGUITY A6
    return float((1.0 - params.pi_m_o) * h_o_val * uniform_pos)


# ---------------------------------------------------------------------------
# 5. The two mixture likelihoods
# ---------------------------------------------------------------------------


def L1(report, bridge_point, q_tilde, r, params):
    """L1(y|b,q~,r) -- the family-PRESENT likelihood.

    Transcribes, verbatim from section 2:

        "L1(y|b,q̃,r) = r[q̃·p_vis + (1−q̃)·p_cens] + (1−r)·p_out;"

    The expression is evaluated in exactly the written associativity so that
    substituting q~ = 0 reproduces `L0` BIT-FOR-BIT in IEEE double arithmetic
    (see `censoring_equality_gap`): with q~ = 0.0 the bracket evaluates to
    0.0*p_vis + 1.0*p_cens = p_cens exactly, and the outer expression becomes
    r*p_cens + (1-r)*p_out, which is the literal `L0` expression.
    """
    params.validate_r(r)
    params.validate_q_tilde(q_tilde)
    vis = p_vis(report, bridge_point, params)
    cens = p_cens(report, params)
    out = p_out(report, params)
    return r * (q_tilde * vis + (1.0 - q_tilde) * cens) + (1.0 - r) * out


def L0(report, r, params):
    """L0(y|r) -- the family-ABSENT likelihood.

    Transcribes, verbatim from section 2:

        "L0(y|r) = r·p_cens + (1−r)·p_out."

    Depends on neither b nor q~.
    """
    params.validate_r(r)
    cens = p_cens(report, params)
    out = p_out(report, params)
    return r * cens + (1.0 - r) * out


def censoring_equality_gap(report, bridge_point, r, params):
    """L1(q~=0) - L0, which section 2 requires to be EXACTLY zero.

    Transcribes, verbatim from section 2:

        "Censoring equality at q̃=0: identical."

    "identical" is read as exact equality of the two expressions, not
    approximate agreement: at q~ = 0 the L1 bracket is p_cens and the outer
    mixture is literally the L0 expression.  This function is kept as a
    callable so tests can assert `== 0.0` (not `abs(...) < tol`).

    Any nonzero return means either (a) the caller's heads are not finite
    (0.0 * inf = nan would poison the bracket), or (b) the two expressions
    were not evaluated in the section-2 written form.  Neither is a licence
    to loosen the assertion.
    """
    return L1(report, bridge_point, 0.0, r, params) - L0(report, r, params)


# ---------------------------------------------------------------------------
# 6. Self-check (does not participate in the transcription)
# ---------------------------------------------------------------------------


def _self_check():
    """Construct valid params and print the section-2 quantities.

    Purely illustrative: it exercises the transcription on one miss report and
    one positional report.  It defines NO spec semantics -- the numeric head
    families used here are arbitrary caller-supplied choices (AMBIGUITY A1).
    """
    v_min, v_max = 0.0, 1.0

    # arbitrary valid caller-supplied heads on [v_min, v_max]:
    #   g_v : triangular-ish, peaked at v=1, integrates to 1
    #   h_c : uniform density 1.0 on [0,1]
    #   h_o : uniform density 1.0 on [0,1]
    def g_v(v):
        return 2.0 * v

    def h_c(v):
        return 1.0

    def h_o(v):
        return 1.0

    params = LikelihoodParams(
        pi_m_v=0.10,
        pi_m_c=0.60,
        pi_m_o=0.30,
        x0=0.0,
        x1=64.0,
        y0=0.0,
        y1=48.0,
        g_pos_sigma=2.0,
        v_min=v_min,
        v_max=v_max,
        r_min=0.05,
        h_floor=1e-6,
        pi_floor=1e-3,
        g_cap=10.0,
        pos_cap=10.0,
        g_v=g_v,
        h_c=h_c,
        h_o=h_o,
    )
    params.validate(r=0.7, d=0.4)

    bridge_point = (32.0, 24.0)
    miss = Report(is_miss=True)
    hit = Report(is_miss=False, v=0.75, pos=(33.0, 23.0))

    r = 0.7
    d_u = 0.4
    q = 0.9
    q_tilde = q * d_u  # section 3: q~ = q * d_u in [0,1]

    print("|D_img|            = %.17g" % params.d_img_area())
    print("q~ (= q * d_u)     = %.17g" % q_tilde)
    print("r                  = %.17g" % r)
    print("")
    for label, rep in (("miss", miss), ("hit", hit)):
        print("-- report: %s --" % label)
        print("  p_vis            = %.17g" % p_vis(rep, bridge_point, params))
        print("  p_cens           = %.17g" % p_cens(rep, params))
        print("  p_out            = %.17g" % p_out(rep, params))
        print("  L1(q~=%.2f)      = %.17g" % (q_tilde, L1(rep, bridge_point, q_tilde, r, params)))
        print("  L1(q~=0)         = %.17g" % L1(rep, bridge_point, 0.0, r, params))
        print("  L0               = %.17g" % L0(rep, r, params))
        gap = censoring_equality_gap(rep, bridge_point, r, params)
        print("  censoring gap    = %r  (exact zero: %s)" % (gap, gap == 0.0))
        print("")
    print("g_pos at bridge centre = %.17g" % g_pos_density(bridge_point, bridge_point, params))


# ---------------------------------------------------------------------------
# 7. Freeze note
# ---------------------------------------------------------------------------

REFERENCE_SHA_NOTE = """
FROZEN ORACLE -- DO NOT EDIT TO MATCH AN IMPLEMENTATION.

This file was written from a single source, in a context with no access to any
EL-GS implementation, implementation plan, or module under `elgs/`: section 2
("Evidence objects") of `research-wiki/operations/elgs-v8-formal-spec.md`.
Its value is exactly that independence.  It is the test oracle against which a
separately written implementation is compared.

Rules:

1. If a test comparing the implementation against this file FAILS, that is a
   finding, not a maintenance chore.  It means one of:
     (a) the implementation is wrong;
     (b) this transcription is wrong;
     (c) the two readings of the section-2 text differ on a point the text
         does not settle (the flagged ambiguities A1-A7 are the known
         candidates).
2. The divergence MUST be adjudicated against the spec text itself, quoting
   the section-2 line at issue.  It must NOT be resolved by editing this file
   until the implementation's numbers are reproduced.  Doing so destroys the
   only independent check that exists and silently converts the oracle into a
   restatement of the implementation.
3. If adjudication concludes this transcription is wrong (case b), fix it and
   record the spec quotation that justified the change in the commit message.
4. If adjudication concludes the SPEC is ambiguous (case c), the spec must be
   amended first; then this file is re-transcribed from the amended text.
5. Any edit to this file invalidates its sha256, which should be re-recorded
   alongside the spec revision it was transcribed from.

Transcribed from: elgs-v8-formal-spec.md, section 2 (spec rev 4 era,
                  lines 198-220), on 2026-08-11.
"""


if __name__ == "__main__":
    _self_check()
