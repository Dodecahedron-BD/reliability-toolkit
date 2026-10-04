"""
Two-parameter Weibull life-data analysis.

Fit     : median-rank regression (Bernard, Johnson-adjusted for suspensions)
          + maximum likelihood estimate as an independent cross-check

Notation follows the NIST/SEMATECH e-Handbook, section 8.1.6.2:
    gamma  = shape parameter (dimensionless)
    alpha  = scale parameter / characteristic life (h)
    mu     = location parameter (h), held at 0 in this module

Times are in hours. Multiply by 3.6e3 for SI seconds.

Dependencies: numpy   (math.gamma is standard library)
"""

from math import gamma as GAMMA, log, exp, sqrt
import numpy as np


def _asarr(v):
    """Coerce None / empty / list / array to a 1-D float array."""
    if v is None:
        return np.array([], dtype=float)
    arr = np.atleast_1d(np.asarray(v, dtype=float))
    return arr


# ----------------------------------------------------------------------
# 1.  PLOTTING POSITIONS
# ----------------------------------------------------------------------
def median_ranks(failures, suspensions=None):
    """
    Return (sorted_failure_times, median_rank_estimates_of_F).

    Complete data : Bernard's approximation  F_i = (i - 0.3) / (n + 0.4)
    With suspensions : Johnson's rank-increment method first, then Bernard
                       applied to the adjusted (non-integer) ranks.

        dI = ((n + 1) - I_prev) / (1 + N_remaining)
        I_j = I_prev + dI
    """
    f = np.sort(np.asarray(failures, dtype=float))
    s = _asarr(suspensions)
    s = np.sort(s)

    if s.size == 0:
        n = f.size
        i = np.arange(1, n + 1)
        return f, (i - 0.3) / (n + 0.4)

    # Build the combined ordered event list, flagged F = failure, S = suspension
    events = [(t, 'F') for t in f] + [(t, 'S') for t in s]
    events.sort(key=lambda e: (e[0], e[1] == 'F'))  # suspension first on ties
    n = len(events)

    adj_ranks, I_prev = [], 0.0
    for k, (t, kind) in enumerate(events):
        n_remaining = n - k            # items at risk, this one included
        if kind == 'F':
            dI = ((n + 1) - I_prev) / (1 + n_remaining)
            I_prev += dI
            adj_ranks.append(I_prev)
    adj = np.asarray(adj_ranks)
    return f, (adj - 0.3) / (n + 0.4)


# ----------------------------------------------------------------------
# 2.  PARAMETER ESTIMATION
# ----------------------------------------------------------------------
def fit_mrr(failures, suspensions=None, regress='x_on_y'):
    """
    Median-rank regression fit of the linearised Weibull CDF

        ln ln[1/(1-F)]  =  gamma * ln t  -  gamma * ln alpha
                 y      =       b * x    +      a0

    regress='x_on_y' (RRX, reliability-engineering standard: scatter is in
    the times, not the ranks).  regress='y_on_x' gives the RRY variant.

    Returns gamma, alpha, r, (x, y) for plotting.
    """
    t, F = median_ranks(failures, suspensions)
    x = np.log(t)
    y = np.log(np.log(1.0 / (1.0 - F)))

    if regress == 'y_on_x':
        b, a0 = np.polyfit(x, y, 1)
        g, a = b, exp(-a0 / b)
    else:                                   # x on y, then invert
        c, d0 = np.polyfit(y, x, 1)
        g, a = 1.0 / c, exp(d0)

    r = float(np.corrcoef(x, y)[0, 1])
    return g, a, r, x, y


def fit_mle(failures, suspensions=None, tol=1e-10, g_lo=0.05, g_hi=50.0):
    """
    Maximum likelihood estimate, censored-data form (NIST 8.4.1.3).

    Solve for gamma:
        sum_all t^g ln t / sum_all t^g  -  1/g  -  (1/r) sum_fail ln t = 0
    then
        alpha = ( sum_all t^g / r ) ** (1/g)

    Monotone in gamma, so plain bisection is robust and needs no scipy.
    """
    f = np.asarray(failures, dtype=float)
    s = _asarr(suspensions)
    allt = np.concatenate([f, s]) if s.size else f
    r = f.size
    mean_ln_fail = float(np.mean(np.log(f)))

    def g_eq(g):
        tg = allt ** g
        return float(np.sum(tg * np.log(allt)) / np.sum(tg)) - 1.0 / g - mean_ln_fail

    lo, hi = g_lo, g_hi
    if g_eq(lo) * g_eq(hi) > 0:
        return float('nan'), float('nan')          # no sign change: give up cleanly
    while hi - lo > tol * max(1.0, lo):
        mid = 0.5 * (lo + hi)
        if g_eq(lo) * g_eq(mid) <= 0:
            hi = mid
        else:
            lo = mid
    g = 0.5 * (lo + hi)
    a = (float(np.sum(allt ** g)) / r) ** (1.0 / g)
    return g, a


# ----------------------------------------------------------------------
# 3.  DISTRIBUTION FUNCTIONS  (mu = 0)
# ----------------------------------------------------------------------
def pdf(t, g, a):
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = t > 0
    out[m] = (g / a) * (t[m] / a) ** (g - 1.0) * np.exp(-(t[m] / a) ** g)
    return out


def cdf(t, g, a):
    t = np.asarray(t, dtype=float)
    return 1.0 - np.exp(-(np.maximum(t, 0.0) / a) ** g)


def reliability(t, g, a):
    t = np.asarray(t, dtype=float)
    return np.exp(-(np.maximum(t, 0.0) / a) ** g)


def hazard(t, g, a):
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = t > 0
    out[m] = (g / a) * (t[m] / a) ** (g - 1.0)
    return out


def cum_hazard(t, g, a):
    return (np.maximum(np.asarray(t, dtype=float), 0.0) / a) ** g


def percentile(p, g, a):
    """B-life: age by which a fraction p has failed."""
    return a * (-log(1.0 - p)) ** (1.0 / g)


# ----------------------------------------------------------------------
# 4.  DERIVED LIFE STATISTICS
# ----------------------------------------------------------------------
def life_stats(g, a):
    """All closed-form Weibull descriptors. Returns a dict."""
    G1 = GAMMA(1.0 + 1.0 / g)
    G2 = GAMMA(1.0 + 2.0 / g)
    G3 = GAMMA(1.0 + 3.0 / g)

    mttf = a * G1
    var = a * a * (G2 - G1 * G1)
    sd = sqrt(var)
    skew = (G3 - 3.0 * G1 * G2 + 2.0 * G1 ** 3) / (G2 - G1 * G1) ** 1.5

    return {
        'gamma_fn_1': G1, 'gamma_fn_2': G2,
        'mttf': mttf, 'sd': sd, 'cov': sd / mttf, 'skew': skew,
        'median': a * log(2.0) ** (1.0 / g),
        'mode': a * ((g - 1.0) / g) ** (1.0 / g) if g > 1.0 else 0.0,
        'b1': percentile(0.01, g, a),
        'b10': percentile(0.10, g, a),
        'b50': percentile(0.50, g, a),
        'b10_over_mttf': percentile(0.10, g, a) / mttf,
        'f_at_mttf': float(cdf(mttf, g, a)),
    }


def mechanism(g):
    """Plain-language reading of the shape parameter."""
    if g < 0.85:
        return "infant mortality - install / manufacturing defects; PM is counter-productive"
    if g < 1.15:
        return "random / shock-driven (near-exponential); PM cannot help, use CBM or redundancy"
    if g < 2.0:
        return "weak wearout mixed with variability; check plot for mixed modes"
    if g < 3.6:
        return "clear wearout (fatigue, erosion, abrasion); age replacement justified"
    return "sharp, tight wearout; age replacement highly effective"