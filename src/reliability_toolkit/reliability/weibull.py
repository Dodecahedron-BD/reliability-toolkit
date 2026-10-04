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


# ----------------------------------------------------------------------
# 5.  REPORT TABLE
# ----------------------------------------------------------------------
def report(failures, suspensions=None, title="Weibull Life-Data Analysis",
           eval_times=None, regress='x_on_y'):
    """Fit, print the parameter table, and return a results dict."""
    f = np.sort(np.asarray(failures, dtype=float))
    s = np.sort(_asarr(suspensions))

    g, a, r, x, y = fit_mrr(f, s, regress=regress)
    g_ml, a_ml = fit_mle(f, s)
    st = life_stats(g, a)

    if eval_times is None:
        eval_times = [st['b10'], st['median'], st['mttf'], a]

    W, L = 66, 34                                   # table width, label width
    line, thin = "=" * W, "-" * W

    def row(label, value, unit=""):
        print("  {:<{L}}{:>14}  {}".format(label, value, unit, L=L))

    def head(txt):
        print("\n" + thin)
        print("  " + txt)
        print(thin)

    print("\n" + line)
    print(title.center(W))
    print(line)

    head("1.  SAMPLE")
    row("Failures, r", "{:d}".format(f.size), "-")
    row("Suspensions (right-censored)", "{:d}".format(s.size), "-")
    row("Total units, n", "{:d}".format(f.size + s.size), "-")
    row("First failure, t(1)", "{:,.1f}".format(f[0]), "h")
    row("Last failure, t(r)", "{:,.1f}".format(f[-1]), "h")
    row("Sample mean, t-bar", "{:,.1f}".format(f.mean()), "h")
    row("Sample std dev, s (n-1)", "{:,.1f}".format(f.std(ddof=1)) if f.size > 1 else "n/a", "h")

    head("2.  FITTED PARAMETERS")
    tag = "RRX, x on y" if regress == 'x_on_y' else "RRY, y on x"
    row("Shape, gamma  [MRR {}]".format(tag), "{:.4f}".format(g), "-")
    row("Scale, alpha  [MRR]", "{:,.1f}".format(a), "h")
    row("Location, mu  (held fixed)", "0.0", "h")
    row("Correlation, r", "{:.5f}".format(r), "-")
    row("Coeff. of determination, r^2", "{:.5f}".format(r * r), "-")
    row("Shape, gamma  [MLE cross-check]", "{:.4f}".format(g_ml), "-")
    row("Scale, alpha  [MLE cross-check]", "{:,.1f}".format(a_ml), "h")
    row("Std error of gamma, 0.78g/sqrt(n)",
        "{:.3f}".format(0.78 * g / sqrt(f.size + s.size)), "-")

    head("3.  CENTRAL MEASURES AND SCATTER")
    row("Mode, t_mode", "{:,.1f}".format(st['mode']), "h")
    row("Median, t_med = a(ln2)^(1/g)", "{:,.1f}".format(st['median']), "h")
    row("MTTF = a*Gamma(1+1/g)", "{:,.1f}".format(st['mttf']), "h")
    row("Characteristic life, alpha (F=63.2%)", "{:,.1f}".format(a), "h")
    row("Std deviation, sigma", "{:,.1f}".format(st['sd']), "h")
    row("Coeff. of variation, CoV", "{:.4f}".format(st['cov']), "-")
    row("Skewness, Sk", "{:+.4f}".format(st['skew']), "-")
    row("Gamma(1+1/g)", "{:.6f}".format(st['gamma_fn_1']), "-")
    row("Gamma(1+2/g)", "{:.6f}".format(st['gamma_fn_2']), "-")

    head("4.  B-LIVES (PERCENTILES)")
    row("B1   life  (R = 99%)", "{:,.1f}".format(st['b1']), "h")
    row("B10  life  (R = 90%)", "{:,.1f}".format(st['b10']), "h")
    row("B50  life  (R = 50%)", "{:,.1f}".format(st['b50']), "h")
    row("B10 / MTTF  [PM efficiency index]", "{:.4f}".format(st['b10_over_mttf']), "-")
    row("F(MTTF)  [fraction dead by mean]", "{:.4f}".format(st['f_at_mttf']), "-")

    head("5.  POINT EVALUATIONS")
    print("  {:>12}  {:>10}  {:>10}  {:>12}  {:>12}".format(
        "t (h)", "R(t) -", "F(t) -", "h(t) /1000h", "H(t) -"))
    for tv in eval_times:
        print("  {:>12,.1f}  {:>10.4f}  {:>10.4f}  {:>12.4f}  {:>12.4f}".format(
            tv, float(reliability(tv, g, a)), float(cdf(tv, g, a)),
            float(hazard(tv, g, a)) * 1000.0, float(cum_hazard(tv, g, a))))

    head("6.  INTERPRETATION")
    print("  Shape reading : {}".format(mechanism(g)))
    print("  Hazard trend  : h(t) proportional to t^{:.3f}".format(g - 1.0))
    print("  Scatter       : sigma is {:.0f}% of mean life".format(100 * st['cov']))
    if st['skew'] > 0:
        print("  Skew          : right-skewed, so mode < median < MTTF")
    else:
        print("  Skew          : left-skewed, so MTTF < median < mode")
    print(line + "\n")

    out = {'gamma': g, 'alpha': a, 'r': r, 'r2': r * r,
           'gamma_mle': g_ml, 'alpha_mle': a_ml,
           'failures': f, 'suspensions': s, 'x': x, 'y': y}
    out.update(st)
    return out


# ----------------------------------------------------------------------
# 6.  FOUR-PANEL PLOT
# ----------------------------------------------------------------------
def plot_four(res, title="Weibull Functions", t_max=None, savepath=None,
              show=True):
    """2x2 panel: pdf, cdf, reliability, hazard rate."""
    g, a = res['gamma'], res['alpha']
    f = res['failures']

    if t_max is None:
        t_max = max(float(percentile(0.999, g, a)), float(f.max()) * 1.15)
    t = np.linspace(0.0, t_max, 1200)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("{}\n$\\gamma$ = {:.3f},  $\\alpha$ = {:,.0f} h,  "
                 "$r^2$ = {:.4f},  n = {:d}"
                 .format(title, g, a, res['r2'], f.size + res['suspensions'].size),
                 fontsize=12)

    # --- (1) pdf -------------------------------------------------------
    A = ax[0, 0]
    A.plot(t, pdf(t, g, a) * 1000.0, lw=2, color='#1f4e9c')
    A.fill_between(t, pdf(t, g, a) * 1000.0, alpha=0.15, color='#1f4e9c')
    for xv, lab, c in [(res['mode'], 'mode', '#c0392b'),
                       (res['median'], 'median', '#27ae60'),
                       (res['mttf'], 'MTTF', '#8e44ad')]:
        if xv > 0:
            A.axvline(xv, ls='--', lw=1, color=c, label='{} = {:,.0f} h'.format(lab, xv))
    A.set_title('(1)  Probability density  $f(t)$')
    A.set_xlabel('Age $t$ (h)')
    A.set_ylabel('$f(t)$  (per 1000 h)')
    A.legend(fontsize=8)
    A.grid(alpha=0.3)

    # --- (2) cdf -------------------------------------------------------
    B = ax[0, 1]
    B.plot(t, cdf(t, g, a), lw=2, color='#c0392b', label='$F(t)$ fitted')
    _, Fi = median_ranks(f, res['suspensions'])
    B.plot(np.sort(f), Fi, 'o', ms=5, mfc='white', mec='#c0392b',
           label='median ranks (data)')
    B.axhline(0.632, ls=':', lw=1, color='grey')
    B.axvline(a, ls=':', lw=1, color='grey')
    B.annotate('$\\alpha$ at $F$ = 0.632', xy=(a, 0.632),
               xytext=(a * 0.35, 0.72), fontsize=8,
               arrowprops=dict(arrowstyle='->', lw=0.8))
    B.set_title('(2)  Cumulative distribution  $F(t)$')
    B.set_xlabel('Age $t$ (h)')
    B.set_ylabel('$F(t)$  (dimensionless)')
    B.set_ylim(0, 1.02)
    B.legend(fontsize=8)
    B.grid(alpha=0.3)

    # --- (3) reliability ----------------------------------------------
    C = ax[1, 0]
    C.plot(t, reliability(t, g, a), lw=2, color='#148f5c')
    C.fill_between(t, reliability(t, g, a), alpha=0.15, color='#148f5c')
    C.axhline(0.90, ls='--', lw=1, color='#d35400')
    C.axvline(res['b10'], ls='--', lw=1, color='#d35400',
              label='$B_{{10}}$ = {:,.0f} h'.format(res['b10']))
    C.axhline(0.50, ls=':', lw=1, color='grey')
    C.axvline(res['median'], ls=':', lw=1, color='grey',
              label='$B_{{50}}$ = {:,.0f} h'.format(res['b50']))
    C.set_title('(3)  Reliability  $R(t)$')
    C.set_xlabel('Age $t$ (h)')
    C.set_ylabel('$R(t)$  (dimensionless)')
    C.set_ylim(0, 1.02)
    C.legend(fontsize=8)
    C.grid(alpha=0.3)

    # --- (4) hazard ----------------------------------------------------
    D = ax[1, 1]
    D.plot(t, hazard(t, g, a) * 1000.0, lw=2, color='#d35400')
    D.fill_between(t, hazard(t, g, a) * 1000.0, alpha=0.15, color='#d35400')
    trend = ('increasing - age replacement justified' if g > 1.05 else
             'decreasing - fix install process, do NOT replace on age' if g < 0.95 else
             'constant - memoryless, PM cannot help')
    D.set_title('(4)  Hazard rate  $h(t)$   [{}]'.format(trend), fontsize=10)
    D.set_xlabel('Age $t$ (h)')
    D.set_ylabel('$h(t)$  (failures per 1000 h)')
    D.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if savepath:
        fig.savefig(savepath, dpi=150)
        print("Figure saved: {}".format(savepath))
    if show:
        plt.show()
    return fig


def analyse(failures, suspensions=None, title="Weibull Life-Data Analysis",
            eval_times=None, savepath=None, show=True):
    """Convenience wrapper: table + four-panel plot in one call."""
    res = report(failures, suspensions, title=title, eval_times=eval_times)
    plot_four(res, title=title, savepath=savepath, show=show)
    return res


# ----------------------------------------------------------------------
# 7.  DEMONSTRATION
# ----------------------------------------------------------------------
if __name__ == '__main__':

    # Capstone data: chilled-water pump drive-end bearings, complete data
    bearing = [3500, 4800, 5900, 6600, 7400, 8100, 9000, 10200, 11800, 14000]

    res = analyse(bearing,
                  title="Chilled-Water Pump Drive-End Bearing - Weibull Analysis",
                  eval_times=[4000, 8760, 12000],
                  savepath="weibull_bearing.png",
                  show=True)

    # Censored example: 7 failures, 5 suspensions
    cam_f = [1100, 1850, 2400, 2950, 3600, 4300, 5100]
    cam_s = [2200, 3000, 4000, 5500, 5500]
    analyse(cam_f, cam_s,
            title="Filling-Machine Cam Follower - Censored Weibull Analysis",
            savepath="weibull_cam.png",
            show=True)