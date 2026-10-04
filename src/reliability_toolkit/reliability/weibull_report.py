"""Formatted parameter table for Weibull life-data analysis."""

import numpy as np
from math import sqrt

from .weibull import (
    _asarr, fit_mrr, fit_mle, life_stats, mechanism,
    reliability, cdf, hazard, cum_hazard,
)

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