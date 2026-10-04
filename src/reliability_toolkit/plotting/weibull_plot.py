"""Four-panel Weibull figure: pdf, cdf, reliability, hazard."""

import numpy as np
import matplotlib.pyplot as plt

from ..reliability.weibull import (
    pdf, cdf, reliability, hazard, median_ranks, percentile,
)
from ..reliability.weibull_report import report

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