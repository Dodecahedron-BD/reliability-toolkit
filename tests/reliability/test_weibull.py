"""Tests for the two-parameter Weibull module."""

import numpy as np
import pytest
from math import log

from reliability_toolkit.reliability import (
    median_ranks, fit_mrr, fit_mle, cdf, hazard, life_stats,
)

G_TRUE, A_TRUE = 2.5, 5000.0


def _synthetic(n=20, g=G_TRUE, a=A_TRUE):
    """Times lying exactly on the Weibull line at Bernard ranks."""
    i = np.arange(1, n + 1)
    F = (i - 0.3) / (n + 0.4)
    return a * (-np.log(1.0 - F)) ** (1.0 / g)


def test_mrr_recovers_parameters_exactly():
    g, a, r, _, _ = fit_mrr(_synthetic())
    assert g == pytest.approx(G_TRUE, rel=1e-8)
    assert a == pytest.approx(A_TRUE, rel=1e-8)
    assert r == pytest.approx(1.0, abs=1e-10)


def test_mle_recovers_parameters_approximately():
    g, a = fit_mle(_synthetic(n=200))
    assert g == pytest.approx(G_TRUE, rel=0.10)
    assert a == pytest.approx(A_TRUE, rel=0.10)


def test_exponential_special_case():
    st = life_stats(1.0, 1000.0)
    assert st["mttf"] == pytest.approx(1000.0)
    assert st["median"] == pytest.approx(1000.0 * log(2.0))
    h = hazard(np.array([10.0, 5000.0]), 1.0, 1000.0)
    assert h[0] == pytest.approx(h[1])


def test_alpha_is_the_632_percent_point():
    assert float(cdf(777.0, 1.8, 777.0)) == pytest.approx(1.0 - np.exp(-1.0))


def test_empty_suspensions_matches_bernard():
    f = [100.0, 200.0, 300.0, 400.0]
    _, F1 = median_ranks(f)
    _, F2 = median_ranks(f, [])
    assert np.allclose(F1, F2)


def test_suspensions_lower_the_ranks():
    f = [100.0, 200.0, 300.0]
    _, F_plain = median_ranks(f)
    _, F_cens = median_ranks(f, [150.0, 250.0])
    assert np.all(F_cens < F_plain)


def test_hazard_increases_when_shape_exceeds_one():
    t = np.array([100.0, 500.0, 2000.0])
    h = hazard(t, 2.5, 1000.0)
    assert np.all(np.diff(h) > 0)