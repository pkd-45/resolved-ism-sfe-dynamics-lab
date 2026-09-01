import numpy as np

from resolved_ism.sfe import (
    compute_sfe_quantities,
    fit_resolved_sk,
    residual_predictor_test,
    resolved_sk_residuals,
)


def test_sfe_and_tdep_are_reciprocal():
    h2 = np.array([10.0, 20.0])
    sfr = np.array([2.0, 4.0])
    sfe, tdep = compute_sfe_quantities(h2, sfr)
    assert np.allclose(sfe * tdep, 1.0)


def test_exact_resolved_sk_fit():
    h2 = np.logspace(0, 2, 100)
    sfr = 0.01 * h2**1.2
    fit = fit_resolved_sk(h2, sfr)
    assert np.isclose(fit.slope, 1.2, atol=1e-10)
    assert np.isclose(fit.intercept, -2.0, atol=1e-10)


def test_residuals_zero_for_exact_relation():
    h2 = np.logspace(0, 2, 50)
    sfr = 0.1 * h2
    fit = fit_resolved_sk(h2, sfr)
    residual = resolved_sk_residuals(h2, sfr, fit)
    assert np.nanmax(np.abs(residual)) < 1e-10


def test_residual_predictor_detects_monotonic_relation():
    residual = np.arange(20.0)
    predictor = 3.0 * residual + 2.0
    result = residual_predictor_test(residual, predictor)
    assert result["rho"] > 0.999
