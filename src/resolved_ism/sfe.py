from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress, spearmanr


@dataclass(frozen=True)
class LinearFit:
    slope: float
    intercept: float
    scatter_dex: float
    n: int


def _finite_positive_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[good], y[good]


def compute_sfe_quantities(sigma_h2, sigma_sfr):
    h2 = np.asarray(sigma_h2, dtype=float)
    sfr = np.asarray(sigma_sfr, dtype=float)
    if h2.shape != sfr.shape:
        raise ValueError("sigma_h2 and sigma_sfr must have the same shape")
    sfe = np.full(h2.shape, np.nan)
    tdep = np.full(h2.shape, np.nan)
    good = np.isfinite(h2) & np.isfinite(sfr) & (h2 > 0) & (sfr > 0)
    sfe[good] = sfr[good] / h2[good]
    tdep[good] = h2[good] / sfr[good]
    return sfe, tdep


def fit_resolved_sk(sigma_h2, sigma_sfr):
    h2, sfr = _finite_positive_pair(sigma_h2, sigma_sfr)
    if h2.size < 3:
        raise ValueError("need at least three finite positive points")
    x = np.log10(h2)
    y = np.log10(sfr)
    result = linregress(x, y)
    residual = y - (result.intercept + result.slope * x)
    return LinearFit(
        slope=float(result.slope),
        intercept=float(result.intercept),
        scatter_dex=float(np.std(residual, ddof=1)),
        n=x.size,
    )


def resolved_sk_residuals(sigma_h2, sigma_sfr, fit):
    h2 = np.asarray(sigma_h2, dtype=float)
    sfr = np.asarray(sigma_sfr, dtype=float)
    if h2.shape != sfr.shape:
        raise ValueError("sigma_h2 and sigma_sfr must have the same shape")
    residual = np.full(h2.shape, np.nan)
    good = np.isfinite(h2) & np.isfinite(sfr) & (h2 > 0) & (sfr > 0)
    residual[good] = (
        np.log10(sfr[good]) - (fit.intercept + fit.slope * np.log10(h2[good]))
    )
    return residual


def residual_predictor_test(residual, predictor):
    residual = np.asarray(residual, dtype=float)
    predictor = np.asarray(predictor, dtype=float)
    if residual.shape != predictor.shape:
        raise ValueError("residual and predictor must have the same shape")
    good = np.isfinite(residual) & np.isfinite(predictor)
    if good.sum() < 3:
        return {"rho": float("nan"), "pvalue": float("nan"), "n": int(good.sum())}
    result = spearmanr(residual[good], predictor[good])
    return {
        "rho": float(result.statistic),
        "pvalue": float(result.pvalue),
        "n": int(good.sum()),
    }


def bootstrap_sk_fit(sigma_h2, sigma_sfr, n_boot=1000, seed=42):
    h2, sfr = _finite_positive_pair(sigma_h2, sigma_sfr)
    if h2.size < 5:
        raise ValueError("need at least five points")
    x = np.log10(h2)
    y = np.log10(sfr)
    rng = np.random.default_rng(seed)
    slopes = np.empty(n_boot)
    intercepts = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, x.size, size=x.size)
        result = linregress(x[idx], y[idx])
        slopes[i] = result.slope
        intercepts[i] = result.intercept

    def summary(values):
        p16, p50, p84 = np.percentile(values, [16, 50, 84])
        return {
            "median": float(p50),
            "minus": float(p50 - p16),
            "plus": float(p84 - p50),
        }

    return {
        "slope": summary(slopes),
        "intercept": summary(intercepts),
        "n_boot": n_boot,
    }


def bootstrap_spearman(residual, predictor, n_boot=1000, seed=42):
    residual = np.asarray(residual, dtype=float)
    predictor = np.asarray(predictor, dtype=float)
    good = np.isfinite(residual) & np.isfinite(predictor)
    x = residual[good]
    y = predictor[good]
    if x.size < 5:
        raise ValueError("need at least five points")
    rng = np.random.default_rng(seed)
    rho = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, x.size, size=x.size)
        rho[i] = spearmanr(x[idx], y[idx]).statistic
    p16, p50, p84 = np.nanpercentile(rho, [16, 50, 84])
    return {
        "rho_median": float(p50),
        "rho_minus": float(p50 - p16),
        "rho_plus": float(p84 - p50),
        "n": x.size,
        "n_boot": n_boot,
    }
