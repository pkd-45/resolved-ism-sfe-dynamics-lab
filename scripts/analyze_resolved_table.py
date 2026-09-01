#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from resolved_ism.dynamics import orbital_time_myr, v_over_sigma
from resolved_ism.sfe import (
    bootstrap_sk_fit,
    bootstrap_spearman,
    compute_sfe_quantities,
    fit_resolved_sk,
    residual_predictor_test,
    resolved_sk_residuals,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--sigma-h2-col", default="sigma_h2")
    parser.add_argument("--sigma-sfr-col", default="sigma_sfr")
    parser.add_argument("--predictors", nargs="*", default=[])
    parser.add_argument("--radius-col")
    parser.add_argument("--vrot-col")
    parser.add_argument("--sigma-col")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=300)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    h2 = pd.to_numeric(df[args.sigma_h2_col], errors="coerce").to_numpy()
    sfr = pd.to_numeric(df[args.sigma_sfr_col], errors="coerce").to_numpy()

    fit = fit_resolved_sk(h2, sfr)
    residual = resolved_sk_residuals(h2, sfr, fit)
    sfe, tdep = compute_sfe_quantities(h2, sfr)

    df["sfe"] = sfe
    df["tdep"] = tdep
    df["delta_rsk"] = residual

    predictors = list(args.predictors)
    if args.radius_col and args.vrot_col:
        radius = pd.to_numeric(df[args.radius_col], errors="coerce").to_numpy()
        vrot = pd.to_numeric(df[args.vrot_col], errors="coerce").to_numpy()
        df["orbital_time_myr"] = orbital_time_myr(radius, vrot)
        predictors.append("orbital_time_myr")

    if args.vrot_col and args.sigma_col:
        vrot = pd.to_numeric(df[args.vrot_col], errors="coerce").to_numpy()
        sigma = pd.to_numeric(df[args.sigma_col], errors="coerce").to_numpy()
        df["v_over_sigma"] = v_over_sigma(vrot, sigma)
        predictors.append("v_over_sigma")

    results = {
        "rsk": {
            "slope": fit.slope,
            "intercept": fit.intercept,
            "scatter_dex": fit.scatter_dex,
            "n": fit.n,
            "bootstrap": bootstrap_sk_fit(h2, sfr, n_boot=args.n_boot),
        },
        "predictors": {},
    }

    for predictor in dict.fromkeys(predictors):
        if predictor not in df.columns:
            continue
        values = pd.to_numeric(df[predictor], errors="coerce").to_numpy()
        test = residual_predictor_test(residual, values)
        if test["n"] >= 5:
            test["bootstrap"] = bootstrap_spearman(
                residual,
                values,
                n_boot=args.n_boot,
            )
        results["predictors"][predictor] = test

    df.to_csv(args.outdir / "resolved_analysis_table.csv", index=False)
    (args.outdir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")

    good = np.isfinite(h2) & np.isfinite(sfr) & (h2 > 0) & (sfr > 0)
    x = np.log10(h2[good])
    y = np.log10(sfr[good])
    grid = np.linspace(np.nanmin(x), np.nanmax(x), 200)

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.scatter(x, y, s=12, alpha=0.35)
    ax.plot(grid, fit.intercept + fit.slope * grid)
    ax.set_xlabel("log10 Sigma_H2")
    ax.set_ylabel("log10 Sigma_SFR")
    ax.set_title("Resolved molecular Schmidt-Kennicutt relation")
    fig.tight_layout()
    fig.savefig(args.outdir / "resolved_sk.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(results, indent=2))
    print("RESOLVED_ISM_ANALYSIS=PASS")


if __name__ == "__main__":
    main()
