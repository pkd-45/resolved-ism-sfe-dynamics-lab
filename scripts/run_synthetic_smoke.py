#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n = 1200
    log_h2 = rng.normal(1.35, 0.45, n)
    radius = rng.uniform(0.5, 9.0, n)
    vrot = 205.0 * (1.0 - np.exp(-radius / 1.8))
    sigma_gas = rng.lognormal(np.log(18.0), 0.25, n)
    vos = vrot / sigma_gas
    dyn = 0.14 * (np.log10(vos) - np.mean(np.log10(vos)))
    log_sfr = -2.65 + 1.02 * log_h2 + dyn + rng.normal(0.0, 0.20, n)

    df = pd.DataFrame(
        {
            "sigma_h2": 10**log_h2,
            "sigma_sfr": 10**log_sfr,
            "radius_kpc": radius,
            "vrot": vrot,
            "sigma_gas": sigma_gas,
        }
    )
    path = args.outdir / "synthetic_resolved_regions.csv"
    df.to_csv(path, index=False)
    print(f"SYNTHETIC_TABLE={path}")
    print("SYNTHETIC_GENERATION=PASS")


if __name__ == "__main__":
    main()
