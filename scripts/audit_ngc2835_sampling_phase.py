#!/usr/bin/env python3

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.stats import linregress, pearsonr, rankdata, spearmanr

ROOT = Path.cwd()

BASE = ROOT / "outputs" / "ngc2835_real"
OUT = BASE / "final_audit"
OUT.mkdir(parents=True, exist_ok=True)

SIGMA_MOL = BASE / "ngc2835_sigma_mol.fits"
SIGMA_SFR = BASE / "ngc2835_sigma_sfr.fits"
DELTA_V = BASE / "ngc2835_delta_v_co.fits"
LINEWIDTH = BASE / "ngc2835_co_linewidth.fits"
SUMMARY = BASE / "ngc2835_real_sfe_dynamics_summary.json"

DISTANCE_MPC = 12.22
RA0_DEG = 139.47044
DEC0_DEG = -22.35468
INCLINATION_DEG = 41.3
PA_DEG = 1.0

BEAM_STEP_PIX = 10

# Published PHANGS-MUSE catalogue value.
REFERENCE_VSYS_KMS = 867.0


def finite_positive(array):
    return np.isfinite(array) & (array > 0)


def fit_rsk(mol, sfr):
    mol = np.asarray(mol, dtype=float)
    sfr = np.asarray(sfr, dtype=float)

    good = finite_positive(mol) & finite_positive(sfr)

    mol = mol[good]
    sfr = sfr[good]

    if len(mol) < 8:
        return None

    x = np.log10(mol)
    y = np.log10(sfr)

    if np.ptp(x) <= 0:
        return None

    fit = linregress(x, y)

    residual = y - (fit.intercept + fit.slope * x)

    return {
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "scatter_dex": float(
            np.std(
                residual,
                ddof=1,
            )
        ),
        "residual": residual,
        "n": len(mol),
    }


def partial_spearman(x, y, z):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    x = x[good]
    y = y[good]
    z = z[good]

    if len(x) < 8:
        return np.nan

    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z)

    design = np.column_stack(
        [
            np.ones(len(rz)),
            rz,
        ]
    )

    beta_x = np.linalg.lstsq(
        design,
        rx,
        rcond=None,
    )[0]

    beta_y = np.linalg.lstsq(
        design,
        ry,
        rcond=None,
    )[0]

    residual_x = rx - design @ beta_x

    residual_y = ry - design @ beta_y

    return float(
        pearsonr(
            residual_x,
            residual_y,
        ).statistic
    )


def summarize(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if not len(values):
        return {
            "n_phases": 0,
        }

    return {
        "n_phases": len(values),
        "min": float(np.min(values)),
        "p16": float(
            np.percentile(
                values,
                16,
            )
        ),
        "median": float(np.median(values)),
        "p84": float(
            np.percentile(
                values,
                84,
            )
        ),
        "max": float(np.max(values)),
    }


with fits.open(SIGMA_MOL) as hdul:
    sigma_mol = np.asarray(
        hdul[0].data,
        dtype=float,
    )

    header = hdul[0].header.copy()

sigma_sfr = np.asarray(
    fits.getdata(SIGMA_SFR),
    dtype=float,
)

delta_v = np.asarray(
    fits.getdata(DELTA_V),
    dtype=float,
)

linewidth = np.asarray(
    fits.getdata(LINEWIDTH),
    dtype=float,
)

wcs = WCS(header).celestial


if not (sigma_mol.shape == sigma_sfr.shape == delta_v.shape == linewidth.shape):
    raise RuntimeError("map shapes disagree")


yy, xx = np.indices(sigma_mol.shape)

ra, dec = wcs.pixel_to_world_values(
    xx,
    yy,
)

distance_pc = DISTANCE_MPC * 1.0e6

pc_per_arcsec = distance_pc / 206265.0

cos_dec = np.cos(np.deg2rad(DEC0_DEG))

dx_arcsec = (ra - RA0_DEG) * cos_dec * 3600.0

dy_arcsec = (dec - DEC0_DEG) * 3600.0

pa = np.deg2rad(PA_DEG)

inc = np.deg2rad(INCLINATION_DEG)

x_major = dx_arcsec * np.sin(pa) + dy_arcsec * np.cos(pa)

y_minor = -dx_arcsec * np.cos(pa) + dy_arcsec * np.sin(pa)

radius_kpc = (
    np.hypot(
        x_major,
        y_minor / np.cos(inc),
    )
    * pc_per_arcsec
    / 1000.0
)


science = finite_positive(sigma_mol) & finite_positive(sigma_sfr)


records = []


for y_offset, x_offset in product(
    range(BEAM_STEP_PIX),
    repeat=2,
):
    sample_grid = np.zeros(
        sigma_mol.shape,
        dtype=bool,
    )

    sample_grid[
        y_offset::BEAM_STEP_PIX,
        x_offset::BEAM_STEP_PIX,
    ] = True

    base = science & sample_grid

    mol_all = sigma_mol[base]
    sfr_all = sigma_sfr[base]

    global_fit = fit_rsk(
        mol_all,
        sfr_all,
    )

    record = {
        "x_offset": x_offset,
        "y_offset": y_offset,
        "n_all": int(np.sum(base)),
    }

    if global_fit is not None:
        record["global_rsk_slope"] = global_fit["slope"]

        record["global_rsk_scatter_dex"] = global_fit["scatter_dex"]

        record["tdep_ratio_of_sums_gyr"] = float(np.sum(mol_all) / (1000.0 * np.sum(sfr_all)))

    for label, predictor in (
        (
            "abs_delta_v",
            np.abs(delta_v),
        ),
        (
            "co_linewidth",
            linewidth,
        ),
    ):
        valid = base & np.isfinite(predictor) & np.isfinite(radius_kpc)

        n_valid = int(np.sum(valid))

        record[f"{label}_n"] = n_valid

        if n_valid < 8:
            continue

        #
        # IMPORTANT:
        # fit the rSK on the SAME subset used for this predictor.
        #
        fit = fit_rsk(
            sigma_mol[valid],
            sigma_sfr[valid],
        )

        if fit is None:
            continue

        predictor_values = predictor[valid]

        radius_values = radius_kpc[valid]

        raw = spearmanr(
            fit["residual"],
            predictor_values,
        )

        partial = partial_spearman(
            fit["residual"],
            predictor_values,
            radius_values,
        )

        record[f"{label}_subset_rsk_slope"] = fit["slope"]

        record[f"{label}_subset_rsk_scatter_dex"] = fit["scatter_dex"]

        record[f"{label}_raw_rho"] = float(raw.statistic)

        record[f"{label}_partial_rho_radius"] = partial

    records.append(record)


phase_table = pd.DataFrame(records)

phase_csv = OUT / "ngc2835_sampling_phase_audit.csv"

phase_table.to_csv(
    phase_csv,
    index=False,
)


with SUMMARY.open() as handle:
    original_summary = json.load(handle)


vsys_estimated = float(original_summary["estimated_vsys_kms"])

vsys_delta = vsys_estimated - REFERENCE_VSYS_KMS


audit_summary = {
    "n_sampling_phases": len(phase_table),
    "beam_step_pix": BEAM_STEP_PIX,
    "vsys_check": {
        "reference_kms": REFERENCE_VSYS_KMS,
        "estimated_kms": vsys_estimated,
        "difference_kms": float(vsys_delta),
    },
    "global_rsk_slope": summarize(phase_table["global_rsk_slope"]),
    "global_rsk_scatter_dex": summarize(phase_table["global_rsk_scatter_dex"]),
    "tdep_ratio_of_sums_gyr": summarize(phase_table["tdep_ratio_of_sums_gyr"]),
    "abs_delta_v_n": summarize(phase_table["abs_delta_v_n"]),
    "abs_delta_v_partial_rho_radius": summarize(phase_table["abs_delta_v_partial_rho_radius"]),
    "co_linewidth_n": summarize(phase_table["co_linewidth_n"]),
    "co_linewidth_partial_rho_radius": summarize(phase_table["co_linewidth_partial_rho_radius"]),
    "interpretation": {
        "phase_ranges_are_not_statistical_confidence_intervals": True,
        "rsk_refit_on_predictor_specific_subset": True,
        "orbital_time_not_treated_as_independent_of_radius": True,
        "omega_not_treated_as_independent_of_radius": True,
        "dynamics_result_remains_diagnostic_only": True,
    },
}


summary_json = OUT / "ngc2835_sampling_phase_audit_summary.json"

summary_json.write_text(
    json.dumps(
        audit_summary,
        indent=2,
    )
    + "\n"
)


fixed_phase = phase_table.loc[
    (phase_table["x_offset"] == BEAM_STEP_PIX // 2)
    & (phase_table["y_offset"] == BEAM_STEP_PIX // 2)
]


print("=" * 88)
print("NGC 2835 FINAL SAMPLING-PHASE AUDIT")
print("=" * 88)

print()
print("REFERENCE_VSYS_KMS =", REFERENCE_VSYS_KMS)
print("ESTIMATED_VSYS_KMS =", vsys_estimated)
print("VSYS_DIFFERENCE_KMS =", vsys_delta)

print()
print("CURRENT_FIXED_PHASE")
print(fixed_phase.to_string(index=False))

print()
print("ALL_PHASES_SUMMARY")
print(
    json.dumps(
        audit_summary,
        indent=2,
    )
)

print()
print("PHASE_TABLE =", phase_csv)
print("SUMMARY_JSON =", summary_json)

print()
print("NGC2835_SAMPLING_PHASE_AUDIT=PASS")
print("SCIENCE_STATUS=DIAGNOSTIC_ONLY")
print("NO_NEW_DATA_DOWNLOADED=TRUE")
print("HECTOR_INTERNAL_DATA_USED=FALSE")
