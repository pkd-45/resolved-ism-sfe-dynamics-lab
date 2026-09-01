#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.signal import fftconvolve
from scipy.stats import linregress, spearmanr

ROOT = Path.cwd()

MAP_DIR = ROOT / "outputs" / "ngc0628_real"
OUT = MAP_DIR / "scale_ladder"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

MOL_PATH = (
    MAP_DIR
    / "ngc0628_sigma_mol.fits"
)

SFR_PATH = (
    MAP_DIR
    / "ngc0628_sigma_sfr.fits"
)

BASE_SUMMARY = (
    MAP_DIR
    / "ngc0628_real_sfe_summary.json"
)

TARGET_SCALES_PC = [
    100.0,
    250.0,
    500.0,
    1000.0,
]


def finite_positive(array):
    arr = np.asarray(
        array,
        dtype=float,
    )

    return (
        np.isfinite(arr)
        & (arr > 0)
    )


def nan_weighted_smooth(
    array,
    kernel_sigma_pix,
):
    """
    NaN-aware Gaussian smoothing.

    This operates on the existing science-ready surface-density maps.
    It is appropriate as a scale-dependence diagnostic, but it does not
    restore emission that was excluded by the original detection masks.
    """

    arr = np.asarray(
        array,
        dtype=float,
    )

    valid = np.isfinite(
        arr
    )

    if kernel_sigma_pix <= 1e-6:
        return np.array(
            arr,
            copy=True,
        )

    size = int(
        np.ceil(
            8.0
            * kernel_sigma_pix
        )
    )

    if size % 2 == 0:
        size += 1

    kernel = Gaussian2DKernel(
        kernel_sigma_pix,
        x_size=size,
        y_size=size,
    ).array

    kernel /= kernel.sum()

    numerator = fftconvolve(
        np.where(
            valid,
            arr,
            0.0,
        ),
        kernel,
        mode="same",
    )

    denominator = fftconvolve(
        valid.astype(float),
        kernel,
        mode="same",
    )

    result = np.full(
        arr.shape,
        np.nan,
        dtype=float,
    )

    #
    # Require meaningful contributing coverage.
    #
    good = (
        denominator
        >= 0.50
    )

    result[
        good
    ] = (
        numerator[
            good
        ]
        / denominator[
            good
        ]
    )

    return result


def fit_rsk(
    sigma_mol,
    sigma_sfr,
):
    good = (
        finite_positive(
            sigma_mol
        )
        & finite_positive(
            sigma_sfr
        )
    )

    x = np.log10(
        sigma_mol[
            good
        ]
    )

    y = np.log10(
        sigma_sfr[
            good
        ]
    )

    if len(x) < 10:
        return None

    fit = linregress(
        x,
        y,
    )

    residual = (
        y
        - (
            fit.intercept
            + fit.slope
            * x
        )
    )

    return {
        "slope": float(
            fit.slope
        ),
        "intercept": float(
            fit.intercept
        ),
        "scatter_dex": float(
            np.std(
                residual,
                ddof=1,
            )
        ),
        "n": len(x),
        "residual": residual,
        "good": good,
    }


with fits.open(
    MOL_PATH
) as hdul:

    sigma_mol_base = np.asarray(
        hdul[0].data,
        dtype=float,
    )

    header = (
        hdul[0].header.copy()
    )


with fits.open(
    SFR_PATH
) as hdul:

    sigma_sfr_base = np.asarray(
        hdul[0].data,
        dtype=float,
    )


if (
    sigma_mol_base.shape
    != sigma_sfr_base.shape
):
    raise RuntimeError(
        "map shapes disagree"
    )


wcs = WCS(
    header
).celestial


pixel_scale_arcsec = float(
    np.mean(
        proj_plane_pixel_scales(
            wcs
        )
    )
    * 3600.0
)


base = json.loads(
    BASE_SUMMARY.read_text()
)


pc_per_arcsec = (
    base[
        "target_resolution_pc"
    ]
    / base[
        "target_psf_arcsec"
    ]
)


base_fwhm_arcsec = float(
    base[
        "target_psf_arcsec"
    ]
)


print(
    "PIXEL_SCALE_ARCSEC =",
    pixel_scale_arcsec,
)

print(
    "PC_PER_ARCSEC =",
    pc_per_arcsec,
)

print(
    "BASE_FWHM_ARCSEC =",
    base_fwhm_arcsec,
)


#
# Radius map from the same geometry used in the real-data pipeline.
#

yy, xx = np.indices(
    sigma_mol_base.shape
)

ra, dec = (
    wcs.pixel_to_world_values(
        xx,
        yy,
    )
)

ra0 = float(
    base[
        "center_ra_deg"
    ]
)

dec0 = float(
    base[
        "center_dec_deg"
    ]
)

pa = np.deg2rad(
    float(
        base[
            "position_angle_deg"
        ]
    )
)

inclination = np.deg2rad(
    float(
        base[
            "inclination_deg"
        ]
    )
)

dx_arcsec = (
    (
        ra
        - ra0
    )
    * np.cos(
        np.deg2rad(
            dec0
        )
    )
    * 3600.0
)

dy_arcsec = (
    (
        dec
        - dec0
    )
    * 3600.0
)

x_major = (
    dx_arcsec
    * np.sin(pa)
    + dy_arcsec
    * np.cos(pa)
)

y_minor = (
    -dx_arcsec
    * np.cos(pa)
    + dy_arcsec
    * np.sin(pa)
)

radius_kpc = (
    np.sqrt(
        x_major**2
        + (
            y_minor
            / np.cos(
                inclination
            )
        )**2
    )
    * pc_per_arcsec
    / 1000.0
)


records = []
all_samples = []


for target_pc in TARGET_SCALES_PC:

    print()
    print(
        "=" * 88
    )

    print(
        f"TARGET SCALE = {target_pc:.1f} pc"
    )

    print(
        "=" * 88
    )


    target_fwhm_arcsec = (
        target_pc
        / pc_per_arcsec
    )


    if (
        target_fwhm_arcsec
        < base_fwhm_arcsec
        - 1e-6
    ):
        raise RuntimeError(
            "target scale finer than base map"
        )


    extra_fwhm_arcsec = np.sqrt(
        max(
            0.0,
            target_fwhm_arcsec**2
            - base_fwhm_arcsec**2,
        )
    )


    extra_sigma_pix = (
        extra_fwhm_arcsec
        / 2.354820045
        / pixel_scale_arcsec
    )


    print(
        "TARGET_FWHM_ARCSEC =",
        target_fwhm_arcsec,
    )

    print(
        "EXTRA_KERNEL_FWHM_ARCSEC =",
        extra_fwhm_arcsec,
    )

    print(
        "EXTRA_KERNEL_SIGMA_PIX =",
        extra_sigma_pix,
    )


    sigma_mol = (
        nan_weighted_smooth(
            sigma_mol_base,
            extra_sigma_pix,
        )
    )

    sigma_sfr = (
        nan_weighted_smooth(
            sigma_sfr_base,
            extra_sigma_pix,
        )
    )


    #
    # Sample at approximately one resolution element.
    #

    beam_step_pix = max(
        1,
        round(
            target_fwhm_arcsec
            / pixel_scale_arcsec
        ),
    )


    sample_grid = np.zeros(
        sigma_mol.shape,
        dtype=bool,
    )

    offset = (
        beam_step_pix
        // 2
    )

    sample_grid[
        offset::beam_step_pix,
        offset::beam_step_pix,
    ] = True


    valid = (
        sample_grid
        & finite_positive(
            sigma_mol
        )
        & finite_positive(
            sigma_sfr
        )
    )


    mol = sigma_mol[
        valid
    ]

    sfr = sigma_sfr[
        valid
    ]

    radius = radius_kpc[
        valid
    ]


    tdep = (
        mol
        / (
            1000.0
            * sfr
        )
    )


    q16, median, q84 = np.percentile(
        tdep,
        [
            16,
            50,
            84,
        ],
    )


    ratio_sums = (
        np.sum(
            mol
        )
        / (
            1000.0
            * np.sum(
                sfr
            )
        )
    )


    fit = fit_rsk(
        mol,
        sfr,
    )


    if fit is None:

        fit_status = (
            "INSUFFICIENT_INDEPENDENT_SAMPLES"
        )

        residual = np.full(
            len(mol),
            np.nan,
            dtype=float,
        )

        rsk_slope = np.nan
        rsk_scatter = np.nan
        radius_rho = np.nan
        radius_p = np.nan

        print(
            "RSK_FIT_SKIPPED =",
            fit_status,
        )

        print(
            "N_INDEPENDENT_SAMPLES =",
            len(mol),
        )

    else:

        fit_status = "PASS"

        residual = fit[
            "residual"
        ]

        rsk_slope = fit[
            "slope"
        ]

        rsk_scatter = fit[
            "scatter_dex"
        ]

        rho_result = spearmanr(
            residual,
            radius,
        )

        radius_rho = (
            rho_result.statistic
        )

        radius_p = (
            rho_result.pvalue
        )


    record = {
        "target_scale_pc": float(
            target_pc
        ),
        "target_fwhm_arcsec": float(
            target_fwhm_arcsec
        ),
        "beam_step_pix": int(
            beam_step_pix
        ),
        "n_samples": len(mol),
        "tdep_p16_gyr": float(
            q16
        ),
        "tdep_median_gyr": float(
            median
        ),
        "tdep_p84_gyr": float(
            q84
        ),
        "tdep_ratio_of_sums_gyr": float(
            ratio_sums
        ),
        "rsk_slope": float(
            rsk_slope
        ),
        "rsk_scatter_dex": float(
            rsk_scatter
        ),
        "radius_residual_rho": float(
            radius_rho
        ),
        "radius_residual_p": float(
            radius_p
        ),
        "rsk_status": fit_status,
    }


    records.append(
        record
    )


    samples = pd.DataFrame(
        {
            "scale_pc": target_pc,
            "sigma_mol": mol,
            "sigma_sfr": sfr,
            "tdep_gyr": tdep,
            "radius_kpc": radius,
        }
    )

    all_samples.append(
        samples
    )


    print(
        json.dumps(
            record,
            indent=2,
        )
    )


summary = pd.DataFrame(
    records
)

summary_path = (
    OUT
    / "ngc0628_scale_ladder_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


samples_path = (
    OUT
    / "ngc0628_scale_ladder_samples.csv"
)

pd.concat(
    all_samples,
    ignore_index=True,
).to_csv(
    samples_path,
    index=False,
)


json_path = (
    OUT
    / "ngc0628_scale_ladder_summary.json"
)

json_path.write_text(
    json.dumps(
        records,
        indent=2,
    )
    + "\n"
)


#
# Figure 1: depletion-time distribution versus physical scale.
#

fig, ax = plt.subplots(
    figsize=(
        6.4,
        4.7,
    )
)

ax.errorbar(
    summary[
        "target_scale_pc"
    ],
    summary[
        "tdep_median_gyr"
    ],
    yerr=[
        (
            summary[
                "tdep_median_gyr"
            ]
            - summary[
                "tdep_p16_gyr"
            ]
        ),
        (
            summary[
                "tdep_p84_gyr"
            ]
            - summary[
                "tdep_median_gyr"
            ]
        ),
    ],
    marker="o",
    capsize=4,
    label="Median local $t_{dep}$",
)

ax.plot(
    summary[
        "target_scale_pc"
    ],
    summary[
        "tdep_ratio_of_sums_gyr"
    ],
    marker="s",
    label="Ratio of summed gas/SFR",
)

ax.set_xscale(
    "log"
)

ax.set_xlabel(
    "Physical resolution [pc]"
)

ax.set_ylabel(
    "Molecular depletion time [Gyr]"
)

ax.set_title(
    "NGC 0628: depletion time versus physical scale"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUT
    / "ngc0628_tdep_vs_scale.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(
    fig
)


#
# Figure 2: rSK scatter and radial residual correlation.
#

fig, ax = plt.subplots(
    figsize=(
        6.4,
        4.7,
    )
)

ax.plot(
    summary[
        "target_scale_pc"
    ],
    summary[
        "rsk_scatter_dex"
    ],
    marker="o",
    label="rSK scatter",
)

ax.set_xscale(
    "log"
)

ax.set_xlabel(
    "Physical resolution [pc]"
)

ax.set_ylabel(
    "rSK scatter [dex]"
)

ax.set_title(
    "NGC 0628: resolved-SF scatter versus scale"
)

fig.tight_layout()

fig.savefig(
    OUT
    / "ngc0628_rsk_scatter_vs_scale.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(
    fig
)


print()
print(
    "=" * 88
)

print(
    "SCALE LADDER SUMMARY"
)

print(
    "=" * 88
)

print(
    summary.to_string(
        index=False
    )
)

print()
print(
    "SUMMARY_CSV =",
    summary_path,
)

print(
    "SAMPLES_CSV =",
    samples_path,
)

print()
print(
    "NGC0628_SCALE_LADDER=PASS"
)

print(
    "INTERPRETATION_STATUS=DIAGNOSTIC_ONLY"
)

print(
    "DYNAMICS_DEFERRED_FACE_ON=TRUE"
)
