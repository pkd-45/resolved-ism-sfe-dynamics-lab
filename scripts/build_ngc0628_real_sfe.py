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
from reproject import reproject_interp
from scipy.signal import fftconvolve

# =============================================================================
# PATHS
# =============================================================================

ROOT = Path.cwd()

DATA = ROOT / "real_data" / "ngc0628"
OUT = ROOT / "outputs" / "ngc0628_real"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

CO_BROAD = (
    DATA
    / "ngc0628_12m+7m+tp_co21_2as_broad_mom0.fits"
)

CO_ERROR = (
    DATA
    / "ngc0628_12m+7m+tp_co21_2as_broad_emom0.fits"
)

CO_STRICT = (
    DATA
    / "ngc0628_12m+7m+tp_co21_2as_strict_mom0.fits"
)

MUSE_MAPS = (
    DATA
    / "NGC0628_MAPS_copt_0.92asec.fits"
)


# =============================================================================
# PHYSICAL / LITERATURE CONSTANTS
# =============================================================================

# PHANGS adopted distance / geometry.
DISTANCE_MPC = 9.84
INCLINATION_DEG = 8.9
POSITION_ANGLE_DEG = 20.7

# Galaxy centre.
RA0_DEG = 24.173855
DEC0_DEG = 15.783643

# Molecular gas conversion.
#
# alpha_CO includes the usual helium correction and is therefore more
# precisely a molecular-gas conversion rather than pure-H2 conversion.
ALPHA_CO = 4.35
R21 = 0.61

# Balmer decrement / attenuation.
BALMER_INTRINSIC = 2.86
K_HA = 2.52
K_HB = 3.66

# Calzetti (2013) / PHANGS convention for a Kroupa IMF.
SFR_HA_FACTOR = 5.5e-42

# Spatial resolutions.
MUSE_PSF_FWHM_ARCSEC = 0.92
TARGET_PSF_FWHM_ARCSEC = 2.00

# Conservative initial measurement cuts.
CO_SNR_MIN = 5.0
HA_SNR_MIN = 10.0
HB_SNR_MIN = 10.0


# =============================================================================
# HELPERS
# =============================================================================


def banner(text):
    print()
    print("=" * 96)
    print(text)
    print("=" * 96)


def finite_stats(array):
    values = np.asarray(
        array,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return {
            "n": 0,
        }

    return {
        "n": int(values.size),
        "min": float(
            np.min(values)
        ),
        "p16": float(
            np.percentile(
                values,
                16,
            )
        ),
        "median": float(
            np.median(values)
        ),
        "p84": float(
            np.percentile(
                values,
                84,
            )
        ),
        "max": float(
            np.max(values)
        ),
    }


def normalized_gaussian_smooth(
    data,
    error,
    sigma_pix,
):
    """
    Gaussian smooth a map with NaN-aware normalization.

    Formal independent-pixel errors are propagated through the kernel.
    MUSE resampling covariance is not represented by this expression, so
    these remain formal errors rather than a complete covariance model.
    """

    data = np.asarray(
        data,
        dtype=float,
    )

    error = np.asarray(
        error,
        dtype=float,
    )

    valid = (
        np.isfinite(data)
        & np.isfinite(error)
        & (error > 0)
    )

    kernel_size = int(
        np.ceil(
            8.0 * sigma_pix
        )
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = Gaussian2DKernel(
        sigma_pix,
        x_size=kernel_size,
        y_size=kernel_size,
    ).array

    kernel = (
        kernel
        / kernel.sum()
    )

    weight = valid.astype(
        float
    )

    numerator = fftconvolve(
        np.where(
            valid,
            data,
            0.0,
        ),
        kernel,
        mode="same",
    )

    denominator = fftconvolve(
        weight,
        kernel,
        mode="same",
    )

    smoothed = np.full(
        data.shape,
        np.nan,
        dtype=float,
    )

    good = denominator > 0.25

    smoothed[good] = (
        numerator[good]
        / denominator[good]
    )

    variance_numerator = fftconvolve(
        np.where(
            valid,
            error**2,
            0.0,
        ),
        kernel**2,
        mode="same",
    )

    smoothed_error = np.full(
        data.shape,
        np.nan,
        dtype=float,
    )

    smoothed_error[good] = (
        np.sqrt(
            variance_numerator[good]
        )
        / denominator[good]
    )

    return (
        smoothed,
        smoothed_error,
        denominator,
    )


def reproject_map(
    array,
    source_wcs,
    target_wcs,
    target_shape,
):
    result, footprint = reproject_interp(
        (
            array,
            source_wcs,
        ),
        target_wcs,
        shape_out=target_shape,
        order="bilinear",
        return_footprint=True,
    )

    result[
        footprint <= 0
    ] = np.nan

    return result


# =============================================================================
# INPUT VALIDATION
# =============================================================================

banner(
    "1. INPUT FILE GATE"
)

required = [
    CO_BROAD,
    CO_ERROR,
    CO_STRICT,
    MUSE_MAPS,
]

for path in required:

    print(
        f"{path.name}: "
        f"{path.exists()}"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

print(
    "INPUT_FILES=PASS"
)


# =============================================================================
# ALMA
# =============================================================================

banner(
    "2. LOAD PHANGS-ALMA"
)

with fits.open(
    CO_BROAD,
    memmap=True,
) as hdul:

    co = np.asarray(
        hdul[0].data,
        dtype=float,
    )

    alma_header = (
        hdul[0].header.copy()
    )

with fits.open(
    CO_ERROR,
    memmap=True,
) as hdul:

    co_error = np.asarray(
        hdul[0].data,
        dtype=float,
    )

with fits.open(
    CO_STRICT,
    memmap=True,
) as hdul:

    co_strict = np.asarray(
        hdul[0].data,
        dtype=float,
    )


if not (
    co.shape
    == co_error.shape
    == co_strict.shape
):
    raise RuntimeError(
        "ALMA shapes disagree"
    )


alma_wcs = WCS(
    alma_header
).celestial

alma_pixel_scale = (
    np.mean(
        proj_plane_pixel_scales(
            alma_wcs
        )
    )
    * 3600.0
)

print(
    "ALMA_SHAPE =",
    co.shape,
)

print(
    "ALMA_PIXEL_SCALE_ARCSEC =",
    alma_pixel_scale,
)

print(
    "ALMA_BEAM_ARCSEC =",
    TARGET_PSF_FWHM_ARCSEC,
)


# =============================================================================
# MUSE
# =============================================================================

banner(
    "3. LOAD PHANGS-MUSE"
)

with fits.open(
    MUSE_MAPS,
    memmap=True,
) as hdul:

    ha_hdu = hdul[
        "HA6562_FLUX"
    ]

    hae_hdu = hdul[
        "HA6562_FLUX_ERR"
    ]

    hb_hdu = hdul[
        "HB4861_FLUX"
    ]

    hbe_hdu = hdul[
        "HB4861_FLUX_ERR"
    ]

    ha = np.asarray(
        ha_hdu.data,
        dtype=float,
    )

    ha_error = np.asarray(
        hae_hdu.data,
        dtype=float,
    )

    hb = np.asarray(
        hb_hdu.data,
        dtype=float,
    )

    hb_error = np.asarray(
        hbe_hdu.data,
        dtype=float,
    )

    muse_wcs = WCS(
        ha_hdu.header
    ).celestial


muse_pixel_scale = (
    np.mean(
        proj_plane_pixel_scales(
            muse_wcs
        )
    )
    * 3600.0
)

muse_spaxel_area_arcsec2 = (
    muse_pixel_scale**2
)

print(
    "MUSE_SHAPE =",
    ha.shape,
)

print(
    "MUSE_PIXEL_SCALE_ARCSEC =",
    muse_pixel_scale,
)

print(
    "MUSE_PSF_FWHM_ARCSEC =",
    MUSE_PSF_FWHM_ARCSEC,
)

print(
    "MUSE_SPAXEL_AREA_ARCSEC2 =",
    muse_spaxel_area_arcsec2,
)

print(
    "MW_FOREGROUND_EXTINCTION="
    "ASSUMED_ALREADY_CORRECTED_BY_PHANGS_MUSE"
)


# =============================================================================
# CONVERT MUSE FLUX/SPAXEL -> SURFACE BRIGHTNESS
# =============================================================================

banner(
    "4. PREPARE MUSE SURFACE-BRIGHTNESS MAPS"
)

# MAPS units:
# 1e-20 erg s^-1 cm^-2 spaxel^-1
#
# Convert first to:
# 1e-20 erg s^-1 cm^-2 arcsec^-2
#
# Reprojection can then safely operate on an intensity-like quantity.

ha_sb = (
    ha
    / muse_spaxel_area_arcsec2
)

ha_error_sb = (
    ha_error
    / muse_spaxel_area_arcsec2
)

hb_sb = (
    hb
    / muse_spaxel_area_arcsec2
)

hb_error_sb = (
    hb_error
    / muse_spaxel_area_arcsec2
)


# =============================================================================
# PSF MATCH MUSE 0.92" -> ALMA 2.0"
# =============================================================================

banner(
    "5. MATCH MUSE PSF TO 2 ARCSEC"
)

if (
    TARGET_PSF_FWHM_ARCSEC
    <= MUSE_PSF_FWHM_ARCSEC
):
    raise RuntimeError(
        "target PSF must exceed MUSE PSF"
    )

kernel_fwhm_arcsec = np.sqrt(
    TARGET_PSF_FWHM_ARCSEC**2
    - MUSE_PSF_FWHM_ARCSEC**2
)

kernel_sigma_pix = (
    kernel_fwhm_arcsec
    / 2.354820045
    / muse_pixel_scale
)

print(
    "MATCHING_KERNEL_FWHM_ARCSEC =",
    kernel_fwhm_arcsec,
)

print(
    "MATCHING_KERNEL_SIGMA_PIX =",
    kernel_sigma_pix,
)


ha_smooth, hae_smooth, ha_cov = (
    normalized_gaussian_smooth(
        ha_sb,
        ha_error_sb,
        kernel_sigma_pix,
    )
)

hb_smooth, hbe_smooth, hb_cov = (
    normalized_gaussian_smooth(
        hb_sb,
        hb_error_sb,
        kernel_sigma_pix,
    )
)

print(
    "MUSE_PSF_MATCH=PASS"
)


# =============================================================================
# REPROJECT MUSE ONTO ALMA GRID
# =============================================================================

banner(
    "6. REPROJECT MUSE -> ALMA WCS"
)

ha_alma = reproject_map(
    ha_smooth,
    muse_wcs,
    alma_wcs,
    co.shape,
)

hae_alma = reproject_map(
    hae_smooth,
    muse_wcs,
    alma_wcs,
    co.shape,
)

hb_alma = reproject_map(
    hb_smooth,
    muse_wcs,
    alma_wcs,
    co.shape,
)

hbe_alma = reproject_map(
    hbe_smooth,
    muse_wcs,
    alma_wcs,
    co.shape,
)

print(
    "MUSE_TO_ALMA_REPROJECTION=PASS"
)


# =============================================================================
# BALMER DECREMENT
# =============================================================================

banner(
    "7. INTERNAL ATTENUATION CORRECTION"
)

ha_snr = (
    ha_alma
    / hae_alma
)

hb_snr = (
    hb_alma
    / hbe_alma
)

balmer_valid = (
    np.isfinite(ha_alma)
    & np.isfinite(hb_alma)
    & np.isfinite(ha_snr)
    & np.isfinite(hb_snr)
    & (ha_alma > 0)
    & (hb_alma > 0)
    & (ha_snr >= HA_SNR_MIN)
    & (hb_snr >= HB_SNR_MIN)
)

balmer_ratio = np.full(
    co.shape,
    np.nan,
)

balmer_ratio[
    balmer_valid
] = (
    ha_alma[
        balmer_valid
    ]
    / hb_alma[
        balmer_valid
    ]
)

a_ha = np.full(
    co.shape,
    np.nan,
)

a_ha[
    balmer_valid
] = 0.0

reddened = (
    balmer_valid
    & (
        balmer_ratio
        > BALMER_INTRINSIC
    )
)

a_ha[
    reddened
] = (
    2.5
    * np.log10(
        balmer_ratio[
            reddened
        ]
        / BALMER_INTRINSIC
    )
    / (
        K_HB / K_HA
        - 1.0
    )
)

ha_corrected = np.full(
    co.shape,
    np.nan,
)

ha_corrected[
    balmer_valid
] = (
    ha_alma[
        balmer_valid
    ]
    * 10.0
    ** (
        0.4
        * a_ha[
            balmer_valid
        ]
    )
)

print(
    "BALMER_VALID_PIXELS =",
    int(
        np.sum(
            balmer_valid
        )
    ),
)

print(
    "BALMER_RATIO_STATS =",
    finite_stats(
        balmer_ratio[
            balmer_valid
        ]
    ),
)

print(
    "A_HA_STATS =",
    finite_stats(
        a_ha[
            balmer_valid
        ]
    ),
)


# =============================================================================
# CO -> MOLECULAR GAS SURFACE DENSITY
# =============================================================================

banner(
    "8. CO -> MOLECULAR GAS"
)

inclination_rad = np.deg2rad(
    INCLINATION_DEG
)

cos_i = np.cos(
    inclination_rad
)

co_snr = (
    co
    / co_error
)

co_valid = (
    np.isfinite(co)
    & np.isfinite(co_error)
    & np.isfinite(co_strict)
    & (co > 0)
    & (co_error > 0)
    & (co_snr >= CO_SNR_MIN)
    & (co_strict > 0)
)

sigma_mol = np.full(
    co.shape,
    np.nan,
)

sigma_mol_error = np.full(
    co.shape,
    np.nan,
)

conversion = (
    ALPHA_CO
    / R21
    * cos_i
)

sigma_mol[
    co_valid
] = (
    conversion
    * co[
        co_valid
    ]
)

sigma_mol_error[
    co_valid
] = (
    conversion
    * co_error[
        co_valid
    ]
)

print(
    "CO_VALID_PIXELS =",
    int(
        np.sum(
            co_valid
        )
    ),
)

print(
    "SIGMA_MOL_STATS =",
    finite_stats(
        sigma_mol[
            co_valid
        ]
    ),
)


# =============================================================================
# Halpha -> SFR SURFACE DENSITY
# =============================================================================

banner(
    "9. Halpha -> SFR SURFACE DENSITY"
)

MPC_TO_CM = (
    3.0856775814913673e24
)

distance_cm = (
    DISTANCE_MPC
    * MPC_TO_CM
)

distance_pc = (
    DISTANCE_MPC
    * 1.0e6
)

pc_per_arcsec = (
    distance_pc
    / 206265.0
)

physical_area_per_arcsec2_kpc2 = (
    pc_per_arcsec**2
    / 1.0e6
)

# ha_corrected is still:
# 1e-20 erg/s/cm^2/arcsec^2

luminosity_surface_density = (
    4.0
    * np.pi
    * distance_cm**2
    * ha_corrected
    * 1.0e-20
)

sigma_sfr = (
    SFR_HA_FACTOR
    * luminosity_surface_density
    / physical_area_per_arcsec2_kpc2
    * cos_i
)

print(
    "PC_PER_ARCSEC =",
    pc_per_arcsec,
)

print(
    "TARGET_RESOLUTION_PC =",
    (
        TARGET_PSF_FWHM_ARCSEC
        * pc_per_arcsec
    ),
)

print(
    "SIGMA_SFR_STATS =",
    finite_stats(
        sigma_sfr[
            balmer_valid
        ]
    ),
)


# =============================================================================
# COMBINED SFE / DEPLETION-TIME MAP
# =============================================================================

banner(
    "10. MATCHED MOLECULAR GAS + SFR"
)

science_valid = (
    co_valid
    & balmer_valid
    & np.isfinite(
        sigma_mol
    )
    & np.isfinite(
        sigma_sfr
    )
    & (sigma_mol > 0)
    & (sigma_sfr > 0)
)

tdep_gyr = np.full(
    co.shape,
    np.nan,
)

# Sigma_mol = Msun pc^-2
# Sigma_SFR = Msun yr^-1 kpc^-2
#
# tdep [Gyr] =
# Sigma_mol * 1e6 / Sigma_SFR / 1e9
# = Sigma_mol / (1000 Sigma_SFR)

tdep_gyr[
    science_valid
] = (
    sigma_mol[
        science_valid
    ]
    / (
        1000.0
        * sigma_sfr[
            science_valid
        ]
    )
)

print(
    "MATCHED_VALID_PIXELS =",
    int(
        np.sum(
            science_valid
        )
    ),
)

print(
    "TDEP_GYR_STATS =",
    finite_stats(
        tdep_gyr[
            science_valid
        ]
    ),
)


# =============================================================================
# WORLD COORDINATES + DEPROJECTED RADIUS
# =============================================================================

banner(
    "11. DEPROJECTED GALACTOCENTRIC RADIUS"
)

yy, xx = np.indices(
    co.shape
)

ra, dec = alma_wcs.pixel_to_world_values(
    xx,
    yy,
)

cos_dec0 = np.cos(
    np.deg2rad(
        DEC0_DEG
    )
)

dx_arcsec = (
    (
        ra
        - RA0_DEG
    )
    * cos_dec0
    * 3600.0
)

dy_arcsec = (
    (
        dec
        - DEC0_DEG
    )
    * 3600.0
)

pa = np.deg2rad(
    POSITION_ANGLE_DEG
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

radius_arcsec = np.sqrt(
    x_major**2
    + (
        y_minor
        / cos_i
    )**2
)

radius_kpc = (
    radius_arcsec
    * pc_per_arcsec
    / 1000.0
)

print(
    "RADIUS_KPC_STATS =",
    finite_stats(
        radius_kpc[
            science_valid
        ]
    ),
)


# =============================================================================
# BEAM-SPACED SAMPLE
# =============================================================================

banner(
    "12. CONSTRUCT APPROXIMATELY BEAM-SPACED SAMPLE"
)

beam_step_pix = round(
        TARGET_PSF_FWHM_ARCSEC
        / alma_pixel_scale
    )

if beam_step_pix < 1:
    raise RuntimeError(
        "invalid beam sampling"
    )

sample_grid = np.zeros(
    co.shape,
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

sample_valid = (
    science_valid
    & sample_grid
)

sy, sx = np.where(
    sample_valid
)

sample_ra, sample_dec = (
    alma_wcs.pixel_to_world_values(
        sx,
        sy,
    )
)

table = pd.DataFrame(
    {
        "x_pix": sx,
        "y_pix": sy,
        "ra_deg": sample_ra,
        "dec_deg": sample_dec,
        "radius_kpc": radius_kpc[
            sample_valid
        ],
        "sigma_mol": sigma_mol[
            sample_valid
        ],
        "sigma_h2": sigma_mol[
            sample_valid
        ],
        "sigma_mol_err": sigma_mol_error[
            sample_valid
        ],
        "sigma_sfr": sigma_sfr[
            sample_valid
        ],
        "tdep_gyr": tdep_gyr[
            sample_valid
        ],
        "co_snr": co_snr[
            sample_valid
        ],
        "ha_snr": ha_snr[
            sample_valid
        ],
        "hb_snr": hb_snr[
            sample_valid
        ],
        "balmer_ratio": balmer_ratio[
            sample_valid
        ],
        "a_ha_mag": a_ha[
            sample_valid
        ],
    }
)

table_path = (
    OUT
    / "ngc0628_beam_spaced_regions.csv"
)

table.to_csv(
    table_path,
    index=False,
)

print(
    "BEAM_STEP_PIX =",
    beam_step_pix,
)

print(
    "BEAM_SAMPLE_N =",
    len(table),
)

print(
    "TABLE =",
    table_path,
)


# =============================================================================
# WRITE MAPS
# =============================================================================

banner(
    "13. WRITE SCIENCE MAP PRODUCTS"
)

products = {
    "SIGMA_MOL": sigma_mol,
    "SIGMA_SFR": sigma_sfr,
    "TDEP_GYR": tdep_gyr,
    "BALMER_RATIO": balmer_ratio,
    "A_HA": a_ha,
}

for name, data in products.items():

    path = (
        OUT
        / f"ngc0628_{name.lower()}.fits"
    )

    fits.PrimaryHDU(
        data=np.asarray(
            data,
            dtype=np.float32,
        ),
        header=alma_header,
    ).writeto(
        path,
        overwrite=True,
    )

    print(
        name,
        "->",
        path,
    )


# =============================================================================
# FIGURE
# =============================================================================

banner(
    "14. MAKE QUICK-LOOK FIGURE"
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        15,
        4.8,
    ),
)

panels = [
    (
        sigma_mol,
        r"$\Sigma_{\rm mol}$",
        "M$_\\odot$ pc$^{-2}$",
    ),
    (
        sigma_sfr,
        r"$\Sigma_{\rm SFR}$",
        "M$_\\odot$ yr$^{-1}$ kpc$^{-2}$",
    ),
    (
        tdep_gyr,
        r"$t_{\rm dep}$",
        "Gyr",
    ),
]

for ax, (
    array,
    title,
    unit,
) in zip(
    axes,
    panels,
):

    view = np.array(
        array,
        copy=True,
    )

    good = (
        science_valid
        & np.isfinite(view)
        & (view > 0)
    )

    log_view = np.full(
        view.shape,
        np.nan,
    )

    log_view[
        good
    ] = np.log10(
        view[
            good
        ]
    )

    values = log_view[
        np.isfinite(
            log_view
        )
    ]

    if values.size:

        vmin, vmax = np.percentile(
            values,
            [
                5,
                95,
            ],
        )

    else:

        vmin, vmax = (
            -1,
            1,
        )

    image = ax.imshow(
        log_view,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "ALMA x pixel"
    )

    ax.set_ylabel(
        "ALMA y pixel"
    )

    cb = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cb.set_label(
        "log10 "
        + unit
    )

fig.suptitle(
    "NGC 0628 — matched PHANGS ALMA + MUSE at 2 arcsec"
)

fig.tight_layout()

figure_path = (
    OUT
    / "ngc0628_real_sfe_maps.png"
)

fig.savefig(
    figure_path,
    dpi=180,
    bbox_inches="tight",
)

plt.close(
    fig
)

print(
    "FIGURE =",
    figure_path,
)


# =============================================================================
# METHOD MANIFEST
# =============================================================================

banner(
    "15. METHOD / PROVENANCE RECORD"
)

summary = {
    "galaxy": "NGC0628",
    "distance_mpc": DISTANCE_MPC,
    "inclination_deg": INCLINATION_DEG,
    "position_angle_deg": POSITION_ANGLE_DEG,
    "center_ra_deg": RA0_DEG,
    "center_dec_deg": DEC0_DEG,
    "muse_input_psf_arcsec": MUSE_PSF_FWHM_ARCSEC,
    "target_psf_arcsec": TARGET_PSF_FWHM_ARCSEC,
    "target_resolution_pc": (
        TARGET_PSF_FWHM_ARCSEC
        * pc_per_arcsec
    ),
    "alpha_co": ALPHA_CO,
    "r21": R21,
    "balmer_intrinsic": BALMER_INTRINSIC,
    "k_ha": K_HA,
    "k_hb": K_HB,
    "sfr_ha_factor": SFR_HA_FACTOR,
    "co_snr_min": CO_SNR_MIN,
    "ha_snr_min": HA_SNR_MIN,
    "hb_snr_min": HB_SNR_MIN,
    "matched_valid_pixels": int(
        np.sum(
            science_valid
        )
    ),
    "beam_sample_n": len(table),
    "sigma_mol": finite_stats(
        sigma_mol[
            science_valid
        ]
    ),
    "sigma_sfr": finite_stats(
        sigma_sfr[
            science_valid
        ]
    ),
    "tdep_gyr": finite_stats(
        tdep_gyr[
            science_valid
        ]
    ),
    "science_guardrails": {
        "raw_alma_reduction_claim": False,
        "hector_internal_data_used": False,
        "ngc0628_dynamics_headline": False,
        "reason": (
            "NGC0628 is nearly face-on; use it to validate "
            "the SFE pipeline, not as the final dynamics test."
        ),
    },
}

summary_path = (
    OUT
    / "ngc0628_real_sfe_summary.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)

print(
    json.dumps(
        summary,
        indent=2,
    )
)

print()
print(
    "NGC0628_REAL_SFE_PIPELINE=PASS"
)

print(
    "DYNAMICS_DEFERRED_FACE_ON=TRUE"
)

print(
    "HECTOR_INTERNAL_DATA_USED=FALSE"
)
