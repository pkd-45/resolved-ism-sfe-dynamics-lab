#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp
from scipy.signal import fftconvolve

ROOT = Path.cwd()
DATA = ROOT / "real_data" / "ngc2835"
OUT = ROOT / "outputs" / "ngc2835_real"
OUT.mkdir(parents=True, exist_ok=True)

MUSE = DATA / "NGC2835_MAPS_copt_1.15asec.fits"

CO = DATA / "ngc2835_12m+7m+tp_co21_2as_broad_mom0.fits"
CO_ERR = DATA / "ngc2835_12m+7m+tp_co21_2as_broad_emom0.fits"
CO_STRICT = DATA / "ngc2835_12m+7m+tp_co21_2as_strict_mom0.fits"

CO_MOM1 = DATA / "ngc2835_12m+7m+tp_co21_2as_mom1wprior.fits"
CO_EMOM1 = DATA / "ngc2835_12m+7m+tp_co21_2as_emom1wprior.fits"

CO_MOM2 = DATA / "ngc2835_12m+7m+tp_co21_2as_strict_mom2.fits"
CO_EMOM2 = DATA / "ngc2835_12m+7m+tp_co21_2as_strict_emom2.fits"

RC_FILE = DATA / "PHANGS_RC_NGC2835_kpc.ecsv"


DISTANCE_MPC = 12.22
INCLINATION_DEG = 41.3
PA_DEG = 1.0

RA0_DEG = 139.47044
DEC0_DEG = -22.35468

MUSE_PSF_ARCSEC = 1.15
TARGET_PSF_ARCSEC = 2.0

ALPHA_CO = 4.35
R21 = 0.65

BALMER_INTRINSIC = 2.86
K_HA = 2.52
K_HB = 3.66
SFR_HA_FACTOR = 5.5e-42

CO_SNR_MIN = 5.0
HA_SNR_MIN = 10.0
HB_SNR_MIN = 10.0

MOM1_ERROR_MAX = 15.0
MOM2_SNR_MIN = 3.0


def stats(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if not len(x):
        return {"n": 0}

    return {
        "n": len(x),
        "p16": float(np.percentile(x, 16)),
        "median": float(np.median(x)),
        "p84": float(np.percentile(x, 84)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def smooth_map(data, error, sigma_pix):
    data = np.asarray(data, dtype=float)
    error = np.asarray(error, dtype=float)

    valid = np.isfinite(data) & np.isfinite(error) & (error > 0)

    size = int(np.ceil(8 * sigma_pix))
    if size % 2 == 0:
        size += 1

    kernel = Gaussian2DKernel(
        sigma_pix,
        x_size=size,
        y_size=size,
    ).array
    kernel /= kernel.sum()

    weight = valid.astype(float)

    numerator = fftconvolve(
        np.where(valid, data, 0.0),
        kernel,
        mode="same",
    )

    denominator = fftconvolve(
        weight,
        kernel,
        mode="same",
    )

    result = np.full(data.shape, np.nan)
    good = denominator > 0.25
    result[good] = numerator[good] / denominator[good]

    variance = fftconvolve(
        np.where(valid, error**2, 0.0),
        kernel**2,
        mode="same",
    )

    result_error = np.full(data.shape, np.nan)
    result_error[good] = np.sqrt(variance[good]) / denominator[good]

    return result, result_error


def reproj(data, source_wcs, target_wcs, shape):
    result, footprint = reproject_interp(
        (data, source_wcs),
        target_wcs,
        shape_out=shape,
        order="bilinear",
        return_footprint=True,
    )

    result[footprint <= 0] = np.nan
    return result


print("=" * 96)
print("NGC 2835 — REAL PHANGS SFE + DYNAMICS")
print("=" * 96)

required = [
    MUSE,
    CO,
    CO_ERR,
    CO_STRICT,
    CO_MOM1,
    CO_EMOM1,
    CO_MOM2,
    CO_EMOM2,
    RC_FILE,
]

for path in required:
    print(f"{path.name}: {path.exists()}")
    if not path.exists():
        raise FileNotFoundError(path)

print("INPUT_GATE=PASS")


with fits.open(CO) as hdul:
    co = np.asarray(hdul[0].data, dtype=float)
    alma_header = hdul[0].header.copy()
    alma_wcs = WCS(hdul[0].header).celestial

co_err = fits.getdata(CO_ERR).astype(float)
co_strict = fits.getdata(CO_STRICT).astype(float)

mom1 = fits.getdata(CO_MOM1).astype(float)
emom1 = fits.getdata(CO_EMOM1).astype(float)

mom2 = fits.getdata(CO_MOM2).astype(float)
emom2 = fits.getdata(CO_EMOM2).astype(float)

alma_pix_arcsec = np.mean(proj_plane_pixel_scales(alma_wcs)) * 3600.0

print("ALMA_SHAPE =", co.shape)
print("ALMA_PIXEL_SCALE_ARCSEC =", alma_pix_arcsec)


with fits.open(MUSE) as hdul:
    ha = np.asarray(hdul["HA6562_FLUX"].data, dtype=float)
    hae = np.asarray(hdul["HA6562_FLUX_ERR"].data, dtype=float)

    hb = np.asarray(hdul["HB4861_FLUX"].data, dtype=float)
    hbe = np.asarray(hdul["HB4861_FLUX_ERR"].data, dtype=float)

    muse_wcs = WCS(hdul["HA6562_FLUX"].header).celestial

muse_pix_arcsec = np.mean(proj_plane_pixel_scales(muse_wcs)) * 3600.0

print("MUSE_PIXEL_SCALE_ARCSEC =", muse_pix_arcsec)
print("MUSE_INPUT_PSF_ARCSEC =", MUSE_PSF_ARCSEC)


muse_pixel_area_arcsec2 = muse_pix_arcsec**2

ha_sb = ha / muse_pixel_area_arcsec2
hae_sb = hae / muse_pixel_area_arcsec2

hb_sb = hb / muse_pixel_area_arcsec2
hbe_sb = hbe / muse_pixel_area_arcsec2


kernel_fwhm = np.sqrt(TARGET_PSF_ARCSEC**2 - MUSE_PSF_ARCSEC**2)

kernel_sigma_pix = kernel_fwhm / 2.354820045 / muse_pix_arcsec

print("PSF_KERNEL_FWHM_ARCSEC =", kernel_fwhm)
print("PSF_KERNEL_SIGMA_PIX =", kernel_sigma_pix)

ha_s, hae_s = smooth_map(
    ha_sb,
    hae_sb,
    kernel_sigma_pix,
)

hb_s, hbe_s = smooth_map(
    hb_sb,
    hbe_sb,
    kernel_sigma_pix,
)

ha_a = reproj(
    ha_s,
    muse_wcs,
    alma_wcs,
    co.shape,
)

hae_a = reproj(
    hae_s,
    muse_wcs,
    alma_wcs,
    co.shape,
)

hb_a = reproj(
    hb_s,
    muse_wcs,
    alma_wcs,
    co.shape,
)

hbe_a = reproj(
    hbe_s,
    muse_wcs,
    alma_wcs,
    co.shape,
)

print("MUSE_PSF_WCS_MATCH=PASS")


ha_snr = ha_a / hae_a
hb_snr = hb_a / hbe_a

balmer_valid = (
    np.isfinite(ha_a)
    & np.isfinite(hb_a)
    & np.isfinite(ha_snr)
    & np.isfinite(hb_snr)
    & (ha_a > 0)
    & (hb_a > 0)
    & (ha_snr >= HA_SNR_MIN)
    & (hb_snr >= HB_SNR_MIN)
)

balmer = np.full(co.shape, np.nan)
balmer[balmer_valid] = ha_a[balmer_valid] / hb_a[balmer_valid]

a_ha = np.full(co.shape, np.nan)
a_ha[balmer_valid] = 0.0

reddened = balmer_valid & (balmer > BALMER_INTRINSIC)

a_ha[reddened] = 2.5 * np.log10(balmer[reddened] / BALMER_INTRINSIC) / (K_HB / K_HA - 1.0)

ha_corr = np.full(co.shape, np.nan)
ha_corr[balmer_valid] = ha_a[balmer_valid] * 10 ** (0.4 * a_ha[balmer_valid])


inc = np.deg2rad(INCLINATION_DEG)
cos_i = np.cos(inc)

co_snr = co / co_err

co_valid = (
    np.isfinite(co)
    & np.isfinite(co_err)
    & np.isfinite(co_strict)
    & (co > 0)
    & (co_err > 0)
    & (co_snr >= CO_SNR_MIN)
    & (co_strict > 0)
)

sigma_mol = np.full(co.shape, np.nan)

sigma_mol[co_valid] = ALPHA_CO / R21 * co[co_valid] * cos_i


mpc_cm = 3.0856775814913673e24
distance_cm = DISTANCE_MPC * mpc_cm
distance_pc = DISTANCE_MPC * 1e6

pc_per_arcsec = distance_pc / 206265.0

area_kpc2_per_arcsec2 = pc_per_arcsec**2 / 1e6

luminosity_sb = 4 * np.pi * distance_cm**2 * ha_corr * 1e-20

sigma_sfr = SFR_HA_FACTOR * luminosity_sb / area_kpc2_per_arcsec2 * cos_i

science_valid = (
    co_valid
    & balmer_valid
    & np.isfinite(sigma_mol)
    & np.isfinite(sigma_sfr)
    & (sigma_mol > 0)
    & (sigma_sfr > 0)
)

tdep_gyr = np.full(co.shape, np.nan)

tdep_gyr[science_valid] = sigma_mol[science_valid] / (1000.0 * sigma_sfr[science_valid])


yy, xx = np.indices(co.shape)

ra, dec = alma_wcs.pixel_to_world_values(
    xx,
    yy,
)

dx = (ra - RA0_DEG) * np.cos(np.deg2rad(DEC0_DEG)) * 3600.0

dy = (dec - DEC0_DEG) * 3600.0

pa = np.deg2rad(PA_DEG)

x_major = dx * np.sin(pa) + dy * np.cos(pa)

y_minor = -dx * np.cos(pa) + dy * np.sin(pa)

y_plane = y_minor / cos_i

radius_arcsec = np.hypot(
    x_major,
    y_plane,
)

theta = np.arctan2(
    y_plane,
    x_major,
)

radius_kpc = radius_arcsec * pc_per_arcsec / 1000.0


rc = Table.read(RC_FILE)

rc_radius = np.asarray(
    rc["radius"],
    dtype=float,
)

rc_vrot = np.asarray(
    rc["Vrot"],
    dtype=float,
)

rc_min = float(np.min(rc_radius))
rc_max = float(np.max(rc_radius))

print("LANG_RC_MIN_KPC =", rc_min)
print("LANG_RC_MAX_KPC =", rc_max)

rc_measured = np.isfinite(radius_kpc) & (radius_kpc >= rc_min) & (radius_kpc <= rc_max)

vrot = np.full(co.shape, np.nan)

vrot[rc_measured] = np.interp(
    radius_kpc[rc_measured],
    rc_radius,
    rc_vrot,
)

orbital_time_myr = np.full(co.shape, np.nan)

good_vrot = rc_measured & np.isfinite(vrot) & (vrot > 0) & (radius_kpc > 0)

KPC_IN_KM = 3.0856775814913673e16
SEC_PER_MYR = 365.25 * 24 * 3600 * 1e6

orbital_time_myr[good_vrot] = (
    2 * np.pi * radius_kpc[good_vrot] * KPC_IN_KM / vrot[good_vrot] / SEC_PER_MYR
)

omega = np.full(co.shape, np.nan)

omega[good_vrot] = vrot[good_vrot] / radius_kpc[good_vrot]


projected_rotation = np.full(co.shape, np.nan)

projected_rotation[good_vrot] = vrot[good_vrot] * np.sin(inc) * np.cos(theta[good_vrot])

mom1_valid = (
    science_valid
    & rc_measured
    & np.isfinite(mom1)
    & np.isfinite(emom1)
    & (emom1 > 0)
    & (emom1 <= MOM1_ERROR_MAX)
    & np.isfinite(projected_rotation)
)

if np.sum(mom1_valid) < 20:
    raise RuntimeError("Too few valid CO velocity pixels")

vsys = float(np.nanmedian(mom1[mom1_valid] - projected_rotation[mom1_valid]))

model_mom1 = vsys + projected_rotation

delta_v_co = np.full(co.shape, np.nan)

delta_v_co[mom1_valid] = mom1[mom1_valid] - model_mom1[mom1_valid]

abs_delta_v_co = np.abs(delta_v_co)


linewidth_valid = (
    science_valid
    & rc_measured
    & np.isfinite(mom2)
    & np.isfinite(emom2)
    & (mom2 > 0)
    & (emom2 > 0)
    & (mom2 / emom2 >= MOM2_SNR_MIN)
)

co_linewidth = np.full(co.shape, np.nan)
co_linewidth[linewidth_valid] = mom2[linewidth_valid]

vrot_over_linewidth = np.full(co.shape, np.nan)

good_linewidth_ratio = linewidth_valid & np.isfinite(vrot) & (vrot > 0)

vrot_over_linewidth[good_linewidth_ratio] = (
    vrot[good_linewidth_ratio] / co_linewidth[good_linewidth_ratio]
)


beam_step = round(TARGET_PSF_ARCSEC / alma_pix_arcsec)

sample_grid = np.zeros(
    co.shape,
    dtype=bool,
)

offset = beam_step // 2

sample_grid[
    offset::beam_step,
    offset::beam_step,
] = True

sample = science_valid & sample_grid

sy, sx = np.where(sample)

sample_ra, sample_dec = alma_wcs.pixel_to_world_values(
    sx,
    sy,
)

table = pd.DataFrame(
    {
        "x_pix": sx,
        "y_pix": sy,
        "ra_deg": sample_ra,
        "dec_deg": sample_dec,
        "radius_kpc": radius_kpc[sample],
        "sigma_mol": sigma_mol[sample],
        "sigma_h2": sigma_mol[sample],
        "sigma_sfr": sigma_sfr[sample],
        "tdep_gyr": tdep_gyr[sample],
        "co_snr": co_snr[sample],
        "vrot_kms": vrot[sample],
        "orbital_time_myr": orbital_time_myr[sample],
        "omega_kms_kpc": omega[sample],
        "co_linewidth_kms": co_linewidth[sample],
        "vrot_over_co_linewidth": vrot_over_linewidth[sample],
        "delta_v_co_kms": delta_v_co[sample],
        "abs_delta_v_co_kms": abs_delta_v_co[sample],
        "rc_measured": rc_measured[sample].astype(int),
    }
)

table_path = OUT / "ngc2835_beam_spaced_regions.csv"

table.to_csv(
    table_path,
    index=False,
)


for name, array in {
    "sigma_mol": sigma_mol,
    "sigma_sfr": sigma_sfr,
    "tdep_gyr": tdep_gyr,
    "co_linewidth": co_linewidth,
    "delta_v_co": delta_v_co,
}.items():
    fits.PrimaryHDU(
        data=np.asarray(
            array,
            dtype=np.float32,
        ),
        header=alma_header,
    ).writeto(
        OUT / f"ngc2835_{name}.fits",
        overwrite=True,
    )


fig, axes = plt.subplots(
    1,
    4,
    figsize=(18, 4.6),
)

panels = [
    (sigma_mol, "Sigma_mol"),
    (sigma_sfr, "Sigma_SFR"),
    (tdep_gyr, "t_dep [Gyr]"),
    (abs_delta_v_co, "|CO velocity residual|"),
]

for ax, (array, title) in zip(
    axes,
    panels,
):
    image = np.array(
        array,
        copy=True,
    )

    good = np.isfinite(image) & (image > 0)

    view = np.full(
        image.shape,
        np.nan,
    )

    view[good] = np.log10(image[good])

    vals = view[np.isfinite(view)]

    if vals.size:
        vmin, vmax = np.percentile(
            vals,
            [5, 95],
        )
    else:
        vmin, vmax = -1, 1

    im = ax.imshow(
        view,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

fig.suptitle("NGC 2835 — PHANGS ALMA + MUSE matched to 2 arcsec")

fig.tight_layout()

fig.savefig(
    OUT / "ngc2835_sfe_dynamics_maps.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)


summary = {
    "galaxy": "NGC2835",
    "distance_mpc": DISTANCE_MPC,
    "inclination_deg": INCLINATION_DEG,
    "pa_deg": PA_DEG,
    "center_ra_deg": RA0_DEG,
    "center_dec_deg": DEC0_DEG,
    "target_psf_arcsec": TARGET_PSF_ARCSEC,
    "target_resolution_pc": (TARGET_PSF_ARCSEC * pc_per_arcsec),
    "alpha_co": ALPHA_CO,
    "r21": R21,
    "lang_rc_min_kpc": rc_min,
    "lang_rc_max_kpc": rc_max,
    "estimated_vsys_kms": vsys,
    "matched_pixels": int(np.sum(science_valid)),
    "beam_sample_n": len(table),
    "beam_samples_in_rc_range": int(np.sum(table["rc_measured"] == 1)),
    "sigma_mol": stats(sigma_mol[science_valid]),
    "sigma_sfr": stats(sigma_sfr[science_valid]),
    "tdep_gyr": stats(tdep_gyr[science_valid]),
    "co_linewidth_kms": stats(co_linewidth[linewidth_valid]),
    "abs_delta_v_co_kms": stats(abs_delta_v_co[mom1_valid]),
    "guardrails": {
        "diagnostic_only": True,
        "constant_alpha_co_baseline": True,
        "metallicity_dependent_alpha_co_pending": True,
        "dig_bpt_mask_pending": True,
        "co_linewidth_beam_smearing_correction_pending": True,
        "stellar_sigma_not_used_yet": True,
        "raw_alma_reduction_claim": False,
        "hector_internal_data_used": False,
    },
}

summary_path = OUT / "ngc2835_real_sfe_dynamics_summary.json"

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)

print()
print("=" * 96)
print("SUMMARY")
print("=" * 96)

print(
    json.dumps(
        summary,
        indent=2,
    )
)

print()
print("TABLE =", table_path)
print("NGC2835_REAL_SFE_DYNAMICS=PASS")
print("INTERPRETATION_STATUS=DIAGNOSTIC_ONLY")
print("HECTOR_INTERNAL_DATA_USED=FALSE")
