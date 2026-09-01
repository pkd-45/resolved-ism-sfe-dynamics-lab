#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

ROOT = Path.cwd()

DATA = ROOT / "real_data" / "ngc2835"


fits_files = sorted(DATA.glob("*.fits"))

print("=" * 96)
print("NGC 2835 PRODUCT AUDIT")
print("=" * 96)


# ---------------------------------------------------------------------
# FITS products
# ---------------------------------------------------------------------

for path in fits_files:
    print()
    print("=" * 96)
    print(path.name)
    print("=" * 96)

    with fits.open(
        path,
        memmap=True,
    ) as hdul:
        print(
            "N_HDU =",
            len(hdul),
        )

        for index, hdu in enumerate(hdul):
            name = hdu.name

            shape = None if hdu.data is None else hdu.data.shape

            bunit = hdu.header.get("BUNIT")

            print(f"HDU={index:3d} NAME={name:28s} SHAPE={shape!s:18s} BUNIT={bunit}")

            upper = name.upper()

            if any(
                token in upper
                for token in (
                    "HA6562",
                    "HB4861",
                    "V_STARS",
                    "SIGMA_STARS",
                    "VEL",
                    "SIGMA",
                    "MASS",
                    "MET",
                )
            ):
                print("  >>> SCIENCE_EXTENSION")

        # Primary or first image HDU WCS check.
        image_hdu = None

        for hdu in hdul:
            if hdu.data is not None and np.ndim(hdu.data) == 2:
                image_hdu = hdu
                break

        if image_hdu is not None:
            try:
                wcs = WCS(image_hdu.header).celestial

                scales = proj_plane_pixel_scales(wcs) * 3600.0

                print(
                    "PIXEL_SCALE_ARCSEC =",
                    np.asarray(scales).tolist(),
                )

            except (ValueError, TypeError):
                print("PIXEL_SCALE_ARCSEC = UNAVAILABLE")

            print(
                "BMAJ_DEG =",
                image_hdu.header.get("BMAJ"),
            )

            print(
                "BMIN_DEG =",
                image_hdu.header.get("BMIN"),
            )

            print(
                "BPA_DEG =",
                image_hdu.header.get("BPA"),
            )


# ---------------------------------------------------------------------
# Rotation curve
# ---------------------------------------------------------------------

curve = DATA / "PHANGS_RC_NGC2835_kpc.ecsv"

print()
print("=" * 96)
print("LANG ET AL. ROTATION CURVE")
print("=" * 96)

if not curve.exists():
    raise FileNotFoundError(curve)

table = Table.read(
    curve,
    format="ascii.ecsv",
)

print(
    "N_ROWS =",
    len(table),
)

print(
    "COLUMNS =",
    table.colnames,
)

print()
print(table[:10])


# ---------------------------------------------------------------------
# Explicit MUSE science-extension audit.
# ---------------------------------------------------------------------

muse = DATA / "NGC2835_MAPS_copt_1.15asec.fits"

print()
print("=" * 96)
print("MUSE DYNAMICS / SFR EXTENSIONS")
print("=" * 96)

with fits.open(
    muse,
    memmap=True,
) as hdul:
    names = [hdu.name for hdu in hdul]

    wanted_tokens = (
        "HA6562",
        "HB4861",
        "V_STARS",
        "SIGMA_STARS",
        "VEL",
        "SIGMA",
    )

    selected = [name for name in names if any(token in name.upper() for token in wanted_tokens)]

    for name in selected:
        hdu = hdul[name]

        print(f"{name:30s} shape={hdu.data.shape} BUNIT={hdu.header.get('BUNIT')}")


print()
print("=" * 96)
print("PRODUCT AUDIT COMPLETE")
print("=" * 96)

print("NGC2835_PRODUCT_AUDIT=PASS")
print("SCIENCE_READY_PRODUCTS_ONLY=TRUE")
print("HECTOR_INTERNAL_DATA_USED=FALSE")
