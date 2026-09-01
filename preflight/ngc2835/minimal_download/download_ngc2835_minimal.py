#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import vos

ROOT = Path.cwd()

OUTDIR = ROOT / "real_data" / "ngc2835"

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


PRODUCTS = {
    # ---------------------------------------------------------------
    # PHANGS-MUSE
    # ---------------------------------------------------------------
    "muse_maps": ("vos:phangs/RELEASES/PHANGS-MUSE/DR1.0/MAPS/NGC2835_MAPS_copt_1.15asec.fits"),
    # ---------------------------------------------------------------
    # PHANGS-ALMA CO(2-1), 2 arcsec products
    # ---------------------------------------------------------------
    "alma_broad_mom0": (
        "vos:phangs/RELEASES/PHANGS-ALMA/"
        "by_galaxy/ngc2835/"
        "ngc2835_12m+7m+tp_co21_2as_broad_mom0.fits"
    ),
    "alma_broad_emom0": (
        "vos:phangs/RELEASES/PHANGS-ALMA/"
        "by_galaxy/ngc2835/"
        "ngc2835_12m+7m+tp_co21_2as_broad_emom0.fits"
    ),
    "alma_strict_mom0": (
        "vos:phangs/RELEASES/PHANGS-ALMA/"
        "by_galaxy/ngc2835/"
        "ngc2835_12m+7m+tp_co21_2as_strict_mom0.fits"
    ),
    # Velocity field with prior.
    "alma_mom1": (
        "vos:phangs/RELEASES/PHANGS-ALMA/"
        "by_galaxy/ngc2835/"
        "ngc2835_12m+7m+tp_co21_2as_mom1wprior.fits"
    ),
    "alma_emom1": (
        "vos:phangs/RELEASES/PHANGS-ALMA/"
        "by_galaxy/ngc2835/"
        "ngc2835_12m+7m+tp_co21_2as_emom1wprior.fits"
    ),
    # CO velocity-dispersion products.
    "alma_mom2": (
        "vos:phangs/RELEASES/PHANGS-ALMA/"
        "by_galaxy/ngc2835/"
        "ngc2835_12m+7m+tp_co21_2as_strict_mom2.fits"
    ),
    "alma_emom2": (
        "vos:phangs/RELEASES/PHANGS-ALMA/"
        "by_galaxy/ngc2835/"
        "ngc2835_12m+7m+tp_co21_2as_strict_emom2.fits"
    ),
    # ---------------------------------------------------------------
    # Published PHANGS rotation curve
    # ---------------------------------------------------------------
    "lang_rotation_curve": (
        "vos:phangs/RELEASES/Lang_etal_2020/By_Galaxy/PHANGS_RC_NGC2835_kpc.ecsv"
    ),
}


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


client = vos.Client()

print("=" * 96)
print("NGC 2835 MINIMAL PUBLIC PRODUCT DOWNLOAD")
print("=" * 96)

manifest = []

for label, remote in PRODUCTS.items():
    print()
    print("-" * 96)
    print("LABEL  =", label)
    print("REMOTE =", remote)

    if not client.isfile(remote):
        raise RuntimeError(f"remote file missing: {remote}")

    remote_size = client.size(remote)

    filename = remote.rsplit(
        "/",
        1,
    )[-1]

    local = OUTDIR / filename

    print(
        "REMOTE_SIZE_BYTES =",
        remote_size,
    )

    print(
        "REMOTE_SIZE_MB =",
        remote_size / 1024**2,
    )

    if local.exists() and local.stat().st_size == remote_size:
        print("LOCAL_FILE_ALREADY_COMPLETE=TRUE")

    else:
        if local.exists():
            print(
                "REMOVING_INCOMPLETE_LOCAL_FILE =",
                local,
            )

            local.unlink()

        print(
            "DOWNLOAD_START =",
            local,
        )

        client.copy(
            remote,
            str(local),
        )

        print(
            "DOWNLOAD_COMPLETE =",
            local,
        )

    local_size = local.stat().st_size

    if local_size != remote_size:
        raise RuntimeError(f"size mismatch for {label}: {local_size} != {remote_size}")

    digest = sha256(local)

    print(
        "LOCAL_SIZE_BYTES =",
        local_size,
    )

    print(
        "SHA256 =",
        digest,
    )

    manifest.append(
        {
            "label": label,
            "remote": remote,
            "local": str(local),
            "size_bytes": local_size,
            "sha256": digest,
        }
    )


manifest_path = OUTDIR / "NGC2835_PUBLIC_DATA_MANIFEST.json"

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    + "\n"
)

print()
print("=" * 96)
print("DOWNLOAD SUMMARY")
print("=" * 96)
print(
    "N_PRODUCTS =",
    len(manifest),
)
print(
    "MANIFEST =",
    manifest_path,
)
print("NGC2835_MINIMAL_DOWNLOAD=PASS")
print("RAW_ALMA_VISIBILITIES_DOWNLOADED=FALSE")
print("MUSE_DATACUBE_DOWNLOADED=FALSE")
print("HECTOR_INTERNAL_DATA_USED=FALSE")
