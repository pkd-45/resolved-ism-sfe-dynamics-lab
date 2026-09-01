from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

KPC_IN_KM = 3.0856775814913673e16
SECONDS_PER_MYR = 365.25 * 24 * 3600 * 1e6


def v_over_sigma(vrot, sigma):
    vrot = np.asarray(vrot, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if vrot.shape != sigma.shape:
        raise ValueError("vrot and sigma must have the same shape")
    out = np.full(vrot.shape, np.nan)
    good = np.isfinite(vrot) & np.isfinite(sigma) & (sigma > 0)
    out[good] = np.abs(vrot[good]) / sigma[good]
    return out


def orbital_time_myr(radius_kpc, vrot_kms):
    radius = np.asarray(radius_kpc, dtype=float)
    velocity = np.asarray(vrot_kms, dtype=float)
    if radius.shape != velocity.shape:
        raise ValueError("radius and velocity must have the same shape")
    out = np.full(radius.shape, np.nan)
    good = np.isfinite(radius) & np.isfinite(velocity) & (radius > 0) & (velocity > 0)
    seconds = 2 * np.pi * radius[good] * KPC_IN_KM / velocity[good]
    out[good] = seconds / SECONDS_PER_MYR
    return out


def specific_angular_momentum(radius, surface_density, velocity):
    radius = np.asarray(radius, dtype=float)
    surface_density = np.asarray(surface_density, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if not (
        radius.ndim == surface_density.ndim == velocity.ndim == 1
        and len(radius) == len(surface_density) == len(velocity)
    ):
        raise ValueError("inputs must be equal-length 1-D arrays")
    good = (
        np.isfinite(radius)
        & np.isfinite(surface_density)
        & np.isfinite(velocity)
        & (radius >= 0)
        & (surface_density >= 0)
    )
    r = radius[good]
    sigma = surface_density[good]
    vel = velocity[good]
    order = np.argsort(r)
    r = r[order]
    sigma = sigma[order]
    vel = vel[order]
    if r.size < 2 or np.any(np.diff(r) <= 0):
        raise ValueError("need at least two unique valid radii")
    numerator = trapezoid(sigma * vel * r**2, r)
    denominator = trapezoid(sigma * r, r)
    return float(numerator / denominator) if denominator > 0 else float("nan")
