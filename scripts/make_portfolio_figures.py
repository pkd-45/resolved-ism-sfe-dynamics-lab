#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

ROOT = Path.cwd()
FIGDIR = ROOT / "docs" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 1. NGC 0628 scale dependence
# ---------------------------------------------------------------------

scale_path = ROOT / "outputs" / "ngc0628_real" / "scale_ladder" / "ngc0628_scale_ladder_summary.csv"

scale = pd.read_csv(scale_path)

good = np.isfinite(scale["rsk_scatter_dex"]) & (scale["rsk_status"] == "PASS")

fig, ax = plt.subplots(figsize=(6.4, 4.8))

ax.plot(
    scale.loc[good, "target_scale_pc"],
    scale.loc[good, "rsk_scatter_dex"],
    marker="o",
)

ax.set_xlabel("Physical resolution [pc]")
ax.set_ylabel("Resolved SK scatter [dex]")
ax.set_title("NGC 0628: rSK scatter decreases with averaging scale")
ax.grid(alpha=0.25)

fig.tight_layout()

fig.savefig(
    FIGDIR / "fig1_ngc0628_rsk_scatter_vs_scale.png",
    dpi=220,
    bbox_inches="tight",
)

fig.savefig(
    FIGDIR / "fig1_ngc0628_rsk_scatter_vs_scale.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# 2. NGC 2835 resolved molecular SK relation
# ---------------------------------------------------------------------

table_path = ROOT / "outputs" / "ngc2835_real" / "regression" / "resolved_analysis_table.csv"

audit_summary_path = (
    ROOT / "outputs" / "ngc2835_real" / "final_audit" / "ngc2835_sampling_phase_audit_summary.json"
)

df = pd.read_csv(table_path)

with audit_summary_path.open() as handle:
    audit = json.load(handle)

good = (
    np.isfinite(df["sigma_mol"])
    & np.isfinite(df["sigma_sfr"])
    & (df["sigma_mol"] > 0)
    & (df["sigma_sfr"] > 0)
)

x = np.log10(df.loc[good, "sigma_mol"].to_numpy())

y = np.log10(df.loc[good, "sigma_sfr"].to_numpy())

fit = linregress(x, y)

grid = np.linspace(
    np.min(x),
    np.max(x),
    200,
)

phase_slope = audit["global_rsk_slope"]
phase_scatter = audit["global_rsk_scatter_dex"]

fig, ax = plt.subplots(figsize=(6.4, 5.2))

ax.scatter(
    x,
    y,
    s=18,
    alpha=0.65,
)

ax.plot(
    grid,
    fit.intercept + fit.slope * grid,
)

text = (
    f"N = {len(x)} beam-spaced regions\n"
    f"fixed-phase slope = {fit.slope:.2f}\n"
    f"phase median slope = {phase_slope['median']:.2f}\n"
    f"phase P16–P84 = "
    f"{phase_slope['p16']:.2f}–{phase_slope['p84']:.2f}\n"
    f"scatter ≈ {phase_scatter['median']:.2f} dex"
)

ax.text(
    0.04,
    0.96,
    text,
    transform=ax.transAxes,
    va="top",
)

ax.set_xlabel(
    r"$\log_{10}\,\Sigma_{\rm mol}$ "
    r"[$M_\odot\,{\rm pc}^{-2}$]"
)

ax.set_ylabel(
    r"$\log_{10}\,\Sigma_{\rm SFR}$ "
    r"[$M_\odot\,{\rm yr}^{-1}\,{\rm kpc}^{-2}$]"
)

ax.set_title("NGC 2835: resolved molecular gas–star formation relation")

ax.grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    FIGDIR / "fig2_ngc2835_resolved_sk.png",
    dpi=220,
    bbox_inches="tight",
)

fig.savefig(
    FIGDIR / "fig2_ngc2835_resolved_sk.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# 3. Sampling-phase robustness of dynamics diagnostics
# ---------------------------------------------------------------------

phase_path = ROOT / "outputs" / "ngc2835_real" / "final_audit" / "ngc2835_sampling_phase_audit.csv"

phase = pd.read_csv(phase_path)

phase_index = np.arange(len(phase))

dv = phase["abs_delta_v_partial_rho_radius"].to_numpy(dtype=float)

lw = phase["co_linewidth_partial_rho_radius"].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(7.2, 4.8))

ax.axhline(
    0.0,
    linewidth=1,
)

ax.scatter(
    phase_index,
    dv,
    s=22,
    alpha=0.7,
    label=r"$|\Delta v_{\rm CO}|$",
)

ax.scatter(
    phase_index,
    lw,
    s=22,
    alpha=0.7,
    label=r"$\sigma_{\rm CO}$",
)

ax.axhline(
    np.nanmedian(dv),
    linestyle="--",
    linewidth=1,
)

ax.axhline(
    np.nanmedian(lw),
    linestyle=":",
    linewidth=1,
)

ax.set_xlabel("Beam-sampling grid phase")
ax.set_ylabel(
    r"Partial Spearman $\rho$ "
    r"(controlling for radius)"
)

ax.set_title("NGC 2835: sensitivity of local dynamics diagnostics")

ax.legend()

ax.text(
    0.02,
    0.03,
    "Grid phases are a robustness test, not statistical confidence intervals.",
    transform=ax.transAxes,
    fontsize=9,
)

ax.grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    FIGDIR / "fig3_ngc2835_dynamics_phase_audit.png",
    dpi=220,
    bbox_inches="tight",
)

fig.savefig(
    FIGDIR / "fig3_ngc2835_dynamics_phase_audit.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# 4. Copy existing matched-map figure
# ---------------------------------------------------------------------

source_map = ROOT / "outputs" / "ngc2835_real" / "ngc2835_sfe_dynamics_maps.png"

destination_map = FIGDIR / "fig4_ngc2835_sfe_dynamics_maps.png"

shutil.copy2(
    source_map,
    destination_map,
)


print("PORTFOLIO_FIGURES=PASS")

for path in sorted(FIGDIR.iterdir()):
    print(path)
