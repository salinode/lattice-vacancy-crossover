"""Publication-grade figure generator for the vacancy clustering and transport study.

Reads simulation outputs directly from `data/*.csv` and writes both vector PDF
and 300-DPI PNG figures into `figures/`.

Styling invariants:
  - Colorblind-safe Okabe-Ito palette (SC: #0072B2, BCC: #D55E00, FCC: #009E73)
  - Consistent publication font hierarchy
  - Clean manuscript titles (zero working-draft tags such as Phase 2/3/7 or Fig B)
  - Non-overlapping annotations with white opaque bbox backgrounds
  - Valid LaTeX notation (\\Lambda, never \\wedge)
  - Figure 6 plotted directly from `data/thermo_controls.csv`
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.lattices import LATTICE_SPECS
from src.statistics import interpolate_spanning_crossing, shared_slope_ancova


# Colorblind-safe Okabe-Ito / Tol palette
PALETTE: Dict[str, str] = {
    "SC": "#0072B2",
    "BCC": "#D55E00",
    "FCC": "#009E73",
    "FCC_HIGH_T": "#882255",
    "BELOW_TC": "#0072B2",
    "ABOVE_TC": "#E69F00",
}

MARKERS: Dict[str, str] = {
    "SC": "o",
    "BCC": "s",
    "FCC": "^",
}

BBOX_STYLE = dict(
    boxstyle="round,pad=0.3",
    facecolor="white",
    edgecolor="#CCCCCC",
    linewidth=0.8,
    alpha=0.94,
)


def apply_publication_style() -> None:
    """Configure consistent Matplotlib rcParams for publication figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.0,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "legend.fontsize": 9.3,
            "legend.framealpha": 0.94,
            "legend.edgecolor": "#CCCCCC",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str, pad_inches: float = 0.14) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=pad_inches)
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)


def _se(series: pd.Series) -> float:
    return float(series.std(ddof=1) / np.sqrt(len(series))) if len(series) > 1 else 0.0


def _sd(series: pd.Series) -> float:
    return float(series.std(ddof=1)) if len(series) > 1 else 0.0


def plot_fig02_diffusivity_vs_cv(df_seed: pd.DataFrame, out_dir: Path) -> None:
    """Figure 2: Tracer diffusivity D vs Cv at matched reduced temperatures (T/Tc = 0.82, 1.25)."""
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    temp_map = {
        "below": {"SC": 0.93, "BCC": 1.30, "FCC": 2.00},
        "above": {"SC": 1.41, "BCC": 1.99, "FCC": 3.06},
    }

    for lat in ("SC", "BCC", "FCC"):
        for regime, ls, mfc, tag in (
            ("above", "--", "white", r"$T/T_c \simeq 1.25$"),
            ("below", "-", PALETTE[lat], r"$T/T_c \simeq 0.82$"),
        ):
            Tv = temp_map[regime][lat]
            d = df_seed[(df_seed["lattice"] == lat) & (np.isclose(df_seed["T"], Tv, atol=0.02))]
            if d.empty:
                continue
            g = d.groupby("Cv", as_index=False).agg(D_mean=("D", "mean"), D_se=("D", _se)).sort_values("Cv")
            ax.errorbar(
                g["Cv"],
                g["D_mean"],
                yerr=g["D_se"],
                label=f"{lat} ($Z={LATTICE_SPECS[lat]['Z']}$, {tag})",
                color=PALETTE[lat],
                marker=MARKERS[lat],
                markerfacecolor=mfc,
                markeredgewidth=1.3,
                linestyle=ls,
                markersize=6.0,
                linewidth=1.8,
                capsize=3.2,
                elinewidth=1.1,
            )

    ax.set_xlabel(r"Vacancy concentration $C_v$")
    ax.set_ylabel(r"Tracer diffusivity $D$ ($a^2/\mathrm{MCS}$)")
    ax.set_ylim(0.009, 0.041)
    ax.legend(loc="upper left", ncol=2, fontsize=8.1, frameon=True, columnspacing=0.9, handlelength=2.1)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig02_diffusivity_vs_cv")


def plot_fig03_normalized_diffusivity(df_seed: pd.DataFrame, out_dir: Path) -> None:
    """Figure 3: Normalized diffusivity D / Cv vs Cv at matched reduced temperatures."""
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    temp_map = {
        "below": {"SC": 0.93, "BCC": 1.30, "FCC": 2.00},
        "above": {"SC": 1.41, "BCC": 1.99, "FCC": 3.06},
    }

    for lat in ("SC", "BCC", "FCC"):
        for regime, ls, mfc, tag in (
            ("above", "--", "white", r"$T/T_c \simeq 1.25$"),
            ("below", "-", PALETTE[lat], r"$T/T_c \simeq 0.82$"),
        ):
            Tv = temp_map[regime][lat]
            d = df_seed[(df_seed["lattice"] == lat) & (np.isclose(df_seed["T"], Tv, atol=0.02))]
            if d.empty:
                continue
            g = (
                d.groupby("Cv", as_index=False)
                .agg(val_mean=("D_over_Cv", "mean"), val_se=("D_over_Cv", _se))
                .sort_values("Cv")
            )
            ax.errorbar(
                g["Cv"],
                g["val_mean"],
                yerr=g["val_se"],
                label=f"{lat} ({tag})",
                color=PALETTE[lat],
                marker=MARKERS[lat],
                markerfacecolor=mfc,
                markeredgewidth=1.3,
                linestyle=ls,
                markersize=6.0,
                linewidth=1.8,
                capsize=3.2,
                elinewidth=1.1,
            )

    ax.set_xlabel(r"Vacancy concentration $C_v$")
    ax.set_ylabel(r"Normalized diffusivity $D / C_v$ ($a^2/\mathrm{MCS}$)")
    ax.set_ylim(0.045, 0.172)
    ax.legend(loc="upper right", ncol=2, fontsize=8.1, frameon=True, columnspacing=0.9, handlelength=2.1)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig03_normalized_diffusivity_vs_cv")


def plot_fig04_fcc_giant_cluster(df_fcc: pd.DataFrame, out_dir: Path) -> None:
    """Figure 4: FCC largest-cluster fraction across the dense concentration sweep."""
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    pc_fcc = float(LATTICE_SPECS["FCC"]["pc"])

    configs = [
        (2.00, r"$T = 2.00$ ($T/T_c \simeq 0.82$, below $T_c$)", PALETTE["FCC"], "o", "-"),
        (3.06, r"$T = 3.06$ ($T/T_c \simeq 1.25$, above $T_c$)", PALETTE["FCC_HIGH_T"], "s", "-"),
    ]
    for T_val, label, color, marker, ls in configs:
        sub = df_fcc[np.isclose(df_fcc["T"], T_val, atol=0.02)]
        if sub.empty:
            continue
        g = (
            sub.groupby("Cv", as_index=False)
            .agg(f_mean=("f_max_pct", "mean"), f_se=("f_max_pct", _se))
            .sort_values("Cv")
        )
        ax.errorbar(
            g["Cv"],
            g["f_mean"],
            yerr=g["f_se"],
            label=label,
            color=color,
            marker=marker,
            linestyle=ls,
            markersize=5.8,
            linewidth=1.8,
            capsize=3.2,
        )

    ax.axvline(pc_fcc, color="#333333", linestyle="--", linewidth=1.3, zorder=1)
    ax.text(
        pc_fcc + 0.0025,
        14.0,
        rf"$p_c^{{\mathrm{{FCC}}}} = {pc_fcc:.4f}$",
        fontsize=9.5,
        color="#222222",
        bbox=BBOX_STYLE,
        va="bottom",
        ha="left",
    )

    ax.set_xlabel(r"Vacancy concentration $C_v$")
    ax.set_ylabel(r"Largest-cluster fraction $f_{\max}$ (%)")
    ax.set_ylim(-2.0, 102.0)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig04_fcc_giant_cluster_vs_cv")


def plot_fig05_spanning_crossover(df_seed: pd.DataFrame, out_dir: Path) -> None:
    """Figure 5: Spanning probability P_span vs Cv at matched subcritical T/Tc = 0.82."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    series_cfg = [
        ("SC", 0.93, r"SC: $T = 0.93$ ($T/T_c = 0.82$)", PALETTE["SC"], "o", "-"),
        ("BCC", 1.30, r"BCC: $T = 1.30$ ($T/T_c = 0.82$)", PALETTE["BCC"], "s", "-"),
        ("FCC", 2.00, r"FCC: $T = 2.00$ ($T/T_c = 0.82$)", PALETTE["FCC"], "^", "-"),
        ("FCC", 3.06, r"FCC: $T = 3.06$ ($T/T_c = 1.25$)", PALETTE["FCC_HIGH_T"], "D", "--"),
    ]

    crossing_notes = []
    for lat, T_val, label, color, marker, ls in series_cfg:
        sub = df_seed[(df_seed["lattice"] == lat) & (np.isclose(df_seed["T"], T_val, atol=0.02))]
        if sub.empty:
            continue
        g = sub.groupby("Cv", as_index=False)["is_spanning"].mean().sort_values("Cv")
        ax.plot(
            g["Cv"],
            g["is_spanning"],
            label=label,
            color=color,
            marker=marker,
            linestyle=ls,
            markersize=5.8,
            linewidth=1.8,
        )
        if T_val < float(LATTICE_SPECS[lat]["Tc"]):
            cv_star = interpolate_spanning_crossing(g["Cv"].to_numpy(), g["is_spanning"].to_numpy(), 0.5)
            pc = float(LATTICE_SPECS[lat]["pc"])
            if cv_star is not None:
                crossing_notes.append(f"{lat}: $C_v^*={cv_star:.4f}$ ($C_v^*/p_c={cv_star / pc:.3f}$)")

    # Horizontal P_span = 0.5 reference
    ax.axhline(0.5, color="#555555", linestyle=":", linewidth=1.2)
    ax.text(
        0.268,
        0.545,
        r"$P_{\mathrm{span}} = 0.5$",
        fontsize=9.0,
        color="#333333",
        bbox=BBOX_STYLE,
        va="bottom",
        ha="left",
    )

    # Vertical random site-percolation thresholds with non-overlapping staggered labels
    pc_label_y = {"FCC": 0.08, "BCC": 0.22, "SC": 0.08}
    for lat in ("FCC", "BCC", "SC"):
        pc = float(LATTICE_SPECS[lat]["pc"])
        ax.axvline(pc, color=PALETTE[lat], linestyle=":", linewidth=1.1, alpha=0.85)
        ax.text(
            pc,
            pc_label_y[lat],
            rf"$p_c^{{\mathrm{{{lat}}}}}={pc:.4f}$",
            fontsize=8.3,
            color=PALETTE[lat],
            bbox=BBOX_STYLE,
            ha="center",
            va="bottom",
        )

    if crossing_notes:
        ax.text(
            0.985,
            0.27,
            "Crossings ($T/T_c = 0.82$):\n" + "\n".join(crossing_notes),
            transform=ax.transAxes,
            fontsize=8.3,
            va="bottom",
            ha="right",
            bbox=BBOX_STYLE,
        )

    ax.set_xlabel(r"Vacancy concentration $C_v$", labelpad=6)
    ax.set_ylabel(r"Spanning probability $P_{\mathrm{span}}$")
    ax.set_xlim(0.132, 0.332)
    ax.set_ylim(-0.04, 1.15)
    ax.legend(loc="upper left", fontsize=8.3, framealpha=0.90, frameon=True)
    fig.tight_layout(pad=1.2)
    _save_fig(fig, out_dir, "fig05_spanning_probability_crossover", pad_inches=0.16)


def plot_fig06_thermo_controls(df_ctrl: pd.DataFrame, out_dir: Path) -> None:
    """Figure 6: Matched reduced-temperature control bars plotted directly from thermo_controls.csv."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.4, 4.4), gridspec_kw={"width_ratios": [1.55, 1.0]}, sharey=True
    )

    # Panel (a): Cv = 0.20 across SC, BCC, FCC
    sub20 = df_ctrl[np.isclose(df_ctrl["Cv"], 0.20, atol=1e-4)]
    lattices = ("SC", "BCC", "FCC")
    x = np.arange(len(lattices), dtype=np.float64)
    width = 0.34

    below_means, below_sds = [], []
    above_means, above_sds = [], []
    for lat in lattices:
        tc = float(LATTICE_SPECS[lat]["Tc"])
        d_lat = sub20[sub20["lattice"] == lat]
        d_lo = d_lat[d_lat["T"] < tc]
        d_hi = d_lat[d_lat["T"] > tc]
        below_means.append(float(d_lo["f_max_pct"].mean()) if not d_lo.empty else 0.0)
        below_sds.append(_sd(d_lo["f_max_pct"]) if not d_lo.empty else 0.0)
        above_means.append(float(d_hi["f_max_pct"].mean()) if not d_hi.empty else 0.0)
        above_sds.append(_sd(d_hi["f_max_pct"]) if not d_hi.empty else 0.0)

    bars1 = ax1.bar(
        x - width / 2,
        below_means,
        width=width,
        yerr=below_sds,
        capsize=4.0,
        color=PALETTE["BELOW_TC"],
        edgecolor="#222222",
        linewidth=0.7,
        label=r"$T / T_c \simeq 0.82$ (below $T_c$)",
    )
    bars2 = ax1.bar(
        x + width / 2,
        above_means,
        width=width,
        yerr=above_sds,
        capsize=4.0,
        color=PALETTE["ABOVE_TC"],
        edgecolor="#222222",
        linewidth=0.7,
        label=r"$T / T_c \simeq 1.25$ (above $T_c$)",
    )

    for bars, sds in ((bars1, below_sds), (bars2, above_sds)):
        for bar, sd_val in zip(bars, sds):
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + sd_val + 2.0,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(lattices)
    ax1.set_ylabel(r"Largest-cluster fraction $f_{\max}$ (%)")
    ax1.set_title(r"(a) Matched-Temperature Control at $C_v = 0.20$")
    ax1.legend(loc="upper left", frameon=True)

    # Panel (b): FCC geometric-support control (Cv = 0.20 vs Cv = 0.15)
    fcc_ctrl = df_ctrl[df_ctrl["lattice"] == "FCC"]
    tc_fcc = float(LATTICE_SPECS["FCC"]["Tc"])
    cvs = (0.20, 0.15)
    x2 = np.arange(len(cvs), dtype=np.float64)
    b_m, b_s, a_m, a_s = [], [], [], []
    for cv in cvs:
        d_cv = fcc_ctrl[np.isclose(fcc_ctrl["Cv"], cv, atol=1e-4)]
        d_lo = d_cv[d_cv["T"] < tc_fcc]
        d_hi = d_cv[d_cv["T"] > tc_fcc]
        b_m.append(float(d_lo["f_max_pct"].mean()) if not d_lo.empty else 0.0)
        b_s.append(_sd(d_lo["f_max_pct"]) if not d_lo.empty else 0.0)
        a_m.append(float(d_hi["f_max_pct"].mean()) if not d_hi.empty else 0.0)
        a_s.append(_sd(d_hi["f_max_pct"]) if not d_hi.empty else 0.0)

    b1 = ax2.bar(
        x2 - width / 2,
        b_m,
        width=width,
        yerr=b_s,
        capsize=4.0,
        color=PALETTE["BELOW_TC"],
        edgecolor="#222222",
        linewidth=0.7,
    )
    b2 = ax2.bar(
        x2 + width / 2,
        a_m,
        width=width,
        yerr=a_s,
        capsize=4.0,
        color=PALETTE["ABOVE_TC"],
        edgecolor="#222222",
        linewidth=0.7,
    )
    for bars, sds in ((b1, b_s), (b2, a_s)):
        for bar, sd_val in zip(bars, sds):
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + sd_val + 2.0,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
            )

    ax2.set_xticks(x2)
    ax2.set_xticklabels([r"FCC $C_v = 0.20$", r"FCC $C_v = 0.15$"])
    ax2.set_title(r"(b) FCC Control Below $p_c^{\mathrm{FCC}}$")
    ax1.set_ylim(0.0, 105.0)

    fig.tight_layout()
    _save_fig(fig, out_dir, "fig06_thermo_controls_cv020")


def plot_fig07_diffusivity_vs_clustering(df_seed: pd.DataFrame, out_dir: Path) -> None:
    """Figure 7: Seed-level tracer diffusivity D vs local clustering index C."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    sub = df_seed[(df_seed["D"] > 0) & (df_seed["C_index"] > 0)].copy()

    for lat in ("SC", "BCC", "FCC"):
        d = sub[sub["lattice"] == lat]
        if d.empty:
            continue
        ax.scatter(
            d["C_index"],
            d["D"],
            s=22,
            alpha=0.78,
            color=PALETTE[lat],
            marker=MARKERS[lat],
            edgecolors="none",
            label=lat,
        )

    # Pooled log-log power-law fit
    lx = np.log(sub["C_index"].to_numpy(dtype=np.float64))
    ly = np.log(sub["D"].to_numpy(dtype=np.float64))
    slope, intercept = np.polyfit(lx, ly, 1)
    x_grid = np.geomspace(sub["C_index"].min() * 0.96, sub["C_index"].max() * 1.04, 150)
    y_fit = np.exp(intercept) * (x_grid ** slope)

    r_lin, _ = stats.pearsonr(sub["C_index"], sub["D"])
    rho, _ = stats.spearmanr(sub["C_index"], sub["D"])
    anc = shared_slope_ancova(sub, "C_index", "D", log_transform=True)
    y_pred_log = intercept + slope * lx
    r2_log = 1.0 - float(np.sum((ly - y_pred_log) ** 2)) / float(np.sum((ly - np.mean(ly)) ** 2))

    ax.plot(
        x_grid,
        y_fit,
        color="#222222",
        linestyle="--",
        linewidth=1.5,
        label=rf"Pooled fit: $D \propto \mathcal{{C}}^{{{slope:.2f}}}$ ($R^2={r2_log:.2f}$)",
    )

    if anc["F"] is not None:
        p_val = float(anc["p_value"])
        if p_val < 1e-3:
            exp = int(np.floor(np.log10(p_val)))
            mant = p_val / (10.0 ** exp)
            p_str = rf"p = {mant:.2f} \times 10^{{{exp}}}"
        else:
            p_str = f"p = {p_val:.4f}"
        f_text = f"Shared-slope $F({anc['df1']}, {anc['df2']}) = {anc['F']:.2f}$ (${p_str}$)"
    else:
        f_text = "Shared-slope F: n/a"
    ax.text(
        0.03,
        0.04,
        f"$n = {len(sub)}$ seeds\n"
        f"Pearson $r = {r_lin:.3f}$, Spearman $\\rho = {rho:.3f}$\n"
        f"{f_text}",
        transform=ax.transAxes,
        fontsize=8.6,
        va="bottom",
        ha="left",
        bbox=BBOX_STYLE,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import FormatStrFormatter, NullFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax.set_xticks([1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(max(0.95, sub["C_index"].min() * 0.94), sub["C_index"].max() * 1.06)
    ax.set_xlabel(r"Local clustering index $\mathcal{C} = \langle z_v \rangle / (Z C_v)$")
    ax.set_ylabel(r"Tracer diffusivity $D$ ($a^2/\mathrm{MCS}$)")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig07_diffusivity_vs_clustering")


def plot_fig08_finite_size_fcc(df_fs: pd.DataFrame, out_dir: Path) -> None:
    """Figure 8: Finite-size comparison for FCC at Cv = 0.20, T = 2.00 (L=20 vs L=40)."""
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    sizes = sorted(df_fs["L"].unique().tolist())
    x = np.arange(len(sizes), dtype=np.float64)

    means = []
    sds = []
    labels = []
    for L_val in sizes:
        s = df_fs[df_fs["L"] == L_val]
        means.append(float(s["f_max_pct"].mean()))
        sds.append(_sd(s["f_max_pct"]))
        n_sites = int(s["N"].iloc[0])
        labels.append(f"$L = {int(L_val)}$\n($N = {n_sites:,}$)")

    bars = ax.bar(
        x,
        means,
        width=0.48,
        yerr=sds,
        capsize=5.0,
        color=[PALETTE["FCC"], PALETTE["SC"]][: len(sizes)],
        edgecolor="#222222",
        linewidth=0.8,
    )
    for bar, sd_val in zip(bars, sds):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + sd_val + 2.2,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            bbox=BBOX_STYLE,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Largest-cluster fraction $f_{\max}$ (%)")
    ax.set_ylim(0.0, max(means) + max(sds) + 18.0)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig08_finite_size_fcc")


def plot_fig09_two_param_master_curve(df_seed: pd.DataFrame, out_dir: Path) -> None:
    """Figure 9: Two-parameter reduced-variable transport collapse D/D0 vs M."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    sub = df_seed[(df_seed["D_over_D0"] > 0) & (df_seed["Cv_over_pc"] > 0) & (df_seed["T_over_Tc"] > 0)].copy()

    M = (sub["Cv_over_pc"].to_numpy(dtype=np.float64) ** 0.75) * sub["T_over_Tc"].to_numpy(dtype=np.float64)
    sub["M"] = M

    for lat in ("SC", "BCC", "FCC"):
        d = sub[sub["lattice"] == lat]
        if d.empty:
            continue
        ax.scatter(
            d["M"],
            d["D_over_D0"],
            s=22,
            alpha=0.78,
            color=PALETTE[lat],
            marker=MARKERS[lat],
            edgecolors="none",
            label=lat,
        )

    # Fixed-exponent master fit: D/D0 = A * M
    ln_y = np.log(sub["D_over_D0"].to_numpy(dtype=np.float64))
    ln_M = np.log(M)
    a_fixed = float(np.mean(ln_y - ln_M))
    A_fixed = float(np.exp(a_fixed))
    r2_fixed = 1.0 - float(np.sum((ln_y - (a_fixed + ln_M)) ** 2)) / float(np.sum((ln_y - np.mean(ln_y)) ** 2))

    # Free two-parameter OLS for annotation
    X = np.column_stack([
        np.ones(len(sub)),
        np.log(sub["Cv_over_pc"].to_numpy(dtype=np.float64)),
        np.log(sub["T_over_Tc"].to_numpy(dtype=np.float64)),
    ])
    beta, _, _, _ = np.linalg.lstsq(X, ln_y, rcond=None)
    alpha_hat, beta_hat = float(beta[1]), float(beta[2])

    m_grid = np.geomspace(M.min() * 0.94, M.max() * 1.06, 150)
    ax.plot(
        m_grid,
        A_fixed * m_grid,
        color="#222222",
        linestyle="--",
        linewidth=1.6,
        label=rf"Master scaling: $D/D_0 = {A_fixed:.4f}\,M$ ($R^2 = {r2_fixed:.4f}$)",
    )

    ax.text(
        0.97,
        0.05,
        f"$n = {len(sub)}$ seeds\n"
        rf"Free fit: $\alpha = {alpha_hat:.3f}$, $\beta_{{T/T_c}} = {beta_hat:.3f}$" "\n"
        rf"Fixed $(3/4,\,1)$: $R^2 = {r2_fixed:.4f}$",
        transform=ax.transAxes,
        fontsize=8.8,
        va="bottom",
        ha="right",
        bbox=BBOX_STYLE,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Two-parameter scaling variable $M = (C_v / p_c)^{3/4}(T / T_c)$")
    ax.set_ylabel(r"Normalized diffusivity $D / D_0$")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig09_two_param_master_curve")


def plot_fig10_single_param_lambda_transport(df_seed: pd.DataFrame, out_dir: Path) -> None:
    """Figure 10: Single-parameter reduced-variable test D/D0 vs Lambda."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    sub = df_seed[(df_seed["D_over_D0"] > 0) & (df_seed["Lambda"] > 0)].copy()

    slope_lines = []
    for lat in ("SC", "BCC", "FCC"):
        d = sub[sub["lattice"] == lat]
        if d.empty:
            continue
        ax.scatter(
            d["Lambda"],
            d["D_over_D0"],
            s=22,
            alpha=0.78,
            color=PALETTE[lat],
            marker=MARKERS[lat],
            edgecolors="none",
            label=lat,
        )
        lx = np.log(d["Lambda"].to_numpy(dtype=np.float64))
        ly = np.log(d["D_over_D0"].to_numpy(dtype=np.float64))
        if len(lx) >= 2:
            m, b = np.polyfit(lx, ly, 1)
            xg = np.geomspace(d["Lambda"].min() * 0.96, d["Lambda"].max() * 1.04, 80)
            ax.plot(xg, np.exp(b) * (xg ** m), color=PALETTE[lat], linestyle="--", linewidth=1.35)
            slope_lines.append(f"{lat}: slope $= {m:.3f}$")

    if slope_lines:
        ax.text(
            0.03,
            0.05,
            "Per-lattice log–log slopes:\n" + "\n".join(slope_lines),
            transform=ax.transAxes,
            fontsize=8.8,
            va="bottom",
            ha="left",
            bbox=BBOX_STYLE,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Single reduced variable $\Lambda = (C_v / p_c)(T_c / T)$")
    ax.set_ylabel(r"Normalized diffusivity $D / D_0$")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig10_single_param_lambda_transport")


def plot_fig11_single_param_lambda_fmax(df_seed: pd.DataFrame, out_dir: Path) -> None:
    """Figure 11: Largest-cluster fraction f_max (%) vs single reduced variable Lambda."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    sub = df_seed[df_seed["Lambda"] > 0].copy()

    for lat in ("SC", "BCC", "FCC"):
        d = sub[sub["lattice"] == lat]
        if d.empty:
            continue
        ax.scatter(
            d["Lambda"],
            d["f_max_pct"],
            s=22,
            alpha=0.78,
            color=PALETTE[lat],
            marker=MARKERS[lat],
            edgecolors="none",
            label=lat,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Single reduced variable $\Lambda = (C_v / p_c)(T_c / T)$", labelpad=6)
    ax.set_ylabel(r"Largest-cluster fraction $f_{\max}$ (%)")
    ax.set_ylim(-2.0, 102.0)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout(pad=1.2)
    _save_fig(fig, out_dir, "fig11_single_param_lambda_fmax", pad_inches=0.16)


def generate_all_figures(data_dir: Path, figures_dir: Path) -> None:
    """Generate all manuscript figures from CSVs in `data_dir`."""
    apply_publication_style()
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed_path = data_dir / "seed_level_465.csv"
    dense_span_path = data_dir / "spanning_dense_subcritical.csv"
    fcc_path = data_dir / "fcc_concentration_sweep.csv"
    ctrl_path = data_dir / "thermo_controls.csv"
    fs_path = data_dir / "finite_size_fcc.csv"

    if seed_path.exists():
        df_seed = pd.read_csv(seed_path)
        df_span = pd.read_csv(dense_span_path) if dense_span_path.exists() else df_seed
        plot_fig02_diffusivity_vs_cv(df_seed, figures_dir)
        plot_fig03_normalized_diffusivity(df_seed, figures_dir)
        plot_fig05_spanning_crossover(df_span, figures_dir)
        plot_fig07_diffusivity_vs_clustering(df_seed, figures_dir)
        plot_fig09_two_param_master_curve(df_seed, figures_dir)
        plot_fig10_single_param_lambda_transport(df_seed, figures_dir)
        plot_fig11_single_param_lambda_fmax(df_seed, figures_dir)

    if fcc_path.exists():
        df_fcc = pd.read_csv(fcc_path)
        plot_fig04_fcc_giant_cluster(df_fcc, figures_dir)

    if ctrl_path.exists():
        df_ctrl = pd.read_csv(ctrl_path)
        plot_fig06_thermo_controls(df_ctrl, figures_dir)

    if fs_path.exists():
        df_fs = pd.read_csv(fs_path)
        plot_fig08_finite_size_fcc(df_fs, figures_dir)

    generated = sorted(p.name for p in figures_dir.glob("*.pdf"))
    print(f"Generated {len(generated)} figure pairs (PDF + 300-DPI PNG) in {figures_dir}:")
    for name in generated:
        print(f"  - {name}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication figures from simulation CSVs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Input CSV directory (default: data).")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
        help="Output figure directory (default: figures).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    generate_all_figures(args.data_dir, args.figures_dir)


if __name__ == "__main__":
    main()
