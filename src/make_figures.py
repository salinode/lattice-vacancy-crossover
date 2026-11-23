"""Publication-grade figure generator for the vacancy clustering and transport study.

Reads simulation outputs directly from `data/*.csv` and writes both vector PDF
and 300-DPI PNG figures into `figures/`.

Styling invariants:
  - Colorblind-safe Okabe-Ito palette (SC: #0072B2, BCC: #D55E00, FCC: #009E73)
  - Consistent APS publication font hierarchy (no top titles on single-panel figures)
  - Non-overlapping annotations with white opaque bbox backgrounds
  - Generous savefig padding (pad_inches >= 0.14) to prevent subscript clipping
  - Valid LaTeX notation (\\Lambda, \\mathcal{C}, D_{\\mathrm{ref}})
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter
import numpy as np
import pandas as pd
from scipy import stats

from src.lattices import LATTICE_SPECS
from src.statistics import (
    _controlled_regression_for_predictor,
    _run_grouped_cv,
    fit_logistic_crossing_with_bootstrap,
    shared_slope_ancova,
    wilson_ci,
)


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

DREF_MAP: Dict[str, float] = {
    k: (1.0 / 3.0) * float(v["f0_dilute"]) * float(v["nn_dist_sq"])
    for k, v in LATTICE_SPECS.items()
}


def apply_publication_style() -> None:
    """Configure consistent Matplotlib rcParams for publication figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10.5,
            "axes.titlesize": 11.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.0,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "legend.fontsize": 9.1,
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


def _fmt_p(p_val: float) -> str:
    if p_val < 1e-300:
        return r"p < 10^{-300}"
    if p_val < 1e-3:
        exp = int(np.floor(np.log10(p_val)))
        mant = p_val / (10.0 ** exp)
        return rf"p = {mant:.2f} \times 10^{{{exp}}}"
    return f"p = {p_val:.4f}"


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


def plot_fig05_spanning_crossover(
    df_span: pd.DataFrame,
    out_dir: Path,
    df_j0: Optional[pd.DataFrame] = None,
) -> None:
    """Figure 5: Spanning probability P_span vs Cv with Wilson 95% CIs, logistic fits, and J=0 baselines."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    series_cfg = [
        ("SC", 0.93, r"SC: $J=1$, $T/T_c = 0.82$", PALETTE["SC"], "o", "-"),
        ("BCC", 1.30, r"BCC: $J=1$, $T/T_c = 0.82$", PALETTE["BCC"], "s", "-"),
        ("FCC", 2.00, r"FCC: $J=1$, $T/T_c = 0.82$", PALETTE["FCC"], "^", "-"),
        ("FCC", 3.06, r"FCC: $J=1$, $T/T_c = 1.25$", PALETTE["FCC_HIGH_T"], "D", "--"),
    ]

    crossing_notes = []
    for idx, (lat, T_val, label, color, marker, ls) in enumerate(series_cfg):
        sub = df_span[(df_span["lattice"] == lat) & (np.isclose(df_span["T"], T_val, atol=0.02))]
        if sub.empty:
            continue
        g = (
            sub.groupby("Cv", as_index=False)
            .agg(n=("is_spanning", "count"), k=("is_spanning", "sum"))
            .sort_values("Cv")
        )
        p_obs = g["k"].to_numpy(dtype=np.float64) / g["n"].to_numpy(dtype=np.float64)
        cv_arr = g["Cv"].to_numpy(dtype=np.float64)
        ci_lo = np.array([wilson_ci(int(k), int(n))[0] for k, n in zip(g["k"], g["n"])])
        ci_hi = np.array([wilson_ci(int(k), int(n))[1] for k, n in zip(g["k"], g["n"])])
        yerr = np.vstack([np.maximum(0.0, p_obs - ci_lo), np.maximum(0.0, ci_hi - p_obs)])

        pc = float(LATTICE_SPECS[lat]["pc"])
        fit_info, _ = fit_logistic_crossing_with_bootstrap(
            cv_arr, g["k"].to_numpy(), g["n"].to_numpy(), pc=pc, n_boot=2000, boot_seed=20262000 + idx
        )
        g0 = fit_info["logistic_gamma0"]
        g1 = fit_info["logistic_gamma1"]
        x_smooth = np.linspace(max(0.132, cv_arr.min() - 0.005), min(0.34, cv_arr.max() + 0.005), 240)
        p_smooth = 1.0 / (1.0 + np.exp(-np.clip(g0 + g1 * x_smooth, -40.0, 40.0)))

        ax.plot(x_smooth, p_smooth, color=color, linestyle=ls, linewidth=1.7, zorder=3)
        ax.errorbar(
            cv_arr,
            p_obs,
            yerr=yerr,
            label=label,
            color=color,
            marker=marker,
            linestyle="none",
            markersize=5.2,
            capsize=2.4,
            elinewidth=0.95,
            zorder=4,
        )

        if T_val < float(LATTICE_SPECS[lat]["Tc"]) and fit_info["Cv_star"] is not None:
            c_s = fit_info["Cv_star"]
            c_lo = fit_info["Cv_star_ci95_lo"]
            c_hi = fit_info["Cv_star_ci95_hi"]
            crossing_notes.append(
                f"{lat}: $C_v^*={c_s:.4f}$ [{c_lo:.4f}, {c_hi:.4f}]"
            )

    # Overlay non-interacting (J=0) L=20 random site-percolation logistic curves
    if df_j0 is not None:
        j0_l20 = df_j0[df_j0["L"] == 20]
        first_j0 = True
        for lat_i, lat in enumerate(("FCC", "BCC", "SC")):
            sj = j0_l20[j0_l20["lattice"] == lat]
            if sj.empty:
                continue
            gj = (
                sj.groupby("Cv", as_index=False)
                .agg(n=("is_spanning", "count"), k=("is_spanning", "sum"))
                .sort_values("Cv")
            )
            cv_j = gj["Cv"].to_numpy(dtype=np.float64)
            pc = float(LATTICE_SPECS[lat]["pc"])
            f_j, _ = fit_logistic_crossing_with_bootstrap(
                cv_j, gj["k"].to_numpy(), gj["n"].to_numpy(), pc=pc, n_boot=500, boot_seed=20265000 + lat_i
            )
            xs = np.linspace(cv_j.min(), cv_j.max(), 240)
            ps = 1.0 / (1.0 + np.exp(-np.clip(f_j["logistic_gamma0"] + f_j["logistic_gamma1"] * xs, -40.0, 40.0)))
            ax.plot(
                xs,
                ps,
                color=PALETTE[lat],
                linestyle=":",
                linewidth=1.55,
                alpha=0.85,
                label=(r"$J=0$ random ($L=20$, color-matched)" if first_j0 else None),
                zorder=2,
            )
            first_j0 = False

    # Horizontal P_span = 0.5 reference
    ax.axhline(0.5, color="#555555", linestyle="-.", linewidth=1.0, alpha=0.8, zorder=1)
    ax.text(
        0.136,
        0.525,
        r"$P_{\mathrm{span}} = 0.5$",
        fontsize=8.5,
        color="#333333",
        bbox=BBOX_STYLE,
        va="bottom",
        ha="left",
    )

    # Vertical thermodynamic random site-percolation thresholds
    pc_y = {"FCC": 0.03, "BCC": 0.03, "SC": 0.03}
    pc_x_off = {"FCC": 0.0, "BCC": 0.0, "SC": 0.0}
    for lat in ("FCC", "BCC", "SC"):
        pc = float(LATTICE_SPECS[lat]["pc"])
        ax.axvline(pc, color=PALETTE[lat], linestyle=(0, (1, 2.5)), linewidth=1.0, alpha=0.65, zorder=1)
        ax.text(
            pc + pc_x_off[lat],
            pc_y[lat],
            rf"$p_c^{{\mathrm{{{lat}}}}}$",
            fontsize=8.3,
            color=PALETTE[lat],
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=PALETTE[lat], linewidth=0.6, alpha=0.92),
            ha="center",
            va="bottom",
        )

    if crossing_notes:
        ax.text(
            0.985,
            0.16,
            r"Logistic $C_v^*$ ($J=1$, $T/T_c=0.82$, 95% CI):" + "\n" + "\n".join(crossing_notes),
            transform=ax.transAxes,
            fontsize=7.8,
            va="bottom",
            ha="right",
            bbox=BBOX_STYLE,
        )

    ax.set_xlabel(r"Vacancy concentration $C_v$", labelpad=6)
    ax.set_ylabel(r"Spanning probability $P_{\mathrm{span}}$")
    ax.set_xlim(0.132, 0.342)
    ax.set_ylim(-0.04, 1.25)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper left", fontsize=8.0, framealpha=0.94, frameon=True, ncol=2, columnspacing=0.8)
    fig.tight_layout(pad=1.2)
    _save_fig(fig, out_dir, "fig05_spanning_probability_crossover", pad_inches=0.16)


def plot_fig06_thermo_controls(df_ctrl: pd.DataFrame, out_dir: Path) -> None:
    """Figure 6: Matched reduced-temperature control bars plotted directly from thermo_controls.csv."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.4, 4.4), gridspec_kw={"width_ratios": [1.55, 1.0]}, sharey=True
    )

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
        x2 - width / 2, b_m, width=width, yerr=b_s, capsize=4.0,
        color=PALETTE["BELOW_TC"], edgecolor="#222222", linewidth=0.7,
    )
    b2 = ax2.bar(
        x2 + width / 2, a_m, width=width, yerr=a_s, capsize=4.0,
        color=PALETTE["ABOVE_TC"], edgecolor="#222222", linewidth=0.7,
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
    """Figure 7: Seed-level tracer diffusivity D vs time-aligned production-window clustering C_bar."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    c_col = "C_index_window_mean" if "C_index_window_mean" in df_seed.columns else "C_index"
    c_sym = r"\bar{\mathcal{C}}" if c_col == "C_index_window_mean" else r"\mathcal{C}"
    sub = df_seed[(df_seed["D"] > 0) & (df_seed[c_col] > 0)].copy()

    for lat in ("SC", "BCC", "FCC"):
        d = sub[sub["lattice"] == lat]
        if d.empty:
            continue
        ax.scatter(
            d[c_col],
            d["D"],
            s=22,
            alpha=0.78,
            color=PALETTE[lat],
            marker=MARKERS[lat],
            edgecolors="none",
            label=lat,
        )

    lx = np.log(sub[c_col].to_numpy(dtype=np.float64))
    ly = np.log(sub["D"].to_numpy(dtype=np.float64))
    slope, intercept = np.polyfit(lx, ly, 1)
    x_grid = np.geomspace(sub[c_col].min() * 0.96, sub[c_col].max() * 1.04, 150)
    y_fit = np.exp(intercept) * (x_grid ** slope)

    r_lin, _ = stats.pearsonr(sub[c_col], sub["D"])
    rho, _ = stats.spearmanr(sub[c_col], sub["D"])
    y_pred_log = intercept + slope * lx
    r2_log = 1.0 - float(np.sum((ly - y_pred_log) ** 2)) / float(np.sum((ly - np.mean(ly)) ** 2))

    ctrl = _controlled_regression_for_predictor(sub, c_col)
    part_r = ctrl["partial_r"]
    b_c = ctrl["b_predictor"]
    b_se = ctrl["b_predictor_se"]
    p_str = _fmt_p(ctrl["b_predictor_p"])

    ax.plot(
        x_grid,
        y_fit,
        color="#222222",
        linestyle="--",
        linewidth=1.5,
        label=rf"Pooled fit: $D \propto {c_sym}^{{{slope:.2f}}}$ ($R^2={r2_log:.2f}$)",
    )

    ax.text(
        0.03,
        0.04,
        f"$n = {len(sub)}$ seeds\n"
        f"Bivariate: $r = {r_lin:.3f}$, $\\rho = {rho:.3f}$\n"
        rf"Controlled ($C_v, T/T_c,$ lattice): $b_{{{c_sym}}} = {b_c:.2f} \pm {b_se:.2f}$" "\n"
        rf"Partial $r = {part_r:.3f}$ (${p_str}$)",
        transform=ax.transAxes,
        fontsize=8.2,
        va="bottom",
        ha="left",
        bbox=BBOX_STYLE,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xticks([1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(max(0.95, sub[c_col].min() * 0.94), sub[c_col].max() * 1.06)
    ax.set_ylim(0.0072, 0.048)
    ax.set_xlabel(rf"Production-window local clustering ${c_sym}$ ($2{{,}}000$–$6{{,}}000$ MCS)")
    ax.set_ylabel(r"Tracer diffusivity $D$ ($a^2/\mathrm{MCS}$)")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig07_diffusivity_vs_clustering")


def plot_fig08_finite_size_fcc(
    df_fs: pd.DataFrame,
    out_dir: Path,
    df_coarsen: Optional[pd.DataFrame] = None,
) -> None:
    """Figure 8: 2-panel finite-size comparison (L=20 n=15 vs L=40 n=10) & coarsening kinetics R_S(t)."""
    if df_coarsen is None or df_coarsen.empty:
        fig, ax1 = plt.subplots(figsize=(5.6, 4.4))
        ax2 = None
    else:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(9.8, 4.35), gridspec_kw={"width_ratios": [1.0, 1.35]}
        )

    sizes = sorted(df_fs["L"].unique().tolist())
    x = np.arange(len(sizes), dtype=np.float64)
    means, sds, labels = [], [], []
    c_vals, d_vals = [], []
    for L_val in sizes:
        s = df_fs[df_fs["L"] == L_val]
        means.append(float(s["f_max_pct"].mean()))
        sds.append(_sd(s["f_max_pct"]))
        c_vals.append(float(s["C_index"].mean()))
        d_vals.append(float(s["D"].mean()))
        n_sites = int(s["N"].iloc[0])
        labels.append(f"$L = {int(L_val)}$ ($n={len(s)}$)\n$N = {n_sites:,}$")

    bars = ax1.bar(
        x,
        means,
        width=0.46,
        yerr=sds,
        capsize=5.0,
        color=[PALETTE["FCC"], PALETTE["SC"]][: len(sizes)],
        edgecolor="#222222",
        linewidth=0.8,
    )
    for bar, sd_val in zip(bars, sds):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + sd_val + 2.0,
            f"{h:.1f}%\n($\\pm${sd_val:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.4,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
        )

    if len(sizes) >= 2:
        rel_c = 100.0 * abs(c_vals[1] - c_vals[0]) / c_vals[0]
        rel_d = 100.0 * abs(d_vals[1] - d_vals[0]) / d_vals[0]
        ax1.text(
            0.97,
            0.96,
            rf"Local invariants:" "\n"
            rf"$\Delta \mathcal{{C}} = {rel_c:.2f}\%$" "\n"
            rf"$\Delta D = {rel_d:.2f}\%$",
            transform=ax1.transAxes,
            fontsize=8.2,
            va="top",
            ha="right",
            bbox=BBOX_STYLE,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel(r"Largest-cluster fraction $f_{\max}$ (%) at $6{,}000$ MCS")
    ax1.set_ylim(0.0, max(means) + max(sds) + 22.0)
    if ax2 is not None:
        ax1.set_title(r"(a) Finite-Size Split at $6{,}000$ MCS")

        # Panel (b): Direct time-resolved domain growth R_S(t) = 2*pi / k_1(t)
        l_styles = {
            20: (PALETTE["FCC"], "o", "-", r"$L = 20$ ($n = 15$)"),
            40: (PALETTE["SC"], "s", "-", r"$L = 40$ ($n = 5$)"),
        }
        fit_notes = []
        t_ref_vals = None
        r_ref_anchor = None
        for L_val in (20, 40):
            sub_c = df_coarsen[df_coarsen["L"] == L_val]
            if sub_c.empty:
                continue
            gt = (
                sub_c.groupby("t_mcs", as_index=False)
                .agg(R_mean=("R_S", "mean"), R_se=("R_S", _se))
                .sort_values("t_mcs")
            )
            col, mk, ls, lbl = l_styles[L_val]
            ax2.errorbar(
                gt["t_mcs"],
                gt["R_mean"],
                yerr=gt["R_se"],
                color=col,
                marker=mk,
                linestyle=ls,
                markersize=5.6,
                linewidth=1.7,
                capsize=2.8,
                label=lbl,
                zorder=4,
            )
            # Fit power law over t >= 1000 MCS
            sub_fit = gt[gt["t_mcs"] >= 1000]
            if len(sub_fit) >= 3:
                lt = np.log(sub_fit["t_mcs"].to_numpy(dtype=np.float64))
                lr = np.log(sub_fit["R_mean"].to_numpy(dtype=np.float64))
                z_hat, b_hat = np.polyfit(lt, lr, 1)
                # Per-seed SE
                s_slopes = [
                    float(np.polyfit(np.log(sd["t_mcs"].to_numpy(dtype=np.float64)), np.log(sd["R_S"].to_numpy(dtype=np.float64)), 1)[0])
                    for _, sd in sub_c[sub_c["t_mcs"] >= 1000].groupby("seed_idx")
                    if len(sd) >= 3
                ]
                z_se = float(np.std(s_slopes, ddof=1) / np.sqrt(len(s_slopes))) if len(s_slopes) > 1 else 0.0
                fit_notes.append(rf"$L={L_val}$: $z_{{\mathrm{{eff}}}} = {z_hat:.3f} \pm {z_se:.3f}$")
                if L_val == 40:
                    t_ref_vals = np.geomspace(1000.0, 10000.0, 60)
                    r_ref_anchor = float(sub_fit["R_mean"].iloc[0])

        if t_ref_vals is not None and r_ref_anchor is not None:
            # LSW asymptotic t^(1/3) slope guide line offset slightly above L=40 curve
            r_lsw = (r_ref_anchor * 1.18) * ((t_ref_vals / 1000.0) ** (1.0 / 3.0))
            ax2.plot(
                t_ref_vals,
                r_lsw,
                color="#333333",
                linestyle="--",
                linewidth=1.35,
                label=r"LSW asymptotic slope $\propto t^{1/3}$",
                zorder=2,
            )

        if fit_notes:
            ax2.text(
                0.97,
                0.05,
                "Power-law fit ($t \\geq 1{,}000$ MCS):\n" + "\n".join(fit_notes),
                transform=ax2.transAxes,
                fontsize=8.2,
                va="bottom",
                ha="right",
                bbox=BBOX_STYLE,
            )

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(r"Monte Carlo time $t$ (MCS)")
        ax2.set_ylabel(r"Characteristic domain length $R_S(t) = 2\pi / k_1(t)$ ($a$)")
        ax2.set_title(r"(b) Conserved Domain Coarsening Kinetics")
        ax2.legend(loc="upper left", fontsize=8.3, frameon=True)

    fig.tight_layout(pad=1.2)
    _save_fig(fig, out_dir, "fig08_finite_size_fcc", pad_inches=0.15)


def plot_fig09_two_param_master_curve(df_seed: pd.DataFrame, out_dir: Path) -> None:
    """Figure 9: Two-panel empirical cross-lattice scaling under zero-free-parameter D/D_ref."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.55), sharey=True)
    sub = df_seed[(df_seed["D"] > 0) & (df_seed["Cv_over_pc"] > 0) & (df_seed["T_over_Tc"] > 0)].copy()

    sub["D_over_Dref"] = sub["D"] / sub["lattice"].map(DREF_MAP)
    y_vals = sub["D_over_Dref"].to_numpy(dtype=np.float64)
    ln_y = np.log(y_vals)
    ln_cv = np.log(sub["Cv_over_pc"].to_numpy(dtype=np.float64))
    ln_t = np.log(sub["T_over_Tc"].to_numpy(dtype=np.float64))

    # Free two-parameter OLS
    X = np.column_stack([np.ones(len(sub)), ln_cv, ln_t])
    beta, _, _, _ = np.linalg.lstsq(X, ln_y, rcond=None)
    a_free, alpha_hat, beta_hat = float(beta[0]), float(beta[1]), float(beta[2])
    A_free = float(np.exp(a_free))
    M_free = (sub["Cv_over_pc"].to_numpy(dtype=np.float64) ** alpha_hat) * (
        sub["T_over_Tc"].to_numpy(dtype=np.float64) ** beta_hat
    )
    sub["M_free"] = M_free
    r2_free = 1.0 - float(np.sum((ln_y - (X @ beta)) ** 2)) / float(np.sum((ln_y - np.mean(ln_y)) ** 2))

    # Constrained rational (3/4, +1) model
    M_fix = (sub["Cv_over_pc"].to_numpy(dtype=np.float64) ** 0.75) * sub["T_over_Tc"].to_numpy(dtype=np.float64)
    sub["M_fix"] = M_fix
    ln_M_fix = np.log(M_fix)
    a_fix = float(np.mean(ln_y - ln_M_fix))
    A_fix = float(np.exp(a_fix))
    r2_fix = 1.0 - float(np.sum((ln_y - (a_fix + ln_M_fix)) ** 2)) / float(np.sum((ln_y - np.mean(ln_y)) ** 2))

    # Grouped out-of-sample CV metrics
    cv_res = _run_grouped_cv(sub, y_vals)
    loco_free = cv_res["LOCO_31fold"]["free_fit"]["R2_log"]
    lolo_free = cv_res["LOLO_3fold_overall"]["free_fit"]["R2_log"]
    loco_fix = cv_res["LOCO_31fold"]["fixed_3_4_and_1"]["R2_log"]
    lolo_fix = cv_res["LOLO_3fold_overall"]["fixed_3_4_and_1"]["R2_log"]

    # Panel (a): Free-exponent fit
    for lat in ("SC", "BCC", "FCC"):
        d = sub[sub["lattice"] == lat]
        ax1.scatter(
            d["M_free"], d["D_over_Dref"], s=20, alpha=0.76,
            color=PALETTE[lat], marker=MARKERS[lat], edgecolors="none", label=lat,
        )
    mg1 = np.geomspace(M_free.min() * 0.94, M_free.max() * 1.06, 150)
    ax1.plot(
        mg1, A_free * mg1, color="#222222", linestyle="--", linewidth=1.6,
        label=rf"Free fit: ${A_free:.4f}\,M_{{\mathrm{{free}}}}$",
    )
    ax1.text(
        0.97,
        0.04,
        f"$n = {len(sub)}$ seeds ($D_{{\\mathrm{{ref}}}} = \\frac{{1}}{{3}}f_0 r_{{\\mathrm{{nn}}}}^2$)\n"
        rf"$\alpha = {alpha_hat:.3f}$, $\beta = {beta_hat:.3f}$" "\n"
        rf"In-sample $R^2_{{\log}} = {r2_free:.4f}$" "\n"
        rf"LOCO-CV $R^2 = {loco_free:.4f}$, LOLO-CV $= {lolo_free:.4f}$",
        transform=ax1.transAxes,
        fontsize=8.0,
        va="bottom",
        ha="right",
        bbox=BBOX_STYLE,
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(rf"Free scaling variable $M_{{\mathrm{{free}}}} = (C_v / p_c)^{{{alpha_hat:.3f}}}(T / T_c)^{{{beta_hat:.3f}}}$")
    ax1.set_ylabel(r"Analytical reduced diffusivity $D / D_{\mathrm{ref}}$")
    ax1.set_title(r"(a) Free-Exponent Empirical Scaling")
    ax1.legend(loc="upper left", fontsize=8.3, frameon=True)

    # Panel (b): Constrained rational (3/4, +1) model
    for lat in ("SC", "BCC", "FCC"):
        d = sub[sub["lattice"] == lat]
        ax2.scatter(
            d["M_fix"], d["D_over_Dref"], s=20, alpha=0.76,
            color=PALETTE[lat], marker=MARKERS[lat], edgecolors="none", label=lat,
        )
    mg2 = np.geomspace(M_fix.min() * 0.94, M_fix.max() * 1.06, 150)
    ax2.plot(
        mg2, A_fix * mg2, color="#222222", linestyle="-.", linewidth=1.6,
        label=rf"Constrained: ${A_fix:.4f}\,M$",
    )
    ax2.text(
        0.97,
        0.04,
        "Constrained exponents:\n"
        r"$\alpha = 3/4$, $\beta = +1$" "\n"
        rf"In-sample $R^2_{{\log}} = {r2_fix:.4f}$" "\n"
        rf"LOCO-CV $R^2 = {loco_fix:.4f}$, LOLO-CV $= {lolo_fix:.4f}$",
        transform=ax2.transAxes,
        fontsize=8.0,
        va="bottom",
        ha="right",
        bbox=BBOX_STYLE,
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Constrained scaling variable $M = (C_v / p_c)^{3/4}(T / T_c)$")
    ax2.set_title(r"(b) Constrained Rational Scaling ($3/4,\,+1$)")
    ax2.legend(loc="upper left", fontsize=8.3, frameon=True)

    fig.tight_layout(pad=1.2)
    _save_fig(fig, out_dir, "fig09_two_param_master_curve", pad_inches=0.15)


def plot_fig10_single_param_lambda_transport(df_seed: pd.DataFrame, out_dir: Path) -> None:
    """Figure 10: Single-parameter reduced-variable test D/D_ref vs Lambda."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    sub = df_seed[(df_seed["D"] > 0) & (df_seed["Lambda"] > 0)].copy()
    sub["D_over_Dref"] = sub["D"] / sub["lattice"].map(DREF_MAP)

    slope_lines = []
    for lat in ("SC", "BCC", "FCC"):
        d = sub[sub["lattice"] == lat]
        if d.empty:
            continue
        ax.scatter(
            d["Lambda"],
            d["D_over_Dref"],
            s=22,
            alpha=0.78,
            color=PALETTE[lat],
            marker=MARKERS[lat],
            edgecolors="none",
            label=lat,
        )
        lx = np.log(d["Lambda"].to_numpy(dtype=np.float64))
        ly = np.log(d["D_over_Dref"].to_numpy(dtype=np.float64))
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
    ax.set_xlabel(r"Single reduced variable $\Lambda = (C_v / p_c)(T_c / T)$", labelpad=6)
    ax.set_ylabel(r"Analytical reduced diffusivity $D / D_{\mathrm{ref}}$")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout(pad=1.2)
    _save_fig(fig, out_dir, "fig10_single_param_lambda_transport", pad_inches=0.16)


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
    j0_path = data_dir / "random_percolation_j0.csv"
    fcc_path = data_dir / "fcc_concentration_sweep.csv"
    ctrl_path = data_dir / "thermo_controls.csv"
    fs_path = data_dir / "finite_size_fcc.csv"
    coarsen_path = data_dir / "coarsening_kinetics.csv"

    df_j0 = pd.read_csv(j0_path) if j0_path.exists() else None
    df_coarsen = pd.read_csv(coarsen_path) if coarsen_path.exists() else None

    if seed_path.exists():
        df_seed = pd.read_csv(seed_path)
        df_span = pd.read_csv(dense_span_path) if dense_span_path.exists() else df_seed
        plot_fig02_diffusivity_vs_cv(df_seed, figures_dir)
        plot_fig03_normalized_diffusivity(df_seed, figures_dir)
        plot_fig05_spanning_crossover(df_span, figures_dir, df_j0=df_j0)
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
        plot_fig08_finite_size_fcc(df_fs, figures_dir, df_coarsen=df_coarsen)

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
