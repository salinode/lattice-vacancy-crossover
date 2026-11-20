"""Statistical analysis and scaling diagnostics for the Kawasaki MC ensemble.

Computes and exports `data/summary_stats.json`:
  1. Dilute-limit D/Cv convergence checks and diffusive MSD slope QA.
  2. Interpolated P_span = 0.5 crossing concentrations Cv* and Cv* / pc
     for SC, BCC, and FCC at matched subcritical reduced temperature T/Tc = 0.82.
  3. Matched reduced-temperature control comparisons (T/Tc = 0.82 vs 1.25).
  4. Finite-size FCC L=20 vs L=40 morphology comparison.
  5. Pearson r, Spearman rho, and shared-slope ANCOVA F-tests for
     D vs f_max and D vs C.
  6. Single-parameter scaling slopes in ln(D/D0) vs ln(Lambda), where
     Lambda = (Cv / pc) * (Tc / T).
  7. Two-parameter OLS regression ln(D/D0) = a + alpha * ln(Cv/pc) + beta * ln(T/Tc),
     fixed-exponent (alpha=3/4, beta=1) R^2, shared-vs-free exponent F-test,
     and comparison against f_max scaling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.lattices import LATTICE_SPECS


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    f = float(val)
    return f if np.isfinite(f) else None


def _ols_fit(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Ordinary least squares y = X @ beta with parameter SEs, RSS, and R^2."""
    n, p = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    rss = float(np.dot(resid, resid))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0.0 else float("nan")
    dof = n - p
    if dof > 0:
        sigma2 = rss / float(dof)
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    else:
        se = np.full(p, np.nan)
    return {
        "beta": beta,
        "se": se,
        "rss": rss,
        "tss": tss,
        "r2": r2,
        "n": int(n),
        "p": int(p),
        "dof": int(dof),
    }


def _nested_f_test(rss_red: float, p_red: int, rss_full: float, p_full: int, n: int) -> Dict[str, Optional[float]]:
    """Compute nested ANOVA F-statistic comparing reduced vs full linear models."""
    df1 = p_full - p_red
    df2 = n - p_full
    if df1 <= 0 or df2 <= 0 or rss_full <= 0.0:
        return {"F": None, "df1": df1, "df2": df2, "p_value": None}
    num = max(0.0, rss_red - rss_full) / float(df1)
    den = rss_full / float(df2)
    f_stat = num / den
    p_val = float(stats.f.sf(f_stat, df1, df2))
    return {"F": float(f_stat), "df1": int(df1), "df2": int(df2), "p_value": p_val}


def shared_slope_ancova(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    log_transform: bool = True,
) -> Dict[str, Any]:
    """Compare separate-intercept shared-slope vs separate-slope models across lattices.

    Reduced model: y = mu_g + gamma * x         (G + 1 parameters)
    Full model:    y = mu_g + gamma_g * x       (2 * G parameters)
    For G = 3 lattices and n = 465 rows, yields F(2, 459).
    """
    sub = df[["lattice", x_col, y_col]].dropna().copy()
    if log_transform:
        sub = sub[(sub[x_col] > 0) & (sub[y_col] > 0)]
        x = np.log(sub[x_col].to_numpy(dtype=np.float64))
        y = np.log(sub[y_col].to_numpy(dtype=np.float64))
    else:
        x = sub[x_col].to_numpy(dtype=np.float64)
        y = sub[y_col].to_numpy(dtype=np.float64)

    groups = sorted(sub["lattice"].unique().tolist())
    n = len(sub)
    G = len(groups)
    if n <= 2 * G or G < 2:
        return {"F": None, "df1": None, "df2": None, "p_value": None, "per_lattice_slopes": {}}

    # Reduced design matrix: [I_g1, I_g2, ..., I_gG, x]
    X_red = np.zeros((n, G + 1), dtype=np.float64)
    # Full design matrix: [I_g1, ..., I_gG, I_g1*x, ..., I_gG*x]
    X_full = np.zeros((n, 2 * G), dtype=np.float64)
    lat_vals = sub["lattice"].to_numpy()
    for gi, g in enumerate(groups):
        mask = (lat_vals == g).astype(np.float64)
        X_red[:, gi] = mask
        X_full[:, gi] = mask
        X_full[:, G + gi] = mask * x
    X_red[:, G] = x

    fit_red = _ols_fit(X_red, y)
    fit_full = _ols_fit(X_full, y)
    ftest = _nested_f_test(fit_red["rss"], fit_red["p"], fit_full["rss"], fit_full["p"], n)

    per_lat = {}
    for gi, g in enumerate(groups):
        per_lat[g] = {
            "slope": _safe_float(fit_full["beta"][G + gi]),
            "slope_se": _safe_float(fit_full["se"][G + gi]),
            "intercept": _safe_float(fit_full["beta"][gi]),
        }

    return {
        **ftest,
        "shared_slope": _safe_float(fit_red["beta"][G]),
        "shared_slope_se": _safe_float(fit_red["se"][G]),
        "per_lattice_slopes": per_lat,
    }


def interpolate_spanning_crossing(
    cv_vals: np.ndarray,
    pspan_vals: np.ndarray,
    target: float = 0.5,
) -> Optional[float]:
    """Linearly interpolate the upward Cv* crossing where P_span(Cv) rises through `target`."""
    order = np.argsort(cv_vals)
    x = np.asarray(cv_vals, dtype=np.float64)[order]
    y = np.asarray(pspan_vals, dtype=np.float64)[order]

    # Prefer upward crossings where y[i] <= target <= y[i + 1] with y[i + 1] > y[i].
    # Taking the last such interval ignores low-Cv finite-sample fluctuations.
    upward_crossings = []
    for i in range(len(x) - 1):
        if y[i] <= target <= y[i + 1] and y[i + 1] > y[i]:
            frac = (target - y[i]) / (y[i + 1] - y[i])
            upward_crossings.append(float(x[i] + frac * (x[i + 1] - x[i])))
    if upward_crossings:
        return upward_crossings[-1]

    # Fallback: exact hit if curve stays flat at target
    exact = np.where(np.isclose(y, target, atol=1e-12))[0]
    if len(exact) > 0:
        return float(x[exact[-1]])

    return None


def analyze_dilute_convergence(df_seed: pd.DataFrame) -> Dict[str, Any]:
    """Compute dilute-limit D/Cv convergence checks and diffusive MSD QA."""
    per_lattice: Dict[str, Any] = {}
    for lat in ("SC", "BCC", "FCC"):
        sub = df_seed[df_seed["lattice"] == lat]
        if sub.empty:
            continue
        by_cv = (
            sub.groupby("Cv", as_index=False)
            .agg(
                n=("D", "count"),
                D_mean=("D", "mean"),
                D_se=("D", lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0),
                D_over_Cv_mean=("D_over_Cv", "mean"),
                D_over_Cv_se=("D_over_Cv", lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0),
            )
            .sort_values("Cv")
        )
        records = by_cv.to_dict(orient="records")
        lowest = records[0]
        highest = records[-1]
        rel_drop = (
            (lowest["D_over_Cv_mean"] - highest["D_over_Cv_mean"]) / lowest["D_over_Cv_mean"]
            if lowest["D_over_Cv_mean"] > 0
            else None
        )
        per_lattice[lat] = {
            "by_Cv": records,
            "lowest_Cv": float(lowest["Cv"]),
            "highest_Cv": float(highest["Cv"]),
            "D_over_Cv_relative_change_low_to_high": _safe_float(rel_drop),
        }

    slopes = df_seed["msd_loglog_slope"].dropna().to_numpy(dtype=np.float64)
    in_bounds = np.mean((slopes >= 0.95) & (slopes <= 1.05)) if len(slopes) > 0 else float("nan")
    return {
        "per_lattice": per_lattice,
        "diffusive_qa": {
            "n_total": int(len(df_seed)),
            "mean_loglog_slope": _safe_float(np.mean(slopes)) if len(slopes) > 0 else None,
            "sd_loglog_slope": _safe_float(np.std(slopes, ddof=1)) if len(slopes) > 1 else None,
            "min_loglog_slope": _safe_float(np.min(slopes)) if len(slopes) > 0 else None,
            "max_loglog_slope": _safe_float(np.max(slopes)) if len(slopes) > 0 else None,
            "fraction_in_095_105": _safe_float(in_bounds),
        },
    }


def analyze_spanning_crossings(df_seed: pd.DataFrame) -> Dict[str, Any]:
    """Compute P_span = 0.5 crossings Cv* and Cv* / pc at matched T/Tc = 0.82."""
    subcritical_T = {"SC": 0.93, "BCC": 1.30, "FCC": 2.00}
    out: Dict[str, Any] = {}

    for lat, T_sub in subcritical_T.items():
        sub = df_seed[(df_seed["lattice"] == lat) & (np.isclose(df_seed["T"], T_sub, atol=0.02))]
        if sub.empty:
            continue
        grp = sub.groupby("Cv", as_index=False)["is_spanning"].mean().sort_values("Cv")
        cv_star = interpolate_spanning_crossing(
            grp["Cv"].to_numpy(), grp["is_spanning"].to_numpy(), target=0.5
        )
        pc = float(LATTICE_SPECS[lat]["pc"])
        out[lat] = {
            "T": T_sub,
            "T_over_Tc": round(T_sub / float(LATTICE_SPECS[lat]["Tc"]), 4),
            "pc": pc,
            "Cv_star": _safe_float(cv_star),
            "Cv_star_over_pc": _safe_float(cv_star / pc) if cv_star is not None else None,
            "curve": [
                {"Cv": float(r["Cv"]), "P_span": float(r["is_spanning"])}
                for _, r in grp.iterrows()
            ],
        }

    # Also compute FCC supercritical reference curve (T = 3.06)
    fcc_hi = df_seed[(df_seed["lattice"] == "FCC") & (np.isclose(df_seed["T"], 3.06, atol=0.02))]
    if not fcc_hi.empty:
        grp = fcc_hi.groupby("Cv", as_index=False)["is_spanning"].mean().sort_values("Cv")
        cv_star = interpolate_spanning_crossing(
            grp["Cv"].to_numpy(), grp["is_spanning"].to_numpy(), target=0.5
        )
        pc = float(LATTICE_SPECS["FCC"]["pc"])
        out["FCC_above_Tc"] = {
            "T": 3.06,
            "T_over_Tc": round(3.06 / float(LATTICE_SPECS["FCC"]["Tc"]), 4),
            "pc": pc,
            "Cv_star": _safe_float(cv_star),
            "Cv_star_over_pc": _safe_float(cv_star / pc) if cv_star is not None else None,
            "curve": [
                {"Cv": float(r["Cv"]), "P_span": float(r["is_spanning"])}
                for _, r in grp.iterrows()
            ],
        }

    return out


def analyze_structure_transport(df_seed: pd.DataFrame) -> Dict[str, Any]:
    """Compute Pearson r, Spearman rho, and shared-slope ANCOVA F-tests."""
    out: Dict[str, Any] = {}
    for name, col in (("D_vs_f_max", "f_max"), ("D_vs_C_index", "C_index")):
        sub = df_seed[["lattice", "D", col]].dropna()
        x = sub[col].to_numpy(dtype=np.float64)
        y = sub["D"].to_numpy(dtype=np.float64)

        r_lin, p_lin = stats.pearsonr(x, y) if len(sub) >= 3 else (np.nan, np.nan)
        rho, p_rho = stats.spearmanr(x, y) if len(sub) >= 3 else (np.nan, np.nan)

        pos = (x > 0) & (y > 0)
        if np.count_nonzero(pos) >= 3:
            lx = np.log(x[pos])
            ly = np.log(y[pos])
            r_log, p_log = stats.pearsonr(lx, ly)
            pooled_fit = _ols_fit(np.column_stack([np.ones_like(lx), lx]), ly)
            pooled_slope = float(pooled_fit["beta"][1])
            pooled_intercept = float(pooled_fit["beta"][0])
            pooled_r2 = float(pooled_fit["r2"])
        else:
            r_log = p_log = pooled_slope = pooled_intercept = pooled_r2 = np.nan

        per_lattice_corr = {}
        for lat in ("SC", "BCC", "FCC"):
            s = sub[sub["lattice"] == lat]
            if len(s) >= 3:
                rl, pl = stats.pearsonr(s[col], s["D"])
                rs, ps = stats.spearmanr(s[col], s["D"])
                per_lattice_corr[lat] = {
                    "n": int(len(s)),
                    "pearson_r": _safe_float(rl),
                    "pearson_p": _safe_float(pl),
                    "spearman_rho": _safe_float(rs),
                    "spearman_p": _safe_float(ps),
                }

        out[name] = {
            "n": int(len(sub)),
            "pearson_r": _safe_float(r_lin),
            "pearson_p": _safe_float(p_lin),
            "spearman_rho": _safe_float(rho),
            "spearman_p": _safe_float(p_rho),
            "loglog_pearson_r": _safe_float(r_log),
            "loglog_pearson_p": _safe_float(p_log),
            "loglog_pooled_slope": _safe_float(pooled_slope),
            "loglog_pooled_intercept": _safe_float(pooled_intercept),
            "loglog_pooled_r2": _safe_float(pooled_r2),
            "per_lattice_correlations": per_lattice_corr,
            "shared_slope_ancova_loglog": shared_slope_ancova(sub, col, "D", log_transform=True),
        }

    return out


def analyze_reduced_scaling(df_seed: pd.DataFrame) -> Dict[str, Any]:
    """Single-parameter Lambda test and two-parameter reduced-variable transport scaling."""
    sub = df_seed[(df_seed["D_over_D0"] > 0) & (df_seed["Cv_over_pc"] > 0) & (df_seed["T_over_Tc"] > 0)].copy()

    ln_D_D0 = np.log(sub["D_over_D0"].to_numpy(dtype=np.float64))
    ln_Lam = np.log(sub["Lambda"].to_numpy(dtype=np.float64))
    ln_Cv_pc = np.log(sub["Cv_over_pc"].to_numpy(dtype=np.float64))
    ln_T_Tc = np.log(sub["T_over_Tc"].to_numpy(dtype=np.float64))

    # 1. Single-parameter scaling: ln(D/D0) vs ln(Lambda)
    single_param: Dict[str, Any] = {}
    pooled_lam = _ols_fit(np.column_stack([np.ones_like(ln_Lam), ln_Lam]), ln_D_D0)
    single_param["pooled"] = {
        "intercept": _safe_float(pooled_lam["beta"][0]),
        "slope": _safe_float(pooled_lam["beta"][1]),
        "slope_se": _safe_float(pooled_lam["se"][1]),
        "R2": _safe_float(pooled_lam["r2"]),
    }
    single_param["per_lattice"] = {}
    for lat in ("SC", "BCC", "FCC"):
        mask = (sub["lattice"] == lat).to_numpy()
        if np.count_nonzero(mask) >= 3:
            fit_l = _ols_fit(np.column_stack([np.ones(np.count_nonzero(mask)), ln_Lam[mask]]), ln_D_D0[mask])
            single_param["per_lattice"][lat] = {
                "n": int(np.count_nonzero(mask)),
                "intercept": _safe_float(fit_l["beta"][0]),
                "slope": _safe_float(fit_l["beta"][1]),
                "slope_se": _safe_float(fit_l["se"][1]),
                "R2": _safe_float(fit_l["r2"]),
            }

    # 2. Two-parameter OLS: ln(D/D0) = a + alpha * ln(Cv/pc) + beta * ln(T/Tc)
    X_pooled = np.column_stack([np.ones_like(ln_Cv_pc), ln_Cv_pc, ln_T_Tc])
    fit_pooled = _ols_fit(X_pooled, ln_D_D0)
    a_hat, alpha_hat, beta_hat = fit_pooled["beta"]
    a_se, alpha_se, beta_se = fit_pooled["se"]

    # Fixed-exponent model: alpha = 3/4, beta_T_over_Tc = +1  (equiv. beta_Tc_over_T = -1)
    # ln(D/D0) = a_fixed + 0.75 * ln(Cv/pc) + 1.0 * ln(T/Tc)
    ln_M = 0.75 * ln_Cv_pc + 1.0 * ln_T_Tc
    a_fixed = float(np.mean(ln_D_D0 - ln_M))
    pred_fixed = a_fixed + ln_M
    rss_fixed = float(np.sum((ln_D_D0 - pred_fixed) ** 2))
    tss_log = float(np.sum((ln_D_D0 - np.mean(ln_D_D0)) ** 2))
    r2_fixed_log = 1.0 - rss_fixed / tss_log if tss_log > 0 else float("nan")

    # Also compute linear-space R^2 of D/D0 = A * (Cv/pc)^(3/4) * (T/Tc)
    y_lin = sub["D_over_D0"].to_numpy(dtype=np.float64)
    A_fixed = float(np.exp(a_fixed))
    y_lin_pred = A_fixed * np.exp(ln_M)
    r2_fixed_linear = 1.0 - float(np.sum((y_lin - y_lin_pred) ** 2)) / float(np.sum((y_lin - np.mean(y_lin)) ** 2))

    # Shared-vs-free exponent F-test across the 3 lattices:
    # Reduced: 3 params (a, alpha, beta) pooled; Full: 9 params (a_g, alpha_g, beta_g) per lattice
    # For n = 465 rows and 3 lattices -> F(6, 456)
    groups = sorted(sub["lattice"].unique().tolist())
    G = len(groups)
    n = len(sub)
    lat_arr = sub["lattice"].to_numpy()
    X_free = np.zeros((n, 3 * G), dtype=np.float64)
    per_lat_two_param: Dict[str, Any] = {}
    for gi, g in enumerate(groups):
        m = (lat_arr == g).astype(np.float64)
        X_free[:, 3 * gi] = m
        X_free[:, 3 * gi + 1] = m * ln_Cv_pc
        X_free[:, 3 * gi + 2] = m * ln_T_Tc

    fit_free = _ols_fit(X_free, ln_D_D0)
    shared_vs_free_f = _nested_f_test(fit_pooled["rss"], fit_pooled["p"], fit_free["rss"], fit_free["p"], n)
    for gi, g in enumerate(groups):
        m_bool = (lat_arr == g)
        X_g = np.column_stack([np.ones(np.count_nonzero(m_bool)), ln_Cv_pc[m_bool], ln_T_Tc[m_bool]])
        fit_g = _ols_fit(X_g, ln_D_D0[m_bool])
        per_lat_two_param[g] = {
            "n": int(np.count_nonzero(m_bool)),
            "a": _safe_float(fit_g["beta"][0]),
            "prefactor": _safe_float(np.exp(fit_g["beta"][0])),
            "alpha": _safe_float(fit_g["beta"][1]),
            "alpha_se": _safe_float(fit_g["se"][1]),
            "beta_T_over_Tc": _safe_float(fit_g["beta"][2]),
            "beta_Tc_over_T": _safe_float(-fit_g["beta"][2]),
            "beta_se": _safe_float(fit_g["se"][2]),
            "R2_log": _safe_float(fit_g["r2"]),
        }

    # Zero-free-parameter analytical scale: D_ref = (1/3) * f0 * r_nn^2
    dref_map = {
        k: (1.0 / 3.0) * float(v["f0_dilute"]) * float(v["nn_dist_sq"])
        for k, v in LATTICE_SPECS.items()
    }
    d_over_dref = (sub["D"] / sub["lattice"].map(dref_map)).to_numpy(dtype=np.float64)
    ln_D_Dref = np.log(d_over_dref)
    fit_ref = _ols_fit(X_pooled, ln_D_Dref)
    a_ref_fix = float(np.mean(ln_D_Dref - ln_M))
    tss_ref = float(np.sum((ln_D_Dref - np.mean(ln_D_Dref)) ** 2))
    r2_ref_fix_log = 1.0 - float(np.sum((ln_D_Dref - (a_ref_fix + ln_M)) ** 2)) / tss_ref
    r2_ref_fix_lin = 1.0 - float(np.sum((d_over_dref - np.exp(a_ref_fix) * np.exp(ln_M)) ** 2)) / float(
        np.sum((d_over_dref - np.mean(d_over_dref)) ** 2)
    )

    # Matched two-temperature subset (n = 420, excluding SC T/Tc = 1.77)
    m420 = ~((sub["lattice"] == "SC") & (sub["T"] > 1.8)).to_numpy()
    fit_420_d0 = _ols_fit(X_pooled[m420], ln_D_D0[m420])
    a420_fix = float(np.mean(ln_D_D0[m420] - ln_M[m420]))
    r2_420_fix_log = 1.0 - float(np.sum((ln_D_D0[m420] - (a420_fix + ln_M[m420])) ** 2)) / float(
        np.sum((ln_D_D0[m420] - np.mean(ln_D_D0[m420])) ** 2)
    )
    fit_420_ref = _ols_fit(X_pooled[m420], ln_D_Dref[m420])
    a420_ref_fix = float(np.mean(ln_D_Dref[m420] - ln_M[m420]))
    r2_420_ref_fix_log = 1.0 - float(np.sum((ln_D_Dref[m420] - (a420_ref_fix + ln_M[m420])) ** 2)) / float(
        np.sum((ln_D_Dref[m420] - np.mean(ln_D_Dref[m420])) ** 2)
    )

    # Comparison: two-parameter regression on ln(f_max)
    sub_f = sub[sub["f_max"] > 0]
    ln_fmax = np.log(sub_f["f_max"].to_numpy(dtype=np.float64))
    X_fmax = np.column_stack([
        np.ones(len(sub_f)),
        np.log(sub_f["Cv_over_pc"].to_numpy(dtype=np.float64)),
        np.log(sub_f["T_over_Tc"].to_numpy(dtype=np.float64)),
    ])
    fit_fmax = _ols_fit(X_fmax, ln_fmax)

    return {
        "single_parameter_Lambda": single_param,
        "two_parameter_transport": {
            "n": int(n),
            "free_fit": {
                "a": _safe_float(a_hat),
                "a_se": _safe_float(a_se),
                "prefactor": _safe_float(np.exp(a_hat)),
                "alpha": _safe_float(alpha_hat),
                "alpha_se": _safe_float(alpha_se),
                "beta_T_over_Tc": _safe_float(beta_hat),
                "beta_Tc_over_T": _safe_float(-beta_hat),
                "beta_se": _safe_float(beta_se),
                "R2": _safe_float(fit_pooled["r2"]),
            },
            "fixed_exponent_3_4_and_1": {
                "a_fixed": _safe_float(a_fixed),
                "prefactor": _safe_float(A_fixed),
                "alpha_fixed": 0.75,
                "beta_T_over_Tc_fixed": 1.0,
                "beta_Tc_over_T_fixed": -1.0,
                "R2_log": _safe_float(r2_fixed_log),
                "R2_linear": _safe_float(r2_fixed_linear),
            },
            "shared_vs_free_exponent_F_test": shared_vs_free_f,
            "per_lattice_free_fits": per_lat_two_param,
            "zero_free_param_Dref": {
                "n": int(n),
                "Dref_map": dref_map,
                "free_fit": {
                    "a": _safe_float(fit_ref["beta"][0]),
                    "prefactor": _safe_float(np.exp(fit_ref["beta"][0])),
                    "alpha": _safe_float(fit_ref["beta"][1]),
                    "alpha_se": _safe_float(fit_ref["se"][1]),
                    "beta_T_over_Tc": _safe_float(fit_ref["beta"][2]),
                    "beta_Tc_over_T": _safe_float(-fit_ref["beta"][2]),
                    "beta_se": _safe_float(fit_ref["se"][2]),
                    "R2_log": _safe_float(fit_ref["r2"]),
                },
                "fixed_exponent_3_4_and_1": {
                    "prefactor": _safe_float(np.exp(a_ref_fix)),
                    "R2_log": _safe_float(r2_ref_fix_log),
                    "R2_linear": _safe_float(r2_ref_fix_lin),
                },
            },
            "matched_two_temp_420": {
                "n": int(np.count_nonzero(m420)),
                "D_over_D0": {
                    "prefactor_free": _safe_float(np.exp(fit_420_d0["beta"][0])),
                    "alpha": _safe_float(fit_420_d0["beta"][1]),
                    "alpha_se": _safe_float(fit_420_d0["se"][1]),
                    "beta_T_over_Tc": _safe_float(fit_420_d0["beta"][2]),
                    "beta_se": _safe_float(fit_420_d0["se"][2]),
                    "R2_log_free": _safe_float(fit_420_d0["r2"]),
                    "prefactor_fixed": _safe_float(np.exp(a420_fix)),
                    "R2_log_fixed": _safe_float(r2_420_fix_log),
                },
                "D_over_Dref": {
                    "prefactor_free": _safe_float(np.exp(fit_420_ref["beta"][0])),
                    "alpha": _safe_float(fit_420_ref["beta"][1]),
                    "alpha_se": _safe_float(fit_420_ref["se"][1]),
                    "beta_T_over_Tc": _safe_float(fit_420_ref["beta"][2]),
                    "beta_se": _safe_float(fit_420_ref["se"][2]),
                    "R2_log_free": _safe_float(fit_420_ref["r2"]),
                    "prefactor_fixed": _safe_float(np.exp(a420_ref_fix)),
                    "R2_log_fixed": _safe_float(r2_420_ref_fix_log),
                },
            },
        },
        "two_parameter_f_max_comparison": {
            "n": int(len(sub_f)),
            "a": _safe_float(fit_fmax["beta"][0]),
            "alpha": _safe_float(fit_fmax["beta"][1]),
            "beta_T_over_Tc": _safe_float(fit_fmax["beta"][2]),
            "R2": _safe_float(fit_fmax["r2"]),
        },
    }


def analyze_thermo_controls(df_ctrl: pd.DataFrame) -> Dict[str, Any]:
    """Summarize matched T/Tc = 0.82 vs 1.25 controls from thermo_controls.csv."""
    summary: List[Dict[str, Any]] = []
    for (lat, cv), grp in df_ctrl.groupby(["lattice", "Cv"]):
        temps = sorted(grp["T"].unique().tolist())
        entry: Dict[str, Any] = {"lattice": str(lat), "Cv": float(cv), "conditions": {}}
        vals_by_T = {}
        for T_val in temps:
            s = grp[np.isclose(grp["T"], T_val)]
            tag = "below_Tc" if float(T_val) < float(LATTICE_SPECS[str(lat)]["Tc"]) else "above_Tc"
            vals_by_T[tag] = s["f_max_pct"].to_numpy(dtype=np.float64)
            entry["conditions"][tag] = {
                "T": float(T_val),
                "T_over_Tc": float(s["T_over_Tc"].mean()),
                "n": int(len(s)),
                "f_max_pct_mean": _safe_float(s["f_max_pct"].mean()),
                "f_max_pct_sd": _safe_float(s["f_max_pct"].std(ddof=1)) if len(s) > 1 else 0.0,
                "P_span": _safe_float(s["is_spanning"].mean()),
                "C_index_mean": _safe_float(s["C_index"].mean()),
                "D_mean": _safe_float(s["D"].mean()),
            }
        if "below_Tc" in vals_by_T and "above_Tc" in vals_by_T and len(vals_by_T["below_Tc"]) > 1:
            t_stat, p_val = stats.ttest_ind(vals_by_T["below_Tc"], vals_by_T["above_Tc"], equal_var=False)
            entry["welch_t_test_f_max"] = {"t": _safe_float(t_stat), "p_value": _safe_float(p_val)}
        summary.append(entry)
    return {"controls": summary}


def analyze_finite_size(df_fs: pd.DataFrame) -> Dict[str, Any]:
    """Summarize FCC Cv=0.20, T=2.00 finite-size comparison (L=20 vs L=40)."""
    out: Dict[str, Any] = {}
    for L_val, grp in df_fs.groupby("L"):
        out[f"L={int(L_val)}"] = {
            "L": int(L_val),
            "N": int(grp["N"].iloc[0]),
            "n": int(len(grp)),
            "sweeps": int(grp["sweeps"].iloc[0]),
            "f_max_pct_mean": _safe_float(grp["f_max_pct"].mean()),
            "f_max_pct_sd": _safe_float(grp["f_max_pct"].std(ddof=1)) if len(grp) > 1 else 0.0,
            "P_span": _safe_float(grp["is_spanning"].mean()),
            "C_index_mean": _safe_float(grp["C_index"].mean()),
            "D_mean": _safe_float(grp["D"].mean()),
        }
    return out


def run_statistics(data_dir: Path, out_json: Path) -> Dict[str, Any]:
    """Run the complete statistical pipeline and save summary_stats.json."""
    seed_path = data_dir / "seed_level_465.csv"
    dense_span_path = data_dir / "spanning_dense_subcritical.csv"
    ctrl_path = data_dir / "thermo_controls.csv"
    fs_path = data_dir / "finite_size_fcc.csv"

    df_seed = pd.read_csv(seed_path)
    df_span = pd.read_csv(dense_span_path) if dense_span_path.exists() else df_seed
    results: Dict[str, Any] = {
        "dataset_rows": {
            "seed_level_465": int(len(df_seed)),
            "spanning_dense_subcritical": int(len(df_span)),
        },
        "dilute_convergence": analyze_dilute_convergence(df_seed),
        "spanning_crossings": analyze_spanning_crossings(df_span),
        "structure_transport": analyze_structure_transport(df_seed),
        "reduced_scaling": analyze_reduced_scaling(df_seed),
    }

    if ctrl_path.exists():
        df_ctrl = pd.read_csv(ctrl_path)
        results["dataset_rows"]["thermo_controls"] = int(len(df_ctrl))
        results["thermo_controls"] = analyze_thermo_controls(df_ctrl)

    if fs_path.exists():
        df_fs = pd.read_csv(fs_path)
        results["dataset_rows"]["finite_size_fcc"] = int(len(df_fs))
        results["finite_size_fcc"] = analyze_finite_size(df_fs)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute statistical summaries from simulation CSVs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Input CSV directory (default: data).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/summary_stats.json"),
        help="Output JSON path (default: data/summary_stats.json).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    stats_dict = run_statistics(args.data_dir, args.output)

    sc = stats_dict["spanning_crossings"]
    rs = stats_dict["reduced_scaling"]["two_parameter_transport"]
    ff = rs["free_fit"]
    fx = rs["fixed_exponent_3_4_and_1"]
    ft = rs["shared_vs_free_exponent_F_test"]

    print(f"Exported summary statistics -> {args.output}")
    print(f"  Rows analyzed: {stats_dict['dataset_rows']['seed_level_465']}")
    print("  Spanning crossings (P_span = 0.5 at T/Tc = 0.82):")
    for lat in ("SC", "BCC", "FCC"):
        if lat in sc:
            print(f"    {lat}: Cv* = {sc[lat]['Cv_star']}, Cv*/pc = {sc[lat]['Cv_star_over_pc']}")
    print(
        f"  Two-parameter transport scaling: alpha = {ff['alpha']:.4f} +/- {ff['alpha_se']:.4f}, "
        f"beta(T/Tc) = {ff['beta_T_over_Tc']:.4f} +/- {ff['beta_se']:.4f}, "
        f"R^2(free) = {ff['R2']:.4f}, R^2(fixed 3/4, 1) = {fx['R2_log']:.4f}"
    )
    if ft["F"] is not None:
        print(f"  Shared-vs-free exponent F-test: F({ft['df1']}, {ft['df2']}) = {ft['F']:.2f}, p = {ft['p_value']:.3e}")


if __name__ == "__main__":
    main()
