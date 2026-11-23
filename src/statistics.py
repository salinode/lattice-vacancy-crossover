"""Statistical analysis, uncertainty quantification, and scaling diagnostics.

Computes and exports `data/summary_stats.json`:
  1. Dilute-limit quantitative validation against analytical D_ref = (1/3) f0 r_nn^2
     (from dilute_validation.csv) and canonical 465-seed diffusive MSD QA.
  2. Non-interacting (J=0) finite-size random site-percolation controls
     (L=20 SC/BCC/FCC, L=30/40 FCC) with Wilson 95% binomial CIs and
     logistic-regression MLE crossings with 2,000-resample bootstrap 95% CIs.
  3. High-precision interacting (J=1) spanning crossovers (n=50 per Cv near
     P_span = 0.5) with Wilson 95% CIs, logistic-regression MLE Cv*, bootstrap
     95% CIs on Cv*, Cv*/pc, and Cv*(J=1)/Cv*(J=0), plus 3-way FCC Cv=0.20 comparison.
  4. Matched reduced-temperature controls (T/Tc = 0.82 vs 1.25) with Wilson CIs
     and Welch t-tests.
  5. Expanded finite-size FCC comparison (L=20 n=15 vs L=40 n=10) with SD/SE and
     direct time-resolved domain coarsening kinetics R_S(t) and R_int(t) power-law fits.
  6. Structure-transport linkage using BOTH final C and time-aligned production-window
     C_bar (over [2000, 6000] MCS), including controlled multiple OLS regressions
     adjusting for ln(Cv), ln(T/Tc), and lattice topology.
  7. Empirical cross-lattice transport scaling under BOTH zero-free-parameter
     analytical D_ref = (1/3) f0 r_nn^2 (primary) and empirical D_scale, with
     grouped out-of-sample cross-validation (LOCO-CV, 5-fold grouped CV, and
     Leave-One-Lattice-Out CV) and nested ANOVA variance decomposition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from src.lattices import LATTICE_SPECS


Z_95 = 1.959963984540054


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    f = float(val)
    return f if np.isfinite(f) else None


def wilson_ci(k: int, n: int, z: float = Z_95) -> Tuple[float, float]:
    """Exact Wilson score binomial confidence interval clamped to [0, 1]."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = float(k) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / float(n)
    center = (p + z2 / (2.0 * float(n))) / denom
    half = (z / denom) * np.sqrt(max(0.0, (p * (1.0 - p)) / float(n) + z2 / (4.0 * float(n) * float(n))))
    return (max(0.0, float(center - half)), min(1.0, float(center + half)))


def _ols_fit(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Ordinary least squares y = X @ beta with parameter SEs, t/p stats, RSS, and R^2."""
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
        with np.errstate(divide="ignore", invalid="ignore"):
            t_vals = np.where(se > 0, beta / se, np.nan)
            p_vals = np.where(np.isfinite(t_vals), 2.0 * stats.t.sf(np.abs(t_vals), dof), np.nan)
    else:
        se = np.full(p, np.nan)
        t_vals = np.full(p, np.nan)
        p_vals = np.full(p, np.nan)
    return {
        "beta": beta,
        "se": se,
        "t": t_vals,
        "p_vals": p_vals,
        "y_hat": y_hat,
        "resid": resid,
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
    """Compare separate-intercept shared-slope vs separate-slope models across lattices."""
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

    X_red = np.zeros((n, G + 1), dtype=np.float64)
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

    upward_crossings = []
    for i in range(len(x) - 1):
        if y[i] <= target <= y[i + 1] and y[i + 1] > y[i]:
            frac = (target - y[i]) / (y[i + 1] - y[i])
            upward_crossings.append(float(x[i] + frac * (x[i + 1] - x[i])))
    if upward_crossings:
        return upward_crossings[-1]

    exact = np.where(np.isclose(y, target, atol=1e-12))[0]
    if len(exact) > 0:
        return float(x[exact[-1]])

    return None


def _fit_binomial_logistic_mle(
    cv: np.ndarray, k: np.ndarray, n: np.ndarray
) -> Tuple[float, float, float]:
    """Fit exact binomial logistic MLE: logit P(Cv) = gamma0 + gamma1 * Cv.

    Returns (gamma0, gamma1, Cv_star = -gamma0 / gamma1).
    """
    cv = np.asarray(cv, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    mu = float(np.mean(cv))
    sd = float(np.std(cv))
    if sd <= 0.0:
        return (float("nan"), float("nan"), float("nan"))
    x = (cv - mu) / sd

    def nll_and_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        eta = theta[0] + theta[1] * x
        # Numerically stable log(1 + exp(eta))
        log1pexp = np.where(eta > 0.0, eta + np.log1p(np.exp(-eta)), np.log1p(np.exp(eta)))
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -40.0, 40.0)))
        nll = float(np.sum(n * log1pexp - k * eta))
        # Tiny L2 guard only active under quasi-complete separation (1e-10 doesn't move normal MLE)
        nll += 1e-10 * float(theta[0] ** 2 + theta[1] ** 2)
        diff = n * p - k
        g0 = float(np.sum(diff)) + 2e-10 * float(theta[0])
        g1 = float(np.sum(diff * x)) + 2e-10 * float(theta[1])
        return nll, np.array([g0, g1], dtype=np.float64)

    res = optimize.minimize(
        nll_and_grad,
        x0=np.array([0.0, 2.0], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-200.0, 200.0), (1e-6, 400.0)],
        options={"ftol": 1e-14, "gtol": 1e-12},
    )
    th0, th1 = float(res.x[0]), float(res.x[1])
    gamma1 = th1 / sd
    gamma0 = th0 - gamma1 * mu
    cv_star = -gamma0 / gamma1 if gamma1 > 0.0 else float("nan")
    return gamma0, gamma1, cv_star


def fit_logistic_crossing_with_bootstrap(
    cv: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    pc: float,
    n_boot: int = 2000,
    boot_seed: int = 20260909,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Fit logistic MLE for P_span(Cv) and compute stratified bootstrap 95% CIs."""
    cv = np.asarray(cv, dtype=np.float64)
    k = np.asarray(k, dtype=np.int64)
    n = np.asarray(n, dtype=np.int64)
    p_obs = k / np.maximum(n, 1)

    g0, g1, cv_star = _fit_binomial_logistic_mle(cv, k, n)
    cv_interp = interpolate_spanning_crossing(cv, p_obs, target=0.5)

    # Fisher information Wald SE for cross-check
    eta = g0 + g1 * cv
    p_hat = 1.0 / (1.0 + np.exp(-np.clip(eta, -40.0, 40.0)))
    w = n * p_hat * (1.0 - p_hat)
    X_des = np.column_stack([np.ones_like(cv), cv])
    info = X_des.T @ (w[:, None] * X_des)
    try:
        cov = np.linalg.inv(info)
        # Delta method for Cv* = -g0 / g1
        grad = np.array([-1.0 / g1, g0 / (g1 * g1)], dtype=np.float64)
        wald_se = float(np.sqrt(max(0.0, float(grad @ cov @ grad))))
    except np.linalg.LinAlgError:
        wald_se = float("nan")

    rng = np.random.default_rng(boot_seed)
    boot_cv = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        k_b = rng.binomial(n, p_obs)
        _, _, c_b = _fit_binomial_logistic_mle(cv, k_b, n)
        boot_cv[b] = c_b

    valid = boot_cv[np.isfinite(boot_cv)]
    ci_lo, ci_hi = np.percentile(valid, [2.5, 97.5]) if len(valid) > 10 else (np.nan, np.nan)
    boot_se = float(np.std(valid, ddof=1)) if len(valid) > 10 else float("nan")

    # Curve table with Wilson 95% CIs
    curve_rows = []
    for c_i, k_i, n_i in zip(cv, k, n):
        w_lo, w_hi = wilson_ci(int(k_i), int(n_i))
        curve_rows.append({
            "Cv": float(c_i),
            "k_span": int(k_i),
            "n": int(n_i),
            "P_span": float(k_i) / float(n_i),
            "wilson_ci_lo": w_lo,
            "wilson_ci_hi": w_hi,
        })

    summary = {
        "Cv_star": _safe_float(cv_star),
        "Cv_star_boot_se": _safe_float(boot_se),
        "Cv_star_wald_se": _safe_float(wald_se),
        "Cv_star_ci95_lo": _safe_float(ci_lo),
        "Cv_star_ci95_hi": _safe_float(ci_hi),
        "Cv_star_over_pc": _safe_float(cv_star / pc) if np.isfinite(cv_star) else None,
        "Cv_star_over_pc_ci95_lo": _safe_float(ci_lo / pc) if np.isfinite(ci_lo) else None,
        "Cv_star_over_pc_ci95_hi": _safe_float(ci_hi / pc) if np.isfinite(ci_hi) else None,
        "Cv_star_linear_interp": _safe_float(cv_interp),
        "logistic_gamma0": _safe_float(g0),
        "logistic_gamma1": _safe_float(g1),
        "n_boot": int(n_boot),
        "total_realizations": int(np.sum(n)),
        "curve": curve_rows,
    }
    return summary, valid


# ---------------------------------------------------------------------------
# 1. Dilute-Limit Quantitative Validation & Canonical QA
# ---------------------------------------------------------------------------

def analyze_dilute_validation(df_dilute: pd.DataFrame) -> Dict[str, Any]:
    """Compare dilute high-T (T=100) simulated D/Cv against analytical (1/3) f0 r_nn^2."""
    out: Dict[str, Any] = {}
    for lat in ("SC", "BCC", "FCC"):
        sub = df_dilute[df_dilute["lattice"] == lat]
        if sub.empty:
            continue
        spec = LATTICE_SPECS[lat]
        d_ref = (1.0 / 3.0) * float(spec["f0_dilute"]) * float(spec["nn_dist_sq"])
        by_cv = []
        d_means_map: Dict[float, float] = {}
        for cv_val, grp in sub.groupby("Cv"):
            n = int(len(grp))
            d_mean = float(grp["D"].mean())
            d_se = float(grp["D"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            dc_mean = float(grp["D_over_Cv"].mean())
            dc_sd = float(grp["D_over_Cv"].std(ddof=1)) if n > 1 else 0.0
            dc_se = dc_sd / np.sqrt(n) if n > 1 else 0.0
            rel_err_pct = 100.0 * (dc_mean - d_ref) / d_ref
            d_means_map[round(float(cv_val), 4)] = d_mean
            by_cv.append({
                "Cv": float(cv_val),
                "n": n,
                "D_mean": d_mean,
                "D_se": d_se,
                "D_over_Cv_mean": dc_mean,
                "D_over_Cv_sd": dc_sd,
                "D_over_Cv_se": dc_se,
                "D_ref_analytical": d_ref,
                "rel_error_pct": rel_err_pct,
                "msd_r2_mean": float(grp["msd_r2"].mean()),
                "msd_loglog_slope_mean": float(grp["msd_loglog_slope"].mean()),
            })
        lin_ratio = None
        if 0.015 in d_means_map and 0.030 in d_means_map and d_means_map[0.015] > 0:
            lin_ratio = d_means_map[0.030] / d_means_map[0.015]
        out[lat] = {
            "Z": int(spec["Z"]),
            "f0_dilute": float(spec["f0_dilute"]),
            "r_nn_sq": float(spec["nn_dist_sq"]),
            "D_ref_analytical": d_ref,
            "by_Cv": by_cv,
            "D_ratio_0030_over_0015": _safe_float(lin_ratio),
        }
    return out


def analyze_dilute_convergence(df_seed: pd.DataFrame) -> Dict[str, Any]:
    """Compute canonical 465-seed D/Cv trends and diffusive MSD QA."""
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


# ---------------------------------------------------------------------------
# 2 & 3. Non-Interacting (J=0) and Interacting (J=1) Percolation Crossovers
# ---------------------------------------------------------------------------

def analyze_j0_percolation(
    df_j0: pd.DataFrame, n_boot: int = 2000
) -> Tuple[Dict[str, Any], Dict[Tuple[str, int], np.ndarray]]:
    """Analyze non-interacting (J=0) finite-size random site-percolation benchmarks."""
    out: Dict[str, Any] = {}
    boot_draws: Dict[Tuple[str, int], np.ndarray] = {}
    seed_offset = 0
    for (lat, L_val), grp in df_j0.groupby(["lattice", "L"]):
        lat_str = str(lat)
        L_int = int(L_val)
        pc = float(LATTICE_SPECS[lat_str]["pc"])
        by_cv = (
            grp.groupby("Cv", as_index=False)
            .agg(
                n=("is_spanning", "count"),
                k_span=("is_spanning", "sum"),
                f_max_pct_mean=("f_max_pct", "mean"),
                f_max_pct_sd=("f_max_pct", lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
                C_index_mean=("C_index", "mean"),
                C_index_sd=("C_index", lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
            )
            .sort_values("Cv")
        )
        fit_info, draws = fit_logistic_crossing_with_bootstrap(
            by_cv["Cv"].to_numpy(),
            by_cv["k_span"].to_numpy(),
            by_cv["n"].to_numpy(),
            pc=pc,
            n_boot=n_boot,
            boot_seed=20261000 + seed_offset,
        )
        seed_offset += 1
        boot_draws[(lat_str, L_int)] = draws
        # Attach f_max and C_index to each curve point
        for row_dict, (_, r) in zip(fit_info["curve"], by_cv.iterrows()):
            row_dict["f_max_pct_mean"] = float(r["f_max_pct_mean"])
            row_dict["f_max_pct_sd"] = float(r["f_max_pct_sd"])
            row_dict["C_index_mean"] = float(r["C_index_mean"])
            row_dict["C_index_sd"] = float(r["C_index_sd"])

        key = f"{lat_str}_L{L_int}"
        out[key] = {
            "lattice": lat_str,
            "L": L_int,
            "N": int(grp["N"].iloc[0]),
            "pc_inf": pc,
            **fit_info,
        }
    return out, boot_draws


def analyze_spanning_crossings(
    df_span: pd.DataFrame,
    df_perc: Optional[pd.DataFrame] = None,
    j0_results: Optional[Dict[str, Any]] = None,
    j0_boot_draws: Optional[Dict[Tuple[str, int], np.ndarray]] = None,
    n_boot: int = 2000,
) -> Dict[str, Any]:
    """Compute logistic MLE Cv* + bootstrap 95% CIs for SC, BCC, FCC (J=1 vs J=0)."""
    targets = [
        ("SC", "SC", 0.93),
        ("BCC", "BCC", 1.30),
        ("FCC", "FCC", 2.00),
        ("FCC_above_Tc", "FCC", 3.06),
    ]
    out: Dict[str, Any] = {}

    for idx, (tag, lat, T_val) in enumerate(targets):
        # Use merged dense curve (which has n=50 in transition window + n=15 tails)
        sub = df_span[(df_span["lattice"] == lat) & (np.isclose(df_span["T"], T_val, atol=0.02))]
        if sub.empty:
            continue
        pc = float(LATTICE_SPECS[lat]["pc"])
        by_cv = (
            sub.groupby("Cv", as_index=False)
            .agg(
                n=("is_spanning", "count"),
                k_span=("is_spanning", "sum"),
                f_max_pct_mean=("f_max_pct", "mean"),
                f_max_pct_sd=("f_max_pct", lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
                C_index_mean=("C_index", "mean"),
            )
            .sort_values("Cv")
        )
        fit_info, draws_j1 = fit_logistic_crossing_with_bootstrap(
            by_cv["Cv"].to_numpy(),
            by_cv["k_span"].to_numpy(),
            by_cv["n"].to_numpy(),
            pc=pc,
            n_boot=n_boot,
            boot_seed=20262000 + idx,
        )
        for row_dict, (_, r) in zip(fit_info["curve"], by_cv.iterrows()):
            row_dict["f_max_pct_mean"] = float(r["f_max_pct_mean"])
            row_dict["f_max_pct_sd"] = float(r["f_max_pct_sd"])
            row_dict["C_index_mean"] = float(r["C_index_mean"])

        entry: Dict[str, Any] = {
            "lattice": lat,
            "T": T_val,
            "T_over_Tc": round(T_val / float(LATTICE_SPECS[lat]["Tc"]), 4),
            "pc": pc,
            **fit_info,
        }

        # Sensitivity check: fit strictly on the uniform n=50 high-precision window
        if df_perc is not None:
            sub_hp = df_perc[(df_perc["lattice"] == lat) & (np.isclose(df_perc["T"], T_val, atol=0.02))]
            if not sub_hp.empty:
                hp_grp = (
                    sub_hp.groupby("Cv", as_index=False)
                    .agg(n=("is_spanning", "count"), k_span=("is_spanning", "sum"))
                    .sort_values("Cv")
                )
                hp_fit, _ = fit_logistic_crossing_with_bootstrap(
                    hp_grp["Cv"].to_numpy(),
                    hp_grp["k_span"].to_numpy(),
                    hp_grp["n"].to_numpy(),
                    pc=pc,
                    n_boot=n_boot,
                    boot_seed=20263000 + idx,
                )
                entry["window_n50_only"] = {
                    "Cv_star": hp_fit["Cv_star"],
                    "Cv_star_ci95_lo": hp_fit["Cv_star_ci95_lo"],
                    "Cv_star_ci95_hi": hp_fit["Cv_star_ci95_hi"],
                    "Cv_star_over_pc": hp_fit["Cv_star_over_pc"],
                }

        # Direct ratio against finite-size L=20 non-interacting (J=0) baseline
        if j0_results and j0_boot_draws and (lat, 20) in j0_boot_draws:
            j0_key = f"{lat}_L20"
            c_j0 = j0_results[j0_key]["Cv_star"]
            draws_j0 = j0_boot_draws[(lat, 20)]
            m = min(len(draws_j1), len(draws_j0))
            if c_j0 is not None and m > 10 and fit_info["Cv_star"] is not None:
                ratio_draws = draws_j1[:m] / draws_j0[:m]
                r_lo, r_hi = np.percentile(ratio_draws, [2.5, 97.5])
                diff_draws = draws_j1[:m] - draws_j0[:m]
                d_lo, d_hi = np.percentile(diff_draws, [2.5, 97.5])
                entry["vs_J0_L20"] = {
                    "Cv_star_J0_L20": float(c_j0),
                    "ratio_J1_over_J0": float(fit_info["Cv_star"] / c_j0),
                    "ratio_ci95_lo": float(r_lo),
                    "ratio_ci95_hi": float(r_hi),
                    "delta_Cv_star": float(fit_info["Cv_star"] - c_j0),
                    "delta_ci95_lo": float(d_lo),
                    "delta_ci95_hi": float(d_hi),
                }

        out[tag] = entry

    return out


def analyze_fcc_three_way_cv020(
    df_seed: pd.DataFrame,
    df_span: pd.DataFrame,
    df_j0: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """Direct 3-way FCC Cv=0.20, L=20 comparison isolating geometry vs attraction."""
    out: Dict[str, Any] = {}

    def _summarize(sub: pd.DataFrame, has_D: bool = True) -> Dict[str, Any]:
        n = int(len(sub))
        k = int(sub["is_spanning"].sum())
        w_lo, w_hi = wilson_ci(k, n)
        d: Dict[str, Any] = {
            "n": n,
            "k_span": k,
            "P_span": float(k) / float(n) if n > 0 else float("nan"),
            "P_span_wilson_lo": w_lo,
            "P_span_wilson_hi": w_hi,
            "f_max_pct_mean": float(sub["f_max_pct"].mean()),
            "f_max_pct_sd": float(sub["f_max_pct"].std(ddof=1)) if n > 1 else 0.0,
            "f_max_pct_se": float(sub["f_max_pct"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "C_index_mean": float(sub["C_index"].mean()),
            "C_index_sd": float(sub["C_index"].std(ddof=1)) if n > 1 else 0.0,
        }
        if has_D and "D" in sub.columns:
            d["D_mean"] = float(sub["D"].mean())
            d["D_se"] = float(sub["D"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        if "C_index_window_mean" in sub.columns:
            d["C_window_mean"] = float(sub["C_index_window_mean"].mean())
        return d

    if df_j0 is not None:
        s_j0 = df_j0[(df_j0["lattice"] == "FCC") & (df_j0["L"] == 20) & (np.isclose(df_j0["Cv"], 0.20, atol=1e-4))]
        if not s_j0.empty:
            out["J0_random_L20"] = _summarize(s_j0, has_D=False)

    s_hi = df_seed[(df_seed["lattice"] == "FCC") & (np.isclose(df_seed["Cv"], 0.20, atol=1e-4)) & (np.isclose(df_seed["T"], 3.06, atol=0.02))]
    if not s_hi.empty:
        out["J1_above_Tc_T306"] = _summarize(s_hi, has_D=True)

    s_lo_15 = df_seed[(df_seed["lattice"] == "FCC") & (np.isclose(df_seed["Cv"], 0.20, atol=1e-4)) & (np.isclose(df_seed["T"], 2.00, atol=0.02))]
    if not s_lo_15.empty:
        out["J1_below_Tc_T200_n15"] = _summarize(s_lo_15, has_D=True)

    s_lo_50 = df_span[(df_span["lattice"] == "FCC") & (np.isclose(df_span["Cv"], 0.20, atol=1e-4)) & (np.isclose(df_span["T"], 2.00, atol=0.02))]
    if not s_lo_50.empty:
        out["J1_below_Tc_T200_n50"] = _summarize(s_lo_50, has_D=True)

    return out


# ---------------------------------------------------------------------------
# 4. Structure-Transport Correlations & Confounding-Controlled Regressions
# ---------------------------------------------------------------------------

def _controlled_regression_for_predictor(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Fit controlled multiple OLS: ln(D) ~ ln(col) + ln(Cv) + ln(T/Tc) + lattice dummies."""
    sub = df[["lattice", "D", "Cv", "T_over_Tc", col]].dropna().copy()
    sub = sub[(sub["D"] > 0) & (sub["Cv"] > 0) & (sub["T_over_Tc"] > 0) & (sub[col] > 0)]
    n = len(sub)
    y = np.log(sub["D"].to_numpy(dtype=np.float64))
    lx = np.log(sub[col].to_numpy(dtype=np.float64))
    lcv = np.log(sub["Cv"].to_numpy(dtype=np.float64))
    lt = np.log(sub["T_over_Tc"].to_numpy(dtype=np.float64))
    lat = sub["lattice"].to_numpy()
    is_bcc = (lat == "BCC").astype(np.float64)
    is_fcc = (lat == "FCC").astype(np.float64)

    # Baseline M0: [1, ln(Cv), ln(T/Tc), I_BCC, I_FCC] (5 params)
    X0 = np.column_stack([np.ones(n), lcv, lt, is_bcc, is_fcc])
    fit0 = _ols_fit(X0, y)

    # Full additive M1: [1, ln(col), ln(Cv), ln(T/Tc), I_BCC, I_FCC] (6 params)
    X1 = np.column_stack([np.ones(n), lx, lcv, lt, is_bcc, is_fcc])
    fit1 = _ols_fit(X1, y)

    # Partial correlation via Frisch-Waugh-Lovell residualization
    res_y = fit0["resid"]
    res_x = _ols_fit(X0, lx)["resid"]
    part_r, part_p = stats.pearsonr(res_x, res_y)
    part_r2 = (fit0["rss"] - fit1["rss"]) / fit0["rss"] if fit0["rss"] > 0 else float("nan")
    f_add = _nested_f_test(fit0["rss"], fit0["p"], fit1["rss"], fit1["p"], n)

    # Interaction M2: allow lattice-specific slope on ln(col) (8 params)
    X2 = np.column_stack([
        np.ones(n), lx, lcv, lt, is_bcc, is_fcc, is_bcc * lx, is_fcc * lx
    ])
    fit2 = _ols_fit(X2, y)
    f_int = _nested_f_test(fit1["rss"], fit1["p"], fit2["rss"], fit2["p"], n)

    # Per-lattice controlled slopes (separately within each lattice: ln(D) ~ 1 + ln(col) + ln(Cv) + ln(T/Tc))
    per_lat = {}
    for g in ("SC", "BCC", "FCC"):
        m = (lat == g)
        if np.count_nonzero(m) >= 6:
            Xg0 = np.column_stack([np.ones(np.count_nonzero(m)), lcv[m], lt[m]])
            Xg1 = np.column_stack([np.ones(np.count_nonzero(m)), lx[m], lcv[m], lt[m]])
            fg0 = _ols_fit(Xg0, y[m])
            fg1 = _ols_fit(Xg1, y[m])
            pr_g, pp_g = stats.pearsonr(_ols_fit(Xg0, lx[m])["resid"], fg0["resid"])
            per_lat[g] = {
                "n": int(np.count_nonzero(m)),
                "b_predictor": _safe_float(fg1["beta"][1]),
                "se": _safe_float(fg1["se"][1]),
                "t": _safe_float(fg1["t"][1]),
                "p_value": _safe_float(fg1["p_vals"][1]),
                "partial_r": _safe_float(pr_g),
                "partial_p": _safe_float(pp_g),
                "R2_base": _safe_float(fg0["r2"]),
                "R2_full": _safe_float(fg1["r2"]),
                "delta_R2": _safe_float(fg1["r2"] - fg0["r2"]),
            }

    return {
        "predictor": col,
        "n": int(n),
        "R2_base_Cv_T_lat": _safe_float(fit0["r2"]),
        "R2_with_predictor": _safe_float(fit1["r2"]),
        "delta_R2": _safe_float(fit1["r2"] - fit0["r2"]),
        "partial_R2": _safe_float(part_r2),
        "partial_r": _safe_float(part_r),
        "partial_p": _safe_float(part_p),
        "b_predictor": _safe_float(fit1["beta"][1]),
        "b_predictor_se": _safe_float(fit1["se"][1]),
        "b_predictor_t": _safe_float(fit1["t"][1]),
        "b_predictor_p": _safe_float(fit1["p_vals"][1]),
        "b_lnCv": _safe_float(fit1["beta"][2]),
        "b_lnCv_se": _safe_float(fit1["se"][2]),
        "b_lnT_Tc": _safe_float(fit1["beta"][3]),
        "b_lnT_Tc_se": _safe_float(fit1["se"][3]),
        "F_add_predictor": f_add,
        "F_lattice_interaction": f_int,
        "R2_with_interaction": _safe_float(fit2["r2"]),
        "per_lattice_controlled": per_lat,
    }


def analyze_structure_transport(df_seed: pd.DataFrame) -> Dict[str, Any]:
    """Compute bivariate correlations, time-aligned C_bar diagnostics, and controlled OLS."""
    out: Dict[str, Any] = {}
    cols_to_check = [("D_vs_f_max", "f_max"), ("D_vs_C_index", "C_index")]
    if "C_index_window_mean" in df_seed.columns:
        cols_to_check.append(("D_vs_C_index_window_mean", "C_index_window_mean"))

    for name, col in cols_to_check:
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

    # Comparison between time-aligned window C_bar and final t=6000 C_index
    if "C_index_window_mean" in df_seed.columns:
        c_win = df_seed["C_index_window_mean"].to_numpy(dtype=np.float64)
        c_fin = df_seed["C_index"].to_numpy(dtype=np.float64)
        r_wf, _ = stats.pearsonr(c_win, c_fin)
        diff = c_fin - c_win
        out["C_window_vs_final_alignment"] = {
            "n": int(len(df_seed)),
            "pearson_r": _safe_float(r_wf),
            "mean_diff_final_minus_window": _safe_float(np.mean(diff)),
            "mean_abs_diff": _safe_float(np.mean(np.abs(diff))),
            "max_abs_diff": _safe_float(np.max(np.abs(diff))),
            "mean_within_window_sd": _safe_float(df_seed["C_index_window_sd"].mean()),
        }

    # Confounding-controlled regressions
    ctrl: Dict[str, Any] = {
        "C_index_final": _controlled_regression_for_predictor(df_seed, "C_index"),
        "f_max": _controlled_regression_for_predictor(df_seed, "f_max"),
    }
    if "C_index_window_mean" in df_seed.columns:
        ctrl["C_index_window_mean"] = _controlled_regression_for_predictor(df_seed, "C_index_window_mean")
    out["controlled_regressions"] = ctrl
    out["condition_fixed_effects"] = analyze_condition_fixed_effects(df_seed)
    return out


def condition_fixed_effects(
    df: pd.DataFrame,
    x_col: str,
    log_x: bool = False,
    n_boot: int = 9999,
    boot_seed: int = 20261120,
) -> Dict[str, Any]:
    """Condition-fixed-effects regression: ln(D_ij) = mu_j + beta * (X_ij - Xbar_j) + eps_ij.

    Reports exact within-condition slope, within-R^2, classical OLS SE, condition-clustered
    CR1 SE (df = G - 1), and Rademacher wild cluster bootstrap (WCB) p-value and 95% CI.
    """
    sub = df[["lattice", "Cv", "T", "D", x_col]].dropna().copy()
    sub = sub[sub["D"] > 0]
    if log_x:
        sub = sub[sub[x_col] > 0]
    sub["_cond"] = (
        sub["lattice"].astype(str)
        + "|"
        + sub["Cv"].round(4).astype(str)
        + "|"
        + sub["T"].round(3).astype(str)
    )
    u_conds = sorted(sub["_cond"].unique().tolist())
    G = len(u_conds)
    n = len(sub)
    if G < 2 or n <= G + 1:
        return {"n": int(n), "G": int(G), "beta_FE": None}

    y = np.log(sub["D"].to_numpy(dtype=np.float64))
    x_raw = sub[x_col].to_numpy(dtype=np.float64)
    x = np.log(x_raw) if log_x else x_raw
    c_arr = sub["_cond"].to_numpy()

    y_dm = np.empty_like(y)
    x_dm = np.empty_like(x)
    c_masks = []
    for c in u_conds:
        m = (c_arr == c)
        c_masks.append(m)
        y_dm[m] = y[m] - np.mean(y[m])
        x_dm[m] = x[m] - np.mean(x[m])

    sxx = float(np.dot(x_dm, x_dm))
    if sxx <= 0.0:
        return {"n": int(n), "G": int(G), "beta_FE": None}

    beta = float(np.dot(x_dm, y_dm)) / sxx
    resid = y_dm - beta * x_dm
    rss_w = float(np.dot(resid, resid))
    tss_w = float(np.dot(y_dm, y_dm))
    tss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_within = 1.0 - rss_w / tss_w if tss_w > 0.0 else float("nan")
    r_within = float(np.sign(beta) * np.sqrt(max(0.0, r2_within))) if np.isfinite(r2_within) else float("nan")
    r2_fe_only = 1.0 - tss_w / tss_tot if tss_tot > 0.0 else float("nan")
    r2_fe_full = 1.0 - rss_w / tss_tot if tss_tot > 0.0 else float("nan")

    dof_ols = n - G - 1
    se_ols = float(np.sqrt((rss_w / float(dof_ols)) / sxx))
    t_ols = beta / se_ols
    p_ols = float(2.0 * stats.t.sf(abs(t_ols), dof_ols))

    # CR1 condition-clustered SE (small-sample corrected, df = G - 1)
    scores = np.array([float(np.dot(x_dm[m], resid[m])) for m in c_masks], dtype=np.float64)
    adj = (float(G) / float(G - 1)) * (float(n - 1) / float(dof_ols))
    se_cr1 = float(np.sqrt(adj * float(np.sum(scores ** 2))) / sxx)
    t_cr1 = beta / se_cr1 if se_cr1 > 0.0 else float("nan")
    p_cr1 = float(2.0 * stats.t.sf(abs(t_cr1), G - 1)) if np.isfinite(t_cr1) else float("nan")

    # Rademacher wild cluster bootstrap (WCR null p-value + unrestricted 95% CI)
    rng = np.random.default_rng(boot_seed)
    W = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_boot, G))
    xy0 = np.array([float(np.dot(x_dm[m], y_dm[m])) for m in c_masks], dtype=np.float64)
    xx_g = np.array([float(np.dot(x_dm[m], x_dm[m])) for m in c_masks], dtype=np.float64)
    beta_null = (W @ xy0) / sxx
    sc_null = W * xy0[None, :] - beta_null[:, None] * xx_g[None, :]
    se_null = np.sqrt(np.maximum(adj * np.sum(sc_null ** 2, axis=1), 1e-30)) / sxx
    t_null = beta_null / se_null
    p_wcb = float((1.0 + np.sum(np.abs(t_null) >= abs(t_cr1))) / float(n_boot + 1))

    xr1 = np.array([float(np.dot(x_dm[m], resid[m])) for m in c_masks], dtype=np.float64)
    beta_unres = beta + (W @ xr1) / sxx
    ci_lo, ci_hi = np.percentile(beta_unres, [2.5, 97.5])

    return {
        "predictor": x_col,
        "log_x": bool(log_x),
        "n": int(n),
        "G_conditions": int(G),
        "beta_FE": _safe_float(beta),
        "se_ols": _safe_float(se_ols),
        "t_ols": _safe_float(t_ols),
        "p_ols": _safe_float(p_ols),
        "se_CR1": _safe_float(se_cr1),
        "t_CR1": _safe_float(t_cr1),
        "df_CR1": int(G - 1),
        "p_CR1": _safe_float(p_cr1),
        "n_boot": int(n_boot),
        "p_WCB": _safe_float(p_wcb),
        "wcb_ci95_lo": _safe_float(ci_lo),
        "wcb_ci95_hi": _safe_float(ci_hi),
        "R2_within": _safe_float(r2_within),
        "r_within": _safe_float(r_within),
        "R2_FE_only": _safe_float(r2_fe_only),
        "R2_FE_plus_X": _safe_float(r2_fe_full),
    }


def analyze_condition_fixed_effects(df_seed: pd.DataFrame) -> Dict[str, Any]:
    """Run condition-fixed-effects robustness checks across all 31 conditions and subgroups."""
    out: Dict[str, Any] = {}
    if "C_index_window_mean" in df_seed.columns:
        out["C_bar_centered_all31"] = condition_fixed_effects(df_seed, "C_index_window_mean", log_x=False, boot_seed=20261120)
        out["ln_C_bar_all31"] = condition_fixed_effects(df_seed, "C_index_window_mean", log_x=True, boot_seed=20261121)
        sub_lo = df_seed[df_seed["T_over_Tc"] < 1.0]
        sub_hi = df_seed[df_seed["T_over_Tc"] > 1.0]
        out["C_bar_centered_subcritical14"] = condition_fixed_effects(sub_lo, "C_index_window_mean", log_x=False, boot_seed=20261122)
        out["ln_C_bar_subcritical14"] = condition_fixed_effects(sub_lo, "C_index_window_mean", log_x=True, boot_seed=20261123)
        out["C_bar_centered_supercritical17"] = condition_fixed_effects(sub_hi, "C_index_window_mean", log_x=False, boot_seed=20261124)
        out["ln_C_bar_supercritical17"] = condition_fixed_effects(sub_hi, "C_index_window_mean", log_x=True, boot_seed=20261125)
    out["C_final_centered_all31"] = condition_fixed_effects(df_seed, "C_index", log_x=False, boot_seed=20261126)
    out["ln_C_final_all31"] = condition_fixed_effects(df_seed, "C_index", log_x=True, boot_seed=20261127)
    out["f_max_centered_all31"] = condition_fixed_effects(df_seed, "f_max", log_x=False, boot_seed=20261128)
    out["ln_f_max_all31"] = condition_fixed_effects(df_seed, "f_max", log_x=True, boot_seed=20261129)
    return out


# ---------------------------------------------------------------------------
# 5. Cross-Lattice Transport Scaling + Grouped Cross-Validation
# ---------------------------------------------------------------------------

def _eval_scaling_predictions(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> Dict[str, float]:
    """Compute log-space R^2, linear-space R^2, RMSE_log, and MAPE (%) from log predictions."""
    rss_log = float(np.sum((y_log_true - y_log_pred) ** 2))
    tss_log = float(np.sum((y_log_true - np.mean(y_log_true)) ** 2))
    r2_log = 1.0 - rss_log / tss_log if tss_log > 0 else float("nan")
    rmse_log = float(np.sqrt(np.mean((y_log_true - y_log_pred) ** 2)))

    y_lin = np.exp(y_log_true)
    y_hat_lin = np.exp(y_log_pred)
    r2_lin = 1.0 - float(np.sum((y_lin - y_hat_lin) ** 2)) / float(np.sum((y_lin - np.mean(y_lin)) ** 2))
    mape_pct = 100.0 * float(np.mean(np.abs(y_lin - y_hat_lin) / y_lin))
    return {
        "R2_log": r2_log,
        "R2_linear": r2_lin,
        "RMSE_log": rmse_log,
        "MAPE_pct": mape_pct,
    }


def _run_grouped_cv(
    sub: pd.DataFrame,
    y_col_vals: np.ndarray,
    alpha_fixed: float = 0.75,
    beta_fixed: float = 1.0,
) -> Dict[str, Any]:
    """Run LOCO-CV (31 conditions), 5-fold grouped CV, and Leave-One-Lattice-Out CV."""
    ln_y = np.log(y_col_vals)
    ln_cv = np.log(sub["Cv_over_pc"].to_numpy(dtype=np.float64))
    ln_t = np.log(sub["T_over_Tc"].to_numpy(dtype=np.float64))
    ln_M = alpha_fixed * ln_cv + beta_fixed * ln_t
    X = np.column_stack([np.ones(len(sub)), ln_cv, ln_t])

    # Identify the 31 distinct (lattice, Cv, T) conditions
    cond_keys = [
        f"{r['lattice']}|{round(float(r['Cv']), 4)}|{round(float(r['T']), 3)}"
        for _, r in sub.iterrows()
    ]
    cond_arr = np.asarray(cond_keys)
    unique_conds = sorted(set(cond_keys))
    n = len(sub)

    def _cv_loop(folds: List[np.ndarray]) -> Dict[str, Any]:
        pred_free = np.empty(n, dtype=np.float64)
        pred_fixed = np.empty(n, dtype=np.float64)
        for test_mask in folds:
            train_mask = ~test_mask
            beta_tr, _, _, _ = np.linalg.lstsq(X[train_mask], ln_y[train_mask], rcond=None)
            pred_free[test_mask] = X[test_mask] @ beta_tr
            a_fix_tr = float(np.mean(ln_y[train_mask] - ln_M[train_mask]))
            pred_fixed[test_mask] = a_fix_tr + ln_M[test_mask]
        return {
            "free_fit": _eval_scaling_predictions(ln_y, pred_free),
            "fixed_3_4_and_1": _eval_scaling_predictions(ln_y, pred_fixed),
        }

    # 1. Leave-One-Condition-Out CV (31 folds, holding out all 15 seeds of a condition together)
    loco_folds = [cond_arr == c for c in unique_conds]
    loco_res = _cv_loop(loco_folds)

    # 2. Stratified 5-fold grouped condition CV (stratified across lattices, deterministic seed)
    rng = np.random.default_rng(20260915)
    cond_df = pd.DataFrame({
        "cond": unique_conds,
        "lat": [c.split("|")[0] for c in unique_conds],
    })
    fold_ids = np.zeros(len(cond_df), dtype=np.int64)
    k_folds = 5
    for lat in ("SC", "BCC", "FCC"):
        idxs = cond_df.index[cond_df["lat"] == lat].to_numpy()
        perm = rng.permutation(idxs)
        for j, idx in enumerate(perm):
            fold_ids[idx] = j % k_folds
    cond_to_fold = dict(zip(cond_df["cond"], fold_ids))
    row_folds = np.asarray([cond_to_fold[c] for c in cond_arr])
    kfold_masks = [row_folds == f for f in range(k_folds)]
    kfold_res = _cv_loop(kfold_masks)

    # 3. Leave-One-Lattice-Out CV (3 folds: zero-shot prediction on each unseen crystal topology!)
    lat_arr = sub["lattice"].to_numpy()
    lolo_folds = [lat_arr == g for g in ("SC", "BCC", "FCC")]
    lolo_overall = _cv_loop(lolo_folds)
    lolo_per_lat: Dict[str, Any] = {}
    for g in ("SC", "BCC", "FCC"):
        test_mask = (lat_arr == g)
        train_mask = ~test_mask
        beta_tr, _, _, _ = np.linalg.lstsq(X[train_mask], ln_y[train_mask], rcond=None)
        p_free = X[test_mask] @ beta_tr
        a_fix_tr = float(np.mean(ln_y[train_mask] - ln_M[train_mask]))
        p_fixed = a_fix_tr + ln_M[test_mask]
        lolo_per_lat[g] = {
            "n_held_out": int(np.count_nonzero(test_mask)),
            "train_alpha": float(beta_tr[1]),
            "train_beta": float(beta_tr[2]),
            "free_fit": _eval_scaling_predictions(ln_y[test_mask], p_free),
            "fixed_3_4_and_1": _eval_scaling_predictions(ln_y[test_mask], p_fixed),
        }

    return {
        "n_conditions": int(len(unique_conds)),
        "LOCO_31fold": loco_res,
        "grouped_5fold": kfold_res,
        "LOLO_3fold_overall": lolo_overall,
        "LOLO_per_held_out_lattice": lolo_per_lat,
    }


def _full_scaling_block(sub: pd.DataFrame, y_vals: np.ndarray) -> Dict[str, Any]:
    """Complete two-parameter fit, fixed (3/4, 1) fit, ANOVA F-test, variance decomp, and CV."""
    ln_y = np.log(y_vals)
    ln_cv = np.log(sub["Cv_over_pc"].to_numpy(dtype=np.float64))
    ln_t = np.log(sub["T_over_Tc"].to_numpy(dtype=np.float64))
    ln_M = 0.75 * ln_cv + 1.0 * ln_t
    X_pooled = np.column_stack([np.ones_like(ln_cv), ln_cv, ln_t])

    fit_pooled = _ols_fit(X_pooled, ln_y)
    a_hat, alpha_hat, beta_hat = fit_pooled["beta"]
    a_se, alpha_se, beta_se = fit_pooled["se"]
    free_metrics = _eval_scaling_predictions(ln_y, fit_pooled["y_hat"])

    a_fixed = float(np.mean(ln_y - ln_M))
    A_fixed = float(np.exp(a_fixed))
    fixed_metrics = _eval_scaling_predictions(ln_y, a_fixed + ln_M)

    # Separate-lattice 9-parameter model & nested F(6, n-9)
    groups = sorted(sub["lattice"].unique().tolist())
    G = len(groups)
    n = len(sub)
    lat_arr = sub["lattice"].to_numpy()
    X_free = np.zeros((n, 3 * G), dtype=np.float64)
    per_lat: Dict[str, Any] = {}
    for gi, g in enumerate(groups):
        m = (lat_arr == g).astype(np.float64)
        X_free[:, 3 * gi] = m
        X_free[:, 3 * gi + 1] = m * ln_cv
        X_free[:, 3 * gi + 2] = m * ln_t
        mb = (lat_arr == g)
        Xg = np.column_stack([np.ones(np.count_nonzero(mb)), ln_cv[mb], ln_t[mb]])
        fg = _ols_fit(Xg, ln_y[mb])
        per_lat[g] = {
            "n": int(np.count_nonzero(mb)),
            "a": _safe_float(fg["beta"][0]),
            "prefactor": _safe_float(np.exp(fg["beta"][0])),
            "alpha": _safe_float(fg["beta"][1]),
            "alpha_se": _safe_float(fg["se"][1]),
            "beta_T_over_Tc": _safe_float(fg["beta"][2]),
            "beta_Tc_over_T": _safe_float(-fg["beta"][2]),
            "beta_se": _safe_float(fg["se"][2]),
            "R2_log": _safe_float(fg["r2"]),
        }

    fit_sep = _ols_fit(X_free, ln_y)
    f_test = _nested_f_test(fit_pooled["rss"], fit_pooled["p"], fit_sep["rss"], fit_sep["p"], n)

    r2_pooled = float(fit_pooled["r2"])
    r2_sep = float(fit_sep["r2"])
    var_decomp = {
        "R2_log_pooled": r2_pooled,
        "R2_log_separate_9param": r2_sep,
        "delta_R2_lattice_specific": r2_sep - r2_pooled,
        "unexplained_within_lattice": 1.0 - r2_sep,
        "unexplained_total": 1.0 - r2_pooled,
    }

    return {
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
            "R2": _safe_float(free_metrics["R2_log"]),
            "R2_log": _safe_float(free_metrics["R2_log"]),
            "R2_linear": _safe_float(free_metrics["R2_linear"]),
            "RMSE_log": _safe_float(free_metrics["RMSE_log"]),
            "MAPE_pct": _safe_float(free_metrics["MAPE_pct"]),
        },
        "fixed_exponent_3_4_and_1": {
            "a_fixed": _safe_float(a_fixed),
            "prefactor": _safe_float(A_fixed),
            "alpha_fixed": 0.75,
            "beta_T_over_Tc_fixed": 1.0,
            "beta_Tc_over_T_fixed": -1.0,
            "R2_log": _safe_float(fixed_metrics["R2_log"]),
            "R2_linear": _safe_float(fixed_metrics["R2_linear"]),
            "RMSE_log": _safe_float(fixed_metrics["RMSE_log"]),
            "MAPE_pct": _safe_float(fixed_metrics["MAPE_pct"]),
        },
        "shared_vs_free_exponent_F_test": f_test,
        "variance_decomposition": var_decomp,
        "per_lattice_free_fits": per_lat,
        "cross_validation": _run_grouped_cv(sub, y_vals),
    }


def analyze_reduced_scaling(df_seed: pd.DataFrame) -> Dict[str, Any]:
    """Single-parameter Lambda test and two-parameter scaling (D_ref primary + D_scale)."""
    sub = df_seed[(df_seed["D_over_D0"] > 0) & (df_seed["Cv_over_pc"] > 0) & (df_seed["T_over_Tc"] > 0)].copy()

    dref_map = {
        k: (1.0 / 3.0) * float(v["f0_dilute"]) * float(v["nn_dist_sq"])
        for k, v in LATTICE_SPECS.items()
    }
    d_over_dref = (sub["D"] / sub["lattice"].map(dref_map)).to_numpy(dtype=np.float64)
    d_over_dscale = sub["D_over_D0"].to_numpy(dtype=np.float64)
    ln_Lam = np.log(sub["Lambda"].to_numpy(dtype=np.float64))

    # 1. Single-parameter Lambda diagnostics (both D/D_ref and D/D_scale)
    def _single_param_block(y_arr: np.ndarray) -> Dict[str, Any]:
        ln_y = np.log(y_arr)
        pooled = _ols_fit(np.column_stack([np.ones_like(ln_Lam), ln_Lam]), ln_y)
        per_l: Dict[str, Any] = {}
        for lat in ("SC", "BCC", "FCC"):
            m = (sub["lattice"] == lat).to_numpy()
            if np.count_nonzero(m) >= 3:
                fl = _ols_fit(np.column_stack([np.ones(np.count_nonzero(m)), ln_Lam[m]]), ln_y[m])
                per_l[lat] = {
                    "n": int(np.count_nonzero(m)),
                    "intercept": _safe_float(fl["beta"][0]),
                    "slope": _safe_float(fl["beta"][1]),
                    "slope_se": _safe_float(fl["se"][1]),
                    "R2": _safe_float(fl["r2"]),
                }
        return {
            "pooled": {
                "intercept": _safe_float(pooled["beta"][0]),
                "slope": _safe_float(pooled["beta"][1]),
                "slope_se": _safe_float(pooled["se"][1]),
                "R2": _safe_float(pooled["r2"]),
            },
            "per_lattice": per_l,
        }

    single_ref = _single_param_block(d_over_dref)
    single_scale = _single_param_block(d_over_dscale)

    # 2. Full two-parameter blocks
    block_dref = _full_scaling_block(sub, d_over_dref)
    block_dref["Dref_map"] = dref_map
    block_dscale = _full_scaling_block(sub, d_over_dscale)

    # Matched two-temperature subset (n = 420, excluding SC T/Tc = 1.77)
    m420 = ~((sub["lattice"] == "SC") & (sub["T"] > 1.8)).to_numpy()
    sub420 = sub[m420].copy()
    block_420_ref = _full_scaling_block(sub420, d_over_dref[m420])
    block_420_scale = _full_scaling_block(sub420, d_over_dscale[m420])

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
        "single_parameter_Lambda_Dref": single_ref,
        "single_parameter_Lambda": single_scale,
        "two_parameter_transport_Dref": block_dref,
        "two_parameter_transport": {
            **block_dscale,
            "zero_free_param_Dref": block_dref,
            "matched_two_temp_420": {
                "n": int(len(sub420)),
                "D_over_Dref": {
                    "prefactor_free": block_420_ref["free_fit"]["prefactor"],
                    "alpha": block_420_ref["free_fit"]["alpha"],
                    "alpha_se": block_420_ref["free_fit"]["alpha_se"],
                    "beta_T_over_Tc": block_420_ref["free_fit"]["beta_T_over_Tc"],
                    "beta_se": block_420_ref["free_fit"]["beta_se"],
                    "R2_log_free": block_420_ref["free_fit"]["R2_log"],
                    "prefactor_fixed": block_420_ref["fixed_exponent_3_4_and_1"]["prefactor"],
                    "R2_log_fixed": block_420_ref["fixed_exponent_3_4_and_1"]["R2_log"],
                    "LOCO_R2_log_free": block_420_ref["cross_validation"]["LOCO_31fold"]["free_fit"]["R2_log"],
                },
                "D_over_Dscale": {
                    "prefactor_free": block_420_scale["free_fit"]["prefactor"],
                    "alpha": block_420_scale["free_fit"]["alpha"],
                    "alpha_se": block_420_scale["free_fit"]["alpha_se"],
                    "beta_T_over_Tc": block_420_scale["free_fit"]["beta_T_over_Tc"],
                    "beta_se": block_420_scale["free_fit"]["beta_se"],
                    "R2_log_free": block_420_scale["free_fit"]["R2_log"],
                    "prefactor_fixed": block_420_scale["fixed_exponent_3_4_and_1"]["prefactor"],
                    "R2_log_fixed": block_420_scale["fixed_exponent_3_4_and_1"]["R2_log"],
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


# ---------------------------------------------------------------------------
# 6. Matched Reduced-Temperature Controls
# ---------------------------------------------------------------------------

def analyze_thermo_controls(df_ctrl: pd.DataFrame) -> Dict[str, Any]:
    """Summarize matched T/Tc = 0.82 vs 1.25 controls with Wilson CIs and Welch t-tests."""
    summary: List[Dict[str, Any]] = []
    for (lat, cv), grp in df_ctrl.groupby(["lattice", "Cv"]):
        temps = sorted(grp["T"].unique().tolist())
        entry: Dict[str, Any] = {"lattice": str(lat), "Cv": float(cv), "conditions": {}}
        vals_by_T = {}
        for T_val in temps:
            s = grp[np.isclose(grp["T"], T_val)]
            n = int(len(s))
            k = int(s["is_spanning"].sum())
            w_lo, w_hi = wilson_ci(k, n)
            tag = "below_Tc" if float(T_val) < float(LATTICE_SPECS[str(lat)]["Tc"]) else "above_Tc"
            vals_by_T[tag] = s["f_max_pct"].to_numpy(dtype=np.float64)
            cond_d: Dict[str, Any] = {
                "T": float(T_val),
                "T_over_Tc": float(s["T_over_Tc"].mean()),
                "n": n,
                "k_span": k,
                "f_max_pct_mean": _safe_float(s["f_max_pct"].mean()),
                "f_max_pct_sd": _safe_float(s["f_max_pct"].std(ddof=1)) if n > 1 else 0.0,
                "P_span": float(k) / float(n) if n > 0 else None,
                "P_span_wilson_lo": w_lo,
                "P_span_wilson_hi": w_hi,
                "C_index_mean": _safe_float(s["C_index"].mean()),
                "C_index_sd": _safe_float(s["C_index"].std(ddof=1)) if n > 1 else 0.0,
                "D_mean": _safe_float(s["D"].mean()),
                "D_se": _safe_float(s["D"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            }
            if "C_index_window_mean" in s.columns:
                cond_d["C_window_mean"] = _safe_float(s["C_index_window_mean"].mean())
            entry["conditions"][tag] = cond_d
        if "below_Tc" in vals_by_T and "above_Tc" in vals_by_T and len(vals_by_T["below_Tc"]) > 1:
            t_stat, p_val = stats.ttest_ind(vals_by_T["below_Tc"], vals_by_T["above_Tc"], equal_var=False)
            entry["welch_t_test_f_max"] = {"t": _safe_float(t_stat), "p_value": _safe_float(p_val)}
        summary.append(entry)
    return {"controls": summary}


# ---------------------------------------------------------------------------
# 7. Expanded Finite-Size FCC & Direct Domain-Coarsening Kinetics R(t)
# ---------------------------------------------------------------------------

def analyze_finite_size(df_fs: pd.DataFrame) -> Dict[str, Any]:
    """Summarize FCC Cv=0.20, T=2.00 finite-size comparison (L=20 n=15 vs L=40 n=10)."""
    out: Dict[str, Any] = {}
    groups_map = {}
    for L_val, grp in df_fs.groupby("L"):
        n = int(len(grp))
        k = int(grp["is_spanning"].sum())
        w_lo, w_hi = wilson_ci(k, n)
        groups_map[int(L_val)] = grp
        entry: Dict[str, Any] = {
            "L": int(L_val),
            "N": int(grp["N"].iloc[0]),
            "n": n,
            "k_span": k,
            "sweeps": int(grp["sweeps"].iloc[0]),
            "f_max_pct_mean": _safe_float(grp["f_max_pct"].mean()),
            "f_max_pct_sd": _safe_float(grp["f_max_pct"].std(ddof=1)) if n > 1 else 0.0,
            "f_max_pct_se": _safe_float(grp["f_max_pct"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "P_span": float(k) / float(n) if n > 0 else None,
            "P_span_wilson_lo": w_lo,
            "P_span_wilson_hi": w_hi,
            "C_index_mean": _safe_float(grp["C_index"].mean()),
            "C_index_sd": _safe_float(grp["C_index"].std(ddof=1)) if n > 1 else 0.0,
            "C_index_se": _safe_float(grp["C_index"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "D_mean": _safe_float(grp["D"].mean()),
            "D_sd": _safe_float(grp["D"].std(ddof=1)) if n > 1 else 0.0,
            "D_se": _safe_float(grp["D"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        }
        if "C_index_window_mean" in grp.columns:
            entry["C_window_mean"] = _safe_float(grp["C_index_window_mean"].mean())
            entry["C_window_sd"] = _safe_float(grp["C_index_window_mean"].std(ddof=1)) if n > 1 else 0.0
            entry["C_window_se"] = _safe_float(grp["C_index_window_mean"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        out[f"L={int(L_val)}"] = entry

    if 20 in groups_map and 40 in groups_map:
        g20, g40 = groups_map[20], groups_map[40]
        comparisons = {}
        for col in ("f_max_pct", "C_index", "D"):
            t_s, p_s = stats.ttest_ind(g20[col], g40[col], equal_var=False)
            rel_pct = 100.0 * (float(g40[col].mean()) - float(g20[col].mean())) / float(g20[col].mean())
            comparisons[col] = {
                "rel_diff_L40_vs_L20_pct": rel_pct,
                "welch_t": _safe_float(t_s),
                "welch_p": _safe_float(p_s),
            }
        out["L40_vs_L20_tests"] = comparisons

    return out


def analyze_coarsening_kinetics(df_coarsen: pd.DataFrame) -> Dict[str, Any]:
    """Analyze direct time-resolved domain growth R_S(t) and R_int(t) power laws."""
    out: Dict[str, Any] = {}
    for L_val, grp in df_coarsen.groupby("L"):
        L_int = int(L_val)
        by_t = (
            grp.groupby("t_mcs", as_index=False)
            .agg(
                n=("R_S", "count"),
                R_S_mean=("R_S", "mean"),
                R_S_sd=("R_S", lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
                R_S_se=("R_S", lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0),
                k1_mean=("k1", "mean"),
                R_int_mean=("R_int", "mean"),
                R_int_sd=("R_int", lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
                R_int_se=("R_int", lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0),
                C_index_mean=("C_index", "mean"),
                f_max_pct_mean=("f_max_pct", "mean"),
                f_max_pct_sd=("f_max_pct", lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
                P_span=("is_spanning", "mean"),
            )
            .sort_values("t_mcs")
        )

        # Fit effective growth exponents z_eff over [1000, 10000] and [2000, 10000] MCS
        fits: Dict[str, Any] = {}
        for metric in ("R_S", "R_int"):
            fits[metric] = {}
            for t_min in (1000, 2000):
                sub_t = by_t[by_t["t_mcs"] >= t_min]
                if len(sub_t) >= 3:
                    lt = np.log(sub_t["t_mcs"].to_numpy(dtype=np.float64))
                    lr = np.log(sub_t[f"{metric}_mean"].to_numpy(dtype=np.float64))
                    fm = _ols_fit(np.column_stack([np.ones_like(lt), lt]), lr)

                    # Also compute per-seed slopes to obtain independent-trajectory SE and 95% CI
                    seed_slopes = []
                    for _, s_df in grp[grp["t_mcs"] >= t_min].groupby("seed_idx"):
                        if len(s_df) >= 3:
                            s_lt = np.log(s_df["t_mcs"].to_numpy(dtype=np.float64))
                            s_lr = np.log(s_df[metric].to_numpy(dtype=np.float64))
                            slope_i = float(np.polyfit(s_lt, s_lr, 1)[0])
                            seed_slopes.append(slope_i)
                    seed_arr = np.asarray(seed_slopes, dtype=np.float64)
                    n_s = len(seed_arr)
                    z_seed_mean = float(np.mean(seed_arr)) if n_s > 0 else float("nan")
                    z_seed_se = float(np.std(seed_arr, ddof=1) / np.sqrt(n_s)) if n_s > 1 else float("nan")
                    t_crit = float(stats.t.ppf(0.975, df=n_s - 1)) if n_s > 1 else float("nan")

                    fits[metric][f"t>={t_min}"] = {
                        "n_timepoints": int(len(sub_t)),
                        "n_seeds": int(n_s),
                        "z_eff_mean_curve": _safe_float(fm["beta"][1]),
                        "z_eff_mean_curve_se": _safe_float(fm["se"][1]),
                        "prefactor_mean_curve": _safe_float(np.exp(fm["beta"][0])),
                        "R2_mean_curve": _safe_float(fm["r2"]),
                        "z_eff_per_seed_mean": _safe_float(z_seed_mean),
                        "z_eff_per_seed_se": _safe_float(z_seed_se),
                        "z_eff_ci95_lo": _safe_float(z_seed_mean - t_crit * z_seed_se) if n_s > 1 else None,
                        "z_eff_ci95_hi": _safe_float(z_seed_mean + t_crit * z_seed_se) if n_s > 1 else None,
                    }

        out[f"L={L_int}"] = {
            "L": L_int,
            "N": int(grp["N"].iloc[0]),
            "n_seeds_at_6000": int(by_t.loc[by_t["t_mcs"] == 6000, "n"].iloc[0]) if (by_t["t_mcs"] == 6000).any() else 0,
            "n_seeds_at_10000": int(by_t.loc[by_t["t_mcs"] == 10000, "n"].iloc[0]) if (by_t["t_mcs"] == 10000).any() else 0,
            "power_law_fits": fits,
            "time_series": by_t.to_dict(orient="records"),
        }
    return out


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_statistics(data_dir: Path, out_json: Path) -> Dict[str, Any]:
    """Run the complete statistical pipeline and save summary_stats.json."""
    seed_path = data_dir / "seed_level_465.csv"
    dense_span_path = data_dir / "spanning_dense_subcritical.csv"
    hp_perc_path = data_dir / "percolation_high_precision.csv"
    j0_path = data_dir / "random_percolation_j0.csv"
    dilute_path = data_dir / "dilute_validation.csv"
    ctrl_path = data_dir / "thermo_controls.csv"
    fs_path = data_dir / "finite_size_fcc.csv"
    coarsen_path = data_dir / "coarsening_kinetics.csv"

    df_seed = pd.read_csv(seed_path)
    df_span = pd.read_csv(dense_span_path) if dense_span_path.exists() else df_seed
    df_perc = pd.read_csv(hp_perc_path) if hp_perc_path.exists() else None
    df_j0 = pd.read_csv(j0_path) if j0_path.exists() else None

    j0_results = None
    j0_draws = None
    if df_j0 is not None:
        j0_results, j0_draws = analyze_j0_percolation(df_j0, n_boot=2000)

    results: Dict[str, Any] = {
        "dataset_rows": {
            "seed_level_465": int(len(df_seed)),
            "spanning_dense_subcritical": int(len(df_span)),
        },
        "dilute_convergence": analyze_dilute_convergence(df_seed),
        "spanning_crossings": analyze_spanning_crossings(
            df_span, df_perc=df_perc, j0_results=j0_results, j0_boot_draws=j0_draws, n_boot=2000
        ),
        "fcc_cv020_three_way": analyze_fcc_three_way_cv020(df_seed, df_span, df_j0),
        "structure_transport": analyze_structure_transport(df_seed),
        "reduced_scaling": analyze_reduced_scaling(df_seed),
    }

    if df_perc is not None:
        results["dataset_rows"]["percolation_high_precision"] = int(len(df_perc))
    if j0_results is not None and df_j0 is not None:
        results["dataset_rows"]["random_percolation_j0"] = int(len(df_j0))
        results["random_percolation_j0"] = j0_results
    if dilute_path.exists():
        df_dil = pd.read_csv(dilute_path)
        results["dataset_rows"]["dilute_validation"] = int(len(df_dil))
        results["dilute_validation"] = analyze_dilute_validation(df_dil)
    if ctrl_path.exists():
        df_ctrl = pd.read_csv(ctrl_path)
        results["dataset_rows"]["thermo_controls"] = int(len(df_ctrl))
        results["thermo_controls"] = analyze_thermo_controls(df_ctrl)
    if fs_path.exists():
        df_fs = pd.read_csv(fs_path)
        results["dataset_rows"]["finite_size_fcc"] = int(len(df_fs))
        results["finite_size_fcc"] = analyze_finite_size(df_fs)
    if coarsen_path.exists():
        df_c = pd.read_csv(coarsen_path)
        results["dataset_rows"]["coarsening_kinetics"] = int(len(df_c))
        results["coarsening_kinetics"] = analyze_coarsening_kinetics(df_c)

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
    rs_ref = stats_dict["reduced_scaling"]["two_parameter_transport_Dref"]
    ff = rs_ref["free_fit"]
    fx = rs_ref["fixed_exponent_3_4_and_1"]
    cv = rs_ref["cross_validation"]

    print(f"Exported summary statistics -> {args.output}")
    print(f"  Canonical rows analyzed: {stats_dict['dataset_rows']['seed_level_465']}")
    print("  Logistic MLE spanning crossings (P_span = 0.5, 95% bootstrap CI):")
    for tag in ("SC", "BCC", "FCC", "FCC_above_Tc"):
        if tag in sc:
            d = sc[tag]
            print(
                f"    {tag:12s}: Cv* = {d['Cv_star']:.4f} "
                f"[{d['Cv_star_ci95_lo']:.4f}, {d['Cv_star_ci95_hi']:.4f}], "
                f"Cv*/pc = {d['Cv_star_over_pc']:.3f}"
            )
    print(
        f"  Primary zero-free-param scaling (D/D_ref): alpha = {ff['alpha']:.4f} +/- {ff['alpha_se']:.4f}, "
        f"beta = {ff['beta_T_over_Tc']:.4f} +/- {ff['beta_se']:.4f}, "
        f"R^2_log(free) = {ff['R2_log']:.4f}, R^2_log(3/4,1) = {fx['R2_log']:.4f}, "
        f"LOCO-CV R^2_log(free) = {cv['LOCO_31fold']['free_fit']['R2_log']:.4f}, "
        f"LOLO-CV R^2_log(free) = {cv['LOLO_3fold_overall']['free_fit']['R2_log']:.4f}"
    )


if __name__ == "__main__":
    main()
