"""Comprehensive reviewer-hardening simulation suite for SC, BCC, and FCC lattice gases.

Executes five deterministic, reproducible computational experiments:
  1. Dilute-limit quantitative validation against analytical (1/3) * f0 * r_nn^2
     (L=20, T=100, Cv in {0.015, 0.030}, n=15 seeds each, 2,000 MCS)
     -> data/dilute_validation.csv
  2. Non-interacting (J=0) finite-size random site-percolation benchmark
     using identical periodic supercells and exact BFS topological wrapping
     (L=20 SC/BCC/FCC n=200 per Cv; plus L=30, 40 FCC n=100 per Cv)
     -> data/random_percolation_j0.csv
  3. High-precision interacting (J=1) percolation crossover sweeps
     (n=50 seeds per concentration across tight grids bracketing P_span=0.5
     for SC/BCC/FCC below Tc and FCC above Tc, L=20, 6,000 MCS)
     -> data/percolation_high_precision.csv
  4. Time-aligned production-window clustering (C_index_window_mean over [2000, 6000] MCS)
     across all 465 canonical seeds (deduplicated with Exp. 3, bit-identical trajectories)
     -> updates data/seed_level_465.csv, data/fcc_concentration_sweep.csv, data/thermo_controls.csv
  5. Expanded L=40 FCC ensemble (n=10 seeds at 6,000 MCS) AND direct time-resolved
     domain-growth coarsening kinetics R_S(t) (via 3D static structure factor S(k, t))
     and R_int(t) (inverse atom-vacancy interface density) up to t=10,000 MCS
     -> updates data/finite_size_fcc.csv and writes data/coarsening_kinetics.csv
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.kawasaki_mc import (
    build_exp_table,
    compute_c_index_fast,
    compute_msd,
    compute_total_energy,
    init_rng_state,
    initialize_lattice_state,
    kawasaki_sweeps_kernel,
    simulate_trajectory,
)
from src.lattices import LATTICE_SPECS, Lattice, build_lattice
from src.observables import (
    compute_cluster_observables,
    compute_local_clustering_index,
    compute_tracer_diffusivity,
)
from src.run_sweeps import (
    Condition,
    _collect_dataframe,
    _warmup_numba_kernels,
    derive_deterministic_seed,
    execute_jobs,
    get_fcc_sweep_conditions,
    get_seed_level_465_conditions,
    get_thermo_control_conditions,
)


BASE_SEED = 20260808


# ---------------------------------------------------------------------------
# 1. Exact 3D Static Structure Factor Characteristic Domain Length R_S(t)
# ---------------------------------------------------------------------------

def compute_structure_factor_length(occ: np.ndarray, lattice: Lattice) -> Tuple[float, float, float]:
    """Compute spherically averaged 3D structure-factor domain length R_S = 2*pi / k1.

    Projects vacancy occupancy (1 - occ) onto the L x L x L conventional unit-cell
    grid (averaging over the n_basis sites per cell), computes the exact 3D FFT
    power spectrum S(k) = |F(k)|^2, and evaluates the first moment
        k1 = sum_{k != 0} |k| S(k) / sum_{k != 0} S(k)
    where |k| = (2*pi / L) * sqrt(nx^2 + ny^2 + nz^2) in inverse unit-cell units (1/a).
    Also returns the exact inverse atom-vacancy interface density R_int = 1 / <z_av>.
    """
    L = lattice.L
    n_basis = lattice.n_basis
    vac_field = (1.0 - occ.astype(np.float64)).reshape(L, L, L, n_basis).mean(axis=-1)
    delta = vac_field - float(np.mean(vac_field))

    fk = np.fft.rfftn(delta)
    sk = (fk.real ** 2 + fk.imag ** 2)

    kx = 2.0 * math.pi * np.fft.fftfreq(L, d=1.0)
    ky = 2.0 * math.pi * np.fft.fftfreq(L, d=1.0)
    kz = 2.0 * math.pi * np.fft.rfftfreq(L, d=1.0)
    kmag = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)

    # Account for Hermitian symmetry weights on rfftn last axis (0 and L/2 have weight 1, interior weight 2)
    w = np.full(kz.shape, 2.0, dtype=np.float64)
    w[0] = 1.0
    if L % 2 == 0:
        w[-1] = 1.0
    sk_w = sk * w[None, None, :]
    sk_w[0, 0, 0] = 0.0

    s_tot = float(np.sum(sk_w))
    if s_tot <= 0.0:
        r_s = float("nan")
        k1 = float("nan")
    else:
        k1 = float(np.sum(kmag * sk_w)) / s_tot
        r_s = (2.0 * math.pi) / k1 if k1 > 0.0 else float("nan")

    # Exact interface density <z_av> = (Z * Cv / (1 - Cv)) * (1 - Cv * C)
    c_idx, _ = compute_local_clustering_index(occ, lattice)
    cv = float(np.mean(occ == 0))
    z_av = (float(lattice.Z) * cv / (1.0 - cv)) * (1.0 - cv * c_idx) if cv < 1.0 else 0.0
    r_int = (1.0 / z_av) if z_av > 0.0 else float("nan")
    return r_s, k1, r_int


# ---------------------------------------------------------------------------
# 2. Experiment 1: Dilute-Limit Validation (T=100, Cv in {0.015, 0.030})
# ---------------------------------------------------------------------------

def _run_dilute_job(job: Tuple[str, int, float, float, int, int, int, int]) -> Dict[str, object]:
    lat_name, L, Cv, T, seed_idx, rng_seed, total_sweeps, burn_in = job
    lat = build_lattice(lat_name, L)
    sim = simulate_trajectory(lat, Cv=Cv, T=T, seed=rng_seed, total_sweeps=total_sweeps, burn_in=burn_in, n_msd_samples=25)
    diff = compute_tracer_diffusivity(sim.msd_times, sim.msd_values, sim.actual_Cv, lat.name)
    spec = LATTICE_SPECS[lat.name]
    d_ref = (1.0 / 3.0) * float(spec["f0_dilute"]) * float(spec["nn_dist_sq"])
    return {
        "lattice": lat.name,
        "L": lat.L,
        "N": lat.N,
        "Z": lat.Z,
        "Cv": Cv,
        "actual_Cv": sim.actual_Cv,
        "T": T,
        "seed_idx": seed_idx,
        "rng_seed": rng_seed,
        "sweeps": total_sweeps,
        "burn_in": burn_in,
        "D": diff.D,
        "D_over_Cv": diff.D_over_Cv,
        "D_ref_analytical": d_ref,
        "rel_error": (diff.D_over_Cv - d_ref) / d_ref,
        "msd_r2": diff.msd_r2,
        "msd_loglog_slope": diff.msd_loglog_slope,
    }


def run_dilute_validation(out_dir: Path, workers: int, n_seeds: int = 15) -> pd.DataFrame:
    jobs = []
    for lat_name in ("SC", "BCC", "FCC"):
        for Cv in (0.015, 0.030):
            for s_idx in range(n_seeds):
                rng_seed = derive_deterministic_seed(BASE_SEED + 101, lat_name, 20, Cv, 100.0, s_idx)
                jobs.append((lat_name, 20, Cv, 100.0, s_idx, rng_seed, 2000, 400))

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run_dilute_job, j) for j in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc="Exp 1: Dilute T=100", unit="sim"):
            rows.append(f.result())

    df = pd.DataFrame(rows)
    ord_map = {"SC": 0, "BCC": 1, "FCC": 2}
    df["_o"] = df["lattice"].map(ord_map)
    df = df.sort_values(["_o", "Cv", "seed_idx"]).drop(columns=["_o"]).reset_index(drop=True)
    out_path = out_dir / "dilute_validation.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")
    return df


# ---------------------------------------------------------------------------
# 3. Experiment 2: Non-Interacting (J=0) Random Percolation Control
# ---------------------------------------------------------------------------

def _run_j0_batch(job: Tuple[str, int, float, int]) -> List[Dict[str, object]]:
    lat_name, L, Cv, n_realizations = job
    lat = build_lattice(lat_name, L)
    rows: List[Dict[str, object]] = []
    for s_idx in range(n_realizations):
        rng_seed = derive_deterministic_seed(BASE_SEED + 202, lat_name, L, Cv, 0.0, s_idx)
        occ, _, _ = initialize_lattice_state(lat, Cv, rng_seed)
        c_idx, mean_zv = compute_local_clustering_index(occ, lat)
        clust = compute_cluster_observables(occ, lat)
        rows.append({
            "lattice": lat.name,
            "L": lat.L,
            "N": lat.N,
            "Z": lat.Z,
            "J": 0.0,
            "Cv": round(Cv, 6),
            "actual_Cv": float(clust.n_vac) / float(lat.N),
            "pc": lat.pc,
            "seed_idx": s_idx,
            "C_index": c_idx,
            "mean_zv": mean_zv,
            "f_max": clust.f_max,
            "f_max_pct": clust.f_max_pct,
            "is_spanning": clust.is_spanning,
            "n_wrap_dims": clust.n_wrap_dims,
        })
    return rows


def run_random_percolation_j0(out_dir: Path, workers: int) -> pd.DataFrame:
    grids = [
        ("SC", 20, (0.20, 0.28, 0.29, 0.295, 0.30, 0.305, 0.31, 0.315, 0.32, 0.325, 0.33, 0.34), 200),
        ("BCC", 20, (0.20, 0.22, 0.23, 0.235, 0.24, 0.245, 0.25, 0.255, 0.26, 0.27), 200),
        ("FCC", 20, (0.15, 0.175, 0.185, 0.19, 0.195, 0.198, 0.20, 0.202, 0.205, 0.21, 0.215, 0.225), 200),
        ("FCC", 30, (0.190, 0.195, 0.198, 0.200, 0.202, 0.205, 0.210), 100),
        ("FCC", 40, (0.190, 0.195, 0.198, 0.200, 0.202, 0.205, 0.210), 100),
    ]
    jobs = []
    for lat_name, L, cvs, n_rep in grids:
        for Cv in cvs:
            jobs.append((lat_name, L, float(Cv), n_rep))

    all_rows: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run_j0_batch, j) for j in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc="Exp 2: J=0 random perc", unit="grid"):
            all_rows.extend(f.result())

    df = pd.DataFrame(all_rows)
    ord_map = {"SC": 0, "BCC": 1, "FCC": 2}
    df["_o"] = df["lattice"].map(ord_map)
    df = df.sort_values(["_o", "L", "Cv", "seed_idx"]).drop(columns=["_o"]).reset_index(drop=True)
    out_path = out_dir / "random_percolation_j0.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")
    return df


# ---------------------------------------------------------------------------
# 4. Experiments 3 & 4: High-Precision Percolation (n=50) + Time-Aligned 465
# ---------------------------------------------------------------------------

def get_high_precision_percolation_conditions() -> List[Condition]:
    conds: List[Condition] = []
    for Cv in (0.18, 0.19, 0.20, 0.21, 0.22, 0.23):
        conds.append(Condition("SC", 20, Cv, 0.93, "subcritical_0.82Tc"))
    for Cv in (0.20, 0.21, 0.22, 0.23, 0.24, 0.25):
        conds.append(Condition("BCC", 20, Cv, 1.30, "subcritical_0.82Tc"))
    for Cv in (0.185, 0.190, 0.195, 0.200, 0.205, 0.210, 0.215):
        conds.append(Condition("FCC", 20, Cv, 2.00, "subcritical_0.82Tc"))
    for Cv in (0.165, 0.170, 0.175, 0.180, 0.185, 0.190):
        conds.append(Condition("FCC", 20, Cv, 3.06, "supercritical_1.25Tc"))
    return conds


def run_percolation_and_time_aligned_465(
    out_dir: Path, workers: int, n_perc_seeds: int = 50
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    conds_465 = get_seed_level_465_conditions()
    conds_perc = get_high_precision_percolation_conditions()
    conds_fcc = get_fcc_sweep_conditions()
    conds_ctrl = get_thermo_control_conditions()

    # Deduplicate jobs so overlapping seed_idx 0..14 share bit-exact trajectories
    unique_jobs: Dict[Tuple[str, int, float, float, int], Tuple[str, int, float, float, int, int, int, int, int]] = {}

    def register(conds: Sequence[Condition], n_s: int) -> None:
        for c in conds:
            for s_idx in range(n_s):
                key = (c.lattice, c.L, round(c.Cv, 6), round(c.T, 6), s_idx)
                if key not in unique_jobs:
                    rng_seed = derive_deterministic_seed(BASE_SEED, c.lattice, c.L, c.Cv, c.T, s_idx)
                    unique_jobs[key] = (c.lattice, c.L, c.Cv, c.T, s_idx, rng_seed, 6000, 2000, 25)

    register(conds_465, 15)
    register(conds_perc, n_perc_seeds)

    print(f"Exp 3+4: Executing {len(unique_jobs)} unique L=20 6,000-MCS trajectories (workers={workers})...")
    results_map = execute_jobs(list(unique_jobs.values()), workers=workers, desc="Exp 3+4: L=20 6k MCS")

    # Save updated canonical 465-seed CSVs (now with C_index_window_mean & C_index_window_sd)
    df_465 = _collect_dataframe(conds_465, {20: 15}, results_map)
    df_fcc = _collect_dataframe(conds_fcc, {20: 15}, results_map)
    df_ctrl = _collect_dataframe(conds_ctrl, {20: 15}, results_map)
    df_perc = _collect_dataframe(conds_perc, {20: n_perc_seeds}, results_map)

    df_465.to_csv(out_dir / "seed_level_465.csv", index=False)
    df_fcc.to_csv(out_dir / "fcc_concentration_sweep.csv", index=False)
    df_ctrl.to_csv(out_dir / "thermo_controls.csv", index=False)
    df_perc.to_csv(out_dir / "percolation_high_precision.csv", index=False)

    # Also build combined spanning dataset merging high-precision (n=50) with outer tail points (n=15)
    tail_rows = []
    perc_keys = {(r["lattice"], round(float(r["T"]), 4), round(float(r["Cv"]), 6)) for _, r in df_perc[["lattice", "T", "Cv"]].drop_duplicates().iterrows()}
    for _, row in df_465.iterrows():
        lat = str(row["lattice"])
        T = round(float(row["T"]), 4)
        Cv = round(float(row["Cv"]), 6)
        keep_T = (
            (lat == "SC" and abs(T - 0.93) < 0.02)
            or (lat == "BCC" and abs(T - 1.30) < 0.02)
            or (lat == "FCC" and (abs(T - 2.00) < 0.02 or abs(T - 3.06) < 0.02))
        )
        if keep_T and (lat, T, Cv) not in perc_keys:
            tail_rows.append(row.to_dict())
    df_span_all = pd.concat([df_perc, pd.DataFrame(tail_rows)], ignore_index=True)
    ord_map = {"SC": 0, "BCC": 1, "FCC": 2}
    df_span_all["_o"] = df_span_all["lattice"].map(ord_map)
    df_span_all = df_span_all.sort_values(["_o", "T", "Cv", "seed_idx"]).drop(columns=["_o"]).reset_index(drop=True)
    df_span_all.to_csv(out_dir / "spanning_dense_subcritical.csv", index=False)

    print(
        f"Wrote updated seed_level_465 ({len(df_465)}), fcc_sweep ({len(df_fcc)}), "
        f"thermo_controls ({len(df_ctrl)}), percolation_high_precision ({len(df_perc)}), "
        f"spanning_dense_subcritical ({len(df_span_all)})"
    )
    return df_465, df_perc


# ---------------------------------------------------------------------------
# 5. Experiment 5: Expanded L=40 (n=10) + Time-Resolved Domain Growth R(t)
# ---------------------------------------------------------------------------

COARSENING_TIMES: Tuple[int, ...] = (250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000)


def _run_coarsening_and_fs_job(
    job: Tuple[int, int, int, bool]
) -> Tuple[List[Dict[str, object]], Dict[str, object] | None]:
    """Run FCC Cv=0.20, T=2.00 up to max_sweeps; collect R_S(t), R_int(t), and 6,000-MCS fs row."""
    L, seed_idx, max_sweeps, need_6k_row = job
    lat = build_lattice("FCC", L)
    Cv = 0.20
    T = 2.00
    rng_seed = derive_deterministic_seed(BASE_SEED, "FCC", L, Cv, T, seed_idx)

    occ, site_to_atom, unwrapped_disp = initialize_lattice_state(lat, Cv, rng_seed)
    mc_seed = np.uint64((int(rng_seed) ^ 0xA5A5A5A55A5A5A5A) & 0xFFFFFFFFFFFFFFFF)
    rng_state = init_rng_state(mc_seed)
    exp_table = build_exp_table(lat.Z, T, J=1.0)

    # Exact checkpoints matching simulate_trajectory(6000, burn_in=2000, n_msd_samples=25)
    msd_rel_checkpoints = np.unique(np.round(np.linspace(1, 4000, 25)).astype(np.int64))
    if msd_rel_checkpoints[-1] != 4000:
        msd_rel_checkpoints = np.append(msd_rel_checkpoints, 4000)
    msd_abs_set = set(int(2000 + dt) for dt in msd_rel_checkpoints)
    coarsen_set = set(t for t in COARSENING_TIMES if t <= max_sweeps)
    all_stops = sorted(msd_abs_set | coarsen_set | {2000, min(6000, max_sweeps)})

    disp_ref = None
    msd_times: List[float] = []
    msd_vals: List[float] = []
    c_win_vals: List[float] = []
    coarsen_rows: List[Dict[str, object]] = []
    fs_row: Dict[str, object] | None = None

    accepted_total = 0
    prev_t = 0
    for t_now in all_stops:
        step = t_now - prev_t
        if step > 0:
            acc, _ = kawasaki_sweeps_kernel(
                occ, site_to_atom, unwrapped_disp, lat.neighbors, lat.displacements, exp_table, rng_state, step
            )
            accepted_total += int(acc)
            prev_t = t_now

        if t_now == 2000:
            disp_ref = unwrapped_disp.copy()

        if t_now in msd_abs_set and disp_ref is not None:
            msd_times.append(float(t_now - 2000))
            msd_vals.append(float(compute_msd(unwrapped_disp, disp_ref)))
            c_win_vals.append(float(compute_c_index_fast(occ, lat.neighbors)))

        if t_now in coarsen_set:
            r_s, k1, r_int = compute_structure_factor_length(occ, lat)
            c_idx, mean_zv = compute_local_clustering_index(occ, lat)
            clust = compute_cluster_observables(occ, lat)
            coarsen_rows.append({
                "lattice": "FCC",
                "L": L,
                "N": lat.N,
                "Cv": Cv,
                "T": T,
                "seed_idx": seed_idx,
                "t_mcs": t_now,
                "R_S": r_s,
                "k1": k1,
                "R_int": r_int,
                "C_index": c_idx,
                "f_max_pct": clust.f_max_pct,
                "is_spanning": clust.is_spanning,
            })

        if need_6k_row and t_now == 6000:
            actual_cv = float(np.sum(occ == 0)) / float(lat.N)
            diff = compute_tracer_diffusivity(np.asarray(msd_times), np.asarray(msd_vals), actual_cv, "FCC")
            c_idx, mean_zv = compute_local_clustering_index(occ, lat)
            clust = compute_cluster_observables(occ, lat)
            e_tot = float(compute_total_energy(occ, lat.neighbors, J=1.0))
            fs_row = {
                "lattice": "FCC",
                "L": L,
                "N": lat.N,
                "Z": lat.Z,
                "Cv": Cv,
                "actual_Cv": actual_cv,
                "T": T,
                "Tc": lat.Tc,
                "pc": lat.pc,
                "D0": lat.D0,
                "T_over_Tc": T / lat.Tc,
                "Cv_over_pc": actual_cv / lat.pc,
                "Lambda": (actual_cv / lat.pc) * (lat.Tc / T),
                "seed_idx": seed_idx,
                "rng_seed": int(rng_seed),
                "sweeps": 6000,
                "burn_in": 2000,
                "D": diff.D,
                "D_over_Cv": diff.D_over_Cv,
                "D_over_D0": diff.D_over_D0,
                "msd_r2": diff.msd_r2,
                "msd_loglog_slope": diff.msd_loglog_slope,
                "is_diffusive": int(diff.is_diffusive),
                "C_index": c_idx,
                "C_index_window_mean": float(np.mean(c_win_vals)),
                "C_index_window_sd": float(np.std(c_win_vals, ddof=1)) if len(c_win_vals) > 1 else 0.0,
                "mean_zv": mean_zv,
                "n_clusters": clust.n_clusters,
                "largest_cluster_size": clust.largest_cluster_size,
                "f_max": clust.f_max,
                "f_max_pct": clust.f_max_pct,
                "is_spanning": clust.is_spanning,
                "wraps_x": int(clust.wraps_x),
                "wraps_y": int(clust.wraps_y),
                "wraps_z": int(clust.wraps_z),
                "n_wrap_dims": clust.n_wrap_dims,
                "acceptance_rate": float(accepted_total) / float(6000 * lat.N),
                "final_energy_per_site": e_tot / float(lat.N),
                "regime_tag": f"finite_size_L{L}",
            }

    return coarsen_rows, fs_row


def run_coarsening_and_expanded_l40(out_dir: Path, workers: int, df_465: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # L=20: 15 seeds up to 10,000 MCS for R(t) coarsening (already have L=20 6k rows in df_465)
    # L=40: 5 seeds (0..4) up to 10,000 MCS (also gives 6k fs rows with C_window) + 5 seeds (5..9) up to 6,000 MCS
    jobs: List[Tuple[int, int, int, bool]] = []
    for s_idx in range(15):
        jobs.append((20, s_idx, 10000, False))
    for s_idx in range(5):
        jobs.append((40, s_idx, 10000, True))
    for s_idx in range(5, 10):
        jobs.append((40, s_idx, 6000, True))

    coarsen_all: List[Dict[str, object]] = []
    l40_fs_rows: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(workers, 7)) as pool:
        futs = [pool.submit(_run_coarsening_and_fs_job, j) for j in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc="Exp 5: Coarsening & L=40", unit="sim"):
            c_rows, fs_r = f.result()
            coarsen_all.extend(c_rows)
            if fs_r is not None:
                l40_fs_rows.append(fs_r)

    df_coarsen = pd.DataFrame(coarsen_all).sort_values(["L", "t_mcs", "seed_idx"]).reset_index(drop=True)
    df_coarsen.to_csv(out_dir / "coarsening_kinetics.csv", index=False)

    # Combine L=20 (n=15) from df_465 + L=40 (n=10) from l40_fs_rows
    l20_rows = df_465[
        (df_465["lattice"] == "FCC") & (np.isclose(df_465["Cv"], 0.20)) & (np.isclose(df_465["T"], 2.00))
    ].copy()
    l20_rows["regime_tag"] = "finite_size_L20"
    df_fs = pd.concat([l20_rows, pd.DataFrame(l40_fs_rows)], ignore_index=True)
    df_fs = df_fs.sort_values(["L", "seed_idx"]).reset_index(drop=True)
    df_fs.to_csv(out_dir / "finite_size_fcc.csv", index=False)

    print(f"Wrote coarsening_kinetics ({len(df_coarsen)} rows) & expanded finite_size_fcc ({len(df_fs)} rows: L20=15, L40=10)")
    return df_coarsen, df_fs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all reviewer-hardening computational experiments.")
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--perc-seeds", type=int, default=50)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _warmup_numba_kernels()

    run_dilute_validation(args.output_dir, workers=args.workers, n_seeds=15)
    run_random_percolation_j0(args.output_dir, workers=args.workers)
    df_465, _ = run_percolation_and_time_aligned_465(args.output_dir, workers=args.workers, n_perc_seeds=args.perc_seeds)
    run_coarsening_and_expanded_l40(args.output_dir, workers=args.workers, df_465=df_465)
    print("ALL 5 REVIEWER EXPERIMENTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
