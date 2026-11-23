"""CLI runner for deterministic, multiprocessed Kawasaki Monte Carlo sweeps.

Generates structured CSV datasets in `data/`:
  - `data/seed_level_465.csv`:         31 (lattice, Cv, T) conditions x 15 seeds = 465 rows
  - `data/fcc_concentration_sweep.csv`: FCC sweep over Cv in {0.14..0.25} at T=2.00, 3.06
  - `data/thermo_controls.csv`:        Matched T/Tc = 0.82 vs 1.25 controls at Cv=0.20 (+ FCC Cv=0.15)
  - `data/finite_size_fcc.csv`:        FCC Cv=0.20, T=2.00 comparing L=20 vs L=40
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from src.kawasaki_mc import simulate_trajectory
from src.lattices import LATTICE_SPECS, build_lattice
from src.observables import (
    compute_cluster_observables,
    compute_local_clustering_index,
    compute_tracer_diffusivity,
)


@dataclass(frozen=True)
class Condition:
    """A single (lattice, L, Cv, T) simulation condition."""

    lattice: str
    L: int
    Cv: float
    T: float
    regime_tag: str = ""

    def key(self) -> Tuple[str, int, float, float]:
        return (self.lattice, self.L, round(self.Cv, 6), round(self.T, 6))


FCC_CONCENTRATIONS: Tuple[float, ...] = (0.14, 0.15, 0.16, 0.18, 0.20, 0.22, 0.24, 0.25)
SC_BCC_CONCENTRATIONS: Tuple[float, ...] = (0.15, 0.20, 0.25)


def get_seed_level_465_conditions() -> List[Condition]:
    """Return the exact 31 (lattice, Cv, T) conditions (31 x 15 = 465 rows)."""
    conditions: List[Condition] = []

    # SC (Tc = 1.13, pc = 0.3116): 3 temperatures x 3 concentrations = 9 conditions
    for T, tag in ((0.93, "subcritical_0.82Tc"), (1.41, "supercritical_1.25Tc"), (2.00, "highT_1.77Tc")):
        for Cv in SC_BCC_CONCENTRATIONS:
            conditions.append(Condition("SC", 20, Cv, T, tag))

    # BCC (Tc = 1.59, pc = 0.2455): 2 temperatures x 3 concentrations = 6 conditions
    for T, tag in ((1.30, "subcritical_0.82Tc"), (1.99, "supercritical_1.25Tc")):
        for Cv in SC_BCC_CONCENTRATIONS:
            conditions.append(Condition("BCC", 20, Cv, T, tag))

    # FCC (Tc = 2.45, pc = 0.1996): 2 temperatures x 8 concentrations = 16 conditions
    for T, tag in ((2.00, "subcritical_0.82Tc"), (3.06, "supercritical_1.25Tc")):
        for Cv in FCC_CONCENTRATIONS:
            conditions.append(Condition("FCC", 20, Cv, T, tag))

    assert len(conditions) == 31, f"Expected 31 conditions, got {len(conditions)}"
    return conditions


def get_fcc_sweep_conditions() -> List[Condition]:
    """Return the 16 FCC concentration sweep conditions (8 Cv x 2 temperatures)."""
    return [c for c in get_seed_level_465_conditions() if c.lattice == "FCC"]


def get_thermo_control_conditions() -> List[Condition]:
    """Return the 8 matched reduced-temperature control conditions."""
    return [
        Condition("SC", 20, 0.20, 0.93, "below_Tc"),
        Condition("SC", 20, 0.20, 1.41, "above_Tc"),
        Condition("BCC", 20, 0.20, 1.30, "below_Tc"),
        Condition("BCC", 20, 0.20, 1.99, "above_Tc"),
        Condition("FCC", 20, 0.20, 2.00, "below_Tc"),
        Condition("FCC", 20, 0.20, 3.06, "above_Tc"),
        Condition("FCC", 20, 0.15, 2.00, "below_Tc"),
        Condition("FCC", 20, 0.15, 3.06, "above_Tc"),
    ]


def get_finite_size_conditions() -> List[Condition]:
    """Return FCC Cv=0.20, T=2.00 finite-size comparison conditions (L=20 vs L=40)."""
    return [
        Condition("FCC", 20, 0.20, 2.00, "finite_size_L20"),
        Condition("FCC", 40, 0.20, 2.00, "finite_size_L40"),
    ]


def derive_deterministic_seed(
    base_seed: int,
    lattice: str,
    L: int,
    Cv: float,
    T: float,
    seed_idx: int,
) -> int:
    """Derive a collision-free, schedule-independent 64-bit seed per simulation job."""
    payload = f"{base_seed}|{lattice.upper()}|L={L}|Cv={Cv:.6f}|T={T:.6f}|s={seed_idx}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _run_single_job(
    job: Tuple[str, int, float, float, int, int, int, int, int]
) -> Dict[str, object]:
    """Worker entrypoint for one (lattice, L, Cv, T, seed_idx) simulation."""
    lattice_name, L, Cv, T, seed_idx, rng_seed, total_sweeps, burn_in, n_msd_samples = job
    lattice = build_lattice(lattice_name, L)
    sim = simulate_trajectory(
        lattice=lattice,
        Cv=Cv,
        T=T,
        seed=rng_seed,
        total_sweeps=total_sweeps,
        burn_in=burn_in,
        n_msd_samples=n_msd_samples,
    )

    diff = compute_tracer_diffusivity(
        msd_times=sim.msd_times,
        msd_values=sim.msd_values,
        Cv=sim.actual_Cv,
        lattice_name=lattice.name,
    )
    c_index, mean_zv = compute_local_clustering_index(sim.occ, lattice)
    clust = compute_cluster_observables(sim.occ, lattice)

    t_over_tc = float(T) / lattice.Tc
    cv_over_pc = sim.actual_Cv / lattice.pc
    lam = cv_over_pc * (lattice.Tc / float(T))

    return {
        "lattice": lattice.name,
        "L": lattice.L,
        "N": lattice.N,
        "Z": lattice.Z,
        "Cv": round(float(Cv), 6),
        "actual_Cv": sim.actual_Cv,
        "T": round(float(T), 6),
        "Tc": lattice.Tc,
        "pc": lattice.pc,
        "D0": lattice.D0,
        "T_over_Tc": t_over_tc,
        "Cv_over_pc": cv_over_pc,
        "Lambda": lam,
        "seed_idx": int(seed_idx),
        "rng_seed": int(rng_seed),
        "sweeps": int(total_sweeps),
        "burn_in": int(sim.burn_in),
        "D": diff.D,
        "D_over_Cv": diff.D_over_Cv,
        "D_over_D0": diff.D_over_D0,
        "msd_r2": diff.msd_r2,
        "msd_loglog_slope": diff.msd_loglog_slope,
        "is_diffusive": int(diff.is_diffusive),
        "C_index": c_index,
        "C_index_window_mean": sim.mean_c_index_window,
        "C_index_window_sd": sim.sd_c_index_window,
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
        "acceptance_rate": sim.acceptance_rate,
        "final_energy_per_site": sim.final_energy_per_site,
    }


def _warmup_numba_kernels() -> None:
    """Compile Numba kernels once in the parent process to populate the disk cache."""
    lat = build_lattice("SC", 4)
    sim = simulate_trajectory(lat, Cv=0.20, T=1.5, seed=42, total_sweeps=4, burn_in=1, n_msd_samples=2)
    compute_local_clustering_index(sim.occ, lat)
    compute_cluster_observables(sim.occ, lat)


def execute_jobs(
    jobs: Sequence[Tuple[str, int, float, float, int, int, int, int, int]],
    workers: int,
    desc: str = "Kawasaki MC",
) -> Dict[Tuple[str, int, float, float, int], Dict[str, object]]:
    """Execute deduplicated simulation jobs using ProcessPoolExecutor or serial mode."""
    results: Dict[Tuple[str, int, float, float, int], Dict[str, object]] = {}
    if not jobs:
        return results

    _warmup_numba_kernels()

    if workers <= 1 or len(jobs) == 1:
        for job in tqdm(jobs, desc=desc, unit="sim"):
            row = _run_single_job(job)
            key = (row["lattice"], int(row["L"]), round(float(row["Cv"]), 6), round(float(row["T"]), 6), int(row["seed_idx"]))
            results[key] = row
        return results

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_run_single_job, job): job for job in jobs}
        for future in tqdm(as_completed(future_map), total=len(jobs), desc=desc, unit="sim"):
            row = future.result()
            key = (row["lattice"], int(row["L"]), round(float(row["Cv"]), 6), round(float(row["T"]), 6), int(row["seed_idx"]))
            results[key] = row

    return results


def _collect_dataframe(
    conditions: Iterable[Condition],
    n_seeds_by_L: Dict[int, int],
    results_map: Dict[Tuple[str, int, float, float, int], Dict[str, object]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    lattice_order = {"SC": 0, "BCC": 1, "FCC": 2}
    for cond in conditions:
        n_s = n_seeds_by_L[cond.L]
        for s_idx in range(n_s):
            k = (cond.lattice, cond.L, round(cond.Cv, 6), round(cond.T, 6), s_idx)
            row = dict(results_map[k])
            if cond.regime_tag:
                row["regime_tag"] = cond.regime_tag
            rows.append(row)

    df = pd.DataFrame(rows)
    df["_lat_ord"] = df["lattice"].map(lattice_order)
    df = df.sort_values(["_lat_ord", "L", "T", "Cv", "seed_idx"]).drop(columns=["_lat_ord"]).reset_index(drop=True)
    return df


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Kawasaki Monte Carlo sweeps for SC, BCC, and FCC lattices."
    )
    parser.add_argument(
        "--suite",
        choices=("all", "seed465", "fcc_sweep", "thermo_controls", "finite_size"),
        default="all",
        help="Which simulation suite to run (default: all).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=15,
        help="Number of independent seeds per L=20 condition (default: 15).",
    )
    parser.add_argument(
        "--l40-seeds",
        type=int,
        default=None,
        help="Number of independent seeds for L=40 FCC finite-size check (default: min(seeds, 5)).",
    )
    parser.add_argument(
        "--sweeps",
        type=int,
        default=6000,
        help="Total Monte Carlo sweeps per trajectory (default: 6000).",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=None,
        help="Burn-in sweeps before MSD production (default: sweeps // 3).",
    )
    parser.add_argument(
        "--msd-samples",
        type=int,
        default=25,
        help="Number of MSD sampling points during production window (default: 25).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of parallel worker processes (default: cpu_count - 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to write CSV outputs (default: data).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260808,
        help="Base integer seed for deterministic per-job PRNG derivation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    burn_in = args.burn_in if args.burn_in is not None else max(1, args.sweeps // 3)
    l40_seeds = args.l40_seeds if args.l40_seeds is not None else min(args.seeds, 5)
    n_seeds_by_L = {20: args.seeds, 40: l40_seeds}

    suites_to_run = (
        ("seed465", "fcc_sweep", "thermo_controls", "finite_size")
        if args.suite == "all"
        else (args.suite,)
    )

    suite_conditions: Dict[str, List[Condition]] = {}
    if "seed465" in suites_to_run:
        suite_conditions["seed465"] = get_seed_level_465_conditions()
    if "fcc_sweep" in suites_to_run:
        suite_conditions["fcc_sweep"] = get_fcc_sweep_conditions()
    if "thermo_controls" in suites_to_run:
        suite_conditions["thermo_controls"] = get_thermo_control_conditions()
    if "finite_size" in suites_to_run:
        suite_conditions["finite_size"] = get_finite_size_conditions()

    # Deduplicate jobs across suites so overlapping conditions share bit-identical runs
    unique_jobs: Dict[Tuple[str, int, float, float, int], Tuple[str, int, float, float, int, int, int, int, int]] = {}
    for cond_list in suite_conditions.values():
        for cond in cond_list:
            n_s = n_seeds_by_L[cond.L]
            for s_idx in range(n_s):
                key = (cond.lattice, cond.L, round(cond.Cv, 6), round(cond.T, 6), s_idx)
                if key not in unique_jobs:
                    rng_seed = derive_deterministic_seed(
                        args.base_seed, cond.lattice, cond.L, cond.Cv, cond.T, s_idx
                    )
                    unique_jobs[key] = (
                        cond.lattice,
                        cond.L,
                        cond.Cv,
                        cond.T,
                        s_idx,
                        rng_seed,
                        args.sweeps,
                        burn_in,
                        args.msd_samples,
                    )

    print(
        f"Running {len(unique_jobs)} unique Kawasaki MC trajectories "
        f"(sweeps={args.sweeps}, burn_in={burn_in}, seeds_L20={args.seeds}, "
        f"seeds_L40={l40_seeds}, workers={args.workers})..."
    )
    results_map = execute_jobs(list(unique_jobs.values()), workers=args.workers)

    output_mapping = {
        "seed465": out_dir / "seed_level_465.csv",
        "fcc_sweep": out_dir / "fcc_concentration_sweep.csv",
        "thermo_controls": out_dir / "thermo_controls.csv",
        "finite_size": out_dir / "finite_size_fcc.csv",
    }

    for suite_name, cond_list in suite_conditions.items():
        df = _collect_dataframe(cond_list, n_seeds_by_L, results_map)
        out_path = output_mapping[suite_name]
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df):4d} rows -> {out_path}")


if __name__ == "__main__":
    main()
