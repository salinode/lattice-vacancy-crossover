"""Run dense intermediate subcritical SC & BCC concentrations to refine P_span = 0.5 crossings.

Adds SC (T=0.93) at Cv in {0.18, 0.22} and BCC (T=1.30) at Cv in {0.18, 0.22, 0.24}
(15 seeds each, L=20, 6,000 MCS, deterministic SHA-256 seeds matching run_sweeps.py),
and merges with existing subcritical SC/BCC/FCC and supercritical FCC rows from
`data/seed_level_465.csv` into `data/spanning_dense_subcritical.csv`.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.run_sweeps import Condition, _collect_dataframe, derive_deterministic_seed, execute_jobs
from src.statistics import analyze_spanning_crossings


def main() -> None:
    new_conds = [
        Condition("SC", 20, 0.18, 0.93, "subcritical_0.82Tc"),
        Condition("SC", 20, 0.22, 0.93, "subcritical_0.82Tc"),
        Condition("BCC", 20, 0.18, 1.30, "subcritical_0.82Tc"),
        Condition("BCC", 20, 0.22, 1.30, "subcritical_0.82Tc"),
        Condition("BCC", 20, 0.24, 1.30, "subcritical_0.82Tc"),
    ]
    base_seed = 20260808
    jobs = []
    for c in new_conds:
        for s_idx in range(15):
            rng_seed = derive_deterministic_seed(base_seed, c.lattice, c.L, c.Cv, c.T, s_idx)
            jobs.append((c.lattice, c.L, c.Cv, c.T, s_idx, rng_seed, 6000, 2000, 25))

    res = execute_jobs(jobs, workers=7, desc="Dense SC/BCC spanning")
    df_new = _collect_dataframe(new_conds, {20: 15}, res)

    df_465 = pd.read_csv(Path("data/seed_level_465.csv"))
    sub_existing = df_465[
        ((df_465["lattice"] == "SC") & (df_465["T"].sub(0.93).abs() < 0.02))
        | ((df_465["lattice"] == "BCC") & (df_465["T"].sub(1.30).abs() < 0.02))
        | (df_465["lattice"] == "FCC")
    ].copy()
    df_dense = pd.concat([sub_existing, df_new], ignore_index=True)
    lat_ord = {"SC": 0, "BCC": 1, "FCC": 2}
    df_dense["_o"] = df_dense["lattice"].map(lat_ord)
    df_dense = df_dense.sort_values(["_o", "T", "Cv", "seed_idx"]).drop(columns=["_o"]).reset_index(drop=True)
    out_path = Path("data/spanning_dense_subcritical.csv")
    df_dense.to_csv(out_path, index=False)
    print(f"Wrote {len(df_dense)} rows -> {out_path}")

    sc = analyze_spanning_crossings(df_dense)
    for k, v in sc.items():
        print(f"\n{k} (T={v['T']}, T/Tc={v['T_over_Tc']}): Cv* = {v['Cv_star']:.4f}, Cv*/pc = {v['Cv_star_over_pc']:.4f}")
        for pt in v["curve"]:
            print(f"  Cv={pt['Cv']:.2f}: P_span={pt['P_span']:.3f}")


if __name__ == "__main__":
    main()
