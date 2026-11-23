"""Generate 3-panel 3D morphological comparison for FCC at Cv = 0.20, L = 20.

Directly contrasts the three physical regimes at fixed FCC vacancy concentration Cv = 0.20
(N = 32,000 sites, N_v = 6,400 vacancies):
  (a) Interacting below Tc (J = 1, T = 2.00, T/Tc = 0.82): compact, locally enriched droplets
  (b) Interacting above Tc (J = 1, T = 3.06, T/Tc = 1.25): space-filling correlated spanning network
  (c) Non-interacting random (J = 0): uncorrelated random site-percolation baseline

Outputs `figures/fig06b_fcc_3way_morphology.(pdf|png)`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.kawasaki_mc import initialize_lattice_state, simulate_trajectory
from src.lattices import LATTICE_SPECS, build_lattice
from src.make_fig01 import _label_largest_cluster_mask
from src.make_figures import _save_fig, apply_publication_style
from src.observables import compute_cluster_observables, compute_local_clustering_index
from src.run_sweeps import derive_deterministic_seed


BASE_SEED = 20260808


def generate_fig06b(out_dir: Path, sweeps: int = 6000) -> None:
    apply_publication_style()
    lat = build_lattice("FCC", L=20)
    tc = float(LATTICE_SPECS["FCC"]["Tc"])
    Cv = 0.20

    # (a) J=1 below Tc, (b) J=1 above Tc, (c) J=0 random baseline
    # Use deterministic canonical seed_idx=0 (matching run_sweeps & run_reviewer_experiments)
    seed_lo = derive_deterministic_seed(BASE_SEED, "FCC", 20, Cv, 2.00, seed_idx=0)
    sim_lo = simulate_trajectory(lat, Cv=Cv, T=2.00, seed=seed_lo, total_sweeps=sweeps)

    seed_hi = derive_deterministic_seed(BASE_SEED, "FCC", 20, Cv, 3.06, seed_idx=0)
    sim_hi = simulate_trajectory(lat, Cv=Cv, T=3.06, seed=seed_hi, total_sweeps=sweeps)

    seed_j0 = derive_deterministic_seed(BASE_SEED + 202, "FCC", 20, Cv, 0.0, seed_idx=0)
    occ_j0, _, _ = initialize_lattice_state(lat, Cv, seed_j0)

    panels = [
        (
            "(a) Interacting below $T_c$ ($J=1$, $T/T_c=0.82$)",
            sim_lo.occ,
        ),
        (
            "(b) Interacting above $T_c$ ($J=1$, $T/T_c=1.25$)",
            sim_hi.occ,
        ),
        (
            "(c) Non-interacting random ($J=0$)",
            occ_j0,
        ),
    ]

    fig = plt.figure(figsize=(11.6, 4.2))
    pt_size = 6.2

    for idx, (header, occ) in enumerate(panels, start=1):
        clust = compute_cluster_observables(occ, lat)
        c_idx, _ = compute_local_clustering_index(occ, lat)
        largest_mask = _label_largest_cluster_mask(occ, lat.neighbors)

        vac_mask = occ == 0
        other_vac_mask = vac_mask & (~largest_mask)
        atom_mask = occ == 1

        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.set_box_aspect((1.0, 1.0, 1.0))

        # Subtle cubic shell atoms for geometric reference
        pos_atoms = lat.positions[atom_mask]
        on_shell = (
            (pos_atoms[:, 0] <= 0.5)
            | (pos_atoms[:, 0] >= lat.L - 1.0)
            | (pos_atoms[:, 1] <= 0.5)
            | (pos_atoms[:, 1] >= lat.L - 1.0)
            | (pos_atoms[:, 2] <= 0.5)
            | (pos_atoms[:, 2] >= lat.L - 1.0)
        )
        shell_atoms = pos_atoms[on_shell]
        stride = max(1, len(shell_atoms) // 1600)
        sa = shell_atoms[::stride]
        ax.scatter(
            sa[:, 0], sa[:, 1], sa[:, 2],
            s=pt_size * 0.45,
            c="#B8BEC6",
            alpha=0.13,
            edgecolors="none",
            depthshade=False,
            rasterized=True,
        )

        # Secondary clusters
        pos_other = lat.positions[other_vac_mask]
        if len(pos_other) > 0:
            ax.scatter(
                pos_other[:, 0], pos_other[:, 1], pos_other[:, 2],
                s=pt_size * 0.72,
                c="#56B4E9",
                alpha=0.50,
                edgecolors="none",
                depthshade=True,
                rasterized=True,
            )

        # Largest cluster
        pos_giant = lat.positions[largest_mask]
        if len(pos_giant) > 0:
            ax.scatter(
                pos_giant[:, 0], pos_giant[:, 1], pos_giant[:, 2],
                s=pt_size * 0.90,
                c="#D55E00",
                alpha=0.80,
                edgecolors="none",
                depthshade=True,
                rasterized=True,
            )

        ax.view_init(elev=22, azim=38)
        ax.set_xlim(0, lat.L)
        ax.set_ylim(0, lat.L)
        ax.set_zlim(0, lat.L)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("#CCCCCC")

        span_str = "spanning" if clust.is_spanning else "non-spanning"
        ax.set_title(
            f"{header}\n"
            rf"$f_{{\max}} = {clust.f_max_pct:.1f}\%$, $\mathcal{{C}} = {c_idx:.2f}$ ({span_str})",
            fontsize=8.6,
            pad=4,
        )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D55E00", markersize=7.5, label="Largest vacancy cluster"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#56B4E9", markersize=7.0, label="Secondary vacancy clusters"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#B8BEC6", markersize=6.5, alpha=0.6, label="Host lattice shell (atoms)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=9.1,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.12, wspace=0.06)
    _save_fig(fig, out_dir, "fig06b_fcc_3way_morphology", pad_inches=0.12)
    print(f"Generated fig06b_fcc_3way_morphology.(pdf|png) in {out_dir}")


if __name__ == "__main__":
    generate_fig06b(Path("figures"))
