"""Generate Figure 1: Representative 3D vacancy configurations at Cv = 0.20 (T/Tc = 0.82).

Simulates deterministic seed_idx=0 trajectories (L=20, 6,000 sweeps) for SC (T=0.93),
BCC (T=1.30), and FCC (T=2.00) using the exact same RNG seed derivation as run_sweeps.py,
and renders a publication-grade 3-panel 3D isometric visualization showing the largest
connected vacancy cluster, secondary vacancy clusters, and the host lattice boundary.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.kawasaki_mc import simulate_trajectory
from src.lattices import LATTICE_SPECS, build_lattice
from src.make_figures import apply_publication_style, _save_fig
from src.observables import compute_cluster_observables, compute_local_clustering_index
from src.run_sweeps import derive_deterministic_seed


def _label_largest_cluster_mask(occ: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """Return boolean mask of vacant sites belonging to the single largest vacancy cluster."""
    n_sites = len(occ)
    visited = np.zeros(n_sites, dtype=np.bool_)
    best_comp: list[int] = []

    for root in range(n_sites):
        if occ[root] != 0 or visited[root]:
            continue
        comp = [root]
        visited[root] = True
        head = 0
        while head < len(comp):
            u = comp[head]
            head += 1
            for v in neighbors[u]:
                if occ[v] == 0 and not visited[v]:
                    visited[v] = True
                    comp.append(int(v))
        if len(comp) > len(best_comp):
            best_comp = comp

    mask = np.zeros(n_sites, dtype=np.bool_)
    if best_comp:
        mask[best_comp] = True
    return mask


def generate_fig01(out_dir: Path, base_seed: int = 20260808, sweeps: int = 6000) -> None:
    apply_publication_style()
    fig = plt.figure(figsize=(11.4, 4.1))

    configs = [
        ("SC", 0.93, 0.20, 14.0),
        ("BCC", 1.30, 0.20, 9.5),
        ("FCC", 2.00, 0.20, 6.2),
    ]

    for idx, (lat_name, T_val, Cv, pt_size) in enumerate(configs, start=1):
        lat = build_lattice(lat_name, L=20)
        rng_seed = derive_deterministic_seed(base_seed, lat_name, 20, Cv, T_val, seed_idx=0)
        sim = simulate_trajectory(lat, Cv=Cv, T=T_val, seed=rng_seed, total_sweeps=sweeps)
        clust = compute_cluster_observables(sim.occ, lat)
        c_idx, _ = compute_local_clustering_index(sim.occ, lat)
        largest_mask = _label_largest_cluster_mask(sim.occ, lat.neighbors)

        vac_mask = sim.occ == 0
        other_vac_mask = vac_mask & (~largest_mask)
        atom_mask = sim.occ == 1

        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.set_box_aspect((1.0, 1.0, 1.0))

        # Subtle surface-shell host atoms for crisp cubic visual framing without occluding interior
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
        # Subsample shell atoms deterministically for FCC/BCC readability
        stride = max(1, len(shell_atoms) // 1600)
        sa = shell_atoms[::stride]
        ax.scatter(
            sa[:, 0], sa[:, 1], sa[:, 2],
            s=pt_size * 0.45,
            c="#B8BEC6",
            alpha=0.14,
            edgecolors="none",
            depthshade=False,
            rasterized=True,
        )

        # Secondary (non-giant) vacancy clusters
        pos_other = lat.positions[other_vac_mask]
        if len(pos_other) > 0:
            ax.scatter(
                pos_other[:, 0], pos_other[:, 1], pos_other[:, 2],
                s=pt_size * 0.72,
                c="#56B4E9",
                alpha=0.52,
                edgecolors="none",
                depthshade=True,
                rasterized=True,
            )

        # Largest connected vacancy cluster
        pos_giant = lat.positions[largest_mask]
        if len(pos_giant) > 0:
            ax.scatter(
                pos_giant[:, 0], pos_giant[:, 1], pos_giant[:, 2],
                s=pt_size * 0.92,
                c="#D55E00",
                alpha=0.82,
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
        t_red = T_val / float(LATTICE_SPECS[lat_name]["Tc"])
        ax.set_title(
            f"({chr(96 + idx)}) {lat_name} ($Z={lat.Z}$, $N={lat.N:,}$)\n"
            rf"$T/T_c = {t_red:.2f}$, $f_{{\max}} = {clust.f_max_pct:.1f}\%$, $\mathcal{{C}} = {c_idx:.2f}$ ({span_str})",
            fontsize=8.9,
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
        fontsize=9.2,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.12, wspace=0.06)
    _save_fig(fig, out_dir, "fig01_vacancy_configurations", pad_inches=0.12)
    print(f"Generated fig01_vacancy_configurations.(pdf|png) in {out_dir}")


if __name__ == "__main__":
    generate_fig01(Path("figures"))
