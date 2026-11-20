"""Transport and structural observables for conserved lattice-gas simulations.

Implements:
1. Tracer diffusivity D from linear fit <R^2(t)> = 6 D t over the production
   window, diffusive log-log MSD slope verification ([0.95, 1.05]), and
   normalization by lattice-specific reference diffusivities D0.
2. Local clustering index C = <z_v> / (Z * C_v).
3. Largest-cluster fraction f_max and exact 3D periodic topological wrapping
   indicator P_span via BFS spanning-tree unwrapped displacement tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from numba import njit

from src.lattices import LATTICE_SPECS, Lattice


D0_REFERENCE: Dict[str, float] = {
    name: float(spec["D0"]) for name, spec in LATTICE_SPECS.items()
}


@dataclass(frozen=True)
class DiffusivityMetrics:
    """Tracer diffusivity and diffusive-regime validation diagnostics."""

    D: float
    D_over_Cv: float
    D_over_D0: float
    msd_intercept: float
    msd_r2: float
    msd_loglog_slope: float
    is_diffusive: bool


@dataclass(frozen=True)
class ClusterMetrics:
    """Vacancy cluster connectivity and topological spanning diagnostics."""

    n_vac: int
    n_clusters: int
    largest_cluster_size: int
    f_max: float
    f_max_pct: float
    is_spanning: int
    wraps_x: bool
    wraps_y: bool
    wraps_z: bool
    n_wrap_dims: int


def compute_tracer_diffusivity(
    msd_times: np.ndarray,
    msd_values: np.ndarray,
    Cv: float,
    lattice_name: str,
    diffusive_bounds: Tuple[float, float] = (0.95, 1.05),
) -> DiffusivityMetrics:
    """Fit <R^2(t)> = 6 D t + b and log(<R^2(t)>) vs log(t) over the production window."""
    t = np.asarray(msd_times, dtype=np.float64)
    y = np.asarray(msd_values, dtype=np.float64)

    mask = (t > 0.0) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if len(t) < 2:
        raise ValueError("Need at least 2 positive time points to fit tracer diffusivity.")

    # Linear OLS fit: y = slope * t + intercept, where slope = 6 * D
    t_mean = float(np.mean(t))
    y_mean = float(np.mean(y))
    dt = t - t_mean
    dy = y - y_mean
    denom = float(np.dot(dt, dt))
    if denom <= 0.0:
        raise ValueError("Degenerate time grid for MSD fit.")

    slope = float(np.dot(dt, dy)) / denom
    intercept = y_mean - slope * t_mean
    D = max(0.0, slope / 6.0)

    y_pred = slope * t + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    msd_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    # Log-log slope over positive MSD points to verify diffusive scaling
    pos_mask = y > 0.0
    if np.count_nonzero(pos_mask) >= 2:
        lt = np.log(t[pos_mask])
        ly = np.log(y[pos_mask])
        lt_c = lt - np.mean(lt)
        ly_c = ly - np.mean(ly)
        lt_denom = float(np.dot(lt_c, lt_c))
        loglog_slope = float(np.dot(lt_c, ly_c)) / lt_denom if lt_denom > 0.0 else float("nan")
    else:
        loglog_slope = float("nan")

    low, high = diffusive_bounds
    is_diffusive = bool(np.isfinite(loglog_slope) and (low <= loglog_slope <= high))

    D0 = D0_REFERENCE[lattice_name.upper()]
    return DiffusivityMetrics(
        D=D,
        D_over_Cv=D / float(Cv) if Cv > 0.0 else float("nan"),
        D_over_D0=D / D0,
        msd_intercept=intercept,
        msd_r2=msd_r2,
        msd_loglog_slope=loglog_slope,
        is_diffusive=is_diffusive,
    )


@njit(cache=True)
def _local_clustering_kernel(occ: np.ndarray, neighbors: np.ndarray) -> Tuple[float, float, int]:
    n_sites = occ.shape[0]
    z_coord = neighbors.shape[1]
    n_vac = 0
    vac_nbr_sum = 0

    for i in range(n_sites):
        if occ[i] == 0:
            n_vac += 1
            for k in range(z_coord):
                if occ[neighbors[i, k]] == 0:
                    vac_nbr_sum += 1

    if n_vac == 0:
        return 0.0, 0.0, 0

    mean_zv = float(vac_nbr_sum) / float(n_vac)
    cv = float(n_vac) / float(n_sites)
    c_index = mean_zv / (float(z_coord) * cv)
    return c_index, mean_zv, n_vac


def compute_local_clustering_index(occ: np.ndarray, lattice: Lattice) -> Tuple[float, float]:
    """Compute dimensionless local clustering index C = <z_v> / (Z * C_v) and <z_v>."""
    c_index, mean_zv, _ = _local_clustering_kernel(occ, lattice.neighbors)
    return float(c_index), float(mean_zv)


@njit(cache=True)
def _vacancy_clusters_bfs_kernel(
    occ: np.ndarray,
    neighbors: np.ndarray,
    displacements: np.ndarray,
    L: int,
) -> Tuple[int, int, int, int, int, int]:
    """3D periodic BFS over vacant sites with exact spanning-tree wrapping detection.

    For each connected component of vacant sites (occ == 0), BFS assigns each
    visited vertex an unwrapped spanning-tree coordinate relative to the component
    root by accumulating exact minimum-image displacement vectors. Whenever BFS
    encounters an edge (u, v) to an already-visited vertex v in the same component,
    the loop displacement delta = (tree_pos[u] + d_{u->v}) - tree_pos[v] is an
    integer multiple of L along each periodic axis. Any nonzero component of delta
    (|delta_alpha| > 0.5 * L) certifies a homotopically non-trivial wrapping cycle
    across periodic boundary axis alpha.
    """
    n_sites = occ.shape[0]
    z_coord = neighbors.shape[1]
    half_L = 0.5 * float(L)

    visited = np.zeros(n_sites, dtype=np.int8)
    queue = np.empty(n_sites, dtype=np.int32)
    tree_pos = np.zeros((n_sites, 3), dtype=np.float64)

    n_vac = 0
    n_clusters = 0
    max_cluster_size = 0
    wraps_x_any = 0
    wraps_y_any = 0
    wraps_z_any = 0

    for root in range(n_sites):
        if occ[root] != 0 or visited[root] != 0:
            continue

        n_clusters += 1
        visited[root] = 1
        tree_pos[root, 0] = 0.0
        tree_pos[root, 1] = 0.0
        tree_pos[root, 2] = 0.0

        head = 0
        tail = 0
        queue[tail] = root
        tail += 1
        cluster_size = 0

        while head < tail:
            u = queue[head]
            head += 1
            cluster_size += 1

            ux = tree_pos[u, 0]
            uy = tree_pos[u, 1]
            uz = tree_pos[u, 2]

            for k in range(z_coord):
                v = neighbors[u, k]
                if occ[v] != 0:
                    continue

                ex = ux + displacements[u, k, 0]
                ey = uy + displacements[u, k, 1]
                ez = uz + displacements[u, k, 2]

                if visited[v] == 0:
                    visited[v] = 1
                    tree_pos[v, 0] = ex
                    tree_pos[v, 1] = ey
                    tree_pos[v, 2] = ez
                    queue[tail] = v
                    tail += 1
                else:
                    # Cycle edge: check for nonzero periodic winding vector
                    if abs(ex - tree_pos[v, 0]) > half_L:
                        wraps_x_any = 1
                    if abs(ey - tree_pos[v, 1]) > half_L:
                        wraps_y_any = 1
                    if abs(ez - tree_pos[v, 2]) > half_L:
                        wraps_z_any = 1

        n_vac += cluster_size
        if cluster_size > max_cluster_size:
            max_cluster_size = cluster_size

    return n_vac, n_clusters, max_cluster_size, wraps_x_any, wraps_y_any, wraps_z_any


def compute_cluster_observables(occ: np.ndarray, lattice: Lattice) -> ClusterMetrics:
    """Compute f_max and exact periodic topological wrapping indicator P_span."""
    n_vac, n_clusters, max_size, wx, wy, wz = _vacancy_clusters_bfs_kernel(
        occ, lattice.neighbors, lattice.displacements, lattice.L
    )
    f_max = float(max_size) / float(n_vac) if n_vac > 0 else 0.0
    n_wrap_dims = int(wx) + int(wy) + int(wz)
    is_spanning = 1 if n_wrap_dims > 0 else 0

    return ClusterMetrics(
        n_vac=int(n_vac),
        n_clusters=int(n_clusters),
        largest_cluster_size=int(max_size),
        f_max=f_max,
        f_max_pct=100.0 * f_max,
        is_spanning=is_spanning,
        wraps_x=bool(wx),
        wraps_y=bool(wy),
        wraps_z=bool(wz),
        n_wrap_dims=n_wrap_dims,
    )
