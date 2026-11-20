"""3D periodic cubic lattice builder (SC, BCC, FCC) with exact neighbor and displacement tables.

All coordinates are expressed in conventional cubic unit-cell lengths (a = 1), giving
nearest-neighbor jump distances:
  - SC  (Z = 6):  d = 1.0          (d^2 = 1.00)
  - BCC (Z = 8):  d = sqrt(3) / 2  (d^2 = 0.75)
  - FCC (Z = 12): d = sqrt(2) / 2  (d^2 = 0.50)

For conventional supercell linear dimension L:
  - SC:  N = L^3      (L=20 -> 8,000 sites)
  - BCC: N = 2 * L^3  (L=20 -> 16,000 sites)
  - FCC: N = 4 * L^3  (L=20 -> 32,000 sites; L=40 -> 256,000 sites)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np


LATTICE_SPECS: Dict[str, Dict[str, object]] = {
    "SC": {
        "Z": 6,
        "Tc": 1.13,
        "pc": 0.3116,
        "D0": 0.1890,
        "f0_dilute": 0.6531,
        "nn_dist_sq": 1.0,
        "basis": (
            (0.0, 0.0, 0.0),
        ),
    },
    "BCC": {
        "Z": 8,
        "Tc": 1.59,
        "pc": 0.2455,
        "D0": 0.1488,
        "f0_dilute": 0.7272,
        "nn_dist_sq": 0.75,
        "basis": (
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
        ),
    },
    "FCC": {
        "Z": 12,
        "Tc": 2.45,
        "pc": 0.1996,
        "D0": 0.1006,
        "f0_dilute": 0.7815,
        "nn_dist_sq": 0.5,
        "basis": (
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ),
    },
}


@dataclass(frozen=True)
class Lattice:
    """Immutable 3D periodic lattice geometry with precomputed neighbor tables."""

    name: str
    L: int
    N: int
    Z: int
    n_basis: int
    Tc: float
    pc: float
    D0: float
    nn_dist: float
    positions: np.ndarray       # shape (N, 3), float64
    neighbors: np.ndarray       # shape (N, Z), int32
    displacements: np.ndarray   # shape (N, Z, 3), float64

    def site_index(self, ix: int, iy: int, iz: int, basis_idx: int = 0) -> int:
        """Return flat site index for unit cell (ix, iy, iz) and basis index."""
        return (((ix % self.L) * self.L + (iy % self.L)) * self.L + (iz % self.L)) * self.n_basis + basis_idx


def _build_local_stencil(
    basis: Tuple[Tuple[float, float, float], ...],
    Z: int,
    nn_dist_sq: float,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact per-basis neighbor stencil over (-1, 0, +1)^3 cell shifts."""
    n_basis = len(basis)
    basis_arr = np.asarray(basis, dtype=np.float64)

    cell_shifts = np.zeros((n_basis, Z, 3), dtype=np.int32)
    target_basis = np.zeros((n_basis, Z), dtype=np.int32)
    disp_vectors = np.zeros((n_basis, Z, 3), dtype=np.float64)

    for b in range(n_basis):
        entries = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for b2 in range(n_basis):
                        vec = np.array([dx, dy, dz], dtype=np.float64) + basis_arr[b2] - basis_arr[b]
                        dist_sq = float(np.dot(vec, vec))
                        if abs(dist_sq - nn_dist_sq) < tol:
                            entries.append((dx, dy, dz, b2, vec))
        if len(entries) != Z:
            raise RuntimeError(
                f"Expected {Z} nearest neighbors for basis site {b}, found {len(entries)}"
            )
        # Deterministic canonical ordering by displacement vector
        entries.sort(key=lambda e: (round(e[4][0], 6), round(e[4][1], 6), round(e[4][2], 6)))
        for k, (dx, dy, dz, b2, vec) in enumerate(entries):
            cell_shifts[b, k] = (dx, dy, dz)
            target_basis[b, k] = b2
            disp_vectors[b, k] = vec

    return cell_shifts, target_basis, disp_vectors


@lru_cache(maxsize=16)
def build_lattice(name: str, L: int = 20) -> Lattice:
    """Build a 3D periodic SC, BCC, or FCC supercell of linear size L.

    Parameters
    ----------
    name : str
        One of "SC", "BCC", or "FCC" (case-insensitive).
    L : int
        Linear number of conventional cubic unit cells along each axis (L >= 3).

    Returns
    -------
    Lattice
        Precomputed lattice object with exact `(N, Z)` neighbor indices and
        `(N, Z, 3)` periodic nearest-neighbor displacement vectors.
    """
    key = name.upper()
    if key not in LATTICE_SPECS:
        raise ValueError(f"Unsupported lattice '{name}'. Expected one of {sorted(LATTICE_SPECS)}.")
    if L < 3:
        raise ValueError(f"Supercell linear dimension L must be >= 3 to avoid self-wrapping, got {L}.")

    spec = LATTICE_SPECS[key]
    Z = int(spec["Z"])
    basis = spec["basis"]
    n_basis = len(basis)
    nn_dist_sq = float(spec["nn_dist_sq"])
    N = (L ** 3) * n_basis

    cell_shifts, target_basis, stencil_disp = _build_local_stencil(basis, Z, nn_dist_sq)

    # Enumerate unit cell coordinates (ix, iy, iz) in C-order: ((ix * L + iy) * L + iz)
    grid = np.indices((L, L, L), dtype=np.int32).reshape(3, -1).T  # shape (L^3, 3)
    n_cells = L ** 3
    basis_arr = np.asarray(basis, dtype=np.float64)

    # Positions: shape (L^3, n_basis, 3) -> (N, 3)
    positions = (grid[:, None, :].astype(np.float64) + basis_arr[None, :, :]).reshape(N, 3)

    # Neighbor indices: shape (L^3, n_basis, Z) -> (N, Z)
    ix = grid[:, 0][:, None, None]  # (n_cells, 1, 1)
    iy = grid[:, 1][:, None, None]
    iz = grid[:, 2][:, None, None]

    jx = (ix + cell_shifts[None, :, :, 0]) % L
    jy = (iy + cell_shifts[None, :, :, 1]) % L
    jz = (iz + cell_shifts[None, :, :, 2]) % L
    jb = np.broadcast_to(target_basis[None, :, :], (n_cells, n_basis, Z))

    neighbors = (((jx * L + jy) * L + jz) * n_basis + jb).reshape(N, Z).astype(np.int32)

    # Displacement vectors: identical for all unit cells for a given basis index
    displacements = np.broadcast_to(
        stencil_disp[None, :, :, :], (n_cells, n_basis, Z, 3)
    ).reshape(N, Z, 3).copy()

    # Make arrays read-only so cached instances cannot be mutated accidentally
    positions.setflags(write=False)
    neighbors.setflags(write=False)
    displacements.setflags(write=False)

    return Lattice(
        name=key,
        L=L,
        N=N,
        Z=Z,
        n_basis=n_basis,
        Tc=float(spec["Tc"]),
        pc=float(spec["pc"]),
        D0=float(spec["D0"]),
        nn_dist=float(np.sqrt(nn_dist_sq)),
        positions=positions,
        neighbors=neighbors,
        displacements=displacements,
    )
