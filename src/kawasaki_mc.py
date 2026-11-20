"""Numba-accelerated conserved Kawasaki Monte Carlo engine for the attractive lattice gas.

Hamiltonian:
    E = -J * sum_{<i, j>} n_i * n_j,   n_i in {0, 1},  J = 1,  k_B = 1

where n_i = 1 denotes an occupied site (atom) and n_i = 0 denotes a vacancy.
One Monte Carlo sweep (MCS) equals N attempted nearest-neighbor atom-vacancy
exchanges using Metropolis acceptance rules. Continuous unwrapped displacements
(N_atoms, 3) are tracked for every tagged atom across periodic boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np
from numba import njit

from src.lattices import Lattice


@njit(cache=True)
def _splitmix64_step(state: np.uint64) -> Tuple[np.uint64, np.uint64]:
    state = np.uint64(state + np.uint64(0x9E3779B97F4A7C15))
    z = state
    z = np.uint64((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9))
    z = np.uint64((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB))
    z = np.uint64(z ^ (z >> np.uint64(31)))
    return state, z


@njit(cache=True)
def init_rng_state(seed: np.uint64) -> np.ndarray:
    """Initialize a 4-word xoshiro256** PRNG state from a 64-bit integer seed."""
    state = np.empty(4, dtype=np.uint64)
    sm = np.uint64(seed)
    for idx in range(4):
        sm, val = _splitmix64_step(sm)
        state[idx] = val
    # Guard against the all-zero absorbing state
    if state[0] == 0 and state[1] == 0 and state[2] == 0 and state[3] == 0:
        state[0] = np.uint64(0x9E3779B97F4A7C15)
    return state


@njit(cache=True)
def _rotl64(x: np.uint64, k: np.uint64) -> np.uint64:
    return np.uint64((x << k) | (x >> np.uint64(np.uint64(64) - k)))


@njit(cache=True)
def _xoshiro_next(s: np.ndarray) -> np.uint64:
    result = np.uint64(_rotl64(np.uint64(s[1] * np.uint64(5)), np.uint64(7)) * np.uint64(9))
    t = np.uint64(s[1] << np.uint64(17))
    s[2] ^= s[0]
    s[3] ^= s[1]
    s[1] ^= s[2]
    s[0] ^= s[3]
    s[2] ^= t
    s[3] = _rotl64(s[3], np.uint64(45))
    return result


@njit(cache=True)
def _rand_uniform(s: np.ndarray) -> float:
    return (_xoshiro_next(s) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def compute_total_energy(occ: np.ndarray, neighbors: np.ndarray, J: float = 1.0) -> float:
    """Compute exact lattice-gas energy E = -J * sum_{<i,j>} n_i * n_j."""
    n_sites = occ.shape[0]
    z_coord = neighbors.shape[1]
    bond_sum = np.int64(0)
    for i in range(n_sites):
        if occ[i] == 1:
            for k in range(z_coord):
                bond_sum += occ[neighbors[i, k]]
    # Each undirected bond <i, j> is counted twice
    return -0.5 * J * float(bond_sum)


@njit(cache=True)
def compute_msd(unwrapped_disp: np.ndarray, disp_ref: np.ndarray) -> float:
    """Compute mean-squared displacement <||r(t) - r(t_0)||^2> over all tagged atoms."""
    n_atoms = unwrapped_disp.shape[0]
    if n_atoms == 0:
        return 0.0
    total = 0.0
    for a in range(n_atoms):
        dx = unwrapped_disp[a, 0] - disp_ref[a, 0]
        dy = unwrapped_disp[a, 1] - disp_ref[a, 1]
        dz = unwrapped_disp[a, 2] - disp_ref[a, 2]
        total += dx * dx + dy * dy + dz * dz
    return total / float(n_atoms)


@njit(cache=True)
def kawasaki_sweeps_kernel(
    occ: np.ndarray,
    site_to_atom: np.ndarray,
    unwrapped_disp: np.ndarray,
    neighbors: np.ndarray,
    displacements: np.ndarray,
    exp_table: np.ndarray,
    rng_state: np.ndarray,
    n_sweeps: int,
) -> Tuple[np.int64, np.int64]:
    """Execute `n_sweeps` Kawasaki Metropolis sweeps (each = N exchange attempts).

    Parameters
    ----------
    occ : int8[N]
        Site occupancy (1 = atom, 0 = vacancy). Mutated in-place.
    site_to_atom : int32[N]
        Maps occupied site -> atom ID in [0, N_atoms), vacant site -> -1. Mutated in-place.
    unwrapped_disp : float64[N_atoms, 3]
        Cumulative unwrapped displacement of each tagged atom. Mutated in-place.
    neighbors : int32[N, Z]
        Precomputed periodic neighbor index table.
    displacements : float64[N, Z, 3]
        Precomputed minimum-image displacement vectors from site i to neighbors[i, k].
    exp_table : float64[Z + 1]
        Precomputed Boltzmann factors exp(-J * k / T) for k = 0, 1, ..., Z.
    rng_state : uint64[4]
        xoshiro256** PRNG state. Mutated in-place.
    n_sweeps : int
        Number of Monte Carlo sweeps to run.

    Returns
    -------
    accepted : int64
        Total number of accepted atom-vacancy exchanges.
    delta_e_int : int64
        Cumulative integer energy change sum(Delta E / J) across accepted moves.
    """
    n_sites = occ.shape[0]
    z_coord = neighbors.shape[1]
    n_sites_u = np.uint64(n_sites)
    z_coord_u = np.uint64(z_coord)

    accepted = np.int64(0)
    delta_e_int = np.int64(0)

    total_attempts = np.int64(n_sweeps) * np.int64(n_sites)
    for _ in range(total_attempts):
        r1 = _xoshiro_next(rng_state)
        i = np.int32(r1 % n_sites_u)
        r2 = _xoshiro_next(rng_state)
        k = np.int32(r2 % z_coord_u)
        j = neighbors[i, k]

        oi = occ[i]
        oj = occ[j]
        if oi == oj:
            continue

        if oi == 1:
            atom_site = i
            vac_site = j
            dx = displacements[i, k, 0]
            dy = displacements[i, k, 1]
            dz = displacements[i, k, 2]
        else:
            atom_site = j
            vac_site = i
            dx = -displacements[i, k, 0]
            dy = -displacements[i, k, 1]
            dz = -displacements[i, k, 2]

        # Count occupied nearest neighbors of atom_site and vac_site
        m_atom = np.int32(0)
        for s in range(z_coord):
            m_atom += occ[neighbors[atom_site, s]]

        m_vac = np.int32(0)
        for s in range(z_coord):
            m_vac += occ[neighbors[vac_site, s]]

        # Before: atom has m_atom bonds (vac_site is vacant, contributes 0).
        # After:  atom at vac_site has (m_vac - 1) bonds (atom_site became vacant).
        # dE = E_after - E_before = J * (m_atom - (m_vac - 1)) = J * (m_atom - m_vac + 1).
        dE = m_atom - m_vac + 1

        if dE <= 0 or _rand_uniform(rng_state) < exp_table[dE]:
            occ[atom_site] = np.int8(0)
            occ[vac_site] = np.int8(1)

            aid = site_to_atom[atom_site]
            site_to_atom[vac_site] = aid
            site_to_atom[atom_site] = np.int32(-1)

            unwrapped_disp[aid, 0] += dx
            unwrapped_disp[aid, 1] += dy
            unwrapped_disp[aid, 2] += dz

            accepted += 1
            delta_e_int += dE

    return accepted, delta_e_int


def build_exp_table(Z: int, T: float, J: float = 1.0) -> np.ndarray:
    """Precompute Boltzmann factors exp(-J * k / T) for integer dE in [0, Z]."""
    if T <= 0.0:
        raise ValueError(f"Temperature T must be positive, got {T}.")
    table = np.ones(Z + 1, dtype=np.float64)
    for k in range(1, Z + 1):
        table[k] = math.exp(-J * float(k) / float(T))
    return table


def initialize_lattice_state(
    lattice: Lattice,
    Cv: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize a random conserved vacancy configuration and atom tracking tables."""
    if not (0.0 < Cv < 1.0):
        raise ValueError(f"Vacancy concentration Cv must be in (0, 1), got {Cv}.")

    n_vac = int(round(Cv * lattice.N))
    n_vac = max(1, min(lattice.N - 1, n_vac))
    n_atoms = lattice.N - n_vac

    rng = np.random.default_rng(int(seed))
    vac_sites = rng.choice(lattice.N, size=n_vac, replace=False)

    occ = np.ones(lattice.N, dtype=np.int8)
    occ[vac_sites] = np.int8(0)

    site_to_atom = np.full(lattice.N, -1, dtype=np.int32)
    occupied_indices = np.flatnonzero(occ == 1)
    site_to_atom[occupied_indices] = np.arange(n_atoms, dtype=np.int32)

    unwrapped_disp = np.zeros((n_atoms, 3), dtype=np.float64)
    return occ, site_to_atom, unwrapped_disp


@dataclass
class SimulationResult:
    """Container for a single Kawasaki Monte Carlo trajectory."""

    lattice_name: str
    L: int
    N: int
    Z: int
    target_Cv: float
    actual_Cv: float
    T: float
    seed: int
    total_sweeps: int
    burn_in: int
    n_vac: int
    n_atoms: int
    occ: np.ndarray
    site_to_atom: np.ndarray
    unwrapped_disp: np.ndarray
    msd_times: np.ndarray
    msd_values: np.ndarray
    accepted_total: int
    attempted_total: int
    acceptance_rate: float
    final_energy: float
    final_energy_per_site: float


def simulate_trajectory(
    lattice: Lattice,
    Cv: float,
    T: float,
    seed: int,
    total_sweeps: int = 6000,
    burn_in: int | None = None,
    n_msd_samples: int = 25,
    J: float = 1.0,
) -> SimulationResult:
    """Run a complete deterministic Kawasaki MC trajectory with burn-in and MSD sampling.

    Parameters
    ----------
    lattice : Lattice
        Precomputed periodic lattice geometry.
    Cv : float
        Target vacancy fraction in (0, 1).
    T : float
        Reduced temperature (k_B = 1, J = 1).
    seed : int
        64-bit integer seed controlling both initial placement and MC updates.
    total_sweeps : int
        Total number of Monte Carlo sweeps (burn-in + production).
    burn_in : int, optional
        Number of equilibration sweeps prior to MSD production. Defaults to
        int(round(total_sweeps / 3)) (e.g., 2,000 sweeps for a 6,000-sweep run).
    n_msd_samples : int
        Number of MSD sampling checkpoints across the production window.
    J : float
        Nearest-neighbor attractive coupling constant (default 1.0).
    """
    if total_sweeps < 2:
        raise ValueError(f"total_sweeps must be >= 2, got {total_sweeps}.")
    if burn_in is None:
        burn_in = int(round(total_sweeps / 3.0))
    burn_in = max(0, min(total_sweeps - 1, int(burn_in)))
    prod_sweeps = total_sweeps - burn_in

    occ, site_to_atom, unwrapped_disp = initialize_lattice_state(lattice, Cv, seed)
    # Mix seed with a fixed odd constant so MC PRNG stream is independent of init shuffle
    mc_seed = np.uint64((int(seed) ^ 0xA5A5A5A55A5A5A5A) & 0xFFFFFFFFFFFFFFFF)
    rng_state = init_rng_state(mc_seed)
    exp_table = build_exp_table(lattice.Z, T, J=J)

    accepted_total = 0
    if burn_in > 0:
        acc_b, _ = kawasaki_sweeps_kernel(
            occ,
            site_to_atom,
            unwrapped_disp,
            lattice.neighbors,
            lattice.displacements,
            exp_table,
            rng_state,
            burn_in,
        )
        accepted_total += int(acc_b)

    disp_ref = unwrapped_disp.copy()

    # Build evenly spaced integer sweep checkpoints across production window
    n_samples = max(2, min(int(n_msd_samples), prod_sweeps))
    checkpoints = np.unique(
        np.round(np.linspace(1, prod_sweeps, n_samples)).astype(np.int64)
    )
    if checkpoints[-1] != prod_sweeps:
        checkpoints = np.append(checkpoints, prod_sweeps)

    msd_times = np.empty(len(checkpoints), dtype=np.float64)
    msd_values = np.empty(len(checkpoints), dtype=np.float64)

    prev_t = 0
    for idx, t_prod in enumerate(checkpoints):
        step_sweeps = int(t_prod - prev_t)
        acc_p, _ = kawasaki_sweeps_kernel(
            occ,
            site_to_atom,
            unwrapped_disp,
            lattice.neighbors,
            lattice.displacements,
            exp_table,
            rng_state,
            step_sweeps,
        )
        accepted_total += int(acc_p)
        msd_times[idx] = float(t_prod)
        msd_values[idx] = float(compute_msd(unwrapped_disp, disp_ref))
        prev_t = int(t_prod)

    n_vac = int(np.sum(occ == 0))
    n_atoms = lattice.N - n_vac
    attempted_total = int(total_sweeps) * int(lattice.N)
    final_energy = float(compute_total_energy(occ, lattice.neighbors, J=J))

    return SimulationResult(
        lattice_name=lattice.name,
        L=lattice.L,
        N=lattice.N,
        Z=lattice.Z,
        target_Cv=float(Cv),
        actual_Cv=float(n_vac) / float(lattice.N),
        T=float(T),
        seed=int(seed),
        total_sweeps=int(total_sweeps),
        burn_in=int(burn_in),
        n_vac=n_vac,
        n_atoms=n_atoms,
        occ=occ,
        site_to_atom=site_to_atom,
        unwrapped_disp=unwrapped_disp,
        msd_times=msd_times,
        msd_values=msd_values,
        accepted_total=accepted_total,
        attempted_total=attempted_total,
        acceptance_rate=float(accepted_total) / float(attempted_total),
        final_energy=final_energy,
        final_energy_per_site=final_energy / float(lattice.N),
    )
