"""Physics invariant tests for SC, BCC, and FCC Kawasaki lattice-gas simulations.

Verifies:
  1. Exact supercell site counts, coordination numbers, nearest-neighbor distances,
     and periodic neighbor & displacement symmetry.
  2. Exact vacancy conservation, atom-ID bijection, Hamiltonian delta-E bookkeeping,
     unwrapped-displacement modular consistency, and bit-exact seed determinism.
  3. True topological periodic wrapping detection on known percolating vs.
     adversarial non-percolating (bounding-box-spanning but non-wrapping) lattices,
     plus local clustering index C exact checks.
  4. Dilute-limit D / Cv linearity and recovery of analytical Compaan-Haven / Manning
     tracer diffusivity values in the non-interacting high-temperature limit.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.kawasaki_mc import (
    build_exp_table,
    compute_total_energy,
    init_rng_state,
    initialize_lattice_state,
    kawasaki_sweeps_kernel,
    simulate_trajectory,
)
from src.lattices import LATTICE_SPECS, build_lattice
from src.observables import (
    compute_cluster_observables,
    compute_local_clustering_index,
    compute_tracer_diffusivity,
)


# ---------------------------------------------------------------------------
# 1. Lattice Geometry & Periodic Neighbor Symmetry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,L,expected_N,expected_Z,expected_d2",
    [
        ("SC", 20, 8_000, 6, 1.0),
        ("BCC", 20, 16_000, 8, 0.75),
        ("FCC", 20, 32_000, 12, 0.50),
        ("FCC", 40, 256_000, 12, 0.50),
    ],
)
def test_supercell_dimensions_and_coordination(
    name: str, L: int, expected_N: int, expected_Z: int, expected_d2: float
) -> None:
    lat = build_lattice(name, L)
    assert lat.N == expected_N
    assert lat.Z == expected_Z
    assert lat.positions.shape == (expected_N, 3)
    assert lat.neighbors.shape == (expected_N, expected_Z)
    assert lat.displacements.shape == (expected_N, expected_Z, 3)
    assert np.allclose(lat.nn_dist ** 2, expected_d2, atol=1e-12)

    # Check exact nearest-neighbor bond lengths on a sample of sites
    sample_idx = np.linspace(0, lat.N - 1, min(lat.N, 500), dtype=np.int32)
    d2 = np.sum(lat.displacements[sample_idx] ** 2, axis=-1)
    assert np.allclose(d2, expected_d2, atol=1e-12)


@pytest.mark.parametrize("name", ["SC", "BCC", "FCC"])
def test_periodic_neighbor_symmetry_and_displacement_antisymmetry(name: str) -> None:
    lat = build_lattice(name, L=6)

    for i in range(lat.N):
        nbrs_i = lat.neighbors[i]
        # No self-neighbors and all Z neighbors are distinct
        assert i not in nbrs_i
        assert len(set(nbrs_i.tolist())) == lat.Z

        for k in range(lat.Z):
            j = int(nbrs_i[k])
            disp_ij = lat.displacements[i, k]

            # Minimum-image position check: (pos[i] + disp_ij) mod L == pos[j] mod L
            pred_j = np.mod(lat.positions[i] + disp_ij, float(lat.L))
            diff = np.mod(pred_j - lat.positions[j] + 0.5 * lat.L, float(lat.L)) - 0.5 * lat.L
            assert np.allclose(diff, 0.0, atol=1e-12)

            # Exact reciprocal neighbor and antisymmetric displacement vector
            matches = np.where(lat.neighbors[j] == i)[0]
            assert len(matches) == 1
            m = int(matches[0])
            assert np.allclose(lat.displacements[j, m], -disp_ij, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Kawasaki MC Conservation, Energy Bookkeeping & Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["SC", "BCC", "FCC"])
def test_vacancy_conservation_energy_and_unwrapped_consistency(name: str) -> None:
    lat = build_lattice(name, L=6)
    Cv = 0.20
    T = 1.5
    seed = 123456

    occ, site_to_atom, unwrapped_disp = initialize_lattice_state(lat, Cv, seed)
    initial_n_vac = int(np.sum(occ == 0))
    initial_n_atoms = int(np.sum(occ == 1))
    assert initial_n_vac + initial_n_atoms == lat.N

    # Record initial site coordinate of every tagged atom
    init_pos = np.empty((initial_n_atoms, 3), dtype=np.float64)
    for site in range(lat.N):
        aid = int(site_to_atom[site])
        if aid >= 0:
            init_pos[aid] = lat.positions[site]

    e_initial = compute_total_energy(occ, lat.neighbors, J=1.0)
    exp_table = build_exp_table(lat.Z, T, J=1.0)
    rng_state = init_rng_state(np.uint64(seed))

    accepted, delta_e_int = kawasaki_sweeps_kernel(
        occ, site_to_atom, unwrapped_disp, lat.neighbors, lat.displacements, exp_table, rng_state, 30
    )
    assert accepted > 0

    # Exact vacancy & atom count conservation
    assert int(np.sum(occ == 0)) == initial_n_vac
    assert int(np.sum(occ == 1)) == initial_n_atoms

    # Exact atom-ID bijection invariant
    vac_mask = occ == 0
    occ_mask = occ == 1
    assert np.all(site_to_atom[vac_mask] == -1)
    active_ids = site_to_atom[occ_mask]
    assert len(active_ids) == initial_n_atoms
    assert np.array_equal(np.sort(active_ids), np.arange(initial_n_atoms, dtype=np.int32))

    # Exact Hamiltonian energy bookkeeping: E_final - E_initial == sum(dE)
    e_final = compute_total_energy(occ, lat.neighbors, J=1.0)
    assert np.isclose(e_final - e_initial, float(delta_e_int), atol=1e-10)

    # Exact periodic consistency of unwrapped atom displacements
    for site in range(lat.N):
        aid = int(site_to_atom[site])
        if aid >= 0:
            expected_wrapped = np.mod(init_pos[aid] + unwrapped_disp[aid], float(lat.L))
            delta = (
                np.mod(expected_wrapped - lat.positions[site] + 0.5 * lat.L, float(lat.L))
                - 0.5 * lat.L
            )
            assert np.allclose(delta, 0.0, atol=1e-10)


def test_bit_exact_seed_determinism() -> None:
    lat = build_lattice("FCC", L=6)
    res1 = simulate_trajectory(lat, Cv=0.20, T=2.0, seed=98765, total_sweeps=40, burn_in=10)
    res2 = simulate_trajectory(lat, Cv=0.20, T=2.0, seed=98765, total_sweeps=40, burn_in=10)
    res3 = simulate_trajectory(lat, Cv=0.20, T=2.0, seed=98766, total_sweeps=40, burn_in=10)

    assert np.array_equal(res1.occ, res2.occ)
    assert np.array_equal(res1.unwrapped_disp, res2.unwrapped_disp)
    assert np.array_equal(res1.msd_values, res2.msd_values)
    assert not np.array_equal(res1.occ, res3.occ)


# ---------------------------------------------------------------------------
# 3. Exact Topological Wrapping & Local Clustering Observables
# ---------------------------------------------------------------------------

def test_topological_wrapping_true_percolating_pillar_sc() -> None:
    """A 1D line of vacancies along x=0..L-1 wraps strictly along x."""
    L = 8
    lat = build_lattice("SC", L)
    occ = np.ones(lat.N, dtype=np.int8)
    for ix in range(L):
        occ[lat.site_index(ix, 2, 3, 0)] = 0

    metrics = compute_cluster_observables(occ, lat)
    assert metrics.n_vac == L
    assert metrics.n_clusters == 1
    assert metrics.largest_cluster_size == L
    assert np.isclose(metrics.f_max, 1.0)
    assert metrics.is_spanning == 1
    assert metrics.wraps_x is True
    assert metrics.wraps_y is False
    assert metrics.wraps_z is False
    assert metrics.n_wrap_dims == 1


def test_topological_wrapping_rejects_nonwrapping_bounding_box_serpentine() -> None:
    """Adversarial test: a U-shaped cluster touches BOTH x=0 and x=L-1 without closing
    across the periodic seam. A naive bounding-box heuristic would falsely flag it as
    spanning, whereas spanning-tree unwrapped displacement tracking MUST return False.
    """
    L = 8
    lat = build_lattice("SC", L)
    occ = np.ones(lat.N, dtype=np.int8)

    # Bottom arm: x = 0..L-2 at y = 1, z = 0
    for ix in range(L - 1):
        occ[lat.site_index(ix, 1, 0, 0)] = 0
    # Vertical turn at x = L-2: y = 1..3, z = 0
    for iy in range(1, 4):
        occ[lat.site_index(L - 2, iy, 0, 0)] = 0
    # Step to x = L-1 at y = 3 (NOTE: (0, 3, 0) is OCCUPIED, so NO periodic bond closes!)
    occ[lat.site_index(L - 1, 3, 0, 0)] = 0

    # Confirm the cluster's x-coordinates include every single plane 0..L-1
    vac_x = np.unique(lat.positions[occ == 0, 0])
    assert len(vac_x) == L  # Naive bounding-box span == L - 1 (full box!)

    metrics = compute_cluster_observables(occ, lat)
    assert metrics.n_clusters == 1
    assert metrics.is_spanning == 0
    assert metrics.wraps_x is False
    assert metrics.wraps_y is False
    assert metrics.wraps_z is False


def test_topological_wrapping_diagonal_helical_fcc() -> None:
    """A helical nearest-neighbor vacancy path on FCC that winds around both x and y."""
    L = 6
    lat = build_lattice("FCC", L)
    occ = np.ones(lat.N, dtype=np.int8)

    # Walk (0,0,0) -> (0.5,0.5,0) -> (1,1,0) -> ... -> (L-0.5, L-0.5, 0) -> wraps to (0,0,0)
    for k in range(L):
        occ[lat.site_index(k, k, 0, 0)] = 0  # basis 0: (k, k, 0)
        occ[lat.site_index(k, k, 0, 1)] = 0  # basis 1: (k+0.5, k+0.5, 0)

    metrics = compute_cluster_observables(occ, lat)
    assert metrics.n_vac == 2 * L
    assert metrics.n_clusters == 1
    assert metrics.is_spanning == 1
    assert metrics.wraps_x is True
    assert metrics.wraps_y is True
    assert metrics.wraps_z is False


def test_local_clustering_index_random_and_isolated() -> None:
    lat = build_lattice("SC", L=10)

    # Isolated non-adjacent vacancies (even coordinates only) -> C == 0
    occ_iso = np.ones(lat.N, dtype=np.int8)
    for ix in range(0, 10, 2):
        for iy in range(0, 10, 2):
            occ_iso[lat.site_index(ix, iy, 0, 0)] = 0
    c_iso, mean_zv_iso = compute_local_clustering_index(occ_iso, lat)
    assert c_iso == 0.0
    assert mean_zv_iso == 0.0

    # Random vacancy placement at Cv = 0.20 -> C approx 1.0
    occ_rand, _, _ = initialize_lattice_state(lat, Cv=0.20, seed=42)
    c_rand, _ = compute_local_clustering_index(occ_rand, lat)
    assert 0.85 < c_rand < 1.15


# ---------------------------------------------------------------------------
# 4. Dilute-Limit D / Cv Recovery (Compaan-Haven / Manning Benchmark)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["SC", "BCC", "FCC"])
def test_dilute_limit_d_over_cv_recovery(name: str) -> None:
    """In the non-interacting dilute limit (high T, small Cv), tracer diffusivity
    must satisfy D proportional to Cv with analytical prefactor:
        D / Cv = (1 / 3) * f_0 * d^2
    where f_0 is the exact Compaan-Haven / Manning correlation factor and
    2 * Cv is the bidirectional jump rate per atom per MCS.
    """
    spec = LATTICE_SPECS[name]
    theoretical_d_over_cv = (1.0 / 3.0) * float(spec["f0_dilute"]) * float(spec["nn_dist_sq"])

    # Use L=12 for fast high-statistics dilute verification
    lat = build_lattice(name, L=12)
    T_high = 100.0  # exp(-dE / T) ~ 1 (non-interacting random-walk limit)

    d_over_cv_measured = []
    for Cv in (0.015, 0.030):
        vals = []
        for s in range(3):
            sim = simulate_trajectory(
                lat, Cv=Cv, T=T_high, seed=7000 + s * 100 + int(Cv * 1000),
                total_sweeps=450, burn_in=50, n_msd_samples=20,
            )
            diff = compute_tracer_diffusivity(sim.msd_times, sim.msd_values, sim.actual_Cv, name)
            assert 0.90 <= diff.msd_loglog_slope <= 1.10
            vals.append(diff.D_over_Cv)
        d_over_cv_measured.append(float(np.mean(vals)))

    # 1. Recover analytical Manning/Compaan-Haven dilute value within 12%
    for measured in d_over_cv_measured:
        rel_err = abs(measured - theoretical_d_over_cv) / theoretical_d_over_cv
        assert rel_err < 0.12, (
            f"{name}: measured D/Cv={measured:.4f} vs theory={theoretical_d_over_cv:.4f}"
        )

    # 2. Linearity across dilute concentrations (constant D/Cv within 10%)
    ratio = d_over_cv_measured[1] / d_over_cv_measured[0]
    assert 0.90 < ratio < 1.10, f"{name}: D/Cv changed across dilute Cv: {d_over_cv_measured}"
