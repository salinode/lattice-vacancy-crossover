# Lattice Topology, Vacancy Clustering, and Transport

Reproducible, deterministic, Numba-accelerated simulation, statistical analysis, figure-generation, and LaTeX manuscript repository for:

> **"Lattice Topology, Vacancy Clustering, and Transport: A Thermodynamic-to-Geometric Crossover in SC, BCC, and FCC Lattices"**
> Salin Adhikari and Nilaksh Khetan

## Repository Structure

```text
lattice-vacancy-crossover/
├── pyproject.toml
├── requirements.txt
├── src/
│   ├── lattices.py                  # Periodic 3D SC (Z=6), BCC (Z=8), FCC (Z=12) supercells (L=20, 30, 40)
│   ├── kawasaki_mc.py               # Numba @njit conserved Kawasaki Metropolis engine + unwrapped tracking + window C_bar
│   ├── observables.py               # Tracer diffusivity D, local clustering C, f_max, exact topological wrapping P_span
│   ├── run_sweeps.py                # Multiprocessed CLI sweep runner -> 465-seed ensemble + controls + finite-size
│   ├── run_dense_spanning.py        # Intermediate subcritical SC & BCC spanning sweep runner
│   ├── run_reviewer_experiments.py  # Full suite: dilute T=100, J=0 percolation, n=50 crossovers, time-aligned C_bar, L=40 & R_S(t)
│   ├── statistics.py                # Logistic MLE + bootstrap CIs, Wilson CIs, controlled OLS, grouped CV -> summary_stats.json
│   ├── make_figures.py              # Publication-ready vector PDF & 300-DPI PNG figures -> figures/
│   ├── make_fig01.py                # Deterministic 3D SC/BCC/FCC vacancy snapshots (Fig. 1) -> figures/
│   └── make_fig_fcc_morphology.py   # 3-way FCC Cv=0.20 morphology comparison (J=1 below/above Tc vs J=0) -> figures/
├── tests/
│   └── test_physics.py              # 22 physics-invariant, structure-factor, & statistical/FE tests
├── data/                            # Verified CSV datasets & summary_stats.json
├── figures/                         # Publication-ready vector PDF & 300-DPI PNG figures
└── manuscript/
    ├── main.tex                     # APS revtex4-2 submission manuscript
    └── references.bib               # Verified 32-entry BibTeX bibliography
```

## Quick Start

### 1. Install Dependencies & Run Verification Suite
```bash
pip install -r requirements.txt
python -m pytest -v
```

### 2. Full Production Reproduction (All Datasets & Figures)
```bash
python -m src.run_reviewer_experiments --perc-seeds 50
python -m src.statistics --data-dir data --output data/summary_stats.json
python -m src.make_figures --data-dir data --figures-dir figures
python -m src.make_fig01
python -m src.make_fig_fcc_morphology
```

### 3. Compile Manuscript (TeX Live / Overleaf)
```bash
cd manuscript
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
