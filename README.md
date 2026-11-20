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
│   ├── lattices.py           # Periodic 3D SC (Z=6), BCC (Z=8), FCC (Z=12) supercells (L=20, L=40)
│   ├── kawasaki_mc.py        # Numba @njit conserved Kawasaki Metropolis engine + unwrapped atom tracking
│   ├── observables.py        # Tracer diffusivity D, local clustering C, f_max, exact topological wrapping P_span
│   ├── run_sweeps.py         # Multiprocessed CLI sweep runner -> data/*.csv (465-seed ensemble + controls + L=40)
│   ├── run_dense_spanning.py # 75 intermediate subcritical SC & BCC trajectories -> data/spanning_dense_subcritical.csv
│   ├── statistics.py         # Scaling regressions, crossing interpolation, ANCOVA F-tests -> data/summary_stats.json
│   ├── make_figures.py       # Publication-ready vector PDF & 300-DPI PNG figures (Figs. 2-11) -> figures/
│   └── make_fig01.py         # Deterministic 3D vacancy configuration snapshots (Fig. 1) -> figures/
├── tests/
│   └── test_physics.py       # 18 physics-invariant tests (conservation, wrapping, dilute Manning recovery)
├── data/                     # Verified 465-seed CSV datasets, dense spanning sweep, & summary_stats.json
├── figures/                  # Publication-ready vector PDF & 300-DPI PNG figures (Figs. 1-11)
└── manuscript/
    ├── main.tex              # APS revtex4-2 submission manuscript
    └── references.bib        # Verified 23-entry BibTeX bibliography
```

## Quick Start

### 1. Install Dependencies & Run Physics Invariant Tests
```bash
pip install -r requirements.txt
python -m pytest -v
```

### 2. Fast End-to-End Smoke Test
```bash
python -m src.run_sweeps --suite all --seeds 2 --sweeps 500
python -m src.statistics --data-dir data --output data/summary_stats.json
python -m src.make_figures --data-dir data --figures-dir figures
```

### 3. Full Production Reproduction (465-Seed Ensemble + Dense Spanning + All 11 Figures)
```bash
python -m src.run_sweeps --suite all --seeds 15 --l40-seeds 5 --sweeps 6000
python -m src.run_dense_spanning
python -m src.statistics --data-dir data --output data/summary_stats.json
python -m src.make_figures --data-dir data --figures-dir figures
python -m src.make_fig01
```

### 4. Compile Manuscript (TeX Live / Overleaf)
```bash
cd manuscript
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
