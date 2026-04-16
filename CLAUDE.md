# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Asteroid mission trajectory optimization for Ae105c. Designs a spacecraft mission from Earth visiting **3 asteroids sequentially**, minimizing total delta-v (fuel cost). Two mission architectures:

1. **Direct**: Earth -> Asteroid 1 -> Asteroid 2 -> Asteroid 3
2. **Mars flyby**: Earth -> Mars (gravity assist) -> A1 -> A2 -> A3

The codebase was originally MATLAB, now **fully ported to Python** in `Python_Consolidated/`. The MATLAB code in `Code/` is legacy and no longer maintained.

## Repository Structure

```
Ae105c-Project/
├── Python_Consolidated/          # Active Python codebase (6 files, ~3000 lines)
│   ├── core.py                   # Constants, pykep Lambert/propagation wrappers, SPICE loading
│   ├── optimization.py           # Delta-v computation, scoring, DE optimizer, beam search
│   ├── greedy.py                 # Greedy/flyby algorithm (legacy approach)
│   ├── visualization.py          # Flightpath animation, orbit plotting
│   ├── scripts.py                # Runner entry points for all workflows
│   ├── tradeoff.py               # Asteroid science/physical scoring (standalone)
│   └── requirements.txt
├── NOTABLE_ASTEROID_BSPs/        # 50 asteroid ephemeris files (.bsp)
├── SPICE_BSPs/                   # Larger asteroid pool (49 BSPs)
├── Renders/Asteroid_Plots/       # Generated images and GIFs (numbered 01-09)
├── asteroid_tradeoff.csv         # Ranked asteroid table (407 asteroids)
├── Code/                         # Legacy MATLAB code (not maintained)
└── CLAUDE.md
```

## Prerequisites

- **Python 3.10+** (3.13 works for everything except pykep)
- **SPICE kernels** at `/Users/rebnoob/Documents/ae105/generic_kernels/` with:
  - `lsk/naif0012.tls`, `spk/planets/de430.bsp`, `spk/satellites/jup310.bsp`
  - `pck/gm_de431.tpc`, `pck/pck00010.tpc`
- **pykep** (ESA): Used for Lambert solving, orbit propagation, flyby computation. Note: no wheel for Python 3.13 yet — code falls back gracefully where needed.
- **spiceypy**: Always required for SPICE kernel management and asteroid ephemerides.

Install dependencies:
```bash
cd Python_Consolidated
pip install -r requirements.txt
```

## Running the Code

All scripts run from the **repository root** (not from `Python_Consolidated/`). The `NOTABLE_ASTEROID_BSPs/` folder must be in the working directory.

### Recommended workflows (in `scripts.py`):

```python
from scripts import *

# 1. Two-level optimization (RECOMMENDED — best quality results)
#    Coarse pass on all N^3 triplets, then fine-tune top 50
results = run_two_level_optimize()

# 2. With science weighting (70% delta-v, 30% science value)
results = run_two_level_optimize(science_csv='asteroid_tradeoff.csv', alpha=0.7)

# 3. Beam search (structured multi-stage, good for large pools)
results = run_beam_search(beam_width=15)

# 4. Brute-force N^3 (original method, slow but thorough)
results = run_asteroid_selector()

# 5. Mars flyby variant
results = run_mars_transfer_selector()

# 6. Visualization
run_graphing_notable_asteroids()
```

## Architecture

### core.py — Foundation layer

Wraps pykep and spiceypy behind a clean interface. All units: **km, km/s, km^3/s^2**.

| Function | Purpose |
|----------|---------|
| `solve_lambert(r1, r2, tof_days, m, mu)` | Wraps `pk.lambert_problem`. Returns (V1, V2, exitflag). |
| `solve_lambert_best(r1, r2, tof_days, mu)` | Tries m=0,1,2 revolutions + both directions, returns best. |
| `two_body_sim(t_final, x0, mu)` | Wraps `pk.propagate_lagrangian`. Returns (X, T) arrays. |
| `compute_flyby_dv(v_in, v_out, v_planet, mu, r)` | Wraps `pk.fb_dv`. Returns powered flyby delta-v. |
| `load_kernels(bsp_folder, generic_path)` | Loads SPICE kernels, returns asteroid list. |
| `get_state(body_id, et)` | Returns (r_km, v_km_s) via spiceypy. |

**Constants**: `DAY=86400s`, `MONTH=30.4375*DAY`, `YEAR=365.25*DAY`, `MAX_MISSION_DURATION=14*YEAR`, `MU_SUN`

### optimization.py — Optimization engine

| Function | Purpose |
|----------|---------|
| `compute_path_deltav(...)` | 3 Lambert solves for a triplet. Returns dict with all delta-v components. **Includes launch dv in total.** |
| `score_paths(input_vec, ...)` | Objective function. Penalizes missions >14yr with 1e3. |
| `optimize_times(...)` | `scipy.optimize.differential_evolution` on 6D time space. |
| `optimize_times_quick(...)` | Fast coarse version (30 iters) for two-level first pass. |
| `two_level_optimize(...)` | Coarse filter all triplets, fine-optimize top-N. Accepts science scores. |
| `beam_search(...)` | Multi-stage K-best search. Keeps top-K at each leg. |
| `optimize_best_architecture(...)` | Tries direct + Moon flyby + Mars flyby, returns best |
| `compute_path_with_flyby(...)` | Delta-v for Earth -> flyby -> A1 -> A2 -> A3 |
| `generate_optimized_data(...)` | Brute-force N^3 loop (legacy). |

### greedy.py — Greedy algorithm (legacy, suboptimal)

Sequential 3-leg optimization with Earth/Mars/direct flyby options. Per commit `9afb21a`: "gives too suboptimal results." Prefer `beam_search` or `two_level_optimize`.

### visualization.py — Plotting and animation

- `flightpath_animation(...)` — 3D MP4 video of a trajectory (requires FFmpeg)
- `graph_asteroids(...)` — Animated asteroid orbit video
- `graph_asteroid_flightpath(...)` — Static 3D trajectory plot

### tradeoff.py — Asteroid ranking (standalone, no SPICE needed)

Reads `sbdb_query_results.csv` from JPL SBDB and outputs `asteroid_tradeoff.csv`. Two versions:
- `run_tradeoff_v1()` — Chebyshev-bin scoring
- `run_tradeoff_v3()` — Enhanced two-component science score (characterization 40% + interest 60%), with manual flags for active asteroids, ice, radar-ambiguous M-types, visited targets

## Key Design Decisions

- **Optimizer**: `scipy.optimize.differential_evolution` (global, multimodal) with `polish=True` (L-BFGS-B refinement). Replaces the old 180-point Chebyshev grid.
- **Time system**: SPICE ET (seconds past J2000). Parsed with `spiceypy.str2et()`.
- **Units**: km / km/s / km^3/s^2 throughout. Pykep (SI) conversions inside core.py.
- **Lambert failure**: Returns `delta_v_total = 1e3` as penalty.
- **Mission duration**: Hard cap at 14 years. Score functions return 1e3 for violations.
- **Delta-v total**: Sum of **all 6 maneuver norms** including Earth departure.
- **Science integration**: `two_level_optimize` and `beam_search` accept `science_scores` + `alpha`. Example: `alpha=0.7` = 70% delta-v + 30% science.
- **Gravity assists**: Moon and Mars flybys are automatically evaluated for every triplet. The optimizer picks the best architecture (direct, Moon flyby, or Mars flyby) per triplet.

## Current Best Results (saved — do NOT re-run unless asked)

Results are saved as pickle files in `optimal_asteroid_paths/`. Load with:
```python
import pickle
with open('optimal_asteroid_paths/results_69ast_ga.pkl', 'rb') as f:
    results = pickle.load(f)  # list of (i, j, k, result_dict)
```

### Best paths (69 asteroids, C+S+X/M diverse, gravity assists)

| Rank | Path | dV (km/s) | Flyby |
|------|------|:---------:|:-----:|
| 1 | **Hertha [X/M] -> Polyxo [C] -> Alkeste [S]** | **9.40** | Mars |
| 2 | Virginia [C] -> Psyche [X/M] -> Parthenope [S] | 9.51 | Moon |
| 3 | Massalia [S] -> Misa [C] -> Psyche [X/M] | 9.77 | Moon |
| 4 | Massalia [S] -> Psyche [X/M] -> Nuwa [C] | 9.96 | Moon |
| 5 | Psyche [X/M] -> Polyxo [C] -> Alkeste [S] | 10.01 | Mars |

### All saved result files

| File | Asteroids | Composition | Gravity Assist | Best dV |
|------|:---------:|:-----------:|:--------------:|:-------:|
| `results_69ast_ga.pkl` | 69 | C+S+X/M | Yes (Moon+Mars) | 9.40 |
| `results_diverse_CSM.pkl` | 50 | C+S+X/M | No | 13.80 |
| `results_diverse_science_weighted.pkl` | 50 | C+S+X/M | No (alpha=0.5) | 14.61 |
| `results_50ast_full.pkl` | 50 | Any | No | 13.07 |

## GCP Compute

- **Project**: `project-8b1249f5-4cb6-4dad-8a9`
- **Machine**: `e2-standard-12` (12 vCPU — max allowed by quota CPUS_ALL_REGIONS=12)
- **Zone**: `us-west1-b`
- **Setup**: Miniconda + Python 3.11 + pykep from conda-forge
- **Scripts**: `Python_Consolidated/gcp/` (run_optimization.py, run_diverse.py, etc.)
- **Auth**: `gcloud auth login` required before each session

## Conventions

- All active Python code lives in `Python_Consolidated/` (6 files + gcp/ subfolder)
- GCP scripts in `Python_Consolidated/gcp/` import from `..` (no code duplication)
- Cross-file imports: `from core import ...`, `from optimization import ...`
- Asteroid BSPs in `NOTABLE_ASTEROID_BSPs/` (69 asteroids, merged from original 50 + extended pool)
- Output saved to `optimal_asteroid_paths/` as `.pkl` (pickle)
- Renders saved to `Renders/Asteroid_Plots/`, numbered (01_, 02_, ...)
- Reference frame: `ECLIPJ2000`, observer: `10` (Sun), aberration: `NONE`

## SPICE Kernel Paths

Generic kernels: `/Users/rebnoob/Documents/ae105/generic_kernels/`

```python
asteroid_list = load_kernels("NOTABLE_ASTEROID_BSPs",
                             "/Users/rebnoob/Documents/ae105/generic_kernels")
```

## Asteroid Pool

69 asteroids in `NOTABLE_ASTEROID_BSPs/` (merged from original 50 notable + 23 from SPICE_BSPs, minus 4 with bad BSP coverage).

Composition: 39 C-complex, 12 S-complex, 5 X/M-complex, 13 Unknown.

Science-ranked table: `asteroid_tradeoff.csv` (407 asteroids).
