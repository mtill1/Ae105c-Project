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

## Conventions

- All active Python code lives in `Python_Consolidated/` (6 files, no subpackages)
- Cross-file imports: `from core import ...`, `from optimization import ...`
- Asteroid BSPs named after the asteroid: `THEMIS.bsp`, `HYGIEA.bsp`
- Output saved to `optimal_asteroid_paths/` as `.pkl` (pickle)
- Renders saved to `Renders/Asteroid_Plots/`, numbered (01_, 02_, ...)
- Reference frame: `ECLIPJ2000`, observer: `10` (Sun), aberration: `NONE`

## SPICE Kernel Paths

Generic kernels: `/Users/rebnoob/Documents/ae105/generic_kernels/`

```python
asteroid_list = load_kernels("NOTABLE_ASTEROID_BSPs",
                             "/Users/rebnoob/Documents/ae105/generic_kernels")
```

## Asteroid Pools

| Pool | Count | Location |
|------|-------|----------|
| Notable (primary) | 50 | `NOTABLE_ASTEROID_BSPs/` |
| Extended | 49 | `SPICE_BSPs/` |
| Science-ranked | 407 | `asteroid_tradeoff.csv` |
