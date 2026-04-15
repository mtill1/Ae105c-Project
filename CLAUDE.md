# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Asteroid mission trajectory optimization system for AE 105c (Caltech/Pomona). Selects optimal 3-asteroid visitation sequences launched from Earth, minimizing total delta-V (fuel cost). Also includes a Mars transfer variant (Earth -> Asteroid -> Mars).

## Prerequisites

- **MATLAB** with Optimization Toolbox (fmincon) and ODE solvers (ode113)
- **MICE** (MATLAB Interface to CSPICE) — JPL's SPICE toolkit for ephemeris computations
- Generic SPICE kernels: `naif0012.tls`, `de430.bsp`, `jup310.bsp`, `gm_de431.tpc`, `pck00010.tpc`
- MICE and kernel paths are user/OS-specific, configured in `LOAD_KERNELS.m` (detects username to set paths)

## Running the Code

All scripts must be run from the **repository root directory** with MATLAB's working directory set there. The `NOTABLE_ASTEROID_BSPs/` and `SPICE_BSPs/` folders must be accessible from the working directory.

### Three main workflows:

1. **Exhaustive optimization**: `asteroid_selector.m` -> `find_best_path.m`
   - Runs fmincon over all (i,j,k) asteroid triplets, saves results to `optimal_asteroid_paths/asteroid_data_{m1}_{m2}_{m3}.mat`
   - `find_best_path.m` loads results and identifies top 10 lowest delta-V paths

2. **Greedy heuristic**: `Code/Asteroid_Trajectory_Selection/GREEDY/greedy_selector.m`
   - Faster but suboptimal; outputs to `greedy_asteroid_paths/` and `asteroid_optimized_data_table.csv`

3. **Mars transfer**: `mars_transfer_selector.m`
   - Earth -> Asteroid -> Mars trajectory variant

## Architecture

### Core pipeline (executed in sequence):

```
LOAD_KERNELS(bsp_folder)          — Initialize SPICE, load ephemeris kernels, return asteroid list
    |
GENERATE_OPTIMIZED_DATA(...)      — Exhaustive loop over all asteroid triplets
    |
    +-- OPTIMIZE_TIMES(...)       — fmincon optimization for one triplet, grid search over initial guesses
        |
        +-- SCORE_PATHS(...)      — Objective function: calls COMPUTE_PATH_DELTAV, returns scalar delta-V
            |
            +-- COMPUTE_PATH_DELTAV(...)  — 3 Lambert solves, computes 5 maneuver delta-Vs
                |
                +-- lambert(...)          — Robust Lambert solver (Izzo + Lancaster/Gooding, ~800 lines)
```

### Key design decisions:

- **Time representation**: All times are SPICE ET (seconds past J2000). Convert with `cspice_str2et` / `cspice_et2utc`.
- **M parameter**: Number of complete orbital revolutions for each Lambert arc (0 = direct transfer). Different M values produce different `.mat` output files.
- **Optimization grid**: `OPTIMIZE_TIMES` uses a Chebyshev-spaced grid of initial guesses (controlled by `N_RES` vector) and keeps the best fmincon result.
- **Delta-V total**: Sum of 5 maneuver magnitudes (arrival/departure at each asteroid, excluding Earth departure from the total in `COMPUTE_PATH_DELTAV` but tracked separately).
- **Lambert failure handling**: Returns `delta_v_total = 1e3` as a penalty when the solver doesn't converge.

### Parallel structure for Mars variant:
`GENERATE_MARS_TRANSFER_OPTIMIZED` / `OPTIMIZE_MARS_TIMES` / `COMPUTE_MARS_PATH_DELTAV` / `SCORE_PATHS_MARS` / `UNPACK_MARS_INPUT` mirror the standard pipeline.

### Greedy variant:
`GREEDY/` subfolder mirrors the pipeline with `_GREEDY` suffixed functions. Uses `GENERATE_GREEDY_OPTIMIZED_DATA` which builds paths incrementally rather than evaluating all triplets.

### Visualization:
- `FLIGHTPATH_ANIMATION.m` / `GREEDY_FLIGHTPATH_ANIMATION.m` — renders `.mp4` trajectory videos
- `GRAPH_ASTEROID_FLIGHTPATH.m` / `GRAPH_ASTEROIDS.m` — static plots
- `TWO_BODY_SIM.m` — numerical propagator for orbit arcs (ode113, Sun gravity only)

### Tradeoff analysis (standalone):
`Code/Tradeoff Table/asteroid_tradeoff.m` scores ~1000 asteroids from JPL SBDB data (`sbdb_query_results.csv`) using weighted criteria: delta-V (30%), eccentricity (15%), inclination (15%), science potential (10%), mass (10%), radius (10%), SMA (5%), rotation (5%). Uses Chebyshev-spaced bins and taxonomy-based density estimates.

## Conventions

- Functions are UPPERCASE (`LOAD_KERNELS`, `OPTIMIZE_TIMES`); scripts and variables are lowercase
- BSP kernel files are named after their asteroid (e.g., `Ceres.bsp`)
- Output `.mat` files are named by M parameters: `asteroid_data_{m1}_{m2}_{m3}.mat`
- Reference frame: `ECLIPJ2000`, observer: `10` (Sun), aberration: `NONE`
