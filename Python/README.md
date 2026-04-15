# Asteroid Trajectory Selection — Python

Python port of the MATLAB codebase for multi-asteroid mission trajectory optimization. Solves Lambert's problem to find minimum delta-v paths visiting three asteroids from Earth, with optional Mars gravity-assist flybys.

## Prerequisites

- **Python 3.10+**
- **SPICE Kernels**: You need a local copy of the generic NAIF/SPICE kernels with the following structure:
  ```
  <generic_kernels_path>/
    lsk/naif0012.tls
    spk/satellites/jup310.bsp
    spk/planets/de430.bsp
    pck/gm_de431.tpc
    pck/pck00010.tpc
  ```
  These can be downloaded from the [NAIF Generic Kernels](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/) server.

- **Asteroid BSP files**: Place asteroid ephemeris `.bsp` files in a folder (e.g., `NOTABLE_ASTEROID_BSPs/`). Each file should contain the trajectory for one asteroid.

- **FFmpeg** (optional): Required for generating `.mp4` animation videos.

## Installation

```bash
cd Python
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `spiceypy`, `matplotlib`, `tqdm`, `pandas`, `imageio`

## Project Structure

```
Python/
├── requirements.txt
├── README.md
├── asteroid_trajectory_selection/     # Main package
│   ├── constants.py                   # Time constants (MINUTE, HOUR, DAY, WEEK, MONTH, YEAR)
│   ├── lambert.py                     # Lambert solver (Izzo + Lancaster-Blanchard fallback)
│   ├── two_body_sim.py                # Two-body ODE propagator (scipy DOP853)
│   ├── load_kernels.py                # SPICE kernel loader
│   ├── unpack_input.py                # Optimizer input vector -> epoch times
│   │
│   ├── compute_path_deltav.py         # Delta-v for a 3-asteroid path
│   ├── compute_mars_path_deltav.py    # Delta-v with Mars gravity-assist flyby
│   ├── score_paths.py                 # Objective function for the optimizer
│   ├── score_paths_mars.py            # Objective function (Mars variant)
│   │
│   ├── optimize_times.py              # Grid-search + L-BFGS-B time optimization
│   ├── optimize_mars_times.py         # Mars flyby time optimization
│   ├── generate_optimized_data.py     # Full N^3 optimization sweep over all asteroid combos
│   ├── generate_mars_transfer_optimized.py
│   ├── get_id_from_asteroid_name.py   # Lookup asteroid SPICE ID by name
│   │
│   ├── flightpath_animation.py        # 3D animated trajectory video
│   ├── graph_asteroids.py             # Animated video of all asteroid orbits
│   ├── graph_asteroid_flightpath.py   # Static 3D trajectory plot
│   │
│   ├── asteroid_selector.py           # Runner: direct Earth -> 3 asteroid optimization
│   ├── mars_transfer_selector.py      # Runner: Mars flyby -> 3 asteroid optimization
│   ├── find_best_path.py              # Find top-N paths from pre-computed results
│   ├── graph_all_best_paths.py        # Animate all best paths
│   ├── graph_example_flightpath.py    # Animate a specific example path
│   ├── graphing_notable_asteroids.py  # Animate notable asteroid orbits
│   │
│   └── greedy/                        # Greedy path-selection algorithm
│       ├── compute_path_deltav_greedy.py   # Delta-v with planetary flyby options
│       ├── score_paths_greedy.py
│       ├── optimize_greedy_times.py        # 3-stage sequential optimizer
│       ├── generate_greedy_optimized_data.py
│       ├── greedy_flightpath_animation.py
│       ├── greedy_selector.py              # Runner: sweep M-value combinations
│       ├── find_all_paths_greedy.py        # Rank paths across all greedy runs
│       ├── find_best_path_greedy.py        # Find top 6 greedy paths
│       ├── plot_best_path.py               # Animate the best greedy path
│       └── plot_dv_range.py                # Surface plot of delta-v landscape
│
└── tradeoff_table/                    # Asteroid scoring/ranking (no trajectory needed)
    ├── asteroid_tradeoff.py           # v1: Chebyshev-bin scoring
    └── asteroid_tradeoff_01.py        # v3: enhanced two-component science scoring
```

## Usage

### 1. Kernel paths

All scripts that load SPICE kernels call `load_kernels(bsp_folder, generic_kernels_path)`. Update the two path arguments in each runner script to match your local setup before running.

### 2. Direct transfer optimization (Earth -> A1 -> A2 -> A3)

Searches every permutation of 3 asteroids and optimizes departure/arrival/stay times to minimize total delta-v.

```bash
python -m asteroid_trajectory_selection.asteroid_selector
```

This performs an N^3 sweep, calling `scipy.optimize.minimize` (L-BFGS-B) at each grid point for each asteroid triplet. Results are saved as pickle files under `optimal_asteroid_paths/`.

### 3. Mars flyby optimization (Earth -> Mars -> A1 -> A2 -> A3)

Same approach but with an additional Mars gravity-assist leg before the first asteroid.

```bash
python -m asteroid_trajectory_selection.mars_transfer_selector
```

Results saved under `optimal_asteroid_paths/mars_transfers/`.

### 4. Greedy path selection

The greedy algorithm optimizes one leg at a time (Earth->A1, then A1->A2, then A2->A3), considering Earth, Mars, or direct transfers at each flyby point. It sweeps over multiple Lambert revolution parameter (`M`) combinations.

```bash
python -m asteroid_trajectory_selection.greedy.greedy_selector
```

Results saved under `greedy_asteroid_paths/`.

### 5. Finding the best paths

After optimization data has been generated, find the top paths:

```bash
# Direct transfer paths
python -m asteroid_trajectory_selection.find_best_path

# Greedy paths (across all M-value runs)
python -m asteroid_trajectory_selection.greedy.find_all_paths_greedy
```

### 6. Generating trajectory animations

```bash
# Animate all 10 best paths
python -m asteroid_trajectory_selection.graph_all_best_paths

# Animate notable asteroid orbits
python -m asteroid_trajectory_selection.graphing_notable_asteroids
```

Videos are written as `.mp4` files using matplotlib's FFMpegWriter (requires FFmpeg installed).

### 7. Asteroid trade-off table

Scores and ranks asteroids based on physical properties (mass, radius, eccentricity, inclination, semi-major axis, rotation period, science potential). Does not require SPICE kernels — only needs `sbdb_query_results.csv` from a [JPL SBDB query](https://ssd.jpl.nasa.gov/tools/sbdb_query.html).

```bash
# Basic Chebyshev-bin scoring
python -m tradeoff_table.asteroid_tradeoff

# Enhanced version with two-component science scoring and manual flags
python -m tradeoff_table.asteroid_tradeoff_01
```

Outputs `asteroid_tradeoff.csv` with per-criterion scores and weighted totals.

## Key Algorithms

### Lambert Solver (`lambert.py`)

Solves the orbital boundary-value problem: given two position vectors and a time of flight, find the connecting orbit. Uses two methods:

1. **Izzo's method** — fast Newton-Raphson iteration on a transformed variable. Handles single and multi-revolution cases.
2. **Lancaster-Blanchard fallback** — more robust Halley's method solver, used when Izzo's method fails to converge.

Based on the MATLAB implementation by Rody Oldenhuis (original algorithm by Dr. D. Izzo, ESA/ACT).

### Optimization Strategy

- **Grid search**: each time-of-flight dimension is discretized (N_RES = [5,3,2,3,2,3] for 6 variables) to avoid local minima.
- **Local optimizer**: `scipy.optimize.minimize` with L-BFGS-B (bounded) refines each grid point.
- **Variables** (scaled by YEAR): launch offset, travel time to A1, stay at A1, travel to A2, stay at A2, travel to A3.
- **Bounds**: minimum stay = 3 months, maximum stay = 1 year, travel times = 2 weeks to 8 years.

### Delta-V Budget

Total mission delta-v includes:
- Launch delta-v (Earth departure)
- Arrival delta-v at each asteroid (orbit matching)
- Departure delta-v from each asteroid
- Mars flyby delta-v (if applicable, computed from powered flyby equations when the required turn angle exceeds the maximum achievable by gravity alone at 200 km altitude)

## MATLAB-to-Python Mapping

| MATLAB | Python |
|--------|--------|
| `cspice_*` (mice toolbox) | `spiceypy` |
| `ode113` | `scipy.integrate.solve_ivp` (DOP853) |
| `fmincon` | `scipy.optimize.minimize` (L-BFGS-B) |
| `VideoWriter` / `writeVideo` | `matplotlib.animation.FFMpegWriter` |
| `waitbar` | `tqdm` progress bars |
| `.mat` files (`save`/`load`) | `pickle` (`.pkl` files) |
| MATLAB structs | Python dicts |
| `readtable` / `writetable` | `pandas` |
| 1-based indexing | 0-based indexing |

## Output Files

| File | Description |
|------|-------------|
| `optimal_asteroid_paths/asteroid_data_M1_M2_M3.pkl` | Optimized trajectory data for each asteroid triplet |
| `optimal_asteroid_paths/mars_transfers/asteroid_data_M1_M2_M3.pkl` | Same with Mars flyby |
| `greedy_asteroid_paths/<name>.pkl` | Greedy optimization results for each M-value combo |
| `asteroid_tradeoff.csv` | Ranked asteroid trade-off table |
| `*.mp4` | Trajectory animation videos |
