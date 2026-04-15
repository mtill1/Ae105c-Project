# Three-Asteroid Main Belt Rendezvous — Trajectory Optimization

## Mission Concept

This project designs an optimal spacecraft trajectory to visit **three main-belt asteroids** sequentially from Earth, minimizing total impulsive delta-v (fuel cost). The spacecraft arrives at each asteroid, matches its orbital velocity, stays for 3-12 months of science operations, then departs for the next target.

Two mission architectures are supported:
- **Direct transfer**: Earth -> Asteroid 1 -> Asteroid 2 -> Asteroid 3
- **Mars gravity assist**: Earth -> Mars (flyby) -> A1 -> A2 -> A3

Developed for Ae105c at Caltech/Pomona College.

## Quick Start

```bash
# Install dependencies
cd Python_Consolidated
pip install -r requirements.txt

# Run from the repository root
cd ..
python3 -c "
import sys; sys.path.insert(0, 'Python_Consolidated')
from scripts import run_two_level_optimize
results = run_two_level_optimize()
"
```

## How the Optimization Works

The optimization pipeline has four stages, each narrowing the candidate pool:

### Stage 1: Asteroid Screening

We query the **JPL Small Body Database** for main-belt asteroids (2.0 < a < 3.5 AU, diameter > 30 km, taxonomy known), yielding 406 candidates. We download individual SPK/BSP ephemeris files from JPL Horizons for each candidate, covering 2025-2050.

### Stage 2: Multi-Criteria Scoring (`tradeoff.py`)

Each asteroid is scored 1-10 on eight criteria and ranked by weighted total:

| Criterion | Weight | Direction | Why it matters |
|-----------|--------|-----------|----------------|
| Delta-v accessibility | 30% | Lower better | Dominates mission feasibility |
| Orbital inclination | 20% | Lower better | Plane changes are expensive |
| Science potential | 14% | Higher better | Mission science return |
| Estimated mass | 12% | Higher better | Larger = richer geology |
| Radius | 12% | Higher better | More surface for mapping |
| Eccentricity | 6% | Lower better | High-e raises rendezvous cost |
| Rotation period | 4% | 6-24h optimal | Operations/imaging constraint |
| Semi-major axis | 2% | Lower better | Closer = shorter transfer |

The **science potential** score combines:
- **Characterization depth (40%)**: How well the asteroid is already known (taxonomy, measured diameter, albedo, rotation, color)
- **Intrinsic interest (60%)**: Aqueous alteration features (+2), active main-belt comet (+3), surface ice detection (+3), ambiguous radar albedo (+2), minus penalty for previously visited targets (-3)

This produces a ranked list of ~50 top candidates, exported to `asteroid_tradeoff.csv`. At this stage, the delta-v weight uses a placeholder since trajectory analysis hasn't been run yet.

### Stage 3: Trajectory Optimization (`optimization.py`)

For each candidate triplet (A1, A2, A3), we solve for the **optimal timing** of all mission events:

**Decision variables** (6-dimensional, normalized by year):

| Variable | Physical meaning | Bounds |
|----------|-----------------|--------|
| x1 | Launch date within window | Jan 2027 - Dec 2035 |
| x2 | Transfer time: Earth -> A1 | 2 weeks - 5 years |
| x3 | Stay duration at A1 | 3 months - 1 year |
| x4 | Transfer time: A1 -> A2 | 2 weeks - 5 years |
| x5 | Stay duration at A2 | 3 months - 1 year |
| x6 | Transfer time: A2 -> A3 | 2 weeks - 5 years |

**Hard constraints**:
- Total mission duration < 14 years (BSP ephemeris coverage ends ~2050)
- Each stay between 3 months and 1 year

**Objective function**: For each set of times, we:
1. Look up the heliocentric positions and velocities of Earth and all three asteroids at the relevant epochs using **SPICE ephemerides** (JPL de430 for planets, individual BSP files for asteroids)
2. Solve **Lambert's problem** for each of the three transfer legs using **pykep** (ESA's astrodynamics library, implementing Izzo's algorithm)
3. Compute the velocity mismatch (delta-v) at each of the six maneuver points: Earth departure, A1 arrival, A1 departure, A2 arrival, A2 departure, A3 arrival
4. Sum all six delta-v magnitudes to get the total mission cost

**Lambert's problem**: Given two position vectors and a time of flight, Lambert's problem finds the connecting Keplerian orbit. The solution gives the required velocities at departure and arrival. The difference between these velocities and the actual body velocities is the delta-v the spacecraft must provide.

**Optimizer**: We use `scipy.optimize.differential_evolution`, a population-based global optimizer that:
1. Maintains a population of candidate time vectors
2. Creates trial solutions by combining differences between random members
3. Keeps improvements, discards failures
4. Finishes with L-BFGS-B gradient refinement (`polish=True`)

This replaced an earlier approach that used a 180-point Chebyshev grid with local optimization at each point. Differential evolution explores the 6D search space more thoroughly and finds better global minima.

### Stage 4: Sequence Selection

Given N candidate asteroids, we need to find the best ordered triplet out of N x (N-1) x (N-2) possibilities. Three algorithms are available:

**Brute force** (`generate_optimized_data`): Run the full optimizer on every triplet. Feasible for N < 15 (~2,730 triplets).

**Two-level optimization** (`two_level_optimize`): First, run a cheap coarse optimizer (30 iterations instead of 300) on all N^3 triplets. Then run the full optimizer on only the top 50 candidates. This is ~10x faster with nearly identical results. Supports optional science weighting.

**Beam search** (`beam_search`): Build the path one leg at a time, keeping only the top-K partial sequences at each stage. For N=50 asteroids with beam width K=10, this evaluates ~1,500 single-leg problems instead of 117,600 full triplets. Supports science-weighted scoring.

### Mars Flyby Variant

The Mars variant adds a 7th decision variable (Earth-to-Mars transfer time) and computes the **powered flyby delta-v** at Mars. The flyby delta-v is zero if Mars's gravity alone provides sufficient turning; otherwise, a periapsis burn is computed using pykep's `fb_dv()` function. Minimum flyby altitude is 200 km above Mars's surface.

## Project Structure

```
Ae105c-Project/
├── Python_Consolidated/          # Active codebase (6 Python files)
│   ├── core.py                   # pykep Lambert wrapper, SPICE loading, constants
│   ├── optimization.py           # Delta-v, scoring, DE optimizer, beam search, two-level
│   ├── greedy.py                 # Legacy greedy algorithm
│   ├── visualization.py          # 3D trajectory animation, orbit plotting
│   ├── scripts.py                # Runner entry points
│   ├── tradeoff.py               # Asteroid science scoring (standalone)
│   └── requirements.txt
├── NOTABLE_ASTEROID_BSPs/        # 50 asteroid ephemeris files
├── SPICE_BSPs/                   # Extended asteroid pool (39 BSPs)
├── Renders/Asteroid_Plots/       # Generated visualizations (numbered 01-09)
├── asteroid_tradeoff.csv         # Ranked asteroid table (406 asteroids)
├── plan.md                       # Full mission design document
├── selection_and_optimization.md # Detailed optimization methodology
├── Code/                         # Legacy MATLAB code (not maintained)
└── CLAUDE.md                     # AI assistant guidance
```

## Key Results

### Top Paths Found (15-asteroid subset, two-level optimization)

| Rank | Path | Total dv (km/s) | Launch dv | Mission Duration |
|------|------|:---:|:---:|:---:|
| 1 | Peraga -> Fortuna -> Thekla | 13.44 | 6.09 | 5.5 yr |
| 2 | Peraga -> Fortuna -> Vibilia | 14.66 | 6.09 | -- |
| 3 | Fortuna -> Thekla -> Erato | 14.78 | 6.71 | -- |
| 4 | Fortuna -> Thekla -> Aegina | 14.87 | 6.71 | -- |
| 5 | Peraga -> Fortuna -> Aegina | 15.46 | 6.09 | -- |

### Best Path Timeline (Earth -> Peraga -> Fortuna -> Thekla)

| Event | Date | Elapsed |
|-------|------|---------|
| Earth departure | 2029 Dec 24 | 0 |
| Arrive 554 Peraga | 2031 Jun 19 | 1.5 yr |
| Depart Peraga | 2031 Oct 7 | 1.8 yr |
| Arrive 19 Fortuna | 2033 Aug 26 | 3.7 yr |
| Depart Fortuna | 2033 Nov 26 | 3.9 yr |
| Arrive 586 Thekla | 2035 Jun 20 | 5.5 yr |

## Visualization

Orbit visualizations are in `Renders/Asteroid_Plots/`:

| File | Description |
|------|-------------|
| `01_Static_TopDown_2027-2035.png` | Top-down ecliptic view of all 50 asteroid orbits |
| `02_Static_3D_2027-2035.png` | 3D perspective view |
| `03_Animated_TopDown_2027-2035.gif` | 2D animated GIF (10s at 30fps) |
| `04_Animated_3D_Rotating_2027-2035.gif` | 3D rotating camera animation |
| `05-09_3D_Keyframe_*.png` | Snapshots at 2027, 2029, 2031, 2033, 2035 |

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Numerical arrays |
| scipy | `differential_evolution` global optimizer |
| spiceypy | SPICE kernel loading, ephemeris queries |
| pykep | Lambert solver, orbit propagation, flyby dv (ESA) |
| matplotlib | Plotting, GIF/video animation |
| tqdm | Progress bars |
| pandas | CSV I/O for tradeoff tables |

## References

- Izzo, D. (2015). "Revisiting Lambert's problem." *Celestial Mechanics & Dynamical Astronomy*, 121(1):1-15.
- Storn, R. & Price, K. (1997). "Differential Evolution." *J. Global Optimization*, 11:341-359.
- Carry, B. (2012). "Density of asteroids." *Planetary & Space Science*, 73, 98-118.
- DeMeo, F.E. et al. (2009). "An extension of the Bus asteroid taxonomy." *Icarus*, 202, 160-180.
- pykep: https://esa.github.io/pykep/
- SPICE/NAIF: https://naif.jpl.nasa.gov/naif/
