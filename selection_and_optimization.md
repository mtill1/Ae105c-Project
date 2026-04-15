# Asteroid Selection & Trajectory Optimization Methodology

## Table of Contents

1. [Overview](#1-overview)
2. [Stage 1: Asteroid Candidate Screening](#2-stage-1-asteroid-candidate-screening)
3. [Stage 2: Multi-Criteria Tradeoff Scoring](#3-stage-2-multi-criteria-tradeoff-scoring)
4. [Stage 3: Trajectory Optimization](#4-stage-3-trajectory-optimization)
5. [Stage 4: Sequence Selection Algorithms](#5-stage-4-sequence-selection-algorithms)
6. [Implementation Details](#6-implementation-details)
7. [Results Summary](#7-results-summary)
8. [How to Run](#8-how-to-run)

---

## 1. Overview

The goal is to select three main-belt asteroids of different compositions (C-complex, S-complex, X/M-complex) and find the optimal visitation order that minimizes the total impulsive delta-V for a rendezvous mission launched from Earth.

The pipeline has four stages:

```
  SBDB Query (406 asteroids)
        |
  [Stage 1] Physical/orbital screening
        |
  [Stage 2] Multi-criteria scoring --> Top 50 candidates
        |
  [Stage 3] Lambert-based trajectory optimization (per triplet)
        |
  [Stage 4] Sequence selection (find best triplets from N asteroids)
        |
  Ranked mission candidates with delta-V, timelines, compositions
```

**Key files:**
- `Python_Consolidated/tradeoff.py` -- Stage 2 scoring
- `Python_Consolidated/optimization.py` -- Stage 3 & 4 optimization
- `Python_Consolidated/core.py` -- Lambert solver, SPICE utilities, constants
- `Python_Consolidated/udp.py` -- pagmo UDP classes (MBH + MGA-DSM)
- `Python_Consolidated/beam_search.py` -- composition-aware beam search
- `Python_Consolidated/scripts.py` -- runner entry points

---

## 2. Stage 1: Asteroid Candidate Screening

### 2.1 Data Source

The initial asteroid catalog comes from the **JPL Small Body Database (SBDB)** query saved as `sbdb_query_results.csv`. The query filters:

- Semi-major axis: 2.0 < a < 3.5 AU (main asteroid belt)
- Diameter: > 30 km (scientifically interesting, well-characterized)
- Spectral classification available (Bus-DeMeo or Tholen taxonomy)

This yields **406 candidate asteroids**.

### 2.2 Required Data per Asteroid

| Column | Description | Source |
|--------|-------------|--------|
| `full_name` | Asteroid designation and name | SBDB |
| `a`, `e`, `i` | Orbital elements (AU, dimensionless, degrees) | SBDB |
| `diameter` | Measured diameter (km), or estimated from H-magnitude | SBDB / H-mag formula |
| `spec_B`, `spec_T` | Bus-DeMeo (SMASSII) and Tholen taxonomic class | SBDB |
| `rot_per` | Rotation period (hours) | SBDB |
| `albedo` | Geometric albedo | SBDB |
| `BV` | B-V color index | SBDB |
| `H` | Absolute magnitude | SBDB |

### 2.3 Derived Quantities

**Diameter from H-magnitude** (when direct measurement unavailable):

```
D = 1329 / sqrt(p_v) * 10^(-H/5)    [km]
```

where `p_v` is the geometric albedo (default 0.15 if unknown).

**Density from taxonomy** (Carry, 2012):

| Taxonomic Group | Density (kg/m^3) | Composition |
|-----------------|-------------------|-------------|
| C, B, F, G, D, T | 1,400 | Carbonaceous (primitive) |
| S, Q, A, V, R, L, K | 2,700 | Silicaceous (stony) |
| M, X, E, P | 3,500 | Metallic / enstatite |
| Unknown | 2,000 | Conservative blend |

**Mass** = (4/3) * pi * r^3 * density

---

## 3. Stage 2: Multi-Criteria Tradeoff Scoring

Two scoring versions are implemented. **Version 3 (enhanced)** is the current default.

### 3.1 Scoring Method

Each asteroid receives a score from 1 to 10 on each criterion. Two scoring functions are available:

- **Chebyshev-spaced bins** (`cheb_score`): Boundaries at `v_min + (v_max - v_min)/2 * (1 - cos(k*pi/10))`, concentrating discrimination at distribution extremes.
- **Log-Chebyshev** (`log_cheb_score`): Same but on log10-transformed values (used for mass, radius).
- **Percentile-rank** (`pct_score`): Linear mapping of percentile rank to 1-10 (used for eccentricity, inclination, SMA, rotation, science).

### 3.2 Scoring Criteria and Weights (v3)

| Criterion | Weight | Direction | Scoring Method | Justification |
|-----------|--------|-----------|----------------|---------------|
| Delta-V accessibility | 30% | Lower is better | (from trajectory optimizer) | Dominates feasibility |
| Inclination | 20% | Lower is better | Percentile-rank | Plane changes are expensive |
| Science potential | 14% | Higher is better | Percentile-rank | Mission science value |
| Mass | 12% | Higher is better | Log-Chebyshev | Larger bodies = richer geology |
| Radius | 12% | Higher is better | Log-Chebyshev | More surface area for mapping |
| Eccentricity | 6% | Lower is better | Percentile-rank | High e increases rendezvous cost |
| Rotation period | 4% | 6-24h optimal | Percentile-rank (log-distance from 12h) | Operations constraint |
| Semi-major axis | 2% | Lower is better | Percentile-rank | Closer = shorter transfer |

### 3.3 Science Potential Score (v3)

The science score has two components:

**Component A -- Characterization depth (40%):**
- Bus-DeMeo classification known: +2 (Tholen only: +1)
- Diameter measured with sigma < 5 km: +2 (measured but imprecise: +1)
- Rotation period known: +2
- Albedo measured: +2
- B-V color index available: +1
- Max 10 points

**Component B -- Intrinsic science interest (60%):**
- Hydrated mineral signature (Ch subclass): +2
- B-type (possible organics/ice): +1
- Active main-belt comet: +3
- Surface ice detection: +3
- Ambiguous M-type (radar anomalous): +2
- Previously visited by orbit: -3 (flyby only: -1)
- Max 10 points

**Combined** = 0.40 * A + 0.60 * B

### 3.4 Output

The top 50 asteroids by weighted score are exported to `top_50_asteroids.csv` with all sub-scores. At this stage, the delta-V score is 0 because trajectory data hasn't been computed yet -- the 30% weight is effectively redistributed among the other criteria.

### 3.5 Composition Distribution in Top 50

| Class | Count | Examples |
|-------|-------|---------|
| C-complex | 37 | Themis, Fortuna, Peraga, Ceres, Doris |
| S-complex | 8 | Vesta, Massalia, Parthenope, Urania |
| X/M-complex | 5 | Psyche, Hertha, Lutetia, Beatrix, Lydia |

---

## 4. Stage 3: Trajectory Optimization

### 4.1 Problem Formulation

For a given triplet of asteroids (A1, A2, A3), find the departure/transfer/stay times that minimize total delta-V for the trajectory:

```
Earth --[leg 1]--> A1 --[stay]--> A1 --[leg 2]--> A2 --[stay]--> A2 --[leg 3]--> A3
```

**Decision variables** (6-dimensional, normalized to years):

| Variable | Meaning | Bounds |
|----------|---------|--------|
| x1 | Launch date offset from window start | [0, window_width] |
| x2 | Transfer time: Earth -> A1 | [2 weeks, 5 years] |
| x3 | Stay time at A1 | [3 months, 1 year] |
| x4 | Transfer time: A1 -> A2 | [2 weeks, 5 years] |
| x5 | Stay time at A2 | [3 months, 1 year] |
| x6 | Transfer time: A2 -> A3 | [2 weeks, 5 years] |

**Constraints:**
- Mission duration < 14 years (BSP ephemeris coverage to 2050)
- Launch window: Jan 1, 2027 - Dec 31, 2035

### 4.2 Objective Function

**Total delta-V** = sum of 5 impulsive maneuvers:

```
dV_total = |dV_A1_arrive| + |dV_A1_depart| + |dV_A2_arrive| + |dV_A2_depart| + |dV_A3_arrive|
```

Each maneuver is the velocity difference between the Lambert transfer arc and the body's actual heliocentric velocity:

```
dV_arrive = V_lambert_arrival - V_asteroid
dV_depart = V_lambert_departure - V_asteroid
```

Earth departure delta-V is computed but excluded from the total (launch vehicle provides C3).

**Penalty handling:** Lambert solver failure returns dV = 1000 km/s. Mission duration violations also return 1000 km/s.

### 4.3 Lambert Solver

The Lambert problem is solved using **pykep** (ESA's astrodynamics library), which implements Izzo's algorithm:

```python
lp = pk.lambert_problem(r1, r2, tof, mu_sun, cw=False, max_revs=0)
V1, V2 = lp.get_v1()[0], lp.get_v2()[0]
```

- **Multi-revolution parameter (M):** For M > 0, the transfer orbit completes M full revolutions. Two branches exist per M value (left/right). M=0 is the direct (shortest) transfer.
- **Clockwise flag (cw):** Selects retrograde transfers (used for Mars flyby legs).
- **Units:** All internal computations use km, km/s (SPICE convention). pykep uses SI (m, m/s), so the wrapper converts at the boundary.

### 4.4 Optimization Algorithm

**Scipy `differential_evolution`** with L-BFGS-B polish:

```python
result = differential_evolution(
    score_paths, bounds,
    maxiter=300, tol=1e-7, seed=42,
    polish=True, updating='deferred'
)
```

Differential Evolution is a population-based global optimizer that:
1. Maintains a population of candidate solutions
2. Creates mutants by combining differences between random population members
3. Crosses over with current individuals
4. Accepts trial solutions that improve fitness
5. Finishes with L-BFGS-B gradient descent for local refinement (polish=True)

This replaced the original Chebyshev grid search (540 initial guesses + L-BFGS-B), providing better global exploration with fewer function evaluations.

**Quick variant** for coarse screening: `maxiter=30, popsize=5, polish=False` (~5-10x faster, ~10-20% less accurate).

### 4.5 Ephemeris System

| Setting | Value |
|---------|-------|
| Reference frame | ECLIPJ2000 |
| Observer (center) | Sun (NAIF ID 10) |
| Aberration correction | NONE |
| Time system | ET (seconds past J2000) |
| Planetary ephemeris | de430.bsp |
| Asteroid ephemeris | Individual BSP files (2025-2050) from JPL Horizons |
| Gravitational parameter | pykep.MU_SUN (converted to km^3/s^2) |

### 4.6 MGA-DSM Variant (Deep Space Maneuvers)

An extended 9-dimensional formulation adds mid-leg Deep Space Maneuvers:

| Additional Variables | Meaning | Bounds |
|---------------------|---------|--------|
| x7 (eta_1) | DSM timing fraction, leg 1 | [0.01, 0.99] |
| x8 (eta_2) | DSM timing fraction, leg 2 | [0.01, 0.99] |
| x9 (eta_3) | DSM timing fraction, leg 3 | [0.01, 0.99] |

Each leg is split at fraction eta:
1. Depart body on Lambert arc
2. Coast for eta * tof
3. Apply impulsive DSM (correction burn)
4. Solve Lambert from DSM point to arrival body for remaining (1-eta) * tof

This is the standard MGA-DSM formulation (Vasile & De Pascale, 2006). When eta -> 0 or 1, the DSM vanishes and the solution reduces to pure Lambert.

Implemented in `udp.py` as `AsteroidTripletDSM_UDP` for use with pagmo's Monotonic Basin Hopping.

---

## 5. Stage 4: Sequence Selection Algorithms

Given N candidate asteroids, the problem is to find the best ordered triplet (A_i, A_j, A_k) that minimizes total delta-V. Three approaches are implemented:

### 5.1 Brute Force (N^3)

Evaluate all N * (N-1) * (N-2) ordered triplets with the full optimizer.

- **Complexity:** O(N^3) full optimizations
- **For N=11:** 990 triplets -- feasible (~hours)
- **For N=50:** 117,600 triplets -- impractical with full optimizer

```python
generate_optimized_data(asteroid_list, m_1, m_2, m_3, launch_min, launch_max)
```

### 5.2 Two-Level Optimization

Coarse-filter all N^3 triplets with a fast optimizer, then fine-optimize only the top candidates.

**Pass 1 (Coarse):** Run `optimize_times_quick` (DE with maxiter=30, popsize=5) on every valid triplet. This evaluates ~100 Lambert solves per triplet instead of ~3000.

**Pass 2 (Fine):** Run full `optimize_times` (DE with maxiter=300 + polish) on the top `top_n` candidates from Pass 1.

- **Complexity:** O(N^3) quick evaluations + O(top_n) full evaluations
- **Speedup:** ~10x vs brute force for same top-k results
- **Optional science weighting:** `alpha` parameter blends delta-V and science scores: `1.0 = pure delta-V`, `0.7 = 70% dV + 30% science`

```python
two_level_optimize(asteroid_list, m_1=0, m_2=0, m_3=0,
                   launch_utc_min, launch_utc_max, top_n=15)
```

### 5.3 Beam Search

Build sequences stage-by-stage, keeping only the top-k (beam width) partial sequences at each depth.

**Stage 1:** Earth -> each of N asteroids. Optimize single-leg transfer, keep top k.

**Stage 2:** For each of k survivors, try extending to each remaining asteroid. Optimize second leg. Keep top k two-asteroid sequences.

**Stage 3:** Same process to get top k complete triplets.

- **Complexity:** O(N * k * 3) single-leg optimizations
- **For N=50, k=10:** ~1,500 leg evaluations vs 117,600 triplet evaluations
- **Composition filtering:** `beam_search.py` supports constraining results to include asteroids from each specified composition class (e.g., `{'C', 'S', 'M'}`)

Two implementations exist:
- `optimization.beam_search(...)` -- uses DE for each leg, includes science weighting
- `beam_search.beam_search_optimize(...)` -- uses quick Lambert screening + full optimization refinement, supports composition filtering

### 5.4 Comparison of Approaches

| Method | N=15 time | N=50 time | Quality | Best for |
|--------|-----------|-----------|---------|----------|
| Brute force N^3 | ~4 min | ~hours | Optimal | Small N |
| Two-level | ~4 min | ~90 min | Near-optimal | Medium N |
| Beam search (k=10) | ~30 sec | ~5 min | Good (may miss) | Large N, quick screening |

---

## 6. Implementation Details

### 6.1 Software Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pykep | 2.6 | Lambert solver, orbit propagation, flyby dV |
| pygmo | 2.19 | Monotonic Basin Hopping, island-model parallelism |
| spiceypy | latest | SPICE kernel management, ephemeris queries |
| scipy | latest | `differential_evolution` optimizer |
| numpy | latest | Numerical arrays |
| pandas | latest | Tradeoff table I/O |
| matplotlib | latest | Visualization, GIF animation |

### 6.2 SPICE Kernel Setup

**Generic kernels** (required, must be in `generic_kernels/` or symlinked):

```
generic_kernels/
  lsk/naif0012.tls           -- leap seconds
  spk/planets/de430.bsp      -- planetary ephemeris
  spk/satellites/jup310.bsp  -- Jupiter satellite ephemeris
  pck/gm_de431.tpc           -- gravitational parameters
  pck/pck00010.tpc           -- planetary constants
```

**Asteroid BSP files** (in `NOTABLE_ASTEROID_BSPs/`, one per asteroid):

Downloaded from JPL Horizons API covering 2025-01-01 to 2050-01-01. NAIF IDs follow the convention `2000XXXX` where `XXXX` is the asteroid number.

```python
# Download example (from JPL Horizons API):
params = {'COMMAND': "'586'", 'EPHEM_TYPE': 'SPK',
          'START_TIME': "'2025-01-01'", 'STOP_TIME': "'2050-01-01'"}
# Response contains base64-encoded SPK in the 'spk' field
```

For "pre-computed major bodies" (Ceres, Vesta), use `'N;'` command format (trailing semicolon) to force small-body lookup.

### 6.3 Unit Conventions

| Quantity | Internal Unit | SPICE Unit | pykep Unit |
|----------|---------------|------------|------------|
| Position | km | km | m |
| Velocity | km/s | km/s | m/s |
| Time | seconds (ET) | seconds (ET) | seconds |
| Gravitational parameter | km^3/s^2 | km^3/s^2 | m^3/s^2 |
| Time of flight (Lambert) | days | -- | seconds |

Conversions happen at the `core.py` wrapper boundary: `_KM2M = 1e3`, `_M2KM = 1e-3`.

### 6.4 Key Constants

```python
MINUTE = 60
HOUR   = 3600
DAY    = 86400
WEEK   = 604800
MONTH  = 2419200      # 4 weeks
YEAR   = 31557600     # 365.25 days (Julian year)
MAX_MISSION_DURATION = 14 * YEAR  # 2035 launch + 14yr < 2050 BSP coverage
```

---

## 7. Results Summary

### 7.1 Top 15 Paths (from 15-asteroid two-level optimization)

| Rank | Path (Earth ->) | Total dV (km/s) | Launch dV (km/s) |
|------|-----------------|-----------------|-----------------|
| 1 | Peraga -> Fortuna -> Thekla | 13.44 | 6.09 |
| 2 | Peraga -> Fortuna -> Vibilia | 14.66 | 6.09 |
| 3 | Fortuna -> Thekla -> Erato | 14.78 | 6.71 |
| 4 | Fortuna -> Thekla -> Aegina | 14.87 | 6.71 |
| 5 | Peraga -> Fortuna -> Aegina | 15.46 | 6.09 |
| 6 | Peraga -> Antiope -> Vibilia | 15.62 | 5.97 |
| 7 | Fortuna -> Peraga -> Themis | 15.81 | 6.49 |
| 8 | Peraga -> Fortuna -> Erato | 15.91 | 6.30 |
| 9 | Fortuna -> Peraga -> Thekla | 16.05 | 5.41 |
| 10 | Fortuna -> Thekla -> Circe | 16.13 | 6.74 |
| 11 | Circe -> Doris -> Concordia | 16.18 | 6.62 |
| 12 | Fortuna -> Vibilia -> Antiope | 16.20 | 6.38 |
| 13 | Fortuna -> Peraga -> Aglaja | 16.33 | 6.34 |
| 14 | Aegina -> Themis -> Doris | 16.49 | 6.24 |
| 15 | Peraga -> Fortuna -> Concordia | 17.54 | 7.09 |

### 7.2 Best Path Timeline

**Earth -> Peraga -> Fortuna -> Thekla** (13.44 km/s total):

| Event | Date | Elapsed |
|-------|------|---------|
| Earth departure | 2029 Dec 24 | 0 |
| Arrive Peraga | 2031 Jun 19 | 1.5 yr |
| Depart Peraga | 2031 Oct 7 | 1.8 yr |
| Arrive Fortuna | 2033 Aug 26 | 3.7 yr |
| Depart Fortuna | 2033 Nov 26 | 3.9 yr |
| Arrive Thekla | 2035 Jun 20 | 5.5 yr |

### 7.3 Composition Note

The top 15 paths above are all C-complex asteroids (Peraga=Ch, Fortuna=Ch, Thekla=Ch, etc.). For **compositional diversity** (C+S+M), the beam search with `composition_filter={'C', 'S', 'M'}` should be used, or the full 50-asteroid optimization expanded to include S-complex (Massalia, Vesta, Urania) and X/M-complex (Hertha, Lutetia, Psyche) targets.

---

## 8. How to Run

All commands assume the working directory is the repository root.

### 8.1 Setup

```bash
# Install dependencies
mamba install -c conda-forge pykep pygmo spiceypy

# Ensure generic SPICE kernels are linked
ln -sf /path/to/generic_kernels ./generic_kernels

# Asteroid BSP files should be in NOTABLE_ASTEROID_BSPs/
ls NOTABLE_ASTEROID_BSPs/*.bsp | wc -l   # should be 50
```

### 8.2 Run Tradeoff Scoring

```python
import sys; sys.path.insert(0, 'Python_Consolidated')
from tradeoff import run_tradeoff_v3
run_tradeoff_v3('sbdb_query_results.csv', 'asteroid_tradeoff.csv')
```

### 8.3 Run Two-Level Optimization (Recommended)

```python
import sys; sys.path.insert(0, 'Python_Consolidated')
from core import load_kernels
from optimization import two_level_optimize

asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs', 'generic_kernels')

# Filter to top N by name if desired
top_names = ['THEMIS', 'THEKLA', 'FORTUNA', 'PERAGA', ...]
subset = [a for a in asteroid_list if a['NAME'] in top_names]

results = two_level_optimize(
    subset, m_1=0, m_2=0, m_3=0,
    launch_utc_min='Jan 1 12:00:00 UTC 2027',
    launch_utc_max='Dec 31 12:00:00 UTC 2035',
    top_n=15
)
```

### 8.4 Run Beam Search with Composition Filter

```python
from beam_search import beam_search_optimize

results = beam_search_optimize(
    asteroid_list,
    launch_utc_min='Jan 1 12:00:00 UTC 2027',
    launch_utc_max='Dec 31 12:00:00 UTC 2035',
    beam_width=15,
    composition_filter={'C', 'S', 'M'}
)
```

### 8.5 Run Brute Force (Small N only)

```python
from optimization import generate_optimized_data

data = generate_optimized_data(
    asteroid_list, 0, 0, 0,
    'Jan 1 12:00:00 UTC 2027', 'Dec 31 12:00:00 UTC 2035'
)
```

### 8.6 Using Script Entry Points

```python
from scripts import run_two_level_optimize, run_beam_search

results = run_two_level_optimize()
# or
results = run_beam_search(beam_width=15)
```

---

## References

- Carry, B. (2012). "Density of asteroids." *Planetary & Space Science*, 73, 98-118.
- DeMeo, F.E. et al. (2009). "An extension of the Bus asteroid taxonomy." *Icarus*, 202, 160-180.
- Izzo, D. (2015). "Revisiting Lambert's problem." *Celestial Mechanics & Dynamical Astronomy*, 121(1):1-15.
- Storn, R. & Price, K. (1997). "Differential Evolution." *J. Global Optimization*, 11:341-359.
- Taylor, C.R. et al. (2018). "A Delta-V map of the known Main Belt Asteroids." *Acta Astronautica*, 146, 73-82.
- Vasile, M. & De Pascale, P. (2006). "On the Preliminary Design of MGA Trajectories." *JGCD*, 29(6):1347-1361.
- Wales, D.J. & Doye, J.P.K. (1997). "Global Optimization by Basin-Hopping." *J. Phys. Chem. A*, 101(28):5111-5116.
- pykep: https://esa.github.io/pykep/
- pagmo: https://esa.github.io/pagmo2/
- SPICE/NAIF: https://naif.jpl.nasa.gov/naif/
