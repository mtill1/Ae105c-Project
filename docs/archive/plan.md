# Mission Design Plan: Three-Asteroid Main Belt Rendezvous

> **New to this project?** Read [`README.md`](README.md) first — it has step-by-step setup and run instructions for someone with no Python experience.
> This document is the deep technical design spec.

## Table of Contents

1. [Mission Statement and Requirements](#1-mission-statement-and-requirements)
2. [Asteroid Selection Methodology](#2-asteroid-selection-methodology)
3. [Trajectory Design and Orbital Dynamics](#3-trajectory-design-and-orbital-dynamics)
4. [Delta-V Budget and Propulsion Trades](#4-delta-v-budget-and-propulsion-trades)
5. [Spacecraft Subsystem Design](#5-spacecraft-subsystem-design)
6. [Mission Operations Plan](#6-mission-operations-plan)
7. [Risk Assessment](#7-risk-assessment)
8. [Key References](#8-key-references)

---

## 1. Mission Statement and Requirements

### 1.1 Mission Objective

Design a spacecraft mission to rendezvous with three (3) main-belt asteroids of different presumed composition, perform in-situ science at each for 3-6 months, and return data to Earth. The mission provides comparative planetology across the three major asteroid compositional classes (C-complex, S-complex, X-complex), addressing fundamental questions about solar system formation, differentiation, and volatile delivery.

### 1.2 Top-Level Requirements

| Requirement | Value | Rationale |
|---|---|---|
| Number of asteroid targets | 3 | Course requirement |
| Compositional diversity | One each: C, S, X/M complex | Different formation environments |
| Stay time per asteroid | 3-6 months | Sufficient for survey + mapping orbits |
| Instrument suite | Visible/IR imaging spectrometer, camera, magnetometer | Specified payload |
| Launch window | 2027-2035 | Course constraint |
| Mission type | Rendezvous (velocity matching) | 3-6 month stays require orbit insertion |

### 1.3 Derived Requirements

| Parameter | Derived Value | Source |
|---|---|---|
| Total mission delta-V (spacecraft) | 10-12 km/s (SEP) or 20-25 km/s (chemical impulsive) | Lambert solver results |
| Mission duration | 8-14 years | Transfer times + stay times |
| Max heliocentric distance | ~3.2 AU | Hygiea/Klymene orbits |
| Min solar array power at max distance | ~1.3 kW | Ion propulsion + bus at 3.15 AU |
| Pointing accuracy | < 1 mrad | IR spectrometer requirement |
| Data downlink rate at 3 AU | 10-120 kbps | X-band, 1.5m HGA, 34m DSN |

---

## 2. Asteroid Selection Methodology

### 2.1 Selection Pipeline

The asteroid selection pipeline proceeds in three stages: (1) initial screening from the JPL Small Body Database, (2) multi-criteria scoring and ranking, and (3) trajectory-constrained combinatorial optimization.

#### Stage 1: SBDB Screening

The `sbdb_query_results.csv` contains 406 main-belt asteroids queried from the JPL Small Body Database. Initial filters applied:

- **Semi-major axis**: 2.0 < a < 3.5 AU (main belt)
- **Diameter**: > 30 km (scientifically interesting, well-characterized)
- **Spectral classification available** (Bus-DeMeo or Tholen taxonomy known)

#### Stage 2: Multi-Criteria Scoring (`asteroid_tradeoff.m`)

Each asteroid is scored 1-10 on seven criteria using Chebyshev-spaced bin boundaries, then combined via weighted sum. The Chebyshev spacing (boundaries at `vmin + (vmax-vmin)/2 * (1 - cos(k*pi/10))`) concentrates discriminating power at the distribution extremes where the best and worst candidates cluster.

**Scoring Weights:**

| Criterion | Weight | Direction | Justification |
|---|---|---|---|
| Delta-V accessibility | 30% | Lower is better | Dominates feasibility; Taylor et al. (2018) showed only 3,986 of ~700,000 MBAs have delta-V < 8 km/s |
| Eccentricity | 15% | Lower is better | High e increases velocity mismatch at rendezvous and complicates phasing |
| Mass | 13% | Higher is better | Larger bodies enable orbit insertion, have richer geology |
| Inclination | 12% | Lower is better | Plane changes are the most expensive maneuver type; cutoff should be i < 15 deg |
| Science potential | 10% | Higher is better | Proxy for pre-mission characterization depth (taxonomy, rotation, albedo, color data known) |
| Radius | 10% | Higher is better | Larger surface area for mapping, stronger gravity for stable orbits |
| Semi-major axis | 5% | Lower is better | Closer to Sun = shorter transfer, more solar power |
| Rotation period | 5% | 6-24h optimal | Operations-driven: too fast (<2h) limits imaging; too slow (>200h) limits full-surface coverage |

**Density estimation from taxonomy** (following Carry, 2012, *Planetary & Space Science*):

| Taxonomic Group | Density (kg/m^3) | Meteorite Analogues |
|---|---|---|
| C/B/F/G/D/T (carbonaceous) | 1,400 | CI/CM carbonaceous chondrites |
| S/Q/A/V/R/L/K (silicaceous) | 2,700 | Ordinary chondrites, HED achondrites |
| M/X/E/P (metallic/enstatite) | 3,500 | Iron meteorites, enstatite achondrites |
| Unknown | 2,000 | Conservative blend |

These values are well-supported by Carry (2012), who compiled mass/volume estimates for 287 small bodies and found average densities of C~1.0-2.0, S~2.5-3.5, M~3.0-5.0 g/cm^3. The Krasinsky et al. mean values (C=1.38, S=2.71, M=5.32 g/cm^3) bracket our selections.

#### Stage 3: Trajectory-Constrained Combinatorial Optimization

The current codebase implements two approaches:

**Exhaustive search** (`asteroid_selector.m` + `GENERATE_OPTIMIZED_DATA.m`): For the 11 notable asteroids, evaluates all 11x10x9 = 990 ordered triplets. For each triplet, `OPTIMIZE_TIMES` runs fmincon with a Chebyshev grid of initial guesses (N_RES = [5,3,2,3,2,3] = 540 starting points) over a 6-dimensional time space. Three M-value (orbital revolution count) combinations have been computed: (0,0,0), (1,0,0), (0,1,0). Results are stored in `optimal_asteroid_paths/asteroid_data_{m1}_{m2}_{m3}.mat`.

**Greedy heuristic** (`GREEDY/greedy_selector.m`): Builds paths incrementally, selecting the cheapest next transfer at each step. Faster (O(N*K) vs O(N^3)) but produces paths 30-100% worse than optimal, consistent with the literature on greedy TSP-like solutions. The commit history notes "it just gives too suboptimal results."

### 2.2 Candidate Asteroids in the Notable Pool

The 11 asteroids in `NOTABLE_ASTEROID_BSPs/` span all three compositional classes:

| Asteroid | Taxonomic Class | Composition | a (AU) | e | i (deg) | Radius (km) |
|---|---|---|---|---|---|---|
| **C-complex targets:** | | | | | | |
| 10 Hygiea | C (Bus) / C (Tholen) | Carbonaceous | 3.14 | 0.117 | 3.83 | ~215 |
| 24 Themis | B (Bus) / C (Tholen) | Primitive carbonaceous | 3.13 | 0.133 | 0.76 | ~99 |
| 48 Doris | Ch (Bus) / CG (Tholen) | Hydrated carbonaceous | 3.11 | 0.069 | 6.55 | ~110 |
| 34 Circe | C | Carbonaceous | 3.00 | 0.108 | 5.50 | ~57 |
| 104 Klymene | C | Carbonaceous | 3.15 | 0.156 | 2.79 | ~62 |
| **S-complex targets:** | | | | | | |
| 20 Massalia | S | Silicaceous stony | 2.41 | 0.143 | 0.71 | ~73 |
| 58 Concordia | S-related | Silicaceous stony | 2.70 | 0.042 | 5.06 | ~47 |
| **X-complex targets:** | | | | | | |
| 110 Lydia | X/M | Metallic | 2.73 | 0.083 | 5.86 | ~43 |
| 135 Hertha | M (Tholen) | Metallic | 2.43 | 0.207 | 2.30 | ~40 |
| 96 Aegina | X | Metallic/enstatite | 2.59 | 0.113 | 2.11 | ~36 |
| 554 Peraga | -- | Unknown | 2.37 | 0.149 | 2.89 | ~33 |

### 2.3 Recommended Target Selection

Based on your greedy optimizer results in `asteroid_optimized_data_table.csv`, the best-performing triplets (lowest asteroid delta-V) with compositional diversity are:

| Rank | Triplet (order visited) | Asteroid dV (km/s) | Launch dV (km/s) | Compositions |
|---|---|---|---|---|
| 1 | **Hertha -> Hygiea -> Klymene** | 22.6 | 7.08 | M -> C -> C |
| 2 | **Lydia -> Hygiea -> Klymene** | 25.8 | 10.2 | X -> C -> C |
| 3 | **Themis -> Hygiea -> Klymene** | 30.2 | 6.0 | B/C -> C -> C |
| 4 | **Massalia -> Peraga -> Klymene** | 32.1 | 4.0 | S -> ? -> C |

**Issue**: None of the top greedy paths achieve full C+S+M compositional diversity. The best path (Hertha->Hygiea->Klymene) visits one M-type and two C-types. This must be addressed through:

1. **Enforcing a composition constraint** in the exhaustive optimizer: filter triplets to require exactly one C, one S, one X/M target before running fmincon. This reduces 990 triplets to the ~200 that satisfy the diversity constraint.
2. **Running the exhaustive optimizer with composition filtering**: Among the notable asteroids, valid diverse triplets include combinations like:
   - Hygiea(C) + Massalia(S) + Hertha(M)
   - Themis(C) + Massalia(S) + Lydia(X)
   - Doris(C) + Concordia(S) + Aegina(X)
   - Klymene(C) + Massalia(S) + Hertha(M)

### 2.4 Taxonomy References

| Reference | Contribution |
|---|---|
| Tholen, D.J. (1984). "Asteroid taxonomy from cluster analysis of photometry." PhD thesis, U. Arizona. | Original 14-class taxonomy from ECAS 8-color photometry |
| Bus, S.J. & Binzel, R.P. (2002). "Phase II of the Small Main-Belt Asteroid Spectroscopic Survey." *Icarus*, 158, 146-177. | Bus/SMASS II taxonomy, 26 classes from 0.44-0.92 um spectra of 1,447 asteroids |
| DeMeo, F.E., Binzel, R.P., Slivan, S.M., Bus, S.J. (2009). "An extension of the Bus asteroid taxonomy into the near-infrared." *Icarus*, 202, 160-180. | Bus-DeMeo taxonomy, extended to 2.45 um with 24 classes; current standard |
| Carry, B. (2012). "Density of asteroids." *Planetary & Space Science*, 73, 98-118. | Compiled bulk densities for 287 bodies by taxonomic class |
| Taylor, C.R., Seager, S., et al. (2018). "A Delta-V map of the known Main Belt Asteroids." *Acta Astronautica*, 146, 73-82. | Delta-V accessibility mapping for all known MBAs |

---

## 3. Trajectory Design and Orbital Dynamics

### 3.1 Lambert Problem Formulation

The trajectory between any two points in the mission is modeled as a Lambert arc: given positions **r1** and **r2** and a time of flight t_f, find the conic orbit connecting them under two-body (Sun) gravity.

The time-of-flight equation for the elliptic case:

```
t_f = sqrt(a^3 / mu) * [(alpha - sin(alpha)) - (beta - sin(beta)) + 2*pi*M]
```

where alpha and beta are geometric parameters derived from the semi-major axis, chord length, and semi-perimeter, and M is the number of complete revolutions.

The codebase implements a two-tier Lambert solver (`lambert.m`, ~800 lines):

1. **Primary: Izzo's algorithm** (Izzo, 2015, *Celestial Mechanics & Dynamical Astronomy*) -- Uses Householder iterations on a logarithmic/tangent transform of the universal variable x. Converges in 3-5 iterations for M=0.
2. **Fallback: Lancaster-Blanchard-Gooding** (Lancaster & Blanchard, 1969, NASA TN D-5368; Gooding, 1990, *Celestial Mechanics*) -- Halley's third-order method on the time-of-flight function. Activated after 15 failed Izzo iterations.

**Multi-revolution solutions (M > 0)**: For each M value, two solutions exist: a "left branch" (lower energy, larger semi-major axis) and a "right branch" (higher energy). Multi-revolution solutions are important for main-belt transfers with flight times > 2 years because they can yield lower delta-V than the direct (M=0) transfer. The codebase sweeps M={0,1} for each leg, with results stored in separate .mat files.

### 3.2 Trajectory Optimization Architecture

The optimization pipeline has been upgraded from the original grid-search + local optimizer to a multi-layer system using global optimization, mid-leg Deep Space Maneuvers (DSMs), and beam search for sequence selection.

#### 3.2.1 Decision Variables

**Pure Lambert mode (6D)** — baseline formulation:

| Variable | Physical Meaning | Bounds |
|---|---|---|
| x1 | Launch date offset from window start | [0, window_end - window_start] |
| x2 | Transfer time: Earth -> Asteroid 1 | [2 weeks, 8 years] |
| x3 | Stay time at Asteroid 1 | [3 months, 1 year] |
| x4 | Transfer time: Asteroid 1 -> Asteroid 2 | [2 weeks, 8 years] |
| x5 | Stay time at Asteroid 2 | [3 months, 1 year] |
| x6 | Transfer time: Asteroid 2 -> Asteroid 3 | [2 weeks, 8 years] |

**MGA-DSM mode (9D)** — adds Deep Space Maneuver timing per leg:

| Variable | Physical Meaning | Bounds |
|---|---|---|
| x1-x6 | Same as above | Same |
| x7 (eta_1) | DSM timing fraction, leg 1 | [0.01, 0.99] |
| x8 (eta_2) | DSM timing fraction, leg 2 | [0.01, 0.99] |
| x9 (eta_3) | DSM timing fraction, leg 3 | [0.01, 0.99] |

Each eta_i splits a transfer leg into two sub-arcs joined by an impulsive Deep Space Maneuver. The spacecraft departs on a Lambert arc, coasts for fraction eta of the transfer time, applies a correction burn (DSM), then follows a new Lambert arc to the destination. This is the standard MGA-DSM formulation used in ESA mission design (Vasile & De Pascale, 2006). When eta ~ 0 or ~1, the DSM vanishes and the solution reduces to a pure Lambert arc, so MGA-DSM can only improve upon the baseline.

#### 3.2.2 Objective Function

The objective minimizes the total impulsive delta-V, the sum of all maneuvers:

**Pure Lambert:** 5 maneuvers (arrive/depart at each asteroid, arrive at final)
```
delta_V_total = |dV_A1_arrive| + |dV_A1_depart| + |dV_A2_arrive| + |dV_A2_depart| + |dV_A3_arrive|
```

**MGA-DSM:** 5 maneuvers + 3 DSM burns (one per leg)
```
delta_V_total = |dV_A1_arr| + |dV_DSM_1| + |dV_A1_dep| + |dV_DSM_2| + |dV_A2_arr| + |dV_A2_dep| + |dV_DSM_3| + |dV_A3_arr|
```

Earth departure delta-V is tracked separately (launch vehicle provides C3). A mission duration constraint (15 years max) is enforced as a penalty (dV = 1000 km/s for violations). Lambert solver failures are penalized identically.

#### 3.2.3 Optimization Methods

Three complementary methods are implemented in `Python_Consolidated/optimization.py`:

**Method 1: Differential Evolution + L-BFGS-B polish** (`optimize_times`)

scipy's `differential_evolution` is a population-based global optimizer that explores the search space stochastically and finishes with an L-BFGS-B gradient descent polish. Replaces the old 540-point Chebyshev grid. Configuration: `maxiter=300`, `tol=1e-7`, `seed=42`, `polish=True`.

**Method 2: pagmo Monotonic Basin Hopping (MBH)** (`udp.py` + pagmo)

pagmo (Parallel Global Multiobjective Optimizer) from ESA provides Monotonic Basin Hopping wrapping NLopt SLSQP as the local solver. MBH was specifically designed for trajectory optimization at ESA's Advanced Concepts Team (Wales & Doye, 1997). It repeatedly perturbs the current best solution and runs local optimization, accepting only monotonically improving results.

The `udp.py` module defines pagmo-compatible User Defined Problem (UDP) classes:
- `AsteroidTripletUDP` — 6D pure Lambert
- `AsteroidTripletDSM_UDP` — 9D MGA-DSM
- `MarsTripletUDP` — 7D Mars flyby variant

Island-model parallelism runs multiple MBH instances across CPU cores (`pg.archipelago`), extracting the best result across all islands.

**Method 3: Two-Level Optimization** (`two_level_optimize`)

For large asteroid catalogs, evaluates all N^3 triplets with a fast coarse optimizer (`differential_evolution` with `maxiter=30`, `popsize=5`), then fine-optimizes only the top candidates with the full optimizer. Reduces computation by ~10x for the same top-k results.

#### 3.2.4 Sequence Selection: Beam Search

The beam search (`beam_search.py` and `optimization.beam_search`) replaces both the O(N^3) exhaustive enumeration and the greedy heuristic (beam width=1) with configurable beam width k:

1. **Stage 1:** Earth -> each of N asteroids. Quick Lambert screen, keep top k.
2. **Stage 2:** For each of k survivors, extend to each remaining asteroid. Keep top k.
3. **Stage 3:** Repeat, producing k complete triplets.
4. **Refinement:** Run full optimization only on the top-k triplets.

**Complexity:** O(N * k * L) quick screens + k full optimizations, vs. N^3 full optimizations for exhaustive. For N=100, k=20: ~6,000 screens + 20 full runs vs. 1,000,000 full runs.

The beam search in `beam_search.py` also supports **composition filtering** — constraining sequences to include at least one asteroid from each compositional class (C, S, M) to guarantee scientific diversity.

#### 3.2.5 Optimization Architecture Summary

```
                    ┌─────────────────────────────────┐
                    │  SEQUENCE SELECTION LAYER        │
                    │  beam_search / two_level_optimize│
                    │  Identifies top-k triplets       │
                    └───────────────┬─────────────────┘
                                    │ top-k triplets
                    ┌───────────────▼─────────────────┐
                    │  GLOBAL OPTIMIZATION LAYER       │
                    │  differential_evolution or        │
                    │  pagmo MBH (island-model)        │
                    │  Optimizes times for each triplet│
                    └───────────────┬─────────────────┘
                                    │ optimal times
                    ┌───────────────▼─────────────────┐
                    │  TRAJECTORY EVALUATION LAYER     │
                    │  Lambert solver (pykep Izzo)     │
                    │  + optional mid-leg DSMs         │
                    │  Computes delta-V breakdown      │
                    └─────────────────────────────────┘
```

#### 3.2.6 Optimization References

| Reference | Contribution |
|---|---|
| Wales, D.J. & Doye, J.P.K. (1997). "Global Optimization by Basin-Hopping." *J. Phys. Chem. A*, 101(28):5111-5116. | Monotonic Basin Hopping algorithm |
| Storn, R. & Price, K. (1997). "Differential Evolution." *J. Global Optimization*, 11:341-359. | Differential Evolution algorithm |
| Vasile, M. & De Pascale, P. (2006). "On the Preliminary Design of Multiple Gravity-Assist Trajectories." *JGCD*, 29(6):1347-1361. | MGA-DSM formulation with eta parameter |
| Sims, J.A. & Flanagan, S.N. (1999). "Preliminary design of low-thrust interplanetary missions." AAS/AIAA Astrodynamics Specialist Conf. | Sims-Flanagan transcription for low-thrust |
| Izzo, D. et al. pagmo: https://esa.github.io/pagmo2/ | Parallel global optimizer framework |
| pykep: https://esa.github.io/pykep/ | ESA trajectory design toolkit |

### 3.3 Reference Frame and Ephemeris

All computations use the SPICE ephemeris system:

| Setting | Value | Justification |
|---|---|---|
| Reference frame | ECLIPJ2000 | Standard for heliocentric mission design |
| Observer (center body) | 10 (Sun) | Heliocentric trajectories |
| Aberration correction | NONE | Geometric positions for trajectory design |
| Time system | ET (Ephemeris Time = TDB) | SPICE standard; seconds past J2000 |
| Gravitational parameter | cspice_bodvcd(10, 'GM', 10) | Sun mu from SPICE PCK |
| Planetary ephemeris | de430.bsp | JPL Development Ephemeris 430 |

### 3.4 Porkchop Plot Analysis

For the project report, generating porkchop plots for individual transfer legs would strengthen the trajectory analysis. The procedure for each leg:

1. Fix departure body and arrival body
2. Create a grid of departure dates (x-axis) and arrival dates (y-axis)
3. At each grid point, solve the Lambert problem using SPICE states
4. Compute C3 = |V_departure - V_body_departure|^2 and V_inf_arrival = |V_arrival - V_body_arrival|
5. Contour plot C3 and V_inf_arrival

**Typical C3 values for main belt missions**:
- Earth to inner belt (2.0-2.5 AU): C3 ~ 10-30 km^2/s^2
- Earth to mid belt (2.5-3.0 AU): C3 ~ 15-50 km^2/s^2
- Earth to outer belt (3.0-3.5 AU): C3 ~ 25-65 km^2/s^2
- Dawn mission actual: C3 = 10.4 km^2/s^2 (very favorable window)

### 3.5 Mars Gravity Assist Option

The Mars transfer variant (`mars_transfer_selector.m` + `COMPUTE_MARS_PATH_DELTAV.m`) adds a Mars flyby between Earth departure and the first asteroid. The maximum gravity assist turn angle is:

```
delta_max = 2 * arcsin(1 / (1 + r_periapsis * V_inf^2 / mu_Mars))
```

With mu_Mars = 42,828 km^3/s^2 and R_Mars = 3,396 km, at a minimum flyby altitude of 200 km:

| V_infinity at Mars (km/s) | Max Turn Angle (deg) | Practical delta-V Savings |
|---|---|---|
| 3 | ~38 | ~2 km/s |
| 5 | ~14 | ~1 km/s |
| 8 | ~5.5 | ~0.5 km/s |

When the desired turn angle exceeds delta_max, a powered flyby at Mars periapsis adds the remaining delta-V. The code correctly implements this in `COMPUTE_MARS_PATH_DELTAV.m`.

**Assessment**: Mars gravity assists are moderately useful for chemical-propulsion main-belt missions (providing ~0.5-2 km/s savings), and are essential for SEP missions (both Dawn and Psyche used Mars flybys). They also provide a free inclination change, valuable for reaching asteroids with i > 5 deg.

### 3.6 Rendezvous vs. Flyby Delta-V Cost

Your mission requires rendezvous (3-6 month stays), not flybys. The delta-V penalty for rendezvous is substantial:

For a main-belt asteroid at 2.7 AU, orbital velocity ~ 18 km/s. The arrival V-infinity from a Lambert arc is typically 2-8 km/s, which must be nulled at each asteroid. Including departure:

| Scenario | Per-Asteroid Cost | 3-Asteroid Total |
|---|---|---|
| Flyby | ~0 km/s (trajectory shaping only) | ~0 km/s maneuver |
| Rendezvous (arrive + depart) | 4-13 km/s | 12-39 km/s |

This is precisely why Dawn required ion propulsion (Isp=3100s) rather than chemical (Isp=320s): the mass ratio for 10 km/s with chemical is exp(10/3.06)=26x, while with ion it is exp(10/30.4)=1.39x.

**Stay time impact on delta-V**: During the 3-6 month stay, the asteroid moves along its orbit (at 2.7 AU, ~18 km/s orbital velocity). In 6 months, it moves ~30 degrees. The optimizer finds the stay time that minimizes the sum of arrival + departure delta-V by balancing departure geometry against next-leg transfer efficiency. Your optimizer bounds (3 months to 1 year) are appropriate.

### 3.7 Trajectory Design References

| Reference | Contribution |
|---|---|
| Izzo, D. (2015). "Revisiting Lambert's problem." *Celestial Mechanics & Dynamical Astronomy*, 121(1):1-15. | Primary Lambert solver algorithm used in code |
| Lancaster, E.R. & Blanchard, R.C. (1969). NASA TN D-5368. | Universal-variable Lambert formulation (fallback solver) |
| Gooding, R.H. (1990). *Celestial Mechanics*, 48:145-165. | Halley iteration for Lambert (fallback solver) |
| Battin, R.H. (1999). *An Introduction to the Methods of Astrodynamics*. AIAA. | Textbook reference for Lambert's theorem |
| Olympio, J.T. (2011). ESA/ACT technical report. | Branch-and-bound multi-gravity-assist trajectory search |
| Chen, Y., Baoyin, H., Li, J. (2014). *Advances in Space Research*, 53(4):697-710. | Design of trajectory visiting 10 asteroids |
| Englander, J.A., Conway, B.A., Williams, T. (2012). *JGCD*, 35(6):1878-1887. | Automated mission planning via evolutionary algorithms |

---

## 4. Delta-V Budget and Propulsion Trades

### 4.1 Current Optimizer Results

From the exhaustive optimizer (M=0,0,0) and greedy searches, the best paths found so far:

**Best greedy paths** (from `asteroid_optimized_data_table.csv`):

| Path | Asteroid dV (km/s) | Launch dV (km/s) | Total (km/s) |
|---|---|---|---|
| Hertha -> Hygiea -> Klymene | 22.6 | 7.1 | 29.7 |
| Lydia -> Hygiea -> Klymene | 25.8 | 10.2 | 36.0 |
| Themis -> Hygiea -> Klymene | 30.2 | 6.0 | 36.2 |
| Massalia -> Peraga -> Klymene | 32.1 | 4.0 | 36.1 |

Note: The exhaustive optimizer results in `asteroid_data_0_0_0.mat` (and M=1,0,0, M=0,1,0) should produce better paths than the greedy, but the `find_best_path.m` results are not yet exported to CSV.

The 22.6 km/s best greedy result is the **total impulsive delta-V excluding Earth departure**. This is the delta-V the spacecraft propulsion system must provide.

### 4.2 Propulsion Trade Study

#### Option A: Chemical Bipropellant (MMH/NTO)

- **Engine**: Aerojet R-4D-11, thrust = 490 N, Isp = 312 s
- **Heritage**: NEAR Shoemaker, Mars Odyssey, many GEO satellites
- **Mass ratio for 22.6 km/s**: m0/mf = exp(22.6 / (0.312 * 9.81)) = exp(7.39) = 1,615 -- **completely infeasible** (99.94% propellant)
- **Mass ratio for 10 km/s**: exp(3.27) = 26.3 -- still infeasible (96.2% propellant)
- **Mass ratio for 5 km/s**: exp(1.63) = 5.12 -- marginal (80.5% propellant)
- **Assessment**: Chemical propulsion cannot deliver the required delta-V for a 3-asteroid rendezvous. Even with extensive gravity assists (Rosetta used 4 to reach a comet with only 2.3 km/s onboard), achieving 3 rendezvous is not feasible with chemical alone.

#### Option B: Solar Electric Propulsion (Ion Engines)

- **Engine**: NSTAR (Dawn heritage), thrust = 19-91 mN, Isp = 3,100 s
- **Mass ratio for 11 km/s**: exp(11 / (3.1 * 9.81)) = exp(0.362) = 1.44 -- **highly feasible** (30.3% propellant)
- **Heritage**: Dawn (11.5 km/s total delta-V, 425 kg xenon, 747 kg dry mass)
- **Assessment**: Ion propulsion is the enabling technology for multi-asteroid rendezvous, exactly as Dawn demonstrated. The trade-off is slower transfers (continuous low-thrust spirals vs. impulsive burns) and the need for large solar arrays at 3+ AU.

#### Option C: Solar Electric Propulsion (Hall Thrusters)

- **Engine**: SPT-140 (Psyche heritage), thrust = 240 mN, Isp = 1,800-2,700 s
- **Mass ratio for 10 km/s at Isp=2000s**: exp(10 / (2.0 * 9.81)) = exp(0.51) = 1.67 -- feasible (40% propellant)
- **Heritage**: Psyche mission (6.7 km/s total, 1,085 kg xenon, 1,648 kg dry mass)
- **Assessment**: Higher thrust than NSTAR enables faster maneuvers, but lower Isp means more propellant. Still far superior to chemical.

#### Option D: Hybrid (SEP cruise + Chemical for proximity ops)

- **Primary**: SEP for all interplanetary transfers and orbit insertion spirals
- **Secondary**: Small hydrazine system for attitude control, momentum dumping, and emergency maneuvers
- **Heritage**: This is exactly what Dawn flew (3 NSTAR + 12 hydrazine 0.9N thrusters)
- **Assessment**: Recommended approach. The hybrid architecture provides the high-Isp cruise efficiency of SEP with the agility of chemical thrusters for proximity operations.

### 4.3 Recommended Delta-V Budget

Assuming SEP with Mars gravity assist (analogous to Dawn):

| Mission Phase | Delta-V (km/s) | Source |
|---|---|---|
| Earth departure (launch vehicle provides C3) | -- | C3 ~ 10-15 km^2/s^2 |
| Earth to Mars (ion thrust) | 1.5-2.0 | Spiral out + plane change |
| Mars gravity assist | -1.5 to -2.5 (free) | Flyby, 200 km min altitude |
| Mars to Asteroid 1 (ion thrust) | 2.0-3.5 | Transfer + spiral capture |
| Asteroid 1 orbit operations | 0.1-0.3 | Orbit changes (survey/HAMO/LAMO) |
| Asteroid 1 to Asteroid 2 (ion thrust) | 2.0-4.0 | Spiral escape + transfer + capture |
| Asteroid 2 orbit operations | 0.1-0.3 | Same |
| Asteroid 2 to Asteroid 3 (ion thrust) | 2.0-4.0 | Same |
| Asteroid 3 orbit operations | 0.1-0.3 | Same |
| Margin (10%) | 1.0-1.5 | Standard |
| **Total spacecraft delta-V** | **~9-16 km/s** | **Within SEP capability** |

**Comparison to Dawn**: Dawn delivered 11.5 km/s total with 425 kg xenon and 747 kg dry mass. Our mission requires a similar or slightly larger delta-V, suggesting a Dawn-class spacecraft is appropriate.

### 4.4 Impact on Lambert Solver Results

The current codebase computes impulsive delta-V via Lambert arcs. For SEP, the actual trajectory is a continuous low-thrust spiral, not an impulsive transfer. However, **impulsive Lambert solutions remain valuable for:**

1. **Screening**: Ranking asteroid triplets by impulsive delta-V correctly identifies the most accessible combinations, even if the absolute values are higher than the SEP solution.
2. **Phasing**: The optimal launch windows and transfer timing from Lambert analysis are approximately correct for SEP.
3. **Lower bound**: The impulsive delta-V provides a lower bound on the total velocity change needed (SEP requires slightly more delta-V due to gravity losses during long thrust arcs, but this is offset by the higher Isp).

For detailed SEP trajectory design, tools like JPL's MALTO (Mission Analysis Low-Thrust Optimization) or ESA's `pykep` library would be needed. These are beyond the current MATLAB codebase but could be a future extension.

### 4.5 Propellant Mass Estimate

Using the Tsiolkovsky rocket equation with the recommended SEP system:

```
m_propellant = m_dry * (exp(delta_V / (Isp * g0)) - 1)
```

| Parameter | Value |
|---|---|
| Target delta-V | 12 km/s (with margin) |
| Isp (NSTAR) | 3,100 s |
| Exhaust velocity | 30.4 km/s |
| Mass ratio | exp(12/30.4) = 1.485 |
| Assumed dry mass | 750 kg (Dawn-class) |
| Propellant mass | 750 * 0.485 = **364 kg xenon** |
| Launch mass | 750 + 364 = **~1,114 kg** |

This is close to Dawn's actual values (425 kg Xe, 1,218 kg launch), providing confidence in the estimate.

---

## 5. Spacecraft Subsystem Design

### 5.1 Propulsion System

**Primary: NSTAR Ion Engine (Dawn heritage)**

| Parameter | Value |
|---|---|
| Number of engines | 3 (1 operating, 2 redundant) |
| Thrust per engine | 19-91 mN (throttleable) |
| Specific impulse | 3,100 s |
| Input power per engine | 0.5-2.3 kW |
| Engine mass | 8.9 kg each |
| Propellant | Xenon gas |
| PPU (Power Processing Unit) mass | ~14 kg each |
| Total propulsion dry mass | ~80 kg (engines + PPU + tankage + plumbing) |
| Xenon propellant | ~400 kg |
| Xenon tank | Composite overwrap, ~30 kg |

**Secondary: Hydrazine RCS**

| Parameter | Value |
|---|---|
| Number of thrusters | 12 (0.9 N each) |
| Isp | 220 s |
| Propellant | 46 kg hydrazine |
| Purpose | Attitude control, momentum dumping, emergency maneuvers |

### 5.2 Power System

**Solar Arrays (Dawn heritage)**

At 3.15 AU (worst case, Hygiea/Klymene distance), solar flux = 1361 / 3.15^2 = 137 W/m^2.

| Parameter | Value |
|---|---|
| Cell type | Triple-junction GaAs (InGaP/GaAs/Ge) |
| Cell efficiency (BOL) | 30% |
| EOL degradation | 15% (10+ year mission) |
| Power per m^2 at 3.15 AU | 137 * 0.30 * 0.85 = 35.0 W/m^2 |
| Required power at 3.15 AU | ~1,500 W (one ion engine at reduced power + bus) |
| Array area needed | 1500 / 35.0 = **43 m^2** |
| Array configuration | Two wings, each ~5.2 m x 4.2 m |
| Array mass | ~1.5 kg/m^2 * 43 = **65 kg** |
| Power at 1 AU | 43 * 1361 * 0.30 * 0.85 = **14.9 kW** |

Note: If operating the ion engine at full power (~2.3 kW), larger arrays (~85 m^2) would be needed, but reduced-power thrusting at 3 AU is the Dawn operations paradigm.

**Energy Storage**

| Parameter | Value |
|---|---|
| Battery type | Li-ion (modern) or NiH2 (Dawn heritage) |
| Capacity | 35 Ah |
| Mass | ~15 kg |
| Purpose | Eclipse operations, peak loads, safe mode |

### 5.3 Thermal Control

At 2.4-3.15 AU, the thermal environment is cold. Solar flux ranges from 218 W/m^2 (2.5 AU) to 137 W/m^2 (3.15 AU).

| Component | Function | Mass (kg) | Power (W) |
|---|---|---|---|
| MLI blankets (10-30 layers) | Insulation | ~15 | 0 (passive) |
| Electric heaters (Kapton film) | Keep components above min temps | ~2 | 100-200 at 3 AU |
| Ammonia heat pipes | Transport electronics waste heat | ~3 | 0 (passive) |
| Louvers (bi-metallic) | Variable heat rejection near Sun | ~4 | 0 (passive) |
| IR detector cryocooler | Cool VIR HgCdTe to ~70 K | ~1.3 | 12.6 |
| **Total thermal** | | **~25 kg** | **~115-215 W** |

**Key thermal requirements**:
- Electronics boxes: -20 to +50 C operating
- Xenon tanks: above -112 C (Xe freezing point)
- Hydrazine lines: above +2 C
- VIR IR detector: ~70 K (dedicated cryocooler)
- Battery: 0 to +30 C

### 5.4 Attitude Determination and Control (ADCS)

| Component | Qty | Unit Mass (kg) | Unit Power (W) | Performance |
|---|---|---|---|---|
| Star tracker (DTU Micro-ASC) | 2 CHU + 1 DPU | 0.5 + 1.5 | 1.9 | Arcsecond accuracy |
| Sun sensor (digital) | 12 | 0.1 | 0.1 | ~0.5 deg, coarse safe mode |
| IMU / IRU (2-axis) | 3 | 2.5 | 10 | Rate sensing, 0.01 deg/hr drift |
| Reaction wheels (Honeywell HR14-50) | 4 | 7.5 | ~50 (total) | 50 Nms storage, 0.2 Nm torque |
| **Total ADCS** | | **~45 kg** | **~75 W peak** | |

Reaction wheels in pyramid (4-wheel) configuration provide 3-axis control with single-fault tolerance. Hydrazine RCS thrusters dump momentum when wheels approach saturation.

**Pointing accuracy**: ~1 mrad (0.057 deg) meets the VIR imaging spectrometer IFOV of ~250 urad with comfortable margin.

### 5.5 Telecommunications

**Link budget at 3 AU (X-band, 8.4 GHz)**:

| Parameter | Value |
|---|---|
| HGA diameter | 1.52 m parabolic (Dawn heritage) |
| HGA gain | ~39.6 dBi |
| Transmitter | 35 W RF (TWTA) |
| EIRP | 55.0 dBW |
| Free space path loss at 3 AU | ~282 dB |
| DSN 34m antenna gain | ~68.3 dBi |
| Achievable data rate | **10-120 kbps** |
| Daily downlink (8 hr/day) | **36-345 Mb/day** |

| Component | Mass (kg) | Power (W) |
|---|---|---|
| HGA dish + feed | 8 | -- |
| SDST transponder | 3 | 15 |
| TWTA | 2.5 | 70 (during TX) |
| LGA (omni, backup) | 0.5 | -- |
| Cabling/waveguide | 2 | -- |
| **Total telecom** | **~16 kg** | **~85 W (TX)** |

**Operational constraint**: During ion thrusting, the HGA cannot point at Earth (spacecraft is oriented for thrust). Downlink occurs during coast arcs or dedicated comm windows. Dawn used alternating thrust and comm periods. An articulated HGA could enable simultaneous thrust + downlink but adds complexity and mass.

### 5.6 Command and Data Handling (C&DH)

| Component | Mass (kg) | Power (W) |
|---|---|---|
| RAD750 flight computer | 4 | 10 |
| Solid-state recorder (128 Gb) | 3 | 5 |
| Interfaces / harness | 3 | 5 |
| **Total C&DH** | **~10 kg** | **~20 W** |

### 5.7 Instrument Suite

#### Visible/IR Imaging Spectrometer (VIR, Dawn heritage)

| Parameter | Value |
|---|---|
| Heritage | Dawn VIR / Rosetta VIRTIS (ASI/INAF Italy) |
| Spectral range | 0.25-1.05 um (CCD) + 0.95-5.1 um (HgCdTe) |
| Spectral pixels | 432 |
| Spatial pixels | 256 per frame |
| IFOV | ~250 urad/pixel |
| Slit FOV | 64 mrad |
| Optical design | Shafer telescope + Offner spectrometer |
| Mass | 14.3 kg |
| Power | 17.6 W + 12.6 W cryocooler = **30.2 W** |
| TRL | 9 (flown on Dawn and Rosetta) |
| Science | Mineral identification (pyroxene/olivine 1-2 um bands), hydration (2.7 um OH), organics, thermal mapping |

#### Framing Camera (Dawn heritage)

| Parameter | Value |
|---|---|
| Heritage | Dawn FC (DLR/MPS Germany) |
| Number of units | 2 (redundant) |
| Detector | 1024 x 1024 frame-transfer CCD |
| FOV | 5.5 x 5.5 deg |
| IFOV | 93.7 urad/pixel |
| Filters | 8-position wheel: 1 clear (450-950 nm) + 7 narrow-band (430-980 nm) |
| Resolution | 9.3 m/pixel at 100 km range |
| Mass | 5.5 kg each, 11 kg total |
| Power | ~14 W each, ~14 W operating (one at a time) |
| TRL | 9 |
| Science | Surface morphology, cratering record, color variations, stereo topography |

#### Magnetometer (Psyche / Rosetta heritage)

| Parameter | Value |
|---|---|
| Heritage | Psyche MAG (MIT/DTU), Rosetta RPC-MAG |
| Type | Dual three-axis fluxgate (gradiometer) |
| Configuration | Two sensors on 2.15 m deployable boom |
| Dynamic range | 0.2 nT to 80,000 nT (multiple ranges) |
| Sensitivity | ~7 pT/sqrt(Hz) at 0.1-1 Hz |
| Sample rate | Up to 50 Hz |
| Mass | ~3 kg (sensors + electronics + boom) |
| Power | ~3 W |
| TRL | 9 |
| Science | Remanent magnetization (differentiation history), solar wind interaction, magnetic anomalies |

#### Instrument Summary

| Instrument | Mass (kg) | Power (W) | Data Rate |
|---|---|---|---|
| VIR spectrometer | 14.3 | 30.2 | ~2 Mbps (imaging) |
| Framing Camera (x2) | 11.0 | 14 | ~1 Mbps |
| Magnetometer | 3.0 | 3 | ~10 kbps |
| Instrument DPU / electronics | 3.0 | 10 | -- |
| **Total payload** | **~31 kg** | **~57 W** | -- |

### 5.8 Structure

| Parameter | Value |
|---|---|
| Primary structure | Central graphite composite cylinder |
| Equipment panels | Aluminum honeycomb with aluminum facesheets |
| Propellant tanks | Titanium (Xe) + titanium diaphragm (hydrazine) |
| Structural mass | ~130 kg (16-20% of dry mass) |
| Min lateral frequency | > 15 Hz |
| Min axial frequency | > 35 Hz |

### 5.9 Mass Budget Summary

| Subsystem | Mass (kg) | % of Dry |
|---|---|---|
| Structure | 130 | 17% |
| Propulsion (SEP, dry) | 80 | 10% |
| Propulsion (RCS, dry) | 15 | 2% |
| Power (arrays + battery + PCDU) | 100 | 13% |
| ADCS | 45 | 6% |
| Thermal | 25 | 3% |
| Telecom | 16 | 2% |
| C&DH | 10 | 1% |
| Instruments | 31 | 4% |
| Harness/cabling | 48 | 6% |
| **Dry mass subtotal** | **500** | **65%** |
| **Margin (30%)** | **150** | |
| **Dry mass with margin** | **650** | |
| Xenon propellant | 400 | |
| Hydrazine propellant | 46 | |
| **Launch mass** | **~1,100 kg** | |

**Launch vehicle**: A Falcon 9 or Atlas V 401 can deliver ~1,100 kg at C3 = 10-15 km^2/s^2, which is well within capability.

---

## 6. Mission Operations Plan

### 6.1 Mission Timeline

Assuming a Hertha(M) -> Massalia(S) -> Hygiea(C) trajectory with Mars gravity assist (the exact dates depend on exhaustive optimizer results with composition constraint):

| Phase | Duration | Cumulative Time | Key Activities |
|---|---|---|---|
| Launch (2029-2031) | -- | T+0 | Launch on Falcon 9, C3 ~ 12 km^2/s^2 |
| Earth-Mars cruise | 6-12 months | T+0.5-1 yr | Ion thrusting, trajectory correction, instrument checkout |
| Mars gravity assist | 1 day | T+1 yr | Flyby at 200+ km altitude, trajectory deflection toward belt |
| Mars-to-Hertha cruise | 1.5-2.5 years | T+2.5-3.5 yr | Continuous ion thrusting |
| **Hertha approach & orbit** | **5 months** | T+3-4 yr | Survey, HAMO, LAMO (see 6.2) |
| Hertha-to-Massalia transfer | 1-2 years | T+4-6 yr | Ion thrust spiral escape + transfer + capture |
| **Massalia approach & orbit** | **5 months** | T+5-7 yr | Survey, HAMO, LAMO |
| Massalia-to-Hygiea transfer | 1.5-3 years | T+7-10 yr | Ion thrust transfer |
| **Hygiea approach & orbit** | **5 months** | T+8-11 yr | Survey, HAMO, LAMO |
| End of mission | -- | T+8-11 yr | Spacecraft decommissioned in orbit at Hygiea |
| **Total mission duration** | | **~8-11 years** | |

### 6.2 Operations at Each Asteroid

Based on Dawn's Vesta/Ceres operations (Russell et al., 2012; Rayman & Mase, 2014):

**Phase 1: Approach (8-12 weeks before orbit insertion)**
- Begin optical navigation imaging at ~1.2 million km range
- Framing Camera acquires target as point source, refines ephemeris
- Search for moons, rings, or debris hazards
- Build initial rotation light curve and shape model
- Ion thruster gradually matches asteroid velocity (spiral approach)
- Navigation transitions from ground-based to onboard OpNav

**Phase 2: Survey Orbit (3-4 weeks)**
- Enter high-altitude polar orbit (~10x asteroid radius)
  - Hertha: ~400 km; Massalia: ~700 km; Hygiea: ~2,000 km
- 5-pass polar survey: nadir camera + spectrometer mapping
- Determine mass (radio science Doppler), spin pole, global shape
- Characterize gravity field to plan lower orbits
- Magnetometer baseline measurements of solar wind at distance

**Phase 3: High-Altitude Mapping Orbit (HAMO, 6-8 weeks)**
- Lower orbit to ~3x asteroid radius
  - Hertha: ~120 km; Massalia: ~220 km; Hygiea: ~650 km
- Systematic mapping in 10-orbit cycles: nadir imaging, stereo for topography
- VIR composition mapping: mineral identification via 1/2 um bands
- Color imaging with FC filter wheel
- Magnetometer measures crustal magnetic field at intermediate range
- Downlink imaging data during coast arcs between orbit change maneuvers

**Phase 4: Low-Altitude Mapping Orbit (LAMO, 6-8 weeks)**
- Lowest stable orbit ~1.5-2x asteroid radius
  - Hertha: ~80 km; Massalia: ~150 km; Hygiea: ~430 km
- High-resolution imaging (FC achieves ~7-15 m/pixel)
- Detailed VIR spectral mapping of regions of interest
- Gravity field determination via precise Doppler tracking
- Magnetometer detects crustal remanent magnetization at close range
- Radio science for internal structure constraints

**Phase 5: Departure (2-4 weeks)**
- Spiral orbit raising via ion propulsion
- Final departure observations (phase angle coverage)
- Begin cruise to next target

### 6.3 Data Management

| Parameter | Value |
|---|---|
| Data rate at 3 AU | 10-120 kbps |
| Daily DSN contact | ~8 hours (34m antenna) |
| Daily downlink volume | ~36-345 Mb/day |
| Onboard storage | 128 Gb |
| Data compression | ICER (lossless/lossy, 4:1 to 16:1) |
| Priority scheme | Magnetometer (continuous, low rate) > camera thumbnails > full-res spectra |

### 6.4 Mission Phases and Power Modes

| Mode | Power Draw (W) | Duration |
|---|---|---|
| Cruise (ion thrusting) | ~1,500-2,300 | Years |
| Cruise (coast/downlink) | ~300 | Hours-days |
| Science (mapping orbit) | ~400 | Months |
| Safe mode | ~150 | Contingency |

---

## 7. Risk Assessment

### 7.1 Risk Matrix

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| **Ion thruster failure** | Medium | High | Carry 3 thrusters (Dawn used single-string operation after wheel failures); design for 2-thruster mission completion |
| **Reaction wheel failure** | Medium | High | 4 wheels (3+1 redundant); RCS thrusters as backup for wheel desaturation; Dawn lost 2 of 4 wheels and continued |
| **Solar array degradation beyond prediction** | Low | High | Conservative degradation model (15% EOL); can reduce ion thrust power |
| **Lambert solver convergence issues** | Medium | Low | Two-tier solver (Izzo + Gooding); penalty function (1e3 km/s) ensures graceful handling; solver is robust for main belt arcs |
| **Unfavorable launch window** | Low | Medium | 8-year launch window (2027-2035) provides multiple opportunities; Mars GA windows repeat every ~26 months |
| **Communication outages at 3 AU** | Low | Medium | Onboard autonomy for thrust arcs; 128 Gb storage for science data buffering; LGA for emergency commands |
| **Thermal system heater failure** | Low | High | Redundant heater circuits; MLI passive insulation provides baseline protection; critical components (hydrazine, battery) have dual heaters |
| **Xenon propellant leak** | Low | Critical | Welded propellant system; pressure monitoring; delta-V margin (10%) covers small leaks |
| **Asteroid gravity field uncertainty** | Medium | Low | Survey orbit characterizes gravity before committing to low orbits; autonomous orbit maintenance |
| **Composition constraint yields poor trajectories** | Medium | Medium | Expand notable asteroid pool beyond 11; consider including ~20-30 candidates from tradeoff table |

### 7.2 Key Design Margins

| Parameter | Nominal | Margin | With Margin |
|---|---|---|---|
| Delta-V | 10.5 km/s | +10% | 11.6 km/s |
| Xenon propellant | 364 kg | +10% | 400 kg |
| Dry mass | 500 kg | +30% | 650 kg |
| Power at 3.15 AU | 1,200 W needed | +25% | 1,500 W generated |
| Data downlink | 36 Mb/day minimum | -- | 345 Mb/day maximum |

---

## 8. Key References

### Trajectory Design and Orbital Mechanics

1. Izzo, D. (2015). "Revisiting Lambert's problem." *Celestial Mechanics and Dynamical Astronomy*, 121(1):1-15.
2. Lancaster, E.R. & Blanchard, R.C. (1969). "A unified form of Lambert's theorem." NASA TN D-5368.
3. Gooding, R.H. (1990). "A procedure for the solution of Lambert's orbital boundary-value problem." *Celestial Mechanics*, 48:145-165.
4. Battin, R.H. (1999). *An Introduction to the Methods of Astrodynamics*. AIAA Education Series.
5. Bate, R.R., Mueller, D.D., White, J.E. (1971). *Fundamentals of Astrodynamics*. Dover.
6. Curtis, H.D. (2014). *Orbital Mechanics for Engineering Students*, 3rd ed. Elsevier.
7. Chen, Y., Baoyin, H., Li, J. (2014). "Design and optimization of a trajectory visiting ten asteroids." *Advances in Space Research*, 53(4):697-710.
8. Englander, J.A., Conway, B.A., Williams, T. (2012). "Automated mission planning via evolutionary algorithms." *JGCD*, 35(6):1878-1887.

### Asteroid Science and Selection

9. DeMeo, F.E., Binzel, R.P., Slivan, S.M., Bus, S.J. (2009). "An extension of the Bus asteroid taxonomy into the near-infrared." *Icarus*, 202, 160-180.
10. Carry, B. (2012). "Density of asteroids." *Planetary & Space Science*, 73, 98-118.
11. Taylor, C.R., Seager, S., et al. (2018). "A Delta-V map of the known Main Belt Asteroids." *Acta Astronautica*, 146, 73-82.
12. Bonilla de la Corte, J., et al. (2025). "Fuzzy Multi-Criteria Decision Framework for Asteroid Selection in Boulder Capture Missions." *Aerospace*, 12(9), 800.

### Mission Heritage

13. Rayman, M.D., et al. (2006). "Dawn: A mission in development for exploration of main belt asteroids Vesta and Ceres." *Acta Astronautica*, 58(11):605-616.
14. Russell, C.T. & Raymond, C.A. (2011). "The Dawn Mission to Vesta and Ceres." *Space Science Reviews*, 163.
15. Levison, H.F., et al. (2021). "Lucy: Surveying the Diversity of the Trojan Asteroids." *Planetary Science Journal*, 2:171.
16. Lauretta, D.S., et al. (2017). "OSIRIS-REx: Sample Return from Asteroid (101955) Bennu." *Space Science Reviews*, 212(1-2):925-984.
17. Elkins-Tanton, L.T., et al. (2020). "Observations, Meteorites, and Models: A Preflight Assessment of the Composition and Formation of (16) Psyche." *JGR: Planets*, 125.
18. Glassmeier, K.-H., et al. (2007). "The Rosetta Mission: Flying Towards the Origin of the Solar System." *Space Science Reviews*, 128.
19. Tsuda, Y., et al. (2013). "System Design of the Hayabusa 2." *Acta Astronautica*, 91.

### SPICE and Ephemerides

20. Acton, C.H. (1996). "Ancillary data services of NASA's Navigation and Ancillary Information Facility." *Planetary & Space Science*, 44(1):65-70.
21. Acton, C.H., Bachman, N., Semenov, B., Wright, E. (2018). "A look towards the future in the handling of space science mission geometry." *Planetary & Space Science*, 150:9-12.

### Spacecraft Engineering

22. Wertz, J.R., Everett, D.F., Puschell, J.J. (2011). *Space Mission Engineering: The New SMAD*. Microcosm Press.
23. Larson, W.J. & Wertz, J.R. (1999). *Space Mission Analysis and Design*, 3rd ed. Microcosm/Kluwer.

---

## Appendix A: Execution Checklist

### Phase 1: Complete Trajectory Optimization (Operations Goals 1 & 2)

- [ ] **Add composition constraint to exhaustive optimizer**: Modify `GENERATE_OPTIMIZED_DATA.m` to skip triplets where the three asteroids are not from three different composition groups (C, S, X/M). This reduces 990 triplets to ~200, cutting runtime by ~80%.
- [ ] **Run exhaustive optimizer for remaining M combinations**: Currently have (0,0,0), (1,0,0), (0,1,0). Complete with (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1) to fully explore multi-revolution solutions.
- [ ] **Export exhaustive results**: Adapt `find_best_path.m` to output the top 10 composition-diverse paths to a CSV file for easy comparison.
- [ ] **Generate porkchop plots**: For the best 3-5 triplets, create 2D contour plots of C3 and arrival V-infinity for each leg. These are critical for the project report.
- [ ] **Run Mars gravity assist variant**: Execute `mars_transfer_selector.m` for the top composition-diverse triplets to compare direct vs. Mars-assist delta-V.
- [ ] **Select final asteroid triplet and trajectory**: Choose based on minimum total delta-V with full C+S+X diversity.

### Phase 2: Spacecraft Design

- [ ] **Size propulsion system**: Based on final delta-V, size the SEP system (number of thrusters, xenon mass, tank volume).
- [ ] **Size solar arrays**: Calculate required area for worst-case heliocentric distance, accounting for ion thruster power + bus load.
- [ ] **Size thermal system**: Estimate heater power at max distance, MLI coverage, cryocooler for VIR.
- [ ] **Size telecom system**: Link budget for worst-case distance, determine HGA size and transmitter power.
- [ ] **Complete mass budget**: Roll up all subsystem masses with 30% margin on dry mass.
- [ ] **Create CAD model**: Basic 3D model for structural mode analysis (lateral > 15 Hz, axial > 35 Hz).
- [ ] **Simple thermal model**: Verify operability at min/max heliocentric distance.

### Phase 3: Mission Operations and Documentation

- [ ] **Write ConOps**: Detailed concept of operations for each mission phase.
- [ ] **Create mission timeline**: Gantt chart of all phases from launch to end of mission.
- [ ] **Compile risk list**: Expand risk matrix with probability x consequence scores.
- [ ] **Generate trajectory visualizations**: Use `FLIGHTPATH_ANIMATION.m` for the selected path.
