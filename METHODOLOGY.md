# Methodology — Algorithm Reference

This document is the technical algorithm reference for the asteroid trajectory
optimization pipeline. For setup and step-by-step "how to run X" instructions,
see [`README.md`](README.md) and the [`Tutorials/`](Tutorials/) folder.
Historical design documents are in [`docs/archive/`](docs/archive/).

---

## Table of Contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Stage 1 — Asteroid screening (SBDB)](#2-stage-1--asteroid-screening-sbdb)
3. [Stage 2 — Multi-criteria scoring](#3-stage-2--multi-criteria-scoring)
4. [Stage 3 — Trajectory model](#4-stage-3--trajectory-model)
5. [Stage 4 — Sequence selection](#5-stage-4--sequence-selection)
6. [Gravity assists (Moon, Mars, Earth)](#6-gravity-assists-moon-mars-earth)
7. [Low-thrust framework (Sims-Flanagan)](#7-low-thrust-framework-sims-flanagan)
8. [Mass-Pareto optimization](#8-mass-pareto-optimization)
9. [Physical-feasibility audit](#9-physical-feasibility-audit)
10. [References](#10-references)

---

## 1. Pipeline overview

The pipeline narrows ~10⁶ candidate three-asteroid missions to one ranked
deliverable through four stages:

```
JPL SBDB → 406 candidates → 50 ranked → 50³ triplets → top-N optimized → top-3 audited
   Stage 1       Stage 2          Stage 3                  Stage 4         Audit
```

| Stage | Tool | Input | Output |
|---|---|---|---|
| 1 | JPL SBDB query | All known asteroids | 406 main-belt candidates |
| 2 | `tradeoff.py` | 406 candidates + science weights | `asteroid_tradeoff.csv` (ranked top ~50) |
| 3 | `optimization.py` | 50 candidates + launch window | Per-triplet Δv via `differential_evolution` |
| 4 | `optimization.py` | All N×(N-1)×(N-2) triplets | Top-K best paths |
| Audit | `core.audit_flyby_geometry` | Saved trajectories | Pass/fail flyby physics |

All stages use **km, km/s, km³/s²** as internal units. SI conversions happen
at the pykep boundary. Reference frame: ECLIPJ2000, observer body `10` (Sun
barycenter), aberration `NONE`.

---

## 2. Stage 1 — Asteroid screening (SBDB)

Source: NASA JPL Small-Body Database (SBDB) bulk export.

Filters applied:
- 2.0 AU < a < 3.5 AU (main belt)
- Diameter > 30 km (estimated from H magnitude when not directly measured)
- Known SMASS-II / Bus-DeMeo taxonomy code
- Has SPK ephemeris available from JPL Horizons covering 2025–2050

When diameter isn't directly measured, the H-magnitude formula gives:

```
D [km] = 1329 / sqrt(p_v) · 10^(-H/5)
```

with default geometric albedo `p_v = 0.07` for C-types and `0.20` for S-types.

Output: 406 candidates, dumped to `sbdb_query_results.csv`.

---

## 3. Stage 2 — Multi-criteria scoring

Implemented in `Python_Consolidated/tradeoff.py` (function `run_tradeoff_v3`).

Each asteroid is scored 1–10 on eight criteria, then ranked by weighted
total. Weights:

| Criterion | Weight | Direction | Why |
|---|---|---|---|
| Δv accessibility (proxy) | 30% | Lower better | Dominates mission feasibility |
| Orbital inclination | 20% | Lower better | Plane changes are expensive |
| Science potential | 14% | Higher better | Mission science return |
| Estimated mass | 12% | Higher better | Larger = richer geology |
| Radius | 12% | Higher better | More surface to map |
| Eccentricity | 6% | Lower better | High-e raises rendezvous cost |
| Rotation period | 4% | 6–24h optimal | Operations / imaging constraint |
| Semi-major axis | 2% | Lower better | Closer = shorter transfer |

The **science potential** sub-score combines:
- **Characterization depth** (40%): how well the body is already known
  (taxonomy, measured diameter, albedo, rotation, color)
- **Intrinsic interest** (60%): aqueous alteration features (+2), active
  main-belt comet (+3), surface ice detection (+3), ambiguous radar
  albedo (+2), penalty for previously visited targets (-3)

The Δv accessibility proxy at this stage is a placeholder (analytical
Hohmann-transfer estimate) since real trajectory analysis hasn't run yet.
Stage 3 supersedes it.

Output: `asteroid_tradeoff.csv` with the top ~50 candidates promoted to Stage 3.

---

## 4. Stage 3 — Trajectory model

Implemented in `Python_Consolidated/optimization.py` (`compute_path_deltav`,
`compute_path_with_flyby`, `score_paths_flyby`).

For each candidate triplet (A1, A2, A3), the optimizer searches a 6-D (direct)
or 7-D (flyby) decision space.

### Decision variables (Mars/Moon-flyby case, 7-D)

| Variable | Physical meaning | Bounds |
|---|---|---|
| x1 | Launch date offset from window start | 0 – 9 yr |
| x2 | Earth → flyby-body transfer time | 0.3 – 3 yr (Mars), 1 – 10 day (Moon) |
| x3 | Flyby → A1 transfer time | 2 wk – 5 yr |
| x4 | Stay duration at A1 | 3 mo – 1 yr |
| x5 | A1 → A2 transfer time | 2 wk – 5 yr |
| x6 | Stay duration at A2 | 3 mo – 1 yr |
| x7 | A2 → A3 transfer time | 2 wk – 5 yr |

Hard constraint: total mission < 14 yr (BSP coverage ends ~2050).

### Objective function (per evaluation)

1. Convert decision vector to absolute SPICE epochs.
2. Look up heliocentric states (r, v) for Earth, the flyby body, and all three
   asteroids at the relevant epochs via `spiceypy.spkezr` in ECLIPJ2000.
3. Solve **Lambert's problem** for each transfer leg using `pk.lambert_problem`
   (Izzo's algorithm, 2015). Try multi-revolution branches m = 0, 1, 2 in both
   directions; keep the lowest-Δv branch.
4. Compute powered Δv at the gravity-assist periapsis via the patched
   `compute_flyby_dv` (see §6 and §9).
5. Sum the magnitudes of all velocity-mismatch vectors at the six maneuver
   points: Earth departure, flyby periapsis, A1 arrival, A1 departure,
   A2 arrival, A2 departure, A3 arrival.

Penalty value `1e3 km/s` returned for any infeasible state (Lambert
non-convergence, mission > 14 yr, infeasible flyby geometry).

### Optimizer

`scipy.optimize.differential_evolution` with:
- maxiter = 200–300
- popsize = 15–18
- mutation = (0.5, 1.3)
- recombination = 0.8
- updating = `'deferred'` (parallelizable)
- `polish=True` (final L-BFGS-B refinement)

DE replaced an earlier 180-point Chebyshev grid because DE explores the 7-D
search space more thoroughly and finds better global minima with fewer total
function evaluations.

Multi-start: `optimize_for_architecture` runs DE with 3 seeds (42, 137, 314)
and 4 Lambert m-revolution combos, keeping the best-of-12.

---

## 5. Stage 4 — Sequence selection

Three algorithms, in `optimization.py`:

### Brute force — `generate_optimized_data`
Run full DE on every triplet. Feasible for N < 15 (~2,730 triplets). Slow
but exhaustive.

### Two-level — `two_level_optimize` (recommended)
1. **Coarse pass:** sampling-based pre-screen on every N×(N-1)×(N-2) triplet
   (~300,000 for N=69). ~5 min on 12 cores.
2. **Fine pass:** full DE on the top 50.

Speedup: ~10× over brute force. Negligible quality loss since the coarse
score correlates strongly with the DE-optimized score.

Optional flags:
- `science_scores` + `alpha`: blends Δv with science weight (e.g. α=0.7 ⇒
  70% Δv + 30% science)
- `comp_map` + `required_compositions`: restricts to triplets that span the
  required taxonomy classes (e.g. C+S+X/M)

### Beam search — `beam_search`
Build the path one leg at a time, keeping the top-K partial sequences. For
N=50 with K=10, evaluates ~1,500 single-leg problems instead of ~117,600 full
triplets. Fastest of the three; slight quality loss vs two-level.

---

## 6. Gravity assists (Moon, Mars, Earth)

`FLYBY_BODIES` table in `optimization.py`:

| Body | SPICE GM ID | SPICE radii ID | Min altitude | TOF window |
|---|:-:|:-:|:-:|:-:|
| Moon | 301 | 301 | 100 km | 1–10 days post-launch |
| Mars | 4   | 499 | 200 km | 0.3–3 years post-launch |
| Earth | 399 | 399 | 300 km | 1–3 years (EGA loop) |

(Note the Mars asymmetry: GM kernel uses BODY4_GM (system barycenter) but the
radii kernel only has BODY499_RADII (planet itself). Functions that need
both — like `audit_flyby_geometry` — use the explicit `mu_body` /
`radii_body` fields.)

### Flyby Δv computation — `core.compute_flyby_dv`

Wraps `pk.fb_dv` for the energy-matching radial-burn calculation, plus a
**geometric feasibility gate**:

```python
sin_in_max  = 1 / (1 + r_p · v_in² / μ)
sin_out_max = 1 / (1 + r_p · v_out² / μ)
δ_max       = arcsin(sin_in_max) + arcsin(sin_out_max)
if δ_required > δ_max + 1e-6:
    return 1e3  # km/s — penalty, optimizer steers away
```

This catches the case where the required turn angle exceeds the natural
maximum at the safe periapsis altitude — a bug `pk.fb_dv` doesn't check
because it only equalizes |v_inf| magnitudes. The patch was added after
auditing 12 of 15 mass-Pareto top results and finding all 12 had
geometrically impossible Mars flybys (turn angles 40°–165°, requiring
periapsis altitudes thousands of km below Mars's surface).

### Mars gravity assist
Adds a 7th decision variable (Earth → Mars transfer time, 0.3–3 yr) and
computes powered Δv at periapsis. Real bipropellant chemical engines deliver
this Δv in seconds (high thrust authority); electric thrusters cannot.

### Lunar gravity assist
Tighter window (1–10 days). Often used for "Moon Oberth maneuvers" — arrive
with 1 km/s v_inf, dive to 100 km altitude, fire chemical engine for 4–5 km/s
burn at periapsis (where speed is highest = Oberth efficiency boost),
depart with 7 km/s v_inf. Boundary cases sit right at the 100 km altitude
limit — mathematically valid, but practically tight.

### Earth gravity assist (EGA)
Long phasing loop (1–3 yr). Spacecraft launches with v_inf ≥ 3 km/s, returns
to Earth, and gets a velocity bend. Constraint: powered Δv at the periapsis
must be ≤ 0.3 km/s (otherwise it's not a "real" GA but a re-launch). EGA
is geometrically attractive when the launcher can't deliver the needed v_inf
directly.

---

## 7. Low-thrust framework (Sims-Flanagan)

Implemented in `Python_Consolidated/lowthrust.py`.

**Why low-thrust?** Chemical Δv at 320 s Isp requires ~95% propellant mass
for a 14 km/s mission. Electric thrust at 3100 s Isp lets a similar mission
fit in ~55% propellant — even though the integrated electric Δv is *higher*
than chemical (because slow burns lose efficiency to gravity), the ~10×
better Isp wins overall.

The choice isn't "which lowers Δv" but "which delivers more spacecraft mass
to the destination."

### Method: Sims-Flanagan direct transcription

Each heliocentric leg is split into N segments. For each segment:

1. Half-segment Kepler coast (`pk.propagate_lagrangian`)
2. Impulsive Δv kick `δv_i = u_i · T·Δt/m_i`
   where `u_i ∈ [-1,1]³` (unit ball) is the throttle vector
3. Half-segment Kepler coast
4. Mass updates per Tsiolkovsky: `m_{i+1} = m_i · exp(-|δv_i| / (Isp·g₀))`

Decision variables: 3·N throttle components per leg. Default N = 15
(15 segments per leg, 45 throttle variables).

### Solver

`scipy.optimize.least_squares` with method `'trf'` (trust region reflective).
Composite residual:

```
[pos_err / 1e6, vel_err · 10, reg_weight · throttles, thrust_penalty]
```

- `pos_err`: rendezvous position mismatch at the leg endpoint (km)
- `vel_err`: rendezvous velocity mismatch (km/s)
- `reg_weight · throttles`: L2 penalty to discourage unnecessary thrust
- `thrust_penalty`: smooth one-sided penalty when |u_i| > 1 (engine
  constraint violation)

Convergence: position error < 1.5×10⁶ km AND velocity error < 0.15 km/s.

Initial seed: Lambert solution split across segments, scaled to the unit-ball
constraint.

### Default spacecraft params

```
m_init  = 1500 kg     (Dawn-class)
thrust  = 0.30 N      (3 × NSTAR ion thruster cluster)
Isp_chem = 320 s      (bipropellant)
Isp_elec = 3100 s     (xenon-fed Hall-effect)
```

### Limitations
- Thrust direction is unconstrained (real engines have gimbal limits ±30°)
- No power-system model (assumes T = const for the full leg)
- Sun pointing not enforced (real solar-electric needs solar arrays facing
  the Sun, which constrains attitude)

These are conscious simplifications for the trajectory-optimization layer;
mission-design adds them downstream.

---

## 8. Mass-Pareto optimization

Implemented in `Python_Consolidated/mass_optimization.py`.

### Problem statement

For each triplet, find the (timing, propulsion architecture) pair that
maximizes delivered final spacecraft mass.

Architecture codes are 3-letter strings labeling the propulsion mode of the
three transfer legs (L2 = flyby→A1, L3 = A1→A2, L4 = A2→A3). L1 (Earth
launch + powered flyby periapsis) is always chemical.

```
C = chemical (Isp 320 s, impulsive)
E = electric (Isp 3100 s, continuous low-thrust)

8 architectures: CCC, CCE, CEC, CEE, ECC, ECE, EEC, EEE
```

### Joint optimization

The naive approach — optimize timing for chemical Δv, then layer LT on top —
gives suboptimal LT trajectories because timings tuned for chemical force
the LT solver into bad geometry.

`mass_optimization.optimize_for_architecture` runs DE *per architecture* with
the objective being delivered mass (Tsiolkovsky chain through 4 legs):

```
m_after_L1 = m_init · exp(-Δv_L1 / (Isp_chem · g₀))
m_after_L2 = m_after_L1 · exp(-Δv_L2_lambert · factor / (Isp_L2 · g₀))
m_after_L3 = m_after_L2 · exp(-Δv_L3_lambert · factor / (Isp_L3 · g₀))
m_final   = m_after_L3 · exp(-Δv_L4_lambert · factor / (Isp_L4 · g₀))
```

where `factor = 1.0` for chemical legs, and `factor = gravity_loss_factor(...)`
for electric legs (a calibrated surrogate, see below).

### Gravity-loss surrogate

We can't run the real Sims-Flanagan solver inside DE — too slow (~15 s/call,
DE wants 30,000 evals = 5 days/triplet). Instead use a closed-form correction
calibrated against Sims-Flanagan output:

```python
ratio = lambert_dv / (thrust · tof / m)   # how much of the thrust ceiling
                                            #  the leg needs

if ratio >= 0.95:      return ∞    # infeasible
factor = 1.20 + 0.55*ratio + 0.40 / (1-ratio) - 0.40
if tof_yr < 1.0:       factor *= (2.0 - tof_yr)   # short-cruise penalty
```

Validated against the Mars→Fortuna, Fortuna→Themis, Themis→Psyche legs from
`ftp_strict_electric_only.pkl`. Surrogate is ~25% optimistic vs. the real
solver — verification stage corrects this.

### Verification stage

`mass_optimization.verify_with_full_lt` re-evaluates the top architectures
per triplet using the real Sims-Flanagan solver on every electric leg. If
the LT solver fails to converge, the architecture is rejected and the
verifier falls back to the next-best.

### Δv equivalent

For ranking heterogeneous architectures, the optimizer emits a
*chemical-Isp-equivalent* Δv:

```
dv_eq = -Isp_chem · g₀ · ln(m_final / m_init) / 1000   (km/s)
```

Lower `dv_eq` ⇔ more delivered mass. This compresses the
mass-and-Δv trade into a single number for sorting. **It is not the
spacecraft's actual integrated Δv** — that's the sum of the per-leg
integrated Δvs (chemical Lambert + electric Sims-Flanagan).

---

## 9. Physical-feasibility audit

Implemented in `Python_Consolidated/core.py` (`audit_flyby_geometry`),
exposed via `main.py verify`.

**Why it exists.** The flyby-physics check inside `compute_flyby_dv` runs
during optimization, but old result pickles predate the patch. Independent
post-hoc audit catches them.

For a saved trajectory's Mars/Moon flyby:

1. Re-derive v_inf vectors from a fresh Lambert solve on either side of
   the flyby epoch.
2. Compute the required turn angle from `arccos(v_in · v_out / (|v_in||v_out|))`.
3. Compute the maximum natural turn at the safe periapsis altitude using
   the asymmetric formula:
   ```
   δ_max = arcsin(1/(1+r·v_in²/μ)) + arcsin(1/(1+r·v_out²/μ))
   ```
4. Pass if `δ_required ≤ δ_max + 1e-6`.

Outputs full v_inf vectors (heliocentric ECLIPJ2000), magnitudes, energy
residual, turn angle, max turn, and the periapsis altitude implied by the
geometry.

For powered-flyby cases (|v_inf_in| ≠ |v_inf_out|), the periapsis altitude
is computed using a ballistic-symmetric approximation that's exact at the
boundary (`δ_required == δ_max`) but underestimates altitude when the
asymmetry is large. The feasibility judgment is independent of this and
uses the exact asymmetric formula.

---

## 10. References

- Izzo, D. (2015). "Revisiting Lambert's problem." *Celestial Mechanics &
  Dynamical Astronomy*, 121(1):1–15.
- Storn, R. & Price, K. (1997). "Differential Evolution — a Simple and
  Efficient Heuristic for Global Optimization over Continuous Spaces."
  *Journal of Global Optimization*, 11:341–359.
- Carry, B. (2012). "Density of asteroids." *Planetary & Space Science*,
  73, 98–118.
- DeMeo, F.E. et al. (2009). "An extension of the Bus asteroid taxonomy
  into the near-infrared." *Icarus*, 202, 160–180.
- Sims, J.A. & Flanagan, S.N. (1999). "Preliminary Design of Low-Thrust
  Interplanetary Missions." AAS/AIAA Astrodynamics Specialist Conference.
- pykep (ESA): https://esa.github.io/pykep/
- SPICE / NAIF (NASA JPL): https://naif.jpl.nasa.gov/naif/
- JPL SBDB: https://ssd.jpl.nasa.gov/tools/sbdb_query.html

---

*Historical design documents (`plan.md`, `selection_and_optimization.md`,
`LOW_THRUST_PLAN.md`, `EARTH_GA_PLAN.md`, `MASS_DV_PARETO_PLAN.md`) are
preserved in [`docs/archive/`](docs/archive/).*
