# Low-Thrust & Earth-Gravity-Assist Extension Plan

> **New to this project?** Read [`README.md`](README.md) first — it has step-by-step setup and run instructions for someone with no Python experience.
> This document is the technical extension spec for the low-thrust framework.

**Status**: research complete, implementation pending. Target: reduce *propellant mass fraction* (not necessarily Δv) for the Earth → [flyby] → A1 → A2 → A3 mission.

Current best (impulsive, Mars GA): **9.40 km/s Δv**, Mars flyby, ~5-10 yr mission. At bipropellant Isp ≈ 320 s this implies a **~95% propellant mass fraction** (essentially impossible to launch).

Target with low-thrust + better flyby selection: **~33% propellant mass fraction** (2.5-5× lighter wet mass) by switching to electric propulsion (Isp ≈ 3000 s) and layering in Earth gravity assists.

---

## 1. Why the Δv number can be misleading

Impulsive design minimizes `Σ ||Δv_i||`. Low-thrust dynamics are `ṁ = -T/(Isp·g0)` and `a = T/m(t)`, so the analog of Δv is the integral `∫ |T|/m dt`. **This integral is almost always LARGER than the impulsive Δv for the same endpoints** — the spacecraft fights gravity continuously and loses Oberth benefit — yet propellant mass drops dramatically because Isp is 10× higher.

Tsiolkovsky, rearranged: `m_prop/m0 = 1 - exp(-Δv/(Isp·g0))`.

| Scenario | Δv | Isp | Prop fraction |
|---|---|---|---|
| Impulsive (chemical) | 9.4 km/s | 320 s | **95%** |
| Low-thrust (NSTAR-class) | 12 km/s | 3100 s | **33%** |
| Low-thrust + EGA | 10 km/s | 3100 s | **28%** |

**Deliverable for the trade study must be final mass (or propellant fraction), not Δv.** Dawn's 11 km/s mission was chemically impossible but cost 425 kg xenon on a 1218 kg wet spacecraft.

---

## 2. Gravity assist survey: Moon, Mars, AND Earth

The existing code has Moon + Mars flybys. We should **add Earth**, because for asteroid main-belt missions it is often the most valuable single assist.

### 2.1 Why add Earth gravity assist (EGA)

- **Free mass**: unlike Mars, Earth is at launch — you don't "spend" anything getting there the first time. The spacecraft launches with low C3, loops around the Sun for ~1-3 years (often with a deep-space maneuver halfway), returns for an EGA, and emerges with much higher C3 aimed at the asteroid belt.
- **Magnitude of the assist**: a single Earth flyby at 300 km altitude can rotate v∞ by up to ~60° and add or subtract several km/s of heliocentric velocity — far more than Moon (which adds ~1 km/s max) or Mars.
- **Flight heritage**: Dawn used one Mars GA + ion propulsion; MESSENGER used 2 Earth + 2 Venus + 3 Mercury flybys; Hayabusa2 used EGA to reach Ryugu; BepiColombo uses 1 Earth + 2 Venus + 6 Mercury flybys. Every modern asteroid / outer-belt mission chains EGAs.
- **ΔV-EGA trajectory family** (Hollister 1969, Farquhar 1985): launch → DSM (deep-space maneuver at aphelion) → 1-2 yr loop → Earth flyby → target. The DSM can be impulsive or replaced by continuous low-thrust.
- **Scales to 2 EGAs**: EGA1 + EGA2 separated by one synodic period (~1 yr) stacks the assist — MESSENGER pattern. Larger search space but big energy gain.

### 2.2 Implementation parameters (add to `FLYBY_BODIES` in `optimization.py:192`)

```python
'earth': {'id': '399', 'mu_body': 399, 'radii_body': 399, 'min_alt': 300,
          'tof_min': 1.0*YEAR, 'tof_max': 3.0*YEAR},   # EGA loop: 1-3 years
```

Note: the Moon `mu_body` bug (was `4`, should be `301`) is a cautionary tale — **add a unit test that asserts `get_mu(id)` returns the expected value for each flyby body**.

### 2.3 Architecture expansion

| Architecture | Legs | Notes |
|---|---|---|
| direct (current) | 3 | Earth → A1 → A2 → A3 |
| Moon GA (current) | 4 | Earth → Moon → A1 → A2 → A3 — small assist, mostly phasing |
| Mars GA (current) | 4 | Earth → Mars → A1 → A2 → A3 — current winner |
| **Earth GA + DSM** | 4 | Earth → DSM loop → Earth → A1 → A2 → A3 |
| **EGA + Mars GA** | 5 | Earth → DSM → Earth → Mars → A1 → A2 → A3 (BepiColombo-style) |
| **2× EGA** | 5 | Earth → Earth → Earth → A1 → A2 → A3 (MESSENGER-style) |

The existing `compute_path_with_flyby(...)` generalizes to Earth with only the flyby-body parameters changing. The ΔV-EGA variant (with deep-space maneuver) needs a 2-Lambert split for the outbound loop — a new function.

---

## 3. Low-thrust optimization: method choice

From the research survey, **Sims-Flanagan direct transcription wrapped in Monotonic Basin Hopping (MBH)** is the standard recipe used by JPL and ESA ACT. It is:

- **Robust**: splits each leg into N segments, each represented as a bounded impulse at the segment center. Continuity enforced via match-point mismatch constraints.
- **Well-supported in pykep**: `pykep.trajopt.mga_lt_nep` implements multi-leg Sims-Flanagan with gravity assists, matching our Earth → [GA] → A1 → A2 → A3 chain exactly.
- **Globalizable**: MBH (`pg.mbh`) wraps a local solver (`pg.nlopt("slsqp")` or IPOPT) with stochastic restarts.

Alternatives and why we're skipping them:

| Method | Verdict |
|---|---|
| Indirect (Pontryagin + costates) | Highest accuracy, smallest decision vector — but tiny convergence radius, costate initial guesses are unphysical. Not a first project. |
| Shape-based (exponential sinusoid) | Fast analytic filter, but planar and assumes thrust direction. Useful for coarse screening — **not shipped in pykep 2.6, would have to write it**. Skip. |
| Q-law (Petropoulos) | Feedback law for planetocentric spirals. Useful only if we do Earth-escape spiral phase. Out of scope. |

---

## 4. pykep API map (verified against v2.6 source)

### Key classes (from `pykep.trajopt`)

- **`direct_pl2pl`** — single-leg low-thrust between two planets. Good for smoke-testing the toolchain.
  ```python
  pk.trajopt.direct_pl2pl(p0="earth", pf="mars",
                          mass=1000, thrust=0.3, isp=3000, nseg=20,
                          t0=[500, 1000], tof=[200, 500],
                          vinf_dep=1e-3, vinf_arr=1e-3, hf=False)
  ```
  Methods: `fitness(z)`, `get_bounds()`, `plot_traj(z)`, `plot_control(z)`, `get_traj(z)`.

- **`mga_lt_nep`** — **this is our target class**. Multi-leg low-thrust with flybys. `seq` is the visit sequence (list of `pk.planet`), `n_seg` per leg, per-leg TOF bounds, launch window, v∞ bounds, spacecraft mass range, `Tmax`, `Isp`.
  ```python
  pk.trajopt.mga_lt_nep(
      seq=[earth, earth, mars, ast1, ast2, ast3],  # EGA + Mars GA
      n_seg=[20, 20, 30, 30, 30],                  # 5 legs
      t0=[9000, 10500], tof=[[300,700],[100,400],[200,900],[200,900],[200,900]],
      vinf_dep=3.5, vinf_arr=0.5, mass=[500., 1500.],
      Tmax=0.2, Isp=3000., high_fidelity=False)
  ```

- **`lt_margo`** — Earth-departure low-thrust to an NEO with optional SEP/NEP modes. Closest built-in to our asteroid-rendezvous use case; useful reference for SPICE-asteroid pattern.

- **`pl2pl_N_impulses`** — N-impulse DSM chemical leg. If we want to enrich impulsive search with DSMs (the ΔV-EGA variant) before going full low-thrust, start here.

### Primitives (`pykep.sims_flanagan`)

`spacecraft(mass, Tmax, Isp)`, `sc_state(r, v, m)`, `leg()` with `set(...)`, `mismatch_constraints()`, `throttles_constraints()`. Needed only if we build a custom UDP.

### SPICE bridge

pykep has its own SPICE caching, separate from `spiceypy.furnsh`. **Both must be loaded**:
```python
pk.util.load_spice_kernel(path_to_bsp)        # for pykep
spiceypy.furnsh(path_to_bsp)                   # for existing code
asteroid = pk.planet.spice(naif_id, '10', 'ECLIPJ2000', 'NONE',
                           pk.MU_SUN, mu_self=1e10, radius=500., safe_radius=500.)
```
Signature: `pk.planet.spice(target, observer, ref_frame, abcorr, mu_central, mu_self, radius, safe_radius)`. **All SI units inside pykep (m, m/s, kg)** — current code is km throughout; convert at the boundary.

### Solver pattern

```python
prob = pg.problem(udp)
prob.c_tol = [1e-5] * prob.get_nc()
uda  = pg.nlopt("slsqp"); uda.maxeval = 500     # or snopt7 if licensed
algo = pg.algorithm(pg.mbh(uda, stop=5, perturb=0.05))
pop  = pg.population(prob, size=1)
pop  = algo.evolve(pop)
print("feasible?", prob.feasibility_x(pop.champion_x))
udp.plot(pop.champion_x)
```

### Reference implementations
- `pykep/examples/_ex3.py` — Earth → Venus → Mercury low-thrust with flyby. **Directly translatable** to our Earth → Earth → Mars → A1 template.
- `pykep/trajopt/gym/_messenger.py`, `_rosetta.py` — GTOC-heritage UDPs closest to our multi-flyby problem.

---

## 5. Codebase integration (verified file:line refs)

### New module: `Python_Consolidated/lowthrust.py`

Self-contained so legacy Lambert code keeps working untouched.

```python
# Proposed API surface
compute_lowthrust_leg(r1, r2, t_flight, m_init, thrust, Isp, n_segments) -> dict
optimize_lowthrust_mission(seq, launch_range, sc_params, n_seg_per_leg) -> dict
build_udp_from_triplet(seq_ids, launch_range, sc_params) -> pk.trajopt.mga_lt_nep
lambert_to_lowthrust_seed(lambert_result, n_segments) -> np.ndarray  # initial guess
```

### Minimal modifications to existing files

| File | Line | Change |
|---|---|---|
| `optimization.py` | 192 | Add `'earth'` entry to `FLYBY_BODIES` |
| `optimization.py` | 333-368 | Add `lowthrust=False` flag to `optimize_best_architecture`; dispatch to `lowthrust.py` when set |
| `optimization.py` | 506-601 | In `two_level_optimize`: keep coarse impulsive, add low-thrust refinement to fine pass only (top-N) |
| `core.py` | — | Add spacecraft constants: `M_DRY_KG`, `M_PROP_KG`, `THRUST_N`, `ISP_S`, `N_SEGMENTS`. Add `get_mu(399)` / `get_radius(399)` for Earth if missing |
| `gcp/run_73ast_full.py` | 28-56 | `_eval_fine` dispatches low-thrust for top-50 |
| `gcp/gcp_config.py` | — | Add `conda install pygmo` to `SETUP_COMMANDS` |

### Why a new module (not extending optimization.py)
- Clean separation — low-thrust uses pygmo/pykep UDPs, not scipy DE.
- Different unit system inside (SI) — keep boundary conversions localized.
- Easier to smoke-test in isolation before wiring into the two-level pipeline.

---

## 6. Result schema extension

Current `result_dict` keys (from `compute_path_deltav`):
`delta_v_launch, delta_v_A1_arrive, delta_v_A1_leave, delta_v_A2_arrive, delta_v_A2_leave, delta_v_A3_arrive, delta_v_total, et_*, architecture`.

**New keys for low-thrust runs:**
```python
{
  ...existing keys...,                          # impulsive analogue kept for comparison
  'propulsion': 'lowthrust',                    # 'impulsive' | 'lowthrust' | 'hybrid'
  'm_initial_kg': 1500.0,
  'm_final_kg': 980.0,
  'm_prop_used_kg': 520.0,
  'prop_mass_fraction': 0.347,                  # KEY METRIC FOR THE TRADE STUDY
  'dv_integral_km_s': 12.3,                     # ∫|T|/m dt — expected > impulsive analogue
  'thrust_N': 0.2, 'isp_s': 3000.,
  'throttle_profile_per_leg': [...],            # list of np.ndarray, shape (n_seg, 3)
  'n_segments_per_leg': [20, 20, 30, 30, 30],
  'solver_converged': True,
  'mbh_feasibility_x': 1e-6,
}
```

Pickle format unchanged: `results.pkl = [(i, j, k, result_dict), ...]`.

---

## 7. Hybrid architecture — per-leg propulsion choice

**Key insight**: low-thrust and chemical each dominate in different regimes, so the globally optimal mission is probably a *mix*: chemical for launch + gravity assist, electric for long belt-to-belt cruises. This matches flight heritage — Dawn, Hayabusa2, Psyche all use hybrid architectures.

### 7.1 Per-leg regime analysis

| Leg | Typical requirements | Natural fit |
|---|---|---|
| L1: Earth → GA body | Fixed launch window, big Δv fast | **Chemical** (launch vehicle + Oberth at periapsis) |
| L2: GA body → A1 | Tied to the GA window timing | Usually chemical |
| L3: A1 → A2 | Heliocentric cruise, flexible TOF | **Electric** candidate |
| L4: A2 → A3 | Heliocentric cruise, flexible TOF | **Electric** candidate |

So the real architecture decision is just legs 3 and 4 → 4 variants: `CC`, `CE`, `EC`, `EE`. This keeps the search space tractable.

### 7.2 Pair-closeness pre-screen (avoid wasted LT compute)

Not every asteroid pair benefits from low-thrust. LT helps only when:
- **Δv demand is moderate** (~1-5 km/s). Below, chemical is already cheap. Above, the 14-yr cap punishes the long TOF LT needs to deliver.
- **Time is available** (typically ≥ 1 yr per leg for LT to reach break-even on mass).

We already have the impulsive Δv and TOF for every pair from the existing Lambert search. So the screen is:

```python
def lt_eligible(leg_impulsive_dv, leg_tof_yr):
    return leg_impulsive_dv < 5.0 and leg_tof_yr >= 1.0
```

Only pairs passing this get a full LT optimization. Typical hit rate: 20-40% of the top-30 triplets' legs.

### 7.3 Unified objective: final mass (not Δv)

Δv does not add up when Isp varies. **Use Tsiolkovsky per leg, then multiply mass fractions across the mission.** Equivalent objective to minimize:

```
J  =  Σ_i  Δv_i  /  Isp_i
```

Here `Isp_i ∈ {Isp_chem, Isp_electric}` depending on which propulsion is used for that leg. Minimizing J maximizes final delivered mass `m_final = m_launch · exp(-J · g0)`.

Result-dict should carry `Isp_per_leg` and `dv_per_leg` so the objective is reproducible downstream.

### 7.4 Decision rule per triplet

For each of the top-30 impulsive triplets:

```
for arch in [CC, CE, EC, EE]:
    for leg ∈ {L3, L4}:
        if arch says E for this leg AND lt_eligible(leg):
            optimize_leg_lt(leg, seeded_from_impulsive)   # pykep mga_lt_nep-style
        else:
            use_impulsive_leg(leg)
    compute_J(arch)
keep arch with minimum J
```

If LT optimization fails (infeasible, didn't converge), fall back to chemical for that leg. Never let a failed LT solve dominate an otherwise-good architecture.

---

## 8. Parallelization & pipeline (updated)

Low-thrust per-leg optimization is **~30 s**, not milliseconds. Running LT on every pair in every triplet is 14,040 × ≥2 legs × 30 s = 230+ CPU-hrs. Prohibitive. The pair-screen + top-N filter keeps it fast.

### 8.1 Four-stage pipeline

| Stage | Method | Work unit | Cost on 12 vCPU |
|---|---|---|---|
| **1. Coarse** | Lambert + flyby quick screen (existing) | 14,040 triplets × 4 archs | ~8 min |
| **2. Fine impulsive** | `differential_evolution` (existing) | top-50 triplets × 4 archs | ~5 min |
| **3. LT pair-screen** | Flag eligible legs in top-30 | ~60 legs (metadata only) | seconds |
| **4. LT refinement** | `sims_flanagan.leg` + MBH per eligible leg; score 4 archs per triplet | ~60 LT solves + 30 × 4 arch evals | ~40 min |

**Total**: ~55 min on current 12-vCPU VM, ~$0.40.

### 8.2 Initial-guess chain
For each LT leg, seed the decision vector from the impulsive Lambert result:
- `et_depart`, `et_arrive`: carry through unchanged
- `m_initial`: the ending mass from the prior leg (chain through)
- `throttles[0..N]`: initialize as `(Δv_lambert / N) * u_lambert` where `u_lambert` is the unit vector of the impulsive Δv. Direction is roughly correct; magnitude normalized to segment count.
- `match_point`: mid-leg.

This seed is almost always inside MBH's convergence basin. Empirically ~80% first-try success rate.

### 8.3 Parallelization
- Stage 1, 2: already parallelized across triplets via `multiprocessing.Pool`.
- Stage 4: parallelize LT leg solves across the 12 vCPUs. Each leg is independent.

---

## 9. Spacecraft parameters (reasonable defaults)

From Dawn heritage:

| Parameter | Value | Source |
|---|---|---|
| Dry mass | 500-1000 kg | undergraduate-project scale |
| Wet mass | 1000-1500 kg | |
| Thrust (continuous) | 0.09 N (NSTAR) or 0.24 N (NEXT-C) | JPL ion thruster data |
| Isp | 3100 s (NSTAR) or 4190 s (NEXT-C) | JPL |
| Solar-array-limited power | ~2.6 kW at 1 AU, drops as 1/r² | |
| Minimum TOF consideration | Thrust accel ~10⁻⁴ m/s², so ~460 days to build 4 km/s | |

**Implication**: 14-year mission cap is *tight* for 3 rendezvous + EGA loop. Low-thrust extends total flight time — expect winners to hit 10-13 yr.

---

### 9.1 Per-leg Isp assignment (for the hybrid objective)

| Leg | Propulsion | Isp used in objective |
|---|---|---|
| L1 (Earth launch + possible dep. burn) | Chemical | **320 s** (bipropellant) |
| L1 periapsis GA burn | Chemical | **320 s** |
| L2 (GA → A1) | Chemical by default | **320 s** |
| L3 (A1 → A2) | **Decided per triplet: C or E** | 320 s OR 3100 s |
| L4 (A2 → A3) | **Decided per triplet: C or E** | 320 s OR 3100 s |

Propellant mass per leg: `m_prop_i = m_start_i · (1 - exp(-Δv_i / (Isp_i · g0)))`. Chained through the mission to get final delivered dry mass.

---

## 10. Reading list

- **Sims & Flanagan (1999)**, *Preliminary Design of Low-Thrust Interplanetary Missions*, AAS 99-338. The original direct-transcription paper.
- **Petropoulos & Longuski (2004)**, *J. Spacecraft & Rockets* 41(5) — shape-based algorithm (useful context even if we skip it).
- **Yam, Di Lorenzo, Izzo (2011)**, *Proc. IMechE Part G* — the MBH+SQP recipe behind pykep's `mga_lt_nep`. Closest to what we'll actually implement.
- **Conway ed. (2010)**, *Spacecraft Trajectory Optimization*, Cambridge — book-length ref; ch. 1-3 cover direct methods.
- **Izzo (2016)**, *Designing Complex Interplanetary Trajectories for GTOC*, Springer — real-world recipe.
- **Farquhar, Muhonen, Church (1985)**, *J. Astro Sci.* — ΔV-EGA trajectory design.
- **pykep sources**: `pykep/trajopt/_mga_lt_nep.py`, `pykep/examples/_ex3.py`, `pykep/trajopt/gym/_messenger.py`.

---

## 11. Risks / gotchas

- **No Python 3.13 wheel for pykep**: pin GCP VM to 3.12. Current `env311` already handles this.
- **pygmo dependency**: `conda install -c conda-forge pygmo` alongside pykep. SNOPT7 is licensed — use `nlopt("slsqp")` or IPOPT.
- **Unit conversions**: pykep is SI, our code is km. Write `_to_pykep_units()` and `_from_pykep_units()` helpers at the boundary.
- **`pk.util.load_spice_kernel` is separate from `spiceypy.furnsh`** — both must be called.
- **MBH is stochastic**: ~50% seed failures normal. Budget enough restarts (`pg.mbh(uda, stop=5, perturb=0.05)`).
- **Match-point feasibility**: always check `prob.feasibility_x(pop.champion_x)` before trusting a result. Infeasible solutions look numerically fine but violate physics.
- **Moon-μ bug precedent**: add a unit test for every flyby body verifying `get_mu(id)` and `get_radius(id)` against known values.
- **v∞ at Earth for EGA**: `vinf_dep` bounds in `mga_lt_nep` are per-leg — must be relaxed (3-6 km/s) for EGA returns to be feasible.

---

## 12. Implementation milestones

1. **[done]** Earth GA added, physically realistic constraints applied. Result: no real EGAs in top-50; Mars GA dominates. Baseline ≈ 9.40 km/s, ~11-yr mission.
2. **Smoke-test pykep low-thrust** in isolation on a local env. Verify `pygmo` + `pykep.sims_flanagan.leg` + SPICE bridge work end-to-end on a single Earth→Mars leg. **~1 day.**
3. **Build `lowthrust.py` with a single-leg API**: `optimize_lt_leg(r0, v0, t0, r1, t1, m0, thrust, Isp)` returning `(m_final, dv_integral, throttle_profile, converged)`. Seeded from Lambert. **~2 days.**
4. **Hybrid scoring wrapper**: `evaluate_hybrid_architecture(triplet, arch ∈ {CC, CE, EC, EE})` returning final mass. Pair-screen filter (`lt_eligible`) applied before calling LT solver. **~1 day.**
5. **Stage 4 on GCP**: re-run the existing pipeline with the new hybrid refinement stage. Compare final-mass ranking to all-impulsive ranking. Produce trade-study table showing: best-impulsive mass, best-hybrid mass, mass savings %, winning architecture per top triplet. **~1 hr compute, 1 day analysis.**
6. **(Stretch)** Expand hybrid space to include L2 (GA→A1) as E-eligible, or allow deep-space maneuvers (DSMs) to enable real ΔV-EGA trajectories. Out of scope for v1.

Total: **~1 week for a working hybrid impulsive+LT pipeline**, ~$1 in GCP spend.
