# Joint Mass + Δv Optimization Plan

## The problem

Current pipeline:
1. `optimize_times_flyby` minimizes **chemical Δv** to find timings.
2. `evaluate_hybrid` then picks the best of 4 propulsion architectures (CC, CE, EC, EE on legs L3 and L4) **on the already-frozen timings**.

Why this is suboptimal: timing optimal for chemical is **not** timing optimal for electric. The Fortuna → Themis leg shows it — chemical wants 3.87 km/s, electric on that same trajectory wants 15.78 km/s, because the cruise window is too short for low-thrust to phase efficiently. A timing tuned for electric would lengthen cruise and reduce integrated dv, but the current pipeline never gets to try.

We want to **co-optimize** timing AND propulsion architecture, with the objective being delivered dry mass (which captures both Δv reduction AND Isp efficiency in one number).

## The objective

For each triplet, find the (timing, architecture) pair that maximizes **final delivered spacecraft mass** subject to:
- Total mission < 14 yr
- Stay times in [3 mo, 1 yr]
- Each transfer leg in [2 wk, 5 yr]
- LT thrust ceiling not exceeded

Equivalently: minimize the chemical-Isp-equivalent Δv:

```
dv_equiv = -Isp_chem * g0 * ln(m_final / m_init) / 1000   (km/s)
```

Lower `dv_equiv` ⇔ more delivered mass. This is the right single-number objective for "optimize both Δv and payload."

## Architecture space

8 propulsion architectures over the 3 transfer legs (L1 launch is always chemical):

| Code | L2 (GA→A1) | L3 (A1→A2) | L4 (A2→A3) |
|------|:---:|:---:|:---:|
| CCC | C | C | C |
| CCE | C | C | E |
| CEC | C | E | C |
| CEE | C | E | E |
| ECC | E | C | C |
| ECE | E | C | E |
| EEC | E | E | C |
| EEE | E | E | E |

Each architecture gets its own DE timing optimization.

## Inner loop: fast surrogate, not full LT solver

We can't run the Sims-Flanagan LT solver inside DE — it's ~15 s per call, DE wants ~30,000 evaluations. We'd need 5 days per triplet.

Surrogate cost model for each leg:

```
if leg_mode == 'C':
    dv_leg = lambert_dv      # exact
    Isp    = 320 s
elif leg_mode == 'E':
    dv_leg = lambert_dv * gravity_loss(tof, lambert_dv, m, thrust)
    Isp    = 3100 s
    if thrust * tof / m < 1.5 * lambert_dv:
        return INFEASIBLE_PENALTY    # not enough thrust authority
```

`gravity_loss(...)` is a closed-form correction calibrated against full LT solutions:
- `factor = 1.15 + 0.30 * (lambert_dv / dv_ceiling)` for tof > 1.5 yr
- `factor = 1.30 + 1.00 * (lambert_dv / dv_ceiling)` for shorter tof
- capped at 3.0 (beyond which it's clearly chemical territory)

Then chain Tsiolkovsky through all four legs to get `m_final`. Optimizer minimizes `-m_final` (or equivalently `dv_equiv`). One DE eval is ~5 ms — same order as the chemical-only Lambert path.

## Verification stage

Surrogate may underestimate. After DE returns the best timing per architecture, refine the top 3 candidates per triplet with the **full Sims-Flanagan solver** (`optimize_lt_leg`) on every electric leg. Re-rank by true delivered mass.

## Implementation steps

### Step 1 — `Python_Consolidated/mass_optimization.py` (new module)

Functions:
- `gravity_loss_factor(lambert_dv, tof_sec, m, thrust)` — surrogate correction
- `compute_path_mass(timing_vec, triplet_ids, launch_range, arch_code, m_init, thrust)` — Tsiolkovsky chain through all 4 legs given architecture, returns `(m_final, dv_per_leg, dv_equiv, feasible)`
- `score_paths_mass(...)` — DE objective wrapper, returns `-m_final` with infeasibility penalty
- `optimize_for_architecture(triplet, launch_range, arch_code, ...)` — single DE run for one architecture
- `pareto_optimize_triplet(triplet_idx, res_seed, ...)` — runs DE over all 8 architectures, returns Pareto-front list of (arch, dv_total, m_final, timings)
- `verify_with_full_lt(top_K, ...)` — refines best K with `optimize_lt_leg`

### Step 2 — `Python_Consolidated/gcp/run_mass_pareto.py` (new GCP runner)

- Load top-50 triplets from `optimal_asteroid_paths/pkl/results_69ast_ega_real.pkl`
- For each triplet, call `pareto_optimize_triplet` with all 8 archs (12-way parallel via mp.Pool)
- Collect Pareto fronts
- Run verification stage on top-3 per triplet
- Save to `optimal_asteroid_paths/pkl/results_mass_pareto_<timestamp>.pkl`
- Print a single ranked table sorted by best `m_final`

### Step 3 — Run on GCP

Compute estimate:
- 50 triplets × 8 arch × ~30 s DE run = 12,000 CPU-seconds = 16 min on 12 vCPU
- Verification: 50 triplets × 3 candidates × 2 LT legs × ~15 s = 7,500 CPU-seconds = 10 min on 12 vCPU
- Total: ~30 min wall clock, ~$0.20

GCP setup steps:
1. `gcloud auth login` (user runs this — interactive)
2. `gcloud compute instances create asteroid-optimizer ... --scopes=storage-ro,default`
3. SCP the new files (gcp/run_mass_pareto.py, mass_optimization.py)
4. SSH in, pull bucket data, install env, run script
5. SCP results back, delete VM

### Step 4 — Analyze + visualize

- Print top-10 by delivered mass (the answer to "lower Δv AND more payload")
- Plot Pareto front per triplet (Δv on x, mass on y) — shows the trade
- Compare against current best (FTP CCC = 14.20 km/s / 66 kg, FTP CCE = 14.87 / 135 kg)
- Identify: which triplet/architecture gives **both** lower Δv AND higher mass than the current FTP best?

## Risks and mitigations

1. **Surrogate inaccuracy** — verification stage with full LT catches this. If verification disagrees by >20%, recalibrate the surrogate against the verification data.
2. **DE convergence per architecture** — multi-start is cheap; we'll run 3 seeds per architecture per triplet and keep the best.
3. **Infeasibility everywhere** — some triplets will return INFEASIBLE for all electric arch (geometry too tight). That's fine; the chemical-only result is the answer for those.
4. **GCP auth** — user must run `gcloud auth login` once before I trigger the deploy script.

## Expected outcome

For each of the top-50 triplets, we'll have:
- Best (timing, architecture) pair
- Delivered mass at Psyche/A3
- Re-optimized Δv (likely lower than current chemical-only optimum for several arch)
- Pareto front showing the cost-of-fuel-efficiency trade

The headline number will be: **the triplet with the highest delivered mass, and what its Δv is.** Likely candidates: any triplet whose A1→A2 or A2→A3 leg has > 2 yr TOF (those have room for electric to shine).

For FTP specifically, expect:
- CCC stays around 13.6 km/s / 66 kg (bound by chemical Tsiolkovsky)
- ECC, ECE, EEE re-optimized to substantially lower integrated electric Δv (lengthened cruise) → much higher delivered mass than current EEE = 662 kg, possibly 800+ kg

For the global winner across all triplets — could be a different triplet entirely than Hertha → Polyxo → Alkeste (current 9.40 km/s leader), if some other triplet's geometry is unusually friendly to low-thrust.
