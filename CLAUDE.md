# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Asteroid mission trajectory optimization for Ae105c. Designs a spacecraft mission from Earth visiting **3 asteroids sequentially**, minimizing total delta-v (fuel cost). Two mission architectures:

1. **Direct**: Earth -> Asteroid 1 -> Asteroid 2 -> Asteroid 3
2. **Mars flyby**: Earth -> Mars (gravity assist) -> A1 -> A2 -> A3
3. **Earth + Mars chain**: Earth -> Earth GA loop -> Mars GA -> A1 -> A2 -> A3 (used by the selected mission concept)

The codebase was originally MATLAB, now **fully ported to Python** in `Python_Consolidated/`. The MATLAB code in `Code/` is legacy and no longer maintained.

**Selected mission concept:** PARTHENOPE → PSYCHE → THEMIS via **LT-after-launch chain** with optional Mars ballistic GA. Architecture: impulsive launch (≤ 7 km/s, excluded from objective), then Sims-Flanagan electric (Isp 3100 s, thrust ≤ 0.30 N) for ALL post-launch legs. See [`MISSION_PARTHENOPE_PSYCHE_THEMIS.md`](MISSION_PARTHENOPE_PSYCHE_THEMIS.md) §0 for the frozen constraint setup and full writeup.

The LT-chain optimizer is in `Python_Consolidated/lt_chain_optimization.py`; the GCP runner is `gcp/run_ppt_lt_chain.py`. Output includes per-leg integrated Δv, throttle-vs-time profiles (15 segments × 3 components per leg), Mars flyby v_∞ vectors / turn / altitude, full date/Δv breakdown.

Frozen constraint setup and required output specification: [`CONSTRAINTS_AND_OUTPUTS.md`](CONSTRAINTS_AND_OUTPUTS.md).

## Repository Structure

```
Ae105c-Project/
├── README.md                     # Friendly entry — overview + tutorial index
├── METHODOLOGY.md                # Algorithm reference (consolidated from old PLAN docs)
├── MISSION_PARTHENOPE_PSYCHE_THEMIS.md  # Selected mission concept + full physics writeup
├── CLAUDE.md                     # This file (AI-assistant guidance)
├── Tutorials/                    # 7 task-focused walkthroughs + FAQ
├── docs/archive/                 # Historical PLAN docs (plan.md, etc.)
├── Python_Consolidated/          # Active Python codebase (7 files)
│   ├── main.py                   # CLI entry point — all user-facing workflows
│   ├── core.py                   # Lambert/SPICE/flyby primitives + audit_flyby_geometry
│   ├── optimization.py           # Δv scoring, DE optimizer, two-level + beam search, gravity assists
│   ├── lowthrust.py              # Sims-Flanagan low-thrust leg solver
│   ├── mass_optimization.py      # Mass-Pareto across 8 propulsion architectures (CCC..EEE)
│   ├── visualization.py          # Flightpath animation, orbit plotting (library)
│   ├── tradeoff.py               # Asteroid science/physical scoring (standalone, no SPICE)
│   ├── check_mission.py          # Independent physics verifier (re-derives all Δv from scratch)
│   ├── api/                      # FastAPI HTTP service: server.py, schemas.py, jobs.py, client.py
│   │                             #   Run: python -m Python_Consolidated.api
│   ├── gcp/                      # GCP cloud-runner scripts (3 files)
│   └── requirements.txt
├── NOTABLE_ASTEROID_BSPs/        # 69 asteroid ephemeris files (.bsp)
├── SPICE_BSPs/                   # Larger asteroid pool (49 BSPs)
├── Renders/                      # Generated images and GIFs
├── optimal_asteroid_paths/pkl/   # Saved optimization results (pickle)
├── asteroid_tradeoff.csv         # Ranked asteroid table (407 asteroids)
└── Code/                         # Legacy MATLAB code (not maintained)
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

Everything runs through `main.py` from the **repository root** (not from `Python_Consolidated/`). All subcommands resolve `NOTABLE_ASTEROID_BSPs/` relative to the repo root. For task-focused walkthroughs see `Tutorials/`.

```bash
# Optimization
python Python_Consolidated/main.py optimize                       # two-level dv-only, all asteroids
python Python_Consolidated/main.py optimize --science 0.7         # 70% dv + 30% science
python Python_Consolidated/main.py optimize --diverse             # require C+S+X/M diversity
python Python_Consolidated/main.py optimize --feasible            # diverse + physical-flyby audit (recommended)
python Python_Consolidated/main.py optimize --beam 15             # beam search, K=15
python Python_Consolidated/main.py optimize --pareto              # mass-Pareto across 8 architectures

# Visualization & verification
python Python_Consolidated/main.py list                           # list saved pkl results
python Python_Consolidated/main.py plot RESULT.pkl                # show top-10 entries
python Python_Consolidated/main.py plot RESULT.pkl --rank 1       # static 3D PNG of #1
python Python_Consolidated/main.py plot RESULT.pkl --rank 1 --gif # animated GIF of #1
python Python_Consolidated/main.py plot RESULT.pkl --names HEDDA BEATRIX PROSERPINA --gif
python Python_Consolidated/main.py verify RESULT.pkl --rank 1     # audit flyby physics (pass/fail)
python Python_Consolidated/main.py inspect RESULT.pkl --rank 1    # full per-leg dump (Lambert V1/V2, Δv vectors, flyby diagnostics)
python Python_Consolidated/main.py inspect RESULT.pkl --top 5     # inspect top 5 entries

# Auxiliary
python Python_Consolidated/main.py rank                           # rebuild asteroid_tradeoff.csv
python Python_Consolidated/main.py animate-asteroids              # MP4 of all asteroid orbits
```

GCP runner env var conventions (for `gcp/run_mars_diverse_science.py`):
- `ALPHA` (0–1): Δv weight in combined score. 1.0 = pure Δv, lower = more science.
- `REQUIRED_ASTEROIDS=A,B`: at least one of {A,B} must appear in every triplet
- `REQUIRE_ALL_ASTEROIDS=A,B`: ALL of {A,B} must appear in every triplet (overrides REQUIRED)

## HTTP API

Same workflows exposed over HTTP — see `Tutorials/07_using_the_api.md`.

```bash
python -m Python_Consolidated.api          # local server at :8000
# Swagger UI: http://localhost:8000/docs
```

API package layout:
- `api/server.py` — FastAPI app, all route definitions
- `api/schemas.py` — Pydantic request/response models
- `api/jobs.py` — SQLite-backed job queue with subprocess + GCP executors
- `api/client.py` — Python client (`Client` class with `Job` wrapper)
- `api/serialization.py` — pkl/numpy → JSON-safe dict
- `api/__main__.py` — `python -m Python_Consolidated.api` entrypoint

Job state: `optimal_asteroid_paths/api_jobs/jobs.db` (SQLite).
Job logs: `optimal_asteroid_paths/api_jobs/<uuid>.log`.

Endpoints (all under `/api/v1/`):
- `GET /asteroids` `GET /results` `GET /results/{f}` `GET /results/{f}/entries/{rank}`
- `POST /verify` `POST /inspect`
- `POST /jobs/optimize` `GET /jobs` `GET /jobs/{id}` `DELETE /jobs/{id}` `GET /jobs/{id}/log`

`main.py plot` and `main.py verify` are generic over the saved-pkl formats — they handle two-level (`(i,j,k,result)` tuples), mass-Pareto (`{'all_results': [...]}`), `diverse_top3_feasible.pkl` (`{'audited': [...]}`), and single-triplet dicts.

## Architecture

### core.py — Foundation layer

Wraps pykep and spiceypy behind a clean interface. All units: **km, km/s, km^3/s^2**.

| Function | Purpose |
|----------|---------|
| `solve_lambert(r1, r2, tof_days, m, mu)` | Wraps `pk.lambert_problem`. Returns (V1, V2, exitflag). |
| `solve_lambert_best(r1, r2, tof_days, mu)` | Tries m=0,1,2 revolutions + both directions, returns best. |
| `two_body_sim(t_final, x0, mu)` | Wraps `pk.propagate_lagrangian`. Returns (X, T) arrays. |
| `compute_flyby_dv(v_in, v_out, v_planet, mu, r)` | Wraps `pk.fb_dv` **with turn-angle feasibility check** (returns 1e3 km/s penalty if geometry impossible). |
| `audit_flyby_geometry(et_launch, et_flyby, et_arr_a1, a1_id, arch)` | Independent post-hoc audit. Returns v_inf vectors, turn angle, max turn at safe r_p, periapsis altitude, feasibility flag. |
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

### lowthrust.py — Low-thrust leg solver

Sims-Flanagan direct transcription via `scipy.optimize.least_squares`. Splits a heliocentric leg into N segments with half-coast / impulsive kick / half-coast. Used by `mass_optimization.verify_with_full_lt` for verification of electric-propulsion legs.

Constants: `ISP_CHEM=320s`, `ISP_ELEC=3100s`, `G0=9.80665`. Defaults: `DEFAULT_M_INIT_KG=1500`, `DEFAULT_THRUST_N=0.30`, `DEFAULT_NSEG=15`.

### mass_optimization.py — Mass-Pareto across propulsion architectures

| Function | Purpose |
|----------|---------|
| `compute_path_mass(...)` | Tsiolkovsky chain through 4 legs (L1=flyby, L2-L4=transfers). Returns `m_final_kg`, `dv_equiv_kms`, feasibility. |
| `gravity_loss_factor(...)` | Surrogate multiplier on Lambert dv to estimate integrated low-thrust dv (calibrated). |
| `optimize_for_architecture(...)` | DE optimization for a single arch_code (e.g. `'EEE'`). |
| `pareto_optimize_triplet(...)` | All 8 architectures × multi-seed × m-revs sweep for one triplet. |
| `verify_with_full_lt(...)` | Re-evaluate top arch_result with the real Sims-Flanagan solver. |

Architecture codes: 3-letter strings for L2/L3/L4 modes — `'C'`=chemical (Isp 320s), `'E'`=electric (Isp 3100s). All 8 = `ARCH_CODES`.

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

⚠ **WARNING:** Many older result pkls predate the **ballistic-flyby fix**
(see `core.compute_flyby_dv` and `core.BALLISTIC_VINF_TOLERANCE_KMS`).
Their flybys may be powered (energy non-conserving), which the project no
longer accepts. Use `python Python_Consolidated/check_mission.py <pkl>` to
verify before quoting any number from the older files. The selected mission
in `MISSION_PARTHENOPE_PSYCHE_THEMIS.md` uses the corrected pipeline.

Results are saved as pickle files in `optimal_asteroid_paths/pkl/`. Load with:
```python
import pickle
with open('optimal_asteroid_paths/pkl/results_69ast_ga.pkl', 'rb') as f:
    results = pickle.load(f)  # list of (i, j, k, result_dict)
```

### Selected mission concept (Earth + Mars GA chain)

| Rank | Path | dV (km/s) | Architecture | Notes |
|------|------|:---------:|:-----:|---|
| **1** | **PARTHENOPE [S] → PSYCHE [X/M] → THEMIS [C]** | **13.45** | Earth GA + Mars GA chain | Earth GA: 0.74 km/s Oberth burn; Mars GA: ballistic. See `MISSION_PARTHENOPE_PSYCHE_THEMIS.md`. |

### Best paths — minimum delta-v, ballistic-only Mars GA (post-fix)

| Rank | Path | dV (km/s) | Flyby | Notes |
|------|------|:---------:|:-----:|---|
| 1 | **HARMONIA [S] → LUTETIA [X/M] → IRMA [C]** | **12.97** | Mars (ballistic) | Verified — passes both geometric + ballistic checks |
| 2 | HARMONIA → LUTETIA → AGLAJA | 13.04 | Mars (ballistic) | |
| 3 | HARMONIA → LUTETIA → HEDDA | 13.05 | Mars (ballistic) | |

### Best paths — minimum delta-v, OLD pre-fix (powered flybys allowed) — DO NOT TRUST

| Rank | Path | dV (km/s) | Flyby |
|------|------|:---------:|:-----:|
| 1 | **Hertha [X/M] -> Polyxo [C] -> Alkeste [S]** | **9.40** | Mars |
| 2 | Virginia [C] -> Psyche [X/M] -> Parthenope [S] | 9.51 | Moon |
| 3 | Massalia [S] -> Misa [C] -> Psyche [X/M] | 9.77 | Moon |

### Best paths — science priority (70% science, 30% dv, gravity assists)

| Rank | Path | dV | Science | Score | Flyby |
|------|------|:--:|:------:|:-----:|:-----:|
| 1 | **Aegina [C] -> Beatrix [X/M] -> Vesta [S]** | **10.8** | **20.1** | **10.14** | Mars |
| 2 | Massalia [S] -> Psyche [X/M] -> Themis [C] | 10.7 | 20.0 | 10.25 | Moon |
| 3 | Massalia [S] -> Psyche [X/M] -> Concordia [C] | 10.1 | 19.7 | 10.25 | Moon |

Science score uses tradeoff table weights (renormalized without dv): 21.4% sci potential + 21.4% inclination + 18.6% radius + 17.1% mass + 10% eccentricity + 7.1% rotation + 4.3% SMA.

### All saved result files

| File | Asteroids | Composition | GA | Science | Best |
|------|:---------:|:-----------:|:--:|:-------:|:----:|
| `pkl/results_69ast_ga.pkl` | 69 | C+S+X/M | Moon+Mars | None | 9.40 km/s |
| `pkl/results_science_priority_v2.pkl` | 69 | C+S+X/M | Moon+Mars | 70% (tradeoff weights) | score 10.14 |
| `pkl/results_diverse_CSM.pkl` | 50 | C+S+X/M | No | None | 13.80 km/s |
| `pkl/results_science_priority.pkl` | 69 | C+S+X/M | Moon+Mars | 70% (old weights) | score 10.05 |
| `pkl/results_diverse_science_weighted.pkl` | 50 | C+S+X/M | No | 50% | 14.61 km/s |
| `pkl/results_50ast_full.pkl` | 50 | Any | No | None | 13.07 km/s |

## GCP Compute

**All GCP config is in `Python_Consolidated/gcp/gcp_config.py` — read that file first.**

Key facts:
- **Project**: `project-8b1249f5-4cb6-4dad-8a9`
- **Machine**: `e2-custom-12-49152` (12 vCPU, max allowed by quota)
- **Zone**: `us-west1-b`
- **GCS bucket**: `gs://ae105c-asteroid-data` (PUBLIC READ — kernels + BSPs stored permanently, ~30 sec pull)
- **NEVER upload kernels via SCP** — always pull from the GCS bucket
- **IMPORTANT**: Create VM with `--scopes=storage-ro,default` so it can read the bucket
- **Auth**: `gcloud auth login` required before each session
- **Scripts**: `Python_Consolidated/gcp/` (run_optimization.py, run_diverse.py, run_science_priority.py, run_73ast_full.py)

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
