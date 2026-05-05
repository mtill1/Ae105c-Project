# Constraints and Required Outputs

This document specifies the **frozen constraint setup** and the **required
output format** for the Ae105c trajectory optimization. The optimizer code
(`lt_chain_optimization.py`) and the GCP runner (`gcp/run_ppt_lt_chain.py`)
enforce these.

---

## 1. Current Constraint Setup (as coded now)

### Mission shape

```
Earth → (optional one flyby body) → A1 → A2 → A3
```

### Objective

Minimize **post-launch mission Δv** (`dv_after_launch_km_s`):
- Sum of integrated low-thrust Δvs across all post-launch legs
- Launch Δv is tracked but **excluded from the objective** (reference-only)

### Launch Δv

| Parameter | Value |
|---|---|
| Type | Impulsive (instantaneous chemical burn at Earth departure) |
| Magnitude constraint | **≤ 7 km/s** |
| Included in objective? | **No** — paid by launch vehicle, reported as reference |

### Post-launch propulsion

| Parameter | Value |
|---|---|
| Engine type | **Electric only** — Sims-Flanagan low-thrust |
| Specific impulse (Isp) | **3100 s** |
| Thrust cap | **0.30 N** per low-thrust leg solve (not a mission-total budget — applies independently to each leg) |
| Post-launch impulsive burns allowed? | **No** |
| Post-launch chemical propulsion allowed? | **No** |

### Spacecraft mass

| Parameter | Value |
|---|---|
| Spacecraft launch mass (total) | **3000 kg** (delivered by launcher to interplanetary v_∞) |
| LT-solve chain initial mass | **1500 kg** (the mass that begins the post-launch electric propulsion phase) |

### Mission duration

| Parameter | Value |
|---|---|
| Hard cap | **30 years** |
| Practical limit | Bounded above by SPICE BSP ephemeris coverage (~2050 for most asteroids) |

### Per-leg timing bounds

| Leg type | Lower bound | Upper bound |
|---|---|---|
| Earth → flyby body (Mars window) | 0.3 yr | 3.0 yr |
| Each LT transfer leg TOF | **1 year** (SF solver handles short legs; engine non-saturating in practice) | min(8 yr, mission_max/4) |
| Stay duration at each asteroid | **3 months** | 1 year |

### Composition diversity

| Parameter | Value |
|---|---|
| Default | Not enforced |
| Opt-in | `--require-diverse-composition` flag |
| When enabled | Requires C + S + X/M across the triplet |
| PARTHENOPE → PSYCHE → THEMIS | Satisfies diversity: S + X/M + C |

### Flyby physics constraints (when a flyby is used)

| Constraint | Value |
|---|---|
| Powered burn at flyby periapsis | **None allowed** — unpowered turn only |
| Energy conservation | `|v_∞_in| ≈ |v_∞_out|` within **0.05 km/s** tolerance |
| Geometric turn | ≤ natural maximum at safe periapsis altitude |
| Minimum altitude — Mars | 200 km above surface |
| Minimum altitude — Earth GA | 300 km above surface |

### Electric model constants

| Constant | Value |
|---|---|
| `ISP_ELEC` | 3100 s |
| `G0` | 9.80665 m/s² |
| `DEFAULT_THRUST_N` | 0.30 N |
| `DEFAULT_NSEG` | 15 segments per Sims-Flanagan leg |
| `DEFAULT_M_INIT_KG` | 1500 kg (start of LT chain) |

---

## 2. Required Output Data

For every optimized trajectory, the following data must be produced and
saved. This is the specification the GCP runner (`run_ppt_lt_chain.py`)
and the verification function (`verify_lt_chain_full`) target.

### 2.1 Earth launch

| Field | Units | Description |
|---|---|---|
| `launch_date_utc` | UTC string | Earth departure epoch |
| `launch_dv_kms` | km/s | Impulsive launch Δv (= v_∞ at Earth). Must be ≤ 7.0. |
| `launch_dv_vector_kms` | 3-vector (km/s) | Heliocentric ECLIPJ2000 Δv vector at Earth departure |

### 2.2 Mars (or other flyby body) gravity assist

| Field | Units | Description |
|---|---|---|
| `flyby_date_utc` | UTC string | Flyby epoch |
| `flyby_body` | string | `'mars'`, `'earth'`, or `None` (direct) |
| `flyby_angle_deg` | degrees | Required turn angle between v_∞_in and v_∞_out |
| `flyby_altitude_km` | km | Periapsis altitude above body surface |
| `flyby_turn_max_deg` | degrees | Maximum natural turn at safe periapsis |
| `v_inf_in_vector_kms` | 3-vector (km/s) | Incoming v_∞ vector (heliocentric ECLIPJ2000) |
| `v_inf_out_vector_kms` | 3-vector (km/s) | Outgoing v_∞ vector (heliocentric ECLIPJ2000) |
| `v_inf_in_magnitude_kms` | km/s | `|v_∞_in|` |
| `v_inf_out_magnitude_kms` | km/s | `|v_∞_out|` |
| `energy_residual_kms` | km/s | `|v_∞_out| − |v_∞_in|` (should be < 0.05 for ballistic) |
| `geometric_feasible` | bool | Turn ≤ natural max? |
| `ballistic_feasible` | bool | `|residual|` < 0.05 km/s? |

### 2.3 Thrust vs time (per low-thrust leg)

Each LT leg (e.g., Mars→A1, A1→A2, A2→A3) has N segments (default 15).
For each segment:

| Field | Units | Description |
|---|---|---|
| `time_yr_from_leg_start` | years | Mid-point time of this segment within the leg |
| `segment_dt_yr` | years | Duration of one segment |
| `throttle_unit_vector` | 3-vector [-1, 1] | Throttle direction in heliocentric frame; `|u| ≤ 1` |
| `thrust_magnitude_N` | Newtons | Actual thrust = `|u| × thrust_max_N` |
| `thrust_max_N` | Newtons | Engine cap (0.30 N) |

Aggregate per-leg statistics:

| Field | Units | Description |
|---|---|---|
| `dv_integral_kms` | km/s | Total integrated Δv for this leg (sum of segment impulses) |
| `tof_yr` | years | Time of flight for this leg |
| `m_in_kg` | kg | Spacecraft mass at start of this leg |
| `m_out_kg` | kg | Spacecraft mass at end of this leg |
| `pos_err_km` | km | Position residual at leg endpoint (Sims-Flanagan convergence) |
| `vel_err_kms` | km/s | Velocity residual at leg endpoint |
| `converged` | bool | Did the Sims-Flanagan solver converge? |
| `mean_thrust_mN` | mN | Mean thrust magnitude across all segments |
| `peak_thrust_mN` | mN | Maximum thrust magnitude |
| `duty_cycle_pct` | % | Fraction of segments with thrust > 5% of max |

### 2.4 Total Δv

| Field | Units | Description |
|---|---|---|
| `total_post_launch_dv_kms` | km/s | **THE OBJECTIVE** — sum of all LT leg `dv_integral_kms` |
| `launch_dv_kms` | km/s | Impulsive launch (reference, not in objective) |
| `total_trip_dv_kms` | km/s | `launch_dv + total_post_launch_dv` (full mission cost, all-inclusive) |

### 2.5 Asteroid arrival and departure dates

For each asteroid (A1 = first visited, A2 = second, A3 = third/final):

| Field | Units | Description |
|---|---|---|
| `arrive_date_utc` | UTC string | Epoch when spacecraft arrives (matches asteroid velocity) |
| `depart_date_utc` | UTC string | Epoch when spacecraft departs for next target |
| `stay_duration_months` | months | Time spent at the asteroid (science phase) |

### 2.6 Mass progression

| Field | Units | Description |
|---|---|---|
| `spacecraft_launch_mass_kg` | kg | 3000 kg (total, delivered by launcher) |
| `lt_chain_initial_mass_kg` | kg | 1500 kg (start of post-launch electric phase) |
| `mass_after_leg_N_kg` | kg | Mass remaining after each LT leg (N = 1, 2, 3) |
| `final_delivered_mass_kg` | kg | Mass at final asteroid arrival |
| `propellant_fraction_pct` | % | `(1 − final/initial) × 100` for the LT chain |

### 2.7 Lambert m-revolutions

| Field | Description |
|---|---|
| `m_revs` | Tuple of integers (one per leg) specifying the Lambert multi-revolution branch used |

---

## 3. Output file format

Results are saved as Python pickle files in `optimal_asteroid_paths/pkl/`.

### Structure of `ppt_lt_chain.pkl`

```python
{
    'triplet_set':    ['PARTHENOPE', 'PSYCHE', 'THEMIS'],
    'best_ordering':  ['THEMIS', 'PARTHENOPE', 'PSYCHE'],    # optimizer-chosen ordering
    'best_flyby':     'mars',                                   # or None for direct
    'config':         { ... LTChainConfig fields ... },
    'surrogate':      { ... surrogate result dict ... },
    'verified': {
        'flyby_name':              'mars',
        'm_revs':                  [0, 1, 0, 0],
        'launch_dv_kms':           6.43,
        'epochs': {
            'et_launch':   ...,    # SPICE ET (seconds past J2000)
            'et_flyby':    ...,
            'et_a1_arr':   ...,
            'et_a1_dep':   ...,
            'et_a2_arr':   ...,
            'et_a2_dep':   ...,
            'et_a3_arr':   ...,
        },
        'flyby_audit': {
            'feasible':          True,
            'geometric_ok':      True,
            'ballistic_ok':      True,
            'v_inf_in_kms':      ...,
            'v_inf_out_kms':     ...,
            'v_inf_in_vec':      [x, y, z],
            'v_inf_out_vec':     [x, y, z],
            'turn_angle_deg':    ...,
            'turn_max_deg':      ...,
            'periapsis_alt_km':  ...,
            'energy_residual_kms': ...,
        },
        'verified_legs': [
            {
                'label':             'mars→A1',
                'et_start':          ...,
                'et_end':            ...,
                'tof_yr':            ...,
                'm_revs':            1,
                'm_in_kg':           1500.0,
                'm_out_kg':          ...,
                'converged':         True,
                'dv_integral_kms':   ...,
                'pos_err_km':        ...,
                'vel_err_kms':       ...,
                'thrust_profile': {
                    'time_yr_from_leg_start': [...],   # 15 mid-segment times
                    'segment_dt_yr':          ...,
                    'throttle_unit_vector':   [[u1x,u1y,u1z], ...],  # 15×3
                    'thrust_magnitude_N':     [...],   # 15 values
                    'thrust_max_N':           0.30,
                },
            },
            ...  # one per LT leg
        ],
        'post_launch_dv_kms_full': ...,
        'm_final_kg_full':         ...,
        'feasibility': {
            'all_legs_converged': True,
        },
    },
    'all_orderings': [
        {'ordering': [...], 'flyby': 'mars', 'feasible': True, 'surrogate': {...}},
        {'ordering': [...], 'flyby': None,   'feasible': False, 'elapsed_s': ...},
        ...  # all 12 combinations (6 orderings × 2 architectures)
    ],
}
```

---

## 4. How to access the output

### CLI

```bash
# List saved results
python Python_Consolidated/main.py list

# Inspect per-leg Lambert velocities, Δv vectors, flyby diagnostics
python Python_Consolidated/main.py inspect ppt_lt_chain.pkl --rank 1

# Physical-feasibility audit (pass/fail + flyby checks)
python Python_Consolidated/main.py verify ppt_lt_chain.pkl --rank 1

# Independent physics checker (re-derives everything from scratch)
python Python_Consolidated/check_mission.py ppt_lt_chain.pkl --rank 1
```

### Python

```python
import pickle
with open('optimal_asteroid_paths/pkl/ppt_lt_chain.pkl', 'rb') as f:
    data = pickle.load(f)

v = data['verified']
print(v['launch_dv_kms'])               # launch
print(v['post_launch_dv_kms_full'])      # objective
print(v['m_final_kg_full'])              # delivered mass

for leg in v['verified_legs']:
    print(leg['label'], leg['dv_integral_kms'], leg['tof_yr'])
    tp = leg['thrust_profile']
    print('  throttles:', tp['throttle_unit_vector'][:3], '...')
    print('  thrust (N):', tp['thrust_magnitude_N'][:3], '...')

fa = v['flyby_audit']
print(fa['v_inf_in_vec'], fa['v_inf_out_vec'])
print(fa['turn_angle_deg'], fa['periapsis_alt_km'])
```

### HTTP API

```bash
curl http://localhost:8000/api/v1/results/ppt_lt_chain.pkl
curl -X POST http://localhost:8000/api/v1/inspect \
     -H "Content-Type: application/json" \
     -d '{"pkl":"ppt_lt_chain.pkl","rank":1}'
```

---

## 5. Verification checklist

Before trusting any result, confirm ALL of the following:

| # | Check | Tool |
|---|---|---|
| 1 | Launch Δv ≤ 7.0 km/s | `check_mission.py` or manual |
| 2 | Mars flyby ballistic (`|v_∞_in| ≈ |v_∞_out|` within 0.05 km/s) | `audit_flyby_geometry()` |
| 3 | Mars flyby geometric (turn ≤ max at 200 km altitude) | `audit_flyby_geometry()` |
| 4 | All LT legs converged (pos_err < 1.5×10⁶ km, vel_err < 0.15 km/s) | `verify_lt_chain_full()` |
| 5 | Mission duration ≤ 30 yr | Epoch difference |
| 6 | Each stay ≥ 3 months | Epoch difference |
| 7 | Each LT leg TOF ≥ 1 year | Epoch difference |
| 8 | Δv breakdown sums correctly | Compare sum of legs to reported total |
| 9 | Mass chain consistent (Tsiolkovsky at Isp 3100 s) | Manual or `check_mission.py` |
