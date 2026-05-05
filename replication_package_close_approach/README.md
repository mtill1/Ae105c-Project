# Replication Package — PPT LT-Chain (Close-Approach Variant)

PARTHENOPE → PSYCHE → THEMIS, exploiting the 2041-04-25 Psyche-Themis close
approach. **Same launch date as the v2 baseline** (2029-07-29), but a
shorter, cheaper Psyche→Themis transfer that leverages a 0.144 AU minimum
separation between the two asteroids.

## Package layout

```
replication_package_close_approach/
├── README.md                      ← this file
│
├── PPT_LT_CHAIN_trajectory.bsp    ← SPICE SPK Type 9 (the trajectory)
├── PPT_LT_CHAIN_trajectory.oem    ← CCSDS OEM v2.0 ASCII
├── PPT_LT_CHAIN_trajectory.csv    ← Lowest-common-denominator CSV
├── PPT_LT_CHAIN_metadata.json     ← Machine-readable mission summary
├── PPT_LT_CHAIN_names.tpc         ← SPICE name binding (-1030 ↔ PPT-LT-CHAIN-CA)
│
├── thrust_vs_time.png             ← Thrust + duty-cycle profile
├── trajectory_3d.gif              ← 3D animation
├── trajectory_2d.gif              ← Top-down ecliptic
│
├── optimization/                  ← Source-of-truth pkl + the optimizer code
│   ├── ppt_lt_chain_close_approach.pkl   ← Saved DE + Sims-Flanagan result
│   ├── run.log                            ← Full GCP run log (80 DE runs + verify)
│   ├── run_ppt_lt_close_approach.py       ← The runner (top-level entry point)
│   ├── lt_chain_optimization.py           ← LT-chain DE driver
│   ├── core.py                            ← SPICE/Lambert primitives
│   ├── lowthrust.py                       ← Sims-Flanagan leg solver
│   ├── optimization.py                    ← legacy Lambert optimizer (used by lt_chain)
│   └── mass_optimization.py               ← gravity-loss surrogate (used by lt_chain)
│
├── verification/                  ← Two independent audits
│   ├── audit_145_checks.md                ← Full Sims-Flanagan re-derivation report (145/145 pass)
│   ├── verify_ppt_lt_chain_full.py        ← Run to regenerate the report
│   └── check_constraints.py               ← Project-level constraint audit (36/36 pass)
│
└── tools/                         ← Scripts to regenerate trajectory + figures from pkl
    ├── export_ppt_trajectory.py           ← pkl → BSP / OEM / CSV / JSON / TPC
    ├── plot_ppt_lt_chain_gif.py           ← pkl → 3D GIF
    ├── plot_ppt_lt_chain_gif_2d.py        ← pkl → 2D top-down GIF
    └── plot_ppt_lt_chain_thrust.py        ← pkl → thrust-vs-time PNG
```

### File summary

| File | Format | Use case |
|---|---|---|
| `PPT_LT_CHAIN_trajectory.bsp` | SPICE SPK Type 9 binary | Native SPICE — load with `spiceypy.furnsh` |
| `PPT_LT_CHAIN_names.tpc`      | SPICE text kernel       | Maps the spacecraft name to NAIF ID **-1030** (`PPT-LT-CHAIN-CA`) |
| `PPT_LT_CHAIN_trajectory.oem` | CCSDS OEM v2.0 ASCII   | Inter-agency standard, no SPICE needed |
| `PPT_LT_CHAIN_trajectory.csv` | CSV                     | Excel, MATLAB, anything |
| `PPT_LT_CHAIN_metadata.json`  | JSON                    | Machine-readable mission summary |
| `thrust_vs_time.png`          | PNG                     | Thrust + duty-cycle profile |
| `trajectory_3d.gif`           | GIF                     | 3D animation |
| `trajectory_2d.gif`           | GIF                     | Top-down ecliptic |

The trajectory has **959 state samples** at ~4.7 day spacing across the full
12.40-year mission. Position/velocity round-trip through SPICE with **0 km
error** at sample points (sub-meter at intermediate times via 5th-order
Lagrange interpolation).

> NAIF ID **-1030** is used here (the v2 baseline package uses -1029) so
> both packages can be furnished simultaneously without conflict.

---

## 1. What is this trajectory?

A **direct, all-low-thrust** mission visiting three main-belt asteroids,
with the final Psyche→Themis leg timed to land on the bodies' close
approach window:

```
Earth (launch, impulsive ≤ 7 km/s)
   ↓
PARTHENOPE [S-type, NAIF 20000011]   — 3.0-month stay
   ↓                                    (electric LT cruise, 3.74 yr)
PSYCHE     [X/M-type, NAIF 20000016] — 3.0-month stay
   ↓                                    (electric LT cruise, 1.34 yr — exploits close approach)
THEMIS     [C-type, NAIF 20000024]   — final rendezvous
```

| | |
|---|---|
| Total mission | **12.40 years** (2029-07-29 → 2041-12-20) |
| Launch Δv (paid by launcher) | 7.0000 km/s |
| Post-launch Δv (sum of LT integrated) | **12.7499 km/s** |
| Spacecraft launch mass | 3000 kg (delivered by launcher) |
| LT-chain initial mass | 1500 kg (start of post-launch electric phase) |
| Final delivered mass | **986.16 kg** (65.7% of 1500 kg LT start) |
| Engine | Solar electric, Isp 3100 s, max thrust 0.30 N |
| Lambert m_revs | (1, 1, 0) — 1 rev each on first two legs, direct on Psyche→Themis |

### Comparison vs the v2 baseline

| | v2 baseline | **Close-approach** | Δ |
|---|:---:|:---:|:---:|
| Post-launch Δv | 13.105 km/s | **12.7499** | **−0.355** |
| Mission duration | 16.79 yr | **12.40 yr** | **−4.39** |
| Final mass | 974.7 kg | **986.16** | **+11.5** |
| Themis arrival | 2046-05-15 | 2041-12-20 | (4.4 yr earlier) |
| Psyche-Themis sep at arrival | — | 0.664 AU | (close approach min: 0.144 AU on 2041-04-25, eight months earlier) |

The close-approach variant **saves 0.36 km/s of Δv and 4.4 years of mission
time** by timing the Psyche→Themis leg to fall during the once-per-50-year
synodic close encounter of the two asteroids.

---

## 2. Reference frame, units, time system

| | |
|---|---|
| **Frame** | ECLIPJ2000 — Earth-Mean-Equator/Equinox of J2000.0 ecliptic |
| **Center body** | Sun barycenter (NAIF ID 10) |
| **Position units** | km |
| **Velocity units** | km/s |
| **Time system** | SPICE Ephemeris Time (TDB seconds past J2000.0) |
| **Aberration correction** | NONE (geometric states, not light-time) |

---

## 3. SPICE kernels you need to also have

The trajectory file alone isn't enough — you need NASA's standard
ephemerides for the bodies it references. Pull from
`https://naif.jpl.nasa.gov/pub/naif/generic_kernels/`:

| Kernel | Purpose |
|---|---|
| `lsk/naif0012.tls` | Leapseconds (any 0012 or later works) |
| `spk/planets/de430.bsp` | Sun + planet ephemerides |
| `pck/gm_de431.tpc` | Gravitational parameters |
| `pck/pck00010.tpc` | Planetary constants (radii, etc.) |
| Asteroid SPK files for **PARTHENOPE, PSYCHE, THEMIS** | — pull from JPL Horizons, see §6 |

Plus the two files in this package (`.bsp` and `.tpc`).

---

## 4. Five-line replication test (Python)

```python
import spiceypy

# 1. Load NASA standard kernels (you must have these locally)
spiceypy.furnsh('naif0012.tls')
spiceypy.furnsh('de430.bsp')
spiceypy.furnsh('gm_de431.tpc')
spiceypy.furnsh('pck00010.tpc')

# 2. Load this replication package
spiceypy.furnsh('PPT_LT_CHAIN_trajectory.bsp')
spiceypy.furnsh('PPT_LT_CHAIN_names.tpc')

# 3. Query the spacecraft state at any time during the mission
et = spiceypy.str2et('2035 JAN 01 00:00:00')
state, _ = spiceypy.spkezr('PPT-LT-CHAIN-CA', et, 'ECLIPJ2000', 'NONE', 'SUN')
print(f'r = {state[0:3]} km   v = {state[3:6]} km/s')

# (Equivalent integer-ID query, works without the names kernel:)
state, _ = spiceypy.spkezr('-1030', et, 'ECLIPJ2000', 'NONE', 'SUN')
```

If your colleague is using GMAT, STK, FreeFlyer, or any other SPICE-aware
tool, just load `PPT_LT_CHAIN_trajectory.bsp` and the standard NASA kernels —
it'll appear as a normal trajectory.

---

## 5. If you don't use SPICE

Use the **CSV** or **OEM** file. Each line gives the spacecraft state
(heliocentric ECLIPJ2000) at one epoch. Interpolate with Lagrange order ≤ 5
or Hermite for accurate intermediate-time queries.

CSV columns:
```
et_seconds_past_J2000, utc_iso, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms, mass_kg
```

OEM follows CCSDS recommendation 502.0-B-2 (the
`OBJECT_NAME = PPT-LT-CHAIN-CA`, `CENTER_NAME = SUN`,
`REF_FRAME = ECLIPJ2000` convention).

---

## 6. Asteroid SPK files for PARTHENOPE, PSYCHE, THEMIS

The trajectory references these by NAIF ID:

| Asteroid | NAIF ID | Comp. | Where to get the SPK |
|---|---|---|---|
| **PARTHENOPE** | 20000011 | S-type | JPL Horizons, target ID `2000011` |
| **PSYCHE**     | 20000016 | X/M-type | JPL Horizons, target ID `2000016` |
| **THEMIS**     | 20000024 | C-type | JPL Horizons, target ID `2000024` |

To pull SPK files for the asteroids:

1. Go to https://ssd.jpl.nasa.gov/horizons/app.html
2. Settings → Ephemeris Type → "SPK File"
3. Target Body → enter `2000011` (or `2000016`, `2000024`)
4. Time span → at least `2029-07-28 → 2042-01-01` (or wider)
5. Submit — download the `.bsp`

(If you have access to the parent repo's `NOTABLE_ASTEROID_BSPs/` directory,
those are exactly these three asteroid BSPs — just copy them.)

---

## 7. To re-run the optimization yourself

The runner and all supporting modules are bundled in `optimization/`. The
runner:

- Locks the launch date to **2029-07-29 04:56:36 UTC** (matches v2)
- Locks ordering to PARTHENOPE → PSYCHE → THEMIS, direct (no Mars flyby)
- Adds a soft penalty steering the Themis arrival toward the
  **2041-04-25 Psyche-Themis close approach** (sep min = 0.144 AU)
- Allows LT cruise as short as 1 yr per leg (the SF solver converges
  cleanly on a 1.34-yr Psyche→Themis transfer with throttle peak |u| = 0.42)
- Searches all 8 m_revs combos × 10 DE seeds = 80 surrogate runs, then runs
  full Sims-Flanagan verification on the winner

The full saved log (`optimization/run.log`, ~13 KB) records every DE run,
the surrogate winner, the SF verification, and the side-by-side comparison
against v2.

To reproduce from scratch:

1. Set up Python 3.11 + the dependencies the modules need
   (`pykep`, `spiceypy`, `scipy`, `numpy`).
2. Download the SPICE kernels listed in §3 + the three asteroid BSPs
   (place them so the runner can find them — see the parent repo's
   `gcp_config.py` and `core.load_kernels()` for the expected layout).
3. Run:

   ```bash
   python3 optimization/run_ppt_lt_close_approach.py
   ```

4. To re-verify physics:
   ```bash
   python3 verification/verify_ppt_lt_chain_full.py \
           optimization/ppt_lt_chain_close_approach.pkl
   ```
5. To re-check project-level constraints:
   ```bash
   python3 verification/check_constraints.py
   ```

The full optimization takes ~9 minutes on a 12-vCPU machine.

To regenerate just the data products (BSP / OEM / CSV / JSON / TPC) and
figures from the saved pkl, without re-running the optimizer:

```bash
python3 tools/export_ppt_trajectory.py \
        optimization/ppt_lt_chain_close_approach.pkl . -1030 PPT-LT-CHAIN-CA
python3 tools/plot_ppt_lt_chain_thrust.py \
        optimization/ppt_lt_chain_close_approach.pkl thrust_vs_time.png
python3 tools/plot_ppt_lt_chain_gif.py \
        optimization/ppt_lt_chain_close_approach.pkl trajectory_3d.gif
python3 tools/plot_ppt_lt_chain_gif_2d.py \
        optimization/ppt_lt_chain_close_approach.pkl trajectory_2d.gif
```

---

## 8. Optimization configuration (frozen; what produced this trajectory)

| Constraint | Value |
|---|---|
| Mission shape | Earth → A1 → A2 → A3 (direct, no flyby) |
| Launch Δv (impulsive) | ≤ 7.0 km/s, **excluded from objective** |
| Post-launch propulsion | Sims-Flanagan electric only, Isp 3100 s, ≤ 0.30 N thrust |
| Mission duration cap | 30 years |
| Stay duration | [3 months, 12 months] at each asteroid |
| LT leg TOF | [1 year, 8 years] per leg |
| LT chain initial mass | 1500 kg |
| Spacecraft launch mass | 3000 kg |
| Themis-arrival soft window | ±1.5 yr around 2041-04-25 close approach |

The optimizer used `scipy.optimize.differential_evolution` with 10 random
seeds × 8 Lambert m-revs combinations = 80 DE runs total. The winner is
direct architecture, Lambert m_revs = `(1, 1, 0)` for
(Earth→PARTHENOPE, PARTHENOPE→PSYCHE, PSYCHE→THEMIS) respectively.

The DE population was seeded with:
- 10% jittered v2-anchor points (proven feasible region)
- 60% close-approach-biased points (target arrival ≈ 2041-04-25)
- 30% uniform random across bounds

---

## 9. Per-leg breakdown

| Leg | TOF (yr) | Lambert m | Δv integrated (km/s) | m_in (kg) | m_out (kg) | Peak thrust (mN) | Duty cycle |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Earth → PARTHENOPE  | 3.74 | 1 | 6.173 | 1500.0 | 1224.3 | 144.9 | 100% |
| PARTHENOPE → PSYCHE | 6.81 | 1 | 2.867 | 1224.3 | 1114.1 |  32.5 |  33% |
| PSYCHE → THEMIS     | 1.34 | 0 | 3.710 | 1114.1 |  986.2 | 127.0 | 100% |

The two short, high-throttle legs (Earth→Parthenope and Psyche→Themis) run
at 100% duty but max |u| ≤ 0.483, so the engine still has thrust margin
(48% of 0.30 N at peak). The middle leg cruises lazily at ~33% duty.

See `thrust_vs_time.png` for the full segment-level profile.

---

## 10. Constraint-compliance audit

Two independent audits ship with the package, both fully passing:

### Sims-Flanagan re-derivation (145/145 checks pass)

`verification/audit_145_checks.md` — re-runs every leg from raw SPICE
states, recomputes launch Δv, Lambert solves, mass evolution, and integrated
Δv. Highlights:

- Launch Δv re-derived from raw SPICE: 7.000001 km/s (matches saved to 1 mm/s)
- Σ per-leg integrated Δv == saved post-launch Δv: 12.749950 = 12.749950
- Each LT leg re-integrated independently (Sims-Flanagan, 15 segments):
  pos error ≤ 50 km, vel error ≤ 1.3×10⁻⁵ km/s, all converged
- Tsiolkovsky predictions agree with integrator outputs to nanometer-level
- Throttle |u| max = 0.483 (well below 1.0 cap)
- Heliocentric distance bounds: 1.02–2.87 AU (sane)
- All 6 epochs within BSP coverage windows

Regenerate with `python3 verification/verify_ppt_lt_chain_full.py optimization/ppt_lt_chain_close_approach.pkl`.

### Project-level constraint audit (36/36 checks pass)

`verification/check_constraints.py` — verifies every bound in
`CONSTRAINTS_AND_OUTPUTS.md` (launch Δv ≤ 7, mission ≤ 30 yr, stays in
[3, 12] mo, LT TOF in [1, 8] yr, throttle |u| ≤ 1, thrust ≤ 0.30 N, mass
strictly decreases, etc.). Run with `python3 verification/check_constraints.py`.

---

## 11. What this package does NOT include

- The launch vehicle's pad-to-LEO trajectory — handled by the launcher's
  own analysis (Falcon Heavy / SLS / Vulcan). The 7 km/s launch Δv is
  delivered to the spacecraft as an instantaneous post-separation v_∞
  at Earth's sphere-of-influence boundary.
- Asteroid arrival / departure orbital insertion — the trajectory ends with
  the spacecraft matching the asteroid's heliocentric velocity (rendezvous);
  parking-orbit design is downstream.
- Power-system mass model — assumes thrust is always available.
- Sun-pointing / attitude constraints on the solar arrays.
- Operational margins — peak thrust hits 48% of the 0.30 N cap; real
  ops would re-derive with reserves.
- N-body perturbations / J2 effects — pure two-body around Sun.

---

## 12. Contact / project info

- Project: Ae105c (Caltech / Pomona College spring term)
- Repository: https://github.com/<your-username>/Ae105c-Project
- Trajectory revision: `ppt_lt_chain_close_approach.pkl`
- Generated by `Python_Consolidated/export_ppt_trajectory.py`

If you find a discrepancy between the SPK and the OEM/CSV, the SPK is
authoritative — the others are derived from the same underlying integration.

---

## 13. Quick sanity-check expected values

If your replication is correct, querying the spacecraft at the launch
event should give you a state on Earth's heliocentric trajectory:

```python
spiceypy.furnsh('PPT_LT_CHAIN_trajectory.bsp')
spiceypy.furnsh('PPT_LT_CHAIN_names.tpc')
et = spiceypy.str2et('2029-07-29T04:56:36')
state, _ = spiceypy.spkezr('-1030', et, 'ECLIPJ2000', 'NONE', 'SUN')
# Expected: spacecraft co-located with Earth at launch, with v = v_Earth + v_∞
earth_state, _ = spiceypy.spkezr('399', et, 'ECLIPJ2000', 'NONE', 'SUN')
# state[:3] should match earth_state[:3] within ~6,400 km (Earth radius);
# state[3:] = earth_state[3:] + v_∞_launch_vector with |v_∞| = 7.000 km/s
```

At the THEMIS arrival event (`2041-12-20T15:31:04`), the spacecraft state
should match THEMIS's heliocentric state within ~50 km (the LT solver's
rendezvous residual on the close-approach leg).
