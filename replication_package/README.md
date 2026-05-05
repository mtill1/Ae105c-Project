# Replication Package — PARTHENOPE → PSYCHE → THEMIS LT-Chain

Everything you need to **independently reproduce, query, and verify** the
selected mission trajectory.

The trajectory is provided in **four formats** so any tool can ingest it:

| File | Format | Use case |
|---|---|---|
| `PPT_LT_CHAIN_trajectory.bsp` | SPICE SPK Type 9 binary | Native SPICE — load with `spiceypy.furnsh` |
| `PPT_LT_CHAIN_names.tpc`      | SPICE text kernel       | Maps the spacecraft name to NAIF ID -1029 |
| `PPT_LT_CHAIN_trajectory.oem` | CCSDS OEM v2.0 ASCII   | Inter-agency standard, no SPICE needed |
| `PPT_LT_CHAIN_trajectory.csv` | CSV                     | Lowest common denominator — Excel, MATLAB, anything |
| `PPT_LT_CHAIN_metadata.json`  | JSON                    | Machine-readable mission summary + constraints |
| **`README.md`** | this file | Human-readable overview |

The trajectory has **959 state samples** at ~6.4 day spacing across the full
16.79-year mission. Position/velocity round-trip through SPICE with **0 km
error** at sample points (sub-meter at intermediate times via 5th-order
Lagrange interpolation).

---

## 1. What is this trajectory?

A **direct, all-low-thrust** mission visiting three main-belt asteroids:

```
Earth (launch, impulsive ≤ 7 km/s)
   ↓
PARTHENOPE [S-type, NAIF 20000011]   — 6.4 month stay
   ↓                                    (electric LT cruise)
PSYCHE     [X/M-type, NAIF 20000016] — 12 month stay
   ↓                                    (electric LT cruise)
THEMIS     [C-type, NAIF 20000024]   — final rendezvous
```

| | |
|---|---|
| Total mission | **16.79 years** (2029-07-28 → 2046-05-15) |
| Launch Δv (paid by launcher) | 7.000 km/s |
| Post-launch Δv (sum of LT integrated) | 13.105 km/s |
| Spacecraft launch mass | 3000 kg (delivered by launcher) |
| LT-chain initial mass | 1500 kg (start of post-launch electric phase) |
| Final delivered mass | 974.7 kg (65% of 1500 kg LT start) |
| Engine | Solar electric, Isp 3100 s, max thrust 0.30 N |

**Selected because:** it's the only architecture/ordering that achieved
post-launch Δv < 14 km/s among the 12 (PARTHENOPE+PSYCHE+THEMIS) ordering ×
architecture combinations evaluated. All checks (146 of them) pass an
independent physics audit — see the verification report in the parent repo
at `docs/verification_ppt_lt_chain.md`.

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
state, _ = spiceypy.spkezr('PPT-LT-CHAIN', et, 'ECLIPJ2000', 'NONE', 'SUN')
print(f'r = {state[0:3]} km   v = {state[3:6]} km/s')

# (Equivalent integer-ID query, works without the names kernel:)
state, _ = spiceypy.spkezr('-1029', et, 'ECLIPJ2000', 'NONE', 'SUN')
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

OEM follows CCSDS recommendation 502.0-B-2 (the `OBJECT_NAME = PPT-LT-CHAIN`,
`CENTER_NAME = SUN`, `REF_FRAME = ECLIPJ2000` convention).

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
4. Time span → at least `2029-07-28 → 2046-05-15` (or wider)
5. Submit — download the `.bsp`

Each asteroid SPK is ~1 MB. Together with the planetary `de430.bsp` and the
kernels listed in §3, that's the full kernel set needed.

(If you have access to the parent repo's `NOTABLE_ASTEROID_BSPs/` directory,
those are exactly these three asteroid BSPs — just copy them.)

---

## 7. To re-run the optimization yourself (instead of just reading the result)

The trajectory was computed by `Python_Consolidated/gcp/run_ppt_lt_chain.py`
in the parent repository. To reproduce:

1. Clone the repo (https://github.com/<...>/Ae105c-Project)
2. Set up Python 3.11 + dependencies (`pip install -r Python_Consolidated/requirements.txt`)
3. Download the SPICE kernels listed in §3 + the three asteroid BSPs
4. Run:

   ```bash
   python3 Python_Consolidated/gcp/run_ppt_lt_chain.py
   ```

   This tries all 6 orderings × {direct, Mars-flyby} = 12 architectures,
   picks the lowest post-launch Δv, runs full Sims-Flanagan verification on
   the winner, and saves the result to
   `optimal_asteroid_paths/pkl/ppt_lt_chain_v2.pkl`.

5. To verify physics: `python3 Python_Consolidated/verify_ppt_lt_chain_full.py`

The full optimization takes ~5 minutes on a 12-vCPU machine (~30 minutes
on a typical laptop with the supplied DE config).

---

## 8. Optimization configuration (frozen; what produced this trajectory)

| Constraint | Value |
|---|---|
| Mission shape | Earth → optional flyby → A1 → A2 → A3 |
| Launch Δv (impulsive) | ≤ 7.0 km/s, **excluded from objective** |
| Post-launch propulsion | Sims-Flanagan electric only, Isp 3100 s, ≤ 0.30 N thrust |
| Mission duration cap | 30 years |
| Stay duration | ≥ 3 months at each asteroid |
| Composition diversity | Required (C + S + X/M) |
| Flyby physics | Ballistic only when used (none in this winning trajectory) |
| LT chain initial mass | 1500 kg |
| Spacecraft launch mass | 3000 kg |

The optimizer used `scipy.optimize.differential_evolution` with 6 random
seeds × 5 Lambert m-revs × 2 architectures = 60 DE runs total. The winning
combination is direct architecture, Lambert m_revs = `(1, 1, 0)` for
(Earth→PARTHENOPE, PARTHENOPE→PSYCHE, PSYCHE→THEMIS) respectively.

---

## 9. What this package does NOT include

- The launch vehicle's pad-to-LEO trajectory — handled by the launcher's
  own analysis (Falcon Heavy / SLS / Vulcan). The 7 km/s launch Δv is
  delivered to the spacecraft as an instantaneous post-separation v_∞
  at Earth's sphere-of-influence boundary.
- Asteroid arrival / departure orbital insertion — the trajectory ends with
  the spacecraft matching the asteroid's heliocentric velocity (rendezvous);
  parking-orbit design is downstream.
- Power-system mass model — assumes thrust is always available.
- Sun-pointing / attitude constraints on the solar arrays.
- Operational margins — the 0.30 N thrust cap is hit by 48% at most; real
  ops would re-derive with reserves.
- N-body perturbations / J2 effects — pure two-body around Sun.

---

## 10. Contact / project info

- Project: Ae105c (Caltech / Pomona College spring term)
- Repository: https://github.com/<your-username>/Ae105c-Project
- Trajectory revision: `ppt_lt_chain_v2.pkl` (post-flyby-physics-fix, post-LT-chain-rework)
- Generated by `Python_Consolidated/export_ppt_trajectory.py`

If you find a discrepancy between the SPK and the OEM/CSV, the SPK is
authoritative — the others are derived from the same underlying integration.

---

## 11. Quick sanity-check expected values

If your replication is correct, querying the spacecraft at the launch
event should give you a state on Earth's heliocentric trajectory:

```python
spiceypy.furnsh('PPT_LT_CHAIN_trajectory.bsp')
et = spiceypy.str2et('2029-07-28T21:06:34')
state, _ = spiceypy.spkezr('-1029', et, 'ECLIPJ2000', 'NONE', 'SUN')
# Expected: spacecraft is essentially co-located with Earth at launch
earth_state, _ = spiceypy.spkezr('399', et, 'ECLIPJ2000', 'NONE', 'SUN')
# state[:3] should match earth_state[:3] within ~6,400 km (Earth radius);
# state[3:] = earth_state[3:] + v_∞_launch_vector with |v_∞| = 7.000 km/s
```

At the THEMIS arrival event (`2046-05-15T00:40:56`), the spacecraft state
should match THEMIS's heliocentric state within ~8 km (the LT solver's
rendezvous residual).
