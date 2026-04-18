# Earth Gravity Assist — Implementation Plan

Quick-win extension before tackling low-thrust. Adds Earth flyby as a 4th architecture alongside direct / Moon GA / Mars GA.

## What an Earth GA looks like

`Earth launch → 1-3 year heliocentric loop → Earth flyby → A1 → A2 → A3`

The 1-3 year loop lets the spacecraft swing back to Earth with non-zero v∞, which the Earth flyby rotates into a trajectory aimed at the asteroid belt. Heritage: Dawn (Mars GA + ion), Hayabusa2 (EGA + ion), MESSENGER (multiple EGAs), Galileo (Earth×2 + Venus).

## Code changes (minimal)

### 1. `optimization.py:192` — add `'earth'` entry

```python
FLYBY_BODIES = {
    'moon':  {'id': '301', 'mu_body': 301, 'radii_body': 301, 'min_alt': 100,
              'tof_min': 1*DAY,   'tof_max': 10*DAY},
    'mars':  {'id': '4',   'mu_body': 4,   'radii_body': 499, 'min_alt': 200,
              'tof_min': 0.3*YEAR,'tof_max': 3*YEAR},
    'earth': {'id': '399', 'mu_body': 399, 'radii_body': 399, 'min_alt': 300,
              'tof_min': 1.0*YEAR,'tof_max': 3.0*YEAR},   # NEW
}
```

Rationale: Earth radius ≈ 6378 km, 300 km min altitude is standard safe flyby (below 200 km hits atmospheric drag, above 500 km throws away leverage). `tof_min` ≥ 1 yr because any shorter and the spacecraft hasn't looped back to Earth's position.

### 2. `compute_path_with_flyby` — multi-rev Lambert for E→E leg

Earth-to-Earth with TOF ≈ 1 yr needs **m=1** (one revolution) in Lambert; at 2 yr maybe m=1 or m=2. Current code hardcodes `m_0`. Fix: for Earth flybys, use `solve_lambert_best` (already handles m=0,1,2 + both directions).

At `optimization.py:226`, change:

```python
e_lv, fb_arr_lv, ef0 = solve_lambert(earth_r, flyby_r,
                                      (et_flyby - et_launch) / DAY, m_0, MU_SUN)
```

to:

```python
if flyby_name == 'earth':
    e_lv, fb_arr_lv, ef0 = solve_lambert_best(earth_r, flyby_r,
                                               (et_flyby - et_launch) / DAY, MU_SUN)
else:
    e_lv, fb_arr_lv, ef0 = solve_lambert(earth_r, flyby_r,
                                          (et_flyby - et_launch) / DAY, m_0, MU_SUN)
```

Import: add `solve_lambert_best` to the top-of-file import at `optimization.py:21`.

### 3. `optimize_times_flyby_quick` — EGA-aware initial guess

At `optimization.py:316`, the coarse evaluator hardcodes flyby-time guesses per body. Add Earth branch:

```python
if flyby_name == 'moon':
    flyby_tofs = [3*DAY/YEAR, 5*DAY/YEAR]
elif flyby_name == 'mars':
    flyby_tofs = [0.6, 1.2]
elif flyby_name == 'earth':
    flyby_tofs = [1.1, 1.5, 2.0]         # NEW: 1-2 yr loop is sweet spot
```

### 4. `optimize_best_architecture` — include Earth in fine pass

At `optimization.py:359`, the loop iterates `['moon', 'mars']`. Change to `['moon', 'mars', 'earth']`. Return value already handles arbitrary architecture names.

### 5. GCP runners — add Earth to coarse screening

In `gcp/run_73ast_full.py:36-44` and `gcp/run_science_priority.py:37-44`, the coarse pass only tries direct + Mars. Add an Earth block:

```python
dv_earth = 1e3
for lf in [0.1, 0.4, 0.7]:
    for et_tof in [1.2, 1.8, 2.5]:       # Earth loop TOF in years
        for tof in [2.0, 3.5]:
            x = np.array([lf*(launch_dates[1]-launch_dates[0])/YEAR,
                          et_tof, tof, 0.4, tof, 0.4, tof])
            dv = score_paths_flyby(x, a1_id, a2_id, a3_id, launch_dates,
                                   'earth', 0, 0, 0, 0)
            if dv < dv_earth:
                dv_earth = dv
```

Then `best_dv = min(dv_direct, dv_mars, dv_earth)` and `best_arch` chosen accordingly.

## Testing

1. **Local smoke test**: evaluate one known triplet (dv-min winner: Hertha → Polyxo → Alkeste) with `architecture='earth'`. Verify no crashes and dv is finite.
2. **Local sanity check**: verify Earth GA beats direct on at least one triplet where Mars doesn't work (e.g., inner-belt targets).
3. **GCP full run**: `run_73ast_full.py` and `run_science_priority.py` on all 14,040 diverse triplets with 4 architectures. Compare new top-15 against current Mars-dominated list.

## Expected outcome

- Current winner (Hertha → Polyxo → Alkeste, Mars GA, 9.40 km/s) may hold. Mars is already a strong assist for main-belt rendezvous.
- EGA likely wins for **lower-Δv asteroids** that don't need the big Mars kick, and for **synodic alignments** where Mars is poorly placed.
- Expected EGA win rate in top-15: **maybe 2-5 entries**. The big win comes from combining EGA + Mars (ΔV-EGA-Mars chain), which requires a 5-leg architecture — deferred to the low-thrust phase.

## Risk items

- **m=0 fallback for Earth**: if `solve_lambert_best` fails (spacecraft can't physically loop back in the given TOF), it returns exitflag=-1 and `compute_path_with_flyby` already returns a 1e3 penalty. No explicit handling needed.
- **Earth-return launch dv**: expect 3-4 km/s launch v∞ for a useful EGA. The optimizer will naturally find this.
- **Flyby altitude too low**: `compute_flyby_dv` (pykep `fb_dv`) returns penalty if the required bend angle exceeds what altitude allows. No new handling needed.
- **Unit test**: add `assert get_mu('399') > 3.98e5 and get_mu('399') < 4.00e5` at test time to catch any future mu-typo regressions (like the Moon bug).
