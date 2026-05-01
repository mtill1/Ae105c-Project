# Tutorial 06 — Audit and inspect a saved trajectory

**Two related tools:**
- **`verify`** — pass/fail audit of flyby physics (is this trajectory flyable?)
- **`inspect`** — comprehensive per-leg dump (Lambert V1/V2 vectors, Δv vectors, full flyby diagnostics)

**Time:** ~30 s each. **No GCP required.**

## Why this matters

Lambert solvers don't enforce physical flyby altitude. The optimizer can find a
trajectory that looks great on paper but requires the spacecraft to dip
*through* Mars's surface — these solutions are mathematically valid Lambert arcs
but physically impossible. Always audit before reporting a result.

## The command

```bash
python Python_Consolidated/main.py verify <pkl_file> [--rank N | --names A B C]
```

## Example

```bash
python Python_Consolidated/main.py verify diverse_top3_feasible.pkl --rank 1
```

Output:

```
Flyby audit — VIRGINIA -> PSYCHE -> PARTHENOPE  via moon
------------------------------------------------------------
  RESULT: FEASIBLE
  v_inf_in  vector       : [+0.8248, +0.6972, +0.0477] km/s
  v_inf_out vector       : [+0.4335, +7.5297, -0.1440] km/s
  |v_inf_in|             : 1.0810 km/s
  |v_inf_out|            : 7.5436 km/s
  Energy residual        : +6.46e+00 km/s
  Turn angle (required)  : 46.628°
  Turn angle (max @ safe): 46.628°
  Periapsis altitude     : 100 km (min allowed: 100 km)
  Moon surface radius    : 1737.4 km
```

## What each line means

| Line | Meaning |
|---|---|
| `RESULT` | `FEASIBLE` if the trajectory is physically achievable; `INFEASIBLE` if not. |
| `v_inf_in/out vector` | Spacecraft hyperbolic excess velocity at the flyby, expressed in the heliocentric ECLIPJ2000 frame. |
| `|v_inf|` magnitudes | If equal → unpowered (ballistic) flyby. If different → spacecraft makes a powered burn at periapsis (Oberth maneuver). |
| `Energy residual` | `|v_inf_out| − |v_inf_in|` in km/s. Zero ⇒ pure gravity assist. Non-zero ⇒ powered. |
| `Turn angle (required)` | The angle between v_inf_in and v_inf_out — what the trajectory needs the flyby body to bend by. |
| `Turn angle (max @ safe)` | The maximum bend Mars/Moon can deliver at the minimum-safe periapsis radius (200 km altitude for Mars, 100 km for the Moon). Computed from `arcsin(1/(1 + r·v²/μ))` summed for the in and out asymptotes. |
| `Periapsis altitude` | The closest altitude the spacecraft reaches above the body's surface. Negative ⇒ goes underground (impossible). |

## How to interpret the result

A valid (feasible) flyby must pass **both** checks:
1. **Geometric:** turn angle ≤ natural max at safe periapsis (no sub-surface flyby)
2. **Ballistic:** `|v_inf_in| ≈ |v_inf_out|` within 0.05 km/s (no powered burn at periapsis)

| What you see | What it means |
|---|---|
| `FEASIBLE`, both checks OK, periapsis > 200 km buffer | ✅ comfortable margin, real-world flyable |
| `FEASIBLE`, both checks OK, altitude ≈ minimum | ⚠️ right at the geometric limit. Will need re-tuning for navigation tolerances in real ops. |
| `INFEASIBLE` (geometric) | ✗ trajectory requires going through the body's surface. |
| `INFEASIBLE` (ballistic, large `|v_inf|` residual) | ✗ Mars/Moon would need a large powered Δv at periapsis. Not a real GA. |
| `lambert_fail` | Optimizer's saved epochs don't produce a converged Lambert solution. The pkl may be corrupt. |

## `inspect` — full per-leg dump

For mission-design-level detail (every Lambert velocity vector, every Δv as a
3-component vector, full flyby diagnostics), use:

```bash
python Python_Consolidated/main.py inspect <pkl_file> [--rank N]
```

Example (the PARTHENOPE → PSYCHE → THEMIS winner):

```bash
python Python_Consolidated/main.py inspect mars_diverse_science_a40_all_PSYCHE_THEMIS.pkl --rank 1
```

What you get (per leg):
- TOF in days and years, Lambert m-revs
- Body velocity vector at start (heliocentric)
- Lambert V1 (departure) vector
- Δv at departure (vector + magnitude)
- Lambert V2 (arrival) vector
- Body velocity vector at end
- Δv at arrival (vector + magnitude)

For the flyby leg (Mars or Moon):
- Full v_inf_in and v_inf_out vectors (heliocentric ECLIPJ2000)
- Magnitudes and energy residual (ballistic vs powered classification)
- Required turn angle and the natural maximum at safe periapsis
- Turn margin in degrees and percent
- Periapsis altitude

Plus a per-burn Δv breakdown table summing to the saved total.

This is what you'd put in a mission-design report or a PDR slide.

Inspect top 5 at once:

```bash
python Python_Consolidated/main.py inspect <pkl> --top 5
```

## Bulk-audit a whole result file

For a full mass-Pareto sweep, audit all top entries:

```bash
for rank in 1 2 3 4 5 6 7 8 9 10; do
  echo "=== rank $rank ==="
  python Python_Consolidated/main.py verify results_mass_pareto_<TS>.pkl --rank $rank
done
```

(Tip: this is exactly how I caught that the previous mass-Pareto top-15 results
all had non-physical Mars flybys.)

## What's next

- **`02_diverse_csm.md`** — re-run an optimization with the physical-flyby
  check baked in
- **`FAQ.md`** — common errors
