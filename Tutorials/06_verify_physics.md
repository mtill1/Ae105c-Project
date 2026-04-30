# Tutorial 06 — Audit a saved trajectory's flyby physics

**Question this tutorial answers:** is this saved trajectory actually flyable?
Does its Mars/Moon gravity assist obey the laws of physics?

**Time:** ~30 s. **No GCP required.**

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

| What you see | What it means |
|---|---|
| `FEASIBLE`, periapsis altitude > 200 km buffer | ✅ comfortable margin, real-world flyable |
| `FEASIBLE`, altitude ≈ minimum | ⚠️ right at the limit. Optimizer pushed against the constraint. Will need re-tuning for navigation tolerances in real ops. |
| `INFEASIBLE` | ✗ trajectory requires going through the body's surface. Don't trust the optimization result. |
| `lambert_fail` | Optimizer's saved epochs don't produce a converged Lambert solution. The pkl may be corrupt. |

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
