# Tutorial 02 — Composition-diverse mission with physical-flyby check

**Question this tutorial answers:** what's the lowest-fuel mission that visits
**one C-type, one S-type, and one X/M-type asteroid** (covering the three main
main-belt taxonomies for a strong science return), AND has a Mars/Moon flyby
that's geometrically achievable in real life?

**Time:** ~10 min on a laptop. **No GCP required.**

This is the **recommended entry point** for a publication-quality result —
it's the only workflow that physically verifies every flyby's altitude.

## Pre-requisites

- Tutorial 00 finished
- `asteroid_tradeoff.csv` exists in repo root (it ships with the repo)

## The command

```bash
python Python_Consolidated/main.py optimize --feasible
```

What it does:

1. **Stage 1** (≈4 min): coarse architecture screen on every C+S+X/M triplet
   (~14,000 of them) using a sampling-based pre-screen. Parallelized 12-way.
2. **Stage 2** (≈1 min): full Differential Evolution on the top 25, evaluating
   direct, Moon-flyby, and Mars-flyby architectures for each.
3. **Stage 3** (≈30 s): independent geometric audit of every flyby —
   re-derives v_inf vectors from a fresh Lambert solve and checks whether the
   required turn angle fits within Mars/Moon's natural maximum at the safe
   periapsis altitude.

## Expected output

You'll see a table like this:

```
  rank triplet                                arch     dv  feas
   1   VIRGINIA   ->PSYCHE     ->PARTHENOPE   moon  12.38  OK
   2   HERTHA     ->POLYXO     ->ALKESTE      moon  12.50  OK
   3   MASSALIA   ->MISA       ->PSYCHE       moon  12.78  OK
   ...

Top 3 with feasible flyby:
  #1  VIRGINIA [C] -> PSYCHE [X/M] -> PARTHENOPE [S]  (moon)  dv=12.383 km/s
  #2  HERTHA [X/M] -> POLYXO [C] -> ALKESTE [S]      (moon)  dv=12.504 km/s
  #3  MASSALIA [S] -> MISA [C] -> PSYCHE [X/M]       (moon)  dv=12.775 km/s

Saved: optimal_asteroid_paths/pkl/diverse_top3_feasible.pkl
```

If you see `Saved:`, ✅ it worked.

## Reading the result

```bash
python Python_Consolidated/main.py plot diverse_top3_feasible.pkl
```

Audit the #1 result's flyby physics:

```bash
python Python_Consolidated/main.py verify diverse_top3_feasible.pkl --rank 1
```

You'll get full v_inf vectors, turn angle vs. natural max, periapsis altitude,
and energy conservation residual.

## Why "feasible" matters

A previous optimization run found a "9.40 km/s" Hertha→Polyxo→Alkeste
solution that turned out to be **non-physical** — the trajectory required the
spacecraft to dip 3,300 km **below Mars's surface**. The patched optimizer
rejects such geometries. Real flyable answers are 12–14 km/s, not 9 km/s.

## Pinning specific asteroids to the path

If you want particular asteroids guaranteed in every triplet (e.g. Decadal
targets like THEMIS or PSYCHE), use the `gcp/run_mars_diverse_science.py`
runner with environment variables. **Local OK if the search space is small**
(< 100 candidate triplets):

```bash
# At least one of THEMIS or PSYCHE must be in the path
ALPHA=0.4 REQUIRED_ASTEROIDS=THEMIS,PSYCHE \
  python Python_Consolidated/gcp/run_mars_diverse_science.py

# BOTH THEMIS and PSYCHE must be in the path
ALPHA=0.4 REQUIRE_ALL_ASTEROIDS=THEMIS,PSYCHE \
  python Python_Consolidated/gcp/run_mars_diverse_science.py
```

`ALPHA` blends Δv vs science: 1.0 = pure Δv, 0.7 = 70% Δv + 30% science,
0.4 = 40% Δv + 60% science.

## What's next

- **`05_visualize.md`** — render the winner as a 3D GIF
- **`06_verify_physics.md`** — `verify` (pass/fail) and `inspect` (full per-leg
  Lambert/Δv/flyby dump) for any saved trajectory
- **`03_mass_pareto_gcp.md`** — if you want to also optimize *propellant mass*
  (not just Δv), trade chemical vs. electric across architectures
