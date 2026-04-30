# Three-Asteroid Main Belt Rendezvous — Trajectory Optimization

## What This Is

This project designs the cheapest-fuel trajectory for a spacecraft that
launches from Earth and visits **three main-belt asteroids** in sequence. It
tries every reasonable launch date, transfer time, and stay time at each
asteroid, picks the best propulsion architecture (chemical vs electric per
leg), audits Mars/Moon gravity-assist physics, and reports the path that
delivers the most spacecraft mass to the final asteroid.

Built for Ae105c at Caltech / Pomona College.

Supports:
- **Direct transfer:** Earth → A1 → A2 → A3
- **Lunar gravity assist:** Earth → Moon flyby → A1 → A2 → A3
- **Mars gravity assist:** Earth → Mars flyby → A1 → A2 → A3
- **Hybrid chemical + electric propulsion** across 8 architectures (CCC, CCE, ..., EEE)
- **Composition-diverse** missions (one C, one S, one X/M asteroid)

---

## Where to start

| If you want to... | Go here |
|---|---|
| Set things up the first time | [`Tutorials/00_setup.md`](Tutorials/00_setup.md) |
| Find the lowest-fuel mission | [`Tutorials/01_minimum_dv.md`](Tutorials/01_minimum_dv.md) |
| Find a science-balanced C+S+X/M mission | [`Tutorials/02_diverse_csm.md`](Tutorials/02_diverse_csm.md) |
| Trade chemical vs electric across triplets (GCP) | [`Tutorials/03_mass_pareto_gcp.md`](Tutorials/03_mass_pareto_gcp.md) |
| Optimize 3 specific asteroids you picked | [`Tutorials/04_single_triplet.md`](Tutorials/04_single_triplet.md) |
| Render a 3D trajectory animation | [`Tutorials/05_visualize.md`](Tutorials/05_visualize.md) |
| Audit a saved trajectory's flyby physics | [`Tutorials/06_verify_physics.md`](Tutorials/06_verify_physics.md) |
| Hit an error | [`Tutorials/FAQ.md`](Tutorials/FAQ.md) |
| Understand the algorithms | [`METHODOLOGY.md`](METHODOLOGY.md) |

---

## The single CLI

Everything goes through one entry point:

```bash
python Python_Consolidated/main.py --help
```

Subcommands:

| Subcommand | What it does |
|---|---|
| `optimize`           | Run trajectory optimization (many flags — see below) |
| `list`               | List all saved result `.pkl` files |
| `plot RESULT.pkl`    | Show top entries / render PNG / render GIF |
| `verify RESULT.pkl`  | Audit a saved trajectory's flyby physics |
| `rank`               | Rebuild `asteroid_tradeoff.csv` from JPL SBDB data |
| `animate-asteroids`  | MP4 of all 69 asteroid orbits |

`optimize` modes (combine flags as documented in the tutorials):

```bash
python Python_Consolidated/main.py optimize                # two-level Δv (default)
python Python_Consolidated/main.py optimize --science 0.7  # 70% Δv + 30% science
python Python_Consolidated/main.py optimize --diverse      # require C+S+X/M
python Python_Consolidated/main.py optimize --feasible     # diverse + flyby physics audit ★
python Python_Consolidated/main.py optimize --beam 15      # beam search
python Python_Consolidated/main.py optimize --pareto       # mass-Pareto across 8 archs
```

★ `--feasible` is the **recommended** workflow for a publication-quality
result (it's the only mode that physically verifies every Mars/Moon flyby).

---

## Project layout

```
Ae105c-Project/
├── README.md                # this file (entry point)
├── METHODOLOGY.md           # algorithm reference
├── CLAUDE.md                # AI-assistant guidance
├── Tutorials/               # 7 task-focused walkthroughs + FAQ
├── Python_Consolidated/     # Active codebase (7 .py files)
│   ├── main.py              # CLI entry point — all user workflows
│   ├── core.py              # Lambert/SPICE/flyby primitives
│   ├── optimization.py      # Δv scoring + DE + composition-diverse search
│   ├── lowthrust.py         # Sims-Flanagan low-thrust solver
│   ├── mass_optimization.py # Mass-Pareto across 8 propulsion architectures
│   ├── visualization.py     # 3D trajectory animation + plotting
│   ├── tradeoff.py          # Asteroid science scoring
│   ├── requirements.txt
│   └── gcp/                 # GCP cloud-runner scripts (3 files)
├── NOTABLE_ASTEROID_BSPs/   # 69 asteroid SPICE ephemerides
├── SPICE_BSPs/              # Extended pool (49 BSPs)
├── optimal_asteroid_paths/  # Saved optimization results (.pkl)
├── Renders/                 # Generated images, GIFs, MP4s
├── asteroid_tradeoff.csv    # Ranked asteroid table
├── docs/archive/            # Historical design docs (plan.md, etc.)
└── generic_kernels/         # Symlink → ~/Documents/ae105/generic_kernels/
```

7 active Python modules, 1 CLI entry, no scattered scripts.

---

## Best results found so far

### Lowest Δv with physically feasible flyby (composition-diverse)

| Rank | Path | Total Δv | Flyby | Mission |
|---|---|:---:|:---:|:---:|
| **#1** | **VIRGINIA [C] → PSYCHE [X/M] → PARTHENOPE [S]** | **12.38 km/s** | Moon | 9.4 yr |
| #2 | HERTHA [X/M] → POLYXO [C] → ALKESTE [S] | 12.50 km/s | Moon | 8.5 yr |
| #3 | MASSALIA [S] → MISA [C] → PSYCHE [X/M] | 12.78 km/s | Moon | 8.4 yr |

All three use **Moon-Oberth maneuvers** at the 100 km altitude limit (powered
chemical burn at periapsis where Oberth efficiency peaks). See
[`Tutorials/02_diverse_csm.md`](Tutorials/02_diverse_csm.md) for how to
reproduce.

### Highest delivered mass (mass-Pareto, single triplet)

| Triplet | Architecture | Δv equivalent | Final mass |
|---|---|:---:|:---:|
| PARTHENOPE → PSYCHE → THEMIS | ECE (mixed) | 2.94 km/s | 588 kg / 1500 kg |
| FORTUNA → THEMIS → PSYCHE (legacy) | EEE (electric) | 1.62 km/s | 894 kg* |

*The 894 kg result used non-physical Mars flyby — corrected to 588 kg with
the patched optimizer. See [`METHODOLOGY.md`](METHODOLOGY.md) §9 for what
changed.

---

## Quick troubleshooting

For comprehensive answers see [`Tutorials/FAQ.md`](Tutorials/FAQ.md).

| Symptom | Quick fix |
|---|---|
| `pip install pykep` fails | You're on Python 3.13. Use 3.11. |
| `KERNELVARNOTFOUND` | Missing SPICE kernel. Re-run setup Step 3. |
| `No module named 'core'` | Run from repo root, not from `Python_Consolidated/`. |
| Optimization returns Δv = 1000 | Penalty for infeasible trajectory. Try a longer launch window. |
| Asteroid name not recognized | Names are uppercase. Run `ls NOTABLE_ASTEROID_BSPs/` to see exact names. |

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy`, `scipy` | Numerics + Differential Evolution |
| `pykep` | Lambert solver, propagation, flyby Δv (ESA) |
| `spiceypy` | NASA SPICE kernel access |
| `matplotlib` | Plotting and animation |
| `pandas` | CSV I/O for the tradeoff table |
| `tqdm` | Progress bars |
| `imageio` | GIF writer |

---

## Citation

If you use this work, please cite:

- Izzo, D. (2015). "Revisiting Lambert's problem." *Celestial Mechanics &
  Dynamical Astronomy*, 121(1):1–15.
- Storn, R. & Price, K. (1997). "Differential Evolution." *J. Global
  Optimization*, 11:341–359.
- Sims, J.A. & Flanagan, S.N. (1999). "Preliminary Design of Low-Thrust
  Interplanetary Missions." AAS/AIAA Astrodynamics Specialist Conference.

Full reference list in [`METHODOLOGY.md`](METHODOLOGY.md) §10.
