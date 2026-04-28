# Three-Asteroid Main Belt Rendezvous — Trajectory Optimization

## What This Is

This project designs the cheapest fuel trajectory for a spacecraft that launches from Earth and visits **three main-belt asteroids** in sequence. It tries every reasonable launch date, transfer time, and stay time at each asteroid, and reports the path that uses the least delta-v (fuel).

It supports:
- **Direct transfer**: Earth -> Asteroid 1 -> Asteroid 2 -> Asteroid 3
- **Lunar gravity assist**: Earth -> Moon flyby -> A1 -> A2 -> A3
- **Mars gravity assist**: Earth -> Mars flyby -> A1 -> A2 -> A3

Built for Ae105c at Caltech / Pomona College.

---

# Part 1: Step-by-step setup (for someone who has never used Python)

If you already have Python and a working terminal, skip to **[Part 2: Running the code](#part-2-running-the-code)**.

> **Heads up — only macOS / Linux are tested.** The instructions below assume macOS (the project author uses macOS). On Windows, install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) first and run all the commands inside the WSL terminal.

## Step 1: Open a terminal

On macOS, press `Cmd + Space`, type `Terminal`, and hit Enter. A black/white window appears. This is where you type all commands. Each command in this guide is written inside a grey box — copy it, paste it into the terminal, hit Enter.

To check the terminal works, type:

```bash
echo "hello"
```

You should see `hello` printed back. If so, you're good.

## Step 2: Install Python 3.11

**Important:** This project does **not** work on Python 3.13 yet, because one of the libraries it depends on (`pykep`, made by the European Space Agency) hasn't released a 3.13 version. Use Python **3.11** specifically.

Check what you have:

```bash
python3 --version
```

If you see something like `Python 3.11.x`, you're done with this step. If you see 3.12, 3.13, or get "command not found", install 3.11:

**On macOS, the easiest way is Homebrew.** First install Homebrew (skip if you already have it):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install Python 3.11:

```bash
brew install python@3.11
```

Verify:

```bash
python3.11 --version
```

You should see `Python 3.11.x`. From now on, use the command `python3.11` (not `python` or `python3`) to be sure you're using the right version.

## Step 3: Get the code

If you haven't cloned this repo yet, do that now. Pick a folder where you want it to live (your Desktop is fine), then:

```bash
cd ~/Desktop
git clone <URL_OF_THIS_REPO> Ae105c-Project
cd Ae105c-Project
```

Replace `<URL_OF_THIS_REPO>` with the GitHub URL. From now on, **every command in this guide assumes you're inside the `Ae105c-Project` folder.** If you ever close the terminal and come back, run `cd ~/Desktop/Ae105c-Project` first.

You can confirm where you are with:

```bash
pwd
```

It should print `/Users/<your-name>/Desktop/Ae105c-Project`.

## Step 4: Create a "virtual environment"

A virtual environment is a private folder of Python libraries used by this project only. This way, installing things here won't mess up other Python work on your computer.

Create one called `.venv`:

```bash
python3.11 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Your prompt should now have `(.venv)` at the front. That means the environment is active. **You need to run `source .venv/bin/activate` every time you open a new terminal for this project.**

To leave the environment later, type `deactivate`.

## Step 5: Install the Python libraries the project uses

With the virtual environment active, run:

```bash
pip install -r Python_Consolidated/requirements.txt
```

This downloads about 8 packages (numpy, scipy, pykep, spiceypy, matplotlib, tqdm, pandas, imageio). It takes a few minutes. You'll see a wall of text — that's normal.

**If `pip install pykep` fails:** see [Troubleshooting](#troubleshooting) at the bottom.

## Step 6: Download the SPICE kernels (NASA ephemeris files)

The optimizer needs to know where the planets and asteroids actually are in space at any given date. NASA publishes this data as "SPICE kernels" — small binary files. You have to download a handful of them once.

Make a folder for them. The project expects them at `~/Documents/ae105/generic_kernels/` by default (you can change this later if you want):

```bash
mkdir -p ~/Documents/ae105/generic_kernels/lsk
mkdir -p ~/Documents/ae105/generic_kernels/spk/planets
mkdir -p ~/Documents/ae105/generic_kernels/spk/satellites
mkdir -p ~/Documents/ae105/generic_kernels/pck
```

Download the five required kernels:

```bash
cd ~/Documents/ae105/generic_kernels

curl -O https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls
mv naif0012.tls lsk/

curl -O https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp
mv de430.bsp spk/planets/

curl -O https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup310.bsp
mv jup310.bsp spk/satellites/

curl -O https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de431.tpc
mv gm_de431.tpc pck/

curl -O https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc
mv pck00010.tpc pck/
```

The biggest one (`de430.bsp`) is about 115 MB and takes a minute on broadband. Now go back to the project folder:

```bash
cd ~/Desktop/Ae105c-Project
```

The project also expects a symlink called `generic_kernels` inside the project that points to the folder you just made. Create it once:

```bash
ln -sf ~/Documents/ae105/generic_kernels generic_kernels
```

You're done with setup. The asteroid ephemeris files (one BSP per asteroid) are already inside the repo at `NOTABLE_ASTEROID_BSPs/` — you don't need to download those.

## Step 7: Verify everything works

Run this one-liner:

```bash
python3.11 -c "import sys; sys.path.insert(0, 'Python_Consolidated'); from core import load_kernels; print('Loaded', len(load_kernels('NOTABLE_ASTEROID_BSPs', 'generic_kernels')), 'asteroids.')"
```

If everything is set up right, you'll see:

```
Loaded 69 asteroids.
```

If you see an error instead, jump to [Troubleshooting](#troubleshooting).

---

# Part 2: Running the code

All commands below assume:
1. You're in the `Ae105c-Project` folder (`cd ~/Desktop/Ae105c-Project`).
2. The virtual environment is active (`source .venv/bin/activate` — you should see `(.venv)` in your prompt).

## Run #1: Find the lowest-delta-v three-asteroid path (recommended first run)

This is the headline workflow. It takes about **20–40 minutes** the first time on a laptop. It evaluates roughly 300,000 candidate trajectories and reports the best one.

```bash
python3.11 -c "
import sys
sys.path.insert(0, 'Python_Consolidated')
from scripts import run_two_level_optimize
results = run_two_level_optimize()
print('Done. Results saved.')
"
```

What it does:
1. Loads all 69 asteroid orbits from the BSP files.
2. Runs a fast/coarse optimizer on every triplet of asteroids (about 300,000 of them).
3. Picks the top 50 most promising triplets.
4. Runs the full slow optimizer on those 50.
5. Returns a sorted list of the best paths and saves them to `optimal_asteroid_paths/pkl/`.

The terminal will show progress bars (that's `tqdm` — perfectly normal).

## Run #2: Same thing, but weighted toward scientifically valuable asteroids

If you'd rather visit the most scientifically interesting asteroids (even if it costs a bit more fuel), pass the science table and an `alpha` weight. `alpha=1.0` means pure delta-v. `alpha=0.7` means 70% delta-v, 30% science score.

```bash
python3.11 -c "
import sys
sys.path.insert(0, 'Python_Consolidated')
from scripts import run_two_level_optimize
results = run_two_level_optimize(science_csv='asteroid_tradeoff.csv', alpha=0.7)
"
```

## Run #3: Composition-diverse path (one C-type, one S-type, one X/M-type)

Forces the three asteroids to span the three main main-belt taxonomies (carbonaceous, silicaceous, metallic). This gives the most scientifically interesting mission.

```bash
python3.11 -c "
import sys
sys.path.insert(0, 'Python_Consolidated')
from scripts import run_diverse_optimize
results = run_diverse_optimize()
"
```

## Run #4: Beam search (faster alternative)

Builds the path one leg at a time and only keeps the top 15 partial paths at each stage. About 5x faster than two-level optimization, with slightly less thorough results.

```bash
python3.11 -c "
import sys
sys.path.insert(0, 'Python_Consolidated')
from scripts import run_beam_search
results = run_beam_search(beam_width=15)
"
```

## Run #5: Just the Mars-flyby variant

```bash
python3.11 -c "
import sys
sys.path.insert(0, 'Python_Consolidated')
from scripts import run_mars_transfer_selector
results = run_mars_transfer_selector()
"
```

## Run #6: Make a 3D animation of the asteroid orbits (no optimization)

```bash
python3.11 -c "
import sys
sys.path.insert(0, 'Python_Consolidated')
from scripts import run_graphing_notable_asteroids
run_graphing_notable_asteroids()
"
```

This creates an MP4 file in the project folder. You need `ffmpeg` installed for video output:

```bash
brew install ffmpeg
```

## Looking at the results

Every optimization run saves its output to a "pickle" file in `optimal_asteroid_paths/pkl/`. To peek inside one, paste this into the terminal:

```bash
python3.11 -c "
import pickle
with open('optimal_asteroid_paths/pkl/results_69ast_ga.pkl', 'rb') as f:
    results = pickle.load(f)
for i, j, k, r in results[:5]:
    print(f\"{r.get('path','?')}: dv = {r.get('delta_v_total',0):.2f} km/s\")
"
```

(Substitute a different filename to look at a different run — see the table below.)

### Pre-computed results that ship with the repo

Don't want to wait an hour? Several runs are already saved:

| File | What it is |
|------|------------|
| `pkl/results_69ast_ga.pkl` | Best pure delta-v path with Moon + Mars flyby options (9.40 km/s — the headline result) |
| `pkl/results_science_priority_v2.pkl` | Best path weighted 70% science, 30% delta-v |
| `pkl/results_diverse_CSM.pkl` | Best path forcing one C-type + one S-type + one X/M-type asteroid |
| `pkl/results_50ast_full.pkl` | Brute-force run over the original 50-asteroid pool |

---

# Part 3: What the code is doing under the hood

(Skip this if you just want to run things. This part is for understanding the algorithm.)

## Stage 1: Asteroid screening

We query the JPL Small-Body Database for main-belt asteroids with semi-major axis between 2.0 and 3.5 AU, diameter > 30 km, and a known taxonomy. That gives 406 candidates. We download an SPK ephemeris file for each from JPL Horizons (covering 2025–2050).

## Stage 2: Multi-criteria ranking (`tradeoff.py`)

Each asteroid is scored 1–10 on eight criteria, then ranked by weighted total:

| Criterion | Weight | Direction | Why it matters |
|-----------|--------|-----------|----------------|
| Delta-v accessibility | 30% | Lower better | Dominates mission feasibility |
| Orbital inclination | 20% | Lower better | Plane changes are expensive |
| Science potential | 14% | Higher better | Mission science return |
| Estimated mass | 12% | Higher better | Larger = richer geology |
| Radius | 12% | Higher better | More surface for mapping |
| Eccentricity | 6% | Lower better | High-e raises rendezvous cost |
| Rotation period | 4% | 6–24h optimal | Operations/imaging constraint |
| Semi-major axis | 2% | Lower better | Closer = shorter transfer |

Output: `asteroid_tradeoff.csv`, with the top ~50 promoted to the next stage.

## Stage 3: Trajectory optimization (`optimization.py`)

For each candidate triplet (A1, A2, A3) we solve for the best timing of all six mission events:

| Variable | Physical meaning | Bounds |
|----------|-----------------|--------|
| x1 | Launch date | Jan 2027 – Dec 2035 |
| x2 | Earth -> A1 transfer time | 2 weeks – 5 years |
| x3 | Stay at A1 | 3 months – 1 year |
| x4 | A1 -> A2 transfer time | 2 weeks – 5 years |
| x5 | Stay at A2 | 3 months – 1 year |
| x6 | A2 -> A3 transfer time | 2 weeks – 5 years |

Hard constraint: total mission < 14 years (BSP coverage ends ~2050).

For each timing vector we:
1. Look up positions/velocities of Earth and the three asteroids using SPICE.
2. Solve **Lambert's problem** for each leg using `pykep` (Izzo's algorithm).
3. Compute the velocity mismatch (delta-v) at all six maneuver points.
4. Sum them.

The optimizer is `scipy.optimize.differential_evolution` (a global, population-based search) with L-BFGS-B refinement at the end.

## Stage 4: Sequence selection

Three algorithms are available — pick one in `scripts.py`:

- **Brute force** (`generate_optimized_data`) — every triplet, full optimizer. Slow.
- **Two-level** (`two_level_optimize`) — coarse pass on all triplets, fine pass on top 50. ~10x faster, nearly identical results. **Recommended.**
- **Beam search** (`beam_search`) — build path leg-by-leg, keep top-K. Even faster.

## Gravity assists

For each triplet, the optimizer evaluates direct, lunar-flyby, and Mars-flyby architectures and picks the best. The Mars variant adds a 7th decision variable (Earth -> Mars transfer time) and computes powered-flyby delta-v with `pykep.fb_dv()`. Minimum flyby altitude is 200 km above the Mars surface.

---

# Part 4: Project layout

```
Ae105c-Project/
├── Python_Consolidated/          # Active codebase
│   ├── core.py                   # pykep / SPICE wrappers, constants
│   ├── optimization.py           # delta-v, scoring, optimizers
│   ├── greedy.py                 # legacy greedy algorithm
│   ├── visualization.py          # 3D animation, plotting
│   ├── scripts.py                # all the run_* entry points
│   ├── tradeoff.py               # asteroid science scoring
│   ├── lowthrust.py              # low-thrust trajectory module
│   ├── hybrid_mission.py         # hybrid impulsive+low-thrust
│   ├── plot_*.py                 # standalone plotting scripts
│   ├── gcp/                      # Google Cloud runner scripts
│   └── requirements.txt
├── NOTABLE_ASTEROID_BSPs/        # 69 asteroid ephemeris files
├── SPICE_BSPs/                   # extended pool (49 BSPs)
├── Renders/Asteroid_Plots/       # generated images and GIFs
├── optimal_asteroid_paths/       # saved optimization results (.pkl)
├── asteroid_tradeoff.csv         # ranked asteroid table (407 rows)
├── plan.md                       # full mission design (technical)
├── selection_and_optimization.md # detailed methodology (technical)
├── LOW_THRUST_PLAN.md            # low-thrust framework spec (technical)
├── EARTH_GA_PLAN.md              # Earth-flyby spec (technical)
└── README.md                     # this file
```

If you want the deeper, theory-heavy explanation, read `plan.md` and `selection_and_optimization.md`. They're written for someone with mission-design background.

---

# Part 5: Top results found so far

### Lowest delta-v (gravity assists allowed)

| Rank | Path | dV (km/s) | Flyby |
|------|------|:---:|:---:|
| 1 | **Hertha [X/M] -> Polyxo [C] -> Alkeste [S]** | **9.40** | Mars |
| 2 | Virginia [C] -> Psyche [X/M] -> Parthenope [S] | 9.51 | Moon |
| 3 | Massalia [S] -> Misa [C] -> Psyche [X/M] | 9.77 | Moon |

### Best science-weighted (70% science, 30% dv)

| Rank | Path | dV | Science | Score | Flyby |
|------|------|:--:|:------:|:-----:|:-----:|
| 1 | **Aegina [C] -> Beatrix [X/M] -> Vesta [S]** | **10.8** | **20.1** | **10.14** | Mars |
| 2 | Massalia [S] -> Psyche [X/M] -> Themis [C] | 10.7 | 20.0 | 10.25 | Moon |
| 3 | Massalia [S] -> Psyche [X/M] -> Concordia [C] | 10.1 | 19.7 | 10.25 | Moon |

---

# Troubleshooting

### `command not found: python3.11`

Homebrew didn't put it on your PATH. Try `/opt/homebrew/bin/python3.11 --version` (Apple Silicon) or `/usr/local/bin/python3.11 --version` (Intel Mac). If that works, add Homebrew to your PATH:

```bash
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### `pip install pykep` fails

`pykep` only ships pre-built wheels for Python 3.10, 3.11, and 3.12 — not 3.13. Run `python3.11 --version` to confirm you're on 3.11. If you accidentally created the venv with a newer Python, delete it (`rm -rf .venv`) and recreate with `python3.11 -m venv .venv`.

### `SpiceyError: Insufficient ephemeris data has been loaded`

You're missing one of the SPICE kernels, or the symlink `generic_kernels` doesn't point where it should. Check:

```bash
ls -la generic_kernels/
ls generic_kernels/spk/planets/de430.bsp
ls generic_kernels/lsk/naif0012.tls
```

All four kernel files must exist. If not, redo Step 6.

### `An error occurred: ... Most likely you need to add NOTABLE_ASTEROID_BSPs to your path`

You ran the script from the wrong directory. Always run from the project root (`Ae105c-Project/`), not from inside `Python_Consolidated/`. Run `pwd` to check.

### Optimization is taking forever

A two-level run on 69 asteroids takes 20–60 minutes on a laptop. If you want quick results, use beam search instead (Run #4) — it finishes in 5–10 minutes.

### "ModuleNotFoundError: No module named 'core'"

You forgot to put `Python_Consolidated/` on the Python path. Always start your one-liners with:

```python
import sys; sys.path.insert(0, 'Python_Consolidated')
```

Or `cd Python_Consolidated && python3.11 ...` and adjust paths to `../NOTABLE_ASTEROID_BSPs` and `../generic_kernels`.

### Video output (MP4 / GIF) doesn't render

Install ffmpeg: `brew install ffmpeg`. Then re-run.

---

# Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | numerical arrays |
| `scipy` | `differential_evolution` global optimizer |
| `spiceypy` | SPICE kernel loading, ephemeris queries |
| `pykep` | Lambert solver, propagation, flyby dv (ESA) |
| `matplotlib` | plotting and animation |
| `tqdm` | progress bars |
| `pandas` | CSV I/O for the tradeoff table |
| `imageio` | GIF writer |

# References

- Izzo, D. (2015). "Revisiting Lambert's problem." *Celestial Mechanics & Dynamical Astronomy*, 121(1):1–15.
- Storn, R. & Price, K. (1997). "Differential Evolution." *J. Global Optimization*, 11:341–359.
- Carry, B. (2012). "Density of asteroids." *Planetary & Space Science*, 73, 98–118.
- DeMeo, F.E. et al. (2009). "An extension of the Bus asteroid taxonomy." *Icarus*, 202, 160–180.
- pykep — https://esa.github.io/pykep/
- SPICE / NAIF — https://naif.jpl.nasa.gov/naif/
