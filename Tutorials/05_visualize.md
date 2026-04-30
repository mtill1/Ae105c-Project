# Tutorial 05 — Render any saved trajectory as a 3D animation

**Question this tutorial answers:** I have a saved `.pkl` result. How do I see
the trajectory?

**Time:** ~30 s for a static PNG, ~2 min for a GIF, ~5 min for a long GIF.
**No GCP required.**

## Pre-requisites

- Tutorial 00 finished
- Some saved result `.pkl` in `optimal_asteroid_paths/pkl/`
- For GIFs only: `Pillow` (already in requirements.txt)

## See what results are saved

```bash
python Python_Consolidated/main.py list
```

Output:

```
N results in optimal_asteroid_paths/pkl/
  results_69ast_ga.pkl                 ...
  diverse_top3_feasible.pkl            ...
  results_mass_pareto_<TS>.pkl         ...
  single_PARTHENOPE_PSYCHE_THEMIS.pkl  ...
```

## See top entries inside a pkl

```bash
python Python_Consolidated/main.py plot diverse_top3_feasible.pkl
```

Prints the top-10 entries with their Δv, architecture, and asteroid names.

## Render the #1 entry as a static PNG

```bash
python Python_Consolidated/main.py plot diverse_top3_feasible.pkl --rank 1
```

Writes `Renders/<NAMES>_trajectory.png` and shows it.

## Render the #1 entry as an animated GIF

```bash
python Python_Consolidated/main.py plot diverse_top3_feasible.pkl --rank 1 --gif
```

Writes `Renders/<NAMES>_trajectory.gif`. Default 240 frames at 24 fps =
10-second loop. Adjust:

```bash
python Python_Consolidated/main.py plot diverse_top3_feasible.pkl --rank 1 --gif \
    --frames 360 --fps 30
```

## Render by name instead of by rank

```bash
python Python_Consolidated/main.py plot diverse_top3_feasible.pkl \
    --names VIRGINIA PSYCHE PARTHENOPE --gif
```

Names must match exactly (uppercase, no spaces).

## What's in the rendered animation

- Sun at the center (yellow)
- Earth (blue), Mars (red), and the three target asteroids (custom colors per
  taxonomy: C-type teal, S-type purple, X/M-type gray)
- Each body's full orbit drawn faintly
- Spacecraft trajectory ticks forward in cyan, building over time
- Date label updates as the frame advances
- Phase label shows what the spacecraft is doing
- Slow camera drift across the run for 3D parallax

## Common gotchas

- **GIF too big to share** (~15 MB): re-render with fewer frames (`--frames 120`),
  or post-process with `gifsicle -O3 --colors 128 in.gif > out.gif`.
- **Trajectory looks like straight lines**: the result has no Lambert
  re-propagation data; the renderer falls back to linear interpolation.
  Re-run the optimization to regenerate the pkl.
- **Asteroid not found**: name mismatch. Use `python Python_Consolidated/main.py list`
  to confirm available pkls, then `--rank N` instead of `--names`.

## What's next

- **`06_verify_physics.md`** — sanity-check the physics behind the trajectory
  you just rendered
