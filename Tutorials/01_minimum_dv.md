# Tutorial 01 — Lowest-fuel 3-asteroid mission

**Question this tutorial answers:** of all the asteroid combinations the
spacecraft could visit, which three (in which order) cost the least fuel?

**Time:** 20–40 min on a laptop. **No GCP required.** Local CPU only.

## Pre-requisites

- Tutorial 00 finished (Python env + kernels + 69 BSPs).
- You're in `~/Desktop/Ae105c-Project` with `(.venv)` in your prompt.

## Want the answer without re-running?

A pre-computed result already ships with the repo. Look at it first:

```bash
python Python_Consolidated/main.py plot results_69ast_ga.pkl
```

You'll see the top-10 ranked paths. The current best is **Hertha → Polyxo →
Alkeste at 9.40 km/s** with a Mars gravity assist. (Note: this old result has a
non-physical Mars flyby — see Tutorial 02 for the corrected answer.)

## Re-running from scratch

```bash
python Python_Consolidated/main.py optimize
```

What it does, in order:

1. Loads all 69 asteroid orbits from `NOTABLE_ASTEROID_BSPs/`
2. Runs a fast/coarse optimizer on every triplet of asteroids (~300,000 of them)
3. Picks the top 50 most promising
4. Runs the full slow optimizer on those 50
5. Saves the ranked list to `optimal_asteroid_paths/pkl/two_level_0_0_0.pkl`

Progress bars print to the terminal — that's normal.

## When it finishes

```bash
python Python_Consolidated/main.py plot two_level_0_0_0.pkl
```

This prints the top-10 paths sorted by Δv. Pick a rank and visualize it:

```bash
python Python_Consolidated/main.py plot two_level_0_0_0.pkl --rank 1 --gif
```

GIF ends up in `Renders/`.

## Common variations

```bash
# Science-weighted: 70% Δv, 30% science score
python Python_Consolidated/main.py optimize --science 0.7

# Faster: beam search instead of two-level
python Python_Consolidated/main.py optimize --beam 15

# Different launch window
python Python_Consolidated/main.py optimize \
    --launch-min "Jan 1 12:00:00 UTC 2030" \
    --launch-max "Dec 31 12:00:00 UTC 2040"
```

## Verification

After it finishes you should see a line like:

```
Saved: optimal_asteroid_paths/pkl/two_level_0_0_0.pkl
```

If you see that, ✅ the run worked.

If you see `KERNELVARNOTFOUND` or `No module named 'core'`, see `FAQ.md`.

## What's next

- The result from this tutorial trusts the optimizer's flyby physics. That
  trust is sometimes misplaced. **`02_diverse_csm.md`** runs the same kind
  of optimization but with a hard physical-flyby check — it's the recommended
  workflow for a real result you'd put in a report.
- **`05_visualize.md`** — render the trajectory you just found.
