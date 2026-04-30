# Tutorials — FAQ

## Setup errors

### `pip install pykep` fails with "no matching distribution"

You're on Python 3.13. pykep doesn't ship a 3.13 wheel.

```bash
brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r Python_Consolidated/requirements.txt
```

### `KERNELVARNOTFOUND BODY4_GM` (or any KERNELVARNOTFOUND)

You're missing a SPICE kernel. Re-run Tutorial 00 Step 3. Confirm with:

```bash
ls generic_kernels/lsk/naif0012.tls
ls generic_kernels/spk/planets/de430.bsp
ls generic_kernels/pck/gm_de431.tpc
```

All three must exist.

### `No module named 'core'`

You're inside `Python_Consolidated/` instead of the repo root. Always run
from the repo root:

```bash
cd ~/Desktop/Ae105c-Project
python Python_Consolidated/main.py <command>
```

### `command not found: python3.11`

Homebrew not on PATH. Run once:

```bash
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

## Optimization errors

### "Optimization returns delta_v_total = 1000"

That's the project's penalty value for an infeasible trajectory. Causes:

- Mission > 14 years (BSP coverage ends ~2050)
- Lambert solver failed to converge for that timing
- **Mars/Moon flyby geometrically impossible at safe periapsis** (this fires
  whenever the required turn angle exceeds gravity's natural max)

The optimizer steers away from these regions, but if every trial returns 1000
the asteroid pair you're optimizing is probably impossible in the launch
window. Try a longer launch window or a different triplet.

### Asteroid name "X" not recognized

Names are uppercase and must match `NOTABLE_ASTEROID_BSPs/` filenames exactly.
Check available names:

```bash
ls NOTABLE_ASTEROID_BSPs/ | sed 's/.bsp//' | sort
```

### Optimization is taking forever

| Workflow | Expected wall time |
|---|---|
| `optimize` (two-level, 69 ast) | 20–40 min |
| `optimize --beam 15` | 5–10 min |
| `optimize --feasible` | 5–10 min |
| `optimize --pareto` (top 50) | 70 min – 3.5 hr (depends on settings) |
| GCP `run_mass_pareto.py` | 30 min – 3.5 hr |
| GCP `run_single_triplet.py` | 5–10 min |

If it's much slower than this, check that Python 3.11 is being used (the
3.13 fallback paths are slower).

## Result file questions

### What's the difference between `results_69ast_ga.pkl` and `diverse_top3_feasible.pkl`?

- `results_69ast_ga.pkl` — old run before the flyby-physics fix. Some entries
  have non-physical Mars flybys.
- `diverse_top3_feasible.pkl` — recent run with the patched `compute_flyby_dv`.
  All entries have audited Mars/Moon altitudes.

Always prefer the newer file for reports.

### Why does the same triplet have different Δv across pkl files?

Different optimization settings (DE seed, m-revs, top-N) find different local
optima. The newest file is usually the most thorough. Check `audited` keys:
the `_feasible` files include flyby audits.

### "Verification failed" on a top result

The surrogate (fast estimator) said it was feasible, but the real
Sims-Flanagan low-thrust solver couldn't fly it. The verifier falls back to
the next-best architecture. This is expected — the surrogate is optimistic.

## GCP-specific

### "No credentials" error from gcloud

```bash
gcloud auth login
```

A browser pops up. Sign in. The credentials persist for ~hours, then expire.

### VM can't read the data bucket (`gs://ae105c-asteroid-data`)

You forgot `--scopes=storage-ro,default` when creating the VM. Delete and
recreate:

```bash
gcloud compute instances delete asteroid-optimizer --zone=us-west1-b --quiet
# then re-run the create command from your tutorial, with the scopes flag
```

### Run finished but I forgot to download results, then deleted the VM

Results are gone with the disk. Always SCP back **before** deleting:

```bash
gcloud compute scp asteroid-optimizer:~/project/optimal_asteroid_paths/pkl/*.pkl \
    ./optimal_asteroid_paths/pkl/ --zone=us-west1-b
gcloud compute instances delete asteroid-optimizer --zone=us-west1-b --quiet
```

## Visualization

### GIF is 16 MB and hard to share

Matplotlib's PillowWriter doesn't optimize palettes. Reduce frames or
post-process:

```bash
gifsicle -O3 --colors 128 Renders/big.gif > Renders/small.gif
```

### Trajectory in the GIF is straight lines, not curves

The pkl is missing Lambert intermediate states. The renderer fell back to
linear interpolation between events. Re-run the optimization to regenerate
the result.

## What does each architecture code mean?

Three letters labeling propulsion mode of the three transfer legs after Earth
launch:

```
Letter 1 = leg flyby_body → A1
Letter 2 = leg A1         → A2
Letter 3 = leg A2         → A3

C = chemical (Isp 320 s, impulsive burns)
E = electric (Isp 3100 s, continuous low-thrust)
```

So `CCE` means chemical Mars→A1, chemical A1→A2, electric A2→A3. The Earth
launch and powered Mars flyby are always chemical (you can't ion-thrust off
the launchpad).

## Anything not covered here

Open an issue or message Donny.
