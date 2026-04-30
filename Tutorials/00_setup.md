# Tutorial 00 — One-time setup

Do this once. Every other tutorial assumes you've finished this one.

## What you need

- macOS or Linux (Windows: install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) first)
- Python 3.11 (not 3.13 — pykep doesn't ship a 3.13 wheel yet)
- ~2 GB free disk for SPICE kernels
- For the GCP tutorials only: a Google account with a GCP project

## Step 1 — Get the code

```bash
cd ~/Desktop
git clone <REPO_URL> Ae105c-Project
cd Ae105c-Project
```

From now on every command assumes you're in `~/Desktop/Ae105c-Project`.

## Step 2 — Python environment

```bash
brew install python@3.11                       # if not already installed
python3.11 -m venv .venv
source .venv/bin/activate                       # do this every new terminal
pip install -r Python_Consolidated/requirements.txt
```

Verify:

```bash
python -c "import pykep, spiceypy; print('OK')"
```

If you see `OK`, the libraries are installed.

## Step 3 — SPICE kernels

NASA's planetary ephemerides. Five files, ~115 MB total. Download once.

```bash
mkdir -p ~/Documents/ae105/generic_kernels/{lsk,spk/planets,spk/satellites,pck}
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

cd ~/Desktop/Ae105c-Project
ln -sf ~/Documents/ae105/generic_kernels generic_kernels
```

## Step 4 — Confirm everything loads

```bash
python Python_Consolidated/main.py list
```

Expected output:

```
N results in optimal_asteroid_paths/pkl/
  results_69ast_ga.pkl                         ...
  results_mass_pareto_<timestamp>.pkl          ...
  ...
```

If you see a list of `.pkl` files, setup is complete. ✅

## (Optional) Step 5 — GCP setup

Skip this section if you only run things locally.

```bash
brew install --cask google-cloud-sdk
gcloud auth login                                # browser pops up
gcloud config set project <YOUR_PROJECT_ID>
```

The project's GCP defaults are in `Python_Consolidated/gcp/gcp_config.py`. The
shared SPICE-kernel bucket (`gs://ae105c-asteroid-data`) is **public-read**, so
you don't need to upload anything yourself.

When creating a VM you **must** include `--scopes=storage-ro,default` so the VM
can read the bucket. The tutorials show the full command.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pip install pykep` fails | You're on Python 3.13. Use 3.11 specifically. `python3.11 -m venv .venv && source .venv/bin/activate` |
| `KERNELVARNOTFOUND BODY4_GM` | Missing `gm_de431.tpc`. Re-run Step 3. |
| `No module named 'core'` | You're inside `Python_Consolidated/` instead of the repo root. Run `cd ~/Desktop/Ae105c-Project`. |
| `command not found: python3.11` | Homebrew not on PATH. Run `echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc`. |

## What's next

Pick a tutorial:

- **`01_minimum_dv.md`** — find the lowest-fuel 3-asteroid mission (no GCP, ~30 min)
- **`02_diverse_csm.md`** — composition-diverse mission with physical-flyby check (~10 min)
- **`03_mass_pareto_gcp.md`** — chemical vs electric trade across 50 triplets (GCP, ~30 min)
- **`04_single_triplet.md`** — deeply optimize 3 specific asteroids you picked
- **`05_visualize.md`** — render any saved trajectory as a 3D GIF
- **`06_verify_physics.md`** — audit a saved solution's flyby physics
- **`FAQ.md`** — common errors and answers
