# Tutorial 04 — Deeply optimize one specific triplet

**Question this tutorial answers:** I picked three asteroids by name. What's
the best mission visiting them in that order, including what propulsion
architecture (chemical / electric per leg) delivers the most mass?

**Time:** ~10–15 min. Works locally OR on GCP. GCP is recommended (12-way
parallelism cuts wall time from ~15 min to ~5 min).

## Pre-requisites

- Tutorial 00 finished
- Asteroid names spelled exactly as in `NOTABLE_ASTEROID_BSPs/` filenames.
  Names are uppercase, no spaces (e.g. `PARTHENOPE`, `PSYCHE`, `THEMIS`).

## Local run

```bash
python Python_Consolidated/main.py optimize \
    --pareto \
    --pareto-seed optimal_asteroid_paths/pkl/results_69ast_ga.pkl \
    --top-n 1
```

Note: the local `--pareto` mode iterates from a seed pkl. To target a triplet
that **isn't** in the seed pkl, use the GCP runner below — it accepts any
triplet by name.

## GCP run (recommended)

```bash
GCLOUD=/opt/homebrew/share/google-cloud-sdk/bin/gcloud

# Spin up an 8-vCPU VM (cheaper than 12-vCPU; we only need 8 archs in parallel)
$GCLOUD compute instances create asteroid-optimizer \
  --zone=us-west1-b \
  --machine-type=e2-standard-8 \
  --boot-disk-size=50GB \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --scopes=storage-ro,default \
  --quiet

# Upload code (no result pickles needed for a single triplet)
cd ~/Desktop/Ae105c-Project
tar czf /tmp/single.tar.gz \
  --exclude='__pycache__' \
  Python_Consolidated/*.py \
  Python_Consolidated/gcp/*.py
sleep 25
$GCLOUD compute scp /tmp/single.tar.gz asteroid-optimizer:~/deploy.tar.gz \
  --zone=us-west1-b

# Setup env + run (TRIPLET is your three asteroid names, comma-separated)
$GCLOUD compute ssh asteroid-optimizer --zone=us-west1-b --command="
  set -e
  mkdir -p ~/project && cd ~/project
  tar xzf ~/deploy.tar.gz && rm ~/deploy.tar.gz

  mkdir -p generic_kernels/lsk generic_kernels/spk/satellites \
           generic_kernels/spk/planets generic_kernels/pck NOTABLE_ASTEROID_BSPs
  gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/naif0012.tls generic_kernels/lsk/
  gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/jup310.bsp   generic_kernels/spk/satellites/
  gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/de430.bsp    generic_kernels/spk/planets/
  gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/gm_de431.tpc generic_kernels/pck/
  gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/pck00010.tpc generic_kernels/pck/
  gsutil -q -m rsync gs://ae105c-asteroid-data/NOTABLE_ASTEROID_BSPs/ NOTABLE_ASTEROID_BSPs/

  if [ ! -d \$HOME/env311 ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p \$HOME/mc3 && rm /tmp/mc.sh
    \$HOME/mc3/bin/conda create -p \$HOME/env311 python=3.11 -y -q
    \$HOME/mc3/bin/conda install -p \$HOME/env311 -c conda-forge \
      pykep spiceypy scipy numpy pandas tqdm matplotlib -y -q
  fi

  # ★ Edit this line to choose your triplet ★
  env TRIPLET=PARTHENOPE,PSYCHE,THEMIS \
    \$HOME/env311/bin/python Python_Consolidated/gcp/run_single_triplet.py \
    > run.log 2>&1
"

# Pull results
$GCLOUD compute scp asteroid-optimizer:~/project/optimal_asteroid_paths/pkl/single_*.pkl \
  ./optimal_asteroid_paths/pkl/ --zone=us-west1-b
$GCLOUD compute scp asteroid-optimizer:~/project/run.log \
  /tmp/single_run.log --zone=us-west1-b

# Delete VM
$GCLOUD compute instances delete asteroid-optimizer \
  --zone=us-west1-b --delete-disks=all --quiet

# Look at result
cat /tmp/single_run.log | tail -30
```

## Expected output

Tail of `run.log`:

```
 BEST: PARTHENOPE -> PSYCHE -> THEMIS via mars flyby, arch=ECE
==========================================================================================
  Verified delivered mass: 587.7 kg / 1500 kg
  Verified Δv-equivalent : 2.94 km/s
  Propellant fraction    : 60.8%
  Launch                 : 2030 JUN 15 18:40:55
  Mars flyby             : 2032 NOV 04 08:42:33
  Arrive PARTHENOPE      : 2036 DEC 11 16:04:45
  ...
  Mission duration       : 11.50 yr
```

The 8 architectures are CCC, CCE, CEC, CEE, ECC, ECE, EEC, EEE — three letters
for the three transfer legs (L2 = flyby→A1, L3 = A1→A2, L4 = A2→A3). C =
chemical (Isp 320 s), E = electric (Isp 3100 s).

## Verify the result is physical

```bash
python Python_Consolidated/main.py verify single_PARTHENOPE_PSYCHE_THEMIS.pkl
```

Look for `RESULT: FEASIBLE` and a positive `Periapsis altitude`.

## What's next

- **`05_visualize.md`** — render the winner as a 3D GIF
- **`06_verify_physics.md`** — what the verification numbers mean
