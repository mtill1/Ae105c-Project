# Tutorial 03 — Mass-Pareto sweep across 50 triplets (GCP)

**Question this tutorial answers:** for each of the 50 best impulsive triplets,
which propulsion architecture (chemical vs. electric on each leg, 8
combinations) delivers the most spacecraft mass to the final asteroid?

**Time:** ~30 min wall clock + ~3 hr if you use the full settings. **GCP
required** (12 vCPUs is the bottleneck).

**Cost:** ~$0.05–$1.50 depending on settings.

## Pre-requisites

- Tutorial 00 finished, including the GCP setup section.
- `pkl/results_69ast_ega_real.pkl` exists (ships with the repo) — this is the
  seed pool of triplets the sweep optimizes.

## The 5 commands

```bash
GCLOUD=/opt/homebrew/share/google-cloud-sdk/bin/gcloud

# 1. Spin up the VM (~30 s)
$GCLOUD compute instances create asteroid-optimizer \
  --zone=us-west1-b \
  --machine-type=e2-custom-12-49152 \
  --boot-disk-size=50GB \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --scopes=storage-ro,default \
  --quiet

# 2. Package and upload (~1 min)
cd ~/Desktop/Ae105c-Project
tar czf /tmp/deploy.tar.gz \
  --exclude='__pycache__' \
  Python_Consolidated/*.py \
  Python_Consolidated/gcp/*.py \
  optimal_asteroid_paths/pkl/results_69ast_ega_real.pkl
sleep 25  # let SSH come up
$GCLOUD compute scp /tmp/deploy.tar.gz asteroid-optimizer:~/deploy.tar.gz \
  --zone=us-west1-b

# 3. Set up env + run (~10 min setup, then 30 min – 3 hr compute)
$GCLOUD compute ssh asteroid-optimizer --zone=us-west1-b --command="
  set -e
  mkdir -p ~/project && cd ~/project
  tar xzf ~/deploy.tar.gz && rm ~/deploy.tar.gz

  # pull SPICE kernels and BSPs from public bucket
  mkdir -p generic_kernels/lsk generic_kernels/spk/satellites \
           generic_kernels/spk/planets generic_kernels/pck NOTABLE_ASTEROID_BSPs
  gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/naif0012.tls generic_kernels/lsk/
  gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/jup310.bsp   generic_kernels/spk/satellites/
  gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/de430.bsp    generic_kernels/spk/planets/
  gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/gm_de431.tpc generic_kernels/pck/
  gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/pck00010.tpc generic_kernels/pck/
  gsutil -q -m rsync gs://ae105c-asteroid-data/NOTABLE_ASTEROID_BSPs/ NOTABLE_ASTEROID_BSPs/

  # install miniconda + pykep
  if [ ! -d \$HOME/mc3 ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p \$HOME/mc3 && rm /tmp/mc.sh
  fi
  if [ ! -d \$HOME/env311 ]; then
    \$HOME/mc3/bin/conda create -p \$HOME/env311 python=3.11 -y -q
    \$HOME/mc3/bin/conda install -p \$HOME/env311 -c conda-forge \
      pykep spiceypy scipy numpy pandas tqdm matplotlib -y -q
  fi

  # run the sweep — output to log, run in background so the SSH session can return
  nohup \$HOME/env311/bin/python Python_Consolidated/gcp/run_mass_pareto.py \
    > run_sweep.log 2>&1 &
  echo \"PID=\$!\"
"

# 4. Watch progress (Ctrl-C to detach; the run keeps going)
$GCLOUD compute ssh asteroid-optimizer --zone=us-west1-b \
  --command='tail -f ~/project/run_sweep.log'

# 5. When done, pull results back AND delete the VM
$GCLOUD compute scp asteroid-optimizer:~/project/optimal_asteroid_paths/pkl/results_mass_pareto_*.pkl \
  ./optimal_asteroid_paths/pkl/ --zone=us-west1-b
$GCLOUD compute instances delete asteroid-optimizer \
  --zone=us-west1-b --delete-disks=all --quiet
```

## Expected output

The log will print a per-triplet table:

```
  #   triplet                              arch   surr_m   ver_m   dv_eq  t(s)
  #1  HERTHA->POLYXO->ALKESTE              EEE   1193.3   900.5    1.60   2134
  ...
  #50 POLANA->MASSALIA->PSYCHE             EEE   1116.5   824.0    1.88   2182
```

`surr_m` = surrogate-predicted final mass (kg). `ver_m` = mass after the real
Sims-Flanagan solver verifies each electric leg. The last line:

```
Saved: /home/rebnoob/project/optimal_asteroid_paths/pkl/results_mass_pareto_<TS>.pkl
```

## ⚠️ Important caveat

The optimizer used in this tutorial does NOT physically verify Mars flybys —
some "winners" require the spacecraft to fly through Mars's surface. The
flyby-physics fix is in the underlying `core.compute_flyby_dv` (so newer runs
are fine), but old result pickles may contain unphysical solutions.

**Always verify with `main.py verify` before trusting a result.** See
Tutorial 06.

## Common gotchas

- **Forgot `--scopes=storage-ro,default`**: the VM can't read the kernel
  bucket. Delete the VM and recreate.
- **Auth expired mid-run**: `gcloud auth login` again. Existing VMs keep
  running; you just need re-auth to interact.
- **Deleted VM before pulling results**: results lost. Always SCP back first.

## What's next

- **`04_single_triplet.md`** — same architecture sweep but on a triplet you
  pick by name
- **`06_verify_physics.md`** — audit the saved results
- **`05_visualize.md`** — render the winner
