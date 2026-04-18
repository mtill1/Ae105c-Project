"""
GCP configuration for asteroid trajectory optimization.

All GCP settings in one place. Import this from any GCP runner script.
Future Claude sessions: read this file first before doing anything with GCP.
"""

# =============================================================================
# GCP PROJECT
# =============================================================================
PROJECT_ID = "project-8b1249f5-4cb6-4dad-8a9"
ZONE = "us-west1-b"
MACHINE_TYPE = "e2-custom-12-49152"  # 12 vCPU, 48 GB RAM (max for CPUS_ALL_REGIONS=12 quota)
VM_NAME = "asteroid-optimizer"
BOOT_DISK_SIZE = "50GB"
IMAGE_FAMILY = "debian-12"
IMAGE_PROJECT = "debian-cloud"

# =============================================================================
# GCS BUCKET (persistent data — never need to re-upload kernels)
# =============================================================================
GCS_BUCKET = "gs://ae105c-asteroid-data"
# Bucket is PUBLIC READ (allUsers:objectViewer) — no auth needed to pull

# Contents:
#   gs://ae105c-asteroid-data/generic_kernels/naif0012.tls     (leapseconds)
#   gs://ae105c-asteroid-data/generic_kernels/jup310.bsp       (Jupiter satellites)
#   gs://ae105c-asteroid-data/generic_kernels/de430.bsp        (planetary ephemeris, 1.1 GB)
#   gs://ae105c-asteroid-data/generic_kernels/gm_de431.tpc     (gravitational parameters)
#   gs://ae105c-asteroid-data/generic_kernels/pck00010.tpc     (planetary constants)
#   gs://ae105c-asteroid-data/NOTABLE_ASTEROID_BSPs/*.bsp      (69 asteroid ephemeris files)
#   gs://ae105c-asteroid-data/asteroid_tradeoff.csv            (science scoring table)

# =============================================================================
# VM SETUP COMMANDS (run after creating VM)
# =============================================================================
SETUP_COMMANDS = """
# 1. Pull data from GCS bucket (~30 sec, no auth needed — bucket is public)
cd ~/project
mkdir -p generic_kernels/lsk generic_kernels/spk/satellites generic_kernels/spk/planets generic_kernels/pck NOTABLE_ASTEROID_BSPs
gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/naif0012.tls generic_kernels/lsk/
gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/jup310.bsp generic_kernels/spk/satellites/
gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/de430.bsp generic_kernels/spk/planets/
gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/gm_de431.tpc generic_kernels/pck/
gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/pck00010.tpc generic_kernels/pck/
gsutil -q -m rsync gs://ae105c-asteroid-data/NOTABLE_ASTEROID_BSPs/ NOTABLE_ASTEROID_BSPs/
gsutil -q cp gs://ae105c-asteroid-data/asteroid_tradeoff.csv .

# 2. Install Miniconda + Python 3.11 + pykep (if not already installed)
if [ ! -d $HOME/mc3 ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p $HOME/mc3 && rm /tmp/mc.sh
fi
export PATH=$HOME/mc3/bin:$PATH
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null
if [ ! -d $HOME/env311 ]; then
    conda create -p $HOME/env311 python=3.11 -y -q
    conda install -p $HOME/env311 -c conda-forge pykep spiceypy scipy numpy pandas tqdm matplotlib -y -q
fi
"""

# =============================================================================
# GCLOUD AUTH
# =============================================================================
# gcloud auth expires frequently. Before any GCP operation, run:
#   /opt/homebrew/share/google-cloud-sdk/bin/gcloud auth login
#
# Then set project:
#   /opt/homebrew/share/google-cloud-sdk/bin/gcloud config set project project-8b1249f5-4cb6-4dad-8a9

# =============================================================================
# SSH KEY
# =============================================================================
SSH_KEY = "/Users/rebnoob/.ssh/google_compute_engine"
# SSH command template:
#   ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i {SSH_KEY} rebnoob@{VM_IP}

# =============================================================================
# QUOTA LIMITS
# =============================================================================
# CPUS_ALL_REGIONS = 12 (global cap, cannot use more than 12 vCPU)
# E2_CPUS = 8 (regional), but e2-custom allows up to global cap
# C2_CPUS = 0 (no compute-optimized machines available)
# To increase: https://console.cloud.google.com/iam-admin/quotas
#   Search "CPUS_ALL_REGIONS", request increase to 64

# =============================================================================
# COST ESTIMATES
# =============================================================================
# e2-custom-12-49152: ~$0.40/hr
# Typical run (14,040 triplets): ~7 min = ~$0.05
# GCS bucket storage: ~$0.02/month for 1.29 GB
# VM deleted after each run — $0 idle cost

# =============================================================================
# QUICK REFERENCE: How to run an optimization
# =============================================================================
# 1. gcloud auth login
# 2. gcloud config set project project-8b1249f5-4cb6-4dad-8a9
# 3. gcloud compute instances create asteroid-optimizer --zone=us-west1-b \
#      --machine-type=e2-custom-12-49152 --boot-disk-size=50GB \
#      --image-family=debian-12 --image-project=debian-cloud \
#      --scopes=storage-ro,default --quiet
# 4. Wait 15s, then SCP code: gcloud compute scp code.tar.gz asteroid-optimizer:~/
# 5. SSH in, run setup commands above, then run the optimizer
# 6. SCP results back, then: gcloud compute instances delete asteroid-optimizer \
#      --zone=us-west1-b --delete-disks=all --quiet
#
# IMPORTANT: Add --scopes=storage-ro,default when creating VM so it can read GCS bucket!
"""
