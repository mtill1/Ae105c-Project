#!/bin/bash
set -e

echo "=== Setting up asteroid optimization VM ==="

# Pull data from GCS bucket (public read, ~30 sec)
# Bucket: gs://ae105c-asteroid-data (allUsers:objectViewer)
echo "Downloading data from GCS bucket..."
cd ~/project
mkdir -p generic_kernels/lsk generic_kernels/spk/satellites generic_kernels/spk/planets generic_kernels/pck NOTABLE_ASTEROID_BSPs

gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/naif0012.tls generic_kernels/lsk/
gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/jup310.bsp generic_kernels/spk/satellites/
gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/de430.bsp generic_kernels/spk/planets/
gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/gm_de431.tpc generic_kernels/pck/
gsutil -q cp gs://ae105c-asteroid-data/generic_kernels/pck00010.tpc generic_kernels/pck/
gsutil -q -m rsync gs://ae105c-asteroid-data/NOTABLE_ASTEROID_BSPs/ NOTABLE_ASTEROID_BSPs/
gsutil -q cp gs://ae105c-asteroid-data/asteroid_tradeoff.csv .
echo "Data downloaded from GCS."

# Install Miniconda + pykep
if [ ! -d "$HOME/mc3" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p $HOME/mc3 2>&1 | tail -1
    rm /tmp/mc.sh
fi

export PATH="$HOME/mc3/bin:$PATH"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if [ ! -d "$HOME/env311" ]; then
    echo "Creating Python 3.11 environment..."
    conda create -p $HOME/env311 python=3.11 -y -q 2>&1 | tail -1
    conda install -p $HOME/env311 -c conda-forge pykep spiceypy scipy numpy pandas tqdm matplotlib -y -q 2>&1 | tail -1
fi

echo ""
echo "=== Verifying ==="
$HOME/env311/bin/python -c "import pykep; print('pykep OK')"
$HOME/env311/bin/python -c "import spiceypy; print('spiceypy OK')"
ls ~/project/generic_kernels/spk/planets/de430.bsp > /dev/null && echo "kernels OK"
echo "$(ls ~/project/NOTABLE_ASTEROID_BSPs/*.bsp | wc -l) BSPs OK"
echo ""
echo "=== Setup complete ==="
