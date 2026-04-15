#!/bin/bash
set -e

echo "=== Setting up asteroid optimization VM ==="

# Install Python 3.11 (pykep compatible) + pip
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3-pip git

# Create virtual environment
python3.11 -m venv ~/astro_env
source ~/astro_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install numpy scipy spiceypy matplotlib tqdm pandas imageio

# Try pykep (may fail on some architectures)
pip install pykep 2>/dev/null || echo "WARNING: pykep not available, will use spiceypy fallback"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source ~/astro_env/bin/activate"
echo "Run from:      cd ~/asteroid_project"
