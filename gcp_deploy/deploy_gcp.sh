#!/bin/bash
# ============================================================
# Deploy asteroid optimization to Google Cloud Compute Engine
#
# Prerequisites:
#   1. gcloud auth login
#   2. gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   bash deploy_gcp.sh
#
# This will:
#   1. Create a Compute Engine VM (e2-standard-4, 4 vCPUs, 16 GB RAM)
#   2. Upload your code and data
#   3. Install Python and dependencies
#   4. Print SSH command to connect
# ============================================================

set -e

# ---- Configuration (edit these) ----
VM_NAME="asteroid-optimizer"
ZONE="us-west1-b"
MACHINE_TYPE="e2-standard-4"    # 4 vCPU, 16 GB RAM (~$0.13/hr)
BOOT_DISK_SIZE="30GB"
IMAGE_FAMILY="debian-12"
IMAGE_PROJECT="debian-cloud"

GCLOUD="/opt/homebrew/share/google-cloud-sdk/bin/gcloud"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Check auth ----
echo "Checking gcloud authentication..."
PROJECT=$($GCLOUD config get-value project 2>/dev/null)
if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
    echo ""
    echo "ERROR: No GCP project set."
    echo "Run these first:"
    echo "  $GCLOUD auth login"
    echo "  $GCLOUD config set project YOUR_PROJECT_ID"
    echo ""
    echo "If you don't have a project, create one at:"
    echo "  https://console.cloud.google.com/projectcreate"
    exit 1
fi
echo "  Project: $PROJECT"
echo ""

# ---- Create VM ----
echo "Creating VM: $VM_NAME ($MACHINE_TYPE in $ZONE)..."
$GCLOUD compute instances create $VM_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --scopes=default \
    --tags=asteroid-opt \
    2>&1 || {
        echo ""
        echo "VM may already exist. Trying to start it..."
        $GCLOUD compute instances start $VM_NAME --zone=$ZONE 2>&1 || true
    }

echo ""
echo "Waiting 15s for VM to boot..."
sleep 15

# ---- Upload code and data ----
echo "Compressing data for upload..."
cd "$SCRIPT_DIR"
tar czf /tmp/asteroid_deploy.tar.gz \
    --exclude='__pycache__' \
    code/ NOTABLE_ASTEROID_BSPs/ generic_kernels/ \
    asteroid_tradeoff.csv setup_vm.sh run_optimization.py

echo "Uploading to VM (~1.3 GB compressed, may take a few minutes)..."
$GCLOUD compute scp /tmp/asteroid_deploy.tar.gz \
    $VM_NAME:~/asteroid_deploy.tar.gz \
    --zone=$ZONE

echo "Extracting on VM..."
$GCLOUD compute ssh $VM_NAME --zone=$ZONE --command="
    mkdir -p ~/asteroid_project
    cd ~/asteroid_project
    tar xzf ~/asteroid_deploy.tar.gz
    rm ~/asteroid_deploy.tar.gz
    echo 'Files extracted successfully'
    ls -la
"

# ---- Install dependencies ----
echo ""
echo "Installing Python and dependencies on VM..."
$GCLOUD compute ssh $VM_NAME --zone=$ZONE --command="
    cd ~/asteroid_project
    bash setup_vm.sh
"

# ---- Print instructions ----
echo ""
echo "========================================================"
echo "  VM READY: $VM_NAME"
echo "========================================================"
echo ""
echo "SSH into the VM:"
echo "  $GCLOUD compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "Once connected, run:"
echo "  source ~/astro_env/bin/activate"
echo "  cd ~/asteroid_project"
echo ""
echo "  # Quick test (5 asteroids, fast)"
echo "  python3 run_optimization.py two_level --subset 5 --top_n 10"
echo ""
echo "  # Full run (all 50 asteroids, ~hours)"
echo "  python3 run_optimization.py two_level --top_n 50"
echo ""
echo "  # With science weighting"
echo "  python3 run_optimization.py two_level --science_csv asteroid_tradeoff.csv --alpha 0.7"
echo ""
echo "  # Beam search"
echo "  python3 run_optimization.py beam --beam_width 15"
echo ""
echo "Download results when done:"
echo "  $GCLOUD compute scp $VM_NAME:~/asteroid_project/results_*.pkl . --zone=$ZONE"
echo ""
echo "STOP the VM when done (saves money):"
echo "  $GCLOUD compute instances stop $VM_NAME --zone=$ZONE"
echo ""
echo "DELETE the VM when completely finished:"
echo "  $GCLOUD compute instances delete $VM_NAME --zone=$ZONE"
echo ""
echo "Cost: ~\$0.13/hour while running, \$0 when stopped."
echo "========================================================"
