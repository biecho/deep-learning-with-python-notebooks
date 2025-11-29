#!/bin/bash
# =============================================================================
# Setup script for Chapter 14 on p12 server
# =============================================================================
# This script creates the virtual environment, installs dependencies,
# and configures the Jupyter kernel with GPU support.
#
# Usage:
#   cd ~/projects/deep-learning-with-python-notebooks
#   bash setup-p12.sh
#
# After running this script:
#   1. Start Jupyter: source .venv-tf215/bin/activate && jupyter lab --no-browser
#   2. From local machine: ssh -L 8888:localhost:8888 p12
#   3. Open notebook and select kernel "DLWP TF2.12 GPU"
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv-tf215"
KERNEL_NAME="dlwp-tf212"
KERNEL_DIR="$HOME/.local/share/jupyter/kernels/$KERNEL_NAME"

echo "=========================================="
echo "Setting up Chapter 14 for p12 server"
echo "=========================================="

# Step 1: Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "Delete it first if you want a fresh install: rm -rf $VENV_DIR"
    exit 1
fi

echo ""
echo "[1/4] Creating virtual environment with Python 3.11..."
python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo ""
echo "[2/4] Installing dependencies from requirements-p12.txt..."
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements-p12.txt"

echo ""
echo "[3/4] Registering Jupyter kernel..."
python -m ipykernel install --user --name="$KERNEL_NAME" --display-name="DLWP TF2.12 GPU"

echo ""
echo "[4/4] Configuring GPU environment in kernel.json..."

# Build the LD_LIBRARY_PATH
SITE_PACKAGES="$VENV_DIR/lib/python3.11/site-packages"
LD_LIBRARY_PATH=""
for lib in cudnn cublas cuda_runtime cuda_nvrtc cufft curand cusolver cusparse; do
    LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/$lib/lib:$LD_LIBRARY_PATH"
done

# Write kernel.json with environment variables
cat > "$KERNEL_DIR/kernel.json" << EOF
{
  "argv": [
    "$VENV_DIR/bin/python",
    "-Xfrozen_modules=off",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "DLWP TF2.12 GPU",
  "language": "python",
  "metadata": {
    "debugger": true
  },
  "env": {
    "LD_LIBRARY_PATH": "$LD_LIBRARY_PATH",
    "XLA_FLAGS": "--xla_gpu_cuda_data_dir=$SITE_PACKAGES/nvidia/cuda_nvcc",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3"
  }
}
EOF

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To start Jupyter:"
echo "  source $VENV_DIR/bin/activate"
echo "  jupyter lab --no-browser"
echo ""
echo "From your local machine, create SSH tunnel:"
echo "  ssh -L 8888:localhost:8888 p12"
echo ""
echo "Then open the notebook and select kernel: DLWP TF2.12 GPU"
echo ""
