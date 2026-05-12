#!/bin/bash
# TCC-IRoNL Python Dependencies Installer
# Creates a virtual environment with all required packages.

set -e

# Find the workspace root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Workspace root is two directories up: <ws>/src/TCC-IRoNL/ -> <ws>
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$WORKSPACE_DIR/TCC-IRoNLEnv"

echo "=========================================="
echo "TCC-IRoNL Dependency Installer"
echo "=========================================="
echo "Workspace: $WORKSPACE_DIR"
echo "Virtualenv: $VENV_DIR"
echo ""

# Fall back to python3 if 3.8 isn't on PATH.
if command -v python3.8 >/dev/null 2>&1; then
    PYTHON_BIN="python3.8"
else
    PYTHON_BIN="python3"
fi

PYTHON_VERSION="$($PYTHON_BIN --version 2>&1 | awk '{print $2}')"
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: Python 3.8+ required. Found: $PYTHON_VERSION ($PYTHON_BIN)"
    exit 1
fi

echo "Python: $PYTHON_VERSION ($PYTHON_BIN)"

# Virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    if [ -t 0 ]; then
        read -p "Recreate? (y/N): " RECREATE
        if [[ $RECREATE =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
            "$PYTHON_BIN" -m venv "$VENV_DIR"
            echo "Virtual environment recreated"
        else
            echo "Using existing virtual environment"
        fi
    else
        echo "Non-interactive shell detected; keeping existing virtual environment"
    fi
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "Virtual environment created"
fi

# Activate and upgrade pip
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

echo ""
echo "Installing core dependencies..."
echo ""

# PyTorch with CUDA 11.8 support
pip install "torch==2.0.1+cu118" "torchvision==0.15.2+cu118" \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Segment Anything Model
pip install "git+https://github.com/facebookresearch/segment-anything.git"

# LLM provider SDKs
pip install "openai>=1.0.0"
pip install "google-generativeai"   # Gemini
pip install "anthropic"             # Claude

# ROS Python bridge
pip install rospkg catkin_pkg

# Computer vision
pip install opencv-python Pillow numpy

# Audio (for voice input)
pip install SpeechRecognition PyAudio gTTS

# Web framework for the chat GUI
pip install flask flask-socketio

# Misc utilities
pip install pyyaml python-dotenv requests tqdm

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To verify the install:"
echo "  python -c \"import torch; print('PyTorch:', torch.__version__)\""
echo "  python -c \"import openai; print('OpenAI:', openai.__version__)\""
echo ""
