#!/bin/bash
# TCC-IRoNL Python Dependencies Installer
# Creates a virtual environment with all required packages

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$WORKSPACE_DIR/TCC-IRoNLEnv"

echo "=========================================="
echo "TCC-IRoNL Dependency Installer"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: Python 3.8+ required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "⚠ Virtual environment already exists at $VENV_DIR"
    read -p "Recreate? (y/N): " RECREATE
    if [[ $RECREATE =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR"
        echo "✓ Virtual environment recreated"
    else
        echo "✓ Using existing virtual environment"
    fi
else
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created"
fi

# Activate and upgrade pip
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

echo ""
echo "Installing core dependencies..."
echo ""

# Install PyTorch with CUDA support
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git

# Install OpenAI API
pip install openai>=1.0.0

# Install other LLM providers
pip install google-generativeai  # Gemini
pip install anthropic          # Claude

# Install ROS Python bridge
pip install rospkg
pip install catkin_pkg

# Install computer vision utilities
pip install opencv-python
pip install Pillow
pip install numpy

# Install audio processing (for voice input)
pip install SpeechRecognition
pip install PyAudio
pip install gTTS  # Google Text-to-Speech

# Install web framework (for chat GUI)
pip install flask
pip install flask-socketio

# Install utilities
pip install pyyaml
pip install python-dotenv
pip install requests
pip install tqdm

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To verify installation:"
echo "  python -c 'import torch; print("PyTorch:", torch.__version__)'"
echo "  python -c 'import openai; print("OpenAI:", openai.__version__)'"
echo ""
