#!/bin/bash
# Faceless YouTube Channel Automation - Setup Script
# This script sets up everything you need to run the automation

set -e

echo "🎬 Faceless YouTube Channel Automation Setup"
echo "============================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.9+${NC}"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}! Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Check FFmpeg
echo ""
echo "Checking FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version | head -n1 | cut -d' ' -f3)
    echo -e "${GREEN}✓ FFmpeg $FFMPEG_VERSION found${NC}"
else
    echo -e "${YELLOW}! FFmpeg not found. Installing...${NC}"
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install ffmpeg
    else
        echo -e "${RED}✗ Please install FFmpeg manually: https://ffmpeg.org/download.html${NC}"
    fi
fi

# Check/Install Ollama
echo ""
echo "Checking Ollama (Local AI)..."
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama found${NC}"
else
    echo -e "${YELLOW}! Ollama not found. Installing...${NC}"
    curl -fsSL https://ollama.com/install.sh | sh
    echo -e "${GREEN}✓ Ollama installed${NC}"
fi

# Pull Ollama model
echo ""
echo "Pulling Ollama model (llama3.2)..."
ollama pull llama3.2 2>/dev/null || echo -e "${YELLOW}! Could not pull model. Run 'ollama pull llama3.2' manually${NC}"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p output temp logs cache data assets/fonts assets/music assets/media models/piper config
echo -e "${GREEN}✓ Directories created${NC}"

# Check config file
echo ""
echo "Checking configuration..."
if [ -f "config/config.yaml" ]; then
    echo -e "${GREEN}✓ Config file exists${NC}"
else
    echo -e "${RED}✗ Config file missing${NC}"
fi

# Check YouTube credentials
echo ""
echo "Checking YouTube credentials..."
if [ -f "config/client_secrets.json" ]; then
    echo -e "${GREEN}✓ YouTube credentials found${NC}"
else
    echo -e "${YELLOW}! YouTube credentials not found${NC}"
    echo "  To upload videos, you need to:"
    echo "  1. Go to https://console.cloud.google.com/"
    echo "  2. Create a project and enable YouTube Data API v3"
    echo "  3. Create OAuth 2.0 credentials"
    echo "  4. Download and save as config/client_secrets.json"
fi

# Summary
echo ""
echo "============================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "============================================="
echo ""
echo "Next steps:"
echo "1. Edit config/config.yaml with your settings"
echo "2. Add API keys for Pexels/Pixabay (optional but recommended)"
echo "3. Add YouTube credentials (config/client_secrets.json)"
echo "4. Start Ollama: ollama serve"
echo "5. Run: python main.py create"
echo ""
echo "For 24/7 automation: python main.py start"
echo ""
