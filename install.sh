#!/bin/bash
# Solana Trading Bot - Installation Script

set -e

echo "=== Solana Trading Bot Setup ==="

# Check Python version
python3 --version || { echo "Python 3 is required"; exit 1; }

# Create virtual environment
echo "[1/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "[2/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Setup environment file
echo "[3/4] Setting up configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from template - please edit with your settings"
fi

# Create directories
echo "[4/4] Creating data directories..."
mkdir -p data logs

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your Telegram bot token and settings"
echo "  2. Generate encryption keys (see README.md)"
echo "  3. Run: python main.py"
echo ""
