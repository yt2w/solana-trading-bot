#!/bin/bash

set -e

echo "=== Solana Trading Bot Setup ==="

python3 --version || { echo "Python 3 is required"; exit 1; }

echo "[1/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[2/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[3/4] Setting up configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from template - please edit with your settings"
fi

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
