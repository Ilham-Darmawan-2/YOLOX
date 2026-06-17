#!/bin/bash
set -e

echo "🐍 Setup Python 3.10..."

# ==============================
# Check Python 3.10
# ==============================
if command -v python3.10 >/dev/null 2>&1; then
  echo "✅ Python 3.10 already installed"
else
  echo "📦 Installing Python 3.10..."

  sudo apt update

  # install add-apt-repository kalau belum ada
  if ! command -v add-apt-repository >/dev/null 2>&1; then
    sudo apt install -y software-properties-common
  fi

  # tambah repo deadsnakes kalau belum ada
  if ! grep -q "deadsnakes" /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
  fi

  sudo apt update

  sudo apt install -y python3.10 python3.10-venv python3.10-dev
fi

# ==============================
# Setup venv
# ==============================
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  python3.10 -m venv venv
else
  echo "✅ venv already exists"
fi

# activate
source venv/bin/activate

python --version

# ==============================
# Install dependencies
# ==============================
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
  pip install duckdb
fi

echo "✅ Python setup completed"
