#!/bin/bash
echo "🚀 Setting up Python environment (gp-env)..."

# Move to project root (one level above configs/)
cd "$(dirname "$0")"/../..

# Remove existing environment if it exists
if [ -d "gp-env" ]; then
    echo "♻️  Removing existing gp-env..."
    rm -rf gp-env
fi

# Create new virtual environment in project root
python -m venv gp-env || { echo "❌ Failed to create venv"; exit 1; }

# Activate it
echo "🔹 Activating environment..."
source gp-env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️  No requirements.txt found in root directory!"
fi

echo "✅ gp-env created successfully in project root!"
