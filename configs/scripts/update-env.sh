#!/bin/bash
set -e  # Exit on any error

echo "🚀 Updating Python environment (gp-env)..."

# Move to project root
cd "$(dirname "$0")"/../..

# Ensure gp-env exists before freezing
if [ ! -d "gp-env" ]; then
    echo "❌ No existing gp-env found. Run setup.sh first."
    exit 1
fi

# Activate current environment to freeze packages
echo "📋 Freezing current dependencies to requirements.txt..."
source gp-env/bin/activate
pip freeze > requirements.txt
deactivate

# Optionally back up old environment
if [ -d "gp-env" ]; then
    echo "♻️  Removing old environment..."
    rm -rf gp-env
fi

# Recreate environment
echo "🐍 Creating fresh gp-env..."
python -m venv gp-env || { echo "❌ Failed to create venv"; exit 1; }

# Activate new env
source gp-env/bin/activate

# Upgrade pip and reinstall dependencies
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing from updated requirements.txt..."
pip install -r requirements.txt

echo "✅ Environment updated and
