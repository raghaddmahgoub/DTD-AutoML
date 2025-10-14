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

# Detect OS-specific activation path
if [ -f "gp-env/Scripts/activate" ]; then
    ACTIVATE_PATH="gp-env/Scripts/activate"   # Windows
else
    ACTIVATE_PATH="gp-env/bin/activate"       # macOS/Linux
fi

# Freeze dependencies
echo "📋 Freezing current dependencies to requirements.txt..."
source "$ACTIVATE_PATH"
pip freeze > requirements.txt
deactivate

# Backup and recreate environment
echo "♻️  Removing old environment..."
rm -rf gp-env

echo "🐍 Creating fresh gp-env..."
python -m venv gp-env || { echo "❌ Failed to create venv"; exit 1; }

# Detect new activate path
if [ -f "gp-env/Scripts/activate" ]; then
    ACTIVATE_PATH="gp-env/Scripts/activate"
else
    ACTIVATE_PATH="gp-env/bin/activate"
fi

# Activate and reinstall dependencies
echo "🔹 Activating new environment..."
source "$ACTIVATE_PATH"

echo "⬆️  Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing from updated requirements.txt..."
pip install -r requirements.txt

echo "✅ Environment updated and synchronized successfully!"
