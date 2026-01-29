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

# Ensure .gitignore exists and contains '*'
if [ ! -f ".gitignore" ]; then
    echo "📝 Creating .gitignore..."
    echo "*" > .gitignore
    echo "✅ Filled .gitignore with * (ignore all files)"
else
    if [ ! -s ".gitignore" ]; then
        echo "*" > .gitignore
        echo "✅ Filled empty .gitignore with *"
    else
        echo "ℹ️  .gitignore already exists and has content — not modified."
    fi
fi

echo "✅ gp-env created successfully in project root!"
