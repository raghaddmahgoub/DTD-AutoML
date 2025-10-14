#!/bin/bash
set -e  # Exit on any error

echo "📦 Updating requirements.txt with new dependencies..."

# Move to project root
cd "$(dirname "$0")"/../..

# Ensure gp-env exists
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

# Activate environment
source "$ACTIVATE_PATH"

# Export current dependencies to a temp file
TMP_REQ="temp_requirements.txt"
pip freeze | sort > "$TMP_REQ"

# Make sure requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "📝 Creating new requirements.txt..."
    touch requirements.txt
fi

# Sort existing and compare
sort requirements.txt -o requirements.txt

# Find packages not in existing requirements.txt
NEW_REQ="new_requirements.txt"
comm -23 "$TMP_REQ" requirements.txt > "$NEW_REQ"

# Append new ones if any
if [ -s "$NEW_REQ" ]; then
    echo "✨ Found new dependencies. Appending..."
    cat "$NEW_REQ" >> requirements.txt
else
    echo "✅ No new dependencies to add."
fi

# Clean up
rm -f "$TMP_REQ" "$NEW_REQ"

deactivate
echo "✅ requirements.txt updated successfully!"
