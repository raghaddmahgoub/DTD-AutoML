#!/bin/bash

ENV_NAME="automl_env_310"

echo "=========================================="
echo "🚀 Running Setup from GP/ Root"
echo "=========================================="

# 1. Check for Python 3.10
if ! command -v python3.10 &> /dev/null
then
    echo "❌ Python 3.10 not found! Please run: brew install python@3.10"
    exit
fi

# 2. Create Virtual Environment
if [ ! -d "$ENV_NAME" ]; then
    python3.10 -m venv $ENV_NAME
fi

# 3. Activate and Install Libraries
source $ENV_NAME/bin/activate
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost requests python-dotenv \
langchain-google-genai langgraph autogluon datasets openpyxl

# 4. Clean up / Fix structure
mkdir -p utils
touch utils/__init__.py  # Critical for 'import utils.logger'

echo "=========================================="
echo "🎉 SETUP COMPLETE"
echo "To run your agent: python test_sklearn_datasets.py"
echo "=========================================="