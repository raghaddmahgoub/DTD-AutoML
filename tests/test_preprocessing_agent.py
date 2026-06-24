"""
Test script for the dynamic PreprocessingAgent.
Demonstrates how to use the preprocessing agent with LangGraph.
"""


import os
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.utils.logger import Logger
from agents.dynamic.preprocessing_agent import PreprocessingAgent
from langchain_google_genai import ChatGoogleGenerativeAI

def test_preprocessing_agent_basic():
    """Test the PreprocessingAgent with a basic dataset."""
    print("\n" + "=" * 70)
    print("PREPROCESSING AGENT — BASIC TEST")
    print("=" * 70 + "\n")

    # Setup
    logger = Logger()
    llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )

    # Data path (using Titanic dataset from uploads)
    data_path = str(PROJECT_ROOT / "uploads/Titanic-Dataset.csv")

    if not Path(data_path).exists():
        print(f"❌ Dataset not found: {data_path}")
        print("Please ensure the Titanic-Dataset.csv exists in uploads/")
        return

    # Create agent
    agent = PreprocessingAgent(logger, llm)

    # Run preprocessing
    result_state = agent.run(
        data_path=data_path,
        prompt="Clean and preprocess the Titanic dataset for binary classification",
        task="Preprocess Titanic dataset for model training",
        target_column="Survived",
        test_size=0.2,
        use_llm=True,
    )

    # Display results
    print("\n" + "=" * 70)
    print("PREPROCESSING RESULTS")
    print("=" * 70)

    status = result_state.get("status", "unknown")
    print(f"\n[OK] Status: {status}")
    print(f"[OK] Step: {result_state.get('step')}")

    if status == "success":
        output = result_state.get("preprocessing_output", {})
        print(f"\n[DATA] Preprocessing Outputs:")
        print(f"   * X_train: {output.get('X_train_path')}")
        print(f"   * X_test:  {output.get('X_test_path')}")
        print(f"   * y_train: {output.get('y_train_path')}")
        print(f"   * y_test:  {output.get('y_test_path')}")
        print(f"\n[METADATA] Metadata:")
        print(f"   * Summary: {output.get('summary_path')}")
        print(f"   * Policy:  {output.get('policy_path')}")
        print(
            f"   * Column Actions (Frontend): {output.get('column_actions_frontend_path')}")

        # Try to load and display summary
        summary_path = output.get("summary_path")
        if summary_path and Path(summary_path).exists():
            import json
            with open(summary_path) as f:
                summary = json.load(f)
            print(f"\n[OK] Training Data Summary:")
            print(f"   * Rows: {summary.get('train_rows')}")
            print(f"   * Features: {summary.get('n_features')}")
            print(f"   * Target: {summary.get('target_column')}")
            print(f"   * Task: {summary.get('task_type')}")
    else:
        error = result_state.get("error", "Unknown error")
        print(f"\n[ERROR] Error: {error}")

    print("\n" + "=" * 70 + "\n")
    return result_state


def test_preprocessing_agent_with_feedback():
    """Test the PreprocessingAgent with feedback (simulated)."""
    print("\n" + "=" * 70)
    print("PREPROCESSING AGENT — FEEDBACK TEST")
    print("=" * 70 + "\n")

    logger = Logger()
    llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )

    data_path = str(
        PROJECT_ROOT / "assets/data/Regression Datasets/Medical Insurance Cost.csv"
    )

    if not Path(data_path).exists():
        print(f"[ERROR] Dataset not found: {data_path}")
        return

    agent = PreprocessingAgent(logger, llm)

    # First run with default settings
    print("-> First preprocessing run (default settings)...")
    result_state = agent.run(
        data_path=data_path,
        prompt="Preprocess insurance cost dataset for regression prediction",
        task="Preprocess Medical Insurance dataset",
        target_column="charges",
        test_size=0.2,
        use_llm=True,
    )

    status = result_state.get("status")
    print(f"[OK] First run status: {status}")

    if status == "success":
        print("\n[OK] Preprocessing completed successfully!")
        print(
            f"[OK] Output folder: {result_state.get('preprocessing_output', {}).get('summary_path', 'N/A')}")
    else:
        print(f"[ERROR] First run failed: {result_state.get('error')}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        # Run basic test
        test_preprocessing_agent_basic()

        # Run feedback test (simulated)
        test_preprocessing_agent_with_feedback()

        print("[OK] All tests completed!")
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Test failed with error:")
        print(traceback.format_exc())
        sys.exit(1)
