"""
Model Training & Selection Agent
Loads preprocessed data from Output folder, trains multiple models, picks the best one.
All in one file - no LLM needed.

Usage:
    python agents/dynamic/model_training/train_and_select.py
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import os
import json
import pickle
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ======================================================================
# CONFIGURATION BLOCK
# ======================================================================

# Data Configuration
DATA_FOLDER = "Output/Preprocessing/Titanic-Dataset"  # Where preprocessed data is stored
DATASET_NAME = "Titanic-Dataset"

# Model Configuration
MODELS_TO_TRY = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "SVM": SVC(kernel='rbf', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
}

# Output Configuration
OUTPUT_FOLDER = "Output/Model_Training"
VERBOSE = True

# ======================================================================
# END CONFIGURATION BLOCK
# ======================================================================


def load_preprocessed_data(data_folder):
    """
    Load preprocessed train/test data from Output folder.
    
    Args:
        data_folder: Path to folder containing X_train.csv, X_test.csv, etc.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    data_path = Path(data_folder)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {data_path}")
    
    X_train_path = data_path / "X_train.csv"
    X_test_path = data_path / "X_test.csv"
    y_train_path = data_path / "y_train.csv"
    y_test_path = data_path / "y_test.csv"
    
    if not all([X_train_path.exists(), X_test_path.exists(), y_train_path.exists(), y_test_path.exists()]):
        raise FileNotFoundError(f"Missing preprocessed data files in {data_path}")
    
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0]  # First column
    y_test = pd.read_csv(y_test_path).iloc[:, 0]     # First column
    
    if VERBOSE:
        print(f"[+] Loaded training data: {X_train.shape}")
        print(f"[+] Loaded test data: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate_models(X_train, X_test, y_train, y_test, models):
    """
    Train all models and evaluate their performance.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        models: Dict of {name: model_instance}
    
    Returns:
        Dict with results for each model
    """
    results = {}
    
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)
    
    for model_name, model in models.items():
        print(f"\n[*] Training {model_name}...", end=" ", flush=True)
        
        try:
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluate
            results[model_name] = {
                "model": model,
                "train_accuracy": accuracy_score(y_train, y_pred_train),
                "test_accuracy": accuracy_score(y_test, y_pred_test),
                "precision": precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
                "f1": f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
            }
            
            print(f"Done! Test Accuracy: {results[model_name]['test_accuracy']:.4f}")
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    return results


def select_best_model(results):
    """
    Select the best model based on test accuracy.
    
    Args:
        results: Dict with model results
    
    Returns:
        Tuple of (best_model_name, best_results)
    """
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if not valid_results:
        raise RuntimeError("No valid models were trained")
    
    best_model_name = max(valid_results, key=lambda x: valid_results[x]["test_accuracy"])
    best_results = valid_results[best_model_name]
    
    return best_model_name, best_results


def save_results(best_model_name, best_results, all_results, dataset_name, output_folder):
    """
    Save trained model and results to disk.
    
    Args:
        best_model_name: Name of best model
        best_results: Results of best model
        all_results: Results of all models
        dataset_name: Dataset name for naming
        output_folder: Where to save
    """
    output_path = Path(output_folder) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save best model
    model_path = output_path / f"best_model_{best_model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_results["model"], f)
    
    # Save results JSON
    results_data = {}
    for model_name, result in all_results.items():
        if "error" not in result:
            results_data[model_name] = {
                "train_accuracy": float(result["train_accuracy"]),
                "test_accuracy": float(result["test_accuracy"]),
                "precision": float(result["precision"]),
                "recall": float(result["recall"]),
                "f1": float(result["f1"]),
            }
        else:
            results_data[model_name] = {"error": result["error"]}
    
    results_path = output_path / "model_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    
    # Save summary
    summary_path = output_path / "best_model_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "best_model": best_model_name,
        "test_accuracy": float(best_results["test_accuracy"]),
        "train_accuracy": float(best_results["train_accuracy"]),
        "precision": float(best_results["precision"]),
        "recall": float(best_results["recall"]),
        "f1": float(best_results["f1"]),
        "model_path": str(model_path),
        "results_path": str(results_path),
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    if VERBOSE:
        print(f"\n[+] Model saved to: {model_path}")
        print(f"[+] Results saved to: {results_path}")
        print(f"[+] Summary saved to: {summary_path}")
    
    return summary


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("MODEL TRAINING & SELECTION AGENT")
    print("=" * 70)
    
    try:
        # Load data
        print("\n[*] Loading preprocessed data...")
        X_train, X_test, y_train, y_test = load_preprocessed_data(DATA_FOLDER)
        
        # Train models
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test, MODELS_TO_TRY)
        
        # Select best
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        
        for model_name, result in results.items():
            if "error" not in result:
                print(f"\n{model_name}:")
                print(f"  Train Accuracy: {result['train_accuracy']:.4f}")
                print(f"  Test Accuracy:  {result['test_accuracy']:.4f}")
                print(f"  Precision:      {result['precision']:.4f}")
                print(f"  Recall:         {result['recall']:.4f}")
                print(f"  F1-Score:       {result['f1']:.4f}")
            else:
                print(f"\n{model_name}: ERROR - {result['error']}")
        
        best_model_name, best_results = select_best_model(results)
        
        print("\n" + "=" * 70)
        print(f"BEST MODEL: {best_model_name}")
        print("=" * 70)
        print(f"Test Accuracy: {best_results['test_accuracy']:.4f}")
        print(f"Precision: {best_results['precision']:.4f}")
        print(f"Recall: {best_results['recall']:.4f}")
        print(f"F1-Score: {best_results['f1']:.4f}")
        
        # Save results
        summary = save_results(best_model_name, best_results, results, DATASET_NAME, OUTPUT_FOLDER)
        
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"Best model: {best_model_name}")
        print(f"Accuracy: {best_results['test_accuracy']:.4f}")
        print(f"Output folder: {OUTPUT_FOLDER}/{DATASET_NAME}/")
        print("=" * 70 + "\n")
        
        return summary
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    main()
