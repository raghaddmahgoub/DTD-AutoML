import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================================
# CONFIGURATION BLOCK - Edit these paths for your project
# ============================================================================
PREPROCESSING_CONFIG = {
    # Input/Output Paths
    "default_input_path": "Datasets/",  # Default folder for input datasets
    "default_output_path": "output/",  # Default folder for outputs
    
    # Preprocessing Parameters
    "test_size": 0.2,  # Train/test split ratio (0.0 - 1.0)
    "random_state": 42,  # Random seed for reproducibility
    "use_llm": True,  # Enable LLM for ambiguous decisions (requires GEMINI_API_KEY)
    "hash_features": 8,  # Number of features for hash encoding
    "max_label_categories": 30,  # Max categories before switching to hash
    
    # API Configuration
    "gemini_api_key_env": "GEMINI_API_KEY",  # Environment variable name for Gemini API key
}
# ============================================================================
#
# USAGE EXAMPLES:
#
# 1. Command Line:
#    python preprocessing_node.py Datasets/data.csv target_column output_folder
#
# 2. Python Script:
#    from preprocessing_node import preprocessing_node
#    result = preprocessing_node({
#        "dataset_path": "Datasets/data.csv",
#        "target_column": "target"
#    })
#
# 3. LangGraph Node:
#    from preprocessing_node import preprocessing_node, PreprocessingState
#    from langgraph.graph import StateGraph
#    workflow = StateGraph(PreprocessingState)
#    workflow.add_node("preprocess", preprocessing_node)
#
# 4. Custom Config:
#    from preprocessing_node import PreprocessingNode
#    custom_config = {"default_output_path": "custom_output/", "test_size": 0.3}
#    node = PreprocessingNode(config=custom_config)
#    result = node.run({"dataset_path": "data.csv", "target_column": "target"})
#
# ============================================================================


class PreprocessingState(TypedDict):
    """State object for LangGraph node."""

    dataset_path: str
    target_column: str
    output_folder: str
    test_size: float
    random_state: int
    use_llm: bool
    hash_features: int
    max_label_categories: int
    X_train_path: Optional[str]
    X_test_path: Optional[str]
    y_train_path: Optional[str]
    y_test_path: Optional[str]
    summary_path: Optional[str]
    column_actions_path: Optional[str]
    status: str
    error: Optional[str]


class PreprocessingNode:
    """
    Preprocessing node for LangGraph orchestrator.

    Usage:
        node = PreprocessingNode()
        state = {
            "dataset_path": "data.csv",
            "target_column": "target",
            "output_folder": "output",
        }
        result = node.run(state)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize preprocessing node with optional custom config."""
        self.config = config or PREPROCESSING_CONFIG.copy()
        api_key_env = self.config.get("gemini_api_key_env", "GEMINI_API_KEY")
        self.api_key = os.getenv(api_key_env)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for LangGraph node.

        Args:
            state: Dictionary with keys:
                - dataset_path (required): Path to CSV dataset
                - target_column (required): Name of target column
                - output_folder (optional): Output directory, default "output"
                - test_size (optional): Test split ratio, default 0.2
                - random_state (optional): Random seed, default 42
                - use_llm (optional): Enable LLM, default True if API key exists
                - hash_features (optional): Hash feature count, default 8
                - max_label_categories (optional): Max label encoding size, default 30

        Returns:
            Updated state dictionary with output paths and status
        """
        try:
            # Extract parameters with config defaults
            dataset_path = Path(state["dataset_path"])
            target_col = state["target_column"]
            output_folder = Path(
                state.get("output_folder", self.config.get(
                    "default_output_path", "output"))
            )
            test_size = state.get(
                "test_size", self.config.get("test_size", 0.2))
            random_state = state.get(
                "random_state", self.config.get("random_state", 42))
            use_llm = state.get("use_llm", self.config.get(
                "use_llm", bool(self.api_key)))
            hash_features = state.get(
                "hash_features", self.config.get("hash_features", 8))
            max_label_categories = state.get(
                "max_label_categories", self.config.get(
                    "max_label_categories", 30)
            )

            # Create output folder
            output_folder.mkdir(parents=True, exist_ok=True)

            # Load dataset
            df = pd.read_csv(dataset_path)
            self._validate_target_column(df, target_col)

            # Preprocess
            X, y, metadata = self._preprocess(
                df=df,
                target_col=target_col,
                use_llm=use_llm,
                hash_features=hash_features,
                max_label_categories=max_label_categories,
            )
            full_df = X.copy()
            full_df[target_col] = y.reset_index(drop=True)
            full_path = output_folder / "full_preprocessed.csv"
            full_df.to_csv(full_path, index=False)

            # Split
            # X_train, X_test, y_train, y_test = train_test_split(
            #     X,
            #     y,
            #     test_size=test_size,
            #     random_state=random_state,
            #     stratify=y if y.nunique() <= 20 else None,
            # )
            # =========================
            # SAFE STRATIFIED SPLIT
            # =========================

            stratify_target = None

            if y.nunique() <= 20:
                class_counts = y.value_counts()

                if class_counts.min() < 2:
                    print("⚠️ Rare classes detected (<2 samples). Dropping them...")
                    valid_classes = class_counts[class_counts >= 2].index
                    mask = y.isin(valid_classes)

                    X = X[mask]
                    y = y[mask]

                    stratify_target = y
                else:
                    stratify_target = y

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_target,
            )

            # Save outputs
            paths = self._save_outputs(
                output_folder=output_folder,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                metadata=metadata,
                dataset_name=dataset_path.name,
                target_col=target_col,
                test_size=test_size,
                random_state=random_state,
                use_llm=use_llm,
                hash_features=hash_features,
            )

            # Update state
            state.update(
                {
                    "X_train_path": str(paths["X_train"]),
                    "X_test_path": str(paths["X_test"]),
                    "y_train_path": str(paths["y_train"]),
                    "y_test_path": str(paths["y_test"]),
                    "full_dataset_path": str(full_path),
                    "summary_path": str(paths["summary"]),
                    "column_actions_path": str(paths["column_actions"]),
                    "status": "success",
                    "error": None,
                    "output_folder": str(output_folder),
                }
            )

            return state

        except Exception as e:
            state.update(
                {
                    "status": "failed",
                    "error": str(e),
                }
            )
            return state

    def _validate_target_column(self, df: pd.DataFrame, target_col: str) -> None:
        if target_col not in df.columns:
            available = ", ".join(df.columns)
            raise ValueError(
                f"Target column '{target_col}' not found. Available: {available}"
            )

    def _is_id_like(self, series: pd.Series) -> bool:
        if series.dtype == "object":
            return False
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        return unique_ratio > 0.98

    def _is_high_cardinality(self, series: pd.Series) -> bool:
        unique_count = series.nunique(dropna=True)
        unique_ratio = unique_count / max(len(series), 1)
        return unique_count > 50 and unique_ratio > 0.2

    def _call_gemini(self, prompt: str) -> Optional[str]:
        if not self.api_key:
            return None
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-1.5-flash:generateContent?key=" + self.api_key
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                body = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError):
            return None
        try:
            parsed = json.loads(body)
            return (
                parsed.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text")
            )
        except (json.JSONDecodeError, IndexError, KeyError, TypeError):
            return None

    def _parse_llm_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start: end + 1])
        except json.JSONDecodeError:
            return None

    def _llm_decide_categorical_action(
        self,
        series: pd.Series,
        max_label_categories: int,
        hash_features: int,
    ) -> Optional[Dict[str, Any]]:
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = unique_count / max(len(series), 1)
        missing_ratio = float(series.isna().mean())
        sample_values = (
            series.dropna().astype(str).head(5).tolist()
            if not series.dropna().empty
            else []
        )
        prompt = (
            "You are a data preprocessing assistant. Choose an action for a categorical column. "
            "Allowed actions: drop, hash, label. Avoid creating huge features. "
            "If unique_count is high, prefer drop or hash. "
            f"Column stats: unique_count={unique_count}, unique_ratio={unique_ratio:.2f}, "
            f"missing_ratio={missing_ratio:.2f}, max_label_categories={max_label_categories}, "
            f"hash_features={hash_features}. Sample values: {sample_values}. "
            "Respond with JSON only: {\"action\":\"drop|hash|label\",\"reason\":\"...\"}."
        )
        response = self._call_gemini(prompt)
        decision = self._parse_llm_json(response) if response else None
        if not decision or "action" not in decision:
            return None
        action = str(decision.get("action", "")).lower().strip()
        if action not in {"drop", "hash", "label"}:
            return None
        return {
            "action": action,
            "reason": str(decision.get("reason", "llm_decision")),
            "raw": decision,
        }

    def _preprocess(
        self,
        df: pd.DataFrame,
        target_col: str,
        use_llm: bool,
        hash_features: int,
        max_label_categories: int,
    ) -> tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Perform preprocessing and return X, y, and metadata."""
        column_actions = {}
        dropped_columns = []
        numeric_cols = []
        categorical_cols = []
        hashed_cols = []
        label_cols = []
        frames = []

        for col in df.columns:
            if col == target_col:
                continue

            series = df[col]
            missing_ratio = series.isna().mean()
            unique_ratio = series.nunique(dropna=True) / max(len(series), 1)

            # Drop high missing
            if missing_ratio > 0.6:
                dropped_columns.append(col)
                column_actions[col] = {
                    "action": "drop",
                    "reason": f"missing_ratio>{missing_ratio:.2f}",
                }
                continue

            # Drop ID-like
            if self._is_id_like(series):
                dropped_columns.append(col)
                column_actions[col] = {
                    "action": "drop",
                    "reason": "id_like_numeric",
                }
                continue

            # Handle categorical
            if series.dtype == "object":
                if unique_ratio > 0.95:
                    dropped_columns.append(col)
                    column_actions[col] = {
                        "action": "drop",
                        "reason": f"high_cardinality_ratio>{unique_ratio:.2f}",
                    }
                    continue

                action = "label"
                reason = "low_cardinality_categorical"
                llm_info = None

                if self._is_high_cardinality(series):
                    action = "drop"
                    reason = "high_cardinality_categorical"
                    if use_llm and self.api_key:
                        llm_info = self._llm_decide_categorical_action(
                            series,
                            max_label_categories=max_label_categories,
                            hash_features=hash_features,
                        )
                        if llm_info:
                            action = llm_info["action"]
                            reason = llm_info["reason"]

                unique_count = int(series.nunique(dropna=True))
                if action == "label" and unique_count > max_label_categories:
                    action = "hash"
                    reason = "label_too_many_categories"

                if action == "drop":
                    dropped_columns.append(col)
                    column_actions[col] = {
                        "action": "drop",
                        "reason": reason,
                        "llm_used": bool(llm_info),
                    }
                    continue

                if action == "hash":
                    cleaned = series.fillna("missing").astype(str)
                    hasher = FeatureHasher(
                        n_features=hash_features, input_type="string"
                    )
                    # FeatureHasher expects iterable of iterables
                    hashed = hasher.transform([[val] for val in cleaned]).toarray()
                    hash_frame = pd.DataFrame(
                        hashed,
                        columns=[f"{col}__hash_{i}" for i in range(
                            hash_features)],
                    )
                    frames.append(hash_frame)
                    hashed_cols.append(col)
                    column_actions[col] = {
                        "type": "categorical",
                        "imputation": "missing",
                        "encoding": "hash",
                        "hash_features": hash_features,
                        "reason": reason,
                        "llm_used": bool(llm_info),
                    }
                    continue

                # Label encode
                cleaned = series.fillna("missing").astype(str)
                categories = sorted(cleaned.unique().tolist())
                mapping = {cat: idx for idx, cat in enumerate(categories)}
                encoded = cleaned.map(mapping).astype(int)
                frames.append(pd.DataFrame({col: encoded}))
                categorical_cols.append(col)
                label_cols.append(col)
                column_actions[col] = {
                    "type": "categorical",
                    "imputation": "missing",
                    "encoding": "label",
                    "mapping": mapping,
                    "reason": reason,
                    "llm_used": bool(llm_info),
                }
                continue

            # Handle numeric
            numeric_series = pd.to_numeric(series, errors="coerce")
            median = (
                float(numeric_series.median())
                if not numeric_series.dropna().empty
                else 0.0
            )
            numeric_series = numeric_series.fillna(median)
            frames.append(pd.DataFrame({col: numeric_series}))
            numeric_cols.append(col)
            column_actions[col] = {
                "type": "numeric",
                "imputation": "median",
                "imputation_value": median,
                "encoding": None,
            }

        X = pd.concat(frames, axis=1) if frames else pd.DataFrame()
        y = df[target_col]

        # Standardize numeric features
        scaler = None
        if numeric_cols:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        metadata = {
            "column_actions": column_actions,
            "dropped_columns": dropped_columns,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "label_encoded_columns": label_cols,
            "hashed_columns": hashed_cols,
            "scaler": {
                "means": scaler.mean_.tolist() if scaler else [],
                "scales": scaler.scale_.tolist() if scaler else [],
                "columns": numeric_cols,
            },
        }

        return X, y, metadata

    def _save_outputs(
        self,
        output_folder: Path,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        metadata: Dict[str, Any],
        dataset_name: str,
        target_col: str,
        test_size: float,
        random_state: int,
        use_llm: bool,
        hash_features: int,
    ) -> Dict[str, Path]:
        """Save all outputs and return file paths."""
        X_train_path = output_folder / "X_train.csv"
        X_test_path = output_folder / "X_test.csv"
        y_train_path = output_folder / "y_train.csv"
        y_test_path = output_folder / "y_test.csv"
        summary_path = output_folder / "preprocessing_summary.json"
        column_actions_path = output_folder / "column_actions.json"

        X_train.to_csv(X_train_path, index=False)
        X_test.to_csv(X_test_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        y_test.to_csv(y_test_path, index=False)

        summary = {
            "dataset": dataset_name,
            "target_column": target_col,
            "rows": len(X_train) + len(X_test),
            "features": int(X_train.shape[1]),
            "numeric_columns": metadata["numeric_columns"],
            "categorical_columns": metadata["categorical_columns"],
            "label_encoded_columns": metadata["label_encoded_columns"],
            "hashed_columns": metadata["hashed_columns"],
            "dropped_columns": metadata["dropped_columns"],
            "test_size": test_size,
            "random_state": random_state,
            "llm": {
                "enabled": use_llm,
                "provider": "gemini",
                "model": "gemini-1.5-flash",
                "env_var": "GEMINI_API_KEY",
            },
            "scaling": {
                "method": "standard",
                "means": metadata["scaler"]["means"],
                "scales": metadata["scaler"]["scales"],
                "columns": metadata["scaler"]["columns"],
            },
            "hashing": {
                "features": hash_features,
                "columns": metadata["hashed_columns"],
            },
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        with open(column_actions_path, "w", encoding="utf-8") as f:
            json.dump(metadata["column_actions"], f, indent=2)

        return {
            "X_train": X_train_path,
            "X_test": X_test_path,
            "y_train": y_train_path,
            "y_test": y_test_path,
            "summary": summary_path,
            "column_actions": column_actions_path,
        }


# LangGraph node function
def preprocessing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node wrapper function.

    Example usage in LangGraph:
        from langgraph.graph import StateGraph
        from preprocessing_node import preprocessing_node, PreprocessingState

        workflow = StateGraph(PreprocessingState)
        workflow.add_node("preprocess", preprocessing_node)
        workflow.set_entry_point("preprocess")
        workflow.set_finish_point("preprocess")
        app = workflow.compile()

        result = app.invoke({
            "dataset_path": "data.csv",
            "target_column": "target",
        })
    """
    node = PreprocessingNode()
    return node.run(state)


if __name__ == "__main__":
    # Example standalone usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python preprocessing_node.py <dataset_path> <target_column>")
        sys.exit(1)

    state = {
        "dataset_path": sys.argv[1],
        "target_column": sys.argv[2],
        "output_folder": sys.argv[3] if len(sys.argv) > 3 else "output",
    }

    result = preprocessing_node(state)

    if result["status"] == "success":
        print(f"✓ Preprocessing successful")
        print(f"  Output folder: {result['output_folder']}")
        print(f"  X_train: {result['X_train_path']}")
        print(f"  Summary: {result['summary_path']}")
    else:
        print(f"✗ Preprocessing failed: {result['error']}")
        sys.exit(1)