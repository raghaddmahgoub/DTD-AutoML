"""Feature engineering tool with AI-guided feature selection."""
from langchain_core.tools import tool
from pathlib import Path
import json
import pandas as pd
import numpy as np

from tools.pipeline_state import ensure_state, merge_state


@tool
def feature_engineering_execution(task, tool_input, prompt, data_path, llm, state=None):
    """
    AI-guided feature engineering: LLM analyzes column names → select top 3-4 by correlation.

    Inputs (via tool_input):
    - X_train_path: str - Path to training features
    - X_test_path: str - Path to test features
    - y_train_path: str - Path to training target
    - y_test_path: str - Path to test target (optional)
    - top_k: int (default 4) - Select top K features (3-4 recommended)
    - use_llm: bool (default True) - Use LLM to suggest meaningful columns
    - output_folder: str (optional)

    Returns:
    - status: success/error
    - X_train_engineered_path, X_test_engineered_path
    - feature_summary with:
      - tried_columns: all columns analyzed
      - correlations: correlation of each to target
      - selected_features: final 3-4 selected
      - selection_reasoning: why LLM selected them
    """
    pipeline_state = ensure_state(state, data_path, prompt)

    try:
        # Extract inputs from state if not provided
        X_train_path = tool_input.get("X_train_path") or pipeline_state.get("X_train_path")
        X_test_path = tool_input.get("X_test_path") or pipeline_state.get("X_test_path")
        y_train_path = tool_input.get("y_train_path") or pipeline_state.get("y_train_path")

        top_k = int(tool_input.get("top_k", 4))
        use_llm = bool(tool_input.get("use_llm", True))
        output_folder = tool_input.get("output_folder") or str(
            Path("Output") / "FeatureEngineering" / Path(data_path).stem
        )

        # Validate paths exist
        if not Path(X_train_path).exists():
            raise FileNotFoundError(f"X_train not found: {X_train_path}")
        if not Path(X_test_path).exists():
            raise FileNotFoundError(f"X_test not found: {X_test_path}")
        if not Path(y_train_path).exists():
            raise FileNotFoundError(f"y_train not found: {y_train_path}")

        # Load data
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path).iloc[:, 0]

        all_columns = X_train.columns.tolist()

        # Step 1: Calculate correlations for all columns
        correlations = _calculate_correlations(X_train, y_train)

        # Step 2: Get LLM suggestions (if enabled)
        llm_suggestions = []
        selection_reasoning = ""
        if use_llm and llm:
            llm_suggestions, selection_reasoning = _get_llm_suggestions(
                all_columns, correlations, llm
            )

        # Step 3: Select top K by correlation
        if llm_suggestions:
            # If LLM suggested columns, use those and rank by correlation
            suggested_cols = [col for col in llm_suggestions if col in all_columns]
            selected_features = sorted(
                suggested_cols,
                key=lambda x: abs(correlations.get(x, 0)),
                reverse=True
            )[:top_k]
        else:
            # Otherwise, select top K by correlation
            selected_features = sorted(
                all_columns,
                key=lambda x: abs(correlations.get(x, 0)),
                reverse=True
            )[:top_k]

        # Filter to selected features
        X_train_engineered = X_train[selected_features]
        X_test_engineered = X_test[selected_features]

        # Create output folder
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save engineered features
        X_train_engineered_path = str(output_path / "X_train_engineered.csv")
        X_test_engineered_path = str(output_path / "X_test_engineered.csv")
        feature_summary_path = str(output_path / "feature_summary.json")

        X_train_engineered.to_csv(X_train_engineered_path, index=False)
        X_test_engineered.to_csv(X_test_engineered_path, index=False)

        # Build comprehensive summary
        summary = {
            "status": "success",
            "tried_columns": all_columns,
            "correlations": {col: float(correlations.get(col, 0)) for col in all_columns},
            "selected_features": selected_features,
            "selected_correlations": {col: float(correlations.get(col, 0)) for col in selected_features},
            "n_original_features": len(X_train.columns),
            "n_engineered_features": len(selected_features),
            "use_llm": use_llm,
            "selection_reasoning": selection_reasoning,
            "X_train_shape": list(X_train_engineered.shape),
            "X_test_shape": list(X_test_engineered.shape),
        }

        with open(feature_summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Update pipeline state
        pipeline_state = merge_state(
            pipeline_state,
            {
                "step": "feature_engineering_complete",
                "status": "success",
                "feature_engineering_output": {
                    "X_train_engineered_path": X_train_engineered_path,
                    "X_test_engineered_path": X_test_engineered_path,
                    "feature_summary_path": feature_summary_path,
                },
                "feature_names": selected_features,
                "n_features": len(selected_features),
            }
        )

        result = {
            "status": "success",
            "message": f"Feature engineering completed: {len(X_train.columns)} → {len(selected_features)} features",
            "feature_engineering_output": {
                "X_train_engineered_path": X_train_engineered_path,
                "X_test_engineered_path": X_test_engineered_path,
                "feature_summary_path": feature_summary_path,
                "tried_columns": all_columns,
                "selected_features": selected_features,
                "correlations": {col: float(correlations.get(col, 0)) for col in all_columns},
            },
        }

        return result, pipeline_state

    except Exception as e:
        import traceback
        error_msg = f"Feature engineering failed: {str(e)}\n{traceback.format_exc()}"
        pipeline_state["step"] = "feature_engineering_failed"
        pipeline_state["error"] = error_msg
        result = {"status": "error", "error": error_msg}
        return result, pipeline_state


def _calculate_correlations(X, y):
    """Calculate correlation of each feature to target."""
    correlations = {}
    for col in X.columns:
        try:
            if pd.api.types.is_numeric_dtype(X[col]):
                corr = X[col].corr(y)
                correlations[col] = corr if not np.isnan(corr) else 0
            else:
                correlations[col] = 0  # Non-numeric: correlation is 0
        except:
            correlations[col] = 0
    return correlations


def _get_llm_suggestions(columns, correlations, llm):
    """
    Use LLM to suggest meaningful columns based on names and correlations.
    
    Returns:
    - suggested_columns: list of suggested column names
    - reasoning: explanation from LLM
    """
    try:
        # Prepare data for LLM
        col_info = []
        for col in columns:
            corr = correlations.get(col, 0)
            col_info.append(f"{col} (correlation: {corr:.3f})")

        prompt = f"""
Analyze these dataset columns and their correlations to the target variable.
Select the 3-4 MOST MEANINGFUL columns for a machine learning model.
Prioritize high correlation AND semantic importance.

Columns with correlations:
{chr(10).join(col_info)}

Return ONLY a JSON object like:
{{
  "selected_columns": ["col1", "col2", "col3"],
  "reasoning": "Brief explanation"
}}
"""

        response = llm.invoke(prompt)
        
        # Parse response
        import re
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result.get("selected_columns", []), result.get("reasoning", "")
        
        return [], ""

    except Exception as e:
        return [], f"LLM suggestion failed: {str(e)}"


def _generate_interactions(X_train, X_test, max_features):
    """
    Generate interaction features (polynomial features, 2-way interactions).
    Keeps top features by variance.
    """
    from sklearn.preprocessing import PolynomialFeatures

    # Only use numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_numeric = X_train[numeric_cols]
    X_test_numeric = X_test[numeric_cols]

    # Generate polynomial features (degree 2 = interactions)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_numeric)
    X_test_poly = poly.transform(X_test_numeric)

    # Get feature names
    feature_names = poly.get_feature_names_out(numeric_cols)

    # Convert back to DataFrame
    X_train_poly = pd.DataFrame(X_train_poly, columns=feature_names)
    X_test_poly = pd.DataFrame(X_test_poly, columns=feature_names)

    # Add non-numeric columns back
    non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        X_train_poly = pd.concat([X_train_poly, X_train[non_numeric_cols].reset_index(drop=True)], axis=1)
        X_test_poly = pd.concat([X_test_poly, X_test[non_numeric_cols].reset_index(drop=True)], axis=1)

    return X_train_poly, X_test_poly


def _select_features(X, y, method="variance", max_features=20):
    """
    Select features based on method: variance, correlation, or RFE.
    """
    if method == "variance":
        from sklearn.feature_selection import VarianceThreshold

        selector = VarianceThreshold(threshold=0.01)
        X_selected = selector.fit_transform(X)
        selected_cols = X.columns[selector.get_support()].tolist()

    elif method == "correlation":
        # Remove features with low correlation to target
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        selected_cols = correlations.head(max_features).index.tolist()

    elif method == "rfe":
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        selector = RFE(rf, n_features_to_select=min(max_features, len(X.columns)))
        selector.fit(X, y)
        selected_cols = X.columns[selector.get_support()].tolist()

    else:
        selected_cols = X.columns.tolist()[:max_features]

    return selected_cols
