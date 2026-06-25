"""
Preprocessing Node — Gemini LLM Policy Driven
=============================================
Edit RUN_CONFIG below, then run:  python preprocessing_node.py
"""

# ---------------------------------------------------------------------------
# RUN CONFIG — edit these values before running
# ---------------------------------------------------------------------------
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, TypedDict
from pathlib import Path
import warnings
import urllib.request
import urllib.error
import re
import os
import logging
import json
from dotenv import load_dotenv
from tools.shared.llm_fallback import call_qwen_fallback

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

RUN_CONFIG = {
    # path to your CSV
    "dataset_path":  "assets\data\Datasets\Regression Datasets\Medical Insurance Cost.csv",
    "target_column": "charges",                       # column to predict
    "use_llm":       True,                           # False = skip Gemini call
}
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Logging — terminal output only
# ---------------------------------------------------------------------------
logger = logging.getLogger("preprocessing_node")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    _fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    _sh = logging.StreamHandler()
    _sh.setLevel(logging.INFO)
    _sh.setFormatter(_fmt)
    logger.addHandler(_sh)


PREPROCESSING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "use_llm": True,
    "llm_final_decision": True,
    "safe_mode": True,
    "max_label_categories": 30,
    "max_onehot_categories": 3,
    "high_cardinality_unique_count": 100,
    "high_cardinality_unique_ratio": 0.98,
    "hash_features": 8,
    "max_row_drop_fraction": 0.02,
    "max_outlier_clip_quantile": 0.01,
    "target_metric_priority": "f1",
    "gemini_model": "gemini-2.5-flash",
    "google_api_key_env": "GOOGLE_API_KEY",
}


class PreprocessingState(TypedDict):
    dataset_path: str
    target_column: str
    output_folder: str
    test_size: float
    random_state: int
    use_llm: bool
    X_train_path: Optional[str]
    X_test_path: Optional[str]
    y_train_path: Optional[str]
    y_test_path: Optional[str]
    summary_path: Optional[str]
    column_actions_path: Optional[str]
    column_actions_frontend_path: Optional[str]
    policy_path: Optional[str]
    evidence_path: Optional[str]
    status: str
    error: Optional[str]


class PreprocessingNode:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = PREPROCESSING_CONFIG.copy()
        if config:
            self.config.update(config)
        # Read API credentials from environment only.
        self.google_api_key = os.getenv(
            self.config.get("google_api_key_env", "GOOGLE_API_KEY")
        ) or os.getenv("GOOGLE_API_KEY")
        self._last_gemini_error = ""
        logger.debug("PreprocessingNode initialised | gemini_credentials=%s", bool(
            self.google_api_key))

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            dataset_path = Path(state["dataset_path"])
            target_col = state["target_column"]
            dataset_stem = Path(state["dataset_path"]).stem
            output_folder = Path(
                state.get("output_folder", PROJECT_ROOT / "Output" / "static" / "Preprocessing" / dataset_stem))
            test_size = float(state.get("test_size", self.config["test_size"]))
            random_state = int(
                state.get("random_state", self.config["random_state"]))
            use_llm = bool(state.get("use_llm", self.config["use_llm"]))

            output_folder.mkdir(parents=True, exist_ok=True)
            logger.info("[START] dataset=%s  target=%s  output=%s",
                        dataset_path.name, target_col, output_folder)

            df = pd.read_csv(dataset_path)
            logger.info("Loaded dataset: %d rows x %d cols",
                        len(df), len(df.columns))
            self._validate_target_column(df, target_col)
            logger.info("CHECK: target column exists -> %s", target_col)

            logger.info("Building evidence profile...")
            evidence = self._build_evidence(df, target_col)
            logger.debug("Evidence: %d cols profiled | imbalance_ratio=%.2f | duplicates=%d",
                         evidence["columns"], evidence["target"]["imbalance_ratio"], evidence["duplicate_rows"])
            logger.info("CHECK: evidence profile completed")

            logger.info("Generating default policy...")
            default_policy = self._default_policy(df, target_col, evidence)

            llm_policy = None
            if use_llm and self._has_llm_credentials():
                logger.info("Calling LLM (gemini / %s) for policy decision...",
                            self.config.get("gemini_model"))
                llm_policy = self._llm_decide_policy(evidence, target_col)
                if llm_policy:
                    logger.info("LLM policy received and will be applied")
                else:
                    logger.warning(
                        "LLM returned no valid policy — falling back to defaults")
            else:
                logger.info("LLM skipped (use_llm=%s, credentials=%s) — using defaults",
                            use_llm, self._has_llm_credentials())

            logger.info("Merging and validating policy...")
            policy = self._merge_and_validate_policy(
                default_policy=default_policy,
                llm_policy=llm_policy,
                evidence=evidence,
                total_rows=len(df),
            )
            safeguard_notes = policy.get("safeguards", {}).get("notes", [])
            if safeguard_notes:
                for note in safeguard_notes:
                    logger.warning("SAFEGUARD: %s", note)
            logger.info(
                "CHECK: policy validated with %d safeguard note(s)", len(safeguard_notes))

            logger.info("Running preprocessing pipeline...")
            X, y, metadata = self._preprocess_with_policy(
                df, target_col, policy)
            metadata["llm_policy_used"] = bool(llm_policy)
            logger.info("Preprocessing done: %d rows x %d features | dropped=%s",
                        len(X), X.shape[1], metadata["dropped_columns"])
            logger.debug("Steps status: %s", metadata["steps_status"])
            logger.info("CHECK: preprocessing completed successfully")

            # clean target before split
            n_before = len(y)
            valid_idx = ~y.isna()
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
            n_after = len(y)

            logger.warning(f"Dropped {n_before - n_after} rows due to missing target")

            logger.info(
                "Splitting train/test (test_size=%.0f%%)...", test_size * 100)
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y if y.nunique() <= 20 else None,
            )
            logger.info("Split done: train=%d rows | test=%d rows",
                        len(X_train), len(X_test))
            logger.info("CHECK: train/test split completed")

            logger.info("Applying imbalance method: %s",
                        policy["imbalance"]["method"])
            X_train, y_train, imbalance_meta = self._apply_imbalance_method(
                X_train,
                y_train,
                method=str(policy["imbalance"]["method"]),
                random_state=random_state,
            )
            logger.info("Imbalance handling: %s", imbalance_meta.get("status"))
            metadata["imbalance"] = imbalance_meta
            logger.info("CHECK: imbalance handling completed")

            logger.info("Saving outputs to %s", output_folder)
            paths = self._save_outputs(
                output_folder=output_folder,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                metadata=metadata,
                policy=policy,
                evidence=evidence,
                dataset_name=dataset_path.name,
                target_col=target_col,
                test_size=test_size,
                random_state=random_state,
                use_llm=use_llm,
            )
            logger.info("[DONE] policy_used=%s | features=%d | output=%s",
                        bool(llm_policy), X_train.shape[1], output_folder)

            state.update(
                {
                    "X_train_path": str(paths["X_train"]),
                    "X_test_path": str(paths["X_test"]),
                    "y_train_path": str(paths["y_train"]),
                    "y_test_path": str(paths["y_test"]),
                    "summary_path": str(paths["summary"]),
                    "column_actions_path": str(paths["column_actions"]),
                    "column_actions_frontend_path": str(paths["column_actions_frontend"]),
                    "policy_path": str(paths["policy"]),
                    "evidence_path": str(paths["evidence"]),
                    "status": "success",
                    "error": None,
                    "output_folder": str(output_folder),
                }
            )
            return state

        except Exception as e:
            logger.error("FAILED: %s", str(e))
            state.update({"status": "failed", "error": str(e)})
            return state

    def _build_frontend_column_actions(self, column_actions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize column actions for frontend consumption as a stable list."""
        frontend_rows: List[Dict[str, Any]] = []

        for column_name in sorted(column_actions.keys()):
            details = column_actions.get(column_name, {})
            action = details.get("action")
            if not action:
                action = "drop" if details.get("reason") in {
                    "high_cardinality_drop", "constant_column_drop", "id_like_column_drop", "policy_drop"
                } else "transform"

            policy_source = details.get("policy_source", "default_policy")
            row: Dict[str, Any] = {
                "column": column_name,
                "action": action,
                "reason": details.get("reason", "policy"),
                "policy_source": policy_source,
            }

            # Keep details dynamic for frontend rendering while preserving a fixed top-level shape.
            dynamic_details = {k: v for k, v in details.items() if k not in {
                "action", "reason", "policy_source"}}
            if dynamic_details:
                row["details"] = dynamic_details

            frontend_rows.append(row)

        return frontend_rows

    def _validate_target_column(self, df: pd.DataFrame, target_col: str) -> None:
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found. Available: {', '.join(df.columns)}"
            )

    def _has_llm_credentials(self) -> bool:
        return bool(self.google_api_key or os.getenv("HF_TOKEN"))

    def _call_gemini(self, prompt: str) -> Optional[str]:
        self._last_gemini_error = ""
        if not self.google_api_key:
            self._last_gemini_error = "Gemini: no API key available"
            logger.warning(self._last_gemini_error)
            return None
        model = str(self.config.get("gemini_model", "gemini-2.5-flash"))
        logger.debug("Gemini: calling model=%s", model)
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key=" + self.google_api_key
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")[:200]
            self._last_gemini_error = f"Gemini HTTP error: {e.code} - {error_body}"
            logger.error(self._last_gemini_error)
            return None
        except (urllib.error.URLError, TimeoutError) as e:
            self._last_gemini_error = f"Gemini connection error: {e}"
            logger.error(self._last_gemini_error)
            return None

        try:
            parsed = json.loads(body)
            text = (
                parsed.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text")
            )
            logger.debug("Gemini: response received (%d chars)",
                         len(text) if text else 0)
            if not text:
                self._last_gemini_error = "Gemini response was empty"
            return text
        except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
            self._last_gemini_error = f"Gemini response parse error: {e}"
            logger.error(self._last_gemini_error)
            return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        print("[LLM] Trying Gemini 2.5 Flash...")
        try:
            text = self._call_gemini(prompt)
            if not text or not str(text).strip():
                raise ValueError(self._last_gemini_error or "Gemini returned an empty response")
            print("[LLM] Gemini succeeded.")
            return text
        except Exception as exc:
            print(f"[LLM] Gemini failed: {exc}")
            print("[LLM] Trying fallback LLM: Qwen2.5-7B...")

        try:
            text = call_qwen_fallback(prompt, temperature=0.0)
            if not text or not str(text).strip():
                raise ValueError("Qwen returned an empty response")
            print("[LLM] Qwen fallback succeeded.")
            return text
        except Exception as exc:
            print(f"[LLM] Qwen fallback failed: {exc}")
            print("[LLM] Returning safe default response.")
            return None

    def _parse_llm_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            logger.debug("LLM JSON parse: empty text")
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.warning(
                "LLM JSON parse: no JSON object found in response (first 200 chars: %s)", text[:200])
            return None
        try:
            parsed = json.loads(text[start: end + 1])
            logger.debug("LLM JSON parse: success, keys=%s",
                         list(parsed.keys()))
            return parsed
        except json.JSONDecodeError as e:
            logger.error("LLM JSON parse failed: %s | raw snippet: %s", str(
                e), text[start:start+200])
            return None

    def _build_evidence(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        y = df[target_col]
        class_counts = y.value_counts(dropna=False).to_dict()
        min_count = min(class_counts.values()) if class_counts else 0
        max_count = max(class_counts.values()) if class_counts else 0
        imbalance_ratio = float(
            max_count / max(min_count, 1)) if class_counts else 1.0

        columns: Dict[str, Dict[str, Any]] = {}
        for col in df.columns:
            if col == target_col:
                continue
            s = df[col]
            non_null = s.dropna()
            if pd.api.types.is_numeric_dtype(s):
                numeric_ratio = 1.0
                numeric_conv = pd.to_numeric(s, errors="coerce")
            elif non_null.empty:
                numeric_ratio = 0.0
                numeric_conv = pd.to_numeric(s, errors="coerce")
            else:
                numeric_conv = pd.to_numeric(non_null, errors="coerce")
                numeric_ratio = float(numeric_conv.notna().mean())

            datetime_ratio = 0.0
            if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                sample_text = " ".join(non_null.astype(
                    str).head(8).tolist()).lower()
                # Avoid expensive datetime parsing for clear non-date text columns.
                if re.search(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", sample_text):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        dt_conv = pd.to_datetime(
                            non_null, errors="coerce") if not non_null.empty else pd.Series(dtype="datetime64[ns]")
                    datetime_ratio = float(
                        dt_conv.notna().mean()) if not non_null.empty else 0.0

            top_values = s.astype(str).value_counts(dropna=False).head(10)
            columns[col] = {
                "dtype": str(s.dtype),
                "missing_ratio": float(s.isna().mean()),
                "unique_count": int(s.nunique(dropna=True)),
                "unique_ratio": float(s.nunique(dropna=True) / max(len(s), 1)),
                "numeric_parse_ratio": numeric_ratio,
                "datetime_parse_ratio": datetime_ratio,
                "sample_values": s.dropna().astype(str).head(8).tolist(),
                "top_values": {str(k): int(v) for k, v in top_values.items()},
                "numeric_stats": {
                    "mean": float(numeric_conv.mean()) if numeric_conv.notna().any() else None,
                    "std": float(numeric_conv.std()) if numeric_conv.notna().any() else None,
                    "q1": float(numeric_conv.quantile(0.25)) if numeric_conv.notna().any() else None,
                    "q3": float(numeric_conv.quantile(0.75)) if numeric_conv.notna().any() else None,
                },
            }
        n_classes = int(y.nunique(dropna=False))
        is_numeric = pd.api.types.is_numeric_dtype(y)

        if is_numeric and n_classes > 30:
            task_type = "regression"
        else:
            task_type = "classification"
        evidence_result = {
            "rows": int(len(df)),
            "columns": int(len(df.columns) - 1),
            "duplicate_rows": int(df.duplicated().sum()),
            "target": {
                "name": target_col,
                "n_classes": int(y.nunique(dropna=False)),
                "class_counts": {str(k): int(v) for k, v in class_counts.items()},
                "imbalance_ratio": imbalance_ratio,
                "task_type": task_type
            },
            "metric_priority": self.config["target_metric_priority"],
            "columns_profile": columns,
        }
        logger.debug("Evidence built: %d rows | %d feature cols | %d duplicates | "
                     "target classes=%s | imbalance=%.2f",
                     evidence_result["rows"], evidence_result["columns"],
                     evidence_result["duplicate_rows"],
                     list(class_counts.keys()), imbalance_ratio)
        return evidence_result

    def _default_policy(self, df: pd.DataFrame, target_col: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        columns_policy: Dict[str, Dict[str, Any]] = {}
        for col in df.columns:
            if col == target_col:
                continue
            profile = evidence["columns_profile"][col]
            is_numeric_dtype = pd.api.types.is_numeric_dtype(df[col])
            dtype_guess = "numeric" if is_numeric_dtype or profile[
                "numeric_parse_ratio"] > 0.85 else "categorical"
            if profile["datetime_parse_ratio"] > 0.85 and not is_numeric_dtype:
                dtype_guess = "datetime"
            # High-cardinality dropping: categorical/text OR numeric ID-like columns (almost unique)
            high_card = (
                (
                    dtype_guess in {"categorical", "text"}
                    and (
                        profile["unique_ratio"] > float(
                            self.config["high_cardinality_unique_ratio"])
                        or (
                            profile["unique_count"] > int(
                                self.config["high_cardinality_unique_count"])
                            and profile["unique_ratio"] > 0.9
                        )
                    )
                )
                or (
                    # Also catch numeric ID-like columns (almost 100% unique)
                    dtype_guess == "numeric"
                    and profile["unique_ratio"] > 0.98
                )
            )
            drop = profile["missing_ratio"] > 0.6 or (
                dtype_guess in {"categorical",
                                "text"} and profile["unique_ratio"] > 0.99
            ) or high_card
            encoding = "none" if dtype_guess == "numeric" else "label"
            if dtype_guess == "categorical":
                if profile["unique_count"] <= int(self.config["max_onehot_categories"]):
                    encoding = "onehot"
                else:
                    encoding = "label"
            columns_policy[col] = {
                "drop": bool(drop),
                "dtype": dtype_guess,
                "missing": "median" if dtype_guess == "numeric" else "mode",
                "outlier": "keep",
                "encoding": encoding,
                "reason": "high_cardinality_drop" if high_card else "default_policy",
            }

        imbalance_method = "none"
        if evidence["target"]["imbalance_ratio"] >= 3.0:
            imbalance_method = "class_weight"

        return {
            "duplicates": {"action": "drop_exact", "reason": "safe_default"},
            "columns": columns_policy,
            # Keep columns interpretable and avoid post-encoding drops.
            "feature_selection": {"method": "none", "threshold": 0.0},
            "feature_creation": {"method": "datetime_parts"},
            "dimensionality_reduction": {"method": "none", "n_components": "auto"},
            "scaling": {"method": "standard"},
            "normalization": {"method": "none"},
            "imbalance": {"method": imbalance_method, "reason": "default_by_ratio"},
        }

    def _llm_decide_policy(self, evidence: Dict[str, Any], target_col: str) -> Optional[Dict[str, Any]]:
        prompt = (
            "You are the final preprocessing policy authority. "
            "Return only JSON with keys: duplicates, columns, feature_selection, feature_creation, "
            "dimensionality_reduction, scaling, normalization, imbalance. "
            "Allowed methods only: duplicates.action=drop_exact|keep; "
            "columns.dtype=numeric|categorical|datetime|text; "
            "columns.missing=median|mean|mode|constant|indicator; "
            "columns.outlier=keep|clip|log_transform|drop_rows; "
            "columns.encoding=none|label|onehot|frequency; "
            "feature_selection.method=none|variance; "
            "feature_creation.method=none|datetime_parts; "
            "dimensionality_reduction.method=none; "
            "scaling.method=none|standard|minmax|robust|quantile|power; "
            "normalization.method=none|l1|l2|max; "
            "imbalance.method=none|class_weight|oversample|undersample. "
            "IMPORTANT RULES: onehot is allowed only when unique_count <= 3; "
            "if unique_count > 3 use label or frequency; "
            "very high-cardinality columns (including numeric ID-like columns) should be dropped before encoding; "
            "do not use pca. "
            "Do not remove more than 20 percent of features. "
            f"Target column is '{target_col}'. "
            f"Evidence: {json.dumps(evidence)}"
        )
        raw = self._call_llm(prompt)
        if not raw:
            return None
        parsed = self._parse_llm_json(raw)
        if not parsed:
            return None
        return parsed

    def _merge_and_validate_policy(
        self,
        default_policy: Dict[str, Any],
        llm_policy: Optional[Dict[str, Any]],
        evidence: Dict[str, Any],
        total_rows: int,
    ) -> Dict[str, Any]:
        policy = default_policy
        policy_source_tracking = {}  # Track which policy each decision came from

        # Track default_policy source for all columns
        if "columns" in default_policy:
            for col in default_policy["columns"]:
                policy_source_tracking[col] = "default_policy"

        if llm_policy and self.config["llm_final_decision"]:
            policy = {**default_policy, **llm_policy}
            if "columns" in llm_policy:
                merged_cols = default_policy["columns"].copy()
                if isinstance(llm_policy["columns"], dict):
                    # Update with llm_policy and track source
                    for col, llm_decision in llm_policy["columns"].items():
                        merged_cols[col] = llm_decision
                        policy_source_tracking[col] = "llm_policy"
                policy["columns"] = merged_cols

        # Store tracking info in policy for later use
        policy["_policy_source_tracking"] = policy_source_tracking

        allowed_scaling = {"none", "standard",
                           "minmax", "robust", "quantile", "power"}
        allowed_norm = {"none", "l1", "l2", "max"}
        allowed_imb = {"none", "class_weight", "oversample", "undersample"}
        allowed_enc = {"none", "label", "onehot", "frequency"}
        allowed_dtype = {"numeric", "categorical", "datetime", "text"}
        allowed_missing = {"median", "mean",
                           "mode", "constant", "indicator"}
        allowed_outlier = {"keep", "clip", "log_transform", "drop_rows"}

        if policy.get("scaling", {}).get("method") not in allowed_scaling:
            policy["scaling"] = {"method": "standard"}
        if policy.get("normalization", {}).get("method") not in allowed_norm:
            policy["normalization"] = {"method": "none"}
        if policy.get("imbalance", {}).get("method") not in allowed_imb:
            policy["imbalance"] = {"method": "none",
                                   "reason": "invalid_replaced"}
        # Force interpretable output column names.
        if policy.get("dimensionality_reduction", {}).get("method") == "pca":
            policy["dimensionality_reduction"] = {
                "method": "none", "reason": "pca_disabled_for_interpretability"}

        safe_notes: List[str] = []
        col_policy = policy.get("columns", {})
        dropped_count = 0
        for col, decisions in col_policy.items():
            if col not in default_policy.get("columns", {}):
                continue
            if decisions.get("dtype") not in allowed_dtype:
                decisions["dtype"] = default_policy["columns"][col]["dtype"]
            if decisions.get("encoding") not in allowed_enc:
                decisions["encoding"] = default_policy["columns"][col]["encoding"]
            if decisions.get("missing") not in allowed_missing:
                decisions["missing"] = default_policy["columns"][col]["missing"]
            if decisions.get("outlier") not in allowed_outlier:
                decisions["outlier"] = "keep"

            # Never allow one-hot when category count is above threshold.
            unique_count = evidence.get("columns_profile", {}).get(
                col, {}).get("unique_count", 0)
            if decisions.get("encoding") == "onehot" and unique_count > int(self.config["max_onehot_categories"]):
                decisions["encoding"] = "label"
                safe_notes.append(
                    f"{col}: onehot replaced with label (unique_count={unique_count} > {self.config['max_onehot_categories']})")

            # Convert any lingering "keep" missing strategy into explicit imputations.
            if decisions.get("missing") == "keep":
                decisions["missing"] = "median" if decisions.get(
                    "dtype") == "numeric" else "mode"
                safe_notes.append(
                    f"{col}: missing strategy 'keep' replaced with {decisions['missing']}")

            # Force drop for very high-cardinality columns: categorical/text OR numeric ID-like.
            col_profile = evidence.get("columns_profile", {}).get(col, {})
            col_dtype = decisions.get("dtype")
            is_high_card_categorical = (
                col_dtype in {"categorical", "text"}
                and (
                    col_profile.get("unique_ratio", 0.0) > float(
                        self.config["high_cardinality_unique_ratio"])
                    or (
                        col_profile.get("unique_count", 0) > int(
                            self.config["high_cardinality_unique_count"])
                        and col_profile.get("unique_ratio", 0.0) > 0.9
                    )
                )
            )
            is_high_card_numeric = (
                col_dtype == "numeric"
                and col_profile.get("unique_ratio", 0.0) > 0.98
            )

            if is_high_card_categorical or is_high_card_numeric:
                decisions["drop"] = True
                decisions["reason"] = "high_cardinality_drop"
                reason_msg = "non-numeric values" if is_high_card_categorical else "numeric ID-like column"
                safe_notes.append(
                    f"{col}: dropped as high-cardinality {reason_msg} before encoding")

            if bool(decisions.get("drop")):
                dropped_count += 1

            if self.config["safe_mode"] and decisions.get("outlier") not in {"keep", "clip", "log_transform", "drop_rows"}:
                decisions["outlier"] = "keep"
                safe_notes.append(f"{col}: unsafe outlier action replaced")

        total_features = max(len(col_policy), 1)
        protected_drop_reasons = {"high_cardinality_drop",
                                  "constant_column_drop", "id_like_column_drop"}
        optional_drop_cols = [
            col
            for col, decisions in col_policy.items()
            if bool(decisions.get("drop"))
            and decisions.get("reason") not in protected_drop_reasons
        ]
        if len(optional_drop_cols) / total_features > 0.35:
            safe_notes.append(
                "Too many columns dropped by policy; capping drops to protect dataset")
            kept = 0
            for col in sorted(optional_drop_cols):
                if col_policy[col].get("drop"):
                    if kept / total_features < 0.35:
                        kept += 1
                    else:
                        col_policy[col]["drop"] = False

        if policy.get("duplicates", {}).get("action") not in {"drop_exact", "keep"}:
            policy["duplicates"] = {
                "action": "drop_exact", "reason": "invalid_replaced"}

        policy["safeguards"] = {
            "safe_mode": bool(self.config["safe_mode"]),
            "notes": safe_notes,
            "max_row_drop_fraction": self.config["max_row_drop_fraction"],
            "rows": total_rows,
            "target_imbalance_ratio": evidence["target"]["imbalance_ratio"],
        }
        return policy

    def _preprocess_with_policy(
        self,
        df: pd.DataFrame,
        target_col: str,
        policy: Dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        steps_status = {
            "missing_values": "handled",
            "outliers": "handled",
            "duplicates": "handled",
            "type_conversion": "handled",
            "categorical_encoding": "handled",
            "feature_selection": "handled",
            "feature_creation": "handled",
            "dimensionality_reduction": "handled",
            "normalization": "handled",
            "scaling": "handled",
            "imbalance": "handled_after_split",
        }

        df_work = df.copy()
        duplicate_rows_removed = 0
        if policy["duplicates"]["action"] == "drop_exact":
            duplicate_rows_removed = int(df_work.duplicated().sum())
            df_work = df_work.drop_duplicates().copy()

        y = df_work[target_col].copy()
        feature_frames: List[pd.DataFrame] = []
        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        dropped_cols: List[str] = []
        column_actions: Dict[str, Any] = {}
        outlier_row_drop_mask = pd.Series(False, index=df_work.index)

        policy_source_tracking = policy.get("_policy_source_tracking", {})

        for col in [c for c in df_work.columns if c != target_col]:
            s = df_work[col]
            p = policy["columns"].get(col, {})
            if bool(p.get("drop", False)):
                dropped_cols.append(col)
                policy_source = policy_source_tracking.get(
                    col, "default_policy")
                drop_reason = p.get("reason", "policy_drop")
                column_actions[col] = {"action": "drop",
                                       "reason": drop_reason,
                                       "policy_source": policy_source}
                logger.debug("  col %-30s -> DROP (%s via %s)", col,
                             drop_reason, policy_source)
                continue

            dtype_choice = p.get("dtype", "categorical")
            missing_method = p.get("missing", "mode")
            outlier_method = p.get("outlier", "keep")
            encoding_method = p.get("encoding", "none")

            # Type conversion
            if dtype_choice == "numeric":
                converted = pd.to_numeric(s, errors="coerce")
            elif dtype_choice == "datetime":
                converted = pd.to_datetime(s, errors="coerce")
            else:
                converted = s.astype("string")

            # Missing values
            if dtype_choice == "numeric":
                if missing_method == "median":
                    fill_val = float(converted.median()
                                     ) if converted.notna().any() else 0.0
                    converted = converted.fillna(fill_val)
                elif missing_method == "mean":
                    fill_val = float(
                        converted.mean()) if converted.notna().any() else 0.0
                    converted = converted.fillna(fill_val)
                elif missing_method == "constant":
                    converted = converted.fillna(0.0)
                elif missing_method == "indicator":
                    indicator = converted.isna().astype(int)
                    feature_frames.append(pd.DataFrame(
                        {f"{col}__missing": indicator}, index=df_work.index))
                    fill_val = float(converted.median()
                                     ) if converted.notna().any() else 0.0
                    converted = converted.fillna(fill_val)
                else:
                    converted = converted.fillna(
                        float(converted.median()) if converted.notna().any() else 0.0)
            elif dtype_choice == "datetime":
                converted = converted.fillna(pd.Timestamp("1970-01-01"))
            else:
                if missing_method in {"mode", "keep"}:
                    mode_val = converted.mode(dropna=True)
                    fill_val = str(
                        mode_val.iloc[0]) if not mode_val.empty else "missing"
                    converted = converted.fillna(fill_val)
                elif missing_method == "constant":
                    converted = converted.fillna("missing")
                elif missing_method == "indicator":
                    indicator = converted.isna().astype(int)
                    feature_frames.append(pd.DataFrame(
                        {f"{col}__missing": indicator}, index=df_work.index))
                    converted = converted.fillna("missing")
                else:
                    converted = converted.fillna("missing")

            # Outliers (numeric only)
            if dtype_choice == "numeric":
                if outlier_method == "clip":
                    q = float(self.config["max_outlier_clip_quantile"])
                    lo, hi = converted.quantile(q), converted.quantile(1.0 - q)
                    converted = converted.clip(lower=lo, upper=hi)
                elif outlier_method == "log_transform":
                    converted = np.sign(converted) * \
                        np.log1p(np.abs(converted))
                elif outlier_method == "drop_rows":
                    q = float(self.config["max_outlier_clip_quantile"])
                    lo, hi = converted.quantile(q), converted.quantile(1.0 - q)
                    col_outlier_mask = (converted < lo) | (converted > hi)
                    outlier_row_drop_mask = outlier_row_drop_mask | col_outlier_mask.fillna(
                        False)
            
            # Feature creation
            if dtype_choice == "datetime":
                if policy.get("feature_creation", {}).get("method") == "datetime_parts":
                    converted = pd.to_datetime(s, errors="coerce", utc=True)
                    if pd.api.types.is_datetime64_any_dtype(converted):
                        feature_frames.append(
                            pd.DataFrame(
                                {
                                    f"{col}__year": converted.dt.year.astype(int),
                                    f"{col}__month": converted.dt.month.astype(int),
                                    f"{col}__day": converted.dt.day.astype(int),
                                },
                                index=df_work.index,
                            )
                        )
                        policy_source = policy_source_tracking.get(
                            col, "default_policy")
                        column_actions[col] = {
                            "type": "datetime",
                            "action": "datetime_parts",
                            "missing": missing_method,
                            "reason": p.get("reason", "policy"),
                            "policy_source": policy_source,
                        }
                    else:
                        logger.warning(f"Column {col} could not be converted to datetime — skipping datetime parts")
                    continue

                feature_frames.append(pd.DataFrame(
                    {col: converted.astype("int64")}, index=df_work.index))
                numeric_cols.append(col)
                policy_source = policy_source_tracking.get(
                    col, "default_policy")
                column_actions[col] = {
                    "type": "datetime", "action": "timestamp", "reason": p.get("reason", "policy"),
                    "policy_source": policy_source
                }
                continue

            if dtype_choice == "numeric":
                feature_frames.append(pd.DataFrame(
                    {col: converted}, index=df_work.index))
                numeric_cols.append(col)
                policy_source = policy_source_tracking.get(
                    col, "default_policy")
                column_actions[col] = {
                    "type": "numeric",
                    "missing": missing_method,
                    "outlier": outlier_method,
                    "encoding": "none",
                    "reason": p.get("reason", "policy"),
                    "policy_source": policy_source,
                }
                continue

            # Categorical / text encoding
            cleaned = converted.astype(str)
            if encoding_method == "hash":
                hasher = FeatureHasher(
                    n_features=int(self.config["hash_features"]), input_type="string"
                )
                hashed = hasher.transform([[val] for val in cleaned]).toarray()
                hash_cols = [f"{col}__hash_{i}" for i in range(
                    self.config["hash_features"])]
                feature_frames.append(pd.DataFrame(
                    hashed, columns=hash_cols, index=df_work.index))
            elif encoding_method == "onehot":
                feature_frames.append(pd.get_dummies(
                    cleaned, prefix=col, dtype=int))
            elif encoding_method == "frequency":
                freq = cleaned.value_counts(normalize=True)
                feature_frames.append(pd.DataFrame(
                    {f"{col}__freq": cleaned.map(freq)}, index=df_work.index))
            elif encoding_method == "label":
                categories = sorted(cleaned.unique().tolist())
                mapping = {cat: idx for idx, cat in enumerate(categories)}
                feature_frames.append(pd.DataFrame(
                    {col: cleaned.map(mapping).astype(int)}, index=df_work.index))
            else:
                feature_frames.append(pd.DataFrame(
                    {col: cleaned}, index=df_work.index))
                categorical_cols.append(col)

            policy_source = policy_source_tracking.get(col, "default_policy")
            column_actions[col] = {
                "type": "categorical",
                "missing": missing_method,
                "encoding": encoding_method,
                "reason": p.get("reason", "policy"),
                "policy_source": policy_source,
            }
            logger.debug("  col %-30s -> CATEGORICAL | missing=%-8s encoding=%s (via %s)",
                         col, missing_method, encoding_method, policy_source)

        if not feature_frames:
            logger.error(
                "No features left after applying preprocessing policy — check column drop rules")
            raise ValueError("No features left after preprocessing policy")

        X = pd.concat(feature_frames, axis=1)
        logger.debug("Feature matrix assembled: %d rows x %d cols",
                     X.shape[0], X.shape[1])

        # Optional row dropping for outliers, guarded by max_row_drop_fraction.
        rows_dropped_outliers = 0
        marked_outlier_rows = int(outlier_row_drop_mask.sum())
        if marked_outlier_rows > 0:
            max_fraction = float(self.config.get(
                "max_row_drop_fraction", 0.02))
            max_rows_to_drop = max(1, int(len(X) * max_fraction))
            outlier_indices = outlier_row_drop_mask[outlier_row_drop_mask].index.tolist(
            )
            if marked_outlier_rows > max_rows_to_drop:
                logger.warning(
                    "Outlier row drop capped by safeguard: marked=%d cap=%d",
                    marked_outlier_rows,
                    max_rows_to_drop,
                )
                outlier_indices = outlier_indices[:max_rows_to_drop]
                steps_status["outliers"] = "drop_rows_capped"
            else:
                steps_status["outliers"] = "drop_rows"
            X = X.drop(index=outlier_indices)
            y = y.drop(index=outlier_indices)
            rows_dropped_outliers = len(outlier_indices)

        # Feature selection
        fs_method = policy.get("feature_selection", {}).get("method", "none")
        if fs_method == "variance":
            constant_cols = [
                c for c in X.columns if X[c].nunique(dropna=False) <= 1]
            if constant_cols:
                logger.info("Feature selection: dropping %d constant column(s): %s", len(
                    constant_cols), constant_cols)
                X = X.drop(columns=constant_cols)
                logger.info("CHECK: post-encoding variance drop applied")
        elif fs_method == "none":
            steps_status["feature_selection"] = "skipped"
            logger.info("CHECK: post-encoding feature drops are disabled")

        # Scaling
        scaler = None
        scaling_method = policy.get("scaling", {}).get("method", "standard")
        numeric_X_cols = [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        logger.debug("Scaling: method=%s on %d numeric columns",
                     scaling_method, len(numeric_X_cols))
        if numeric_X_cols and scaling_method != "none":
            if scaling_method == "standard":
                scaler = StandardScaler()
            elif scaling_method == "minmax":
                scaler = MinMaxScaler()
            elif scaling_method == "robust":
                scaler = RobustScaler()
            elif scaling_method == "quantile":
                scaler = QuantileTransformer(
                    output_distribution="normal", random_state=42)
            elif scaling_method == "power":
                scaler = PowerTransformer()
            if scaler is not None:
                X[numeric_X_cols] = scaler.fit_transform(X[numeric_X_cols])
        else:
            steps_status["scaling"] = "skipped"

        # Normalization
        norm_method = policy.get("normalization", {}).get("method", "none")
        logger.debug("Normalization: method=%s", norm_method)
        if norm_method in {"l1", "l2", "max"} and not X.empty:
            norm = Normalizer(norm=norm_method)
            X = pd.DataFrame(norm.fit_transform(
                X), columns=X.columns, index=X.index)
        else:
            steps_status["normalization"] = "skipped"

        # Dimensionality reduction
        dr_method = policy.get("dimensionality_reduction",
                               {}).get("method", "none")
        logger.debug("Dimensionality reduction: method=%s", dr_method)
        if dr_method == "pca":
            if X.shape[1] > 2:
                n_components = min(
                    max(2, int(np.sqrt(X.shape[1]))), X.shape[1])
                pca = PCA(n_components=n_components, random_state=42)
                reduced = pca.fit_transform(X)
                X = pd.DataFrame(
                    reduced,
                    columns=[f"pca_{i+1}" for i in range(reduced.shape[1])],
                    index=X.index,
                )
            else:
                steps_status["dimensionality_reduction"] = "skipped_not_enough_features"
        else:
            steps_status["dimensionality_reduction"] = "skipped"

        metadata = {
            "column_actions": column_actions,
            "dropped_columns": dropped_cols,
            "rows_dropped_outliers": rows_dropped_outliers,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "duplicates_removed": duplicate_rows_removed,
            "steps_status": steps_status,
            "scaler": {
                "method": scaling_method,
                "columns": numeric_X_cols,
                "means": scaler.mean_.tolist() if hasattr(scaler, "mean_") else [],
                "scales": scaler.scale_.tolist() if hasattr(scaler, "scale_") else [],
            },
        }

        return X, y.loc[X.index], metadata

    def _apply_imbalance_method(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method: str,
        random_state: int,
    ) -> tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        counts = y_train.value_counts()
        if counts.empty:
            return X_train, y_train, {"method": "none", "status": "skipped_empty_target"}

        min_count = int(counts.min())
        max_count = int(counts.max())
        ratio = float(max_count / max(min_count, 1))
        logger.debug("Imbalance check: method=%s | ratio=%.2f | min_class=%d | max_class=%d",
                     method, ratio, min_count, max_count)
        meta = {
            "method": method,
            "before_counts": {str(k): int(v) for k, v in counts.items()},
            "imbalance_ratio": ratio,
            "status": "handled",
        }

        if method == "none" or ratio < 1.5:
            meta["status"] = "skipped_not_needed"
            return X_train, y_train, meta

        if method == "class_weight":
            class_weights = {str(c): float(max_count / max(int(v), 1))
                             for c, v in counts.items()}
            meta["class_weights"] = class_weights
            meta["status"] = "handled_class_weight_metadata_only"
            return X_train, y_train, meta

        if method not in {"oversample", "undersample"}:
            meta["status"] = "skipped_unknown_method"
            return X_train, y_train, meta

        combined = X_train.copy()
        combined["__target__"] = y_train.values
        groups = [g for _, g in combined.groupby("__target__")]

        if method == "oversample":
            target_n = max_count
            rebalanced = []
            for g in groups:
                if len(g) < target_n:
                    sampled = g.sample(n=target_n - len(g),
                                       replace=True, random_state=random_state)
                    g = pd.concat([g, sampled], axis=0)
                rebalanced.append(g)
            out = pd.concat(rebalanced, axis=0).sample(
                frac=1.0, random_state=random_state)
        else:
            # Conservative undersampling: do not keep less than 60% of original training rows.
            target_n = min_count
            projected_rows = target_n * len(groups)
            min_rows_allowed = int(0.6 * len(combined))
            if projected_rows < min_rows_allowed:
                meta["status"] = "skipped_undersample_too_aggressive"
                return X_train, y_train, meta
            rebalanced = []
            for g in groups:
                if len(g) > target_n:
                    g = g.sample(n=target_n, replace=False,
                                 random_state=random_state)
                rebalanced.append(g)
            out = pd.concat(rebalanced, axis=0).sample(
                frac=1.0, random_state=random_state)

        X_out = out.drop(columns=["__target__"])
        y_out = out["__target__"]
        meta["after_counts"] = {str(k): int(v)
                                for k, v in y_out.value_counts().items()}
        logger.info("Imbalance %s: before=%s  after=%s",
                    method, meta["before_counts"], meta["after_counts"])
        return X_out, y_out, meta

    def _save_outputs(
        self,
        output_folder: Path,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        metadata: Dict[str, Any],
        policy: Dict[str, Any],
        evidence: Dict[str, Any],
        dataset_name: str,
        target_col: str,
        test_size: float,
        random_state: int,
        use_llm: bool,
    ) -> Dict[str, Path]:
        X_train_path = output_folder / "X_train.csv"
        X_test_path = output_folder / "X_test.csv"
        y_train_path = output_folder / "y_train.csv"
        y_test_path = output_folder / "y_test.csv"
        summary_path = output_folder / "preprocessing_summary.json"
        column_actions_path = output_folder / "column_actions.json"
        column_actions_frontend_path = output_folder / "column_actions_frontend.json"
        policy_path = output_folder / "llm_policy.json"
        evidence_path = output_folder / "evidence_snapshot.json"

        X_train.to_csv(X_train_path, index=False)
        X_test.to_csv(X_test_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        y_test.to_csv(y_test_path, index=False)

        summary = {
            "dataset": dataset_name,
            "target_column": target_col,
            "rows": len(X_train) + len(X_test),
            "features": int(X_train.shape[1]),
            "dropped_columns": metadata["dropped_columns"],
            "duplicates_removed": metadata["duplicates_removed"],
            "test_size": test_size,
            "random_state": random_state,
            "llm": {
                "enabled": use_llm,
                "policy_used": bool(metadata.get("llm_policy_used", False)),
                "provider": "gemini",
                "model": self.config.get("gemini_model", "gemini-2.5-flash"),
                "final_decision_enabled": bool(self.config.get("llm_final_decision", True)),
            },
            "steps_status": metadata["steps_status"],
            "imbalance": metadata.get("imbalance", {}),
            "scaling": metadata["scaler"],
            "safeguards": policy.get("safeguards", {}),
            "task_type": evidence["target"]["task_type"],
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(column_actions_path, "w", encoding="utf-8") as f:
            json.dump(metadata["column_actions"], f, indent=2)
        with open(column_actions_frontend_path, "w", encoding="utf-8") as f:
            json.dump(self._build_frontend_column_actions(
                metadata["column_actions"]), f, indent=2)
        with open(policy_path, "w", encoding="utf-8") as f:
            json.dump(policy, f, indent=2)
        with open(evidence_path, "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2)

        return {
            "X_train": X_train_path,
            "X_test": X_test_path,
            "y_train": y_train_path,
            "y_test": y_test_path,
            "summary": summary_path,
            "column_actions": column_actions_path,
            "column_actions_frontend": column_actions_frontend_path,
            "policy": policy_path,
            "evidence": evidence_path,
        }


def preprocessing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    node = PreprocessingNode()
    return node.run(state)


if __name__ == "__main__":
    state = {
        "dataset_path":  RUN_CONFIG["dataset_path"],
        "target_column": RUN_CONFIG["target_column"],
        "use_llm":       RUN_CONFIG["use_llm"],
    }

    result = preprocessing_node(state)

    if result["status"] != "success":
        raise SystemExit(f"Preprocessing failed: {result['error']}")
