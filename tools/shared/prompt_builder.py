"""
tools/prompt_builder.py
D.T.D (Data To Deployment) — Multi-Agent AutoML Pipeline

Tool: Prompt Builder
Responsibility:
    Central store of ALL agent prompts — system and user — for every agent.

    Every build_prompt_*() function returns a PromptPair namedtuple:
        .system  — the SYSTEM prompt  (role, rules, output format)
        .user    — the USER prompt    (runtime data: query, schema, state values)

    The agent then calls:
        self.llm.invoke([
            SystemMessage(content=prompts.system),
            HumanMessage(content=prompts.user),
        ])

    Why split?
        - System prompt  = stable instructions that rarely change
        - User prompt    = dynamic data injected at runtime (query, schema, paths)
        - Keeping them separate gives the LLM proper role context and
          makes each independently editable and testable

    Current coverage: Agent 0 (Intent Detector) — fully implemented.
    Stubs for Agents 1-7 included; fill each when that agent is built.
"""

from typing import NamedTuple


# ─────────────────────────────────────────────
# Return type for all build_prompt_* functions
# ─────────────────────────────────────────────

class PromptPair(NamedTuple):
    """
    Holds the system and user prompt for one LLM call.

    Usage in an agent:
        from langchain_core.messages import SystemMessage, HumanMessage

        prompts = build_prompt_intent_detector(...)
        response = self.llm.invoke([
            SystemMessage(content=prompts.system),
            HumanMessage(content=prompts.user),
        ])
    """
    system: str
    user:   str


# ═════════════════════════════════════════════════════════════════════════════
# Agent 0 — Intent Detector & Router
# ═════════════════════════════════════════════════════════════════════════════

# ── System prompt ─────────────────────────────────────────────────────────────
# Tells the LLM WHO it is, WHAT rules to follow, and WHAT format to output.
# Does NOT contain any runtime data (no paths, no column names, no user text).

_INTENT_DETECTOR_SYSTEM = """\
You are the Intent Detector for the D.T.D AutoML pipeline.

Your job is to read the user's request and the dataset schema, then decide
which pipeline steps to activate and what the target column and task type are.

Available pipeline steps:
  1. EDA                — dataset profiling, statistics, visualizations
  2. Preprocessing      — missing values, encoding, scaling, class imbalance
  3. Feature Engineering — feature synthesis, selection, dimensionality reduction
  4. Model Selection    — choose training backend and model families
  5. Model Training     — AutoGluon / XGBoost / sklearn + Optuna HPO
  7. Deployment         — FastAPI endpoint + Dockerfile

Routing rules:
  - "just preprocess" / "only clean"           → only preprocessing = true
  - "train a model" / "build a model"          → preprocessing, model_selection, training = true
  - "full pipeline" / "end to end" / "all"     → all flags = true
  - "analyse" / "analysis" / "explore"         → eda = true
  - "deploy" / "serve" / "api"                 → deployment = true (+ training chain if no model yet)
  - If target column is explicitly named        → use it
  - If target column is not named              → infer from schema or set null
  - If task type is ambiguous                  → set "unknown"

Output rules:
  - Return ONLY a valid JSON object — no explanation, no markdown, no code fences
  - Every field in the schema below is required
  - Boolean values must be true or false (lowercase), not strings

Output schema:
{
  "eda":                true | false,
  "preprocessing":      true | false,
  "feature_engineering":true | false,
  "model_selection":    true | false,
  "training":           true | false,
  "deployment":         true | false,
  "target_column":          "column_name" | null,
  "task_type":              "classification" | "regression" | "clustering" | "unknown"
}
"""

# ── User prompt template ──────────────────────────────────────────────────────
# Contains ONLY runtime data injected at call time.
# The {placeholders} are filled by build_prompt_intent_detector().

_INTENT_DETECTOR_USER = """\
User request:
  {nl_query}

Dataset schema:
  File       : {data_path}
  Shape      : {n_rows} rows × {n_cols} columns
  Columns    : {columns}
  Data types : {dtypes}

Decide which pipeline steps to activate and return the JSON object.
"""


def build_prompt_intent_detector(
    nl_query:  str,
    data_path: str,
    columns:   list,
    dtypes:    dict,
    shape:     tuple,
) -> PromptPair:
    """
    Build the system + user prompt pair for Agent 0 (Intent Detector).

    Args:
        nl_query:  User's natural-language request.
        data_path: Path to the dataset file (shown to LLM for context).
        columns:   List of column name strings from schema extraction.
        dtypes:    {"col": "dtype"} dict from schema extraction.
        shape:     (n_rows, n_cols) tuple from schema extraction.

    Returns:
        PromptPair(system=..., user=...)

    Example:
        schema  = extract_schema("data/titanic.csv")
        prompts = build_prompt_intent_detector(
            nl_query  = "i want to make analysis on this data",
            data_path = "data/titanic.csv",
            columns   = schema["columns"],
            dtypes    = schema["dtypes"],
            shape     = schema["shape"],
        )
        response = llm.invoke([
            SystemMessage(content=prompts.system),
            HumanMessage(content=prompts.user),
        ])
    """
    user = _INTENT_DETECTOR_USER.format(
        nl_query  = nl_query,
        data_path = data_path,
        n_rows    = shape[0],
        n_cols    = shape[1],
        columns   = ", ".join(columns),
        dtypes    = ", ".join(f"{c}: {t}" for c, t in dtypes.items()),
    )
    return PromptPair(system=_INTENT_DETECTOR_SYSTEM, user=user)


# ═════════════════════════════════════════════════════════════════════════════
# Agent 1 — EDA Agent
# ═════════════════════════════════════════════════════════════════════════════
# The EDA Agent uses EDAAgent.run() for all structured computation.
# The LLM's only job is to write a human-readable narrative report
# ON TOP of the already-computed numbers.

_EDA_SYSTEM = """You are a senior data scientist writing a human-readable narrative report
on top of structured EDA statistics already computed by code.

Produce a JSON object with this exact schema — no other text:
{
  "title": "<short report title>",
  "summary": "<= 80 words summarising the dataset and task>",
  "sections": [
    {
      "title": "<section name>",
      "content": [
        {"type": "text|bullet|warning|metric", "label": "<label>", "value": "<value>"}
      ]
    }
  ],
  "recommendations": ["<max 3 actionable strings>"]
}

Rules:
  - Return ONLY valid JSON — no markdown, no code fences, no preamble
  - Tailor summary and recommendations to the user task and any detected issues
  - Flag class imbalance, high missingness, outliers, multicollinearity if present
  - Every recommendation must be concrete and actionable
"""

_EDA_USER = """User task    : {nl_query}
Dataset path : {data_path}
Run type     : {run_type}
Shape        : {n_rows} rows x {n_cols} columns
Target column: {target_column}
Task type    : {task_type}

Feedback from user (if any): {feedback_context}

Dataset statistics (JSON) are appended below. Write the narrative JSON report.
"""


def build_prompt_eda(
    data_path:        str,
    run_type:         str,
    shape:            tuple,
    target_column:    str,
    task_type:        str,
    feedback_context: str = "",
    nl_query:         str = "",
) -> PromptPair:
    """
    Build the system + user prompt pair for Agent 1 (EDA narrative).

    Note: eda_agent.py appends the actual dataset_info JSON after
    prompts.user before sending to the LLM, so the numbers are always
    present without being hardcoded here.
    """
    user = _EDA_USER.format(
        nl_query         = nl_query or "explore the dataset",
        data_path        = data_path,
        run_type         = run_type,
        n_rows           = shape[0],
        n_cols           = shape[1],
        target_column    = target_column or "unknown (please infer)",
        task_type        = task_type,
        feedback_context = feedback_context or "none",
    )
    return PromptPair(system=_EDA_SYSTEM, user=user)


# ═════════════════════════════════════════════════════════════════════════════
# Agent 2 — Preprocessing Agent
# ═════════════════════════════════════════════════════════════════════════════

_PREPROCESSING_SYSTEM = """\
You are a data preprocessing expert.

For EACH column in the dataset, decide:
  - Missing values  : drop_row | mean | median | mode | knn_impute | constant
  - Outliers        : clip_iqr | remove | keep
  - Encoding        : one_hot (≤15 unique) | ordinal (>15 unique) | target_encode
  - Scaling         : standard | minmax | robust | none
  - Imbalance (target only): none | class_weight | smote | oversample | undersample

Hard constraints you must never violate:
  - Never one-hot encode columns with more than 15 unique values
  - Never drop more than 30% of rows
  - Never apply PCA here (that belongs to Feature Engineering)

Output a JSON policy object with one entry per column, then apply all
transforms and save: X_train.csv, X_test.csv, y_train.csv, y_test.csv,
preprocessing_summary.json, column_actions_frontend.json.
"""

_PREPROCESSING_USER = """\
Dataset path         : {data_path}
Target column        : {target_column}
Task type            : {task_type}
Test size            : {test_size}
EDA preprocessing context (use as guidance):
{preprocessing_context}

Feedback from user (if any): {feedback_context}

Build the per-column policy and apply all transforms.
"""


def build_prompt_preprocessing(
    data_path:             str,
    target_column:         str,
    task_type:             str,
    preprocessing_context: dict,
    test_size:             float = 0.2,
    feedback_context:      str   = "",
) -> PromptPair:
    import json
    user = _PREPROCESSING_USER.format(
        data_path             = data_path,
        target_column         = target_column,
        task_type             = task_type,
        test_size             = test_size,
        preprocessing_context = json.dumps(preprocessing_context, separators=(',', ':')) if preprocessing_context else "not available",
        feedback_context      = feedback_context or "none",
    )
    return PromptPair(system=_PREPROCESSING_SYSTEM, user=user)


# ═════════════════════════════════════════════════════════════════════════════
# Agent 3 — Feature Engineering Agent
# ═════════════════════════════════════════════════════════════════════════════

_FEATURE_ENGINEERING_SYSTEM = """\
You are a feature engineering specialist.

Steps to perform:
  1. Feature synthesis  : ratio features (A/B), interaction terms (A*B),
                          polynomial features (degree=2) for top-correlated pairs
  2. Feature selection  : remove zero-variance columns, remove pairwise corr > 0.95
  3. Importance pruning : if >50 features, keep top-N by RandomForest importance
  4. Time-series        : for datetime columns extract year, month, day, hour,
                          day_of_week, lag-1, rolling_mean_7

For each transformation explain WHY it was applied.
Save: X_train_engineered.csv, X_test_engineered.csv, feature_report.json
"""

_FEATURE_ENGINEERING_USER = """\
Train data   : {X_train_path}
Test data    : {X_test_path}
Target column: {target_column}
Task type    : {task_type}
Preprocessing summary:
{preprocessing_summary}

Feedback from user (if any): {feedback_context}

Apply feature engineering and save the enriched datasets.
"""


def build_prompt_feature_engineering(
    X_train_path:          str,
    X_test_path:           str,
    target_column:         str,
    task_type:             str,
    preprocessing_summary: dict,
    feedback_context:      str = "",
) -> PromptPair:
    import json
    user = _FEATURE_ENGINEERING_USER.format(
        X_train_path          = X_train_path,
        X_test_path           = X_test_path,
        target_column         = target_column,
        task_type             = task_type,
        preprocessing_summary = json.dumps(preprocessing_summary, separators=(',', ':')) if preprocessing_summary else "not available",
        feedback_context      = feedback_context or "none",
    )
    return PromptPair(system=_FEATURE_ENGINEERING_SYSTEM, user=user)


# ═════════════════════════════════════════════════════════════════════════════
# Agent 4 — Model Selection Agent
# ═════════════════════════════════════════════════════════════════════════════

_MODEL_SELECTION_SYSTEM = """\
You are an AutoML model selection expert.

Decision logic:
  - rows > 700,000                            → use dask_xgboost
  - rows > 10,000 and task is classif/regress → use autogluon
  - otherwise                                 → use sklearn + Optuna HPO

Also consider:
  - High cardinality categoricals → prefer tree-based models
  - Severe class imbalance        → prefer models with class_weight support

Output ONLY this JSON — no explanation, no markdown:
{
  "use_automl":           true | false,
  "automl_backend":       "autogluon" | "dask_xgboost" | "sklearn",
  "candidate_models":     ["ModelName", ...],
  "reasoning":            "one sentence",
  "time_budget_seconds":  300,
  "hyperparameter_search": true | false
}
"""

_MODEL_SELECTION_USER = """\
Dataset statistics:
  Rows             : {n_rows}
  Features         : {n_features}
  Task type        : {task_type}
  Class distribution: {class_distribution}
  Feature types    : {feature_types}

AutoML directives from EDA:
{automl_directives}

Feedback from user (if any): {feedback_context}

Choose the best training backend and return the JSON config.
"""


def build_prompt_model_selection(
    n_rows:             int,
    n_features:         int,
    task_type:          str,
    class_distribution: dict,
    feature_types:      dict,
    automl_directives:  dict,
    feedback_context:   str = "",
) -> PromptPair:
    import json
    user = _MODEL_SELECTION_USER.format(
        n_rows             = n_rows,
        n_features         = n_features,
        task_type          = task_type,
        class_distribution = json.dumps(class_distribution, separators=(',', ':')) if class_distribution else "{}",
        feature_types      = json.dumps(feature_types, separators=(',', ':'))      if feature_types      else "{}",
        automl_directives  = json.dumps(automl_directives, separators=(',', ':')) if automl_directives else "not available",
        feedback_context   = feedback_context or "none",
    )
    return PromptPair(system=_MODEL_SELECTION_SYSTEM, user=user)


# ═════════════════════════════════════════════════════════════════════════════
# Agent 5 — Training Agent
# ═════════════════════════════════════════════════════════════════════════════

_TRAINING_SYSTEM = """\
You are a model training specialist.

Execution rules:
  - backend == "autogluon"    → TabularPredictor.fit(train_data, label=target, time_limit=budget)
  - backend == "dask_xgboost" → load via Dask, run distributed XGBoost
  - backend == "sklearn"      → Optuna study (30 trials, TPE sampler, MedianPruner),
                                refit best params on full train set

Log all trials with MLflow: log_params(), log_metrics(), log_model().
Save: best_model_{timestamp}.pkl, training_log.json, model_leaderboard.csv (AutoGluon only).
"""

_TRAINING_USER = """\
Training config  : {automl_config}
Train features   : {X_train_path}
Train labels     : {y_train_path}
Test features    : {X_test_path}
Test labels      : {y_test_path}
Task type        : {task_type}
Time budget      : {time_budget_seconds}s

Feedback from user (if any): {feedback_context}

Execute training and save the best model.
"""


def build_prompt_training(
    automl_config:        dict,
    X_train_path:         str,
    y_train_path:         str,
    X_test_path:          str,
    y_test_path:          str,
    task_type:            str,
    time_budget_seconds:  int = 300,
    feedback_context:     str = "",
) -> PromptPair:
    import json
    user = _TRAINING_USER.format(
        automl_config        = json.dumps(automl_config, separators=(',', ':')) if automl_config else "{}",
        X_train_path         = X_train_path,
        y_train_path         = y_train_path,
        X_test_path          = X_test_path,
        y_test_path          = y_test_path,
        task_type            = task_type,
        time_budget_seconds  = time_budget_seconds,
        feedback_context     = feedback_context or "none",
    )
    return PromptPair(system=_TRAINING_SYSTEM, user=user)

# ═════════════════════════════════════════════════════════════════════════════
# Agent 7 — Deployment Agent
# ═════════════════════════════════════════════════════════════════════════════

_DEPLOYMENT_SYSTEM = """\
You are an MLOps deployment engineer.

Create a production-ready deployment package at output/deployment/{run_id}/:
  1. Serialize model   : joblib.dump(model, "model_{version}.pkl")
  2. Generate FastAPI app (api_server.py):
       POST /predict  — accepts JSON matching feature_schema,
                        returns {prediction, confidence, model_version}
       GET  /health   — returns {model_version, task_type, metrics, status}
       Input validated by auto-generated Pydantic model from feature_schema
  3. Generate Dockerfile:
       FROM python:3.11-slim
       COPY . /app
       RUN pip install -r requirements.txt
       CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
  4. Register in MLflow Model Registry → transition to "Staging"
"""

_DEPLOYMENT_USER = """\
Trained model  : {trained_model_path}
Task type      : {task_type}
Feature schema : {feature_schema}
Model metrics  : {model_metrics}

Feedback from user (if any): {feedback_context}

Generate the full deployment package.
"""

def build_prompt_deployment(
    trained_model_path: str,
    task_type:          str,
    feature_schema:     dict,
    model_metrics:      dict,
    feedback_context:   str = "",
) -> PromptPair:
    import json
    user = _DEPLOYMENT_USER.format(
        trained_model_path = trained_model_path,
        task_type          = task_type,
        feature_schema     = json.dumps(feature_schema, separators=(',', ':')) if feature_schema else "{}",
        model_metrics      = json.dumps(model_metrics,  separators=(',', ':')) if model_metrics  else "{}",
        feedback_context   = feedback_context or "none",
    )
    return PromptPair(system=_DEPLOYMENT_SYSTEM, user=user)