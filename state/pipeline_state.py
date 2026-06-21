"""
Single source of truth for the LangGraph StateGraph state.

Rules:
    - Every key that any node reads OR writes must be declared here.
    - Nodes return ONLY the keys they changed — LangGraph merges the
      returned dict into the existing state automatically.
    - No agent imports another agent. PipelineState is the only
      communication channel between all agents.
    - Optional fields default to None at graph invocation time.
      Always initialise with a full state dict (see EMPTY_STATE below).
"""

from typing import Optional, TypedDict


# ─────────────────────────────────────────────
# IntentFlags sub-dict (Agent 0 output)
# ─────────────────────────────────────────────

class IntentFlagsDict(TypedDict):
    """
    Serialised form of the Pydantic IntentFlags model.
    Stored directly in PipelineState after .model_dump().
    Controls which LangGraph nodes are activated for this run.
    """
    run_eda:                 bool
    run_preprocessing:       bool
    run_feature_engineering: bool
    run_model_selection:     bool
    run_training:            bool
    run_evaluation:          bool
    run_deployment:          bool
    target_column:           Optional[str]
    task_type:               str  # "classification"|"regression"|"clustering"|"unknown"


# ─────────────────────────────────────────────
# Full Pipeline State
# ─────────────────────────────────────────────

class PipelineState(TypedDict):

    # ── Inputs ────────────────────────────────────────────────────────────────
    data_path:  str        # absolute path to the uploaded dataset file
    nl_query:   str        # user's natural-language request

    # ── Intent (written by Agent 0 — intent_detector_node) ───────────────────
    intent_flags:  IntentFlagsDict
    target_column: Optional[str]
    task_type:     str

    # ── EDA outputs (written by Agent 1 — eda_node) ───────────────────────────
    analysis_report_path:  Optional[str]
    visualization_paths:   Optional[list]
    preprocessing_context: Optional[dict]
    automl_directives:     Optional[dict]

    # ── Preprocessing outputs (written by Agent 2 — preprocessing_node) ───────
    clean_data_path:       Optional[str]
    X_train_path:          Optional[str]
    X_test_path:           Optional[str]
    y_train_path:          Optional[str]
    y_test_path:           Optional[str]
    preprocessing_summary: Optional[dict]

    # ── Feature Engineering outputs (written by Agent 3) ─────────────────────
    X_train_engineered_path: Optional[str]
    X_test_engineered_path:  Optional[str]
    feature_report:          Optional[dict]

    # ── Model Selection outputs (written by Agent 4) ──────────────────────────
    automl_config:             Optional[dict]
    model_selection_reasoning: Optional[str]

    # ── Training outputs (written by Agent 5) ─────────────────────────────────
    trained_model_path: Optional[str]
    training_log:       Optional[dict]
    model_leaderboard:  Optional[str]

    # ── Evaluation outputs (written by Agent 6) ───────────────────────────────
    model_metrics:         Optional[dict]
    shap_plot_path:        Optional[str]
    diagnostic_report:     Optional[dict]
    confusion_matrix_path: Optional[str]
    roc_curve_path:        Optional[str]

    # ── Deployment outputs (written by Agent 7) ───────────────────────────────
    deployment_package_path: Optional[str]
    mlflow_run_id:           Optional[str]
    endpoint_url:            Optional[str]

    # ── HITL checkpoint fields ────────────────────────────────────────────────
    # Written by every *_checkpoint_node; read by the agent node on re-run.
    user_decision: Optional[str]   # "accept" | "feedback"
    feedback_text: Optional[str]   # free-text from the user

    # Full audit trail of all feedback given across the session
    feedback_history: list   # [{"agent": str, "feedback_text": str, "iteration": int}]

    # ── Shared error field ────────────────────────────────────────────────────
    error: Optional[str]

    # ── Per-agent UI output store ─────────────────────────────────────────────
    # Populated by each agent node so the frontend can render per-panel output.
    # Keys match agent names: "eda", "preprocessing", "feature_engineering", etc.
    agent_outputs: dict


# ─────────────────────────────────────────────
# Empty state factory
# ─────────────────────────────────────────────

def make_initial_state(data_path: str, nl_query: str) -> PipelineState:
    """
    Return a fully initialised PipelineState with all Optional fields
    set to None and list/dict fields set to empty containers.

    Always use this when invoking the graph to avoid KeyError on any
    node that reads an Optional field before it has been written.

    Usage:
        initial = make_initial_state("data/train.csv", "run full pipeline")
        result  = app.invoke(initial, config={"configurable": {"thread_id": "run-001"}})
    """
    return PipelineState(
        # Inputs
        data_path=data_path,
        nl_query=nl_query,

        # Intent
        intent_flags=IntentFlagsDict(
            run_eda=False,
            run_preprocessing=False,
            run_feature_engineering=False,
            run_model_selection=False,
            run_training=False,
            run_evaluation=False,
            run_deployment=False,
            target_column=None,
            task_type="unknown",
        ),
        target_column=None,
        task_type="unknown",

        # EDA
        analysis_report_path=None,
        visualization_paths=None,
        preprocessing_context=None,
        automl_directives=None,

        # Preprocessing
        clean_data_path=None,
        X_train_path=None,
        X_test_path=None,
        y_train_path=None,
        y_test_path=None,
        preprocessing_summary=None,

        # Feature Engineering
        X_train_engineered_path=None,
        X_test_engineered_path=None,
        feature_report=None,

        # Model Selection
        automl_config=None,
        model_selection_reasoning=None,

        # Training
        trained_model_path=None,
        training_log=None,
        model_leaderboard=None,

        # Evaluation
        model_metrics=None,
        shap_plot_path=None,
        diagnostic_report=None,
        confusion_matrix_path=None,
        roc_curve_path=None,

        # Deployment
        deployment_package_path=None,
        mlflow_run_id=None,
        endpoint_url=None,

        # HITL
        user_decision=None,
        feedback_text=None,
        feedback_history=[],

        # Shared
        error=None,
        agent_outputs={},
    )