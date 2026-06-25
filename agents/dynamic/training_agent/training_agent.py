"""Training Agent — LangGraph orchestrator nodes and routing."""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from state.pipeline_state import PipelineState
from tools.shared import get_llm
from tools.shared.training_common import align_features_for_model
from tools.training import train_simple, train_simple_optuna, train_autogluon
from graph.knowledge_graph import update_agent_progress

logger = logging.getLogger(__name__)


def _build_feedback_context(state: PipelineState, agent_name: str) -> str:
    """
    Pull this agent's own feedback entries from state["feedback_history"]
    and format them for the prompt.
    """
    history = state.get("feedback_history", []) or []
    own = [h["feedback_text"] for h in history if h.get("agent") == agent_name]
    if own:
        return f"\n\nUser Feedback History for {agent_name}:\n" + "\n".join(f"- {f}" for f in own)
    return ""


def _run_evaluation_subnode(state: dict) -> dict:
    """
    Post-training evaluation: generates confusion matrix, ROC curve,
    feature importance plot, and a diagnostic JSON report.
    Returns a dict of PipelineState updates (paths + diagnostic_report).
    All failures are caught individually so a broken plot never aborts the run.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    updates: dict = {}
    metrics = dict(state.get("model_metrics") or {})
    problem_type = state.get("task_type") or state.get("problem_type") or "classification"
    trained_model_path = state.get("trained_model_path")
    X_test_path = state.get("X_test_path")
    y_test_path = state.get("y_test_path")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = Path("output") / "dynamic_pipeline" / timestamp / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load model and test data
    model, X_test, y_test, preds = None, None, None, None
    is_autogluon = metrics.get("autogluon_used", False)
    try:
        if trained_model_path and Path(trained_model_path).exists():
            with open(trained_model_path, "rb") as f:
                model = pickle.load(f)
        if X_test_path and Path(X_test_path).exists():
            X_test = pd.read_csv(X_test_path)
        if y_test_path and Path(y_test_path).exists():
            y_test = pd.read_csv(y_test_path).iloc[:, 0]
        if model is not None and X_test is not None:
            if not is_autogluon:
                X_test = align_features_for_model(model, X_test)
            preds = model.predict(X_test)
    except Exception as exc:
        logger.warning("[TrainingAgent/Evaluation] Could not load model/data: %s", exc)

    # Confusion matrix
    if problem_type == "classification" and metrics.get("confusion_matrix"):
        try:
            from sklearn.metrics import ConfusionMatrixDisplay
            cm = np.array(metrics["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, colorbar=True)
            ax.set_title("Confusion Matrix")
            cm_path = str(eval_dir / "confusion_matrix.png")
            fig.savefig(cm_path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            updates["confusion_matrix_path"] = cm_path
        except Exception as exc:
            logger.warning("[TrainingAgent/Evaluation] Confusion matrix failed: %s", exc)

    # ROC curve (binary classification only)
    if (
        problem_type == "classification"
        and model is not None
        and X_test is not None
        and y_test is not None
        and not is_autogluon
        and hasattr(model, "predict_proba")
    ):
        try:
            from sklearn.metrics import roc_curve, auc
            classes = np.unique(np.asarray(y_test))
            if len(classes) == 2:
                y_score = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                ax.plot([0, 1], [0, 1], "k--")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend(loc="lower right")
                roc_path = str(eval_dir / "roc_curve.png")
                fig.savefig(roc_path, bbox_inches="tight", dpi=120)
                plt.close(fig)
                updates["roc_curve_path"] = roc_path
                metrics["roc_auc"] = round(roc_auc, 4)
        except Exception as exc:
            logger.warning("[TrainingAgent/Evaluation] ROC curve failed: %s", exc)

    # Feature importance
    if model is not None and X_test is not None and not is_autogluon:
        try:
            importances = None
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                coef = model.coef_
                importances = np.abs(coef[0] if coef.ndim > 1 else coef)
            if importances is not None:
                feat_names = np.array(list(X_test.columns))
                top_idx = np.argsort(importances)[-20:]
                fig, ax = plt.subplots(figsize=(8, max(4, len(top_idx) * 0.35)))
                ax.barh(feat_names[top_idx], importances[top_idx])
                ax.set_title("Feature Importance (Top 20)")
                ax.set_xlabel("Importance")
                fi_path = str(eval_dir / "feature_importance.png")
                fig.savefig(fi_path, bbox_inches="tight", dpi=120)
                plt.close(fig)
                updates["feature_importance_path"] = fi_path
        except Exception as exc:
            logger.warning("[TrainingAgent/Evaluation] Feature importance failed: %s", exc)

    # Diagnostic report JSON
    try:
        diagnostic = {
            "timestamp": timestamp,
            "problem_type": problem_type,
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, list, bool, type(None)))},
            "artifacts": {
                "confusion_matrix_path": updates.get("confusion_matrix_path"),
                "roc_curve_path": updates.get("roc_curve_path"),
                "feature_importance_path": updates.get("feature_importance_path"),
            },
        }
        diag_path = str(eval_dir / "diagnostic_report.json")
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diagnostic, f, indent=2, default=str)
        updates["diagnostic_report"] = diagnostic
    except Exception as exc:
        logger.warning("[TrainingAgent/Evaluation] Diagnostic report failed: %s", exc)

    if metrics:
        updates["model_metrics"] = metrics

    score = metrics.get("best_score")
    method = metrics.get("training_method", "")
    summary = f"Evaluated {method}"
    if score is not None:
        summary += f" — score: {score:.4f}"
    if updates.get("roc_curve_path"):
        summary += f", AUC: {metrics.get('roc_auc', '?')}"
    updates["_eval_summary"] = summary

    return updates


def training_node(state: PipelineState) -> dict:
    plan = state.get("training_plan") or {}
    train_tool_name = plan.get("train_tool")
    run_id = state.get("run_id")

    sub_nodes = [
        {"name": "Initialization", "description": f"Initializing training configurations for {train_tool_name}.", "status": "pending"},
        {"name": "Training", "description": f"Running model fit via {train_tool_name}.", "status": "pending"},
        {"name": "Save Artifact", "description": "Saving serialized model file.", "status": "pending"},
        {"name": "Evaluation", "description": "Evaluating trained model on held-out test set.", "status": "pending"},
    ]
    agent_output = {
        "status": "running",
        "sub_nodes": sub_nodes
    }

    # Initialize progress in DB
    update_agent_progress(run_id, "training", agent_output)

    def update_step(name: str, status: str, description: str = None):
        for node in sub_nodes:
            if node["name"] == name:
                node["status"] = status
                if description:
                    node["description"] = description
                break
        update_agent_progress(run_id, "training", agent_output)

    if not plan.get("approved"):
        logger.error("[TrainingAgent] Training plan not approved")
        for node in sub_nodes:
            node["status"] = "skipped"
        agent_output["status"] = "skipped"
        agent_output["sub_nodes"] = sub_nodes
        update_agent_progress(run_id, "training", agent_output)

        outputs = dict(state.get("agent_outputs", {}))
        outputs["training"] = agent_output

        return {
            **state,
            "error": "Training plan not approved",
            "step": "train_skipped",
            "agent_outputs": outputs
        }

    tool_map = {
        "train_simple": train_simple,
        "train_simple_optuna": train_simple_optuna,
        "train_autogluon": train_autogluon,
    }

    tool = tool_map.get(train_tool_name)
    if tool is None:
        logger.error("[TrainingAgent] Unknown train tool: %s", train_tool_name)
        sub_nodes[0]["status"] = "failed"
        sub_nodes[0]["description"] = f"Unknown train tool: {train_tool_name}"
        for node in sub_nodes[1:]:
            node["status"] = "skipped"
        agent_output["status"] = "failed"
        agent_output["sub_nodes"] = sub_nodes
        update_agent_progress(run_id, "training", agent_output)

        outputs = dict(state.get("agent_outputs", {}))
        outputs["training"] = agent_output

        return {
            **state,
            "error": f"Unknown train tool: {train_tool_name}",
            "step": "error",
            "agent_outputs": outputs
        }

    llm = get_llm()
    tool_input = {}
    if train_tool_name == "train_simple_optuna":
        plan_optuna = plan.get("optuna_config") or {}
        tool_input["optuna_trials"] = plan_optuna.get("n_trials", 30)

    task = f"Train with {train_tool_name}"
    feedback = _build_feedback_context(state, "training")
    prompt = state.get("nl_query", state.get("prompt", "")) + feedback

    update_step("Initialization", "completed", f"Selected tool '{train_tool_name}' and initialized model parameters.")
    update_step("Training", "running")
    update_step("Save Artifact", "running")

    result, updated_state = tool.invoke({
        "task": task,
        "tool_input": tool_input,
        "prompt": prompt,
        "data_path": state.get("data_path", ""),
        "llm": llm,
        "state": state,
    })

    # Sync fields to PipelineState compatibility
    pickle_path = updated_state.get("saved_files", {}).get("pickle")
    if pickle_path:
        updated_state["trained_model_path"] = pickle_path

    # Set UI agent output
    error_val = None
    status_val = result.get("status", "success")
    if status_val == "error":
        error_val = result.get("error", f"{train_tool_name} failed")

    best_model = result.get("best_model", "Unknown Model")
    best_score = result.get("best_score")
    score_desc = f" (score: {best_score:.4f})" if isinstance(best_score, float) else (f" (score: {best_score})" if best_score is not None else "")
    pickle_file = (result.get("saved_files") or {}).get("pickle", "")

    if status_val == "success":
        update_step("Training", "completed", f"Successfully trained best model: {best_model}{score_desc}.")
        update_step("Save Artifact", "completed", f"Saved model to: {pickle_file}." if pickle_file else "Saved model artifact.")
    else:
        update_step("Training", "failed", f"Failed: {error_val}")
        update_step("Save Artifact", "skipped")
        update_step("Evaluation", "skipped")

    # ── Evaluation subnode ────────────────────────────────────────────────────
    if status_val == "success":
        update_step("Evaluation", "running")
        eval_updates = _run_evaluation_subnode(updated_state)
        eval_summary = eval_updates.pop("_eval_summary", "Evaluation complete.")
        updated_state.update(eval_updates)
        update_step("Evaluation", "completed", eval_summary)
    # ─────────────────────────────────────────────────────────────────────────

    agent_output = {
        "status": status_val,
        "training_method": result.get("training_method"),
        "best_model": best_model,
        "best_score": best_score,
        "saved_files": result.get("saved_files"),
        "confusion_matrix_path": updated_state.get("confusion_matrix_path"),
        "roc_curve_path": updated_state.get("roc_curve_path"),
        "feature_importance_path": updated_state.get("feature_importance_path"),
        "diagnostic_report": updated_state.get("diagnostic_report"),
        "error": error_val,
        "sub_nodes": sub_nodes,
    }
    update_agent_progress(run_id, "training", agent_output)

    merged_outputs = dict(updated_state.get("agent_outputs", {}))
    merged_outputs["training"] = agent_output
    updated_state["agent_outputs"] = merged_outputs

    if error_val:
        updated_state["error"] = error_val

    return updated_state

def route_after_training(state: PipelineState) -> str:
    flags = state["intent_flags"]
    if flags.get("deployment"):
        return "deployment_agent"
    return "pipeline_done"
