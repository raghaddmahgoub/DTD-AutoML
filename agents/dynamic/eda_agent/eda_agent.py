"""
Agent 1: EDA Agent

Responsibility:
    Profile the dataset, compute statistics deterministically (tools/eda_tools.py),
    generate standard plots (tools/eda_plots.py), then make ONE LLM call that:
        - writes a human-readable narrative report on top of the computed stats
        - optionally requests a small number of EXTRA targeted plots
    Writes analysis_report_path, visualization_paths, preprocessing_context,
    and automl_directives into PipelineState.

    No iteration inside this node — one LLM call per run. Re-runs happen by
    looping the whole node again (via the HITL checkpoint's "feedback" path),
    with the user's feedback_text injected into the prompt on the next pass.

What this file exports for graph_builder.py:
    eda_node(state)            — the LangGraph node function (Agent 1 execution)
    route_after_eda(state)     — conditional edge: where to go after ACCEPT

Imports only from tools/ and state/ (per project rule: no agent imports another agent):
    tools.eda_tools     → load_dataframe(), compute_*(), build_*(), save_eda_report()
    tools.eda_plots     → generate_all_plots()
    tools.prompt_builder→ build_prompt_eda()
    tools.llm_client    → get_llm()
    tools.target_suggester → TargetSuggestionAgent (fallback only)
    state.pipeline_state→ PipelineState (type hint only)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt

from tools.eda import (
    load_dataframe,
    compute_dataset_summary,
    compute_column_profiles,
    compute_target_analysis,
    compute_data_quality,
    compute_relationships,
    compute_warnings,
    compute_signal_analysis,
    build_preprocessing_context,
    save_eda_report,
    generate_all_plots,
    generate_llm_requested_plot,
)
from tools.shared import build_prompt_eda, get_llm, TargetSuggestionAgent
from state.pipeline_state import PipelineState

logger = logging.getLogger(__name__)

_AGENT_NAME = "eda"
_OUTPUT_DIR_TEMPLATE = "../../../Output/eda/{run_id}"


# ─────────────────────────────────────────────
# Pydantic schema — LLM structured output
# ─────────────────────────────────────────────
# Mirrors the JSON schema described in _EDA_SYSTEM (tools/prompt_builder.py),
# extended with an optional `visualizations` list so the LLM can request a
# handful of targeted extra plots on top of the deterministic standard ones.

class ReportContentItem(BaseModel):
    type:  Literal["text", "bullet", "warning", "metric"]
    label: str
    value: str


class ReportSection(BaseModel):
    title:   str
    content: List[ReportContentItem]


class RequestedVisualization(BaseModel):
    """One extra plot the LLM wants beyond the standard deterministic set."""
    plot_type: Literal[
        "histogram", "boxplot", "scatterplot",
        "heatmap", "countplot", "missing_values",
    ]
    columns: List[str] = Field(default_factory=list)
    title:   str
    reason:  str = Field(description="Why this plot is useful")


class EDANarrativeReport(BaseModel):
    """
    Structured output bound to the LLM via .with_structured_output().
    `visualizations` is capped at 5 — the EDA agent should not flood the
    output directory with redundant plots.
    """
    title:           str
    summary:         str = Field(description="<= 80 words")
    sections:        List[ReportSection]
    recommendations: List[str] = Field(description="max 3 actionable strings")
    visualizations:  List[RequestedVisualization] = Field(default_factory=list)


# ─────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────

class EDAAgent:

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.6,
        google_api_key: Optional[str] = None,
    ):
        base_llm = get_llm(model_name=model_name, temperature=temperature, google_api_key=google_api_key)
        self.llm = base_llm.with_structured_output(EDANarrativeReport)
        self.suggester = TargetSuggestionAgent()

    def run(
        self,
        data_path: str,
        target_column: Optional[str],
        task_type: str,
        nl_query: str = "",
        feedback_context: str = "",
        run_id: str = "run",
    ) -> dict:
        """
        Execute Agent 1 and return a partial PipelineState dict.

        Returns dict with keys:
            analysis_report_path, visualization_paths,
            preprocessing_context, automl_directives,
            target_column, task_type, agent_outputs
        """
        logger.info(
            "[EDAAgent] data_path=%s target=%s task=%s",
            data_path, target_column, task_type,
        )

        output_dir = _OUTPUT_DIR_TEMPLATE.format(run_id=run_id)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Step 1 — load data via tools/eda_tools.py
        df = load_dataframe(data_path)

        # Step 2 — fallback target/task_type resolution (mirrors Agent 0's pattern)
        if not target_column or target_column not in df.columns:
            logger.info("[EDAAgent] target missing/invalid — running TargetSuggestionAgent")
            target_column = self.suggester.suggest(df)

        if task_type == "unknown" and target_column:
            task_type = self.suggester.suggest_task_type(df, target_column)
            logger.info("[EDAAgent] task_type unknown — inferred: %s", task_type)

        # Step 3 — deterministic computation (tools/eda_tools.py)
        dataset_summary  = compute_dataset_summary(df, target_column)
        column_profiles  = compute_column_profiles(df)
        target_analysis  = compute_target_analysis(df, target_column)
        data_quality     = compute_data_quality(df)
        relationships    = compute_relationships(df, target_column)
        warnings_list    = compute_warnings(dataset_summary, column_profiles, target_analysis)
        signal_analysis  = compute_signal_analysis(df, target_column, task_type)
        preprocessing_context = build_preprocessing_context(df, column_profiles, target_column)

        computed_stats = {
            "dataset_summary":  dataset_summary,
            "column_profiles":  column_profiles,
            "target_analysis":  target_analysis,
            "data_quality":     data_quality,
            "relationships":    relationships,
            "warnings":         warnings_list,
            "signal_analysis":  signal_analysis,
        }

        # Step 4 — standard deterministic plots (always generated)
        standard_plots = generate_all_plots(
            df=df,
            column_profiles=column_profiles,
            target_column=target_column,
            task_type=task_type,
            output_dir=output_dir,
        )

        # Step 5 — prompt via tools/prompt_builder.py
        # build_prompt_eda() returns PromptPair(system=..., user=...).
        # The raw computed_stats JSON is appended after prompts.user, per the
        # contract documented in prompt_builder.py ("eda_agent.py appends the
        # actual dataset_info JSON after prompts.user").
        prompts = build_prompt_eda(
            data_path        = data_path,
            run_type         = "raw",
            shape            = (dataset_summary["n_rows"], dataset_summary["n_columns"]),
            target_column    = target_column,
            task_type        = task_type,
            feedback_context = feedback_context,
            nl_query         = nl_query,
        )

        user_prompt = (
            f"{prompts.user}\n\n"
            f"Dataset statistics (JSON):\n"
            f"{json.dumps(computed_stats, indent=2, default=str)}\n\n"
            f"You may additionally request UP TO 5 extra targeted plots in "
            f"`visualizations` (only if they add real analytical value beyond "
            f"the standard plots already generated: missing values, correlation "
            f"heatmap, numeric/categorical distributions, target distribution)."
        )

        logger.info("[EDAAgent] Invoking LLM…")
        report: EDANarrativeReport = self.llm.invoke([
            SystemMessage(content=prompts.system),
            HumanMessage(content=user_prompt),
        ])
        logger.info("[EDAAgent] Report title=%r | %d sections | %d extra viz requested",
                     report.title, len(report.sections), len(report.visualizations))

        # Step 6 — LLM-requested extra plots (capped at 5 regardless of schema)
        llm_viz_dicts = [v.model_dump() for v in report.visualizations[:5]]
        llm_only_plots = []
        for idx, viz in enumerate(llm_viz_dicts):
            result = generate_llm_requested_plot(df, viz, output_dir, idx)
            if result:
                llm_only_plots.append(result)

        all_plots = standard_plots + llm_only_plots
        visualization_paths = [p["local_path"] for p in all_plots]

        # Step 7 — automl_directives: compact summary fed to Model Selection Agent
        automl_directives = {
            "n_rows":               dataset_summary["n_rows"],
            "n_columns":            dataset_summary["n_columns"],
            "target_column":        target_column,
            "task_type":            task_type,
            "class_distribution":   (target_analysis or {}).get("class_distribution"),
            "imbalance_severity":   (target_analysis or {}).get("imbalance_severity"),
            "high_cardinality_columns": [
                c for c, s in column_profiles.items()
                if s.get("is_high_cardinality")
            ],
            "warnings": warnings_list,
        }

        # Step 8 — persist full report JSON (tools/eda_tools.py)
        full_report = {
            "narrative":       report.model_dump(),
            "computed_stats":  computed_stats,
            "plots":           all_plots,
            "automl_directives": automl_directives,
        }
        analysis_report_path = save_eda_report(full_report, output_dir, filename="eda_report.json")

        # Step 9 — per-agent UI payload (consumed by the HITL checkpoint + frontend)
        agent_output = {
            "title":            report.title,
            "summary":          report.summary,
            "sections":         [s.model_dump() for s in report.sections],
            "recommendations":  report.recommendations,
            "warnings":         warnings_list,
            "visualization_paths": visualization_paths,
            "analysis_report_path": analysis_report_path,
        }

        logger.info("[EDAAgent] Done — %d plots, report → %s", len(all_plots), analysis_report_path)

        return {
            "analysis_report_path":  analysis_report_path,
            "visualization_paths":   visualization_paths,
            "preprocessing_context": {"columns": preprocessing_context, "warnings": warnings_list},
            "automl_directives":     automl_directives,
            "target_column":         target_column,
            "task_type":             task_type,
            "agent_outputs":         {_AGENT_NAME: agent_output},
        }


# ─────────────────────────────────────────────
# Feedback helper
# ─────────────────────────────────────────────

def _build_feedback_context(state: PipelineState) -> str:
    """
    Pull this agent's own feedback entries from state["feedback_history"]
    and join them into a single string for the prompt's {feedback_context}.
    Most-recent feedback last, so the LLM treats it as the latest instruction.
    """
    history = state.get("feedback_history", []) or []
    own = [h["feedback_text"] for h in history if h.get("agent") == _AGENT_NAME]
    return "\n".join(own)


# ─────────────────────────────────────────────
# LangGraph node function — Agent 1 execution
# ─────────────────────────────────────────────

def eda_node(state: PipelineState) -> dict:
    """
    LangGraph node for Agent 1 (EDA).

    Reads:   state["data_path"], state["target_column"], state["task_type"],
             state["nl_query"], state["feedback_history"]
    Returns: partial state dict — see EDAAgent.run().

    Note: agent_outputs from EDAAgent.run() is merged with any existing
    agent_outputs in state, since PipelineState["agent_outputs"] is a
    single shared dict across ALL agents (per pipeline_state.py).

    Note on run_id: PipelineState does not carry the LangGraph thread_id
    (it lives in the invoke config, not the state dict — see
    controller_agent.py), so output directories are namespaced by the
    dataset's filename stem instead. Good enough for one run per dataset;
    swap in a real run_id if/when it gets threaded into PipelineState.
    """
    agent = EDAAgent()

    result = agent.run(
        data_path        = state["data_path"],
        target_column    = state.get("target_column"),
        task_type         = state.get("task_type", "unknown"),
        nl_query          = state.get("nl_query", ""),
        feedback_context  = _build_feedback_context(state),
        run_id            = Path(state["data_path"]).stem,
    )

    # Merge into existing agent_outputs dict rather than overwrite it
    merged_outputs = dict(state.get("agent_outputs", {}))
    merged_outputs.update(result["agent_outputs"])
    result["agent_outputs"] = merged_outputs

    return result

# ─────────────────────────────────────────────
# Conditional edge function — after ACCEPT
# ─────────────────────────────────────────────
# graph_builder.py wraps this with _make_checkpoint_router("eda_agent", route_after_eda),
# which already handles the "feedback" → loop-back-to-eda_agent case.
# This function ONLY needs to decide where to go when the user ACCEPTS.

def route_after_eda(state: PipelineState) -> str:
    """
    Decide which node to visit after the EDA checkpoint is accepted.
    Mirrors route_after_intent()'s flag-checking pattern, starting
    one step further down the pipeline (preprocessing onward).
    """
    flags = state["intent_flags"]
    if flags["run_preprocessing"]:
        return "preprocessing_agent"
    if flags["run_feature_engineering"]:
        return "feature_engineering_agent"
    if flags["run_model_selection"]:
        return "model_selection_agent"
    if flags["run_training"]:
        return "training_agent"
    if flags["run_evaluation"]:
        return "evaluation_agent"
    # if flags["run_deployment"]:
    #     return "deployment_agent"
    return "pipeline_done"