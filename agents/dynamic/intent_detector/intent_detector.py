"""
Agent 0: Intent Detector & Router

Responsibility:
    Parse the user's NL query and dataset schema in a SINGLE LLM call.
    Emit IntentFlags booleans written into PipelineState.
    No iteration, no interrupt() — runs to completion in one pass.

What this file exports for graph_builder.py:
    intent_detector_node(state)     — the LangGraph node function

    route_after_intent(state) -> str — conditional edge function:
        always returns "eda_agent" or "preprocessing_gate" etc.
        (intent detector has no checkpoint — it routes immediately)

Imports only from tools/ and state/:
    tools.schema_extractor  → extract_schema()
    tools.target_suggester  → TargetSuggestionAgent
    tools.prompt_builder    → build_prompt_intent_detector()
    tools.llm_client        → get_llm()
    state.pipeline_state    → PipelineState (type hint only)
"""

import logging
from typing import Optional, Literal
from graph.knowledge_graph import store_initial_knowledge_graph
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from tools.shared import (
    extract_schema,
    TargetSuggestionAgent,
    build_prompt_intent_detector,
    get_llm,
)
from state.pipeline_state   import PipelineState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Pydantic schema — LLM structured output
# ─────────────────────────────────────────────

class IntentFlags(BaseModel):
    """
    Structured output bound to the LLM via .with_structured_output().
    Parsed directly from the LLM response — no manual JSON handling.
    Serialised to dict via .model_dump() before writing into PipelineState.
    """
    eda:                 bool  = Field(description="Run EDA agent")
    preprocessing:       bool  = Field(description="Run Preprocessing agent")
    feature_engineering: bool  = Field(description="Run Feature Engineering agent")
    model_selection:     bool  = Field(description="Run Model Selection agent")
    training:            bool  = Field(description="Run Training agent")
    evaluation:          bool  = Field(description="Run Evaluation agent")
    # deployment:          bool  = Field(description="Run Deployment agent")
    target_column: Optional[str]  = Field(default=None)
    task_type: Literal["classification", "regression", "clustering", "unknown"]


# ─────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────

class IntentDetectorAgent:

    def __init__(
        self,
        model_name: str = "gemma-4-31b-it",
        temperature: float = 0.0,
        google_api_key: Optional[str] = None,
    ):
        base_llm   = get_llm(model_name=model_name, temperature=temperature, google_api_key=google_api_key)
        self.llm   = base_llm.with_structured_output(IntentFlags)
        self.suggester = TargetSuggestionAgent()

    def run(self, data_path: str, nl_query: str, run_id: Optional[str] = None) -> dict:
        """
        Execute Agent 0 and return a partial PipelineState dict.

        Returns dict with keys:
            intent_flags, target_column, task_type, knowledge_graph
        """
        logger.info("[IntentDetector] query=%r | file=%s", nl_query, data_path)

        # Step 1 — schema via tools/schema_extractor.py
        schema = extract_schema(data_path)

        # Step 2 — prompt via tools/prompt_builder.py
        # Returns PromptPair(system=..., user=...) — two separate prompts
        prompts = build_prompt_intent_detector(
            nl_query  = nl_query,
            data_path = data_path,
            columns   = schema["columns"],
            dtypes    = schema["dtypes"],
            shape     = schema["shape"],
        )

        logger.info("[IntentDetector] System prompt:%s", prompts.system)
        logger.info("[IntentDetector] User prompt:%s", prompts.user)

        # Step 3 — single LLM call with proper SystemMessage + HumanMessage
        logger.info("[IntentDetector] Invoking LLM…")
        flags: IntentFlags = self.llm.invoke([
            SystemMessage(content=prompts.system),
            HumanMessage(content=prompts.user),
        ])
        
        knowledge_graph = store_initial_knowledge_graph(state={"intent_flags": flags.model_dump()}, run_id=run_id)
        print(f"[IntentDetector] flags={flags.model_dump()}")
        logger.info("[IntentDetector] flags=%s", flags.model_dump())

        # Step 4 — fallback: target_column
        target_column = flags.target_column
        if target_column is None:
            logger.info("[IntentDetector] target null — running TargetSuggestionAgent")
            target_column = self.suggester.suggest(schema["df"])
            flags = flags.model_copy(update={"target_column": target_column})

        # Step 5 — fallback: task_type
        if flags.task_type == "unknown" and target_column:
            inferred = self.suggester.suggest_task_type(schema["df"], target_column)
            logger.info("[IntentDetector] task_type unknown — inferred: %s", inferred)
            flags = flags.model_copy(update={"task_type": inferred})

        logger.info(
            "[IntentDetector] Done — target='%s' task='%s' active=%s",
            target_column, flags.task_type,
            [k for k, v in flags.model_dump().items() if k.startswith("run_") and v],
        )

        return {
            "intent_flags":  flags.model_dump(),
            "target_column": target_column,
            "task_type":     flags.task_type,
            "knowledge_graph": knowledge_graph,
        }


# ─────────────────────────────────────────────
# LangGraph node function
# ─────────────────────────────────────────────

def intent_detector_node(state: PipelineState, config: RunnableConfig) -> dict:
    """
    LangGraph node for Agent 0.

    Reads:   state["data_path"], state["nl_query"]
    Returns: partial state dict with intent_flags, target_column, task_type, knowledge_graph.
             LangGraph merges this into the full PipelineState automatically.

    No interrupt() — this node never pauses for human input.
    After this node graph_builder wires a conditional edge to route_after_intent().
    """
    agent = IntentDetectorAgent()
    run_id = config.get("configurable", {}).get("thread_id")
    return agent.run(
        data_path=state["data_path"],
        nl_query=state["nl_query"],
        run_id=run_id,
    )


# ─────────────────────────────────────────────
# Conditional edge function
# ─────────────────────────────────────────────
# graph_builder.py registers this as:
#   graph.add_conditional_edges("intent_detector", route_after_intent)
#
# Agent 0 has NO checkpoint — it routes directly to the first active agent gate.

def route_after_intent(state: PipelineState) -> str:
    """
    Decide which node to visit after Agent 0 completes.
    Returns the name of the first active agent, or skips to its gate.

    Called by LangGraph as a conditional edge — must return a node name
    that exists in the StateGraph.
    """
    flags = state["intent_flags"]
    if flags["eda"]:
        return "eda_agent"
    if flags["preprocessing"]:
        return "preprocessing_agent"
    if flags["feature_engineering"]:
        return "feature_engineering_agent"
    if flags["model_selection"]:
        return "model_selection_agent"
    if flags["training"]:
        return "training_agent"
    if flags["evaluation"]:
        return "evaluation_agent"
    # if flags["deployment"]:
    #     return "deployment_agent"
    return "pipeline_done"