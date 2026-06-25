"""
THE single file that assembles the full LangGraph StateGraph.

Rules enforced here:
    - Every add_node(), add_edge(), add_conditional_edges() lives HERE only.
      No agent file registers itself — agents only export node functions
      and conditional edge functions.
    - Every agent node is immediately followed by a *_checkpoint_node that
      calls interrupt() for Human-in-the-Loop (HITL).
    - Conditional edges after each checkpoint decide:
          "accept"   → move to the next active agent (or pipeline_done)
          "feedback" → loop back to the same agent node for re-execution
    - The graph is compiled with MemorySaver so interrupt()/resume works
      across HTTP requests.

Node naming convention:
    "<agent_name>_agent"       — the agent execution node
    "<agent_name>_checkpoint"  — the HITL interrupt node after it
    "pipeline_done"            — terminal node

How to add a new agent:
    1. Implement agent_node() and route_after_<name>() in agents/<name>.py
       (NO checkpoint_node needed — the factory below generates it)
    2. Import both here
    3. Add add_node() + add_conditional_edges() below, using _make_checkpoint_node()
    4. Wire the checkpoint's conditional edge to the next agent or done

Usage (from agents.static.orchestrator.py):
    from graph.graph_builder import build_graph
    app = build_graph()
    result = app.invoke(initial_state, config={"configurable": {"thread_id": "run-001"}})
"""

import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from state.pipeline_state import PipelineState

# ── Agent node imports ────────────────────────────────────────────────────────
# Each agents/*.py exports exactly:
#   <name>_node(state)          — executes the agent, returns partial state dict
#   route_after_<name>(state)   — conditional edge: "accept" path next node name

from agents.dynamic.intent_detector.intent_detector import (
    intent_detector_node,
    route_after_intent,
)

from agents.dynamic.eda_agent.eda_agent import (
    eda_node,
    route_after_eda,
)

from agents.dynamic.model_selection_agent import (
    model_selection_node,
    route_after_model_selection,
)

from agents.dynamic.training_agent import (
    training_node,
    route_after_training,
)


from agents.dynamic.preprocessing_agent import (
    preprocessing_node,
    route_after_preprocessing,
)

from agents.dynamic.feature_engineering_agent import (
    feature_engineering_node,
    route_after_feature_engineering,
)

from agents.dynamic.deployment_agent import (
    deployment_node,
    route_after_deployment,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint node factory
# ─────────────────────────────────────────────────────────────────────────────
# Each agent gets a HITL checkpoint node.
# interrupt() pauses the graph and surfaces agent_output to the frontend.
# The /resume endpoint calls graph.invoke(None, config) with updated state
# containing user_decision and feedback_text.

def _make_checkpoint_node(agent_name: str):
    """
    Returns a LangGraph node function that:
        1. Calls interrupt() — pauses the graph, pushes agent output to the UI
        2. On resume, reads user_decision from the interrupt response
        3. If feedback: appends to feedback_history, clears user_decision
        4. Returns partial state dict with user_decision + feedback_text

    Args:
        agent_name: used as the key in agent_outputs and feedback_history.

    Returns:
        A node function compatible with graph.add_node().
    """
    def checkpoint_node(state: PipelineState) -> dict:
        logger.info("[%sCheckpoint] Interrupting for human review", agent_name)

        # Pause here — surface the agent's latest output to the frontend
        human_response: dict = interrupt({
            "agent":        agent_name,
            "agent_output": state["agent_outputs"].get(agent_name, {}),
        })

        # human_response arrives via /resume endpoint:
        # {"decision": "accept" | "feedback", "text": "<optional feedback>"}
        decision      = human_response.get("decision", "accept")
        feedback_text = human_response.get("text", "")

        updates: dict = {
            "user_decision": decision,
            "feedback_text": feedback_text,
        }

        # On feedback: record in history so the agent can record it on re-run
        if decision == "feedback" and feedback_text:
            history = list(state.get("feedback_history", []))
            history.append({
                "agent":         agent_name,
                "feedback_text": feedback_text,
                "iteration":     len([h for h in history if h["agent"] == agent_name]) + 1,
            })
            updates["feedback_history"] = history

        logger.info("[%sCheckpoint] decision=%s", agent_name, decision)
        return updates

    checkpoint_node.__name__ = f"{agent_name}_checkpoint_node"
    return checkpoint_node


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint routing factory
# ─────────────────────────────────────────────────────────────────────────────
# After each checkpoint, route:
#   "accept"   → next_node  (the next active agent, resolved by intent_flags)
#   "feedback" → same_agent (re-run the agent with updated feedback_text)

def _make_checkpoint_router(agent_node_name: str, accept_router, feedback_target: str = None):
    """
    Returns a conditional edge function for the checkpoint after an agent.

    Args:
        agent_node_name: name of the agent node this checkpoint follows
                          (used for logging and as the default feedback target).
        accept_router:   the route_after_<agent>() function from the agent file,
                         called when the user accepts to find the next node.
        feedback_target: node to loop back to on "feedback". Defaults to
                         agent_node_name (re-run the same agent). Pass a
                         different node when feedback on this checkpoint
                         should be re-planned upstream instead — e.g. the
                         training checkpoint reroutes to model_selection_agent
                         (see build_graph()) because plan_training already
                         has a reliable LLM step to turn "use XGBoost instead"
                         into a concrete model change, instead of training_agent
                         guessing at it from free text.

    Returns:
        A conditional edge function compatible with add_conditional_edges().
    """
    target = feedback_target or agent_node_name

    def router(state: PipelineState) -> str:
        if state.get("user_decision") == "feedback":
            logger.info("[CheckpointRouter] feedback → re-running %s", target)
            return target
        logger.info("[CheckpointRouter] accept → routing forward")
        return accept_router(state)

    router.__name__ = f"route_{agent_node_name}_checkpoint"
    return router


# ─────────────────────────────────────────────────────────────────────────────
# Terminal node
# ─────────────────────────────────────────────────────────────────────────────

def pipeline_done_node(state: PipelineState) -> dict:
    """
    Terminal node — signals the pipeline completed successfully.
    Writes a summary entry into agent_outputs for the UI.
    """
    logger.info("[PipelineDone] Pipeline complete.")
    outputs = dict(state.get("agent_outputs", {}))
    outputs["pipeline_done"] = {
        "status":       "complete",
        "task_type":    state.get("task_type"),
        "target":       state.get("target_column"),
        "model_path":   state.get("trained_model_path"),
        "endpoint_url": state.get("endpoint_url"),
    }
    return {"agent_outputs": outputs}


# ─────────────────────────────────────────────────────────────────────────────
# Stub agent + checkpoint nodes (replace with real imports progressively)
# ─────────────────────────────────────────────────────────────────────────────
# These stubs let the graph compile and run Agent 0 end-to-end immediately.
# Replace each stub pair with a real import once the agent is implemented.

def _stub_node(name: str):
    def node(state: PipelineState) -> dict:
        logger.info("[STUB] %s — not yet implemented", name)
        outputs = dict(state.get("agent_outputs", {}))
        outputs[name] = {"status": "stub — not yet implemented"}
        return {"agent_outputs": outputs}
    node.__name__ = name
    return node


def _make_stub_router(next_active_flags: list[tuple[str, str]]):
    """
    Stub conditional edge: skip inactive agents, land on the first active one.
    next_active_flags: [(flag_key, node_name), ...]  in pipeline order
    """
    def router(state: PipelineState) -> str:
        flags = state.get("intent_flags", {})
        for flag_key, node_name in next_active_flags:
            if flags.get(flag_key):
                return node_name
        return "pipeline_done"
    return router


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder — THE single place that assembles everything
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> any:
    """
    Assemble and compile the full D.T.D LangGraph StateGraph.

    Returns:
        A compiled LangGraph app (CompiledStateGraph) with MemorySaver
        checkpointer attached — required for interrupt()/resume HITL flow.

    Node sequence (each agent followed by its HITL checkpoint):
        START
          → intent_detector                    (Agent 0 — no checkpoint)
          → eda_agent → eda_checkpoint
          → preprocessing_agent → preprocessing_checkpoint
          → feature_engineering_agent → feature_engineering_checkpoint
          → model_selection_agent → model_selection_checkpoint
          → training_agent → training_checkpoint
          → (evaluation runs as subnode inside training_agent)
          → deployment_agent → deployment_checkpoint
          → pipeline_done
        END
    """
    graph = StateGraph(PipelineState)

    # ── Agent 0: Intent Detector (no HITL — runs once, no re-run) ────────────
    graph.add_node("intent_detector", intent_detector_node)
    graph.set_entry_point("intent_detector")
    graph.add_conditional_edges("intent_detector", route_after_intent)

    # ── Agent 1: EDA ──────────────────────────────────────────────────────────
    # Real implementation — imported from agents.dynamic.eda_agent.eda_agent
    eda_checkpoint_node = _make_checkpoint_node("eda")
    graph.add_node("eda_agent",      eda_node)
    graph.add_node("eda_checkpoint", eda_checkpoint_node)
    graph.add_edge("eda_agent", "eda_checkpoint")
    graph.add_conditional_edges(
        "eda_checkpoint",
        _make_checkpoint_router("eda_agent", route_after_eda),
    )

    # ── Agent 2: Preprocessing ────────────────────────────────────────────────
    preprocessing_checkpoint_node = _make_checkpoint_node("preprocessing")
    graph.add_node("preprocessing_agent",      preprocessing_node)
    graph.add_node("preprocessing_checkpoint", preprocessing_checkpoint_node)
    graph.add_edge("preprocessing_agent", "preprocessing_checkpoint")
    graph.add_conditional_edges(
        "preprocessing_checkpoint",
        _make_checkpoint_router("preprocessing_agent", route_after_preprocessing),
    )

    # ── Agent 3: Feature Engineering ──────────────────────────────────────────
    feature_engineering_checkpoint = _make_checkpoint_node("feature_engineering")
    graph.add_node("feature_engineering_agent",      feature_engineering_node)
    graph.add_node("feature_engineering_checkpoint", feature_engineering_checkpoint)
    graph.add_edge("feature_engineering_agent", "feature_engineering_checkpoint")
    graph.add_conditional_edges(
        "feature_engineering_checkpoint",
        _make_checkpoint_router("feature_engineering_agent", route_after_feature_engineering),
    )

    # ── Agent 4: Model Selection ───────────────────────────────────────────────
    model_selection_checkpoint_node = _make_checkpoint_node("model_selection")
    graph.add_node("model_selection_agent",      model_selection_node)
    graph.add_node("model_selection_checkpoint", model_selection_checkpoint_node)
    graph.add_edge("model_selection_agent", "model_selection_checkpoint")
    graph.add_conditional_edges(
        "model_selection_checkpoint",
        _make_checkpoint_router("model_selection_agent", route_after_model_selection),
    )

    # ── Agent 5: Training ──────────────────────────────────────────────────────
    training_checkpoint_node = _make_checkpoint_node("training")
    graph.add_node("training_agent",      training_node)
    graph.add_node("training_checkpoint", training_checkpoint_node)
    graph.add_edge("training_agent", "training_checkpoint")
    # Feedback on the training checkpoint reroutes to model_selection_agent
    # (not training_agent) — see _make_checkpoint_router()'s feedback_target note.
    graph.add_conditional_edges(
        "training_checkpoint",
        _make_checkpoint_router(
            "training_agent", route_after_training, feedback_target="model_selection_agent"
        ),
    )

    # ── Agent 6: Deployment ────────────────────────────────────────────────────
    deployment_checkpoint = _make_checkpoint_node("deployment")
    graph.add_node("deployment_agent",      deployment_node)
    graph.add_node("deployment_checkpoint", deployment_checkpoint)
    graph.add_edge("deployment_agent", "deployment_checkpoint")
    graph.add_conditional_edges(
        "deployment_checkpoint",
        _make_checkpoint_router("deployment_agent", route_after_deployment),
    )

    # ── Terminal node ──────────────────────────────────────────────────────────
    graph.add_node("pipeline_done", pipeline_done_node)
    graph.add_edge("pipeline_done", END)

    # ── Compile with MemorySaver (required for interrupt()/resume) ─────────────
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    logger.info("[GraphBuilder] Graph compiled successfully.")
    return app