"""
D.T.D (Data To Deployment) — Multi-Agent AutoML Pipeline

What changed vs the old version:
    OLD: while True → LLM picks a tool → tool.invoke() → repeat
         (custom ReAct loop, registry-based, no state persistence)

    NEW: build_graph() → app.invoke(initial_state) → LangGraph drives everything
         - Agent 0 (Intent Detector) replaces the LLM tool-picking loop:
           it decides WHICH agents run and in what combination
         - Each agent is a LangGraph node, not a registry tool
         - HITL checkpoints use interrupt() — the graph pauses and waits
           for human input instead of the LLM deciding the next step
         - MemorySaver persists state across interrupt/resume cycles
         - run_id (thread_id) lets you resume a paused graph later

    KEPT:
         - Same __init__ signature (logger, llm, registry) — registry is
           accepted but ignored; kept so existing call sites don't break
         - Same run(inputs) entry point
         - Same inputs dict keys: data_path, target_column, prompt
         - Console output style matches the original

How to run (CLI):
    python agents/controller_agent.py \\
        --data   path/to/dataset.csv \\
        --query  "train a classifier, target column is Survived" \\
        --target Survived

    Or import and call directly:
        from agents.controller_agent import ControllerAgent
        agent = ControllerAgent(logger=my_logger, llm=None, registry=None)
        result = agent.run({
            "data_path":     "data/titanic.csv",
            "target_column": "Survived",
            "prompt":        "run full pipeline",
        })

HITL resume (after an interrupt):
    result = agent.resume(
        run_id="run-001",
        decision="accept",       # or "feedback"
        feedback_text="",        # only needed when decision == "feedback"
    )

    Or from the command line (after the graph printed a paused run_id):
        python agents/controller_agent.py \\
            --resume  run-001 \\
            --decision accept

        python agents/controller_agent.py \\
            --resume   run-001 \\
            --decision feedback \\
            --feedback "please also compute ROC curve"
"""

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

def _find_project_root() -> Path:
    """
    Walk up the directory tree from this file until we find the folder
    that contains both 'state/' and 'graph/' subdirectories.
    That folder is the project root.
    Falls back to 3 levels up from this file if the marker dirs aren't found.
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):                          # max 10 levels up
        if (current / "state").is_dir() and (current / "graph").is_dir():
            return current
        parent = current.parent
        if parent == current:                    # reached filesystem root
            break
        current = parent

    return Path(__file__).resolve().parents[3]

_PROJECT_ROOT = _find_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from state.pipeline_state import make_initial_state
from graph.graph_builder  import build_graph

logger = logging.getLogger(__name__)
try:
    from dotenv import load_dotenv
    _env_file = _PROJECT_ROOT / ".env"
    if _env_file.exists():
        load_dotenv(dotenv_path=_env_file)   # loads from project root
    else:
        load_dotenv()                         # fallback: searches parent dirs
except ImportError:
    pass

class ControllerAgent:
    """
    Entry point for the D.T.D pipeline.

    Responsibilities:
        1. Accept user inputs (data_path, nl_query, target_column)
        2. Build (or reuse) the compiled LangGraph app
        3. Invoke the graph — LangGraph drives all agent execution
        4. Handle interrupt() pauses — surface them to the caller
        5. Accept resume() calls to continue after human feedback
    """

    def __init__(self, logger=None, llm=None, registry=None):
        """
        Args:
            logger:   Optional logger. Falls back to module-level logger.
            llm:      Accepted for backward compatibility — not used.
                      Each agent constructs its own LLM via tools/llm_client.py.
            registry: Accepted for backward compatibility — not used.
                      Agents are LangGraph nodes, not registry tools.
        """
        self.logger = logger or logging.getLogger(__name__)

        # llm and registry are kept in signature so existing call sites
        if llm is not None:
            self.logger.info(
                "[ControllerAgent] Note: llm argument is no longer used. "
                "Each agent builds its own LLM via tools/llm_client.py."
            )
        if registry is not None:
            self.logger.info(
                "[ControllerAgent] Note: registry argument is no longer used. "
                "Agents are LangGraph nodes registered in graph/graph_builder.py."
            )

        # Build and cache the compiled graph once per ControllerAgent instance.
        # MemorySaver is wired inside build_graph() — it persists state
        # across interrupt/resume cycles keyed by thread_id (run_id).
        self.logger.info("[ControllerAgent] Building LangGraph pipeline…")
        self.app = build_graph()
        self.logger.info("[ControllerAgent] Pipeline ready.")

    # ─────────────────────────────────────────────
    # Primary entry point
    # ─────────────────────────────────────────────

    def run(self, inputs: dict) -> dict:
        """
        Start a new pipeline run.

        Args:
            inputs: dict with keys:
                data_path     (str)  — path to the dataset file
                target_column (str)  — target/label column name (optional; Intent Detector will try to infer it)
                prompt        (str)  — natural-language request
                                       e.g. "run full pipeline"
                                            "just preprocess my data"
                                            "train a classifier on the churn column"
                run_id        (str)  — optional; auto-generated if not provided

        Returns:
            Final PipelineState dict after the pipeline completes or pauses.
            Check result["__interrupted__"] to know if a HITL pause occurred.
        """
        data_path     = inputs.get("data_path")
        target_column = inputs.get("target_column")
        nl_query      = inputs.get("prompt") or inputs.get("nl_query", "")
        report_id     = inputs.get("report_id") or inputs.get("run_id") or f"run-{uuid.uuid4().hex[:8]}"
        run_id        = report_id  # run_id corresponds to report_id in database
        print("Report ID:", report_id)
        print("Run ID:", run_id)
        if not data_path:
            raise ValueError("inputs['data_path'] is required")
        if not nl_query:
            raise ValueError("inputs['prompt'] or inputs['nl_query'] is required")

        self.logger.info(
            "\n" + "=" * 60 + "\n"
            "D.T.D PIPELINE — NEW RUN\n"
            + "=" * 60 + "\n"
            f"report_id   : {run_id}\n"
            f"data_path   : {data_path}\n"
            f"nl_query    : {nl_query}\n"
            f"target_col  : {target_column or '(will be inferred)'}"
        )

        # Build initial state
        # If target_column was explicitly provided, pre-populate it so
        # Intent Detector uses it directly instead of inferring.
        initial_state = make_initial_state(data_path, nl_query)
        if target_column:
            initial_state["target_column"] = target_column

        # LangGraph config — thread_id is the resume key for MemorySaver
        config = {"configurable": {"thread_id": run_id}}

        return self._invoke(initial_state, config, run_id)

    # ─────────────────────────────────────────────
    # Resume after HITL interrupt
    # ─────────────────────────────────────────────

    def resume(
        self,
        run_id: str,
        decision: str,              # "accept" | "feedback"
        feedback_text: str = "",
    ) -> dict:
        """
        Resume a paused pipeline after a Human-in-the-Loop checkpoint.

        Args:
            run_id:        The run_id returned by the run() that paused.
            decision:      "accept" — approve agent output and continue.
                           "feedback" — reject and re-run with feedback_text.
            feedback_text: Free-text feedback used when decision == "feedback".

        Returns:
            Updated PipelineState dict.
            May contain another "__interrupted__" if the next checkpoint fires.

        How it works internally:
            LangGraph's MemorySaver checkpointer has saved the full state at the
            interrupted checkpoint. We load that state and inject the human response
            via app.invoke(None, config) — passing None tells LangGraph to load
            the saved state instead of creating a fresh one. The resume payload is
            passed via the checkpoint node's interrupt response mechanism.
        """
        from langgraph.types import Command

        self.logger.info("\n" + "=" * 60)
        self.logger.info("D.T.D PIPELINE — RESUME")
        self.logger.info("=" * 60)
        self.logger.info(f"run_id   : {run_id}")
        self.logger.info(f"decision : {decision}")
        if feedback_text:
                self.logger.info(f"feedback : {feedback_text}")
        if decision == "feedback":
            self.logger.info("paused")

        config = {"configurable": {"thread_id": run_id}}

        # Create resume command with human response
        # Include full feedback context for the graph to re-run agent if feedback provided
        resume_payload = {
            "decision": decision,
            "feedback_text": feedback_text,
            "text": feedback_text,  # backward compatibility
        }
        # When decision is "feedback", we need to pass the feedback through the interrupt response
        # so the graph can re-run the agent with the user's feedback
        command = Command(resume=resume_payload)
        return self._invoke(command, config, run_id)

    # ─────────────────────────────────────────────
    # Internal invoke wrapper
    # ─────────────────────────────────────────────

    def _invoke(self, input_or_command, config: dict, run_id: str) -> dict:
        """
        Invoke the graph and handle GraphInterrupt (HITL pause) gracefully.

        When interrupt() fires inside a checkpoint node, LangGraph raises
        GraphInterrupt. We catch it, log the pause details, and return a
        state dict with "__interrupted__": True so the caller knows to
        call resume() later.
        """
        from langgraph.errors import GraphInterrupt

        try:
            final_state = self.app.invoke(input_or_command, config)

            # Check if execution paused on an interrupt (newer LangGraph returns interrupt state directly)
            if isinstance(final_state, dict) and final_state.get("__interrupt__"):
                raw = final_state["__interrupt__"]
                interrupt_data = {}
                if isinstance(raw, list) and len(raw) > 0:
                    if hasattr(raw[0], "value"):
                        interrupt_data = raw[0].value
                    else:
                        # try dict or object access
                        try:
                            interrupt_data = raw[0].get("value", raw[0])
                        except Exception:
                            interrupt_data = raw[0]
                elif isinstance(raw, dict):
                    interrupt_data = raw

                agent_name = "unknown"
                agent_output = {}
                if isinstance(interrupt_data, dict):
                    agent_name    = interrupt_data.get("agent", "unknown")
                    agent_output  = interrupt_data.get("agent_output", {})
                    if interrupt_data.get("decision") == "feedback" or interrupt_data.get("feedback_text"):
                            self.logger.info("paused")

                self.logger.info("\n" + "─" * 60)
                self.logger.info(f"[HITL CHECKPOINT] Pipeline paused at: {agent_name}")
                self.logger.info(f"report_id: {run_id}  (use this to resume)")
                self.logger.info("─" * 60)
                self.logger.info("[AGENT OUTPUT PREVIEW]")
                self.logger.info(json.dumps(agent_output, indent=2, default=str)[:1000])
                self.logger.info("─" * 60)
                self.logger.info(
                    "To resume, call:\n"
                    f"  agent.resume(run_id='{run_id}', decision='accept')\n"
                    f"  agent.resume(run_id='{run_id}', decision='feedback', "
                    "feedback_text='your note here')"
                )

                partial_state = dict(final_state)
                partial_state["__interrupted__"] = True
                partial_state["__paused_at__"]   = agent_name
                partial_state["__report_id__"]   = run_id
                partial_state["__run_id__"]      = run_id
                return partial_state

            self.logger.info("\n[ControllerAgent] Pipeline completed successfully.")
            self._log_summary(final_state)
            return final_state

        except GraphInterrupt as interrupt_event:
            # Extract what the checkpoint node surfaced via interrupt({...})
            interrupt_data = {}
            if interrupt_event.args:
                raw = interrupt_event.args[0]
                # LangGraph wraps interrupt value in a list of Interrupt objects
                if isinstance(raw, list) and hasattr(raw[0], "value"):
                    interrupt_data = raw[0].value
                elif isinstance(raw, dict):
                    interrupt_data = raw

            agent_name    = interrupt_data.get("agent", "unknown")
            agent_output  = interrupt_data.get("agent_output", {})
            if interrupt_data.get("decision") == "feedback" or interrupt_data.get("feedback_text"):
                self.logger.info("paused")

            self.logger.info("\n" + "─" * 60)
            self.logger.info(f"[HITL CHECKPOINT] Pipeline paused at: {agent_name}")
            self.logger.info(f"report_id: {run_id}  (use this to resume)")
            self.logger.info("─" * 60)
            self.logger.info("[AGENT OUTPUT PREVIEW]")
            self.logger.info(json.dumps(agent_output, indent=2, default=str)[:1000])
            self.logger.info("─" * 60)
            self.logger.info(
                "To resume, call:\n"
                f"  agent.resume(run_id='{run_id}', decision='accept')\n"
                f"  agent.resume(run_id='{run_id}', decision='feedback', "
                "feedback_text='your note here')"
            )

            # Return a partial state so callers can inspect what ran so far
            # Get current state snapshot from MemorySaver
            try:
                current_snapshot = self.app.get_state(
                    {"configurable": {"thread_id": run_id}}
                )
                partial_state = dict(current_snapshot.values) if current_snapshot else {}
            except Exception:
                partial_state = {}

            partial_state["__interrupted__"] = True
            partial_state["__paused_at__"]   = agent_name
            partial_state["__report_id__"]   = run_id
            partial_state["__run_id__"]      = run_id
            return partial_state

        except Exception as exc:
            self.logger.error(f"[ControllerAgent] Pipeline error: {exc}")
            return {
                "__error__": str(exc),
                "__report_id__": run_id,
                "__run_id__": run_id,
            }

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _log_summary(self, state: dict) -> None:
        """Print a compact summary of what ran and what was produced."""
        self.logger.info("\n── PIPELINE SUMMARY ──────────────────────────────────")
        flags = state.get("intent_flags", {})
        ran   = [k.replace("run_", "") for k, v in flags.items()
                 if k.startswith("run_") and v]
        self.logger.info(f"Agents activated : {', '.join(ran) if ran else 'none'}")
        self.logger.info(f"Target column    : {state.get('target_column')}")
        self.logger.info(f"Task type        : {state.get('task_type')}")
        self.logger.info(f"Trained model    : {state.get('trained_model_path') or '—'}")
        self.logger.info(f"Model metrics    : {state.get('model_metrics') or '—'}")
        self.logger.info(f"Endpoint URL     : {state.get('endpoint_url') or '—'}")
        self.logger.info("─" * 54)


# ─────────────────────────────────────────────────────────────────────────────
# CLI runner
# ─────────────────────────────────────────────────────────────────────────────

def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="controller_agent",
        description="D.T.D Multi-Agent AutoML Pipeline — CLI runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
────────

Start a new full pipeline run:
  python agents/controller_agent.py \\
      --data   data/titanic.csv \\
      --query  "run full pipeline, predict survival" \\
      --target Survived

Run only preprocessing + training:
  python agents/controller_agent.py \\
      --data  data/churn.csv \\
      --query "preprocess then train a classifier on the Churn column"

Run with an explicit run_id (useful for reproducibility):
  python agents/controller_agent.py \\
      --data   data/titanic.csv \\
      --query  "full pipeline" \\
      --run-id my-experiment-01

Resume after a HITL pause (accept):
  python agents/controller_agent.py \\
      --resume   run-4f3a9b12 \\
      --decision accept

Resume after a HITL pause (feedback):
  python agents/controller_agent.py \\
      --resume   run-4f3a9b12 \\
      --decision feedback \\
      --feedback "please also generate a correlation heatmap"
""",
    )

    # ── New run arguments ─────────────────────────────────────────────────────
    new_run = p.add_argument_group("New run")
    new_run.add_argument(
        "--data", "-d",
        metavar="PATH",
        help="Path to dataset file (.csv / .xlsx / .parquet / .json)",
    )
    new_run.add_argument(
        "--query", "-q",
        metavar="TEXT",
        help='Natural-language request. e.g. "run full pipeline"',
    )
    new_run.add_argument(
        "--target", "-t",
        metavar="COLUMN",
        default=None,
        help="Target column name (optional — will be inferred if omitted)",
    )
    new_run.add_argument(
        "--run-id",
        metavar="ID",
        default=None,
        help="Explicit run ID (auto-generated if omitted)",
    )

    # ── Resume arguments ──────────────────────────────────────────────────────
    resume_grp = p.add_argument_group("Resume after HITL pause")
    resume_grp.add_argument(
        "--resume", "-r",
        metavar="RUN_ID",
        default=None,
        help="run_id of a previously paused pipeline to resume",
    )
    resume_grp.add_argument(
        "--decision",
        choices=["accept", "feedback"],
        default="accept",
        help="HITL decision: 'accept' continues, 'feedback' re-runs the agent",
    )
    resume_grp.add_argument(
        "--feedback", "-f",
        metavar="TEXT",
        default="",
        help="Feedback text (only used when --decision feedback)",
    )

    return p


def main():

    
    agent = ControllerAgent(logger=logger, llm=None, registry=None)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_cli_parser()
    args   = parser.parse_args()

    agent = ControllerAgent()

    # ── Resume mode ───────────────────────────────────────────────────────────
    if args.resume:
        result = agent.resume(
            run_id=args.resume,
            decision=args.decision,
            feedback_text=args.feedback,
        )

    # ── New run mode ──────────────────────────────────────────────────────────
    else:
        # if not args.data or not args.query:
        #     parser.error("--data and --query are required for a new run.\n"
        #                  "Use --help to see all options.")

        # result = agent.run({
        #     "data_path":     args.data,
        #     "prompt":        args.query,
        #     "target_column": args.target,
        #     "run_id":        args.run_id,
        # })
        data_path = args.data
        prompt = args.query
        target_column = args.target

        if not data_path:
            default_path = _PROJECT_ROOT / "assets/data/Classification Datasets/Titanic-Dataset.csv"
            if default_path.exists():
                data_path = str(default_path)
            else:
                default_path_alt = _PROJECT_ROOT / "assets/data/Datasets/Classification Datasets/Iris.csv"
                if default_path_alt.exists():
                    data_path = str(default_path_alt)
                else:
                    data_path = "assets/data/Classification Datasets/Titanic-Dataset.csv"
        
        if not prompt:
            prompt = "analyze this data and train a model"

        result = agent.run({
            "data_path":     data_path,
            "prompt":        prompt,
            "target_column": target_column,
            "run_id":        args.run_id,
            "report_id":     args.run_id,
        })

    # ── Output ────────────────────────────────────────────────────────────────
    if result.get("__interrupted__"):
        print(f"\n⏸  Pipeline paused at: {result['__paused_at__']}")
        print(f"   run_id: {result['__run_id__']}")
        print("\n   To accept:   python agents/controller_agent.py "
              f"--resume {result['__run_id__']} --decision accept")
        print("   To provide feedback:  python agents/controller_agent.py "
              f"--resume {result['__run_id__']} --decision feedback "
              "--feedback \"your note\"")
        sys.exit(0)

    if result.get("__error__"):
        print(f"\n❌  Pipeline error: {result['__error__']}")
        sys.exit(1)

    print("\n✅  Pipeline complete.")
    print(f"   Target column : {result.get('target_column')}")
    print(f"   Task type     : {result.get('task_type')}")
    print(f"   Trained model : {result.get('trained_model_path') or '—'}")
    print(f"   Endpoint URL  : {result.get('endpoint_url') or '—'}")
    sys.exit(0)


if __name__ == "__main__":
    main()