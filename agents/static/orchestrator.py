from importlib.resources import path
import os
import sys
import time
import pandas as pd
from typing import TypedDict, Optional, Generator
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import json
from pathlib import Path

# Add project root to sys.path to allow absolute imports when run directly from package subfolders
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force UTF-8 encoding for standard streams to prevent UnicodeEncodeError on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Import modular agents from your folders
from agents.static.eda_agent.eda_agent import EDAAgent
from agents.static.automl_agent.automl_agent import AutoMLAgent
from agents.static.preprocessing_agent.preprocessing_node import preprocessing_node

from cache.cache_manager import PipelineCacheManager

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORCHESTRATOR_CONFIG = {
    "data_path": str(PROJECT_ROOT / "assets/data/Classification Datasets/Titanic-Dataset.csv"),
    "target_column": "Survived",
    "preprocessing_output_root": str(PROJECT_ROOT / "Output" / "static" / "Preprocessing"),
    "use_preprocessing_llm": True,
}

class AgentState(TypedDict, total=False):
    """Shared pipeline memory."""
    data_path: str
    cache_key: str
    dataset_snapshot: dict
    clean_data_path: str
    X_train_path: Optional[str]
    X_test_path: Optional[str]
    y_train_path: Optional[str]
    y_test_path: Optional[str]
    target_column: str
    task_type: str
    analysis_report_path: Optional[str]
    automl_directives: Optional[dict]
    final_metrics: Optional[dict]
    saved_files: Optional[dict]
    agent_output: Optional[dict]
    error: Optional[str]
    _cache_hit: Optional[bool]
    _cached_stages: Optional[dict]

class DTDPipeline:
    def __init__(self):
        self.cache = PipelineCacheManager(cache_root=PROJECT_ROOT / "Output" / "static" / "cache")
        self.workflow = self._build_graph()

    def _get_dataset_name(self, path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    def _get_preprocessing_output_folder(self, data_path: str) -> str:
        dataset_name = self._get_dataset_name(data_path)
        return os.path.join(ORCHESTRATOR_CONFIG["preprocessing_output_root"], dataset_name)

    def _build_clean_dataset_from_splits(
        self,
        X_train_path: str,
        X_test_path: str,
        y_train_path: str,
        y_test_path: str,
        target_column: str,
        output_folder: str,
    ) -> str:
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path).squeeze("columns")
        y_test = pd.read_csv(y_test_path).squeeze("columns")

        train_df = X_train.copy()
        train_df[target_column] = y_train.reset_index(drop=True)

        test_df = X_test.copy()
        test_df[target_column] = y_test.reset_index(drop=True)

        clean_df = pd.concat([train_df, test_df], ignore_index=True)
        clean_data_path = os.path.join(output_folder, "full_preprocessed.csv")
        clean_df.to_csv(clean_data_path, index=False)
        return clean_data_path

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("cache_check",     self.stage_cache_check)
        # 1. Define Nodes (The 4 Stages)
        builder.add_node("raw_analysis",    self.stage_raw_analysis)
        builder.add_node("preprocessing",   self.stage_preprocessing)
        builder.add_node("clean_analysis",  self.stage_clean_analysis)
        builder.add_node("automl_training", self.stage_automl)

        # 2. Define Flow
        builder.set_entry_point("cache_check")
        builder.add_conditional_edges(
            "cache_check",
            lambda s: "hit" if s.get("_cache_hit") else "miss",
            {
                "hit":  END,
                "miss": "raw_analysis",
            },
        )

        builder.add_edge("raw_analysis",    "preprocessing")
        builder.add_edge("preprocessing",   "clean_analysis")
        builder.add_edge("clean_analysis",  "automl_training")
        builder.add_edge("automl_training", END)
        # builder.add_edge("clean_analysis",  END)

        return builder.compile()

    def stream_stages(self, inputs: dict) -> Generator[dict, None, None]:
        """
        Runs the pipeline stage-by-stage and yields a dict the moment each
        stage finishes.  The API converts each dict to an SSE event immediately
        so the frontend sees results arriving one stage at a time.
 
        Yielded shape:
            {
                "node_name":    str,
                "agent_output": dict | None,
                "error":        str  | None,
                "cache_hit":    bool,
                "_state":       dict,   # full state (internal — not forwarded to client)
            }
        """
        state: AgentState = dict(inputs)
        STAGE_ORDER = ["raw_analysis", "preprocessing", "clean_analysis", "automl_training"]
        # ── cache check ───────────────────────────────────────────────────────
        print("\n🔎 [Cache] Checking for cached results…")
        hit, cached = self.cache.lookup(state["data_path"], state["target_column"])
 
        if hit:
            print("⚡ Cache HIT — replaying stages from disk.")
            stages = cached["stages"]
 
            # Banner event (no output — just signals a cache hit to the frontend)
            yield {"node_name": "cache_check", "agent_output": None, "error": None,
                   "cache_hit": True, "_state": state}
 
            # One event per stage, same shape as a live run
            for stage in STAGE_ORDER:
                stage_output = stages.get(stage)
                if stage_output is None:
                    continue
                print(f"  ↳ replaying: {stage}")
                yield {"node_name": stage, "agent_output": stage_output, "error": None,
                       "cache_hit": True, "_state": state}
            return
 
        # ── live pipeline ─────────────────────────────────────────────────────
        print("📭 Cache MISS — running full pipeline.")
 
        for stage_fn, stage_name in [
            (self.stage_raw_analysis,   "raw_analysis"),
            (self.stage_preprocessing,  "preprocessing"),
            (self.stage_clean_analysis, "clean_analysis"),
            (self.stage_automl,         "automl_training"),
        ]:
            state = stage_fn(state)
            yield {
                "node_name":    stage_name,
                "agent_output": state.get("agent_output"),
                "error":        state.get("error"),
                "cache_hit":    False,
                "_state":       state,
            }
            if state.get("error"):
                return 

    def stage_cache_check(self, state: AgentState) -> AgentState:
        """
        Strict content-based cache lookup.
        Hashes the raw CSV bytes + target_column — filename is never used.
        """
        print("\n🔎 [Cache] Checking for cached results…")
 
        hit, cached = self.cache.lookup(state["data_path"], state["target_column"])
 
        if hit:
            print("⚡ Returning cached pipeline results — skipping all stages.")
            stages = cached["stages"]
 
            # Restore the pieces of state the downstream caller may need
            automl_out = stages.get("automl_training", {})
            state["_cache_hit"]     = True
            state["_cached_stages"] = stages
            state["task_type"]      = cached["meta"].get("task_type")
            state["final_metrics"]  = (
                automl_out.get("training_results")
                or automl_out.get("final_metrics")
            )
            state["agent_output"] = {
                "source":         "cache",
                "cache_meta":     cached["meta"],
                "artifacts_dir":  cached["artifacts_dir"],
                "stages":         stages,
            }
        else:
            print("📭 No valid cache entry found — running full pipeline.")
            state["_cache_hit"] = False
 
        return state

    # --- Node Implementations ---

    def stage_raw_analysis(self, state: AgentState):
        """First Analysis: Identify issues for the Preprocessor."""
        print("\n🔍 [Stage 1] Running Raw Data Analysis...")
        
        df = pd.read_csv(state['data_path'])
        agent = EDAAgent(df,target_column=state['target_column'],df_name=self._get_dataset_name(state['data_path']))
        agent.run(run_type="raw")
        results = agent.export(output_dir=str(PROJECT_ROOT / "Output" / "static" / "orchestrator"))

        frontend_json_path = results.get("frontend_json_path")
        with open(frontend_json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)

        agent_output = {
            "stage":        "raw_analysis",
            "raw_analysis": analysis_data,
        }
        state["agent_output"] = agent_output


        self.cache.save_stage(
            state["data_path"], state["target_column"],
            "raw_analysis", agent_output,
        )

        return state

    def stage_preprocessing(self, state: AgentState):
        """Preprocessing using PreprocessingNode."""
        print("\n🛠️ [Stage 2] Running Preprocessing Node...")
        
        output_folder = self._get_preprocessing_output_folder(
            state["data_path"])

        preprocessing_state = {
            "dataset_path":  state["data_path"],
            "target_column": state["target_column"],
            "output_folder": output_folder,
            "use_llm": ORCHESTRATOR_CONFIG["use_preprocessing_llm"],
        }

        result_state = preprocessing_node(preprocessing_state)

        if result_state.get("status") != "success":
            print(f"❌ Preprocessing failed: {result_state.get('error')}")
            state["error"] = result_state.get("error")
            state["agent_output"] = {
                "stage": "preprocessing",
                "error": state["error"]
            }
            return state
        summary_path = result_state.get("summary_path")

        if summary_path and os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)

                state["task_type"] = summary.get("task_type")   # ✅ SET IT HERE

            except Exception as e:
                print(f"⚠️ Failed to read task_type from preprocessing summary: {e}")

        state["X_train_path"] = result_state["X_train_path"]
        state["X_test_path"] = result_state["X_test_path"]
        state["y_train_path"] = result_state["y_train_path"]
        state["y_test_path"] = result_state["y_test_path"]
        state["clean_data_path"] = self._build_clean_dataset_from_splits(
            X_train_path=state["X_train_path"],
            X_test_path=state["X_test_path"],
            y_train_path=state["y_train_path"],
            y_test_path=state["y_test_path"],
            target_column=state["target_column"],
            output_folder=result_state["output_folder"],
        )

        column_actions_frontend = None
        column_actions_path = result_state.get("column_actions_frontend_path")

        if column_actions_path and os.path.exists(column_actions_path):
            try:
                with open(column_actions_path, "r", encoding="utf-8") as f:
                    column_actions_frontend = json.load(f)
            except Exception as e:
                column_actions_frontend = {"error": f"Failed to load JSON: {str(e)}"}

        agent_output = {
            "stage":          "preprocessing",
            "task_type":      state.get("task_type"),
            "column_actions": column_actions_frontend,
        }
        state["agent_output"] = agent_output
 
        self.cache.save_stage(
            state["data_path"], state["target_column"],
            "preprocessing", agent_output,
        )

        # print(column_actions_frontend)
        print(f"✅ Preprocessing complete.")
        print(f"📊 Problem Type: {state['task_type']}")
        print(f"📂 Output folder: {result_state['output_folder']}")
        print(f"📄 Rebuilt full dataset: {state['clean_data_path']}")

        return state

    def stage_clean_analysis(self, state: AgentState):
        """Second Analysis: Generate Directives for AutoML."""
        print("\n📊 [Stage 3] Running Post-Prep Analysis...")
        
        df = pd.read_csv(state['clean_data_path'], low_memory=False)
        agent = EDAAgent(df, target_column=state['target_column'], df_name=self._get_dataset_name(
            state['clean_data_path']))
        agent.run(run_type="clean")
        results = agent.export(output_dir=str(PROJECT_ROOT / "Output" / "static" / "orchestrator"))

        state['automl_directives'] = results.get("automl_context")

        # Safeguard: ensure report & target_analysis exist
        directives = state['automl_directives'] or {}
        if 'report' not in directives or directives['report'] is None:
            directives['report'] = {
                "target_analysis": {
                    "column": state.get('target_column', 'Survived'),
                    "skew_severity": "N/A"
                }
            }
        elif 'target_analysis' not in directives['report'] or directives['report']['target_analysis'] is None:
            directives['report']['target_analysis'] = {
                "column": state.get('target_column', 'Survived'),
                "skew_severity": "N/A"
            }

        state['automl_directives'] = directives

        frontend_json_path = results.get("frontend_json_path")
        with open(frontend_json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)

        agent_output = {
            "stage":          "clean_analysis",
            "clean_analysis": analysis_data,
        }
        state["agent_output"] = agent_output
 
        self.cache.save_stage(
            state["data_path"], state["target_column"],
            "clean_analysis", agent_output,
        )


        return state

    def stage_automl(self, state: AgentState):
        """Final Stage: Model Selection & Training."""
        print("\n🤖 [Stage 4] Running AutoML Training...")
        
        # 1. Extract directives generated by Stage 3
        directives = state.get('automl_directives')
        if not directives:
            print("❌ Error: No automl_directives found in state.")
            state['error'] = "Missing analysis directives for training."
            return state

        # 2. Extract target and task info
        target_info = directives['report']['target_analysis']
        target_col = target_info['column']
        task_type = state.get('task_type') or directives.get('task_type')

        print(f"🎯 Target identified: {target_col}")
        print(f"📊 Problem Type: {task_type}")

        # 3. Instantiate AutoMLAgent
        automl_agent_instance = AutoMLAgent()

        try:
            # 4. Use run() so _save_outputs() is triggered automatically
            print(f"⏳ Training models for {target_col}...")
            final_subagent_state = automl_agent_instance.run(
                data_path=state['clean_data_path'],
                target_column=target_col,
                output_dir=str(PROJECT_ROOT / "Output" / "static" / "automl"),
                automl_directives=directives,
                problem_type=task_type,
              
            )

            # 5. Capture results back into orchestrator state
            if final_subagent_state.get('error'):
                state['error'] = final_subagent_state['error']
                print(f"❌ Training failed: {state['error']}")
                state["agent_output"] = {
                    "stage": "automl_training",
                    "error": state["error"]
                }

            else:
                raw_metrics = final_subagent_state.get('model_metrics', {})

                # Flatten structure
                if "all_metrics" in raw_metrics:
                    metrics = raw_metrics["all_metrics"]
                else:
                    metrics = raw_metrics

                # Add missing fields
            
                from datetime import datetime

                
                state['final_metrics'] = metrics
                state['saved_files'] = final_subagent_state.get(
                    'saved_files', {})
                
                agent_output = {
                    "run_timestamp":  final_subagent_state.get("run_timestamp"),
                    "data_path":      final_subagent_state.get("data_path"),
                    "target_column":  target_col,
                    "problem_type":   task_type,
                    "model_selection": {
                        "use_automl":               final_subagent_state.get("use_automl"),
                        "automl_config":            final_subagent_state.get("automl_config"),
                        "selected_models":          final_subagent_state.get("selected_models"),
                        "model_selection_reasoning": final_subagent_state.get("model_selection_reasoning"),
                    },
                    "training_results": metrics,
                    "agent_messages":   final_subagent_state.get("agent_messages", []),
                    "workflow": {
                        "final_step": final_subagent_state.get("step"),
                        "error":      final_subagent_state.get("error"),
                    },
                }

                state["agent_output"] = agent_output
 
                self.cache.save_stage(
                    state["data_path"], state["target_column"],
                    "automl_training", agent_output,
                )
 
                self.cache.finalize(
                    state["data_path"],
                    state["target_column"],
                    state,
                )

                print(
                    f"✅ Training complete. Best Model: {state['final_metrics'].get('best_model')}")
                print(
                    f"📈 Final Score: {state['final_metrics'].get('best_score'):.4f}")
                print(f"💾 Saved outputs:")
                for ftype, fpath in state.get('saved_files', {}).items():
                    print(f"   {ftype:10s} → {fpath}")

        except Exception as e:
            print(f"❌ Exception in Stage 4: {str(e)}")
            state['error'] = f"AutoML Stage failed: {str(e)}"
            state["agent_output"] = {
                "stage": "automl_training",
                "error": state["error"]
            }

        return state

    def visualize_graph(self, output_path="pipeline_graph.png"):
        """Generates a PNG image of the LangGraph workflow."""
        try:
            graph_image = self.workflow.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(graph_image)
            print(f"✅ Pipeline visualization saved to: {output_path}")
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            print("Falling back to ASCII representation:")
            print(self.workflow.get_graph().print_ascii())


# --- Main Execution ---
if __name__ == "__main__":
    runtime_start=time.time()
    pipeline = DTDPipeline()

    inputs = {
        "data_path": ORCHESTRATOR_CONFIG["data_path"],
        "target_column": ORCHESTRATOR_CONFIG["target_column"],
    }
    final_state = inputs
    for event in pipeline.stream_stages(inputs):
        print(f"[{event['node_name']}] done — error={event['error']}")
        final_state = event["_state"]

    runtime_end=time.time()
    runtime_duration = runtime_end - runtime_start
    minutes = int(runtime_duration // 60)
    seconds = runtime_duration % 60

    print(f"\n🏁 Pipeline Finished in {minutes} min {seconds:.2f} sec.")
    
    if final_state.get("error"):
        print(f"⚠️  Completed with error: {final_state['error']}")
    else:
        print(f"Best Model Metrics: {final_state.get('final_metrics')}")

    # if result.get('error'):
    #     print(f"⚠️  Completed with error: {result['error']}")
    # else:
    #     print(f"Best Model Metrics: {result.get('final_metrics')}")
