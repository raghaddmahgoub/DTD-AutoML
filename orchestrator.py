from importlib.resources import path
import os
import pandas as pd
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import json
from agents.preprocessing_agent.preprocessing_pipeline import PreprocessingPipelineAgent

# Import modular agents from your folders
from agents.eda_agent.eda_agent import EDAAgent
from agents.automl_agent.automl_agent import AutoMLAgent
from agents.preprocessing_agent.preprocessing_node import preprocessing_node

load_dotenv()

ORCHESTRATOR_CONFIG = {
    "data_path": "assets/data/Datasets/Classification Datasets/Titanic-Dataset.csv",
    "target_column": "Survived",
    "task_type": "classification",
    "preprocessing_output_root": os.path.join("Output", "Preprocessing"),
    "use_preprocessing_llm": False,
}

class AgentState(TypedDict):
    """Shared pipeline memory."""
    data_path: str
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


class DTDPipeline:
    def __init__(self):
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

        # 1. Define Nodes (The 4 Stages)
        builder.add_node("raw_analysis",    self.stage_raw_analysis)
        builder.add_node("preprocessing",   self.stage_preprocessing)
        builder.add_node("clean_analysis",  self.stage_clean_analysis)
        builder.add_node("automl_training", self.stage_automl)

        # 2. Define Flow
        builder.set_entry_point("raw_analysis")
        builder.add_edge("raw_analysis",    "preprocessing")
        builder.add_edge("preprocessing",   "clean_analysis")
        builder.add_edge("clean_analysis",  "automl_training")
        builder.add_edge("automl_training", END)
        # builder.add_edge("clean_analysis",  END)

        return builder.compile()

    # --- Node Implementations ---

    def stage_raw_analysis(self, state: AgentState):
        """First Analysis: Identify issues for the Preprocessor."""
        print("\n🔍 [Stage 1] Running Raw Data Analysis...")

        df = pd.read_csv(state['data_path'])
        agent = EDAAgent(df, target_column=state['target_column'], df_name=self._get_dataset_name(
            state['data_path']))
        agent.run(run_type="raw")
        results = agent.export(output_dir="Output")

        frontend_json_path = results.get("frontend_json_path")
        with open(frontend_json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)

        state["agent_output"] = {
            "stage":        "raw_analysis",
            "raw_analysis": analysis_data,
        }

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

        # state["agent_output"] = {
        #     "stage":          "preprocessing",
        #     "X_train":        result_state["X_train_path"],
        #     "X_test":         result_state["X_test_path"],
        #     "y_train":        result_state["y_train_path"],
        #     "y_test":         result_state["y_test_path"],
        #     "full_dataset":   state["clean_data_path"],
        #     "summary":        result_state["summary_path"],
        #     "column_actions": result_state["column_actions_path"],
        #     "column_actions_frontend": result_state.get("column_actions_frontend_path"),
        #     "policy":         result_state.get("policy_path"),
        #     "evidence":       result_state.get("evidence_path"),
        # }

        column_actions_frontend = None
        column_actions_path = result_state.get("column_actions_frontend_path")

        if column_actions_path and os.path.exists(column_actions_path):
            try:
                with open(column_actions_path, "r", encoding="utf-8") as f:
                    column_actions_frontend = json.load(f)
            except Exception as e:
                column_actions_frontend = {"error": f"Failed to load JSON: {str(e)}"}
        state["agent_output"] = {
            "stage":"preprocessing",  
            "column_actions": column_actions_frontend,
        }
        # print(column_actions_frontend)
        print(f"✅ Preprocessing complete.")
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
        results = agent.export(output_dir="Output")

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

        state["agent_output"] = {
            "stage":          "clean_analysis",
            "clean_analysis": analysis_data,
        }

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
                output_dir="Output/automl",
                automl_directives=directives,
                problem_type=task_type,
                # X_train_path=state.get("X_train_path"),
                # X_test_path=state.get("X_test_path"),
                # y_train_path=state.get("y_train_path"),
                # y_test_path=state.get("y_test_path"),
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
                # metrics["optuna_refined_config"] = raw_metrics.get("optuna_refined_config")
                from datetime import datetime

                # metrics = final_subagent_state.get('model_metrics')
                # run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                state['final_metrics'] = metrics
                state['saved_files'] = final_subagent_state.get(
                    'saved_files', {})
                state["agent_output"] = {
                    "run_timestamp": final_subagent_state.get("run_timestamp"),
                    "data_path": final_subagent_state.get("data_path"),
                    "target_column": target_col,
                    "problem_type": task_type,
                     "model_selection": {
                        "use_automl": final_subagent_state.get("use_automl"),
                        "automl_config": final_subagent_state.get("automl_config"),
                        "selected_models": final_subagent_state.get("selected_models"),
                        "model_selection_reasoning": final_subagent_state.get("model_selection_reasoning"),
                    },
                    "training_results": metrics, # metrics is final_subagent_state['model_metrics']
                    "agent_messages": final_subagent_state.get("agent_messages",[]),
                    "workflow":  {
                        "final_step": final_subagent_state.get("step"),
                        "error": final_subagent_state.get("error")
                    }
                }

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
    pipeline = DTDPipeline()

    inputs = {
        "data_path":     ORCHESTRATOR_CONFIG["data_path"],
        "target_column": ORCHESTRATOR_CONFIG["target_column"],
        "task_type":     ORCHESTRATOR_CONFIG["task_type"],
        }
    # inputs = {
    #     "data_path":     "assets/data/Datasets/Regression Datasets/car_prices.csv",
    #     "target_column": "sellingprice",
    #     "task_type":     "regression"
    # }
    # inputs = {
    #     "data_path":     "assets/data/Datasets/Classification Datasets/Titanic-Dataset.csv",
    #     "target_column": "Survived",
    #     "task_type":     "classification"  # or "regression" based on your dataset
    # }

    #------------------------------Large Datasets
    # inputs = {
    #     "data_path":     "assets/data/Datasets/Classification Datasets/customer_spending_1M_2018_2025.csv",
    #     "target_column": "Referral",
    #     "task_type":     "classification"
    # }
    # inputs = {
    #     "data_path":     "assets/data/Datasets/Regression Datasets/student_performance.csv",
    #     "target_column": "total_score",
    #     "task_type":     "regression"
    # }

    result = pipeline.workflow.invoke(inputs)

    print("\n🏁 Pipeline Finished.")
    if result.get('error'):
        print(f"⚠️  Completed with error: {result['error']}")
    else:
        print(f"Best Model Metrics: {result.get('final_metrics')}")
