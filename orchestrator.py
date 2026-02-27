import os
import pandas as pd
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import json
from agents.preprocessing_agent.preprocessing_pipeline import PreprocessingPipelineAgent

# Import modular agents from your folders
from agents.eda_agent.eda_agent import EDAAgent
# from agents.preprocessing_agent.preprocessor import PreprocessingAgent
from agents.automl_agent.automl_agent import AutoMLAgent
from agents.preprocessing_agent.preprocessing_node import preprocessing_node

load_dotenv()

class AgentState(TypedDict):
    """Shared pipeline memory."""
    data_path: str
    clean_data_path: str
    target_column: str
    task_type: str
    analysis_report_path: Optional[str]
    automl_directives: Optional[dict]
    final_metrics: Optional[dict]
    agent_output: Optional[dict]

class DTDPipeline:
    def __init__(self):
        self.workflow = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)

        # 1. Define Nodes (The 4 Stages)
        builder.add_node("raw_analysis", self.stage_raw_analysis)
        builder.add_node("preprocessing", self.stage_preprocessing)
        builder.add_node("clean_analysis", self.stage_clean_analysis)
        builder.add_node("automl_training", self.stage_automl)

        # 2. Define Flow
        builder.set_entry_point("raw_analysis")
        builder.add_edge("raw_analysis", "preprocessing")
        builder.add_edge("preprocessing", "clean_analysis")
        builder.add_edge("clean_analysis", "automl_training")
        builder.add_edge("automl_training", END)
        # builder.add_edge("clean_analysis", END)

        return builder.compile()

    # --- Node Implementations ---

    def stage_raw_analysis(self, state: AgentState):
        """First Analysis: Identify issues for the Preprocessor."""
        print("🔍 [Stage 1] Running Raw Data Analysis...")
        df = pd.read_csv(state['data_path'])
        agent = EDAAgent(df, target_column=state['target_column'], df_name="raw_data")
        
        agent.run(run_type="raw") # Internal route for raw analysis
        results = agent.export(output_dir="Output/raw")
        
        frontend_json_path = results.get("frontend_json_path")
        with open(frontend_json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)

        state["agent_output"] = {
            "stage": "raw_analysis",
            "raw_analysis": analysis_data, 
        }

        return state

    # def stage_preprocessing(self, state: AgentState):
    #     """Preprocessing: Clean data based on Raw Analysis."""
    #     print("🛠️ [Stage 2] Running Preprocessing Agent...")
    #     agent = PreprocessingPipelineAgent()
    #     # clean_path = state['data_path'].replace(".csv", "_clean.csv")
    #     # state['clean_data_path'] = clean_path 
    #     prep_results = agent.run(
    #         data_path=state['data_path'],
    #         target_column=state['target_column']
    #     )

    #     # Save full preprocessing output
    #     state['preprocessing_results'] = prep_results

    #     # ✅ Use exported FULL preprocessed dataset for next stages
    #     state['clean_data_path'] = prep_results['exports']['full']

    #     # ✅ Update inferred task type from preprocessing agent
    #     state['task_type'] = prep_results['task_type']

    #     print(f"✅ Preprocessed dataset saved at: {state['clean_data_path']}")
    #     print(f"📊 Inferred task type: {state['task_type']}")
    #     state["agent_output"] = {
    #         "stage": "preprocessing",
    #         "task_type": prep_results["task_type"],
    #         "best_cv_score": prep_results["best_cv_score"],
    #         "exported_files": prep_results["exports"]
    #     }

    #     return state
    def stage_preprocessing(self, state: AgentState):
        """Preprocessing using PreprocessingNode from test.py"""
        print("🛠️ [Stage 2] Running Preprocessing Node...")

        # Prepare state expected by preprocessing_node
        preprocessing_state = {
            "dataset_path": state["data_path"],
            "target_column": state["target_column"],
            "output_folder": "Output/Preprocessing"
        }

        # Run the node
        result_state = preprocessing_node(preprocessing_state)

        # Handle failure
        if result_state.get("status") != "success":
            print(f"❌ Preprocessing failed: {result_state.get('error')}")
            state["error"] = result_state.get("error")
            state["agent_output"] = {
                "stage": "preprocessing",
                "error": state["error"]
            }
            return state

        state["clean_data_path"] = result_state["full_dataset_path"]

        state["agent_output"] = {
            "stage": "preprocessing",
            "X_train": result_state["X_train_path"],
            "X_test": result_state["X_test_path"],
            "y_train": result_state["y_train_path"],
            "y_test": result_state["y_test_path"],
            "full_dataset": result_state["full_dataset_path"],
            "summary": result_state["summary_path"],
            "column_actions": result_state["column_actions_path"]
        }

        print(f"✅ Preprocessing complete.")
        print(f"📂 Output folder: {result_state['output_folder']}")

        return state
    
    def stage_clean_analysis(self, state: AgentState):
        """Second Analysis: Generate Directives for AutoML."""
        print("📊 [Stage 3] Running Post-Prep Analysis...")
        df = pd.read_csv(state['clean_data_path'])
        # df = pd.read_csv(state['data_path'])
        agent = EDAAgent(df, target_column=state['target_column'], df_name="clean_data")
        
        agent.run(run_type="clean") # Internal route for clean analysis
        results = agent.export(output_dir="Output/clean")
        
        # Save the reformatted JSON directives for the AutoML agent
        state['automl_directives'] = results.get("automl_context")
        frontend_json_path = results.get("frontend_json_path")
        with open(frontend_json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        state["agent_output"] = {
            "stage": "clean_analysis",
            "clean_analysis": analysis_data, # Send the actual JSON object
        }

        return state

    def stage_automl(self, state: AgentState):
        """
        Final Stage: Model Selection & Training.
        """
        print("🤖 [Stage 4] Running AutoML Training...")
        
        # 1. Extract directives generated by the Analysis Agent in Stage 3
        directives = state.get('automl_directives')
        if not directives:
            print("❌ Error: No automl_directives found in state.")
            state['error'] = "Missing analysis directives for training."
            return state

        # 2. Extract target and task info for logging/verification
        # Accessing nested info as defined in eda_agent2.py
        target_info = directives['report']['target_analysis']
        target_col = target_info['column']
        task_type = state.get('task_type') or directives.get('task_type')
        
        print(f"🎯 Target identified: {target_col}")
        print(f"📊 Problem Type: {task_type}")

        # 3. Instantiate and configure the AutoMLAgent
        # This agent uses LangGraph to manage model selection and training
        automl_agent_instance = AutoMLAgent()

        # 4. Prepare the initial state for the AutoMLAgent
        # We pass the external directives into the sub-agent's starting state
        subagent_initial_state = {
            'data_path': state['clean_data_path'],
            # 'data_path': state['data_path'],
            'target_column': target_col,
            'automl_directives': directives, # Handshake JSON
            'problem_type': task_type,
            'step': 'initialized',
            'agent_messages': []
        }

        try:
            # 5. Invoke the AutoMLAgent's internal graph
            print(f"⏳ Training models for {target_col}...")
            final_subagent_state = automl_agent_instance.graph.invoke(subagent_initial_state)

            # 6. Capture results back into the main orchestrator state
            if final_subagent_state.get('error'):
                state['error'] = final_subagent_state['error']
                print(f"❌ Training failed: {state['error']}")
            else:
                metrics = final_subagent_state.get('model_metrics')
                state['final_metrics'] = metrics

                state["agent_output"] = {
                    "stage": "automl_training",
                    "best_model": metrics.get("best_model"),
                    "best_score": metrics.get("best_score"),
                    "all_metrics": metrics
                }
                print(f"✅ Training complete. Best Model: {state['final_metrics'].get('best_model')}")
                print(f"📈 Final Score: {state['final_metrics'].get('best_score'):.4f}")

        except Exception as e:
            print(f"❌ Exception in Stage 4: {str(e)}")
            state['error'] = f"AutoML Stage failed: {str(e)}"
            state["agent_output"] = {
            "stage": "automl_training",
            "error": state["error"]
        }

        return state
    
    def visualize_graph(self, output_path="pipeline_graph.png"):
        """
        Generates a PNG image of the LangGraph workflow.
        """
        try:
            # Generate the graph image using the Mermaid format or ASCII
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
    # Starting state
    # inputs = {
    #     "data_path": "assets/data/Datasets/Classification Datasets/Titanic-Dataset.csv",
    #     "target_column": "Survived",
    #     "task_type": "classification"
    # }   
    # inputs = {
    #     "data_path": "assets/data/Datasets/Classification Datasets/Iris.csv",
    #     "target_column": "Species",
    #     "task_type": "classification"
    # }   
    inputs = {
        "data_path": "assets/data/Datasets/Regression Datasets/car_prices.csv",
        "target_column": "sellingprice",
        "task_type": "regression"
    }   
    result = pipeline.workflow.invoke(inputs)
    print("\n🏁 Pipeline Finished. Best Model Metrics:", result['final_metrics'])

    # pipeline.visualize_graph()
    # # To get a string compatible with Mermaid live editors or frontend renderers
    # mermaid_config = pipeline.workflow.get_graph().draw_mermaid()
    # print(mermaid_config)