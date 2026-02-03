import os
import pandas as pd
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Import modular agents from your folders
from agents.eda_agent.eda_agent2 import EDAAgent
# from agents.preprocessing_agent.preprocessor import PreprocessingAgent
from agents.automl_agent.automl_agent import AutoMLAgent

load_dotenv()

class AgentState(TypedDict):
    """Shared pipeline memory."""
    data_path: str
    target_column: str
    task_type: str
    analysis_report_path: Optional[str]
    automl_directives: Optional[dict]
    final_metrics: Optional[dict]

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

        return builder.compile()

    # --- Node Implementations ---

    def stage_raw_analysis(self, state: AgentState):
        """First Analysis: Identify issues for the Preprocessor."""
        print("🔍 [Stage 1] Running Raw Data Analysis...")
        df = pd.read_csv(state['data_path'])
        agent = EDAAgent(df, target_column=state['target_column'], df_name="raw_data")
        
        agent.run(run_type="raw") # Internal route for raw analysis
        results = agent.export(output_dir="Output/raw")
        
        print(f"✅ Raw analysis complete. Report: {results['report_path']}")
        return state

    def stage_preprocessing(self, state: AgentState):
        """Preprocessing: Clean data based on Raw Analysis."""
        print("🛠️ [Stage 2] Running Preprocessing Agent...")
        # Simulate preprocessing output
        clean_path = state['data_path'].replace(".csv", "_clean.csv")
        state['clean_data_path'] = clean_path 
        return state

    def stage_clean_analysis(self, state: AgentState):
        """Second Analysis: Generate Directives for AutoML."""
        print("📊 [Stage 3] Running Post-Prep Analysis...")
        df = pd.read_csv(state['clean_data_path'])
        agent = EDAAgent(df, target_column=state['target_column'], df_name="clean_data")
        
        agent.run(run_type="clean") # Internal route for clean analysis
        results = agent.export(output_dir="Output/clean")
        
        # Save the reformatted JSON directives for the AutoML agent
        state['automl_directives'] = results.get("automl_context")
        return state

    def stage_automl(self, state: AgentState):
        """Final Stage: Model Selection & Training."""
        print("🤖 [Stage 4] Running AutoML Training...")
        directives = state['automl_directives']
        
        # Accessing the meaningful reformatted keys we created
        print(f"Targeting: {directives['target']['column']} for {state['task_type']}")
        print(f"Applying Encoding Hints: {directives['encoding_hints']}")
        
        state['final_metrics'] = {"best_model": "XGBoost", "score": 0.89}
        return state

# --- Main Execution ---
if __name__ == "__main__":
    pipeline = DTDPipeline()
    
    # Starting state
    inputs = {
        "data_path": "assets/data/Datasets/Classification Datasets/Iris.csv",
        "target_column": "Species",
        "task_type": "classification"
    }
    
    result = pipeline.workflow.invoke(inputs)
    print("\n🏁 Pipeline Finished. Best Model Metrics:", result['final_metrics'])