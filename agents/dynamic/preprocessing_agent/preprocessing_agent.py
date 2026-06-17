"""PreprocessingAgent — LangGraph orchestrator for preprocessing pipeline."""
from __future__ import annotations
from tools.pipeline_state import empty_state
from agents.dynamic.preprocessing_agent.state import PreprocessingAgentState
from agents.dynamic.preprocessing_agent.graph import build_preprocessing_graph

import sys
from pathlib import Path
from typing import Any

# Add project root to Python path so imports work from anywhere
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class PreprocessingAgent:
    """
    LangGraph agent that runs the preprocessing workflow by calling the tools layer:
      preprocessing_execution → handles cleaning, scaling, encoding, imbalance correction
    """

    def __init__(self, logger: Any, llm: Any, registry: Any):
        self.logger = logger
        self.llm = llm
        self.registry = registry

    def run(
        self,
        data_path: str,
        prompt: str,
        pipeline_state: dict | None = None,
        *,
        task: str = "Execute preprocessing pipeline",
        target_column: str = "",
        test_size: float = 0.2,
        use_llm: bool = True,
        preprocessing_input: dict | None = None,
    ) -> dict:
        """
        Execute the preprocessing workflow.

        Args:
            data_path: Path to input dataset CSV
            prompt: NL description of preprocessing needs (optional)
            pipeline_state: Existing pipeline state dict (carries forward data_path, target, etc.)
            task: Task description for logging
            target_column: Column to predict (will be added to preprocessing_input)
            test_size: Train/test split ratio
            use_llm: Whether to use LLM for preprocessing policy decisions
            preprocessing_input: Additional tool input parameters

        Returns:
            Updated pipeline_state dict with:
            - X_train_path, X_test_path, y_train_path, y_test_path
            - preprocessing_output: full results from preprocessing tool
            - status: success/error
            - error: error message if failed
        """
        config = {
            "target_column": target_column,
            "test_size": test_size,
            "use_llm": use_llm,
            "preprocessing_input": preprocessing_input or {},
        }

        graph = build_preprocessing_graph(self.llm, self.registry, config)
        initial: PreprocessingAgentState = {
            "data_path": data_path,
            "prompt": prompt,
            "task": task,
            "pipeline_state": pipeline_state or empty_state(data_path, prompt),
            "step": "preprocessing_agent_start",
        }

        self.logger.info("\n" + "=" * 50)
        self.logger.info("PREPROCESSING AGENT (LangGraph)")
        self.logger.info("=" * 50)

        final_state: PreprocessingAgentState = graph.invoke(initial)
        pipeline_state = final_state.get(
            "pipeline_state") or initial["pipeline_state"]

        if final_state.get("error"):
            self.logger.warning(
                f"PreprocessingAgent finished with error: {final_state['error']}")
        else:
            self.logger.info(
                f"PreprocessingAgent finished — step={pipeline_state.get('step')} "
                f"status={pipeline_state.get('status')}"
            )

        return pipeline_state


if __name__ == "__main__":
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI
    from src.utils.logger import Logger
    from tools.registry import ToolRegistry
    from tools.preprocessing_execution import preprocessing_execution

    # Load env FIRST (before reading any config)
    load_dotenv()

    # ======================================================================
    # CONFIGURATION BLOCK - Edit these settings to customize behavior
    # ======================================================================

    # Dataset Configuration
    DATASET_NAME = "Titanic-Dataset.csv"  # File in uploads/ folder
    DATASET_PATH = None  # Set to override default path, e.g., "uploads/your_data.csv"

    # Preprocessing Configuration
    TARGET_COLUMN = "Survived"  # Column to predict
    TEST_SIZE = 0.2  # Train/test split ratio (0.2 = 80/20)
    # Use LLM for preprocessing policy (True) or default policy (False)
    USE_LLM = True

    # LLM Configuration
    LLM_MODEL = "gemini-2.5-flash"  # Model name
    LLM_TEMPERATURE = 0.3  # Creativity (0.0 = deterministic, 1.0 = creative)
    LLM_API_KEY = os.getenv("GOOGLE_API_KEY")  # From .env file

    # Task Configuration
    TASK_NAME = "Preprocess Titanic dataset"
    TASK_PROMPT = "Clean and preprocess the Titanic dataset for classification"

    # Output Configuration
    VERBOSE = True  # Print detailed results

    # ======================================================================
    # END OF CONFIGURATION BLOCK
    # ======================================================================

    # Setup
    logger = Logger()
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=LLM_API_KEY,
        temperature=LLM_TEMPERATURE,
    )
    registry = ToolRegistry()
    registry.register("preprocessing_execution", preprocessing_execution)

    # Determine data path
    if DATASET_PATH:
        data_path = DATASET_PATH
    else:
        data_path = str(
            Path(__file__).resolve().parent.parent.parent.parent
            / f"uploads/{DATASET_NAME}"
        )

    if not Path(data_path).exists():
        print(f"Dataset not found: {data_path}")
        exit(1)

    agent = PreprocessingAgent(logger, llm, registry)
    result = agent.run(
        data_path=data_path,
        prompt=TASK_PROMPT,
        task=TASK_NAME,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        use_llm=USE_LLM,
    )

    # Display results
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Status: {result.get('status')}")
    if result.get('status') == 'success':
        output = result.get("preprocessing_output", {})
        print(f"X_train: {output.get('X_train_path')}")
        print(f"X_test:  {output.get('X_test_path')}")
        print(f"y_train: {output.get('y_train_path')}")
        print(f"y_test:  {output.get('y_test_path')}")
    else:
        print(f"Error: {result.get('error')}")
    print("=" * 70 + "\n")
