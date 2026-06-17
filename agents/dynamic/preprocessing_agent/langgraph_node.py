"""
LangGraph Node for Preprocessing Agent

This is a comprehensive, reusable LangGraph node that can be added to any StateGraph.
It handles preprocessing execution with proper state threading.

INCLUDES:
- Node creation functions (create, simple, configured)
- 3 working examples (simple, multi-step, with retry)
- Full documentation and usage examples

USAGE:
    from langgraph.graph import StateGraph
    from agents.dynamic.preprocessing_agent.langgraph_node import create_preprocessing_node

    graph = StateGraph(YourState)
    graph.add_node("preprocessing", create_preprocessing_node(llm, registry))
    graph.set_entry_point("preprocessing")
    graph.add_edge("preprocessing", END)

EXAMPLES:
    # Run all 3 examples:
    python agents/dynamic/preprocessing_agent/langgraph_node.py
"""

from __future__ import annotations
from agents.dynamic.preprocessing_agent.tool_runner import invoke_tool
from agents.dynamic.preprocessing_agent.state import PreprocessingAgentState
from langgraph.graph import StateGraph, END

import sys
from pathlib import Path
from typing import Any, Callable

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ============================================================================
# NODE CREATION FUNCTIONS
# ============================================================================

def create_preprocessing_node(
    llm: Any,
    registry: Any,
    config: dict | None = None,
) -> Callable[[PreprocessingAgentState], PreprocessingAgentState]:
    """
    Create a reusable LangGraph node for preprocessing.

    This node:
    1. Retrieves the preprocessing_execution tool from registry
    2. Executes preprocessing on the input data
    3. Updates state with results and metadata

    Args:
        llm: Language model instance (GoogleGenerativeAI, etc.)
        registry: ToolRegistry instance with "preprocessing_execution" registered
        config: Optional configuration dict with:
            - preprocessing_input: dict with tool parameters
            - target_column: str (optional)
            - test_size: float (optional, default 0.2)
            - use_llm: bool (optional, default True)

    Returns:
        Node function: (state) -> state
        Compatible with: langgraph.graph.StateGraph

    Raises:
        RuntimeError: If preprocessing_execution tool is not registered

    Example:
        >>> from langgraph.graph import StateGraph
        >>> graph = StateGraph(PreprocessingAgentState)
        >>> node_func = create_preprocessing_node(llm, registry)
        >>> graph.add_node("preprocessing", node_func)
    """
    preprocessing_tool = registry.get("preprocessing_execution")
    if preprocessing_tool is None:
        raise RuntimeError(
            "preprocessing_execution tool is not registered in ToolRegistry. "
            "Register it with: registry.register('preprocessing_execution', preprocessing_execution)"
        )

    config = config or {}

    def preprocessing_node(state: PreprocessingAgentState) -> PreprocessingAgentState:
        """
        Execute preprocessing and update state.

        Flow:
        1. Extract current pipeline_state from input state
        2. Build tool_input from config
        3. Call preprocessing_execution tool
        4. Update state with results
        5. Return updated state

        Args:
            state: PreprocessingAgentState with:
                - data_path: str
                - prompt: str
                - task: str
                - pipeline_state: dict
                - step: str

        Returns:
            Updated PreprocessingAgentState with:
                - pipeline_state: Updated with preprocessing results
                - last_tool: "preprocessing_execution"
                - last_result: Tool result dict
                - step: Current step
                - error: Error message if failed
        """
        # Get current pipeline state
        pipeline_state = state.get("pipeline_state") or {}

        # Build tool input from config
        tool_input = dict(config.get("preprocessing_input") or {})

        # Apply config overrides (these take precedence)
        if config.get("target_column"):
            tool_input["target_column"] = config["target_column"]
        if config.get("test_size") is not None:
            tool_input["test_size"] = config["test_size"]
        if config.get("use_llm") is not None:
            tool_input["use_llm"] = config["use_llm"]

        # Set defaults if not specified
        if "test_size" not in tool_input:
            tool_input["test_size"] = 0.2
        if "use_llm" not in tool_input:
            tool_input["use_llm"] = True

        # Invoke the preprocessing tool
        result, updated_pipeline_state = invoke_tool(
            preprocessing_tool,
            task=state.get("task") or "Execute preprocessing pipeline",
            tool_input=tool_input,
            prompt=state.get("prompt", ""),
            data_path=state.get(
                "data_path", pipeline_state.get("data_path", "")),
            llm=llm,
            pipeline_state=pipeline_state,
        )

        # Build updated state
        updated_state: PreprocessingAgentState = {
            **state,
            "pipeline_state": updated_pipeline_state,
            "last_tool": "preprocessing_execution",
            "last_result": result,
            "step": updated_pipeline_state.get("step", "preprocessing_complete"),
        }

        # Track errors
        if result.get("status") == "error":
            updated_state["error"] = result.get(
                "error", "Unknown error in preprocessing")

        return updated_state

    return preprocessing_node


def simple_preprocessing_node(
    llm: Any,
    registry: Any,
) -> Callable[[PreprocessingAgentState], PreprocessingAgentState]:
    """
    Create a preprocessing node with default configuration.

    Useful for simple preprocessing without custom config.

    Args:
        llm: Language model instance
        registry: ToolRegistry instance

    Returns:
        Node function for LangGraph

    Example:
        >>> node = simple_preprocessing_node(llm, registry)
        >>> graph.add_node("preprocessing", node)
    """
    return create_preprocessing_node(llm, registry, config={})


def configured_preprocessing_node(
    llm: Any,
    registry: Any,
    target_column: str,
    test_size: float = 0.2,
    use_llm: bool = True,
) -> Callable[[PreprocessingAgentState], PreprocessingAgentState]:
    """
    Create a preprocessing node with predefined configuration.

    Useful when you want to lock in specific preprocessing parameters.

    Args:
        llm: Language model instance
        registry: ToolRegistry instance
        target_column: Column to predict
        test_size: Train/test split ratio (default 0.2)
        use_llm: Use LLM for policy generation (default True)

    Returns:
        Node function for LangGraph

    Example:
        >>> node = configured_preprocessing_node(
        ...     llm, registry,
        ...     target_column="Survived",
        ...     test_size=0.15,
        ...     use_llm=True,
        ... )
        >>> graph.add_node("preprocessing", node)
    """
    config = {
        "target_column": target_column,
        "test_size": test_size,
        "use_llm": use_llm,
        "preprocessing_input": {},
    }
    return create_preprocessing_node(llm, registry, config)


# ============================================================================
# EXAMPLE GRAPHS (3 working examples)
# ============================================================================

def build_simple_preprocessing_graph(llm, registry):
    """
    EXAMPLE 1: Simple preprocessing graph (just preprocessing, no other nodes)

    Graph structure:
        START → preprocessing → END
    """
    graph = StateGraph(PreprocessingAgentState)

    # Create the preprocessing node
    preprocessing_node = create_preprocessing_node(
        llm,
        registry,
        config={
            "target_column": "Survived",
            "test_size": 0.2,
            "use_llm": True,
        },
    )

    # Add node to graph
    graph.add_node("preprocessing", preprocessing_node)

    # Set entry point and edge
    graph.set_entry_point("preprocessing")
    graph.add_edge("preprocessing", END)

    # Compile and return
    return graph.compile()


def build_multi_step_pipeline(llm, registry):
    """
    EXAMPLE 2: Multi-step pipeline with preprocessing as one step

    Graph structure:
        START → validate_data → preprocessing → feature_engineering → END

    (feature_engineering and validate_data are stubs for demonstration)
    """
    graph = StateGraph(PreprocessingAgentState)

    # Validation node (example)
    def validate_data(state: PreprocessingAgentState) -> PreprocessingAgentState:
        """Quick validation that data exists."""
        data_path = state.get("data_path", "")
        if not data_path:
            state["error"] = "No data_path provided"
        return state

    # Preprocessing node
    preprocessing_node = create_preprocessing_node(
        llm,
        registry,
        config={"target_column": "Survived", "test_size": 0.2},
    )

    # Feature engineering stub (placeholder)
    def feature_engineering(state: PreprocessingAgentState) -> PreprocessingAgentState:
        """Placeholder for feature engineering."""
        state["step"] = "feature_engineering_complete"
        return state

    # Add nodes
    graph.add_node("validate", validate_data)
    graph.add_node("preprocessing", preprocessing_node)
    graph.add_node("feature_engineering", feature_engineering)

    # Set flow
    graph.set_entry_point("validate")
    graph.add_edge("validate", "preprocessing")
    graph.add_edge("preprocessing", "feature_engineering")
    graph.add_edge("feature_engineering", END)

    return graph.compile()


def build_preprocessing_with_retry(llm, registry):
    """
    EXAMPLE 3: Preprocessing with error handling and retry logic

    Graph structure:
        START → preprocessing → check_error → (success) → END
                                           → (error) → retry → preprocessing
    """
    graph = StateGraph(PreprocessingAgentState)

    # Preprocessing node
    preprocessing_node = create_preprocessing_node(
        llm,
        registry,
        config={"target_column": "Survived", "test_size": 0.2},
    )

    # Error checker
    def check_error(state: PreprocessingAgentState):
        """Route based on success or error."""
        if state.get("error"):
            return "retry"
        return "success"

    # Retry handler
    def retry_handler(state: PreprocessingAgentState) -> PreprocessingAgentState:
        """Handle retries - clear error and try again."""
        retry_count = state.get("retry_count", 0)
        if retry_count >= 2:
            state["error"] = "Max retries exceeded"
            return state
        state["retry_count"] = retry_count + 1
        state["error"] = None
        state["step"] = "retrying_preprocessing"
        return state

    # Add nodes
    graph.add_node("preprocessing", preprocessing_node)
    graph.add_node("check_error", check_error)
    graph.add_node("retry", retry_handler)

    # Set flow
    graph.set_entry_point("preprocessing")
    graph.add_edge("preprocessing", "check_error")
    graph.add_conditional_edges(
        "check_error",
        lambda x: x,  # Router function
        {
            "success": END,
            "retry": "preprocessing",
        },
    )

    return graph.compile()


# ============================================================================
# DEMO: Run all 3 examples
# ============================================================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI
    from src.utils.logger import Logger
    from tools.registry import ToolRegistry
    from tools.preprocessing_execution import preprocessing_execution

    # Load environment
    load_dotenv()

    # Setup
    logger = Logger()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )
    registry = ToolRegistry()
    registry.register("preprocessing_execution", preprocessing_execution)

    # Data path
    data_path = str(_project_root / "uploads/Titanic-Dataset.csv")

    print("\n" + "=" * 70)
    print("LANGGRAPH NODE - 3 WORKING EXAMPLES")
    print("=" * 70)

    # Example 1: Simple preprocessing graph
    print("\n[Example 1] Simple Preprocessing Graph")
    print("-" * 70)
    try:
        graph1 = build_simple_preprocessing_graph(llm, registry)
        result1 = graph1.invoke(
            {
                "data_path": data_path,
                "prompt": "Preprocess Titanic dataset",
                "task": "Preprocessing",
                "step": "preprocessing_start",
            }
        )
        print(f"Status: {result1.get('status')}")
        print(f"Step: {result1.get('step')}")
        if result1.get("error"):
            print(f"Error: {result1.get('error')}")
    except Exception as e:
        print(f"Error: {str(e)}")

    # Example 2: Multi-step pipeline
    print("\n[Example 2] Multi-Step Pipeline")
    print("-" * 70)
    try:
        graph2 = build_multi_step_pipeline(llm, registry)
        result2 = graph2.invoke(
            {
                "data_path": data_path,
                "prompt": "Full preprocessing pipeline",
                "task": "Full pipeline",
                "step": "validate_start",
            }
        )
        print(f"Final Step: {result2.get('step')}")
        print(f"Status: {result2.get('status')}")
    except Exception as e:
        print(f"Error: {str(e)}")

    # Example 3: With retry logic
    print("\n[Example 3] Preprocessing with Retry Logic")
    print("-" * 70)
    try:
        graph3 = build_preprocessing_with_retry(llm, registry)
        result3 = graph3.invoke(
            {
                "data_path": data_path,
                "prompt": "Preprocessing with error handling",
                "task": "Preprocessing with retry",
                "step": "preprocessing_start",
            }
        )
        print(f"Final Step: {result3.get('step')}")
        print(f"Status: {result3.get('status')}")
        if result3.get("error"):
            print(f"Error: {result3.get('error')}")
    except Exception as e:
        print(f"Error: {str(e)}")

    print("\n" + "=" * 70 + "\n")
