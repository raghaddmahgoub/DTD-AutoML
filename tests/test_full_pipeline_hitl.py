"""
Test file for running the full LangGraph pipeline with HITL (Human-in-the-Loop).
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Setup project root import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from agents.dynamic.controller_agent.controller_agent import ControllerAgent
from src.utils.logger import Logger

def test_full_pipeline_hitl():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = Logger()
    
    # 1. Initialize ControllerAgent
    logger.info("Initializing ControllerAgent...")
    controller = ControllerAgent()
    
    # Data path (using Titanic dataset from uploads)
    data_path = str(PROJECT_ROOT / "uploads/Titanic-Dataset.csv")
    if not Path(data_path).exists():
        logger.error(f"Titanic dataset not found at {data_path}. Please place Titanic-Dataset.csv in uploads/")
        return

    # User request to trigger the full pipeline
    prompt = "Run full pipeline on Titanic dataset, predict Survived column, and use Simple training approach with RandomForest"
    run_id = f"test-hitl-{os.urandom(3).hex()}"
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Starting pipeline with prompt: '{prompt}' and run_id: '{run_id}'")
    logger.info("=" * 80 + "\n")
    
    state = controller.run({
        "data_path": data_path,
        "prompt": prompt,
        # "target_column": "Survived",
        "run_id": run_id,
    })
    
    # Step 2: Loop to resume the pipeline on every HITL checkpoint pause
    iteration = 0
    max_steps = 15
    
    while state.get("__interrupted__") and iteration < max_steps:
        paused_at = state.get("__paused_at__")
        logger.info("\n" + "-" * 80)
        logger.info(f"[PAUSED] [HITL Checkpoint] Pipeline paused at agent: '{paused_at}'")
        logger.info(f"Checkpoint Outputs preview:")
        logger.info(str(state.get("agent_outputs", {}).get(paused_at, {}))[:300] + "...")
        logger.info("-" * 80)
        
        # Test simulated feedback for a specific agent (e.g. preprocessing) to show feedback loops,
        # and then accept on the second try. Otherwise accept.
        decision = "accept"
        feedback = ""
        
        if paused_at == "preprocessing" and not any(h.get("agent") == "preprocessing" for h in state.get("feedback_history", [])):
            decision = "feedback"
            feedback = "Please impute missing categorical columns with mode."
            logger.info(f"[HITL] [Simulated HITL] Injecting feedback loop back to '{paused_at}': '{feedback}'")
        else:
            logger.info(f"[HITL] [Simulated HITL] Resuming with decision: 'accept'")
            
        # Resume the pipeline
        state = controller.resume(
            run_id=run_id,
            decision=decision,
            feedback_text=feedback,
        )
        iteration += 1
        
    logger.info("\n" + "=" * 80)
    if state.get("__error__"):
        logger.error(f"[ERROR] Pipeline failed with error: {state.get('__error__')}")
    elif state.get("error"):
        logger.error(f"[ERROR] Pipeline failed with internal state error: {state.get('error')}")
    else:
        logger.info("[OK] Pipeline completed successfully!")
        logger.info(f"   Target column: {state.get('target_column')}")
        logger.info(f"   Task type:     {state.get('task_type')}")
        logger.info(f"   Trained model: {state.get('trained_model_path') or 'N/A'}")
        logger.info(f"   Saved files:   {list((state.get('saved_files') or {}).keys())}")
    logger.info("=" * 80 + "\n")

if __name__ == "__main__":
    test_full_pipeline_hitl()
