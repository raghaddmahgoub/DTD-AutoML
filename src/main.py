"""
Main entry point for the AutoML Agent.
"""
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.automl_agent.automl_agent import AutoMLAgent
from src.utils.logger import Logger

logger = Logger()


def main():
    """Main function to run the AutoML agent."""
    # Example usage
    data_path = "assets/data/Datasets/Classification Datasets/Titanic-Dataset.csv"
    
    # Check if file exists
    if not os.path.exists(data_path):
        logger.warn(f"Data file not found: {data_path}")
        logger.info("Please provide a valid data path.")
        logger.info("Example datasets available in assets/data/Datasets/")
        return
    
    logger.info("=" * 60)
    logger.info("AutoML Agent - Model Selection and Training")
    logger.info("=" * 60)
    
    # Initialize agent
    agent = AutoMLAgent()
    
    # Run the workflow
    # You can specify target_column or let it auto-detect
    results = agent.run(
        data_path=data_path,
        target_column=None  # Auto-detect target column
    )
    
    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("WORKFLOW RESULTS")
    logger.info("=" * 60)
    
    if results.get('error'):
        logger.error(f"Error: {results['error']}")
        return
    
    logger.info(f"Step: {results.get('step', 'unknown')}")
    logger.info(f"Target Column: {results.get('target_column', 'N/A')}")
    logger.info(f"Problem Type: {results.get('problem_type', 'N/A')}")
    
    if results.get('selected_models'):
        logger.info(f"Selected Models: {', '.join(results['selected_models'])}")
    
    if results.get('reasoning'):
        logger.info(f"\nReasoning:\n{results['reasoning']}")
    
    if results.get('model_metrics'):
        metrics = results['model_metrics']
        logger.info(f"\nModel Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Workflow completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
