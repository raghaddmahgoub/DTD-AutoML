import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.automl_agent.automl_agent import AutoMLAgent
from src.utils.logger import Logger

logger = Logger()


def run():
    data_path = PROJECT_ROOT  / "assets/data/Classification Datasets/iris.csv"

    if not os.path.exists(data_path):
        logger.warn(f"Data file not found: {data_path}")
        return

    logger.info("=" * 60)
    logger.info("DETERMINISTIC MODE")
    logger.info("=" * 60)

    agent = AutoMLAgent()

    results = agent.run(
        data_path=data_path,
        target_column=None
    )

    # inline printing (no utils)
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    if results.get("error"):
        logger.error(results["error"])
        return

    logger.info(f"Target: {results.get('target_column')}")
    logger.info(f"Problem: {results.get('problem_type')}")

    if results.get("model_metrics"):
        for k, v in results["model_metrics"].items():
            logger.info(f"{k}: {v}")


if __name__ == "__main__":
    run()