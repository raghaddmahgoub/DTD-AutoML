from utils.logger_config import logger
from utils.reasoning_client import ReasoningClient

class AutoMLAgent:
    def __init__(self):
        self.reasoner = ReasoningClient()

    def analyze_dataset(self, dataset_path: str):
        logger.info(f"Starting analysis for dataset: {dataset_path}")
        # Example prompt
        prompt = f"Suggest preprocessing methods for dataset located at {dataset_path}"
        response = self.reasoner.ask(prompt)
        logger.info(f"Reasoning response: {response}")
        return response
