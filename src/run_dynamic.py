import os
import sys
from pathlib import Path
from dotenv import load_dotenv 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.dynamic.controller_agent.controller_agent import ControllerAgent
from src.utils.logger import Logger
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv(PROJECT_ROOT / ".env")

#tools
from tools.registry import ToolRegistry

from tools.data_understanding import data_understanding
from tools.data_cleaning import data_cleaning
from tools.feature_engineering import feature_engineering
from tools.plan_training import plan_training
from tools.train_simple import train_simple
from tools.train_simple_optuna import train_simple_optuna
from tools.train_autogluon import train_autogluon
from tools.evaluate import evaluate

tool_registry = ToolRegistry()
tool_registry.register("data_understanding", data_understanding)
tool_registry.register("data_cleaning", data_cleaning)
tool_registry.register("feature_engineering", feature_engineering)
tool_registry.register("plan_training", plan_training)
tool_registry.register("train_simple", train_simple)
tool_registry.register("train_simple_optuna", train_simple_optuna)
tool_registry.register("train_autogluon", train_autogluon)
tool_registry.register("evaluate", evaluate)

def run():
    logger = Logger()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )
    data = str(PROJECT_ROOT / "assets/data/Classification Datasets/Titanic-Dataset.csv")
    controller = ControllerAgent(logger, llm, tool_registry)
    controller.run(data,"Analyze the dataset and train a model to predict the target variable.")



if __name__ == "__main__":
    run()