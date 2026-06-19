from tools.evaluate import evaluate
from tools.train_autogluon import train_autogluon
from tools.train_simple_optuna import train_simple_optuna
from tools.train_simple import train_simple
from tools.preprocessing_execution import preprocessing_execution
from tools.feature_engineering_execution import feature_engineering_execution
from tools.plan_training import plan_training
from tools.registry import ToolRegistry

from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils.logger import Logger
from agents.dynamic.controller_agent.controller_agent import ControllerAgent

import os
from dotenv import load_dotenv

load_dotenv()

tool_registry = ToolRegistry()
tool_registry.register("preprocessing_execution", preprocessing_execution)
tool_registry.register("feature_engineering_execution", feature_engineering_execution)
tool_registry.register("plan_training", plan_training)
tool_registry.register("train_simple", train_simple)
tool_registry.register("train_simple_optuna", train_simple_optuna)
tool_registry.register("train_autogluon", train_autogluon)
tool_registry.register("evaluate", evaluate)


def run_dynamic(inputs: dict):
    logger = Logger()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )

    controller = ControllerAgent(
        logger=logger,
        llm=llm,
        registry=tool_registry,
    )

    return controller.run(inputs)