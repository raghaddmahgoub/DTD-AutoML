"""
tools/training/
───────────────
All @tool-decorated training and evaluation tools.
Each tool uses the standard call signature:

    tool.invoke({
        "task": str,
        "tool_input": dict,
        "prompt": str,
        "data_path": str,
        "llm": ChatGoogleGenerativeAI,
        "state": dict | None,
    })

Exports:
    plan_training        — LLM-driven model selection + plan generation
    train_simple         — Scikit-learn train with default hyperparameters
    train_simple_optuna  — Scikit-learn train + Optuna HPO
    train_autogluon      — AutoGluon automated training
    evaluate             — Model evaluation (metrics + artefacts)
"""

from .plan_training import plan_training
from .train_simple import train_simple
from .train_simple_optuna import train_simple_optuna
from .train_autogluon import train_autogluon
from .evaluate import evaluate

__all__ = [
    "plan_training",
    "train_simple",
    "train_simple_optuna",
    "train_autogluon",
    "evaluate",
]
