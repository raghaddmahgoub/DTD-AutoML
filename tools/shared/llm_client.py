"""
tools/llm_client.py
D.T.D (Data To Deployment) — Multi-Agent AutoML Pipeline

Tool: LLM Client Factory
Responsibility:
    Single place that constructs and returns a configured
    ChatGoogleGenerativeAI instance (Gemini 2.5 Flash).

    All agents call get_llm() so model name, API key handling,
    and defaults are never duplicated across agent files.

Consumers:
    - agents/intent_detector.py
    - agents/model_selection_agent.py
    - agents/feature_engineering_agent.py
    - agents/evaluation_agent.py
    - agents/deployment_agent.py
"""

import os
import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-2.5-flash-lite"


def get_llm(
    model_name: str = _DEFAULT_MODEL,
    temperature: float = 0.0,
    google_api_key: Optional[str] = None,
) -> ChatGoogleGenerativeAI:
    """
    Construct and return a ChatGoogleGenerativeAI instance.

    API key resolution order:
        1. Explicit google_api_key argument
        2. GOOGLE_API_KEY environment variable

    Args:
        model_name:     Gemini model string. Defaults to "gemini-2.5-flash-lite".
        temperature:    0.0 = deterministic (default for all structured calls).
        google_api_key: Optional explicit key (overrides env var).

    Returns:
        ChatGoogleGenerativeAI instance.
        Bind structured output with:
            llm = get_llm().with_structured_output(MyPydanticModel)

    Raises:
        EnvironmentError: if no API key found.
    """
    api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not found.\n"
            "  export GOOGLE_API_KEY='your-key'  — or —\n"
            "  get_llm(google_api_key='your-key')"
        )

    logger.debug("[LLMClient] Building %s (temp=%.1f)", model_name, temperature)

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
    )