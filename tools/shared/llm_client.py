"""
tools/llm_client.py
D.T.D (Data To Deployment) — Multi-Agent AutoML Pipeline

Tool: LLM Client Factory
Responsibility:
    Single place that constructs and returns a configured
    LangChain-compatible LLM wrapper.

    All agents call get_llm() so model name, API key handling,
    and defaults are never duplicated across agent files.

Consumers:
    - agents/intent_detector.py
    - agents/model_selection_agent.py
    - agents/feature_engineering_agent.py
    - agents/deployment_agent.py
"""

import os
import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from tools.shared.llm_fallback import MultiProviderFallbackLLM, PRIMARY_MODEL

logger = logging.getLogger(__name__)

_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


class _MissingGeminiLLM:
    """Placeholder that lets fallback logic handle missing Gemini credentials."""

    def __init__(self, reason: str):
        self.reason = reason

    def invoke(self, *args, **kwargs):
        raise EnvironmentError(self.reason)

    def with_structured_output(self, *args, **kwargs):
        return self


def get_llm(
    model_name: str = _DEFAULT_GEMINI_MODEL,
    temperature: float = 0.0,
    google_api_key: Optional[str] = None,
    primary_model_name: Optional[str] = None,
) -> MultiProviderFallbackLLM:
    """
    Construct and return the shared fallback LLM wrapper.

    API key resolution order:
        1. Explicit google_api_key argument
        2. GOOGLE_API_KEY environment variable

    Args:
        model_name:     Gemini fallback model string.
        temperature:    0.0 = deterministic (default for all structured calls).
        google_api_key: Optional explicit key (overrides env var).
        primary_model_name: Optional primary model override.

    Returns:
        LangChain-compatible LLM wrapper.
        Bind structured output with:
            llm = get_llm().with_structured_output(MyPydanticModel)

    Raises:
        EnvironmentError: if no API key found.
    """
    api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        reason = "GOOGLE_API_KEY not found."
        logger.warning(reason)
        missing_llm = _MissingGeminiLLM(reason)
        return MultiProviderFallbackLLM(
            missing_llm,
            missing_llm,
            temperature=temperature,
        )

    primary_model = (
        primary_model_name
        or os.getenv("PRIMARY_LLM_MODEL")
        or os.getenv("GEMMA_MODEL")
        or PRIMARY_MODEL
    )
    gemini_model = os.getenv("GEMINI_FALLBACK_MODEL") or model_name

    logger.debug(
        "[LLMClient] Building primary=%s fallback=%s (temp=%.1f)",
        primary_model,
        gemini_model,
        temperature,
    )

    primary_llm = ChatGoogleGenerativeAI(
        model=primary_model,
        temperature=temperature,
        google_api_key=api_key,
    )

    gemini_llm = ChatGoogleGenerativeAI(
        model=gemini_model,
        temperature=temperature,
        google_api_key=api_key,
    )

    return MultiProviderFallbackLLM(
        primary_llm,
        gemini_llm,
        temperature=temperature,
    )
