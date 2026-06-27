"""Shared LLM fallback helpers.

Provider order:
    1. Primary Gemma model
    2. Gemini fallback
    3. Qwen fallback
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Optional

import requests
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

PRIMARY_MODEL = os.getenv("PRIMARY_LLM_MODEL") or os.getenv("GEMMA_MODEL") or "gemma-4-31b-it"
QWEN_MODEL = os.getenv("QWEN_FALLBACK_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
HF_LEGACY_URL = f"https://api-inference.huggingface.co/models/{QWEN_MODEL}"
HF_TIMEOUT_SECONDS = 60
HF_MAX_TOKENS = 1200


def _terminal_log(message: str) -> None:
    """Print the required LLM fallback logs and also send them to logging."""
    print(message)
    logger.info(message)


def _raise_or_return_safe_default(
    safe_default: Optional[Any],
    primary_error: Optional[BaseException],
    gemini_error: Optional[BaseException],
    qwen_error: Optional[BaseException],
) -> Any:
    _terminal_log("All LLM providers failed")
    if safe_default is not None:
        return safe_default
    error = qwen_error or gemini_error or primary_error
    if error is not None:
        raise error
    raise RuntimeError("All LLM providers failed")


def _response_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return json.dumps(response, ensure_ascii=False)
    if isinstance(response, list):
        return "\n".join(_response_text(item) for item in response)
    content = getattr(response, "content", None)
    if content is None:
        return str(response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_response_text(item) for item in content)
    return str(content)


def _is_empty_response(response: Any) -> bool:
    text = _response_text(response).strip()
    return not text


def _prompt_to_text(prompt: Any) -> str:
    """Convert LangChain messages or a plain prompt into one text prompt."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: list[str] = []
        for message in prompt:
            content = getattr(message, "content", message)
            role = message.__class__.__name__.replace("Message", "")
            if role and role != "str":
                parts.append(f"{role}: {_response_text(content)}")
            else:
                parts.append(_response_text(content))
        return "\n\n".join(part for part in parts if part)
    return _response_text(prompt)


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Qwen response did not contain a JSON object")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Qwen JSON response must be an object")
    return parsed


def _schema_instruction(schema: Any) -> str:
    try:
        schema_json = schema.model_json_schema()
    except Exception:
        try:
            schema_json = schema.schema()
        except Exception:
            schema_json = str(schema)
    return (
        "Return ONLY valid JSON. Do not include markdown, code fences, or explanations.\n"
        "The JSON must match this schema:\n"
        f"{json.dumps(schema_json, ensure_ascii=False, default=str)}"
    )


def _parse_schema(schema: Any, text: str) -> Any:
    data = _extract_json_object(text)
    if hasattr(schema, "model_validate"):
        return schema.model_validate(data)
    if hasattr(schema, "parse_obj"):
        return schema.parse_obj(data)
    return data


def call_qwen_fallback(
    prompt: Any,
    *,
    temperature: float = 0.0,
    max_tokens: int = HF_MAX_TOKENS,
    timeout: int = HF_TIMEOUT_SECONDS,
    schema: Any = None,
) -> str:
    """Call Qwen through Hugging Face and return plain generated text."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not found")

    prompt_text = _prompt_to_text(prompt)
    if schema is not None:
        prompt_text = f"{_schema_instruction(schema)}\n\n{prompt_text}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    router_payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            HF_ROUTER_URL,
            headers=headers,
            json=router_payload,
            timeout=timeout,
        )
        if response.status_code == 200:
            data = response.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if str(text).strip():
                return str(text).strip()
            raise RuntimeError(f"Empty Hugging Face router response: {data}")
        router_error = f"HF router error {response.status_code}: {response.text[:500]}"
    except Exception as exc:
        router_error = str(exc)

    legacy_payload = {
        "inputs": prompt_text,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "return_full_text": False,
        },
    }
    response = requests.post(
        HF_LEGACY_URL,
        headers=headers,
        json=legacy_payload,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"{router_error}; HF legacy error {response.status_code}: {response.text[:500]}"
        )
    data = response.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        text = str(data[0]["generated_text"]).strip()
        if text:
            return text
    if isinstance(data, dict) and data.get("generated_text"):
        text = str(data["generated_text"]).strip()
        if text:
            return text
    raise RuntimeError(f"{router_error}; unexpected HF legacy response: {data}")


def call_text_llm_with_fallback(
    prompt: Any,
    primary_call: Callable[[Any], Any],
    gemini_call: Callable[[Any], Any],
    *,
    safe_default: Optional[Any] = None,
    temperature: float = 0.0,
    qwen_as_message: bool = True,
) -> Any:
    """Run primary Gemma, then Gemini, then Qwen if needed."""
    primary_error: Optional[BaseException] = None
    gemini_error: Optional[BaseException] = None
    qwen_error: Optional[BaseException] = None

    _terminal_log(f"Trying primary LLM: {PRIMARY_MODEL}")
    try:
        response = primary_call(prompt)
        if _is_empty_response(response):
            raise ValueError("Primary LLM returned an empty response")
        _terminal_log("Primary LLM succeeded")
        return response
    except Exception as exc:
        primary_error = exc
        _terminal_log(f"Primary LLM failed: {exc}")

    _terminal_log("Trying first fallback LLM: Gemini")
    try:
        response = gemini_call(prompt)
        if _is_empty_response(response):
            raise ValueError("Gemini returned an empty response")
        _terminal_log("Gemini fallback succeeded")
        return response
    except Exception as exc:
        gemini_error = exc
        _terminal_log(f"Gemini fallback failed: {exc}")

    _terminal_log("Trying second fallback LLM: Qwen")
    try:
        text = call_qwen_fallback(prompt, temperature=temperature)
        if not text.strip():
            raise ValueError("Qwen returned an empty response")
        _terminal_log("Qwen fallback succeeded")
        return AIMessage(content=text) if qwen_as_message else text
    except Exception as exc:
        qwen_error = exc
        _terminal_log(f"Qwen fallback failed: {exc}")
        return _raise_or_return_safe_default(
            safe_default, primary_error, gemini_error, qwen_error
        )


class MultiProviderFallbackLLM:
    """Small adapter preserving the LangChain .invoke/.with_structured_output shape."""

    def __init__(self, primary_llm: Any, gemini_llm: Any, *, temperature: float = 0.0):
        self.primary_llm = primary_llm
        self.gemini_llm = gemini_llm
        self.temperature = temperature

    def invoke(self, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        return call_text_llm_with_fallback(
            prompt,
            lambda value: self.primary_llm.invoke(value, *args, **kwargs),
            lambda value: self.gemini_llm.invoke(value, *args, **kwargs),
            temperature=self.temperature,
        )

    def with_structured_output(self, schema: Any, *args: Any, **kwargs: Any) -> "MultiProviderStructuredFallbackLLM":
        structured_primary = self.primary_llm.with_structured_output(
            schema, *args, **kwargs
        )
        structured_gemini = self.gemini_llm.with_structured_output(
            schema, *args, **kwargs
        )
        return MultiProviderStructuredFallbackLLM(
            structured_primary,
            structured_gemini,
            schema,
            temperature=self.temperature,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.gemini_llm, name)


class MultiProviderStructuredFallbackLLM:
    """Adapter for structured output with Gemma, Gemini, then Qwen JSON fallback."""

    def __init__(self, primary_llm: Any, gemini_llm: Any, schema: Any, *, temperature: float = 0.0):
        self.primary_llm = primary_llm
        self.gemini_llm = gemini_llm
        self.schema = schema
        self.temperature = temperature

    def invoke(self, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        primary_error: Optional[BaseException] = None
        gemini_error: Optional[BaseException] = None
        qwen_error: Optional[BaseException] = None

        _terminal_log(f"Trying primary LLM: {PRIMARY_MODEL}")
        try:
            response = self.primary_llm.invoke(prompt, *args, **kwargs)
            if response is None:
                raise ValueError("Primary LLM returned an empty response")
            _terminal_log("Primary LLM succeeded")
            return response
        except Exception as exc:
            primary_error = exc
            _terminal_log(f"Primary LLM failed: {exc}")

        _terminal_log("Trying first fallback LLM: Gemini")
        try:
            response = self.gemini_llm.invoke(prompt, *args, **kwargs)
            if response is None:
                raise ValueError("Gemini returned an empty response")
            _terminal_log("Gemini fallback succeeded")
            return response
        except Exception as exc:
            gemini_error = exc
            _terminal_log(f"Gemini fallback failed: {exc}")

        _terminal_log("Trying second fallback LLM: Qwen")
        try:
            text = call_qwen_fallback(
                prompt,
                temperature=self.temperature,
                schema=self.schema,
            )
            if not text.strip():
                raise ValueError("Qwen returned an empty response")
            parsed = _parse_schema(self.schema, text)
            _terminal_log("Qwen fallback succeeded")
            return parsed
        except Exception as exc:
            qwen_error = exc
            _terminal_log(f"Qwen fallback failed: {exc}")
            return _raise_or_return_safe_default(
                None, primary_error, gemini_error, qwen_error
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.gemini_llm, name)


class GeminiQwenFallbackLLM(MultiProviderFallbackLLM):
    """Compatibility wrapper for older Gemini-first imports."""

    def __init__(self, gemini_llm: Any, *, temperature: float = 0.0):
        super().__init__(gemini_llm, gemini_llm, temperature=temperature)


class GeminiQwenStructuredFallbackLLM(MultiProviderStructuredFallbackLLM):
    """Compatibility wrapper for older Gemini-first structured imports."""

    def __init__(self, gemini_llm: Any, schema: Any, *, temperature: float = 0.0):
        super().__init__(gemini_llm, gemini_llm, schema, temperature=temperature)
