from __future__ import annotations

from typing import Dict, Iterable, Optional

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

from .logging_utils import log_event, summarize_text


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return ""


def _safe_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _safe_get_num_tokens(model: object, text: str) -> Optional[int]:
    if not model:
        return None
    try:
        return int(model.get_num_tokens(text))
    except Exception:
        return None


def _safe_get_num_tokens_from_messages(
    model: object, messages: Iterable[BaseMessage]
) -> Optional[int]:
    if not model:
        return None
    fn = getattr(model, "get_num_tokens_from_messages", None)
    if not callable(fn):
        return None
    try:
        return int(fn(list(messages)))
    except Exception:
        return None


def _get_model_name(model: object) -> Optional[str]:
    for attr in ("model_name", "model", "model_id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_llm_input_token_attrs(
    model: object,
    *,
    system_prompt: object,
    user_prompt: object,
) -> Dict[str, object]:
    system_text = _normalize_text(system_prompt)
    user_text = _normalize_text(user_prompt)
    attrs: Dict[str, object] = {}
    model_name = _get_model_name(model)
    if model_name:
        attrs["llm.model"] = model_name

    system_tokens: Optional[int]
    user_tokens: Optional[int]
    if system_text:
        system_tokens = _safe_get_num_tokens(model, system_text)
    else:
        system_tokens = 0
    if user_text:
        user_tokens = _safe_get_num_tokens(model, user_text)
    else:
        user_tokens = 0

    if system_tokens is not None:
        attrs["llm.input_tokens.system"] = system_tokens
    if user_tokens is not None:
        attrs["llm.input_tokens.user"] = user_tokens
    if system_tokens is not None and user_tokens is not None:
        attrs["llm.input_tokens.total_text"] = system_tokens + user_tokens

    messages: list[BaseMessage] = []
    if system_text:
        messages.append(SystemMessage(content=system_text))
    if user_text:
        messages.append(HumanMessage(content=user_text))
    total_messages = _safe_get_num_tokens_from_messages(model, messages)
    if total_messages is not None:
        attrs["llm.input_tokens.total_messages"] = total_messages
    return attrs


def _extract_usage_metadata(result: object) -> Optional[dict]:
    usage = getattr(result, "usage_metadata", None)
    if isinstance(usage, dict):
        return usage
    response_metadata = getattr(result, "response_metadata", None)
    if isinstance(response_metadata, dict):
        usage = response_metadata.get("usage") or response_metadata.get("token_usage")
        if isinstance(usage, dict):
            return usage
    return None


def build_llm_output_token_attrs(result: object) -> Dict[str, object]:
    attrs: Dict[str, object] = {}
    usage = _extract_usage_metadata(result)
    if not usage:
        return attrs
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    ):
        value = _safe_int(usage.get(key))
        if value is not None:
            attrs[f"llm.usage.{key}"] = value
    return attrs


def build_llm_request_log_fields(
    model: object,
    *,
    system_prompt: object,
    user_prompt: object,
) -> Dict[str, object]:
    attrs = build_llm_input_token_attrs(
        model, system_prompt=system_prompt, user_prompt=user_prompt
    )
    fields: Dict[str, object] = {
        "model": _get_model_name(model),
        "system_prompt_summary": summarize_text(_normalize_text(system_prompt), limit=800),
        "user_prompt_summary": summarize_text(_normalize_text(user_prompt), limit=1200),
    }
    for key in (
        "llm.input_tokens.system",
        "llm.input_tokens.user",
        "llm.input_tokens.total_text",
        "llm.input_tokens.total_messages",
        "llm.model",
    ):
        if key in attrs:
            fields[key] = attrs[key]
    return fields


def build_llm_response_log_fields(
    model: object,
    result: object,
    *,
    latency_ms: Optional[float] = None,
    response_text: object = None,
) -> Dict[str, object]:
    attrs = build_llm_output_token_attrs(result)
    fields: Dict[str, object] = {
        "model": _get_model_name(model),
    }
    if latency_ms is not None:
        fields["latency_ms"] = round(float(latency_ms), 2)
    if response_text is not None:
        fields["response_summary"] = summarize_text(
            _normalize_text(response_text), limit=1200
        )
    for key in (
        "llm.usage.prompt_tokens",
        "llm.usage.completion_tokens",
        "llm.usage.total_tokens",
        "llm.usage.input_tokens",
        "llm.usage.output_tokens",
    ):
        if key in attrs:
            fields[key] = attrs[key]
    return fields


def log_llm_request(
    operation: str,
    *,
    model: object,
    system_prompt: object,
    user_prompt: object,
) -> None:
    log_event(
        "llm_api_request",
        operation=operation,
        **build_llm_request_log_fields(
            model, system_prompt=system_prompt, user_prompt=user_prompt
        ),
    )


def log_llm_response(
    operation: str,
    *,
    model: object,
    result: object,
    latency_ms: Optional[float] = None,
    response_text: object = None,
) -> None:
    log_event(
        "llm_api_response",
        operation=operation,
        **build_llm_response_log_fields(
            model, result, latency_ms=latency_ms, response_text=response_text
        ),
    )


def log_llm_error(
    operation: str,
    *,
    model: object,
    system_prompt: object,
    user_prompt: object,
    error: object,
    latency_ms: Optional[float] = None,
) -> None:
    fields = build_llm_request_log_fields(
        model, system_prompt=system_prompt, user_prompt=user_prompt
    )
    if latency_ms is not None:
        fields["latency_ms"] = round(float(latency_ms), 2)
    fields["error"] = _normalize_text(error)
    log_event("llm_api_error", operation=operation, **fields)


def apply_span_attributes(span: object, attributes: Dict[str, object]) -> None:
    if not span or not attributes:
        return None
    for key, value in attributes.items():
        if value is None:
            continue
        try:
            span.set_attribute(key, value)
        except Exception:
            try:
                span.set_attribute(key, str(value))
            except Exception:
                pass
