from __future__ import annotations

import time
from typing import Any, Dict, Type

from pydantic import BaseModel

from .llm import get_extractor_model
from ..observability.logging_utils import log_event, summarize_text
from ..observability.llm_usage import (
    apply_span_attributes,
    build_llm_input_token_attrs,
    build_llm_output_token_attrs,
    log_llm_error,
    log_llm_request,
    log_llm_response,
)
from ..observability.otel import record_exception, start_span


def llm_structured_extract(
    prompt: str, *, schema: Type[BaseModel], system_prompt: str
) -> Dict[str, Any]:
    if not prompt:
        return {}
    try:
        llm = get_extractor_model()
    except Exception as exc:
        log_event(
            "llm_extract_model_error",
            error=str(exc),
            prompt_summary=summarize_text(prompt),
            schema=schema.__name__,
        )
        return {}
    try:
        extractor = llm.with_structured_output(schema)
        log_event(
            "llm_extract_call",
            prompt=prompt,
            system_prompt=system_prompt,
            schema=schema.__name__,
        )
        log_llm_request(
            f"structured_extract:{schema.__name__}",
            model=llm,
            system_prompt=system_prompt,
            user_prompt=prompt,
        )
        started_at = time.perf_counter()
        span_attrs = build_llm_input_token_attrs(
            llm, system_prompt=system_prompt, user_prompt=prompt
        )
        with start_span("llm.extract", attributes=span_attrs) as span:
            try:
                result = extractor.invoke(
                    [
                        ("system", system_prompt),
                        ("human", prompt),
                    ]
                )
            except Exception as exc:
                log_llm_error(
                    f"structured_extract:{schema.__name__}",
                    model=llm,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    error=exc,
                    latency_ms=(time.perf_counter() - started_at) * 1000.0,
                )
                record_exception(span, exc)
                raise
            apply_span_attributes(
                span, build_llm_output_token_attrs(result)
            )
        payload = result.model_dump(exclude_none=True)
        log_llm_response(
            f"structured_extract:{schema.__name__}",
            model=llm,
            result=result,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            response_text=payload,
        )
        log_event(
            "llm_extract_response",
            response_summary=summarize_text(payload),
            response_keys=sorted(payload.keys()) if isinstance(payload, dict) else [],
        )
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        log_event(
            "llm_extract_error",
            error=str(exc),
            prompt_summary=summarize_text(prompt),
            schema=schema.__name__,
        )
        return {}
