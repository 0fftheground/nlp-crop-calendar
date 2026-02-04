from __future__ import annotations

from typing import Any, Dict, Type

from pydantic import BaseModel

from .llm import get_extractor_model
from ..observability.logging_utils import log_event, summarize_text
from ..observability.llm_usage import (
    apply_span_attributes,
    build_llm_input_token_attrs,
    build_llm_output_token_attrs,
)
from ..observability.otel import record_exception, start_span


def llm_structured_extract(
    prompt: str, *, schema: Type[BaseModel], system_prompt: str
) -> Dict[str, Any]:
    if not prompt:
        return {}
    try:
        llm = get_extractor_model()
    except Exception:
        return {}
    try:
        extractor = llm.with_structured_output(schema)
        log_event(
            "llm_extract_call",
            prompt=prompt,
            system_prompt=system_prompt,
            schema=schema.__name__,
        )
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
                record_exception(span, exc)
                raise
            apply_span_attributes(
                span, build_llm_output_token_attrs(result)
            )
        payload = result.model_dump(exclude_none=True)
        log_event(
            "llm_extract_response",
            response_summary=summarize_text(payload),
            response_keys=sorted(payload.keys()) if isinstance(payload, dict) else [],
        )
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
