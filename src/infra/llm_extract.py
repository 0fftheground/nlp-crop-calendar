from __future__ import annotations

from typing import Any, Dict, Type

from pydantic import BaseModel

from .llm import get_extractor_model
from ..observability.logging_utils import log_event, summarize_text


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
        result = extractor.invoke(
            [
                ("system", system_prompt),
                ("human", prompt),
            ]
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
