from __future__ import annotations

from typing import Any, Dict, Type

from pydantic import BaseModel

from .llm import get_extractor_model


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
        result = extractor.invoke(
            [
                ("system", system_prompt),
                ("human", prompt),
            ]
        )
        payload = result.model_dump(exclude_none=True)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
