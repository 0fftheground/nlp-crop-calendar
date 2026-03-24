from __future__ import annotations

import os
import math
import json
from urllib.parse import quote
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import httpx
import yaml


def ensure_dataset_path(path: str) -> Path:
    dataset_path = Path(path)
    if dataset_path.is_absolute():
        return dataset_path
    return Path.cwd() / dataset_path


def load_yaml_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset must be a mapping: {path}")
    return payload


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump(mode="json", exclude_none=True))
    return value


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        jsonable = to_jsonable(value)
        if isinstance(jsonable, str):
            return jsonable
        return json.dumps(jsonable, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def get_model_name(model: object) -> Optional[str]:
    for attr in ("model_name", "model", "model_id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def estimate_tokens(model: object, value: Any) -> Optional[int]:
    if not model:
        return None
    text = normalize_text(value)
    try:
        return int(model.get_num_tokens(text))
    except Exception:
        return None


def estimate_message_tokens(
    model: object,
    *,
    system_prompt: Any,
    user_prompt: Any,
) -> Optional[int]:
    if not model:
        return None
    fn = getattr(model, "get_num_tokens_from_messages", None)
    if not callable(fn):
        return None
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = []
    system_text = normalize_text(system_prompt)
    user_text = normalize_text(user_prompt)
    if system_text:
        messages.append(SystemMessage(content=system_text))
    if user_text:
        messages.append(HumanMessage(content=user_text))
    try:
        return int(fn(messages))
    except Exception:
        return None


def pick_token_estimate(*values: Optional[int]) -> Optional[int]:
    for value in values:
        if value is not None:
            return int(value)
    return None


def percentile(values: Iterable[float], pct: float) -> float:
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return 0.0
    if len(numbers) == 1:
        return round(numbers[0], 2)
    rank = max(0, min(len(numbers) - 1, math.ceil((pct / 100.0) * len(numbers)) - 1))
    return round(numbers[rank], 2)


def resolve_openai_base_url(base_url: Optional[str]) -> str:
    base = (base_url or "https://api.openai.com/v1").strip()
    return base.rstrip("/")


def validate_openai_model_available(
    *,
    model_name: str,
    api_key: Optional[str],
    base_url: Optional[str],
    timeout_seconds: int,
    label: str,
) -> None:
    model_name = str(model_name or "").strip()
    if not model_name:
        raise ValueError(f"{label} model is empty")
    if not api_key:
        raise ValueError(f"{label} API key is not configured")
    url = f"{resolve_openai_base_url(base_url)}/models/{quote(model_name, safe='')}"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    try:
        response = httpx.get(url, headers=headers, timeout=max(1, int(timeout_seconds or 90)))
    except Exception as exc:
        raise RuntimeError(f"failed validating {label} model '{model_name}': {exc}") from exc
    if response.status_code == 200:
        return
    if response.status_code == 404:
        raise ValueError(f"{label} model does not exist or is not accessible: {model_name}")
    raise RuntimeError(
        f"failed validating {label} model '{model_name}': "
        f"status={response.status_code} body={response.text[:200]}"
    )


@contextmanager
def temporary_model_overrides(
    *,
    llm_model: Optional[str] = None,
    extractor_model: Optional[str] = None,
) -> Iterator[None]:
    from ..infra.config import get_config

    backup = {
        "LLM_MODEL": os.environ.get("LLM_MODEL"),
        "EXTRACTOR_MODEL": os.environ.get("EXTRACTOR_MODEL"),
    }
    try:
        if llm_model:
            os.environ["LLM_MODEL"] = llm_model
        if extractor_model:
            os.environ["EXTRACTOR_MODEL"] = extractor_model
        get_config.cache_clear()
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_config.cache_clear()
