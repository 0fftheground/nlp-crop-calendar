from functools import lru_cache
import time
from uuid import uuid4

import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from ..agent.router import RequestRouter
from ..infra.config import get_config
from ..infra.export_store import resolve_export_path
from ..infra.interaction_store import get_interaction_store
from ..observability.logging_utils import (
    init_logging,
    log_event,
    reset_trace_id,
    set_trace_id,
    summarize_text,
)
from ..observability.otel import (
    init_otel,
    instrument_fastapi,
    instrument_httpx,
)
from ..schemas.models import HandleResponse, UserRequest


@lru_cache(maxsize=1)
def get_router():
    router = RequestRouter()
    return router


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_logging(log_path=str(_OBS_LOG_PATH))
    init_otel()
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LOG_PATH.touch(exist_ok=True)
    except Exception:
        pass
    print(f"API error log path: {_LOG_PATH}")
    log_event(
        "startup",
        log_path=str(_LOG_PATH),
        observability_log=str(_OBS_LOG_PATH),
    )
    instrument_fastapi(app)
    instrument_httpx()
    yield


app = FastAPI(title="Crop Calendar Planner", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOG_PATH = _PROJECT_ROOT / "api_errors.log"
_OBS_LOG_PATH = _PROJECT_ROOT / "observability.log"


def _append_error_log(message: str, tb: str = "") -> None:
    timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n{tb}\n")
    except Exception:
        pass


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    _append_error_log(f"Unhandled error at {request.url.path}: {exc}", tb)
    return JSONResponse(
        status_code=500,
        content={"detail": {"error": str(exc), "traceback": tb}},
    )


@app.get("/health")
async def health():
    cfg = get_config()
    return {"status": "ok", "llm": cfg.llm_provider}


@app.post("/api/v1/handle", response_model=HandleResponse)
async def handle_request(request: UserRequest):
    router = get_router()
    store = get_interaction_store()
    started_at = time.time()
    trace_id = uuid4().hex
    token = set_trace_id(trace_id)
    try:
        log_event(
            "request_received",
            input_schema=request.model_dump(mode="json"),
        )
        response = router.handle(request)
    except Exception as exc:
        tb = traceback.format_exc()
        _append_error_log(str(exc), tb)
        print(f"handle_request failed: {exc}\n{tb}")
        reset_trace_id(token)
        raise HTTPException(status_code=500, detail={"error": str(exc), "traceback": tb})
    latency_ms = int((time.time() - started_at) * 1000)
    try:
        store.record(request, response, latency_ms)
    except Exception:
        pass
    message = ""
    if response.mode == "tool" and response.tool:
        message = response.tool.message
    elif response.plan:
        message = response.plan.message
    session_id = request.session_id or request.user_id
    log_event(
        "response_ready",
        mode=response.mode,
        message_summary=summarize_text(message),
        latency_ms=latency_ms,
        session_id=session_id,
        user_id=request.user_id,
    )
    reset_trace_id(token)
    return response


@app.get("/api/v1/download/{file_id}")
async def download_export(file_id: str):
    try:
        path = resolve_export_path(file_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="invalid download id")
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"{file_id}.csv",
    )
