from functools import lru_cache

import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from ..agent.router import RequestRouter
from ..infra.config import get_config
from ..schemas.models import HandleResponse, UserRequest


@lru_cache(maxsize=1)
def get_router():
    router = RequestRouter()
    return router


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LOG_PATH.touch(exist_ok=True)
    except Exception:
        pass
    print(f"API error log path: {_LOG_PATH}")
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


def _append_error_log(message: str, tb: str = "") -> None:
    timestamp = datetime.utcnow().isoformat()
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}Z] {message}\n{tb}\n")
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
    try:
        return router.handle(request)
    except Exception as exc:
        tb = traceback.format_exc()
        _append_error_log(str(exc), tb)
        print(f"handle_request failed: {exc}\n{tb}")
        raise HTTPException(status_code=500, detail={"error": str(exc), "traceback": tb})
