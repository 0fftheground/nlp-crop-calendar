from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..agent.router import RequestRouter
from ..data.config import get_config
from ..schemas.models import HandleResponse, UserRequest


@lru_cache(maxsize=1)
def get_router():
    router = RequestRouter()
    return router


app = FastAPI(title="Crop Calendar Planner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    cfg = get_config()
    return {"status": "ok", "llm": cfg.llm_provider}


@app.post("/api/v1/handle", response_model=HandleResponse)
async def handle_request(request: UserRequest):
    router = get_router()
    return router.handle(request)
