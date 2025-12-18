from functools import lru_cache
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import get_config
from .models import HandleResponse, PlanRequest
from .router import RequestRouter


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
async def handle_request(request: PlanRequest):
    router = get_router()
    return router.handle(request)
