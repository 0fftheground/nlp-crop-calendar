from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_config
from .models import HandleResponse, PlanRequest, PlanResponse
from .router import RequestRouter
from .workflows.crop_graph import build_graph


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


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


@app.post("/api/v1/plan", response_model=PlanResponse)
async def run_plan(request: PlanRequest):
    graph = get_graph()
    initial_state = {"user_prompt": request.prompt, "trace": []}
    state = graph.invoke(initial_state)
    return PlanResponse(
        query=state["query"],
        recommendations=state.get("recommendations", []),
        message=state.get("message", ""),
        trace=state.get("trace", []),
    )


@app.post("/api/v1/handle", response_model=HandleResponse)
async def handle_request(request: PlanRequest):
    router = get_router()
    return router.handle(request)
