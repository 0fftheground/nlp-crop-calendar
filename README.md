# NLP Crop Calendar (Chainlit + FastAPI + LangGraph)

This project demonstrates an end-to-end workflow for generating crop recommendations from farmer questions. A Chainlit UI collects inputs, a FastAPI backend hosts a LangChain tool-calling agent, and LangGraph powers the fixed-step crop planning workflow. When a request is simple, the agent can call a single tool; for planning, it invokes the LangGraph workflow tool. **OpenAI GPT models are required--no mock LLM is provided.**

## Components
- **Chainlit Frontend (`chainlit_app.py`)** - chat interface that sends prompts to the backend and renders results/trace info.
- **FastAPI Backend (`src/api/server.py`)** - exposes `/api/v1/handle` for agent-driven routing between tools and workflow.
- **Request Router (`src/agent/router.py`)** - tool-calling agent that can run standalone tools (`src/tools/registry.py`) or invoke the LangGraph workflow tool.
- **LangGraph Workflow (`src/agent/workflows/crop_graph.py`)** - fixed-step state machine for the crop planning flow (LLM extraction + follow-up + tools).

## Getting Started
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

1. **Start both backend + Chainlit**
   ```bash
   python run_all.py
   ```
   This launches `uvicorn` (`src.api.server:app`) and `chainlit run chainlit_app.py --watch` in parallel. Press `Ctrl+C` once to stop both.
2. Open the Chainlit URL from the console and converse with the assistant.

## Environment
Create a `.env` file (see `.env.example`) to specify LLM provider and API keys, e.g.:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
EXTRACTOR_PROVIDER=openai
EXTRACTOR_MODEL=gpt-4o-mini
EXTRACTOR_API_KEY=
EXTRACTOR_API_BASE=
EXTRACTOR_TEMPERATURE=0.0
DEFAULT_REGION=global
VARIETY_PROVIDER=mock
VARIETY_API_URL=
VARIETY_API_KEY=
WEATHER_PROVIDER=mock
WEATHER_API_URL=
WEATHER_API_KEY=
GROWTH_STAGE_PROVIDER=mock
GROWTH_STAGE_API_URL=
GROWTH_STAGE_API_KEY=
RECOMMENDATION_PROVIDER=mock
RECOMMENDATION_API_URL=
RECOMMENDATION_API_KEY=
```
If `EXTRACTOR_API_KEY` is empty, the extractor falls back to `OPENAI_API_KEY`.
Tool providers default to `mock`. Set `*_PROVIDER=intranet` and supply the `*_API_URL`/`*_API_KEY` fields to switch to intranet endpoints. Growth stage prediction requires `GROWTH_STAGE_PROVIDER=intranet`.

## Development Notes
- `src/agent/router.py` + `src/tools/registry.py` implement the tool-calling agent logic. Add tool handlers or adjust the agent prompt to expand coverage.
- LangGraph state types live in `src/agent/workflows/state.py`. Adding nodes/branches only requires editing `crop_graph.py`.
- Workflow details: LLM extracts planting details, missing fields trigger follow-up questions (up to 2 times), then weather/variety tools run in parallel, and `farming_recommendation` consumes the combined context.
- `src/api/server.py` wires HTTP handlers to the router/graph runner; extend it with authentication, logging, or persistence as needed.
- An OpenAI API key is mandatory. The system instantiates `ChatOpenAI` for both routing and extraction, with extraction optionally using a lighter model via `EXTRACTOR_*` settings.
- `growth_stage_prediction` uses structured extraction (PlantingDetailsDraft) and will ask for missing planting fields before calling the intranet growth stage service.
- Infrastructure adapters live under `src/infra` (config, LLM clients, structured extraction).
- For unrelated prompts, the router can return `mode="none"` and skip tool/workflow execution.
- A minimal local variety store lives in `src/resources/varieties.json`, used for retrieval hints during extraction.
- Variety retrieval prefers semantic similarity via embeddings (falls back to fuzzy match); optional env: `EMBEDDING_MODEL`.
- If Qdrant is configured, retrieval will query Qdrant first (`QDRANT_URL`, `QDRANT_COLLECTIONS` with `"variety"` key).

## Tests
```bash
python -m unittest
```
For routing checks without LLM calls:
```bash
python scripts/intent_routing_test.py --strategy rule
```

## Variety Embedding (Qdrant)
If you run a local Qdrant instance, build the variety embeddings and upsert once:
```bash
python scripts/build_variety_qdrant.py --qdrant-url http://localhost:6333
```
Ensure `.env` includes:
```
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTIONS={"variety":"varieties"}
EMBEDDING_MODEL=text-embedding-3-small
```

More details live in `TECHNICAL_DETAILS.md`.


