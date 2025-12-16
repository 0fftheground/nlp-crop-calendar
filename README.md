# NLP Crop Calendar (Chainlit + FastAPI + LangGraph)

This project demonstrates an end-to-end workflow for generating crop recommendations from farmer questions. A Chainlit UI collects inputs, a FastAPI backend orchestrates LLM reasoning, and LangGraph powers the multi-step agent that interprets intent, retrieves agronomy knowledge, and returns actionable plans. When a request is simple, the backend can short-circuit to a single LangChain-style tool (e.g., weather lookup) instead of running the full workflow. **OpenAI GPT models are required—no mock LLM is provided.**

## Components
- **Chainlit Frontend (`chainlit_app.py`)** – chat interface that sends prompts to the backend and renders results/trace info.
- **FastAPI Backend (`src/server.py`)** – exposes `/api/v1/handle` (auto route between tool & workflow) and `/api/v1/plan` (force LangGraph workflow).
- **Request Router (`src/router.py`)** – LLM-powered router that decides whether to execute a standalone tool (`src/tools.py`) or fall back to the LangGraph workflow.
- **LangGraph Workflow (`src/workflows/crop_graph.py`)** – state machine with parsing, planning, and recommendation nodes backed by LangChain models.
- **Knowledge Base (`src/knowledge_base.py`)** – curated crop-stage-task facts consumed by the planner.

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
   This launches `uvicorn` (`src.server:app`) and `chainlit run chainlit_app.py --watch` in parallel. Press `Ctrl+C` once to stop both.
2. Open the Chainlit URL from the console and converse with the assistant.

## Environment
Create a `.env` file (see `.env.example`) to specify LLM provider and API keys, e.g.:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
DEFAULT_REGION=global
```

## Development Notes
- `src/router.py` + `src/tool_selector.py` + `src/tools.py` implement the LLM-based “tool vs. workflow” decision logic. Add tool handlers or adjust the routing prompt to expand coverage.
- LangGraph state types live in `src/workflows/state.py`. Adding nodes/branches only requires editing `crop_graph.py`.
- `src/server.py` wires HTTP handlers to the router/graph runner; extend it with authentication, logging, or persistence as needed.
- An OpenAI API key is mandatory. The system instantiates `ChatOpenAI` (`gpt-4o-mini` by default) for both routing and workflow nodes.

More details live in `TECHNICAL_DETAILS.md`.
