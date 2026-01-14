# NLP Crop Calendar (Chainlit + FastAPI + LangGraph)

本项目展示了从农户问题生成种植建议的端到端流程。Chainlit 负责收集输入，FastAPI 提供后端服务，路由采用 LLM 驱动的 Planner+Executor，LangGraph 负责固定步骤的种植规划流程。简单问题由 planner 选择单一工具处理，复杂规划则进入 LangGraph 工作流。**必须使用 OpenAI GPT 模型（不提供 mock LLM）。**

## 组件
- **Chainlit 前端 (`chainlit_app.py`)** - 对话界面，向后端发送请求并展示结果/trace。
- **FastAPI 后端 (`src/api/server.py`)** - 对外提供 `/api/v1/handle`，由 planner 决定工具或工作流。
- **请求路由 (`src/agent/router.py`)** - Planner+Executor：通过 `src/agent/planner.py` 选择 tool/workflow 并执行，同时管理追问状态。
- **LangGraph 工作流 (`src/agent/workflows/crop_calendar_graph.py`/`src/agent/workflows/growth_stage_graph.py`)** - 固定步骤的种植规划流程（LLM 抽取 + 追问 + 工具并行）。

## 快速开始
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

1. **启动后端与 Chainlit**
   ```bash
   python run_all.py
   ```
   该命令会并行启动 `uvicorn`（`src.api.server:app`）和 `chainlit run chainlit_app.py --watch`。按一次 `Ctrl+C` 停止。
2. 打开控制台输出的 Chainlit URL 与助手对话。

## 环境变量
创建 `.env` 文件（参考 `.env.example`），配置 LLM 与各工具的 API 参数，例如：
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
EXTRACTOR_PROVIDER=openai
EXTRACTOR_MODEL=gpt-4o-mini
EXTRACTOR_API_KEY=
EXTRACTOR_API_BASE=
EXTRACTOR_TEMPERATURE=0.0
DEFAULT_REGION=global
VARIETY_PROVIDER=local
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
若 `EXTRACTOR_API_KEY` 为空，抽取模型将回退使用 `OPENAI_API_KEY`。
工具默认使用 `mock`。品种查询默认读取本地 SQLite（`VARIETY_PROVIDER=local`），仅在切换到内网接口时设为 `VARIETY_PROVIDER=intranet`；其他工具将 `*_PROVIDER` 设为 `intranet` 并配置 `*_API_URL`/`*_API_KEY` 可切换到内网接口；生育期预测必须使用 `GROWTH_STAGE_PROVIDER=intranet`。

## 开发说明
- `src/agent/router.py` + `src/agent/planner.py` 实现 Planner+Executor 逻辑，可通过调整 planner 提示词或新增工具扩展能力（提示词在 `src/prompts/planner.py`）。
- LangGraph 状态类型在 `src/agent/workflows/state.py`，新增节点/分支分别在 `src/agent/workflows/crop_calendar_graph.py` 与 `src/agent/workflows/growth_stage_graph.py` 编辑。
- 工作流流程：LLM 抽取种植信息，缺失字段触发追问（最多 2 次），随后气象/品种工具并行执行，`farming_recommendation` 消费上下文输出推荐。
- 业务服务集中在 `src/application/services`，工具层为薄适配器调用应用服务。
- LLM 提示词与工作流用户文案统一在 `src/prompts`（planner / 抽取 / workflow 文案 / tool 兜底）。
- `src/api/server.py` 负责将 HTTP 请求绑定到 router/graph，可按需扩展鉴权、日志或持久化。
- 必须提供 OpenAI API key。系统使用 `ChatOpenAI` 完成规划与抽取，抽取可通过 `EXTRACTOR_*` 选择轻量模型。
- `growth_stage_prediction` 仅用于读取历史缓存结果；生育期预测必须走 workflow，由工作流完成抽取/追问/内网调用。
- 作物日历/生育期 workflow 结果会基于标准化 `PlantingDetails` 生成缓存 key，命中则直接返回。
- 命中生育期缓存时，需要传入 `PlantingDetails` JSON（或包含 `planting` 字段的 JSON）。
- 追问控制：pending 状态会传入 LLM planner；显式“取消追问”会清理 pending，从而允许切换到新问题。
- 记忆偏好：标准化后的种植信息可按 session 缓存（TTL），使用前会提示确认；回复“清除记忆”可重置。
- 基础设施适配器集中在 `src/infra`（配置、LLM 客户端、结构化抽取等）。
- 与农事无关的请求会返回 `mode="none"`，跳过工具/工作流执行。
- 品种抽取使用候选名称匹配 + 模糊 token；数据来源使用 `resources/rice_variety_approvals.sqlite3`（可用 `VARIETY_DB_PATH` 覆盖）。

## 测试
```bash
python -m unittest
```
不调用 LLM 的意图测试：
```bash
python scripts/intent_routing_test.py --strategy rule
```

更多细节请查看 `TECHNICAL_DETAILS.md`。


