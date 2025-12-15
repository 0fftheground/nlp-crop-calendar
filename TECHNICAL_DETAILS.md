# 技术细节（面向开发者）

## 架构总览
- 入口：`NLPApp`（`src/app.py`）负责调度，优先用 `MultiTurnAgent`，失败降级到 LLM 意图识别 + API 调用。
- Agent：`src/agent.py` 使用 LangChain，支持 `openai_functions` / `react`，加载全部工具（`src/agent_tools.py`），`MultiTurnAgent` 维护对话历史。
- 工具集：`src/agent_tools.py` 暴露 API 调用、意图识别、搜索、天气、翻译、模式切换等；`toggle_mock_mode` 直接切换 `APICaller`。
- 意图识别：`src/intent_recognition_manager.py` + `src/llm_intent_recognizer.py`，LLM 驱动，带工具 schema 感知，fallback 返回 `unknown`。
- API 调用：`src/api_caller.py` 封装 requests，支持环境变量 `USE_MOCK_API` 或运行时切换 Mock。
- 前端/服务：Chainlit（`chainlit_app.py` + `run_chainlit.py`）和 FastAPI（`src/web_app_fastapi.py` + `run_web.py`）。

## 流程（Agent 优先，失败降级）
1) `NLPApp.process_user_input(user_input)` → 若 Agent 可用则 `MultiTurnAgent.chat_with_history`。
2) Agent 失败时降级：`IntentRecognitionManager.recognize` → 拿到 intent/tool/schema → `_prepare_api_params` 填默认参数 → `APICaller.call_api`。
3) 返回结构包含 `mode`、`intent`、`confidence`、`api_response` 或 Agent 响应。

## Chainlit 前端请求链路
- **会话初始化**：`chainlit_app.py:37-96` 中的 `@cl.on_chat_start` 启动 `NLPApp(use_agent=True, llm_provider=DEFAULT_LLM_PROVIDER, agent_type=DEFAULT_AGENT_TYPE)`，并把实例、Agent 信息、消息计数等写入 `cl.user_session`，方便跨消息持久化。
- **消息入口**：每当前端发送文本，`@cl.on_message`（`chainlit_app.py:222-276`）读取 `nlp_app`，做空输入校验后先回复“处理中…”占位，再调用 `nlp_app.process_input(user_input)` 获取结果。
- **结果格式化**：`format_response`（`chainlit_app.py:176-217`）优先展示 Agent 的自然语言回复；若走意图识别，则拼意图名、置信度、API 响应或错误，并在底部附 `mode/confidence`。
- **日志与统计**：`cl.user_session["message_count"]` 记录会话内调用次数；`logger.info` 输出输入摘要与命中模式。
- **多轮上下文**：Chainlit 只负责 session 粘性，真正的对话历史由 `MultiTurnAgent.chat_with_history` 在 Agent 内部维护，因此 Chainlit 重载 `nlp_app` 或清 session 时历史会被丢弃。

## LLM 意图识别与工具集成
- 工具 schema：`AGENT_TOOLS_SCHEMA`（`intent_recognition_manager.py`）列出工具名、描述、参数。
- 提示词：`LLMIntentRecognizer._build_system_prompt` 向模型提供完整 schema，要求返回 JSON（intent、confidence、tool、required_params、clarification）。
- 关联策略：`_find_associated_tool` 先名称匹配，再 details.tool，最后描述搜索。
- 辅助能力：`get_suggestions`、`extract_params`、`clarify_intent`、`batch_recognize_intents`。

## Mock 模式实现
- 开关：环境变量 `USE_MOCK_API=true` 或 `APICaller.set_mock_mode(True/False)`；链路中可用工具 `toggle_mock_mode`。
- 数据：`MockAPIData.get_mock_response` 针对天气/翻译/搜索返回固定结构；未匹配则返回通用搜索结果。
- 日志：启动或切换时会打印当前模式；Mock 请求带 `[Mock]`。

## 关键文件与职责
- `src/app.py`：主调度、参数填充、历史管理。
- `src/agent.py`：Agent 构建、执行、流式接口、历史封装。
- `src/agent_tools.py`：工具定义和列表。
- `src/intent_recognition_manager.py`：LLM 意图识别管理和工具 schema 绑定。
- `src/llm_intent_recognizer.py`：基于 LLM 的提示构造、解析、参数提取。
- `src/api_caller.py`：requests 封装、Mock 数据、模式切换。
- `chainlit_app.py` / `run_chainlit.py`：Chainlit 前端。
- `src/web_app_fastapi.py` / `run_web.py`：FastAPI 服务与 WebSocket。

## 现存注意点
- `NLPApp` 暴露 `process_user_input`，而 Chainlit/FastAPI 当前调用 `process_input`，需对齐后端方法名以免运行时报错。
- `intent_recognition_manager` 调用 `init_llm`，但 `llm_config.py` 暂未实现同名函数；如需使用该路径需补齐或改为 `get_llm/init_*`。

## 开发与测试建议
- 默认使用 Mock 进行本地调试；切换真实 LLM 需配置 `OPENAI_API_KEY` 或本地 Ollama。
- 若引入新工具，需同时更新 `AGENT_TOOLS_SCHEMA` 和 `get_all_tools` 列表，保证意图识别与 Agent 同步。
- 运行接口调试：`python run_web.py --reload --llm mock` 后访问 `/docs`；Chainlit 调试用 `python run_chainlit.py --watch`.
