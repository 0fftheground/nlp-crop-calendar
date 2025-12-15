# 用户指南（面向使用者）

## 快速开始
- 安装依赖：`pip install -r requirements.txt`
- Chainlit 对话界面（推荐演示）：`python run_chainlit.py` → 浏览器访问 `http://localhost:8000`
- FastAPI 接口（高性能 API）：`python run_web.py` → 浏览器访问 `http://localhost:5000/docs`

### 常用参数
- Chainlit：`--port 8080`、`--host 0.0.0.0`、`--watch`（热重载）、`--headless`（不自动开浏览器）、`--llm openai|ollama|mock`
- FastAPI：`--port 8081`、`--host 0.0.0.0`、`--reload`（热重载）、`--llm openai|ollama|mock`

## Mock 模式（无网可用）
- 启动前设置环境变量：`USE_MOCK_API=true python run_chainlit.py`
- 运行时切换：在聊天里输入“启用Mock模式”或“禁用Mock模式”（调用 `toggle_mock_mode` 工具）。
- 覆盖范围：天气、翻译、搜索示例接口都会返回内置 Mock 数据。

## Chainlit 使用提示
- 首页会显示系统信息、能力说明、可用意图。
- 快速启动按钮：问候、能力演示、天气查询、任务列表。
- 聊天模式：标准 / 详细（显示思考链） / 快速。
- 常见问题：端口占用可换端口；未初始化提示可重启；需要外观修改可编辑 `.chainlit/config.toml`。

## 典型场景
1) 离线/受限网络演示：`USE_MOCK_API=true python run_chainlit.py`
2) API 接入测试：`python run_web.py --llm openai` 并在 `/docs` 调用
3) 快速体验工具：在聊天中询问天气、翻译或搜索问题，或直接输入“展示能力”

## 文件导航
- 应用入口：`run_chainlit.py`（Chainlit），`run_web.py`（FastAPI）
- 对话逻辑：`src/app.py`，`src/agent.py`，`src/agent_tools.py`
- 模式切换：`src/api_caller.py`（Mock/真实）
- 配置示例：`.chainlit/config.toml`

## 故障排查
- “Chainlit 未安装”：`pip install chainlit`
- “消息不能为空”或无响应：确认输入非空；检查终端日志。
- 外网受限：确认 `USE_MOCK_API=true` 已生效（日志会打印 Mock 模式）。
