"""
FastAPI Web 应用 - NLP Agent 对话界面
支持实时对话、多轮交互和 WebSocket 通信
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, List
import json
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.app import NLPApp

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================

class Message(BaseModel):
    """消息模型"""
    id: int
    content: str
    role: str  # "user" 或 "assistant"
    timestamp: str


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str


class ChatResponse(BaseModel):
    """聊天响应"""
    success: bool
    user_message: Optional[Message] = None
    assistant_message: Optional[Message] = None
    error: Optional[str] = None


class StatusResponse(BaseModel):
    """状态响应"""
    status: str
    version: str
    timestamp: str


class AgentInfo(BaseModel):
    """Agent 信息"""
    agent_type: str
    llm_provider: str
    model_name: str
    capabilities: List[str]


class IntentItem(BaseModel):
    """意图项"""
    name: str
    description: str
    examples: List[str]


class MessagesResponse(BaseModel):
    """消息列表响应"""
    messages: List[Message]
    count: int


class ClearResponse(BaseModel):
    """清除响应"""
    success: bool
    message: str


class AgentInfoResponse(BaseModel):
    """Agent 信息响应"""
    agent_type: str
    llm_provider: str
    model_name: str
    capabilities: List[str]
    session_id: str


# ==================== ChatSession 类 ====================

class ChatSession:
    """用户聊天会话"""

    def __init__(self, session_id: str, llm_provider: str = "mock", agent_type: str = "react"):
        self.session_id = session_id
        self.messages: List[Message] = []
        self.app = NLPApp(use_agent=True, llm_provider=llm_provider, agent_type=agent_type)
        self.created_at = datetime.now()
        self.llm_provider = llm_provider
        self.agent_type = agent_type

    def add_message(self, content: str, role: str = "user") -> Message:
        """添加消息到历史"""
        message = Message(
            id=len(self.messages),
            content=content,
            role=role,
            timestamp=datetime.now().isoformat()
        )
        self.messages.append(message)
        return message

    def process_input(self, user_input: str) -> dict:
        """处理用户输入"""
        try:
            result = self.app.process_input(user_input)
            return result
        except Exception as e:
            logger.error(f"处理输入时出错: {str(e)}")
            return {"error": str(e), "response": None}

    def get_messages(self) -> List[Message]:
        """获取所有消息"""
        return self.messages

    def clear_messages(self):
        """清除所有消息"""
        self.messages = []

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "created_at": self.created_at.isoformat(),
            "llm_provider": self.llm_provider,
            "agent_type": self.agent_type
        }


# ==================== 全局状态 ====================

# 存储用户会话
user_sessions: Dict[str, ChatSession] = {}

# FastAPI 应用实例
app: FastAPI = None


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    global app

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理"""
        logger.info("FastAPI 应用启动")
        yield
        logger.info("FastAPI 应用关闭")

    app = FastAPI(
        title="NLP Agent 对话系统",
        description="基于 LangChain Agent 的自然语言对话应用",
        version="2.0.0",
        lifespan=lifespan
    )

    # ==================== CORS 配置 ====================
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ==================== 静态文件 ====================
    # 挂载静态文件
    static_path = os.path.join(os.path.dirname(__file__), "..", "static")
    if os.path.exists(static_path):
        app.mount("/static", StaticFiles(directory=static_path), name="static")

    # ==================== 辅助函数 ====================

    def get_or_create_session(session_id: Optional[str] = None, llm_provider: str = "mock", 
                             agent_type: str = "react") -> tuple[str, ChatSession]:
        """获取或创建用户会话"""
        if not session_id or session_id not in user_sessions:
            session_id = str(uuid.uuid4())
            user_sessions[session_id] = ChatSession(session_id, llm_provider, agent_type)
            logger.info(f"创建新会话: {session_id}")
        return session_id, user_sessions[session_id]

    def get_intents() -> List[IntentItem]:
        """获取可用意图列表"""
        try:
            intents_path = os.path.join(os.path.dirname(__file__), "..", "config", "intents.json")
            if os.path.exists(intents_path):
                with open(intents_path, "r", encoding="utf-8") as f:
                    intents_data = json.load(f)
                    return [
                        IntentItem(
                            name=intent.get("name", ""),
                            description=intent.get("description", ""),
                            examples=intent.get("examples", [])
                        )
                        for intent in intents_data.get("intents", [])
                    ]
        except Exception as e:
            logger.error(f"加载意图时出错: {str(e)}")
        return []

    # ==================== REST API 路由 ====================

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """主页 - 返回 HTML 模板"""
        template_path = os.path.join(os.path.dirname(__file__), "..", "templates", "index.html")
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>NLP Agent 对话系统</h1><p>找不到模板文件</p>"

    @app.get("/api/status", response_model=StatusResponse)
    async def get_status():
        """获取应用状态"""
        return StatusResponse(
            status="running",
            version="2.0.0",
            timestamp=datetime.now().isoformat()
        )

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """处理聊天请求"""
        try:
            user_input = request.message.strip()
            if not user_input:
                raise HTTPException(status_code=400, detail="消息不能为空")

            # 使用默认会话
            session_id, chat_session = get_or_create_session()

            # 添加用户消息
            user_msg = chat_session.add_message(user_input, role="user")

            # 处理输入
            result = chat_session.process_input(user_input)

            # 获取响应
            response_text = result.get("response") or result.get("error", "无响应")

            # 添加 AI 消息
            assistant_msg = chat_session.add_message(response_text, role="assistant")

            return ChatResponse(
                success=True,
                user_message=user_msg,
                assistant_message=assistant_msg
            )

        except Exception as e:
            logger.error(f"处理聊天请求时出错: {str(e)}")
            return ChatResponse(
                success=False,
                error=str(e)
            )

    @app.get("/api/messages", response_model=MessagesResponse)
    async def get_messages():
        """获取聊天历史"""
        session_id, chat_session = get_or_create_session()
        messages = chat_session.get_messages()
        return MessagesResponse(
            messages=messages,
            count=len(messages)
        )

    @app.post("/api/clear", response_model=ClearResponse)
    async def clear_history():
        """清除聊天历史"""
        try:
            session_id, chat_session = get_or_create_session()
            chat_session.clear_messages()
            return ClearResponse(
                success=True,
                message="聊天历史已清除"
            )
        except Exception as e:
            logger.error(f"清除历史时出错: {str(e)}")
            return ClearResponse(
                success=False,
                message=str(e)
            )

    @app.get("/api/intents")
    async def get_intents_endpoint():
        """获取可用意图列表"""
        try:
            intents = get_intents()
            return {
                "success": True,
                "intents": intents,
                "count": len(intents)
            }
        except Exception as e:
            logger.error(f"获取意图时出错: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "intents": [],
                "count": 0
            }

    @app.get("/api/agent-info", response_model=AgentInfoResponse)
    async def get_agent_info():
        """获取 Agent 信息"""
        try:
            session_id, chat_session = get_or_create_session()
            return AgentInfoResponse(
                agent_type=chat_session.agent_type,
                llm_provider=chat_session.llm_provider,
                model_name="gpt-3.5-turbo" if chat_session.llm_provider == "openai" else "local",
                capabilities=["intent_recognition", "api_calling", "conversation"],
                session_id=session_id
            )
        except Exception as e:
            logger.error(f"获取 Agent 信息时出错: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    # ==================== WebSocket 路由 ====================

    class ConnectionManager:
        """WebSocket 连接管理器"""

        def __init__(self):
            self.active_connections: Dict[str, List[WebSocket]] = {}

        async def connect(self, websocket: WebSocket, session_id: str):
            """建立连接"""
            await websocket.accept()
            if session_id not in self.active_connections:
                self.active_connections[session_id] = []
            self.active_connections[session_id].append(websocket)
            logger.info(f"WebSocket 连接建立: {session_id}")

        def disconnect(self, websocket: WebSocket, session_id: str):
            """断开连接"""
            if session_id in self.active_connections:
                self.active_connections[session_id].remove(websocket)
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
            logger.info(f"WebSocket 连接断开: {session_id}")

        async def broadcast(self, session_id: str, message: dict):
            """广播消息到所有连接"""
            if session_id in self.active_connections:
                for connection in self.active_connections[session_id]:
                    try:
                        await connection.send_json(message)
                    except Exception as e:
                        logger.error(f"发送消息失败: {str(e)}")

    manager = ConnectionManager()

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket 端点"""
        # 获取或创建会话
        session_id, chat_session = get_or_create_session(session_id)

        # 建立连接
        await manager.connect(websocket, session_id)

        try:
            # 发送连接确认
            await websocket.send_json({
                "type": "connected",
                "session_id": session_id,
                "message": "已连接到服务器"
            })

            # 接收消息循环
            while True:
                data = await websocket.receive_json()
                message_text = data.get("message", "").strip()

                if not message_text:
                    await websocket.send_json({
                        "type": "error",
                        "message": "消息不能为空"
                    })
                    continue

                try:
                    # 发送用户消息
                    user_msg = chat_session.add_message(message_text, role="user")
                    await manager.broadcast(session_id, {
                        "type": "user_message",
                        "message": user_msg.model_dump()
                    })

                    # 发送处理中状态
                    await manager.broadcast(session_id, {
                        "type": "processing",
                        "message": "处理中..."
                    })

                    # 处理消息
                    result = chat_session.process_input(message_text)

                    # 获取响应
                    response_text = result.get("response") or result.get("error", "无响应")

                    # 添加 AI 消息
                    assistant_msg = chat_session.add_message(response_text, role="assistant")

                    # 发送 AI 响应
                    await manager.broadcast(session_id, {
                        "type": "assistant_message",
                        "message": assistant_msg.model_dump(),
                        "mode": result.get("mode", "unknown")
                    })

                except Exception as e:
                    logger.error(f"处理 WebSocket 消息时出错: {str(e)}")
                    await manager.broadcast(session_id, {
                        "type": "error",
                        "message": f"处理失败: {str(e)}"
                    })

        except WebSocketDisconnect:
            manager.disconnect(websocket, session_id)
            logger.info(f"客户端断开连接: {session_id}")
        except Exception as e:
            logger.error(f"WebSocket 错误: {str(e)}")
            manager.disconnect(websocket, session_id)

    return app


# ==================== 应用入口 ====================

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )
