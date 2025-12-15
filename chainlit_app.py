"""
Chainlit 应用 - NLP Agent 对话界面
基于 Chainlit 框架，与 LangChain Agent 无缝集成
"""

import os
import json
import logging
from typing import Optional
from datetime import datetime

import chainlit as cl
from chainlit.input_widget import Slider, Select, TextInput

from src.app import NLPApp
from src.agent import NLPAgent, MultiTurnAgent

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 全局配置 ====================

# 应用配置
APP_NAME = "NLP Agent 对话系统"
APP_VERSION = "3.0.0"

# 默认配置
DEFAULT_LLM_PROVIDER = "mock"
DEFAULT_AGENT_TYPE = "react"

# 全局 NLP 应用实例
nlp_app: Optional[NLPApp] = None

# ==================== 初始化和配置 ====================

@cl.on_chat_start
async def start():
    """应用启动时执行"""
    global nlp_app
    
    logger.info("Chainlit 应用启动")
    
    # 初始化 NLP 应用
    nlp_app = NLPApp(
        use_agent=True,
        llm_provider=DEFAULT_LLM_PROVIDER,
        agent_type=DEFAULT_AGENT_TYPE
    )
    
    # 获取应用信息
    agent_info = get_agent_info()
    intents = load_intents()
    
    # 欢迎消息
    welcome_msg = f"""
# 欢迎使用 {APP_NAME} v{APP_VERSION}

## 系统信息
- **Agent 类型**: {agent_info['agent_type']}
- **LLM 提供商**: {agent_info['llm_provider']}
- **模型**: {agent_info['model_name']}
- **能力**: {', '.join(agent_info['capabilities'])}

## 可用意图 ({len(intents)} 个)
"""
    
    # 添加意图列表
    for intent in intents[:5]:  # 显示前 5 个意图
        welcome_msg += f"\n- **{intent['name']}**: {intent['description']}"
    
    if len(intents) > 5:
        welcome_msg += f"\n- ... 及其他 {len(intents) - 5} 个意图"
    
    welcome_msg += """

## 功能说明
- 💬 **自然语言理解** - 自动识别用户意图
- 🔗 **API 调用** - 智能调用相关 API
- 🧠 **智能推理** - 基于 LangChain Agent 的推理能力
- 📊 **多轮对话** - 支持完整的对话上下文

## 使用提示
- 尝试提出自然语言问题
- 描述你想执行的任务
- 系统会自动理解和处理你的请求
"""
    
    # 发送欢迎消息
    await cl.Message(content=welcome_msg).send()
    
    # 存储应用配置到会话
    cl.user_session.set("nlp_app", nlp_app)
    cl.user_session.set("agent_info", agent_info)
    cl.user_session.set("message_count", 0)


@cl.on_settings_update
async def setup_agent(settings):
    """更新 Agent 配置"""
    global nlp_app
    
    logger.info(f"更新配置: {settings}")
    
    # 获取新的配置
    llm_provider = settings.get("llm_provider", DEFAULT_LLM_PROVIDER)
    agent_type = settings.get("agent_type", DEFAULT_AGENT_TYPE)
    
    # 重新初始化应用
    nlp_app = NLPApp(
        use_agent=True,
        llm_provider=llm_provider,
        agent_type=agent_type
    )
    
    cl.user_session.set("nlp_app", nlp_app)
    
    await cl.Message(
        content=f"✅ 已更新配置:\n- LLM: {llm_provider}\n- Agent 类型: {agent_type}"
    ).send()


# ==================== 辅助函数 ====================

def get_agent_info() -> dict:
    """获取 Agent 信息"""
    return {
        "agent_type": DEFAULT_AGENT_TYPE,
        "llm_provider": DEFAULT_LLM_PROVIDER,
        "model_name": "gpt-3.5-turbo" if DEFAULT_LLM_PROVIDER == "openai" else "local-model",
        "capabilities": [
            "intent_recognition",
            "api_calling",
            "multi_turn_conversation",
            "knowledge_base_search"
        ]
    }


def load_intents() -> list:
    """加载意图配置"""
    try:
        intents_path = os.path.join(
            os.path.dirname(__file__),
            "config",
            "intents.json"
        )
        
        if os.path.exists(intents_path):
            with open(intents_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("intents", [])
    except Exception as e:
        logger.error(f"加载意图失败: {str(e)}")
    
    # 返回默认意图
    return [
        {
            "name": "greeting",
            "description": "问候和基本对话",
            "examples": ["你好", "早上好", "谢谢"]
        },
        {
            "name": "question_answering",
            "description": "回答用户问题",
            "examples": ["天气如何?", "今天几号?", "是什么意思?"]
        },
        {
            "name": "task_execution",
            "description": "执行特定任务",
            "examples": ["帮我查一下", "给我发送", "列出所有"]
        }
    ]


def format_response(result: dict) -> str:
    """格式化 Agent 响应"""
    mode = result.get("mode", "unknown")
    confidence = result.get("confidence", 0.0)
    success = result.get("success", False)

    # 优先使用模型响应
    if result.get("response"):
        formatted = result["response"]
    else:
        chunks = []

        if success:
            # 无直接 response 时，展示意图识别/工具调用信息
            intent = result.get("intent")
            if intent:
                chunks.append(f"意图: {intent}")
                chunks.append(f"置信度: {confidence:.0%}")

            api_resp = result.get("api_response")
            error = result.get("error")
            if api_resp is not None:
                try:
                    chunks.append(f"API 响应: {json.dumps(api_resp, ensure_ascii=False, indent=2)}")
                except Exception:
                    chunks.append(f"API 响应: {api_resp}")
            if error:
                chunks.append(f"错误: {error}")

            if not chunks:
                chunks.append("处理完成，但无可展示内容")
        else:
            error = result.get("error") or result.get("message") or "处理失败"
            chunks = [f"错误: {error}"]

        formatted = "\n".join(chunks)

    # 附加模式信息
    if mode and mode != "unknown":
        formatted += f"\n\n---\n*模式: {mode} | 置信度: {confidence:.0%}*"

    return formatted


# ==================== 主消息处理 ====================

@cl.on_message
async def main(message: cl.Message):
    """处理用户消息"""
    try:
        # 获取 NLP 应用
        nlp_app = cl.user_session.get("nlp_app")
        if not nlp_app:
            await cl.Message(
                content="❌ 错误: 应用未初始化，请刷新页面"
            ).send()
            return
        
        # 获取用户输入
        user_input = message.content.strip()
        
        if not user_input:
            await cl.Message(
                content="⚠️ 请输入有效的问题或指令"
            ).send()
            return
        
        # 更新消息计数
        message_count = cl.user_session.get("message_count", 0) + 1
        cl.user_session.set("message_count", message_count)
        
        # 处理输入
        try:
            result = nlp_app.process_input(user_input)
            
            # 格式化响应
            response_text = format_response(result)
            
            # 直接发送结果
            await cl.Message(content=response_text).send()
            
            # 记录日志
            logger.info(
                f"处理请求 #{message_count}: "
                f"输入={user_input[:50]}, "
                f"模式={result.get('mode', 'unknown')}"
            )
            
        except Exception as e:
            logger.error(f"处理消息时出错: {str(e)}")
            await cl.Message(content=f"❌ 处理失败: {str(e)}").send()
    
    except Exception as e:
        logger.error(f"消息处理出错: {str(e)}")
        await cl.Message(
            content=f"❌ 发生错误: {str(e)}"
        ).send()


# ==================== 自定义操作 ====================

@cl.action_callback("show_info")
async def show_info():
    """显示系统信息"""
    agent_info = get_agent_info()
    info_text = f"""
# 系统信息

| 项目 | 值 |
|------|-----|
| Agent 类型 | {agent_info['agent_type']} |
| LLM 提供商 | {agent_info['llm_provider']} |
| 模型名称 | {agent_info['model_name']} |
| 应用版本 | v{APP_VERSION} |
| 时间戳 | {datetime.now().isoformat()} |

## 能力清单
"""
    for cap in agent_info['capabilities']:
        info_text += f"- ✅ {cap}\n"
    
    await cl.Message(content=info_text).send()


@cl.action_callback("show_intents")
async def show_intents():
    """显示可用意图"""
    intents = load_intents()
    
    intents_text = f"# 可用意图列表 ({len(intents)} 个)\n\n"
    
    for i, intent in enumerate(intents, 1):
        intents_text += f"## {i}. {intent.get('name', 'unknown')}\n"
        intents_text += f"**描述**: {intent.get('description', '')}\n"
        intents_text += f"**示例**: {', '.join(intent.get('examples', []))}\n\n"
    
    await cl.Message(content=intents_text).send()


@cl.action_callback("clear_history")
async def clear_history():
    """清除聊天历史"""
    cl.user_session.set("message_count", 0)
    await cl.Message(
        content="✅ 聊天历史已清除"
    ).send()


# ==================== 自定义设置 ====================

@cl.set_starters
async def set_starters():
    """设置快速启动按钮"""
    return [
        cl.Starter(
            label="问候",
            message="你好，请问你是谁？",
            icon="👋",
        ),
        cl.Starter(
            label="能力演示",
            message="请展示你的能力和功能",
            icon="🚀",
        ),
        cl.Starter(
            label="天气查询",
            message="请查一下今天的天气",
            icon="⛅",
        ),
        cl.Starter(
            label="任务列表",
            message="请列出你能处理的所有任务",
            icon="📋",
        ),
    ]


@cl.set_chat_profiles
async def chat_profiles():
    """设置聊天模式"""
    return [
        cl.ChatProfile(
            name="标准模式",
            markdown_description="标准的对话模式",
            icon="⚙️",
        ),
        cl.ChatProfile(
            name="详细模式",
            markdown_description="显示完整的思考过程",
            icon="🔍",
        ),
        cl.ChatProfile(
            name="快速模式",
            markdown_description="快速响应，简洁回答",
            icon="⚡",
        ),
    ]


# ==================== 应用配置 ====================

# 设置应用名称和描述
def setup_chainlit():
    """设置 Chainlit 应用配置"""
    # 这些配置通过 .chainlit/config.toml 设置
    pass


if __name__ == "__main__":
    # 注意: Chainlit 应用需要通过 chainlit run 启动
    # 不能直接运行此文件
    logger.warning(
        "请使用以下命令启动应用:\n"
        "chainlit run chainlit_app.py"
    )
