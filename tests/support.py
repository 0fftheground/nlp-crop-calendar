import os
from unittest.mock import patch


class DummyLLM:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return {"action": "none", "response": "noop"}


class DummySessionActionLLM:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        prompt = ""
        if isinstance(messages, list) and messages:
            last = messages[-1]
            prompt = str(getattr(last, "content", "") or "")
        if "换个问题" in prompt or "新任务" in prompt:
            return {
                "thread_switch": True,
                "action_type": "none",
                "confidence": 0.95,
            }
        if "删除" in prompt or "移除" in prompt:
            return {
                "thread_switch": False,
                "action_type": "delete_plan",
                "confidence": 0.95,
            }
        if "生育期" in prompt or "生长阶段" in prompt or "成熟期" in prompt:
            return {
                "thread_switch": False,
                "action_type": "query_growth_stage",
                "confidence": 0.95,
            }
        if any(token in prompt for token in ("记录", "录入", "登记", "新增", "添加")):
            return {
                "thread_switch": False,
                "action_type": "record_task",
                "confidence": 0.95,
            }
        return {
            "thread_switch": False,
            "action_type": "none",
            "confidence": 0.2,
        }


def build_test_router():
    from src.agent.router import RequestRouter

    with patch("src.agent.planner.get_chat_model", return_value=DummyLLM()):
        with patch(
            "src.agent.fast_intent.get_extractor_model", return_value=DummyLLM()
        ):
            with patch(
                "src.agent.session_context.get_extractor_model",
                return_value=DummySessionActionLLM(),
            ):
                return RequestRouter()


def memory_env():
    return {
        "PENDING_STORE": os.environ.get("PENDING_STORE"),
        "INTENT_ROUTING_MODE": os.environ.get("INTENT_ROUTING_MODE"),
    }
