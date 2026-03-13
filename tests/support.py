import os
from unittest.mock import patch


class DummyLLM:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return {"action": "none", "response": "noop"}


def build_test_router():
    from src.agent.router import RequestRouter

    with patch("src.agent.planner.get_chat_model", return_value=DummyLLM()):
        with patch(
            "src.agent.fast_intent.get_extractor_model", return_value=DummyLLM()
        ):
            return RequestRouter()


def memory_env():
    return {
        "PENDING_STORE": os.environ.get("PENDING_STORE"),
        "INTENT_ROUTING_MODE": os.environ.get("INTENT_ROUTING_MODE"),
    }
