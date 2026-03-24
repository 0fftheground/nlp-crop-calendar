import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_MISSING_PYDANTIC_SETTINGS = importlib.util.find_spec("pydantic_settings") is None


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class LlmConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = {
            "LLM_PROVIDER": os.environ.get("LLM_PROVIDER"),
            "LLM_MODEL": os.environ.get("LLM_MODEL"),
            "AUDIT_JUDGE_MODEL": os.environ.get("AUDIT_JUDGE_MODEL"),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "OPENAI_API_BASE": os.environ.get("OPENAI_API_BASE"),
            "BACKEND_TIMEOUT_SECONDS": os.environ.get("BACKEND_TIMEOUT_SECONDS"),
        }
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["LLM_MODEL"] = "gpt-test-chat"
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["BACKEND_TIMEOUT_SECONDS"] = "19"
        from src.infra.config import get_config

        get_config.cache_clear()

    def tearDown(self) -> None:
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        from src.infra.config import get_config

        get_config.cache_clear()

    def test_get_chat_model_uses_configured_model_and_timeout(self) -> None:
        from src.infra.llm import get_chat_model

        with patch("src.infra.llm.ChatOpenAI", return_value=object()) as mocked_chat:
            get_chat_model()

        kwargs = mocked_chat.call_args.kwargs
        self.assertEqual(kwargs.get("model"), "gpt-test-chat")
        self.assertEqual(kwargs.get("timeout"), 19)

    def test_get_audit_judge_model_uses_dedicated_model_when_configured(self) -> None:
        from src.infra.config import get_config
        from src.infra.llm import get_audit_judge_model

        os.environ["AUDIT_JUDGE_MODEL"] = "gpt-test-judge"
        get_config.cache_clear()

        with patch("src.infra.llm.ChatOpenAI", return_value=object()) as mocked_chat:
            get_audit_judge_model()

        kwargs = mocked_chat.call_args.kwargs
        self.assertEqual(kwargs.get("model"), "gpt-test-judge")
        self.assertEqual(kwargs.get("timeout"), 19)
        self.assertEqual(kwargs.get("temperature"), 0.0)

    def test_get_audit_judge_model_falls_back_to_llm_model(self) -> None:
        from src.infra.llm import get_audit_judge_model

        with patch("src.infra.llm.ChatOpenAI", return_value=object()) as mocked_chat:
            get_audit_judge_model()

        kwargs = mocked_chat.call_args.kwargs
        self.assertEqual(kwargs.get("model"), "gpt-test-chat")
