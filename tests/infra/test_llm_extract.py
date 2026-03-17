import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_MISSING_PYDANTIC_SETTINGS = importlib.util.find_spec("pydantic_settings") is None


class _ExtractSchema(BaseModel):
    value: str | None = None


class _FailingExtractorModel:
    model_name = "fake-extractor"

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        raise RuntimeError("extract boom")


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class LlmExtractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = {
            "LLM_PROVIDER": os.environ.get("LLM_PROVIDER"),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "OPENAI_API_BASE": os.environ.get("OPENAI_API_BASE"),
            "EXTRACTOR_PROVIDER": os.environ.get("EXTRACTOR_PROVIDER"),
            "EXTRACTOR_MODEL": os.environ.get("EXTRACTOR_MODEL"),
            "EXTRACTOR_API_KEY": os.environ.get("EXTRACTOR_API_KEY"),
            "EXTRACTOR_API_BASE": os.environ.get("EXTRACTOR_API_BASE"),
            "BACKEND_TIMEOUT_SECONDS": os.environ.get("BACKEND_TIMEOUT_SECONDS"),
        }
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["EXTRACTOR_PROVIDER"] = "openai"
        os.environ["EXTRACTOR_MODEL"] = "gpt-4.1-mini"
        os.environ["BACKEND_TIMEOUT_SECONDS"] = "17"
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

    def test_get_extractor_model_passes_backend_timeout(self) -> None:
        from src.infra.llm import get_extractor_model

        with patch("src.infra.llm.ChatOpenAI", return_value=object()) as mocked_chat:
            get_extractor_model()

        kwargs = mocked_chat.call_args.kwargs
        self.assertEqual(kwargs.get("timeout"), 17)

    def test_llm_structured_extract_logs_error_when_invoke_fails(self) -> None:
        from src.infra.llm_extract import llm_structured_extract

        with patch(
            "src.infra.llm_extract.get_extractor_model",
            return_value=_FailingExtractorModel(),
        ):
            with patch("src.infra.llm_extract.log_event") as mocked_log:
                result = llm_structured_extract(
                    "美香占2号，直播",
                    schema=_ExtractSchema,
                    system_prompt="extract",
                )

        self.assertEqual(result, {})
        event_names = [call.args[0] for call in mocked_log.call_args_list]
        self.assertIn("llm_extract_call", event_names)
        self.assertIn("llm_extract_error", event_names)


if __name__ == "__main__":
    unittest.main()
