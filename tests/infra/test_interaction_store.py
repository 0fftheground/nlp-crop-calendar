import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_MISSING_PYDANTIC_SETTINGS = importlib.util.find_spec("pydantic_settings") is None


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class InteractionStoreLineageTests(unittest.TestCase):
    def tearDown(self) -> None:
        from src.observability.interaction_context import (
            reset_interaction_context,
            set_interaction_context,
        )

        token = set_interaction_context({})
        reset_interaction_context(token)

    def test_memory_store_records_request_and_thread_ids(self) -> None:
        from src.infra.interaction_store import MemoryInteractionStore
        from src.observability.interaction_context import set_interaction_context
        from src.schemas.models import HandleResponse, UserRequest, WorkflowResponse

        store = MemoryInteractionStore(max_items=10)
        token = set_interaction_context(
            {
                "request_id": "req-1",
                "thread_id": "thread-1",
                "parent_interaction_id": None,
                "continuity_type": "standalone",
                "continuity_source": "none",
                "dialogue_act": "start_new_task",
                "task_type": "none",
            }
        )
        try:
            store.record(
                UserRequest(prompt="你能提供哪些功能", session_id="s1"),
                HandleResponse(mode="none", plan=WorkflowResponse(message="ok")),
                12,
            )
        finally:
            from src.observability.interaction_context import reset_interaction_context

            reset_interaction_context(token)

        latest = store.get_latest_session_lineage("s1")
        self.assertIsNotNone(latest)
        self.assertEqual(latest["interaction_id"], 1)
        self.assertEqual(latest["thread_id"], "thread-1")
        self.assertEqual(store._items[-1]["request"]["dialogue_act"], "start_new_task")
        self.assertEqual(store._items[-1]["request"]["task_type"], "none")

    def test_memory_store_records_parent_interaction_for_continued_thread(self) -> None:
        from src.infra.interaction_store import MemoryInteractionStore
        from src.observability.interaction_context import set_interaction_context
        from src.schemas.models import HandleResponse, ToolInvocation, UserRequest

        store = MemoryInteractionStore(max_items=10)

        token = set_interaction_context(
            {
                "request_id": "req-1",
                "thread_id": "thread-1",
                "parent_interaction_id": None,
                "continuity_type": "standalone",
                "continuity_source": "none",
                "dialogue_act": "start_new_task",
                "task_type": "variety_lookup",
            }
        )
        try:
            store.record(
                UserRequest(prompt="帮我创建一个种植计划", session_id="s2"),
                HandleResponse(
                    mode="tool",
                    tool=ToolInvocation(name="variety_lookup", message="ok", data={}),
                ),
                10,
            )
        finally:
            from src.observability.interaction_context import reset_interaction_context

            reset_interaction_context(token)

        latest = store.get_latest_session_lineage("s2")
        token = set_interaction_context(
            {
                "request_id": "req-2",
                "thread_id": latest["thread_id"],
                "parent_interaction_id": latest["interaction_id"],
                "continuity_type": "pending_resume",
                "continuity_source": "pending",
                "dialogue_act": "update_fields",
                "task_type": "variety_lookup",
            }
        )
        try:
            store.record(
                UserRequest(prompt="常德", session_id="s2"),
                HandleResponse(
                    mode="tool",
                    tool=ToolInvocation(name="variety_lookup", message="ok", data={}),
                ),
                11,
            )
        finally:
            from src.observability.interaction_context import reset_interaction_context

            reset_interaction_context(token)

        item = store._items[-1]
        self.assertEqual(item["request"]["thread_id"], "thread-1")
        self.assertEqual(item["request"]["parent_interaction_id"], 1)
        self.assertEqual(item["request"]["continuity_type"], "pending_resume")
        self.assertEqual(item["request"]["dialogue_act"], "update_fields")
        self.assertEqual(item["request"]["task_type"], "variety_lookup")

    def test_row_value_supports_mapping_rows(self) -> None:
        from src.infra.interaction_store import _row_value

        row = {
            "id": 12,
            "request_id": "req-12",
            "thread_id": "thread-9",
            "parent_interaction_id": 11,
        }

        self.assertEqual(_row_value(row, "id"), 12)
        self.assertEqual(_row_value(row, "request_id"), "req-12")
        self.assertEqual(_row_value(row, 0, "fallback"), "fallback")
