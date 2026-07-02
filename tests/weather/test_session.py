import importlib.util
import json
import os
import sys
import unittest
from contextlib import ExitStack
from datetime import date as _date
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_MISSING_PYDANTIC_SETTINGS = importlib.util.find_spec("pydantic_settings") is None

if not _MISSING_PYDANTIC_SETTINGS:
    from tests.scenario_loader import load_yaml_scenarios
    from tests.support import build_test_router
    from src.agent.session_context import build_contextual_candidate


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class WeatherSessionContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = {
            "PENDING_STORE": os.environ.get("PENDING_STORE"),
            "INTENT_ROUTING_MODE": os.environ.get("INTENT_ROUTING_MODE"),
        }
        os.environ["PENDING_STORE"] = "memory"
        os.environ["INTENT_ROUTING_MODE"] = "hybrid"
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

    def _make_tool_invocation(self, payload: dict):
        from src.schemas.models import ToolInvocation

        return ToolInvocation(
            name=str(payload.get("name") or "weather_lookup"),
            message=str(payload.get("message") or ""),
            data=dict(payload.get("data") or {}),
        )

    def _build_fake_date(self, iso_date: str):
        year, month, day = (int(part) for part in iso_date.split("-"))

        class FakeDate(_date):
            @classmethod
            def today(cls):
                return cls(year, month, day)

        return FakeDate

    def test_weather_session_context_scenarios(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import UserRequest

        for scenario in load_yaml_scenarios("weather/session.yaml"):
            with self.subTest(scenario=scenario["id"]):
                router = build_test_router()
                session_id = scenario["session_id"]
                seed_plan = ActionPlan(
                    action="tool",
                    name="weather_lookup",
                    input=dict(scenario["seed_turn"]["plan_input"]),
                )
                seed_payload = self._make_tool_invocation(
                    scenario["seed_turn"]["tool_result"]
                )
                with patch.object(router._intent_router, "plan", return_value=seed_plan):
                    with patch(
                        "src.agent.router.execute_tool", return_value=seed_payload
                    ):
                        router.handle(
                            UserRequest(
                                prompt=scenario["seed_turn"]["prompt"],
                                session_id=session_id,
                            )
                        )

                followup_payload = self._make_tool_invocation(
                    scenario["followup_turn"]["tool_result"]
                )
                patches = []
                if scenario.get("today"):
                    fake_date = self._build_fake_date(scenario["today"])
                    patches.extend(
                        [
                            patch("src.application.services.weather_service.date", fake_date),
                            patch("src.agent.session_context.date", fake_date),
                        ]
                    )
                if scenario.get("force_rule_match"):
                    patches.append(
                        patch.object(router._rule_engine, "match", return_value=object())
                    )

                with patch.object(
                    router._intent_router, "plan", return_value=None
                ) as mocked_plan:
                    with patch(
                        "src.agent.router.execute_tool", return_value=followup_payload
                    ) as mocked_execute:
                        with ExitStack() as stack:
                            for item in patches:
                                stack.enter_context(item)
                            result = router.handle(
                                UserRequest(
                                    prompt=scenario["followup_turn"]["prompt"],
                                    session_id=session_id,
                                )
                            )

                self.assertEqual(result.mode, "tool")
                self.assertEqual(result.tool.name, "weather_lookup")
                if scenario["expected"].get("message_contains"):
                    self.assertIn(
                        scenario["expected"]["message_contains"], result.tool.message
                    )
                payload = json.loads(mocked_execute.call_args[0][1])
                for key, value in scenario["expected"]["payload"].items():
                    self.assertEqual(payload.get(key), value)

    def test_session_context_only_uses_last_context(self) -> None:
        from src.schemas.models import ToolInvocation, UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "s-last",
            {
                "tool_contexts": {
                    "sowing_suitability_lookup": {
                        "variety": "美香占2号",
                        "culti_type": "早稻",
                        "planting_method": "direct_seeding",
                        "region_id": "长沙",
                        "crop": "水稻",
                    },
                    "weather_lookup": {
                        "region": "长沙",
                        "start_date": "2026-03-25",
                        "end_date": "2026-03-31",
                        "granularity": "daily",
                    },
                },
                "last_context": {"kind": "tool", "name": "weather_lookup"},
            },
        )

        weather_payload = ToolInvocation(
            name="weather_lookup",
            message="ok",
            data={
                "region": "芜湖",
                "start_date": "2026-03-25",
                "end_date": "2026-03-31",
                "granularity": "daily",
                "requested_operations": ["施肥"],
                "points": [],
            },
        )
        with patch.object(
            router._intent_router, "plan", return_value=None
        ) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=weather_payload
            ) as mocked_execute:
                result = router.handle(
                    UserRequest(prompt="芜湖今天适合施肥吗", session_id="s-last")
                )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "weather_lookup")
        mocked_plan.assert_called_once()
        self.assertEqual(mocked_execute.call_args[0][0], "weather_lookup")

    def test_session_context_can_fallback_to_non_last_context_when_last_has_no_candidate(self) -> None:
        from src.schemas.models import ToolInvocation, UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "s-context-fallback",
            {
                "tool_contexts": {
                    "sowing_suitability_lookup": {
                        "variety": "美香占2号",
                        "culti_type": "早稻",
                        "planting_method": "direct_seeding",
                        "region_id": "长沙",
                        "crop": "水稻",
                    },
                    "weather_lookup": {
                        "region": "长沙",
                        "start_date": "2026-03-25",
                        "end_date": "2026-03-31",
                        "granularity": "daily",
                    },
                },
                "last_context": {"kind": "tool", "name": "weather_lookup"},
            },
        )

        tool_payload = ToolInvocation(
            name="sowing_suitability_lookup",
            message="ok",
            data={"resolved": {"region_id": "长沙", "variety": "美香占2号"}},
        )
        with patch.object(
            router._intent_router, "plan", return_value=None
        ) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=tool_payload
            ) as mocked_execute:
                result = router.handle(
                    UserRequest(prompt="那播种适宜期呢", session_id="s-context-fallback")
                )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "sowing_suitability_lookup")
        mocked_plan.assert_called_once()
        self.assertEqual(mocked_execute.call_args[0][0], "sowing_suitability_lookup")

    def test_sowing_session_context_reuses_farm_id_without_validation_followup(self) -> None:
        from src.schemas.models import ToolInvocation, UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "s-sowing-farm",
            {
                "tool_contexts": {
                    "sowing_suitability_lookup": {
                        "variety": "南粳46",
                        "culti_type": "中稻",
                        "planting_method": "直播",
                        "farm_id": 12,
                        "crop": "水稻",
                    }
                },
                "last_context": {"kind": "tool", "name": "sowing_suitability_lookup"},
            },
        )

        tool_payload = ToolInvocation(
            name="sowing_suitability_lookup",
            message="success",
            data={"resolved": {"farm_id": 12}},
        )
        with patch.object(
            router._intent_router, "plan", return_value=None
        ) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=tool_payload
            ) as mocked_execute:
                result = router.handle(
                    UserRequest(prompt="最近适合播种嘛", session_id="s-sowing-farm")
                )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "sowing_suitability_lookup")
        mocked_plan.assert_called_once()
        self.assertEqual(mocked_execute.call_args[0][0], "sowing_suitability_lookup")
        payload = json.loads(mocked_execute.call_args[0][1])
        self.assertEqual(payload.get("farm_id"), "12")
        self.assertEqual(payload.get("variety"), "南粳46")

    def test_extract_sowing_session_context_coerces_farm_id_to_string(self) -> None:
        from src.agent.session_context import extract_session_context_from_tool
        from src.schemas.models import ToolInvocation

        tool = ToolInvocation(
            name="sowing_suitability_lookup",
            message="success",
            data={"resolved": {"farm_id": 12, "variety": "南粳46"}},
        )

        name, context = extract_session_context_from_tool(tool) or (None, None)
        self.assertEqual(name, "sowing_suitability_lookup")
        self.assertEqual(context.get("farm_id"), "12")

    def test_weather_relative_followup_preserves_last_requested_operation(self) -> None:
        from src.schemas.models import UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "s-weather-op",
            {
                "tool_contexts": {
                    "weather_lookup": {
                        "region": "长沙",
                        "start_date": "2026-03-13",
                        "end_date": "2026-03-19",
                        "granularity": "daily",
                        "requested_operations": ["施肥"],
                    }
                },
                "last_context": {"kind": "tool", "name": "weather_lookup"},
            },
        )

        weather_payload = self._make_tool_invocation(
            {
                "message": "ok",
                "data": {
                    "region": "长沙",
                    "start_date": "2026-03-16",
                    "end_date": "2026-03-22",
                    "granularity": "daily",
                    "requested_operations": ["施肥"],
                    "points": [],
                },
            }
        )
        fake_date = self._build_fake_date("2026-03-13")
        with patch.object(
            router._intent_router, "plan", return_value=None
        ) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=weather_payload
            ) as mocked_execute:
                with patch("src.application.services.weather_service.date", fake_date):
                    with patch("src.agent.session_context.date", fake_date):
                        result = router.handle(
                            UserRequest(prompt="下周呢", session_id="s-weather-op")
                        )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "weather_lookup")
        mocked_plan.assert_called_once()
        payload = json.loads(mocked_execute.call_args[0][1])
        self.assertEqual(payload.get("region"), "长沙")
        self.assertEqual(payload.get("start_date"), "2026-03-16")
        self.assertEqual(payload.get("end_date"), "2026-03-22")
        self.assertEqual(payload.get("requested_operations"), ["施肥"])

    def test_weather_today_followup_overrides_stale_context_dates(self) -> None:
        from src.schemas.models import UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "s-weather-today",
            {
                "tool_contexts": {
                    "weather_lookup": {
                        "region": "长沙",
                        "start_date": "2024-06-01",
                        "end_date": "2024-06-07",
                        "granularity": "daily",
                        "requested_operations": ["施肥"],
                    }
                },
                "last_context": {"kind": "tool", "name": "weather_lookup"},
            },
        )

        weather_payload = self._make_tool_invocation(
            {
                "message": "ok",
                "data": {
                    "region": "长沙",
                    "start_date": "2026-03-13",
                    "end_date": "2026-03-13",
                    "granularity": "daily",
                    "requested_operations": ["施肥"],
                    "points": [],
                },
            }
        )
        fake_date = self._build_fake_date("2026-03-13")
        with patch.object(
            router._intent_router, "plan", return_value=None
        ) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=weather_payload
            ) as mocked_execute:
                with patch("src.application.services.weather_service.date", fake_date):
                    with patch("src.agent.session_context.date", fake_date):
                        result = router.handle(
                            UserRequest(prompt="今天适合施肥吗", session_id="s-weather-today")
                        )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "weather_lookup")
        mocked_plan.assert_called_once()
        payload = json.loads(mocked_execute.call_args[0][1])
        self.assertEqual(payload.get("region"), "长沙")
        self.assertEqual(payload.get("start_date"), "2026-03-13")
        self.assertEqual(payload.get("end_date"), "2026-03-13")
        self.assertEqual(payload.get("requested_operations"), ["施肥"])

    def test_weather_operation_followup_preserves_region_for_generic_fertilizer_phrase(
        self,
    ) -> None:
        from src.schemas.models import UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "s-weather-generic-op",
            {
                "tool_contexts": {
                    "weather_lookup": {
                        "region": "芜湖",
                        "start_date": "2026-03-23",
                        "end_date": "2026-03-29",
                        "granularity": "daily",
                        "requested_operations": ["施肥"],
                    }
                },
                "last_context": {"kind": "tool", "name": "weather_lookup"},
            },
        )

        weather_payload = self._make_tool_invocation(
            {
                "message": "ok",
                "data": {
                    "region": "芜湖",
                    "start_date": "2026-03-23",
                    "end_date": "2026-03-29",
                    "granularity": "daily",
                    "requested_operations": ["施肥"],
                    "points": [],
                },
            }
        )
        with patch.object(
            router._intent_router, "plan", return_value=None
        ) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=weather_payload
            ) as mocked_execute:
                result = router.handle(
                    UserRequest(prompt="施穗肥呢", session_id="s-weather-generic-op")
                )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "weather_lookup")
        mocked_plan.assert_called_once()
        payload = json.loads(mocked_execute.call_args[0][1])
        self.assertEqual(payload.get("region"), "芜湖")
        self.assertEqual(payload.get("start_date"), "2026-03-23")
        self.assertEqual(payload.get("end_date"), "2026-03-29")
        self.assertEqual(payload.get("requested_operations"), ["施肥"])

    def test_session_resolver_prefers_contextual_weather_slots_for_same_tool(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "s-weather-resolver",
            {
                "tool_contexts": {
                    "weather_lookup": {
                        "region": "长沙",
                        "start_date": "2024-06-01",
                        "end_date": "2024-06-07",
                        "granularity": "daily",
                    }
                },
                "last_context": {"kind": "tool", "name": "weather_lookup"},
            },
        )

        standalone_plan = ActionPlan(
            action="tool",
            name="weather_lookup",
            input={
                "start_date": "2026-03-13",
                "end_date": "2026-03-13",
                "year": 2026,
            },
            reason="standalone:weather",
        )
        weather_payload = self._make_tool_invocation(
            {
                "message": "ok",
                "data": {
                    "region": "长沙",
                    "start_date": "2026-03-13",
                    "end_date": "2026-03-13",
                    "granularity": "daily",
                    "points": [],
                },
            }
        )
        fake_date = self._build_fake_date("2026-03-13")
        with patch.object(
            router._intent_router, "plan", return_value=standalone_plan
        ) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=weather_payload
            ) as mocked_execute:
                with patch("src.application.services.weather_service.date", fake_date):
                    with patch("src.agent.session_context.date", fake_date):
                        result = router.handle(
                            UserRequest(
                                prompt="今天的情况请帮我详细看一下",
                                session_id="s-weather-resolver",
                            )
                        )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "weather_lookup")
        mocked_plan.assert_called_once()
        payload = json.loads(mocked_execute.call_args[0][1])
        self.assertEqual(payload.get("region"), "长沙")
        self.assertEqual(payload.get("start_date"), "2026-03-13")
        self.assertEqual(payload.get("end_date"), "2026-03-13")
        self.assertEqual(payload.get("requested_operations"), [])

    def test_weather_followup_with_prefixed_region_overrides_default_farm(self) -> None:
        from src.schemas.models import UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "s-weather-region-override",
            {
                "tool_contexts": {
                    "weather_lookup": {
                        "start_date": "2026-03-16",
                        "end_date": "2026-03-16",
                        "granularity": "daily",
                        "requested_operations": ["施肥"],
                    }
                },
                "last_context": {"kind": "tool", "name": "weather_lookup"},
            },
        )

        weather_payload = self._make_tool_invocation(
            {
                "message": "ok",
                "data": {
                    "region": "芜湖",
                    "start_date": "2026-03-16",
                    "end_date": "2026-03-16",
                    "granularity": "daily",
                    "requested_operations": ["施肥"],
                    "points": [],
                },
            }
        )
        fake_date = self._build_fake_date("2026-03-16")
        with patch.object(
            router._intent_router, "plan", return_value=None
        ) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=weather_payload
            ) as mocked_execute:
                with patch("src.application.services.weather_service.date", fake_date):
                    with patch("src.agent.session_context.date", fake_date):
                        result = router.handle(
                            UserRequest(
                                prompt="芜湖今天可以施肥吗",
                                session_id="s-weather-region-override",
                            )
                        )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "weather_lookup")
        mocked_plan.assert_called_once()
        payload = json.loads(mocked_execute.call_args[0][1])
        self.assertEqual(payload.get("region"), "芜湖")
        self.assertEqual(payload.get("start_date"), "2026-03-16")
        self.assertEqual(payload.get("end_date"), "2026-03-16")
        self.assertEqual(payload.get("requested_operations"), ["施肥"])

    def test_session_context_trace_annotations_are_emitted(self) -> None:
        from src.schemas.models import UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "s-session-trace",
            {
                "tool_contexts": {
                    "weather_lookup": {
                        "region": "长沙",
                        "start_date": "2026-03-16",
                        "end_date": "2026-03-16",
                        "granularity": "daily",
                        "requested_operations": ["施肥"],
                    }
                },
                "last_context": {"kind": "tool", "name": "weather_lookup"},
            },
        )

        weather_payload = self._make_tool_invocation(
            {
                "message": "ok",
                "data": {
                    "region": "长沙",
                    "start_date": "2026-03-16",
                    "end_date": "2026-03-16",
                    "granularity": "daily",
                    "requested_operations": ["施肥"],
                    "points": [],
                },
            }
        )
        fake_date = self._build_fake_date("2026-03-16")
        with patch("src.agent.router.annotate_current_span") as mocked_annotate:
            with patch(
                "src.agent.router.execute_tool", return_value=weather_payload
            ):
                with patch("src.application.services.weather_service.date", fake_date):
                    with patch("src.agent.session_context.date", fake_date):
                        router.handle(
                            UserRequest(
                                prompt="今天适合施肥吗",
                                session_id="s-session-trace",
                            )
                        )

        self.assertGreaterEqual(mocked_annotate.call_count, 2)

    def test_weather_relative_two_weeks_followup_uses_correct_week(self) -> None:
        from src.schemas.models import UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "s-weather-two-weeks",
            {
                "tool_contexts": {
                    "weather_lookup": {
                        "region": "长沙",
                        "start_date": "2026-03-13",
                        "end_date": "2026-03-19",
                        "granularity": "daily",
                        "requested_operations": ["施肥"],
                    }
                },
                "last_context": {"kind": "tool", "name": "weather_lookup"},
            },
        )

        weather_payload = self._make_tool_invocation(
            {
                "message": "ok",
                "data": {
                    "region": "长沙",
                    "start_date": "2026-03-23",
                    "end_date": "2026-03-29",
                    "granularity": "daily",
                    "requested_operations": ["施肥"],
                    "points": [],
                },
            }
        )
        fake_date = self._build_fake_date("2026-03-13")
        with patch.object(
            router._intent_router, "plan", return_value=None
        ) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=weather_payload
            ) as mocked_execute:
                with patch("src.application.services.weather_service.date", fake_date):
                    with patch("src.agent.session_context.date", fake_date):
                        result = router.handle(
                            UserRequest(
                                prompt="下下周呢", session_id="s-weather-two-weeks"
                            )
                        )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "weather_lookup")
        mocked_plan.assert_called_once()
        payload = json.loads(mocked_execute.call_args[0][1])
        self.assertEqual(payload.get("region"), "长沙")
        self.assertEqual(payload.get("start_date"), "2026-03-23")
        self.assertEqual(payload.get("end_date"), "2026-03-29")
        self.assertEqual(payload.get("requested_operations"), ["施肥"])

    def test_weather_question_after_sowing_context_does_not_get_hijacked(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import ToolInvocation, UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "s-sowing-to-weather",
            {
                "tool_contexts": {
                    "sowing_suitability_lookup": {
                        "variety": "湘早籼24号",
                        "culti_type": "早稻",
                        "planting_method": "移栽",
                        "region_id": "湖南常德",
                        "crop": "水稻",
                    }
                },
                "last_context": {"kind": "tool", "name": "sowing_suitability_lookup"},
            },
        )

        weather_plan = ActionPlan(
            action="tool",
            name="weather_lookup",
            input={
                "region": "湖南常德",
                "start_date": "2026-03-13",
                "end_date": "2026-03-13",
                "year": 2026,
                "requested_operations": ["施肥"],
            },
        )
        weather_payload = ToolInvocation(
            name="weather_lookup",
            message="ok",
            data={
                "region": "湖南常德",
                "start_date": "2026-03-13",
                "end_date": "2026-03-13",
                "requested_operations": ["施肥"],
                "points": [],
            },
        )
        with patch.object(router._intent_router, "plan", return_value=weather_plan) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=weather_payload
            ) as mocked_execute:
                result = router.handle(
                    UserRequest(prompt="今天适合施肥吗", session_id="s-sowing-to-weather")
                )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "weather_lookup")
        mocked_plan.assert_called_once()
        self.assertEqual(mocked_execute.call_args[0][0], "weather_lookup")

    def test_pending_resumed_sowing_tool_updates_session_context_for_followup(self) -> None:
        from src.schemas.models import ToolInvocation, UserRequest

        router = build_test_router()
        session_id = "s-pending-sowing-context"
        router._session_context_store.set(
            session_id,
            {
                "tool_contexts": {
                    "weather_lookup": {
                        "region": "芜湖",
                        "start_date": "2026-03-17",
                        "end_date": "2026-03-17",
                        "granularity": "daily",
                        "requested_operations": ["打药"],
                    }
                },
                "last_context": {"kind": "tool", "name": "weather_lookup"},
            },
        )
        router._pending_store.set(
            session_id,
            {
                "mode": "tool",
                "tool_name": "sowing_suitability_lookup",
                "query": "适合种美香占2号吗",
                "draft": {"variety": "美香占2号"},
                "missing_fields": ["culti_type", "planting_method"],
                "followup_count": 0,
            },
        )

        first_payload = ToolInvocation(
            name="sowing_suitability_lookup",
            message="success",
            data={
                "resolved": {
                    "variety": "美香占2号",
                    "culti_type": "早稻",
                    "planting_method": "direct_seeding",
                    "region_id": "芜湖",
                    "crop": "水稻",
                }
            },
        )
        second_payload = ToolInvocation(
            name="sowing_suitability_lookup",
            message="success",
            data={
                "resolved": {
                    "variety": "美香占2号",
                    "culti_type": "早稻",
                    "planting_method": "direct_seeding",
                    "region_id": "常德",
                    "crop": "水稻",
                }
            },
        )
        with patch.object(
            router._intent_router, "plan", return_value=None
        ) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool",
                side_effect=[first_payload, second_payload],
            ) as mocked_execute:
                first_result = router.handle(
                    UserRequest(prompt="早稻，直播", session_id=session_id)
                )
                second_result = router.handle(
                    UserRequest(prompt="常德呢", session_id=session_id)
                )

        self.assertEqual(first_result.mode, "tool")
        self.assertEqual(first_result.tool.name, "sowing_suitability_lookup")
        self.assertEqual(second_result.mode, "tool")
        self.assertEqual(second_result.tool.name, "sowing_suitability_lookup")
        self.assertGreaterEqual(mocked_plan.call_count, 1)
        self.assertEqual(mocked_execute.call_args_list[0][0][0], "sowing_suitability_lookup")
        self.assertEqual(mocked_execute.call_args_list[1][0][0], "sowing_suitability_lookup")
        second_call_payload = json.loads(mocked_execute.call_args_list[1][0][1])
        self.assertEqual(second_call_payload.get("region_id"), "常德")
        session_payload = router._session_context_store.get(session_id) or {}
        self.assertEqual(
            session_payload.get("last_context"),
            {"kind": "tool", "name": "sowing_suitability_lookup"},
        )

    def test_full_sowing_question_does_not_get_hijacked_by_weather_context(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import ToolInvocation, UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "s-weather-to-sowing",
            {
                "tool_contexts": {
                    "weather_lookup": {
                        "region": "长沙",
                        "start_date": "2026-03-16",
                        "end_date": "2026-03-22",
                        "granularity": "daily",
                        "requested_operations": ["施肥"],
                    }
                },
                "last_context": {"kind": "tool", "name": "weather_lookup"},
            },
        )

        sowing_payload = ToolInvocation(
            name="sowing_suitability_lookup",
            message="ok",
            data={"resolved": {"variety": "美香占2号", "region_id": "常德"}},
        )
        sowing_plan = ActionPlan(
            action="tool",
            name="sowing_suitability_lookup",
            input={"query": "美香占2号在常德种什么时候播种"},
            reason="standalone:sowing",
        )
        with patch.object(router._intent_router, "plan", return_value=sowing_plan):
            with patch(
                "src.agent.router.execute_tool", return_value=sowing_payload
            ) as mocked_execute:
                result = router.handle(
                    UserRequest(prompt="美香占2号在常德种什么时候播种", session_id="s-weather-to-sowing")
                )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "sowing_suitability_lookup")
        self.assertEqual(mocked_execute.call_args[0][0], "sowing_suitability_lookup")
        payload = json.loads(mocked_execute.call_args[0][1])
        self.assertEqual(payload.get("query"), "美香占2号在常德种什么时候播种")

    def test_plain_sowing_followup_routes_to_sowing_tool_not_weather(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import ToolInvocation, UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "s-weather-to-plain-sowing",
            {
                "tool_contexts": {
                    "weather_lookup": {
                        "region": "长沙",
                        "start_date": "2026-03-16",
                        "end_date": "2026-03-22",
                        "granularity": "daily",
                        "requested_operations": ["施肥"],
                    }
                },
                "last_context": {"kind": "tool", "name": "weather_lookup"},
            },
        )

        sowing_payload = ToolInvocation(
            name="sowing_suitability_lookup",
            message="请补充品种、稻作类型、播种方式和区域，我才能给出播期推荐。",
            data={},
        )
        sowing_plan = ActionPlan(
            action="tool",
            name="sowing_suitability_lookup",
            input={"query": "那播种怎么样"},
            reason="standalone:sowing",
        )
        with patch.object(router._intent_router, "plan", return_value=sowing_plan):
            with patch(
                "src.agent.router.execute_tool", return_value=sowing_payload
            ) as mocked_execute:
                result = router.handle(
                    UserRequest(prompt="那播种怎么样", session_id="s-weather-to-plain-sowing")
                )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "sowing_suitability_lookup")
        self.assertEqual(mocked_execute.call_args[0][0], "sowing_suitability_lookup")
        payload = json.loads(mocked_execute.call_args[0][1])
        self.assertEqual(payload.get("query"), "那播种怎么样")

    def test_transplanting_sowing_question_does_not_get_hijacked_by_weather_context(
        self,
    ) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import ToolInvocation, UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "s-weather-to-transplant-sowing",
            {
                "tool_contexts": {
                    "weather_lookup": {
                        "region": "湖南常德",
                        "start_date": "2024-06-01",
                        "end_date": "2024-06-30",
                        "granularity": "daily",
                        "requested_operations": ["移栽"],
                    }
                },
                "last_context": {"kind": "tool", "name": "weather_lookup"},
            },
        )

        sowing_payload = ToolInvocation(
            name="sowing_suitability_lookup",
            message="ok",
            data={"resolved": {"variety": "湘早籼24", "region_id": "湖南常德"}},
        )
        sowing_plan = ActionPlan(
            action="tool",
            name="sowing_suitability_lookup",
            input={"query": "我在常德种植早稻湘早籼24，移栽什么时候播种合适"},
            reason="standalone:sowing",
        )
        with patch.object(router._intent_router, "plan", return_value=sowing_plan):
            with patch(
                "src.agent.router.execute_tool", return_value=sowing_payload
            ) as mocked_execute:
                result = router.handle(
                    UserRequest(
                        prompt="我在常德种植早稻湘早籼24，移栽什么时候播种合适",
                        session_id="s-weather-to-transplant-sowing",
                    )
                )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "sowing_suitability_lookup")
        self.assertEqual(mocked_execute.call_args[0][0], "sowing_suitability_lookup")
        payload = json.loads(mocked_execute.call_args[0][1])
        self.assertEqual(
            payload.get("query"), "我在常德种植早稻湘早籼24，移栽什么时候播种合适"
        )

    def test_crop_calendar_plan_prompt_does_not_build_weather_contextual_candidate(
        self,
    ) -> None:
        payload = {
            "tool_contexts": {
                "weather_lookup": {
                    "region": "湖南常德",
                    "start_date": "2026-03-25",
                    "end_date": "2026-03-31",
                    "granularity": "daily",
                    "requested_operations": ["移栽"],
                }
            },
            "last_context": {"kind": "tool", "name": "weather_lookup"},
        }

        candidate = build_contextual_candidate(
            "我想建立一个在湖南常德种植的湘早籼24号的移栽方案",
            payload,
        )

        self.assertIsNone(candidate)

    def test_session_context_adapter_registry_covers_current_tools_and_workflow(self) -> None:
        from src.agent.session_context import get_session_context_adapter

        self.assertIsNotNone(get_session_context_adapter("tool", "weather_lookup"))
        self.assertIsNotNone(get_session_context_adapter("tool", "variety_lookup"))
        self.assertIsNotNone(
            get_session_context_adapter("tool", "sowing_suitability_lookup")
        )
        self.assertIsNotNone(
            get_session_context_adapter("tool", "plant_plan_list_active")
        )
        self.assertIsNotNone(get_session_context_adapter("tool", "plant_plan_delete"))
        self.assertIsNotNone(get_session_context_adapter("tool", "plant_task_create"))
        self.assertIsNotNone(get_session_context_adapter("tool", "growth_stage_lookup"))
        self.assertIsNotNone(
            get_session_context_adapter("workflow", "crop_calendar_workflow")
        )

    def test_recent_sowing_suitability_question_routes_to_sowing_tool(self) -> None:
        from src.schemas.models import ToolInvocation, UserRequest

        router = build_test_router()
        sowing_payload = ToolInvocation(
            name="sowing_suitability_lookup",
            message="请补充品种、稻作类型、播种方式和区域，我才能给出播期推荐。",
            data={},
        )
        with patch(
            "src.agent.router.execute_tool", return_value=sowing_payload
        ) as mocked_execute:
            result = router.handle(
                UserRequest(prompt="最近适合播种嘛", session_id="s-recent-sowing")
            )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "sowing_suitability_lookup")
        self.assertEqual(mocked_execute.call_args[0][0], "sowing_suitability_lookup")
        payload = json.loads(mocked_execute.call_args[0][1])
        self.assertEqual(payload.get("query"), "最近适合播种嘛")


if __name__ == "__main__":
    unittest.main()
