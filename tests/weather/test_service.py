import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_MISSING_PYDANTIC_SETTINGS = importlib.util.find_spec("pydantic_settings") is None

if not _MISSING_PYDANTIC_SETTINGS:
    from tests.scenario_loader import load_yaml_scenarios
    from src.application.adapters import (
        DEFAULT_CONFIG_ADAPTER,
        DEFAULT_HTTP_ADAPTER,
        DEFAULT_SQL_ADAPTER,
    )
    from src.application.services.weather_service import (
        configure_weather_ports,
        lookup_weather,
        normalize_weather_prompt,
    )


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class WeatherServiceTests(unittest.TestCase):
    def tearDown(self) -> None:
        configure_weather_ports(
            config_port=DEFAULT_CONFIG_ADAPTER,
            http_port=DEFAULT_HTTP_ADAPTER,
            sql_port=DEFAULT_SQL_ADAPTER,
        )

    def _configure_stub_weather_ports(self, payload: dict) -> None:
        class StubConfig:
            agri_db_url = "postgresql://example"
            business_api_key = None
            business_api_base_url = "http://example.test"
            farm_weather_api_url = None
            default_farm_id = "9"
            db_region_lookup_candidates = []
            region_db_table = None

        class StubResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self):
                return payload

        class StubHttp:
            def post(self, url, *, json_payload, headers=None, timeout=10.0):
                return StubResponse()

        class StubSql:
            def fetch_all(self, url, sql, params=()):
                return []

            def quote_identifier(self, name: str) -> str:
                return f'"{name}"'

        configure_weather_ports(
            config_port=type("P", (), {"get": staticmethod(lambda: StubConfig())})(),
            http_port=StubHttp(),
            sql_port=StubSql(),
        )

    def test_lookup_weather_uses_suitability_api_with_region_id(self) -> None:
        class StubConfig:
            agri_db_url = "postgresql://example"
            business_api_key = None
            business_api_base_url = "http://example.test"
            farm_weather_api_url = None
            default_farm_id = "9"
            db_region_lookup_candidates = []
            region_db_table = None

        class StubResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self):
                return self._payload

        class StubHttp:
            def __init__(self) -> None:
                self.calls = []

            def get(self, *args, **kwargs):
                raise AssertionError("weather service should use POST")

            def post(self, url, *, json_payload, headers=None, timeout=10.0):
                self.calls.append(
                    {
                        "url": url,
                        "json_payload": json_payload,
                        "headers": headers,
                        "timeout": timeout,
                    }
                )
                return StubResponse(
                    {
                        "code": 200,
                        "message": "成功获取20250101至20250103数据，共3条。",
                        "data": [
                            {
                                "date": "2025-01-01",
                                "tmax": 20.0,
                                "tmin": 10.0,
                                "tavg": 15.0,
                                "wins": 3.5,
                                "pre": 1.2,
                                "rh": 80.0,
                                "sf_ws": 0.6,
                                "sf_reason": "有小雨，不建议施肥。",
                                "lm_ws": 0.5,
                                "lm_reason": "适合炼苗。",
                            },
                            {
                                "date": "2025-01-02",
                                "tmax": 21.0,
                                "tmin": 11.0,
                                "tavg": 16.0,
                                "wins": 3.0,
                                "pre": 0.0,
                                "rh": 78.0,
                            },
                            {
                                "date": "2025-01-03",
                                "tmax": 22.0,
                                "tmin": 12.0,
                                "tavg": 17.0,
                                "wins": 2.8,
                                "pre": 0.5,
                                "rh": 76.0,
                            },
                        ],
                    }
                )

        class StubSql:
            def fetch_all(self, url, sql, params=()):
                return [{"region_id": 430100, "region_name": "长沙市"}]

            def quote_identifier(self, name: str) -> str:
                return f'"{name}"'

        http = StubHttp()
        configure_weather_ports(
            config_port=type("P", (), {"get": staticmethod(lambda: StubConfig())})(),
            http_port=http,
            sql_port=StubSql(),
        )

        result = lookup_weather(
            '{"region":"长沙","start_date":"2025-01-01","end_date":"2025-01-03"}'
        )

        self.assertEqual(result.name, "weather_lookup")
        self.assertEqual(result.message, "成功获取20250101至20250103数据，共3条。")
        self.assertEqual(len(http.calls), 1)
        self.assertEqual(http.calls[0]["url"], "http://example.test/suit_rili")
        self.assertEqual(
            http.calls[0]["json_payload"],
            {
                "region_id": "430100",
                "start_date": "20250101",
                "end_date": "20250103",
            },
        )
        data = result.data
        self.assertEqual(data.get("region"), "长沙")
        self.assertEqual(data.get("source"), "agri_weather_api")
        points = data.get("points") or []
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0].get("sf_ws"), 0.6)
        self.assertEqual(points[0].get("sf_reason"), "有小雨，不建议施肥。")
        self.assertEqual(points[0].get("lm_ws"), 0.5)
        self.assertEqual(points[0].get("lm_reason"), "适合炼苗。")

    def test_lookup_weather_uses_default_farm_when_region_missing(self) -> None:
        class StubConfig:
            agri_db_url = "postgresql://example"
            business_api_key = None
            business_api_base_url = "http://example.test"
            farm_weather_api_url = None
            default_farm_id = "12"
            db_region_lookup_candidates = []
            region_db_table = None

        class StubResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self):
                return {
                    "code": 200,
                    "message": "success",
                    "data": [
                        {
                            "date": "2025-01-01",
                            "tmax": 20.0,
                            "tmin": 10.0,
                            "tavg": 15.0,
                            "wins": 3.5,
                            "pre": 1.2,
                            "rh": 80.0,
                        }
                    ],
                }

        class StubHttp:
            def __init__(self) -> None:
                self.last_payload = None

            def get(self, *args, **kwargs):
                raise AssertionError("weather service should use POST")

            def post(self, url, *, json_payload, headers=None, timeout=10.0):
                self.last_payload = json_payload
                return StubResponse()

        class StubSql:
            def fetch_all(self, url, sql, params=()):
                return []

            def quote_identifier(self, name: str) -> str:
                return f'"{name}"'

        http = StubHttp()
        configure_weather_ports(
            config_port=type("P", (), {"get": staticmethod(lambda: StubConfig())})(),
            http_port=http,
            sql_port=StubSql(),
        )

    def test_lookup_weather_uses_requested_operations_from_query_payload(self) -> None:
        self._configure_stub_weather_ports(
            {
                "code": 200,
                "message": "success",
                "data": [
                    {
                        "date": "2026-03-16",
                        "tmax": 12.0,
                        "tmin": 7.0,
                        "sf_ws": 0.2,
                        "sf_reason": "降雨明显且气温偏低，不适合施肥。",
                        "dy_ws": 0.9,
                        "dy_reason": "适合打药。",
                    }
                ],
            }
        )

        result = lookup_weather(
            json.dumps(
                {
                    "query": "下周呢",
                    "start_date": "2026-03-16",
                    "end_date": "2026-03-22",
                    "requested_operations": ["施肥"],
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(result.data.get("requested_operations"), ["施肥"])
        points = result.data.get("points") or []
        self.assertEqual(points[0].get("sf_ws"), 0.2)
        self.assertNotIn("dy_ws", points[0])

        result = lookup_weather('{"start_date":"2025-01-01","end_date":"2025-01-01"}')

        self.assertEqual(
            http.last_payload,
            {
                "farm_id": "12",
                "start_date": "20250101",
                "end_date": "20250101",
            },
        )
        self.assertEqual(result.data.get("region"), "farm:12")

    def test_normalize_weather_prompt_scenario_ranges(self) -> None:
        from datetime import date as _date

        class FakeDate(_date):
            @classmethod
            def today(cls):
                return cls(2026, 3, 13)

        scenarios = load_yaml_scenarios("weather/service.yaml")
        relative_dates = scenarios["relative_dates"]
        with patch("src.application.services.weather_service.date", FakeDate):
            for case in relative_dates["cases"]:
                with self.subTest(prompt=case["prompt"]):
                    payload = json.dumps({"query": case["prompt"]}, ensure_ascii=False)
                    _, query = normalize_weather_prompt(payload)
                    self.assertIsNotNone(query)
                    self.assertEqual(query.start_date.isoformat(), case["start_date"])
                    self.assertEqual(query.end_date.isoformat(), case["end_date"])

    def test_lookup_weather_operation_scenarios(self) -> None:
        scenarios = load_yaml_scenarios("weather/service.yaml")
        base_payload = scenarios["operation_cases"]["base_payload"]
        response_payload = {
            "code": 200,
            "message": "success",
            "data": [dict(base_payload["points"][0])],
        }
        self._configure_stub_weather_ports(response_payload)

        for group_name in ("direct", "followup"):
            for case in scenarios["operation_cases"][group_name]:
                with self.subTest(group=group_name, query=case["query"]):
                    payload = {
                        "query": case["query"],
                        "start_date": base_payload["start_date"],
                        "end_date": base_payload["end_date"],
                    }
                    result = lookup_weather(json.dumps(payload, ensure_ascii=False))
                    self.assertEqual(
                        result.data.get("requested_operations"),
                        case["requested_operations"],
                    )
                    point = (result.data.get("points") or [{}])[0]
                    for field in case.get("present_fields", []):
                        self.assertIn(field, point)
                    for field in case.get("absent_fields", []):
                        self.assertNotIn(field, point)

    def test_lookup_weather_unsupported_operation_scenarios(self) -> None:
        scenarios = load_yaml_scenarios("weather/service.yaml")
        for case in scenarios["unsupported_cases"]:
            with self.subTest(query=case["query"]):
                payload = {"query": case["query"]}
                if "start_date" in case:
                    payload["start_date"] = case["start_date"]
                    payload["end_date"] = case["end_date"]
                result = lookup_weather(json.dumps(payload, ensure_ascii=False))
                self.assertEqual(result.name, "weather_lookup")
                self.assertEqual(result.data, {})
                for fragment in case["message_contains"]:
                    self.assertIn(fragment, result.message)
