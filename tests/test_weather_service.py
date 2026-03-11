import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_MISSING_PYDANTIC_SETTINGS = importlib.util.find_spec("pydantic_settings") is None

if not _MISSING_PYDANTIC_SETTINGS:
    from src.application.adapters import (
        DEFAULT_CONFIG_ADAPTER,
        DEFAULT_HTTP_ADAPTER,
        DEFAULT_SQL_ADAPTER,
    )
    from src.application.services.weather_service import (
        configure_weather_ports,
        lookup_weather,
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
