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
    from src.application.services.sowing_suitability_service import (
        configure_sowing_suitability_ports,
        lookup_sowing_suitability,
    )


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class SowingSuitabilityServiceTests(unittest.TestCase):
    def tearDown(self) -> None:
        configure_sowing_suitability_ports(
            config_port=DEFAULT_CONFIG_ADAPTER,
            http_port=DEFAULT_HTTP_ADAPTER,
            sql_port=DEFAULT_SQL_ADAPTER,
        )

    def test_lookup_sowing_suitability_uses_region_id_and_variety_sub_type(self) -> None:
        class StubConfig:
            agri_db_url = "postgresql://example"
            business_api_key = None
            business_api_base_url = "http://example.test"
            sowing_suitability_api_url = None
            default_farm_id = "8"
            db_region_lookup_candidates = []
            region_db_table = None

        class StubResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self):
                return {
                    "code": 200,
                    "message": "success",
                    "data": {
                        "suitDate": ["2026-03-25", "2026-03-26"],
                        "unsuitDate": [""],
                        "unsuitReasons": [""],
                    },
                }

        class StubHttp:
            def __init__(self) -> None:
                self.calls = []

            def get(self, *args, **kwargs):
                raise AssertionError("sowing suitability service should use POST")

            def post(self, url, *, json_payload, headers=None, timeout=10.0):
                self.calls.append(
                    {
                        "url": url,
                        "json_payload": json_payload,
                        "headers": headers,
                        "timeout": timeout,
                    }
                )
                return StubResponse()

        class StubSql:
            def fetch_all(self, url, sql, params=()):
                text = " ".join(str(item) for item in (sql, params))
                if "SELECT code, code_name FROM agri_code_dict" in text:
                    if "sowingmtd" in text:
                        return [{"code": 0, "code_name": "直播"}]
                    if "culti_type" in text:
                        return [{"code": 4, "code_name": "一季晚稻"}]
                    if "sub_type" in text:
                        return [{"code": 9, "code_name": "籼稻"}]
                if 'AS sub_type' in text and 'WHERE "name" = ' in text:
                    return [{"sub_type": 9}]
                if "ILIKE" in text:
                    return [{"region_id": 430100, "region_name": "长沙市"}]
                return []

            def quote_identifier(self, name: str) -> str:
                return f'"{name}"'

        http = StubHttp()
        configure_sowing_suitability_ports(
            config_port=type("P", (), {"get": staticmethod(lambda: StubConfig())})(),
            http_port=http,
            sql_port=StubSql(),
        )

        result = lookup_sowing_suitability(
            '{"variety":"美香占2号","culti_type":"一季晚稻","planting_method":"直播","region_id":"长沙"}'
        )

        self.assertEqual(result.name, "sowing_suitability_lookup")
        self.assertEqual(result.message, "success")
        self.assertEqual(len(http.calls), 1)
        self.assertEqual(http.calls[0]["url"], "http://example.test/bozhong_syd")
        self.assertEqual(
            http.calls[0]["json_payload"],
            {
                "region_id": 430100,
                "culti_type": 4,
                "sowing_method": 0,
                "sub_type": 9,
                "crop": 0,
            },
        )
        self.assertEqual(
            result.data.get("result", {}).get("suitDate"),
            ["2026-03-25", "2026-03-26"],
        )
        self.assertEqual(result.data.get("resolved", {}).get("sub_type"), 9)

    def test_lookup_sowing_suitability_falls_back_to_default_farm(self) -> None:
        class StubConfig:
            agri_db_url = "postgresql://example"
            business_api_key = None
            business_api_base_url = "http://example.test"
            sowing_suitability_api_url = None
            default_farm_id = "12"
            db_region_lookup_candidates = []
            region_db_table = None

        class StubResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self):
                return {"code": 200, "message": "success", "data": {"suitDate": []}}

        class StubHttp:
            def __init__(self) -> None:
                self.last_payload = None

            def get(self, *args, **kwargs):
                raise AssertionError("sowing suitability service should use POST")

            def post(self, url, *, json_payload, headers=None, timeout=10.0):
                self.last_payload = json_payload
                return StubResponse()

        class StubSql:
            def fetch_all(self, url, sql, params=()):
                text = " ".join(str(item) for item in (sql, params))
                if "SELECT code, code_name FROM agri_code_dict" in text:
                    if "sowingmtd" in text:
                        return [{"code": 1, "code_name": "插秧"}]
                    if "culti_type" in text:
                        return [{"code": 4, "code_name": "一季晚稻"}]
                if 'AS sub_type' in text and 'WHERE "name" = ' in text:
                    return [{"sub_type": 9}]
                return []

            def quote_identifier(self, name: str) -> str:
                return f'"{name}"'

        http = StubHttp()
        configure_sowing_suitability_ports(
            config_port=type("P", (), {"get": staticmethod(lambda: StubConfig())})(),
            http_port=http,
            sql_port=StubSql(),
        )

        result = lookup_sowing_suitability(
            '{"variety":"美香占2号","culti_type":"一季晚稻","planting_method":"移栽"}'
        )

        self.assertEqual(
            http.last_payload,
            {
                "farm_id": 12,
                "culti_type": 4,
                "sowing_method": 1,
                "sub_type": 9,
                "crop": 0,
            },
        )
        self.assertEqual(result.data.get("resolved", {}).get("farm_id"), 12)

    def test_lookup_sowing_suitability_returns_followup_when_missing_fields(self) -> None:
        result = lookup_sowing_suitability("帮我查播种适宜期")

        self.assertEqual(result.name, "sowing_suitability_lookup")
        self.assertIn("请补充", result.message)
        self.assertEqual(
            result.data.get("missing_fields"),
            ["variety", "culti_type", "planting_method"],
        )

    def test_lookup_sowing_suitability_followup_merges_planting_method(self) -> None:
        class StubConfig:
            agri_db_url = "postgresql://example"
            business_api_key = None
            business_api_base_url = "http://example.test"
            sowing_suitability_api_url = None
            default_farm_id = "12"
            db_region_lookup_candidates = []
            region_db_table = None

        class StubResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self):
                return {"code": 200, "message": "success", "data": {"suitDate": []}}

        class StubHttp:
            def __init__(self) -> None:
                self.last_payload = None

            def get(self, *args, **kwargs):
                raise AssertionError("sowing suitability service should use POST")

            def post(self, url, *, json_payload, headers=None, timeout=10.0):
                self.last_payload = json_payload
                return StubResponse()

        class StubSql:
            def fetch_all(self, url, sql, params=()):
                text = " ".join(str(item) for item in (sql, params))
                if "SELECT code, code_name FROM agri_code_dict" in text:
                    if "sowingmtd" in text:
                        return [{"code": 0, "code_name": "直播"}]
                    if "culti_type" in text:
                        return [{"code": 4, "code_name": "一季晚稻"}]
                if 'AS sub_type' in text and 'WHERE "name" = ' in text:
                    return [{"sub_type": 9}]
                if "ILIKE" in text:
                    return [{"region_id": 430100, "region_name": "长沙市"}]
                return []

            def quote_identifier(self, name: str) -> str:
                return f'"{name}"'

        http = StubHttp()
        configure_sowing_suitability_ports(
            config_port=type("P", (), {"get": staticmethod(lambda: StubConfig())})(),
            http_port=http,
            sql_port=StubSql(),
        )

        result = lookup_sowing_suitability(
            '{"query":"美香占2号适合在湖南当早稻种吗","followup":{"prompt":"直播","draft":{"variety":"美香占2号","culti_type":"一季晚稻","region_id":"长沙"},"missing_fields":["planting_method"],"followup_count":1}}'
        )

        self.assertEqual(result.message, "success")
        self.assertEqual(
            http.last_payload,
            {
                "region_id": 430100,
                "culti_type": 4,
                "sowing_method": 0,
                "sub_type": 9,
                "crop": 0,
            },
        )

    def test_lookup_sowing_suitability_returns_unsupported_region_message(self) -> None:
        class StubConfig:
            agri_db_url = "postgresql://example"
            business_api_key = None
            business_api_base_url = "http://example.test"
            sowing_suitability_api_url = None
            default_farm_id = "12"
            db_region_lookup_candidates = []
            region_db_table = None

        class StubHttp:
            def post(self, *args, **kwargs):
                raise AssertionError("unsupported region should not call API")

            def get(self, *args, **kwargs):
                raise AssertionError("sowing suitability service should use POST")

        class StubSql:
            def fetch_all(self, url, sql, params=()):
                text = " ".join(str(item) for item in (sql, params))
                if "SELECT code, code_name FROM agri_code_dict" in text:
                    if "sowingmtd" in text:
                        return [{"code": 0, "code_name": "直播"}]
                    if "culti_type" in text:
                        return [{"code": 4, "code_name": "一季晚稻"}]
                if 'AS sub_type' in text and 'WHERE "name" = ' in text:
                    return [{"sub_type": 9}]
                return []

            def quote_identifier(self, name: str) -> str:
                return f'"{name}"'

        configure_sowing_suitability_ports(
            config_port=type("P", (), {"get": staticmethod(lambda: StubConfig())})(),
            http_port=StubHttp(),
            sql_port=StubSql(),
        )

        result = lookup_sowing_suitability(
            '{"variety":"美香占2号","culti_type":"一季晚稻","planting_method":"直播","region_id":"芜湖"}'
        )

        self.assertEqual(result.name, "sowing_suitability_lookup")
        self.assertEqual(result.message, "暂不支持该区域的播期推荐：芜湖")


if __name__ == "__main__":
    unittest.main()
