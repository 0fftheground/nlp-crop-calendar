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
    from src.application.services.sowing_suitability_service import (
        _extract_variety_hint,
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

    def test_lookup_sowing_suitability_service_scenarios(self) -> None:
        scenarios = load_yaml_scenarios("sowing/service.yaml")["sowing_service_cases"]

        class StubResponse:
            def __init__(self, payload) -> None:
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self):
                return self._payload

        class StubHttp:
            def __init__(self, payload=None, *, forbid_post: bool = False) -> None:
                self._payload = payload
                self._forbid_post = forbid_post
                self.calls = []

            def get(self, *args, **kwargs):
                raise AssertionError("sowing suitability service should use POST")

            def post(self, url, *, json_payload, headers=None, timeout=10.0):
                if self._forbid_post:
                    raise AssertionError("unsupported region should not call API")
                self.calls.append(
                    {
                        "url": url,
                        "json_payload": json_payload,
                        "headers": headers,
                        "timeout": timeout,
                    }
                )
                return StubResponse(self._payload)

        for scenario in scenarios:
            with self.subTest(scenario=scenario["id"]):
                if scenario["id"] == "returns_followup_when_missing_fields":
                    result = lookup_sowing_suitability(scenario["raw_request"])
                    self.assertEqual(result.name, scenario["expected"]["name"])
                    for snippet in scenario["expected"]["message_contains"]:
                        self.assertIn(snippet, result.message)
                    self.assertEqual(
                        result.data.get("missing_fields"),
                        scenario["expected"]["missing_fields"],
                    )
                    continue

                config = scenario["config"]
                stub_config = type("StubConfig", (), dict(config))
                http = StubHttp(
                    scenario.get("http_response"),
                    forbid_post=scenario["id"] == "returns_unsupported_region_message",
                )
                sql_data = scenario["sql"]

                class StubSql:
                    def fetch_all(self, url, sql, params=()):
                        text = " ".join(str(item) for item in (sql, params))
                        if "SELECT code, code_name FROM agri_code_dict" in text:
                            if "sowingmtd" in text:
                                return list(sql_data["dict_rows"].get("sowingmtd", []))
                            if "culti_type" in text:
                                return list(sql_data["dict_rows"].get("culti_type", []))
                            if "sub_type" in text:
                                return list(sql_data["dict_rows"].get("sub_type", []))
                        if 'AS approve_region' in text and 'WHERE "name" = ' in text:
                            return list(sql_data.get("variety_rows", []))
                        if 'AS sub_type' in text and 'WHERE "name" = ' in text:
                            return list(sql_data.get("sub_type_rows", []))
                        if "ILIKE" in text:
                            return list(sql_data.get("region_rows", []))
                        return []

                    def quote_identifier(self, name: str) -> str:
                        return f'"{name}"'

                configure_sowing_suitability_ports(
                    config_port=type(
                        "P", (), {"get": staticmethod(lambda: stub_config())}
                    )(),
                    http_port=http,
                    sql_port=StubSql(),
                )

                if "raw_request_json" in scenario:
                    raw_request = json.dumps(
                        scenario["raw_request_json"], ensure_ascii=False
                    )
                else:
                    raw_request = json.dumps(
                        scenario["request_payload"], ensure_ascii=False
                    )
                result = lookup_sowing_suitability(raw_request)

                self.assertEqual(result.name, scenario["expected"].get("name", "sowing_suitability_lookup"))
                self.assertEqual(result.message, scenario["expected"]["message"])
                if scenario["id"] == "returns_unsupported_region_message":
                    self.assertEqual(len(http.calls), 0)
                    continue
                self.assertEqual(len(http.calls), 1)
                if scenario["expected"].get("url"):
                    self.assertEqual(http.calls[0]["url"], scenario["expected"]["url"])
                self.assertEqual(
                    http.calls[0]["json_payload"],
                    scenario["expected"]["payload"],
                )
                if scenario["expected"].get("result_suit_dates") is not None:
                    self.assertEqual(
                        result.data.get("result", {}).get("suitDate"),
                        scenario["expected"]["result_suit_dates"],
                    )
                if scenario["expected"].get("resolved"):
                    for key, value in scenario["expected"]["resolved"].items():
                        self.assertEqual(result.data.get("resolved", {}).get(key), value)

    def test_lookup_sowing_suitability_infers_culti_type_from_variety_region_record(self) -> None:
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
                self.calls = []

            def post(self, url, *, json_payload, headers=None, timeout=10.0):
                self.calls.append(json_payload)
                return StubResponse()

        class StubSql:
            def fetch_all(self, url, sql, params=()):
                text = " ".join(str(item) for item in (sql, params))
                if "SELECT code, code_name FROM agri_code_dict" in text:
                    if "sowingmtd" in text:
                        return [{"code": 0, "code_name": "直播"}]
                    if "culti_type" in text:
                        return [{"code": 4, "code_name": "一季晚稻"}]
                if 'AS approve_region' in text and 'WHERE "name" = ' in text:
                    return [
                        {
                            "name": "美香占2号",
                            "sub_type": 9,
                            "culti_type": "一季晚稻",
                            "approve_region": "湖南省",
                            "approve_year": 2024,
                        }
                    ]
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

        result = lookup_sowing_suitability("美香占2号在长沙什么时候适合播种，直播")

        self.assertEqual(result.name, "sowing_suitability_lookup")
        self.assertEqual(result.message, "success")
        self.assertEqual(len(http.calls), 1)
        self.assertEqual(
            http.calls[0],
            {
                "region_id": 430100,
                "culti_type": 4,
                "sowing_method": 0,
                "sub_type": 9,
                "crop": 0,
            },
        )
        self.assertEqual(result.data.get("resolved", {}).get("culti_type"), "一季晚稻")

    def test_lookup_sowing_suitability_returns_unapproved_region_message_for_variety(self) -> None:
        class StubConfig:
            agri_db_url = "postgresql://example"
            business_api_key = None
            business_api_base_url = "http://example.test"
            sowing_suitability_api_url = None
            default_farm_id = "12"
            db_region_lookup_candidates = []
            region_db_table = None

        class StubSql:
            def fetch_all(self, url, sql, params=()):
                text = " ".join(str(item) for item in (sql, params))
                if "SELECT code, code_name FROM agri_code_dict" in text:
                    if "sowingmtd" in text:
                        return [{"code": 0, "code_name": "直播"}]
                if 'AS approve_region' in text and 'WHERE "name" = ' in text:
                    return [
                        {
                            "name": "美香占2号",
                            "sub_type": 9,
                            "culti_type": "一季晚稻",
                            "approve_region": "湖南省",
                            "approve_year": 2024,
                        }
                    ]
                return []

            def quote_identifier(self, name: str) -> str:
                return f'"{name}"'

        configure_sowing_suitability_ports(
            config_port=type("P", (), {"get": staticmethod(lambda: StubConfig())})(),
            sql_port=StubSql(),
        )

        result = lookup_sowing_suitability("美香占2号在芜湖什么时候适合播种，直播")

        self.assertEqual(result.name, "sowing_suitability_lookup")
        self.assertEqual(result.message, "品种 美香占2号 未在 芜湖 审定。")

    def test_lookup_sowing_suitability_skips_llm_extraction_for_vague_prompt(self) -> None:
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
                self.calls = []

            def post(self, url, *, json_payload, headers=None, timeout=10.0):
                self.calls.append(json_payload)
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
                return []

            def quote_identifier(self, name: str) -> str:
                return f'"{name}"'

        http = StubHttp()
        configure_sowing_suitability_ports(
            config_port=type("P", (), {"get": staticmethod(lambda: StubConfig())})(),
            http_port=http,
            sql_port=StubSql(),
        )
        with patch(
            "src.application.services.sowing_suitability_service._llm_extract_planting_for_sowing"
        ) as mocked_extract:
            result = lookup_sowing_suitability("最近适合播种嘛")

        self.assertEqual(result.name, "sowing_suitability_lookup")
        self.assertIn("请补充", result.message)
        self.assertEqual(len(http.calls), 0)
        mocked_extract.assert_not_called()

    def test_lookup_sowing_suitability_returns_variety_candidates_for_partial_name(self) -> None:
        class StubConfig:
            agri_db_url = "postgresql://example"
            business_api_key = None
            business_api_base_url = "http://example.test"
            sowing_suitability_api_url = None
            default_farm_id = "12"
            db_region_lookup_candidates = []
            region_db_table = None

        class StubSql:
            def fetch_all(self, url, sql, params=()):
                text = " ".join(str(item) for item in (sql, params))
                if "SELECT code, code_name FROM agri_code_dict" in text:
                    if "sowingmtd" in text:
                        return [{"code": 0, "code_name": "直播"}]
                    if "culti_type" in text:
                        return [{"code": 3, "code_name": "中稻"}]
                if 'AS sub_type' in text and 'WHERE "name" = ' in text:
                    return []
                return []

            def quote_identifier(self, name: str) -> str:
                return f'"{name}"'

        configure_sowing_suitability_ports(
            config_port=type("P", (), {"get": staticmethod(lambda: StubConfig())})(),
            sql_port=StubSql(),
        )
        with patch(
            "src.application.services.sowing_suitability_service.retrieve_variety_candidates",
            return_value=["南粳46", "南粳9108"],
        ), patch(
            "src.application.services.sowing_suitability_service.find_exact_variety_in_text",
            return_value=None,
        ):
            result = lookup_sowing_suitability("南粳，中稻，直播")

        self.assertEqual(result.name, "sowing_suitability_lookup")
        self.assertIn("未找到完全匹配的品种", result.message)
        self.assertEqual(result.data.get("missing_fields"), ["variety"])
        self.assertEqual(result.data.get("options"), ["南粳46", "南粳9108"])
        self.assertTrue(result.data.get("choice_hint"))
        self.assertTrue(result.data.get("strict_options_only"))
        self.assertEqual(
            result.data.get("draft", {}).get("candidates"), ["南粳46", "南粳9108"]
        )

    def test_extract_variety_hint_ignores_sentence_prefix_and_keeps_actual_variety(self) -> None:
        self.assertEqual(
            _extract_variety_hint("我在湖南省常德种植早稻湘早籼24号，移栽什么时候播种合适"),
            "湘早籼24号",
        )

    def test_lookup_sowing_suitability_resolves_variety_candidate_followup(self) -> None:
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
                self.calls = []

            def post(self, url, *, json_payload, headers=None, timeout=10.0):
                self.calls.append(json_payload)
                return StubResponse()

        class StubSql:
            def fetch_all(self, url, sql, params=()):
                text = " ".join(str(item) for item in (sql, params))
                if "SELECT code, code_name FROM agri_code_dict" in text:
                    if "sowingmtd" in text:
                        return [{"code": 0, "code_name": "直播"}]
                    if "culti_type" in text:
                        return [{"code": 3, "code_name": "中稻"}]
                if 'AS sub_type' in text and 'WHERE "name" = ' in text:
                    if params == ("南粳46",):
                        return [{"sub_type": 9}]
                    return []
                return []

            def quote_identifier(self, name: str) -> str:
                return f'"{name}"'

        http = StubHttp()
        configure_sowing_suitability_ports(
            config_port=type("P", (), {"get": staticmethod(lambda: StubConfig())})(),
            http_port=http,
            sql_port=StubSql(),
        )
        raw_request = json.dumps(
            {
                "query": "南粳，中稻，直播",
                "followup": {
                    "prompt": "1",
                    "draft": {
                        "crop": "水稻",
                        "culti_type": "中稻",
                        "planting_method": "直播",
                        "candidates": ["南粳46", "南粳9108"],
                    },
                    "missing_fields": ["variety"],
                    "followup_count": 1,
                },
            },
            ensure_ascii=False,
        )
        result = lookup_sowing_suitability(raw_request)

        self.assertEqual(result.name, "sowing_suitability_lookup")
        self.assertEqual(result.message, "success")
        self.assertEqual(len(http.calls), 1)
        self.assertEqual(
            http.calls[0],
            {
                "farm_id": 12,
                "culti_type": 3,
                "sowing_method": 0,
                "sub_type": 9,
                "crop": 0,
            },
        )
        self.assertEqual(result.data.get("resolved", {}).get("variety"), "南粳46")


if __name__ == "__main__":
    unittest.main()
