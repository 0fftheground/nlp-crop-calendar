"""
API调用模块
处理HTTP请求的发送和响应处理
支持真实API调用和Mock模式
"""

import requests
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, ValidationError
import json
import os


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 获取Mock模式配置（从环境变量或默认False）
USE_MOCK_MODE = os.getenv("USE_MOCK_API", "false").lower() == "true"


class APIResponse(BaseModel):
    """API响应数据模型"""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


MOCK_API_HOST = "https://mock.api.local"


class MockAPIData:
    """Mock API 数据提供器"""

    @staticmethod
    def get_mock_response(endpoint: str, method: str = "GET", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        根据 endpoint 返回 Mock 数据

        Args:
            endpoint: API端点
            method: HTTP方法
            params: 请求参数

        Returns:
            Mock 响应数据
        """
        if params is None:
            params = {}

        normalized_endpoint = endpoint.lower()

        # 天气 API Mock
        if (
            "weatherapi.com" in normalized_endpoint
            or normalized_endpoint.startswith(f"{MOCK_API_HOST}/weather")
            or "weather" in normalized_endpoint
        ):
            return {
                "location": {
                    "name": params.get("q", "Beijing"),
                    "region": "China",
                    "country": "China",
                    "lat": 39.92,
                    "lon": 116.41,
                    "tz_id": "Asia/Shanghai",
                    "localtime_epoch": 1702300800,
                    "localtime": "2023-12-11 12:00"
                },
                "current": {
                    "temp_c": 5.2,
                    "temp_f": 41.4,
                    "is_day": 1,
                    "condition": {
                        "text": "晴",
                        "icon": "//cdn.weatherapi.com/weather/128x128/day/113.png",
                        "code": 1000
                    },
                    "wind_kph": 12.3,
                    "wind_dir": "NE",
                    "humidity": 45,
                    "feelslike_c": 2.8,
                }
            }

        # 翻译 API Mock
        elif (
            "mymemory.translated.net" in normalized_endpoint
            or normalized_endpoint.startswith(f"{MOCK_API_HOST}/translate")
            or "translat" in normalized_endpoint
        ):
            query = params.get("q", "Hello")
            langpair = params.get("langpair", "en|zh")
            
            # 简单的翻译Mock映射
            translation_map = {
                "hello": "你好",
                "world": "世界",
                "good morning": "早上好",
                "thank you": "谢谢你",
                "how are you": "你好吗",
            }
            
            source_lang, target_lang = langpair.split("|")
            translated = translation_map.get(query.lower(), f"[{target_lang}翻译] {query}")
            
            return {
                "responseStatus": 200,
                "responseData": {
                    "translatedText": translated,
                    "match": 1.0,
                },
                "quotaFinished": False,
                "mtLangSupported": None,
            }

        # 新闻/搜索 API Mock
        elif (
            normalized_endpoint.startswith(f"{MOCK_API_HOST}/info")
            or normalized_endpoint.startswith(f"{MOCK_API_HOST}/search")
            or "search" in normalized_endpoint
        ):
            query = params.get("q", params.get("query", "search"))
            return {
                "query": query,
                "results": [
                    {
                        "title": f"搜索结果: {query} - 1",
                        "url": f"https://example.com/result1",
                        "snippet": "这是关于搜索结果的相关内容摘要，包含了用户查询的关键词。"
                    },
                    {
                        "title": f"搜索结果: {query} - 2",
                        "url": f"https://example.com/result2",
                        "snippet": "这是第二个搜索结果，提供了更多相关的信息。"
                    },
                    {
                        "title": f"搜索结果: {query} - 3",
                        "url": f"https://example.com/result3",
                        "snippet": "第三个搜索结果，帮助用户更好地了解相关内容。"
                    }
                ],
                "total_results": 1000,
                "page": 1,
                "results_per_page": 3,
            }

        # 任务状态/运营数据 Mock
        elif normalized_endpoint.startswith(f"{MOCK_API_HOST}/tasks"):
            status = params.get("status", "in_progress")
            return {
                "status": status,
                "updated_at": "2023-12-11T12:00:00Z",
                "items": [
                    {"id": "task-001", "title": "数据清洗", "status": "in_progress"},
                    {"id": "task-002", "title": "报告撰写", "status": "completed"},
                ],
            }

        # 默认响应
        else:
            return {
                "message": "Mock endpoint 未定义，返回占位内容",
                "endpoint": endpoint,
            }


class APICaller:
    """
    API调用器
    支持GET和POST请求，可切换真实API和Mock模式
    """

    def __init__(self, timeout: int = 10, use_mock: bool = None):
        """
        初始化API调用器

        Args:
            timeout: 请求超时时间（秒）
            use_mock: 是否使用Mock模式（None表示使用环境变量设置）
        """
        self.timeout = timeout
        self.session = requests.Session()
        
        # 确定是否使用Mock模式
        if use_mock is None:
            self.use_mock = USE_MOCK_MODE
        else:
            self.use_mock = use_mock
            
        if self.use_mock:
            logger.info("🔧 已启用 Mock 模式 - 所有 API 调用将使用 Mock 数据")
        else:
            logger.info("🌐 已启用真实 API 模式 - 将调用真实网络接口")

    def set_mock_mode(self, use_mock: bool):
        """
        动态设置Mock模式

        Args:
            use_mock: 是否使用Mock模式
        """
        self.use_mock = use_mock
        mode_str = "Mock 模式" if use_mock else "真实 API 模式"
        logger.info(f"已切换到: {mode_str}")

    def call_api(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> APIResponse:
        """
        调用API

        Args:
            endpoint: API端点URL
            method: HTTP方法 (GET 或 POST)
            params: 请求参数
            headers: 请求头
            **kwargs: 其他请求参数

        Returns:
            APIResponse对象
        """
        try:
            # 如果启用Mock模式，直接返回Mock数据
            if self.use_mock:
                logger.info(f"[Mock] 调用API: {method} {endpoint}")
                mock_data = MockAPIData.get_mock_response(endpoint, method, params)
                return APIResponse(success=True, data=mock_data, status_code=200)

            # 真实API调用模式
            # 设置默认请求头
            if headers is None:
                headers = {}
            headers.setdefault("Content-Type", "application/json")
            headers.setdefault("User-Agent", "NLP-App/1.0")

            # 发送请求
            method = method.upper()
            logger.info(f"调用API: {method} {endpoint}")

            if method == "GET":
                response = self.session.get(
                    endpoint, params=params, headers=headers, timeout=self.timeout
                )
            elif method == "POST":
                response = self.session.post(
                    endpoint,
                    json=params,
                    headers=headers,
                    timeout=self.timeout,
                    **kwargs,
                )
            else:
                return APIResponse(
                    success=False, error=f"不支持的HTTP方法: {method}"
                )

            # 处理响应
            if response.status_code == 200:
                try:
                    data = response.json()
                    return APIResponse(success=True, data=data, status_code=200)
                except ValueError:
                    # 如果响应不是JSON格式
                    return APIResponse(
                        success=True,
                        data={"raw_text": response.text},
                        status_code=200,
                    )
            else:
                return APIResponse(
                    success=False,
                    error=f"API请求失败: {response.status_code} {response.reason}",
                    status_code=response.status_code,
                )

        except requests.exceptions.Timeout:
            error_msg = f"请求超时 (>{self.timeout}秒)"
            logger.error(error_msg)
            return APIResponse(success=False, error=error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求错误: {str(e)}"
            logger.error(error_msg)
            return APIResponse(success=False, error=error_msg)

        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(error_msg)
            return APIResponse(success=False, error=error_msg)

    def close(self):
        """关闭会话"""
        self.session.close()
