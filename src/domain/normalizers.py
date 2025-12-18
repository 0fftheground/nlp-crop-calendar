import re
from enum import Enum
from typing import Any, Type
from .enums import PlantingMethod

class EnumNormalizer:
    # 每个枚举一张表：alias -> canonical
    ALIASES: dict[Type[Enum], dict[str, str]] = {
        PlantingMethod: {
            # direct_seeding
            "直播": "direct_seeding",
            "撒播": "direct_seeding",
            "直接播种": "direct_seeding",
            "direct seeding": "direct_seeding",
            "direct-seeding": "direct_seeding",
            "direct_seeding": "direct_seeding",

            # transplanting
            "移栽": "transplanting",
            "插秧": "transplanting",
            "机插": "transplanting",
            "抛秧": "transplanting",
            "transplanting": "transplanting",
        },
        
    }

    @staticmethod
    def _canon_key(x: Any) -> str:
        # 统一：去空格、小写、把连续空白压缩成一个空格
        s = str(x).strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    @classmethod
    def normalize(cls, enum_cls: Type[Enum], value: Any) -> Any:
        if value is None:
            return value

        key = cls._canon_key(value)

        # 允许直接给 Enum/标准值
        if isinstance(value, enum_cls):
            return value.value

        # 命中 alias
        aliases = cls.ALIASES.get(enum_cls, {})
        return aliases.get(key, value)  # 未命中则原样返回，让 Pydantic 报错
