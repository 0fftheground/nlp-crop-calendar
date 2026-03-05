from __future__ import annotations


BASE_PLANTING_EXTRACT_PROMPT = (
    "你是农事助手，负责从用户描述中抽取种植信息。"
    "只输出可确定的信息；不确定或未提及时保持为空。"
    "种植方式使用 direct_seeding 或 transplanting。"
    "日期格式为 YYYY-MM-DD。"
    "若用户给出种植区域（例如某省/市/县），写入 region_id（可先写区域名称）。"
    "稻作类型/熟制（如早稻、晚稻、双季晚稻）填写到 culti_type。"
    "当用户说“早稻/晚稻”等时，作物仍为水稻。"
)


def build_planting_extract_prompt(hint: str = "") -> str:
    if hint:
        return f"{BASE_PLANTING_EXTRACT_PROMPT}{hint}"
    return BASE_PLANTING_EXTRACT_PROMPT
