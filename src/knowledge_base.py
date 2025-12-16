from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Task:
    title: str
    description: str
    reasoning: str
    months: List[str]
    regions: List[str]


CROP_PLAYBOOK: Dict[str, Dict[str, List[Task]]] = {
    "rice": {
        "panicle_initiation": [
            Task(
                title="保持浅水层",
                description="抽穗前维持 3-5cm 水层并适度晒田。",
                reasoning="稳定分蘖向成穗转化并抑制杂草。",
                months=["august", "september"],
                regions=["global", "asia", "sichuan"],
            ),
            Task(
                title="稻飞虱统防统治",
                description="集中时间喷施吡蚜酮等低毒药剂并监测虫口基数。",
                reasoning="在孕穗期压低虫口可减少空壳率。",
                months=["august", "september", "october"],
                regions=["global", "asia", "sichuan"],
            ),
        ],
        "tillering": [
            Task(
                title="分蘖肥补充",
                description="每亩追施尿素 8-10kg 并结合轻度晒田。",
                reasoning="补氮促成有效分蘖。",
                months=["june", "july"],
                regions=["global", "asia"],
            )
        ],
    },
    "corn": {
        "vegetative": [
            Task(
                title="叶面肥管理",
                description="喷施含锌叶面肥并补足水分。",
                reasoning="提高光合效率，缓解低温抑制。",
                months=["may", "june"],
                regions=["global", "north_america"],
            ),
            Task(
                title="病虫监测",
                description="重点巡查灰斑病、玉米螟并清除病株。",
                reasoning="早期控制避免爆发。",
                months=["june", "july"],
                regions=["global", "north_america"],
            ),
        ]
    },
    "tomato": {
        "fruiting": [
            Task(
                title="夜间防寒覆盖",
                description="提前准备无纺布，温度低于12℃覆盖。",
                reasoning="防止低温引起落花落果。",
                months=["january", "february"],
                regions=["global", "california"],
            ),
            Task(
                title="钾肥滴灌",
                description="结合灌溉补充硫酸钾 8-10kg/亩。",
                reasoning="维持膨果和糖度。",
                months=["february", "march"],
                regions=["global", "california"],
            ),
        ]
    },
}
