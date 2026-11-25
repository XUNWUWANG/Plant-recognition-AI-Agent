"""兼容转发壳(CR-006):新代码请直接依赖 llm / vision / advice。"""

from advice.service import plant_advice, plant_advice_offline
from vision.api import classify_plant
from vision.local import classify_local

__all__ = ["classify_plant", "classify_local",
           "plant_advice", "plant_advice_offline"]
