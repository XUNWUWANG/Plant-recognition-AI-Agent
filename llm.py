"""大模型客户端:懒加载 OpenAI 兼容客户端,无 Key 时只在真正调用时报错。"""

from openai import OpenAI

import config as cfg


def get_client():
    return OpenAI(api_key=cfg.require_api_key(), base_url=cfg.BASE_URL)
