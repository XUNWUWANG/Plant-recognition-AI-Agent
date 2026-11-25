"""养护建议服务:检索知识库后生成建议;离线路线直接返回原文。"""

import config as cfg
from llm import get_client
from retrieval import rag

_ADVICE_SYSTEM = "你是植物养护助手,基于知识库资料输出建议,简洁口语化。"

_ADVICE_PROMPT = (
    "以下是植物知识库中关于「{plant}」的资料:\n{context}\n\n"
    "请基于资料输出该植物的养护建议,按以下格式、每条一行:\n"
    "特征:…\n光照:…\n浇水:…\n温度:…\n施肥:…\n备注:…"
)

_ADVICE_PROMPT_Q = (
    "以下是植物知识库中与问题相关的资料:\n{context}\n\n"
    "用户的问题:{question}\n"
    "请先基于资料回答该问题,再补充必要的养护建议,按以下格式、每条一行:\n"
    "特征:…\n光照:…\n浇水:…\n温度:…\n施肥:…\n备注:…"
)

_GENERIC_ADVICE_PROMPT = (
    "「{plant}」不在我的植物知识库里。请给出通用养护建议,按以下格式、每条一行:\n"
    "特征:…\n光照:…\n浇水:…\n温度:…\n施肥:…\n备注:(说明知识库暂无此植物,建议谨慎参考)"
)

_GENERIC_ADVICE_PROMPT_Q = (
    "「{plant}」不在我的植物知识库里。用户的问题:{question}\n"
    "请针对问题给出通用养护建议,按以下格式、每条一行:\n"
    "特征:…\n光照:…\n浇水:…\n温度:…\n施肥:…\n备注:(说明知识库暂无此植物,建议谨慎参考)"
)


def plant_advice(plant_name=None, question=None, topk=2):
    """先检索知识库(RAG),再让大模型基于资料生成养护建议。

    返回 {content: 建议文本, source: 'kb' | 'generic'}。
    """
    q = (question or "").strip()
    query = q or (plant_name or "")
    ctx, hit = rag.retrieve_hit(query, topk=topk, plant_hint=plant_name)
    label = plant_name or "未知植物"
    if hit and ctx:
        if q:
            prompt = _ADVICE_PROMPT_Q.format(plant=label, context="\n".join(ctx),
                                             question=q)
        else:
            prompt = _ADVICE_PROMPT.format(plant=label, context="\n".join(ctx))
        source = "kb"
    else:
        if q:
            prompt = _GENERIC_ADVICE_PROMPT_Q.format(plant=label, question=q)
        else:
            prompt = _GENERIC_ADVICE_PROMPT.format(plant=label)
        source = "generic"

    resp = get_client().chat.completions.create(
        model=cfg.MODEL,
        messages=[
            {"role": "system", "content": _ADVICE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return {"content": resp.choices[0].message.content, "source": source}


def plant_advice_offline(plant_name=None, question=None, topk=1):
    """完全离线的养护建议:直接取知识库条目原文返回,不调用大模型 API。

    用于“本地模型识别/纯问答”路线,使该路线真正做到零 API、零额度消耗。
    知识库未命中时返回一句提示,建议改用大模型路线获取更完善建议。
    """
    q = (question or "").strip()
    query = q or (plant_name or "")
    ctx, hit = rag.retrieve_hit(query, topk=topk, plant_hint=plant_name)
    if hit and ctx:
        body = ctx[0].split("\n", 1)[1] if "\n" in ctx[0] else ctx[0]
        return {"content": body, "source": "kb"}
    label = plant_name or "这个问题"
    return {"content": f"知识库暂无「{label}」的详细资料。可切换“大模型识别”获取更完善的建议。",
            "source": "generic"}
