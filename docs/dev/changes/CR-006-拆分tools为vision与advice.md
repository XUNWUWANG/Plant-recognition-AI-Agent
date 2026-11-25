# CR-006 拆分 tools 为 vision 与 advice

- 状态:提案(待评审,未实施)
- 日期:2026-11-25
- 依赖:CR-003(统一编排)、CR-004(配置分层)
- 建议顺序:4/5

## 1. 变更原因

`tools.py` 同时承担四类职责:大模型识图、本地 CV 推理、LLM 建议生成、离线建议。它被 `agent`/`app`/`compare` 同时依赖,任一处改动都影响全部调用方;`compare.py` 只需要识图,却被迫依赖 openai/torch/retrieval。

## 2. 影响范围

- 新增:`llm.py`、`vision/api.py`、`vision/local.py`、`advice/service.py`
- 修改:`agent.py`(改用新模块)、`app.py`(若 CR-003 已完成则只依赖 agent)、`tests/test_compat.py`
- 保留:`tools.py` 作为**纯转发壳**(见下方决策点)
- 不动:`train.py`、`compare.py`、`retrieval/`

## 3. 方案要点

| 新模块 | 内容 | 依赖 |
|---|---|---|
| `llm.py` | `get_client()` 懒加载 OpenAI 客户端 | `config` |
| `vision/api.py` | `classify_plant`、`_image_to_data_url`、`_parse_json` | `llm`、PIL |
| `vision/local.py` | `classify_local`、模型懒加载缓存 | torch/torchvision、`config` |
| `advice/service.py` | prompt、`plant_advice`、`plant_advice_offline` | `llm`、`retrieval.rag` |

`tools.py` 改为纯转发(无业务逻辑):
```python
from vision.api import classify_plant
from vision.local import classify_local
from advice.service import plant_advice, plant_advice_offline
```

**决策点**:任务书要求"不动 compare.py",而 `compare.py` 目前 `from tools import ...`。
- 方案 A(推荐):保留 `tools.py` 转发壳,`compare.py` 零修改;
- 方案 B(更干净):允许修改 `compare.py` 一行 import,删除 `tools.py`。需你明确批准打破"不动"约束。

默认按方案 A 实施;答辩后再评估移除转发壳。CR-008 生效后,方案 B 不再需要修改任务书,但仍需你明确批准。

**不做**:不改识图/建议行为;不改 prompt;不改检索。

## 4. 验收标准

- `tools.py` 内无函数定义,只有 import;
- `grep -rn "import tools\|from tools import" --include="*.py" .` 只剩 `compare.py`;
- 本地识图、纯文字问答、图片+追问三条路径输出不变;
- `python -c "from tools import classify_local, classify_plant, plant_advice, plant_advice_offline"` 成功(为 compare 兼容);
- 测试 15 绿;五组评测指标不变。

## 5. 回滚方案

删除新模块,恢复 `tools.py` 原内容,还原 `agent.py`/tests 的 import;`git revert` 单提交。

## 6. 执行记录

- 日期:2026-11-25
- 结果:新增 `llm.py`、`vision/api.py`、`vision/local.py`、`advice/service.py`;`tools.py` 变为纯转发壳;`agent.py` 直接依赖新模块;`compare.py` 保持原 import。
- 验收:`tools.py` 无函数定义;`import tools` 仅剩 `compare.py` 与兼容测试;`from tools import ...` 四函数成功;五条路径输出与改造前一致;16 测试绿;五组指标不变。
- 决策点:按方案 A 保留转发壳,未修改 `compare.py`。
