# 植物识别 AI Agent —— 代码说明

基于 DeepSeek 多模态大模型 + 本地 MobileNet 的植物识别智能体:上传植物照片,输出植物名称(含置信度与 Top-3 候选)与养护建议;若为花卉,附带花语。

## 界面预览

![界面预览](screenshots/界面预览.png)

## 一、目录结构

```
code/
├── config.py           配置(API Key、模型名、路径)
├── tools.py            工具层:大模型识别 / 本地识别 / 知识生成
├── rag.py              本地知识库检索(检索增强生成,RAG)
├── agent.py            编排:识别 → 检索 → 生成,拼装输出
├── app.py              Gradio 网页界面(发布)
├── train.py            本地模型训练(支持 --mode full|reg|frozen)
├── compare.py          双路线对比评测(输出 指标/对比表.md)
├── download_images.py  数据集采集脚本(按关键词批量下载并清洗图片)
├── kb/                 植物养护知识库(文本)
├── models/             训练好的模型与类别表、训练历史
├── 指标/               对比评测结果
└── test_set/           测试图片(未随仓库提供,可用脚本采集)
```

## 二、环境与依赖

- Python 3.10+(开发环境 Python 3.13)
- 安装依赖:

```bash
pip install -r requirements.txt
```

- 若需 GPU 训练(可选,推荐):安装 CUDA 版 PyTorch

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 三、配置 API Key(必做)

1. 复制 `.env.example` 为 `.env`;
2. 在 `.env` 中填入自己的 DeepSeek API Key:

```
DEEPSEEK_API_KEY=sk-你的key
```

> 说明:源码中不含任何密钥;`.env` 不随代码提交。

## 四、运行

```bash
# 启动网页界面(浏览器打开 http://127.0.0.1:7860)
python app.py

# 训练本地模型(三种策略)
python train.py --mode frozen     # 冻结骨干 + 早停(最终采用)
python train.py --mode full       # 全参数微调
python train.py --mode reg        # 全参数 + 强增强 + 权重衰减

# 双路线对比评测(生成 指标/对比表.md)
python compare.py
```

## 五、数据集说明

训练数据集(20 类、共 763 张)体积较大,**未随代码提供**。可用 `download_images.py` 重新采集:

```bash
python download_images.py "绿萝" 40     # 关键词 + 张数
```

或向作者索取。测试集图片(`test_set/`)同样**未随仓库提供**,可自行采集若干张用于评测。

> **数据来源说明**:本项目所用植物图片均通过搜索引擎公开图片批量采集,仅用于课程实践与学习研究,版权归原作者所有;若涉及侵权请联系删除。

## 六、模型与结果

- `models/best.pt`:最终采用的本地模型(20 类,冻结骨干 MobileNetV3-small),配合 `models/classes.json` 使用;
- `models/hist_full.json`、`hist_reg.json`、`hist_frozen.json`:三种训练策略的逐轮记录;
- `指标/对比表.md`:本地模型与大模型在同一测试集上的对比结果。

## 七、许可

本项目代码采用 MIT 许可证,详见 [LICENSE](LICENSE)。
