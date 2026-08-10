# 多模态驱动的电商服饰细粒度语义增强与智能解析系统

本项目面向电商服饰场景，构建“商品图片 + 自然语言查询”联合驱动的细粒度语义增强与智能解析系统。系统目标包括服饰实例分割、语言引导局部区域定位、细粒度属性提取、多模态问答、商家内容生成，以及基于时尚专业知识库的 RAG 增强回答。

## 项目状态

PRD 3.1.1 服饰实例分割的功能主线已经完成并阶段收口。项目已实现 DeepFashion2 与 Fashionpedia 到统一八类 COCO 数据的转换、Mask2Former 混合训练、双数据集完整验证、FP16 精度与延迟实验、八类可视化验收，以及可复用模型实例的 FastAPI 推理接口。2,000 次混合训练 checkpoint 在完整 DeepFashion2 上达到 mask AP `60.58`，在 Fashionpedia 上达到 mask AP `45.74`，鞋子、包包和配饰均获得非零正式指标。默认部署配置继续指向该八分类 checkpoint 的 `512/853 + FP16 + 0.6` 推理档；主查询链路会在服饰分割前自动选择主要人物，并保留 `0.35` 的上下文区域。同参数 DeepFashion2 500 图复验达到 mask AP `45.68`，阈值 `0.6` 下匹配 Mask 平均 IoU 为 `89.81%`。默认 API 的八类代表样例全部命中，返回的 Mask 均非空且 Box 均有效。小物件专项微调结果保留为实验候选，但因完整 DeepFashion2 跨域验证未完成，不替换当前默认 checkpoint。RTX 3090 尚未达到 50 ms 目标；小物件、TensorRT 和更高性能 GPU 优化均延后到主流程打通之后。

PRD 3.1.2 目前只有历史原型和实验基线，尚未形成符合正式技术栈的交付实现。已有 Fashionpedia 19 类局部 Mask、人物 ROI、Grounding DINO 候选框、SAM-HQ Mask 细化、Mask 派生 Box 和 `POST /v1/localize` 路由；这些资产可用于数据、接口和失败分析，但 Grounding DINO 与固定类别 Hybrid 都不是 PRD 指定的 `DINOv2 区域特征 + 文本特征相似度匹配 + SAM-HQ` 主路径。现有代表样例、固定类别指标和开放词汇 smoke 均不能证明定位准确率达到 `92%` 或定位时间达到 `30 ms`。后续主线必须使用 Python 3.10.12，并按 PRD 实现 DINOv2 区域表示、完整语言表达的跨模态匹配、SAM-HQ Mask 细化及 TensorRT 优化；原型代码仅作为历史对照，不作为验收后端。

导师反馈后，3.1.2 的任务边界已修正为开放语言指代表达定位，而不是用自然语言选择固定 N 类部位。现有 19 类和 Hybrid 路径只保留为历史辅助基线；`targeted_3000` 是最佳固定类别实验模型，但不属于最终 3.1.2 主路径。下一阶段使用 `data/benchmarks/localization/referring_smoke_v1.template.json` 建立包含部件、方位、属性、关系及新部件的查询级验证集，并严格依据 PRD 实现 `DINOv2 区域特征 + 完整文本特征匹配 + SAM-HQ`，将功能 smoke 与 `92%`、`30 ms` 正式验收分开。

首轮真实开放词汇历史 smoke 已完成：4 条带 Fashionpedia GT 的查询在 Mask IoU `0.50` 下为 `0/4`，完整表达、名词提示和人物 ROI 都没有解决拉链与口袋的候选召回。领口候选 Box IoU 达到 `90.77%`，但 SAM-HQ Mask IoU 只有 `45.82%`。低阈值固定类别候选和极端坐标重排同样失败，因此停止继续调 Grounding、固定类别阈值和手工方位规则；这些结果只用于说明旧路线失败，不再决定 PRD 主路径。

分割服务入口为 `POST /v1/segment`，局部定位入口为 `POST /v1/localize`。`POST /v1/query` 默认启用自动人物 ROI，一般服饰查询返回 3.1.1 分割结果，已知局部部位查询同时返回 3.1.2 定位结果；属性提取和最终问答仍属于后续 PRD 模块，不能由当前分割或定位接口替代。

## 目录结构

```text
.
├── configs/                  # 配置文件
├── data/                     # 数据目录，不提交大文件
│   ├── knowledge/            # RAG 知识库原始/清洗数据
│   ├── processed/            # 处理后的数据
│   └── raw/                  # 原始数据
├── docs/                     # 项目文档
├── models/                   # 模型相关文件，不提交大权重
│   └── checkpoints/          # 本地模型权重目录
├── outputs/                  # 推理输出和实验结果
├── scripts/                  # 开发和运维脚本
├── src/
│   └── fashion_semantic_parser/
│       ├── api/              # FastAPI 接口层
│       ├── common/           # 公共工具、异常、常量
│       ├── dao/              # 数据访问层
│       ├── models/           # 领域数据模型
│       └── service/          # 业务编排层
└── tests/                    # 测试目录
```

## 环境要求

- Python 3.10
- CUDA 12.x + PyTorch 2.x（云 GPU 开发环境）
- 推荐 GPU：RTX 3090 24GB 或更高

## 本地开发

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
pytest
```

也可以使用 `pyproject.toml` 的开发依赖安装方式：

```bash
python -m pip install -e ".[dev]"
```

## 路径约定

项目代码中禁止硬编码个人电脑绝对路径。所有数据、模型、配置和输出路径都应从项目根目录按相对路径解析。

示例：

```python
from fashion_semantic_parser.common.paths import resolve_project_path

image_path = resolve_project_path("data/raw/example.jpg")
```
