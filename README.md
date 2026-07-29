# 多模态驱动的电商服饰细粒度语义增强与智能解析系统

本项目面向电商服饰场景，构建“商品图片 + 自然语言查询”联合驱动的细粒度语义增强与智能解析系统。系统目标包括服饰实例分割、语言引导局部区域定位、细粒度属性提取、多模态问答、商家内容生成，以及基于时尚专业知识库的 RAG 增强回答。

## 项目状态

当前阶段聚焦 PRD 3.1.1 服饰实例分割功能。已经完成 DeepFashion2 与 Fashionpedia 到统一八类 COCO 数据的转换、Mask2Former 混合训练、双数据集完整验证、FP16 精度与延迟实验、八类可视化验收，以及可复用模型实例的 FastAPI 推理接口。2,000 次混合训练 checkpoint 在完整 DeepFashion2 上达到 mask AP `60.58`，在 Fashionpedia 上达到 mask AP `45.74`，鞋子、包包和配饰均获得非零正式指标。默认部署配置现已指向该八分类 checkpoint 的 `512/853 + FP16 + 0.6` 推理档；主查询链路会在服饰分割前自动选择主要人物，并保留 `0.35` 的上下文区域。同参数 DeepFashion2 500 图复验达到 mask AP `45.68`，阈值 `0.6` 下匹配 Mask 平均 IoU 为 `89.81%`。默认 API 的八类代表样例全部命中，返回的 Mask 均非空且 Box 均有效。除延迟外，PRD 3.1.1 的数据、模型、精度评估、可视化和服务链路均已完成；RTX 3090 尚未达到 50 ms 目标，后续使用更高性能 GPU 或 TensorRT 等优化运行时处理。

分割服务入口为 `POST /v1/segment`，返回每个服饰实例的 Mask、边界框、类别和置信度。现有 `POST /v1/query` 也已接入同一分割服务，并默认启用自动人物 ROI；语言引导定位、属性提取和最终问答仍属于后续 PRD 模块，不能由当前分割结果替代。

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
