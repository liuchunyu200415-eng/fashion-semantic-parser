# 多模态驱动的电商服饰细粒度语义增强与智能解析系统

本项目面向电商服饰场景，构建“商品图片 + 自然语言查询”联合驱动的细粒度语义增强与智能解析系统。系统目标包括服饰实例分割、语言引导局部区域定位、细粒度属性提取、多模态问答、商家内容生成，以及基于时尚专业知识库的 RAG 增强回答。

## 项目状态

当前阶段聚焦 PRD 3.1.1 服饰实例分割功能。已经完成 DeepFashion2 到 COCO 的数据转换、Mask2Former 训练与完整验证集评估、FP16 部署配置、延迟基准、可视化验收，以及可复用模型实例的 FastAPI 推理接口。当前经过训练和验证的类别为上衣、裤子、裙子、外套和连衣裙。工程已加入 Fashionpedia 八类映射、标注审计和 COCO 转换入口，用于补充鞋子、包包和配饰；完整数据下载、混合训练及三类正式指标仍未完成。

分割服务入口为 `POST /v1/segment`，返回每个服饰实例的 Mask、边界框、类别和置信度。现有 `POST /v1/query` 也已接入同一分割服务，但语言引导定位、属性提取和最终问答仍属于后续 PRD 模块，不能由当前分割结果替代。

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
