# 多模态驱动的电商服饰细粒度语义增强与智能解析系统

本项目面向电商服饰场景，构建“商品图片 + 自然语言查询”联合驱动的细粒度语义增强与智能解析系统。系统目标包括服饰实例分割、语言引导局部区域定位、细粒度属性提取、多模态问答、商家内容生成，以及基于时尚专业知识库的 RAG 增强回答。

## 项目状态

PRD 3.1.1 服饰实例分割的功能主线已经完成并阶段收口。项目已实现 DeepFashion2 与 Fashionpedia 到统一八类 COCO 数据的转换、Mask2Former 混合训练、双数据集完整验证、FP16 精度与延迟实验、八类可视化验收，以及可复用模型实例的 FastAPI 推理接口。2,000 次混合训练 checkpoint 在完整 DeepFashion2 上达到 mask AP `60.58`，在 Fashionpedia 上达到 mask AP `45.74`，鞋子、包包和配饰均获得非零正式指标。默认部署配置继续指向该八分类 checkpoint 的 `512/853 + FP16 + 0.6` 推理档；主查询链路会在服饰分割前自动选择主要人物，并保留 `0.35` 的上下文区域。同参数 DeepFashion2 500 图复验达到 mask AP `45.68`，阈值 `0.6` 下匹配 Mask 平均 IoU 为 `89.81%`。默认 API 的八类代表样例全部命中，返回的 Mask 均非空且 Box 均有效。小物件专项微调结果保留为实验候选，但因完整 DeepFashion2 跨域验证未完成，不替换当前默认 checkpoint。RTX 3090 尚未达到 50 ms 目标；小物件、TensorRT 和更高性能 GPU 优化均延后到主流程打通之后。

PRD 3.1.2 语言引导局部区域定位的第一版端到端工程链路已经完成，但质量验收仍在进行。项目现有独立的 Fashionpedia 19 类局部 Mask 数据、中文到英文部位提示词归一化、人物 ROI、Grounding DINO 候选框、SAM-HQ Mask 细化、Mask 派生 Box，以及懒加载的 `POST /v1/localize` 路由。Fashionpedia 可直接覆盖领口、口袋和装饰，对肩部只有肩章这一部分监督，袖口、下摆、腰部和通用图案仍缺少直接 Mask 标注。Grounding DINO/SAM-HQ 环境与权重已在 AutoDL 验证，10 张衣领样例的 Top-1 `P50/R50/F1` 均为 `20%`，降低框阈值后的 Top-10 Recall 也仅为 `40%`。基于 `170,332` 个训练 Mask 的 19 类监督式 Mask2Former 已训练到 10,000 次，正式验证达到 mask AP `8.88`、AP50 `16.95`、AP75 `7.81`；衣领、翻领、袖子、口袋和领口 AP 分别为 `19.87`、`19.03`、`61.04`、`13.06` 和 `9.08`。10,000 次 checkpoint 被选为流程联调基线：已标注类别优先走监督式模型；缺失肩章预测时，肩部从 3.1.1 上衣、外套或连衣裙 Mask 的上缘两侧推导；袖口从监督式袖子 Mask 的主轴远端推导；下摆和腰部复用 3.1.1 服饰 Mask 的轮廓与位置；图案从服饰 Mask 内部的外观差异推导；无稳定结果及开放词汇查询再回退到 Grounding DINO + SAM-HQ。`POST /v1/query` 会复用同一次 3.1.1 分割结果组合 3.1.2 定位。首轮 8 类 API 检查虽然满足标签、ROI、Mask 和 Box 的结构条件，但视觉复核发现 shoulder 覆盖整个人，因此该轮 `8/8 PASS` 已被否决。验收现已对领口、口袋和 ruffle 加入 Mask IoU `>=0.50` 门槛及 GT 对照图；肩章只作为完整肩部的部分参考，肩部与其他无直接标注类别明确保留为视觉检查。严格复验中 ruffle 与 GT 的 Mask IoU 为 `92.0%`，说明该样例的整片裙摆本身就是 Fashionpedia 标注的荷叶边区域。这仍不代表达到 `92%` 总体准确率或 `30 ms` 延迟目标。

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
