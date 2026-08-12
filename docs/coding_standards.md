# 编码规范与自动门禁

本项目以团队《编码规范》为准，并通过自动化检查防止新增不规范代码。新增或修改代码必须使用 Python 3.10，遵循 Black、isort、Flake8、Mypy 和 Pylint 的检查结果；公共模块、类、函数和方法必须提供清晰的文档字符串，复杂函数使用 Google 风格的 `Args`、`Returns` 和 `Raises` 段落。

## 本地检查

安装开发依赖后运行：

```bash
python -m pip install -e ".[dev]"
bash scripts/check_quality.sh
```

该命令依次检查格式、导入顺序、静态规范、类型、严重 Pylint 问题、项目专项编码规则、测试以及最低 `75%` 的包级覆盖率。GitHub Actions 会在 `main` 分支推送和 Pull Request 上执行同一套门禁。

## 历史技术债基线

`configs/coding_standard_baseline.json` 记录现有历史代码中的结构性技术债，包括缺失文档、超长模块、参数过多、复杂 `Any` 字典和隐式字符串拼接。该文件不是豁免清单：检查脚本禁止任何指标增加；修复历史问题时必须同步降低对应基线，不能用提高基线掩盖回归。

以下规则按零容忍执行：

- 模块缺失文档字符串；
- 可变默认参数和裸 `except`；
- 相对导入和通配符导入；
- 测试代码直接使用 `print`；
- Tab 字符或不规范的文件末尾换行。

## 提交信息

所有新提交使用 Conventional Commits：

```text
feat(localization): add spatial reranking
fix(training): validate empty masks
docs: explain PRD 3.1.2 acceptance scope
```

允许的类型为 `build`、`chore`、`ci`、`docs`、`feat`、`fix`、`perf`、`refactor`、`revert`、`style` 和 `test`。CI 会检查 Pull Request 中的每个提交以及直接推送到 `main` 的提交。
