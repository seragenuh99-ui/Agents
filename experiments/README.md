# 实验脚本说明

所有脚本从**仓库根目录**执行：`python3 experiments/<name>.py`

产物默认写入 [`../output/`](../output/README.md)。

## 主基准与质量（默认 **full24 = 24 题**）

| 脚本 | 作用 |
|------|------|
| **`../run.py`** | **统一入口**（中文向导、`--mode 1,2,3`、`--suite continuous12` 等） |
| **`full24_controlled_comparison.py`** | **控制变量对照**（A/C/B + 对抗；`--skip-pure-text` 复用已有 A） |
| **`benchmark_full24.py`** | 结构化子实验（全功能 + 无缓存 + 对抗） |
| `pure_text_baseline.py` | 纯文本基线（`--suite full24`） |
| `optimized_benchmark.py` | 快速基准（默认 `--suite full24`，可 `--suite core12`） |
| `quality_benchmark.py` | 质量验证 + 生成 `docs/quality_records.md` |
| `rejudge_quality.py` | 对已有 quality_report 补 LLM 评判 |
| `multi_run_benchmark.py` | 多次重复测方差 |

## 基线与消融

| 脚本 | 作用 |
|------|------|
| `pure_text_baseline.py` | 纯文本多 Agent 基线 A |
| `comparison_benchmark.py` | 有/无缓存对比 |
| `ablation_analysis.py` | 汇总 A/B/C 消融表 |
| `single_agent_baseline.py` | 单 Agent 强基线 |

## 实验矩阵

| 脚本 | 作用 |
|------|------|
| `experiment_matrix.py` | v1 矩阵（`--tier1` / `--all`） |
| `experiment_matrix_v2.py` | **v2 矩阵**（`--phase all`） |
| `experiment_common.py` | 矩阵共享运行逻辑 |
| `analyze_experiment_matrix.py` | 从矩阵报告生成优化建议 |
| `benchmark_suites.py` | core12 / extended6 / adversarial6 |
| `benchmark_tasks.py` | 12 题定义 |

## 工具与其它

| 脚本 | 作用 |
|------|------|
| `collect_system_metrics.py` | 微基准（无 API） |
| `embedding_matrix.py` | 任务嵌入相似度矩阵 |
| `full_evaluation_suite.py` | E1–E10 一键套件 |
| `factcheck_export.py` | 导出人工核查包 |
| `runner.py` | `main.py experiment` 调用 |
| `tasks.py` | `main.py` 演示任务组 |

## 入口对照

| 用途 | 命令 |
|------|------|
| 交互对话 | `python3 chat.py` |
| 单任务调试 | `python3 main.py single "..."` |
| 对比实验 | `python3 main.py experiment` |
