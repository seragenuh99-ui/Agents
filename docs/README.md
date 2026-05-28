# 702solver 文档索引

> **主入口**：[STATUS_SUMMARY.md](STATUS_SUMMARY.md) — **目前情况一页总览**  
> **全记录**：[PROJECT_RECORD.md](PROJECT_RECORD.md)  
> **最后整理**：2026-05-20 · 当前版本 **v0.11.1** · 默认实验 **full24**

---

## 一、答辩 / 写稿（优先读这些）

| 文档 | 说明 |
|------|------|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | **系统架构 + 课程要求对照 + 实验入口（答辩主文档）** |
| **[STATUS_SUMMARY.md](STATUS_SUMMARY.md)** | **目前情况总结（数据一页纸）** |
| **[full24_controlled_comparison.md](full24_controlled_comparison.md)** | **控制变量对照（纯文本 vs 结构化，同 24 题）** |
| [full24_benchmark_results.md](full24_benchmark_results.md) | 24 题结构化子实验 |
| [PROJECT_RECORD.md](PROJECT_RECORD.md) | 全记录：指标、实验、对话、代码索引 |
| [report_v1_complete.md](report_v1_complete.md) | 论文体技术报告（Token 建议对照 v0.9：16,339） |
| [innovation_and_limitations.md](innovation_and_limitations.md) | 创新点、不足、30 秒答辩稿 |
| **[CURRENT_STATUS_AND_ROADMAP.md](CURRENT_STATUS_AND_ROADMAP.md)** | **通俗现状 + v2 数据 + 后续优化（推荐先读）** |
| [VERSION_HISTORY.md](VERSION_HISTORY.md) | 版本迭代表 |
| [experiment_design_v2.md](experiment_design_v2.md) | **当前**实验设计（full24、对抗、消融） |

---

## 二、运行与体验

| 文档 | 说明 |
|------|------|
| [guides/chat_usage.md](guides/chat_usage.md) | `python3 chat.py` 对话模式 |
| [quality_validation_guide.md](quality_validation_guide.md) | 质量评测方法与命令 |
| [human_factcheck_pack.md](human_factcheck_pack.md) | 人工 6 题事实核查表 |

---

## 三、策略与文献

| 文档 | 说明 |
|------|------|
| [strategy/improvement_strategy_v2.md](strategy/improvement_strategy_v2.md) | 优化战略 v2（含 v0.7 路线图附录） |
| [strategy/literature_review_2024_2026.md](strategy/literature_review_2024_2026.md) | 2024–2026 学界/工业调研 |

---

## 四、实验设计（历史 + 当前）

| 文档 | 说明 |
|------|------|
| [experiment_design_v2.md](experiment_design_v2.md) | v2 矩阵（推荐） |
| [experiments/design_v1/comprehensive_suite.md](experiments/design_v1/comprehensive_suite.md) | v0.6 E1–E10 套件设计 |
| [experiments/design_v1/matrix_v1.md](experiments/design_v1/matrix_v1.md) | v1 变体矩阵 H1–H8 |

---

## 五、自动生成 / 快照（勿手改）

| 文件 | 生成命令 |
|------|----------|
| [quality_records.md](quality_records.md) | `python3 experiments/quality_benchmark.py` |
| [experiment_matrix_results.md](experiment_matrix_results.md) | `python3 experiments/experiment_matrix.py` |
| [experiment_matrix_v2_results.md](experiment_matrix_v2_results.md) | `experiment_matrix_v2.py`（跑完后出现） |
| [full_evaluation_summary.md](full_evaluation_summary.md) | `python3 experiments/full_evaluation_suite.py` |
| [optimization_from_matrix.md](optimization_from_matrix.md) | `analyze_experiment_matrix.py` |

运行时 JSON / 日志 / 数据库：见 [`../output/README.md`](../output/README.md)（如 `output/results/optimized_benchmark_results.json`）。

---

## 六、开发历程（归档，细节完整保留）

### 优化轮次日志

| 文档 | 内容 |
|------|------|
| [history/rounds/round2_token_savings.md](history/rounds/round2_token_savings.md) | 第二轮：Planner/Retriever 缓存 |
| [history/rounds/round3_template_json.md](history/rounds/round3_template_json.md) | 第三轮：模板填空 + JSON |
| [history/rounds/round4_research_sns_e2e.md](history/rounds/round4_research_sns_e2e.md) | 第四轮：SNS/E2E/Dropout + 文献 |
| [history/rounds/round5_v0.5.0_stability.md](history/rounds/round5_v0.5.0_stability.md) | 第五轮：v0.5.0 稳定性 |
| [history/optimization_implementation_log.md](history/optimization_implementation_log.md) | 基础设施实施记录 |

### 长文叙事与早期方案

| 文档 | 内容 |
|------|------|
| [history/project_narrative.md](history/project_narrative.md) | 原 `project_overview.md` 全文 |
| [archive/improvement_plan_9fixes.md](archive/improvement_plan_9fixes.md) | 9 项生产修复 |
| [archive/solution_proposal_phases.md](archive/solution_proposal_phases.md) | 向量原生方案 Phase 1–4 |
| [archive/problems_and_literature_deep_dive.md](archive/problems_and_literature_deep_dive.md) | 7 问题 + 论文深读 |
| [archive/optimization_analysis_early.md](archive/optimization_analysis_early.md) | 早期工程 backlog |
| [archive/optimization_roadmap_v0.7.md](archive/optimization_roadmap_v0.7.md) | v0.7 数据驱动路线图（全文） |

### 已过时报告（仅作对照，勿引用数据）

| 文档 | 说明 |
|------|------|
| [archive/final_report_mock.md](archive/final_report_mock.md) | Mock 72.1%、MiniLM |
| [archive/experiment_report_v0.5.md](archive/experiment_report_v0.5.md) | 早期实验报告 |

---

## 七、目录结构

```
docs/
├── README.md                 ← 本索引
├── PROJECT_RECORD.md         ← 主入口
├── VERSION_HISTORY.md
├── report_v1_complete.md
├── experiment_design_v2.md
├── innovation_and_limitations.md
├── quality_validation_guide.md
├── quality_records.md        # 生成
├── experiment_matrix_results.md
├── guides/chat_usage.md
├── strategy/
├── experiments/design_v1/
├── history/rounds/
└── archive/
```

根目录：`README.md`、`claude.md`（Agent 系统规格）、`EXPERIMENT_STATUS.md`（已合并进 PROJECT_RECORD §8）。

---

## 八、勿混淆

| 避免 | 使用 |
|------|------|
| `archive/final_report_mock.md` 的 72.1% | `report_v1_complete.md` + v0.9 JSON |
| `history/project_narrative.md` 页眉 v0.6.1 | `VERSION_HISTORY.md` + `PROJECT_RECORD.md` |
| 调研「第四轮」编号 | `history/rounds/round4_*` vs 叙事中的 Prompt 压缩阶段 |
