# 702solver 版本迭代记录

| 版本 | 日期 | 阶段 | 关键变化 | 12 任务真 API 指标（如有） |
|:----:|:----:|------|----------|---------------------------|
| — | — | R1–R3 | 四 Agent + Orchestrator + MessagePack + SQLite/FAISS；287 tests | Mock only |
| — | — | R4 (SNS) | SNS 速记、E2E 缓存、证据 Jaccard | Mock 5 任务：省 **72.1%** |
| — | — | R5–R6 | P0 LLM judge、P1 领域模板 | Mock：缓存 **60%** |
| — | — | 真 API 切换 | 暴露 role/agent_role、HNSW 虚高、跨域污染等 | 崩溃，rel≈0 |
| — | — | 9 项修复 | FlatIP、tag 过滤、BGE 嵌入、Summarizer retry | 48,292 tok，rel **0.664** |
| — | — | Prompt 压缩 | 模板骨架、记忆 2×120、AgentDropout | 文档最佳 **17,093** tok，rel **0.710** |
| **v0.5.0** | **2026-05-19** | **Round 5 稳定性** | JSON mode、调度分阶段、错误摘要不入库、relevance 评测修正 | **16,480** tok（**-66.4%**），rel **0.838**，25 calls |
| **v0.5.1** | **2026-05-19** | **质量验证** | `quality_validator`、问答记录、多轮 benchmark | 见 `quality_report.json` |
| **v0.6.0** | **2026-05-19** | **文献对标 + 实验充分性** | DB KB、tag 路由、utility 重排、`full_evaluation_suite` | 见 suite |
| **v0.6.1** | **2026-05-19** | **剪枝 + 消融** | 角色级 memory 注入、证据 900 字上限 | **16,523** tok，rel **0.848** |
| **v0.7.0** | **2026-05-19** | **P0 文献落地** | 0.78–0.82 模板带、Executor 免 CodeAct、摘要封顶 | 16,615 tok，rel 0.849 |
| **v0.8.0** | **2026-05-19** | **战略 v2 落地** | SafeSieve-lite、证据去重、Summarizer schema | **16,788** tok，rel **0.853** |
| **v0.9.0** | **2026-05-19** | **实验矩阵 + 质量 SafeSieve** | `RunOptions`、`experiment_matrix.py`、证据 750 | **16,339** tok（−66.7%） |
| **v0.10.0** | **2026-05-19** | **Token P0** | Summarizer 适配、Planner delta、无 judge 域匹配、Dropout 0.65 | **14,680** tok（**−70.1%**），DROP **8/12** |
| **v0.11.1** | **2026-05-19** | **意图门控** | `task_intent`、对比题禁摘要误复用、e4 修复 | **16,127** tok（**−67.1%**），rel **0.837**，**304 tests** |
| **v2 实验** | **2026-05-19** | **实验重设计** | full24、adversarial6、单 Agent 基线 | `experiment_design_v2.md` |
| **对话** | **2026-05-19** | **chat.py** | 全流水线交互、`chat_session.db` | `guides/chat_usage.md` |

## 文档索引（整理后）

| 类别 | 路径 |
|------|------|
| **总索引** | [`docs/README.md`](README.md) |
| **全记录** | [`PROJECT_RECORD.md`](PROJECT_RECORD.md) |
| **论文报告** | [`report_v1_complete.md`](report_v1_complete.md) |
| **答辩要点** | [`innovation_and_limitations.md`](innovation_and_limitations.md) |
| **实验 v2** | [`experiment_design_v2.md`](experiment_design_v2.md) |
| **战略 / 文献** | [`strategy/improvement_strategy_v2.md`](strategy/improvement_strategy_v2.md)、[`strategy/literature_review_2024_2026.md`](strategy/literature_review_2024_2026.md) |
| **实验 v1** | [`experiments/design_v1/`](experiments/design_v1/) |
| **优化轮次** | [`history/rounds/`](history/rounds/) |
| **长文叙事** | [`history/project_narrative.md`](history/project_narrative.md) |
| **归档** | [`archive/`](archive/)（Mock 报告、早期方案） |
| **生成** | `quality_records.md`、`experiment_matrix_results.md`、`full_evaluation_summary.md` |

旧文件名（如 `final_report.md`、`project_overview.md`）在 `docs/` 根目录保留**跳转存根**，正文已迁移。

## 当前推荐基线

**版本**：**v0.11.1**（质量优先：e4 修复；Token 报告同时保留 v0.10 **14,680** 作对照）  
**入口**：[`PROJECT_RECORD.md`](PROJECT_RECORD.md) · [`README.md`](README.md)  
**报告**：[`report_v1_complete.md`](report_v1_complete.md)（Token 最低 **14,680** / 当前推荐 **16,127**）  
**优化记录**：[`history/rounds/optimization_v0.11_intent_gate.md`](history/rounds/optimization_v0.11_intent_gate.md)  
**实验**：`python3 experiments/experiment_matrix_v2.py --phase all`  
**对话**：`python3 chat.py` → [`guides/chat_usage.md`](guides/chat_usage.md)

勿混淆：

- `archive/final_report_mock.md` — Mock 72.1%，已过时  
- `history/rounds/round4_*` 的「第四轮」≠ 叙事里 Prompt 压缩阶段  
- `history/project_narrative.md` 页眉可能停在 v0.6.1，以本表与 `PROJECT_RECORD` 为准
