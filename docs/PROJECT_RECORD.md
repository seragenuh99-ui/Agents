# 702solver 项目全记录（实验 · 对话 · 文档索引）

> **最后更新**：2026-05-20  
> **当前代码版本**：v0.11.1  
> **默认实验**：full24（24 题）— `python3 experiments/benchmark_full24.py`  
> **一页总览**：[STATUS_SUMMARY.md](STATUS_SUMMARY.md)  
> **本文档**：汇总「做了什么、效果如何、怎么跑、产物在哪」，作为答辩与交接的主入口。

---

## 1. 项目是什么

多 Agent 协作系统（Planner / Retriever / Executor / Summarizer + Orchestrator），面向赛题三维：

| 维度 | 实现要点 |
|------|----------|
| 低开销通信 | 短 prompt、JSON/MessagePack、**少调 LLM**（缓存 + 模板 + AgentDropout） |
| 非文本状态传递 | BGE 384 维向量，固定约 1536 bytes/次 |
| 共享记忆 | SQLite + FAISS，跨任务复用计划与摘要 |

---

## 2. 版本与核心数据（真 API）

| 版本 | 12 题 Token | vs 纯文本 49,076 | 备注 |
|------|------------:|------------------:|------|
| v0.7 | 16,615 | −66.1% | 论文报告 v1 主数据 |
| v0.8 | 16,788 | −65.8% | SafeSieve-lite + schema |
| **v0.9** | **16,339** | **−66.7%** | RunOptions + 实验框架 |

| 指标 | 数值 | 文件 |
|------|------|------|
| 纯文本基线 A | 49,076 tok | `output/results/pure_text_baseline_results.json` |
| 结构化无缓存 B | 37,208 tok | `output/results/cache_comparison_results.json` |
| 质量 12/12 | composite ≈0.95 | `output/results/quality_report.json` |
| 单元测试 | **296** passed | `pytest tests/` |
| 3-run 方差 (v0.7) | 15,963 ± 640 tok | `output/results/multi_run_benchmark_results.json` |

**勿用**：`docs/archive/final_report_mock.md`（原 `final_report.md`，Mock 72.1%）。

---

## 3. 实验体系（两套）

### 3.1 v1 实验矩阵（已完成一次）

- 设计：`docs/experiments/design_v1/`（`comprehensive_suite.md`、`matrix_v1.md`）
- 执行：`python3 experiments/experiment_matrix.py --tier1`
- 汇总：`output/results/experiment_matrix_report.json`、`docs/experiment_matrix_results.md`

### 3.2 v2 实验矩阵（针对不足重设计）— **推荐**

- 设计：**`docs/experiment_design_v2.md`**
- 任务集：**`experiments/benchmark_suites.py`**
  - `core12` — 原 12 题
  - `extended6` — 扩展 6 题
  - `adversarial6` — 对抗缓存 6 题（禁止误 E2E）
  - `full24` — 合计 24 题
- 执行：

```bash
cd /home/chen/projects/702solver

# 全流程（约 2–3h API）
python3 experiments/experiment_matrix_v2.py --phase all

# 分阶段 + 断点续跑
python3 experiments/experiment_matrix_v2.py --phase micro
python3 experiments/experiment_matrix_v2.py --phase core --skip-existing
python3 experiments/experiment_matrix_v2.py --phase extended
python3 experiments/experiment_matrix_v2.py --phase adversarial
python3 experiments/experiment_matrix_v2.py --phase quality
python3 experiments/experiment_matrix_v2.py --phase factcheck
```

- 日志（后台跑时）：`output/logs/experiment_matrix_v2_run.log`
- 产物：

| 文件 | 内容 |
|------|------|
| `output/results/experiment_matrix_v2_report.json` | 总汇总 |
| `docs/experiment_matrix_v2_results.md` | 表格 |
| `output/experiment_matrix_v2/*.json` | 各变体明细 |
| `single_agent_baseline_results.json` | 单 Agent 强基线 |
| `docs/human_factcheck_pack.md` | 人工 6 题核查表 |
| `docs/optimization_from_matrix.md` | 自动优化建议（跑完后生成） |

### 3.3 单任务 / 主基准（快速）

```bash
python3 experiments/optimized_benchmark.py          # → output/results/optimized_benchmark_results.json
python3 experiments/quality_benchmark.py            # 质量 + quality_records.md
python3 experiments/rejudge_quality.py              # LLM 评判
python3 experiments/multi_run_benchmark.py -n 3     # 方差
python3 experiments/single_agent_baseline.py        # 单 Agent 对照
```

### 3.4 v2 变体一览

| 变体 ID | 说明 |
|---------|------|
| `C_full_core12` | 全功能，12 题 |
| `B_nocache_core12` | 无 FAISS/缓存 |
| `C_no_dropout_core12` | 无 Executor Dropout |
| `C_no_e2e_core12` | 无 E2E |
| `C_warm_core12` / `C_cold_core12` | 热/冷记忆 |
| `C_full_full24` | 24 题 |
| `C_adversarial6_warm` | 对抗集 |
| `C_full_core12_llm_judge` | LLM 裁判质量 |

---

## 4. 对话功能（v0.9 全流水线）

**不是**单轮 ChatGPT，而是每条消息走完整多 Agent + **会话级共享记忆**。

### 启动

```bash
cd /home/chen/projects/702solver
python3 chat.py
# 或
python3 main.py chat
```

### 参数

```bash
python3 chat.py --db chat_session.db    # 记忆持久化（默认）
python3 chat.py --mode structured       # 省 Token（默认）
python3 chat.py --mode text             # 对比用
```

### 会话命令

| 命令 | 作用 |
|------|------|
| 直接输入 | 提问 |
| `/clear` | 清空记忆 |
| `/stats` | 系统/记忆状态 |
| `/tags solar energy` | 设置领域标签 |
| `/mode structured` | 切换模式 |
| `/quit` | 退出 |

### 代码位置

| 路径 | 说明 |
|------|------|
| `chat.py` | CLI 入口 |
| `src/chat/session.py` | `ChatSession` 封装 Orchestrator |
| `docs/guides/chat_usage.md` | 使用说明 |

### 环境

`.env` 中需配置：

```
DEEPSEEK_API_KEY=...
```

可选：`CHAT_MODEL=deepseek-chat`、`EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`

---

## 5. 文档地图（按用途）

> 完整目录树见 **`docs/README.md`**。旧路径保留短跳转存根，正文均在子目录。

| 用途 | 文档 |
|------|------|
| **文档索引** | `docs/README.md` |
| **通俗现状与优化路线** | `docs/CURRENT_STATUS_AND_ROADMAP.md` |
| **全记录（本文）** | `docs/PROJECT_RECORD.md` |
| **答辩主报告** | `docs/report_v1_complete.md`（Token 建议 v0.9：**16,339**） |
| **创新点与不足** | `docs/innovation_and_limitations.md` |
| **版本迭代** | `docs/VERSION_HISTORY.md` |
| **优化战略 / 文献** | `docs/strategy/improvement_strategy_v2.md`、`literature_review_2024_2026.md` |
| **实验设计 v2** | `docs/experiment_design_v2.md` |
| **实验设计 v1** | `docs/experiments/design_v1/` |
| **12 题问答** | `docs/quality_records.md`（生成） |
| **质量方法** | `docs/quality_validation_guide.md` |
| **人工核查** | `docs/human_factcheck_pack.md` |
| **对话** | `docs/guides/chat_usage.md` |
| **开发历程** | `docs/history/rounds/`、`docs/history/project_narrative.md` |
| **归档（勿引数据）** | `docs/archive/` |

---

## 6. 代码模块索引

| 模块 | 路径 |
|------|------|
| 编排 | `src/orchestrator.py` |
| 四 Agent | `src/agents/` |
| 协议 | `src/protocol/` |
| 记忆 | `src/memory/` |
| 向量状态 | `src/state/` |
| 质量评测 | `src/evaluation/quality_validator.py` |
| 实验开关 | `src/run_options.py` |
| 对话 | `src/chat/` |
| 实验脚本 | `experiments/` |

---

## 7. 技术方法一句话（答辩用）

1. **省 Token**：主要靠 **跨任务缓存 + 模板填空 + Executor Dropout**，不是单靠 MessagePack。  
2. **传状态**：384 维 BGE 向量 + memory_ref，不传长文。  
3. **记忆**：四级缓存（E2E / 模板 / LLM judge / 全生成）+ 标签防跨域 + SafeSieve 反馈。  
4. **实验 v2**：24 题 + 对抗集 + 单 Agent 基线 + 热/冷 + 消融。

---

## 8. 实验运行状态板（请自行更新）

| 任务 | 命令 | 日志/产物 | 状态 |
|------|------|-----------|------|
| v2 全流程 | `experiment_matrix_v2.py --phase all` | `output/logs/` | ✅ 2026-05-19 完成 `EXIT:0`（803s） |
| 主基准 | `optimized_benchmark.py` | `output/results/optimized_benchmark_results.json` | ✅ v0.9 已有 |
| v1 矩阵 | `experiment_matrix.py --tier1` | `output/results/experiment_matrix_report.json` | 见 `experiment_matrix_results.md` |

**检查 v2 是否结束**：

```bash
tail -30 output/logs/experiment_matrix_v2_run.log
grep EXIT output/logs/experiment_matrix_v2_run.log
ls -la output/results/experiment_matrix_v2_report.json docs/experiment_matrix_v2_results.md 2>/dev/null
ls output/experiment_matrix_v2/*.json 2>/dev/null | wc -l
```

**已有稳定数据（通常无需重跑）**：

| 数据 | 文件 |
|------|------|
| v0.9 主基准 | `output/results/optimized_benchmark_results.json` |
| 纯文本 A | `output/results/pure_text_baseline_results.json` |
| 质量 12/12 | `output/results/quality_report.json` |
| v1 矩阵 | `output/results/experiment_matrix_report.json` |

**对话**：`python3 chat.py` · 默认 `output/databases/chat_session.db`（勿提交含隐私的 db）

---

## 9. 环境要求

- Python 3.10+
- `pip install -r requirements.txt`（含 sentence-transformers、faiss、msgpack 等）
- DeepSeek API Key（`.env`）
- 跑全量实验约需 **2–3 小时** API 时间

---

## 10. 变更日志（摘要）

| 日期 | 内容 |
|------|------|
| 2026-05-19 | v0.5–v0.7 真 API 稳定、论文报告 v1 |
| 2026-05-19 | v0.8 SafeSieve-lite、证据截断、Summarizer schema |
| 2026-05-19 | v0.9 RunOptions、experiment_matrix、质量反馈修复 |
| 2026-05-19 | 实验 v2 设计（full24、adversarial、单 Agent） |
| 2026-05-19 | **对话功能** `chat.py` + `PROJECT_RECORD.md` |
| 2026-05-19 | **文档整理**：`docs/README.md`、`archive/`、`history/`、`strategy/` |
| 2026-05-19 | **目录整理**：`output/` 统一产物，`src/paths.py` |
| 2026-05-19 | **v2 实验跑完** + `CURRENT_STATUS_AND_ROADMAP.md` 记录 |

---

*维护：重大实验跑完后请更新第 2 节数据表与第 8 节状态。*
