# 702solver：面向低开销通信与共享记忆的多智能体协作系统

**技术报告 v1.0（首个相对完善版本）**  
**系统版本**：v0.7.0  
**日期**：2026-05-19  
**LLM 后端**：DeepSeek-V3（`deepseek-chat`）  
**嵌入模型**：`BAAI/bge-small-en-v1.5`（384 维）

---

## 摘要

本文报告多智能体系统 **702solver** 的设计、实现与评测。系统由 Orchestrator 调度 Planner、Retriever、Executor、Summarizer 四类 Agent，通过 MessagePack 结构化协议通信，以 384 维语义向量实现非文本状态传递，并以 SQLite+FAISS 构建跨任务共享记忆。在借鉴 SNS-Core 速记、AgentDropout 动态剪枝、E2E/模板多级缓存等机制的基础上，v0.7.0 在 12 项跨领域基准任务（能源、安全、数据库）上完成真 API 评测。

**主要结果**（相对纯文本长报告基线 49,076 tokens）：

| 指标 | 数值 |
|------|------|
| 单次完整基准总 Token | **16,615**（节省 **66.1%**） |
| 3 次独立重复实验 | **15,963 ± 640** tokens（节省 **67.5 ± 1.3%**） |
| 平均语义相关性（BGE） | **0.849**（3-run：**0.835 ± 0.008**） |
| 质量验证（12/12 通过） | 综合分 **0.948**（LLM Judge） |
| E2E 缓存命中 | s3 任务 **29 ms / 0 API 调用** |
| 单元测试 | **294** 项通过 |

本报告取代 `docs/archive/final_report_mock.md`（原 `final_report.md`）中基于 MockLLM 的早期数据（如 72.1%、MiniLM），作为当前答辩与写稿的**唯一推荐数据源**。

**关键词**：多智能体系统；低开销通信；MessagePack；语义向量状态传递；FAISS；跨任务缓存；Token 效率

---

## 1 引言

### 1.1 背景与赛题要求

赛题要求构建多 Agent 协作系统，在三个维度达到可度量目标：

1. **低开销通信**：Agent 间避免冗长自然语言，采用结构化、可压缩的消息格式。  
2. **非文本状态传递**：用固定维度向量等方式传递状态，接收方直接用于检索或相似度计算。  
3. **共享记忆复用**：跨任务积累知识，同域或相似任务自动复用历史结果。

纯端到端「结构化 vs 文本」对比易受 LLM 输出长度波动干扰；因此本系统将**协议层微基准**、**LLM API 用量**与**任务质量验证**分离测量。

### 1.2 贡献概述

- 四 Agent + Orchestrator 架构，拓扑分层并行调度。  
- MessagePack 短字段映射 + 消息引用化（memory ref 代替全文）。  
- BGE 384 维向量经 StateExchangeBus 传递；FAISS FlatIP + 标签过滤 + utility 重排。  
- 四级缓存：E2E（cos≥0.85）→ P1 领域模板 → P0 LLM Judge（0.55–0.85）→ 全新生成。  
- v0.7 专项优化：0.78–0.82 模板带、KB 命中时 Executor 跳过 CodeAct LLM、Summarizer 输出封顶、Planner 列表 JSON 修复。

---

## 2 相关工作

| 方向 | 代表工作 | 与本系统关系 |
|------|----------|--------------|
| 速记通信 | SNS-Core | ✅ SNS 风格 prompt 骨架 |
| 动态剪枝 | AgentDropout (ACL 2025) | ✅ template_score≥0.70 时 Executor Dropout |
| 图剪枝 | AgentPrune / Cut the Crap | ⚠️ 部分（证据截断、tag 路由） |
| 代码化交互 | CodeAgents | ⚠️ 结构化 JSON 输出，未完全代码化 |
| 共享记忆 | Memory Sharing, PlugMem | ✅ SQLite+FAISS；❌ 无知识图谱分层 |
| 渐进压缩 | SafeSieve | ❌ 未实现（路线图 P4） |

详细文献表见 `docs/strategy/literature_review_2024_2026.md`。

---

## 3 系统设计

### 3.1 总体架构

```
任务输入 → Orchestrator
              ├─ Planner（DAG 计划 + 缓存策略）
              ├─ Retriever（记忆 / KB / 外部检索）  ← 同层并行
              ├─ Executor（沙箱代码，可选跳过 LLM）
              └─ Summarizer（结构化 JSON 报告）
                    ↓
         SQLite 元数据 + FAISS 向量索引（跨任务复用）
```

**协作阶段**（v0.5+）：Retriever → Executor → Summarizer **分阶段串行**，避免 Summarizer 与 Executor 并行竞态；同层 subtask 仍由 ThreadPoolExecutor 并行。

### 3.2 结构化通信协议

- 消息头 `h`、动作 `a`、状态 `s`、载荷 `p` 分区编码。  
- `ActionType` 枚举：plan / retrieve / execute / summarize 等。  
- 调度类消息传 **memory_refs** 与能力列表，不传 evidence 全文。

### 3.3 非文本状态传递

- `EmbeddingEngine`：`BAAI/bge-small-en-v1.5`，输出 float32[384]。  
- `StateExchangeBus.transfer()`：将上下文编码为固定 **1,536 bytes** 向量包，供下游相似度与记忆检索。

### 3.4 共享记忆与缓存

| 层级 | 触发条件 | 行为 |
|------|----------|------|
| E2E | cos≥0.85 + 标签一致 | 0 次 LLM，直接返回历史摘要 |
| P1 模板 | 同域 ≥2 条 evidence | 领域模板 + 填空 |
| P0 Judge | 0.55≤cos<0.85 | LLM 判定是否同类 → 模板填空 |
| FULL_GEN | 其余 | 完整流水线 |

**检索增强**（v0.6+）：`0.85×相似度 + 0.15×访问计数` utility 重排；Retriever 按 tags 路由 database KB；证据单条上限 **900 字符**。

### 3.5 v0.7.0 关键实现

| 模块 | 变更 |
|------|------|
| Planner | 0.82 直接命中、0.78 模板带、列表型 plan JSON 规范化 |
| Executor | KB 命中且非代码任务时跳过 `_generate_code` |
| Summarizer | 最多 5 findings / 3 facts；拒绝 `_error` 摘要入库 |
| Orchestrator | AgentDropout：高模板分跳过 Executor LLM |

---

## 4 实验设置

### 4.1 环境

| 项目 | 配置 |
|------|------|
| OS | Linux 6.6.87 (WSL2) |
| Python | 3.10 |
| LLM | DeepSeek `deepseek-chat` |
| Embedding | `BAAI/bge-small-en-v1.5` |
| 向量索引 | FAISS IndexFlatIP |
| 元数据 | SQLite3 |

### 4.2 基准任务集

12 项任务，覆盖 **energy（4）**、**security（4）**、**database（4）**，定义于 `experiments/benchmark_tasks.py`。每项含 `expected_topics` 供质量验证。

### 4.3 对比与消融

| 代号 | 配置 | 脚本 |
|------|------|------|
| **A** | 纯文本长报告基线（每任务独立、无缓存） | `pure_text_baseline.py` |
| **B** | 结构化流水线，关闭跨任务 FAISS 缓存 | `comparison_benchmark.py` |
| **C** | 全功能 v0.7（E2E+模板+Dropout） | `optimized_benchmark.py` |

### 4.4 评测指标

- **Token**：DeepSeek API 返回的 prompt + completion tokens 累计。  
- **Relevance**：任务描述 embedding 与摘要 embedding 的余弦相似度（BGE）。  
- **Quality**：`quality_validator` 综合分（主题覆盖 + 相关性 + 可选 LLM Judge，阈值 0.65）。  
- **稳定性**：`multi_run_benchmark.py -n 3` 报告 mean ± std。

### 4.5 可复现命令

```bash
# 主基准（单次）
python3 experiments/optimized_benchmark.py

# 消融
python3 experiments/pure_text_baseline.py
python3 experiments/comparison_benchmark.py
python3 experiments/ablation_analysis.py

# 3 次重复
python3 experiments/multi_run_benchmark.py -n 3

# 质量 + 问答留档
python3 experiments/quality_benchmark.py
python3 experiments/rejudge_quality.py

# 系统微基准（无 API）
python3 experiments/collect_system_metrics.py
```

---

## 5 实验结果

### 5.1 主结果：通信效率（真 API）

**表 1 — 12 任务汇总（v0.7.0 单次运行，`optimized_benchmark_results.json`）**

| 指标 | 纯文本 A | 结构化无缓存 B | 全功能 C (v0.7) |
|------|:--------:|:--------------:|:---------------:|
| 总 Token | 49,076 | 37,208 | **16,615** |
| 相对 A 节省 | — | 24.2% | **66.1%** |
| API 调用次数 | 36 | — | **32** |
| 平均 Relevance | — | — | **0.849** |
| 总耗时 | ~501 s | — | **120 s** |

**表 2 — 三次独立重复实验（`multi_run_benchmark_results.json`）**

| 指标 | Run 1 | Run 2 | Run 3 | Mean ± Std |
|------|:-----:|:-----:|:-----:|:----------:|
| 总 Token | 15,476 | 16,688 | 15,725 | **15,963 ± 640** |
| 平均 Relevance | 0.842 | 0.826 | 0.835 | **0.835 ± 0.008** |
| API 调用 | 31 | 32 | 30 | **31.0 ± 1.0** |
| vs A 节省 | 68.5% | 66.0% | 68.0% | **67.5 ± 1.3%** |

> 说明：LLM 输出长度存在固有方差（Run 2 中 s2、d2  completion 偏高）；3 次运行的 Token 标准差约 **4%**，仍显著优于纯文本基线。

### 5.2 消融：各层贡献

**表 3 — 消融（`ablation_analysis.json`）**

| 配置 | Token | 相对纯文本节省 | 说明 |
|------|:-----:|:--------------:|------|
| A 纯文本 | 49,076 | 0% | 每任务长报告 |
| B 结构化无缓存 | 37,208 | 24.2% | SNS+JSON+refs |
| C 全功能 v0.7 | 16,615 | **66.1%** | +E2E/模板/Dropout |
| Δ (B→C) | −20,593 | +44.7 pp | 缓存层单独贡献 |

B→C 在 B 的基础上再节省 **55.3%**，表明跨任务缓存与模板填空是主要增益来源。

### 5.3 策略分布

**表 4 — 12 任务策略命中（v0.7 单次）**

| 策略 | 任务数 | 代表任务 |
|------|:------:|----------|
| FULL_GEN | 4 | e1, s1, d1, d2 |
| TEMPLATE_FILL_DROP | 4 | e2, s4, d3, d4 |
| TEMPLATE_FILL | 2 | e3, s2 |
| P1_TEMPLATE_DROP | 1 | e4 |
| E2E_EXACT | 1 | **s3**（0 call, ~29ms） |

### 5.4 逐任务明细（v0.7.0）

| 任务 ID | 领域 | 策略 | API | Token | Relevance | 耗时(s) |
|---------|------|------|:---:|:-----:|:---------:|:-------:|
| e1_solar_basics | solar | FULL_GEN | 2 | 1,023 | 0.772 | 8.9 |
| e2_solar_advanced | solar | TEMPLATE_FILL_DROP | 5 | 2,121 | 0.897 | 12.9 |
| e3_wind_energy | wind | TEMPLATE_FILL | 3 | 1,219 | 0.864 | 7.8 |
| e4_renewable_comparison | solar | P1_TEMPLATE_DROP | 2 | 1,206 | 0.790 | 8.6 |
| s1_python_vuln | security | FULL_GEN | 3 | 2,051 | 0.906 | 14.8 |
| s2_web_security | security | TEMPLATE_FILL | 4 | 1,802 | 0.842 | 11.3 |
| s3_python_vuln_v2 | security | **E2E_EXACT** | **0** | **0** | 0.800 | **0.03** |
| s4_code_review | security | TEMPLATE_FILL_DROP | 2 | 1,154 | 0.825 | 8.4 |
| d1_query_optimization | database | FULL_GEN | 2 | 1,265 | 0.895 | 11.7 |
| d2_nosql_comparison | database | FULL_GEN | 4 | 2,231 | 0.864 | 15.9 |
| d3_db_performance | database | TEMPLATE_FILL_DROP | 2 | 1,172 | 0.859 | 9.0 |
| d4_data_modeling | database | TEMPLATE_FILL_DROP | 3 | 1,371 | 0.871 | 9.9 |
| **合计** | | | **32** | **16,615** | **0.849** | **120** |

### 5.5 任务嵌入相似度与 E2E 触发

对 12 项任务描述做 BGE 编码，仅 **s1_python_vuln ↔ s3_python_vuln_v2** 余弦相似度 **0.882 > 0.85**，与 s3 的 E2E 命中一致（`embedding_similarity_matrix.json`）。其余跨域对均 <0.85，避免错误复用。

### 5.6 质量验证

**表 5 — 质量基准（`quality_report.json`，含 LLM Judge）**

| 指标 | 数值 |
|------|------|
| 综合通过率 | **12 / 12** |
| 平均综合分 | **0.948** |
| 平均相关性 | **0.846** |
| 平均主题覆盖率 | **95.8%** |
| 总 Token | 16,799 |

完整问答见 `docs/quality_records.md`（2026-05-19 重新生成）。

### 5.7 协议与状态微基准

**表 6 — 系统微基准（`system_metrics.json`，无 API）**

| 子系统 | 指标 | 数值 |
|--------|------|------|
| 协议（调度消息，无 embedding） | MessagePack / JSON | **196 / 310 B** |
| | 体积减少 | **36.8%** |
| 协议（含 384d 状态向量） | MessagePack / JSON | 3634 / 2207 B |
| | 说明 | 向量载荷使 msgpack 变大；调度层与状态层分开评测 |
| 状态传递 | 向量固定体积 | **1,536 B** |
| | 等效文本（~6.9 KB 上下文） | 6,950 B |
| | 体积减少 | **77.9%** |
| | 吞吐 | **23,096 transfers/s** |
| 记忆检索 (200 条) | 写入延迟 | 4.36 ms/条 |
| | 语义搜索 p50 | **0.68 ms** |
| 测试 | pytest 通过 | **294** |

### 5.8 与早期 Mock 数据对照

| 指标 | Mock 阶段 (`final_report.md`) | 本报告 v0.7 |
|------|------------------------------|-------------|
| Token 节省 | 72.1%（5 任务估算） | **66.1%**（12 任务真 API） |
| 嵌入模型 | MiniLM | **BGE small en v1.5** |
| E2E 延迟 | 5 ms（模拟） | **29 ms**（真 API 路径） |
| 质量验证 | 无 | **12/12，0.948** |

Mock 阶段比例更高，因任务少且 LLM 为规则模拟；真 API 下 **66–67%** 仍为可复现的稳健结果。

---

## 6 讨论

### 6.1 为何协议节省与端到端节省分离

结构化协议在微基准上相对 JSON 节省约 **37%**（调度消息），但端到端 Token 主要由 **LLM prompt/completion** 决定。本系统通过 SNS 骨架、模板填空、E2E 跳过 LLM 在应用层实现主要节省，与文献中 AgentDropout、SNS-Core 的结论一致。

### 6.2 质量与效率权衡

v0.7 在压缩 Token 的同时，相关性从修复初期的 **0.664** 提升至 **0.849**，质量验证 **12/12**。模板填空任务（e2、d4）在综合分上常 >0.95，说明缓存未显著损害主题覆盖。

### 6.3 局限性

1. **基准规模**：12 任务、3 次重复；未覆盖工具调用、多轮对话、人机协作。  
2. **单 LLM 供应商**：仅 DeepSeek；未测试 GPT-4/Claude 泛化。  
3. **无 LangGraph/CrewAI 对照**：工业框架基线待补。  
4. **SafeSieve / 轨迹蒸馏**：文献中的渐进压缩与 MemCollab 式蒸馏尚未实现。  
5. **HNSW**：生产数据量大时可切换；当前基准用 FlatIP 保证精确 top-k。

---

## 7 结论

702solver v0.7.0 在赛题三维目标上取得可度量进展：

- **低开销通信**：真 API 下相对纯文本节省 **66–67%**（16.6k vs 49.1k tokens）。  
- **非文本状态传递**：384 维固定向量，相对同等语义文本体积减少 **~78%**。  
- **共享记忆复用**：E2E 命中实现 **0 API / 29 ms**；多级模板与 utility 重排支撑跨任务复用。

本报告 v1.0 整合单次主基准、3-run 方差、消融、质量验证与系统微基准，作为项目首个相对完善的正式技术报告。后续版本计划：SafeSieve-lite 缓存反馈、更大任务集、多框架对照实验。

---

## 参考文献

1. AgentDropout: Dynamic Agent Elimination for Token-Efficient Multi-Agent Collaboration. ACL, 2025. arXiv:2503.18891  
2. Cut the Crap / AgentPrune: Economical Communication Pipeline for LLM-based Multi-Agent Systems. ICLR, 2025. arXiv:2410.02506  
3. CodeAgents: … arXiv:2507.03254  
4. SafeSieve: … arXiv:2508.11733  
5. Memory Sharing in Multi-Agent Systems. arXiv:2404.09982  
6. BGE Embeddings: `BAAI/bge-small-en-v1.5` (Hugging Face)

---

## 附录 A：数据文件索引

| 文件 | 内容 |
|------|------|
| `optimized_benchmark_results.json` | v0.7 主基准 |
| `multi_run_benchmark_results.json` | 3-run 方差 |
| `pure_text_baseline_results.json` | 消融 A |
| `cache_comparison_results.json` | 消融 B |
| `ablation_analysis.json` | 消融汇总 |
| `embedding_similarity_matrix.json` | 任务相似度矩阵 |
| `quality_report.json` | 质量验证原始数据 |
| `system_metrics.json` | 协议/状态/记忆微基准 |
| `docs/quality_records.md` | 12 题问答留档 |
| `docs/VERSION_HISTORY.md` | 版本迭代 |

## 附录 B：版本演进摘要

| 版本 | 12 任务 Token | 平均 Relevance | 备注 |
|------|:-------------:|:--------------:|------|
| 真 API 修复后 | 48,292 | 0.664 | 9 项 bugfix |
| Prompt 压缩 | 17,093 | 0.710 | 模板骨架 |
| v0.5.0 | 16,480 | 0.838 | JSON mode、分阶段调度 |
| v0.6.1 | 16,523 | 0.848 | 角色级 memory 限制 |
| **v0.7.0** | **16,615** | **0.849** | Executor 剪枝、摘要封顶 |

---

*报告生成：实验脚本自动产出 + 本文档人工编排。复现请先配置 `.env` 中 `DEEPSEEK_API_KEY`。*
