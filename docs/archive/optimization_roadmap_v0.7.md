# 方案对比与进一步优化路线图

> 基于完整实验矩阵（A/B/C 消融 + 质量验证 + 嵌入矩阵）与 2024–2026 文献调研。  
> 数据文件：`full_evaluation_report.json`、`ablation_analysis.json`、`embedding_similarity_matrix.json`

---

## 一、实验结论：三层节省从哪来

| 层级 | 模式 | Token | 相对纯文本 A | 机制 |
|:----:|------|------:|:------------:|------|
| **A** | 纯文本多 Agent | 49,076 | 0% | 长 prompt、散文输出、全文传递 |
| **B** | 结构化、无跨任务缓存 | 37,208 | **24.2%** | SNS + MessagePack + JSON + refs |
| **C** | 全功能 v0.7.0（实测） | **16,615** | **66.1%** | B + 缓存 + Dropout + KB + 剪枝 |

**B→C 再省 55.6%**（37,208→16,523）：说明赛题核心不是「换 JSON」，而是 **少调用 LLM + 少传上下文**。

**v0.7.0 相对 v0.6.1**：token 16,523→16,615（持平）；**relevance 0.848→0.855**；**TEMPLATE_FILL_DROP 1→4**（Executor 免 CodeAct LLM 生效）；e2 relevance **0.899**。

**v0.7 新增机制**：
- Planner：cos>0.82 直复用 + 0.78–0.82 免 judge 模板带
- Executor：有 KB 且非代码任务 → 不调 CodeAct LLM
- Summarizer：最多 5 findings / 3 facts
- 记忆访问计数参与 utility 重排

### 策略分布（12 任务，C 模式）

| 策略 | 次数 | 含义 |
|------|:----:|------|
| TEMPLATE_FILL | 7 | 嵌入相似 → 模板填空 |
| FULL_GEN | 3 | 三域冷启动 |
| P1_TEMPLATE_DROP | 1 | 领域模板 + 跳过 Executor |
| E2E_EXACT | 1 | s3 复用 s1，0 token |

### 嵌入矩阵启示（BGE-small）

| cos 区间 | 任务对示例 | 系统行为 |
|:--------:|------------|----------|
| >0.85 | s1↔s3 (0.882) | E2E 全复用 ✅ |
| 0.70–0.85 | e1↔e2 (0.775) | 模板填空 ✅ |
| 0.50–0.70 | e1↔e3 (0.71) | 可走 P0 judge + 模板 |
| <0.50 跨域 | energy↔security | 正确不缓存 ✅ |

**瓶颈**：同域不同角度（如 solar 技术 vs solar 政策）cos 常落在 0.60–0.75，依赖 P0 judge，每次 ~50 token。

---

## 二、方案横向对比（我们 vs 文献 vs 工业）

| 方案 | Token 潜力 | 质量风险 | API 可实现 | 我们 |
|------|:----------:|:--------:|:----------:|:----:|
| 纯文本 Agent | 基线 | 低 | ✅ | A 对照 |
| SNS / 短 prompt | 高 | 低 | ✅ | ✅ |
| 消息 refs | 高 | 低 | ✅ | ✅ |
| E2E / Planner 缓存 | 很高 | 中（错域） | ✅ | ✅ + tag 过滤 |
| AgentDropout | 中 | 中 | ✅ | ✅ Executor |
| AgentPrune 边剪枝 | 中 | 低 | ✅ | ⚠️ v0.6.1 收紧 memory 注入 |
| CodeAgents 伪代码 | 高 | 中 | ✅ | ⚠️ 部分（模板骨架） |
| SafeSieve 渐进剪枝 | 中 | 低 | ✅ | ❌ 建议 P5 |
| LLM 长 prompt cache | 低/负 | — | ✅ | ❌ 已证得不偿失 |
| LatentMAS 隐状态 | 很高 | — | ❌ | 不可行 |
| 真实 RAG / 搜索 | 质量↑ | 低 | ✅ | ❌ 仍为 KB 模拟 |
| Mem0 式蒸馏记忆 | 中 | 低 | ✅ | ❌ P4 |

---

## 三、已实施优化清单（v0.5 → v0.6.1）

| 版本 | 优化 | 预期效果 |
|------|------|----------|
| v0.5.0 | JSON mode、调度分阶段、错误摘要不入库 | 稳定性、relevance↑ |
| v0.5.1 | quality_benchmark + LLM judge | 可审计 12 题 Q&A |
| v0.6.0 | Database KB + tag 路由 | DB 域 evidence 对齐 |
| v0.6.0 | Memory utility 重排 | 复用更准的记忆 |
| v0.6.0 | 主题同义词评测 | 减少误报 |
| v0.6.1 | 按角色收紧 `suggested_memories` | prompt 再瘦身 |

---

## 四、还能怎么优化（按 ROI 排序）

### P0 — 高收益、低风险（建议下一 sprint）

| # | 措施 | 依据 | 预估 |
|---|------|------|------|
| 1 | **跑 v0.6 benchmark + 3 轮方差** | 答辩「充分性」 | 数据可信 |
| 2 | **Summarizer 输出 schema 固定字段** | CodeAgents | completion↓10–15% |
| 3 | **Retriever 证据上限**（combined_results 最多 800 字） | AgentPrune | prompt↓5–10% |
| 4 | **e1↔e2 类对启用「轻量 judge」**（cos∈[0.70,0.85]） | 矩阵显示 0.775 | 多 1–2 次 E2E/模板 |

### P1 — 中收益、中工作量

| # | 措施 | 说明 |
|---|------|------|
| 5 | **SafeSieve-lite** | 记录每次 template fill 的 `composite_pass`，失败模板降权 |
| 6 | **P4 证据蒸馏** | 同域 ≥4 条 evidence 合并为 1 条 `distilled` |
| 7 | **升级嵌入 bge-base / e5** | 扩大 0.70–0.85 模板区，略增 CPU |
| 8 | **单 Agent 强基线** | 1 个 LLM 长链对照，回应「多 Agent 是否必要」 |

### P2 — 质量向（非 token）

| # | 措施 |
|---|------|
| 9 | 接真实检索 API 或更大 KB |
| 10 | 5% 人工 fact-check 写入报告 |
| 11 | CrewAI/LangGraph 同任务 token 对照（若时间允许） |

### 不建议

- 拉长 system prompt 蹭 DeepSeek KV cache（已证 completion 膨胀）
- 无 tag 过滤的 E2E（跨域污染）
- 为省 token 去掉 Summarizer（赛题要求）

---

## 五、推荐实验完善 checklist

```bash
# 1. v0.6 主 benchmark（更新 optimized_benchmark_results.json）
python3 experiments/optimized_benchmark.py

# 2. 消融表
python3 experiments/ablation_analysis.py

# 3. 质量 + 问答（v0.6 代码路径）
python3 experiments/quality_benchmark.py --no-judge
python3 experiments/rejudge_quality.py

# 4. 方差
python3 experiments/multi_run_benchmark.py -n 3

# 5. 总表
python3 experiments/full_evaluation_suite.py --skip-existing
```

**答辩最小数据集**：A/B/C 表 + `quality_records.md` 2 页 + 策略饼图 + s1/s3 E2E 延迟截图。

---

## 六、综合判断

| 维度 | 当前水平 | 再优化上限（API 约束下） |
|------|----------|-------------------------|
| Token | **66.4%** | 70–75%（P0 项叠加） |
| 质量（自动） | 12/12，0.936 综合分 | 维持；靠 KB/RAG 提升事实性 |
| 实验充分性 | 缺 v0.6 新跑分、缺 3σ | 补 multi_run 即可 |
| 学术对标 | SNS/AgentDropout 达标 | AgentPrune/SafeSieve 部分对标 |

**一句话**：通信与缓存架构已接近纯 API 方案的天花板；下一步重点是 **v0.6 实测确认 DB 域收益**、**消融与方差写进报告**、**P0 的 prompt/证据裁剪** 再抠 5–10% token。
