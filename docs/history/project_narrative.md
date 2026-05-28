# 702solver 项目报告（开发历程长文）

> **归档说明**：本文为历史叙事全文（指标可能停在 v0.6.1）。**当前数据**见 [`../PROJECT_RECORD.md`](../PROJECT_RECORD.md)、[`../VERSION_HISTORY.md`](../VERSION_HISTORY.md)。  
> **文档索引**：[`../README.md`](../README.md)。

## 项目是什么

一个多 Agent 协作系统。4 个 Agent（Planner / Retriever / Executor / Summarizer）分工完成复杂任务，由 Orchestrator 统一调度。

核心创新点：
- Agent 之间用 **384 维语义向量**代替文本传递状态，接收方直接用向量做 FAISS 检索
- 用 **SQLite + FAISS** 构建跨任务共享记忆，同领域任务自动复用历史结果
- 用 **SNS 速记符号**（编程风格）替代自然语言 prompt，大幅压缩 token

```
任务输入 → Orchestrator
              ├─ Planner（拆任务，生成 subtask 计划）
              ├─ Retriever（查信息）  ←─┐ 同层可并行
              ├─ Executor（跑代码）   ←─┘
              └─ Summarizer（写总结，输出 JSON）
```

---

## 系统架构

### 模块组成

| 模块 | 做什么 | 关键技术 |
|------|--------|---------|
| Planner Agent | 把复杂任务拆成 subtask 计划 | 4 层缓存策略（E2E → P1 模板 → P0 LLM judge → 全新生成） |
| Retriever Agent | 检索相关信息 | LLM 模拟搜索 + FAISS 语义匹配 |
| Executor Agent | 执行代码、处理数据 | subprocess 沙箱 + AST 验证，19 种危险模式拦截 |
| Summarizer Agent | 综合所有结果生成最终报告 | 结构化 JSON 输出 + 语义缓存 |
| Orchestrator | 调度、依赖排序、并行执行 | 拓扑分层 + ThreadPoolExecutor |
| 通信层 | Agent 间消息传递 | MessagePack 二进制序列化 + 消息引用化（传 ref ID 不传全文） |
| 状态层 | 非文本状态传递 | 384 维 BGE 向量 + StateExchangeBus |
| 记忆层 | 跨任务知识复用 | SQLite 元数据 + FAISS FlatIP 精确索引 |
| 沙箱层 | 代码安全执行 | subprocess 隔离 + AST pattern 拦截 |

### 缓存体系（4 层）

```
新任务到来
  ├─ Layer 1: E2E 全缓存 → cos>0.85 + 同标签 → 结果完整复用，0 次 API 调用
  ├─ Layer 2: P1 标签模板 → 同领域积累 ≥2 条自动提炼模板 → LLM 只填空差异
  ├─ Layer 3: P0 LLM 判断 → cos 0.50-0.85 → LLM 确认是否同类任务 → YES 则模板填空
  └─ Layer 4: 全新生成 → 前面都不通 → 从零调 LLM
```

---

## 项目历程

### Round 1-3：搭建基础系统（MockLLM 模拟阶段）

**做了什么**：搭建 4 Agent + Orchestrator 框架、MessagePack 通信、SQLite+FAISS 记忆、沙箱、状态传递。

**状态**：287 个单元测试全过。一切用 MockLLM 模拟（不调真实 API，用规则模拟 LLM 输出）。

**问题**：无。MockLLM 永远输出完美格式，所有边界情况被掩盖。

---

### Round 4：SNS 速记 + E2E 缓存（MockLLM，5 任务实验）

**做了什么**：
- 借鉴 SNS-Core 论文，用编程符号风格写 prompt（如 `task→subtasks{RETR|EXEC|SUMM}`）
- E2E 缓存：cos>0.90 直接复用全部结果，连 LLM 都不用调
- Summarizer 用 evidence_refs Jaccard 相似度复用历史摘要

**MockLLM 数据**（5 任务）：

| 指标 | 数值 |
|------|:----:|
| LLM 调用 | 7 次 |
| Prompt Tokens | 2,724 |
| Completion Tokens | 1,510 |
| 总 Token | 4,234 |
| vs 纯文本估算 | **省 72.1%** |
| 缓存命中 | 2/5（E2E 命中） |

**暴露的问题**（MockLLM 阶段已发现但未修复）：

| # | 问题 | 严重度 |
|---|------|:----:|
| 1 | MiniLM 嵌入模型太弱，同域任务 cos 只有 0.59 | 核心 |
| 2 | 模板区间窄，0.70-0.85 很少出现 | 中 |
| 3 | Summarizer 缓存用 Jaccard-on-evidence_refs，新任务永不相同 | 中 |
| 4 | 缓存判断无二次验证，纯靠 cos 值 | 中 |
| 5 | 记忆扁平无层级，无压缩合并 | 中 |

---

### Round 5-6：P0 LLM 判断 + P1 层次化模板（MockLLM，5 任务实验）

**P0：LLM 二次判断**
- FAISS 粗筛 cos>0.50 的候选 → LLM 判断是否同类任务 → YES 则复用
- 缓存命中率从 40%（2/5）提升到 60%（3/5）
- 每次 judge 只花 ~50 tokens

**P1：层次化记忆模板**
- 同域积累 ≥3 条 strategy 后，自动调用 LLM 提炼领域模板
- 新任务命中模板后走"模板填空"路径，成本显著低于全量生成
- 跨域任务正确拒绝（security 不受 solar 模板干扰）

**MockLLM 数据**：

| 指标 | Round 4 | Round 5 (P0) | Round 6 (P1) |
|------|:------:|:----------:|:----------:|
| 缓存命中率 | 40% | 60% | 60%+模板 |
| LLM Judge 准确率 | — | 100% (2/2) | — |
| 模板数 | 0 | 0 | 1（solar 域） |

---

### 转折点：切换到真实 DeepSeek API

切换到真实 LLM 后，MockLLM 阶段掩盖的问题全面爆发：

**崩溃级问题（系统完全不可用）**：

| 问题 | 表现 | 根因 |
|------|------|------|
| Summarizer 从未被调用 | 12 个任务 relevance=0 | LLM 输出字段名 `role`，orchestrator 检查 `agent_role` |
| 跨域缓存污染 | Wind 任务拿到 Solar 的 summary | LLM 失败时 raw text 被当缓存结果 |

**严重问题（数据全错）**：

| 问题 | 表现 | 根因 |
|------|------|------|
| HNSW score 膨胀 46% | cos=0.58 被当成 0.85 触发 E2E 复用 | FAISS IndexHNSWFlat 近似索引返回假分数 |
| 标签共用串台 | Wind 匹配到 Solar 模板 | "energy" 标签在 solar/wind 间共用 |
| P1 模板缺 summarizer | Wind 任务输出空 summary | 模板的 subtask_pattern 只有 retriever |

**效率问题（缓存反而更费 token）**：

| 问题 | 表现 | 根因 |
|------|------|------|
| 模板 JSON 膨胀 | 模板命中后 prompt 反而更大 | 整套模板 JSON（500+ tokens）塞进 prompt |
| 记忆上下文冗余 | prompt 被历史记忆撑大 | 注入 5 条记忆 × 500 字符/条 |
| LLM judge 过多 | 数据库任务调了 6 次跨域 judge | 对所有 cos>0.50 候选都调 LLM |

**切换后的初始数据**（真实 DeepSeek API）：

| 指标 | 数值 |
|------|:----:|
| 总 Token | 61,708 |
| API 调用 | 43 |
| 平均 Relevance | ~0.52 |
| 低 Relevance (<0.3) | 2/12 |
| Summarizer 报错 | 4/12 |
| 跨域污染 | 有 |

---

### 修复阶段：9 项修复（真实 DeepSeek API）

| # | 问题 | 修复 | 影响 |
|:--:|------|------|:--:|
| 1 | role/agent_role 不一致 | Orchestrator 加字段 normalizer | 致命修复 |
| 2 | 非结构化 summary 缓存 | 只接受 `isinstance(dict)` 的缓存，拒收 raw text | 致命修复 |
| 3 | HNSW score 膨胀 | `HNSW_THRESHOLD = 100_000`，用 FlatIP 精确内积 | 严重修复 |
| 4 | P1 模板缺 summarizer | 自动追加 summarizer subtask | 严重修复 |
| 5 | E2E 跨域误命中 | Judge 前检查标签必须有重叠 | 严重修复 |
| 6 | P3 summarizer 跨域 | 同上，加标签过滤 | 严重修复 |
| 7 | P0 planner judge 过多 | Judge 前先过滤标签 | 优化 |
| 8 | P1 模板跨域匹配 | Primary tag（第一个标签）必须匹配 | 严重修复 |
| 9 | Summarizer LLM 超时 | Retry + 更短 prompt + 更大 max_tokens | 严重修复 |

**修复后数据**：

| 指标 | 修复前 | 修复后 |
|------|:-----:|:-----:|
| API 总调用 | 43 | 27 |
| 总 Token | 61,708 | 48,292 |
| 平均 Relevance | ~0.52 | 0.664 |
| 低 Relevance | 2/12 | 0/12 |
| 跨域污染 | 有 | **0** |
| Summarizer 报错 | 4/12 | 0/12 |

系统终于能正常工作了。但还有一个问题：**缓存机制仍然是亏本的**。

---

### 优化阶段：Prompt 压缩（真实 DeepSeek API）

**根因分析**：为什么修复后缓存还是亏本？

算一笔账：
- 模板命中 → 把 500+ token 的 JSON 塞进 prompt → 省了 ~100 token completion → **净亏 400+ token**
- 注入 5 条历史记忆 → prompt 膨胀 ~2500 字符 → 几乎没用 → **净亏**
- `_suggest_memories` 每个 subtask 调一次 → 结果注入 agent prompt → 进一步膨胀

核心矛盾：**prompt 膨胀的 token 远超缓存节省的 token**。

**优化措施**（借鉴 SNS-Core / AgentDropout 论文思路）：

1. **模板压缩**：不传完整 JSON，只传结构骨架
   ```
   旧: {"plan_id":"template_abc","subtasks":[{"step":1,"desc":"Research solar...","role":"retriever",...}],...}  (500+ tokens)
   新: retriever/retrieve[deps:-]|executor/execute[deps:1]|summarizer/summarize[deps:2]  (30 tokens)
   ```

2. **记忆精简**：只注最相关的 2 条，只含 120 字符 summary，不传 content
3. **Summarizer 压缩**：去 markdown 标题，用最短标记（`Plan:` `Retrieved:` `Analysis:` `Past:`）
4. **AgentDropout**：模板填空命中且 cos≥0.70 时，跳过 Executor，Summarizer 直接从 Retriever 结果生成。省 1 次 LLM 调用/任务。

### DeepSeek Prompt Caching 实验（未采用）

测试了 DeepSeek 原生的 prompt caching 机制：
- 需要 128+ token 的 system_prompt 才能触发缓存（`cached_tokens > 0`）
- 实测：143 token system_prompt → 第二个请求 128 tokens 被缓存
- **但未采用**：长 prompt 会"引导" LLM 输出更长回复，completion 从 10,303 涨到 14,296，得不偿失
- **结论**：SNS 短 prompt 策略与 DeepSeek prompt caching 矛盾，选 SNS

### 优化后数据（12 任务，真实 DeepSeek API）

> **当前基线**：**v0.6.1**（2026-05-19）。详见 `docs/optimization_roadmap_analysis.md`、`docs/full_evaluation_summary.md`、`docs/VERSION_HISTORY.md`。

| 指标 | 纯文本 | 修复后 | Prompt 压缩 (R4*) | **v0.6.1（当前）** |
|------|:-----:|:-----:|:-------------------:|:------------------:|
| 总 Token | 49,076 | 48,292 | 17,093 | **16,523** |
| vs 纯文本 | 基线 | 省 1.6% | 省 65.2% | **省 66.3%** |
| Prompt tokens | 12,753 | — | 6,790 | **6,373** |
| Completion tokens | 36,323 | — | 10,303 | **10,107** |
| API 调用 | 36 | 27 | 29 | **25** |
| 平均 Relevance | ~0.52 | 0.664 | 0.710 | **0.838** |
| 低 Relevance | 2/12 | 0/12 | 0/12 | **0/12** |
| Summarizer 硬错误 | — | 0 | 0 | **0** |
| 跨域污染 | — | 0 | 0 | 0 |

\* R4* = Prompt 压缩 + AgentDropout 阶段（`project_overview` 内「优化阶段」），非 `optimization_round4_research.md` 的调研轮次编号。

---

## 关键技术总结

### 1. 通信效率：Token 从 49,076 降到 16,523（省 66.3%）

四组对照实验，控制变量分离各因素贡献：

| 模式 | Token | vs 纯文本 | 说明 |
|------|:-----:|:------:|------|
| 纯文本 | 49,076 | 基线 | 长 prompt + 散文输出 + 全文传递 |
| Structured 无缓存 | 37,208 | 省 24.2% | SNS + JSON + refs，无跨任务缓存 |
| Structured 旧版 | 46,326 | 省 5.6% | 加缓存反而更差（模板 JSON 膨胀） |
| Structured 优化版 (R4*) | 17,093 | 省 65.2% | 压缩模板 + 精简记忆 |
| **v0.6.1（当前）** | **16,523** | **省 66.3%** | v0.6 DB KB + v0.6.1 剪枝 |

各优化项贡献：

| 优化项 | 节省 | 机制 |
|--------|:----:|------|
| JSON 结构化输出 | 省 65% completion | 散文→JSON，36,323→10,303 |
| 消息引用化 | 省 ~46% prompt | KB 级全文→bytes 级 ref ID |
| 模板 JSON 压缩 | 省 ~80% prompt | 500+ token→30 token 骨架 |
| 记忆上下文精简 | 省 ~90% prompt | 5条×500字→2条×120字 |
| SNS 速记 prompt | 省 ~60% prompt | 自然语言→编程符号 |

### 2. 缓存命中率：75%（9/12）

| 策略 | 命中 | 说明 |
|------|:---:|------|
| E2E EXACT | 1 | cos>0.85+同标签，0 次 LLM 调用（s3 复用 s1） |
| P1 TEMPLATE_DROP | 1 | 同主标签模板 + AgentDropout（e4） |
| TEMPLATE FILL | 7 | embedding 相似触发模板填空 |
| FULL GEN | 3 | 冷启动（每域第一个任务） |

### 3. 稳定性：Relevance 0.838，0 硬错误

| 指标 | 修复前 | 修复后 | R4* 最佳 | **v0.5.0** |
|------|:-----:|:-----:|:--------:|:-----------:|
| 平均 Relevance | ~0.52 | 0.664 | 0.710 | **0.838** |
| 低 Relevance (<0.3) | 2/12 | 0 | 0 | 0 |
| 跨域缓存污染 | 有 | 0 | 0 | 0 |
| Summarizer 硬错误 | 4/12 | 0 | 0 | **0** |
| 单元测试 | 287 | 287 | 287 | 287 |

---

## 优化版 12 任务详情（v0.5.0，2026-05-19）

| Task | Domain | Strategy | Calls | Tokens | Rel |
|------|--------|----------|:----:|:------:|:---:|
| e1_solar_basics | solar | FULL_GEN | 2 | 1,104 | 0.775 |
| e2_solar_advanced | solar | TEMPLATE_FILL | 4 | 2,025 | 0.878 |
| e3_wind_energy | wind | TEMPLATE_FILL | 2 | 1,276 | 0.872 |
| e4_renewable_comparison | solar | P1_TEMPLATE_DROP | 2 | 1,485 | 0.841 |
| s1_python_vuln | security | FULL_GEN | 2 | 1,392 | 0.869 |
| s2_web_security | security | TEMPLATE_FILL | 3 | 1,657 | 0.834 |
| s3_python_vuln_v2 | security | E2E_EXACT | 0 | 0 | 0.731 |
| s4_code_review | security | TEMPLATE_FILL | 2 | 1,429 | 0.746 |
| d1_query_optimization | database | FULL_GEN | 2 | 1,411 | 0.901 |
| d2_nosql_comparison | database | TEMPLATE_FILL | 2 | 1,430 | 0.863 |
| d3_db_performance | database | TEMPLATE_FILL | 2 | 1,686 | 0.833 |
| d4_data_modeling | database | TEMPLATE_FILL | 2 | 1,585 | 0.909 |

---

## 当前状态与局限

### 已解决的问题
- Token 节省 **66.3%**（16,523 vs 49,076），为真 API 12 任务当前最低记录
- 缓存命中率 75%（9/12），且缓存命中**真正产生净收益**
- AgentDropout 生效（e4：`P1_TEMPLATE_DROP`）
- 跨域污染 **0 例**
- Relevance 均值 **0.838**，三域均衡（security 0.73–0.87，database 0.83–0.91）
- Summarizer **0 硬错误**（Round 5 修复调度竞态 + JSON mode）
- 287 单元测试全过

### v0.5.1 质量验证（2026-05-19）

针对「无问答留档、无事实核查」补充：

| 产出 | 说明 |
|------|------|
| `docs/quality_records.md` | 每题：**问题** + **最终回答** + **验证表** |
| `quality_report.json` | 机器可读全量记录 |
| `src/evaluation/quality_validator.py` | 启发式 + 可选 LLM 评判 |

**最近一次质量验证**（12 任务，启发式 + LLM 评判）：

| 指标 | 启发式 only | + LLM 评判 |
|------|:-----------:|:----------:|
| 综合通过 | 12/12 | **12/12** |
| 平均综合分 | 0.920 | **0.936** |
| 平均嵌入相关性 | 0.849 | 0.849 |
| 平均主题词覆盖 | 90.8% | 90.8% |
| LLM 切题率 | — | **12/12**（评分 4–5/5） |

```bash
python3 experiments/quality_benchmark.py          # 含 LLM 评判
python3 experiments/quality_benchmark.py --no-judge  # 仅启发式（已生成 records）
python3 experiments/rejudge_quality.py            # 对已有 JSON 补评判
python3 experiments/multi_run_benchmark.py -n 3     # 3 次均值±方差
```

详见 `docs/quality_validation_guide.md`。

### 仍存在的局限
1. **LLM 输出方差**：可用 `multi_run_benchmark.py -n 3` 报均值±方差
2. **Retriever 仍为内置 KB 模拟**，非真实搜索引擎或 RAG
3. **缺少横向对比**：无 AutoGen/CrewAI 同条件对照
4. **自动验证 ≠ 事实正确**：主题词 + LLM 评判可发现错域，不能代替人工 fact-check
5. **P2 / P4 / P5** 尚未实施

### 和论文的对比（v0.5.0）

| 论文 | 报告效果 | 我们 | 对比 |
|------|:------:|:---:|------|
| SNS-Core | 60-85% token↓ | **66.4%** | 达标 |
| LatentMAS | 70-83% token↓ | 66.4% | API 不可行，差约 4–17pp |
| AgentPrune | 28-73% token↓ | 66.4% | 达标 |
| AgentDropout | 21.6% prompt↓ | **已实施** | Executor dropout |

---

## 修复时间线总览

```
MockLLM 阶段                  真实 LLM 阶段
────────────────────────────────────────────────────────────────────────────
R1-R3   R4       R5-R6        切换      9项修复    Prompt压缩   v0.5.0 Round5
搭建    SNS+E2E  P0+P1        崩溃      rel 0.664  17k tok      16.5k tok
287test 72.1%*  缓存60%*      →0        48k tok    rel 0.710    rel 0.838
        (*Mock)  (*Mock)                 省1.6%     省65.2%      省66.4%
                                                      ↓            ↓
                                              AgentDropout   调度+JSON+缓存卫生
```

版本明细见 `docs/VERSION_HISTORY.md`；Round 5 详见 `docs/history/rounds/round5_v0.5.0_stability.md`。

**核心教训**：
1. MockLLM 掩盖一切——`role`vs`agent_role`、HNSW score 膨胀、模板 JSON 膨胀在 MockLLM 下永远发现不了
2. 缓存机制不一定是好的——只有当 prompt 开销 < LLM 调用节省时，缓存才产生净收益
3. SNS 短 prompt 是最大的单项优化，比 DeepSeek 原生 prompt caching 更有效（短 prompt 不等同于 cache miss，紧凑输出省更多）
