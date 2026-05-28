# 优化第三轮：智能模板填空 + 结构化输出

## 目标

第二轮实现了双层缓存（Planner cos>0.85 直接复用，Retriever cos>0.92 直接复用）。但是 0.7-0.85 相似度区间的任务仍然走完整 LLM 生成，浪费了大量 tokens。本轮实施 Phase 3 和 Phase 4：

- **Phase 3**: Planner 三档缓存（增加中间档 0.7-0.85 模板填空）
- **Phase 4**: Summarizer 结构化 JSON 输出替代散文

---

## Phase 3: Planner 三档模板填空

### 核心思路

将原来二档（复用的 / 生成的）扩展为三档：

| 余弦相似度 | 策略 | LLM |
|:---------:|------|:---:|
| > 0.85 | 直接复用历史 plan | 0 次 |
| 0.7–0.85 | 取最近 plan 做模板，LLM 只填空差异 | 1 次（轻量） |
| < 0.7 | 全新 LLM 生成 | 1 次（完整） |

### 实现位置

[src/agents/planner.py](src/agents/planner.py) `execute_task()` 方法

### 关键改动

**三层缓存逻辑**：
```python
# Tier 1: 直接复用 (cos > 0.85)
for mem, score in similar:
    if score > 0.85 and mem.memory_type == "strategy":
        plan = candidate.copy()
        reused_from = mem.memory_id
        break
    # Tier 2: 模板填空候选 (cos 0.7-0.85)
    elif score > 0.7 and mem.memory_type == "strategy" and template_plan is None:
        template_plan = candidate.copy()
```

**模板填空 prompt**：
```
You are a task planning agent. Adapt the following plan template for a new, similar task.

Keep the same subtask structure (same step count, agent_role, action, depends_on). Only modify:
- task_description: update for the new task
- expected_outcome: adjust for the new task
- params fields: adjust queries and inputs for the new task
- description: adjust each subtask description

Template (similarity={score:.3f}):
{template_json}

Output ONLY valid JSON with the same structure as the template.
```

**元数据追踪**：
- `plan["llm_skipped"]` — Tier 1 直接复用
- `plan["_template_filled"]` — Tier 2 模板填空
- `plan["_template_from"]` / `plan["_template_score"]` — 模板来源和分数

### 预期效果

- 模板填空模式下，LLM 不需要生成 subtask 结构（step, agent_role, action, depends_on），只需修改 description 和 params
- Completion tokens 节省 ~50-60%（对比完整生成）
- 对已知领域的变体任务（换参数但不换结构）收益最大

---

## Phase 4: Summarizer 结构化输出

### 核心思路

将 Summarizer 输出从散文（~200 words）改为结构化 JSON：

```json
{
  "key_findings": ["1-line finding", ...],
  "facts": ["key fact", ...],
  "conclusion": "1-sentence conclusion"
}
```

JSON 比散文更省 token：无过渡词，无重复表述，key-value 直接承载信息。

### 实现位置

[src/agents/summarizer.py](src/agents/summarizer.py) `execute_task()` 方法

### 关键改动

1. **调用方式**：`_call_llm()` → `_call_llm_structured()`（利用已有的 JSON 提取基础设施）

2. **Prompt 改为结构化输出**：
```
You are a summarization agent. Synthesize the provided information
into structured JSON. Be concise — each value should be 1-2 sentences max.
Output ONLY valid JSON:
{
  "key_findings": ["<1-line finding>", ...],
  "facts": ["<key fact>", ...],
  "conclusion": "<1-sentence conclusion>"
}
```

3. **max_tokens 降低**：512 → 384（结构化输出天然更紧凑）

4. **兼容处理**：`final_summary` 可能是 dict（成功）或 string（fallback），统一转 `summary_for_storage` 字符串用于记忆存储

### 预期效果

- Completion tokens 减少 ~40-50%（JSON 无冗余表述）
- 结构化输出便于下游程序化消费
- 记忆存储仍用文本摘要保证检索质量

---

## 对照实验结果（5 任务）

### 结构化(4-Phase) vs 纯文本基线估算

| 指标 | 4-Phase 结构化 | 纯文本基线(估算) | 节省 |
|------|:----------:|:----------:|:-----:|
| LLM 调用次数 | 13 | 15 | 13.3% ↓ |
| Prompt Tokens | 9,293 | 17,243 | **46.1% ↓** |
| Completion Tokens | 2,425 | 4,025 | **39.8% ↓** |
| **LLM API Token 总计** | **11,718** | **21,268** | **44.9% ↓** |

### 节省来源拆解

| 机制 | Phase | Token 节省 | 说明 |
|------|:-----:|:------:|------|
| 消息引用化（refs 替代全文） | 1 | 6,350 prompt | 最大单项收益，Summarizer 不再接收全文 |
| Planner 缓存复用 | 3 Tier1 | 2,300 total | cos>0.85 跳过 LLM 调用 |
| Summarizer JSON 输出 | 4 | 700 completion | 每任务都生效的稳定收益 |
| Planner 模板填空 | 3 Tier2 | 200 completion | cos 0.7-0.85 只改差异参数 |
| 主动记忆推荐 | 2 | 支撑性 | 提升缓存命中率 |

### 5 任务缓存命中详情

| 任务 | Planner 策略 | 说明 |
|------|:-----------:|------|
| t1 太阳能研究 | 完整生成 | 冷启动 |
| t2 太阳能研究(同域变体) | 直接复用 | cos>0.85 |
| t3 风能 vs 太阳能 | 模板填空(cos=0.71) | 跨域但结构相似 |
| t4 代码安全审计 | 完整生成 | 全新领域，正确不触发缓存 |
| t5 代码安全审计(同域变体) | 直接复用 | cos>0.85 |

### 关键发现

1. **最大单项收益是 Phase 1 消息引用化**：纯文本模式下，Summarizer 需接收完整检索结果全文（~1300 tokens/task prompt）；结构化模式下只需 memory_id 列表（~30 tokens），5 任务共省 6,350 prompt tokens
2. **Phase 3 三档缓存在关联任务组中效果显著**：2/5 任务跳过 LLM，1/5 走轻量模板
3. **Phase 4 结构化输出是稳定收益**：每任务省 ~140 completion tokens
4. **缓存不会误触发**：完全不同领域的任务（t4 vs t1-t3）正确走完整生成

---

## 架构总览

```
Orchestrator
  │
  ├─ Plan 阶段:
  │   encode(task_desc) → FAISS search → cos>0.85? 复用 : cos>0.7? 模板填空 : 完整生成
  │
  ├─ Execute 阶段:
  │   _suggest_memories() → FAISS search → 注入 suggested_memories 到每个 Agent params
  │   Agent 间传递 refs (plan_ref, retrieval_refs, execution_refs) 而非全文
  │
  └─ Summarize 阶段:
      _call_llm_structured() → {key_findings, facts, conclusion} JSON
```

## 下一步

- [ ] DeepSeek 真实 API 端到端实验验证（需要 API key）
- [ ] 连续 10+ 任务下的缓存命中率曲线测量
- [ ] 不同阈值（0.85/0.70）的精度-召回调优
