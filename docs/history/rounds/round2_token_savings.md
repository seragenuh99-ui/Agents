# 优化第二轮：实际 Token 节省

## 目标

上一轮实验表明，端到端的 structured vs text token 对比被 LLM 输出波动主导——协议层压缩（MessagePack 比 JSON 小 34%）的收益是固定的 bytes 级别，在 LLM 输出的 KB 级别面前几乎不可见。

本轮聚焦**真正能省 token 的方向**：减少不必要的 LLM 调用。

---

## 实施内容

### 1. Planner 计划缓存（最大收益）

**核心思路**：新任务到达时，先搜索历史记忆中的相似 plan。如果相似度 > 0.85，直接复用历史 plan，跳过 LLM 调用。

**实现位置**：[src/agents/planner.py](src/agents/planner.py) `execute_task()` 方法

**流程**：
```
新任务 → 编码 task_description → FAISS 语义搜索 "strategy" 类型记忆
         ↓ cos > 0.85
    复用历史 plan（设置 llm_skipped=True）
         ↓ cos ≤ 0.85
    正常调用 LLM 生成 plan
```

**关键代码**：
```python
# Cache check: reuse plan if near-identical task was done before
if self.memory_store._index is not None and self.memory_store._index.ntotal > 0:
    task_emb = self.embedding_engine.encode(task_desc)
    similar = self.memory_store.search_by_similarity(task_emb, limit=3)
    for mem, score in similar:
        if score > 0.85 and mem.memory_type == "strategy":
            candidate = json.loads(mem.content)
            if isinstance(candidate, dict) and "subtasks" in candidate:
                plan = candidate.copy()
                reused_from = mem.memory_id
                break
```

### 2. 精简 Prompt（中等收益）

**Planner prompt**：添加 "Keep all descriptions brief (1 sentence max)" 约束，减少 LLM 输出长度。

**Summarizer prompt**：从 "Include: 1. Key findings 2. Main conclusions 3. Recommendations" 改为 "under 200 words. Use bullet points. Do NOT repeat the input."

**Summarizer max_tokens**：从 1024 降至 512。

**实现位置**：
- [src/agents/planner.py:115-138](src/agents/planner.py#L115-L138) — 精简的系统 prompt
- [src/agents/summarizer.py:118-119](src/agents/summarizer.py#L118-L119) — 精简的汇总 prompt
- [src/agents/summarizer.py:125](src/agents/summarizer.py#L125) — max_tokens 512

### 3. Retriever 结果缓存（中等收益）

**核心思路**：检索查询到达时，先用 embedding 搜索是否有相同或极相似的查询已完成。如果 cos > 0.92，直接复用缓存结果。

**实现位置**：[src/agents/retriever.py](src/agents/retriever.py) `execute_task()` 方法

```python
# Cache check: reuse cached retrieval results for near-identical queries
if self.memory_store._index is not None and self.memory_store._index.ntotal > 0:
    query_emb = self.embedding_engine.encode(query)
    similar = self.memory_store.search_by_similarity(query_emb, limit=3)
    for mem, score in similar:
        if score > 0.92 and mem.memory_type == "evidence":
            cached = json.loads(mem.content)
            if isinstance(cached, dict) and "memory_hits" in cached:
                results = cached.copy()
                results["_cached_from"] = mem.memory_id
                break
```

### 4. Embedding 对齐修复

**问题**：`BaseAgent.store_memory()` 用 `topic+summary+tags` 计算 embedding，但搜索时用原始 `task_description`。两者文本差异导致余弦相似度从 0.945 降到 0.798，缓存无法命中。

**修复**：给 `store_memory()` 添加 `embedding_text` 参数，planner 传 `task_desc`，retriever 传 `query`。

**实现位置**：[src/agents/base.py:396](src/agents/base.py#L396)

---

## 实验验证

### 缓存机制验证（DeepSeek 真实 API）

**测试设计**：
1. Task 1: "Research solar energy costs: installation, LCOE, maintenance, and 2024 price trends"
2. Task 2: "Research solar energy costs: installation expenses, LCOE metrics, maintenance costs, and pricing trends in 2024" (极相似)
3. Task 3: "Analyze Python security vulnerabilities: SQL injection, XSS, CSRF detection patterns" (完全不同的主题)

**结果**：

| 任务 | Planner 缓存 | Retriever 缓存 | LLM 调用数 |
|------|:----------:|:------------:|:---------:|
| Task 1 (首次) | ✗ (首次) | ✗ (首次) | 2 (planner + summarizer) |
| Task 2 (极相似) | ✅ 复用 4c050db2 | ✅ 4/4 步骤命中 | 1 (仅 summarizer) |
| Task 3 (不同) | ✗ (正确) | ✗ (正确) | 5 (完整 pipeline) |

**Task 2 节省**：
- 跳过 1 次 Planner LLM 调用（~1200 prompt + ~500 completion tokens）
- 跳过 4 次检索步骤的全部查询操作
- 总 LLM 调用：8 次（3 个任务，含缓存优化）

**余弦相似度**：
- Task 1 vs Task 2 (task_desc): **0.963** — 远超 0.85 阈值
- Task 1 vs Task 3 (task_desc): 远低于阈值 — 正确区分

---

## 预期收益分析

### 单次计划复用节省

| 节省项 | Token 量 |
|--------|:------:|
| Planner system prompt | ~800 tokens |
| Planner context (历史记忆) | ~400 tokens |
| Planner 输出的 plan JSON | ~500 tokens |
| **合计** | **~1700 tokens** |

### 规模化收益估算

假设系统处理 10 个相关领域任务，其中 30-50% 可复用：

| 场景 | 无缓存 (LLM 调用) | 有缓存 (LLM 调用) | Token 节省 |
|------|:----------------:|:----------------:|:----------:|
| 10 tasks, 30% 相似 | 10 planner calls | 7 planner calls | ~5,100 tokens |
| 10 tasks, 50% 相似 | 10 planner calls | 5 planner calls | ~8,500 tokens |
| 连续模式 (高关联) | 10 planner calls | 2-3 planner calls | ~12,000+ tokens |

Plus retriever cache 额外节省了检索步骤的执行成本（非 LLM，但减少了 CPU 开销和响应延迟）。

---

## 与第一轮优化的协同效果

| 优化层 | 作用域 | 节省类型 |
|--------|--------|----------|
| MessagePack 协议 | 每一条消息 | ~34% bytes (固定) |
| 非文本状态传递 | Agent 间状态 | ~70% bytes (固定) |
| Prompt Caching | LLM prompt | 重复前缀免费 (API 层) |
| **Planner 缓存** | 相似任务 plan | 省 1 次 LLM 调用 (~1700 tokens) |
| **Retriever 缓存** | 相似查询结果 | 省检索执行 + 减少重复 |
| 并行执行 | 同级子任务 | 时延减少 ~30% |
| HNSW 索引 | 记忆搜索 | O(log N) vs O(N) |

---

## 下一步

- [ ] 连续模式 (10+ rounds) 下测量缓存命中率
- [ ] 动态调整缓存阈值（基于 false positive rate）
- [ ] Embedding 模型升级 (bge-large-zh-v1.5) 提升中文语义精度
