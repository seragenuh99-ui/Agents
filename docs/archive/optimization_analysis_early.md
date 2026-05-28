# 702solver 系统优化分析

## 当前状态总览

| 模块 | 当前方案 | 瓶颈/问题 |
|------|----------|-----------|
| 通信协议 | MessagePack (schema-less) | 无零拷贝，无增量传输 |
| LLM 调用 | DeepSeek API，每次全新请求 | 无缓存，重复传输 Agent system prompt |
| 状态传递 | all-MiniLM-L6-v2, 384维, float32 | 模型较小，未量化 |
| 共享记忆 | SQLite + FAISS IndexFlatIP | FlatIP 暴力搜索，大规模退化 |
| Agent 编排 | 固定 4 Agent 串行流水线 | 无并行，无动态路由 |
| 沙箱 | subprocess隔离 + AST验证 | 冷启动慢，无会话复用 |

---

## 一、通信协议层优化

### 1.1 增量消息传输（高优先级）

当前问题：Agent 间每轮通信发送完整消息，包含重复的 context/memory_refs 等字段。

```
现状: Msg1{plan + context + 5 refs} → Msg2{plan + context + 5 refs + 新增1 ref}  // 大量重复
优化: Msg1{full} → Msg2{delta: +1 ref, _base: msg1.id}  // 仅传差异
```

预期效果：多轮任务消息体积再降 40-60%。

### 1.2 字段编码再压缩（中优先级）

当前字段名已缩写（`h.fr`, `a.pm` 等），可进一步用 varint 数字 ID 替代字符串 key：

| 方案 | 字段 key 开销（6字段） | 说明 |
|------|:---:|------|
| 当前 (msgpack str) | ~12 bytes | 短字符串 |
| 优化 (varint id) | ~6 bytes | 数字 ID + varint 编码 |
| Protobuf 方案 | ~3 bytes | schema 预定义，tag+type 1 byte |

预期效果：每条消息再省 5-10 bytes，累计约 5% 额外压缩。

### 1.3 零拷贝反序列化 (FlatBuffers/Cap'n Proto)（低优先级）

适合高吞吐场景（1000+ msg/s），但不适合当前规模。引入 schema 定义增加维护成本。**暂不推荐**。

---

## 二、LLM 调用优化

### 2.1 Prompt Caching（高优先级）★

DeepSeek 天然支持 KVCache，且粒度仅 64 tokens（Anthropic: 1024, OpenAI: 1024），是最容易落地的优化。

当前问题：

```
每次 Agent 调用都发送完整 system_prompt + user_prompt
PlannerAgent system_prompt: ~800 chars (~200 tokens)  ← 每次重复传输
```

优化方案：

```python
# 在 LLMBackend.chat() 中将 system prompt 缓存
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},  # ← 缓存前缀
    {"role": "user", "content": user_prompt},       # ← 动态部分放最后
]
# DeepSeek 自动缓存 ≥64 tokens 的前缀
```

预期效果：
- 缓存命中时输入 token **成本降低 75%**（DeepSeek: $0.14/M vs $0.55/M 命中 token）
- TTFT（首 token 延迟）从 2-3s 降至 **0.5s 以内**
- 尤其适合 Agent 场景：4 个 Agent 的 system_prompt 在多轮任务中完全不变

### 2.2 请求合并 / Batching（中优先级）

当前 orchestrator 对每个 Agent 顺序调用 LLM。对于独立的 subtask（如并行检索多个数据源），可合并请求：

```
现状: Retriever→LLM(查询1) → Retriever→LLM(查询2) → Retriever→LLM(查询3)  // 串行
优化: Retriever→LLM([查询1, 查询2, 查询3])  // 单次请求批处理
```

DeepSeek 支持在一次请求中传入多个 message 序列，可减少 API 调用次数。

### 2.3 双模型路由（中优先级）

不是所有 Agent 调用都需要 DeepSeek-V3 级别的模型：

```
检索 Agent: deepseek-chat (小模型) → 关键词匹配足够
规划 Agent: deepseek-chat (大模型) → 需要推理
执行 Agent: deepseek-chat (中模型) → CodeAct 生成
总结 Agent: deepseek-chat (大模型) → 需要综合
```

可接入 DeepSeek 的轻量模型处理简单任务，降低 50%+ 成本。

---

## 三、非文本状态传递优化

### 3.1 Embedding 模型升级（高优先级）

当前 `all-MiniLM-L6-v2` (384维) 是 2021 年的轻量英文模型：

| 模型 | 维度 | MTEB 得分 | 模型大小 |
|------|:---:|:---------:|:-------:|
| all-MiniLM-L6-v2 (当前) | 384 | 56.3 | 80MB |
| **bge-large-zh-v1.5** | **1024** | **64.5** (C-MTEB) | 1.3GB |
| **multilingual-e5-large** | 1024 | 61.5 | 2.2GB |
| **gte-Qwen2-1.5B** | 1536 | 65.1 | 3GB |

推荐 `bge-large-zh-v1.5`（中文场景）或 `multilingual-e5-large`（中英混合），语义检索精度提升 15-20%。

### 3.2 Embedding 量化（中优先级）

当前每个 384 维 float32 向量 = 1536 bytes。可量化到 8-bit：

```
当前: 384 × 4 bytes = 1536 bytes  (float32)
量化: 384 × 1 byte  = 384 bytes   (int8, 75% 压缩)
```

结合 FAISS IVF_SQ8 索引直接支持量化向量存储和检索，无需额外转换。

### 3.3 增量 State 传递（中优先级）

类似于协议的增量传输，状态也只需传变化部分：

```
首轮: StatePacket(full_embedding)
续轮: StatePacket(delta_embedding, base_id=prev)  // 只传残差
```

接收方通过 embedding fusion 组合历史状态和增量状态。

---

## 四、共享记忆系统优化

### 4.1 FAISS 索引升级（高优先级）★

当前使用 `IndexFlatIP`（暴力内积），查询复杂度 O(N)：

| 索引类型 | 1K 向量 | 10K 向量 | 100K 向量 | 内存 |
|----------|:-------:|:--------:|:---------:|:----:|
| **IndexFlatIP (当前)** | 0.3 ms | 3 ms | 30 ms | 大 |
| **IndexHNSW** | **0.05 ms** | **0.08 ms** | **0.15 ms** | 大 |
| **IndexIVFFlat** | 0.2 ms | 0.3 ms | 0.5 ms | 中 |
| **IndexIVFPQ** | 0.5 ms | 0.8 ms | 1.2 ms | **小** |

推荐分阶段升级：
1. 短期（向量 < 5000）：保持 FlatIP
2. 中期（向量 5000-50000）：切换 HNSW，查询 10-30× 加速
3. 长期（向量 > 50000）：IVFPQ，内存压缩 75%+

### 4.2 记忆去重与压缩（中优先级）

当前每条记忆独立存储，可能出现高度重复：

```python
# 记忆去重：写入前检查语义相似度
existing = store.search_by_similarity(new_embedding, limit=1)
if existing[0][1] > 0.95:  # 几乎重复
    store.record_access(existing[0][0].memory_id)  # 更新访问计数
    return existing[0][0].memory_id  # 复用已有
```

### 4.3 记忆分层（中优先级）

按访问频率将记忆分级存储：

```
Hot 层 (最近 50 条 + 高频访问) → FAISS GPU / HNSW     ← 毫秒级
Warm 层 (最近 500 条)           → FAISS IVF            ← 十毫秒级
Cold 层 (归档)                  → SQLite + 按需加载     ← 离线
```

### 4.4 混合检索增强（低优先级）

当前已有关键词 + 标签 + 语义三重检索。可增加 BM25 稀疏检索作为第四路，提升精确术语匹配精度。

---

## 五、Agent 架构优化

### 5.1 并行子任务执行（高优先级）★

当前 orchestrator 对所有 subtask 串行处理：

```python
# 现状（串行）
for subtask in plan["subtasks"]:
    if subtask["agent_role"] == "retriever":
        result = self.retriever.execute_task(msg)  # 阻塞
```

优化为并行执行无依赖的 subtask：

```python
# 优化（并行）
independent_groups = group_by_dependency(plan["subtasks"])
for group in independent_groups:
    results = run_parallel([agent.execute_task(t) for t in group])
```

预期效果：多检索任务场景时延降低 40-60%。

### 5.2 动态 Agent 路由（中优先级）

当前固定 4 Agent。可增加 Agent 能力注册和动态发现：

```
当前: Orchestrator 硬编码调用 planner → retriever → executor → summarizer
优化: Orchestrator 查询 Registry → 发现可用 Agent → 动态组装 Pipeline
```

使系统可扩展更多 Agent（如 Validator、Translator、Visualizer）。

### 5.3 层级规划（中优先级）

复杂任务单层分解可能不够精细：

```
当前: 任务 → Planner → [subtask1, subtask2, subtask3]  // 单层
优化: 任务 → Planner → [subtask1, [subtask2.1, subtask2.2], subtask3]  // 递归
```

---

## 六、沙箱优化

### 6.1 会话级沙箱复用（高优先级）

当前每次 CodeAct 执行启动新 subprocess（~30ms 冷启动）：

```python
# 现状：每次 subprocess.run()
proc = subprocess.run([sys.executable, "-c", code], ...)  # 30ms 冷启动
```

改为持久化 subprocess 会话：

```python
# 优化：复用进程
session = SandboxSession()       # 一次性启动
session.execute("x = [1,2,3]")   # 1ms
session.execute("y = sum(x)")    # 1ms (变量可跨调用保留)
```

预期效果：CodeAct 执行延迟降低 90%+。

### 6.2 WASM 沙箱（低优先级）

如 smolagents 的 WasmExecutor 方案，使用 Pyodide 在 WASM 中执行代码。优点是更安全的隔离（capability-based）和更快的冷启动。但 Python 生态兼容性有损失（部分库无法运行）。

---

## 七、评测与监控增强

### 7.1 Token 核算改进（高优先级）

当前 token 统计混合了协议开销和 LLM 输出，导致对比数据不够精准。应拆分为：

```
协议 Token = 结构化消息字段的 MessagePack → Token 估算
LLM Token  = API 返回的实际 usage.prompt_tokens + usage.completion_tokens
总 Token   = 协议 Token + LLM Token
```

这样结构化 vs 纯文本的对比才能排除 LLM 输出波动的影响。

### 7.2 成本追踪

增加基于 DeepSeek 定价的实时成本估算：

```
成本 = prompt_tokens × $0.55/M + completion_tokens × $2.19/M
```

让系统在终端实时显示累计花费。

---

## 八、优化优先级排序

| 优先级 | 优化项 | 预期收益 | 实现难度 | 建议顺序 |
|:------:|--------|----------|:--------:|:--------:|
| ★★★ | Prompt Caching | 成本↓75%, 延迟↓80% | 低 | **1** |
| ★★★ | 并行子任务执行 | 时延↓40-60% | 中 | **2** |
| ★★★ | FAISS HNSW 索引 | 搜索加速 10-30× | 低 | **3** |
| ★★★ | Token 核算改进 | 对比数据更精准 | 低 | **4** |
| ★★ | Embedding 模型升级 | 检索精度↑15-20% | 中 | 5 |
| ★★ | 增量消息传输 | 消息体积再↓40% | 中 | 6 |
| ★★ | 会话级沙箱复用 | CodeAct 延迟↓90% | 中 | 7 |
| ★★ | 记忆去重压缩 | 记忆库质量提升 | 低 | 8 |
| ★ | Embedding 量化 | 存储↓75% | 低 | 9 |
| ★ | 双模型路由 | 成本↓30-50% | 高 | 10 |
| ★ | 动态 Agent 路由 | 可扩展性提升 | 高 | 11 |
| ★ | 层级规划 | 复杂任务质量提升 | 高 | 12 |

---

## 九、赛题评分维度对应

针对评分细则的优化映射：

| 评分维度 | 当前得分潜力 | 核心优化方向 |
|----------|:----------:|-------------|
| 通信效率 (25分) | 20-22 | 增量传输(★2) + Token核算(★4) |
| 状态传递创新 (20分) | 17-19 | Embedding升级(★5) + 量化(★9) + 增量State |
| 记忆复用效果 (20分) | 18-20 | FAISS HNSW(★3) + 去重(★8) |
| 系统完整性 (20分) | 18-20 | 并行执行(★2) + 会话沙箱(★7) |
| 实验验证 (15分) | 13-15 | Token核算(★4) + 成本追踪 |

建议按优先级依次实施 **1→2→3→4→5**，每个约 1-2 小时，5 项下来整体提升显著。
