# 多Agent共享记忆驱动的工业软件故障诊断与操作知识图谱构建研究

> 开题报告 | 2026-05-28

---

## 一、研究背景与问题提出

### 1.1 工业软件运维的现实困境

工业软件（MES、SCADA、PLC 控制系统、工业物联网平台等）运行在复杂的异构环境中，故障诊断面临三个核心瓶颈：

1. **知识碎片化**：故障排查经验分散在运维人员的头脑、工单系统、日志文件和操作手册中，缺乏统一的结构化表示
2. **经验难复用**：相似的故障场景反复出现，但每次诊断都从零开始，历史案例的检索依赖人工记忆和关键词匹配
3. **知识衰减快**：系统持续迭代升级，旧的故障模式文档很快过时，静态知识库的维护成本极高

### 1.2 现有方法的不足

| 方法 | 代表技术 | 关键局限 |
|------|----------|----------|
| 基于规则 | 专家系统（如 CLIPS） | 规则覆盖率低，维护成本高，无法处理未预见故障 |
| 基于统计 | 异常检测（如 LSTM、Transformer） | 给出异常信号但不给解释，运维人员不信任黑箱 |
| 静态知识图谱 | Neo4j 工业 KG、OWL 本体 | 依赖专家手工构建，更新滞后，无法自动从新数据中学习 |
| LLM 单次诊断 | GPT-4 直接分析日志 | 无记忆能力，每次独立诊断，相似案例间无法复用推理 |

### 1.3 核心问题

**如何在工业软件运行过程中，自动从故障处理经验中构建可演化的操作知识图谱，并使新故障能有效复用历史诊断知识？**

这涉及三个子问题：

- **SP1**：如何将非结构化的故障处理过程自动转化为结构化的知识单元？
- **SP2**：如何在保证安全性的前提下，实现跨故障案例的知识复用？
- **SP3**：如何让知识图谱随新故障的积累自我演化，而非依赖人工维护？

---

## 二、研究现状与切入点

### 2.1 工业故障知识图谱（Fault KG）

传统方法依赖领域本体（如 FMEA、FTA 形式化），由专家定义故障类、现象类、原因类和操作类的实体与关系。代表性工作包括：
- 基于设备层次结构（系统→子系统→部件→零件）构建故障传播图
- 利用文本抽取（NER + Relation Extraction）从维修工单中半自动构建

**不足**：(1) 本体预定义的是封闭世界，无法覆盖未知故障模式；(2) 关系抽取依赖大量标注数据；(3) 知识图谱是静态的，没有"学习"机制。

### 2.2 LLM + Agent 协作系统

近两年，基于大语言模型的多 Agent 框架（AutoGen、CrewAI、MetaGPT 等）展示了在复杂任务中分工协作的潜力。但大多数工作集中在通用的对话和代码生成任务，**专门面向工业故障诊断场景的 Agent 协作与知识沉淀机制尚无成熟方案**。

### 2.3 记忆增强的 LLM 系统

MemGPT、Mem0 等项目探索了为 LLM 添加外部记忆的能力，但其记忆模型是通用的键值存储或对话摘要，缺乏面向诊断任务的专门设计（如证据链追溯、跨案例因果关联、故障类型感知的复用控制）。

### 2.4 本研究的切入点

将多 Agent 协作框架中的**共享记忆模块**（结构化存储 + 语义检索 + 意图门控 + 证据链）与工业故障诊断场景结合，设计一套**从故障案例中自动构建并演化操作知识图谱**的方法。核心理念是"每处理一次故障，知识图谱就增长一分"——即诊断过程本身就是知识图谱的构建过程。

---

## 三、研究目标与内容

### 3.1 总体目标

设计并实现一套**多 Agent 共享记忆驱动的工业软件故障诊断与操作知识图谱自动构建系统**，实现以下能力：

1. **自构建**：从故障描述和诊断过程中自动提取实体、关系与操作步骤，存入结构化记忆
2. **可复用**：对新故障通过语义检索和意图门控匹配历史案例，复用诊断策略而非从零开始
3. **可演化**：记忆随案例积累自动抽象为模板和原则，新案例修正错误记忆
4. **安全门控**：防止跨故障类型的错误知识迁移

### 3.2 研究内容

**研究内容一：面向故障诊断的记忆单元扩展**

现有 `MemoryUnit`（`src/memory/models.py`）已具备通用知识存储的核心字段：`memory_id`、`summary`、`content`（JSON 正文）、`tags`、`evidence_chain`、`embedding`（384 维）、`memory_type`（result/strategy/evidence/fact/error）、`abstraction_level`（0 实例/1 模板/2 原则）、`confidence`、`fill_attempts`/`fill_successes`。本研究在此基础上扩展故障诊断专属字段：

```
FaultMemoryUnit(MemoryUnit)    # 继承现有 MemoryUnit
├── 通用字段（直接复用）
│   ├── memory_id, source_agent, created_at
│   ├── summary, content, tags
│   ├── evidence_chain         # 诊断因果链：从症状到根因的记忆引用
│   ├── embedding              # BGE 384 维语义向量
│   ├── memory_type            # 复用现有枚举，增加 "diagnosis" 子类型
│   ├── abstraction_level      # 0=具体故障案例, 1=诊断策略模板, 2=诊断原则
│   ├── confidence             # 诊断置信度
│   └── fill_attempts/fill_successes  # SafeSieve 质量追踪
│
└── 故障诊断扩展字段
    ├── fault_category   → 故障大类（database/network/config/code/resource）
    ├── symptoms         → 结构化症状列表 [{indicator, value, unit}, ...]
    ├── root_cause       → 根因 memory_id（指向 evidence_chain 末端的因果记忆）
    ├── affected_components → 受影响组件列表
    ├── severity         → P0/P1/P2/P3
    └── resolution       → 修复方案摘要（JSON）
```

关键点：**这不是重新设计**。现有 `MemoryUnit` 的 `content` 字段已经是 JSON 格式，`tags` 已经承载领域分类，`evidence_chain` 已经记录推导链路。故障扩展字段本质上是对 `content` JSON schema 的领域特化、对 `tags` 体系增加故障分类维度、以及新增若干索引列以支持按故障类型和严重等级过滤。底层 `MemoryStore`（SQLite + FAISS）无需改动。

**研究内容二：基于语义检索与意图门控的故障案例复用**

现有系统已实现完整的三级复用管道（`src/orchestrator.py`），本研究将其适配到故障诊断场景：

| 复用层级 | 现有实现 | 诊断场景适配 |
|----------|----------|-------------|
| **E2E 整案复用** | `_try_e2e_reuse()` — 语义检索 + 意图门控 + `e2e_threshold=0.85` | 症状向量相似度 ≥ 0.85 + 同故障大类 → 直接复用诊断步骤和修复方案 |
| **领域计划注入** | `_load_domain_plan_hint()` — 检索 `strategy` 类型记忆，注入 Planner | 检索同故障类的诊断 DAG 模板，Planner 只改差异参数（模板填空，`max_tokens=256`） |
| **Agent 级记忆提示** | `_suggest_memories()` — 各 Agent 执行前检索相关记忆 | Executor 获得相似故障的修复步骤提示，Retriever 获得相关组件的已知问题提示 |

意图门控直接复用现有 `task_intent.py` 的机制：
- `detect_intent()` 扩展故障分类关键词（database_performance、network_timeout、config_error、memory_leak、cpu_spike 等），映射到 `fault_category`
- `cache_intents_compatible()` 确保不同故障大类之间不发生 E2E 复用
- `requires_fresh_synthesis()` 对高危故障类型（数据丢失、安全漏洞）强制从零诊断
- `blocks_executor_dropout()` 对需要验证的故障类型保留 Executor 执行通道

语义检索复用现有 `MemoryStore.search_by_similarity()`（FAISS `IndexFlatIP`，内积等价余弦相似度），综合效用排序公式已在 `search_by_similarity` 中实现：

```
utility = 0.75 × cosine_sim + 0.15 × min(1.0, access_count/10) + 0.10 × template_success_rate
```

这使得"高频命中且诊断成功"的案例排序靠前，而非只靠向量相似度。

**研究内容三：evidence_chain 驱动的因果推理**

现有 `evidence_chain`（`MemoryUnit` 中的 `List[str]`，存储支撑记忆的 `memory_id`）已记录 Agent 间的推导链路：Planner 计划 → Retriever 检索 → Executor 执行 → Summarizer 综合，下游记忆的 `evidence_chain` 指向上游记忆。本研究将其从被动记录提升为主动推理工具：

```
故障案例 A: "订单服务响应超时"
│  evidence_chain: [m1, m2, m3]
├── m1 (evidence): "数据库连接池使用率 98%"
├── m2 (evidence): "慢查询堆积 > 30s"
└── m3 (evidence): "orders 表缺少 status+created_at 联合索引"  ← 根因

故障案例 B: "支付服务响应超时"
│  evidence_chain: [m4, m5, m6]
├── m4 (evidence): "数据库连接池使用率 95%"
├── m5 (evidence): "慢查询堆积 > 20s"
└── m6 (evidence): "payments 表缺少 user_id+created_at 联合索引"  ← 同类根因
```

两个案例的症状不同（订单 vs 支付），但因果链结构高度相似。现有 `evidence_chain` 数据已就绪，增量工作在于：

1. **因果链结构匹配**：沿 `evidence_chain` 递归读取记忆，比较链上节点的 `memory_type` 序列和嵌入相似度，识别因果结构相似的案例对
2. **根因聚类**：当多个案例的 `evidence_chain` 末端指向嵌入相似的记忆时（如上述 m3 和 m6），自动聚类为同类根因，新故障出现时优先检索该聚类
3. **排除式诊断**：若某候选案例的 `evidence_chain` 中某一环节与当前故障的已确认事实矛盾（如 "连接池使用率正常" vs 候选案例假设 "连接池满"），通过 `confidence` 字段降权该候选

**研究内容四：知识图谱的自动抽象与演化**

现有三级 `abstraction_level` 体系已经定义了知识抽象的层级结构。增量工作是利用该体系驱动自动归纳：

```
Level 0: 具体故障案例 (abstraction_level=0)
  memory_type="result"
  "2024-03-15，MES订单模块超时，根因：缺少联合索引"

Level 1: 诊断策略模板 (abstraction_level=1) — 同类案例 ≥3 时自动生成
  memory_type="strategy"
  "数据库连接池类故障诊断模板":
    Step 1: 检查连接池使用率和等待队列
    Step 2: 分析慢查询日志，定位高耗时 SQL
    Step 3: 检查索引覆盖和执行计划
    Step 4: 验证修复后性能指标

Level 2: 诊断原则 (abstraction_level=2) — LLM 跨模板归纳
  memory_type="strategy"
  "对于数据密集型服务的响应延迟问题，
   优先排查数据库层（连接池→慢查询→索引），
   再排查应用层（代码逻辑→GC→线程池）"
  置信度随支持案例数增长
```

现有 SafeSieve-lite（`MemoryStore.effective_similarity()` + `record_template_outcome()`）直接用于模板质量追踪：模板填充分数连续失败（`fill_attempts ≥ 2` 且 `fill_successes/fill_attempts < 0.25`）的模板，相似度分数自动 ×0.45，自然被淘汰。这个机制在故障场景中尤为重要——过时的诊断模板如果被复用，可能导致运维人员按错误流程排查。

### 3.3 研究边界

- **范围限定**：工业软件的**运行类故障**（性能、可用性、数据一致性），不包括硬件故障和物理安全问题
- **输入假设**：故障报告为自然语言描述 + 结构化日志片段，不要求完整的传感器时序数据
- **人机协作**：诊断方案为推荐而非自动执行，最终操作由运维人员确认

---

## 四、技术方案

> 本节所有引用的模块、类、方法均已在 702solver v0.11.2 中实现并验证。标注为"增量工作"的部分为本课题需要新增或修改的内容。

### 4.1 现有技术基础全景

以下架构已在 702solver 中完整运行，是本研究直接继承的工程基础：

```
                        main.py / chat.py / experiments/
                              │
┌─────────────────────────────▼──────────────────────────────────┐
│                   Orchestrator (src/orchestrator.py)            │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              多 Agent 运行时                                │ │
│  │  PlannerAgent  │  RetrieverAgent  │  ExecutorAgent  │  SummarizerAgent │
│  │  (src/agents/   │  (src/agents/    │  (src/agents/   │  (src/agents/    │
│  │   planner.py)   │   retriever.py)  │   executor.py)  │   summarizer.py) │
│  └───────────────────┬─────────────────────────────────────────┘ │
│                      │                                            │
│  ┌───────────────────▼─────────────────────────────────────────┐ │
│  │         协议调度层 (src/protocol/scheduler.py)                │ │
│  │  Scheduler  │  MessageBus  │  AgentRegistry  │  ProtocolParser│ │
│  │  握手+路由   │  消息队列+统计 │  能力注册+发现    │  MessagePack   │ │
│  └───────────────────┬─────────────────────────────────────────┘ │
│                      │                                            │
│  ┌───────────────────▼─────────────────────────────────────────┐ │
│  │         状态层 (src/state/)                                   │ │
│  │  EmbeddingEngine (embeddings.py)  │  StateExchangeBus         │ │
│  │  BAAI/bge-small-en-v1.5 → 384维  │  (exchange.py)            │ │
│  └───────────────────┬─────────────────────────────────────────┘ │
│                      │                                            │
│  ┌───────────────────▼─────────────────────────────────────────┐ │
│  │         共享记忆层 (src/memory/)                              │ │
│  │  ┌─────────────────────┐  ┌──────────────────────────────┐  │ │
│  │  │ MemoryUnit (models)  │  │ MemoryStore (store.py)       │  │ │
│  │  │ 17 字段              │  │ SQLite 元数据 + FAISS 向量   │  │ │
│  │  │ evidence_chain       │  │ IndexFlatIP → HNSW (100K)   │  │ │
│  │  │ abstraction_level    │  │ 关键词/标签/语义 三路检索     │  │ │
│  │  │ fill_attempts/       │  │ effective_similarity()       │  │ │
│  │  │ fill_successes       │  │ record_template_outcome()    │  │ │
│  │  └─────────────────────┘  └──────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │       安全层 (src/task_intent.py)                              │ │
│  │  detect_intent()  │  cache_intents_compatible()               │ │
│  │  requires_fresh_synthesis()  │  blocks_executor_dropout()     │ │
│  │  7 类意图 → 严格意图 4 类：compare/policy/compliance/vuln     │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │       评测层 (src/evaluation/)                                 │ │
│  │  MetricsCollector  │  quality_validator  │  Reporter          │ │
│  └──────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

### 4.2 关键技术一：MemoryUnit + MemoryStore 双引擎存储

**现有实现**（`src/memory/models.py` + `src/memory/store.py`）：

`MemoryUnit` 是一条记忆的完整数据模型。核心字段：

| 字段 | 说明 | 在故障诊断中的作用 |
|------|------|-------------------|
| `memory_id` | UUID 全局唯一 | 故障案例的唯一标识 |
| `summary` | ≤200 字符摘要 | 故障一句话描述，用于列表展示 |
| `content` | JSON 正文 | 诊断步骤、修复方案、监控数据等结构化内容 |
| `tags` | 标签列表 | 故障分类标签：`["database", "performance", "timeout"]` |
| `evidence_chain` | 指向支撑记忆的 ID 列表 | 记录"症状→中间原因→根因"的推导链路 |
| `embedding` | 384 维 float32 向量 | 用于语义相似度检索 |
| `memory_type` | result/strategy/evidence/fact/error | 区分诊断结果、策略模板、证据、事实、错误 |
| `abstraction_level` | 0=实例 / 1=模板 / 2=原则 | 知识抽象层级 |
| `fill_attempts` / `fill_successes` | SafeSieve 计数 | 模板被诊断复用的成功/失败次数 |

`MemoryStore` 采用双引擎架构：

- **SQLite**：存储元数据（17 个字段），建索引于 `task_topic`、`tags`、`memory_type`、`abstraction_level`
- **FAISS**：存储 384 维向量，`IndexFlatIP`（内积 = 余弦相似度），向量数 ≥100K 时自动切换 `IndexHNSWFlat`（O(log N) 图搜索）

写入时执行语义去重（`store()` 的 `dedup` 参数，`cos > 0.95` 且同 `memory_type` 则复用已有记录）。现有 `access_log` 表记录每次访问的 agent、时间和 task_id。

**增量工作**：在 `MemoryUnit` 子类中增加 6 个故障诊断字段（见 3.2 节 RC1），底层 `MemoryStore` 不变，仅增加 `fault_category` 和 `severity` 的 SQLite 索引列。

### 4.3 关键技术二：三路检索 + 综合效用排序

**现有实现**（`src/agents/base.py:368-409` — `BaseAgent.query_memory()`）：

```
query_memory(query, tags, use_embedding=True, limit=5)
         │
         ├─→ search_by_keyword(query)        → SQL LIKE 匹配 topic/summary/tags/content
         ├─→ search_by_tags(tags)             → 标签 OR 匹配，去重
         └─→ search_by_similarity(embedding)  → FAISS Top-K，过滤 cos < 0.3
                      │
                合并去重，记录访问日志，返回 Top-K
```

其中 `search_by_similarity()`（`src/memory/store.py:351-391`）的排序不是纯相似度，而是综合效用分数：

```python
# store.py:384-388
utility = 0.75 × cosine_sim + 0.15 × min(1.0, access_count/10) + 0.10 × template_success_rate
```

这个公式让"被频繁使用且产出好结果"的记忆排名靠前。

**增量工作**：在检索时增加 `fault_category` 过滤参数（已有 `memory_type` 过滤的代码模式可直接复用），确保跨故障大类的记忆不被混入检索结果。

### 4.4 关键技术三：三级缓存复用管道

**现有实现**（`src/orchestrator.py` + `src/run_options.py`）：

这是 token 节省（68.3%）的核心机制，已经在 `Orchestrator.execute_task()` 中完整运行：

**第一级：E2E 整案复用** — `_try_e2e_reuse()`（`orchestrator.py:260-326`）

```
新任务到达
  │
  ├─ 编码任务描述 → embedding
  ├─ search_by_similarity(embedding, limit=12, memory_type="result")
  ├─ 遍历候选记忆：
  │   ├─ _tags_overlap(task_tags, cached_tags)  → 标签必须交集
  │   ├─ enable_intent_cache_gate?
  │   │   ├─ requires_fresh_synthesis()?  → COMPARE/COMPLIANCE/REVIEW 禁 E2E
  │   │   └─ cache_intents_compatible()?  → 严格意图只同类型
  │   ├─ _effective_score(mem, raw)       → SafeSieve 降权
  │   └─ score ≥ e2e_threshold (0.85)?
  │       └─ _finish_e2e() → 直接返回缓存，0 LLM 调用
  └─ 未命中 → 进入第二级
```

**第二级：领域计划注入** — `_load_domain_plan_hint()`（`orchestrator.py:328-359`）

```
search_by_similarity(embedding, limit=8, memory_type="strategy")
  ├─ 标签匹配 + 意图兼容检查
  ├─ _effective_score ≥ domain_plan_threshold (0.68)?
  └─ 注入 Planner 系统提示："参考以下诊断模板，只修改差异字段"
```

Planner 拿到模板后，LLM 被指示只修改 `query` 和 `context` 字段（`planner_template_max_tokens=256`），不重写整个计划 JSON。Planner 生成的新计划写入 MemoryStore，`abstraction_level=1`（领域模板），供后续同类故障复用。

**第三级：Agent 级记忆提示** — `_suggest_memories()`（`orchestrator.py:151-176`）

每个 Agent 执行前，`_suggest_memories(text, limit=2, min_score=0.35)` 检索相关记忆，以 compact 格式注入 Agent 参数：

```python
# orchestrator.py:170-175
{"memory_id": mem.memory_id,
 "summary": mem.summary[:200],
 "type": mem.memory_type,
 "relevance": round(score, 3)}
```

**Executor Dropout**（`orchestrator.py:650-662`）：当任务不需要代码执行时跳过 Executor Agent——条件为 `blocks_executor_dropout()` 返回 False 且模板匹配分数 ≥ `executor_dropout_threshold (0.65)`。

**关键阈值一览**（`src/run_options.py:27-41`）：

| 阈值 | 默认值 | 控制什么 |
|------|--------|----------|
| `e2e_threshold` | 0.85 | E2E 整案复用门槛 |
| `summarizer_e2e_threshold` | 0.85 | Summarizer 摘要复用 |
| `summarizer_adapt_threshold` | 0.72 | Summarizer 轻量适配（~280 completion tok） |
| `domain_plan_threshold` | 0.68 | Planner 领域模板注入 |
| `executor_dropout_threshold` | 0.65 | 跳过 Executor |
| `retriever_cache_threshold` | 0.88 | Retriever 检索结果复用 |

**增量工作**：以上所有机制和阈值直接在故障诊断场景中使用。主要适配工作为：
- 扩展 `task_intent.py` 的 `_STRICT_INTENTS` 增加故障大类边界
- 新增 `_VULN_MARKERS` 类似的故障关键词标记（如 `_DB_PERF_MARKERS`、`_NETWORK_MARKERS` 等）
- 对高危故障类型（数据损坏、安全漏洞）在 `requires_fresh_synthesis()` 和 `blocks_executor_dropout()` 中加白名单

### 4.5 关键技术四：意图门控

**现有实现**（`src/task_intent.py`）：

意图门控是一套基于关键词匹配的分类与路由机制，防止不同性质的任务错误共享缓存。核心函数：

- `detect_intent(description, tags)` → 7 类意图（compare/policy/compliance/vulnerability/review/analyze/general）
- `cache_intents_compatible(src_desc, tgt_desc, src_tags, tgt_tags)` → 严格意图（COMPARE/POLICY/COMPLIANCE/VULNERABILITY）只能同类型复用
- `requires_fresh_synthesis(description, tags)` → COMPARE/COMPLIANCE/REVIEW 禁止 E2E 和 Summarizer 缓存
- `blocks_executor_dropout(description, tags)` → REVIEW/COMPLIANCE 保留 Executor 通道

三个门控生效位置（全部在 `orchestrator.py` 中）：

| 位置 | 函数 | 行号 | 作用 |
|------|------|------|------|
| E2E 缓存检查 | `_try_e2e_reuse()` | L285-294 | 严格意图禁 E2E + 意图不兼容禁复用 |
| 领域计划注入 | `_load_domain_plan_hint()` | L343-350 | 计划模板也需意图兼容 |
| Summarizer 缓存 | `execute_task()` | L560-561 | 摘要复用前检查意图兼容 |

**增量工作**：扩展 `detect_intent()` 的故障分类能力。现有架构已支持——新增故障类型关键词标记组（类似现有 `_VULN_MARKERS`），在 `TaskIntent` 枚举中增加 `DATABASE_FAULT`、`NETWORK_FAULT` 等类型，门控逻辑无需改动。故障场景特有的需求是**操作安全验证**：复用修复方案前检查是否包含高风险操作（如 `DROP TABLE`、`kill -9`、`iptables -F`），该检查在 `_finish_e2e()` 中增加一个安全过滤步骤即可。

### 4.6 关键技术五：SafeSieve-lite 模板质量反馈

**现有实现**（`src/memory/store.py:261-285`）：

这是知识图谱自校正的核心机制。当 Planner 的策略模板被填充并执行后，Summarizer 的输出经过质量验证，结果反馈到模板记忆的计数器中：

```python
# store.py:274-285
def record_template_outcome(self, memory_id: str, success: bool):
    """记录模板填充的成败"""
    # UPDATE memories SET
    #   fill_attempts = fill_attempts + 1,
    #   fill_successes = fill_successes + (1 if success else 0)

def effective_similarity(self, mem, raw_score: float):
    """根据历史成功率调整相似度分数"""
    rate = fill_successes / fill_attempts
    if fill_attempts >= 2 and rate < 0.25:
        return raw_score * 0.45      # 严重降权
    return raw_score * (0.65 + 0.35 * rate)  # 平滑调整
```

这个机制的特点是**自动运行、无需人工标注**。在故障诊断场景中其价值更大：
- 过时的诊断模板（如"先重启服务"在新版本中不再有效）会自动降权
- 被多次验证有效的模板（如"数据库超时先查连接池"）权重自然升高
- 降权而非删除，当新案例再次验证其有效时可以恢复

**增量工作**：此机制无需修改，直接使用。需要在故障场景的 quality_validator 中增加诊断正确性的自动检查项（如修复方案是否匹配故障类型）。

### 4.7 关键技术六：evidence_chain 证据链

**现有实现**（`src/memory/models.py:27` + Agent 执行流程中的自动记录）：

`evidence_chain` 是 `MemoryUnit` 中的一个 `List[str]` 字段，存储支撑此记忆的其他记忆 ID。在 Agent 执行过程中自动记录——Planner 的计划 ID、Retriever 的检索结果 ID、Executor 的执行输出 ID 被聚合到 Summarizer 的综合报告的 `evidence_chain` 中，形成推导链路。

`MemoryStore` 提供完整的基础操作：
- `get(memory_id)` — 按 ID 读取单条记忆（`store.py:247-258`）
- `search_by_similarity()` — 语义检索（`store.py:351-391`）
- `get_concrete_memories()` — 按标签 + 类型 + abstraction_level=0 查具体案例（`store.py:432-451`）
- `get_templates()` — 按标签 + 类型 + abstraction_level≥1 查领域模板（`store.py:411-430`）

**增量工作**：利用上述已有 API 实现三个因果推理功能：

1. **因果链遍历**：沿 `evidence_chain` 递归 `get(memory_id)`，组装完整推导链路 → 用于向运维人员展示"为什么是这个诊断结论"
2. **因果链相似度**：比较两个案例的 evidence_chain 上各节点的嵌入向量和类型序列，识别"症状不同但根因相同"的案例对 → 用于根因聚类
3. **矛盾检测**：新故障的中间发现与候选案例 evidence_chain 中某节点矛盾 → 降低该候选的 `confidence` 分数

这些功能都在现有 `MemoryStore` API 之上实现，不需要新的存储引擎或索引结构。

### 4.8 关键技术七：评估框架

**现有实现**（`src/evaluation/metrics.py` + `src/evaluation/quality_validator.py`）：

`MetricsCollector` / `TaskMetrics` 已采集的指标（`ARCHITECTURE.md:253-259`）：

| 指标 | 字段 | 诊断场景意义 |
|------|------|-------------|
| Token 消耗 | `llm_prompt_tokens`, `llm_completion_tokens` | 衡量诊断的 LLM 成本 |
| 记忆命中 | `memory_hits`, `memory_hit_rate` | 衡量知识复用程度 |
| 消息数 | `messages_sent/received` | Agent 间通信开销 |
| 向量传输 | `state_transfers`, `state_data_bytes` | 非文本状态交换量 |
| 墙钟时间 | `total_elapsed_ms` | 诊断响应速度 |

已有对比实验框架（`experiments/full24_controlled_comparison.py`）支持 A/B/C 三方对照（纯文本基线 / 结构化无缓存 / 结构化全功能），可直接用于诊断场景的消融实验。

`RunOptions`（`src/run_options.py`）提供 11 个可配置开关和 10 个可调阈值，支持逐功能消融以量化每个机制的贡献。

**增量工作**：构建故障诊断基准案例集（≥50 个案例，覆盖 5 类故障），增加诊断准确率（Top-1/Top-3）的评估逻辑。

---

## 五、创新点

| # | 创新点 | 说明 |
|---|--------|------|
| **1** | **诊断即建图** | 放弃"先建 KG 再用 KG"的两阶段范式，Agent 执行诊断的同时自动将经验结构化存入 MemoryStore，知识图谱是诊断过程的"副产品" |
| **2** | **因果链驱动的案例匹配** | 不只匹配表面症状语义，而是沿 evidence_chain 比较因果结构，提供可解释的诊断依据 |
| **3** | **三级知识抽象的自演化** | 利用现有 abstraction_level 体系，从具体案例(level=0)→诊断模板(level=1)→诊断原则(level=2)自动归纳 |
| **4** | **SafeSieve 驱动的质量闭环** | 现有 fill_attempts/fill_successes + effective_similarity 机制让系统自动淘汰低质量诊断模板，无需人工审核 |

---

## 六、研究计划

| 阶段 | 时间 | 工作内容 | 产出 |
|------|------|----------|------|
| **Phase 1** | 第 1–2 月 | 扩展 MemoryUnit 故障字段；构建故障案例集（≥50 案例，5 类故障）；扩展 task_intent.py 故障关键词 | FaultMemoryUnit 模型、基准案例集 |
| **Phase 2** | 第 3–5 月 | Agent 链路适配故障诊断流程；因果链遍历与匹配算法；故障意图门控扩展 | 可运行的诊断原型 |
| **Phase 3** | 第 6–7 月 | 案例聚类与模板自动归纳；LLM 跨模板原则提取；SafeSieve 接入诊断质量反馈 | 知识演化引擎 |
| **Phase 4** | 第 8–9 月 | 完整消融实验（A/B/C 三方对照 + 各功能开关）；与基线方法对比 | 实验数据与对比报告 |
| **Phase 5** | 第 10–12 月 | 撰写论文、完善系统、准备答辩 | 学位论文 |

---

## 七、预期成果

1. **系统原型**：多 Agent 共享记忆驱动的工业软件故障诊断原型，完整链路：故障报告输入 → Agent 协作诊断 → 知识自动沉淀 → 案例智能复用 → 知识演化
2. **评估数据**：诊断准确率、token 节省比例、知识模板质量等指标，以及与规则系统、静态 KG、纯 LLM 诊断的对比
3. **学位论文**：系统性阐述方法的理论基础、技术方案和实验分析

---

## 八、可行性分析

### 8.1 技术基础（已就绪）

以下模块已在 702solver v0.11.2 中完整实现并通过 24 题基准验证（质量 24/24，token 节省 68.3%，对抗 0/6 误用）：

| 模块 | 文件 | 状态 |
|------|------|------|
| MemoryUnit 17 字段数据模型 | `src/memory/models.py` | 已实现 |
| MemoryStore SQLite + FAISS 双引擎 | `src/memory/store.py` | 已实现 |
| 三路检索（关键词/标签/语义）+ 综合效用排序 | `src/memory/store.py` + `src/agents/base.py:query_memory()` | 已实现 |
| evidence_chain 记录 + 递归读取 | `src/memory/models.py` + `store.py:get()` | 已实现 |
| 三级缓存复用管道（E2E + 领域计划注入 + Agent 提示） | `src/orchestrator.py:_try_e2e_reuse()` / `_load_domain_plan_hint()` / `_suggest_memories()` | 已实现 |
| 7 类意图识别 + 三级门控 | `src/task_intent.py` | 已实现 |
| SafeSieve-lite 模板质量反馈 | `src/memory/store.py:effective_similarity()` + `record_template_outcome()` | 已实现 |
| RunOptions 11 开关 + 10 阈值可配置 | `src/run_options.py` | 已实现 |
| A/B/C 三方对照实验框架 + MetricsCollector | `experiments/` + `src/evaluation/` | 已实现 |
| Executor Dropout + 模板填空 + Summarizer 轻量适配 | `src/orchestrator.py` + `src/agents/` | 已实现 |

### 8.2 增量工作估算

本课题的核心增量工作是**领域适配**而非系统重构：

| 工作项 | 类型 | 估时 |
|--------|------|------|
| MemoryUnit 扩展 6 个故障字段 | 子类继承 | 1 周 |
| task_intent.py 扩展故障关键词 | 增加标记组 | 1 周 |
| 故障案例集构建 | 数据准备 | 3 周 |
| 因果链遍历与匹配算法 | 新增函数 | 3 周 |
| 案例聚类 + 模板自动归纳 | 新增模块 | 4 周 |
| 故障 quality_validator 扩展 | 修改现有 | 1 周 |
| 消融实验 + 对比基线 | 实验 | 3 周 |

总计约 16 周核心开发工作量，分布在 9 个月中，节奏合理。

### 8.3 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 故障案例集获取困难 | 优先合成案例 + 公开 IT 运维数据集；真实工单作为迁移验证 |
| LLM 诊断不准确 | 系统为人机协作模式（推荐而非自动执行），核心指标是"正确诊断在推荐列表中的排名" |
| 模板演化失控 | SafeSieve 自动降权 + 意图门控 + 运维人员确认，三重约束 |
| FAISS 大规模检索性能 | 已实现 IndexFlatIP → HNSW 自动切换（100K 阈值），可扩展至百万级 |

---

## 九、参考文献（初步）

1. Pan, Y. H., et al. (2024). "A Survey on Knowledge Graphs for Industrial Fault Diagnosis." *IEEE Transactions on Industrial Informatics*.
2. Wu, Q., et al. (2024). "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation." *arXiv*.
3. Packer, C., et al. (2024). "MemGPT: Towards LLMs as Operating Systems." *arXiv*.
4. Li, Y., et al. (2024). "CrewAI: Multi-Agent Collaboration Framework."
5. BAAI. "BGE: BAAI General Embedding." *arXiv 2023*.
6. Johnson, J., et al. (2019). "Billion-scale similarity search with GPUs." *IEEE Transactions on Big Data (FAISS)*.
7. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*.
8. 中国电子技术标准化研究院. (2023). "工业软件标准体系白皮书."
