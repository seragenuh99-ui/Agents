# 问题诊断与学界解决方案调研

## 一、当前系统问题清单

通过 Round 1–4 的迭代实验和 10 轮稳定性测试，识别出以下 7 个问题：

### 问题 1：嵌入模型能力瓶颈（核心问题）

**现象**：MiniLM-L6-v2（384 维）只捕捉措辞层面的相似度，不识别领域/任务层面的关联。

| 任务对 | cos | 是否触发缓存 |
|--------|:---:|:----------:|
| "研究太阳能技术（光伏、效率、成本、环境影响）" → "研究太阳能技术：光伏、效率测量、安装成本、环境足迹" | **0.959** | ✅ E2E 全复用 |
| "研究太阳能技术…" → "分析太阳能政策激励及其对住宅光伏采用率的影响" | **0.609** | ❌ 视为全新任务 |
| "研究太阳能技术…" → "研究风能技术并对比太阳能" | **0.579** | ❌ 视为全新任务 |

**根因**：all-MiniLM-L6-v2 在短文本上主要比较关键词重叠。"solar energy technologies" 和 "solar energy policy" 共享 "solar energy" 但模型对 "technologies" vs "policy" 的区分权重很高。

**影响**：同域不同角度的任务——这是最常见的复用场景——完全无法复用。系统只在措辞几乎相同的任务对之间有效。

---

### 问题 2：模板填空区间太窄

**现象**：cos 0.70–0.85 的 Planner 模板填空区间，在 5 任务和 10 轮测试中仅命中 1 次。

**根因**：MiniLM 的余弦分布集中在两极——要么非常相似（>0.90，同义改写），要么不太相似（<0.65，同一领域不同角度）。0.70–0.85 的中间地带很少出现。

---

### 问题 3：Summarizer 缓存几乎不命中

**现象**：Summarizer 结果缓存使用 evidence_refs 的 Jaccard 相似度（阈值 0.5），但不同任务即使内容相似也会产生不同的 evidence_refs（新的 memory_id），Jaccard 通常为 0。

**根因**：evidence_refs 是 memory_id 集合，每个任务生成新的 memory_id，几乎不可能与历史任务的集合重叠超过 50%。

---

### 问题 4：E2E 缓存依赖单一嵌入模型

**现象**：E2E 缓存完全依赖 MiniLM 的余弦相似度判断，没有二次验证机制。

**风险**：如果 MiniLM 对两个实际上不相关的任务产生了高余弦（假阳性），系统会错误复用。反之，真实相似任务可能因余弦不够高而错过（假阴性）。

---

### 问题 5：记忆存储是扁平的

**现象**：所有记忆平铺在 SQLite + FAISS 中，按 memory_type 标签区分，但没有层级结构、没有压缩合并、没有遗忘机制。

**风险**：随着任务累积，记忆数量线性增长。FAISS HNSW 在 500 条时性能没问题（1ms），但万条级别可能退化。且低质量记忆会污染检索结果。

---

### 问题 6：未区分任务类型进行差异化复用

**现象**：当前只有一条复用路径：编码 → FAISS 搜索 → 按 cos 阈值决定策略。对不同类型的关系（同任务改写、子任务包含、互补任务、先决任务）没有任何区分。

---

### 问题 7：稳定性验证不充分

**现象**：只测试了 10 轮 22 条记忆的规模。未测试：
- 500+ 条记忆时 FAISS 搜索是否退化
- 跨 100+ 轮任务的记忆累积和检索质量
- 并发场景（ThreadPoolExecutor 多任务并行时的竞态）
- 记忆去重在大规模下的表现

---

## 二、学界方案调研

针对以上问题，查阅了 2024–2025 年 arXiv、NeurIPS、ICLR、ACL、EMNLP 的相关论文。

### 方案一：两阶段检索 + LLM 重排序

**来源**：MemGuide (2025)、JudgeRank (2025)、Rationale-Augmented Retrieval (2025)、SCORE (2025)

**核心思路**：不单纯依赖嵌入余弦相似度。第一阶段用轻量嵌入做粗筛（召回 top-K），第二阶段用 LLM 判断任务间的真实相关性（精排）。

```
当前:  task → embedding → FAISS top-3 → cos>0.90? → E2E缓存
改进:  task → embedding → FAISS top-10 → LLM判断是否"同类任务" → 如LLM确认 → E2E缓存
```

**为什么有效**：
- LLM 对 task description 的理解远超 384 维向量——它能识别"研究太阳能技术"和"分析太阳能政策"同属太阳能领域
- MemGuide 用 LLaMA-8B 做重排序，+11% 任务成功率
- SCORE 用 LLM 自评估检索结果，"推荐系统的 LLM 自评重排序优于纯嵌入"

**对 702solver 的适用性**：
- ✅ 可在 E2E 缓存和 Planner 缓存前增加 LLM 二次判断
- ✅ LLM 只需判断"这两个任务是否足够相似"——单轮轻量调用（~50 tokens）
- ⚠️ 增加一次 LLM 调用，但如果能命中缓存（省 3 次调用），净赚 2 次

**关键论文**：
- MemGuide: [arXiv:2505.20231](https://arxiv.org/abs/2505.20231) — 意图驱动记忆选择
- SCORE: [arXiv:2505.19464](https://arxiv.org/abs/2505.19464) — LLM 自评检索
- Rationale-Augmented Retrieval: [arXiv:2510.05131](https://arxiv.org/abs/2510.05131)

---

### 方案二：层次化记忆结构

**来源**：G-Memory (NeurIPS 2025)、EVOLVE-MEM (NeurIPS 2025)、MemVerse (2025)

**核心思路**：记忆不应平铺。应按抽象层级组织：

```
当前（扁平）:
  memories: [strategy1, strategy2, evidence1, result1, result2, ...]

改进（层次化）:
  Level 2 - 领域原则: "太阳能研究类任务通常包含：光伏类型、效率对比、成本分析、环境影响"
  Level 1 - 任务模板:   "研究[X]能源技术，包括[类型]、[效率]、[成本]、[环境]"
  Level 0 - 具体记忆:   每次执行的 plan/retrieval/execution/summary 实例
```

**为什么有效**：
- G-Memory 的三层图（Insight/Query/Interaction）在具身任务上 +20.89% 成功率
- EVOLVE-MEM 从 L0（原始）→ L1（摘要）→ L2（原则）自动聚类提炼
- MEM1 的核心思想：记忆不应只是"存和取"，而应该在每次访问时**重组和提炼**

**对 702solver 的适用性**：
- ✅ 可在现有 memory_type 之上增加 abstraction_level 维度
- ✅ 当同一领域的 strategy 积累超过 N 条时，自动触发 LLM 提炼为领域原则
- ⚠️ 提炼是离线操作（可在空闲时执行），不影响在线时延

**关键论文**：
- G-Memory: [arXiv:2506.07398](https://arxiv.org/abs/2506.07398) — 三层图层次记忆
- EVOLVE-MEM: NeurIPS 2025 — 自适应层次记忆，58.3% LoCoMo 准确率
- MEM1: [arXiv:2506.15841](https://arxiv.org/abs/2506.15841) — RL 统一推理与记忆合并，3.5× 性能 + 3.7× 内存压缩

---

### 方案三：记忆自我进化与去重

**来源**：SEDM (NeurIPS 2025)、ReasoningBank (2025)、Evo-Memory/ReMem (Google DeepMind, 2025)

**核心思路**：记忆系统不应是被动仓库，应是主动的、自优化的。

**SEDM 的三个关键机制**：
1. **可验证写入**：新记忆写入前，用 Docker 容器重放验证其效用，只有真正有用的才入库
2. **自调度控制器**：周期性评估每条记忆的 utility 权重，合并冗余、删除低质
3. **跨域知识扩散**：从不同任务中提取可复用的抽象见解

**ReasoningBank 的记忆提取流程**：
1. LLM 自判断任务成功/失败
2. 从成功轨迹中提取 `{Title, Description, Content}` 三元组
3. 从失败轨迹中也提取（"什么不该做"）
4. 新任务 → embedding 检索 → top-K 记忆注入 prompt

**ReMem 的 Think→Act→Refine 循环**：
- Think: 分析当前任务需要什么类型的记忆
- Act: 检索并注入记忆
- Refine: 评估记忆是否有用，更新记忆的 utility 分数

**对 702solver 的适用性**：
- ✅ E2E 缓存的写入端可增加效用验证（验证缓存结果是否真的正确）
- ✅ 可增加 per-memory utility 分数，根据被命中次数和任务成功率动态调整
- ✅ 周期性合并：当同一 domain 的 strategy 超过 5 条时，自动压缩为领域模板
- ⚠️ SEDM 的 Docker 重放太重，可用更轻量的 LLM 自评替代

**关键论文**：
- SEDM: [arXiv:2509.09498](https://arxiv.org/abs/2509.09498) — 可验证自进化记忆
- ReasoningBank: [arXiv:2509.25140](https://arxiv.org/abs/2509.25140) — Google，记忆自进化
- Evo-Memory/ReMem: [arXiv:2511.20857](https://arxiv.org/abs/2511.20857) — DeepMind，经验复用

---

### 方案四：渐进式记忆压缩（MemGPT 模式）

**来源**：MemGPT (2023→Letta 2025)、Recursive Summarization (Neurocomputing 2025)

**核心思路**：借鉴操作系统的虚拟内存管理。具体记忆（episodic）随时间推移逐步压缩为抽象摘要（semantic）。

```
原始记忆（500 tokens）
  ↓ 1天后，如果未被访问
压缩摘要（100 tokens）
  ↓ 再经过 N 次同类任务
领域知识（30 tokens，进入 working memory）
```

**对 702solver 的适用性**：
- ✅ 可解决长期运行的记忆膨胀问题
- ✅ Planner 的 fallback plan 可以作为"领域知识"直接进入 working context
- ⚠️ MemGPT 的递归总结需要 LLM 参与，增加计算开销

**关键论文**：
- MemGPT: 2023 (UC Berkeley) → Letta 框架（16.7K GitHub stars）
- Recursive Summarization: Neurocomputing 2025 — 验证递归总结有效性，+3% BLEU

---

### 方案五：用 LLM 直接判断任务相似度

**来源**：综合分析

**核心思路**：对于任务相似度判断这个特定问题，LLM 比静态嵌入模型强得多。

| 对比维度 | MiniLM-L6-v2 (384d) | LLM (如 DeepSeek-V3) |
|----------|:---:|:---:|
| "太阳能技术" vs "太阳能政策" | cos=0.609（不相似） | 可识别为"同属可再生能源领域" |
| "SQL注入检测" vs "XSS漏洞扫描" | cos≈0.5（不相似） | 可识别为"同属Web安全审计" |
| 成本 | 0（本地计算） | ~50 tokens/次判断 |
| 延迟 | <1ms | ~100ms |

**推荐的混合方案**：
1. FAISS 粗筛（免费，<1ms）：召回 cos>0.50 的 top-10 候选
2. LLM 精排（~50 tokens）：判断候选任务是否"可复用"
3. 根据 LLM 判断结果决定缓存策略

**为什么这是最优方案**：
- FAISS 保证召回率（不放过可能的候选）
- LLM 保证精确率（不误触发缓存）
- LLM 调用量极少（每次缓存判断只需 1 次轻量调用，如果命中则省 3 次重量调用）
- MemGuide、SCORE、JudgeRank 等论文都验证了"embedding 粗筛 + LLM 精排"范式的有效性

---

## 三、改进方案路线图

按投入产出比排序：

| 优先级 | 改进 | 解决问题 | 预估增益 | 实现难度 | 状态 | 参考文献 |
|:------:|------|------|:------:|:--:|:--:|------|
| **P0** | LLM 二次判断任务相似度 | 问题1, 4 | 大幅扩大复用范围 | 低 | ✅ 已完成 | MemGuide, SCORE |
| **P1** | 层次化记忆（模板/实例两级） | 问题2, 5, 6 | 提高模板填空命中率 | 中 | ✅ 已完成 | G-Memory, EVOLVE-MEM |
| **P2** | 记忆 utility 评分 + 自动去重 | 问题5 | 防止记忆膨胀和质量退化 | 中 | ⏳ 待开始 | SEDM, ReMem |
| **P3** | Summarizer 缓存改为语义匹配 | 问题3 | 提高摘要复用率 | 低 | ✅ 已完成 | —（直接用 embedding 替代 Jaccard） |
| **P4** | 渐进式记忆压缩 | 问题5 | 长期运行稳定性 | 高 | ⏳ 待开始 | MemGPT, Recursive Summarization |
| **P5** | 大规模压力测试 | 问题7 | 发现隐藏问题 | 中 | ⏳ 待开始 | — |

### P0 实际效果（5 任务实验验证）

E2E 缓存命中率从 40% → 60%（3/5 vs 2/5）。LLM judge 2/2 判断准确，每次仅 ~53 tokens。

| 场景 | Round 4 | P0 后（实测） | P0+P1 后（预估） |
|------|:----:|:-----:|:--------:|
| 同义改写任务 | E2E 全复用（cos>0.90 直接命中） | 同左（t2, cos=0.902） | 同左 |
| 同域不同角度（风能 vs 太阳能） | 模板填空（cos=0.71） | **E2E 缓存**（cos=0.593, LLM judge=YES） | 领域模板优先 |
| 跨域（能源→安全） | 完整生成 | 完整生成（cos<0.50, 正确拒绝） | 同左 |
| 同域近变体（安全审计变体） | E2E 缓存（cos=0.93） | E2E 缓存（cos=0.841, LLM judge=YES） | 同左 |

### 预期效果

| 场景 | 当前 | P0 后 | P0+P1 后 |
|------|:----:|:-----:|:--------:|
| 同义改写任务 | E2E 全复用（72% 节省） | 同左 | 同左 |
| 同域不同角度（太阳能技术→太阳能政策） | 0% 复用 | E2E 缓存（LLM judge） | 领域模板复用 |
| 跨域结构相似（能源研究→代码审计） | 0% 复用 | 0% 复用（正确拒绝） | 0% 复用（正确拒绝） |
| 10 轮同域递进任务 | 0% 复用 | 逐轮累积复用 | 逐轮累积+领域原则提炼 |

---

## 四、参考文献

| 论文 | 年份/会议 | 链接 | 核心贡献 |
|------|:--------:|------|------|
| MemGuide | 2025 | [arXiv:2505.20231](https://arxiv.org/abs/2505.20231) | 意图驱动记忆选择 + LLM重排序 |
| SCORE | 2025 | [arXiv:2505.19464](https://arxiv.org/abs/2505.19464) | LLM自评检索 |
| Rationale-Augmented Retrieval | 2025 | [arXiv:2510.05131](https://arxiv.org/abs/2510.05131) | 混合检索 + LLM约束重排 |
| G-Memory | NeurIPS 2025 | [arXiv:2506.07398](https://arxiv.org/abs/2506.07398) | 三层图层次记忆，+20.89% |
| EVOLVE-MEM | NeurIPS 2025 | NeurIPS 2025 | 三级自适应层次记忆 |
| MEM1 | NeurIPS 2025 | [arXiv:2506.15841](https://arxiv.org/abs/2506.15841) | RL统一推理与记忆合并 |
| SEDM | NeurIPS 2025 | [arXiv:2509.09498](https://arxiv.org/abs/2509.09498) | 可验证自进化记忆 |
| ReasoningBank | 2025 | [arXiv:2509.25140](https://arxiv.org/abs/2509.25140) | Google，成功/失败轨迹记忆提取 |
| Evo-Memory/ReMem | 2025 | [arXiv:2511.20857](https://arxiv.org/abs/2511.20857) | DeepMind，经验复用框架 |
| MOSAIC | 2025 | [arXiv:2506.05577](https://arxiv.org/abs/2506.05577) | Wasserstein任务嵌入 + 策略组合 |
| MemGPT | 2023 | UC Berkeley → Letta | 层次化虚拟内存管理 |
| Recursive Summarization | 2025 | Neurocomputing | 递归总结有效性验证 |
| MemInsight | 2025 | [arXiv:2503.21760](https://arxiv.org/abs/2503.21760) | Amazon，结构化记忆增强，+34% Recall |
