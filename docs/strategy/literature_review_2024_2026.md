# 学界与工业界方案调研（2024–2026）及本项目的可落地借鉴

> 面向赛题：多 Agent 低开销通信、非文本状态传递、共享记忆复用。  
> 当前基线：**v0.5.0**（16,480 tokens / -66.4% / relevance 0.838）

---

## 一、通信与 Token 效率

| 工作 | 来源 | 核心机制 | 报告效果 | API 可落地 | 本项目状态 |
|------|------|----------|----------|:----------:|:----------:|
| **SNS-Core** | GitHub / 社区 | 速记符号 prompt | 60–85% ↓ | ✅ | ✅ 已用 |
| **AgentPrune** | ICLR 2025, [arXiv:2410.02506](https://arxiv.org/abs/2410.02506) | 时空消息图剪枝，去冗余边 | 28–73% ↓ | ✅ 启发式 | ⚠️ 部分（Executor Dropout） |
| **AgentDropout** | ACL 2025, [arXiv:2503.18891](https://arxiv.org/abs/2503.18891) | 动态去 Agent/边 | ~21.6% prompt ↓ | ✅ 规则 | ✅ Executor 跳过 |
| **CodeAgents** | [arXiv:2507.03254](https://arxiv.org/abs/2507.03254) | 伪代码/控制流表示交互 | 55–87% I/O ↓ | ✅ | ⚠️ SNS 骨架，可加深 |
| **SafeSieve** | [arXiv:2508.11733](https://arxiv.org/abs/2508.11733) | 渐进剪枝 + 语义评估 + 经验反馈 | 12–28% ↓，保准确率 | ✅ | ❌ 未做 |
| **Cut the Crap** | OpenReview LkzuPorQ5L | 经济型通信管道 | 显著 ↓成本 | ✅ | ❌ 未做 |
| **ANX Protocol** | [arXiv:2604.04820](https://arxiv.org/abs/2604.04820) | 高密度机器可执行 SOP | 47–66% ↓ | ✅ | ❌ 调研级 |
| **LatentMAS** | arXiv:2511.20639 | Hidden state / KV 直传 | 70–83% ↓ | ❌ 需模型内部 | ❌ 不可行 |

### 建议采纳（v0.6 方向）

1. **AgentPrune 启发式**：空结果不传、同轮重复 ref 合并、低相关 `suggested_memories` 不注入（已实现 tag 路由 + 记忆 utility 重排）。
2. **SafeSieve 轻量版**：对 borderline 缓存（cos 0.5–0.85）用「历史成功率」而非仅 LLM judge——记录每次 template fill 的 `composite_pass`，失败则降权。
3. **CodeAgents**：Retriever/Executor 输出改为 `{findings:[], refs:[]}` 固定 schema，减少散文。

---

## 二、共享记忆与跨任务复用

| 工作 | 来源 | 核心机制 | 本项目差距 |
|------|------|----------|------------|
| **Memory Sharing (MS)** | [arXiv:2404.09982](https://arxiv.org/abs/2404.09982) | 多 Agent 共享记忆池、query-response 对 | 有 SQLite+FAISS，缺「按 Agent 角色」分池 |
| **PlugMem** | [arXiv:2603.03296](https://arxiv.org/abs/2603.03296) | 知识图式记忆、任务无关插件 | 扁平 memory，无命题/规范分层 |
| **MemCollab** | [arXiv:2603.23234](https://arxiv.org/abs/2603.23234) | 对比轨迹蒸馏、去模型偏差 | 无轨迹蒸馏 |
| **MemoriesDB** | [arXiv:2511.06179](https://arxiv.org/abs/2511.06179) | 时序-语义-关系图 | 无时间衰减/关系边 |
| **Collaborative Memory** | [arXiv:2505.18279](https://arxiv.org/abs/2505.18279) | 多用户 ACL、溯源 | 单用户、有 tags 过滤 |

### 建议采纳

1. **P2 utility 评分**（已做 lite）：`search_by_similarity` 用 `0.85*sim + 0.15*access_count` 重排。
2. **P4 渐进压缩**：同域 ≥5 条 `evidence` 合并为 1 条 `distilled`（abstraction_level=2）。
3. **任务感知检索**：Retriever KB 按 tags 路由（**v0.6 已加 database KB**）。

---

## 三、工业界实践（非论文）

| 方案 | 做法 | 可借鉴 |
|------|------|--------|
| **LangGraph / CrewAI** | 状态图 + checkpoint | 对照实验基线（未做） |
| **Mem0 / Zep** | 长期记忆 API、自动摘要 | 蒸馏合并策略 |
| **OpenAI Prompt Caching** | 前缀 KV 缓存 | 已测：长 prompt 反而涨 completion，未采用 |
| **DeepSeek API** | `cached_tokens` | 短 system prompt 策略保留 |

---

## 四、与当前实验的对应关系

| 不足（自评） | 文献/工业对策 | 本仓库动作 |
|--------------|----------------|------------|
| 只跑 1 次 | 多次 run + 置信区间 | `multi_run_benchmark.py`、**`full_evaluation_suite.py`** |
| 无消融 | AgentPrune / 去缓存对照 | `comparison_benchmark.py` + suite 汇总 |
| 无问答留档 | 人工 + 自动评判 | `quality_benchmark.py` → `quality_records.md` |
| DB 域 evidence 偏 | 任务感知 KB | **database systems KB + tag 路由** |
| 主题词误报 | 同义词扩展 | **TOPIC_SYNONYMS** |
| 摘要截断 | 提高 max_tokens | Summarizer **1024** |

---

## 五、参考文献（ Bib 简表）

```bibtex
@inproceedings{agentdropout2025,
  title={AgentDropout: Dynamic Agent Elimination for Token-Efficient Multi-Agent Collaboration},
  booktitle={ACL},
  year={2025}
}
@inproceedings{agentprune2025,
  title={Cut the Crap: Economical Communication Pipeline for LLM-based Multi-Agent Systems},
  booktitle={ICLR},
  year={2025}
}
@article{memorysharing2024,
  title={Memory Sharing for Large Language Model based Agents},
  journal={arXiv:2404.09982},
  year={2024}
}
```

完整实验矩阵见 `docs/experiments/design_v1/comprehensive_suite.md`（v1）；当前 v2 见 `docs/experiment_design_v2.md`。
