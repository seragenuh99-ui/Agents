# 702solver 优化战略 v2（文献调研 + 根因分析 + 落地路线）

> **基线**：v0.7.0 — 16,615 tokens（−66.1%）、relevance 0.849、质量 12/12 composite 0.948  
> **本版目标**：v0.8.0 — Token 再降 8–15%，方差收敛，缓存误用可反馈  
> **v0.8 实测**（单次）：**16,788 tokens**（−65.8%）、avg rel **0.853**、TEMPLATE_FILL_DROP **5/12**  
> **关联报告**：`docs/report_v1_complete.md`

---

## 一、根本结论：效果由什么决定

消融（A/B/C）表明：

| 层级 | Token | 相对纯文本 |
|------|------:|-----------:|
| A 纯文本 | 49,076 | 0% |
| B 结构化无缓存 | 37,208 | 24% |
| C 全功能 | 16,615 | **66%** |

**B→C 再省 55%** → 赛题核心收益来自 **少调 LLM + 少塞上下文**，而非 MessagePack 字节。

提升应沿三条轴：

1. **扩大跳过 LLM 面积**（E2E / 强模板 / Dropout）  
2. **压住 completion 膨胀**（schema、证据预算）  
3. **实验可信度**（序列任务、Pareto、对抗集）

---

## 二、文献对照（2024–2026，API 可落地）

| 方向 | 工作 | 报告效果 | 我们 v0.7 | v0.8 动作 |
|------|------|----------|-----------|-----------|
| 动态剪枝 | [AgentDropout](https://aclanthology.org/2025.acl-long.1170/) ACL 2025 | prompt ↓~22% | Executor Dropout | 保持 |
| 消息图剪枝 | [AgentPrune](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bbc461518c59a2a8d64e70e2c38c4a0e-Abstract-Conference.html) ICLR 2025 | ↓28–73% | 证据 900 字 | **全局 800 字 + 去重** |
| 经验式剪枝 | [SafeSieve](https://arxiv.org/html/2508.11733v3) | ↓12–28%，保准确率 | 未做 | **fill 成功率反馈** |
| 程序记忆 | [LEGOMem](https://arxiv.org/html/2510.04851) | 小模型受益 | 扁平 memory | P1：子任务模板 |
| 检索质量 | [RAGentA](https://arxiv.org/html/2506.16988v1) | 忠实度↑ | KB 模拟 | P2：CRAG 门控 |
| 联合训练 | [OPTIMA](https://arxiv.org/abs/2410.08115) | 少 Token 高 F1 | 无 | ❌ 需 SFT |
| 图联合剪枝 | [AGP](https://arxiv.org/abs/2506.02951v2) | 极高 Token↓ | 固定拓扑 | ❌ 需训练 |

**不宜追逐**：LatentMAS / 模型内 KV 直传、加长 system prompt 蹭缓存（已证 completion 膨胀）、无 tag 的 E2E。

---

## 三、当前瓶颈（数据驱动）

| 现象 | 根因 | 对策 |
|------|------|------|
| 仅 s1↔s3 E2E（cos 0.882） | 阈值 0.85 + 仅一对超线 | 嵌入升级 / Summarizer-E2E（P1） |
| e1↔e2 cos 0.775 | 落在模板带，仍 2–5 次 API | SafeSieve + 0.78 带优化 |
| 3-run 方差 ±640 tok | s2 等 completion 尖峰 | **schema + 证据预算** |
| composite 0.948 已高 | 评测≠事实；KB 模拟 | 真实 RAG / 人工抽检（P2） |
| FULL_GEN ×4 | 冷启动域 | 同域序列 benchmark |

---

## 四、v0.8.0 已实施（本 sprint）

| # | 措施 | 模块 | 机制 |
|---|------|------|------|
| 1 | **SafeSieve-lite** | `memory/store.py`, `planner.py`, `orchestrator.py` | `fill_attempts/successes`；低成功率模板降权；任务结束写回 |
| 2 | **Summarizer 硬 schema** | `summarizer.py` | 固定 3 字段；max_tokens 768；证据上限 1800 字；校验剔除多余键 |
| 3 | **Retriever 证据预算** | `retriever.py` | memory_id 去重；全局 **800** 字上限（原 900 无去重） |

---

## 五、路线图（按 ROI）

### P0 — 下一 sprint（中高收益）

| # | 措施 | 预估 |
|---|------|------|
| 4 | 跑 v0.8 benchmark + 3-run 对比 v0.7 | 数据闭环 |
| 5 | `bge-base-en-v1.5` 重跑嵌入矩阵 | 多 1–2 次 E2E/模板 |
| 6 | Summarizer-E2E（plan 模板 + 摘要 cos>0.85） | Token ↓ |
| 7 | Planner 模板只填 `delta_fields` | completion ↓ |

### P1 — 质量与记忆

| # | 措施 |
|---|------|
| 8 | SafeSieve 接入 `quality_validator` composite_pass |
| 9 | P4 证据蒸馏（同域 ≥4 条 → distilled） |
| 10 | LEGOMem 式子任务 DAG 模板 |
| 11 | CRAG：低分 evidence 不进入 Summarizer |

### P2 — 实验与答辩

| # | 措施 |
|---|------|
| 12 | 同域 5 题序列（测缓存累积） |
| 13 | 阈值 Pareto（0.82/0.85/0.88 × template_score） |
| 14 | 单 Agent 强基线 |
| 15 | LangGraph 同任务对照（可选） |

---

## 六、合理上限（不训练、不重构）

| 维度 | v0.7 | 合理上限 | 手段 |
|------|------|----------|------|
| Token | −66% | **−72%~−78%** | 更多 E2E + completion 约束 |
| relevance | 0.849 | 0.86–0.88 | 嵌入升级 + CRAG |
| composite | 0.948 | 0.95+（自动评测） | KB/检索升级才突破 |

---

## 七、复现命令

```bash
# v0.8 主基准
python3 experiments/optimized_benchmark.py

# 对比 v0.7 结果
python3 experiments/ablation_analysis.py

# 3-run 方差
python3 experiments/multi_run_benchmark.py -n 3

# 质量
python3 experiments/quality_benchmark.py && python3 experiments/rejudge_quality.py

pytest tests/ -q
```

---

## 八、参考文献（简表）

1. AgentDropout — ACL 2025 — arXiv:2503.18891  
2. AgentPrune / Cut the Crap — ICLR 2025 — arXiv:2410.02506  
3. SafeSieve — arXiv:2508.11733  
4. OPTIMA — ACL Findings 2025 — arXiv:2410.08115  
5. LEGOMem — arXiv:2510.04851  
6. RAGentA — arXiv:2506.16988  
7. Memory Sharing — arXiv:2404.09982  

---

## 附录 A：v0.7 实验矩阵数据驱动路线图

> 以下为 `docs/archive/optimization_roadmap_v0.7.md` 的完整保留副本（与 v0.8+ 战略互补，数据以 v0.7 为准）。

### A.1 三层节省

| 层级 | 模式 | Token | 相对纯文本 A |
|:----:|------|------:|:------------:|
| A | 纯文本多 Agent | 49,076 | 0% |
| B | 结构化、无跨任务缓存 | 37,208 | **24.2%** |
| C | 全功能 v0.7.0 | **16,615** | **66.1%** |

B→C 再省 **55.6%**。v0.7 策略分布：TEMPLATE_FILL×7、FULL_GEN×3、P1_TEMPLATE_DROP×1、E2E_EXACT×1（s3←s1）。

### A.2 嵌入矩阵 cos 区间

| cos | 示例 | 行为 |
|:---:|:-----|------|
| >0.85 | s1↔s3 (0.882) | E2E ✅ |
| 0.70–0.85 | e1↔e2 (0.775) | 模板填空 ✅ |
| 0.50–0.70 | e1↔e3 (0.71) | P0 judge + 模板 |
| <0.50 跨域 | energy↔security | 不缓存 ✅ |

瓶颈：同域不同角度 cos 常 0.60–0.75，依赖 P0 judge（~50 tok/次）。

### A.3 方案横向对比（摘要）

| 方案 | 我们 |
|------|:----:|
| SNS / refs / E2E / AgentDropout | ✅ |
| AgentPrune / CodeAgents 骨架 | ⚠️ 部分 |
| SafeSieve / 真实 RAG / LatentMAS | ❌ / 待做 |

### A.4 ROI 排序（v0.7 时点）

**P0**：3-run 方差、Summarizer schema、证据 800 字上限、e1↔e2 轻量 judge。  
**P1**：SafeSieve-lite、证据蒸馏、bge-base、单 Agent 基线。  
**P2**：真实 RAG、人工 fact-check、LangGraph 对照。  
**不建议**：拉长 prompt 蹭 KV cache、无 tag 的 E2E、去掉 Summarizer。

完整表格与 checklist 见 [`../archive/optimization_roadmap_v0.7.md`](../archive/optimization_roadmap_v0.7.md)。

---

*文档版本：v2.1 | 对应代码：v0.9.0 | 更新：2026-05-19*
