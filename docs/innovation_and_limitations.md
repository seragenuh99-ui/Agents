# 创新点与实验不足（答辩参考）

> 与 [`PROJECT_RECORD.md`](PROJECT_RECORD.md) 配套。数据以 v0.9 + v2 实验为准。文档索引见 [`README.md`](README.md)。

---

## 一、创新点

### 1. 可消融的「少调 LLM」架构

消融表明：结构化协议约省 24% Token，**跨任务缓存层再省约 55%**（B→C）。创新叙事应强调 **缓存 + Dropout**，而非仅 MessagePack。

### 2. 四级缓存 + 标签门控

E2E（cos≥0.85）→ 模板填空（0.78+）→ LLM 判同类（0.55+）→ 全量生成；`no-e2e-cache` 标签用于对抗题禁止误复用。

### 3. AgentDropout 工程化

模板分≥0.70 跳过 Executor LLM；矩阵实验 `C_no_dropout` vs `C_full` 可量化节省。

### 4. 向量 + 引用一体化

固定维状态传递 + memory_ref，微基准体积降约 78%（相对等长文本）。

### 5. 实验 v2 充分性

- 24 题（含 extended / adversarial）
- 单 Agent 强基线
- 热/冷记忆、LLM judge 子实验、人工 6 题包

### 6. 对话模式

`chat.py`：用户可直接使用 v0.9 全流水线，会话记忆持久化。

---

## 二、不足与对策（v2 已部分解决）

| 不足 | v2 对策 | 状态 |
|------|---------|------|
| 仅 12 题 | full24 + adversarial6 | 已设计 |
| 无强基线 | `single_agent_baseline.py` | 已实现 |
| 质量≠事实 | forbidden_topics + LLM judge + 人工包 | 已实现 |
| E2E 难验证 | adversarial6 + `adversarial_e2e_safe` | 已设计 |
| 无 LangGraph 对照 | 文档说明 scope；单 Agent 作代理 | 文档 |
| 单模型 | 保留 DeepSeek；`EMBEDDING_MODEL` 可换 | 可选 |
| 模拟 KB | 未接真实 RAG | 待做 |

---

## 三、答辩 30 秒稿

我们实现了带共享记忆的多 Agent 系统，在真 API 下相对纯文本长报告节省约 **67% Token**，12 题自动质量全通过。消融证明收益来自缓存与工序剪枝。v2 实验扩展至 24 题并设对抗集验证缓存安全，提供单 Agent 对照与对话接口可直接体验系统。
