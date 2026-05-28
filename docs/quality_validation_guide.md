# 质量验证与问答记录说明

## 目的

弥补「仅有 embedding relevance、无事实核查、无问答留档」的不足，提供：

1. **每题问题 + 最终回答** 的可读记录（`docs/quality_records.md`）
2. **多层自动验证**（启发式 + 可选 LLM 评判）
3. **多次跑分统计**（均值 ± 标准差）

---

## 验证方法设计

### 第一层：启发式（无额外 API 成本）

| 检查项 | 方法 | 通过条件 |
|--------|------|----------|
| 嵌入相关性 | BGE 向量 cosine(task, answer) | 计入综合分 35% |
| 主题覆盖 | `expected_topics` 关键词是否出现在回答中 | ≥40% 词命中 |
| 域一致性 | `CROSS_DOMAIN_MARKERS` 禁词检测 | 无跨域污染词 |
| 结构卫生 | `_error` / `_fallback` 标记 | 无 `_error` |
| 非空 | 回答长度 | >40 字符 |

**启发式综合分** = 0.35×相关性 + 0.30×主题覆盖 + 0.20×域一致 + 0.15×无错误

默认通过阈值：**0.65**

### 第二层：LLM 评判（可选，每题约 +200 token）

调用 DeepSeek，输出 JSON：

```json
{"on_topic": true, "score": 4, "reason": "...", "issues": []}
```

- `on_topic`：是否切题、是否错域  
- `score`：1–5 分  
- 纳入综合分：0.55×启发式 + 0.30×(score/5) + 0.15×on_topic

---

## 命令

```bash
# 1. 跑 12 任务 + 验证 + 生成问答记录（含 LLM 评判）
python3 experiments/quality_benchmark.py

# 2. 仅启发式（更快、更省 token）
python3 experiments/quality_benchmark.py --no-judge

# 3. 对已有 quality_report.json 补跑 LLM 评判
python3 experiments/rejudge_quality.py

# 4. 多次跑分（默认 3 次）得均值±方差
python3 experiments/multi_run_benchmark.py -n 3
```

## 输出文件

| 文件 | 内容 |
|------|------|
| `quality_report.json` | 完整结构化记录（问题、回答、验证、指标） |
| `docs/quality_records.md` | 人类可读的 12 题问答 + 验证表 |
| `multi_run_benchmark_results.json` | 多轮 token/relevance 统计 |

---

## 如何阅读 `quality_records.md`

每个任务三节：

1. **问题**：原始任务描述（评审可对照）  
2. **最终回答**：要点列表 + 结论（来自 Summarizer JSON）  
3. **验证**：PASS/FAIL、各指标、未覆盖主题词、问题列表  

`⚠️ _fallback` 表示 LLM 失败后用证据行拼装的兜底摘要，不算硬错误但质量略低。

---

## 局限（诚实说明）

- 主题词覆盖 ≠ 事实正确（「提到 SQL」不代表修复方案对）  
- LLM 评判仍有偏差，不能替代人工专家抽检  
- Retriever 仍为内置 KB，验证的是「系统输出是否切题」而非「检索源是否真实」  

建议答辩时：自动验证 + 每域抽 1 题人工快速扫一眼 `quality_records.md`。
