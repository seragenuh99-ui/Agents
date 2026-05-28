# 充分实验设计（赛题答辩版）

> **最新完整矩阵**：见 [`matrix_v1.md`](matrix_v1.md)，执行 `python3 experiments/experiment_matrix.py --tier1`。  
> **当前推荐**：[`experiment_design_v2.md`](../../experiment_design_v2.md)。

## 一、实验目标与假设

| 假设 ID | 内容 | 验证方式 |
|---------|------|----------|
| H1 | 结构化协议 + 引用化可显著降低 token | 纯文本 vs 结构化（无缓存） |
| H2 | 共享记忆与 E2E/模板缓存进一步降低 token | 结构化无缓存 vs 全功能 |
| H3 | 非文本向量传递体积小于等效文本 | `test_performance` / 指标导出 |
| H4 | 同域连续任务可复用记忆，降延迟 | 12 任务顺序跑 + 策略分布 |
| H5 | 优化后摘要仍切题 | `quality_benchmark` + LLM judge |
| H6 | 结果对随机性不敏感 | `multi_run_benchmark -n 3` |

---

## 二、实验矩阵（必做 + 选做）

### 2.1 必做（答辩最小集）

| # | 实验 | 脚本 | 输出 | 预计 API 成本 |
|---|------|------|------|---------------|
| E1 | 纯文本基线 12 任务 | `pure_text_baseline.py` | `pure_text_baseline_results.json` | ~49k tok |
| E2 | 全功能优化 12 任务 | `optimized_benchmark.py` | `optimized_benchmark_results.json` | ~16–20k tok |
| E3 | 结构化 **无缓存** 12 任务 | `comparison_benchmark.py` | `comparison_benchmark_results.json` | ~37k tok |
| E4 | 质量验证 + 问答记录 | `quality_benchmark.py` | `quality_report.json`, `docs/quality_records.md` | ~30k tok |
| E5 | **一键汇总** | `full_evaluation_suite.py` | `full_evaluation_report.json`, `docs/full_evaluation_summary.md` | E1–E4 可选子集 |

### 2.2 选做（冲高分）

| # | 实验 | 脚本 | 说明 |
|---|------|------|------|
| E6 | 3 次重复 + 方差 | `multi_run_benchmark.py -n 3` | 报告 mean±std |
| E7 | 嵌入相似度矩阵 | `experiments/embedding_matrix.py` | 5–12 任务 cos 热力数据 |
| E8 | 冷启动 vs 热记忆 | suite `--warm-only` 第二遍 12 任务 | 量化缓存增益 |
| E9 | 消融：无 AgentDropout | suite `--no-dropout` | 需代码 flag |
| E10 | 10 轮稳定性 | `main.py continuous --num-tasks 10` | 赛题要求 |

---

## 三、对照组定义

```
A. pure_text          长 prompt + 散文 + 无记忆
B. structured_nocache  SNS + JSON + refs，FAISS 关闭
C. structured_full    B + E2E + P0/P1 + AgentDropout + utility 重排
D. structured_v06     C + database KB + tag 路由（当前 main）
```

**核心对比表**（论文 Table 1）：

| 模式 | Token | Δ vs A | Relevance | API calls |
|------|-------|--------|-----------|-----------|
| A | T_A | — | — | C_A |
| B | T_B | (A-B)/A | R_B | C_B |
| C | T_C | (A-C)/A | R_C | C_C |

---

## 四、指标清单（赛题评分对齐）

### 通信效率（25 分）
- 总 token、prompt/completion 拆分
- 单任务平均 token
- vs 纯文本节省率
- MessagePack 消息体积（`test_protocol`）

### 状态传递（20 分）
- 传递次数、bytes/次（384×4）
- vs 文本等效压缩率

### 记忆复用（20 分）
- 缓存策略分布（E2E / TEMPLATE / FULL）
- 跨任务 memory hit（orchestrator `_suggest_memories`）
- 同域第 2 任务延迟下降

### 系统完整性（20 分）
- 287+ pytest
- 12 任务 0 硬错误
- Docker / openEuler 可运行

### 实验验证（15 分）
- 多轮重复、消融、质量评判、问答记录

---

## 五、推荐执行顺序（约 1–2 小时 API 时间）

```bash
cd /home/chen/projects/702solver

# 一键（跳过已存在结果可 --skip-existing）
python3 experiments/full_evaluation_suite.py

# 或分步：
python3 experiments/pure_text_baseline.py          # E1
python3 experiments/comparison_benchmark.py        # E3 (含 no-cache)
python3 experiments/optimized_benchmark.py         # E2
python3 experiments/quality_benchmark.py --no-judge  # E4 启发式
python3 experiments/rejudge_quality.py             # E4 LLM 评判
python3 experiments/multi_run_benchmark.py -n 3      # E6
python3 experiments/embedding_matrix.py            # E7
```

---

## 六、v0.6 代码变更（相对 v0.5.0）

| 变更 | 对应文献/工业 |
|------|----------------|
| Retriever 增加 `database systems` KB | 任务感知检索 / PlugMem |
| Tag → KB 类别路由 | MemCollab task-aware |
| Memory utility 重排 | MS / PlugMem 复用加权 |
| 主题同义词验证 | 减少评测误报 |
| Summarizer max_tokens 1024 | 修复 LLM judge「截断」 |
| `full_evaluation_suite.py` | 实验充分性 |

---

## 七、答辩叙事建议

1. **先展示矩阵**：A/B/C 三列 token 柱状图（数据来自 `full_evaluation_report.json`）。
2. **再展示 1 页问答**：打开 `quality_records.md` 任选 security + database 各 1 题。
3. **诚实边界**：LatentMAS 不可 API 实现；事实正确性需人工抽检；Retriever 为可控 KB 非联网搜索。
