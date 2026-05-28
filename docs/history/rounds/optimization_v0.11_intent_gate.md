# v0.11 优化记录：意图门控 + 质量修复

> **日期**：2026-05-19  
> **版本**：v0.11.0 → **v0.11.1**（微调 Dropout 策略）  
> **动机**：v0.10 主基准中 `e4_renewable_comparison` 相关度仅 **0.667**，摘要与 `e3_wind_energy` 完全相同。

---

## 一、问题诊断

| 现象 | 数据 |
|------|------|
| e4 策略（v0.10） | `P1_TEMPLATE_DROP`（2 次 API） |
| e4 相关度（v0.10） | **0.667** |
| e3/e4 摘要 | `summary_preview` 逐字相同 |
| 根因 | Summarizer 对同域结果做缓存复用，未区分 **对比类** vs **单主题分析** |

---

## 二、改动清单

### 1. 新增 `src/task_intent.py`

- `detect_intent()` / `cache_intents_compatible()` / `requires_fresh_synthesis()`
- 对比题禁止 Summarizer E2E；E2E/Planner 直接复用前做意图校验

### 2. Orchestrator / Planner / Summarizer

- E2E 与计划直接复用：意图门控
- Summarizer：对比题 `skip_summary_cache`；E2E 前嵌入相关度 ≥0.72
- Summarizer 显式接收 `task_description`

### 3. RunOptions

- `enable_intent_cache_gate: bool = True`
- `summarizer_cache_min_relevance: float = 0.72`

### 4. v0.11.1 微调

- **撤销**对比题禁止 Executor Dropout（质量靠 Summarizer 门控即可）
- 对比题恢复 `P1_TEMPLATE_DROP`，Token 回落

### 5. 测试与脚本

- `tests/test_task_intent.py`、`tests/test_compare_cache_gate.py`
- `experiments/threshold_grid.py`

---

## 三、复现命令

```bash
pytest tests/test_task_intent.py tests/test_compare_cache_gate.py -q
python3 experiments/optimized_benchmark.py
python3 experiments/threshold_grid.py --dry-run
```

---

## 四、结果记录

| 指标 | v0.10 | v0.11.0（初跑） | **v0.11.1** | 说明 |
|------|------:|----------------:|------------:|------|
| 总 Token | 14,680 | 21,359 | **16,127** | 比 v0.10 +1,447；质量换 Token |
| 平均相关度 | 0.815 | 0.837 | **0.837** | +0.022 |
| e4 相关度 | 0.667 | 0.811 | **0.816** | 已修复 |
| e4 策略 | P1_TEMPLATE_DROP | P1_TEMPLATE | **P1_TEMPLATE_DROP** | v0.11.1 恢复 Dropout |
| E2E 命中 | 1 | 1 | **1** | s3 |
| Dropout 命中 | 8 | 1 | **5** | |
| 测试 | 296 | 303 | **304** | |

**产物**：`output/results/optimized_benchmark_results.json`（version: v0.11.1）

**e4 摘要（v0.11.1，节选）**：Solar LCOE / wind onshore LCOE / capacity factors 对比 — 不再与 e3 风能单题相同。

---

## 五、结论

- **质量**：e4 误复用已消除，平均相关度提升。
- **Token**：相对 v0.10 多花约 **10%**（+1.4k/12 题），相对纯文本仍省 **67.1%**。
- **原则**：对比/政策类意图 → **只门控 Summarizer 缓存**，不必一律禁止 Dropout。

---

## 六、后续

- [ ] `EMBEDDING_MODEL=BAAI/bge-base-en-v1.5` 重跑
- [ ] `threshold_grid.py` 全量 9 组合
- [x] `quality_benchmark.py` 12/12；full24 `benchmark_full24.py` **24/24**
- [x] 同步 `CURRENT_STATUS_AND_ROADMAP.md`、`STATUS_SUMMARY.md`、默认 full24

---

*维护：大实验后更新第四节。*
