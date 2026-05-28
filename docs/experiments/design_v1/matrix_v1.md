# 完整实验矩阵设计 v1

> **目的**：用可复现的对照实验验证假设，再数据驱动优化（v0.9+）。  
> **执行入口**：`python3 experiments/experiment_matrix.py --tier1`  
> **输出**：`experiment_matrix_report.json`、`docs/experiment_matrix_results.md`、`docs/optimization_from_matrix.md`

---

## 一、研究假设

| ID | 假设 | 验证变体 |
|----|------|----------|
| H1 | 结构化协议降低通信冗余 | A vs B（见既有 pure_text / nocache） |
| H2 | 跨任务缓存是 Token 主因 | `B_nocache` vs `C_full` |
| H3 | 向量状态传递体积 < 文本 | `collect_system_metrics.py`（无 API） |
| H4 | 热记忆累积降低 Token / 增 E2E | `C_warm_12` vs `C_cold_12`；`C_energy_seq` |
| H5 | AgentDropout 净节省 Token | `C_no_dropout` vs `C_full` |
| H6 | E2E 精确复用有效且省 API | `C_no_e2e` vs `C_full`；`embedding_matrix` |
| H7 | 优化后仍切题 | 每变体 `validate_task` heuristic |
| H8 | 结果对随机性稳健 | `multi_run_benchmark -n 3`（`--all`） |

---

## 二、实验变体（对照组）

| 变体 ID | RunOptions 要点 | 任务集 | DB 策略 |
|---------|----------------|--------|---------|
| `C_full` | 默认全功能 | 12 | 单库顺序 |
| `B_nocache` | 关闭 index + 全部缓存 | 12 | 单库 |
| `C_no_dropout` | `enable_executor_dropout=False` | 12 | 单库 |
| `C_no_e2e` | `enable_e2e_cache=False` | 12 | 单库 |
| `C_warm_12` | 全功能 | 12 | **同一 DB** 顺序跑 |
| `C_cold_12` | 全功能 | 12 | **每题清空 DB** |
| `C_energy_seq` | 全功能 | 4 energy | 单库顺序 |
| `A_pure_text` | text mode | 12 | `pure_text_baseline.py`（`--all`） |

---

## 三、指标

### 必报（每变体）

- `total_tokens`, `prompt_tokens`, `completion_tokens`
- `total_api_calls`, `avg_relevance`
- `strategy_distribution`, `e2e_hits`, `dropout_hits`
- `quality_pass_count`（heuristic composite ≥ 0.65）
- `save_vs_pure_text_pct`

### 微基准（无 API）

- `system_metrics.json`：协议 / 状态 / 记忆
- `embedding_similarity_matrix.json`：E2E 候选对

---

## 四、执行计划

### Tier 1（约 40–60 min API，答辩推荐）

```bash
cd /home/chen/projects/702solver
python3 experiments/experiment_matrix.py --tier1
```

包含：4 消融 + warm/cold + energy 序列 + 微基准 + 自动分析。

### Tier 2（完整，约 2h+ API）

```bash
python3 experiments/experiment_matrix.py --all
python3 experiments/quality_benchmark.py
python3 experiments/rejudge_quality.py
```

### 单变体调试

```bash
python3 experiments/experiment_matrix.py --variant C_no_dropout
```

---

## 五、决策规则（实验 → 优化）

| 观测 | 动作 |
|------|------|
| `C_no_dropout` − `C_full` > 500 tok | 保持 Dropout；可试 threshold 0.65 增覆盖 |
| `C_no_dropout` < `C_full` | 提高 `executor_dropout_threshold` → 0.75 |
| `C_warm` E2E > `C_cold` E2E + 1 | 答辩强调「顺序任务记忆累积」 |
| `B_nocache` − `C_full` > 50% | 论文强调缓存层，非协议 alone |
| `quality_pass` < 12 | 收紧 SafeSieve / e2e_threshold |
| `avg_relevance` < 0.82 | 检查跨域缓存；升嵌入模型 |

分析脚本：`experiments/analyze_experiment_matrix.py` → `docs/optimization_from_matrix.md`

---

## 六、与赛题评分映射

| 赛题维度 | 对应实验 |
|----------|----------|
| 通信效率 25 | A/B/C Token 表 + 协议微基准 |
| 状态传递 20 | system_metrics state_transfer |
| 记忆复用 20 | warm/cold + energy_seq + E2E 矩阵 |
| 系统完整 20 | pytest 294+ |
| 实验验证 15 | 矩阵 7 变体 + 假设表 + 质量 heuristic |

---

## 七、文件索引

| 文件 | 说明 |
|------|------|
| `src/run_options.py` | 消融开关与阈值 |
| `experiments/experiment_common.py` | 统一跑批 |
| `experiments/experiment_matrix.py` | 矩阵执行器 |
| `experiment_matrix/*.json` | 各变体原始结果 |
| `experiment_matrix_report.json` | 汇总 |
| `docs/strategy/improvement_strategy_v2.md` | 文献与 ROI 路线图 |

---

*v1 | 2026-05-19*
