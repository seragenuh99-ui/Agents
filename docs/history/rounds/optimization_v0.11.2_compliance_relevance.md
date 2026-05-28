# v0.11.2 优化记录：合规意图 + 复测 full24

> **时间**：2026-05-21  
> **版本**：v0.11.1 → **v0.11.2**

## 改动

1. **意图**：`COMPLIANCE` / `VULNERABILITY` 严格隔离；合规题禁止 executor dropout、禁止 summarizer 摘要缓存。
2. **模板门控**：orchestrator domain hint、planner template fill 全路径加 intent 检查。
3. **贴题**：`summarizer_cache_min_relevance` 0.72 → **0.75**；`REVIEW`（含 security audit）强制 fresh synthesis。
4. **实验**：`full24_controlled_comparison.py` 支持 `--skip-pure-text`、B 逐题落盘、A/B/C 逐题表。
5. **稳定性**：`run_task_batch` 单 Orchestrator 复用；`MemoryStore.get` 加锁。

## full24 控制变量复测（A 沿用 v0.11.1 实测）

| 变体 | Token | 质量 | 相关度 | 墙钟 |
|------|------:|------|-------:|-----:|
| A 纯文本 | 95,377 | 24/24 | 0.868 | 928s |
| **C 结构化** | **30,221** | **24/24** | **0.841** | 209s |
| B 无缓存 | 49,540 | 24/24 | 0.843 | 292s |

- **C vs A**：省 **68.31%**
- **对抗 6 题**：E2E 误用 0/6，质量 **6/6**
- **x3 合规**：`FULL_GEN`，composite 通过（不再套漏洞模板）
- **s2 审计**：C token 1079（v0.11.1 约 2320），相关度 0.821

## 命令

```bash
python3 experiments/full24_controlled_comparison.py --skip-pure-text
```
