# 实验矩阵 v2 结果

> **注意**：full24 主数据请以 **v0.11.1** 为准 → [`full24_benchmark_results.md`](full24_benchmark_results.md)（2026-05-20，28,561 tok）。

- 时间: 2026-05-19T07:30:00Z（core 消融）；full24 见上
- 设计: `docs/experiment_design_v2.md`

## 变体汇总

| 变体 | Token | vs 纯文本 | Relevance | E2E | 质量通过 | 对抗 E2E 安全 |
|------|------:|----------:|----------:|----:|---------:|:-------------:|
| B_nocache_core12 | 20,767 | 57.7% | 0.838 | 0 | 12/12 | — |
| C_adversarial6_warm | 15,517 | 68.4% | 0.848 | 0 | 6/6 | ✓ |
| C_cold_core12 | 22,131 | 54.9% | 0.835 | 0 | 12/12 | — |
| C_full_core12 | 17,558 | 64.2% | 0.838 | 1 | 12/12 | — |
| C_full_core12_llm_judge | 17,344 | 64.7% | 0.836 | 1 | 12/12 | — |
| C_full_full24 | 35,110 | 28.5% | 0.848 | 1 | 24/24 | ✓ |
| C_no_dropout_core12 | 22,352 | 54.5% | 0.848 | 1 | 12/12 | — |
| C_no_e2e_core12 | 19,472 | 60.3% | 0.841 | 0 | 12/12 | — |
| C_warm_core12 | 17,226 | 64.9% | 0.839 | 1 | 12/12 | — |
| warm_core12_for_adv | 0 | — | 0.000 | 0 | — | — |

## 单 Agent 强基线

- Token: 4,568
- 质量通过: 12

人工抽检: `docs/human_factcheck_pack.md`
