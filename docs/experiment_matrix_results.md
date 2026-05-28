# 实验矩阵结果

- 生成时间: 2026-05-19T06:27:09Z
- 版本: v0.9.0

## 变体汇总

| 变体 | Token | vs 纯文本 | Relevance | E2E | Dropout | 质量通过 |
|------|------:|----------:|----------:|----:|--------:|---------:|
| B_nocache | 20,668 | 57.9% | 0.846 | 0 | 1 | 0/12 |
| C_cold_12 | 21,274 | 56.6% | 0.844 | 0 | 0 | 0/12 |
| C_energy_seq | 6,499 | 86.8% | 0.836 | 0 | 2 | 0/12 |
| C_full | 18,092 | 63.1% | 0.840 | 1 | 5 | 0/12 |
| C_no_dropout | 22,340 | 54.5% | 0.836 | 1 | 0 | 0/12 |
| C_no_e2e | 19,756 | 59.7% | 0.844 | 0 | 5 | 0/12 |
| C_warm_12 | 16,463 | 66.5% | 0.830 | 1 | 5 | 0/12 |

## 假设检验

### H2_cache_layer
```json
{
  "B_tokens": 20668,
  "C_tokens": 18092,
  "delta_pct": 12.46
}
```

### H_dropout
```json
{
  "no_dropout_tokens": 22340,
  "full_tokens": 18092,
  "dropout_saves_tokens": 4248
}
```

### H4_warm_memory
```json
{
  "warm_tokens": 16463,
  "cold_tokens": 21274,
  "warm_e2e_hits": 1,
  "cold_e2e_hits": 0
}
```

### H4_domain_sequence
```json
{
  "energy_only_tokens": 6499,
  "per_task": [
    {
      "task_id": "e1_solar_basics",
      "tokens": 1082,
      "strategy": "FULL_GEN"
    },
    {
      "task_id": "e2_solar_advanced",
      "tokens": 2043,
      "strategy": "TEMPLATE_FILL_DROP"
    },
    {
      "task_id": "e3_wind_energy",
      "tokens": 2208,
      "strategy": "TEMPLATE_FILL"
    },
    {
      "task_id": "e4_renewable_comparison",
      "tokens": 1166,
      "strategy": "P1_TEMPLATE_DROP"
    }
  ]
}
```

设计文档: `docs/experiments/design_v1/matrix_v1.md`