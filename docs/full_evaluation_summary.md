# 完整实验汇总

- 生成时间: 2026-05-19T05:36:18Z
- 版本: v0.6.0

## 1. Token 对照（赛题通信效率）

| 模式 | 总 Token | vs 纯文本 | 平均 Relevance | API 调用 |
|------|----------|----------|----------------|----------|
| A 纯文本基线 | 49,076 | 0.0% | — | 36 |
| B 结构化无缓存 | 37,208 | 24.2% | 0.7584166666666666 | 30 |
| C 全功能优化 | 16,615 | 66.1% | 0.84875 | 32 |

## 2. 质量验证

- 综合通过: 12/12
- 平均综合分: 0.936
- 详见: `docs/quality_records.md`

## 3. 缓存相似度（E2E 候选对）

- s1_python_vuln ↔ s3_python_vuln_v2: cos=0.882

## 4. 实验文件索引

| 实验 | 文件 |
|------|------|
| E1 纯文本 | `pure_text_baseline_results.json` |
| E2 全功能 | `optimized_benchmark_results.json` |
| E3 无缓存 | `comparison_benchmark_results.json` |
| E4 质量 | `quality_report.json`, `docs/quality_records.md` |
| E6 多轮 | `multi_run_benchmark_results.json` |
| E7 相似度 | `embedding_similarity_matrix.json` |

设计说明: `docs/experiments/design_v1/comprehensive_suite.md`
文献调研: `docs/strategy/literature_review_2024_2026.md`