# 实验矩阵驱动的优化建议

基于 `/home/chen/projects/702solver/experiment_matrix_report.json` 自动生成。

## 结论摘要

缓存层（B→C）贡献约 **12.5%** token 节省 (20,668 → 18,092)。
AgentDropout 节省 **4,248** tokens；保持 enable_executor_dropout=True。
E2E 缓存节省约 **1,664** tokens（关闭 E2E 时更高）。
热记忆 vs 冷启动：warm 省 **4,811** tokens，E2E 多 **1** 次。

## 建议代码动作

- 保持 Executor Dropout（threshold=0.70）
- 考虑 bge-base 或 Summarizer-E2E 扩大 E2E 命中（当前 E2E 偏少）
- 质量未全通过：SafeSieve 已接 heuristic，可接 LLM judge 反馈
