# full24 对照实验（控制变量）

- **时间**: 2026-05-21T02:32:13Z
- **代码**: v0.11.2
- **控制**: 同一任务集 `full24`（24 题）、同一模型 `deepseek-chat`、真 API
- **命令**: `python3 experiments/full24_controlled_comparison.py`

## 实验设计

| 变体 | 条件 |
|------|------|
| **A 纯文本** | 每题 3 次 LLM（规划+检索+综合），长 prompt、散文输出、**无记忆、无结构化协议** |
| **C 结构化全功能** | Orchestrator + 记忆 + 模板/E2E/Dropout（v0.11.2） |
| **B 结构化无缓存** | 同 C，但关闭全部缓存与 Dropout |

## 主表（同条件对比）

| 变体 | Token | API 调用 | 平均相关度 | 质量通过 | 墙钟总耗时 | 平均每题 |
|------|------:|---------:|----------:|---------:|-----------:|---------:|
| A_pure_text_full24 | 95,377 | 72 | 0.868 | 24/24 | 928.1s | 38.67s |
| C_structured_full24 | 30,221 | 52 | 0.841 | 24/24 | 208.5s | 8.69s |
| B_nocache_full24 | 49,540 | 75 | 0.843 | 24/24 | 292.1s | 12.17s |

## Token 节省（相对对照）

| 对比 | 节省 Token | 说明 |
|------|----------:|------|
| C vs A（主结论） | **68.31%** | 95,377 → 30,221 |
| C vs B | **39.0%** | 记忆+剪枝相对无缓存 |
| B vs A | 48.06% | 仅结构化协议、无记忆 |

## 准确性

- **A_pure_text_full24**: 质量 24/24，相关度 0.868
- **C_structured_full24**: 质量 24/24，相关度 0.841
- **B_nocache_full24**: 质量 24/24，相关度 0.843

## 对抗子集（C 结构化，热记忆后 6 题）

- 误用 E2E: 0/6
- 对抗质量通过: 6/6
- 安全: True

## 逐题对比（A / B / C）

| 题号 | A Token | B Token | C Token | C省% | A相关 | C相关 | A | B | C | C策略 |
|------|--------:|--------:|--------:|-----:|------:|------:|:---:|:---:|:---:|------|
| `e1_solar_basics` | 3,803 | 958 | 1,024 | 73.1% | 0.816 | 0.778 | ✓ | ✓ | ✓ | FULL_GEN |
| `e2_solar_advanced` | 3,977 | 3,217 | 3,261 | 18.0% | 0.923 | 0.876 | ✓ | ✓ | ✓ | FULL_GEN |
| `e3_wind_energy` | 4,002 | 1,901 | 1,914 | 52.2% | 0.811 | 0.857 | ✓ | ✓ | ✓ | FULL_GEN |
| `e4_renewable_comparison` | 4,078 | 2,263 | 1,087 | 73.3% | 0.766 | 0.807 | ✓ | ✓ | ✓ | P1_TEMPLATE_DROP |
| `s1_python_vuln` | 4,050 | 2,447 | 1,941 | 52.1% | 0.930 | 0.887 | ✓ | ✓ | ✓ | FULL_GEN |
| `s2_web_security` | 4,037 | 2,431 | 1,079 | 73.3% | 0.930 | 0.821 | ✓ | ✓ | ✓ | TEMPLATE_FILL_DROP |
| `s3_python_vuln_v2` | 3,996 | 2,359 | 0 | 100.0% | 0.940 | 0.763 | ✓ | ✓ | ✓ | E2E_EXACT |
| `s4_code_review` | 4,001 | 1,918 | 1,038 | 74.1% | 0.841 | 0.728 | ✓ | ✓ | ✓ | TEMPLATE_FILL_DROP |
| `d1_query_optimization` | 4,028 | 1,915 | 1,185 | 70.6% | 0.876 | 0.902 | ✓ | ✓ | ✓ | FULL_GEN |
| `d2_nosql_comparison` | 4,048 | 2,111 | 1,156 | 71.4% | 0.887 | 0.903 | ✓ | ✓ | ✓ | FULL_GEN |
| `d3_db_performance` | 4,039 | 2,472 | 1,129 | 72.0% | 0.925 | 0.823 | ✓ | ✓ | ✓ | TEMPLATE_FILL_DROP |
| `d4_data_modeling` | 4,020 | 2,004 | 1,124 | 72.0% | 0.869 | 0.913 | ✓ | ✓ | ✓ | TEMPLATE_FILL_DROP |
| `e5_hydro_storage` | 4,015 | 1,956 | 1,146 | 71.5% | 0.856 | 0.874 | ✓ | ✓ | ✓ | TEMPLATE_FILL_DROP |
| `e6_grid_integration` | 3,915 | 1,108 | 1,088 | 72.2% | 0.847 | 0.819 | ✓ | ✓ | ✓ | TEMPLATE_FILL_DROP |
| `s5_secrets_management` | 3,937 | 1,107 | 1,049 | 73.4% | 0.826 | 0.924 | ✓ | ✓ | ✓ | TEMPLATE_FILL_DROP |
| `s6_supply_chain` | 3,905 | 1,140 | 1,062 | 72.8% | 0.923 | 0.841 | ✓ | ✓ | ✓ | TEMPLATE_FILL_DROP |
| `d5_transactions` | 3,902 | 1,198 | 1,086 | 72.2% | 0.882 | 0.856 | ✓ | ✓ | ✓ | TEMPLATE_FILL_DROP |
| `d6_backup_recovery` | 4,001 | 1,156 | 1,068 | 73.3% | 0.892 | 0.861 | ✓ | ✓ | ✓ | TEMPLATE_FILL_DROP |
| `x1_solar_policy_only` | 3,639 | 1,925 | 974 | 73.2% | 0.801 | 0.761 | ✓ | ✓ | ✓ | P1_TEMPLATE_DROP |
| `x2_wind_economics_only` | 3,847 | 1,635 | 1,126 | 70.7% | 0.854 | 0.821 | ✓ | ✓ | ✓ | TEMPLATE_FILL_DROP |
| `x3_security_compliance_only` | 4,063 | 4,100 | 2,415 | 40.6% | 0.863 | 0.837 | ✓ | ✓ | ✓ | FULL_GEN |
| `x4_web_tls_only` | 4,245 | 1,112 | 1,128 | 73.4% | 0.854 | 0.813 | ✓ | ✓ | ✓ | TEMPLATE_FILL_DROP |
| `x5_db_migration_only` | 3,901 | 4,210 | 1,084 | 72.2% | 0.907 | 0.833 | ✓ | ✓ | ✓ | TEMPLATE_FILL_DROP |
| `x6_db_cap_only` | 3,928 | 2,897 | 1,057 | 73.1% | 0.820 | 0.879 | ✓ | ✓ | ✓ | TEMPLATE_FILL_DROP |

