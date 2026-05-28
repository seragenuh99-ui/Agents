# 实验设计 v2（针对不足项的完整重设计）

> **目标**：回应 v1 局限（任务少、基线弱、质量≠事实、无对抗集、实验未闭环），形成可答辩的充分实验体系。  
> **执行**：`python3 experiments/experiment_matrix_v2.py --phase all`  
> **输出**：`experiment_matrix_v2_report.json`、`docs/experiment_matrix_v2_results.md`

---

## 一、v1 不足 → v2 对策

| v1 不足 | v2 对策 | 产物 |
|---------|---------|------|
| 仅 12 题 | **core12 + extended6 + adversarial6 = full24** | `benchmark_suites.py` |
| 无强基线 | **单 Agent 长链** + 既有纯文本 A | `single_agent_baseline.py` |
| 质量≠事实 | **禁止词检测** + **LLM judge 子实验** + **人工 6 题包** | `quality_validator` + `factcheck_export.py` |
| E2E 难验证 | **对抗集**（相似表述、不同意图，`no-e2e-cache`） | `adversarial6` + `adversarial_e2e_safe` 指标 |
| 热/冷记忆说不清 | **C_warm vs C_cold** 同 core12 | matrix v2 `core` 阶段 |
| 消融不完整 | **B / no_dropout / no_e2e** on core12 | matrix v2 |
| 单模型 | 保留 DeepSeek；**EMBEDDING_MODEL** 可换 bge-base（可选） | 环境变量 |
| 无工业框架对照 | 文档标明 scope；单 Agent 作「必要性」代理基线 | 设计说明 |

---

## 二、任务集设计

### 2.1 core12（与 v1 相同，可对比历史）

能源 / 安全 / 数据库 各 4 题，见 `benchmark_tasks.py`。

### 2.2 extended6（覆盖扩展）

| ID | 意图 |
|----|------|
| e5_hydro_storage | 水电与抽蓄 |
| e6_grid_integration | 电网集成 |
| s5_secrets_management | 密钥管理（非 Web 审计） |
| s6_supply_chain | 供应链安全 |
| d5_transactions | 事务隔离级别 |
| d6_backup_recovery | 备份与容灾 |

### 2.3 adversarial6（缓存压力 / 误复用检测）

题目表述与 core 题**表面相似**，但：

- `must_not_e2e: true`
- tags 含 `no-e2e-cache`（Orchestrator **禁止 E2E**）
- `forbidden_answer_topics`：答案出现即判质量问题

| ID | 陷阱 |
|----|------|
| x1_solar_policy_only | 要政策，不要光伏技术 |
| x2_wind_economics_only | 要经济学，不要 DFIG/PMSG |
| x3_security_compliance_only | 要合规，不要漏洞代码 |
| x4_web_tls_only | 要 TLS，不要 CSRF/JWT |
| x5_db_migration_only | 要迁移，不要索引选型 |
| x6_db_cap_only | 要 CAP，不要 Redis/Mongo 对比 |

**跑法**：先 warm-run core12 写入记忆库，再跑 adversarial6，统计 `adversarial_false_e2e`（应为 0）。

---

## 三、对照组（完整矩阵）

| ID | 类型 | 任务集 | 说明 |
|----|------|--------|------|
| A_pure_text | 基线 | core12 | 已有 `pure_text_baseline_results.json` |
| D_single_agent | 强基线 | core12 | 单 LLM JSON 链，无多 Agent 缓存 |
| B_nocache | 消融 | core12 | 关闭 FAISS/全部缓存 |
| C_no_dropout | 消融 | core12 | 关闭 Executor Dropout |
| C_no_e2e | 消融 | core12 | 关闭 E2E |
| C_cold | 条件 | core12 | 每题清空 DB |
| C_warm | 条件 | core12 | 12 题共库 |
| C_full_core12 | **主系统** | core12 | 默认 RunOptions |
| C_full_full24 | 扩展 | full24 | 规模与领域覆盖 |
| C_adversarial6_warm | 安全 | adversarial6 | 热库后对抗 |
| C_full_core12_llm_judge | 质量 | core12 | 开启 LLM 评判 |

---

## 四、指标

### 必报

- Token（总 / prompt / completion）、API 次数  
- `avg_relevance`（BGE）  
- `strategy_distribution`、e2e_hits、dropout_hits  
- `quality_pass_count` / `composite`（heuristic，修复后含 expected_topics）  
- `save_vs_pure_text_pct`  
- **adversarial_e2e_safe**（对抗集无 false E2E）

### 微基准（无 API）

- `system_metrics.json`  
- `embedding_similarity_matrix.json`（`--suite full24`）

### 人工（抽样）

- `docs/human_factcheck_pack.md`（6 题）

---

## 五、假设检验（更新）

| 假设 | 判定 |
|------|------|
| H2 缓存主因 | `B_nocache` vs `C_full_core12` Token 差 |
| H5 Dropout | `C_no_dropout` vs `C_full` |
| H6 E2E | `C_no_e2e` vs `C_full`；对抗集 false E2E=0 |
| H4 热记忆 | `C_warm` vs `C_cold` |
| H9 多 Agent 必要？ | `C_full` vs `D_single_agent`（Token 与质量） |
| H10 规模 | `C_full_full24` 策略分布与通过率 |

---

## 六、推荐执行顺序

```bash
cd /home/chen/projects/702solver

# 全流程（约 2–3h API，可 --skip-existing 断点续跑）
python3 experiments/experiment_matrix_v2.py --phase all

# 或分阶段
python3 experiments/experiment_matrix_v2.py --phase micro      # 无 API
python3 experiments/experiment_matrix_v2.py --phase core       # 消融+单Agent
python3 experiments/experiment_matrix_v2.py --phase extended   # 24 题
python3 experiments/experiment_matrix_v2.py --phase adversarial
python3 experiments/experiment_matrix_v2.py --phase quality    # LLM judge
python3 experiments/experiment_matrix_v2.py --phase factcheck  # 人工包

# 可选：3-run 方差（core12）
python3 experiments/multi_run_benchmark.py -n 5
```

---

## 七、答辩叙事（v2）

1. **规模**：24 题含 6 道对抗题，不是只报 12 题漂亮数字。  
2. **因果**：消融证明省 Token 来自缓存与 Dropout，不是协议 alone。  
3. **安全**：对抗集 + 禁止 E2E 标签 + 禁止词检测。  
4. **可信度**：自动 + LLM judge 子实验 + 6 题人工表。  
5. **必要性**：单 Agent 基线对比多 Agent 全功能。

---

## 八、仍保留的局限（诚实）

- 检索仍为内置 KB，非真实 RAG API  
- 未做 LangGraph 自动化对照（可人工补 1 题）  
- 人工 fact-check 仅 6 题样本  
- DeepSeek 单模型为主  

---

*v2.0 | 2026-05-19*
