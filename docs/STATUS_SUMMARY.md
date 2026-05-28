# 702solver 目前情况总结

> **更新**：2026-05-21（**v0.11.2 full24 复测：24/24**）  
> **代码版本**：v0.11.2  
> **主数据**：[full24_controlled_comparison.md](full24_controlled_comparison.md)

---

## 1. 系统在做什么

多 Agent + 共享记忆，能复用就复用；v0.11.2 意图门控区分对比/合规/漏洞，防止套错模板。

---

## 2. 标准实验（控制变量，同一 full24、同一 deepseek-chat）

**命令**：`python3 experiments/full24_controlled_comparison.py`（可加 `--skip-pure-text` 复用已有纯文本 A）

| 变体 | 条件 | Token | 质量 | 相关度 | 墙钟 | 均题耗时 |
|------|------|------:|------|-------:|-----:|---------:|
| **A 纯文本** | 3 次 LLM/题，长 prompt，无记忆 | **95,377** | **24/24** | **0.868** | 928s | 38.7s |
| **C 结构化全功能** | 记忆+模板+Dropout（v0.11.2） | **30,221** | **24/24** | **0.841** | 209s | 8.7s |
| **B 结构化无缓存** | 关全部缓存 | 49,540 | 24/24 | 0.843 | 292s | 12.2s |

### Token 节省（同条件，非折算）

| 对比 | 节省 |
|------|------|
| **C vs A（答辩主数字）** | **68.31%** |
| C vs B（记忆贡献） | 39.0% |
| B vs A（仅协议） | 48.1% |

### 其它

- 对抗 6 题：误用 E2E **0/6**；对抗集质量 **6/6**
- 旧 12 题纯文本 A：**49,076**（题数不同勿混比）

---

## 3. 文件索引

| 文件 | 内容 |
|------|------|
| [full24_controlled_comparison.md](full24_controlled_comparison.md) | **控制变量对照表（含 A/B/C 逐题）** |
| `output/results/full24_controlled_comparison.json` | 机器可读 |
| `output/results/structured_nocache_full24.json` | B 无缓存逐题 |
| [optimization_v0.11.2_compliance_relevance.md](history/rounds/optimization_v0.11.2_compliance_relevance.md) | 本轮优化说明 |

---

## 4. 还没做

- bge-base 正式对照（须清库）
- 人工 6 题 factcheck 填表

---

*勿再用「49k×2 折算」当 24 题纯文本基线；以 **95,377** 为准。*
