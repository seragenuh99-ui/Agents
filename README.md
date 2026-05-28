# 702solver

多 Agent 协作系统：低开销通信 · 非文本状态传递 · 共享记忆复用。

| 入口 | 说明 |
|------|------|
| **`./启动实验.sh`** 或双击 **`启动实验.bat`**（Windows） | **实验汇报入口**（控制台 / 答辩快测 / 对照） |
| **`./启动实验_答辩12题.sh`** | **一键 core12 · 模式 1+2 对照** |
| **`python3 run.py`** | **实验控制台（benchmark / 演示 / 切换模式）** |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | **系统架构 + 课程要求对照** |
| **[docs/STATUS_SUMMARY.md](docs/STATUS_SUMMARY.md)** | **目前情况一页总览（最新数据）** |
| **[docs/PROJECT_RECORD.md](docs/PROJECT_RECORD.md)** | 全记录：指标、实验、对话、产物 |
| **[docs/CURRENT_STATUS_AND_ROADMAP.md](docs/CURRENT_STATUS_AND_ROADMAP.md)** | 通俗解读 + 后续优化路线 |
| **[docs/full24_benchmark_results.md](docs/full24_benchmark_results.md)** | 24 题标准基准 |
| **[docs/README.md](docs/README.md)** | 文档目录 |
| **[experiments/README.md](experiments/README.md)** | 实验脚本说明 |
| **[output/README.md](output/README.md)** | 运行时产物（JSON / 日志 / 数据库） |

## 仓库结构

```
702solver/
├── main.py              # CLI：experiment / single / chat / demo / stats
├── chat.py              # 交互对话（推荐入口）
├── src/                 # 核心代码
│   ├── agents/          # Planner, Retriever, Executor, Summarizer
│   ├── orchestrator.py
│   ├── memory/          # SQLite + FAISS
│   ├── protocol/        # MessagePack 调度
│   ├── state/           # BGE 向量状态
│   ├── chat/            # 对话会话
│   ├── evaluation/      # 指标与质量验证
│   └── paths.py         # output/ 路径常量
├── experiments/         # 基准与实验矩阵脚本
├── tests/               # 单元与集成测试
├── docs/                # 文档（报告、设计、归档）
└── output/              # 生成物（勿与源码混放）
    ├── results/         # *.json 报告
    ├── logs/            # *.log
    ├── databases/       # *.db 记忆库
    ├── experiment_matrix/
    └── experiment_matrix_v2/
```

## 快速开始

```bash
pip install -r requirements.txt
echo 'DEEPSEEK_API_KEY=你的密钥' >> .env

python3 chat.py                                    # 对话
python3 main.py single "你的任务描述"             # 单任务
python3 experiments/full24_controlled_comparison.py  # 控制变量对照（推荐）
python3 experiments/benchmark_full24.py              # 结构化子实验
```

## 核心结果（v0.11.2，full24 控制变量实验）

| 变体 | Token | 质量 |
|------|------:|------|
| 纯文本 A（同 24 题） | **95,377** | 24/24 |
| **结构化 C（全功能）** | **30,221** | **24/24** |
| 结构化 B（无缓存） | 49,540 | 24/24 |

**C vs A 省 Token：68.3%**（同条件，非折算）· 对抗题 **6/6**

数据：[docs/full24_controlled_comparison.md](docs/full24_controlled_comparison.md)
