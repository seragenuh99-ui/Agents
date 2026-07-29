# 702solver

多 Agent 协作诊断系统：低开销结构化通信、非文本状态传递、共享记忆复用。

## 快速入口

| 入口 | 说明 |
|------|------|
| `./启动实验.sh` 或双击 `启动实验.bat` | 实验汇报入口，包含控制台、答辩快测和对照实验 |
| `./启动实验_答辩12题.sh` | 一键运行 core12，模式 1+2 对照 |
| `python3 run.py` | 实验控制台，支持 benchmark、演示和模式切换 |
| `python3 main.py demo` | 离线 mock 演示，无需 API key |
| `python3 main.py experiment --mock` | mock LLM 完整对比实验 |
| `python3 chat.py` | 交互式对话会话 |

## openEuler 可执行版本

本仓库已提供面向检查环境的 openEuler 24.03 LTS-SP3 版演示程序：

```bash
cd dist/702solver_full24_demo_openeuler
./702solver_full24_demo
```

该 bundle 在 `openeuler/openeuler:24.03` 容器中构建，并已在干净 openEuler 24.03 基础镜像中通过 `--help` 冒烟验证：

```bash
docker run --rm -v "$PWD:/app" -w /app openeuler/openeuler:24.03 \
  dist/702solver_full24_demo_openeuler/702solver_full24_demo --help
```

如需重新生成 openEuler 版可执行目录：

```bash
docker run --rm -v "$PWD:/app" -w /app openeuler/openeuler:24.03 \
  bash scripts/build_openeuler_demo_bundle.sh

docker run --rm -v "$PWD:/app" -w /app openeuler/openeuler:24.03 \
  bash scripts/vendor_openeuler_gui_libs.sh
```

说明：之前的 `dist/702solver_full24_demo.exe` 是 Windows 可执行文件；openEuler/Linux 检查环境请使用 `dist/702solver_full24_demo_openeuler/` 目录中的无后缀 ELF 启动文件。

## 环境与依赖

```bash
pip install -r requirements.txt
echo 'DEEPSEEK_API_KEY=你的密钥' >> .env
```

测试不需要真实 LLM API key，默认使用 `MockLLM` 和确定性 hash embeddings。

## 常用命令

```bash
python3 run.py
python3 run.py --mode 2 --suite core12
python3 run.py --mode 1,2 --suite full24
python3 run.py --mode 2 --ask "你的问题"

python3 main.py demo
python3 main.py experiment --mock
python3 main.py single "任务描述"
python3 chat.py

python3 -m pytest tests/ -v
python3 -m pytest tests/ -x --tb=short
```

## 仓库结构

```text
702solver/
├── main.py                  # CLI：experiment / single / chat / demo / stats
├── run.py                   # 实验控制台入口
├── chat.py                  # 交互式对话
├── demo_dashboard.py        # Qt 演示仪表盘
├── src/                     # 核心代码
│   ├── agents/              # Planner / Retriever / Executor / Summarizer
│   ├── memory/              # SQLite + FAISS 共享记忆
│   ├── protocol/            # MessagePack 结构化协议
│   ├── state/               # embedding 状态传递
│   ├── evaluation/          # 指标与质量验证
│   └── sandbox/             # AST 校验 + 安全执行
├── experiments/             # 基准与对照实验脚本
├── tests/                   # 单元与集成测试
├── docs/                    # 架构、报告、项目记录
├── dist/                    # 可执行交付物
└── output/                  # 运行时结果、日志和数据库
```

## 核心结果

full24 控制变量实验结果：

| 变体 | Token | 质量 |
|------|------:|------|
| 纯文本 A，同 24 题 | 95,377 | 24/24 |
| 结构化 C，全功能 | 30,221 | 24/24 |
| 结构化 B，无缓存 | 49,540 | 24/24 |

结构化全功能 C 相比纯文本 A 节省 Token：68.3%，对抗题通过 6/6。

详细数据见 [docs/full24_controlled_comparison.md](docs/full24_controlled_comparison.md)。

## 关键文档

| 文档 | 说明 |
|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 系统架构与课程要求对照 |
| [docs/STATUS_SUMMARY.md](docs/STATUS_SUMMARY.md) | 当前状态总览 |
| [docs/PROJECT_RECORD.md](docs/PROJECT_RECORD.md) | 指标、实验、对话和产物记录 |
| [docs/CURRENT_STATUS_AND_ROADMAP.md](docs/CURRENT_STATUS_AND_ROADMAP.md) | 通俗解读与后续路线 |
| [experiments/README.md](experiments/README.md) | 实验脚本说明 |
