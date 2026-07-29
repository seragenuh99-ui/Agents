# 702solver

多 Agent 协作诊断系统：低开销结构化通信、非文本状态传递、共享记忆复用。

本仓库主分支已整理为比赛交付版，只保留源码、测试、实验脚本、关键文档、最终汇报 PPT 和 openEuler 可执行演示 bundle。

## 快速入口

| 入口 | 说明 |
|------|------|
| `DELIVERY.md` | 交付说明和检查环境运行方法 |
| `dist/702solver_full24_demo_openeuler/702solver_full24_demo` | openEuler 可执行演示程序 |
| `702solver_competition_results.pptx` | 最终汇报 PPT |
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

命令行验证：

```bash
dist/702solver_full24_demo_openeuler/702solver_full24_demo --help
```

该 bundle 在 `openeuler/openeuler:24.03` 容器中构建，并已在同版本基础镜像中通过验证。

如需重新生成 openEuler 版可执行目录：

```bash
docker run --rm -v "$PWD:/app" -w /app openeuler/openeuler:24.03 \
  bash scripts/build_openeuler_demo_bundle.sh

docker run --rm -v "$PWD:/app" -w /app openeuler/openeuler:24.03 \
  bash scripts/vendor_openeuler_gui_libs.sh
```

## 环境与依赖

```bash
pip install -r requirements.txt
```

如需真实 LLM，请自行创建 `.env` 并写入 API key；交付仓库不包含密钥文件。测试默认使用 `MockLLM` 和确定性 hash embeddings，不依赖外部 API。

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
├── README.md
├── DELIVERY.md
├── 702solver_competition_results.pptx
├── main.py
├── run.py
├── chat.py
├── demo_dashboard.py
├── src/
├── tests/
├── experiments/
├── docs/
├── scripts/
└── dist/702solver_full24_demo_openeuler/
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
| [docs/PROJECT_RECORD.md](docs/PROJECT_RECORD.md) | 指标、实验和产物记录 |
| [docs/final_report.md](docs/final_report.md) | 最终报告 |
| [docs/full24_controlled_comparison.md](docs/full24_controlled_comparison.md) | 24 题控制变量实验 |
| [experiments/README.md](experiments/README.md) | 实验脚本说明 |
