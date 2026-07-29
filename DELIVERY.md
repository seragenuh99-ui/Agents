# 702solver 交付说明

本交付包面向 openEuler 检查环境，主分支已清理为比赛交付版，包含源码、测试、实验脚本、关键文档、答辩材料和已适配的 openEuler 可执行演示程序。

## 直接运行 openEuler 演示

```bash
cd dist/702solver_full24_demo_openeuler
./702solver_full24_demo
```

命令行冒烟验证：

```bash
dist/702solver_full24_demo_openeuler/702solver_full24_demo --help
```

该程序在 `openeuler/openeuler:24.03` 环境中构建，并已用同版本基础镜像验证。

## 从源码运行

```bash
pip install -r requirements.txt
python3 main.py demo
python3 run.py --mode 1,2 --suite full24
```

如需真实 LLM，请在本机自行创建 `.env` 并写入 API key；交付包不会包含密钥文件。

## 测试

```bash
python3 -m pytest tests/ -v
```

测试默认使用 `MockLLM` 和确定性 hash embeddings，不依赖外部 API。

## 关键材料

| 文件/目录 | 用途 |
|---|---|
| `README.md` | 项目总览和运行入口 |
| `src/` | 核心 Multi-Agent 系统源码 |
| `tests/` | 单元、集成、性能和安全测试 |
| `experiments/` | full24 等实验脚本 |
| `docs/` | 架构、实验结果、项目记录和报告 |
| `702solver_competition_results.pptx` | 最终汇报 PPT |
| `dist/702solver_full24_demo_openeuler/` | openEuler 可执行演示 bundle |
| `scripts/` | openEuler bundle 构建与依赖 vendoring 脚本 |

## 已清理内容

仓库中已移除 Codex/Claude 工作说明、旧 PPT 草稿、历史过程文档、Windows `.exe`、PPT 渲染检查残留、旧指标 JSON 和 Python 缓存等非交付必需产物；本地 `.env`、旧实验数据库、`build/`、`.pytest_cache/` 等也不应随交付包发送。
