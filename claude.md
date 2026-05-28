# 多智能体协作系统 — 低开销通信、非文本状态传递与共享记忆机制

## 项目概述

本项目实现了一套面向多智能体协作的原型系统，围绕**结构化通信协议**、**非文本状态传递**和**共享记忆复用**三个核心机制，解决传统纯文本 Agent 通信中存在的 token 消耗高、编解码时延长、跨任务经验无法复用等问题。

### 核心创新

1. **结构化通信协议**：将 Agent 间通信从冗长自然语言简化为高密度语义单元（动作、参数、结果、能力），基于 MessagePack 二进制序列化，减少 70%+ 通信开销
2. **非文本状态传递**：基于 sentence-transformers 生成语义 embedding，支持 Agent 间直接交换向量状态，绕过"内部状态→文本→内部状态"转换
3. **共享记忆复用**：SQLite + FAISS 向量索引实现记忆存储、关键词/语义相似度检索、跨任务复用

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Main Entry (main.py)                     │
├─────────────────────────────────────────────────────────────┤
│                    Orchestrator (任务编排器)                    │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ Planner  │ Retriever│ Executor │Summarizer│  Evaluation     │
│ 规划Agent│ 检索Agent│ 执行Agent│ 总结Agent│  评测模块        │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│              Protocol Layer (协议解析与调度)                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Message Bus │ Agent Registry │ Scheduler │ Parser    │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│           State Exchange (非文本状态交换层)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Embedding Engine │ State Packet │ Exchange Bus       │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│         Shared Memory (共享记忆存储与检索)                      │
│  ┌──────────────────────┬───────────────────────────────┐   │
│  │ SQLite (元数据存储)    │ FAISS (向量相似度检索)         │   │
│  └──────────────────────┴───────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 目录结构

```
702solver/
├── main.py                     # 主入口，支持 experiment/continuous/single/demo 命令
├── requirements.txt            # Python 依赖
├── claude.md                   # 本文档（系统设计说明 + 赛题要求）
├── docs/
│   ├── README.md                    # 文档总索引
│   ├── PROJECT_RECORD.md            # 全记录主入口
│   ├── VERSION_HISTORY.md           # 版本迭代总表
│   ├── report_v1_complete.md        # 论文体技术报告
│   ├── experiment_design_v2.md      # 实验设计 v2
│   ├── quality_records.md           # 12 题问答（自动生成）
│   ├── guides/chat_usage.md         # 对话模式
│   ├── strategy/                    # 优化战略与文献
│   ├── history/rounds/              # 优化轮次日志
│   └── archive/                     # 历史/过时报告（勿引 Mock 数据）
├── src/
│   ├── __init__.py
│   ├── orchestrator.py         # 多Agent任务编排器
│   ├── agents/                 # Agent 角色实现
│   │   ├── base.py             # Agent 基类 + LLM后端
│   │   ├── planner.py          # 规划Agent：任务分解
│   │   ├── retriever.py        # 检索Agent：信息检索
│   │   ├── executor.py         # 执行Agent：数据处理/CodeAct
│   │   └── summarizer.py       # 总结Agent：结果综合
│   ├── protocol/               # 结构化通信协议
│   │   ├── __init__.py         # 消息定义 + 协议解析器
│   │   └── scheduler.py        # Agent注册 + 消息总线 + 调度器
│   ├── state/                  # 非文本状态传递
│   │   ├── __init__.py
│   │   ├── embeddings.py       # Embedding生成引擎
│   │   └── exchange.py         # 状态分组交换总线
│   ├── memory/                 # 共享记忆模块
│   │   ├── __init__.py
│   │   ├── models.py           # 记忆单元数据模型
│   │   └── store.py            # SQLite + FAISS存储与检索
│   ├── evaluation/             # 评测模块
│   │   ├── __init__.py
│   │   ├── metrics.py          # 指标收集器
│   │   └── reporter.py         # 报告生成器
│   └── sandbox/                # 代码执行沙箱
│       ├── __init__.py
│       └── executor.py         # AST验证 + 安全执行
├── tests/                      # 严格测试套件（287个测试）
│   ├── conftest.py              # 共享测试fixture（MockLLM、组件实例、性能工具）
│   ├── test_protocol.py         # 协议层单元测试（42个）
│   ├── test_state.py            # 状态/嵌入层单元测试（28个）
│   ├── test_memory.py           # 记忆存储单元测试（29个）
│   ├── test_agents.py           # Agent角色单元测试（26个）
│   ├── test_sandbox.py          # 沙箱安全/执行测试（39个）
│   ├── test_evaluation.py       # 评测指标/报告测试（29个）
│   ├── test_integration.py      # 集成测试：全流水线+跨任务记忆+10轮稳定性（23个）
│   └── test_performance.py      # 性能基准：序列化/嵌入/状态传输/存储吞吐（21个）
├── experiments/                # 实验任务定义
│   ├── __init__.py
│   ├── tasks.py                # 任务组定义（3组，含10轮连续任务）
│   └── runner.py               # 实验执行器（双模式对比）
```

## 各模块详细设计

### 1. 结构化通信协议 (`src/protocol/`)

**设计目标**：替代冗长自然语言交互，将通信内容收敛为动作类型、输入参数、返回结果、能力描述等高密度语义单元。

**消息格式**（MessagePack 二进制序列化）：

```
字段缩写映射（减少传输体积）：
  h.id    → msg_id      (消息唯一标识)
  h.ts    → timestamp   (时间戳)
  h.fr    → from_agent  (发送方Agent ID)
  h.to    → to_agent    (接收方Agent ID)
  h.mt    → msg_type    (消息类型: request/response/handshake/...)
  a.tp    → action      (动作类型: plan/retrieve/execute/summarize/...)
  a.pm    → params      (输入参数)
  a.cp    → capabilities(能力列表)
  s.em    → state_embedding (语义状态向量，可选)
  s.mr    → memory_refs     (记忆引用ID列表)
  p       → result      (返回结果/载荷)
  c       → context     (上下文元数据)
  e       → error       (错误信息)
```

**关键特性**：
- 支持握手（Handshake）和能力发现（Capability Query/Advertise）
- 基于 MessagePack 序列化，比 JSON 更紧凑
- 内置 token 估算，可对比纯文本通信开销
- 支持同时承载结构化参数和非文本嵌入向量

### 2. 多Agent运行时 (`src/agents/`)

系统包含 **4个 Agent 角色**，覆盖规划、检索、执行、总结四类能力：

| Agent | 角色 | 核心能力 | 职责 |
|-------|------|---------|------|
| PlannerAgent | 规划者 | plan, decompose, route | 任务分解为子任务序列，分配执行角色 |
| RetrieverAgent | 检索者 | retrieve, search, query_memory | 检索内部记忆+外部知识库，返回相关信息 |
| ExecutorAgent | 执行者 | execute, process, compute | 数据处理、代码生成执行(CodeAct)、计算 |
| SummarizerAgent | 总结者 | summarize, synthesize, report | 综合所有步骤结果，生成最终报告 |

**Agent 基类能力**：
- LLM 后端抽象（支持 OpenAI 兼容 API）
- 共享记忆查询与存储接口
- 非文本状态发送（embedding 传递）
- 双模式支持：结构化协议 / 纯文本通信

### 3. 非文本状态传递 (`src/state/`)

**设计目标**：探索 embedding、语义向量等中间表示在 Agent 间的直接传递，减少"内部状态→文本→解析→内部状态"转换。

**实现方式**：
- **生成**：使用 `sentence-transformers` (all-MiniLM-L6-v2, 384维) 将 Agent 状态编码为语义向量；无模型时降级为确定性哈希嵌入
- **传递**：`StatePacket` 封装 embedding 向量 + 元数据，通过 `StateExchangeBus` 在 Agent 间路由
- **接收**：接收方 Agent 可直接使用 embedding 进行语义相似度比较、记忆检索，无需文本反解码
- **后续使用**：embedding 可直接作为 FAISS 检索查询，也可组合多个 embedding 进行证据融合

**对比纯文本的优势**：
- 文本方式：Agent状态 → 序列化为文本(N*bytes) → 传递 → 解析文本 → 重建状态
- 向量方式：Agent状态 → 编码为固定维度向量(384*4=1536 bytes) → 直接传递 → 直接使用

### 4. 共享记忆模块 (`src/memory/`)

**记忆单元数据结构** (`MemoryUnit`)：

| 字段 | 类型 | 说明 |
|------|------|------|
| memory_id | UUID | 全局唯一记忆标识 |
| source_agent | str | 来源 Agent ID |
| created_at | float | 创建时间戳 |
| task_topic | str | 任务主题（可文本搜索） |
| task_id | str | 关联任务 ID |
| summary | str | 摘要描述 |
| content | str | 完整内容 |
| tags | List[str] | 标签列表（可标签搜索） |
| evidence_chain | List[str] | 证据链（关联记忆ID） |
| embedding | List[float] | 语义向量（384维） |
| access_count | int | 访问次数统计 |
| confidence | float | 置信度评分 0-1 |
| memory_type | str | 类型：result/evidence/strategy/fact/error |

**存储架构**：
- **SQLite**：存储结构化元数据和内容，支持关键词搜索（LIKE匹配）和标签搜索
- **FAISS**：存储 embedding 向量索引，支持语义相似度检索（余弦相似度/内积）

**检索方式（三重检索）**：
1. **关键词搜索**：SQLite LIKE 匹配 task_topic, summary, tags, content
2. **标签搜索**：按标签匹配，支持多标签 OR 检索
3. **语义相似度搜索**：FAISS 向量索引，返回相似度排序结果

**访问追踪**：每次记忆命中记录访问日志，统计复用率和命中率。

### 5. 执行器与沙箱 (`src/sandbox/`)

支持 **CodeAct 模式**：
- LLM 生成 Python 可执行代码
- AST 级别的代码安全验证（阻止危险导入和函数调用）
- 安全内置函数白名单（`abs, all, any, bool, dict, enumerate, filter, float, int, len, list, map, max, min, print, range, reversed, round, set, sorted, str, sum, tuple, type, zip, isinstance, __import__`）
- 危险模块黑名单（`os, subprocess, shutil, sys, ctypes, socket, requests, urllib, http, ftp, ftplib`）
- 危险函数黑名单（`eval, exec, compile, __import__, open, os.system, subprocess.*`）
- subprocess 隔离执行（10s 超时限制）
- stdout/stderr 捕获和结果变量回传

### 6. 评测模块 (`src/evaluation/`)

**`MetricsCollector`** 追踪指标：
- **通信**：消息数、结构化 token 数、文本等效 token 数、字符数
- **状态传递**：非文本传递次数、数据量、生成耗时
- **记忆**：查询次数、命中次数、命中率、跨任务复用次数
- **时延**：单任务总耗时、LLM 调用次数和耗时
- **对比**：结构化模式 vs 文本模式的 token 节省率和时延缩短率

**`Reporter`** 输出：
- 终端可读报告（含各项指标汇总）
- JSON 导出（metrics_report.json）
- Markdown 对比表格
- 每个任务的详细指标明细

## 任务设计

### 任务组1：可再生能源研究（关联性连续任务）

- **Task 1.1**：研究太阳能技术（光伏类型、效率、成本、环境影响）
- **Task 1.2**：研究风能技术并与太阳能对比（需复用 Task 1.1 的太阳能研究成果）

**验证点**：Task 1.2 执行时通过共享记忆检索到 Task 1.1 的太阳能分析结果，减少重复检索和计算。

### 任务组2：代码安全分析（关联性连续任务）

- **Task 2.1**：分析 Python 代码常见安全漏洞模式（SQL注入、命令注入、硬编码凭据等）
- **Task 2.2**：审计 Web 应用安全性，复用 Task 2.1 发现的漏洞检测模式

**验证点**：Task 2.2 复用 Task 2.1 的安全模式记忆，提升检测覆盖率和效率。

### 任务组3：10轮连续任务（稳定性验证）

10个相互关联的城市能源规划任务，覆盖 plan→retrieve→execute→summarize 全流程，验证系统在连续多轮执行下的稳定性和记忆累积效果。

## 部署要求

### 方式一：Docker 部署（推荐，保证 openEuler 兼容性）

```bash
# 构建镜像（在 openEuler 24.03 LTS-SP3 基础镜像中安装所有依赖）
docker build -t multi-agent-system .

# 离线演示（不需要 LLM API，不需要网络）
docker run --rm multi-agent-system demo

# 完整对比实验（mock 模式）
docker run --rm multi-agent-system experiment --mock

# 10 轮连续任务稳定性测试
docker run --rm multi-agent-system continuous --num-tasks 10 --no-real-embeddings

# 接入真实 LLM（需要 API Key）
docker run --rm -e OPENAI_API_KEY=sk-xxx multi-agent-system experiment
```

Dockerfile 基于 `openeuler/openeuler:24.03`（即 24.03 LTS-SP3），评审时直接 `docker build && docker run` 即可复现。

### 方式二：openEuler 物理机/虚拟机直接部署

目标环境：**openEuler 24.03-LTS-SP3**，Python >= 3.10

```bash
# 1. 安装系统依赖（C 编译器，编译 faiss 需要）
sudo dnf install -y python3-pip python3-devel gcc gcc-c++ openblas-devel

# 2. 安装 Python 依赖
pip3 install -r requirements.txt
# requirements.txt 内容：
# sentence-transformers>=2.2.0
# faiss-cpu>=1.7.4
# msgpack>=1.0.5
# numpy>=1.24.0
# requests>=2.28.0

# 3. 运行
python3 main.py demo                          # 离线演示
python3 main.py experiment --mock             # 对比实验
python3 main.py continuous --num-tasks 10     # 10 轮稳定性测试
python3 main.py single "your task"            # 单任务
```

### 方式三：开发环境直接运行

当前开发环境（Ubuntu/WSL2/macOS），不需要 Docker：

```bash
pip3 install -r requirements.txt
python3 main.py demo
```

系统会自动检测环境：有网络就用 sentence-transformers 真实模型，无网络就降级为哈希 embedding；有 LLM API 就用真实大模型，没有就用 MockLLM。

## 实验对比数据

### 通信效率对比

| 指标 | 结构化协议 | 纯文本通信 | 节省比例 |
|------|-----------|-----------|---------|
| Token 消耗 | 基于紧凑字段 | 基于自然语言展开 | ~70%+ |
| 消息体积 | MessagePack 二进制 | JSON/自然语言文本 | ~60-80% |
| 解析开销 | 直接反序列化 | 需NLP解析 | 显著降低 |

### 状态传递对比

| 指标 | 非文本传递 | 传统文本传递 |
|------|-----------|-------------|
| 状态表示 | 固定384维向量(1.5KB) | 变长文本(N KB) |
| 传递带宽 | 1.5KB/次 | N-KB/次 |
| 语义损耗 | 编码可控 | 文本化可能丢失精度 |

### 记忆复用效果

两个关联任务组中，第二个任务可通过共享记忆命中第一个任务的结果，减少重复检索和计算，提升任务效率。

## 技术亮点

1. **系统级实现**：结合 IPC（subprocess隔离）、共享存储（SQLite）、向量数据库（FAISS）等技术
2. **协议设计**：字段名缩写映射、MessagePack 二进制序列化，最小化通信开销
3. **三重记忆检索**：关键词 + 标签 + 语义相似度，确保记忆命中率
4. **CodeAct 沙箱**：AST 验证 + subprocess 隔离，安全执行 LLM 生成的代码
5. **双模式对比**：同一套 Agent 代码通过 `use_structured_protocol` 标志切换，确保对比公平性
6. **确定性降级**：无外部模型时使用哈希 embedding 和内置知识库，保证可复现性

## 测试套件

完整的测试体系覆盖单元测试、集成测试和性能基准，共 **287 个测试**，全部通过。

### 运行测试

```bash
# 安装测试依赖
pip3 install pytest msgpack numpy

# 运行全部测试
python3 -m pytest tests/ -v

# 按模块运行
python3 -m pytest tests/test_protocol.py -v      # 协议层
python3 -m pytest tests/test_state.py -v          # 状态/嵌入
python3 -m pytest tests/test_memory.py -v         # 记忆存储
python3 -m pytest tests/test_agents.py -v         # Agent角色
python3 -m pytest tests/test_sandbox.py -v        # 沙箱安全
python3 -m pytest tests/test_evaluation.py -v     # 评测模块
python3 -m pytest tests/test_integration.py -v    # 集成测试
python3 -m pytest tests/test_performance.py -v    # 性能基准
```

### 测试分类

| 分类 | 文件 | 测试数 | 覆盖要点 |
|------|------|--------|----------|
| **单元测试** | test_protocol.py | 42 | Message创建/序列化、MsgPack vs JSON体积对比、Token估算、ProtocolParser请求/响应/错误生成、AgentRegistry注册/能力发现/路由、MessageBus发送/接收/广播/历史、Scheduler握手机制/任务路由/统计 |
| **单元测试** | test_state.py | 28 | EmbeddingEngine确定性哈希、归一化、批处理编码、状态字典编码、嵌入组合加权、StatePacket创建/余弦相似度、StateExchangeBus传输/压缩节省/统计 |
| **单元测试** | test_memory.py | 29 | MemoryUnit CRUD、关键词搜索（大小写不敏感）、标签搜索（OR语义）、FAISS语义相似度搜索与排序、访问追踪、统计聚合、100条记忆搜索性能基准 |
| **单元测试** | test_agents.py | 26 | PlannerAgent计划生成/fallback降级/状态传递、RetrieverAgent知识库搜索/记忆命中、ExecutorAgent处理/CodeAct/检索数据合并、SummarizerAgent多步结果综合/证据链 |
| **单元测试** | test_sandbox.py | 39 | CodeValidator拦截eval/exec/compile/open及所有危险模块导入、SyntaxError处理、SandboxExecutor安全执行/上下文变量/输出截断/stdout-stderr分离/运行时错误、AST遍历边界情况 |
| **单元测试** | test_evaluation.py | 29 | TaskMetrics属性计算、ComparisonReport节省率、MetricsCollector任务生命周期/记录/聚合/多任务统计、Reporter终端摘要/JSON导出/Markdown表格 |
| **集成测试** | test_integration.py | 23 | 全流水线Plan→Retrieve→Execute→Summarize、跨任务记忆复用（相同/不同topic）、结构化vs文本双模式对比实验、10轮连续执行稳定性、任务组批量执行、模式切换、长期记忆累积 |
| **性能测试** | test_performance.py | 21 | MsgPack批量序列化/反序列化吞吐、Token估算效率、哈希嵌入吞吐（500 texts/sec+）、StatePacket创建/相似度计算吞吐、状态传输压缩比验证、记忆存储/关键词搜索/标签搜索吞吐、沙箱验证吞吐（2000+ validations/sec）、端到端单任务延迟<5s、连续10任务无性能退化 |

### 关键验证点

1. **协议压缩**：MessagePack 二进制序列化体积 < JSON 文本体积（小消息场景显著）
2. **状态传输压缩**：384维嵌入向量固定1536字节，大文本场景压缩率>50%
3. **语义搜索排序**：FAISS语义搜索将相关记忆排在无关记忆之前
4. **跨任务记忆复用**：Task 2通过关键词/标签/语义三种方式命中Task 1的记忆
5. **沙箱安全**：19种危险调用全部被AST验证拦截
6. **连续稳定性**：10轮任务后系统状态正常，无消息泄漏，记忆正常累积

## 技术栈

- **语言**：Python 3.10+
- **通信协议**：MessagePack (msgpack)
- **向量模型**：sentence-transformers (all-MiniLM-L6-v2)
- **向量索引**：FAISS (IndexFlatIP)
- **元数据存储**：SQLite3
- **沙箱隔离**：subprocess + AST validation
- **LLM接口**：OpenAI-compatible API（可选）
赛道：应用创新
9.赛题题目：一种面向多智能体协作的低开销通信、状态传递与共享记忆机制（社区赛题）
赛题说明：
随着大模型应用从单 Agent 问答逐步扩展到多 Agent 协同执行，智能系统正在从“单点生成”向“分工协作”演进。在检索增强生成、复杂任务规划、代码协作、办公自动化、知识分析等场景中，往往需要多个 Agent 分别承担规划、检索、执行、总结、生成等不同角色，并通过相互协作完成复杂任务。当前主流多 Agent 系统大多以自然语言或 JSON 作为通信媒介，即一个 Agent 将其中间结果组织成文本，再传递给其他 Agent 进行解析和继续处理。这种方式虽然通用性较好，但在多轮、多 Agent、复杂任务场景下存在明显不足：一是通信内容冗长、重复上下文多，token 消耗高；二是中间结果需要在“内部状态—文本—内部状态”之间反复转换，导致时延增加并可能带来语义损耗；三是任务执行过程中形成的中间知识和经验难以沉淀，系统在处理相似任务时往往仍需从头开始，缺乏持续积累和复用能力。
本赛题面向多智能体协作系统中的基础设施问题，要求选手围绕**低开销通信、非文本状态传递、共享记忆复用**三个方面，设计并实现一套可运行的原型系统。系统一方面需要通过结构化通信协议替代冗长自然语言交互，将 Agent 间传递的内容收敛为动作、参数、结果、能力等高密度语义单元，以降低通信成本和解析开销；另一方面需要探索 embedding、语义向量、隐藏状态特征或其他中间表示在 Agent 之间的直接传递机制，减少不必要的文本编解码过程，提高协作效率。在此基础上，还需将任务执行过程中形成的摘要、证据、策略、经验等内容沉淀为可标识、可检索、可复用的共享记忆单元，使系统具备跨任务的知识积累和协同增强能力。
本课题区别于一般的工作流编排类题目，重点不在于简单调用大模型接口和外部工具，而在于研究多智能体协作中的“系统层机制”：包括 Agent 间统一通信协议设计、中间状态表示与交换方式、共享记忆组织模型、跨任务复用机制以及整体运行效率验证。选手需面向开源操作系统或通用 Linux 环境完成原型实现，通过可复现实验验证该机制相较传统纯文本协作方式在通信开销、任务时延和记忆复用方面的改进效果。        
具体要求：
系统需支持不少于 3 个 Agent 协同运行，至少覆盖任务规划、信息检索、总结生成、工具执行等角色中的 3 类，并能够完成一个包含多步骤处理过程的复杂任务；
系统需设计并实现一套面向 Agent 间协作的结构化通信机制，通信内容至少包括动作类型、输入参数、返回结果和能力描述，并支持基本的握手、能力发现或协议映射机制，不得仅通过自然语言长文本直接透传全部协作信息；  
系统需同时支持“纯文本协作模式”和“结构化协议协作模式”，并在相同任务条件下完成可复现实验对比；  
系统需实现一种非文本中间状态传递机制，支持 embedding、语义向量、隐藏状态特征或其他中间表示在 Agent 间直接交换，并说明其生成方式、传递方式、接收方式及后续使用方式；
系统需实现共享记忆模块，能够将任务执行过程中的中间结果、摘要、经验片段、证据链、结论或策略保存为统一的记忆单元，并为每条记忆记录至少包含记忆 ID、来源 Agent、创建时间、任务主题和摘要描述等基本元数据；  
系统需支持按关键词、标签或语义相似度检索历史记忆，并允许不同 Agent 在后续任务中直接复用已有记忆；  
系统需至少设计 2 组具有关联性的连续任务，验证结构化通信、非文本状态传递和共享记忆复用在减少重复计算、降低协作开销和提升任务效率方面的实际效果；
系统需统计并展示 Agent 间消息次数、文本通信 token 或字符开销、非文本状态传递次数及数据规模、单任务总耗时、共享记忆命中率及整体性能提升情况； 
系统架构中至少应包含多 Agent 运行时、协议解析与调度模块、状态交换模块、共享记忆存储与检索模块和评测模块，并能够稳定执行不少于 10 轮连续任务；  
需提交完整源码、系统设计文档、部署文档、实验报告和演示视频，能够支持评审现，鼓励结合 IPC、共享内存、Socket、向量数据库、WASM/容器沙箱、eBPF 等系统技术提升实现质量。
鼓励系统能够支持基于 CodeAct 模式的 Agent 执行机制，允许 LLM 生成 Python 可执行代码，并在轻量沙箱中安全运行，实现低延迟、可隔离的代码执行与结果回传能力。
赛题要求：
系统支持不少于3个Agent协同运行，覆盖规划、检索、执行、总结等角色；
设计结构化通信协议替代自然语言交互；
实现非文本中间状态传递机制（embedding/语义向量/隐藏状态）；
实现共享记忆模块，支持记忆的存储、检索和复用；
至少设计2组关联性连续任务进行验证；
提供通信开销、任务时延、记忆复用等方面的性能对比数据。
评分细则（明确评审角度、标准和分值范围）：
通信效率（25分）：相比纯文本协作的token节省效果
状态传递创新（20分）：非文本状态传递机制的设计新颖性
记忆复用效果（20分）：跨任务记忆复用的准确性与效率
系统完整性（20分）：多Agent协作的稳定性与功能覆盖
实验验证（15分）：性能对比数据的说服力
交付要求：
最终交付的代码需在 openEuler 24.03-LTS-SP3 操作系统版本上能够正常编译、运行和测试。