# 对话模式使用说明

> 全项目记录见 [`../PROJECT_RECORD.md`](../PROJECT_RECORD.md) · 文档索引 [`../README.md`](../README.md)。

使用 **v0.9 全功能多 Agent 流水线**（非单纯 ChatGPT 单轮），带 **跨轮共享记忆**。

## 启动

```bash
cd /home/chen/projects/702solver

# 确保 .env 中有 DEEPSEEK_API_KEY=
python3 chat.py
```

可选参数：

```bash
python3 chat.py --db my_session.db    # 指定记忆库文件（可跨次启动保留记忆）
python3 chat.py --mode text           # 文本基线模式对比
python3 chat.py --model deepseek-chat
```

## 会话内命令

| 命令 | 作用 |
|------|------|
| `/help` | 帮助 |
| `/clear` | 清空记忆（新话题建议执行） |
| `/stats` | 查看记忆条数、系统状态 |
| `/tags security python` | 设置领域标签，影响缓存命中域 |
| `/mode structured` | 结构化模式（默认，省 Token） |
| `/quit` | 退出 |

## 行为说明

- 每条消息作为独立 **task** 走：Planner → Retriever → Executor（可跳过）→ Summarizer  
- **相似问题**可能触发 E2E/模板复用，终端会标注「记忆复用」  
- 同一 `--db` 文件下次启动仍可复用历史记忆  

## 与 benchmark 的区别

| | benchmark | chat |
|--|-----------|------|
| 任务 | 固定 12/24 题 | 你自由输入 |
| 记忆 | 实验用临时 DB | 默认 `chat_session.db` |
| 输出 | JSON 指标 | 格式化要点+结论 |
