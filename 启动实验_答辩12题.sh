#!/usr/bin/env bash
# 一键答辩：core12 标准 12 题，模式 1+2 对照（生成 token 对比表）
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec "$ROOT/启动实验.sh" --preset defense12-compare -y
