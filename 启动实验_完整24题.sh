#!/usr/bin/env bash
# 一键完整 benchmark：full24 · 模式 2（耗时较长，约 35 分钟）
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo ""
echo "即将运行 full24 完整 24 题（模式 2），预计约 35 分钟。"
read -r -p "确认开始？(y/n) " ans
if [[ ! "$ans" =~ ^[yY是]$ ]]; then
  echo "已取消。"
  read -r -p "按回车键退出..." _ </dev/tty 2>/dev/null || true
  exit 0
fi

exec "$ROOT/启动实验.sh" --preset full24 -y
