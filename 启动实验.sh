#!/usr/bin/env bash
# 702solver 实验汇报入口 — 双击或在终端执行均可
# 无参数：先选「实验控制台 / 答辩快测」；有参数则原样传给 run.py，例如：
#   ./启动实验.sh --preset defense12 -y
#   ./启动实验.sh --mode 1,2 --suite core12 -y

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

_pause_on_exit() {
  read -r -p "按回车键退出..." _ </dev/tty 2>/dev/null || true
}

_show_banner() {
  echo ""
  echo "========================================================================"
  echo "  702solver — 多 Agent 实验系统"
  echo "  结构化通信 · 共享记忆 · 可对照纯文本基线"
  echo "  目录: $ROOT"
  echo "========================================================================"
  echo ""
}

_check_python() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "错误：未找到 python3，请先安装 Python 3。"
    _pause_on_exit
    exit 1
  fi
}

_check_api_key() {
  if [[ ! -f "$ROOT/.env" ]] || ! grep -qE '^DEEPSEEK_API_KEY=.+' "$ROOT/.env" 2>/dev/null; then
    if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
      echo "提示：未检测到 DEEPSEEK_API_KEY。"
      echo "  请在 $ROOT/.env 中添加一行："
      echo "  DEEPSEEK_API_KEY=你的密钥"
      echo ""
      read -r -p "仍要继续尝试运行吗？(y/n) " ans
      if [[ ! "$ans" =~ ^[yY是]$ ]]; then
        _pause_on_exit
        exit 1
      fi
    fi
  fi
}

_launcher_menu() {
  echo "  答辩主数据来自 benchmark 题库（core12 / full24），非自定义闲聊。"
  echo ""
  echo "  【推荐】"
  echo "  1  实验控制台（选跑题库 / 对照 / 演示，默认）"
  echo "  2  答辩快测：core12 · 仅模式 2（结构化，~12 分钟）"
  echo "  3  答辩对照：core12 · 模式 1+2（省 token 对比，~25 分钟）"
  echo ""
  echo "  【其它】"
  echo "  4  查看中文帮助"
  echo "  q  退出"
  echo ""
  read -r -p "请选择 [1]: " choice
  choice="${choice:-1}"

  case "$choice" in
    1|"")
      echo ""
      echo "进入实验控制台…"
      echo ""
      python3 "$ROOT/run.py"
      ;;
    2)
      echo ""
      echo "启动答辩快测（core12 · 模式 2）…"
      echo ""
      python3 "$ROOT/run.py" --preset defense12 -y
      ;;
    3)
      echo ""
      echo "启动答辩对照（core12 · 模式 1+2）…"
      echo ""
      python3 "$ROOT/run.py" --preset defense12-compare -y
      ;;
    4)
      python3 "$ROOT/run.py" --help-cn
      _pause_on_exit
      exit 0
      ;;
    q|Q|0)
      exit 0
      ;;
    *)
      echo "  无效选项，进入实验控制台。"
      echo ""
      python3 "$ROOT/run.py"
      ;;
  esac
}

_show_banner
_check_python
_check_api_key

if [[ $# -eq 0 ]]; then
  _launcher_menu
else
  python3 "$ROOT/run.py" "$@"
fi

code=$?
echo ""
if [[ $code -ne 0 ]]; then
  echo "运行结束，退出码: $code"
  _pause_on_exit
elif [[ $# -eq 0 ]]; then
  echo "（实验控制台已正常退出）"
else
  _pause_on_exit
fi
