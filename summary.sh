#!/bin/bash

# 激活虚拟环境（假设 .venv 位于脚本同级目录）
VENV_PATH="$(dirname "$0")/.venv/bin/activate"
SCRIPT_PATH="$(dirname "$0")/summary.py"

if [ ! -f "$VENV_PATH" ]; then
    echo "错误：未找到虚拟环境文件 $VENV_PATH"
    exit 1
fi

. "$VENV_PATH" || { echo "激活虚拟环境失败"; exit 1; }

# 执行 Python 脚本并动态传递所有参数
python $SCRIPT_PATH "$@"

# 退出虚拟环境
deactivate