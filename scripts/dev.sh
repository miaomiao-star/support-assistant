#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# 单进程运行，重启清空历史；不要使用多个 worker。
exec uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
