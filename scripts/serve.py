"""跨平台 Web 启动器；Windows 使用 Psycopg 支持的 Selector 事件循环。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from app.workflow.persistence import run_async

if __name__ == "__main__":
    # 命令行只决定监听地址/端口，模型配置仍由应用读取现有.env。
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = uvicorn.Server(
        uvicorn.Config("app.main:app", host=args.host, port=args.port)
    )
    # 不用server.run自行创建默认循环；把serve协程交给支持Psycopg的统一启动器。
    run_async(server.serve())
