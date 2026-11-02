"""跨平台 Web 启动器；Windows 使用 Psycopg 支持的 Selector 事件循环。"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from app.workflow.persistence import run_async

if __name__ == "__main__":
    # 命令行决定监听地址和上下文演示档位，模型/上游/凭据仍从原配置读取。
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        '--context-profile', choices=['default', 'demo'], default='default'
    )
    args = parser.parse_args()
    if args.context_profile == 'demo':
        # 只覆盖用户指定六个非敏感演示参数；不改私有.env和模型配置。
        os.environ.update(
            MODEL_CONTEXT_WINDOW='18000',
            MAX_OUTPUT_TOKENS='2000',
            MAX_USER_INPUT_TOKENS='2000',
            MAX_AGENT_STEPS='3',
            TOOL_RESULT_MAX_TOKENS='1200',
            RERANK_TOP_K='5',
        )
    # 传应用导入字符串让运行期读取上述环境；不在启动器里提前创建含旧预算的Settings。
    server = uvicorn.Server(
        uvicorn.Config("app.main:app", host=args.host, port=args.port)
    )
    # 不用server.run自行创建默认循环；把serve协程交给支持Psycopg的统一启动器。
    run_async(server.serve())
