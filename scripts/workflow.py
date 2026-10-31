"""Ch05 私有配置与官方 checkpoint 初始化，不修改 .env 模型凭据。"""

import argparse
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.workflow.persistence import open_checkpointer, run_async
from app.workflow.graph import build_graph
from app.workflow.service import WorkflowService


def init_config():
    """仅在缺少本地文件时生成独立图数据库配置，不覆盖模型或已有数据库凭据。"""
    path = ROOT / ".env.graph"
    if path.exists():
        print("已有 .env.graph，保持原样。")
        return
    # 十六进制随机密码没有DSN分隔字符，可同时用于容器环境和连接串。
    password = secrets.token_hex(24)
    # x 模式防并发覆盖，输出只报告路径，绝不回显密码或 DSN。
    with path.open("x", encoding="utf-8") as output:
        output.write(
            f"GRAPH_POSTGRES_PASSWORD={password}\nGRAPH_POSTGRES_DSN=postgresql://mewhelp_graph:{password}@127.0.0.1:5433/mewhelp_graph\n"
        )
    print("已生成私有 .env.graph（未修改模型配置）。")


async def initialize():
    """显式执行官方saver迁移，只创建/升级图状态表，不碰MySQL业务表。"""
    async with open_checkpointer(get_settings(), setup=True):
        print("官方 Postgres checkpoint 表初始化成功。")


async def inspect_state(session_id):
    """读取同一图线程的状态摘要，不运行图节点、不发起LLM调用。"""
    async with open_checkpointer(get_settings()) as saver:
        state = (
            await build_graph(saver).aget_state(WorkflowService.config(session_id))
        ).values
        if not state:
            print("未找到这个会话的图状态。")
            return
        import json

        # 只展示路径、计数和停止原因，不打印原始对话、完整证据或私有DSN。
        print(
            json.dumps(
                dict(
                    intent=state["intent"],
                    route=state["route"],
                    trace=state["trace"],
                    agent_steps=state["agent_steps"],
                    history_messages=len(state["history"]),
                    saved_offers=len(state.get("offers", {})),
                    stop_reason=state["stop_reason"] or "answer",
                ),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    # 配置生成、迁移、只读查看分成三个显式命令，启动Web不会偷偷执行完整初始化。
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init-config", "init", "inspect"])
    parser.add_argument("--session-id")
    args = parser.parse_args()
    try:
        if args.command == "init-config":
            init_config()
        elif args.command == "init":
            run_async(initialize())
        else:
            if not args.session_id:
                parser.error("inspect 需要 --session-id")
            run_async(inspect_state(args.session_id))
    except Exception as exc:
        # 异常原文可能带连接信息，只输出类型与操作提示，不输出密码/DSN。
        print(
            "操作未完成："
            + type(exc).__name__
            + "；请检查独立 Postgres 服务与本地配置。",
            file=sys.stderr,
        )
        raise SystemExit(1)
