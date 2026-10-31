"""官方 Postgres checkpointer 的生命周期；不提供生产内存降级。"""

import asyncio
import sys
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


def run_async(coroutine):
    """Windows Psycopg 要求 Selector；在创建事件循环前明确选择。"""
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    # Runner负责创建/关闭循环；先运行uvicorn再设置循环策略已经来不及。
    with asyncio.Runner(loop_factory=factory) as runner:
        return runner.run(coroutine)


@asynccontextmanager
async def open_checkpointer(settings, setup=False):
    """打开官方持久化资源；setup仅供显式初始化，普通启动只检查表是否就绪。"""
    if settings.graph_postgres_dsn is None:
        raise ValueError("请先运行 scripts/workflow.py init-config 并启动独立 Postgres")
    # 这些连接参数与已核对的官方 saver 实现一致，池随 lifespan 统一关闭。
    pool = AsyncConnectionPool(
        settings.graph_postgres_dsn.get_secret_value(),
        min_size=1,
        max_size=5,
        # 显式await打开，避免构造函数依赖尚不存在的事件循环或未来默认值变化。
        open=False,
        timeout=10,
        reconnect_timeout=5,
        # saver自行安排数据库操作；迁移要求autocommit，查询结果按列名读取。
        # prepare_threshold=0与官方from_conn_string所用连接参数保持一致。
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    try:
        # wait=True让连接失败在启动阶段暴露，不把错误推迟到第一条聊天请求。
        await pool.open(wait=True, timeout=10)
        saver = AsyncPostgresSaver(pool)
        if setup:
            await saver.setup()
        else:
            # 缺少迁移时启动失败，不能运行到第一条用户请求才暴露问题。
            await saver.aget_tuple({"configurable": {"thread_id": "__startup_check__"}})
        yield saver
    finally:
        # Web关闭、初始化异常或启动检查失败，均关闭池中的数据库连接。
        await pool.close()
