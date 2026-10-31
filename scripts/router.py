"""Ch06显式初始化演示退款申请表；不在Web启动时迁移业务表。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.db import Database, DatabaseSettings
from app.refunds import RefundApplication
from app.workflow.persistence import run_async


async def initialize():
    database = Database(DatabaseSettings().url())
    try:
        async with database.engine.begin() as connection:
            await connection.run_sync(
                lambda sync: RefundApplication.__table__.create(sync, checkfirst=True)
            )
        print("演示退款申请表已就绪，已有业务表和数据保持。")
    finally:
        await database.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init"])
    parser.parse_args()
    try:
        run_async(initialize())
    except Exception as exc:
        print(
            "初始化未完成：" + type(exc).__name__ + "；请检查原MySQL服务和本地配置。",
            file=sys.stderr,
        )
        raise SystemExit(1)
