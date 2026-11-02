"""Ch07显式迁移与预算检查，不读取或打印私有配置正文。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import Database, DatabaseSettings
from app.context.migration import migrate
from app.config import get_settings
from app.context.budget import ContextBudget
from app.workflow.persistence import run_async


async def main(command):
    """budget只打印非敏感推导结果；init才连接MySQL并显式执行原DDL。"""
    if command == 'budget':
        print(ContextBudget.from_settings(get_settings()))
        return
    database = Database(DatabaseSettings().url())
    try:
        await migrate(database, ROOT)
        print('已按顺序执行ch07-ddl.sql、ch07-layers.sql（已有完整列跳过）。')
    finally:
        # 脚本失败也释放连接；迁移的DDL隐式提交语义由migration模块明确处理。
        await database.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['init', 'budget'])
    run_async(main(parser.parse_args().command))
