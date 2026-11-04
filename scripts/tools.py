"""Ch08显式审计建表；不打印数据库配置或凭据。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.db import Database, DatabaseSettings
from app.tools.audit import migrate_audit
from app.workflow.persistence import run_async


async def main():
    # 显式迁移命令：已存在只核验结构，不清表，也不由Web启动隐式执行DDL。
    db = Database(DatabaseSettings().url())
    try:
        await migrate_audit(db, ROOT)
        print('tool_audit_logs已按原ch08-ddl建表/核验，无外键。')
    finally:
        # 无论建表或核验成功与否都关闭连接池，异常继续向命令行报告。
        await db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['init'])
    parser.parse_args()
    run_async(main())
