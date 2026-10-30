"""Ch04环境初始化、完整重建与显式切换；普通pending补跑仍用kb.py。"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sqlalchemy import inspect
from app.db import Database, DatabaseSettings
from app.knowledge.runtime import build_service
from app.config import get_settings
from app.rag.models import LowConfidenceQuestion, FaithCase
from app.knowledge.splitter import embedding_text


def sql_statements(sql):
    """仅在引号外按分号分句，保留COMMENT中的分号与SQL转义引号。"""
    sql = re.sub(r'^\s*--.*$', '', sql, flags=re.M)
    statements = []
    start = 0
    quote = None
    i = 0
    # quote记录当前是否在单/双引号或标识符引号中，只有引号外的分号才结束语句。
    while i < len(sql):
        char = sql[i]
        if quote:
            # 支持SQL反斜线转义和两个连续引号，不把COMMENT正文拆成多条SQL。
            if char == '\\':
                i += 2
                continue
            if char == quote:
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    i += 2
                    continue
                quote = None
        elif char in "'\"`":
            quote = char
        elif char == ';':
            if sql[start:i].strip():
                statements.append(sql[start:i].strip())
            start = i + 1
        i += 1
    if sql[start:].strip():
        statements.append(sql[start:].strip())
    return statements


async def initialize(db):
    """按原DDL创建缺失表，不重建或清空现有业务表。"""
    sql = (ROOT / 'sql/ch04-ddl.sql').read_text(encoding='utf-8')
    async with db.engine.begin() as conn:
        tables = set(await conn.run_sync(lambda c: inspect(c).get_table_names()))
        for statement in sql_statements(sql):
            statement = statement.strip()
            if not statement:
                continue
            match = re.search(r'CREATE TABLE (\w+)', statement)
            if match and match[1] in tables:
                print('已存在，保留：' + match[1])
                continue
            await conn.exec_driver_sql(statement)
            if match:
                print('已建表：' + match[1])


def index_matches(expected, actual):
    """切换校验必须精确比对主键和正文/元数据，空库不允许验收通过。"""
    return bool(expected) and sorted(expected, key=lambda r: r['id']) == sorted(
        actual, key=lambda r: r['id']
    )


async def verify_index(service):
    # 对照的是完整权威原文，不只看数量：相同行数也可能是错误主键或旧正文。
    rows = await service.store.all_chunks()
    expected = [
        {
            'id': str(r['id']),
            'text': embedding_text(r),
            'category': r['category'],
            'content_type': r['content_type'] or '',
            'product_category': service.vectors.product_category(r),
        }
        for r in rows
    ]

    def read():
        # 同步Milvus SDK通过已有线程封装执行，连接和加载操作共用线程锁。
        with service.vectors.lock:
            if not service.vectors._ready(1024):
                return []
            return service.vectors.client.query(
                service.vectors.collection,
                filter='',
                limit=16384,
                output_fields=[
                    'id',
                    'text',
                    'category',
                    'content_type',
                    'product_category',
                ],
                consistency_level='Strong',
            )

    actual = await service.vectors._call(read)
    return {
        'mysql_rows': len(rows),
        'milvus_rows': len(actual),
        'passed': index_matches(expected, actual)
        and all(r['vectorize_status'] == 'done' for r in rows),
    }


async def main(args):
    db = Database(DatabaseSettings().url())
    service = build_service(db, rag_enabled=True)
    try:
        if args.command == 'init':
            await initialize(db)
        elif args.command in ('rebuild', 'activate'):
            # activate只核验/切换；rebuild会包含MySQL里已经done的行，重建新集合。
            result = (
                await service.vectorize(rebuild=True)
                if args.command == 'rebuild'
                else None
            )
            report = await verify_index(service)
            if result:
                report['result'] = result
                report['passed'] = (
                    report['passed']
                    and result['failed'] == 0
                    and result['succeeded'] == report['mysql_rows']
                )
            report_path = ROOT / 'dev-notes' / f'ch04-migration-{time.time_ns()}.json'
            if args.activate or args.command == 'activate':
                # 全量精确比对失败时不改开关，继续保留当前可用服务。
                if not report['passed']:
                    raise ValueError('全量校验未通过，没有切换应用')
                settings = get_settings()
                # 只写非敏感开关；真实.env与旧Lite数据完全保留。
                (ROOT / '.env.rag').write_text(
                    f'RAG_ENABLED=true\nRAG_MILVUS_URI={settings.rag_milvus_uri}\nRAG_COLLECTION={settings.rag_collection}\n',
                    encoding='utf-8',
                )
                report['activated'] = True
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            print(json.dumps(report, ensure_ascii=False), flush=True)
        elif args.command == 'search':
            result = await service.retriever.search(args.query, strategy=args.strategy)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        elif args.command == 'status':
            print(json.dumps(await service.store.stats(), ensure_ascii=False))
    finally:
        # 异常也要释放HTTP、Milvus和数据库资源，避免命令退出后留下连接。
        await service.close()
        await db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'command', choices=['init', 'rebuild', 'activate', 'search', 'status']
    )
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--query', default='邮费是多少')
    parser.add_argument(
        '--strategy',
        default='hybrid_rerank',
        choices=['dense', 'bm25', 'hybrid', 'hybrid_rerank'],
    )
    asyncio.run(main(parser.parse_args()))
