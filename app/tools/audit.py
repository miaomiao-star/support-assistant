"""原审计表的独立短事务与显式迁移；不在应用启动时建业务表。"""

from pathlib import Path
from sqlalchemy import inspect, text
from app.models import ToolAuditLog
from app.context.migration import split_sql


class AuditStore:
    def __init__(self, database):
        self.db = database

    async def write(self, **values):
        # sessions.begin创建自己的Session/事务：正常退出提交，异常回滚并由引擎隔离。
        # 不能借用工具业务事务，否则日志故障可能回滚已经完成的工单。
        async with self.db.sessions.begin() as session:
            session.add(ToolAuditLog(**values))


async def validate_audit_table(database):
    # run_sync只桥接当前异步连接上的反射操作，不另建同步数据库连接。
    async with database.engine.connect() as connection:
        columns, foreign, indexes = await connection.run_sync(
            lambda conn: (
                inspect(conn).get_columns('tool_audit_logs'),
                inspect(conn).get_foreign_keys('tool_audit_logs'),
                inspect(conn).get_indexes('tool_audit_logs'),
            )
        )
        expected = set(ToolAuditLog.__table__.columns.keys())
        if {column['name'] for column in columns} != expected or foreign:
            raise RuntimeError('审计表列或外键与原ch08-ddl不一致')
        if {item['name'] for item in indexes} != {
            'idx_conversation_id',
            'idx_tool_name',
            'idx_status',
        }:
            raise RuntimeError('审计表索引与原ch08-ddl不一致')
        if connection.dialect.name == 'mysql':
            # SQLite隔离测试不具备MySQL unsigned/ENUM；真实库再逐列比较原DDL类型。
            for actual in columns:
                expected_column = ToolAuditLog.__table__.columns[actual['name']]
                expected_type = str(
                    expected_column.type.compile(dialect=connection.dialect)
                )
                actual_type = str(actual['type'].compile(dialect=connection.dialect))
                if (
                    expected_type != actual_type
                    or actual['nullable'] != expected_column.nullable
                ):
                    raise RuntimeError('审计列类型/可空性冲突：' + actual['name'])
            states = next(c['type'].enums for c in columns if c['name'] == 'status')
            if states != ['成功', '失败', '超时', '校验拦下', '权限拒绝']:
                raise RuntimeError('审计状态枚举与原DDL不一致')


async def migrate_audit(database, root):
    """只完整不存在时执行用户原文件；重跑核验现有结构，不改写CREATE。"""
    async with database.engine.connect() as connection:
        exists = await connection.run_sync(
            lambda conn: inspect(conn).has_table('tool_audit_logs')
        )
        if not exists:
            # 只由显式命令执行用户原SQL；表已存在也必须核验，不能静默接受漂移。
            for statement in split_sql(
                (Path(root) / 'sql/ch08-ddl.sql').read_text(encoding='utf-8-sig')
            ):
                await connection.execute(text(statement))
                await connection.commit()
    await validate_audit_table(database)
