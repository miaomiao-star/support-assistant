"""按原SQL顺序显式迁移；已有/部分迁移只核验，不猜测改写DDL。"""

from pathlib import Path
from sqlalchemy import inspect
from app.context.migration import split_sql
from app.learning.models import ReviewQueue, EvalRun
from app.rag.models import LowConfidenceQuestion


async def validate_schema(database):
    async with database.engine.connect() as connection:
        for model in (ReviewQueue, EvalRun, LowConfidenceQuestion):
            columns = await connection.run_sync(
                lambda c: inspect(c).get_columns(model.__tablename__)
            )
            actual = {c['name']: c for c in columns}
            expected = model.__table__.columns
            if set(actual) != set(expected.keys()):
                raise RuntimeError('Ch09表列冲突：' + model.__tablename__)
            if connection.dialect.name == 'mysql':
                for column in expected:
                    found = actual[column.name]
                    if (
                        str(column.type.compile(dialect=connection.dialect))
                        != str(found['type'].compile(dialect=connection.dialect))
                        or column.nullable != found['nullable']
                    ):
                        raise RuntimeError(
                            'Ch09列类型/可空性冲突：'
                            + model.__tablename__
                            + '.'
                            + column.name
                        )
                indexes = await connection.run_sync(
                    lambda c: inspect(c).get_indexes(model.__tablename__)
                )
                required = {
                    index.name
                    for index in model.__table__.indexes
                    if index.name.startswith('idx_')
                }
                if not required <= {i['name'] for i in indexes}:
                    raise RuntimeError('Ch09索引缺失：' + model.__tablename__)
        foreign = await connection.run_sync(
            lambda c: inspect(c).get_foreign_keys('low_confidence_questions')
        )
        match = next(
            (f for f in foreign if f['constrained_columns'] == ['matched_review_id']),
            None,
        )
        if (
            not match
            or match['referred_table'] != 'review_queue'
            or match['referred_columns'] != ['id']
            or match.get('options', {}).get('ondelete', '').upper() != 'SET NULL'
        ):
            raise RuntimeError('Ch09原归并外键不匹配')


async def migrate(database, root):
    statements = split_sql(
        (Path(root) / 'sql/ch09-ddl.sql').read_text(encoding='utf-8-sig')
    )
    if len(statements) != 4:
        raise RuntimeError('Ch09原SQL结构变化，停止以便核对')
    async with database.engine.connect() as connection:
        # 原DDL的JSON注释含冒号，不能交text()误解析成绑定参数；原样送驱动。
        await connection.exec_driver_sql(statements[0])
        for table, statement in zip(('review_queue', 'eval_runs'), statements[1:3]):
            exists = await connection.run_sync(lambda c: inspect(c).has_table(table))
            if not exists:
                await connection.exec_driver_sql(statement)
                await connection.commit()
        columns = await connection.run_sync(
            lambda c: inspect(c).get_columns('low_confidence_questions')
        )
        present = {c['name'] for c in columns} & {
            'retrieved_chunks',
            'matched_review_id',
        }
        if present and len(present) != 2:
            raise RuntimeError('Ch09部分列已存在，不改写原ALTER，请核对')
        if not present:
            await connection.exec_driver_sql(statements[3])
            await connection.commit()
    await validate_schema(database)
