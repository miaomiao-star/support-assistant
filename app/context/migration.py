"""按用户指定顺序执行原SQL；分句识别引号中的分号，不改写DDL。"""

from pathlib import Path
from sqlalchemy import BigInteger, Text, inspect, text


def validate_columns(definitions, names):
    """重复执行前核验已存在目标列的类型/可空性；不替用户自动修复冲突结构。"""
    for column in definitions:
        if column['name'] not in names:
            continue
        expected = Text if column['name'] == 'summary' else BigInteger
        if not isinstance(column['type'], expected) or not column['nullable']:
            raise RuntimeError(f"Ch07列{column['name']}定义冲突，请核对原DDL")
        if expected is BigInteger and not getattr(column['type'], 'unsigned', True):
            raise RuntimeError(f"Ch07列{column['name']}定义冲突：要求UNSIGNED")


def split_sql(source):
    """针对本章原DDL逐字符分句，引号内分号及SQL注释都不能作为语句结束。"""
    # quote记录当前引号，escaped处理引号内反斜线；只在普通状态识别注释或分号。
    statements, buffer, quote, escaped = [], [], None, False
    index = 0
    while index < len(source):
        char = source[index]
        if quote is None and source.startswith('--', index):
            # 本章行注释也含分号，必须先跳注释，不能直接对源文本split(';')。
            end = source.find('\n', index)
            index = len(source) if end < 0 else end
            continue
        if quote is None and source.startswith('/*', index):
            end = source.find('*/', index + 2)
            if end < 0:
                raise ValueError('Unclosed SQL comment')
            index = end + 2
            continue
        buffer.append(char)
        if escaped:
            escaped = False
        elif char == '\\' and quote:
            escaped = True
        elif quote:
            if char == quote:
                quote = None
        elif char in ("'", '"', '`'):
            quote = char
        elif char == ';':
            # 到这里必然不在引号中，才将本条完整语句交给数据库执行。
            statements.append(''.join(buffer[:-1]).strip())
            buffer = []
        index += 1
    if ''.join(buffer).strip():
        statements.append(''.join(buffer).strip())
    return statements


async def migrate(database, root):
    """按用户指定顺序执行两份原文件；只跳过已完整应用的目标ALTER。"""
    async with database.engine.connect() as connection:
        for filename, columns in [
            ('ch07-ddl.sql', {'summary', 'summary_upto_msg_id'}),
            ('ch07-layers.sql', {'layer1_from_msg_id'}),
        ]:
            definitions = await connection.run_sync(
                lambda conn: inspect(conn).get_columns('conversations')
            )
            existing = {column['name'] for column in definitions}
            present = columns & existing
            if present and present != columns:
                # 第一份ALTER一次增加两列，部分存在时不可按缺列自行改写原SQL继续。
                raise RuntimeError(f'{filename}部分列已存在，停止迁移以便核对')
            validate_columns(definitions, present)
            for statement in split_sql(
                (Path(root) / 'sql' / filename).read_text(encoding='utf-8-sig')
            ):
                if (
                    statement.lstrip().upper().startswith('ALTER TABLE')
                    and present == columns
                ):
                    continue
                # CREATE TABLE IF NOT EXISTS仍交数据库处理；此工具不是通用迁移/回滚框架。
                await connection.execute(text(statement))
                await connection.commit()  # MySQL DDL本身隐式提交，不伪称两文件原子事务。
