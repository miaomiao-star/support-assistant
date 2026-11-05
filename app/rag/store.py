"""质量问题留痕与编造个案处置；外部模型请求不进入事务。"""

from datetime import datetime, timezone
from sqlalchemy import select
from app.rag.models import LowConfidenceQuestion, FaithCase


def values(row):
    # 只返回普通列值，时间转ISO字符串，避免把已关闭Session的ORM对象交给API。
    result = {}
    for column in row.__table__.columns:
        value = getattr(row, column.name)
        if isinstance(value, datetime):
            value = value.isoformat()
        result[column.name] = value
    return result


class QualityStore:
    def __init__(self, db):
        self.db = db

    async def low_confidence(self, conversation_id, raw_question, source, reason, *, retrieved_chunks=None):
        """三入口先持久化原话/当轮快照；后台失败仍可从未归并行恢复。"""
        if source not in ('retrieval_low_conf', 'self_check', 'user_feedback'):
            raise ValueError('问题池来源无效')
        async with self.db.sessions.begin() as session:
            row = LowConfidenceQuestion(
                    conversation_id=conversation_id,
                    raw_question=raw_question,
                    source=source,
                    reason=reason,
                    retrieved_chunks=retrieved_chunks,
            )
            session.add(row)
            await session.flush()
            return row.id

    async def low_questions(self):
        # 管理页只看最近200条；这里的上限不是数据库保留条数限制。
        async with self.db.sessions() as session:
            return [
                values(r)
                for r in await session.scalars(
                    select(LowConfidenceQuestion)
                    .order_by(LowConfidenceQuestion.id.desc())
                    .limit(200)
                )
            ]

    async def record_faith_case(self, case):
        """正式策略的同题重复出现更新最近快照、增加次数，并重新打开已处置个案。"""
        if case['strategy'] != 'hybrid_rerank':
            raise ValueError('长期台账仅记录正式策略')
        async with self.db.sessions.begin() as session:
            row = await session.scalar(
                select(FaithCase)
                .where(FaithCase.eval_id == case['eval_id'])
                .with_for_update()
            )
            # 锁住已有行后更新计数，避免已存在个案的并发更新互相覆盖。
            if row is None:
                session.add(FaithCase(**case))
                return
            for key, value in case.items():
                setattr(row, key, value)
            row.seen_count += 1
            row.status = '未解决'
            row.resolution = None
            row.last_seen_at = datetime.now(timezone.utc).replace(tzinfo=None)
            # resolved_at特意保留，用于识别“以前解决过、现在再次出现”。

    async def faith_cases(self):
        # 按最近一次出现时间排序，复发问题会重新浮到列表前面。
        async with self.db.sessions() as session:
            return [
                values(r)
                for r in await session.scalars(
                    select(FaithCase).order_by(FaithCase.last_seen_at.desc()).limit(200)
                )
            ]

    async def resolve(self, key, status, resolution):
        """处置说明是必填交代；退回未解决时清空说明，保留历史处置时间。"""
        if status not in ('未解决', '已解决', '无需解决'):
            raise ValueError('处置状态无效')
        if status != '未解决' and not resolution.strip():
            raise ValueError('处置时必须填写说明')
        if len(resolution) > 300:
            raise ValueError('处置说明过长')
        async with self.db.sessions.begin() as session:
            row = await session.scalar(
                select(FaithCase).where(FaithCase.id == key).with_for_update()
            )
            if row is None:
                raise ValueError('个案不存在')
            row.status = status
            row.resolution = resolution.strip() if status != '未解决' else None
            # 只有明确处置才刷新时间；复发保留此时间，以便识别曾处置过的个案。
            if status != '未解决':
                row.resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
            await session.flush()
            return values(row)
