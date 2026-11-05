"""持久待处理池→模型标准化/查重→短事务归并；模型调用不占SQL事务。"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from sqlalchemy import select
from app.learning.models import ReviewQueue
from app.learning.prompts import (
    FaqDraft,
    SemanticMatch,
    NORMALIZE_PROMPT,
    DEDUPE_PROMPT,
)
from app.rag.models import LowConfidenceQuestion
from app.rag.store import values
from app.knowledge.materials import manual_chunks
from app.knowledge.mining import structured
from app.rag.retrieval import preserve_identifiers


class Flywheel:
    def __init__(self, db, knowledge, model, *, normalizer=None, matcher=None):
        self.db, self.knowledge, self.model = db, knowledge, model
        self.normalizer, self.matcher = normalizer, matcher
        if not hasattr(db, 'flywheel_lock'):
            db.flywheel_lock = asyncio.Lock()

    @asynccontextmanager
    async def serial(self):
        # 与CLI共同使用本机文件锁；模型期间不持SQL事务，仍串行化“发现候选→归并”。
        async with self.db.flywheel_lock:
            handle = None
            try:
                if self.db.engine.dialect.name == 'mysql':
                    root = Path(__file__).resolve().parents[2] / 'work/ch09'
                    root.mkdir(parents=True, exist_ok=True)
                    handle = (root / 'flywheel.lock').open('a+b')
                    handle.seek(0)
                    if not handle.read(1):
                        handle.write(b'0')
                        handle.flush()
                    handle.seek(0)
                    import os

                    if os.name == 'nt':
                        import msvcrt

                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                yield
            finally:
                if handle:
                    handle.close()

    async def list(self, status=None):
        async with self.db.sessions() as session:
            query = select(ReviewQueue).order_by(
                ReviewQueue.occurrence_count.desc(), ReviewQueue.id.desc()
            )
            if status:
                query = query.where(ReviewQueue.review_status == status)
            return [values(r) for r in await session.scalars(query.limit(200))]

    async def detail(self, key):
        async with self.db.sessions() as session:
            row = await session.get(ReviewQueue, key)
            if not row:
                raise LookupError('待审缺口不存在')
            questions = [
                values(r)
                for r in await session.scalars(
                    select(LowConfidenceQuestion)
                    .where(LowConfidenceQuestion.matched_review_id == key)
                    .order_by(LowConfidenceQuestion.id)
                )
            ]
            for question in questions:
                if question['conversation_id'] is not None:
                    question['conversation_id'] = str(question['conversation_id'])
            return {**values(row), 'questions': questions}

    async def has_pending(self):
        # 空轮询只做最小存在性查询；只有真实待处理事实才进入模型/trace。
        async with self.db.sessions() as session:
            pending = await session.scalar(
                select(LowConfidenceQuestion.id)
                .where(LowConfidenceQuestion.matched_review_id.is_(None))
                .limit(1)
            )
            return pending is not None

    async def process(self, limit=20):
        processed = 0
        async with self.serial():
            async with self.db.sessions() as session:
                incoming = [
                    values(r)
                    for r in await session.scalars(
                        select(LowConfidenceQuestion)
                        .where(LowConfidenceQuestion.matched_review_id.is_(None))
                        .order_by(LowConfidenceQuestion.id)
                        .limit(limit)
                    )
                ]
            for item in incoming:
                if self.normalizer:
                    draft = await self.normalizer(
                        item['raw_question'], item['retrieved_chunks']
                    )
                else:
                    result = await structured(
                        self.model,
                        FaqDraft,
                        NORMALIZE_PROMPT,
                        {
                            'question': item['raw_question'],
                            'retrieved_chunks': item['retrieved_chunks'],
                        },
                    )
                    draft = result.model_dump()
                    if (
                        preserve_identifiers(
                            item['raw_question'], draft['normalized_question']
                        )
                        != draft['normalized_question']
                    ):
                        raise ValueError('标准化改变了型号/数字，保留原池行待重试')
                draft = FaqDraft.model_validate(draft).model_dump()
                async with self.db.sessions() as session:
                    candidates = [
                        values(r)
                        for r in await session.scalars(
                            select(ReviewQueue)
                            .where(ReviewQueue.review_status == '待审')
                            .order_by(ReviewQueue.id)
                        )
                    ]
                matched = None
                # 分批覆盖完整待审队列，不能只对页面最近200条查重。
                for offset in range(0, len(candidates), 16):
                    batch = candidates[offset : offset + 16]
                    payload = [
                        dict(id=r['id'], normalized_question=r['normalized_question'])
                        for r in batch
                    ]
                    if self.matcher:
                        matched = await self.matcher(
                            draft['normalized_question'], payload
                        )
                    else:
                        result = await structured(
                            self.model,
                            SemanticMatch,
                            DEDUPE_PROMPT,
                            {
                                'question': draft['normalized_question'],
                                'candidates': payload,
                            },
                        )
                        matched = result.matched_review_id
                    if matched is not None and (
                        type(matched) is not int
                        or matched not in {r['id'] for r in batch}
                    ):
                        raise ValueError('查重返回了不存在的候选')
                    if matched is not None:
                        break
                # 匹配与次数、原话归并锚点同事务；重启扫描时已匹配行不会重复累加。
                async with self.db.sessions.begin() as session:
                    row = await session.get(
                        LowConfidenceQuestion, item['id'], with_for_update=True
                    )
                    if row.matched_review_id is not None:
                        continue
                    target = (
                        await session.get(ReviewQueue, matched, with_for_update=True)
                        if matched
                        else None
                    )
                    if target and target.review_status != '待审':
                        raise ValueError('候选审核状态已变化，请重新处理')
                    if target:
                        target.occurrence_count += 1
                    else:
                        target = ReviewQueue(**draft)
                        session.add(target)
                        await session.flush()
                    row.matched_review_id = target.id
                processed += 1
        return dict(processed=processed)

    async def reject(self, key):
        async with self.serial():
            async with self.db.sessions.begin() as session:
                row = await session.get(ReviewQueue, key, with_for_update=True)
                if not row:
                    raise LookupError('待审缺口不存在')
                if row.review_status == '通过':
                    raise ValueError('不能用驳回撤销已核准知识')
                row.review_status = '驳回'
        return await self.detail(key)

    async def approve(self, key, answer, *, category='客服FAQ', content_type='faq'):
        if not answer.strip():
            raise ValueError('必须填写人工核准答案')
        async with self.serial():
            async with self.db.sessions.begin() as session:
                row = await session.get(ReviewQueue, key, with_for_update=True)
                if not row:
                    raise LookupError('待审缺口不存在')
                if row.review_status == '驳回':
                    raise ValueError('已驳回记录保留历史，请从新待审项处理')
                if (
                    row.review_status == '通过'
                    and row.approved_answer != answer.strip()
                ):
                    raise ValueError('本次核准答案已冻结，不能重放时改写')
                question = row.normalized_question
                row.approved_answer, row.review_status = answer.strip(), '通过'
            # 人工决定先保存；原文来源稳定，崩溃后同答案重放不会重复导入。
            chunks, _ = manual_chunks(
                dict(
                    category=category,
                    questions=question,
                    answer=answer.strip(),
                    section_path='飞轮人工核准',
                    content_type=content_type,
                    is_key_clause=False,
                )
            )
            source = f'flywheel:{key}'
            chunks = [replace(c, section_path=source + ' / 人工核准') for c in chunks]
            imported = await self.knowledge.store.import_chunks(chunks, source)
            result = await self.knowledge.vectorize()
            known = {r['id']: r for r in await self.knowledge.store.all_chunks()}
            indexed = all(
                known[k]['vectorize_status'] == 'done' for k in imported['ids']
            )
            return dict(
                id=key,
                review_status='通过',
                knowledge_ids=imported['ids'],
                indexed=indexed,
                vectorization=result,
            )
