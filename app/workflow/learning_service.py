"""当轮反馈按PG记录定位，入池事实在SQL防重；不通过更新图破坏当前暂停。"""

from copy import deepcopy
from sqlalchemy import select
from app.workflow.tooling_service import ToolingService
from app.rag.models import LowConfidenceQuestion
from app.rag.store import QualityStore


async def feedback_records(db, cid, state):
    async with db.sessions() as session:
        reasons = set(
            (
                await session.scalars(
                    select(LowConfidenceQuestion.reason).where(
                        LowConfidenceQuestion.conversation_id == cid,
                        LowConfidenceQuestion.source == 'user_feedback',
                    )
                )
            ).all()
        )
    return [
        dict(
            feedback_id=r['feedback_id'],
            answer_message_id=r['answer_message_id'],
            session_id=f'conv7:{cid}',
            voted=('feedback:' + r['feedback_id']) in reasons,
        )
        for r in state.get('learning_turns', {}).values()
    ]


class LearningService(ToolingService):
    def __init__(self, *args, learning, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning = learning

    async def context(self, session_id, user_text):
        context = await super().context(session_id, user_text)
        context.learning = self.learning
        return context

    async def stream(self, session_id, user_text, filters=None):
        with self.learning.tracing.request(
            'customer-chat', {'question': user_text}, session_id
        ):
            async for event in super().stream(session_id, user_text, filters):
                yield event

    async def resume_ticket(self, body):
        with self.learning.tracing.request(
            'ticket-resume',
            {'preview_id': body.preview_id, 'confirmed': body.confirmed},
            body.session_id,
        ):
            await self._resume_intent(body.session_id)
            async for event in super().resume_ticket(body):
                yield event

    async def _resume_intent(self, session_id, key=None):
        cid = await self.repo.open_conversation(session_id)
        state = (await self.graph.aget_state(self.config(f'conv7:{cid}'))).values
        source = state.get('learning_turns', {}).get(key, state) if key else state
        self.learning.tracing.intent(
            source.get('intent', '未分类'), source.get('operation', 'general')
        )

    async def resume(self, body):
        with self.learning.tracing.request(
            'order-resume', {'selection_id': body.selection_id}, body.session_id
        ):
            await self._resume_intent(body.session_id)
            async for event in super().resume(body):
                yield event

    async def confirm_ticket(self, body):
        with self.learning.tracing.request(
            'ticket-button', {'offer_id': body.offer_id}, body.session_id
        ):
            await self._resume_intent(body.session_id, body.offer_id)
            return await super().confirm_ticket(body)

    async def confirm_refund(self, body):
        with self.learning.tracing.request(
            'refund-button', {'offer_id': body.offer_id}, body.session_id
        ):
            await self._resume_intent(body.session_id, body.offer_id)
            return await super().confirm_refund(body)

    async def run(self, session_id, value, context):
        async for event in super().run(session_id, value, context):
            yield event
        snapshot = await self.graph.aget_state(self.config(session_id))
        if not snapshot.next:
            record = snapshot.values.get('learning_turns', {}).get(
                snapshot.values.get('request_id')
            )
            if record:
                yield dict(
                    type='answer',
                    feedback_id=record['feedback_id'],
                    answer_message_id=record['answer_message_id'],
                    session_id=session_id,
                )

    async def feedback(self, body):
        cid = await self.repo.open_conversation(body.session_id)
        snapshot = await self.graph.aget_state(self.config(f'conv7:{cid}'))
        record = snapshot.values.get('learning_turns', {}).get(body.feedback_id)
        if not record:
            raise LookupError('回答轮次不存在或尚未完成')
        reason = 'feedback:' + body.feedback_id
        # 与调用方同一会话锁协调；SQL已提交但响应/PG失败时仍找回原行，不重复累加。
        async with self.repo.db.sessions() as session:
            existing = await session.scalar(
                select(LowConfidenceQuestion.id).where(
                    LowConfidenceQuestion.conversation_id == cid,
                    LowConfidenceQuestion.source == 'user_feedback',
                    LowConfidenceQuestion.reason == reason,
                )
            )
        if existing is not None:
            return dict(pool_id=existing, duplicate=True)
        chunks = deepcopy(record.get('retrieved_chunks'))
        if record.get('retrieved') and chunks is None:
            # 兼容缺快照的记录时仅回捞相同request，绝不用当前另一轮的检索。
            chunks = (
                deepcopy(snapshot.values.get('retrieved_chunks') or [])
                if snapshot.values.get('request_id') == body.feedback_id
                else []
            )
        with self.learning.tracing.request(
            'user-feedback', {'feedback_id': body.feedback_id}, body.session_id
        ):
            self.learning.tracing.intent(
                record['intent'], record.get('operation', 'general')
            )
            key = await QualityStore(self.repo.db).low_confidence(
                cid,
                record['raw_question'],
                'user_feedback',
                reason,
                retrieved_chunks=chunks,
            )
        return dict(pool_id=key, duplicate=False)
