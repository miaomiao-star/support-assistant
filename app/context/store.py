"""Ch07业务存储：匿名分组、完整正文、单调边界与只追加摘要。"""

import hashlib
import json

from langchain_core.messages import ToolMessage
from sqlalchemy import select, update, func
from sqlalchemy.exc import IntegrityError

from app.models import Conversation, ConversationSummary, Message, Ticket
from app.repositories.chat import ChatRepository, decode_message


class ContextRepository(ChatRepository):
    # 继承旧业务工具与消息写入方法，仅覆盖Ch07变化：归属分组、全量读取、工具观察与摘要。
    def __init__(self, db, knowledge=None, *, visitor=None):
        super().__init__(db, knowledge)
        self.visitor = visitor
        self._owners = {}  # 请求级旧客户端兼容凭据，不是共享登录态或跨会话记忆。

    def owner(self, session_id):
        """新客户端按访客分组；旧客户端只有原session时仍沿用会话摘要归属。"""
        source = 'visitor:' + self.visitor if self.visitor else session_id
        return hashlib.sha256(source.encode()).hexdigest()

    async def open_conversation(self, session_id):
        """解析会话别名并核验归属；只有原始session入口允许创建新会话。"""
        digest = hashlib.sha256(session_id.encode()).hexdigest()
        canonical = session_id.startswith('conv7:')
        if canonical:
            # 列表返回的conv7:<cid>可直接找回旧行，但不能借此猜出并创建不存在的会话。
            try:
                cid = int(session_id[6:])
            except ValueError:
                raise LookupError('Invalid conversation') from None
            if cid <= 0 or cid > (1 << 63) - 1:
                raise LookupError('Invalid conversation')
            if not self.visitor and cid not in self._owners:
                raise LookupError('Visitor required for canonical conversation')
        else:
            # 稳定63位正整数便于沿用既有BIGINT主键；SHA256不作为登录认证机制。
            cid = (int(digest[:16], 16) & ((1 << 63) - 1)) or 1
        owner = (
            self._owners[cid]
            if canonical and not self.visitor
            else self.owner(session_id)
        )
        try:
            # 仓库共享引擎/Session工厂，每次方法调用独占Session和短事务。
            async with self.db.sessions.begin() as session:
                row = await session.get(Conversation, cid)
                if row is None and not canonical:
                    session.add(Conversation(id=cid, user_id=owner))
                elif row is None or row.user_id != owner:
                    raise LookupError('Conversation not owned by visitor')
        except IntegrityError:
            # 两个首次请求同时创建同一ID时，只接受归属相同的已存在行，不能吞其他约束错。
            async with self.db.sessions() as session:
                row = await session.get(Conversation, cid)
                if row is None or row.user_id != owner:
                    raise
        # 老客户端原UUID先证明归属后，本请求内部可规范化；新请求裸猜cid仍拒绝。
        self._owners[cid] = owner
        return cid

    async def raw_history(self, cid):
        """按SQL行ID回载该会话全部流水；这里不做旧章200条限制或完整回合筛选。"""
        async with self.db.sessions() as session:
            rows = (
                await session.scalars(
                    select(Message)
                    .where(Message.conversation_id == cid)
                    .order_by(Message.id)
                )
            ).all()
            result = []
            for row in rows:
                message = decode_message(row)
                # LangGraph用id识别同一消息；分层函数另外用sql_msg_id比较SQL锚点。
                message.id = f'm:{row.id}'
                message.additional_kwargs['sql_msg_id'] = row.id
                result.append(message)
            return result

    async def record_tool_result(self, cid, call_id, result, after_message_id=0):
        """构造待图节点追加的ToolMessage，本方法不执行INSERT。"""
        # 新工具正文只进checkpoint；不把大块观察写回逐条消息表。
        # 工具没有自己的SQL行，anchor_id借用申请行；call_id只负责上游协议配对。
        return ToolMessage(
            content=json.dumps(result, ensure_ascii=False),
            tool_call_id=call_id,
            id=f't:{after_message_id}:{call_id}',
            additional_kwargs={'anchor_id': after_message_id},
        )

    async def create_ticket(self, cid, description, ticket_type, ticket_no, call_id):
        """确认接口仍写真实演示工单，但不再把成功工具结果顺便插进messages。"""
        async with self.db.sessions.begin() as session:
            existing = await session.get(Ticket, ticket_no)
            if existing and existing.conversation_id != cid:
                raise ValueError('Ticket ownership mismatch')
            if not existing:
                # 稳定ticket_no承接重复确认幂等；业务提交与后续PG回写仍不是跨库事务。
                session.add(
                    Ticket(
                        ticket_no=ticket_no,
                        conversation_id=cid,
                        description=description,
                        ticket_type=ticket_type,
                    )
                )
        return dict(ok=True, ticket_no=ticket_no, status='待处理')

    async def list_conversations(self):
        """只返回同一匿名访客的会话壳和首问预览，不把其他会话正文交给模型。"""
        if not self.visitor:
            return []
        async with self.db.sessions() as session:
            rows = (
                await session.scalars(
                    select(Conversation)
                    .where(Conversation.user_id == self.owner(''))
                    .order_by(Conversation.created_at.desc(), Conversation.id.desc())
                )
            ).all()
            result = []
            for row in rows:
                preview = await session.scalar(
                    select(Message.content)
                    .where(Message.conversation_id == row.id, Message.role == 'user')
                    .order_by(Message.id)
                    .limit(1)
                )
                result.append(
                    dict(
                        id=str(row.id),
                        # BIGINT作为字符串出JSON，避免浏览器Number丢失高位导致切错会话。
                        session_id=f'conv7:{row.id}',
                        preview=(preview or '新对话')[:60],
                        summarized=bool(row.summary_upto_msg_id),
                        created_at=row.created_at.isoformat(),
                    )
                )
            return result

    async def metadata(self, cid):
        """读取权威边界与原始摘要段；模型每次整理都会重读，避免依赖过期图缓存。"""
        async with self.db.sessions() as session:
            row = await session.get(Conversation, cid)
            if row is None:
                raise LookupError('Unknown conversation')
            segments = (
                await session.scalars(
                    select(ConversationSummary)
                    .where(ConversationSummary.conversation_id == cid)
                    .order_by(ConversationSummary.seq)
                )
            ).all()
            return dict(
                summary_upto=row.summary_upto_msg_id or 0,
                layer1_from=row.layer1_from_msg_id or 0,
                segments=[
                    dict(
                        seq=part.seq,
                        content=part.content,
                        from_msg_id=part.from_msg_id,
                        upto_msg_id=part.upto_msg_id,
                    )
                    for part in segments
                ],
            )

    async def advance_layer1(self, cid, boundary):
        """只允许层1边界向新消息方向移动；不搬消息，也不提前推进摘要覆盖。"""
        async with self.db.sessions.begin() as session:
            await session.execute(
                update(Conversation)
                .where(
                    Conversation.id == cid,
                    func.coalesce(Conversation.layer1_from_msg_id, 0) < boundary,
                )
                .values(layer1_from_msg_id=boundary)
            )

    async def append_summary(self, cid, expected, first, last, content):
        """原子提交新段、投影和覆盖锚点；expected过期则整次不提交。"""
        # CAS保护多任务/多进程；不在生成摘要时持有事务。边界冻结，不追未来layer1。
        async with self.db.sessions.begin() as session:
            # 原子UPDATE比较旧锚点并锁住本事务命中的行，不是先SELECT再无条件覆盖。
            # layer1_from>=last只核验源仍在可摘要范围，不把last扩成新的实时边界。
            changed = await session.execute(
                update(Conversation)
                .where(
                    Conversation.id == cid,
                    func.coalesce(Conversation.summary_upto_msg_id, 0) == expected,
                    Conversation.layer1_from_msg_id >= last,
                )
                .values(summary_upto_msg_id=last)
            )
            if not changed.rowcount:
                # 0行表示另一个任务已经推进边界或源范围失效；不是“成功生成就一定能落库”。
                return False
            parts = (
                await session.scalars(
                    select(ConversationSummary)
                    .where(ConversationSummary.conversation_id == cid)
                    .order_by(ConversationSummary.seq)
                )
            ).all()
            if content.strip():
                # 空寒暄批可推进覆盖但不造空段；有事实才分配下一个seq并插入新行。
                session.add(
                    ConversationSummary(
                        conversation_id=cid,
                        seq=(parts[-1].seq + 1 if parts else 1),
                        from_msg_id=first,
                        upto_msg_id=last,
                        content=content.strip(),
                    )
                )
            projection = '\n'.join(
                [part.content for part in parts]
                + ([content.strip()] if content.strip() else [])
            )
            # 这是字符串拼接缓存，不是再次调用LLM压缩旧段；段表中旧content从不UPDATE。
            # 全部段保留在段表；投影只是便于观察，不参与递归合并。
            await session.execute(
                update(Conversation)
                .where(Conversation.id == cid)
                .values(summary=projection)
            )
            return True
