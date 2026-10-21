"""数据库访问层：不依赖 HTTP 或模型，所有事务都短于模型生成时间。"""
import hashlib
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models import Conversation, FAQ, Message, Ticket


def encode_message(cid, message):
    """
    LangChain 的消息对象 → SQLAlchemy Message 行；cid 是会话表主键。
    **values 将字典展开为构造参数，避免每个角色重复填写公共字段。
    """
    values = dict(conversation_id=cid, content=message.content)
    if isinstance(message, HumanMessage):
        return Message(**values, role='user')
    if isinstance(message, ToolMessage):
        return Message(**values, role='tool', tool_call_id=message.tool_call_id)
    # 服务层只传 human/assistant/tool 三类，剩余分支对应 AIMessage。
    # 工具申请是 assistant 的属性，不能把它当成 role='tool'。
    return Message(**values, role='assistant', tool_calls=message.tool_calls or None)


def decode_message(row):
    """
    数据库行 → LangChain 消息，恢复后可直接放回下一次模型上下文。
    空正文统一还原为 ''，工具参数和关联 ID 则保留原结构。
    """
    if row.role == 'user':
        return HumanMessage(content=row.content or '')
    if row.role == 'tool':
        return ToolMessage(content=row.content or '', tool_call_id=row.tool_call_id)
    return AIMessage(content=row.content or '', tool_calls=row.tool_calls or [])


class ChatRepository:
    # 仓库负责持久化与查询，不选择工具，不生成用户回答，也不负责 SSE。
    def __init__(self, db, knowledge=None):
        self.db = db
        self.knowledge = knowledge

    async def open_conversation(self, session_id):
        """
        同一 session_id 总是得到相同摘要，所以服务重启后仍能定位同一行。
        取摘要前 64 位并清掉最高位，得到正 63 位 ID；or 1 排除零。
        不使用 Python hash()，因为其结果可能随进程重启变化。
        """
        digest = hashlib.sha256(session_id.encode()).hexdigest()
        cid = (int(digest[:16], 16) & ((1 << 63) - 1)) or 1
        try:
            # sessions.begin() 同时创建 Session 和事务：正常退出提交，
            # 异常退出回滚，最后关闭 Session。每次操作独占自己的事务。
            async with self.db.sessions.begin() as session:
                row = await session.get(Conversation, cid)
                if row is None:
                    session.add(Conversation(id=cid, user_id=digest))
                elif row.user_id != digest:
                    # 截短摘要理论上可能碰撞，因此用完整摘要二次检查。
                    # 这里只区分演示会话，不能证明浏览器是真实登录用户。
                    raise ValueError('Session hash collision')
        except IntegrityError:
            # 并发首次创建只接受相同摘要，不掩盖其他约束错误。
            async with self.db.sessions() as session:
                row = await session.get(Conversation, cid)
                if row is None or row.user_id != digest:
                    raise
        return cid

    async def add_messages(self, cid, messages):
        async with self.db.sessions.begin() as session:
            rows = [encode_message(cid, message) for message in messages]
            session.add_all(rows)
            # flush 把待写入对象同步为 SQL，获得数据库生成的主键；
            # 它不等于 commit，后续异常仍可回滚这些写入。
            await session.flush()
            # 即使 return 写在 with 内，也会先完成上下文退出/提交再返回值。
            return rows[-1].id if rows else 0

    async def history(self, cid):
        """
        倒序取最近 200 条，再反转为时间正序。中途被截断的旧回合会跳过。
        这里先控制读库条数，service 中还会按上下文预算裁剪完整回合。
        """
        async with self.db.sessions() as session:
            rows = list((await session.scalars(select(Message).where(
                Message.conversation_id == cid).order_by(Message.id.desc()).limit(200))).all())
        # 只回放已经有最终 assistant 的完整回合；断线留下的流水仍可审计。
        complete, turn, pending = [], [], set()
        # turn 是当前临时回合；pending 是发出申请但尚未收到结果的调用 ID 集合。
        # 只有没有未配对申请、且见到最终 assistant，才纳入可回放历史。
        for row in reversed(rows):
            message = decode_message(row)
            if row.role == 'user':
                # 新用户消息意味着新一轮开始；上一轮若未结束就不回灌模型。
                # 原始流水依旧在数据库，此处只决定“哪些可以拿来作为历史”。
                turn, pending = [message], set()
            elif turn:
                turn.append(message)
                if isinstance(message, ToolMessage):
                    # 工具结果归还对应申请；discard 在集合没有该值时也不会报错。
                    pending.discard(message.tool_call_id)
                elif message.tool_calls:
                    pending.update(c['id'] for c in message.tool_calls)
                elif not pending:
                    # 此时是无 tool_calls 的 assistant，代表一轮完整回答结束。
                    complete.extend(turn)
                    turn = []
        return complete

    async def query_faq(self, keyword):
        """Ch03：dense 召回主键，再从 MySQL 读取原文，保留原返回契约。"""
        if self.knowledge is None:
            from app.knowledge.runtime import build_service
            self.knowledge = build_service(self.db)
        # 沿用keyword原话参数和四字段返回；向量服务故障交给原工具错误流程，不回退LIKE。
        return await self.knowledge.search(keyword)

    async def record_tool_result(self, cid, call_id, result, after_message_id=0):
        """
        只在当前会话、当前 assistant 申请之后查已有结果。
        call_id 来自上游，不能假设它在所有请求中全局唯一。
        """
        async with self.db.sessions.begin() as session:
            row = await session.scalar(select(Message).where(
                Message.conversation_id == cid, Message.tool_call_id == call_id,
                Message.role == 'tool', Message.id > after_message_id
            ).order_by(Message.id.desc()).limit(1))
            if row:
                # 写工具可能已经与工单一起存好了结果；直接复用，不重复加消息。
                return decode_message(row)
            message = ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call_id)
            # 模型协议要求 tool 结果带匹配的 tool_call_id；JSON 转成正文字符串保存。
            session.add(encode_message(cid, message))
            return message

    async def create_ticket(self, cid, description, ticket_type, ticket_no, call_id):
        """
        这个事务只包住业务写入和成功结果，不包住后续慢速模型回答。
        """
        async with self.db.sessions.begin() as session:
            existing = await session.get(Ticket, ticket_no)
            result = dict(ok=True, ticket_no=ticket_no, status='待处理')
            if existing:
                # 同一次工具执行的相同工单号不再插入。该机制不覆盖用户重发请求。
                if existing.conversation_id != cid:
                    raise ValueError('Ticket ownership mismatch')
                return result
            session.add(Ticket(ticket_no=ticket_no, conversation_id=cid,
                               description=description, ticket_type=ticket_type))
            # add 是登记 ORM 对象，不等于提交；后续查询可能触发 autoflush，
            # 但仍属于当前事务，只有整个 with 成功退出才提交。
            conversation = await session.get(Conversation, cid)
            conversation.status = '已转人工'
            # 工单和成功结果原子提交：模型回复失败也能追溯已经创建的工单。
            # 任意写入失败则整个事务回滚，避免只建工单却没有对应成功结果。
            session.add(encode_message(cid, ToolMessage(
                content=json.dumps(result, ensure_ascii=False), tool_call_id=call_id)))
            return result
