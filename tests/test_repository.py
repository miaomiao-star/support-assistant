import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import func, select

from app.db import Database
from app.models import Base, FAQ, Ticket
from app.repositories.chat import ChatRepository


async def repository():
    """
    每个用例创建独立的内存数据库，不读取 .env.mysql，也不修改 Docker MySQL。
    create_all 只出现在测试准备中；真实表仍由 sql/ 下的脚本初始化。
    """
    db = Database('sqlite+aiosqlite://')
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with db.sessions.begin() as session:
        session.add_all([
            FAQ(question='退货政策', answer='七天退货，以平台规则为准', category='售后'),
            FAQ(question='运费怎么算', answer='满99包邮', category='物流'),
        ])
    return db, ChatRepository(db)


def test_faq_uses_semantic_service_without_changing_contract():
    async def run():
        from unittest.mock import AsyncMock
        db, repo = await repository()
        expected=[dict(id=10,question='运费怎么算',answer='满99包邮',category='物流')]
        service=AsyncMock()
        service.search.return_value=expected
        repo.knowledge=service
        try:
            assert await repo.query_faq('邮费') == expected
            service.search.assert_awaited_once_with('邮费')
            service.search.side_effect=ConnectionError('unavailable')
            import pytest
            with pytest.raises(ConnectionError): await repo.query_faq('退货政策')
        finally: await db.close()
    asyncio.run(run())


def test_persisted_history_survives_new_repository_and_skips_incomplete_turn():
    """
    更换仓库实例模拟“没有旧 Python 对象，只剩数据库”的恢复场景。
    同时写入一轮未完成工具申请，验证它仍存库但不会进入可回放历史。
    """
    async def run():
        db, repo = await repository()
        try:
            cid = await repo.open_conversation('same-session')
            assert cid == await ChatRepository(db).open_conversation('same-session')
            call = {'name': 'query_faq', 'args': {'keyword': '退货政策'}, 'id': 'c1', 'type': 'tool_call'}
            complete = [HumanMessage(content='退货政策'), AIMessage(content='', tool_calls=[call]),
                        ToolMessage(content='七天退货', tool_call_id='c1'), AIMessage(content='答复')]
            await repo.add_messages(cid, complete)
            await repo.add_messages(cid, [HumanMessage(content='中途断线'), AIMessage(content='', tool_calls=[call])])
            restored = await ChatRepository(db).history(cid)
            assert [m.type for m in restored] == ['human', 'ai', 'tool', 'ai']
            assert restored[1].tool_calls[0]['id'] == restored[2].tool_call_id
            assert await repo.history(await repo.open_conversation('other')) == []
        finally:
            await db.close()
    asyncio.run(run())


def test_ticket_and_tool_result_atomic_and_idempotent():
    """
    同一工单号执行两次仅保留一张工单；已经落库的成功结果不能被超时误报覆盖。
    """
    async def run():
        db, repo = await repository()
        try:
            cid = await repo.open_conversation('ticket-session')
            first = await repo.create_ticket(cid, '商品损坏，请人工处理', '售后', 'Ttest001', 'call-ticket')
            second = await repo.create_ticket(cid, '商品损坏，请人工处理', '售后', 'Ttest001', 'call-ticket')
            assert first == second
            msg = await repo.record_tool_result(cid, 'call-ticket', {'ok': False, 'error': 'timeout'})
            assert json.loads(msg.content)['ticket_no'] == 'Ttest001'
            async with db.sessions() as session:
                assert await session.scalar(select(func.count()).select_from(Ticket)) == 1
            assert first['status'] == '待处理'
        finally:
            await db.close()
    asyncio.run(run())


def test_reused_upstream_call_id_cannot_return_previous_turn_result():
    """
    故意让两轮共用同一个上游 call_id；本次 assistant 主键应隔离两轮结果。
    """
    async def run():
        db, repo = await repository()
        try:
            cid = await repo.open_conversation('repeat-id')
            call = {'name': 'query_order', 'args': {'order_id': '1'}, 'id': 'same'}
            anchor1 = await repo.add_messages(cid, [HumanMessage(content='查1'), AIMessage(content='', tool_calls=[call])])
            await repo.record_tool_result(cid, 'same', {'ok': True, 'data': 'first'}, after_message_id=anchor1)
            await repo.add_messages(cid, [AIMessage(content='第一轮')])
            anchor2 = await repo.add_messages(cid, [HumanMessage(content='查2'), AIMessage(content='', tool_calls=[call])])
            result = await repo.record_tool_result(cid, 'same', {'ok': True, 'data': 'second'}, after_message_id=anchor2)
            assert json.loads(result.content)['data'] == 'second'
        finally:
            await db.close()
    asyncio.run(run())


def test_ticket_rolls_back_when_audit_result_cannot_be_written(monkeypatch):
    """
    人为让成功结果编码失败，检查整个事务回滚，而不是留下没有审计的工单。
    """
    async def run():
        db, repo = await repository()
        try:
            cid = await repo.open_conversation('rollback')
            def fail(*args):
                raise RuntimeError('audit write failed')
            monkeypatch.setattr('app.repositories.chat.encode_message', fail)
            try:
                await repo.create_ticket(cid, '损坏', '售后', 'Trollback', 'c')
                assert False
            except RuntimeError:
                pass
            async with db.sessions() as session:
                assert await session.get(Ticket, 'Trollback') is None
        finally:
            await db.close()
    asyncio.run(run())
