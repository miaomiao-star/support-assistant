import asyncio
from unittest.mock import AsyncMock

from langchain.tools import tool

from app.tools.registry import ToolRegistry


def registry(text='订单1001，查询退货政策，帮我创建人工工单', timeout=1):
    """
    AsyncMock 提供可 await 的仓库替身：验证工具路由而不访问真实数据库。
    """
    repo = AsyncMock()
    repo.query_faq.return_value = []
    repo.create_ticket.return_value = {'ok': True, 'ticket_no': 'T1', 'status': '待处理'}
    return ToolRegistry(repo, 123, text, timeout=timeout), repo


def test_five_tools_schemas_and_mock_data():
    """
    随机值不固定比较；检查 mock 标记、订单号透传、五工具注册和参数边界。
    """
    async def run():
        reg, _ = registry()
        assert set(reg.tools) == {'query_order', 'query_product', 'query_logistics', 'query_faq', 'create_ticket'}
        for name in ('query_order', 'query_logistics'):
            result = await reg.execute({'name': name, 'args': {'order_id': '1001'}, 'id': 'c'})
            assert result['ok'] and result['data']['mock']
            assert result['data']['order_id'] == '1001'
        product = await reg.execute({'name': 'query_product', 'args': {'keyword': '猫粮'}, 'id': 'p'})
        assert product['data']['mock']
        assert 'conversation_id' not in reg.tools['create_ticket'].args_schema.model_fields
    asyncio.run(run())


def test_invalid_unknown_and_literal_faq_keyword():
    """
    参数有效不等于业务允许：FAQ 还必须保留当前原话，不能擅自同义改写。
    """
    async def run():
        reg, repo = registry(text='邮费是多少')
        assert (await reg.execute({'name': 'shell', 'args': {}, 'id': 'x'}))['error'] == 'unknown_tool'
        assert (await reg.execute({'name': 'query_order', 'args': {'order_id': ''}, 'id': 'x'}))['error'] == 'invalid_arguments'
        assert (await reg.execute({'name': 'query_order', 'args': {'order_id': '1', 'sql': 'bad'}, 'id': 'x'}))['error'] == 'invalid_arguments'
        assert (await reg.execute({'name': 'query_faq', 'args': {'keyword': '运费'}, 'id': 'x'}))['error'] == 'keyword_not_literal'
        result = await reg.execute({'name': 'query_faq', 'args': {'keyword': '邮费'}, 'id': 'f'})
        assert result['ok'] and result['data'] == []
        repo.query_faq.assert_awaited_once_with('邮费')
    asyncio.run(run())


def test_read_timeout_retries_once_and_cleans_up():
    """
    工具故意等得比超时长：确认总尝试 2 次，而且两次都进入 finally 清理。
    """
    async def run():
        reg, _ = registry(timeout=.01)
        attempts, closed = [], []
        @tool
        async def slow() -> dict:
            """等待测试。"""
            attempts.append(1)
            try:
                await asyncio.sleep(1)
            finally:
                closed.append(1)
        reg.register(slow)
        result = await reg.execute({'name': 'slow', 'args': {}, 'id': 's'})
        assert result['error'] == 'timeout'
        assert len(attempts) == len(closed) == 2
    asyncio.run(run())


def test_write_timeout_never_blindly_retries_and_requires_explicit_request():
    """
    写操作与只读的重试策略不同；没有人工/工单关键词时不能调用写库函数。
    """
    async def run():
        reg, repo = registry(timeout=.01)
        async def wait(*args):
            await asyncio.sleep(1)
        repo.create_ticket.side_effect = wait
        result = await reg.execute({'name': 'create_ticket', 'args': {'description': '商品损坏', 'ticket_type': '售后'}, 'id': 't'})
        assert result['error'] == 'timeout'
        assert repo.create_ticket.await_count == 1
        reg, repo = registry(text='退货政策是什么')
        result = await reg.execute({'name': 'create_ticket', 'args': {'description': '商品损坏', 'ticket_type': '售后'}, 'id': 't'})
        assert result['error'] == 'confirmation_required'
        repo.create_ticket.assert_not_awaited()
    asyncio.run(run())


def test_errors_are_safe_and_cancel_propagates():
    """
    普通故障返回安全错误码；取消信号继续传播，不能被吞掉后假装执行完成。
    """
    async def run():
        reg, repo = registry()
        repo.query_faq.side_effect = RuntimeError('secret-do-not-expose')
        result = await reg.execute({'name': 'query_faq', 'args': {'keyword': '退货政策'}, 'id': 'f'})
        assert result == {'ok': False, 'error': 'execution_failed'}
        repo.query_faq.side_effect = asyncio.CancelledError()
        try:
            await reg.execute({'name': 'query_faq', 'args': {'keyword': '退货政策'}, 'id': 'f'})
            assert False
        except asyncio.CancelledError:
            pass
    asyncio.run(run())
