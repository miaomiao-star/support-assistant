import asyncio
import json
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessageChunk, ToolMessage

from app.services.chat import ChatService


class Model:
    # 假模型分开模拟“绑定工具的选择流”和“原模型的最终回答流”。
    # 这样能断言模型调用次数以及回灌消息，不消耗真实模型费用。
    def __init__(self, calls=None, fail=False):
        self.tool_calls = calls or []
        self.bound_count = self.final_count = 0
        self.prompts = []
        self.fail = fail
        self.closed = False

    def bind_tools(self, tools, **kwargs):
        self.bound_count += 1
        outer = self
        class Bound:
            # 本测试直接返回完整申请；真实网络中参数还可能分段到达。
            async def astream(self, prompt):
                outer.prompts.append(prompt.to_messages())
                yield AIMessageChunk(content='' if outer.tool_calls else '你好', tool_calls=outer.tool_calls)
        return Bound()

    async def astream(self, prompt):
        self.final_count += 1
        self.prompts.append(prompt.to_messages())
        try:
            yield AIMessageChunk(content='根据查询结果')
            if self.fail:
                raise RuntimeError('sensitive-value')
            yield AIMessageChunk(content='回答')
        finally:
            self.closed = True


def dependencies():
    """
    仓库是异步替身；record 还原实际返回类型，便于测试消息配对。
    """
    repo = AsyncMock()
    repo.open_conversation.return_value = 1
    repo.history.return_value = []
    repo.query_faq.return_value = []
    async def record(cid, call_id, result, **kwargs):
        return ToolMessage(content=json.dumps(result), tool_call_id=call_id)
    repo.record_tool_result.side_effect = record
    return repo


def service(repo, model):
    return ChatService(repo, model, text_budget=2000, tool_budget=8000, chars_per_token=1.2, tool_timeout=1)


def test_single_tool_status_result_and_streamed_final():
    """
    验证正常主线：选择一次、执行一次、running/success、结果回灌、最终文本落库。
    """
    async def run():
        repo = dependencies()
        model = Model([{'name': 'query_logistics', 'args': {'order_id': '1001'}, 'id': 'c'}])
        events = [e async for e in service(repo, model).stream('s', '订单1001物流到哪了')]
        assert [(e['status']) for e in events if e['type'] == 'tool'] == ['running', 'success']
        assert ''.join(e['delta'] for e in events if e['type'] == 'delta') == '根据查询结果回答'
        assert model.bound_count == model.final_count == 1
        assert [m.type for m in model.prompts[1]][-2:] == ['ai', 'tool']
        assert model.prompts[1][-1].tool_call_id == 'c'
        repo.record_tool_result.assert_awaited_once()
        assert repo.add_messages.await_count == 3  # user、assistant 工具请求、完整最终回复
    asyncio.run(run())


def test_multiple_calls_rejected_without_execution_and_no_loop():
    """
    上游即使忽略禁并行参数，也不能突破后端“最多一个工具”的规则。
    """
    async def run():
        repo = dependencies()
        model = Model([{'name': 'query_faq', 'args': {'keyword': '退货'}, 'id': 'a'},
                       {'name': 'query_faq', 'args': {'keyword': '邮费'}, 'id': 'b'}])
        events = [e async for e in service(repo, model).stream('s', '退货和邮费')]
        repo.query_faq.assert_not_awaited()
        assert model.final_count == 1
        assert len([e for e in events if e.get('status') == 'error']) == 2
        assert repo.record_tool_result.await_count == 2
    asyncio.run(run())


def test_final_failure_keeps_tool_audit_but_not_partial_final():
    """
    模拟工具已完成后最终模型流失败：审计保留，半截最终回复不保存。
    """
    async def run():
        repo = dependencies()
        model = Model([{'name': 'query_faq', 'args': {'keyword': '邮费'}, 'id': 'c'}], fail=True)
        try:
            _ = [e async for e in service(repo, model).stream('s', '邮费是多少')]
            assert False
        except RuntimeError:
            pass
        assert model.closed
        assert repo.add_messages.await_count == 2
        repo.record_tool_result.assert_awaited_once()
    asyncio.run(run())


def test_cancel_closes_upstream_and_does_not_save_partial_final():
    """
    手动 aclose 模拟消费者提前离开，检查 aclosing 是否传递到模型流。
    """
    async def run():
        repo = dependencies()
        model = Model([{'name': 'query_faq', 'args': {'keyword': '邮费'}, 'id': 'c'}])
        iterator = service(repo, model).stream('s', '邮费是多少')
        while (await anext(iterator))['type'] != 'delta':
            pass
        await iterator.aclose()
        assert model.closed
        assert repo.add_messages.await_count == 2
    asyncio.run(run())


def test_duplicate_call_ids_are_rejected_before_persisting_tool_request():
    """
    同一回复中重复 ID 会造成协议歧义，应在任何业务执行前拒绝。
    """
    async def run():
        repo = dependencies()
        model = Model([{'name': 'query_faq', 'args': {'keyword': '退货'}, 'id': 'same'},
                       {'name': 'query_faq', 'args': {'keyword': '邮费'}, 'id': 'same'}])
        try:
            _ = [e async for e in service(repo, model).stream('s', '退货和邮费')]
            assert False, 'duplicate ids must fail'
        except ValueError:
            pass
        repo.query_faq.assert_not_awaited()
        assert repo.add_messages.await_count == 1
    asyncio.run(run())


def test_model_requesting_second_tool_is_never_executed():
    """
    第二次模型本应只回答；即使返回了工具申请，也不能再执行写操作。
    """
    async def run():
        repo = dependencies()
        class SecondToolModel(Model):
            async def astream(self, prompt):
                yield AIMessageChunk(content='', tool_calls=[{'name':'create_ticket',
                    'args':{'description':'再次写入', 'ticket_type':'咨询'}, 'id':'second'}])
        model = SecondToolModel([{'name': 'query_faq', 'args': {'keyword': '邮费'}, 'id': 'first'}])
        try:
            _ = [e async for e in service(repo, model).stream('s', '邮费是多少')]
            assert False
        except ValueError:
            pass
        repo.create_ticket.assert_not_awaited()
        assert repo.query_faq.await_count == 1
    asyncio.run(run())
