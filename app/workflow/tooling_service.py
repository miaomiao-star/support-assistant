"""统一工具图的HTTP适配：工单预览确认独立于旧建议按钮。"""

from types import SimpleNamespace
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage
from app.workflow.conversation_service import ConversationService
from app.workflow import tooling
from app.context.budget import ContextBudget, count_tokens, text_tokens
from app.tools.engine import Approval
import json


class ToolingService(ConversationService):
    # 继承会话锁/三层投影/持久化适配，仅增加动态目录和新工单确认入口。
    def __init__(self, *args, tools, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_runtime = tools
        # 新章实际工具前缀独立预留，沿用原公式，不修改旧章预算测试或私有env。
        self.settings = self.settings.model_copy(
            update={'context_system_reserve': self.settings.tool_system_reserve}
        )
        ContextBudget.from_settings(self.settings)

    async def context(self, session_id, user_text):
        context = await super().context(session_id, user_text)
        context.settings = self.settings
        context.registry = await self.tool_runtime.registry(
            self.repo, context.conversation_id, user_text, self.settings
        )
        context.registry.system_prompt = tooling.SYSTEM
        # 热新增工具会扩大Schema；每段都核验真实前缀，不能只相信固定预留数。
        prefix = count_tokens(
            [AIMessage(content=tooling.SYSTEM)], self.settings
        ) + text_tokens(
            json.dumps(context.registry.model_tools(), ensure_ascii=False),
            self.settings,
        )
        if prefix > self.settings.context_system_reserve:
            raise ValueError(
                '动态工具前缀超过TOOL_SYSTEM_RESERVE，请校准预算或精简Schema'
            )
        return context

    async def run(self, session_id, value, context):
        async for event in super().run(session_id, value, context):
            yield event
        snapshot = await self.graph.aget_state(self.config(session_id))
        # interrupt已保存到checkpoint后再发卡片；HTTP连接无需一直等用户点击。
        if snapshot.next == ('ticket_preview',):
            yield dict(type='ticket_preview', **snapshot.values['ticket_preview'])

    async def stream(self, session_id, user_text, filters=None):
        cid = await self.repo.open_conversation(session_id)
        canonical = f'conv7:{cid}'
        snapshot = await self.graph.aget_state(self.config(canonical))
        if snapshot.next == ('ticket_preview',):
            # 新话题显式结束旧申请，取消本身进入引擎拒绝审计，不能偷偷丢掉旧调用。
            preview = snapshot.values['ticket_preview']
            async for _ in self.resume_ticket(
                SimpleNamespace(
                    session_id=canonical,
                    preview_id=preview['preview_id'],
                    confirmed=False,
                )
            ):
                pass
        async for event in super().stream(session_id, user_text, filters):
            yield event

    async def resume_ticket(self, body):
        cid = await self.repo.open_conversation(body.session_id)
        canonical = f'conv7:{cid}'
        # 旧UUID与规范conv7入口统一到同一PG线程；归属/加锁在外层API处理。
        config = self.config(canonical)
        snapshot = await self.graph.aget_state(config)
        state = snapshot.values
        completed = state.get('completed_tickets', {}).get(body.preview_id)
        if completed:
            # 双击/响应丢失后的同值确认返回原结果；改变选择不能撤销已处理业务。
            if completed['confirmed'] != body.confirmed:
                raise LookupError('工单已按另一选择处理')
            if snapshot.next in (('agent',), ('log',)) and body.preview_id.startswith(
                state['request_id'] + ':'
            ):
                # 业务已完成而回执生成/保存失败：只续后继节点，不能停留在半轮状态，
                # 也不能再次执行ticket_execute或覆盖另一轮的暂停位置。
                context = await self.context(canonical, state['raw_question'])
                async for event in self.run(canonical, None, context):
                    yield event
                return
            yield dict(type='delta', delta=completed['answer'])
            return
        preview = state.get('ticket_preview')
        # 仅允许当前预览及其后继恢复；不会拿旧凭据推进其他轮的订单选择等暂停。
        if (
            not preview
            or preview['preview_id'] != body.preview_id
            or snapshot.next
            not in (('ticket_preview',), ('ticket_execute',), ('agent',), ('log',))
        ):
            raise LookupError('工单预览已失效')
        if (
            state.get('ticket_decision') is not None
            and state['ticket_decision'] != body.confirmed
        ):
            raise LookupError('工单已经按另一选择开始处理')
        context = await self.context(canonical, state['raw_question'])
        value = (
            # 只有仍在interrupt时传Command；已过确认节点的失败从checkpoint继续。
            Command(resume={'preview_id': body.preview_id, 'confirmed': body.confirmed})
            if snapshot.next == ('ticket_preview',)
            else None
        )
        async for event in self.run(canonical, value, context):
            yield event

    async def confirm_ticket(self, body):
        """旧投诉按钮仍一次确认；复用统一引擎并保留稳定offer编号。"""
        cid = await self.repo.open_conversation(body.session_id)
        canonical = f'conv7:{cid}'
        config = self.config(canonical)
        snapshot = await self.graph.aget_state(config)
        offers = dict(snapshot.values.get('offers', {}))
        offer = offers.get(body.offer_id)
        if (
            not offer
            or offer['conversation_id'] != cid
            or 'create_ticket' not in offer['actions']
        ):
            raise LookupError('没有该工单建议')
        if offer.get('result'):
            return offer['result']
        context = await self.context(canonical, body.description)
        registry = context.registry
        registry.request_key = body.offer_id
        # 旧按钮本身已是一次确认，沿offer生成稳定调用；不再进入新预览二次确认。
        registry.ticket_requested = True
        call = dict(
            name='create_ticket',
            id='ticket_' + body.offer_id,
            args=dict(description=body.description, ticket_type=body.ticket_type),
        )
        human = HumanMessage(content='我确认创建工单：' + body.description)
        request = AIMessage(content='', tool_calls=[call])
        after_id = await self.repo.add_messages(cid, [human, request])
        result = await registry.execute(
            call, approval=Approval.for_call(registry.engine().context, call)
        )
        if not result.get('ok'):
            raise RuntimeError('工单写入结果未确认')
        await self.repo.add_messages(
            cid, [AIMessage(content=f'已创建待处理工单 {result["ticket_no"]}。')]
        )
        if not snapshot.next:
            # 当前没有暂停才把按钮结果同步进图；正在等订单/工单选择时保持原暂停。
            offers[body.offer_id] = {**offer, 'result': result}
            known = {
                m.additional_kwargs.get('sql_msg_id')
                for m in snapshot.values.get('messages', [])
            }
            additions = [
                m
                for m in await self.repo.raw_history(cid)
                if m.additional_kwargs.get('sql_msg_id') not in known
            ]
            await self.graph.aupdate_state(
                config, dict(offers=offers, messages=additions), as_node='log'
            )
        return result
