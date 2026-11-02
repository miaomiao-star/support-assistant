"""Ch07会话适配：规范图线程、原文持久化、摘要生命周期和显式动作。"""

from uuid import uuid4
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.workflow.routing_service import RoutingService
from app.workflow.conversation import Context, history_view
from app.tools.registry import ToolRegistry


class ConversationService(RoutingService):
    # 沿用Ch06的SSE事件/选单/显式确认协议；覆盖会话定位及完整消息准备。
    def __init__(self, *args, summaries=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.summaries = summaries

    async def context(self, session_id, user_text):
        """每个HTTP执行段创建独立仓库上下文和可变工具注册表，不把连接放入State。"""
        cid = await self.repo.open_conversation(session_id)
        registry = ToolRegistry(
            self.repo, cid, user_text, timeout=self.settings.tool_timeout_seconds
        )
        return Context(
            self.repo,
            self.model,
            self.rag,
            self.settings,
            registry,
            cid,
            self.classifier,
            self.understander,
            self.expander,
            self.small_model,
            self.summaries,
        )

    async def canonical(self, body):
        """先核验归属再复制请求为规范ID，让UUID入口和侧栏入口命中同一图线程。"""
        cid = await self.repo.open_conversation(body.session_id)
        return body.model_copy(update={'session_id': f'conv7:{cid}'})

    async def stream(self, session_id, user_text, filters=None):
        """保存当前原话，向完整State追加新消息，再转发图事件并检查摘要触发。"""
        context = await self.context(session_id, user_text)
        canonical = f'conv7:{context.conversation_id}'
        config = self.config(canonical)
        snapshot = await self.graph.aget_state(config)
        if snapshot.next:
            # 新问题可取消尚未选单的旧轮；其他未完成节点不能被默默跳过当作已恢复。
            if snapshot.next == ('pick_order',):
                async for _ in self.run(
                    canonical, Command(resume={'cancelled': True}), context
                ):
                    pass
            else:
                raise RuntimeError('Previous graph execution incomplete')
        # 独立确认可能在暂停期间写SQL审计；下次执行按稳定SQL ID补入，不丢原State工具。
        known = {
            message.additional_kwargs.get('sql_msg_id')
            for message in snapshot.values.get('messages', [])
        }
        bootstrap = [
            message
            for message in await self.repo.raw_history(context.conversation_id)
            if message.additional_kwargs.get('sql_msg_id') not in known
        ]
        # 仅追加SQL中State尚未见过的行；SQL没有新工具正文，所以不能用它替换已有State。
        human = HumanMessage(content=user_text)
        message_id = await self.repo.add_messages(context.conversation_id, [human])
        human.id = f'm:{message_id}'
        # reducer身份(id)与分层坐标(sql_msg_id)分开；两者都需稳定，前者不能每次重新随机。
        human.additional_kwargs['sql_msg_id'] = message_id
        yield dict(
            type='conversation', id=str(context.conversation_id), session_id=canonical
        )
        # conversation事件告诉页面本轮所属cid；正式回答/工具/选单事件由父类run统一转发。
        async for item in self.run(
            canonical,
            dict(
                messages=[*bootstrap, human],
                current_user_id=message_id,
                raw_question=user_text,
                request_id=uuid4().hex,
                filters=filters or {},
            ),
            context,
        ):
            yield item
        state = (await self.graph.aget_state(config)).values
        if not (await self.graph.aget_state(config)).next:
            # 轮末把刚完成的原文也纳入预算。这里只安排任务，不等待后台摘要模型返回。
            await history_view(state, context, completed=True)

    async def resume(self, body):
        """恢复同一暂停轮；重复/失效选择的核验仍由Ch06父类执行。"""
        body = await self.canonical(body)
        async for item in super().resume(body):
            yield item
        snapshot = await self.graph.aget_state(self.config(body.session_id))
        if not snapshot.next:
            # 成功续跑后也检查层级；仍在暂停时不把未完成轮当作最终答复处理。
            context = await self.context(
                body.session_id, snapshot.values.get('raw_question', '')
            )
            await history_view(snapshot.values, context, completed=True)

    async def confirm_refund(self, body):
        # 只统一定位键，退款保存边界和幂等契约仍复用父类，不额外调用模型。
        return await super().confirm_refund(await self.canonical(body))

    async def confirm_ticket(self, body):
        # 旧确认服务的history字段不用于Ch07模型；新增SQL审计随后并入完整State。
        body = await self.canonical(body)
        result = await super().confirm_ticket(body)
        config = self.config(body.session_id)
        snapshot = await self.graph.aget_state(config)
        if not snapshot.next:
            # as_node='log'会影响图后继，只能在已结束轮回写；暂停中留待下次stream补SQL审计。
            cid = await self.repo.open_conversation(body.session_id)
            known = {
                message.additional_kwargs.get('sql_msg_id')
                for message in snapshot.values.get('messages', [])
            }
            additions = [
                message
                for message in await self.repo.raw_history(cid)
                if message.additional_kwargs.get('sql_msg_id') not in known
            ]
            await self.graph.aupdate_state(
                config, dict(messages=additions), as_node='log'
            )
        return result
