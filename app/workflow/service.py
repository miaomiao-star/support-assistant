"""HTTP 与图之间的适配；连接和模型走 Runtime context，绝不进 State。"""

import asyncio
import hashlib
from contextlib import aclosing
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from app.tools.registry import ToolRegistry
from app.workflow.graph import Context
from app.workflow.protocol import OPTIONS


class WorkflowService:
    def __init__(self, graph, repo, model, settings, rag=None, classifier=None):
        """绑定应用共享图与本次请求依赖；对象本身不承担持久化。"""
        self.graph, self.repo, self.model = graph, repo, model
        self.settings, self.rag, self.classifier = settings, rag, classifier

    @staticmethod
    def config(session_id):
        """生成稳定图线程标识及图执行步数保险，不把它当登录身份。"""
        # 不把浏览器提供的会话原值写入图线程标识或日志。
        return {
            "configurable": {
                "thread_id": hashlib.sha256(session_id.encode()).hexdigest()
            },
            # 图节点步数不同于Agent模型决策数；后者另由agent_max_steps限制。
            "recursion_limit": 64,
        }

    async def stream(self, session_id, user_text, filters=None):
        """准备本轮执行上下文并转发内部事件，成功落盘后再发可点击建议。"""
        cid = await self.repo.open_conversation(session_id)
        # 注册表带可变call_id/request_key，必须每轮新建，不能跨请求共享。
        registry = ToolRegistry(
            self.repo, cid, user_text, timeout=self.settings.tool_timeout_seconds
        )
        context = Context(
            self.repo,
            self.model,
            self.rag,
            self.settings,
            registry,
            cid,
            self.classifier,
        )
        # 原始输入先存MySQL以便审计；后续失败不等于用户没有发过这条消息。
        # 是否允许下轮回放，由图中的完整history与原仓库完整回合规则决定。
        await self.repo.add_messages(cid, [HumanMessage(content=user_text)])
        config = self.config(session_id)
        # 总超时涵盖分类、检索、所有模型/工具步；取消继续向外传播。
        async with asyncio.timeout(self.settings.workflow_timeout_seconds):
            async with aclosing(
                self.graph.astream(
                    dict(
                        raw_question=user_text,
                        # 同一个会话的thread_id不变，每一轮request_id则不同。
                        request_id=uuid4().hex,
                        filters=filters or {},
                    ),
                    config,
                    context=context,
                    # 这里只接收节点writer发出的事件，不把State整包暴露给浏览器。
                    stream_mode="custom",
                )
            ) as events:
                async for event in events:
                    yield event
            # 只有 checkpoint 成功落盘以后才给可点击选项，失败轮不能凭空建单。
            state = (await self.graph.aget_state(config)).values
            offer_id = state.get("last_offer")
            if offer_id:
                offer = state["offers"][offer_id]
                yield dict(
                    type="actions",
                    offer_id=offer_id,
                    options=[
                        dict(action=name, label=OPTIONS[name])
                        for name in offer["actions"]
                    ],
                    description=offer["description"],
                    ticket_type=offer["ticket_type"],
                )

    async def confirm_ticket(self, body):
        """仅由确认接口调用；稳定建议 ID 让写成功但响应丢失时可安全重试。"""
        config = self.config(body.session_id)
        # 重新从服务端checkpoint查建议，不信任前端自己拼的动作或会话归属。
        # API层和普通聊天使用同一把会话锁，避免本进程内边聊天边确认覆盖状态。
        snapshot = await self.graph.aget_state(config)
        state = snapshot.values
        offers = dict(state.get("offers", {}))
        offer = offers.get(body.offer_id)
        if not offer or "create_ticket" not in offer["actions"]:
            raise LookupError("No ticket offer")
        if offer.get("result"):
            # 上次已确认成功，直接返回原结果；再次编辑描述不会改写已创建的工单。
            return offer["result"]
        cid = await self.repo.open_conversation(body.session_id)
        if offer["conversation_id"] != cid:
            raise LookupError("Offer conversation mismatch")
        # “确认建工单”来自已校验的用户点击，不能由 Agent 的工具申请触发。
        registry = ToolRegistry(
            self.repo,
            cid,
            "用户确认建工单：" + body.description,
            timeout=self.settings.tool_timeout_seconds,
        )
        # 普通请求默认生成随机request_key；这里改用稳定offer_id，才能跨重试同号。
        # 既有ToolRegistry会用会话/request_key/call_id共同派生ticket_no。
        registry.request_key = body.offer_id
        call = dict(
            name="create_ticket",
            id="ticket_" + body.offer_id,
            args=dict(description=body.description, ticket_type=body.ticket_type),
        )
        # 确认操作也形成完整消息链：用户确认 → assistant申请 → tool结果 → 回执。
        # assistant申请由后端构造，并没有为建单再调用一次模型。
        human = HumanMessage(content="我确认创建工单：" + body.description)
        request = AIMessage(content="", tool_calls=[call])
        after_id = await self.repo.add_messages(cid, [human, request])
        result = await registry.execute(call)
        # 旧工单工具会将工单和成功ToolMessage同事务提交；这里优先取回同一条观察。
        tool_message = await self.repo.record_tool_result(
            cid, call["id"], result, after_message_id=after_id
        )
        if not result.get("ok"):
            raise RuntimeError("Ticket creation unconfirmed")
        receipt = AIMessage(content=f'已创建待处理工单 {result["ticket_no"]}。')
        await self.repo.add_messages(cid, [receipt])
        if snapshot.next:
            # Ch06可在新轮等待选单时确认旧工单建议；业务与完整审计已写MySQL。
            # 此处不推进或覆盖新轮的图暂停节点，同offer重试仍由稳定工单号兜住。
            return result
        offers[body.offer_id] = {**offer, "result": result}
        # 不因写入工单切换对话模式，也不自动触发下一次 Agent。
        # MySQL已提交后才写Postgres，这不是跨库事务；回写失败时同offer重试仍同号。
        await self.graph.aupdate_state(
            config,
            dict(
                offers=offers,
                history=[
                    *state.get("history", []),
                    human,
                    request,
                    tool_message,
                    receipt,
                ],
            ),
            # 将本次确认视为终点状态更新，不重新执行route/agent/tools节点。
            as_node="log",
        )
        return result
