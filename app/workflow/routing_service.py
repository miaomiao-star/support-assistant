"""正式图适配：聊天启动、选单恢复、演示退款确认各自具有明确凭据。"""

import asyncio
from contextlib import aclosing
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.workflow.service import WorkflowService
from app.workflow.routing import Context
from app.workflow.protocol import OPTIONS
from app.tools.registry import ToolRegistry
from app.refunds import RefundStore, REASONS


class RoutingService(WorkflowService):
    def __init__(
        self, *args, understander=None, expander=None, small_model=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.understander, self.expander, self.small_model = (
            understander,
            expander,
            small_model,
        )

    async def context(self, session_id, user_text):
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
        )

    async def run(self, session_id, value, context):
        """执行一个有限HTTP段；等待用户时图暂停并结束网络流，不持锁等待点击。"""
        config = self.config(session_id)
        async with asyncio.timeout(self.settings.workflow_timeout_seconds):
            async with aclosing(
                self.graph.astream(value, config, context=context, stream_mode="custom")
            ) as events:
                async for event in events:
                    yield event
            snapshot = await self.graph.aget_state(config)
            state = snapshot.values
            if snapshot.next and state.get("selection"):
                yield dict(type="delta", delta="请选择需要处理的演示订单。")
                yield dict(type="order_selection", **state["selection"])
                return
            if state.get("last_offer"):
                offer_id = state["last_offer"]
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
            if state.get("last_refund_offer"):
                offer_id = state["last_refund_offer"]
                yield dict(
                    type="refund_offer",
                    offer_id=offer_id,
                    order=state["refund_offers"][offer_id]["order"],
                    reasons=REASONS,
                )

    async def stream(self, session_id, user_text, filters=None):
        context = await self.context(session_id, user_text)
        config = self.config(session_id)
        snapshot = await self.graph.aget_state(config)
        if snapshot.next and snapshot.values.get("selection"):
            # 新话题取消旧等待，再作为新一轮输入；不能让用户文本误当选单结果。
            if snapshot.next == ("pick_order",):
                async for _ in self.run(
                    session_id, Command(resume={"cancelled": True}), context
                ):
                    pass
            else:
                await self.graph.aupdate_state(
                    config,
                    dict(selection=None, stop_reason="selection_cancelled"),
                    as_node="log",
                )
        await self.repo.add_messages(
            context.conversation_id, [HumanMessage(content=user_text)]
        )
        async for event in self.run(
            session_id,
            dict(raw_question=user_text, request_id=uuid4().hex, filters=filters or {}),
            context,
        ):
            yield event

    async def resume(self, body):
        """服务端先核验候选与轮次；重复已完成选择返回原结果，不重跑模型。"""
        snapshot = await self.graph.aget_state(self.config(body.session_id))
        state = snapshot.values
        selection = state.get("selection")
        complete = state.get("completed_selections", {}).get(body.selection_id)
        if complete:
            if complete["order_id"] != body.order_id:
                raise LookupError("Selection already completed")
            yield dict(type="delta", delta=complete["answer"])
            offer_id = complete.get("refund_offer_id")
            if offer_id and offer_id in state.get("refund_offers", {}):
                yield dict(
                    type="refund_offer",
                    offer_id=offer_id,
                    order=state["refund_offers"][offer_id]["order"],
                    reasons=REASONS,
                )
            return
        if (
            not snapshot.next
            or not selection
            or selection["selection_id"] != body.selection_id
        ):
            raise LookupError("No pending order selection")
        if body.order_id not in {order["order_id"] for order in selection["orders"]}:
            raise LookupError("Order not offered")
        if (
            state.get("selected_order_id")
            and state["selected_order_id"] != body.order_id
        ):
            raise LookupError("Selection already started")
        context = await self.context(body.session_id, state["raw_question"])
        value = (
            Command(resume=dict(selection_id=body.selection_id, order_id=body.order_id))
            if snapshot.next == ("pick_order",)
            else None
        )
        async for event in self.run(body.session_id, value, context):
            yield event

    async def confirm_refund(self, body):
        """显式确认才写申请；订单快照来自服务端建议，前端不能改订单或金额。"""
        config = self.config(body.session_id)
        snapshot = await self.graph.aget_state(config)
        state = snapshot.values
        offers = dict(state.get("refund_offers", {}))
        offer = offers.get(body.offer_id)
        cid = await self.repo.open_conversation(body.session_id)
        if not offer or offer["conversation_id"] != cid:
            raise LookupError("No refund offer")
        if offer.get("result"):
            return offer["result"]
        result = await RefundStore(self.repo.db).submit(
            cid, body.offer_id, offer["order"], body.reason, body.description
        )
        if snapshot.next:
            # 用户可在另一轮等选单时确认旧建议。此时不能以log节点回写并覆盖暂停位置。
            # MySQL稳定主键仍保证重复确认幂等；结果缓存延后，不影响业务申请已保存。
            return result
        offers[body.offer_id] = {**offer, "result": result}
        # 两次独立提交不假装跨库事务，MySQL稳定主键承担回写失败的重试幂等。
        await self.graph.aupdate_state(
            config, dict(refund_offers=offers), as_node="log"
        )
        return result
