import asyncio
import json
import logging
import re
from contextlib import aclosing
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.core.llm import get_chat_model
from app.core.memory import BudgetExceeded, prepare_messages
from app.core.prompts import CUSTOMER_SERVICE_SYSTEM
from app.repositories.chat import ChatRepository
from app.services.chat import ChatService
from app.schemas.chat import ChatRequest
from app.schemas.actions import TicketConfirmation
from app.workflow.service import WorkflowService
from app.workflow.graph import AGENT_SYSTEM
from app.workflow.routing_service import RoutingService
from app.refunds import OrderSelection, RefundSubmission
from app.context.store import ContextRepository
from app.context.budget import text_tokens, ContextBudget
from app.workflow.conversation_service import ConversationService
from app.workflow.tooling_service import ToolingService
from app.schemas.tooling import TicketDecision

router = APIRouter()
logger = logging.getLogger(__name__)


def workflow_service(request, repo, model, settings):
    """正式图使用可恢复选单服务；历史隔离图显式使用旧服务。"""
    graph = request.app.state.workflow
    arguments = dict(
        rag=getattr(request.app.state, 'rag', None),
        classifier=getattr(request.app.state, 'workflow_classifier', None),
    )
    if not getattr(graph, 'formal_router', False):
        return WorkflowService(graph, repo, model, settings, **arguments)
    small = None
    if settings.intent_small_model:
        # 显式小模型配置仍沿用同一兼容上游与凭据，不更改主聊天模型。
        small = model.model_copy(update={'model_name': settings.intent_small_model})
    service_class = (
        ConversationService
        if getattr(graph, 'layered_context', False)
        else RoutingService
    )
    # 根据应用编译的图选择服务，不由用户参数切换版本；旧图保留隔离教学回归入口。
    if service_class is ConversationService:
        # 注册表应用共享，但其余仓库/工具上下文仍按请求独立，不能共享可变AsyncSession。
        arguments['summaries'] = request.app.state.summary_tasks
        if getattr(graph, 'unified_tools', False):
            # 共用应用目录，服务内再冻结本段Schema/权限，不能把请求仓库放进共享runtime。
            service_class = ToolingService
            arguments['tools'] = request.app.state.tool_runtime
    return service_class(
        graph,
        repo,
        model,
        settings,
        **arguments,
        understander=getattr(request.app.state, 'workflow_understander', None),
        expander=getattr(request.app.state, 'workflow_expander', None),
        small_model=small,
    )


def get_repository(request: Request):
    """
    Depends 每次请求调用这里，注入轻量仓库对象。
    仓库共享的是引擎/Session 工厂，不是把一个打开的事务共享给所有请求。
    """
    # 仓库按请求创建，但注入应用共享的知识服务，避免每轮重新建立Milvus客户端。
    if getattr(getattr(request.app.state, 'workflow', None), 'layered_context', False):
        # 访客标识用于演示会话分组/归属核验，不是登录令牌；模型看不到其他会话正文。
        visitor = request.headers.get('X-Visitor-Id')
        if visitor and (
            len(visitor) > 128 or not re.fullmatch(r'[A-Za-z0-9-]+', visitor)
        ):
            raise HTTPException(422, '访客标识格式错误')
        return ContextRepository(
            request.app.state.database, request.app.state.knowledge, visitor=visitor
        )
    return ChatRepository(request.app.state.database, request.app.state.knowledge)


async def lock_identity(repo, session_id):
    """先统一定位键，再取得本进程锁；两个会话别名不能各占一把锁同时推进图。"""
    # 原UUID与列表规范ID指向同一会话，必须共用一把锁。
    if isinstance(repo, ContextRepository):
        return f'conv7:{await repo.open_conversation(session_id)}'
    return session_id


def sse_frame(data: dict | str, event: str | None = None) -> str:
    """构造一个 SSE 事件：可选 event 行 + data 行 + 空行。

    JSON 编码会转义正文换行，防止正文被误解析为额外 SSE 行。
    [DONE] 是本项目约定的结束标记，不是 SSE 协议自带的关键字。
    """
    # json.dumps 默认 ensure_ascii=True，会把中文转义为 \uXXXX，前端显示不友好。
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return (f"event: {event}\n" if event else "") + f"data: {payload}\n\n"


@router.post("/chat", response_class=StreamingResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    model=Depends(get_chat_model),
    repo=Depends(get_repository),
):
    """
    处理聊天请求，返回流式响应。
    - body: 请求体，包含 session_id 和 message。
    - request: FastAPI 请求对象，用于访问应用状态。
    - settings: 应用配置，通过依赖注入获取。
    - model: 聊天模型，通过依赖注入获取。
    """
    # FastAPI 校验 body；Depends 取得配置/模型。Request 则提供应用共享状态。
    # store只提供进程内锁；正式历史在PG checkpoint，SQL补入规则由各章服务处理。
    store = request.app.state.sessions
    try:
        # 在发送响应头之前检查当前输入。开始流以后无法改 HTTP 状态为 422。
        # 预检必须使用实际路径的system，避免旧Prompt能放下、新Agent Prompt却超预算。
        system = (
            AGENT_SYSTEM
            if getattr(request.app.state, "workflow", None) is not None
            else CUSTOMER_SERVICE_SYSTEM
        )
        if getattr(
            getattr(request.app.state, 'workflow', None), 'layered_context', False
        ):
            # 本章已将系统/证据/摘要/工具峰值计入窗口预算，不再拿旧文本2000额度裁历史。
            ContextBudget.from_settings(settings)
            if text_tokens(body.message, settings) > settings.max_user_input_tokens:
                raise BudgetExceeded('当前消息超过本轮输入预算，请缩短消息')
            # 归属检查在响应头之前，跨访客不能启动图或看到其暂停信息。
            await repo.open_conversation(body.session_id)
        else:
            prepare_messages(
                [],
                body.message,
                system,
                settings.llm_token_budget,
                settings.zh_chars_pre_token,
            )
    except BudgetExceeded as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except LookupError:
        raise HTTPException(404, '会话不存在') from None

    lock_key = await lock_identity(repo, body.session_id)
    # 检查/定位可在响应前完成，锁则覆盖后面整段流式执行，避免同会话消息顺序交错。

    async def stream():
        """
        async generator 的 yield 交出一帧，而不是等全部生成后才返回。
        """
        async with store.lock(lock_key):
            # 锁覆盖整个生成过程，避免同一会话的并发问答顺序交错。
            try:
                workflow = getattr(request.app.state, "workflow", None)
                if workflow is not None:
                    # 共享的是已编译图；每个HTTP请求仍创建独立服务/仓库/工具上下文。
                    service = workflow_service(request, repo, model, settings)
                    event_stream = service.stream(
                        body.session_id,
                        body.message,
                        body.filters.model_dump(exclude_none=True),
                    )
                else:
                    # 仅兼容显式关闭Workflow的旧章节回归，不是Postgres故障时自动降级。
                    service = ChatService(
                        repo,
                        model,
                        text_budget=settings.llm_token_budget,
                        tool_budget=settings.tool_context_budget,
                        chars_per_token=settings.zh_chars_pre_token,
                        tool_timeout=settings.tool_timeout_seconds,
                        rag=getattr(request.app.state, "rag", None),
                        filters=body.filters.model_dump(exclude_none=True),
                    )
                    event_stream = service.stream(body.session_id, body.message)
                # aclosing 保证页面断开时关闭内层生成器及其上游模型流。
                async with aclosing(event_stream) as events:
                    async for event in events:
                        # service 产生内部 dict 事件；HTTP/SSE 格式在 API 层转换。
                        # tool 走命名事件，delta 延用原文本事件，前端分别处理。
                        if event["type"] in (
                            "tool",
                            "citations",
                            "workflow",
                            "actions",
                            "order_selection",
                            "refund_offer",
                            "conversation",
                            "ticket_preview",
                        ):
                            # 命名事件分别携带工具状态、来源、执行事实和按钮建议。
                            # type只用于后端分流；发送event名称后去掉内部type键。
                            yield sse_frame(
                                {k: v for k, v in event.items() if k != "type"},
                                event["type"],
                            )
                        else:
                            yield sse_frame({"delta": event["delta"]})
                        await asyncio.sleep(0)
                # 服务生成器正常结束表示最终回复已落库，才允许发送成功标记。
                yield sse_frame("[DONE]")
            except Exception as exc:
                # 流已开始，错误用 SSE 事件告知；不发送代表成功的 DONE。
                # asyncio.CancelledError 不在普通 Exception 范围内，会向外传播。
                # 不回显上游异常文本：其中可能含请求头、URL 或其他敏感内容。
                logger.warning("Chat upstream failure: %s", type(exc).__name__)
                yield sse_frame(
                    {
                        "error": "upstream_error",
                        "message": "本轮服务调用失败，请稍后重试；已完成的工具操作会保留记录",
                    },
                    "error",
                )

    return StreamingResponse(
        # Response 逐步消费 stream()；禁缓存/禁代理缓冲有助于及时送达。
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post('/chat/orders', response_class=StreamingResponse)
async def select_order(
    body: OrderSelection,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    model=Depends(get_chat_model),
    repo=Depends(get_repository),
):
    """点选仅恢复当前暂停图；服务端校验候选集合，非法选择用明确错误结束。"""
    if not getattr(
        getattr(request.app.state, 'workflow', None), 'formal_router', False
    ):
        raise HTTPException(503, '正式分流尚未启用')
    try:
        lock_key = await lock_identity(repo, body.session_id)
    except LookupError:
        raise HTTPException(404, '会话不存在') from None

    async def stream():
        async with request.app.state.sessions.lock(lock_key):
            try:
                service = workflow_service(request, repo, model, settings)
                async with aclosing(service.resume(body)) as events:
                    async for item in events:
                        if item['type'] == 'delta':
                            yield sse_frame({'delta': item['delta']})
                        else:
                            yield sse_frame(
                                {
                                    key: value
                                    for key, value in item.items()
                                    if key != 'type'
                                },
                                item['type'],
                            )
                        await asyncio.sleep(0)
                yield sse_frame('[DONE]')
            except LookupError:
                yield sse_frame(
                    dict(
                        error='selection_expired',
                        message='这张订单卡片已失效，请重新发送问题选择订单',
                    ),
                    'error',
                )
            except Exception as exc:
                logger.warning('Order resume failure: %s', type(exc).__name__)
                yield sse_frame(
                    dict(
                        error='upstream_error',
                        message='选单后处理未完成，请用同一订单卡片重试核对',
                    ),
                    'error',
                )

    return StreamingResponse(
        stream(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@router.post('/chat/tickets/resume', response_class=StreamingResponse)
async def resume_ticket_preview(
    body: TicketDecision,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    model=Depends(get_chat_model),
    repo=Depends(get_repository),
):
    """页面只传凭据与确认选择，实际工具参数只从服务端checkpoint读取。"""
    if not getattr(
        getattr(request.app.state, 'workflow', None), 'unified_tools', False
    ):
        raise HTTPException(503, '工具确认流程未启用')
    try:
        lock_key = await lock_identity(repo, body.session_id)
    except LookupError:
        raise HTTPException(404, '会话不存在') from None

    async def stream():
        async with request.app.state.sessions.lock(lock_key):
            # 会话锁只覆盖本次resume执行，不跨预览等待持有；同会话双击按顺序核验。
            try:
                service = workflow_service(request, repo, model, settings)
                async with aclosing(service.resume_ticket(body)) as events:
                    async for item in events:
                        if item['type'] == 'delta':
                            yield sse_frame({'delta': item['delta']})
                        else:
                            yield sse_frame(
                                {k: v for k, v in item.items() if k != 'type'},
                                item['type'],
                            )
                yield sse_frame('[DONE]')
            except LookupError:
                yield sse_frame(
                    dict(
                        error='preview_expired',
                        message='这张工单预览已失效，请重新发送建单需求。',
                    ),
                    'error',
                )
            except Exception as exc:
                logger.warning('Ticket resume failure: %s', type(exc).__name__)
                # 错误可能发生在写入后回执生成阶段，提示同预览重试，不能说“一定没建”。
                yield sse_frame(
                    dict(
                        error='upstream_error',
                        message='本次处理未完成，请用同一预览重试核对；不要重复新建。',
                    ),
                    'error',
                )

    return StreamingResponse(
        stream(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@router.post('/chat/refunds')
async def submit_refund(
    body: RefundSubmission,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    repo=Depends(get_repository),
):
    """用户确认保存演示申请；不会执行退款、建工单或转人工。"""
    graph = getattr(request.app.state, 'workflow', None)
    if not getattr(graph, 'formal_router', False):
        raise HTTPException(503, '正式分流尚未启用')
    try:
        lock_key = await lock_identity(repo, body.session_id)
    except LookupError:
        raise HTTPException(404, '会话不存在') from None
    async with request.app.state.sessions.lock(lock_key):
        try:
            return await workflow_service(request, repo, None, settings).confirm_refund(
                body
            )
        except LookupError:
            raise HTTPException(404, '本会话没有这条退款申请建议') from None
        except Exception as exc:
            logger.warning('Refund submission failure: %s', type(exc).__name__)
            raise HTTPException(
                503, '暂时无法确认申请结果，请使用同一表单重试核对'
            ) from None


@router.post("/chat/tickets")
async def confirm_ticket(
    body: TicketConfirmation,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    repo=Depends(get_repository),
):
    """校验独立建单确认并调用服务；没有转人工动作，也不会请求模型。

    与聊天共用会话锁，避免本进程内两个请求同时覆盖同一份checkpoint建议表。
    无建议返回404；提交结果不确定返回503，引导同建议重试而不是新开一单。
    """
    workflow = getattr(request.app.state, "workflow", None)
    if workflow is None:
        raise HTTPException(503, "Workflow 尚未启用")
    try:
        lock_key = await lock_identity(repo, body.session_id)
    except LookupError:
        raise HTTPException(404, '会话不存在') from None
    async with request.app.state.sessions.lock(lock_key):
        # 此路径不调用模型，显式传None，写入依据是已确认的请求和服务器保存的建议。
        service = workflow_service(request, repo, None, settings)
        try:
            return await service.confirm_ticket(body)
        except LookupError:
            raise HTTPException(404, "本会话没有这条可用的建工单建议") from None
        except Exception as exc:
            logger.warning("Ticket confirmation failure: %s", type(exc).__name__)
            raise HTTPException(
                503, "暂时无法确认建单结果，请使用同一个按钮重试核对"
            ) from None


@router.get('/conversations')
async def list_conversations(repo=Depends(get_repository)):
    # 只读会话壳接口；旧图/无访客情形返回空列表，不把全库会话当同一个用户展示。
    if not isinstance(repo, ContextRepository):
        return []
    return await repo.list_conversations()


@router.get('/conversations/{conversation_id}/messages')
async def conversation_messages(
    conversation_id: int, request: Request, repo=Depends(get_repository)
):
    # 页面回载读SQL原文，和Agent整理的三层窗口是两条独立读取路径。
    if not isinstance(repo, ContextRepository):
        raise HTTPException(404, '会话不存在')
    try:
        cid = await repo.open_conversation(f'conv7:{conversation_id}')
    except LookupError:
        raise HTTPException(404, '会话不存在') from None
    messages = await repo.raw_history(cid)
    result = dict(
        id=str(cid),
        session_id=f'conv7:{cid}',
        messages=[
            dict(
                id=str(message.additional_kwargs['sql_msg_id']),
                role=message.type,
                content=message.content,
            )
            for message in messages
            if message.type == 'human'
            or (message.type == 'ai' and bool(message.content))
        ],
    )
    # 有可见正文的工具申请也需回载，不能把已经流出的“我先查询”一起过滤掉。
    # 空申请和Tool正文不作为聊天气泡；完整协议仍在checkpoint，接口也不触发模型。
    graph = getattr(request.app.state, 'workflow', None)
    if getattr(graph, 'layered_context', False):
        snapshot = await graph.aget_state(ConversationService.config(f'conv7:{cid}'))
        state = snapshot.values
        if snapshot.next == ('ticket_preview',) and state.get('ticket_preview'):
            # 刷新/切回只恢复服务端预览，不生成批准，不执行工具，也不推进图。
            result['ticket_preview'] = state['ticket_preview']
        elif snapshot.next and state.get('selection'):
            # 刷新/切回不能让还有效的订单卡片消失；这里只读暂停凭据，不推进图。
            result['order_selection'] = state['selection']
        elif state:
            final = state.get('messages', [])[-1] if state.get('messages') else None
            result['target_message_id'] = (
                str(final.additional_kwargs.get('sql_msg_id', '')) if final else None
            )
            # 由SQL行ID把当前有效建议挂回对应答复，不能把它挂到页面最后一个任意气泡。
            last = state.get('last_offer')
            if last:
                offer = state['offers'][last]
                if not offer.get('result'):
                    from app.workflow.protocol import OPTIONS

                    result['actions'] = dict(
                        offer_id=last,
                        options=[
                            dict(action=name, label=OPTIONS[name])
                            for name in offer['actions']
                        ],
                        description=offer['description'],
                        ticket_type=offer['ticket_type'],
                    )
            refund = state.get('last_refund_offer')
            if refund and not state['refund_offers'][refund].get('result'):
                from app.refunds import REASONS

                result['refund_offer'] = dict(
                    offer_id=refund,
                    order=state['refund_offers'][refund]['order'],
                    reasons=REASONS,
                )
    return result
