import asyncio
import json
import logging
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

router = APIRouter()
logger = logging.getLogger(__name__)


def get_repository(request: Request):
    """
    Depends 每次请求调用这里，注入轻量仓库对象。
    仓库共享的是引擎/Session 工厂，不是把一个打开的事务共享给所有请求。
    """
    # 仓库按请求创建，但注入应用共享的知识服务，避免每轮重新建立Milvus客户端。
    return ChatRepository(request.app.state.database, request.app.state.knowledge)


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
    # store只提供进程内锁；Ch05完整历史由checkpoint延续，首次才从MySQL引入。
    store = request.app.state.sessions
    try:
        # 在发送响应头之前检查当前输入。开始流以后无法改 HTTP 状态为 422。
        # 预检必须使用实际路径的system，避免旧Prompt能放下、新Agent Prompt却超预算。
        system = (
            AGENT_SYSTEM
            if getattr(request.app.state, "workflow", None) is not None
            else CUSTOMER_SERVICE_SYSTEM
        )
        prepare_messages(
            [],
            body.message,
            system,
            settings.llm_token_budget,
            settings.zh_chars_pre_token,
        )
    except BudgetExceeded as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    async def stream():
        """
        async generator 的 yield 交出一帧，而不是等全部生成后才返回。
        """
        async with store.lock(body.session_id):
            # 锁覆盖整个生成过程，避免同一会话的并发问答顺序交错。
            try:
                workflow = getattr(request.app.state, "workflow", None)
                if workflow is not None:
                    # 共享的是已编译图；每个HTTP请求仍创建独立服务/仓库/工具上下文。
                    service = WorkflowService(
                        workflow,
                        repo,
                        model,
                        settings,
                        rag=getattr(request.app.state, "rag", None),
                        classifier=getattr(
                            request.app.state, "workflow_classifier", None
                        ),
                    )
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
    async with request.app.state.sessions.lock(body.session_id):
        # 此路径不调用模型，显式传None，写入依据是已确认的请求和服务器保存的建议。
        service = WorkflowService(workflow, repo, None, settings)
        try:
            return await service.confirm_ticket(body)
        except LookupError:
            raise HTTPException(404, "本会话没有这条可用的建工单建议") from None
        except Exception as exc:
            logger.warning("Ticket confirmation failure: %s", type(exc).__name__)
            raise HTTPException(
                503, "暂时无法确认建单结果，请使用同一个按钮重试核对"
            ) from None
