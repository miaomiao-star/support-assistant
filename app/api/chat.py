import asyncio
import json
import logging
from contextlib import aclosing
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage

from app.config import Settings, get_settings
from app.core.llm import get_chat_model
from app.core.memory import BudgetExceeded, prepare_messages
from app.core.prompts import CUSTOMER_SERVICE_PROMPT, CUSTOMER_SERVICE_SYSTEM
from app.schemas.chat import ChatRequest

router = APIRouter()
logger = logging.getLogger(__name__)


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
    body: ChatRequest, request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    model=Depends(get_chat_model),
):
    """
    处理聊天请求，返回流式响应。
    - body: 请求体，包含 session_id 和 message。
    - request: FastAPI 请求对象，用于访问应用状态。
    - settings: 应用配置，通过依赖注入获取。
    - model: 聊天模型，通过依赖注入获取。
    """
    # FastAPI 校验 body；Depends 取得配置/模型。Request 则提供应用共享状态。
    store = request.app.state.sessions
    try:
        # 在发送响应头之前检查当前输入。开始流以后无法改 HTTP 状态为 422。
        prepare_messages([], body.message, CUSTOMER_SERVICE_SYSTEM,
                         settings.llm_token_budget, settings.zh_chars_pre_token)
    except BudgetExceeded as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    async def stream():
        # async generator 的 yield 交出一帧，而不是等全部生成后才返回。
        async with store.lock(body.session_id):
            # 锁覆盖整个生成过程，避免同一会话的并发问答顺序交错。
            try:
                # 先准备输入，裁掉过长历史；再格式化 Prompt，最后请求模型。
                messages = prepare_messages(
                    store.history(body.session_id), body.message, CUSTOMER_SERVICE_SYSTEM,
                    settings.llm_token_budget, settings.zh_chars_pre_token,
                )
                prompt = await CUSTOMER_SERVICE_PROMPT.ainvoke({"history": messages[1:]})
                # 模板自己有 system，所以切掉 messages[0]，避免重复系统提示词。
                parts = []
                async with aclosing(model.astream(prompt)) as chunks:
                    # 页面断开或生成器关闭时，也立即关闭内层上游流。
                    async for chunk in chunks:
                        content = chunk.content
                        if isinstance(content, str):
                            delta = content
                        else:
                            # 部分模型返回内容块列表；只转发 text，不展示其他块。
                            delta = "".join(
                                block.get("text", "") for block in content
                                if isinstance(block, dict) and block.get("type") == "text"
                            )
                        if delta:
                            # 一份发给浏览器，一份积累为完整 AIMessage。
                            # chunk 不一定等于一个 token，前端会另做逐字动画。
                            parts.append(delta)
                            yield sse_frame({"delta": delta})
                        await asyncio.sleep(0)
                        # 让出执行机会，使其他请求以及取消信号能得到处理。
                if not parts:
                    raise ValueError("Empty model reply")
                store.commit(
                    # 只有上游完整成功才提交；失败不把半截回答写进历史。
                    body.session_id, [*messages[1:], AIMessage(content="".join(parts))],
                    system=CUSTOMER_SERVICE_SYSTEM, budget=settings.llm_token_budget,
                    chars_per_token=settings.zh_chars_pre_token,
                )
                yield sse_frame("[DONE]")
            except Exception as exc:
                # 流已开始，错误用 SSE 事件告知；不发送代表成功的 DONE。
                # asyncio.CancelledError 不在普通 Exception 范围内，会向外传播。
                # 不回显上游异常文本：其中可能含请求头、URL 或其他敏感内容。
                logger.warning("Chat upstream failure: %s", type(exc).__name__)
                yield sse_frame({"error": "upstream_error", "message": "模型服务调用失败，请检查配置或稍后重试"}, "error")

    return StreamingResponse(
        # Response 逐步消费 stream()；禁缓存/禁代理缓冲有助于及时送达。
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
