import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.llm import get_extract_chain
from app.schemas.extract import AfterSalesTicket, ExtractRequest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/extract", response_model=AfterSalesTicket)
async def extract(body: ExtractRequest, chain=Depends(get_extract_chain)):
    """独立提取接口：请求文本 → Prompt → JSON 模式 → Pydantic → JSON 响应。"""
    try:
        ticket = await chain.ainvoke({"text": body.text})
        # 订单号必须在原文中出现，防止结构正确但凭空编造的输出。
        if ticket.order_id is not None and ticket.order_id not in body.text:
            # 这是基础子串核验，不是完整的订单合法性校验或数据库查询。
            raise ValueError("Order ID not found in source")
        return ticket
    except Exception as exc:
        # 非流式接口可以直接返回 502；不把可能含敏感信息的异常正文回显。
        logger.warning("Extract upstream failure: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="模型提取失败，请检查配置或稍后重试") from None
