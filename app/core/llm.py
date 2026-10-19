"""OpenAI 协议模型工厂；所有上游配置均来自 .env。"""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.core.prompts import EXTRACT_PROMPT
from app.schemas.extract import AfterSalesTicket


@lru_cache
def get_chat_model() -> ChatOpenAI:
    """创建并缓存客户端；不是把对话历史放进模型对象。"""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.chat_model,
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key.get_secret_value(),
        temperature=0,  # 尽量稳定输出，但不能保证每次结果完全相同。
        timeout=60,  # SDK 网络超时配置，不等同于整个请求绝对耗时上限。
        max_retries=1,  # 有限重试；不是业务操作失败后无限重试。
        use_responses_api=False,  # 使用上游兼容的 Chat Completions 协议。
    )


@lru_cache
def get_extract_chain():
    """
    JSON 模式不引入工具调用；Pydantic 解析校验在 LangChain 内完成。
    """
    model = get_chat_model().with_structured_output(AfterSalesTicket, method="json_mode")
    # | 是 LangChain 的流水线：先格式化 Prompt，再请求模型并解析结果。
    # JSON 模式保证方向是 JSON；字段有效性由 Pydantic 校验，事实仍需核对。
    return EXTRACT_PROMPT | model
