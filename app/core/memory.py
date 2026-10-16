"""单进程会话存储。预算是字符估算值，不是模型 tokenizer 的精确值。"""

import asyncio
from math import ceil

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, trim_messages


class BudgetExceeded(ValueError):
    """无法保留 system 和当前消息时抛出，由 API 转换为 422。"""
    pass


def estimate_tokens(messages: list[BaseMessage], chars_per_token: float = 1.2) -> int:
    """每条消息估算为 4 个开销 token + 字符数/系数，向上取整。

    4 是本项目经验值，不是服务商计费规则。英文、中文和真实分词存在差异。
    """
    # 每条消息计入角色/协议开销；仅支持本章使用的文本消息。
    return sum(4 + ceil(len(str(message.content)) / chars_per_token) for message in messages)


def prepare_messages(
    history: list[BaseMessage], message: str, system: str,
    budget: int, chars_per_token: float,
) -> list[BaseMessage]:
    """准备一次请求的输入：[system, 最近完整历史轮次..., 当前 human]。"""
    system_message = SystemMessage(content=system)
    current = HumanMessage(content=message)
    if estimate_tokens([system_message, current], chars_per_token) > budget:
        raise BudgetExceeded("当前消息与系统提示词超过上下文预算，请缩短消息或提高 LLM_TOKEN_BUDGET")
    retained = trim_messages(
        # 先为当前消息留预算，再裁历史，避免用户本轮问题被静默删除。
        [system_message, *history],
        max_tokens=budget - estimate_tokens([current], chars_per_token),
        token_counter=lambda messages: estimate_tokens(messages, chars_per_token),
        strategy="last", include_system=True, start_on="human", end_on="ai",
        # last 留最近消息；首尾分别为 human/ai，避免只保留孤立的回复。
        allow_partial=False,
    )
    # 空历史或所有历史被裁剪时，显式保留 system。
    if not retained or not isinstance(retained[0], SystemMessage):
        retained = [system_message, *retained]
    return [*retained, current]


class SessionStore:
    """内存版会话仓库；锁只协调本进程，本章未做会话过期与总数限制。"""
    def __init__(self) -> None:
        self._histories: dict[str, list[BaseMessage]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def history(self, session_id: str) -> list[BaseMessage]:
        # 返回浅拷贝，调用方改列表不会改变仓库；消息对象本身仍共享。
        return list(self._histories.get(session_id, []))

    def lock(self, session_id: str) -> asyncio.Lock:
        # 同一会话复用同一把锁；不同会话可以同时请求上游。
        return self._locks.setdefault(session_id, asyncio.Lock())

    def commit(
        self, session_id: str, messages: list[BaseMessage], *, system: str = "",
        budget: int = 2000, chars_per_token: float = 1.2,
    ) -> None:
        """完整生成成功后保存历史，并再次裁剪，避免单个会话无限增长。

        若一个完整问答轮次太长而放不下，整轮会被丢弃，而非截断回复。
        """
        retained = trim_messages(
            [SystemMessage(content=system), *messages], max_tokens=budget,
            token_counter=lambda items: estimate_tokens(items, chars_per_token),
            strategy="last", include_system=True, start_on="human", end_on="ai",
            allow_partial=False,
        )
        self._histories[session_id] = [m for m in retained if not isinstance(m, SystemMessage)]
