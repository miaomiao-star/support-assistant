"""单进程会话存储。预算是字符估算值，不是模型 tokenizer 的精确值。"""

import asyncio
import json
from math import ceil

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, trim_messages


class BudgetExceeded(ValueError):
    """无法保留 system 和当前消息时抛出，由 API 转换为 422。"""
    pass


def estimate_tokens(messages: list[BaseMessage], chars_per_token: float = 1.2) -> int:
    """每条消息估算为 4 个开销 token + 字符数/系数，向上取整。

    4 是本项目经验值，不是服务商计费规则。英文、中文和真实分词存在差异。
    """
    # 工具参数和结果也占上下文；仍是字符估算，不是精确 tokenizer。
    total = 0
    for message in messages:
        content = str(message.content)
        if getattr(message, 'tool_calls', None):
            # 文本正文可能为空，但申请里的工具名、参数 JSON 仍然占用上下文。
            content += json.dumps(message.tool_calls, ensure_ascii=False)
        content += getattr(message, 'tool_call_id', '')
        total += 4 + ceil(len(content) / chars_per_token)
    return total


def prepare_messages(
    history: list[BaseMessage], message: str, system: str,
    budget: int, chars_per_token: float,
) -> list[BaseMessage]:
    """准备一次请求的输入：[system, 最近完整历史轮次..., 当前 human]。"""
    system_message = SystemMessage(content=system)
    current = HumanMessage(content=message)
    if estimate_tokens([system_message, current], chars_per_token) > budget:
        raise BudgetExceeded("当前消息与系统提示词超过上下文预算，请缩短消息或提高 LLM_TOKEN_BUDGET")
    # 单条消息完整不等于工具回合完整，按 user 到最终 assistant 分组裁剪。
    groups, group, pending = [], [], set()
    # 例如 [human, ai(申请), tool(结果), ai(回答)] 要作为一组保留/删除。
    # pending 防止在工具结果尚未齐全时，把中间状态误当成完整问答。
    for item in history:
        if item.type == 'human':
            group, pending = [item], set()
        elif group:
            group.append(item)
            if item.type == 'tool':
                pending.discard(item.tool_call_id)
            elif getattr(item, 'tool_calls', None):
                pending.update(call['id'] for call in item.tool_calls)
            elif item.type == 'ai' and not pending:
                groups.append(group)
                group = []
    retained = []
    for group in reversed(groups):
        # 从最近回合向前放；某组放不下就停止，不只保留它的尾部助手消息。
        if estimate_tokens([system_message, *group, *retained, current], chars_per_token) > budget:
            break
        retained = [*group, *retained]
    return [system_message, *retained, current]


class SessionStore:
    """内存版会话仓库；锁只协调本进程，本章未做会话过期与总数限制。"""
    def __init__(self) -> None:
        """
        兼容第一章的内存历史接口；Ch02 聊天路径只依赖 _locks，
        _histories/history()/commit() 不再是新接口的历史数据来源。
        """
        self._histories: dict[str, list[BaseMessage]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def history(self, session_id: str) -> list[BaseMessage]:
        """
        返回浅拷贝，调用方改列表不会改变仓库；消息对象本身仍共享。
        """
        return list(self._histories.get(session_id, []))

    def lock(self, session_id: str) -> asyncio.Lock:
        """
        同一会话复用同一把锁；不同会话可以同时请求上游。
        """
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
