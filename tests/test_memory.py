import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.memory import BudgetExceeded, SessionStore, estimate_tokens, prepare_messages


def test_session_isolation_and_copy():
    store = SessionStore()
    store.commit("a", [HumanMessage(content="猫粮"), AIMessage(content="您好")])
    assert store.history("b") == []
    copy = store.history("a")
    copy.clear()
    assert len(store.history("a")) == 2


def test_trim_keeps_recent_complete_turns_and_system():
    history = [HumanMessage(content="旧" * 20), AIMessage(content="旧回复" * 20),
               HumanMessage(content="猫粮"), AIMessage(content="已记住")]
    result = prepare_messages(history, "刚才说的商品?", "客服", 40, 1.2)
    assert isinstance(result[0], SystemMessage)
    assert [m.content for m in result[1:]] == ["猫粮", "已记住", "刚才说的商品?"]
    assert estimate_tokens(result, 1.2) <= 40


def test_orphan_reply_is_not_kept():
    history = [HumanMessage(content="长" * 100), AIMessage(content="短回复")]
    result = prepare_messages(history, "你好", "客服", 20, 1.2)
    assert [m.type for m in result] == ["system", "human"]


def test_current_message_cannot_be_silently_dropped():
    with pytest.raises(BudgetExceeded):
        prepare_messages([], "长" * 100, "客服", 20, 1.2)


def test_storage_stays_within_budget():
    store = SessionStore()
    store.commit("a", [HumanMessage(content="猫粮"), AIMessage(content="答复" * 100)],
                 system="客服", budget=30, chars_per_token=1.2)
    assert store.history("a") == []


def test_same_session_lock_serializes_requests():
    async def run():
        store = SessionStore()
        events = []
        async def work(number):
            async with store.lock("same"):
                events.append((number, "start"))
                await asyncio.sleep(0)
                events.append((number, "end"))
        await asyncio.gather(work(1), work(2))
        assert events == [(1, "start"), (1, "end"), (2, "start"), (2, "end")]
    asyncio.run(run())
