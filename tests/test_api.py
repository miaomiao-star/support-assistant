import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from app.config import Settings, get_settings
from app.core.llm import get_chat_model, get_extract_chain
from app.main import create_app
from app.schemas.extract import AfterSalesTicket


class FakeLLM:
    def __init__(self):
        self.calls = []
        self.fail = False

    async def astream(self, prompt):
        self.calls.append(prompt.to_messages())
        yield AIMessageChunk(content="你好\n")
        if self.fail:
            raise RuntimeError("sensitive-upstream-detail")
        await asyncio.sleep(0)
        yield AIMessageChunk(content="小喵")


class FakeExtract:
    async def ainvoke(self, value):
        return AfterSalesTicket(request_type="退款", expected_solution="退还货款")


@pytest.fixture
def api(monkeypatch):
    settings = Settings(_env_file=None, CHAT_API_KEY="fake-test-key",
                        CHAT_BASE_URL="https://example.invalid/v1", CHAT_MODEL="fake")
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    fake = FakeLLM()
    app.dependency_overrides[get_chat_model] = lambda: fake
    app.dependency_overrides[get_extract_chain] = lambda: FakeExtract()
    with TestClient(app) as client:
        yield client, app, fake


def deltas(response):
    frames = response.text.strip().split("\n\n")
    return [json.loads(f[6:])["delta"] for f in frames if f.startswith('data: {')]


def test_sse_frames_and_two_turn_history(api):
    client, app, fake = api
    response = client.post("/api/chat", json={"session_id": "s", "message": "我要猫粮"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert deltas(response) == ["你好\n", "小喵"]
    assert response.text.endswith("data: [DONE]\n\n")
    client.post("/api/chat", json={"session_id": "s", "message": "我刚才想买什么?"})
    assert [m.type for m in fake.calls[1]] == ["system", "human", "ai", "human"]
    assert fake.calls[1][1].content == "我要猫粮"
    assert fake.calls[1][2].content == "你好\n小喵"
    client.post("/api/chat", json={"session_id": "other", "message": "你好"})
    assert len(fake.calls[2]) == 2


def test_sse_failure_does_not_commit_partial_reply(api):
    client, app, fake = api
    fake.fail = True
    response = client.post("/api/chat", json={"session_id": "s", "message": "你好"})
    assert "event: error\n" in response.text
    assert "sensitive-upstream-detail" not in response.text
    assert "[DONE]" not in response.text
    assert app.state.sessions.history("s") == []
    fake.fail = False
    client.post("/api/chat", json={"session_id": "s", "message": "重试"})
    assert len(fake.calls[-1]) == 2


def test_over_budget_rejected_before_stream(api):
    client, app, fake = api
    response = client.post("/api/chat", json={"session_id": "s", "message": "长" * 4000})
    assert response.status_code == 422
    assert fake.calls == []


def test_extract_returns_null_order(api):
    client, _, _ = api
    response = client.post("/api/extract", json={"text": "商品坏了,我要退款"})
    assert response.status_code == 200
    assert response.json() == {"order_id": None, "request_type": "退款", "expected_solution": "退还货款"}


def test_extract_failure_returns_safe_502(api):
    client, app, _ = api
    class Broken:
        async def ainvoke(self, value):
            raise RuntimeError("sensitive-upstream-detail")
    app.dependency_overrides[get_extract_chain] = lambda: Broken()
    response = client.post("/api/extract", json={"text": "我要退款"})
    assert response.status_code == 502
    assert "sensitive-upstream-detail" not in response.text


def test_extract_rejects_invented_order(api):
    client, app, _ = api
    class Invented:
        async def ainvoke(self, value):
            return AfterSalesTicket(order_id="MH999", request_type="退款", expected_solution="退款")
    app.dependency_overrides[get_extract_chain] = lambda: Invented()
    assert client.post("/api/extract", json={"text": "我要退款"}).status_code == 502


@pytest.mark.parametrize("path,data", [
    ("/api/chat", {"session_id": "s", "message": " "}),
    ("/api/extract", {"text": " "}),
])
def test_blank_inputs(api, path, data):
    client, _, _ = api
    assert client.post(path, json=data).status_code == 422


def test_empty_model_reply_is_error_and_not_saved(api):
    client, app, _ = api
    class Empty:
        async def astream(self, prompt):
            yield AIMessageChunk(content="")
    app.dependency_overrides[get_chat_model] = lambda: Empty()
    response = client.post("/api/chat", json={"session_id": "empty", "message": "你好"})
    assert "event: error" in response.text
    assert "[DONE]" not in response.text
    assert app.state.sessions.history("empty") == []


def test_cancelled_stream_releases_lock_without_commit():
    from types import SimpleNamespace
    from app.api.chat import chat
    from app.core.memory import SessionStore

    async def run():
        store = SessionStore()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(sessions=store)))
        settings = Settings(_env_file=None, CHAT_API_KEY="test-key",
                            CHAT_BASE_URL="https://example.invalid/v1", CHAT_MODEL="fake")
        class Closable:
            closed = False
            async def astream(self, prompt):
                try:
                    yield AIMessageChunk(content="部分回复")
                    await asyncio.Event().wait()
                finally:
                    self.closed = True
        fake = Closable()
        response = await chat(ChatRequest(session_id="s", message="你好"), request, settings, fake)
        iterator = response.body_iterator
        await anext(iterator)
        await iterator.aclose()
        assert store.history("s") == []
        assert not store.lock("s").locked()
        assert fake.closed
    from app.schemas.chat import ChatRequest
    asyncio.run(run())
