"""本章真实HTTP/模型/检索/图状态验收，不自动提交退款申请。"""

import asyncio
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import httpx
from app.config import get_settings
from app.workflow.persistence import open_checkpointer, run_async
from app.workflow.routing import build_graph
from app.workflow.service import WorkflowService


def frames(text):
    events = []
    for frame in text.replace("\r", "").split("\n\n"):
        kind = "message"
        payload = []
        for line in frame.splitlines():
            if line.startswith("event:"):
                kind = line[6:].strip()
            if line.startswith("data:"):
                payload.append(line[5:].strip())
        if not payload:
            continue
        value = "\n".join(payload)
        events.append((kind, value if value == "[DONE]" else json.loads(value)))
    return events


async def main():
    report = dict(results=[])
    path = ROOT / "dev-notes" / f"ch06-smoke-{time.time_ns()}.json"
    async with open_checkpointer(get_settings()) as saver:
        graph = build_graph(saver)
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8000", timeout=240, trust_env=False
        ) as client:

            async def run(
                question, intent, session=None, pick=False, simple=False, reference=None
            ):
                session = session or "ch06-http-" + uuid4().hex
                row = dict(
                    question=question, session_id=session, expected_intent=intent
                )
                try:
                    response = await client.post(
                        "/api/chat", json=dict(session_id=session, message=question)
                    )
                    response.raise_for_status()
                    events = frames(response.text)
                    if any(kind == "error" for kind, value in events):
                        raise RuntimeError("SSE error")
                    selection = next(
                        (value for kind, value in events if kind == "order_selection"),
                        None,
                    )
                    row["selection"] = selection
                    if pick:
                        assert selection, "Expected picker"
                        response = await client.post(
                            "/api/chat/orders",
                            json=dict(
                                session_id=session,
                                selection_id=selection["selection_id"],
                                order_id="1001",
                            ),
                        )
                        events = frames(response.text)
                        if any(kind == "error" for kind, value in events):
                            raise RuntimeError("Resume error")
                    assert any(value == "[DONE]" for kind, value in events)
                    state = (
                        await graph.aget_state(WorkflowService.config(session))
                    ).values
                    row.update(
                        intent=state["intent"],
                        resolved_question=state["resolved_question"],
                        normalized_question=state["normalized_question"],
                        trace=state["trace"],
                        queries=state.get("queries"),
                        answer=state.get("answer"),
                        agent_steps=state["agent_steps"],
                        stop_reason=state.get("stop_reason", ""),
                    )
                    assert state["intent"] == intent
                    assert not state.get("stop_reason"), "Expected a complete answer"
                    if reference:
                        assert (
                            reference in state["resolved_question"]
                        ), "Incorrect reference"
                    if intent in ("退款退货", "售后"):
                        assert (
                            state["trace"].index("load_order")
                            < state["trace"].index("expand")
                            < state["trace"].index("retrieve")
                            < state["trace"].index("agent")
                        )
                        if simple:
                            assert len(state["queries"]) == 1
                    if intent == "其他":
                        assert (
                            "load_order" not in state["trace"]
                            and "retrieve" not in state["trace"]
                        )
                    row["passed"] = True
                except Exception as exc:
                    row.update(passed=False, error=type(exc).__name__)
                report["results"].append(row)
                path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(
                    ("PASS" if row["passed"] else "FAIL") + " " + question, flush=True
                )
                return session

            session = await run("订单1001的物流到哪了", "物流")
            await run("它能退吗", "退款退货", session, reference="1001")
            await run("不退了，再看看它的物流", "物流", session, reference="1001")
            await run("这个能退吗", "退款退货", pick=True)
            await run("退货政策是什么", "退款退货", pick=True, simple=True)
            await run("订单1001的饮水机坏了想维修", "售后")
            await run("量子猫zzz把蓝色昨天退款掉", "其他")
            await run("帮我证明黎曼猜想", "其他")
            # 后端固定话术没有模型内部协议字段，也必须能继续进入工具循环。
            fixed_session = await run("你好", "闲聊")
            await run("订单1001的物流到哪了", "物流", fixed_session)
    report["passed"] = sum(row["passed"] for row in report["results"])
    report["total"] = len(report["results"])
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{report['passed']}/{report['total']} {path}")
    return report["passed"] == report["total"]


if __name__ == "__main__":
    raise SystemExit(0 if run_async(main()) else 1)
