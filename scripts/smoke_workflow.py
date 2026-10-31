"""Ch05 真实 HTTP/模型/数据库验收；--confirm-ticket 会创建一张演示工单。"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import Database, DatabaseSettings
from app.models import Conversation, Ticket
from app.repositories.chat import ChatRepository
from app.workflow.persistence import run_async


async def collect(client, base, session_id, question, filters=None):
    """通过真实HTTP消费SSE，把命名事件、正文增量及DONE还原成便于断言的列表。"""
    events = []
    async with client.stream(
        "POST",
        base + "/api/chat",
        json=dict(session_id=session_id, message=question, filters=filters or {}),
    ) as response:
        response.raise_for_status()
        kind = "message"
        async for line in response.aiter_lines():
            # aiter_lines处理网络分片；event名称只作用于下一段以空行分隔的事件。
            if line.startswith("event:"):
                kind = line[6:].strip()
            elif line.startswith("data:"):
                value = line[5:].strip()
                if value == "[DONE]":
                    events.append(dict(type="done"))
                else:
                    events.append(dict(type=kind, **json.loads(value)))
            elif not line:
                kind = "message"
    return events


async def main(args):
    """验证真实分类/检索/Agent/数据库协作；确认建单由独立命令选项控制。"""
    report = dict(kind="live_http_workflow", rows=[], completed=False)
    path = ROOT / "dev-notes" / f"ch05-smoke-{time.time_ns()}.json"
    db = Database(DatabaseSettings().url())
    repo = ChatRepository(db)
    cases = [
        ("policy", "退货政策是什么", "knowledge", None),
        ("logistics", "订单1001的物流到哪了", "business", None),
        ("complaint", "我要投诉", "complaint", None),
        ("smalltalk", "你好，谢谢你", "smalltalk", None),
        (
            "multi_step",
            "先查订单1001的状态，再查这个订单的物流，最后汇总两项结果",
            "business",
            None,
        ),
        ("missing_order", "帮我查一下物流，我还没提供订单号", "business", None),
        (
            "weak_evidence",
            "MH-W99的水箱容量是多少",
            "knowledge",
            # 特意制造空候选来验证前置闸，不声称该分数能识别所有未知型号。
            {"product_category": "Ch05验收空品类"},
        ),
        (
            "refund_order",
            "先查退货政策，再查订单1001的状态，分别告诉我政策条件与订单演示状态",
            "knowledge",
            None,
        ),
    ]
    try:
        # 本机回环请求不走系统代理；这个设置不修改服务端的外部模型代理。
        async with httpx.AsyncClient(timeout=240, trust_env=False) as client:
            for name, question, expected, filters in cases:
                session_id = "ch05-smoke-" + uuid4().hex
                start = time.monotonic()
                row = dict(name=name, question=question, session_id=session_id)
                try:
                    events = await collect(
                        client, args.base, session_id, question, filters
                    )
                    # 从真实执行事件取路径，而不是从预期标签推断“应该走过”的节点。
                    nodes = [e["node"] for e in events if e["type"] == "workflow"]
                    tools = [
                        e["name"]
                        for e in events
                        if e["type"] == "tool" and e["status"] == "running"
                    ]
                    answer = "".join(
                        e.get("delta", "") for e in events if e["type"] == "message"
                    )
                    row.update(
                        nodes=nodes,
                        tools=tools,
                        answer=answer,
                        delta_events=sum(e["type"] == "message" for e in events),
                        elapsed=round(time.monotonic() - start, 3),
                    )
                    assert not any(
                        e["type"] == "error" for e in events
                    ), "SSE service error"
                    assert events[-1]["type"] == "done", "missing DONE"
                    route = next(
                        e["route"]
                        for e in events
                        if e["type"] == "workflow" and e["node"] == "route"
                    )
                    assert route == expected, "wrong route"
                    if expected == "knowledge":
                        # 不仅验证有检索，还验证其在闸门和首次Agent调用之前发生。
                        assert nodes.index("retrieve") < nodes.index("gate")
                        gate = next(
                            e
                            for e in events
                            if e["type"] == "workflow" and e["node"] == "gate"
                        )
                        row["gate"] = gate
                        if name == "weak_evidence":
                            assert not gate["allowed"] and "agent" not in nodes
                        else:
                            assert gate["allowed"] and nodes.index(
                                "gate"
                            ) < nodes.index("agent")
                            sources = next(
                                e["citations"]
                                for e in events
                                if e["type"] == "citations"
                            )
                            actual = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
                            # 这里验证编号存在，不将引用格式通过等同于事实语义完全忠实。
                            assert actual and actual <= {
                                s["n"] for s in sources
                            }, "invalid/missing citations"
                    else:
                        assert "retrieve" not in nodes and "gate" not in nodes
                    if name == "logistics":
                        assert tools == ["query_logistics"]
                    if name == "multi_step":
                        assert tools == ["query_order", "query_logistics"]
                    if name == "refund_order":
                        assert "query_order" in tools
                    if name == "missing_order":
                        assert not tools and "订单" in answer
                    if name in ("complaint", "smalltalk"):
                        assert "agent" not in nodes
                    if name == "complaint":
                        # 建议出现之后先查真实数据库，确认未点击时没有工单副作用。
                        offer = next(e for e in events if e["type"] == "actions")
                        assert {o["action"] for o in offer["options"]} == {
                            "handoff",
                            "create_ticket",
                        }
                        cid = await repo.open_conversation(session_id)
                        async with db.sessions() as session:
                            before = len(
                                (
                                    await session.scalars(
                                        select(Ticket).where(
                                            Ticket.conversation_id == cid
                                        )
                                    )
                                ).all()
                            )
                        assert before == 0, "ticket created without click"
                        if args.confirm_ticket:
                            # 只有显式--confirm-ticket才发确认请求；重复提交验证同号幂等。
                            payload = dict(
                                session_id=session_id,
                                offer_id=offer["offer_id"],
                                confirmed=True,
                                description="Ch05真实HTTP验收：用户确认投诉工单",
                                ticket_type="投诉",
                            )
                            first = await client.post(
                                args.base + "/api/chat/tickets", json=payload
                            )
                            first.raise_for_status()
                            repeat = await client.post(
                                args.base + "/api/chat/tickets", json=payload
                            )
                            repeat.raise_for_status()
                            assert (
                                first.json()["ticket_no"] == repeat.json()["ticket_no"]
                            )
                            async with db.sessions() as session:
                                after = len(
                                    (
                                        await session.scalars(
                                            select(Ticket).where(
                                                Ticket.conversation_id == cid
                                            )
                                        )
                                    ).all()
                                )
                                status = (await session.get(Conversation, cid)).status
                            assert after == 1 and status == "进行中"
                            row["ticket"] = first.json()["ticket_no"]
                            row["tickets_before_after"] = [before, after]
                        continued = await collect(client, args.base, session_id, "谢谢")
                        assert not any(e["type"] == "error" for e in continued)
                    row["passed"] = True
                except Exception as exc:
                    row.update(
                        passed=False,
                        error=type(exc).__name__,
                        reason=(
                            str(exc)
                            if isinstance(exc, AssertionError)
                            else "服务或连接异常，详见本机服务日志"
                        ),
                    )
                report["rows"].append(row)
                # 逐场景落盘，即使后续服务故障也保留本轮已得到的真实结果。
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(name, "PASS" if row["passed"] else "FAIL", flush=True)
        report["completed"] = True
    finally:
        # 数据库连接生命周期不跨整个脚本退出；失败场景也要关闭引擎。
        await db.close()
        report["passed"] = sum(row["passed"] for row in report["rows"])
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f'{report["passed"]}/{len(cases)}；报告 {path.name}')
    return 0 if report["passed"] == len(cases) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--confirm-ticket", action="store_true")
    raise SystemExit(run_async(main(parser.parse_args())))
