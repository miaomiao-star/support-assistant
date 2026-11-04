"""Ch08真实HTTP/SQL/PG验收；按用户本章授权确认一张演示工单并取消一张。"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import httpx
from sqlalchemy import select, func
from app.db import Database, DatabaseSettings
from app.models import Ticket, ToolAuditLog
from app.workflow.persistence import run_async, open_checkpointer
from app.workflow.service import WorkflowService
from app.workflow.tooling import build_graph
from app.config import get_settings
from app.tools.runtime import ToolRuntime
from app.tools.catalog import ToolSpec, LocalPolicy
from app.tools.engine import ToolEngine, ExecutionContext, Approval
from app.tools.audit import AuditStore


def frames(text):
    # SSE按空行划帧，delta仍可能拆字；检查完整工单号前需拼接所有message事件。
    result = []
    for frame in text.replace('\r', '').split('\n\n'):
        kind = 'message'
        payload = []
        for line in frame.splitlines():
            if line.startswith('event:'):
                kind = line[6:].strip()
            if line.startswith('data:'):
                payload.append(line[5:].strip())
        if payload:
            value = '\n'.join(payload)
            result.append((kind, value if value == '[DONE]' else json.loads(value)))
    return result


async def main(url):
    # 真实HTTP/模型/数据库联调，会创建演示会话与一张工单；不是日常隔离回归命令。
    db = Database(DatabaseSettings().url())
    report = {'url': url, 'visitor': 'ch08-' + uuid4().hex, 'results': []}
    path = ROOT / 'dev-notes' / f'ch08-smoke-{time.time_ns()}.json'
    # 前置建池也可能失败，先创建报告，避免环境故障没有任何证据文件。
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    def save(row):
        report['results'].append(row)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print(('PASS' if row['passed'] else 'FAIL') + ' ' + row['name'], flush=True)

    try:
        async with open_checkpointer(get_settings()) as saver:
            graph = build_graph(saver)
            async with httpx.AsyncClient(
                base_url=url,
                timeout=240,
                trust_env=False,
                headers={'X-Visitor-Id': report['visitor']},
            ) as client:

                async def chat(session, question):
                    response = await client.post(
                        '/api/chat', json={'session_id': session, 'message': question}
                    )
                    response.raise_for_status()
                    events = frames(response.text)
                    assert not any(k == 'error' for k, v in events), events
                    assert any(v == '[DONE]' for k, v in events)
                    conversation = next(v for k, v in events if k == 'conversation')
                    config = WorkflowService.config(conversation['session_id'])
                    snapshot = await graph.aget_state(config)
                    return events, conversation, snapshot

                for question, expected in [
                    # 分别验证两个独立MCP来源；审计来源必须为mcp，不能回退内置物流。
                    ('请查询订单1001的完整物流轨迹', 'query_logistics'),
                    ('请查询订单1001目前是否在保，还剩多少保修天数', 'query_warranty'),
                    ('请查询订单1001当前的退货进度', 'query_return_progress'),
                ]:
                    row = {'name': expected, 'question': question}
                    try:
                        events, conv, snapshot = await chat(
                            'ch08-' + uuid4().hex, question
                        )
                        cid = int(conv['id'])
                        async with db.sessions() as session:
                            rows = (
                                await session.scalars(
                                    select(ToolAuditLog).where(
                                        ToolAuditLog.conversation_id == cid,
                                        ToolAuditLog.tool_name == expected,
                                    )
                                )
                            ).all()
                        row.update(
                            conversation=conv,
                            answer=snapshot.values.get('answer'),
                            audit=[
                                dict(
                                    status=r.status,
                                    source=r.tool_source,
                                    server=r.mcp_server,
                                )
                                for r in rows
                            ],
                        )
                        assert (
                            rows
                            and rows[-1].status == '成功'
                            and rows[-1].tool_source == 'mcp'
                        )
                        assert '模拟' in row['answer'] or '演示' in row['answer']
                        row['passed'] = True
                    except Exception as exc:
                        row.update(
                            passed=False,
                            error=type(exc).__name__,
                            detail=str(exc)[:500],
                        )
                    save(row)
                for confirmed in (True, False):
                    # 两个独立会话覆盖“缺描述→追问→预览→确认/取消”，避免前次状态干扰。
                    row = {'name': '工单确认' if confirmed else '工单取消'}
                    try:
                        session_id = 'ch08-ticket-' + uuid4().hex
                        events, conv, snapshot = await chat(session_id, '帮我建个工单')
                        assert not any(k == 'ticket_preview' for k, v in events)
                        first = snapshot.values.get('answer', '')
                        assert any(w in first for w in ('问题', '情况', '描述', '说明'))
                        events, conv, snapshot = await chat(
                            session_id, '饮水机漏水，需要维修'
                        )
                        preview = next(v for k, v in events if k == 'ticket_preview')
                        assert '漏水' in preview['description']
                        cid = int(conv['id'])

                        async def count():
                            async with db.sessions() as sql:
                                return await sql.scalar(
                                    select(func.count())
                                    .select_from(Ticket)
                                    .where(Ticket.conversation_id == cid)
                                )

                        before = await count()
                        # 另建同样的图，使用同一真实PG池验证暂停能被重建恢复。
                        restored = await build_graph(saver).aget_state(
                            WorkflowService.config(conv['session_id'])
                        )
                        assert restored.next == ('ticket_preview',)
                        history = (
                            await client.get(f'/api/conversations/{cid}/messages')
                        ).json()
                        assert history['ticket_preview'] == preview
                        body = dict(
                            session_id=conv['session_id'],
                            preview_id=preview['preview_id'],
                            confirmed=confirmed,
                        )
                        response = await client.post(
                            '/api/chat/tickets/resume', json=body
                        )
                        resumed = frames(response.text)
                        assert not any(k == 'error' for k, v in resumed), resumed
                        after = await count()
                        repeated = await client.post(
                            '/api/chat/tickets/resume', json=body
                        )
                        assert '[DONE]' in repeated.text and await count() == after
                        # 同选择重放不能多建或多审计；SQL计数独立于模型正文和SSE状态。
                        async with db.sessions() as sql:
                            audit = (
                                await sql.scalars(
                                    select(ToolAuditLog).where(
                                        ToolAuditLog.conversation_id == cid,
                                        ToolAuditLog.tool_name == 'create_ticket',
                                    )
                                )
                            ).all()
                            ticket = await sql.scalar(
                                select(Ticket).where(Ticket.conversation_id == cid)
                            )
                        assert after - before == int(confirmed)
                        assert len(audit) == 1 and audit[0].status == (
                            '成功' if confirmed else '权限拒绝'
                        )
                        answer = ''.join(
                            v.get('delta', '')
                            for k, v in resumed
                            if k == 'message' and isinstance(v, dict)
                        )
                        if ticket:
                            assert ticket.ticket_no in answer
                        row.update(
                            passed=True,
                            conversation=conv,
                            preview=preview,
                            before=before,
                            after=after,
                            status=audit[0].status,
                            answer=answer,
                            ticket_no=ticket.ticket_no if ticket else None,
                            pg_restored=True,
                        )
                    except Exception as exc:
                        row.update(
                            passed=False,
                            error=type(exc).__name__,
                            detail=str(exc)[:500],
                        )
                    save(row)
        # 人为超时只用专用替身，不触碰实际tickets写入；真实MySQL审计照常记录。
        runtime = ToolRuntime()
        calls = {'read': 0, 'write': 0}

        async def slow_read(args, context):
            calls['read'] += 1
            await asyncio.sleep(1)

        async def slow_write(args, context):
            calls['write'] += 1
            await asyncio.sleep(1)

        specs = {
            'timeout_read': ToolSpec(
                'timeout_read', '专用超时验收', {'type': 'object'}, slow_read
            ),
            'create_ticket': ToolSpec(
                'create_ticket',
                '专用写超时验收',
                {'type': 'object'},
                slow_write,
                access='write',
            ),
        }
        context = ExecutionContext(
            None, request_id='ch08-timeout-' + uuid4().hex, ticket_requested=True
        )
        engine = ToolEngine(
            specs, LocalPolicy(), context, audit=AuditStore(db).write, timeout=0.05
        )
        for name, key in [('timeout_read', 'read'), ('create_ticket', 'write')]:
            call = dict(name=name, args={}, id=context.request_id + key)
            result = await engine.execute(
                call,
                approval=Approval.for_call(context, call) if key == 'write' else None,
            )
            async with db.sessions() as sql:
                audit = await sql.scalar(
                    select(ToolAuditLog).where(ToolAuditLog.tool_call_id == call['id'])
                )
            save(
                dict(
                    name=key + '超时审计',
                    passed=result['error'] == 'timeout'
                    and calls[key] == (2 if key == 'read' else 1)
                    and audit.retry_count == (1 if key == 'read' else 0)
                    and audit.status == '超时',
                    calls=calls[key],
                    retry_count=audit.retry_count,
                    duration_ms=audit.duration_ms,
                    status=audit.status,
                    result=result,
                )
            )
    except Exception as exc:
        save(dict(name='联调前置/执行故障', passed=False, error=type(exc).__name__))
    finally:
        await db.close()
    report.update(
        passed=sum(r['passed'] for r in report['results']), total=len(report['results'])
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"{report['passed']}/{report['total']} {path}")
    return report['passed'] == report['total']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    raise SystemExit(0 if run_async(main(parser.parse_args().url)) else 1)
