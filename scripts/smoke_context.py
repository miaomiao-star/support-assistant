"""定向真实回归与SQL/checkpoint/log证据核验，不提交退款或工单。"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import httpx
from app.config import get_settings
from app.db import Database, DatabaseSettings
from app.context.store import ContextRepository
from app.workflow.conversation import build_graph
from app.workflow.persistence import open_checkpointer, run_async
from app.workflow.service import WorkflowService
from scripts.smoke_router import frames


async def main(args):
    """先审计已有压力报告；非audit-only时再运行少量真实HTTP语义回归。"""
    database = Database(DatabaseSettings().url())
    report = dict(checks=[], results=[])
    try:
        async with open_checkpointer(get_settings()) as saver:
            graph = build_graph(saver)
            sources = []
            for profile, explicit in [
                ('default', args.default_report),
                ('demo', args.demo_report),
            ]:
                # 默认取最新报告，也可固定路径复查旧证据；没有报告时不自动启动收费22轮。
                candidates = list((ROOT / 'dev-notes').glob(f'ch07-{profile}-*.json'))
                if not explicit and not candidates:
                    raise RuntimeError('请先运行默认/演示22轮评估，或显式指定报告路径')
                sources.append(
                    Path(explicit)
                    if explicit
                    else max(candidates, key=lambda path: path.stat().st_mtime)
                )
            for filename in sources:
                saved = json.loads(filename.read_text(encoding='utf-8'))
                repo = ContextRepository(database, visitor=saved['visitor'])
                cid = int(saved['conversation_id'])
                sql = await repo.raw_history(cid)
                state = (
                    await graph.aget_state(WorkflowService.config(f'conv7:{cid}'))
                ).values
                assert len(state['messages']) == saved['results'][-1]['state_messages']
                # SQL不应新写tool正文，PG却必须保留；两边条数故意不同，不是数据丢失。
                assert not any(message.type == 'tool' for message in sql)
                assert any(message.type == 'tool' for message in state['messages'])
                assert len({message.id for message in state['messages']}) == len(
                    state['messages']
                )
                entries = []
                for line in (
                    (ROOT / 'log' / 'app.log').read_text(encoding='utf-8').splitlines()
                ):
                    if '{' not in line:
                        continue
                    value = json.loads(line[line.index('{') :])
                    if value.get('conversation_id') == cid:
                        entries.append(
                            (line[:23], line[24 : line.index('{')].strip(), value)
                        )
                payloads = [
                    value for stamp, kind, value in entries if kind == 'model_ctx'
                ]
                prefix_hashes = {
                    # 比较实际SDK日志中的system+tools，只证明前缀稳定，不冒称测得缓存命中率。
                    hashlib.sha256(
                        json.dumps(
                            dict(system=value['messages'][0], tools=value['tools']),
                            sort_keys=True,
                            ensure_ascii=False,
                        ).encode()
                    ).hexdigest()
                    for value in payloads
                }
                assert len(prefix_hashes) == 1
                asynchronous = []
                for stamp, kind, value in entries:
                    if kind != 'summary trigger':
                        continue
                    done = next(
                        (
                            entry[0]
                            for entry in entries
                            if entry[1] == 'summary done'
                            and entry[2]['upto'] == value['upto']
                        ),
                        None,
                    )
                    following = next(
                        (
                            entry[0]
                            for entry in entries
                            if entry[1] == 'history_ctx' and entry[0] > stamp
                        ),
                        None,
                    )
                    asynchronous.append(
                        dict(
                            trigger=stamp,
                            next_history=following,
                            done=done,
                            nonblocking=bool(done and following and following < done),
                        )
                    )
                # 下一轮history_ctx在摘要done之前，才是“用户可以不等摘要继续聊”的时序证据。
                if saved['profile'] == 'demo':
                    assert asynchronous and all(
                        item['nonblocking'] for item in asynchronous
                    )
                report['checks'].append(
                    dict(
                        profile=saved['profile'],
                        conversation_id=str(cid),
                        sql_messages=len(sql),
                        checkpoint_messages=len(state['messages']),
                        checkpoint_tools=sum(
                            message.type == 'tool' for message in state['messages']
                        ),
                        static_prefix_hash=next(iter(prefix_hashes)),
                        history_ctx_count=sum(
                            kind == 'history_ctx' for stamp, kind, value in entries
                        ),
                        model_ctx_count=len(payloads),
                        async_evidence=asynchronous,
                        passed=True,
                    )
                )
            if args.audit_only:
                # 本分支仍连接真实SQL/PG、读取本地日志，但不会发起新的聊天模型调用。
                report['passed'] = True
                return
            visitor = 'ch07-smoke-' + uuid4().hex
            repo = ContextRepository(database, visitor=visitor)
            session = 'ch07-smoke-' + uuid4().hex
            async with httpx.AsyncClient(
                base_url=args.url,
                headers={'X-Visitor-Id': visitor},
                timeout=180,
                trust_env=False,
            ) as client:
                for question, expected in [
                    ('量子猫zzz把昨天蓝色时间退款掉', '其他'),
                    ('你好，谢谢', '闲聊'),
                    ('这个能退吗', '退款退货'),
                    ('不退了，只查它的物流', '物流'),
                ]:
                    # 只发送问题与用户选单；不自动确认退款/工单，业务写入边界保持原契约。
                    response = await client.post(
                        '/api/chat', json=dict(session_id=session, message=question)
                    )
                    response.raise_for_status()
                    events = frames(response.text)
                    assert any(value == '[DONE]' for kind, value in events), events
                    selection = next(
                        (value for kind, value in events if kind == 'order_selection'),
                        None,
                    )
                    if selection:
                        response = await client.post(
                            '/api/chat/orders',
                            json=dict(
                                session_id=session,
                                selection_id=selection['selection_id'],
                                order_id='1001',
                            ),
                        )
                        assert any(
                            value == '[DONE]' for kind, value in frames(response.text)
                        )
                    cid = await repo.open_conversation(session)
                    state = (
                        await graph.aget_state(WorkflowService.config(f'conv7:{cid}'))
                    ).values
                    assert state['intent'] == expected, (state['intent'], expected)
                    assert not state.get('stop_reason'), state.get('stop_reason')
                    if expected == '其他':
                        assert (
                            not state['selected_order_id']
                            and '商品' not in state['resolved_question']
                        )
                    if expected in ('退款退货', '物流'):
                        assert '模拟' in state['answer'], state['answer']
                    report['results'].append(
                        dict(
                            question=question,
                            intent=state['intent'],
                            resolved=state['resolved_question'],
                            answer=state['answer'],
                            picker=bool(selection),
                            passed=True,
                        )
                    )
                    print(expected, 'PASS', flush=True)
                listing = (await client.get('/api/conversations')).json()
                restored = (
                    await client.get(f'/api/conversations/{cid}/messages')
                ).json()
                assert listing[0]['session_id'] == restored['session_id']
                continued = await client.post(
                    # 列表返回的规范ID必须能继续原线程，不能只验“历史看得见”。
                    '/api/chat',
                    json=dict(session_id=restored['session_id'], message='谢谢，再见'),
                )
                assert any(value == '[DONE]' for kind, value in frames(continued.text))
                report['canonical_continue'] = True
        report['passed'] = True
    finally:
        path = ROOT / 'dev-notes' / f'ch07-audit-{time.time_ns()}.json'
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print(path, flush=True)
        await database.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--default-report')
    parser.add_argument('--demo-report')
    parser.add_argument('--audit-only', action='store_true')
    run_async(main(parser.parse_args()))
