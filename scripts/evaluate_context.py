"""真实上游22轮验证；分别在default/demo服务上运行，保留逐轮原始结果与图证据。"""

import argparse
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
from app.db import Database, DatabaseSettings
from app.config import get_settings
from app.context.store import ContextRepository
from app.workflow.persistence import open_checkpointer, run_async
from app.workflow.conversation import build_graph
from app.workflow.service import WorkflowService
from scripts.smoke_router import frames


async def main(args):
    """在指定已启动服务跑22轮真实链路，同时只读SQL/PG取证并逐轮保存报告。"""
    # 每次评估新建访客/会话，防止上次历史影响“默认不压缩”或“早期事实召回”结果。
    visitor = 'ch07-eval-' + uuid4().hex
    session = 'ch07-' + args.profile + '-' + uuid4().hex
    path = ROOT / 'dev-notes' / f'ch07-{args.profile}-{time.time_ns()}.json'
    report = dict(profile=args.profile, visitor=visitor, session=session, results=[])
    database = Database(DatabaseSettings().url())
    repo = ContextRepository(database, visitor=visitor)
    try:
        async with (
            open_checkpointer(get_settings()) as saver,
            httpx.AsyncClient(
                base_url=args.url,
                headers={'X-Visitor-Id': visitor},
                timeout=240,
                trust_env=False,
            ) as client,
        ):
            graph = build_graph(saver)
            # 本地graph仅用于读checkpoint；真正执行的是HTTP目标服务的图和配置。
            questions = [
                '订单1001的MH-W40饮水机漏水，保留包装，还没使用，想退货，怎么处理？'
            ]
            # 用受控长原话增长token，而不是靠短闲聊消息条数硬触发摘要。
            note = (
                '补充我的收货偏好：平日白天家里没有人，晚上才方便签收，快递员请先联系；楼下有代收点，但我更希望送到门口。商品是给家里宠物准备的，计划周末整理收货位置，所以想先了解配送进度。这里只补充收货安排，没有请求取消、退货或投诉。'
            ) * 5
            for index in range(2, 22):
                questions.append(
                    f'这轮仅查询订单1002的物流进度，第{index}次核对。' + note
                )
            questions.append(
                '最开始那个订单后来怎么说？请回顾当时的商品、问题和诉求，不要查现在这单物流。'
            )
            for index, question in enumerate(questions, 1):
                start = time.monotonic()
                row = dict(round=index, question=question)
                try:
                    response = await client.post(
                        '/api/chat', json=dict(session_id=session, message=question)
                    )
                    response.raise_for_status()
                    events = frames(response.text)
                    errors = [value for kind, value in events if kind == 'error']
                    assert not errors, errors
                    assert any(value == '[DONE]' for kind, value in events)
                    selection = next(
                        (value for kind, value in events if kind == 'order_selection'),
                        None,
                    )
                    if selection:
                        # 核验真实缺槽位路径时按当前轮应处理的演示订单续跑，不直接注入图字段。
                        selected = await client.post(
                            '/api/chat/orders',
                            json=dict(
                                session_id=session,
                                selection_id=selection['selection_id'],
                                order_id='1001' if index in (1, 22) else '1002',
                            ),
                        )
                        resumed = frames(selected.text)
                        assert any(
                            value == '[DONE]' for kind, value in resumed
                        ), resumed
                        events += resumed
                    cid = await repo.open_conversation(session)
                    state = (
                        await graph.aget_state(WorkflowService.config(f'conv7:{cid}'))
                    ).values
                    metadata = await repo.metadata(cid)
                    row.update(
                        intent=state['intent'],
                        resolved=state['resolved_question'],
                        normalized=state['normalized_question'],
                        answer=state['answer'],
                        trace=state['trace'],
                        stop_reason=state.get('stop_reason'),
                        state_messages=len(state['messages']),
                        metadata=metadata,
                        actual_tokens=state.get('actual_tokens'),
                        estimated_tokens=state.get('estimated_tokens'),
                    )
                    assert not state.get('stop_reason'), state.get('stop_reason')
                    # 收到DONE只证明HTTP成功，仍需排除“因步数/输出限制而安全停止”的伪成功。
                    if args.profile == 'default':
                        assert (
                            metadata['layer1_from'] == metadata['summary_upto'] == 0
                        ), '默认预算意外压缩'
                    row['passed'] = True
                except Exception as exc:
                    row.update(
                        passed=False,
                        error_type=type(exc).__name__,
                        error=str(exc)[:500],
                    )
                row['elapsed_ms'] = round((time.monotonic() - start) * 1000)
                report['results'].append(row)
                path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
                )
                # 每轮落盘，失败保留当时输出；不会把中断报告当成22轮全部通过。
                print(
                    f"{args.profile} {index}/22 {'PASS' if row['passed'] else 'FAIL'} {row.get('intent','')} {row['elapsed_ms']}ms",
                    flush=True,
                )
                if not row['passed']:
                    break
            cid = await repo.open_conversation(session)
            report['metadata'] = await repo.metadata(cid)
            report['conversation_id'] = str(cid)
            report['passed'] = len(report['results']) == 22 and all(
                row['passed'] for row in report['results']
            )
            if args.profile == 'demo':
                # 号码/诉求短词检查只是自动底线，摘要与最终答复的语义仍需逐条看报告。
                report['passed'] = report['passed'] and bool(
                    report['metadata']['segments']
                )
                if report['results'] and report['results'][-1].get('answer'):
                    answer = report['results'][-1]['answer']
                    report['recall_passed'] = '1001' in answer and (
                        '漏水' in answer or '退货' in answer
                    )
                    report['passed'] = report['passed'] and report['recall_passed']
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            print(path, flush=True)
            if not report['passed']:
                raise SystemExit(1)
    finally:
        await database.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--profile', choices=['default', 'demo'], default='default')
    run_async(main(parser.parse_args()))
