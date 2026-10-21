"""真实上游 + MySQL 验收，先启动服务；会创建演示会话和一张人工工单。"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

import requests
from sqlalchemy import select

ROOT = Path(__file__).resolve().parent.parent
# 允许直接运行 python scripts/evaluate_tools.py 时，从项目根目录导入 app。
sys.path.insert(0, str(ROOT))
from app.db import Database, DatabaseSettings
from app.models import Message, Ticket
from app.repositories.chat import ChatRepository


def chat(base_url, session_id, message):
    """
    这是验收客户端，真实请求 FastAPI；不同于 tests 中不联网的假模型。
    chunks 只存文本，tools 存工具轨迹，arrivals 记录文本增量到达耗时。
    """
    chunks, tools, arrivals = [], [], []
    event, data, done = 'message', [], False
    started = time.monotonic()
    # requests 的超时为连接/读取等待超时，不是整轮请求的绝对耗时上限。
    with requests.post(f'{base_url}/api/chat', json=dict(session_id=session_id, message=message),
                       stream=True, timeout=(10, 180)) as response:
        response.raise_for_status()
        response.encoding = 'utf-8'
        # SSE 以空行结束一帧，event: 指类型，多条 data: 需拼成一个 payload。
        # chunk_size=1 减小客户端缓冲，不代表上游一个字节等于一个 token。
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if line.startswith('event:'):
                event = line[6:].strip()
            elif line.startswith('data:'):
                data.append(line[5:].lstrip())
            elif line == '' and data:
                payload = '\n'.join(data)
                if payload == '[DONE]':
                    # 收到成功结束标记；只断开连接而没 DONE 不能当作通过。
                    done = True
                    break
                item = json.loads(payload)
                if event == 'error':
                    raise RuntimeError(item['message'])
                if event == 'tool':
                    # 状态帧不能混进回复文字，也不能算成模型文本增量。
                    tools.append(item)
                elif 'delta' in item:
                    chunks.append(item['delta'])
                    arrivals.append(round(time.monotonic() - started, 3))
                event, data = 'message', []
                # 重置单帧状态，避免上一帧的 tool 类型影响下一帧的普通文本。
    return dict(reply=''.join(chunks), tools=tools, chunk_count=len(chunks),
                arrival_seconds=arrivals, done=done)


async def inspect(session_id):
    """
    从实际 MySQL 读取同会话流水，与 HTTP 结果交叉验证，
    不能仅凭“模型说查过了/工单建好了”判断业务真的执行成功。
    """
    db = Database(DatabaseSettings().url())
    try:
        repo = ChatRepository(db)
        cid = await repo.open_conversation(session_id)
        async with db.sessions() as session:
            rows = list((await session.scalars(select(Message).where(
                Message.conversation_id == cid).order_by(Message.id))).all())
            tickets = list((await session.scalars(select(Ticket).where(Ticket.conversation_id == cid))).all())
        calls = [c for r in rows for c in (r.tool_calls or [])]
        # tool_calls 在库中是结构化 JSON，tool 正文保存的是 JSON 字符串。
        results = [json.loads(r.content) for r in rows if r.role == 'tool']
        return dict(conversation_id=cid, roles=[r.role for r in rows], calls=calls,
                    results=results, tickets=[t.ticket_no for t in tickets])
    finally:
        # 每个样例检查完成都释放自己的连接池，避免测试脚本残留数据库连接。
        await db.close()


def evaluate(base_url):
    """
    标注数据明确指定预期工具、FAQ 关键词和是否命中；Prompt 通过这些例子验证。
    """
    cases = json.loads((ROOT / 'tests/fixtures/tool_cases.json').read_text(encoding='utf-8'))
    report = dict(results=[], retrieval='Ch03 dense semantic retrieval; original keyword contract preserved')
    for case in cases:
        # 样例各用新会话，避免上一条历史影响这条的工具选择。
        session_id = 'tools-eval-' + uuid4().hex
        try:
            actual = chat(base_url, session_id, case['message'])
            stored = asyncio.run(inspect(session_id))
            # 同时检查用户能看见的流、工具选择和数据库事实；只“回答正确”不够。
            ok = actual['done'] and actual['chunk_count'] > 1 and len(stored['calls']) == 1
            ok = ok and stored['calls'][0]['name'] == case['tool']
            ok = ok and stored['roles'] == ['user', 'assistant', 'tool', 'assistant']
            ok = ok and [t['status'] for t in actual['tools']] == ['running', 'success']
            result = stored['results'][0] if stored['results'] else {}
            ok = ok and result.get('ok', False)
            if 'keyword' in case:
                # “邮费”保留原词，靠向量语义命中，不能改词掩盖检索问题。
                ok = ok and stored['calls'][0]['args']['keyword'] == case['keyword']
                ok = ok and bool(result.get('data')) == case['hit']
                if not case['hit']:
                    ok = ok and any(w in actual['reply'] for w in ('未查到', '没查到', '未找到', '没有查到'))
                else:
                    ok = ok and all(token in actual['reply'] for token in case.get('reply_tokens', []))
            if case.get('mock'):
                # 不要求每次随机数相同；验证回答明确模拟，并与本次工具状态一致。
                ok = ok and result.get('data', {}).get('mock') and '模拟' in actual['reply']
                if case['tool'] in ('query_order', 'query_logistics'):
                    data = result['data']
                    ok = ok and data['order_id'] == '1001' and data['status'] in actual['reply']
            if case['tool'] == 'create_ticket':
                # 真正创建演示工单会改变本地数据库，所以该脚本不是无副作用单测。
                ok = ok and len(stored['tickets']) == 1 and result['ticket_no'] in actual['reply']
            row = dict(case=case, session_id=session_id, actual=actual, stored=stored, passed=bool(ok))
        except Exception as exc:
            # 不把底层网络异常/请求头写入报告。
            row = dict(case=case, session_id=session_id, passed=False, error=type(exc).__name__)
        report['results'].append(row)
        print(f"{case['id']}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    report['passed'] = all(r['passed'] for r in report['results'])
    return report


if __name__ == '__main__':
    # 只有从命令行直接运行才请求上游；被其他脚本 import 不会自动跑评估。
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    report = evaluate(args.base_url.rstrip('/'))
    # 每次独立报告，不覆盖失败历史。
    path = ROOT / 'dev-notes' / f'ch03-tools-{time.time_ns()}.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    # 命令行退出码便于人工/CI 判断结果：0 为全通过，1 为至少一个失败。
    print(f'Report: {path}')
    raise SystemExit(0 if report['passed'] else 1)
