"""Ch09真实聊天标注验证：先经审核页发布演示FAQ；每次报告保留，调用现有模型。"""

import argparse
import json
from pathlib import Path
import re
import time
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    dict(id='F1', question='MH-W40 演示版的极光节能模式怎么开启？', kind='knowledge'),
    dict(id='F2', question='MH-W40 的水箱容量是多少？', kind='knowledge'),
    dict(id='L1', question='订单1001的物流轨迹是什么？', kind='logistics'),
]


def main(only=None):
    cases = [c for c in CASES if not only or c['id'] == only]
    report = ROOT / 'dev-notes' / f'ch09-chat-{time.time_ns()}.json'
    rows = []
    visitor = 'ch09-check-' + uuid4().hex
    with httpx.Client(
        base_url='http://127.0.0.1:8000',
        headers={'X-Visitor-Id': visitor},
        timeout=180,
        trust_env=False,
    ) as client:
        for case in cases:
            row = {**case, 'session_id': uuid4().hex, 'visitor': visitor}
            started = time.monotonic()
            try:
                response = client.post(
                    '/api/chat',
                    json={'session_id': row['session_id'], 'message': case['question']},
                )
                response.raise_for_status()
                events = [
                    json.loads(line[6:])
                    for line in response.text.splitlines()
                    if line.startswith('data: {')
                ]
                answer = ''.join(e.get('delta', '') for e in events)
                row.update(answer=answer, events=events)
                passed = bool(answer) and not any(
                    e.get('type') == 'error' for e in events
                )
                if case['kind'] == 'knowledge':
                    passed = (
                        passed
                        and bool(re.search(r'\[\d+\]', answer))
                        and '可靠的依据' not in answer
                        and '安全处理边界' not in answer
                    )
                    if case['id'] == 'F1':
                        passed = passed and '星河喵喵' in answer and '演示' in answer
                    else:
                        compact = answer.replace(' ', '').upper()
                        passed = passed and ('4L' in compact or '4升' in compact)
                else:
                    passed = (
                        passed
                        and '模拟' in answer
                        and 'query_logistics' in json.dumps(events, ensure_ascii=False)
                    )
                row['passed'] = bool(passed)
            except Exception as exc:
                row.update(passed=False, error=type(exc).__name__)
            row['elapsed_ms'] = round((time.monotonic() - started) * 1000)
            rows.append(row)
            report.write_text(
                json.dumps(
                    dict(
                        results=rows,
                        passed=sum(r['passed'] for r in rows),
                        total=len(cases),
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding='utf-8',
            )
            print(('PASS' if row['passed'] else 'FAIL') + ' ' + case['id'], flush=True)
    print(f"{sum(r['passed'] for r in rows)}/{len(cases)} {report}")
    return all(r['passed'] for r in rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=[c['id'] for c in CASES])
    raise SystemExit(0 if main(parser.parse_args().only) else 1)
