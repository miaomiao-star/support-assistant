"""本地HTTP验收：原生型号命中、引用来源、真实拒答入池和元数据预过滤。"""

import asyncio
import json
import sys
import time
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import httpx


async def main():
    report = {}
    # 本机验收直连回环地址，不让系统代理代转localhost。
    async with httpx.AsyncClient(
        base_url='http://127.0.0.1:8000', timeout=180, trust_env=False
    ) as client:
        # 先证实原生BM25能按型号命中，且品类过滤没有让其他商品混入。
        response = await client.post(
            '/api/kb/search',
            json={
                'query': 'MH-W40水箱几升',
                'strategy': 'bm25',
                'filters': {'product_category': '饮水机'},
            },
        )
        response.raise_for_status()
        report['bm25'] = response.json()
        assert 'MH-W40' in report['bm25']['data'][0]['section_path']
        assert all('饮水机' in r['question'] for r in report['bm25']['data'])
        key = report['bm25']['data'][0]['id']
        source = await client.get(f'/api/kb/source/{key}')
        assert source.json()['answer'] == report['bm25']['data'][0]['answer']
        assert (await client.get(f'/kb/source/{key}')).status_code == 200
        report['source'] = source.json()
        chat = await client.post(
            '/api/chat',
            json={
                'session_id': 'ch04-citations-' + str(time.time_ns()),
                'message': '邮费是多少',
            },
        )
        assert 'data: [DONE]' in chat.text and 'event: error' not in chat.text
        evidence = []
        answer = ''
        # SSE命名事件与普通delta分开解析；网络完成标记不能当作答案内容。
        for frame in chat.text.split('\n\n'):
            payload = next(
                (line[6:] for line in frame.splitlines() if line.startswith('data: ')),
                None,
            )
            if not payload or payload == '[DONE]':
                continue
            value = json.loads(payload)
            if frame.startswith('event: citations'):
                evidence = value['citations']
            elif not frame.startswith('event:'):
                answer += value.get('delta', '')
        refs = {int(n) for n in re.findall(r'\[(\d+)\]', answer)}
        assert (
            refs
            and refs <= {c['n'] for c in evidence}
            and all(w in answer for w in ('99', '10', '偏远'))
        )
        cited = next(c for c in evidence if c['n'] in refs)
        # 引用快照要与MySQL来源API一致，不能只是页面上出现一个可点击数字。
        original = (await client.get('/api/kb/source/' + str(cited['chunk_id']))).json()
        assert original['answer'] == cited['answer']
        report['citation_sse'] = chat.text
        before = (await client.get('/api/kb/quality')).json()['low_confidence']
        # 真实HTTP聊天会写演示会话及拒答池，所以该命令需要服务已就绪。
        raw = 'MH-W99水箱容量多少'
        chat = await client.post(
            '/api/chat',
            json={'session_id': 'ch04-smoke-' + str(time.time_ns()), 'message': raw},
        )
        assert 'data: [DONE]' in chat.text and 'event: error' not in chat.text
        after = (await client.get('/api/kb/quality')).json()['low_confidence']
        assert len(after) > len(before) and after[0]['raw_question'] == raw
        assert '无法确认' in chat.text
        report['refusal_sse'] = chat.text
        report['pool_row'] = after[0]
        report['passed'] = True
    path = ROOT / 'dev-notes' / f'ch04-smoke-{time.time_ns()}.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(path, flush=True)


if __name__ == '__main__':
    asyncio.run(main())
